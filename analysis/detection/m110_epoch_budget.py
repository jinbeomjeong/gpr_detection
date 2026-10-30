"""M1-10 epoch budget analysis (read-only aggregation, no training).

Reads the M1-7 fold-0 runs (30 epochs x 4 augmentation strengths) and the
AIHub split manifest, then prints:
  1. convergence statistics per run (mAP50 / mAP50-95 windows, late slope,
     train/val loss gap, lr schedule),
  2. a saturating fit of val mAP50 against cumulative learning rate, used only
     as a weak hint of what a longer linear-decay schedule could add,
  3. per-fold train/val sizes, iterations per epoch and the epoch count where
     Ultralytics optimizer=auto switches from AdamW to MuSGD (>10000 iters),
  4. wall-time estimates per fold and in total for candidate epoch budgets.

Usage:
  conda run -n gpr-dev python -I analysis/detection/m110_epoch_budget.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
RUNS = ["m17_aug0", "m17_aug0.5", "m17_aug1", "m17_aug2"]
MANIFEST = ROOT / "data" / "splits" / "aihub_yolo_v1.csv"

BATCH = 64
LR_FIT = 0.00125          # optimizer=auto AdamW lr for nc=4
LRF = 0.01
WARMUP_EPOCHS = 3.0
AUTO_SWITCH_ITERS = 10000  # ultralytics 8.4.192 trainer.build_optimizer
CANDIDATES = [30, 50, 80, 100]

# Timing constants measured in M1-7 (fold 0, b64, w2, mask 0xFF).
STEADY_S = 29.7           # mean epoch time incl. val, epochs 2-30
FIRST_EXTRA_S = 14.0      # first epoch 42-46 s -> ~+14 s over steady
FOLD0_TRAIN = 11276
CPU_OVERLAP_FACTOR = 34.2 / 29.7


def read_results(run: str) -> dict[str, np.ndarray]:
    path = ROOT / "runs" / "detect" / run / "results.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cols = {k.strip(): np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return cols


def linear_lr(epoch_idx: np.ndarray, epochs: int) -> np.ndarray:
    """Post-warmup lr at the end of epoch index e (0-based) for a linear schedule."""
    return LR_FIT * (np.maximum(1 - epoch_idx / epochs, 0) * (1 - LRF) + LRF)


def slope_with_se(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    a = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(a, y, rcond=None)
    n = len(x)
    resid = y - a @ coef
    s2 = float(resid @ resid) / (n - 2)
    se = math.sqrt(s2 / float(((x - x.mean()) ** 2).sum()))
    return float(coef[0]), se


def sat(s, a, b, tau):
    return a - b * np.exp(-s / tau)


def analyse_runs() -> dict:
    out = {}
    for run in RUNS:
        c = read_results(run)
        ep = c["epoch"].astype(int)
        m50 = c["metrics/mAP50(B)"] * 100
        m5095 = c["metrics/mAP50-95(B)"] * 100
        vbox = c["val/box_loss"]
        tbox = c["train/box_loss"]
        vcls = c["val/cls_loss"]
        tcls = c["train/cls_loss"]
        lr = c["lr/pg0"]
        t = c["time"]
        dt = np.diff(t)
        late = ep >= 21
        s50, se50 = slope_with_se(ep[late].astype(float), m50[late])
        s95, se95 = slope_with_se(ep[late].astype(float), m5095[late])
        sv, sev = slope_with_se(ep[ep >= 13].astype(float), vbox[ep >= 13])
        # Cumulative lr (lr logged per epoch is the value used during that epoch).
        cum_lr = np.cumsum(lr)
        fit = {}
        try:
            mask = ep >= 4  # skip warmup
            p, _ = curve_fit(sat, cum_lr[mask] / LR_FIT, m50[mask], p0=[85, 30, 5], maxfev=20000)
            pred = {}
            for e_tot in CANDIDATES:
                idx = np.arange(e_tot, dtype=float)
                lr_sched = linear_lr(idx, e_tot)
                # warmup: roughly linear ramp over first 3 epochs
                lr_sched[:3] = LR_FIT * (idx[:3] + 0.5) / WARMUP_EPOCHS
                s_tot = lr_sched.sum() / LR_FIT
                pred[e_tot] = round(float(sat(s_tot, *p)), 2)
            resid = m50[mask] - sat(cum_lr[mask] / LR_FIT, *p)
            fit = {"A_asymptote": round(float(p[0]), 2), "B": round(float(p[1]), 2),
                   "tau_epochs_at_full_lr": round(float(p[2]), 2),
                   "rmse": round(float(np.sqrt((resid ** 2).mean())), 2),
                   "pred_final_mAP50_by_budget": pred}
        except RuntimeError as exc:  # fit failed
            fit = {"error": str(exc)}
        best95 = int(ep[np.argmax(m5095)])
        best50 = int(ep[np.argmax(m50)])
        out[run] = {
            "epochs": int(ep[-1]),
            "time_last_s": round(float(t[-1]), 1),
            "epoch1_s": round(float(t[0]), 1),
            "steady_epoch_s_mean_2_30": round(float(dt.mean()), 2),
            "steady_epoch_s_median": round(float(np.median(dt)), 2),
            "mAP50_win": {f"{a}-{b}": round(float(m50[(ep >= a) & (ep <= b)].mean()), 2)
                          for a, b in [(1, 10), (11, 20), (21, 25), (26, 30)]},
            "mAP50_last5_mean_sd": [round(float(m50[-5:].mean()), 2), round(float(m50[-5:].std(ddof=1)), 2)],
            "mAP50_max_epoch": [round(float(m50.max()), 2), best50],
            "mAP50-95_win": {f"{a}-{b}": round(float(m5095[(ep >= a) & (ep <= b)].mean()), 2)
                             for a, b in [(11, 20), (21, 25), (26, 30)]},
            "mAP50-95_max_epoch(best.pt)": [round(float(m5095.max()), 2), best95],
            "slope_mAP50_ep21_30_pp_per_epoch": [round(s50, 3), round(se50, 3)],
            "slope_mAP50-95_ep21_30_pp_per_epoch": [round(s95, 3), round(se95, 3)],
            "slope_val_box_ep13_30_per_epoch": [round(sv, 4), round(sev, 4)],
            "train_box_ep13_ep30": [round(float(tbox[12]), 3), round(float(tbox[-1]), 3)],
            "val_box_ep13_min_ep30": [round(float(vbox[12]), 3), round(float(vbox.min()), 3),
                                      int(ep[np.argmin(vbox)]), round(float(vbox[-1]), 3)],
            "train_cls_ep13_ep30": [round(float(tcls[12]), 3), round(float(tcls[-1]), 3)],
            "val_cls_ep13_min_ep30": [round(float(vcls[12]), 3), round(float(vcls.min()), 3),
                                      int(ep[np.argmin(vcls)]), round(float(vcls[-1]), 3)],
            "lr_ep1_ep4_ep15_ep30": [float(lr[0]), float(lr[3]), float(lr[14]), float(lr[-1])],
            "cumlr_fit": fit,
        }
    return out


def fold_sizes() -> dict:
    counts: dict[int, dict] = {}
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            k = int(r["fold"])
            d = counts.setdefault(k, {"images": 0, "crops": set(), "sessions": set(), "groups": set()})
            d["images"] += 1
            d["crops"].add(r["crop_key"])
            d["sessions"].add(r["session"])
            d["groups"].add(r["split_group"])
    folds = {k: {"images": v["images"], "crops": len(v["crops"]), "sessions": len(v["sessions"]),
                 "groups": len(v["groups"])} for k, v in sorted(counts.items())}
    pool = sum(v["images"] for k, v in folds.items() if k >= 0)
    for k in range(5):
        n_tr = pool - folds[k]["images"]
        it = math.ceil(n_tr / BATCH)
        folds[k].update({"train_images": n_tr, "val_images": folds[k]["images"], "iters_per_epoch": it,
                         "auto_switch_to_MuSGD_from_epoch": math.floor(AUTO_SWITCH_ITERS / it) + 1})
    return {"pool_images": pool, "folds": folds}


def time_estimates(fs: dict) -> dict:
    res = {}
    for e in CANDIDATES:
        per_fold_a, per_fold_b = [], []
        for k in range(5):
            n_tr = fs["folds"][k]["train_images"]
            t_a = STEADY_S * n_tr / FOLD0_TRAIN      # model A: epoch time ~ train size
            t_b = STEADY_S                            # model B: train+val = pool, constant
            overhead = FIRST_EXTRA_S + 40.0           # first epoch + start-up + final best.pt val
            per_fold_a.append(e * t_a + overhead)
            per_fold_b.append(e * t_b + overhead)
        tot_a, tot_b = sum(per_fold_a), sum(per_fold_b)
        res[e] = {
            "per_fold_min_A": [round(x / 60, 1) for x in per_fold_a],
            "total_h_A": round(tot_a / 3600, 2),
            "total_h_B": round(tot_b / 3600, 2),
            "total_h_A_cpu_overlap": round(tot_a * CPU_OVERLAP_FACTOR / 3600, 2),
            "max_fold_min_A": round(max(per_fold_a) / 60, 1),
        }
    return res


def main() -> None:
    fs = fold_sizes()
    report = {"runs": analyse_runs(), "fold_sizes": fs, "time_estimates": time_estimates(fs)}
    out = ROOT / "analysis" / "detection" / "m110_epoch_budget.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
