"""GPR-SD: are the four roads independent units? (plan docs/processing/m36_gprsd_plan.md 2-3, 6 item 4)

Evidence only from metadata (marks.csv, radargrams_index.csv) and a coarse amplitude profile of the stored npy
(block-mean |amp|, no pipe/hyperbola detection, no M3-6 output). Read-only on the dataset.

Run: conda run -n gpr-dev python -I analysis/data/gprsd_m34_road_independence.py
Output: data/labels/gprsd_m34/road_independence.json (git-ignored) and a stdout summary.
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PKG = Path(r"C:\workspace\GPR_Data\GPR-SD_중국USTC_도로GPR_가공본_261016")
SEED = 20261126
BLOCK = 50  # traces per block (1.5 m)
MAXLAG = 120  # blocks (6000 traces)


def main() -> int:
    with open(PKG / "01_메타데이터" / "marks.csv", encoding="utf-8-sig", newline="") as f:
        marks = list(csv.DictReader(f))
    with open(PKG / "01_메타데이터" / "radargrams_index.csv", encoding="utf-8-sig", newline="") as f:
        index = {r["file_id"]: r for r in csv.DictReader(f)}
    roads = sorted({m["road"] for m in marks})
    surface = {r["road"]: r["surface"] for r in index.values()}
    length = {r: [int(v["n_traces"]) for v in index.values() if v["road"] == r] for r in roads}

    pos = {r: sorted((int(m["location_trace"]), m["type"]) for m in marks if m["road"] == r) for r in roads}
    rng = np.random.default_rng(SEED)
    out: dict = {"seed": SEED, "surface": surface, "n_traces_per_line": length, "n_marks": {r: len(pos[r]) for r in roads}}

    # A. mark coincidences (exact equal trace number) for all pairs, with shift null
    pairs = {}
    for a, b in itertools.combinations(roads, 2):
        ta = np.array([p[0] for p in pos[a]])
        tb = np.array([p[0] for p in pos[b]])
        la, lb = max(length[a]), max(length[b])
        L = min(la, lb)
        coin = [(int(x), ta_t, tb_t) for (x, ta_t) in pos[a] for (y, tb_t) in pos[b] if x == y]
        obs = len(coin)
        null = []
        for _ in range(4000):
            sh = int(rng.integers(0, L // 25)) * 25
            tb_s = (tb + sh) % L
            null.append(int(sum(int(x) in set(tb_s.tolist()) for x in ta)))
        null = np.array(null)
        pairs[f"{a}|{b}"] = {
            "observed_same_trace": obs,
            "coincidences": [{"trace": c[0], "type_a": c[1], "type_b": c[2]} for c in coin],
            "same_type_at_coincidence": sum(c[1] == c[2] for c in coin),
            "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p_ge_obs": float((null >= obs).mean()),
        }
    out["mark_coincidence_pairs"] = pairs

    # B. block-mean log|amp| profile correlation between lines
    prof: dict[str, np.ndarray] = {}
    for fid in sorted(index):
        a = np.load(PKG / "05_radargrams_npy" / f"{fid}.npy", mmap_mode="r")
        n = (a.shape[1] // BLOCK) * BLOCK
        m = np.abs(np.asarray(a[:, :n])).reshape(a.shape[0], n // BLOCK, BLOCK).mean(axis=(0, 2))
        prof[fid] = np.log(m + 1e-6)

    def corr(x: np.ndarray, y: np.ndarray, lag: int) -> float:
        if lag >= 0:
            x2, y2 = x[lag:], y[: len(y) - lag] if lag else y
        else:
            x2, y2 = x[: len(x) + lag], y[-lag:]
        n = min(len(x2), len(y2))
        x2, y2 = x2[:n], y2[:n]
        if n < 50:
            return float("nan")
        return float(np.corrcoef(x2, y2)[0, 1])

    fids = sorted(index)
    cm = {}
    for fa, fb in itertools.combinations(fids, 2):
        c0 = corr(prof[fa], prof[fb], 0)
        best = max(((corr(prof[fa], prof[fb], lag), lag) for lag in range(-MAXLAG, MAXLAG + 1)), key=lambda t: (np.nan_to_num(t[0], nan=-9)))
        cm[f"{fa}|{fb}"] = {"corr_lag0": c0, "best_corr": best[0], "best_lag_blocks": best[1], "same_road": index[fa]["road"] == index[fb]["road"]}
    out["profile_corr"] = cm

    def summ(sel):
        v = [x["corr_lag0"] for x in cm.values() if sel(x)]
        return {"n": len(v), "min": float(np.nanmin(v)), "median": float(np.nanmedian(v)), "max": float(np.nanmax(v))}

    out["profile_corr_summary_lag0"] = {
        "same_road": summ(lambda x: x["same_road"]),
        "different_road": summ(lambda x: not x["same_road"]),
    }
    road_of = {f: index[f]["road"] for f in fids}
    by_pair: dict[str, list] = {}
    for k, v in cm.items():
        fa, fb = k.split("|")
        ra, rb = road_of[fa], road_of[fb]
        key = f"{ra}|{rb}" if ra <= rb else f"{rb}|{ra}"
        by_pair.setdefault(key, []).append(v["corr_lag0"])
    out["profile_corr_by_road_pair_lag0_median"] = {k: float(np.nanmedian(v)) for k, v in sorted(by_pair.items())}

    # B2. same comparison after removing the slow trend (moving mean over 21 blocks = 31.5 m)
    def hp(x: np.ndarray) -> np.ndarray:
        k = np.ones(21) / 21
        return x - np.convolve(np.pad(x, 10, mode="edge"), k, mode="valid")

    hpd = {f: hp(prof[f]) for f in fids}
    hp_by: dict[str, list] = {}
    hp_best_lag: dict[str, list] = {}
    for fa, fb in itertools.combinations(fids, 2):
        ra, rb = road_of[fa], road_of[fb]
        key = f"{ra}|{rb}" if ra <= rb else f"{rb}|{ra}"
        lags = [(corr(hpd[fa], hpd[fb], lag), lag) for lag in range(-MAXLAG, MAXLAG + 1)]
        b = max(lags, key=lambda t: np.nan_to_num(t[0], nan=-9))
        hp_by.setdefault(key, []).append(corr(hpd[fa], hpd[fb], 0))
        hp_best_lag.setdefault(key, []).append(b[1])
    out["hp_corr_by_road_pair_lag0_median"] = {k: float(np.nanmedian(v)) for k, v in sorted(hp_by.items())}
    out["hp_best_lag_blocks_by_road_pair_median"] = {k: float(np.median(v)) for k, v in sorted(hp_best_lag.items())}
    print(json.dumps({"hp_lag0_median": out["hp_corr_by_road_pair_lag0_median"], "hp_best_lag_median": out["hp_best_lag_blocks_by_road_pair_median"]}, indent=1))

    p = REPO / "data" / "labels" / "gprsd_m34" / "road_independence.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("surface", "n_marks", "profile_corr_summary_lag0", "profile_corr_by_road_pair_lag0_median")}, ensure_ascii=False, indent=1))
    for k, v in pairs.items():
        print(k, "obs", v["observed_same_trace"], "same_type", v["same_type_at_coincidence"], "null_mean %.2f p95 %.0f p>=obs %.3f" % (v["null_mean"], v["null_p95"], v["p_ge_obs"]), [(c["trace"], c["type_a"], c["type_b"]) for c in v["coincidences"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
