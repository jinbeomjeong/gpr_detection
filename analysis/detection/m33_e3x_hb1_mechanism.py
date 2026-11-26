"""Post-judgment exploration of the E3x HB1 mechanism and the R-f table (report only).

Reads (read only):
  * runs/eval/m33/e3x_retry/injected/{band_records,hyperbola_records}.csv
  * runs/eval/m33/e3x_retry/judgment/judgment.json
  * data/splits/aihub_yolo_v1.csv + data/aihub_yolo_v1/labels/*.txt (train_pool only,
    one gain copy per crop_key; the test sessions are never read)
Never reads the first-attempt folders runs/eval/m33/e3x/{uninjected001,injected,judgment}.

Writes analysis/detection/m33_e3x_hb1_mechanism.json. Labels and decisions are not touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RETRY = ROOT / "runs" / "eval" / "m33" / "e3x_retry"
BAND = RETRY / "injected" / "band_records.csv"
HYP = RETRY / "injected" / "hyperbola_records.csv"
JUDG = RETRY / "judgment" / "judgment.json"
SPLIT = ROOT / "data" / "splits" / "aihub_yolo_v1.csv"
LABELS = ROOT / "data" / "aihub_yolo_v1" / "labels"
OUT = ROOT / "analysis" / "detection" / "m33_e3x_hb1_mechanism.json"

IMG_H = 256.0
TRANSLATE = 0.1          # Ultralytics translate (N2-a' config)
P_CONTEXT = 0.5
CONTEXT_SCALE = (1.0, 1.5)
PLAIN_SCALE = (0.55, 1.5)
MC_DRAWS = 40
MC_SEED = 20261126
PROBE_ROWS = [20, 28, 33, 47, 48, 63, 68, 90]
WINDOW = 4.0             # +-px window for training density at a probe row


def r(x, nd=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def training_prior() -> dict:
    split = pd.read_csv(SPLIT, dtype=str)
    pool = split[split["subset"] == "trainpool"].drop_duplicates("crop_key")
    rows = {0: [], 1: []}
    n_files = 0
    for name in pool["file_name"]:
        lab = LABELS / (Path(name).stem + ".txt")
        if not lab.exists():
            continue
        n_files += 1
        for line in lab.read_text().split("\n"):
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(parts[0])
            if c not in rows:
                continue
            yc, h = float(parts[2]) * IMG_H, float(parts[4]) * IMG_H
            rows[c].append((yc - h / 2.0, yc + h / 2.0))
    out = {"n_trainpool_crops_read": n_files}
    rng = np.random.default_rng(MC_SEED)
    for c, nm in ((0, "pipe_hyperbola"), (1, "pipe_band")):
        a = np.asarray(rows[c])
        y1, y2 = a[:, 0], a[:, 1]
        q = np.percentile(y1, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        raw = {"n_boxes": int(len(a)), "y1_quantiles_px_0_1_5_25_50_75_95_99_100": [r(v, 2) for v in q]}
        # Monte Carlo of the N2-a' geometric chain on the box top row:
        # top-anchored isotropic scale, then Ultralytics vertical translate.
        n = len(a)
        ctx = rng.random((MC_DRAWS, n)) < P_CONTEXT
        s = np.where(ctx, rng.uniform(*CONTEXT_SCALE, (MC_DRAWS, n)), rng.uniform(*PLAIN_SCALE, (MC_DRAWS, n)))
        dy = rng.uniform(-TRANSLATE, TRANSLATE, (MC_DRAWS, n)) * IMG_H
        t1 = s * y1[None, :] + dy
        t2 = s * y2[None, :] + dy
        vis = (t2 > 0) & (t1 < IMG_H)
        t1v = np.clip(t1[vis], 0, IMG_H)
        qa = np.percentile(t1v, [0, 1, 5, 25, 50, 75, 95])
        dens = {}
        for pr in PROBE_ROWS:
            dens[str(pr)] = r(np.mean(np.abs(t1v - pr) <= WINDOW), 5)
        raw_dens = {str(pr): r(np.mean(np.abs(y1 - pr) <= WINDOW), 5) for pr in PROBE_ROWS}
        out[nm] = {
            "raw": raw,
            "raw_fraction_y1_within_4px": raw_dens,
            "raw_fraction_y1_below_37": r(np.mean(y1 < 37.0), 5),
            "aug_y1_quantiles_px_0_1_5_25_50_75_95": [r(v, 2) for v in qa],
            "aug_fraction_y1_within_4px": dens,
            "aug_fraction_y1_below_37": r(np.mean(t1v < 37.0), 5),
            "aug_fraction_y1_below_30": r(np.mean(t1v < 30.0), 5),
            "aug_fraction_y1_below_24": r(np.mean(t1v < 24.0), 5),
        }
    return out


def unit_means(df: pd.DataFrame, keys: list[str], col: str) -> pd.DataFrame:
    return df.groupby(keys, dropna=False)[col].mean().reset_index()


def band_analysis() -> dict:
    df = pd.read_csv(BAND, dtype={"cluster_id": str, "position_id": str, "line_id": str})
    df = df[df["accepted"] == True]  # noqa: E712
    keys = ["position_id", "dataset", "bin_target_px", "length_arm", "texture", "recipe", "snr_db", "polarity"]
    df["no_box"] = df["best_score_001_band_no_box"].astype(str) == "True"
    df["any_cov"] = df["covered_any_class"].astype(str) == "True"
    agg = df.groupby(keys).agg(
        coverage=("coverage", "mean"),
        y1_img=("realized_y1_px_image", "median"),
        apex_P=("apex_P_below_zero_target", "median"),
        to_end_P=("rows_to_record_end_P", "median"),
        compression=("compression", "median"),
        no_box=("no_box", "mean"),
        best=("best_score_001_band", "median"),
        c_lat=("c_lat", "median"),
        c_vert=("c_vert", "median"),
        c_vert_bg=("c_vert_bg", "median"),
        period_inj=("period_inj", "median"),
    ).reset_index()
    res = {}
    base = agg[(agg.length_arm == "long") & (agg.texture == "none")]

    # (1) r0 long/flat, 12+18 dB: by row, overall and per dataset (unit mean then position mean)
    r0 = base[(base.recipe == "r0") & (base.snr_db.isin([12.0, 18.0]))]
    tab = []
    for (ds, b), g in r0.groupby(["dataset", "bin_target_px"]):
        pos = g.groupby("position_id")["coverage"].mean()
        tab.append({"dataset": ds, "row": b, "n_pos": int(len(pos)), "coverage": r(pos.mean()),
                    "y1_img_med": r(g.y1_img.median(), 2), "apex_P_med": r(g.apex_P.median(), 2),
                    "to_end_P_med": r(g.to_end_P.median(), 2), "no_box_share": r(g.no_box.mean()),
                    "best_score_med": r(g.best.median()), "c_lat_med": r(g.c_lat.median(), 3),
                    "c_vert_med": r(g.c_vert.median(), 3), "c_vert_bg_med": r(g.c_vert_bg.median(), 3)})
    res["r0_long_flat_12_18_by_row"] = tab

    # (2) R-b alignment: long/flat 18 dB, positions with all four rows in both recipes
    s18 = base[base.snr_db == 18.0]
    piv = s18.pivot_table(index=["position_id", "dataset"], columns=["recipe", "bin_target_px"],
                          values="coverage", aggfunc="mean")
    y1p = s18.pivot_table(index=["position_id", "dataset"], columns=["recipe", "bin_target_px"],
                          values="y1_img", aggfunc="median")
    rows4 = [28.0, 47.0, 68.0, 90.0]
    cols = [(rc, b) for rc in ("r0", "e3x_p42") for b in rows4]
    full = piv.dropna(subset=[c for c in cols if c in piv.columns])
    curves = {}
    for scope, sub in (("all", full), ("4tu", full.xs("4tu", level="dataset", drop_level=False)),
                       ("ids", full.xs("gz_ids", level="dataset", drop_level=False)
                        if "gz_ids" in full.index.get_level_values("dataset") else full.iloc[0:0])):
        if len(sub) == 0:
            continue
        pts = []
        for rc in ("r0", "e3x_p42"):
            for b in rows4:
                pts.append({"recipe": rc, "row_target": b,
                            "y1_img_med": r(y1p.loc[sub.index, (rc, b)].median(), 2),
                            "coverage": r(sub[(rc, b)].mean())})
        # Alignment test. Model P (period-unit / zero-distance): R'(b) = R(b) + c.
        # Model px (absolute px row): R'(b) = R_r0(px'_b) + c, R_r0 linear interp in px.
        r0x = np.array([p["y1_img_med"] for p in pts[:4]])
        r0y = np.array([p["coverage"] for p in pts[:4]])
        p42x = np.array([p["y1_img_med"] for p in pts[4:]])
        p42y = np.array([p["coverage"] for p in pts[4:]])
        idx = [1, 2, 3]  # 47, 68, 90 -> 33, 48, 63 (inside the r0 px range)
        predP = r0y[idx]
        predX = np.interp(p42x[idx], r0x, r0y)
        cP = float(np.mean(p42y[idx] - predP))
        cX = float(np.mean(p42y[idx] - predX))
        rssP = float(np.sum((p42y[idx] - predP - cP) ** 2))
        rssX = float(np.sum((p42y[idx] - predX - cX) ** 2))
        curves[scope] = {"n_positions": int(len(sub)), "points": pts,
                         "model_P_offset": r(cP), "model_P_rss": r(rssP, 6),
                         "model_px_offset": r(cX), "model_px_rss": r(rssX, 6),
                         "model_px_pred_33_48_63": [r(v) for v in predX],
                         "p42_obs_33_48_63": [r(v) for v in p42y[idx]],
                         "p42_row20_vs_r0_row28": [r(p42y[0]), r(r0y[0])]}
    res["R_b_alignment_long_flat_18db"] = curves

    # (3) zero-band proximity check inside r0 28 px arms: does coverage track c_vert_bg
    #     (background vertical contrast; zero band in its denominator makes it negative)?
    prox = {}
    for b in (28.0, 47.0, 68.0):
        g = r0[r0.bin_target_px == b]
        pos = g.groupby(["position_id", "dataset"]).agg(cov=("coverage", "mean"), cvb=("c_vert_bg", "median"),
                                                         apexP=("apex_P", "median"), y1=("y1_img", "median"))
        out_b = {}
        for ds in ("4tu", "gz_ids"):
            p = pos.xs(ds, level="dataset") if ds in pos.index.get_level_values("dataset") else None
            if p is None or len(p) < 5:
                continue
            out_b[ds] = {"n": int(len(p)),
                         "spearman_cov_vs_c_vert_bg": r(p["cov"].corr(p["cvb"], method="spearman"), 3),
                         "spearman_cov_vs_apex_P": r(p["cov"].corr(p["apexP"], method="spearman"), 3),
                         "apex_P_range": [r(p.apexP.min(), 2), r(p.apexP.max(), 2)],
                         "c_vert_bg_tertile_cov": [r(v) for v in p.groupby(pd.qcut(p["cvb"], 3, labels=False,
                                                                                      duplicates="drop"))["cov"].mean()]}
        prox[str(int(b))] = out_b
    res["r0_proximity_within_row"] = prox

    # (4) collinearity of px row and P-below-zero in r0 (zero row in image px)
    rr = df[(df.recipe == "r0")]
    p_img = rr["realized_y1_px_image"] / rr["apex_P_below_zero_target"].where(rr["apex_P_below_zero_target"] > 0)
    res["r0_px_per_P_below_zero_quantiles_5_50_95"] = [r(v, 3) for v in np.nanpercentile(p_img, [5, 50, 95])]
    return res


def hyperbola_analysis() -> dict:
    df = pd.read_csv(HYP, dtype={"cluster_id": str, "position_id": str, "line_id": str})
    df = df[(df["accepted"] == True) & (df["taper"] == "on")]  # noqa: E712
    df["hit_f"] = df["hit"].astype(str) == "True"
    sj = df[df.snr_db.isin([12.0, 18.0])]
    pos = sj.groupby(["position_id", "dataset", "model"]).agg(y=("hit_f", "mean"), apex_px=("apex_y_px", "median"),
                                                              apex_P=("apex_depth_P", "median")).reset_index()
    bins = [0, 28, 37, 47, 57, 79, 400]
    pos["apex_bin"] = pd.cut(pos["apex_px"], bins, right=False)
    by_bin = [{"apex_px_bin": str(k), "n_pos": int(len(g)), "Y_on_mean": r(g.y.mean())}
              for k, g in pos.groupby("apex_bin", observed=True)]
    # R-f stability: label per dataset and per fold with the sealed thresholds (report only)
    def label(g):
        n = len(g)
        if n >= 6 and (g >= 0.5).sum() >= 0.75 * n:
            return "informative"
        return "variable_or_other"
    per_model = []
    for m, g in pos.groupby("model"):
        row = {"model": m, "n_pos": int(len(g)), "Y_on_mean": r(g.y.mean()),
               "share_pos_ge_0.5": r((g.y >= 0.5).mean()), "label_all": label(g.y),
               "apex_px_med": r(g.apex_px.median(), 1)}
        for ds in ("4tu", "gz_ids"):
            gd = g[g.dataset == ds]
            row[f"Y_on_{ds}"] = r(gd.y.mean()) if len(gd) else None
            row[f"n_{ds}"] = int(len(gd))
        fl = []
        for f in sorted(sj.fold.unique()):
            gf = sj[(sj.model == m) & (sj.fold == f)].groupby("position_id")["hit_f"].mean()
            fl.append(label(gf))
        row["label_by_fold"] = fl
        per_model.append(row)
    return {"hyperbola_on_12_18_by_apex_px": by_bin, "R_f_stability": per_model}


def main() -> None:
    judg = json.loads(JUDG.read_text(encoding="utf-8"))
    out = {
        "note": "report only, post-judgment exploration; labels unchanged",
        "inputs": {"band_records": str(BAND.relative_to(ROOT)), "hyperbola_records": str(HYP.relative_to(ROOT)),
                   "judgment": str(JUDG.relative_to(ROOT)), "split": str(SPLIT.relative_to(ROOT))},
        "mc": {"seed": MC_SEED, "draws_per_box": MC_DRAWS, "translate": TRANSLATE, "p_context": P_CONTEXT,
               "context_scale": CONTEXT_SCALE, "plain_scale": PLAIN_SCALE, "window_px": WINDOW},
        "R_f_from_judgment": judg["report_only"]["R-f"],
        "R_b_from_judgment": judg["report_only"]["R-b"],
        "training_prior": training_prior(),
        "band": band_analysis(),
        "hyperbola": hyperbola_analysis(),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"status": "ok", "out": str(OUT.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
