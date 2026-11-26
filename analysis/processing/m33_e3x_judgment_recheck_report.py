"""Spot re-aggregation of report-only E3x items (R-a, R-c, R-b rows, HB1 contrast).

Reads runs/eval/m33/e3x_retry/injected/band_records.csv and
runs/eval/m33/e3x_retry/judgment/units_HH3.csv only.  Point values only
(no bootstrap); compared by hand with judgment.json report_only.
Output: analysis/processing/m33_e3x_judgment_recheck_report.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INJ = ROOT / "runs" / "eval" / "m33" / "e3x_retry" / "injected"
JDG = ROOT / "runs" / "eval" / "m33" / "e3x_retry" / "judgment"
OUT = ROOT / "analysis" / "processing" / "m33_e3x_judgment_recheck_report.json"
BGS = ("4tu", "gz_ids")

b = pd.read_csv(INJ / "band_records.csv", dtype={"cluster_id": str, "line_id": str}, low_memory=False)
b = b[b.accepted.astype(str).str.lower().isin(["true", "1"])]
unit_cols = ["position_id", "bin_target_px", "length_arm", "texture", "recipe", "snr_db", "polarity"]
u = b.groupby(unit_cols, as_index=False).agg(y=("coverage", "mean"), dataset=("dataset", "first"),
                                             c_lat=("c_lat", "first"), coh=("coh", "first"))


def arm(df, row, length, texture, recipe, snrs):
    s = df[(df.bin_target_px == row) & (df.length_arm == length) & (df.texture == texture)
           & (df.recipe == recipe) & (df.snr_db.isin(snrs))]
    g = s.groupby("position_id")
    n = g.size()
    full = n[n == 2 * len(snrs)].index
    return g.y.mean().loc[full], s.groupby("position_id").dataset.first()


POS_DS = b.groupby("position_id").dataset.first()


def bg_mean(diff, ds=None):
    ds = POS_DS
    vals = [diff[ds.loc[diff.index] == g].mean() for g in BGS if (ds.loc[diff.index] == g).any()]
    return float(np.mean(vals))


res = {}
# R-a: 24 - 18 dB, long arm, all four rows? -> report both variants
for rows, name in (((28, 47, 68, 90), "rows_all_long"), ((47, 68), "rows_47_68_long")):
    diffs = []
    for r in rows:
        a24, ds = arm(u, r, "long", "none", "r0", [24.0])
        a18, _ = arm(u, r, "long", "none", "r0", [18.0])
        diffs.append((a24 - a18).dropna().rename(r))
    d = pd.concat(diffs, axis=1)
    dm = d.mean(axis=1, skipna=True)
    res[f"R-a_{name}"] = dict(estimate=bg_mean(dm, ds), n_positions=int(dm.notna().sum()))
# R-c: texture - flat, long arm, 47/68, 12/18 dB
for tex in ("amp", "jitter0.1", "jitter0.2"):
    parts = []
    for r in (47, 68):
        t, ds = arm(u, r, "long", tex, "r0", [12.0, 18.0])
        f, _ = arm(u, r, "long", "none", "r0", [12.0, 18.0])
        parts.append((t - f).rename(r))
    d = pd.concat(parts, axis=1).mean(axis=1)
    sub = u[(u.texture == tex) & (u.length_arm == "long") & (u.bin_target_px.isin([47, 68]))
            & (u.recipe == "r0") & (u.snr_db.isin([12.0, 18.0]))]
    res[f"R-c_{tex}"] = dict(estimate=bg_mean(d, ds), n_positions=int(len(d)),
                             coh_median=float(sub.coh.median()), c_lat_median=float(sub.c_lat.median()))
sub = u[(u.texture == "none") & (u.length_arm == "long") & (u.bin_target_px.isin([47, 68]))
        & (u.recipe == "r0") & (u.snr_db.isin([12.0, 18.0]))]
res["R-c_none"] = dict(coh_median=float(sub.coh.median()), c_lat_median=float(sub.c_lat.median()))
# HB1 contrast medians: injected c_lat by row, long arm, r0, none, 12/18 dB (unit level)
med = {}
for r in (28, 47, 68, 90):
    s = u[(u.bin_target_px == r) & (u.length_arm == "long") & (u.texture == "none") & (u.recipe == "r0")
          & (u.snr_db.isin([12.0, 18.0]))]
    med[str(r)] = dict(unit_level=float(s.c_lat.median()), n_units=int(len(s)))
    s2 = b[(b.bin_target_px == r) & (b.length_arm == "long") & (b.texture == "none") & (b.recipe == "r0")
           & (b.snr_db.isin([12.0, 18.0]))]
    med[str(r)]["record_level"] = float(s2.c_lat.median())
res["HB1_contrast_c_lat_median_db"] = med
# R-b compression and realized rows
comp = b.groupby("recipe").compression.describe()[["count", "min", "50%", "max"]]
res["R-b_compression"] = comp.to_dict(orient="index")
# HH3 by background: simple position mean vs model-weighted
hh3 = pd.read_csv(JDG / "units_HH3.csv", dtype={"cluster": str})
res["HH3_by_background"] = {
    g: dict(simple_mean=float(hh3[hh3.background == g].difference.mean()),
            model_weighted=float(hh3[hh3.background == g].groupby("model").difference.mean().mean()),
            n=int((hh3.background == g).sum()))
    for g in BGS}
hh1 = pd.read_csv(JDG / "units_HH1.csv", dtype={"cluster": str})
res["HH1_by_background"] = {
    g: dict(simple_mean=float(hh1[hh1.background == g].difference.mean()),
            model_weighted=float(hh1[hh1.background == g].groupby("model").difference.mean().mean()),
            n=int((hh1.background == g).sum()))
    for g in BGS}
res["HH1_positions_by_model"] = hh1.groupby("model").size().to_dict()
res["HH1_model_mean_D"] = hh1.groupby("model").difference.mean().round(6).to_dict()
res["HH3_model_mean_D"] = hh3.groupby("model").difference.mean().round(6).to_dict()
for k in ("HB1", "HB1b", "HB2"):
    df = pd.read_csv(JDG / f"units_{k}.csv", dtype={"cluster": str})
    res[f"{k}_positions_by_background"] = df.groupby("background").size().to_dict()
    res[f"{k}_n_negative_D"] = int((df.difference < 0).sum())
    res[f"{k}_n_positive_D"] = int((df.difference > 0).sum())
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps({"status": "ok", "out": str(OUT.relative_to(ROOT))}))
