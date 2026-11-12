"""M2-10 (g) U-2 record: read-only aggregation of the M2-9 match csv (already-seen 4TU, exploration).

Counts the pipe radii behind the top/centre assumptions and the size of the
residual bias, and fits the one R-sensitive check that does not need the bias
level: regress the top-assumption error on the radius (with the depth as a second
regressor). If the truth is the pipe top, the R slope is ~0; if it is the centre,
~ -1 (e_top = e_centre - R). Activity bootstrap, seed 20261131 (checked unused).
No rule, no threshold, no judgement.

Inputs: docs/processing/depth_4tu_v2_matches.csv, data/processing/m210d_truth_full.csv (R per truth;
(activity, depth) keys with more than one radius are left out)
Output: docs/processing/m210g_u2_scan.json (LF, relative paths)
Run: conda run -n gpr-dev python -I analysis/processing/m210g_u2_scan.py
"""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs" / "processing" / "depth_4tu_v2_matches.csv"
TRUTH = ROOT / "data" / "processing" / "m210d_truth_full.csv"
OUT = ROOT / "docs" / "processing" / "m210g_u2_scan.json"
SEED, N_BOOT = 20261131, 2000


def fit(df):
    x = np.column_stack([np.ones(len(df)), df["R"].values, df["d"].values])
    coef, *_ = np.linalg.lstsq(x, df["e_top"].values, rcond=None)
    return coef


def main() -> None:
    m = pd.read_csv(SRC, dtype={"activity_id": str})
    base = m[(m.gt_set == "exclude_x") & (m.zero_variant == "detected")]
    t = pd.read_csv(TRUTH, dtype={"activity_id": str})
    t = t[t.radius_known]
    grp = t.groupby(["activity_id", t.gt_depth_m.round(4)]).gt_center_shift_m
    rmap = {k: float(v.iloc[0]) for k, v in grp if v.nunique() == 1}
    out = {"note": "already-seen 4TU M2-9 matches; exploration only, U-2 is not decided here",
           "input_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
           "truth_sha256": hashlib.sha256(TRUTH.read_bytes()).hexdigest(), "seed": SEED}
    for variant in ("main", "b"):
        top = base[(base.variant == variant) & (base.assumption == "top")].reset_index(drop=True)
        cen = base[(base.variant == variant) & (base.assumption == "center")].reset_index(drop=True)
        same_pairs = (len(top) == len(cen)) and bool(
            (top[["activity_id", "est_depth_m"]].values == cen[["activity_id", "est_depth_m"]].values).all()
        )
        rk = top[top.radius_known].copy()
        res = {
            "n_pairs": int(len(top)),
            "n_radius_known": int(len(rk)),
            "pairs_identical_between_assumptions": same_pairs,
            "bias_top_all": round(float(top.error_m.mean()), 4),
            "bias_center_all": round(float(cen.error_m.mean()), 4),
        }
        if True:
            top["R"] = [rmap.get((a, round(float(d), 4)), np.nan) for a, d in zip(top.activity_id, top.gt_depth_m)]
            rk = top[top.radius_known & top.R.notna()].copy()
            res["n_radius_known_unambiguous"] = int(len(rk))
            res["R_m_q10_q50_q90_radius_known"] = [round(float(q), 3) for q in np.quantile(rk.R, [0.1, 0.5, 0.9])]
            res["R_m_min_max"] = [round(float(rk.R.min()), 3), round(float(rk.R.max()), 3)]
            res["R_sd_m"] = round(float(rk.R.std()), 3)
            res["sites_radius_known"] = {str(k): int(v) for k, v in rk.site.value_counts().sort_index().items()}
            res["bias_top_radius_known"] = round(float(rk.error_m.mean()), 4)
            res["bias_center_radius_known"] = round(float(rk.error_m.mean() + rk.R.mean()), 4)  # e_centre = e_top + R
            rk = rk.assign(e_top=rk.error_m, d=rk.gt_depth_m)
            coef = fit(rk)
            acts = rk.activity_id.unique()
            by = {a: rk[rk.activity_id == a] for a in acts}
            rng = np.random.default_rng(SEED)
            bs = np.array([fit(pd.concat([by[a] for a in rng.choice(acts, len(acts))])) for _ in range(N_BOOT)])
            res["regress_e_top_on_R_and_d"] = {
                "n": int(len(rk)), "n_activities": int(len(acts)),
                "coef_R": round(float(coef[1]), 3),
                "coef_R_ci95": [round(float(np.quantile(bs[:, 1], 0.025)), 3), round(float(np.quantile(bs[:, 1], 0.975)), 3)],
                "coef_d": round(float(coef[2]), 3),
                "expect": "top truth -> coef_R ~ 0; centre truth -> coef_R ~ -1",
            }
        out[variant] = res
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"out": "docs/processing/m210g_u2_scan.json"}))


if __name__ == "__main__":
    main()
