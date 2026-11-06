"""M2-9 C step 1c: alternative QC rules on the reviewed candidates (diagnostic only).

Reads docs/processing/m29c_apex_review.csv (review labels + curvature metrics).
Thresholds below are fixed from physics / sampling arguments, not searched on the
labels (except the Youden band already reported in m29c_summary.json, labelled
as label-tuned). Ground truth depths are not read.

  band_0.7_1.3   : rule under test (out -> drop, unknown -> keep)
  band_0.7_1.6   : asymmetric upper bound (R=0 bias + small air gap + oblique)
  rms_meta_0.5   : picks fit by meta-v hyperbola within 1/(4 f_c) = 0.5 ns RMS
  band13_rms     : band_0.7_1.3 and (rms_meta <= 0.5 ns when known)
  band16_rms     : band_0.7_1.6 and (rms_meta <= 0.5 ns when known)
  unknown_drop   : band_0.7_1.3 with unknown dropped
  support_3      : >= 3 other lines of the activity with a candidate within 0.10 m

Writes docs/processing/m29c_alternatives.csv and prints radius-bias rows.
Run: conda run -n gpr-dev python -I analysis/processing/m29c_alternatives.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DOCS = Path(__file__).resolve().parents[2] / "docs" / "processing"
RMS_LIMIT_NS = 0.5


def main() -> None:
    d = pd.read_csv(DOCS / "m29c_apex_review.csv", dtype={"activity_id": str})
    true = d["type"] == "T"
    known = d["curv_flag"] != "unknown"
    r = d["r"].where(known)
    rms_ok = d["rms_meta_ns"].isna() | (d["rms_meta_ns"] <= RMS_LIMIT_NS) | ~known
    rules = {
        "band_0.7_1.3": ~(known & ~r.between(0.7, 1.3)),
        "band_0.7_1.6": ~(known & ~r.between(0.7, 1.6)),
        "rms_meta_0.5": rms_ok,
        "band13_rms": ~(known & ~r.between(0.7, 1.3)) & rms_ok,
        "band16_rms": ~(known & ~r.between(0.7, 1.6)) & rms_ok,
        "unknown_drop": known & r.between(0.7, 1.3),
        "support_3": d["support_lines"] >= 3,
        "band16_rms_support3": ~(known & ~r.between(0.7, 1.6)) & rms_ok & (d["support_lines"] >= 3),
    }
    rows = []
    for name, keep in rules.items():
        for scope, mask in {
            "all": np.ones(len(d), bool),
            "auto": (d["source"] == "auto_v1").to_numpy(),
            "bin0": (d["depth_bin"] == 0).to_numpy(),
            "bin1": (d["depth_bin"] == 1).to_numpy(),
            "bin2": (d["depth_bin"] == 2).to_numpy(),
        }.items():
            k, t = keep[mask], true[mask]
            n_keep = int(k.sum())
            rows.append(
                dict(
                    rule=name,
                    scope=scope,
                    n_true=int(t.sum()),
                    n_false=int((~t).sum()),
                    true_kept=int((k & t).sum()),
                    false_rejected=int((~k & ~t).sum()),
                    true_kept_rate=float((k & t).sum() / max(t.sum(), 1)),
                    false_rejected_rate=float((~k & ~t).sum() / max((~t).sum(), 1)),
                    kept_true_fraction=float((k & t).sum() / max(n_keep, 1)),
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(DOCS / "m29c_alternatives.csv", index=False, float_format="%.3f")
    print(table[table["scope"].isin(["all", "auto"])].round(3).to_string())
    print(table[table["scope"].str.startswith("bin")].round(3).to_string())

    print("support by type:")
    print(d.groupby("type")["support_lines"].describe()[["count", "25%", "50%", "75%"]])
    print("rms_meta by type:")
    print(d.groupby("type")["rms_meta_ns"].describe()[["count", "25%", "50%", "75%"]])

    b = pd.read_csv(DOCS / "m29c_refraction_bias.csv")
    sel = b[(b["eps_r"] == 9.0) & (b["half_aperture_m"] == 0.3) & b["h_m"].isin([0.0, 0.05, 0.1])]
    print(sel.pivot_table(index=["h_m", "radius_m"], columns="d_top_m", values="r").round(3))
    sel = b[(b["eps_r"] == 16.0) & (b["half_aperture_m"] == 0.3) & b["h_m"].isin([0.0, 0.05, 0.1])]
    print(sel.pivot_table(index=["h_m", "radius_m"], columns="d_top_m", values="r").round(3))
    print("aperture of T picks:", d.loc[true, "aperture_m"].describe().round(3).to_dict())


if __name__ == "__main__":
    main()
