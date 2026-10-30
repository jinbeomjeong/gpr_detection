"""4TU inventory for M2-8: activities, lines, traces, dt, eps (no ground-truth depths read).

Read-only on C:\\workspace\\GPR_Data. Prints a per-site / per-activity summary and
writes docs/processing/fourtu_lines_v1.csv (one row per line, metadata only).
Run:  conda run -n gpr-dev python -I analysis/processing/fourtu_inventory.py
"""

from pathlib import Path

import pandas as pd

from gpr_detection.io.fourtu import FourTUTable, load_fourtu_table, site_of_activity

BASE = Path(r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015")
OUT = Path(__file__).resolve().parents[2] / "docs" / "processing"


def main() -> None:
    idx = load_fourtu_table(BASE, FourTUTable.RADARGRAMS_INDEX)
    for col in ("n_samples", "n_traces", "dt_ns", "eps_r_header", "eps_r_used", "length_m"):
        idx[col] = pd.to_numeric(idx[col])
    idx["site"] = idx["activity_id"].map(site_of_activity)
    npy = BASE / "05_radargrams_npy"
    idx["npy_exists"] = [
        (npy / a / f"{p}.npy").exists() for a, p in zip(idx["activity_id"], idx["path"], strict=True)
    ]
    cols = ["activity_id", "site", "path", "n_samples", "n_traces", "dt_ns", "length_m",
            "eps_r_header", "eps_r_used", "control_unit", "npy_exists"]
    idx[cols].to_csv(OUT / "fourtu_lines_v1.csv", index=False, encoding="utf-8")
    print(f"lines {len(idx)}, activities {idx['activity_id'].nunique()}, sites {idx['site'].nunique()}, "
          f"npy missing {(~idx['npy_exists']).sum()}, total traces {int(idx['n_traces'].sum())}")
    g = idx.groupby("site").agg(
        activities=("activity_id", "nunique"), lines=("path", "size"),
        traces=("n_traces", "sum"), dt=("dt_ns", lambda s: "/".join(sorted({f"{v:g}" for v in s}))),
        eps_used=("eps_r_used", lambda s: "/".join(sorted({f"{v:g}" for v in s}))),
        eq_header=("eps_r_used", "size"),
    )
    eq = idx.assign(eq=idx["eps_r_used"] == idx["eps_r_header"]).groupby("site")["eq"].mean()
    g["eps_used_eq_header_frac"] = eq.round(2)
    g["long_lines_ge30m"] = idx.assign(lg=idx["length_m"] >= 30).groupby("site")["lg"].sum()
    print(g.drop(columns="eq_header").to_string())


if __name__ == "__main__":
    main()
