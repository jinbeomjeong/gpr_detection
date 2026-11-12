"""M2-10 (a) design scan: structure of the M2-9 4TU outputs (already-seen data, read only).

Counts only - no eps fit, no bias decomposition (that is the analysis itself and
waits for the user decision). Reports, per site:
* meta eps_r values in the candidate csv (per site and whether an activity mixes values),
* number of main-path pairs (top / detected / exclude_x) and of unclustered ('b') pairs,
* how many pairs have a known pipe radius,
* gt depth range (to judge whether d-dependent offsets are separable from scale).

Inputs: docs/processing/depth_4tu_v2_matches.csv, depth_4tu_v2_candidates.csv.
Output: docs/processing/m210a_design_scan.json (LF, relative paths only).

Run: conda run -n gpr-dev python -I analysis/processing/m210a_design_scan.py
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "processing"
IDS = {"activity_id": str, "path": str, "line_id": str}


def main() -> None:
    m = pd.read_csv(DOC / "depth_4tu_v2_matches.csv", dtype=IDS)
    c = pd.read_csv(DOC / "depth_4tu_v2_candidates.csv", dtype=IDS)
    base = m[(m.gt_set == "exclude_x") & (m.assumption == "top") & (m.zero_variant == "detected")]
    out = {"note": "already-seen 4TU outputs; structure counts only, no fits", "sites": {}}
    eps_by_act = c.groupby("activity_id")["eps_r_used"].nunique()
    for site, g in c.groupby("site"):
        acts = g["activity_id"].unique()
        main = base[(base.variant == "main") & (base.site == site)]
        unc = base[(base.variant == "b") & (base.site == site)]
        out["sites"][str(site)] = {
            "eps_meta_values": sorted(round(float(x), 3) for x in g["eps_r_used"].dropna().unique()),
            "n_activities_candidates": int(len(acts)),
            "n_activities_mixed_eps": int((eps_by_act.reindex(acts) > 1).sum()),
            "n_pairs_main": int(len(main)),
            "n_pairs_main_radius_known": int(main["radius_known"].sum()) if len(main) else 0,
            "n_pairs_unclustered": int(len(unc)),
            "gt_depth_m_min_max_main": (
                [round(float(main.gt_depth_m.min()), 3), round(float(main.gt_depth_m.max()), 3)] if len(main) else None
            ),
            "n_activities_with_main_pairs": int(main["activity_id"].nunique()),
        }
    out["totals"] = {
        "n_pairs_main": int((base.variant == "main").sum()),
        "n_pairs_unclustered": int((base.variant == "b").sum()),
        "n_candidates_ok": int((c.status == "ok").sum()),
    }
    (DOC / "m210a_design_scan.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"out": "docs/processing/m210a_design_scan.json"}))


if __name__ == "__main__":
    main()
