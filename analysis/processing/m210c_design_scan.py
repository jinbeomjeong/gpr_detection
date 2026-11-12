"""M2-10 (c) design scan: where the QC flags of P3-P5 would act (already-seen 4TU outputs, counts only).

No new QC is computed. Reads the committed candidate csv of the M2-9 run and counts:
* lines (activity, path) carrying ``zero_window_drift`` and other zero flags, per site;
* candidates / QC-passing candidates on those lines, and whether their activity has truth pairs;
* activities whose meta eps_r equals 9.0 (the 4TU header default named in the CHECKLIST) per site;
* candidates with a straight-ray curvature ratio (``qc_r``) per site (input that P4 would use).

Output: docs/processing/m210c_design_scan.json (LF, relative paths only).
Run: conda run -n gpr-dev python -I analysis/processing/m210c_design_scan.py
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
    c = pd.read_csv(DOC / "depth_4tu_v2_candidates.csv", dtype=IDS)
    m = pd.read_csv(DOC / "depth_4tu_v2_matches.csv", dtype=IDS)
    main_pairs = m[(m.gt_set == "exclude_x") & (m.assumption == "top") & (m.zero_variant == "detected") & (m.variant == "main")]
    acts_with_pairs = set(main_pairs.activity_id)
    c["flags"] = c["flags"].fillna("")
    lines = c.groupby(["activity_id", "path"]).agg(
        site=("site", "first"),
        flags=("flags", lambda s: sorted({f for x in s for f in x.split() if f})),
        n_cand=("status", "size"),
        n_qc_in=("qc_verdict", lambda s: int((s == "in").sum())),
        eps=("eps_r_used", "first"),
        zero_rule=("zero_rule", "first"),
    ).reset_index()
    all_flags = sorted({f for fl in lines["flags"] for f in fl})
    out = {"note": "already-seen 4TU M2-9 outputs; counts only", "n_lines_with_candidates": int(len(lines)),
           "flag_line_counts": {f: int(sum(f in fl for fl in lines["flags"])) for f in all_flags}}
    drift = lines[lines["flags"].map(lambda fl: "zero_window_drift" in fl)]
    out["zero_window_drift"] = {
        "n_lines": int(len(drift)),
        "by_site": {str(k): int(v) for k, v in drift["site"].value_counts().sort_index().items()},
        "n_candidates": int(drift["n_cand"].sum()),
        "n_qc_in": int(drift["n_qc_in"].sum()),
        "n_activities": int(drift["activity_id"].nunique()),
        "n_activities_with_main_pairs": int(len(set(drift["activity_id"]) & acts_with_pairs)),
        "zero_rules": {str(k): int(v) for k, v in drift["zero_rule"].value_counts().items()},
    }
    acts = lines.groupby("activity_id").agg(site=("site", "first"), eps=("eps", "first")).reset_index()
    out["eps_equals_9_activities_by_site"] = {
        str(s): {"n_activities": int(len(g)), "n_eps9": int((g.eps == 9.0).sum())}
        for s, g in acts.groupby("site")
    }
    q = c[c.qc_r.notna()]
    out["qc_r_available_by_site"] = {
        str(s): {"n": int(len(g)), "n_in": int((g.qc_verdict == "in").sum()),
                 "median_in": round(float(g[g.qc_verdict == "in"].qc_r.median()), 3) if (g.qc_verdict == "in").any() else None}
        for s, g in q.groupby("site")
    }
    (DOC / "m210c_design_scan.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                                                encoding="utf-8", newline="\n")
    print(json.dumps({"out": "docs/processing/m210c_design_scan.json"}))


if __name__ == "__main__":
    main()
