"""Summaries of fourtu_apex_candidates_v1.csv / fourtu_apex_candidate_lines_v1.csv (no ground truth read).

Writes docs/processing/fourtu_apex_candidates_v1_summary.csv (per activity) and prints the
per-site table, the no-candidate reasons and the manual_v0 recall.
Run:  conda run -n gpr-dev python -I analysis/processing/fourtu_apex_candidate_summary.py
"""

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "docs" / "processing"


def main() -> None:
    candidates = pd.read_csv(
        OUT / "fourtu_apex_candidates_v1.csv", dtype={"activity_id": str}
    )
    lines = pd.read_csv(
        OUT / "fourtu_apex_candidate_lines_v1.csv", dtype={"activity_id": str}
    )
    lines["n_apex_candidates"] = lines["n_apex_candidates"].fillna(0).astype(int)
    auto = candidates[candidates["source"] == "auto_v1"]
    manual = candidates[candidates["source"] == "manual_v0"]
    per_act = lines.groupby("activity_id").agg(
        site=("site", "first"),
        lines=("path", "size"),
        lines_with_apex_candidate=("n_apex_candidates", lambda s: int((s > 0).sum())),
        auto_apex_candidates=("n_apex_candidates", "sum"),
    )
    per_act["manual_apex_candidates"] = (
        manual.groupby("activity_id")
        .size()
        .reindex(per_act.index)
        .fillna(0)
        .astype(int)
    )
    per_act.to_csv(OUT / "fourtu_apex_candidates_v1_summary.csv", encoding="utf-8")
    site = per_act.groupby("site").agg(
        activities=("lines", "size"),
        lines=("lines", "sum"),
        lines_with_apex_candidate=("lines_with_apex_candidate", "sum"),
        auto_apex_candidates=("auto_apex_candidates", "sum"),
        acts_without_apex_candidate=(
            "auto_apex_candidates",
            lambda s: int((s == 0).sum()),
        ),
    )
    print(site.to_string())
    print(
        f"TOTAL activities {len(per_act)}, lines {len(lines)}, lines with candidate "
        f"{int((lines['n_apex_candidates'] > 0).sum())}, auto candidates {len(auto)}, manual {len(manual)}"
    )
    no_candidate = lines[lines["n_apex_candidates"] == 0].copy()
    no_candidate["kind"] = no_candidate["reason"].fillna("").str.split(":").str[0]
    print(
        "no-candidate lines by reason:", no_candidate["kind"].value_counts().to_dict()
    )
    acts0 = per_act[per_act["auto_apex_candidates"] == 0]
    print("activities without any auto candidate:", ", ".join(acts0.index))
    print("zero rules:", lines["zero_rule"].value_counts(dropna=False).to_dict())
    print("manual_v0:", manual["note"].str.split(";").str[0].value_counts().to_dict())
    print(
        "candidates per line (lines with candidates) quantiles:",
        lines.loc[lines["n_apex_candidates"] > 0, "n_apex_candidates"]
        .quantile([0, 0.25, 0.5, 0.75, 1])
        .tolist(),
    )
    print(
        "tau_ns quantiles:",
        auto["tau_ns"].quantile([0, 0.05, 0.5, 0.95, 1]).round(2).tolist(),
    )


if __name__ == "__main__":
    main()
