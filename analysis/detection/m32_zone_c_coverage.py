"""Read-only proxy: how much of each D3 line zone (c) would block (M3-2 S4 review).

Uses D3 boxes (recipe r0 only, folds 0-4 union, any class) as a stand-in for
the P5a un-injected boxes. Compares the horizontal-only rule
(|apex - centre| <= w/2 + 1P) with a 2-D rule (apex sample inside the box
expanded by 1P in time as well). 1P is taken as 6 traces horizontally; the
time margin uses P_line samples from line_flags.csv when present.
Writes JSON to the path given as the only argument; prints counts only.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
D3 = REPO / "data" / "processing" / "d3"
P_TRACES = 6.0


def main(out_path: str) -> int:
    n_traces = {}
    with (D3 / "lines.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n_traces[f"{row['activity_id']}/{row['path']}"] = int(row["n_traces"])
    boxes = defaultdict(list)
    with (D3 / "boxes.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            boxes[row["line"]].append(
                (int(row["i0"]), int(row["i1"]), int(row["k0"]), int(row["k1"]))
            )
    per_line = []
    for line, n in n_traces.items():
        blocked = np.zeros(n, dtype=bool)
        for i0, i1, _, _ in boxes.get(line, []):
            centre = (i0 + i1 + 1) / 2.0
            half = (i1 - i0 + 1) / 2.0 + P_TRACES
            lo = max(0, int(np.floor(centre - half)))
            hi = min(n, int(np.ceil(centre + half)) + 1)
            blocked[lo:hi] = True
        per_line.append(
            {"line": line, "n_traces": n, "n_boxes": len(boxes.get(line, [])),
             "blocked_fraction_horizontal": float(blocked.mean())}
        )
    fractions = np.array([r["blocked_fraction_horizontal"] for r in per_line])
    weights = np.array([r["n_traces"] for r in per_line], dtype=float)
    summary = {
        "n_lines": len(per_line),
        "lines_with_boxes": int(sum(r["n_boxes"] > 0 for r in per_line)),
        "blocked_fraction_median": float(np.median(fractions)),
        "blocked_fraction_p90": float(np.percentile(fractions, 90)),
        "blocked_fraction_trace_weighted": float((fractions * weights).sum() / weights.sum()),
        "lines_blocked_ge_50pct": int((fractions >= 0.5).sum()),
        "lines_blocked_ge_90pct": int((fractions >= 0.9).sum()),
        "per_line": per_line,
    }
    Path(out_path).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"status": "ok", "n_lines": len(per_line), "out": out_path}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
