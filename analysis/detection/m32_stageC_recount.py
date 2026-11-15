"""Read-only recount of the M3-2 stage C R1 verdict from raw outputs (judge cross-check).

Independent of m32_recipe_judge.py: recomputes C-1 point values (unstratified;
pi = 0 in F2'), per-fold deltas, per-dataset deltas, G-in, C-6 point, C-7 M4
counts and C-8 apex sums, plus F2' position counts for the disclosures.
Writes JSON to the only argument; stdout carries a status line only.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "runs" / "eval" / "m32_recipe_v2"
RECIPES = ("r0", "r5", "r6")
CELLS = {("4tu", "A0"): 0.5, ("gz_ids", "A0"): 0.5}
SNRS = ("0.0", "6.0", "12.0", "18.0")


def read(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def cell_recall(rows, keep):
    acc = defaultdict(lambda: [0, 0])
    for row in rows:
        if row["k"] != "1.3" or not keep(row):
            continue
        key = (row["dataset"], row["model_group"], row["snr_db"], row["polarity"])
        acc[key][0] += row["hit"] == "True"
        acc[key][1] += 1
    return acc


def weighted(acc, cells=CELLS):
    total = 0.0
    for (dataset, group), weight in cells.items():
        values = [
            acc[(dataset, group, snr, pol)][0] / acc[(dataset, group, snr, pol)][1]
            for snr in SNRS
            for pol in ("1", "-1")
        ]
        total += weight * sum(values) / len(values)
    return total


def main(out_path: str) -> int:
    data = {
        (r, k): read(ROOT / "injected" / r / f"fold{k}" / "inj_matches.csv")
        for r in RECIPES
        for k in range(5)
    }
    main_keep = lambda row: row["main_cell"] == "True" and row["model_group"] == "A0"
    in_keep = lambda row: main_keep(row) and row["in_distribution"] == "True"
    result = {"c1": {}, "gin": {}, "c4": {}, "c6": {}, "c7": {}, "c8": {}}
    for keep_name, keep, target in (("c1", main_keep, "c1"), ("gin", in_keep, "gin")):
        r0 = [weighted(cell_recall(data[("r0", k)], keep)) for k in range(5)]
        for r in ("r5", "r6"):
            per_fold = [weighted(cell_recall(data[(r, k)], keep)) - r0[k] for k in range(5)]
            result[target][r] = {
                "mean_pp": round(100 * sum(per_fold) / 5, 3),
                "per_fold_pp": [round(100 * v, 3) for v in per_fold],
            }
    for r in ("r5", "r6"):
        by = {}
        for dataset, group in CELLS:
            one = {(dataset, group): 1.0}
            diffs = [
                weighted(cell_recall(data[(r, k)], main_keep), one)
                - weighted(cell_recall(data[("r0", k)], main_keep), one)
                for k in range(5)
            ]
            by[dataset] = round(100 * sum(diffs) / 5, 3)
        result["c4"][r] = by
    bottom = lambda row: row["k"] == "1.3" and row["bottom_stratum"] == "True"
    def pooled(rows):
        sel = [row for row in rows if bottom(row)]
        return sum(row["hit"] == "True" for row in sel) / len(sel), len(sel)
    for r in ("r5", "r6"):
        diffs = [pooled(data[(r, k)])[0] - pooled(data[("r0", k)])[0] for k in range(5)]
        result["c6"][r] = {"point_pp": round(100 * sum(diffs) / 5, 3), "n_fold0": pooled(data[(r, 0)])[1]}
    for r in RECIPES:
        per_fold, m4c, lines = {}, 0, set()
        for k in range(5):
            n = 0
            for row in read(ROOT / "uninjected" / r / f"fold{k}" / "boxes.csv"):
                if row["orientation"] != "original":
                    continue
                if row["m4"] in ("M4a", "M4b", "M4d"):
                    n += 1
                    lines.add(row["line_id"])
                m4c += row["m4c"] == "True"
            per_fold[k] = n
        result["c7"][r] = {"per_fold": per_fold, "total": sum(per_fold.values()), "lines": len(lines), "m4c_total": m4c}
        hits = defaultdict(int)
        fold_hits = defaultdict(int)
        for k in range(5):
            for row in read(ROOT / "uninjected" / r / f"fold{k}" / "apex_hits.csv"):
                h = row["hit"] == "True"
                hits[row["blind_id"]] += h
                fold_hits[k] += h
        result["c8"][r] = {"sum": sum(hits.values()), "any": sum(v > 0 for v in hits.values()), "per_fold": dict(fold_hits)}
    positions = read(ROOT / "plan" / "positions.csv")
    counts = defaultdict(int)
    bottom_lines = set()
    for row in positions:
        cell = f"{row['dataset']}/{row['model_group']}"
        counts["total"] += 1
        counts[f"cell:{cell}"] += 1
        if row["main_cell"] == "True":
            counts[f"main:{cell}"] += 1
            if row["in_distribution"] == "True":
                counts[f"main_in:{cell}"] += 1
        else:
            counts[f"excluded:{cell}"] += 1
            counts["excluded_box_clipped" if row["box_clipped"] == "True" else "excluded_wing_only"] += 1
        if row["bottom_stratum"] == "True":
            counts[f"bottom:{cell}"] += 1
            bottom_lines.add(row["line_id"])
        counts[f"height:{row['height_class']}"] += 1
        counts["lines"] += 0
    counts["bottom_lines"] = len(bottom_lines)
    counts["lines_total"] = len({row["line_id"] for row in positions})
    result["positions"] = dict(counts)
    Path(out_path).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({"status": "ok", "out": out_path}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
