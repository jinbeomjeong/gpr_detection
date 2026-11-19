"""Read-only cross-check of runs/eval/m33/r3_change_report for P4/P5 judgment §10.

Recounts the change flags from realizations.csv, compares the start/end R3
tree hash lists with the tracked hash lists, and lists per-line counts of
period changes > 5 %. Writes nothing; prints one JSON object.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "runs/eval/m33/r3_change_report"
RESULT_LIST = ROOT / "analysis/detection/m33_r3_result_hashes.csv"
UPSTREAM_LIST = ROOT / "docs/experiments/m33_r3_upstream_hashes.csv"


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> None:
    rows = read_rows(REPORT / "realizations.csv")
    out: dict = {"n_rows": len(rows)}
    keys = ["zero_changed", "zero_changed_exact", "period_changed", "period_changed_exact",
            "zero_label_changed", "zero_failed_changed", "zero_rule_changed", "zero_events_changed",
            "period_failed"]
    out["counts"] = {k: sum(truth(r[k]) for r in rows) for k in keys}
    out["stored_match"] = {
        "zero": sum(truth(r["zero_changed_exact"]) == truth(r["stored_zero_changed"]) for r in rows),
        "period": sum(truth(r["period_changed_exact"]) == truth(r["stored_period_changed"]) for r in rows),
        "infer": sum(r["infer_i0"] == r["stored_infer_i0"] and r["infer_i1"] == r["stored_infer_i1"]
                     for r in rows),
    }
    # Independent recomputation of the new definitions from raw columns.
    zero_new = 0
    period_new = 0
    period_ratio_form = 0
    max_dk_over_p = 0.0
    for r in rows:
        p_bg = float(r["period_bg_samples"])
        dk = float(r["zero_dk_samples"])
        label = r["zero_label_bg"] != r["zero_label_inj"]
        zero_new += label or abs(dk) > 0.1 * p_bg
        max_dk_over_p = max(max_dk_over_p, abs(dk) / p_bg)
        p_inj = float(r["period_inj_samples"])
        period_new += abs(p_inj - p_bg) > 0.05 * p_bg
        period_ratio_form += abs(p_inj / p_bg - 1.0) > 0.05
    out["recomputed"] = {"zero_changed": zero_new, "period_changed": period_new,
                         "period_changed_ratio_form": period_ratio_form,
                         "max_abs_dk_over_p": max_dk_over_p}
    by_line = Counter(r["line_id"] for r in rows if truth(r["period_changed"]))
    ds = {r["line_id"]: (r["dataset"], r["ids_freq_stratum"]) for r in rows}
    out["period_changed_lines"] = {
        "n_lines": len(by_line),
        "by_dataset_lines": dict(Counter(ds[k][0] for k in by_line)),
        "top": [[k, ds[k][0], ds[k][1], v] for k, v in by_line.most_common(12)],
    }
    ids_lines = {k: v for k, v in by_line.items() if ds[k][0] != "4tu"}
    out["ids_period_changed"] = {
        "n_lines": len(ids_lines),
        "n_lines_total": len({r["line_id"] for r in rows if r["dataset"] != "4tu"}),
        "sorted_counts": sorted(ids_lines.values(), reverse=True),
    }
    # Tree hash lists.
    start = read_rows(REPORT / "r3_tree_sha256_start.csv")
    end = read_rows(REPORT / "r3_tree_sha256_end.csv")
    expected = {r["path"]: r["sha256"] for r in read_rows(RESULT_LIST)}
    expected.update({r["path"]: r["sha256"] for r in read_rows(UPSTREAM_LIST)
                     if r["root"] == "runs/eval/m33/r3/uninjected_ids"})
    s = {r["path"]: r["sha256"] for r in start}
    e = {r["path"]: r["sha256"] for r in end}
    out["tree"] = {
        "n_start": len(s), "n_end": len(e), "n_expected": len(expected),
        "start_eq_end": s == e,
        "start_eq_expected": s == expected,
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
