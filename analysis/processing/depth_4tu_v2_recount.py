"""Recount the M2-9 4TU re-evaluation (depth_4tu_v2) outputs for the manual judgment.

Read-only: reads the four ``docs/processing/depth_4tu_v2_*.csv`` files written by
the re-evaluation run and writes one JSON summary. No randomness, no ground-truth
file is read beyond what the csv files already hold.

The recount filters ``depth_4tu_v2_matches.csv`` by ``gt_set``, ``assumption``,
``zero_variant`` and ``variant`` and counts ``within_30cm == True``; the summary
csv supplies the baseline numbers it is checked against. Identifier columns are
read as strings: read as floats, activity ids such as "05.1" and "05.10" collapse
into one (the first recount of 2026-11-11 had that defect).

Run (single process, a few seconds):
    conda run -n gpr-dev python -I analysis/processing/depth_4tu_v2_recount.py \
        --out <folder>/recount_v2.json

Without ``--out`` the JSON goes to stdout.
"""

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "docs" / "processing"
FILE_PREFIX = "depth_4tu_v2"

# Identifier columns must stay strings ("05.1" vs "05.10" collapse as float).
ID_DTYPES = {
    "activity_id": str,
    "path": str,
    "line_id": str,
    "duplicate_of": str,
    "scope": str,
    "gt_code": str,
}

# Activities the auditor listed as having ground truth but no estimate (main combo).
AUDIT_ACTIVITIES = {
    "010.4",
    "010.5",
    "010.7",
    "011.1",
    "011.11",
    "011.13",
    "011.15",
    "011.16",
    "011.6",
    "011.7",
    "012.4",
    "013.1",
    "013.2",
    "013.4",
    "04.2",
    "04.6",
    "04.8",
    "05.9",
}

VARIANTS = ["main", "bd", "noqc", "a", "b"]
GT_SETS = ["exclude_x", "include_x"]
ASSUMPTIONS = ["top", "center"]
ZERO_VARIANTS = ["detected", "est", "act"]

# Baseline trial counts of the re-evaluation run (main 1000, auxiliary 200).
MAIN_BASELINE_TRIALS = 1000
AUX_BASELINE_TRIALS = 200
# U-3 rule: baseline p <= 0.049 and at least 25 % of GT items within 30 cm.
U3_MAX_BASELINE_P = 0.049
U3_MIN_PASS_FRACTION = 0.25
FLOAT_SLACK = 1e-9

DEPTH_PASS_LIMIT_M = 0.3
# S14: apex index minus search start k0, "near the top" window in samples.
S14_NEAR_MIN, S14_NEAR_MAX = 0, 2
TOP_FLAG_COUNT = 15
TOP_FAILED_FLAG_COUNT = 10
DUP_HEAD_ROWS = 3


def _read_outputs(input_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in ("summary", "matches", "candidates", "dup_sensitivity"):
        path = input_dir / f"{FILE_PREFIX}_{name}.csv"
        tables[name] = pd.read_csv(path, dtype=ID_DTYPES)
    return tables


def _is_true(column: pd.Series) -> pd.Series:
    """csv booleans may arrive as bool or text; compare their lower-case text."""
    return column.astype(str).str.lower() == "true"


def _activity_counts(matches: pd.DataFrame) -> dict[str, object]:
    # GT activities are the union of activity ids in any exclude_x match row
    # (any variant, zero, assumption); variant b (no clustering, 823 estimates)
    # should cover every GT activity with any candidate. The auditor list is
    # checked against this.
    excluded = matches[matches.gt_set == "exclude_x"]
    is_main_combo = (
        (excluded.assumption == "top")
        & (excluded.zero_variant == "detected")
        & (excluded.variant == "main")
    )
    main_activities = set(excluded[is_main_combo].activity_id)
    any_activities = set(excluded.activity_id)
    per_variant = {
        variant: len(set(excluded[excluded.variant == variant].activity_id))
        for variant in sorted(excluded.variant.unique())
    }
    return {
        "main_acts_with_pairs": len(main_activities),
        "acts_in_any_exclude_x_match": len(any_activities),
        "acts_per_variant": per_variant,
        "zero_est_from_any_union": sorted(any_activities - main_activities),
        "audit_list_n": len(AUDIT_ACTIVITIES),
        "audit_overlap_main": sorted(AUDIT_ACTIVITIES & main_activities),
        "audit_minus_union": sorted(AUDIT_ACTIVITIES - any_activities),
        "union_minus_main_equals_audit": (any_activities - main_activities)
        == AUDIT_ACTIVITIES,
        "float_collapse_check": {
            "n_unique_str": int(matches.activity_id.nunique()),
            "n_unique_float": int(pd.to_numeric(matches.activity_id).nunique()),
        },
    }


def _unique_values(table: pd.DataFrame, columns: list[str]) -> dict[str, list[str]]:
    return {
        column: sorted(map(str, table[column].dropna().unique())) for column in columns
    }


def _summary_rows(
    summary: pd.DataFrame,
    scope: str = "all",
    gt_set: str = "exclude_x",
    assumption: str = "top",
    zero_variant: str = "detected",
    variant: str = "main",
) -> pd.DataFrame:
    return summary[
        (summary.scope == scope)
        & (summary.gt_set == gt_set)
        & (summary.assumption == assumption)
        & (summary.zero_variant == zero_variant)
        & (summary.variant == variant)
    ]


def _count_matches_within_limit(
    matches: pd.DataFrame, gt_set: str, assumption: str, zero_variant: str, variant: str
) -> tuple[int, int, pd.DataFrame]:
    """Return (pairs within 30 cm, pairs, the selected match rows)."""
    selected = matches[
        (matches.gt_set == gt_set)
        & (matches.assumption == assumption)
        & (matches.zero_variant == zero_variant)
        & (matches.variant == variant)
    ]
    n_pass = int(_is_true(selected.within_30cm).sum())
    return n_pass, len(selected), selected


def _combo_record(
    row: pd.Series, n_pass: int, n_match_rows: int, labels: dict[str, str]
) -> dict[str, object]:
    n_gt = int(row.n_gt)
    trials = MAIN_BASELINE_TRIALS if labels["v"] == "main" else AUX_BASELINE_TRIALS
    baseline_exceed_count = row.baseline_p * trials
    is_u3_pass = (
        baseline_exceed_count <= U3_MAX_BASELINE_P * trials + FLOAT_SLACK
    ) and n_pass >= math.ceil(U3_MIN_PASS_FRACTION * n_gt)
    return {
        **labels,
        "n_gt": n_gt,
        "n_est": int(row.n_est),
        "n_pairs": int(row.n_pairs),
        "match_rows": n_match_rows,
        "n_pass_matches": n_pass,
        "n_pass_from_pass_b": round(row.pass_b * n_gt, 3),
        "pass_b": row.pass_b,
        "pass_a": row.pass_a,
        "mae": row.mae_m,
        "bias": row.bias_m,
        "med": row.med_abs_m,
        "p90": row.p90_abs_m,
        "missed": int(row.missed),
        "false_est": int(row.false_est),
        "bl_b_mean": row.baseline_pass_b_mean,
        "bl_b_p025": row.baseline_pass_b_p025,
        "bl_b_p975": row.baseline_pass_b_p975,
        "bl_a_mean": row.baseline_pass_a_mean,
        "bl_a_p025": row.baseline_pass_a_p025,
        "bl_a_p975": row.baseline_pass_a_p975,
        "baseline_p": row.baseline_p,
        "k": baseline_exceed_count,
        "trials": trials,
        "bmax": round(int(row.n_pairs) / n_gt, 4),
        "bl_a_from_b": round(row.baseline_pass_b_mean * n_gt / int(row.n_pairs), 4),
        "glob_b_mean": row.baseline_global_pass_b_mean,
        "glob_b_p025": row.baseline_global_pass_b_p025,
        "glob_b_p975": row.baseline_global_pass_b_p975,
        "glob_p": row.baseline_global_p,
        "gate_pairs": row.gate_pairs,
        "gate_recall": row.gate_recall,
        "gate_precision": row.gate_precision,
        "gate_f1": row.gate_f1,
        "gate_p": row.baseline_gate_p,
        "u3_rule_pass": bool(is_u3_pass),
    }


def _combos(summary: pd.DataFrame, matches: pd.DataFrame) -> list[dict[str, object]]:
    """One record per (variant, gt set, assumption, zero variant) of the summary."""
    combos: list[dict[str, object]] = []
    # product() varies the last list fastest, the order of the original nested loops.
    for variant, gt_set, assumption, zero_variant in itertools.product(
        VARIANTS, GT_SETS, ASSUMPTIONS, ZERO_VARIANTS
    ):
        labels = {"v": variant, "gt": gt_set, "a": assumption, "z": zero_variant}
        rows = _summary_rows(
            summary,
            gt_set=gt_set,
            assumption=assumption,
            zero_variant=zero_variant,
            variant=variant,
        )
        if len(rows) != 1:
            combos.append({**labels, "rows": len(rows)})
            continue
        n_pass, n_rows, _ = _count_matches_within_limit(
            matches, gt_set, assumption, zero_variant, variant
        )
        combos.append(_combo_record(rows.iloc[0], n_pass, n_rows, labels))
    return combos


def _main_detail(
    n_pass: int, n_rows: int, main_matches: pd.DataFrame
) -> dict[str, object]:
    errors_m = main_matches.error_m
    quantiles = errors_m.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n_pass": n_pass,
        "pairs": n_rows,
        "fail": int(n_rows - n_pass),
        "fail_shallow_e_lt_-0.3": int((errors_m < -DEPTH_PASS_LIMIT_M).sum()),
        "fail_deep_e_gt_0.3": int((errors_m > DEPTH_PASS_LIMIT_M).sum()),
        "e_neg_frac": round(float((errors_m < 0).mean()), 4),
        "e_quantiles_5_25_50_75_95": [round(float(value), 4) for value in quantiles],
        "mae_recalc": round(float(errors_m.abs().mean()), 4),
        "bias_recalc": round(float(errors_m.mean()), 4),
        "radius_known_pairs": int(_is_true(main_matches.radius_known).sum()),
        "gt_code_counts": main_matches.gt_code.value_counts().to_dict(),
        "multi_trench_pairs": int(_is_true(main_matches.multi_trench).sum()),
    }


def _per_site_from_matches(main_matches: pd.DataFrame) -> dict[int, dict[str, object]]:
    per_site = main_matches.groupby("site").agg(
        pairs=("error_m", "size"),
        bias=("error_m", "mean"),
        mae=("error_m", lambda errors: errors.abs().mean()),
        npass=("within_30cm", lambda flags: int(_is_true(flags).sum())),
    )
    by_site: dict[int, dict[str, object]] = {}
    # iterrows upcasts each row to float64, so every value takes the float branch.
    for site, site_row in per_site.iterrows():
        by_site[int(site)] = {
            name: (round(float(value), 4) if isinstance(value, float) else int(value))
            for name, value in site_row.items()
        }
    return by_site


def _site_rows(summary: pd.DataFrame) -> list[dict[str, object]]:
    sites = summary[
        (summary.scope.str.startswith("site_"))
        & (summary.gt_set == "exclude_x")
        & (summary.variant == "main")
        & (summary.zero_variant == "detected")
    ]
    columns = [
        "scope",
        "assumption",
        "n_gt",
        "n_est",
        "n_pairs",
        "mae_m",
        "bias_m",
        "pass_b",
        "baseline_pass_b_mean",
        "baseline_p",
    ]
    return sites[columns].to_dict("records")


def _eps_by_site(candidates: pd.DataFrame) -> dict[int, list[float]]:
    """Relative permittivity values used per site (rounded, distinct)."""
    eps = candidates.groupby("site").eps_r_used.agg(
        lambda values: sorted({round(float(value), 3) for value in values.dropna()})
    )
    return {int(site): values for site, values in eps.items()}


def _apex_offset_stats(subset: pd.DataFrame) -> dict[str, object]:
    offsets = subset.apex_minus_search_k0.dropna()
    is_near = (offsets >= S14_NEAR_MIN) & (offsets <= S14_NEAR_MAX)
    quantiles = offsets.quantile([0, 0.05, 0.5, 0.95])
    return {
        "n": len(offsets),
        "n_0_2": int(is_near.sum()),
        "frac_0_2": round(float(is_near.mean()), 4),
        "n_neg": int((offsets < 0).sum()),
        "quantiles_0_5_50_95": [round(float(value), 2) for value in quantiles],
    }


def _flag_counts(candidates: pd.DataFrame, top_n: int) -> dict[str, int]:
    flags = candidates["flags"].fillna("").str.split(" ").explode()
    return flags.value_counts().head(top_n).to_dict()


def _candidate_sections(candidates: pd.DataFrame) -> dict[str, object]:
    fitted = candidates[candidates.status == "ok"]
    sections: dict[str, object] = {
        "S14_all_nonnull": _apex_offset_stats(candidates),
        "S14_ok": _apex_offset_stats(fitted),
    }
    near_top = fitted[
        (fitted.apex_minus_search_k0 >= S14_NEAR_MIN)
        & (fitted.apex_minus_search_k0 <= S14_NEAR_MAX)
    ]
    sections["S14_near_flags"] = _flag_counts(near_top, TOP_FLAG_COUNT)
    sections["S14_near_qc_excluded"] = (
        near_top.qc_excluded.astype(str).value_counts().to_dict()
    )
    sections["status_counts"] = candidates.status.value_counts().to_dict()
    failed = candidates[candidates.status != "ok"]
    sections["fit_failed_flags"] = _flag_counts(failed, TOP_FAILED_FLAG_COUNT)
    sections["zero_rule_counts"] = candidates.zero_rule.value_counts().to_dict()
    return sections


def recount(input_dir: Path) -> dict[str, object]:
    """Recount the depth_4tu_v2 csv outputs in ``input_dir``.

    Args:
        input_dir: Folder holding ``depth_4tu_v2_{summary,matches,candidates,
            dup_sensitivity}.csv``.

    Returns:
        The JSON-ready summary; key order is part of the output format.
    """
    tables = _read_outputs(input_dir)
    summary, matches = tables["summary"], tables["matches"]
    candidates, dup_sensitivity = tables["candidates"], tables["dup_sensitivity"]

    out: dict[str, object] = {
        "activity_counts": _activity_counts(matches),
        "summary_cols_unique": _unique_values(
            summary, ["scope", "gt_set", "variant", "zero_variant", "assumption"]
        ),
        "matches_cols_unique": _unique_values(
            matches, ["gt_set", "variant", "zero_variant", "assumption"]
        ),
        "combos": _combos(summary, matches),
    }
    n_pass, n_rows, main_matches = _count_matches_within_limit(
        matches, "exclude_x", "top", "detected", "main"
    )
    out["main_detail"] = _main_detail(n_pass, n_rows, main_matches)
    out["main_per_site_matches"] = _per_site_from_matches(main_matches)
    out["main_activities_with_pairs"] = int(main_matches.activity_id.nunique())
    out["site_rows"] = _site_rows(summary)
    out["eps_by_site"] = _eps_by_site(candidates)
    out.update(_candidate_sections(candidates))
    out["dup_cols"] = list(dup_sensitivity.columns)
    out["dup_head"] = dup_sensitivity.head(DUP_HEAD_ROWS).to_dict("records")
    return out


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recount the depth_4tu_v2 csv outputs (read-only)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="folder of the depth_4tu_v2_*.csv files (default: docs/processing)",
    )
    parser.add_argument(
        "--out", type=Path, help="JSON file to write (default: print to stdout)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the recount and write the JSON summary."""
    args = _parse_arguments(argv)
    summary = recount(args.input_dir)
    text = json.dumps(summary, ensure_ascii=False, indent=1, default=str)
    if args.out is None:
        sys.stdout.write(text)
    else:
        # LF on every platform; the stdout form follows the console convention.
        args.out.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
