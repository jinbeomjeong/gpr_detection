"""Projection of 4TU re-evaluation estimate counts and the pass-B ceiling.

Used for the target proposal of the 4TU re-evaluation pre-registration
(docs/processing/preregistration_4tu_reeval_v1.md); the output is a projection,
not a result.

No ground-truth depth is printed or used: only the number of ground-truth
items per activity (exclude_x set) enters the ceiling sum(min(n_est, n_gt)).
Candidate depth proxy = v_meta * tau / 2 with v_meta = c / sqrt(eps_r_used).
QC retention is modelled by independent Bernoulli thinning with the review
sample rates (m29c_track_diagnosis.md 12-5); this is an assumption.

Run:
  conda run -n gpr-dev python -I analysis/processing/prereg_4tu_ceiling_projection.py
      [--out docs/processing/preregistration_4tu_reeval_v1_projection.json]
Seed: numpy default_rng(20261108), one stream over the scenarios in order.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import eval_depth_4tu as eval4tu  # noqa: E402

from gpr_detection.io.paths import load_dataset_paths  # noqa: E402

PROCESSING_DOCS = PROJECT_ROOT / "docs" / "processing"
CANDIDATES_CSV = PROCESSING_DOCS / "fourtu_apex_candidates_v1.csv"
LINES_CSV = PROCESSING_DOCS / "fourtu_lines_v1.csv"

C_M_PER_NS = 0.299792458
SEED = 20261108
DRAWS = 200
CEILING_DECIMALS = 4

# Review-sample retention (status-independent counts, 12-5 table, v1.9).
RETENTION = {
    "main_BDE": {"auto_v1": 28 / 130, "manual_v0": 21 / 39},
    "bd_BD": {"auto_v1": 91 / 130, "manual_v0": 35 / 39},
    "noqc": {"auto_v1": 1.0, "manual_v0": 1.0},
}
# Without QC thinning every draw is identical, so one draw suffices.
DETERMINISTIC_SCENARIOS = {"noqc"}

METADATA = {
    "purpose": (
        "Pass-B ceiling projection for the 4TU re-evaluation pre-registration "
        "(target proposal only, not a result)"
    ),
    "author": "gpr-processing-expert",
    "script": "analysis/processing/prereg_4tu_ceiling_projection.py",
    "inputs": [
        "docs/processing/fourtu_apex_candidates_v1.csv (tau_ns, source)",
        "docs/processing/fourtu_lines_v1.csv (eps_r_used)",
        "docs/datasets/fourtu_duplicate_radargrams.csv "
        "(within-activity copies dropped)",
        "4TU ground truth: number of exclude_x items per activity only "
        "(no depth value used or printed)",
    ],
}
ASSUMPTIONS = [
    "QC retention independent per candidate "
    "(real retention is correlated within a pipe and site)",
    "apex re-selection within +-0.5 period ignored (depth proxy from candidate tau)",
    "review-sample rates applied to the whole population without site weighting",
]


def _model_description(n_gt_total: int) -> dict[str, str]:
    return {
        "depth_proxy": f"c / sqrt(eps_r_used) * tau_ns / 2, c = {C_M_PER_NS} m/ns",
        "qc_retention": (
            "independent Bernoulli thinning with the review-sample rates "
            "(m29c_track_diagnosis.md 12-5, v1.9): main B+D+E auto 28/130, "
            "manual 21/39; bd B+D auto 91/130, manual 35/39; noqc 1"
        ),
        "clustering": (
            "eval_depth_4tu.cluster_depths complete linkage, width "
            f"{eval4tu.CLUSTER_WIDTH_M:.2f} m, support filter >= "
            f"{eval4tu.MIN_SUPPORT_LINES} lines (eval_depth_4tu.supported)"
        ),
        "ceiling": f"sum over gt activities of min(n_est, n_gt) / {n_gt_total}",
    }


def _load_gt_counts(lines_per_activity: dict[str, int]) -> dict[str, int]:
    """Number of exclude_x ground-truth items per evaluated activity.

    The loader builds full items, but only their count leaves this function.
    """
    gt_sets = eval4tu.load_ground_truth_sets(load_dataset_paths().fourtu)
    return {
        activity: len(items)
        for activity, items in gt_sets["exclude_x"].items()
        if activity in lines_per_activity
    }


def _candidate_groups(
    lines: pd.DataFrame, copies: set[tuple[str, str]]
) -> tuple[dict[str, np.ndarray], int]:
    """Per-activity ``[depth_proxy_m, path, source]`` rows and the candidate count."""
    candidates = pd.read_csv(CANDIDATES_CSV, dtype={"activity_id": str, "path": str})
    eps_by_line = {
        (row.activity_id, row.path): float(row.eps_r_used)
        for row in lines.itertuples(index=False)
    }
    is_original = np.array(
        [
            (activity, path) not in copies
            for activity, path in zip(
                candidates["activity_id"], candidates["path"], strict=True
            )
        ]
    )
    has_tau = candidates["tau_ns"].notna().to_numpy()
    used = candidates[is_original & has_tau].copy()
    used["eps"] = [
        eps_by_line[(activity, path)]
        for activity, path in zip(used["activity_id"], used["path"], strict=True)
    ]
    used["depth_proxy_m"] = C_M_PER_NS / np.sqrt(used["eps"]) * used["tau_ns"] / 2.0
    groups = {
        activity: group[["depth_proxy_m", "path", "source"]].to_numpy()
        for activity, group in used.groupby("activity_id")
    }
    return groups, len(used)


def _project_once(
    rates: dict[str, float],
    groups: dict[str, np.ndarray],
    lines_per_activity: dict[str, int],
    n_gt: dict[str, int],
    rng: np.random.Generator,
) -> dict[str, float]:
    """One QC-thinning draw: cluster the kept candidates and count estimates."""
    n_est: dict[str, int] = {}
    n_input = 0
    for activity in lines_per_activity:
        rows = groups.get(activity)
        if rows is None:
            n_est[activity] = 0
            continue
        keep_probability = np.array([rates[source] for source in rows[:, 2]])
        is_kept = rng.random(len(rows)) < keep_probability
        depths = [float(depth) for depth in rows[is_kept, 0]]
        paths = [str(path) for path in rows[is_kept, 1]]
        n_input += len(depths)
        clusters = eval4tu.cluster_depths(
            depths, paths, eval4tu.CLUSTER_WIDTH_M, "complete"
        )
        supported = eval4tu.supported(
            clusters, lines_per_activity[activity], eval4tu.MIN_SUPPORT_LINES
        )
        n_est[activity] = len(supported)
    pairs = sum(min(n_est[activity], count) for activity, count in n_gt.items())
    return {
        "n_input": n_input,
        "n_est_all": sum(n_est.values()),
        "n_est_gt_acts": sum(n_est[activity] for activity in n_gt),
        "pairs_max": pairs,
        "ceiling_pass_b": pairs / sum(n_gt.values()),
        "gt_acts_without_estimate": sum(n_est[activity] == 0 for activity in n_gt),
    }


def _summarise_draws(draws: list[dict[str, float]]) -> dict[str, object]:
    """Mean and 5/50/95 percentiles per key, or the plain values for one draw."""
    if len(draws) == 1:
        (only,) = draws
        return {
            key: round(value, CEILING_DECIMALS) if isinstance(value, float) else value
            for key, value in only.items()
        }
    summary: dict[str, object] = {}
    for key in draws[0]:
        values = np.array([draw[key] for draw in draws], dtype=float)
        summary[key] = {
            "mean": round(float(values.mean()), CEILING_DECIMALS),
            "p05": round(float(np.percentile(values, 5)), CEILING_DECIMALS),
            "p50": round(float(np.percentile(values, 50)), CEILING_DECIMALS),
            "p95": round(float(np.percentile(values, 95)), CEILING_DECIMALS),
        }
    return summary


def _format_json(projection: dict[str, object]) -> str:
    """Indented JSON with each mean/percentile summary kept on one line."""
    text = json.dumps(projection, ensure_ascii=True, indent=1)

    def collapse(match: re.Match[str]) -> str:
        return "{" + re.sub(r",\n\s+", ", ", match.group(1)) + "}"

    return re.sub(r"\{\n\s+(\"mean\": [^{}]*?)\n\s+\}", collapse, text) + "\n"


def build_projection() -> dict[str, object]:
    """Run all retention scenarios and return the projection document."""
    lines = pd.read_csv(LINES_CSV, dtype={"activity_id": str, "path": str})
    duplicates = eval4tu.load_duplicate_table(eval4tu.DEFAULT_DUPLICATES)
    lines_per_activity = eval4tu.unique_line_counts(lines, duplicates)
    n_gt = _load_gt_counts(lines_per_activity)
    n_gt = dict(sorted(n_gt.items()))
    groups, n_candidates = _candidate_groups(lines, duplicates.within_copies())
    n_gt_total = sum(n_gt.values())

    rng = np.random.default_rng(SEED)
    scenarios: dict[str, object] = {}
    for scenario, rates in RETENTION.items():
        n_draws = 1 if scenario in DETERMINISTIC_SCENARIOS else DRAWS
        draws = [
            _project_once(rates, groups, lines_per_activity, n_gt, rng)
            for _ in range(n_draws)
        ]
        scenarios[scenario] = _summarise_draws(draws)

    return {
        "purpose": METADATA["purpose"],
        "created": datetime.now().astimezone().isoformat(timespec="minutes"),
        "author": METADATA["author"],
        "script": METADATA["script"],
        "inputs": METADATA["inputs"],
        "model": _model_description(n_gt_total),
        "assumptions": ASSUMPTIONS,
        "seed": SEED,
        "draws": DRAWS,
        "n_candidates_used": n_candidates,
        "n_gt_total_exclude_x": n_gt_total,
        "n_gt_activities": len(n_gt),
        "activities": len(lines_per_activity),
        "scenarios": scenarios,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write the JSON here as well")
    args = parser.parse_args()
    text = _format_json(build_projection())
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
