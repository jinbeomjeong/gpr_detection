"""Aggregate the per-fold evaluation metrics of a 5-fold experiment (M1-10).

Plan: docs/experiments/m110_baseline_plan.md section 5-2. Reads
``<run>/eval/<weights>_<unit>/metrics.json`` (written by
``scripts/eval_detector.py``) and ``<run>/gpr_train_settings.yaml`` of each run
and writes ``<out>.json`` and ``<out>.csv``: one row per fold (fold, seed,
images, instances, metrics) and the unweighted mean, sample standard deviation
(ddof=1), minimum and maximum over the folds. Metric values stay fractions
(0-1) as in metrics.json.

Refuses (exit 1) when the runs are not one each of folds 0-4, when their
``ultralytics`` training blocks differ in anything but ``name`` and ``data``,
or when their evaluations used different settings or class names.

Example:
    conda run -n gpr-dev python -I scripts/aggregate_folds.py \\
        --runs runs/detect/yolo11m_aihubv1_20261030_fold0 ... _fold4 \\
        --weights last --unit crop --out analysis/detection/m110_fold_summary
"""

import argparse
import csv
import json
import logging
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gpr_detection.run_info import git_state, local_timestamp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE_NAME = "gpr_train_settings.yaml"
EVAL_DIR_NAME = "eval"
EXPECTED_FOLDS = ("0", "1", "2", "3", "4")
PER_RUN_ULTRALYTICS_KEYS = frozenset({"name", "data"})
"""Training keys that legitimately differ between the folds of one experiment."""
STATISTICS = ("mean", "std", "min", "max")
OVERALL_METRICS = ("map50", "map50_95", "pipe_ap50_mean")

EXIT_OK = 0
EXIT_INCONSISTENT = 1

LOGGER = logging.getLogger("aggregate_folds")


class InconsistentFoldsError(Exception):
    """The runs do not form one comparable 5-fold experiment (exit code 1)."""


@dataclass(frozen=True)
class FoldRun:
    """Training settings and evaluation metrics of one fold."""

    run_dir: Path
    train_settings: dict[str, Any]
    metrics: dict[str, Any]

    @property
    def fold(self) -> str:
        return str(self.train_settings["fold"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", default="last", choices=("last", "best"))
    parser.add_argument("--unit", default="crop", choices=("crop", "image"))
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output path without suffix; .json and .csv are written",
    )
    return parser


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_fold_run(run_dir: Path, weights: str, unit: str) -> FoldRun:
    settings_path = run_dir / SETTINGS_FILE_NAME
    metrics_path = run_dir / EVAL_DIR_NAME / f"{weights}_{unit}" / "metrics.json"
    for required in (settings_path, metrics_path):
        if not required.is_file():
            raise InconsistentFoldsError(f"missing {required}")
    return FoldRun(
        run_dir=run_dir,
        train_settings=yaml.safe_load(settings_path.read_text(encoding="utf-8")),
        metrics=json.loads(metrics_path.read_text(encoding="utf-8")),
    )


def _check_fold_set(runs: Sequence[FoldRun]) -> None:
    folds = sorted(run.fold for run in runs)
    if folds != sorted(EXPECTED_FOLDS):
        raise InconsistentFoldsError(
            f"need folds {', '.join(EXPECTED_FOLDS)} exactly once each, got {folds}"
        )
    for run in runs:
        if str(run.metrics["fold"]) != run.fold:
            raise InconsistentFoldsError(
                f"{run.run_dir}: trained with fold {run.fold} but evaluated on "
                f"fold {run.metrics['fold']}"
            )


def _differing_keys(first: Mapping[str, Any], other: Mapping[str, Any]) -> list[str]:
    keys = set(first) | set(other)
    return sorted(key for key in keys if first.get(key) != other.get(key))


def _comparable_eval(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluation fields that must match across folds (device may differ).

    Every ``settings`` key counts, including ``conf`` and ``confusion_conf``.
    """
    settings = {
        key: value for key, value in metrics["settings"].items() if key != "device"
    }
    return {
        **{f"settings.{key}": value for key, value in settings.items()},
        "class_names": metrics["class_names"],
        "unit": metrics["unit"],
        "weights.name": metrics["weights"]["name"],
    }


def _check_same_settings(runs: Sequence[FoldRun]) -> None:
    reference = runs[0]
    for run in runs[1:]:
        training_diff = [
            key
            for key in _differing_keys(
                reference.train_settings["ultralytics"],
                run.train_settings["ultralytics"],
            )
            if key not in PER_RUN_ULTRALYTICS_KEYS
        ]
        if training_diff:
            raise InconsistentFoldsError(
                f"ultralytics settings of {run.run_dir} differ from "
                f"{reference.run_dir} in {training_diff}"
            )
        eval_diff = _differing_keys(
            _comparable_eval(reference.metrics), _comparable_eval(run.metrics)
        )
        if eval_diff:
            raise InconsistentFoldsError(
                f"evaluation of {run.run_dir} differs from {reference.run_dir} "
                f"in {eval_diff}"
            )


def _fold_row(run: FoldRun) -> dict[str, Any]:
    metrics = run.metrics
    per_class = metrics["per_class"]
    row: dict[str, Any] = {
        "fold": run.fold,
        "seed": run.train_settings["ultralytics"].get("seed"),
        "run_dir": str(run.run_dir),
        "images": metrics["eval_images"]["count"],
        "instances": sum(item["instances"] for item in per_class),
    }
    for item in per_class:
        row[f"instances_{item['name']}"] = item["instances"]
    row["map50"] = metrics["map50"]
    row["map50_95"] = metrics["map50_95"]
    row["pipe_ap50_mean"] = metrics["pipe_ap50_mean"]["value"]
    for item in per_class:
        row[f"ap50_{item['name']}"] = item["ap50"]
        row[f"ap50_95_{item['name']}"] = item["ap50_95"]
    return row


def _metric_columns(class_names: Sequence[str]) -> list[str]:
    columns = list(OVERALL_METRICS)
    for name in class_names:
        columns += [f"ap50_{name}", f"ap50_95_{name}"]
    return columns


def _summarise(values: Sequence[float | None]) -> dict[str, float | int | None]:
    """Mean, sample std (ddof=1), min, max over the folds that have a value."""
    present = [value for value in values if value is not None]
    if not present:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(present),
        "mean": statistics.fmean(present),
        "std": statistics.stdev(present) if len(present) > 1 else None,
        "min": min(present),
        "max": max(present),
    }


def _write_csv(
    path: Path,
    fold_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Mapping[str, Any]],
) -> None:
    columns = list(fold_rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(fold_rows)
        for statistic in STATISTICS:
            row = {"fold": statistic}
            for column, stats in summary.items():
                row[column] = stats[statistic]
            writer.writerow(row)


def aggregate(runs: Sequence[FoldRun], weights: str, unit: str) -> dict[str, Any]:
    """Check the runs and build the summary document.

    Raises:
        InconsistentFoldsError: If the runs are not one comparable 5-fold set.
    """
    _check_fold_set(runs)
    _check_same_settings(runs)
    ordered = sorted(runs, key=lambda run: run.fold)
    fold_rows = [_fold_row(run) for run in ordered]
    metric_columns = _metric_columns(ordered[0].metrics["class_names"])
    summary = {
        column: _summarise([row[column] for row in fold_rows])
        for column in metric_columns
    }
    return {
        "created_at": local_timestamp(),
        "weights": weights,
        "unit": unit,
        "values_are_fractions": True,
        "std": "sample standard deviation over folds (ddof=1), folds weighted equally",
        "eval_settings": ordered[0].metrics["settings"],
        "ultralytics": {
            key: value
            for key, value in ordered[0].train_settings["ultralytics"].items()
            if key not in PER_RUN_ULTRALYTICS_KEYS
        },
        "folds": fold_rows,
        "summary": summary,
        "git": git_state(PROJECT_ROOT),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Aggregate the folds and write ``<out>.json`` and ``<out>.csv``.

    Returns:
        Process exit code: 0 written, 1 runs missing or inconsistent.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        runs = [
            _read_fold_run(_resolve_project_path(run_dir), args.weights, args.unit)
            for run_dir in args.runs
        ]
        document = aggregate(runs, args.weights, args.unit)
    except InconsistentFoldsError as error:
        LOGGER.error("%s", error)
        return EXIT_INCONSISTENT

    out = _resolve_project_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_name(f"{out.name}.json")
    csv_path = out.with_name(f"{out.name}.csv")
    json_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(csv_path, document["folds"], document["summary"])
    mean_map50 = document["summary"]["map50"]["mean"]
    LOGGER.info("mAP50 fold mean %s -> %s, %s", mean_map50, json_path, csv_path)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
