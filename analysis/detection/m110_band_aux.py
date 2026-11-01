"""pipe_band auxiliary metrics of the M1-10 baseline (U-3, 2026-11-01). CPU only.

Reads the per-image predictions that ``export_val_predictions.py`` wrote for
each fold (``runs/analysis/m110_band_aux/fold<k>_<unit>/``) and computes, with
``gpr_detection.detect.band_metrics``:

* horizontal-extent recall and precision of ``pipe_band`` at conf > 0.25
  (length-weighted totals = main value, image means for reference),
* fragment counts (predictions per GT box and per merged GT band),
* band AP50 per session (manifest ``session``) from the same predictions.

Two checks prove the predictions are those of the evaluation path (exit 1 if
either fails): the conf 0.25 confusion matrix rebuilt from ``predictions.csv``
equals ``eval/last_<unit>/confusion_matrix_conf0.25.csv`` cell by cell, and the
per-class AP50 recomputed from all predictions equals ``metrics.json``.

Writes ``analysis/detection/m110_band_aux.json`` and ``.csv`` (aggregates only:
no image names or coordinates, the repository is public). The target rule of
U-3 is fold mean - 1 SD (ddof=1) of the crop-unit values.

Usage: python -I analysis/detection/m110_band_aux.py [--units crop,image]
The last stdout line is a JSON summary. No ultralytics import.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpr_detection.detect.band_metrics import (  # noqa: E402
    DEFAULT_MIN_VERTICAL_FRACTION,
    EMPTY_COVERAGE,
    UNASSIGNED,
    BandCoverage,
    average_precision_50,
    band_coverage,
    box_iou,
    fragment_counts,
    greedy_assignment,
    group_layers,
    merged_band_fragment_counts,
    total_length,
    true_positives_at_iou,
)

RUN_NAME = "yolo11m_aihubv1_20261101_fold{fold}"
FOLDS = (0, 1, 2, 3, 4)
CLASS_NAMES = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")
BAND_CLASS = 1
SCORE_THRESHOLD = 0.25
"""Operating threshold; kept when score > threshold, as the confusion matrix."""
CONFUSION_IOU = 0.45
"""IoU a confusion-matrix match must exceed; equality with the validator's
matrix is checked against ``confusion_matrix_conf0.25.csv``."""
AP_TOLERANCE = 1e-6
FRAGMENT_BINS = ("0", "1", "2", "3", "4", "5+")
OUT_STEM = PROJECT_ROOT / "analysis" / "detection" / "m110_band_aux"
SCHEMA = "gpr_band_aux/1"


@dataclass
class ImageRecord:
    """GT and predictions of one validated image, original pixels."""

    file_name: str
    session: str
    gt_classes: np.ndarray
    gt_boxes: np.ndarray
    pred_classes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    pred_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pred_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))

    def gt_of(self, class_id: int) -> np.ndarray:
        return self.gt_boxes[self.gt_classes == class_id]

    def operating_predictions_of(self, class_id: int) -> np.ndarray:
        keep = (self.pred_classes == class_id) & (self.pred_scores > SCORE_THRESHOLD)
        return self.pred_boxes[keep]


# ---------------------------------------------------------------- reading


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_sessions(manifest: Path) -> dict[str, str]:
    return {row["file_name"]: row["session"] for row in _read_csv(manifest)}


def _read_yolo_labels(label_path: Path, width: int, height: int):
    classes, boxes = [], []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            x_center, y_center, box_w, box_h = (float(v) for v in parts[1:5])
            classes.append(int(parts[0]))
            boxes.append(
                [
                    (x_center - box_w / 2) * width,
                    (y_center - box_h / 2) * height,
                    (x_center + box_w / 2) * width,
                    (y_center + box_h / 2) * height,
                ]
            )
    return np.array(classes, dtype=int), np.array(boxes, dtype=float).reshape(-1, 4)


def _load_records(
    eval_dir: Path, prediction_dir: Path, sessions: dict[str, str]
) -> list[ImageRecord]:
    """Image records in validator order; checks the image list matches."""
    eval_paths = [
        Path(line.strip())
        for line in (eval_dir / "eval_images.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    path_by_name = {path.name: path for path in eval_paths}
    image_rows = _read_csv(prediction_dir / "images.csv")
    if sorted(row["file_name"] for row in image_rows) != sorted(path_by_name):
        raise ValueError(f"{prediction_dir} images differ from {eval_dir} list")
    records = {}
    for row in image_rows:
        image_path = path_by_name[row["file_name"]]
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        gt_classes, gt_boxes = _read_yolo_labels(
            label_path, int(row["width"]), int(row["height"])
        )
        records[row["file_name"]] = ImageRecord(
            row["file_name"], sessions[row["file_name"]], gt_classes, gt_boxes
        )
    _attach_predictions(records, prediction_dir / "predictions.csv")
    return list(records.values())


def _attach_predictions(records: dict[str, ImageRecord], path: Path) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(path):
        grouped[row["file_name"]].append(row)
    for file_name, rows in grouped.items():
        record = records[file_name]
        record.pred_classes = np.array([int(r["cls"]) for r in rows], dtype=int)
        record.pred_scores = np.array([float(r["conf"]) for r in rows])
        record.pred_boxes = np.array(
            [[float(r[key]) for key in ("x1", "y1", "x2", "y2")] for r in rows]
        )


# ---------------------------------------------------------------- checks


def confusion_assignment(
    gt_boxes: np.ndarray, pred_boxes: np.ndarray, pred_scores: np.ndarray
) -> np.ndarray:
    """GT box index per prediction for the confusion matrix (``UNASSIGNED`` = none).

    Class-agnostic one-to-one matching: predictions, in descending score (ties
    keep their input order), each take the free GT box they overlap most when
    the IoU is above ``CONFUSION_IOU``. The class is ignored here so that a box
    found with the wrong class lands in an off-diagonal cell.
    """
    score_order = np.argsort(-pred_scores, kind="stable")
    iou = box_iou(pred_boxes[score_order], gt_boxes)
    gt_of_ranked = greedy_assignment(iou, CONFUSION_IOU, is_threshold_inclusive=False)
    gt_of_prediction = np.full(len(pred_scores), UNASSIGNED, dtype=int)
    gt_of_prediction[score_order] = gt_of_ranked
    return gt_of_prediction


def _add_image_to_confusion(
    matrix: np.ndarray,
    gt_classes: np.ndarray,
    pred_classes: np.ndarray,
    gt_of_prediction: np.ndarray,
) -> None:
    """Count matched pairs, missed GT (background row), false positives (column)."""
    background = len(CLASS_NAMES)
    is_matched = gt_of_prediction != UNASSIGNED
    is_gt_found = np.zeros(len(gt_classes), dtype=bool)
    is_gt_found[gt_of_prediction[is_matched]] = True
    np.add.at(
        matrix,
        (pred_classes[is_matched], gt_classes[gt_of_prediction[is_matched]]),
        1,
    )
    np.add.at(matrix, (background, gt_classes[~is_gt_found]), 1)
    np.add.at(matrix, (pred_classes[~is_matched], background), 1)


def rebuild_confusion_matrix(records: list[ImageRecord]) -> np.ndarray:
    """``[predicted, true]`` matrix at score > 0.25 with a background row/column."""
    background = len(CLASS_NAMES)
    matrix = np.zeros((background + 1, background + 1), dtype=int)
    for record in records:
        keep = record.pred_scores > SCORE_THRESHOLD
        gt_of_prediction = confusion_assignment(
            record.gt_boxes, record.pred_boxes[keep], record.pred_scores[keep]
        )
        _add_image_to_confusion(
            matrix, record.gt_classes, record.pred_classes[keep], gt_of_prediction
        )
    return matrix


def _read_confusion_csv(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return np.array([[int(value) for value in row[1:]] for row in rows[1:]])


def _ap50_inputs(records: list[ImageRecord]):
    """Concatenated (correct, score, class) in validator order, plus GT counts."""
    correct, scores, classes, gt_counts = [], [], [], Counter()
    for record in records:
        correct.append(
            true_positives_at_iou(
                record.gt_boxes,
                record.gt_classes,
                record.pred_boxes,
                record.pred_classes,
            )
        )
        scores.append(record.pred_scores)
        classes.append(record.pred_classes)
        gt_counts.update(record.gt_classes.tolist())
    return (
        np.concatenate(correct),
        np.concatenate(scores),
        np.concatenate(classes),
        gt_counts,
    )


def ap50_per_class(records: list[ImageRecord]) -> dict[int, float | None]:
    correct, scores, classes, gt_counts = _ap50_inputs(records)
    return {
        class_id: average_precision_50(
            correct, scores, classes, gt_counts[class_id], class_id
        )
        for class_id in range(len(CLASS_NAMES))
    }


def _ap50_differences(
    recomputed: dict[int, float | None], metrics_json: Path
) -> dict[str, float]:
    reference = json.loads(metrics_json.read_text(encoding="utf-8"))["per_class"]
    differences = {}
    for row in reference:
        value = recomputed[row["class_id"]]
        if row["ap50"] is None or value is None:
            differences[row["name"]] = 0.0 if row["ap50"] == value else float("inf")
        else:
            differences[row["name"]] = abs(value - row["ap50"])
    return differences


# ---------------------------------------------------------------- metrics


def _coverage_summary(coverages: list[BandCoverage]) -> dict[str, Any]:
    total = sum(coverages, EMPTY_COVERAGE)
    recalls = [c.recall for c in coverages if c.recall is not None]
    precisions = [c.precision for c in coverages if c.precision is not None]
    return {
        "recall": total.recall,
        "precision": total.precision,
        "recall_image_mean": float(np.mean(recalls)) if recalls else None,
        "precision_image_mean": float(np.mean(precisions)) if precisions else None,
        "gt_length_px": total.gt_length_px,
        "covered_gt_length_px": total.covered_gt_length_px,
        "prediction_length_px": total.prediction_length_px,
        "matched_prediction_length_px": total.matched_prediction_length_px,
        "images_with_gt": len(recalls),
        "images_with_prediction": len(precisions),
    }


def _fragment_summary(counts: np.ndarray) -> dict[str, Any]:
    histogram = Counter(min(int(value), len(FRAGMENT_BINS) - 1) for value in counts)
    return {
        "n": len(counts),
        "mean": float(counts.mean()) if len(counts) else None,
        "mean_when_found": float(counts[counts > 0].mean())
        if (counts > 0).any()
        else None,
        "distribution": {
            label: histogram.get(index, 0) for index, label in enumerate(FRAGMENT_BINS)
        },
    }


def _gap_length_within_layers(gt_boxes: np.ndarray) -> float:
    """Unlabelled x-length between the GT pieces of each layer (span - union).

    A band labelled as pieces with gaps loses the gaps from the GT length, so a
    prediction that bridges them lowers the precision; this measures how much.
    """
    gap = 0.0
    for layer in group_layers(gt_boxes, DEFAULT_MIN_VERTICAL_FRACTION):
        intervals = gt_boxes[layer][:, [0, 2]]
        span = intervals[:, 1].max() - intervals[:, 0].min()
        gap += float(span) - total_length(intervals)
    return gap


def band_metrics_of(records: list[ImageRecord]) -> dict[str, Any]:
    """Coverage, fragment counts and band AP50 of a set of images."""
    coverages, per_box, per_band = [], [], []
    n_predictions = 0
    gap_length = 0.0
    for record in records:
        gt = record.gt_of(BAND_CLASS)
        predictions = record.operating_predictions_of(BAND_CLASS)
        n_predictions += len(predictions)
        gap_length += _gap_length_within_layers(gt)
        coverages.append(band_coverage(gt, predictions, DEFAULT_MIN_VERTICAL_FRACTION))
        per_box.append(fragment_counts(gt, predictions))
        per_band.append(merged_band_fragment_counts(gt, predictions))
    return {
        "n_images": len(records),
        "n_gt_band_boxes": int(sum(len(r.gt_of(BAND_CLASS)) for r in records)),
        "n_band_predictions_conf0.25": n_predictions,
        "band_ap50": ap50_per_class(records)[BAND_CLASS],
        "coverage": _coverage_summary(coverages),
        "gt_gap_within_layers_px": gap_length,
        "fragments_per_gt_box": _fragment_summary(np.concatenate(per_box)),
        "fragments_per_merged_gt_band": _fragment_summary(np.concatenate(per_band)),
    }


def session_table(records: list[ImageRecord]) -> list[dict[str, Any]]:
    """Band metrics per session that has a pipe_band GT, plus the rest pooled."""
    by_session: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_session[record.session].append(record)
    rows, others = [], []
    for session, session_records in sorted(by_session.items()):
        if any(len(r.gt_of(BAND_CLASS)) for r in session_records):
            rows.append({"session": session, **band_metrics_of(session_records)})
        else:
            others.extend(session_records)
    if others:
        pooled = band_metrics_of(others)
        pooled["n_sessions"] = len({r.session for r in others})
        rows.append({"session": "(sessions without pipe_band GT)", **pooled})
    return rows


def evaluate_fold(fold: int, unit: str, prediction_root: Path, sessions) -> dict:
    """All metrics and checks of one fold and unit."""
    run_dir = PROJECT_ROOT / "runs" / "detect" / RUN_NAME.format(fold=fold)
    eval_dir = run_dir / "eval" / f"last_{unit}"
    records = _load_records(eval_dir, prediction_root / f"fold{fold}_{unit}", sessions)
    rebuilt = rebuild_confusion_matrix(records)
    reference = _read_confusion_csv(eval_dir / "confusion_matrix_conf0.25.csv")
    ap_differences = _ap50_differences(
        ap50_per_class(records), eval_dir / "metrics.json"
    )
    background = len(CLASS_NAMES)
    return {
        "fold": fold,
        "unit": unit,
        "checks": {
            "confusion_conf0.25_cells_equal": int((rebuilt == reference).sum()),
            "confusion_conf0.25_cells_total": int(reference.size),
            "confusion_matches": bool(np.array_equal(rebuilt, reference)),
            "band_tp_fn_fp_conf0.25": [
                int(rebuilt[BAND_CLASS, BAND_CLASS]),
                int(rebuilt[background, BAND_CLASS]),
                int(rebuilt[BAND_CLASS, background]),
            ],
            "ap50_max_abs_diff_vs_metrics_json": max(ap_differences.values()),
            "ap50_matches": max(ap_differences.values()) <= AP_TOLERANCE,
        },
        **band_metrics_of(records),
        "sessions": session_table(records),
    }


# ---------------------------------------------------------------- summary

SUMMARY_KEYS = (
    ("recall", ("coverage", "recall")),
    ("precision", ("coverage", "precision")),
    ("recall_image_mean", ("coverage", "recall_image_mean")),
    ("precision_image_mean", ("coverage", "precision_image_mean")),
    ("band_ap50", ("band_ap50",)),
    ("fragments_per_gt_box_mean", ("fragments_per_gt_box", "mean")),
    ("fragments_per_merged_gt_band_mean", ("fragments_per_merged_gt_band", "mean")),
)


def _lookup(result: dict, path: tuple[str, ...]) -> float:
    value: Any = result
    for key in path:
        value = value[key]
    return float(value)


def fold_statistics(results: list[dict]) -> dict[str, dict[str, float]]:
    """Unweighted fold mean, SD (ddof=1) and mean - 1 SD of each summary value."""
    statistics = {}
    for name, path in SUMMARY_KEYS:
        values = np.array([_lookup(result, path) for result in results])
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        statistics[name] = {
            "folds": values.tolist(),
            "mean": mean,
            "sd": sd,
            "mean_minus_1sd": mean - sd,
        }
    return statistics


def _write_csv(units: dict[str, dict], path: Path) -> None:
    names = [name for name, _ in SUMMARY_KEYS]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unit", "row", *names])
        for unit, block in units.items():
            stats = block["statistics"]
            for fold_position, fold in enumerate(block["fold_ids"]):
                writer.writerow(
                    [unit, f"fold{fold}"]
                    + [f"{stats[n]['folds'][fold_position]:.6f}" for n in names]
                )
            for row in ("mean", "sd", "mean_minus_1sd"):
                writer.writerow([unit, row] + [f"{stats[n][row]:.6f}" for n in names])


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--units", default="crop,image")
    parser.add_argument(
        "--folds",
        default=",".join(str(fold) for fold in FOLDS),
        help="comma list; the json/csv are written for these folds only",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "analysis" / "m110_band_aux",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "aihub_yolo_v1.csv",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Evaluate every fold and unit, write the json/csv, print a JSON summary."""
    args = _parse_args(argv)
    sessions = _read_sessions(args.manifest)
    folds = [int(fold) for fold in args.folds.split(",")]
    units = {}
    for unit in args.units.split(","):
        results = [
            evaluate_fold(f, unit, args.prediction_root, sessions) for f in folds
        ]
        units[unit] = {
            "fold_ids": folds,
            "statistics": fold_statistics(results),
            "folds": results,
        }
    all_checks = [r["checks"] for block in units.values() for r in block["folds"]]
    passed = all(c["confusion_matches"] and c["ap50_matches"] for c in all_checks)
    document = {
        "schema": SCHEMA,
        "status": "ok" if passed else "check_failed",
        "definition": {
            "class": "pipe_band",
            "score_threshold": f"conf > {SCORE_THRESHOLD}",
            "nms_iou": 0.7,
            "min_vertical_overlap_fraction_of_gt_height": DEFAULT_MIN_VERTICAL_FRACTION,
            "layers": "boxes linked when vertical overlap >= 0.5 x smaller height",
            "main_value": "length-weighted totals over all images of a fold",
            "target_rule": "crop-unit fold mean - 1 SD (ddof=1)",
            "weights": "runs/detect/yolo11m_aihubv1_20261101_fold<k>/weights/last.pt",
            "eval_lists": "eval/last_<unit>/eval_images.txt (imgsz 256, batch 64)",
        },
        "units": units,
    }
    OUT_STEM.with_suffix(".json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(units, OUT_STEM.with_suffix(".csv"))
    crop = units.get("crop", next(iter(units.values())))["statistics"]
    print(
        json.dumps(
            {
                "status": document["status"],
                "recall": crop["recall"],
                "precision": crop["precision"],
            }
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
