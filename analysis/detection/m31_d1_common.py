"""Shared code of the M3-1 D1 r2 scripts: labels, Ultralytics-equal scoring, identity.

Spec: docs/experiments/m31_detection_input_tiling.md v0.5 §8-4 (a) label edges,
(b) scoring equal to Ultralytics 8.4.192, (c) the same-path check G1 inputs,
(e) the ``run_identity`` block and (g) LF-only json/csv output. Used by
``m31_d1_shift_scale.py`` (run) and ``m31_d1_judge.py`` (verdict).
"""

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO / "runs" / "detect"
RUN_PREFIX = "yolo11m_aihubv1_20261101_fold"
LABEL_DIR = REPO / "data" / "aihub_yolo_v1" / "labels"
EXPORT_ROOT = REPO / "runs" / "analysis" / "m110_band_aux"
FOLDS = (0, 1, 2, 3, 4)

CLASS_NAMES = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")
CANVAS_PX = 256
IOU_MATCH = 0.5
CONF_RECALL = 0.25
IOU_EPS = 1e-7  # ultralytics.utils.metrics.box_iou
RECALL_EPS = 1e-16  # ultralytics.utils.metrics.ap_per_class
AP_RECALL_LEVELS = np.linspace(0.0, 1.0, 101)

# Audit record (verif-protocol-auditor, 2026-11-13) and M1-10 instance counts.
EXPECTED_WEIGHTS_SHA8 = ("5300f34b", "cf9423f1", "8dfa5713", "43ae65ee", "19602860")
EXPECTED_N_GT = {
    "pipe_hyperbola": (71, 45, 98, 147, 142),
    "pipe_band": (966, 503, 258, 62, 189),
    "cavity": (365, 367, 364, 367, 364),
    "manhole": (408, 425, 418, 439, 446),
}

# pipe_hyperbola GT that gp6_dx-64 / gp6_dx+64 lose by geometry, folds 0-4
# (v0.5.1 §8-2 (3) G3; same deterministic count as the 1st run).
EXPECTED_GP6_64_HYP_IGNORED = {
    "gp6_dx-64": (0, 0, 0, 3, 2),
    "gp6_dx+64": (0, 0, 3, 3, 0),
}

EXPORT_FILES = ("predictions.csv", "images.csv", "val_metrics.json")
GIT_TIMEOUT_S = 30


class InputError(Exception):
    """A missing or malformed input; the scripts exit with code 2."""


# ----------------------------------------------------------------- labels


def read_yolo_label(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO label file as px boxes clipped to the 256 px crop (§8-4 (a)).

    Returns:
        ``(boxes [n, 4] xyxy px, classes [n] int)``; see
        :func:`read_yolo_label_indexed`.
    """
    boxes, classes, _ = read_yolo_label_indexed(path)
    return boxes, classes


def read_yolo_label_indexed(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a YOLO label file with the line number of every box (§8-4 (a), k1).

    Labels are normalised to 6 decimals, so a box edge can lie up to about
    1.3e-4 px outside the crop; Ultralytics counts those boxes as GT, and the
    clip keeps them inside. A missing file means no objects. Lines that are not
    5 fields are skipped but still counted, so a box keeps its line number.

    Returns:
        ``(boxes [n, 4] xyxy px, classes [n] int, gt_index [n] int)``;
        ``gt_index`` is the 0-based line number in the label file.
    """
    rows = []
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                continue
            x_center, y_center, width, height = map(float, parts[1:])
            rows.append(
                (
                    line_number,
                    int(parts[0]),
                    (x_center - width / 2) * CANVAS_PX,
                    (y_center - height / 2) * CANVAS_PX,
                    (x_center + width / 2) * CANVAS_PX,
                    (y_center + height / 2) * CANVAS_PX,
                )
            )
    table = np.asarray(rows, dtype=float).reshape(-1, 6)
    boxes = table[:, 2:6].clip(0.0, CANVAS_PX)
    return boxes, table[:, 1].astype(int), table[:, 0].astype(int)


# ---------------------------------------------------------------- scoring


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU matrix ``[len(a), len(b)]`` with the Ultralytics eps in the union."""
    boxes_a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (bottom_right - top_left).clip(0.0).prod(axis=2)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).prod(axis=1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).prod(axis=1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection + IOU_EPS)


def match_image(
    pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match one image's predictions of one class to its GT boxes at IoU 0.5.

    ``BaseValidator.match_predictions`` (Ultralytics 8.4.192, ``use_scipy``
    False, IoU 0.5 column): predictions in descending score (ties keep input
    order) that overlap some GT box with IoU >= 0.5 each claim the unclaimed GT
    box with the highest IoU (lowest index on ties) when that IoU is >= 0.5.

    Returns:
        ``(is_true_positive [n_pred] in input order, claim_score [n_gt])``;
        ``claim_score`` is the score of the prediction that claimed the GT box,
        or -1 when none did.
    """
    pred_scores = np.asarray(pred_scores, dtype=float)
    order = np.argsort(-pred_scores, kind="stable")
    is_tp_sorted = np.zeros(len(order), dtype=bool)
    claim_score = np.full(len(gt_boxes), -1.0)
    if len(order) and len(gt_boxes):
        iou = box_iou(gt_boxes, np.asarray(pred_boxes)[order])
        is_claimed = np.zeros(len(gt_boxes), dtype=bool)
        for rank in np.flatnonzero((iou >= IOU_MATCH).any(axis=0)):
            available = np.where(is_claimed, 0.0, iou[:, rank])
            best_gt = int(available.argmax())
            if available[best_gt] >= IOU_MATCH:
                is_tp_sorted[rank] = True
                is_claimed[best_gt] = True
                claim_score[best_gt] = pred_scores[order[rank]]
    is_tp = np.zeros(len(order), dtype=bool)
    is_tp[order] = is_tp_sorted
    return is_tp, claim_score


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point AP of one PR curve, ``ultralytics.utils.metrics.compute_ap``."""
    # The (last recall, 0) sentinel ends the curve with a vertical drop; without
    # it the 1st D1 run added a sloped area up to recall 1 (cause A, §8-3 (4)).
    recall_curve = np.concatenate(([0.0], recall, [recall[-1]], [1.0]))
    precision_curve = np.concatenate(([1.0], precision, [0.0], [0.0]))
    envelope = np.flip(np.maximum.accumulate(np.flip(precision_curve)))
    sampled = np.interp(AP_RECALL_LEVELS, recall_curve, envelope)
    return float(np.trapezoid(sampled, AP_RECALL_LEVELS))


def ap50_per_class(
    scores: np.ndarray,
    pred_classes: np.ndarray,
    is_true_positive: np.ndarray,
    n_gt_per_class: Sequence[int],
) -> list[float | None]:
    """Per-class AP50 as ``ultralytics.utils.metrics.ap_per_class``.

    All predictions are ranked together with numpy's default sort of the
    float32 scores, so tied scores fall in the order the validator sees.

    Returns:
        AP50 per class id; None when the class has no GT or no prediction.
    """
    ranking = np.argsort(-np.asarray(scores, dtype=np.float32))
    ranked_classes = np.asarray(pred_classes)[ranking]
    ranked_tp = np.asarray(is_true_positive, dtype=float)[ranking]
    results: list[float | None] = []
    for class_id, n_gt in enumerate(n_gt_per_class):
        hits = ranked_tp[ranked_classes == class_id]
        if n_gt == 0 or len(hits) == 0:
            results.append(None)
            continue
        true_positives = hits.cumsum()
        false_positives = (1.0 - hits).cumsum()
        recall = true_positives / (n_gt + RECALL_EPS)
        precision = true_positives / (true_positives + false_positives)
        results.append(compute_ap(recall, precision))
    return results


class ScoreAccumulator:
    """Collects matches image by image and gives AP50 and recall per class."""

    def __init__(self, n_classes: int = len(CLASS_NAMES)) -> None:
        self._n_classes = n_classes
        self._scores: list[np.ndarray] = []
        self._classes: list[np.ndarray] = []
        self._is_tp: list[np.ndarray] = []
        self._n_gt = np.zeros(n_classes, dtype=int)
        self._n_hit = np.zeros(n_classes, dtype=int)

    def add_image(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_classes: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
    ) -> np.ndarray:
        """Score one image (predictions in detector order).

        Returns:
            Per GT box (input order) the score of the conf > 0.25 prediction
            that claimed it, or -1 when it was missed at conf > 0.25.
        """
        pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
        pred_scores = np.asarray(pred_scores, dtype=float).reshape(-1)
        pred_classes = np.asarray(pred_classes, dtype=int).reshape(-1)
        gt_classes = np.asarray(gt_classes, dtype=int).reshape(-1)
        is_tp = np.zeros(len(pred_scores), dtype=bool)
        hit_score = np.full(len(gt_classes), -1.0)
        for class_id in range(self._n_classes):
            is_pred = pred_classes == class_id
            is_gt = gt_classes == class_id
            gt_of_class = np.asarray(gt_boxes)[is_gt]
            is_tp[is_pred], _ = match_image(
                pred_boxes[is_pred], pred_scores[is_pred], gt_of_class
            )
            is_confident = is_pred & (pred_scores > CONF_RECALL)
            _, claim_score = match_image(
                pred_boxes[is_confident], pred_scores[is_confident], gt_of_class
            )
            hit_score[is_gt] = claim_score
            self._n_gt[class_id] += int(is_gt.sum())
            self._n_hit[class_id] += int((claim_score >= 0).sum())
        self._scores.append(pred_scores.astype(np.float32))
        self._classes.append(pred_classes)
        self._is_tp.append(is_tp)
        return hit_score

    def class_results(self) -> dict[str, dict]:
        """``{class name: {"n_gt", "ap50", "recall_conf025"}}`` (fractions)."""
        aps = ap50_per_class(
            _concatenate(self._scores, np.float32),
            _concatenate(self._classes, int),
            _concatenate(self._is_tp, bool),
            self._n_gt.tolist(),
        )
        results = {}
        for class_id, name in enumerate(CLASS_NAMES[: self._n_classes]):
            n_gt = int(self._n_gt[class_id])
            recall = self._n_hit[class_id] / n_gt if n_gt else None
            results[name] = {
                "n_gt": n_gt,
                "ap50": aps[class_id],
                "recall_conf025": recall,
            }
        return results


def _concatenate(parts: list[np.ndarray], dtype: type) -> np.ndarray:
    return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)


# ------------------------------------------------- G1 inputs (§8-4 (c))


@dataclass(frozen=True)
class ValExport:
    """M1-10 validator detections of one fold (``export_val_predictions.py``).

    Attributes:
        folder: ``runs/analysis/m110_band_aux/fold<k>_crop``.
        stems: Image stems in validation order (``images.csv`` ``order``).
        n_predictions_listed: Sum of ``images.csv`` ``n_predictions``.
        predictions: Stem -> ``[n, 6]`` rows ``cls, conf, x1, y1, x2, y2``.
        n_prediction_rows: Rows of ``predictions.csv``.
        weights: ``val_metrics.json`` ``weights`` path.
        sha256: File name -> sha256 of the three input files.
    """

    folder: Path
    stems: tuple[str, ...]
    n_predictions_listed: int
    predictions: dict[str, np.ndarray]
    n_prediction_rows: int
    weights: Path
    sha256: dict[str, str]


def export_folder(export_root: Path, fold: int) -> Path:
    """Folder of one fold's validator detections (crop unit)."""
    return export_root / f"fold{fold}_crop"


def export_sha256(folder: Path) -> dict[str, str] | None:
    """sha256 of the three G1 input files, or None when any is missing."""
    paths = [folder / name for name in EXPORT_FILES]
    if not all(path.is_file() for path in paths):
        return None
    return {path.name: sha256_file(path) for path in paths}


def load_val_export(folder: Path) -> ValExport:
    """Read the three G1 input files of one fold.

    Raises:
        InputError: If a file is missing or lacks a needed column or key.
    """
    hashes = export_sha256(folder)
    if hashes is None:
        raise InputError(f"G1 input files missing in {folder}")
    try:
        image_rows = _read_csv_rows(folder / "images.csv")
        image_rows.sort(key=lambda row: int(row["order"]))
        stems = tuple(Path(row["file_name"]).stem for row in image_rows)
        n_listed = sum(int(row["n_predictions"]) for row in image_rows)
        grouped: dict[str, list[tuple[float, ...]]] = {}
        prediction_rows = _read_csv_rows(folder / "predictions.csv")
        for row in prediction_rows:
            values = tuple(
                float(row[key]) for key in ("cls", "conf", "x1", "y1", "x2", "y2")
            )
            grouped.setdefault(Path(row["file_name"]).stem, []).append(values)
        metrics = json.loads((folder / "val_metrics.json").read_text(encoding="utf-8"))
        weights = Path(metrics["weights"])
    except (KeyError, ValueError) as error:
        raise InputError(f"malformed G1 input in {folder}: {error!r}") from error
    predictions = {
        stem: np.asarray(rows, dtype=float) for stem, rows in grouped.items()
    }
    return ValExport(
        folder, stems, n_listed, predictions, len(prediction_rows), weights, hashes
    )


def check_val_export(
    export: ValExport, eval_stems: Sequence[str], weights_path: Path
) -> dict[str, bool]:
    """The G1 input assertions of §8-4 (c), one flag per assertion."""
    return {
        "image_set_equals_eval_list": (
            set(export.stems) == set(eval_stems)
            and len(export.stems) == len(set(export.stems))
        ),
        "n_predictions_sum_equals_rows": (
            export.n_predictions_listed == export.n_prediction_rows
        ),
        "prediction_images_listed": set(export.predictions) <= set(export.stems),
        "weights_path_is_this_fold": Path(export.weights) == Path(weights_path),
    }


def score_val_export(export: ValExport, label_dir: Path) -> dict[str, dict]:
    """Score the validator detections with the D1 code (G1)."""
    accumulator = ScoreAccumulator()
    empty = np.zeros((0, 6))
    for stem in export.stems:
        gt_boxes, gt_classes = read_yolo_label(label_dir / f"{stem}.txt")
        rows = export.predictions.get(stem, empty)
        accumulator.add_image(
            rows[:, 2:6], rows[:, 1], rows[:, 0], gt_boxes, gt_classes
        )
    return accumulator.class_results()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ------------------------------------------------- identity (§8-4 (e))


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eval_list_stems(eval_list: Path) -> list[str]:
    """Image stems of an ``eval_images.txt`` list, in list order."""
    lines = eval_list.read_text(encoding="utf-8").splitlines()
    return [Path(line.strip()).stem for line in lines if line.strip()]


def local_now() -> str:
    """Local time, ISO 8601 with the UTC offset."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def git_state(repo: Path = REPO) -> dict:
    """HEAD and working-tree state; on failure the values are None plus an error."""
    try:
        head = _git(repo, "rev-parse", "HEAD").strip()
        status = [
            line for line in _git(repo, "status", "--porcelain").splitlines() if line
        ]
    except (OSError, subprocess.SubprocessError) as error:
        return {"head": None, "dirty": None, "status": None, "error": repr(error)}
    return {"head": head, "dirty": bool(status), "status": status, "error": None}


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=GIT_TIMEOUT_S,
        check=True,
    )
    return completed.stdout


def library_versions() -> dict[str, str | None]:
    """Versions of the libraries that can change detections or scores."""
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": None,
        "torch": None,
        "torch_cuda": None,
        "ultralytics": _distribution_version("ultralytics"),
        "gpr_detection": _distribution_version("gpr-detection"),
    }
    try:
        import cv2

        versions["opencv"] = cv2.__version__
    except ImportError:
        pass
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["torch_cuda"] = torch.version.cuda
    except ImportError:
        pass
    return versions


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu_name(device: str) -> str:
    """``torch.cuda.get_device_name`` for a GPU device string, else ``"cpu"``."""
    if device.strip().lower() == "cpu":
        return "cpu"
    import torch

    index = int(device) if device.strip().isdigit() else 0
    return torch.cuda.get_device_name(index)


def new_run_identity() -> dict:
    """Start a ``run_identity`` block; the caller adds hashes and ``finished_at``."""
    return {
        "run_id": str(uuid.uuid4()),
        "started_at": local_now(),
        "finished_at": None,
        "argv": list(sys.argv),
        "git": git_state(),
        "versions": library_versions(),
        "gpu": None,
        "weights_sha256": {},
        "eval_list_sha256": {},
        "g1_inputs_sha256": {},
    }


# --------------------------------------------------- output (§8-4 (d)(g))


def write_json(path: Path, document: dict) -> None:
    """Write JSON with LF line ends (UTF-8, non-ASCII kept)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=1, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    """Write CSV with LF line ends."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def print_final_line(exit_code: int, out: Path | str | None) -> int:
    """Print the closing stdout JSON line (no metric values) and return the code."""
    status = "ok" if exit_code == 0 else "error"
    line = {
        "status": status,
        "exit_code": exit_code,
        "out": None if out is None else str(out),
    }
    print(json.dumps(line, ensure_ascii=False), flush=True)
    return exit_code
