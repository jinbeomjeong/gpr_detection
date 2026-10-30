"""Ultralytics YOLO validation turned into plain numbers (M1-10, plan section 5-1).

:func:`evaluate_yolo` runs ``YOLO(weights).val`` on one data yaml and returns a
JSON-ready dict. :func:`metrics_to_dict` does the conversion and needs no
Ultralytics, so it can be tested with a stand-in metrics object.

Facts of Ultralytics 8.4.192 this module relies on:

* ``metrics.box.p/r/ap50/ap`` hold one value per entry of
  ``metrics.ap_class_index`` (the classes that have ground truth), not per
  class id. Classes absent from the evaluated images get ``None`` here.
* ``map50``/``map`` average over those present classes only.
* The confusion matrix is indexed ``[predicted, true]`` with a trailing
  background row/column. It is filled only when ``plots=True``, uses the
  validation ``conf`` as its score threshold (0.001 by default, not 0.25) and
  matches boxes at IoU > 0.45.

``ultralytics`` is imported only inside :func:`evaluate_yolo` (ADR-001).
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gpr_detection.io.classes import DetectionClass

PIPE_CLASS_IDS = (DetectionClass.PIPE_HYPERBOLA.value, DetectionClass.PIPE_BAND.value)
"""Classes averaged into ``pipe_ap50_mean`` (plan section 5-1)."""
CONFUSION_MATRIX_IOU = 0.45
"""IoU match threshold Ultralytics 8.4.192 hard-codes for the confusion matrix."""
BACKGROUND_LABEL = "background"


def _count_or_zero(counts: Any, class_id: int) -> int:
    """Ultralytics leaves the count arrays ``None`` when nothing was evaluated."""
    return 0 if counts is None else int(counts[class_id])


def _class_row(
    metrics: Any, class_id: int, name: str, position: int | None
) -> dict[str, Any]:
    """Metrics of one class; ``position`` indexes the per-present-class arrays."""
    instances = _count_or_zero(metrics.nt_per_class, class_id)
    images = _count_or_zero(metrics.nt_per_image, class_id)
    row: dict[str, Any] = {
        "class_id": class_id,
        "name": name,
        "instances": instances,
        "images": images,
        "precision": None,
        "recall": None,
        "ap50": None,
        "ap50_95": None,
    }
    if position is not None:
        row["precision"] = float(metrics.box.p[position])
        row["recall"] = float(metrics.box.r[position])
        row["ap50"] = float(metrics.box.ap50[position])
        row["ap50_95"] = float(metrics.box.ap[position])
    return row


def _pipe_ap50_mean(per_class: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean AP50 of the pipe classes that are present (``None`` if neither is)."""
    present = [
        row
        for row in per_class
        if row["class_id"] in PIPE_CLASS_IDS and row["ap50"] is not None
    ]
    mean = sum(row["ap50"] for row in present) / len(present) if present else None
    return {"value": mean, "classes": [row["name"] for row in present]}


def _confusion_matrix(
    metrics: Any, class_names: Sequence[str], conf: float, has_plots: bool
) -> dict[str, Any] | None:
    matrix_holder = getattr(metrics, "confusion_matrix", None)
    if not has_plots or matrix_holder is None or matrix_holder.matrix is None:
        return None
    # Ultralytics stores counts as floats.
    matrix = [[round(float(count)) for count in row] for row in matrix_holder.matrix]
    return {
        "labels": [*class_names, BACKGROUND_LABEL],
        "rows": "predicted class; last row = background (ground truth missed)",
        "columns": "true class; last column = background (false positive)",
        "conf_threshold": conf,
        "iou_match_threshold": CONFUSION_MATRIX_IOU,
        "matrix": matrix,
    }


def metrics_to_dict(
    metrics: Any, *, conf: float, has_plots: bool = True
) -> dict[str, Any]:
    """Convert Ultralytics ``DetMetrics`` to plain Python values.

    Args:
        metrics: Result of ``YOLO.val`` (or an object with the same attributes:
            ``names``, ``ap_class_index``, ``nt_per_class``, ``nt_per_image``,
            ``box.p/r/ap50/ap/map50/map``, optionally ``confusion_matrix.matrix``).
        conf: Validation score threshold, recorded with the confusion matrix.
        has_plots: Whether validation ran with ``plots=True``; without it
            Ultralytics leaves the confusion matrix empty and ``None`` is stored.

    Returns:
        Dict with ``per_class`` (every class id, ``None`` metrics for classes
        without ground truth), ``classes_present``, ``map50``, ``map50_95``
        (``None`` when no class is present), ``pipe_ap50_mean`` and
        ``confusion_matrix``.
    """
    names = {int(class_id): str(name) for class_id, name in metrics.names.items()}
    class_ids = sorted(names)
    position_by_class = {
        int(class_id): position
        for position, class_id in enumerate(metrics.ap_class_index)
    }
    per_class = [
        _class_row(metrics, class_id, names[class_id], position_by_class.get(class_id))
        for class_id in class_ids
    ]
    classes_present = sorted(position_by_class)
    has_any_class = bool(classes_present)
    return {
        "class_names": [names[class_id] for class_id in class_ids],
        "classes_present": classes_present,
        "per_class": per_class,
        "map50": float(metrics.box.map50) if has_any_class else None,
        "map50_95": float(metrics.box.map) if has_any_class else None,
        "pipe_ap50_mean": _pipe_ap50_mean(per_class),
        "confusion_matrix": _confusion_matrix(
            metrics, [names[class_id] for class_id in class_ids], conf, has_plots
        ),
    }


def evaluate_yolo(
    weights: Path,
    data_yaml: Path,
    out_dir: Path,
    *,
    imgsz: int = 256,
    batch: int = 64,
    conf: float = 0.001,
    iou: float = 0.7,
    device: str | None = None,
    plots: bool = True,
) -> dict[str, Any]:
    """Validate YOLO weights on the ``val`` list of ``data_yaml``.

    The settings default to the M1-7 re-evaluation (``imgsz 256, batch 64,
    conf 0.001, iou 0.7``). ``YOLO.val`` adds ``rect=True`` on its own.

    Args:
        weights: ``.pt`` checkpoint written by the trainer.
        data_yaml: Ultralytics data yaml (``channels: 1``) whose ``val`` entry
            lists the images to evaluate.
        out_dir: Folder for the Ultralytics plots; reused if it exists.
        imgsz: Input size in pixels.
        batch: Validation batch size.
        conf: Score threshold of the predictions (and the confusion matrix).
        iou: NMS IoU threshold.
        device: Ultralytics device string (``"0"``, ``"cpu"``); ``None`` lets
            Ultralytics choose.
        plots: Save PR curves and the confusion matrix; also required for the
            confusion matrix numbers.

    Returns:
        :func:`metrics_to_dict` result plus ``checkpoint_epoch`` (0-based epoch
        stored in the checkpoint, ``None`` if missing) and ``save_dir``.
    """
    from ultralytics import YOLO  # lazy (ADR-001)

    model = YOLO(str(weights))
    checkpoint = model.ckpt or {}
    val_args: dict[str, Any] = {
        "data": str(data_yaml),
        "split": "val",
        "imgsz": imgsz,
        "batch": batch,
        "conf": conf,
        "iou": iou,
        "plots": plots,
        "project": str(Path(out_dir).parent),
        "name": Path(out_dir).name,
        "exist_ok": True,
    }
    if device is not None:
        val_args["device"] = device
    metrics = model.val(**val_args)

    result = metrics_to_dict(metrics, conf=conf, has_plots=plots)
    epoch = checkpoint.get("epoch")
    result["checkpoint_epoch"] = None if epoch is None else int(epoch)
    result["save_dir"] = str(metrics.save_dir)
    return result
