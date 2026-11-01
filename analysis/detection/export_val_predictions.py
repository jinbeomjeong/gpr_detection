"""Export the per-image detections of one ``YOLO.val`` run (GPU inference).

Runs ``YOLO(weights).val`` the way ``scripts/eval_detector.py`` does (same
``eval_data.yaml``, imgsz 256, batch 64, conf 0.001, NMS iou 0.7, rect batches,
FP16 on GPU) and records, by wrapping ``ConfusionMatrix.process_batch`` in this
process only, the exact detections the validator scores. This is the capture
method of ``m17_pipe_band_fp_gallery.py``, made general by arguments.

Outputs in ``--out-dir`` (put it under ``runs/``, which git ignores):

* ``predictions.csv`` - ``file_name, cls, conf, x1, y1, x2, y2``; boxes in
  original image pixels. Rows keep the validator order (images in rect-batch
  order, detections in NMS order = descending score), which an exact AP
  re-computation needs for tied scores. ``conf`` is
  written with full precision.
* ``images.csv`` - ``order, file_name, width, height, n_predictions`` for every
  validated image, including those without detections.
* ``val_metrics.json`` - per-class AP50 / AP50-95 of this run and the settings.
* ``val/`` - the Ultralytics plots of this run.

Usage::

    python -I analysis/detection/export_val_predictions.py \
        --weights runs/detect/<run>/weights/last.pt \
        --data-yaml runs/detect/<run>/eval/last_crop/eval_data.yaml \
        --out-dir runs/analysis/<name>

The last stdout line is a JSON summary. Exit codes: 0 ok, 2 input error.
Imports ultralytics (AGPL-3.0) as an analysis-only script, like
``analysis/detection/m17_pipe_band_fp_gallery.py``; keep it out of ``src/``
(outside ``detect/``) and ``scripts/``.
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_IMGSZ = 256
DEFAULT_BATCH = 64
DEFAULT_CONF = 0.001
DEFAULT_NMS_IOU = 0.7
SEALED_TEST_MARKER = "test"
"""Val lists whose path mentions this word are refused (test is sealed)."""


@dataclass(frozen=True)
class CapturedImage:
    """Detections of one image in original pixels, in validator order."""

    file_name: str
    width: int
    height: int
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_NMS_IOU)
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def _check_inputs(args: argparse.Namespace) -> list[str]:
    """Image paths of the val list; raises ``ValueError`` on bad input."""
    if not args.weights.is_file():
        raise ValueError(f"missing weights {args.weights}")
    data = yaml.safe_load(args.data_yaml.read_text(encoding="utf-8"))
    val_list = Path(data["val"])
    if any(SEALED_TEST_MARKER in part.lower() for part in val_list.parts):
        raise ValueError(f"refusing a test list (sealed): {val_list}")
    images = [
        line.strip()
        for line in val_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not images:
        raise ValueError(f"empty val list {val_list}")
    return images


def _to_original_pixels(boxes: np.ndarray, ratio_pad: Any) -> np.ndarray:
    """Undo the letterbox: ``ratio_pad = ((gain, gain), (pad_w, pad_h))``."""
    (gain, *_), (pad_w, pad_h) = ratio_pad[0], ratio_pad[1]
    offset = np.array([float(pad_w), float(pad_h), float(pad_w), float(pad_h)])
    return (boxes.astype(float) - offset) / float(gain)


def _run_validation(args: argparse.Namespace) -> tuple[list[CapturedImage], Any]:
    from ultralytics import YOLO
    from ultralytics.utils.metrics import ConfusionMatrix

    captured: list[CapturedImage] = []
    original_process_batch = ConfusionMatrix.process_batch

    def recording_process_batch(self, detections, batch, conf=0.25, iou_thres=0.45):
        height, width = (int(value) for value in batch["ori_shape"])
        boxes = detections["bboxes"].detach().float().cpu().numpy()
        captured.append(
            CapturedImage(
                file_name=Path(batch["im_file"]).name,
                width=width,
                height=height,
                boxes=_to_original_pixels(boxes.reshape(-1, 4), batch["ratio_pad"]),
                scores=detections["conf"].detach().float().cpu().numpy().copy(),
                classes=detections["cls"].detach().cpu().numpy().astype(int).copy(),
            )
        )
        return original_process_batch(
            self, detections, batch, conf=conf, iou_thres=iou_thres
        )

    ConfusionMatrix.process_batch = recording_process_batch
    try:
        model = YOLO(str(args.weights))
        # plots=True: the validator calls process_batch only when plotting.
        metrics = model.val(
            data=str(args.data_yaml),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            plots=True,
            project=str(args.out_dir),
            name="val",
            exist_ok=True,
            verbose=False,
        )
    finally:
        ConfusionMatrix.process_batch = original_process_batch
    return captured, metrics


def _write_predictions(captured: list[CapturedImage], out_dir: Path) -> int:
    rows = 0
    with (out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "cls", "conf", "x1", "y1", "x2", "y2"])
        for image in captured:
            for box, score, class_id in zip(
                image.boxes, image.scores, image.classes, strict=True
            ):
                coordinates = (repr(float(value)) for value in box)
                writer.writerow(
                    [image.file_name, int(class_id), repr(float(score)), *coordinates]
                )
                rows += 1
    return rows


def _write_images(captured: list[CapturedImage], out_dir: Path) -> None:
    with (out_dir / "images.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order", "file_name", "width", "height", "n_predictions"])
        for order, image in enumerate(captured):
            writer.writerow(
                [order, image.file_name, image.width, image.height, len(image.scores)]
            )


def _per_class_metrics(metrics: Any) -> dict[str, dict[str, float]]:
    per_class = {}
    for position, class_id in enumerate(metrics.ap_class_index):
        per_class[str(int(class_id))] = {
            "ap50": float(metrics.box.ap50[position]),
            "ap50_95": float(metrics.box.ap[position]),
        }
    return per_class


def main(argv: list[str] | None = None) -> int:
    """Validate once, write the three output files, print a JSON summary."""
    args = _parse_args(argv)
    try:
        images = _check_inputs(args)
    except (OSError, KeyError, ValueError, yaml.YAMLError) as error:
        print(json.dumps({"status": "input_error", "error": repr(error)}))
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    captured, metrics = _run_validation(args)
    if len(captured) != len(images):
        raise RuntimeError(f"captured {len(captured)} images, val list {len(images)}")
    n_rows = _write_predictions(captured, args.out_dir)
    _write_images(captured, args.out_dir)
    summary = {
        "status": "ok",
        "weights": str(args.weights.resolve()),
        "data_yaml": str(args.data_yaml.resolve()),
        "settings": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "conf": args.conf,
            "nms_iou": args.iou,
            "device": args.device,
        },
        "n_images": len(captured),
        "n_predictions": n_rows,
        "per_class": _per_class_metrics(metrics),
    }
    (args.out_dir / "val_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
