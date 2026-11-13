"""R1 of M3-1 N2-a' (report only): recall of one object seen at two offsets.

Plan docs/experiments/m31_n2a2_retrain_plan.md v1.0 §5 (R1, U2-c) and §7-1.
For every validation fold k, adjacent crops of the same line and gain
(consecutive by ``a`` in ``crops.csv``, positive overlap, both crops the
crop-level representative gain of their crop key: 1.0 > 0.5 > 0.25) are
searched for the same object: same class, IoU >= 0.5 in original x, and the
box lies wholly inside the overlap in both crops. The two crops show it at
different horizontal offsets from the crop centre, without any resampling or
fill, so this is real context independent of the training transforms.

Each object is detected (or not) in each crop with ``last.pt`` of fold k
(``YoloDetector`` at conf 0.001, then a prediction counts if conf > 0.25 and it
claims the box at IoU >= 0.5, same class). Per class and pooled the output
lists, by offset bin (|offset| in 0-32, 32-64, 64-96, > 96 px), the "near"
(smaller |offset|) and "far" objects with their detected counts, and per bin
of the far offset the paired outcomes (both / near only / far only / neither).
Samples are small (tens of hyperbola pairs per fold); counts are reported as
they are, no test statistic.

Always blind (§6): stdout carries only the last line {"status", "exit_code",
"out"}; --blind is accepted so runner commands look alike. --out must not
exist. Exit codes: 0 written, 2 input missing or --out exists, 3 internal error.

Example (runner, PowerShell):
    conda run -n gpr-dev --no-capture-output python -I \
        analysis/detection/m31_r1_overlap_shift.py \
        --run-prefix yolo11m_aihubv1n2a2_<D>_fold --folds 0,1,2,3,4 \
        --device 0 --out analysis/detection/m31_r1_n2a2.json --blind
"""

import argparse
import itertools
import logging
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Same reason as in scripts/train_detector.py: MKL's default OpenMP layer aborts
# once torch is loaded; must be set before numpy is imported.
os.environ.setdefault("MKL_THREADING_LAYER", "TBB")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common

from gpr_detection.augment_geometry import (
    CONTEXT_INDEX_HASHED_FILES,
    CROPS_FILE_NAME,
    TRAIN_POOL_SUBSET,
    ContextCrop,
    file_sha256,
    read_crops_csv,
)
from gpr_detection.data.splits import select_crop_representatives

REPO = common.REPO
DEFAULT_RUN_ROOT = common.RUN_ROOT
DEFAULT_INDEX_DIR = REPO / "data" / "aihub_context_index"
DEFAULT_IMAGE_DIR = REPO / "data" / "aihub_yolo_v1" / "images"
DEFAULT_LABEL_DIR = common.LABEL_DIR
WEIGHTS_RELATIVE = Path("weights") / "last.pt"
PREDICT_CONF = 0.001
RECALL_CONF = common.CONF_RECALL
"""A prediction counts above this score (strictly greater, §5 R1)."""
SAME_OBJECT_IOU = 0.5
OFFSET_BIN_EDGES_PX = (32.0, 64.0, 96.0)
OFFSET_BIN_LABELS = ("0-32", "32-64", "64-96", ">96")
OVERLAP_EDGE_TOLERANCE_PX = 1e-6
"""Label boxes are clipped to the crop; a box ending on the overlap edge is inside."""
ALL_CLASSES = "all"
OUTCOMES = ("both", "near_only", "far_only", "neither")
SCHEMA = "m31_r1_overlap_shift/1"
RULES = "m31_n2a2_retrain_plan.md v1.0 §5 R1 (report only), §7-1"

logger = logging.getLogger("m31_r1_overlap_shift")


@dataclass(frozen=True)
class PairedObject:
    """One object found in two adjacent crops.

    Attributes:
        class_id: Project class id.
        near_file: Crop showing the object nearer to its centre.
        near_gt_index: Line number of the box in the near crop's label file.
        near_offset_px: Box centre x minus crop centre x in the near crop.
        far_file: The other crop.
        far_gt_index: Line number of the box in the far crop's label file.
        far_offset_px: Offset in the far crop.
    """

    class_id: int
    near_file: str
    near_gt_index: int
    near_offset_px: float
    far_file: str
    far_gt_index: int
    far_offset_px: float


@dataclass(frozen=True)
class CropLabels:
    """Label boxes of one crop (crop px, clipped) with their line numbers."""

    boxes: np.ndarray
    classes: np.ndarray
    gt_index: np.ndarray


# ------------------------------------------------------------- pairing


def offset_bin(offset_px: float) -> str:
    """Bin label of ``|offset|``: [0, 32), [32, 64), [64, 96), >= 96 px."""
    magnitude = abs(offset_px)
    for edge, label in zip(OFFSET_BIN_EDGES_PX, OFFSET_BIN_LABELS, strict=False):
        if magnitude < edge:
            return label
    return OFFSET_BIN_LABELS[-1]


def adjacent_pairs(
    crops: Iterable[ContextCrop],
) -> list[tuple[ContextCrop, ContextCrop]]:
    """Consecutive crops (by ``a``) of each ``(line, gain)`` that overlap (> 0 px)."""
    groups: dict[tuple[str, str], list[ContextCrop]] = {}
    for crop in crops:
        groups.setdefault((crop.line, crop.gain), []).append(crop)
    pairs = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda crop: (crop.x1_px, crop.file_name))
        for first, second in itertools.pairwise(ordered):
            if min(first.x2_px, second.x2_px) > max(first.x1_px, second.x1_px):
                pairs.append((first, second))
    return pairs


def _inside_overlap_indices(
    crop: ContextCrop, labels: CropLabels, overlap_x1: int, overlap_x2: int
) -> list[int]:
    inside = []
    for position, box in enumerate(labels.boxes):
        x1 = box[0] + crop.x1_px
        x2 = box[2] + crop.x1_px
        if (
            x1 >= overlap_x1 - OVERLAP_EDGE_TOLERANCE_PX
            and x2 <= overlap_x2 + OVERLAP_EDGE_TOLERANCE_PX
        ):
            inside.append(position)
    return inside


def _original_x(box: np.ndarray, crop: ContextCrop) -> np.ndarray:
    moved = np.array(box, dtype=float)
    moved[[0, 2]] += crop.x1_px
    return moved


def _offset_px(box: np.ndarray, crop: ContextCrop) -> float:
    return float((box[0] + box[2]) / 2 - crop.width_px / 2)


def pair_objects(
    first: ContextCrop,
    second: ContextCrop,
    first_labels: CropLabels,
    second_labels: CropLabels,
) -> list[PairedObject]:
    """Objects labelled wholly inside the overlap of both crops, matched 1:1.

    Matching: same class and IoU >= 0.5 in original x; pairs are taken in
    descending IoU (ties: label order), each box at most once.
    """
    overlap_x1 = max(first.x1_px, second.x1_px)
    overlap_x2 = min(first.x2_px, second.x2_px)
    first_inside = _inside_overlap_indices(first, first_labels, overlap_x1, overlap_x2)
    second_inside = _inside_overlap_indices(
        second, second_labels, overlap_x1, overlap_x2
    )
    candidates = []
    for i in first_inside:
        for j in second_inside:
            if first_labels.classes[i] != second_labels.classes[j]:
                continue
            iou = common.box_iou(
                _original_x(first_labels.boxes[i], first),
                _original_x(second_labels.boxes[j], second),
            )[0, 0]
            if iou >= SAME_OBJECT_IOU:
                candidates.append((-float(iou), i, j))
    used_first: set[int] = set()
    used_second: set[int] = set()
    paired = []
    for _, i, j in sorted(candidates):
        if i in used_first or j in used_second:
            continue
        used_first.add(i)
        used_second.add(j)
        paired.append(_paired_object(first, second, first_labels, second_labels, i, j))
    return paired


def _paired_object(
    first: ContextCrop,
    second: ContextCrop,
    first_labels: CropLabels,
    second_labels: CropLabels,
    i: int,
    j: int,
) -> PairedObject:
    first_offset = _offset_px(first_labels.boxes[i], first)
    second_offset = _offset_px(second_labels.boxes[j], second)
    first_is_near = abs(first_offset) <= abs(second_offset)
    near = (first, first_labels, i, first_offset)
    far = (second, second_labels, j, second_offset)
    if not first_is_near:
        near, far = far, near
    return PairedObject(
        class_id=int(first_labels.classes[i]),
        near_file=near[0].file_name,
        near_gt_index=int(near[1].gt_index[near[2]]),
        near_offset_px=near[3],
        far_file=far[0].file_name,
        far_gt_index=int(far[1].gt_index[far[2]]),
        far_offset_px=far[3],
    )


# ------------------------------------------------------------ summary


def _empty_class_summary() -> dict:
    return {
        "objects_by_offset_bin": {
            label: {
                "near": {"n": 0, "detected": 0},
                "far": {"n": 0, "detected": 0},
            }
            for label in OFFSET_BIN_LABELS
        },
        "pairs_by_far_offset_bin": {
            label: {"n": 0, **dict.fromkeys(OUTCOMES, 0)} for label in OFFSET_BIN_LABELS
        },
    }


def _outcome(is_near_detected: bool, is_far_detected: bool) -> str:
    if is_near_detected and is_far_detected:
        return "both"
    if is_near_detected:
        return "near_only"
    if is_far_detected:
        return "far_only"
    return "neither"


def summarize(
    objects: Sequence[PairedObject],
    detected: dict[tuple[str, int], bool],
    class_names: Sequence[str] = common.CLASS_NAMES,
) -> dict:
    """Counts per class and pooled; ``detected[(file, gt_index)]`` per object."""
    summary = {name: _empty_class_summary() for name in (*class_names, ALL_CLASSES)}
    for item in objects:
        is_near_detected = detected[(item.near_file, item.near_gt_index)]
        is_far_detected = detected[(item.far_file, item.far_gt_index)]
        for name in (class_names[item.class_id], ALL_CLASSES):
            bins = summary[name]["objects_by_offset_bin"]
            near = bins[offset_bin(item.near_offset_px)]["near"]
            near["n"] += 1
            near["detected"] += int(is_near_detected)
            far = bins[offset_bin(item.far_offset_px)]["far"]
            far["n"] += 1
            far["detected"] += int(is_far_detected)
            pair_row = summary[name]["pairs_by_far_offset_bin"][
                offset_bin(item.far_offset_px)
            ]
            pair_row["n"] += 1
            pair_row[_outcome(is_near_detected, is_far_detected)] += 1
    return summary


def _add_recalls(summary: dict) -> dict:
    for class_summary in summary.values():
        for row in class_summary["objects_by_offset_bin"].values():
            for side in row.values():
                side["recall"] = side["detected"] / side["n"] if side["n"] else None
    return summary


# ------------------------------------------------------------- inputs


def validation_crops(crops: Iterable[ContextCrop], fold: int) -> list[ContextCrop]:
    """Train-pool crops of ``fold`` that are their crop's representative gain."""
    in_fold = [
        crop for crop in crops if crop.fold == fold and crop.subset == TRAIN_POOL_SUBSET
    ]
    rows = [
        {"crop_key": crop.crop_key, "gain": crop.gain, "file_name": crop.file_name}
        for crop in in_fold
    ]
    chosen = {str(row["file_name"]) for row in select_crop_representatives(rows)}
    return [crop for crop in in_fold if crop.file_name in chosen]


def read_labels(label_dir: Path, file_name: str) -> CropLabels:
    boxes, classes, gt_index = common.read_yolo_label_indexed(
        label_dir / f"{Path(file_name).stem}.txt"
    )
    return CropLabels(boxes, classes, gt_index)


def fold_objects(
    crops: Sequence[ContextCrop], fold: int, label_dir: Path
) -> tuple[list[PairedObject], int]:
    """Paired objects of one validation fold and the number of crop pairs."""
    pairs = adjacent_pairs(validation_crops(crops, fold))
    objects = []
    for first, second in pairs:
        objects.extend(
            pair_objects(
                first,
                second,
                read_labels(label_dir, first.file_name),
                read_labels(label_dir, second.file_name),
            )
        )
    return objects, len(pairs)


# ----------------------------------------------------------- detection


def detected_objects(
    detector, image_dir: Path, label_dir: Path, objects: Sequence[PairedObject]
) -> dict[tuple[str, int], bool]:
    """Whether each object's box is detected in its crop (conf > 0.25, IoU 0.5)."""
    import cv2  # lazy: the pairing functions need no OpenCV

    files = sorted(
        {item.near_file for item in objects} | {item.far_file for item in objects}
    )
    detected: dict[tuple[str, int], bool] = {}
    for file_name in files:
        image = cv2.imread(str(image_dir / file_name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise common.InputError(f"cannot read {image_dir / file_name}")
        labels = read_labels(label_dir, file_name)
        output = detector.predict(image)
        for gt_position, is_hit in enumerate(_gt_hits(output, labels)):
            detected[(file_name, int(labels.gt_index[gt_position]))] = is_hit
    return detected


def _gt_hits(output, labels: CropLabels) -> list[bool]:
    transform = output.transform
    hits = [False] * len(labels.boxes)
    for class_id in np.unique(labels.classes):
        gt_positions = np.flatnonzero(labels.classes == class_id)
        kept = [
            detection
            for detection in output.detections
            if detection.class_id == class_id and detection.score > RECALL_CONF
        ]
        boxes = np.array(
            [
                [
                    (d.box.x1 - transform.pad_x) / transform.scale_x,
                    (d.box.y1 - transform.pad_y) / transform.scale_y,
                    (d.box.x2 - transform.pad_x) / transform.scale_x,
                    (d.box.y2 - transform.pad_y) / transform.scale_y,
                ]
                for d in kept
            ]
        ).reshape(-1, 4)
        scores = np.array([d.score for d in kept], dtype=float)
        _, claim_score = common.match_image(boxes, scores, labels.boxes[gt_positions])
        for position, score in zip(gt_positions, claim_score, strict=True):
            hits[int(position)] = bool(score >= 0)
    return hits


# ----------------------------------------------------------------- main


def _parse_folds(text: str) -> list[int]:
    folds = [int(part) for part in text.split(",") if part.strip()]
    if not folds or any(fold not in common.FOLDS for fold in folds):
        raise argparse.ArgumentTypeError(f"folds must be among 0-4, got {text!r}")
    return folds


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-prefix",
        required=True,
        help="run folder name without the fold digit "
        "(e.g. yolo11m_aihubv1n2a2_<D>_fold)",
    )
    parser.add_argument("--folds", type=_parse_folds, default=list(common.FOLDS))
    parser.add_argument("--device", default="0", help="e.g. 0 or cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument(
        "--blind",
        action="store_true",
        help="accepted for uniform runner commands; this script is always blind",
    )
    return parser.parse_args(argv)


def _load_detector(weights: Path, device: str):
    from gpr_detection.detect.yolo import load_yolo_detector
    from gpr_detection.detect.yolo_logging import quiet_ultralytics_logger

    quiet_ultralytics_logger()
    return load_yolo_detector(weights, confidence_threshold=PREDICT_CONF, device=device)


def run(args: argparse.Namespace, identity: dict) -> int:
    """Pair, detect and write the report; returns the exit code."""
    if args.out.exists():
        raise common.InputError(f"--out exists (not overwritten): {args.out}")
    weights_by_fold = {}
    for fold in args.folds:
        weights = args.run_root / f"{args.run_prefix}{fold}" / WEIGHTS_RELATIVE
        if not weights.is_file():
            raise common.InputError(f"missing {weights}")
        weights_by_fold[fold] = weights
    identity["gpu"] = common.gpu_name(args.device)
    identity["index_sha256"] = {
        name: file_sha256(args.index_dir / name) for name in CONTEXT_INDEX_HASHED_FILES
    }
    crops = read_crops_csv(args.index_dir / CROPS_FILE_NAME)
    per_fold = {}
    pooled_objects: list[PairedObject] = []
    pooled_detected: dict[tuple[str, int], bool] = {}
    for fold, weights in weights_by_fold.items():
        identity["weights_sha256"][str(fold)] = common.sha256_file(weights)
        objects, pair_count = fold_objects(crops, fold, args.label_dir)
        detector = _load_detector(weights, args.device)
        detected = detected_objects(detector, args.image_dir, args.label_dir, objects)
        per_fold[str(fold)] = {
            "crop_pairs": pair_count,
            "paired_objects": len(objects),
            "by_class": _add_recalls(summarize(objects, detected)),
        }
        pooled_objects.extend(objects)
        pooled_detected.update(detected)
    identity["finished_at"] = common.local_now()
    document = {
        "schema": SCHEMA,
        "rules": RULES,
        "note": "report only (U2-c); not a judgement input",
        "run_identity": identity,
        "inputs": {
            "run_root": str(args.run_root),
            "run_prefix": args.run_prefix,
            "folds": args.folds,
            "index_dir": str(args.index_dir),
            "label_dir": str(args.label_dir),
            "image_dir": str(args.image_dir),
            "predict_conf": PREDICT_CONF,
            "recall_conf_exclusive": RECALL_CONF,
            "offset_bins_px": list(OFFSET_BIN_LABELS),
        },
        "per_fold": per_fold,
        "pooled": {
            "paired_objects": len(pooled_objects),
            "by_class": _add_recalls(summarize(pooled_objects, pooled_detected)),
        },
    }
    common.write_json(args.out, document)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry; returns 0 written, 2 input missing or --out exists, 3 error."""
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
    args = _parse_args(argv)
    identity = common.new_run_identity()
    try:
        code = run(args, identity)
    except common.InputError as error:
        logger.error("input error: %s", error)
        return common.print_final_line(2, args.out)
    except Exception:
        logger.exception("internal error")
        return common.print_final_line(3, args.out)
    return common.print_final_line(code, args.out)


if __name__ == "__main__":
    sys.exit(main())
