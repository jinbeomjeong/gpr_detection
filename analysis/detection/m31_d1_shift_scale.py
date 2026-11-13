"""M3-1 D1: position (dx) and scale (g) sensitivity of the M1-10 YOLO11m models.

Spec: docs/experiments/m31_detection_input_tiling.md v0.5. Design and
pre-registered decision rules §8-2, 1st-run results §8-3, r2 changes §8-4,
r2 results §8-5.

For every fold k the model ``runs/detect/<prefix><k>/weights/last.pt`` is run on
its own validation crops (crop unit, ``eval/last_crop/eval_images.txt``, the
list M1-10 scored). Each 256 x 256 crop is rescaled by ``g`` (isotropic, area
interpolation when shrinking, linear when enlarging), placed on a 256 x 256
canvas filled with ``--fill`` (top row at y = 0 so the surface stays on top,
crop centre column at x = 128 + dx) and cut to the canvas. Labels are read
clipped to the crop ([0, 256] px) and follow the same map. A box that is not
inside the canvas within ``EDGE_TOL_PX`` (or reaches into a fill strip of the
fill-control conditions) becomes an *ignore* box: predictions overlapping it
(IoU >= 0.5 or IoA >= 0.5) are dropped before scoring.

Inference goes through the project backend ``gpr_detection.detect.yolo``
(conf 0.001, NMS IoU 0.7, input 256, one image per call). Because every canvas
is 256 x 256 the backend letterbox is the identity, asserted per image.

Metrics per fold and condition: per-class AP50 and recall at conf > 0.25, both
scored as Ultralytics 8.4.192 does (``m31_d1_common``). The validation set is
never test: lists whose path contains 'test' are refused. ``--dry-labels`` runs
only the label map and writes ``n_gt`` / ``n_ignored`` (no model, no metrics).

Outputs (written once, at the end; existing outputs stop the run with exit 2
unless ``--overwrite``):
  <run>/eval/<out-subdir>/metrics.json and objects.csv per fold (runs/ is
  git-ignored), and the --summary JSON; each carries a ``run_identity`` block.
  objects.csv has one row per kept GT box and condition; ``gt_index`` (label
  line number) pairs the same object across conditions (§8-4 (k) k1).
Stdout carries progress only, never metric values; its last line is
``{"status", "exit_code", "out"}``. Exit codes: 0 done, 2 input missing or
output exists, 3 internal error.
"""

import argparse
import json
import logging
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common

CANVAS = common.CANVAS_PX
CONF_AP = 0.001
NMS_IOU = 0.7
IGNORE_OVERLAP = 0.5
FILL_CONTROL_PX = 64
P_STAR_RATIO = 6.0 / 4.06
# Labels overshoot the crop by up to 1.28e-4 px (6-decimal rounding); 1e-3 px
# keeps them while any real cut (>= 1 px) still becomes an ignore box (§8-4 (a)).
EDGE_TOL_PX = 1e-3
N_EXAMPLE_IMAGES = 4
DEFAULT_OUT_SUBDIR = "m31_d1_r2"
# 1st-run and M1-10 outputs that --overwrite must never delete (§8-4 (j)).
PROTECTED_SUBDIRS = frozenset(
    {"m31_d1", "last_crop", "best_crop", "last_image", "best_image"}
)
PROTECTED_SUMMARIES = frozenset(
    {"m31_d1_summary.json", "m31_d1_verdict.json", "m31_d1_baseline_diag.json"}
)
METRICS_SCHEMA = "m31_d1/2"
DRY_LABELS_SCHEMA = "m31_d1_dry_labels/2"
OBJECTS_HEADER = (
    "fold",
    "condition",
    "image",
    "class",
    "gt_index",
    "x_center",
    "w",
    "h",
    "hit_conf025",
    "score",
)

# name -> (g, dx, fill_strip) ; fill_strip: None, "L", "R"
SCALES_DX0 = (0.5, 0.74, 0.87, 1.0, 1.13, 1.28, P_STAR_RATIO, 1.67)
SHIFTS_G1 = (-96, -64, -32, 32, 64, 96)
SHIFTS_G6 = (-64, -32, 0, 32, 64)

logger = logging.getLogger("m31_d1_shift_scale")


def _cond_name(g: float, dx: int, strip: str | None = None) -> str:
    gtxt = "p6" if abs(g - P_STAR_RATIO) < 1e-9 else f"{g:.2f}"
    base = f"g{gtxt}_dx{dx:+d}"
    return base if strip is None else f"{base}_fill{strip}{FILL_CONTROL_PX}"


def default_conditions() -> dict[str, tuple[float, int, str | None]]:
    """The 20 D1 conditions of §8-2 (2), name -> (g, dx, fill strip)."""
    conds: dict[str, tuple[float, int, str | None]] = {}
    for g in SCALES_DX0:
        conds[_cond_name(g, 0)] = (g, 0, None)
    for dx in SHIFTS_G1:
        conds[_cond_name(1.0, dx)] = (1.0, dx, None)
    for dx in SHIFTS_G6:
        conds[_cond_name(P_STAR_RATIO, dx)] = (P_STAR_RATIO, dx, None)
    conds[_cond_name(1.0, 0, "L")] = (1.0, 0, "L")
    conds[_cond_name(1.0, 0, "R")] = (1.0, 0, "R")
    return conds


@dataclass
class Sample:
    """One validation crop; ``image`` is None in ``--dry-labels`` mode."""

    stem: str
    image: np.ndarray | None  # uint8 [256, 256]
    boxes: np.ndarray  # [n, 4] xyxy px, clipped to [0, 256]
    classes: np.ndarray  # [n]
    gt_index: np.ndarray | None = None  # [n] label line numbers; None -> 0..n-1

    def __post_init__(self) -> None:
        if self.gt_index is None:
            self.gt_index = np.arange(len(self.classes))


@dataclass(frozen=True)
class Placement:
    """Where a rescaled crop lands on the canvas.

    Attributes:
        size: Side of the rescaled crop in px.
        off_x: Canvas column of the crop's left edge (may be negative).
        off_y: Canvas row of the crop's top edge (always 0).
        valid: ``(x0, y0, x1, y1)`` canvas area showing crop pixels, fill
            strips excluded.
    """

    size: int
    off_x: int
    off_y: int
    valid: tuple[float, float, float, float]

    @property
    def scale(self) -> float:
        return self.size / CANVAS


def read_samples(
    eval_list: Path, limit: int | None, load_images: bool = True
) -> list[Sample]:
    """Read the crops of an eval list and their labels (clipped to the crop).

    Raises:
        common.InputError: If the list looks like test or an image is unreadable.
    """
    if "test" in str(eval_list).lower():
        raise common.InputError(f"refusing a list that looks like test: {eval_list}")
    samples = []
    for line in eval_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        path = Path(line)
        image = None
        if load_images:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (CANVAS, CANVAS):
                raise common.InputError(f"bad image {path}")
        label = common.LABEL_DIR / f"{path.stem}.txt"
        boxes, classes, gt_index = common.read_yolo_label_indexed(label)
        samples.append(Sample(path.stem, image, boxes, classes, gt_index))
        if limit is not None and len(samples) >= limit:
            break
    return samples


def place(g: float, dx: int, strip: str | None) -> Placement:
    """Canvas placement of a crop rescaled by ``g`` with its centre at 128 + dx."""
    size = max(1, round(CANVAS * g))
    off_x = round(CANVAS / 2 + dx - size / 2)
    off_y = 0
    x0, x1 = max(0, off_x), min(CANVAS, off_x + size)
    y0, y1 = max(0, off_y), min(CANVAS, off_y + size)
    if strip == "L":
        x0 = max(x0, FILL_CONTROL_PX)
    elif strip == "R":
        x1 = min(x1, CANVAS - FILL_CONTROL_PX)
    return Placement(size, off_x, off_y, (float(x0), float(y0), float(x1), float(y1)))


def render_canvas(
    image: np.ndarray, g: float, placement: Placement, strip: str | None, fill: int
) -> np.ndarray:
    """The 256 x 256 model input of one crop (pixels unchanged since the 1st run)."""
    size, off_x, off_y = placement.size, placement.off_x, placement.off_y
    interp = cv2.INTER_AREA if g < 1 else cv2.INTER_LINEAR
    resized = (
        image
        if size == CANVAS
        else cv2.resize(image, (size, size), interpolation=interp)
    )
    canvas = np.full((CANVAS, CANVAS), fill, dtype=np.uint8)
    x0, x1 = max(0, off_x), min(CANVAS, off_x + size)
    y0, y1 = max(0, off_y), min(CANVAS, off_y + size)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = resized[y0 - off_y : y1 - off_y, x0 - off_x : x1 - off_x]
    if strip == "L":
        canvas[:, :FILL_CONTROL_PX] = fill
    elif strip == "R":
        canvas[:, CANVAS - FILL_CONTROL_PX :] = fill
    return canvas


def map_labels(
    sample: Sample, placement: Placement
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map GT boxes onto the canvas; return (kept_boxes, kept_classes, ignore_boxes)."""
    kept_boxes, kept_classes, _, ignore = map_labels_indexed(sample, placement)
    return kept_boxes, kept_classes, ignore


def map_labels_indexed(
    sample: Sample, placement: Placement
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """As :func:`map_labels`, plus the kept boxes' ``gt_index`` (third item)."""
    offset = np.array([placement.off_x, placement.off_y] * 2, dtype=float)
    boxes = sample.boxes * placement.scale + offset
    x0, y0, x1, y1 = placement.valid
    inside = (
        (boxes[:, 0] >= x0 - EDGE_TOL_PX)
        & (boxes[:, 1] >= y0 - EDGE_TOL_PX)
        & (boxes[:, 2] <= x1 + EDGE_TOL_PX)
        & (boxes[:, 3] <= y1 + EDGE_TOL_PX)
    )
    ignore = boxes[~inside].copy()
    if ignore.size:
        ignore[:, [0, 2]] = ignore[:, [0, 2]].clip(x0, x1)
        ignore[:, [1, 3]] = ignore[:, [1, 3]].clip(y0, y1)
        ignore = ignore[(ignore[:, 2] > ignore[:, 0]) & (ignore[:, 3] > ignore[:, 1])]
    kept_index = sample.gt_index[inside]
    return boxes[inside], sample.classes[inside], kept_index, ignore.reshape(-1, 4)


def transform(
    sample: Sample, g: float, dx: int, strip: str | None, fill: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (canvas, kept_boxes, kept_classes, ignore_boxes) for one condition."""
    placement = place(g, dx, strip)
    canvas = render_canvas(sample.image, g, placement, strip, fill)
    kept_boxes, kept_classes, ignore = map_labels(sample, placement)
    return canvas, kept_boxes, kept_classes, ignore


def ioa_pred(pred: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Intersection over prediction area, [n_pred, n_region]."""
    if len(pred) == 0 or len(region) == 0:
        return np.zeros((len(pred), len(region)))
    ix = np.clip(
        np.minimum(pred[:, None, 2], region[None, :, 2])
        - np.maximum(pred[:, None, 0], region[None, :, 0]),
        0,
        None,
    )
    iy = np.clip(
        np.minimum(pred[:, None, 3], region[None, :, 3])
        - np.maximum(pred[:, None, 1], region[None, :, 1]),
        0,
        None,
    )
    area = np.maximum((pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]), 1e-9)
    return ix * iy / area[:, None]


def _lost_per_class(sample: Sample, kept_classes: np.ndarray) -> np.ndarray:
    n_classes = len(common.CLASS_NAMES)
    return np.bincount(sample.classes, minlength=n_classes) - np.bincount(
        kept_classes, minlength=n_classes
    )


def _predict_identity_letterbox(
    detector, canvas: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = detector.predict(canvas)
    t = out.transform
    if not (t.scale_x == 1 and t.scale_y == 1 and t.pad_x == 0 and t.pad_y == 0):
        raise RuntimeError(f"letterbox not identity: {t}")
    if not out.detections:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    boxes = np.array([d.box.xyxy for d in out.detections], dtype=float)
    scores = np.array([d.score for d in out.detections], dtype=float)
    classes = np.array([d.class_id for d in out.detections], dtype=int)
    return boxes, scores, classes


def _drop_ignored(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, ignore: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (len(ignore) and len(boxes)):
        return boxes, scores, classes
    drop = (common.box_iou(boxes, ignore).max(axis=1) >= IGNORE_OVERLAP) | (
        ioa_pred(boxes, ignore).max(axis=1) >= IGNORE_OVERLAP
    )
    return boxes[~drop], scores[~drop], classes[~drop]


def _save_example(
    save_dir: Path,
    name: str,
    idx: int,
    canvas: np.ndarray,
    gt: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
) -> None:
    vis = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for b in gt:
        cv2.rectangle(
            vis,
            (int(b[0]), int(b[1])),
            (math.ceil(b[2]) - 1, math.ceil(b[3]) - 1),
            (0, 255, 0),
            1,
        )
    for b, s in zip(boxes, scores, strict=True):
        if s > common.CONF_RECALL:
            cv2.rectangle(
                vis,
                (int(b[0]), int(b[1])),
                (math.ceil(b[2]) - 1, math.ceil(b[3]) - 1),
                (0, 0, 255),
                1,
            )
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / f"{name}_{idx}.png"), vis)


def _object_rows(
    fold: int,
    name: str,
    stem: str,
    gt: np.ndarray,
    gt_cls: np.ndarray,
    gt_index: np.ndarray,
    hit_score: np.ndarray,
) -> list[list]:
    rows = []
    for box, class_id, index, score in zip(
        gt, gt_cls, gt_index, hit_score, strict=True
    ):
        rows.append(
            [
                fold,
                name,
                stem,
                common.CLASS_NAMES[class_id],
                int(index),
                round((box[0] + box[2]) / 2, 2),
                round(box[2] - box[0], 2),
                round(box[3] - box[1], 2),
                int(score >= 0),
                round(float(score), 4),
            ]
        )
    return rows


def evaluate_condition(
    detector,
    samples: list[Sample],
    cond: tuple,
    fill: int,
    fold: int,
    name: str,
    object_rows: list,
    save_dir: Path | None,
) -> dict:
    """Run one condition on one fold and score it (fractions, not printed)."""
    g, dx, strip = cond
    placement = place(g, dx, strip)
    accumulator = common.ScoreAccumulator()
    n_ignored = np.zeros(len(common.CLASS_NAMES), dtype=int)
    for idx, sample in enumerate(samples):
        canvas = render_canvas(sample.image, g, placement, strip, fill)
        gt, gt_cls, gt_index, ignore = map_labels_indexed(sample, placement)
        n_ignored += _lost_per_class(sample, gt_cls)
        boxes, scores, classes = _predict_identity_letterbox(detector, canvas)
        boxes, scores, classes = _drop_ignored(boxes, scores, classes, ignore)
        hit_score = accumulator.add_image(boxes, scores, classes, gt, gt_cls)
        object_rows.extend(
            _object_rows(fold, name, sample.stem, gt, gt_cls, gt_index, hit_score)
        )
        if save_dir is not None and idx < N_EXAMPLE_IMAGES:
            _save_example(save_dir, name, idx, canvas, gt, boxes, scores)
    result = {"g": g, "dx": dx, "fill_strip": strip, "classes": {}}
    aps = []
    for class_id, (class_name, scored) in enumerate(
        accumulator.class_results().items()
    ):
        result["classes"][class_name] = {
            "n_gt": scored["n_gt"],
            "n_ignored": int(n_ignored[class_id]),
            "ap50": scored["ap50"],
            "recall_conf025": scored["recall_conf025"],
        }
        if scored["ap50"] is not None:
            aps.append(scored["ap50"])
    result["map50"] = float(np.mean(aps)) if aps else None
    return result


def count_labels(samples: list[Sample], cond: tuple) -> dict:
    """``--dry-labels``: per-class ``n_gt`` and ``n_ignored`` of one condition."""
    g, dx, strip = cond
    placement = place(g, dx, strip)
    n_gt = np.zeros(len(common.CLASS_NAMES), dtype=int)
    n_ignored = np.zeros(len(common.CLASS_NAMES), dtype=int)
    for sample in samples:
        _, kept_classes, _ = map_labels(sample, placement)
        n_gt += np.bincount(kept_classes, minlength=len(common.CLASS_NAMES))
        n_ignored += _lost_per_class(sample, kept_classes)
    classes = {
        name: {"n_gt": int(n_gt[i]), "n_ignored": int(n_ignored[i])}
        for i, name in enumerate(common.CLASS_NAMES)
    }
    return {"g": g, "dx": dx, "fill_strip": strip, "classes": classes}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, default=common.RUN_ROOT)
    parser.add_argument("--run-prefix", default=common.RUN_PREFIX)
    parser.add_argument(
        "--export-root",
        type=Path,
        default=None,
        help="folder of the fold<k>_crop validator detections whose sha256 is "
        f"recorded as the G1 input (G2-c); default: M1-10 ({common.EXPORT_ROOT})",
    )
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--device", default="0", help="'0' (GPU) or 'cpu'")
    parser.add_argument(
        "--fill", type=int, default=128, help="assumed fill grey (Q3 deferred to M3-2)"
    )
    parser.add_argument(
        "--conditions", default="all", help="comma list of names or 'all'"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="images per fold (smoke only)"
    )
    parser.add_argument("--out-subdir", default=DEFAULT_OUT_SUBDIR)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--save-examples", action="store_true")
    parser.add_argument("--list-conditions", action="store_true")
    parser.add_argument(
        "--dry-labels",
        action="store_true",
        help="label map only: write n_gt / n_ignored, no model, no metrics",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete existing outputs of this --out-subdir / --summary first",
    )
    return parser.parse_args(argv)


def _select_conditions(spec: str) -> dict[str, tuple[float, int, str | None]]:
    conds = default_conditions()
    if spec == "all":
        return conds
    wanted = [c.strip() for c in spec.split(",") if c.strip()]
    missing = [c for c in wanted if c not in conds]
    if missing:
        raise common.InputError(f"unknown conditions {missing}")
    return {c: conds[c] for c in wanted}


def _fold_paths(args: argparse.Namespace, fold: int) -> tuple[Path, Path, Path]:
    run_dir = args.run_root / f"{args.run_prefix}{fold}"
    weights = run_dir / "weights" / "last.pt"
    eval_list = run_dir / "eval" / "last_crop" / "eval_images.txt"
    if not weights.is_file() or not eval_list.is_file():
        raise common.InputError(f"missing weights or eval list for fold {fold}")
    return weights, eval_list, run_dir / "eval" / args.out_subdir


def _prepare_outputs(args: argparse.Namespace, out_dirs: list[Path]) -> None:
    """Refuse (or, with --overwrite, delete) existing outputs before any write."""
    subdir = Path(args.out_subdir)
    if len(subdir.parts) != 1 or args.out_subdir in {".", ".."}:
        raise common.InputError(
            f"--out-subdir must be a plain folder name: {args.out_subdir}"
        )
    existing = [path for path in out_dirs if path.exists()]
    if args.summary is not None and args.summary.exists():
        existing.append(args.summary)
    if not existing:
        return
    if not args.overwrite:
        raise common.InputError(
            f"outputs exist (use --overwrite): {[str(p) for p in existing]}"
        )
    if args.out_subdir in PROTECTED_SUBDIRS or (
        args.summary is not None and args.summary.name in PROTECTED_SUMMARIES
    ):
        raise common.InputError("refusing to overwrite 1st-run or M1-10 outputs")
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _record_inputs(
    identity: dict,
    fold: int,
    weights: Path,
    eval_list: Path,
    export_root: Path | None = None,
) -> None:
    """Hash this fold's inputs; ``export_root`` None = ``common.EXPORT_ROOT``."""
    identity["weights_sha256"][str(fold)] = common.sha256_file(weights)
    identity["eval_list_sha256"][str(fold)] = common.sha256_file(eval_list)
    root = common.EXPORT_ROOT if export_root is None else export_root
    folder = common.export_folder(root, fold)
    identity["g1_inputs_sha256"][str(fold)] = common.export_sha256(folder)


def _load_detector(weights: Path, device: str):
    from gpr_detection.detect.yolo import (
        load_yolo_detector,
    )  # lazy (AGPL inside detect/)

    return load_yolo_detector(
        weights,
        input_size=CANVAS,
        confidence_threshold=CONF_AP,
        iou_threshold=NMS_IOU,
        device=device,
    )


def _run_fold(
    args: argparse.Namespace,
    fold: int,
    conds: dict,
    weights: Path,
    eval_list: Path,
    out_dir: Path,
    started: float,
) -> tuple[dict, list]:
    samples = read_samples(eval_list, args.limit, load_images=not args.dry_labels)
    detector = None if args.dry_labels else _load_detector(weights, args.device)
    object_rows: list = []
    fold_result = {
        "weights": str(weights),
        "eval_list": str(eval_list),
        "n_images": len(samples),
        "conditions": {},
    }
    for name, cond in conds.items():
        t0 = time.time()
        if args.dry_labels:
            res = count_labels(samples, cond)
        else:
            save_dir = out_dir / "examples" if args.save_examples else None
            res = evaluate_condition(
                detector, samples, cond, args.fill, fold, name, object_rows, save_dir
            )
        res["seconds"] = round(time.time() - t0, 1)
        fold_result["conditions"][name] = res
        print(
            f"fold {fold} {name}: {len(samples)} images, {res['seconds']} s "
            f"(elapsed {time.time() - started:.0f} s)",
            flush=True,
        )
    return fold_result, object_rows


def _write_outputs(args: argparse.Namespace, summary: dict, fold_outputs: dict) -> None:
    identity = summary["run_identity"]
    for fold, (out_dir, fold_result, object_rows) in fold_outputs.items():
        common.write_json(
            out_dir / "metrics.json",
            {"schema": summary["schema"], "run_identity": identity, **fold_result},
        )
        if not args.dry_labels:
            common.write_csv(out_dir / "objects.csv", OBJECTS_HEADER, object_rows)
        summary["folds"][str(fold)] = fold_result
    if args.summary is not None:
        common.write_json(args.summary, summary)


def run(args: argparse.Namespace) -> None:
    """Evaluate (or count labels for) the selected folds and write all outputs."""
    conds = _select_conditions(args.conditions)
    folds = [int(f) for f in args.folds.split(",")]
    paths = {fold: _fold_paths(args, fold) for fold in folds}
    _prepare_outputs(args, [out_dir for _, _, out_dir in paths.values()])
    identity = common.new_run_identity()
    identity["gpu"] = "cpu" if args.dry_labels else common.gpu_name(args.device)
    summary = {
        "schema": DRY_LABELS_SCHEMA if args.dry_labels else METRICS_SCHEMA,
        "run_identity": identity,
        "fill": args.fill,
        "limit": args.limit,
        "device": args.device,
        "dry_labels": args.dry_labels,
        "conditions": {k: list(v) for k, v in conds.items()},
        "folds": {},
    }
    started = time.time()
    fold_outputs = {}
    for fold, (weights, eval_list, out_dir) in paths.items():
        _record_inputs(identity, fold, weights, eval_list, args.export_root)
        fold_result, object_rows = _run_fold(
            args, fold, conds, weights, eval_list, out_dir, started
        )
        if not args.dry_labels:
            m110 = json.loads(
                (eval_list.parent / "metrics.json").read_text(encoding="utf-8")
            )
            fold_result["m110_ap50"] = {
                pc["name"]: pc["ap50"] for pc in m110["per_class"]
            }
        fold_outputs[fold] = (out_dir, fold_result, object_rows)
    summary["seconds_total"] = round(time.time() - started, 1)
    identity["finished_at"] = common.local_now()
    _write_outputs(args, summary, fold_outputs)


def _out_label(args: argparse.Namespace) -> Path:
    if args.summary is not None:
        return args.summary
    return args.run_root / f"{args.run_prefix}<k>" / "eval" / args.out_subdir


def main(argv: list[str] | None = None) -> int:
    """CLI entry; returns the exit code (0 done, 2 input/output refused, 3 error)."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    if args.list_conditions:
        print(json.dumps(list(default_conditions())))
        return 0
    try:
        run(args)
    except common.InputError as error:
        logger.error("input error: %s", error)
        return common.print_final_line(2, _out_label(args))
    except Exception:
        logger.exception("internal error")
        return common.print_final_line(3, _out_label(args))
    return common.print_final_line(0, _out_label(args))


if __name__ == "__main__":
    sys.exit(main())
