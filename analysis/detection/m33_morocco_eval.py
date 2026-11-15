"""M3-3 R1 Morocco cross-domain evaluation (P5 check) with N2-a' folds 0-4.

Spec: ``docs/experiments/m33_runner_spec.md`` v0.3.6 §4 ([RUN] R1) and §1
(common checks through ``m33_inputs``); plan
``docs/experiments/m33_cross_domain_plan.md`` v0.2.8 §2-1, §3-5 P5.

Data: ``data/morocco_eval_v1`` (233 images, YOLO labels) and the manifest
``data/splits/morocco_eval_v1.csv``; the X7 sha of both, of the 233 images
and of the 233 labels are asserted (LF for the text files), as are no
forbidden augmentation copy (every ``aug_transform`` geometry preserving)
and one image per group.

Path: the S3 predict path (``gpr_detection.detect.yolo``, conf 0.001, NMS
0.7, one image at a time, 1-channel). Main condition ``main``: the 224 px
image resized to 256 x 256 (``cv2.INTER_LINEAR``, the Ultralytics letterbox
resize). Sensitivity conditions (report only): ``s1_pad`` (no resize, centred
on the 128-grey canvas), ``s2_scale_0.75`` / ``s2_scale_1.33`` (isotropic
around the main size, centred pad or crop), ``s3_top_mask`` (rows 0-27 of
the source set to 128 first), ``s4_near_duplicate`` (the main predictions
without ``U_239_aug_6``), ``s5_period`` (isotropic ``INTER_AREA`` so the
X6-method period becomes ``P_A`` 4.06 px, centred on the canvas).

Scoring per fold, condition and PP version (``pre`` = S0, ``post`` = the
sealed C0b set; Morocco crops have ``y_bot`` 256 and no zero, so only PP-c
could act): ``pipe_hyperbola`` AP50 (Ultralytics-equal,
``m31_d1_common.ScoreAccumulator``), AP50-95, recall and precision at conf
0.25 on ``utilities`` + ``intact``; false alarms per ``intact`` image per
class; ``cavity`` AP50 and recall on ``cavities`` (kept apart); a confusion
matrix with background; an image bootstrap 95 % CI (B 2000, seed 20261115)
of the main condition. P5 (``p5_adoption_input``) is the fold mean of the
post - pre ``pipe_hyperbola`` AP50; it must be exactly 0 in every fold
(``p5_structural_zero``), else exit 1.

Outputs: ``fold<k>/predictions.csv``, ``fold<k>/objects.csv`` (full
precision), ``metrics.json`` and ``summary.json`` (``judgment``,
``timings``). A full run happens once: an existing output folder is refused
(no ``--overwrite``); ``--smoke`` (8 images, fold 0) writes its own folder.

Blind: values go to the files only. The last stdout line is ``{"status",
"out_dir", "n_images", "elapsed_s"}``. Exit codes: 0 done, 1 P5 not zero
(``gate_failed``) or not judgeable, 2 input refused, 3 internal error.
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common
import m32_aihub_perturb as perturb
import m32_recipe_inputs as inputs
import m33_inputs
import numpy as np

from gpr_detection.data.morocco_eval import is_geometry_preserving
from gpr_detection.io.radargram import Radargram
from gpr_detection.pipeline import PP_SETS, ContentRows, PPConfig, PPSetId, pp_decision
from gpr_detection.preprocess.filters import prepare_fitting_branch

REPO = common.REPO
SCHEMA = "m33_morocco_eval/1"
CANVAS_PX = common.CANVAS_PX
PAD_GRAY = 128
CONF_AP = perturb.CONF_AP
CONF_OPERATING = common.CONF_RECALL
NMS_IOU = perturb.NMS_IOU
PERIOD_TARGET_PX = perturb.PERIOD_AIHUB_PX
"""``P_A`` 4.06 px: (s5) scales every image to this period ([RUN] §4)."""
SEED = 20261115
BOOTSTRAP_B = 2000
IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
TOP_MASK_ROWS = 28
"""(s3): source rows 0-27 (the top 1/8 of 224) are set to 128."""
SCALES = (0.75, 1.33)
NEAR_DUPLICATE_DROPPED = "U_239_aug_6"
"""(s4): the near-duplicate pair U_233 / U_239 counts once ([X7] §1 1)."""
N_IMAGES = 233
SMOKE_IMAGES = 8
X6_JSON = REPO / "analysis" / "processing" / "m33_x6_period_snr.json"
X6_QUANTILES = (5, 25, 50, 75, 95)
"""``m33_x6_period_snr.py`` ``pct``: these percentiles, rounded to 3 decimals."""
N_CASES = 12
CASE_DIR = "cases"
SMOKE_FOLD = 0

HYPERBOLA = 0
CAVITY = 2
SUBSET_UTILITIES = "utilities"
SUBSET_INTACT = "intact"
SUBSET_CAVITIES = "cavities"
MAIN_SUBSETS = (SUBSET_UTILITIES, SUBSET_INTACT)
PP_PRE = "pre"
PP_POST = "post"
MAIN = "main"
S4 = "s4_near_duplicate"
CONDITIONS = (
    MAIN,
    "s1_pad",
    "s2_scale_0.75",
    "s2_scale_1.33",
    "s3_top_mask",
    "s5_period",
)
"""Inferred conditions; ``s4_near_duplicate`` reuses the main predictions."""

STATUS_GATE_FAILED = "gate_failed"
EXIT_CODES = {
    m33_inputs.STATUS_OK: 0,
    STATUS_GATE_FAILED: 1,
    m33_inputs.STATUS_NOT_JUDGEABLE: 1,
    m33_inputs.STATUS_INPUT_ERROR: 2,
    "error": 3,
}
PREDICTION_COLUMNS = (
    "fold",
    "condition",
    "pp",
    "image",
    "subset",
    "x1",
    "y1",
    "x2",
    "y2",
    "score",
    "class",
    "pp_action",
)
OBJECT_COLUMNS = (
    "fold",
    "condition",
    "pp",
    "image",
    "subset",
    "class",
    "gt_index",
    "x1",
    "y1",
    "x2",
    "y2",
    "hit_conf025",
    "hit_score",
)

logger = logging.getLogger("m33_morocco_eval")


# ----------------------------------------------------------------- inputs


@dataclass(frozen=True)
class MoroccoImage:
    """One evaluation image with its labels (normalised, source frame)."""

    stem: str
    subset: str
    group_id: str
    pixels: np.ndarray
    boxes_norm: np.ndarray
    classes: np.ndarray
    gt_index: np.ndarray
    period_px: float | None = None
    """X6-method pixel period (``period_native_px``), set once per run (s5)."""


def _sha_list(paths: Iterable[Path]) -> str:
    """X7 aggregate: sha256 of the file sha256 joined by newlines."""
    joined = "\n".join(common.sha256_file(path) for path in paths)
    return hashlib.sha256(joined.encode()).hexdigest()


def assert_inputs(data_dir: Path, manifest_path: Path) -> dict[str, str]:
    """X7 sha of the manifest, list, images and labels; copies and groups.

    Raises:
        inputs.InputError: For any difference (exit 2).
    """
    hashes = {
        "manifest": inputs.checked_text_sha256(
            manifest_path, m33_inputs.MOROCCO_SHA256_LF["manifest"], "manifest"
        ),
        "eval_images": inputs.checked_text_sha256(
            data_dir / "eval_images.txt",
            m33_inputs.MOROCCO_SHA256_LF["eval_images"],
            "eval_images.txt",
        ),
    }
    raw = {
        "manifest": common.sha256_file(manifest_path),
        "eval_images": common.sha256_file(data_dir / "eval_images.txt"),
    }
    if raw != m33_inputs.MOROCCO_SHA256_RAW:
        raise inputs.InputError(f"Morocco raw sha differ from X7: {raw}")
    rows = inputs.read_csv(manifest_path)
    names = sorted(row["file_name"] for row in rows)
    images = _sha_list(data_dir / "images" / name for name in names)
    labels = _sha_list(data_dir / "labels" / f"{Path(name).stem}.txt" for name in names)
    if images != m33_inputs.MOROCCO_IMAGES_SHA256:
        raise inputs.InputError("Morocco image sha differ from X7")
    if labels != m33_inputs.MOROCCO_LABELS_SHA256:
        raise inputs.InputError("Morocco label sha differ from X7")
    check_manifest(rows)
    return hashes | {"images_aggregate": images, "labels_aggregate": labels}


def check_manifest(rows: Sequence[Mapping[str, str]]) -> None:
    """233 rows, no forbidden augmentation copy, one image per group.

    Raises:
        inputs.InputError: For any of the three.
    """
    if len(rows) != N_IMAGES:
        raise inputs.InputError(f"manifest has {len(rows)} rows, want {N_IMAGES}")
    forbidden = [
        r["image_id"] for r in rows if not is_geometry_preserving(r["aug_transform"])
    ]
    if forbidden:
        raise inputs.InputError(f"forbidden augmentation copies: {forbidden[:5]}")
    groups = Counter(row["group_id"] for row in rows)
    repeated = [group for group, n in groups.items() if n != 1]
    if repeated:
        raise inputs.InputError(f"groups with more than one image: {repeated[:5]}")


def smoke_selection(images: Sequence[MoroccoImage]) -> list[MoroccoImage]:
    """Eight images for the smoke: 3 utilities, 3 intact, 2 cavities."""
    quota = {SUBSET_UTILITIES: 3, SUBSET_INTACT: 3, SUBSET_CAVITIES: 2}
    chosen = []
    for image in images:
        if quota.get(image.subset, 0) > 0:
            chosen.append(image)
            quota[image.subset] -= 1
    return chosen


def read_images(data_dir: Path, manifest_path: Path) -> list[MoroccoImage]:
    """Images of ``eval_images.txt`` in list order with labels and subset."""
    subset_of = {
        row["file_name"]: (row["subset"], row["group_id"])
        for row in inputs.read_csv(manifest_path)
    }
    images = []
    for name in perturb.read_eval_names(data_dir / "eval_images.txt"):
        pixels = cv2.imread(str(data_dir / "images" / name), cv2.IMREAD_GRAYSCALE)
        if pixels is None:
            raise inputs.InputError(f"unreadable image {name}")
        boxes, classes, gt_index = common.read_yolo_label_indexed(
            data_dir / "labels" / f"{Path(name).stem}.txt"
        )
        subset, group_id = subset_of[name]
        images.append(
            MoroccoImage(
                stem=Path(name).stem,
                subset=subset,
                group_id=group_id,
                pixels=pixels,
                boxes_norm=boxes / CANVAS_PX,
                classes=classes,
                gt_index=gt_index,
            )
        )
    return images


# ------------------------------------------------------------- conditions


@dataclass(frozen=True)
class CanvasImage:
    """A 256 x 256 network input and its ground truth in canvas px."""

    pixels: np.ndarray
    boxes: np.ndarray
    classes: np.ndarray
    gt_index: np.ndarray
    n_dropped: int = 0
    """Ground-truth boxes with no area left on the canvas (s2 1.33 crop)."""


def _place(scaled: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Centre an image on the 128-grey canvas (pad or crop); its offsets."""
    canvas = np.full((CANVAS_PX, CANVAS_PX), PAD_GRAY, dtype=np.uint8)
    height, width = scaled.shape
    off_y, off_x = (CANVAS_PX - height) // 2, (CANVAS_PX - width) // 2
    src_y, src_x = max(0, -off_y), max(0, -off_x)
    dst_y, dst_x = max(0, off_y), max(0, off_x)
    rows = min(height - src_y, CANVAS_PX - dst_y)
    cols = min(width - src_x, CANVAS_PX - dst_x)
    canvas[dst_y : dst_y + rows, dst_x : dst_x + cols] = scaled[
        src_y : src_y + rows, src_x : src_x + cols
    ]
    return canvas, off_x, off_y


def _transform_boxes(
    image: MoroccoImage, size: tuple[int, int], offset: tuple[int, int]
) -> CanvasImage:
    """Ground truth scaled to ``size`` (w, h), shifted and clipped to the canvas.

    Boxes with no area left on the canvas are dropped.
    """
    width, height = size
    boxes = image.boxes_norm * np.array([width, height, width, height], dtype=float)
    boxes = (boxes + np.array([*offset, *offset], dtype=float)).clip(0.0, CANVAS_PX)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return CanvasImage(
        np.empty(0),
        boxes[keep],
        image.classes[keep],
        image.gt_index[keep],
        int((~keep).sum()),
    )


def _scaled(image: MoroccoImage, pixels: np.ndarray, size: tuple[int, int], area: bool):
    interpolation = cv2.INTER_AREA if area else cv2.INTER_LINEAR
    scaled = cv2.resize(pixels, size, interpolation=interpolation)
    canvas, off_x, off_y = _place(scaled)
    truth = _transform_boxes(image, size, (off_x, off_y))
    return replace(truth, pixels=canvas)


def period_native_px(pixels: np.ndarray, stem: str) -> float | None:
    """X6 method: ``prepare_fitting_branch`` pixel period of the grey image."""
    radargram = Radargram(
        data=pixels.astype(np.float32),
        dt_ns=None,
        dx_m=None,
        air_gap="unknown",
        amplitude_processed=False,
        dataset="morocco",
        line_id=stem,
    )
    return prepare_fitting_branch(radargram).period_px


def with_periods(images: Sequence[MoroccoImage]) -> list[MoroccoImage]:
    """Images with their X6-method period, measured once per run."""
    return [
        replace(image, period_px=period_native_px(image.pixels, image.stem))
        for image in images
    ]


def period_quantiles(periods: Sequence[float | None]) -> dict | None:
    """X6 ``pct``: linear percentiles 5/25/50/75/95 rounded to 3 decimals, n."""
    values = np.asarray([p for p in periods if p is not None], dtype=np.float64)
    if values.size == 0:
        return None
    quantiles = {
        f"p{q}": round(float(np.percentile(values, q)), 3) for q in X6_QUANTILES
    }
    return quantiles | {"n": int(values.size)}


def period_record(images: Sequence[MoroccoImage], x6_path: Path = X6_JSON) -> dict:
    """Re-measured period quantiles beside the X6 json ones, per subset and all."""
    x6 = json.loads(x6_path.read_text(encoding="utf-8"))["morocco"]["subsets"]
    groups: dict[str, list[MoroccoImage]] = {"all": list(images)}
    for image in images:
        groups.setdefault(image.subset, []).append(image)
    record = {}
    for name, members in sorted(groups.items()):
        periods = [image.period_px for image in members]
        record[name] = {
            "measured": period_quantiles(periods),
            "n_no_period": sum(period is None for period in periods),
            "x6": (x6.get(name) or {}).get("P_native_px"),
        }
    return record | {"x6_sha256_lf": inputs.sha256_file_lf(x6_path)}


def assert_periods_match_x6(record: Mapping) -> None:
    """Full run: the 233 re-measured periods give the X6 json quantiles.

    gpr-processing check 2026-11-16: X6 measured the same function on the
    same images; ``n_no_period`` must be 0 as well.

    Raises:
        inputs.InputError: For another quantile or a missing period (exit 2).
    """
    every = record["all"]
    if every["n_no_period"] != 0 or every["measured"] != every["x6"]:
        raise inputs.InputError(
            f"s5 periods differ from X6: {every['measured']} vs {every['x6']}, "
            f"{every['n_no_period']} without a period"
        )


def condition_image(image: MoroccoImage, condition: str) -> CanvasImage | None:
    """Network input of one condition (``None``: no period for ``s5``)."""
    pixels = image.pixels
    height, width = pixels.shape
    if condition == MAIN:
        return _scaled(image, pixels, (CANVAS_PX, CANVAS_PX), area=False)
    if condition == "s1_pad":
        return _scaled(image, pixels, (width, height), area=False)
    if condition.startswith("s2_scale_"):
        factor = float(condition.removeprefix("s2_scale_"))
        side = round(CANVAS_PX * factor)
        return _scaled(image, pixels, (side, side), area=factor < 1)
    if condition == "s3_top_mask":
        masked = pixels.copy()
        masked[:TOP_MASK_ROWS, :] = PAD_GRAY
        return _scaled(image, masked, (CANVAS_PX, CANVAS_PX), area=False)
    if condition == "s5_period":
        period = image.period_px
        if period is None:
            return None
        scale = PERIOD_TARGET_PX / period
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return _scaled(image, pixels, size, area=True)
    raise ValueError(f"unknown condition {condition}")


# ------------------------------------------------------------- inference


@dataclass(frozen=True)
class Predictions:
    """Detector output of one canvas (px), with the PP action of each box."""

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    actions: tuple[str, ...] = ()

    def subset(self, keep: np.ndarray) -> "Predictions":
        return Predictions(
            self.boxes[keep],
            self.scores[keep],
            self.classes[keep],
            tuple(a for a, k in zip(self.actions, keep, strict=True) if k),
        )


CROP_CONTENT = ContentRows(
    y_bot_px=float(CANVAS_PX), period_px=PERIOD_TARGET_PX, zero_px=None
)
"""Square crops: no padding (``y_bot`` 256), no zero ([RUN] §4 PP)."""


def apply_crop_pp(
    predictions: Predictions, config: PPConfig
) -> tuple[Predictions, Predictions]:
    """``(pre, post)``: the same boxes with their actions, and the PP output."""
    actions, keep, boxes = [], [], predictions.boxes.copy()
    for index, (box, score, class_id) in enumerate(
        zip(predictions.boxes, predictions.scores, predictions.classes, strict=True)
    ):
        decision = pp_decision(
            int(class_id),
            float(score),
            float(box[1]),
            float(box[3]),
            CROP_CONTENT,
            config,
        )
        actions.append(decision.action.value)
        keep.append(not decision.action.is_removed)
        boxes[index, 3] = decision.y2_px
    pre = Predictions(
        predictions.boxes, predictions.scores, predictions.classes, tuple(actions)
    )
    post = Predictions(boxes, predictions.scores, predictions.classes, tuple(actions))
    return pre, post.subset(np.asarray(keep, dtype=bool))


def load_detector(weights: Path, device: str):
    """N2-a' fold model at the AP threshold (S3 path)."""
    # Lazy: ultralytics stays inside gpr_detection.detect (AGPL boundary).
    from gpr_detection.detect.yolo import load_yolo_detector
    from gpr_detection.detect.yolo_logging import quiet_ultralytics_logger

    quiet_ultralytics_logger()
    return load_yolo_detector(
        weights,
        input_size=CANVAS_PX,
        confidence_threshold=CONF_AP,
        iou_threshold=NMS_IOU,
        device=device,
    )


def predict(detector, canvas: np.ndarray) -> Predictions:
    """One image through the S3 predict path (identity letterbox asserted)."""
    boxes, scores, classes = perturb.predict(detector, canvas)
    return Predictions(boxes, scores, classes)


# ---------------------------------------------------------------- scoring


@dataclass
class Scored:
    """Predictions and ground truth of one condition, image by image."""

    images: list[str] = field(default_factory=list)
    subsets: list[str] = field(default_factory=list)
    truths: list[CanvasImage] = field(default_factory=list)
    predictions: list[Predictions] = field(default_factory=list)

    def add(self, image: MoroccoImage, truth: CanvasImage, prediction: Predictions):
        self.images.append(image.stem)
        self.subsets.append(image.subset)
        self.truths.append(truth)
        self.predictions.append(prediction)

    def where(self, keep: Callable[[str, str], bool]) -> "Scored":
        chosen = Scored()
        for name, subset, truth, prediction in zip(
            self.images, self.subsets, self.truths, self.predictions, strict=True
        ):
            if keep(name, subset):
                chosen.images.append(name)
                chosen.subsets.append(subset)
                chosen.truths.append(truth)
                chosen.predictions.append(prediction)
        return chosen


def match_at(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """``m31_d1_common.match_image`` at any IoU threshold (TP flags, input order)."""
    order = np.argsort(-np.asarray(pred_scores, dtype=float), kind="stable")
    is_tp_sorted = np.zeros(len(order), dtype=bool)
    if len(order) and len(gt_boxes):
        iou = common.box_iou(gt_boxes, np.asarray(pred_boxes)[order])
        is_claimed = np.zeros(len(gt_boxes), dtype=bool)
        for rank in np.flatnonzero((iou >= threshold).any(axis=0)):
            available = np.where(is_claimed, 0.0, iou[:, rank])
            best_gt = int(available.argmax())
            if available[best_gt] >= threshold:
                is_tp_sorted[rank] = True
                is_claimed[best_gt] = True
    is_tp = np.zeros(len(order), dtype=bool)
    is_tp[order] = is_tp_sorted
    return is_tp


@dataclass(frozen=True)
class ImageMatches:
    """Per image: TP flags per IoU threshold for every prediction (own class)."""

    scores: np.ndarray
    classes: np.ndarray
    is_tp: np.ndarray  # [n_thresholds, n_pred]
    n_gt: np.ndarray  # [n_classes]


def image_matches(truth: CanvasImage, prediction: Predictions) -> ImageMatches:
    n_classes = len(common.CLASS_NAMES)
    is_tp = np.zeros((len(IOU_THRESHOLDS), len(prediction.scores)), dtype=bool)
    for class_id in range(n_classes):
        is_pred = prediction.classes == class_id
        gt = truth.boxes[truth.classes == class_id]
        for row, threshold in enumerate(IOU_THRESHOLDS):
            is_tp[row, is_pred] = match_at(
                prediction.boxes[is_pred], prediction.scores[is_pred], gt, threshold
            )
    n_gt = np.bincount(truth.classes, minlength=n_classes)
    return ImageMatches(prediction.scores, prediction.classes, is_tp, n_gt)


def ap_from_matches(matches: Sequence[ImageMatches], class_id: int) -> dict:
    """AP50 and AP50-95 of one class (Ultralytics ranking and 101-point AP)."""
    if not matches:
        return {"ap50": None, "ap50_95": None, "n_gt": 0}
    scores = np.concatenate([m.scores for m in matches])
    classes = np.concatenate([m.classes for m in matches])
    n_gt = sum(m.n_gt for m in matches).tolist()
    per_threshold = []
    for row in range(len(IOU_THRESHOLDS)):
        is_tp = np.concatenate([m.is_tp[row] for m in matches])
        per_threshold.append(
            common.ap50_per_class(scores, classes, is_tp, n_gt)[class_id]
        )
    ap50 = per_threshold[0]
    ap50_95 = None if ap50 is None else float(np.mean(per_threshold))
    return {"ap50": ap50, "ap50_95": ap50_95, "n_gt": int(n_gt[class_id])}


def image_hits(
    truth: CanvasImage, prediction: Predictions, class_id: int
) -> tuple[int, int, int]:
    """``(n_gt, n_hit, n_pred)`` of one image at conf > 0.25, IoU 0.5."""
    confident = (prediction.classes == class_id) & (prediction.scores > CONF_OPERATING)
    gt = truth.boxes[truth.classes == class_id]
    is_tp = match_at(prediction.boxes[confident], prediction.scores[confident], gt, 0.5)
    return len(gt), int(is_tp.sum()), int(confident.sum())


def recall_precision(scored: Scored, class_id: int) -> dict:
    """Recall and precision at conf > 0.25, IoU 0.5 (one class)."""
    n_gt = n_hit = n_pred = 0
    for truth, prediction in zip(scored.truths, scored.predictions, strict=True):
        image_gt, image_hit, image_pred = image_hits(truth, prediction, class_id)
        n_gt += image_gt
        n_hit += image_hit
        n_pred += image_pred
    return {
        "recall_conf025": n_hit / n_gt if n_gt else None,
        "precision_conf025": n_hit / n_pred if n_pred else None,
        "n_gt": n_gt,
        "n_pred_conf025": n_pred,
    }


def false_alarms_per_image(scored: Scored) -> dict:
    """Conf > 0.25 boxes per image, per class (the ``intact`` images)."""
    n_images = len(scored.images)
    counts = Counter()
    for prediction in scored.predictions:
        for class_id in prediction.classes[prediction.scores > CONF_OPERATING]:
            counts[common.CLASS_NAMES[int(class_id)]] += 1
    return {
        "n_images": n_images,
        "per_image": {
            name: (counts[name] / n_images if n_images else None)
            for name in common.CLASS_NAMES
        },
    }


def confusion_matrix(scored: Scored) -> dict:
    """Class-agnostic greedy IoU >= 0.5 match at conf > 0.25, with background.

    Rows are predicted classes plus ``background`` (a missed GT), columns GT
    classes plus ``background`` (a prediction with no GT).
    """
    names = [*common.CLASS_NAMES, "background"]
    matrix = {row: dict.fromkeys(names, 0) for row in names}
    for truth, prediction in zip(scored.truths, scored.predictions, strict=True):
        confident = prediction.scores > CONF_OPERATING
        boxes, classes = prediction.boxes[confident], prediction.classes[confident]
        pairs = []
        if len(boxes) and len(truth.boxes):
            iou = common.box_iou(boxes, truth.boxes)
            pairs = sorted(
                (
                    (iou[p, g], p, g)
                    for p, g in zip(*np.nonzero(iou >= 0.5), strict=True)
                ),
                reverse=True,
            )
        used_p, used_g = set(), set()
        for _, p, g in pairs:
            if p in used_p or g in used_g:
                continue
            used_p.add(p)
            used_g.add(g)
            matrix[names[int(classes[p])]][names[int(truth.classes[g])]] += 1
        for p in set(range(len(boxes))) - used_p:
            matrix[names[int(classes[p])]]["background"] += 1
        for g in set(range(len(truth.boxes))) - used_g:
            matrix["background"][names[int(truth.classes[g])]] += 1
    return matrix


def is_main_subset(_: str, subset: str) -> bool:
    return subset in MAIN_SUBSETS


def condition_metrics(scored: Scored) -> dict:
    """Every [RUN] §4 metric of one condition and PP version."""
    main = scored.where(is_main_subset)
    intact = scored.where(lambda _, subset: subset == SUBSET_INTACT)
    cavities = scored.where(lambda _, subset: subset == SUBSET_CAVITIES)
    main_matches = [
        image_matches(t, p) for t, p in zip(main.truths, main.predictions, strict=True)
    ]
    cavity_matches = [
        image_matches(t, p)
        for t, p in zip(cavities.truths, cavities.predictions, strict=True)
    ]
    return {
        "utilities_intact": {
            "pipe_hyperbola": ap_from_matches(main_matches, HYPERBOLA)
            | recall_precision(main, HYPERBOLA),
            "n_images": len(main.images),
        },
        "intact_false_alarms": false_alarms_per_image(intact),
        "cavities": {
            "cavity": ap_from_matches(cavity_matches, CAVITY)
            | recall_precision(cavities, CAVITY),
            "n_images": len(cavities.images),
        },
        "confusion_utilities_intact": confusion_matrix(main),
    }


def bootstrap_fold_mean(
    scored_by_fold: Mapping[int, Scored],
    seed: int = SEED,
    n_draws: int = BOOTSTRAP_B,
) -> dict:
    """Image bootstrap 95 % CI of the 5-fold mean ([RUN] §4 구현 해석 (5)).

    One draw resamples the images once; every fold scores that same sample
    and the fold values are averaged. Main path, PP before.
    """
    rng = np.random.default_rng(seed)

    def subset(keep: Callable[[str, str], bool]) -> dict[int, Scored]:
        return {fold: scored.where(keep) for fold, scored in scored_by_fold.items()}

    cavities = subset(lambda _, name: name == SUBSET_CAVITIES)
    return {
        "utilities_intact_pipe_hyperbola": _bootstrap_folds(
            subset(is_main_subset), HYPERBOLA, rng, n_draws
        ),
        "cavities_cavity": _bootstrap_folds(cavities, CAVITY, rng, n_draws),
        "folds": sorted(scored_by_fold),
        "n_draws": n_draws,
        "seed": seed,
    }


@dataclass(frozen=True)
class _FoldSample:
    """Per-image matches and conf > 0.25 hits of one fold (reused per draw)."""

    matches: list[ImageMatches]
    hits: np.ndarray  # [n_images, 2]: n_gt, n_hit


def _fold_sample(scored: Scored, class_id: int) -> _FoldSample:
    pairs = list(zip(scored.truths, scored.predictions, strict=True))
    return _FoldSample(
        [image_matches(truth, prediction) for truth, prediction in pairs],
        np.array([image_hits(t, p, class_id)[:2] for t, p in pairs]).reshape(-1, 2),
    )


def _bootstrap_folds(
    scored_by_fold: Mapping[int, Scored], class_id: int, rng, n_draws: int
) -> dict:
    """Images drawn with replacement; every fold scores the same draw."""
    orders = {tuple(scored.images) for scored in scored_by_fold.values()}
    if len(orders) != 1:
        raise RuntimeError("folds scored different image lists")
    n_images = len(next(iter(orders)))
    if n_images == 0:
        return {"ap50": None, "recall_conf025": None}
    samples = [_fold_sample(s, class_id) for s in scored_by_fold.values()]
    ap_means, recall_means = [], []
    for _ in range(n_draws):
        chosen = rng.integers(0, n_images, n_images)
        aps, recalls = [], []
        for sample in samples:
            ap = ap_from_matches([sample.matches[i] for i in chosen], class_id)["ap50"]
            n_gt, n_hit = sample.hits[chosen].sum(axis=0)
            if ap is not None:
                aps.append(ap)
            if n_gt:
                recalls.append(n_hit / n_gt)
        if aps:
            ap_means.append(float(np.mean(aps)))
        if recalls:
            recall_means.append(float(np.mean(recalls)))
    return {
        "ap50": _interval(ap_means),
        "recall_conf025": _interval(recall_means),
    }


def _interval(values: Sequence[float]) -> dict | None:
    if not values:
        return None
    low, high = np.percentile(np.asarray(values), [2.5, 97.5])
    return {"low": float(low), "high": float(high), "n": len(values)}


# ---------------------------------------------------------------- cases


def image_errors(truth: CanvasImage, prediction: Predictions) -> dict:
    """Misses and false alarms at conf > 0.25, same-class IoU >= 0.5 greedy."""
    n_gt = n_pred = n_tp = 0
    matched_scores: list[float] = []
    for class_id in range(len(common.CLASS_NAMES)):
        confident = (prediction.classes == class_id) & (
            prediction.scores > CONF_OPERATING
        )
        gt = truth.boxes[truth.classes == class_id]
        is_tp = match_at(
            prediction.boxes[confident], prediction.scores[confident], gt, 0.5
        )
        n_gt += len(gt)
        n_pred += int(confident.sum())
        n_tp += int(is_tp.sum())
        matched_scores += prediction.scores[confident][is_tp].tolist()
    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "errors": (n_gt - n_tp) + (n_pred - n_tp),
        "mean_matched_score": float(np.mean(matched_scores)) if matched_scores else 0.0,
    }


def rank_cases(scored: Scored, n_cases: int = N_CASES) -> dict[str, list[int]]:
    """Image indices of the case figures ([RUN] §4 사례 PNG 순위).

    ``bottom``: most errors first, ties by name. ``top``: images with ground
    truth, fewest errors first, ties by the higher mean matched score, then
    by name.
    """
    rows = [
        image_errors(truth, prediction)
        for truth, prediction in zip(scored.truths, scored.predictions, strict=True)
    ]
    names = scored.images
    bottom = sorted(range(len(rows)), key=lambda i: (-rows[i]["errors"], names[i]))
    with_truth = [i for i in range(len(rows)) if rows[i]["n_gt"] > 0]
    top = sorted(
        with_truth,
        key=lambda i: (rows[i]["errors"], -rows[i]["mean_matched_score"], names[i]),
    )
    return {"top": top[:n_cases], "bottom": bottom[:n_cases]}


def draw_case(truth: CanvasImage, prediction: Predictions) -> np.ndarray:
    """Canvas with ground truth (green) and conf > 0.25 boxes (red, class id)."""
    figure = cv2.cvtColor(truth.pixels, cv2.COLOR_GRAY2BGR)
    for box in truth.boxes:
        x1, y1, x2, y2 = (round(float(v)) for v in box)
        cv2.rectangle(figure, (x1, y1), (x2, y2), (0, 200, 0), 1)
    confident = prediction.scores > CONF_OPERATING
    for box, class_id in zip(
        prediction.boxes[confident], prediction.classes[confident], strict=True
    ):
        x1, y1, x2, y2 = (round(float(v)) for v in box)
        cv2.rectangle(figure, (x1, y1), (x2, y2), (0, 0, 220), 1)
        cv2.putText(
            figure, str(int(class_id)), (x1 + 2, y1 + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 220), 1,
        )  # fmt: skip
    return figure


def write_cases(out_root: Path, scored: Scored) -> Path:
    """Top and bottom case PNGs of fold 0, main path, PP before."""
    case_dir = out_root / CASE_DIR
    case_dir.mkdir(parents=True, exist_ok=True)
    for kind, indices in rank_cases(scored).items():
        for rank, index in enumerate(indices, start=1):
            path = case_dir / f"{kind}_{rank:02d}_{scored.images[index]}.png"
            figure = draw_case(scored.truths[index], scored.predictions[index])
            if not cv2.imwrite(str(path), figure):
                raise OSError(f"cannot write {path}")
    return case_dir


# ------------------------------------------------------------------ rows


def prediction_rows(
    fold: int, condition: str, pp: str, image: str, subset: str, prediction: Predictions
) -> list[list]:
    return [
        [
            fold,
            condition,
            pp,
            image,
            subset,
            *(float(v) for v in box),
            float(score),
            common.CLASS_NAMES[int(class_id)],
            action,
        ]
        for box, score, class_id, action in zip(
            prediction.boxes,
            prediction.scores,
            prediction.classes,
            prediction.actions or ("",) * len(prediction.scores),
            strict=True,
        )
    ]


def object_rows(
    fold: int,
    condition: str,
    pp: str,
    image: str,
    subset: str,
    truth: CanvasImage,
    prediction: Predictions,
) -> list[list]:
    """Ground truth boxes with their conf > 0.25 hit (full precision)."""
    accumulator = common.ScoreAccumulator()
    hit_score = accumulator.add_image(
        prediction.boxes,
        prediction.scores,
        prediction.classes,
        truth.boxes,
        truth.classes,
    )
    return [
        [
            fold, condition, pp, image, subset, common.CLASS_NAMES[int(class_id)],
            int(index), *(float(v) for v in box), bool(score >= 0), float(score),
        ]
        for box, class_id, index, score in zip(
            truth.boxes, truth.classes, truth.gt_index, hit_score, strict=True
        )
    ]  # fmt: skip


# ----------------------------------------------------------------- folds


@dataclass
class FoldRun:
    """Scored conditions of one fold and its rows."""

    scored: dict[tuple[str, str], Scored] = field(default_factory=dict)
    prediction_rows: list[list] = field(default_factory=list)
    object_rows: list[list] = field(default_factory=list)
    n_no_period: int = 0
    n_gt_dropped: Counter = field(default_factory=Counter)
    """Ground-truth boxes with no area left on the canvas, per condition."""


def run_fold(
    detector, images: Sequence[MoroccoImage], fold: int, config: PPConfig, timer
) -> FoldRun:
    """Every condition of one fold: infer, PP, keep rows and scores."""
    result = FoldRun()
    for condition in CONDITIONS:
        with timer.phase(f"infer_{condition}"):
            for image in images:
                canvas = condition_image(image, condition)
                if canvas is None:
                    result.n_no_period += 1
                    continue
                result.n_gt_dropped[condition] += canvas.n_dropped
                _add_image(
                    result,
                    fold,
                    condition,
                    image,
                    canvas,
                    predict(detector, canvas.pixels),
                    config,
                )
    for pp in (PP_PRE, PP_POST):
        result.scored[(S4, pp)] = result.scored[(MAIN, pp)].where(
            lambda name, _: name != NEAR_DUPLICATE_DROPPED
        )
    return result


def _add_image(result, fold, condition, image, canvas, raw, config) -> None:
    for pp, prediction in zip(
        (PP_PRE, PP_POST), apply_crop_pp(raw, config), strict=True
    ):
        result.scored.setdefault((condition, pp), Scored()).add(
            image, canvas, prediction
        )
        result.prediction_rows += prediction_rows(
            fold, condition, pp, image.stem, image.subset, prediction
        )
        result.object_rows += object_rows(
            fold, condition, pp, image.stem, image.subset, canvas, prediction
        )


def fold_metrics(run: FoldRun) -> dict:
    return {
        f"{condition}/{pp}": condition_metrics(scored)
        for (condition, pp), scored in sorted(run.scored.items())
    }


def p5_record(metrics_by_fold: Mapping[int, Mapping]) -> dict:
    """``p5_adoption_input``: fold mean of post - pre hyperbola AP50 (main).

    The structural-zero check is "PP before and after give the same AP50 in
    every fold" ([RUN] §4 PP): ``None`` (no ``pipe_hyperbola`` prediction in
    scope, e.g. a few smoke images) on both sides is the same value, so it
    passes; ``None`` on one side only does not. A fold with ``None`` on both
    sides has no difference and stays out of the mean.
    """
    differences: dict[str, float | None] = {}
    unchanged = []
    for fold, metrics in metrics_by_fold.items():
        pre = metrics[f"{MAIN}/{PP_PRE}"]["utilities_intact"]["pipe_hyperbola"]["ap50"]
        post = metrics[f"{MAIN}/{PP_POST}"]["utilities_intact"]["pipe_hyperbola"][
            "ap50"
        ]
        unchanged.append(pre == post)
        differences[str(fold)] = None if pre is None or post is None else post - pre
    values = [v for v in differences.values() if v is not None]
    return {
        "p5_adoption_input": float(np.mean(values)) if values else None,
        "p5_by_fold": differences,
        "p5_structural_zero": all(unchanged),
        "p5_folds_without_ap50": sum(v is None for v in differences.values()),
    }


def fold_summary(metrics_by_fold: Mapping[int, Mapping]) -> dict:
    """Mean and SD over the folds of the headline values per condition."""
    keys = (
        sorted(next(iter(metrics_by_fold.values())).keys()) if metrics_by_fold else []
    )
    headline = {
        "pipe_hyperbola_ap50": ("utilities_intact", "pipe_hyperbola", "ap50"),
        "pipe_hyperbola_ap50_95": ("utilities_intact", "pipe_hyperbola", "ap50_95"),
        "pipe_hyperbola_recall_conf025": (
            "utilities_intact",
            "pipe_hyperbola",
            "recall_conf025",
        ),
        "cavity_ap50": ("cavities", "cavity", "ap50"),
    }
    table = {}
    for key in keys:
        table[key] = {
            name: perturb.mean_and_sd(
                [m[key][scope][cls][item] for m in metrics_by_fold.values()]
            )
            for name, (scope, cls, item) in headline.items()
        }
    return table


# ------------------------------------------------------------------- run


def code_paths() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [
        Path(__file__).resolve(),
        here / "m31_d1_common.py",
        here / "m32_aihub_perturb.py",
        here / "m32_recipe_inputs.py",
        here / "m33_inputs.py",
        REPO / "src" / "gpr_detection" / "pipeline.py",
        REPO / "src" / "gpr_detection" / "detect" / "yolo.py",
        REPO / "src" / "gpr_detection" / "data" / "morocco_eval.py",
        REPO / "src" / "gpr_detection" / "preprocess" / "filters.py",
    ]


@dataclass(frozen=True)
class Settings:
    """Command-line choices."""

    out_root: Path
    folds: tuple[int, ...]
    device: str
    smoke: bool
    limit: int | None
    allow_dirty: bool
    run_root: Path
    data_dir: Path = m33_inputs.MOROCCO_DIR
    manifest: Path = m33_inputs.MOROCCO_MANIFEST

    @property
    def uses_every_image(self) -> bool:
        """All 233 images (no smoke, no ``--limit``): the s5 periods are asserted."""
        return not self.smoke and self.limit is None

    def scope_blockers(self) -> list[str]:
        blockers = []
        if self.smoke:
            blockers.append(m33_inputs.BLOCKER_SMOKE)
        if not m33_inputs.has_all_folds(self.folds):
            blockers.append(m33_inputs.BLOCKER_PARTIAL_FOLDS)
        if self.limit is not None:
            blockers.append(m33_inputs.BLOCKER_LINE_LIMIT)
        return blockers


def weights_path(run_root: Path, fold: int) -> Path:
    return run_root / f"{perturb.RUN_PREFIX}{fold}" / "weights" / "last.pt"


def prepare_out_root(settings: Settings) -> None:
    """A full run writes once; the smoke folder is replaced.

    Raises:
        inputs.InputError: For an existing output folder of a non-smoke run.
    """
    if settings.out_root.exists():
        if not settings.smoke:
            raise inputs.InputError(
                f"output exists (R1 runs once): {settings.out_root}"
            )
        shutil.rmtree(settings.out_root)
    settings.out_root.mkdir(parents=True)


def run(settings: Settings, started: float) -> tuple[str, int]:
    """Assert the inputs, run every fold and write the outputs.

    Returns:
        ``(status, n_images)``.

    Raises:
        inputs.InputError: For a refused input (exit 2).
    """
    m33_inputs.check_folds(settings.folds)
    required = [*code_paths(), X6_JSON]
    start_tree = m33_inputs.tree_state(required, code_paths())
    if not settings.allow_dirty and start_tree["problems"]:
        raise inputs.InputError(f"tree not clean: {start_tree['problems']}")
    sealed = m33_inputs.assert_sealed_pp(start_tree["code_blobs"])
    settings_check = m33_inputs.assert_settings(CONF_OPERATING, CONF_AP, NMS_IOU)
    config = PP_SETS[PPSetId(sealed["pp_set_id"])]
    hashes = assert_inputs(settings.data_dir, settings.manifest)
    weights = _checked_weights(settings)
    timer = PhaseTimer()
    with timer.phase("load"):
        images = select_images(settings)
    periods = period_record(images)
    if settings.uses_every_image:
        assert_periods_match_x6(periods)
    prepare_out_root(settings)
    folds = run_folds(settings, images, config, timer, started)
    with timer.phase("bootstrap"):
        bootstrap = bootstrap_fold_mean(
            {fold: run.scored[(MAIN, PP_PRE)] for fold, run in folds.runs.items()}
        )
    if 0 in folds.runs:
        case_dir = write_cases(settings.out_root, folds.runs[0].scored[(MAIN, PP_PRE)])
        print(f"case png: {case_dir}", flush=True)
    for name, value in timer.seconds.items():
        print(f"timing {name}: {value:.1f} s", flush=True)
    metrics_by_fold = folds.metrics
    first_run = next(iter(folds.runs.values()))
    n_no_period = first_run.n_no_period
    end_tree = m33_inputs.tree_state(required, code_paths())
    judgment = m33_inputs.judgment_record(
        settings.folds, settings.scope_blockers(), start_tree, end_tree
    )
    p5 = p5_record(metrics_by_fold)
    base = m33_inputs.STATUS_OK if p5["p5_structural_zero"] else STATUS_GATE_FAILED
    status = m33_inputs.final_status(base, judgment)
    identity = _identity(settings, start_tree, end_tree, sealed, hashes, weights)
    identity["settings_assertion"] = settings_check
    common.write_json(
        settings.out_root / "metrics.json",
        {str(fold): m for fold, m in metrics_by_fold.items()},
    )
    common.write_json(
        settings.out_root / "summary.json",
        {
            "schema": SCHEMA,
            "status": status,
            "judgment": judgment,
            "run_identity": identity,
            "n_images": len(images),
            "folds": list(settings.folds),
            "conditions": [*CONDITIONS, S4],
            "n_images_without_period_s5": n_no_period,
            "n_gt_dropped_by_condition": dict(first_run.n_gt_dropped),
            "s5_periods": periods | {"asserted": settings.uses_every_image},
            **p5,
            "fold_summary": fold_summary(metrics_by_fold),
            "bootstrap_fold_mean": bootstrap,
            "case_rule": "fold 0, main, PP before, conf > 0.25, same-class IoU 0.5",
            "timings": dict(timer.seconds),
        },
    )
    return status, len(images)


def _checked_weights(settings: Settings) -> dict[str, str]:
    return {
        str(fold): inputs.checked_sha256(
            weights_path(settings.run_root, fold),
            perturb.EXPECTED_WEIGHTS_SHA256[fold],
            f"fold {fold} weights",
        )
        for fold in settings.folds
    }


def select_images(settings: Settings) -> list[MoroccoImage]:
    """The run's images with their periods (smoke: 8, ``--limit``: first n)."""
    images = read_images(settings.data_dir, settings.manifest)
    images = smoke_selection(images) if settings.smoke else images
    images = images if settings.limit is None else images[: settings.limit]
    return with_periods(images)


@dataclass
class FoldsRun:
    """Every fold's scored conditions and metrics."""

    runs: dict[int, FoldRun] = field(default_factory=dict)
    metrics: dict[int, dict] = field(default_factory=dict)


def run_folds(
    settings: Settings,
    images: Sequence[MoroccoImage],
    config: PPConfig,
    timer: "PhaseTimer",
    started: float,
) -> FoldsRun:
    """Infer, score and write each fold (predictions and objects)."""
    result = FoldsRun()
    for fold in settings.folds:
        detector = load_detector(weights_path(settings.run_root, fold), settings.device)
        fold_run = run_fold(detector, images, fold, config, timer)
        with timer.phase("score"):
            result.metrics[fold] = fold_metrics(fold_run)
        _write_fold(settings.out_root, fold, fold_run)
        result.runs[fold] = fold_run
        print(f"fold {fold}: {len(images)} images x {len(CONDITIONS)} conditions "
              f"({time.time() - started:.0f} s)", flush=True)  # fmt: skip
    return result


@dataclass
class PhaseTimer:
    """Wall seconds per named phase (``timings``)."""

    seconds: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        begin = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - begin
            self.seconds[name] = self.seconds.get(name, 0.0) + elapsed


def _identity(settings, start_tree, end_tree, sealed, hashes, weights) -> dict:
    identity = common.new_run_identity()
    del identity["g1_inputs_sha256"]
    identity["gpu"] = common.gpu_name(settings.device)
    identity["dirty"] = start_tree["dirty"]
    identity["tree_problems"] = start_tree["problems"]
    identity["allow_dirty"] = settings.allow_dirty
    identity["code_blobs"] = start_tree["code_blobs"]
    identity["src_tree_id"] = start_tree["src_tree_id"]
    identity["tree_end"] = end_tree
    identity["sealed_pp_assertion"] = sealed
    identity["weights_sha256"] = weights
    identity["inputs_sha256_lf"] = hashes
    identity["inputs_sha256"] = {
        "manifest": common.sha256_file(settings.manifest),
        "eval_images": common.sha256_file(settings.data_dir / "eval_images.txt"),
    }
    identity["settings"] = {
        "conf_ap": CONF_AP,
        "nms_iou": NMS_IOU,
        "seed": SEED,
        "bootstrap_b": BOOTSTRAP_B,
        "period_target_px": PERIOD_TARGET_PX,
        "top_mask_rows": TOP_MASK_ROWS,
        "scales": list(SCALES),
    }
    identity["finished_at"] = common.local_now()
    return identity


def _write_fold(out_root: Path, fold: int, fold_run: FoldRun) -> None:
    fold_dir = out_root / f"fold{fold}"
    common.write_csv(
        fold_dir / "predictions.csv", PREDICTION_COLUMNS, fold_run.prediction_rows
    )
    common.write_csv(fold_dir / "objects.csv", OBJECT_COLUMNS, fold_run.object_rows)


# -------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None) -> Settings:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--device", default="0", help="'0' (GPU) or 'cpu'")
    parser.add_argument("--smoke", action="store_true", help="8 images, fold 0")
    parser.add_argument(
        "--limit", type=int, default=None, help="images (not judgeable)"
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--run-root", type=Path, default=common.RUN_ROOT)
    args = parser.parse_args(argv)
    try:
        folds = tuple(int(part) for part in args.folds.split(",") if part.strip())
    except ValueError:
        parser.error(f"--folds must be integers: {args.folds!r}")
    if args.smoke:
        folds = (SMOKE_FOLD,)
    default_root = (
        m33_inputs.R1_SMOKE_OUT_ROOT if args.smoke else m33_inputs.R1_OUT_ROOT
    )
    return Settings(
        out_root=args.out_root or default_root,
        folds=folds,
        device=args.device,
        smoke=args.smoke,
        limit=args.limit,
        allow_dirty=args.allow_dirty or args.smoke,
        run_root=args.run_root,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry; returns the exit code (see the module docstring)."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    started = time.time()
    settings = parse_args(argv)
    n_images = 0
    try:
        status, n_images = run(settings, started)
    except inputs.InputError as error:
        logger.error("input error: %s", error)
        status = m33_inputs.STATUS_INPUT_ERROR
    except Exception:
        logger.exception("internal error")
        status = "error"
    line = {
        "status": status,
        "out_dir": str(settings.out_root),
        "n_images": n_images,
        "elapsed_s": round(time.time() - started, 1),
    }
    print(json.dumps(line, ensure_ascii=False), flush=True)
    return EXIT_CODES[status]


if __name__ == "__main__":
    sys.exit(main())
