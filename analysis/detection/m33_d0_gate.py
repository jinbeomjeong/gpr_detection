"""M3-3 D0 development gate of the boundary post-processing PP (sets S1-S5).

Spec: ``docs/experiments/m33_pp_spec.md`` v0.2.3 §4 ([PP]), physics review
``docs/processing/m33_x2_pp_physics_review.md``, plan
``docs/experiments/m33_cross_domain_plan.md`` v0.2.6 §3-§4. PP itself is
``gpr_detection.pipeline.apply_pp`` / ``pp_decision``.

Data: AIHub validation folds (N2-a' fold k model on its own crops) and the
gprMax models M01-M20 only. Every path the runner opens goes through
:class:`InputGuard` (allow list [PP] §4-1, forbidden list §4-2; a forbidden
or unlisted path stops the run with exit 2). The AIHub test seal is asserted
with ``analysis/data/m33_assert_no_aihub_test.py`` and the M3-2 split checks.

Conditions ([PP] §4-3), one inference each; PP sets S0-S5 are applied to the
same predictions afterwards:

* A0 crop as is (``y_bot`` 256, ``y_z`` 0); A1 S3 ``pad_p0`` (``y_bot`` 256 -
  ``h_pad``); A2 (report) A1 + 8 px flat top band, bottom cut to keep 256
  (``y_z`` 8); A3 the A0 predictions with ``y_z`` = surface-band row (row of
  largest row energy in rows 0-63).
* Y0 the 20 models through the r0 C' path (5 folds, merged, ``select_zero``);
  Y1 (report) the same records cut at apex + {0.5, 1, 2, 3} ``P_line``.

Gates per set ([PP] v0.2.3 §4-4, worst of A0, A1, A3, unweighted fold mean):
G1 total hyperbola recall loss <= 0.5 %p (GT hit by S0 and missed by S_j,
gains reported apart), G2 total band coverage loss <= 1.0 %p (per-GT
decrease of the covered length), G3 Y0 apex loss = 0 with ``H(S0) >= 1``,
G4 new-rule increment ``processed(S_j) - processed(S_(j-1))`` > 0. Selection:
the largest passing set of S1 ⊂ ... ⊂ S5, stopping at the first failure; an
S1 failure is ``s1_defect`` (G1-G3) or ``s1_no_effect`` (G4 only).

Outputs ([PP] §4-5): ``fold<k>/predictions.csv`` (crop only; synthetic boxes
in ``synthetic_boxes.csv``), ``fold<k>/objects.csv``,
``fold<k>/synthetic_boxes.csv`` (every Y0 / Y1 merged box before PP with its
action and ``y2`` per set), ``synthetic_apex.csv`` and ``d0_gate.json``.

Blind output: values go to ``<out>/d0_gate.json`` and the CSV files only.
Stdout carries stages, counts, elapsed time and paths; the last line is
``{"status", "out_dir", "n_conditions", "n_images", "elapsed_s"}`` with
``status`` ``ok``, ``s1_defect``, ``s1_no_effect``, ``not_judgeable``,
``input_error`` or ``error``. Exit codes: 0 done, 1 S1 failed (either
reason) or not judgeable, 2 input refused (forbidden or unlisted path, sha
mismatch, dirty tree, test crops) or the tree changed during the run, 3
internal error (also a negative G4 increment).

Judgment scope (verif audit 2026-11-15): only a run over exactly folds 0-4,
every crop and all 20 models on a tree that was clean at the start can be
judged. Any other run (``--smoke``, ``--folds`` subset, ``--limit``,
``--synthetic-limit``, ``--allow-dirty`` on a dirty tree) still writes its
gates, but ``d0_gate.json`` ``judgment.judgeable`` is false and the status
is ``not_judgeable``. HEAD, the tracked-tree state and the code blobs are
measured at the start and again before the outputs are written; a
difference is recorded (``judgment.tree_changes``) and the status is
``input_error``. Score and coordinate columns are written at full precision.

Sealed PP ([PP] v0.2.4 A-1, the D0 re-run): before any inference the ``PPConfig``
hashes of S0-S5 must equal the C0b record and the ``src`` tree id and the
``pipeline.py`` blob must equal C0a (53c4e16); otherwise exit 2. The
result is kept in ``run_identity.sealed_pp_assertion``. An existing output
folder is refused (exit 2) unless ``--overwrite`` is given; the C0b result
folder ``runs/eval/m33_d0`` (also the default) and any folder inside it are
refused even with it.
"""

import argparse
import csv
import fnmatch
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common
import m32_aihub_perturb as perturb

from gpr_detection.detect.band_metrics import (
    EMPTY_COVERAGE,
    BandCoverage,
    band_coverage,
    gt_covered_lengths,
)
from gpr_detection.io.classes import DetectionClass
from gpr_detection.io.radargram import GridBox, Radargram
from gpr_detection.io.synthetic import load_synthetic, load_synthetic_truth
from gpr_detection.pipeline import (
    DEFAULT_MERGE_CONFIG,
    PP_SETS,
    ContentRows,
    LineBox,
    PPAction,
    PPConfig,
    PPSetId,
    apply_pp,
    content_rows_for,
    image_box_to_grid_box,
    is_removed_inside_content,
    kept_after_pp,
    line_boxes_to_grid,
    line_zero,
    merge_line_boxes,
    pp_category,
    pp_decision,
    pp_sets_sha256,
    tile_output_line_boxes,
)
from gpr_detection.preprocess.to_image import DetectionImage, make_detection_image

REPO = common.REPO
SCHEMA = "m33_d0_gate/1"
DEFAULT_OUT_ROOT = REPO / "runs" / "eval" / "m33_d0"
SMOKE_OUT_ROOT = REPO / "runs" / "eval" / "m33_d0_smoke"
C0B_OUT_ROOT = REPO / "runs" / "eval" / "m33_d0"
"""C0b D0 results (sealed by 0f11c34): never an output folder, even with
``--overwrite``. It is also ``DEFAULT_OUT_ROOT``, so a run without
``--out-root`` stops with exit 2 ([PP] A-1, user decision 2026-11-15)."""
RUN_PREFIX = perturb.RUN_PREFIX
RUN_ROOT = REPO / "runs" / "detect"
LABEL_DIR = REPO / "data" / "aihub_yolo_v1" / "labels"
MANIFEST = REPO / "data" / "splits" / "aihub_yolo_v1.csv"
PAD_HEIGHTS_CSV = REPO / "analysis" / "processing" / "m32_content_height_lines.csv"
SYNTHETIC_ROOT = REPO / "data" / "synthetic_gprmax"
D1_R4_VERDICT = REPO / "analysis" / "detection" / "m31_d1_r4_verdict.json"
NO_TEST_SCRIPT = REPO / "analysis" / "data" / "m33_assert_no_aihub_test.py"
GPR_DATA_ROOT = Path(r"C:\workspace\GPR_Data")

SEED = perturb.SEED
PERIOD_AIHUB_PX = perturb.PERIOD_AIHUB_PX
CANVAS_PX = common.CANVAS_PX
PAD_GRAY = perturb.PAD_GRAY
TOP_BAND_PX = 8
"""A2: ``h_top`` = 2 ``P_A`` rounded to 8 px ([PP] §4-3)."""
SURFACE_SEARCH_ROWS = 64
"""A3: the surface-band row is searched in rows 0-63 (top quarter)."""
SYNTHETIC_MODELS = tuple(f"M{index:02d}" for index in range(1, 21))
Y1_CUT_PERIODS = (0.5, 1.0, 2.0, 3.0)
CONF_AP = perturb.CONF_AP
CONF_RECALL = common.CONF_RECALL
CONF_LINE = 0.25
"""Line inference threshold of the M3-2 S5 path (``m32_recipe_eval``)."""
NMS_IOU = perturb.NMS_IOU
BATCH_SIZE = 64
MATCH_TRACE_MARGIN = 3
MATCH_SAMPLE_PERIODS = 0.5

GATE_CONDITIONS = ("A0", "A1", "A3")
REPORT_CONDITIONS = ("A2",)
CROP_CONDITIONS = ("A0", "A1", "A2", "A3")
INFERRED_IMAGES = ("A0", "A1", "A2")
"""A3 reuses the A0 predictions ([PP] §4-3)."""
G4_CROP_CONDITIONS = ("A1", "A3")
GATE_SETS = (PPSetId.S1, PPSetId.S2, PPSetId.S3, PPSetId.S4, PPSetId.S5)
G1_MAX_LOSS_PP = 0.5
G2_MAX_LOSS_PP = 1.0
HYPERBOLA_NAME = "pipe_hyperbola"

PAD_P0 = perturb.Perturbation("pad_p0", pad=perturb.PadMimic.FLAT)
EXPECTED_FOLDS = (0, 1, 2, 3, 4)
"""Folds a judgeable run must cover: the gates are means over all five."""
STATUS_NOT_JUDGEABLE = "not_judgeable"
STATUS_INPUT_ERROR = "input_error"
EXIT_CODES = {
    "ok": 0,
    "s1_defect": 1,
    "s1_no_effect": 1,
    STATUS_NOT_JUDGEABLE: 1,
    STATUS_INPUT_ERROR: 2,
    "error": 3,
}
"""Final-line ``status`` -> exit code ([PP] §4-5); both S1 stops exit 1.

``not_judgeable`` (smoke, partial folds, crop or model limits, dirty tree)
also exits 1, so a partial run never ends with 0.
"""
SMOKE_CROPS = 8
SMOKE_MODELS = 2

C0B_PP_SET_SHA256 = {
    PPSetId.S0: "fc9f0c8eab77e84f578f38e12fd4b7d401b6e18be3ec06187b00336acbfd2e4d",
    PPSetId.S1: "78de0a20390e71922ebc0335030696b0519207525a9138a9c27169424bf7640a",
    PPSetId.S2: "184c07e44c0c24742d6aa5ddc0e87901329b9543e96bb3ebf0a6886df7bb9d63",
    PPSetId.S3: "eb2796ddfaa9611e5030342c4ed66b00c91ff2ed6a6698ee5979b9afe080602f",
    PPSetId.S4: "af29b47126e790abb05aa6a5644c74f316be6c810457e933004eed558b7a35e3",
    PPSetId.S5: "b87bbc145e78061b4e8ada82fba93d7d481c065374dfaf9ecf09d72204a64a58",
}
"""``PPConfig.sha256()`` per set in the C0b ``d0_gate.json`` (0f11c34)."""
C0B_PP_SETS_SHA256 = "1f2b0cf2379815ff5dc73571fb483848b16838e37b856c362c9f045d6fc9ec4d"
"""``pp_sets_sha256()`` of S0-S5 in the C0b ``d0_gate.json``."""
C0B_SELECTED_SET = PPSetId.S3
C0A_SRC_TREE_ID = "164f09909ad51c1a8d26545cadb0417b1da3acc6"
"""``git rev-parse 53c4e16:src`` (C0a)."""
C0A_PIPELINE_BLOB = "dd5e7bcba4397efd732e7bf3ca1c5c92fb2404a1"
"""``git rev-parse 53c4e16:src/gpr_detection/pipeline.py`` (C0a)."""
PIPELINE_PATH = "src/gpr_detection/pipeline.py"
"""Repo-relative key of the PP implementation in ``code_blobs``."""

PROCESSED = tuple(action for action in PPAction if action.is_processed)
"""Actions that count as processed (``removed_*``, ``clipped_b``)."""

PREDICTIONS_HEADER = (
    "fold",
    "condition",
    "image",
    "gain",
    "x1",
    "y1",
    "x2",
    "y2",
    "score",
    "class",
    "category",
    *(f"pp_action_{key}" for key in GATE_SETS),
)
"""``fold<k>/predictions.csv``: crop only; synthetic boxes in synthetic_boxes.csv."""
OBJECTS_HEADER = (
    "fold",
    "condition",
    "image",
    "gain",
    "class",
    "gt_index",
    *(f"hit_{key}" for key in PPSetId),
)
SYNTHETIC_HEADER = (
    "fold",
    "condition",
    "model",
    "apex_trace",
    "apex_sample",
    "n_samples_kept",
    "period_samples",
    "period_source",
    "p_over_p_y0",
    "zero_px",
    "y_bot_px",
    "y_bot_minus_apex_over_p_ell",
    "n_merged_boxes",
    *(f"hit_{key}" for key in PPSetId),
    *(f"n_processed_{key}" for key in GATE_SETS),
    "n_exposed_boxes_S0",
    *(f"n_{action}_{key}" for key in GATE_SETS for action in PROCESSED),
)
SYNTHETIC_BOXES_HEADER = (
    "fold",
    "condition",
    "model",
    "x1",
    "y1",
    "x2",
    "y2",
    "i0",
    "i1",
    "k0",
    "k1",
    "score",
    "class",
    "category",
    "tile_truncated",
    "n_tiles_seen",
    *(f"pp_action_{key}" for key in GATE_SETS),
    *(f"y2_after_{key}" for key in GATE_SETS),
)
"""``fold<k>/synthetic_boxes.csv``: Y0 and Y1 merged boxes before PP ([PP] §4-5)."""

logger = logging.getLogger("m33_d0_gate")


def exact(value: float) -> float:
    """A csv score, coordinate or row value at full precision.

    ``csv`` writes a Python float as its ``repr``, which reads back to the
    same double, so conf > 0.25 and the PP rows can be recounted from the
    files (rounding wrote a 0.2500x score as 0.25, verif audit 2026-11-15).
    ``float`` also turns numpy scalars into Python floats.
    """
    return float(value)


# ------------------------------------------------------------ input guard


class ForbiddenPathError(common.InputError):
    """The runner tried to open a path of the forbidden list ([PP] §4-2)."""


FORBIDDEN_PATTERNS = (
    "data/morocco_eval_v1/*",
    "data/processing/*",
    "runs/eval/m32_recipe*",
    "runs/eval/m32_aihub_perturb*",
    "*utilities_groundtruth*",
    "docs/processing/fourtu_*",
    "*test.yaml",
)
"""Repo-relative patterns (``*`` crosses folders); GPR_Data is checked apart."""


def _allowed_patterns() -> tuple[str, ...]:
    """[PP] §4-1 inputs plus the code that runs (repo-relative, lower case)."""
    runs = f"runs/detect/{RUN_PREFIX}[0-4]"
    synthetic = [
        f"data/synthetic_gprmax/{model}{suffix}"
        for model in SYNTHETIC_MODELS
        for suffix in (".npy", "_bg.npy", ".json")
    ]
    return (
        f"{runs}/weights/last.pt",
        f"{runs}/eval/last_crop/eval_images.txt",
        "data/aihub_yolo_v1/images/*",
        "data/aihub_yolo_v1/labels/*",
        "data/splits/aihub_yolo_v1.csv",
        "analysis/processing/m32_content_height_lines.csv",
        "analysis/detection/m31_d1_r4_verdict.json",
        *synthetic,
        "src/gpr_detection/*",
        "analysis/detection/m31_d1_common.py",
        "analysis/detection/m32_aihub_perturb.py",
        "analysis/detection/m33_d0_gate.py",
        "analysis/data/m33_assert_no_aihub_test.py",
    )


@dataclass
class InputGuard:
    """Asserts every opened path against the allow and forbidden lists.

    Attributes:
        repo: Repository root; patterns are relative to it.
        opened: Repo-relative paths checked so far (``run_identity``).
    """

    repo: Path = REPO
    opened: list[str] = field(default_factory=list)

    def check(self, path: Path) -> Path:
        """Return ``path`` when allowed.

        Raises:
            ForbiddenPathError: For a path of the forbidden list (exit 2).
            common.InputError: For a path outside the allow list (exit 2).
        """
        resolved = Path(path).resolve()
        if _is_inside(resolved, GPR_DATA_ROOT):
            raise ForbiddenPathError(f"forbidden input (raw datasets): {resolved}")
        if not _is_inside(resolved, self.repo.resolve()):
            raise common.InputError(f"input outside the allow list: {resolved}")
        relative = resolved.relative_to(self.repo.resolve()).as_posix().lower()
        if any(fnmatch.fnmatchcase(relative, p) for p in FORBIDDEN_PATTERNS):
            raise ForbiddenPathError(f"forbidden input: {relative}")
        if not any(
            fnmatch.fnmatchcase(relative, p.lower()) for p in _allowed_patterns()
        ):
            raise common.InputError(f"input outside the allow list: {relative}")
        if relative not in self.opened:
            self.opened.append(relative)
        return resolved


def _is_inside(path: Path, root: Path) -> bool:
    return path == root or root.resolve() in path.parents


# ------------------------------------------------------------ crop views


@dataclass(frozen=True)
class CropView:
    """One crop under one condition: what PP and the scoring see.

    Attributes:
        condition: ``A0``-``A3``.
        image_key: Inferred image whose predictions it uses (A3 -> A0).
        content: ``y_bot``, ``P_A``, ``y_z``.
        gt_boxes: Kept GT (centre above ``y_bot``), ``y2`` cut to ``y_bot``.
        gt_classes: Their classes.
        gt_index: Their label line numbers.
        ignore_boxes: GT with the centre in the padding or cut off (S3 rule).
    """

    condition: str
    image_key: str
    content: ContentRows
    gt_boxes: np.ndarray
    gt_classes: np.ndarray
    gt_index: np.ndarray
    ignore_boxes: np.ndarray


def top_band_image(image: np.ndarray, top_px: int = TOP_BAND_PX) -> np.ndarray:
    """A2: a flat ``top_px`` band on top, the bottom cut so the height stays."""
    shifted = np.full_like(image, PAD_GRAY)
    shifted[top_px:] = image[: image.shape[0] - top_px]
    return shifted


def surface_band_row(image: np.ndarray, search_rows: int = SURFACE_SEARCH_ROWS) -> int:
    """A3 ``y_z``: row of largest ``mean_x (q - q_mean)^2`` in rows 0-63.

    ``q_mean`` is the mean of the whole crop; ties take the smaller row.
    """
    values = image.astype(np.float64)
    energy = ((values[:search_rows] - values.mean()) ** 2).mean(axis=1)
    return int(np.argmax(energy))


def _split_and_cut(
    boxes: np.ndarray, classes: np.ndarray, index: np.ndarray, y_bot: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kept GT and ignore boxes, decided on the uncut box centre ([PP] §4-3).

    (ii) a box whose centre lies at or below ``y_bot`` is ignored; its IoU /
    IoA rows are ``[y1, min(y2, 256)]``. (iii) only kept boxes get ``y2`` cut
    to ``y_bot``. ``boxes`` may run past the canvas (A2 moved them down).
    """
    centre = (boxes[:, 1] + boxes[:, 3]) / 2.0
    kept = centre < y_bot
    kept_boxes = boxes[kept].copy()
    kept_boxes[:, 3] = np.minimum(kept_boxes[:, 3], y_bot)
    ignore = boxes[~kept].reshape(-1, 4).copy()
    ignore[:, 3] = np.minimum(ignore[:, 3], CANVAS_PX)
    return kept_boxes, classes[kept], index[kept], ignore


def crop_views(
    sample: perturb.Sample, images: Mapping[str, np.ndarray]
) -> list[CropView]:
    """The four condition views of one crop ([PP] §4-3).

    A2: (i) the boxes move down by ``h_top`` and are not cut, then
    :func:`_split_and_cut` decides on the moved centre, so for ``h_pad >= 8``
    the kept set equals that of A1 with rows + 8.
    """
    boxes = np.asarray(sample.boxes, dtype=float).reshape(-1, 4)
    pad = sample.pad_height_px
    a1_bot = float(CANVAS_PX - pad)
    shifted = boxes.copy()
    shifted[:, [1, 3]] += TOP_BAND_PX
    a2_bot = float(CANVAS_PX - max(pad - TOP_BAND_PX, 0))
    surface = float(surface_band_row(images["A0"]))
    specs = (
        ("A0", "A0", boxes, float(CANVAS_PX), 0.0),
        ("A1", "A1", boxes, a1_bot, 0.0),
        ("A2", "A2", shifted, a2_bot, float(TOP_BAND_PX)),
        ("A3", "A0", boxes, float(CANVAS_PX), surface),
    )
    views = []
    for condition, image_key, gt, y_bot, zero in specs:
        kept, classes, index, ignore = _split_and_cut(
            gt, sample.classes, sample.gt_index, y_bot
        )
        content = ContentRows(y_bot, PERIOD_AIHUB_PX, zero)
        views.append(
            CropView(condition, image_key, content, kept, classes, index, ignore)
        )
    return views


def condition_images(sample: perturb.Sample) -> dict[str, np.ndarray]:
    """The three inferred images A0, A1 (``pad_p0``) and A2 (A1 + top band)."""
    a1 = perturb.perturb_image(sample.image, PAD_P0, sample.pad_height_px)
    return {"A0": sample.image, "A1": a1, "A2": top_band_image(a1)}


# --------------------------------------------------------- crop PP + score


@dataclass(frozen=True)
class Predictions:
    """Predictions of one image: ``[n, 4]`` xyxy boxes, scores, classes."""

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray

    def subset(self, keep: np.ndarray) -> "Predictions":
        return Predictions(self.boxes[keep], self.scores[keep], self.classes[keep])


@dataclass(frozen=True)
class CropPP:
    """PP of every prediction of one crop view under one set.

    Attributes:
        actions: ``pp_action`` per prediction.
        kept: Predictions after PP (``y2`` of PP-b clips replaced).
    """

    actions: tuple[PPAction, ...]
    kept: Predictions


def pp_on_predictions(
    predictions: Predictions, content: ContentRows, config: PPConfig
) -> CropPP:
    """Apply one PP set to one crop's predictions (boxes are independent)."""
    actions = []
    boxes = predictions.boxes.copy()
    for row, (box, score, class_id) in enumerate(
        zip(predictions.boxes, predictions.scores, predictions.classes, strict=True)
    ):
        decision = pp_decision(
            int(class_id), float(score), float(box[1]), float(box[3]), content, config
        )
        actions.append(decision.action)
        boxes[row, 3] = decision.y2_px
    keep = np.array([not action.is_removed for action in actions], dtype=bool)
    clipped = Predictions(boxes, predictions.scores, predictions.classes)
    return CropPP(tuple(actions), clipped.subset(keep.reshape(-1)))


@dataclass
class SetScores:
    """Scores of one (condition, set) on one fold.

    Attributes:
        overall: Per-class AP50 and recall at conf > 0.25 (D1 code).
        by_gain: The same per gain copy.
        band: ``pipe_band`` horizontal coverage (conf > 0.25).
        processed: Action -> count over all predictions.
        processed_confident: Action -> count over conf > 0.25 predictions.
        pp_a_inside: PP-a removals with the top inside the content (all, conf).
    """

    overall: common.ScoreAccumulator = field(default_factory=common.ScoreAccumulator)
    by_gain: dict[str, common.ScoreAccumulator] = field(
        default_factory=lambda: {g: common.ScoreAccumulator() for g in perturb.GAINS}
    )
    band: BandCoverage = EMPTY_COVERAGE
    processed: Counter = field(default_factory=Counter)
    processed_confident: Counter = field(default_factory=Counter)
    pp_a_inside: list[int] = field(default_factory=lambda: [0, 0])
    _classes: dict | None = None

    def class_results(self) -> dict[str, dict]:
        """``overall.class_results()``, computed once (AP sorts every score)."""
        if self._classes is None:
            self._classes = self.overall.class_results()
        return self._classes


def _band_part(boxes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return boxes[classes == DetectionClass.PIPE_BAND].reshape(-1, 4)


def score_view(
    scores: SetScores,
    view: CropView,
    original: Predictions,
    crop_pp: CropPP,
    gain: str,
) -> np.ndarray:
    """Add one crop view under one set; return the conf > 0.25 hit per GT."""
    kept = crop_pp.kept
    hit_score = scores.overall.add_image(
        kept.boxes, kept.scores, kept.classes, view.gt_boxes, view.gt_classes
    )
    scores.by_gain[gain].add_image(
        kept.boxes, kept.scores, kept.classes, view.gt_boxes, view.gt_classes
    )
    confident = kept.scores > CONF_RECALL
    scores.band = scores.band + band_coverage(
        _band_part(view.gt_boxes, view.gt_classes),
        _band_part(kept.boxes[confident], kept.classes[confident]),
    )
    for action, box, score in zip(
        crop_pp.actions, original.boxes, original.scores, strict=True
    ):
        is_confident = score > CONF_RECALL
        scores.processed[action.value] += 1
        scores.processed_confident[action.value] += int(is_confident)
        if is_removed_inside_content(float(box[1]), action, view.content):
            scores.pp_a_inside[0] += 1
            scores.pp_a_inside[1] += int(is_confident)
    return hit_score >= 0


# ------------------------------------------------------- synthetic (Y0/Y1)


PERIOD_MEASURED = "measured"
PERIOD_Y0_FIXED = "Y0_fixed"
"""``period_source`` labels of ``synthetic_apex.csv`` ([PP] §4-3 Y1)."""


@dataclass(frozen=True)
class SyntheticLine:
    """One synthetic line image (Y0 or a Y1 cut) and its truth apex in L0.

    Attributes:
        condition: ``Y0`` or ``Y1_<f>P``.
        model: ``M01``-``M20``.
        image: Line L1 image; a Y1 cut uses the Y0 ``P_line``.
        content: ``y_bot``, ``P_ell``, ``y_z`` of the (cut) record.
        apex_trace: Truth apex trace (json ``apex.trace_index``).
        apex_sample: Truth apex sample ``k_apex`` (json ``envelope_peak_ns``).
        n_samples_kept: Record length after the cut.
        period_source: ``measured`` (Y0) or ``Y0_fixed`` (Y1).
        p_over_p_y0: Period re-measured on the cut record / Y0 period
            (report only); ``None`` when it could not be measured.
    """

    condition: str
    model: str
    image: DetectionImage
    content: ContentRows
    apex_trace: float
    apex_sample: float
    n_samples_kept: int
    period_source: str = PERIOD_MEASURED
    p_over_p_y0: float | None = 1.0

    @property
    def period_samples(self) -> float:
        """``P_line`` used for the image and the +-0.5 P match margin."""
        return self.image.period_samples

    @property
    def apex_bottom_periods(self) -> float:
        """``(y_bot - apex) / P_ell`` in line L1 px."""
        apex_px = self.apex_sample / self.image.grid_map.samples_per_px
        return (self.content.y_bot_px - apex_px) / self.content.period_px


def apex_hit(
    grid_boxes: Iterable[tuple[int, GridBox]],
    apex_trace: float,
    apex_sample: float,
    period_samples: float,
) -> bool:
    """M3-2 match for one apex: a ``pipe_hyperbola`` box contains it.

    Box grown by +-3 traces and +-0.5 ``P_line``. With a single apex the
    M3-2 one-to-one rule assigns every containing box to it, so "hit" equals
    "some box contains it" (checked against ``match_one_to_one`` in the tests).
    """
    return any(
        class_id == DetectionClass.PIPE_HYPERBOLA
        and _box_contains_apex(box, apex_trace, apex_sample, period_samples)
        for class_id, box in grid_boxes
    )


def _box_contains_apex(
    box: GridBox, apex_trace: float, apex_sample: float, period_samples: float
) -> bool:
    """Apex inside the box grown by +-3 traces and +-0.5 ``P_line`` samples."""
    sample_margin = MATCH_SAMPLE_PERIODS * period_samples
    inside_traces = (
        box.i0 - MATCH_TRACE_MARGIN <= apex_trace <= box.i1 + MATCH_TRACE_MARGIN
    )
    inside_samples = box.k0 - sample_margin <= apex_sample <= box.k1 + sample_margin
    return inside_traces and inside_samples


def _remeasured_period_ratio(cut: Radargram, y0_period: float) -> float | None:
    """Report only: the period measured on the cut record over the Y0 period."""
    try:
        return make_detection_image(cut).period_samples / y0_period
    except ValueError:
        return None


def y1_condition_name(periods: float) -> str:
    return f"Y1_{periods:g}P"


def y1_samples_kept(apex_sample: float, periods: float, period: float, n: int) -> int:
    """Samples ``0 .. floor(k_apex + f P_line)`` inclusive, at most the record."""
    return min(n, math.floor(apex_sample + periods * period) + 1)


def synthetic_lines(
    model: str, guard: InputGuard, root: Path = SYNTHETIC_ROOT
) -> list[SyntheticLine]:
    """Y0 and the four Y1 cuts of one model (truth apex from the json).

    Y1 keeps the Y0 r0 ``period_samples`` (a short record re-measures it up
    to 3.8x off, [PP] §4-3); zero, content range, gain and image come from
    the cut record.
    """
    for suffix in (".npy", ".json"):
        guard.check(root / f"{model}{suffix}")
    radargram = load_synthetic(model, root=root)
    truth = load_synthetic_truth(model, root=root)
    dt_ns = radargram.dt_ns
    assert dt_ns is not None
    origin_ns = float(radargram.metadata.get("time_of_sample0_ns", 0.0))
    apex_trace = float(truth.apex_trace_index)
    apex_sample = (truth.apex_envelope_peak_ns - origin_ns) / dt_ns
    y0_image = make_detection_image(radargram)
    lines = [
        SyntheticLine(
            "Y0",
            model,
            y0_image,
            content_rows_for(y0_image, line_zero(radargram)),
            apex_trace,
            apex_sample,
            radargram.n_samples,
        )
    ]
    period = y0_image.period_samples
    for periods in Y1_CUT_PERIODS:
        n_keep = y1_samples_kept(apex_sample, periods, period, radargram.n_samples)
        cut = replace(radargram, data=np.ascontiguousarray(radargram.data[:n_keep]))
        image = make_detection_image(cut, period_samples_override=period)
        lines.append(
            SyntheticLine(
                y1_condition_name(periods),
                model,
                image,
                content_rows_for(image, line_zero(cut)),
                apex_trace,
                apex_sample,
                n_keep,
                PERIOD_Y0_FIXED,
                _remeasured_period_ratio(cut, period),
            )
        )
    return lines


def synthetic_row(
    fold: int, line: SyntheticLine, merged: Sequence[LineBox]
) -> tuple[list, dict[PPSetId, Counter]]:
    """Hits per set and processed counts of one line on one fold."""
    hits = {}
    processed: dict[PPSetId, Counter] = {}
    for key in PPSetId:
        outcomes = apply_pp(merged, line.content, PP_SETS[key])
        pairs = line_boxes_to_grid(kept_after_pp(outcomes), line.image)
        grid = [(box.class_id, grid_box) for box, grid_box in pairs]
        hits[key] = apex_hit(
            grid, line.apex_trace, line.apex_sample, line.period_samples
        )
        processed[key] = Counter(outcome.action.value for outcome in outcomes)
    zero_px = line.content.zero_px
    row = [
        fold,
        line.condition,
        line.model,
        line.apex_trace,
        exact(line.apex_sample),
        line.n_samples_kept,
        exact(line.period_samples),
        line.period_source,
        "" if line.p_over_p_y0 is None else exact(line.p_over_p_y0),
        "" if zero_px is None else exact(zero_px),
        exact(line.content.y_bot_px),
        exact(line.apex_bottom_periods),
        len(merged),
        *(int(hits[key]) for key in PPSetId),
        *(_n_processed(processed[key]) for key in GATE_SETS),
        exposed_apex_boxes(line, merged) if hits[PPSetId.S0] else 0,
        *(processed[key][action.value] for key in GATE_SETS for action in PROCESSED),
    ]
    return row, processed


def synthetic_box_rows(
    fold: int, line: SyntheticLine, merged: Sequence[LineBox]
) -> list[list]:
    """``synthetic_boxes.csv`` rows: every merged box before PP, one per row.

    L0 columns are the outward-rounded box (empty when it misses the grid).
    A PP-b clip only moves ``y2`` to ``y_bot``, the grid bottom, so the L0
    box of a kept box is the same for every set; ``y2_after_<set>`` is empty
    for a removed box.
    """
    outcomes = {key: apply_pp(merged, line.content, PP_SETS[key]) for key in GATE_SETS}
    rows = []
    for index, box in enumerate(merged):
        grid = image_box_to_grid_box(box.box, line.image.grid_map)
        category = pp_category(box.box.y1, box.box.y2, line.content)
        after = [outcomes[key][index].output for key in GATE_SETS]
        rows.append(
            [
                fold,
                line.condition,
                line.model,
                *(exact(v) for v in box.box.xyxy),
                *(
                    ("", "", "", "")
                    if grid is None
                    else (grid.i0, grid.i1, grid.k0, grid.k1)
                ),
                exact(box.score),
                common.CLASS_NAMES[box.class_id],
                "" if category is None else category.value,
                int(box.tile_truncated),
                box.n_tiles_seen,
                *(outcomes[key][index].action.value for key in GATE_SETS),
                *("" if out is None else exact(out.box.y2) for out in after),
            ]
        )
    return rows


def exposed_apex_boxes(line: SyntheticLine, merged: Sequence[LineBox]) -> int:
    """G3 exposure: S0 apex boxes (``pipe_hyperbola``, containing the apex)
    whose original box has a PP category a, b or d ([PP] §4-7 4).
    """
    count = 0
    for box, grid_box in line_boxes_to_grid(merged, line.image):
        if box.class_id != DetectionClass.PIPE_HYPERBOLA:
            continue
        if not _box_contains_apex(
            grid_box, line.apex_trace, line.apex_sample, line.period_samples
        ):
            continue
        if pp_category(box.box.y1, box.box.y2, line.content) is not None:
            count += 1
    return count


def _n_processed(counts: Mapping[str, int]) -> int:
    return sum(n for action, n in counts.items() if PPAction(action).is_processed)


# ----------------------------------------------------------- gates


@dataclass(frozen=True)
class GateResult:
    """G1-G4 of one set ([PP] v0.2.3 §4-4).

    Attributes:
        g1_loss_pp: Worst total hyperbola recall loss over A0, A1, A3 (%p):
            GT hit by S0 and missed by S_j over all GT, gains not netted.
            ``None`` when a condition x fold cell has no GT (fails).
        g2_loss_pp: Worst total band coverage loss (%p): per GT
            ``max(0, covered S0 - covered S_j)`` over the GT length.
        g3_loss: Y0 apex hits ``H(S0) - H(S_j)``.
        g4_increment: Processed boxes of S_j minus those of S_(j-1).
        g3_h_s0: ``H(S0)``; G3 is only defined for ``H(S0) >= 1``.
    """

    g1_loss_pp: float | None
    g2_loss_pp: float | None
    g3_loss: int
    g4_increment: int
    g3_h_s0: int = 1

    @property
    def checks(self) -> dict[str, bool]:
        return {
            "G1": self.g1_loss_pp is not None and self.g1_loss_pp <= G1_MAX_LOSS_PP,
            # No band GT means no measurement: fail rather than pass.
            "G2": self.g2_loss_pp is not None and self.g2_loss_pp <= G2_MAX_LOSS_PP,
            "G3": self.g3_h_s0 >= 1 and self.g3_loss == 0,
            "G4": self.g4_increment > 0,
        }

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    @property
    def tp_gates_failed(self) -> bool:
        """G1, G2 or G3 failed (an S1 failure of this kind is ``s1_defect``)."""
        checks = self.checks
        return not (checks["G1"] and checks["G2"] and checks["G3"])


@dataclass(frozen=True)
class LossCell:
    """Total loss and gain of one condition x fold (G1 counts, G2 lengths).

    Attributes:
        lost: GT hit by S0 and missed by S_j (or covered length lost).
        gained: GT missed by S0 and hit by S_j (or covered length gained).
        total: GT count (or GT length); 0 means the cell is empty.
    """

    lost: float
    gained: float
    total: float

    @property
    def loss_pp(self) -> float | None:
        return None if self.total <= 0 else 100.0 * self.lost / self.total

    @property
    def gain_pp(self) -> float | None:
        return None if self.total <= 0 else 100.0 * self.gained / self.total


def worst_fold_mean_pp(
    cells: Mapping[tuple[str, int], LossCell],
    folds: Sequence[int],
    conditions: Sequence[str] = GATE_CONDITIONS,
) -> tuple[float | None, dict[str, float | None]]:
    """Worst over ``conditions`` of the unweighted fold mean of ``loss_pp``.

    A condition whose cells are not all defined has ``None``, and then the
    worst is ``None`` (the gate fails, [PP] §4-7 3).
    """
    by_condition: dict[str, float | None] = {}
    for name in conditions:
        values = [
            cells[(name, f)].loss_pp if (name, f) in cells else None for f in folds
        ]
        present = [value for value in values if value is not None]
        defined = bool(folds) and len(present) == len(values)
        by_condition[name] = float(np.mean(present)) if defined else None
    values = list(by_condition.values())
    if not values or any(value is None for value in values):
        return None, by_condition
    return max(value for value in values if value is not None), by_condition


def mean_fold_loss_pp(
    reference: Mapping[int, float | None],
    candidate: Mapping[int, float | None],
    folds: Sequence[int],
) -> float | None:
    """Unweighted mean over ``folds`` of ``R(S0) - R(S_j)`` in %p.

    ``None`` when any fold lacks a value ([PP] §4-7 3: no mean over the
    remaining folds).
    """
    losses = []
    for fold in folds:
        base, value = reference.get(fold), candidate.get(fold)
        if base is None or value is None:
            return None
        losses.append(100.0 * (base - value))
    return float(np.mean(losses)) if losses else None


def worst_loss_pp(
    recalls: Mapping[tuple[str, PPSetId], Mapping[int, float | None]],
    set_id: PPSetId,
    folds: Sequence[int],
    conditions: Sequence[str] = GATE_CONDITIONS,
) -> tuple[float | None, dict[str, float | None]]:
    """Largest fold-mean loss over ``conditions`` and the per-condition values.

    The worst value is ``None`` (the gate fails) when any condition x fold
    cell has no recall ([PP] §4-7 3); the per-condition values are reported.
    """
    by_condition = {
        name: mean_fold_loss_pp(
            recalls.get((name, PPSetId.S0), {}), recalls.get((name, set_id), {}), folds
        )
        for name in conditions
    }
    values = list(by_condition.values())
    if not values or any(value is None for value in values):
        return None, by_condition
    return max(value for value in values if value is not None), by_condition


STOP_S1_DEFECT = "s1_defect"
STOP_S1_NO_EFFECT = "s1_no_effect"
STOP_GATE_FAILED = "gate_failed"
"""``selection.stop_reason`` values besides ``None`` ([PP] §3, §4-5)."""


@dataclass(frozen=True)
class Selection:
    """Selection rule of [PP] §3.

    Attributes:
        pp_set_id: Largest passing set before the first failure; ``None``
            when S1 fails.
        evaluated: Sets whose gates were read, in order.
        stopped_at: First failing set (``None`` when all pass).
        stop_reason: ``None`` (all passed), ``s1_defect`` (S1 failed G1-G3,
            whatever G4), ``s1_no_effect`` (S1 failed G4 only) or
            ``gate_failed`` (a later set failed).
    """

    pp_set_id: PPSetId | None
    evaluated: tuple[PPSetId, ...]
    stopped_at: PPSetId | None
    stop_reason: str | None = None

    @property
    def s1_failed(self) -> bool:
        return self.stopped_at is PPSetId.S1


def run_status(stop_reason: str | None) -> str:
    """Final-line ``status``: the S1 stop reason, else ``ok`` (S2+ stops too)."""
    if stop_reason is not None and stop_reason in (STOP_S1_DEFECT, STOP_S1_NO_EFFECT):
        return stop_reason
    return "ok"


BLOCKER_SMOKE = "smoke"
BLOCKER_PARTIAL_FOLDS = "partial_folds"
BLOCKER_CROP_LIMIT = "crop_limit"
BLOCKER_PARTIAL_MODELS = "partial_models"
BLOCKER_DIRTY_TREE = "dirty_tree"
"""``judgment.blockers`` values: why a finished run is not a D0 judgment."""


def run_scope_blockers(
    folds_run: Iterable[int],
    models: Sequence[str],
    crop_limit: int | None,
    smoke: bool,
) -> list[str]:
    """Scope reasons a run cannot be judged; empty for a full run.

    A full run covers exactly folds 0-4, every crop and the 20 models. The
    gates of any other run are means over what ran, so they are reported
    but never read as a selection.
    """
    blockers = []
    if smoke:
        blockers.append(BLOCKER_SMOKE)
    if tuple(sorted(folds_run)) != EXPECTED_FOLDS:
        blockers.append(BLOCKER_PARTIAL_FOLDS)
    if crop_limit is not None:
        blockers.append(BLOCKER_CROP_LIMIT)
    if tuple(models) != SYNTHETIC_MODELS:
        blockers.append(BLOCKER_PARTIAL_MODELS)
    return blockers


def final_status(
    stop_reason: str | None, blockers: Sequence[str], tree_changes: Sequence[str]
) -> str:
    """Final-line ``status`` of a finished run.

    A tree that changed during the run is ``input_error`` (the code that ran
    is not the code recorded); any blocker makes it ``not_judgeable``;
    otherwise the selection decides (:func:`run_status`).
    """
    if tree_changes:
        return STATUS_INPUT_ERROR
    if blockers:
        return STATUS_NOT_JUDGEABLE
    return run_status(stop_reason)


def select_pp_set(gates: Mapping[PPSetId, GateResult]) -> Selection:
    """Walk S1 ⊂ ... ⊂ S5; keep the last passing set, stop at the first failure."""
    selected = None
    evaluated = []
    for set_id in GATE_SETS:
        evaluated.append(set_id)
        gate = gates[set_id]
        if gate.passed:
            selected = set_id
            continue
        reason = STOP_GATE_FAILED
        if set_id is PPSetId.S1:
            reason = STOP_S1_DEFECT if gate.tp_gates_failed else STOP_S1_NO_EFFECT
        return Selection(selected, tuple(evaluated), set_id, reason)
    return Selection(selected, tuple(evaluated), None, None)


# ----------------------------------------------------------- inference


def _load_detector(weights: Path, device: str, confidence: float):
    # Lazy: ultralytics stays inside gpr_detection.detect (AGPL boundary).
    from gpr_detection.detect.yolo import load_yolo_detector
    from gpr_detection.detect.yolo_logging import quiet_ultralytics_logger

    quiet_ultralytics_logger()
    return load_yolo_detector(
        weights,
        input_size=CANVAS_PX,
        confidence_threshold=confidence,
        iou_threshold=NMS_IOU,
        device=device,
    )


def predict_batched(detector, images: Sequence[np.ndarray]) -> list:
    """``predict_batch`` in chunks of 64 (the M3-2 S5 batch path)."""
    outputs = []
    for start in range(0, len(images), BATCH_SIZE):
        outputs.extend(detector.predict_batch(list(images[start : start + BATCH_SIZE])))
    return outputs


def crop_predictions(output) -> Predictions:
    """Boxes of one 256 x 256 crop; the letterbox must be the identity."""
    transform = output.transform
    if (transform.scale_x, transform.scale_y, transform.pad_x, transform.pad_y) != (
        1,
        1,
        0,
        0,
    ):
        raise RuntimeError(f"letterbox not identity: {transform}")
    if not output.detections:
        return Predictions(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))
    return Predictions(
        np.array([d.box.xyxy for d in output.detections], dtype=float),
        np.array([d.score for d in output.detections], dtype=float),
        np.array([d.class_id for d in output.detections], dtype=int),
    )


def synthetic_merged(
    detector, lines: Sequence[SyntheticLine]
) -> list[tuple[LineBox, ...]]:
    """Merged line boxes of every synthetic line (all tiles in shared batches)."""
    tiles = [(j, tile) for j, line in enumerate(lines) for tile in line.image.tiles]
    pixels = [lines[j].image.tile_image(tile.index) for j, tile in tiles]
    tile_boxes: list[list[LineBox]] = [[] for _ in lines]
    for (j, tile), output in zip(tiles, predict_batched(detector, pixels), strict=True):
        tile_boxes[j].extend(
            tile_output_line_boxes(output, tile, DEFAULT_MERGE_CONFIG.edge_px)
        )
    return [
        merge_line_boxes(boxes, line.image, DEFAULT_MERGE_CONFIG)
        for boxes, line in zip(tile_boxes, lines, strict=True)
    ]


# ------------------------------------------------------------- fold run


@dataclass
class FoldOutput:
    """Everything one fold produces (values stay in files).

    ``write_predictions`` receives the ``predictions.csv`` rows chunk by chunk
    (conf 0.001 boxes are too many to hold for a whole fold).
    """

    write_predictions: Callable[[list[list]], None]
    scores: dict[tuple[str, PPSetId], SetScores] = field(default_factory=dict)
    object_rows: list[list] = field(default_factory=list)
    band_changes: dict[tuple[str, PPSetId], list[float]] = field(default_factory=dict)
    """(condition, set) -> [lost, gained, GT length] of the band coverage (G2)."""
    synthetic_rows: list[list] = field(default_factory=list)
    synthetic_processed: Counter = field(default_factory=Counter)
    synthetic_box_rows: list[list] = field(default_factory=list)
    synthetic_rules: dict[tuple[str, PPSetId], Counter] = field(default_factory=dict)
    """(Y0 / Y1_<f>P, set) -> processed count per action (all merged boxes)."""
    surface_rows: Counter = field(default_factory=Counter)
    n_images: int = 0


def read_crops(
    eval_list: Path, guard: InputGuard, gains: Mapping[str, str], pad_heights, limit
) -> list[perturb.Sample]:
    """Validation crops of one fold with labels, gain copy and pad height."""
    samples = []
    for line in guard.check(eval_list).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        path = guard.check(Path(line.strip()))
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (CANVAS_PX, CANVAS_PX):
            raise common.InputError(f"bad image {path}")
        label = guard.check(LABEL_DIR / f"{path.stem}.txt")
        boxes, classes, gt_index = common.read_yolo_label_indexed(label)
        pad = perturb.draw_pad_height(path.stem, pad_heights)
        samples.append(
            perturb.Sample(
                path.stem, gains[path.name], image, boxes, classes, gt_index, pad
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def band_gt_coverage(view: CropView, kept: Predictions) -> np.ndarray:
    """Covered x length of every ``pipe_band`` GT of the view (conf > 0.25)."""
    confident = kept.scores > CONF_RECALL
    return gt_covered_lengths(
        _band_part(view.gt_boxes, view.gt_classes),
        _band_part(kept.boxes[confident], kept.classes[confident]),
    )


def add_band_changes(
    output: FoldOutput, view: CropView, covered: Mapping[PPSetId, np.ndarray]
) -> None:
    """Add per-GT coverage loss and gain of every set against S0 (G2)."""
    gt = _band_part(view.gt_boxes, view.gt_classes)
    length = float(np.clip(gt[:, 2] - gt[:, 0], 0.0, None).sum())
    reference = covered[PPSetId.S0]
    for key, values in covered.items():
        change = values - reference
        cell = output.band_changes.setdefault((view.condition, key), [0.0, 0.0, 0.0])
        cell[0] += float(np.clip(-change, 0.0, None).sum())
        cell[1] += float(np.clip(change, 0.0, None).sum())
        cell[2] += length


def _prediction_rows(
    fold: int,
    view: CropView,
    sample: perturb.Sample,
    predictions: Predictions,
    actions: Mapping[PPSetId, Sequence[PPAction]],
) -> list[list]:
    rows = []
    for index, (box, score, class_id) in enumerate(
        zip(predictions.boxes, predictions.scores, predictions.classes, strict=True)
    ):
        category = pp_category(float(box[1]), float(box[3]), view.content)
        rows.append(
            [
                fold,
                view.condition,
                sample.stem,
                sample.gain,
                *(exact(v) for v in box),
                exact(score),
                common.CLASS_NAMES[int(class_id)],
                "" if category is None else category.value,
                *(actions[key][index].value for key in GATE_SETS),
            ]
        )
    return rows


def _score_crop(
    fold: int,
    sample: perturb.Sample,
    images: Mapping[str, np.ndarray],
    predictions: Mapping[str, Predictions],
    output: FoldOutput,
) -> None:
    for view in crop_views(sample, images):
        raw = predictions[view.image_key]
        kept = perturb.drop_ignored(
            raw.boxes, raw.scores, raw.classes, view.ignore_boxes
        )
        original = Predictions(*kept)
        hits = {}
        actions = {}
        covered = {}
        for key in PPSetId:
            crop_pp = pp_on_predictions(original, view.content, PP_SETS[key])
            actions[key] = crop_pp.actions
            scores = output.scores.setdefault((view.condition, key), SetScores())
            hits[key] = score_view(scores, view, original, crop_pp, sample.gain)
            covered[key] = band_gt_coverage(view, crop_pp.kept)
        add_band_changes(output, view, covered)
        output.write_predictions(
            _prediction_rows(fold, view, sample, original, actions)
        )
        for gt, (class_id, index) in enumerate(
            zip(view.gt_classes, view.gt_index, strict=True)
        ):
            output.object_rows.append(
                [
                    fold,
                    view.condition,
                    sample.stem,
                    sample.gain,
                    common.CLASS_NAMES[int(class_id)],
                    int(index),
                    *(int(hits[key][gt]) for key in PPSetId),
                ]
            )
        if view.condition == "A3":
            output.surface_rows[int(view.content.zero_px or 0)] += 1


CROP_CHUNK = 512
"""Crops perturbed and inferred together (memory bound, batches stay 64)."""


def run_crops(
    detector, samples: Sequence[perturb.Sample], fold: int, output: FoldOutput
) -> None:
    """Infer A0-A2 once per crop, then score A0-A3 under S0-S5."""
    for start in range(0, len(samples), CROP_CHUNK):
        chunk = samples[start : start + CROP_CHUNK]
        images = [condition_images(sample) for sample in chunk]
        predictions: dict[str, list[Predictions]] = {}
        for key in INFERRED_IMAGES:
            outputs = predict_batched(detector, [item[key] for item in images])
            predictions[key] = [crop_predictions(o) for o in outputs]
        for index, sample in enumerate(chunk):
            per_image = {key: predictions[key][index] for key in INFERRED_IMAGES}
            _score_crop(fold, sample, images[index], per_image, output)
    output.n_images += len(samples) * len(INFERRED_IMAGES)


def run_synthetic(
    detector, lines: Sequence[SyntheticLine], fold: int, output: FoldOutput
) -> None:
    """Y0 and Y1 lines of one fold: merge, PP S0-S5, apex match."""
    for line, merged in zip(lines, synthetic_merged(detector, lines), strict=True):
        add_synthetic_line(output, fold, line, merged)
    output.n_images += sum(len(line.image.tiles) for line in lines)


def add_synthetic_line(
    output: FoldOutput, fold: int, line: SyntheticLine, merged: Sequence[LineBox]
) -> None:
    """Rows, box rows and per-rule counts of one merged synthetic line."""
    row, processed = synthetic_row(fold, line, merged)
    output.synthetic_rows.append(row)
    output.synthetic_box_rows.extend(synthetic_box_rows(fold, line, merged))
    for key in GATE_SETS:
        rules = output.synthetic_rules.setdefault((line.condition, key), Counter())
        rules.update({a.value: processed[key][a.value] for a in PROCESSED})
        if line.condition == "Y0":
            output.synthetic_processed[key.value] += _n_processed(processed[key])


# ---------------------------------------------------------- aggregation


def _recalls_by_fold(
    folds: Mapping[int, FoldOutput], metric: str
) -> dict[tuple[str, PPSetId], dict[int, float | None]]:
    table: dict[tuple[str, PPSetId], dict[int, float | None]] = {}
    for fold, output in folds.items():
        for key, scores in output.scores.items():
            if metric == "band":
                value = scores.band.recall
            else:
                value = scores.class_results()[metric]["recall_conf025"]
            table.setdefault(key, {})[fold] = value
    return table


def hyperbola_loss_cells(
    folds: Mapping[int, FoldOutput], set_id: PPSetId, class_name: str = HYPERBOLA_NAME
) -> dict[tuple[str, int], LossCell]:
    """G1 cells: paired GT of ``objects.csv`` (``hit_S0`` vs ``hit_Sj``).

    Lost = hit by S0 and missed by S_j, gained = missed by S0 and hit by S_j
    (reported, never subtracted); total = GT of the class in the cell.
    """
    column = {name: i for i, name in enumerate(OBJECTS_HEADER)}
    reference, candidate = column[f"hit_{PPSetId.S0}"], column[f"hit_{set_id}"]
    counts: dict[tuple[str, int], list[int]] = {}
    for fold, output in folds.items():
        for row in output.object_rows:
            if row[column["class"]] != class_name:
                continue
            cell = counts.setdefault((row[column["condition"]], fold), [0, 0, 0])
            cell[0] += int(row[reference] == 1 and row[candidate] == 0)
            cell[1] += int(row[reference] == 0 and row[candidate] == 1)
            cell[2] += 1
    return {key: LossCell(*map(float, value)) for key, value in counts.items()}


def band_loss_cells(
    folds: Mapping[int, FoldOutput], set_id: PPSetId
) -> dict[tuple[str, int], LossCell]:
    """G2 cells: per-GT covered length decrease / increase against S0."""
    cells = {}
    for fold, output in folds.items():
        for (condition, key), (lost, gained, total) in output.band_changes.items():
            if key is set_id:
                cells[(condition, fold)] = LossCell(lost, gained, total)
    return cells


def _synthetic_sum(folds: Mapping[int, FoldOutput], name: str, condition: str) -> int:
    column = {header: i for i, header in enumerate(SYNTHETIC_HEADER)}
    return sum(
        int(row[column[name]])
        for output in folds.values()
        for row in output.synthetic_rows
        if row[column["condition"]] == condition
    )


def _synthetic_hits(folds: Mapping[int, FoldOutput], condition: str) -> dict:
    return {key: _synthetic_sum(folds, f"hit_{key}", condition) for key in PPSetId}


def _g4_processed(folds: Mapping[int, FoldOutput], set_id: PPSetId) -> int:
    """Cumulative processed boxes of a set: A1 + A3 (score > 0.25) + Y0."""
    crops = sum(
        _n_processed(output.scores[(name, set_id)].processed_confident)
        for output in folds.values()
        for name in G4_CROP_CONDITIONS
        if (name, set_id) in output.scores
    )
    synthetic = sum(o.synthetic_processed[set_id.value] for o in folds.values())
    return crops + synthetic


class GateDefectError(RuntimeError):
    """A structurally impossible gate value (negative G4 increment): exit 3."""


def g4_increments(folds: Mapping[int, FoldOutput]) -> dict[PPSetId, int]:
    """``processed(S_j) - processed(S_(j-1))`` with ``processed(S0) = 0``.

    Raises:
        GateDefectError: If an increment is negative ([PP] §4-7 2: the sets
            are nested and the categories exclusive, so that is a defect).
    """
    increments = {}
    previous = 0
    for set_id in GATE_SETS:
        cumulative = _g4_processed(folds, set_id)
        increments[set_id] = cumulative - previous
        if increments[set_id] < 0:
            raise GateDefectError(f"G4 increment of {set_id} is negative")
        previous = cumulative
    return increments


def compute_gates(folds: Mapping[int, FoldOutput]) -> dict[PPSetId, GateResult]:
    """G1-G4 of every set S1-S5 from the fold outputs ([PP] v0.2.3 §4-4).

    The fold means run over the folds in ``folds``; whether those are the
    five a judgment needs is :func:`run_scope_blockers`' call, not this one.

    Raises:
        GateDefectError: For a negative G4 increment or per-rule G4 counts
            that do not add up to the cumulative count.
    """
    fold_ids = sorted(folds)
    y0_hits = _synthetic_hits(folds, "Y0")
    increments = g4_increments(folds)
    check_g4_rules(folds)
    gates = {}
    for set_id in GATE_SETS:
        gates[set_id] = GateResult(
            g1_loss_pp=worst_fold_mean_pp(
                hyperbola_loss_cells(folds, set_id), fold_ids
            )[0],
            g2_loss_pp=worst_fold_mean_pp(band_loss_cells(folds, set_id), fold_ids)[0],
            g3_loss=y0_hits[PPSetId.S0] - y0_hits[set_id],
            g4_increment=increments[set_id],
            g3_h_s0=y0_hits[PPSetId.S0],
        )
    return gates


def _cell_table(cells: Mapping[tuple[str, int], LossCell]) -> dict:
    """Condition -> fold -> lost, gained, total and the two %p values."""
    table: dict = {}
    for (condition, fold), cell in sorted(cells.items()):
        table.setdefault(condition, {})[str(fold)] = {
            **asdict(cell),
            "loss_pp": cell.loss_pp,
            "gain_pp": cell.gain_pp,
        }
    return table


def _gain_by_condition(
    cells: Mapping[tuple[str, int], LossCell], folds: Sequence[int]
) -> dict[str, float | None]:
    gains = {key: LossCell(cell.gained, 0.0, cell.total) for key, cell in cells.items()}
    return worst_fold_mean_pp(gains, folds, CROP_CONDITIONS)[1]


def _loss_gate_row(
    cells: Mapping[tuple[str, int], LossCell],
    folds: Sequence[int],
    loss_pp: float | None,
    threshold_pp: float,
) -> dict:
    return {
        "total_loss_pp": loss_pp,
        "by_condition": worst_fold_mean_pp(cells, folds)[1],
        "gain_pp_by_condition": _gain_by_condition(cells, folds),
        "report_conditions": worst_fold_mean_pp(cells, folds, REPORT_CONDITIONS)[1],
        "cells": _cell_table(cells),
        "threshold_pp": threshold_pp,
    }


def _gate_row(
    folds: Mapping[int, FoldOutput], set_id: PPSetId, result: GateResult
) -> dict:
    fold_ids = sorted(folds)
    y0_hits = _synthetic_hits(folds, "Y0")
    return {
        "G1": _loss_gate_row(
            hyperbola_loss_cells(folds, set_id),
            fold_ids,
            result.g1_loss_pp,
            G1_MAX_LOSS_PP,
        ),
        "G2": _loss_gate_row(
            band_loss_cells(folds, set_id), fold_ids, result.g2_loss_pp, G2_MAX_LOSS_PP
        ),
        "G3": {
            "loss": result.g3_loss,
            "h_s0": y0_hits[PPSetId.S0],
            "h_sj": y0_hits[set_id],
            "exposed_boxes_s0": _synthetic_sum(folds, "n_exposed_boxes_S0", "Y0"),
            "required": "h_s0 >= 1 and loss == 0",
        },
        "G4": {
            "increment": result.g4_increment,
            "cumulative": _g4_processed(folds, set_id),
            **rule_counts(folds, set_id, G4_CONDITIONS),
            "required": "> 0",
        },
        "checks": result.checks,
        "pass": result.passed,
    }


G4_CONDITIONS = (*G4_CROP_CONDITIONS, "Y0")
OUTSIDE_GATE_CONDITIONS = (
    "A0",
    "A2",
    *(y1_condition_name(periods) for periods in Y1_CUT_PERIODS),
)
"""Report-only rule counts in the G4 format ([PP] §4-5): A0, A2, Y1 cuts."""


def _condition_rules(
    folds: Mapping[int, FoldOutput], condition: str, set_id: PPSetId
) -> Counter[str]:
    """Processed count per action of one condition and set, over the folds.

    Crop conditions count predictions with the original score > 0.25;
    synthetic ones (Y0, Y1_<f>P) count every merged box ([PP] §4-7 2).
    """
    counts: Counter[str] = Counter()
    for output in folds.values():
        scores = output.scores.get((condition, set_id))
        if scores is not None:
            counts.update(scores.processed_confident)
        counts.update(output.synthetic_rules.get((condition, set_id), Counter()))
    return Counter({a.value: counts[a.value] for a in PROCESSED if counts[a.value]})


def rule_counts(
    folds: Mapping[int, FoldOutput], set_id: PPSetId, conditions: Sequence[str]
) -> dict:
    """``{"by_condition": {condition: {action: n}}, "by_rule": {action: n}}``."""
    by_condition = {name: _condition_rules(folds, name, set_id) for name in conditions}
    total: Counter[str] = Counter()
    for counts in by_condition.values():
        total.update(counts)
    return {
        "by_condition": {
            name: dict(sorted(c.items())) for name, c in by_condition.items()
        },
        "by_rule": dict(sorted(total.items())),
    }


def check_g4_rules(folds: Mapping[int, FoldOutput]) -> None:
    """``sum(by_rule) == cumulative`` for every set ([PP] §4-5 G4).

    Raises:
        GateDefectError: When the per-rule counts of A1 + A3 + Y0 do not add
            up to the cumulative G4 count (exit 3).
    """
    for set_id in GATE_SETS:
        by_rule = rule_counts(folds, set_id, G4_CONDITIONS)["by_rule"]
        cumulative = _g4_processed(folds, set_id)
        if sum(by_rule.values()) != cumulative:
            raise GateDefectError(
                f"G4 of {set_id}: sum(by_rule) {sum(by_rule.values())} "
                f"!= cumulative {cumulative}"
            )


def _class_tables(folds: Mapping[int, FoldOutput]) -> dict:
    tables: dict = {}
    for fold, output in folds.items():
        for (condition, set_id), scores in output.scores.items():
            classes = scores.class_results()
            by_gain = {
                gain: accumulator.class_results()
                for gain, accumulator in scores.by_gain.items()
            }
            tables.setdefault(condition, {}).setdefault(set_id.value, {})[fold] = {
                "classes": classes,
                "map50": perturb._map50(classes),
                "band_recall": scores.band.recall,
                "band_lengths": asdict(scores.band),
                "by_gain": by_gain,
                "processed": dict(scores.processed),
                "processed_conf025": dict(scores.processed_confident),
                "pp_a_inside_content": {
                    "all": scores.pp_a_inside[0],
                    "conf025": scores.pp_a_inside[1],
                },
            }
    return tables


def _report(folds: Mapping[int, FoldOutput]) -> dict:
    """Report-only values ([PP] §4-4 보고): A2, Y1, other classes, mAP50, ..."""
    fold_ids = sorted(folds)
    report: dict = {"fold_tables": _class_tables(folds)}
    for class_name in common.CLASS_NAMES:
        recalls = _recalls_by_fold(folds, class_name)
        report.setdefault("recall_loss_pp", {})[class_name] = {
            set_id.value: worst_loss_pp(recalls, set_id, fold_ids, CROP_CONDITIONS)[1]
            for set_id in GATE_SETS
        }
    band = _recalls_by_fold(folds, "band")
    report["band_recall_loss_pp"] = {
        set_id.value: worst_loss_pp(band, set_id, fold_ids, CROP_CONDITIONS)[1]
        for set_id in GATE_SETS
    }
    report["synthetic_apex_hits"] = {
        condition: {k.value: v for k, v in _synthetic_hits(folds, condition).items()}
        for condition in ("Y0", *(y1_condition_name(p) for p in Y1_CUT_PERIODS))
    }
    report["rule_counts_outside_gate"] = {
        set_id.value: rule_counts(folds, set_id, OUTSIDE_GATE_CONDITIONS)
        for set_id in GATE_SETS
    }
    surface: Counter[int] = Counter()
    for output in folds.values():
        surface.update(output.surface_rows)
    report["a3_zero_row_counts"] = {str(row): n for row, n in sorted(surface.items())}
    return report


def gate_document(folds: Mapping[int, FoldOutput]) -> dict:
    """Gates, selection and report of ``d0_gate.json`` ([PP] §4-5).

    Raises:
        GateDefectError: For a negative G4 increment.
    """
    gates = compute_gates(folds)
    selection = select_pp_set(gates)
    gate_rows = {
        set_id.value: _gate_row(folds, set_id, result)
        for set_id, result in gates.items()
    }
    selected = selection.pp_set_id
    stopped = selection.stopped_at
    return {
        "gates": gate_rows,
        "selection": {
            "pp_set_id": None if selected is None else selected.value,
            "pp_config_sha256": None
            if selected is None
            else PP_SETS[selected].sha256(),
            "evaluated": [key.value for key in selection.evaluated],
            "stopped_at": None if stopped is None else stopped.value,
            "stop_reason": selection.stop_reason,
            "s1_failed": selection.s1_failed,
        },
        "report": _report(folds),
    }


# ------------------------------------------------------------ identity


def code_paths() -> list[Path]:
    """Code whose git blobs identify the run ([PP] §4-1 코드)."""
    return [
        Path(__file__).resolve(),
        Path(common.__file__).resolve(),
        Path(perturb.__file__).resolve(),
        REPO / "src" / "gpr_detection" / "pipeline.py",
        REPO / "src" / "gpr_detection" / "preprocess" / "to_image.py",
        NO_TEST_SCRIPT,
    ]


def git_blobs(paths: Iterable[Path]) -> dict[str, str]:
    """``git hash-object`` of each file (repo-relative keys)."""
    blobs = {}
    for path in paths:
        relative = path.resolve().relative_to(REPO.resolve()).as_posix()
        blobs[relative] = _git("hash-object", "--", relative).strip()
    return blobs


TREE_KEYS = ("head", "dirty", "problems", "code_blobs", "src_tree_id")
"""``tree_state`` fields compared at the start and the end of a run."""


def tree_state() -> dict:
    """HEAD, tracked-tree problems, code blobs and the ``src`` tree id now.

    ``dirty`` counts modified tracked files anywhere in the repository and
    untracked required code (:func:`perturb.tracked_tree_problems`).
    """
    problems = perturb.tracked_tree_problems(code_paths())
    return {
        "head": common.git_state()["head"],
        "dirty": bool(problems),
        "problems": problems,
        "code_blobs": git_blobs(code_paths()),
        "src_tree_id": _git("rev-parse", "HEAD:src").strip(),
    }


def tree_changes(start: Mapping[str, Any], end: Mapping[str, Any]) -> list[str]:
    """``TREE_KEYS`` that differ between two snapshots (an unknown HEAD counts)."""
    changed = [key for key in TREE_KEYS if start.get(key) != end.get(key)]
    if "head" not in changed and start.get("head") is None:
        changed.insert(0, "head")
    return changed


def assert_sealed_pp(start_tree: Mapping[str, Any]) -> dict:
    """Assert the PP sets and the PP code are the sealed ones ([PP] A-1).

    The ``PPConfig`` hashes must equal the C0b record (selected S3
    included) and the ``src`` tree and ``pipeline.py`` blob of the start
    snapshot must equal C0a (53c4e16): a re-run of D0 is only comparable
    with C0b when the PP it applies is the same.

    Returns:
        ``run_identity.sealed_pp_assertion``: the checks, the expected and
        the measured values, and ``"result": "pass"``.

    Raises:
        common.InputError: When any check fails (exit 2, before inference).
    """
    measured = {
        "pp_set_sha256": {key.value: PP_SETS[key].sha256() for key in PPSetId},
        "pp_sets_sha256": pp_sets_sha256(),
        "src_tree_id": start_tree["src_tree_id"],
        "pipeline_blob": start_tree["code_blobs"].get(PIPELINE_PATH),
    }
    expected = {
        "pp_set_sha256": {key.value: sha for key, sha in C0B_PP_SET_SHA256.items()},
        "pp_sets_sha256": C0B_PP_SETS_SHA256,
        "src_tree_id": C0A_SRC_TREE_ID,
        "pipeline_blob": C0A_PIPELINE_BLOB,
    }
    selected = C0B_SELECTED_SET.value
    checks = {name: measured[name] == expected[name] for name in expected}
    checks["selected_set_sha256"] = (
        measured["pp_set_sha256"][selected] == expected["pp_set_sha256"][selected]
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise common.InputError(f"PP differs from C0a/C0b: {failed}")
    return {
        "checks": checks,
        "expected": expected | {"selected_set": selected},
        "measured": measured,
        "result": "pass",
    }


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=common.GIT_TIMEOUT_S,
        check=True,
    )
    return completed.stdout


def assert_no_test_crops(eval_lists: Sequence[Path], guard: InputGuard) -> dict:
    """Run the X7 assertion script on the crop lists; any hit refuses the run."""
    script = guard.check(NO_TEST_SCRIPT)
    for path in eval_lists:
        guard.check(path)
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(script), *map(str, eval_lists)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        raise common.InputError(
            f"AIHub test assertion failed (exit {completed.returncode})"
        )
    return check_no_test_output(completed.stdout, len(eval_lists))


def check_no_test_output(stdout: str, n_lists: int) -> dict:
    """Parse the X7 script output; every list must be fully in the manifest.

    One JSON line per list, then a summary line. ``not_in_manifest`` > 0
    means crops the test assertion could not look up.

    Raises:
        common.InputError: For a missing list line or an unknown crop.
    """
    records = [json.loads(line) for line in stdout.strip().splitlines() if line]
    *per_list, summary = records
    if len(per_list) != n_lists:
        raise common.InputError(f"test assertion printed {len(per_list)} lists")
    unknown = {
        r["list"]: r["not_in_manifest"] for r in per_list if r["not_in_manifest"]
    }
    if unknown:
        raise common.InputError(f"crops missing from the manifest: {unknown}")
    return {"exit_code": 0, "lists": per_list, "summary": summary}


def synthetic_sha256(models: Sequence[str], guard: InputGuard) -> dict[str, str]:
    hashes = {}
    for model in models:
        for suffix in (".npy", ".json"):
            path = guard.check(SYNTHETIC_ROOT / f"{model}{suffix}")
            hashes[path.name] = common.sha256_file(path)
    return hashes


# ----------------------------------------------------------------- CLI


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--device", default="0", help="'0' (GPU) or 'cpu'")
    parser.add_argument("--limit", type=int, default=None, help="crops per fold")
    parser.add_argument("--synthetic-limit", type=int, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"fold 0, {SMOKE_CROPS} crops, {SMOKE_MODELS} models, smoke folder",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--probe-path",
        type=Path,
        default=None,
        help="check one path against the allow and forbidden lists and exit",
    )
    args = parser.parse_args(argv)
    if args.smoke:
        args.folds = "0"
        args.limit = SMOKE_CROPS if args.limit is None else args.limit
        args.synthetic_limit = args.synthetic_limit or SMOKE_MODELS
        args.out_root = args.out_root or SMOKE_OUT_ROOT
    args.out_root = args.out_root or DEFAULT_OUT_ROOT
    return args


@dataclass(frozen=True)
class RunInputs:
    """Inputs shared by every fold, read through the guard."""

    manifest: list[dict[str, str]]
    pad_heights: np.ndarray
    models: tuple[str, ...]
    folds: tuple[int, ...]


def parse_folds(text: str) -> tuple[int, ...]:
    """``--folds`` as fold ids; duplicates and folds outside 0-4 are refused.

    Raises:
        common.InputError: For a non-integer, repeated or unknown fold.
    """
    try:
        folds = tuple(int(part) for part in text.split(","))
    except ValueError as error:
        raise common.InputError(f"bad --folds {text!r}") from error
    if len(set(folds)) != len(folds):
        raise common.InputError(f"repeated fold in --folds {text!r}")
    unknown = sorted(set(folds) - set(EXPECTED_FOLDS))
    if unknown:
        raise common.InputError(f"unknown folds {unknown} in --folds {text!r}")
    return folds


def _load_inputs(args: argparse.Namespace, guard: InputGuard) -> RunInputs:
    models = SYNTHETIC_MODELS[: args.synthetic_limit]
    return RunInputs(
        manifest=perturb.read_manifest(guard.check(MANIFEST)),
        pad_heights=perturb.load_pad_heights(guard.check(PAD_HEIGHTS_CSV)),
        models=tuple(models),
        folds=parse_folds(args.folds),
    )


def _fold_paths(fold: int) -> tuple[Path, Path]:
    run_dir = RUN_ROOT / f"{RUN_PREFIX}{fold}"
    return (
        run_dir / "weights" / "last.pt",
        run_dir / "eval" / "last_crop" / "eval_images.txt",
    )


def _check_fixed_inputs(
    inputs: RunInputs, guard: InputGuard, run_identity: dict
) -> None:
    """Weights, crop lists, split and test seal, before any inference."""
    expected_lists = perturb.read_expected_eval_list_sha256(guard.check(D1_R4_VERDICT))
    eval_lists = []
    for fold in inputs.folds:
        weights, eval_list = _fold_paths(fold)
        run_identity["weights_sha256"][str(fold)] = perturb._checked_sha256(
            guard.check(weights), perturb.EXPECTED_WEIGHTS_SHA256[fold], "weights"
        )
        run_identity["eval_list_sha256"][str(fold)] = perturb._checked_sha256(
            guard.check(eval_list), expected_lists.get(fold), "eval list"
        )
        names = perturb.read_eval_names(eval_list)
        split = perturb.check_fold_split(inputs.manifest, names, fold)
        failed = [name for name, ok in split["checks"].items() if not ok]
        if failed:
            raise common.InputError(f"fold {fold} split assertions failed: {failed}")
        eval_lists.append(eval_list)
    run_identity["no_aihub_test"] = assert_no_test_crops(eval_lists, guard)


TEXT_INPUTS = {
    "manifest": MANIFEST,
    "pad_heights": PAD_HEIGHTS_CSV,
    "d1_r4_verdict": D1_R4_VERDICT,
}
"""Text tables whose sha is also recorded with CRLF read as LF."""


def _new_run_identity(
    args: argparse.Namespace,
    inputs: RunInputs,
    guard: InputGuard,
    start_tree: Mapping[str, Any],
) -> dict:
    run_identity = common.new_run_identity()
    # M3-1 D1 field; this runner reads no validator export.
    del run_identity["g1_inputs_sha256"]
    run_identity["gpu"] = common.gpu_name(args.device)
    run_identity["dirty"] = start_tree["dirty"]
    run_identity["tree_problems"] = start_tree["problems"]
    run_identity["allow_dirty"] = bool(args.allow_dirty or args.smoke)
    run_identity["code_blobs"] = start_tree["code_blobs"]
    run_identity["src_tree_id"] = start_tree["src_tree_id"]
    run_identity["inputs_sha256"] = {
        **{name: common.sha256_file(path) for name, path in TEXT_INPUTS.items()},
        "synthetic": synthetic_sha256(inputs.models, guard),
    }
    # The work tree may hold a tracked table with CRLF (checked out before
    # .gitattributes set eol=lf); the LF sha matches the committed content.
    run_identity["inputs_sha256_lf"] = {
        name: common.sha256_file_lf(path) for name, path in TEXT_INPUTS.items()
    }
    run_identity["pp_sets"] = {
        key.value: {
            "config": json.loads(PP_SETS[key].canonical_json()),
            "sha256": PP_SETS[key].sha256(),
        }
        for key in PPSetId
    }
    run_identity["pp_sets_sha256"] = pp_sets_sha256()
    return run_identity


def _settings(args: argparse.Namespace, inputs: RunInputs) -> dict:
    return {
        "seed": SEED,
        "period_aihub_px": PERIOD_AIHUB_PX,
        "top_band_px": TOP_BAND_PX,
        "surface_search_rows": SURFACE_SEARCH_ROWS,
        "y1_cut_periods": list(Y1_CUT_PERIODS),
        "conf_ap": CONF_AP,
        "conf_recall": CONF_RECALL,
        "conf_line": CONF_LINE,
        "nms_iou": NMS_IOU,
        "batch_size": BATCH_SIZE,
        "folds": list(inputs.folds),
        "models": list(inputs.models),
        "limit": args.limit,
        "smoke": args.smoke,
        "expected_folds": list(EXPECTED_FOLDS),
        "full_run": not run_scope_blockers(
            inputs.folds, inputs.models, args.limit, args.smoke
        ),
        "gate_conditions": list(GATE_CONDITIONS),
        "report_conditions": list(REPORT_CONDITIONS),
        "device": args.device,
    }


def _run_fold(
    args: argparse.Namespace,
    inputs: RunInputs,
    fold: int,
    guard: InputGuard,
    lines: Sequence[SyntheticLine],
    started: float,
) -> FoldOutput:
    weights, eval_list = _fold_paths(fold)
    gains = perturb.gain_by_name(inputs.manifest)
    samples = read_crops(eval_list, guard, gains, inputs.pad_heights, args.limit)
    fold_dir = args.out_root / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    with (fold_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(PREDICTIONS_HEADER)
        output = FoldOutput(write_predictions=writer.writerows)
        crop_detector = _load_detector(weights, args.device, CONF_AP)
        run_crops(crop_detector, samples, fold, output)
    elapsed = time.time() - started
    print(f"fold {fold}: {len(samples)} crops x 3 images ({elapsed:.0f} s)", flush=True)
    line_detector = _load_detector(weights, args.device, CONF_LINE)
    run_synthetic(line_detector, lines, fold, output)
    elapsed = time.time() - started
    print(f"fold {fold}: {len(lines)} synthetic lines ({elapsed:.0f} s)", flush=True)
    return output


def _write_outputs(
    out_root: Path, folds: Mapping[int, FoldOutput], document: dict
) -> None:
    """``predictions.csv`` (crop only) is streamed by ``_run_fold``; the rest here."""
    for fold, output in folds.items():
        fold_dir = out_root / f"fold{fold}"
        common.write_csv(fold_dir / "objects.csv", OBJECTS_HEADER, output.object_rows)
        common.write_csv(
            fold_dir / "synthetic_boxes.csv",
            SYNTHETIC_BOXES_HEADER,
            output.synthetic_box_rows,
        )
    synthetic = [row for output in folds.values() for row in output.synthetic_rows]
    common.write_csv(out_root / "synthetic_apex.csv", SYNTHETIC_HEADER, synthetic)
    common.write_json(out_root / "d0_gate.json", document)


def judgment_record(
    folds_run: Sequence[int],
    scope_blockers: Sequence[str],
    start_tree: Mapping[str, Any],
    end_tree: Mapping[str, Any],
) -> dict:
    """``judgment`` block of ``d0_gate.json``: is this run a D0 judgment?

    Blockers are the scope reasons plus a tree dirty at the start; a tree
    that changed between the start and the end is listed apart
    (``tree_changes``, status ``input_error``).
    """
    blockers = list(scope_blockers)
    if start_tree["dirty"]:
        blockers.append(BLOCKER_DIRTY_TREE)
    changes = tree_changes(start_tree, end_tree)
    return {
        "judgeable": not blockers and not changes,
        "blockers": blockers,
        "expected_folds": list(EXPECTED_FOLDS),
        "folds_run": list(folds_run),
        "tree_stable": not changes,
        "tree_changes": changes,
    }


def _folder_key(path: Path) -> Path:
    # normcase: Windows paths that differ only in case name the same folder.
    return Path(os.path.normcase(str(Path(path).resolve())))


def is_c0b_out_root(out_root: Path) -> bool:
    """``out_root`` is the C0b result folder ``runs/eval/m33_d0`` or inside it.

    Paths are resolved first (relative to the working directory), so
    relative, absolute and differently cased spellings all match; a folder
    below C0b (``runs/eval/m33_d0/rerun``) would also write into the sealed
    results and is refused too.
    """
    return _folder_key(out_root).is_relative_to(_folder_key(C0B_OUT_ROOT))


def run(args: argparse.Namespace, guard: InputGuard) -> dict:
    """Assert the inputs, run every fold and write the outputs.

    The C0b result folder is refused as the output folder whatever the
    flags (exit 2); any other existing folder needs ``--overwrite``.

    Returns:
        ``{"status", "n_conditions", "n_images"}`` for the closing line;
        ``status`` comes from :func:`final_status`.

    Raises:
        common.InputError: For a refused input (exit 2).
    """
    started = time.time()
    if is_c0b_out_root(args.out_root):
        raise common.InputError(
            f"{args.out_root} is the sealed C0b result folder; "
            "give another --out-root (--overwrite does not apply)"
        )
    start_tree = tree_state()
    if not (args.allow_dirty or args.smoke) and start_tree["problems"]:
        raise common.InputError(f"tree not clean: {start_tree['problems']}")
    sealed_pp = assert_sealed_pp(start_tree)
    inputs = _load_inputs(args, guard)
    run_identity = _new_run_identity(args, inputs, guard, start_tree)
    run_identity["sealed_pp_assertion"] = sealed_pp
    _check_fixed_inputs(inputs, guard, run_identity)
    if args.out_root.exists():
        if not args.overwrite:
            raise common.InputError(f"output exists (use --overwrite): {args.out_root}")
        shutil.rmtree(args.out_root)
    lines = [line for model in inputs.models for line in synthetic_lines(model, guard)]
    print(
        f"inputs asserted: {len(inputs.folds)} folds, {len(lines)} synthetic lines",
        flush=True,
    )
    folds = {
        fold: _run_fold(args, inputs, fold, guard, lines, started)
        for fold in inputs.folds
    }
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "run_identity": run_identity,
        "settings": _settings(args, inputs),
    }
    document |= gate_document(folds)
    end_tree = tree_state()
    run_identity["tree_end"] = end_tree
    judgment = judgment_record(
        sorted(folds),
        run_scope_blockers(folds, inputs.models, args.limit, args.smoke),
        start_tree,
        end_tree,
    )
    if judgment["tree_changes"]:
        logger.error("tree changed during the run: %s", judgment["tree_changes"])
    document["judgment"] = judgment
    selection = document["selection"]
    selection["judgeable"] = judgment["judgeable"]
    status = final_status(
        selection["stop_reason"], judgment["blockers"], judgment["tree_changes"]
    )
    document["status"] = status
    run_identity["opened_paths"] = list(guard.opened)
    run_identity["allow_list_assertion"] = "pass"
    run_identity["finished_at"] = common.local_now()
    document["resources"] = {
        "seconds_total": round(time.time() - started, 1),
        "peak_vram_reserved_mb": perturb._peak_vram_mb(args.device),
    }
    _write_outputs(args.out_root, folds, document)
    print(f"wrote {args.out_root}", flush=True)
    return {
        "status": status,
        "n_conditions": len(CROP_CONDITIONS) + 1 + len(Y1_CUT_PERIODS),
        "n_images": sum(output.n_images for output in folds.values()),
    }


def print_final_line(
    status: str, out_root: Path | None, counts: dict, started: float
) -> None:
    """Closing stdout JSON line; it never carries metric values."""
    line = {
        "status": status,
        "out_dir": None if out_root is None else str(out_root),
        "n_conditions": counts.get("n_conditions"),
        "n_images": counts.get("n_images"),
        "elapsed_s": round(time.time() - started, 1),
    }
    print(json.dumps(line, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry; returns the exit code (0 ok, 1 S1 failed or not judgeable,
    2 refused or tree changed, 3 error)."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    started = time.time()
    args = _parse_args(argv)
    guard = InputGuard()
    counts: dict = {}
    try:
        if args.probe_path is not None:
            guard.check(args.probe_path)
            status = "ok"
        else:
            counts = run(args, guard)
            status = counts["status"]
    except common.InputError as error:
        logger.error("input error: %s", error)
        status = "input_error"
    except Exception:
        logger.exception("internal error")
        status = "error"
    print_final_line(status, args.out_root, counts, started)
    return EXIT_CODES[status]


if __name__ == "__main__":
    sys.exit(main())
