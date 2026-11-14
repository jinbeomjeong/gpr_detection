"""M3-2 S3 (stage B, measurement M1): AIHub perturbation recall of the N2-a' models.

Spec: ``docs/processing/m32_image_quality_plan.md`` v0.2.1 (§3-1 M1, §4-3 B,
§4-4, §5-3 S3) and ``docs/experiments/m32_cv_review.md`` §4-1 (the cv
requirements, the authority for this script).

For every fold k the N2-a' model ``runs/detect/<prefix><k>/weights/last.pt``
(sha256 asserted, no ensemble) sees only its own validation crops, the crop
list ``eval/last_crop/eval_images.txt`` written by ``scripts/eval_detector.py``
(one gain copy per crop). The split manifest is checked first: every crop is
in fold k of the train pool (test seal: zero test crops), its session never
appears on the training side, and all gain copies of a crop sit in fold k.

Each crop is perturbed by up to 16 conditions that mimic the M3-2 recipe steps
on the AIHub grey image (amplitude ``a = q - 127.5``): background (row median
over the crop width, B0 / B2), gain (AGC over ``8 P_A`` rows, G0 / G2), bottom
padding (P0 flat 128 / P1 cosine taper; the height is drawn per crop with a
fixed seed from the 4TU + IDS tile-weighted ``h_pad`` distribution, §1-5),
grey mapping (M1 clip p98 / p99.5, M2 histogram matching, M0 for gain mimics)
and polarity inversion (M3). ``P_A`` = 4.06 px (§1-6). GT boxes whose centre
lies in the padding become ignore boxes: predictions overlapping them
(IoU >= 0.5 or IoA >= 0.5) are dropped before scoring.

Inference goes through ``gpr_detection.detect.yolo`` (conf 0.001, NMS IoU 0.7,
input 256, one image per call; ``ultralytics`` is never imported here).
Scoring is the D1 code (``m31_d1_common``, equal to Ultralytics 8.4.192): per
class AP50 and recall at conf > 0.25 (IoU 0.5), split by gain copy.

Identity gate: the unperturbed condition must reproduce D1 r4
``g1.00_dx+0`` of the same models (same predict path): per-class ``n_gt`` as
expected, nothing ignored, AP50 within 0.01 %p and the ``pipe_hyperbola``
recall equal. The difference to the Ultralytics validator
(``eval/last_crop/metrics.json``) is written as a report item only, because
the predict path and the validator path differ (D1 ``path_difference``).
Before any inference the D1 r4 reference files and the crop lists are
asserted by sha256 ([CVR] appendix A-1), and the gate also requires the GPU
D1 r4 ran on.

Blind output: metrics go to ``<out-root>/fold<k>/metrics.json``,
``objects.csv`` and ``<out-root>/summary.json`` only. Stdout carries counts,
elapsed time, paths and sha256 values; its last line is
``{"status", "out_dir", "n_conditions", "n_images", "elapsed_s"}``.
Exit codes: 0 done, 1 identity gate failed, 2 input refused (weights sha,
split, output exists, dirty tree), 3 internal error.
"""

import argparse
import csv
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common

from gpr_detection.preprocess.filters import agc, agc_rms
from gpr_detection.preprocess.to_image import (
    DetectionImageConfig,
    content_taper_weights,
)

REPO = common.REPO
RUN_PREFIX = "yolo11m_aihubv1n2a2_20261114_fold"
MANIFEST = REPO / "data" / "splits" / "aihub_yolo_v1.csv"
PAD_HEIGHTS_CSV = REPO / "analysis" / "processing" / "m32_content_height_lines.csv"
HIST_CDF_JSON = REPO / "analysis" / "processing" / "m32_aihub_nm_cdf.json"
DEFAULT_OUT_ROOT = REPO / "runs" / "eval" / "m32_aihub_perturb"
SMOKE_OUT_ROOT = REPO / "runs" / "eval" / "m32_aihub_perturb_smoke"
SCHEMA = "m32_aihub_perturb/1"

# N2-a' baseline (docs/experiments/2026-11-14_n2a2_baseline.md §1), folds 0-4.
EXPECTED_WEIGHTS_SHA256 = (
    "7ead71c7fa62d85edb11f5614e78b11619062b455dca62f5cc66c93b2dcf8d92",
    "f4b7e0443799d66983adb21356a390bf5cf68ec98964e9425f0981cd3b079b15",
    "32c0e854ed7e6ac6e4847265f956f038030bb968f4d01ab8a550a40e2d98a99b",
    "0238d4d62722ff7318e8c8fa8c283471b008f01905e95d326510547882b74de8",
    "98922d2e79b547fd7cf4714b400e693b131d9082c58d293b9e0f5163e039c6b2",
)
# Gate reference files (runs/, outside git), fixed by [CVR] appendix A-1.
EXPECTED_D1_REFERENCE_SHA256 = (
    "1c0d070852bfa7adb4c27740889228d2d0852083e4ff13a91de973b52d435767",
    "a9ed73ecad1437eddb278426401f5780fafbe4fecfb3de5d2507eabbaddc85a1",
    "98b637b1ee8f2fab78072da9c366f0a235d47af46c80929244cdde45f86b8e58",
    "d205d8e178bb4fd070dbeb78978697cec8d26047293cf59d322a4057f80e7546",
    "960d4021393a6e76ded3e7d7086bdf3289e54ce9fe9af42bf6a5fe5403b06fd2",
)
D1_R4_VERDICT = REPO / "analysis" / "detection" / "m31_d1_r4_verdict.json"
EXPECTED_N_IMAGES_TOTAL = 5473
D1_REFERENCE_SUBDIR = "m31_d1_r4"
D1_REFERENCE_CONDITION = "g1.00_dx+0"
VALIDATOR_SUBDIR = "last_crop"

SEED = 20261114
PERIOD_AIHUB_PX = 4.06  # P_A, AIHub tpipe apex crop carrier period p50 (§1-6)
NS_PER_S = 1e9
GRAY_CENTER = 127.5  # zero amplitude before rounding, as in to_image
PAD_GRAY = 128
GRAY_MAX = 255
PAD_DATASETS = ("4tu", "gz_ids")
EXPECTED_PAD_TILES = 14749  # 4TU + IDS tile-weighted distribution (§1-5)
MAX_CONDITIONS = 16
MAX_STAGE_A_RECIPES = 4
CONF_AP = 0.001
NMS_IOU = 0.7
IGNORE_OVERLAP = 0.5
GATE_AP50_TOL = 1e-4  # 0.01 %p as a fraction
GATE_RECALL_TOL = 1e-12
GATE_RECALL_CLASS = "pipe_hyperbola"
SMOKE_LIMIT = 8
IDENTITY = "identity"
GAINS = ("0.25", "0.5", "1.0")
EXIT_CODES = {"ok": 0, "gate_failed": 1, "input_error": 2, "error": 3}
DISCLOSURE = {
    "m2_reference_cdf": (
        "train_pool NM (pipe_hyperbola, pipe_band) gain 1.0, 600 crops, labels "
        "unused, test 0. It includes validation crops of all 5 folds (per fold "
        "170/110/112/104/104; each fold's own crops are 17-28 % of the "
        "reference), so the M2 recall drop may be slightly underestimated "
        "([CVR] A-2)."
    ),
    "units": (
        "Padding heights are px of the real-line tile distribution (P* 6, "
        "14,749 tiles, plan §1-5); taper length and AGC windows are in "
        "P_A = 4.06 px (plan §1-6)."
    ),
    "paired_recall": (
        "Main paired recall pairs each condition with the identity on the "
        "objects both keep (condition ∩ identity); conditions without padding "
        "therefore use all objects, and the polarity threshold (plan §4-3 B) "
        "reads this value. paired_recall_all_common (objects kept by every "
        "condition) is report only ([CVR] A-3 (1))."
    ),
    "gate": (
        "Identity gate against D1 r4 g1.00_dx+0 (same predict path, same GPU); "
        "the validator difference is report only ([CVR] A-1). Judgement uses "
        "runs with settings.gated true and n_images_total 5,473 only."
    ),
}
"""Statements the stage-B report must carry ([CVR] appendix A)."""
OBJECTS_HEADER = (
    "fold",
    "condition",
    "image",
    "gain",
    "class",
    "gt_index",
    "x_center",
    "y_center",
    "w",
    "h",
    "hit_conf025",
    "score",
)

logger = logging.getLogger("m32_aihub_perturb")


class GateFailedError(Exception):
    """The identity condition did not reproduce the recorded D1 r4 values.

    Carries the fold's partial result so the gate details reach the files.
    """

    def __init__(self, fold: int, fold_result: dict, object_rows: list) -> None:
        super().__init__(f"fold {fold}")
        self.fold = fold
        self.fold_result = fold_result
        self.object_rows = object_rows


# ----------------------------------------------------------------- conditions


class BackgroundMimic(StrEnum):
    """Step (4) mimic: subtract ``alpha`` times the row median over the crop."""

    NONE = "none"
    ROW_MEDIAN = "row_median"


class GainMimic(StrEnum):
    """Step (5) mimic: AGC (G0 / G1) or capped AGC (G2) along the rows."""

    NONE = "none"
    AGC = "agc"
    AGC_CAPPED = "agc_capped"


class PadMimic(StrEnum):
    """Bottom padding with the drawn height: none, P0 flat or P1 taper."""

    NONE = "none"
    FLAT = "flat"
    TAPER = "taper"


class GreyMimic(StrEnum):
    """Amplitude -> grey: keep the AIHub scale, M0 / M1 symmetric clip, M2."""

    KEEP = "keep"
    SYMMETRIC = "symmetric"
    HIST = "hist"


@dataclass(frozen=True)
class Perturbation:
    """One stage-B condition, applied in the order B -> G -> M -> P -> polarity.

    Attributes:
        name: Condition name used in every output.
        remove_trace_mean: Subtract each column's mean over the content rows
            first, like the DC removal of step (1). Set for the gain-only
            conditions: the NM centre grey is 125, not 127.5, and an AGC
            would amplify that offset into artificial shading ([CVR] A).
        background: :class:`BackgroundMimic`.
        background_alpha: ``alpha`` of ``a - alpha med_row(a)`` (B0 1, B2 < 1).
        gain: :class:`GainMimic`.
        agc_window_periods: AGC window in ``P_A`` rows (G0 8, G1 32).
        agc_floor_beta: ``beta`` of the capped AGC (G2).
        pad: :class:`PadMimic`.
        pad_taper_periods: ``n_tau`` of P1 in ``P_A``.
        mapping: :class:`GreyMimic`.
        clip_percentile: ``|a|`` percentile of the symmetric mapping.
        invert_polarity: M3: content grey ``q`` becomes ``255 - q``.
        recipe_id: Stage-A recipe this condition mimics, if any.
    """

    name: str
    remove_trace_mean: bool = False
    background: BackgroundMimic = BackgroundMimic.NONE
    background_alpha: float = 1.0
    gain: GainMimic = GainMimic.NONE
    agc_window_periods: float = 8.0
    agc_floor_beta: float = 0.5
    pad: PadMimic = PadMimic.NONE
    pad_taper_periods: float = 1.0
    mapping: GreyMimic = GreyMimic.KEEP
    clip_percentile: float = 99.0
    invert_polarity: bool = False
    recipe_id: str | None = None

    def __post_init__(self) -> None:
        if self.gain != GainMimic.NONE and self.mapping == GreyMimic.KEEP:
            # A gained section has no grey scale of its own; it must be mapped.
            raise ValueError(f"{self.name}: a gain mimic needs a grey mapping")
        if not 0.0 < self.background_alpha <= 1.0:
            raise ValueError(f"{self.name}: background_alpha must be in (0, 1]")

    @property
    def is_identity(self) -> bool:
        return self == Perturbation(self.name)

    @property
    def uses_pad(self) -> bool:
        return self.pad != PadMimic.NONE


BASE_CONDITIONS = (
    Perturbation(IDENTITY),
    Perturbation("pad_p0", pad=PadMimic.FLAT),
    Perturbation("pad_p1_n1", pad=PadMimic.TAPER, pad_taper_periods=1.0),
    Perturbation("pad_p1_n2", pad=PadMimic.TAPER, pad_taper_periods=2.0),
    Perturbation("map_m1_p98", mapping=GreyMimic.SYMMETRIC, clip_percentile=98.0),
    Perturbation("map_m1_p99.5", mapping=GreyMimic.SYMMETRIC, clip_percentile=99.5),
    Perturbation("map_m2_hist", mapping=GreyMimic.HIST),
    Perturbation("polarity_m3", invert_polarity=True),
    Perturbation("bg_b0_row_median", background=BackgroundMimic.ROW_MEDIAN),
    Perturbation(
        "bg_b2_a0.5", background=BackgroundMimic.ROW_MEDIAN, background_alpha=0.5
    ),
    Perturbation(
        "gain_g0_agc8",
        remove_trace_mean=True,
        gain=GainMimic.AGC,
        mapping=GreyMimic.SYMMETRIC,
    ),
    Perturbation(
        "gain_g2_b0.5",
        remove_trace_mean=True,
        gain=GainMimic.AGC_CAPPED,
        mapping=GreyMimic.SYMMETRIC,
    ),
)
"""Conditions (1)-(12) of [CVR] §4-1; (13)-(16) come from stage A."""

_R0_MIMIC = Perturbation(
    "recipe_r0",
    background=BackgroundMimic.ROW_MEDIAN,
    gain=GainMimic.AGC,
    pad=PadMimic.FLAT,
    mapping=GreyMimic.SYMMETRIC,
    recipe_id="r0",
)
_G2_P1_M0 = Perturbation(
    "recipe",
    gain=GainMimic.AGC_CAPPED,
    pad=PadMimic.TAPER,
    mapping=GreyMimic.SYMMETRIC,
)
RECIPE_MIMICS: dict[str, Perturbation] = {
    "r0": _R0_MIMIC,
    "r1": replace(_G2_P1_M0, name="recipe_r1", recipe_id="r1"),
    "r2": replace(
        _G2_P1_M0,
        name="recipe_r2",
        background=BackgroundMimic.ROW_MEDIAN,
        background_alpha=0.5,
        recipe_id="r2",
    ),
    "r5": replace(
        _G2_P1_M0,
        name="recipe_r5",
        background=BackgroundMimic.ROW_MEDIAN,
        recipe_id="r5",
    ),
    "r6": replace(
        _R0_MIMIC,
        name="recipe_r6",
        background_alpha=0.8,
        agc_window_periods=32.0,
        recipe_id="r6",
    ),
    "r7": replace(_G2_P1_M0, name="recipe_r7", mapping=GreyMimic.HIST, recipe_id="r7"),
}
"""Stage-A recipes of plan §4-1 as AIHub mimics (B, G, P, M in that table)."""

UNSUPPORTED_RECIPES = {
    "r3": "B3 gates on t_z, which cannot be located at the AIHub crop top ([AF])",
    "r4": "G3 multiplies by record time t^n, unknown on AIHub crops",
}


def build_conditions(stage_a_recipes: Sequence[str] = ()) -> tuple[Perturbation, ...]:
    """The 12 base conditions plus up to four stage-A recipe mimics.

    Raises:
        common.InputError: For an unknown, unsupported or repeated recipe, or
            more than :data:`MAX_CONDITIONS` conditions.
    """
    recipes = list(stage_a_recipes)
    if len(recipes) != len(set(recipes)):
        raise common.InputError(f"repeated stage-A recipe in {recipes}")
    if len(recipes) > MAX_STAGE_A_RECIPES:
        raise common.InputError(f"at most {MAX_STAGE_A_RECIPES} stage-A recipes")
    for recipe in recipes:
        if recipe in UNSUPPORTED_RECIPES:
            reason = UNSUPPORTED_RECIPES[recipe]
            raise common.InputError(f"recipe {recipe} has no AIHub mimic: {reason}")
        if recipe not in RECIPE_MIMICS:
            raise common.InputError(f"unknown stage-A recipe {recipe!r}")
    conditions = (*BASE_CONDITIONS, *(RECIPE_MIMICS[r] for r in recipes))
    if len(conditions) > MAX_CONDITIONS:
        raise common.InputError(f"{len(conditions)} conditions > {MAX_CONDITIONS}")
    return conditions


def conditions_sha256(conditions: Sequence[Perturbation]) -> str:
    """sha256 of the canonical JSON of the condition list (order included)."""
    payload = json.dumps([asdict(c) for c in conditions], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# -------------------------------------------------------------- perturbation


def symmetric_grey_levels(amplitudes: np.ndarray, clip_percentile: float) -> np.ndarray:
    """M0 / M1: ``127.5 + 127.5 clip(a / c, -1, 1)``, ``c`` the ``|a|`` percentile.

    Same rule as ``to_image._symmetric_gray_levels``; a zero clip level gives
    the padding grey everywhere, as a flat line does in ``to_image``.
    """
    clip_level = float(np.percentile(np.abs(amplitudes), clip_percentile))
    if clip_level == 0:
        return np.full(amplitudes.shape, float(PAD_GRAY))
    scaled = np.clip(amplitudes / clip_level, -1.0, 1.0)
    return GRAY_CENTER + GRAY_CENTER * scaled


def hist_matched_grey_levels(
    values: np.ndarray, reference_cdf: np.ndarray
) -> np.ndarray:
    """M2: the grey whose reference CDF first reaches each value's mid-rank quantile.

    Same rule as ``to_image._hist_matched_gray_levels`` (equal values share a
    grey, the amplitude order is kept).
    """
    flat = values.ravel()
    ordered = np.sort(flat)
    below = np.searchsorted(ordered, flat, side="left")
    up_to = np.searchsorted(ordered, flat, side="right")
    quantile = (below + up_to) / (2.0 * flat.size)
    levels = np.searchsorted(reference_cdf, quantile, side="left")
    return np.minimum(levels, GRAY_MAX).astype(np.float64).reshape(values.shape)


def _remove_row_background(amplitudes: np.ndarray, alpha: float) -> np.ndarray:
    row_median = np.median(amplitudes, axis=1, keepdims=True)
    return amplitudes - alpha * row_median


def _apply_gain(amplitudes: np.ndarray, condition: Perturbation) -> np.ndarray:
    # Rows act as 1 ns samples with f_c = 1 / P_A, so the window is
    # window_periods * P_A rows (8 P_A = 32.5 -> 33 odd rows).
    f_c_hz = NS_PER_S / PERIOD_AIHUB_PX
    window = condition.agc_window_periods
    if condition.gain == GainMimic.AGC:
        return agc(amplitudes, 1.0, f_c_hz, window_periods=window)
    rms = agc_rms(amplitudes, 1.0, f_c_hz, window_periods=window)
    floor = condition.agc_floor_beta * np.median(rms, axis=0, keepdims=True)
    return amplitudes / np.maximum(rms, floor)


def _grey_levels(
    amplitudes: np.ndarray, condition: Perturbation, reference_cdf: np.ndarray | None
) -> np.ndarray:
    if condition.mapping == GreyMimic.SYMMETRIC:
        return symmetric_grey_levels(amplitudes, condition.clip_percentile)
    if condition.mapping == GreyMimic.HIST:
        if reference_cdf is None:
            raise ValueError(f"{condition.name} needs the reference CDF")
        return hist_matched_grey_levels(amplitudes, reference_cdf)
    return np.clip(GRAY_CENTER + amplitudes, 0.0, float(GRAY_MAX))


def _taper_to_pad(levels: np.ndarray, taper_px: float) -> np.ndarray:
    weights = content_taper_weights(levels.shape[0], taper_px)
    rows = weights < 1.0  # rows above the taper keep their exact levels
    levels[rows] = PAD_GRAY + weights[rows, None] * (levels[rows] - PAD_GRAY)
    return levels


def perturb_image(
    image: np.ndarray,
    condition: Perturbation,
    pad_height_px: int = 0,
    reference_cdf: np.ndarray | None = None,
) -> np.ndarray:
    """Return the perturbed uint8 crop of one condition (input unchanged).

    The bottom ``pad_height_px`` rows (only for padding conditions) end the
    record: background, gain and mapping see the content rows only, as in
    ``to_image`` where they never see the padding.

    Args:
        image: uint8 ``[H, W]`` AIHub crop.
        condition: The condition to apply.
        pad_height_px: Drawn padding height ``h_pad`` (< H).
        reference_cdf: 256-value grey CDF for :attr:`GreyMimic.HIST`.
    """
    if condition.is_identity:
        return image.copy()
    height = image.shape[0]
    pad_px = pad_height_px if condition.uses_pad else 0
    if not 0 <= pad_px < height:
        raise ValueError(f"pad height {pad_px} outside [0, {height})")
    amplitudes = image[: height - pad_px].astype(np.float64) - GRAY_CENTER
    if condition.remove_trace_mean:
        amplitudes -= amplitudes.mean(axis=0, keepdims=True)
    if condition.background == BackgroundMimic.ROW_MEDIAN:
        amplitudes = _remove_row_background(amplitudes, condition.background_alpha)
    if condition.gain != GainMimic.NONE:
        amplitudes = _apply_gain(amplitudes, condition)
    levels = _grey_levels(amplitudes, condition, reference_cdf)
    if condition.pad == PadMimic.TAPER:
        levels = _taper_to_pad(levels, condition.pad_taper_periods * PERIOD_AIHUB_PX)
    content = np.rint(levels).astype(np.uint8)
    if condition.invert_polarity:
        content = GRAY_MAX - content  # the padding keeps its grey
    perturbed = np.full_like(image, PAD_GRAY)
    perturbed[: height - pad_px] = content
    return perturbed


def load_pad_heights(path: Path) -> np.ndarray:
    """Sorted tile-weighted ``h_pad`` values of 4TU + IDS (one per tile, §1-5).

    Raises:
        common.InputError: If the file is missing or the tile count is not the
            documented 14,749.
    """
    if not path.is_file():
        raise common.InputError(f"pad height table missing: {path}")
    heights: list[int] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] in PAD_DATASETS:
                heights.extend([int(row["pad_px"])] * int(row["n_tiles"]))
    if len(heights) != EXPECTED_PAD_TILES:
        raise common.InputError(
            f"{path.name}: {len(heights)} tiles, expected {EXPECTED_PAD_TILES}"
        )
    return np.sort(np.asarray(heights, dtype=int))


def draw_pad_height(stem: str, pad_heights: np.ndarray, seed: int = SEED) -> int:
    """Padding height of one crop, fixed by the seed and the crop name only."""
    stem_key = int.from_bytes(hashlib.sha256(stem.encode("utf-8")).digest()[:8])
    rng = np.random.default_rng([seed, stem_key])
    return int(pad_heights[rng.integers(len(pad_heights))])


def load_reference_cdf(path: Path) -> np.ndarray:
    """The M2 reference CDF (``cdf``, 256 values), validated like the S1 config.

    Raises:
        common.InputError: If the file is missing or the CDF is malformed.
    """
    if not path.is_file():
        raise common.InputError(f"M2 reference CDF missing: {path}")
    cdf = json.loads(path.read_text(encoding="utf-8")).get("cdf")
    try:
        DetectionImageConfig(
            gray_mapping="hist_aihub",
            hist_reference_cdf=tuple(cdf or ()),
            recipe_id="m2_cdf_check",
        )
    except ValueError as error:
        raise common.InputError(f"{path.name}: {error}") from error
    return np.asarray(cdf, dtype=np.float64)


# ------------------------------------------------------------- split checks


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Rows of the split manifest ``aihub_yolo_v1.csv``."""
    if not path.is_file():
        raise common.InputError(f"split manifest missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_fold_split(
    manifest: Sequence[dict[str, str]], eval_names: Sequence[str], fold: int
) -> dict:
    """Leak and test-seal assertions of one fold's crop list ([CVR] §4-1).

    Args:
        manifest: Manifest rows (``file_name``, ``session``, ``crop_key``,
            ``subset``, ``fold``).
        eval_names: Image file names of the fold's crop list.
        fold: The fold whose model sees these crops.

    Returns:
        ``{"checks": {name: bool}, "counts": {...}}``; all checks must hold.
    """
    by_name = {row["file_name"]: row for row in manifest}
    rows = [by_name[name] for name in eval_names if name in by_name]
    sessions = {row["session"] for row in rows}
    crop_keys = {row["crop_key"] for row in rows}
    train_sessions = {
        row["session"]
        for row in manifest
        if row["subset"] == "trainpool" and int(row["fold"]) != fold
    }
    copies = [row for row in manifest if row["crop_key"] in crop_keys]
    n_test = sum(row["subset"] == "test" for row in rows)
    checks = {
        "all_in_manifest": len(rows) == len(eval_names),
        "no_duplicates": len(set(eval_names)) == len(eval_names),
        "test_seal_zero_test_crops": n_test == 0,
        "all_in_this_fold": all(_is_fold(row, fold) for row in rows),
        "sessions_absent_from_training": not (sessions & train_sessions),
        "gain_copies_same_side": all(_is_fold(row, fold) for row in copies),
    }
    counts = {
        "n_images": len(eval_names),
        "n_test": n_test,
        "n_sessions": len(sessions),
        "n_gain_copies_in_manifest": len(copies),
    }
    return {"checks": checks, "counts": counts}


def _is_fold(row: dict[str, str], fold: int) -> bool:
    return row["subset"] == "trainpool" and int(row["fold"]) == fold


def gain_by_name(manifest: Sequence[dict[str, str]]) -> dict[str, str]:
    """Manifest gain copy (``"0.25"``, ``"0.5"``, ``"1.0"``) of every image file."""
    return {row["file_name"]: row["gain"] for row in manifest}


# ------------------------------------------------------------------- samples


@dataclass
class Sample:
    """One validation crop with its labels, gain copy and drawn pad height."""

    stem: str
    gain: str
    image: np.ndarray  # uint8 [256, 256]
    boxes: np.ndarray  # [n, 4] xyxy px, clipped to the crop
    classes: np.ndarray  # [n]
    gt_index: np.ndarray  # [n] label line numbers
    pad_height_px: int


def read_eval_names(eval_list: Path) -> list[str]:
    """Image file names of an ``eval_images.txt`` list, in list order."""
    lines = eval_list.read_text(encoding="utf-8").splitlines()
    return [Path(line.strip()).name for line in lines if line.strip()]


def read_samples(
    eval_list: Path,
    gains: dict[str, str],
    pad_heights: np.ndarray,
    label_dir: Path,
    limit: int | None,
) -> list[Sample]:
    """Load the crops of a fold list with labels, gain and pad height.

    Raises:
        common.InputError: If an image is unreadable or not 256 x 256.
    """
    samples = []
    for line in eval_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        path = Path(line.strip())
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (common.CANVAS_PX, common.CANVAS_PX):
            raise common.InputError(f"bad image {path}")
        boxes, classes, gt_index = common.read_yolo_label_indexed(
            label_dir / f"{path.stem}.txt"
        )
        pad = draw_pad_height(path.stem, pad_heights)
        samples.append(
            Sample(path.stem, gains[path.name], image, boxes, classes, gt_index, pad)
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def split_ground_truth(
    sample: Sample, content_height_px: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kept GT (centre above the padding) and ignore boxes (centre in it).

    Returns:
        ``(kept_boxes, kept_classes, kept_gt_index, ignore_boxes)``.
    """
    centre_y = (sample.boxes[:, 1] + sample.boxes[:, 3]) / 2
    is_kept = centre_y < content_height_px
    return (
        sample.boxes[is_kept],
        sample.classes[is_kept],
        sample.gt_index[is_kept],
        sample.boxes[~is_kept].reshape(-1, 4),
    )


# ------------------------------------------------------------------- scoring


def _ioa_of_predictions(predictions: np.ndarray, regions: np.ndarray) -> np.ndarray:
    top_left = np.maximum(predictions[:, None, :2], regions[None, :, :2])
    bottom_right = np.minimum(predictions[:, None, 2:], regions[None, :, 2:])
    intersection = (bottom_right - top_left).clip(0.0).prod(axis=2)
    area = (predictions[:, 2:] - predictions[:, :2]).prod(axis=1).clip(1e-9)
    return intersection / area[:, None]


def drop_ignored(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, ignore: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop predictions with IoU >= 0.5 or IoA >= 0.5 on an ignore box (D1 rule)."""
    if not (len(ignore) and len(boxes)):
        return boxes, scores, classes
    overlaps = np.maximum(
        common.box_iou(boxes, ignore).max(axis=1),
        _ioa_of_predictions(boxes, ignore).max(axis=1),
    )
    keep = overlaps < IGNORE_OVERLAP
    return boxes[keep], scores[keep], classes[keep]


def predict(detector, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boxes, scores and classes of one 256 x 256 image (identity letterbox)."""
    output = detector.predict(image)
    transform = output.transform
    is_identity = (transform.scale_x, transform.scale_y) == (1, 1) and (
        transform.pad_x,
        transform.pad_y,
    ) == (0, 0)
    if not is_identity:
        raise RuntimeError(f"letterbox not identity: {transform}")
    if not output.detections:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    boxes = np.array([d.box.xyxy for d in output.detections], dtype=float)
    scores = np.array([d.score for d in output.detections], dtype=float)
    classes = np.array([d.class_id for d in output.detections], dtype=int)
    return boxes, scores, classes


def _class_table(accumulator: common.ScoreAccumulator, n_ignored: np.ndarray) -> dict:
    table = {}
    for class_id, (name, scored) in enumerate(accumulator.class_results().items()):
        table[name] = {**scored, "n_ignored": int(n_ignored[class_id])}
    return table


def _map50(classes: dict) -> float | None:
    aps = [row["ap50"] for row in classes.values() if row["ap50"] is not None]
    return float(np.mean(aps)) if aps else None


def _object_rows(
    fold: int, condition: str, sample: Sample, kept: tuple, hit_score: np.ndarray
) -> list[list]:
    boxes, classes, gt_index = kept
    rows = []
    for box, class_id, index, score in zip(
        boxes, classes, gt_index, hit_score, strict=True
    ):
        rows.append(
            [
                fold,
                condition,
                sample.stem,
                sample.gain,
                common.CLASS_NAMES[class_id],
                int(index),
                round((box[0] + box[2]) / 2, 2),
                round((box[1] + box[3]) / 2, 2),
                round(box[2] - box[0], 2),
                round(box[3] - box[1], 2),
                int(score >= 0),
                round(float(score), 4),
            ]
        )
    return rows


def evaluate_condition(
    detector,
    samples: Sequence[Sample],
    condition: Perturbation,
    reference_cdf: np.ndarray | None,
    fold: int,
    object_rows: list,
) -> dict:
    """Run one condition on one fold and score it (values go to files only)."""
    overall = common.ScoreAccumulator()
    by_gain = {gain: common.ScoreAccumulator() for gain in GAINS}
    n_classes = len(common.CLASS_NAMES)
    ignored_by_gain = {gain: np.zeros(n_classes, dtype=int) for gain in GAINS}
    for sample in samples:
        pad_px = sample.pad_height_px if condition.uses_pad else 0
        image = perturb_image(sample.image, condition, pad_px, reference_cdf)
        boxes, classes, gt_index, ignore = split_ground_truth(
            sample, common.CANVAS_PX - pad_px
        )
        ignored_by_gain[sample.gain] += np.bincount(
            sample.classes, minlength=n_classes
        ) - np.bincount(classes, minlength=n_classes)
        predictions = drop_ignored(*predict(detector, image), ignore)
        hit_score = overall.add_image(*predictions, boxes, classes)
        by_gain[sample.gain].add_image(*predictions, boxes, classes)
        object_rows.extend(
            _object_rows(
                fold, condition.name, sample, (boxes, classes, gt_index), hit_score
            )
        )
    classes_table = _class_table(overall, sum(ignored_by_gain.values()))
    gains_table = {}
    for gain, accumulator in by_gain.items():
        gain_classes = _class_table(accumulator, ignored_by_gain[gain])
        gains_table[gain] = {"classes": gain_classes, "map50": _map50(gain_classes)}
    return {
        "classes": classes_table,
        "map50": _map50(classes_table),
        "by_gain": gains_table,
    }


# ---------------------------------------------------------------------- gate


def check_identity_gate(
    identity: dict,
    d1_reference: dict,
    validator_reference: dict,
    expected_n_gt: dict[str, int],
    run_gpu: str | None = None,
    reference_gpu: str | None = None,
) -> dict:
    """Identity gate of one fold ([CVR] §4-1 정체 조건, predict-path form, A-1).

    Args:
        identity: ``classes`` table of the identity condition.
        d1_reference: ``classes`` table of D1 r4 ``g1.00_dx+0`` (same models,
            same predict path).
        validator_reference: Class name -> AP50 of ``eval/last_crop``
            (Ultralytics validator); reported, not gated.
        expected_n_gt: Class name -> expected ``n_gt`` of this fold.
        run_gpu: Device name of this run (``run_identity.gpu``).
        reference_gpu: Device name recorded by D1 r4. The 1e-12 recall
            tolerance holds on the same device only, so they must match.

    Returns:
        ``{"pass": bool, "checks": [...], "validator_ap50_diff_pp": {...}}``.
    """
    checks = [_check("same_gpu_as_d1_r4", None, run_gpu == reference_gpu)]
    for name, row in identity.items():
        reference = d1_reference[name]
        checks.append(_check("n_gt", name, row["n_gt"] == expected_n_gt[name]))
        checks.append(_check("n_ignored_zero", name, row["n_ignored"] == 0))
        checks.append(
            _check(
                "ap50_equals_d1_r4",
                name,
                _close(row["ap50"], reference["ap50"], GATE_AP50_TOL),
            )
        )
        if name == GATE_RECALL_CLASS:
            is_equal = _close(
                row["recall_conf025"], reference["recall_conf025"], GATE_RECALL_TOL
            )
            checks.append(_check("recall_equals_d1_r4", name, is_equal))
    validator_diff = {
        name: _difference_pp(row["ap50"], validator_reference.get(name))
        for name, row in identity.items()
    }
    return {
        "pass": all(check["pass"] for check in checks),
        "checks": checks,
        "validator_ap50_diff_pp": validator_diff,
    }


def _check(kind: str, class_name: str | None, passed: bool) -> dict:
    return {"check": kind, "class": class_name, "pass": bool(passed)}


def _close(value: float | None, reference: float | None, tolerance: float) -> bool:
    if value is None or reference is None:
        return value is None and reference is None
    return abs(value - reference) <= tolerance


def _difference_pp(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return 100.0 * (value - reference)


@dataclass(frozen=True)
class GateReference:
    """What the identity gate of one fold compares against.

    Attributes:
        d1_classes: ``classes`` table of D1 r4 ``g1.00_dx+0``.
        d1_gpu: ``run_identity.gpu`` of the D1 r4 file.
        validator_ap50: Class name -> AP50 of ``eval/last_crop`` (report only).
        sha256: sha256 of the two reference files.
    """

    d1_classes: dict
    d1_gpu: str | None
    validator_ap50: dict[str, float | None]
    sha256: dict[str, str]


def d1_reference_path(run_dir: Path) -> Path:
    """The D1 r4 metrics file of one fold run folder."""
    return run_dir / "eval" / D1_REFERENCE_SUBDIR / "metrics.json"


def read_gate_reference(run_dir: Path) -> GateReference:
    """Read the gate reference files of one fold.

    Raises:
        common.InputError: If a reference file or key is missing.
    """
    d1_path = d1_reference_path(run_dir)
    validator_path = run_dir / "eval" / VALIDATOR_SUBDIR / "metrics.json"
    try:
        d1 = json.loads(d1_path.read_text(encoding="utf-8"))
        d1_classes = d1["conditions"][D1_REFERENCE_CONDITION]["classes"]
        d1_gpu = d1["run_identity"]["gpu"]
        validator = json.loads(validator_path.read_text(encoding="utf-8"))
        validator_ap50 = {row["name"]: row["ap50"] for row in validator["per_class"]}
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise common.InputError(
            f"gate reference unreadable in {run_dir}: {error!r}"
        ) from error
    hashes = {
        "d1_r4_metrics": common.sha256_file(d1_path),
        "validator_metrics": common.sha256_file(validator_path),
    }
    return GateReference(d1_classes, d1_gpu, validator_ap50, hashes)


def read_expected_eval_list_sha256(verdict_path: Path) -> dict[int, str]:
    """Fold -> ``eval_list_sha256`` recorded by the committed D1 r4 verdict (G2).

    Raises:
        common.InputError: If the verdict file or its G2 records are missing.
    """
    try:
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        return {
            int(record["fold"]): record["eval_list_sha256"]
            for record in verdict["gates"]["G2"]
        }
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise common.InputError(
            f"D1 r4 verdict unreadable: {verdict_path}: {error!r}"
        ) from error


# ---------------------------------------------------------------- aggregation


def mean_and_sd(values: Sequence[float | None]) -> dict:
    """Unweighted mean and sample SD (ddof 1) over the non-missing values."""
    present = [float(v) for v in values if v is not None]
    mean = float(np.mean(present)) if present else None
    sd = float(np.std(present, ddof=1)) if len(present) > 1 else None
    return {"n": len(present), "mean": mean, "sd": sd}


def _hits_by_condition(
    object_rows: Iterable[Sequence], condition_names: Sequence[str]
) -> dict[str, dict[tuple, int]]:
    """Condition -> {(image, gt_index, class): hit at conf > 0.25 (0 / 1)}."""
    column = {name: i for i, name in enumerate(OBJECTS_HEADER)}
    hits: dict[str, dict[tuple, int]] = {name: {} for name in condition_names}
    for row in object_rows:
        key = (row[column["image"]], row[column["gt_index"]], row[column["class"]])
        hits[row[column["condition"]]][key] = int(row[column["hit_conf025"]])
    return hits


def _recall_pair(
    hits: dict[tuple, int], identity_hits: dict[tuple, int], keys: set[tuple]
) -> dict:
    """Per class: recall of a condition and of the identity on the same objects."""
    result = {}
    for class_name in common.CLASS_NAMES:
        class_keys = [key for key in keys if key[2] == class_name]
        row: dict = {"n_common": len(class_keys), "recall": None}
        row |= {"identity_recall": None, "delta_vs_identity": None}
        if class_keys:
            row["recall"] = sum(hits[key] for key in class_keys) / len(class_keys)
            identity = sum(identity_hits[key] for key in class_keys) / len(class_keys)
            row["identity_recall"] = identity
            row["delta_vs_identity"] = row["recall"] - identity
        result[class_name] = row
    return result


def paired_recall(
    object_rows: Iterable[Sequence], condition_names: Sequence[str]
) -> dict:
    """Main paired recall: each condition against the identity on the objects
    both keep (condition ∩ identity, [CVR] A-3 (1)).

    A condition without padding is therefore paired on every object, so the
    polarity threshold (plan §4-3 B) reads all objects.

    Args:
        object_rows: ``objects.csv`` rows of one fold (:data:`OBJECTS_HEADER`).
        condition_names: All conditions of the run; the first is the identity.

    Returns:
        ``{condition: {class: {"n_common", "recall", "identity_recall",
        "delta_vs_identity"}}}``.
    """
    hits = _hits_by_condition(object_rows, condition_names)
    identity_hits = hits[condition_names[0]]
    return {
        name: _recall_pair(
            hits[name], identity_hits, set(hits[name]) & set(identity_hits)
        )
        for name in condition_names
    }


def paired_recall_all_common(
    object_rows: Iterable[Sequence], condition_names: Sequence[str]
) -> dict:
    """Report only: the same pairing on the objects kept by every condition."""
    hits = _hits_by_condition(object_rows, condition_names)
    common_keys = set.intersection(*(set(h) for h in hits.values()))
    identity_hits = hits[condition_names[0]]
    return {
        name: _recall_pair(hits[name], identity_hits, common_keys)
        for name in condition_names
    }


def aggregate_folds(
    fold_results: dict[int, dict], condition_names: Sequence[str]
) -> dict:
    """Per condition and class: fold values, mean and SD of AP50, recall and the
    paired common-set recall difference to the identity condition."""
    aggregate: dict[str, dict] = {}
    for name in condition_names:
        aggregate[name] = {"classes": {}, "map50": None}
        map50_values = [r["conditions"][name]["map50"] for r in fold_results.values()]
        aggregate[name]["map50"] = mean_and_sd(map50_values)
        for class_name in common.CLASS_NAMES:
            aggregate[name]["classes"][class_name] = _class_aggregate(
                fold_results, name, class_name
            )
    return aggregate


def _class_aggregate(fold_results: dict[int, dict], name: str, class_name: str) -> dict:
    rows = [r["conditions"][name]["classes"][class_name] for r in fold_results.values()]
    paired = [r["paired_recall"][name][class_name] for r in fold_results.values()]
    all_common = [
        r["paired_recall_all_common"][name][class_name] for r in fold_results.values()
    ]
    return {
        "ap50": mean_and_sd([row["ap50"] for row in rows]),
        "recall_conf025": mean_and_sd([row["recall_conf025"] for row in rows]),
        "paired_recall_delta": mean_and_sd([p["delta_vs_identity"] for p in paired]),
        "paired_recall_delta_all_common": mean_and_sd(
            [p["delta_vs_identity"] for p in all_common]
        ),
        "n_ignored_total": sum(row["n_ignored"] for row in rows),
    }


# ----------------------------------------------------------------- identity


def tracked_tree_problems(
    required_paths: Sequence[Path], repo: Path = REPO
) -> list[str]:
    """Modified tracked files and required inputs that git does not track.

    Untracked files of other work do not count: they cannot change this run
    unless they are one of ``required_paths``.
    """
    problems = []
    try:
        status = _git(repo, "status", "--porcelain", "--untracked-files=no")
        problems += [f"modified: {line}" for line in status.splitlines() if line]
        for path in required_paths:
            relative = path.resolve().relative_to(repo.resolve()).as_posix()
            if not _git(repo, "ls-files", "--", relative).strip():
                problems.append(f"not tracked: {relative}")
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        problems.append(f"git state unknown: {error!r}")
    return problems


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=common.GIT_TIMEOUT_S,
        check=True,
    )
    return completed.stdout


def _peak_vram_mb(device: str) -> float | None:
    if device.strip().lower() == "cpu":
        return None
    import torch

    return round(torch.cuda.max_memory_reserved() / 2**20, 1)


# ------------------------------------------------------------------------ run


@dataclass(frozen=True)
class RunInputs:
    """Inputs shared by all folds."""

    conditions: tuple[Perturbation, ...]
    manifest: list[dict[str, str]]
    pad_heights: np.ndarray
    reference_cdf: np.ndarray | None
    label_dir: Path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, default=common.RUN_ROOT)
    parser.add_argument("--run-prefix", default=RUN_PREFIX)
    parser.add_argument("--label-dir", type=Path, default=common.LABEL_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--pad-heights", type=Path, default=PAD_HEIGHTS_CSV)
    parser.add_argument("--hist-cdf", type=Path, default=HIST_CDF_JSON)
    parser.add_argument("--d1-verdict", type=Path, default=D1_R4_VERDICT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--device", default="0", help="'0' (GPU) or 'cpu'")
    parser.add_argument(
        "--stage-a-recipes",
        default="",
        help="comma list of stage-A recipe ids kept by S2 (max 4, e.g. r0,r2,r5)",
    )
    parser.add_argument("--limit", type=int, default=None, help="images per fold")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"fold 0, {SMOKE_LIMIT} images, smoke folder; gate and clean tree skipped",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-conditions", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.folds = "0"
        args.limit = SMOKE_LIMIT if args.limit is None else args.limit
        args.out_root = args.out_root or SMOKE_OUT_ROOT
    args.out_root = args.out_root or DEFAULT_OUT_ROOT
    return args


def _stage_a_list(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def _is_gated(args: argparse.Namespace) -> bool:
    """The identity gate applies to full runs only (all crops of a fold)."""
    return args.limit is None


def _prepare_out_root(args: argparse.Namespace) -> None:
    if not args.out_root.exists():
        return
    if not args.overwrite:
        raise common.InputError(f"output exists (use --overwrite): {args.out_root}")
    shutil.rmtree(args.out_root)


def _checked_sha256(path: Path, expected: str | None, what: str) -> str:
    if not path.is_file():
        raise common.InputError(f"{what} missing: {path}")
    digest = common.sha256_file(path)
    if digest != expected:
        raise common.InputError(f"{what} sha256 {digest} != expected {expected}")
    return digest


def check_fold_inputs(
    run_dir: Path, fold: int, expected_eval_lists: dict[int, str]
) -> dict[str, str]:
    """Assert the fixed inputs of one fold before any inference.

    Weights against the N2-a' baseline, the crop list against the committed
    D1 r4 verdict and the D1 r4 gate reference against [CVR] A-1.

    Returns:
        ``{"weights", "eval_list", "d1_r4_metrics"}`` sha256 values.

    Raises:
        common.InputError: On a missing file or a sha256 mismatch (exit 2).
    """
    return {
        "weights": _checked_sha256(
            run_dir / "weights" / "last.pt",
            EXPECTED_WEIGHTS_SHA256[fold],
            f"fold {fold} weights",
        ),
        "eval_list": _checked_sha256(
            run_dir / "eval" / VALIDATOR_SUBDIR / "eval_images.txt",
            expected_eval_lists.get(fold),
            f"fold {fold} eval list",
        ),
        "d1_r4_metrics": _checked_sha256(
            d1_reference_path(run_dir),
            EXPECTED_D1_REFERENCE_SHA256[fold],
            f"fold {fold} D1 r4 gate reference",
        ),
    }


def label_dir_sha256(label_dir: Path) -> str:
    """sha256 over the sorted label file names and their content hashes."""
    digest = hashlib.sha256()
    for path in sorted(label_dir.glob("*.txt")):
        digest.update(f"{path.name}\t{common.sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _load_inputs(args: argparse.Namespace) -> RunInputs:
    conditions = build_conditions(_stage_a_list(args.stage_a_recipes))
    needs_cdf = any(c.mapping == GreyMimic.HIST for c in conditions)
    return RunInputs(
        conditions=conditions,
        manifest=read_manifest(args.manifest),
        pad_heights=load_pad_heights(args.pad_heights),
        reference_cdf=load_reference_cdf(args.hist_cdf) if needs_cdf else None,
        label_dir=args.label_dir,
    )


def _load_detector(weights: Path, device: str):
    # Lazy: ultralytics stays inside gpr_detection.detect (AGPL boundary).
    from gpr_detection.detect.yolo import load_yolo_detector
    from gpr_detection.detect.yolo_logging import quiet_ultralytics_logger

    quiet_ultralytics_logger()
    return load_yolo_detector(
        weights,
        input_size=common.CANVAS_PX,
        confidence_threshold=CONF_AP,
        iou_threshold=NMS_IOU,
        device=device,
    )


def _check_split(inputs: RunInputs, eval_list: Path, fold: int) -> dict:
    split = check_fold_split(inputs.manifest, read_eval_names(eval_list), fold)
    failed = [name for name, ok in split["checks"].items() if not ok]
    if failed:
        raise common.InputError(f"fold {fold} split assertions failed: {failed}")
    return split


def _new_fold_result(
    weights: Path, eval_list: Path, split: dict, samples: Sequence[Sample]
) -> dict:
    return {
        "weights": str(weights),
        "eval_list": str(eval_list),
        "n_images": len(samples),
        "split": split,
        "pad_heights_px": {sample.stem: sample.pad_height_px for sample in samples},
        "conditions": {},
        "gate": None,
    }


def _run_fold(
    args: argparse.Namespace,
    inputs: RunInputs,
    fold: int,
    run_identity: dict,
    started: float,
) -> tuple[dict, list]:
    """Check one fold's split, then run every condition on its crops.

    The fold's file hashes were asserted by :func:`check_fold_inputs` first.

    Raises:
        GateFailedError: If the identity condition misses the gate.
    """
    run_dir = args.run_root / f"{args.run_prefix}{fold}"
    weights = run_dir / "weights" / "last.pt"
    eval_list = run_dir / "eval" / VALIDATOR_SUBDIR / "eval_images.txt"
    split = _check_split(inputs, eval_list, fold)
    weights_sha256 = run_identity["weights_sha256"][str(fold)]
    print(f"fold {fold}: weights sha256 {weights_sha256}", flush=True)
    samples = read_samples(
        eval_list,
        gain_by_name(inputs.manifest),
        inputs.pad_heights,
        inputs.label_dir,
        args.limit,
    )
    detector = _load_detector(weights, args.device)
    fold_result = _new_fold_result(weights, eval_list, split, samples)
    object_rows: list = []
    for condition in inputs.conditions:
        condition_started = time.time()
        result = evaluate_condition(
            detector, samples, condition, inputs.reference_cdf, fold, object_rows
        )
        result["seconds"] = round(time.time() - condition_started, 1)
        fold_result["conditions"][condition.name] = result
        print(
            f"fold {fold} {condition.name}: {len(samples)} images, "
            f"{result['seconds']} s (elapsed {time.time() - started:.0f} s)",
            flush=True,
        )
        if condition.is_identity and _is_gated(args):
            fold_result["gate"] = _gate_fold(run_dir, fold, result, run_identity)
            if not fold_result["gate"]["pass"]:
                raise GateFailedError(fold, fold_result, object_rows)
    names = [condition.name for condition in inputs.conditions]
    fold_result["paired_recall"] = paired_recall(object_rows, names)
    fold_result["paired_recall_all_common"] = paired_recall_all_common(
        object_rows, names
    )
    return fold_result, object_rows


def _gate_fold(
    run_dir: Path, fold: int, identity_result: dict, run_identity: dict
) -> dict:
    reference = read_gate_reference(run_dir)
    run_identity["gate_inputs_sha256"][str(fold)] = reference.sha256
    expected = {name: counts[fold] for name, counts in common.EXPECTED_N_GT.items()}
    return check_identity_gate(
        identity_result["classes"],
        reference.d1_classes,
        reference.validator_ap50,
        expected,
        run_gpu=run_identity["gpu"],
        reference_gpu=reference.d1_gpu,
    )


def _new_run_identity(args: argparse.Namespace, inputs: RunInputs) -> dict:
    run_identity = common.new_run_identity()
    run_identity["gpu"] = common.gpu_name(args.device)
    run_identity["gate_inputs_sha256"] = {}
    run_identity["seed"] = SEED
    run_identity["conditions_sha256"] = conditions_sha256(inputs.conditions)
    hist_cdf_sha256 = None
    if inputs.reference_cdf is not None:
        hist_cdf_sha256 = common.sha256_file(args.hist_cdf)
    run_identity["inputs_sha256"] = {
        "manifest": common.sha256_file(args.manifest),
        "pad_heights": common.sha256_file(args.pad_heights),
        "hist_cdf": hist_cdf_sha256,
        "d1_r4_verdict": common.sha256_file(args.d1_verdict),
        "label_dir": label_dir_sha256(args.label_dir),
    }
    run_identity["allow_dirty"] = bool(args.allow_dirty or args.smoke)
    return run_identity


def _required_tracked_paths(args: argparse.Namespace, inputs: RunInputs) -> list[Path]:
    paths = [Path(__file__), Path(common.__file__), args.pad_heights, args.d1_verdict]
    if inputs.reference_cdf is not None:
        paths.append(args.hist_cdf)
    return paths


def _crop_list_sha256(fold_results: dict[int, dict]) -> str:
    digest = hashlib.sha256()
    for fold in sorted(fold_results):
        for stem in fold_results[fold]["pad_heights_px"]:
            digest.update(f"{fold}\t{stem}\n".encode())
    return digest.hexdigest()


def _settings(args: argparse.Namespace, inputs: RunInputs) -> dict:
    return {
        "seed": SEED,
        "period_aihub_px": PERIOD_AIHUB_PX,
        "conf_ap": CONF_AP,
        "conf_recall": common.CONF_RECALL,
        "nms_iou": NMS_IOU,
        "ignore_overlap": IGNORE_OVERLAP,
        "limit": args.limit,
        "smoke": args.smoke,
        "gated": _is_gated(args),
        "device": args.device,
        "conditions": [asdict(c) for c in inputs.conditions],
        "disclosure": DISCLOSURE,
    }


def _write_outputs(
    args: argparse.Namespace, summary: dict, fold_outputs: dict[int, tuple]
) -> None:
    for fold, (fold_result, object_rows) in fold_outputs.items():
        fold_dir = args.out_root / f"fold{fold}"
        common.write_json(
            fold_dir / "metrics.json",
            {"schema": SCHEMA, "run_identity": summary["run_identity"], **fold_result},
        )
        common.write_csv(fold_dir / "objects.csv", OBJECTS_HEADER, object_rows)
    common.write_json(args.out_root / "summary.json", summary)


def run(args: argparse.Namespace) -> dict:
    """Evaluate the selected folds and write all outputs.

    Returns:
        ``{"status", "n_conditions", "n_images"}`` for the closing stdout line.

    Raises:
        common.InputError: For a refused input or output (exit 2).
    """
    inputs = _load_inputs(args)
    if not (args.allow_dirty or args.smoke):
        problems = tracked_tree_problems(_required_tracked_paths(args, inputs))
        if problems:
            raise common.InputError(f"tree not clean for a full run: {problems}")
    folds = [int(f) for f in args.folds.split(",")]
    expected_eval_lists = read_expected_eval_list_sha256(args.d1_verdict)
    fold_hashes = {
        fold: check_fold_inputs(
            args.run_root / f"{args.run_prefix}{fold}", fold, expected_eval_lists
        )
        for fold in folds
    }
    _prepare_out_root(args)
    run_identity = _new_run_identity(args, inputs)
    for fold, hashes in fold_hashes.items():
        run_identity["weights_sha256"][str(fold)] = hashes["weights"]
        run_identity["eval_list_sha256"][str(fold)] = hashes["eval_list"]
    summary = {
        "schema": SCHEMA,
        "run_identity": run_identity,
        "settings": _settings(args, inputs),
    }
    started = time.time()
    fold_outputs: dict[int, tuple] = {}
    status = "ok"
    for fold in folds:
        try:
            fold_outputs[fold] = _run_fold(args, inputs, fold, run_identity, started)
        except GateFailedError as failure:
            # Stop at once: the other conditions would measure an unknown path.
            fold_outputs[fold] = (failure.fold_result, failure.object_rows)
            status = "gate_failed"
            logger.error("identity gate failed: %s (details in files only)", failure)
            break
    _finish_summary(args, summary, fold_outputs, inputs, started, status)
    return {
        "status": status,
        "n_conditions": len(inputs.conditions),
        "n_images": sum(result["n_images"] for result, _ in fold_outputs.values()),
    }


def _finish_summary(
    args: argparse.Namespace,
    summary: dict,
    fold_outputs: dict[int, tuple],
    inputs: RunInputs,
    started: float,
    status: str,
) -> None:
    fold_results = {fold: result for fold, (result, _) in fold_outputs.items()}
    names = [condition.name for condition in inputs.conditions]
    run_identity = summary["run_identity"]
    run_identity["crop_list_sha256"] = _crop_list_sha256(fold_results)
    run_identity["finished_at"] = common.local_now()
    summary["status"] = status
    summary["n_images_total"] = sum(r["n_images"] for r in fold_results.values())
    summary["n_images_expected_total"] = EXPECTED_N_IMAGES_TOTAL
    summary["gate"] = {str(fold): r["gate"] for fold, r in fold_results.items()}
    is_complete = status == "ok" and bool(fold_results)
    summary["aggregate"] = aggregate_folds(fold_results, names) if is_complete else {}
    summary["resources"] = {
        "seconds_total": round(time.time() - started, 1),
        "peak_vram_reserved_mb": _peak_vram_mb(args.device),
    }
    _write_outputs(args, summary, fold_outputs)


def print_final_line(
    status: str, args: argparse.Namespace | None, counts: dict, started: float
) -> None:
    """The closing stdout JSON line; it never carries metric values."""
    line = {
        "status": status,
        "out_dir": None if args is None else str(args.out_root),
        "n_conditions": counts.get("n_conditions"),
        "n_images": counts.get("n_images"),
        "elapsed_s": round(time.time() - started, 1),
    }
    print(json.dumps(line, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry; returns the exit code (0 ok, 1 gate failed, 2 refused, 3 error)."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    started = time.time()
    args = _parse_args(argv)
    counts: dict = {}
    try:
        if args.list_conditions:
            conditions = build_conditions(_stage_a_list(args.stage_a_recipes))
            print(json.dumps([condition.name for condition in conditions]))
            return 0
        counts = run(args)
        status = counts["status"]
    except common.InputError as error:
        logger.error("input error: %s", error)
        status = "input_error"
    except Exception:
        logger.exception("internal error")
        status = "error"
    print_final_line(status, args, counts, started)
    return EXIT_CODES[status]


if __name__ == "__main__":
    sys.exit(main())
