"""N2-a″ pre-check S-0..S-4: band labels left by rule K32 (CPU, no model).

Plan docs/experiments/m31_n2a3_retrain_plan.md v1.0 §5. Samples ``--images``
fold-0 training crops per seed (``default_rng(seed)``, as
``n2a2_band_drop_diag.py``) and pushes every crop through the augmentation
functions twice, once with the baseline (N2-a') config and once with the K32
(N2-a″) config, using the same generator ``default_rng([seed, i])`` for image
``i`` in both:

* R-ctx: context mode forced (``p_context`` = 1), ``augment_context_scale``
  only;
* R-nat: config probabilities, ``augment_context_scale`` then
  ``augment_shift_fill``.

Measured: the visible output width of the band labels left after the window
(by clipped / not clipped and by centre / neighbour origin), the band boxes K32
dropped, the centre-band drop share, a KS distance to the original crop band
widths, and the original distribution of the fold's training band labels.
Checks (both seeds):

  S-0  baseline, R-ctx: share of left band labels < 32 px >= 5 %
  S-1  K32,      R-ctx: share of left band labels < 32 px <= 2 %
  S-2  both,     R-nat: every image sha256 equal, non-band labels equal,
       K32 band labels a subset of the baseline band labels
  S-3  K32,      R-ctx: images with a centre band and no band left <= 1 %
  S-4  both,     R-ctx: (K32-dropped visible width sum) / (baseline left band
       visible width sum) <= 5 %

Centre labels are read like the neighbours (float64 from the label txt, not
the float32 Ultralytics cache). No training, GPU or weights.

Original crop band counts (report only, not a check) come in two definitions,
both unrounded with half-open bins [low, high):

* ``counts_plan_convention`` (plan v1.0 §1-2): width = normalised txt width
  x 256, edge = x1 <= 2 px or x2 >= 254 px;
* ``counts_raw``: width = ``x2 - x1`` of the float64 xyxy px boxes
  ((x_centre +- w / 2) x 256, as the context stage sees them), edge tolerance
  0.5 px. Float rounding of the subtraction puts some boxes of txt width
  exactly 64 px just below 64, into [32, 64); their number is
  ``width_64_counted_below_64_in_raw``.

stdout: status lines, then one summary JSON line {"status", "exit_code",
"out", "checks"}. --out must not exist unless --overwrite. Exit codes: 0 all
S checks pass, 1 an S check fails, 2 input error, 3 internal error.

Example (PowerShell, pinned, BelowNormal):
    conda run -n gpr-dev python -I analysis/detection/n2a3_band_label_diag.py \
        --baseline-config configs/train_yolo11m_n2a2.yaml \
        --config configs/train_yolo11m_n2a3.yaml --validation-fold 0 \
        --images 2000 --seeds 0,1 --out analysis/detection/n2a3_band_label_diag.json
"""

import argparse
import dataclasses
import hashlib
import itertools
import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

os.environ.setdefault("MKL_THREADING_LAYER", "TBB")

import numpy as np
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common

from gpr_detection.augment_geometry import (
    ORIGIN_CENTRE,
    BoxKeepRule,
    ContextIndex,
    ContextMode,
    ContextScaleConfig,
    CropContent,
    GeometricAugmentConfig,
    assemble_context_strip,
    augment_context_scale,
    augment_shift_fill,
    context_config_from_dict,
    crop_only_strip,
    geometric_config_from_dict,
)
from gpr_detection.detect.yolo_train import FolderCropLoader

REPO = common.REPO
SCRIPT = Path(__file__).resolve()
DATASET_DIR = REPO / "data" / "aihub_yolo_v1"
SCHEMA = "n2a3_band_label_diag/1"
BAND = common.CLASS_NAMES.index("pipe_band")
CANVAS_PX = common.CANVAS_PX
K32_KEY = "band_min_clipped_width_px"
SHORT_BAND_PX = 32.0
WIDTH_EDGES_PX = (0, 8, 16, 24, 32, 64, 128, 192, 256)
CLIPPED_TOLERANCE_PX = 1.0
"""Plan §3-1 2: clipped when visible < full width - 1 px."""
PLAN_EDGE_TOLERANCE_PX = 2.0
RAW_EDGE_TOLERANCE_PX = 0.5
"""Edge tolerances of ``counts_plan_convention`` (plan v1.0 §1-2) and ``counts_raw``."""
LABEL_DIR = DATASET_DIR / "labels"
YOLO_LABEL_FIELDS = 5
QUANTILES = (1, 5, 25, 50, 95)
PLAN_ORIGINAL = {"bands": 2748, "below_32": 0, "from_32_to_64": 155, "edge": 687}
"""Plan §1-2 (fold 0 training set, v1 labels); recomputed and compared."""
S0_MIN_SHORT_SHARE = Fraction(5, 100)
S1_MAX_SHORT_SHARE = Fraction(2, 100)
S3_MAX_NO_BAND_SHARE = Fraction(1, 100)
S4_MAX_WIDTH_SHARE = Fraction(5, 100)
CENTRE, NEIGHBOUR = "centre", "neighbour"
BASELINE, K32 = "baseline", "k32"

logger = logging.getLogger("n2a3_band_label_diag")


# ----------------------------------------------------------------- inputs


@dataclass(frozen=True)
class StageSettings:
    """The two settings of one config this script uses."""

    context: ContextScaleConfig
    geometric: GeometricAugmentConfig

    @property
    def keep_rule(self) -> BoxKeepRule:
        return self.geometric.keep_rule


def _read_config(path: Path) -> dict:
    if not path.is_file():
        raise common.InputError(f"missing config {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _stage_settings(config: dict) -> StageSettings:
    if config.get("gpr_context") is None or config.get("gpr_geometric") is None:
        raise common.InputError("config needs gpr_context and gpr_geometric blocks")
    try:
        return StageSettings(
            context_config_from_dict(config["gpr_context"]),
            geometric_config_from_dict(
                config["gpr_geometric"], scale_owned_by_context=True
            ),
        )
    except ValueError as error:
        raise common.InputError(f"bad config: {error}") from error


def check_config_pair(baseline: dict, k32: dict) -> None:
    """The K32 config must be the baseline plus a positive K32 key only.

    Raises:
        common.InputError: If the pair is not such a pair.
    """
    if float(baseline.get("gpr_context", {}).get(K32_KEY, 0)) != 0:
        raise common.InputError(f"--baseline-config must not turn {K32_KEY} on")
    k32_block = dict(k32.get("gpr_context") or {})
    if float(k32_block.pop(K32_KEY, 0)) <= 0:
        raise common.InputError(f"--config needs gpr_context.{K32_KEY} > 0")
    if {**k32, "gpr_context": k32_block} != baseline:
        raise common.InputError("--config differs from --baseline-config beyond K32")


def _list_names(list_file: Path) -> list[str]:
    if not list_file.is_file():
        raise common.InputError(f"missing {list_file}")
    lines = list_file.read_text(encoding="utf-8").splitlines()
    return [Path(line.strip()).name for line in lines if line.strip()]


def sample_names(train: Sequence[str], count: int, seed: int) -> list[str]:
    """``count`` training crops drawn with ``default_rng(seed)``, in list order."""
    sample_rng = np.random.default_rng(seed)
    count = min(count, len(train))
    return [train[i] for i in sorted(sample_rng.choice(len(train), count, False))]


# ------------------------------------------------------------- one image


@dataclass(frozen=True)
class ContextOutcome:
    """One image after ``augment_context_scale`` (and the shift/fill in R-nat)."""

    image_sha256: str
    boxes: np.ndarray
    classes: np.ndarray
    keep: np.ndarray
    is_clipped: np.ndarray
    origins: np.ndarray
    k32_dropped: tuple[tuple[float, int], ...]
    rng_state: str
    final: tuple[str, np.ndarray, np.ndarray] | None = None
    """R-nat only: (image sha256, kept boxes, kept classes) after shift/fill."""


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _window_factor_x(strip_width_px: int, s: float) -> float:
    """``place_window``'s horizontal factor (round of the scaled strip width)."""
    return 1.0 if s == 1.0 else round(strip_width_px * s) / strip_width_px


def _clip_flags(
    name: str,
    content: CropContent,
    index: ContextIndex,
    loader: FolderCropLoader,
    settings: StageSettings,
    boxes_out: np.ndarray,
    record: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Clipped flag (plan §3-1 2) and origin of every output box.

    The strip is rebuilt (deterministic, no random draws) to get each box's
    source width; plain mode uses the crop alone.
    """
    if record.window.mode is ContextMode.CONTEXT:
        strip = assemble_context_strip(name, content, index, loader, settings.context)
    else:
        strip = crop_only_strip(content)
    factor_x = _window_factor_x(strip.image.shape[1], record.window.s)
    full_widths = strip.source_widths_px * factor_x
    visible = boxes_out[:, 2] - boxes_out[:, 0]
    return visible < full_widths - CLIPPED_TOLERANCE_PX, strip.origins


def run_image(
    name: str,
    content: CropContent,
    index: ContextIndex,
    loader: FolderCropLoader,
    settings: StageSettings,
    rng: np.random.Generator,
    with_shift_fill: bool,
) -> ContextOutcome:
    """The augmentation stage(s) on one image."""
    image, boxes, classes, keep, record = augment_context_scale(
        name, content, index, loader, settings.context, settings.keep_rule, rng
    )
    final = None
    if with_shift_fill:
        shifted, shift_boxes, shift_keep, _ = augment_shift_fill(
            image, boxes[keep], settings.geometric, rng
        )
        final = (_sha256(shifted), shift_boxes[shift_keep], classes[keep][shift_keep])
    is_clipped, origins = _clip_flags(
        name, content, index, loader, settings, boxes, record
    )
    return ContextOutcome(
        image_sha256=_sha256(image),
        boxes=boxes,
        classes=classes,
        keep=keep,
        is_clipped=is_clipped,
        origins=origins,
        k32_dropped=record.band_k32_dropped,
        rng_state=repr(rng.bit_generator.state),
        final=final,
    )


# ------------------------------------------------------------ accumulators


def width_bin(width_px: float) -> str:
    """Bin of a width over :data:`WIDTH_EDGES_PX`; the top edge is in the last bin."""
    pairs = list(itertools.pairwise(WIDTH_EDGES_PX))
    for position, (low, high) in enumerate(pairs):
        is_last = position == len(pairs) - 1
        if low <= width_px < high or (is_last and width_px == high):
            return f"{low}-{high}"
    return f">{WIDTH_EDGES_PX[-1]}"


def _empty_bins() -> Counter:
    return Counter({width_bin(low): 0 for low, _ in itertools.pairwise(WIDTH_EDGES_PX)})


@dataclass
class ContextCounts:
    """R-ctx counts of one config and seed."""

    images: int = 0
    left_widths: list[float] = field(default_factory=list)
    left_bins: dict[str, Counter] = field(default_factory=dict)
    k32_dropped_widths: list[float] = field(default_factory=list)
    k32_dropped_by_origin: Counter = field(default_factory=Counter)
    centre_band_entering: int = 0
    centre_band_dropped: int = 0
    images_with_centre_band: int = 0
    images_with_centre_band_none_left: int = 0

    def add(self, outcome: ContextOutcome, has_centre_band: bool) -> None:
        self.images += 1
        is_band = outcome.classes == BAND
        widths = outcome.boxes[:, 2] - outcome.boxes[:, 0]
        for position in np.flatnonzero(is_band & outcome.keep):
            width = float(widths[position])
            self.left_widths.append(width)
            clipped = "clipped" if outcome.is_clipped[position] else "not_clipped"
            origin = CENTRE if outcome.origins[position] == ORIGIN_CENTRE else NEIGHBOUR
            for key in ("all", clipped, origin):
                self.left_bins.setdefault(key, _empty_bins())[width_bin(width)] += 1
        for width, origin in outcome.k32_dropped:
            self.k32_dropped_widths.append(width)
            self.k32_dropped_by_origin[
                CENTRE if origin == ORIGIN_CENTRE else NEIGHBOUR
            ] += 1
        is_centre_band = is_band & (outcome.origins == ORIGIN_CENTRE)
        self.centre_band_entering += int(is_centre_band.sum())
        self.centre_band_dropped += int((is_centre_band & ~outcome.keep).sum())
        if has_centre_band:
            self.images_with_centre_band += 1
            self.images_with_centre_band_none_left += not bool(
                (is_band & outcome.keep).any()
            )

    @property
    def short_share(self) -> Fraction | None:
        if not self.left_widths:
            return None
        short = sum(width < SHORT_BAND_PX for width in self.left_widths)
        return Fraction(short, len(self.left_widths))


@dataclass
class NaturalComparison:
    """R-nat pairing of the two configs for one seed (S-2)."""

    images: int = 0
    image_sha_differs: int = 0
    rng_state_differs: int = 0
    non_band_labels_differ: int = 0
    band_not_subset: int = 0
    k32_dropped: int = 0
    final_band_widths: dict[str, Counter] = field(
        default_factory=lambda: {BASELINE: _empty_bins(), K32: _empty_bins()}
    )

    def add(self, baseline: ContextOutcome, k32: ContextOutcome) -> None:
        self.images += 1
        assert baseline.final is not None and k32.final is not None
        self.image_sha_differs += (
            baseline.image_sha256 != k32.image_sha256
            or baseline.final[0] != k32.final[0]
        )
        self.rng_state_differs += baseline.rng_state != k32.rng_state
        base_rows = _label_rows(*baseline.final[1:])
        k32_rows = _label_rows(*k32.final[1:])
        self.non_band_labels_differ += _non_band(base_rows) != _non_band(k32_rows)
        self.band_not_subset += not (_band(k32_rows) <= _band(base_rows))
        self.k32_dropped += len(k32.k32_dropped)
        for key, outcome in ((BASELINE, baseline), (K32, k32)):
            boxes, classes = outcome.final[1:]
            for box in boxes[classes == BAND]:
                self.final_band_widths[key][width_bin(float(box[2] - box[0]))] += 1

    @property
    def passes(self) -> bool:
        return (
            self.image_sha_differs == 0
            and self.non_band_labels_differ == 0
            and self.band_not_subset == 0
        )


def _label_rows(boxes: np.ndarray, classes: np.ndarray) -> Counter:
    """Multiset of exact ``(class, x1, y1, x2, y2)`` rows."""
    return Counter(
        (int(class_id), *(float(value) for value in box))
        for box, class_id in zip(boxes, classes, strict=True)
    )


def _non_band(rows: Counter) -> Counter:
    return Counter({row: n for row, n in rows.items() if row[0] != BAND})


def _band(rows: Counter) -> Counter:
    return Counter({row: n for row, n in rows.items() if row[0] == BAND})


# ---------------------------------------------------------------- summary


def _fraction_view(value: Fraction | None) -> dict | None:
    if value is None:
        return None
    return {"value": float(value), "exact": f"{value.numerator}/{value.denominator}"}


def _ks_distance(widths: Sequence[float], original: Sequence[float]) -> float | None:
    if not widths or not original:
        return None
    return float(stats.ks_2samp(widths, original).statistic)


def context_summary(counts: ContextCounts, original: Sequence[float]) -> dict:
    """JSON view of one R-ctx run."""
    left_sum = sum(counts.left_widths)
    return {
        "images": counts.images,
        "left_band_labels": len(counts.left_widths),
        "left_band_visible_width_sum_px": left_sum,
        "left_band_short_labels": sum(
            width < SHORT_BAND_PX for width in counts.left_widths
        ),
        "left_band_short_share": _fraction_view(counts.short_share),
        "left_band_width_bins": {
            key: dict(bins) for key, bins in sorted(counts.left_bins.items())
        },
        "k32_dropped": len(counts.k32_dropped_widths),
        "k32_dropped_by_origin": dict(sorted(counts.k32_dropped_by_origin.items())),
        "k32_dropped_visible_width_sum_px": sum(counts.k32_dropped_widths),
        "centre_band_entering_window": counts.centre_band_entering,
        "centre_band_dropped": counts.centre_band_dropped,
        "centre_band_drop_share": _fraction_view(
            Fraction(counts.centre_band_dropped, counts.centre_band_entering)
            if counts.centre_band_entering
            else None
        ),
        "images_with_centre_band": counts.images_with_centre_band,
        "centre_band_images_without_band_left": (
            counts.images_with_centre_band_none_left
        ),
        "centre_band_images_without_band_left_share": _fraction_view(
            _no_band_share(counts)
        ),
        "ks_distance_to_original": _ks_distance(counts.left_widths, original),
    }


def _no_band_share(counts: ContextCounts) -> Fraction | None:
    if not counts.images_with_centre_band:
        return None
    return Fraction(
        counts.images_with_centre_band_none_left, counts.images_with_centre_band
    )


def natural_summary(comparison: NaturalComparison) -> dict:
    """JSON view of one R-nat pairing."""
    return {
        "images": comparison.images,
        "image_sha256_differs": comparison.image_sha_differs,
        "rng_state_differs": comparison.rng_state_differs,
        "non_band_labels_differ": comparison.non_band_labels_differ,
        "band_labels_not_subset": comparison.band_not_subset,
        "k32_dropped": comparison.k32_dropped,
        "final_band_width_bins": {
            key: dict(bins) for key, bins in comparison.final_band_widths.items()
        },
    }


def s_checks(seed_results: dict[int, dict]) -> dict:
    """S-0..S-4 per seed and over both seeds (plan §5-2)."""
    rows: dict[str, dict] = {}
    for seed, result in seed_results.items():
        base, k32 = result["context"][BASELINE], result["context"][K32]
        base_short, k32_short = base.short_share, k32.short_share
        no_band = _no_band_share(k32)
        left_sum = sum(base.left_widths)
        width_share = (
            Fraction(sum(k32.k32_dropped_widths)) / Fraction(left_sum)
            if left_sum
            else None
        )
        values = {
            "S-0": (
                base_short,
                base_short is not None and base_short >= S0_MIN_SHORT_SHARE,
            ),
            "S-1": (
                k32_short,
                k32_short is not None and k32_short <= S1_MAX_SHORT_SHARE,
            ),
            "S-2": (None, result["natural"].passes),
            "S-3": (no_band, no_band is not None and no_band <= S3_MAX_NO_BAND_SHARE),
            "S-4": (
                width_share,
                width_share is not None and width_share <= S4_MAX_WIDTH_SHARE,
            ),
        }
        for check, (value, is_pass) in values.items():
            rows.setdefault(check, {"per_seed": {}})["per_seed"][str(seed)] = {
                "value": _fraction_view(value),
                "pass": is_pass,
            }
    for row in rows.values():
        row["pass"] = all(seed["pass"] for seed in row["per_seed"].values())
    return rows


S_RULES = {
    "S-0": "baseline R-ctx: left band labels < 32 px share >= 5 %",
    "S-1": "K32 R-ctx: left band labels < 32 px share <= 2 %",
    "S-2": "R-nat: image sha256 equal, non-band labels equal, K32 band subset",
    "S-3": "K32 R-ctx: centre-band images with no band left <= 1 %",
    "S-4": "R-ctx: K32-dropped visible width / baseline left band width <= 5 %",
}


# -------------------------------------------------------- original widths


def _txt_band_widths(label_path: Path) -> list[float]:
    """Normalised band widths x 256 from a label txt, in the loader's row order.

    Duplicate rows are skipped as ``FolderCropLoader.labels`` skips them, so
    the list lines up with the loader's band boxes.
    """
    if not label_path.is_file():
        return []
    seen: set[tuple[float, ...]] = set()
    widths = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != YOLO_LABEL_FIELDS:
            continue
        row = tuple(float(value) for value in parts)
        if row in seen:
            continue
        seen.add(row)
        if int(row[0]) == BAND:
            widths.append(row[3] * CANVAS_PX)
    return widths


def original_band_boxes(
    names: Sequence[str], loader: FolderCropLoader
) -> tuple[np.ndarray, np.ndarray]:
    """Band labels of ``names``: ``[B, 4]`` xyxy boxes and ``[B]`` txt widths.

    Raises:
        common.InputError: If a label txt and the loader disagree on the band
            count of an image.
    """
    boxes_rows: list[np.ndarray] = []
    txt_widths: list[float] = []
    for name in names:
        boxes, classes = loader.labels(name)
        bands = np.asarray(boxes)[np.asarray(classes) == BAND]
        widths = _txt_band_widths(LABEL_DIR / f"{Path(name).stem}.txt")
        if len(widths) != len(bands):
            raise common.InputError(f"{name}: label txt and loader band counts differ")
        boxes_rows.extend(bands)
        txt_widths.extend(widths)
    return (
        np.array(boxes_rows, dtype=np.float64).reshape(-1, 4),
        np.array(txt_widths, dtype=np.float64),
    )


def _width_counts(widths: np.ndarray, boxes: np.ndarray, edge_tolerance: float):
    touches = (boxes[:, 0] <= edge_tolerance) | (
        boxes[:, 2] >= CANVAS_PX - edge_tolerance
    )
    return {
        "bands": len(widths),
        "below_32": int((widths < 32).sum()),
        "from_32_to_64": int(((widths >= 32) & (widths < 64)).sum()),
        "edge": int(touches.sum()),
    }


def original_summary(boxes: np.ndarray, txt_widths: np.ndarray) -> dict:
    """Plan §1-2 statistics, its counts and the raw counts with their definitions.

    * ``counts_plan_convention`` (plan v1.0 §1-2): width = normalised txt width
      x 256, no rounding, half-open bins; edge tolerance 2 px.
    * ``counts_raw``: width = ``x2 - x1`` of the xyxy px boxes the context stage
      uses (``FolderCropLoader``: (x_centre +- w / 2) x 256 in float64), no
      rounding, half-open bins; edge tolerance 0.5 px. Float rounding of the
      subtraction leaves some boxes of txt width exactly 64 px just short of
      64, so they fall into [32, 64) here (``width_64_counted_below_64_in_raw``).
    """
    widths = boxes[:, 2] - boxes[:, 0]
    bins = _empty_bins()
    for width in widths:
        bins[width_bin(float(width))] += 1
    plan_counts = _width_counts(txt_widths, boxes, PLAN_EDGE_TOLERANCE_PX)
    quantiles = {
        f"p{q}": float(np.percentile(widths, q)) if len(widths) else None
        for q in QUANTILES
    }
    return {
        "counts_plan_convention": plan_counts,
        "counts_raw": _width_counts(widths, boxes, RAW_EDGE_TOLERANCE_PX),
        "width_64_counted_below_64_in_raw": int(
            ((txt_widths == 64) & (widths < 64)).sum()
        ),
        "quantiles_px": quantiles,
        "width_bins": dict(bins),
        "plan_1_2_counts": PLAN_ORIGINAL,
        "matches_plan_1_2": plan_counts == PLAN_ORIGINAL,
        "definitions": {
            "counts_plan_convention": "width = txt normalised width x 256, no "
            "rounding, bins [low, high); edge = x1 <= "
            f"{PLAN_EDGE_TOLERANCE_PX} or x2 >= {CANVAS_PX} - {PLAN_EDGE_TOLERANCE_PX}",
            "counts_raw": "width = x2 - x1 of the float64 xyxy px boxes "
            "((x_centre +- w / 2) x 256), no rounding, bins [low, high); edge = "
            f"x1 <= {RAW_EDGE_TOLERANCE_PX} or x2 >= {CANVAS_PX} - "
            f"{RAW_EDGE_TOLERANCE_PX}",
            "quantiles_and_width_bins": "x2 - x1 widths as counts_raw",
        },
    }


# -------------------------------------------------------------------- run


def run_seed(
    names: Sequence[str],
    seed: int,
    index: ContextIndex,
    loader: FolderCropLoader,
    settings: dict[str, StageSettings],
) -> dict:
    """R-ctx and R-nat of both configs on one sample."""
    forced = {
        key: dataclasses.replace(
            value, context=dataclasses.replace(value.context, p_context=1.0)
        )
        for key, value in settings.items()
    }
    context = {BASELINE: ContextCounts(), K32: ContextCounts()}
    natural = NaturalComparison()
    for i, name in enumerate(names):
        boxes, classes = loader.labels(name)
        content = CropContent(loader.image(name), boxes, classes)
        has_centre_band = bool((np.asarray(classes) == BAND).any())
        for key in (BASELINE, K32):
            rng = np.random.default_rng([seed, i])
            outcome = run_image(name, content, index, loader, forced[key], rng, False)
            context[key].add(outcome, has_centre_band)
        paired = [
            run_image(
                name,
                content,
                index,
                loader,
                settings[key],
                np.random.default_rng([seed, i]),
                True,
            )
            for key in (BASELINE, K32)
        ]
        natural.add(*paired)
    return {"context": context, "natural": natural}


def _training_names(fold: int) -> tuple[list[str], list[str]]:
    train = [
        name
        for k in common.FOLDS
        if k != fold
        for name in _list_names(DATASET_DIR / f"fold{k}_images.txt")
    ]
    return train, _list_names(DATASET_DIR / f"fold{fold}_images.txt")


def run(args: argparse.Namespace, identity: dict) -> tuple[int, dict]:
    if args.out.exists() and not args.overwrite:
        raise common.InputError(f"--out exists (use --overwrite): {args.out}")
    baseline_config = _read_config(args.baseline_config)
    k32_config = _read_config(args.config)
    settings = {
        BASELINE: _stage_settings(baseline_config),
        K32: _stage_settings(k32_config),
    }
    check_config_pair(baseline_config, k32_config)
    train, validation = _training_names(args.validation_fold)
    index_dir = REPO / settings[BASELINE].context.index_dir
    index = ContextIndex.from_csv(index_dir, train, validation)
    loader = FolderCropLoader(DATASET_DIR / "images", CANVAS_PX, CANVAS_PX)
    original_boxes, original_txt_widths = original_band_boxes(train, loader)
    original_widths = list(original_boxes[:, 2] - original_boxes[:, 0])
    print(f"original band labels: {len(original_boxes)}", flush=True)

    seed_results = {}
    seeds_view = {}
    for seed in args.seeds:
        names = sample_names(train, args.images, seed)
        result = run_seed(names, seed, index, loader, settings)
        seed_results[seed] = result
        seeds_view[str(seed)] = {
            "sampled_images": len(names),
            "r_ctx": {
                key: context_summary(counts, original_widths)
                for key, counts in result["context"].items()
            },
            "r_nat": natural_summary(result["natural"]),
        }
        print(f"seed {seed}: R-ctx and R-nat done ({len(names)} images)", flush=True)
    checks = s_checks(seed_results)

    identity["finished_at"] = common.local_now()
    identity["baseline_config_sha256"] = common.sha256_file(args.baseline_config)
    identity["config_sha256"] = common.sha256_file(args.config)
    identity["index_sha256"] = index.file_sha256
    identity["script_sha256"] = common.sha256_file(SCRIPT)
    document = {
        "schema": SCHEMA,
        "note": "pre-check before training; augmentation labels, not a metric",
        "plan": "docs/experiments/m31_n2a3_retrain_plan.md v1.0 §5",
        "run_identity": identity,
        "inputs": {
            "baseline_config": str(args.baseline_config),
            "config": str(args.config),
            "validation_fold": args.validation_fold,
            "train_images": len(train),
            "images_per_seed": args.images,
            "seeds": list(args.seeds),
            "per_image_rng": "default_rng([seed, i])",
            "width_edges_px": list(WIDTH_EDGES_PX),
        },
        "original_crop_band_widths": original_summary(
            original_boxes, original_txt_widths
        ),
        "seeds": seeds_view,
        "s_rules": S_RULES,
        "s_checks": checks,
        "s_all_pass": all(row["pass"] for row in checks.values()),
    }
    common.write_json(args.out, document)
    summary = {check: row["pass"] for check, row in checks.items()}
    return (0 if document["s_all_pass"] else 1), summary


# -------------------------------------------------------------------- CLI


def _seed_list(text: str) -> list[int]:
    try:
        seeds = [int(item) for item in text.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"bad --seeds {text!r}") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(f"--seeds needs distinct ints, got {text!r}")
    return seeds


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-fold", type=int, default=0, choices=common.FOLDS)
    parser.add_argument("--images", type=int, default=2000)
    parser.add_argument("--seeds", type=_seed_list, default=[0, 1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.images < 1:
        parser.error("--images must be >= 1")
    return args


def _final_line(exit_code: int, out: Path, checks: dict | None) -> int:
    line = {
        "status": "ok" if exit_code in (0, 1) else "error",
        "exit_code": exit_code,
        "out": str(out),
        "checks": checks,
    }
    print(json.dumps(line, ensure_ascii=False), flush=True)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """CLI entry; 0 all S pass, 1 an S check fails, 2 input error, 3 internal."""
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
    try:
        args = _parse_args(argv)
    except SystemExit as exit_request:
        return 2 if exit_request.code else 0
    identity = common.new_run_identity()
    identity["gpu"] = "cpu"
    try:
        code, checks = run(args, identity)
    except common.InputError as error:
        logger.error("input error: %s", error)
        return _final_line(2, args.out, None)
    except Exception:
        logger.exception("internal error")
        return _final_line(3, args.out, None)
    return _final_line(code, args.out, checks)


if __name__ == "__main__":
    sys.exit(main())
