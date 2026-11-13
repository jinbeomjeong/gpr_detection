"""Backend-neutral geometric augmentations for grayscale GPR B-scans (M3-1 N2-a).

Plan: ``docs/experiments/m31_n2a_retrain_plan.md`` v0.2 section 1-1 and 5-1.
Three augmentations widen the position and scale range the detector sees:

* Isotropic scale ``s ~ U[0.55, 1.5]``, anchored at the top row and the
  horizontal centre. Shrinking uses ``cv2.INTER_AREA`` (area averaging, no
  aliasing of the ~4 px carrier), enlarging ``cv2.INTER_LINEAR``; the uncovered
  area is filled with gray 128.
* Horizontal shift by an integer number of columns (no resampling); the
  emptied columns are filled with 128. There is deliberately no vertical shift:
  the surface row must stay at the top.
* Edge fill band: a constant 128 band on the left, right or bottom edge, as
  short survey lines and the C' bottom padding produce in real tiles.

Boxes are ``x1, y1, x2, y2`` in pixels. After each stage a box is clipped to
the visible area and kept only if its visible area is at least half of the
reference area and, if clipping cut it, both visible sides are at least 4 px
(plan section 1-1-a, v0.3 section 5-7 (다), :class:`BoxKeepRule`). Every
function returns new arrays; inputs are never modified.

N2-a' context scale (``docs/experiments/m31_n2a2_retrain_plan.md`` v1.0
section 2-2, 3, 4 and 7-1) replaces the N2-a isotropic scale: per image a
context mode (strip of overlapping train crops of the same line, gain and fold,
scale ``U[1.0, 1.5]``, window moved inside real content) or a plain mode (the
crop alone, scale ``U[0.55, 1.5]``, centred window). See
:class:`ContextIndex`, :func:`assemble_context_strip`, :func:`place_window`
and :func:`augment_context_scale`.

This module lives next to :mod:`gpr_detection.augment` instead of inside it
because the architecture rule 5 keeps ``augment`` free of cv2
(``docs/architecture/README.md``).
"""

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import cv2
import numpy as np

GRAY_LEVEL_MAX = 255
MAX_SHIFT_LIMIT_PX = 128
"""Largest allowed ``max_shift_px`` (plan section 5-1)."""
SCALE_FACTOR_LIMIT = 2.0
"""Largest allowed ``scale_max``."""
SIDE_WEIGHT_SUM_TOLERANCE = 1e-9
CLIPPED_AREA_TOLERANCE = 1e-9
"""A box counts as clipped when it loses more than this share of its area."""

VisibleRect = tuple[int, int, int, int]
"""Visible region ``(x1, y1, x2, y2)`` in pixels, half-open like a slice."""


class EdgeSide(StrEnum):
    """Edge that an edge fill band covers; the order matches ``side_weights``."""

    LEFT = "left"
    RIGHT = "right"
    BOTTOM = "bottom"


EDGE_SIDES = (EdgeSide.LEFT, EdgeSide.RIGHT, EdgeSide.BOTTOM)


def _check_probability(probability: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")


def _check_fill(fill: int) -> None:
    if not 0 <= fill <= GRAY_LEVEL_MAX:
        raise ValueError(f"fill must be in [0, {GRAY_LEVEL_MAX}], got {fill}")


def _check_size_range(name: str, low: int, high: int) -> None:
    if not 0 < low <= high:
        raise ValueError(f"{name} range must satisfy 0 < min <= max, got {low}, {high}")


@dataclass(frozen=True)
class IsotropicScaleConfig:
    """Isotropic scale drawn once per image.

    Attributes:
        probability: Chance of applying the scale to one image.
        scale_min: Smallest factor (shrink).
        scale_max: Largest factor (enlarge).
        fill: Gray value of the area the shrunk image no longer covers.
    """

    probability: float = 1.0
    scale_min: float = 0.55
    scale_max: float = 1.5
    fill: int = 128

    def __post_init__(self) -> None:
        _check_probability(self.probability)
        if not 0.0 < self.scale_min <= 1.0 <= self.scale_max <= SCALE_FACTOR_LIMIT:
            raise ValueError(
                "scale range must satisfy 0 < scale_min <= 1 <= scale_max <= "
                f"{SCALE_FACTOR_LIMIT}, got {self.scale_min}, {self.scale_max}"
            )
        _check_fill(self.fill)


@dataclass(frozen=True)
class HorizontalShiftConfig:
    """Integer column shift ``dx ~ U{-max_shift_px, ..., max_shift_px}``.

    Attributes:
        probability: Chance of applying the shift to one image.
        max_shift_px: Largest shift in columns, either direction.
        fill: Gray value of the emptied columns.
    """

    probability: float = 0.5
    max_shift_px: int = 96
    fill: int = 128

    def __post_init__(self) -> None:
        _check_probability(self.probability)
        if not 0 <= self.max_shift_px <= MAX_SHIFT_LIMIT_PX:
            raise ValueError(
                f"max_shift_px must be in [0, {MAX_SHIFT_LIMIT_PX}], "
                f"got {self.max_shift_px}"
            )
        _check_fill(self.fill)


@dataclass(frozen=True)
class EdgeFillConfig:
    """Constant band on one image edge.

    Attributes:
        probability: Chance of drawing a band on one image.
        side_weights: Probabilities of left, right and bottom; must sum to 1.
        width_min_px: Narrowest left/right band, columns.
        width_max_px: Widest left/right band, columns.
        height_min_px: Lowest bottom band, rows.
        height_max_px: Highest bottom band, rows.
        fill: Gray value of the band.
    """

    probability: float = 0.3
    side_weights: tuple[float, float, float] = (0.35, 0.35, 0.30)
    width_min_px: int = 8
    width_max_px: int = 96
    height_min_px: int = 16
    height_max_px: int = 128
    fill: int = 128

    def __post_init__(self) -> None:
        _check_probability(self.probability)
        weights = tuple(float(weight) for weight in self.side_weights)
        object.__setattr__(self, "side_weights", weights)  # YAML gives a list
        if len(weights) != len(EDGE_SIDES) or min(weights) < 0.0:
            raise ValueError(
                f"side_weights must be three non-negative values, got {weights}"
            )
        if abs(sum(weights) - 1.0) > SIDE_WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"side_weights must sum to 1, got {sum(weights)}")
        _check_size_range("width_px", self.width_min_px, self.width_max_px)
        _check_size_range("height_px", self.height_min_px, self.height_max_px)
        _check_fill(self.fill)


@dataclass(frozen=True)
class BoxKeepRule:
    """When a clipped box keeps its label (plan section 1-1-a).

    Attributes:
        min_visible_area_fraction: Visible area / reference area must reach this.
        min_side_px: Visible width and height must both reach this, checked
            only for boxes the visible region cut (a whole box may be smaller).
    """

    min_visible_area_fraction: float = 0.5
    min_side_px: float = 4.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_visible_area_fraction <= 1.0:
            raise ValueError(
                "min_visible_area_fraction must be in [0, 1], got "
                f"{self.min_visible_area_fraction}"
            )
        if self.min_side_px < 0:
            raise ValueError(f"min_side_px must be >= 0, got {self.min_side_px}")


@dataclass(frozen=True)
class GeometricAugmentConfig:
    """Settings of the three geometric augmentations and the box keep rule.

    Independent of :class:`gpr_detection.augment.AugmentConfig` and its
    ``strength``. The defaults are the N2-a values.
    """

    scale: IsotropicScaleConfig = field(default_factory=IsotropicScaleConfig)
    shift: HorizontalShiftConfig = field(default_factory=HorizontalShiftConfig)
    edge_fill: EdgeFillConfig = field(default_factory=EdgeFillConfig)
    keep_rule: BoxKeepRule = field(default_factory=BoxKeepRule)


@dataclass(frozen=True)
class ShiftFillRecord:
    """What :func:`augment_shift_fill` drew; ``None`` means the stage was skipped.

    Attributes:
        shift_dx_px: Drawn column shift.
        edge_side: Drawn band edge.
        edge_size_px: Drawn band width or height.
        shift_keep_mask: ``[N]`` keep mask of the shift stage alone.
        edge_keep_mask: ``[N]`` keep mask of the edge fill stage alone.
    """

    shift_dx_px: int | None
    edge_side: EdgeSide | None
    edge_size_px: int | None
    shift_keep_mask: np.ndarray | None = None
    edge_keep_mask: np.ndarray | None = None


_CONFIG_SECTIONS = {
    "scale": ("scale", IsotropicScaleConfig),
    "horizontal_shift": ("shift", HorizontalShiftConfig),
    "edge_fill": ("edge_fill", EdgeFillConfig),
    "keep_rule": ("keep_rule", BoxKeepRule),
}
"""YAML section name -> (config field, section class)."""


def geometric_config_from_dict(
    settings: Mapping[str, Any], *, scale_owned_by_context: bool = False
) -> GeometricAugmentConfig:
    """Build a :class:`GeometricAugmentConfig` from the ``gpr_geometric`` YAML block.

    Missing sections keep their defaults.

    Args:
        settings: Mapping with optional sections ``scale``, ``horizontal_shift``,
            ``edge_fill`` and ``keep_rule``.
        scale_owned_by_context: ``True`` when a ``gpr_context`` block is on
            (N2-a'): :class:`YoloContextScale` then owns the isotropic scale,
            so a missing ``scale`` section means "off" (probability 0) instead
            of the N2-a default.

    Raises:
        ValueError: If the block has an unknown section (typo protection), or
            ``scale_owned_by_context`` is set and the ``scale`` section is on
            (two scale stages would multiply, docs/experiments/
            m31_n2a2_retrain_plan.md v1.0 section 7-1, test C-19).
        TypeError: If a section contains an unknown key.
    """
    unknown = sorted(set(settings) - set(_CONFIG_SECTIONS))
    if unknown:
        raise ValueError(
            f"unknown gpr_geometric sections {unknown}; "
            f"expected {sorted(_CONFIG_SECTIONS)}"
        )
    sections = {
        field_name: section_class(**settings.get(yaml_name, {}))
        for yaml_name, (field_name, section_class) in _CONFIG_SECTIONS.items()
    }
    if scale_owned_by_context:
        if "scale" in settings and sections["scale"].probability > 0:
            raise ValueError(
                "gpr_geometric.scale must be absent or off when gpr_context is "
                "on (YoloContextScale owns the isotropic scale; "
                "docs/experiments/m31_n2a2_retrain_plan.md section 7-1)"
            )
        sections["scale"] = IsotropicScaleConfig(probability=0.0)
    return GeometricAugmentConfig(**sections)


# ----------------------------------------------------------------- helpers


def _check_gray_image(image: np.ndarray) -> None:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(
            f"image must be a 2-D uint8 array [H, W], got {image.dtype} {image.shape}"
        )


def _as_box_array(boxes_xyxy: np.ndarray) -> np.ndarray:
    boxes = np.array(boxes_xyxy, dtype=np.float64)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes_xyxy must have shape [N, 4], got {boxes.shape}")
    return boxes


def box_areas(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Areas of ``[N, 4]`` xyxy boxes in px² (negative sizes count as 0)."""
    boxes = _as_box_array(boxes_xyxy)
    widths = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    return widths * heights


def _full_image_rect(image: np.ndarray) -> VisibleRect:
    height_px, width_px = image.shape
    return (0, 0, width_px, height_px)


def _clip_to_image(boxes: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Clip label boxes to the image frame before a stage.

    Some AIHub labels overshoot the frame by float rounding (e.g. y1 =
    -0.000128 px). Without this, the keep rule would count such a box as
    "clipped" by the stage and apply the 4 px side limit to it.
    """
    height_px, width_px = image.shape
    clipped = boxes.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0.0, width_px)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0.0, height_px)
    return clipped


# -------------------------------------------------------------- operations


def scale_isotropic(
    image: np.ndarray, boxes_xyxy: np.ndarray, s: float, fill: int
) -> tuple[np.ndarray, np.ndarray, VisibleRect]:
    """Scale by ``s`` about the top row and horizontal centre, keeping ``[H, W]``.

    The resampled size is ``round(W*s) x round(H*s)``; the exact factors
    ``kx = new_w / W`` and ``ky = new_h / H`` move the boxes. Shrinking fills
    the bottom and both sides with ``fill``; enlarging crops them.

    Args:
        image: Grayscale uint8 image ``[H, W]``.
        boxes_xyxy: Boxes ``[N, 4]`` in px.
        s: Scale factor (> 0). ``s == 1`` returns copies without resampling.
        fill: Gray value of the uncovered area.

    Returns:
        ``(image, boxes, visible_rect)``: the new image, the scaled boxes (not
        clipped) and the region covered by the resampled image.

    Raises:
        ValueError: If the image is not 2-D uint8, the boxes are not ``[N, 4]``,
            ``s <= 0`` or ``fill`` is outside [0, 255].
    """
    _check_gray_image(image)
    boxes = _as_box_array(boxes_xyxy)
    if s <= 0:
        raise ValueError(f"scale factor must be > 0, got {s}")
    _check_fill(fill)
    if s == 1.0:
        return image.copy(), boxes, _full_image_rect(image)

    height_px, width_px = image.shape
    new_width_px = round(width_px * s)
    new_height_px = round(height_px * s)
    interpolation = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    resampled = cv2.resize(
        image, (new_width_px, new_height_px), interpolation=interpolation
    )
    # Positive when shrinking (centred with margins), negative when enlarging.
    offset_x_px = (width_px - new_width_px) // 2
    visible_x1 = max(0, offset_x_px)
    visible_x2 = min(width_px, offset_x_px + new_width_px)
    visible_y2 = min(height_px, new_height_px)

    result = np.full_like(image, fill)
    result[:visible_y2, visible_x1:visible_x2] = resampled[
        :visible_y2, visible_x1 - offset_x_px : visible_x2 - offset_x_px
    ]

    factor_x = new_width_px / width_px
    factor_y = new_height_px / height_px
    scaled = boxes * np.array([factor_x, factor_y, factor_x, factor_y])
    scaled[:, [0, 2]] += offset_x_px
    return result, scaled, (visible_x1, 0, visible_x2, visible_y2)


def shift_horizontal(
    image: np.ndarray, boxes_xyxy: np.ndarray, dx: int, fill: int
) -> tuple[np.ndarray, np.ndarray, VisibleRect]:
    """Shift columns by ``dx`` (positive = right) without resampling.

    Args:
        image: Grayscale uint8 image ``[H, W]``.
        boxes_xyxy: Boxes ``[N, 4]`` in px.
        dx: Integer shift; ``|dx|`` must be smaller than ``W``.
        fill: Gray value of the emptied columns.

    Returns:
        ``(image, boxes, visible_rect)``: the new image, the shifted boxes (not
        clipped) and the columns that still show image content.

    Raises:
        ValueError: On a bad image, boxes, fill, or ``|dx| >= W``.
    """
    _check_gray_image(image)
    boxes = _as_box_array(boxes_xyxy)
    _check_fill(fill)
    height_px, width_px = image.shape
    if abs(dx) >= width_px:
        raise ValueError(f"|dx| must be < image width {width_px}, got {dx}")

    result = np.full_like(image, fill)
    if dx >= 0:
        result[:, dx:] = image[:, : width_px - dx]
    else:
        result[:, :dx] = image[:, -dx:]
    shifted = boxes + np.array([dx, 0.0, dx, 0.0])
    visible = (max(0, dx), 0, min(width_px, width_px + dx), height_px)
    return result, shifted, visible


def apply_edge_fill(
    image: np.ndarray, side: EdgeSide | str, size_px: int, fill: int
) -> tuple[np.ndarray, VisibleRect]:
    """Overwrite a band of ``size_px`` columns (left/right) or rows (bottom).

    Args:
        image: Grayscale uint8 image ``[H, W]``.
        side: ``"left"``, ``"right"`` or ``"bottom"``.
        size_px: Band width (left/right) or height (bottom), within the image.
        fill: Gray value of the band.

    Returns:
        ``(image, visible_rect)``: the new image and the region outside the band.

    Raises:
        ValueError: On a bad image, side, fill, or a band larger than the image.
    """
    _check_gray_image(image)
    _check_fill(fill)
    edge = EdgeSide(side)
    height_px, width_px = image.shape
    limit_px = height_px if edge is EdgeSide.BOTTOM else width_px
    if not 0 <= size_px <= limit_px:
        raise ValueError(f"{edge} band size must be in [0, {limit_px}], got {size_px}")

    result = image.copy()
    if edge is EdgeSide.LEFT:
        result[:, :size_px] = fill
        return result, (size_px, 0, width_px, height_px)
    if edge is EdgeSide.RIGHT:
        result[:, width_px - size_px :] = fill
        return result, (0, 0, width_px - size_px, height_px)
    result[height_px - size_px :, :] = fill
    return result, (0, 0, width_px, height_px - size_px)


def filter_boxes(
    boxes_ref_area: np.ndarray,
    boxes: np.ndarray,
    visible_rect: VisibleRect,
    rule: BoxKeepRule,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip boxes to ``visible_rect`` and decide which labels survive.

    Args:
        boxes_ref_area: ``[N]`` reference areas in px²: the box area just before
            the stage (for the scale stage the area times ``kx * ky``, which is
            the area of the scaled, unclipped box).
        boxes: ``[N, 4]`` boxes after the stage, not clipped.
        visible_rect: Region that shows image content after the stage.
        rule: Keep thresholds.

    Returns:
        ``(boxes_clipped, keep_mask)``: all ``N`` boxes clipped to the visible
        region (dropped ones included, possibly with zero size) and a boolean
        ``[N]`` mask of the labels to keep.

    Raises:
        ValueError: If the inputs have different lengths.
    """
    clipped = _as_box_array(boxes)
    reference_area = np.asarray(boxes_ref_area, dtype=np.float64).reshape(-1)
    if len(reference_area) != len(clipped):
        raise ValueError(
            f"{len(reference_area)} reference areas for {len(clipped)} boxes"
        )
    pre_clip_area = box_areas(clipped)
    visible_x1, visible_y1, visible_x2, visible_y2 = visible_rect
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], visible_x1, visible_x2)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], visible_y1, visible_y2)
    visible_width = clipped[:, 2] - clipped[:, 0]
    visible_height = clipped[:, 3] - clipped[:, 1]
    visible_area = visible_width * visible_height
    # The side limit removes fragments made by clipping (plan v0.3 §5-7 (다)); a
    # small but whole box, e.g. a 4.79 px manhole shrunk to 2.6 px, stays.
    is_clipped = visible_area < pre_clip_area * (1.0 - CLIPPED_AREA_TOLERANCE)
    has_short_side = (visible_width < rule.min_side_px) | (
        visible_height < rule.min_side_px
    )
    keep_mask = (visible_area >= rule.min_visible_area_fraction * reference_area) & ~(
        is_clipped & has_short_side
    )
    return clipped, keep_mask


# ------------------------------------------------------------ random stages


def augment_scale(
    image: np.ndarray,
    boxes: np.ndarray,
    config: GeometricAugmentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    """Draw and apply the isotropic scale, then apply the keep rule.

    Args:
        image: Grayscale uint8 image ``[H, W]``; not modified.
        boxes: Boxes ``[N, 4]`` in px.
        config: Uses ``config.scale`` and ``config.keep_rule``.
        rng: Random generator.

    Returns:
        ``(image, boxes, keep_mask, s)``. ``boxes`` holds all ``N`` boxes,
        clipped; ``keep_mask`` marks the labels to keep. ``s`` is the drawn
        factor, or ``None`` when the probability check skipped the stage (the
        image and boxes then come back as unchanged copies).
    """
    _check_gray_image(image)
    box_array = _clip_to_image(_as_box_array(boxes), image)
    scale = config.scale
    if rng.random() >= scale.probability:
        return image.copy(), box_array, np.ones(len(box_array), dtype=bool), None
    s = float(rng.uniform(scale.scale_min, scale.scale_max))
    scaled_image, scaled_boxes, visible = scale_isotropic(
        image, box_array, s, scale.fill
    )
    clipped, keep_mask = filter_boxes(
        box_areas(scaled_boxes), scaled_boxes, visible, config.keep_rule
    )
    return scaled_image, clipped, keep_mask, s


def _draw_edge_band(
    edge_fill: EdgeFillConfig, rng: np.random.Generator
) -> tuple[EdgeSide, int]:
    side = EDGE_SIDES[int(rng.choice(len(EDGE_SIDES), p=edge_fill.side_weights))]
    if side is EdgeSide.BOTTOM:
        low, high = edge_fill.height_min_px, edge_fill.height_max_px
    else:
        low, high = edge_fill.width_min_px, edge_fill.width_max_px
    return side, int(rng.integers(low, high, endpoint=True))


def augment_shift_fill(
    image: np.ndarray,
    boxes: np.ndarray,
    config: GeometricAugmentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ShiftFillRecord]:
    """Horizontal shift, then edge fill band, each with its own probability.

    The keep rule runs after each applied stage with the box area just before
    that stage as reference; the two keep masks are combined.

    Args:
        image: Grayscale uint8 image ``[H, W]``; not modified.
        boxes: Boxes ``[N, 4]`` in px.
        config: Uses ``shift``, ``edge_fill`` and ``keep_rule``.
        rng: Random generator.

    Returns:
        ``(image, boxes, keep_mask, record)``. ``boxes`` holds all ``N`` boxes,
        clipped; ``record`` holds the drawn shift and band.
    """
    _check_gray_image(image)
    result = image.copy()
    current = _clip_to_image(_as_box_array(boxes), image)
    shift_dx_px = shift_mask = None
    edge_side = edge_size_px = edge_mask = None

    shift = config.shift
    if rng.random() < shift.probability:
        shift_dx_px = int(
            rng.integers(-shift.max_shift_px, shift.max_shift_px, endpoint=True)
        )
        reference_area = box_areas(current)
        result, shifted, visible = shift_horizontal(
            result, current, shift_dx_px, shift.fill
        )
        current, shift_mask = filter_boxes(
            reference_area, shifted, visible, config.keep_rule
        )

    edge_fill = config.edge_fill
    if rng.random() < edge_fill.probability:
        edge_side, edge_size_px = _draw_edge_band(edge_fill, rng)
        result, visible = apply_edge_fill(
            result, edge_side, edge_size_px, edge_fill.fill
        )
        current, edge_mask = filter_boxes(
            box_areas(current), current, visible, config.keep_rule
        )

    keep_mask = np.ones(len(current), dtype=bool)
    for stage_mask in (shift_mask, edge_mask):
        if stage_mask is not None:
            keep_mask &= stage_mask
    record = ShiftFillRecord(
        shift_dx_px, edge_side, edge_size_px, shift_mask, edge_mask
    )
    return result, current, keep_mask, record


# ===================================================== N2-a' context scale
#
# Plan docs/experiments/m31_n2a2_retrain_plan.md v1.0: section 2-2 (stage),
# 3-1 (neighbours and brightness), 3-2 (labels), 4 (leak assertions), 7-1.
# Coordinates: "original x" is the column of the AIHub B-scan line a crop was
# cut from (file name ``x(a-b)``); "strip x" is original x minus the strip's
# first column. Rows are never moved (top row = surface).

TRAIN_POOL_SUBSET = "trainpool"
PIPE_BAND_CLASS_ID = 1
"""``pipe_band`` (project class order, CLAUDE.md); merged by columns (§3-2)."""
DUPLICATE_IOU_MIN = 0.5
"""Same-class boxes of two crops with at least this IoU are one object (§3-2)."""
CONTAINED_IOMIN_MIN = 0.7
CONTAINED_AREA_RATIO_MIN = 1.25
"""Rule (가) of §3-2 v0.3: intersection / smaller area and larger / smaller area
from which the larger box replaces the smaller (a box cut at the crop edge)."""
CONTEXT_INDEX_HASHED_FILES = ("crops.csv", "groups.csv")
"""Index files whose sha256 a run records (§4)."""
CROPS_FILE_NAME = "crops.csv"
_REPORTED_PROBLEMS = 5


class ContextMode(StrEnum):
    """Mode of :class:`YoloContextScale` for one image (§2-2 step 1)."""

    CONTEXT = "context"
    PLAIN = "plain"


class ContextSource(StrEnum):
    """Where the content around the crop came from (§2-2 step 6).

    ``none``: plain mode. ``c1``: context mode without a usable neighbour (only
    the enlargement margin). ``a``: context mode with at least one stitched
    neighbour crop.
    """

    NONE = "none"
    C1 = "c1"
    A = "a"


class RejectReason(StrEnum):
    """Why the chain stopped at a neighbour (§3-1)."""

    OVERLAP = "overlap"
    CORRELATION = "correlation"
    ALPHA = "alpha"


def _float_pair(name: str, values: Sequence[float]) -> tuple[float, float]:
    pair = tuple(float(value) for value in values)
    if len(pair) != 2 or pair[0] > pair[1]:
        raise ValueError(f"{name} must be [low, high] with low <= high, got {values}")
    return (pair[0], pair[1])


@dataclass(frozen=True)
class ContextScaleConfig:
    """The ``gpr_context`` YAML block (§7-1).

    Attributes:
        index_dir: Folder with ``crops.csv`` and ``groups.csv`` (project
            relative or absolute).
        p_context: Chance of the context mode for one image.
        context_scale: ``[low, high]`` of ``s`` in the context mode; low >= 1,
            so the 256 px window always fits inside real content.
        plain_scale: ``[low, high]`` of ``s`` in the plain mode.
        max_shift_px: Largest window shift either way in the context mode.
        extend_px: Context columns gathered on each side of the crop (E).
        min_overlap_px: Narrowest overlap with which a neighbour is stitched.
        corr_min: Smallest overlap correlation before brightness correction.
        alpha_range: Allowed ``[low, high]`` of the brightness gain alpha.
        fill: Gray value of the area a shrunk window does not cover.
    """

    index_dir: str = "data/aihub_context_index"
    p_context: float = 0.5
    context_scale: tuple[float, float] = (1.0, 1.5)
    plain_scale: tuple[float, float] = (0.55, 1.5)
    max_shift_px: int = 96
    extend_px: int = 128
    min_overlap_px: int = 16
    corr_min: float = 0.98
    alpha_range: tuple[float, float] = (0.8, 1.25)
    fill: int = 128

    def __post_init__(self) -> None:
        _check_probability(self.p_context)
        # YAML gives lists; tuples keep the config hashable and comparable.
        for name in ("context_scale", "plain_scale", "alpha_range"):
            object.__setattr__(self, name, _float_pair(name, getattr(self, name)))
        context_low, context_high = self.context_scale
        if not 1.0 <= context_low <= context_high <= SCALE_FACTOR_LIMIT:
            raise ValueError(
                "context_scale must satisfy 1 <= low <= high <= "
                f"{SCALE_FACTOR_LIMIT}, got {self.context_scale}"
            )
        plain_low, plain_high = self.plain_scale
        if not 0.0 < plain_low <= plain_high <= SCALE_FACTOR_LIMIT:
            raise ValueError(
                f"plain_scale must satisfy 0 < low <= high <= {SCALE_FACTOR_LIMIT}, "
                f"got {self.plain_scale}"
            )
        if not 0 <= self.max_shift_px <= MAX_SHIFT_LIMIT_PX:
            raise ValueError(
                f"max_shift_px must be in [0, {MAX_SHIFT_LIMIT_PX}], "
                f"got {self.max_shift_px}"
            )
        if self.extend_px < 0 or self.min_overlap_px < 1:
            raise ValueError(
                "need extend_px >= 0 and min_overlap_px >= 1, got "
                f"{self.extend_px}, {self.min_overlap_px}"
            )
        if not -1.0 <= self.corr_min <= 1.0:
            raise ValueError(f"corr_min must be in [-1, 1], got {self.corr_min}")
        if self.alpha_range[0] <= 0.0:
            raise ValueError(f"alpha_range must be positive, got {self.alpha_range}")
        _check_fill(self.fill)


def context_config_from_dict(settings: Mapping[str, Any]) -> ContextScaleConfig:
    """Build a :class:`ContextScaleConfig` from the ``gpr_context`` YAML block.

    Raises:
        ValueError: If the block has an unknown key (typo protection) or a
            value is out of range.
    """
    known = set(ContextScaleConfig.__dataclass_fields__)
    unknown = sorted(set(settings) - known)
    if unknown:
        raise ValueError(
            f"unknown gpr_context keys {unknown}; expected {sorted(known)}"
        )
    return ContextScaleConfig(**settings)


def settings_block_sha256(block: Mapping[str, Any]) -> str:
    """sha256 of a settings block as canonical JSON (sorted keys, no spaces).

    Training records it and the A1 judge compares it with the sealed config.
    """
    text = json.dumps(block, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hex sha256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------ context index


@dataclass(frozen=True)
class ContextCrop:
    """One row of ``crops.csv`` (the columns N2-a' uses).

    Attributes:
        file_name: Image file name (``..._g<gain>_x(a-b).jpg``).
        line: Session + channel; context only comes from the same line.
        gain: Display gain as written in the index (``"0.5"``).
        x1_px: First original column ``a``.
        x2_px: End original column ``b`` (exclusive).
        subset: ``"trainpool"`` or ``"test"``.
        fold: Fold 0-4, -1 for the test subset.
        primary_class: Class the crop was cut for.
        crop_key: Gain-free crop key.
    """

    file_name: str
    line: str
    gain: str
    x1_px: int
    x2_px: int
    subset: str
    fold: int
    primary_class: str = ""
    crop_key: str = ""

    @property
    def width_px(self) -> int:
        return self.x2_px - self.x1_px

    @property
    def center_x_px(self) -> float:
        return (self.x1_px + self.x2_px) / 2


def check_neighbor_admissible(
    center: ContextCrop, neighbor: ContextCrop, validation_folds: Iterable[int]
) -> None:
    """Leak assertions for one neighbour of ``center`` (§4 "이웃마다").

    Raises:
        ValueError: If the neighbour is not a train-pool crop, lies in another
            fold or in this run's validation fold, or has another line or gain.
    """
    problems = []
    if neighbor.subset != TRAIN_POOL_SUBSET:
        problems.append(f"subset {neighbor.subset!r} != {TRAIN_POOL_SUBSET!r}")
    if neighbor.fold != center.fold:
        problems.append(f"fold {neighbor.fold} != centre fold {center.fold}")
    if neighbor.fold in set(validation_folds):
        problems.append(f"fold {neighbor.fold} is a validation fold of this run")
    if neighbor.line != center.line or neighbor.gain != center.gain:
        problems.append("line or gain differs from the centre")
    if problems:
        raise ValueError(
            f"context leak: neighbour {neighbor.file_name} of {center.file_name}: "
            + "; ".join(problems)
        )


def _check_line_assignment(crops: Iterable[ContextCrop]) -> None:
    """Each line must map to one (fold, subset) (§4; curator: line = session)."""
    assignments: dict[str, set[tuple[int, str]]] = {}
    for crop in crops:
        assignments.setdefault(crop.line, set()).add((crop.fold, crop.subset))
    split_lines = sorted(line for line, seen in assignments.items() if len(seen) > 1)
    if split_lines:
        raise ValueError(
            f"context index: {len(split_lines)} lines span several (fold, subset), "
            f"e.g. {split_lines[:_REPORTED_PROBLEMS]}"
        )


def read_crops_csv(path: Path) -> list[ContextCrop]:
    """Rows of the context index ``crops.csv`` (the columns N2-a' uses)."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [
            ContextCrop(
                file_name=row["file_name"],
                line=row["line"],
                gain=row["gain"],
                x1_px=int(row["a"]),
                x2_px=int(row["b"]),
                subset=row["subset"],
                fold=int(row["fold"]),
                primary_class=row.get("primary_class", ""),
                crop_key=row.get("crop_key", ""),
            )
            for row in csv.DictReader(handle)
        ]


class ContextIndex:
    """Overlapping-crop lookup with the start-up leak assertions of §4.

    Neighbours of a crop are the crops of the same ``(line, gain)``; only crops
    in this run's train list are used as context.

    Args:
        crops: Rows of ``crops.csv``.
        train_file_names: Image file names of the train list (``fold<k>.yaml``
            ``train``); only these are context candidates.
        validation_file_names: Image file names of the validation list; their
            folds are the folds no neighbour may come from.
        file_sha256: sha256 of the index files, recorded with the run.

    Raises:
        ValueError: If a file name repeats, a line spans several (fold,
            subset), a train image is missing from the index or is not a
            train-pool crop, or a train fold is also a validation fold.
    """

    def __init__(
        self,
        crops: Iterable[ContextCrop],
        train_file_names: Iterable[str],
        validation_file_names: Iterable[str] = (),
        file_sha256: Mapping[str, str] | None = None,
    ) -> None:
        crop_list = list(crops)
        self._crops_by_name: dict[str, ContextCrop] = {}
        for crop in crop_list:
            if crop.file_name in self._crops_by_name:
                raise ValueError(f"context index: duplicate file {crop.file_name}")
            self._crops_by_name[crop.file_name] = crop
        _check_line_assignment(crop_list)
        self._train_names = frozenset(train_file_names)
        self.train_folds = self._train_folds()
        self.validation_folds = frozenset(
            self._crops_by_name[name].fold
            for name in validation_file_names
            if name in self._crops_by_name
        )
        shared = sorted(self.train_folds & self.validation_folds)
        if shared:
            raise ValueError(f"context index: folds {shared} are train and validation")
        self._by_line_gain: dict[tuple[str, str], list[ContextCrop]] = {}
        for crop in crop_list:
            self._by_line_gain.setdefault((crop.line, crop.gain), []).append(crop)
        for group in self._by_line_gain.values():
            group.sort(key=lambda crop: crop.x1_px)
        self.file_sha256 = dict(file_sha256 or {})

    def _train_folds(self) -> frozenset[int]:
        missing = sorted(self._train_names - set(self._crops_by_name))
        if missing:
            raise ValueError(
                f"context index: {len(missing)} train images are not in the index, "
                f"e.g. {missing[:_REPORTED_PROBLEMS]}"
            )
        train_crops = [self._crops_by_name[name] for name in self._train_names]
        outside_pool = sorted(
            crop.file_name for crop in train_crops if crop.subset != TRAIN_POOL_SUBSET
        )
        if outside_pool:
            raise ValueError(
                f"context index: {len(outside_pool)} train images are not train-pool "
                f"crops, e.g. {outside_pool[:_REPORTED_PROBLEMS]}"
            )
        return frozenset(crop.fold for crop in train_crops)

    @classmethod
    def from_csv(
        cls,
        index_dir: Path,
        train_file_names: Iterable[str],
        validation_file_names: Iterable[str] = (),
    ) -> "ContextIndex":
        """Read ``<index_dir>/crops.csv`` and hash the index files (§4)."""
        index_dir = Path(index_dir)
        hashes = {
            name: file_sha256(index_dir / name) for name in CONTEXT_INDEX_HASHED_FILES
        }
        crops = read_crops_csv(index_dir / CROPS_FILE_NAME)
        return cls(crops, train_file_names, validation_file_names, hashes)

    def crop(self, file_name: str) -> ContextCrop:
        """Index row of an image file name.

        Raises:
            ValueError: If the file is not in the index.
        """
        try:
            return self._crops_by_name[file_name]
        except KeyError:
            raise ValueError(f"context index has no crop {file_name!r}") from None

    def neighbors(self, center: ContextCrop, extend_px: int) -> list[ContextCrop]:
        """Train crops of the centre's line and gain within ``[a - E, b + E)``.

        Every crop found is checked with :func:`check_neighbor_admissible`
        before the train-list filter, so a leak raises instead of being
        silently skipped. Crops of another gain are never looked at (§3-1).

        Raises:
            ValueError: On a leak (see :func:`check_neighbor_admissible`).
        """
        reach_x1 = center.x1_px - extend_px
        reach_x2 = center.x2_px + extend_px
        found = []
        for crop in self._by_line_gain.get((center.line, center.gain), []):
            if crop.file_name == center.file_name:
                continue
            if crop.x2_px <= reach_x1 or crop.x1_px >= reach_x2:
                continue
            check_neighbor_admissible(center, crop, self.validation_folds)
            if crop.file_name in self._train_names:
                found.append(crop)
        return found

    def summary(self) -> dict[str, Any]:
        """Facts a run records next to its settings."""
        return {
            "file_sha256": dict(self.file_sha256),
            "crops": len(self._crops_by_name),
            "train_images": len(self._train_names),
            "train_folds": sorted(self.train_folds),
            "validation_folds": sorted(self.validation_folds),
        }


# --------------------------------------------------------- brightness (§3-1)


class BrightnessFit(NamedTuple):
    """Least-squares fit ``reference ~ alpha * neighbour + beta`` on the overlap."""

    alpha: float
    beta: float
    corr: float
    ok: bool


def fit_brightness(
    ref_overlap: np.ndarray,
    nb_overlap: np.ndarray,
    corr_min: float = 0.98,
    alpha_range: tuple[float, float] = (0.8, 1.25),
) -> BrightnessFit:
    """Fit the neighbour's brightness to the reference on the shared columns.

    Args:
        ref_overlap: Reference crop pixels of the overlap (all rows).
        nb_overlap: Neighbour pixels of the same original columns.
        corr_min: Smallest accepted Pearson correlation (before correction).
        alpha_range: Accepted ``[low, high]`` of alpha.

    Returns:
        ``(alpha, beta, corr, ok)``; a flat overlap (zero variance) gives NaN
        values and ``ok = False``.

    Raises:
        ValueError: If the two overlaps differ in shape or are empty.
    """
    reference = np.asarray(ref_overlap, dtype=np.float64).ravel()
    neighbor = np.asarray(nb_overlap, dtype=np.float64).ravel()
    if reference.shape != neighbor.shape or reference.size == 0:
        raise ValueError(
            f"overlaps must have the same non-empty shape, got "
            f"{np.shape(ref_overlap)} and {np.shape(nb_overlap)}"
        )
    reference_centered = reference - reference.mean()
    neighbor_centered = neighbor - neighbor.mean()
    neighbor_sum_squares = float(np.sum(neighbor_centered * neighbor_centered))
    reference_sum_squares = float(np.sum(reference_centered * reference_centered))
    if neighbor_sum_squares == 0.0 or reference_sum_squares == 0.0:
        return BrightnessFit(float("nan"), float("nan"), float("nan"), False)
    cross = float(np.sum(neighbor_centered * reference_centered))
    alpha = cross / neighbor_sum_squares
    beta = float(reference.mean()) - alpha * float(neighbor.mean())
    corr = cross / float(np.sqrt(neighbor_sum_squares * reference_sum_squares))
    alpha_low, alpha_high = alpha_range
    is_ok = alpha_low <= alpha <= alpha_high and corr >= corr_min
    return BrightnessFit(alpha, beta, corr, is_ok)


def apply_brightness(image: np.ndarray, fit: BrightnessFit) -> np.ndarray:
    """``clip(round(alpha * image + beta), 0, 255)`` as uint8 (§3-1)."""
    corrected = np.rint(fit.alpha * image.astype(np.float64) + fit.beta)
    return np.clip(corrected, 0, GRAY_LEVEL_MAX).astype(np.uint8)


def _reject_reason(fit: BrightnessFit, corr_min: float) -> RejectReason:
    # NaN correlation (flat overlap) fails the ">=" test, so it counts here.
    if not fit.corr >= corr_min:
        return RejectReason.CORRELATION
    return RejectReason.ALPHA


# ------------------------------------------------------------ labels (§3-2)


@dataclass(frozen=True)
class StripPiece:
    """One crop of a context strip with the columns it contributes.

    Attributes:
        file_name: Image file name.
        x1_px: Crop start ``a`` (original x).
        x2_px: Crop end ``b`` (original x, exclusive).
        owned_x1_px: First strip column taken from this crop (original x).
        owned_x2_px: End of the columns taken from this crop.
        boxes: ``[N, 4]`` label boxes in original x (y unchanged), clipped to
            the crop frame.
        classes: ``[N]`` class ids.
    """

    file_name: str
    x1_px: int
    x2_px: int
    owned_x1_px: int
    owned_x2_px: int
    boxes: np.ndarray
    classes: np.ndarray

    @property
    def center_x_px(self) -> float:
        return (self.x1_px + self.x2_px) / 2


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    top_left = np.maximum(box[:2], others[:, :2])
    bottom_right = np.minimum(box[2:], others[:, 2:])
    intersection = np.clip(bottom_right - top_left, 0.0, None).prod(axis=1)
    union = box_areas(box[None])[0] + box_areas(others) - intersection
    return np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )


def _voronoi_cells(pieces: Sequence[StripPiece]) -> list[tuple[float, float]]:
    """``[low, high)`` of each piece's cell; borders are midpoints of crop centres."""
    order = sorted(range(len(pieces)), key=lambda index: pieces[index].center_x_px)
    cells: list[tuple[float, float]] = [(0.0, 0.0)] * len(pieces)
    for rank, index in enumerate(order):
        low = -np.inf
        high = np.inf
        if rank > 0:
            low = (pieces[order[rank - 1]].center_x_px + pieces[index].center_x_px) / 2
        if rank < len(order) - 1:
            high = (pieces[index].center_x_px + pieces[order[rank + 1]].center_x_px) / 2
        cells[index] = (low, high)
    return cells


def _priority_order(pieces: Sequence[StripPiece], center_index: int) -> list[int]:
    """Centre first, then by distance to the centre crop (left first on ties)."""
    center_x = pieces[center_index].center_x_px
    return sorted(
        range(len(pieces)),
        key=lambda index: (
            abs(pieces[index].center_x_px - center_x),
            pieces[index].center_x_px,
        ),
    )


def _intersection_over_min(box: np.ndarray, other: np.ndarray) -> float:
    top_left = np.maximum(box[:2], other[:2])
    bottom_right = np.minimum(box[2:], other[2:])
    intersection = float(np.clip(bottom_right - top_left, 0.0, None).prod())
    smaller_area = float(min(box_areas(box[None])[0], box_areas(other[None])[0]))
    return intersection / smaller_area if smaller_area > 0 else 0.0


def _area_ratio(box: np.ndarray, other: np.ndarray) -> float:
    """Larger area / smaller area (inf when the smaller box is empty)."""
    areas = sorted(float(box_areas(item[None])[0]) for item in (box, other))
    return areas[1] / areas[0] if areas[0] > 0 else np.inf


def _is_contained_pair(box: np.ndarray, other: np.ndarray) -> bool:
    """Rule (가): one box mostly inside a clearly larger one (§3-2 v0.3)."""
    return (
        _intersection_over_min(box, other) >= CONTAINED_IOMIN_MIN
        and _area_ratio(box, other) >= CONTAINED_AREA_RATIO_MIN
    )


def resolve_duplicate(
    candidate: np.ndarray, kept: Sequence[np.ndarray]
) -> tuple[bool, list[int]]:
    """Same-class duplicate rules of §3-2 (v0.3) for one later box.

    ``kept`` boxes come from the centre crop or crops nearer to it. Against
    each, rule (가) first: IoMin >= 0.7 and area ratio >= 1.25 keeps the larger
    box; otherwise rule (나): IoU >= 0.5 keeps the earlier (kept) box.

    Returns:
        ``(is_kept, replaced)``: whether ``candidate`` stays and the positions
        in ``kept`` it replaces (rule (가) with the candidate larger).
    """
    replaced = []
    candidate_area = float(box_areas(candidate[None])[0])
    for position, kept_box in enumerate(kept):
        if _is_contained_pair(candidate, kept_box):
            if candidate_area > float(box_areas(kept_box[None])[0]):
                replaced.append(position)
                continue
            return False, []
        if _iou(candidate, kept_box[None])[0] >= DUPLICATE_IOU_MIN:
            return False, []
    return True, replaced


def _neighbor_box_is_owned(
    box: np.ndarray, center: StripPiece, cell: tuple[float, float]
) -> bool:
    """Rule 2: outside the centre columns and inside its own neighbour cell."""
    center_x = (box[0] + box[2]) / 2
    if center.x1_px <= center_x < center.x2_px:
        return False  # the centre label is the reference there
    cell_low, cell_high = cell
    return bool(cell_low <= center_x < cell_high)


def _entry_box(entry: tuple[np.ndarray, int] | None) -> np.ndarray:
    if entry is None:
        raise AssertionError("a removed box is still listed as kept")
    return entry[0]


def _band_piece(box: np.ndarray, piece: StripPiece) -> np.ndarray | None:
    cut = box.copy()
    cut[[0, 2]] = np.clip(cut[[0, 2]], piece.owned_x1_px, piece.owned_x2_px)
    return cut if cut[2] > cut[0] else None


def merge_strip_labels(
    pieces: Sequence[StripPiece],
    center_index: int,
    band_class_id: int = PIPE_BAND_CLASS_ID,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge the label boxes of a strip's crops (§3-2 v0.3), in original x.

    * ``pipe_band``: every crop keeps its band boxes cut to the columns it owns
      (the centre owns ``[a, b)``; outer crops own the columns no crop nearer
      to the centre covers). ``y1``/``y2`` stay as labelled.
    * Other classes: every centre-crop box stays. A neighbour box is dropped
      when its centre x lies in the centre columns ``[a, b)``; otherwise it
      stays only inside its own cell of the Voronoi split between the
      neighbour crop centres. Same-class duplicates then follow
      :func:`resolve_duplicate` (centre first, then nearer crops).

    Args:
        pieces: The strip's crops.
        center_index: Position of the centre crop in ``pieces``.
        band_class_id: Class id of ``pipe_band``.

    Returns:
        ``(boxes [M, 4], classes [M])``: centre crop first, then the other
        crops by distance to the centre, each in label order (a box that
        replaces an earlier one by rule (가) takes its place).
    """
    center = pieces[center_index]
    neighbor_indices = [index for index in range(len(pieces)) if index != center_index]
    neighbor_cells = _voronoi_cells([pieces[index] for index in neighbor_indices])
    cell_of = dict(zip(neighbor_indices, neighbor_cells, strict=True))
    merged: list[tuple[np.ndarray, int] | None] = []
    # Class id -> positions in ``merged`` of the kept non-band boxes.
    kept_by_class: dict[int, list[int]] = {}
    for index in _priority_order(pieces, center_index):
        piece = pieces[index]
        for box, class_value in zip(piece.boxes, piece.classes, strict=True):
            class_id = int(class_value)
            if class_id == band_class_id:
                cut = _band_piece(box, piece)
                if cut is not None:
                    merged.append((cut, class_id))
                continue
            is_center = index == center_index
            if not is_center and not _neighbor_box_is_owned(
                box, center, cell_of[index]
            ):
                continue
            positions = kept_by_class.setdefault(class_id, [])
            kept = [_entry_box(merged[position]) for position in positions]
            is_kept, replaced = resolve_duplicate(box, kept)
            if not is_kept:
                continue
            if replaced:
                first, *others = (positions[position] for position in replaced)
                merged[first] = (box, class_id)
                for position in others:
                    merged[position] = None
                    positions.remove(position)
                continue
            positions.append(len(merged))
            merged.append((box, class_id))
    entries = [entry for entry in merged if entry is not None]
    if not entries:
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.int64)
    boxes = np.array([box for box, _ in entries], dtype=np.float64)
    classes = np.array([class_id for _, class_id in entries], dtype=np.int64)
    return boxes, classes


# ---------------------------------------------------------- strip (§3-1)


@dataclass(frozen=True)
class CropContent:
    """Pixels and labels of one crop.

    Attributes:
        image: Grayscale uint8 ``[H, W]``.
        boxes: ``[N, 4]`` xyxy px in crop coordinates.
        classes: ``[N]`` class ids.
    """

    image: np.ndarray
    boxes: np.ndarray
    classes: np.ndarray


class CropLoader(Protocol):
    """Reads neighbour crops; injected so tests need no files."""

    def image(self, file_name: str) -> np.ndarray:
        """Grayscale uint8 ``[H, W]`` of a crop."""
        ...

    def labels(self, file_name: str) -> tuple[np.ndarray, np.ndarray]:
        """``(boxes [N, 4] xyxy px in crop coordinates, classes [N])``."""
        ...


@dataclass(frozen=True)
class ContextStrip:
    """A crop with its stitched context (§3).

    Attributes:
        image: Grayscale uint8 ``[H, W_s]``.
        x0_px: Original x of strip column 0.
        crop_left_px: Strip column of the centre crop's first column.
        boxes: ``[M, 4]`` merged label boxes in strip x.
        classes: ``[M]`` class ids.
        neighbor_files: Stitched neighbour files, left chain then right chain.
        rejections: Why each chain stopped, if it stopped at a neighbour.
    """

    image: np.ndarray
    x0_px: int
    crop_left_px: int
    boxes: np.ndarray
    classes: np.ndarray
    neighbor_files: tuple[str, ...] = ()
    rejections: tuple[RejectReason, ...] = ()

    @property
    def used_neighbors(self) -> int:
        return len(self.neighbor_files)


def crop_only_strip(content: CropContent) -> ContextStrip:
    """The crop alone as a strip (plain mode, or no usable neighbour)."""
    _check_gray_image(content.image)
    boxes = _clip_to_image(_as_box_array(content.boxes), content.image)
    classes = np.asarray(content.classes).reshape(-1)
    return ContextStrip(content.image, 0, 0, boxes, classes)


@dataclass
class _ChainLink:
    crop: ContextCrop
    pixels: np.ndarray
    owned_x1_px: int
    owned_x2_px: int

    def columns(self, x1_px: int, x2_px: int) -> np.ndarray:
        """Pixels of original columns ``[x1, x2)``."""
        return self.pixels[:, x1_px - self.crop.x1_px : x2_px - self.crop.x1_px]


def _grow_chain(
    center: _ChainLink,
    candidates: Sequence[ContextCrop],
    loader: CropLoader,
    config: ContextScaleConfig,
    is_rightward: bool,
) -> tuple[list[_ChainLink], RejectReason | None]:
    """Stitch neighbours outwards on one side until E is reached or one fails.

    ``candidates`` are ordered nearest first. Each new crop is fitted to the
    last accepted one (the reference) and only adds the columns beyond the
    already covered edge (pixels of crops nearer to the centre win).
    """
    reference = center
    if is_rightward:
        covered_edge, limit = center.crop.x2_px, center.crop.x2_px + config.extend_px
    else:
        covered_edge, limit = center.crop.x1_px, center.crop.x1_px - config.extend_px
    links: list[_ChainLink] = []
    for neighbor in candidates:
        is_done = covered_edge >= limit if is_rightward else covered_edge <= limit
        if is_done:
            break
        adds_columns = (
            neighbor.x2_px > covered_edge
            if is_rightward
            else neighbor.x1_px < covered_edge
        )
        if not adds_columns:
            continue
        overlap_x1 = max(reference.crop.x1_px, neighbor.x1_px)
        overlap_x2 = min(reference.crop.x2_px, neighbor.x2_px)
        if overlap_x2 - overlap_x1 < config.min_overlap_px:
            return links, RejectReason.OVERLAP
        pixels = loader.image(neighbor.file_name)
        if pixels.shape != center.pixels.shape or pixels.dtype != np.uint8:
            raise ValueError(
                f"neighbour {neighbor.file_name} is {pixels.dtype} {pixels.shape}, "
                f"centre is {center.pixels.dtype} {center.pixels.shape}"
            )
        neighbor_overlap = pixels[
            :, overlap_x1 - neighbor.x1_px : overlap_x2 - neighbor.x1_px
        ]
        fit = fit_brightness(
            reference.columns(overlap_x1, overlap_x2),
            neighbor_overlap,
            config.corr_min,
            config.alpha_range,
        )
        if not fit.ok:
            return links, _reject_reason(fit, config.corr_min)
        if is_rightward:
            owned = (covered_edge, min(neighbor.x2_px, limit))
        else:
            owned = (max(neighbor.x1_px, limit), covered_edge)
        link = _ChainLink(neighbor, apply_brightness(pixels, fit), *owned)
        links.append(link)
        reference = link
        covered_edge = owned[1] if is_rightward else owned[0]
    return links, None


def _piece(link: _ChainLink, boxes: np.ndarray, classes: np.ndarray) -> StripPiece:
    frame_boxes = _clip_to_image(_as_box_array(boxes), link.pixels)
    frame_boxes[:, [0, 2]] += link.crop.x1_px
    return StripPiece(
        file_name=link.crop.file_name,
        x1_px=link.crop.x1_px,
        x2_px=link.crop.x2_px,
        owned_x1_px=link.owned_x1_px,
        owned_x2_px=link.owned_x2_px,
        boxes=frame_boxes,
        classes=np.asarray(classes).reshape(-1),
    )


def assemble_context_strip(
    center_file: str,
    content: CropContent,
    index: ContextIndex,
    loader: CropLoader,
    config: ContextScaleConfig,
) -> ContextStrip:
    """Stitch the overlapping train crops around one crop (§3-1, §3-2).

    Args:
        center_file: Image file name of the crop (index key).
        content: The crop's own pixels and labels; its columns stay as given.
        index: Context index of this run.
        loader: Reads neighbour images and labels.
        config: ``extend_px``, ``min_overlap_px``, ``corr_min``, ``alpha_range``.

    Returns:
        The strip; the crop alone when no neighbour is usable.

    Raises:
        ValueError: On a leak, an unknown crop, or a crop whose width differs
            from its index interval.
    """
    _check_gray_image(content.image)
    center = index.crop(center_file)
    if content.image.shape[1] != center.width_px:
        raise ValueError(
            f"{center_file}: image width {content.image.shape[1]} != index width "
            f"{center.width_px}"
        )
    center_link = _ChainLink(center, content.image, center.x1_px, center.x2_px)
    candidates = index.neighbors(center, config.extend_px)
    right = sorted(
        (crop for crop in candidates if crop.x1_px > center.x1_px),
        key=lambda crop: crop.x1_px,
    )
    left = sorted(
        (crop for crop in candidates if crop.x1_px < center.x1_px),
        key=lambda crop: -crop.x1_px,
    )
    left_links, left_stop = _grow_chain(center_link, left, loader, config, False)
    right_links, right_stop = _grow_chain(center_link, right, loader, config, True)
    if not left_links and not right_links:
        strip = crop_only_strip(content)
        rejections = tuple(stop for stop in (left_stop, right_stop) if stop)
        return ContextStrip(
            strip.image, center.x1_px, 0, strip.boxes, strip.classes, (), rejections
        )

    links = [center_link, *left_links, *right_links]
    x0_px = min(link.owned_x1_px for link in links)
    x_end_px = max(link.owned_x2_px for link in links)
    image = np.empty((content.image.shape[0], x_end_px - x0_px), dtype=np.uint8)
    for link in links:
        image[:, link.owned_x1_px - x0_px : link.owned_x2_px - x0_px] = link.columns(
            link.owned_x1_px, link.owned_x2_px
        )

    pieces = [_piece(center_link, content.boxes, content.classes)]
    for link in [*left_links, *right_links]:
        boxes, classes = loader.labels(link.crop.file_name)
        pieces.append(_piece(link, boxes, classes))
    boxes, classes = merge_strip_labels(pieces, center_index=0)
    boxes[:, [0, 2]] -= x0_px
    return ContextStrip(
        image=image,
        x0_px=x0_px,
        crop_left_px=center.x1_px - x0_px,
        boxes=boxes,
        classes=classes,
        neighbor_files=tuple(
            link.crop.file_name for link in [*left_links, *right_links]
        ),
        rejections=tuple(stop for stop in (left_stop, right_stop) if stop),
    )


# ----------------------------------------------------------- window (§2-2)


@dataclass(frozen=True)
class WindowRecord:
    """What :func:`place_window` did.

    Attributes:
        mode: Context or plain.
        s: Scale factor.
        dx_px: Window shift; ``> 0`` moves the content right in the output.
        dx_min_px: Lowest allowed shift (feasible range ∩ ±max_shift).
        dx_max_px: Highest allowed shift.
        visible_rect: Output region that shows strip content.
    """

    mode: ContextMode
    s: float
    dx_px: int
    dx_min_px: int
    dx_max_px: int
    visible_rect: VisibleRect


def _centred_window_left(
    crop_left_px: int, window_width_px: int, strip_width_px: int, scaled_width_px: int
) -> int:
    """Scaled-strip column of the window start that centres the crop.

    ``ceil(c * k - W / 2)`` with the crop centre ``c`` (strip px) and
    ``k = scaled_width / strip_width``, in integers; for a strip that is the
    crop alone this is ``-((W - scaled_width) // 2)``, the N2-a placement.
    """
    twice_center_px = 2 * crop_left_px + window_width_px
    numerator = twice_center_px * scaled_width_px - window_width_px * strip_width_px
    denominator = 2 * strip_width_px
    return -((-numerator) // denominator)


def place_window(
    strip: np.ndarray,
    boxes: np.ndarray,
    s: float,
    mode: ContextMode | str,
    rng: np.random.Generator,
    crop_left_px: int = 0,
    window_width_px: int = 256,
    max_shift_px: int = 96,
    fill: int = 128,
    keep_rule: BoxKeepRule | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, WindowRecord]:
    """Scale the strip by ``s`` and cut the output window (§2-2 steps 3-5).

    The strip is resampled as a whole (``INTER_AREA`` down, ``INTER_LINEAR``
    up, no resampling at ``s = 1``) about its top row. The window has the
    strip height and ``window_width_px`` columns and starts at the top row.

    * Plain mode: the crop centre goes to the output centre (``dx = 0``);
      uncovered output is ``fill``. With the crop alone as strip this is
      exactly :func:`scale_isotropic`.
    * Context mode: ``dx`` is drawn uniformly from the integers for which the
      window lies fully inside the scaled strip, intersected with
      ``[-max_shift_px, max_shift_px]``. Needs ``s >= 1``.

    Args:
        strip: Grayscale uint8 ``[H, W_s]``.
        boxes: ``[N, 4]`` boxes in strip px.
        s: Scale factor (> 0).
        mode: ``"context"`` or ``"plain"``.
        rng: Generator for ``dx`` (context mode only).
        crop_left_px: Strip column of the centre crop's first column.
        window_width_px: Output width (the crop width).
        max_shift_px: Shift limit of the context mode.
        fill: Gray value of uncovered output.
        keep_rule: Box keep rule (default :class:`BoxKeepRule`).

    Returns:
        ``(image, boxes, keep_mask, record)``: the output, all ``N`` boxes
        clipped to the visible region, the keep mask and the record.

    Raises:
        ValueError: On a bad image or boxes, ``s <= 0``, a fill outside
            [0, 255], or a context mode window that cannot fit (``s < 1``).
    """
    _check_gray_image(strip)
    box_array = _as_box_array(boxes)
    window_mode = ContextMode(mode)
    if s <= 0:
        raise ValueError(f"scale factor must be > 0, got {s}")
    _check_fill(fill)
    rule = keep_rule if keep_rule is not None else BoxKeepRule()

    height_px, strip_width_px = strip.shape
    if s == 1.0:
        scaled_strip = strip
        scaled_width_px, scaled_height_px = strip_width_px, height_px
    else:
        scaled_width_px = round(strip_width_px * s)
        scaled_height_px = round(height_px * s)
        interpolation = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
        scaled_strip = cv2.resize(
            strip, (scaled_width_px, scaled_height_px), interpolation=interpolation
        )

    centred_left_px = _centred_window_left(
        crop_left_px, window_width_px, strip_width_px, scaled_width_px
    )
    dx_px = dx_min_px = dx_max_px = 0
    if window_mode is ContextMode.CONTEXT:
        if scaled_height_px < height_px:
            raise ValueError(f"context mode needs s >= 1, got {s}")
        # Window inside content: 0 <= left and left + W <= scaled width.
        dx_min_px = max(
            centred_left_px + window_width_px - scaled_width_px, -max_shift_px
        )
        dx_max_px = min(centred_left_px, max_shift_px)
        if dx_min_px > dx_max_px:
            raise ValueError(f"no window position fits inside the strip at s = {s}")
        dx_px = int(rng.integers(dx_min_px, dx_max_px, endpoint=True))

    # Output column j shows scaled-strip column j - offset_x_px.
    offset_x_px = dx_px - centred_left_px
    visible_x1 = max(0, offset_x_px)
    visible_x2 = min(window_width_px, offset_x_px + scaled_width_px)
    visible_y2 = min(height_px, scaled_height_px)
    result = np.full((height_px, window_width_px), fill, dtype=np.uint8)
    result[:visible_y2, visible_x1:visible_x2] = scaled_strip[
        :visible_y2, visible_x1 - offset_x_px : visible_x2 - offset_x_px
    ]

    factor_x = scaled_width_px / strip_width_px
    factor_y = scaled_height_px / height_px
    scaled_boxes = box_array * np.array([factor_x, factor_y, factor_x, factor_y])
    scaled_boxes[:, [0, 2]] += offset_x_px
    visible = (visible_x1, 0, visible_x2, visible_y2)
    clipped, keep_mask = filter_boxes(
        box_areas(scaled_boxes), scaled_boxes, visible, rule
    )
    record = WindowRecord(window_mode, s, dx_px, dx_min_px, dx_max_px, visible)
    return result, clipped, keep_mask, record


# ------------------------------------------------------------ stage (§2-2)


@dataclass(frozen=True)
class ContextRecord:
    """What :func:`augment_context_scale` drew for one image.

    Attributes:
        window: Mode, s, dx and the allowed dx range.
        source: Origin of the context (none / c1 / a).
        used_neighbors: Number of stitched neighbour crops.
        rejections: Why a chain stopped at a neighbour.
    """

    window: WindowRecord
    source: ContextSource
    used_neighbors: int
    rejections: tuple[RejectReason, ...]


def augment_context_scale(
    center_file: str,
    content: CropContent,
    index: ContextIndex,
    loader: CropLoader,
    config: ContextScaleConfig,
    keep_rule: BoxKeepRule,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ContextRecord]:
    """The N2-a' context scale stage for one image (§2-2).

    Draw order: mode, then ``s``, then ``dx`` (context mode only).

    Args:
        center_file: Image file name (index key).
        content: The crop's pixels and labels (crop coordinates).
        index: Context index (used in the context mode only).
        loader: Reads neighbour crops (context mode only).
        config: The ``gpr_context`` settings.
        keep_rule: Box keep rule (``gpr_geometric.keep_rule``).
        rng: Random generator.

    Returns:
        ``(image, boxes, classes, keep_mask, record)``. ``boxes`` and
        ``classes`` are the merged labels (more than the input when
        neighbours add objects), clipped; ``keep_mask`` marks those to keep.
    """
    is_context = bool(rng.random() < config.p_context)
    if is_context:
        strip = assemble_context_strip(center_file, content, index, loader, config)
        low, high = config.context_scale
    else:
        strip = crop_only_strip(content)
        low, high = config.plain_scale
    s = float(rng.uniform(low, high))
    mode = ContextMode.CONTEXT if is_context else ContextMode.PLAIN
    image, boxes, keep_mask, window = place_window(
        strip.image,
        strip.boxes,
        s,
        mode,
        rng,
        crop_left_px=strip.crop_left_px,
        window_width_px=content.image.shape[1],
        max_shift_px=config.max_shift_px,
        fill=config.fill,
        keep_rule=keep_rule,
    )
    if not is_context:
        source = ContextSource.NONE
    elif strip.used_neighbors:
        source = ContextSource.A
    else:
        source = ContextSource.C1
    record = ContextRecord(window, source, strip.used_neighbors, strip.rejections)
    return image, boxes, strip.classes, keep_mask, record
