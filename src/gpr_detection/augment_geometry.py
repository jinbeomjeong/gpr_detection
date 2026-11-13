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

This module lives next to :mod:`gpr_detection.augment` instead of inside it
because the architecture rule 5 keeps ``augment`` free of cv2
(``docs/architecture/README.md``).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

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


def geometric_config_from_dict(settings: Mapping[str, Any]) -> GeometricAugmentConfig:
    """Build a :class:`GeometricAugmentConfig` from the ``gpr_geometric`` YAML block.

    Missing sections keep their defaults.

    Args:
        settings: Mapping with optional sections ``scale``, ``horizontal_shift``,
            ``edge_fill`` and ``keep_rule``.

    Raises:
        ValueError: If the block has an unknown section (typo protection).
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
