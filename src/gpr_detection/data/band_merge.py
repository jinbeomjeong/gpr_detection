"""Merge fragmented ``pipe_band`` boxes of one image into one box per layer.

AIHub ``lpipe`` labels often split one horizontal reflection into several
boxes that a gap of a few pixels separates (label audits v1 and v2,
``docs/datasets/audit_pipe_band_shape_v2.md`` section 3). Two boxes of the
same image are merged into their union when all of these hold:

* **same row**: their y overlap is at least ``min_y_overlap_fraction`` of the
  smaller box height;
* **close**: the x gap between them is at most ``max_gap_px`` (overlapping or
  touching boxes have a gap <= 0 and always qualify);
* **adjacent**: no third box lies inside the gap and shares the row with
  either of the two (``labeling/pipe_band_relabel_plan.md`` section 8);
* **no height blow-up**: the union is at most ``max_height_ratio`` times as
  tall as the taller of the two.

The closest pair is merged first and the search restarts until no pair
qualifies, so the result does not depend on the input order beyond ties.
Coordinates are pixels of the original crop (256 x 256 for AIHub v1).
"""

from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_MIN_Y_OVERLAP_FRACTION = 0.5
DEFAULT_MAX_HEIGHT_RATIO = 1.5


@dataclass(frozen=True)
class PixelBox:
    """Axis-aligned box in image pixels, ``x1 < x2`` and ``y1 < y2``."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def union(self, other: "PixelBox") -> "PixelBox":
        """Smallest box that contains both boxes."""
        return PixelBox(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
        )


@dataclass(frozen=True)
class MergeRule:
    """Thresholds of the band merge.

    Attributes:
        max_gap_px: Largest x gap (pixels) that is still bridged.
        min_y_overlap_fraction: Least y overlap, as a fraction of the smaller
            box height, for two boxes to be on the same row.
        max_height_ratio: Largest union height relative to the taller box;
            ``None`` disables the guard.
        require_adjacent: Refuse pairs with a same-row box inside the gap.
    """

    max_gap_px: float
    min_y_overlap_fraction: float = DEFAULT_MIN_Y_OVERLAP_FRACTION
    max_height_ratio: float | None = DEFAULT_MAX_HEIGHT_RATIO
    require_adjacent: bool = True


@dataclass(frozen=True)
class MergeResult:
    """Outcome of :func:`merge_band_boxes`.

    Attributes:
        boxes: Boxes after merging.
        member_indices: For each output box, the sorted input indices it
            covers (a single index when the box was not merged).
        merge_gaps_px: X gap of every merge, in the order the merges happened
            (one entry per merge, so ``len(boxes) + len(merge_gaps_px)`` is
            the input count).
        height_blocked_pairs: Pairs of output boxes that qualify on every
            condition except ``max_height_ratio``.
    """

    boxes: tuple[PixelBox, ...]
    member_indices: tuple[tuple[int, ...], ...]
    merge_gaps_px: tuple[float, ...]
    height_blocked_pairs: int


def y_overlap_fraction(first: PixelBox, second: PixelBox) -> float:
    """Y overlap divided by the smaller height (0 when they do not overlap)."""
    overlap = min(first.y2, second.y2) - max(first.y1, second.y1)
    smaller_height = min(first.height, second.height)
    if overlap <= 0 or smaller_height <= 0:
        return 0.0
    return overlap / smaller_height


def x_gap_px(first: PixelBox, second: PixelBox) -> float:
    """Horizontal gap between the boxes; <= 0 when they touch or overlap in x."""
    return max(first.x1, second.x1) - min(first.x2, second.x2)


def _is_same_row(first: PixelBox, second: PixelBox, rule: MergeRule) -> bool:
    return y_overlap_fraction(first, second) >= rule.min_y_overlap_fraction


def _has_same_row_box_in_gap(
    first: PixelBox, second: PixelBox, others: Sequence[PixelBox], rule: MergeRule
) -> bool:
    """True if a box of ``others`` reaches into the x gap on the same row.

    Matches the audit definition: the third box overlaps the open gap
    interval in x and is on the same row as either of the two boxes.
    """
    gap_left = min(first.x2, second.x2)
    gap_right = max(first.x1, second.x1)
    for other in others:
        reaches_into_gap = other.x1 < gap_right and other.x2 > gap_left
        if not reaches_into_gap:
            continue
        if _is_same_row(other, first, rule) or _is_same_row(other, second, rule):
            return True
    return False


def _is_too_tall(first: PixelBox, second: PixelBox, rule: MergeRule) -> bool:
    if rule.max_height_ratio is None:
        return False
    taller_height = max(first.height, second.height)
    return first.union(second).height > rule.max_height_ratio * taller_height


def _qualifies_without_height(
    boxes: Sequence[PixelBox], i: int, j: int, rule: MergeRule
) -> bool:
    first, second = boxes[i], boxes[j]
    if not _is_same_row(first, second, rule):
        return False
    gap = x_gap_px(first, second)
    if gap > rule.max_gap_px:
        return False
    if gap <= 0 or not rule.require_adjacent:
        return True
    others = [box for k, box in enumerate(boxes) if k not in (i, j)]
    return not _has_same_row_box_in_gap(first, second, others, rule)


def _closest_mergeable_pair(
    boxes: Sequence[PixelBox], rule: MergeRule
) -> tuple[float, int, int] | None:
    """``(gap, i, j)`` of the closest pair that may merge, or ``None``."""
    best: tuple[float, int, int] | None = None
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if not _qualifies_without_height(boxes, i, j, rule):
                continue
            if _is_too_tall(boxes[i], boxes[j], rule):
                continue
            candidate = (x_gap_px(boxes[i], boxes[j]), i, j)
            if best is None or candidate < best:
                best = candidate
    return best


def _count_height_blocked_pairs(boxes: Sequence[PixelBox], rule: MergeRule) -> int:
    blocked = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _qualifies_without_height(boxes, i, j, rule) and _is_too_tall(
                boxes[i], boxes[j], rule
            ):
                blocked += 1
    return blocked


def merge_band_boxes(boxes: Sequence[PixelBox], rule: MergeRule) -> MergeResult:
    """Merge same-row band fragments of one image, closest pair first.

    Args:
        boxes: ``pipe_band`` boxes of one image in pixels. Boxes of other
            classes must not be passed.
        rule: Merge thresholds.

    Returns:
        The merged boxes (a merged box takes the place of the earlier of its
        two parts), which input boxes each covers, and the merge gaps.

    Raises:
        ValueError: If ``rule.max_gap_px`` is negative.
    """
    if rule.max_gap_px < 0:
        raise ValueError(f"max_gap_px must be >= 0, got {rule.max_gap_px}")
    current = list(boxes)
    members: list[tuple[int, ...]] = [(index,) for index in range(len(boxes))]
    merge_gaps_px = []
    while (pair := _closest_mergeable_pair(current, rule)) is not None:
        gap, i, j = pair
        current[i] = current[i].union(current[j])
        members[i] = tuple(sorted(members[i] + members[j]))
        del current[j]
        del members[j]
        merge_gaps_px.append(gap)
    return MergeResult(
        boxes=tuple(current),
        member_indices=tuple(members),
        merge_gaps_px=tuple(merge_gaps_px),
        height_blocked_pairs=_count_height_blocked_pairs(current, rule),
    )
