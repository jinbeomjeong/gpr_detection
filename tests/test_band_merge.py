"""Tests for the pipe_band fragment merge (plan section 8, audit v2 section 3)."""

import pytest

from gpr_detection.data.aihub_yolo_v2_merge import merge_label_text
from gpr_detection.data.band_merge import (
    MergeRule,
    PixelBox,
    merge_band_boxes,
    x_gap_px,
    y_overlap_fraction,
)

RULE_32 = MergeRule(max_gap_px=32)
ROW_TOP_PX = 40.0
ROW_BOTTOM_PX = 56.0  # 16 px tall, the v1 median band height
IMAGE_SIDE_PX = 256


def band(x1: float, x2: float, y1: float = ROW_TOP_PX, y2: float = ROW_BOTTOM_PX):
    return PixelBox(x1, y1, x2, y2)


@pytest.mark.parametrize(
    ("gap_px", "expected_boxes"), [(8, 1), (32, 1), (33, 2), (0, 1), (-10, 1)]
)
def test_adjacent_pair_merges_up_to_the_gap_limit(
    gap_px: float, expected_boxes: int
) -> None:
    result = merge_band_boxes([band(10, 60), band(60 + gap_px, 120)], RULE_32)

    assert len(result.boxes) == expected_boxes
    if expected_boxes == 1:
        assert result.boxes[0] == band(10, 120)
        assert result.merge_gaps_px == (gap_px,)
        assert result.member_indices == ((0, 1),)


def test_mergeable_box_between_joins_the_chain() -> None:
    boxes = [band(0, 50), band(65, 75), band(90, 140)]

    result = merge_band_boxes(boxes, RULE_32)

    assert result.boxes == (band(0, 140),)
    assert result.merge_gaps_px == (15, 15)


def test_box_between_on_the_same_row_makes_the_outer_pair_non_adjacent() -> None:
    # The middle box sits in the 20 px gap on the row of both outer boxes but
    # may not merge with them (its union would be 32 px tall > 1.25 x 24 px).
    left, right = band(0, 50), band(70, 120)
    middle = PixelBox(55, 48, 65, 72)
    rule = MergeRule(max_gap_px=32, max_height_ratio=1.25)

    result = merge_band_boxes([left, middle, right], rule)

    assert x_gap_px(left, right) == 20
    assert y_overlap_fraction(middle, left) == 0.5
    assert result.boxes == (left, middle, right)
    assert result.height_blocked_pairs == 2  # middle-left and middle-right

    no_adjacency = MergeRule(
        max_gap_px=32, max_height_ratio=1.25, require_adjacent=False
    )
    assert merge_band_boxes([left, middle, right], no_adjacency).boxes == (
        band(0, 120),
        middle,
    )


@pytest.mark.parametrize(
    ("shifted_top_px", "expected_boxes"),
    [(ROW_TOP_PX + 8, 1), (ROW_TOP_PX + 8.16, 2)],  # overlap 50 % / 49 %
)
def test_y_overlap_must_reach_half_of_the_smaller_height(
    shifted_top_px: float, expected_boxes: int
) -> None:
    shifted = band(70, 120, shifted_top_px, shifted_top_px + 16)
    rule = MergeRule(max_gap_px=32, max_height_ratio=None)

    result = merge_band_boxes([band(10, 60), shifted], rule)

    assert len(result.boxes) == expected_boxes


def test_merging_repeats_until_stable_closest_pair_first() -> None:
    boxes = [band(0, 40), band(70, 100), band(108, 150), band(170, 256)]

    result = merge_band_boxes(boxes, RULE_32)

    assert result.boxes == (band(0, 256),)
    # Gaps 8 (100-108), then 20 (150-170), then 30 (40-70).
    assert result.merge_gaps_px == (8, 20, 30)
    assert result.member_indices == ((0, 1, 2, 3),)


def test_stacked_rows_stay_separate() -> None:
    upper = band(0, 100, 20, 36)
    lower = band(0, 100, 60, 76)

    result = merge_band_boxes([upper, lower], RULE_32)

    assert result.boxes == (upper, lower)
    assert result.merge_gaps_px == ()


def test_height_guard_blocks_tall_unions() -> None:
    short = band(0, 50, 40, 56)
    tall_offset = band(60, 120, 47, 71)  # overlap 9/16 >= 0.5, union 31 px
    guarded = MergeRule(max_gap_px=32, max_height_ratio=1.25)

    result = merge_band_boxes([short, tall_offset], guarded)

    assert len(result.boxes) == 2
    assert result.height_blocked_pairs == 1
    assert len(merge_band_boxes([short, tall_offset], RULE_32).boxes) == 1


def test_negative_gap_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_gap_px"):
        merge_band_boxes([band(0, 10)], MergeRule(max_gap_px=-1))


def _line(class_id: int, x1: float, y1: float, x2: float, y2: float) -> str:
    values = (
        (x1 + x2) / 2 / IMAGE_SIDE_PX,
        (y1 + y2) / 2 / IMAGE_SIDE_PX,
        (x2 - x1) / IMAGE_SIDE_PX,
        (y2 - y1) / IMAGE_SIDE_PX,
    )
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in values)


def test_label_merge_leaves_other_classes_and_unmerged_bands_untouched() -> None:
    cavity = _line(2, 10, 120, 60, 170)
    hyperbola_same_row = _line(0, 62, 40, 68, 56)  # class 0 inside the gap
    band_left = _line(1, 0, 40, 60, 56)
    band_right = _line(1, 70, 40, 128, 56)
    band_far = _line(1, 200, 100, 240, 116)
    text = "\n".join([cavity, band_left, hyperbola_same_row, band_right, band_far])

    merged = merge_label_text(text + "\n", RULE_32, IMAGE_SIDE_PX, IMAGE_SIDE_PX)

    output_lines = merged.text.splitlines()
    assert [output_lines[0], *output_lines[2:]] == [
        cavity,
        hyperbola_same_row,
        band_far,
    ]
    union_fields = output_lines[1].split()
    assert union_fields[0] == "1"
    expected_union = [float(v) for v in _line(1, 0, 40, 128, 56).split()[1:]]
    assert [float(v) for v in union_fields[1:]] == pytest.approx(
        expected_union, abs=2e-6
    )
    assert (merged.band_boxes_before, merged.band_boxes_after) == (3, 2)
    assert merged.merge_gaps_px == pytest.approx((10,), abs=1e-3)


def test_label_merge_without_bands_returns_the_same_lines() -> None:
    text = _line(3, 0, 0, 100, 100) + "\n" + _line(0, 50, 50, 70, 90) + "\n"

    merged = merge_label_text(text, RULE_32, IMAGE_SIDE_PX, IMAGE_SIDE_PX)

    assert merged.text == text
    assert merged.band_boxes_before == merged.band_boxes_after == 0
