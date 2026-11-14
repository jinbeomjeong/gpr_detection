"""M3-2 S5 match and box rules (``analysis/detection/m32_recipe_match.py``).

One-to-one apex match, M4a-M4d classification on the content range, the
C-5 conf grid evaluation, the M5 mirror pairing and the D3 M6 rule.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "detection"))
import m32_recipe_match as match

from gpr_detection.io.classes import DetectionClass
from gpr_detection.io.radargram import GridBox

HYPERBOLA = DetectionClass.PIPE_HYPERBOLA
BAND = DetectionClass.PIPE_BAND
PERIOD_SAMPLES = 20.0
PERIOD_TRACES = 6.0


def _box(i0: int, i1: int, k0: int, k1: int, score: float = 0.5, cls: int = HYPERBOLA):
    return match.ScoredBox(cls, score, GridBox(i0, i1, k0, k1))


def _apex(key: str, trace: float, sample: float) -> match.TruthApex:
    return match.TruthApex(key, trace, sample)


# ----------------------------------------------------------- one-to-one


def test_box_grown_by_three_traces_and_half_a_period_contains_the_apex() -> None:
    box = GridBox(10, 20, 100, 140)
    assert match.box_contains_apex(box, _apex("a", 7.0, 90.0), PERIOD_SAMPLES)
    assert match.box_contains_apex(box, _apex("a", 23.0, 150.0), PERIOD_SAMPLES)
    assert not match.box_contains_apex(box, _apex("a", 6.9, 120.0), PERIOD_SAMPLES)
    assert not match.box_contains_apex(box, _apex("a", 15.0, 150.1), PERIOD_SAMPLES)


def test_one_box_over_two_apexes_hits_only_the_one_nearest_its_centre() -> None:
    apexes = [_apex("left", 10.0, 120.0), _apex("right", 40.0, 120.0)]
    wide = _box(5, 45, 100, 140)  # centre trace 25.5: right is 14.5 away, left 15.5

    outcomes = match.match_one_to_one(apexes, [wide], PERIOD_TRACES, PERIOD_SAMPLES)

    assert [o.hit for o in outcomes] == [False, True]
    assert outcomes[1].box_index == 0


def test_two_boxes_on_one_apex_count_one_hit_with_the_best_score() -> None:
    apexes = [_apex("a", 20.0, 120.0)]
    boxes = [_box(15, 25, 110, 130, 0.4), _box(14, 26, 108, 132, 0.9)]

    (outcome,) = match.match_one_to_one(apexes, boxes, PERIOD_TRACES, PERIOD_SAMPLES)

    assert outcome == match.MatchOutcome(hit=True, box_index=1, n_boxes=2)


def test_only_pipe_hyperbola_boxes_match_unless_any_class_is_asked() -> None:
    apexes = [_apex("a", 20.0, 120.0)]
    band = [_box(0, 60, 110, 130, cls=BAND)]

    assert not match.match_one_to_one(apexes, band, PERIOD_TRACES, PERIOD_SAMPLES)[
        0
    ].hit
    agnostic = match.match_one_to_one(
        apexes, band, PERIOD_TRACES, PERIOD_SAMPLES, class_id=None
    )
    assert agnostic[0].hit


def test_equal_distances_go_to_the_earlier_apex() -> None:
    apexes = [_apex("first", 10.0, 120.0), _apex("second", 30.0, 120.0)]
    box = _box(10, 29, 100, 139)  # continuous centre trace 20.0

    outcomes = match.match_one_to_one(apexes, [box], PERIOD_TRACES, PERIOD_SAMPLES)

    assert [o.hit for o in outcomes] == [True, False]


def test_no_boxes_means_no_hits() -> None:
    outcomes = match.match_one_to_one(
        [_apex("a", 1.0, 1.0)], [], PERIOD_TRACES, PERIOD_SAMPLES
    )
    assert outcomes == [match.NO_MATCH]


def test_grid_iou_counts_cells() -> None:
    a, b = GridBox(0, 9, 0, 9), GridBox(5, 14, 0, 9)
    assert match.grid_iou(a, b) == pytest.approx(50 / 150)
    assert match.grid_iou(a, GridBox(20, 30, 0, 9)) == 0.0


# ------------------------------------------------------------------ M4


CONTENT = match.ContentRange(
    y_top_px=0.0, y_bot_px=120.0, period_l1_px=6.0, zero_px=10.0
)


@pytest.mark.parametrize(
    ("y1", "y2", "expected"),
    [
        (118.0, 140.0, match.M4Category.A),  # centre 129 in the padding
        (100.0, 130.0, match.M4Category.B),  # straddles 118..122
        (100.0, 121.0, None),  # bottom edge inside the 2 px margin
        (5.0, 20.0, match.M4Category.D),  # centre 12.5 above zero + 3
        (5.0, 21.0, None),  # centre 13.0 is not above 13.0
        (50.0, 70.0, None),
    ],
)
def test_m4_category(y1: float, y2: float, expected) -> None:
    assert match.m4_category(y1, y2, CONTENT) == expected


def test_m4d_is_skipped_when_the_zero_failed() -> None:
    no_zero = match.ContentRange(0.0, 120.0, 6.0, zero_px=None)
    assert match.m4_category(5.0, 20.0, no_zero) is None


def test_m4c_is_a_band_near_the_bottom_that_is_not_m4a_or_m4b() -> None:
    assert match.is_m4c(BAND, 90.0, 102.0, CONTENT)  # y2 >= 120 - 18
    assert not match.is_m4c(BAND, 90.0, 101.9, CONTENT)
    assert not match.is_m4c(BAND, 100.0, 130.0, CONTENT)  # M4b
    assert not match.is_m4c(HYPERBOLA, 90.0, 110.0, CONTENT)
    assert match.is_m4c(BAND, 100.0, 112.0, CONTENT, match.M4C_D3_WINDOW_PX)
    assert not match.is_m4c(BAND, 100.0, 111.0, CONTENT, match.M4C_D3_WINDOW_PX)


# -------------------------------------------------------------- conf grid


def test_conf_grid_is_025_to_095_by_001() -> None:
    grid = match.conf_grid()
    assert grid[0] == 0.25 and grid[-1] == 0.95 and len(grid) == 71
    assert all(round(b - a, 2) == 0.01 for a, b in itertools.pairwise(grid))


def test_evaluate_on_conf_grid_keeps_strictly_higher_scores_and_caches() -> None:
    calls = []

    def count(keep: np.ndarray) -> int:
        calls.append(keep.copy())
        return int(keep.sum())

    counts = match.evaluate_on_conf_grid([0.3, 0.5, 0.26], count)

    assert counts[0] == 3  # 0.25
    assert counts[1] == 2  # 0.26: 0.26 is not > 0.26
    assert counts[match.conf_grid().index(0.3)] == 1
    assert counts[-1] == 0
    assert len(calls) == 4  # one call per distinct kept set


def test_hit_mask_text() -> None:
    assert match.hit_mask_text([True, False, True]) == "101"


# --------------------------------------------------------------- M5 / M6


def test_flip_grid_box_mirrors_the_trace_range() -> None:
    assert match.flip_grid_box(GridBox(0, 9, 3, 4), 100) == GridBox(90, 99, 3, 4)


def test_flip_pairs_match_same_class_iou_and_use_a_box_once() -> None:
    original = [_box(0, 9, 0, 9, 0.9), _box(1, 10, 0, 9, 0.8), _box(50, 60, 0, 9, 0.7)]
    flipped = [_box(0, 9, 0, 9), _box(50, 60, 0, 9, cls=BAND)]

    pairs = match.flip_pairs(original, flipped)

    assert pairs[0][:2] == (0, 0)
    assert pairs[1][1] is None  # its best flipped box is already used
    assert pairs[2][1] is None and pairs[2][2] == 0.0  # other class only


def test_m6_uses_the_d3_rule_without_one_to_one() -> None:
    box = [_box(10, 20, 100, 140)]
    assert match.apex_hit_d3(15, 120.0, box, PERIOD_SAMPLES)
    assert match.apex_hit_d3(16, 121.0, box, PERIOD_SAMPLES)  # same box, again a hit
    assert not match.apex_hit_d3(30, 120.0, box, PERIOD_SAMPLES)
