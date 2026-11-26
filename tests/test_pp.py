"""M3-3 boundary post-processing PP (``docs/experiments/m33_pp_spec.md`` v0.2.3 §5).

1. ``pp_mode`` off (``S0``, named since the M3-3 close made S3 the default)
   is bit-identical (fixture line; opt-in: the r0 11 lines). The operational
   default itself is pinned in ``tests/test_pp_default.py``.
2. The category is exclusive (a > b > d) and equals M3-2 ``m4_category``.
3. PP-b clips to ``y_bot``, removes below 1 ``P_ell``, keeps boxes within +-e.
4. PP-d is off on a failed zero and switches at ``y_z + 0.5 P_ell``.
5. PP-c: c(0.35) removals are a subset of c(0.50) removals; bands only.
6. Shuffling the boxes does not change any decision.
7. The D0 runner refuses a forbidden input path with exit 2.

Also pins the ``PPConfig`` hashes (C0a identity) and the nested set sequence.
"""

import itertools
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fake_line_detector import TileStatisticsDetector
from gpr_detection import pipeline
from gpr_detection.detect.base import PixelBox
from gpr_detection.pipeline import (
    PP_SETS,
    ContentRows,
    PPAction,
    PPCategory,
    PPConfig,
    PPRule,
    PPSetId,
    apply_pp,
    content_rows_for,
    detect_line,
    detect_tiles,
    is_removed_inside_content,
    kept_after_pp,
    line_boxes_to_grid,
    merge_line_boxes,
    pp_category,
    pp_decision,
    pp_sets_sha256,
)
from gpr_detection.preprocess.timezero import given_zero
from test_pipeline_tiles import (
    BAND,
    CAVITY,
    HYPERBOLA,
    VisiblePartDetector,
    fixture_image,
    line_box,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DETECTION = PROJECT_ROOT / "analysis" / "detection"
MANHOLE = 3
Y_BOT = 81.0  # fixture_image content height
PERIOD = 4.05
ZERO_PX = 10.0
CONTENT = ContentRows(y_bot_px=Y_BOT, period_px=PERIOD, zero_px=ZERO_PX)
NO_ZERO = replace(CONTENT, zero_px=None)
EDGE = 2.0

PINNED_SET_SHA256 = {
    PPSetId.S0: "fc9f0c8eab77",
    PPSetId.S1: "78de0a20390e",
    PPSetId.S2: "184c07e44c0c",
    PPSetId.S3: "eb2796ddfaa9",
    PPSetId.S4: "af29b47126e7",
    PPSetId.S5: "b87bbc145e78",
}
"""Hash prefixes of the C0a constants; a changed constant must fail here."""


def _box(y1: float, y2: float, class_id: int = HYPERBOLA, score: float = 0.9):
    return line_box(10.0, y1, 30.0, y2, 0, class_id=class_id, score=score)


def _action(y1, y2, set_id, content=CONTENT, class_id=HYPERBOLA, score=0.9):
    return pp_decision(class_id, score, y1, y2, content, PP_SETS[set_id]).action


# ------------------------------------------------------- sets and hashes


def test_sets_are_nested_and_c_cut_rises():
    rules = [set(PP_SETS[key].rules) for key in PPSetId]
    assert rules[0] == set()
    for smaller, larger in itertools.pairwise(rules[1:]):
        assert smaller <= larger
    assert PP_SETS[PPSetId.S4].band_score_cut == 0.35
    assert PP_SETS[PPSetId.S5].band_score_cut == 0.50
    assert PP_SETS[PPSetId.S3].rules == (PPRule.A, PPRule.B, PPRule.D)


def test_config_constants_and_pinned_hashes():
    for key, config in PP_SETS.items():
        assert config.edge_px == 2.0
        assert config.band_window_px == 18.0
        assert config.zero_margin_periods == 0.5
        assert config.min_clipped_height_periods == 1.0
        assert config.sha256().startswith(PINNED_SET_SHA256[key]), key
    assert len(pp_sets_sha256()) == 64
    assert PP_SETS[PPSetId.S0].is_off and not PP_SETS[PPSetId.S1].is_off


def test_config_rejects_unsorted_rules_and_a_stray_cut():
    with pytest.raises(ValueError, match="sorted"):
        PPConfig(PPSetId.S2, (PPRule.D, PPRule.A))
    with pytest.raises(ValueError, match="rule c"):
        PPConfig(PPSetId.S1, (PPRule.A,), band_score_cut=0.3)
    with pytest.raises(ValueError, match="rule c"):
        PPConfig(PPSetId.S4, (PPRule.A, PPRule.C))


# ------------------------------------------------------- 1 off is identical


def _boundary_boxes() -> list[tuple[int, float, PixelBox]]:
    """Boxes hitting every rule on the fixture (y_bot 81, P_L1 4.05)."""
    return [
        (HYPERBOLA, 0.9, PixelBox(100, 30, 140, 60)),  # untouched
        (HYPERBOLA, 0.8, PixelBox(300, 70, 340, 100)),  # a: centre 85 > 81
        (BAND, 0.7, PixelBox(500, 60, 700, 90)),  # b: clipped to 81
        (CAVITY, 0.6, PixelBox(800, 79, 830, 84)),  # inside +-e: untouched
        (HYPERBOLA, 0.5, PixelBox(900, 0, 930, 8)),  # d with zero 10
        (BAND, 0.3, PixelBox(1000, 66, 1200, 72)),  # c(0.35) and c(0.50)
        (BAND, 0.4, PixelBox(1300, 66, 1500, 72)),  # c(0.50) only
        (MANHOLE, 0.1, PixelBox(1600, 66, 1630, 72)),  # not a band: no c
    ]


def _pp_fixture_image():
    """T8 fixture whose grid-map bottom is exactly H_c 81 (as real images)."""
    image = fixture_image(width_px=2250)
    grid = replace(image.grid_map, samples_per_px=image.grid_map.n_samples / Y_BOT)
    return replace(image, grid_map=grid)


def _use_fixture(monkeypatch, image, zero_k: float = 63.2):
    monkeypatch.setattr(pipeline, "make_detection_image", lambda *a, **k: image)
    zero = given_zero(zero_k, 1.0)  # dt 1 ns: k_zero = zero_k
    monkeypatch.setattr(pipeline, "line_zero", lambda radargram: zero)


def test_1_pp_off_is_the_pre_pp_path_bit_for_bit(monkeypatch):
    image = _pp_fixture_image()
    boxes = _boundary_boxes()
    _use_fixture(monkeypatch, image)

    off = detect_line(
        object(),  # type: ignore[arg-type]
        VisiblePartDetector(image, boxes),
        pp_mode=PPSetId.S0,
    )
    merged = merge_line_boxes(
        detect_tiles(image, VisiblePartDetector(image, boxes)), image
    )
    reference = line_boxes_to_grid(merged, image)

    assert off.boxes == tuple(reference)
    assert off.pp_outcomes == () and off.pp_config is None
    content = content_rows_for(image, given_zero(63.2, 1.0))
    outcomes = apply_pp(merged, content, PP_SETS[PPSetId.S0])
    assert all(o.output is o.original and o.action is PPAction.NONE for o in outcomes)
    assert kept_after_pp(outcomes) == merged


@pytest.mark.data
@pytest.mark.realdata
def test_1_pp_off_is_identical_on_the_r0_eleven_lines():
    """Opt-in (-m realdata): the spec section 9 lines through r0 with PP off."""
    from gpr_detection.io.paths import load_dataset_paths
    from gpr_detection.preprocess.to_image import make_detection_image
    from test_detection_image_data import CANDIDATES

    paths = load_dataset_paths()
    assert len(CANDIDATES) == 11
    for candidate in CANDIDATES:
        root = paths.root(candidate.dataset)
        if not root.exists():
            pytest.skip(f"dataset root missing: {candidate.dataset}")
        radargram = candidate.load(root)
        off = detect_line(radargram, TileStatisticsDetector(), pp_mode=PPSetId.S0)
        image = make_detection_image(radargram)
        merged = merge_line_boxes(detect_tiles(image, TileStatisticsDetector()), image)
        assert off.boxes == tuple(line_boxes_to_grid(merged, image))
        on = detect_line(radargram, TileStatisticsDetector(), pp_mode=PPSetId.S5)
        assert len(on.pp_outcomes) == len(merged), candidate.name


def test_pp_on_drops_and_clips_before_the_grid(monkeypatch):
    image = _pp_fixture_image()
    _use_fixture(monkeypatch, image)
    detections = detect_line(
        object(),  # type: ignore[arg-type]
        VisiblePartDetector(image, _boundary_boxes()),
        pp_mode=PPSetId.S5,
    )
    outcomes = detections.pp_outcomes
    assert {o.action for o in outcomes} == set(PPAction) - {PPAction.REMOVED_B}
    kept = [o.output for o in outcomes if o.output is not None]
    assert [box for box, _ in detections.boxes] == kept
    for outcome in outcomes:
        if outcome.action is PPAction.CLIPPED_B:
            assert outcome.output is not None and outcome.output.box.y2 == Y_BOT
        if outcome.action is PPAction.REMOVED_C:
            assert outcome.original.class_id == BAND
    manholes = [o for o in outcomes if o.original.class_id == MANHOLE]
    assert manholes and all(o.action is PPAction.NONE for o in manholes)
    assert detections.pp_config is PP_SETS[PPSetId.S5]


# ------------------------------------------------------- 2 categories


def _m4_module():
    sys.path.insert(0, str(ANALYSIS_DETECTION))
    try:
        import m32_recipe_match
    finally:
        sys.path.remove(str(ANALYSIS_DETECTION))
    return m32_recipe_match


def test_2_category_is_exclusive_and_equals_m4_category():
    match = _m4_module()
    rng = np.random.default_rng(20261115)
    for zero in (None, 0.0, 10.0, 37.3):
        rows = ContentRows(Y_BOT, PERIOD, zero)
        m4_rows = match.ContentRange(0.0, Y_BOT, PERIOD, zero)
        for _ in range(4000):
            y1 = float(rng.uniform(-5, 120))
            y2 = y1 + float(rng.uniform(0.5, 60))
            ours = pp_category(y1, y2, rows)
            theirs = match.m4_category(y1, y2, m4_rows)
            assert (None if theirs is None else theirs.value[-1]) == (
                None if ours is None else ours.value
            )


def test_2_category_order_a_before_b_before_d():
    low_zero = ContentRows(Y_BOT, PERIOD, zero_px=200.0)  # every centre is "above"
    assert pp_category(70, 100, low_zero) is PPCategory.A
    assert pp_category(60, 90, low_zero) is PPCategory.B
    assert pp_category(30, 60, low_zero) is PPCategory.D


# ------------------------------------------------------- 3 PP-b


def test_3_b_clips_to_the_content_bottom():
    decision = pp_decision(HYPERBOLA, 0.9, 60, 90, CONTENT, PP_SETS[PPSetId.S3])
    assert decision.action is PPAction.CLIPPED_B and decision.y2_px == Y_BOT


def test_3_b_removes_a_clipped_box_lower_than_one_period():
    top = Y_BOT - PERIOD
    bottom = Y_BOT + 3.0  # straddles by more than e, centre above y_bot
    assert _action(top + 0.01, bottom, PPSetId.S3) is PPAction.REMOVED_B
    assert _action(top - 1e-9, bottom, PPSetId.S3) is PPAction.CLIPPED_B
    assert _action(top - 0.01, bottom, PPSetId.S3) is PPAction.CLIPPED_B


def test_3_b_keeps_boxes_within_the_edge_margin():
    assert _action(60, Y_BOT + EDGE, PPSetId.S3) is PPAction.NONE
    assert _action(Y_BOT - EDGE, 85, PPSetId.S5) is not PPAction.CLIPPED_B
    assert _action(60, Y_BOT + EDGE + 1e-9, PPSetId.S3) is PPAction.CLIPPED_B


def test_3_b_is_inactive_in_s1_and_s2():
    for set_id in (PPSetId.S1, PPSetId.S2):
        assert _action(60, 90, set_id) is PPAction.NONE


def test_clip_changes_only_y2(monkeypatch):
    box = replace(_box(60, 90, class_id=BAND, score=0.42), n_tiles_seen=3)
    box = replace(box, tile_truncated=True, merged_from_tiles=(0, 1, 2))
    (outcome,) = apply_pp([box], CONTENT, PP_SETS[PPSetId.S3])
    assert outcome.output == replace(box, box=replace(box.box, y2=Y_BOT))


# ------------------------------------------------------- PP-a


def test_a_removes_a_centre_below_the_content_bottom():
    assert _action(72, 90.02, PPSetId.S1) is PPAction.REMOVED_A
    assert _action(72, 90.0, PPSetId.S1) is not PPAction.REMOVED_A  # centre == y_bot
    assert _action(72, 90.02, PPSetId.S0) is PPAction.NONE


def test_a_diagnostic_counts_tops_inside_the_content():
    assert is_removed_inside_content(Y_BOT - PERIOD - 0.1, PPAction.REMOVED_A, CONTENT)
    assert not is_removed_inside_content(Y_BOT - PERIOD, PPAction.REMOVED_A, CONTENT)
    assert not is_removed_inside_content(10.0, PPAction.REMOVED_D, CONTENT)


# ------------------------------------------------------- 4 PP-d


def test_4_d_is_off_when_the_zero_failed():
    assert _action(0, 8, PPSetId.S3, content=NO_ZERO) is PPAction.NONE
    assert pp_category(0, 8, NO_ZERO) is None


def test_4_d_boundary_at_zero_plus_half_period():
    limit = ZERO_PX + 0.5 * PERIOD
    half_height = 3.0
    below = limit - 1e-6
    above = limit + 1e-6
    assert (
        _action(below - half_height, below + half_height, PPSetId.S2)
        is PPAction.REMOVED_D
    )
    assert _action(limit - half_height, limit + half_height, PPSetId.S2) is (
        PPAction.NONE
    )
    assert _action(above - half_height, above + half_height, PPSetId.S2) is (
        PPAction.NONE
    )
    assert _action(0, 8, PPSetId.S1) is PPAction.NONE  # d not in S1


# ------------------------------------------------------- 5 PP-c


def _c_removed(boxes, set_id):
    outcomes = apply_pp(boxes, CONTENT, PP_SETS[set_id])
    return {id(o.original) for o in outcomes if o.action is PPAction.REMOVED_C}


def test_5_c_cut_sequence_is_nested_and_band_only():
    rng = random.Random(5)
    boxes = []
    for _ in range(600):
        y1 = rng.uniform(15, 75)
        y2 = min(y1 + rng.uniform(1, 20), Y_BOT + EDGE)
        class_id = rng.choice([HYPERBOLA, BAND, CAVITY, MANHOLE])
        boxes.append(_box(y1, y2, class_id=class_id, score=rng.uniform(0, 1)))
    lower, upper = _c_removed(boxes, PPSetId.S4), _c_removed(boxes, PPSetId.S5)
    assert lower and lower < upper
    by_id = {id(box): box for box in boxes}
    assert all(by_id[key].class_id == BAND for key in upper)
    assert not _c_removed(boxes, PPSetId.S3)


def test_5_c_window_and_score_boundaries():
    window_top = Y_BOT - 18.0
    assert _action(50, window_top, PPSetId.S5, class_id=BAND, score=0.49) is (
        PPAction.REMOVED_C
    )
    assert _action(50, window_top - 1e-6, PPSetId.S5, class_id=BAND, score=0.1) is (
        PPAction.NONE
    )
    assert _action(50, 70, PPSetId.S5, class_id=BAND, score=0.50) is PPAction.NONE
    assert _action(50, 70, PPSetId.S4, class_id=BAND, score=0.35) is PPAction.NONE
    assert _action(50, 70, PPSetId.S4, class_id=BAND, score=0.3499) is (
        PPAction.REMOVED_C
    )


def test_5_c_never_looks_at_a_categorised_box():
    # Category d (zero 10) with an active d: removed_d, never removed_c.
    assert _action(0, 8, PPSetId.S5, class_id=BAND, score=0.1) is PPAction.REMOVED_D
    # Category b straddles the bottom: clipped, not removed by c.
    assert _action(60, 90, PPSetId.S5, class_id=BAND, score=0.1) is PPAction.CLIPPED_B


# ------------------------------------------------------- 6 order


def test_6_shuffled_boxes_give_the_same_decisions():
    rng = random.Random(6)
    boxes = []
    for _ in range(200):
        y1 = rng.uniform(-5, 110)
        class_id = rng.choice([HYPERBOLA, BAND, CAVITY, MANHOLE])
        boxes.append(_box(y1, y1 + rng.uniform(1, 40), class_id, rng.uniform(0, 1)))
    for set_id in PPSetId:
        reference = {
            id(o.original): (o.action, o.output)
            for o in apply_pp(boxes, CONTENT, PP_SETS[set_id])
        }
        shuffled = boxes[:]
        rng.shuffle(shuffled)
        again = {
            id(o.original): (o.action, o.output)
            for o in apply_pp(shuffled, CONTENT, PP_SETS[set_id])
        }
        assert again == reference


# ------------------------------------------- Y1 period override (§4-3 Y1)


def _image_fields(image) -> dict:
    fields = dict(vars(image))
    fields.pop("line_image")
    return fields


@pytest.mark.parametrize("processed", [False, True])
def test_period_override_none_is_bit_identical(processed):
    """``make_detection_image(..., period_samples_override=None)`` == before."""
    from gpr_detection.preprocess.to_image import make_detection_image
    from synthetic_bscan import BScanSpec, make_radargram

    radargram = make_radargram(BScanSpec(), amplitude_processed=processed)
    default = make_detection_image(radargram)
    explicit = make_detection_image(radargram, period_samples_override=None)
    np.testing.assert_array_equal(default.line_image, explicit.line_image)
    assert default.line_image.dtype == explicit.line_image.dtype
    assert _image_fields(default) == _image_fields(explicit)


def test_period_override_fixes_only_the_period():
    from gpr_detection.preprocess.to_image import PeriodSource, make_detection_image
    from synthetic_bscan import BScanSpec, make_radargram

    radargram = make_radargram(BScanSpec())
    measured = make_detection_image(radargram)
    same = make_detection_image(
        radargram, period_samples_override=measured.period_samples
    )
    np.testing.assert_array_equal(measured.line_image, same.line_image)
    assert same.period_source == PeriodSource.OVERRIDE
    fields = _image_fields(same)
    fields["period_source"] = measured.period_source
    assert fields == _image_fields(measured)
    other = make_detection_image(
        radargram, period_samples_override=1.5 * measured.period_samples
    )
    assert other.period_samples == 1.5 * measured.period_samples
    assert "period_fallback" not in other.flags
    with pytest.raises(ValueError, match="positive"):
        make_detection_image(radargram, period_samples_override=0.0)


# ------------------------------------------------------- 7 D0 forbidden path


def _d0_module():
    import importlib.util

    path = ANALYSIS_DETECTION / "m33_d0_gate.py"
    spec = importlib.util.spec_from_file_location("m33_d0_gate_for_pp", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "forbidden",
    [
        r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015\x.npy",
        "data/morocco_eval_v1/images/a.jpg",
        "data/processing/d3/lines.csv",
        "runs/eval/m32_recipe/uninjected/lines.csv",
        "runs/eval/m32_aihub_perturb/summary.json",
        "docs/processing/fourtu_apex_candidates_v1.csv",
        "data/aihub_yolo_v1/test.yaml",
    ],
)
def test_7_d0_refuses_a_forbidden_input_with_exit_2(forbidden, capsys):
    d0 = _d0_module()
    path = Path(forbidden)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    assert d0.main(["--probe-path", str(path)]) == 2
    final = capsys.readouterr().out.strip().splitlines()[-1]
    assert '"status": "input_error"' in final
