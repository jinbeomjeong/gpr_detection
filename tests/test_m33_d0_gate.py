"""M3-3 D0 gate runner (``analysis/detection/m33_d0_gate.py``) without a model.

Covers the input guard, the condition views (A1 padding, A2 top band, A3
surface row), PP on crop predictions, the single-apex match, the gate
values G1-G4 and the selection rule (largest passing set, stop at the first
failure, S1 failure). A stand-in detector replaces YOLO; no D0 result is read.
"""

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from gpr_detection.detect.base import (
    Detection,
    DetectorOutput,
    InputTransform,
    PixelBox,
)
from gpr_detection.io.radargram import GridBox
from gpr_detection.pipeline import PP_SETS, ContentRows, PPAction, PPSetId

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DETECTION = PROJECT_ROOT / "analysis" / "detection"
METRIC_WORDS = re.compile(r"recall|ap50|map50|loss|delta", re.IGNORECASE)
CANVAS = 256
P_A = 4.06


@pytest.fixture(scope="module")
def d0():
    path = ANALYSIS_DETECTION / "m33_d0_gate.py"
    spec = importlib.util.spec_from_file_location("m33_d0_gate_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def match():
    sys.path.insert(0, str(ANALYSIS_DETECTION))
    try:
        import m32_recipe_match
    finally:
        sys.path.remove(str(ANALYSIS_DETECTION))
    return m32_recipe_match


# ---------------------------------------------------------------- guard


def test_guard_allows_the_listed_inputs(d0):
    guard = d0.InputGuard()
    run = PROJECT_ROOT / "runs" / "detect" / f"{d0.RUN_PREFIX}3"
    guard.check(run / "weights" / "last.pt")
    guard.check(run / "eval" / "last_crop" / "eval_images.txt")
    guard.check(PROJECT_ROOT / "data" / "aihub_yolo_v1" / "images" / "a(1).jpg")
    guard.check(PROJECT_ROOT / "data" / "synthetic_gprmax" / "M20.json")
    assert len(guard.opened) == 4


@pytest.mark.parametrize(
    "relative",
    ["data/synthetic_gprmax/M21.npy", "data/aihub_yolo_v1/fold0.yaml", "README.md"],
)
def test_guard_refuses_unlisted_inputs(d0, relative):
    with pytest.raises(d0.common.InputError, match="allow list"):
        d0.InputGuard().check(PROJECT_ROOT / relative)


def test_guard_forbidden_paths_raise_the_forbidden_error(d0):
    guard = d0.InputGuard()
    with pytest.raises(d0.ForbiddenPathError):
        guard.check(Path(r"C:\workspace\GPR_Data\x\y.npy"))
    with pytest.raises(d0.ForbiddenPathError):
        guard.check(PROJECT_ROOT / "runs" / "eval" / "m32_recipe_v2" / "a.json")
    assert guard.opened == []


def test_probe_prints_a_value_free_final_line(d0, capsys):
    path = PROJECT_ROOT / "data" / "splits" / "aihub_yolo_v1.csv"
    assert d0.main(["--probe-path", str(path)]) == 0
    final = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert set(final) == {"status", "out_dir", "n_conditions", "n_images", "elapsed_s"}
    assert final["status"] == "ok"
    assert not METRIC_WORDS.search(json.dumps(final))


# ------------------------------------------------------- condition views


def _sample(d0, pad: int, boxes, classes):
    image = np.full((CANVAS, CANVAS), 120, dtype=np.uint8)
    image[20] = 250  # the surface band row
    return d0.perturb.Sample(
        stem="crop",
        gain="1.0",
        image=image,
        boxes=np.asarray(boxes, dtype=float).reshape(-1, 4),
        classes=np.asarray(classes, dtype=int),
        gt_index=np.arange(len(classes)),
        pad_height_px=pad,
    )


def _views(d0, sample):
    return {
        view.condition: view
        for view in d0.crop_views(sample, d0.condition_images(sample))
    }


@pytest.mark.parametrize(
    ("y1", "y2", "pad"), [(242, 256, 0), (244, 256, 4), (220, 256, 20)]
)
def test_a2_ignores_on_the_moved_uncut_centre(d0, y1, y2, pad):
    """[PP] §4-3 A2 (a): the +8 move is not cut at 256 before the centre test."""
    sample = _sample(d0, pad=pad, boxes=[[10, y1, 60, y2]], classes=[0])
    a2 = _views(d0, sample)["A2"]
    assert len(a2.gt_boxes) == 0
    assert a2.ignore_boxes.tolist() == [[10, y1 + 8, 60, 256]]  # IoU/IoA rows


@pytest.mark.parametrize("pad", [8, 9, 30, 120])
def test_a2_keeps_the_a1_set_moved_by_8_when_the_pad_is_8_or_more(d0, pad):
    """[PP] §4-3 A2 (b): A1 <-> A2 invariance for h_pad >= 8."""
    rng = np.random.default_rng(pad)
    boxes = []
    for _ in range(60):
        y1 = float(rng.uniform(0, 250))
        boxes.append([10, y1, 60, min(256.0, y1 + float(rng.uniform(2, 80)))])
    sample = _sample(d0, pad=pad, boxes=boxes, classes=[0] * len(boxes))
    views = _views(d0, sample)
    a1, a2 = views["A1"], views["A2"]
    assert a2.gt_index.tolist() == a1.gt_index.tolist()
    moved = a1.gt_boxes.copy()
    moved[:, [1, 3]] += 8
    np.testing.assert_array_equal(a2.gt_boxes, moved)


def test_surface_row_is_the_top_quarter_energy_maximum(d0):
    image = np.full((CANVAS, CANVAS), 100, dtype=np.uint8)
    image[30] = 200
    image[45] = 200  # same row energy: the smaller row wins
    image[100] = 255  # outside rows 0-63
    assert d0.surface_band_row(image) == 30


def test_views_set_y_bot_zero_and_ignore_boxes(d0):
    boxes = [[10, 40, 60, 80], [10, 200, 60, 240], [10, 180, 60, 230]]
    sample = _sample(d0, pad=50, boxes=boxes, classes=[0, 1, 2])
    views = {
        view.condition: view
        for view in d0.crop_views(sample, d0.condition_images(sample))
    }
    assert views["A0"].content == ContentRows(256.0, P_A, 0.0)
    assert views["A1"].content.y_bot_px == 206.0
    assert len(views["A1"].gt_boxes) == 2 and len(views["A1"].ignore_boxes) == 1
    assert views["A1"].gt_boxes[:, 3].max() <= 206.0  # cut with the content
    assert views["A2"].content == ContentRows(214.0, P_A, 8.0)
    assert views["A2"].gt_boxes[0].tolist() == [10, 48, 60, 88]
    assert views["A3"].image_key == "A0"
    assert views["A3"].content.zero_px == 20.0


def test_a2_image_is_a1_moved_down_under_a_flat_band(d0):
    sample = _sample(d0, pad=50, boxes=[[0, 0, 1, 1]], classes=[0])
    images = d0.condition_images(sample)
    assert (images["A2"][:8] == d0.PAD_GRAY).all()
    np.testing.assert_array_equal(images["A2"][8:], images["A1"][:-8])
    assert (images["A1"][206:] == d0.PAD_GRAY).all()


# ------------------------------------------------------------ crop PP


def test_pp_on_predictions_removes_and_clips(d0):
    predictions = d0.Predictions(
        boxes=np.array([[0, 190, 40, 250], [0, 150, 40, 230], [0, 50, 40, 90]], float),
        scores=np.array([0.9, 0.8, 0.7]),
        classes=np.array([0, 1, 0]),
    )
    content = ContentRows(206.0, P_A, 0.0)
    crop = d0.pp_on_predictions(predictions, content, PP_SETS[PPSetId.S3])
    assert crop.actions == (PPAction.REMOVED_A, PPAction.CLIPPED_B, PPAction.NONE)
    assert crop.kept.boxes.tolist() == [[0, 150, 40, 206], [0, 50, 40, 90]]
    off = d0.pp_on_predictions(predictions, content, PP_SETS[PPSetId.S0])
    np.testing.assert_array_equal(off.kept.boxes, predictions.boxes)


# ------------------------------------------------------- apex match


def test_single_apex_hit_equals_the_m32_one_to_one_rule(d0, match):
    rng = np.random.default_rng(7)
    for _ in range(300):
        boxes = []
        for _ in range(int(rng.integers(0, 4))):
            i0, k0 = int(rng.integers(0, 80)), int(rng.integers(0, 300))
            grid = GridBox(
                i0, i0 + int(rng.integers(0, 30)), k0, k0 + int(rng.integers(0, 60))
            )
            boxes.append((int(rng.integers(0, 2)), grid))
        apex = (float(rng.uniform(0, 110)), float(rng.uniform(0, 360)))
        period = float(rng.uniform(10, 40))
        ours = d0.apex_hit(boxes, *apex, period)
        scored = [match.ScoredBox(c, 0.5, g) for c, g in boxes]
        (outcome,) = match.match_one_to_one(
            [match.TruthApex("a", *apex)], scored, period * 0.2, period
        )
        assert ours == outcome.hit


# ------------------------------------------------------ gates, selection


def _gate(d0, g1=0.0, g2=0.0, g3=0, g4=1, h_s0=1):
    return d0.GateResult(
        g1_loss_pp=g1, g2_loss_pp=g2, g3_loss=g3, g4_increment=g4, g3_h_s0=h_s0
    )


def test_gate_thresholds(d0):
    assert _gate(d0, g1=0.5, g2=1.0).passed
    assert not _gate(d0, g1=0.5001).passed
    assert not _gate(d0, g2=1.0001).passed
    assert not _gate(d0, g2=None).passed  # no band GT: nothing measured, fail
    assert not _gate(d0, g1=None).passed
    assert not _gate(d0, g3=1).passed
    assert not _gate(d0, g3=-1).passed  # impossible, counted as failure


def test_g3_fails_without_any_s0_hit(d0):
    """[PP] v0.2.3 §4-7 4: G3 is defined only for H(S0) >= 1."""
    assert not _gate(d0, g3=0, h_s0=0).passed
    assert not _gate(d0, h_s0=0).checks["G3"]
    assert _gate(d0, h_s0=1).checks["G3"]


def test_g4_zero_increment_fails(d0):
    assert not _gate(d0, g4=0).passed
    assert _gate(d0, g4=1).passed


def _fold_with_processed(d0, by_set: dict[str, int]):
    output = d0.FoldOutput(write_predictions=[].extend)
    output.synthetic_processed.update(by_set)
    return output


def test_g4_is_the_increment_over_the_previous_set(d0):
    cumulative = {"S1": 3, "S2": 3, "S3": 7, "S4": 9, "S5": 12}
    increments = d0.g4_increments({0: _fold_with_processed(d0, cumulative)})
    assert [increments[key] for key in d0.GATE_SETS] == [3, 0, 4, 2, 3]
    shrinking = dict(cumulative, S3=2)
    with pytest.raises(d0.GateDefectError, match="S3"):
        d0.g4_increments({0: _fold_with_processed(d0, shrinking)})


def _gates(d0, failing: set[str]):
    return {
        key: _gate(d0, g1=1.0 if key.value in failing else 0.0) for key in d0.GATE_SETS
    }


def test_selection_takes_the_largest_passing_set(d0):
    selection = d0.select_pp_set(_gates(d0, set()))
    assert selection.pp_set_id is PPSetId.S5 and selection.stopped_at is None


def test_selection_stops_at_the_first_failure(d0):
    selection = d0.select_pp_set(_gates(d0, {"S3", "S5"}))
    assert selection.pp_set_id is PPSetId.S2
    assert selection.stopped_at is PPSetId.S3
    assert selection.evaluated == (PPSetId.S1, PPSetId.S2, PPSetId.S3)
    later_only = d0.select_pp_set(_gates(d0, {"S5"}))
    assert later_only.pp_set_id is PPSetId.S4


def test_s1_failure_stops_everything(d0):
    selection = d0.select_pp_set(_gates(d0, {"S1"}))
    assert selection.pp_set_id is None and selection.s1_failed
    assert selection.evaluated == (PPSetId.S1,)
    assert selection.stop_reason == "s1_defect"


def test_stop_reasons_tell_a_defect_from_no_effect(d0):
    gates = _gates(d0, set())
    assert d0.select_pp_set(gates).stop_reason is None
    no_effect = dict(gates, S1=_gate(d0, g4=0))
    assert d0.select_pp_set(no_effect).stop_reason == "s1_no_effect"
    # G1-G3 failing wins even when G4 is also 0.
    defect = dict(gates, S1=_gate(d0, g1=1.0, g4=0))
    assert d0.select_pp_set(defect).stop_reason == "s1_defect"
    no_hits = dict(gates, S1=_gate(d0, h_s0=0, g4=0))
    assert d0.select_pp_set(no_hits).stop_reason == "s1_defect"
    later = dict(gates, S3=_gate(d0, g4=0))
    assert d0.select_pp_set(later).stop_reason == "gate_failed"
    for reason, status in [
        (None, "ok"),
        ("gate_failed", "ok"),
        ("s1_defect", "s1_defect"),
        ("s1_no_effect", "s1_no_effect"),
    ]:
        assert d0.run_status(reason) == status
    assert d0.EXIT_CODES["s1_defect"] == d0.EXIT_CODES["s1_no_effect"] == 1
    assert d0.EXIT_CODES["ok"] == 0 and d0.EXIT_CODES["error"] == 3


def _object_row(d0, condition, fold, index, hits):
    """An ``objects.csv`` row of a hyperbola GT with ``hits`` for S0..S5."""
    return [fold, condition, "crop", "1.0", "pipe_hyperbola", index, *hits]


def test_g1_is_the_total_loss_a_gain_never_hides_it(d0):
    """One GT lost and another gained: recall is unchanged, G1 is 50 %p."""
    output = d0.FoldOutput(write_predictions=[].extend)
    for condition in d0.GATE_CONDITIONS:
        output.object_rows += [
            _object_row(d0, condition, 0, 0, [1, 0, 0, 0, 0, 0]),
            _object_row(d0, condition, 0, 1, [0, 1, 1, 1, 1, 1]),
        ]
    cells = d0.hyperbola_loss_cells({0: output}, PPSetId.S1)
    worst, by_condition = d0.worst_fold_mean_pp(cells, [0])
    assert worst == pytest.approx(50.0)
    assert cells[("A0", 0)].gain_pp == pytest.approx(50.0)
    assert by_condition == {name: pytest.approx(50.0) for name in d0.GATE_CONDITIONS}
    # One empty cell of the 3 x folds still fails the gate.
    output.object_rows = [r for r in output.object_rows if r[1] != "A3"]
    cells = d0.hyperbola_loss_cells({0: output}, PPSetId.S1)
    assert d0.worst_fold_mean_pp(cells, [0])[0] is None


def test_gt_covered_lengths_are_per_gt_box():
    from gpr_detection.detect.band_metrics import band_coverage, gt_covered_lengths

    gt = [[0, 10, 100, 20], [100, 10, 200, 20], [0, 50, 100, 60]]
    pred = [[50, 10, 150, 20], [0, 50, 30, 60], [0, 80, 300, 90]]
    covered = gt_covered_lengths(gt, pred)
    np.testing.assert_allclose(covered, [50.0, 50.0, 30.0])
    assert covered.sum() == pytest.approx(band_coverage(gt, pred).covered_gt_length_px)


def test_g2_sums_the_per_gt_decrease(d0):
    """S0 covers band 1, S_j covers band 2 instead: total equal, G2 = 50 %p."""
    gt = np.array([[0, 10, 100, 20], [0, 50, 100, 60]], dtype=float)
    view = d0.CropView(
        "A0",
        "A0",
        ContentRows(256.0, P_A, 0.0),
        gt,
        np.array([1, 1]),
        np.array([0, 1]),
        np.zeros((0, 4)),
    )
    covered = {key: np.array([0.0, 100.0]) for key in PPSetId}
    covered[PPSetId.S0] = np.array([100.0, 0.0])
    output = d0.FoldOutput(write_predictions=[].extend)
    d0.add_band_changes(output, view, covered)
    cells = d0.band_loss_cells({0: output}, PPSetId.S2)
    assert cells[("A0", 0)] == d0.LossCell(lost=100.0, gained=100.0, total=200.0)
    assert cells[("A0", 0)].loss_pp == pytest.approx(50.0)
    assert d0.band_loss_cells({0: output}, PPSetId.S0)[("A0", 0)].loss_pp == 0.0


def test_worst_loss_is_the_worst_gate_condition_fold_mean(d0):
    s0, s1 = PPSetId.S0, PPSetId.S1
    recalls = {
        ("A0", s0): {0: 0.8, 1: 0.6},
        ("A0", s1): {0: 0.79, 1: 0.6},  # 0.5 %p
        ("A1", s0): {0: 0.5, 1: 0.5},
        ("A1", s1): {0: 0.48, 1: 0.5},  # 1.0 %p
        ("A3", s0): {0: 0.7, 1: None},
        ("A3", s1): {0: 0.71, 1: None},  # -1.0 %p
        ("A2", s0): {0: 0.9, 1: 0.9},
        ("A2", s1): {0: 0.0, 1: 0.0},  # report only: never the worst
    }
    complete = dict(recalls)
    complete[("A3", s0)] = {0: 0.7, 1: 0.7}
    complete[("A3", s1)] = {0: 0.71, 1: 0.71}  # -1.0 %p
    worst, by_condition = d0.worst_loss_pp(complete, s1, [0, 1])
    assert worst == pytest.approx(1.0)
    assert by_condition["A0"] == pytest.approx(0.5)
    assert by_condition["A3"] == pytest.approx(-1.0)
    assert "A2" not in by_condition
    # [PP] §4-7 3: one empty cell of the 3 x folds makes the gate value None.
    worst, by_condition = d0.worst_loss_pp(recalls, s1, [0, 1])
    assert worst is None and by_condition["A3"] is None
    assert by_condition["A1"] == pytest.approx(1.0)
    # A fold that never ran counts as an empty cell too.
    assert d0.worst_loss_pp(complete, s1, [0, 1, 2])[0] is None


def test_g2_fails_when_one_fold_has_no_band_truth(d0):
    """[PP] §4-7 3: band GT in fold 0 only -> G2 fails, G1 still has a value."""
    outputs = {}
    for fold, (boxes, classes) in enumerate(
        [
            ([[10, 40, 60, 80], [0, 100, 250, 110]], [0, 1]),
            ([[10, 40, 60, 80]], [0]),
        ]
    ):
        rows: list[list] = []
        output = d0.FoldOutput(write_predictions=rows.extend)
        sample = _sample(d0, pad=50, boxes=boxes, classes=classes)
        detections = [
            (0, 0.9, PixelBox(10, 40, 60, 80)),
            (1, 0.9, PixelBox(0, 100, 250, 110)),
        ]
        d0.run_crops(BoxListDetector(detections), [sample], fold, output)
        outputs[fold] = output
    gates = d0.compute_gates(outputs)
    for set_id in d0.GATE_SETS:
        assert gates[set_id].g1_loss_pp == 0.0
        assert gates[set_id].g2_loss_pp is None
        assert not gates[set_id].checks["G2"]


def test_no_test_output_requires_every_crop_in_the_manifest(d0):
    line = '{{"list": "{name}", "n": 3, "not_in_manifest": {unknown}}}'
    summary = '{"total_hits": 0, "assertion_ok": true}'
    good = "\n".join([line.format(name="a", unknown=0), summary])
    assert d0.check_no_test_output(good, 1)["summary"]["total_hits"] == 0
    bad = "\n".join([line.format(name="a", unknown=2), summary])
    with pytest.raises(d0.common.InputError, match="missing from the manifest"):
        d0.check_no_test_output(bad, 1)
    with pytest.raises(d0.common.InputError, match="lists"):
        d0.check_no_test_output(good, 2)


SYNTHETIC_MODEL = PROJECT_ROOT / "data" / "synthetic_gprmax" / "M04.npy"


@pytest.fixture(scope="module")
def m04_lines(d0):
    if not SYNTHETIC_MODEL.is_file():
        pytest.skip("synthetic model M04 missing")
    return d0.synthetic_lines("M04", d0.InputGuard())


@pytest.mark.data
def test_synthetic_lines_cut_the_record_after_the_apex(d0, m04_lines):
    assert [line.condition for line in m04_lines] == [
        "Y0",
        "Y1_0.5P",
        "Y1_1P",
        "Y1_2P",
        "Y1_3P",
    ]
    y0 = m04_lines[0]
    period = y0.image.period_samples
    assert (y0.period_source, y0.p_over_p_y0) == ("measured", 1.0)
    for line, periods in zip(m04_lines[1:], d0.Y1_CUT_PERIODS, strict=True):
        expected = min(
            y0.n_samples_kept, int(np.floor(y0.apex_sample + periods * period)) + 1
        )
        assert line.n_samples_kept == expected  # 0 .. floor(k + f P), inclusive
        assert line.image.grid_map.n_samples == expected
        assert (line.apex_trace, line.apex_sample) == (y0.apex_trace, y0.apex_sample)
        assert line.content.y_bot_px == pytest.approx(line.image.content_height_px)
        # [PP] §4-3 Y1: the Y0 period is kept; zero and content are rebuilt.
        assert line.period_samples == period
        assert line.period_source == "Y0_fixed"
        assert line.content.period_px == pytest.approx(line.image.period_l1_px)
        assert line.p_over_p_y0 is not None and line.p_over_p_y0 > 0
        # The kept end sample adds at most one sample past k + f P.
        assert line.apex_bottom_periods <= periods + 1.0 / period + 1e-9


@pytest.mark.data
def test_synthetic_row_carries_the_period_columns(d0, m04_lines):
    column = {name: i for i, name in enumerate(d0.SYNTHETIC_HEADER)}
    for line in m04_lines:
        row, _ = d0.synthetic_row(0, line, ())
        assert len(row) == len(d0.SYNTHETIC_HEADER)
        assert row[column["period_source"]] == line.period_source
        assert row[column["period_samples"]] == round(line.period_samples, 4)
        assert row[column["y_bot_px"]] == round(line.content.y_bot_px, 3)
        assert row[column["y_bot_minus_apex_over_p_ell"]] == round(
            line.apex_bottom_periods, 3
        )


@pytest.mark.data
def test_synthetic_row_matches_the_apex_and_counts_pp(d0, m04_lines):
    from gpr_detection.pipeline import LineBox, grid_box_to_image_box

    line = m04_lines[0]
    apex = GridBox(
        round(line.apex_trace) - 5,
        round(line.apex_trace) + 5,
        round(line.apex_sample) - 2,
        round(line.apex_sample) + 40,
    )
    around_apex = grid_box_to_image_box(apex, line.image.grid_map)
    y_bot = line.content.y_bot_px
    in_padding = PixelBox(around_apex.x1, y_bot + 5, around_apex.x2, y_bot + 40)
    merged = (
        LineBox(0, "pipe_hyperbola", 0.9, around_apex, 0),
        LineBox(0, "pipe_hyperbola", 0.8, in_padding, 0),
    )
    row, processed = d0.synthetic_row(0, line, merged)
    column = {name: i for i, name in enumerate(d0.SYNTHETIC_HEADER)}
    assert all(row[column[f"hit_{key}"]] == 1 for key in PPSetId)
    assert row[column["n_processed_S1"]] == 1
    assert processed[PPSetId.S1][PPAction.REMOVED_A.value] == 1


def _rule_fixture_boxes(line):
    """Merged boxes hitting PP-a, -b, -d and -c on one synthetic line."""
    from gpr_detection.pipeline import LineBox, grid_box_to_image_box

    apex = GridBox(
        round(line.apex_trace) - 5,
        round(line.apex_trace) + 5,
        round(line.apex_sample) - 2,
        min(round(line.apex_sample) + 20, line.n_samples_kept - 1),
    )
    around = grid_box_to_image_box(apex, line.image.grid_map)
    y_bot, period = line.content.y_bot_px, line.content.period_px
    zero = line.content.zero_px or 0.0

    def box(class_id, score, y1, y2, x_shift=0.0):
        pixel = PixelBox(around.x1 + x_shift, y1, around.x2 + x_shift, y2)
        return LineBox(class_id, d0_names[class_id], score, pixel, 0)

    d0_names = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")
    return (
        LineBox(0, "pipe_hyperbola", 0.9, around, 0, n_tiles_seen=2),
        box(0, 0.8, y_bot + 2, y_bot + 30, 3.0),  # PP-a
        box(2, 0.7, y_bot - 3 * period, y_bot + 10, 6.0),  # PP-b clip
        box(1, 0.6, max(zero - 1.0, 0.0), zero + 0.2 * period, 9.0),  # PP-d
        box(1, 0.3, y_bot - 10, y_bot - 1, 12.0),  # PP-c (0.35 and 0.50)
    )


@pytest.mark.data
def test_g4_by_rule_adds_up_to_the_cumulative_count(d0, m04_lines):
    """[PP] §4-5 G4: sum(by_rule) over A1, A3 and Y0 == cumulative, per set."""
    output, _ = _fold_with(
        d0,
        [
            (0, 0.9, PixelBox(10, 40, 60, 80)),
            (0, 0.8, PixelBox(100, 200, 140, 250)),
            (2, 0.6, PixelBox(100, 2, 140, 12)),
        ],
    )
    for line in m04_lines:
        d0.add_synthetic_line(output, 0, line, _rule_fixture_boxes(line))
    folds = {0: output}
    d0.check_g4_rules(folds)
    for set_id in d0.GATE_SETS:
        counts = d0.rule_counts(folds, set_id, d0.G4_CONDITIONS)
        assert set(counts["by_condition"]) == {"A1", "A3", "Y0"}
        assert sum(counts["by_rule"].values()) == d0._g4_processed(folds, set_id)
    y0 = d0.rule_counts(folds, PPSetId.S5, d0.G4_CONDITIONS)["by_condition"]["Y0"]
    assert y0.get("removed_a") and y0.get("clipped_b")
    outside = d0.rule_counts(folds, PPSetId.S5, d0.OUTSIDE_GATE_CONDITIONS)
    assert set(outside["by_condition"]) == {
        "A0",
        "A2",
        "Y1_0.5P",
        "Y1_1P",
        "Y1_2P",
        "Y1_3P",
    }
    output.synthetic_processed["S3"] += 1  # the totals no longer agree
    with pytest.raises(d0.GateDefectError, match="by_rule"):
        d0.check_g4_rules(folds)


def _read_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.data
def test_synthetic_boxes_csv_reproduces_the_apex_table(d0, m04_lines, tmp_path):
    """[PP] §4-5: hits and processed counts from synthetic_boxes.csv alone."""
    output = d0.FoldOutput(write_predictions=[].extend)
    for fold in (0, 1):
        for line in m04_lines:
            d0.add_synthetic_line(output, fold, line, _rule_fixture_boxes(line))
    boxes_path = tmp_path / "synthetic_boxes.csv"
    apex_path = tmp_path / "synthetic_apex.csv"
    d0.common.write_csv(
        boxes_path, d0.SYNTHETIC_BOXES_HEADER, output.synthetic_box_rows
    )
    d0.common.write_csv(apex_path, d0.SYNTHETIC_HEADER, output.synthetic_rows)
    boxes = _read_rows(boxes_path)
    apex_rows = _read_rows(apex_path)
    assert len(apex_rows) == 2 * len(m04_lines)
    for apex in apex_rows:
        key = (apex["fold"], apex["condition"], apex["model"])
        mine = [b for b in boxes if (b["fold"], b["condition"], b["model"]) == key]
        assert len(mine) == int(apex["n_merged_boxes"])
        for set_id in PPSetId:
            kept = [
                b
                for b in mine
                if set_id is PPSetId.S0
                or not PPAction(b[f"pp_action_{set_id}"]).is_removed
            ]
            assert int(apex[f"hit_{set_id}"]) == _hit_from_rows(d0, kept, apex), (
                key,
                set_id,
            )
        for set_id in d0.GATE_SETS:
            actions = [PPAction(b[f"pp_action_{set_id}"]) for b in mine]
            for action in d0.PROCESSED:
                assert int(apex[f"n_{action}_{set_id}"]) == actions.count(action)
            assert int(apex[f"n_processed_{set_id}"]) == sum(
                a.is_processed for a in actions
            )
            for row in mine:
                after = row[f"y2_after_{set_id}"]
                removed = PPAction(row[f"pp_action_{set_id}"]).is_removed
                assert (after == "") == removed


def _hit_from_rows(d0, kept, apex) -> int:
    for row in kept:
        if row["class"] != "pipe_hyperbola" or row["i0"] == "":
            continue
        grid = GridBox(int(row["i0"]), int(row["i1"]), int(row["k0"]), int(row["k1"]))
        if d0._box_contains_apex(
            grid,
            float(apex["apex_trace"]),
            float(apex["apex_sample"]),
            float(apex["period_samples"]),
        ):
            return 1
    return 0


class BoxListDetector:
    """Stand-in backend: the same detections (crop px) for every image."""

    class_names = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")

    def __init__(self, detections: list[tuple[int, float, PixelBox]]) -> None:
        self._detections = detections

    def predict_batch(self, images):
        transform = InputTransform(1.0, 1.0, 0.0, 0.0, CANVAS, CANVAS)
        output = DetectorOutput(
            tuple(
                Detection(c, self.class_names[c], s, b) for c, s, b in self._detections
            ),
            transform,
        )
        return [output for _ in images]


def _fold_with(d0, detections, pad=50):
    rows: list[list] = []
    output = d0.FoldOutput(write_predictions=rows.extend)
    sample = _sample(d0, pad=pad, boxes=[[10, 40, 60, 80]], classes=[0])
    d0.run_crops(BoxListDetector(detections), [sample], 0, output)
    return output, rows


def test_crop_run_scores_every_condition_and_set(d0):
    detections = [
        (0, 0.9, PixelBox(10, 40, 60, 80)),  # the GT: hit everywhere
        (0, 0.8, PixelBox(100, 200, 140, 250)),  # in the A1 padding: PP-a
        (2, 0.6, PixelBox(100, 2, 140, 12)),  # above y_z + 0.5 P_A in A3: PP-d
    ]
    output, rows = _fold_with(d0, detections)
    assert len(rows) == 3 * 4
    assert {key for key in output.scores} == {
        (c, s) for c in d0.CROP_CONDITIONS for s in PPSetId
    }
    for key, scores in output.scores.items():
        assert scores.class_results()["pipe_hyperbola"]["recall_conf025"] == 1.0, key
    a1 = output.scores[("A1", PPSetId.S1)].processed_confident
    assert a1[PPAction.REMOVED_A.value] == 1
    a3 = output.scores[("A3", PPSetId.S2)].processed_confident
    assert a3[PPAction.REMOVED_D.value] == 1
    gates = d0.compute_gates({0: output})
    assert gates[PPSetId.S1].g1_loss_pp == 0.0
    assert gates[PPSetId.S1].g4_increment >= 1
    assert gates[PPSetId.S2].g4_increment >= 1  # PP-d adds the A3 removal
    assert output.surface_rows == Counter({20: 1})


def test_removing_the_only_hit_shows_as_a_g1_loss(d0):
    # A3 zero = surface row 20: a hit centred above 20 + 0.5 P_A is PP-d.
    rows: list[list] = []
    output = d0.FoldOutput(write_predictions=rows.extend)
    sample = _sample(d0, pad=40, boxes=[[10, 5, 60, 30]], classes=[0])
    detector = BoxListDetector([(0, 0.9, PixelBox(10, 5, 60, 30))])
    d0.run_crops(detector, [sample], 0, output)
    gates = d0.compute_gates({0: output})
    assert gates[PPSetId.S1].g1_loss_pp == 0.0
    assert gates[PPSetId.S1].g4_increment == 0  # nothing processed: S1 fails G4
    assert gates[PPSetId.S2].g1_loss_pp == pytest.approx(100.0)
    selection = d0.select_pp_set(gates)
    assert selection.s1_failed and selection.pp_set_id is None
    # No band GT also leaves G2 undefined, so S1 is a defect, not "no effect".
    assert selection.stop_reason == "s1_defect"
