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
        # Full precision (verif audit 2026-11-15): no rounding of row values.
        assert row[column["apex_sample"]] == line.apex_sample
        assert row[column["period_samples"]] == line.period_samples
        assert row[column["y_bot_px"]] == line.content.y_bot_px
        assert row[column["y_bot_minus_apex_over_p_ell"]] == line.apex_bottom_periods


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


# ---------------------------------------- full-precision csv (audit 1)

JUST_ABOVE_CONF = float(np.nextafter(np.float32(0.25), np.float32(1.0)))
"""The float32 score right above 0.25; four digits wrote it as 0.25."""


def test_prediction_csv_keeps_scores_and_coordinates_exact(d0, tmp_path):
    box = PixelBox(10.123456789, 40.000001, 60.5, 80.333333333)
    _, rows = _fold_with(d0, [(0, JUST_ABOVE_CONF, box)])
    path = tmp_path / "predictions.csv"
    d0.common.write_csv(path, d0.PREDICTIONS_HEADER, rows)
    written = _read_rows(path)
    assert [row["condition"] for row in written] == list(d0.CROP_CONDITIONS)
    for row in written:
        assert float(row["score"]) == JUST_ABOVE_CONF
        assert float(row["score"]) > d0.CONF_RECALL  # recountable from the csv
        assert [float(row[key]) for key in ("x1", "y1", "x2", "y2")] == list(box.xyxy)


@pytest.mark.data
def test_synthetic_box_csv_keeps_scores_and_coordinates_exact(d0, m04_lines, tmp_path):
    from dataclasses import replace

    from gpr_detection.pipeline import apply_pp

    line = m04_lines[0]
    merged = tuple(
        replace(box, score=JUST_ABOVE_CONF) for box in _rule_fixture_boxes(line)
    )
    path = tmp_path / "synthetic_boxes.csv"
    rows = d0.synthetic_box_rows(0, line, merged)
    d0.common.write_csv(path, d0.SYNTHETIC_BOXES_HEADER, rows)
    written = _read_rows(path)
    s1 = apply_pp(merged, line.content, PP_SETS[PPSetId.S1])
    for box, row, outcome in zip(merged, written, s1, strict=True):
        assert float(row["score"]) == box.score == JUST_ABOVE_CONF
        coordinates = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
        assert coordinates == list(box.box.xyxy)
        if outcome.output is not None:
            assert float(row["y2_after_S1"]) == outcome.output.box.y2


# ------------------------------------- judgment scope (audit 2, 3, 4, 5)


def test_only_folds_0_to_4_with_every_crop_and_model_are_a_full_run(d0):
    models = d0.SYNTHETIC_MODELS
    assert d0.run_scope_blockers([4, 3, 2, 1, 0], models, None, smoke=False) == []
    assert d0.run_scope_blockers([0, 1, 2, 3], models, None, smoke=False) == [
        "partial_folds"
    ]
    assert d0.run_scope_blockers(range(5), models, 100_000, smoke=False) == [
        "crop_limit"
    ]
    assert d0.run_scope_blockers(range(5), models[:19], None, smoke=False) == [
        "partial_models"
    ]
    assert d0.run_scope_blockers([0], models[:2], 8, smoke=True) == [
        "smoke",
        "partial_folds",
        "crop_limit",
        "partial_models",
    ]


@pytest.mark.parametrize("text", ["0,0,1,2,3", "0,1,2,3,5", "0;1", ""])
def test_fold_argument_refuses_repeated_and_unknown_folds(d0, text):
    with pytest.raises(d0.common.InputError):
        d0.parse_folds(text)


def test_fold_argument_keeps_the_given_order(d0):
    assert d0.parse_folds("4,0,1,2,3") == (4, 0, 1, 2, 3)


def test_final_status_is_never_ok_for_a_partial_or_changed_run(d0):
    for reason in (None, "gate_failed", "s1_defect", "s1_no_effect"):
        assert d0.final_status(reason, ["partial_folds"], []) == "not_judgeable"
        assert d0.final_status(reason, [], ["head"]) == "input_error"
        assert d0.final_status(reason, ["smoke"], ["code_blobs"]) == "input_error"
        assert d0.final_status(reason, [], []) == d0.run_status(reason)
    assert d0.EXIT_CODES["not_judgeable"] == 1
    assert d0.EXIT_CODES["input_error"] == 2


C0A_SRC_TREE_ID = "164f09909ad51c1a8d26545cadb0417b1da3acc6"
C0A_PIPELINE_BLOB = "dd5e7bcba4397efd732e7bf3ca1c5c92fb2404a1"
"""``git rev-parse 53c4e16:src`` and ``53c4e16:src/gpr_detection/pipeline.py``."""


def _tree(**changes):
    """A ``tree_state`` snapshot whose PP code is the C0a one."""
    state = {
        "head": "a" * 40,
        "dirty": False,
        "problems": [],
        "code_blobs": {
            "analysis/detection/m33_d0_gate.py": "1" * 40,
            "src/gpr_detection/pipeline.py": C0A_PIPELINE_BLOB,
        },
        "src_tree_id": C0A_SRC_TREE_ID,
    }
    return state | changes


def test_tree_changes_name_every_field_that_moved(d0):
    start = _tree()
    assert d0.tree_changes(start, _tree()) == []
    assert d0.tree_changes(start, _tree(head="c" * 40)) == ["head"]
    dirty = _tree(dirty=True, problems=["modified:  M docs/a.md"])
    assert d0.tree_changes(start, dirty) == ["dirty", "problems"]
    edited = _tree(code_blobs={"analysis/detection/m33_d0_gate.py": "2" * 40})
    assert d0.tree_changes(start, edited) == ["code_blobs"]
    unknown = _tree(head=None)
    assert d0.tree_changes(unknown, unknown) == ["head"]  # never "unchanged"


def test_judgment_blocks_a_dirty_start_and_a_changed_tree(d0):
    folds = list(d0.EXPECTED_FOLDS)
    clean = d0.judgment_record(folds, [], _tree(), _tree())
    assert clean["judgeable"] and clean["tree_stable"] and clean["blockers"] == []
    dirty_tree = _tree(dirty=True, problems=["modified:  M docs/a.md"])
    dirty = d0.judgment_record(folds, [], dirty_tree, dirty_tree)
    assert dirty["blockers"] == ["dirty_tree"] and dirty["tree_stable"]
    assert not dirty["judgeable"]
    moved = d0.judgment_record(folds, [], _tree(), _tree(head="c" * 40))
    assert moved["tree_changes"] == ["head"] and not moved["judgeable"]
    assert moved["expected_folds"] == folds


def test_tree_state_reads_head_blobs_and_problems(d0):
    state = d0.tree_state()
    assert tuple(state) == d0.TREE_KEYS
    assert re.fullmatch(r"[0-9a-f]{40}", state["head"])
    assert "analysis/detection/m33_d0_gate.py" in state["code_blobs"]
    assert state["dirty"] == bool(state["problems"])


def _patch_run(d0, monkeypatch, trees, documents):
    """Replace inputs, inference and writing; ``trees`` are start and end."""
    states = iter(trees)
    monkeypatch.setattr(d0, "tree_state", lambda: next(states))

    def load_inputs(args, guard):
        n_models = args.synthetic_limit or len(d0.SYNTHETIC_MODELS)
        return d0.RunInputs(
            manifest=[],
            pad_heights=np.zeros(1),
            models=d0.SYNTHETIC_MODELS[:n_models],
            folds=d0.parse_folds(args.folds),
        )

    def run_fold(args, inputs, fold, guard, lines, started):
        output, _ = _fold_with(d0, [(0, 0.9, PixelBox(10, 40, 60, 80))])
        return output

    monkeypatch.setattr(d0, "_load_inputs", load_inputs)
    monkeypatch.setattr(d0, "_new_run_identity", lambda *args: {})
    monkeypatch.setattr(d0, "_check_fixed_inputs", lambda *args: None)
    monkeypatch.setattr(d0, "synthetic_lines", lambda model, guard: [])

    def write_outputs(out_root, folds, document):
        documents.append(document)

    monkeypatch.setattr(d0, "_run_fold", run_fold)
    monkeypatch.setattr(d0, "_write_outputs", write_outputs)


def _run_main(d0, capsys, tmp_path, *extra):
    argv = ["--out-root", str(tmp_path / "out"), "--device", "cpu", *extra]
    exit_code = d0.main(argv)
    final = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    return exit_code, final


def test_a_full_run_on_a_stable_tree_is_judged(d0, monkeypatch, capsys, tmp_path):
    documents: list[dict] = []
    _patch_run(d0, monkeypatch, [_tree(), _tree()], documents)
    exit_code, final = _run_main(d0, capsys, tmp_path)
    (document,) = documents
    assert document["judgment"]["judgeable"] and document["settings"]["full_run"]
    assert document["judgment"]["folds_run"] == list(d0.EXPECTED_FOLDS)
    assert document["run_identity"]["sealed_pp_assertion"]["result"] == "pass"
    assert document["selection"]["judgeable"]
    expected = d0.run_status(document["selection"]["stop_reason"])
    assert final["status"] == document["status"] == expected
    assert exit_code == d0.EXIT_CODES[expected]


@pytest.mark.parametrize(
    ("extra", "blockers"),
    [
        (["--folds", "0,1"], ["partial_folds"]),
        (["--folds", "0,1,2,3,4", "--limit", "5"], ["crop_limit"]),
        (
            ["--smoke"],
            ["smoke", "partial_folds", "crop_limit", "partial_models"],
        ),
    ],
)
def test_a_partial_run_is_not_judgeable_and_exits_1(
    d0, monkeypatch, capsys, tmp_path, extra, blockers
):
    documents: list[dict] = []
    _patch_run(d0, monkeypatch, [_tree(), _tree()], documents)
    exit_code, final = _run_main(d0, capsys, tmp_path, *extra)
    (document,) = documents
    assert final["status"] == document["status"] == "not_judgeable"
    assert exit_code == 1
    assert document["judgment"]["blockers"] == blockers
    assert not document["judgment"]["judgeable"]
    assert not document["selection"]["judgeable"]
    assert not document["settings"]["full_run"]


def test_a_tree_change_during_the_run_is_an_input_error(
    d0, monkeypatch, capsys, tmp_path
):
    documents: list[dict] = []
    _patch_run(d0, monkeypatch, [_tree(), _tree(head="c" * 40)], documents)
    exit_code, final = _run_main(d0, capsys, tmp_path)
    (document,) = documents
    assert final["status"] == document["status"] == "input_error"
    assert exit_code == 2
    assert document["judgment"]["tree_changes"] == ["head"]
    assert document["run_identity"]["tree_end"]["head"] == "c" * 40
    assert not METRIC_WORDS.search(json.dumps(final))


def test_repeated_folds_are_refused_before_any_fold_runs(
    d0, monkeypatch, capsys, tmp_path
):
    documents: list[dict] = []
    _patch_run(d0, monkeypatch, [_tree(), _tree()], documents)
    exit_code, final = _run_main(d0, capsys, tmp_path, "--folds", "0,0,1,2,3")
    assert (exit_code, final["status"]) == (2, "input_error")
    assert documents == []


def test_run_identity_has_lf_shas_and_no_g1_field(d0, monkeypatch, tmp_path):
    """Audit 4 and 5: LF sha beside the raw sha; no empty M3-1 G1 field."""
    crlf = tmp_path / "table.csv"
    crlf.write_bytes(b"a,b\r\n1,2\r\n")
    lf = tmp_path / "table_lf.csv"
    lf.write_bytes(b"a,b\n1,2\n")
    monkeypatch.setattr(d0, "TEXT_INPUTS", {"pad_heights": crlf})
    monkeypatch.setattr(d0, "synthetic_sha256", lambda models, guard: {})
    monkeypatch.setattr(d0.common, "gpu_name", lambda device: "cpu")
    args = d0._parse_args(["--device", "cpu"])
    inputs = d0.RunInputs([], np.zeros(1), (), d0.EXPECTED_FOLDS)
    identity = d0._new_run_identity(args, inputs, d0.InputGuard(), _tree())
    assert "g1_inputs_sha256" not in identity
    raw = identity["inputs_sha256"]["pad_heights"]
    assert raw == d0.common.sha256_file(crlf) != d0.common.sha256_file(lf)
    assert identity["inputs_sha256_lf"]["pad_heights"] == d0.common.sha256_file(lf)
    assert identity["code_blobs"] == _tree()["code_blobs"]


# --------------------------------------- sealed PP for D0' ([PP] v0.2.4 A-1)

C0B_PP_SET_PREFIXES = {
    "S0": "fc9f0c8eab77",
    "S1": "78de0a20390e",
    "S2": "184c07e44c0c",
    "S3": "eb2796ddfaa9",
    "S4": "af29b47126e7",
    "S5": "b87bbc145e78",
}
"""Prefixes of the C0b ``d0_gate.json`` ``pp_sets`` hashes (0f11c34)."""


def test_sealed_constants_are_the_c0b_and_c0a_values(d0):
    for key, prefix in C0B_PP_SET_PREFIXES.items():
        assert d0.C0B_PP_SET_SHA256[PPSetId(key)].startswith(prefix)
    assert d0.C0B_SELECTED_SET is PPSetId.S3
    assert d0.C0B_PP_SET_SHA256[PPSetId.S3] == (
        "eb2796ddfaa9611e5030342c4ed66b00c91ff2ed6a6698ee5979b9afe080602f"
    )
    assert d0.C0B_PP_SETS_SHA256.startswith("1f2b0cf2")
    assert (d0.C0A_SRC_TREE_ID, d0.C0A_PIPELINE_BLOB) == (
        C0A_SRC_TREE_ID,
        C0A_PIPELINE_BLOB,
    )


def test_sealed_constants_match_the_c0a_commit(d0):
    import subprocess

    def rev_parse(spec: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", spec],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            pytest.skip("commit 53c4e16 not in this clone")
        return completed.stdout.strip()

    assert rev_parse("53c4e16:src") == d0.C0A_SRC_TREE_ID
    assert rev_parse(f"53c4e16:{d0.PIPELINE_PATH}") == d0.C0A_PIPELINE_BLOB


def test_sealed_pp_passes_on_the_c0a_tree(d0):
    record = d0.assert_sealed_pp(_tree())
    assert record["result"] == "pass" and all(record["checks"].values())
    assert set(record["checks"]) == {
        "pp_set_sha256",
        "pp_sets_sha256",
        "src_tree_id",
        "pipeline_blob",
        "selected_set_sha256",
    }
    assert record["expected"]["selected_set"] == "S3"
    assert record["measured"]["pp_set_sha256"]["S3"] == PP_SETS[PPSetId.S3].sha256()


@pytest.mark.parametrize(
    ("tree", "failed"),
    [
        (_tree(src_tree_id="0" * 40), "src_tree_id"),
        (
            _tree(code_blobs={"src/gpr_detection/pipeline.py": "0" * 40}),
            "pipeline_blob",
        ),
        (_tree(code_blobs={}), "pipeline_blob"),
    ],
)
def test_sealed_pp_refuses_other_pp_code(d0, tree, failed):
    with pytest.raises(d0.common.InputError, match=failed):
        d0.assert_sealed_pp(tree)


def test_sealed_pp_refuses_another_pp_config(d0, monkeypatch):
    changed = dict(d0.PP_SETS)
    changed[PPSetId.S3] = d0.PP_SETS[PPSetId.S2]
    monkeypatch.setattr(d0, "PP_SETS", changed)
    with pytest.raises(d0.common.InputError) as raised:
        d0.assert_sealed_pp(_tree())
    assert "pp_set_sha256" in str(raised.value)
    assert "selected_set_sha256" in str(raised.value)


def test_other_pp_code_stops_the_run_before_any_fold(d0, monkeypatch, capsys, tmp_path):
    documents: list[dict] = []
    other_src = _tree(src_tree_id="0" * 40)
    _patch_run(d0, monkeypatch, [other_src, other_src], documents)
    exit_code, final = _run_main(d0, capsys, tmp_path)
    assert (exit_code, final["status"]) == (2, "input_error")
    assert documents == []


IS_WINDOWS = sys.platform == "win32"


def test_the_c0b_folder_is_the_default_and_matches_any_spelling(d0, monkeypatch):
    c0b = PROJECT_ROOT / "runs" / "eval" / "m33_d0"
    assert d0.C0B_OUT_ROOT == d0.DEFAULT_OUT_ROOT == c0b
    assert d0.is_c0b_out_root(d0._parse_args([]).out_root)  # --out-root left out
    monkeypatch.chdir(PROJECT_ROOT)
    assert d0.is_c0b_out_root(Path("runs/eval/m33_d0"))
    assert d0.is_c0b_out_root(Path("runs/eval/m33_d0r/../m33_d0"))
    assert d0.is_c0b_out_root(c0b)
    if IS_WINDOWS:
        assert d0.is_c0b_out_root(Path("RUNS/Eval/M33_D0"))
        assert d0.is_c0b_out_root(Path(str(c0b).upper()))
    assert not d0.is_c0b_out_root(Path("runs/eval/m33_d0r"))
    assert not d0.is_c0b_out_root(Path("runs/eval/m33_d0_smoke"))


def _c0b_spellings() -> list[str]:
    spellings = ["absolute", "relative", "dotdot"]
    return [*spellings, "upper_absolute", "upper_relative"] if IS_WINDOWS else spellings


def _spell(sealed: Path, base: Path, spelling: str) -> str:
    """``sealed`` written as ``spelling``; relative ones are relative to ``base``."""
    relative = sealed.relative_to(base).as_posix()
    return {
        "absolute": str(sealed),
        "relative": relative,
        "dotdot": f"{relative}r/../{sealed.name}",
        "upper_absolute": str(sealed).upper(),
        "upper_relative": relative.upper(),
    }[spelling]


@pytest.mark.parametrize("spelling", _c0b_spellings())
def test_overwrite_never_deletes_the_c0b_folder(
    d0, monkeypatch, capsys, tmp_path, spelling
):
    """User decision 2026-11-15: --overwrite does not apply to the C0b folder.

    A stand-in C0b folder under ``tmp_path`` replaces the real one, and
    ``rmtree`` fails loudly, so the real results are never at risk.
    """
    sealed = tmp_path / "runs" / "eval" / "m33_d0"
    sealed.mkdir(parents=True)
    (sealed / "d0_gate.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(d0, "C0B_OUT_ROOT", sealed)
    monkeypatch.setattr(d0.shutil, "rmtree", lambda path: pytest.fail(f"rmtree {path}"))
    monkeypatch.chdir(tmp_path)
    documents: list[dict] = []
    _patch_run(d0, monkeypatch, [_tree(), _tree()], documents)
    out_root = _spell(sealed, tmp_path, spelling)
    exit_code = d0.main(["--out-root", out_root, "--overwrite", "--device", "cpu"])
    final = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (exit_code, final["status"]) == (2, "input_error")
    assert documents == []
    assert (sealed / "d0_gate.json").read_text(encoding="utf-8") == "{}"


def test_overwrite_still_replaces_another_folder(d0, monkeypatch, capsys, tmp_path):
    existing = tmp_path / "out"
    existing.mkdir()
    (existing / "old.csv").write_text("x\n", encoding="utf-8")
    documents: list[dict] = []
    _patch_run(d0, monkeypatch, [_tree(), _tree()], documents)
    exit_code, final = _run_main(d0, capsys, tmp_path, "--overwrite")
    assert len(documents) == 1
    assert not (existing / "old.csv").exists()
    assert final["status"] == documents[0]["status"] != "input_error"
    assert exit_code == d0.EXIT_CODES[final["status"]]


def test_an_existing_output_folder_needs_overwrite(d0, monkeypatch, capsys, tmp_path):
    """[PP] A-1: the C0b folder can only be replaced on purpose."""
    existing = tmp_path / "out"
    existing.mkdir()
    (existing / "d0_gate.json").write_text("{}", encoding="utf-8")
    documents: list[dict] = []
    _patch_run(d0, monkeypatch, [_tree(), _tree()], documents)
    exit_code, final = _run_main(d0, capsys, tmp_path)
    assert (exit_code, final["status"]) == (2, "input_error")
    assert documents == []
    assert (existing / "d0_gate.json").read_text(encoding="utf-8") == "{}"
