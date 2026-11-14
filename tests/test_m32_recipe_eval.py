"""M3-2 S5 recipe runner (``analysis/detection/m32_recipe_eval.py``) and loader.

A stand-in detector replaces YOLO (one ``pipe_hyperbola`` box over the whole
tile and one box in the padding); the line and the pipe response are the
synthetic ones of ``tests/test_inject.py``. No model runs and no stage-C
result is read. Covers batching, the P5a rows (M4, conf counts, M5, M6), the
D3 identity gate, the P5b match rows and conf mask, the sample rule, the
stdout contract, sha refusals (exit 2) and resume.
"""

import importlib.util
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "detection"))
import m32_pipe_responses as responses_io
import m32_recipe_eval as runner
import m32_recipe_inputs as inputs
import m32_recipe_match as match

from gpr_detection.detect.base import (
    Detection,
    DetectorOutput,
    InputTransform,
    PixelBox,
)
from gpr_detection.eval import inject
from gpr_detection.io.classes import DetectionClass

CLASS_NAMES = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")
HYPERBOLA_SCORE = 0.6
METRIC_WORDS = re.compile(r"recall|hit|m4|delta|ap50", re.IGNORECASE)
FINAL_KEYS = {
    "status",
    "stage",
    "out_dir",
    "n_lines",
    "n_tiles",
    "n_injections",
    "elapsed_s",
}


@pytest.fixture(scope="module")
def helpers() -> ModuleType:
    """``tests/test_inject.py`` (synthetic line and response builders)."""
    path = Path(__file__).resolve().parent / "test_inject.py"
    spec = importlib.util.spec_from_file_location("inject_test_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDetector:
    """A whole-tile ``pipe_hyperbola`` box and a ``pipe_band`` box low in the tile."""

    class_names = CLASS_NAMES

    def __init__(self) -> None:
        self.n_calls = 0
        self.n_batch_calls = 0

    def predict(self, image: np.ndarray) -> DetectorOutput:
        self.n_calls += 1
        size = image.shape[0]
        detections = (
            Detection(0, "pipe_hyperbola", HYPERBOLA_SCORE, PixelBox(0, 0, size, size)),
            Detection(1, "pipe_band", 0.3, PixelBox(10, size - 40, 60, size - 10)),
        )
        return DetectorOutput(detections, InputTransform(1, 1, 0, 0, size, size))


class FakeBatchDetector(FakeDetector):
    def predict_batch(self, images):
        self.n_batch_calls += 1
        return [self.predict(image) for image in images]


@pytest.fixture(scope="module")
def line(helpers) -> runner.LineData:
    radargram = helpers._radargram(helpers._background_data())
    ref = inputs.LineRef(inputs.DATASET_4TU, helpers.LINE_ID, inputs.ROLE_K)
    branch = runner.prepare_fitting_branch(radargram)
    return runner.LineData(
        ref, radargram, runner.find_time_zero(radargram, branch.filtered)
    )


def _settings(tmp_path: Path, **changes) -> runner.RunSettings:
    settings = runner.parse_args(
        [
            "--stage",
            "uninjected",
            "--recipes",
            "r0",
            "--folds",
            "0",
            "--device",
            "cpu",
            "--allow-dirty",
            "--out-root",
            str(tmp_path),
        ]
    )
    return replace(settings, **changes)


# --------------------------------------------------------------- batching


def test_predict_tiles_uses_batches_when_offered() -> None:
    tiles = [np.zeros((256, 256), np.uint8)] * 5
    batched, single = FakeBatchDetector(), FakeDetector()

    out_batched = runner.predict_tiles(batched, tiles, batch_size=2)
    out_single = runner.predict_tiles(single, tiles, batch_size=2)

    assert batched.n_batch_calls == 3 and len(out_batched) == 5
    assert single.n_calls == 5 and out_single == out_batched
    assert runner.predict_tiles(FakeBatchDetector(), tiles, batch_size=1) == out_single


# ------------------------------------------------------------------- P5a


def test_recipes_keep_the_line_geometry(line) -> None:
    images = {
        recipe: runner.make_detection_image(
            line.radargram, inputs.recipe_config(recipe)
        )
        for recipe in inputs.STAGE_C_RECIPES
    }
    runner.check_recipe_invariance(images)  # S6 (f): raises on a change
    content = runner.content_range(images["r0"], line.zero)
    assert content.y_bot_px == images["r0"].content_height_px


def test_uninjected_rows_classify_m4_count_conf_and_pair_mirrors(
    line, tmp_path
) -> None:
    settings = _settings(tmp_path)
    apex = inputs.KnownApex("G1", line.ref.line_id, 100, 200.0)

    part = runner.evaluate_uninjected_line(line, {0: FakeDetector()}, settings, [apex])

    originals = [row for row in part["boxes"] if row["orientation"] == runner.ORIGINAL]
    mirrored = [row for row in part["boxes"] if row["orientation"] == runner.FLIPPED]
    band = next(row for row in originals if row["class_id"] == DetectionClass.PIPE_BAND)
    assert band["m4"] == "M4a"  # centre in the padding of the content range
    assert all(row["m4"] == "" for row in mirrored)
    (counts,) = part["conf_counts"]
    assert counts["c0.25"] == 1 and counts["c0.59"] == 1 and counts["c0.60"] == 0
    assert part["apex_hits"] == [
        {
            "recipe": "r0",
            "fold": 0,
            "blind_id": "G1",
            "line_id": line.ref.line_id,
            "trace": 100,
            "sample": 200.0,
            "hit": True,
        }
    ]
    assert all(row["consistent"] for row in part["flip_pairs"])
    assert part["line"][0]["zero_ok"] is True


def test_identity_gate_needs_the_d3_counts() -> None:
    rows = []
    for apex in range(41):
        for fold in range(5):
            hit = (
                (fold == 0 and apex < 15)
                or (fold == 1 and apex < 30)
                or (fold == 2 and apex < 30)
                or (fold == 3 and apex < 15)
            )
            rows.append(
                {"recipe": "r0", "fold": fold, "blind_id": f"G{apex}", "hit": hit}
            )
    gate = runner.identity_gate(rows)
    assert gate["observed"] == {
        "sum_hits": 90,
        "any_fold": 30,
        "fold0": 15,
        "n_apexes": 41,
    }
    assert gate["pass"]
    rows[0]["hit"] = False
    assert not runner.identity_gate(rows)["pass"]


def _positions(n: int, n_lines: int, n_in: int, n_bottom: int) -> list[dict]:
    return [
        {
            "line_id": f"L{i % n_lines}",
            "dataset": "4tu",
            "model_group": "A1",
            "in_distribution": i < n_in,
            "bottom_stratum": i < n_bottom,
            "period_ringing": i == 0,
        }
        for i in range(n)
    ]


def test_sample_report_decides_the_path_and_the_bottom_rule() -> None:
    reports = [
        {
            "patch_failures": {"M1": "period_check_failed"},
            "n_dropped": 1,
            "n_period_ringing": 2,
        }
    ]
    full = runner.sample_report(_positions(450, 70, 400, 100), reports)
    assert full["sample_ok"] and full["contributing_lines"] == 70
    assert full["primary_stratum"] == "in" and full["bottom_rule"] == "point"
    assert full["patch_failure_reasons"] == {"period_check_failed": 1}
    assert full["period_ringing_positions"] == 1 and full["period_ringing_patches"] == 2
    short = runner.sample_report(_positions(450, 70, 399, 99), reports)
    assert short["sample_ok"]  # fewer than 400 in-distribution no longer stops
    assert short["primary_stratum"] == "in_or_out"
    assert short["bottom_rule"] == "upper_bound"
    few_lines = runner.sample_report(_positions(450, 59, 450, 200), reports)
    assert not few_lines["sample_ok"]  # only < 60 contributing lines stops


def _level_row(widths: dict[float, float], aperture: float | None = None) -> dict:
    row = {}
    for level, width in widths.items():
        tag = runner.level_tag(level)
        row[f"width_{tag}"] = width
        row[f"height_{tag}"] = 4.0
        row[f"aperture_{tag}"] = level == aperture
    return row


def test_level_w_is_the_median_width_nearest_11_36_p_in_log_distance() -> None:
    rows = [
        _level_row({-6.0: 5.0, -10.0: 9.0, -12.0: 12.0, -20.0: 20.0}),
        _level_row({-6.0: 6.0, -10.0: 10.0, -12.0: 13.0, -20.0: 22.0}),
    ]
    level, report = runner.select_level_w(rows)
    assert level == -12.0  # |ln(12.5 / 11.36)| beats |ln(9.5 / 11.36)|
    medians = {tag: item["censored_median"] for tag, item in report.items()}
    assert medians == {"m6": 5.5, "m10": 9.5, "m12": 12.5, "m20": 21.0}
    tie = [_level_row({-6.0: 1.0, -10.0: 11.36 / 1.1, -12.0: 100.0, -20.0: 12.496})]
    assert runner.select_level_w(tie)[0] == -20.0  # equal log distance -> -20 dB
    assert runner.select_level_w([])[0] == -20.0


def test_aperture_limited_widths_are_censored_in_the_level_choice() -> None:
    rows = [
        _level_row({-6.0: 5.0, -10.0: 9.0, -12.0: 11.0, -20.0: 11.36}, aperture=-20.0),
        _level_row({-6.0: 5.0, -10.0: 9.0, -12.0: 11.0, -20.0: 11.36}, aperture=-20.0),
        _level_row({-6.0: 5.0, -10.0: 9.0, -12.0: 11.0, -20.0: 11.36}),
    ]
    level, report = runner.select_level_w(rows)
    assert report["m20"] == {
        "censored_median": None,
        "aperture_limited_share": 2 / 3,
        "candidate": False,
    }
    assert level == -12.0  # -20 dB would be exact but its median is infinite
    assert runner.censored_median([1.0, 2.0, 3.0], [False, False, True]) == 2.0


@pytest.mark.parametrize(
    ("widths", "limited", "expected"),
    [
        # Exactly half limited (even count): not more than half -> finite.
        ([4.0, 1.0, 9.0, 9.0], [False, False, True, True], 4.0),
        ([2.0, 7.0], [False, True], 2.0),
        # More than half limited -> +inf.
        ([1.0, 2.0, 3.0, 4.0], [False, True, True, True], math.inf),
        ([1.0, 2.0, 3.0], [False, True, True], math.inf),
        # Limited widths rank above every finite width, whatever their value.
        ([5.0, 6.0, 0.5, 8.0], [False, False, True, False], 7.0),
        ([3.0, 1.0, 2.0, 4.0], [False] * 4, 2.5),
    ],
)
def test_censored_median_is_infinite_only_above_half(widths, limited, expected) -> None:
    assert runner.censored_median(widths, limited) == expected


def test_censored_median_refuses_mismatched_or_empty_input() -> None:
    with pytest.raises(ValueError):
        runner.censored_median([1.0, 2.0], [False])
    with pytest.raises(ValueError):
        runner.censored_median([], [])


def test_natural_bottom_share_pools_candidates_per_cell() -> None:
    result = runner.LinePlanResult(
        specs=[],
        cost=None,  # type: ignore[arg-type]
        positions=[],
        report={"dataset": "4tu"},
        candidates={"M12": (3, 10), "M02": (0, 30)},
    )
    groups = {"M12": responses_io.ModelGroup.A1, "M02": responses_io.ModelGroup.A0}
    assert runner.natural_bottom_share([result, result], groups) == {
        "4tu/A0": 0.0,
        "4tu/A1": 0.3,
    }


def test_classify_positions_counts_aperture_limited_as_too_wide() -> None:
    rows = [
        _level_row({-12.0: 10.0, -20.0: 12.0}),
        _level_row({-12.0: 10.0, -20.0: 12.0}, aperture=-20.0),
        _level_row({-12.0: 5.0, -20.0: 7.0}),
    ]
    runner.classify_positions(rows, -20.0)
    assert [row["in_distribution"] for row in rows] == [True, False, False]
    assert [row["width_class"] for row in rows] == [
        "in",
        "width_over_lower_bound",
        "width_under",
    ]
    assert all(row["height_class"] == "height_under" for row in rows)
    summary = runner.level_summary(rows, -20.0)
    assert summary["in_distribution"] == 1 and summary["aperture_limited"] == 1
    assert summary["height_under"] == 3


# ------------------------------------------------------------------- P5b


@pytest.fixture(scope="module")
def library(helpers) -> responses_io.ResponseLibrary:
    response = helpers._response()
    return responses_io.ResponseLibrary(
        air={"M12": response},
        ground={},
        groups={"M12": responses_io.ModelGroup.A1},
        sha256={},
    )


def test_injected_rows_match_one_to_one_and_mask_the_conf_grid(
    helpers, line, library, tmp_path
) -> None:
    specs = [
        helpers._spec(realization_id="g00"),
        helpers._spec(index=1, realization_id="g00", apex_trace=helpers.APEX_TRACE + 1),
    ]
    settings = _settings(tmp_path, stage=runner.Stage.INJECTED)

    part = runner.evaluate_injected_line(
        line,
        {"g00": specs},
        inject.NO_EXCLUSIONS,
        library,
        {0: FakeDetector()},
        settings,
        level_w_db=-20.0,
    )

    assert part["n_accepted"] == 1  # the second row breaks the spacing (e')
    assert [row["reason"] for row in part["rejections"]] == [
        "spacing_below_width_plus_1p"
    ]
    (row,) = part["inj_matches"]
    assert row["hit"] and row["model_group"] == "A1" and row["n_boxes"] == 1
    grid = match.conf_grid()
    expected = "".join("1" if level < HYPERBOLA_SCORE else "0" for level in grid)
    assert row["hit_mask"] == expected
    for column in runner.MATCH_COLUMNS:
        assert column in row, column
    assert row["iou_visible_box"] > 0 and row["period_ringing"] is False
    assert row["n_clipped_cells"] == 0 and row["width_class"] in {
        "in",
        "width_over",
        "width_under",
        "width_over_lower_bound",
    }


def test_touched_tiles_cover_the_wing_box_plus_one_stride(helpers, library) -> None:
    long_data = np.tile(helpers._background_data(), (1, 8))
    radargram = helpers._radargram(long_data)
    image = runner.make_detection_image(radargram, inputs.recipe_config("r0"))
    assert len(image.tiles) > 3
    context = inject.prepare_line_context(radargram)
    result = inject.inject_realization(
        context, [helpers._spec()], {"M12": library.air["M12"]}
    )

    touched = runner.touched_tiles(image, result.truths)

    assert 0 in touched and len(touched) < len(image.tiles)


# -------------------------------------------------------------------- CLI


@pytest.fixture
def offline_run(monkeypatch, line):
    """Stage inputs replaced by the synthetic line and the fake detector."""
    detector = FakeBatchDetector()
    monkeypatch.setattr(inputs, "assert_list_hashes", lambda: {"lists": "x"})
    monkeypatch.setattr(inputs, "uninjected_lines", lambda: [line.ref])
    monkeypatch.setattr(inputs, "known_apexes", lambda: [])
    monkeypatch.setattr(runner, "assert_weights", lambda *_: {"0": "w"})
    monkeypatch.setattr(runner, "load_radargram", lambda ref: line.radargram)
    monkeypatch.setattr(runner, "load_detectors", lambda settings: {0: detector})
    return detector


def _final_line(capsys) -> tuple[dict, list[str]]:
    lines = capsys.readouterr().out.strip().splitlines()
    return json.loads(lines[-1]), lines


def _uninjected_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--stage",
        "uninjected",
        "--recipes",
        "r0",
        "--folds",
        "0",
        "--device",
        "cpu",
        "--allow-dirty",
        "--out-root",
        str(tmp_path),
        *extra,
    ]


def test_uninjected_cli_writes_files_and_prints_no_metric(
    offline_run, tmp_path, capsys
) -> None:
    code = runner.main(_uninjected_argv(tmp_path))

    final, lines = _final_line(capsys)
    assert code == 0 and final["status"] == "ok" and set(final) == FINAL_KEYS
    assert final["n_lines"] == 1 and final["n_tiles"] > 0
    assert not any(METRIC_WORDS.search(text) for text in lines)
    stage = tmp_path / "uninjected"
    for name in ("boxes", "conf_counts", "apex_hits", "flip_pairs"):
        assert (stage / "r0" / "fold0" / f"{name}.csv").is_file()
    summary = json.loads((stage / "summary.json").read_text(encoding="utf-8"))
    assert (
        summary["full_run"] is False and summary["run_identity"]["seed"] == inputs.SEED
    )


def test_existing_output_needs_resume_and_resume_reuses_parts(
    offline_run, tmp_path, capsys
) -> None:
    assert runner.main(_uninjected_argv(tmp_path)) == 0
    calls = offline_run.n_calls
    assert runner.main(_uninjected_argv(tmp_path)) == 2
    assert _final_line(capsys)[0]["status"] == "input_error"
    assert runner.main(_uninjected_argv(tmp_path, "--resume")) == 0
    assert offline_run.n_calls == calls  # the line part was reused
    assert runner.main(_uninjected_argv(tmp_path, "--resume", "--no-flip")) == 2


def test_list_sha_mismatch_refuses_with_exit_2(monkeypatch, tmp_path, capsys) -> None:
    changed = dict(inputs.EXPECTED_SHA256, k_apexes="0" * 64)
    monkeypatch.setattr(inputs, "EXPECTED_SHA256", changed)

    code = runner.main(_uninjected_argv(tmp_path))

    final, _ = _final_line(capsys)
    assert code == 2 and final["status"] == "input_error"
    assert not (tmp_path / "uninjected").exists()


def test_weights_sha_mismatch_refuses_with_exit_2(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(inputs, "assert_list_hashes", lambda: {})
    weights = (
        tmp_path / "runs" / f"{runner.stage_b.RUN_PREFIX}0" / "weights" / "last.pt"
    )
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not the N2-a' weights")

    code = runner.main(
        _uninjected_argv(tmp_path / "out", "--run-root", str(tmp_path / "runs"))
    )

    assert code == 2 and _final_line(capsys)[0]["status"] == "input_error"


def test_injection_table_sha_mismatch_refuses_with_exit_2(
    monkeypatch, tmp_path, capsys, helpers
) -> None:
    monkeypatch.setattr(inputs, "assert_list_hashes", lambda: {})
    monkeypatch.setattr(runner, "assert_weights", lambda *_: {})
    plan = tmp_path / "plan"
    plan.mkdir()
    inject.write_injection_table(plan / "injection_table.csv", [helpers._spec()])
    exclusions_sha = inputs.write_exclusions(plan / "exclusions.csv", {})
    record = tmp_path / "f2.json"
    record.write_text(
        json.dumps(
            {"injection_table_sha256": "0" * 64, "exclusions_sha256": exclusions_sha}
        ),
        encoding="utf-8",
    )

    code = runner.main(
        [
            "--stage",
            "injected",
            "--device",
            "cpu",
            "--allow-dirty",
            "--out-root",
            str(tmp_path),
            "--f2-record",
            str(record),
        ]
    )

    assert code == 2 and _final_line(capsys)[0]["status"] == "input_error"


def test_gate_runs_r0_on_k_lines_at_batch_size_one(line, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "load_radargram", lambda ref: line.radargram)
    settings = _settings(tmp_path, batch_size=4)
    apex = inputs.KnownApex("G1", line.ref.line_id, 100, 200.0)
    detector = FakeBatchDetector()

    gate = runner.run_identity_gate(settings, [line.ref], {0: detector}, [apex])

    assert gate["batch_size"] == 1
    assert gate["apex_rows"][0]["hit"] is True
    assert gate["box_counts"] == {f"{line.ref.line_id}|0": 2}
    assert gate["tile_score_max_difference_batch_vs_1"] == 0.0
    assert detector.n_batch_calls >= 1  # the batch path ran for the comparison
    assert not gate["result"]["pass"]  # one apex is not the D3 set of 41

    part = {"apex_hits": gate["apex_rows"], "boxes": []}
    report = runner.batch_path_report(gate, [part])
    assert report["apex_hit_differences"] == 0
    assert report["box_count_difference_conf025"] == 2


def test_full_run_stops_on_a_failed_gate(
    offline_run, monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(runner.RunSettings, "is_full", property(lambda self: True))
    code = runner.main(_uninjected_argv(tmp_path))
    final, _ = _final_line(capsys)
    assert code == 1 and final["status"] == "gate_failed"
    summary = json.loads(
        (tmp_path / "uninjected" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "gate_failed"
    assert not (tmp_path / "uninjected" / "r0").exists()  # no measurement


def test_f2_record_without_the_stratum_choices_is_refused(
    monkeypatch, tmp_path, capsys, helpers
) -> None:
    monkeypatch.setattr(inputs, "assert_list_hashes", lambda: {})
    monkeypatch.setattr(runner, "assert_weights", lambda *_: {})
    plan = tmp_path / "plan"
    plan.mkdir()
    table_sha = inject.write_injection_table(
        plan / "injection_table.csv", [helpers._spec()]
    )
    exclusions_sha = inputs.write_exclusions(plan / "exclusions.csv", {})
    record = tmp_path / "f2.json"
    record.write_text(
        json.dumps(
            {"injection_table_sha256": table_sha, "exclusions_sha256": exclusions_sha}
        ),
        encoding="utf-8",
    )

    code = runner.main(
        [
            "--stage",
            "injected",
            "--device",
            "cpu",
            "--allow-dirty",
            "--out-root",
            str(tmp_path),
            "--f2-record",
            str(record),
        ]
    )

    assert code == 2 and _final_line(capsys)[0]["status"] == "input_error"


def test_rejection_conditions_are_read_from_the_realization_id() -> None:
    rows = runner.with_condition(
        [
            {"realization_id": "01.1/Path3:g00:snr18:k1.3:p+1", "reason": "x"},
            {"realization_id": "plain", "reason": "y"},
        ]
    )
    assert (rows[0]["snr_db"], rows[0]["k"]) == ("18", "1.3")
    assert (rows[1]["snr_db"], rows[1]["k"]) == ("", "")


def test_stage_and_smoke_are_exclusive() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--stage", "plan", "--smoke"])
    with pytest.raises(SystemExit):
        runner.parse_args([])
    smoke = runner.parse_args(["--smoke"])
    assert smoke.recipes == ("r0",) and smoke.folds == (0,) and smoke.allow_dirty


# ----------------------------------------------------------------- loader


@pytest.mark.skipif(
    not responses_io.SYNTHETIC_DIR.is_dir()
    or not responses_io.SYNTHETIC_K7_DIR.is_dir(),
    reason="gprMax synthetic set (data/) not present",
)
def test_response_library_groups_and_quality() -> None:
    library = responses_io.load_library()

    assert set(library.air) == {*responses_io.A1_MODELS, *responses_io.A0_MODELS}
    assert set(library.ground) == set(responses_io.A0_MODELS)
    assert "M20" not in library.air
    for response in library.air.values():
        assert inject.response_quality_failure(response) is None, response.response_id
        assert response.coupling == "air" and response.source_peak_ns is not None
    m14 = library.air["M14"]
    assert m14.apex_time_ns == pytest.approx(11.09, abs=0.1)  # top, not the inner ring
    short = library.for_line(inputs.DATASET_IDS, 10.0)
    assert short and all(r.top_depth_m <= 1.0 for r in short.values())
    assert set(library.for_line(inputs.DATASET_IDS, 100.0)) == set(
        responses_io.A0_MODELS
    )


# ------------------------------------------------------------------ smoke


def test_p5b_time_counts_injection_once_and_inference_per_recipe_and_fold() -> None:
    import m32_recipe_smoke as smoke

    rates = smoke.Rates(
        load_per_line_tile=0.0,
        l1_per_line_tile=0.01,
        infer_per_tile=0.002,
        post_per_tile=0.0,
        context_per_line_tile=0.1,
        inject_per_realization=0.5,
        plan_per_line_tile=0.0,
    )
    work = [smoke.LineWork("L", l1_builds=20, inferred_tiles=40)]

    seconds = smoke.p5b_seconds(rates, work, {"L": 4}, n_recipes=3, n_folds=5)

    # context 0.4 + inject 20 x 0.5 + L1 3 x 20 x 4 x 0.01 + infer 15 x 40 x 0.002
    assert seconds == pytest.approx(0.4 + 10.0 + 2.4 + 1.2)
