"""M3-3 R3 band runner (``m32_recipe_eval.py --injectant band``, ``m33_r3_band``).

The synthetic line and pipe response of ``tests/test_inject.py`` stand in for
an IDS 400 MHz line and an A0 model; a stand-in detector replaces YOLO (no
box on the un-injected run, one ``pipe_band`` box over each tile on the
injected run). The integration test runs uninjected -> plan -> injected
through ``main`` and checks the files, the P4 block and the stdout contract;
no stored result and no model are read.
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
import m33_inputs
import m33_r3_band as band
import m33_r3_p4 as p4

from gpr_detection.detect.base import (
    Detection,
    DetectorOutput,
    InputTransform,
    PixelBox,
)
from gpr_detection.eval import inject
from gpr_detection.pipeline import (
    PP_SETS,
    PPAction,
    PPSetId,
    grid_box_to_image_box,
)

CLASS_NAMES = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")
METRIC_WORDS = re.compile(r"recall|ghost|survival|delta|ap50|cover", re.IGNORECASE)
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
    spec = importlib.util.spec_from_file_location("inject_band_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EmptyDetector:
    """No box at all (the un-injected run leaves zone (c') empty)."""

    class_names = CLASS_NAMES

    def predict(self, image: np.ndarray) -> DetectorOutput:
        size = image.shape[0]
        return DetectorOutput((), InputTransform(1, 1, 0, 0, size, size))


class BandDetector(EmptyDetector):
    """One ``pipe_band`` box over the whole tile (covers any band it sees)."""

    def predict(self, image: np.ndarray) -> DetectorOutput:
        size = image.shape[0]
        box = Detection(1, "pipe_band", 0.6, PixelBox(0, 0, size, size))
        return DetectorOutput((box,), InputTransform(1, 1, 0, 0, size, size))


# --------------------------------------------------------------- planning


@pytest.fixture(scope="module")
def context(helpers) -> inject.LineContext:
    return inject.prepare_line_context(helpers._radargram(helpers._background_data()))


@pytest.fixture(scope="module")
def patch(helpers, context) -> inject.ResponsePatch:
    built = inject.build_response_patch(helpers._response(), context, inject.K_MAIN)
    assert isinstance(built, inject.ResponsePatch)
    return built


def test_band_plan_is_seeded_spaced_and_injectable(helpers, context, patch) -> None:
    plan = inject.plan_band_positions(context, [patch])
    again = inject.plan_band_positions(context, [patch])

    assert plan == again and plan.positions  # seeded: same plan twice
    w_eff = inject.effective_median_window(context)
    for group in {position.group for position in plan.positions}:
        members = [p for p in plan.positions if p.group == group]
        placed = []
        for position in members:
            assert inject.band_spacing_ok(
                position.center_trace, position.length_traces, placed, w_eff
            )
            placed.append((position.center_trace, position.length_traces))
    specs = inject.build_band_specs(
        helpers.LINE_ID, plan.positions, inject.band_conditions()
    )
    assert len(specs) == len(plan.positions) * 8  # SNR 4 x polarity 2
    first = specs[0]
    result = inject.inject_band_realization(
        context,
        [s for s in specs if s.realization_id == first.realization_id],
        {"M12": helpers._response()},
    )
    assert result.truths and not result.rejected
    truth = result.truths[0]
    assert band.position_id_of(truth.injection_id) == inject.band_position_id(
        helpers.LINE_ID, plan.positions[0].group, 0
    )


def test_4tu_lengths_keep_the_fresnel_margin() -> None:
    assert inject.band_length_fits(200, 14, 235, keep_fresnel_margin=False)
    assert not inject.band_length_fits(222, 14, 235, keep_fresnel_margin=False)
    assert not inject.band_length_fits(208, 14, 235, keep_fresnel_margin=True)
    assert inject.band_length_fits(207, 14, 235, keep_fresnel_margin=True)


def test_plan_truth_l1_box_is_the_cell_edge_map(context, patch) -> None:
    """[RUN] §3-1 4 좌표: L1 truth = L0 cell edges, ``y2 = (k1 + 1) / sigma_y``."""
    built = inject.build_band_patch(patch, context, 40)
    assert isinstance(built, inject.BandPatch)
    center = 117.5
    truth = inject.band_truth_box(built, center, context.background.n_samples)
    pixel = grid_box_to_image_box(truth, context.grid_map)
    grid = context.grid_map
    assert (grid.trace_offset, grid.sample_offset) == (0, 0)
    assert pixel.y1 == truth.k0 / grid.samples_per_px
    assert pixel.y2 == (truth.k1 + 1) / grid.samples_per_px
    assert pixel.x2 == (truth.i1 + 1) / grid.traces_per_px
    position = inject.PlannedBand(0, "M12", center, 40, built.taper_traces)
    row = band._position_row(
        inputs.LineRef(inputs.DATASET_IDS, context.background.line_id, "background"),
        context,
        built,
        position,
        0,
        {"M12": responses_io.ModelGroup.A0},
        band.FREQ_MAIN,
        PP_SETS[PPSetId.S3],
    )
    assert row["truth_y2_px"] == pixel.y2 and row["truth_x1_px"] == pixel.x1
    content = band.content_of(context)
    assert row["y_bot_px"] == content.y_bot_px == grid.n_samples / grid.samples_per_px


def test_freq_strata_and_px_strata() -> None:
    assert band.freq_stratum_of(400.0) == "main_400"
    assert band.freq_stratum_of(2000.0) == "excluded_2ghz"
    assert band.freq_stratum_of(900.0) == "report_other"
    assert [band.px_stratum(v) for v in (63.9, 64.0, 192.0, 192.1)] == [
        "lt64",
        "64_192",
        "64_192",
        "gt192",
    ]


def test_box_rows_round_trip_to_pp_boxes() -> None:
    rows = [
        {"fold": "0", "line_id": "L", "box_index": "0", "pp": "pre", "class_id": "1",
         "score": "0.6", "in_grid": "True", "i0": "1", "i1": "9", "k0": "2",
         "k1": "8", "pp_action": "clipped_b"},
        {"fold": "0", "line_id": "L", "box_index": "0", "pp": "post", "class_id": "1",
         "score": "0.6", "in_grid": "True", "i0": "1", "i1": "9", "k0": "2",
         "k1": "6", "pp_action": "clipped_b"},
        {"fold": "0", "line_id": "L", "box_index": "1", "pp": "pre", "class_id": "1",
         "score": "0.3", "in_grid": True, "i0": 3, "i1": 4, "k0": 5,
         "k1": 6, "pp_action": "removed_d"},
        {"fold": "0", "line_id": "L", "box_index": "2", "pp": "pre", "class_id": "1",
         "score": "0.3", "in_grid": "False", "i0": "", "i1": "", "k0": "",
         "k1": "", "pp_action": "removed_a"},
    ]  # fmt: skip
    (clipped, removed) = band.pp_boxes_from_rows(rows, "line_id")[("L", 0)]
    assert clipped.action is PPAction.CLIPPED_B
    assert (clipped.before.k1, clipped.after.k1) == (8, 6)
    assert removed.after is None and removed.action is PPAction.REMOVED_D


@pytest.fixture(scope="module")
def long_context(helpers) -> inject.LineContext:
    """1,880 traces: the moving median (W 301) and several bands per group."""
    data = np.tile(helpers._background_data(), (1, 8))
    return inject.prepare_line_context(helpers._radargram(data))


@pytest.fixture(scope="module")
def long_patch(helpers, long_context) -> inject.ResponsePatch:
    built = inject.build_response_patch(
        helpers._response(), long_context, inject.K_MAIN
    )
    assert isinstance(built, inject.ResponsePatch)
    return built


def test_spacing_inside_a_group_follows_the_w_eff_rule(long_context, long_patch):
    """Runner spec §3-1 4: |c1 - c2| >= (L1 + L2) / 2 + W_eff in a group;
    bands of other groups keep their injected columns apart."""
    plan = inject.plan_band_positions(long_context, [long_patch])
    w_eff = inject.effective_median_window(long_context)
    assert w_eff == 301
    groups = {}
    for position in plan.positions:
        groups.setdefault(position.group, []).append(position)
    assert max(len(members) for members in groups.values()) >= 2
    for first in plan.positions:
        for second in plan.positions:
            if first is second:
                continue
            gap = abs(first.center_trace - second.center_trace)
            if first.group == second.group:
                needed = (first.length_traces + second.length_traces) / 2 + w_eff
            else:
                needed = (first.n_columns + second.n_columns) / 2
            assert gap >= needed, (first, second)


def test_preferred_models_are_visited_first(helpers, long_context, long_patch):
    """900 MHz line: M11, M14, M16 before the other models (§3-1 1, [EMR] D-6)."""
    other = inject.build_response_patch(
        helpers._response("M14"), long_context, inject.K_MAIN
    )
    assert isinstance(other, inject.ResponsePatch)
    patches = [long_patch, other]
    plain = inject.plan_band_positions(long_context, patches)
    assert (
        inject.plan_band_positions(long_context, patches, preferred_models=()) == plain
    )
    preferred = inject.plan_band_positions(
        long_context, patches, preferred_models=("M14",)
    )
    m14 = [p for p in preferred.positions if p.response_id == "M14"]
    # Every M14 candidate is tried before any M12 one, so M14 fills the line
    # first and M12 can only take what is left.
    assert len(m14) >= sum(p.response_id == "M14" for p in plain.positions)
    assert m14 and preferred.positions[0].response_id in ("M14", "M12")
    assert band.preferred_models(900.0) == ("M11", "M14", "M16")
    assert band.preferred_models(400.0) == ()


def test_inference_covers_every_tile(long_context) -> None:
    """r0 has whole-line steps (period, AGC floor, 99th percentile clip)."""
    image = runner.make_detection_image(
        long_context.background, inputs.recipe_config("r0")
    )
    tiles = band.inference_tiles(image)
    assert tiles == tuple(range(len(image.tiles)))
    assert band.tile_trace_range(image, tiles) == (0, image.grid_map.n_traces - 1)
    first, last = band.tile_trace_range(image, tiles[:1])
    assert first == 0 and last < image.grid_map.n_traces - 1


def test_a_band_changes_the_r0_image_far_beyond_the_median_window(
    helpers, long_context, long_patch
) -> None:
    """Item 2 evidence: pixels more than W_eff / 2 from the band still change."""
    built = inject.build_band_patch(long_patch, long_context, 300)
    assert isinstance(built, inject.BandPatch)
    center = 200 + (built.n_columns - 1) / 2  # left part of the line
    spec = inject.InjectionSpec(
        "x:00", "x", helpers.LINE_ID, center, "M12", 18.0, 1, inject.K_MAIN, 0,
        mode=inject.InjectionMode.BAND, band_length_traces=300,
        band_center_trace=center,
    )  # fmt: skip
    result = inject.inject_band_realization(
        long_context, [spec], {"M12": helpers._response()}
    )
    assert result.truths
    config = inputs.recipe_config("r0")
    background = runner.make_detection_image(long_context.background, config)
    injected = runner.make_detection_image(
        replace(long_context.background, data=result.data.astype(np.float32)), config
    )
    far_trace = result.truths[0].box.i1 + built.taper_traces + 301
    far_px = math.ceil(far_trace / background.grid_map.traces_per_px)
    assert far_px < background.width_px
    changed = injected.line_image[:, far_px:] != background.line_image[:, far_px:]
    assert changed.any()


def test_batched_survival_ratios_equal_the_single_band_function(
    helpers, long_context
) -> None:
    patch = inject.build_response_patch(
        helpers._response(), long_context, inject.K_MAIN
    )
    plan = inject.plan_band_positions(long_context, [patch])
    specs = inject.build_band_specs(
        helpers.LINE_ID, plan.positions, inject.band_conditions()
    )
    first = specs[0].realization_id
    result = inject.inject_band_realization(
        long_context,
        [s for s in specs if s.realization_id == first],
        {"M12": helpers._response()},
    )
    assert len(result.truths) >= 2
    section = inject.background_step4_section(long_context)
    batched = inject.band_survival_ratios(
        long_context, result.data, result.truths, section
    )
    single = [
        inject.band_survival_ratio(long_context, result.data, truth)
        for truth in result.truths
    ]
    assert batched == single


def test_realization_images_are_built_once_for_every_fold(
    helpers, monkeypatch, context
) -> None:
    """Item 8 (a): more folds infer more, but build no extra r0 image."""
    plan = inject.plan_band_positions(
        context,
        [inject.build_response_patch(helpers._response(), context, inject.K_MAIN)],
    )
    specs = inject.build_band_specs(
        helpers.LINE_ID, plan.positions, inject.band_conditions()
    )
    realizations = runner.group_specs(specs)[helpers.LINE_ID]
    built = []
    original = band.make_detection_image

    def counting(radargram, config):
        built.append(1)
        return original(radargram, config)

    monkeypatch.setattr(band, "make_detection_image", counting)
    ref = inputs.LineRef(inputs.DATASET_IDS, helpers.LINE_ID, "background")
    line = runner.LineData(ref, context.background, context.zero)
    settings = runner.parse_args(
        ["--injectant", "band", "--stage", "injected", "--device", "cpu"]
    )
    library = responses_io.ResponseLibrary(
        air={}, ground={"M12": helpers._response()},
        groups={"M12": responses_io.ModelGroup.A0}, sha256={},
    )  # fmt: skip
    parts = {}
    for folds in ((0,), (0, 1)):
        built.clear()
        line_inputs = band.BandLineInputs(
            realizations=realizations,
            exclusions=inject.NO_EXCLUSIONS,
            library=library,
            detectors=dict.fromkeys(folds, BandDetector()),
            config=PP_SETS[PPSetId.S3],
            freq="main_400",
        )
        parts[folds] = band.evaluate_band_line(line, line_inputs, settings)
        assert len(built) == len(realizations)  # one image per realization
    fold0 = [row for row in parts[(0, 1)]["boxes"] if row["fold"] == 0]
    assert fold0 == parts[(0,)]["boxes"]
    assert parts[(0,)]["truths"] == parts[(0, 1)]["truths"]
    assert set(parts[(0,)]["timings"]) >= {
        band.PHASE_INJECT,
        band.PHASE_L1,
        band.PHASE_INFER,
        band.PHASE_SURVIVAL,
        band.PHASE_PP,
    }


def test_padding_acts_and_realization_counts() -> None:
    rows = [
        {"pp": "pre", "in_grid": "False", "class_id": "1", "score": "0.6",
         "pp_action": "removed_a"},
        {"pp": "pre", "in_grid": "False", "class_id": "1", "score": "0.2",
         "pp_action": "removed_a"},  # under the operating threshold
        {"pp": "pre", "in_grid": "False", "class_id": "0", "score": "0.6",
         "pp_action": "removed_a"},  # not pipe_band
        {"pp": "pre", "in_grid": "True", "class_id": "1", "score": "0.6",
         "pp_action": "removed_a"},  # on the grid: matched, not padding
        {"pp": "pre", "in_grid": False, "class_id": 1, "score": 0.9,
         "pp_action": "none"},
    ]  # fmt: skip
    assert band.padding_only_acts(rows) == {"total": 1, "by_rule": {"removed_a": 1}}
    parts = [
        {"realizations": [
            {"period_changed": True, "zero_changed": False},
            {"period_changed": "False", "zero_changed": "True"},
        ]},
    ]  # fmt: skip
    assert band.realization_counts(parts) == {
        "n_realizations": 2,
        "n_period_changed": 1,
        "n_zero_changed": 1,
    }


# ------------------------------------------------------- end-to-end run


@pytest.fixture
def offline_band(monkeypatch, helpers, tmp_path_factory):
    """One IDS 400 MHz stand-in line, an A0 stand-in model, no stored input."""
    radargram = helpers._radargram(helpers._background_data())
    ref = inputs.LineRef(inputs.DATASET_IDS, helpers.LINE_ID, "background")
    library = responses_io.ResponseLibrary(
        air={},
        ground={"M12": helpers._response()},
        groups={"M12": responses_io.ModelGroup.A0},
        sha256={},
    )
    c0b = tmp_path_factory.mktemp("c0b") / "d0_gate.json"
    selection = {"pp_set_id": "S3", "pp_config_sha256": PP_SETS[PPSetId.S3].sha256()}
    c0b.write_text(json.dumps({"selection": selection}), encoding="utf-8")
    monkeypatch.setattr(m33_inputs, "C0B_D0_GATE_JSON", c0b)
    monkeypatch.setattr(m33_inputs, "C0B_D0_GATE_SHA256", inputs.sha256_file(c0b))
    monkeypatch.setattr(inputs, "assert_list_hashes", lambda: {})
    monkeypatch.setattr(inputs, "background_lines", lambda: [ref])
    monkeypatch.setattr(
        inputs, "candidate_apexes", lambda: inputs.CandidateApexes({}, {})
    )
    monkeypatch.setattr(band, "ids_freq_strata", lambda: {ref.line_id: "main_400"})
    monkeypatch.setattr(
        p4,
        "IDS_FREQ_STRATUM_COUNTS",
        {"main_400": 1, "report_other": 0, "excluded_2ghz": 0},
    )
    monkeypatch.setattr(responses_io, "load_library", lambda *_: library)
    monkeypatch.setattr(runner, "assert_weights", lambda *_: {"0": "w"})
    monkeypatch.setattr(runner, "load_radargram", lambda ref: radargram)
    detectors = {"uninjected": EmptyDetector(), "injected": BandDetector()}
    monkeypatch.setattr(
        runner,
        "load_detectors",
        lambda settings: {0: detectors[settings.stage.value]},
    )
    return ref


def _argv(out_root: Path, stage: str, *extra: str) -> list[str]:
    return [
        "--injectant",
        "band",
        "--stage",
        stage,
        "--folds",
        "0",
        "--device",
        "cpu",
        "--allow-dirty",
        "--backgrounds",
        "gz_ids",
        "--out-root",
        str(out_root),
        *extra,
    ]


def _final(capsys) -> tuple[dict, list[str]]:
    lines = capsys.readouterr().out.strip().splitlines()
    return json.loads(lines[-1]), lines


def _run_all(out: Path, capsys) -> list[tuple[int, dict, list[str]]]:
    results = []
    for stage, extra in (
        ("uninjected", ()),
        ("plan", ()),
        ("injected", ("--c1b-record", str(out / "plan" / "c1b_record_draft.json"))),
    ):
        code = runner.main(_argv(out, stage, *extra))
        final, lines = _final(capsys)
        results.append((code, final, lines))
    return results


def test_plan_to_injected_runs_end_to_end(offline_band, tmp_path, capsys) -> None:
    results = _run_all(tmp_path, capsys)

    for code, final, lines in results:
        assert (code, final["status"]) == (1, "not_judgeable"), final
        assert set(final) == FINAL_KEYS
        assert not any(METRIC_WORDS.search(text) for text in lines)
    assert (tmp_path / "uninjected_ids" / "fold0" / "boxes.csv").is_file()

    plan = tmp_path / "plan"
    positions = inputs.read_csv(plan / "positions.csv")
    assert positions and all(row["dataset"] == "gz_ids" for row in positions)
    report = json.loads((plan / "plan_report.json").read_text(encoding="utf-8"))
    assert report["p4_plan"]["pp_set_id"] == "S3"
    assert report["p4_plan"]["mde_pp"] is None
    assert report["judgment"]["folds_run"] is None
    record = json.loads((plan / "c1b_record_draft.json").read_text(encoding="utf-8"))
    assert record["positions_sha256"] == inputs.sha256_file_lf(plan / "positions.csv")

    injected = tmp_path / "injected"
    truths = inputs.read_csv(injected / "truths.csv")
    assert truths and {row["mode"] for row in truths} == {"band"}
    assert all(row["survival_ratio"] != "" for row in truths)
    assert {row["position_id"] for row in truths} <= {
        row["position_id"] for row in positions
    }
    matches = inputs.read_csv(injected / "fold0" / "inj_matches.csv")
    assert len(matches) == len(truths)
    assert all(int(row["covered_pre"]) == int(row["length_traces"]) for row in matches)
    for name in ("inj_boxes", "ghosts"):
        assert (injected / "fold0" / f"{name}.csv").is_file()
    assert (injected / p4.ACTED_FILE).is_file()
    assert (injected / p4.ARTIFACT_FILE).is_file()

    summary = json.loads((injected / "summary.json").read_text(encoding="utf-8"))
    p4_block = summary["p4_adoption_input"]
    assert p4_block["n_lines"] == 1 and p4_block["defects"] == []
    assert p4_block["pp_set_id"] == "S3" and p4_block["mde_pp"] is None
    assert p4_block["p4_status"] in {s.value for s in p4.P4Status}
    scope = [row for row in positions if row["p4_scope"] == "True"]
    in_strata = [row for row in scope if row["lw_stratum"] in p4.P4_STRATA]
    assert sum(p4_block["n_positions_by_stratum"].values()) == len(in_strata)
    assert in_strata, "the fixture should reach the P4 strata"
    assert sum(p4_block["n_truth_folds_by_stratum"].values()) > 0
    assert summary["judgment"]["blockers"]  # partial: never a judgment input
    assert "recall" in summary and "survival_ratio" in summary
    assert summary["bottom"]["report_only"] is True and "recall" in summary["bottom"]
    assert summary["n_realizations"] > 0 and "n_zero_changed" in summary
    assert summary["pp_acted_padding_only"]["total"] >= 0
    assert "pp_acted_padding_only" in p4_block
    assert {band.PHASE_INFER, band.PHASE_MATCH, band.PHASE_P4} <= set(
        summary["timings"]
    )
    plan_timings = report["timings"]
    assert band.PHASE_PLAN in plan_timings
    uninjected = json.loads(
        (tmp_path / "uninjected_ids" / "summary.json").read_text(encoding="utf-8")
    )
    assert band.PHASE_UNINJECTED_INFER in uninjected["timings"]


def test_smoke_runs_the_three_stages_and_is_never_judgeable(
    offline_band, tmp_path, capsys
) -> None:
    argv = ["--injectant", "band", "--smoke", "--device", "cpu"]
    code = runner.main([*argv, "--out-root", str(tmp_path)])

    final, lines = _final(capsys)
    assert (code, final["status"], final["stage"]) == (1, "not_judgeable", "smoke")
    assert not any(METRIC_WORDS.search(text) for text in lines)
    summary = json.loads(
        (tmp_path / "injected" / "summary.json").read_text(encoding="utf-8")
    )
    assert "smoke" in summary["judgment"]["blockers"]
    assert summary["p4_adoption_input"]["defects"] == []
    estimate = json.loads(
        (tmp_path / "smoke_estimate.json").read_text(encoding="utf-8")
    )
    assert estimate["full_seconds_upper"]["total"] > 0
    assert estimate["rates_s"]["injected_gpu_per_realization_tile_fold"] > 0
    assert any(line.startswith("timing ") for line in lines)


def test_a_p4_defect_ends_the_injected_stage_with_exit_1(
    offline_band, monkeypatch, tmp_path, capsys
) -> None:
    _run_all(tmp_path, capsys)

    def defect(inputs_p4, stage_dir):
        raise p4.P4DefectError("x", {"defects": ["x"], "p4_status": "no_effect"})

    monkeypatch.setattr(p4, "run_p4", defect)
    extra = ("--c1b-record", str(tmp_path / "plan" / "c1b_record_draft.json"))
    code = runner.main(_argv(tmp_path, "injected", *extra, "--overwrite"))

    assert (code, _final(capsys)[0]["status"]) == (1, "p4_defect")
    summary = json.loads(
        (tmp_path / "injected" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "p4_defect"
    assert summary["p4_adoption_input"]["defects"] == ["x"]


def test_injected_refuses_a_changed_plan_table(offline_band, tmp_path, capsys) -> None:
    _run_all(tmp_path, capsys)
    table = tmp_path / "plan" / "positions.csv"
    table.write_text(table.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    extra = ("--c1b-record", str(tmp_path / "plan" / "c1b_record_draft.json"))
    code = runner.main(_argv(tmp_path, "injected", *extra, "--overwrite"))
    assert (code, _final(capsys)[0]["status"]) == (2, "input_error")


def test_band_settings_and_blockers(tmp_path) -> None:
    settings = runner.parse_args(_argv(tmp_path, "plan"))
    assert settings.is_band and settings.recipes == ("r0",)
    assert settings.scope_blockers() == ["line_limit"]  # 4TU background missing
    full = runner.parse_args(
        ["--injectant", "band", "--stage", "injected", "--out-root", str(tmp_path)]
    )
    assert full.scope_blockers() == [] and full.expected_recipes == ("r0",)
    assert full.stage_dir == tmp_path / "injected"
    uninjected = runner.parse_args(
        ["--injectant", "band", "--stage", "uninjected", "--out-root", str(tmp_path)]
    )
    assert uninjected.stage_dir == tmp_path / "uninjected_ids"
    smoke = runner.parse_args(["--injectant", "band", "--smoke"])
    assert smoke.out_root == m33_inputs.R3_SMOKE_OUT_ROOT and smoke.folds == (0,)
    with pytest.raises(SystemExit):
        runner.parse_args(_argv(tmp_path, "plan", "--recipes", "r5"))
