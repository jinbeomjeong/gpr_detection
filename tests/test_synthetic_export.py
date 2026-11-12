"""Tests for ``scripts/synthetic/export_dataset.py`` (multi-pipe S1g, one-pipe schema).

Fake gprMax h5 files hold Ricker pulses at the ray times the exporter itself
predicts, so the checks are about the per-pipe bookkeeping (apex trace, picks,
json keys), not about the ray physics. The regression against the stored M01-M20
export is opt-in (``-m synthetic``) like the depth regression.
"""

import csv
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import pytest

from gpr_detection.io.synthetic import (
    load_synthetic,
    load_synthetic_pipe_truths,
    load_synthetic_truth,
)

PROJECT = Path(__file__).resolve().parents[1]
STORED_EXPORT = PROJECT / "data" / "synthetic_gprmax"
REGRESSION_MODEL = "M12"  # air-coupled with a free-space reference: covers every branch

DT_NS = 0.0118
N_SAMPLES = 2400
FREQ_HZ = 500e6
VELOCITY_M_PER_NS = 0.299792458 / 3.0
SCAN_START_M = 0.4
TRACE_STEP_M = 0.04
N_TRACES = 31
ANTENNA_HEIGHT_M = 0.1  # y_antenna_m - y_surface_m in the fake models json
BACKGROUND_LINE_DRIFT = 1e-6  # relative gain change per trace

# Single-pipe json schema of the M01-M20 export, in file order.
SINGLE_PIPE_KEYS = [
    "model_id",
    "array_layout",
    "dt_ns",
    "n_samples",
    "n_traces",
    "dx_m",
    "time_of_sample0_ns",
    "trace_x_m_first",
    "trace_x_m",
    "eps_r",
    "soil_sigma_S_per_m",
    "velocity_m_per_ns",
    "pipe_material",
    "pipe_wall_m",
    "pipe_center_x_m",
    "pipe_top_depth_m",
    "pipe_center_depth_m",
    "pipe_radius_m",
    "center_freq_hz",
    "wavelet",
    "air_gap_m",
    "antenna_offset_m",
    "fdtd_cell_m",
    "pml_cells",
    "time_window_ns",
    "t0_definition",
    "t0_top_ns",
    "t0_center_ns",
    "air_two_way_ns",
    "ray_reflection_time_ns",
    "ray_reflection_time_definition",
    "time_zero_candidates_ns",
    "surface_reflection_geom_ns",
    "direct_air_geom_ns",
    "apex",
    "t0_effective_ns",
    "depth_from_t0_effective_m",
    "velocity_check",
    "trace_envelope_pick_ns",
    "checks",
    "files",
]


@pytest.fixture
def exporter(load_script, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # The script prepends its own folder to sys.path; keep that local to the test.
    monkeypatch.setattr(sys, "path", list(sys.path))
    return load_script("synthetic/export_dataset")


def _ricker(t_ns: np.ndarray, centre_ns: float) -> np.ndarray:
    arg = (math.pi * FREQ_HZ * 1e-9 * (t_ns - centre_ns)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def _write_h5(path: Path, traces: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["dt"] = DT_NS * 1e-9
        handle.attrs["Iterations"] = traces.shape[0]
        handle["rxs/rx1/Ez"] = traces.astype(np.float32)


def _models_json(tmp_path: Path) -> Path:
    common = {
        "eps_r": 9.0,
        "soil_sigma_S_per_m": 0.001,
        "velocity_m_per_ns": VELOCITY_M_PER_NS,
        "center_freq_hz": FREQ_HZ,
        "air_gap_m": 0.1,
        "antenna_offset_m": 0.1,
        "fdtd_cell_m": 0.005,
        "pml_cells": 20,
        "y_surface_m": 0.7,
        "y_antenna_m": 0.8,
        "time_window_ns": 28.0,
        "pipe_material": "pec",
        "pipe_radius_m": 0.05,
        "pipe_top_depth_m": 0.3,
    }
    scan = {
        "scan_start_m": SCAN_START_M,
        "trace_step_m": TRACE_STEP_M,
        "n_traces": N_TRACES,
    }
    models = [
        {
            "id": "T_D060",
            "input": "in/T_D060.in",
            "pipe_center_x_m": [1.4, 0.8],
            "apex_trace_index": [10, 25],
            "output": "h5/T_D060_merged.h5",
            **scan,
        },
        {
            "id": "T_C1",
            "input": "in/T_C1.in",
            "pipe_center_x_m": [1.0],
            "apex_trace_index": [15],
            "output": "h5/T_C1_merged.h5",
            **scan,
        },
        {
            "id": "T_BGL",
            "input": "in/T_BGL.in",
            "pipe_center_x_m": [],
            "apex_trace_index": [],
            "output": "h5/T_BGL_merged.h5",
            **scan,
        },
        {"id": "T_bg", "pipe_center_x_m": [], "output": "h5/T_bg_merged.h5"},
        {"id": "T_fs", "pipe_center_x_m": [], "output": "h5/T_fs.h5"},
    ]
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"common": common, "models": models}), encoding="utf-8")
    return path


def _write_fake_runs(exporter: ModuleType, tmp_path: Path) -> Path:
    """Write background, free-space and pipe B-scans with echoes at the ray times."""
    h5_dir = tmp_path / "h5"
    h5_dir.mkdir()
    t_ns = np.arange(N_SAMPLES) * DT_NS
    source_peak_ns = math.sqrt(2.0) / (FREQ_HZ * 1e-9)
    direct = _ricker(t_ns, source_peak_ns + exporter.direct_air_time(0.1))
    surface = 0.5 * _ricker(
        t_ns, source_peak_ns + exporter.surface_reflection_time(0.1, 0.1)
    )
    background = direct + surface
    _write_h5(h5_dir / "T_bg_merged.h5", np.tile(background[:, None], (1, 3)))
    _write_h5(h5_dir / "T_fs.h5", direct[:, None])

    x_mid = SCAN_START_M + np.arange(N_TRACES) * TRACE_STEP_M
    # A small lateral gain drift stands in for the FDTD lateral residual of a real line.
    lateral_gain = 1.0 + BACKGROUND_LINE_DRIFT * np.arange(N_TRACES)
    for model_id, pipe_xs in (("T_D060", (0.8, 1.4)), ("T_C1", (1.0,)), ("T_BGL", ())):
        bscan = background[:, None] * lateral_gain[None, :]
        for amplitude, x0 in zip((1.0, 0.8), pipe_xs, strict=False):
            ray_ns = exporter.trace_reflection_times(
                x_mid, 0.1, 0.1, x0, 0.3, 0.05, VELOCITY_M_PER_NS
            )
            for i, ray in enumerate(ray_ns):
                bscan[:, i] += amplitude * _ricker(t_ns, source_peak_ns + ray)
        _write_h5(h5_dir / f"{model_id}_merged.h5", bscan)
    return h5_dir


@pytest.fixture
def s1g_export(exporter: ModuleType, tmp_path: Path) -> tuple[ModuleType, Path]:
    h5_dir = _write_fake_runs(exporter, tmp_path)
    out_dir = tmp_path / "out"
    exporter.main(
        [
            "--s1g",
            str(_models_json(tmp_path)),
            "--h5-dir",
            str(h5_dir),
            "--out-dir",
            str(out_dir),
        ]
    )
    return exporter, out_dir


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_two_pipe_json_lists_every_pipe_with_its_own_apex(s1g_export):
    _, out_dir = s1g_export
    meta = _read_json(out_dir / "T_D060.json")
    assert meta["n_pipes"] == 2
    pipes = meta["pipes"]
    assert [p["pipe_center_x_m"] for p in pipes] == [0.8, 1.4]  # sorted by x
    assert [p["apex_trace_index"] for p in pipes] == [10, 25]
    for pipe in pipes:
        assert pipe["pipe_top_depth_m"] == 0.3
        assert pipe["pipe_center_depth_m"] == pytest.approx(0.35)
        assert pipe["t0_top_ns"] == pytest.approx(2 * 0.3 / VELOCITY_M_PER_NS)
        ray = pipe["ray_reflection_time_ns"]
        assert len(ray) == N_TRACES
        assert int(np.argmin(ray)) == pipe["apex_trace_index"]
        apex = pipe["apex"]
        assert apex["trace_index"] == pipe["apex_trace_index"]
        assert apex["wavelet_lag_ns"] == pytest.approx(0.0, abs=0.02)
        assert pipe["velocity_check"][
            "apparent_v_ratio_from_pick_slope"
        ] == pytest.approx(1.0, abs=0.01)


def test_velocity_picks_stay_on_their_own_side_of_the_midpoint(s1g_export):
    _, out_dir = s1g_export
    pipes = _read_json(out_dir / "T_D060.json")["pipes"]
    x_mid = SCAN_START_M + np.arange(N_TRACES) * TRACE_STEP_M
    midpoint_m = 1.1
    left_picks = [
        x
        for x, pick in zip(x_mid, pipes[0]["trace_envelope_pick_ns"], strict=True)
        if pick is not None
    ]
    right_picks = [
        x
        for x, pick in zip(x_mid, pipes[1]["trace_envelope_pick_ns"], strict=True)
        if pick is not None
    ]
    assert left_picks and max(left_picks) < midpoint_m
    assert right_picks and min(right_picks) > midpoint_m
    assert pipes[0]["velocity_check"]["n_picks"] == len(left_picks)


def test_two_pipe_json_keeps_single_pipe_keys_for_pipe_zero(s1g_export):
    _, out_dir = s1g_export
    meta = _read_json(out_dir / "T_D060.json")
    assert list(meta)[: len(SINGLE_PIPE_KEYS)] == SINGLE_PIPE_KEYS
    first = meta["pipes"][0]
    for key in (
        "pipe_center_x_m",
        "pipe_top_depth_m",
        "ray_reflection_time_ns",
        "apex",
        "velocity_check",
    ):
        assert meta[key] == first[key]
    assert meta["files"]["gprmax_input"] == "in/T_D060.in"
    assert "bg_lateral_diff_rel_x0.4_x3.6" in meta["checks"]
    truth = load_synthetic_truth("T_D060", root=out_dir)
    assert truth.pipe_center_x_m == 0.8
    assert truth.apex_trace_index == 10


def test_single_pipe_s1g_model_also_gets_a_pipe_list(s1g_export):
    _, out_dir = s1g_export
    meta = _read_json(out_dir / "T_C1.json")
    assert meta["n_pipes"] == 1
    assert meta["pipes"][0]["apex_trace_index"] == 15
    assert meta["apex"] == meta["pipes"][0]["apex"]


def test_background_line_has_no_pipe_and_null_pipe_keys(s1g_export):
    _, out_dir = s1g_export
    meta = _read_json(out_dir / "T_BGL.json")
    assert meta["n_pipes"] == 0
    assert meta["pipes"] == []
    assert list(meta)[: len(SINGLE_PIPE_KEYS)] == SINGLE_PIPE_KEYS
    for key in (
        "pipe_material", "pipe_center_x_m", "pipe_top_depth_m", "pipe_radius_m",
        "t0_top_ns", "ray_reflection_time_ns", "apex", "velocity_check",
        "trace_envelope_pick_ns",
    ):  # fmt: skip
        assert meta[key] is None, key
    assert meta["checks"]["pre_arrival_residual_rel"] is None
    # Last trace drifts by DRIFT * (N - 1) in amplitude; the envelope peak is >= |peak|.
    largest_drift = BACKGROUND_LINE_DRIFT * (N_TRACES - 1)
    assert (
        0.5 * largest_drift < meta["checks"]["line_residual_max_rel"] <= largest_drift
    )
    assert meta["time_zero_candidates_ns"]["surface_reflection_peak"] is not None


@pytest.mark.parametrize("model_id", ["T_BGL", "T_C1", "T_D060"])
def test_every_line_has_the_harness_axes_and_antenna_height(s1g_export, model_id):
    _, out_dir = s1g_export
    meta = _read_json(out_dir / f"{model_id}.json")
    assert meta["trace_x_m_first"] == pytest.approx(SCAN_START_M, abs=1e-12)
    assert meta["dx_m"] == TRACE_STEP_M
    assert meta["time_of_sample0_ns"] == 0.0
    assert meta["dt_ns"] == pytest.approx(DT_NS, rel=1e-12)
    assert meta["air_gap_m"] == ANTENNA_HEIGHT_M


@pytest.mark.parametrize("model_id", ["T_C1", "T_D060"])
def test_pipe_lines_load_like_the_harness(s1g_export, model_id):
    _, out_dir = s1g_export
    radargram = load_synthetic(
        model_id,
        root=out_dir,
        include_eps=True,
        include_radius=False,
        include_height=True,
    )
    assert radargram.antenna_height_m == ANTENNA_HEIGHT_M
    assert radargram.dx_m == TRACE_STEP_M
    assert radargram.metadata["time_of_sample0_ns"] == 0.0


@pytest.mark.parametrize("include_radius", [False, True])
def test_background_line_loads_like_the_harness(s1g_export, include_radius):
    _, out_dir = s1g_export
    meta = _read_json(out_dir / "T_BGL.json")
    radargram = load_synthetic(
        "T_BGL",
        root=out_dir,
        include_eps=True,
        include_radius=include_radius,
        include_height=True,
    )
    assert radargram.data.shape == (meta["n_samples"], N_TRACES)
    assert radargram.eps_r_meta == meta["eps_r"]
    assert radargram.antenna_height_m == ANTENNA_HEIGHT_M
    assert radargram.pipe_radius_m is None
    assert radargram.metadata["trace_x_m_first"] == meta["trace_x_m_first"]


def test_background_line_has_no_pipe_truth(s1g_export):
    _, out_dir = s1g_export
    assert load_synthetic_pipe_truths("T_BGL", root=out_dir) == []
    with pytest.raises(ValueError, match="background line"):
        load_synthetic_truth("T_BGL", root=out_dir)


def test_pipe_truths_list_every_pipe(s1g_export):
    _, out_dir = s1g_export
    two = load_synthetic_pipe_truths("T_D060", root=out_dir)
    assert [t.pipe_center_x_m for t in two] == [0.8, 1.4]
    assert [t.apex_trace_index for t in two] == [10, 25]
    assert all(t.pipe_top_depth_m == 0.3 for t in two)
    assert two[0] == load_synthetic_truth("T_D060", root=out_dir)
    one = load_synthetic_pipe_truths("T_C1", root=out_dir)
    assert one == [load_synthetic_truth("T_C1", root=out_dir)]
    assert one[0].apex_trace_index == 15


def test_s1g_manifest_joins_per_pipe_values(s1g_export):
    _, out_dir = s1g_export
    with (out_dir / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["model_id"]: row for row in csv.DictReader(handle)}
    assert set(rows) == {"T_D060", "T_C1", "T_BGL"}
    assert rows["T_BGL"]["n_pipes"] == "0"
    assert rows["T_BGL"]["pipe_center_x_m"] == ""
    assert rows["T_D060"]["n_pipes"] == "2"
    assert rows["T_D060"]["pipe_center_x_m"] == "0.8;1.4"
    assert len(rows["T_D060"]["apex_envelope_peak_ns"].split(";")) == 2
    assert rows["T_C1"]["pipe_center_x_m"] == "1.0"


def test_apex_trace_mismatch_with_the_spec_is_an_error(exporter, tmp_path):
    h5_dir = _write_fake_runs(exporter, tmp_path)
    spec, sources = exporter.load_s1g_specs(_models_json(tmp_path), h5_dir)[0]
    spec["expected_apex_trace_index"] = [11, 25]
    with pytest.raises(ValueError, match="apex traces"):
        exporter.export_model(spec, sources, tmp_path / "out")


def test_spec_without_pipe_list_keeps_the_single_pipe_schema(exporter, tmp_path):
    h5_dir = _write_fake_runs(exporter, tmp_path)
    spec, sources = exporter.load_s1g_specs(_models_json(tmp_path), h5_dir)[1]
    del spec["pipes"], spec["expected_apex_trace_index"], spec["gprmax_input"]
    meta = exporter.export_model(spec, sources, tmp_path / "out")
    assert list(meta) == SINGLE_PIPE_KEYS
    assert meta["files"]["gprmax_input"] == "scripts/synthetic/inputs/T_C1.in"
    exporter.write_manifest(tmp_path / "out", [meta])
    header = (
        (tmp_path / "out" / "manifest.csv").read_text(encoding="utf-8").splitlines()[0]
    )
    assert header.split(",") == exporter.MANIFEST_FIELDS


# ----------------------------------------------------------------- S4/S6 (M2-10 (b))

S4S6_PAVING = {"brick": {"eps": 5.0}, "joint_x_m": [0.2, 0.42]}
S4S6_STONES = {"x_and_centre_depth_m": [[0.8, 0.15]]}


def _s4s6_model(model_id, clutter, has_pipe):
    model = {
        "id": model_id,
        "family": model_id[:2],
        "batch": 1,
        "role": "test",
        "input": f"in/{model_id}.in",
        "soil_sigma_S_per_m": 0.01,
        "clutter": clutter,
        "n_traces": N_TRACES,
        "scan_start_m": SCAN_START_M,
        "trace_step_m": TRACE_STEP_M,
        "output": f"h5/{model_id}_merged.h5",
        "pipe_material": "pec" if has_pipe else None,
    }
    if has_pipe:
        model.update(
            pipe_radius_m=0.05,
            pipe_top_depth_m=0.3,
            pipe_center_x_m=1.0,
            apex_trace_index=15,
            expected_apex_envelope_peak_ns_approx=9.0,
            two_way_vertical_ns_surface_to_top=6.0,
        )
    return model


def _s4s6_models_json(tmp_path: Path) -> Path:
    common = {
        "eps_r_soil": 9.0,
        "center_freq_hz": FREQ_HZ,
        "air_gap_m": 0.1,
        "antenna_offset_m": 0.1,
        "fdtd_cell_m": 0.005,
        "pml_cells": 20,
        "y_surface_m": 0.7,
        "y_antenna_m": 0.8,
        "time_window_ns": 28.0,
    }
    models = [
        _s4s6_model("S4N_D03_S010", "none", True),
        _s4s6_model("S6P_D03_S010", "paving", True),
        _s4s6_model("S6P_NP_S010", "paving", False),
        _s4s6_model("S4S_D03_S010", "stones", True),
    ]
    references = [
        {"id": "BG_N_S010", "n_traces": 3},
        {"id": "BG_PF_S010", "n_traces": 3},
        {"id": "FS", "n_traces": 1},
    ]
    meta = {
        "common": common,
        "paving": S4S6_PAVING,
        "stones": S4S6_STONES,
        "models": models,
        "references": references,
    }
    path = tmp_path / "models_s4s6.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


def _write_fake_s4s6_runs(exporter: ModuleType, tmp_path: Path) -> Path:
    h5_dir = tmp_path / "h5"
    h5_dir.mkdir()
    t_ns = np.arange(N_SAMPLES) * DT_NS
    source_peak_ns = math.sqrt(2.0) / (FREQ_HZ * 1e-9)
    direct = _ricker(t_ns, source_peak_ns + exporter.direct_air_time(0.1))
    surface = 0.5 * _ricker(
        t_ns, source_peak_ns + exporter.surface_reflection_time(0.1, 0.1)
    )
    background = direct + surface
    for reference in ("BG_N_S010", "BG_PF_S010"):
        _write_h5(
            h5_dir / f"{reference}_merged.h5", np.tile(background[:, None], (1, 3))
        )
    _write_h5(h5_dir / "FS.h5", direct[:, None])
    x_mid = SCAN_START_M + np.arange(N_TRACES) * TRACE_STEP_M
    lateral_gain = 1.0 + BACKGROUND_LINE_DRIFT * np.arange(N_TRACES)
    ray_ns = exporter.trace_reflection_times(
        x_mid, 0.1, 0.1, 1.0, 0.3, 0.05, VELOCITY_M_PER_NS
    )
    for model_id in ("S4N_D03_S010", "S6P_D03_S010", "S6P_NP_S010", "S4S_D03_S010"):
        bscan = background[:, None] * lateral_gain[None, :]
        if model_id != "S6P_NP_S010":
            for i, ray in enumerate(ray_ns):
                bscan[:, i] += _ricker(t_ns, source_peak_ns + ray)
        _write_h5(h5_dir / f"{model_id}_merged.h5", bscan)
    return h5_dir


@pytest.fixture
def s4s6_export(exporter: ModuleType, tmp_path: Path) -> tuple[ModuleType, Path]:
    h5_dir = _write_fake_s4s6_runs(exporter, tmp_path)
    out_dir = tmp_path / "out"
    argv = [
        "--s4s6", str(_s4s6_models_json(tmp_path)),
        "--h5-dir", str(h5_dir),
        "--out-dir", str(out_dir),
    ]  # fmt: skip
    exporter.main(argv)
    return exporter, out_dir


def test_s4s6_backgrounds_follow_the_clutter(exporter, tmp_path):
    h5_dir = _write_fake_s4s6_runs(exporter, tmp_path)
    jobs = exporter.load_s4s6_specs(_s4s6_models_json(tmp_path), h5_dir)
    backgrounds = {spec["model_id"]: src.background_h5.name for spec, src in jobs}
    assert backgrounds == {
        "S4N_D03_S010": "BG_N_S010_merged.h5",
        "S6P_D03_S010": "BG_PF_S010_merged.h5",
        "S6P_NP_S010": "BG_PF_S010_merged.h5",
        "S4S_D03_S010": "BG_N_S010_merged.h5",
    }
    assert all(src.free_space_h5 == h5_dir / "FS.h5" for _, src in jobs)
    with pytest.raises(ValueError, match="soil tag"):
        exporter.s4s6_background_id(
            {"id": "S4N_D15_S030", "clutter": "none", "soil_sigma_S_per_m": 0.01}
        )


def test_s4s6_v2_ids_and_per_model_joints(exporter):
    model = {
        "id": "S6P_D14_S040_J2",
        "clutter": "paving",
        "soil_sigma_S_per_m": 0.04,
        "family": "S6",
        "batch": "v2",
        "role": "test",
        "pair": "P0",
        "joint_rng_seed": 20261112,
        "joint_x_m": [0.21, 0.4],
    }
    assert exporter.s4s6_background_id(model) == "BG_PF_S040"
    extra = exporter._s4s6_extra_meta({"paving": S4S6_PAVING}, model, "BG_PF_S040")
    s4s6 = extra["s4s6"]
    assert (s4s6["pair"], s4s6["joint_rng_seed"]) == ("P0", 20261112)
    assert s4s6["joint_x_m"] == [0.21, 0.4]
    per_model_keys = ("pair", "joint_rng_seed", "joint_x_m")
    batch1 = {key: value for key, value in model.items() if key not in per_model_keys}
    batch1["id"] = "S6P_D15_S040"
    assert "pair" not in exporter._s4s6_extra_meta({"paving": {}}, batch1, "x")["s4s6"]


def test_s4s6_reference_output_paths_are_used(exporter, tmp_path):
    meta_path = _s4s6_models_json(tmp_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["references"][2]["output"] = "elsewhere/FS.h5"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    jobs = exporter.load_s4s6_specs(meta_path)
    assert jobs[0][1].background_h5.name == "BG_N_S010_merged.h5"
    assert jobs[0][1].free_space_h5 is None  # elsewhere/FS.h5 does not exist
    _write_fake_s4s6_runs(exporter, tmp_path)
    with_dir = exporter.load_s4s6_specs(meta_path, tmp_path / "h5")
    assert with_dir[0][1].free_space_h5 == tmp_path / "h5" / "FS.h5"


def test_s4s6_batch_v3_exports_only_its_models(exporter, tmp_path):
    meta_path = _s4s6_models_json(tmp_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for model in meta["models"]:
        if model["id"] in ("S4N_D03_S010", "S6P_NP_S010"):
            model["batch"] = "v3"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    h5_dir = _write_fake_s4s6_runs(exporter, tmp_path)
    out_dir = tmp_path / "out_v3"
    argv = [
        "--s4s6", str(meta_path), "--batch", "v3",
        "--h5-dir", str(h5_dir), "--out-dir", str(out_dir),
    ]  # fmt: skip
    exporter.main(argv)
    exported = sorted(p.stem for p in out_dir.glob("*.json"))
    assert exported == ["S4N_D03_S010", "S6P_NP_S010"]
    assert _read_json(out_dir / "S4N_D03_S010.json")["s4s6"]["batch"] == "v3"


def test_s4s6_batch_folders_never_fall_back_to_batch_1(exporter):
    project = exporter.PROJECT
    assert exporter.s4s6_out_dir("v3", None) == (
        project / "data" / "synthetic_gprmax_s4s6_v3"
    )
    assert exporter.s4s6_out_dir("v2", None) == (
        project / "data" / "synthetic_gprmax_s4s6_v2"
    )
    assert exporter.s4s6_out_dir(None, None) == exporter.S4S6_OUT
    with pytest.raises(SystemExit, match="no default folder"):
        exporter.s4s6_out_dir("v4", None)
    with pytest.raises(SystemExit, match="--batch needs --s4s6"):
        exporter.main(["--batch", "v3"])


def test_s4s6_pipe_line_json_has_the_harness_keys(s4s6_export):
    _, out_dir = s4s6_export
    meta = _read_json(out_dir / "S6P_D03_S010.json")
    assert list(meta)[: len(SINGLE_PIPE_KEYS)] == SINGLE_PIPE_KEYS
    assert meta["n_pipes"] == 1 and meta["pipes"][0]["apex_trace_index"] == 15
    assert meta["apex_trace_index"] == 15
    assert meta["expected_apex_envelope_peak_ns_approx"] == 9.0
    assert meta["air_gap_m"] == ANTENNA_HEIGHT_M
    extra = meta["s4s6"]
    assert extra["clutter"] == "paving" and extra["clutter_meta"] == S4S6_PAVING
    assert extra["background_reference"] == "BG_PF_S010"
    assert meta["velocity_m_per_ns"] == pytest.approx(VELOCITY_M_PER_NS, rel=1e-12)
    stones = _read_json(out_dir / "S4S_D03_S010.json")["s4s6"]
    assert stones["clutter_meta"] == S4S6_STONES
    assert _read_json(out_dir / "S4N_D03_S010.json")["s4s6"]["clutter_meta"] is None
    radargram = load_synthetic(
        "S6P_D03_S010",
        root=out_dir,
        include_eps=True,
        include_radius=False,
        include_height=True,
    )
    assert radargram.antenna_height_m == ANTENNA_HEIGHT_M


def test_s4s6_no_pipe_line_is_a_background_line(s4s6_export):
    _, out_dir = s4s6_export
    meta = _read_json(out_dir / "S6P_NP_S010.json")
    assert meta["n_pipes"] == 0 and meta["pipes"] == []
    assert meta["apex"] is None and meta["apex_trace_index"] is None
    assert meta["expected_apex_envelope_peak_ns_approx"] is None
    assert load_synthetic_pipe_truths("S6P_NP_S010", root=out_dir) == []
    with (out_dir / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["model_id"]: row for row in csv.DictReader(handle)}
    assert rows["S6P_NP_S010"]["n_pipes"] == "0"
    assert rows["S4N_D03_S010"]["n_pipes"] == "1"


def _assert_same_json(actual, expected, path="$"):
    """Same keys in the same order and same values; floats within 1e-9 relative.

    The stored files were written in env gprmax-acceleration; numpy/scipy builds of
    other envs can change the last printed digit of a float, so exact bytes are only
    expected there (checked by the before/after export run, not by this test).
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert list(actual) == list(expected), path
        for key in expected:
            _assert_same_json(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_same_json(item, expected_item, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), path
    else:
        assert actual == expected, path


@pytest.mark.synthetic
def test_single_pipe_export_matches_the_stored_files(exporter, tmp_path):
    stored_json = STORED_EXPORT / f"{REGRESSION_MODEL}.json"
    raw_dir = STORED_EXPORT / "_raw" / REGRESSION_MODEL
    if not (stored_json.exists() and raw_dir.exists()):
        pytest.skip(f"{STORED_EXPORT} with _raw/{REGRESSION_MODEL} is not available")
    specs = json.loads(
        (PROJECT / "scripts" / "synthetic" / "model_specs.json").read_text(
            encoding="utf-8"
        )
    )
    spec = next(s for s in specs if s["model_id"] == REGRESSION_MODEL)
    exporter.export_model(
        spec, exporter.legacy_source_files(spec, STORED_EXPORT / "_raw"), tmp_path
    )
    for name in (
        f"{REGRESSION_MODEL}.npy",
        f"{REGRESSION_MODEL}_bg.npy",
        f"{REGRESSION_MODEL}_fs.npy",
    ):
        assert (tmp_path / name).read_bytes() == (STORED_EXPORT / name).read_bytes(), (
            name
        )
    _assert_same_json(_read_json(tmp_path / stored_json.name), _read_json(stored_json))


# ----------------------------------------------------------------- K7 (M2-10 K7)


def _k7_model(model_id: str, pipe_xs: tuple[float, ...], material: str = "pec"):
    pipes = [
        {
            "pipe_index": i,
            "x_m": x,
            "top_depth_m": 0.3,
            "radius_m": 0.05,
            "material": material,
            "wall_m": None,
            "fill": None,
            "apex_trace_index": round((x - SCAN_START_M) / TRACE_STEP_M),
        }
        for i, x in enumerate(pipe_xs)
    ]
    return {
        "id": model_id,
        "role": "test",
        "target_mechanism": "a",
        "function": "defect",
        "batch": "k7",
        "roles": ["defect_a"],
        "soil": {"eps_r": 9.0, "sigma_S_per_m": 0.001},
        "layer": None,
        "pipes": pipes,
        "input": f"in/{model_id}.in",
        "n_traces": N_TRACES,
        "scan_start_m": SCAN_START_M,
        "trace_step_m": TRACE_STEP_M,
        "output": f"h5/{model_id}_merged.h5",
        "bg_ref": "K7_REF_T",
        "fs_ref": "FS",
    }


def _k7_models_json(tmp_path: Path, fs_path: Path, models=None) -> Path:
    common = {
        "eps_r_soil": 9.0,
        "center_freq_hz": FREQ_HZ,
        "air_gap_m": 0.1,
        "antenna_offset_m": 0.1,
        "fdtd_cell_m": 0.005,
        "pml_cells": 20,
        "y_surface_m": 0.7,
        "y_antenna_m": 0.8,
        "time_window_ns": 28.0,
    }
    references = [
        {"id": "K7_REF_T", "n_traces": 3, "output": "h5/K7_REF_T_merged.h5"},
        # Absolute: a reference reused from another folder (as the S4/S6 FS).
        {"id": "FS", "n_traces": 1, "output": str(fs_path)},
    ]
    models = models or [_k7_model("K7_T1", (0.8, 1.4)), _k7_model("K7_TBG", ())]
    meta = {"common": common, "models": models, "references": references}
    path = tmp_path / "models_k7.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


def _write_fake_k7_runs(exporter: ModuleType, tmp_path: Path) -> tuple[Path, Path]:
    h5_dir = tmp_path / "h5"
    h5_dir.mkdir()
    fs_dir = tmp_path / "reused"
    fs_dir.mkdir()
    t_ns = np.arange(N_SAMPLES) * DT_NS
    source_peak_ns = math.sqrt(2.0) / (FREQ_HZ * 1e-9)
    direct = _ricker(t_ns, source_peak_ns + exporter.direct_air_time(0.1))
    surface = 0.5 * _ricker(
        t_ns, source_peak_ns + exporter.surface_reflection_time(0.1, 0.1)
    )
    background = direct + surface
    _write_h5(h5_dir / "K7_REF_T_merged.h5", np.tile(background[:, None], (1, 3)))
    _write_h5(fs_dir / "FS.h5", direct[:, None])
    x_mid = SCAN_START_M + np.arange(N_TRACES) * TRACE_STEP_M
    lateral_gain = 1.0 + BACKGROUND_LINE_DRIFT * np.arange(N_TRACES)
    for model_id, pipe_xs in (("K7_T1", (0.8, 1.4)), ("K7_TBG", ())):
        bscan = background[:, None] * lateral_gain[None, :]
        for x0 in pipe_xs:
            ray_ns = exporter.trace_reflection_times(
                x_mid, 0.1, 0.1, x0, 0.3, 0.05, VELOCITY_M_PER_NS
            )
            for i, ray in enumerate(ray_ns):
                bscan[:, i] += _ricker(t_ns, source_peak_ns + ray)
        _write_h5(h5_dir / f"{model_id}_merged.h5", bscan)
    return h5_dir, fs_dir / "FS.h5"


def test_k7_export_writes_the_s1g_schema_with_k7_keys(exporter, tmp_path):
    h5_dir, fs_path = _write_fake_k7_runs(exporter, tmp_path)
    models_json = _k7_models_json(tmp_path, fs_path)
    out_dir = tmp_path / "out"
    exporter.main(
        ["--k7", str(models_json), "--h5-dir", str(h5_dir), "--out-dir", str(out_dir)]
    )
    meta = _read_json(out_dir / "K7_T1.json")
    assert list(meta)[: len(SINGLE_PIPE_KEYS)] == SINGLE_PIPE_KEYS
    assert meta["n_pipes"] == 2
    assert [p["apex_trace_index"] for p in meta["pipes"]] == [10, 25]
    assert meta["k7"]["background_reference"] == "K7_REF_T"
    assert meta["k7"]["batch"] == "k7"
    truths = load_synthetic_pipe_truths("K7_T1", root=out_dir)
    assert [t.apex_trace_index for t in truths] == [10, 25]
    background = _read_json(out_dir / "K7_TBG.json")
    assert background["n_pipes"] == 0 and background["pipe_material"] is None
    assert load_synthetic_pipe_truths("K7_TBG", root=out_dir) == []


def test_k7_specs_read_the_references(exporter, tmp_path):
    h5_dir, fs_path = _write_fake_k7_runs(exporter, tmp_path)
    jobs = exporter.load_k7_specs(_k7_models_json(tmp_path, fs_path), h5_dir)
    for _, sources in jobs:
        assert sources.background_h5 == h5_dir / "K7_REF_T_merged.h5"
        assert sources.free_space_h5 == fs_path
    assert exporter.K7_OUT == exporter.PROJECT / "data" / "synthetic_gprmax_k7"


def test_k7_errors(exporter, tmp_path):
    h5_dir, fs_path = _write_fake_k7_runs(exporter, tmp_path)
    missing_fs = _k7_models_json(tmp_path, tmp_path / "absent" / "FS.h5")
    with pytest.raises(ValueError, match="free-space reference"):
        exporter.load_k7_specs(missing_fs, h5_dir)
    mixed = _k7_model("K7_MIX", (0.8, 1.4))
    mixed["pipes"][1]["material"] = "pvc_air"
    mixed_json = _k7_models_json(tmp_path, fs_path, [mixed])
    with pytest.raises(ValueError, match="differ in material"):
        exporter.load_k7_specs(mixed_json, h5_dir)
    with pytest.raises(SystemExit, match="exclusive"):
        exporter.main(["--k7", "a.json", "--s1g", "b.json"])


def test_k7_em_json_specs_use_the_shared_references(exporter):
    """Dry run on the committed models_k7.json (no h5 is read)."""
    models_json = PROJECT / "analysis" / "synthetic" / "k7" / "models_k7.json"
    fs_path = PROJECT / "data" / "synthetic_gprmax_s4s6" / "h5" / "FS.h5"
    if not (models_json.exists() and fs_path.exists()):
        pytest.skip("models_k7.json or the reused S4/S6 FS.h5 not available")
    jobs = exporter.load_k7_specs(models_json)
    assert len(jobs) == 8
    backgrounds = {spec["model_id"]: src.background_h5.name for spec, src in jobs}
    assert backgrounds["K7_BG_L"] == backgrounds["K7_P2_D070_L"]
    assert backgrounds["K7_P2_D070"] == "K7_REF_S001_merged.h5"
    assert all(src.free_space_h5 == fs_path for _, src in jobs)
    for spec, _ in jobs:
        for pipe in spec["extra_meta"]["k7"]["pipes_json"]:
            offset = (pipe["x_m"] - spec["scan_start_m"]) / spec["trace_step_m"]
            assert round(offset) == pipe["apex_trace_index"]
