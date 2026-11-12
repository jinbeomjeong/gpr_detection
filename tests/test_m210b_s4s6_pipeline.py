"""S4/S6 harness (m210b_selector_floor.md 4-2, 8-3): resampling, the seed formula,
the lost-reason table and fake lines end to end. No gprMax output is needed."""

import csv
import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from synthetic_bscan import T_SRC_NS, BScanSpec, cylinder_times, make_radargram

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = PROJECT_ROOT / "analysis" / "processing" / "m210b_s4s6_pipeline.py"
F_C_HZ = 500e6
GPRMAX_DT_NS = 0.011793271683748419
FAKE_DT_NS = 0.02
X_FIRST_M = 0.40
APEX_TRACE = 80
T_A0_NS = 20.0
EXPECTED_RUN_COLUMNS = [
    "model_id", "rho", "seed", "rule", "delta_ns", "ramp_noise", "dt_ns",
    "t_zero_ns", "zero_rule", "L", "F", "sigma_n", "rho_real", "Lambda",
    "argmax_level_periods", "accepted_apex", "peak_on_crest", "kept_apex",
    "kept_apex_trace", "kept_apex_sample", "lost_reason", "floor_held_at_apex",
    "n_peaks", "cap300_hit", "n_false_deep", "n_false_near_surface",
    "n_false_record_end", "rho_real_cal",
]  # fmt: skip
"""8-3 columns; ``rho_real_cal`` appended at the end (user decision D1)."""
V2_EXTRA_COLUMNS = [
    "kept_nearest_trace",
    "kept_nearest_sample",
    "kept_nearest_t_ns",
    "n_kept_in_window",
    "depth_nearest_m",
    "depth_nearest_status",
]
"""v2 H4 and H6 columns after the batch-1 columns (``runs_v2.csv``)."""


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("m210b_s4s6_pipeline", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["m210b_s4s6_pipeline"] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------- resampling


def _sine_line(dt_ns: float, f_hz: float, n_samples: int) -> object:
    t_ns = np.arange(n_samples) * dt_ns
    column = np.sin(2 * np.pi * f_hz * 1e-9 * t_ns)
    spec = BScanSpec(f_c_hz=F_C_HZ, dt_ns=dt_ns, n_traces=3, n_samples=n_samples)
    radargram = make_radargram(spec)
    data = np.tile(column[:, None], (1, 3)).astype(np.float32)
    return dataclasses.replace(radargram, data=data, metadata={"time_of_sample0_ns": 0})


def test_resample_grid_follows_the_protocol(harness) -> None:
    radargram = _sine_line(GPRMAX_DT_NS, 300e6, 4665)
    resampled = harness.resample_to(radargram, 0.098)
    t_end = (4665 - 1) * GPRMAX_DT_NS
    assert resampled.dt_ns == 0.098
    assert resampled.n_samples == math.floor(t_end / 0.098) + 1
    assert resampled.data.dtype == np.float32
    assert resampled.n_traces == 3


@pytest.mark.parametrize(
    ("f_hz", "max_error"),
    [
        (300e6, 1e-4),
        # Linear interpolation error bound (omega dt)^2 / 8 = 1.7e-4 at 500 MHz:
        # 4-2's "about 4e-5" does not hold as a maximum at the centre frequency.
        (500e6, (2 * np.pi * 500e6 * 1e-9 * GPRMAX_DT_NS) ** 2 / 8 * 1.05),
    ],
)
def test_resample_sine_error(harness, f_hz: float, max_error: float) -> None:
    radargram = _sine_line(GPRMAX_DT_NS, f_hz, 4665)
    resampled = harness.resample_to(radargram, 0.098)
    t_new = np.arange(resampled.n_samples) * 0.098
    exact = np.sin(2 * np.pi * f_hz * 1e-9 * t_new)
    # float32 storage adds up to ~6e-8.
    assert np.max(np.abs(resampled.data[:, 0] - exact)) < max_error


# ----------------------------------------------------------------- seeds


def test_seed_formula_and_shared_pairs(harness) -> None:
    assert harness.noise_seed(0, 0, 0) == 20261120
    assert harness.noise_seed(4, 1, 9) == 20261120 + 4000 + 100 + 9
    assert harness.PAIR_INDEX == {
        "D12_S010": 0,
        "D15_S010": 1,
        "D15_S030": 2,
        "NP_S010": 3,
        "D15_S040": 4,
    }
    assert harness.model_pair("S6P_D15_S030") == harness.model_pair("S4N_D15_S030")
    assert harness.calibration_pair("S6P_NP_S010") == "D12_S010"
    assert harness.calibration_pair("S6P_D15_S040") == "D15_S040"
    first = harness.white_noise((4, 3), harness.noise_seed(1, 0, 2))
    again = harness.white_noise((4, 3), harness.noise_seed(1, 0, 2))
    np.testing.assert_array_equal(first, again)


def test_ramp_profile_runs_from_0_3_to_1_5(harness) -> None:
    ramp = harness.ramp_profile(561, 0.098)
    assert ramp.shape == (561, 1)
    assert ramp[0, 0] == pytest.approx(0.3)
    assert ramp[-1, 0] == pytest.approx(1.5)


# ----------------------------------------------------------------- realised rho


def _clean_with_noise(harness, a0_self):
    """A noise-free zero section and a unit-noise copy (same noise for every call)."""
    spec = BScanSpec(f_c_hz=F_C_HZ, dt_ns=0.098, n_traces=41, n_samples=300)
    radargram = make_radargram(spec)
    clean_removed = np.zeros((300, 41), dtype=np.float32)
    noise = harness.white_noise((300, 41), 20261120).astype(np.float32)
    clean = harness.CleanLine(
        radargram=radargram,
        background_removed=clean_removed,
        t_zero_ns=3.0,
        zero_rule="air_single_event",
        target=None,
        a0_self=a0_self,
        t_early_ns=None,
        level_max_by_delta={},
    )
    return clean, clean_removed + noise


def test_rho_real_uses_the_models_own_apex(harness) -> None:
    a0_cal = 0.25
    clean, noisy_removed = _clean_with_noise(harness, a0_self=2 * a0_cal)
    rho = harness.realised_rho(clean, noisy_removed, a0_cal)
    assert rho.own == pytest.approx(2 * rho.calibration, rel=1e-12)
    assert rho.calibration == pytest.approx(a0_cal / 1.0, rel=0.1)  # unit noise


def test_rho_real_is_blank_without_a_pipe(harness) -> None:
    clean, noisy_removed = _clean_with_noise(harness, a0_self=None)
    rho = harness.realised_rho(clean, noisy_removed, 0.25)
    assert rho.own is None and rho.calibration > 0
    assert harness._blank_or_rounded(rho.own, 4) == ""


# ----------------------------------------------------------------- lost reason


def _row(trace: int, t_ns: float) -> dict:
    return {"trace": trace, "t_ns": t_ns, "sample": t_ns / 0.098, "tau_ns": t_ns - 3.0}


def _fate(fate: str, winner: int | None) -> dict:
    return {"fate": fate, "winner": winner}


def _line(accepted, fates, peaks) -> dict:
    return {"_accepted_all": accepted, "_accepted_fate": fates, "_peaks": peaks}


def _peak(trace: int, t_ns: float) -> dict:
    return {"trace": trace, "sample": t_ns / 0.098}


@pytest.fixture
def target(harness):
    return harness.ApexTarget(APEX_TRACE, T_A0_NS, F_C_HZ)


@pytest.mark.parametrize(
    ("winner_t_ns", "winner_trace", "reason"),
    [
        (T_A0_NS - 3.0, APEX_TRACE, "early_lobe_nms"),  # t_a0 - 1.5/f_c, included
        (T_A0_NS - 2.0, APEX_TRACE + 3, "early_lobe_nms"),
        (T_A0_NS - 1.0 - 1e-6, APEX_TRACE, "early_lobe_nms"),
        (T_A0_NS - 3.01, APEX_TRACE, "other_nms"),
        (T_A0_NS - 2.0, APEX_TRACE + 4, "other_nms"),
    ],
)
def test_nms_reasons_split_at_the_early_lobe_box(
    harness, target, winner_t_ns, winner_trace, reason
) -> None:
    winner = _row(winner_trace, winner_t_ns)
    apex = _row(APEX_TRACE, T_A0_NS)
    line = _line([winner, apex], [_fate("kept", 0), _fate("nms", 0)], [])
    outcome = harness.apex_outcome(line, [winner], 0.098, target)
    assert (outcome.accepted_apex, outcome.kept_apex) == (True, False)
    assert outcome.lost_reason == reason


def test_window_boundaries(harness) -> None:
    box, window = harness.is_in_early_lobe_box, harness.is_in_apex_window
    # [t_a0 - 1.5/f_c, t_a0 - 0.5/f_c): the upper end belongs to the apex window.
    assert box(APEX_TRACE, T_A0_NS - 3.0, APEX_TRACE, T_A0_NS, F_C_HZ)
    assert not box(APEX_TRACE, T_A0_NS - 1.0, APEX_TRACE, T_A0_NS, F_C_HZ)
    assert window(APEX_TRACE, T_A0_NS - 1.0, APEX_TRACE, T_A0_NS, F_C_HZ)
    assert window(APEX_TRACE - 3, T_A0_NS + 1.0, APEX_TRACE, T_A0_NS, F_C_HZ)
    assert not window(APEX_TRACE - 4, T_A0_NS, APEX_TRACE, T_A0_NS, F_C_HZ)
    assert not box(APEX_TRACE + 4, T_A0_NS - 2.0, APEX_TRACE, T_A0_NS, F_C_HZ)


def test_cap6_kept_not_accepted_and_no_peak(harness, target) -> None:
    apex = _row(APEX_TRACE, T_A0_NS)
    others = [_row(10 * i, 10.0) for i in range(6)]
    capped = _line(
        [*others, apex],
        [_fate("kept", i) for i in range(6)] + [_fate("cap6", None)],
        [],
    )
    assert harness.apex_outcome(capped, others, 0.098, target).lost_reason == "cap6"

    edge = _row(APEX_TRACE + 3, T_A0_NS + 1.0)  # window boundaries included
    kept = harness.apex_outcome(
        _line([edge], [_fate("kept", 0)], []), [edge], 0.098, target
    )
    assert (kept.kept_apex, kept.lost_reason, kept.kept_trace) == (
        True,
        "",
        APEX_TRACE + 3,
    )

    on_crest = _line([], [], [_peak(APEX_TRACE + 1, T_A0_NS - 0.5)])
    rejected = harness.apex_outcome(on_crest, [], 0.098, target)
    assert (rejected.peak_on_crest, rejected.lost_reason) == (True, "not_accepted")

    far = _line([], [], [_peak(APEX_TRACE + 4, T_A0_NS)])
    assert harness.apex_outcome(far, [], 0.098, target).lost_reason == "no_peak"


def test_false_counts_split_by_depth_and_record_end(harness, target) -> None:
    velocity = 0.1
    kept = [
        _row(APEX_TRACE, T_A0_NS),  # the apex itself: not counted
        _row(20, 3.0 + 2 * 0.30 / velocity),  # 0.30 m: deep
        _row(30, 3.0 + 2 * 0.29 / velocity),  # near surface
        _row(40, 50.0),  # within 5 ns of t_end 55
    ]
    counts = harness.false_counts(kept, target, velocity, 55.0)
    assert (counts.deep, counts.near_surface, counts.record_end) == (1, 1, 1)
    assert harness.false_counts(kept, None, velocity, 55.0).deep == 2


# ----------------------------------------------------------------- end to end


def _fake_model(model_id: str, has_pipe: bool, has_clutter: bool):
    """Air-coupled line on a fine grid: optional pipe at 0.9 m, optional clutter."""
    spec = BScanSpec(
        f_c_hz=F_C_HZ,
        dt_ns=FAKE_DT_NS,
        dx_m=0.02,
        n_traces=161,
        n_samples=1400,
        snr_db=None,
        air_gap_m=0.001,
        offset_m=0.0,
        direct_amplitude=30.0,
        surface_amplitude=-20.0,
        pipe_amplitude=1.0 if has_pipe else 0.0,
        eps_r=9.0,
    )
    spec.pipe_times_ns = cylinder_times(spec.x_m, spec.x0_m, 0.9, spec.velocity, 0.0)
    if has_clutter:
        lateral = 3.0 * np.sin(2 * np.pi * spec.x_m / 0.2)
        spec.extra_events = [(np.full(spec.n_traces, 1.8), lateral)]
    radargram = make_radargram(spec, line_id=f"S4S6/{model_id}")
    radargram = dataclasses.replace(
        radargram, metadata={"trace_x_m_first": X_FIRST_M, "time_of_sample0_ns": 0.0}
    )
    t_apex = T_SRC_NS + float(spec.pipe_times_ns[APEX_TRACE])
    meta = {
        "apex_trace_index": APEX_TRACE if has_pipe else None,
        "expected_apex_envelope_peak_ns_approx": t_apex if has_pipe else None,
        "air_gap_m": 0.001,
    }
    return radargram, meta


FAKE_MODELS = {
    "S4N_D12_S010": (True, False),
    "S6P_D12_S010": (True, True),
    "S6P_NP_S010": (False, True),
}
FAKE_V2_MODELS = {
    "S4N_D14_S040": (True, False),
    "S6P_D14_S040_J2": (True, True),
    "S6P_NP_S040_J2": (False, True),
}
SMALL_PLAN = {
    "n_seeds": 1,
    "rules": ("current", "B_full"),
    "deltas_ns": (0.0, 1.0),
    "original_dt_rules": ("current",),
}


def _clean_seal(harness, seal=None, dirty=()):
    """A seal record with every sealed file matching (git is not consulted)."""
    commit = "bc9a474" if seal is None else seal.commit
    return {
        "seal_commit": commit,
        "seal_commit_full": commit,
        "seal_commit_time": "2026-11-11T19:29:19+09:00",
        "sealed_files": {"docs/x.md": {"blob_at_seal": "a", "matches": True}},
        "head_commit": commit,
        "harness_blob_working_tree": "h",
        "dirty_files": list(dirty),
        "git_failures": [],
        "taken_at": "2026-11-11T21:00:00+09:00",
    }


def _placeholder_inputs(root: Path, model_ids) -> None:
    root.mkdir()
    for model_id in model_ids:
        for suffix in (".npy", ".json"):
            (root / f"{model_id}{suffix}").write_bytes(b"placeholder")


def _caching_loader(models):
    cache = {}

    def loader(model_id, root):
        if model_id not in cache:
            cache[model_id] = _fake_model(model_id, *models[model_id])
        return cache[model_id]

    return loader


def _read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return reader.fieldnames, rows


def test_main_writes_runs_and_run_info(harness, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(harness, "seal_record", lambda seal=None: _clean_seal(harness))
    data_root = tmp_path / "data"
    _placeholder_inputs(data_root, FAKE_MODELS)
    models_json = tmp_path / "models.json"
    models_json.write_text("{}", encoding="utf-8")
    plan = harness.HarnessPlan(**SMALL_PLAN)
    out = tmp_path / "out"
    argv = [
        "--data-root", str(data_root),
        "--models-json", str(models_json),
        "--out-dir", str(out),
        "--models", "S6P_D12_S010,S6P_NP_S010",
        "--regression-activities", "none",
    ]  # fmt: skip
    assert harness.main(argv, plan, _caching_loader(FAKE_MODELS)) == 0

    fieldnames, rows = _read_rows(out / "runs.csv")
    assert fieldnames == EXPECTED_RUN_COLUMNS
    assert list(harness.RUN_COLUMNS) == EXPECTED_RUN_COLUMNS
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    assert set(info["protocol_detail"]["columns"]) >= {"rho_real", "rho_real_cal"}
    noisy_runs = 2 * 1 * 2 * 2  # rho x seed x rule x delta
    ramp_runs = 1 * 2 * 2
    original_runs = 1
    assert (
        len(rows) == noisy_runs + original_runs + noisy_runs + ramp_runs + original_runs
    )
    pipe_rows = [r for r in rows if r["model_id"] == "S6P_D12_S010"]
    assert {r["dt_ns"] for r in pipe_rows} == {"0.098", str(FAKE_DT_NS)}
    reasons = {"", "early_lobe_nms", "cap6", "other_nms", "not_accepted", "no_peak"}
    assert all(r["lost_reason"] in reasons for r in pipe_rows)
    assert all(float(r["rho_real"]) > 0 for r in pipe_rows)
    assert all(float(r["rho_real_cal"]) > 0 for r in pipe_rows)
    assert any(r["kept_apex"] == "True" for r in pipe_rows if r["rule"] == "current")
    no_pipe = [r for r in rows if r["model_id"] == "S6P_NP_S010"]
    assert {r["ramp_noise"] for r in no_pipe} == {"0", "1"}
    assert all(r["accepted_apex"] == "" and r["Lambda"] == "" for r in no_pipe)
    # D1: no own apex -> rho_real blank; the calibration ratio is still written.
    assert all(r["rho_real"] == "" and float(r["rho_real_cal"]) > 0 for r in no_pipe)
    # S6P shares the S4N seeds of its pair; the no-pipe line has its own p = 3.
    s6p_seeds = {int(r["seed"]) for r in pipe_rows if r["dt_ns"] == "0.098"}
    assert s6p_seeds == {harness.noise_seed(0, 0, 0), harness.noise_seed(0, 1, 0)}
    np_seeds = {int(r["seed"]) for r in no_pipe}
    assert np_seeds == {harness.noise_seed(3, 0, 0), harness.noise_seed(3, 1, 0)}

    assert info["seal"]["seal_commit"] == "bc9a474"
    assert info["guards"] == {"pass": True, "failures": []}
    assert info["seed_overlap"] is None  # batch-1 protocol
    assert info["mkl_cbwr"]
    environment = info["environment"]
    assert {"scipy", "skimage", "executable", "thread_variables"} <= set(environment)
    assert environment["interpretation_doc"]["blob_working_tree"]
    assert info["wrapper_regression"] == {"run": False}
    assert set(info["inputs"]) == {
        "models_json",
        *(f"{m}{s}" for m in FAKE_MODELS for s in (".npy", ".json")),
    }
    record = info["models"]["S6P_D12_S010"]["harness"]
    assert record["calibration"]["control_id"] == "S4S6/S4N_D12_S010"
    assert record["t_a0_ns"] is not None and record["A0_self"] > 0
    assert set(record["Lambda"]) == {"+0.00", "+1.00"}
    assert set(record["argmax_level_periods_noise_free"]) == {"+0.00", "+1.00"}
    assert info["models"]["S6P_NP_S010"]["calibration_pair"] == "D12_S010"
    assert (out / "pre_run.txt").read_text(encoding="utf-8").startswith("time ")
    assert "dirty 0" in (out / "post_run.txt").read_text(encoding="utf-8")


def test_sealed_blobs_list_the_v1_seal(harness) -> None:
    sealed = harness.sealed_blobs()
    assert "docs/processing/m210b_selector_floor.md" in sealed
    assert any(path.startswith("analysis/synthetic/s4s6/") for path in sealed)
    assert all(record["blob_at_seal"] for record in sealed.values())


# ----------------------------------------------------------------- guards


@pytest.mark.parametrize(
    ("change", "needle"),
    [
        ({"sealed": False}, "sealed files differ"),
        ({"dirty_start": ["?? x.py"]}, "dirty working tree at start"),
        ({"dirty_end": [" M y.py"]}, "dirty working tree at end"),
        ({"end_head": "other"}, "HEAD changed"),
        ({"regression": False}, "wrapper regression differs"),
        ({"inputs_changed": ["S4N_D14_S040.npy"]}, "inputs changed"),
    ],
)
def test_guards_name_every_failure(harness, change, needle) -> None:
    start = _clean_seal(harness, dirty=change.get("dirty_start", ()))
    end = _clean_seal(harness, dirty=change.get("dirty_end", ()))
    if change.get("sealed") is False:
        start["sealed_files"]["docs/x.md"]["matches"] = False
    if "end_head" in change:
        end["head_commit"] = change["end_head"]
    regression = {
        "run": True,
        "a.csv": {"byte_identical": change.get("regression", True)},
    }
    failures = harness.guard_failures(
        start, end, change.get("inputs_changed", []), regression
    )
    assert any(needle in failure for failure in failures), failures


def test_guards_pass_on_a_clean_run(harness) -> None:
    clean = _clean_seal(harness)
    assert harness.guard_failures(clean, clean, [], {"run": False}) == []


def test_main_returns_1_when_a_guard_fails(harness, tmp_path, monkeypatch) -> None:
    dirty = _clean_seal(harness, dirty=["?? stray.py"])
    monkeypatch.setattr(harness, "seal_record", lambda seal=None: dirty)
    data_root = tmp_path / "data"
    _placeholder_inputs(data_root, ["S4N_D12_S010"])
    models_json = tmp_path / "models.json"
    models_json.write_text("{}", encoding="utf-8")
    plan = harness.HarnessPlan(**{**SMALL_PLAN, "rhos": (10,), "deltas_ns": (0.0,)})
    argv = [
        "--data-root", str(data_root), "--models-json", str(models_json),
        "--out-dir", str(tmp_path / "out"), "--models", "S4N_D12_S010",
        "--regression-activities", "none",
    ]  # fmt: skip
    calls = []

    def loader(model_id, root):
        calls.append(model_id)
        return _fake_model(model_id, *FAKE_MODELS[model_id])

    assert harness.main(argv, plan, loader) == 1
    info = json.loads((tmp_path / "out" / "run_info.json").read_text(encoding="utf-8"))
    assert info["guards"]["pass"] is False and info["guards"]["stage"] == "start"
    # A start guard stops before any model is loaded or run.
    assert calls == [] and not (tmp_path / "out" / "runs.csv").exists()
    assert (tmp_path / "out" / "pre_run.txt").exists()


def _small_v1_argv(tmp_path, data_root, models_json):
    return [
        "--data-root", str(data_root), "--models-json", str(models_json),
        "--out-dir", str(tmp_path / "out"), "--models", "S4N_D12_S010",
        "--regression-activities", "none",
    ]  # fmt: skip


def test_end_guard_failure_keeps_the_outputs(harness, tmp_path, monkeypatch) -> None:
    records = iter([_clean_seal(harness), {**_clean_seal(harness), "head_commit": "x"}])
    monkeypatch.setattr(harness, "seal_record", lambda seal=None: next(records))
    data_root = tmp_path / "data"
    _placeholder_inputs(data_root, ["S4N_D12_S010"])
    models_json = tmp_path / "models.json"
    models_json.write_text("{}", encoding="utf-8")
    plan = harness.HarnessPlan(**{**SMALL_PLAN, "rhos": (10,), "deltas_ns": (0.0,)})
    argv = _small_v1_argv(tmp_path, data_root, models_json)
    assert harness.main(argv, plan, _caching_loader(FAKE_MODELS)) == 1
    info = json.loads((tmp_path / "out" / "run_info.json").read_text(encoding="utf-8"))
    assert any("HEAD changed" in f for f in info["guards"]["failures"])
    assert (tmp_path / "out" / "runs.csv").exists()


def test_inputs_are_hashed_before_the_runs(harness, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(harness, "seal_record", lambda seal=None: _clean_seal(harness))
    data_root = tmp_path / "data"
    _placeholder_inputs(data_root, ["S4N_D12_S010"])
    (data_root / "manifest.csv").write_text("model_id\n", encoding="utf-8")
    models_json = tmp_path / "models.json"
    models_json.write_text("{}", encoding="utf-8")

    def loader(model_id, root):
        # A file changed while the models run must be caught by the end guard.
        (data_root / "manifest.csv").write_text("changed\n", encoding="utf-8")
        return _fake_model(model_id, *FAKE_MODELS[model_id])

    plan = harness.HarnessPlan(**{**SMALL_PLAN, "rhos": (10,), "deltas_ns": (0.0,)})
    argv = _small_v1_argv(tmp_path, data_root, models_json)
    assert harness.main(argv, plan, loader) == 1
    info = json.loads((tmp_path / "out" / "run_info.json").read_text(encoding="utf-8"))
    assert set(info["inputs"]) == {
        "models_json", "S4N_D12_S010.npy", "S4N_D12_S010.json", "manifest.csv",
    }  # fmt: skip
    assert info["inputs_changed_during_run"] == ["manifest.csv"]


def test_failed_git_query_is_not_a_clean_tree(harness, monkeypatch) -> None:
    def git(*args):
        return None if args[0] == "status" else "abc"

    monkeypatch.setattr(harness, "_git", git)
    monkeypatch.setattr(harness, "sealed_blobs", lambda seal: {"a": {"matches": True}})
    record = harness.seal_record()
    assert record["dirty_files"] == [] and record["git_failures"] == ["status"]
    failures = harness.start_guard_failures(record, head_must_be_seal=False)
    assert any("git queries failed" in f for f in failures)


def test_end_sealed_records_are_compared_with_the_start(harness) -> None:
    start = _clean_seal(harness)
    end = _clean_seal(harness)
    end["sealed_files"] = {"docs/x.md": {"blob_at_seal": "b", "matches": True}}
    failures = harness.end_guard_failures(start, end, [], {"run": False})
    assert failures == ["sealed file records changed during the run"]


def test_v2_start_guard_needs_head_at_the_seal(harness) -> None:
    record = _clean_seal(harness, harness.v2_seal("v2seal"))
    assert harness.start_guard_failures(record, head_must_be_seal=True) == []
    record["head_commit"] = "later"
    failures = harness.start_guard_failures(record, head_must_be_seal=True)
    assert len(failures) == 1 and "is not the seal commit" in failures[0]
    assert harness.start_guard_failures(record, head_must_be_seal=False) == []


def test_lambda_column_keeps_4_digits_and_run_info_the_exact_value(
    harness, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(harness, "seal_record", lambda seal=None: _clean_seal(harness))
    data_root = tmp_path / "data"
    _placeholder_inputs(data_root, ["S6P_D12_S010", "S4N_D12_S010"])
    models_json = tmp_path / "models.json"
    models_json.write_text("{}", encoding="utf-8")
    plan = harness.HarnessPlan(**{**SMALL_PLAN, "rhos": (10,), "deltas_ns": (0.0,)})
    argv = _small_v1_argv(tmp_path, data_root, models_json)
    argv[argv.index("S4N_D12_S010")] = "S6P_D12_S010"
    assert harness.main(argv, plan, _caching_loader(FAKE_MODELS)) == 0
    _, rows = _read_rows(tmp_path / "out" / "runs.csv")
    info = json.loads((tmp_path / "out" / "run_info.json").read_text(encoding="utf-8"))
    exact = info["models"]["S6P_D12_S010"]["harness"]["Lambda"]["+0.00"]
    column = {r["Lambda"] for r in rows if r["dt_ns"] == "0.098"}
    assert column == {str(round(exact, 4))}


# ----------------------------------------------------------------- v2 (H2, H4, H8)


def test_v2_seed_formula_and_batch1_disjointness(harness) -> None:
    plan = harness.V2_PLAN
    assert harness.noise_seed(0, 0, 0, plan.seed_base) == 20271120
    assert harness.noise_seed(4, 1, 19, plan.seed_base) == 20271120 + 4000 + 100 + 19
    assert plan.n_seeds == 20 and plan.unit_noise_seed == 20271199
    assert plan.pair_index == {
        "D14_S040": 0, "D16_S040": 1, "D15_S045": 2, "D13_S010": 3, "NP_S040": 4,
    }  # fmt: skip
    assert plan.rules == ("current", "B_full", "B_min", "B_noise", "C_none")
    batch1 = harness.plan_seeds(harness.DEFAULT_PLAN)
    assert len(batch1) == 5 * 2 * 10 + 1 and 20261199 in batch1
    v2_seeds = harness.plan_seeds(plan)
    assert len(v2_seeds) == 5 * 2 * 20 + 1
    assert harness.assert_no_seed_overlap(v2_seeds, batch1) == 0
    with pytest.raises(harness.SeedOverlapError):
        harness.assert_no_seed_overlap(v2_seeds | {20261120}, batch1)


def test_v2_seal_lists_the_three_test_files(harness) -> None:
    sealed = set(harness.V2_SEALED_FILES)
    assert {
        "tests/test_m210b_s4s6_pipeline.py",
        "tests/test_m210b_v2_judge.py",
        "tests/test_synthetic_export.py",
        "analysis/processing/m210b_v2_judge.py",
        "analysis/synthetic/s4s6/models_s4s6_v2.json",
    } <= sealed


def test_module_docstring_names_the_v2_json_and_decisions(harness) -> None:
    doc = harness.__doc__
    assert "analysis/synthetic/s4s6/models_s4s6_v2.json" in doc
    assert "W1-W10" in doc and "s4s6_v2/models" not in doc


def test_v2_model_pairs_strip_the_joint_tag(harness) -> None:
    assert harness.model_pair("S6P_D14_S040_J2") == "D14_S040"
    assert harness.model_pair("S4N_D14_S040") == "D14_S040"
    assert harness.model_pair("S6P_NP_S040_J2") == "NP_S040"
    plan = dataclasses.replace(harness.V2_PLAN, no_pipe_calibration_pair="D14_S040")
    assert harness.calibration_pair("S6P_NP_S040_J2", plan) == "D14_S040"
    assert harness.calibration_pair("S6P_NP_S010") == "D12_S010"  # v1.4 unchanged


def _kept(trace: int, t_ns: float) -> dict:
    return {"trace": trace, "t_ns": t_ns, "sample": round(t_ns / 0.098, 2)}


def test_nearest_kept_tie_rules(harness, target) -> None:
    # 1: smallest |t - t_a0| wins over a rank-first candidate.
    kept = [_kept(APEX_TRACE, T_A0_NS + 0.5), _kept(APEX_TRACE + 2, T_A0_NS + 0.1)]
    nearest, count = harness.nearest_kept(kept, target)
    assert (nearest["trace"], count) == (APEX_TRACE + 2, 2)
    # 2: equal time -> smallest |trace - apex|.
    kept = [_kept(APEX_TRACE + 2, T_A0_NS + 0.2), _kept(APEX_TRACE - 1, T_A0_NS - 0.2)]
    assert harness.nearest_kept(kept, target)[0]["trace"] == APEX_TRACE - 1
    # 3: equal time and equal |trace - apex| -> smallest trace index.
    kept = [
        _kept(APEX_TRACE + 1, T_A0_NS + 0.25),
        _kept(APEX_TRACE - 1, T_A0_NS - 0.25),
    ]
    assert harness.nearest_kept(kept, target)[0]["trace"] == APEX_TRACE - 1
    kept.reverse()
    assert harness.nearest_kept(kept, target)[0]["trace"] == APEX_TRACE - 1


def test_nearest_kept_window_is_closed(harness, target) -> None:
    edge = [_kept(APEX_TRACE + 3, T_A0_NS + 1.0)]  # +3 traces, +0.5/f_c
    assert harness.nearest_kept(edge, target) == (edge[0], 1)
    outside = [_kept(APEX_TRACE + 4, T_A0_NS), _kept(APEX_TRACE, T_A0_NS + 1.01)]
    assert harness.nearest_kept(outside, target) == (None, 0)


def _v2_models_json(path: Path, with_no_pipe_pair: bool = True) -> Path:
    models = [
        {"id": "S4N_D14_S040", "batch": "v2"},
        {"id": "S6P_D14_S040_J2", "batch": "v2"},
        {"id": "S6P_NP_S040_J2", "batch": "v2"},
        {"id": "S6P_D16_S040_J3", "batch": "v2_excluded", "reason": "Lambda_est < 15"},
    ]
    meta = {"models": models}
    if with_no_pipe_pair:
        meta["no_pipe_calibration_pair"] = "D14_S040"
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_v2_models_come_from_json_batch(harness, tmp_path) -> None:
    included, excluded, pair = harness.load_v2_models(
        _v2_models_json(tmp_path / "m.json")
    )
    assert included == ["S4N_D14_S040", "S6P_D14_S040_J2", "S6P_NP_S040_J2"]
    assert excluded == ["S6P_D16_S040_J3"] and pair == "D14_S040"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"models": [{"id": "S6P_D16_S040_J3", "batch": "v2"}]}),
                   encoding="utf-8")  # fmt: skip
    with pytest.raises(ValueError, match="control"):
        harness.load_v2_models(bad)
    wrong_label = tmp_path / "label.json"
    entry = {"id": "S4N_D14_S040", "batch": "v2", "pair": "P1"}
    wrong_label.write_text(json.dumps({"models": [entry]}), encoding="utf-8")
    with pytest.raises(ValueError, match="json pair"):
        harness.load_v2_models(wrong_label)


def test_v2_needs_a_seal_commit_and_a_no_pipe_pair(harness, tmp_path) -> None:
    models_json = _v2_models_json(tmp_path / "m.json", with_no_pipe_pair=False)
    with pytest.raises(SystemExit, match="seal-commit"):
        harness.main(["--protocol", "v2", "--models-json", str(models_json)])
    with pytest.raises(SystemExit, match="calibration pair"):
        harness.main(["--protocol", "v2", "--models-json", str(models_json),
                      "--seal-commit", "abc"])  # fmt: skip


def test_v2_main_writes_runs_v2_with_nearest_columns(
    harness, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        harness, "seal_record", lambda seal=None: _clean_seal(harness, seal)
    )
    data_root = tmp_path / "data"
    _placeholder_inputs(data_root, FAKE_V2_MODELS)
    models_json = _v2_models_json(tmp_path / "m.json")
    plan = dataclasses.replace(harness.V2_PLAN, **SMALL_PLAN)
    out = tmp_path / "out"
    argv = [
        "--protocol", "v2", "--models-json", str(models_json),
        "--input-dir", str(data_root), "--out-dir", str(out),
        "--seal-commit", "v2seal", "--regression-activities", "none",
    ]  # fmt: skip
    assert harness.main(argv, plan, _caching_loader(FAKE_V2_MODELS)) == 0
    assert not (out / "runs.csv").exists()
    fieldnames, rows = _read_rows(out / "runs_v2.csv")
    assert fieldnames == [*EXPECTED_RUN_COLUMNS, *V2_EXTRA_COLUMNS]
    pipe = [
        r for r in rows if r["model_id"] == "S6P_D14_S040_J2" and r["dt_ns"] == "0.098"
    ]
    assert {int(r["seed"]) for r in pipe} == {20271120, 20271220}
    with_nearest = [r for r in pipe if r["kept_nearest_trace"] != ""]
    assert with_nearest, "the fake pipe must give a nearest kept"
    for r in with_nearest:
        assert int(r["n_kept_in_window"]) >= 1
        assert abs(int(r["kept_nearest_trace"]) - APEX_TRACE) <= 3
        assert r["depth_nearest_status"]
    assert all(
        r["kept_nearest_trace"] == "" for r in pipe if r["n_kept_in_window"] == "0"
    )
    no_pipe = [r for r in rows if r["model_id"] == "S6P_NP_S040_J2"]
    assert {int(r["seed"]) for r in no_pipe} == {20275120, 20275220}
    assert all(r["n_kept_in_window"] == "" for r in no_pipe)
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    assert info["protocol"] == "v2" and info["seed_overlap"] == 0
    assert info["seal"]["seal_commit"] == "v2seal"
    assert info["models_run"] == list(FAKE_V2_MODELS)
    assert info["models_excluded_in_json"] == ["S6P_D16_S040_J3"]
    assert info["runs_csv"] == "runs_v2.csv"
    assert info["models"]["S6P_NP_S040_J2"]["calibration_pair"] == "D14_S040"
    # The judge's W10 check reads these harness keys (same names, same values).
    judge_path = PROJECT_ROOT / "analysis" / "processing" / "m210b_v2_judge.py"
    judge_spec = importlib.util.spec_from_file_location(
        "m210b_v2_judge_w10", judge_path
    )
    judge = importlib.util.module_from_spec(judge_spec)
    sys.modules[judge_spec.name] = judge
    judge_spec.loader.exec_module(judge)
    assert judge.w10_calibration_check(info)["pass"] is True
    record = info["models"]["S6P_D14_S040_J2"]["harness"]
    assert set(record["argmax_level_periods_noise_free"]) == {"+0.00", "+1.00"}
    assert info["environment"]["qc_constants"]["sha256"]


# ----------------------------------------------------------------- batch 1 replay

BATCH1_RUNS = PROJECT_ROOT / "data" / "processing" / "m210b_s4s6" / "runs.csv"
BATCH1_DATA = PROJECT_ROOT / "data" / "synthetic_gprmax_s4s6"


@pytest.mark.data
def test_v1_defaults_reproduce_batch1_rows(harness, tmp_path) -> None:
    """Default protocol: a subset of the stored batch-1 rows comes back unchanged."""
    if not (BATCH1_RUNS.exists() and (BATCH1_DATA / "S4N_D12_S010.npy").exists()):
        pytest.skip("batch-1 runs.csv or exports are not available")
    plan = harness.HarnessPlan(**{**SMALL_PLAN, "deltas_ns": (0.0, 2.35)})
    models = harness.ModelSet(BATCH1_DATA, plan)
    new_rows = harness.model_runs("S4N_D12_S010", models)
    written = tmp_path / "subset.csv"
    harness.write_runs(written, new_rows, plan.columns)
    _, regenerated = _read_rows(written)
    _, stored = _read_rows(BATCH1_RUNS)
    key_columns = ("model_id", "rho", "seed", "rule", "delta_ns", "ramp_noise", "dt_ns")
    stored_by_key = {tuple(r[c] for c in key_columns): r for r in stored}
    assert len(regenerated) == 2 * 2 * 2 + 1
    for row in regenerated:
        assert row == stored_by_key[tuple(row[c] for c in key_columns)]


# ----------------------------------------------------------------- v3 (V3-1 to V3-6)

FAKE_V3_MODELS = {
    "S4N_D14_S038": (True, False),
    "S6P_D14_S038_J6": (True, True),
    "S6P_NP_S038_J6": (False, True),
}
V3_EXTRA_COLUMNS = ["accepted_apex_strong_start", "n_strong_starts"]
"""v3 V3-2 columns after the v2 columns (``runs_v3.csv``)."""


def test_v3_plan_follows_the_spec(harness) -> None:
    plan = harness.V3_PLAN
    assert plan.n_seeds == 40 and plan.unit_noise_seed == 20281199
    assert harness.noise_seed(0, 0, 0, plan.seed_base) == 20281120
    assert harness.noise_seed(5, 1, 39, plan.seed_base) == 20281120 + 5000 + 100 + 39
    assert plan.rules == ("current", "B_full", "B_min", "C_none")
    assert plan.pair_index == {
        "D14_S038": 0, "D13_S042": 1, "D16_S045": 2, "D15_S048": 3,
        "D13_S012": 4, "NP_S038": 5,
    }  # fmt: skip
    assert list(plan.columns) == [
        *EXPECTED_RUN_COLUMNS, *V2_EXTRA_COLUMNS, *V3_EXTRA_COLUMNS,
    ]  # fmt: skip
    assert (
        harness.calibration_pair(
            "S6P_NP_S038_J6",
            dataclasses.replace(plan, no_pipe_calibration_pair="D14_S038"),
        )
        == "D14_S038"
    )
    assert harness.calibration_pair("S4N_D16_S045", plan) == "D16_S045"


def test_v3_seeds_are_disjoint_from_batch1_v2_and_prototype(harness) -> None:
    record = harness.seed_record(harness.V3_PLAN, "v3")
    assert record["seed_overlap"] == 0
    assert set(record["asserted_against"]) == {"batch1", "v2", "prototype"}
    assert record["n_v3_seeds"] == 6 * 2 * 40 + 1
    v3_seeds = harness.plan_seeds(harness.V3_PLAN)
    for earlier in (harness.BATCH1_SEEDS, harness.V2_SEEDS, harness.PROTOTYPE_SEED_SET):
        assert not v3_seeds & earlier


@pytest.mark.parametrize("seed_base", [20271120, 20261120, 20261111])
def test_v3_seed_overlap_is_asserted(harness, seed_base) -> None:
    # A v3 plan drawing v2, batch-1 or prototype seeds must not start.
    plan = dataclasses.replace(harness.V3_PLAN, seed_base=seed_base)
    with pytest.raises(harness.SeedOverlapError):
        harness.seed_record(plan, "v3")


def test_v2_seed_record_is_unchanged(harness) -> None:
    record = harness.seed_record(harness.V2_PLAN, "v2")
    assert record["seed_overlap"] == 0
    assert set(record["asserted_against"]) == {"batch1"}
    assert "prototype_seed_overlap" in record and "batch1_seed_set" in record


def _strong_start_case(harness, envelope_at_peak: float, threshold: float):
    """A one-peak line whose start envelope is ``envelope_at_peak``."""
    background_removed = np.tile(np.array([[-1.0], [1.0]]), (30, 8))  # MAD 1
    envelope = np.zeros((60, 8))
    envelope[20, 3] = envelope_at_peak
    line = {
        "_floor": {"k_hi": 44},
        "_peaks": [
            {"row": 20, "trace0": 3, "trace": 4, "sample": T_A0_NS / 0.098,
             "status": "accepted"},
        ],
    }  # fmt: skip
    target = harness.ApexTarget(4, T_A0_NS, F_C_HZ)
    sigma_n = harness.floor_rules.noise_sigma(background_removed[:44])
    assert threshold == pytest.approx(harness.floor_rules.K_NOISE * sigma_n)
    return harness.strong_start_outcome(
        line, envelope, background_removed, target, 0.098
    )


def test_strong_start_boundary_is_included(harness) -> None:
    threshold = 5 * 1.4826
    at = _strong_start_case(harness, threshold, threshold)
    assert (at.n_strong_starts, at.accepted_apex) == (1, True)
    below = _strong_start_case(harness, np.nextafter(threshold, 0), threshold)
    assert (below.n_strong_starts, below.accepted_apex) == (0, False)


def test_strong_start_needs_an_accepted_point_in_the_window(harness) -> None:
    background_removed = np.tile(np.array([[-1.0], [1.0]]), (30, 8))
    envelope = np.full((60, 8), 100.0)
    target = harness.ApexTarget(4, T_A0_NS, F_C_HZ)
    peaks = [
        {"row": 10, "trace0": 1, "trace": 4, "sample": T_A0_NS / 0.098,
         "status": "few_picks"},
        {"row": 12, "trace0": 2, "trace": 8, "sample": T_A0_NS / 0.098,
         "status": "accepted"},  # 4 traces from the apex
    ]  # fmt: skip
    line = {"_floor": {"k_hi": 44}, "_peaks": peaks}
    outcome = harness.strong_start_outcome(
        line, envelope, background_removed, target, 0.098
    )
    assert (outcome.n_strong_starts, outcome.accepted_apex) == (2, False)
    no_pipe = harness.strong_start_outcome(
        line, envelope, background_removed, None, 0.098
    )
    assert (no_pipe.n_strong_starts, no_pipe.accepted_apex) == (2, None)
    no_zero = harness.strong_start_outcome({}, None, background_removed, target, 0.098)
    assert (no_zero.n_strong_starts, no_zero.accepted_apex) == (0, False)


def _v3_models_json(path: Path) -> Path:
    models = [
        {"id": "S4N_D14_S038", "batch": "v3", "pair": "R1"},
        {"id": "S6P_D14_S038_J6", "batch": "v3", "pair": "R1"},
        {"id": "S6P_NP_S038_J6", "batch": "v3", "pair": "NP"},
        {"id": "S6P_D13_S042_J7", "batch": "v3_excluded", "pair": "R2"},
    ]
    meta = {"models": models, "no_pipe_calibration_pair": "D14_S038"}
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_v3_models_come_from_json_batch(harness, tmp_path) -> None:
    included, excluded, pair = harness.load_protocol_models(
        _v3_models_json(tmp_path / "m.json"), harness.V3_PROTOCOL
    )
    assert included == ["S4N_D14_S038", "S6P_D14_S038_J6", "S6P_NP_S038_J6"]
    assert excluded == ["S6P_D13_S042_J7"] and pair == "D14_S038"
    wrong = tmp_path / "wrong.json"
    entry = {"id": "S4N_D16_S045", "batch": "v3", "pair": "F2"}
    wrong.write_text(json.dumps({"models": [entry]}), encoding="utf-8")
    with pytest.raises(ValueError, match="json pair"):
        harness.load_protocol_models(wrong, harness.V3_PROTOCOL)


def test_v3_main_writes_runs_v3_with_strong_start_columns(
    harness, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        harness, "seal_record", lambda seal=None: _clean_seal(harness, seal)
    )
    data_root = tmp_path / "data"
    _placeholder_inputs(data_root, FAKE_V3_MODELS)
    models_json = _v3_models_json(tmp_path / "m.json")
    plan = dataclasses.replace(harness.V3_PLAN, **SMALL_PLAN)
    out = tmp_path / "out"
    argv = [
        "--protocol", "v3", "--models-json", str(models_json),
        "--input-dir", str(data_root), "--out-dir", str(out),
        "--seal-commit", "v3seal", "--regression-activities", "none",
    ]  # fmt: skip
    assert harness.main(argv, plan, _caching_loader(FAKE_V3_MODELS)) == 0
    assert not (out / "runs_v2.csv").exists()
    fieldnames, rows = _read_rows(out / "runs_v3.csv")
    assert fieldnames == [*EXPECTED_RUN_COLUMNS, *V2_EXTRA_COLUMNS, *V3_EXTRA_COLUMNS]
    pipe = [
        r for r in rows if r["model_id"] == "S6P_D14_S038_J6" and r["dt_ns"] == "0.098"
    ]
    assert {int(r["seed"]) for r in pipe} == {20281120, 20281220}
    current = [r for r in pipe if r["rule"] == "current"]
    assert current and all(r["n_strong_starts"] != "" for r in current)
    assert all(r["accepted_apex_strong_start"] in {"True", "False"} for r in current)
    others = [r for r in rows if r["rule"] != "current"]
    assert all(
        r["n_strong_starts"] == "" and r["accepted_apex_strong_start"] == ""
        for r in others
    )
    no_pipe = [r for r in rows if r["model_id"] == "S6P_NP_S038_J6"]
    assert {int(r["seed"]) for r in no_pipe} == {20286120, 20286220}
    assert all(
        r["accepted_apex_strong_start"] == "" for r in no_pipe if r["rule"] == "current"
    )
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    assert info["protocol"] == "v3" and info["seed_overlap"] == 0
    assert set(info["seeds"]["asserted_against"]) == {"batch1", "v2", "prototype"}
    assert info["runs_csv"] == "runs_v3.csv"
    assert info["models_excluded_in_json"] == ["S6P_D13_S042_J7"]
    assert info["models"]["S6P_NP_S038_J6"]["calibration_pair"] == "D14_S038"
    assert "m210b_selector_floor_v3" in info["spec"]
    judge_path = PROJECT_ROOT / "analysis" / "processing" / "m210b_v3_judge.py"
    judge_spec = importlib.util.spec_from_file_location(
        "m210b_v3_judge_w10", judge_path
    )
    judge = importlib.util.module_from_spec(judge_spec)
    sys.modules[judge_spec.name] = judge
    judge_spec.loader.exec_module(judge)
    assert judge.w10_calibration_check(info)["pass"] is True


def test_v3_start_guard_needs_head_at_the_seal(harness, tmp_path, monkeypatch) -> None:
    def seal_record(seal=None):
        record = _clean_seal(harness, seal)
        record["head_commit"] = "later"
        return record

    monkeypatch.setattr(harness, "seal_record", seal_record)
    data_root = tmp_path / "data"
    _placeholder_inputs(data_root, FAKE_V3_MODELS)
    argv = [
        "--protocol", "v3", "--models-json", str(_v3_models_json(tmp_path / "m.json")),
        "--input-dir", str(data_root), "--out-dir", str(tmp_path / "out"),
        "--seal-commit", "v3seal", "--regression-activities", "none",
    ]  # fmt: skip
    plan = dataclasses.replace(harness.V3_PLAN, **SMALL_PLAN)
    assert harness.main(argv, plan, _caching_loader(FAKE_V3_MODELS)) == 1
    assert not (tmp_path / "out" / "runs_v3.csv").exists()


def test_v3_seal_lists_its_files(harness) -> None:
    sealed = set(harness.V3_SEALED_FILES)
    assert {
        "docs/processing/m210b_selector_floor_v3.md",
        "docs/processing/m210b_selector_floor_v2.md",
        "analysis/synthetic/s4s6/models_s4s6_v3.json",
        "analysis/processing/m210b_v3_judge.py",
        "tests/test_m210b_v3_judge.py",
        "tests/test_m210b_s4s6_pipeline.py",
        "tests/test_synthetic_export.py",
        "analysis/processing/m210b_floor_rules.py",
        "analysis/processing/fourtu_apex_candidate_select.py",
    } <= sealed


@pytest.mark.parametrize(
    ("path", "prefix", "is_unchanged"),
    [
        ("analysis/processing/m210b_floor_rules.py", "bad0f95c", True),
        # M2-10 K7 adds generator flags (m210_k7_generator.md 4-2): the blob
        # changes; the default path is checked against the 752e783 wrapper by
        # tests/test_m210_k7_generator.py (T6) and the K7 G0 regression.
        ("analysis/processing/fourtu_apex_candidate_select.py", "7cc2bf2b", False),
    ],
)
def test_floor_rules_and_wrapper_are_unchanged_since_752e783(
    harness, path, prefix, is_unchanged
) -> None:
    sealed = harness._git("rev-parse", f"752e783:{path}")
    working = harness._git("hash-object", "--", path)
    if sealed is None or working is None:
        pytest.skip("git history with 752e783 is not available")
    assert sealed.startswith(prefix)
    if is_unchanged:
        assert working == sealed


# ----------------------------------------------------------------- v2 replay

V2_RUNS = PROJECT_ROOT / "data" / "processing" / "m210b_s4s6_v2" / "runs_v2.csv"
V2_DATA = PROJECT_ROOT / "data" / "synthetic_gprmax_s4s6_v2"


@pytest.mark.data
def test_v2_protocol_reproduces_v2_rows(harness, tmp_path) -> None:
    """Protocol v2: a subset of the stored ``runs_v2.csv`` rows comes back unchanged."""
    if not (V2_RUNS.exists() and (V2_DATA / "S4N_D14_S040.npy").exists()):
        pytest.skip("v2 runs_v2.csv or exports are not available")
    plan = dataclasses.replace(
        harness.V2_PLAN,
        n_seeds=1,
        rules=("current", "B_full"),
        deltas_ns=(0.0, 2.35),
        original_dt_rules=("current",),
        no_pipe_calibration_pair="D14_S040",
    )
    models = harness.ModelSet(V2_DATA, plan)
    new_rows = harness.model_runs("S4N_D14_S040", models)
    written = tmp_path / "subset.csv"
    harness.write_runs(written, new_rows, plan.columns)
    _, regenerated = _read_rows(written)
    _, stored = _read_rows(V2_RUNS)
    key_columns = ("model_id", "rho", "seed", "rule", "delta_ns", "ramp_noise", "dt_ns")
    stored_by_key = {tuple(r[c] for c in key_columns): r for r in stored}
    assert len(regenerated) == 2 * 2 * 2 + 1
    for row in regenerated:
        assert row == stored_by_key[tuple(row[c] for c in key_columns)]
