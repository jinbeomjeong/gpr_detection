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


def test_main_writes_runs_and_run_info(harness, tmp_path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    for model_id in FAKE_MODELS:
        for suffix in (".npy", ".json"):
            (data_root / f"{model_id}{suffix}").write_bytes(b"placeholder")
    models_json = tmp_path / "models.json"
    models_json.write_text("{}", encoding="utf-8")
    cache = {}

    def loader(model_id, root):
        if model_id not in cache:
            cache[model_id] = _fake_model(model_id, *FAKE_MODELS[model_id])
        return cache[model_id]

    plan = harness.HarnessPlan(
        n_seeds=1,
        rules=("current", "B_full"),
        deltas_ns=(0.0, 1.0),
        original_dt_rules=("current",),
    )
    out = tmp_path / "out"
    argv = [
        "--data-root", str(data_root),
        "--models-json", str(models_json),
        "--out-dir", str(out),
        "--models", "S6P_D12_S010,S6P_NP_S010",
        "--regression-activities", "none",
    ]  # fmt: skip
    assert harness.main(argv, plan, loader) == 0

    with (out / "runs.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == EXPECTED_RUN_COLUMNS
    assert list(harness.RUN_COLUMNS) == EXPECTED_RUN_COLUMNS
    assert set(json.loads((out / "run_info.json").read_text(encoding="utf-8"))[
        "protocol"]["columns"]) >= {"rho_real", "rho_real_cal"}  # fmt: skip
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

    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    assert info["seal"]["seal_commit"] == "bc9a474"
    sealed = info["seal"]["sealed_files"]
    assert "docs/processing/m210b_selector_floor.md" in sealed
    assert any(path.startswith("analysis/synthetic/s4s6/") for path in sealed)
    assert info["mkl_cbwr"]
    assert info["wrapper_regression"] == {"run": False}
    assert set(info["inputs"]) == {
        "models_json",
        *(f"{m}{s}" for m in FAKE_MODELS for s in (".npy", ".json")),
    }
    record = info["models"]["S6P_D12_S010"]["harness"]
    assert record["calibration"]["control_id"] == "S4S6/S4N_D12_S010"
    assert record["t_a0_ns"] is not None and record["A0_self"] > 0
    assert set(record["Lambda"]) == {"+0.00", "+1.00"}
    assert info["models"]["S6P_NP_S010"]["calibration_pair"] == "D12_S010"
