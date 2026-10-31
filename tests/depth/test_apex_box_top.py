"""Apex hint and box-top guard (spec_depth_v1 section 5-7, M2-9 improvement A).

In the M2-8 4TU run most ``ok`` apexes moved about 3 ns (the box top margin
``1.5 / f_c``) earlier than the apex candidate. Two kinds were found:

* box-top boundary effect (76.5 % of the moves): an integer peak index 1-2 at
  the top of the search window (the cut-off tail or front of a shallower
  event). Section 5-7 item 2 rejects those peaks (A2a, A2b).
* a real shallow event inside the top margin (A1): only the apex hint
  (item 1) resolves it, so A1(b) without a hint stays ``xfail(strict=True)``.

Generator setup (4TU-like): air-coupled single surface event, 500 MHz,
eps_r 9, pipe top 0.6 m, h 0.05 m. Shallow events span +-1.0 m only on a
401-trace line, so the 301-trace background median keeps them like local
clutter. A6 (synthetic set, ``-m synthetic``) checks that the true-apex hint
leaves the 20 gprMax depths unchanged.
"""

import math

import numpy as np
import pytest

from gpr_detection.depth.config import DepthConfig
from gpr_detection.depth.estimate import estimate_depth, prepare_depth_inputs
from gpr_detection.depth.picks import (
    ApexHint,
    extract_picks,
    find_apex_candidate,
    hint_peak_floor,
    noise_sigma,
    pick_signal,
    widen_box,
)
from gpr_detection.depth.refraction import reflection_times
from gpr_detection.io.paths import DEFAULT_SYNTHETIC_ROOT
from gpr_detection.io.radargram import GridBox, Radargram
from synthetic_bscan import DT_NS, T_SRC_NS, BScanSpec, make_radargram
from synthetic_bscan import F_C_HZ as U1_F_C_HZ

F_C_HZ = 500e6
EPS_R = 9.0
PIPE_DEPTH_M = 0.6
AIR_GAP_M = 0.05
OFFSET_M = 0.1
N_TRACES = 401
"""8 m line: the +-1.0 m shallow event covers under half the background window."""

LAYER_LEAD_NS = 2.5
LAYER_AMPLITUDE = 2.0
LAYER_HALF_WIDTH_M = 1.0

FOURTU_DT_NS = 0.098
A2B_TOP_INDEX = 1.5
"""A2b: shallow-event envelope peak 1.5 samples below the search top (index 1-2)."""

APEX_TOLERANCE_PERIODS = 0.5
APEX_TOLERANCE_TRACES = 3
DEPTH_TOLERANCE_M = 0.02
A6_DEPTH_TOLERANCE_M = 0.001

PERIOD_NS = 1e9 / F_C_HZ
SECOND_TYPE_REASON = (
    "second type: shallow event inside the top margin; only apex_hint "
    "(5-7 item 1) resolves it"
)


# ----------------------------------------------------------------- generators


def _spec(dt_ns: float = DT_NS, snr_db: float | None = 20.0) -> BScanSpec:
    spec = BScanSpec(
        f_c_hz=F_C_HZ,
        eps_r=EPS_R,
        d_top_m=PIPE_DEPTH_M,
        air_gap_m=AIR_GAP_M,
        offset_m=OFFSET_M,
        n_traces=N_TRACES,
        dt_ns=dt_ns,
        snr_db=snr_db,
    )
    spec.pipe_times_ns = reflection_times(
        spec.x_m,
        offset_m=OFFSET_M,
        h=AIR_GAP_M,
        x0_m=spec.x0_m,
        d_top_m=PIPE_DEPTH_M,
        radius_m=0.0,
        v=spec.velocity,
    )
    return spec


def _add_shallow_event(spec: BScanSpec, time_ns: float) -> None:
    """Horizontal event of twice the pipe amplitude over +-1.0 m (record clock)."""
    inside = np.abs(spec.x_m - spec.x0_m) <= LAYER_HALF_WIDTH_M
    times = np.full(spec.n_traces, time_ns - T_SRC_NS)
    spec.extra_events.append((times, LAYER_AMPLITUDE * inside))


def _true_apex(spec: BScanSpec) -> tuple[int, float]:
    """True apex trace and envelope-peak sample of ``spec``."""
    trace = spec.n_traces // 2
    return trace, (T_SRC_NS + float(spec.pipe_times_ns[trace])) / spec.dt_ns


def _line(with_layer: bool) -> tuple[Radargram, int, float]:
    """A1 line: shallow event 2.5 ns above the apex (inside the top margin)."""
    spec = _spec()
    true_trace, true_sample = _true_apex(spec)
    if with_layer:
        _add_shallow_event(spec, true_sample * DT_NS - LAYER_LEAD_NS)
    radargram = make_radargram(spec, line_id="01.1/PathT")
    return radargram, true_trace, true_sample


def _box(load_script, radargram: Radargram, trace: int, sample: float) -> GridBox:
    """The 4TU evaluation box around a candidate placed on the true apex."""
    candidate_box = load_script("eval_depth_4tu").candidate_box
    zero = prepare_depth_inputs(radargram).zero
    tau_ns = sample * radargram.dt_ns - zero.t_zero_ns
    return candidate_box(radargram, trace, sample, tau_ns)


def _a2b_line(load_script) -> tuple[Radargram, GridBox, int, float]:
    """A2b: dt 0.098 ns, shallow event cut by the box top (peak at index 1-2)."""
    spec = _spec(dt_ns=FOURTU_DT_NS, snr_db=None)
    true_trace, true_sample = _true_apex(spec)
    event_sample = true_sample - LAYER_LEAD_NS / FOURTU_DT_NS
    _add_shallow_event(spec, event_sample * FOURTU_DT_NS)
    radargram = make_radargram(spec, line_id="01.1/PathT")
    wide_box = _box(load_script, radargram, true_trace, true_sample)
    top = math.floor(event_sample - A2B_TOP_INDEX + 0.5)
    box = GridBox(i0=wide_box.i0, i1=wide_box.i1, k0=top, k1=wide_box.k1)
    return radargram, box, true_trace, true_sample


def _assert_apex_near_truth(
    apex_trace: int,
    apex_sample: float,
    true_trace: int,
    true_sample: float,
    dt_ns: float = DT_NS,
) -> None:
    sample_limit = APEX_TOLERANCE_PERIODS * PERIOD_NS / dt_ns
    assert abs(apex_sample - true_sample) <= sample_limit
    assert abs(apex_trace - true_trace) <= APEX_TOLERANCE_TRACES


# ----------------------------------------------------------------- A1


def test_a1_box_without_layer_keeps_the_true_apex(load_script) -> None:
    """Control: the same box without the layer finds the true apex."""
    radargram, true_trace, true_sample = _line(with_layer=False)
    box = _box(load_script, radargram, true_trace, true_sample)
    result = estimate_depth(radargram, box)
    assert result.apex_trace is not None and result.apex_sample is not None
    _assert_apex_near_truth(
        result.apex_trace, result.apex_sample, true_trace, true_sample
    )


def test_a1_layer_lies_inside_the_box_top_margin(load_script) -> None:
    """Precondition: the layer is in the box, above the apex, below the guard."""
    radargram, true_trace, true_sample = _line(with_layer=True)
    box = _box(load_script, radargram, true_trace, true_sample)
    layer_sample = true_sample - LAYER_LEAD_NS / DT_NS
    assert box.k0 + DepthConfig().top_guard_samples < layer_sample < true_sample
    assert true_sample - box.k0 == pytest.approx(1.5 * PERIOD_NS / DT_NS, abs=1.0)


def test_a1a_hint_on_the_true_apex_ignores_the_layer(load_script) -> None:
    radargram, true_trace, true_sample = _line(with_layer=True)
    box = _box(load_script, radargram, true_trace, true_sample)
    hint = ApexHint(trace=true_trace, sample=true_sample)
    result = estimate_depth(radargram, box, apex_hint=hint)
    assert result.status == "ok"
    assert result.apex_trace is not None and result.apex_sample is not None
    _assert_apex_near_truth(
        result.apex_trace, result.apex_sample, true_trace, true_sample
    )
    assert result.depth_top_m == pytest.approx(PIPE_DEPTH_M, abs=DEPTH_TOLERANCE_M)
    assert "apex_at_box_edge" not in result.flags


@pytest.mark.xfail(strict=True, reason=SECOND_TYPE_REASON)
def test_a1b_without_hint_extract_picks_ignores_the_layer(load_script) -> None:
    radargram, true_trace, true_sample = _line(with_layer=True)
    box = _box(load_script, radargram, true_trace, true_sample)
    inputs = prepare_depth_inputs(radargram)
    picks = extract_picks(
        inputs.branch.background_removed,
        box,
        inputs.zero,
        dt_ns=radargram.dt_ns,
        dx_m=radargram.dx_m,
        f_c_hz=radargram.f_c_hz,
    )
    assert picks.apex is not None
    _assert_apex_near_truth(picks.apex.trace, picks.apex.k, true_trace, true_sample)


@pytest.mark.xfail(strict=True, reason=SECOND_TYPE_REASON)
def test_a1b_without_hint_estimate_depth_ignores_the_layer(load_script) -> None:
    radargram, true_trace, true_sample = _line(with_layer=True)
    box = _box(load_script, radargram, true_trace, true_sample)
    result = estimate_depth(radargram, box)
    assert result.apex_trace is not None and result.apex_sample is not None
    _assert_apex_near_truth(
        result.apex_trace, result.apex_sample, true_trace, true_sample
    )


# ----------------------------------------------------------------- A2


A2A_TOP_K = 100
A2A_N_SAMPLES = 60
A2A_N_COLUMNS = 7
A2A_TRUE_INDEX = 30


def _envelope_with_peaks(top_index: int) -> np.ndarray:
    """Envelope columns: a 0.8 peak at ``top_index`` and the 1.0 apex at 30."""
    index = np.arange(A2A_N_SAMPLES, dtype=float)[:, None]
    column = 0.05 + 0.8 * np.exp(-0.5 * ((index - top_index) / 0.6) ** 2)
    column = column + 1.0 * np.exp(-0.5 * ((index - A2A_TRUE_INDEX) / 2.0) ** 2)
    signal = np.zeros((A2A_TOP_K + A2A_N_SAMPLES + 20, A2A_N_COLUMNS))
    signal[A2A_TOP_K : A2A_TOP_K + A2A_N_SAMPLES] = column
    return signal


def _a2a_box() -> GridBox:
    return GridBox(
        i0=0, i1=A2A_N_COLUMNS - 1, k0=A2A_TOP_K, k1=A2A_TOP_K + A2A_N_SAMPLES - 1
    )


def test_a2a_top_guard_rejects_the_index_1_peak() -> None:
    signal = _envelope_with_peaks(top_index=1)
    candidate, flags = find_apex_candidate(
        signal, _a2a_box(), 0.3, top_guard=DepthConfig().top_guard_samples
    )
    assert candidate is not None
    assert candidate.k == pytest.approx(A2A_TOP_K + A2A_TRUE_INDEX, abs=0.5)
    assert "apex_at_box_top" in flags


def test_a2a_control_peak_at_index_3_is_kept() -> None:
    signal = _envelope_with_peaks(top_index=3)
    candidate, flags = find_apex_candidate(
        signal, _a2a_box(), 0.3, top_guard=DepthConfig().top_guard_samples
    )
    assert candidate is not None
    assert candidate.k == pytest.approx(A2A_TOP_K + 3, abs=0.5)
    assert flags == ()


def test_a2b_cut_shallow_event_peaks_at_index_1_to_2(load_script) -> None:
    """Precondition: the cut event gives an integer peak 1-2 above 0.3 E_max."""
    from scipy.signal import find_peaks

    radargram, box, true_trace, _ = _a2b_line(load_script)
    inputs = prepare_depth_inputs(radargram)
    searched = widen_box(box, radargram.n_traces, DepthConfig().box_expand_frac)
    envelope = pick_signal(inputs.branch.background_removed, "envelope")
    window = envelope[searched.sample_slice, searched.trace_slice]
    column = window[:, true_trace - searched.i0]
    peaks, _ = find_peaks(column, height=0.3 * float(window.max()))
    assert 1 <= peaks[0] <= 2


def test_a2b_cut_shallow_event_is_rejected_at_the_box_top(load_script) -> None:
    radargram, box, true_trace, true_sample = _a2b_line(load_script)
    result = estimate_depth(radargram, box)
    assert result.status == "ok"
    assert result.apex_trace is not None and result.apex_sample is not None
    _assert_apex_near_truth(
        result.apex_trace,
        result.apex_sample,
        true_trace,
        true_sample,
        dt_ns=FOURTU_DT_NS,
    )
    assert result.depth_top_m == pytest.approx(PIPE_DEPTH_M, abs=DEPTH_TOLERANCE_M)
    assert "apex_at_box_top" in result.flags


# ----------------------------------------------------------------- A3, A4

U1_BOX = GridBox(i0=25, i1=125, k0=270, k1=560)
"""Ground-coupled U1 box (spec 11-1), as in ``test_estimate.py``."""
U_PERIOD_NS = 1e9 / U1_F_C_HZ
WRONG_HINT_OFFSETS_NS = (3.0, 5.0, -2.0)
RIGHT_HINT_OFFSET_NS = 0.3
A4_WRONG_HINT_PX = 60


def _u1_line(snr_db: float | None) -> tuple[Radargram, int, float]:
    """U1 data (ground-coupled, 400 MHz, d 0.6 m) and its true apex."""
    spec = BScanSpec(snr_db=snr_db)
    radargram = make_radargram(spec)
    true_sample = (T_SRC_NS + 2 * spec.d_top_m / spec.velocity) / DT_NS
    return radargram, spec.n_traces // 2, true_sample


@pytest.mark.parametrize("snr_db", [20.0, None], ids=["snr20", "noise_free"])
@pytest.mark.parametrize("offset_ns", WRONG_HINT_OFFSETS_NS)
def test_a3_misplaced_hint_does_not_fall_back(snr_db, offset_ns) -> None:
    radargram, true_trace, true_sample = _u1_line(snr_db)
    hint = ApexHint(trace=true_trace, sample=true_sample + offset_ns / DT_NS)
    result = estimate_depth(radargram, U1_BOX, apex_hint=hint)
    assert result.status == "fit_failed"
    assert result.apex_trace is None  # no fallback to the whole box
    assert {"no_apex_candidate", "apex_hint_unmatched"} <= set(result.flags)


@pytest.mark.parametrize("snr_db", [20.0, None], ids=["snr20", "noise_free"])
def test_a3_positive_control_hint_near_the_apex(snr_db) -> None:
    radargram, true_trace, true_sample = _u1_line(snr_db)
    hint = ApexHint(trace=true_trace, sample=true_sample + RIGHT_HINT_OFFSET_NS / DT_NS)
    result = estimate_depth(radargram, U1_BOX, apex_hint=hint)
    assert result.status == "ok"
    assert result.apex_trace == pytest.approx(true_trace, abs=1)
    sample_limit = APEX_TOLERANCE_PERIODS * U_PERIOD_NS / DT_NS
    assert result.apex_sample == pytest.approx(true_sample, abs=sample_limit)


def test_a7_noise_free_floor_falls_back_to_the_box_fraction() -> None:
    """Without noise sigma_n is small, so the r x box-maximum floor (ii) decides.

    sigma_n is not exactly 0 here: the MAD region (search top to record end)
    holds the hyperbola and filter tails (5 sigma_n is about 2 % of the box
    maximum on U1).
    """
    radargram, _, _ = _u1_line(snr_db=None)
    config = DepthConfig()
    inputs = prepare_depth_inputs(radargram)
    section = inputs.branch.background_removed
    searched = widen_box(U1_BOX, radargram.n_traces, config.box_expand_frac)
    signal = pick_signal(section[:, searched.trace_slice], config.pick_mode)
    local_box = GridBox(i0=0, i1=searched.n_traces - 1, k0=searched.k0, k1=searched.k1)
    box_max = float(signal[local_box.sample_slice, local_box.trace_slice].max())
    noise_floor = config.hint_noise_k * noise_sigma(section, searched)
    floor = hint_peak_floor(section, signal, searched, local_box, config)
    assert noise_floor < config.hint_box_rel_floor * box_max
    assert floor == pytest.approx(config.hint_box_rel_floor * box_max)


def test_a4_hint_without_scale_uses_a_pixel_window() -> None:
    spec = BScanSpec()  # U7: U1 data without scale, SNR 20 dB
    radargram = make_radargram(spec, dt_ns=None, dx_m=None, f_c_hz=None)
    true_trace = spec.n_traces // 2
    true_sample = (T_SRC_NS + 2 * spec.d_top_m / spec.velocity) / DT_NS
    hint = ApexHint(trace=true_trace, sample=true_sample)
    result = estimate_depth(radargram, U1_BOX, apex_hint=hint)
    assert result.status == "no_scale"
    assert result.apex_trace == pytest.approx(true_trace, abs=1)
    assert result.apex_sample == pytest.approx(true_sample, abs=2)

    wrong = ApexHint(trace=true_trace, sample=true_sample + A4_WRONG_HINT_PX)
    wrong_result = estimate_depth(radargram, U1_BOX, apex_hint=wrong)
    assert wrong_result.apex_trace is None
    assert "apex_hint_unmatched" in wrong_result.flags


# ----------------------------------------------------------------- A6


@pytest.mark.synthetic
def test_a6_true_apex_hint_keeps_the_synthetic_depths(load_script) -> None:
    from gpr_detection.io.synthetic import (
        list_synthetic_models,
        load_synthetic,
        load_synthetic_truth,
    )

    if not (DEFAULT_SYNTHETIC_ROOT / "manifest.csv").is_file():
        pytest.skip(f"synthetic data not found at {DEFAULT_SYNTHETIC_ROOT}")
    supply_box = load_script("eval_depth_synthetic").supply_box
    differences_m = {}
    for model_id in list_synthetic_models():
        truth = load_synthetic_truth(model_id)
        radargram = load_synthetic(model_id)
        box = supply_box(truth, radargram)
        inputs = prepare_depth_inputs(radargram)
        plain = estimate_depth(radargram, box, inputs=inputs)
        hint = ApexHint(
            trace=truth.apex_trace_index,
            sample=truth.apex_envelope_peak_ns / radargram.dt_ns,
        )
        hinted = estimate_depth(radargram, box, inputs=inputs, apex_hint=hint)
        assert plain.depth_top_m is not None and hinted.depth_top_m is not None
        differences_m[model_id] = abs(hinted.depth_top_m - plain.depth_top_m)
    assert len(differences_m) == 20
    assert max(differences_m.values()) <= A6_DEPTH_TOLERANCE_M, differences_m
