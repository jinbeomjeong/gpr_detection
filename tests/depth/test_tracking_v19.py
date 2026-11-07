"""Tracking improvements T1 (apex descent) and T2 (reverse rule), spec_depth_v1 11-3.

TK1-TK10 of spec_depth v1.9 section 11-3 and TK11-TK14 of v1.9.2
(diagnosis ``m29c_track_diagnosis.md``).
The generator has the form of ``base_section`` in
``analysis/processing/m29c_track_repro.py`` but lives here (tests never import
analysis scripts): 500 MHz Ricker, dt 0.098 ns, dx 0.02 m, eps_r 9, 161 traces,
a ground-coupled direct wave (+3) at ``t_src + 0.7`` ns, and an R = 0 point
pipe whose wing amplitude falls as ``(d / r)^4`` (beam / spreading
**assumption**; it makes the ``0.2 E_a`` stop end the wings at 0.5-0.7 m).
Expected values come from ``docs/processing/m29c_track_repro_det.csv``.

"Before the fix" expectations (strict xfail in the spec table) are checked as
controls instead: the v1.8 tracker is replayed through ``_track_one_side``
with the v1.8 limits (previous pick, 0.1 ns), and the v1.8 apex comes from
``extract_picks`` with ``descend_apex`` patched to the identity.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
import pytest

import gpr_detection.depth.picks as picks_module
from gpr_detection.depth.config import DepthConfig
from gpr_detection.depth.estimate import DepthInputs, prepare_depth_inputs
from gpr_detection.depth.hyperbola import fit_curvature, fixed_velocity_apex
from gpr_detection.depth.picks import (
    APEX_DESCENDED,
    ApexCandidate,
    ApexHint,
    HyperbolaPicks,
    TrackStop,
    extract_picks,
    reverse_tolerance_ns,
)
from gpr_detection.io.radargram import GridBox, Radargram, velocity_from_eps
from gpr_detection.preprocess.signals import pick_signal
from synthetic_bscan import ricker

F_C_HZ = 500e6
PERIOD_NS = 1e9 / F_C_HZ
DT_NS = 0.098
DX_M = 0.02
EPS_R = 9.0
VELOCITY = velocity_from_eps(EPS_R)
N_TRACES = 161
N_SAMPLES = 420
T_SRC_NS = 3.0
DIRECT_DELAY_NS = 0.7
DIRECT_AMPLITUDE = 3.0
APEX_TRACE = N_TRACES // 2
PIPE_DEPTH_M = 0.5
CONFIG = DepthConfig()

V18_REVERSE_NS = 0.1
"""The v1.8 reverse tolerance, kept only for the before-the-fix controls."""
JITTER_HALF_TRACES = 6
"""D1: alternating shifts on the apex and 6 traces per side."""
CLEAN_RATIO = 1.0316
"""D0 curvature ratio ``v_curv / v`` (m29c_track_repro_det.csv)."""
INTRUDER_MARGIN_NS = 0.05


# ----------------------------------------------------------------- generator


@dataclass(frozen=True)
class Event:
    """One extra reflection: delay after ``t_src`` and amplitude, per trace."""

    times_ns: np.ndarray
    amplitudes: np.ndarray


@dataclass(frozen=True)
class Line:
    """A synthetic line ready for ``extract_picks``."""

    radargram: Radargram
    inputs: DepthInputs
    pipe_times_ns: np.ndarray | None
    """Two-way time after ``t_src`` of the pipe per trace (shifts included)."""


def _place(data: np.ndarray, times_ns: np.ndarray, amplitudes: np.ndarray) -> None:
    t_ns = np.arange(N_SAMPLES)[:, None] * DT_NS
    data += np.asarray(amplitudes)[None, :] * ricker(t_ns - times_ns[None, :], F_C_HZ)


def _offsets_m() -> np.ndarray:
    return (np.arange(N_TRACES) - APEX_TRACE) * DX_M


def _pipe_times_ns(depth_m: float) -> np.ndarray:
    return 2.0 / VELOCITY * np.hypot(depth_m, _offsets_m())


def _line(
    depth_m: float | None = PIPE_DEPTH_M,
    shifts_ns: np.ndarray | None = None,
    extra: tuple[Event, ...] = (),
) -> Line:
    """Direct wave, optional point pipe (``None``: no pipe) and extra events."""
    data = np.zeros((N_SAMPLES, N_TRACES))
    _place(
        data,
        np.full(N_TRACES, T_SRC_NS + DIRECT_DELAY_NS),
        np.full(N_TRACES, DIRECT_AMPLITUDE),
    )
    pipe_times = None
    if depth_m is not None:
        pipe_times = _pipe_times_ns(depth_m)
        if shifts_ns is not None:
            pipe_times = pipe_times + shifts_ns
        wing_amplitude = (depth_m / np.hypot(depth_m, _offsets_m())) ** 4
        _place(data, T_SRC_NS + pipe_times, wing_amplitude)
    for event in extra:
        _place(data, T_SRC_NS + event.times_ns, event.amplitudes)
    radargram = Radargram(
        data=data.astype(np.float32),
        dt_ns=DT_NS,
        dx_m=DX_M,
        air_gap="ground",
        amplitude_processed=False,
        dataset="unit",
        line_id="tk/track",
        f_c_hz=F_C_HZ,
        eps_r_meta=EPS_R,
        eps_r_meta_source="synthetic_truth",
        antenna_offset_m=0.1,
    )
    return Line(radargram, prepare_depth_inputs(radargram, CONFIG), pipe_times)


def _jitter_shifts(amplitude_ns: float) -> np.ndarray:
    """D1: ``+s0`` on even ``|j - j_a|``, ``-s0`` on odd, within 6 traces."""
    distance = np.abs(np.arange(N_TRACES) - APEX_TRACE)
    alternating = np.where(distance % 2 == 0, amplitude_ns, -amplitude_ns)
    return np.where(distance <= JITTER_HALF_TRACES, alternating, 0.0)


def _true_apex_sample(line: Line, depth_m: float) -> float:
    """Envelope-peak sample of the processed apex trace within +-0.5 period."""
    background_removed = line.inputs.branch.background_removed
    envelope = pick_signal(background_removed[:, [APEX_TRACE]], "envelope")[:, 0]
    centre = (T_SRC_NS + 2.0 / VELOCITY * depth_m) / DT_NS
    lo = int(centre - 0.5 * PERIOD_NS / DT_NS)
    hi = int(centre + 0.5 * PERIOD_NS / DT_NS)
    k_peak = lo + int(np.argmax(envelope[lo : hi + 1]))
    return float(picks_module.parabolic_peak(envelope, k_peak)[0])


def _box(load_script, line: Line, hint: ApexHint) -> GridBox:
    candidate_box = load_script("eval_depth_4tu").candidate_box
    tau_ns = hint.sample * DT_NS - _t_zero_ns(line)
    return candidate_box(line.radargram, hint.trace, hint.sample, tau_ns)


def _picks(load_script, line: Line, hint: ApexHint | None) -> HyperbolaPicks:
    """``extract_picks`` as in the 4TU evaluation (box from the hint position)."""
    box_hint = hint if hint is not None else _apex_hint(line, PIPE_DEPTH_M)
    return extract_picks(
        line.inputs.branch.background_removed,
        _box(load_script, line, box_hint),
        line.inputs.zero,
        dt_ns=DT_NS,
        dx_m=DX_M,
        f_c_hz=F_C_HZ,
        config=CONFIG,
        apex_hint=hint,
        period_px=line.inputs.branch.period_px,
    )


def _apex_hint(line: Line, depth_m: float) -> ApexHint:
    return ApexHint(trace=APEX_TRACE, sample=_true_apex_sample(line, depth_m))


def _curvature_ratio(picks: HyperbolaPicks) -> float:
    curvature = fit_curvature(
        picks, radius_m=0.0, air_gap="ground", v_fallback=VELOCITY, config=CONFIG
    )
    assert curvature.fit is not None
    return curvature.fit.v_m_per_ns / VELOCITY


def _apex(picks: HyperbolaPicks) -> ApexCandidate:
    assert picks.apex is not None
    return picks.apex


def _t_zero_ns(line: Line) -> float:
    t_zero_ns = line.inputs.zero.t_zero_ns
    assert t_zero_ns is not None
    return t_zero_ns


# ----------------------------------------------------------------- v1.8 replay


def _limits(reverse_ns: float, reference_n: int) -> picks_module._TrackLimits:
    return picks_module._TrackLimits(
        back_samples=CONFIG.track_back_ns / DT_NS,
        forward_samples=(2.0 * DX_M / CONFIG.v_min + CONFIG.track_back_ns) / DT_NS,
        reverse_samples=reverse_ns / DT_NS,
        reverse_ref_n=reference_n,
        stop_amplitude=0.0,  # set per apex below
        max_steps=math.floor(CONFIG.a_max_m / DX_M + 1e-9),
    )


def _replay_wings(
    line: Line, picks: HyperbolaPicks, reverse_ns: float, reference_n: int
) -> dict[int, tuple[list[tuple[int, float, float]], TrackStop]]:
    """Track both wings again from ``picks.apex`` with other reverse limits.

    Returns ``{-1: (left picks, stop), +1: (right picks, stop)}`` in the
    local (searched-box) trace coordinates of ``_track_one_side``.
    """
    assert picks.apex is not None
    searched = picks.box
    signal = pick_signal(
        line.inputs.branch.background_removed[:, searched.trace_slice],
        CONFIG.pick_mode,
    )
    local_box = GridBox(i0=0, i1=searched.n_traces - 1, k0=searched.k0, k1=searched.k1)
    apex = ApexCandidate(
        trace=_apex(picks).trace - searched.i0,
        k=_apex(picks).k,
        amplitude=_apex(picks).amplitude,
    )
    limits = replace(
        _limits(reverse_ns, reference_n),
        stop_amplitude=CONFIG.track_stop_rel * apex.amplitude,
    )
    return {
        step: picks_module._track_one_side(signal, apex, step, local_box, limits)
        for step in (-1, +1)
    }


def _v18_wings(line: Line, picks: HyperbolaPicks):
    return _replay_wings(line, picks, V18_REVERSE_NS, reference_n=1)


def _same_wings(picks: HyperbolaPicks, wings) -> bool:
    """``picks`` hold exactly the replayed wings around the apex."""
    left, _ = wings[-1]
    right, _ = wings[+1]
    replay_k = [k for _, k, _ in reversed(left)]
    replay_k.append(_apex(picks).k)
    replay_k.extend(k for _, k, _ in right)
    return np.array_equal(picks.k_pick, np.array(replay_k))


@pytest.fixture
def without_descent(monkeypatch: pytest.MonkeyPatch) -> None:
    """``extract_picks`` without T1: the v1.8 hint candidate."""

    def identity(signal, candidate, *args, **kwargs):
        return candidate

    monkeypatch.setattr(picks_module, "descend_apex", identity)


# ----------------------------------------------------------------- TK1


@pytest.mark.parametrize(
    ("f_c_hz", "expected_ns"),
    [(500e6, 0.2064), (400e6, 0.2580), (None, 0.2064)],
)
def test_tk1_reverse_tolerance(f_c_hz: float | None, expected_ns: float) -> None:
    assert reverse_tolerance_ns(f_c_hz, CONFIG) == pytest.approx(expected_ns, abs=1e-4)


# ----------------------------------------------------------------- TK2, TK3 (D1)


def test_tk2_jitter_006_recovers_both_wings(load_script) -> None:
    clean = _line()
    clean_picks = _picks(load_script, clean, _apex_hint(clean, PIPE_DEPTH_M))
    line = _line(shifts_ns=_jitter_shifts(0.06))
    picks = _picks(load_script, line, _apex_hint(line, PIPE_DEPTH_M))

    assert picks.is_enough_for_curvature(CONFIG)
    assert picks.n_left >= 20 and picks.n_right >= 20
    assert _curvature_ratio(picks) == pytest.approx(
        _curvature_ratio(clean_picks), abs=0.005
    )
    assert _curvature_ratio(clean_picks) == pytest.approx(CLEAN_RATIO, abs=1e-3)

    before = _v18_wings(line, picks)
    assert min(len(before[-1][0]), len(before[+1][0])) < CONFIG.min_side_picks


def test_tk3_jitter_012_needs_the_median_reference(load_script) -> None:
    line = _line(shifts_ns=_jitter_shifts(0.12))
    picks = _picks(load_script, line, _apex_hint(line, PIPE_DEPTH_M))

    assert picks.is_enough_for_curvature(CONFIG)

    previous_pick_reference = _replay_wings(
        line, picks, reverse_tolerance_ns(F_C_HZ, CONFIG), reference_n=1
    )
    shortest = min(len(wing) for wing, _ in previous_pick_reference.values())
    assert shortest < CONFIG.min_side_picks


# ----------------------------------------------------------------- TK4 (D4)


def test_tk4_rising_event_is_followed_at_most_three_traces(load_script) -> None:
    pipe_times = _pipe_times_ns(PIPE_DEPTH_M)
    meet_trace = APEX_TRACE + 6  # x_a + 0.12 m
    traces = np.arange(N_TRACES)
    layer_times = pipe_times[meet_trace] - 0.15 * (traces - meet_trace)
    layer_amplitudes = np.where(traces >= meet_trace - 2, 1.5, 0.0)
    line = _line(extra=(Event(layer_times, layer_amplitudes),))
    clean = _line()
    picks = _picks(load_script, line, _apex_hint(line, PIPE_DEPTH_M))
    clean_picks = _picks(load_script, clean, _apex_hint(clean, PIPE_DEPTH_M))

    right = picks.trace_idx > _apex(picks).trace
    pick_times = picks.k_pick[right] * DT_NS
    to_pipe = np.abs(pick_times - (T_SRC_NS + pipe_times[picks.trace_idx[right]]))
    to_layer = np.abs(pick_times - (T_SRC_NS + layer_times[picks.trace_idx[right]]))
    n_on_layer = int(np.sum(to_layer < to_pipe - INTRUDER_MARGIN_NS))

    assert n_on_layer <= 3
    assert _curvature_ratio(picks) == pytest.approx(
        _curvature_ratio(clean_picks), abs=0.01
    )
    assert picks.stop_right in {TrackStop.REVERSE, TrackStop.REVERSE_TOP}


# ----------------------------------------------------------------- TK5 (D3)


def test_tk5_crossing_reflector_picks_unchanged(load_script) -> None:
    shallower_m = PIPE_DEPTH_M - 0.06
    lateral = _offsets_m() - 0.35
    crossing_times = 2.0 / VELOCITY * np.hypot(shallower_m, lateral)
    crossing_amplitudes = (shallower_m / np.hypot(shallower_m, lateral)) ** 4
    line = _line(extra=(Event(crossing_times, crossing_amplitudes),))
    picks = _picks(load_script, line, _apex_hint(line, PIPE_DEPTH_M))

    assert APEX_DESCENDED not in picks.flags
    assert _same_wings(picks, _v18_wings(line, picks))
    assert (picks.n_left, picks.n_right) == (30, 8)


# ----------------------------------------------------------------- TK6 (D2)


def _truncated_hint_line() -> tuple[Line, ApexHint]:
    """D2: d 0.25 m, hint 3 traces right, its window top hides traces j_a..j_a+2."""
    depth_m = 0.25
    line = _line(depth_m=depth_m)
    assert line.pipe_times_ns is not None
    k_second_neighbour = (T_SRC_NS + line.pipe_times_ns[APEX_TRACE + 2]) / DT_NS
    window_top = math.floor(k_second_neighbour + 0.5)
    hint_sample = window_top + 0.5 * PERIOD_NS / DT_NS - 0.5
    return line, ApexHint(trace=APEX_TRACE + 3, sample=hint_sample)


def _t0_error_ns(line: Line, picks: HyperbolaPicks, depth_m: float) -> float:
    vertex = fixed_velocity_apex(picks, CONFIG)
    zero_offset_ns = _t_zero_ns(line) - T_SRC_NS
    return vertex.t0_ns - (2.0 / VELOCITY * depth_m - zero_offset_ns)


def test_tk6_truncated_hint_descends_to_the_apex(load_script) -> None:
    line, hint = _truncated_hint_line()
    picks = _picks(load_script, line, hint)

    assert _apex(picks).trace == APEX_TRACE
    assert APEX_DESCENDED in picks.flags
    assert abs(_t0_error_ns(line, picks, 0.25)) <= 0.005


def test_tk6_control_without_descent_the_apex_is_three_traces_off(
    load_script, without_descent
) -> None:
    line, hint = _truncated_hint_line()
    picks = _picks(load_script, line, hint)

    assert _apex(picks).trace == APEX_TRACE + 3
    assert APEX_DESCENDED not in picks.flags


# ----------------------------------------------------------------- TK7 (D0)


def test_tk7_clean_apex_hint_changes_nothing(load_script) -> None:
    line = _line()
    picks = _picks(load_script, line, _apex_hint(line, PIPE_DEPTH_M))

    assert APEX_DESCENDED not in picks.flags
    assert _same_wings(picks, _v18_wings(line, picks))
    assert (picks.n_left, picks.n_right) == (27, 27)


# ----------------------------------------------------------------- TK8


def test_tk8_descent_stays_inside_the_hint_traces(load_script) -> None:
    """A ridge that keeps rising to the right (no apex) stops at ``j_h + 3``."""
    traces = np.arange(N_TRACES)
    ridge_half_traces = 20
    centre_time_ns = 2.0 / VELOCITY * PIPE_DEPTH_M
    ridge_times = centre_time_ns - 0.1 * (traces - APEX_TRACE)
    ridge_amplitudes = np.where(
        np.abs(traces - APEX_TRACE) <= ridge_half_traces, 1.0, 0.0
    )
    line = _line(depth_m=None, extra=(Event(ridge_times, ridge_amplitudes),))
    # Late hint: the window top sits 1 sample above the ridge at j_h, so the
    # earlier traces to the right are hidden and the candidate starts at j_h.
    late_ns = 0.5 * PERIOD_NS - DT_NS
    hint = ApexHint(
        trace=APEX_TRACE,
        sample=(T_SRC_NS + centre_time_ns + late_ns) / DT_NS,
    )
    picks = _picks(load_script, line, hint)

    assert APEX_DESCENDED in picks.flags
    assert _apex(picks).trace == APEX_TRACE + CONFIG.hint_traces


# ----------------------------------------------------------------- TK9


def test_tk9_no_hint_path_never_descends(
    load_script, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []

    def recording(signal, candidate, *args, **kwargs):
        calls.append(candidate)
        return candidate

    monkeypatch.setattr(picks_module, "descend_apex", recording)
    line = _line(shifts_ns=_jitter_shifts(0.06))
    picks = _picks(load_script, line, hint=None)

    assert picks.apex is not None
    assert calls == []
    assert APEX_DESCENDED not in picks.flags


# ----------------------------------------------------------------- TK10


def test_tk10_stop_reasons_on_synthetic_lines(load_script) -> None:
    clean = _line()
    clean_picks = _picks(load_script, clean, _apex_hint(clean, PIPE_DEPTH_M))
    wing_ends = {TrackStop.AMPLITUDE, TrackStop.BOX_EDGE}
    assert clean_picks.stop_left in wing_ends
    assert clean_picks.stop_right in wing_ends

    jittered = _line(shifts_ns=_jitter_shifts(0.06))
    picks = _picks(load_script, jittered, _apex_hint(jittered, PIPE_DEPTH_M))
    before = _v18_wings(jittered, picks)
    cut_wing, cut_stop = min(before.values(), key=lambda wing: len(wing[0]))
    assert len(cut_wing) < CONFIG.min_side_picks
    assert cut_stop == TrackStop.REVERSE


def test_tk10_no_scale_has_no_stop_reason(load_script) -> None:
    line = _line()
    hint = _apex_hint(line, PIPE_DEPTH_M)
    picks = extract_picks(
        line.inputs.branch.background_removed,
        _box(load_script, line, hint),
        line.inputs.zero,
        dt_ns=None,
        dx_m=None,
        f_c_hz=F_C_HZ,
        config=CONFIG,
    )

    assert picks.apex is not None
    assert (picks.stop_left, picks.stop_right) == (None, None)


def _single_peak_signal(peak_k: float, n_samples: int = 100) -> np.ndarray:
    """Two columns: column 0 holds the apex at k 50, column 1 one Gaussian peak."""
    k = np.arange(n_samples, dtype=float)
    signal = np.zeros((n_samples, 2))
    signal[:, 0] = np.exp(-0.5 * ((k - 50.0) / 2.0) ** 2)
    signal[:, 1] = np.exp(-0.5 * ((k - peak_k) / 2.0) ** 2)
    return signal


@pytest.mark.parametrize(
    ("peak_k", "back_samples", "expected"),
    [
        (44.0, 3.0, TrackStop.REVERSE_TOP),  # window [47, ...]: peak above the top
        (46.0, 6.0, TrackStop.REVERSE),  # window [44, ...]: interior, too early
        (50.0, 3.0, TrackStop.BOX_EDGE),  # accepted, then the box ends
    ],
)
def test_tk10_stop_reason_of_one_step(
    peak_k: float, back_samples: float, expected: TrackStop
) -> None:
    signal = _single_peak_signal(peak_k)
    apex = ApexCandidate(trace=0, k=50.0, amplitude=1.0)
    limits = picks_module._TrackLimits(
        back_samples=back_samples,
        forward_samples=5.0,
        reverse_samples=2.0,
        reverse_ref_n=3,
        stop_amplitude=0.2,
        max_steps=5,
    )
    box = GridBox(i0=0, i1=1, k0=0, k1=99)

    _, stop = picks_module._track_one_side(signal, apex, +1, box, limits)

    assert stop == expected


def test_tk10_amplitude_window_and_aperture_stops() -> None:
    signal = _single_peak_signal(50.0)
    apex = ApexCandidate(trace=0, k=50.0, amplitude=1.0)
    box = GridBox(i0=0, i1=1, k0=0, k1=99)
    limits = picks_module._TrackLimits(
        back_samples=3.0,
        forward_samples=5.0,
        reverse_samples=2.0,
        reverse_ref_n=3,
        stop_amplitude=0.2,
        max_steps=1,
    )

    faded = replace(limits, stop_amplitude=2.0)
    at_record_end = ApexCandidate(trace=0, k=99.0, amplitude=1.0)
    no_window = replace(limits, forward_samples=0.0, back_samples=0.0)

    assert picks_module._track_one_side(signal, apex, +1, box, limits)[1] == (
        TrackStop.A_MAX
    )
    assert picks_module._track_one_side(signal, apex, +1, box, faded)[1] == (
        TrackStop.AMPLITUDE
    )
    assert picks_module._track_one_side(signal, at_record_end, +1, box, no_window)[
        1
    ] == (TrackStop.WINDOW)


# ----------------------------------------------------------------- TK11-TK14 (v1.9.2)

PURE_N_COLUMNS = 7
PURE_HINT_TRACE = 3
PURE_FLOOR_AMPLITUDE = 0.2
RELATIVE_TOL = 1e-12


def _triangle_signal(n_samples: int, peaks: dict[int, tuple[int, float]]) -> np.ndarray:
    """Zero signal with ``s[k-1] = a/2, s[k] = a, s[k+1] = a/2`` per column."""
    signal = np.zeros((n_samples, PURE_N_COLUMNS))
    for column, (k, amplitude) in peaks.items():
        signal[k - 1 : k + 2, column] = (0.5 * amplitude, amplitude, 0.5 * amplitude)
    return signal


def _descend(
    signal: np.ndarray,
    candidate: ApexCandidate,
    box: GridBox,
    back_samples: float,
    floor_amplitude: float = PURE_FLOOR_AMPLITUDE,
) -> ApexCandidate:
    return picks_module.descend_apex(
        signal,
        candidate,
        box,
        ApexHint(trace=PURE_HINT_TRACE, sample=candidate.k),
        hint_traces=CONFIG.hint_traces,
        back_samples=back_samples,
        floor_amplitude=floor_amplitude,
    )


def _as_tuple(candidate: ApexCandidate) -> tuple[int, float, float]:
    return candidate.trace, candidate.k, candidate.amplitude


@pytest.mark.parametrize(
    ("floor_amplitude", "expected"),
    [
        (PURE_FLOOR_AMPLITUDE, (4, 38.0, 0.5)),
        # Control: a floor of 0.1 (what a floor re-based on the current
        # candidate, 0.2 x 0.5, would give) walks on to column 5.
        (0.1, (5, 36.0, 0.15)),
    ],
)
def test_tk11_descent_floor_is_the_hint_candidate_value(
    floor_amplitude: float, expected: tuple[int, float, float]
) -> None:
    """TK11 (a): the floor stays ``0.2 x`` the pre-descent candidate value."""
    signal = _triangle_signal(64, {3: (40, 1.0), 4: (38, 0.5), 5: (36, 0.15)})
    candidate = ApexCandidate(trace=PURE_HINT_TRACE, k=40.0, amplitude=1.0)
    box = GridBox(i0=0, i1=6, k0=10, k1=60)

    descended = _descend(signal, candidate, box, 3.0, floor_amplitude)

    assert _as_tuple(descended) == pytest.approx(expected, abs=1e-12)


@dataclass
class _DescentRecord:
    """Arguments and result of ``descend_apex`` and the wing-tracking calls."""

    descent_inputs: list[tuple[ApexCandidate, float]]
    descent_results: list[ApexCandidate]
    tracked: list[tuple[ApexCandidate, float]]


@pytest.fixture
def recorded_tracking(monkeypatch: pytest.MonkeyPatch) -> _DescentRecord:
    """Wrap ``descend_apex`` and ``_track_one_side`` without changing them."""
    record = _DescentRecord([], [], [])
    real_descend = picks_module.descend_apex
    real_track = picks_module._track_one_side

    def descend(signal, candidate, box, hint, **kwargs):
        record.descent_inputs.append((candidate, kwargs["floor_amplitude"]))
        result = real_descend(signal, candidate, box, hint, **kwargs)
        record.descent_results.append(result)
        return result

    def track(signal, candidate, step, box, limits):
        record.tracked.append((candidate, limits.stop_amplitude))
        return real_track(signal, candidate, step, box, limits)

    monkeypatch.setattr(picks_module, "descend_apex", descend)
    monkeypatch.setattr(picks_module, "_track_one_side", track)
    return record


def test_tk11_descent_floor_on_the_truncated_hint_line(
    load_script, recorded_tracking: _DescentRecord
) -> None:
    """TK11 (b): on the TK6 line the floor is ``track_stop_rel x`` the hint value."""
    line, hint = _truncated_hint_line()
    picks = _picks(load_script, line, hint)

    ((hint_candidate, floor_amplitude),) = recorded_tracking.descent_inputs
    assert floor_amplitude == pytest.approx(
        CONFIG.track_stop_rel * hint_candidate.amplitude, rel=RELATIVE_TOL
    )
    assert APEX_DESCENDED in picks.flags


def test_tk12_track_stop_uses_the_descended_value(
    load_script, recorded_tracking: _DescentRecord
) -> None:
    """TK12: both wings start from the descended candidate and use its value."""
    line, hint = _truncated_hint_line()
    picks = _picks(load_script, line, hint)

    ((hint_candidate, _),) = recorded_tracking.descent_inputs
    (descended,) = recorded_tracking.descent_results
    # Premise: the two values differ enough for the test to tell them apart.
    relative_change = abs(hint_candidate.amplitude - descended.amplitude)
    assert relative_change / descended.amplitude > 0.01
    assert _apex(picks).trace == APEX_TRACE
    assert len(recorded_tracking.tracked) == 2
    for tracked_candidate, stop_amplitude in recorded_tracking.tracked:
        assert tracked_candidate is descended
        assert stop_amplitude == pytest.approx(
            CONFIG.track_stop_rel * descended.amplitude, rel=RELATIVE_TOL
        )


TK13_BACK_SAMPLES = 4.0


def test_tk13_descent_ignores_the_box_top() -> None:
    """TK13 (a): the descent goes above the box top ``k0`` 20."""
    signal = _triangle_signal(40, {4: (19, 0.8)})
    candidate = ApexCandidate(trace=PURE_HINT_TRACE, k=22.0, amplitude=1.0)
    box = GridBox(i0=0, i1=6, k0=20, k1=38)

    descended = _descend(signal, candidate, box, TK13_BACK_SAMPLES)

    assert (descended.trace, descended.k) == (4, pytest.approx(19.0, abs=1e-12))


def test_tk13_descent_near_the_record_top() -> None:
    """TK13 (b): window ``[1, 8]`` holds an interior maximum at k 2."""
    signal = _triangle_signal(40, {4: (2, 0.8)})
    candidate = ApexCandidate(trace=PURE_HINT_TRACE, k=4.0, amplitude=1.0)
    box = GridBox(i0=0, i1=6, k0=0, k1=38)

    descended = _descend(signal, candidate, box, TK13_BACK_SAMPLES)

    assert (descended.trace, descended.k) == (4, pytest.approx(2.0, abs=1e-12))


def test_tk13_no_descent_next_to_the_first_record_sample() -> None:
    """TK13 (c): a peak at k 1 is the window end, so nothing moves."""
    signal = _triangle_signal(40, {4: (1, 0.8)})
    candidate = ApexCandidate(trace=PURE_HINT_TRACE, k=4.0, amplitude=1.0)
    box = GridBox(i0=0, i1=6, k0=0, k1=38)

    descended = _descend(signal, candidate, box, TK13_BACK_SAMPLES)

    assert descended is candidate


def test_tk14_descent_stays_inside_the_hint_traces_to_the_left(load_script) -> None:
    """TK14: the TK8 ridge mirrored (earlier to the left) stops at ``j_h - 3``."""
    traces = np.arange(N_TRACES)
    ridge_half_traces = 20
    centre_time_ns = 2.0 / VELOCITY * PIPE_DEPTH_M
    ridge_times = centre_time_ns + 0.1 * (traces - APEX_TRACE)
    ridge_amplitudes = np.where(
        np.abs(traces - APEX_TRACE) <= ridge_half_traces, 1.0, 0.0
    )
    line = _line(depth_m=None, extra=(Event(ridge_times, ridge_amplitudes),))
    late_ns = 0.5 * PERIOD_NS - DT_NS
    hint = ApexHint(
        trace=APEX_TRACE,
        sample=(T_SRC_NS + centre_time_ns + late_ns) / DT_NS,
    )
    picks = _picks(load_script, line, hint)

    assert APEX_DESCENDED in picks.flags
    assert _apex(picks).trace == APEX_TRACE - CONFIG.hint_traces
