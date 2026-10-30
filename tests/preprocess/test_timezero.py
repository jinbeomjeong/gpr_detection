"""Unit tests of time-zero detection and selection (spec_depth_v1 v1.2 section 4).

Z2-Z7 are the section 11-1 zero tests; Z1 lives in ``test_zero_dewow_z1.py``.
"""

import numpy as np
import pytest

from gpr_detection.io.radargram import SPEED_OF_LIGHT_M_PER_NS
from gpr_detection.preprocess.filters import prepare_fitting_branch
from gpr_detection.preprocess.timezero import (
    GroundZeroShift,
    ZeroConfig,
    classify_meta_t0,
    detect_zero_events,
    direct_wave_delay_ns,
    find_time_zero,
    given_zero,
)
from synthetic_bscan import (
    DIRECT_WAVE_DELAY_NS,
    T_SRC_NS,
    BScanSpec,
    cylinder_times,
    make_radargram,
    ricker,
)

PIPE_DEPTH_M = 1.0
"""Pipe far below the zero window; only the early events matter here."""

SURFACE_LATE_MAX_NS = 0.2
"""Z7: the surface peak may be late by up to 0.2 ns after (1)-(3)."""

SHALLOW_TAU_NS = 3.5
"""Shallow flat reflection inside the ground zero window (Z3, Z5), 2.8 ns after
the direct wave; opposite polarity keeps the two envelope peaks apart."""


def _air_spec(air_gap_m: float = 0.3, f_c_hz: float = 500e6, **fields) -> BScanSpec:
    """Air-coupled spec; the pipe gets a vertical air leg (outside the zero window)."""
    spec = BScanSpec(
        f_c_hz=f_c_hz, air_gap_m=air_gap_m, offset_m=0.1, d_top_m=PIPE_DEPTH_M, **fields
    )
    soil_ns = cylinder_times(spec.x_m, spec.x0_m, PIPE_DEPTH_M, spec.velocity, 0.0)
    spec.pipe_times_ns = soil_ns + 2.0 * air_gap_m / SPEED_OF_LIGHT_M_PER_NS
    return spec


def _surface_geometric_ns(spec: BScanSpec) -> float:
    """``t_geom = t_src + 2 sqrt(h^2 + s^2/4) / c``."""
    path_m = 2.0 * np.hypot(spec.air_gap_m, spec.offset_m / 2)
    return T_SRC_NS + path_m / SPEED_OF_LIGHT_M_PER_NS


def _zero_of(radargram, **kwargs):
    filtered = prepare_fitting_branch(radargram).filtered
    return find_time_zero(radargram, filtered, **kwargs)


def _assert_surface_zero_z7(zero, spec: BScanSpec) -> None:
    """Z7: two events, surface peak 0-0.2 ns late, h_est 0.30-0.36 m.

    The late shift is the direct-wave interference bias, not a processing
    fault (``docs/processing/surface_zero_shift.csv``).
    """
    assert zero.zero_rule == "air_surface_late"
    assert len(zero.events_ns) == 2
    late_by_ns = zero.t_zero_ns - _surface_geometric_ns(spec)
    assert 0.0 < late_by_ns <= SURFACE_LATE_MAX_NS
    assert 0.30 <= zero.h_est_m <= 0.36


def _two_event_radargram(gap_ns: float):
    """Direct event plus an opposite-polarity (-2/3) event ``gap_ns`` later."""
    time_ns = np.arange(800)[:, None] * 0.05
    trace = ricker(time_ns - 3.0, 500e6) - (2.0 / 3.0) * ricker(
        time_ns - 3.0 - gap_ns, 500e6
    )
    data = np.tile(trace, (1, 20)).astype(np.float32)
    return make_radargram(BScanSpec(), data=data, f_c_hz=500e6, air_gap="air")


# ------------------------------------------------------------------ rules


def test_ground_zero_is_the_direct_wave_peak() -> None:
    zero = _zero_of(make_radargram(BScanSpec()))
    assert zero.zero_rule == "ground_direct_peak"
    assert zero.zero_event_label == "direct_wave"
    assert zero.t_zero_ns == pytest.approx(T_SRC_NS + DIRECT_WAVE_DELAY_NS, abs=0.05)
    assert zero.k_zero == pytest.approx(zero.t_zero_ns / 0.05)
    assert zero.sigma_zero_ns == 0.15
    assert zero.bias_notes == ("ground_direct_zero: shallow 0.53-1.03 ns",)


def test_z7_air_gap_with_two_events_uses_the_late_surface_reflection() -> None:
    spec = _air_spec()
    zero = _zero_of(make_radargram(spec))
    _assert_surface_zero_z7(zero, spec)
    assert zero.zero_event_label == "surface_reflection"
    assert zero.sigma_zero_ns == 0.3
    assert "direct-wave interference" in zero.bias_notes[0]


def test_small_air_gap_merges_into_a_single_event() -> None:
    zero = _zero_of(make_radargram(_air_spec(air_gap_m=0.1, f_c_hz=400e6)))
    assert zero.zero_rule == "air_single_event"
    assert zero.zero_event_label == "merged"
    assert zero.h_est_m is None
    assert zero.sigma_zero_ns == 0.25


def test_single_event_bias_note_gives_the_numeric_upper_bound() -> None:
    zero = _zero_of(make_radargram(_air_spec(air_gap_m=0.1)))
    # 0.375 * c / 500 MHz / sqrt(9) = 0.0749 m
    assert "deep by up to 0.075 m" in zero.bias_notes[0]


def test_single_event_bias_note_without_eps_gives_the_formula() -> None:
    radargram = make_radargram(
        _air_spec(air_gap_m=0.1), eps_r_meta=None, eps_r_meta_source=""
    )
    note = _zero_of(radargram).bias_notes[0]
    assert note.endswith("deep by up to 0.375 c / (f_c sqrt(eps_r))")


def test_empty_record_fails() -> None:
    radargram = make_radargram(BScanSpec(), data=np.zeros((400, 20), np.float32))
    zero = _zero_of(radargram)
    assert zero.failed
    assert zero.t_zero_ns is None
    assert zero.zero_event_label is None
    assert "zero_failed" in zero.flags


def test_processed_amplitudes_use_sample_zero() -> None:
    zero = _zero_of(make_radargram(BScanSpec(), amplitude_processed=True))
    assert zero.zero_rule == "meta_sample0"
    assert zero.t_zero_ns == 0.0
    assert {"zero_unverified", "amplitude_processed"} <= set(zero.flags)


def test_unknown_air_gap_with_two_events_is_flagged() -> None:
    radargram = make_radargram(_air_spec(), air_gap="unknown")
    zero = _zero_of(radargram)
    assert zero.zero_rule == "ground_direct_peak"
    assert {"air_gap_unknown", "air_gap_unknown_multi_event"} <= set(zero.flags)


def test_no_scale_input_gives_sample_position_only() -> None:
    radargram = make_radargram(BScanSpec(), dt_ns=None, dx_m=None, f_c_hz=None)
    zero = _zero_of(radargram)
    assert zero.t_zero_ns is None
    assert zero.k_zero == pytest.approx((T_SRC_NS + DIRECT_WAVE_DELAY_NS) / 0.05, abs=2)
    assert "zero_px_unverified" in zero.flags


def test_given_zero_and_invalid_pick_mode() -> None:
    zero = given_zero(3.0, 0.05)
    assert zero.zero_rule == "given"
    assert zero.k_zero == pytest.approx(60.0)
    with pytest.raises(ValueError, match="pick_mode"):
        given_zero(3.0, 0.05, pick_mode="onset")


# ------------------------------------------------------------------ detection


@pytest.mark.parametrize(("gap_ns", "n_events"), [(1.2, 1), (1.8, 2)])
def test_z2_opposite_polarity_events_split_from_0_75_periods(gap_ns, n_events) -> None:
    zero = _zero_of(_two_event_radargram(gap_ns))
    assert len(zero.events_ns) == n_events


def test_z6_surface_reflection_alone_keeps_its_geometric_time() -> None:
    spec = _air_spec(direct_amplitude=0.0, snr_db=None)
    zero = _zero_of(make_radargram(spec))
    assert len(zero.events_ns) == 1
    assert abs(zero.events_ns[0] - _surface_geometric_ns(spec)) <= 0.01


def test_z7_excluding_band_traces_keeps_the_surface_reflection() -> None:
    """UB14: a shallow band inside the zero window is taken as surface reflection."""
    spec = _air_spec()
    band_traces = np.arange(int(0.6 * spec.n_traces))
    band_amplitude = np.zeros(spec.n_traces)
    band_amplitude[band_traces] = 2.0
    band_tau_ns = _surface_geometric_ns(spec) - T_SRC_NS + 2 * 0.1 / spec.velocity
    spec.extra_events = [(np.full(spec.n_traces, band_tau_ns), band_amplitude)]
    radargram = make_radargram(spec)

    without_mask = _zero_of(radargram)
    with_mask = _zero_of(radargram, exclude_traces=band_traces)

    assert without_mask.t_zero_ns == pytest.approx(T_SRC_NS + band_tau_ns, abs=0.15)
    _assert_surface_zero_z7(with_mask, spec)
    assert with_mask.excluded_traces == len(band_traces)


def test_excluding_every_trace_is_ignored_and_flagged() -> None:
    radargram = make_radargram(BScanSpec())
    filtered = prepare_fitting_branch(radargram).filtered
    mask = np.ones(radargram.n_traces, dtype=bool)
    events = detect_zero_events(filtered, 0.05, 400e6, exclude_traces=mask)
    assert "zero_band_overlap" in events.flags
    assert events.excluded_traces == 0


# ------------------------------------------------------------------ window QC


def test_long_profile_tracks_the_zero_between_windows() -> None:
    zero = _zero_of(make_radargram(BScanSpec(n_traces=700)))
    assert zero.window_std_ns is not None
    assert zero.window_std_ns < 0.05
    assert not {"zero_window_missing", "zero_window_drift"} & set(zero.flags)


def test_z5_window_qc_follows_the_direct_wave_not_the_last_peak() -> None:
    spec = BScanSpec(n_traces=700)
    shallow_amplitude = np.zeros(spec.n_traces)
    shallow_amplitude[:350] = -1.5
    spec.extra_events = [(np.full(spec.n_traces, SHALLOW_TAU_NS), shallow_amplitude)]
    zero = _zero_of(make_radargram(spec))
    assert zero.zero_rule == "ground_direct_peak"
    assert zero.window_std_ns < 0.05
    assert "zero_window_event_count_mismatch" in zero.flags


# ------------------------------------------------------------------ metadata zeros


def test_meta_labels_follow_the_air_two_event_rule() -> None:
    labels = classify_meta_t0(
        {"est": 3.2, "act": 5.1, "far": 9.0}, "air_surface_late", (3.3, 5.0)
    )
    assert labels == {
        "est": (3.2, "direct_wave"),
        "act": (5.1, "surface_reflection"),
        "far": (9.0, "unclassified"),
    }


def test_meta_labels_single_event_use_merged_with_half_period_tolerance() -> None:
    labels = classify_meta_t0(
        {"lobe": 4.78, "far": 4.0 + 1.1}, "air_single_event", (4.0,), f_c_hz=500e6
    )
    assert labels == {"lobe": (4.78, "merged"), "far": (5.1, "unclassified")}


def test_meta_labels_ground_rule_knows_only_the_direct_wave() -> None:
    labels = classify_meta_t0({"late": 6.0}, "ground_direct_peak", (3.7, 6.0))
    assert labels == {"late": (6.0, "unclassified")}


def test_meta_zeros_from_the_radargram_are_labelled() -> None:
    radargram = make_radargram(
        _air_spec(), t0_candidates_ns={"t0_sample_est": T_SRC_NS + 0.3}
    )
    zero = _zero_of(radargram)
    assert zero.meta_t0["t0_sample_est"][1] == "direct_wave"


def test_z3_ground_late_peak_meta_zero_is_unclassified() -> None:
    spec = BScanSpec()
    spec.extra_events = [(np.full(spec.n_traces, SHALLOW_TAU_NS), -1.5)]
    first_pass = _zero_of(make_radargram(spec))
    assert len(first_pass.events_ns) == 2
    candidates = {
        "on_late_peak": first_pass.events_ns[-1],
        "on_direct": first_pass.events_ns[0],
    }
    zero = _zero_of(make_radargram(spec, t0_candidates_ns=candidates))
    assert zero.meta_t0["on_late_peak"][1] == "unclassified"
    assert zero.meta_t0["on_direct"][1] == "direct_wave"


@pytest.mark.parametrize("lobe_offset_ns", [-0.78, 0.78])
def test_z4_single_air_event_meta_zero_on_a_lobe_is_merged(lobe_offset_ns) -> None:
    spec = _air_spec(air_gap_m=0.1)
    peak_ns = _zero_of(make_radargram(spec)).t_zero_ns
    candidates = {"t0_sample_est": peak_ns + lobe_offset_ns}
    zero = _zero_of(make_radargram(spec, t0_candidates_ns=candidates))
    assert zero.zero_rule == "air_single_event"
    assert zero.meta_t0["t0_sample_est"][1] == "merged"


# ------------------------------------------------------------------ table shift


def test_direct_wave_delay_table_interpolates_in_sqrt_eps() -> None:
    assert direct_wave_delay_ns(9.0, 400e6) == pytest.approx(0.725)
    assert direct_wave_delay_ns(16.0, 500e6) == pytest.approx(1.03)
    assert direct_wave_delay_ns(6.25, 400e6) == pytest.approx((0.53 + 0.725) / 2)


def test_ground_zero_shift_subtracts_the_table_delay() -> None:
    radargram = make_radargram(BScanSpec())
    config = ZeroConfig(ground_zero_shift=GroundZeroShift.SYNTHETIC_TABLE)
    plain = _zero_of(radargram)
    shifted = _zero_of(radargram, config=config)
    assert shifted.zero_rule == "ground_direct_shifted"
    assert plain.t_zero_ns - shifted.t_zero_ns == pytest.approx(0.725)


@pytest.mark.parametrize(
    "overrides",
    [
        {"eps_r_meta": None, "eps_r_meta_source": ""},
        {"f_c_hz": 300e6},
        {"eps_r_meta": 20.0},
        {"antenna_offset_m": 0.2},
    ],
)
def test_ground_zero_shift_outside_the_table_range_falls_back(overrides) -> None:
    radargram = make_radargram(BScanSpec(), **overrides)
    config = ZeroConfig(ground_zero_shift=GroundZeroShift.SYNTHETIC_TABLE)
    zero = _zero_of(radargram, config=config)
    assert zero.zero_rule == "ground_direct_peak"
    assert "ground_zero_shift_unavailable" in zero.flags
