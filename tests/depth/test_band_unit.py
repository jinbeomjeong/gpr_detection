"""Unit tests of ``pipe_band`` depth (spec_band_depth_v1 section 9-1, UB1-UB22).

Analytic B-scans from ``synthetic_bscan``: 400 MHz Ricker, dt 0.05 ns, dx
0.02 m, SNR 20 dB, eps_r 9, d_top 0.6 m, R 0.05 m, ground-coupled, 200
traces. The band is an extra event with a per-trace time ``tau(x)`` and a
per-trace amplitude (0 outside the band). The zero is the transmit time
(``given_zero``) unless a test says otherwise; the velocity is ``meta_fixed``.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from gpr_detection.depth.band import (
    LAYER_AMBIGUITY_NOTE,
    PROTECTIVE_PLATE_NOTE,
    band_exclude_traces,
    estimate_band_depths,
    extract_band_picks,
    group_band_boxes,
    pixel_limits,
)
from gpr_detection.depth.config import DEFAULT_BAND_CONFIG, BandConfig
from gpr_detection.depth.velocity import VelocityEstimate, select_band_velocity
from gpr_detection.io.radargram import SPEED_OF_LIGHT_M_PER_NS, GridBox
from gpr_detection.preprocess.filters import prepare_fitting_branch, remove_background
from gpr_detection.preprocess.signals import pick_signal
from gpr_detection.preprocess.timezero import find_time_zero, given_zero
from synthetic_bscan import DT_NS, DX_M, F_C_HZ, T_SRC_NS, BScanSpec, make_radargram

N_TRACES = 200
X_M = np.arange(N_TRACES) * DX_M
D_TOP_M = 0.6
RADIUS_M = 0.05
PERIOD_NS = 1e9 / F_C_HZ
FAR_PIPE_NS = 1000.0
"""The generator's own pipe is pushed beyond the record; the band is an extra event."""

V9 = SPEED_OF_LIGHT_M_PER_NS / 3.0
TAU_FLAT_NS = 2 * D_TOP_M / V9
BAND_TRACES = slice(60, 141)


def _amplitude(traces: slice, value: float = 1.0) -> np.ndarray:
    amplitude = np.zeros(N_TRACES)
    amplitude[traces] = value
    return amplitude


def _radargram(events, **overrides):
    """Ground profile with ``events = [(tau per trace, amplitude per trace)]``."""
    spec = BScanSpec(
        n_traces=N_TRACES,
        radius_m=RADIUS_M,
        pipe_times_ns=np.full(N_TRACES, FAR_PIPE_NS),
    )
    spec.extra_events = list(events)
    fields = {"pipe_radius_m": RADIUS_M, **overrides}
    return make_radargram(spec, **fields)


def _box(
    i0: int, i1: int, tau_min: float, tau_max: float, dt: float = DT_NS
) -> GridBox:
    """Detection-like box: 0.4 period above to 0.8 period below the band."""
    period = PERIOD_NS * dt / DT_NS
    return GridBox(
        i0=i0,
        i1=i1,
        k0=math.floor((T_SRC_NS * dt / DT_NS + tau_min - 0.4 * period) / dt),
        k1=math.ceil((T_SRC_NS * dt / DT_NS + tau_max + 0.8 * period) / dt),
    )


def _estimate(radargram, boxes, cfg=DEFAULT_BAND_CONFIG, pool=(), t_zero_ns=T_SRC_NS):
    zero = given_zero(t_zero_ns, radargram.dt_ns, pick_mode=cfg.pick_mode)
    return estimate_band_depths(radargram, boxes, zero, pool, cfg=cfg)


def _flat_band(value: float = 1.0):
    return (np.full(N_TRACES, TAU_FLAT_NS), _amplitude(BAND_TRACES, value))


def _flat_box() -> GridBox:
    return _box(60, 140, TAU_FLAT_NS, TAU_FLAT_NS)


def test_ub1_flat_band_depth() -> None:
    (result,) = _estimate(_radargram([_flat_band()]), [_flat_box()])
    assert result.status == "ok"
    assert result.target_class == "pipe_band"
    assert result.band.model == "flat"
    assert result.v_rule == "meta_fixed"
    assert result.depth_top_m == pytest.approx(D_TOP_M, abs=0.02)
    assert result.depth_center_m == pytest.approx(result.depth_top_m + RADIUS_M)
    assert result.apex_trace is None
    assert result.band.bg_mode == "masked_median"
    assert result.band.coverage >= 0.95


def test_ub2_full_line_band_skips_background_removal() -> None:
    band = (np.full(N_TRACES, TAU_FLAT_NS), np.ones(N_TRACES))
    radargram = _radargram([band])
    box = _box(0, N_TRACES - 1, TAU_FLAT_NS, TAU_FLAT_NS)
    (result,) = _estimate(radargram, [box])
    assert result.band.bg_mode == "none"
    assert "bg_none" in result.flags
    assert result.depth_top_m == pytest.approx(D_TOP_M, abs=0.02)

    # A full-line median would erase the band. The trace mean measures the band
    # amplitude without the per-trace noise peaks.
    filtered = prepare_fitting_branch(radargram).filtered
    rows = box.sample_slice
    kept = pick_signal(filtered.mean(axis=1), "envelope")[rows].max()
    erased_section = remove_background(filtered)
    erased = pick_signal(erased_section.mean(axis=1), "envelope")[rows].max()
    assert erased / kept < 0.05


def test_ub3_sloped_band_grade_and_centre_depth() -> None:
    grade = 0.03
    alpha = math.atan(grade)
    centre_x = X_M[100]
    depth = D_TOP_M + grade * (X_M - centre_x)
    tau = 2 * depth * math.cos(alpha) / V9
    radargram = _radargram([(tau, _amplitude(BAND_TRACES))])
    (result,) = _estimate(radargram, [_box(60, 140, tau[60], tau[140])])
    assert result.band.model == "sloped"
    assert result.band.grade == pytest.approx(grade, abs=0.005)
    assert result.depth_top_m == pytest.approx(D_TOP_M, abs=0.02)
    assert len(result.band.depth_profile) == 3


@pytest.mark.parametrize("theta_deg", [10.0, 15.0, 20.0])
def test_ub4_oblique_crossing_gives_curved_model_and_angle(theta_deg: float) -> None:
    theta = math.radians(theta_deg)
    x0 = X_M[100]
    centre = D_TOP_M + RADIUS_M
    tau = (2 / V9) * (
        np.sqrt(centre**2 + ((X_M - x0) * math.sin(theta)) ** 2) - RADIUS_M
    )
    radargram = _radargram([(tau, _amplitude(slice(50, 151)))])
    (result,) = _estimate(radargram, [_box(50, 150, tau[100], tau[50])])
    assert result.band.model == "curved"
    assert result.band.theta_est_deg == pytest.approx(theta_deg, abs=2.0)
    assert result.depth_top_m == pytest.approx(D_TOP_M, abs=0.01)


def test_ub5_lateral_offset_is_reported_not_corrected() -> None:
    offset_y = 0.2
    centre = D_TOP_M + RADIUS_M
    tau = np.full(N_TRACES, (2 / V9) * (math.hypot(centre, offset_y) - RADIUS_M))
    radargram = _radargram([(tau, _amplitude(BAND_TRACES))])
    (result,) = _estimate(radargram, [_box(60, 140, tau[0], tau[0])])
    expected = math.hypot(centre, offset_y) - RADIUS_M
    assert result.depth_top_m == pytest.approx(expected, abs=0.01)
    assert result.band.lateral_bias_max_m > 0
    assert any(note.startswith("lateral_offset_unknown") for note in result.bias_notes)


def test_ub6_same_depth_fragments_merge_and_different_ones_do_not() -> None:
    continuous = (np.full(N_TRACES, TAU_FLAT_NS), _amplitude(slice(40, 151)))
    radargram = _radargram([continuous])
    boxes = [
        _box(40, 90, TAU_FLAT_NS, TAU_FLAT_NS),
        _box(97, 150, TAU_FLAT_NS, TAU_FLAT_NS),
    ]
    (merged,) = _estimate(radargram, boxes)
    assert merged.band.member_indices == (0, 1)
    assert merged.band.trace_range == (40, 150)
    # Coverage counts member traces only; the 6 gap-trace picks feed the fit.
    member_traces = (90 - 40 + 1) + (150 - 97 + 1)
    assert merged.band.coverage <= 1.0
    member_picks = round(merged.band.coverage * member_traces)
    assert merged.band.n_picks - member_picks == 6

    deeper = TAU_FLAT_NS + 2.0
    tau = np.where(np.arange(N_TRACES) >= 97, deeper, TAU_FLAT_NS)
    radargram = _radargram([(tau, _amplitude(slice(40, 151)))])
    boxes = [_box(40, 90, TAU_FLAT_NS, TAU_FLAT_NS), _box(97, 150, deeper, deeper)]
    separate = _estimate(radargram, boxes)
    assert len(separate) == 2


def test_group_band_boxes_requires_matching_ends() -> None:
    boxes = [GridBox(0, 10, 100, 120), GridBox(15, 30, 100, 120)]
    assert group_band_boxes(boxes, None) == [(0, 1)]
    edges = [(110.0, 110.0), (130.0, 130.0)]
    assert group_band_boxes(boxes, None, edge_samples=edges) == [(0,), (1,)]


def _sloped_tau(grade: float, centre_trace: int = 100) -> np.ndarray:
    """Band-top times of a sloped band (the UB3 formula)."""
    depth = D_TOP_M + grade * (X_M - X_M[centre_trace])
    return 2 * depth * math.cos(math.atan(grade)) / V9


def test_ub16_x_overlapping_fragments_of_a_flat_band_merge() -> None:
    band = (np.full(N_TRACES, TAU_FLAT_NS), _amplitude(slice(40, 181)))
    radargram = _radargram([band])
    boxes = [
        _box(40, 160, TAU_FLAT_NS, TAU_FLAT_NS),
        _box(120, 180, TAU_FLAT_NS, TAU_FLAT_NS),
    ]
    (result,) = _estimate(radargram, boxes)
    assert result.band.member_indices == (0, 1)
    assert result.band.trace_range == (40, 180)
    assert result.depth_top_m == pytest.approx(D_TOP_M, abs=0.02)


def test_ub17_contained_fragment_of_a_sloped_band_merges() -> None:
    """Facing ends 96 traces apart differ by ~1.9 ns; shared traces agree."""
    grade = 0.05
    tau = _sloped_tau(grade)
    radargram = _radargram([(tau, _amplitude(slice(30, 171)))])
    boxes = [_box(40, 160, tau[40], tau[160]), _box(60, 100, tau[60], tau[100])]
    results = _estimate(radargram, boxes)
    assert len(results) == 1, f"got {len(results)} results"
    (result,) = results
    assert result.band.member_indices == (0, 1)
    assert result.band.model == "sloped"
    assert result.band.grade == pytest.approx(grade, abs=0.005)


def test_ub18_overlapping_boxes_on_different_events_stay_apart() -> None:
    """A spans both bands; on the shared traces A picks tau1 and B picks tau2."""
    tau_deep = TAU_FLAT_NS + 6.0
    shallow = (np.full(N_TRACES, TAU_FLAT_NS), _amplitude(slice(40, 161)))
    deep = (np.full(N_TRACES, tau_deep), _amplitude(slice(100, 181)))
    radargram = _radargram([shallow, deep])
    boxes = [_box(40, 160, TAU_FLAT_NS, tau_deep), _box(100, 180, tau_deep, tau_deep)]
    results = _estimate(radargram, boxes)
    assert len(results) == 2
    assert all("fragment_of_group" in result.flags for result in results)


def test_ub19_default_no_scale_merge_gap() -> None:
    assert DEFAULT_BAND_CONFIG.merge_gap_px == 32


UB19_GAP_20_BOXES = (
    _box(40, 90, TAU_FLAT_NS, TAU_FLAT_NS),
    _box(111, 170, TAU_FLAT_NS, TAU_FLAT_NS),
)
UB19_GAP_40_BOXES = (
    _box(40, 80, TAU_FLAT_NS, TAU_FLAT_NS),
    _box(121, 170, TAU_FLAT_NS, TAU_FLAT_NS),
)


@pytest.mark.parametrize(
    ("boxes", "merge_gap_px", "expected_count"),
    [
        pytest.param(UB19_GAP_20_BOXES, None, 1, id="a-gap20-default"),
        pytest.param(UB19_GAP_40_BOXES, None, 2, id="b-gap40-default"),
        pytest.param(UB19_GAP_20_BOXES, 8, 2, id="c-gap20-G8"),
    ],
)
def test_ub19_no_scale_merge_gap_is_32_px(
    boxes: tuple[GridBox, ...], merge_gap_px: int | None, expected_count: int
) -> None:
    cfg = DEFAULT_BAND_CONFIG
    if merge_gap_px is not None:
        cfg = replace(cfg, merge_gap_px=merge_gap_px)
    band = (np.full(N_TRACES, TAU_FLAT_NS), _amplitude(slice(40, 171)))
    radargram = _radargram([band], dt_ns=None, dx_m=None, f_c_hz=None)
    zero = find_time_zero(radargram, prepare_fitting_branch(radargram).filtered)
    results = estimate_band_depths(radargram, list(boxes), zero, cfg=cfg)
    assert len(results) == expected_count, f"got {len(results)} results"
    assert all(result.status == "no_scale" for result in results)
    if expected_count == 1:
        assert results[0].band.member_indices == (0, 1)


def test_ub7_water_filled_pipe_keeps_the_top_reflection() -> None:
    bottom = (np.full(N_TRACES, TAU_FLAT_NS + 12.0), _amplitude(BAND_TRACES, 2.0))
    radargram = _radargram([_flat_band(), bottom])
    box = _box(60, 140, TAU_FLAT_NS, TAU_FLAT_NS + 12.0)
    (result,) = _estimate(radargram, [box])
    assert result.depth_top_m == pytest.approx(D_TOP_M, abs=0.02)


def test_ub8_surface_multiple_is_flagged_and_ignored() -> None:
    multiple = (np.full(N_TRACES, 2 * TAU_FLAT_NS), _amplitude(BAND_TRACES, 0.8))
    radargram = _radargram([_flat_band(), multiple])
    (result,) = _estimate(radargram, [_flat_box()])
    assert result.depth_top_m == pytest.approx(D_TOP_M, abs=0.02)
    assert "surface_multiple_candidate" in result.flags


def test_ub9_polarity_does_not_change_the_envelope_depth() -> None:
    (normal,) = _estimate(_radargram([_flat_band()]), [_flat_box()])
    (inverted,) = _estimate(_radargram([_flat_band(-1.0)]), [_flat_box()])
    assert abs(inverted.depth_top_m - normal.depth_top_m) <= 0.001


def test_ub10_band_in_the_zero_zone() -> None:
    tau = np.full(N_TRACES, 2 * 0.05 / V9)
    radargram = _radargram([(tau, _amplitude(BAND_TRACES))])
    (result,) = _estimate(radargram, [_box(60, 140, tau[0], tau[0])])
    assert result.status == "band_in_zero_zone"
    assert result.depth_top_m is None


def test_ub11_no_scale_gives_pixels_only() -> None:
    radargram = _radargram([_flat_band()], dt_ns=None, dx_m=None, f_c_hz=None)
    zero = find_time_zero(radargram, prepare_fitting_branch(radargram).filtered)
    (result,) = estimate_band_depths(radargram, [_flat_box()], zero)
    assert result.status == "no_scale"
    scaled_fields = (
        result.depth_top_m,
        result.depth_center_m,
        result.sigma_depth_m,
        result.v_m_per_ns,
        result.sigma_v,
        result.eps_r_used,
        result.t_zero_ns,
        result.t0_ns,
        result.t_apex_ns,
        result.x0_m,
    )
    assert all(value is None for value in scaled_fields)
    band = result.band
    assert band.x_ref_m is None and band.t_band_ns is None
    assert band.slope_ns_per_m is None and band.grade is None
    assert band.depth_profile is None and band.lateral_bias_max_m is None
    true_sample = (T_SRC_NS + TAU_FLAT_NS) / DT_NS
    assert band.ref_sample == pytest.approx(true_sample, abs=1.0)


def test_ub11_pixel_limits_scale_with_the_dominant_period() -> None:
    """Band spec 3-5 v1.1: no-scale limits are fractions of P_px."""
    radargram = _radargram([_flat_band()], dt_ns=None, dx_m=None, f_c_hz=None)
    period_px = prepare_fitting_branch(radargram).period_px
    # Ricker period is 50 px; direct wave and band together put the spectral
    # peak one FFT bin lower (57 px on 800 samples), so only a range is asserted.
    assert 2.2 <= period_px <= 128.0
    assert 40.0 <= period_px <= 60.0

    limits = pixel_limits(period_px)
    assert limits.f_scale == pytest.approx(0.05 * period_px)
    assert limits.res_max == pytest.approx(0.25 * period_px)
    assert limits.slope_min == pytest.approx(0.1 * period_px)
    assert limits.merge_tolerance == pytest.approx(max(0.5 * period_px, 3.0))
    assert limits.q_res_scale == pytest.approx(0.125 * period_px)
    doubled = pixel_limits(2 * period_px)
    assert doubled.res_max == pytest.approx(2 * limits.res_max)
    assert doubled.slope_min == pytest.approx(2 * limits.slope_min)

    fixed = pixel_limits(None)  # bandpass_px skipped
    assert (fixed.f_scale, fixed.res_max, fixed.slope_min) == (1.0, 2.0, 2.0)
    assert (fixed.merge_tolerance, fixed.q_res_scale) == (3.0, 1.0)


def test_pad_up_periods_raises_the_search_window() -> None:
    """Band spec 8-2 v1.3: the window padding follows ``pad_*_periods``."""
    radargram = _radargram([_flat_band()])
    a_bg = prepare_fitting_branch(radargram).filtered
    box = _flat_box()
    one = extract_band_picks(a_bg, box, None, DT_NS, F_C_HZ, BandConfig())
    two = extract_band_picks(
        a_bg, box, None, DT_NS, F_C_HZ, BandConfig(pad_up_periods=2.0)
    )
    period_samples = round(PERIOD_NS / DT_NS)
    assert one.window[0] == box.k0 - period_samples
    assert two.window[0] == one.window[0] - period_samples
    assert two.window[1] == one.window[1]


def test_no_scale_failure_reports_no_scale_with_the_reason_flag() -> None:
    """Band spec 6-2 v1.3: no_scale wins; the failure name becomes a flag."""
    radargram = _radargram([_flat_band()], dt_ns=None, dx_m=None, f_c_hz=None)
    zero = find_time_zero(radargram, prepare_fitting_branch(radargram).filtered)
    (result,) = estimate_band_depths(
        radargram, [_flat_box()], zero, cfg=BandConfig(n_picks_min=10_000)
    )
    assert result.status == "no_scale"
    assert "insufficient_picks" in result.flags


def test_branch_of_another_profile_is_refused() -> None:
    radargram = _radargram([_flat_band()])
    other = prepare_fitting_branch(make_radargram(BScanSpec(n_traces=50)))
    zero = given_zero(T_SRC_NS, DT_NS)
    with pytest.raises(ValueError, match="branch shape"):
        estimate_band_depths(radargram, [_flat_box()], zero, branch=other)


def test_neighbor_without_known_radius_adds_the_17_percent_note() -> None:
    radargram = _radargram([_flat_band()])
    entry = replace(_pool_entry(0.1, TAU_FLAT_NS), radius_known=False)
    choice = select_band_velocity(radargram, TAU_FLAT_NS, [entry], ())
    assert "neighbor_v_radius_unknown: v up to +17 %" in choice.bias_notes


def _pool_entry(
    v: float, tau_ns: float, line_id: str = "unit/line"
) -> VelocityEstimate:
    return VelocityEstimate(
        v_m_per_ns=v,
        sigma_v=0.002,
        tau_ns=tau_ns,
        line_id=line_id,
        survey_key=None,
        radius_known=True,
        source_index=0,
        eligible=True,
    )


def test_ub12_velocity_sources() -> None:
    radargram = _radargram([_flat_band()])
    pool = [_pool_entry(0.1, TAU_FLAT_NS)]
    # (a) ground + eligible neighbour hyperbola
    choice = select_band_velocity(radargram, TAU_FLAT_NS, pool, ())
    assert choice.v_rule == "neighbor_hyperbola"
    assert choice.alt_meta_v is not None  # rule 2a
    # (b) air-coupled ignores the pool
    air = replace(radargram, air_gap="air")
    assert select_band_velocity(air, TAU_FLAT_NS, pool, ()).v_rule == "meta_fixed"
    # (c) ground, no pool and no meta eps: tau only
    no_meta = _radargram([_flat_band()], eps_r_meta=None, eps_r_meta_source="")
    (result,) = _estimate(no_meta, [_flat_box()])
    assert result.status == "no_velocity"
    assert result.t0_ns == pytest.approx(TAU_FLAT_NS, abs=0.1)
    assert result.depth_top_m is None
    # (d) the dataset default is off unless set
    assert select_band_velocity(no_meta, TAU_FLAT_NS, (), ()).v_rule == "none"
    assumed = select_band_velocity(
        no_meta, TAU_FLAT_NS, (), (), BandConfig(dataset_default_eps=9.0)
    )
    assert assumed.v_rule == "dataset_default"
    assert "v_assumed" in assumed.flags


def test_ub13_dt_hypothesis_keeps_the_band_depth() -> None:
    stretch = 1.18
    radargram = _radargram([_flat_band()])
    stretched = _radargram(
        [_flat_band()], dt_ns=DT_NS * stretch, f_c_hz=F_C_HZ / stretch
    )
    box = _flat_box()
    (plain,) = _estimate(radargram, [box], pool=[_pool_entry(V9, TAU_FLAT_NS)])
    (scaled,) = _estimate(
        stretched,
        [box],
        pool=[_pool_entry(V9 / stretch, TAU_FLAT_NS * stretch)],
        t_zero_ns=T_SRC_NS * stretch,
    )
    assert plain.v_rule == scaled.v_rule == "neighbor_hyperbola"
    assert abs(scaled.depth_top_m - plain.depth_top_m) <= 0.001


def test_ub14_shallow_band_is_kept_out_of_the_zero() -> None:
    """Air-coupled: a band inside the zero window must not become the zero."""
    air_gap, offset = 0.3, 0.1
    spec = BScanSpec(
        f_c_hz=500e6,
        air_gap_m=air_gap,
        offset_m=offset,
        n_traces=N_TRACES,
        pipe_times_ns=np.full(N_TRACES, FAR_PIPE_NS),
    )
    surface_ns = 2 * math.hypot(air_gap, offset / 2) / SPEED_OF_LIGHT_M_PER_NS
    # d 0.1 m, not the spec's 0.15 m: with h 0.3 m and 500 MHz the zero window
    # ends near 7.7 ns, and a 0.15 m band (8.0 ns) would lie outside it.
    band_tau = surface_ns + 2 * 0.10 / V9
    band_traces = slice(0, int(0.6 * N_TRACES))
    spec.extra_events = [(np.full(N_TRACES, band_tau), _amplitude(band_traces, 2.0))]
    radargram = make_radargram(spec)
    filtered = prepare_fitting_branch(radargram).filtered
    box = GridBox(0, band_traces.stop - 1, 0, 10)
    excluded = band_exclude_traces([box], N_TRACES, DX_M)

    without = find_time_zero(radargram, filtered)
    with_mask = find_time_zero(radargram, filtered, exclude_traces=excluded)

    geometric_ns = T_SRC_NS + surface_ns
    assert without.t_zero_ns == pytest.approx(T_SRC_NS + band_tau, abs=0.15)
    assert 0.0 < with_mask.t_zero_ns - geometric_ns <= 0.2  # Z7 window


def test_ub20_layer_inside_the_window_is_flagged_not_corrected() -> None:
    """Band spec 13-1 R2: a layer one period above the band top is picked first."""
    layer = (np.full(N_TRACES, TAU_FLAT_NS - PERIOD_NS), _amplitude(BAND_TRACES))
    (result,) = _estimate(_radargram([_flat_band(), layer]), [_flat_box()])
    assert "band_top_multi_event" in result.flags
    assert LAYER_AMBIGUITY_NOTE in result.bias_notes
    assert result.depth_top_m < D_TOP_M - 0.02


def test_ub21_single_band_has_the_plate_note_only() -> None:
    """Relies on the Ricker envelope having no later lobe above 0.3 E_box."""
    (result,) = _estimate(_radargram([_flat_band()]), [_flat_box()])
    assert result.band.bg_mode == "masked_median"
    assert "band_top_multi_event" not in result.flags
    assert PROTECTIVE_PLATE_NOTE in result.bias_notes
    assert LAYER_AMBIGUITY_NOTE not in result.bias_notes


def test_ub22_plate_note_without_background_and_without_scale() -> None:
    full_line = (np.full(N_TRACES, TAU_FLAT_NS), np.ones(N_TRACES))
    box = _box(0, N_TRACES - 1, TAU_FLAT_NS, TAU_FLAT_NS)
    (no_background,) = _estimate(_radargram([full_line]), [box])
    assert "bg_none" in no_background.flags
    assert PROTECTIVE_PLATE_NOTE in no_background.bias_notes
    assert LAYER_AMBIGUITY_NOTE in no_background.bias_notes
    assert no_background.bias_notes.count(LAYER_AMBIGUITY_NOTE) == 1

    radargram = _radargram([_flat_band()], dt_ns=None, dx_m=None, f_c_hz=None)
    zero = find_time_zero(radargram, prepare_fitting_branch(radargram).filtered)
    (no_scale,) = estimate_band_depths(radargram, [_flat_box()], zero)
    assert no_scale.status == "no_scale"
    assert PROTECTIVE_PLATE_NOTE in no_scale.bias_notes
    assert no_scale.depth_top_m is None and no_scale.t0_ns is None
    assert no_scale.band.x_ref_m is None and no_scale.band.t_band_ns is None


def test_ub15_pick_mode_mismatch_is_refused() -> None:
    radargram = _radargram([_flat_band()])
    zero = given_zero(T_SRC_NS, DT_NS, pick_mode="envelope")
    with pytest.raises(ValueError, match="phase reference"):
        estimate_band_depths(
            radargram, [_flat_box()], zero, cfg=BandConfig(pick_mode="signed_max")
        )
