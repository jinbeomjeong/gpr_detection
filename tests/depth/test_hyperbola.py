"""Unit tests of picks and the fixed-radius hyperbola fit (spec section 11-1, M2-3).

U1-U3, U5, U6, U8, the U4 refusal and the U7 picks (no-scale apex candidate) at the
pick/fit level. The zero is the transmit time (``given_zero``), as U1 specifies.
"""

from dataclasses import replace

import numpy as np
import pytest

from gpr_detection.depth.config import DepthConfig
from gpr_detection.depth.hyperbola import (
    FitStatus,
    apex_vertex,
    fit_curvature,
    fit_hyperbola,
    fixed_velocity_apex,
    hyperbola_time,
    ransac_init,
)
from gpr_detection.depth.picks import extract_picks, widen_box
from gpr_detection.io.radargram import GridBox
from gpr_detection.preprocess.filters import prepare_fitting_branch
from gpr_detection.preprocess.timezero import given_zero
from synthetic_bscan import DT_NS, F_C_HZ, T_SRC_NS, BScanSpec, make_radargram

CONFIG = DepthConfig()
U1_BOX = GridBox(i0=25, i1=125, k0=270, k1=560)
U5_STRETCH = 1.18
"""GPR-SD dt hypothesis ratio B/A (spec section 7)."""


def _picks(radargram, box: GridBox = U1_BOX, *, t_zero_ns=T_SRC_NS, config=CONFIG):
    section = prepare_fitting_branch(radargram).background_removed
    zero = given_zero(t_zero_ns, radargram.dt_ns or 1.0)
    return extract_picks(
        section,
        box,
        zero,
        dt_ns=radargram.dt_ns,
        dx_m=radargram.dx_m,
        f_c_hz=radargram.f_c_hz,
        config=config,
    )


def _curvature(picks, radius_m: float):
    return fit_curvature(picks, radius_m=radius_m, air_gap="ground", config=CONFIG)


def test_u1_point_scatterer_recovers_velocity_and_depth() -> None:
    spec = BScanSpec(radius_m=0.0)
    picks = _picks(make_radargram(spec))
    assert picks.is_enough_for_curvature(CONFIG)
    assert picks.n_left == picks.n_right == 50  # a_max 1.0 m / 0.02 m
    result = _curvature(picks, radius_m=0.0)
    assert result.status == FitStatus.OK
    assert result.init.accepted
    assert result.fit.v_m_per_ns / spec.velocity == pytest.approx(1.0, abs=0.05)
    assert result.fit.depth_top_m == pytest.approx(0.6, abs=0.02)
    assert result.fit.x0_m == pytest.approx(spec.x0_m, abs=0.02)
    assert result.fit.sigma_v > 0


def test_u2_cylinder_needs_the_fixed_radius_and_r0_overestimates_v() -> None:
    spec = BScanSpec(radius_m=0.1)
    picks = _picks(make_radargram(spec))
    fixed = _curvature(picks, radius_m=0.1).fit
    point = _curvature(picks, radius_m=0.0).fit
    assert fixed.v_m_per_ns / spec.velocity == pytest.approx(1.0, abs=0.05)
    assert fixed.depth_top_m == pytest.approx(0.6, abs=0.02)
    assert point.v_m_per_ns > spec.velocity


def test_u3_outliers_are_rejected() -> None:
    spec = BScanSpec(radius_m=0.0)
    picks = _picks(make_radargram(spec))
    x, tau = picks.x_m, picks.tau_ns
    rng = np.random.default_rng(0)
    corrupted = tau.copy()
    chosen = rng.choice(len(tau), size=len(tau) // 5, replace=False)
    signs = rng.choice([-1.0, 1.0], size=len(chosen))
    corrupted[chosen] += signs * rng.uniform(3.0, 5.0, size=len(chosen))
    init = ransac_init(x, corrupted, picks.apex_x_m, picks.apex_tau_ns, config=CONFIG)
    fit = fit_hyperbola(
        x, corrupted, radius_m=0.0, init=init, air_gap="ground", config=CONFIG
    )
    assert fit.v_m_per_ns / spec.velocity == pytest.approx(1.0, abs=0.05)
    assert fit.depth_top_m == pytest.approx(0.6, abs=0.02)
    assert fit.inlier_frac >= 0.75
    # v1.3: covariance over inliers only, so outlier size is not fit uncertainty.
    assert fit.sigma_v / fit.v_m_per_ns < 0.01


def test_u4_air_coupled_curvature_fit_is_refused() -> None:
    picks = _picks(make_radargram(BScanSpec()))
    with pytest.raises(ValueError, match="air-coupled"):
        fit_curvature(picks, radius_m=0.0, air_gap="air")
    x = np.linspace(-1, 1, 21)
    tau = hyperbola_time(x, 0.0, 12.0, 0.1, 0.0)
    with pytest.raises(ValueError, match="air-coupled"):
        fit_hyperbola(
            x, tau, radius_m=0.0, init=ransac_init(x, tau, 0.0, 12.0), air_gap="air"
        )


def test_u5_stretched_time_axis_keeps_depth_and_scales_velocity() -> None:
    spec = BScanSpec(radius_m=0.0)
    plain = make_radargram(spec)
    stretched = make_radargram(
        spec, dt_ns=DT_NS * U5_STRETCH, f_c_hz=F_C_HZ / U5_STRETCH
    )
    fit_a = _curvature(_picks(plain), radius_m=0.0).fit
    fit_b = _curvature(
        _picks(stretched, t_zero_ns=T_SRC_NS * U5_STRETCH), radius_m=0.0
    ).fit
    assert abs(fit_b.depth_top_m - fit_a.depth_top_m) <= 0.001
    assert fit_b.v_m_per_ns * U5_STRETCH / fit_a.v_m_per_ns == pytest.approx(
        1.0, abs=0.005
    )


def test_u6_mixed_pick_modes_are_refused() -> None:
    radargram = make_radargram(BScanSpec())
    section = prepare_fitting_branch(radargram).background_removed
    zero = given_zero(T_SRC_NS, DT_NS, pick_mode="envelope")
    signed = DepthConfig(pick_mode="signed_max")
    with pytest.raises(ValueError, match="phase reference"):
        extract_picks(
            section, U1_BOX, zero, dt_ns=DT_NS, dx_m=0.02, f_c_hz=F_C_HZ, config=signed
        )


def test_u8_candidate_takes_the_first_peak_not_the_stronger_late_event() -> None:
    spec = BScanSpec(radius_m=0.1)
    top_ns = (2.0 / spec.velocity) * (
        np.sqrt((0.6 + 0.1) ** 2 + (spec.x_m - spec.x0_m) ** 2) - 0.1
    )
    spec.extra_events = [(top_ns + 6.0, 2.0)]
    picks = _picks(make_radargram(spec), GridBox(i0=25, i1=125, k0=270, k1=700))
    assert picks.apex_tau_ns == pytest.approx(2 * 0.6 / spec.velocity, abs=0.1)


def test_fixed_velocity_apex_matches_the_true_apex_time() -> None:
    spec = BScanSpec(radius_m=0.0)
    vertex = fixed_velocity_apex(_picks(make_radargram(spec)))
    assert not vertex.used_fallback
    assert vertex.n_points >= CONFIG.apex_min_points
    assert vertex.t0_ns == pytest.approx(2 * 0.6 / spec.velocity, abs=0.05)
    assert vertex.x0_m == pytest.approx(spec.x0_m, abs=0.02)


def test_narrow_aperture_gives_insufficient_picks() -> None:
    narrow = DepthConfig(a_max_m=0.04)  # two traces per side
    picks = _picks(make_radargram(BScanSpec()), config=narrow)
    result = fit_curvature(picks, radius_m=0.0, air_gap="ground", config=narrow)
    assert result.status == FitStatus.INSUFFICIENT_PICKS
    assert result.fit is None


def test_box_above_the_mute_fails_the_curvature_fit() -> None:
    picks = _picks(make_radargram(BScanSpec()), GridBox(i0=25, i1=125, k0=0, k1=120))
    result = _curvature(picks, radius_m=0.0)
    assert result.status == FitStatus.FIT_FAILED
    assert "box_in_zero_zone" in result.flags
    assert result.fit is None


def test_all_zero_box_fails_the_curvature_fit_without_an_apex_candidate() -> None:
    radargram = make_radargram(BScanSpec(), data=np.zeros((800, 151), np.float32))
    result = _curvature(_picks(radargram), radius_m=0.0)
    assert result.status == FitStatus.FIT_FAILED
    assert "no_apex_candidate" in result.flags


def test_box_above_the_mute_gives_no_picks() -> None:
    picks = _picks(make_radargram(BScanSpec()), GridBox(i0=25, i1=125, k0=0, k1=120))
    assert picks.n_picks == 0
    assert picks.flags == ("box_in_zero_zone",)


def test_box_top_inside_the_mute_is_cut_at_the_mute() -> None:
    picks = _picks(make_radargram(BScanSpec()), GridBox(i0=25, i1=125, k0=0, k1=560))
    mute_k = (T_SRC_NS + 1e9 / F_C_HZ) / DT_NS
    assert picks.box.k0 == pytest.approx(mute_k, abs=1)
    assert picks.apex_tau_ns == pytest.approx(12.0, abs=0.1)


def test_apex_near_the_box_side_is_flagged() -> None:
    picks = _picks(make_radargram(BScanSpec()), GridBox(i0=74, i1=125, k0=270, k1=560))
    assert "apex_at_box_edge" in picks.flags


def test_no_scale_returns_the_unmuted_apex_candidate_only() -> None:
    """U7 picks (SNR 20 dB): pixel band-pass keeps noise below the candidate rule."""
    spec = BScanSpec()
    radargram = make_radargram(spec, dt_ns=None, dx_m=None, f_c_hz=None)
    picks = _picks(radargram)
    true_apex_k = (T_SRC_NS + 2 * 0.6 / spec.velocity) / DT_NS
    assert picks.n_picks == 1
    assert picks.apex.trace == pytest.approx(75, abs=1)
    assert picks.apex.k == pytest.approx(true_apex_k, abs=2)
    with pytest.raises(ValueError, match="physical scale"):
        _ = picks.tau_ns


def test_failed_zero_tracks_without_a_mute_and_flags_it() -> None:
    radargram = make_radargram(BScanSpec())
    section = prepare_fitting_branch(radargram).background_removed
    zero = given_zero(T_SRC_NS, DT_NS)
    unknown_zero = replace(zero, t_zero_ns=None)
    picks = extract_picks(
        section, U1_BOX, unknown_zero, dt_ns=DT_NS, dx_m=0.02, f_c_hz=F_C_HZ
    )
    assert "mute_unavailable" in picks.flags
    assert picks.n_picks > 1


def test_widen_box_adds_ten_percent_per_side_and_clips() -> None:
    widened = widen_box(GridBox(i0=5, i1=104, k0=0, k1=9), n_traces=110, fraction=0.1)
    assert (widened.i0, widened.i1) == (0, 109)


def test_ransac_rejects_a_concave_parabola() -> None:
    x = np.linspace(-1, 1, 21)
    tau = 10.0 - 2.0 * x**2
    init = ransac_init(x, tau, 0.0, 10.0, v_fallback=0.08)
    assert not init.accepted
    assert (init.x0_m, init.t0_ns, init.v_m_per_ns) == (0.0, 10.0, 0.08)


def test_apex_vertex_recovers_an_offset_vertex() -> None:
    x = np.arange(-0.1, 0.1001, 0.02)
    tau = 12.0 + 3.0 * (x - 0.03) ** 2
    vertex = apex_vertex(x, tau, 0.0, float(tau[5]))
    assert not vertex.used_fallback
    assert vertex.x0_m == pytest.approx(0.03)
    assert vertex.t0_ns == pytest.approx(12.0)


def test_apex_vertex_falls_back_with_too_few_points() -> None:
    vertex = apex_vertex(np.array([0.0, 0.02]), np.array([12.0, 12.1]), 0.0, 12.0)
    assert vertex.used_fallback
    assert vertex.t0_ns == 12.0
