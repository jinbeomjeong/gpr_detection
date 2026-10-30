"""Unit tests of ``estimate_depth`` (spec sections 2, 7-10; U4, U5, U7, table rows).

Analytic B-scans only. Ground-coupled depths use the detected direct-wave zero,
so they come out shallow by about ``v * 0.7 ns / 2`` = 3.5 cm (spec 4-2, R8).
"""

import numpy as np
import pytest

from gpr_detection.depth import estimate as estimate_module
from gpr_detection.depth import hyperbola as hyperbola_module
from gpr_detection.depth.config import DepthConfig
from gpr_detection.depth.estimate import (
    depth_from_time,
    depth_uncertainty,
    estimate_depth,
    prepare_depth_inputs,
)
from gpr_detection.depth.hyperbola import (
    CurvatureFit,
    FitStatus,
    RansacInit,
    fixed_velocity_apex,
)
from gpr_detection.depth.picks import extract_picks
from gpr_detection.depth.refraction import RefractionQC, reflection_times
from gpr_detection.io.radargram import GridBox, velocity_from_eps
from synthetic_bscan import DT_NS, F_C_HZ, T_SRC_NS, BScanSpec, make_radargram

GROUND_BOX = GridBox(i0=25, i1=125, k0=270, k1=560)
DIRECT_ZERO_SHALLOW_M = (-0.05, 0.0)
"""Ground-coupled depth error range with the direct-wave zero (spec R8)."""

QC_GRID = DepthConfig(refraction_eps_min=4.0, refraction_eps_max=14.0)
"""eps_r 4-14 keeps the two refraction QC runs short; the algorithm is the same."""

U5_STRETCH = 1.18


def _depth_error_m(result, true_depth_m: float = 0.6) -> float:
    return result.depth_top_m - true_depth_m


def _air_radargram(air_gap_m: float, radius_m: float = 0.05, **overrides):
    """500 MHz air-coupled profile; pipe times from the refraction model."""
    spec = BScanSpec(
        f_c_hz=500e6, air_gap_m=air_gap_m, offset_m=0.1, d_top_m=0.6, radius_m=radius_m
    )
    spec.pipe_times_ns = reflection_times(
        spec.x_m,
        offset_m=spec.offset_m,
        h=air_gap_m,
        x0_m=spec.x0_m,
        d_top_m=0.6,
        radius_m=radius_m,
        v=spec.velocity,
    )
    k_top = int((T_SRC_NS + spec.pipe_times_ns.min()) / DT_NS) - 20
    box = GridBox(i0=25, i1=125, k0=k_top, k1=k_top + 300)
    return make_radargram(spec, **overrides), box


# ------------------------------------------------------------------ formulas


def test_depth_formulas_of_section_8() -> None:
    assert depth_from_time(0.1, 12.0, air_gap="ground", offset_m=None) == (
        pytest.approx(0.6)
    )
    assert depth_from_time(0.1, 12.0, air_gap="ground", offset_m=0.1) == (
        pytest.approx(np.sqrt(0.36 - 0.0025))
    )
    assert depth_from_time(0.1, 12.0, air_gap="air", offset_m=0.1) == (
        pytest.approx(0.6)
    )


def test_uncertainty_formula_of_section_9() -> None:
    sigma = depth_uncertainty(0.6, 0.1, 0.0075, 0.15, 0.05)
    expected = np.hypot(0.6 * 0.075, 0.05 * np.hypot(0.15, 0.05))
    assert sigma == pytest.approx(expected)


# ------------------------------------------------------------------ ground rules


def test_rule3_ground_curvature_depth_and_3a_meta_alternative() -> None:
    result = estimate_depth(make_radargram(BScanSpec()), GROUND_BOX)
    assert result.status == "ok"
    assert result.target_class == "pipe_hyperbola"
    assert result.v_rule == "curvature_R_fixed"
    low, high = DIRECT_ZERO_SHALLOW_M
    assert low <= _depth_error_m(result) < high
    assert result.zero_rule == "ground_direct_peak"
    assert "meta_fixed" in result.alt_depths
    assert "radius_unknown: v,d over by up to +17 %" in result.bias_notes
    assert result.depth_center_m is None
    assert result.sigma_depth_m > 0
    assert result.n_picks == 101


def test_rule3a_meta_alternative_uses_the_section_6_4_vertex() -> None:
    radargram = make_radargram(BScanSpec())
    config = DepthConfig()
    inputs = prepare_depth_inputs(radargram, config)
    result = estimate_depth(radargram, GROUND_BOX, config, inputs=inputs)
    picks = extract_picks(
        inputs.branch.background_removed,
        GROUND_BOX,
        inputs.zero,
        dt_ns=DT_NS,
        dx_m=radargram.dx_m,
        f_c_hz=F_C_HZ,
    )
    vertex_t0_ns = fixed_velocity_apex(picks).t0_ns
    expected = depth_from_time(
        velocity_from_eps(9.0), vertex_t0_ns, air_gap="ground", offset_m=0.1
    )
    assert result.v_rule == "curvature_R_fixed"
    assert result.alt_depths["meta_fixed"].depth_top_m == pytest.approx(expected)
    assert vertex_t0_ns != pytest.approx(result.t0_ns, abs=1e-9)


def test_rule6_unknown_air_gap_has_no_velocity(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("fit_curvature must not run for an unknown air gap")

    monkeypatch.setattr(estimate_module, "fit_curvature", forbidden)
    result = estimate_depth(make_radargram(BScanSpec(), air_gap="unknown"), GROUND_BOX)
    assert result.status == "no_velocity"
    assert result.depth_top_m is None
    assert "meta_fixed" in result.alt_depths
    assert "air_gap_unknown" in result.flags


def test_known_radius_gives_the_centre_depth() -> None:
    spec = BScanSpec(radius_m=0.1)
    result = estimate_depth(make_radargram(spec, pipe_radius_m=0.1), GROUND_BOX)
    assert result.radius_used_m == 0.1
    assert result.depth_center_m == pytest.approx(result.depth_top_m + 0.1)


def test_rule4_too_few_picks_with_meta_uses_meta_fixed() -> None:
    narrow = DepthConfig(a_max_m=0.04)
    result = estimate_depth(make_radargram(BScanSpec()), GROUND_BOX, narrow)
    assert result.status == "ok"
    assert result.v_rule == "meta_fixed"
    assert "meta_eps_unverified" in result.flags
    low, high = DIRECT_ZERO_SHALLOW_M
    assert low <= _depth_error_m(result) < high


def test_rule5_too_few_picks_without_meta_gives_no_depth() -> None:
    radargram = make_radargram(BScanSpec(), eps_r_meta=None, eps_r_meta_source="")
    result = estimate_depth(radargram, GROUND_BOX, DepthConfig(a_max_m=0.04))
    assert result.status == "insufficient_picks"
    assert result.v_rule == "none"
    assert result.depth_top_m is None


def test_u5_dt_hypothesis_keeps_depth_and_scales_velocity() -> None:
    spec = BScanSpec()
    plain = estimate_depth(make_radargram(spec), GROUND_BOX)
    stretched = estimate_depth(
        make_radargram(spec, dt_ns=DT_NS * U5_STRETCH, f_c_hz=F_C_HZ / U5_STRETCH),
        GROUND_BOX,
    )
    assert abs(stretched.depth_top_m - plain.depth_top_m) <= 0.001
    ratio = stretched.v_m_per_ns * U5_STRETCH / plain.v_m_per_ns
    assert ratio == pytest.approx(1.0, abs=0.005)


# ------------------------------------------------------------------ air rules


def test_u4_air_coupled_meta_fixed_depth_and_refraction_interval() -> None:
    """U4 (b) depth +-2 cm and (c) the 2a interval contains eps 9, no suspicion."""
    radargram, box = _air_radargram(air_gap_m=0.3)
    result = estimate_depth(radargram, box, QC_GRID)
    assert result.status == "ok"
    assert result.zero_rule == "air_surface_late"
    assert result.v_rule == "meta_fixed"
    assert abs(_depth_error_m(result)) <= 0.02
    eps_lo, eps_hi = result.eps_r_refraction_qc_range
    assert eps_lo <= 9.0 <= eps_hi
    assert result.eps_r_refraction_qc == eps_hi
    assert QC_GRID.refraction_eps_min < eps_lo  # not stuck on the grid ends
    assert eps_hi < QC_GRID.refraction_eps_max
    assert "meta_eps_suspect" not in result.flags
    assert result.inlier_frac is None  # no straight-ray curvature fit


def test_u4_curvature_like_meta_eps_is_suspected() -> None:
    """U4 (d), required: eps_r_meta 2.7 mimics the M20 straight-ray curvature eps."""
    radargram, box = _air_radargram(air_gap_m=0.3, eps_r_meta=2.7)
    result = estimate_depth(radargram, box, QC_GRID)
    assert "meta_eps_suspect" in result.flags


def test_u4_wrong_meta_eps_is_suspected() -> None:
    """U4 (d), optional: eps_r_meta 4.5, just below the lower limit.

    The interval here is about (6.85, 9.70); the lower limit 0.7 * 6.85 = 4.79
    lies below 5, so eps_r_meta 5 is not suspected under the v1.4 rule.
    """
    radargram, box = _air_radargram(air_gap_m=0.3, eps_r_meta=4.5)
    result = estimate_depth(radargram, box, QC_GRID)
    eps_lo, _ = result.eps_r_refraction_qc_range
    assert 0.7 * eps_lo > 4.5
    assert "meta_eps_suspect" in result.flags


def test_u4_air_coupled_straight_ray_fit_is_never_called(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("fit_curvature must not run for air-coupled data")

    monkeypatch.setattr(estimate_module, "fit_curvature", forbidden)
    monkeypatch.setattr(estimate_module, "refraction_qc", _fake_qc(9.0))
    radargram, box = _air_radargram(air_gap_m=0.3)
    assert estimate_depth(radargram, box).v_rule == "meta_fixed"


def _fake_qc(eps_fit: float):
    def qc(*args, **kwargs) -> RefractionQC:
        return RefractionQC(eps_fit, 0.3 / np.sqrt(eps_fit), 0.6, 0.01, 50, 0.3, 0.1)

    return qc


def test_rule2a_refraction_qc_far_from_meta_flags_meta_eps_suspect(
    monkeypatch,
) -> None:
    monkeypatch.setattr(estimate_module, "refraction_qc", _fake_qc(5.0))
    radargram, box = _air_radargram(air_gap_m=0.3)
    result = estimate_depth(radargram, box)
    assert result.eps_r_refraction_qc == 5.0
    assert "meta_eps_suspect" in result.flags
    assert result.v_rule == "meta_fixed"


def test_air_single_event_is_deep_and_skips_the_refraction_qc(monkeypatch) -> None:
    monkeypatch.setattr(estimate_module, "refraction_qc", _fake_qc(9.0))
    radargram, box = _air_radargram(air_gap_m=0.1)
    result = estimate_depth(radargram, box)
    assert result.zero_rule == "air_single_event"
    assert 0.0 < _depth_error_m(result) <= 0.05
    assert result.eps_r_refraction_qc is None
    assert any("deep by up to" in note for note in result.bias_notes)


def test_air_coupled_without_meta_has_no_velocity() -> None:
    radargram, box = _air_radargram(
        air_gap_m=0.1, eps_r_meta=None, eps_r_meta_source=""
    )
    result = estimate_depth(radargram, box)
    assert result.status == "no_velocity"
    assert result.depth_top_m is None


def test_meta_zeros_get_labelled_alternative_depths() -> None:
    radargram, box = _air_radargram(
        air_gap_m=0.1, t0_candidates_ns={"t0_sample_est": T_SRC_NS + 0.6}
    )
    result = estimate_depth(radargram, box)
    alternative = result.alt_depths["t0_sample_est"]
    assert alternative.label == "merged"
    assert alternative.t_zero_ns == T_SRC_NS + 0.6


# ------------------------------------------------------------------ no scale, failures


def test_u7_no_scale_gives_the_apex_pixel_only() -> None:
    spec = BScanSpec()
    radargram = make_radargram(spec, dt_ns=None, dx_m=None, f_c_hz=None)
    result = estimate_depth(radargram, GROUND_BOX)
    assert result.status == "no_scale"
    assert result.depth_top_m is None
    assert result.v_m_per_ns is None
    assert result.t_zero_ns is None
    assert result.apex_trace == pytest.approx(75, abs=1)
    assert result.apex_sample == pytest.approx(
        (T_SRC_NS + 2 * 0.6 / spec.velocity) / DT_NS, abs=2
    )


def test_empty_record_fails_the_zero() -> None:
    radargram = make_radargram(BScanSpec(), data=np.zeros((800, 151), np.float32))
    assert estimate_depth(radargram, GROUND_BOX).status == "zero_failed"


def test_box_in_the_zero_zone_fails_the_fit() -> None:
    box = GridBox(i0=25, i1=125, k0=0, k1=120)
    result = estimate_depth(make_radargram(BScanSpec()), box)
    assert result.status == "fit_failed"
    assert "box_in_zero_zone" in result.flags


def test_failed_curvature_fit_keeps_the_meta_depth_as_alternative(
    monkeypatch,
) -> None:
    def failed(*args, **kwargs) -> CurvatureFit:
        return CurvatureFit(FitStatus.FIT_FAILED, None, None, ())

    monkeypatch.setattr(estimate_module, "fit_curvature", failed)
    result = estimate_depth(make_radargram(BScanSpec()), GROUND_BOX)
    assert result.status == "fit_failed"
    assert result.depth_top_m is None
    low, high = DIRECT_ZERO_SHALLOW_M
    assert low <= result.alt_depths["meta_fixed"].depth_top_m - 0.6 < high


def test_ransac_rejection_keeps_a_good_curvature_result(monkeypatch) -> None:
    def rejected(x_m, tau_ns, x_apex_m, tau_apex_ns, v_fallback=None, config=None):
        return RansacInit(
            x_apex_m, tau_apex_ns, v_fallback or 0.1, np.zeros(len(x_m), bool), False
        )

    monkeypatch.setattr(hyperbola_module, "ransac_init", rejected)
    result = estimate_depth(make_radargram(BScanSpec()), GROUND_BOX)
    assert result.status == "ok"
    assert result.v_rule == "curvature_R_fixed"
    assert "ransac_rejected" in result.flags


def test_saturated_apex_is_flagged() -> None:
    spec = BScanSpec(pipe_amplitude=40000.0, snr_db=None)
    result = estimate_depth(make_radargram(spec), GROUND_BOX)
    assert "saturated_apex" in result.flags


def test_dataset_and_processing_flags_are_carried_over() -> None:
    radargram = make_radargram(
        BScanSpec(), amplitude_processed=True, metadata={"dt_hypothesis": "B"}
    )
    result = estimate_depth(radargram, GROUND_BOX)
    assert result.zero_rule == "meta_sample0"
    assert {"zero_unverified", "amplitude_processed", "dt_hypothesis_B"} <= set(
        result.flags
    )


def test_shared_inputs_must_match_the_pick_mode() -> None:
    radargram = make_radargram(BScanSpec())
    inputs = prepare_depth_inputs(radargram)
    with pytest.raises(ValueError, match="phase reference"):
        estimate_depth(
            radargram, GROUND_BOX, DepthConfig(pick_mode="signed_max"), inputs=inputs
        )
