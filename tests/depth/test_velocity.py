"""Unit tests of the section 7 velocity table (``depth/velocity.py``)."""

import numpy as np
import pytest

from gpr_detection.depth.hyperbola import CurvatureFit, FitStatus, HyperbolaFit
from gpr_detection.depth.velocity import (
    RADIUS_UNKNOWN_NOTE,
    VelocityRule,
    select_hyperbola_velocity,
)
from gpr_detection.io.radargram import velocity_from_eps
from synthetic_bscan import BScanSpec, make_radargram

V_FIT = 0.1
SIGMA_V_FIT = 0.002


def _fit(status: FitStatus) -> CurvatureFit:
    fit = HyperbolaFit(
        x0_m=1.5,
        t0_ns=12.0,
        v_m_per_ns=V_FIT,
        sigma_x0_m=0.001,
        sigma_t0_ns=0.01,
        sigma_v=SIGMA_V_FIT,
        radius_m=0.0,
        rms_ns=0.05,
        inlier_frac=1.0,
        n_picks=101,
        converged=status == FitStatus.OK,
        at_bound=False,
    )
    if status == FitStatus.INSUFFICIENT_PICKS:
        return CurvatureFit(status, None, None, ())
    return CurvatureFit(status, None, fit, ())


def _radargram(**overrides):
    return make_radargram(
        BScanSpec(snr_db=None, n_traces=11, n_samples=50), **overrides
    )


def test_rule1_no_scale_has_no_velocity() -> None:
    choice = select_hyperbola_velocity(
        _radargram(dt_ns=None, dx_m=None), _fit(FitStatus.OK)
    )
    assert choice.v_rule == VelocityRule.NONE
    assert not choice.has_velocity


def test_rule2_air_coupled_uses_meta_eps_with_7_5_percent() -> None:
    choice = select_hyperbola_velocity(_radargram(air_gap="air"), None)
    assert choice.v_rule == "meta_fixed"
    assert choice.v_m_per_ns == pytest.approx(velocity_from_eps(9.0))
    assert choice.sigma_v / choice.v_m_per_ns == pytest.approx(0.075)
    assert "meta_eps_unverified" in choice.flags


def test_rule2_air_coupled_without_meta_has_no_velocity() -> None:
    radargram = _radargram(air_gap="air", eps_r_meta=None, eps_r_meta_source="")
    assert not select_hyperbola_velocity(radargram, None).has_velocity


def test_rule3_curvature_adds_the_3_percent_floor_and_3a_keeps_meta() -> None:
    choice = select_hyperbola_velocity(_radargram(), _fit(FitStatus.OK))
    assert choice.v_rule == "curvature_R_fixed"
    expected_rel = np.hypot(SIGMA_V_FIT / V_FIT, 0.03)
    assert choice.sigma_v / V_FIT == pytest.approx(expected_rel)
    assert choice.alt_meta_v == pytest.approx(velocity_from_eps(9.0))
    assert choice.bias_notes == (RADIUS_UNKNOWN_NOTE,)


def test_rule3_known_radius_has_no_radius_note() -> None:
    choice = select_hyperbola_velocity(
        _radargram(pipe_radius_m=0.1), _fit(FitStatus.OK)
    )
    assert choice.bias_notes == ()


def test_rule4_too_few_picks_with_meta_is_meta_fixed() -> None:
    choice = select_hyperbola_velocity(_radargram(), _fit(FitStatus.INSUFFICIENT_PICKS))
    assert choice.v_rule == "meta_fixed"


def test_rule5_too_few_picks_without_meta_has_no_velocity() -> None:
    radargram = _radargram(eps_r_meta=None, eps_r_meta_source="")
    choice = select_hyperbola_velocity(radargram, _fit(FitStatus.INSUFFICIENT_PICKS))
    assert choice.v_rule == "none"
    assert choice.alt_meta_v is None


def test_rule6_unknown_air_gap_has_no_velocity_and_meta_alternative() -> None:
    choice = select_hyperbola_velocity(_radargram(air_gap="unknown"), None)
    assert choice.v_rule == "none"
    assert choice.alt_meta_v == pytest.approx(velocity_from_eps(9.0))


def test_failed_fit_keeps_meta_velocity_as_alternative_only() -> None:
    choice = select_hyperbola_velocity(_radargram(), _fit(FitStatus.FIT_FAILED))
    assert choice.v_rule == "none"
    assert choice.alt_meta_v == pytest.approx(velocity_from_eps(9.0))
