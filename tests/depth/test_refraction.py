"""Unit tests of the air-gap ray model and the refraction QC (spec 6-1, 7 rule 2a)."""

import numpy as np
import pytest

from gpr_detection.depth.config import DepthConfig
from gpr_detection.depth.refraction import (
    leg_time,
    reflection_times,
    refraction_qc,
    surface_reflection_time,
)
from gpr_detection.io.radargram import SPEED_OF_LIGHT_M_PER_NS, velocity_from_eps
from synthetic_bscan import cylinder_times

C = SPEED_OF_LIGHT_M_PER_NS
V9 = velocity_from_eps(9.0)
X_M = np.linspace(0.5, 2.5, 41)
X0_M = 1.5
NARROW_QC_GRID = DepthConfig(refraction_eps_min=6.0, refraction_eps_max=12.0)
"""eps_r 6-12 instead of 3-30 keeps the QC test short; the algorithm is the same."""


def test_leg_without_air_gap_is_a_straight_soil_path() -> None:
    assert leg_time(0.0, 0.0, 0.3, -0.4, V9) == pytest.approx(0.5 / V9)


def test_leg_matches_a_brute_force_search_over_crossing_points() -> None:
    h, px, pz = 0.3, 0.8, -0.6
    crossings = np.linspace(0.0, px, 200_001)
    brute = np.min(np.hypot(crossings, h) / C + np.hypot(px - crossings, pz) / V9)
    assert leg_time(0.0, h, px, pz, V9) == pytest.approx(brute, abs=1e-6)


def test_vertical_leg_is_air_plus_soil_time() -> None:
    assert leg_time(0.0, 0.3, 0.0, -0.6, V9) == pytest.approx(0.3 / C + 0.6 / V9)


def test_ground_coupled_point_scatterer_is_the_straight_ray_hyperbola() -> None:
    times = reflection_times(
        X_M, offset_m=0.0, h=0.0, x0_m=X0_M, d_top_m=0.6, radius_m=0.0, v=V9
    )
    np.testing.assert_allclose(times, cylinder_times(X_M, X0_M, 0.6, V9, 0.0))


def test_ground_coupled_cylinder_matches_the_fixed_radius_model() -> None:
    times = reflection_times(
        X_M, offset_m=0.0, h=0.0, x0_m=X0_M, d_top_m=0.6, radius_m=0.1, v=V9
    )
    expected = cylinder_times(X_M, X0_M, 0.6, V9, 0.1)
    np.testing.assert_allclose(times, expected, atol=1e-5)


def test_air_gap_apex_time_is_air_plus_soil_round_trip() -> None:
    apex = reflection_times(
        np.array([X0_M]),
        offset_m=0.0,
        h=0.3,
        x0_m=X0_M,
        d_top_m=0.6,
        radius_m=0.1,
        v=V9,
    )[0]
    assert apex == pytest.approx(2 * 0.3 / C + 2 * 0.6 / V9, abs=1e-5)


def test_air_gap_flattens_the_wings() -> None:
    """Refraction makes the far wing earlier than the straight soil path."""
    common = {"offset_m": 0.0, "x0_m": X0_M, "d_top_m": 0.6, "radius_m": 0.0, "v": V9}
    air = reflection_times(X_M, h=0.3, **common) - 2 * 0.3 / C
    soil = reflection_times(X_M, h=0.0, **common)
    assert air[0] < soil[0]
    assert air[20] == pytest.approx(soil[20], abs=1e-6)


def test_surface_reflection_time() -> None:
    assert surface_reflection_time(0.3, 0.1) == pytest.approx(
        2 * np.hypot(0.3, 0.05) / C
    )


def test_refraction_qc_recovers_eps_from_exact_picks() -> None:
    h, s, radius = 0.3, 0.1, 0.05
    times = reflection_times(
        X_M, offset_m=s, h=h, x0_m=X0_M, d_top_m=0.6, radius_m=radius, v=V9
    )
    tau = times - surface_reflection_time(h, s)
    qc = refraction_qc(
        X_M,
        tau,
        x0_m=X0_M,
        tau_apex_ns=float(tau.min()),
        h_m=h,
        offset_m=s,
        radius_m=radius,
        config=NARROW_QC_GRID,
    )
    assert qc.eps_fit == pytest.approx(9.0, abs=0.1)
    assert qc.d_top_m == pytest.approx(0.6, abs=0.01)
    assert qc.rms_ns < 0.01
