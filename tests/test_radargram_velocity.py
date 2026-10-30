"""Unit tests of the speed-of-light helpers in ``io/radargram.py``."""

import pytest

from gpr_detection.io.radargram import (
    SPEED_OF_LIGHT_M_PER_NS,
    eps_from_velocity,
    velocity_from_eps,
)


def test_speed_of_light_is_the_exact_si_value_in_m_per_ns() -> None:
    assert SPEED_OF_LIGHT_M_PER_NS == 0.299792458


def test_eps_nine_gives_a_third_of_c() -> None:
    assert velocity_from_eps(9.0) == pytest.approx(0.0999308, abs=1e-6)
    assert velocity_from_eps(1.0) == SPEED_OF_LIGHT_M_PER_NS


@pytest.mark.parametrize("eps_r", [2.03, 9.0, 19.46, 81.0])
def test_velocity_and_eps_round_trip(eps_r: float) -> None:
    assert eps_from_velocity(velocity_from_eps(eps_r)) == pytest.approx(eps_r)


@pytest.mark.parametrize("eps_r", [0.0, -1.0])
def test_velocity_from_eps_rejects_non_positive(eps_r: float) -> None:
    with pytest.raises(ValueError, match="eps_r must be positive"):
        velocity_from_eps(eps_r)


@pytest.mark.parametrize("velocity", [0.0, -0.1])
def test_eps_from_velocity_rejects_non_positive(velocity: float) -> None:
    with pytest.raises(ValueError, match="velocity must be positive"):
        eps_from_velocity(velocity)
