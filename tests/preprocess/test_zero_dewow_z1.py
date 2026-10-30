"""Z1 (spec_depth_v1 section 11-1): dewow must not create a second zero event.

The same data also checks that the test can tell the kernels apart: with the
v1 box dewow (3 / f_c moving average) a ghost lobe becomes a second event.
"""

import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

from gpr_detection.preprocess import filters
from gpr_detection.preprocess.filters import prepare_fitting_branch
from gpr_detection.preprocess.timezero import find_time_zero
from synthetic_bscan import BScanSpec, make_radargram, ricker

Z1_F_C_HZ = 500e6
Z1_DT_NS = 0.098
Z1_N_SAMPLES = 512
Z1_EVENT_NS = 5.0
Z1_TAIL_AMPLITUDE = 0.8
Z1_TAIL_DECAY_NS = 1.5
Z1_N_TRACES = 20

V1_BOX_WINDOW_PERIODS = 3.0
"""v1 dewow window ``T_w = 3 / f_c`` (replaced by the Gaussian in v1.1)."""


def _z1_radargram():
    """Single Ricker plus a slow positive tail, air-coupled."""
    time_ns = np.arange(Z1_N_SAMPLES) * Z1_DT_NS
    after_event = time_ns >= Z1_EVENT_NS
    tail = np.where(
        after_event,
        Z1_TAIL_AMPLITUDE
        * np.exp(-(time_ns - Z1_EVENT_NS) / Z1_TAIL_DECAY_NS * after_event),
        0.0,
    )
    trace = ricker(time_ns - Z1_EVENT_NS, Z1_F_C_HZ) + tail
    data = np.tile(trace[:, None], (1, Z1_N_TRACES)).astype(np.float32)
    return make_radargram(
        BScanSpec(), data=data, dt_ns=Z1_DT_NS, f_c_hz=Z1_F_C_HZ, air_gap="air"
    )


def _v1_box_dewow(data: np.ndarray, dt_ns: float, f_c_hz: float) -> np.ndarray:
    """The v1 dewow: subtract a centred box average of odd length 3 / f_c."""
    values = np.asarray(data, dtype=np.float64)
    window = max(1, round(V1_BOX_WINDOW_PERIODS * 1e9 / f_c_hz / dt_ns))
    if window % 2 == 0:
        window += 1
    return values - uniform_filter1d(values, size=window, axis=0, mode="reflect")


def _z1_zero():
    radargram = _z1_radargram()
    return find_time_zero(radargram, prepare_fitting_branch(radargram).filtered)


def test_z1_slow_tail_leaves_a_single_air_coupled_event() -> None:
    """Z1 (spec section 11-1): a box dewow made a ghost lobe here."""
    zero = _z1_zero()

    assert len(zero.events_ns) == 1
    assert zero.zero_rule == "air_single_event"


def test_z1_box_dewow_would_split_the_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discrimination check: only dewow changes, and the box kernel fails Z1."""
    gaussian = _z1_zero()
    monkeypatch.setattr(filters, "dewow", _v1_box_dewow)
    box = _z1_zero()

    assert len(gaussian.events_ns) == 1
    assert gaussian.zero_rule == "air_single_event"
    assert len(box.events_ns) == 2
    assert box.zero_rule == "air_surface_late"
