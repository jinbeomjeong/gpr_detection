"""Analytic B-scan generator for unit tests (spec_depth_v1 section 11-1).

Traces sit at ``x_j = j * dx``. A Ricker wavelet is placed at
``t_src + travel_time`` for every event: the pipe reflection follows the
straight-ray cylinder curve (ground-coupled; air-coupled profiles must pass
``pipe_times_ns``), the direct wave sits at ``t_src + 0.7`` ns (air: ``s / c``)
and, for air-coupled profiles, the surface reflection (opposite polarity) at
``t_src + 2 sqrt(h^2 + s^2/4) / c``. White noise is scaled to the SNR
relative to the apex peak. The seed is always 0.
"""

from dataclasses import dataclass, field

import numpy as np

from gpr_detection.io.radargram import SPEED_OF_LIGHT_M_PER_NS, Radargram

DT_NS = 0.05
DX_M = 0.02
F_C_HZ = 400e6
T_SRC_NS = 3.0
DIRECT_WAVE_DELAY_NS = 0.7
SNR_DB = 20.0
SEED = 0


def ricker(t_ns: np.ndarray, f_c_hz: float) -> np.ndarray:
    """Ricker wavelet with its (envelope and signed) peak at ``t = 0``."""
    arg = (np.pi * f_c_hz * 1e-9 * t_ns) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def cylinder_times(
    x_m: np.ndarray, x0_m: float, d_top_m: float, v: float, radius_m: float
) -> np.ndarray:
    """Straight-ray two-way time to a cylinder top, zero offset (ns)."""
    centre = d_top_m + radius_m
    return (2.0 / v) * (np.sqrt(centre**2 + (x_m - x0_m) ** 2) - radius_m)


@dataclass
class BScanSpec:
    """Inputs of :func:`make_bscan`; defaults follow spec section 11-1 (U1)."""

    eps_r: float = 9.0
    d_top_m: float = 0.6
    radius_m: float = 0.0
    n_traces: int = 151
    n_samples: int = 800
    air_gap_m: float = 0.0
    offset_m: float = 0.1
    f_c_hz: float = F_C_HZ
    dt_ns: float = DT_NS
    dx_m: float = DX_M
    pipe_amplitude: float = 1.0
    direct_amplitude: float = 3.0
    # Air -> soil reflection coefficient is negative: opposite polarity to the
    # direct wave, which is what keeps the two envelope peaks apart.
    surface_amplitude: float = -2.0
    snr_db: float | None = SNR_DB
    pipe_times_ns: np.ndarray | None = None
    extra_events: list[tuple[np.ndarray, np.ndarray | float]] = field(
        default_factory=list
    )

    @property
    def velocity(self) -> float:
        return SPEED_OF_LIGHT_M_PER_NS / np.sqrt(self.eps_r)

    @property
    def x_m(self) -> np.ndarray:
        return np.arange(self.n_traces) * self.dx_m

    @property
    def x0_m(self) -> float:
        return float(self.x_m[self.n_traces // 2])


def _place(data: np.ndarray, times_ns: np.ndarray, amplitude, spec: BScanSpec):
    t = np.arange(spec.n_samples)[:, None] * spec.dt_ns
    data += np.asarray(amplitude) * ricker(t - times_ns[None, :], spec.f_c_hz)


def make_bscan(spec: BScanSpec) -> np.ndarray:
    """Return float32 ``[n_samples, n_traces]`` amplitudes for ``spec``."""
    data = np.zeros((spec.n_samples, spec.n_traces))
    ones = np.ones(spec.n_traces)
    if spec.air_gap_m > 0:
        direct_ns = T_SRC_NS + spec.offset_m / SPEED_OF_LIGHT_M_PER_NS
        surface_ns = (
            T_SRC_NS
            + 2.0
            * np.hypot(spec.air_gap_m, spec.offset_m / 2)
            / SPEED_OF_LIGHT_M_PER_NS
        )
        _place(data, surface_ns * ones, spec.surface_amplitude, spec)
    else:
        direct_ns = T_SRC_NS + DIRECT_WAVE_DELAY_NS
    _place(data, direct_ns * ones, spec.direct_amplitude, spec)

    pipe_times = spec.pipe_times_ns
    if pipe_times is None and spec.air_gap_m > 0:
        # cylinder_times has no air leg; air-coupled pipe times must come from
        # the caller (later: the refraction model).
        raise ValueError("air_gap_m > 0 needs explicit pipe_times_ns")
    if pipe_times is None:
        pipe_times = cylinder_times(
            spec.x_m, spec.x0_m, spec.d_top_m, spec.velocity, spec.radius_m
        )
    _place(data, T_SRC_NS + pipe_times, spec.pipe_amplitude, spec)
    for times, amplitude in spec.extra_events:
        _place(data, T_SRC_NS + times, amplitude, spec)

    if spec.snr_db is not None:
        rng = np.random.default_rng(SEED)
        noise_sigma = spec.pipe_amplitude / 10 ** (spec.snr_db / 20)
        data += rng.normal(0.0, noise_sigma, data.shape)
    return data.astype(np.float32)


def make_radargram(spec: BScanSpec, **overrides) -> Radargram:
    """Wrap :func:`make_bscan` in a :class:`Radargram` (ground or air coupled)."""
    fields = {
        "data": make_bscan(spec),
        "dt_ns": spec.dt_ns,
        "dx_m": spec.dx_m,
        "air_gap": "air" if spec.air_gap_m > 0 else "ground",
        "amplitude_processed": False,
        "dataset": "unit",
        "line_id": "unit/line",
        "f_c_hz": spec.f_c_hz,
        "eps_r_meta": spec.eps_r,
        "eps_r_meta_source": "synthetic_truth",
        "antenna_offset_m": spec.offset_m,
    }
    fields.update(overrides)
    return Radargram(**fields)
