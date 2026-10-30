"""Physical B-scan container shared by all loaders (spec_depth_v1 section 1-1)."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import numpy as np

AirGap = Literal["air", "ground", "unknown"]
_AIR_GAP_VALUES: tuple[str, ...] = get_args(AirGap)

SPEED_OF_LIGHT_M_PER_NS = 0.299792458
"""Vacuum speed of light ``c`` in m/ns (spec_depth_v1 symbol table)."""


def velocity_from_eps(eps_r: float) -> float:
    """Wave speed ``v = c / sqrt(eps_r)`` in m/ns for a relative permittivity."""
    if not eps_r > 0:
        raise ValueError(f"eps_r must be positive, got {eps_r}")
    return SPEED_OF_LIGHT_M_PER_NS / float(np.sqrt(eps_r))


def eps_from_velocity(v_m_per_ns: float) -> float:
    """Relative permittivity ``(c / v)^2`` for a wave speed in m/ns."""
    if not v_m_per_ns > 0:
        raise ValueError(f"velocity must be positive, got {v_m_per_ns}")
    return (SPEED_OF_LIGHT_M_PER_NS / v_m_per_ns) ** 2


@dataclass(frozen=True)
class Radargram:
    """One GPR profile with the physical scale needed for depth estimation.

    Attributes:
        data: Raw amplitudes, float32, shape ``[n_samples, n_traces]`` (time first).
        dt_ns: Sample interval in ns. ``None`` forbids depth output.
        dx_m: Trace interval in m. ``None`` forbids depth output.
        air_gap: ``"air"`` for air-launched antennas (4TU), ``"ground"`` for
            ground-coupled ones, ``"unknown"`` when not recorded.
        amplitude_processed: True when background removal or gain was already
            applied upstream (GPR-SD, Guangzhou MALA).
        dataset: Dataset key, e.g. ``"4tu"``.
        line_id: Profile identifier inside the dataset, e.g. ``"01.1/Path3"``.
        f_c_hz: Antenna centre frequency in Hz, if known.
        eps_r_meta: Relative permittivity from metadata. Not a verified value.
        eps_r_meta_source: Where ``eps_r_meta`` came from, e.g.
            ``"4tu_metadata_csv"``. Empty when ``eps_r_meta`` is None.
        antenna_offset_m: Tx-Rx separation in m, if known.
        t0_candidates_ns: Time-zero values given by metadata, already converted
            from sample index to ns (``sample * dt_ns``). Keys name the source
            field, e.g. ``"t0_sample_est"``.
        pipe_radius_m: Known pipe radius (outer diameter / 2) in m.
        metadata: Remaining source metadata, unmodified, for traceability.
    """

    data: np.ndarray
    dt_ns: float | None
    dx_m: float | None
    air_gap: AirGap
    amplitude_processed: bool
    dataset: str
    line_id: str
    f_c_hz: float | None = None
    eps_r_meta: float | None = None
    eps_r_meta_source: str = ""
    antenna_offset_m: float | None = None
    t0_candidates_ns: Mapping[str, float] = field(default_factory=dict)
    pipe_radius_m: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(
                f"data must be 2-D [samples, traces], got {self.data.shape}"
            )
        if self.data.dtype != np.float32:
            raise TypeError(f"data must be float32, got {self.data.dtype}")
        if self.air_gap not in _AIR_GAP_VALUES:
            raise ValueError(
                f"air_gap must be one of {_AIR_GAP_VALUES}, got {self.air_gap!r}"
            )
        for name in ("dt_ns", "dx_m"):
            value = getattr(self, name)
            if value is not None and not value > 0:
                raise ValueError(f"{name} must be positive or None, got {value}")

    @property
    def has_physical_scale(self) -> bool:
        """True when ``dt_ns`` and ``dx_m`` are both known (depth in m allowed)."""
        return self.dt_ns is not None and self.dx_m is not None

    @property
    def n_samples(self) -> int:
        """Number of time samples per trace."""
        return self.data.shape[0]

    @property
    def n_traces(self) -> int:
        """Number of traces along the profile."""
        return self.data.shape[1]


@dataclass(frozen=True)
class GridBox:
    """Box on the original radargram grid (coordinate layer L0).

    Closed integer intervals: the box covers traces ``i0..i1`` and samples
    ``k0..k1`` inclusive, i.e. ``radargram.data[k0:k1 + 1, i0:i1 + 1]``. Depth
    estimation only receives boxes in this layer
    (``docs/architecture/README.md`` section 2-3).

    Attributes:
        i0: First trace index.
        i1: Last trace index (inclusive).
        k0: First sample index (earliest time).
        k1: Last sample index (inclusive).
    """

    i0: int
    i1: int
    k0: int
    k1: int

    def __post_init__(self) -> None:
        if min(self.i0, self.k0) < 0:
            raise ValueError(f"indices must be >= 0, got {self}")
        if self.i1 < self.i0 or self.k1 < self.k0:
            raise ValueError(f"need i0 <= i1 and k0 <= k1, got {self}")

    @property
    def n_traces(self) -> int:
        """Number of traces covered."""
        return self.i1 - self.i0 + 1

    @property
    def n_samples(self) -> int:
        """Number of samples covered."""
        return self.k1 - self.k0 + 1

    @property
    def trace_slice(self) -> slice:
        """Column slice for ``data[:, trace_slice]``."""
        return slice(self.i0, self.i1 + 1)

    @property
    def sample_slice(self) -> slice:
        """Row slice for ``data[sample_slice, :]``."""
        return slice(self.k0, self.k1 + 1)


def sample_to_ns(sample_index: float, dt_ns: float) -> float:
    """Convert a sample index from metadata to a time in ns."""
    return float(sample_index) * dt_ns
