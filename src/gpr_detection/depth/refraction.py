"""Air-gap (refraction) ray model and the refraction QC (spec_depth_v1 6-1, 7 rule 2a).

Frame: ``x`` along the profile, ``z`` up, ground surface at ``z = 0``, soil
below. Tx and Rx sit at height ``h >= 0`` and ``x -/+ s/2`` around the trace
midpoint. The pipe is a circle of radius ``R`` with centre ``(x0, -(d_top + R))``.

The top reflection time is the minimum over points ``P`` on the upper half of
the circle of ``leg(Tx, P) + leg(P, Rx)``. Each leg obeys Fermat's principle
across the air-soil interface: its refraction point is found by bisection on
the derivative of the leg time, which increases monotonically.

The algorithm follows ``scripts/synthetic/ray_geometry.py`` (``leg_time``,
``reflection_time``) and is re-implemented here because ``src`` never imports
``scripts``. Times are in ns, lengths in m, velocities in m/ns.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from gpr_detection.depth.config import DEFAULT_DEPTH_CONFIG, DepthConfig
from gpr_detection.io.radargram import SPEED_OF_LIGHT_M_PER_NS, velocity_from_eps

BISECTION_STEPS = 40
"""Refraction-point bisection steps per leg (spec 6-1: 40 are enough)."""

ANGLE_GRID_POINTS = 721
"""Points of the angle grid on the pipe's upper half circle.

Single stage, spec 6-1 v1.4.
"""

REFINE_EPS_TOLERANCE = 1e-3
"""``minimize_scalar`` eps_r tolerance after the grid (assumed; QC precision)."""

_DEPTH_BISECTION_STEPS = 40
_DEPTH_SEARCH_M = (0.0, 10.0)
"""Pipe-top depth range searched when the apex time fixes ``d`` (assumed)."""

_FLAT_AIR_GAP_M = 1e-12
"""Below this antenna height the legs are straight soil paths."""


def leg_time(
    x_antenna: np.ndarray | float,
    h: float,
    px: np.ndarray | float,
    pz: np.ndarray | float,
    v: np.ndarray | float,
) -> np.ndarray:
    """Fastest air -> soil travel time from an antenna to soil points (ns).

    Args:
        x_antenna: Antenna position(s) along x, at height ``h``.
        h: Antenna height above the ground (m).
        px: Soil point x (m); broadcast with ``x_antenna``, ``pz`` and ``v``.
        pz: Soil point z (m, <= 0).
        v: Soil velocity (m/ns), scalar or broadcastable array.
    """
    x_antenna, px, pz, v = np.broadcast_arrays(
        np.asarray(x_antenna, dtype=float),
        np.asarray(px, dtype=float),
        np.asarray(pz, dtype=float),
        np.asarray(v, dtype=float),
    )
    if h <= _FLAT_AIR_GAP_M:
        return np.hypot(px - x_antenna, pz) / v
    depth = np.abs(pz)
    lo = np.minimum(x_antenna, px)
    hi = np.maximum(x_antenna, px)
    for _ in range(BISECTION_STEPS):
        crossing = 0.5 * (lo + hi)
        # d(time)/d(crossing) is increasing; its root is Snell's law.
        slope = (crossing - x_antenna) / (
            SPEED_OF_LIGHT_M_PER_NS * np.hypot(crossing - x_antenna, h)
        ) + (crossing - px) / (v * np.hypot(crossing - px, depth))
        past_root = slope > 0
        hi = np.where(past_root, crossing, hi)
        lo = np.where(past_root, lo, crossing)
    crossing = 0.5 * (lo + hi)
    air_ns = np.hypot(crossing - x_antenna, h) / SPEED_OF_LIGHT_M_PER_NS
    return air_ns + np.hypot(crossing - px, depth) / v


def _two_leg_time(x_tx, x_rx, h, px, pz, v) -> np.ndarray:
    return leg_time(x_tx, h, px, pz, v) + leg_time(x_rx, h, px, pz, v)


def reflection_times(
    x_mid_m: np.ndarray,
    *,
    offset_m: float,
    h: float,
    x0_m: float,
    d_top_m: float,
    radius_m: float,
    v: float,
) -> np.ndarray:
    """Specular top-reflection time for every trace midpoint (ns, from transmit).

    For ``R > 0`` the minimum over a 721-point angle grid on the upper half
    circle is taken. ``ray_geometry.reflection_time`` refines the angle on a
    second grid; that step is left out because only the time is needed and the
    time is stationary at the specular point: an angle error of half a grid
    step (pi/1440) changes the time by about 1e-6 ns for R = 0.1 m.
    ``R = 0`` is a point scatterer.
    """
    x_mid = np.asarray(x_mid_m, dtype=float)[:, None]
    x_tx, x_rx = x_mid - offset_m / 2.0, x_mid + offset_m / 2.0
    centre_z = -(d_top_m + radius_m)
    if radius_m <= 0:
        return _two_leg_time(x_tx, x_rx, h, x0_m, centre_z, v)[:, 0]

    theta = np.linspace(0.0, np.pi, ANGLE_GRID_POINTS)[None, :]
    px = x0_m + radius_m * np.cos(theta)
    pz = centre_z + radius_m * np.sin(theta)
    return _two_leg_time(x_tx, x_rx, h, px, pz, v).min(axis=1)


def surface_reflection_time(h: float, offset_m: float) -> float:
    """Specular ground-surface reflection ``2 sqrt(h^2 + s^2/4) / c`` (ns)."""
    return 2.0 * float(np.hypot(h, offset_m / 2.0)) / SPEED_OF_LIGHT_M_PER_NS


def _apex_depths(
    tau_apex_ns: float, h: float, offset_m: float, velocities: np.ndarray
) -> np.ndarray:
    """``d_top`` per velocity so that the apex time after the surface zero matches.

    At the apex the Tx/Rx pair is symmetric about the pipe, so the specular
    point is the circle top ``(x0, -d_top)``; the radius does not enter.
    """
    # Symmetric pair: both legs take the same time, so one leg is half the target.
    target_leg_ns = (tau_apex_ns + surface_reflection_time(h, offset_m)) / 2.0
    lo = np.full_like(velocities, _DEPTH_SEARCH_M[0])
    hi = np.full_like(velocities, _DEPTH_SEARCH_M[1])
    for _ in range(_DEPTH_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        too_late = leg_time(-offset_m / 2.0, h, 0.0, -mid, velocities) > target_leg_ns
        hi = np.where(too_late, mid, hi)
        lo = np.where(too_late, lo, mid)
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class RefractionQC:
    """Refraction-model eps_r check of an air-coupled hyperbola (7 rule 2a).

    QC only: the main depth stays ``meta_fixed``.

    Attributes:
        eps_fit: eps_r with the smallest pick RMS.
        v_fit: ``c / sqrt(eps_fit)`` (m/ns).
        d_top_m: Depth that reproduces the apex time at ``eps_fit``.
        rms_ns: Pick RMS at ``eps_fit``.
        n_picks: Picks used.
        h_m: Antenna height used (from the two zero events).
        offset_m: Tx-Rx separation used (0 when unknown).
    """

    eps_fit: float
    v_fit: float
    d_top_m: float
    rms_ns: float
    n_picks: int
    h_m: float
    offset_m: float


def refraction_qc(
    x_m: np.ndarray,
    tau_ns: np.ndarray,
    *,
    x0_m: float,
    tau_apex_ns: float,
    h_m: float,
    offset_m: float,
    radius_m: float,
    config: DepthConfig = DEFAULT_DEPTH_CONFIG,
) -> RefractionQC:
    """Grid eps_r 3-30 (step 0.25), then ``minimize_scalar`` around the best.

    For each eps_r the depth is tied to the apex time, and the RMS of
    ``tau_model(x) - tau`` over all picks is computed with
    ``tau_model = T_reflection - T_surface`` (zero = surface reflection).

    Args:
        x_m: Pick positions.
        tau_ns: Pick times after the surface-reflection zero.
        x0_m: Apex position.
        tau_apex_ns: Apex time after the zero.
        h_m: Antenna height (from the direct/surface gap).
        offset_m: Tx-Rx separation (0 when unknown).
        radius_m: Fixed pipe radius.
        config: eps_r grid settings.
    """
    x = np.asarray(x_m, dtype=float)
    tau = np.asarray(tau_ns, dtype=float)
    surface_ns = surface_reflection_time(h_m, offset_m)

    def pick_rms(v: float, d_top: float) -> float:
        model = reflection_times(
            x,
            offset_m=offset_m,
            h=h_m,
            x0_m=x0_m,
            d_top_m=d_top,
            radius_m=radius_m,
            v=v,
        )
        return float(np.sqrt(np.mean((model - surface_ns - tau) ** 2)))

    def rms_at(eps_r: float) -> float:
        v = velocity_from_eps(eps_r)
        d_top = float(_apex_depths(tau_apex_ns, h_m, offset_m, np.array([v]))[0])
        return pick_rms(v, d_top)

    grid = np.arange(
        config.refraction_eps_min,
        config.refraction_eps_max + 0.5 * config.refraction_eps_step,
        config.refraction_eps_step,
    )
    grid_v = SPEED_OF_LIGHT_M_PER_NS / np.sqrt(grid)
    # One vectorised depth solve for the whole grid (the costly part per eps).
    grid_depths = _apex_depths(tau_apex_ns, h_m, offset_m, grid_v)
    grid_rms = [pick_rms(v, d) for v, d in zip(grid_v, grid_depths, strict=True)]
    best = int(np.argmin(grid_rms))
    bracket = (
        float(grid[max(best - 1, 0)]),
        float(grid[min(best + 1, len(grid) - 1)]),
    )
    refined = minimize_scalar(
        rms_at,
        bounds=bracket,
        method="bounded",
        options={"xatol": REFINE_EPS_TOLERANCE},
    )
    eps_fit = float(refined.x)
    v_fit = velocity_from_eps(eps_fit)
    d_top = float(_apex_depths(tau_apex_ns, h_m, offset_m, np.array([v_fit]))[0])
    rms_ns = pick_rms(v_fit, d_top)
    return RefractionQC(
        eps_fit=eps_fit,
        v_fit=v_fit,
        d_top_m=d_top,
        rms_ns=rms_ns,
        n_picks=len(x),
        h_m=h_m,
        offset_m=offset_m,
    )
