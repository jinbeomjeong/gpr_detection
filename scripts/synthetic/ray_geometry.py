"""Ray (geometric-optics) travel times for the synthetic pipe models.

Frame: x horizontal, z vertical up, ground surface at z = 0, soil below.
Antennas are at height h >= 0 above the surface. The pipe is a circle with
centre (xc, zc), zc = -(d_top + R), radius R. The top-surface (specular)
reflection time is the minimum, over points P on the upper half of the circle,
of leg(Tx, P) + leg(P, Rx), where each leg obeys Fermat's principle across the
air-soil interface (refraction point found by bisection).
"""

import numpy as np

C0_M_PER_NS = 0.299792458


def leg_time(xa, h, px, pz, v):
    """Fastest air->soil path time (ns) from antenna (xa, h) to soil points (px, pz <= 0)."""
    px = np.asarray(px, dtype=float)
    pz = np.asarray(pz, dtype=float)
    if h <= 1e-12:
        return np.hypot(px - xa, pz) / v
    lo = np.minimum(xa, px)
    hi = np.maximum(xa, px)
    depth = np.abs(pz)
    for _ in range(64):
        a = 0.5 * (lo + hi)
        # derivative of total time w.r.t. crossing point a (monotonically increasing)
        g = (a - xa) / (C0_M_PER_NS * np.hypot(a - xa, h)) + (a - px) / (v * np.hypot(a - px, depth))
        hi = np.where(g > 0, a, hi)
        lo = np.where(g > 0, lo, a)
    a = 0.5 * (lo + hi)
    return np.hypot(a - xa, h) / C0_M_PER_NS + np.hypot(a - px, depth) / v


def reflection_time(x_tx, x_rx, h, xc, zc, radius, v, n_theta=4001):
    """Bistatic specular top reflection time (ns) and specular point for one Tx/Rx pair."""

    def total(theta):
        px = xc + radius * np.cos(theta)
        pz = zc + radius * np.sin(theta)
        return leg_time(x_tx, h, px, pz, v) + leg_time(x_rx, h, px, pz, v)

    theta = np.linspace(0.0, np.pi, n_theta)
    t = total(theta)
    k = int(np.argmin(t))
    fine = np.linspace(theta[max(k - 1, 0)], theta[min(k + 1, n_theta - 1)], 2001)
    tf = total(fine)
    j = int(np.argmin(tf))
    th = fine[j]
    return float(tf[j]), (float(xc + radius * np.cos(th)), float(zc + radius * np.sin(th)))


def trace_reflection_times(x_mid, offset, h, xc, d_top, radius, v):
    """Reflection times (ns) for all trace midpoints x_mid (common offset)."""
    zc = -(d_top + radius)
    out = np.empty(len(x_mid))
    for i, xm in enumerate(x_mid):
        out[i], _ = reflection_time(xm - offset / 2, xm + offset / 2, h, xc, zc, radius, v)
    return out


def surface_reflection_time(offset, h):
    """Specular ground-surface reflection, Tx/Rx at height h, separation offset (ns)."""
    return 2.0 * np.hypot(h, offset / 2) / C0_M_PER_NS


def direct_air_time(offset):
    return offset / C0_M_PER_NS
