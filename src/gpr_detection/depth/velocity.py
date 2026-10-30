"""Velocity rules for one hyperbola (spec_depth_v1 section 7).

:func:`select_hyperbola_velocity` turns the radargram metadata and the
curvature fit of one box into a :class:`VelocityChoice`. ``estimate_depth``
only calls it. M2-6b adds the band-side velocity pool here
(``VelocityEstimate``, ``select_band_velocity``, spec_band_depth_v1 section 5)
and reuses :class:`VelocityChoice` for it.

Decision table (spec 7):

====  =====================================  ======================
rule  condition                              ``v_rule``
====  =====================================  ======================
1     no ``dt_ns`` or ``dx_m``               ``none`` (no_scale)
2     ``air_gap="air"``                      ``meta_fixed``
2b    air, no ``eps_r_meta``                 ``none`` (no_velocity)
3     ground, curvature fit ok               ``curvature_R_fixed``
3a    3 + ``eps_r_meta``                     3, meta v in ``alt_meta_v``
4     ground, too few picks, meta eps_r      ``meta_fixed``
5     ground, too few picks, no meta         ``none``
6     ``air_gap="unknown"`` with scale       ``none``, meta v in ``alt_meta_v``
====  =====================================  ======================

A failed curvature fit gives ``none`` with the metadata velocity, if any, in
``alt_meta_v`` (section 10).
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from gpr_detection.depth.config import DEFAULT_DEPTH_CONFIG, DepthConfig
from gpr_detection.depth.hyperbola import CurvatureFit, FitStatus
from gpr_detection.io.radargram import Radargram, eps_from_velocity, velocity_from_eps

RADIUS_UNKNOWN_NOTE = "radius_unknown: v,d over by up to +17 %"
"""R = 0 curvature fit with unknown R: v'/v up to 1.17 (spec 7 v1.6, M2-5 R11)."""


class VelocityRule(StrEnum):
    """``v_rule`` names (spec 7; band rules join in M2-6b)."""

    NONE = "none"
    META_FIXED = "meta_fixed"
    CURVATURE_R_FIXED = "curvature_R_fixed"


@dataclass(frozen=True)
class VelocityChoice:
    """Chosen velocity of one target (field set of spec_band_depth_v1 8-2).

    Attributes:
        v_rule: :class:`VelocityRule` value.
        v_m_per_ns: Velocity for the main depth; ``None`` for ``none``.
        sigma_v: Its standard uncertainty (m/ns).
        eps_r_used: ``(c / v)^2``.
        v_sources: Indices of pooled hyperbolas used (band only; empty here).
        alt_meta_v: Metadata velocity for ``alt_depths["meta_fixed"]`` when the
            main result is not ``meta_fixed`` (rule 3a, failed fit).
        bias_notes: e.g. ``radius_unknown: ...``.
        flags: e.g. ``meta_eps_unverified``.
    """

    v_rule: VelocityRule
    v_m_per_ns: float | None
    sigma_v: float | None
    eps_r_used: float | None
    v_sources: tuple[int, ...] = ()
    alt_meta_v: float | None = None
    bias_notes: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()

    @property
    def has_velocity(self) -> bool:
        """True when a main depth can be computed."""
        return self.v_m_per_ns is not None


def meta_velocity(radargram: Radargram) -> float | None:
    """``c / sqrt(eps_r_meta)``, or ``None`` without a metadata eps_r."""
    if radargram.eps_r_meta is None:
        return None
    return velocity_from_eps(radargram.eps_r_meta)


def _no_velocity(alt_meta_v: float | None = None) -> VelocityChoice:
    return VelocityChoice(VelocityRule.NONE, None, None, None, alt_meta_v=alt_meta_v)


def _meta_fixed(v_meta: float, config: DepthConfig) -> VelocityChoice:
    """Rule 2/4: metadata eps_r; ``sigma_v / v = sigma_eps / (2 eps)`` (section 9)."""
    sigma_v_rel = 0.5 * config.sigma_eps_rel
    return VelocityChoice(
        v_rule=VelocityRule.META_FIXED,
        v_m_per_ns=v_meta,
        sigma_v=sigma_v_rel * v_meta,
        eps_r_used=eps_from_velocity(v_meta),
        flags=("meta_eps_unverified",),
    )


def _curvature(
    curvature: CurvatureFit,
    radius_known: bool,
    v_meta: float | None,
    config: DepthConfig,
) -> VelocityChoice:
    """Rule 3 (3a): fitted v with ``sqrt((sigma_fit / v)^2 + 0.03^2)`` (section 9)."""
    fit = curvature.fit
    assert fit is not None  # status OK guarantees a fit
    v = fit.v_m_per_ns
    fit_rel = fit.sigma_v / v if np.isfinite(fit.sigma_v) else 0.0
    sigma_v_rel = float(np.hypot(fit_rel, config.curvature_sigma_floor))
    return VelocityChoice(
        v_rule=VelocityRule.CURVATURE_R_FIXED,
        v_m_per_ns=v,
        sigma_v=sigma_v_rel * v,
        eps_r_used=eps_from_velocity(v),
        alt_meta_v=v_meta,
        bias_notes=() if radius_known else (RADIUS_UNKNOWN_NOTE,),
    )


def select_hyperbola_velocity(
    radargram: Radargram,
    curvature: CurvatureFit | None,
    config: DepthConfig = DEFAULT_DEPTH_CONFIG,
) -> VelocityChoice:
    """Apply the section 7 table to one hyperbola.

    Args:
        radargram: Scale, coupling, ``eps_r_meta`` and ``pipe_radius_m``.
        curvature: Result of ``fit_curvature``; ``None`` for air-coupled data,
            where the straight-ray curvature fit is forbidden.
        config: ``sigma_eps_rel`` and ``curvature_sigma_floor``.

    Returns:
        The choice; ``v_rule="none"`` when no depth can be given.
    """
    if not radargram.has_physical_scale:
        return _no_velocity()
    v_meta = meta_velocity(radargram)
    if radargram.air_gap == "unknown":
        # Rule 6: could be air-coupled, where a failed curvature fit has a small
        # residual; no main velocity, meta v only as an alternative.
        return _no_velocity(alt_meta_v=v_meta)
    if radargram.air_gap == "air" or curvature is None:
        return _no_velocity() if v_meta is None else _meta_fixed(v_meta, config)

    if curvature.status == FitStatus.OK:
        radius_known = radargram.pipe_radius_m is not None
        return _curvature(curvature, radius_known, v_meta, config)
    if curvature.status == FitStatus.INSUFFICIENT_PICKS and v_meta is not None:
        return _meta_fixed(v_meta, config)
    # Rule 5, or a failed fit: no main velocity; meta v only as an alternative.
    return _no_velocity(alt_meta_v=v_meta)
