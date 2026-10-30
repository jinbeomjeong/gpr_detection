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

Band velocity (spec_band_depth_v1 section 5): a band has no curvature, so
:func:`select_band_velocity` borrows a velocity. The order is eligible
``pipe_hyperbola`` results of the same radargram (the velocity pool, ground
only), then ``eps_r_meta``, then the survey-unit median, else ``none``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from gpr_detection.depth.config import (
    DEFAULT_BAND_CONFIG,
    DEFAULT_DEPTH_CONFIG,
    BandConfig,
    DepthConfig,
)
from gpr_detection.depth.hyperbola import CurvatureFit, FitStatus
from gpr_detection.depth.result import DepthResult, DepthStatus
from gpr_detection.io.radargram import Radargram, eps_from_velocity, velocity_from_eps

ELIGIBLE_MIN_PICKS = 9
ELIGIBLE_MIN_INLIER_FRAC = 0.75
ELIGIBLE_MAX_RMS_NS = 0.3
ELIGIBLE_V_RANGE = (0.05, 0.20)
"""Neighbour hyperbola eligibility (band spec 5-2; v range eps_r 2.2-36, assumed)."""

NEAR_DEPTH_REL = 0.5
"""Prefer hyperbolas with ``|tau_hyp - tau_band| <= 0.5 tau_band`` (assumed)."""

FAR_DEPTH_SIGMA_REL = 0.05
"""Extra ``sigma_v / v`` when no hyperbola is near the band depth (assumed)."""

CURVATURE_SIGMA_FLOOR_REL = 0.03
SURVEY_MIN_HYPERBOLAS = 3
SURVEY_SIGMA_FLOOR_REL = 0.10
DATASET_DEFAULT_SIGMA_REL = 0.15
IQR_TO_SIGMA = 1.349

NEIGHBOR_RADIUS_UNKNOWN_NOTE = "neighbor_v_radius_unknown: v up to +17 %"
"""Same bound as the hyperbola note (spec_depth_v1 v1.6 R11: up to 1.165)."""
NEIGHBOR_OBLIQUE_NOTE = "neighbor_v_oblique_unknown: v,d over by 1/sin(theta)"

RADIUS_UNKNOWN_NOTE = "radius_unknown: v,d over by up to +17 %"
"""R = 0 curvature fit with unknown R: v'/v up to 1.17 (spec 7 v1.6, M2-5 R11)."""


class VelocityRule(StrEnum):
    """``v_rule`` names (spec 7 and spec_band_depth_v1 section 5-1)."""

    NONE = "none"
    META_FIXED = "meta_fixed"
    CURVATURE_R_FIXED = "curvature_R_fixed"
    NEIGHBOR_HYPERBOLA = "neighbor_hyperbola"
    SURVEY_HYPERBOLA_MEDIAN = "survey_hyperbola_median"
    DATASET_DEFAULT = "dataset_default"


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


# ----------------------------------------------------------------- band velocity


@dataclass(frozen=True)
class VelocityEstimate:
    """One ``pipe_hyperbola`` velocity offered to the band (band spec 8-2).

    Attributes:
        v_m_per_ns: Curvature velocity.
        sigma_v: Its standard uncertainty (m/ns).
        tau_ns: Apex two-way time after the zero (``t0_ns``).
        line_id: Radargram of the hyperbola.
        survey_key: Survey unit (Guangzhou zone, GPR-SD road), if any.
        radius_known: The fit used a known radius.
        source_index: Index of the hyperbola result in the caller's list.
        eligible: Section 5-2 checks passed.
    """

    v_m_per_ns: float
    sigma_v: float
    tau_ns: float
    line_id: str
    survey_key: str | None
    radius_known: bool
    source_index: int
    eligible: bool


def _is_eligible(result: DepthResult) -> bool:
    """Band spec 5-2 checks on a hyperbola result."""
    if (
        result.status != DepthStatus.OK
        or result.v_rule != VelocityRule.CURVATURE_R_FIXED
    ):
        return False
    v_low, v_high = ELIGIBLE_V_RANGE
    return (
        result.n_picks >= ELIGIBLE_MIN_PICKS
        and result.inlier_frac is not None
        and result.inlier_frac >= ELIGIBLE_MIN_INLIER_FRAC
        and result.rms_ns is not None
        and result.rms_ns <= ELIGIBLE_MAX_RMS_NS
        and "apex_at_box_edge" not in result.flags
        and result.v_m_per_ns is not None
        and v_low <= result.v_m_per_ns <= v_high
    )


def velocity_estimate_from(
    result: DepthResult,
    *,
    line_id: str,
    source_index: int,
    survey_key: str | None = None,
) -> VelocityEstimate | None:
    """Turn a ``pipe_hyperbola`` result into a pool entry (``None`` without v).

    Both-side pick counts are implied: ``curvature_R_fixed`` only runs when
    every side has enough picks (spec_depth_v1 section 5-4).
    """
    if result.v_m_per_ns is None or result.t0_ns is None or result.sigma_v is None:
        return None
    return VelocityEstimate(
        v_m_per_ns=result.v_m_per_ns,
        sigma_v=result.sigma_v,
        tau_ns=result.t0_ns,
        line_id=line_id,
        survey_key=survey_key,
        radius_known=bool(result.radius_used_m),
        source_index=source_index,
        eligible=_is_eligible(result),
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
    return float(values[order][index])


def _iqr(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def _neighbor_choice(
    usable: Sequence[VelocityEstimate], tau_band_ns: float, v_meta: float | None
) -> VelocityChoice:
    """Rule 2 (2a): weighted median of eligible same-line hyperbolas."""
    near = [
        e for e in usable if abs(e.tau_ns - tau_band_ns) <= NEAR_DEPTH_REL * tau_band_ns
    ]
    chosen = near or list(usable)
    depth_sigma_rel = 0.0 if near else FAR_DEPTH_SIGMA_REL
    velocities = np.array([e.v_m_per_ns for e in chosen])
    weights = np.array([1.0 / max(e.sigma_v, 1e-9) ** 2 for e in chosen])
    v = _weighted_median(velocities, weights)
    statistical_rel = 1.0 / np.sqrt(weights.sum()) / v
    spread_rel = _iqr(velocities) / IQR_TO_SIGMA / v
    sigma_rel = float(
        np.sqrt(
            statistical_rel**2
            + spread_rel**2
            + CURVATURE_SIGMA_FLOOR_REL**2
            + depth_sigma_rel**2
        )
    )
    notes = [NEIGHBOR_OBLIQUE_NOTE]
    if any(not e.radius_known for e in chosen):
        notes.insert(0, NEIGHBOR_RADIUS_UNKNOWN_NOTE)
    return VelocityChoice(
        v_rule=VelocityRule.NEIGHBOR_HYPERBOLA,
        v_m_per_ns=v,
        sigma_v=sigma_rel * v,
        eps_r_used=eps_from_velocity(v),
        v_sources=tuple(e.source_index for e in chosen),
        alt_meta_v=v_meta,
        bias_notes=tuple(notes),
    )


def _survey_choice(survey: Sequence[VelocityEstimate]) -> VelocityChoice:
    """Rule 4: survey-unit median, ``sigma_v / v = max(0.10, IQR / 1.349 / v)``."""
    velocities = np.array([e.v_m_per_ns for e in survey])
    v = float(np.median(velocities))
    sigma_rel = max(SURVEY_SIGMA_FLOOR_REL, _iqr(velocities) / IQR_TO_SIGMA / v)
    return VelocityChoice(
        v_rule=VelocityRule.SURVEY_HYPERBOLA_MEDIAN,
        v_m_per_ns=v,
        sigma_v=sigma_rel * v,
        eps_r_used=eps_from_velocity(v),
        v_sources=tuple(e.source_index for e in survey),
        bias_notes=(NEIGHBOR_OBLIQUE_NOTE,),
    )


def _dataset_default_choice(eps_r: float) -> VelocityChoice:
    v = velocity_from_eps(eps_r)
    return VelocityChoice(
        v_rule=VelocityRule.DATASET_DEFAULT,
        v_m_per_ns=v,
        sigma_v=DATASET_DEFAULT_SIGMA_REL * v,
        eps_r_used=eps_r,
        flags=("v_assumed",),
    )


def select_band_velocity(
    radargram: Radargram,
    tau_band_ns: float,
    pool: Sequence[VelocityEstimate],
    survey_pool: Sequence[VelocityEstimate],
    cfg: BandConfig = DEFAULT_BAND_CONFIG,
) -> VelocityChoice:
    """Band velocity rules (spec_band_depth_v1 section 5-1).

    Args:
        radargram: Scale, coupling, ``eps_r_meta`` and ``line_id``.
        tau_band_ns: Band two-way time after the zero (near-depth preference).
        pool: Hyperbola velocities of this radargram.
        survey_pool: Hyperbola velocities of the same survey unit.
        cfg: ``dataset_default_eps`` (off by default, user decision Q1).
    """
    if not radargram.has_physical_scale or radargram.air_gap == "unknown":
        return _no_velocity()
    v_meta = meta_velocity(radargram)
    if radargram.air_gap == "air":
        # Air-coupled curvature is forbidden, so neighbour v is never used.
        return (
            _no_velocity()
            if v_meta is None
            else _meta_fixed(v_meta, DEFAULT_DEPTH_CONFIG)
        )

    same_line = [e for e in pool if e.eligible and e.line_id == radargram.line_id]
    if same_line:
        return _neighbor_choice(same_line, tau_band_ns, v_meta)
    if v_meta is not None:
        return _meta_fixed(v_meta, DEFAULT_DEPTH_CONFIG)
    survey = [e for e in survey_pool if e.eligible]
    if len(survey) >= SURVEY_MIN_HYPERBOLAS:
        return _survey_choice(survey)
    if cfg.dataset_default_eps is not None:
        return _dataset_default_choice(cfg.dataset_default_eps)
    return _no_velocity()
