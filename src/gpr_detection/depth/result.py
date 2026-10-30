"""Depth result shared by ``pipe_hyperbola`` (M2-4) and ``pipe_band`` (M2-6b).

Fields follow spec_depth_v1 section 2; ``target_class`` comes from
spec_band_depth_v1 section 6-1 (the ``band`` geometry field is added in M2-6b).
List-valued fields are tuples so the result stays immutable.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

AltDepthLabel = Literal["direct_wave", "surface_reflection", "merged", "unclassified"]


class DepthStatus(StrEnum):
    """``DepthResult.status`` values (spec 2; ``no_velocity`` from the band spec)."""

    OK = "ok"
    NO_SCALE = "no_scale"
    INSUFFICIENT_PICKS = "insufficient_picks"
    FIT_FAILED = "fit_failed"
    ZERO_FAILED = "zero_failed"
    NO_VELOCITY = "no_velocity"


@dataclass(frozen=True)
class AltDepth:
    """A depth from another zero or velocity (spec 2 ``AltDepth``, 4-3, 7 rule 3a).

    Attributes:
        t_zero_ns: Zero used for this depth.
        label: Event the zero sits on.
        depth_top_m: Pipe-top depth.
    """

    t_zero_ns: float
    label: AltDepthLabel
    depth_top_m: float


@dataclass(frozen=True)
class DepthResult:
    """Depth of one detection box (spec_depth_v1 section 2).

    Fields that cannot be computed are ``None``; without scale only
    ``apex_trace``/``apex_sample`` (and the zero sample) are filled.

    Attributes:
        status: :class:`DepthStatus` value.
        target_class: ``"pipe_hyperbola"`` (or ``"pipe_band"`` from M2-6b).
        depth_top_m: Main result, pipe-top depth (m).
        depth_center_m: ``depth_top_m + R`` when the radius is known.
        sigma_depth_m: Combined standard uncertainty (section 9).
        bias_notes: Signed systematic biases, not added to the uncertainty.
        v_m_per_ns: Velocity used for the main depth.
        sigma_v: Its standard uncertainty (m/ns).
        eps_r_used: ``(c / v)^2``.
        x0_m: Apex position along the profile.
        t_apex_ns: Apex time on the record clock (``t_a``).
        t_zero_ns: Depth zero (``t_z``).
        t0_ns: ``t_a - t_z``.
        apex_trace: Apex seed trace (L0), also without scale.
        apex_sample: Apex sample (L0, fractional), also without scale.
        zero_rule: Section 4 rule name.
        zero_event_label: Event the zero sits on.
        v_rule: Section 7 rule name.
        radius_used_m: Radius used in the fit (0 or known).
        n_picks: Tracked picks.
        inlier_frac: Curvature fit inlier fraction.
        rms_ns: Residual RMS of the fit that gave ``t0_ns``.
        h_est_m: Antenna height from the two zero events.
        alt_depths: Depths from metadata zeros (section 4-3) and ``meta_fixed``
            (rule 3a, or the fixed-v fallback when the curvature fit failed).
        flags: Processing flags.
        eps_r_refraction_qc: eps_r of the refraction QC at ``h_est`` (rule 2a,
            flag only, never used for the depth).
        eps_r_refraction_qc_range: ``(eps_lo, eps_hi)`` over ``h_est - 0.06 m``
            to ``h_est`` (rule 2a, v1.4).
        sigma_zero_ns: Zero uncertainty used in ``sigma_depth_m``.
        sigma_apex_ns: Apex-time uncertainty used in ``sigma_depth_m``.
    """

    status: DepthStatus
    target_class: str = "pipe_hyperbola"
    depth_top_m: float | None = None
    depth_center_m: float | None = None
    sigma_depth_m: float | None = None
    bias_notes: tuple[str, ...] = ()
    v_m_per_ns: float | None = None
    sigma_v: float | None = None
    eps_r_used: float | None = None
    x0_m: float | None = None
    t_apex_ns: float | None = None
    t_zero_ns: float | None = None
    t0_ns: float | None = None
    apex_trace: int | None = None
    apex_sample: float | None = None
    zero_rule: str | None = None
    zero_event_label: str | None = None
    v_rule: str | None = None
    radius_used_m: float | None = None
    n_picks: int = 0
    inlier_frac: float | None = None
    rms_ns: float | None = None
    h_est_m: float | None = None
    alt_depths: Mapping[str, AltDepth] = field(default_factory=dict)
    flags: tuple[str, ...] = ()
    eps_r_refraction_qc: float | None = None
    eps_r_refraction_qc_range: tuple[float, float] | None = None
    sigma_zero_ns: float | None = None
    sigma_apex_ns: float | None = None
