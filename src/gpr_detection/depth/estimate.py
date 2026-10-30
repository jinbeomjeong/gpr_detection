"""Depth of one ``pipe_hyperbola`` box (spec_depth_v1 sections 2, 7-10).

:func:`estimate_depth` runs, for one radargram and one L0 box::

    prepare_fitting_branch -> find_time_zero -> extract_picks
        -> fit_curvature (ground only) -> select_hyperbola_velocity
        -> apex time -> depth (section 8) -> uncertainty (section 9)

The preprocessing and the zero belong to the whole radargram; pass them in as
:class:`DepthInputs` (:func:`prepare_depth_inputs`) when a radargram has
several boxes, so they are computed once.
"""

from dataclasses import dataclass

import numpy as np

from gpr_detection.depth.config import DEFAULT_DEPTH_CONFIG, DepthConfig
from gpr_detection.depth.hyperbola import (
    CurvatureFit,
    FitStatus,
    fit_curvature,
    fixed_velocity_apex,
)
from gpr_detection.depth.picks import HyperbolaPicks, extract_picks
from gpr_detection.depth.refraction import refraction_qc
from gpr_detection.depth.result import AltDepth, AltDepthLabel, DepthResult, DepthStatus
from gpr_detection.depth.velocity import (
    VelocityChoice,
    VelocityRule,
    meta_velocity,
    select_hyperbola_velocity,
)
from gpr_detection.io.radargram import GridBox, Radargram, eps_from_velocity
from gpr_detection.preprocess.filters import (
    FittingBranch,
    period_ns_or_default,
    prepare_fitting_branch,
)
from gpr_detection.preprocess.timezero import (
    DEFAULT_PERIOD_NS,
    ZeroResult,
    ZeroRule,
    find_time_zero,
)

SIGMA_APEX_FLOOR_NS = 0.05
"""Lower bound of the apex-time uncertainty (5 mm grid delay, section 9)."""

NARROW_PEAK_SAMPLES = 3.0
"""Envelope peaks this few samples wide get ``sigma_a >= dt / 2`` (section 10)."""

INT16_FULL_SCALE = 32767.0
SATURATION_HALF_WINDOW_SAMPLES = 2
"""``saturated_apex`` when a full-scale sample lies within +-2 samples of the apex."""

_ALT_LABELS: frozenset[str] = frozenset(
    {"direct_wave", "surface_reflection", "merged", "unclassified"}
)


@dataclass(frozen=True)
class DepthInputs:
    """Per-radargram inputs shared by all its boxes.

    Attributes:
        branch: Fitting-branch sections (steps (1)-(4)).
        zero: Time zero of the radargram (pick mode = ``config.pick_mode``).
    """

    branch: FittingBranch
    zero: ZeroResult


@dataclass(frozen=True)
class _ApexTime:
    """Apex used for the main depth: from the curvature fit or section 6-4."""

    x0_m: float
    t0_ns: float
    rms_ns: float
    n_points: int


def prepare_depth_inputs(
    radargram: Radargram,
    config: DepthConfig = DEFAULT_DEPTH_CONFIG,
    *,
    exclude_traces: np.ndarray | None = None,
) -> DepthInputs:
    """Preprocess once and find the time zero of one radargram.

    Args:
        radargram: The profile.
        config: Background window, zero settings and pick mode.
        exclude_traces: Traces left out of the zero mean trace
            (``pipe_band`` boxes, spec_band_depth_v1 section 3-1).
    """
    branch = prepare_fitting_branch(radargram, config.bg_window)
    zero = find_time_zero(
        radargram,
        branch.filtered,
        pick_mode=config.pick_mode,
        exclude_traces=exclude_traces,
        config=config.zero_config(),
    )
    return DepthInputs(branch, zero)


def depth_from_time(
    v_m_per_ns: float, t0_ns: float, *, air_gap: str, offset_m: float | None
) -> float:
    """Pipe-top depth from a two-way time after the zero (section 8).

    Air-coupled: ``v t0 / 2`` (zero = surface reflection). Ground-coupled:
    ``sqrt((v t0 / 2)^2 - (s / 2)^2)`` when ``s`` is known, else ``v t0 / 2``.
    A slant path shorter than ``s / 2`` gives 0.
    """
    half_path = v_m_per_ns * t0_ns / 2.0
    if air_gap == "air" or offset_m is None:
        return half_path
    return float(np.sqrt(max(half_path**2 - (offset_m / 2.0) ** 2, 0.0)))


def depth_uncertainty(
    depth_m: float,
    v_m_per_ns: float,
    sigma_v: float,
    sigma_zero_ns: float,
    sigma_apex_ns: float,
) -> float:
    """``sqrt((d sigma_v / v)^2 + (v / 2)^2 (sigma_z^2 + sigma_a^2))`` (section 9)."""
    velocity_term = depth_m * sigma_v / v_m_per_ns
    time_term = v_m_per_ns / 2.0 * float(np.hypot(sigma_zero_ns, sigma_apex_ns))
    return float(np.hypot(velocity_term, time_term))


def _dataset_flags(radargram: Radargram) -> list[str]:
    hypothesis = radargram.metadata.get("dt_hypothesis")
    return [] if hypothesis is None else [f"dt_hypothesis_{hypothesis}"]


def _saturation_flags(radargram: Radargram, picks: HyperbolaPicks) -> list[str]:
    """``saturated_apex`` for int16 full scale within +-2 samples (section 10)."""
    apex = picks.apex
    if apex is None:
        return []
    centre = round(apex.k)
    lo = max(0, centre - SATURATION_HALF_WINDOW_SAMPLES)
    hi = min(radargram.n_samples, centre + SATURATION_HALF_WINDOW_SAMPLES + 1)
    near_apex = np.abs(radargram.data[lo:hi, apex.trace])
    return ["saturated_apex"] if np.any(near_apex >= INT16_FULL_SCALE) else []


def _apex_time(
    choice: VelocityChoice,
    curvature: CurvatureFit | None,
    picks: HyperbolaPicks,
    config: DepthConfig,
) -> _ApexTime:
    """Curvature fit apex for ``curvature_R_fixed``, else the section 6-4 vertex."""
    if choice.v_rule == VelocityRule.CURVATURE_R_FIXED:
        assert curvature is not None and curvature.fit is not None
        fit = curvature.fit
        near_apex = np.abs(picks.x_m - fit.x0_m) <= config.apex_halfwidth_m + 1e-9
        return _ApexTime(fit.x0_m, fit.t0_ns, fit.rms_ns, max(1, int(near_apex.sum())))
    vertex = fixed_velocity_apex(picks, config)
    return _ApexTime(vertex.x0_m, vertex.t0_ns, vertex.rms_ns, vertex.n_points)


def _sigma_apex_ns(apex: _ApexTime, radargram: Radargram) -> float:
    """``max(0.05, rms / sqrt(n_apex))``, and ``>= dt / 2`` for narrow peaks."""
    sigma = max(SIGMA_APEX_FLOOR_NS, apex.rms_ns / np.sqrt(apex.n_points))
    dt_ns = radargram.dt_ns
    period_ns = period_ns_or_default(radargram.f_c_hz, DEFAULT_PERIOD_NS)
    if dt_ns is not None and period_ns / dt_ns <= NARROW_PEAK_SAMPLES:
        sigma = max(sigma, dt_ns / 2.0)
    return float(sigma)


def _alt_label(label: str | None) -> AltDepthLabel:
    return label if label in _ALT_LABELS else "unclassified"  # type: ignore[return-value]


def alternative_depths(
    radargram: Radargram,
    zero: ZeroResult,
    main_t0_ns: float,
    vertex_t0_ns: float,
    v_main: float | None,
    v_alt_meta: float | None,
) -> dict[str, AltDepth]:
    """``alt_depths`` rules (a) and (b) of spec section 2 (v1.4).

    (a) Metadata zeros: main apex time ``t_a`` minus the metadata zero, with
    the main v (meta v when there is none). (b) ``meta_fixed``: meta v, main
    zero and the section 6-4 vertex time (v fixed -> apex from 6-4).
    """
    assert zero.t_zero_ns is not None
    t_apex_ns = zero.t_zero_ns + main_t0_ns
    alternatives: dict[str, AltDepth] = {}
    v_for_meta_zeros = v_main if v_main is not None else v_alt_meta
    if v_for_meta_zeros is not None:
        for name, (t_meta_ns, label) in zero.meta_t0.items():
            depth = depth_from_time(
                v_for_meta_zeros,
                t_apex_ns - t_meta_ns,
                air_gap=radargram.air_gap,
                offset_m=radargram.antenna_offset_m,
            )
            alternatives[name] = AltDepth(t_meta_ns, label, depth)
    if v_alt_meta is not None:
        depth = depth_from_time(
            v_alt_meta,
            vertex_t0_ns,
            air_gap=radargram.air_gap,
            offset_m=radargram.antenna_offset_m,
        )
        label = _alt_label(zero.zero_event_label)
        alternatives[VelocityRule.META_FIXED.value] = AltDepth(
            zero.t_zero_ns, label, depth
        )
    return alternatives


@dataclass(frozen=True)
class _RefractionCheck:
    """Rule 2a outcome: eps_r at ``h_est`` and the interval over the h margin."""

    eps_at_h_est: float
    eps_range: tuple[float, float]
    flags: tuple[str, ...]


def _refraction_check(
    radargram: Radargram, zero: ZeroResult, picks: HyperbolaPicks, config: DepthConfig
) -> _RefractionCheck | None:
    """Rule 2a (v1.4): refraction QC at ``h_est`` and ``h_est - 0.06 m``.

    ``h_est`` is biased high by direct-wave interference and eps_r rises about
    8 % per cm of h, so the QC is an interval ``[eps(h_est - dh), eps(h_est)]``.
    ``meta_eps_suspect`` only when the metadata eps_r is below ``0.7 eps_lo`` or
    above ``1.3 eps_hi``. Returns ``None`` when the QC does not apply.
    """
    runs = (
        radargram.air_gap == "air"
        and zero.zero_rule == ZeroRule.AIR_SURFACE_LATE
        and zero.h_est_m is not None
        and picks.is_enough_for_curvature(config)
    )
    if not runs:
        return None
    vertex = fixed_velocity_apex(picks, config)

    def eps_at(h_m: float) -> float:
        return refraction_qc(
            picks.x_m,
            picks.tau_ns,
            x0_m=vertex.x0_m,
            tau_apex_ns=vertex.t0_ns,
            h_m=h_m,
            offset_m=radargram.antenna_offset_m or 0.0,
            radius_m=radargram.pipe_radius_m or 0.0,
            config=config,
        ).eps_fit

    eps_high = eps_at(zero.h_est_m)
    eps_low = eps_at(max(zero.h_est_m - config.refraction_h_margin_m, 0.0))
    eps_range = (min(eps_low, eps_high), max(eps_low, eps_high))
    flags: tuple[str, ...] = ()
    eps_meta = radargram.eps_r_meta
    if eps_meta is not None:
        lower_limit = (1.0 - config.meta_eps_suspect_rel) * eps_range[0]
        upper_limit = (1.0 + config.meta_eps_suspect_rel) * eps_range[1]
        if eps_meta < lower_limit or eps_meta > upper_limit:
            flags = ("meta_eps_suspect",)
    return _RefractionCheck(eps_high, eps_range, flags)


def _status_without_velocity(
    radargram: Radargram, curvature: CurvatureFit | None
) -> DepthStatus:
    if curvature is None:
        # Rule 2b (air without eps_r_meta) or rule 6 (unknown air gap).
        return DepthStatus.NO_VELOCITY
    if curvature.status == FitStatus.INSUFFICIENT_PICKS:
        return DepthStatus.INSUFFICIENT_PICKS
    return DepthStatus.FIT_FAILED


def _fit_statistics(curvature: CurvatureFit | None) -> dict:
    if curvature is None or curvature.fit is None:
        return {}
    return {
        "inlier_frac": curvature.fit.inlier_frac,
        "rms_ns": curvature.fit.rms_ns,
    }


def estimate_depth(
    radargram: Radargram,
    box: GridBox,
    config: DepthConfig = DEFAULT_DEPTH_CONFIG,
    *,
    inputs: DepthInputs | None = None,
) -> DepthResult:
    """Pipe-top depth of one ``pipe_hyperbola`` box (spec_depth_v1 sections 2-10).

    Args:
        radargram: The profile; physical scale decides whether depth is allowed.
        box: Detection box on the original grid (L0).
        config: All depth settings.
        inputs: Preprocessing and zero of this radargram from
            :func:`prepare_depth_inputs`; computed here when ``None``.

    Returns:
        One :class:`DepthResult`. Statuses: ``no_scale`` (apex pixel only),
        ``zero_failed``, ``fit_failed`` (no apex seed, box in the zero zone or a
        failed curvature fit), ``insufficient_picks``, ``no_velocity``
        (air-coupled without ``eps_r_meta``, or unknown air gap) and ``ok``.

    Raises:
        ValueError: If ``inputs.zero`` was picked with another pick mode.
    """
    if inputs is None:
        inputs = prepare_depth_inputs(radargram, config)
    zero = inputs.zero
    picks = extract_picks(
        inputs.branch.background_removed,
        box,
        zero,
        dt_ns=radargram.dt_ns,
        dx_m=radargram.dx_m,
        f_c_hz=radargram.f_c_hz,
        config=config,
    )
    flags = [
        *inputs.branch.flags,
        *zero.flags,
        *picks.flags,
        *_dataset_flags(radargram),
        *_saturation_flags(radargram, picks),
    ]
    common = {
        "apex_trace": None if picks.apex is None else picks.apex.trace,
        "apex_sample": None if picks.apex is None else picks.apex.k,
        "zero_rule": zero.zero_rule,
        "zero_event_label": zero.zero_event_label,
        "t_zero_ns": zero.t_zero_ns,
        "h_est_m": zero.h_est_m,
        "n_picks": picks.n_picks,
        "bias_notes": zero.bias_notes,
        "sigma_zero_ns": zero.sigma_zero_ns,
    }

    if not radargram.has_physical_scale:
        return DepthResult(DepthStatus.NO_SCALE, flags=_unique(flags), **common)
    if zero.failed:
        return DepthResult(DepthStatus.ZERO_FAILED, flags=_unique(flags), **common)
    if picks.apex is None:
        return DepthResult(DepthStatus.FIT_FAILED, flags=_unique(flags), **common)
    return _depth_with_apex(radargram, zero, picks, config, flags, common)


def _depth_with_apex(
    radargram: Radargram,
    zero: ZeroResult,
    picks: HyperbolaPicks,
    config: DepthConfig,
    flags: list[str],
    common: dict,
) -> DepthResult:
    """Sections 7-9 once scale, zero and apex seed exist."""
    radius_m = radargram.pipe_radius_m or 0.0
    curvature = None
    # Rules 2 and 6: no straight-ray curvature fit when the gap is air or unknown.
    if radargram.air_gap == "ground":
        curvature = fit_curvature(
            picks,
            radius_m=radius_m,
            air_gap=radargram.air_gap,
            v_fallback=meta_velocity(radargram),
            config=config,
        )
        flags.extend(curvature.flags)
    choice = select_hyperbola_velocity(radargram, curvature, config)
    flags.extend(choice.flags)
    refraction = _refraction_check(radargram, zero, picks, config)
    if refraction is not None:
        flags.extend(refraction.flags)

    apex = _apex_time(choice, curvature, picks, config)
    vertex = fixed_velocity_apex(picks, config)
    eps_qc, eps_qc_range = (
        (None, None)
        if refraction is None
        else (refraction.eps_at_h_est, refraction.eps_range)
    )
    common = {
        **common,
        "bias_notes": (*common["bias_notes"], *choice.bias_notes),
        "radius_used_m": radius_m,
        "x0_m": apex.x0_m,
        "t0_ns": apex.t0_ns,
        "t_apex_ns": zero.t_zero_ns + apex.t0_ns,
        "v_rule": choice.v_rule.value,
        "eps_r_refraction_qc": eps_qc,
        "eps_r_refraction_qc_range": eps_qc_range,
        "alt_depths": alternative_depths(
            radargram,
            zero,
            apex.t0_ns,
            vertex.t0_ns,
            choice.v_m_per_ns,
            choice.alt_meta_v,
        ),
        **_fit_statistics(curvature),
    }
    if not choice.has_velocity:
        status = _status_without_velocity(radargram, curvature)
        return DepthResult(status, flags=_unique(flags), **common)

    v = choice.v_m_per_ns
    depth_m = depth_from_time(
        v, apex.t0_ns, air_gap=radargram.air_gap, offset_m=radargram.antenna_offset_m
    )
    sigma_apex = _sigma_apex_ns(apex, radargram)
    if "rms_ns" not in common:
        common["rms_ns"] = apex.rms_ns
    return DepthResult(
        DepthStatus.OK,
        depth_top_m=depth_m,
        depth_center_m=(
            None if radargram.pipe_radius_m is None else depth_m + radius_m
        ),
        sigma_depth_m=depth_uncertainty(
            depth_m, v, choice.sigma_v, zero.sigma_zero_ns, sigma_apex
        ),
        v_m_per_ns=v,
        sigma_v=choice.sigma_v,
        eps_r_used=eps_from_velocity(v),
        sigma_apex_ns=sigma_apex,
        flags=_unique(flags),
        **common,
    )


def _unique(flags: list[str]) -> tuple[str, ...]:
    """Flags in first-seen order without repeats."""
    return tuple(dict.fromkeys(flags))
