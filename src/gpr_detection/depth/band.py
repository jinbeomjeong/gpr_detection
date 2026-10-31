"""Depth of ``pipe_band`` boxes (spec_band_depth_v1, checklist M2-6b).

A pipe parallel to the survey line gives on every trace the response of a
crossing pipe's apex trace, so the band top is the apex-time rule of
spec_depth_v1 applied trace by trace:

1. Zero: the radargram zero with the band traces left out of the mean trace
   (:func:`band_exclude_traces`, section 3-1); the caller passes it in.
2. Background: **masked median** over traces outside every band box and its
   margin (:func:`masked_background`); the plain moving median would erase a
   long band (section 3-2).
3. Picks: per trace, the first envelope peak above 0.3 x the box maximum in a
   window widened one period upwards (:func:`extract_band_picks`, 3-3). A
   second event within two periods sets ``band_top_multi_event`` (13 R2).
4. Fragments of one band are merged (:func:`group_band_boxes`, 3-4).
5. A flat / sloped / curved model gives the representative time
   (:func:`fit_band_profile`, 3-5); the depth uses a borrowed velocity
   (``velocity.select_band_velocity``, section 5).

Without a physical scale everything runs in pixels and no m or ns field is
filled (QA-4). Lateral-offset and oblique-neighbour biases are reported as
notes only, never corrected (user decision Q2); so are a possible protective
plate and a layer picked instead of the pipe top (band spec 13 R2-R3).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from gpr_detection.depth.config import DEFAULT_BAND_CONFIG, BandConfig
from gpr_detection.depth.estimate import (
    alternative_depths,
    depth_from_time,
    depth_uncertainty,
)
from gpr_detection.depth.picks import mute_sample
from gpr_detection.depth.result import (
    BandBackgroundMode,
    BandGeometry,
    BandModel,
    DepthResult,
    DepthStatus,
)
from gpr_detection.depth.velocity import (
    VelocityChoice,
    VelocityEstimate,
    meta_velocity,
    select_band_velocity,
)
from gpr_detection.io.radargram import GridBox, Radargram
from gpr_detection.preprocess.filters import (
    FittingBranch,
    period_ns_or_default,
    prepare_fitting_branch,
)
from gpr_detection.preprocess.signals import (
    parabolic_peak,
    pick_signal,
    require_same_pick_mode,
)
from gpr_detection.preprocess.timezero import DEFAULT_PERIOD_NS, ZeroResult

TARGET_CLASS = "pipe_band"

MIN_MARGIN_TRACES = 5
"""Lower bound of the masked-median margin ``m`` (also the no-scale value)."""

EDGE_PICK_COUNT = 5
"""Picks at each fragment end compared when merging (section 3-4 item 2)."""

OVERLAP_MIN_TRACES = 5
"""Shared traces, and picks per fragment on them, for the overlap comparison.

Section 3-4 item 2 (a), v1.4. Same value as ``EDGE_PICK_COUNT`` but a separate
rule, so the two can change independently.
"""

MERGE_TOLERANCE_PERIODS = 0.5
MERGE_TOLERANCE_SAMPLES = 3.0
"""Merge when the end-pick difference is within ``max(0.5 / f_c, 3 dt)``."""

WINDOW_EDGE_SAMPLES = 2
EDGE_FLAG_FRACTION = 0.3
"""``top_at_window_edge``: pick within 2 samples of the window top on > 30 %."""

MULTIPLE_REL = 0.5
MULTIPLE_RATIO_TOLERANCE = 0.1
INNER_REFLECTION_PERIODS = 2.0
"""Later-event flags (section 3-3 item 6), information only."""

MULTI_EVENT_PERIODS = 2.0
"""``band_top_multi_event`` search length after the first pick (band spec 13 R2)."""

# Band spec 13-2: the wording is fixed by the spec, character for character.
PROTECTIVE_PLATE_NOTE = (
    "protective_plate_possible: under a steel protective plate (city gas KGS "
    "FS451/FS551, KOGAS GSM-2121) the band is the plate, not the pipe; the pipe top "
    "is >= 0.3 m deeper (>= 0.1 m under the KGS 2022 exception); not corrected"
)
LAYER_AMBIGUITY_NOTE = (
    "band_top_layer_ambiguity: a layer within ~1.4 periods (short wavelet) or "
    "~2 periods (ringing wavelet) of the pipe top can be picked instead; "
    "bias -1.2..+0.5 periods (synthetic band_top_synthetic_v1); not corrected"
)

VERTEX_EDGE_TRACES = 5
"""A curved-model vertex must be at least 5 traces inside the pick range."""

MAD_TO_SIGMA = 1.4826
SIGMA_BAND_FLOOR_NS = 0.05

# No-scale inputs (band spec 3-5 v1.1): limits scale with the dominant period
# P_px of ``bandpass_px``; the fixed px values apply only when it was skipped.
F_SCALE_PERIODS_PX = 0.05
SLOPE_MIN_PERIODS_PX = 0.1
F_SCALE_PX = 1.0
RES_MAX_PX = 2.0
SLOPE_MIN_PX = 2.0
Q_RES_SCALE_PX = 1.0
"""Fixed px fallbacks (``bandpass_px_skipped``): f_scale, residual, slope, q_res."""

PAD_UP_FRACTION_NO_FC = 0.25
PAD_DOWN_FRACTION_NO_FC = 0.1
"""Window padding as a fraction of the box height when ``f_c`` is unknown."""

STEEP_GRADE = 0.2
ZERO_DRIFT_NS = 0.2
CONFIDENCE_COVERAGE = 0.6
CONFIDENCE_RES_PERIODS = 0.125
CONFIDENCE_SNR_LOW_DB = 6.0
CONFIDENCE_SNR_SPAN_DB = 14.0
CONFIDENCE_NO_BACKGROUND = 0.8

_HZ_PER_GHZ = 1e9


# ----------------------------------------------------------------- data classes


@dataclass(frozen=True)
class BandPicks:
    """First-peak picks of one (merged) band box (band spec 8-2).

    Attributes:
        box: Merged (union) box actually searched.
        member_indices: Input box indices in this band.
        trace_idx: Traces with a valid pick (int).
        k_pick: Fractional sample of the first envelope peak (L0).
        t_pick_ns: ``k_pick * dt_ns``; ``None`` without scale.
        amp: Envelope value at each pick.
        n_box_traces: Coverage denominator (member traces, gaps excluded).
        window: ``(k_lo, k_hi)`` search window in samples.
        bg_mode: Background removal used.
        noise_sigma: Robust noise level for the SNR term of the confidence.
        flags: ``top_at_window_edge``, ``surface_multiple_candidate`` ...
        n_member_picks: Coverage numerator: picks on member-box traces. Picks in
            the gap between merged fragments feed the fit only (spec 3-4 v1.3).
    """

    box: GridBox
    member_indices: tuple[int, ...]
    trace_idx: np.ndarray
    k_pick: np.ndarray
    t_pick_ns: np.ndarray | None
    amp: np.ndarray
    n_box_traces: int
    window: tuple[int, int]
    bg_mode: BandBackgroundMode
    noise_sigma: float | None
    flags: tuple[str, ...]
    n_member_picks: int = 0

    @property
    def n_picks(self) -> int:
        return len(self.trace_idx)

    @property
    def coverage(self) -> float:
        """Member-trace picks / member traces (at most 1)."""
        return self.n_member_picks / self.n_box_traces if self.n_box_traces else 0.0


@dataclass(frozen=True)
class BandFit:
    """Trace-direction model of the band-top times (band spec 3-5, 8-2).

    ``xi = x - x_c``; positions in m (traces without scale), times after the
    zero in ns (samples after the zero sample without scale).

    Attributes:
        model: Chosen model.
        params: flat ``(p0,)``, sloped ``(q0, q1)``, curved ``(p0, p1, p2)``.
        cov: Parameter covariance (inliers), ``None`` for flat.
        x_c: Centre of the pick range.
        ref_pos: Representative position.
        tau_ref: Representative time after the zero.
        sigma_tau_ref: Its uncertainty ``sigma_b`` (section 6-3).
        sigma_res: ``1.4826 MAD`` of the chosen model's residuals.
        slope: ``q1`` (ns/m) for ``sloped``.
        theta_est_deg: Crossing angle for ``curved`` with a known velocity.
        x_range: ``(x_min, x_max)`` of the picks.
        flags: ``crossing_outside_box``, ``band_concave``, ``band_oblique_large``,
            ``inconsistent_picks``.
    """

    model: BandModel
    params: tuple[float, ...]
    cov: np.ndarray | None
    x_c: float
    ref_pos: float
    tau_ref: float
    sigma_tau_ref: float
    sigma_res: float
    slope: float | None
    theta_est_deg: float | None
    x_range: tuple[float, float]
    flags: tuple[str, ...] = field(default=())

    def tau_at(self, x: float) -> float:
        """Model time at position ``x`` (flat: the representative time)."""
        if self.model == "flat":
            return self.tau_ref
        xi = x - self.x_c
        return float(sum(p * xi**power for power, p in enumerate(self.params)))


# ----------------------------------------------------------------- helpers


def _margin_traces(dx_m: float | None, cfg: BandConfig) -> int:
    """Masked-median margin ``m = max(5, ceil(0.1 m / dx))`` (5 without scale)."""
    if dx_m is None:
        return MIN_MARGIN_TRACES
    return max(MIN_MARGIN_TRACES, math.ceil(cfg.bg_margin_m / dx_m - 1e-9))


def band_exclude_traces(
    band_boxes: Sequence[GridBox],
    n_traces: int,
    dx_m: float | None,
    cfg: BandConfig = DEFAULT_BAND_CONFIG,
) -> np.ndarray:
    """Boolean mask of band-box traces (with margin) to leave out of the zero.

    Pass it as ``exclude_traces`` to ``find_time_zero`` /
    ``prepare_depth_inputs`` (section 3-1 item 2).
    """
    margin = _margin_traces(dx_m, cfg)
    excluded = np.zeros(n_traces, dtype=bool)
    for box in band_boxes:
        excluded[max(0, box.i0 - margin) : min(n_traces, box.i1 + margin + 1)] = True
    return excluded


def masked_background(
    a: np.ndarray,
    band_boxes: Sequence[GridBox],
    center_trace: int,
    dx_m: float | None,
    cfg: BandConfig = DEFAULT_BAND_CONFIG,
) -> tuple[np.ndarray | None, BandBackgroundMode]:
    """Median trace over traces outside every band box (section 3-2).

    The window is ``bg_window`` traces centred on ``center_trace`` (the whole
    line when shorter). Fewer than ``bg_min_outside`` outside traces return
    ``(None, "none")``: no background removal.
    """
    n_traces = a.shape[1]
    half = cfg.bg_window // 2
    if n_traces <= cfg.bg_window:
        window = np.arange(n_traces)
    else:
        window = np.arange(
            max(0, center_trace - half), min(n_traces, center_trace + half + 1)
        )
    excluded = band_exclude_traces(band_boxes, n_traces, dx_m, cfg)
    outside = window[~excluded[window]]
    if len(outside) < cfg.bg_min_outside:
        return None, "none"
    return np.median(a[:, outside], axis=1), "masked_median"


def _window_padding(
    box: GridBox, dt_ns: float | None, f_c_hz: float | None, cfg: BandConfig
) -> tuple[int, int]:
    """``(p_up, p_down)`` samples: ``pad_*_periods / f_c``, else box fractions."""
    if dt_ns is None or f_c_hz is None:
        height = box.n_samples
        return (
            max(2, math.ceil(PAD_UP_FRACTION_NO_FC * height)),
            max(1, math.ceil(PAD_DOWN_FRACTION_NO_FC * height)),
        )
    period_samples = _HZ_PER_GHZ / f_c_hz / dt_ns
    return (
        math.ceil(cfg.pad_up_periods * period_samples - 1e-9),
        math.ceil(cfg.pad_down_periods * period_samples - 1e-9),
    )


def _robust_sigma(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(MAD_TO_SIGMA * np.median(np.abs(values - np.median(values))))


def _noise_sigma(
    a_bg: np.ndarray, box: GridBox, k_lo: int, k_hi: int, k_mute: int | None, span: int
) -> float | None:
    """Noise above the window (after the mute), else the same length below it."""
    columns = a_bg[:, box.trace_slice]
    start = max(k_lo - span, k_mute or 0, 0)
    if start < k_lo:
        return _robust_sigma(columns[start:k_lo].ravel())
    below = columns[k_hi + 1 : k_hi + 1 + span]
    return _robust_sigma(below.ravel()) if below.size else None


def _later_event_flags(
    signal: np.ndarray,
    k_first: float,
    k_zero: float | None,
    e_box: float,
    inner_samples: float | None,
) -> list[str]:
    """Section 3-3 item 6 for one trace (information only)."""
    later, _ = find_peaks(signal[int(k_first) + 2 :], height=MULTIPLE_REL * e_box)
    if len(later) == 0:
        return []
    k_second = int(k_first) + 2 + int(later[0])
    flags = []
    if k_zero is not None and k_first > k_zero:
        ratio = (k_second - k_zero) / (k_first - k_zero)
        if abs(ratio - 2.0) < MULTIPLE_RATIO_TOLERANCE:
            flags.append("surface_multiple_candidate")
    if inner_samples is not None and k_second - k_first < inner_samples:
        flags.append("inner_reflection_candidate")
    return flags


def _multi_event_samples(
    dt_ns: float | None, f_c_hz: float | None, period_px: float | None
) -> float | None:
    """Search length of the multi-event check in samples; ``None`` skips it."""
    if dt_ns is not None and f_c_hz is not None:
        return MULTI_EVENT_PERIODS * _HZ_PER_GHZ / f_c_hz / dt_ns
    if period_px is not None:
        return MULTI_EVENT_PERIODS * period_px
    return None


def _has_later_event(
    signal: np.ndarray, k_first: float, threshold: float, span_samples: float
) -> bool:
    """Another peak ``>= threshold`` in ``(k_first + 2, k_first + span]`` (13 R2)."""
    start = int(k_first) + 2
    last = int(k_first + span_samples)
    if last <= start:
        return False
    # One sample past ``last`` so that a peak exactly at ``last`` is detectable;
    # find_peaks never reports the first or last element of its input.
    peaks, _ = find_peaks(signal[start : last + 2], height=threshold)
    return bool(np.any(peaks <= last - start))


# ----------------------------------------------------------------- picks


def extract_band_picks(
    a_bg: np.ndarray,
    box: GridBox,
    k_mute: int | None,
    dt_ns: float | None,
    f_c_hz: float | None,
    cfg: BandConfig = DEFAULT_BAND_CONFIG,
    *,
    k_zero: float | None = None,
    member_traces: np.ndarray | None = None,
    member_indices: tuple[int, ...] = (),
    bg_mode: BandBackgroundMode = "masked_median",
    period_px: float | None = None,
) -> BandPicks:
    """First-envelope-peak band-top picks (section 3-3).

    Args:
        a_bg: Steps (1)-(3), background removed, ``[n_samples, n_traces]``.
        box: (Merged) band box on the L0 grid.
        k_mute: First sample after ``t_z + 1 / f_c``; ``None`` = no mute.
        dt_ns: Sample interval (``None`` without scale).
        f_c_hz: Centre frequency (``None`` uses box-fraction padding).
        cfg: Thresholds and pick mode.
        k_zero: Zero sample, for the surface-multiple flag.
        member_traces: Traces that belong to member boxes (coverage
            denominator); default every box trace.
        member_indices: Input box indices merged into ``box``.
        bg_mode: Background mode recorded in the result.
        period_px: No-scale dominant period; sets the ``band_top_multi_event``
            search length when ``dt_ns`` or ``f_c_hz`` is missing (band spec
            13 R2). Without either the check is skipped.
    """
    p_up, p_down = _window_padding(box, dt_ns, f_c_hz, cfg)
    k_lo = max(box.k0 - p_up, k_mute or 0)
    k_hi = min(box.k1 + p_down, a_bg.shape[0] - 1)
    signal = pick_signal(a_bg[:, box.trace_slice], cfg.pick_mode)
    e_box = float(signal[box.sample_slice].max())
    threshold = cfg.candidate_rel * e_box
    inner_samples = None
    if dt_ns is not None and f_c_hz is not None:
        inner_samples = INNER_REFLECTION_PERIODS * _HZ_PER_GHZ / f_c_hz / dt_ns
    multi_event_samples = _multi_event_samples(dt_ns, f_c_hz, period_px)

    traces, k_picks, amplitudes, trace_flags = [], [], [], []
    edge_hits = 0
    for column, trace in enumerate(range(box.i0, box.i1 + 1)):
        window = signal[k_lo : k_hi + 1, column]
        if threshold <= 0 or len(window) < 3:
            continue
        peaks, _ = find_peaks(window, height=threshold)
        if len(peaks) == 0:
            continue
        k_local, amplitude = parabolic_peak(window, int(peaks[0]))
        k = k_lo + k_local
        edge_hits += int(peaks[0]) < WINDOW_EDGE_SAMPLES
        traces.append(trace)
        k_picks.append(k)
        amplitudes.append(amplitude)
        trace_flags.extend(
            _later_event_flags(signal[:, column], k, k_zero, e_box, inner_samples)
        )
        if multi_event_samples is not None and _has_later_event(
            signal[:, column], k, threshold, multi_event_samples
        ):
            trace_flags.append("band_top_multi_event")

    n_picks = len(traces)
    flags = []
    if n_picks and edge_hits / n_picks > EDGE_FLAG_FRACTION:
        flags.append("top_at_window_edge")
    for name in (
        "surface_multiple_candidate",
        "inner_reflection_candidate",
        "band_top_multi_event",
    ):
        if n_picks and trace_flags.count(name) / n_picks > EDGE_FLAG_FRACTION:
            flags.append(name)

    k_array = np.array(k_picks, dtype=float)
    if member_traces is None:
        n_box_traces, n_member_picks = box.n_traces, n_picks
    else:
        n_box_traces = len(member_traces)
        n_member_picks = int(np.isin(traces, member_traces).sum())
    return BandPicks(
        box=box,
        member_indices=member_indices,
        trace_idx=np.array(traces, dtype=int),
        k_pick=k_array,
        t_pick_ns=None if dt_ns is None else k_array * dt_ns,
        amp=np.array(amplitudes, dtype=float),
        n_box_traces=n_box_traces,
        window=(k_lo, k_hi),
        bg_mode=bg_mode,
        noise_sigma=_noise_sigma(a_bg, box, k_lo, k_hi, k_mute, p_up + p_down),
        flags=tuple(flags),
        n_member_picks=n_member_picks,
    )


# ----------------------------------------------------------------- profile fit


@dataclass(frozen=True)
class _Units:
    """Positions and times of the fit, with their unit-dependent limits."""

    x: np.ndarray
    tau: np.ndarray
    f_scale: float
    res_max: float
    slope_min: float
    vertex_margin: float
    sigma_floor: float
    footprint_traces: float


@dataclass(frozen=True)
class PixelLimits:
    """No-scale limits in px (band spec 3-5 and 6-4, v1.1).

    Attributes:
        f_scale: soft-L1 scale, ``0.05 P`` (1 px).
        res_max: ``inconsistent_picks`` limit, ``0.25 P`` (2 px).
        slope_min: Minimum change for ``sloped``, ``0.1 P`` (2 px).
        merge_tolerance: Fragment end difference, ``max(0.5 P, 3)`` (3 px).
        q_res_scale: Residual scale of the confidence, ``0.125 P`` (1 px).
    """

    f_scale: float
    res_max: float
    slope_min: float
    merge_tolerance: float
    q_res_scale: float


def pixel_limits(
    period_px: float | None, cfg: BandConfig = DEFAULT_BAND_CONFIG
) -> PixelLimits:
    """Limits tied to the dominant period ``P`` (fixed px values when ``None``)."""
    if period_px is None:
        return PixelLimits(
            f_scale=F_SCALE_PX,
            res_max=RES_MAX_PX,
            slope_min=SLOPE_MIN_PX,
            merge_tolerance=MERGE_TOLERANCE_SAMPLES,
            q_res_scale=Q_RES_SCALE_PX,
        )
    return PixelLimits(
        f_scale=F_SCALE_PERIODS_PX * period_px,
        res_max=cfg.res_max_periods * period_px,
        slope_min=SLOPE_MIN_PERIODS_PX * period_px,
        merge_tolerance=max(
            MERGE_TOLERANCE_PERIODS * period_px, MERGE_TOLERANCE_SAMPLES
        ),
        q_res_scale=CONFIDENCE_RES_PERIODS * period_px,
    )


def _fit_units(
    picks: BandPicks,
    dx_m: float | None,
    dt_ns: float | None,
    f_c_hz: float | None,
    k_zero: float | None,
    cfg: BandConfig,
    period_px: float | None = None,
) -> _Units:
    k_ref = 0.0 if k_zero is None else k_zero
    if dx_m is None or dt_ns is None:
        limits = pixel_limits(period_px, cfg)
        return _Units(
            x=picks.trace_idx.astype(float),
            tau=picks.k_pick - k_ref,
            f_scale=limits.f_scale,
            res_max=limits.res_max,
            slope_min=limits.slope_min,
            vertex_margin=float(VERTEX_EDGE_TRACES),
            sigma_floor=0.0,
            footprint_traces=1.0,
        )
    period_ns = period_ns_or_default(f_c_hz, DEFAULT_PERIOD_NS)
    return _Units(
        x=picks.trace_idx * dx_m,
        tau=(picks.k_pick - k_ref) * dt_ns,
        f_scale=cfg.f_scale_ns,
        res_max=cfg.res_max_periods * period_ns,
        slope_min=cfg.slope_min_ns,
        vertex_margin=VERTEX_EDGE_TRACES * dx_m,
        sigma_floor=SIGMA_BAND_FLOOR_NS,
        footprint_traces=cfg.footprint_corr_m / dx_m,
    )


@dataclass(frozen=True)
class _PolyFit:
    params: np.ndarray
    cov: np.ndarray
    residuals: np.ndarray

    def sigma(self, index: int) -> float:
        return float(np.sqrt(max(self.cov[index, index], 0.0)))


def _robust_polynomial(
    xi: np.ndarray, tau: np.ndarray, degree: int, f_scale: float
) -> _PolyFit:
    """soft-L1 polynomial fit; covariance ``s^2 (J^T J)^+`` over inliers."""
    design = np.vander(xi, degree + 1, increasing=True)
    start = np.linalg.lstsq(design, tau, rcond=None)[0]
    solution = least_squares(
        lambda p: design @ p - tau,
        start,
        jac=lambda p: design,
        loss="soft_l1",
        f_scale=f_scale,
    )
    residuals = design @ solution.x - tau
    inliers = np.abs(residuals) < 3.0 * f_scale
    if inliers.sum() <= degree + 1:
        inliers = np.ones_like(inliers)
    n_params = degree + 1
    used_design, used_res = design[inliers], residuals[inliers]
    dof = max(len(used_res) - n_params, 1)
    variance = float(used_res @ used_res) / dof
    cov = variance * np.linalg.pinv(used_design.T @ used_design)
    return _PolyFit(solution.x, cov, residuals)


def _theta_deg(
    p2: float, tau_ref: float, v: float | None, radius_m: float
) -> float | None:
    """``theta = arcsin(v sqrt(p2 tau_c / 2))``, ``tau_c = tau_ref + 2R/v``."""
    if v is None:
        return None
    tau_c = tau_ref + 2.0 * radius_m / v
    argument = v * math.sqrt(max(p2 * tau_c / 2.0, 0.0))
    return math.degrees(math.asin(min(argument, 1.0)))


def _vertex_sigma(fit: _PolyFit, floor: float) -> float:
    _, p1, p2 = fit.params
    gradient = np.array([1.0, -p1 / (2.0 * p2), p1**2 / (4.0 * p2**2)])
    return max(floor, float(np.sqrt(max(gradient @ fit.cov @ gradient, 0.0))))


def _flat_or_sloped(
    units: _Units, xi: np.ndarray, x_c: float, force_sloped: bool, cfg: BandConfig
) -> tuple[
    BandModel,
    tuple[float, ...],
    np.ndarray | None,
    float,
    float,
    float | None,
    np.ndarray,
]:
    """Sloped when its slope is significant (or forced), else flat."""
    sloped = _robust_polynomial(xi, units.tau, 1, units.f_scale)
    q0, q1 = (float(p) for p in sloped.params)
    length = float(units.x.max() - units.x.min())
    significant = abs(q1) * length >= units.slope_min and abs(
        q1
    ) >= cfg.signif_sigma * sloped.sigma(1)
    if significant or force_sloped:
        return "sloped", (q0, q1), sloped.cov, q0, x_c, q1, sloped.residuals
    tau_median = float(np.median(units.tau))
    return (
        "flat",
        (tau_median,),
        None,
        tau_median,
        float(np.median(units.x)),
        None,
        units.tau - tau_median,
    )


def fit_band_profile(
    picks: BandPicks,
    dx_m: float | None,
    v_m_per_ns: float | None,
    radius_m: float,
    cfg: BandConfig = DEFAULT_BAND_CONFIG,
    *,
    dt_ns: float | None = None,
    f_c_hz: float | None = None,
    k_zero: float | None = None,
    period_px: float | None = None,
) -> BandFit:
    """Choose curved / sloped / flat and the representative time (section 3-5).

    Args:
        picks: Band picks (at least two).
        dx_m: Trace interval (``None``: positions in traces).
        v_m_per_ns: Velocity for ``theta_est`` (``None``: no angle).
        radius_m: Pipe radius for ``tau_c`` (0 when unknown).
        cfg: Significance and residual limits.
        dt_ns: Sample interval (``None``: times in samples).
        f_c_hz: Centre frequency for the residual limit.
        k_zero: Zero sample; times are measured from it.
        period_px: No-scale dominant period (``FittingBranch.period_px``);
            the px limits scale with it (:func:`pixel_limits`).
    """
    units = _fit_units(picks, dx_m, dt_ns, f_c_hz, k_zero, cfg, period_px)
    x_min, x_max = float(units.x.min()), float(units.x.max())
    x_c = 0.5 * (x_min + x_max)
    xi = units.x - x_c
    flags: list[str] = []

    model: BandModel
    curved = (
        _robust_polynomial(xi, units.tau, 2, units.f_scale) if len(xi) >= 4 else None
    )
    force_sloped = False
    if curved is not None:
        p0, p1, p2 = (float(p) for p in curved.params)
        significant = abs(p2) >= cfg.signif_sigma * curved.sigma(2)
        if significant and p2 > 0:
            xi_vertex = -p1 / (2.0 * p2)
            inside = (
                x_min + units.vertex_margin
                <= x_c + xi_vertex
                <= x_max - units.vertex_margin
            )
            if inside:
                tau_ref = p0 - p1**2 / (4.0 * p2)
                theta = _theta_deg(p2, tau_ref, v_m_per_ns if dt_ns else None, radius_m)
                if theta is not None and theta > cfg.oblique_flag_deg:
                    flags.append("band_oblique_large")
                return _finish_fit(
                    "curved",
                    (p0, p1, p2),
                    curved.cov,
                    x_c,
                    x_c + xi_vertex,
                    tau_ref,
                    _vertex_sigma(curved, units.sigma_floor),
                    curved.residuals,
                    None,
                    theta,
                    (x_min, x_max),
                    flags,
                    units,
                )
            flags.append("crossing_outside_box")
            force_sloped = True
        elif significant and p2 < 0:
            flags.append("band_concave")

    model, params, cov, tau_ref, ref_pos, slope, residuals = _flat_or_sloped(
        units, xi, x_c, force_sloped, cfg
    )
    sigma_res = _robust_sigma(residuals)
    n_eff = max(1.0, len(xi) / units.footprint_traces)
    sigma_b = max(units.sigma_floor, sigma_res / math.sqrt(n_eff))
    return _finish_fit(
        model,
        params,
        cov,
        x_c,
        ref_pos,
        tau_ref,
        sigma_b,
        residuals,
        slope,
        None,
        (x_min, x_max),
        flags,
        units,
    )


def _finish_fit(
    model: BandModel,
    params: tuple[float, ...],
    cov: np.ndarray | None,
    x_c: float,
    ref_pos: float,
    tau_ref: float,
    sigma_tau_ref: float,
    residuals: np.ndarray,
    slope: float | None,
    theta: float | None,
    x_range: tuple[float, float],
    flags: list[str],
    units: _Units,
) -> BandFit:
    sigma_res = _robust_sigma(residuals)
    if sigma_res > units.res_max:
        flags.append("inconsistent_picks")
    return BandFit(
        model=model,
        params=tuple(float(p) for p in params),
        cov=cov,
        x_c=x_c,
        ref_pos=ref_pos,
        tau_ref=float(tau_ref),
        sigma_tau_ref=float(sigma_tau_ref),
        sigma_res=sigma_res,
        slope=None if slope is None else float(slope),
        theta_est_deg=theta,
        x_range=x_range,
        flags=tuple(flags),
    )


# ----------------------------------------------------------------- fragments


def _merge_gap_traces(dx_m: float | None, cfg: BandConfig) -> int:
    if dx_m is None:
        return cfg.merge_gap_px
    return math.ceil(cfg.merge_gap_m / dx_m - 1e-9)


def _are_candidates(a: GridBox, b: GridBox, gap_limit: int, cfg: BandConfig) -> bool:
    """Time overlap >= 50 % of the smaller height and trace gap <= limit."""
    left, right = (a, b) if a.i0 <= b.i0 else (b, a)
    gap = right.i0 - left.i1 - 1
    overlap = min(a.k1, b.k1) - max(a.k0, b.k0) + 1
    smaller = min(a.n_samples, b.n_samples)
    return gap <= gap_limit and overlap >= cfg.merge_overlap_min * smaller


def _edge_difference(
    a_index: int,
    b_index: int,
    boxes: Sequence[GridBox],
    edge_samples: Sequence[tuple[float, float]],
) -> float:
    """Sample difference of the facing fragment ends (``right(A) - left(B)``)."""
    left, right = (
        (a_index, b_index)
        if boxes[a_index].i0 <= boxes[b_index].i0
        else (b_index, a_index)
    )
    return float(edge_samples[left][1] - edge_samples[right][0])


def _left_right(
    a_index: int, b_index: int, boxes: Sequence[GridBox]
) -> tuple[int, int]:
    """``(left, right)``: smaller ``i0`` first, the smaller index on a tie."""
    left, right = sorted((a_index, b_index), key=lambda index: (boxes[index].i0, index))
    return left, right


def _median_pick_on(picks: BandPicks, first: int, last: int) -> float | None:
    """Median pick on traces ``first..last``; ``None`` with too few picks there."""
    on_shared = (picks.trace_idx >= first) & (picks.trace_idx <= last)
    if on_shared.sum() < OVERLAP_MIN_TRACES:
        return None
    return float(np.median(picks.k_pick[on_shared]))


def pair_pick_difference(
    a_index: int,
    b_index: int,
    boxes: Sequence[GridBox],
    fragment_picks: Sequence[BandPicks],
) -> float:
    """Pick difference of two fragments in samples, ``left - right`` (3-4 item 2).

    Left is the box with the smaller ``i0`` (the smaller index on a tie).

    (a) When the boxes share at least ``OVERLAP_MIN_TRACES`` traces and each
    fragment has that many picks on them, the medians over the shared traces
    are compared: the facing ends of overlapping boxes lie on different traces,
    so a sloped band would differ there although it is one event (UB17).
    (b) Otherwise the facing ends are compared: median of the right-end picks of
    left minus median of the left-end picks of right.

    Args:
        a_index: First box index.
        b_index: Second box index.
        boxes: All band boxes of the radargram.
        fragment_picks: Picks of every box searched alone (same order as boxes).

    Returns:
        The difference in samples; NaN when a fragment has no picks.
    """
    left, right = _left_right(a_index, b_index, boxes)
    first_shared = max(boxes[left].i0, boxes[right].i0)
    last_shared = min(boxes[left].i1, boxes[right].i1)
    if last_shared - first_shared + 1 >= OVERLAP_MIN_TRACES:
        left_median = _median_pick_on(fragment_picks[left], first_shared, last_shared)
        right_median = _median_pick_on(fragment_picks[right], first_shared, last_shared)
        if left_median is not None and right_median is not None:
            return left_median - right_median
    left_end = _edge_picks(fragment_picks[left])[1]
    right_start = _edge_picks(fragment_picks[right])[0]
    return left_end - right_start


def _pair_difference(
    a_index: int,
    b_index: int,
    boxes: Sequence[GridBox],
    edge_samples: Sequence[tuple[float, float]] | None,
    fragment_picks: Sequence[BandPicks] | None,
) -> float | None:
    """Confirmation difference of a candidate pair; ``None`` skips the check."""
    if fragment_picks is not None:
        return pair_pick_difference(a_index, b_index, boxes, fragment_picks)
    if edge_samples is not None:
        return _edge_difference(a_index, b_index, boxes, edge_samples)
    return None


def candidate_pairs(
    boxes: Sequence[GridBox], dx_m: float | None, cfg: BandConfig = DEFAULT_BAND_CONFIG
) -> list[tuple[int, int]]:
    """Index pairs that pass the box-geometry test of section 3-4 item 1."""
    gap_limit = _merge_gap_traces(dx_m, cfg)
    return [
        (i, j)
        for i in range(len(boxes))
        for j in range(i + 1, len(boxes))
        if _are_candidates(boxes[i], boxes[j], gap_limit, cfg)
    ]


def group_band_boxes(
    boxes: Sequence[GridBox],
    dx_m: float | None,
    cfg: BandConfig = DEFAULT_BAND_CONFIG,
    *,
    edge_samples: Sequence[tuple[float, float]] | None = None,
    tolerance_samples: float = MERGE_TOLERANCE_SAMPLES,
    fragment_picks: Sequence[BandPicks] | None = None,
) -> list[tuple[int, ...]]:
    """Merge fragments of one band, transitively (section 3-4).

    Candidates are confirmed only when their pick difference is at most
    ``tolerance_samples``. ``fragment_picks`` takes precedence over
    ``edge_samples``; with neither, only the box geometry is checked.

    Args:
        boxes: All ``pipe_band`` boxes of one radargram.
        dx_m: Trace interval (gap limit 0.2 m), or ``None``
            (``cfg.merge_gap_px``).
        cfg: Gap and overlap settings.
        edge_samples: Per box ``(left end, right end)`` median pick sample of
            the 5 end picks; the facing ends are compared (kept for backward
            compatibility).
        tolerance_samples: ``max(0.5 / f_c, 3 dt)`` in samples (3 px).
        fragment_picks: Picks of every box searched alone (v1.4); compared by
            :func:`pair_pick_difference`.

    Returns:
        Groups of box indices (sorted), every box exactly once.
    """
    parent = list(range(len(boxes)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, j in candidate_pairs(boxes, dx_m, cfg):
        difference = _pair_difference(i, j, boxes, edge_samples, fragment_picks)
        # A NaN difference (fragment without picks) still merges: band spec
        # section 11 keeps this open in v1.4.
        if difference is not None and abs(difference) > tolerance_samples:
            continue
        parent[root(j)] = root(i)

    groups: dict[int, list[int]] = {}
    for index in range(len(boxes)):
        groups.setdefault(root(index), []).append(index)
    return sorted(tuple(sorted(group)) for group in groups.values())


def _union_box(boxes: Sequence[GridBox]) -> GridBox:
    return GridBox(
        i0=min(b.i0 for b in boxes),
        i1=max(b.i1 for b in boxes),
        k0=min(b.k0 for b in boxes),
        k1=max(b.k1 for b in boxes),
    )


def _member_traces(boxes: Sequence[GridBox]) -> np.ndarray:
    traces: set[int] = set()
    for box in boxes:
        traces.update(range(box.i0, box.i1 + 1))
    return np.array(sorted(traces), dtype=int)


def _edge_picks(picks: BandPicks) -> tuple[float, float]:
    if picks.n_picks == 0:
        return float("nan"), float("nan")
    return (
        float(np.median(picks.k_pick[:EDGE_PICK_COUNT])),
        float(np.median(picks.k_pick[-EDGE_PICK_COUNT:])),
    )


# ----------------------------------------------------------------- estimate


@dataclass(frozen=True)
class _BandContext:
    """Per-radargram values shared by every band group."""

    radargram: Radargram
    zero: ZeroResult
    section: np.ndarray
    k_mute: int | None
    k_zero: float | None
    period_ns: float | None
    period_px: float | None
    upstream: bool
    flags: tuple[str, ...]


def _background_removed(
    context: _BandContext, boxes: Sequence[GridBox], box: GridBox, cfg: BandConfig
) -> tuple[np.ndarray, BandBackgroundMode]:
    if context.upstream:
        return context.section, "upstream"
    center = (box.i0 + box.i1) // 2
    background, mode = masked_background(
        context.section, boxes, center, context.radargram.dx_m, cfg
    )
    if background is None:
        return context.section, mode
    return context.section - background[:, None], mode


def _picks_for(
    context: _BandContext,
    boxes: Sequence[GridBox],
    members: tuple[int, ...],
    cfg: BandConfig,
) -> BandPicks:
    member_boxes = [boxes[i] for i in members]
    union = _union_box(member_boxes)
    a_bg, mode = _background_removed(context, boxes, union, cfg)
    return extract_band_picks(
        a_bg,
        union,
        context.k_mute,
        context.radargram.dt_ns,
        context.radargram.f_c_hz,
        cfg,
        k_zero=context.k_zero,
        member_traces=_member_traces(member_boxes),
        member_indices=members,
        bg_mode=mode,
        period_px=context.period_px,
    )


def _in_zero_zone(box: GridBox, context: _BandContext) -> bool:
    """Section 3-7: box above the mute, or less than one period left below it."""
    if context.k_mute is None:
        return False
    if box.k1 < context.k_mute:
        return True
    if box.k0 >= context.k_mute or context.period_ns is None:
        return False
    dt_ns = context.radargram.dt_ns
    assert dt_ns is not None
    return (box.k1 - context.k_mute + 1) < context.period_ns / dt_ns


def _confidence(
    picks: BandPicks, fit: BandFit, period_ns: float | None, period_px: float | None
) -> float:
    """``q = q_cov q_res q_snr`` (section 6-4; all coefficients assumed)."""
    q_cov = min(1.0, picks.coverage / CONFIDENCE_COVERAGE)
    if period_ns is not None:
        res_scale = CONFIDENCE_RES_PERIODS * period_ns
    else:
        res_scale = pixel_limits(period_px).q_res_scale
    q_res = math.exp(-((fit.sigma_res / res_scale) ** 2))
    q_snr = 0.0
    if picks.noise_sigma and picks.noise_sigma > 0 and picks.n_picks:
        snr_db = 20.0 * math.log10(float(np.median(picks.amp)) / picks.noise_sigma)
        q_snr = min(
            1.0, max(0.0, (snr_db - CONFIDENCE_SNR_LOW_DB) / CONFIDENCE_SNR_SPAN_DB)
        )
    quality = q_cov * q_res * q_snr
    if picks.bg_mode != "masked_median":
        quality *= CONFIDENCE_NO_BACKGROUND
    return float(quality)


def _lateral_bias_m(
    depth_m: float, v: float, eps_r: float, f_c_hz: float | None, radius_m: float
) -> float | None:
    """``sqrt(d_c^2 + y_max^2) - d_c``, ``y_max = lambda/4 + d / sqrt(eps + 1)``."""
    if f_c_hz is None:
        return None
    wavelength_m = v * _HZ_PER_GHZ / f_c_hz
    y_max = wavelength_m / 4.0 + depth_m / math.sqrt(eps_r + 1.0)
    centre = depth_m + radius_m
    return math.hypot(centre, y_max) - centre


def _edge_flags(box: GridBox, n_traces: int) -> list[str]:
    flags = []
    if box.i0 == 0:
        flags.append("touches_profile_edge_left")
    if box.i1 == n_traces - 1:
        flags.append("touches_profile_edge_right")
    return flags


@dataclass(frozen=True)
class _GroupResult:
    """Pieces assembled into one ``DepthResult``."""

    picks: BandPicks
    fit: BandFit | None
    status: DepthStatus


def estimate_band_depths(
    radargram: Radargram,
    band_boxes: Sequence[GridBox],
    zero: ZeroResult,
    pool: Sequence[VelocityEstimate] = (),
    survey_pool: Sequence[VelocityEstimate] = (),
    cfg: BandConfig = DEFAULT_BAND_CONFIG,
    *,
    branch: FittingBranch | None = None,
) -> list[DepthResult]:
    """Depth of every (merged) ``pipe_band`` box of one radargram.

    Args:
        radargram: The profile.
        band_boxes: All ``pipe_band`` boxes (L0).
        zero: Zero found with the band traces excluded (:func:`band_exclude_traces`).
        pool: Velocities of this radargram's hyperbolas (section 5).
        survey_pool: Velocities of the survey unit (rule 4).
        cfg: Band settings.
        branch: Steps (1)-(3) of this radargram, reused when given.

    Returns:
        One result per merged group, in group order.

    Raises:
        ValueError: If ``zero`` was picked with another pick mode, or ``branch``
            does not have the radargram's shape (it belongs to another profile).
    """
    require_same_pick_mode(zero.pick_mode, cfg.pick_mode)
    if branch is not None and branch.filtered.shape != radargram.data.shape:
        raise ValueError(
            f"branch shape {branch.filtered.shape} != radargram shape "
            f"{radargram.data.shape}"
        )
    if not band_boxes:
        return []
    if branch is None:
        branch = prepare_fitting_branch(radargram, cfg.bg_window)
    context = _context(radargram, zero, branch)
    # One box needs no fragment picks: it is its own group and has no pairs.
    fragment_picks = (
        _fragment_picks(context, band_boxes, cfg) if len(band_boxes) > 1 else None
    )
    groups = _merged_groups(context, band_boxes, cfg, fragment_picks)
    neighbors = _unmerged_neighbors(context, band_boxes, groups, cfg, fragment_picks)
    results = []
    for members in groups:
        group = _evaluate_group(context, band_boxes, members, cfg)
        results.append(
            _assemble(
                context, group, pool, survey_pool, cfg, neighbors.get(members, ())
            )
        )
    return results


def _context(
    radargram: Radargram, zero: ZeroResult, branch: FittingBranch
) -> _BandContext:
    scaled = radargram.has_physical_scale
    period_ns = None
    if scaled:
        period_ns = period_ns_or_default(radargram.f_c_hz, DEFAULT_PERIOD_NS)
    k_mute = None
    if scaled and zero.t_zero_ns is not None:
        assert radargram.dt_ns is not None and period_ns is not None
        k_mute = mute_sample(zero.t_zero_ns, period_ns, radargram.dt_ns)
    hypothesis = radargram.metadata.get("dt_hypothesis")
    flags = [*branch.flags, *zero.flags]
    if hypothesis is not None:
        flags.append(f"dt_hypothesis_{hypothesis}")
    upstream = radargram.amplitude_processed
    if upstream:
        flags.append("band_bg_removed_upstream")
    return _BandContext(
        radargram=radargram,
        zero=zero,
        section=branch.filtered,
        k_mute=k_mute,
        k_zero=zero.k_zero,
        period_ns=period_ns,
        period_px=None if scaled else branch.period_px,
        upstream=upstream,
        flags=tuple(flags),
    )


def _merge_tolerance(context: _BandContext) -> float:
    dt_ns = context.radargram.dt_ns
    if dt_ns is None or context.period_ns is None:
        return pixel_limits(context.period_px).merge_tolerance
    return max(
        MERGE_TOLERANCE_PERIODS * context.period_ns / dt_ns, MERGE_TOLERANCE_SAMPLES
    )


def _fragment_picks(
    context: _BandContext, boxes: Sequence[GridBox], cfg: BandConfig
) -> list[BandPicks]:
    """Picks of every box searched alone (section 3-4 item 2)."""
    return [_picks_for(context, boxes, (i,), cfg) for i in range(len(boxes))]


def _merged_groups(
    context: _BandContext,
    boxes: Sequence[GridBox],
    cfg: BandConfig,
    fragment_picks: Sequence[BandPicks] | None = None,
) -> list[tuple[int, ...]]:
    """Groups of ``boxes``; ``fragment_picks`` is computed here when ``None``."""
    if len(boxes) == 1:
        return [(0,)]
    if fragment_picks is None:
        fragment_picks = _fragment_picks(context, boxes, cfg)
    return group_band_boxes(
        boxes,
        context.radargram.dx_m,
        cfg,
        tolerance_samples=_merge_tolerance(context),
        fragment_picks=fragment_picks,
    )


def _unmerged_neighbors(
    context: _BandContext,
    boxes: Sequence[GridBox],
    groups: Sequence[tuple[int, ...]],
    cfg: BandConfig,
    fragment_picks: Sequence[BandPicks] | None = None,
) -> dict[tuple[int, ...], tuple[tuple[int, float], ...]]:
    """Candidate fragments left in different groups (section 3-4 item 4)."""
    pairs = candidate_pairs(boxes, context.radargram.dx_m, cfg)
    if not pairs:
        return {}
    group_of = {index: group for group in groups for index in group}
    if fragment_picks is None:
        fragment_picks = _fragment_picks(context, boxes, cfg)
    scale = context.radargram.dt_ns or 1.0
    neighbors: dict[tuple[int, ...], list[tuple[int, float]]] = {}
    for i, j in pairs:
        if group_of[i] == group_of[j]:
            continue
        delta = pair_pick_difference(i, j, boxes, fragment_picks) * scale
        neighbors.setdefault(group_of[i], []).append((j, delta))
        neighbors.setdefault(group_of[j], []).append((i, -delta))
    return {group: tuple(items) for group, items in neighbors.items()}


def _evaluate_group(
    context: _BandContext,
    boxes: Sequence[GridBox],
    members: tuple[int, ...],
    cfg: BandConfig,
) -> _GroupResult:
    picks = _picks_for(context, boxes, members, cfg)
    if _in_zero_zone(picks.box, context):
        return _GroupResult(picks, None, DepthStatus.BAND_IN_ZERO_ZONE)
    if picks.coverage < cfg.coverage_min or picks.n_picks < cfg.n_picks_min:
        return _GroupResult(picks, None, DepthStatus.INSUFFICIENT_PICKS)
    radargram = context.radargram
    fit = fit_band_profile(
        picks,
        radargram.dx_m if radargram.has_physical_scale else None,
        None,
        radargram.pipe_radius_m or 0.0,
        cfg,
        dt_ns=radargram.dt_ns if radargram.has_physical_scale else None,
        f_c_hz=radargram.f_c_hz,
        k_zero=context.k_zero,
        period_px=context.period_px,
    )
    if "inconsistent_picks" in fit.flags:
        return _GroupResult(picks, fit, DepthStatus.INCONSISTENT_PICKS)
    return _GroupResult(picks, fit, DepthStatus.OK)


def _ref_sample(fit: BandFit, context: _BandContext, scaled: bool) -> float:
    """Representative sample on the L0 grid from the representative time."""
    k_ref = 0.0 if context.k_zero is None else context.k_zero
    if not scaled:
        return fit.tau_ref + k_ref
    dt_ns = context.radargram.dt_ns
    assert dt_ns is not None
    return fit.tau_ref / dt_ns + k_ref


def _ref_trace(fit: BandFit, context: _BandContext, scaled: bool) -> float:
    if not scaled:
        return fit.ref_pos
    dx_m = context.radargram.dx_m
    assert dx_m is not None
    return fit.ref_pos / dx_m


def _geometry(
    context: _BandContext,
    group: _GroupResult,
    neighbors: tuple[tuple[int, float], ...],
    *,
    v: float | None = None,
    eps_r: float | None = None,
    depth_m: float | None = None,
) -> BandGeometry | None:
    picks, fit = group.picks, group.fit
    if fit is None:
        return None
    radargram = context.radargram
    scaled = radargram.has_physical_scale
    radius_m = radargram.pipe_radius_m or 0.0
    theta = None
    grade = profile = lateral = None
    if scaled and v is not None:
        if fit.model == "curved":
            theta = _theta_deg(fit.params[2], fit.tau_ref, v, radius_m)
        if fit.model == "sloped" and fit.slope is not None:
            grade = v / 2.0 * fit.slope
            profile = _depth_profile(fit, v, radargram)
        if depth_m is not None and eps_r is not None:
            lateral = _lateral_bias_m(depth_m, v, eps_r, radargram.f_c_hz, radius_m)
    return BandGeometry(
        model=fit.model,
        ref_trace=_ref_trace(fit, context, scaled),
        ref_sample=_ref_sample(fit, context, scaled),
        x_ref_m=fit.ref_pos if scaled else None,
        t_band_ns=(
            fit.tau_ref + context.zero.t_zero_ns
            if scaled and context.zero.t_zero_ns is not None
            else None
        ),
        trace_range=(picks.box.i0, picks.box.i1),
        member_indices=picks.member_indices,
        coverage=picks.coverage,
        n_picks=picks.n_picks,
        sigma_res=fit.sigma_res,
        slope_ns_per_m=fit.slope if scaled else None,
        grade=grade,
        theta_est_deg=theta,
        depth_profile=profile,
        bg_mode=picks.bg_mode,
        lateral_bias_max_m=lateral,
        confidence=_confidence(picks, fit, context.period_ns, context.period_px),
        fragment_neighbors=neighbors,
    )


def _depth_profile(
    fit: BandFit, v: float, radargram: Radargram
) -> tuple[tuple[float, float], ...]:
    x_min, x_max = fit.x_range
    points = []
    for x in (x_min, fit.x_c, x_max):
        depth = depth_from_time(
            v,
            fit.tau_at(x),
            air_gap=radargram.air_gap,
            offset_m=radargram.antenna_offset_m,
        )
        points.append((float(x), depth))
    return tuple(points)


def _band_top_notes(picks: BandPicks) -> tuple[str, ...]:
    """Band spec 13 R2-R3: notes on what the band top may be instead of the pipe.

    The plate note goes on every result because a plate cannot be told apart
    from a pipe by the band alone. Without background removal no layer can be
    erased, so the layer note is added even when no second event was seen.
    """
    notes = [PROTECTIVE_PLATE_NOTE]
    if "band_top_multi_event" in picks.flags or picks.bg_mode == "none":
        notes.append(LAYER_AMBIGUITY_NOTE)
    return tuple(notes)


def _assemble(
    context: _BandContext,
    group: _GroupResult,
    pool: Sequence[VelocityEstimate],
    survey_pool: Sequence[VelocityEstimate],
    cfg: BandConfig,
    neighbors: tuple[tuple[int, float], ...],
) -> DepthResult:
    radargram, zero, picks, fit = (
        context.radargram,
        context.zero,
        group.picks,
        group.fit,
    )
    flags = [*context.flags, *picks.flags, *_edge_flags(picks.box, radargram.n_traces)]
    if picks.bg_mode == "none":
        flags.append("bg_none")
    if neighbors:
        flags.append("fragment_of_group")
    if fit is not None:
        flags.extend(fit.flags)
    common = {
        "target_class": TARGET_CLASS,
        "zero_rule": zero.zero_rule,
        "zero_event_label": zero.zero_event_label,
        "n_picks": picks.n_picks,
        "bias_notes": _unique([*zero.bias_notes, *_band_top_notes(picks)]),
    }
    scaled = radargram.has_physical_scale
    if not scaled:
        if zero.k_zero is not None:
            flags.append("zero_px_unverified")
        # Band spec 6-2 v1.3: no_scale wins over every failure; the failure
        # name is kept as a flag.
        if group.status != DepthStatus.OK:
            flags.append(group.status.value)
        return DepthResult(
            DepthStatus.NO_SCALE,
            band=_geometry(context, group, neighbors),
            flags=_unique(flags),
            **common,
        )
    scaled_common = {
        **common,
        "t_zero_ns": zero.t_zero_ns,
        "h_est_m": zero.h_est_m,
        "sigma_zero_ns": zero.sigma_zero_ns,
        "radius_used_m": radargram.pipe_radius_m or 0.0,
    }
    if zero.failed:
        return DepthResult(
            DepthStatus.ZERO_FAILED,
            band=_geometry(context, group, neighbors),
            flags=_unique(flags),
            **scaled_common,
        )
    if group.status != DepthStatus.OK or fit is None:
        return DepthResult(
            group.status,
            band=_geometry(context, group, neighbors),
            flags=_unique(flags),
            **scaled_common,
        )
    return _with_velocity(
        context, group, pool, survey_pool, cfg, neighbors, flags, scaled_common
    )


def _with_velocity(
    context: _BandContext,
    group: _GroupResult,
    pool: Sequence[VelocityEstimate],
    survey_pool: Sequence[VelocityEstimate],
    cfg: BandConfig,
    neighbors: tuple[tuple[int, float], ...],
    flags: list[str],
    common: dict,
) -> DepthResult:
    radargram, zero = context.radargram, context.zero
    fit = group.fit
    assert fit is not None and zero.t_zero_ns is not None
    choice: VelocityChoice = select_band_velocity(
        radargram, fit.tau_ref, pool, survey_pool, cfg
    )
    flags.extend(choice.flags)
    drifting_zero = (
        zero.window_std_ns is not None and zero.window_std_ns > ZERO_DRIFT_NS
    )
    if fit.model == "sloped" and radargram.air_gap == "air" and drifting_zero:
        flags.append("slope_may_include_zero_drift")
    common = {
        **common,
        "t0_ns": fit.tau_ref,
        "v_rule": choice.v_rule.value,
        "bias_notes": (*common["bias_notes"], *choice.bias_notes),
        "sigma_apex_ns": fit.sigma_tau_ref,
        "alt_depths": alternative_depths(
            radargram,
            zero,
            fit.tau_ref,
            fit.tau_ref,
            choice.v_m_per_ns,
            choice.alt_meta_v,
        ),
    }
    if not choice.has_velocity:
        return DepthResult(
            DepthStatus.NO_VELOCITY,
            band=_geometry(context, group, neighbors),
            flags=_unique(flags),
            **common,
        )
    v = choice.v_m_per_ns
    assert (
        v is not None and choice.sigma_v is not None and choice.eps_r_used is not None
    )
    depth_m = depth_from_time(
        v, fit.tau_ref, air_gap=radargram.air_gap, offset_m=radargram.antenna_offset_m
    )
    geometry = _geometry(
        context, group, neighbors, v=v, eps_r=choice.eps_r_used, depth_m=depth_m
    )
    assert geometry is not None
    notes = list(common["bias_notes"])
    if geometry.lateral_bias_max_m is not None:
        notes.append(
            f"lateral_offset_unknown: deep by up to {geometry.lateral_bias_max_m:.3f} m"
        )
    if geometry.grade is not None and abs(geometry.grade) > STEEP_GRADE:
        flags.append("steep_band")
    common["bias_notes"] = tuple(notes)
    radius = radargram.pipe_radius_m
    return DepthResult(
        DepthStatus.OK,
        depth_top_m=depth_m,
        depth_center_m=None if radius is None else depth_m + radius,
        sigma_depth_m=depth_uncertainty(
            depth_m, v, choice.sigma_v, zero.sigma_zero_ns, fit.sigma_tau_ref
        ),
        v_m_per_ns=v,
        sigma_v=choice.sigma_v,
        eps_r_used=choice.eps_r_used,
        band=geometry,
        flags=_unique(flags),
        **common,
    )


def _unique(flags: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(flags))


__all__ = [
    "LAYER_AMBIGUITY_NOTE",
    "PROTECTIVE_PLATE_NOTE",
    "BandConfig",
    "BandFit",
    "BandPicks",
    "PixelLimits",
    "band_exclude_traces",
    "candidate_pairs",
    "estimate_band_depths",
    "extract_band_picks",
    "fit_band_profile",
    "group_band_boxes",
    "masked_background",
    "meta_velocity",
    "pair_pick_difference",
    "pixel_limits",
]
