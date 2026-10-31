"""Hyperbola point extraction inside one L0 box (spec_depth_v1 section 5).

1. Widen the box by 10 % per side along traces.
2. **Apex candidate**: per trace, the *first* peak after the mute that reaches 30 %
   of the box maximum; the trace with the earliest such peak is the apex. The
   first-peak rule keeps the pipe top when a later event is stronger
   (water-filled pipe bottom, M18: 2.17 m with the global maximum, 1.027 m now).
   Peaks within 2 samples of the search top are skipped (box-top boundary
   effect, section 5-7); an optional apex hint limits the search to +-3 traces
   and +-0.5 period around a known apex.
3. **Tracking**: walk out trace by trace, taking the window maximum after the
   previous pick, until the signal fades, the aperture ends or the pick jumps
   earlier (another event).

Input is the background-removed, **un-gained** section and the radargram's
``ZeroResult`` (mute and shared pick mode). Without a physical scale only the
apex candidate is returned, unmuted (tracking windows are in ns and m).
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from gpr_detection.depth.config import DEFAULT_DEPTH_CONFIG, DepthConfig
from gpr_detection.io.radargram import GridBox
from gpr_detection.preprocess.filters import period_ns_or_default
from gpr_detection.preprocess.signals import (
    parabolic_peak,
    pick_signal,
    require_same_pick_mode,
)
from gpr_detection.preprocess.timezero import DEFAULT_PERIOD_NS, ZeroResult

APEX_AT_BOX_TOP = "apex_at_box_top"
"""The plain rule would have taken a peak on the search top (section 5-7 item 2)."""
APEX_HINT_UNMATCHED = "apex_hint_unmatched"
"""No peak above the threshold inside the apex-hint window (section 5-7 item 1)."""
MAD_TO_SIGMA = 1.4826
"""Gaussian sigma from the median absolute deviation."""
MIN_HINT_HALF_PX = 2.0
"""No-scale hint window: at least 2 px above and below the hinted sample."""


@dataclass(frozen=True)
class ApexCandidate:
    """Earliest first peak in the box (L0 indices)."""

    trace: int
    k: float
    amplitude: float


@dataclass(frozen=True)
class ApexHint:
    """Known apex position that restricts the apex search (section 5-7 item 1).

    Given only where an apex candidate already exists (4TU evaluation).

    Attributes:
        trace: Apex trace (L0).
        sample: Apex sample (L0, fractional).
    """

    trace: int
    sample: float


@dataclass(frozen=True)
class PickScale:
    """Grid scale and zero needed to turn L0 picks into ``(x, tau)``."""

    dt_ns: float | None
    dx_m: float | None
    t_zero_ns: float | None

    def require(self) -> tuple[float, float, float]:
        if self.dt_ns is None or self.dx_m is None or self.t_zero_ns is None:
            raise ValueError("picks have no physical scale or no time zero")
        return self.dt_ns, self.dx_m, self.t_zero_ns


@dataclass(frozen=True)
class HyperbolaPicks:
    """Picks of one hyperbola, sorted by trace.

    Attributes:
        box: Box actually searched (widened along traces, top cut at the mute).
        trace_idx: Trace index of every pick, ascending (int).
        k_pick: Pick position in fractional samples (L0).
        amplitude: Pick-signal value at each pick.
        apex: The candidate, or ``None`` when no peak reached the threshold.
        n_left: Picks left of the apex (apex excluded).
        n_right: Picks right of the apex (apex excluded).
        scale: ``dt_ns``, ``dx_m`` and ``t_zero_ns`` used for ``x_m``/``tau_ns``.
        flags: e.g. ``apex_at_box_edge``, ``box_in_zero_zone``, ``no_apex_candidate``,
            ``mute_unavailable``.
    """

    box: GridBox
    trace_idx: np.ndarray
    k_pick: np.ndarray
    amplitude: np.ndarray
    apex: ApexCandidate | None
    n_left: int
    n_right: int
    scale: PickScale
    flags: tuple[str, ...]

    @property
    def n_picks(self) -> int:
        """Number of picks, apex included."""
        return len(self.trace_idx)

    @property
    def x_m(self) -> np.ndarray:
        """Pick positions ``trace * dx`` in m (raises without scale)."""
        _, dx_m, _ = self.scale.require()
        return self.trace_idx * dx_m

    @property
    def tau_ns(self) -> np.ndarray:
        """Pick times after the zero, ``k * dt - t_z`` in ns (raises without scale)."""
        dt_ns, _, t_zero_ns = self.scale.require()
        return self.k_pick * dt_ns - t_zero_ns

    @property
    def apex_x_m(self) -> float:
        """Apex candidate position in m."""
        _, dx_m, _ = self.scale.require()
        return self._require_apex().trace * dx_m

    @property
    def apex_tau_ns(self) -> float:
        """Apex candidate time after the zero in ns (``t_a - t_z``)."""
        dt_ns, _, t_zero_ns = self.scale.require()
        return self._require_apex().k * dt_ns - t_zero_ns

    def _require_apex(self) -> ApexCandidate:
        if self.apex is None:
            raise ValueError("no apex candidate")
        return self.apex

    def is_enough_for_curvature(self, config: DepthConfig) -> bool:
        """Section 5-4: at least ``min_picks`` and ``min_side_picks`` per side."""
        return (
            self.n_picks >= config.min_picks
            and min(self.n_left, self.n_right) >= config.min_side_picks
        )


def widen_box(box: GridBox, n_traces: int, fraction: float) -> GridBox:
    """Widen ``box`` by ``fraction`` of its width on each side, clipped to the line."""
    margin = math.ceil(fraction * box.n_traces)
    return GridBox(
        i0=max(0, box.i0 - margin),
        i1=min(n_traces - 1, box.i1 + margin),
        k0=box.k0,
        k1=box.k1,
    )


def _hint_window(
    box: GridBox, hint: ApexHint, hint_halfwidth: tuple[int, float]
) -> GridBox | None:
    """Hint window intersected with ``box``; ``None`` when nothing is left."""
    half_traces, half_samples = hint_halfwidth
    i0 = max(box.i0, hint.trace - half_traces)
    i1 = min(box.i1, hint.trace + half_traces)
    k0 = max(box.k0, math.ceil(hint.sample - half_samples))
    k1 = min(box.k1, math.floor(hint.sample + half_samples))
    if i0 > i1 or k1 <= k0:
        return None
    return GridBox(i0=i0, i1=i1, k0=k0, k1=k1)


def find_apex_candidate(
    signal: np.ndarray,
    box: GridBox,
    candidate_rel: float,
    *,
    top_guard: int,
    hint: ApexHint | None = None,
    hint_halfwidth: tuple[int, float] | None = None,
    peak_floor: float = 0.0,
) -> tuple[ApexCandidate | None, tuple[str, ...]]:
    """Earliest first peak above ``candidate_rel`` x maximum (sections 5-2, 5-7).

    Per trace the first peak above the threshold is taken, skipping peaks
    whose integer index lies ``top_guard`` or fewer rows below ``box.k0``
    (box-top boundary peaks: the cut tail or front of a shallower event).
    With a hint the search, including the maximum for the threshold, is
    limited to the hint window and never falls back to the whole box.

    Args:
        signal: Pick signal (envelope or signed), ``[n_samples, n_columns]``.
        box: Search box in the coordinates of ``signal``; peaks are looked for
            in rows ``k0..k1``. ``box.k0`` is the search top ``k_top`` (the
            later of the widened box top and the mute).
        candidate_rel: Threshold relative to the maximum inside the search area.
        top_guard: Integer peak index ``k_p - k_top <= top_guard`` is rejected.
        hint: Known apex in the coordinates of ``signal``.
        hint_halfwidth: ``(traces, samples)`` half-size of the hint window;
            required with ``hint``.
        peak_floor: Absolute minimum peak value. The hint path passes
            :func:`hint_peak_floor` (v1.8 supplement); 0 keeps the relative
            rule only (the path without a hint).

    Returns:
        The candidate (or ``None``) and flags: ``apex_at_box_top`` when the
        plain rule would have taken a rejected peak, ``apex_hint_unmatched``
        when a hint window holds no accepted peak.

    Raises:
        ValueError: If ``hint`` is given without ``hint_halfwidth``.
    """
    if hint is not None and hint_halfwidth is None:
        raise ValueError("hint needs hint_halfwidth")
    unmatched = () if hint is None else (APEX_HINT_UNMATCHED,)
    search = box if hint is None else _hint_window(box, hint, hint_halfwidth)
    if search is None:
        return None, unmatched
    window = signal[search.sample_slice, search.trace_slice]
    threshold = max(candidate_rel * float(window.max()), peak_floor)
    if not threshold > 0:
        return None, unmatched

    best: ApexCandidate | None = None
    plain_best_k = math.inf
    plain_best_rejected = False
    for column, trace in enumerate(range(search.i0, search.i1 + 1)):
        values = window[:, column]
        peaks, _ = find_peaks(values, height=threshold)
        if len(peaks) == 0:
            continue
        # Integer indices relative to the search top decide the rejection.
        is_rejected = search.k0 + peaks - box.k0 <= top_guard
        plain_k = search.k0 + parabolic_peak(values, int(peaks[0]))[0]
        if plain_k < plain_best_k:
            plain_best_k, plain_best_rejected = plain_k, bool(is_rejected[0])
        accepted = peaks[~is_rejected]
        if len(accepted) == 0:
            continue
        k_local, amplitude = parabolic_peak(values, int(accepted[0]))
        k = search.k0 + k_local
        if best is None or k < best.k:
            best = ApexCandidate(trace=trace, k=k, amplitude=amplitude)

    flags = (APEX_AT_BOX_TOP,) if plain_best_rejected else ()
    if best is None:
        flags = (*flags, *unmatched)
    return best, flags


@dataclass(frozen=True)
class _TrackLimits:
    """Tracking window and stop rules in samples and traces."""

    back_samples: float
    forward_samples: float
    reverse_samples: float
    stop_amplitude: float
    max_steps: int


def _track_one_side(
    signal: np.ndarray,
    candidate: ApexCandidate,
    step: int,
    box: GridBox,
    limits: _TrackLimits,
) -> list[tuple[int, float, float]]:
    """Picks ``(trace, k, amplitude)`` walking from the apex in direction ``step``."""
    picks: list[tuple[int, float, float]] = []
    previous_k = candidate.k
    n_samples = signal.shape[0]
    for distance in range(1, limits.max_steps + 1):
        trace = candidate.trace + step * distance
        if not box.i0 <= trace <= box.i1:
            break
        lo = max(0, math.floor(previous_k - limits.back_samples))
        hi = min(n_samples - 1, math.ceil(previous_k + limits.forward_samples))
        if hi <= lo:
            break
        column = signal[:, trace]
        k_peak = lo + int(np.argmax(column[lo : hi + 1]))
        k, amplitude = parabolic_peak(column, k_peak)
        if amplitude < limits.stop_amplitude:
            break
        if k < previous_k - limits.reverse_samples:
            break
        picks.append((trace, k, amplitude))
        previous_k = k
    return picks


def mute_sample(t_zero_ns: float, period_ns: float, dt_ns: float) -> int:
    """First sample after ``t_z + 1 / f_c`` (section 5-2 ``k_mute``)."""
    return math.ceil((t_zero_ns + period_ns) / dt_ns)


def _empty_picks(
    box: GridBox, scale: PickScale, flags: tuple[str, ...]
) -> HyperbolaPicks:
    empty_int = np.zeros(0, dtype=int)
    empty = np.zeros(0)
    return HyperbolaPicks(box, empty_int, empty, empty, None, 0, 0, scale, flags)


def _clip_box_to_mute(
    box: GridBox, k_mute: int | None, min_height_samples: float
) -> GridBox | None:
    """Cut the box top at the mute (section 10); ``None`` if too little remains."""
    if k_mute is None or box.k0 >= k_mute:
        return box
    if box.k1 - k_mute + 1 < min_height_samples:
        return None
    return GridBox(i0=box.i0, i1=box.i1, k0=k_mute, k1=box.k1)


def _mute_for(
    zero: ZeroResult, dt_ns: float | None, period_ns: float
) -> tuple[int | None, list[str]]:
    """``k_mute`` from the zero; no mute without scale or without a zero time."""
    if dt_ns is None:
        return None, []
    if zero.t_zero_ns is None:
        return None, ["mute_unavailable"]
    return mute_sample(zero.t_zero_ns, period_ns, dt_ns), []


def noise_sigma(section: np.ndarray, searched: GridBox) -> float:
    """``1.4826 MAD`` of the background-removed amplitude (section 5-7 item 1 (i)).

    Taken over the widened-box traces from the search top to the record end;
    events are sparse there, so this is the noise plus clutter level.
    """
    values = np.asarray(section[searched.k0 :, searched.trace_slice], dtype=float)
    if values.size == 0:
        return 0.0
    deviation = np.abs(values - np.median(values))
    return MAD_TO_SIGMA * float(np.median(deviation))


def hint_peak_floor(
    section: np.ndarray,
    signal: np.ndarray,
    searched: GridBox,
    local_box: GridBox,
    config: DepthConfig,
) -> float:
    """Hint-path event floor ``max(k sigma_n, r E_box_max)`` (5-7 item 1, v1.8).

    Args:
        section: Background-removed section (all traces, L0).
        signal: Pick signal of the searched columns (``local_box`` coordinates).
        searched: Widened, mute-clipped box on the original grid.
        local_box: The same box in ``signal`` coordinates.
        config: ``hint_noise_k`` and ``hint_box_rel_floor``.
    """
    noise_floor = config.hint_noise_k * noise_sigma(section, searched)
    box_max = float(signal[local_box.sample_slice, local_box.trace_slice].max())
    return max(noise_floor, config.hint_box_rel_floor * box_max)


def _hint_halfwidth(
    config: DepthConfig,
    dt_ns: float | None,
    period_ns: float,
    period_px: float | None,
) -> tuple[int, float]:
    """Hint window half-size: 3 traces and 0.5 period (no scale: max(2, 0.5 P_px))."""
    if dt_ns is not None:
        return config.hint_traces, config.hint_periods * period_ns / dt_ns
    half_px = MIN_HINT_HALF_PX
    if period_px is not None:
        half_px = max(MIN_HINT_HALF_PX, config.hint_periods * period_px)
    return config.hint_traces, half_px


def extract_picks(
    section: np.ndarray,
    box: GridBox,
    zero: ZeroResult,
    *,
    dt_ns: float | None,
    dx_m: float | None,
    f_c_hz: float | None,
    config: DepthConfig = DEFAULT_DEPTH_CONFIG,
    apex_hint: ApexHint | None = None,
    period_px: float | None = None,
) -> HyperbolaPicks:
    """Find the apex candidate and track the hyperbola picks (spec_depth_v1 section 5).

    Args:
        section: Background-removed, un-gained data ``[n_samples, n_traces]``.
        box: Detection box on the original grid (L0).
        zero: Time zero of the same radargram; sets the mute ``t_z + 1 / f_c``
            and must share ``config.pick_mode``.
        dt_ns: Sample interval; ``None`` returns the apex candidate only, unmuted.
        dx_m: Trace interval; ``None`` returns the apex candidate only.
        f_c_hz: Centre frequency; ``None`` assumes a 2 ns period.
        config: Thresholds, aperture and pick mode.
        apex_hint: Known apex (L0) limiting the apex search to the hint window
            (section 5-7 item 1); tracking still uses the widened box.
        period_px: No-scale dominant period (``FittingBranch.period_px``) that
            sizes the hint window ``max(2, 0.5 P_px)`` px; 2 px when ``None``.

    Returns:
        Picks sorted by trace. ``box_in_zero_zone`` (too little box below the
        mute) and ``no_apex_candidate`` give no picks; with a hint the latter
        comes with ``apex_hint_unmatched``.

    Raises:
        ValueError: If the zero was picked with another ``pick_mode`` (U6).
    """
    require_same_pick_mode(zero.pick_mode, config.pick_mode)
    period_ns = period_ns_or_default(f_c_hz, DEFAULT_PERIOD_NS)
    k_mute, flags = _mute_for(zero, dt_ns, period_ns)
    scale = PickScale(dt_ns, dx_m, zero.t_zero_ns)

    widened = widen_box(box, section.shape[1], config.box_expand_frac)
    min_height = 0.0 if dt_ns is None else 2.0 * period_ns / dt_ns
    searched = _clip_box_to_mute(widened, k_mute, min_height)
    if searched is None:
        return _empty_picks(widened, scale, (*flags, "box_in_zero_zone"))

    # Work on the searched columns only; local trace j is global searched.i0 + j.
    signal = pick_signal(section[:, searched.trace_slice], config.pick_mode)
    local_box = GridBox(i0=0, i1=searched.n_traces - 1, k0=searched.k0, k1=searched.k1)
    local_hint = (
        None
        if apex_hint is None
        else ApexHint(trace=apex_hint.trace - searched.i0, sample=apex_hint.sample)
    )
    candidate, candidate_flags = find_apex_candidate(
        signal,
        local_box,
        config.candidate_rel,
        top_guard=config.top_guard_samples
        if dt_ns is not None
        else config.top_guard_px,
        hint=local_hint,
        hint_halfwidth=_hint_halfwidth(config, dt_ns, period_ns, period_px),
        peak_floor=0.0
        if apex_hint is None
        else hint_peak_floor(section, signal, searched, local_box, config),
    )
    flags.extend(candidate_flags)
    if candidate is None:
        return _empty_picks(searched, scale, (*flags, "no_apex_candidate"))

    apex_trace = searched.i0 + candidate.trace
    near_left = apex_trace - box.i0 < config.apex_edge_traces
    near_right = box.i1 - apex_trace < config.apex_edge_traces
    if near_left or near_right:
        flags.append("apex_at_box_edge")

    left: list[tuple[int, float, float]] = []
    right: list[tuple[int, float, float]] = []
    if dt_ns is not None and dx_m is not None:
        limits = _TrackLimits(
            back_samples=config.track_back_ns / dt_ns,
            forward_samples=(2.0 * dx_m / config.v_min + config.track_back_ns) / dt_ns,
            reverse_samples=config.track_reverse_ns / dt_ns,
            stop_amplitude=config.track_stop_rel * candidate.amplitude,
            max_steps=math.floor(config.a_max_m / dx_m + 1e-9),
        )
        left = _track_one_side(signal, candidate, -1, local_box, limits)
        right = _track_one_side(signal, candidate, +1, local_box, limits)

    ordered = [
        *reversed(left),
        (candidate.trace, candidate.k, candidate.amplitude),
        *right,
    ]
    return HyperbolaPicks(
        box=searched,
        trace_idx=np.array([searched.i0 + j for j, _, _ in ordered], dtype=int),
        k_pick=np.array([k for _, k, _ in ordered]),
        amplitude=np.array([amplitude for _, _, amplitude in ordered]),
        apex=ApexCandidate(
            trace=apex_trace, k=candidate.k, amplitude=candidate.amplitude
        ),
        n_left=len(left),
        n_right=len(right),
        scale=scale,
        flags=tuple(flags),
    )
