"""Hyperbola point extraction inside one L0 box (spec_depth_v1 section 5).

1. Widen the box by 10 % per side along traces.
2. **Apex candidate**: per trace, the *first* peak after the mute that reaches 30 %
   of the box maximum; the trace with the earliest such peak is the apex. The
   first-peak rule keeps the pipe top when a later event is stronger
   (water-filled pipe bottom, M18: 2.17 m with the global maximum, 1.027 m now).
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


@dataclass(frozen=True)
class ApexCandidate:
    """Earliest first peak in the box (L0 indices)."""

    trace: int
    k: float
    amplitude: float


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


def find_apex_candidate(
    signal: np.ndarray, box: GridBox, candidate_rel: float
) -> ApexCandidate | None:
    """Earliest first peak above ``candidate_rel`` x box maximum (section 5-2).

    Args:
        signal: Pick signal (envelope or signed), ``[n_samples, n_columns]``.
        box: Search box in the coordinates of ``signal``; peaks are looked for
            in rows ``k0..k1``.
        candidate_rel: Threshold relative to the maximum inside ``box``.
    """
    window = signal[box.sample_slice, box.trace_slice]
    threshold = candidate_rel * float(window.max())
    if not threshold > 0:
        return None

    best: ApexCandidate | None = None
    for column, trace in enumerate(range(box.i0, box.i1 + 1)):
        peaks, _ = find_peaks(window[:, column], height=threshold)
        if len(peaks) == 0:
            continue
        k_local, amplitude = parabolic_peak(window[:, column], int(peaks[0]))
        k = box.k0 + k_local
        if best is None or k < best.k:
            best = ApexCandidate(trace=trace, k=k, amplitude=amplitude)
    return best


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


def extract_picks(
    section: np.ndarray,
    box: GridBox,
    zero: ZeroResult,
    *,
    dt_ns: float | None,
    dx_m: float | None,
    f_c_hz: float | None,
    config: DepthConfig = DEFAULT_DEPTH_CONFIG,
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

    Returns:
        Picks sorted by trace. ``box_in_zero_zone`` (too little box below the
        mute) and ``no_apex_candidate`` give no picks.

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
    candidate = find_apex_candidate(signal, local_box, config.candidate_rel)
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
