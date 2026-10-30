"""Time-zero detection and selection (spec_depth_v1 v1.2 section 4).

Two stages:

1. :func:`detect_zero_events` finds the early events (direct wave, surface
   reflection) on the mean trace of the band-passed data **before** background
   removal. Traces of ``pipe_band`` boxes can be left out of the mean
   (``exclude_traces``, spec_band_depth_v1 section 3-1). Long profiles also get
   the same detection per trace window, for QC only.
2. :func:`select_zero` applies the dataset rule (air: late event, ground: direct
   peak, processed data: sample 0) and returns a :class:`ZeroResult`. It also
   tracks the chosen event through the trace windows (``window_std_ns``) and
   labels the metadata time zeros by the rule's own events (section 4-3).

Times are in ns when the radargram has ``dt_ns``; for no-scale input only the
sample (pixel) positions are filled.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal

import numpy as np
from scipy.signal import find_peaks

from gpr_detection.io.radargram import SPEED_OF_LIGHT_M_PER_NS, Radargram
from gpr_detection.preprocess.filters import (
    BACKGROUND_WINDOW_TRACES,
    period_ns_or_default,
)
from gpr_detection.preprocess.signals import (
    PickMode,
    check_pick_mode,
    parabolic_peak,
    pick_signal,
)

DEFAULT_PERIOD_NS = 2.0
"""``1 / f_c`` used when ``f_c`` is unknown (500 MHz, assumed)."""

DIRECT_WAVE_DELAY_TABLE_NS: dict[float, tuple[tuple[float, ...], tuple[float, ...]]] = {
    400e6: ((4.0, 9.0, 16.0), (0.53, 0.725, 0.96)),
    500e6: ((4.0, 9.0, 16.0), (0.53, 0.727, 1.03)),
}
"""Direct-wave peak delay after transmission, ``eps_r -> ns`` (2-D, s = 0.1 m)."""

DELAY_TABLE_F_C_RANGE_HZ = (350e6, 550e6)
DELAY_TABLE_EPS_RANGE = (4.0, 16.0)
DELAY_TABLE_OFFSET_M = 0.1
DELAY_TABLE_OFFSET_TOLERANCE_M = 0.02
"""Validity of the direct-wave delay table; outside it no shift is applied."""

SEPARATION_PERIODS = 0.75
"""Opposite-polarity direct and surface events split from a gap of 0.75 / f_c."""

SIGMA_ZERO_NS: dict[str, float] = {
    "air_surface_late": 0.3,
    "air_single_event": 0.25,
    "ground_direct_peak": 0.15,
    "ground_direct_shifted": 0.15,
    "meta_sample0": 0.5,
}
"""Standard uncertainty of each zero rule (spec_depth_v1 section 9).

``air_surface_late`` is 0.3 ns (user decision 2026-10-30): direct-wave
interference delays the surface peak by 0.12-0.29 ns after steps (1)-(3).
"""

ZeroEventLabel = Literal["surface_reflection", "direct_wave", "merged", "meta"]
MetaT0Label = Literal["direct_wave", "surface_reflection", "merged", "unclassified"]


class ZeroRule(StrEnum):
    """Time-zero rule names (spec_depth_v1 section 4-2)."""

    AIR_SURFACE_LATE = "air_surface_late"
    AIR_SINGLE_EVENT = "air_single_event"
    GROUND_DIRECT_PEAK = "ground_direct_peak"
    GROUND_DIRECT_SHIFTED = "ground_direct_shifted"
    META_SAMPLE0 = "meta_sample0"
    GIVEN = "given"
    """Zero supplied by the caller (tests, known transmit time)."""
    FAILED = "zero_failed"


class GroundZeroShift(StrEnum):
    """Optional correction of the ground-coupled direct-wave zero."""

    NONE = "none"
    SYNTHETIC_TABLE = "synthetic_table"


@dataclass(frozen=True)
class ZeroConfig:
    """Settings of time-zero detection (spec_depth_v1 sections 4-1 to 4-3).

    Attributes:
        h_max_m: Largest antenna height searched for the surface reflection.
        early_fraction: Fraction of the record searched for the strongest event.
        height_rel: ``find_peaks`` height, relative to the window maximum.
        prominence_rel: ``find_peaks`` prominence, relative to the window maximum.
        window_traces: Trace window of the per-window QC.
        window_match_ns: A window peak this close to the chosen event is a match.
        window_drift_ns: ``window_std_ns`` above this flags ``zero_window_drift``.
        meta_tolerance_ns: Max distance for labelling a metadata zero by an
            event (single air events use ``max(this, 0.5 / f_c)``).
        ground_zero_shift: ``"synthetic_table"`` subtracts the tabulated
            direct-wave delay; off by default (only valid for s = 0.1 m).
    """

    h_max_m: float = 0.5
    early_fraction: float = 0.25
    height_rel: float = 0.1
    prominence_rel: float = 0.05
    window_traces: int = BACKGROUND_WINDOW_TRACES
    window_match_ns: float = 0.5
    window_drift_ns: float = 0.2
    meta_tolerance_ns: float = 0.5
    ground_zero_shift: GroundZeroShift = GroundZeroShift.NONE


DEFAULT_ZERO_CONFIG = ZeroConfig()


@dataclass(frozen=True)
class ZeroEvents:
    """Early events found on the mean trace (spec_depth_v1 section 4-1).

    Attributes:
        events_k: Peak positions in fractional samples, earliest first.
        amplitudes: Pick-signal value at each peak.
        dt_ns: Sample interval, ``None`` for no-scale input.
        f_c_hz: Centre frequency used for the windows, if known.
        pick_mode: Phase reference used for the peaks.
        window_end_k: Last sample of the search window.
        h_est_m: Antenna height from the direct/late gap (two events and scale).
        window_events_k: Peaks of each trace window (QC), empty when the
            profile is not longer than one window or has no scale.
        excluded_traces: Number of traces left out of the mean trace.
        flags: e.g. ``zero_band_overlap``, ``h_est_offset_unknown``.
    """

    events_k: tuple[float, ...]
    amplitudes: tuple[float, ...]
    dt_ns: float | None
    f_c_hz: float | None
    pick_mode: PickMode
    window_end_k: int
    h_est_m: float | None
    window_events_k: tuple[tuple[float, ...], ...]
    excluded_traces: int
    flags: tuple[str, ...]

    @property
    def n_events(self) -> int:
        """Number of events in the search window."""
        return len(self.events_k)

    @property
    def events_ns(self) -> tuple[float, ...]:
        """Event times in ns (empty without scale)."""
        if self.dt_ns is None:
            return ()
        return tuple(k * self.dt_ns for k in self.events_k)


@dataclass(frozen=True)
class ZeroResult:
    """Selected depth zero for one radargram (spec_depth_v1 section 13-1).

    Attributes:
        t_zero_ns: Depth zero in ns; ``None`` without scale or on failure.
        k_zero: Depth zero in fractional samples (also for no-scale input).
        zero_rule: Rule name; ``"zero_failed"`` when no event was found.
        zero_event_label: Which event the zero is; ``None`` only on failure.
        pick_mode: Phase reference; the apex must use the same one.
        events_ns: All early events in ns (empty without scale).
        h_est_m: Antenna height estimate (air-coupled, two events).
        t_zero_per_trace_ns: Optional per-trace zero (not computed in v1.1).
        meta_t0: ``{name: (ns, label)}`` for metadata zeros (section 4-3).
        excluded_traces: Traces left out of the mean trace.
        flags: e.g. ``zero_failed``, ``zero_band_overlap``, ``zero_window_drift``.
        sigma_zero_ns: Standard uncertainty of the zero (section 9).
        bias_notes: Signed systematic biases of the chosen rule.
        window_std_ns: Std of the chosen event between trace windows (QC).
    """

    t_zero_ns: float | None
    k_zero: float | None
    zero_rule: str
    zero_event_label: ZeroEventLabel | None
    pick_mode: PickMode
    events_ns: tuple[float, ...]
    h_est_m: float | None
    t_zero_per_trace_ns: np.ndarray | None
    meta_t0: Mapping[str, tuple[float, MetaT0Label]]
    excluded_traces: int
    flags: tuple[str, ...]
    sigma_zero_ns: float = 0.0
    bias_notes: tuple[str, ...] = ()
    window_std_ns: float | None = None

    @property
    def failed(self) -> bool:
        """True when no zero could be selected."""
        return self.zero_rule == ZeroRule.FAILED


# --------------------------------------------------------------------- detection


def _included_trace_mask(
    n_traces: int, exclude_traces: np.ndarray | None
) -> tuple[np.ndarray, list[str]]:
    """Boolean mask of traces in the mean; all traces if everything is excluded."""
    keep = np.ones(n_traces, dtype=bool)
    if exclude_traces is None:
        return keep, []
    excluded = np.asarray(exclude_traces)
    if excluded.dtype == bool:
        if excluded.shape != (n_traces,):
            raise ValueError(
                f"boolean exclude_traces needs shape ({n_traces},), "
                f"got {excluded.shape}"
            )
        keep = ~excluded
    else:
        keep[excluded.astype(int)] = False
    if not keep.any():
        return np.ones(n_traces, dtype=bool), ["zero_band_overlap"]
    return keep, []


def _search_window_end(
    signal: np.ndarray, dt_ns: float | None, f_c_hz: float | None, config: ZeroConfig
) -> int:
    """Last sample of ``[0, t_max + 2 h_max / c + 0.5 / f_c]`` (section 4-1 item 2)."""
    n_early = max(3, int(len(signal) * config.early_fraction))
    if dt_ns is None:
        # No time axis: the early quarter in pixels (relative reference only).
        return n_early - 1
    k_max = int(np.argmax(signal[:n_early]))
    period_ns = period_ns_or_default(f_c_hz, DEFAULT_PERIOD_NS)
    extra_ns = 2.0 * config.h_max_m / SPEED_OF_LIGHT_M_PER_NS + 0.5 * period_ns
    return min(len(signal) - 1, k_max + int(np.ceil(extra_ns / dt_ns)))


def _early_peaks(
    mean_trace: np.ndarray,
    dt_ns: float | None,
    f_c_hz: float | None,
    pick_mode: PickMode,
    config: ZeroConfig,
) -> tuple[list[tuple[float, float]], int]:
    """``[(k, amplitude)]`` of the early events and the window end sample."""
    signal = pick_signal(mean_trace, pick_mode)
    window_end = _search_window_end(signal, dt_ns, f_c_hz, config)
    window = signal[: window_end + 1]
    window_max = float(window.max())
    if not window_max > 0:
        return [], window_end
    peak_indices, _ = find_peaks(
        window,
        height=config.height_rel * window_max,
        prominence=config.prominence_rel * window_max,
    )
    return [parabolic_peak(window, int(k)) for k in peak_indices], window_end


def _antenna_height_m(gap_ns: float, offset_m: float | None) -> tuple[float, list[str]]:
    """Solve ``t_surf - t_dir = (2 sqrt(h^2 + s^2/4) - s) / c`` for ``h``."""
    if offset_m is None:
        return SPEED_OF_LIGHT_M_PER_NS * gap_ns / 2.0, ["h_est_offset_unknown"]
    half_path = (SPEED_OF_LIGHT_M_PER_NS * gap_ns + offset_m) / 2.0
    return float(np.sqrt(max(half_path**2 - (offset_m / 2.0) ** 2, 0.0))), []


def _window_events_k(
    section: np.ndarray,
    keep: np.ndarray,
    dt_ns: float | None,
    f_c_hz: float | None,
    pick_mode: PickMode,
    config: ZeroConfig,
) -> tuple[tuple[float, ...], ...]:
    """Peaks of every ``window_traces`` window of included traces (QC only)."""
    included = np.flatnonzero(keep)
    if dt_ns is None or len(included) <= config.window_traces:
        return ()
    n_windows = len(included) // config.window_traces
    per_window = []
    for chunk in np.array_split(included, n_windows):
        peaks, _ = _early_peaks(
            section[:, chunk].mean(axis=1), dt_ns, f_c_hz, pick_mode, config
        )
        per_window.append(tuple(k for k, _ in peaks))
    return tuple(per_window)


def detect_zero_events(
    filtered: np.ndarray,
    dt_ns: float | None,
    f_c_hz: float | None,
    *,
    pick_mode: PickMode = "envelope",
    antenna_offset_m: float | None = None,
    exclude_traces: np.ndarray | None = None,
    config: ZeroConfig = DEFAULT_ZERO_CONFIG,
) -> ZeroEvents:
    """Find the early events on the mean trace (spec_depth_v1 section 4-1).

    Args:
        filtered: Data after steps (1)-(3), **before** background removal,
            ``[n_samples, n_traces]``.
        dt_ns: Sample interval; ``None`` gives sample positions only.
        f_c_hz: Centre frequency; ``None`` assumes a 2 ns period.
        pick_mode: Phase reference of the peaks.
        antenna_offset_m: Tx-Rx separation ``s`` for the height estimate.
        exclude_traces: Trace indices (int) or a boolean mask of traces to leave
            out of the mean, e.g. ``pipe_band`` boxes with their margins. If it
            covers every trace, nothing is excluded and ``zero_band_overlap``
            is flagged.
        config: Detection settings.

    Returns:
        Events of the global mean trace, earliest first, plus per-window peaks.
    """
    check_pick_mode(pick_mode)
    section = np.asarray(filtered, dtype=np.float64)
    keep, flags = _included_trace_mask(section.shape[1], exclude_traces)
    peaks, window_end = _early_peaks(
        section[:, keep].mean(axis=1), dt_ns, f_c_hz, pick_mode, config
    )

    h_est_m = None
    if len(peaks) >= 2 and dt_ns is not None:
        gap_ns = (peaks[-1][0] - peaks[0][0]) * dt_ns
        h_est_m, height_flags = _antenna_height_m(gap_ns, antenna_offset_m)
        flags.extend(height_flags)

    return ZeroEvents(
        events_k=tuple(k for k, _ in peaks),
        amplitudes=tuple(amplitude for _, amplitude in peaks),
        dt_ns=dt_ns,
        f_c_hz=f_c_hz,
        pick_mode=pick_mode,
        window_end_k=window_end,
        h_est_m=h_est_m,
        window_events_k=_window_events_k(
            section, keep, dt_ns, f_c_hz, pick_mode, config
        ),
        excluded_traces=int((~keep).sum()),
        flags=tuple(flags),
    )


# --------------------------------------------------------------------- labelling


def direct_wave_delay_ns(eps_r: float, f_c_hz: float) -> float:
    """Tabulated direct-wave peak delay, linear in ``sqrt(eps_r)``.

    The table row with the nearest frequency (400 or 500 MHz) is used. Values
    outside eps_r 4-16 are clamped here; :func:`select_zero` does not apply the
    table outside its validity range at all.
    """
    nearest_hz = min(DIRECT_WAVE_DELAY_TABLE_NS, key=lambda hz: abs(hz - f_c_hz))
    eps_points, delays_ns = DIRECT_WAVE_DELAY_TABLE_NS[nearest_hz]
    return float(np.interp(np.sqrt(eps_r), np.sqrt(eps_points), delays_ns))


def _meta_label_events(
    zero_rule: str,
    events_ns: Sequence[float],
    f_c_hz: float | None,
    tolerance_ns: float,
) -> tuple[list[tuple[float, MetaT0Label]], float]:
    """Events a metadata zero may be labelled with, and the tolerance (4-3 table)."""
    if not events_ns:
        return [], tolerance_ns
    if zero_rule == ZeroRule.AIR_SURFACE_LATE:
        events = [(events_ns[0], "direct_wave"), (events_ns[-1], "surface_reflection")]
        return events, tolerance_ns
    if zero_rule == ZeroRule.AIR_SINGLE_EVENT:
        half_period_ns = 0.5 * period_ns_or_default(f_c_hz, DEFAULT_PERIOD_NS)
        return [(events_ns[0], "merged")], max(tolerance_ns, half_period_ns)
    # Ground rules, meta_sample0, failed, given: only the direct wave. A later
    # peak in a ground-coupled window is not a separate surface reflection.
    return [(events_ns[0], "direct_wave")], tolerance_ns


def classify_meta_t0(
    candidates_ns: Mapping[str, float],
    zero_rule: str,
    events_ns: Sequence[float],
    f_c_hz: float | None = None,
    tolerance_ns: float = 0.5,
) -> dict[str, tuple[float, MetaT0Label]]:
    """Label each metadata zero by the chosen rule's events (section 4-3, v1.1).

    * ``air_surface_late``: nearest of ``direct_wave`` (first event) and
      ``surface_reflection`` (last event), within ``tolerance_ns``.
    * ``air_single_event``: ``merged`` within ``max(tolerance_ns, 0.5 / f_c)``
      (metadata zeros sit on a +-0.4 / f_c lobe of the merged band).
    * Every other rule: ``direct_wave`` (first event) only.

    Args:
        candidates_ns: Metadata zeros in ns, e.g. ``t0_sample_est``.
        zero_rule: Rule chosen by :func:`select_zero`.
        events_ns: Detected events in ns, earliest first.
        f_c_hz: Centre frequency (single-event tolerance); ``None`` = 500 MHz.
        tolerance_ns: Base tolerance; farther zeros are ``"unclassified"``.

    Returns:
        ``{name: (t_ns, label)}`` in the input order.
    """
    events, tolerance = _meta_label_events(zero_rule, events_ns, f_c_hz, tolerance_ns)
    labelled: dict[str, tuple[float, MetaT0Label]] = {}
    for name, t_ns in candidates_ns.items():
        label: MetaT0Label = "unclassified"
        if events:
            nearest_ns, nearest_label = min(events, key=lambda e: abs(e[0] - t_ns))
            if abs(nearest_ns - t_ns) <= tolerance:
                label = nearest_label
        labelled[name] = (float(t_ns), label)
    return labelled


# --------------------------------------------------------------------- rules


@dataclass(frozen=True)
class _RuleChoice:
    """One rule's outcome before it becomes a :class:`ZeroResult`."""

    rule: ZeroRule
    k_event: float
    """Sample of the chosen event (before any table shift); tracked in windows."""
    k_zero: float
    label: ZeroEventLabel
    bias_notes: tuple[str, ...]
    flags: tuple[str, ...] = ()


def _delay_table_applies(radargram: Radargram) -> bool:
    """Section 4-2 validity of the direct-wave delay table (v1.1)."""
    f_c_hz, eps_r, offset_m = (
        radargram.f_c_hz,
        radargram.eps_r_meta,
        radargram.antenna_offset_m,
    )
    if f_c_hz is None or eps_r is None:
        return False
    f_lo, f_hi = DELAY_TABLE_F_C_RANGE_HZ
    eps_lo, eps_hi = DELAY_TABLE_EPS_RANGE
    offset_ok = (
        offset_m is None
        or abs(offset_m - DELAY_TABLE_OFFSET_M) <= DELAY_TABLE_OFFSET_TOLERANCE_M
    )
    return f_lo <= f_c_hz <= f_hi and eps_lo <= eps_r <= eps_hi and offset_ok


def _ground_zero(
    radargram: Radargram, k_direct: float, config: ZeroConfig
) -> _RuleChoice:
    """Ground-coupled rule: direct peak, optionally shifted by the table delay."""
    peak = _RuleChoice(
        rule=ZeroRule.GROUND_DIRECT_PEAK,
        k_event=k_direct,
        k_zero=k_direct,
        label="direct_wave",
        bias_notes=("ground_direct_zero: shallow 0.53-1.03 ns",),
    )
    if config.ground_zero_shift != GroundZeroShift.SYNTHETIC_TABLE:
        return peak
    dt_ns = radargram.dt_ns
    if dt_ns is None or not _delay_table_applies(radargram):
        return replace(peak, flags=("ground_zero_shift_unavailable",))
    delay_ns = direct_wave_delay_ns(radargram.eps_r_meta, radargram.f_c_hz)
    return _RuleChoice(
        rule=ZeroRule.GROUND_DIRECT_SHIFTED,
        k_event=k_direct,
        k_zero=k_direct - delay_ns / dt_ns,
        label="direct_wave",
        bias_notes=("ground_direct_shifted: table from 2-D synthetics, s = 0.1 m",),
    )


def _single_event_bias_note(radargram: Radargram) -> str:
    """Upper bound ``h_sep / sqrt(eps_r)`` of the merged-peak depth bias."""
    formula = "0.375 c / (f_c sqrt(eps_r))"
    if radargram.f_c_hz is None or radargram.eps_r_meta is None:
        return f"air_single_event: merged peak -> deep by up to {formula}"
    period_ns = 1e9 / radargram.f_c_hz
    h_sep_m = SEPARATION_PERIODS / 2.0 * SPEED_OF_LIGHT_M_PER_NS * period_ns
    bound_m = h_sep_m / float(np.sqrt(radargram.eps_r_meta))
    return f"air_single_event: merged peak -> deep by up to {bound_m:.3f} m ({formula})"


def _air_zero(radargram: Radargram, events: ZeroEvents) -> _RuleChoice:
    """Air-coupled rule: late event if two or more, else the merged single peak."""
    if events.n_events >= 2:
        note = (
            "air_surface_late: 0.12-0.29 ns late after (1)-(3) -> 0.6-1.5 cm "
            "shallow at eps 9 (direct-wave interference)"
        )
        k_late = events.events_k[-1]
        return _RuleChoice(
            ZeroRule.AIR_SURFACE_LATE, k_late, k_late, "surface_reflection", (note,)
        )
    k_single = events.events_k[0]
    note = _single_event_bias_note(radargram)
    return _RuleChoice(ZeroRule.AIR_SINGLE_EVENT, k_single, k_single, "merged", (note,))


def _window_qc(
    events: ZeroEvents, k_event: float, config: ZeroConfig
) -> tuple[float | None, list[str]]:
    """Track the chosen event through the trace windows (section 4-1 item 1)."""
    if not events.window_events_k or events.dt_ns is None:
        return None, []
    flags = []
    match_samples = config.window_match_ns / events.dt_ns
    matched_ns = []
    for window_peaks in events.window_events_k:
        if len(window_peaks) != events.n_events:
            flags.append("zero_window_event_count_mismatch")
        nearest = min(window_peaks, key=lambda k: abs(k - k_event), default=None)
        if nearest is None or abs(nearest - k_event) > match_samples:
            flags.append("zero_window_missing")
            continue
        matched_ns.append(nearest * events.dt_ns)
    window_std_ns = float(np.std(matched_ns)) if len(matched_ns) >= 2 else None
    if window_std_ns is not None and window_std_ns > config.window_drift_ns:
        flags.append("zero_window_drift")
    return window_std_ns, list(dict.fromkeys(flags))


def _choose_rule(
    radargram: Radargram, events: ZeroEvents, config: ZeroConfig
) -> _RuleChoice:
    """Section 4-2 table for a radargram with at least one event."""
    if radargram.air_gap == "air":
        return _air_zero(radargram, events)
    choice = _ground_zero(radargram, events.events_k[0], config)
    if radargram.air_gap != "unknown":
        return choice
    unknown_flags = ["air_gap_unknown"]
    if events.n_events >= 2:
        unknown_flags.append("air_gap_unknown_multi_event")
    return replace(choice, flags=(*choice.flags, *unknown_flags))


def _meta_sample0_zero(
    radargram: Radargram, events: ZeroEvents, config: ZeroConfig
) -> ZeroResult:
    meta_t0 = classify_meta_t0(
        radargram.t0_candidates_ns,
        ZeroRule.META_SAMPLE0,
        events.events_ns,
        radargram.f_c_hz,
        config.meta_tolerance_ns,
    )
    return ZeroResult(
        t_zero_ns=None if radargram.dt_ns is None else 0.0,
        k_zero=0.0,
        zero_rule=ZeroRule.META_SAMPLE0,
        zero_event_label="meta",
        pick_mode=events.pick_mode,
        events_ns=events.events_ns,
        h_est_m=None,
        t_zero_per_trace_ns=None,
        meta_t0=meta_t0,
        excluded_traces=events.excluded_traces,
        flags=(*events.flags, "zero_unverified", "amplitude_processed"),
        sigma_zero_ns=SIGMA_ZERO_NS[ZeroRule.META_SAMPLE0],
        bias_notes=("meta_sample0: zero assumed at sample 0",),
    )


def _failed_zero(radargram: Radargram, events: ZeroEvents) -> ZeroResult:
    return ZeroResult(
        t_zero_ns=None,
        k_zero=None,
        zero_rule=ZeroRule.FAILED,
        zero_event_label=None,
        pick_mode=events.pick_mode,
        events_ns=(),
        h_est_m=None,
        t_zero_per_trace_ns=None,
        meta_t0=classify_meta_t0(radargram.t0_candidates_ns, ZeroRule.FAILED, ()),
        excluded_traces=events.excluded_traces,
        flags=(*events.flags, "zero_failed"),
    )


def select_zero(
    radargram: Radargram, events: ZeroEvents, config: ZeroConfig = DEFAULT_ZERO_CONFIG
) -> ZeroResult:
    """Apply the dataset zero rule (spec_depth_v1 section 4-2).

    * ``amplitude_processed`` (GPR-SD, Guangzhou MALA): ``meta_sample0``.
    * ``air_gap="air"``: ``air_surface_late`` with two or more events, else
      ``air_single_event``.
    * ``air_gap="ground"``: ``ground_direct_peak`` (or ``ground_direct_shifted``
      inside the table's validity range).
    * ``air_gap="unknown"``: treated like ground, flagged ``air_gap_unknown``
      (and ``air_gap_unknown_multi_event`` with two or more events).

    No event at all gives ``zero_rule="zero_failed"``.
    """
    if radargram.amplitude_processed:
        return _meta_sample0_zero(radargram, events, config)
    if events.n_events == 0:
        return _failed_zero(radargram, events)

    choice = _choose_rule(radargram, events, config)
    window_std_ns, window_flags = _window_qc(events, choice.k_event, config)
    flags = [*events.flags, *choice.flags, *window_flags]
    dt_ns = radargram.dt_ns
    if dt_ns is None:
        flags.append("zero_px_unverified")

    return ZeroResult(
        t_zero_ns=None if dt_ns is None else choice.k_zero * dt_ns,
        k_zero=choice.k_zero,
        zero_rule=choice.rule,
        zero_event_label=choice.label,
        pick_mode=events.pick_mode,
        events_ns=events.events_ns,
        h_est_m=events.h_est_m if choice.rule == ZeroRule.AIR_SURFACE_LATE else None,
        t_zero_per_trace_ns=None,
        meta_t0=classify_meta_t0(
            radargram.t0_candidates_ns,
            choice.rule,
            events.events_ns,
            radargram.f_c_hz,
            config.meta_tolerance_ns,
        ),
        excluded_traces=events.excluded_traces,
        flags=tuple(flags),
        sigma_zero_ns=SIGMA_ZERO_NS[choice.rule],
        bias_notes=choice.bias_notes,
        window_std_ns=window_std_ns,
    )


def find_time_zero(
    radargram: Radargram,
    filtered: np.ndarray,
    *,
    pick_mode: PickMode = "envelope",
    exclude_traces: np.ndarray | None = None,
    config: ZeroConfig = DEFAULT_ZERO_CONFIG,
) -> ZeroResult:
    """Detect the early events and select the zero in one call.

    Args:
        radargram: Profile (scale, coupling, metadata zeros).
        filtered: Its data after steps (1)-(3), before background removal.
        pick_mode: Phase reference shared with the apex picks.
        exclude_traces: Traces left out of the mean (``pipe_band`` boxes).
        config: Detection settings.
    """
    events = detect_zero_events(
        filtered,
        radargram.dt_ns,
        radargram.f_c_hz,
        pick_mode=pick_mode,
        antenna_offset_m=radargram.antenna_offset_m,
        exclude_traces=exclude_traces,
        config=config,
    )
    return select_zero(radargram, events, config)


def given_zero(
    t_zero_ns: float,
    dt_ns: float,
    *,
    pick_mode: PickMode = "envelope",
    sigma_zero_ns: float = 0.0,
) -> ZeroResult:
    """Zero supplied by the caller, e.g. a known transmit time in a test.

    The result has ``zero_rule="given"`` and label ``"meta"``.
    """
    check_pick_mode(pick_mode)
    return ZeroResult(
        t_zero_ns=t_zero_ns,
        k_zero=t_zero_ns / dt_ns,
        zero_rule=ZeroRule.GIVEN,
        zero_event_label="meta",
        pick_mode=pick_mode,
        events_ns=(),
        h_est_m=None,
        t_zero_per_trace_ns=None,
        meta_t0={},
        excluded_traces=0,
        flags=("zero_given",),
        sigma_zero_ns=sigma_zero_ns,
    )
