"""Time-preserving filters for the fitting branch and gain for the detection branch.

Order (spec_depth_v1 section 3)::

    raw -> (1) DC -> (2) dewow -> (3) bandpass -+-> time-zero detection (no bg removal)
                                                +-> (4) background removal -> picks/fit
                                                +-> (4) -> (5) gain -> detector images

Every filter returns a new float64 array with time along axis 0
(``[n_samples, n_traces]``). Gain (:func:`agc`) moves envelope peaks along its
slope, so it is for the detection branch only and is never applied by
:func:`prepare_fitting_branch`.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from gpr_detection.io.radargram import Radargram

DEWOW_SIGMA_PERIODS = 1.0
"""Gaussian dewow ``sigma = 1 / f_c``; its -3 dB point is 0.25 f_c (band-pass low)."""

DEWOW_DEFAULT_SIGMA_NS = 2.0
"""Dewow sigma when ``f_c`` is unknown (equals 1 / 500 MHz, assumed)."""

DEWOW_TRUNCATE_SIGMAS = 4.0
"""Gaussian kernel half-width in sigmas."""

BANDPASS_ORDER = 4
BANDPASS_LOW_FRACTION = 0.25
"""Low cut-off as a fraction of ``f_c``."""

BANDPASS_HIGH_FRACTION = 2.0
"""High cut-off as a multiple of ``f_c`` (capped below Nyquist)."""

BANDPASS_NYQUIST_CAP = 0.45
"""High cut-off cap ``0.45 / dt``, i.e. 90 % of the Nyquist frequency."""

BACKGROUND_WINDOW_TRACES = 301
"""Moving-median window (6 m at 0.02 m); much wider than a hyperbola (~100 traces)."""

AGC_WINDOW_PERIODS = 8.0
"""AGC RMS window ``8 / f_c`` (assumed value, spec_depth_v1 section 14)."""

AGC_DEFAULT_WINDOW_NS = 16.0
"""AGC window when ``f_c`` is unknown (equals 8 / 500 MHz, assumed)."""

PIXEL_PERIOD_RANGE_PX = (4.0, 128.0)
"""No-scale input: the pixel band-pass runs only for dominant periods in this range."""

_HZ_PER_GHZ = 1e9


@dataclass(frozen=True)
class FittingBranch:
    """Fitting-branch sections of one radargram (spec_depth_v1 section 3).

    Attributes:
        filtered: After steps (1)-(3). Input of time-zero detection, which must
            run before background removal (air-coupled surface reflection is flat).
        background_removed: After step (4). Input of hyperbola picks.
        steps: Names of the steps that actually ran, in order, e.g.
            ``("dc", "dewow", "bandpass", "background_median")``.
        flags: e.g. ``bandpass_px_skipped``.
    """

    filtered: np.ndarray
    background_removed: np.ndarray
    steps: tuple[str, ...]
    flags: tuple[str, ...] = ()


def _period_ns(f_c_hz: float) -> float:
    return _HZ_PER_GHZ / f_c_hz


def _odd_window(length_ns: float, dt_ns: float) -> int:
    samples = max(1, round(length_ns / dt_ns))
    return samples if samples % 2 == 1 else samples + 1


def remove_dc(data: np.ndarray) -> np.ndarray:
    """Step (1): subtract each trace's mean over the whole record."""
    values = np.asarray(data, dtype=np.float64)
    return values - values.mean(axis=0, keepdims=True)


def dewow_sigma_samples(dt_ns: float, f_c_hz: float | None) -> float:
    """Gaussian dewow sigma in samples: ``1 / f_c`` (2 ns if ``f_c`` is unknown)."""
    if f_c_hz is None:
        sigma_ns = DEWOW_DEFAULT_SIGMA_NS
    else:
        sigma_ns = DEWOW_SIGMA_PERIODS * _period_ns(f_c_hz)
    return sigma_ns / dt_ns


def dewow(data: np.ndarray, dt_ns: float, f_c_hz: float | None) -> np.ndarray:
    """Step (2): subtract a Gaussian moving average along time (spec v1.1).

    The symmetric kernel has zero phase delay, so peak times do not move. A box
    window would leave ghost lobes at +-T_w/2 that the air-coupled ``late``
    zero rule can mistake for the surface reflection.

    Args:
        data: ``[n_samples, n_traces]`` amplitudes.
        dt_ns: Sample interval in ns.
        f_c_hz: Antenna centre frequency; ``None`` uses sigma 2 ns.
    """
    values = np.asarray(data, dtype=np.float64)
    trend = gaussian_filter1d(
        values,
        sigma=dewow_sigma_samples(dt_ns, f_c_hz),
        axis=0,
        mode="reflect",
        truncate=DEWOW_TRUNCATE_SIGMAS,
    )
    return values - trend


def bandpass_corners_hz(dt_ns: float, f_c_hz: float) -> tuple[float, float]:
    """Pass band ``[0.25 f_c, min(2 f_c, 0.45 / dt)]`` in Hz."""
    low_hz = BANDPASS_LOW_FRACTION * f_c_hz
    high_hz = min(
        BANDPASS_HIGH_FRACTION * f_c_hz, BANDPASS_NYQUIST_CAP / dt_ns * _HZ_PER_GHZ
    )
    return low_hz, high_hz


def bandpass(data: np.ndarray, dt_ns: float, f_c_hz: float) -> np.ndarray:
    """Step (3): zero-phase 4th-order Butterworth band-pass along time.

    ``sosfiltfilt`` runs the filter forwards and backwards, so the zero and the
    apex are shifted identically (by nothing).

    Raises:
        ValueError: If the sample interval is too coarse for the pass band.
    """
    low_hz, high_hz = bandpass_corners_hz(dt_ns, f_c_hz)
    if not high_hz > low_hz:
        raise ValueError(
            f"empty pass band {low_hz:.3g}-{high_hz:.3g} Hz for dt {dt_ns} ns"
        )
    sampling_hz = _HZ_PER_GHZ / dt_ns
    sos = butter(
        BANDPASS_ORDER,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sampling_hz,
        output="sos",
    )
    return sosfiltfilt(sos, np.asarray(data, dtype=np.float64), axis=0)


def estimate_period_px(data: np.ndarray) -> float | None:
    """Dominant period in pixels (samples) of DC-removed data (spec v1.3 section 3).

    The trace-averaged ``|rfft|`` along axis 0, without the DC bin, peaks at
    ``f_px`` cycles/px; the period is ``1 / f_px``.

    Returns:
        The period, or ``None`` when it lies outside 4-128 px or the data are flat.
    """
    values = np.asarray(data, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(values, axis=0)).mean(axis=1)
    frequencies = np.fft.rfftfreq(values.shape[0], d=1.0)
    if len(spectrum) < 2 or not spectrum[1:].max() > 0:
        return None
    f_px = float(frequencies[1 + int(np.argmax(spectrum[1:]))])
    period_px = 1.0 / f_px
    low_px, high_px = PIXEL_PERIOD_RANGE_PX
    if not low_px <= period_px <= high_px:
        return None
    return period_px


def bandpass_px(data: np.ndarray, period_px: float) -> np.ndarray:
    """No-scale step (3): band-pass ``[0.25 f_px, min(2 f_px, 0.45)]`` cycles/px.

    Same 4th-order Butterworth ``sosfiltfilt`` as :func:`bandpass`, with the
    sample index as time (``fs = 1``) and ``f_px = 1 / period_px`` cycles/px.
    """
    f_px = 1.0 / period_px
    low = BANDPASS_LOW_FRACTION * f_px
    high = min(BANDPASS_HIGH_FRACTION * f_px, BANDPASS_NYQUIST_CAP)
    sos = butter(BANDPASS_ORDER, [low, high], btype="bandpass", fs=1.0, output="sos")
    return sosfiltfilt(sos, np.asarray(data, dtype=np.float64), axis=0)


def remove_background(
    data: np.ndarray, window_traces: int = BACKGROUND_WINDOW_TRACES
) -> np.ndarray:
    """Step (4): subtract a moving **median** trace.

    The median, unlike the mean, does not drag the apex later (spec_depth_v1
    section 12 table 2). Each trace uses the ``window_traces`` traces centred on
    it; near the profile ends the window is shifted inwards so it keeps its
    full width. A profile not longer than the window uses its global median.

    Args:
        data: ``[n_samples, n_traces]`` amplitudes.
        window_traces: Median window in traces. An even value is raised by one
            so the window is centred on its trace.
    """
    values = np.asarray(data, dtype=np.float64)
    n_traces = values.shape[1]
    if window_traces < 1:
        raise ValueError(f"window_traces must be >= 1, got {window_traces}")
    if window_traces % 2 == 0:
        window_traces += 1
    if n_traces <= window_traces:
        return values - np.median(values, axis=1, keepdims=True)

    # Row by row: SciPy's 1-D rank filter is ~30x faster than a (1, W) footprint
    # on 2-D input (0.2 s vs 6.5 s for 512 x 1500, W = 301) and gives the same values.
    background = np.stack(
        [median_filter(row, size=window_traces, mode="nearest") for row in values]
    )
    half = window_traces // 2
    # Edge traces: the inward-shifted window is the first / last full window,
    # instead of the edge-replicated padding median_filter would use.
    background[:, :half] = np.median(values[:, :window_traces], axis=1, keepdims=True)
    background[:, n_traces - half :] = np.median(
        values[:, n_traces - window_traces :], axis=1, keepdims=True
    )
    return values - background


def agc(data: np.ndarray, dt_ns: float, f_c_hz: float | None) -> np.ndarray:
    """Step (5), detection branch only: divide by a moving RMS along time.

    Never use the result for time zero, picks or fits: a time-varying gain
    shifts envelope peaks along the gain slope.

    Args:
        data: ``[n_samples, n_traces]`` amplitudes.
        dt_ns: Sample interval in ns.
        f_c_hz: Centre frequency; the window is ``8 / f_c`` (16 ns if unknown).
    """
    values = np.asarray(data, dtype=np.float64)
    if f_c_hz is None:
        window_ns = AGC_DEFAULT_WINDOW_NS
    else:
        window_ns = AGC_WINDOW_PERIODS * _period_ns(f_c_hz)
    window = _odd_window(window_ns, dt_ns)
    mean_power = uniform_filter1d(values**2, size=window, axis=0, mode="reflect")
    peak_power = float(mean_power.max())
    floor = peak_power * 1e-12 if peak_power > 0 else 1.0
    return values / np.sqrt(mean_power + floor)


def prepare_fitting_branch(
    radargram: Radargram, background_window: int = BACKGROUND_WINDOW_TRACES
) -> FittingBranch:
    """Run steps (1)-(4) of the fitting branch (no gain).

    * ``amplitude_processed=True`` (GPR-SD, Guangzhou MALA): nothing is redone;
      both sections are the stored amplitudes (step name ``"upstream"``).
    * ``dt_ns is None`` (no-scale JPG input, spec v1.3): DC, then a pixel
      band-pass around the dominant period (``bandpass_px``, skipped with flag
      ``bandpass_px_skipped`` outside 4-128 px), then background removal.
      Dewow needs a time axis and is not run.
    * ``f_c_hz is None``: band-pass is skipped, dewow uses sigma 2 ns.

    Args:
        radargram: Input profile.
        background_window: Median window in traces for step (4).
    """
    if radargram.amplitude_processed:
        stored = np.asarray(radargram.data, dtype=np.float64)
        return FittingBranch(stored, stored, ("upstream",))

    steps = ["dc"]
    flags: list[str] = []
    section = remove_dc(radargram.data)
    if radargram.dt_ns is None:
        period_px = estimate_period_px(section)
        if period_px is None:
            flags.append("bandpass_px_skipped")
        else:
            section = bandpass_px(section, period_px)
            steps.append("bandpass_px")
    else:
        section = dewow(section, radargram.dt_ns, radargram.f_c_hz)
        steps.append("dewow")
        if radargram.f_c_hz is not None:
            section = bandpass(section, radargram.dt_ns, radargram.f_c_hz)
            steps.append("bandpass")
    background_removed = remove_background(section, background_window)
    steps.append("background_median")
    return FittingBranch(section, background_removed, tuple(steps), tuple(flags))


def period_ns_or_default(f_c_hz: float | None, default_ns: float) -> float:
    """One period ``1 / f_c`` in ns, or ``default_ns`` when ``f_c`` is unknown."""
    if f_c_hz is None:
        return default_ns
    return _period_ns(f_c_hz)


def samples_for_ns(length_ns: float, dt_ns: float) -> int:
    """Number of whole samples covering ``length_ns`` (rounded up)."""
    return math.ceil(length_ns / dt_ns - 1e-9)
