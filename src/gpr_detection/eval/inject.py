"""Real-background injection of gprMax pipe responses (M3-2 S4, measurement M3).

Sources: plan ``docs/processing/m32_image_quality_plan.md`` v0.2.3 section 3-2,
EM review ``docs/processing/m32_em_review.md`` v1.2 sections 1-2 to 1-5 and
appendix A ([EMR]), cv review ``docs/experiments/m32_cv_review.md`` section 2-2
and appendix B ([CVR]).

One realization is one injected copy of a background line::

    PipeResponse --quality gate--> 2.5-D time weight -> window-end taper
        -> waveform replacement H(f) -> low-pass 0.8 Nyquist(line)
        -> band-limited resample on the line time axis (time alignment)
        -> aperture taper = ResponsePatch (unit field, built once per line and k)
    LineContext (t_z, P_line, ImageGridMap, saturation, sections, exclusions)
        + InjectionSpec rows -> zones (a)-(d), spacing (e') -> SNR scale
        -> InjectionResult

Time alignment: a synthetic time ``tau`` lands at line time
``t = tau + t_z - reference_time_ns``, so the truth apex is
``t_apex = t_z + (apex_time_ns - reference_time_ns)``.

Coordinates (ADR-004): truth boxes are L0 ``GridBox`` values read directly from
the injected grid cells. The only L1 quantity, the box size in periods for the
distribution stratum, uses the ``ImageGridMap`` forward factors on L0 lengths;
nothing here maps L1 back to L0 or rounds L1 values.

Import rule 8 (architecture README section 2-2): stdlib, numpy, scipy,
``gpr_detection.io`` and ``gpr_detection.preprocess`` only.
"""

import csv
import dataclasses
import hashlib
import math
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks, hilbert, peak_widths

from gpr_detection.io.radargram import GridBox, Radargram
from gpr_detection.preprocess.filters import (
    bandpass,
    dewow,
    prepare_fitting_branch,
    remove_dc,
)
from gpr_detection.preprocess.timezero import ZeroResult, find_time_zero
from gpr_detection.preprocess.to_image import (
    DEFAULT_DETECTION_IMAGE_CONFIG,
    DetectionImageConfig,
    ImageGridMap,
    LineTile,
    make_detection_image,
    measure_period_samples,
)

# ------------------------------------------------------------------- constants

MIRROR_DIFF_KEY = "mirror_diff"
"""``PipeResponse.quality`` key: max mirror difference / apex peak ([EMR] 1-6)."""

EARLY_ENERGY_KEY = "early_energy_ratio"
"""``PipeResponse.quality`` key: energy before apex - 1 P relative to the apex."""

MIRROR_DIFF_MAX = 1e-3
EARLY_ENERGY_MAX = 0.01

WAVEFORM_EPSILON = 0.03
"""Water level ``epsilon`` of the waveform replacement filter (plan 3-2)."""

SYNTHETIC_WINDOW_PERIODS = 3.0
"""``W_syn`` is the spectrum of the apex trace inside apex +- 3 P."""

K_MAIN = 1.3
"""Gabor FWHM ``k P_line`` of the main metric (field-like, [EMR] 1-3)."""

K_REPORT = 0.8
"""Ricker-like Gabor FWHM, report only."""

LOWPASS_NYQUIST_FRACTION = 0.8
PERIOD_CHECK_TOLERANCE = 0.05
"""Filtered response period must be within +-5 % of ``P_line``."""

SNR_LADDER_DB = (0.0, 6.0, 12.0, 18.0)
POLARITIES = (1, -1)

APERTURE_TAPER_M = 0.2
WINDOW_END_TAPER_PERIODS = 1.0

HALF_WIDTH_AMPLITUDE = 0.5
"""-6 dB: envelope ratio of the half width used for the SNR noise window."""

WING_AMPLITUDE = 0.1
"""-20 dB: envelope ratio of the truth box and of the candidate zone (d)."""

INT16_SATURATION = 32767
INT16_MIN, INT16_MAX = -32768, 32767

SATURATION_MARGIN_PERIODS = 2.0
BAND_MARGIN_PERIODS = 2.0
STRONG_LAYER_MARGIN_PERIODS = 1.0
STRONG_LAYER_FACTOR = 4.0
"""(b'): mean-trace envelope above 4 x its moving median (plan v0.2.3)."""

STRONG_LAYER_BASELINE_PERIODS = 3.0
"""(b'): the moving median spans +-3 P, so a 1 P reflector does not lift it."""

LONG_LAYER_PERIODS = 2.0
"""Diagnostic: strong layers with an envelope FWHM above 2 P are counted."""

LONG_LAYER_BASELINE_PERIODS = 20.0
"""Diagnostic baseline: moving median over +-10 P (window 20 P)."""

STRONG_LAYER_BASELINE_FLOOR = 1e-30
"""Keeps the (b') comparison defined where the moving median is exactly 0."""

BOX_MARGIN_PERIODS = 1.0
"""(c'): un-injected boxes grow by 1 P in traces (L1 isotropy) and in samples."""

MATCH_TRACE_MARGIN = 3
MATCH_SAMPLE_PERIODS = 0.5
"""Apex match range of the S5 matching rule: +-3 traces, +-0.5 ``P_line``."""

REALIZATION_GAP_PERIODS = 1.0
"""(e'): centres in one realization are >= W_-20 + 1 P apart ([CVR] B-4)."""

LINE_GAP_WIDTHS = 0.5
"""(e'): any two positions of a line are >= W_-20 / 2 apart (no near-duplicates)."""

MAX_REALIZATIONS_PER_LINE = 2

SQRT2_PERIODS_SOURCE_PEAK = math.sqrt(2.0)
"""gprMax Ricker transmit peak at ``sqrt(2) / f_c`` ([EMR] appendix A, L1)."""

NS_PER_S = 1e9

AIHUB_WIDTH_PERIODS = (7.79, 14.67)
AIHUB_HEIGHT_PERIODS = (5.22, 8.35)
"""p5-p95 of AIHub ``pipe_hyperbola`` box sizes in periods ([CVR] 2-1).

The AIHub period constant ``P_A`` is 4.06 px; the ranges are already in P.
Since plan v0.2.3 (amendment 5) only the width decides the stratum; the
height range is reported (four-way reasons) and never filters.
"""

AIHUB_WIDTH_MEDIAN_PERIODS = 11.36
"""Median AIHub ``pipe_hyperbola`` label width in P (choice of ``l_W``)."""

LEVEL_AMPLITUDES: dict[float, float] = {
    -6.0: 0.5,
    -10.0: 0.316,
    -12.0: 0.251,
    -20.0: 0.1,
}
"""Envelope levels (dB -> amplitude ratio) of the stratum boxes (plan 3-2)."""

VISIBLE_FLOOR_AMPLITUDE = 0.1
VISIBLE_ZERO_DB_AMPLITUDE = 0.5
"""IoU-report box: ``max(10^(-s/20), 0.1)`` x apex envelope; -6 dB at 0 dB."""

CLIPPED_INJECTION_RATIO_MAX = 0.05
"""int16 rule (ii): a clipped cell may carry at most 5 % of the apex injection."""

BOTTOM_STRATUM_PERIODS = 3.0
BOTTOM_PRIORITY_PER_LINE = 1
"""Bottom-stratum candidates placed first per line (amendment 6)."""

FIRST_REFLECTION_LEAD_PERIODS = 2.0
"""The period check region starts 2 ``P_line`` before the apex (amendment 6)."""
PLAN_SEED = 20261114
MAX_INJECTIONS_PER_LINE = 15
"""Per line and SNR step; positions are shared by all SNR steps."""

GRID_TOLERANCE = 1e-6
"""Apex trace offsets closer than this to an integer count as on the grid."""

INJECTION_TABLE_COLUMNS = (
    "injection_id",
    "realization_id",
    "line_id",
    "apex_trace",
    "response_id",
    "snr_db",
    "polarity",
    "k",
    "seed",
)

Coupling = Literal["air", "ground"]
_COUPLINGS: tuple[str, ...] = ("air", "ground")


class RejectReason(StrEnum):
    """Why a response or an injection was not used."""

    QUALITY_MISSING = "quality_missing"
    MIRROR_DIFF = "mirror_diff_above_1e-3"
    EARLY_ENERGY = "early_energy_above_1pct"
    COUPLING_MISMATCH = "coupling_mismatch"
    ZERO_FAILED = "zero_failed"
    PERIOD_CHECK = "period_check_failed"
    APEX_OUTSIDE_RECORD = "apex_outside_record"
    BEYOND_SYNTHETIC_WINDOW = "beyond_synthetic_window"
    NO_TRUTH_CELLS = "no_truth_cells"
    UNKNOWN_RESPONSE = "unknown_response"
    OFF_GRID_APEX = "apex_trace_off_grid"
    APERTURE_OUTSIDE_LINE = "aperture_outside_line"
    SATURATION_ZONE = "saturation_zone"
    BAND_ZONE = "band_zone"
    STRONG_LAYER_ZONE = "strong_layer_zone"
    UNINJECTED_BOX_ZONE = "uninjected_box_zone"
    CANDIDATE_APEX_ZONE_K = "candidate_apex_zone_k"
    CANDIDATE_APEX_ZONE_AUTO = "candidate_apex_zone_auto"
    SPACING = "spacing_below_width_plus_1p"
    ZERO_NOISE = "zero_noise"
    INT16_OVERFLOW = "int16_overflow"
    """Only for clipped cells in the truth box / SNR window or above 5 %."""


class WidthClass(StrEnum):
    """Stratum of the ``l_W`` box width (plan 3-2 "분포 층", amendment 5)."""

    IN = "in"
    WIDTH_OVER = "width_over"
    WIDTH_UNDER = "width_under"
    WIDTH_OVER_LOWER_BOUND = "width_over_lower_bound"
    """The box touches the aperture taper, so its width is a lower bound."""


class HeightClass(StrEnum):
    """Report-only height reason against the AIHub range 5.22-8.35 P."""

    IN = "in"
    HEIGHT_OVER = "height_over"
    HEIGHT_UNDER = "height_under"


# ---------------------------------------------------------------- data types


@dataclass(frozen=True, eq=False)
class PipeResponse:
    """Pipe-only gprMax response ``d = u_full - u_bg`` prepared by an analysis script.

    Attributes:
        response_id: Model id, e.g. ``"M12"`` or ``"K7_P1_D060"``.
        data: float64 ``[n_t, n_x]`` pipe-only field (full minus ``bg[:, [0]]``).
        dt_ns: Synthetic sample interval.
        dx_m: Synthetic trace interval.
        apex_trace: Mirror axis in traces (may be fractional).
        apex_time_ns: Envelope peak of the apex trace after steps (1)-(3).
        reference_time_ns: Air: envelope peak of the ``_bg`` band after (1)-(3);
            ground: direct-wave peak. Aligned to the line zero ``t_z``.
        coupling: ``"air"`` or ``"ground"``.
        air_gap_m: Antenna height of the model (0 for ground-coupled).
        eps_r: Soil permittivity of the model.
        f_c_hz: Source centre frequency.
        top_depth_m: Pipe top depth.
        od_m: Pipe outer diameter.
        material: Pipe material, e.g. ``"pec"``.
        quality: ``MIRROR_DIFF_KEY`` and ``EARLY_ENERGY_KEY`` values measured by
            the analysis script; the thresholds are checked here.
        source_peak_ns: Transmit-pulse peak time for the 2.5-D weight
            ``sqrt(t_a / t)``; ``None`` uses the gprMax Ricker peak
            ``sqrt(2) / f_c`` ([EMR] appendix A, L1).
    """

    response_id: str
    data: np.ndarray
    dt_ns: float
    dx_m: float
    apex_trace: float
    apex_time_ns: float
    reference_time_ns: float
    coupling: Coupling
    air_gap_m: float
    eps_r: float
    f_c_hz: float
    top_depth_m: float
    od_m: float
    material: str
    quality: Mapping[str, float] = field(default_factory=dict)
    source_peak_ns: float | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != 2 or self.data.dtype != np.float64:
            raise ValueError(
                f"{self.response_id}: data must be 2-D float64, got "
                f"{self.data.dtype} {self.data.shape}"
            )
        if not (self.dt_ns > 0 and self.dx_m > 0):
            raise ValueError(f"{self.response_id}: dt_ns and dx_m must be positive")
        if self.coupling not in _COUPLINGS:
            raise ValueError(f"{self.response_id}: coupling must be air or ground")
        if not 0 <= self.apex_trace <= self.data.shape[1] - 1:
            raise ValueError(f"{self.response_id}: apex_trace outside the aperture")
        if not self.apex_time_ns > self.time_origin_ns:
            raise ValueError(
                f"{self.response_id}: apex_time_ns must follow the time origin"
            )

    @property
    def time_origin_ns(self) -> float:
        """Transmit peak: ``t = 0`` of the 2.5-D weight (travel time from the source).

        ``reference_time_ns`` is not used: the band (air) and the direct wave
        (ground) arrive after the transmit peak.
        """
        if self.source_peak_ns is None:
            return SQRT2_PERIODS_SOURCE_PEAK / self.f_c_hz * NS_PER_S
        return self.source_peak_ns

    @property
    def n_traces(self) -> int:
        """Aperture width in traces."""
        return self.data.shape[1]

    @property
    def window_ns(self) -> float:
        """Time of the last synthetic sample."""
        return (self.data.shape[0] - 1) * self.dt_ns


@dataclass(frozen=True)
class InjectionSpec:
    """One row of the injection table (plan 3-2, pairing)."""

    injection_id: str
    realization_id: str
    line_id: str
    apex_trace: float
    response_id: str
    snr_db: float
    polarity: int
    k: float
    seed: int

    def __post_init__(self) -> None:
        if self.polarity not in POLARITIES:
            raise ValueError(f"{self.injection_id}: polarity must be +1 or -1")
        if not self.k > 0:
            raise ValueError(f"{self.injection_id}: k must be positive")


@dataclass(frozen=True)
class InjectionCondition:
    """SNR, polarity and Gabor width shared by every injection of a realization."""

    snr_db: float
    polarity: int
    k: float


@dataclass(frozen=True)
class LineExclusions:
    """L0 exclusion inputs prepared by the analysis script.

    Attributes:
        uninjected_boxes: Union of the un-injected run boxes (all candidate
            recipes x 5 folds, any class), zone (c').
        k_apex_traces: The 41 K-bundle apexes (M6 guard), zone (d) within the
            -20 dB half width.
        auto_candidate_apex_traces: ``fourtu_apex_candidates_v1.csv`` apexes,
            zone (d) within the -6 dB half width + 1 P (amendment 5).
    """

    uninjected_boxes: tuple[GridBox, ...] = ()
    k_apex_traces: tuple[float, ...] = ()
    auto_candidate_apex_traces: tuple[float, ...] = ()


NO_EXCLUSIONS = LineExclusions()


@dataclass(frozen=True, eq=False)
class LineContext:
    """Everything about one background line that injection needs.

    Attributes:
        background: The raw line (not modified).
        zero: ``find_time_zero`` result (``select_zero`` default rule).
        period_samples: ``P_line`` of the line detection image.
        grid_map: Line L1 -> L0 map of the line detection image.
        filtered: Background after steps (1)-(3).
        background_removed: Background after steps (1)-(4) (median 301).
        last_saturated_sample: Last row with ``|D| >= 32767``; ``None`` if none.
        exclusions: Zones (c') and (d).
        image_config: Config used for ``P_line`` and the period check.
        tiles: Tiles of the line detection image (inference cost table).
        tile_stride_px: Tile stride ``L - O`` in line L1 px.
    """

    background: Radargram
    zero: ZeroResult
    period_samples: float
    grid_map: ImageGridMap
    filtered: np.ndarray
    background_removed: np.ndarray
    last_saturated_sample: int | None
    exclusions: LineExclusions
    image_config: DetectionImageConfig
    tiles: tuple[LineTile, ...] = ()
    tile_stride_px: int = 0

    @property
    def dt_ns(self) -> float:
        """Line sample interval."""
        dt_ns = self.background.dt_ns
        assert dt_ns is not None  # checked by prepare_line_context
        return dt_ns

    @property
    def dx_m(self) -> float:
        """Line trace interval."""
        dx_m = self.background.dx_m
        assert dx_m is not None  # checked by prepare_line_context
        return dx_m

    @property
    def period_ns(self) -> float:
        """``P_line`` in ns."""
        return self.period_samples * self.dt_ns

    @property
    def period_traces(self) -> float:
        """One period as a horizontal length in traces (L1 is depth-isotropic)."""
        period_l1_px = self.period_samples / self.grid_map.samples_per_px
        return period_l1_px * self.grid_map.traces_per_px


@dataclass(frozen=True)
class LevelBox:
    """Apex-seeded envelope component of the untapered unit field at one level.

    Attributes:
        bbox: ``(col0, col1, row0, row1)`` in patch cells.
        width_periods: L1 width in L1 periods (forward ``ImageGridMap``).
        height_periods: L1 height in L1 periods.
        aperture_limited: The component touches an aperture-taper column, so
            the width is a lower bound ([EMR] appendix B-1 (a)).
    """

    bbox: tuple[int, int, int, int]
    width_periods: float
    height_periods: float
    aperture_limited: bool


@dataclass(frozen=True, eq=False)
class ResponsePatch:
    """Unit pipe response of one model on one line's time axis.

    Patch column ``j`` is synthetic trace ``j``; row ``r`` is line sample ``r``
    (rows may run past the record end so the wing box is not cut).

    Attributes:
        response: Source response.
        k: Gabor FWHM in periods.
        field: Injected field before SNR scaling, both tapers applied.
        signal_amplitude: ``A``: apex envelope peak of ``field`` after (1)-(3).
        apex_sample: Truth apex row ``t_apex / dt``.
        half_width_6db: -6 dB half width in traces (SNR noise window).
        half_width_20db: -20 dB half width in traces (zones (d), (e')).
        core_bbox: ``(col0, col1, row0, row1)`` of the -6 dB component of
            ``field`` (zone (c')).
        wing_bbox: ``(col0, col1, row0, row1)`` of the -20 dB component before
            any taper (distribution stratum).
        truth_bbox: The same component without the taper cells (truth box).
        period_samples: ``measure_period_samples`` of ``field`` inside the
            first-reflection region (plan 3-2, amendment 5).
        effective_eps_r: ``eps_r`` after re-labelling the trace spacing. The
            re-labelling is exact for kinematics only: the Tx-Rx offset and the
            wing amplitude-angle relation scale too, and an air gap does not,
            so air-coupled lines with ``dx`` ratio != 1 need a new check
            ([EMR] appendix A, L3).
        period_ringing: The period check failed on a water-filled PVC pipe;
            the patch is kept and flagged (in-pipe water resonance).
        level_boxes: ``LEVEL_AMPLITUDES`` level (dB) -> ``LevelBox``.
        envelope: Envelope of ``field`` after (1)-(3) (IoU-report box).
        n_aperture_taper: Aperture-taper columns at each end.
    """

    response: PipeResponse
    k: float
    field: np.ndarray
    signal_amplitude: float
    apex_sample: float
    half_width_6db: float
    half_width_20db: float
    core_bbox: tuple[int, int, int, int]
    wing_bbox: tuple[int, int, int, int]
    truth_bbox: tuple[int, int, int, int]
    period_samples: float
    effective_eps_r: float
    period_ringing: bool = False
    # ``field`` is an attribute name here, so the factory is spelled in full.
    level_boxes: Mapping[float, LevelBox] = dataclasses.field(default_factory=dict)
    envelope: np.ndarray | None = None
    n_aperture_taper: int = 0

    @property
    def expected_width(self) -> float:
        """-20 dB full width ``W_-20`` in traces."""
        return 2.0 * self.half_width_20db


@dataclass(frozen=True)
class InjectionTruth:
    """Truth of one accepted injection, all in L0.

    Attributes:
        injection_id: Table id.
        response_id: Model id.
        snr_db: SNR step.
        polarity: +1 or -1.
        k: Gabor FWHM in periods.
        box: -20 dB box without the taper cells, clipped to the record.
        wing_box: -20 dB box before tapering (may run past the record end).
        core_box: -6 dB component box, clipped to the record. With the apex
            this lets S5 apply the one-to-one match rule.
        apex_trace: Apex trace (L0, fractional).
        apex_sample: Apex sample (L0, fractional).
        width_periods: L1 width of ``wing_box`` in L1 periods.
        height_periods: L1 height of ``wing_box`` in L1 periods.
        bottom_stratum: Apex within 3 ``P_line`` of the record end.
        gain: Factor applied to the unit field (polarity included).
        signal_amplitude: ``A`` of the unit field.
        noise_rms: ``sigma`` of the background.
        effective_eps_r: See ``ResponsePatch``.
        box_clipped: True when ``box`` was cut at the record end.
        in_horizontal_box_zone: The apex would have been blocked by the old
            horizontal-only zone (c); reported separately ([CVR] B-1).
        period_ringing: See ``ResponsePatch``.
        n_clipped_cells: Cells of the sum cut to the int16 range.
        max_clipped_injection_ratio: Largest ``|injection| / (|gain| A)`` of a
            clipped cell (0 when nothing was cut).
        width_periods_by_level: Level (dB) -> L1 width in P of the stratum box.
        height_periods_by_level: Level (dB) -> L1 height in P.
        aperture_limited_by_level: Level (dB) -> the width is a lower bound.
        visible_box: IoU-report box at ``max(10^(-s/20), 0.1)`` of the apex
            envelope (-6 dB at 0 dB), clipped to the record; not a truth,
            stratum or zone box.
    """

    injection_id: str
    response_id: str
    snr_db: float
    polarity: int
    k: float
    box: GridBox
    wing_box: GridBox
    core_box: GridBox
    apex_trace: float
    apex_sample: float
    width_periods: float
    height_periods: float
    bottom_stratum: bool
    gain: float
    signal_amplitude: float
    noise_rms: float
    effective_eps_r: float
    box_clipped: bool
    in_horizontal_box_zone: bool
    visible_box: GridBox
    period_ringing: bool = False
    n_clipped_cells: int = 0
    max_clipped_injection_ratio: float = 0.0
    width_periods_by_level: Mapping[float, float] = field(
        default_factory=dict, hash=False
    )
    height_periods_by_level: Mapping[float, float] = field(
        default_factory=dict, hash=False
    )
    aperture_limited_by_level: Mapping[float, bool] = field(
        default_factory=dict, hash=False
    )

    def width_class(self, level_w_db: float) -> "WidthClass":
        """Stratum reason of the ``level_w_db`` box (no default level)."""
        return width_class(
            self.width_periods_by_level[level_w_db],
            self.aperture_limited_by_level[level_w_db],
        )

    def in_distribution(self, level_w_db: float) -> bool:
        """Width stratum of plan 3-2 at the fixed level ``l_W``."""
        return self.width_class(level_w_db) is WidthClass.IN


@dataclass(frozen=True, eq=False)
class InjectionResult:
    """One injected copy of a background line.

    Attributes:
        realization_id: Shared by every spec of the call.
        line_id: Background line.
        data: float64 copy of the raw line plus the accepted injections.
        truths: Accepted injections in table order.
        rejected: ``(injection_id, reason)`` of the others.
    """

    realization_id: str
    line_id: str
    data: np.ndarray
    truths: tuple[InjectionTruth, ...]
    rejected: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PlannedPosition:
    """One planned apex: realization group, model and L0 apex trace."""

    group: int
    response_id: str
    apex_trace: float


@dataclass(frozen=True)
class LinePlan:
    """Planned positions of one line and what the plan had to leave out.

    Attributes:
        line_id: Background line.
        positions: Chosen positions, at most ``MAX_REALIZATIONS_PER_LINE``
            groups (realizations).
        n_candidates: Allowed (model, apex trace) pairs before spacing.
        n_dropped: Candidates that kept the line spacing ``W_-20 / 2`` but fit
            in no realization (``[CVR]`` B-4: dropped and counted).
        candidates_by_response: ``response_id -> (bottom, all)`` allowed
            candidates before any priority (natural bottom share, [CVR] D-3).
    """

    line_id: str
    positions: tuple[PlannedPosition, ...]
    n_candidates: int
    n_dropped: int
    candidates_by_response: Mapping[str, tuple[int, int]] = field(
        default_factory=dict, hash=False
    )

    @property
    def n_realizations(self) -> int:
        """Number of position groups."""
        return len({position.group for position in self.positions})


@dataclass(frozen=True)
class LineCost:
    """P5b work of one line per recipe and fold (``[CVR]`` B-3).

    Attributes:
        line_id: Background line.
        n_positions: Planned positions.
        n_realizations: Injected copies per condition.
        n_conditions: Conditions (SNR x polarity x k).
        l1_builds: ``n_realizations x n_conditions`` line L1 images.
        inferred_tiles: Tiles touching an injection, summed over realizations,
            times ``n_conditions``.
        long_layer_runs: Diagnostic for gpr-processing: strong layers with an
            envelope FWHM above 2 P on the mean trace of the whole line
            (``long_strong_layer_runs``); S5 reports it.
    """

    line_id: str
    n_positions: int
    n_realizations: int
    n_conditions: int
    l1_builds: int
    inferred_tiles: int
    long_layer_runs: int


# ------------------------------------------------------------ signal helpers


def _envelope(section: np.ndarray) -> np.ndarray:
    """Hilbert envelope along time, zero padded to 2 N against wrap-around.

    ``hilbert`` uses an unpadded FFT, so energy at the last rows of an
    untapered field would leak into the first rows ([EMR] appendix A, L2).
    """
    n_samples = section.shape[0]
    analytic = hilbert(section, N=2 * n_samples, axis=0)
    return np.abs(analytic[:n_samples])


def _linear_steps(section: np.ndarray, background: Radargram) -> np.ndarray:
    """Steps (1)-(3) as ``prepare_fitting_branch`` runs them on ``background``.

    Repeated here because ``prepare_fitting_branch`` takes a float32
    ``Radargram``; the injected field is float64.
    """
    values = np.asarray(section, dtype=np.float64)
    if background.amplitude_processed:
        return values
    dt_ns = background.dt_ns
    assert dt_ns is not None
    values = dewow(remove_dc(values), dt_ns, background.f_c_hz)
    if background.f_c_hz is not None:
        values = bandpass(values, dt_ns, background.f_c_hz)
    return values


def edge_taper_weights(n_samples: int, n_taper: int) -> np.ndarray:
    """Cosine taper rising from near 0 over ``n_taper`` cells at both ends.

    Cell ``j < n_taper`` gets ``0.5 (1 - cos(pi (j + 0.5) / n_taper))``.
    """
    weights = np.ones(n_samples)
    n_taper = min(n_taper, n_samples // 2)
    if n_taper <= 0:
        return weights
    rising = 0.5 * (1.0 - np.cos(np.pi * (np.arange(n_taper) + 0.5) / n_taper))
    weights[:n_taper] = rising
    weights[n_samples - n_taper :] = rising[::-1]
    return weights


def _window_end_weights(n_samples: int, n_taper: int) -> np.ndarray:
    weights = np.ones(n_samples)
    n_taper = min(n_taper, n_samples)
    if n_taper > 0:
        falling = 0.5 * (1.0 + np.cos(np.pi * (np.arange(n_taper) + 0.5) / n_taper))
        weights[n_samples - n_taper :] = falling
    return weights


def gabor_magnitude(
    frequencies_ghz: np.ndarray, period_ns: float, fwhm_ns: float
) -> np.ndarray:
    """``|W_tgt(f)|`` of a Gabor wavelet with carrier ``1 / P`` (peak 1).

    The envelope ``exp(-t^2 / (2 s^2))`` has FWHM ``2 sqrt(2 ln 2) s``; its
    spectrum is a Gaussian of standard deviation ``1 / (2 pi s)`` around
    ``+-1 / P``.
    """
    sigma_ns = fwhm_ns / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    carrier_ghz = 1.0 / period_ns
    spread = 2.0 * (math.pi * sigma_ns) ** 2
    return np.exp(-spread * (frequencies_ghz - carrier_ghz) ** 2) + np.exp(
        -spread * (frequencies_ghz + carrier_ghz) ** 2
    )


def waveform_replacement_filter(
    apex_trace_values: np.ndarray,
    dt_ns: float,
    apex_time_ns: float,
    period_ns: float,
    k: float,
    n_fft: int,
) -> np.ndarray:
    """Zero-phase ``H(f) = |W_tgt| / (|W_syn| + eps max |W_syn|)`` on rfft bins.

    Args:
        apex_trace_values: Synthetic apex trace.
        dt_ns: Its sample interval.
        apex_time_ns: Apex time; ``W_syn`` uses apex +- 3 ``period_ns``.
        period_ns: ``P_line`` (Gabor carrier ``1 / P_line``).
        k: Gabor envelope FWHM in periods.
        n_fft: FFT length of the filtered section.
    """
    times_ns = np.arange(len(apex_trace_values)) * dt_ns
    half_window_ns = SYNTHETIC_WINDOW_PERIODS * period_ns
    inside = np.abs(times_ns - apex_time_ns) <= half_window_ns
    windowed = np.where(inside, apex_trace_values, 0.0)
    synthetic = np.abs(np.fft.rfft(windowed, n_fft))
    frequencies = np.fft.rfftfreq(n_fft, d=dt_ns)
    target = gabor_magnitude(frequencies, period_ns, k * period_ns)
    return target / (synthetic + WAVEFORM_EPSILON * synthetic.max())


def period_check_passes(measured_samples: float | None, line_period: float) -> bool:
    """True when the filtered period is within +-5 % of ``P_line``."""
    if measured_samples is None:
        return False
    return abs(measured_samples / line_period - 1.0) <= PERIOD_CHECK_TOLERANCE


def _band_limited_resample(
    spectrum: np.ndarray,
    frequencies_ghz: np.ndarray,
    n_fft: int,
    line_times_ns: np.ndarray,
    synthetic_window_ns: float,
) -> np.ndarray:
    """Evaluate the inverse rfft of ``spectrum`` at arbitrary synthetic times.

    This is FFT resampling with the sub-sample shift done as a phase shift:
    ``x(tau) = (1/N) sum_m c_m Re(X_m exp(i 2 pi f_m tau))`` with ``c_0 = 1``,
    ``c_m = 2``. Only bins with a non-zero spectrum enter; times outside the
    synthetic window give 0.
    """
    keep = np.flatnonzero(np.any(spectrum != 0, axis=1))
    output = np.zeros((len(line_times_ns), spectrum.shape[1]))
    inside = (line_times_ns >= 0) & (line_times_ns <= synthetic_window_ns)
    if keep.size == 0 or not inside.any():
        return output
    weights = np.where(keep == 0, 1.0, 2.0) / n_fft
    phases = np.exp(2j * np.pi * np.outer(line_times_ns[inside], frequencies_ghz[keep]))
    output[inside] = ((phases * weights) @ spectrum[keep]).real
    return output


# ----------------------------------------------------------- patch building


def response_quality_failure(response: PipeResponse) -> RejectReason | None:
    """[EMR] 1-6 gate: mirror difference <= 1e-3 and early energy <= 1 %."""
    mirror = response.quality.get(MIRROR_DIFF_KEY)
    early = response.quality.get(EARLY_ENERGY_KEY)
    if mirror is None or early is None:
        return RejectReason.QUALITY_MISSING
    if not mirror <= MIRROR_DIFF_MAX:
        return RejectReason.MIRROR_DIFF
    if not early <= EARLY_ENERGY_MAX:
        return RejectReason.EARLY_ENERGY
    return None


def _time_weighted(response: PipeResponse, apply_2p5d_weight: bool) -> np.ndarray:
    """2.5-D weight ``sqrt(t_a / t)`` after the apex, 1 before it ([EMR] 1-2).

    ``t`` counts from the transmit peak. Times before the apex keep weight 1
    (the quality gate leaves no pipe energy there) so the factor never blows
    up near the transmit peak.
    """
    if not apply_2p5d_weight:
        return response.data.copy()
    origin_ns = response.time_origin_ns
    apex_lag_ns = response.apex_time_ns - origin_ns
    times_ns = np.arange(response.data.shape[0]) * response.dt_ns
    lag_ns = np.maximum(times_ns - origin_ns, apex_lag_ns)
    return response.data * np.sqrt(apex_lag_ns / lag_ns)[:, None]


def _alignment_shift_ns(response: PipeResponse, context: LineContext) -> float:
    """Line time minus synthetic time (``t_z - reference_time_ns``)."""
    t_zero_ns = context.zero.t_zero_ns
    assert t_zero_ns is not None  # zero failures are rejected before this
    return t_zero_ns - response.reference_time_ns


def _patch_rows(response: PipeResponse, context: LineContext, shift_ns: float) -> int:
    """Line rows up to the synthetic window end (at least the record length)."""
    last_row = math.floor((response.window_ns + shift_ns) / context.dt_ns)
    return max(context.background.n_samples, last_row + 1)


def _line_field(
    section: np.ndarray,
    filter_gain: np.ndarray,
    response: PipeResponse,
    context: LineContext,
    n_rows: int,
) -> np.ndarray:
    """Waveform filter + low-pass + resample of a synthetic section (no taper)."""
    n_fft = 2 * (len(filter_gain) - 1)
    frequencies = np.fft.rfftfreq(n_fft, d=response.dt_ns)
    cutoff_ghz = LOWPASS_NYQUIST_FRACTION * 0.5 / context.dt_ns
    lowpass = (frequencies <= cutoff_ghz) & (np.arange(len(frequencies)) < n_fft // 2)
    spectrum = np.fft.rfft(section, n_fft, axis=0) * (filter_gain * lowpass)[:, None]
    shift_ns = _alignment_shift_ns(response, context)
    synthetic_times_ns = np.arange(n_rows) * context.dt_ns - shift_ns
    return _band_limited_resample(
        spectrum, frequencies, n_fft, synthetic_times_ns, response.window_ns
    )


def _seed_component(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """Cells of the 4-connected component of ``mask`` that contains ``seed``."""
    labels, _ = ndimage.label(mask)
    label = labels[seed]
    if label == 0:
        return np.zeros_like(mask)
    return labels == label


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """``(col0, col1, row0, row1)`` of the true cells, ``None`` if there are none."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())


def _half_width(bbox: tuple[int, int, int, int], apex_trace: float) -> float:
    col0, col1, _, _ = bbox
    return max(apex_trace - col0, col1 - apex_trace)


@dataclass(frozen=True)
class _ApexPeak:
    amplitude: float
    seed: tuple[int, int]


def _apex_peak(
    envelope: np.ndarray, apex_column: int, apex_sample: float, period: float
) -> _ApexPeak:
    """Envelope maximum of the apex column within apex +- 1 P."""
    row0 = max(0, math.floor(apex_sample - period))
    row1 = min(envelope.shape[0] - 1, math.ceil(apex_sample + period))
    window = envelope[row0 : row1 + 1, apex_column]
    peak_row = row0 + int(np.argmax(window))
    return _ApexPeak(float(window.max()), (peak_row, apex_column))


def _taper_free_cells(
    shape: tuple[int, int], n_aperture_taper: int, window_end_row: float
) -> np.ndarray:
    """Cells outside the aperture taper and before the window-end taper."""
    n_rows, n_cols = shape
    cells = np.zeros(shape, dtype=bool)
    last_row = min(n_rows - 1, math.floor(window_end_row))
    cells[: last_row + 1, n_aperture_taper : n_cols - n_aperture_taper] = True
    return cells


def _filtered_fields(
    response: PipeResponse, context: LineContext, k: float, apply_2p5d_weight: bool
) -> tuple[np.ndarray, np.ndarray, int]:
    """Tapered and untapered unit fields on the line grid, and the taper width."""
    weighted = _time_weighted(response, apply_2p5d_weight)
    n_fft = 1 << math.ceil(math.log2(2 * weighted.shape[0]))
    apex_column = round(response.apex_trace)
    filter_gain = waveform_replacement_filter(
        weighted[:, apex_column],
        response.dt_ns,
        response.apex_time_ns,
        context.period_ns,
        k,
        n_fft,
    )
    n_rows = _patch_rows(response, context, _alignment_shift_ns(response, context))
    untapered = _line_field(weighted, filter_gain, response, context, n_rows)
    n_end_taper = round(WINDOW_END_TAPER_PERIODS * context.period_ns / response.dt_ns)
    end_weights = _window_end_weights(weighted.shape[0], n_end_taper)
    n_aperture_taper = round(APERTURE_TAPER_M / context.dx_m)
    tapered = _line_field(
        weighted * end_weights[:, None], filter_gain, response, context, n_rows
    )
    tapered *= edge_taper_weights(response.n_traces, n_aperture_taper)[None, :]
    return tapered, untapered, n_aperture_taper


def _patch_failure(response: PipeResponse, context: LineContext) -> RejectReason | None:
    quality = response_quality_failure(response)
    if quality is not None:
        return quality
    if response.coupling != context.background.air_gap:
        return RejectReason.COUPLING_MISMATCH
    if context.zero.t_zero_ns is None:
        return RejectReason.ZERO_FAILED
    return None


def build_response_patch(
    response: PipeResponse,
    context: LineContext,
    k: float = K_MAIN,
    *,
    apply_2p5d_weight: bool = True,
) -> ResponsePatch | RejectReason:
    """Unit response of ``response`` on the time axis of ``context`` (plan 3-2).

    The period check runs on the first-reflection region only (amendment 5,
    [EMR] appendix B-2); a water-filled PVC pipe that fails it is kept with
    ``period_ringing`` set.

    Returns:
        The patch, or the reason the model cannot be used on this line
        (quality gate, coupling, failed zero, apex outside the record, wing
        past the synthetic window - 1 P, period check).
    """
    failure = _patch_failure(response, context)
    if failure is not None:
        return failure
    tapered, untapered, n_aperture_taper = _filtered_fields(
        response, context, k, apply_2p5d_weight
    )
    shift_ns = _alignment_shift_ns(response, context)
    apex_sample = (response.apex_time_ns + shift_ns) / context.dt_ns
    if not 0 <= apex_sample <= context.background.n_samples - 1:
        return RejectReason.APEX_OUTSIDE_RECORD
    # Synthetic window end - 1 P on the line axis: wings must end before it.
    window_end_row = (response.window_ns - context.period_ns + shift_ns) / context.dt_ns
    frame = _PatchFrame(
        round(response.apex_trace),
        apex_sample,
        context.period_samples,
        n_aperture_taper,
        window_end_row,
    )
    envelope = _envelope(_linear_steps(tapered, context.background))
    wing_envelope = _envelope(_linear_steps(untapered, context.background))
    geometry = _patch_geometry(envelope, wing_envelope, frame)
    if isinstance(geometry, RejectReason):
        return geometry
    measured, period_ringing = _first_reflection_period(
        tapered, geometry.wing_mask, response, context, apex_sample
    )
    if measured is None:
        return RejectReason.PERIOD_CHECK
    dx_ratio = context.dx_m / response.dx_m
    return ResponsePatch(
        response=response,
        k=k,
        field=tapered,
        signal_amplitude=geometry.signal_amplitude,
        apex_sample=apex_sample,
        half_width_6db=_half_width(geometry.half_bbox, response.apex_trace),
        half_width_20db=_half_width(geometry.wing_bbox, response.apex_trace),
        core_bbox=geometry.half_bbox,
        wing_bbox=geometry.wing_bbox,
        truth_bbox=geometry.truth_bbox,
        period_samples=measured,
        effective_eps_r=response.eps_r / dx_ratio**2,
        period_ringing=period_ringing,
        level_boxes=_level_boxes(wing_envelope, geometry.wing_seed, frame, context),
        envelope=envelope,
        n_aperture_taper=n_aperture_taper,
    )


def is_water_filled_pvc(material: str) -> bool:
    """PVC pipe filled with water, read from the json material field."""
    name = material.lower()
    return "pvc" in name and "water" in name


def first_reflection_mask(
    wing_mask: np.ndarray,
    response: PipeResponse,
    context: LineContext,
    apex_sample: float,
) -> np.ndarray:
    """Cells of the -20 dB component with apex - 2 P <= row <= apex + tau_m - 1 P.

    ``tau_m`` = apex time - transmit peak (pipe-surface two-way time): later
    cells hold pipe-surface multiples, 2-D tails and the window end, which
    pull the averaged spectrum ([EMR] appendix B-2). The lower bound
    (amendment 6) leaves out weak precursors more than 2 P before the apex.
    """
    tau_m_ns = response.apex_time_ns - response.time_origin_ns
    first_row = apex_sample - FIRST_REFLECTION_LEAD_PERIODS * context.period_samples
    last_row = apex_sample + (tau_m_ns - context.period_ns) / context.dt_ns
    rows = np.arange(wing_mask.shape[0])[:, None]
    return wing_mask & (rows >= first_row) & (rows <= last_row)


def _first_reflection_period(
    tapered: np.ndarray,
    wing_mask: np.ndarray,
    response: PipeResponse,
    context: LineContext,
    apex_sample: float,
) -> tuple[float | None, bool]:
    """Period of the first reflection; ``(None, False)`` means rejected.

    Returns:
        ``(period, period_ringing)``; a failing water-filled PVC pipe keeps
        its measured period and gets ``period_ringing`` True.
    """
    region = first_reflection_mask(wing_mask, response, context, apex_sample)
    measured = measure_period_samples(
        np.where(region, tapered, 0.0),
        pad_factor=context.image_config.period_pad_factor,
        min_period=context.image_config.period_min_samples,
    )
    if period_check_passes(measured, context.period_samples):
        return measured, False
    if measured is not None and is_water_filled_pvc(response.material):
        return measured, True
    return None, False


def _level_boxes(
    wing_envelope: np.ndarray,
    seed: tuple[int, int],
    frame: "_PatchFrame",
    context: LineContext,
) -> dict[float, LevelBox]:
    """Stratum boxes at every level of ``LEVEL_AMPLITUDES`` (untapered field)."""
    peak = float(wing_envelope[seed])
    n_columns = wing_envelope.shape[1]
    boxes = {}
    for level_db, amplitude in LEVEL_AMPLITUDES.items():
        bbox = _bbox(_seed_component(wing_envelope >= amplitude * peak, seed))
        if bbox is None:  # cannot happen: the seed is the peak itself
            continue
        col0, col1, row0, row1 = bbox
        width, height = box_size_in_periods(
            GridBox(0, col1 - col0, row0, row1),
            context.grid_map,
            context.period_samples,
        )
        touches_taper = (
            col0 < frame.n_aperture_taper or col1 >= n_columns - frame.n_aperture_taper
        )
        boxes[level_db] = LevelBox(bbox, width, height, touches_taper)
    return boxes


@dataclass(frozen=True)
class _PatchFrame:
    """Where the apex and the taper cells sit inside a patch."""

    apex_column: int
    apex_sample: float
    period_samples: float
    n_aperture_taper: int
    window_end_row: float


@dataclass(frozen=True)
class _PatchGeometry:
    signal_amplitude: float
    half_bbox: tuple[int, int, int, int]
    wing_bbox: tuple[int, int, int, int]
    truth_bbox: tuple[int, int, int, int]
    wing_mask: np.ndarray
    wing_seed: tuple[int, int]


def _patch_geometry(
    envelope: np.ndarray, wing_envelope: np.ndarray, frame: _PatchFrame
) -> _PatchGeometry | RejectReason:
    """``A``, the -6 dB component and the -20 dB wing and truth boxes.

    ``envelope`` is of the tapered (injected) field, ``wing_envelope`` of the
    untapered one; each threshold is relative to its own apex peak.
    """
    apex = (frame.apex_column, frame.apex_sample, frame.period_samples)
    peak = _apex_peak(envelope, *apex)
    wing_peak = _apex_peak(wing_envelope, *apex)
    if not (peak.amplitude > 0 and wing_peak.amplitude > 0):
        return RejectReason.NO_TRUTH_CELLS
    half_mask = _seed_component(
        envelope >= HALF_WIDTH_AMPLITUDE * peak.amplitude, peak.seed
    )
    wing_mask = _seed_component(
        wing_envelope >= WING_AMPLITUDE * wing_peak.amplitude, wing_peak.seed
    )
    half_bbox, wing_bbox = _bbox(half_mask), _bbox(wing_mask)
    if half_bbox is None or wing_bbox is None:
        return RejectReason.NO_TRUTH_CELLS
    if wing_bbox[3] > frame.window_end_row:
        return RejectReason.BEYOND_SYNTHETIC_WINDOW
    core = _taper_free_cells(
        wing_mask.shape, frame.n_aperture_taper, frame.window_end_row
    )
    truth_bbox = _bbox(wing_mask & core)
    if truth_bbox is None:
        return RejectReason.NO_TRUTH_CELLS
    return _PatchGeometry(
        peak.amplitude, half_bbox, wing_bbox, truth_bbox, wing_mask, wing_peak.seed
    )


# ------------------------------------------------------------ line context


def last_saturated_sample(data: np.ndarray) -> int | None:
    """Last row holding an int16-saturated sample (``|D| >= 32767``)."""
    saturated = np.abs(np.asarray(data, dtype=np.float64)) >= INT16_SATURATION
    rows = np.flatnonzero(saturated.any(axis=1))
    if rows.size == 0:
        return None
    return int(rows[-1])


def prepare_line_context(
    background: Radargram,
    exclusions: LineExclusions = NO_EXCLUSIONS,
    image_config: DetectionImageConfig = DEFAULT_DETECTION_IMAGE_CONFIG,
) -> LineContext:
    """Zero, period, grid map, sections and saturation of one background line.

    ``P_line`` and the ``ImageGridMap`` come from ``make_detection_image`` so
    they are exactly the ones of the evaluated detection image (they do not
    depend on the recipe, S6 (f)).

    Raises:
        ValueError: If the line has no physical scale.
    """
    if not background.has_physical_scale:
        raise ValueError(f"{background.line_id}: injection needs dt_ns and dx_m")
    branch = prepare_fitting_branch(background)
    image = make_detection_image(background, image_config)
    return LineContext(
        background=background,
        zero=find_time_zero(background, branch.filtered),
        period_samples=image.period_samples,
        grid_map=image.grid_map,
        filtered=branch.filtered,
        background_removed=branch.background_removed,
        last_saturated_sample=last_saturated_sample(background.data),
        exclusions=exclusions,
        image_config=image_config,
        tiles=image.tiles,
        tile_stride_px=image.tile_px - image.overlap_px,
    )


# --------------------------------------------------------- exclusion zones


def _aperture_offset(patch: ResponsePatch, apex_trace: float) -> int | None:
    """Integer column offset patch -> line, ``None`` if off the trace grid.

    Traces are re-labelled, never interpolated ([EMR] 1-3), so the apex
    offset must be a whole number of traces.
    """
    offset = apex_trace - patch.response.apex_trace
    if abs(offset - round(offset)) > GRID_TOLERANCE:
        return None
    return round(offset)


def strong_layer_samples(context: LineContext, trace0: int, trace1: int) -> np.ndarray:
    """Rows of a strong flat reflector under traces ``trace0..trace1`` (zone (b')).

    Mean-trace envelope above 4 x its moving median over +-3 P
    (depth-localised reflector, plan v0.2.3). The envelope and the median
    run over the whole record; only rows from band + 2 P on can be flagged.
    A moving (not global) median follows the attenuation trend, so the
    shallow end of the range is not flagged just for being louder.
    """
    _, flagged = _strong_layer_flags(context, trace0, trace1)
    return np.flatnonzero(flagged)


def strong_layer_peaks(context: LineContext, trace0: int, trace1: int) -> np.ndarray:
    """Depth of each strong layer: the envelope peak row of every flagged run.

    Zone (b') measures the 1 P distance from these rows, so an apex is
    excluded only within 1 P of the reflector itself, not of the flanks of
    its envelope (test (j) (ii)).
    """
    envelope, flagged = _strong_layer_flags(context, trace0, trace1)
    runs, n_runs = ndimage.label(flagged)
    if n_runs == 0:
        return np.zeros(0, dtype=int)
    peaks = ndimage.maximum_position(envelope, runs, index=np.arange(1, n_runs + 1))
    return np.array([position[0] for position in peaks], dtype=int)


def long_strong_layer_runs(
    context: LineContext, trace0: int | None = None, trace1: int | None = None
) -> int:
    """Diagnostic: strong layers whose envelope FWHM exceeds 2 P (gpr-processing (a)).

    Long ringing layers lift the +-3 P baseline of zone (b') and are not
    excluded there (decision (a): they stay injectable). This count reports
    them. The mean-trace envelope ``e`` is the one of (b'); the baseline is a
    moving median over +-10 P; peaks above 4 x that baseline and deeper than
    band + 2 P count when their half-height width is over 2 P.

    Limit: a layer longer than about +-10 P lifts this baseline too and is
    not counted. Default traces: the whole line; because the mean trace
    spans the whole line, a layer under only part of it is diluted and not
    counted, so 0 means "no long layer across the line" (amendment 5).
    """
    first = 0 if trace0 is None else trace0
    last = context.background.n_traces - 1 if trace1 is None else trace1
    period = context.period_samples
    envelope = _envelope(context.filtered[:, first : last + 1].mean(axis=1))
    baseline = ndimage.median_filter(
        envelope, size=_odd_window(LONG_LAYER_BASELINE_PERIODS * period), mode="nearest"
    )
    peaks, _ = find_peaks(envelope, height=STRONG_LAYER_FACTOR * baseline)
    k_zero = context.zero.k_zero
    assert k_zero is not None
    peaks = peaks[peaks >= math.ceil(k_zero + BAND_MARGIN_PERIODS * period)]
    if peaks.size == 0:
        return 0
    widths = peak_widths(envelope, peaks, rel_height=0.5)[0]
    return int(np.count_nonzero(widths / period > LONG_LAYER_PERIODS))


def _odd_window(length_samples: float) -> int:
    """``round(length)`` raised to the next odd number, at least 3."""
    window = round(length_samples)
    if window % 2 == 0:
        window += 1
    return max(3, window)


def _strong_layer_flags(
    context: LineContext, trace0: int, trace1: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-trace envelope and its (b') flags over the whole record."""
    k_zero = context.zero.k_zero
    assert k_zero is not None
    period = context.period_samples
    envelope = _envelope(context.filtered[:, trace0 : trace1 + 1].mean(axis=1))
    baseline = ndimage.median_filter(
        envelope, size=strong_layer_window(period), mode="nearest"
    )
    threshold = STRONG_LAYER_FACTOR * np.maximum(baseline, STRONG_LAYER_BASELINE_FLOOR)
    region_start = math.ceil(k_zero + BAND_MARGIN_PERIODS * period)
    in_region = np.arange(len(envelope)) >= region_start
    return envelope, in_region & (envelope > threshold)


def strong_layer_window(period_samples: float) -> int:
    """Odd moving-median length ``round(2 x 3 P)``, at least 3 samples."""
    return _odd_window(2.0 * STRONG_LAYER_BASELINE_PERIODS * period_samples)


def _depth_zone_failure(
    context: LineContext, patch: ResponsePatch, apex_trace: float
) -> RejectReason | None:
    """Zones (a), (b) and (b') on the apex sample."""
    period = context.period_samples
    apex_sample = patch.apex_sample
    saturated = context.last_saturated_sample
    if (
        saturated is not None
        and apex_sample < saturated + SATURATION_MARGIN_PERIODS * period
    ):
        return RejectReason.SATURATION_ZONE
    k_zero = context.zero.k_zero
    assert k_zero is not None
    if apex_sample < k_zero + BAND_MARGIN_PERIODS * period:
        return RejectReason.BAND_ZONE
    trace0 = max(0, math.floor(apex_trace - patch.half_width_6db))
    trace1 = min(
        context.background.n_traces - 1, math.ceil(apex_trace + patch.half_width_6db)
    )
    layers = strong_layer_peaks(context, trace0, trace1)
    if np.any(np.abs(layers - apex_sample) <= STRONG_LAYER_MARGIN_PERIODS * period):
        return RejectReason.STRONG_LAYER_ZONE
    return None


@dataclass(frozen=True)
class _Rectangle:
    """Closed L0 rectangle ``[trace0, trace1] x [sample0, sample1]`` (floats)."""

    trace0: float
    trace1: float
    sample0: float
    sample1: float

    def overlaps(self, other: "_Rectangle") -> bool:
        return (
            self.trace0 <= other.trace1
            and other.trace0 <= self.trace1
            and self.sample0 <= other.sample1
            and other.sample0 <= self.sample1
        )


def _grown_box(box: GridBox, context: LineContext) -> _Rectangle:
    """Un-injected box grown by 1 P in traces (L1 isotropy) and in samples."""
    trace_margin = BOX_MARGIN_PERIODS * context.period_traces
    sample_margin = BOX_MARGIN_PERIODS * context.period_samples
    return _Rectangle(
        box.i0 - trace_margin,
        box.i1 + trace_margin,
        box.k0 - sample_margin,
        box.k1 + sample_margin,
    )


def _injection_rectangles(
    context: LineContext, patch: ResponsePatch, apex_trace: float
) -> tuple[_Rectangle, _Rectangle]:
    """The -6 dB core and the apex match range of an injection, in L0."""
    offset = round(apex_trace - patch.response.apex_trace)
    col0, col1, row0, row1 = patch.core_bbox
    core = _Rectangle(offset + col0, offset + col1, row0, row1)
    sample_margin = MATCH_SAMPLE_PERIODS * context.period_samples
    match_range = _Rectangle(
        apex_trace - MATCH_TRACE_MARGIN,
        apex_trace + MATCH_TRACE_MARGIN,
        patch.apex_sample - sample_margin,
        patch.apex_sample + sample_margin,
    )
    return core, match_range


def _box_zone_hit(
    context: LineContext, patch: ResponsePatch, apex_trace: float
) -> bool:
    """Zone (c'): a grown un-injected box meets the core or the match range.

    Uses only the background boxes and the injection itself, so the blocked
    set is the same for every recipe (fixed denominator, [CVR] B-1).
    """
    core, match_range = _injection_rectangles(context, patch, apex_trace)
    for box in context.exclusions.uninjected_boxes:
        grown = _grown_box(box, context)
        if grown.overlaps(core) or grown.overlaps(match_range):
            return True
    return False


def in_horizontal_box_zone(context: LineContext, apex_trace: float) -> bool:
    """Old horizontal-only zone (c): ``centre +- (box width / 2 + 1 P)``.

    Not a rejection any more; kept so S5 can report the recall of the
    injections it would have blocked ([CVR] B-1).
    """
    margin = BOX_MARGIN_PERIODS * context.period_traces
    for box in context.exclusions.uninjected_boxes:
        centre = (box.i0 + box.i1 + 1) / 2.0
        if abs(apex_trace - centre) <= box.n_traces / 2.0 + margin:
            return True
    return False


def _lateral_zone_failure(
    context: LineContext, patch: ResponsePatch, apex_trace: float
) -> RejectReason | None:
    """Zones (c') and (d).

    Zone (d) keeps the -20 dB half width around the 41 K apexes (M6 guard
    independence) and only the core, -6 dB half width + 1 P, around the
    automatic candidates (amendment 5, [CVR] appendix C-2).
    """
    if _box_zone_hit(context, patch, apex_trace):
        return RejectReason.UNINJECTED_BOX_ZONE
    exclusions = context.exclusions
    for candidate in exclusions.k_apex_traces:
        if abs(apex_trace - candidate) <= patch.half_width_20db:
            return RejectReason.CANDIDATE_APEX_ZONE_K
    auto_radius = patch.half_width_6db + context.period_traces
    for candidate in exclusions.auto_candidate_apex_traces:
        if abs(apex_trace - candidate) <= auto_radius:
            return RejectReason.CANDIDATE_APEX_ZONE_AUTO
    return None


def placement_failure(
    context: LineContext, patch: ResponsePatch, apex_trace: float
) -> RejectReason | None:
    """Why ``patch`` may not sit at L0 ``apex_trace`` on this line (zones (a)-(d)).

    Zone (c') is two-dimensional: un-injected boxes grown by 1 P block only
    an injection whose -6 dB core or apex match range they touch.
    """
    offset = _aperture_offset(patch, apex_trace)
    if offset is None:
        return RejectReason.OFF_GRID_APEX
    if offset < 0 or offset + patch.response.n_traces > context.background.n_traces:
        return RejectReason.APERTURE_OUTSIDE_LINE
    depth_failure = _depth_zone_failure(context, patch, apex_trace)
    if depth_failure is not None:
        return depth_failure
    return _lateral_zone_failure(context, patch, apex_trace)


PlacedApex = tuple[float, float]
"""``(apex_trace, half_width_20db)`` of an already placed injection."""


def realization_spacing_ok(
    apex_trace: float,
    half_width: float,
    placed: Iterable[PlacedApex],
    period_traces: float,
) -> bool:
    """(e') inside one realization: centres >= ``W_-20 + 1 P`` apart.

    With two models the -20 dB half widths add (``W_-20`` for equal models),
    so the two wings stay apart by at least 1 P.
    """
    gap_traces = REALIZATION_GAP_PERIODS * period_traces
    return all(
        abs(apex_trace - other) >= half_width + other_half_width + gap_traces
        for other, other_half_width in placed
    )


def line_spacing_ok(
    apex_trace: float, half_width: float, placed: Iterable[PlacedApex]
) -> bool:
    """(e') across all realizations of a line: centres >= ``W_-20 / 2`` apart.

    ``W_-20 / 2`` of the wider model, so no two positions reuse nearly the
    same background (near-duplicate samples, [CVR] B-3).
    """
    return all(
        abs(apex_trace - other)
        >= LINE_GAP_WIDTHS * 2.0 * max(half_width, other_half_width)
        for other, other_half_width in placed
    )


# --------------------------------------------------------------- injection


def box_size_in_periods(
    box: GridBox, grid_map: ImageGridMap, period_samples: float
) -> tuple[float, float]:
    """L1 width and height of an L0 box in L1 periods (forward map only).

    The L1 period is ``P_line / samples_per_px`` px; the box spans
    ``n_traces / traces_per_px`` by ``n_samples / samples_per_px`` px.
    """
    period_l1_px = period_samples / grid_map.samples_per_px
    width_px = box.n_traces / grid_map.traces_per_px
    height_px = box.n_samples / grid_map.samples_per_px
    return width_px / period_l1_px, height_px / period_l1_px


def width_class(width_periods: float, aperture_limited: bool) -> WidthClass:
    """Width stratum against the AIHub p5-p95 range 7.79-14.67 P.

    An aperture-limited width is a lower bound and counts as too wide.
    """
    if aperture_limited:
        return WidthClass.WIDTH_OVER_LOWER_BOUND
    width_lo, width_hi = AIHUB_WIDTH_PERIODS
    if width_periods < width_lo:
        return WidthClass.WIDTH_UNDER
    if width_periods > width_hi:
        return WidthClass.WIDTH_OVER
    return WidthClass.IN


def height_class(height_periods: float) -> HeightClass:
    """Report-only height reason against 5.22-8.35 P (never filters)."""
    height_lo, height_hi = AIHUB_HEIGHT_PERIODS
    if height_periods < height_lo:
        return HeightClass.HEIGHT_UNDER
    if height_periods > height_hi:
        return HeightClass.HEIGHT_OVER
    return HeightClass.IN


def is_in_distribution(
    widths_by_level: Mapping[float, float],
    aperture_limited_by_level: Mapping[float, bool],
    level_w_db: float,
) -> bool:
    """Width-only stratum (user decision (B), amendment 5) at level ``l_W``.

    ``level_w_db`` has no default: it is chosen once from the position table
    and fixed at F2.
    """
    return (
        width_class(widths_by_level[level_w_db], aperture_limited_by_level[level_w_db])
        is WidthClass.IN
    )


def snr_window(
    context: LineContext, patch: ResponsePatch, apex_trace: float
) -> GridBox:
    """SNR window in L0: apex +- 1 P samples x -6 dB half-width traces."""
    period = context.period_samples
    trace0 = max(0, math.floor(apex_trace - patch.half_width_6db))
    trace1 = min(
        context.background.n_traces - 1, math.ceil(apex_trace + patch.half_width_6db)
    )
    row0 = max(0, math.floor(patch.apex_sample - period))
    row1 = min(context.background.n_samples - 1, math.ceil(patch.apex_sample + period))
    return GridBox(i0=trace0, i1=trace1, k0=row0, k1=row1)


def _noise_rms(context: LineContext, patch: ResponsePatch, apex_trace: float) -> float:
    """``sigma``: RMS of the (1)-(4) background in the SNR window."""
    window = snr_window(context, patch, apex_trace)
    values = context.background_removed[window.sample_slice, window.trace_slice]
    return float(np.sqrt(np.mean(values**2)))


@dataclass(frozen=True)
class _TruthBoxes:
    box: GridBox
    wing_box: GridBox
    core_box: GridBox
    box_clipped: bool


def _placed_box(
    bbox: tuple[int, int, int, int], offset: int, last_row: int | None = None
) -> GridBox:
    """Patch bbox moved to the line; rows cut at ``last_row`` when given."""
    col0, col1, row0, row1 = bbox
    if last_row is not None:
        row1 = min(row1, last_row)
    return GridBox(i0=offset + col0, i1=offset + col1, k0=row0, k1=row1)


def _truth_boxes(patch: ResponsePatch, offset: int, n_samples: int) -> _TruthBoxes:
    """Truth and core boxes (clipped to the record) and the unclipped wing box."""
    last_row = n_samples - 1
    box = _placed_box(patch.truth_bbox, offset, last_row)
    return _TruthBoxes(
        box=box,
        wing_box=_placed_box(patch.wing_bbox, offset),
        core_box=_placed_box(patch.core_bbox, offset, last_row),
        box_clipped=box.k1 < patch.truth_bbox[3],
    )


def _make_truth(
    spec: InjectionSpec,
    context: LineContext,
    patch: ResponsePatch,
    gain: float,
    noise_rms: float,
    clip: "ClipOutcome",
) -> InjectionTruth:
    offset = round(spec.apex_trace - patch.response.apex_trace)
    boxes = _truth_boxes(patch, offset, context.background.n_samples)
    width_periods, height_periods = box_size_in_periods(
        boxes.wing_box, context.grid_map, context.period_samples
    )
    bottom_limit = (
        context.background.n_samples - BOTTOM_STRATUM_PERIODS * context.period_samples
    )
    return InjectionTruth(
        injection_id=spec.injection_id,
        response_id=spec.response_id,
        snr_db=spec.snr_db,
        polarity=spec.polarity,
        k=spec.k,
        box=boxes.box,
        wing_box=boxes.wing_box,
        core_box=boxes.core_box,
        apex_trace=spec.apex_trace,
        apex_sample=patch.apex_sample,
        width_periods=width_periods,
        height_periods=height_periods,
        bottom_stratum=patch.apex_sample >= bottom_limit,
        gain=gain,
        signal_amplitude=patch.signal_amplitude,
        noise_rms=noise_rms,
        effective_eps_r=patch.effective_eps_r,
        box_clipped=boxes.box_clipped,
        in_horizontal_box_zone=in_horizontal_box_zone(context, spec.apex_trace),
        visible_box=visible_box(
            patch, offset, spec.snr_db, context.background.n_samples
        ),
        period_ringing=patch.period_ringing,
        n_clipped_cells=clip.n_clipped_cells,
        max_clipped_injection_ratio=clip.max_ratio,
        width_periods_by_level={
            level: box.width_periods for level, box in patch.level_boxes.items()
        },
        height_periods_by_level={
            level: box.height_periods for level, box in patch.level_boxes.items()
        },
        aperture_limited_by_level={
            level: box.aperture_limited for level, box in patch.level_boxes.items()
        },
    )


def visible_amplitude(snr_db: float) -> float:
    """IoU-report threshold ``max(10^(-s/20), 0.1)``; -6 dB (0.5) at 0 dB."""
    if snr_db <= 0:
        return VISIBLE_ZERO_DB_AMPLITUDE
    return max(10.0 ** (-snr_db / 20.0), VISIBLE_FLOOR_AMPLITUDE)


def visible_box(
    patch: ResponsePatch, offset: int, snr_db: float, n_samples: int
) -> GridBox:
    """Apex-seeded component of the injected envelope above the visible level.

    Report only ([EMR] appendix B-1 (b)): not a truth, stratum or zone box.
    """
    envelope = patch.envelope
    assert envelope is not None  # set by build_response_patch
    peak = _apex_peak(
        envelope,
        round(patch.response.apex_trace),
        patch.apex_sample,
        patch.period_samples,
    )
    mask = envelope >= visible_amplitude(snr_db) * peak.amplitude
    bbox = _bbox(_seed_component(mask, peak.seed))
    assert bbox is not None  # the seed is the peak cell
    return _placed_box(bbox, offset, n_samples - 1)


@dataclass(frozen=True)
class ClipOutcome:
    """int16 clipping of one injection (amendment 5, [EMR] appendix B-3).

    Attributes:
        n_clipped_cells: Cells of the sum outside ``[-32768, 32767]``.
        max_ratio: Largest ``|field| / A`` of a clipped cell, i.e. the
            injected value over the apex injection amplitude (0 if none).
        protected_hit: A clipped cell lies in the truth box or SNR window.
    """

    n_clipped_cells: int
    max_ratio: float
    protected_hit: bool

    @property
    def rejects(self) -> bool:
        """Rules (i) and (ii): the only two int16 rejections."""
        return self.protected_hit or self.max_ratio > CLIPPED_INJECTION_RATIO_MAX


def _any_inside(rows: np.ndarray, columns: np.ndarray, box: GridBox) -> bool:
    inside = (
        (rows >= box.k0) & (rows <= box.k1) & (columns >= box.i0) & (columns <= box.i1)
    )
    return bool(inside.any())


def _clip_sum(
    data: np.ndarray,
    patch: ResponsePatch,
    offset: int,
    gain: float,
    protected: Sequence[GridBox],
) -> tuple[np.ndarray, ClipOutcome]:
    """``data + gain * field`` on the patch columns, clipped to int16.

    A clipped cell on an already saturated sample keeps the recorded value,
    the best physical guess when the injection is small there.
    """
    n_samples = data.shape[0]
    columns = slice(offset, offset + patch.response.n_traces)
    field_rows = patch.field[:n_samples]
    summed = data[:, columns] + gain * field_rows
    clipped = (summed < INT16_MIN) | (summed > INT16_MAX)
    rows, patch_columns = np.nonzero(clipped)
    max_ratio = 0.0
    if rows.size:
        max_ratio = float(np.abs(field_rows[clipped]).max() / patch.signal_amplitude)
    line_columns = patch_columns + offset
    outcome = ClipOutcome(
        n_clipped_cells=int(rows.size),
        max_ratio=max_ratio,
        protected_hit=any(_any_inside(rows, line_columns, box) for box in protected),
    )
    return np.clip(summed, INT16_MIN, INT16_MAX), outcome


@dataclass
class _RealizationState:
    """Mutable bookkeeping of one ``inject_realization`` call."""

    data: np.ndarray
    patches: dict[tuple[str, float], ResponsePatch | RejectReason] = field(
        default_factory=dict
    )
    placed: list[PlacedApex] = field(default_factory=list)


def _inject_one(
    spec: InjectionSpec,
    context: LineContext,
    responses: Mapping[str, PipeResponse],
    state: _RealizationState,
    apply_2p5d_weight: bool,
) -> InjectionTruth | RejectReason:
    response = responses.get(spec.response_id)
    if response is None:
        return RejectReason.UNKNOWN_RESPONSE
    key = (spec.response_id, spec.k)
    if key not in state.patches:
        state.patches[key] = build_response_patch(
            response, context, spec.k, apply_2p5d_weight=apply_2p5d_weight
        )
    patch = state.patches[key]
    if isinstance(patch, RejectReason):
        return patch
    failure = placement_failure(context, patch, spec.apex_trace)
    if failure is not None:
        return failure
    if not realization_spacing_ok(
        spec.apex_trace, patch.half_width_20db, state.placed, context.period_traces
    ):
        return RejectReason.SPACING
    noise_rms = _noise_rms(context, patch, spec.apex_trace)
    if not noise_rms > 0:
        return RejectReason.ZERO_NOISE
    gain = (
        spec.polarity
        * 10.0 ** (spec.snr_db / 20.0)
        * noise_rms
        / patch.signal_amplitude
    )
    offset = round(spec.apex_trace - response.apex_trace)
    protected = (
        _truth_boxes(patch, offset, context.background.n_samples).box,
        snr_window(context, patch, spec.apex_trace),
    )
    summed, clip = _clip_sum(state.data, patch, offset, gain, protected)
    if clip.rejects:
        return RejectReason.INT16_OVERFLOW
    state.data[:, offset : offset + response.n_traces] = summed
    state.placed.append((spec.apex_trace, patch.half_width_20db))
    return _make_truth(spec, context, patch, gain, noise_rms, clip)


def _check_one_realization(specs: Sequence[InjectionSpec], line_id: str) -> str:
    realization_ids = {spec.realization_id for spec in specs}
    if len(realization_ids) != 1:
        raise ValueError(f"specs must share one realization_id, got {realization_ids}")
    wrong_lines = {spec.line_id for spec in specs} - {line_id}
    if wrong_lines:
        raise ValueError(f"specs name other lines {wrong_lines} than {line_id!r}")
    return realization_ids.pop()


def inject_realization(
    context: LineContext,
    specs: Sequence[InjectionSpec],
    responses: Mapping[str, PipeResponse],
    *,
    apply_2p5d_weight: bool = True,
) -> InjectionResult:
    """Inject the rows of one realization into a float64 copy of the line.

    Rows are processed in order; a row that fails a zone, the spacing to an
    earlier accepted row or the int16 range is rejected and leaves the data
    unchanged. ``sigma`` always comes from the background alone.

    Args:
        context: Background line (``prepare_line_context``).
        specs: Table rows with one ``realization_id`` and this line's id.
        responses: Pipe responses by ``response_id``.
        apply_2p5d_weight: 2.5-D time weight (on by default, plan 3-2).

    Raises:
        ValueError: If the rows mix realizations or name another line.
    """
    realization_id = _check_one_realization(specs, context.background.line_id)
    state = _RealizationState(
        data=np.asarray(context.background.data, dtype=np.float64).copy()
    )
    truths: list[InjectionTruth] = []
    rejected: list[tuple[str, str]] = []
    for spec in specs:
        outcome = _inject_one(spec, context, responses, state, apply_2p5d_weight)
        if isinstance(outcome, RejectReason):
            rejected.append((spec.injection_id, outcome.value))
        else:
            truths.append(outcome)
    return InjectionResult(
        realization_id=realization_id,
        line_id=context.background.line_id,
        data=state.data,
        truths=tuple(truths),
        rejected=tuple(rejected),
    )


# ---------------------------------------------------------------- planning


def _line_seed(seed: int, line_id: str) -> np.random.Generator:
    """Generator fixed by the plan seed and the line id (stable across runs)."""
    return np.random.default_rng([seed, zlib.crc32(line_id.encode("utf-8"))])


def _allowed_apex_traces(context: LineContext, patch: ResponsePatch) -> list[float]:
    n_offsets = context.background.n_traces - patch.response.n_traces + 1
    candidates = (
        offset + patch.response.apex_trace for offset in range(max(0, n_offsets))
    )
    return [
        apex_trace
        for apex_trace in candidates
        if placement_failure(context, patch, apex_trace) is None
    ]


def plan_line_positions(
    context: LineContext,
    patches: Sequence[ResponsePatch],
    *,
    max_injections: int = MAX_INJECTIONS_PER_LINE,
    seed: int = PLAN_SEED,
) -> LinePlan:
    """Seeded apex positions of one line in at most two realizations ([CVR] B-4).

    Allowed (model, apex trace) pairs are visited in a seeded random order;
    the first bottom-stratum candidate of that order (apex >= ``n_s - 3
    P_line``) goes first, one per line (amendment 6). A candidate is skipped when
    it is closer than ``W_-20 / 2`` to any chosen position of the line;
    otherwise it joins the first of the
    ``MAX_REALIZATIONS_PER_LINE`` groups where the realization spacing
    ``W_-20 + 1 P`` holds, or is dropped and counted. The plan stops at
    ``max_injections`` positions.

    Args:
        context: Background line with its exclusions.
        patches: Usable models on this line (main ``k``).
        max_injections: Positions per line (per SNR step).
        seed: Plan seed (``PLAN_SEED``); combined with the line id.
    """
    candidates = [
        (patch, apex_trace)
        for patch in patches
        for apex_trace in _allowed_apex_traces(context, patch)
    ]
    is_bottom = [is_bottom_stratum(context, patch) for patch, _ in candidates]
    order = _bottom_first_order(is_bottom, _line_seed(seed, context.background.line_id))
    groups: list[list[PlacedApex]] = [[] for _ in range(MAX_REALIZATIONS_PER_LINE)]
    chosen: list[PlannedPosition] = []
    n_dropped = 0
    for index in order:
        if len(chosen) >= max_injections:
            break
        patch, apex_trace = candidates[int(index)]
        placed_on_line = [apex for group in groups for apex in group]
        if not line_spacing_ok(apex_trace, patch.half_width_20db, placed_on_line):
            continue
        group = _fitting_group(groups, apex_trace, patch, context.period_traces)
        if group is None:
            n_dropped += 1
            continue
        groups[group].append((apex_trace, patch.half_width_20db))
        chosen.append(PlannedPosition(group, patch.response.response_id, apex_trace))
    positions = sorted(
        chosen, key=lambda position: (position.group, position.apex_trace)
    )
    return LinePlan(
        line_id=context.background.line_id,
        positions=tuple(positions),
        n_candidates=len(candidates),
        n_dropped=n_dropped,
        candidates_by_response=_candidate_counts(candidates, is_bottom),
    )


def _candidate_counts(
    candidates: Sequence[tuple[ResponsePatch, float]], is_bottom: Sequence[bool]
) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = {}
    for (patch, _), bottom in zip(candidates, is_bottom, strict=True):
        tally = counts.setdefault(patch.response.response_id, [0, 0])
        tally[0] += int(bottom)
        tally[1] += 1
    return {key: (value[0], value[1]) for key, value in counts.items()}


def is_bottom_stratum(context: LineContext, patch: ResponsePatch) -> bool:
    """Apex within 3 ``P_line`` of the record end (plan 3-2 "아래 끝 층")."""
    bottom_limit = (
        context.background.n_samples - BOTTOM_STRATUM_PERIODS * context.period_samples
    )
    return patch.apex_sample >= bottom_limit


def _bottom_first_order(
    is_bottom: Sequence[bool], rng: np.random.Generator
) -> list[int]:
    """One seeded permutation; its first ``BOTTOM_PRIORITY_PER_LINE`` bottom
    candidates move to the front, everything else keeps the permutation order.
    """
    order = [int(index) for index in rng.permutation(len(is_bottom))]
    first = [index for index in order if is_bottom[index]][:BOTTOM_PRIORITY_PER_LINE]
    return first + [index for index in order if index not in first]


def _fitting_group(
    groups: list[list[PlacedApex]],
    apex_trace: float,
    patch: ResponsePatch,
    period_traces: float,
) -> int | None:
    for index, placed in enumerate(groups):
        if realization_spacing_ok(
            apex_trace, patch.half_width_20db, placed, period_traces
        ):
            return index
    return None


# --------------------------------------------------------------- cost table


def _wing_span_px(
    context: LineContext, patch: ResponsePatch, apex_trace: float
) -> tuple[float, float]:
    """Line L1 x range of the -20 dB wing box (``ImageGridMap`` forward map)."""
    offset = round(apex_trace - patch.response.apex_trace)
    col0, col1, _, _ = patch.wing_bbox
    traces_per_px = context.grid_map.traces_per_px
    return (offset + col0) / traces_per_px, (offset + col1 + 1) / traces_per_px


def _tiles_touching(
    context: LineContext, spans_px: Iterable[tuple[float, float]]
) -> int:
    """Tiles overlapping any span grown by one tile stride ([CVR] B-3)."""
    touched: set[int] = set()
    for start_px, stop_px in spans_px:
        low, high = start_px - context.tile_stride_px, stop_px + context.tile_stride_px
        for tile in context.tiles:
            if tile.x0_px < high and low < tile.x0_px + tile.content_width_px:
                touched.add(tile.index)
    return len(touched)


def line_cost(
    context: LineContext,
    plan: LinePlan,
    patches: Mapping[str, ResponsePatch],
    n_conditions: int,
) -> LineCost:
    """P5b work of one planned line per recipe and fold.

    Args:
        context: The planned line.
        plan: ``plan_line_positions`` result.
        patches: Main-``k`` patches by ``response_id`` (wing boxes for tiles).
        n_conditions: Conditions per realization (10 for ``default_conditions``).
    """
    tiles_per_condition = 0
    for group in sorted({position.group for position in plan.positions}):
        spans = [
            _wing_span_px(context, patches[position.response_id], position.apex_trace)
            for position in plan.positions
            if position.group == group
        ]
        tiles_per_condition += _tiles_touching(context, spans)
    return LineCost(
        line_id=plan.line_id,
        n_positions=len(plan.positions),
        n_realizations=plan.n_realizations,
        n_conditions=n_conditions,
        l1_builds=plan.n_realizations * n_conditions,
        inferred_tiles=tiles_per_condition * n_conditions,
        long_layer_runs=long_strong_layer_runs(context),
    )


COST_TABLE_COLUMNS = (
    "line_id",
    "n_positions",
    "n_realizations",
    "n_conditions",
    "l1_builds",
    "inferred_tiles",
    "long_layer_runs",
)


def write_cost_table(path: Path, costs: Sequence[LineCost]) -> str:
    """Write the per-line cost table (UTF-8 CSV) and return its SHA-256."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COST_TABLE_COLUMNS)
        for cost in costs:
            writer.writerow([getattr(cost, name) for name in COST_TABLE_COLUMNS])
    return file_sha256(path)


def default_conditions() -> tuple[InjectionCondition, ...]:
    """Main SNR 0/6/12/18 dB x polarity +-1 at ``k`` 1.3; report 12 dB x +-1 at 0.8."""
    main = [
        InjectionCondition(snr_db, polarity, K_MAIN)
        for snr_db in SNR_LADDER_DB
        for polarity in POLARITIES
    ]
    report = [InjectionCondition(12.0, polarity, K_REPORT) for polarity in POLARITIES]
    return (*main, *report)


def realization_name(line_id: str, group: int, condition: InjectionCondition) -> str:
    """``<line>:g<group>:snr<dB>:k<k>:p<+1|-1>``."""
    return (
        f"{line_id}:g{group:02d}:snr{condition.snr_db:g}:k{condition.k:g}"
        f":p{condition.polarity:+d}"
    )


def build_injection_specs(
    line_id: str,
    positions: Sequence[PlannedPosition],
    conditions: Sequence[InjectionCondition],
    seed: int = PLAN_SEED,
) -> list[InjectionSpec]:
    """Table rows: every position group under every condition.

    The same positions serve all SNR steps, polarities and ``k`` values, which
    is the pairing of [CVR] 2-2 item 1.
    """
    specs = []
    for condition in conditions:
        for group in sorted({position.group for position in positions}):
            realization_id = realization_name(line_id, group, condition)
            members = [position for position in positions if position.group == group]
            for index, position in enumerate(members):
                specs.append(
                    InjectionSpec(
                        injection_id=f"{realization_id}:{index:02d}",
                        realization_id=realization_id,
                        line_id=line_id,
                        apex_trace=position.apex_trace,
                        response_id=position.response_id,
                        snr_db=condition.snr_db,
                        polarity=condition.polarity,
                        k=condition.k,
                        seed=seed,
                    )
                )
    return specs


# -------------------------------------------------------------- table I/O


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of the file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_injection_table(path: Path, specs: Sequence[InjectionSpec]) -> str:
    """Write the table as UTF-8 CSV and return its SHA-256.

    Floats are written with ``repr`` so reading gives the same values.

    Raises:
        ValueError: If an ``injection_id`` repeats.
    """
    ids = [spec.injection_id for spec in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("injection_id values must be unique")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(INJECTION_TABLE_COLUMNS)
        for spec in specs:
            writer.writerow(
                [_table_cell(getattr(spec, name)) for name in INJECTION_TABLE_COLUMNS]
            )
    return file_sha256(path)


def _table_cell(value: object) -> str:
    if isinstance(value, float):
        return repr(value)
    return str(value)


def read_injection_table(path: Path) -> tuple[list[InjectionSpec], str]:
    """Rows of an injection table and the file's SHA-256.

    Raises:
        ValueError: If the header differs from ``INJECTION_TABLE_COLUMNS``.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != INJECTION_TABLE_COLUMNS:
            raise ValueError(
                f"{path}: header {reader.fieldnames} is not the table header"
            )
        specs = [_spec_from_row(row) for row in reader]
    return specs, file_sha256(path)


def _spec_from_row(row: Mapping[str, str]) -> InjectionSpec:
    return InjectionSpec(
        injection_id=row["injection_id"],
        realization_id=row["realization_id"],
        line_id=row["line_id"],
        apex_trace=float(row["apex_trace"]),
        response_id=row["response_id"],
        snr_db=float(row["snr_db"]),
        polarity=int(row["polarity"]),
        k=float(row["k"]),
        seed=int(row["seed"]),
    )


# -------------------------------------------------------------- pseudo band


def make_pseudo_band(
    apex_trace_values: np.ndarray,
    background_trace: np.ndarray,
    n_traces: int,
    dx_m: float,
    taper_m: float = APERTURE_TAPER_M,
) -> np.ndarray:
    """Synthetic pseudo band for M2: the apex trace repeated plus ``_bg``.

    The repeated apex trace is cosine-tapered over ``taper_m`` at both ends
    ([EMR] E-6, >= 0.2 m); the background trace is added untapered.

    Returns:
        float64 ``[n_t, n_traces]``.
    """
    weights = edge_taper_weights(n_traces, round(taper_m / dx_m))
    band = np.asarray(apex_trace_values, dtype=np.float64)[:, None] * weights[None, :]
    return band + np.asarray(background_trace, dtype=np.float64)[:, None]
