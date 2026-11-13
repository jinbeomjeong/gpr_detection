"""Detection-image layer (L1) and its mapping to the radargram grid (L0).

``docs/architecture/README.md`` section 2-3 defines three coordinate layers:
L0 radargram grid (trace ``i``, sample ``k``), L1 detection image pixels
(``u``, ``v``) and L2 model input pixels. This module owns the L1 <-> L0 map.

Pixel convention used by every layer: coordinates are continuous and pixel
(or trace) ``j`` covers the half-open interval ``[j, j + 1)``. A box that
covers columns 10..19 therefore spans ``x1 = 10`` to ``x2 = 20``, which is
also how Ultralytics reports ``xyxy`` boxes.

Two ways to get an L1 image:

* Scale-free inputs (AIHub, Morocco JPG) use :meth:`ImageGridMap.identity`
  because their pixels already are the grid.
* Inputs with a physical scale (4TU, Guangzhou, GPR-SD) go through
  :func:`make_detection_image` (spec_detection_image_v1 section 4, M3-1): one
  line L1 image per profile with a period-matched vertical scale and a
  depth-isotropic horizontal scale, cut into overlapping 256 x 256 tiles at
  integer pixel origins. The line has a single :class:`ImageGridMap`.

State (spec v0.4, M3-1): the line image, tiling and overlap rules are
implemented here; the defaults are ``P*`` 6.0 px and overlap 192 px. The
padding grey, the grey mapping and the background window are M3-2
assumptions, listed in ``DetectionImage.assumed_params``.
"""

import itertools
import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from gpr_detection.io.radargram import Radargram, velocity_from_eps
from gpr_detection.preprocess.filters import agc, prepare_fitting_branch, remove_dc

CEIL_TOLERANCE_PX = 1e-9
"""Float noise ignored when rounding a pixel width up (e.g. 65.0000000001 -> 65)."""

NS_PER_S = 1e9
DEPTH_PER_SAMPLE_KEY = "depth_per_sample_m_rough"
"""GPR-SD json field with the rough depth per sample (m), independent of dt."""

GRAY_CENTER = 127.5
"""Grey level of zero amplitude before rounding (``rint`` gives 128)."""

SYMMETRIC_GRAY_MAPPING = "symmetric"


@dataclass(frozen=True)
class ImageGridMap:
    """Linear map from detection-image pixels (L1) to the radargram grid (L0).

    ``trace = trace_offset + u * traces_per_px`` and
    ``sample = sample_offset + v * samples_per_px``. The two axes have separate
    factors because a resized B-scan need not keep a square pixel.

    Attributes:
        trace_offset: Grid trace at image column 0 (line L1 origin, 0 for
            ``DetectionImage``; the tile origin is ``LineTile.x0_px``).
        sample_offset: Grid sample at image row 0 (line L1 origin, 0 for
            ``DetectionImage``).
        traces_per_px: Traces per image pixel along x.
        samples_per_px: Samples per image pixel along y.
        n_traces: Trace count of the whole radargram, for clipping.
        n_samples: Sample count of the whole radargram, for clipping.
    """

    trace_offset: int
    sample_offset: int
    traces_per_px: float
    samples_per_px: float
    n_traces: int
    n_samples: int

    def __post_init__(self) -> None:
        if self.traces_per_px <= 0 or self.samples_per_px <= 0:
            raise ValueError(f"grid factors must be positive, got {self}")
        if self.n_traces <= 0 or self.n_samples <= 0:
            raise ValueError(f"grid size must be positive, got {self}")
        if not (
            0 <= self.trace_offset < self.n_traces
            and 0 <= self.sample_offset < self.n_samples
        ):
            raise ValueError(
                f"line L1 origin (0 for DetectionImage) must lie inside the grid, "
                f"got {self}"
            )

    @classmethod
    def identity(cls, width_px: int, height_px: int) -> "ImageGridMap":
        """Map for images whose pixels are the grid (AIHub, Morocco JPG).

        Args:
            width_px: Image width = number of traces.
            height_px: Image height = number of samples.
        """
        return cls(
            trace_offset=0,
            sample_offset=0,
            traces_per_px=1.0,
            samples_per_px=1.0,
            n_traces=width_px,
            n_samples=height_px,
        )


# --------------------------------------------------------------- configuration


@dataclass(frozen=True)
class DetectionImageConfig:
    """Settings of the line L1 image (spec_detection_image_v1 section 4-2).

    Attributes:
        p_target_px: Target period ``P*`` in L1 px (D1 r4, N2-a' baseline,
            2026-11-14; 4.06 was the earlier default).
        overlap_px: Detector-side overlap ``O_cv`` (D1 r4 rule O: 75 %, final).
        tile_px: Tile side ``L`` (training images are 256 x 256).
        overlap_min_px: Lower bound ``O_min`` of the physical overlap.
        a_max_m: Pick tracking half-aperture limit (spec_depth section 5).
        overlap_width_periods: Widest AIHub ``tpipe`` box in periods (19.9).
        overlap_max_px: Overlap cap; keeps the stride at least 32 px.
        compress_max: Compression ``c`` above this flags ``scale_out_of_range``,
            i.e. outside the scales with confirmed detector recall (1 / 0.74,
            N2-a' scale curve, spec section 11-2 item 5).
        period_pad_factor: Zero padding of the period spectrum.
        period_min_samples: Shortest valid period; the longest is ``n_s / 4``.
        clip_percentile: ``|a|`` percentile used as the grey clip level.
        v_assumed_m_per_ns: Horizontal-scale velocity without metadata
            (Guangzhou, Q7).
        pad_gray: Grey of the padding (M3-2 assumption, Q3).
        gray_mapping: Amplitude -> grey rule; only ``"symmetric"`` exists
            (M3-2 assumption, Q3).
        detection_bg_window: Background median window of the detection image
            in traces (M3-2 assumption, Q5).
    """

    p_target_px: float = 6.0
    overlap_px: int = 192
    tile_px: int = 256
    overlap_min_px: int = 64
    a_max_m: float = 1.0
    overlap_width_periods: float = 20.0
    overlap_max_px: int = 224
    compress_max: float = 1.35
    period_pad_factor: int = 8
    period_min_samples: float = 4.0
    clip_percentile: float = 99.0
    v_assumed_m_per_ns: float = 0.10
    pad_gray: int = 128
    gray_mapping: str = SYMMETRIC_GRAY_MAPPING
    detection_bg_window: int = 301

    def __post_init__(self) -> None:
        if not self.p_target_px > 0:
            raise ValueError(f"p_target_px must be positive, got {self.p_target_px}")
        if not 0 <= self.overlap_px < self.tile_px:
            raise ValueError(f"need 0 <= overlap_px < tile_px, got {self}")
        if not 0 <= self.overlap_max_px < self.tile_px:
            raise ValueError(f"need 0 <= overlap_max_px < tile_px, got {self}")
        if self.gray_mapping != SYMMETRIC_GRAY_MAPPING:
            raise ValueError(f"unknown gray_mapping {self.gray_mapping!r}")
        if not 0 <= self.pad_gray <= 255:
            raise ValueError(f"pad_gray must be a uint8 grey, got {self.pad_gray}")


DEFAULT_DETECTION_IMAGE_CONFIG = DetectionImageConfig()

DATASET_DEFAULT_PERIOD: dict[str, tuple[float, str]] = {
    "4tu": (2.51, "ns"),
    "guangzhou": (2.5, "ns"),
    "gprsd": (51.2, "samples"),
}
"""Fallback period per dataset when the measurement fails (spec section 2-2)."""

M3_2_ASSUMED = ("pad_gray", "gray_mapping", "detection_bg_window")
"""Config fields that are assumptions until M3-2 (Q3, Q5)."""


class ImageFlag(StrEnum):
    """Values of ``DetectionImage.flags``."""

    AMPLITUDE_PROCESSED = "amplitude_processed"
    PERIOD_FALLBACK = "period_fallback"
    V_SCALE_ASSUMED = "v_scale_assumed"
    PERIOD_COMPRESSED = "period_compressed"
    SCALE_OUT_OF_RANGE = "scale_out_of_range"
    FLAT_LINE = "flat_line"
    OVERLAP_PERIOD_CAPPED = "overlap_period_capped"
    SINGLE_TILE = "single_tile"
    SHORT_LINE_PADDED = "short_line_padded"


class PeriodSource(StrEnum):
    """Where ``DetectionImage.period_samples`` came from."""

    MEASURED = "measured"
    DATASET_DEFAULT = "dataset_default"
    F_C = "f_c"


class VelocityScaleSource(StrEnum):
    """Where the horizontal-scale velocity ``v_s`` came from."""

    EPS_R_META = "eps_r_meta"
    DEPTH_PER_SAMPLE_ROUGH = "depth_per_sample_rough"
    ASSUMED = "assumed"


# ------------------------------------------------------------------ data types


@dataclass(frozen=True)
class LineScale:
    """Size and scale of one line L1 image (spec section 4-4 item 3).

    Attributes:
        width_px: ``W1``.
        content_height_px: ``H_c`` (rows below it are padding).
        traces_per_px: Actual ``sigma_x = N / W1``.
        samples_per_px: Actual ``sigma_y = n_s / H_c``.
        s_x_nominal: ``s_x`` before rounding (after compression).
        s_y_nominal: ``s_y`` before rounding (after compression).
        compression: ``c``; 1.0 when the content fits the tile height.
    """

    width_px: int
    content_height_px: int
    traces_per_px: float
    samples_per_px: float
    s_x_nominal: float
    s_y_nominal: float
    compression: float


@dataclass(frozen=True)
class LineTile:
    """One tile of a line L1 image (spec section 4-3).

    Attributes:
        index: Tile number, left to right.
        x0_px: Origin in line L1 (integer px).
        content_width_px: ``L``, or ``W1`` when ``W1 <= L`` (the tile image is
            always ``L x L``).
        owner_px: Half-open ``[lo, hi)`` of line L1 columns this tile owns.
        inner_left: The left edge touches a neighbour tile.
        inner_right: The right edge touches a neighbour tile.
    """

    index: int
    x0_px: int
    content_width_px: int
    owner_px: tuple[float, float]
    inner_left: bool
    inner_right: bool

    @property
    def seams_px(self) -> tuple[int, ...]:
        """Line L1 columns of the inner edges of this tile."""
        seams = []
        if self.inner_left:
            seams.append(self.x0_px)
        if self.inner_right:
            seams.append(self.x0_px + self.content_width_px)
        return tuple(seams)

    def owns(self, x_px: float) -> bool:
        """True when line L1 column ``x_px`` lies in ``owner_px``."""
        low, high = self.owner_px
        return low <= x_px < high


@dataclass(frozen=True, eq=False)
class DetectionImage:
    """Line L1 image of one radargram with its tiles (spec section 4-3).

    Attributes:
        line_image: uint8 ``[L, W1]``; rows ``>= content_height_px`` are padding.
        grid_map: Line L1 -> L0 map (offsets 0, ``N / W1``, ``n_s / H_c``).
        tiles: Tiles left to right.
        content_height_px: ``H_c``.
        period_samples: ``P_line`` in samples.
        period_source: :class:`PeriodSource` value.
        depth_m_per_sample: ``delta_z = v_s dt / 2`` (m).
        dx_m: Trace spacing of the line (m).
        v_scale_m_per_ns: ``v_s`` used for the horizontal scale only.
        v_scale_source: :class:`VelocityScaleSource` value.
        p_target_px: ``P*`` used.
        overlap_px: Overlap ``O`` actually used.
        overlap_phys_px: Physical lower bound ``O_phys``.
        compression: ``c`` (1.0 when not compressed).
        flags: :class:`ImageFlag` values.
        assumed_params: Config fields that are M3-2 assumptions.
        pad_gray: Grey of the padding (used by :meth:`tile_image`).
    """

    line_image: np.ndarray
    grid_map: ImageGridMap
    tiles: tuple[LineTile, ...]
    content_height_px: int
    period_samples: float
    period_source: str
    depth_m_per_sample: float
    dx_m: float
    v_scale_m_per_ns: float
    v_scale_source: str
    p_target_px: float
    overlap_px: int
    overlap_phys_px: int
    compression: float
    flags: tuple[str, ...]
    assumed_params: tuple[str, ...]
    pad_gray: int = 128

    @property
    def tile_px(self) -> int:
        """Tile side ``L`` (= image height)."""
        return self.line_image.shape[0]

    @property
    def width_px(self) -> int:
        """Line L1 width ``W1``."""
        return self.line_image.shape[1]

    @property
    def period_l1_px(self) -> float:
        """``P_line / samples_per_px``: the line period in L1 px."""
        return self.period_samples / self.grid_map.samples_per_px

    def tile_image(self, k: int) -> np.ndarray:
        """uint8 ``[L, L]`` C-contiguous copy of tile ``k``.

        Columns ``>= content_width_px`` (short lines only) are ``pad_gray``.
        """
        tile = self.tiles[k]
        image = np.full((self.tile_px, self.tile_px), self.pad_gray, dtype=np.uint8)
        width = tile.content_width_px
        image[:, :width] = self.line_image[:, tile.x0_px : tile.x0_px + width]
        return image


@dataclass(frozen=True)
class _VelocityScale:
    depth_m_per_sample: float
    v_m_per_ns: float
    source: VelocityScaleSource


# ------------------------------------------------------------- private helpers


def _resample_axis0_area(values: np.ndarray, out_length: int) -> np.ndarray:
    """Integrated area mean along axis 0 (``out_length <= input length``)."""
    n_in = values.shape[0]
    prefix = np.zeros((n_in + 1, *values.shape[1:]))
    np.cumsum(values, axis=0, out=prefix[1:])
    # Cell edges j * n / m; written as (j * n) / m so integer ratios stay exact.
    edges = np.arange(out_length + 1) * n_in / out_length
    whole = np.minimum(np.floor(edges).astype(int), n_in - 1)
    fraction = (edges - whole).reshape(-1, *([1] * (values.ndim - 1)))
    cumulative = prefix[whole] + fraction * values[whole]
    return np.diff(cumulative, axis=0) / (n_in / out_length)


def _resample_axis0_linear(values: np.ndarray, out_length: int) -> np.ndarray:
    """Linear interpolation at cell centres along axis 0, ends held."""
    n_in = values.shape[0]
    if n_in == 1:
        return np.repeat(values, out_length, axis=0)
    centres = (np.arange(out_length) + 0.5) * n_in / out_length - 0.5
    centres = np.clip(centres, 0.0, n_in - 1.0)
    lower = np.minimum(np.floor(centres).astype(int), n_in - 2)
    weight = (centres - lower).reshape(-1, *([1] * (values.ndim - 1)))
    return values[lower] * (1.0 - weight) + values[lower + 1] * weight


def _resample_axis0(values: np.ndarray, out_length: int) -> np.ndarray:
    if out_length <= values.shape[0]:
        return _resample_axis0_area(values, out_length)
    return _resample_axis0_linear(values, out_length)


def _detection_section(
    radargram: Radargram, config: DetectionImageConfig
) -> tuple[np.ndarray, list[str]]:
    """Step 1: background-removed section (or DC-removed processed amplitudes)."""
    if radargram.amplitude_processed:
        return remove_dc(radargram.data), [ImageFlag.AMPLITUDE_PROCESSED]
    branch = prepare_fitting_branch(radargram, config.detection_bg_window)
    return branch.background_removed, []


def _line_period(
    radargram: Radargram, section: np.ndarray, config: DetectionImageConfig
) -> tuple[float, PeriodSource]:
    """Step 2: measured period, else the dataset default, else ``1 / f_c``."""
    dt_ns = radargram.dt_ns
    assert dt_ns is not None  # checked by make_detection_image
    measured = measure_period_samples(
        section,
        pad_factor=config.period_pad_factor,
        min_period=config.period_min_samples,
    )
    if measured is not None:
        return measured, PeriodSource.MEASURED
    default = DATASET_DEFAULT_PERIOD.get(radargram.dataset)
    if default is not None:
        value, unit = default
        period = value / dt_ns if unit == "ns" else value
        return period, PeriodSource.DATASET_DEFAULT
    if radargram.f_c_hz is not None:
        return NS_PER_S / radargram.f_c_hz / dt_ns, PeriodSource.F_C
    raise ValueError(
        f"{radargram.line_id}: no period (measurement failed, no dataset "
        f"default for {radargram.dataset!r}, no f_c)"
    )


def _velocity_scale(
    radargram: Radargram, config: DetectionImageConfig
) -> _VelocityScale:
    """Step 3: ``delta_z`` and ``v_s`` from eps_r, the GPR-SD field or the default."""
    dt_ns = radargram.dt_ns
    assert dt_ns is not None  # checked by make_detection_image
    if radargram.eps_r_meta is not None:
        v = velocity_from_eps(radargram.eps_r_meta)
        return _VelocityScale(v * dt_ns / 2.0, v, VelocityScaleSource.EPS_R_META)
    depth_per_sample = radargram.metadata.get(DEPTH_PER_SAMPLE_KEY)
    if depth_per_sample is not None:
        # GPR-SD: use delta_z directly, so both dt hypotheses give the same image.
        depth_m = float(depth_per_sample)
        return _VelocityScale(
            depth_m, 2.0 * depth_m / dt_ns, VelocityScaleSource.DEPTH_PER_SAMPLE_ROUGH
        )
    v = config.v_assumed_m_per_ns
    return _VelocityScale(v * dt_ns / 2.0, v, VelocityScaleSource.ASSUMED)


def _line_gray_image(
    amplitudes: np.ndarray, scale: LineScale, config: DetectionImageConfig
) -> tuple[np.ndarray, bool]:
    """Steps 5-8: clip level, area resample, grey mapping, bottom padding.

    Returns:
        ``(line_image, is_flat)``; a flat line is all ``pad_gray``.
    """
    line_image = np.full(
        (config.tile_px, scale.width_px), config.pad_gray, dtype=np.uint8
    )
    clip_level = float(np.percentile(np.abs(amplitudes), config.clip_percentile))
    if clip_level == 0:
        return line_image, True
    resampled = resample_area(amplitudes, (scale.content_height_px, scale.width_px))
    line_image[: scale.content_height_px, :] = to_gray(resampled, clip_level)
    return line_image, False


# ------------------------------------------------------------------ public API


def measure_period_samples(
    section: np.ndarray, *, pad_factor: int = 8, min_period: float = 4.0
) -> float | None:
    """Dominant period of a section in samples (spec section 4-4 item 2).

    The trace-averaged ``|rfft|`` (zero padded to ``2^ceil(log2(pad n_s))``) of
    the DC-removed traces is searched for its maximum between ``n_s / 4`` and
    ``min_period`` samples.

    Args:
        section: ``[n_samples, n_traces]`` amplitudes.
        pad_factor: Zero-padding factor.
        min_period: Shortest valid period in samples.

    Returns:
        ``1 / f_peak``, or ``None`` when the search band is empty, the spectrum
        is flat (maximum 0) or the maximum sits on an end bin of the band.
    """
    values = np.asarray(section, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    n_samples = values.shape[0]
    values = values - values.mean(axis=0, keepdims=True)
    n_fft = 1 << math.ceil(math.log2(pad_factor * n_samples))
    spectrum = np.abs(np.fft.rfft(values, n_fft, axis=0)).mean(axis=1)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0)
    in_band = np.flatnonzero(
        (frequencies >= 1.0 / (n_samples / 4.0)) & (frequencies <= 1.0 / min_period)
    )
    if len(in_band) == 0:
        return None
    band_spectrum = spectrum[in_band]
    if not band_spectrum.max() > 0:
        return None
    peak = int(np.argmax(band_spectrum))
    if peak in (0, len(in_band) - 1):
        return None
    return 1.0 / float(frequencies[in_band[peak]])


def line_scale(
    n_samples: int,
    n_traces: int,
    dx_m: float,
    period_samples: float,
    depth_m_per_sample: float,
    p_target_px: float,
    tile_px: int = 256,
) -> LineScale:
    """Line L1 size and scale (spec section 4-4 item 3).

    ``s_y = P_line / P*`` matches the period; ``s_x = s_y delta_z / dx`` makes
    a pixel the same depth in both axes. A content taller than ``tile_px`` is
    compressed isotropically by ``c = (n_s / L) / s_y`` instead of being cut.

    Raises:
        ValueError: If a size or scale is not positive.
    """
    for name, value in (
        ("n_samples", n_samples),
        ("n_traces", n_traces),
        ("dx_m", dx_m),
        ("period_samples", period_samples),
        ("depth_m_per_sample", depth_m_per_sample),
        ("p_target_px", p_target_px),
    ):
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value}")
    s_y = period_samples / p_target_px
    s_x = s_y * depth_m_per_sample / dx_m
    compression = 1.0
    if round(n_samples / s_y) > tile_px:
        compression = (n_samples / tile_px) / s_y
        s_y *= compression
        s_x *= compression
    height_px = max(1, round(n_samples / s_y))
    width_px = max(1, round(n_traces / s_x))
    return LineScale(
        width_px=width_px,
        content_height_px=height_px,
        traces_per_px=n_traces / width_px,
        samples_per_px=n_samples / height_px,
        s_x_nominal=s_x,
        s_y_nominal=s_y,
        compression=compression,
    )


def overlap_for(
    traces_per_px: float,
    dx_m: float,
    p_target_px: float,
    config: DetectionImageConfig,
) -> tuple[int, int, bool]:
    """Tile overlap from the physical lower bound and ``O_cv`` (spec 4-4 item 9).

    ``O_phys = max(O_min, ceil(min(2 a_max / (sigma_x dx), w_P P*)))``: a box is
    never wider than the tracking aperture nor than the widest box the
    detector learned (``w_P`` periods; ``lambda / 2 = P*`` px in an isotropic
    image).

    Returns:
        ``(overlap_px, overlap_phys_px, period_capped)``; ``period_capped`` is
        True when the period term is the narrower one.
    """
    tracking_width_px = 2.0 * config.a_max_m / (traces_per_px * dx_m)
    period_width_px = config.overlap_width_periods * p_target_px
    period_capped = period_width_px < tracking_width_px
    widest_box_px = min(tracking_width_px, period_width_px)
    overlap_phys_px = max(
        config.overlap_min_px, math.ceil(widest_box_px - CEIL_TOLERANCE_PX)
    )
    overlap_px = min(config.overlap_max_px, max(overlap_phys_px, config.overlap_px))
    return overlap_px, overlap_phys_px, period_capped


def tile_layout(width_px: int, tile_px: int, overlap_px: int) -> tuple[LineTile, ...]:
    """Tiles of a line L1 image of width ``W1`` (spec section 4-4 item 10).

    Tiles start at ``k S`` (``S = L - O``) while they end inside the line; the
    last tile is aligned to the line end. Each tile owns the columns up to the
    middle of its overlaps.

    Raises:
        ValueError: If ``width_px < 1`` or not ``0 <= overlap_px < tile_px``.
    """
    if width_px < 1:
        raise ValueError(f"width_px must be >= 1, got {width_px}")
    if not 0 <= overlap_px < tile_px:
        raise ValueError(f"need 0 <= overlap_px < tile_px, got {overlap_px}")
    if width_px <= tile_px:
        return (LineTile(0, 0, width_px, (0.0, float(width_px)), False, False),)

    stride_px = tile_px - overlap_px
    origins = []
    k = 0
    while k * stride_px + tile_px < width_px:
        origins.append(k * stride_px)
        k += 1
    origins.append(width_px - tile_px)

    bounds = [0.0]
    for left, right in itertools.pairwise(origins):
        bounds.append((left + tile_px + right) / 2.0)
    bounds.append(float(width_px))
    last = len(origins) - 1
    return tuple(
        LineTile(
            index=k,
            x0_px=x0,
            content_width_px=tile_px,
            owner_px=(bounds[k], bounds[k + 1]),
            inner_left=k > 0,
            inner_right=k < last,
        )
        for k, x0 in enumerate(origins)
    )


def resample_area(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample a 2-D array to ``shape``, time axis first (spec 4-4 item 6).

    Shrinking uses the integrated area mean (``cv2.INTER_AREA`` for integer
    factors); enlarging uses linear interpolation at cell centres.

    Returns:
        float64 array of ``shape``.
    """
    values = np.asarray(a, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"need a 2-D array, got shape {values.shape}")
    height, width = shape
    if height < 1 or width < 1:
        raise ValueError(f"output shape must be positive, got {shape}")
    along_time = _resample_axis0(values, height)
    return _resample_axis0(along_time.T, width).T


def to_gray(a: np.ndarray, clip_level: float) -> np.ndarray:
    """Symmetric grey mapping ``rint(127.5 + 127.5 clip(a / c, -1, 1))`` (uint8).

    Zero amplitude becomes 128 (spec section 4-4 item 7).

    Raises:
        ValueError: If ``clip_level`` is not positive.
    """
    if not clip_level > 0:
        raise ValueError(f"clip_level must be positive, got {clip_level}")
    scaled = np.clip(np.asarray(a, dtype=np.float64) / clip_level, -1.0, 1.0)
    return np.rint(GRAY_CENTER + GRAY_CENTER * scaled).astype(np.uint8)


def make_detection_image(
    radargram: Radargram,
    config: DetectionImageConfig = DEFAULT_DETECTION_IMAGE_CONFIG,
) -> DetectionImage:
    """Line L1 image and tiles of one radargram (spec section 4-4).

    Steps (1)-(6) run once on the whole line (L0), so a reflection in an
    overlap has the same pixels in both tiles.

    Raises:
        ValueError: If the radargram has no physical scale, or no period can
            be found.
    """
    if radargram.dt_ns is None or radargram.dx_m is None:
        raise ValueError(
            "DetectionImage needs a physical scale; use ImageGridMap.identity"
        )
    section, flags = _detection_section(radargram, config)
    period_samples, period_source = _line_period(radargram, section, config)
    if period_source != PeriodSource.MEASURED:
        flags.append(ImageFlag.PERIOD_FALLBACK)
    velocity = _velocity_scale(radargram, config)
    if velocity.source == VelocityScaleSource.ASSUMED:
        flags.append(ImageFlag.V_SCALE_ASSUMED)

    scale = line_scale(
        radargram.n_samples,
        radargram.n_traces,
        radargram.dx_m,
        period_samples,
        velocity.depth_m_per_sample,
        config.p_target_px,
        config.tile_px,
    )
    if scale.compression != 1.0:
        flags.append(ImageFlag.PERIOD_COMPRESSED)
        if scale.compression > config.compress_max:
            flags.append(ImageFlag.SCALE_OUT_OF_RANGE)

    if radargram.amplitude_processed:
        amplitudes = section
    else:
        amplitudes = agc(section, radargram.dt_ns, radargram.f_c_hz)
    line_image, is_flat = _line_gray_image(amplitudes, scale, config)
    if is_flat:
        flags.append(ImageFlag.FLAT_LINE)

    overlap_px, overlap_phys_px, period_capped = overlap_for(
        scale.traces_per_px, radargram.dx_m, config.p_target_px, config
    )
    if period_capped:
        flags.append(ImageFlag.OVERLAP_PERIOD_CAPPED)
    tiles = tile_layout(scale.width_px, config.tile_px, overlap_px)
    if scale.width_px <= config.tile_px:
        flags.append(ImageFlag.SINGLE_TILE)
        if scale.width_px < config.tile_px:
            flags.append(ImageFlag.SHORT_LINE_PADDED)

    return DetectionImage(
        line_image=line_image,
        grid_map=ImageGridMap(
            trace_offset=0,
            sample_offset=0,
            traces_per_px=scale.traces_per_px,
            samples_per_px=scale.samples_per_px,
            n_traces=radargram.n_traces,
            n_samples=radargram.n_samples,
        ),
        tiles=tiles,
        content_height_px=scale.content_height_px,
        period_samples=period_samples,
        period_source=period_source,
        depth_m_per_sample=velocity.depth_m_per_sample,
        dx_m=radargram.dx_m,
        v_scale_m_per_ns=velocity.v_m_per_ns,
        v_scale_source=velocity.source,
        p_target_px=config.p_target_px,
        overlap_px=overlap_px,
        overlap_phys_px=overlap_phys_px,
        compression=scale.compression,
        flags=tuple(flags),
        assumed_params=M3_2_ASSUMED,
        pad_gray=config.pad_gray,
    )
