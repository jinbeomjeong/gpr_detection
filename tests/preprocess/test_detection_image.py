"""Line L1 detection image (spec_detection_image_v1 v0.3.4 section 8-1, T2-T7).

Expected values of T3 are the printed output of
``analysis/processing/detection_image_expected_values.py`` (the spec's
canonical source, section 4-7). The real-data part of T7 (e) lives in
``tests/test_detection_image_data.py``.
"""

import csv
import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import hilbert

from gpr_detection.detect.base import PixelBox, letterbox
from gpr_detection.io.radargram import SPEED_OF_LIGHT_M_PER_NS, Radargram
from gpr_detection.pipeline import image_box_to_grid_box
from gpr_detection.preprocess import to_image
from gpr_detection.preprocess.filters import prepare_fitting_branch
from gpr_detection.preprocess.to_image import (
    M3_2_ASSUMED,
    DetectionImageConfig,
    line_scale,
    make_detection_image,
    measure_period_samples,
    overlap_for,
    resample_area,
    tile_layout,
    to_gray,
)
from synthetic_bscan import BScanSpec, make_radargram

SEED = 20261113
TILE_PX = 256
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOURTU_LINES_CSV = PROJECT_ROOT / "docs" / "processing" / "fourtu_lines_v1.csv"
FOURTU_DX_M = 0.02
FOURTU_PERIOD_NS = 2.51
V_EPS9 = SPEED_OF_LIGHT_M_PER_NS / 3.0
REL_TOLERANCE = 1e-9
GRAY_ZERO = 127.5
"""Grey of zero amplitude, removed before measuring a period in the image."""

# (n_s, N, dx, P_line, delta_z, P*) in the precise form of spec 4-7.
SCALE_CASE_INPUTS = {
    "4tu_01.1_path1": (512, 304, 0.02, 2.51 / 0.097, 0.097 * V_EPS9 / 2, 4.06),
    "4tu_09.1_path2": (512, 12186, 0.02, 2.51 / 0.195, 0.195 * V_EPS9 / 2, 4.06),
    "4tu_09.1_path2_p8": (512, 12186, 0.02, 2.51 / 0.195, 0.195 * V_EPS9 / 2, 8.0),
    "gz_1030_30ns": (512, 710, 0.024, 2.93 / (30 / 512), (30 / 512) * 0.10 / 2, 4.06),
    "gprsd_cementroad2_1": (321, 9411, 0.03, 51.2, 0.012, 4.06),
    "4tu_03.9_path5": (512, 9, 0.02, 2.51 / 0.098, 0.098 * V_EPS9 / 2, 4.06),
}
# Printed by detection_image_expected_values.py:
# (s_y 4 dp, s_x 4 dp, H_c, W1, sigma_x 5 dp, sigma_y 5 dp, compressed,
#  O_phys, tiles at O_cv 128, tiles at O_cv 192)
SCALE_CASE_EXPECTED = {
    "4tu_01.1_path1": (6.3735, 1.5445, 80, 197, 1.54315, 6.4, False, 65, 1, 1),
    "4tu_09.1_path2": (3.1704, 1.5445, 161, 7890, 1.54449, 3.18012, False, 65, 61, 121),
    "4tu_09.1_path2_p8": (2.0, 0.9743, 256, 12507, 0.97433, 2.0, True, 103, 97, 193),
    "gz_1030_30ns": (12.3166, 1.5035, 42, 472, 1.50424, 12.19048, False, 64, 3, 5),
    "gprsd_cementroad2_1": (
        12.6108,
        5.0443,
        25,
        1866,
        5.04341,
        12.84,
        False,
        64,
        14,
        27,
    ),
    "4tu_03.9_path5": (6.3084, 1.5445, 81, 6, 1.5, 6.32099, False, 67, 1, 1),
}

T2_WIDTHS = (1, 97, 255, 256, 257, 300, 383, 512, 2250, 14516, 27628)
T2_OVERLAPS = (0, 64, 128, 141, 192)
T2_EXPECTED_TILES = {
    1: (1, 1, 1, 1, 1),
    97: (1, 1, 1, 1, 1),
    255: (1, 1, 1, 1, 1),
    256: (1, 1, 1, 1, 1),
    257: (2, 2, 2, 2, 2),
    300: (2, 2, 2, 2, 2),
    383: (2, 2, 2, 3, 3),
    512: (2, 3, 3, 4, 5),
    2250: (9, 12, 17, 19, 33),
    14516: (57, 76, 113, 125, 224),
    27628: (108, 144, 215, 240, 429),
}


def _scale_radargram(
    n_samples: int,
    n_traces: int,
    dt_ns: float,
    dx_m: float,
    **fields,
) -> Radargram:
    """Random-noise radargram of a given grid (period supplied by monkeypatch)."""
    rng = np.random.default_rng(SEED)
    data = rng.normal(0.0, 1.0, (n_samples, n_traces)).astype(np.float32)
    defaults = {
        "air_gap": "ground",
        "amplitude_processed": False,
        "dataset": "unit",
        "line_id": "unit/line",
        "f_c_hz": 400e6,
    }
    defaults.update(fields)
    return Radargram(data=data, dt_ns=dt_ns, dx_m=dx_m, **defaults)


def _fixed_period(monkeypatch: pytest.MonkeyPatch, period_samples: float) -> None:
    monkeypatch.setattr(
        to_image, "measure_period_samples", lambda *args, **kwargs: period_samples
    )


# ------------------------------------------------------------------ T2 (C1)


@pytest.mark.parametrize("width_px", T2_WIDTHS)
def test_t2_tile_layout_counts_and_ownership(width_px: int) -> None:
    """T2 ([CV] C1): tile counts, coverage, ownership, widths <= O fit a tile."""
    for overlap_px, expected in zip(
        T2_OVERLAPS, T2_EXPECTED_TILES[width_px], strict=True
    ):
        tiles = tile_layout(width_px, TILE_PX, overlap_px)
        assert len(tiles) == expected, (width_px, overlap_px)

        # Owner intervals partition [0, W1) without gaps or overlaps.
        assert tiles[0].owner_px[0] == 0.0
        assert tiles[-1].owner_px[1] == float(width_px)
        for left, right in itertools.pairwise(tiles):
            assert left.owner_px[1] == right.owner_px[0]
        # Every column is covered; the last tile ends at W1.
        last = tiles[-1]
        assert last.x0_px + last.content_width_px == width_px
        covered = np.zeros(width_px, dtype=bool)
        for tile in tiles:
            covered[tile.x0_px : tile.x0_px + tile.content_width_px] = True
        assert covered.all()

        if width_px <= TILE_PX:
            assert tiles[0].seams_px == ()
            continue
        assert [t.inner_left for t in tiles] == [False] + [True] * (len(tiles) - 1)
        assert [t.inner_right for t in tiles] == [True] * (len(tiles) - 1) + [False]
        if overlap_px > 0:
            _assert_intervals_of_width_fit(tiles, width_px, overlap_px)


def _assert_intervals_of_width_fit(tiles, width_px: int, width: int) -> None:
    """Every integer interval [x, x + width) lies inside some tile."""
    origins = np.array([tile.x0_px for tile in tiles])
    starts = np.arange(0, width_px - width + 1)
    # The tile with the largest origin <= x is the best candidate.
    candidate = np.searchsorted(origins, starts, side="right") - 1
    assert np.all(starts + width <= origins[candidate] + TILE_PX)


def test_t2_short_line_single_tile_flags() -> None:
    (tile,) = tile_layout(97, TILE_PX, 192)
    assert (tile.x0_px, tile.content_width_px, tile.owner_px) == (0, 97, (0.0, 97.0))
    assert not tile.inner_left and not tile.inner_right


def test_t2_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_px"):
        tile_layout(1000, TILE_PX, TILE_PX)


# ----------------------------------------------------------------------- T3


@pytest.mark.parametrize("case", sorted(SCALE_CASE_INPUTS))
def test_t3_line_scale_matches_expected_values(case: str) -> None:
    """T3: line_scale / overlap_for / tile_layout vs the spec 4-7 table."""
    n_s, n_traces, dx_m, period, depth_per_sample, p_star = SCALE_CASE_INPUTS[case]
    (s_y, s_x, height, width, sigma_x, sigma_y, compressed, o_phys, tiles_128,
     tiles_192) = SCALE_CASE_EXPECTED[case]  # fmt: skip

    scale = line_scale(n_s, n_traces, dx_m, period, depth_per_sample, p_star)

    assert (scale.content_height_px, scale.width_px) == (height, width)
    assert round(scale.s_y_nominal, 4) == s_y
    assert round(scale.s_x_nominal, 4) == s_x
    assert round(scale.traces_per_px, 5) == sigma_x
    assert round(scale.samples_per_px, 5) == sigma_y
    assert (scale.compression != 1.0) == compressed
    assert scale.traces_per_px == pytest.approx(n_traces / width, rel=REL_TOLERANCE)
    assert scale.samples_per_px == pytest.approx(n_s / height, rel=REL_TOLERANCE)
    # Closed form (spec 4-4 item 3) with the precise inputs.
    nominal_s_y = period / p_star * scale.compression
    assert scale.s_y_nominal == pytest.approx(nominal_s_y, rel=REL_TOLERANCE)
    assert scale.s_x_nominal == pytest.approx(
        nominal_s_y * depth_per_sample / dx_m, rel=REL_TOLERANCE
    )

    for overlap_cv, expected_tiles in ((128, tiles_128), (192, tiles_192)):
        config = DetectionImageConfig(p_target_px=p_star, overlap_px=overlap_cv)
        overlap, overlap_phys, _ = overlap_for(
            scale.traces_per_px, dx_m, p_star, config
        )
        assert overlap_phys == o_phys
        assert len(tile_layout(width, TILE_PX, overlap)) == expected_tiles


# "P* 6.0 (v0.4 default)" lines of detection_image_expected_values.py:
# (s_y 4 dp, s_x 4 dp, H_c, W1, sigma_x 5 dp, sigma_y 5 dp, O_phys, tiles at 192)
DEFAULT_SCALE_CASE_EXPECTED = {
    "4tu_01.1_path1": (4.3127, 1.0451, 119, 291, 1.04467, 4.30252, 96, 2),
    "4tu_09.1_path2": (2.1453, 1.0451, 239, 11660, 1.04511, 2.14226, 96, 180),
    "gz_1030_30ns": (8.3342, 1.0174, 61, 698, 1.01719, 8.39344, 82, 8),
    "gprsd_cementroad2_1": (8.5333, 3.4133, 38, 2757, 3.41349, 8.44737, 64, 41),
    "4tu_03.9_path5": (4.2687, 1.0451, 120, 9, 1.0, 4.26667, 100, 1),
}


@pytest.mark.parametrize("case", sorted(DEFAULT_SCALE_CASE_EXPECTED))
def test_t3_default_p_target_6_matches_expected_values(case: str) -> None:
    """T3 (v0.4): the spec 4-7 cases with the default config (P* 6.0, O 192)."""
    n_s, n_traces, dx_m, period, depth_per_sample, _ = SCALE_CASE_INPUTS[case]
    s_y, s_x, height, width, sigma_x, sigma_y, o_phys, tiles = (
        DEFAULT_SCALE_CASE_EXPECTED[case]
    )
    config = DetectionImageConfig()
    p_star = config.p_target_px

    scale = line_scale(n_s, n_traces, dx_m, period, depth_per_sample, p_star)

    assert (scale.content_height_px, scale.width_px) == (height, width)
    assert round(scale.s_y_nominal, 4) == s_y
    assert round(scale.s_x_nominal, 4) == s_x
    assert round(scale.traces_per_px, 5) == sigma_x
    assert round(scale.samples_per_px, 5) == sigma_y
    assert scale.compression == 1.0
    overlap, overlap_phys, _ = overlap_for(scale.traces_per_px, dx_m, p_star, config)
    assert overlap_phys == o_phys
    assert overlap == max(o_phys, 192)
    assert len(tile_layout(width, TILE_PX, overlap)) == tiles


@pytest.mark.parametrize("case", sorted(SCALE_CASE_INPUTS))
def test_t3_vertical_and_horizontal_lines_land_on_their_cells(case: str) -> None:
    """T3: a line at trace i (sample k) peaks in an L1 column (row) mapping to i+-1."""
    n_s, n_traces, dx_m, period, depth_per_sample, p_star = SCALE_CASE_INPUTS[case]
    scale = line_scale(n_s, n_traces, dx_m, period, depth_per_sample, p_star)
    rng = np.random.default_rng(SEED)
    shape = (scale.content_height_px, scale.width_px)
    for trace in rng.integers(0, n_traces, 3):
        section = np.zeros((n_s, n_traces))
        section[:, trace] = 1.0
        column = int(np.argmax(resample_area(section, shape).max(axis=0)))
        assert _cell_holds(column, scale.traces_per_px, int(trace))
    for sample in rng.integers(0, n_s, 3):
        section = np.zeros((n_s, n_traces))
        section[sample, :] = 1.0
        row = int(np.argmax(resample_area(section, shape).max(axis=1)))
        assert _cell_holds(row, scale.samples_per_px, int(sample))


def _cell_holds(cell: int, grid_per_px: float, index: int) -> bool:
    """L1 cell ``[cell, cell + 1)`` maps to L0 ``[cell s, (cell + 1) s)``; +-1 index.

    For ``s < 2`` (the trace axis of most cases) this is the spec form
    ``floor(u s) in {i - 1, i, i + 1}``; for the coarse sample axis
    (``s`` 3-13) the whole cell span is the meaningful comparison.
    """
    return cell * grid_per_px - 1 <= index < (cell + 1) * grid_per_px + 1


# ----------------------------------------------------------------------- T4


T4_SNR_CASES = (None, 20.0)
T4_APEX_SKIP_SAMPLES = 60
"""Direct-wave rows skipped when locating the synthetic apex (spec T4 (b))."""
T4_EXPECTED_WINDOW_PERIOD_PX = {
    (None, 4.06): 4.06,
    (None, 6.0): 5.95,
    (20.0, 4.06): 4.41,
    (20.0, 6.0): 6.48,
}
"""Printed by analysis/processing/detection_image_t4_check.py (reference only)."""


def _t4_radargram(snr_db: float | None) -> Radargram:
    spec = BScanSpec(
        f_c_hz=500e6,
        dt_ns=0.098,
        n_traces=401,
        n_samples=512,
        air_gap_m=0,
        snr_db=snr_db,
    )
    return make_radargram(spec)


def _t4_apex(radargram: Radargram) -> tuple[int, int]:
    """Centre trace and its envelope maximum after (1)-(4), first rows skipped."""
    section = prepare_fitting_branch(radargram).background_removed
    centre = section.shape[1] // 2
    envelope = np.abs(hilbert(section[:, centre]))
    envelope[:T4_APEX_SKIP_SAMPLES] = 0
    return centre, int(np.argmax(envelope))


def _apex_window_period(image, apex: tuple[int, int]) -> float | None:
    """Spec T4 (b): rows r - 2P* .. r + 4P*, columns c +- 7, grey - 127.5."""
    trace, sample = apex
    row = sample / image.grid_map.samples_per_px
    column = trace / image.grid_map.traces_per_px
    p_star = image.p_target_px
    rows = slice(max(0, int(row - 2 * p_star)), int(row + 4 * p_star) + 1)
    columns = slice(max(0, int(column) - 7), int(column) + 8)
    window = image.line_image[rows, columns].astype(np.float64) - GRAY_ZERO
    return measure_period_samples(window, min_period=2.0)


@pytest.mark.parametrize("snr_db", T4_SNR_CASES)
@pytest.mark.parametrize("p_target_px", [4.06, 6.0])
def test_t4a_line_period_is_normalised_to_target(
    p_target_px: float, snr_db: float | None
) -> None:
    """T4 (a): ``period_l1_px`` is P* within +-1 %."""
    image = make_detection_image(
        _t4_radargram(snr_db), DetectionImageConfig(p_target_px=p_target_px)
    )
    assert image.period_source == "measured"
    assert image.period_l1_px == pytest.approx(p_target_px, rel=0.01)


@pytest.mark.parametrize("snr_db", T4_SNR_CASES)
@pytest.mark.parametrize("p_target_px", [4.06, 6.0])
def test_t4b_apex_window_period_is_target(
    p_target_px: float, snr_db: float | None
) -> None:
    """T4 (b) (v0.3.5): the period in the apex window is P* within +-10 %.

    The whole-content period is not judged (AGC lifts the noise rows, R2).
    """
    radargram = _t4_radargram(snr_db)
    image = make_detection_image(
        radargram, DetectionImageConfig(p_target_px=p_target_px)
    )
    period = _apex_window_period(image, _t4_apex(radargram))
    assert period == pytest.approx(p_target_px, rel=0.10)
    expected = T4_EXPECTED_WINDOW_PERIOD_PX[(snr_db, p_target_px)]
    assert period == pytest.approx(expected, abs=0.01)


def test_t4_measure_period_samples_edge_cases() -> None:
    n_samples = 512
    samples = np.arange(n_samples)[:, None]
    sine = np.sin(2 * np.pi * samples / 20.0) * np.ones((1, 30))
    assert measure_period_samples(sine) == pytest.approx(20.0, rel=0.01)
    assert measure_period_samples(np.zeros((n_samples, 4))) is None
    # A ramp's spectrum falls with frequency: the maximum is the low band end.
    ramp = samples * np.ones((1, 4), dtype=float)
    assert measure_period_samples(ramp) is None
    # Too short to have a band at all (n_s / 4 < min_period).
    assert measure_period_samples(np.ones((10, 2))) is None


# ----------------------------------------------------------------------- T5


@pytest.mark.parametrize("factor", [3.0, 6.3, 12.6])
def test_t5_area_resample_suppresses_aliasing(factor: float) -> None:
    """T5: 1.5x new-Nyquist sine is attenuated, the carrier is kept.

    ``round(4096 / s)`` output samples make the factor non-integer, so the
    ratio sits a little below the box response 0.333 / 0.307 / 0.302.
    """
    n_in = 4096
    n_out = round(n_in / factor)
    samples = np.arange(n_in)[:, None]

    def rms_ratio(cycles_per_sample: float) -> float:
        signal = np.sin(2 * np.pi * cycles_per_sample * samples)
        out = resample_area(signal, (n_out, 1))
        return float(np.sqrt(np.mean(out**2)) / np.sqrt(np.mean(signal**2)))

    assert rms_ratio(0.75 / factor) <= 0.36
    assert rms_ratio(0.246 / factor) >= 0.85


def test_t5_resample_area_integer_factor_is_block_mean() -> None:
    values = np.arange(24, dtype=float).reshape(6, 4)
    out = resample_area(values, (3, 2))
    expected = values.reshape(3, 2, 2, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)


def test_t5_resample_upsampling_interpolates_and_holds_ends() -> None:
    values = np.array([[0.0], [1.0]])
    out = resample_area(values, (4, 1))
    np.testing.assert_allclose(out[:, 0], [0.0, 0.25, 0.75, 1.0])


def test_to_gray_symmetric_mapping() -> None:
    gray = to_gray(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), clip_level=1.0)
    np.testing.assert_array_equal(gray, [0, 0, 128, 255, 255])
    assert gray.dtype == np.uint8
    with pytest.raises(ValueError, match="clip_level"):
        to_gray(np.zeros(3), 0.0)


# ----------------------------------------------------------------------- T6


@pytest.mark.parametrize("n_traces", [9, 401, 2000])
def test_t6_every_tile_is_letterbox_identity(n_traces: int) -> None:
    """T6 ([CV] choice (c)): tiles are 256 x 256 and the letterbox is identity."""
    radargram = make_radargram(BScanSpec(n_traces=n_traces, n_samples=512))
    image = make_detection_image(radargram)

    for tile in image.tiles:
        tile_image = image.tile_image(tile.index)
        assert tile_image.shape == (TILE_PX, TILE_PX)
        assert tile_image.flags["C_CONTIGUOUS"]
        _, transform = letterbox(tile_image, TILE_PX)
        assert transform.scale_x == transform.scale_y == 1.0
        assert transform.pad_x == transform.pad_y == 0.0

    if image.width_px < TILE_PX:
        tile_image = image.tile_image(0)
        assert np.all(tile_image[:, image.width_px :] == image.pad_gray)
        pad_box = PixelBox(image.width_px + 1.0, 10.0, TILE_PX - 1.0, 40.0)
        assert image_box_to_grid_box(pad_box, image.grid_map) is None
    # Rows below the content are padding as well.
    assert np.all(image.line_image[image.content_height_px :] == image.pad_gray)


# ----------------------------------------------------------------------- T7


def test_t7a_no_scale_is_rejected() -> None:
    radargram = make_radargram(BScanSpec(n_traces=20), dt_ns=None)
    with pytest.raises(ValueError, match="physical scale"):
        make_detection_image(radargram)


def test_t7b_nine_trace_line_is_one_padded_tile() -> None:
    n_s, n_traces, dx_m, period, depth_per_sample, p_star = SCALE_CASE_INPUTS[
        "4tu_03.9_path5"
    ]
    assert (
        line_scale(n_s, n_traces, dx_m, period, depth_per_sample, p_star).width_px == 6
    )

    radargram = make_radargram(BScanSpec(n_traces=9, n_samples=512, dt_ns=0.098))
    image = make_detection_image(radargram)
    assert len(image.tiles) == 1
    assert {"single_tile", "short_line_padded"} <= set(image.flags)
    assert image.assumed_params == M3_2_ASSUMED


COMPRESS_MAX_V04 = 1.35
"""Spec v0.4 section 11-2 item 5; asserted as a constant so a default change fails."""


def test_t7c_default_config_values() -> None:
    config = DetectionImageConfig()
    assert config.compress_max == COMPRESS_MAX_V04
    assert (config.p_target_px, config.overlap_px) == (6.0, 192)


def test_t7c_compression_within_detector_scale(monkeypatch) -> None:
    """P* 8, 4TU dt 0.195 type: compressed by 1.243 <= 1.35."""
    _fixed_period(monkeypatch, 2.51 / 0.195)
    radargram = _scale_radargram(512, 401, 0.195, 0.02, eps_r_meta=9.0)
    image = make_detection_image(radargram, DetectionImageConfig(p_target_px=8.0))
    assert image.content_height_px == TILE_PX
    assert image.compression == pytest.approx(1.243, abs=5e-4)
    assert "period_compressed" in image.flags
    assert "scale_out_of_range" not in image.flags


def _mala_radargram() -> Radargram:
    return _scale_radargram(
        512, 208, 0.391, 0.0256, amplitude_processed=True, dataset="guangzhou"
    )


@pytest.mark.parametrize(("p_star", "expected_c"), [(4.06, 1.825), (6.0, 2.697)])
def test_t7c_mala_compression_out_of_detector_scale(
    monkeypatch, p_star: float, expected_c: float
) -> None:
    """MALA type (P 4.45 samples): c 1.825 at P* 4.06, 2.697 at the default 6.0."""
    _fixed_period(monkeypatch, 4.45)
    config = DetectionImageConfig(p_target_px=p_star)
    image = make_detection_image(_mala_radargram(), config)
    assert image.compression == pytest.approx(expected_c, abs=5e-4)
    assert {"period_compressed", "scale_out_of_range"} <= set(image.flags)
    assert {"amplitude_processed", "v_scale_assumed"} <= set(image.flags)
    assert image.v_scale_source == "assumed"


@pytest.mark.parametrize(
    ("compression", "is_out_of_range"), [(1.30, False), (1.40, True)]
)
def test_t7c_compress_max_boundary(
    monkeypatch, compression: float, is_out_of_range: bool
) -> None:
    """Default config: c 1.30 -> compressed only, c 1.40 -> out of range too."""
    p_star = DetectionImageConfig().p_target_px
    # c = (n_s / L) / (P / P*) -> P = (n_s / L) P* / c
    _fixed_period(monkeypatch, (512 / TILE_PX) * p_star / compression)
    image = make_detection_image(_mala_radargram())
    assert image.compression == pytest.approx(compression, rel=1e-9)
    assert "period_compressed" in image.flags
    assert ("scale_out_of_range" in image.flags) == is_out_of_range


def test_t7d_flat_line_is_all_pad_gray() -> None:
    radargram = Radargram(
        data=np.zeros((512, 50), dtype=np.float32),
        dt_ns=0.1,
        dx_m=0.02,
        air_gap="ground",
        amplitude_processed=False,
        dataset="unit",
        line_id="unit/flat",
        f_c_hz=400e6,
    )
    image = make_detection_image(radargram)
    assert "flat_line" in image.flags
    assert image.period_source == "f_c"
    assert "period_fallback" in image.flags
    assert np.all(image.line_image == image.pad_gray)


def test_t7_period_fallback_uses_dataset_default(monkeypatch) -> None:
    monkeypatch.setattr(to_image, "measure_period_samples", lambda *a, **k: None)
    radargram = _scale_radargram(512, 100, 0.098, 0.02, dataset="4tu", eps_r_meta=9.0)
    image = make_detection_image(radargram)
    assert image.period_source == "dataset_default"
    assert image.period_samples == pytest.approx(2.51 / 0.098)
    no_default = _scale_radargram(512, 100, 0.098, 0.02, f_c_hz=None)
    with pytest.raises(ValueError, match="no period"):
        make_detection_image(no_default)


T7F_CASES = [
    # (dx, P_line, delta_z, P*, O_phys, capped); expected_values.py "T7f" lines
    (0.01, 12.82, 0.00195, 4.06, 82, True),  # high frequency (Guangzhou 2000 MHz)
    (0.02, 25.61, 0.004897, 4.06, 65, False),  # 4TU eps_r 9
    (0.02, 25.61, 0.003526, 4.06, 82, True),  # 4TU eps_r 17.36
    (0.01, 12.82, 0.00195, 6.0, 120, True),
    (0.02, 25.61, 0.004897, 6.0, 96, False),
    (0.02, 25.61, 0.003526, 6.0, 120, True),
]


@pytest.mark.parametrize("overlap_cv", [192, 128])
def test_t7f_overlap_period_cap(overlap_cv: int) -> None:
    """T7 (f): the 20-period cap sets O_phys; the overlap stays O_cv."""
    for (
        dx_m,
        period,
        depth_per_sample,
        p_star,
        expected_phys,
        expected_capped,
    ) in T7F_CASES:
        config = DetectionImageConfig(p_target_px=p_star, overlap_px=overlap_cv)
        scale = line_scale(512, 1000, dx_m, period, depth_per_sample, p_star)
        overlap, overlap_phys, capped = overlap_for(
            scale.traces_per_px, dx_m, p_star, config
        )
        assert (overlap, overlap_phys, capped) == (
            overlap_cv,
            expected_phys,
            expected_capped,
        ), (p_star, dx_m, depth_per_sample)


@pytest.mark.parametrize(("p_star", "expected_phys"), [(4.06, 82), (6.0, 120)])
def test_t7f_capped_flag_on_image(monkeypatch, p_star: float, expected_phys: int):
    _fixed_period(monkeypatch, 12.82)
    radargram = _scale_radargram(512, 1000, 0.039, 0.01, dataset="guangzhou")
    image = make_detection_image(radargram, DetectionImageConfig(p_target_px=p_star))
    assert image.depth_m_per_sample == pytest.approx(0.00195)
    assert image.overlap_phys_px == expected_phys
    assert image.overlap_px == 192
    assert "overlap_period_capped" in image.flags


def _fourtu_rows() -> list[dict[str, str]]:
    with FOURTU_LINES_CSV.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize(("p_star", "expected_capped"), [(4.06, 92), (6.0, 91)])
def test_t7g_fourtu_metadata_capped_count(p_star: float, expected_capped: int):
    """T7 (g): 759 4TU lines, P 2.51 ns and metadata eps_r (no raw data)."""
    rows = _fourtu_rows()
    assert len(rows) == 759
    for overlap_cv in (192, 128):
        config = DetectionImageConfig(p_target_px=p_star, overlap_px=overlap_cv)
        capped = 0
        for row in rows:
            dt_ns = float(row["dt_ns"])
            v = SPEED_OF_LIGHT_M_PER_NS / math.sqrt(float(row["eps_r_used"]))
            scale = line_scale(
                int(row["n_samples"]),
                int(row["n_traces"]),
                FOURTU_DX_M,
                FOURTU_PERIOD_NS / dt_ns,
                v * dt_ns / 2,
                p_star,
            )
            overlap, _, is_capped = overlap_for(
                scale.traces_per_px, FOURTU_DX_M, p_star, config
            )
            assert overlap == overlap_cv
            capped += is_capped
        assert capped == expected_capped


def test_gprsd_depth_per_sample_does_not_depend_on_dt(monkeypatch) -> None:
    """GPR-SD scale uses delta_z only, so dt hypotheses A and B give one image."""
    _fixed_period(monkeypatch, 51.2)
    images = []
    for dt_ns in (0.3022, 0.2553):
        radargram = _scale_radargram(
            321,
            600,
            dt_ns,
            0.03,
            amplitude_processed=True,
            dataset="gprsd",
            f_c_hz=None,
            metadata={"depth_per_sample_m_rough": 0.012},
        )
        images.append(make_detection_image(radargram))
    assert images[0].v_scale_source == "depth_per_sample_rough"
    np.testing.assert_array_equal(images[0].line_image, images[1].line_image)
    assert images[0].grid_map == images[1].grid_map


def test_config_rejects_unknown_gray_mapping() -> None:
    with pytest.raises(ValueError, match="gray_mapping"):
        DetectionImageConfig(gray_mapping="log")
