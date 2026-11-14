"""M3-2 S1 recipe options of the line L1 image (plan S6 (a)-(f), S1 share).

``docs/processing/m32_image_quality_plan.md`` v0.2.1 section 5-3: the options
B0-B4 (background), G0-G4 (gain), P0-P1 (padding) and M0-M3 (grey mapping) of
``DetectionImageConfig``. The plan's recipe table r0-r7 is not part of ``src``
(S5); the option sets below only name the plan symbols they stand for.

* (a) The default configuration (r0) gives the M3-1 images bit for bit. The
  sha256 values were recorded from commit a1a9895 (before S1) in env
  ``gpr-dev``; numpy / scipy upgrades may change them legitimately.
* (b)-(e) identities between the options.
* (f) Period, scale, ``ImageGridMap``, content range and tiles do not depend
  on the detection-branch options (the period section is fixed by
  ``period_bg_window``).

Real-data parts are opt-in like T7 (e): ``pytest -m realdata``.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from gpr_detection.io.fourtu import load_fourtu_radargram
from gpr_detection.io.gprsd import DtHypothesis, load_gprsd_radargram
from gpr_detection.io.guangzhou import load_guangzhou_radargram
from gpr_detection.io.paths import DatasetName, DatasetPaths, load_dataset_paths
from gpr_detection.io.radargram import Radargram
from gpr_detection.preprocess import to_image
from gpr_detection.preprocess.filters import agc, agc_rms, prepare_fitting_branch
from gpr_detection.preprocess.to_image import (
    M3_2_ASSUMED,
    DetectionBackgroundMode,
    DetectionImage,
    DetectionImageConfig,
    GainMode,
    GrayMapping,
    ImageFlag,
    PadMode,
    background_gate_weights,
    make_detection_image,
)
from synthetic_bscan import BScanSpec, make_radargram

GRAY_LEVELS = 256
TAPER_END_TOLERANCE_GRAY = 3
"""Last content row of a P1 taper: within this many grey levels of ``pad_gray``.

The row centre sits half a pixel above the content end, where the
``sin^2`` weight is ``sin^2(pi / (4 L_t))`` (0.017 for ``L_t`` 6 px).
"""

SYNTHETIC_SPECS = {
    "ground_401": BScanSpec(n_traces=401, n_samples=512, dt_ns=0.098, f_c_hz=500e6),
    "ground_301_dt0195": BScanSpec(n_traces=301, n_samples=512, dt_ns=0.195),
    "ground_900_noise": BScanSpec(n_traces=900, n_samples=700, snr_db=10.0),
}
SYNTHETIC_SHA256 = {
    "ground_401": "86341f946c7089679253c629e482cda3ddaf6b542dc4f7eaa58d36a1b2875db1",
    "ground_301_dt0195": (
        "407cf8ab1ed682fa5f7654d1ce7c5a930f82bda5312cdd252cca45d296767c1f"
    ),
    "ground_900_noise": (
        "08a54ea46d3075d6f4cd8499d6bacce662017c2025d301b73840cad7ea14e0c5"
    ),
}

# One option set per plan symbol that changes the detection branch (section 2).
OPTION_SETS: dict[str, dict[str, object]] = {
    "B1": {"detection_bg_mode": "none"},
    "B2_a0.5": {"detection_bg_mode": "partial", "detection_bg_alpha": 0.5},
    "B3_ng2": {"detection_bg_mode": "gated", "detection_bg_gate_periods": 2.0},
    "B4": {"detection_bg_window": 1501},
    "G1": {"agc_window_periods": 32.0},
    "G2_b0.5": {"gain_mode": "agc_capped", "agc_floor_beta": 0.5},
    "G3": {"gain_mode": "tpow", "tpow_exponent": 1.5},
    "G4": {"gain_mode": "none"},
    "P1_nt1": {"pad_mode": "taper", "pad_taper_periods": 1.0},
    "M1_p98": {"clip_percentile": 98.0},
    "M2": {"gray_mapping": "hist_aihub", "hist_reference_cdf": "uniform"},
    "M3": {"invert_polarity": True},
    "r7_like": {
        "detection_bg_mode": "none",
        "gain_mode": "agc_capped",
        "agc_floor_beta": 0.5,
        "pad_mode": "taper",
        "pad_taper_periods": 1.0,
        "gray_mapping": "hist_aihub",
        "hist_reference_cdf": "uniform",
    },
}


def _uniform_reference_cdf() -> tuple[float, ...]:
    return tuple((np.arange(GRAY_LEVELS) + 1.0) / GRAY_LEVELS)


def _config(option_set: str, **extra: object) -> DetectionImageConfig:
    fields = dict(OPTION_SETS[option_set])
    if fields.get("hist_reference_cdf") == "uniform":
        fields["hist_reference_cdf"] = _uniform_reference_cdf()
    fields.update(extra)
    return DetectionImageConfig(recipe_id=option_set, **fields)  # type: ignore[arg-type]


def _sha256(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _geometry(image: DetectionImage) -> tuple[object, ...]:
    """Everything of an image that the recipe must not change (S6 (f))."""
    return (
        image.period_samples,
        image.period_source,
        image.grid_map,
        image.tiles,
        image.content_height_px,
        image.compression,
        image.depth_m_per_sample,
        image.v_scale_m_per_ns,
        image.overlap_px,
        image.overlap_phys_px,
        image.line_image.shape,
    )


@pytest.fixture(scope="module")
def ground_line() -> Radargram:
    return make_radargram(SYNTHETIC_SPECS["ground_401"])


# ------------------------------------------------------------- (a) bit identity


@pytest.mark.parametrize("name", sorted(SYNTHETIC_SPECS))
def test_a_default_config_is_bit_identical_on_synthetic_lines(name: str) -> None:
    image = make_detection_image(make_radargram(SYNTHETIC_SPECS[name]))
    assert _sha256(image.line_image) == SYNTHETIC_SHA256[name]


def test_a_default_config_is_recipe_r0() -> None:
    config = DetectionImageConfig()
    assert config.recipe_id == "r0"
    assert config.detection_bg_mode == DetectionBackgroundMode.MEDIAN
    assert config.detection_bg_window == config.period_bg_window == 301
    assert config.gain_mode == GainMode.AGC
    assert config.agc_window_periods == 8.0
    assert config.pad_mode == PadMode.FLAT
    assert config.pad_gray == 128
    assert config.gray_mapping == GrayMapping.SYMMETRIC
    assert config.clip_percentile == 99.0
    assert config.invert_polarity is False


def test_a_assumed_params_record_every_option_and_the_recipe(ground_line) -> None:
    config = _config("r7_like")
    image = make_detection_image(ground_line, config)
    assert image.assumed_params == M3_2_ASSUMED
    assert image.recipe_id == "r7_like"
    values = dict(image.assumed_values)
    assert tuple(values) == M3_2_ASSUMED
    assert values["recipe_id"] == "r7_like"
    assert values["detection_bg_mode"] == "none"
    assert values["gain_mode"] == "agc_capped"
    assert values["agc_floor_beta"] == 0.5
    assert values["pad_mode"] == "taper"
    assert values["gray_mapping"] == "hist_aihub"
    assert "period_bg_window" not in values  # fixed, not an M3-2 assumption


# ------------------------------------------------------- (b) partial background


def test_b_partial_alpha_0_is_none_and_alpha_1_is_median(ground_line) -> None:
    none_image = make_detection_image(
        ground_line,
        DetectionImageConfig(
            recipe_id="B1", detection_bg_mode=DetectionBackgroundMode.NONE
        ),
    )
    median_image = make_detection_image(ground_line)
    for alpha, expected in ((0.0, none_image), (1.0, median_image)):
        partial = make_detection_image(
            ground_line,
            DetectionImageConfig(
                recipe_id="B2",
                detection_bg_mode=DetectionBackgroundMode.PARTIAL,
                detection_bg_alpha=alpha,
            ),
        )
        np.testing.assert_array_equal(partial.line_image, expected.line_image)
    assert not np.array_equal(none_image.line_image, median_image.line_image)


# ---------------------------------------------------------- (c) gated weights


@pytest.mark.parametrize("period_samples", [4.45, 12.15, 25.6])
@pytest.mark.parametrize("gate_periods", [2.0, 4.0])
def test_c_gate_weight_is_continuous(period_samples, gate_periods) -> None:
    k_zero = 37.3
    weights = background_gate_weights(512, k_zero, gate_periods, period_samples)
    samples = np.arange(512)
    gate_end = k_zero + gate_periods * period_samples

    assert np.all(weights[samples < gate_end] == 1.0)
    assert np.all(weights[samples >= gate_end + period_samples] == 0.0)
    assert np.all(np.diff(weights) <= 0.0)
    # A cosine of one period never steps more than its slope bound pi / (2 P).
    assert np.max(np.abs(np.diff(weights))) <= np.pi / (2 * period_samples) + 1e-12


def test_c_gated_removes_background_only_before_the_gate(ground_line) -> None:
    config = DetectionImageConfig(
        recipe_id="B3",
        detection_bg_mode=DetectionBackgroundMode.GATED,
        gain_mode=GainMode.NONE,
    )
    gated = to_image._line_amplitudes(ground_line, config)
    assert gated.gate_k_zero is not None
    branch = prepare_fitting_branch(ground_line)
    weights = background_gate_weights(
        ground_line.n_samples,
        gated.gate_k_zero,
        config.detection_bg_gate_periods,
        gated.period_samples,
    )
    early = weights == 1.0
    late = weights == 0.0
    np.testing.assert_array_equal(
        gated.amplitudes[early], branch.background_removed[early]
    )
    np.testing.assert_array_equal(gated.amplitudes[late], branch.filtered[late])
    assert early.any() and late.any()


def test_c_gated_without_zero_falls_back_to_median(ground_line, monkeypatch) -> None:
    filtered = prepare_fitting_branch(ground_line).filtered
    failed = replace(
        to_image.find_time_zero(ground_line, filtered),
        t_zero_ns=None,
        k_zero=None,
        zero_rule="zero_failed",
    )
    monkeypatch.setattr(to_image, "find_time_zero", lambda *args, **kwargs: failed)
    gated = make_detection_image(
        ground_line,
        DetectionImageConfig(
            recipe_id="B3", detection_bg_mode=DetectionBackgroundMode.GATED
        ),
    )
    median = make_detection_image(ground_line)
    assert ImageFlag.DETECTION_BG_GATE_FALLBACK in gated.flags
    np.testing.assert_array_equal(gated.line_image, median.line_image)


# ------------------------------------------------------------------ (d) taper


@pytest.mark.parametrize("taper_periods", [1.0, 2.0])
def test_d_taper_reaches_pad_gray_at_the_content_end(ground_line, taper_periods):
    flat = make_detection_image(ground_line)
    tapered = make_detection_image(
        ground_line,
        DetectionImageConfig(
            recipe_id="P1", pad_mode=PadMode.TAPER, pad_taper_periods=taper_periods
        ),
    )
    height = flat.content_height_px
    taper_rows = taper_periods * flat.period_l1_px
    untouched = int(np.floor(height - taper_rows))
    last_row = tapered.line_image[height - 1].astype(int)

    assert np.max(np.abs(last_row - tapered.pad_gray)) <= TAPER_END_TOLERANCE_GRAY
    np.testing.assert_array_equal(
        tapered.line_image[:untouched], flat.line_image[:untouched]
    )
    np.testing.assert_array_equal(tapered.line_image[height:], flat.line_image[height:])
    # Grey distance to the padding shrinks towards the content end.
    distance = np.abs(tapered.line_image[untouched:height].astype(int) - 128)
    assert distance.mean(axis=1)[-1] < distance.mean(axis=1)[0]


# ---------------------------------------------------------------- (e) capped AGC


def test_e_agc_capped_beta_0_is_agc(ground_line) -> None:
    agc_image = make_detection_image(ground_line)
    capped = make_detection_image(
        ground_line,
        DetectionImageConfig(
            recipe_id="G2", gain_mode=GainMode.AGC_CAPPED, agc_floor_beta=0.0
        ),
    )
    np.testing.assert_array_equal(capped.line_image, agc_image.line_image)


def test_e_agc_capped_limits_the_gain_of_quiet_samples() -> None:
    section = np.ones((400, 3))
    section[200:] = 1e-3  # quiet late part of the traces
    beta = 0.5
    gained = to_image._capped_agc(section, 0.1, 400e6, 8.0, beta)
    rms = agc_rms(section, 0.1, 400e6)
    cap = beta * np.median(rms, axis=0)
    np.testing.assert_allclose(gained, section / np.maximum(rms, cap))
    assert np.max(np.abs(gained[300:])) < np.max(np.abs(agc(section, 0.1, 400e6)[300:]))


def test_e_agc_window_option_matches_the_filter_window() -> None:
    rng = np.random.default_rng(20261114)
    section = rng.normal(size=(300, 4))
    np.testing.assert_array_equal(
        agc(section, 0.1, 400e6, window_periods=8.0), agc(section, 0.1, 400e6)
    )
    assert not np.array_equal(
        agc(section, 0.1, 400e6, window_periods=32.0), agc(section, 0.1, 400e6)
    )


# --------------------------------------------------- (f) geometry is recipe-free


@pytest.mark.parametrize("option_set", sorted(OPTION_SETS))
@pytest.mark.parametrize("spec_name", sorted(SYNTHETIC_SPECS))
def test_f_geometry_does_not_depend_on_the_recipe(spec_name, option_set) -> None:
    radargram = make_radargram(SYNTHETIC_SPECS[spec_name])
    default = make_detection_image(radargram)
    recipe = make_detection_image(radargram, _config(option_set))
    assert _geometry(recipe) == _geometry(default)
    # Only the detection-branch options may add a flag (the gate fallback).
    assert set(recipe.flags) - set(default.flags) <= {
        ImageFlag.DETECTION_BG_GATE_FALLBACK
    }


def test_f_period_section_ignores_the_detection_background(monkeypatch) -> None:
    """Changing only ``detection_bg_window`` leaves the period input unchanged."""
    seen: list[np.ndarray] = []
    measure = to_image.measure_period_samples

    def spy(section, **kwargs):
        seen.append(np.array(section))
        return measure(section, **kwargs)

    monkeypatch.setattr(to_image, "measure_period_samples", spy)
    radargram = make_radargram(SYNTHETIC_SPECS["ground_401"])
    make_detection_image(radargram)
    make_detection_image(
        radargram, DetectionImageConfig(recipe_id="B4", detection_bg_window=51)
    )
    np.testing.assert_array_equal(seen[0], seen[1])


# ------------------------------------------------------- other option behaviour


def test_m3_inverts_the_content_and_keeps_the_padding(ground_line) -> None:
    normal = make_detection_image(ground_line)
    inverted = make_detection_image(
        ground_line, DetectionImageConfig(recipe_id="M3", invert_polarity=True)
    )
    height = normal.content_height_px
    np.testing.assert_array_equal(
        inverted.line_image[:height], 255 - normal.line_image[:height]
    )
    np.testing.assert_array_equal(
        inverted.line_image[height:], normal.line_image[height:]
    )


def test_m2_histogram_matching_follows_the_reference_cdf(ground_line) -> None:
    image = make_detection_image(ground_line, _config("M2"))
    content = image.line_image[: image.content_height_px]
    # A uniform reference spreads the content evenly over the grey levels.
    counts = np.bincount(content.ravel(), minlength=GRAY_LEVELS)
    assert counts.max() <= 3 * content.size / GRAY_LEVELS


def test_m2_matching_keeps_the_amplitude_order_and_ties() -> None:
    rng = np.random.default_rng(20261114)
    values = rng.normal(size=(40, 30))
    values[0, :5] = 0.25  # ties get one grey
    reference = _uniform_reference_cdf()
    levels = to_image._hist_matched_gray_levels(values, reference)

    order = np.argsort(values.ravel())
    assert np.all(np.diff(levels.ravel()[order]) >= 0)
    assert np.unique(levels[0, :5]).size == 1
    # A point-mass reference sends every value to that grey level.
    point_mass = tuple(float(level >= 200) for level in range(GRAY_LEVELS))
    assert np.all(to_image._hist_matched_gray_levels(values, point_mass) == 200)


def test_g3_tpow_exponent_0_is_no_gain(ground_line) -> None:
    no_gain = make_detection_image(
        ground_line, DetectionImageConfig(recipe_id="G4", gain_mode=GainMode.NONE)
    )
    tpow_0 = make_detection_image(
        ground_line,
        DetectionImageConfig(
            recipe_id="G3", gain_mode=GainMode.TPOW, tpow_exponent=0.0
        ),
    )
    np.testing.assert_array_equal(tpow_0.line_image, no_gain.line_image)


def test_g3_tpow_scales_by_record_time() -> None:
    section = np.ones((5, 2))
    gained = to_image._tpow_gain(section, dt_ns=0.5, exponent=2.0)
    np.testing.assert_array_equal(gained[:, 0], [0.0, 0.25, 1.0, 2.25, 4.0])


def test_processed_amplitudes_ignore_background_and_gain_options() -> None:
    radargram = make_radargram(
        BScanSpec(n_traces=200, n_samples=321), amplitude_processed=True
    )
    default = make_detection_image(radargram)
    changed = make_detection_image(
        radargram,
        DetectionImageConfig(
            recipe_id="x",
            detection_bg_mode=DetectionBackgroundMode.GATED,
            gain_mode=GainMode.AGC_CAPPED,
        ),
    )
    np.testing.assert_array_equal(changed.line_image, default.line_image)


# -------------------------------------------------------------- config checks


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"detection_bg_mode": "mean"}, "detection_bg_mode"),
        ({"gain_mode": "log"}, "gain_mode"),
        ({"pad_mode": "noise"}, "pad_mode"),
        ({"gray_mapping": "log"}, "gray_mapping"),
        ({"detection_bg_alpha": 1.5}, "detection_bg_alpha"),
        ({"detection_bg_gate_periods": 0.0}, "detection_bg_gate_periods"),
        ({"agc_window_periods": 0.0}, "agc_window_periods"),
        ({"agc_floor_beta": -0.1}, "agc_floor_beta"),
        ({"gain_mode": "tpow"}, "tpow_exponent"),
        ({"gain_mode": "tpow", "tpow_exponent": -0.5}, "tpow_exponent"),
        ({"gain_mode": "tpow", "tpow_exponent": float("nan")}, "tpow_exponent"),
        ({"gain_mode": "tpow", "tpow_exponent": float("inf")}, "tpow_exponent"),
        ({"pad_taper_periods": 0.0}, "pad_taper_periods"),
        ({"period_bg_window": 0}, "period_bg_window"),
        ({"detection_bg_window": 0}, "detection_bg_window"),
        ({"gray_mapping": "hist_aihub"}, "hist_reference_cdf"),
        ({"gray_mapping": "hist_aihub", "hist_reference_cdf": (0.5, 1.0)}, "256"),
        (
            {
                "gray_mapping": "hist_aihub",
                "hist_reference_cdf": tuple(np.linspace(0.0, 0.9, GRAY_LEVELS)),
            },
            "end at 1",
        ),
        ({"recipe_id": ""}, "recipe_id"),
    ],
)
def test_config_rejects_bad_options(fields, message) -> None:
    fields = {"recipe_id": "x", **fields}
    with pytest.raises(ValueError, match=message):
        DetectionImageConfig(**fields)  # type: ignore[arg-type]


def test_config_rejects_a_non_monotonic_reference_cdf() -> None:
    reference = list(_uniform_reference_cdf())
    reference[10], reference[11] = reference[11], reference[10]
    with pytest.raises(ValueError, match="non-decreasing"):
        DetectionImageConfig(
            recipe_id="M2",
            gray_mapping=GrayMapping.HIST_AIHUB,
            hist_reference_cdf=tuple(reference),
        )


def test_r0_label_needs_r0_values() -> None:
    with pytest.raises(ValueError, match="recipe_id"):
        DetectionImageConfig(gain_mode=GainMode.NONE)
    with pytest.raises(ValueError, match="recipe_id"):
        DetectionImageConfig(clip_percentile=98.0)
    # Non-recipe settings keep the r0 label.
    assert DetectionImageConfig(p_target_px=4.06, overlap_px=128).recipe_id == "r0"


def test_option_strings_become_enums() -> None:
    """Recipe tables from JSON pass plain strings; they are stored as enums."""
    config = DetectionImageConfig(
        recipe_id="x",
        detection_bg_mode="gated",  # type: ignore[arg-type]
        gain_mode="agc_capped",  # type: ignore[arg-type]
    )
    assert config.detection_bg_mode is DetectionBackgroundMode.GATED
    assert config.gain_mode is GainMode.AGC_CAPPED


# ------------------------------------------------------------ real data (opt-in)


@dataclass(frozen=True)
class RealLine:
    """One spec section 9 file with its line_image sha256 before S1."""

    name: str
    dataset: DatasetName
    load: Callable[[Path], Radargram]
    sha256: str


def _fourtu(activity_id: str, path: str, sha256: str) -> RealLine:
    return RealLine(
        f"4tu/{activity_id}/{path}",
        DatasetName.FOURTU,
        lambda root: load_fourtu_radargram(root, activity_id, path),
        sha256,
    )


def _guangzhou(file_id: str, sha256: str) -> RealLine:
    return RealLine(
        f"gz/{file_id}",
        DatasetName.GUANGZHOU,
        lambda root: load_guangzhou_radargram(root, file_id),
        sha256,
    )


def _gprsd(file_id: str, sha256: str) -> RealLine:
    return RealLine(
        f"gprsd/{file_id}",
        DatasetName.GPRSD,
        lambda root: load_gprsd_radargram(root, file_id, DtHypothesis.A),
        sha256,
    )


SPEC9_LINES = (
    _fourtu(
        "03.9",
        "Path5",
        "a0891dbfd7efda9a862ddc8a3b0a0ebfd898f821750c07ba85e0a68275d94c79",
    ),
    _fourtu(
        "01.1",
        "Path1",
        "10e8bc9ad7c671a766f9797d9a385dbf8b46b04867ba7d79b61d6d600019a62f",
    ),
    _fourtu(
        "09.1",
        "Path2",
        "66863f4caa028cbca8ee93f173cb7cb5692733a38f61cf66d464316f9cc33ffd",
    ),
    _fourtu(
        "09.4",
        "Path1",
        "78fad5249da1ceb1cf83c752eb8d7096f22be009c471ca190819ca15f075b660",
    ),
    _guangzhou(
        "pipe/001_190321AA/LID10001",
        "b1de7022cc398c91bfa3d34c329938868dbc1559fa878eeef75ae12192d13d78",
    ),
    _guangzhou(
        "pipe/1030_190117AF/LID10001",
        "a3a40315e42ac81602759c7490b98b6c0306bbb4461f0cc536c51031bd7699da",
    ),
    _guangzhou(
        "pipe/190321A_190321AH/LID10001",
        "788f8d22fcccb6b26015d4b699739122922afdd7898d94a8cad0639039e22e46",
    ),
    _guangzhou(
        "pipe/MALA_07011/LA010001",
        "2a0707a491e844808cb29bc0450eba977004e27a76d96dc458e5c75a12d98596",
    ),
    _guangzhou(
        "pipe/1030_181224AB/LID10001",
        "d73a8135a5c9c86075b319c3fd711584af08fe9f88e8e38c069fa6f4ab4e049e",
    ),
    _gprsd(
        "CementRoad2_1",
        "292ede9cdffdef2507692a9418aed58107546832c28af417ebae357a6ce9c9e6",
    ),
    _gprsd(
        "AsphaltRoad2_1",
        "30b732908110f45f581673bd0dd6765f5b4ca68c492dee603957bfa8a9b9fb89",
    ),
)
DT0195_LINE = _fourtu(
    "011.17",
    "Profile_6409",
    "3f1b927cae5abb969ed7f0c273f9007fa0be37268ad6e695d18b2135274c973d",
)
"""Shortest-period Delta t 0.195 line (plan section 1-4, 1.4 % height margin)."""

REAL_OPTION_SETS = ("B1", "B2_a0.5", "B3_ng2", "B4", "G2_b0.5", "P1_nt1", "r7_like")


@pytest.fixture(scope="module")
def dataset_paths() -> DatasetPaths:
    return load_dataset_paths()


def _load(line: RealLine, paths: DatasetPaths) -> Radargram:
    root = paths.root(line.dataset)
    if not root.is_dir():
        pytest.skip(f"{line.dataset.value} data not found at {root}")
    return line.load(root)


@pytest.mark.data
@pytest.mark.realdata
@pytest.mark.parametrize(
    "line", (*SPEC9_LINES, DT0195_LINE), ids=lambda line: line.name
)
def test_a_f_real_lines_default_bits_and_recipe_free_geometry(
    line: RealLine, dataset_paths: DatasetPaths
) -> None:
    radargram = _load(line, dataset_paths)
    default = make_detection_image(radargram)
    assert _sha256(default.line_image) == line.sha256
    for option_set in REAL_OPTION_SETS:
        recipe = make_detection_image(radargram, _config(option_set))
        assert _geometry(recipe) == _geometry(default), option_set
