"""Tests for the backend-neutral B-scan augmentations (M1-7)."""

import numpy as np
import pytest

from gpr_detection.augment import (
    AugmentConfig,
    ContrastGammaConfig,
    GaussianNoiseConfig,
    RingingConfig,
    TraceDropoutConfig,
    TraceFill,
    add_gaussian_noise,
    add_ringing,
    adjust_contrast_gamma,
    augment_bscan,
    augment_config_from_dict,
    drop_traces,
)

IMAGE_SHAPE = (256, 256)
MID_GRAY = 128
SEED = 7
ALWAYS = 1.0
NEVER = 0.0


def _random_image(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        40, 216, size=IMAGE_SHAPE, dtype=np.uint8
    )


def _flat_image() -> np.ndarray:
    return np.full(IMAGE_SHAPE, MID_GRAY, dtype=np.uint8)


def _always_on_config() -> AugmentConfig:
    return AugmentConfig(
        gaussian_noise=GaussianNoiseConfig(probability=ALWAYS),
        contrast_gamma=ContrastGammaConfig(probability=ALWAYS),
        ringing=RingingConfig(probability=ALWAYS),
        trace_dropout=TraceDropoutConfig(probability=ALWAYS),
    )


@pytest.mark.parametrize(
    ("function", "config"),
    [
        (add_gaussian_noise, GaussianNoiseConfig()),
        (adjust_contrast_gamma, ContrastGammaConfig()),
        (add_ringing, RingingConfig()),
        (drop_traces, TraceDropoutConfig()),
    ],
)
def test_each_augmentation_returns_new_uint8_image(function, config) -> None:
    image = _random_image()
    original = image.copy()

    result = function(image, config, np.random.default_rng(SEED))

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result is not image
    np.testing.assert_array_equal(image, original)
    assert not np.array_equal(result, image)


@pytest.mark.parametrize(
    "function", [add_gaussian_noise, adjust_contrast_gamma, add_ringing, drop_traces]
)
def test_rejects_non_gray_images(function) -> None:
    color = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="2-D uint8"):
        function(color, GaussianNoiseConfig(), np.random.default_rng(SEED))


def test_gaussian_noise_sigma_matches_config() -> None:
    sigma_fraction = 0.04
    config = GaussianNoiseConfig(sigma_min=sigma_fraction, sigma_max=sigma_fraction)

    result = add_gaussian_noise(_flat_image(), config, np.random.default_rng(SEED))

    measured_sigma = np.std(result.astype(float) - MID_GRAY)
    assert measured_sigma == pytest.approx(sigma_fraction * 255, rel=0.05)


def test_zero_sigma_noise_is_identity() -> None:
    image = _random_image()
    config = GaussianNoiseConfig(sigma_min=0.0, sigma_max=0.0)
    result = add_gaussian_noise(image, config, np.random.default_rng(SEED))
    np.testing.assert_array_equal(result, image)


def test_contrast_gamma_with_zero_delta_is_identity() -> None:
    image = _random_image()
    config = ContrastGammaConfig(contrast_max_delta=0.0, gamma_max_delta=0.0)
    result = adjust_contrast_gamma(image, config, np.random.default_rng(SEED))
    np.testing.assert_array_equal(result, image)


def test_contrast_gamma_preserves_gray_level_order() -> None:
    ramp = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    config = ContrastGammaConfig(contrast_max_delta=0.5, gamma_max_delta=0.5)
    for seed in range(20):
        result = adjust_contrast_gamma(ramp, config, np.random.default_rng(seed))
        assert np.all(np.diff(result[0].astype(int)) >= 0)


def test_ringing_is_constant_along_each_row_and_low_amplitude() -> None:
    config = RingingConfig(amplitude_min=0.03, amplitude_max=0.03)

    result = add_ringing(_flat_image(), config, np.random.default_rng(SEED))

    difference = result.astype(int) - MID_GRAY
    assert np.all(difference == difference[:, :1])
    assert np.abs(difference).max() <= round(0.03 * 255)
    assert np.abs(difference).max() >= round(0.03 * 255) - 1


@pytest.mark.parametrize(("fill", "expected"), [(TraceFill.ZERO, 0), ("mean", None)])
def test_trace_dropout_blanks_expected_column_count(fill, expected) -> None:
    image = _random_image()
    fraction = 0.05
    config = TraceDropoutConfig(fraction_min=fraction, fraction_max=fraction, fill=fill)

    result = drop_traces(image, config, np.random.default_rng(SEED))

    changed_columns = np.flatnonzero(np.any(result != image, axis=0))
    assert len(changed_columns) == round(fraction * IMAGE_SHAPE[1])
    fill_value = round(float(image.mean())) if expected is None else expected
    assert np.all(result[:, changed_columns] == fill_value)


def test_trace_dropout_rounding_to_zero_columns_is_identity() -> None:
    image = _random_image()
    config = TraceDropoutConfig(fraction_min=0.001, fraction_max=0.001)
    result = drop_traces(image, config, np.random.default_rng(SEED))
    np.testing.assert_array_equal(result, image)


def test_augment_bscan_is_reproducible_with_same_seed() -> None:
    image = _random_image()
    boxes = np.array([[10.0, 20.0, 50.0, 80.0]])
    config = _always_on_config()

    first, _ = augment_bscan(image, boxes, config, np.random.default_rng(SEED))
    second, _ = augment_bscan(image, boxes, config, np.random.default_rng(SEED))
    third, _ = augment_bscan(image, boxes, config, np.random.default_rng(SEED + 1))

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_augment_bscan_keeps_boxes_and_input() -> None:
    image = _random_image()
    original = image.copy()
    boxes = np.array([[10.0, 20.0, 50.0, 80.0], [0.0, 0.0, 256.0, 256.0]])

    _, returned_boxes = augment_bscan(
        image, boxes, _always_on_config(), np.random.default_rng(SEED)
    )

    np.testing.assert_array_equal(returned_boxes, boxes)
    assert returned_boxes is not boxes
    np.testing.assert_array_equal(image, original)


def test_augment_bscan_accepts_empty_boxes() -> None:
    _, boxes = augment_bscan(
        _random_image(), np.zeros((0,)), AugmentConfig(), np.random.default_rng(SEED)
    )
    assert boxes.shape == (0, 4)


def test_augment_bscan_rejects_bad_box_shape() -> None:
    with pytest.raises(ValueError, match=r"\[N, 4\]"):
        augment_bscan(
            _random_image(),
            np.zeros((2, 3)),
            AugmentConfig(),
            np.random.default_rng(SEED),
        )


def test_zero_strength_turns_everything_off() -> None:
    image = _random_image()
    config = _always_on_config().scaled(0.0)
    result, _ = augment_bscan(image, np.zeros((0, 4)), config, np.random.default_rng(1))
    np.testing.assert_array_equal(result, image)


def test_zero_probability_turns_everything_off() -> None:
    image = _random_image()
    config = AugmentConfig(
        gaussian_noise=GaussianNoiseConfig(probability=NEVER),
        contrast_gamma=ContrastGammaConfig(probability=NEVER),
        ringing=RingingConfig(probability=NEVER),
        trace_dropout=TraceDropoutConfig(probability=NEVER),
    )
    result, _ = augment_bscan(image, np.zeros((0, 4)), config, np.random.default_rng(1))
    np.testing.assert_array_equal(result, image)


def test_scaled_multiplies_intensities_only() -> None:
    base = AugmentConfig()
    doubled = base.scaled(2.0)

    assert doubled.gaussian_noise.sigma_max == pytest.approx(0.10)
    assert doubled.contrast_gamma.contrast_max_delta == pytest.approx(0.30)
    assert doubled.ringing.amplitude_max == pytest.approx(0.06)
    assert doubled.trace_dropout.fraction_max == pytest.approx(0.10)
    assert doubled.ringing.period_max_px == base.ringing.period_max_px
    assert doubled.gaussian_noise.probability == base.gaussian_noise.probability


def test_scaled_caps_deltas_and_fraction() -> None:
    huge = AugmentConfig().scaled(100.0)
    assert huge.contrast_gamma.contrast_max_delta == pytest.approx(0.95)
    assert huge.trace_dropout.fraction_max == 1.0


def test_negative_strength_is_rejected() -> None:
    with pytest.raises(ValueError, match="strength"):
        AugmentConfig().scaled(-0.5)


@pytest.mark.parametrize(
    "build",
    [
        lambda: GaussianNoiseConfig(probability=1.5),
        lambda: GaussianNoiseConfig(sigma_min=0.1, sigma_max=0.05),
        lambda: ContrastGammaConfig(gamma_max_delta=1.0),
        lambda: RingingConfig(period_min_px=0.0),
        lambda: TraceDropoutConfig(fraction_max=1.5),
        lambda: TraceDropoutConfig(fill="median"),
    ],
)
def test_invalid_settings_are_rejected(build) -> None:
    with pytest.raises(ValueError):
        build()


def test_config_from_dict_applies_strength_and_keeps_defaults() -> None:
    config = augment_config_from_dict(
        {"strength": 0.5, "ringing": {"probability": 0.1}}
    )
    assert config.ringing.probability == 0.1
    assert config.gaussian_noise.sigma_max == pytest.approx(0.025)
    assert augment_config_from_dict({}) == AugmentConfig()
