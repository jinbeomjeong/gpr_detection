"""Backend-neutral photometric augmentations for grayscale GPR B-scans (M1-7).

The four augmentations mimic acquisition and display variation without touching
geometry, so the time axis (rows) and the hyperbola shapes stay intact
(``docs/experiments/augmentation_policy.md`` section 1):

* Gaussian noise - receiver / thermal noise.
* Contrast and gamma - display gain differences between surveys.
* Horizontal band noise ("ringing") - antenna ringing that repeats down every
  trace as faint horizontal stripes.
* Trace dropout - missing or blanked traces (vertical columns).

Every function takes a uint8 image ``[H, W]`` (rows = time samples, columns =
traces) and a seeded ``numpy.random.Generator`` and returns a **new** array; the
input is never modified (Ultralytics keeps decoded images in a buffer and hands
the same array out again).

Intensity parameters are fractions of the full 8-bit range (255) or of the
image width, so they do not depend on image size. ``AugmentConfig.scaled``
multiplies every intensity by one factor for strength comparison runs; the
probabilities stay unchanged. ``strength=0`` turns every augmentation into a
no-op.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import numpy as np

FULL_SCALE_GRAY_LEVEL = 255.0
"""Maximum value of a uint8 image; noise and ringing amplitudes are fractions of it."""

_UINT8_LEVEL_COUNT = 256
_MAX_CONTRAST_OR_GAMMA_DELTA = 0.95
"""Upper bound for scaled contrast/gamma deltas so the factors stay positive."""


class TraceFill(StrEnum):
    """Value written into dropped traces."""

    ZERO = "zero"
    """Black (0). A strong, unrealistic edge; useful only as an ablation."""
    MEAN = "mean"
    """Image mean. Close to a zero-amplitude trace after display scaling."""


@dataclass(frozen=True)
class GaussianNoiseConfig:
    """Additive Gaussian noise.

    Attributes:
        probability: Chance of applying the augmentation to one image.
        sigma_min: Lower bound of the noise standard deviation, fraction of 255.
        sigma_max: Upper bound of the noise standard deviation, fraction of 255.
    """

    probability: float = 0.3
    sigma_min: float = 0.02
    sigma_max: float = 0.05

    def __post_init__(self) -> None:
        _check_probability(self.probability)
        _check_range("sigma", self.sigma_min, self.sigma_max)


@dataclass(frozen=True)
class ContrastGammaConfig:
    """Contrast stretch around the image mean followed by a gamma curve.

    Attributes:
        probability: Chance of applying the augmentation to one image.
        contrast_max_delta: Contrast factor is drawn from ``1 +- delta``.
        gamma_max_delta: Gamma exponent is drawn from ``1 +- delta``.
    """

    probability: float = 0.3
    contrast_max_delta: float = 0.15
    gamma_max_delta: float = 0.15

    def __post_init__(self) -> None:
        _check_probability(self.probability)
        for name in ("contrast_max_delta", "gamma_max_delta"):
            value = getattr(self, name)
            if not 0.0 <= value <= _MAX_CONTRAST_OR_GAMMA_DELTA:
                raise ValueError(
                    f"{name} must be in [0, {_MAX_CONTRAST_OR_GAMMA_DELTA}], "
                    f"got {value}"
                )


@dataclass(frozen=True)
class RingingConfig:
    """Faint horizontal stripes, constant along each row (antenna ringing).

    The amplitude is kept low on purpose: strong stripes look like the
    horizontal band of a longitudinal pipe (class ``pipe_band``) and could teach
    the model to ignore real bands or to see bands in clutter. The strength
    comparison run checks this effect on ``pipe_band`` AP.

    Attributes:
        probability: Chance of applying the augmentation to one image.
        amplitude_min: Lower bound of the stripe amplitude, fraction of 255.
        amplitude_max: Upper bound of the stripe amplitude, fraction of 255.
        period_min_px: Shortest stripe period along the time axis, in rows.
        period_max_px: Longest stripe period along the time axis, in rows.
    """

    probability: float = 0.2
    amplitude_min: float = 0.01
    amplitude_max: float = 0.03
    period_min_px: float = 6.0
    period_max_px: float = 24.0

    def __post_init__(self) -> None:
        _check_probability(self.probability)
        _check_range("amplitude", self.amplitude_min, self.amplitude_max)
        if not 0.0 < self.period_min_px <= self.period_max_px:
            raise ValueError(
                "period must satisfy 0 < period_min_px <= period_max_px, got "
                f"{self.period_min_px}, {self.period_max_px}"
            )


@dataclass(frozen=True)
class TraceDropoutConfig:
    """Replace a few randomly chosen traces (columns) with a constant.

    Attributes:
        probability: Chance of applying the augmentation to one image.
        fraction_min: Lower bound of the dropped share of columns.
        fraction_max: Upper bound of the dropped share of columns.
        fill: Value written into the dropped columns.
    """

    probability: float = 0.3
    fraction_min: float = 0.01
    fraction_max: float = 0.05
    fill: TraceFill = TraceFill.MEAN

    def __post_init__(self) -> None:
        _check_probability(self.probability)
        _check_range("fraction", self.fraction_min, self.fraction_max)
        if self.fraction_max > 1.0:
            raise ValueError(f"fraction_max must be <= 1, got {self.fraction_max}")
        object.__setattr__(self, "fill", TraceFill(self.fill))


@dataclass(frozen=True)
class AugmentConfig:
    """Probabilities and intensities of the four B-scan augmentations.

    Attributes:
        gaussian_noise: Additive Gaussian noise settings.
        contrast_gamma: Contrast and gamma settings.
        ringing: Horizontal band noise settings.
        trace_dropout: Trace dropout settings.
    """

    gaussian_noise: GaussianNoiseConfig = field(default_factory=GaussianNoiseConfig)
    contrast_gamma: ContrastGammaConfig = field(default_factory=ContrastGammaConfig)
    ringing: RingingConfig = field(default_factory=RingingConfig)
    trace_dropout: TraceDropoutConfig = field(default_factory=TraceDropoutConfig)

    def scaled(self, strength: float) -> "AugmentConfig":
        """Return a copy with every intensity multiplied by ``strength``.

        Scaled: noise sigma, contrast and gamma deltas, ringing amplitude and
        dropped column share. Not scaled: probabilities, ringing period, fill.
        Contrast/gamma deltas are capped at 0.95 and the dropout share at 1.

        Args:
            strength: Non-negative multiplier. 1 keeps the config, 0 disables
                every augmentation, 2 doubles the intensities.

        Raises:
            ValueError: If ``strength`` is negative.
        """
        if strength < 0:
            raise ValueError(f"strength must be >= 0, got {strength}")
        noise = self.gaussian_noise
        contrast = self.contrast_gamma
        ringing = self.ringing
        dropout = self.trace_dropout
        return AugmentConfig(
            gaussian_noise=replace(
                noise,
                sigma_min=noise.sigma_min * strength,
                sigma_max=noise.sigma_max * strength,
            ),
            contrast_gamma=replace(
                contrast,
                contrast_max_delta=_capped_delta(contrast.contrast_max_delta, strength),
                gamma_max_delta=_capped_delta(contrast.gamma_max_delta, strength),
            ),
            ringing=replace(
                ringing,
                amplitude_min=ringing.amplitude_min * strength,
                amplitude_max=ringing.amplitude_max * strength,
            ),
            trace_dropout=replace(
                dropout,
                fraction_min=min(dropout.fraction_min * strength, 1.0),
                fraction_max=min(dropout.fraction_max * strength, 1.0),
            ),
        )


def _check_probability(probability: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")


def _check_range(name: str, low: float, high: float) -> None:
    if not 0.0 <= low <= high:
        raise ValueError(
            f"{name} range must satisfy 0 <= min <= max, got {low}, {high}"
        )


def _capped_delta(delta: float, strength: float) -> float:
    return min(delta * strength, _MAX_CONTRAST_OR_GAMMA_DELTA)


def _check_gray_image(image: np.ndarray) -> None:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(
            f"image must be a 2-D uint8 array [H, W], got {image.dtype} {image.shape}"
        )


def _as_box_array(boxes_xyxy: np.ndarray) -> np.ndarray:
    boxes = np.array(boxes_xyxy, dtype=np.float64)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes_xyxy must have shape [N, 4], got {boxes.shape}")
    return boxes


def _to_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, FULL_SCALE_GRAY_LEVEL).astype(np.uint8)


def add_gaussian_noise(
    image: np.ndarray, config: GaussianNoiseConfig, rng: np.random.Generator
) -> np.ndarray:
    """Add zero-mean Gaussian noise with a random sigma (always applied).

    Args:
        image: Grayscale uint8 image ``[H, W]``.
        config: Sigma range (fractions of 255). The probability is ignored here.
        rng: Random generator.

    Returns:
        New uint8 image of the same shape.
    """
    _check_gray_image(image)
    sigma_gray_levels = rng.uniform(config.sigma_min, config.sigma_max)
    sigma_gray_levels *= FULL_SCALE_GRAY_LEVEL
    if sigma_gray_levels == 0:
        return image.copy()
    noise = rng.normal(0.0, sigma_gray_levels, size=image.shape)
    return _to_uint8(image + noise)


def adjust_contrast_gamma(
    image: np.ndarray, config: ContrastGammaConfig, rng: np.random.Generator
) -> np.ndarray:
    """Stretch contrast around the image mean, then apply a gamma curve.

    Both factors are drawn independently from ``1 +- delta``. Implemented as a
    256-entry lookup table, so the cost does not depend on image size.

    Args:
        image: Grayscale uint8 image ``[H, W]``.
        config: Contrast and gamma deltas. The probability is ignored here.
        rng: Random generator.

    Returns:
        New uint8 image of the same shape.
    """
    _check_gray_image(image)
    contrast = 1.0 + rng.uniform(-config.contrast_max_delta, config.contrast_max_delta)
    gamma = 1.0 + rng.uniform(-config.gamma_max_delta, config.gamma_max_delta)
    mean_level = float(image.mean())
    levels = np.arange(_UINT8_LEVEL_COUNT, dtype=np.float64)
    stretched = np.clip(
        mean_level + contrast * (levels - mean_level), 0.0, FULL_SCALE_GRAY_LEVEL
    )
    curved = FULL_SCALE_GRAY_LEVEL * (stretched / FULL_SCALE_GRAY_LEVEL) ** gamma
    lookup_table = _to_uint8(curved)
    return lookup_table[image]


def add_ringing(
    image: np.ndarray, config: RingingConfig, rng: np.random.Generator
) -> np.ndarray:
    """Add a sinusoidal stripe pattern that is constant along each row.

    Args:
        image: Grayscale uint8 image ``[H, W]``.
        config: Amplitude (fraction of 255) and period (rows). The probability
            is ignored here.
        rng: Random generator.

    Returns:
        New uint8 image of the same shape.
    """
    _check_gray_image(image)
    amplitude_gray_levels = rng.uniform(config.amplitude_min, config.amplitude_max)
    amplitude_gray_levels *= FULL_SCALE_GRAY_LEVEL
    period_rows = rng.uniform(config.period_min_px, config.period_max_px)
    phase_rad = rng.uniform(0.0, 2.0 * np.pi)
    rows = np.arange(image.shape[0], dtype=np.float64)
    row_profile = amplitude_gray_levels * np.sin(
        2.0 * np.pi * rows / period_rows + phase_rad
    )
    return _to_uint8(image + row_profile[:, np.newaxis])


def drop_traces(
    image: np.ndarray, config: TraceDropoutConfig, rng: np.random.Generator
) -> np.ndarray:
    """Overwrite randomly chosen columns (traces) with zero or the image mean.

    The number of columns is ``round(fraction * W)``; when that rounds to zero
    the image is returned unchanged.

    Args:
        image: Grayscale uint8 image ``[H, W]``.
        config: Dropped column share and fill value. The probability is ignored.
        rng: Random generator.

    Returns:
        New uint8 image of the same shape.
    """
    _check_gray_image(image)
    width = image.shape[1]
    fraction = rng.uniform(config.fraction_min, config.fraction_max)
    dropped_count = round(fraction * width)
    result = image.copy()
    if dropped_count == 0:
        return result
    columns = rng.choice(width, size=dropped_count, replace=False)
    fill_value = 0 if config.fill is TraceFill.ZERO else round(float(image.mean()))
    result[:, columns] = fill_value
    return result


def augment_bscan(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    config: AugmentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply each augmentation with its own probability.

    Order: contrast/gamma (display of the clean scan), ringing, Gaussian noise,
    then trace dropout last so dropped traces stay perfectly flat like real
    blank traces.

    Args:
        image: Grayscale uint8 image ``[H, W]``; not modified.
        boxes_xyxy: Boxes ``[N, 4]`` as ``x1, y1, x2, y2`` in px. None of the
            four augmentations moves pixels, so the boxes come back unchanged;
            the argument keeps the signature ready for geometric augmentations.
        config: Probabilities and intensities.
        rng: Random generator; pass a seeded one for reproducible runs.

    Returns:
        ``(augmented_image, boxes_xyxy)`` - a new image and a copy of the boxes.

    Raises:
        ValueError: If the image is not 2-D uint8 or the boxes are not ``[N, 4]``.
    """
    _check_gray_image(image)
    boxes = _as_box_array(boxes_xyxy)

    result = image.copy()
    if rng.random() < config.contrast_gamma.probability:
        result = adjust_contrast_gamma(result, config.contrast_gamma, rng)
    if rng.random() < config.ringing.probability:
        result = add_ringing(result, config.ringing, rng)
    if rng.random() < config.gaussian_noise.probability:
        result = add_gaussian_noise(result, config.gaussian_noise, rng)
    if rng.random() < config.trace_dropout.probability:
        result = drop_traces(result, config.trace_dropout, rng)
    return result, boxes


def augment_config_from_dict(settings: Mapping[str, Any]) -> AugmentConfig:
    """Build an ``AugmentConfig`` from a nested mapping (e.g. a YAML block).

    Missing sections or keys keep their defaults. A top-level ``strength`` key
    is applied with :meth:`AugmentConfig.scaled`.

    Args:
        settings: Mapping with optional keys ``strength``, ``gaussian_noise``,
            ``contrast_gamma``, ``ringing`` and ``trace_dropout``.

    Raises:
        TypeError: If a section contains an unknown key.
    """
    base = AugmentConfig(
        gaussian_noise=GaussianNoiseConfig(**settings.get("gaussian_noise", {})),
        contrast_gamma=ContrastGammaConfig(**settings.get("contrast_gamma", {})),
        ringing=RingingConfig(**settings.get("ringing", {})),
        trace_dropout=TraceDropoutConfig(**settings.get("trace_dropout", {})),
    )
    return base.scaled(float(settings.get("strength", 1.0)))
