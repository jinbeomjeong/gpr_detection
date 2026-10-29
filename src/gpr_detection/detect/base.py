"""Backend-neutral detector contract (ADR-001, architecture README section 2-3).

A backend (``yolo.py`` now, ``rfdetr.py`` in M4) receives a detection image in
layer L1, applies its own resize/padding to reach the model input (layer L2)
and returns boxes **in L2 unchanged** together with the ``InputTransform`` it
applied. The inverse mapping L2 -> L1 -> L0 lives in one place,
:mod:`gpr_detection.pipeline`.

Pixel convention: continuous coordinates, pixel ``j`` covers ``[j, j + 1)``
(see :mod:`gpr_detection.preprocess.to_image`).

This module must not import any detector library (ultralytics, rfdetr), so the
package stays importable in every conda env.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

LETTERBOX_PAD_VALUE = 114
"""Gray level of letterbox padding; the same value Ultralytics uses."""

_ULTRALYTICS_PAD_ROUNDING_BIAS = 0.1
"""Ultralytics rounds ``pad - 0.1`` so odd paddings put the extra pixel right/bottom."""


@dataclass(frozen=True)
class PixelBox:
    """Axis-aligned box in continuous pixel coordinates of one layer (L1 or L2).

    Attributes:
        x1: Left edge in px.
        y1: Top edge in px.
        x2: Right edge in px.
        y2: Bottom edge in px.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"need x1 <= x2 and y1 <= y2, got {self}")

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Box corners as ``(x1, y1, x2, y2)``."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class InputTransform:
    """The L1 -> L2 transform a backend applied: per-axis scale, then padding.

    ``x_L2 = x_L1 * scale_x + pad_x`` and ``y_L2 = y_L1 * scale_y + pad_y``.

    Attributes:
        scale_x: L2 px per L1 px along x.
        scale_y: L2 px per L1 px along y (differs from ``scale_x`` when a
            backend stretches to a square).
        pad_x: Left padding in L2 px.
        pad_y: Top padding in L2 px.
        input_w: Model input width in px (L2 size, for checks).
        input_h: Model input height in px.
    """

    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float
    input_w: int
    input_h: int

    def __post_init__(self) -> None:
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError(f"scales must be positive, got {self}")
        if self.input_w <= 0 or self.input_h <= 0:
            raise ValueError(f"input size must be positive, got {self}")

    def apply(self, box: PixelBox) -> PixelBox:
        """Map an L1 box to L2 (forward direction, used by backends and tests)."""
        return PixelBox(
            x1=box.x1 * self.scale_x + self.pad_x,
            y1=box.y1 * self.scale_y + self.pad_y,
            x2=box.x2 * self.scale_x + self.pad_x,
            y2=box.y2 * self.scale_y + self.pad_y,
        )


@dataclass(frozen=True)
class Detection:
    """One detected object, box in model input coordinates (L2).

    Attributes:
        class_id: Project class index (``DetectionClass`` value).
        class_name: Class name, e.g. ``"pipe_hyperbola"``.
        score: Confidence in ``[0, 1]``.
        box: Box in L2 px, exactly as the model produced it.
    """

    class_id: int
    class_name: str
    score: float
    box: PixelBox


@dataclass(frozen=True)
class DetectorOutput:
    """Result of one ``Detector.predict`` call.

    Attributes:
        detections: Detections in L2 coordinates.
        transform: The L1 -> L2 transform used for this image.
    """

    detections: tuple[Detection, ...]
    transform: InputTransform


@runtime_checkable
class Detector(Protocol):
    """Interface every detection backend implements."""

    @property
    def class_names(self) -> tuple[str, ...]:
        """Class names in class-id order."""
        ...

    def predict(self, image: np.ndarray) -> DetectorOutput:
        """Detect objects in one L1 image (uint8 ``[H, W]`` or ``[H, W, 3]``)."""
        ...


def letterbox(
    image: np.ndarray, input_size: int, pad_value: int = LETTERBOX_PAD_VALUE
) -> tuple[np.ndarray, InputTransform]:
    """Resize with the aspect ratio kept, then pad to ``input_size`` square.

    Mirrors Ultralytics ``LetterBox(center=True)``: one scale for both axes,
    rounded output size, padding split left/right and top/bottom. Because the
    resized size is rounded, the exact per-axis scales are reported.

    Args:
        image: uint8 image ``[H, W]`` or ``[H, W, C]``.
        input_size: Side of the square model input in px.
        pad_value: Gray level of the padding.

    Returns:
        ``(padded_image, transform)`` where ``padded_image`` has shape
        ``[input_size, input_size(, C)]``.

    Raises:
        ValueError: If the image is empty or ``input_size`` is not positive.
    """
    height, width = image.shape[:2]
    if height == 0 or width == 0 or input_size <= 0:
        raise ValueError(f"bad letterbox input {image.shape} -> {input_size}")

    ratio = min(input_size / height, input_size / width)
    resized_w = round(width * ratio)
    resized_h = round(height * ratio)
    resized = image
    if (resized_w, resized_h) != (width, height):
        resized = cv2.resize(
            image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )
        # cv2 drops a trailing single channel ([H, W, 1] -> [H, W]).
        resized = resized.reshape(resized_h, resized_w, *image.shape[2:])
    pad_left = round((input_size - resized_w) / 2 - _ULTRALYTICS_PAD_ROUNDING_BIAS)
    pad_top = round((input_size - resized_h) / 2 - _ULTRALYTICS_PAD_ROUNDING_BIAS)
    padded = np.full(
        (input_size, input_size, *image.shape[2:]), pad_value, dtype=image.dtype
    )
    padded[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized
    transform = InputTransform(
        scale_x=resized_w / width,
        scale_y=resized_h / height,
        pad_x=float(pad_left),
        pad_y=float(pad_top),
        input_w=input_size,
        input_h=input_size,
    )
    return padded, transform
