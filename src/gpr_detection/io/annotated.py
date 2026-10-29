"""Labelled detection image shared by the AIHub and Morocco loaders."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gpr_detection.io.classes import DetectionClass


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates of the original image.

    Attributes:
        class_id: Project detection class.
        x1: Left edge in px.
        y1: Top edge in px.
        x2: Right edge in px.
        y2: Bottom edge in px.
        source_label: Native label in the source dataset (e.g. ``"tpipe"``).
    """

    class_id: DetectionClass
    x1: float
    y1: float
    x2: float
    y2: float
    source_label: str

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Box corners as ``(x1, y1, x2, y2)``."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class AnnotatedImage:
    """One detection image with its boxes and the keys needed for leak-free splits.

    Attributes:
        image: Grayscale uint8 image, shape ``[height, width]``.
        boxes: Labelled boxes (may be empty for background images).
        dataset: Dataset key, e.g. ``"aihub"``.
        image_id: Unique image identifier inside the dataset.
        session: Primary split-group key (AIHub ``<source>``, Morocco ``group_id``).
        session_from_name: Session derived from the file name (AIHub only); kept
            next to ``session`` because the two disagree for some AIHub files.
        crop_key: Key shared by copies of the same crop (AIHub gain copies,
            Morocco augmentations of one original patch).
        gain: AIHub gain value (0.25, 0.5, 1.0); None for other datasets.
        metadata: Remaining source fields, for traceability.
    """

    image: np.ndarray
    boxes: tuple[Box, ...]
    dataset: str
    image_id: str
    session: str | None = None
    session_from_name: str | None = None
    crop_key: str | None = None
    gain: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image.ndim != 2 or self.image.dtype != np.uint8:
            raise ValueError(
                f"image must be 2-D uint8, got {self.image.dtype} {self.image.shape}"
            )

    @property
    def height(self) -> int:
        """Image height in px."""
        return self.image.shape[0]

    @property
    def width(self) -> int:
        """Image width in px."""
        return self.image.shape[1]
