"""Ultralytics YOLO backend for the ``Detector`` protocol (inference side).

``ultralytics`` (AGPL-3.0) is imported lazily inside :func:`load_yolo_detector`
only (ADR-001, CON-1): importing this module never requires the library.

The wrapper letterboxes the L1 image itself and hands Ultralytics a ready
``[1, C, S, S]`` tensor. Ultralytics skips its own letterbox for tensor input
and, because the "original" image then has the input size, reports boxes in
model input coordinates (L2). So the ``InputTransform`` returned here is the
exact transform behind the boxes, with no hidden ``ratio_pad`` to recover.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gpr_detection.detect.base import (
    Detection,
    DetectorOutput,
    PixelBox,
    letterbox,
)

DEFAULT_INPUT_SIZE_PX = 256
"""AIHub B-scans are 256 x 256 and training uses imgsz 256 (dataset_spec_m1)."""
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.7
MODEL_STRIDE_PX = 32
"""Largest YOLO11 stride; tensor inputs must be a multiple of it."""
_GRAY_CHANNELS = 1
_COLOR_CHANNELS = 3


def _to_numpy(values: Any) -> np.ndarray:
    """Convert a torch tensor (any device) or array-like to a numpy array."""
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        return values.numpy()
    return np.asarray(values)


def _class_names_in_order(names: Mapping[int, str]) -> tuple[str, ...]:
    return tuple(names[class_id] for class_id in sorted(names))


def _model_input_channels(model: Any) -> int:
    """Channel count stored in the model yaml (Ultralytics default is 3)."""
    model_yaml = getattr(model.model, "yaml", {}) or {}
    return int(model_yaml.get("channels", _COLOR_CHANNELS))


class YoloDetector:
    """``Detector`` backed by an Ultralytics YOLO model.

    Build it with :func:`load_yolo_detector`; the constructor takes an already
    loaded model object so tests can pass a stand-in.

    Args:
        model: Object with ``predict(source=..., **kwargs)`` returning
            Ultralytics ``Results`` and a ``names`` mapping.
        input_channels: Channels the network expects (1 for this project).
        input_size: Side of the square model input in px.
        confidence_threshold: Minimum score kept by Ultralytics NMS.
        iou_threshold: IoU threshold of Ultralytics NMS.
        device: Torch device string, e.g. ``"cuda:0"`` or ``"cpu"``; None lets
            Ultralytics choose.
    """

    def __init__(
        self,
        model: Any,
        input_channels: int,
        input_size: int = DEFAULT_INPUT_SIZE_PX,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        device: str | None = None,
    ) -> None:
        if input_size % MODEL_STRIDE_PX:
            raise ValueError(
                f"input_size must be a multiple of {MODEL_STRIDE_PX}, got {input_size}"
            )
        if input_channels not in (_GRAY_CHANNELS, _COLOR_CHANNELS):
            raise ValueError(f"input_channels must be 1 or 3, got {input_channels}")
        self._model = model
        self._input_channels = input_channels
        self._input_size = input_size
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._device = device
        self._class_names = _class_names_in_order(model.names)

    @property
    def class_names(self) -> tuple[str, ...]:
        """Class names in class-id order, as stored in the weights."""
        return self._class_names

    def predict(self, image: np.ndarray) -> DetectorOutput:
        """Detect objects in one L1 image.

        Args:
            image: uint8 ``[H, W]`` grayscale, or ``[H, W, 3]`` BGR (OpenCV
                order) which is converted to grayscale first.

        Returns:
            Detections in L2 px and the letterbox ``InputTransform``.
        """
        gray = _to_gray(image)
        padded, transform = letterbox(gray, self._input_size)
        model_input = self._to_tensor(padded)
        results = self._model.predict(
            source=model_input,
            imgsz=self._input_size,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            device=self._device,
            verbose=False,
        )
        detections = self._parse_result(results[0])
        return DetectorOutput(detections=detections, transform=transform)

    def _to_tensor(self, padded_gray: np.ndarray) -> Any:
        """``[S, S]`` uint8 -> float tensor ``[1, C, S, S]`` scaled to 0-1."""
        import torch  # lazy: keeps torch out of package import time

        planes = np.repeat(padded_gray[np.newaxis], self._input_channels, axis=0)
        return torch.from_numpy(planes[np.newaxis].astype(np.float32) / 255.0)

    def _parse_result(self, result: Any) -> tuple[Detection, ...]:
        boxes = result.boxes
        corners = _to_numpy(boxes.xyxy).reshape(-1, 4)
        scores = _to_numpy(boxes.conf).reshape(-1)
        class_ids = _to_numpy(boxes.cls).reshape(-1).astype(int)
        detections = []
        for (x1, y1, x2, y2), score, class_id in zip(
            corners, scores, class_ids, strict=True
        ):
            detections.append(
                Detection(
                    class_id=int(class_id),
                    class_name=self._class_names[class_id],
                    score=float(score),
                    box=PixelBox(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return tuple(detections)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {image.dtype}")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == _COLOR_CHANNELS:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == _GRAY_CHANNELS:
        return image[..., 0]
    raise ValueError(f"image must be [H, W], [H, W, 1] or [H, W, 3], got {image.shape}")


def load_yolo_detector(
    weights_path: Path,
    input_size: int = DEFAULT_INPUT_SIZE_PX,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    device: str | None = None,
) -> YoloDetector:
    """Load trained YOLO weights (``best.pt``) as a ``YoloDetector``.

    Args:
        weights_path: Ultralytics checkpoint path.
        input_size: Side of the square model input in px (training imgsz).
        confidence_threshold: Minimum score kept.
        iou_threshold: NMS IoU threshold.
        device: Torch device string or None.

    Raises:
        FileNotFoundError: If ``weights_path`` does not exist.
    """
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    from ultralytics import YOLO  # lazy by design (ADR-001)

    model = YOLO(str(weights_path))
    return YoloDetector(
        model=model,
        input_channels=_model_input_channels(model),
        input_size=input_size,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )
