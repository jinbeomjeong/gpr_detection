"""Detection backends behind the ``Detector`` protocol (ADR-001).

Import rule: ``ultralytics`` and ``rfdetr`` are imported **only** inside
functions of the backend modules in this package (``yolo.py``,
``yolo_train.py``; later ``rfdetr.py``). This ``__init__`` re-exports the
backend-neutral contract only, so ``import gpr_detection.detect`` works in the
``gpr-detr`` env where ultralytics is absent.
"""

from gpr_detection.detect.base import (
    Detection,
    Detector,
    DetectorOutput,
    InputTransform,
    PixelBox,
    letterbox,
)

__all__ = [
    "Detection",
    "Detector",
    "DetectorOutput",
    "InputTransform",
    "PixelBox",
    "letterbox",
]
