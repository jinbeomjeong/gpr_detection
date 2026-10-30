"""Dataset loaders and the shared data classes.

Physical radargrams (depth-capable when scaled): :mod:`.fourtu`,
:mod:`.guangzhou`, :mod:`.gprsd`, :mod:`.synthetic` (gprMax models, truth kept
separate) -> :class:`Radargram`.
Detection images without physical scale: :mod:`.aihub`, :mod:`.morocco`
-> :class:`AnnotatedImage`.
"""

from gpr_detection.io.annotated import AnnotatedImage, Box
from gpr_detection.io.classes import DetectionClass, class_names
from gpr_detection.io.paths import DatasetName, DatasetPaths, load_dataset_paths
from gpr_detection.io.radargram import AirGap, GridBox, Radargram

__all__ = [
    "AirGap",
    "AnnotatedImage",
    "Box",
    "DatasetName",
    "DatasetPaths",
    "DetectionClass",
    "GridBox",
    "Radargram",
    "class_names",
    "load_dataset_paths",
]
