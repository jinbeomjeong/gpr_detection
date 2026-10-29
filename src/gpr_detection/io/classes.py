"""Detection classes and the mapping from each dataset's native labels.

Index order is fixed (CLAUDE.md, 2026-10-29):

* 0 ``pipe_hyperbola`` - AIHub ``tpipe`` (transverse pipe) and Morocco ``pipe``
* 1 ``pipe_band`` - AIHub ``lpipe`` (longitudinal pipe, horizontal band)
* 2 ``cavity`` - AIHub ``cavity`` and Morocco ``cavity``
* 3 ``manhole`` - AIHub ``manh``
"""

from enum import IntEnum


class DetectionClass(IntEnum):
    """Project detection classes; the integer value is the YOLO class id."""

    PIPE_HYPERBOLA = 0
    PIPE_BAND = 1
    CAVITY = 2
    MANHOLE = 3

    @property
    def label(self) -> str:
        """Lower-case class name used in data.yaml and reports."""
        return self.name.lower()


AIHUB_LABEL_TO_CLASS: dict[str, DetectionClass] = {
    "tpipe": DetectionClass.PIPE_HYPERBOLA,
    "lpipe": DetectionClass.PIPE_BAND,
    "cavity": DetectionClass.CAVITY,
    "manh": DetectionClass.MANHOLE,
}

# Morocco data.yaml: 0 pipe, 1 cavity.
MOROCCO_CLASS_ID_TO_CLASS: dict[int, DetectionClass] = {
    0: DetectionClass.PIPE_HYPERBOLA,
    1: DetectionClass.CAVITY,
}
MOROCCO_EXPECTED_NAMES: dict[int, str] = {0: "pipe", 1: "cavity"}


def class_names() -> dict[int, str]:
    """Return ``{class_id: name}`` in index order, as written to data.yaml."""
    return {member.value: member.label for member in DetectionClass}


def map_aihub_label(label: str) -> DetectionClass:
    """Map an AIHub VOC ``<object><name>`` to a project class.

    Raises:
        KeyError: If the label is not one of ``tpipe``, ``lpipe``, ``cavity``,
            ``manh``.
    """
    try:
        return AIHUB_LABEL_TO_CLASS[label]
    except KeyError:
        known = sorted(AIHUB_LABEL_TO_CLASS)
        raise KeyError(
            f"Unknown AIHub label {label!r}; expected one of {known}"
        ) from None


def map_morocco_class_id(class_id: int) -> DetectionClass:
    """Map a Morocco YOLO class id (0 pipe, 1 cavity) to a project class.

    Raises:
        KeyError: If the id is not 0 or 1.
    """
    try:
        return MOROCCO_CLASS_ID_TO_CLASS[class_id]
    except KeyError:
        raise KeyError(
            f"Unknown Morocco class id {class_id}; expected 0 or 1"
        ) from None
