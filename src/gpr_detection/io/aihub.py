"""AIHub 71327 B-scan reader that works directly on the zips (never extracts them).

Facts used here (docs/datasets/aihub_71327.md, checked against the files):

* Images are in ``<split>/01.원천데이터/{TS_,VS_}B scan_<category>.zip`` and VOC
  labels in ``<split>/02.라벨링데이터/{TL_,VL_}B scan_<category>.zip``. Members
  sit at the zip root with a leading ``/``, e.g.
  ``/NM_20221017(134135)_N00_B11_g0.5_x(4118-4374).jpg``.
* Images and labels are paired by **zip member name**. The XML ``<filename>``
  differs from the member name in 1,870 files, so it is kept only as metadata.
* The split group key is the XML ``<source>`` session. The session derived from
  the member name (``NM_20221017(134135)`` -> ``NM_2022-10-17_(13_41_35)``)
  is kept next to it because the two differ for 224 cavity files.
* Each crop exists in gain 0.25 / 0.5 / 1.0 copies; ``crop_key`` drops the
  ``_g<gain>_`` part so the copies can be kept on the same side of a split.

Open one :class:`AIHubZipReader` per worker process and reuse it: opening a zip
costs 7-158 ms (central directory), reading a member about 0.2 ms.
"""

import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Self

import numpy as np

from gpr_detection.io._common import decode_grayscale
from gpr_detection.io.annotated import AnnotatedImage, Box
from gpr_detection.io.classes import map_aihub_label

DATASET_KEY = "aihub"
IMAGE_FOLDER = "01.원천데이터"
LABEL_FOLDER = "02.라벨링데이터"
BSCAN_ZIP_INFIX = "B scan_"
IMAGE_SUFFIX = ".jpg"
LABEL_SUFFIX = ".xml"

_NAME_SESSION_PATTERN = re.compile(
    r"^(?P<equipment>[A-Z]{2})_"
    r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})"
    r"\((?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})\)"
)
_GAIN_PATTERN = re.compile(r"_g(?P<gain>\d+(?:\.\d+)?)_")


class AIHubSplit(StrEnum):
    """Provided AIHub split folders (they share sessions; re-split before use)."""

    TRAINING = "Training"
    VALIDATION = "Validation"


_ZIP_PREFIXES: dict[AIHubSplit, tuple[str, str]] = {
    # (image zip prefix, label zip prefix)
    AIHubSplit.TRAINING: ("TS_", "TL_"),
    AIHubSplit.VALIDATION: ("VS_", "VL_"),
}


@dataclass(frozen=True)
class AIHubZipPair:
    """Image zip and label zip of one B-scan category in one split.

    Attributes:
        split: Provided split folder.
        category: Category suffix, e.g. ``"비공동_횡배관"``.
        image_zip: Path of the ``TS_``/``VS_`` image zip.
        label_zip: Path of the ``TL_``/``VL_`` label zip.
    """

    split: AIHubSplit
    category: str
    image_zip: Path
    label_zip: Path


@dataclass(frozen=True)
class AIHubLabelRecord:
    """One parsed VOC label, keyed by its zip member name.

    Attributes:
        member_stem: Member name without leading ``/`` and suffix; also the image id.
        split: Provided split folder.
        category: Category suffix of the zip.
        boxes: Boxes in pixels of the 256x256 crop.
        session: XML ``<source>`` (primary split key).
        session_from_name: Session derived from ``member_stem``.
        crop_key: ``member_stem`` without the ``_g<gain>_`` part.
        gain: Gain parsed from ``member_stem``.
        filename_field: XML ``<filename>`` (may not match ``member_stem``).
        xml_fields: All leaf fields directly under ``<annotation>`` as text.
    """

    member_stem: str
    split: AIHubSplit
    category: str
    boxes: tuple[Box, ...]
    session: str
    session_from_name: str
    crop_key: str
    gain: float
    filename_field: str
    xml_fields: Mapping[str, str] = field(default_factory=dict)

    @property
    def has_session_mismatch(self) -> bool:
        """True when ``<source>`` disagrees with the name-derived session."""
        return self.session != self.session_from_name

    @property
    def has_filename_mismatch(self) -> bool:
        """True when ``<filename>`` disagrees with the member name."""
        return self.filename_field != self.member_stem


def session_from_member_name(member_stem: str) -> str:
    """Derive the ``<source>``-style session from a member name.

    ``"NM_20221017(134135)_N00_..."`` -> ``"NM_2022-10-17_(13_41_35)"``.

    Raises:
        ValueError: If the name does not start with ``XX_YYYYMMDD(HHMMSS)``.
    """
    match = _NAME_SESSION_PATTERN.match(member_stem)
    if match is None:
        raise ValueError(f"Cannot derive session from member name {member_stem!r}")
    parts = match.groupdict()
    return (
        f"{parts['equipment']}_{parts['year']}-{parts['month']}-{parts['day']}"
        f"_({parts['hour']}_{parts['minute']}_{parts['second']})"
    )


def gain_from_member_name(member_stem: str) -> float:
    """Return the gain encoded as ``_g<gain>_`` in a member name.

    Raises:
        ValueError: If the name has no gain part.
    """
    match = _GAIN_PATTERN.search(member_stem)
    if match is None:
        raise ValueError(f"No _g<gain>_ part in member name {member_stem!r}")
    return float(match.group("gain"))


def crop_key_from_member_name(member_stem: str) -> str:
    """Drop the ``_g<gain>_`` part so the three gain copies share one key."""
    return _GAIN_PATTERN.sub("_", member_stem, count=1)


def member_stem(member_name: str) -> str:
    """``"/NM_..._x(1-257).xml"`` -> ``"NM_..._x(1-257)"``."""
    return PurePosixPath(member_name).stem


def _parse_box(object_element: ET.Element, description: str) -> Box:
    label = object_element.findtext("name", default="")
    bndbox = object_element.find("bndbox")
    if bndbox is None:
        raise ValueError(f"{description}: <object> without <bndbox>")
    x1, y1, x2, y2 = (
        float(bndbox.findtext(tag, default="nan"))
        for tag in ("xmin", "ymin", "xmax", "ymax")
    )
    return Box(map_aihub_label(label), x1, y1, x2, y2, source_label=label)


def parse_voc_xml(
    xml_bytes: bytes, description: str
) -> tuple[tuple[Box, ...], dict[str, str]]:
    """Parse an AIHub VOC annotation.

    Args:
        xml_bytes: Raw XML.
        description: Used in error messages (e.g. zip and member name).

    Returns:
        Boxes and the leaf fields directly under ``<annotation>`` (``filename``,
        ``source``, ``gainvalue``, ``equipment``, ...), plus ``width``/``height``
        from ``<size>``.

    Raises:
        ValueError: On a malformed object.
        KeyError: On an unknown class label.
    """
    # The XML comes from the AIHub release kept read-only on local disk.
    root = ET.fromstring(xml_bytes)
    fields = {
        child.tag: (child.text or "").strip() for child in root if len(child) == 0
    }
    fields["width"] = root.findtext("size/width", default="")
    fields["height"] = root.findtext("size/height", default="")
    boxes = tuple(_parse_box(element, description) for element in root.iter("object"))
    return boxes, fields


def list_aihub_bscan_zip_pairs(root: Path, split: AIHubSplit) -> list[AIHubZipPair]:
    """Find the B-scan image/label zip pairs of one split.

    Args:
        root: AIHub ``01-1.정식개방데이터`` folder.
        split: Provided split folder.

    Raises:
        FileNotFoundError: If a label zip has no matching image zip.
    """
    image_prefix, label_prefix = _ZIP_PREFIXES[split]
    label_dir = root / split.value / LABEL_FOLDER
    image_dir = root / split.value / IMAGE_FOLDER
    pairs = []
    for label_zip in sorted(label_dir.glob(f"{label_prefix}{BSCAN_ZIP_INFIX}*.zip")):
        image_zip = image_dir / label_zip.name.replace(label_prefix, image_prefix, 1)
        if not image_zip.is_file():
            raise FileNotFoundError(
                f"No image zip for {label_zip}: expected {image_zip}"
            )
        category = label_zip.stem.removeprefix(f"{label_prefix}{BSCAN_ZIP_INFIX}")
        pairs.append(AIHubZipPair(split, category, image_zip, label_zip))
    return pairs


class AIHubZipReader:
    """Read labels and images of one :class:`AIHubZipPair` with handles opened once.

    Use as a context manager. Not safe to share across processes: create one per
    DataLoader worker.
    """

    def __init__(self, pair: AIHubZipPair) -> None:
        self.pair = pair
        self._label_zip = zipfile.ZipFile(pair.label_zip)
        try:
            self._image_zip = zipfile.ZipFile(pair.image_zip)
        except (OSError, zipfile.BadZipFile):
            self._label_zip.close()
            raise
        self._image_member_by_stem = {
            member_stem(name): name
            for name in self._image_zip.namelist()
            if name.lower().endswith(IMAGE_SUFFIX)
        }

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close both zip handles."""
        self._label_zip.close()
        self._image_zip.close()

    def label_member_names(self) -> list[str]:
        """XML member names in stored order."""
        return [
            name
            for name in self._label_zip.namelist()
            if name.lower().endswith(LABEL_SUFFIX)
        ]

    def read_record(self, label_member: str) -> AIHubLabelRecord:
        """Parse one XML member into a record."""
        stem = member_stem(label_member)
        description = f"{self.pair.label_zip.name}:{label_member}"
        boxes, fields = parse_voc_xml(self._label_zip.read(label_member), description)
        return AIHubLabelRecord(
            member_stem=stem,
            split=self.pair.split,
            category=self.pair.category,
            boxes=boxes,
            session=fields.get("source", ""),
            session_from_name=session_from_member_name(stem),
            crop_key=crop_key_from_member_name(stem),
            gain=gain_from_member_name(stem),
            filename_field=fields.get("filename", ""),
            xml_fields=fields,
        )

    def iter_records(self, limit: int | None = None) -> Iterator[AIHubLabelRecord]:
        """Yield label records in stored order without decoding images."""
        for label_member in self.label_member_names()[:limit]:
            yield self.read_record(label_member)

    def image_member_name(self, stem: str) -> str:
        """Return the image zip member (e.g. ``"/NM_....jpg"``) paired with ``stem``.

        Raises:
            KeyError: If the image zip has no member with that stem.
        """
        image_member = self._image_member_by_stem.get(stem)
        if image_member is None:
            raise KeyError(
                f"{self.pair.image_zip.name} has no image for label {stem!r}"
            )
        return image_member

    def read_image_bytes(self, stem: str) -> bytes:
        """Return the encoded JPG bytes of the image paired with ``stem``.

        Raises:
            KeyError: If the image zip has no member with that stem.
        """
        return self._image_zip.read(self.image_member_name(stem))

    def read_image(self, stem: str) -> np.ndarray:
        """Decode the image whose member stem is ``stem`` (grayscale uint8).

        Raises:
            KeyError: If the image zip has no member with that stem.
        """
        image_member = self.image_member_name(stem)
        description = f"{self.pair.image_zip.name}:{image_member}"
        return decode_grayscale(self._image_zip.read(image_member), description)

    def to_annotated(self, record: AIHubLabelRecord) -> AnnotatedImage:
        """Attach the decoded image to a label record."""
        return AnnotatedImage(
            image=self.read_image(record.member_stem),
            boxes=record.boxes,
            dataset=DATASET_KEY,
            image_id=record.member_stem,
            session=record.session,
            session_from_name=record.session_from_name,
            crop_key=record.crop_key,
            gain=record.gain,
            metadata={
                "split": record.split.value,
                "category": record.category,
                **record.xml_fields,
            },
        )

    def iter_annotated(self, limit: int | None = None) -> Iterator[AnnotatedImage]:
        """Yield decoded, labelled images in stored order."""
        for record in self.iter_records(limit):
            yield self.to_annotated(record)
