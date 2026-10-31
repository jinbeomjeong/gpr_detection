"""Derive the auto-merged ``pipe_band`` dataset v2 from the AIHub YOLO dataset v1.

User decision 2026-10-31 (``labeling/pipe_band_relabel_plan.md`` section 8):
only the class 1 (``pipe_band``) boxes change, by :mod:`band_merge`; every
other byte of the dataset stays as in v1. The v1 folder is only read.

Layout under ``out_dir`` (same as v1, ``aihub_yolo_writer``):

* ``images/*.jpg`` - hard links to the v1 images (a copy when linking fails)
* ``labels/*.txt`` - v1 label lines; merged band boxes replace their parts
* ``fold{k}_images.txt``, ``test_images.txt`` - byte copies of the v1 lists
* ``fold{k}.yaml``, ``test.yaml`` - the v1 yamls with ``path`` set to ``out_dir``
* ``test_v1/`` - the test subset with the **v1** labels, for scoring the
  test hold-out against both ground truths: ``images/`` (hard links),
  ``labels/`` (v1 bytes), ``test_images.txt`` and ``test.yaml`` (trains on the
  v2 train pool, validates on the v1 test labels)
* ``relabel_log.csv`` - one row per image: band boxes before and after, gaps
* ``summary.json`` - rule, box counts per class, subset and fold, merge gaps

Plus the manifest ``data/splits/<out name>.csv``: the v1 rows and columns,
``n_boxes`` recounted, and three new columns ``band_relabel``,
``band_boxes_v1`` and ``band_boxes_v2``.
"""

import bisect
import csv
import itertools
import json
import logging
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from gpr_detection.data.aihub_yolo import MANIFEST_COLUMNS, YOLO_COORDINATE_DECIMALS
from gpr_detection.data.band_merge import MergeRule, PixelBox, merge_band_boxes
from gpr_detection.data.splits import DEFAULT_GAIN_PRIORITY, select_crop_representatives
from gpr_detection.io._common import decode_grayscale
from gpr_detection.io.classes import DetectionClass, class_names

logger = logging.getLogger(__name__)

IMAGES_DIR_NAME = "images"
LABELS_DIR_NAME = "labels"
TEST_V1_DIR_NAME = "test_v1"
TEST_LIST_NAME = "test_images.txt"
TEST_YAML_NAME = "test.yaml"
RELABEL_LOG_NAME = "relabel_log.csv"
SUMMARY_FILE_NAME = "summary.json"
FOLD_COUNT = 5
TEST_FOLD = "-1"
SUBSET_TEST = "test"
BAND_CLASS_ID = int(DetectionClass.PIPE_BAND)
NEW_MANIFEST_COLUMNS = ("band_relabel", "band_boxes_v1", "band_boxes_v2")
V2_MANIFEST_COLUMNS = (*MANIFEST_COLUMNS, *NEW_MANIFEST_COLUMNS)
RELABEL_LOG_COLUMNS = (
    "file_name",
    "crop_key",
    "gain",
    "subset",
    "fold",
    "session",
    "band_boxes_v1",
    "band_boxes_v2",
    "merges",
    "merge_gaps_px",
    "height_blocked_pairs",
)
# Upper edges of the merge-gap histogram bins in pixels (gap <= edge).
GAP_BIN_EDGES_PX = (0.0, 8.0, 16.0, 32.0)


class BandRelabel(StrEnum):
    """Manifest ``band_relabel`` value of one image."""

    AUTO_MERGED = "auto_merged"
    AUTO_UNCHANGED = "auto_unchanged"
    NOT_IN_SCOPE = "not_in_scope"


@dataclass(frozen=True)
class MergeBuildConfig:
    """Inputs and outputs of one v2 build.

    Attributes:
        source_dir: v1 dataset folder (read only).
        source_manifest: v1 manifest CSV (read only).
        out_dir: New dataset folder; must be empty or absent.
        manifest_path: New manifest CSV; must not exist.
        rule: Band merge thresholds.
        dry_run: Merge in memory and report; write nothing.
    """

    source_dir: Path
    source_manifest: Path
    out_dir: Path
    manifest_path: Path
    rule: MergeRule
    dry_run: bool = False


@dataclass(frozen=True)
class LabelLine:
    """One YOLO label line, keeping the original text for unchanged boxes."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    text: str

    def to_pixel_box(self, image_width_px: int, image_height_px: int) -> PixelBox:
        return PixelBox(
            (self.x_center - self.width / 2) * image_width_px,
            (self.y_center - self.height / 2) * image_height_px,
            (self.x_center + self.width / 2) * image_width_px,
            (self.y_center + self.height / 2) * image_height_px,
        )


@dataclass(frozen=True)
class LabelMerge:
    """Merged label text of one image and what changed."""

    text: str
    band_boxes_before: int
    band_boxes_after: int
    merge_gaps_px: tuple[float, ...]
    height_blocked_pairs: int


@dataclass(frozen=True)
class _ImageRecord:
    """One manifest row with its v1 image, v1 label text and merge result."""

    row: dict[str, str]
    image_path: Path
    v1_text: str
    merge: LabelMerge


# --- label text ------------------------------------------------------------------


def parse_label_text(text: str, description: str) -> list[LabelLine]:
    """Parse YOLO ``class xc yc w h`` lines (blank lines are skipped).

    Raises:
        ValueError: On a line without exactly five fields.
    """
    lines = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 5:
            raise ValueError(f"{description}:{line_number}: expected 5 fields")
        x_center, y_center, width, height = (float(value) for value in fields[1:])
        lines.append(
            LabelLine(int(fields[0]), x_center, y_center, width, height, stripped)
        )
    return lines


def _union_line(members: Sequence[LabelLine]) -> str:
    """Union of normalised boxes, written like ``YoloBox.to_line``."""
    left = min(line.x_center - line.width / 2 for line in members)
    right = max(line.x_center + line.width / 2 for line in members)
    top = min(line.y_center - line.height / 2 for line in members)
    bottom = max(line.y_center + line.height / 2 for line in members)
    values = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
    coordinates = " ".join(f"{value:.{YOLO_COORDINATE_DECIMALS}f}" for value in values)
    return f"{BAND_CLASS_ID} {coordinates}"


def merge_label_text(
    text: str,
    rule: MergeRule,
    image_width_px: int,
    image_height_px: int,
    description: str = "label",
) -> LabelMerge:
    """Merge the ``pipe_band`` boxes of one label file.

    Lines of other classes and band boxes that were not merged keep their
    original text and position. A merged box is written where its first part
    was; its other parts are dropped.

    Args:
        text: Label file content.
        rule: Band merge thresholds (gap in pixels of this image).
        image_width_px: Image width used to convert to pixels.
        image_height_px: Image height used to convert to pixels.
        description: Used in error messages.

    Returns:
        The new text (one line per box, ``\\n`` ends) and the merge counts.
    """
    lines = parse_label_text(text, description)
    band_positions = [
        i for i, line in enumerate(lines) if line.class_id == BAND_CLASS_ID
    ]
    band_boxes = [
        lines[i].to_pixel_box(image_width_px, image_height_px) for i in band_positions
    ]
    result = merge_band_boxes(band_boxes, rule)

    replacement_at: dict[int, str] = {}
    dropped: set[int] = set()
    for members in result.member_indices:
        positions = [band_positions[member] for member in members]
        if len(positions) == 1:
            continue
        replacement_at[positions[0]] = _union_line([lines[p] for p in positions])
        dropped.update(positions[1:])

    output_lines = []
    for position, line in enumerate(lines):
        if position in dropped:
            continue
        output_lines.append(replacement_at.get(position, line.text))
    return LabelMerge(
        text="".join(f"{line}\n" for line in output_lines),
        band_boxes_before=len(band_boxes),
        band_boxes_after=len(result.boxes),
        merge_gaps_px=result.merge_gaps_px,
        height_blocked_pairs=result.height_blocked_pairs,
    )


# --- reading v1 ------------------------------------------------------------------


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read a manifest CSV (UTF-8, optional BOM) as a list of row dicts."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _label_path(dataset_dir: Path, file_name: str) -> Path:
    return (dataset_dir / LABELS_DIR_NAME / file_name).with_suffix(".txt")


def _image_size_px(image_path: Path) -> tuple[int, int]:
    """``(width, height)`` of a JPG, decoded once to read its real size."""
    image = decode_grayscale(image_path.read_bytes(), str(image_path))
    height_px, width_px = image.shape[:2]
    return width_px, height_px


def _merge_images(
    rows: Sequence[dict[str, str]], source_dir: Path, rule: MergeRule
) -> list[_ImageRecord]:
    records = []
    for row in rows:
        file_name = row["file_name"]
        image_path = source_dir / IMAGES_DIR_NAME / file_name
        width_px, height_px = _image_size_px(image_path)
        v1_text = _label_path(source_dir, file_name).read_text(encoding="utf-8")
        merge = merge_label_text(v1_text, rule, width_px, height_px, file_name)
        records.append(_ImageRecord(row, image_path, v1_text, merge))
    return records


def _check_gain_copies_agree(records: Sequence[_ImageRecord]) -> None:
    """Gain copies of a crop share their v1 labels, so they must merge alike."""
    texts_by_crop: dict[str, set[str]] = defaultdict(set)
    for record in records:
        texts_by_crop[record.row["crop_key"]].add(record.merge.text)
    differing = sorted(key for key, texts in texts_by_crop.items() if len(texts) > 1)
    if differing:
        raise ValueError(
            f"{len(differing)} crops have gain copies with different merged labels, "
            f"e.g. {differing[:3]}"
        )


# --- manifest and log rows -------------------------------------------------------


def _band_relabel(merge: LabelMerge) -> BandRelabel:
    if merge.band_boxes_before == 0:
        return BandRelabel.NOT_IN_SCOPE
    if merge.merge_gaps_px:
        return BandRelabel.AUTO_MERGED
    return BandRelabel.AUTO_UNCHANGED


def _manifest_row(record: _ImageRecord) -> dict[str, str]:
    merge = record.merge
    row = {column: record.row[column] for column in MANIFEST_COLUMNS}
    box_count_change = merge.band_boxes_before - merge.band_boxes_after
    row["n_boxes"] = str(int(row["n_boxes"]) - box_count_change)
    row["band_relabel"] = _band_relabel(merge).value
    row["band_boxes_v1"] = str(merge.band_boxes_before)
    row["band_boxes_v2"] = str(merge.band_boxes_after)
    return row


def _log_row(record: _ImageRecord) -> dict[str, str]:
    merge, row = record.merge, record.row
    return {
        "file_name": row["file_name"],
        "crop_key": row["crop_key"],
        "gain": row["gain"],
        "subset": row["subset"],
        "fold": row["fold"],
        "session": row["session"],
        "band_boxes_v1": str(merge.band_boxes_before),
        "band_boxes_v2": str(merge.band_boxes_after),
        "merges": str(len(merge.merge_gaps_px)),
        "merge_gaps_px": ";".join(f"{gap:.3f}" for gap in merge.merge_gaps_px),
        "height_blocked_pairs": str(merge.height_blocked_pairs),
    }


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# --- summary ---------------------------------------------------------------------


def _gap_bin_labels() -> list[str]:
    """Bin labels in ascending order: ``<=0``, ``(0,8]``, ..., ``>32``."""
    labels = [f"<={GAP_BIN_EDGES_PX[0]:g}"]
    for lower, upper in itertools.pairwise(GAP_BIN_EDGES_PX):
        labels.append(f"({lower:g},{upper:g}]")
    labels.append(f">{GAP_BIN_EDGES_PX[-1]:g}")
    return labels


def _gap_histogram(gaps_px: Iterable[float]) -> dict[str, int]:
    labels = _gap_bin_labels()
    counts = dict.fromkeys(labels, 0)
    for gap in gaps_px:
        bin_index = bisect.bisect_left(GAP_BIN_EDGES_PX, gap)
        counts[labels[bin_index]] += 1
    return counts


def _class_box_counts(texts: Iterable[str]) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for text in texts:
        counts.update(line.class_id for line in parse_label_text(text, "summary"))
    return {member.label: counts[member.value] for member in DetectionClass}


def _merge_counts(records: Sequence[_ImageRecord]) -> dict[str, Any]:
    gaps = [gap for record in records for gap in record.merge.merge_gaps_px]
    band_records = [r for r in records if r.merge.band_boxes_before > 0]
    return {
        "images": len(records),
        "images_with_band": len(band_records),
        "images_changed": sum(bool(r.merge.merge_gaps_px) for r in records),
        "band_boxes_v1": sum(r.merge.band_boxes_before for r in records),
        "band_boxes_v2": sum(r.merge.band_boxes_after for r in records),
        "merged_pairs": len(gaps),
        "height_blocked_pairs": sum(r.merge.height_blocked_pairs for r in records),
        "gap_histogram_px": _gap_histogram(gaps),
        "classes_v2": _class_box_counts(r.merge.text for r in records),
    }


def _crop_representatives(records: Sequence[_ImageRecord]) -> list[_ImageRecord]:
    by_file = {record.row["file_name"]: record for record in records}
    chosen = select_crop_representatives(
        [record.row for record in records], DEFAULT_GAIN_PRIORITY
    )
    return [by_file[str(row["file_name"])] for row in chosen]


def _part_label(row: Mapping[str, str]) -> str:
    return SUBSET_TEST if row["fold"] == TEST_FOLD else f"fold{row['fold']}"


def _summarize(
    records: Sequence[_ImageRecord], config: MergeBuildConfig
) -> dict[str, Any]:
    """Counts for ``summary.json``: whole set, per part, per image and per crop."""
    by_part: dict[str, list[_ImageRecord]] = defaultdict(list)
    for record in records:
        by_part[_part_label(record.row)].append(record)
    crops = _crop_representatives(records)
    crops_by_part: dict[str, list[_ImageRecord]] = defaultdict(list)
    for record in crops:
        crops_by_part[_part_label(record.row)].append(record)
    return {
        "dataset": "aihub_71327_bscan_band_auto_merge",
        "source_dir": str(config.source_dir),
        "source_manifest": str(config.source_manifest),
        "rule": asdict(config.rule),
        "gap_unit": "pixels of the original crop image",
        "class_names": class_names(),
        "classes_v1": _class_box_counts(record.v1_text for record in records),
        "image_level": _merge_counts(records),
        "crop_level": _merge_counts(crops),
        "parts_image_level": {
            part: _merge_counts(by_part[part]) for part in sorted(by_part)
        },
        "parts_crop_level": {
            part: _merge_counts(crops_by_part[part]) for part in sorted(crops_by_part)
        },
        "test_v1_dir": TEST_V1_DIR_NAME,
    }


# --- writing ---------------------------------------------------------------------


def _ensure_new_outputs(config: MergeBuildConfig) -> None:
    if config.out_dir.exists() and any(config.out_dir.iterdir()):
        raise FileExistsError(f"Output folder {config.out_dir} is not empty")
    if config.manifest_path.exists():
        raise FileExistsError(f"Manifest {config.manifest_path} already exists")
    if config.out_dir.resolve() == config.source_dir.resolve():
        raise ValueError("out_dir must differ from source_dir")


def _link_or_copy(source: Path, destination: Path) -> bool:
    """Hard-link ``destination`` to ``source``; copy if linking fails.

    Returns:
        True if a hard link was made.
    """
    try:
        destination.hardlink_to(source)
    except OSError as error:
        logger.warning("hard link failed (%s); copying %s", error, source.name)
        shutil.copy2(source, destination)
        return False
    return True


def _write_images_and_labels(
    records: Sequence[_ImageRecord], dataset_dir: Path, label_texts: Sequence[str]
) -> int:
    """Link the images and write the labels; return the number of hard links."""
    images_dir = dataset_dir / IMAGES_DIR_NAME
    labels_dir = dataset_dir / LABELS_DIR_NAME
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for record, label_text in zip(records, label_texts, strict=True):
        file_name = record.row["file_name"]
        linked += _link_or_copy(record.image_path, images_dir / file_name)
        _label_path(dataset_dir, file_name).write_text(label_text, encoding="utf-8")
    return linked


def fold_list_name(fold: str) -> str:
    """Image list of a manifest fold (``"-1"`` is the test list)."""
    return TEST_LIST_NAME if fold == TEST_FOLD else f"fold{fold}_images.txt"


def fold_yaml_name(fold: str) -> str:
    """Ultralytics data yaml of a manifest fold (``"-1"`` is ``test.yaml``)."""
    return TEST_YAML_NAME if fold == TEST_FOLD else f"fold{fold}.yaml"


def _dump_yaml(path: Path, data: Mapping[str, Any]) -> None:
    text = yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")


def _write_lists_and_yamls(source_dir: Path, out_dir: Path) -> None:
    """Copy the v1 lists byte for byte; rewrite the yamls' ``path`` only."""
    folds = [*(str(k) for k in range(FOLD_COUNT)), TEST_FOLD]
    for fold in folds:
        shutil.copyfile(
            source_dir / fold_list_name(fold), out_dir / fold_list_name(fold)
        )
        source_yaml = source_dir / fold_yaml_name(fold)
        data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
        data["path"] = str(out_dir.resolve())
        _dump_yaml(out_dir / fold_yaml_name(fold), data)


def _write_test_v1(
    records: Sequence[_ImageRecord], source_dir: Path, out_dir: Path
) -> int:
    """Test subset with the v1 labels; returns the number of hard links."""
    test_dir = out_dir / TEST_V1_DIR_NAME
    test_records = [r for r in records if r.row["fold"] == TEST_FOLD]
    v1_label_texts = [record.v1_text for record in test_records]
    linked = _write_images_and_labels(test_records, test_dir, v1_label_texts)
    shutil.copyfile(source_dir / TEST_LIST_NAME, test_dir / TEST_LIST_NAME)

    train_lists = [
        str((out_dir / fold_list_name(str(k))).resolve()) for k in range(FOLD_COUNT)
    ]
    data = {
        "path": str(test_dir.resolve()),
        # Train = the v2 train pool (same as ../test.yaml); val = v1 test labels.
        "train": train_lists,
        "val": TEST_LIST_NAME,
        "test": TEST_LIST_NAME,
        "names": class_names(),
    }
    _dump_yaml(test_dir / TEST_YAML_NAME, data)
    return linked


def build_merged_dataset(config: MergeBuildConfig) -> dict[str, Any]:
    """Merge the v1 band boxes and write the v2 dataset (unless dry run).

    Returns:
        The summary (also written to ``summary.json`` when not a dry run).

    Raises:
        FileExistsError: If the outputs already exist (not for dry runs).
        ValueError: If gain copies of one crop merge differently.
    """
    if not config.dry_run:
        _ensure_new_outputs(config)
    rows = read_manifest(config.source_manifest)
    records = _merge_images(rows, config.source_dir, config.rule)
    _check_gain_copies_agree(records)
    summary = _summarize(records, config)
    if config.dry_run:
        return summary

    out_dir = config.out_dir
    linked = _write_images_and_labels(
        records, out_dir, [record.merge.text for record in records]
    )
    linked += _write_test_v1(records, config.source_dir, out_dir)
    _write_lists_and_yamls(config.source_dir, out_dir)
    _write_csv(config.manifest_path, V2_MANIFEST_COLUMNS, map(_manifest_row, records))
    _write_csv(out_dir / RELABEL_LOG_NAME, RELABEL_LOG_COLUMNS, map(_log_row, records))
    summary["hard_links"] = linked
    summary["manifest"] = str(config.manifest_path)
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    (out_dir / SUMMARY_FILE_NAME).write_text(summary_text, encoding="utf-8")
    logger.info("wrote %d images to %s (%d hard links)", len(records), out_dir, linked)
    return summary
