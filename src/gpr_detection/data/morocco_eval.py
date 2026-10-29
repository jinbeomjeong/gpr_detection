"""Morocco cross-domain evaluation set: one near-original copy per split group.

Decisions implemented here (docs/CHECKLIST.md M1-4, user decision 2026-10-29):

* Evaluation only, never training. The provided train/val/test split is kept
  in the manifest column ``split_src`` but all images form one evaluation set.
* Rotation and elastic copies are dropped (``is_geometry_distorting``). Each
  provided split group (``group_id``, near-duplicate stems share one) then
  contributes exactly **one** image:

  1. **Preferred**: the copy readme_morocco.md section 5 calls "almost identical
     to the original" - ``_aug_6`` (utilities), ``_aug_13`` (cavities),
     ``_aug_12`` (intact).
  2. **Fallback** when the group has no preferred copy: among the copies whose
     ``aug_transform`` leaves the geometry untouched (noise, frequency shift,
     "almost identical"; see ``GEOMETRY_PRESERVING_TRANSFORM_PREFIXES``), the
     first in deterministic order.
  3. A group without any such copy is left out and listed in the summary.

  When a group holds several stems, candidates are ordered by the stem's
  position in ``split_groups.csv`` (the first stem names the group), then by
  augmentation number.
* Intact images are kept with empty label files (false-positive evaluation).
* Cavity images are kept and flagged ``anisotropic_stretch=True``: their
  originals were stretched to 224x224 ignoring the aspect ratio, so the
  hyperbolas are distorted and are reported separately.
* Image bytes are copied unchanged at their original size (no resizing, no
  re-encoding). Class ids are remapped to the project's four classes.

Output layout under ``out_dir``: ``images/*.jpg``, ``labels/*.txt``,
``eval_images.txt``, ``eval.yaml`` and ``summary.json``; plus the manifest CSV.
"""

import csv
import json
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from gpr_detection.data.aihub_yolo import YoloBox
from gpr_detection.data.aihub_yolo_writer import (
    IMAGES_DIR_NAME,
    LABELS_DIR_NAME,
    SUMMARY_FILE_NAME,
    _ensure_empty_output,
)
from gpr_detection.io._common import decode_grayscale
from gpr_detection.io.classes import DetectionClass, class_names, map_morocco_class_id
from gpr_detection.io.morocco import (
    YOLO_FIELDS_PER_LINE,
    MoroccoImageRef,
    is_geometry_distorting,
    list_morocco_images,
    read_morocco_group_stems,
)

logger = logging.getLogger(__name__)

DATASET_NAME = "morocco_eval"
EVAL_LIST_NAME = "eval_images.txt"
EVAL_YAML_NAME = "eval.yaml"
IMAGE_SUFFIX = ".jpg"
_AUG_NUMBER_PATTERN = re.compile(r"_aug_(\d+)$")
# Transforms that change pixel values but not where the reflections are.
# "잡음" is plain noise, "강한 잡음" strong noise (intact subset only).
GEOMETRY_PRESERVING_TRANSFORM_PREFIXES = (
    "원본과 거의 동일",
    "주파수 이동",
    "가우시안 잡음",
    "잡음",
    "강한 잡음",
)
_YAML_HEADER = (
    "# Morocco cross-domain EVALUATION set (checklist M1-4). Never train on it:\n"
    "# 'train' points at the eval list only because Ultralytics requires the key.\n"
)

MANIFEST_COLUMNS = (
    "file_name",
    "image_id",
    "subset",
    "group_id",
    "stem",
    "split_src",
    "aug_number",
    "aug_transform",
    "selection_rule",
    "width_px",
    "height_px",
    "classes",
    "n_boxes",
    "anisotropic_stretch",
    "source_image",
)


class MoroccoSubset(StrEnum):
    """Values of the ``subset`` column of images_index.csv."""

    UTILITIES = "utilities"
    CAVITIES = "cavities"
    INTACT = "intact"


class SelectionRule(StrEnum):
    """How the copy of a group was chosen (manifest ``selection_rule``)."""

    PREFERRED = "preferred"
    FALLBACK = "fallback"


PREFERRED_AUG_NUMBER: dict[MoroccoSubset, int] = {
    MoroccoSubset.UTILITIES: 6,
    MoroccoSubset.CAVITIES: 13,
    MoroccoSubset.INTACT: 12,
}
# Classes each subset may contain; intact images must have no boxes.
ALLOWED_CLASSES: dict[MoroccoSubset, frozenset[DetectionClass]] = {
    MoroccoSubset.UTILITIES: frozenset({DetectionClass.PIPE_HYPERBOLA}),
    MoroccoSubset.CAVITIES: frozenset({DetectionClass.CAVITY}),
    MoroccoSubset.INTACT: frozenset(),
}
ANISOTROPIC_SUBSETS = frozenset({MoroccoSubset.CAVITIES})


class DuplicateGroupError(ValueError):
    """Raised when a split group contributes more than one evaluation image."""


@dataclass(frozen=True)
class GroupSelection:
    """The copy chosen for one split group.

    Attributes:
        ref: Chosen image.
        rule: Preferred copy or fallback.
    """

    ref: MoroccoImageRef
    rule: SelectionRule

    @property
    def subset(self) -> MoroccoSubset:
        """Subset of the chosen image."""
        return MoroccoSubset(self.ref.subset)


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of :func:`select_eval_images`.

    Attributes:
        selections: One entry per group, sorted by subset and group id.
        groups_without_copy: Groups with no geometry-preserving copy, sorted.
        labelled_images: Images of the YOLO folder before any exclusion.
        geometry_distorting_images: Rotation and elastic copies dropped.
    """

    selections: tuple[GroupSelection, ...]
    groups_without_copy: tuple[str, ...]
    labelled_images: int
    geometry_distorting_images: int


@dataclass(frozen=True)
class EvalItem:
    """A selected image with its decoded size and remapped boxes.

    Attributes:
        selection: Group selection the item comes from.
        width_px: Decoded image width.
        height_px: Decoded image height.
        boxes: Boxes with project class ids (empty for intact images).
    """

    selection: GroupSelection
    width_px: int
    height_px: int
    boxes: tuple[YoloBox, ...]

    @property
    def file_name(self) -> str:
        """Output image file name (the Morocco ``image_id`` is ASCII-safe)."""
        return f"{self.selection.ref.image_id}{IMAGE_SUFFIX}"

    @property
    def class_ids(self) -> tuple[DetectionClass, ...]:
        """Distinct box classes, ascending."""
        return tuple(sorted({box.class_id for box in self.boxes}))


@dataclass(frozen=True)
class MoroccoEvalConfig:
    """Options of one evaluation-set build.

    Attributes:
        morocco_root: Morocco package root (read only).
        out_dir: Output folder for images, labels, list, yaml and summary.
        manifest_path: Output manifest CSV.
        dry_run: Select and read labels only; write nothing.
    """

    morocco_root: Path
    out_dir: Path
    manifest_path: Path
    dry_run: bool = False


# --- selection -----------------------------------------------------------------


def augmentation_number(image_id: str) -> int:
    """Return N of an image id ending in ``_aug_N``.

    Raises:
        ValueError: If the id has no ``_aug_N`` suffix.
    """
    match = _AUG_NUMBER_PATTERN.search(image_id)
    if match is None:
        raise ValueError(f"Image id {image_id!r} does not end in _aug_<number>")
    return int(match.group(1))


def is_geometry_preserving(aug_transform: str) -> bool:
    """True for noise, frequency-shift and "almost identical" copies."""
    return aug_transform.startswith(GEOMETRY_PRESERVING_TRANSFORM_PREFIXES)


def _candidate_order(
    group_refs: Iterable[MoroccoImageRef], group_stems: Sequence[str]
) -> list[MoroccoImageRef]:
    """Sort by the stem's position in the group, then by augmentation number."""

    def sort_key(ref: MoroccoImageRef) -> tuple[int, int]:
        if ref.stem not in group_stems:
            raise ValueError(
                f"{ref.image_id}: stem {ref.stem!r} is not listed for group "
                f"{ref.group_id} ({list(group_stems)})"
            )
        return (group_stems.index(ref.stem), augmentation_number(ref.image_id))

    return sorted(group_refs, key=sort_key)


def select_group_copy(
    group_refs: Sequence[MoroccoImageRef], group_stems: Sequence[str]
) -> GroupSelection | None:
    """Choose the single evaluation image of one split group.

    Args:
        group_refs: Copies of the group, rotation and elastic already removed.
        group_stems: Stems of the group in ``split_groups.csv`` order.

    Returns:
        The preferred copy if present, else the first geometry-preserving copy,
        else None.

    Raises:
        ValueError: If the refs mix subsets, or a preferred copy's transform is
            not geometry preserving.
    """
    subsets = {ref.subset for ref in group_refs}
    if len(subsets) != 1:
        raise ValueError(f"Group refs must share one subset, got {sorted(subsets)}")
    preferred_number = PREFERRED_AUG_NUMBER[MoroccoSubset(subsets.pop())]
    ordered = _candidate_order(group_refs, group_stems)

    for ref in ordered:
        if augmentation_number(ref.image_id) != preferred_number:
            continue
        if not is_geometry_preserving(ref.aug_transform):
            raise ValueError(
                f"{ref.image_id}: preferred copy has transform {ref.aug_transform!r}"
            )
        return GroupSelection(ref, SelectionRule.PREFERRED)

    for ref in ordered:
        if is_geometry_preserving(ref.aug_transform):
            return GroupSelection(ref, SelectionRule.FALLBACK)
    return None


def assert_one_image_per_group(selections: Iterable[GroupSelection]) -> None:
    """Check that no split group appears twice.

    Raises:
        DuplicateGroupError: Listing the repeated group ids.
    """
    counts = Counter(selection.ref.group_id for selection in selections)
    repeated = sorted(group for group, count in counts.items() if count > 1)
    if repeated:
        raise DuplicateGroupError(f"Groups selected more than once: {repeated}")


def select_eval_images(
    refs: Sequence[MoroccoImageRef], group_stems: Mapping[str, Sequence[str]]
) -> SelectionResult:
    """Drop rotation/elastic copies and pick one image per split group.

    Args:
        refs: Every labelled image of the YOLO folder.
        group_stems: Stems per group id (``read_morocco_group_stems``).

    Raises:
        KeyError: If a ref's group is missing from ``group_stems``.
        DuplicateGroupError: Never expected; guards the one-per-group rule.
    """
    kept_refs = [ref for ref in refs if not is_geometry_distorting(ref.aug_transform)]
    refs_by_group: dict[str, list[MoroccoImageRef]] = defaultdict(list)
    for ref in kept_refs:
        refs_by_group[ref.group_id].append(ref)

    selections = []
    groups_without_copy = []
    for group_id in sorted(refs_by_group):
        selection = select_group_copy(refs_by_group[group_id], group_stems[group_id])
        if selection is None:
            groups_without_copy.append(group_id)
        else:
            selections.append(selection)
    selections.sort(key=lambda s: (s.ref.subset, s.ref.group_id))
    assert_one_image_per_group(selections)

    # Groups whose every copy is rotated or elastic never reach refs_by_group.
    fully_distorted = {ref.group_id for ref in refs} - set(refs_by_group)
    return SelectionResult(
        selections=tuple(selections),
        groups_without_copy=tuple(sorted([*groups_without_copy, *fully_distorted])),
        labelled_images=len(refs),
        geometry_distorting_images=len(refs) - len(kept_refs),
    )


# --- labels and images ---------------------------------------------------------


def parse_morocco_label(label_text: str, description: str) -> tuple[YoloBox, ...]:
    """Parse a Morocco YOLO label and remap class ids (0 pipe->0, 1 cavity->2).

    Coordinates are kept as given (the files use six decimals, which
    ``YoloBox.to_line`` writes back unchanged).

    Raises:
        ValueError: On a line without five fields or a coordinate outside 0-1.
        KeyError: On a class id other than 0 or 1.
    """
    boxes = []
    for line in label_text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != YOLO_FIELDS_PER_LINE:
            raise ValueError(f"{description}: expected 5 fields, got {line!r}")
        coordinates = [float(value) for value in fields[1:]]
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"{description}: coordinate outside 0-1 in {line!r}")
        x_center, y_center, width, height = coordinates
        boxes.append(
            YoloBox(
                class_id=map_morocco_class_id(int(fields[0])),
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )
    return tuple(boxes)


def check_subset_classes(
    subset: MoroccoSubset, boxes: Iterable[YoloBox], description: str
) -> None:
    """Check that the boxes only use the classes allowed for ``subset``.

    Raises:
        ValueError: E.g. a cavity box in a utilities image or any intact box.
    """
    allowed = ALLOWED_CLASSES[subset]
    unexpected = sorted({box.class_id for box in boxes} - allowed)
    if unexpected:
        names = [detection_class.label for detection_class in unexpected]
        raise ValueError(f"{description}: {subset.value} image has classes {names}")


def load_eval_item(root: Path, selection: GroupSelection) -> EvalItem:
    """Decode the image (size check) and read the remapped boxes of a selection.

    Raises:
        ValueError: If the image does not decode or the label breaks a rule.
        FileNotFoundError: If the image or label file is missing.
    """
    ref = selection.ref
    image = decode_grayscale((root / ref.image_relpath).read_bytes(), ref.image_id)
    label_text = (root / ref.label_relpath).read_text(encoding="utf-8")
    boxes = parse_morocco_label(label_text, ref.label_relpath)
    check_subset_classes(selection.subset, boxes, ref.image_id)
    height_px, width_px = image.shape
    return EvalItem(selection, width_px, height_px, boxes)


# --- manifest and summary ------------------------------------------------------


def manifest_rows(items: Sequence[EvalItem]) -> list[dict[str, Any]]:
    """One manifest row per item, columns in ``MANIFEST_COLUMNS`` order."""
    rows = []
    for item in items:
        ref = item.selection.ref
        rows.append(
            {
                "file_name": item.file_name,
                "image_id": ref.image_id,
                "subset": ref.subset,
                "group_id": ref.group_id,
                "stem": ref.stem,
                "split_src": ref.split,
                "aug_number": augmentation_number(ref.image_id),
                "aug_transform": ref.aug_transform,
                "selection_rule": item.selection.rule.value,
                "width_px": item.width_px,
                "height_px": item.height_px,
                "classes": ";".join(str(int(c)) for c in item.class_ids),
                "n_boxes": len(item.boxes),
                "anisotropic_stretch": item.selection.subset in ANISOTROPIC_SUBSETS,
                "source_image": ref.image_relpath,
            }
        )
    return rows


def _size_counts(items: Iterable[EvalItem]) -> dict[str, int]:
    sizes = Counter(f"{item.width_px}x{item.height_px}" for item in items)
    return dict(sorted(sizes.items()))


def _part_summary(items: Sequence[EvalItem]) -> dict[str, Any]:
    """Groups, images, boxes per class, sizes and provided splits of ``items``."""
    boxes_by_class = Counter(box.class_id.label for item in items for box in item.boxes)
    rules = Counter(item.selection.rule.value for item in items)
    return {
        "groups": len({item.selection.ref.group_id for item in items}),
        "images": len(items),
        "boxes": sum(len(item.boxes) for item in items),
        "boxes_by_class": dict(sorted(boxes_by_class.items())),
        "images_without_boxes": sum(not item.boxes for item in items),
        "selection_rules": dict(sorted(rules.items())),
        "sizes": _size_counts(items),
        "split_src": dict(
            sorted(Counter(i.selection.ref.split for i in items).items())
        ),
    }


def summarize(selection: SelectionResult, items: Sequence[EvalItem]) -> dict[str, Any]:
    """Counts for ``summary.json`` and the dry-run report."""
    items_by_subset: dict[MoroccoSubset, list[EvalItem]] = defaultdict(list)
    for item in items:
        items_by_subset[item.selection.subset].append(item)
    fallback_ids = [
        item.selection.ref.image_id
        for item in items
        if item.selection.rule is SelectionRule.FALLBACK
    ]
    selected_count = len(selection.selections)
    return {
        "dataset": DATASET_NAME,
        "purpose": "cross-domain evaluation only (never train on it)",
        "class_names": class_names(),
        "preferred_aug_number": {s.value: n for s, n in PREFERRED_AUG_NUMBER.items()},
        "anisotropic_stretch_subsets": sorted(s.value for s in ANISOTROPIC_SUBSETS),
        "source": {
            "labelled_images": selection.labelled_images,
            "excluded_rotation_elastic": selection.geometry_distorting_images,
            "excluded_other_copies": (
                selection.labelled_images
                - selection.geometry_distorting_images
                - selected_count
            ),
            "groups_without_geometry_preserving_copy": list(
                selection.groups_without_copy
            ),
            "originals_used": 0,
        },
        "fallback": {"count": len(fallback_ids), "image_ids": fallback_ids},
        "total": _part_summary(items),
        "subsets": {
            subset.value: _part_summary(items_by_subset[subset])
            for subset in MoroccoSubset
        },
    }


# --- writing -------------------------------------------------------------------


def write_images_and_labels(
    root: Path, items: Sequence[EvalItem], out_dir: Path
) -> None:
    """Copy each selected JPG byte for byte and write its remapped label file.

    Intact images get an empty label file (background sample).
    """
    images_dir = out_dir / IMAGES_DIR_NAME
    labels_dir = out_dir / LABELS_DIR_NAME
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        source_bytes = (root / item.selection.ref.image_relpath).read_bytes()
        (images_dir / item.file_name).write_bytes(source_bytes)
        label_text = "".join(f"{box.to_line()}\n" for box in item.boxes)
        label_path = (labels_dir / item.file_name).with_suffix(".txt")
        label_path.write_text(label_text, encoding="utf-8")


def write_eval_list_and_yaml(items: Sequence[EvalItem], out_dir: Path) -> None:
    """Write ``eval_images.txt`` and the Ultralytics ``eval.yaml``.

    List lines start with ``./`` because Ultralytics resolves them against the
    folder of the list file. ``val`` and ``test`` point at the list; ``train``
    does too only because Ultralytics refuses a data yaml without it.
    """
    lines = "".join(f"./{IMAGES_DIR_NAME}/{item.file_name}\n" for item in items)
    (out_dir / EVAL_LIST_NAME).write_text(lines, encoding="utf-8")
    data_config = {
        "path": str(out_dir.resolve()),
        "train": EVAL_LIST_NAME,
        "val": EVAL_LIST_NAME,
        "test": EVAL_LIST_NAME,
        "names": class_names(),
    }
    yaml_text = yaml.safe_dump(data_config, sort_keys=False, allow_unicode=True)
    (out_dir / EVAL_YAML_NAME).write_text(_YAML_HEADER + yaml_text, encoding="utf-8")


def write_manifest(items: Sequence[EvalItem], manifest_path: Path) -> None:
    """Write the manifest CSV (UTF-8, one row per image)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows(items))


def _assert_manifest_groups_unique(manifest_path: Path) -> None:
    """Re-read the written manifest so the one-per-group check covers the file."""
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        counts = Counter(row["group_id"] for row in csv.DictReader(handle))
    repeated = sorted(group for group, count in counts.items() if count > 1)
    if repeated:
        raise DuplicateGroupError(f"{manifest_path}: groups repeated {repeated}")


def build_morocco_eval(config: MoroccoEvalConfig) -> dict[str, Any]:
    """Select one image per group, read labels and write the set (unless dry run).

    Returns:
        The summary (also written to ``summary.json`` when not a dry run).

    Raises:
        FileExistsError: If ``config.out_dir`` is not empty (not for dry runs).
        DuplicateGroupError: If a group would contribute two images.
        ValueError: On unexpected data.yaml names, labels or classes.
    """
    if not config.dry_run:
        _ensure_empty_output(config.out_dir)

    root = config.morocco_root
    refs = list_morocco_images(root)
    selection = select_eval_images(refs, read_morocco_group_stems(root))
    items = [load_eval_item(root, chosen) for chosen in selection.selections]
    summary = summarize(selection, items)
    logger.info(
        "Selected %d images from %d groups (%d fallback)",
        len(items),
        summary["total"]["groups"],
        summary["fallback"]["count"],
    )
    if config.dry_run:
        return summary

    write_images_and_labels(root, items, config.out_dir)
    write_eval_list_and_yaml(items, config.out_dir)
    write_manifest(items, config.manifest_path)
    _assert_manifest_groups_unique(config.manifest_path)
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    (config.out_dir / SUMMARY_FILE_NAME).write_text(summary_text, encoding="utf-8")
    return summary
