"""Plan the AIHub 71327 B-scan YOLO dataset: test hold-out and session-grouped k-fold.

Decisions implemented here (docs/CHECKLIST.md "M1 결정 사항", ADR-003, user
decision of 2026-10-29 on a held-out test set):

* B-scan zips only, four classes (``classes.py``). Images and labels are paired
  by zip member name, never by the XML ``<filename>``.
* Split groups are the connected components of ``<source>``, the name-derived
  session and the crop key (``splits.connected_group_ids``), so no session and
  no gain copy of a crop can sit on two sides of any split.
* About ``test_fraction`` of the groups form a fixed **test** subset first:
  at least ``min_test_pipe_sessions`` tpipe and lpipe sessions each (drawn from
  per-class session lists), and the same fraction of the cavity-only and
  manhole-only groups. A pipe draw is accepted only if tpipe and lpipe each
  have ``pipe_share_min``-``pipe_share_max`` of their images in test; draws are
  tried on the streams ``"{seed}:test:{attempt}"`` and the first accepted
  attempt is recorded. The rest is the **train pool**.
* In each subset every pipe image is kept and cavity and manhole are each cut
  down to about ``balance_ratio`` x that subset's pipe image count. The draw is
  per gain-free crop key (all gain copies in or out together) and round-robin
  over split groups, so every session keeps a share before any gets a second.
* The provided Training/Validation split is ignored. The train pool is split by
  ``StratifiedGroupKFold`` over split groups, stratified by the primary class.
  Test images carry fold ``TEST_FOLD`` (-1).

Writing the files is in :mod:`gpr_detection.data.aihub_yolo_writer`.
"""

import logging
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from gpr_detection.data.splits import (
    AIHUB_LEAK_KEY_COLUMNS,
    assert_no_fold_leakage,
    assign_group_folds,
    connected_group_ids,
)
from gpr_detection.io.aihub import (
    AIHubLabelRecord,
    AIHubSplit,
    AIHubZipPair,
    AIHubZipReader,
)
from gpr_detection.io.annotated import Box
from gpr_detection.io.classes import DetectionClass

logger = logging.getLogger(__name__)

PIPE_CLASSES = (DetectionClass.PIPE_HYPERBOLA, DetectionClass.PIPE_BAND)
DOWNSAMPLED_CLASSES = (DetectionClass.CAVITY, DetectionClass.MANHOLE)
PIPE_BALANCE_CLASS = "pipe"
TEST_FOLD = -1
# Boxes thinner than one pixel after clipping to the image carry no object.
MIN_BOX_SIDE_PX = 1.0
YOLO_COORDINATE_DECIMALS = 6
IMAGE_SUFFIX = ".jpg"
_UNSAFE_FILE_NAME_CHARS = re.compile(r"[^A-Za-z0-9()._-]")

MANIFEST_COLUMNS = (
    "file_name",
    "member",
    "split_src",
    "session",
    "session_from_name",
    "crop_key",
    "gain",
    "classes",
    "n_boxes",
    "subset",
    "fold",
    "seed",
    "test_attempt",
    "category",
    "primary_class",
    "split_group",
)


class Subset(StrEnum):
    """Top-level partition written to the manifest ``subset`` column."""

    TEST = "test"
    TRAINPOOL = "trainpool"


@dataclass(frozen=True)
class YoloBox:
    """One YOLO label line: class id and box centre/size normalised to 0-1."""

    class_id: DetectionClass
    x_center: float
    y_center: float
    width: float
    height: float

    def to_line(self) -> str:
        """Format as ``"<class> <xc> <yc> <w> <h>"``."""
        values = (self.x_center, self.y_center, self.width, self.height)
        coordinates = " ".join(
            f"{value:.{YOLO_COORDINATE_DECIMALS}f}" for value in values
        )
        return f"{int(self.class_id)} {coordinates}"


@dataclass(frozen=True)
class AIHubYoloItem:
    """One AIHub B-scan image with its YOLO boxes and split keys.

    Attributes:
        member_stem: Zip member stem shared by the JPG and the XML.
        image_member: Image zip member name (with the leading ``/``).
        split: Provided split folder the image came from.
        category: Category suffix of the zip, e.g. ``"비공동_횡배관"``.
        session: XML ``<source>``.
        session_from_name: Session derived from ``member_stem``.
        crop_key: ``member_stem`` without the gain part.
        gain: Gain parsed from ``member_stem``.
        image_width_px: XML ``<size><width>``.
        image_height_px: XML ``<size><height>``.
        boxes: Valid boxes after clipping.
        dropped_box_count: Boxes discarded as degenerate after clipping.
        has_filename_mismatch: XML ``<filename>`` differs from ``member_stem``.
    """

    member_stem: str
    image_member: str
    split: AIHubSplit
    category: str
    session: str
    session_from_name: str
    crop_key: str
    gain: float
    image_width_px: int
    image_height_px: int
    boxes: tuple[YoloBox, ...]
    dropped_box_count: int = 0
    has_filename_mismatch: bool = False

    @property
    def class_ids(self) -> tuple[DetectionClass, ...]:
        """Distinct classes of the boxes, ascending."""
        return tuple(sorted({box.class_id for box in self.boxes}))

    @property
    def primary_class(self) -> DetectionClass:
        """Most frequent box class (ties go to the lower id).

        Raises:
            ValueError: If the item has no boxes.
        """
        if not self.boxes:
            raise ValueError(f"{self.member_stem} has no boxes")
        counts = Counter(box.class_id for box in self.boxes)
        return min(counts, key=lambda class_id: (-counts[class_id], class_id))

    @property
    def has_pipe(self) -> bool:
        """True when any box is a pipe class."""
        return any(box.class_id in PIPE_CLASSES for box in self.boxes)

    @property
    def has_session_mismatch(self) -> bool:
        """True when ``<source>`` disagrees with the name-derived session."""
        return self.session != self.session_from_name

    @property
    def pair_key(self) -> tuple[AIHubSplit, str]:
        """``(split, category)``: identifies the zip pair the item came from."""
        return (self.split, self.category)


@dataclass(frozen=True)
class BalanceResult:
    """Outcome of :func:`select_balanced_items` for one subset.

    Attributes:
        kept_indices: Indices into the input items, ascending.
        target_images_per_class: Image target for each downsampled class.
        available_images: Images per balance class before the draw.
        kept_images: Images per balance class after the draw.
        available_crops: Crop keys per balance class before the draw.
        kept_crops: Crop keys per balance class after the draw.
    """

    kept_indices: tuple[int, ...]
    target_images_per_class: int
    available_images: dict[str, int]
    kept_images: dict[str, int]
    available_crops: dict[str, int]
    kept_crops: dict[str, int]


@dataclass(frozen=True)
class SplitOptions:
    """How to hold out the test subset, downsample and fold the rest.

    Attributes:
        seed: Seed of the test draw, the downsampling draw and the fold split.
        k: Number of folds of the train pool.
        balance_ratio: Cavity/manhole image target as a multiple of pipe images.
        test_fraction: Share of split groups held out as the test subset.
        min_test_pipe_sessions: Minimum tpipe and lpipe sessions each in test.
        pipe_share_min: Lowest accepted test share of each pipe class's images.
        pipe_share_max: Highest accepted test share of each pipe class's images.
        max_test_attempts: Test draws tried before giving up.
    """

    seed: int
    k: int = 5
    balance_ratio: float = 1.0
    test_fraction: float = 0.15
    min_test_pipe_sessions: int = 2
    pipe_share_min: float = 0.10
    pipe_share_max: float = 0.25
    max_test_attempts: int = 1000


class TestSelectionError(ValueError):
    """Raised when no test draw meets the pipe image share bounds."""

    __test__ = False  # keep pytest from collecting this class


@dataclass(frozen=True)
class TestSelection:
    """Accepted test draw.

    Attributes:
        groups: Split groups held out as test.
        attempt: Attempt number; the draw used the random stream
            ``rng_stream_for_test_draw(seed, attempt)``.
        pipe_image_shares: Test share of each pipe class's images (None when
            the class has no images).
    """

    __test__ = False  # keep pytest from collecting this class

    groups: frozenset[int]
    attempt: int
    pipe_image_shares: dict[str, float | None]


@dataclass(frozen=True)
class DatasetPlan:
    """Everything needed to write (or only report) the dataset.

    The per-item tuples are parallel and in manifest order.

    Attributes:
        items: Selected items.
        file_names: Output image file name per item.
        subsets: Test or train pool per item.
        folds: Fold index per item (``TEST_FOLD`` for test items).
        split_groups: Split group id per item.
        options: Options the plan was made with.
        balance: Downsampling outcome per subset.
        test_selection: Accepted test draw (attempt and pipe shares).
        test_sessions: ``<source>`` sessions of all test groups, sorted.
        source_stats: Counts over all label files read (before selection).
    """

    items: tuple[AIHubYoloItem, ...]
    file_names: tuple[str, ...]
    subsets: tuple[Subset, ...]
    folds: tuple[int, ...]
    split_groups: tuple[int, ...]
    options: SplitOptions
    balance: dict[Subset, BalanceResult]
    test_selection: TestSelection
    test_sessions: tuple[str, ...]
    source_stats: dict[str, int] = field(default_factory=dict)


# --- record -> item ------------------------------------------------------------


def to_yolo_box(
    box: Box, image_width_px: float, image_height_px: float
) -> YoloBox | None:
    """Convert a pixel box to normalised YOLO centre/size.

    The box is clipped to the image first. Swapped corners are reordered.

    Returns:
        The YOLO box, or None if a side is shorter than ``MIN_BOX_SIDE_PX`` after
        clipping (including NaN coordinates).
    """
    x_low, x_high = sorted((box.x1, box.x2))
    y_low, y_high = sorted((box.y1, box.y2))
    x_low, x_high = max(x_low, 0.0), min(x_high, image_width_px)
    y_low, y_high = max(y_low, 0.0), min(y_high, image_height_px)
    width_px, height_px = x_high - x_low, y_high - y_low
    # Written as "not >=" so that NaN sides are rejected as well.
    if not (width_px >= MIN_BOX_SIDE_PX and height_px >= MIN_BOX_SIDE_PX):
        return None
    return YoloBox(
        class_id=box.class_id,
        x_center=(x_low + x_high) / 2 / image_width_px,
        y_center=(y_low + y_high) / 2 / image_height_px,
        width=width_px / image_width_px,
        height=height_px / image_height_px,
    )


def _image_size_px(record: AIHubLabelRecord) -> tuple[int, int]:
    try:
        width_px = int(record.xml_fields["width"])
        height_px = int(record.xml_fields["height"])
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"{record.member_stem}: missing or invalid <size> in XML"
        ) from error
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"{record.member_stem}: <size> {width_px}x{height_px}")
    return width_px, height_px


def item_from_record(record: AIHubLabelRecord, image_member: str) -> AIHubYoloItem:
    """Build an item from a parsed label and the name of its paired image member.

    Raises:
        ValueError: If the XML has no usable ``<size>``.
    """
    width_px, height_px = _image_size_px(record)
    converted = [to_yolo_box(box, width_px, height_px) for box in record.boxes]
    valid_boxes = tuple(box for box in converted if box is not None)
    return AIHubYoloItem(
        member_stem=record.member_stem,
        image_member=image_member,
        split=record.split,
        category=record.category,
        session=record.session,
        session_from_name=record.session_from_name,
        crop_key=record.crop_key,
        gain=record.gain,
        image_width_px=width_px,
        image_height_px=height_px,
        boxes=valid_boxes,
        dropped_box_count=len(converted) - len(valid_boxes),
        has_filename_mismatch=record.has_filename_mismatch,
    )


def _sort_key(item: AIHubYoloItem) -> tuple[str, str, str]:
    return (item.split.value, item.category, item.member_stem)


def collect_items(
    pairs: Iterable[AIHubZipPair], limit_per_zip: int | None = None
) -> list[AIHubYoloItem]:
    """Read the labels of every zip pair (images are not decoded).

    Returns:
        Items sorted by split, category and member stem, so later steps do not
        depend on zip storage order.
    """
    items = []
    for pair in pairs:
        with AIHubZipReader(pair) as reader:
            pair_items = [
                item_from_record(record, reader.image_member_name(record.member_stem))
                for record in reader.iter_records(limit_per_zip)
            ]
        logger.info(
            "%s %s: %d labels", pair.split.value, pair.category, len(pair_items)
        )
        items.extend(pair_items)
    return sorted(items, key=_sort_key)


# --- grouping ------------------------------------------------------------------


def split_group_ids(items: Sequence[AIHubYoloItem]) -> list[int]:
    """Group items that share a ``<source>`` session, name session or crop key."""
    keys_per_item = []
    for item in items:
        keys: list[tuple[str, str]] = [
            ("session_from_name", item.session_from_name),
            ("crop_key", item.crop_key),
        ]
        # An empty <source> would otherwise join every such file into one group.
        if item.session:
            keys.append(("session", item.session))
        keys_per_item.append(keys)
    return connected_group_ids(keys_per_item)


def _classes_by_group(
    items: Sequence[AIHubYoloItem], group_ids: Sequence[int]
) -> dict[int, Counter[DetectionClass]]:
    """Per group, the number of images that contain each class."""
    counts: dict[int, Counter[DetectionClass]] = defaultdict(Counter)
    for item, group in zip(items, group_ids, strict=True):
        for class_id in item.class_ids:
            counts[group][class_id] += 1
    return counts


def _draw_groups(
    candidates: Iterable[int], wanted: int, already_taken: set[int], rng: random.Random
) -> set[int]:
    """Shuffle ``candidates`` and take groups until ``wanted`` of them are taken,
    counting candidates that are already in ``already_taken``."""
    shuffled = sorted(candidates)
    rng.shuffle(shuffled)
    taken = {group for group in shuffled if group in already_taken}
    for group in shuffled:
        if len(taken) >= wanted:
            break
        taken.add(group)
    return taken


def rng_stream_for_test_draw(seed: int, attempt: int) -> str:
    """Name of the random stream of one test-selection attempt."""
    return f"{seed}:test:{attempt}"


def _pipe_test_group_counts(
    classes_by_group: Mapping[int, Counter[DetectionClass]], options: SplitOptions
) -> dict[DetectionClass, tuple[list[int], int]]:
    """Per pipe class: its groups and how many of them go to test.

    The count is ``max(min_test_pipe_sessions, round(test_fraction x n))``,
    capped so that at least one group stays in the train pool.
    """
    counts = {}
    for pipe_class in PIPE_CLASSES:
        candidates = [g for g, c in classes_by_group.items() if pipe_class in c]
        wanted = max(
            options.min_test_pipe_sessions,
            round(options.test_fraction * len(candidates)),
        )
        capped = max(min(wanted, len(candidates) - 1), 0)
        if capped < options.min_test_pipe_sessions:
            logger.warning(
                "Only %d %s groups: %d go to test (wanted %d)",
                len(candidates),
                pipe_class.label,
                capped,
                options.min_test_pipe_sessions,
            )
        counts[pipe_class] = (candidates, capped)
    return counts


def pipe_image_shares(
    classes_by_group: Mapping[int, Counter[DetectionClass]],
    test_groups: Iterable[int],
) -> dict[str, float | None]:
    """Share of each pipe class's images that lies in ``test_groups``.

    Pipe images are never downsampled, so the share before downsampling equals
    the share after it. None when the class has no images at all.
    """
    test_group_set = set(test_groups)
    shares: dict[str, float | None] = {}
    for pipe_class in PIPE_CLASSES:
        total = sum(counts[pipe_class] for counts in classes_by_group.values())
        in_test = sum(
            counts[pipe_class]
            for group, counts in classes_by_group.items()
            if group in test_group_set
        )
        shares[pipe_class.label] = in_test / total if total else None
    return shares


def _shares_within_bounds(
    shares: Mapping[str, float | None], options: SplitOptions
) -> bool:
    return all(
        options.pipe_share_min <= share <= options.pipe_share_max
        for share in shares.values()
        if share is not None
    )


def _draw_non_pipe_groups(
    classes_by_group: Mapping[int, Counter[DetectionClass]],
    test_fraction: float,
    rng: random.Random,
) -> set[int]:
    """Draw ``round(test_fraction x n)`` pipe-free groups per primary class."""
    non_pipe_by_class: dict[DetectionClass, list[int]] = defaultdict(list)
    for group, counts in classes_by_group.items():
        if not any(pipe_class in counts for pipe_class in PIPE_CLASSES):
            primary = min(counts, key=lambda c: (-counts[c], c))
            non_pipe_by_class[primary].append(group)
    drawn: set[int] = set()
    for detection_class in sorted(non_pipe_by_class):
        candidates = non_pipe_by_class[detection_class]
        wanted = min(round(test_fraction * len(candidates)), len(candidates) - 1)
        drawn |= _draw_groups(candidates, wanted, set(), rng)
    return drawn


def _validate_test_options(options: SplitOptions) -> None:
    if not 0 < options.test_fraction < 1:
        raise ValueError(
            f"test_fraction must be in (0, 1), got {options.test_fraction}"
        )
    if not 0 <= options.pipe_share_min <= options.pipe_share_max <= 1:
        raise ValueError(
            "need 0 <= pipe_share_min <= pipe_share_max <= 1, got "
            f"{options.pipe_share_min} and {options.pipe_share_max}"
        )
    if options.max_test_attempts < 1:
        raise ValueError(
            f"max_test_attempts must be >= 1, got {options.max_test_attempts}"
        )


def select_test_groups(
    items: Sequence[AIHubYoloItem],
    group_ids: Sequence[int],
    options: SplitOptions,
) -> TestSelection:
    """Choose the split groups held out as the test subset.

    Each attempt uses the stream ``rng_stream_for_test_draw(seed, attempt)``. Pipe
    classes go first, one class at a time: the groups containing tpipe, then
    those containing lpipe, each a shuffled list from which groups are taken
    until ``max(min_test_pipe_sessions, round(test_fraction x n))`` of that
    class's groups are in test (groups holding both pipe classes count for
    both). The attempt is accepted when every pipe class's test image share is
    within ``[pipe_share_min, pipe_share_max]``; otherwise the next attempt is
    tried. Pipe-free groups are then drawn per primary class (cavity, manhole)
    at ``round(test_fraction x n)`` from the accepted attempt's stream. At
    least one group of each class always stays in the train pool.

    Args:
        items: Items with at least one box each.
        group_ids: Split group id per item.
        options: Seed, test fraction, pipe minimum, share bounds, attempt limit.

    Raises:
        ValueError: If an option is out of range.
        TestSelectionError: If no attempt meets the share bounds.
    """
    _validate_test_options(options)
    classes_by_group = _classes_by_group(items, group_ids)
    pipe_counts = _pipe_test_group_counts(classes_by_group, options)

    for attempt in range(options.max_test_attempts):
        rng = random.Random(rng_stream_for_test_draw(options.seed, attempt))
        test_groups: set[int] = set()
        for candidates, wanted in pipe_counts.values():
            test_groups |= _draw_groups(candidates, wanted, test_groups, rng)
        shares = pipe_image_shares(classes_by_group, test_groups)
        if not _shares_within_bounds(shares, options):
            continue
        test_groups |= _draw_non_pipe_groups(
            classes_by_group, options.test_fraction, rng
        )
        logger.info("Test selection: attempt %d, pipe shares %s", attempt, shares)
        return TestSelection(frozenset(test_groups), attempt, shares)

    raise TestSelectionError(
        f"No test selection in {options.max_test_attempts} attempts (seed "
        f"{options.seed}) puts each pipe class's image share within "
        f"[{options.pipe_share_min}, {options.pipe_share_max}]"
    )


# --- downsampling --------------------------------------------------------------


def _round_robin_crop_order(
    crops_by_group: dict[int, list[str]], rng: random.Random
) -> list[str]:
    """Shuffle groups and the crops inside each group, then deal one crop per
    group per round, so small groups are not crowded out by large ones."""
    group_order = sorted(crops_by_group)
    rng.shuffle(group_order)
    shuffled = {}
    for group in group_order:
        crops = sorted(crops_by_group[group])
        rng.shuffle(crops)
        shuffled[group] = crops

    order = []
    deepest_group = max((len(crops) for crops in shuffled.values()), default=0)
    for round_index in range(deepest_group):
        for group in group_order:
            if round_index < len(shuffled[group]):
                order.append(shuffled[group][round_index])
    return order


def _crop_balance_class(crop_items: Sequence[AIHubYoloItem]) -> str:
    """``"pipe"`` if any copy has a pipe box, else the first copy's primary class."""
    if any(item.has_pipe for item in crop_items):
        return PIPE_BALANCE_CLASS
    return crop_items[0].primary_class.label


def _take_until(
    order: Sequence[str], image_counts: dict[str, list[int]], target: int
) -> list[str]:
    kept: list[str] = []
    kept_image_count = 0
    for crop_key in order:
        if kept_image_count >= target:
            break
        kept.append(crop_key)
        kept_image_count += len(image_counts[crop_key])
    return kept


def select_balanced_items(
    items: Sequence[AIHubYoloItem],
    group_ids: Sequence[int],
    balance_ratio: float,
    seed: int,
) -> BalanceResult:
    """Keep all pipe crops and draw cavity/manhole crops down to the pipe count.

    Each downsampled class gets a target of ``round(balance_ratio x pipe
    images)`` images. Crops (all gain copies together) are taken in round-robin
    order over split groups until the target is reached, so the result may
    exceed the target by the copies of the last crop.

    Args:
        items: Items with at least one box each.
        group_ids: Split group id per item.
        balance_ratio: Target as a multiple of the pipe image count (> 0).
        seed: Seed of the draw; each class uses its own derived stream.

    Raises:
        ValueError: If ``balance_ratio`` is not positive.
    """
    if balance_ratio <= 0:
        raise ValueError(f"balance_ratio must be > 0, got {balance_ratio}")

    indices_by_crop: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        indices_by_crop[item.crop_key].append(index)

    crops_by_class: dict[str, dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for crop_key, indices in indices_by_crop.items():
        balance_class = _crop_balance_class([items[i] for i in indices])
        crops_by_class[balance_class][group_ids[indices[0]]].append(crop_key)

    def image_count(crop_keys: Iterable[str]) -> int:
        return sum(len(indices_by_crop[crop_key]) for crop_key in crop_keys)

    def all_crops(balance_class: str) -> list[str]:
        groups = crops_by_class.get(balance_class, {})
        return [crop for crops in groups.values() for crop in crops]

    target = round(balance_ratio * image_count(all_crops(PIPE_BALANCE_CLASS)))
    downsampled_labels = {member.label for member in DOWNSAMPLED_CLASSES}

    kept_crops_by_class: dict[str, list[str]] = {}
    for balance_class in sorted(crops_by_class):
        if balance_class not in downsampled_labels:
            kept_crops_by_class[balance_class] = all_crops(balance_class)
            continue
        rng = random.Random(f"{seed}:{balance_class}")
        order = _round_robin_crop_order(crops_by_class[balance_class], rng)
        kept_crops_by_class[balance_class] = _take_until(order, indices_by_crop, target)

    kept_indices = sorted(
        index
        for crops in kept_crops_by_class.values()
        for crop_key in crops
        for index in indices_by_crop[crop_key]
    )
    return BalanceResult(
        kept_indices=tuple(kept_indices),
        target_images_per_class=target,
        available_images={c: image_count(all_crops(c)) for c in crops_by_class},
        kept_images={c: image_count(kept) for c, kept in kept_crops_by_class.items()},
        available_crops={c: len(all_crops(c)) for c in crops_by_class},
        kept_crops={c: len(kept) for c, kept in kept_crops_by_class.items()},
    )


# --- naming and planning -------------------------------------------------------


def output_file_names(items: Sequence[AIHubYoloItem]) -> list[str]:
    """Return a unique ``.jpg`` file name per item, based on the member stem.

    The stem already starts with the equipment, date and time of the session.
    Characters other than ``A-Z a-z 0-9 ( ) . _ -`` become ``_``; a stem seen
    again (e.g. in another zip) gets ``__dup<n>`` in input order.
    """
    seen: Counter[str] = Counter()
    names = []
    for item in items:
        base = _UNSAFE_FILE_NAME_CHARS.sub("_", item.member_stem)
        duplicate_number = seen[base]
        seen[base] += 1
        suffix = f"__dup{duplicate_number}" if duplicate_number else ""
        names.append(f"{base}{suffix}{IMAGE_SUFFIX}")
    return names


def _source_stats(
    all_items: Sequence[AIHubYoloItem], group_ids: Sequence[int]
) -> dict[str, int]:
    sessions_per_group: dict[int, set[str]] = defaultdict(set)
    for item, group in zip(all_items, group_ids, strict=True):
        sessions_per_group[group].add(item.session)
    return {
        "label_files": len(all_items),
        "images_without_valid_boxes": sum(not item.boxes for item in all_items),
        "dropped_boxes": sum(item.dropped_box_count for item in all_items),
        "session_mismatch_files": sum(item.has_session_mismatch for item in all_items),
        "filename_mismatch_files": sum(
            item.has_filename_mismatch for item in all_items
        ),
        "missing_source_files": sum(not item.session for item in all_items),
        "mixed_class_images": sum(len(item.class_ids) > 1 for item in all_items),
        "source_sessions": len({item.session for item in all_items}),
        "name_sessions": len({item.session_from_name for item in all_items}),
        "crop_keys": len({item.crop_key for item in all_items}),
        "split_groups": len(sessions_per_group),
        "split_groups_with_several_sessions": sum(
            len(sessions) > 1 for sessions in sessions_per_group.values()
        ),
    }


@dataclass(frozen=True)
class _PlannedItem:
    item: AIHubYoloItem
    subset: Subset
    fold: int
    group: int


def _plan_subset(
    items: Sequence[AIHubYoloItem],
    group_ids: Sequence[int],
    subset: Subset,
    options: SplitOptions,
) -> tuple[list[_PlannedItem], BalanceResult]:
    """Downsample one subset; fold the train pool, mark test items TEST_FOLD."""
    balance = select_balanced_items(
        items, group_ids, options.balance_ratio, options.seed
    )
    kept_items = [items[i] for i in balance.kept_indices]
    kept_groups = [group_ids[i] for i in balance.kept_indices]
    if subset is Subset.TEST:
        folds = [TEST_FOLD] * len(kept_items)
    else:
        # Stratify on the primary class (4 levels) rather than "has pipe"
        # (2 levels): tpipe and lpipe have only 11 and 13 sessions, so each needs
        # its own stratum to be spread over the folds. On the full AIHub set
        # (2026-10-29, 21 seeds, no test hold-out) "has pipe" left one fold
        # without any tpipe for every seed; the primary class put both pipe
        # classes in every fold.
        strata = [int(item.primary_class) for item in kept_items]
        folds = assign_group_folds(strata, kept_groups, options.k, options.seed)
    planned = [
        _PlannedItem(item, subset, fold, group)
        for item, fold, group in zip(kept_items, folds, kept_groups, strict=True)
    ]
    return planned, balance


def plan_dataset(
    all_items: Sequence[AIHubYoloItem], options: SplitOptions
) -> DatasetPlan:
    """Drop box-less images, hold out test, downsample, fold and check leakage.

    Args:
        all_items: Every label read, sorted as by :func:`collect_items`.
        options: Seed, k, balance ratio and test hold-out settings.

    Raises:
        TestSelectionError: If no test draw meets the pipe image share bounds.
        SplitLeakageError: If a session, name session or crop key spans test and
            the train pool or two folds.
    """
    labelled_items = [item for item in all_items if item.boxes]
    group_ids = split_group_ids(labelled_items)
    test_selection = select_test_groups(labelled_items, group_ids, options)
    test_groups = test_selection.groups

    planned: list[_PlannedItem] = []
    balance: dict[Subset, BalanceResult] = {}
    for subset in Subset:
        is_test = subset is Subset.TEST
        indices = [
            i for i, group in enumerate(group_ids) if (group in test_groups) == is_test
        ]
        subset_planned, balance[subset] = _plan_subset(
            [labelled_items[i] for i in indices],
            [group_ids[i] for i in indices],
            subset,
            options,
        )
        planned.extend(subset_planned)
    planned.sort(key=lambda entry: _sort_key(entry.item))

    items = tuple(entry.item for entry in planned)
    test_sessions = {
        item.session
        for item, group in zip(labelled_items, group_ids, strict=True)
        if group in test_groups
    }
    plan = DatasetPlan(
        items=items,
        file_names=tuple(output_file_names(items)),
        subsets=tuple(entry.subset for entry in planned),
        folds=tuple(entry.fold for entry in planned),
        split_groups=tuple(entry.group for entry in planned),
        options=options,
        balance=balance,
        test_selection=test_selection,
        test_sessions=tuple(sorted(test_sessions)),
        source_stats=_source_stats(all_items, split_group_ids(all_items)),
    )
    # The test subset carries its own fold value, so one check covers test vs
    # train pool and fold vs fold.
    assert_no_fold_leakage(manifest_rows(plan), AIHUB_LEAK_KEY_COLUMNS)
    return plan


def manifest_rows(plan: DatasetPlan) -> list[dict[str, Any]]:
    """One manifest row per selected image, columns in ``MANIFEST_COLUMNS`` order."""
    rows = []
    for index, item in enumerate(plan.items):
        rows.append(
            {
                "file_name": plan.file_names[index],
                "member": item.image_member,
                "split_src": item.split.value,
                "session": item.session,
                "session_from_name": item.session_from_name,
                "crop_key": item.crop_key,
                "gain": item.gain,
                "classes": ";".join(str(int(c)) for c in item.class_ids),
                "n_boxes": len(item.boxes),
                "subset": plan.subsets[index].value,
                "fold": plan.folds[index],
                "seed": plan.options.seed,
                "test_attempt": plan.test_selection.attempt,
                "category": item.category,
                "primary_class": item.primary_class.label,
                "split_group": plan.split_groups[index],
            }
        )
    return rows
