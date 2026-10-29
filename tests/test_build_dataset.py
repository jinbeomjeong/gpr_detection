"""Tests for the AIHub -> YOLO conversion and the session-grouped fold split (M1-3).

Unit tests use synthetic items. The two integration tests read the first few
labels of each real zip (``build_dataset`` in-process, and the CLI script with
``--dry-run``) and are skipped when the AIHub root is missing.
"""

import csv
import json
import math
import subprocess
import sys
import zipfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from gpr_detection.data.aihub_yolo import (
    MANIFEST_COLUMNS,
    TEST_FOLD,
    AIHubYoloItem,
    SplitOptions,
    Subset,
    TestSelectionError,
    YoloBox,
    item_from_record,
    manifest_rows,
    output_file_names,
    plan_dataset,
    select_balanced_items,
    select_test_groups,
    split_group_ids,
    to_yolo_box,
)
from gpr_detection.data.aihub_yolo_writer import (
    TEST_LIST_NAME,
    TEST_YAML_NAME,
    BuildConfig,
    build_dataset,
    fold_list_name,
    write_fold_files,
)
from gpr_detection.data.splits import (
    SplitLeakageError,
    assert_no_fold_leakage,
    assign_group_folds,
    connected_group_ids,
)
from gpr_detection.io import aihub
from gpr_detection.io.aihub import AIHubLabelRecord, AIHubSplit
from gpr_detection.io.annotated import Box
from gpr_detection.io.classes import DetectionClass
from gpr_detection.io.paths import load_dataset_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "build_yolo_dataset.py"
IMAGE_SIDE_PX = 256
GAINS = (0.25, 0.5, 1.0)
LABELS_PER_ZIP = 20
SCRIPT_TIMEOUT_S = 300


# --- synthetic items -------------------------------------------------------------


def make_stem(session_index: int, crop_index: int, gain: float) -> str:
    """Member stem whose name session is distinct per ``session_index``."""
    hour, minute = divmod(session_index, 60)
    x_start = crop_index * IMAGE_SIDE_PX
    return (
        f"NM_20221017({hour:02d}{minute:02d}00)_N00_B1_g{gain}"
        f"_x({x_start}-{x_start + IMAGE_SIDE_PX})"
    )


def make_item(
    session_index: int,
    crop_index: int,
    gain: float,
    class_ids: tuple[DetectionClass, ...] = (DetectionClass.CAVITY,),
    source: str | None = None,
) -> AIHubYoloItem:
    stem = make_stem(session_index, crop_index, gain)
    session_from_name = aihub.session_from_member_name(stem)
    boxes = tuple(YoloBox(class_id, 0.5, 0.5, 0.1, 0.1) for class_id in class_ids)
    return AIHubYoloItem(
        member_stem=stem,
        image_member=f"/{stem}.jpg",
        split=AIHubSplit.TRAINING,
        category="test",
        session=session_from_name if source is None else source,
        session_from_name=session_from_name,
        crop_key=aihub.crop_key_from_member_name(stem),
        gain=gain,
        image_width_px=IMAGE_SIDE_PX,
        image_height_px=IMAGE_SIDE_PX,
        boxes=boxes,
    )


def make_crop(
    session_index: int,
    crop_index: int,
    class_ids: tuple[DetectionClass, ...] = (DetectionClass.CAVITY,),
) -> list[AIHubYoloItem]:
    """The three gain copies of one crop."""
    return [make_item(session_index, crop_index, gain, class_ids) for gain in GAINS]


def make_balanced_fixture() -> list[AIHubYoloItem]:
    """2 pipe sessions x 1 crop (6 images); 6 cavity, 4 manhole sessions x 3 crops."""
    items: list[AIHubYoloItem] = []
    items += make_crop(0, 0, (DetectionClass.PIPE_HYPERBOLA,))
    items += make_crop(1, 0, (DetectionClass.PIPE_BAND, DetectionClass.PIPE_BAND))
    for session_index in range(10, 16):
        for crop_index in range(3):
            items += make_crop(session_index, crop_index)
    for session_index in range(20, 24):
        for crop_index in range(3):
            items += make_crop(session_index, crop_index, (DetectionClass.MANHOLE,))
    return items


def pixel_box(x1: float, y1: float, x2: float, y2: float) -> Box:
    return Box(DetectionClass.CAVITY, x1, y1, x2, y2, source_label="cavity")


# --- YOLO box conversion -------------------------------------------------------


def test_to_yolo_box_normalises_centre_and_size() -> None:
    yolo = to_yolo_box(pixel_box(64, 32, 128, 96), IMAGE_SIDE_PX, IMAGE_SIDE_PX)
    assert yolo == YoloBox(DetectionClass.CAVITY, 0.375, 0.25, 0.25, 0.25)
    assert yolo.to_line() == "2 0.375000 0.250000 0.250000 0.250000"


def test_to_yolo_box_full_image_is_exactly_one() -> None:
    yolo = to_yolo_box(pixel_box(0, 0, 256, 256), IMAGE_SIDE_PX, IMAGE_SIDE_PX)
    assert yolo is not None
    assert (yolo.x_center, yolo.y_center, yolo.width, yolo.height) == (
        0.5,
        0.5,
        1.0,
        1.0,
    )


def test_to_yolo_box_clips_to_image_and_stays_in_unit_range() -> None:
    yolo = to_yolo_box(pixel_box(-20, 200, 300, 270), IMAGE_SIDE_PX, IMAGE_SIDE_PX)
    assert yolo is not None
    assert yolo.width == pytest.approx(1.0)
    assert yolo.height == pytest.approx(56 / 256)
    assert yolo.y_center + yolo.height / 2 == pytest.approx(1.0)
    for value in (yolo.x_center, yolo.y_center, yolo.width, yolo.height):
        assert 0.0 <= value <= 1.0


def test_to_yolo_box_reorders_swapped_corners() -> None:
    swapped = to_yolo_box(pixel_box(128, 96, 64, 32), IMAGE_SIDE_PX, IMAGE_SIDE_PX)
    ordered = to_yolo_box(pixel_box(64, 32, 128, 96), IMAGE_SIDE_PX, IMAGE_SIDE_PX)
    assert swapped == ordered


@pytest.mark.parametrize(
    "corners",
    [
        (10, 10, 10, 50),  # zero width
        (10, 10, 10.5, 50),  # thinner than one pixel
        (300, 10, 400, 50),  # entirely right of the image
        (math.nan, 10, 20, 50),  # unparsable coordinate
    ],
)
def test_to_yolo_box_rejects_degenerate_boxes(
    corners: tuple[float, float, float, float],
) -> None:
    assert to_yolo_box(pixel_box(*corners), IMAGE_SIDE_PX, IMAGE_SIDE_PX) is None


def make_record(boxes: tuple[Box, ...], xml_fields: dict[str, str]) -> AIHubLabelRecord:
    stem = make_stem(0, 0, 0.5)
    return AIHubLabelRecord(
        member_stem=stem,
        split=AIHubSplit.VALIDATION,
        category="test",
        boxes=boxes,
        session="NM_2022-10-17_(00_00_00)",
        session_from_name=aihub.session_from_member_name(stem),
        crop_key=aihub.crop_key_from_member_name(stem),
        gain=0.5,
        filename_field="something_else",
        xml_fields=xml_fields,
    )


def test_item_from_record_counts_dropped_boxes() -> None:
    record = make_record(
        (pixel_box(0, 0, 128, 128), pixel_box(5, 5, 5, 9)),
        {"width": "256", "height": "256"},
    )
    item = item_from_record(record, "/image.jpg")
    assert len(item.boxes) == 1
    assert item.dropped_box_count == 1
    assert item.has_filename_mismatch
    assert item.image_member == "/image.jpg"


@pytest.mark.parametrize(
    "xml_fields", [{}, {"width": "", "height": "256"}, {"width": "0", "height": "256"}]
)
def test_item_from_record_requires_xml_size(xml_fields: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="size"):
        item_from_record(make_record((), xml_fields), "/image.jpg")


def test_primary_class_prefers_majority_then_lower_id() -> None:
    majority = make_item(
        0, 0, 0.5, (DetectionClass.MANHOLE,) * 2 + (DetectionClass.PIPE_BAND,)
    )
    tie = make_item(0, 0, 0.5, (DetectionClass.MANHOLE, DetectionClass.PIPE_BAND))
    assert majority.primary_class is DetectionClass.MANHOLE
    assert tie.primary_class is DetectionClass.PIPE_BAND
    assert tie.has_pipe


# --- grouping --------------------------------------------------------------------


def test_connected_group_ids_follow_key_chains() -> None:
    keys = [["a", "x"], ["b"], ["x", "c"], ["c", "d"], ["b", "e"], ["f"]]
    assert connected_group_ids(keys) == [0, 1, 0, 0, 1, 2]


def test_connected_group_ids_rejects_item_without_keys() -> None:
    with pytest.raises(ValueError, match="no grouping keys"):
        connected_group_ids([["a"], []])


def test_split_groups_merge_sessions_linked_by_mismatched_source() -> None:
    first = make_item(0, 0, 0.5)
    second = make_item(1, 0, 0.5)
    # <source> names session 1 although the member name says session 0.
    mismatched = make_item(0, 1, 0.5, source=second.session)
    unrelated = make_item(2, 0, 0.5)
    groups = split_group_ids([first, second, mismatched, unrelated])
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_split_groups_ignore_empty_source() -> None:
    items = [make_item(0, 0, 0.5, source=""), make_item(1, 0, 0.5, source="")]
    groups = split_group_ids(items)
    assert groups[0] != groups[1]


# --- downsampling ----------------------------------------------------------------


def test_balance_keeps_all_pipe_and_draws_others_near_pipe_count() -> None:
    items = make_balanced_fixture()
    result = select_balanced_items(items, split_group_ids(items), 1.0, seed=7)

    kept = [items[i] for i in result.kept_indices]
    pipe_images = sum(item.has_pipe for item in items)
    assert result.target_images_per_class == pipe_images == 6
    assert sum(item.has_pipe for item in kept) == pipe_images
    for detection_class in (DetectionClass.CAVITY, DetectionClass.MANHOLE):
        kept_count = sum(item.primary_class is detection_class for item in kept)
        # Whole crops of three copies are added until the target is reached.
        assert pipe_images <= kept_count < pipe_images + len(GAINS)
        assert result.kept_images[detection_class.label] == kept_count


def test_balance_takes_all_gain_copies_of_a_crop_together() -> None:
    items = make_balanced_fixture()
    result = select_balanced_items(items, split_group_ids(items), 1.0, seed=3)
    copies_by_crop: dict[str, int] = defaultdict(int)
    for index in result.kept_indices:
        copies_by_crop[items[index].crop_key] += 1
    assert set(copies_by_crop.values()) == {len(GAINS)}


def test_balance_spreads_over_sessions_before_repeating_one() -> None:
    items = make_balanced_fixture()
    # 6 cavity sessions x 3 crops; a target of 4 crops must hit 4 sessions.
    result = select_balanced_items(items, split_group_ids(items), 2.0, seed=11)
    kept_cavity = [
        items[i]
        for i in result.kept_indices
        if items[i].primary_class is DetectionClass.CAVITY
    ]
    crops_per_session: dict[str, set[str]] = defaultdict(set)
    for item in kept_cavity:
        crops_per_session[item.session].add(item.crop_key)
    assert len(crops_per_session) == 4
    assert all(len(crops) == 1 for crops in crops_per_session.values())


def test_balance_is_reproducible_and_seed_dependent() -> None:
    items = make_balanced_fixture()
    groups = split_group_ids(items)
    first = select_balanced_items(items, groups, 1.0, seed=5)
    again = select_balanced_items(items, groups, 1.0, seed=5)
    assert first == again
    draws = {
        select_balanced_items(items, groups, 1.0, seed).kept_indices
        for seed in range(10)
    }
    assert len(draws) > 1


def test_balance_ratio_must_be_positive() -> None:
    items = make_balanced_fixture()
    with pytest.raises(ValueError, match="balance_ratio"):
        select_balanced_items(items, split_group_ids(items), 0.0, seed=1)


def test_balance_takes_everything_when_target_exceeds_supply() -> None:
    items = make_balanced_fixture()
    result = select_balanced_items(items, split_group_ids(items), 100.0, seed=1)
    assert len(result.kept_indices) == len(items)


# --- folds and leakage -----------------------------------------------------------


def test_assign_group_folds_keeps_each_group_in_one_fold() -> None:
    groups = [index // 3 for index in range(60)]
    strata = [group % 2 for group in groups]
    folds = assign_group_folds(strata, groups, k=5, seed=0)

    folds_by_group: dict[int, set[int]] = defaultdict(set)
    for group, fold in zip(groups, folds, strict=True):
        folds_by_group[group].add(fold)
    assert all(len(group_folds) == 1 for group_folds in folds_by_group.values())
    assert set(folds) == set(range(5))
    assert folds == assign_group_folds(strata, groups, k=5, seed=0)


def test_assign_group_folds_needs_enough_groups() -> None:
    with pytest.raises(ValueError, match="at least k=5 groups"):
        assign_group_folds([0, 1, 0, 1], [0, 0, 1, 2], k=5, seed=0)
    with pytest.raises(ValueError, match="strata"):
        assign_group_folds([0, 1], [0], k=2, seed=0)


def test_leakage_assertion_passes_on_disjoint_folds() -> None:
    rows = [
        {"fold": 0, "session": "s1", "session_from_name": "s1", "crop_key": "c1"},
        {"fold": 0, "session": "s1", "session_from_name": "s1", "crop_key": "c1"},
        {"fold": 1, "session": "s2", "session_from_name": "s2", "crop_key": "c2"},
    ]
    assert_no_fold_leakage(rows)


@pytest.mark.parametrize("leaking_column", ["session", "session_from_name", "crop_key"])
def test_leakage_assertion_fails_on_shared_key(leaking_column: str) -> None:
    rows = [
        {"fold": 0, "session": "s1", "session_from_name": "s1", "crop_key": "c1"},
        {"fold": 1, "session": "s2", "session_from_name": "s2", "crop_key": "c2"},
    ]
    rows[1][leaking_column] = rows[0][leaking_column]
    with pytest.raises(SplitLeakageError, match=leaking_column):
        assert_no_fold_leakage(rows)


def make_plan_fixture() -> list[AIHubYoloItem]:
    """Enough sessions of every class for five folds, plus one box-less image.

    12 equal tpipe and 12 equal lpipe sessions, so two test sessions per pipe
    class hold 2/12 = 17 % of its images (inside the default 10-25 % bounds).
    """
    items: list[AIHubYoloItem] = []
    pipe_classes = (DetectionClass.PIPE_HYPERBOLA, DetectionClass.PIPE_BAND)
    for session_index in range(100, 124):
        items += make_crop(session_index, 0, (pipe_classes[session_index % 2],))
    for session_index in range(10, 30):
        items += make_crop(session_index, 0)
        items += make_crop(session_index + 30, 0, (DetectionClass.MANHOLE,))
    items.append(make_item(99, 0, 0.5, class_ids=()))
    return items


def test_plan_dataset_drops_boxless_images_and_has_no_leaks() -> None:
    plan = plan_dataset(make_plan_fixture(), SplitOptions(seed=1))

    assert all(item.boxes for item in plan.items)
    assert plan.source_stats["images_without_valid_boxes"] == 1
    assert set(plan.folds) == {TEST_FOLD, *range(5)}
    rows = manifest_rows(plan)
    assert list(rows[0]) == list(MANIFEST_COLUMNS)
    assert {row["seed"] for row in rows} == {1}
    assert_no_fold_leakage(rows)
    pipe_folds = {
        fold for item, fold in zip(plan.items, plan.folds, strict=True) if item.has_pipe
    }
    assert pipe_folds == {TEST_FOLD, *range(5)}


def test_plan_dataset_test_subset_matches_fold_and_sessions() -> None:
    plan = plan_dataset(make_plan_fixture(), SplitOptions(seed=2))

    test_indices = [i for i, s in enumerate(plan.subsets) if s is Subset.TEST]
    assert test_indices
    assert all(plan.folds[i] == TEST_FOLD for i in test_indices)
    assert {plan.items[i].session for i in test_indices} == set(plan.test_sessions)
    for pipe_class in (DetectionClass.PIPE_HYPERBOLA, DetectionClass.PIPE_BAND):
        test_sessions = {
            plan.items[i].session
            for i in test_indices
            if pipe_class in plan.items[i].class_ids
        }
        assert len(test_sessions) >= 2
    # Pipe images of the test subset are never downsampled.
    test_balance = plan.balance[Subset.TEST]
    assert test_balance.kept_images["pipe"] == test_balance.available_images["pipe"]


def groups_with_class(
    items: list[AIHubYoloItem], groups: list[int], detection_class: DetectionClass
) -> set[int]:
    return {
        group
        for item, group in zip(items, groups, strict=True)
        if detection_class in item.class_ids
    }


def test_select_test_groups_meets_pipe_minimum_and_fraction() -> None:
    items = [item for item in make_plan_fixture() if item.boxes]
    groups = split_group_ids(items)
    options = SplitOptions(seed=3)
    selection = select_test_groups(items, groups, options)

    tpipe_groups = groups_with_class(items, groups, DetectionClass.PIPE_HYPERBOLA)
    assert len(selection.groups & tpipe_groups) == 2  # max(2, round(0.15 x 12))
    cavity_groups = {
        g
        for item, g in zip(items, groups, strict=True)
        if item.primary_class is DetectionClass.CAVITY
    }
    assert len(selection.groups & cavity_groups) == 3  # round(0.15 x 20)
    assert selection.pipe_image_shares == {
        "pipe_hyperbola": pytest.approx(2 / 12),
        "pipe_band": pytest.approx(2 / 12),
    }
    assert selection == select_test_groups(items, groups, options)


def make_uneven_pipe_items() -> list[AIHubYoloItem]:
    """Per pipe class: 10 one-crop sessions (3 images) and 2 ten-crop sessions
    (30 images), 90 images in total. Two test sessions give a share of 6.7 %,
    36.7 % or 66.7 %, depending on how many large sessions are drawn."""
    items: list[AIHubYoloItem] = []
    for offset, pipe_class in (
        (200, DetectionClass.PIPE_HYPERBOLA),
        (300, DetectionClass.PIPE_BAND),
    ):
        for session_index in range(offset, offset + 10):
            items += make_crop(session_index, 0, (pipe_class,))
        for session_index in range(offset + 10, offset + 12):
            for crop_index in range(10):
                items += make_crop(session_index, crop_index, (pipe_class,))
    return items


def test_select_test_groups_searches_attempts_until_shares_fit() -> None:
    items = make_uneven_pipe_items()
    groups = split_group_ids(items)
    # Only "one large + one small session" (33 of 90 images) fits for each class.
    options = SplitOptions(seed=4, pipe_share_min=0.30, pipe_share_max=0.40)
    selection = select_test_groups(items, groups, options)

    for share in selection.pipe_image_shares.values():
        assert share == pytest.approx(33 / 90)
    assert selection == select_test_groups(items, groups, options)
    if selection.attempt > 0:
        # Every earlier attempt failed, so the accepted one is the first fit.
        earlier = replace(options, max_test_attempts=selection.attempt)
        with pytest.raises(TestSelectionError):
            select_test_groups(items, groups, earlier)


def test_select_test_groups_fails_clearly_when_bounds_unreachable() -> None:
    items = [item for item in make_plan_fixture() if item.boxes]
    # Two of twelve equal sessions always give 17 %, never 50-60 %.
    options = SplitOptions(
        seed=0, pipe_share_min=0.50, pipe_share_max=0.60, max_test_attempts=5
    )
    with pytest.raises(TestSelectionError, match="5 attempts"):
        select_test_groups(items, split_group_ids(items), options)


def test_select_test_groups_keeps_one_group_for_training() -> None:
    items = make_crop(0, 0, (DetectionClass.PIPE_HYPERBOLA,)) + make_crop(1, 0)
    options = SplitOptions(seed=0, test_fraction=0.5, pipe_share_min=0.0)
    selection = select_test_groups(items, split_group_ids(items), options)
    assert selection.groups == frozenset()
    assert selection.pipe_image_shares["pipe_band"] is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"test_fraction": 0.0}, "test_fraction"),
        ({"test_fraction": 1.0}, "test_fraction"),
        ({"pipe_share_min": 0.3, "pipe_share_max": 0.2}, "pipe_share_min"),
        ({"max_test_attempts": 0}, "max_test_attempts"),
    ],
)
def test_select_test_groups_rejects_bad_options(
    changes: dict[str, float], message: str
) -> None:
    items = make_plan_fixture()[:3]
    options = replace(SplitOptions(seed=0), **changes)
    with pytest.raises(ValueError, match=message):
        select_test_groups(items, split_group_ids(items), options)


def test_output_file_names_are_unique_and_safe() -> None:
    item = make_item(0, 0, 0.5)
    odd = replace(item, member_stem="a b/c")
    names = output_file_names([item, item, odd])
    assert names[0] == f"{item.member_stem}.jpg"
    assert names[1] == f"{item.member_stem}__dup1.jpg"
    assert names[2] == "a_b_c.jpg"


def test_write_fold_files_point_train_to_the_other_folds(tmp_path: Path) -> None:
    plan = plan_dataset(make_plan_fixture(), SplitOptions(seed=1))
    write_fold_files(plan, tmp_path)

    listed: set[str] = set()
    for fold in range(5):
        lines = (tmp_path / fold_list_name(fold)).read_text(encoding="utf-8").split()
        assert all(line.startswith("./images/") for line in lines)
        listed.update(lines)
        config = yaml.safe_load(
            (tmp_path / f"fold{fold}.yaml").read_text(encoding="utf-8")
        )
        assert config["val"] == fold_list_name(fold)
        assert fold_list_name(fold) not in config["train"]
        assert len(config["train"]) == 4
        assert config["names"] == {
            0: "pipe_hyperbola",
            1: "pipe_band",
            2: "cavity",
            3: "manhole",
        }
    test_lines = (tmp_path / TEST_LIST_NAME).read_text(encoding="utf-8").split()
    assert test_lines
    assert not listed & set(test_lines)
    assert len(listed) + len(test_lines) == len(plan.items)

    test_config = yaml.safe_load(
        (tmp_path / TEST_YAML_NAME).read_text(encoding="utf-8")
    )
    assert test_config["val"] == test_config["test"] == TEST_LIST_NAME
    assert test_config["train"] == [fold_list_name(fold) for fold in range(5)]


# --- real zips -------------------------------------------------------------------


@pytest.fixture(scope="module")
def aihub_root() -> Path:
    root = load_dataset_paths().aihub
    if not root.is_dir():
        pytest.skip(f"AIHub data not found at {root}")
    return root


def run_build_script(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(BUILD_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=SCRIPT_TIMEOUT_S,
        check=False,
    )


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_label_file_valid(label_path: Path) -> int:
    lines = label_path.read_text(encoding="utf-8").splitlines()
    assert lines
    for line in lines:
        class_id, *coordinates = line.split()
        assert int(class_id) in {member.value for member in DetectionClass}
        assert all(0.0 <= float(value) <= 1.0 for value in coordinates)
    return len(lines)


@pytest.mark.data
def test_build_dataset_writes_consistent_dataset(
    aihub_root: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "aihub_yolo_test"
    manifest_path = tmp_path / "splits" / "aihub_yolo_test.csv"
    config = BuildConfig(
        aihub_root=aihub_root,
        out_dir=out_dir,
        manifest_path=manifest_path,
        # --limit leaves too few pipe sessions for the 10-25 % share bounds.
        options=SplitOptions(seed=1, pipe_share_min=0.0, pipe_share_max=1.0),
        limit_per_zip=LABELS_PER_ZIP,
    )
    build_dataset(config)

    rows = read_manifest(manifest_path)
    assert list(rows[0]) == list(MANIFEST_COLUMNS)
    assert_no_fold_leakage(rows)
    assert {row["fold"] for row in rows} == {str(f) for f in [TEST_FOLD, *range(5)]}
    for row in rows:
        is_test_fold = row["fold"] == str(TEST_FOLD)
        assert (row["subset"] == Subset.TEST.value) == is_test_fold
    images = sorted(path.name for path in (out_dir / "images").iterdir())
    assert images == sorted(row["file_name"] for row in rows)
    for row in rows:
        label_path = (out_dir / "labels" / row["file_name"]).with_suffix(".txt")
        assert assert_label_file_valid(label_path) == int(row["n_boxes"])

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["selected"]["images"] == len(rows)
    assert len(list(out_dir.glob("fold*.yaml"))) == 5
    assert (out_dir / TEST_YAML_NAME).is_file()

    # The copied JPG must be byte-identical to the zip member.
    first = rows[0]
    pairs = aihub.list_aihub_bscan_zip_pairs(aihub_root, AIHubSplit(first["split_src"]))
    pair = next(pair for pair in pairs if pair.category == first["category"])
    with zipfile.ZipFile(pair.image_zip) as image_zip:
        original = image_zip.read(first["member"])
    assert (out_dir / "images" / first["file_name"]).read_bytes() == original

    with pytest.raises(FileExistsError):
        build_dataset(config)


@pytest.mark.data
def test_build_script_dry_run_writes_nothing(aihub_root: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "dry"
    manifest_path = tmp_path / "dry.csv"
    completed = run_build_script(
        "--aihub-root",
        str(aihub_root),
        "--out",
        str(out_dir),
        "--manifest",
        str(manifest_path),
        "--limit",
        str(LABELS_PER_ZIP),
        "--dry-run",
        "--pipe-share-min",
        "0",
        "--pipe-share-max",
        "1",
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["selected"]["images"] > 0
    assert not out_dir.exists()
    assert not manifest_path.exists()
