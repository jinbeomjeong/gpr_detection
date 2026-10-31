"""Leakage re-assertions (1)-(6) for a v2 dataset derived from AIHub YOLO v1.

``labeling/pipe_band_relabel_plan.md`` section 4. Every check re-reads the
written files (manifests, lists, yamls, labels) instead of trusting the
builder's memory:

1. Same rows as v1 (``file_name`` set) with the same split columns, and the
   ``images/`` folder holds exactly those files with the v1 bytes.
2. No ``session``, ``session_from_name``, ``crop_key`` or ``split_group`` value
   is shared by test and train pool, or by two of the six parts (test, fold
   0-4): 16 part pairs x 4 keys = 64 intersections, all empty.
3. The gain copies of a crop have byte-identical labels in v2 (and in
   ``test_v1``).
4. The fold lists together are the manifest, each list holds only its fold,
   and no yaml trains on a file of its validation list.
5. Lines of classes 0, 2 and 3 are unchanged; ``test_v1`` labels are the v1
   bytes.
6. ``relabel_log.csv`` has one row per image and its band counts match the
   v1 and v2 label files.
"""

import csv
import itertools
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gpr_detection.data.aihub_yolo_v2_merge import (
    BAND_CLASS_ID,
    FOLD_COUNT,
    IMAGES_DIR_NAME,
    LABELS_DIR_NAME,
    RELABEL_LOG_NAME,
    TEST_FOLD,
    TEST_LIST_NAME,
    TEST_V1_DIR_NAME,
    TEST_YAML_NAME,
    fold_list_name,
    fold_yaml_name,
    parse_label_text,
    read_manifest,
)

SPLIT_COLUMNS = (
    "file_name",
    "subset",
    "fold",
    "session",
    "session_from_name",
    "crop_key",
    "gain",
    "split_group",
    "seed",
    "test_attempt",
)
LEAK_KEY_COLUMNS = ("session", "session_from_name", "crop_key", "split_group")
_MAX_EXAMPLES = 5


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one re-assertion."""

    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckInputs:
    """Folders and manifests compared by the checks."""

    v1_dir: Path
    v1_manifest: Path
    v2_dir: Path
    v2_manifest: Path


def _label_text(dataset_dir: Path, file_name: str) -> str:
    path = (dataset_dir / LABELS_DIR_NAME / file_name).with_suffix(".txt")
    return path.read_text(encoding="utf-8")


def _list_file_names(list_path: Path) -> list[str]:
    lines = list_path.read_text(encoding="utf-8-sig").splitlines()
    return [Path(line.strip()).name for line in lines if line.strip()]


def _same_bytes(first: Path, second: Path) -> bool:
    if first.samefile(second):
        return True
    return first.read_bytes() == second.read_bytes()


# --- (1) rows -------------------------------------------------------------------


def check_rows(inputs: CheckInputs) -> CheckResult:
    """(1) v2 rows, split columns and images equal v1."""
    v1_rows = {row["file_name"]: row for row in read_manifest(inputs.v1_manifest)}
    v2_rows = {row["file_name"]: row for row in read_manifest(inputs.v2_manifest)}
    mismatched = sorted(
        name
        for name in set(v1_rows) & set(v2_rows)
        if any(v1_rows[name][c] != v2_rows[name][c] for c in SPLIT_COLUMNS)
    )
    images_dir = inputs.v2_dir / IMAGES_DIR_NAME
    on_disk = {path.name for path in images_dir.iterdir()}
    differing_images = sorted(
        name
        for name in set(v1_rows) & on_disk
        if not _same_bytes(images_dir / name, inputs.v1_dir / IMAGES_DIR_NAME / name)
    )
    details = {
        "rows_v1": len(v1_rows),
        "rows_v2": len(v2_rows),
        "only_v1": len(set(v1_rows) - set(v2_rows)),
        "only_v2": len(set(v2_rows) - set(v1_rows)),
        "split_column_mismatches": len(mismatched),
        "images_on_disk": len(on_disk),
        "images_not_in_manifest": len(on_disk - set(v2_rows)),
        "manifest_images_missing": len(set(v2_rows) - on_disk),
        "images_differing_from_v1": len(differing_images),
        "examples": (mismatched + differing_images)[:_MAX_EXAMPLES],
    }
    passed = (
        details["only_v1"] == details["only_v2"] == 0
        and not mismatched
        and on_disk == set(v2_rows)
        and not differing_images
    )
    return CheckResult("1_rows_equal_v1", passed, details)


# --- (2) leakage ----------------------------------------------------------------


def _part_of(row: Mapping[str, str]) -> str:
    return "test" if row["fold"] == TEST_FOLD else f"fold{row['fold']}"


def _values_by_group(
    rows: Iterable[Mapping[str, str]], group_of: Callable[[Mapping[str, str]], str]
) -> dict[str, dict[str, set[str]]]:
    """``{key column: {group: values}}``."""
    values: dict[str, dict[str, set[str]]] = {
        c: defaultdict(set) for c in LEAK_KEY_COLUMNS
    }
    for row in rows:
        for column in LEAK_KEY_COLUMNS:
            values[column][group_of(row)].add(row[column])
    return values


def check_leakage(inputs: CheckInputs) -> CheckResult:
    """(2) 64 key intersections between test/train pool and the six parts."""
    rows = read_manifest(inputs.v2_manifest)
    by_subset = _values_by_group(rows, lambda row: row["subset"])
    by_part = _values_by_group(rows, _part_of)
    subset_pairs = [("test", "trainpool")]
    part_pairs = list(itertools.combinations(sorted(by_part["crop_key"]), 2))

    leaks = []
    checked = 0
    for column in LEAK_KEY_COLUMNS:
        for grouping, pairs in ((by_subset, subset_pairs), (by_part, part_pairs)):
            for first, second in pairs:
                shared = grouping[column][first] & grouping[column][second]
                checked += 1
                if shared:
                    leaks.append(f"{column} {first}/{second}: {sorted(shared)[:3]}")
    details = {
        "intersections_checked": checked,
        "non_empty": len(leaks),
        "leaks": leaks,
    }
    expected = len(LEAK_KEY_COLUMNS) * (1 + (FOLD_COUNT + 1) * FOLD_COUNT // 2)
    passed = not leaks and checked == expected
    return CheckResult("2_no_key_shared_across_parts", passed, details)


# --- (3) gain copies ------------------------------------------------------------


def _crops_with_differing_labels(
    rows: Iterable[Mapping[str, str]], dataset_dir: Path
) -> list[str]:
    texts_by_crop: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        texts_by_crop[row["crop_key"]].add(_label_text(dataset_dir, row["file_name"]))
    return sorted(key for key, texts in texts_by_crop.items() if len(texts) > 1)


def check_gain_copies(inputs: CheckInputs) -> CheckResult:
    """(3) Gain copies of a crop share one label text in v2 and test_v1."""
    rows = read_manifest(inputs.v2_manifest)
    test_rows = [row for row in rows if row["fold"] == TEST_FOLD]
    differing_v2 = _crops_with_differing_labels(rows, inputs.v2_dir)
    differing_test_v1 = _crops_with_differing_labels(
        test_rows, inputs.v2_dir / TEST_V1_DIR_NAME
    )
    crop_count = len({row["crop_key"] for row in rows})
    details = {
        "crops": crop_count,
        "crops_differing_v2": len(differing_v2),
        "crops_differing_test_v1": len(differing_test_v1),
        "examples": (differing_v2 + differing_test_v1)[:_MAX_EXAMPLES],
    }
    passed = not differing_v2 and not differing_test_v1
    return CheckResult("3_gain_copies_identical", passed, details)


# --- (4) lists and yamls --------------------------------------------------------


def _train_val_overlap(dataset_dir: Path, yaml_name: str) -> dict[str, Any]:
    """Files shared by the train lists and the val list of one data yaml."""
    data = yaml.safe_load((dataset_dir / yaml_name).read_text(encoding="utf-8"))
    root = Path(data["path"])
    train_entries = (
        data["train"] if isinstance(data["train"], list) else [data["train"]]
    )
    train_files: set[str] = set()
    for entry in train_entries:
        train_files.update(_list_file_names(root / entry))
    val_files = set(_list_file_names(root / data["val"]))
    val_listed_as_train = any(
        Path(entry).name == data["val"] for entry in train_entries
    )
    return {
        "yaml": f"{dataset_dir.name}/{yaml_name}",
        "path_is_this_folder": root.resolve() == dataset_dir.resolve(),
        "val_list_in_train": val_listed_as_train,
        "train_files": len(train_files),
        "val_files": len(val_files),
        "shared_files": len(train_files & val_files),
    }


def check_lists(inputs: CheckInputs) -> CheckResult:
    """(4) Lists = manifest, one fold per list, no yaml trains on its val files."""
    rows = read_manifest(inputs.v2_manifest)
    fold_of = {row["file_name"]: row["fold"] for row in rows}
    folds = [*(str(k) for k in range(FOLD_COUNT)), TEST_FOLD]

    listed: list[str] = []
    wrong_fold = 0
    for fold in folds:
        names = _list_file_names(inputs.v2_dir / fold_list_name(fold))
        listed.extend(names)
        wrong_fold += sum(fold_of.get(name) != fold for name in names)
    test_v1_names = _list_file_names(inputs.v2_dir / TEST_V1_DIR_NAME / TEST_LIST_NAME)
    test_names = {name for name, fold in fold_of.items() if fold == TEST_FOLD}
    duplicate_count = len(listed) - len(set(listed))
    is_test_v1_list_complete = set(test_v1_names) == test_names

    yaml_checks = [_train_val_overlap(inputs.v2_dir, fold_yaml_name(f)) for f in folds]
    yaml_checks.append(
        _train_val_overlap(inputs.v2_dir / TEST_V1_DIR_NAME, TEST_YAML_NAME)
    )
    details = {
        "listed_files": len(listed),
        "duplicates": duplicate_count,
        "manifest_files": len(fold_of),
        "files_in_wrong_list": wrong_fold,
        "test_v1_list_equals_test": is_test_v1_list_complete,
        "yamls": yaml_checks,
    }
    yamls_ok = all(
        check["path_is_this_folder"]
        and not check["val_list_in_train"]
        and check["shared_files"] == 0
        for check in yaml_checks
    )
    passed = (
        set(listed) == set(fold_of)
        and duplicate_count == 0
        and wrong_fold == 0
        and is_test_v1_list_complete
        and yamls_ok
    )
    return CheckResult("4_lists_match_manifest", passed, details)


# --- (5) other classes ----------------------------------------------------------


def _non_band_lines(text: str, description: str) -> list[str]:
    lines = parse_label_text(text, description)
    return [line.text for line in lines if line.class_id != BAND_CLASS_ID]


def check_other_classes(inputs: CheckInputs) -> CheckResult:
    """(5) Classes 0, 2, 3 unchanged; test_v1 labels are the v1 bytes."""
    rows = read_manifest(inputs.v2_manifest)
    changed = []
    unchanged_band_free = 0
    band_free_files = 0
    for row in rows:
        name = row["file_name"]
        v1_text = _label_text(inputs.v1_dir, name)
        v2_text = _label_text(inputs.v2_dir, name)
        if _non_band_lines(v1_text, name) != _non_band_lines(v2_text, name):
            changed.append(name)
        if row["band_boxes_v1"] == "0":
            band_free_files += 1
            unchanged_band_free += v1_text == v2_text
    test_v1_dir = inputs.v2_dir / TEST_V1_DIR_NAME
    test_v1_differing = [
        row["file_name"]
        for row in rows
        if row["fold"] == TEST_FOLD
        and _label_text(test_v1_dir, row["file_name"])
        != _label_text(inputs.v1_dir, row["file_name"])
    ]
    details = {
        "files_with_changed_other_classes": len(changed),
        "band_free_files": band_free_files,
        "band_free_files_byte_identical": unchanged_band_free,
        "test_v1_labels_differing_from_v1": len(test_v1_differing),
        "examples": (changed + test_v1_differing)[:_MAX_EXAMPLES],
    }
    passed = (
        not changed and unchanged_band_free == band_free_files and not test_v1_differing
    )
    return CheckResult("5_other_classes_unchanged", passed, details)


# --- (6) band counts and log ----------------------------------------------------


def _band_count(text: str, description: str) -> int:
    lines = parse_label_text(text, description)
    return sum(line.class_id == BAND_CLASS_ID for line in lines)


def check_band_log(inputs: CheckInputs) -> CheckResult:
    """(6) relabel_log.csv covers every image and matches the label files."""
    rows = read_manifest(inputs.v2_manifest)
    with (inputs.v2_dir / RELABEL_LOG_NAME).open(
        encoding="utf-8", newline=""
    ) as handle:
        log_by_file = {row["file_name"]: row for row in csv.DictReader(handle)}
    mismatched = []
    grown = 0
    totals = {"v1": 0, "v2": 0}
    for row in rows:
        name = row["file_name"]
        count_v1 = _band_count(_label_text(inputs.v1_dir, name), name)
        count_v2 = _band_count(_label_text(inputs.v2_dir, name), name)
        totals["v1"] += count_v1
        totals["v2"] += count_v2
        grown += count_v2 > count_v1
        log_row = log_by_file.get(name)
        expected = (str(count_v1), str(count_v2))
        if (
            log_row is None
            or (log_row["band_boxes_v1"], log_row["band_boxes_v2"]) != expected
            or (row["band_boxes_v1"], row["band_boxes_v2"]) != expected
        ):
            mismatched.append(name)
    details = {
        "log_rows": len(log_by_file),
        "manifest_rows": len(rows),
        "band_boxes_v1_images": totals["v1"],
        "band_boxes_v2_images": totals["v2"],
        "files_with_more_band_boxes": grown,
        "count_mismatches": len(mismatched),
        "examples": mismatched[:_MAX_EXAMPLES],
    }
    passed = len(log_by_file) == len(rows) and not mismatched and grown == 0
    return CheckResult("6_band_counts_logged", passed, details)


def run_all_checks(inputs: CheckInputs) -> list[CheckResult]:
    """Run re-assertions (1)-(6) in order."""
    checks: tuple[Callable[[CheckInputs], CheckResult], ...] = (
        check_rows,
        check_leakage,
        check_gain_copies,
        check_lists,
        check_other_classes,
        check_band_log,
    )
    return [check(inputs) for check in checks]
