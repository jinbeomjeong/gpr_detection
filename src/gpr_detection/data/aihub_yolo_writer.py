"""Write the planned AIHub YOLO dataset: images, labels, lists, yamls, manifest.

Layout under ``out_dir``:

* ``images/*.jpg`` - original JPG bytes, unchanged (each decoded once with cv2
  IMREAD_GRAYSCALE only to check it is readable and matches the XML size)
* ``labels/*.txt`` - YOLO ``class xc yc w h``
* ``fold{k}_images.txt``, ``test_images.txt`` - image lists
* ``fold{k}.yaml`` (train = other folds, val = fold k) and ``test.yaml``
  (train = all folds, val = test = test list) for Ultralytics
* ``summary.json`` - counts per subset, fold and class
"""

import csv
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gpr_detection.data.aihub_yolo import (
    MANIFEST_COLUMNS,
    PIPE_CLASSES,
    TEST_FOLD,
    AIHubYoloItem,
    DatasetPlan,
    SplitOptions,
    Subset,
    collect_items,
    manifest_rows,
    plan_dataset,
    rng_stream_for_test_draw,
)
from gpr_detection.data.splits import AIHUB_LEAK_KEY_COLUMNS, assert_no_fold_leakage
from gpr_detection.io._common import decode_grayscale
from gpr_detection.io.aihub import (
    AIHubSplit,
    AIHubZipPair,
    AIHubZipReader,
    list_aihub_bscan_zip_pairs,
)
from gpr_detection.io.classes import DetectionClass, class_names

logger = logging.getLogger(__name__)

IMAGES_DIR_NAME = "images"
LABELS_DIR_NAME = "labels"
SUMMARY_FILE_NAME = "summary.json"
TEST_LIST_NAME = "test_images.txt"
TEST_YAML_NAME = "test.yaml"


@dataclass(frozen=True)
class BuildConfig:
    """Options of one dataset build.

    Attributes:
        aihub_root: AIHub ``01-1.정식개방데이터`` folder (read only).
        out_dir: Output folder for images, labels, lists and yamls.
        manifest_path: Output manifest CSV.
        options: Seed, k, balance ratio and test hold-out settings.
        limit_per_zip: Read only the first N labels of each zip (for tests).
        dry_run: Plan and report only; write nothing.
    """

    aihub_root: Path
    out_dir: Path
    manifest_path: Path
    options: SplitOptions
    limit_per_zip: int | None = None
    dry_run: bool = False


# --- summary -------------------------------------------------------------------


def _class_counts(items: Sequence[AIHubYoloItem]) -> dict[str, dict[str, int]]:
    """Images containing each class, its boxes, crops and ``<source>`` sessions."""
    counts = {}
    for detection_class in DetectionClass:
        with_class = [item for item in items if detection_class in item.class_ids]
        counts[detection_class.label] = {
            "images": len(with_class),
            "boxes": sum(
                box.class_id == detection_class for item in items for box in item.boxes
            ),
            "crops": len({item.crop_key for item in with_class}),
            "sessions": len({item.session for item in with_class}),
        }
    return counts


def _part_summary(plan: DatasetPlan, indices: Sequence[int]) -> dict[str, Any]:
    """Totals and per-class counts of the items at ``indices``."""
    items = [plan.items[i] for i in indices]
    return {
        "images": len(items),
        "boxes": sum(len(item.boxes) for item in items),
        "crops": len({item.crop_key for item in items}),
        "sessions": len({item.session for item in items}),
        "split_groups": len({plan.split_groups[i] for i in indices}),
        "gains": dict(sorted(Counter(str(item.gain) for item in items).items())),
        "classes": _class_counts(items),
    }


def _test_selection_summary(plan: DatasetPlan) -> dict[str, Any]:
    """Accepted attempt, its pipe shares and the test sessions of each pipe class."""
    options, selection = plan.options, plan.test_selection
    pipe_sessions = {}
    for pipe_class in PIPE_CLASSES:
        pipe_sessions[pipe_class.label] = sorted(
            {
                item.session
                for item, subset in zip(plan.items, plan.subsets, strict=True)
                if subset is Subset.TEST and pipe_class in item.class_ids
            }
        )
    return {
        "attempt": selection.attempt,
        "rng_stream": rng_stream_for_test_draw(options.seed, selection.attempt),
        "pipe_image_share": selection.pipe_image_shares,
        "pipe_share_bounds": [options.pipe_share_min, options.pipe_share_max],
        "max_attempts": options.max_test_attempts,
        "pipe_test_sessions": pipe_sessions,
    }


def summarize(plan: DatasetPlan, limit_per_zip: int | None) -> dict[str, Any]:
    """Counts for ``summary.json`` and the dry-run report."""
    options = plan.options
    indices_by_subset: dict[Subset, list[int]] = defaultdict(list)
    indices_by_fold: dict[int, list[int]] = defaultdict(list)
    for index, (subset, fold) in enumerate(zip(plan.subsets, plan.folds, strict=True)):
        indices_by_subset[subset].append(index)
        indices_by_fold[fold].append(index)
    return {
        "dataset": "aihub_71327_bscan",
        "seed": options.seed,
        "k": options.k,
        "balance_ratio": options.balance_ratio,
        "test_fraction": options.test_fraction,
        "min_test_pipe_sessions": options.min_test_pipe_sessions,
        "limit_per_zip": limit_per_zip,
        "class_names": class_names(),
        "source": plan.source_stats,
        "balance": {
            subset.value: {
                "target_images_per_downsampled_class": result.target_images_per_class,
                "available_images": result.available_images,
                "kept_images": result.kept_images,
                "available_crops": result.available_crops,
                "kept_crops": result.kept_crops,
            }
            for subset, result in plan.balance.items()
        },
        "selected": {
            **_part_summary(plan, range(len(plan.items))),
            "renamed_duplicate_stems": sum("__dup" in n for n in plan.file_names),
        },
        "subsets": {
            subset.value: _part_summary(plan, indices_by_subset[subset])
            for subset in Subset
        },
        "test_selection": _test_selection_summary(plan),
        "test_sessions": list(plan.test_sessions),
        "folds": [
            {"fold": fold, **_part_summary(plan, indices_by_fold[fold])}
            for fold in range(options.k)
        ],
    }


# --- writing -------------------------------------------------------------------


def _ensure_empty_output(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Output folder {out_dir} is not empty; remove it or choose another --out"
        )


def _items_by_pair(
    plan: DatasetPlan,
) -> Iterator[tuple[tuple[AIHubSplit, str], list[int]]]:
    indices_by_pair: dict[tuple[AIHubSplit, str], list[int]] = defaultdict(list)
    for index, item in enumerate(plan.items):
        indices_by_pair[item.pair_key].append(index)
    yield from sorted(indices_by_pair.items(), key=lambda entry: entry[0])


def _copy_checked_image(
    reader: AIHubZipReader, item: AIHubYoloItem, destination: Path
) -> None:
    """Write the original JPG bytes after checking they decode to the XML size."""
    encoded = reader.read_image_bytes(item.member_stem)
    decoded = decode_grayscale(encoded, f"{item.split.value}:{item.image_member}")
    expected_shape = (item.image_height_px, item.image_width_px)
    if decoded.shape != expected_shape:
        raise ValueError(
            f"{item.image_member}: image is {decoded.shape}, XML says {expected_shape}"
        )
    destination.write_bytes(encoded)


def write_images_and_labels(
    plan: DatasetPlan, pairs: Sequence[AIHubZipPair], out_dir: Path
) -> None:
    """Copy each selected JPG unchanged and write its YOLO label file."""
    images_dir = out_dir / IMAGES_DIR_NAME
    labels_dir = out_dir / LABELS_DIR_NAME
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    pair_by_key = {(pair.split, pair.category): pair for pair in pairs}

    for pair_key, indices in _items_by_pair(plan):
        with AIHubZipReader(pair_by_key[pair_key]) as reader:
            for index in indices:
                item, file_name = plan.items[index], plan.file_names[index]
                _copy_checked_image(reader, item, images_dir / file_name)
                label_text = "".join(f"{box.to_line()}\n" for box in item.boxes)
                label_path = (labels_dir / file_name).with_suffix(".txt")
                label_path.write_text(label_text, encoding="utf-8")
        logger.info(
            "%s %s: wrote %d images", pair_key[0].value, pair_key[1], len(indices)
        )


def fold_list_name(fold: int) -> str:
    """Image list file of one fold (or of test for ``TEST_FOLD``), in the root."""
    return TEST_LIST_NAME if fold == TEST_FOLD else f"fold{fold}_images.txt"


def _write_yaml(path: Path, out_dir: Path, train: list[str], val: str) -> None:
    data_config: dict[str, Any] = {
        "path": str(out_dir.resolve()),
        "train": train,
        "val": val,
    }
    if val == TEST_LIST_NAME:
        data_config["test"] = TEST_LIST_NAME
    data_config["names"] = class_names()
    yaml_text = yaml.safe_dump(data_config, sort_keys=False, allow_unicode=True)
    path.write_text(yaml_text, encoding="utf-8")


def write_fold_files(plan: DatasetPlan, out_dir: Path) -> None:
    """Write the image lists and the Ultralytics data yamls.

    ``fold{k}.yaml`` trains on the other train-pool folds and validates on fold
    k. ``test.yaml`` trains on the whole train pool and validates on test.
    List lines start with ``./`` because Ultralytics resolves such lines against
    the folder of the list file.
    """
    names_by_fold: dict[int, list[str]] = defaultdict(list)
    for file_name, fold in zip(plan.file_names, plan.folds, strict=True):
        names_by_fold[fold].append(file_name)
    for fold in [*range(plan.options.k), TEST_FOLD]:
        lines = "".join(f"./{IMAGES_DIR_NAME}/{n}\n" for n in names_by_fold[fold])
        (out_dir / fold_list_name(fold)).write_text(lines, encoding="utf-8")

    all_folds = [fold_list_name(fold) for fold in range(plan.options.k)]
    for eval_fold in range(plan.options.k):
        train = [name for name in all_folds if name != fold_list_name(eval_fold)]
        _write_yaml(
            out_dir / f"fold{eval_fold}.yaml", out_dir, train, fold_list_name(eval_fold)
        )
    _write_yaml(out_dir / TEST_YAML_NAME, out_dir, all_folds, TEST_LIST_NAME)


def write_manifest(plan: DatasetPlan, manifest_path: Path) -> None:
    """Write the fold manifest CSV (UTF-8, one row per image)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows(plan))


def build_dataset(config: BuildConfig) -> dict[str, Any]:
    """Read the AIHub zips, plan the dataset and write it (unless dry run).

    Returns:
        The summary (also written to ``summary.json`` when not a dry run).

    Raises:
        FileExistsError: If ``config.out_dir`` is not empty (not for dry runs).
        FileNotFoundError: If no B-scan zip pair is found.
        SplitLeakageError: If a session or crop key spans test/train pool/folds.
    """
    if not config.dry_run:
        _ensure_empty_output(config.out_dir)

    pairs = [
        pair
        for split in AIHubSplit
        for pair in list_aihub_bscan_zip_pairs(config.aihub_root, split)
    ]
    if not pairs:
        raise FileNotFoundError(f"No B-scan zip pairs under {config.aihub_root}")
    plan = plan_dataset(collect_items(pairs, config.limit_per_zip), config.options)
    summary = summarize(plan, config.limit_per_zip)
    if config.dry_run:
        return summary

    write_images_and_labels(plan, pairs, config.out_dir)
    write_fold_files(plan, config.out_dir)
    write_manifest(plan, config.manifest_path)
    # Re-read the written manifest so the check covers the file itself.
    with config.manifest_path.open(encoding="utf-8", newline="") as handle:
        assert_no_fold_leakage(csv.DictReader(handle), AIHUB_LEAK_KEY_COLUMNS)
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    (config.out_dir / SUMMARY_FILE_NAME).write_text(summary_text, encoding="utf-8")
    return summary
