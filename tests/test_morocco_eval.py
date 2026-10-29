"""Tests for the Morocco cross-domain evaluation set (M1-4).

Unit tests use synthetic refs and a tiny synthetic package written to
``tmp_path``. The integration test builds the full set from the real Morocco
package and is skipped when it is missing.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from gpr_detection.data.morocco_eval import (
    EVAL_LIST_NAME,
    EVAL_YAML_NAME,
    MANIFEST_COLUMNS,
    DuplicateGroupError,
    GroupSelection,
    MoroccoEvalConfig,
    MoroccoSubset,
    SelectionRule,
    assert_one_image_per_group,
    augmentation_number,
    build_morocco_eval,
    check_subset_classes,
    parse_morocco_label,
    select_eval_images,
    select_group_copy,
)
from gpr_detection.io.classes import DetectionClass, class_names
from gpr_detection.io.morocco import MoroccoImageRef
from gpr_detection.io.paths import load_dataset_paths

SUBSET_PREFIX = {"utilities": "U", "cavities": "C", "intact": "I"}
SYNTHETIC_SIDE_PX = 32
EXPECTED_GROUPS = 233


def make_ref(
    subset: str,
    stem: str,
    aug_number: int,
    aug_transform: str,
    group_id: str | None = None,
    split: str = "train",
) -> MoroccoImageRef:
    prefix = SUBSET_PREFIX[subset]
    image_id = f"{prefix}_{stem}_aug_{aug_number}"
    return MoroccoImageRef(
        image_id=image_id,
        split=split,
        subset=subset,
        stem=stem,
        group_id=group_id or f"{prefix}_{stem}",
        aug_transform=aug_transform,
        image_relpath=f"03_yolo_dataset/images/{split}/{image_id}.jpg",
    )


# --- selection rules ------------------------------------------------------------------


def test_augmentation_number_parses_suffix() -> None:
    assert augmentation_number("I_1-2_aug_12") == 12
    with pytest.raises(ValueError, match="_aug_"):
        augmentation_number("U_001")


@pytest.mark.parametrize(
    ("subset", "preferred_number"),
    [("utilities", 6), ("cavities", 13), ("intact", 12)],
)
def test_preferred_copy_per_subset(subset: str, preferred_number: int) -> None:
    refs = [
        make_ref(subset, "7", 1, "가우시안 잡음"),
        make_ref(subset, "7", preferred_number, "원본과 거의 동일"),
        make_ref(subset, "7", 4, "좌우 반전"),
    ]
    selection = select_group_copy(refs, ["7"])
    assert selection is not None
    assert selection.rule is SelectionRule.PREFERRED
    assert augmentation_number(selection.ref.image_id) == preferred_number


def test_preferred_copy_ties_go_to_first_group_stem() -> None:
    refs = [
        make_ref("utilities", "179", 6, "주파수 이동(원본과 거의 동일)", "U_177"),
        make_ref("utilities", "177", 6, "주파수 이동(원본과 거의 동일)", "U_177"),
    ]
    selection = select_group_copy(refs, ["177", "179"])
    assert selection is not None
    assert selection.ref.stem == "177"


def test_fallback_takes_first_geometry_preserving_copy() -> None:
    refs = [
        make_ref("utilities", "9", 2, "가로 이동 +20px"),
        make_ref("utilities", "9", 4, "좌우 반전"),
        make_ref("utilities", "9", 1, "가우시안 잡음"),
    ]
    selection = select_group_copy(refs, ["9"])
    assert selection is not None
    assert selection.rule is SelectionRule.FALLBACK
    assert selection.ref.image_id == "U_9_aug_1"


def test_group_without_geometry_preserving_copy_is_left_out() -> None:
    shifted = make_ref("cavities", "5", 8, "가로 이동 +20px")
    rotated = make_ref("cavities", "6", 10, "회전 +15°")
    kept = make_ref("cavities", "4", 13, "주파수 이동(원본과 거의 동일)")
    groups = {"C_5": ["5"], "C_6": ["6"], "C_4": ["4"]}

    result = select_eval_images([shifted, rotated, kept], groups)

    assert [s.ref.image_id for s in result.selections] == ["C_4_aug_13"]
    assert result.groups_without_copy == ("C_5", "C_6")
    assert result.geometry_distorting_images == 1
    assert result.labelled_images == 3


def test_preferred_copy_must_preserve_geometry() -> None:
    refs = [make_ref("intact", "3", 12, "회전 +15°")]
    with pytest.raises(ValueError, match="preferred copy"):
        select_group_copy(refs, ["3"])


def test_stem_missing_from_group_list_is_an_error() -> None:
    refs = [make_ref("intact", "3", 12, "원본과 거의 동일")]
    with pytest.raises(ValueError, match="not listed"):
        select_group_copy(refs, ["4"])


def test_duplicate_group_is_rejected() -> None:
    ref = make_ref("intact", "3", 12, "원본과 거의 동일")
    selection = GroupSelection(ref, SelectionRule.PREFERRED)
    assert_one_image_per_group([selection])
    with pytest.raises(DuplicateGroupError, match="I_3"):
        assert_one_image_per_group([selection, selection])


# --- labels ---------------------------------------------------------------------------


def test_label_classes_are_remapped_and_coordinates_kept() -> None:
    text = "0 0.599828 0.417667 0.497427 0.523156\n1 0.1 0.2 0.3 0.4\n\n"
    boxes = parse_morocco_label(text, "sample")
    assert [box.class_id for box in boxes] == [
        DetectionClass.PIPE_HYPERBOLA,
        DetectionClass.CAVITY,
    ]
    assert boxes[0].to_line() == "0 0.599828 0.417667 0.497427 0.523156"
    assert boxes[1].to_line().startswith("2 ")


def test_empty_label_gives_no_boxes() -> None:
    assert parse_morocco_label("", "intact") == ()
    assert parse_morocco_label("\n  \n", "intact") == ()


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("2 0.5 0.5 0.1 0.1", KeyError),
        ("0 0.5 0.5 0.1", ValueError),
        ("0 0.5 1.5 0.1 0.1", ValueError),
    ],
)
def test_bad_label_lines_are_rejected(text: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        parse_morocco_label(text, "bad")


def test_subset_class_check() -> None:
    pipe = parse_morocco_label("0 0.5 0.5 0.1 0.1", "pipe")
    cavity = parse_morocco_label("1 0.5 0.5 0.1 0.1", "cavity")
    check_subset_classes(MoroccoSubset.UTILITIES, pipe, "ok")
    check_subset_classes(MoroccoSubset.CAVITIES, cavity, "ok")
    with pytest.raises(ValueError, match="cavity"):
        check_subset_classes(MoroccoSubset.UTILITIES, cavity, "mixed")
    with pytest.raises(ValueError, match="intact"):
        check_subset_classes(MoroccoSubset.INTACT, pipe, "intact with box")


# --- synthetic package build ----------------------------------------------------------

INDEX_HEADER = "image_id,subset,stem,group_id,aug_transform,split,yolo_path"
SYNTHETIC_IMAGES = (
    # (subset, stem, aug, transform, group, split, label)
    ("utilities", "001", 1, "가우시안 잡음", "U_001", "val", "0 0.5 0.5 0.2 0.2"),
    ("utilities", "001", 3, "회전 +15°", "U_001", "val", "0 0.5 0.5 0.2 0.2"),
    (
        "utilities",
        "001",
        6,
        "주파수 이동(원본과 거의 동일)",
        "U_001",
        "val",
        "0 0.25 0.75 0.125 0.25",
    ),
    ("utilities", "002", 1, "가우시안 잡음", "U_002", "test", "0 0.5 0.5 0.3 0.3"),
    ("utilities", "002", 2, "가로 이동 +20px", "U_002", "test", "0 0.5 0.5 0.3 0.3"),
    (
        "cavities",
        "18",
        13,
        "주파수 이동(원본과 거의 동일)",
        "C_18",
        "train",
        "1 0.5 0.5 0.4 0.4",
    ),
    (
        "cavities",
        "23",
        13,
        "주파수 이동(원본과 거의 동일)",
        "C_18",
        "train",
        "1 0.5 0.5 0.4 0.4",
    ),
    ("intact", "1", 12, "원본과 거의 동일", "I_1", "train", ""),
    ("intact", "1", 9, "336x336 리사이즈", "I_1", "train", ""),
)
SYNTHETIC_GROUPS = (
    ("U_001", "utilities", "001"),
    ("U_002", "utilities", "002"),
    ("C_18", "cavities", "18 | 23"),
    ("I_1", "intact", "1"),
)


def _write_synthetic_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (SYNTHETIC_SIDE_PX, SYNTHETIC_SIDE_PX), np.uint8)
    is_encoded, encoded = cv2.imencode(".jpg", image)
    assert is_encoded
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def make_synthetic_package(root: Path) -> None:
    """Write a minimal Morocco package with the columns the loaders read."""
    meta = root / "01_메타데이터"
    meta.mkdir(parents=True)
    yolo = root / "03_yolo_dataset"
    yolo.mkdir()
    (yolo / "data.yaml").write_text(
        "names:\n  0: pipe\n  1: cavity\n", encoding="utf-8"
    )
    index_lines = [INDEX_HEADER]
    for seed, (subset, stem, aug, transform, group, split, label) in enumerate(
        SYNTHETIC_IMAGES
    ):
        image_id = f"{SUBSET_PREFIX[subset]}_{stem}_aug_{aug}"
        relpath = f"03_yolo_dataset/images/{split}/{image_id}.jpg"
        index_lines.append(
            f"{image_id},{subset},{stem},{group},{transform},{split},{relpath}"
        )
        _write_synthetic_image(root / relpath, seed)
        label_path = yolo / "labels" / split / f"{image_id}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(label + "\n" if label else "", encoding="utf-8")
    (meta / "images_index.csv").write_text(
        "\n".join(index_lines) + "\n", encoding="utf-8-sig"
    )
    group_lines = ["group_id,subset,stems"]
    group_lines += [",".join(row) for row in SYNTHETIC_GROUPS]
    (meta / "split_groups.csv").write_text(
        "\n".join(group_lines) + "\n", encoding="utf-8-sig"
    )


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_build_on_synthetic_package(tmp_path: Path) -> None:
    root = tmp_path / "morocco"
    make_synthetic_package(root)
    out_dir = tmp_path / "eval"
    manifest_path = tmp_path / "splits" / "eval.csv"

    summary = build_morocco_eval(MoroccoEvalConfig(root, out_dir, manifest_path))

    rows = read_manifest(manifest_path)
    assert tuple(rows[0]) == MANIFEST_COLUMNS
    by_group = {row["group_id"]: row for row in rows}
    assert sorted(by_group) == ["C_18", "I_1", "U_001", "U_002"]
    assert by_group["U_001"]["image_id"] == "U_001_aug_6"
    assert by_group["U_002"]["selection_rule"] == "fallback"
    assert by_group["U_002"]["image_id"] == "U_002_aug_1"
    assert by_group["C_18"]["image_id"] == "C_18_aug_13"
    assert by_group["C_18"]["anisotropic_stretch"] == "True"
    assert by_group["U_001"]["anisotropic_stretch"] == "False"
    assert {row["split_src"] for row in rows} == {"val", "test", "train"}

    for row in rows:
        copied = (out_dir / "images" / row["file_name"]).read_bytes()
        assert copied == (root / row["source_image"]).read_bytes()
    labels_dir = out_dir / "labels"
    assert (labels_dir / "U_001_aug_6.txt").read_text(encoding="utf-8") == (
        "0 0.250000 0.750000 0.125000 0.250000\n"
    )
    assert (labels_dir / "C_18_aug_13.txt").read_text(encoding="utf-8").startswith("2 ")
    assert (labels_dir / "I_1_aug_12.txt").read_text(encoding="utf-8") == ""

    eval_list = (out_dir / EVAL_LIST_NAME).read_text(encoding="utf-8").splitlines()
    assert sorted(eval_list) == sorted(f"./images/{r['file_name']}" for r in rows)
    config = yaml.safe_load((out_dir / EVAL_YAML_NAME).read_text(encoding="utf-8"))
    assert config["val"] == config["test"] == EVAL_LIST_NAME
    assert config["names"] == class_names()

    assert summary["fallback"]["count"] == 1
    assert summary["source"]["excluded_rotation_elastic"] == 1
    assert summary["subsets"]["intact"]["images_without_boxes"] == 1
    assert summary["subsets"]["cavities"]["boxes_by_class"] == {"cavity": 1}
    written = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    # JSON turns the integer class ids into string keys.
    assert written == json.loads(json.dumps(summary))


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "morocco"
    make_synthetic_package(root)
    out_dir = tmp_path / "eval"
    manifest_path = tmp_path / "eval.csv"

    summary = build_morocco_eval(
        MoroccoEvalConfig(root, out_dir, manifest_path, dry_run=True)
    )

    assert summary["total"]["images"] == len(SYNTHETIC_GROUPS)
    assert not out_dir.exists()
    assert not manifest_path.exists()


def test_build_refuses_non_empty_output(tmp_path: Path) -> None:
    root = tmp_path / "morocco"
    make_synthetic_package(root)
    out_dir = tmp_path / "eval"
    out_dir.mkdir()
    (out_dir / "old.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_morocco_eval(MoroccoEvalConfig(root, out_dir, tmp_path / "m.csv"))


# --- real data ------------------------------------------------------------------------


@pytest.mark.data
def test_build_on_real_morocco_package(tmp_path: Path) -> None:
    root = load_dataset_paths().morocco
    if not root.is_dir():
        pytest.skip(f"Morocco data not found at {root}")
    manifest_path = tmp_path / "morocco_eval.csv"

    summary = build_morocco_eval(
        MoroccoEvalConfig(root, tmp_path / "eval", manifest_path)
    )

    rows = read_manifest(manifest_path)
    groups = Counter(row["group_id"] for row in rows)
    assert len(groups) == len(rows) == summary["total"]["groups"]
    assert len(rows) == EXPECTED_GROUPS - len(
        summary["source"]["groups_without_geometry_preserving_copy"]
    )
    allowed_classes = {"utilities": {"0"}, "cavities": {"2"}, "intact": set()}
    for row in rows:
        classes = set(row["classes"].split(";")) - {""}
        assert classes <= allowed_classes[row["subset"]]
        assert (row["subset"] == "cavities") == (row["anisotropic_stretch"] == "True")
    assert summary["subsets"]["intact"]["boxes"] == 0
