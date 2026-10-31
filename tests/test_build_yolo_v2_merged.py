"""End-to-end tests of scripts/build_yolo_v2_merged.py on a tiny synthetic v1."""

import csv
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest
import yaml

from gpr_detection.data.aihub_yolo import MANIFEST_COLUMNS
from gpr_detection.data.aihub_yolo_v2_checks import (
    CheckInputs,
    check_leakage,
    run_all_checks,
)
from gpr_detection.data.aihub_yolo_v2_merge import read_manifest

IMAGE_SIDE_PX = 256
GAINS = ("0.5", "1.0")
PARTS = ("0", "1", "2", "3", "4", "-1")
# Two band fragments 16 px apart on one row, a cavity box and a far band.
FRAGMENTED_LABEL = (
    "2 0.200000 0.700000 0.100000 0.100000\n"
    "1 0.100000 0.187500 0.200000 0.062500\n"
    "1 0.362500 0.187500 0.200000 0.062500\n"
    "1 0.800000 0.500000 0.100000 0.062500\n"
)
CAVITY_ONLY_LABEL = "2 0.500000 0.500000 0.200000 0.200000\n"


def _write_image(path: Path) -> None:
    pixels = np.full((IMAGE_SIDE_PX, IMAGE_SIDE_PX), 128, dtype=np.uint8)
    is_encoded, encoded = cv2.imencode(".jpg", pixels)
    assert is_encoded
    path.write_bytes(encoded.tobytes())


def _manifest_row(file_name: str, part: str, crop: str, gain: str) -> dict[str, str]:
    row = dict.fromkeys(MANIFEST_COLUMNS, "")
    session = f"S{part}"
    row.update(
        file_name=file_name,
        session=session,
        session_from_name=session,
        crop_key=crop,
        gain=gain,
        n_boxes="0",
        subset="test" if part == "-1" else "trainpool",
        fold=part,
        seed="20261029",
        test_attempt="0",
        split_group=f"G{part}",
    )
    return row


def _list_name(part: str) -> str:
    return "test_images.txt" if part == "-1" else f"fold{part}_images.txt"


def _write_yamls(dataset_dir: Path) -> None:
    fold_lists = [_list_name(part) for part in PARTS[:-1]]
    names = {0: "pipe_hyperbola", 1: "pipe_band", 2: "cavity", 3: "manhole"}
    for part in PARTS:
        train = (
            fold_lists
            if part == "-1"
            else [n for n in fold_lists if n != _list_name(part)]
        )
        data = {"path": str(dataset_dir), "train": train, "val": _list_name(part)}
        if part == "-1":
            data["test"] = _list_name(part)
        data["names"] = names
        yaml_name = "test.yaml" if part == "-1" else f"fold{part}.yaml"
        (dataset_dir / yaml_name).write_text(yaml.safe_dump(data, sort_keys=False))


def make_v1(root: Path) -> tuple[Path, Path]:
    """Write a v1-shaped dataset: per part two crops x two gains."""
    dataset_dir = root / "aihub_yolo_v1"
    (dataset_dir / "images").mkdir(parents=True)
    (dataset_dir / "labels").mkdir()
    rows = []
    for part in PARTS:
        list_lines = []
        for crop_index, label in enumerate((FRAGMENTED_LABEL, CAVITY_ONLY_LABEL)):
            crop = f"P{part}_C{crop_index}"
            for gain in GAINS:
                file_name = f"{crop}_g{gain}.jpg"
                _write_image(dataset_dir / "images" / file_name)
                label_path = dataset_dir / "labels" / f"{crop}_g{gain}.txt"
                label_path.write_text(label, encoding="utf-8")
                row = _manifest_row(file_name, part, crop, gain)
                row["n_boxes"] = str(len(label.splitlines()))
                rows.append(row)
                list_lines.append(f"./images/{file_name}\n")
        (dataset_dir / _list_name(part)).write_text("".join(list_lines))
    _write_yamls(dataset_dir)
    manifest = root / "aihub_yolo_v1.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return dataset_dir, manifest


def _tree_digest(*paths: Path) -> dict[str, str]:
    digests = {}
    for path in paths:
        files = [path] if path.is_file() else sorted(path.rglob("*"))
        for file in files:
            if file.is_file():
                digests[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    return digests


@pytest.fixture
def built(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> tuple[int, CheckInputs, dict[str, str]]:
    v1_dir, v1_manifest = make_v1(tmp_path)
    before = _tree_digest(v1_dir, v1_manifest)
    inputs = CheckInputs(v1_dir, v1_manifest, tmp_path / "v2", tmp_path / "v2.csv")
    script = load_script("build_yolo_v2_merged")
    exit_code = script.main(
        [
            "--source", str(v1_dir),
            "--source-manifest", str(v1_manifest),
            "--out", str(inputs.v2_dir),
            "--manifest", str(inputs.v2_manifest),
        ]
    )  # fmt: skip
    assert _tree_digest(v1_dir, v1_manifest) == before, "v1 must stay unchanged"
    return exit_code, inputs, before


def test_build_merges_bands_and_passes_every_check(
    built: tuple[int, CheckInputs, dict[str, str]],
) -> None:
    exit_code, inputs, _ = built

    assert exit_code == 0
    checks = json.loads((inputs.v2_dir / "leak_checks.json").read_text("utf-8"))
    assert [check["passed"] for check in checks] == [True] * 6
    merged = (inputs.v2_dir / "labels" / "P0_C0_g1.0.txt").read_text("utf-8")
    assert merged.splitlines() == [
        "2 0.200000 0.700000 0.100000 0.100000",
        "1 0.231250 0.187500 0.462500 0.062500",
        "1 0.800000 0.500000 0.100000 0.062500",
    ]


def test_manifest_keeps_v1_rows_and_adds_band_columns(
    built: tuple[int, CheckInputs, dict[str, str]],
) -> None:
    _, inputs, _ = built
    rows = {row["file_name"]: row for row in read_manifest(inputs.v2_manifest)}

    merged_row, cavity_row = rows["P-1_C0_g0.5.jpg"], rows["P-1_C1_g0.5.jpg"]
    assert (merged_row["band_relabel"], merged_row["n_boxes"]) == ("auto_merged", "3")
    assert (merged_row["band_boxes_v1"], merged_row["band_boxes_v2"]) == ("3", "2")
    assert cavity_row["band_relabel"] == "not_in_scope"


def test_test_v1_keeps_the_v1_test_labels(
    built: tuple[int, CheckInputs, dict[str, str]],
) -> None:
    _, inputs, _ = built
    test_v1 = inputs.v2_dir / "test_v1"

    v1_label = (inputs.v1_dir / "labels" / "P-1_C0_g1.0.txt").read_bytes()
    assert (test_v1 / "labels" / "P-1_C0_g1.0.txt").read_bytes() == v1_label
    assert sorted(p.name for p in (test_v1 / "images").iterdir()) == [
        "P-1_C0_g0.5.jpg",
        "P-1_C0_g1.0.jpg",
        "P-1_C1_g0.5.jpg",
        "P-1_C1_g1.0.jpg",
    ]
    data = yaml.safe_load((test_v1 / "test.yaml").read_text(encoding="utf-8"))
    assert Path(data["path"]) == test_v1.resolve()
    assert [Path(entry).name for entry in data["train"]] == [
        f"fold{k}_images.txt" for k in range(5)
    ]


def test_checks_catch_a_crop_moved_to_another_fold(
    built: tuple[int, CheckInputs, dict[str, str]],
) -> None:
    _, inputs, _ = built
    rows = read_manifest(inputs.v2_manifest)
    rows[0]["fold"] = "4" if rows[0]["fold"] != "4" else "3"
    with inputs.v2_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert not check_leakage(inputs).passed
    assert not all(check.passed for check in run_all_checks(inputs))


def test_existing_output_is_refused(
    built: tuple[int, CheckInputs, dict[str, str]],
    load_script: Callable[[str], ModuleType],
) -> None:
    _, inputs, _ = built
    script = load_script("build_yolo_v2_merged")

    exit_code = script.main(
        [
            "--source", str(inputs.v1_dir),
            "--source-manifest", str(inputs.v1_manifest),
            "--out", str(inputs.v2_dir),
            "--manifest", str(inputs.v2_manifest),
        ]
    )  # fmt: skip

    assert exit_code == 2
