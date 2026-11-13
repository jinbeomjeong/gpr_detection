"""Real-data round trip of the line L1 image (spec_detection_image_v1 T7 (e), QA-8).

Opt-in: ``pytest -m realdata`` (also marked ``data``; skipped when a dataset
root is missing). Reads the section 9 candidate files read-only, builds the
line L1 image, repeats the T1 exact round trip on 200 random grid boxes per
line and checks the contract fields. Period-dependent flags
(``overlap_period_capped``, ``period_compressed``, ``scale_out_of_range``) and
the compression are only recorded, in ``data/processing/
detection_image_flags.csv`` (git-ignored) and the log; judging them is
gpr-processing-expert's job.
"""

import csv
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from gpr_detection.detect.base import PixelBox, letterbox
from gpr_detection.io.fourtu import load_fourtu_radargram
from gpr_detection.io.gprsd import DtHypothesis, load_gprsd_radargram
from gpr_detection.io.guangzhou import load_guangzhou_radargram
from gpr_detection.io.paths import DatasetName, DatasetPaths, load_dataset_paths
from gpr_detection.io.radargram import GridBox, Radargram
from gpr_detection.pipeline import (
    grid_box_to_image_box,
    image_box_to_grid_box,
    model_box_to_image_box,
    tile_box_to_line_box,
)
from gpr_detection.preprocess.to_image import (
    M3_2_ASSUMED,
    DetectionImage,
    make_detection_image,
)

pytestmark = [pytest.mark.data, pytest.mark.realdata]

LOGGER = logging.getLogger(__name__)
SEED = 20261113
N_ROUND_TRIP_BOXES = 200
TILE_PX = 256
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLAGS_CSV = PROJECT_ROOT / "data" / "processing" / "detection_image_flags.csv"
REPORTED_FLAGS = ("overlap_period_capped", "period_compressed", "scale_out_of_range")


@dataclass(frozen=True)
class Candidate:
    """One spec section 9 file and the contract values it must show."""

    name: str
    dataset: DatasetName
    load: Callable[[Path], Radargram]
    v_scale_source: str
    expected_flags: tuple[str, ...] = ()
    note: str = ""


def _fourtu(activity_id: str, path: str, note: str = "") -> Candidate:
    return Candidate(
        name=f"4tu/{activity_id}/{path}",
        dataset=DatasetName.FOURTU,
        load=lambda root: load_fourtu_radargram(root, activity_id, path),
        v_scale_source="eps_r_meta",
        note=note,
    )


def _guangzhou(file_id: str, extra_flags: tuple[str, ...] = (), note: str = ""):
    return Candidate(
        name=f"gz/{file_id}",
        dataset=DatasetName.GUANGZHOU,
        load=lambda root: load_guangzhou_radargram(root, file_id),
        v_scale_source="assumed",
        expected_flags=("v_scale_assumed", *extra_flags),
        note=note,
    )


def _gprsd(file_id: str) -> Candidate:
    return Candidate(
        name=f"gprsd/{file_id}",
        dataset=DatasetName.GPRSD,
        load=lambda root: load_gprsd_radargram(root, file_id, DtHypothesis.A),
        v_scale_source="depth_per_sample_rough",
        expected_flags=("amplitude_processed",),
    )


CANDIDATES = (
    _fourtu("03.9", "Path5", note="9 traces, one padded tile"),
    _fourtu("01.1", "Path1"),
    _fourtu("09.1", "Path2"),
    _fourtu("09.4", "Path1"),
    _guangzhou("pipe/001_190321AA/LID10001"),
    _guangzhou("pipe/1030_190117AF/LID10001"),
    _guangzhou("pipe/190321A_190321AH/LID10001"),
    _guangzhou(
        "pipe/MALA_07011/LA010001",
        ("amplitude_processed",),
        note="spec v0.4 expects compression ~2.7 + scale_out_of_range (P* 6)",
    ),
    _guangzhou(
        "pipe/1030_181224AB/LID10001",
        note="2000 MHz: spec expects overlap_period_capped",
    ),
    _gprsd("CementRoad2_1"),
    _gprsd("AsphaltRoad2_1"),
)


@pytest.fixture(scope="module")
def dataset_paths() -> DatasetPaths:
    return load_dataset_paths()


@pytest.fixture(scope="module")
def flag_rows():
    """Collects one csv row per candidate; written once after the module."""
    rows: list[dict[str, object]] = []
    yield rows
    if not rows:
        return
    FLAGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with FLAGS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {flag: sum(bool(row[flag]) for row in rows) for flag in REPORTED_FLAGS}
    LOGGER.info("T7(e) flag counts over %d lines: %s", len(rows), counts)


def _require_root(paths: DatasetPaths, dataset: DatasetName) -> Path:
    root = paths.root(dataset)
    if not root.is_dir():
        pytest.skip(f"{dataset.value} data not found at {root}")
    return root


def round_trip_mismatches(image: DetectionImage, n_boxes: int, seed: int) -> int:
    """T1 on a real line: random grid boxes through every containing tile.

    Returns:
        Number of (box, tile) paths whose L0 box differs from the original
        (0 is the exact round trip of QA-8).
    """
    grid_map = image.grid_map
    _, identity = letterbox(np.zeros((TILE_PX, TILE_PX), np.uint8), TILE_PX)
    rng = np.random.default_rng(seed)
    max_width = max(1, math.floor(image.overlap_px * grid_map.traces_per_px))
    mismatches = 0
    for _ in range(n_boxes):
        width = int(rng.integers(1, min(max_width, grid_map.n_traces) + 1))
        i0 = int(rng.integers(0, grid_map.n_traces - width + 1))
        k0 = int(rng.integers(0, grid_map.n_samples))
        k1 = int(rng.integers(k0, grid_map.n_samples))
        original = GridBox(i0, i0 + width - 1, k0, k1)
        line_box = grid_box_to_image_box(original, grid_map)
        containing = [
            tile
            for tile in image.tiles
            if tile.x0_px - 1e-9 <= line_box.x1
            and line_box.x2 <= tile.x0_px + tile.content_width_px + 1e-9
        ]
        if not containing:
            mismatches += 1
        for tile in containing:
            tile_box = PixelBox(
                line_box.x1 - tile.x0_px, line_box.y1, line_box.x2 - tile.x0_px,
                line_box.y2,
            )  # fmt: skip
            back = model_box_to_image_box(identity.apply(tile_box), identity)
            back_line = tile_box_to_line_box(back, tile)
            mismatches += image_box_to_grid_box(back_line, grid_map) != original
    return mismatches


def _flag_row(candidate: Candidate, image: DetectionImage, mismatches: int) -> dict:
    return {
        "line": candidate.name,
        "width_px": image.width_px,
        "content_height_px": image.content_height_px,
        "n_tiles": len(image.tiles),
        "period_samples": round(image.period_samples, 4),
        "period_source": str(image.period_source),
        "traces_per_px": round(image.grid_map.traces_per_px, 6),
        "samples_per_px": round(image.grid_map.samples_per_px, 6),
        "overlap_px": image.overlap_px,
        "overlap_phys_px": image.overlap_phys_px,
        "compression": round(image.compression, 4),
        **{flag: flag in image.flags for flag in REPORTED_FLAGS},
        "flags": ";".join(image.flags),
        "round_trip_mismatches": mismatches,
        "note": candidate.note,
    }


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: c.name)
def test_t7e_real_line_round_trip_and_contract(
    candidate: Candidate, dataset_paths: DatasetPaths, flag_rows: list
) -> None:
    root = _require_root(dataset_paths, candidate.dataset)
    radargram = candidate.load(root)
    data_before = radargram.data.copy()
    image = make_detection_image(radargram)

    mismatches = round_trip_mismatches(image, N_ROUND_TRIP_BOXES, SEED)
    flag_rows.append(_flag_row(candidate, image, mismatches))
    LOGGER.info("T7(e) %s", flag_rows[-1])

    assert mismatches == 0
    assert image.assumed_params == M3_2_ASSUMED
    assert image.v_scale_source == candidate.v_scale_source
    assert set(candidate.expected_flags) <= set(image.flags)
    for k in range(len(image.tiles)):
        assert image.tile_image(k).shape == (TILE_PX, TILE_PX)
    np.testing.assert_array_equal(radargram.data, data_before)


@pytest.mark.parametrize("file_id", ["CementRoad2_1", "AsphaltRoad2_1"])
def test_t7e_gprsd_dt_hypotheses_give_one_image(
    file_id: str, dataset_paths: DatasetPaths
) -> None:
    root = _require_root(dataset_paths, DatasetName.GPRSD)
    images = [
        make_detection_image(load_gprsd_radargram(root, file_id, hypothesis))
        for hypothesis in (DtHypothesis.A, DtHypothesis.B)
    ]
    np.testing.assert_array_equal(images[0].line_image, images[1].line_image)
    assert images[0].grid_map == images[1].grid_map
