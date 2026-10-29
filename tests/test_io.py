"""Tests for gpr_detection.io: data classes, class mapping and one sample per dataset.

Dataset tests read the read-only roots from configs/paths.yaml and are skipped
when a root is missing on this machine.
"""

from pathlib import Path

import numpy as np
import pytest

from gpr_detection.io import aihub, fourtu, gprsd, guangzhou, morocco
from gpr_detection.io.annotated import AnnotatedImage, Box
from gpr_detection.io.classes import (
    DetectionClass,
    class_names,
    map_aihub_label,
    map_morocco_class_id,
)
from gpr_detection.io.paths import DatasetName, DatasetPaths, load_dataset_paths
from gpr_detection.io.radargram import Radargram

AIHUB_IMAGE_SIDE_PX = 256
AIHUB_GAINS = {0.25, 0.5, 1.0}
AIHUB_SAMPLE_COUNT = 5
FOURTU_SAMPLES = 512
GPRSD_SAMPLES = 321
MOROCCO_SIDES_PX = {224, 268, 336}

SAMPLE_VOC_XML = """<annotation verified="yes">
  <folder>횡배관</folder>
  <filename>NM_20221017(134135)_N00_B5_g0.5_x(4432-4688)</filename>
  <source>NM_2022-10-17_(13_41_35)</source>
  <size><width>256</width><height>256</height><depth>1</depth></size>
  <gainvalue>0.5</gainvalue>
  <equipment>NM</equipment>
  <object>
    <name>tpipe</name>
    <bndbox><xmin>106.819</xmin><ymin>30.865</ymin>
      <xmax>149.217</xmax><ymax>63.546</ymax></bndbox>
  </object>
</annotation>""".encode()


# --- helpers -----------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset_paths() -> DatasetPaths:
    return load_dataset_paths()


def require_root(paths: DatasetPaths, dataset: DatasetName) -> Path:
    root = paths.root(dataset)
    if not root.is_dir():
        pytest.skip(f"{dataset.value} data not found at {root}")
    return root


def assert_boxes_inside(sample: AnnotatedImage, tolerance_px: float = 0.5) -> None:
    for box in sample.boxes:
        assert -tolerance_px <= box.x1 < box.x2 <= sample.width + tolerance_px
        assert -tolerance_px <= box.y1 < box.y2 <= sample.height + tolerance_px


def make_radargram(**overrides) -> Radargram:
    fields = {
        "data": np.zeros((4, 3), dtype=np.float32),
        "dt_ns": 0.1,
        "dx_m": 0.02,
        "air_gap": "ground",
        "amplitude_processed": False,
        "dataset": "unit",
        "line_id": "line",
    }
    fields.update(overrides)
    return Radargram(**fields)


# --- data classes and mapping (no data needed) -------------------------------


def test_class_indices_are_fixed() -> None:
    assert class_names() == {
        0: "pipe_hyperbola",
        1: "pipe_band",
        2: "cavity",
        3: "manhole",
    }


def test_aihub_labels_map_to_project_classes() -> None:
    assert map_aihub_label("tpipe") is DetectionClass.PIPE_HYPERBOLA
    assert map_aihub_label("lpipe") is DetectionClass.PIPE_BAND
    assert map_aihub_label("cavity") is DetectionClass.CAVITY
    assert map_aihub_label("manh") is DetectionClass.MANHOLE
    with pytest.raises(KeyError, match="pipe"):
        map_aihub_label("pipe")


def test_morocco_ids_map_to_project_classes() -> None:
    assert map_morocco_class_id(0) is DetectionClass.PIPE_HYPERBOLA
    assert map_morocco_class_id(1) is DetectionClass.CAVITY
    with pytest.raises(KeyError):
        map_morocco_class_id(2)


def test_radargram_physical_scale_flag() -> None:
    assert make_radargram().has_physical_scale
    assert not make_radargram(dt_ns=None).has_physical_scale
    assert not make_radargram(dx_m=None).has_physical_scale


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"data": np.zeros((4, 3), dtype=np.int16)}, TypeError),
        ({"data": np.zeros(4, dtype=np.float32)}, ValueError),
        ({"air_gap": "water"}, ValueError),
        ({"dt_ns": 0.0}, ValueError),
    ],
)
def test_radargram_rejects_invalid_fields(overrides: dict, error: type) -> None:
    with pytest.raises(error):
        make_radargram(**overrides)


def test_annotated_image_requires_2d_uint8() -> None:
    with pytest.raises(ValueError):
        AnnotatedImage(np.zeros((2, 2, 3), np.uint8), (), "unit", "x")


def test_paths_config_resolves_relative_entries(tmp_path: Path) -> None:
    config = tmp_path / "paths.yaml"
    config.write_text(
        "gpr_data_root: D:/data\n"
        "datasets:\n"
        "  aihub: a\n  morocco: m\n  fourtu: f\n  guangzhou: g\n"
        "  gprsd: E:/elsewhere/s\n",
        encoding="utf-8",
    )
    paths = load_dataset_paths(config)
    assert paths.aihub == Path("D:/data/a")
    assert paths.gprsd == Path("E:/elsewhere/s")


def test_aihub_name_helpers() -> None:
    stem = "MA_20171227(115437)_N00_B4_g0.5_x(20293-20549)"
    assert aihub.session_from_member_name(stem) == "MA_2017-12-27_(11_54_37)"
    assert aihub.gain_from_member_name(stem) == 0.5
    assert (
        aihub.crop_key_from_member_name(stem)
        == "MA_20171227(115437)_N00_B4_x(20293-20549)"
    )
    assert aihub.member_stem(f"/{stem}.jpg") == stem


def test_aihub_voc_parsing() -> None:
    boxes, fields = aihub.parse_voc_xml(SAMPLE_VOC_XML, "inline")
    assert boxes == (
        Box(DetectionClass.PIPE_HYPERBOLA, 106.819, 30.865, 149.217, 63.546, "tpipe"),
    )
    assert fields["source"] == "NM_2022-10-17_(13_41_35)"
    assert fields["width"] == "256"


# --- one sample per dataset ---------------------------------------------------


@pytest.mark.data
def test_fourtu_sample(dataset_paths: DatasetPaths) -> None:
    root = require_root(dataset_paths, DatasetName.FOURTU)
    activity_id, path = fourtu.list_fourtu_lines(root)[0]
    radargram = fourtu.load_fourtu_radargram(root, activity_id, path)

    meta = radargram.metadata
    assert radargram.data.dtype == np.float32
    assert radargram.data.shape == (FOURTU_SAMPLES, meta["n_traces"])
    assert radargram.air_gap == "air"
    assert radargram.has_physical_scale
    assert radargram.dx_m == pytest.approx(0.02)
    assert radargram.eps_r_meta == pytest.approx(meta["eps_r_used"])
    assert radargram.t0_candidates_ns == {
        "t0_sample_est": pytest.approx(meta["t0_sample_est"] * meta["dt_ns"]),
        "t0_sample_act_median": pytest.approx(
            meta["t0_sample_act_median"] * meta["dt_ns"]
        ),
    }
    only_est = fourtu.load_fourtu_radargram(
        root, activity_id, path, t0_choice=fourtu.T0Choice.EST
    )
    assert set(only_est.t0_candidates_ns) == {"t0_sample_est"}


@pytest.mark.data
def test_fourtu_groundtruth_table(dataset_paths: DatasetPaths) -> None:
    root = require_root(dataset_paths, DatasetName.FOURTU)
    table = fourtu.load_fourtu_table(root, fourtu.FourTUTable.GROUNDTRUTH)
    assert {"activity_id", "site", "distance_m", "depth_m", "code"} <= set(
        table.columns
    )
    assert table["activity_id"].iloc[0] == "01.1"  # text dtype keeps the leading zero


@pytest.mark.data
def test_guangzhou_ids_sample(dataset_paths: DatasetPaths) -> None:
    root = require_root(dataset_paths, DatasetName.GUANGZHOU)
    file_id = "pipe/2020wate_200113AA/LID10002"
    radargram = guangzhou.load_guangzhou_radargram(root, file_id)

    meta = radargram.metadata
    assert radargram.data.dtype == np.float32
    assert radargram.data.shape == (meta["n_samples"], meta["n_traces"])
    assert radargram.dt_ns == pytest.approx(meta["time_window_ns"] / meta["n_samples"])
    assert radargram.air_gap == "ground"
    assert not radargram.amplitude_processed
    assert radargram.f_c_hz == pytest.approx(400e6)
    assert radargram.t0_candidates_ns["t0_sample_used"] == pytest.approx(
        meta["t0_sample_used"] * radargram.dt_ns
    )


@pytest.mark.data
def test_guangzhou_mala_is_flagged_processed(dataset_paths: DatasetPaths) -> None:
    root = require_root(dataset_paths, DatasetName.GUANGZHOU)
    mala_ids = [
        file_id
        for file_id in guangzhou.list_guangzhou_files(root)
        if file_id.startswith("pipe/MALA_")
    ]
    radargram = guangzhou.load_guangzhou_radargram(root, mala_ids[0])
    assert radargram.amplitude_processed
    assert radargram.f_c_hz is None
    assert radargram.t0_candidates_ns["t0_sample_used"] == 0.0


@pytest.mark.data
def test_gprsd_dt_hypotheses_and_clip(dataset_paths: DatasetPaths) -> None:
    root = require_root(dataset_paths, DatasetName.GPRSD)
    file_id = gprsd.list_gprsd_files(root)[0]
    hypothesis_a = gprsd.load_gprsd_radargram(root, file_id, gprsd.DtHypothesis.A)
    hypothesis_b = gprsd.load_gprsd_radargram(
        root, file_id, gprsd.DtHypothesis.B, clip_percentile=99.0
    )

    assert hypothesis_a.data.dtype == np.float32
    assert hypothesis_a.data.shape[0] == GPRSD_SAMPLES
    assert hypothesis_a.dt_ns == pytest.approx(97.0 / 321, abs=1e-4)
    assert hypothesis_b.dt_ns == pytest.approx(97.0 / 380, abs=1e-4)
    assert hypothesis_a.amplitude_processed and hypothesis_a.air_gap == "ground"
    clip_level = hypothesis_b.metadata["clip_level"]
    assert np.abs(hypothesis_b.data).max() <= clip_level
    assert np.abs(hypothesis_a.data).max() > clip_level


@pytest.mark.data
def test_morocco_sample(dataset_paths: DatasetPaths) -> None:
    root = require_root(dataset_paths, DatasetName.MOROCCO)
    refs = morocco.list_morocco_images(root, split="train")
    labelled = next(ref for ref in refs if ref.subset == "utilities")
    sample = morocco.load_morocco_image(root, labelled)

    assert sample.image.dtype == np.uint8
    assert sample.height == sample.width and sample.width in MOROCCO_SIDES_PX
    assert sample.boxes
    assert {box.class_id for box in sample.boxes} == {DetectionClass.PIPE_HYPERBOLA}
    assert_boxes_inside(sample)
    assert sample.session == labelled.group_id


@pytest.mark.data
def test_morocco_drops_rotated_and_elastic(dataset_paths: DatasetPaths) -> None:
    root = require_root(dataset_paths, DatasetName.MOROCCO)
    all_refs = morocco.list_morocco_images(root)
    kept_refs = morocco.list_morocco_images(root, drop_rotated_and_elastic=True)

    assert len(kept_refs) < len(all_refs)
    assert not any(
        morocco.is_geometry_distorting(ref.aug_transform) for ref in kept_refs
    )
    all_transforms = {ref.aug_transform for ref in all_refs}
    kept_transforms = {ref.aug_transform for ref in kept_refs}
    dropped_transforms = all_transforms - kept_transforms
    assert dropped_transforms
    assert all(t.startswith(("회전", "탄성")) for t in dropped_transforms)


@pytest.mark.data
def test_aihub_first_samples(dataset_paths: DatasetPaths) -> None:
    root = require_root(dataset_paths, DatasetName.AIHUB)
    pairs = aihub.list_aihub_bscan_zip_pairs(root, aihub.AIHubSplit.TRAINING)
    transverse = next(pair for pair in pairs if pair.category == "비공동_횡배관")

    with aihub.AIHubZipReader(transverse) as reader:
        samples = list(reader.iter_annotated(limit=AIHUB_SAMPLE_COUNT))

    assert len(samples) == AIHUB_SAMPLE_COUNT
    for sample in samples:
        assert sample.image.shape == (AIHUB_IMAGE_SIDE_PX, AIHUB_IMAGE_SIDE_PX)
        assert sample.image.dtype == np.uint8
        assert {box.class_id for box in sample.boxes} == {DetectionClass.PIPE_HYPERBOLA}
        assert_boxes_inside(sample)
        assert sample.gain in AIHUB_GAINS
        assert "_g" not in sample.crop_key
        assert sample.session == sample.metadata["source"]
        assert sample.session_from_name is not None
