"""N2-a geometric augmentations: tests S-1..S-12, G-1..G-17 and G-10b.

Expected values are those of docs/experiments/m31_n2a_retrain_plan.md v0.2
section 5-6 (256 x 256 synthetic images, xyxy px boxes) and the v0.3 section
5-7 (다) side-limit rule (S-11, S-12, G-10b; S-12 reads the AIHub label files
only and is skipped without them). CPU only; the
pipeline tests push a few synthetic images through the real Ultralytics
``build_dataset`` path without training.
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from scipy import stats

from gpr_detection import augment_geometry
from gpr_detection.augment import AugmentConfig
from gpr_detection.augment_geometry import (
    BoxKeepRule,
    EdgeFillConfig,
    EdgeSide,
    GeometricAugmentConfig,
    HorizontalShiftConfig,
    IsotropicScaleConfig,
    apply_edge_fill,
    augment_scale,
    augment_shift_fill,
    box_areas,
    filter_boxes,
    geometric_config_from_dict,
    scale_isotropic,
    shift_horizontal,
)
from gpr_detection.detect import aug_stats, yolo_train
from gpr_detection.detect.yolo_train import (
    YoloGeometricAugment,
    YoloGprAugment,
    YoloIsotropicScale,
    build_gpr_trainer,
    check_augmentation_policy,
    write_training_data_yaml,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
M110_CONFIG = PROJECT_ROOT / "configs" / "train_yolo11m_m110.yaml"
N2A_CONFIG = PROJECT_ROOT / "configs" / "train_yolo11m_n2a.yaml"
SIZE_PX = 256
FILL = 128
RULE = BoxKeepRule()
SEED = 7
DRAWS = 10_000
CLASS_NAMES = {0: "pipe_hyperbola", 1: "pipe_band", 2: "cavity", 3: "manhole"}


def _gradient() -> np.ndarray:
    """Value = column index mod 256 (section 5-6)."""
    columns = np.arange(SIZE_PX, dtype=np.uint8)
    return np.tile(columns, (SIZE_PX, 1))


def _boxes(*rows: tuple[float, float, float, float]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def _scaled_and_kept(box, s: float):
    _, scaled, visible = scale_isotropic(_gradient(), _boxes(box), s, FILL)
    clipped, keep = filter_boxes(box_areas(scaled), scaled, visible, RULE)
    return scaled[0], clipped[0], bool(keep[0])


def _shifted_and_kept(box, dx: int):
    _, shifted, visible = shift_horizontal(_gradient(), _boxes(box), dx, FILL)
    clipped, keep = filter_boxes(box_areas(_boxes(box)), shifted, visible, RULE)
    return clipped[0], bool(keep[0])


def _band_and_kept(box, side: str, size_px: int):
    image, visible = apply_edge_fill(_gradient(), side, size_px, FILL)
    clipped, keep = filter_boxes(box_areas(_boxes(box)), _boxes(box), visible, RULE)
    return image, clipped[0], bool(keep[0])


def _off_config() -> GeometricAugmentConfig:
    return GeometricAugmentConfig(
        scale=IsotropicScaleConfig(probability=0.0),
        shift=HorizontalShiftConfig(probability=0.0),
        edge_fill=EdgeFillConfig(probability=0.0),
    )


def _always_on() -> GeometricAugmentConfig:
    return GeometricAugmentConfig(
        shift=HorizontalShiftConfig(probability=1.0),
        edge_fill=EdgeFillConfig(probability=1.0),
    )


def _yolo_labels(boxes_xyxy_px: np.ndarray, normalized_xywh: bool) -> dict:
    instance_module = pytest.importorskip("ultralytics.utils.instance")
    boxes = boxes_xyxy_px.astype(np.float32)
    if normalized_xywh:
        centre = (boxes[:, :2] + boxes[:, 2:]) / 2 / SIZE_PX
        size = (boxes[:, 2:] - boxes[:, :2]) / SIZE_PX
        boxes = np.concatenate([centre, size], axis=1)
    instances = instance_module.Instances(
        bboxes=boxes,
        bbox_format="xywh" if normalized_xywh else "xyxy",
        normalized=normalized_xywh,
    )
    classes = np.arange(len(boxes), dtype=np.float32).reshape(-1, 1)
    return {
        "img": _gradient()[..., None],
        "instances": instances,
        "cls": classes,
    }


# --------------------------------------------------------------- S: scale


def test_s1_shrink_box_offset_and_fill() -> None:
    image, scaled, visible = scale_isotropic(
        _gradient(), _boxes((100, 50, 150, 80)), 0.55, FILL
    )
    assert visible == (57, 0, 198, 141)  # new 141, off_x 57
    np.testing.assert_array_equal(
        scaled[0], [112.078125, 27.5390625, 139.6171875, 44.0625]
    )
    assert np.all(image[:, :57] == FILL)
    assert np.all(image[:, 198:] == FILL)
    assert np.all(image[141:, :] == FILL)
    assert not np.all(image[:141, 57:198] == FILL)


def test_s2_enlarge_keeps_central_box() -> None:
    scaled, clipped, kept = _scaled_and_kept((100, 50, 150, 80), 1.5)
    np.testing.assert_array_equal(scaled, [86, 75, 161, 120])  # new 384, off -64
    np.testing.assert_array_equal(clipped, [86, 75, 161, 120])
    assert kept


def test_s3_enlarge_drops_box_pushed_out_left() -> None:
    scaled, clipped, kept = _scaled_and_kept((20, 50, 50, 80), 1.5)
    np.testing.assert_array_equal(scaled, [-34, 75, 11, 120])
    np.testing.assert_array_equal(clipped, [0, 75, 11, 120])  # 11/45 = 0.244
    assert not kept


def test_s4_enlarge_keeps_box_cut_at_bottom() -> None:
    scaled, clipped, kept = _scaled_and_kept((100, 140, 150, 180), 1.5)
    np.testing.assert_array_equal(scaled, [86, 210, 161, 270])
    np.testing.assert_array_equal(clipped, [86, 210, 161, 256])  # 46/60 = 0.767
    assert kept


def test_s5_enlarge_drops_box_mostly_below() -> None:
    scaled, _, kept = _scaled_and_kept((100, 150, 150, 200), 1.5)
    np.testing.assert_array_equal(scaled, [86, 225, 161, 300])  # 31/75 = 0.413
    assert not kept


def test_s6_unit_scale_copies_without_resampling(monkeypatch) -> None:
    def no_resize(*args, **kwargs):
        raise AssertionError("cv2.resize must not run for s = 1")

    monkeypatch.setattr(augment_geometry.cv2, "resize", no_resize)
    source = _gradient()
    boxes = _boxes((100, 50, 150, 80))
    image, scaled, visible = scale_isotropic(source, boxes, 1.0, FILL)
    assert image.tobytes() == source.tobytes()
    assert image is not source
    assert scaled.tobytes() == boxes.tobytes()
    assert visible == (0, 0, SIZE_PX, SIZE_PX)


def _row0_std(image: np.ndarray, x1: int, x2: int) -> float:
    return float(image[0, x1:x2].astype(np.float64).std())


def test_s7_shrink_uses_area_interpolation_against_aliasing() -> None:
    stripes = np.zeros((SIZE_PX, SIZE_PX), dtype=np.uint8)
    stripes[:, 1::2] = 255
    image, _, (x1, _, x2, _) = scale_isotropic(stripes, _boxes(), 0.55, FILL)
    linear = cv2.resize(stripes, (141, 141), interpolation=cv2.INTER_LINEAR)
    assert _row0_std(image, x1, x2) < 20  # pre-check 11.7
    assert _row0_std(linear, 0, 141) > 60  # pre-check 73.6


def test_s8_enlarge_equals_linear_resize_with_same_placement() -> None:
    source = np.random.default_rng(0).integers(0, 256, (SIZE_PX, SIZE_PX), np.uint8)
    image, _, visible = scale_isotropic(source, _boxes(), 1.2, FILL)
    new_px = round(SIZE_PX * 1.2)  # 307
    offset_px = (SIZE_PX - new_px) // 2  # -26
    resized = cv2.resize(source, (new_px, new_px), interpolation=cv2.INTER_LINEAR)
    expected = resized[:SIZE_PX, -offset_px : -offset_px + SIZE_PX]
    assert visible == (0, 0, SIZE_PX, SIZE_PX)
    assert image.tobytes() == expected.tobytes()


def test_s9_scale_draws_are_uniform() -> None:
    rng = np.random.default_rng(SEED)
    small = np.full((8, 8), 100, dtype=np.uint8)
    config = GeometricAugmentConfig()
    draws = np.array(
        [augment_scale(small, _boxes(), config, rng)[3] for _ in range(DRAWS)]
    )
    assert stats.kstest(draws, "uniform", args=(0.55, 0.95)).pvalue > 0.01
    assert abs(np.mean(draws < 1.0) - 0.474) <= 0.02


def test_s10_double_scaling_is_refused() -> None:
    overrides = dict(_yaml(N2A_CONFIG)["ultralytics"])
    geometric = GeometricAugmentConfig()
    check_augmentation_policy(overrides, geometric)  # scale 0 is fine
    with pytest.raises(ValueError, match="double scaling"):
        check_augmentation_policy({**overrides, "scale": 0.2}, geometric)
    without_scale = {key: value for key, value in overrides.items() if key != "scale"}
    with pytest.raises(ValueError, match="double scaling"):
        check_augmentation_policy(without_scale, geometric)


def test_s11_narrow_whole_box_survives_smallest_scale() -> None:
    """v0.3 §5-7 (다): the AIHub minimum manhole width 4.79 px shrinks to ~2.6 px."""
    box = (125.0, 100.0, 129.79, 140.0)
    _, clipped, kept = _scaled_and_kept(box, 0.55)
    assert clipped[2] - clipped[0] < 4.0
    assert kept


def _aihub_label_boxes_xyxy_px(label_dir: Path) -> np.ndarray:
    """Every box of the YOLO label files (normalized xywh) as xyxy px of 256 px."""
    rows = []
    for path in sorted(label_dir.glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == len(("cls", "x", "y", "w", "h")):
                rows.append([float(value) for value in parts[1:]])
    centre_size = np.array(rows, dtype=np.float64) * SIZE_PX
    half = centre_size[:, 2:] / 2
    return np.concatenate(
        [centre_size[:, :2] - half, centre_size[:, :2] + half], axis=1
    )


@pytest.mark.data
def test_s12_no_aihub_label_is_dropped_at_smallest_scale() -> None:
    label_dir = PROJECT_ROOT / "data" / "aihub_yolo_v1" / "labels"
    if not label_dir.is_dir():
        pytest.skip(f"derived dataset absent: {label_dir}")
    boxes = _aihub_label_boxes_xyxy_px(label_dir)
    # Labels only: a blank canvas stands in for the images (AIHub crops are
    # 256 x 256), the boxes follow the same geometry. The full stage runs, so
    # label overshoot (y1 = -0.000128 px) is clipped to the frame first.
    blank = np.zeros((SIZE_PX, SIZE_PX), dtype=np.uint8)
    _, _, keep, s = augment_scale(blank, boxes, GeometricAugmentConfig(), _FixedDraw())
    assert s == 0.55
    assert len(boxes) > 0
    assert int((~keep).sum()) == 0


def test_label_overshoot_is_not_counted_as_clipping() -> None:
    """A 6.15 px box starting 0.000128 px above the frame (AIHub label end)."""
    boxes = _boxes((125.05792, -0.000128, 131.21088, 52.17088))
    blank = np.zeros((SIZE_PX, SIZE_PX), dtype=np.uint8)
    _, scaled, keep, _ = augment_scale(
        blank, boxes, GeometricAugmentConfig(), _FixedDraw()
    )
    assert scaled[0, 2] - scaled[0, 0] < 4.0
    assert keep[0]


class _FixedDraw:
    """Generator stand-in: always applies the stage and draws the minimum."""

    def random(self) -> float:
        return 0.0

    def uniform(self, low: float, high: float) -> float:
        return low


# ----------------------------------------------------------- G: shift / fill


def test_g1_shift_right() -> None:
    image, boxes, visible = shift_horizontal(
        _gradient(), _boxes((100, 50, 150, 80)), 40, FILL
    )
    np.testing.assert_array_equal(boxes[0], [140, 50, 190, 80])
    assert visible == (40, 0, SIZE_PX, SIZE_PX)
    assert np.all(image[:, :40] == FILL)
    np.testing.assert_array_equal(image[:, 40:], _gradient()[:, : SIZE_PX - 40])


def test_g2_shift_left() -> None:
    image, boxes, _ = shift_horizontal(
        _gradient(), _boxes((100, 50, 150, 80)), -40, FILL
    )
    np.testing.assert_array_equal(boxes[0], [60, 50, 110, 80])
    assert np.all(image[:, 216:] == FILL)
    np.testing.assert_array_equal(image[:, :216], _gradient()[:, 40:])


@pytest.mark.parametrize(
    ("dx", "expected", "kept"),
    [
        (120, [220, 50, 256, 80], True),  # G-3: 36/50 = 0.72
        (130, [230, 50, 256, 80], True),  # G-4: 26/50 = 0.52
        (140, [240, 50, 256, 80], False),  # G-5: 16/50 = 0.32
    ],
)
def test_g3_g5_shift_cut_at_right_edge(dx, expected, kept) -> None:
    clipped, is_kept = _shifted_and_kept((100, 50, 150, 80), dx)
    np.testing.assert_array_equal(clipped, expected)
    assert is_kept is kept


def test_g6_left_band_keeps_half_visible_box() -> None:
    image, clipped, kept = _band_and_kept((40, 50, 90, 80), "left", 64)
    assert np.all(image[:, :64] == FILL)
    np.testing.assert_array_equal(image[:, 64:], _gradient()[:, 64:])
    np.testing.assert_array_equal(clipped, [64, 50, 90, 80])  # 0.52
    assert kept


def test_g7_left_band_drops_mostly_covered_box() -> None:
    _, _, kept = _band_and_kept((30, 50, 80, 80), "left", 64)  # 16/50
    assert not kept


def test_g8_bottom_band_keeps_box_above() -> None:
    image, clipped, kept = _band_and_kept((10, 120, 60, 180), "bottom", 100)
    assert np.all(image[156:, :] == FILL)
    np.testing.assert_array_equal(image[:156], _gradient()[:156])
    np.testing.assert_array_equal(clipped, [10, 120, 60, 156])  # 36/60 = 0.6
    assert kept


def test_g9_bottom_band_drops_box_inside() -> None:
    _, _, kept = _band_and_kept((10, 170, 60, 200), "bottom", 100)
    assert not kept


def test_g10_clipped_three_pixel_fragment_is_dropped() -> None:
    # 5 px wide, 3 px visible: area share 0.6 passes, the clipped side fails.
    boxes = _boxes((-2, 50, 3, 80))
    clipped, keep = filter_boxes(
        box_areas(boxes), boxes, (0, 0, SIZE_PX, SIZE_PX), RULE
    )
    np.testing.assert_array_equal(clipped[0], [0, 50, 3, 80])
    assert not keep[0]


def test_unclipped_three_pixel_box_is_kept() -> None:  # G-10b
    """v0.3 §5-7 (다): the side limit applies to clipped boxes only."""
    boxes = _boxes((100, 50, 103, 80))
    _, keep = filter_boxes(box_areas(boxes), boxes, (0, 0, SIZE_PX, SIZE_PX), RULE)
    assert keep[0]


def test_g11_all_probabilities_zero_change_nothing() -> None:
    rng = np.random.default_rng(SEED)
    source = _gradient()
    boxes = _boxes((100, 50, 150, 80), (20, 30, 60, 90))
    config = _off_config()

    scaled, scaled_boxes, scale_mask, s = augment_scale(source, boxes, config, rng)
    shifted, shifted_boxes, shift_mask, record = augment_shift_fill(
        scaled, scaled_boxes, config, rng
    )

    assert s is None
    assert record.shift_dx_px is None and record.edge_side is None
    for image in (scaled, shifted):
        assert image.tobytes() == source.tobytes()
    for result in (scaled_boxes, shifted_boxes):
        assert result.tobytes() == boxes.tobytes()
    assert scale_mask.all() and shift_mask.all()


def _step_output(step, labels: dict) -> tuple[bytes, bytes]:
    result = step(labels)
    return result["img"].tobytes(), result["instances"].bboxes.tobytes()


@pytest.mark.parametrize("step_class", [YoloIsotropicScale, YoloGeometricAugment])
def test_g12_seed_reproducible_and_worker_seed_differs(step_class, monkeypatch) -> None:
    boxes = _boxes((100, 50, 150, 80))
    first = _step_output(step_class(_always_on(), SEED), _yolo_labels(boxes, True))
    again = _step_output(step_class(_always_on(), SEED), _yolo_labels(boxes, True))
    monkeypatch.setattr(yolo_train, "_data_loader_worker_seed", lambda: 12345)
    other_worker = _step_output(
        step_class(_always_on(), SEED), _yolo_labels(boxes, True)
    )
    assert first == again
    assert first != other_worker


def test_g13_shift_and_fill_draw_distribution() -> None:
    rng = np.random.default_rng(SEED)
    image = np.full((SIZE_PX, SIZE_PX), 100, dtype=np.uint8)
    config = GeometricAugmentConfig()
    records = [
        augment_shift_fill(image, _boxes(), config, rng)[3] for _ in range(DRAWS)
    ]
    shifts = [r.shift_dx_px for r in records if r.shift_dx_px is not None]
    bands = [r for r in records if r.edge_side is not None]

    assert abs(len(shifts) / DRAWS - 0.5) <= 0.02
    counts = np.bincount(np.asarray(shifts) + 96, minlength=193)
    assert len(counts) == 193 and min(shifts) == -96 and max(shifts) == 96
    assert stats.chisquare(counts).pvalue > 0.01
    assert abs(len(bands) / DRAWS - 0.3) <= 0.02
    for side, share in zip(EdgeSide, (0.35, 0.35, 0.30), strict=True):
        side_bands = [r for r in bands if r.edge_side is side]
        assert abs(len(side_bands) / len(bands) - share) <= 0.02
        sizes = [r.edge_size_px for r in side_bands]
        low, high = (16, 128) if side is EdgeSide.BOTTOM else (8, 96)
        assert min(sizes) == low and max(sizes) == high


@pytest.mark.parametrize("normalized_xywh", [True, False])
def test_g14_adapter_restores_format_and_filters_cls(normalized_xywh) -> None:
    left_band_64 = GeometricAugmentConfig(
        shift=HorizontalShiftConfig(probability=0.0),
        edge_fill=EdgeFillConfig(
            probability=1.0,
            side_weights=(1.0, 0.0, 0.0),
            width_min_px=64,
            width_max_px=64,
        ),
    )
    labels = _yolo_labels(_boxes((40, 50, 90, 80), (30, 50, 80, 80)), normalized_xywh)

    result = YoloGeometricAugment(left_band_64, SEED)(labels)

    instances = result["instances"]
    assert result["img"].shape == (SIZE_PX, SIZE_PX, 1)
    assert len(instances) == len(result["cls"]) == 1
    assert result["cls"].tolist() == [[0.0]]  # the G-7 box (class 1) is dropped
    assert instances.normalized is normalized_xywh
    assert instances._bboxes.format == ("xywh" if normalized_xywh else "xyxy")
    expected_xyxy = np.array([[64, 50, 90, 80]], dtype=np.float32)
    if normalized_xywh:
        centre = (expected_xyxy[:, :2] + expected_xyxy[:, 2:]) / 2 / SIZE_PX
        size = (expected_xyxy[:, 2:] - expected_xyxy[:, :2]) / SIZE_PX
        expected = np.concatenate([centre, size], axis=1)
    else:
        expected = expected_xyxy
    np.testing.assert_allclose(instances.bboxes, expected, rtol=0, atol=1e-6)


def test_g16_policy_errors() -> None:
    overrides = dict(_yaml(N2A_CONFIG)["ultralytics"])
    with pytest.raises(ValueError, match="flipud"):
        check_augmentation_policy(
            {**overrides, "flipud": 0.5}, GeometricAugmentConfig()
        )
    with pytest.raises(ValueError, match="max_shift_px"):
        HorizontalShiftConfig(max_shift_px=129)
    with pytest.raises(ValueError, match="fill"):
        EdgeFillConfig(fill=256)


@pytest.mark.parametrize("step_class", [YoloIsotropicScale, YoloGeometricAugment])
def test_g17_pickled_step_gives_same_result(step_class) -> None:
    boxes = _boxes((100, 50, 150, 80))
    fresh = step_class(_always_on(), SEED)
    restored = pickle.loads(pickle.dumps(step_class(_always_on(), SEED)))
    assert restored._generator is None
    assert _step_output(restored, _yolo_labels(boxes, True)) == _step_output(
        fresh, _yolo_labels(boxes, True)
    )


# --------------------------------------------------------------- config file


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_n2a_config_differs_from_m110_only_in_scale_and_geometric_block() -> None:
    m110 = _yaml(M110_CONFIG)
    n2a = _yaml(N2A_CONFIG)
    geometric = geometric_config_from_dict(n2a.pop("gpr_geometric"))
    assert n2a["ultralytics"].pop("scale") == 0.0
    assert m110["ultralytics"].pop("scale") == 0.2
    assert n2a == m110
    assert geometric == GeometricAugmentConfig()
    assert geometric.edge_fill.side_weights == (0.35, 0.35, 0.30)


def test_unknown_geometric_section_is_refused() -> None:
    with pytest.raises(ValueError, match="horizontal_shfit"):
        geometric_config_from_dict({"horizontal_shfit": {}})


# ------------------------------------------ G-15: real build_dataset on CPU


def _write_tiny_dataset(root: Path) -> Path:
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    rng = np.random.default_rng(0)
    for index in range(4):
        image = rng.integers(60, 120, (SIZE_PX, SIZE_PX), dtype=np.uint8)
        image[40:120, 90:170] = 230
        cv2.imwrite(str(images / f"img{index}.png"), image)
        (labels / f"img{index}.txt").write_text(
            f"{index % 4} 0.5 0.3125 0.3125 0.3125\n", encoding="utf-8"
        )
    list_file = root / "all.txt"
    list_file.write_text(
        "\n".join(str(path) for path in sorted(images.glob("*.png"))),
        encoding="utf-8",
    )
    source_yaml = root / "tiny.yaml"
    source_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "all.txt",
                "val": "all.txt",
                "names": CLASS_NAMES,
            }
        ),
        encoding="utf-8",
    )
    return source_yaml


def _trainer(tmp_path: Path, geometric: GeometricAugmentConfig | None):
    pytest.importorskip("ultralytics")
    data_yaml = write_training_data_yaml(
        _write_tiny_dataset(tmp_path / "tiny"), tmp_path / "tiny_ch1.yaml"
    )
    overrides = dict(_yaml(N2A_CONFIG if geometric else M110_CONFIG)["ultralytics"])
    overrides.update(
        model="yolo11n.yaml",
        data=str(data_yaml),
        imgsz=SIZE_PX,
        batch=4,
        workers=0,
        device="cpu",
        project=str(tmp_path / "runs"),
        name="build",
        plots=False,
        amp=False,
    )
    trainer = build_gpr_trainer(overrides, AugmentConfig(), SEED, geometric)
    # build_dataset reads the model stride; train() would build the model first.
    trainer.model = trainer.get_model(cfg="yolo11n.yaml", verbose=False)
    return trainer


def _step_names(dataset) -> list[str]:
    return [type(step).__name__ for step in dataset.transforms.transforms]


def test_g15_build_dataset_inserts_steps_in_order_and_runs(tmp_path) -> None:
    trainer = _trainer(tmp_path, _always_on())
    train_set = trainer.build_dataset(trainer.data["train"], mode="train", batch=4)
    val_set = trainer.build_dataset(trainer.data["val"], mode="val", batch=4)

    names = _step_names(train_set)
    assert names[:2] == ["YoloGprAugment", "YoloIsotropicScale"]
    assert names[-3:] == ["RandomFlip", "YoloGeometricAugment", "Format"]
    assert train_set.transforms.transforms[-3].direction == "horizontal"
    for step_class in (YoloGprAugment, YoloIsotropicScale, YoloGeometricAugment):
        assert names.count(step_class.__name__) == 1
    pre_transform = train_set.transforms.transforms[2]
    affine = [s for s in pre_transform.transforms if hasattr(s, "translate")]
    assert len(affine) == 1 and affine[0].scale == 0
    assert not {"YoloGprAugment", "YoloIsotropicScale", "YoloGeometricAugment"} & set(
        _step_names(val_set)
    )

    # A few images through the whole train pipeline (CPU, no training).
    for index in range(len(train_set)):
        sample = train_set[index]
        assert tuple(sample["img"].shape) == (1, SIZE_PX, SIZE_PX)
        assert len(sample["bboxes"]) == len(sample["cls"])
    steps = (train_set.transforms.transforms[1], train_set.transforms.transforms[-2])
    written = aug_stats.write_run_stats(
        Path(trainer.save_dir), 1, {"seed": SEED}, flushable_steps=steps
    )
    document = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert document["stages"]["scale"]["images"] == len(train_set)
    assert document["stages"]["scale"]["applied"] == len(train_set)
    assert document["stages"]["shift"]["applied"] == len(train_set)
    assert sum(document["scale_histogram"]["counts"]) == len(train_set)
    assert sum(document["edge_sides"].values()) == len(train_set)


def test_without_geometric_config_the_m110_transforms_are_unchanged(tmp_path) -> None:
    trainer = _trainer(tmp_path, None)
    train_set = trainer.build_dataset(trainer.data["train"], mode="train", batch=4)
    names = _step_names(train_set)
    assert names[0] == "YoloGprAugment"
    assert "YoloIsotropicScale" not in names
    assert "YoloGeometricAugment" not in names
    assert not (Path(trainer.save_dir) / aug_stats.WORKER_STATS_DIR_NAME).exists()
