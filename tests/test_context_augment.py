"""N2-a' context scale: tests C-1..C-25 and R-1; N2-a″ rule K32: K-1..K-10.

Expected values are those of docs/experiments/m31_n2a2_retrain_plan.md v1.0
section 7-2 (C-22..C-25 and the C-21 definition from section 7-5) and of
docs/experiments/m31_n2a3_retrain_plan.md v1.0 §3·§7-0 (K-1..K-10).
Synthetic crops are vertical stripes whose value is the original column
``x mod 251``, so a strip column tells which original column it shows.
CPU only; C-20 pushes a few synthetic crops through the real Ultralytics
``build_dataset`` path without training, C-21 reads the real context index
(skipped without ``data/``).
"""

import csv
import dataclasses
import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest
import yaml
from scipy import stats

from gpr_detection import augment_geometry
from gpr_detection.augment import AugmentConfig
from gpr_detection.augment_geometry import (
    BoxKeepRule,
    ContextCrop,
    ContextIndex,
    ContextMode,
    ContextScaleConfig,
    ContextSource,
    CropContent,
    GeometricAugmentConfig,
    RejectReason,
    StripPiece,
    assemble_context_strip,
    augment_context_scale,
    augment_scale,
    check_neighbor_admissible,
    context_config_from_dict,
    crop_only_strip,
    drop_short_clipped_bands,
    fit_brightness,
    geometric_config_from_dict,
    merge_strip_labels,
    place_window,
    scale_isotropic,
)
from gpr_detection.detect import aug_stats
from gpr_detection.detect.yolo_train import (
    YoloContextScale,
    build_gpr_trainer,
    check_augmentation_policy,
    context_index_for_data_yaml,
    write_training_data_yaml,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N2A_CONFIG = PROJECT_ROOT / "configs" / "train_yolo11m_n2a.yaml"
N2A2_CONFIG = PROJECT_ROOT / "configs" / "train_yolo11m_n2a2.yaml"
N2A3_CONFIG = PROJECT_ROOT / "configs" / "train_yolo11m_n2a3.yaml"
R1_SCRIPT = PROJECT_ROOT / "analysis" / "detection" / "m31_r1_overlap_shift.py"
REAL_INDEX_DIR = PROJECT_ROOT / "data" / "aihub_context_index"
REAL_DATASET_DIR = PROJECT_ROOT / "data" / "aihub_yolo_v1"
SIZE_PX = 256
STRIPE_PERIOD = 251
FILL = 128
SEED = 7
C18_SEED = 2
DRAWS = 10_000
LINE = "DS_20170101(000000)_N00_B1"
CLASS_NAMES = {0: "pipe_hyperbola", 1: "pipe_band", 2: "cavity", 3: "manhole"}
HYPERBOLA, BAND, CAVITY = 0, 1, 2
CONFIG = ContextScaleConfig()


# ---------------------------------------------------------------- helpers


def _stripes(x1_px: int, width_px: int = SIZE_PX) -> np.ndarray:
    """Crop of original columns ``[x1, x1 + width)``: value = x mod 251."""
    columns = (np.arange(x1_px, x1_px + width_px) % STRIPE_PERIOD).astype(np.uint8)
    return np.tile(columns, (SIZE_PX, 1))


def _name(x1_px: int, gain: str = "1.0", line: str = LINE) -> str:
    return f"{line}_g{gain}_x({x1_px}-{x1_px + SIZE_PX}).jpg"


def _crop(
    x1_px: int,
    fold: int = 1,
    subset: str = "trainpool",
    gain: str = "1.0",
    line: str = LINE,
) -> ContextCrop:
    return ContextCrop(
        file_name=_name(x1_px, gain, line),
        line=line,
        gain=gain,
        x1_px=x1_px,
        x2_px=x1_px + SIZE_PX,
        subset=subset,
        fold=fold,
        primary_class="pipe_band",
        crop_key=f"{line}_x({x1_px}-{x1_px + SIZE_PX})",
    )


def _index(
    crops: list[ContextCrop], validation: list[ContextCrop] = ()
) -> ContextIndex:
    train = [crop.file_name for crop in crops if crop.subset == "trainpool"]
    train = [name for name in train if name not in {c.file_name for c in validation}]
    return ContextIndex(
        [*crops, *validation], train, [crop.file_name for crop in validation]
    )


class _MemoryLoader:
    """Crop loader over in-memory images and labels."""

    def __init__(self) -> None:
        self.images: dict[str, np.ndarray] = {}
        self.boxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.loaded: list[str] = []

    def add(self, crop: ContextCrop, image: np.ndarray, boxes=(), classes=()) -> None:
        self.images[crop.file_name] = image
        self.boxes[crop.file_name] = (
            np.array(boxes, dtype=np.float64).reshape(-1, 4),
            np.array(classes, dtype=np.int64),
        )

    def image(self, file_name: str) -> np.ndarray:
        self.loaded.append(file_name)
        return self.images[file_name]

    def labels(self, file_name: str) -> tuple[np.ndarray, np.ndarray]:
        return self.boxes[file_name]


def _content(crop: ContextCrop, image=None, boxes=(), classes=()) -> CropContent:
    return CropContent(
        _stripes(crop.x1_px) if image is None else image,
        np.array(boxes, dtype=np.float64).reshape(-1, 4),
        np.array(classes, dtype=np.int64),
    )


def _strip(center: ContextCrop, neighbors: list[tuple[ContextCrop, np.ndarray]], **kw):
    loader = _MemoryLoader()
    for crop, image in neighbors:
        loader.add(crop, image)
    index = _index([center, *(crop for crop, _ in neighbors)])
    config = ContextScaleConfig(**kw)
    return assemble_context_strip(
        center.file_name, _content(center), index, loader, config
    )


class _FixedDraw:
    """Generator stand-in with fixed draws."""

    def __init__(self, mode_draw: float, s: float, dx: int = 0) -> None:
        self.mode_draw = mode_draw
        self.s = s
        self.dx = dx

    def random(self) -> float:
        return self.mode_draw

    def uniform(self, low: float, high: float) -> float:
        return self.s

    def integers(self, low: int, high: int, endpoint: bool = False) -> int:
        return self.dx


# ------------------------------------------------------- C-1..C-7: strip


def test_c1_strip_columns_are_original_columns() -> None:
    center, neighbor = _crop(1000), _crop(1104)
    strip = _strip(center, [(neighbor, _stripes(1104))])
    assert strip.x0_px == 1000 and strip.image.shape == (SIZE_PX, 360)
    expected = (np.arange(1000, 1360) % STRIPE_PERIOD).astype(np.uint8)
    np.testing.assert_array_equal(strip.image[0], expected)
    assert strip.image[:, :SIZE_PX].tobytes() == _stripes(1000).tobytes()
    assert strip.crop_left_px == 0 and strip.used_neighbors == 1


def test_c2_brightness_fit_and_correction() -> None:
    original = _stripes(1104).astype(np.float64)
    fit = fit_brightness(original[:, :152], 0.9 * original[:, :152] + 10)
    assert fit.ok
    assert abs(fit.alpha - 1 / 0.9) < 1e-3
    assert abs(fit.beta - (-10 / 0.9)) < 1e-3

    darker = np.clip(np.rint(0.9 * original + 10), 0, 255).astype(np.uint8)
    center, neighbor = _crop(1000), _crop(1104)
    strip = _strip(center, [(neighbor, darker)])
    assert strip.image.shape[1] == 360
    rounded = fit_brightness(_stripes(1000)[:, 104:], darker[:, :152])
    corrected = augment_geometry.apply_brightness(darker[:, :152], rounded)
    difference = corrected.astype(int) - _stripes(1000)[:, 104:].astype(int)
    assert np.mean(np.abs(difference)) < 1
    expected_right = (np.arange(1256, 1360) % STRIPE_PERIOD).astype(int)
    assert np.mean(np.abs(strip.image[0, 256:].astype(int) - expected_right)) < 1


def _noisy_overlap_neighbor(target_corr: float) -> tuple[np.ndarray, float]:
    """Neighbour 1104 with noise on the overlap columns giving ``target_corr``."""
    image = _stripes(1104).astype(np.float64)
    overlap = image[:, :152]
    noise_sigma = overlap.std() * np.sqrt(1 / target_corr**2 - 1)
    noise = np.random.default_rng(0).normal(0.0, noise_sigma, overlap.shape)
    image[:, :152] = overlap + noise
    noisy = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    corr = fit_brightness(_stripes(1000)[:, 104:], noisy[:, :152]).corr
    return noisy, corr


def test_c3_low_correlation_drops_neighbor() -> None:
    noisy, corr = _noisy_overlap_neighbor(0.95)
    assert 0.94 <= corr <= 0.96
    strip = _strip(_crop(1000), [(_crop(1104), noisy)])
    assert strip.image.shape[1] == SIZE_PX and strip.used_neighbors == 0
    assert strip.rejections == (RejectReason.CORRELATION,)


def test_c4_alpha_out_of_range_drops_neighbor() -> None:
    doubled = np.clip(2 * _stripes(1104).astype(int), 0, 255).astype(np.uint8)
    strip = _strip(_crop(1000), [(_crop(1104), doubled)])
    assert strip.used_neighbors == 0 and strip.image.shape[1] == SIZE_PX
    assert len(strip.rejections) == 1
    exact = fit_brightness(_stripes(1000)[:, 104:], 2.0 * _stripes(1104)[:, :152])
    assert not exact.ok and abs(exact.alpha - 0.5) < 1e-12 and exact.corr > 0.999


def test_c5_touching_crop_is_not_stitched() -> None:
    strip = _strip(_crop(1000), [(_crop(1256), _stripes(1256))])
    assert strip.used_neighbors == 0 and strip.image.shape[1] == SIZE_PX
    assert strip.rejections == (RejectReason.OVERLAP,)


def test_c6_ten_pixel_overlap_is_not_stitched() -> None:
    strip = _strip(_crop(1000), [(_crop(1246), _stripes(1246))])
    assert strip.used_neighbors == 0
    assert strip.rejections == (RejectReason.OVERLAP,)


def test_c7_chain_is_cut_at_e_and_fitted_to_previous_link(monkeypatch) -> None:
    fitted_overlaps = []
    original_fit = augment_geometry.fit_brightness

    def recording_fit(reference, neighbor, *args):
        fitted_overlaps.append((reference.copy(), neighbor.shape))
        return original_fit(reference, neighbor, *args)

    monkeypatch.setattr(augment_geometry, "fit_brightness", recording_fit)
    second, third = _crop(1104), _crop(1208)
    strip = _strip(_crop(1000), [(second, _stripes(1104)), (third, _stripes(1208))])
    assert strip.x0_px == 1000 and strip.image.shape[1] == 1384 - 1000
    expected = (np.arange(1000, 1384) % STRIPE_PERIOD).astype(np.uint8)
    np.testing.assert_array_equal(strip.image[0], expected)
    assert strip.used_neighbors == 2
    # The third crop is fitted on its 152 px overlap with the second (not 48 with
    # the centre), against the second crop's columns 1208-1360.
    assert [shape for _, shape in fitted_overlaps] == [(SIZE_PX, 152), (SIZE_PX, 152)]
    np.testing.assert_array_equal(fitted_overlaps[1][0], _stripes(1104)[:, 104:256])


# ------------------------------------------------------ C-8..C-11: leaks


def test_c8_neighbor_in_other_fold_raises() -> None:
    center, other_fold = _crop(1000, fold=1), _crop(1104, fold=2)
    with pytest.raises(ValueError, match="fold 2 != centre fold 1"):
        check_neighbor_admissible(center, other_fold, validation_folds=())
    with pytest.raises(ValueError, match="span several"):
        _index([center, other_fold])


def test_c9_test_subset_neighbor_raises() -> None:
    center, test_crop = _crop(1000), _crop(1104, fold=-1, subset="test")
    with pytest.raises(ValueError, match="subset 'test'"):
        check_neighbor_admissible(center, test_crop, validation_folds=())
    with pytest.raises(ValueError, match="span several"):
        _index([center, test_crop])


def test_c10_validation_fold_neighbor_raises() -> None:
    center, neighbor = _crop(1000, fold=1), _crop(1104, fold=1)
    with pytest.raises(ValueError, match="validation fold"):
        check_neighbor_admissible(center, neighbor, validation_folds={1})
    with pytest.raises(ValueError, match="train and validation"):
        ContextIndex([center, neighbor], [center.file_name], [neighbor.file_name])
    # The per-neighbour assertion also fires inside the lookup itself.
    index = _index([center, neighbor])
    index.validation_folds = frozenset({1})
    with pytest.raises(ValueError, match="context leak"):
        index.neighbors(center, CONFIG.extend_px)


def test_c11_other_gain_is_not_a_candidate() -> None:
    center, other_gain = _crop(1000, gain="1.0"), _crop(1104, gain="0.5")
    index = _index([center, other_gain])
    assert index.neighbors(center, CONFIG.extend_px) == []
    strip = _strip(center, [(other_gain, _stripes(1104))])
    assert strip.used_neighbors == 0 and strip.image.shape[1] == SIZE_PX


def test_train_image_missing_from_index_raises() -> None:
    with pytest.raises(ValueError, match="not in the index"):
        ContextIndex([_crop(1000)], [_crop(1000).file_name, "unknown.jpg"])


# ---------------------------------------------------- C-12, C-13: labels


def _piece(x1_px: int, owned: tuple[int, int], boxes, classes) -> StripPiece:
    return StripPiece(
        file_name=_name(x1_px),
        x1_px=x1_px,
        x2_px=x1_px + SIZE_PX,
        owned_x1_px=owned[0],
        owned_x2_px=owned[1],
        boxes=np.array(boxes, dtype=np.float64).reshape(-1, 4),
        classes=np.array(classes, dtype=np.int64),
    )


def test_c12_same_hyperbola_in_two_crops_is_kept_once() -> None:
    center = _piece(1000, (1000, 1256), [(1108, 50, 1148, 90)], [HYPERBOLA])
    neighbor = _piece(1104, (1256, 1360), [(1110, 50, 1150, 90)], [HYPERBOLA])
    boxes, classes = merge_strip_labels([center, neighbor], center_index=0)
    np.testing.assert_array_equal(boxes, [[1108, 50, 1148, 90]])
    assert classes.tolist() == [HYPERBOLA]


def test_c22_centre_box_is_kept_whatever_the_cells() -> None:
    """v0.3: a centre box near the crop end stays though the neighbour lacks it."""
    center = _piece(1000, (1000, 1256), [(1244, 50, 1256, 80)], [HYPERBOLA])
    neighbor = _piece(1104, (1256, 1360), [], [])
    boxes, classes = merge_strip_labels([center, neighbor], center_index=0)
    np.testing.assert_array_equal(boxes, [[1244, 50, 1256, 80]])
    assert classes.tolist() == [HYPERBOLA]


def test_c23_cut_centre_box_is_replaced_by_the_whole_neighbour_box() -> None:
    """Rule (가): IoMin 1.0, area ratio 2.0 -> the larger neighbour box."""
    center = _piece(1000, (1000, 1256), [(1236, 50, 1256, 80)], [HYPERBOLA])
    neighbor = _piece(1104, (1256, 1360), [(1236, 50, 1276, 80)], [HYPERBOLA])
    boxes, classes = merge_strip_labels([center, neighbor], center_index=0)
    np.testing.assert_array_equal(boxes, [[1236, 50, 1276, 80]])
    assert classes.tolist() == [HYPERBOLA]


def test_c24_neighbour_box_inside_centre_columns_is_dropped() -> None:
    center = _piece(1000, (1000, 1256), [], [])
    neighbor = _piece(1104, (1256, 1360), [(1180, 50, 1220, 90)], [CAVITY])
    boxes, _ = merge_strip_labels([center, neighbor], center_index=0)
    assert len(boxes) == 0


def test_neighbour_boxes_are_split_by_neighbour_cells() -> None:
    """Two right neighbours (centres 1232, 1336): border 1284."""
    center = _piece(1000, (1000, 1256), [], [])
    near = _piece(1104, (1256, 1360), [(1290, 50, 1310, 70)], [CAVITY])
    far = _piece(1208, (1360, 1384), [(1292, 50, 1312, 70)], [CAVITY])
    boxes, _ = merge_strip_labels([center, near, far], center_index=0)
    np.testing.assert_array_equal(boxes, [[1292, 50, 1312, 70]])  # far's cell


def test_duplicate_rules_order() -> None:
    kept = [np.array([100.0, 50, 140, 90])]
    # (나) similar size (ratio 1), IoU 0.90 -> the kept (centre or nearer) stays.
    assert augment_geometry.resolve_duplicate(np.array([102.0, 50, 142, 90]), kept) == (
        False,
        [],
    )
    # (가) a larger box containing it replaces it.
    larger = np.array([95.0, 45, 150, 95])
    assert augment_geometry.resolve_duplicate(larger, kept) == (True, [0])
    # (가) a smaller box inside a kept larger one goes.
    smaller = np.array([110.0, 60, 125, 80])
    assert augment_geometry.resolve_duplicate(smaller, kept) == (False, [])
    # Unrelated box stays.
    assert augment_geometry.resolve_duplicate(np.array([200.0, 50, 240, 90]), kept) == (
        True,
        [],
    )


def test_c13_band_inside_centre_columns_comes_from_centre_only() -> None:
    center = _piece(1000, (1000, 1256), [(1050, 70, 1200, 86)], [BAND])
    neighbor = _piece(1104, (1256, 1360), [(1150, 72, 1340, 88)], [BAND])
    boxes, classes = merge_strip_labels([center, neighbor], center_index=0)
    np.testing.assert_array_equal(boxes, [[1050, 70, 1200, 86], [1256, 72, 1340, 88]])
    assert classes.tolist() == [BAND, BAND]


def test_c12_c13_through_the_strip_in_strip_coordinates() -> None:
    center, neighbor = _crop(1000), _crop(1104)
    loader = _MemoryLoader()
    # Neighbour labels in its own crop px: 1110 - 1104 = 6, 1150 - 1104 = 46.
    loader.add(
        neighbor,
        _stripes(1104),
        [(6, 50, 46, 90), (46, 72, 236, 88)],
        [HYPERBOLA, BAND],
    )
    content = _content(
        center, boxes=[(108, 50, 148, 90), (50, 70, 200, 86)], classes=[HYPERBOLA, BAND]
    )
    strip = assemble_context_strip(
        center.file_name, content, _index([center, neighbor]), loader, CONFIG
    )
    np.testing.assert_array_equal(
        strip.boxes, [[108, 50, 148, 90], [50, 70, 200, 86], [256, 72, 340, 88]]
    )
    assert strip.classes.tolist() == [HYPERBOLA, BAND, BAND]


# -------------------------------------------------- C-14..C-17: window


def test_c14_plain_mode_equals_n2a_scale() -> None:
    strip = crop_only_strip(
        _content(_crop(1000), boxes=[(100, 50, 150, 80)], classes=[0])
    )
    image, boxes, keep, record = place_window(
        strip.image, strip.boxes, 0.55, "plain", np.random.default_rng(0)
    )
    np.testing.assert_array_equal(
        boxes[0], [112.078125, 27.5390625, 139.6171875, 44.0625]
    )
    assert keep.tolist() == [True]
    assert record.dx_px == 0 and record.visible_rect == (57, 0, 198, 141)
    reference, _, _ = scale_isotropic(strip.image, strip.boxes, 0.55, FILL)
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("s", [0.55, 0.8, 1.0, 1.2, 1.5])
def test_c14_plain_stage_is_byte_equal_to_n2a_augment_scale(s) -> None:
    source = np.random.default_rng(1).integers(0, 256, (SIZE_PX, SIZE_PX), np.uint8)
    boxes = np.array([[100, 50, 150, 80], [-0.0001, 10, 30, 60]], dtype=np.float64)
    crop = _crop(1000)
    image, new_boxes, classes, keep, record = augment_context_scale(
        crop.file_name,
        CropContent(source, boxes, np.array([0, 3])),
        _index([crop]),
        _MemoryLoader(),
        CONFIG,
        BoxKeepRule(),
        _FixedDraw(mode_draw=0.99, s=s),
    )
    n2a = augment_scale(source, boxes, GeometricAugmentConfig(), _FixedDraw(0.0, s))
    assert (
        record.window.mode is ContextMode.PLAIN and record.source is ContextSource.NONE
    )
    assert image.tobytes() == n2a[0].tobytes()
    assert new_boxes.tobytes() == n2a[1].tobytes()
    assert keep.tobytes() == n2a[2].tobytes()
    assert classes.tolist() == [0, 3]


def _context_draws(strip_image: np.ndarray, s: float, crop_left_px: int = 0):
    rng = np.random.default_rng(SEED)
    return [
        place_window(strip_image, np.zeros((0, 4)), s, "context", rng, crop_left_px)[3]
        for _ in range(DRAWS)
    ]


def test_c15_context_shift_range_and_uniformity() -> None:
    records = _context_draws(_stripes(1000), 1.25)
    shifts = np.array([record.dx_px for record in records])
    assert {(r.dx_min_px, r.dx_max_px) for r in records} == {(-32, 32)}
    assert shifts.min() >= -32 and shifts.max() <= 32
    counts = np.bincount(shifts + 32, minlength=65)
    assert stats.chisquare(counts).pvalue > 0.01
    assert {record.visible_rect for record in records} == {(0, 0, SIZE_PX, SIZE_PX)}


def test_c16_context_unit_scale_without_neighbor_is_identity() -> None:
    source = _stripes(1000)
    image, _, _, record = place_window(
        source, np.zeros((0, 4)), 1.0, "context", np.random.default_rng(0)
    )
    assert record.dx_px == 0 and (record.dx_min_px, record.dx_max_px) == (0, 0)
    assert image.tobytes() == source.tobytes()


def test_c17_wide_strip_moves_objects_left_only() -> None:
    strip = _strip(_crop(1000), [(_crop(1104), _stripes(1104))])
    records = _context_draws(strip.image, 1.0, strip.crop_left_px)
    assert {(r.dx_min_px, r.dx_max_px) for r in records} == {(-96, 0)}
    assert {record.visible_rect for record in records} == {(0, 0, SIZE_PX, SIZE_PX)}
    image, boxes, _, _ = place_window(
        strip.image,
        np.array([[100.0, 50, 150, 80]]),
        1.0,
        "context",
        _FixedDraw(0.0, 1.0, dx=-40),
    )
    np.testing.assert_array_equal(boxes[0], [60, 50, 110, 80])  # moved left by 40
    np.testing.assert_array_equal(image, strip.image[:, 40:296])


# --------------------------------------------------- C-18: mode and s


def test_c18_mode_share_and_scale_distributions() -> None:
    crop = _crop(1000)
    index = _index([crop])
    # Seed 7 gives a context-mode KS p of 0.008 (seeds 0-11: p uniform over
    # [0.008, 0.98], i.e. no bias); a p > 0.01 rule fails 1 seed in 100 by design.
    rng = np.random.default_rng(C18_SEED)
    small = _content(crop)
    records = [
        augment_context_scale(
            crop.file_name, small, index, _MemoryLoader(), CONFIG, BoxKeepRule(), rng
        )[4]
        for _ in range(DRAWS)
    ]
    context = [r.window.s for r in records if r.window.mode is ContextMode.CONTEXT]
    plain = [r.window.s for r in records if r.window.mode is ContextMode.PLAIN]
    assert abs(len(context) / DRAWS - 0.5) <= 0.02
    assert stats.kstest(context, "uniform", args=(1.0, 0.5)).pvalue > 0.01
    assert stats.kstest(plain, "uniform", args=(0.55, 0.95)).pvalue > 0.01
    assert {r.source for r in records} == {ContextSource.NONE, ContextSource.C1}


# --------------------------------------------- C-19, C-20: config, steps


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_c19_context_and_geometric_scale_together_raise() -> None:
    n2a2 = _yaml(N2A2_CONFIG)
    context = context_config_from_dict(n2a2["gpr_context"])
    with_scale = {**n2a2["gpr_geometric"], "scale": {"probability": 1.0}}
    with pytest.raises(ValueError, match="gpr_context"):
        geometric_config_from_dict(with_scale, scale_owned_by_context=True)
    overrides = dict(n2a2["ultralytics"])
    with pytest.raises(ValueError, match="double scaling"):
        check_augmentation_policy(overrides, GeometricAugmentConfig(), context)
    geometric = geometric_config_from_dict(
        n2a2["gpr_geometric"], scale_owned_by_context=True
    )
    assert geometric.scale.probability == 0
    check_augmentation_policy(overrides, geometric, context)  # fine
    with pytest.raises(ValueError, match="gpr_context is on"):
        check_augmentation_policy({**overrides, "scale": 0.2}, geometric, context)


def test_n2a2_config_differs_from_n2a_only_as_planned() -> None:
    n2a, n2a2 = _yaml(N2A_CONFIG), _yaml(N2A2_CONFIG)
    context = context_config_from_dict(n2a2.pop("gpr_context"))
    assert context == ContextScaleConfig()
    geometric_n2a = n2a.pop("gpr_geometric")
    geometric_n2a2 = n2a2.pop("gpr_geometric")
    assert n2a == n2a2
    assert "scale" not in geometric_n2a2
    assert geometric_n2a2["horizontal_shift"]["probability"] == 0.15
    assert geometric_n2a2["edge_fill"]["probability"] == 0.15
    assert geometric_n2a2["edge_fill"]["side_weights"] == [0.25, 0.25, 0.5]
    for section in ("horizontal_shift", "edge_fill"):
        for key in set(geometric_n2a[section]) - {"probability", "side_weights"}:
            assert geometric_n2a2[section][key] == geometric_n2a[section][key]
    assert geometric_n2a2["keep_rule"] == geometric_n2a["keep_rule"]


def test_unknown_context_key_is_refused() -> None:
    with pytest.raises(ValueError, match="p_contxt"):
        context_config_from_dict({"p_contxt": 0.5})


CHAIN_STARTS = (1000, 1104, 1208)
VAL_START = 5000


CROPS_CSV_COLUMNS = (
    "file_name",
    "crop_key",
    "line",
    "gain",
    "a",
    "b",
    "primary_class",
    "subset",
    "fold",
)


def _write_crops_csv(path: Path, crops: list[ContextCrop]) -> None:
    """The ``crops.csv`` columns the context index and R1 read."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CROPS_CSV_COLUMNS)
        for crop in crops:
            writer.writerow(
                [
                    crop.file_name,
                    crop.crop_key,
                    crop.line,
                    crop.gain,
                    crop.x1_px,
                    crop.x2_px,
                    crop.primary_class,
                    crop.subset,
                    crop.fold,
                ]
            )


DEFAULT_LABEL_TEXT = "0 0.5 0.3125 0.15625 0.15625\n1 0.5 0.6 0.8 0.0625\n"
"""One centred hyperbola and one band per crop (normalised xywh)."""


def _write_context_dataset(
    root: Path, label_text_by_x1: dict[int, str] | None = None
) -> tuple[Path, Path]:
    """Three overlapping crops of one line (train) and one other line (val).

    ``label_text_by_x1`` replaces the label txt of the crops starting there.
    """
    images, labels, index_dir = root / "images", root / "labels", root / "index"
    for folder in (images, labels, index_dir):
        folder.mkdir(parents=True)
    rows = []
    for x1_px in CHAIN_STARTS:
        rows.append(_crop(x1_px, fold=1))
    rows.append(_crop(VAL_START, fold=0, line=LINE.replace("B1", "B2")))
    special_labels = label_text_by_x1 or {}
    for crop in rows:
        cv2.imwrite(str(images / crop.file_name), _stripes(crop.x1_px))
        stem = Path(crop.file_name).stem
        label_text = special_labels.get(crop.x1_px, DEFAULT_LABEL_TEXT)
        (labels / f"{stem}.txt").write_text(label_text, encoding="utf-8")
    _write_crops_csv(index_dir / "crops.csv", rows)
    (index_dir / "groups.csv").write_text("group_id\n0\n", encoding="utf-8")
    (root / "train.txt").write_text(
        "\n".join(f"./images/{crop.file_name}" for crop in rows[:3]), encoding="utf-8"
    )
    (root / "val.txt").write_text(f"./images/{rows[3].file_name}", encoding="utf-8")
    source_yaml = root / "fold.yaml"
    source_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": ["train.txt"],
                "val": "val.txt",
                "names": CLASS_NAMES,
            }
        ),
        encoding="utf-8",
    )
    return source_yaml, index_dir


def _context_trainer(tmp_path: Path, context: ContextScaleConfig):
    pytest.importorskip("ultralytics")
    source_yaml, index_dir = _write_context_dataset(tmp_path / "ctx")
    data_yaml = write_training_data_yaml(source_yaml, tmp_path / "ctx_ch1.yaml")
    index = context_index_for_data_yaml(index_dir, data_yaml)
    n2a2 = _yaml(N2A2_CONFIG)
    geometric = geometric_config_from_dict(
        n2a2["gpr_geometric"], scale_owned_by_context=True
    )
    overrides = dict(n2a2["ultralytics"])
    overrides.update(
        model="yolo11n.yaml",
        data=str(data_yaml),
        imgsz=SIZE_PX,
        batch=3,
        workers=0,
        device="cpu",
        project=str(tmp_path / "runs"),
        name="ctx",
        plots=False,
        amp=False,
    )
    # Strength 0 as in the N2-a' config: GPR noise on the centre crop would
    # lower its overlap correlation with the (raw) neighbours below corr_min.
    no_gpr_noise = AugmentConfig().scaled(0.0)
    trainer = build_gpr_trainer(
        overrides, no_gpr_noise, SEED, geometric, context, index
    )
    trainer.model = trainer.get_model(cfg="yolo11n.yaml", verbose=False)
    return trainer, index


def _step_names(dataset) -> list[str]:
    return [type(step).__name__ for step in dataset.transforms.transforms]


def test_c20_context_step_at_index_1_and_runs(tmp_path) -> None:
    always_context = ContextScaleConfig(p_context=1.0)
    trainer, index = _context_trainer(tmp_path, always_context)
    assert index.train_folds == {1} and index.validation_folds == {0}
    train_set = trainer.build_dataset(trainer.data["train"], mode="train", batch=3)
    names = _step_names(train_set)
    assert names[:2] == ["YoloGprAugment", "YoloContextScale"]
    assert names[-3:] == ["RandomFlip", "YoloGeometricAugment", "Format"]
    assert names.count("YoloContextScale") == 1
    assert "YoloIsotropicScale" not in names
    affine = [
        step
        for step in train_set.transforms.transforms[2].transforms
        if hasattr(step, "translate")
    ]
    assert len(affine) == 1 and affine[0].scale == 0

    for position in range(len(train_set)):
        sample = train_set[position]
        assert tuple(sample["img"].shape) == (1, SIZE_PX, SIZE_PX)
        assert len(sample["bboxes"]) == len(sample["cls"])
    steps = (train_set.transforms.transforms[1], train_set.transforms.transforms[-2])
    written = aug_stats.write_run_stats(
        Path(trainer.save_dir), 1, {"seed": SEED}, flushable_steps=steps
    )
    document = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert document["stages"]["context_scale"]["images"] == len(train_set)
    assert document["counters"]["mode"] == {"context": len(train_set)}
    assert document["counters"]["source"] == {"a": len(train_set)}
    assert sum(document["histograms"]["s_context"]["counts"]) == len(train_set)


def _context_labels(root: Path, x1_px: int) -> dict:
    instance_module = pytest.importorskip("ultralytics.utils.instance")
    boxes = np.array(
        [[0.5, 0.3125, 0.15625, 0.15625], [0.5, 0.6, 0.8, 0.0625]], np.float32
    )
    return {
        "img": _stripes(x1_px)[..., None],
        "instances": instance_module.Instances(
            bboxes=boxes, bbox_format="xywh", normalized=True
        ),
        "cls": np.array([[0.0], [1.0]], dtype=np.float32),
        "im_file": str(root / "images" / _name(x1_px)),
    }


def test_c20_pickled_step_gives_same_result(tmp_path) -> None:
    source_yaml, index_dir = _write_context_dataset(tmp_path / "ctx")
    index = context_index_for_data_yaml(index_dir, source_yaml)
    config = ContextScaleConfig(p_context=1.0)
    fresh = YoloContextScale(config, index, SEED)
    restored = pickle.loads(pickle.dumps(YoloContextScale(config, index, SEED)))
    assert restored._generator is None
    results = []
    for step in (fresh, restored):
        labels = step(_context_labels(tmp_path / "ctx", 1104))
        results.append((labels["img"].tobytes(), labels["instances"].bboxes.tobytes()))
    assert results[0] == results[1]
    assert len(labels["cls"]) == len(labels["instances"]) >= 1


def _n2a2_config_copy(tmp_path: Path, index_dir: Path, scale_on: bool) -> Path:
    config = _yaml(N2A2_CONFIG)
    config["gpr_context"]["index_dir"] = str(index_dir)
    if scale_on:
        config["gpr_geometric"]["scale"] = {"probability": 1.0}
    path = tmp_path / ("n2a2_scale_on.yaml" if scale_on else "n2a2.yaml")
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_train_detector_passes_and_records_the_context(load_script, tmp_path):
    module = load_script("train_detector")
    runs_dir = tmp_path / "runs" / "detect"
    # A fresh module per test (load_script), so plain assignment is enough.
    module.RUNS_DIR = runs_dir
    module.DATA_YAML_DIR = runs_dir / "_data"
    calls = []

    class _Trainer:
        def __init__(self, save_dir: Path) -> None:
            self.save_dir = save_dir

        def train(self) -> None:
            return None

    def fake_build_trainer(overrides, augment_config, augment_seed, **kwargs):
        calls.append(kwargs)
        return _Trainer(Path(overrides["project"]) / overrides["name"])

    module.build_gpr_trainer = fake_build_trainer
    source_yaml, index_dir = _write_context_dataset(tmp_path / "ctx")
    config = _n2a2_config_copy(tmp_path, index_dir, scale_on=False)
    arguments = ["--config", str(config), "--data", str(source_yaml), "--name", "ctx"]
    assert module.main(arguments) == 0

    kwargs = calls[0]
    assert kwargs["geometric_config"].scale.probability == 0
    assert kwargs["context_config"] == ContextScaleConfig(index_dir=str(index_dir))
    assert kwargs["context_index"].train_folds == {1}
    settings = _yaml(runs_dir / "ctx" / "gpr_train_settings.yaml")
    block = _yaml(config)["gpr_context"]
    assert settings["gpr_context_sha256"] == augment_geometry.settings_block_sha256(
        block
    )
    assert settings["gpr_context"]["context_scale"] == [1.0, 1.5]
    assert settings["gpr_context_index"]["validation_folds"] == [0]
    assert set(settings["gpr_context_index"]["file_sha256"]) == {
        "crops.csv",
        "groups.csv",
    }

    both_scales = _n2a2_config_copy(tmp_path, index_dir, scale_on=True)
    with pytest.raises(ValueError, match="gpr_context"):
        module.main(["--config", str(both_scales), "--data", str(source_yaml)])


def test_c25_gpr_noise_with_context_is_refused(load_script, tmp_path) -> None:
    module = load_script("train_detector")
    source_yaml, index_dir = _write_context_dataset(tmp_path / "ctx")
    config = _n2a2_config_copy(tmp_path, index_dir, scale_on=False)
    with pytest.raises(ValueError, match="strength"):
        module.main(
            [
                *("--config", str(config), "--data", str(source_yaml)),
                *("--augment-strength", "0.5", "--dry-run"),
            ]
        )
    # The trainer itself refuses non-zero GPR intensities with a context.
    pytest.importorskip("ultralytics")
    index = context_index_for_data_yaml(index_dir, source_yaml)
    with pytest.raises(ValueError, match="strength 0"):
        build_gpr_trainer(
            dict(_yaml(N2A2_CONFIG)["ultralytics"]),
            AugmentConfig(),
            SEED,
            geometric_config_from_dict(
                _yaml(N2A2_CONFIG)["gpr_geometric"], scale_owned_by_context=True
            ),
            ContextScaleConfig(),
            index,
        )


def test_context_config_without_index_is_refused() -> None:
    pytest.importorskip("ultralytics")
    with pytest.raises(ValueError, match="together"):
        build_gpr_trainer(
            dict(_yaml(N2A_CONFIG)["ultralytics"]),
            AugmentConfig(),
            SEED,
            GeometricAugmentConfig(),
            context_config=ContextScaleConfig(),
        )


# ----------------------------------------------- C-21: real index smoke


def _list_names(list_file: Path) -> list[str]:
    lines = list_file.read_text(encoding="utf-8").splitlines()
    return [Path(line.strip()).name for line in lines if line.strip()]


def _hyperbola_neighbor_counts(
    index: ContextIndex, names: list[str], min_overlap_px: int
) -> tuple[int, int]:
    """(hyperbola crops with a neighbour overlapping >= min_overlap_px, all)."""
    with_neighbor = 0
    hyperbola = [index.crop(name) for name in names]
    hyperbola = [crop for crop in hyperbola if crop.primary_class == "pipe_hyperbola"]
    for crop in hyperbola:
        neighbors = index.neighbors(crop, CONFIG.extend_px)  # leak checks run here
        overlaps = [
            min(crop.x2_px, n.x2_px) - max(crop.x1_px, n.x1_px) for n in neighbors
        ]
        with_neighbor += any(overlap >= min_overlap_px for overlap in overlaps)
    return with_neighbor, len(hyperbola)


@pytest.mark.data
def test_c21_real_index_fold0_has_no_leak_and_stitchable_share() -> None:
    if not (REAL_INDEX_DIR / "crops.csv").is_file() or not REAL_DATASET_DIR.is_dir():
        pytest.skip("context index or derived dataset absent")
    train = [
        name
        for k in (1, 2, 3, 4)
        for name in _list_names(REAL_DATASET_DIR / f"fold{k}_images.txt")
    ]
    val = _list_names(REAL_DATASET_DIR / "fold0_images.txt")
    index = ContextIndex.from_csv(REAL_INDEX_DIR, train, val)
    assert index.train_folds == {1, 2, 3, 4} and index.validation_folds == {0}
    for name in train:  # every train crop's lookup passes the leak assertions
        index.neighbors(index.crop(name), CONFIG.extend_px)
    # Plan v1.0 C-21: index only, overlap >= 16 px, fold 0 training set.
    assert _hyperbola_neighbor_counts(index, train, CONFIG.min_overlap_px) == (
        306,
        1172,
    )


# ---------------------------------------------------------- R-1: pairing


def _load_r1(monkeypatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("m31_r1_overlap_shift", R1_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "m31_r1_overlap_shift", module)
    spec.loader.exec_module(module)
    return module


def test_r1_pairing_bins_and_overlap_filter(monkeypatch) -> None:
    r1 = _load_r1(monkeypatch)
    first, second = _crop(1000), _crop(920)  # overlap [1000, 1176)
    # Same hyperbola at original 1118-1138: offset 0 in the first crop, 80 in
    # the second. A cavity at 1180-1200 lies outside the overlap.
    first_labels = r1.CropLabels(
        np.array([[118.0, 40, 138, 60], [180, 40, 200, 60]]),
        np.array([0, 2]),
        np.array([0, 1]),
    )
    second_labels = r1.CropLabels(
        np.array([[198.0, 40, 218, 60], [250, 40, 256, 60]]),
        np.array([0, 2]),
        np.array([0, 1]),
    )
    pairs = r1.adjacent_pairs([first, second])
    assert pairs == [(second, first)]
    objects = r1.pair_objects(first, second, first_labels, second_labels)
    assert len(objects) == 1
    item = objects[0]
    assert (item.near_file, item.near_offset_px) == (first.file_name, 0.0)
    assert (item.far_file, item.far_offset_px) == (second.file_name, 80.0)

    detected = {(first.file_name, 0): True, (second.file_name, 0): False}
    summary = r1.summarize(objects, detected)
    bins = summary["all"]["objects_by_offset_bin"]
    assert bins["0-32"]["near"] == {"n": 1, "detected": 1}
    assert bins["64-96"]["far"] == {"n": 1, "detected": 0}
    total = sum(side["n"] for row in bins.values() for side in row.values())
    assert total == 2
    assert (
        summary["pipe_hyperbola"]["pairs_by_far_offset_bin"]["64-96"]["near_only"] == 1
    )
    assert summary["cavity"]["pairs_by_far_offset_bin"]["0-32"]["n"] == 0
    assert [r1.offset_bin(v) for v in (0, 31.9, 32, 64, 96, -100)] == [
        "0-32",
        "0-32",
        "32-64",
        "64-96",
        ">96",
        ">96",
    ]


def test_r1_hit_needs_score_above_threshold_and_iou(monkeypatch) -> None:
    r1 = _load_r1(monkeypatch)
    from gpr_detection.detect.base import (
        Detection,
        DetectorOutput,
        InputTransform,
        PixelBox,
    )

    labels = r1.CropLabels(
        np.array([[100.0, 40, 140, 80], [10, 10, 30, 30]]),
        np.array([0, 3]),
        np.array([0, 1]),
    )
    detections = (
        Detection(0, "pipe_hyperbola", 0.30, PixelBox(101, 41, 141, 81)),
        Detection(3, "manhole", 0.25, PixelBox(10, 10, 30, 30)),  # not > 0.25
    )
    identity = InputTransform(1.0, 1.0, 0.0, 0.0, SIZE_PX, SIZE_PX)
    hits = r1._gt_hits(DetectorOutput(detections, identity), labels)
    assert hits == [True, False]


# ------------------------------------------------ A1 judge: context gate


def _load_judge(monkeypatch) -> ModuleType:
    path = PROJECT_ROOT / "analysis" / "detection" / "n2a_a1_judge.py"
    spec = importlib.util.spec_from_file_location("n2a_a1_judge", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "n2a_a1_judge", module)
    spec.loader.exec_module(module)
    return module


def _judge_fold(judge, tmp_path: Path, train_settings: dict):
    weights = tmp_path / "last.pt"
    weights.write_bytes(b"w")
    eval_list = tmp_path / "eval_images.txt"
    eval_list.write_text("a.png\n", encoding="utf-8")
    weights_sha = augment_geometry.file_sha256(weights)
    list_sha = augment_geometry.file_sha256(eval_list)
    metrics = {
        "eval_images": {"sha256": list_sha},
        "weights": {"sha256": weights_sha, "name": "last"},
        "unit": "crop",
        "fold": "0",
    }
    args = {"scale": 0.0, "translate": 0.1, "flipud": 0.0, "epochs": 50, "seed": 0}
    return judge.FoldFiles(
        fold=0,
        run_dir=tmp_path,
        metrics=metrics,
        m110_metrics=metrics,
        args_yaml=args,
        train_settings={"fold": "0", "config_sha256": "seal", **train_settings},
        train_record={"weights_sha256": {"last": weights_sha}},
        weights=weights,
        eval_list=eval_list,
    )


def test_a1_judge_checks_the_sealed_context_block(monkeypatch, tmp_path) -> None:
    judge = _load_judge(monkeypatch)
    assert judge.sealed_context_sha256(N2A_CONFIG.read_bytes()) is None
    sealed = judge.sealed_context_sha256(N2A2_CONFIG.read_bytes())
    expected = augment_geometry.settings_block_sha256(_yaml(N2A2_CONFIG)["gpr_context"])
    assert sealed == expected

    good = _judge_fold(judge, tmp_path, {"gpr_context_sha256": expected})
    row = judge.check_fold(good, "seal", sealed)
    assert row["pass"] and row["checks"]["gpr_context_matches_seal"]
    changed = _judge_fold(judge, tmp_path, {"gpr_context_sha256": "0" * 64})
    assert not judge.check_fold(changed, "seal", sealed)["pass"]
    # N2-a judging is unchanged: no context check without a sealed block.
    assert "gpr_context_matches_seal" not in judge.check_fold(good, "seal")["checks"]


# ------------------------------------------- R1 end to end (fake model)


class _NearOnlyDetector:
    """Finds the hyperbola only in crop 1000 (column 0 = 1000 mod 251 = 247).

    Crop 920 starts with 920 mod 251 = 167, so the first pixel tells the crops
    apart even after JPEG compression.
    """

    def predict(self, image: np.ndarray):
        from gpr_detection.detect.base import (
            Detection,
            DetectorOutput,
            InputTransform,
            PixelBox,
        )

        identity = InputTransform(1.0, 1.0, 0.0, 0.0, SIZE_PX, SIZE_PX)
        detections = ()
        if image[SIZE_PX // 2, 0] > 207:
            box = PixelBox(118.0, 41.2, 138.0, 61.2)
            detections = (Detection(HYPERBOLA, "pipe_hyperbola", 0.9, box),)
        return DetectorOutput(detections, identity)


def _write_r1_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    """Fold 0: a 1.0/0.5 gain pair at (1000, 920) and a lone crop; fold 1 empty."""
    index_dir, image_dir, label_dir = root / "index", root / "images", root / "labels"
    for folder in (index_dir, image_dir, label_dir):
        folder.mkdir(parents=True)
    crops = [
        _crop(1000, fold=0),
        _crop(920, fold=0),
        _crop(1000, fold=0, gain="0.5"),  # not the representative gain
        _crop(920, fold=0, gain="0.5"),
    ]
    # Hyperbola at original 1118-1138 (offset 0 in 1000, 80 in 920).
    label_x = {1000: 128 / SIZE_PX, 920: 208 / SIZE_PX}
    _write_crops_csv(index_dir / "crops.csv", crops)
    for crop in crops:
        cv2.imwrite(str(image_dir / crop.file_name), _stripes(crop.x1_px))
        (label_dir / f"{Path(crop.file_name).stem}.txt").write_text(
            f"0 {label_x[crop.x1_px]:.6f} 0.2 0.078125 0.078125\n", encoding="utf-8"
        )
    (index_dir / "groups.csv").write_text("group_id\n", encoding="utf-8")
    run_root = root / "runs"
    for fold in (0, 1):
        weights = run_root / f"r1_fold{fold}" / "weights" / "last.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"w")
    return index_dir, image_dir, label_dir, run_root


def test_r1_end_to_end_with_a_fake_model(monkeypatch, tmp_path, capsys) -> None:
    r1 = _load_r1(monkeypatch)
    index_dir, image_dir, label_dir, run_root = _write_r1_inputs(tmp_path / "r1")
    monkeypatch.setattr(r1, "_load_detector", lambda w, d: _NearOnlyDetector())
    out = tmp_path / "r1.json"
    arguments = [
        *("--run-prefix", "r1_fold", "--folds", "0,1", "--device", "cpu"),
        *("--out", str(out), "--run-root", str(run_root), "--index-dir"),
        *(str(index_dir), "--image-dir", str(image_dir), "--label-dir"),
        *(str(label_dir), "--blind"),
    ]
    assert r1.main(arguments) == 0
    assert capsys.readouterr().out.strip().count("\n") == 0  # status line only
    report = yaml.safe_load(out.read_text(encoding="utf-8"))
    fold0 = report["per_fold"]["0"]
    assert (fold0["crop_pairs"], fold0["paired_objects"]) == (1, 1)
    bins = fold0["by_class"]["pipe_hyperbola"]["objects_by_offset_bin"]
    assert bins["0-32"]["near"] == {"n": 1, "detected": 1, "recall": 1.0}
    assert bins["64-96"]["far"] == {"n": 1, "detected": 0, "recall": 0.0}
    assert report["per_fold"]["1"]["paired_objects"] == 0
    assert set(report["run_identity"]["weights_sha256"]) == {"0", "1"}
    assert r1.main(arguments) == 2  # existing --out is refused


# ------------------------------- A1 judge end to end: index sha gate


JUDGE_PREFIX = "n2a2_fold"
JUDGE_M110_PREFIX = "m110_fold"
INDEX_SHA = {"crops.csv": "a" * 64, "groups.csv": "b" * 64}


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return completed.stdout.strip()


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _judge_inputs(root: Path, odd_index_fold: int | None) -> list[str]:
    """Five synthetic N2-a' folds; ``odd_index_fold`` records another index sha."""
    repo = root / "repo"
    (repo / "configs").mkdir(parents=True)
    config = repo / "configs" / "n2a2.yaml"
    config.write_bytes(N2A2_CONFIG.read_bytes())
    _git(repo, "init", "-q")
    _git(repo, "add", "configs/n2a2.yaml")
    _git(repo, "commit", "-q", "-m", "seal")
    context_sha = augment_geometry.settings_block_sha256(_yaml(config)["gpr_context"])
    run_root = root / "runs"
    for k in range(5):
        run = run_root / f"{JUDGE_PREFIX}{k}"
        weights = run / "weights" / "last.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(f"weights {k}".encode())
        eval_list = run / "eval" / "last_crop" / "eval_images.txt"
        eval_list.parent.mkdir(parents=True)
        eval_list.write_text(f"a{k}.png\n", encoding="utf-8")
        metrics = {
            "fold": str(k),
            "unit": "crop",
            "weights": {
                "name": "last",
                "sha256": augment_geometry.file_sha256(weights),
            },
            "eval_images": {"sha256": augment_geometry.file_sha256(eval_list)},
            "map50": 0.8,
            "map50_95": 0.4,
            "per_class": [{"name": "pipe_hyperbola", "ap50": 0.95}],
        }
        _write_json(eval_list.parent / "metrics.json", metrics)
        _write_json(
            run_root
            / f"{JUDGE_M110_PREFIX}{k}"
            / "eval"
            / "last_crop"
            / "metrics.json",
            metrics,
        )
        args = {"scale": 0.0, "translate": 0.1, "flipud": 0.0, "epochs": 50, "seed": 0}
        (run / "args.yaml").write_text(yaml.safe_dump(args), encoding="utf-8")
        index_sha = dict(INDEX_SHA)
        if k == odd_index_fold:
            index_sha["crops.csv"] = "c" * 64
        settings = {
            "fold": str(k),
            "config_sha256": augment_geometry.file_sha256(config),
            "gpr_context_sha256": context_sha,
            "gpr_context_index": {"file_sha256": index_sha},
        }
        (run / "gpr_train_settings.yaml").write_text(
            yaml.safe_dump(settings), encoding="utf-8"
        )
        _write_json(
            run / "train_record.json",
            {"weights_sha256": {"last": augment_geometry.file_sha256(weights)}},
        )
    return [
        *("--run-prefix", JUDGE_PREFIX, "--m110-prefix", JUDGE_M110_PREFIX),
        *("--run-root", str(run_root), "--config", str(config)),
        *("--seal-commit", _git(repo, "rev-parse", "HEAD"), "--repo", str(repo)),
        *("--out", str(root / "verdict.json"), "--blind"),
    ]


@pytest.mark.parametrize(("odd_index_fold", "expected_exit"), [(None, 0), (3, 1)])
def test_a1_judge_index_sha_must_match_in_all_folds(
    monkeypatch, tmp_path, odd_index_fold, expected_exit
) -> None:
    judge = _load_judge(monkeypatch)
    arguments = _judge_inputs(tmp_path, odd_index_fold)
    assert judge.main(arguments) == expected_exit
    verdict = json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8"))
    gate = verdict["gates"]["config"]["context_index_same_in_all_folds"]
    assert gate is (odd_index_fold is None)
    assert verdict["rules"].endswith("v1.0 §5 (A1 + gpr_context gate), §7-3")
    if odd_index_fold is not None:
        assert verdict["verdict"] == "no verdict: gate failed"


# ----------------------------------------- N2-a″ rule K32: K-1..K-10

K32_WIDTH_PX = 32.0
K32_IMAGES = 1000
K32_OFF = ContextScaleConfig()
K32_ON = ContextScaleConfig(band_min_clipped_width_px=K32_WIDTH_PX)
# C-1 layout: centre (1000, 1256), neighbour (1104, 1360), strip width 360.
K32_CENTRE_LABELS = (
    [(108, 50, 148, 90), (50, 70, 200, 86), (200, 100, 256, 116), (230, 140, 256, 156)],
    [HYPERBOLA, BAND, BAND, BAND],
)
# Neighbour crop px: the band 46-166 is original 1150-1270, cut to the 14 px
# piece [1256, 1270) by column ownership (C-13, K-8); 150-256 gives 104 px.
K32_NEIGHBOUR_LABELS = (
    [(6, 50, 46, 90), (46, 72, 166, 88), (150, 120, 256, 136), (200, 30, 240, 60)],
    [HYPERBOLA, BAND, BAND, CAVITY],
)


def _k32_inputs(
    centre_labels=K32_CENTRE_LABELS, neighbour_labels=K32_NEIGHBOUR_LABELS
) -> tuple[str, CropContent, ContextIndex, _MemoryLoader]:
    center, neighbor = _crop(1000), _crop(1104)
    loader = _MemoryLoader()
    loader.add(neighbor, _stripes(1104), *neighbour_labels)
    content = _content(center, boxes=centre_labels[0], classes=centre_labels[1])
    return center.file_name, content, _index([center, neighbor]), loader


def _run_stage(config: ContextScaleConfig, rng, inputs=None):
    center_file, content, index, loader = inputs or _k32_inputs()
    return augment_context_scale(
        center_file, content, index, loader, config, BoxKeepRule(), rng
    )


def _rows(boxes: np.ndarray, classes: np.ndarray, mask: np.ndarray) -> set:
    return {
        (int(class_id), *map(float, box))
        for box, class_id, is_kept in zip(boxes, classes, mask, strict=True)
        if is_kept
    }


def test_k1_k2_k32_keeps_pixels_non_band_labels_and_draws() -> None:
    """K-1 (image and label identity, band subset) and K-2 (rng state)."""
    k32_drops = 0
    for i in range(K32_IMAGES):
        rng_off = np.random.default_rng([SEED, i])
        rng_on = np.random.default_rng([SEED, i])
        image_off, boxes_off, classes_off, keep_off, _ = _run_stage(K32_OFF, rng_off)
        image_on, boxes_on, classes_on, keep_on, record_on = _run_stage(K32_ON, rng_on)
        assert image_on.tobytes() == image_off.tobytes(), i
        assert rng_on.bit_generator.state == rng_off.bit_generator.state, i
        is_band_off, is_band_on = classes_off == BAND, classes_on == BAND
        assert _rows(boxes_on, classes_on, keep_on & ~is_band_on) == _rows(
            boxes_off, classes_off, keep_off & ~is_band_off
        ), i
        bands_on = _rows(boxes_on, classes_on, keep_on & is_band_on)
        assert bands_on <= _rows(boxes_off, classes_off, keep_off & is_band_off), i
        assert len(record_on.band_k32_dropped) == int((keep_off & ~keep_on).sum())
        k32_drops += len(record_on.band_k32_dropped)
    assert k32_drops >= 1  # the inputs really reach the rule


def test_k3_threshold_is_strictly_below_32() -> None:
    boxes = np.array([[10.0, 0, 41.9, 10], [100.0, 0, 132.0, 10]])
    keep, dropped = drop_short_clipped_bands(
        boxes, np.array([BAND, BAND]), np.array([True, True]), [100.0, 100.0], 32.0
    )
    assert keep.tolist() == [False, True] and dropped.tolist() == [0]


def test_k4_unclipped_short_band_stays() -> None:
    boxes = np.array([[10.0, 0, 30.0, 10], [50.0, 0, 69.2, 10]])
    masks = np.array([True, True])
    keep, dropped = drop_short_clipped_bands(
        boxes, np.array([BAND, BAND]), masks, [20.0, 20.0], 32.0
    )
    assert keep.tolist() == [True, True] and len(dropped) == 0
    assert masks.tolist() == [True, True]  # the input mask is not modified
    # Rule off (min width <= 0): a copy of the mask, nothing dropped.
    clipped = np.array([[0.0, 0, 10, 10]])
    keep, dropped = drop_short_clipped_bands(clipped, [BAND], [True], [100.0], 0.0)
    assert keep.tolist() == [True] and len(dropped) == 0


def test_k5_plain_mode_is_untouched() -> None:
    # Bands touching both crop ends; the 26 px one at the right end is clipped
    # below 32 px with >= 50 % visible for s in (1, 1.113], so K32 would drop it.
    centre_labels = ([(0, 70, 256, 86), (230, 100, 256, 116)], [BAND, BAND])
    inputs = _k32_inputs(centre_labels=centre_labels)
    plain_off = dataclasses.replace(K32_OFF, p_context=0.0)
    plain_on = dataclasses.replace(K32_ON, p_context=0.0)
    has_short_clipped_kept = False
    for i in range(K32_IMAGES):
        off = _run_stage(plain_off, np.random.default_rng([SEED, i]), inputs)
        on = _run_stage(plain_on, np.random.default_rng([SEED, i]), inputs)
        assert on[0].tobytes() == off[0].tobytes()
        assert on[1].tobytes() == off[1].tobytes()
        assert on[3].tobytes() == off[3].tobytes()
        assert on[4].window.mode is ContextMode.PLAIN and on[4].band_k32_dropped == ()
        visible_width = on[1][1, 2] - on[1][1, 0]
        has_short_clipped_kept |= bool(on[3][1] and visible_width < K32_WIDTH_PX)
    assert has_short_clipped_kept  # the test reaches boxes K32 would drop


def test_k6_missing_or_zero_key_is_n2a2_behaviour() -> None:
    n2a2_block = _yaml(N2A2_CONFIG)["gpr_context"]
    assert context_config_from_dict(n2a2_block).band_min_clipped_width_px == 0.0
    zero = context_config_from_dict({**n2a2_block, "band_min_clipped_width_px": 0})
    assert zero.band_min_clipped_width_px == 0.0
    missing = context_config_from_dict(n2a2_block)
    for i in range(K32_IMAGES):
        rng_missing = np.random.default_rng([SEED, i])
        rng_zero = np.random.default_rng([SEED, i])
        first = _run_stage(missing, rng_missing)
        second = _run_stage(zero, rng_zero)
        for left, right in zip(first[:4], second[:4], strict=True):
            assert left.tobytes() == right.tobytes()
        assert first[4] == second[4] and first[4].band_k32_dropped == ()
        assert rng_missing.bit_generator.state == rng_zero.bit_generator.state


def test_k7_k32_does_not_touch_other_classes() -> None:
    # Plan K-7 values on the function itself: hyperbola, visible 20 of 60 px.
    keep, dropped = drop_short_clipped_bands(
        np.array([[0.0, 0, 20, 40]]), np.array([HYPERBOLA]), [True], [60.0], 32.0
    )
    assert keep.tolist() == [True] and len(dropped) == 0
    # Through the stage: a 36 px hyperbola and band cut to 20 px at the window's
    # left edge (dx = -40 shows strip columns 40-296); 20 / 36 >= 50 % visible.
    centre_labels = ([(24, 50, 60, 90), (24, 100, 60, 116)], [HYPERBOLA, BAND])
    inputs = _k32_inputs(centre_labels=centre_labels, neighbour_labels=((), ()))
    draw = _FixedDraw(mode_draw=0.0, s=1.0, dx=-40)
    _, boxes_off, _, keep_off, _ = _run_stage(K32_OFF, draw, inputs)
    _, boxes_on, classes_on, keep_on, record = _run_stage(K32_ON, draw, inputs)
    np.testing.assert_array_equal(boxes_on[:2], [[0, 50, 20, 90], [0, 100, 20, 116]])
    np.testing.assert_array_equal(boxes_on, boxes_off)  # K32 only changes the mask
    assert classes_on[:2].tolist() == [HYPERBOLA, BAND]
    assert keep_off[:2].tolist() == [True, True]
    assert keep_on[:2].tolist() == [True, False]
    assert record.band_k32_dropped == ((20.0, augment_geometry.ORIGIN_CENTRE),)


def test_k8_column_ownership_cut_counts_as_clipped() -> None:
    # C-13 layout. The plan's dx = 0 would leave the piece at strip 256-270
    # outside the window; dx = -40 shows strip 40-296, so it is fully inside.
    centre_labels = ([(50, 70, 200, 86)], [BAND])
    neighbour_labels = ([(46, 72, 166, 88)], [BAND])
    inputs = _k32_inputs(centre_labels, neighbour_labels)
    draw = _FixedDraw(mode_draw=0.0, s=1.0, dx=-40)
    _, boxes_off, classes_off, keep_off, record_off = _run_stage(K32_OFF, draw, inputs)
    _, boxes_on, _, keep_on, record_on = _run_stage(K32_ON, draw, inputs)
    np.testing.assert_array_equal(boxes_off, [[10, 70, 160, 86], [216, 72, 230, 88]])
    np.testing.assert_array_equal(boxes_on, boxes_off)
    assert classes_off.tolist() == [BAND, BAND]
    assert keep_off.tolist() == [True, True] and record_off.band_k32_dropped == ()
    assert keep_on.tolist() == [True, False]
    assert record_on.band_k32_dropped == ((14.0, augment_geometry.ORIGIN_NEIGHBOUR),)
    strip = assemble_context_strip(*inputs[:3], inputs[3], K32_ON)
    assert strip.source_widths_px.tolist() == [150.0, 120.0]
    assert strip.origins.tolist() == [0, 1]


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "32", True])
def test_k9_bad_k32_value_is_refused(value) -> None:
    with pytest.raises(ValueError, match="band_min_clipped_width_px"):
        ContextScaleConfig(band_min_clipped_width_px=value)
    with pytest.raises(ValueError, match="band_min_clipped_width_px"):
        context_config_from_dict({"band_min_clipped_width_px": value})


def _k10_neighbour_band_text() -> str:
    """Hyperbola plus a band at crop px 46-166 (original 1150-1270) for 1104."""
    return "0 0.5 0.3125 0.15625 0.15625\n1 0.4140625 0.3125 0.46875 0.0625\n"


def test_k10_pickled_step_same_result_and_band_k32_stats(tmp_path) -> None:
    root = tmp_path / "ctx"
    source_yaml, index_dir = _write_context_dataset(
        root, {1104: _k10_neighbour_band_text()}
    )
    index = context_index_for_data_yaml(index_dir, source_yaml)
    config = ContextScaleConfig(p_context=1.0, band_min_clipped_width_px=32)
    fresh = YoloContextScale(config, index, SEED)
    restored = pickle.loads(pickle.dumps(YoloContextScale(config, index, SEED)))
    outputs = {"fresh": [], "restored": []}
    for name, step in (("fresh", fresh), ("restored", restored)):
        for _ in range(40):
            labels = step(_context_labels(root, 1000))
            outputs[name].append(
                (
                    labels["img"].tobytes(),
                    labels["instances"].bboxes.tobytes(),
                    labels["cls"].tobytes(),
                )
            )
    assert outputs["fresh"] == outputs["restored"]
    block = restored._stats_document()[aug_stats.BAND_K32_KEY]
    assert block["dropped"] >= 1
    assert block["dropped"] == sum(block["by_origin"].values())
    assert block["dropped"] == sum(block["width_bins"].values())
    assert set(block["width_bins"]) == {"0-8", "8-16", "16-24", "24-32"}
    stage = restored._stats_document()["stages"]["context_scale"]
    assert stage["boxes_dropped_by_class"]["1"] >= block["dropped"]

    restored.stats_dir = tmp_path / "workers"
    restored.stats_dir.mkdir()
    restored.flush_stats()
    aggregated = aug_stats.aggregate_worker_stats(restored.stats_dir)
    assert aggregated[aug_stats.BAND_K32_KEY] == block
    # N2-a' (rule off): the stats document keeps exactly its old keys.
    off_step = YoloContextScale(ContextScaleConfig(p_context=1.0), index, SEED)
    off_step(_context_labels(root, 1000))
    assert aug_stats.BAND_K32_KEY not in off_step._stats_document()


def test_n2a3_config_differs_from_n2a2_only_by_k32() -> None:
    n2a2, n2a3 = _yaml(N2A2_CONFIG), _yaml(N2A3_CONFIG)
    k32 = n2a3["gpr_context"].pop("band_min_clipped_width_px")
    assert k32 == 32
    assert n2a3 == n2a2
    assert context_config_from_dict(_yaml(N2A3_CONFIG)["gpr_context"]) == K32_ON


def test_n2a3_training_record_carries_k32_for_the_a1_judge(
    load_script, monkeypatch, tmp_path
) -> None:
    """Plan §6-1: the recorded block has the key and matches the judge's sha."""
    module = load_script("train_detector")
    runs_dir = tmp_path / "runs" / "detect"
    module.RUNS_DIR = runs_dir
    module.DATA_YAML_DIR = runs_dir / "_data"

    class _Trainer:
        def __init__(self, save_dir: Path) -> None:
            self.save_dir = save_dir

        def train(self) -> None:
            return None

    module.build_gpr_trainer = lambda overrides, *args, **kwargs: _Trainer(
        Path(overrides["project"]) / overrides["name"]
    )
    source_yaml, index_dir = _write_context_dataset(tmp_path / "ctx")
    config = _yaml(N2A3_CONFIG)
    config["gpr_context"]["index_dir"] = str(index_dir)
    config_path = tmp_path / "n2a3.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    arguments = ["--config", str(config_path), "--data", str(source_yaml)]
    assert module.main([*arguments, "--name", "ctx"]) == 0
    settings = _yaml(runs_dir / "ctx" / "gpr_train_settings.yaml")
    assert settings["gpr_context"]["band_min_clipped_width_px"] == 32
    judge = _load_judge(monkeypatch)
    sealed = judge.sealed_context_sha256(config_path.read_bytes())
    assert settings["gpr_context_sha256"] == sealed
    n2a2_like = dict(config["gpr_context"])
    del n2a2_like["band_min_clipped_width_px"]
    assert sealed != augment_geometry.settings_block_sha256(n2a2_like)
