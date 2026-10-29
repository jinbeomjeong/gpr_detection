"""Tests for the Ultralytics training adapter (M1-7 / M1-8).

The last test trains a tiny one-channel YOLO11n for one epoch on CPU on four
synthetic 64 px images inside ``tmp_path``. It exercises the real
Ultralytics path: ``channels: 1`` data yaml, the inserted augmentation step and
one forward/backward pass.
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from gpr_detection.augment import (
    AugmentConfig,
    ContrastGammaConfig,
    GaussianNoiseConfig,
    RingingConfig,
    TraceDropoutConfig,
    augment_config_from_dict,
)
from gpr_detection.detect.yolo_train import (
    YoloGprAugment,
    attach_gpr_augment,
    build_gpr_trainer,
    check_augmentation_policy,
    write_training_data_yaml,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs" / "train_yolo11m_base.yaml"
CLASS_NAMES = {0: "pipe_hyperbola", 1: "pipe_band", 2: "cavity", 3: "manhole"}
SEED = 11


def _always_on() -> AugmentConfig:
    return AugmentConfig(
        gaussian_noise=GaussianNoiseConfig(probability=1.0),
        contrast_gamma=ContrastGammaConfig(probability=1.0),
        ringing=RingingConfig(probability=1.0),
        trace_dropout=TraceDropoutConfig(probability=1.0),
    )


def _labels(image: np.ndarray):
    instance_module = pytest.importorskip("ultralytics.utils.instance")
    instances = instance_module.Instances(
        bboxes=np.array([[0.5, 0.5, 0.25, 0.5]], dtype=np.float32),
        bbox_format="xywh",
        normalized=True,
    )
    return {"img": image, "instances": instances, "cls": np.array([[0.0]])}


def _base_config() -> dict:
    return yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))


# --- YoloGprAugment ---------------------------------------------------------------


def test_augment_step_changes_image_but_not_buffer_or_boxes() -> None:
    image = np.full((64, 64, 1), 128, dtype=np.uint8)
    labels = _labels(image)
    boxes_before = labels["instances"].bboxes.copy()

    result = YoloGprAugment(_always_on(), seed=SEED)(labels)

    assert result["img"].shape == (64, 64, 1)
    assert result["img"].dtype == np.uint8
    assert not np.array_equal(result["img"], image)
    assert np.all(image == 128), "decoded-image buffer must not be modified"
    np.testing.assert_array_equal(result["instances"].bboxes, boxes_before)
    assert result["instances"].normalized


def test_augment_step_accepts_two_dimensional_image() -> None:
    image = np.full((32, 32), 100, dtype=np.uint8)
    result = YoloGprAugment(_always_on(), seed=SEED)(_labels(image))
    assert result["img"].shape == (32, 32)


def test_augment_step_rejects_three_channel_images() -> None:
    labels = _labels(np.zeros((16, 16, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="channels: 1"):
        YoloGprAugment(_always_on(), seed=SEED)(labels)


def test_augment_step_is_reproducible_and_picklable() -> None:
    image = np.random.default_rng(0).integers(0, 256, (48, 48, 1), dtype=np.uint8)
    first = YoloGprAugment(_always_on(), seed=SEED)
    second = YoloGprAugment(_always_on(), seed=SEED)

    out_first = first(_labels(image))["img"]
    out_second = second(_labels(image))["img"]
    restored = pickle.loads(pickle.dumps(first))

    np.testing.assert_array_equal(out_first, out_second)
    assert restored._generator is None
    assert restored.config == first.config
    assert "seed=11" in repr(restored)


def test_attach_inserts_once_at_front() -> None:
    augment_module = pytest.importorskip("ultralytics.data.augment")
    flip = augment_module.RandomFlip(direction="horizontal", p=0.5)
    dataset = type("FakeDataset", (), {})()
    dataset.transforms = augment_module.Compose([flip])
    step = YoloGprAugment(AugmentConfig(), seed=SEED)

    attach_gpr_augment(dataset, step)
    attach_gpr_augment(dataset, step)

    assert dataset.transforms.transforms == [step, flip]


def test_augmenting_before_affine_keeps_padding_border_clean() -> None:
    """Front insertion means noise and stripes never reach the gray (114) border."""
    augment_module = pytest.importorskip("ultralytics.data.augment")
    shrink_to_half = augment_module.RandomPerspective(
        translate=0.0, scale=(0.5, 0.5), size=(64, 64)
    )
    pipeline = augment_module.Compose(
        [YoloGprAugment(_always_on().scaled(3.0), seed=SEED), shrink_to_half]
    )
    labels = _labels(np.full((64, 64, 1), 128, dtype=np.uint8))

    image = pipeline(labels)["img"]

    border_value = 114
    assert np.all(image[:12] == border_value)
    assert np.all(image[-12:] == border_value)
    assert not np.all(image[24:40, 24:40] == 128)


# --- data yaml and policy -----------------------------------------------------------


def test_training_data_yaml_gets_one_channel_and_absolute_path(tmp_path) -> None:
    source = tmp_path / "dataset" / "fold0.yaml"
    source.parent.mkdir()
    source.write_text(
        yaml.safe_dump({"path": ".", "train": ["a.txt"], "val": "b.txt", "names": {}}),
        encoding="utf-8",
    )

    written = write_training_data_yaml(source, tmp_path / "runs" / "fold0.yaml")

    data = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert data["channels"] == 1
    assert Path(data["path"]) == source.parent.resolve()
    assert data["train"] == ["a.txt"]
    assert "channels" not in yaml.safe_load(source.read_text(encoding="utf-8"))


def test_base_config_follows_policy_and_code_defaults() -> None:
    config = _base_config()

    warnings = check_augmentation_policy(config["ultralytics"])

    assert config["model"] == "yolo11m.yaml"
    assert config["ultralytics"]["imgsz"] == 256
    assert len(warnings) == 1 and "hsv_v" in warnings[0]
    assert augment_config_from_dict(config["gpr_augment"]) == AugmentConfig()


@pytest.mark.parametrize("key", ["flipud", "degrees", "shear", "perspective"])
def test_policy_rejects_forbidden_augmentations(key) -> None:
    overrides = dict(_base_config()["ultralytics"])
    overrides[key] = 0.5
    with pytest.raises(ValueError, match=key):
        check_augmentation_policy(overrides)
    del overrides[key]
    with pytest.raises(ValueError, match=key):
        check_augmentation_policy(overrides)


def test_policy_warns_about_ablation_settings() -> None:
    overrides = dict(_base_config()["ultralytics"], mosaic=0.5, hsv_v=0.0)
    assert check_augmentation_policy(overrides) == [
        "mosaic=0.5 is an ablation-only setting"
    ]


# --- one-epoch smoke on synthetic data ---------------------------------------------


def _write_tiny_dataset(root: Path) -> Path:
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    rng = np.random.default_rng(0)
    for index in range(4):
        image = rng.integers(60, 120, (64, 64), dtype=np.uint8)
        image[20:40, 16:48] = 230
        cv2.imwrite(str(images / f"img{index}.png"), image)
        (labels / f"img{index}.txt").write_text(
            f"{index % 4} 0.5 0.47 0.5 0.31\n", encoding="utf-8"
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


def test_one_epoch_cpu_training_uses_one_channel_and_gpr_augment(tmp_path) -> None:
    pytest.importorskip("ultralytics")
    data_yaml = write_training_data_yaml(
        _write_tiny_dataset(tmp_path / "tiny"), tmp_path / "tiny_ch1.yaml"
    )
    overrides = dict(_base_config()["ultralytics"])
    overrides.update(
        model="yolo11n.yaml",
        data=str(data_yaml),
        imgsz=64,
        epochs=1,
        # Mosaic on, closed for the last (only) epoch: the transforms are rebuilt
        # and the GPR step must be re-attached (asserted below).
        mosaic=0.5,
        close_mosaic=1,
        batch=4,
        workers=0,
        device="cpu",
        project=str(tmp_path / "runs"),
        name="smoke",
        plots=False,
        amp=False,
    )

    trainer = build_gpr_trainer(overrides, _always_on(), augment_seed=SEED)
    trainer.train()

    assert trainer.data["channels"] == 1
    first_conv = trainer.model.model[0].conv
    assert first_conv.in_channels == 1
    steps = trainer.train_loader.dataset.transforms.transforms
    assert isinstance(steps[0], YoloGprAugment)
    assert (Path(trainer.save_dir) / "weights" / "last.pt").is_file()
