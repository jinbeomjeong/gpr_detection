"""Ultralytics YOLO training adapter: one-channel data and the GPR augmentations.

How the project augmentations reach Ultralytics 8.4.192 (least invasive path):

* ``DetectionTrainer`` is subclassed (``build_dataset`` is the documented
  customisation hook). After Ultralytics builds the train dataset, a
  :class:`YoloGprAugment` step is inserted at the **front** of
  ``dataset.transforms``, i.e. on the decoded image before
  Mosaic/RandomPerspective. So the noise falls on the B-scan only, not on the
  gray (114) border that the affine step adds.
* ``_close_dataloader_mosaic`` rebuilds the transforms; the subclass re-inserts
  the step there (only relevant to mosaic ablations with ``close_mosaic > 0``).
* Validation datasets are left untouched.

One-channel input uses the official ``channels: 1`` key of the data yaml: the
dataset then decodes with ``cv2.IMREAD_GRAYSCALE`` and ``DetectionModel`` is
built with ``ch=1``. :func:`write_training_data_yaml` adds that key to a copy
of the fold yaml, so the derived dataset under ``data/`` is never modified.

``ultralytics`` is imported only inside functions (ADR-001).
"""

import copy
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gpr_detection.augment import AugmentConfig, augment_bscan

LOGGER = logging.getLogger(__name__)

GRAY_CHANNELS = 1
_NO_WORKER_SEED = 0
"""Seed component used in the main process (``workers=0``)."""

FORBIDDEN_AUGMENTATION_KEYS = ("flipud", "degrees", "shear", "perspective")
"""Must be 0: they flip the time axis or tilt hyperbolae (augmentation_policy §1)."""
ABLATION_ONLY_AUGMENTATION_KEYS = ("mosaic", "mixup", "cutmix", "hsv_h", "hsv_s")
"""0 in the baseline; non-zero only in a deliberate ablation (warned)."""


def _boxes_xyxy_px(instances: Any, height_px: int, width_px: int) -> np.ndarray:
    """Read Ultralytics ``Instances`` as ``xyxy`` pixel boxes without mutating them."""
    boxes = copy.deepcopy(instances)
    boxes.convert_bbox(format="xyxy")
    if boxes.normalized:
        boxes.denormalize(width_px, height_px)
    return np.asarray(boxes.bboxes, dtype=np.float64).reshape(-1, 4)


def _data_loader_worker_seed() -> int:
    import torch.utils.data  # lazy: torch is only needed while training

    worker_info = torch.utils.data.get_worker_info()
    return _NO_WORKER_SEED if worker_info is None else int(worker_info.seed)


class YoloGprAugment:
    """Ultralytics transform step that runs :func:`gpr_detection.augment.augment_bscan`.

    Picklable (Windows DataLoader workers are spawned and receive a copy). Each
    worker process derives its own generator from ``seed`` and the worker seed
    torch assigns, so workers do not repeat each other's noise while a run
    with the same ``seed`` stays reproducible.

    Args:
        config: Augmentation probabilities and intensities.
        seed: Base seed, normally the training ``seed``.
    """

    def __init__(self, config: AugmentConfig, seed: int) -> None:
        self.config = config
        self.seed = seed
        self._generator: np.random.Generator | None = None
        self._generator_pid: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_generator"] = None  # every process seeds its own generator
        state["_generator_pid"] = None
        return state

    def __repr__(self) -> str:
        return f"YoloGprAugment(seed={self.seed}, config={self.config})"

    def _rng(self) -> np.random.Generator:
        if self._generator is None or self._generator_pid != os.getpid():
            seed_sequence = np.random.SeedSequence(
                [self.seed, _data_loader_worker_seed()]
            )
            self._generator = np.random.default_rng(seed_sequence)
            self._generator_pid = os.getpid()
        return self._generator

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Augment ``labels["img"]`` (uint8 ``[H, W, 1]``) in a new array.

        Raises:
            ValueError: If the image is not one-channel.
        """
        image = labels["img"]
        if image.ndim == 3 and image.shape[2] == GRAY_CHANNELS:
            gray = image[..., 0]
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError(
                f"YoloGprAugment expects a one-channel image, got {image.shape}; "
                "set 'channels: 1' in the data yaml"
            )
        height_px, width_px = gray.shape
        boxes = _boxes_xyxy_px(labels["instances"], height_px, width_px)
        augmented, _ = augment_bscan(gray, boxes, self.config, self._rng())
        labels["img"] = augmented.reshape(image.shape)
        return labels


def attach_gpr_augment(dataset: Any, augment: YoloGprAugment) -> None:
    """Insert ``augment`` at the front of ``dataset.transforms`` once.

    Args:
        dataset: Ultralytics ``YOLODataset`` built with ``augment=True``.
        augment: The step to insert.
    """
    transforms = dataset.transforms
    if any(isinstance(step, YoloGprAugment) for step in transforms.transforms):
        return
    transforms.insert(0, augment)


def check_augmentation_policy(overrides: Mapping[str, Any]) -> list[str]:
    """Validate Ultralytics augmentation arguments against the project policy.

    Args:
        overrides: Ultralytics train arguments.

    Returns:
        Warnings for ablation-only settings and for ``hsv_v``, which Ultralytics
        8.4.192 silently skips on one-channel images (``RandomHSV`` only
        touches 3-channel input).

    Raises:
        ValueError: If a forbidden augmentation is non-zero or missing.
    """
    missing_or_active = [
        key for key in FORBIDDEN_AUGMENTATION_KEYS if overrides.get(key) != 0
    ]
    if missing_or_active:
        raise ValueError(
            f"augmentations {missing_or_active} must be set explicitly to 0 "
            "(docs/experiments/augmentation_policy.md)"
        )
    warnings = [
        f"{key}={overrides[key]} is an ablation-only setting"
        for key in ABLATION_ONLY_AUGMENTATION_KEYS
        if overrides.get(key, 0) != 0
    ]
    if overrides.get("hsv_v", 0) != 0:
        warnings.append(
            f"hsv_v={overrides['hsv_v']} has no effect on one-channel images "
            "(Ultralytics RandomHSV skips them); brightness variation comes "
            "from the GPR contrast/gamma augmentation"
        )
    return warnings


def write_training_data_yaml(source_yaml: Path, out_yaml: Path) -> Path:
    """Copy an Ultralytics data yaml and set ``channels: 1``.

    A relative or missing ``path`` is made absolute against the source folder so
    the copy can live anywhere (e.g. under ``runs/``).

    Args:
        source_yaml: Fold yaml written by ``scripts/build_yolo_dataset.py``.
        out_yaml: Where to write the one-channel copy.

    Returns:
        ``out_yaml``.
    """
    data = yaml.safe_load(Path(source_yaml).read_text(encoding="utf-8"))
    dataset_root = Path(data.get("path") or Path(source_yaml).parent)
    if not dataset_root.is_absolute():
        dataset_root = (Path(source_yaml).parent / dataset_root).resolve()
    data["path"] = str(dataset_root)
    data["channels"] = GRAY_CHANNELS
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(
        "# Generated by gpr_detection.detect.yolo_train from "
        f"{Path(source_yaml).name}; adds channels: 1\n"
        + yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out_yaml


def build_gpr_trainer(
    overrides: Mapping[str, Any], augment_config: AugmentConfig, augment_seed: int
) -> Any:
    """Create an Ultralytics ``DetectionTrainer`` that applies the GPR augmentations.

    Args:
        overrides: Ultralytics train arguments (``model``, ``data``, ``imgsz``,
            ``epochs``, augmentation hyperparameters, ...).
        augment_config: Settings of the four GPR augmentations.
        augment_seed: Base seed of the augmentation generators.

    Returns:
        The trainer; call ``.train()`` to start.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer  # lazy (ADR-001)

    augment = YoloGprAugment(augment_config, augment_seed)

    class GprDetectionTrainer(DetectionTrainer):
        """DetectionTrainer whose train dataset starts with ``YoloGprAugment``."""

        def build_dataset(
            self, img_path: str, mode: str = "train", batch: int | None = None
        ) -> Any:
            dataset = super().build_dataset(img_path, mode, batch)
            if mode == "train":
                attach_gpr_augment(dataset, augment)
                LOGGER.info("GPR augmentation attached: %r", augment)
            return dataset

        def _close_dataloader_mosaic(self) -> None:
            super()._close_dataloader_mosaic()
            attach_gpr_augment(self.train_loader.dataset, augment)

    return GprDetectionTrainer(overrides=dict(overrides))
