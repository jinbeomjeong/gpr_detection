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

N2-a geometric augmentations (``docs/experiments/m31_n2a_retrain_plan.md``
v0.2 section 1-1-b, 5-2), only when a ``gpr_geometric`` config is given:

* :class:`YoloIsotropicScale` goes to index 1 (right after
  :class:`YoloGprAugment`, before the Ultralytics pre-transform block), because
  ``INTER_AREA`` shrinking and the top-centre anchor cannot be done by the
  Ultralytics affine step.
* :class:`YoloGeometricAugment` (horizontal shift, then edge fill band) goes
  right before ``Format``, so shift and band sizes are exact in the final
  input frame.

Train ``Compose`` order checked on the installed Ultralytics 8.4.192
(``v8_transforms`` + ``YOLODataset.build_transforms``, ``copy_paste_mode``
"flip")::

    YoloGprAugment -> YoloIsotropicScale
    -> Compose[Mosaic, CopyPaste, RandomPerspective] -> MixUp -> CutMix
    -> Albumentations -> RandomHSV -> RandomFlip(vertical)
    -> RandomFlip(horizontal) -> YoloGeometricAugment -> Format

With a geometric config the Ultralytics ``scale`` must be 0 (``ValueError``
otherwise), so the affine step only translates and the two scale stages never
multiply.

One-channel input uses the official ``channels: 1`` key of the data yaml: the
dataset then decodes with ``cv2.IMREAD_GRAYSCALE`` and ``DetectionModel`` is
built with ``ch=1``. :func:`write_training_data_yaml` adds that key to a copy
of the fold yaml, so the derived dataset under ``data/`` is never modified.

``ultralytics`` is imported only inside functions (ADR-001).
"""

import copy
import dataclasses
import logging
import os
import time
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gpr_detection.augment import AugmentConfig, augment_bscan
from gpr_detection.augment_geometry import (
    GeometricAugmentConfig,
    ShiftFillRecord,
    augment_scale,
    augment_shift_fill,
)
from gpr_detection.detect import aug_stats

LOGGER = logging.getLogger(__name__)

GRAY_CHANNELS = 1
_NO_WORKER_SEED = 0
"""Seed component used in the main process (``workers=0``)."""
SCALE_STREAM_ID = 1
SHIFT_FILL_STREAM_ID = 2
"""Third ``SeedSequence`` entry: keeps the geometric streams apart from
:class:`YoloGprAugment`, whose sequence is ``[seed, worker_seed]``."""
SCALE_STEP_INDEX = 1
"""Position of :class:`YoloIsotropicScale` in the train ``Compose``."""

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


# ------------------------------------------------------- geometric steps (N2-a)


def _gray_plane(image: np.ndarray, step_name: str) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == GRAY_CHANNELS:
        return image[..., 0]
    if image.ndim == 2:
        return image
    raise ValueError(
        f"{step_name} expects a one-channel image, got {image.shape}; "
        "set 'channels: 1' in the data yaml"
    )


def _class_ids(labels: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(labels["cls"]).reshape(-1)


def _replace_boxes(
    labels: dict[str, Any],
    boxes_xyxy: np.ndarray,
    keep_mask: np.ndarray,
    height_px: int,
    width_px: int,
) -> None:
    """Keep ``keep_mask`` labels, set their boxes, restore the original format.

    Raises:
        ValueError: If the instances carry segments or keypoints, which these
            steps do not transform.
    """
    instances = labels["instances"]
    if len(instances.segments) or instances.keypoints is not None:
        raise ValueError("geometric GPR steps support boxes only (no segments)")
    # Ultralytics 8.4.192 has no public getter for the box format.
    bbox_format = instances._bboxes.format
    was_normalized = instances.normalized
    box_dtype = instances.bboxes.dtype
    kept = instances[keep_mask]
    kept.convert_bbox(format="xyxy")
    kept.denormalize(width_px, height_px)
    kept.update(bboxes=boxes_xyxy[keep_mask].astype(box_dtype))
    if was_normalized:
        kept.normalize(width_px, height_px)
    kept.convert_bbox(format=bbox_format)
    labels["instances"] = kept
    labels["cls"] = labels["cls"][keep_mask]


class _GeometricStep:
    """Shared seeding, pickling and stats-file handling of the N2-a steps.

    Each process derives its generator from
    ``SeedSequence([seed, worker_seed, stream_id])`` and keeps its own label
    counts (reset when the step is pickled into a DataLoader worker).
    """

    stats_name = "geometric"
    stream_id = 0

    def __init__(self, config: GeometricAugmentConfig, seed: int) -> None:
        self.config = config
        self.seed = seed
        self.stats_dir: Path | None = None
        """Folder for the per-process stats file; ``None`` writes nothing."""
        self._reset_process_state()

    def _reset_process_state(self) -> None:
        self._generator: np.random.Generator | None = None
        self._process_id: int | None = None
        self._calls_since_flush = 0
        self._last_flush_s = time.monotonic()
        self._stats = self._new_stats()

    def _new_stats(self) -> dict[str, Any]:
        raise NotImplementedError

    def _stats_document(self) -> dict[str, Any]:
        raise NotImplementedError

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_generator"] = None  # every process seeds its own generator
        state["_process_id"] = None
        state["_stats"] = self._new_stats()
        state["_calls_since_flush"] = 0
        return state

    def __repr__(self) -> str:
        return f"{type(self).__name__}(seed={self.seed}, config={self.config})"

    def _rng(self) -> np.random.Generator:
        if self._process_id != os.getpid():
            self._reset_process_state()
            seed_sequence = np.random.SeedSequence(
                [self.seed, _data_loader_worker_seed(), self.stream_id]
            )
            self._generator = np.random.default_rng(seed_sequence)
            self._process_id = os.getpid()
        assert self._generator is not None
        return self._generator

    def _count_call(self) -> None:
        self._calls_since_flush += 1
        is_first_call = self._stats_calls() == 1
        is_due = (
            self._calls_since_flush >= aug_stats.FLUSH_EVERY_CALLS
            or time.monotonic() - self._last_flush_s >= aug_stats.FLUSH_EVERY_S
        )
        if is_first_call or is_due:
            self.flush_stats()

    def _stats_calls(self) -> int:
        return int(self._stats["calls"])

    def flush_stats(self) -> None:
        """Write this process's counts to ``stats_dir`` (no-op without a folder)."""
        self._calls_since_flush = 0
        self._last_flush_s = time.monotonic()
        if self.stats_dir is None or self._process_id is None:
            return
        path = self.stats_dir / f"{self.stats_name}_pid{self._process_id}.json"
        try:
            aug_stats.write_json_atomically(path, self._stats_document())
        except OSError as error:
            # Counts are a diagnostic; a locked file must not stop training.
            LOGGER.warning("could not write augmentation stats %s: %r", path, error)


class YoloIsotropicScale(_GeometricStep):
    """Ultralytics step: project isotropic scale (``augment_scale``), index 1.

    Args:
        config: Geometric settings; uses ``scale`` and ``keep_rule``.
        seed: Base seed, normally the training ``seed``.
    """

    stats_name = "scale"
    stream_id = SCALE_STREAM_ID

    def _new_stats(self) -> dict[str, Any]:
        scale = self.config.scale
        return {
            "calls": 0,
            "scale": aug_stats.StageStats(),
            "edges": aug_stats.scale_histogram_edges(scale.scale_min, scale.scale_max),
            "counts": [0] * aug_stats.SCALE_HISTOGRAM_BINS,
        }

    def _stats_document(self) -> dict[str, Any]:
        stats = self._stats
        return {
            "step": self.stats_name,
            "pid": self._process_id,
            "calls": stats["calls"],
            "stages": {"scale": stats["scale"].to_dict()},
            "scale_histogram": {"edges": stats["edges"], "counts": stats["counts"]},
        }

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Scale ``labels["img"]`` and its boxes; drop labels the rule rejects.

        Raises:
            ValueError: If the image is not one-channel.
        """
        image = labels["img"]
        gray = _gray_plane(image, type(self).__name__)
        height_px, width_px = gray.shape
        boxes = _boxes_xyxy_px(labels["instances"], height_px, width_px)
        classes = _class_ids(labels)
        rng = self._rng()
        scaled, new_boxes, keep_mask, s = augment_scale(gray, boxes, self.config, rng)

        stats = self._stats
        stats["calls"] += 1
        stats["scale"].record(s is not None, classes, keep_mask)
        if s is not None:
            stats["counts"][aug_stats.scale_histogram_bin(s, stats["edges"])] += 1
            labels["img"] = scaled.reshape(image.shape)
            _replace_boxes(labels, new_boxes, keep_mask, height_px, width_px)
        self._count_call()
        return labels


class YoloGeometricAugment(_GeometricStep):
    """Ultralytics step: horizontal shift then edge fill band, before ``Format``.

    Args:
        config: Geometric settings; uses ``shift``, ``edge_fill``, ``keep_rule``.
        seed: Base seed, normally the training ``seed``.
    """

    stats_name = "shift_fill"
    stream_id = SHIFT_FILL_STREAM_ID

    def _new_stats(self) -> dict[str, Any]:
        return {
            "calls": 0,
            "shift": aug_stats.StageStats(),
            "edge_fill": aug_stats.StageStats(),
            "edge_sides": {},
        }

    def _stats_document(self) -> dict[str, Any]:
        stats = self._stats
        return {
            "step": self.stats_name,
            "pid": self._process_id,
            "calls": stats["calls"],
            "stages": {
                "shift": stats["shift"].to_dict(),
                "edge_fill": stats["edge_fill"].to_dict(),
            },
            "edge_sides": dict(sorted(stats["edge_sides"].items())),
        }

    def _record(self, classes: np.ndarray, record: ShiftFillRecord) -> None:
        stats = self._stats
        stats["calls"] += 1
        stats["shift"].record(
            record.shift_dx_px is not None, classes, record.shift_keep_mask
        )
        stats["edge_fill"].record(
            record.edge_side is not None,
            classes,
            record.edge_keep_mask,
            considered_mask=record.shift_keep_mask,
        )
        if record.edge_side is not None:
            side = str(record.edge_side)
            stats["edge_sides"][side] = stats["edge_sides"].get(side, 0) + 1

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Shift and band ``labels["img"]``; drop labels the rule rejects.

        Raises:
            ValueError: If the image is not one-channel.
        """
        image = labels["img"]
        gray = _gray_plane(image, type(self).__name__)
        height_px, width_px = gray.shape
        boxes = _boxes_xyxy_px(labels["instances"], height_px, width_px)
        classes = _class_ids(labels)
        rng = self._rng()
        result, new_boxes, keep_mask, record = augment_shift_fill(
            gray, boxes, self.config, rng
        )
        self._record(classes, record)
        if record.shift_dx_px is not None or record.edge_side is not None:
            labels["img"] = result.reshape(image.shape)
            _replace_boxes(labels, new_boxes, keep_mask, height_px, width_px)
        self._count_call()
        return labels


def _format_step_index(steps: list[Any]) -> int:
    from ultralytics.data.augment import Format  # lazy (ADR-001)

    indices = [index for index, step in enumerate(steps) if isinstance(step, Format)]
    if not indices:
        raise ValueError("train transforms have no Ultralytics Format step")
    return indices[-1]


def attach_geometric_augment(
    dataset: Any, scale_step: YoloIsotropicScale, shift_fill_step: YoloGeometricAugment
) -> None:
    """Insert the two N2-a steps once: scale at index 1, shift/fill before Format.

    Call after :func:`attach_gpr_augment`.

    Raises:
        ValueError: If ``YoloGprAugment`` is not the first step or there is no
            ``Format`` step.
    """
    transforms = dataset.transforms
    steps = transforms.transforms
    if not steps or not isinstance(steps[0], YoloGprAugment):
        raise ValueError("attach_gpr_augment must run before attach_geometric_augment")
    if not any(isinstance(step, YoloIsotropicScale) for step in steps):
        transforms.insert(SCALE_STEP_INDEX, scale_step)
    if not any(isinstance(step, YoloGeometricAugment) for step in steps):
        transforms.insert(_format_step_index(transforms.transforms), shift_fill_step)


def _plain(value: Any) -> Any:
    """Dataclasses, enums and tuples as JSON/YAML-friendly values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {key: _plain(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def geometric_settings(config: GeometricAugmentConfig) -> dict[str, Any]:
    """``gpr_geometric`` config as a plain dict (for settings files)."""
    return _plain(config)


def check_single_scale_stage(
    overrides: Mapping[str, Any], geometric: GeometricAugmentConfig | None
) -> None:
    """With the project scale on, the Ultralytics ``scale`` must be exactly 0.

    A missing ``scale`` key counts as non-zero (Ultralytics default 0.5).

    Raises:
        ValueError: If both scale stages would run (factors would multiply).
    """
    if geometric is None or geometric.scale.probability == 0:
        return
    if overrides.get("scale") != 0:
        raise ValueError(
            f"Ultralytics scale={overrides.get('scale')!r} must be 0 when "
            "gpr_geometric.scale is on (double scaling; "
            "docs/experiments/m31_n2a_retrain_plan.md section 1-1-b)"
        )


def check_augmentation_policy(
    overrides: Mapping[str, Any], geometric: GeometricAugmentConfig | None = None
) -> list[str]:
    """Validate Ultralytics augmentation arguments against the project policy.

    Args:
        overrides: Ultralytics train arguments.
        geometric: N2-a geometric settings, or ``None`` when they are off.

    Returns:
        Warnings for ablation-only settings and for ``hsv_v``, which Ultralytics
        8.4.192 silently skips on one-channel images (``RandomHSV`` only
        touches 3-channel input).

    Raises:
        ValueError: If a forbidden augmentation is non-zero or missing, or the
            project scale is on while the Ultralytics ``scale`` is not 0.
    """
    check_single_scale_stage(overrides, geometric)
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


def _stats_header(trainer: Any, seed: int, geometric: GeometricAugmentConfig) -> dict:
    names = trainer.data.get("names", {}) if isinstance(trainer.data, dict) else {}
    return {
        "seed": seed,
        "class_names": {str(key): str(value) for key, value in dict(names).items()},
        "gpr_geometric": geometric_settings(geometric),
    }


def build_gpr_trainer(
    overrides: Mapping[str, Any],
    augment_config: AugmentConfig,
    augment_seed: int,
    geometric_config: GeometricAugmentConfig | None = None,
) -> Any:
    """Create an Ultralytics ``DetectionTrainer`` that applies the GPR augmentations.

    Args:
        overrides: Ultralytics train arguments (``model``, ``data``, ``imgsz``,
            ``epochs``, augmentation hyperparameters, ...).
        augment_config: Settings of the four GPR augmentations.
        augment_seed: Base seed of the augmentation generators.
        geometric_config: N2-a geometric settings; ``None`` (default) inserts no
            geometric step, which reproduces the M1-10 transforms exactly.

    Returns:
        The trainer; call ``.train()`` to start. With a geometric config it
        writes ``aug_stats.json`` into the run folder after every epoch.

    Raises:
        ValueError: If the geometric scale is on and Ultralytics ``scale`` != 0.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer  # lazy (ADR-001)

    check_single_scale_stage(overrides, geometric_config)
    augment = YoloGprAugment(augment_config, augment_seed)
    geometric_steps: tuple[YoloIsotropicScale, YoloGeometricAugment] | None = None
    if geometric_config is not None:
        geometric_steps = (
            YoloIsotropicScale(geometric_config, augment_seed),
            YoloGeometricAugment(geometric_config, augment_seed),
        )

    class GprDetectionTrainer(DetectionTrainer):
        """DetectionTrainer whose train dataset starts with ``YoloGprAugment``."""

        def _attach_project_steps(self, dataset: Any) -> None:
            attach_gpr_augment(dataset, augment)
            if geometric_steps is None:
                return
            # Second guard on the resolved arguments the dataset was built with.
            check_single_scale_stage(vars(self.args), geometric_config)
            stats_dir = Path(self.save_dir) / aug_stats.WORKER_STATS_DIR_NAME
            stats_dir.mkdir(parents=True, exist_ok=True)
            for step in geometric_steps:
                step.stats_dir = stats_dir
            attach_geometric_augment(dataset, *geometric_steps)

        def build_dataset(
            self, img_path: str, mode: str = "train", batch: int | None = None
        ) -> Any:
            dataset = super().build_dataset(img_path, mode, batch)
            if mode == "train":
                self._attach_project_steps(dataset)
                LOGGER.info("GPR augmentation attached: %r", augment)
                if geometric_steps is not None:
                    LOGGER.info("GPR geometric steps attached: %r", geometric_steps)
            return dataset

        def _close_dataloader_mosaic(self) -> None:
            super()._close_dataloader_mosaic()
            self._attach_project_steps(self.train_loader.dataset)

    trainer = GprDetectionTrainer(overrides=dict(overrides))
    if geometric_config is not None and geometric_steps is not None:

        def write_aug_stats(active_trainer: Any) -> None:
            aug_stats.write_run_stats(
                Path(active_trainer.save_dir),
                epochs_completed=int(active_trainer.epoch) + 1,
                header=_stats_header(active_trainer, augment_seed, geometric_config),
                flushable_steps=geometric_steps,
            )

        trainer.add_callback("on_train_epoch_end", write_aug_stats)
    return trainer
