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

N2-a' (``docs/experiments/m31_n2a2_retrain_plan.md`` v1.0 section 2-1, 7-1),
only when a ``gpr_context`` config and its :class:`ContextIndex` are given:
:class:`YoloContextScale` takes the place of :class:`YoloIsotropicScale` at
index 1 (the two never run together) and :class:`YoloGeometricAugment` stays
before ``Format``::

    YoloGprAugment -> YoloContextScale -> ... -> RandomFlip(horizontal)
    -> YoloGeometricAugment -> Format

The context step reads neighbour crops next to ``labels["im_file"]`` (image
with ``cv2.IMREAD_GRAYSCALE``, label txt at the Ultralytics label path).

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
from collections import Counter
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from gpr_detection.augment import AugmentConfig, augment_bscan
from gpr_detection.augment_geometry import (
    PIPE_BAND_CLASS_ID,
    BoxKeepRule,
    ContextIndex,
    ContextMode,
    ContextRecord,
    ContextScaleConfig,
    CropContent,
    GeometricAugmentConfig,
    ShiftFillRecord,
    augment_context_scale,
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
CONTEXT_STREAM_ID = 3
"""Stream of :class:`YoloContextScale` (N2-a' plan section 7-1, k = 3)."""
SCALE_STEP_INDEX = 1
"""Position of :class:`YoloIsotropicScale` / :class:`YoloContextScale`."""
PIPE_BAND_NAME = "pipe_band"
YOLO_LABEL_FIELDS = 5
"""``class x_center y_center width height`` (normalised)."""
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
S_HISTOGRAM_BINS = 10
DX_HISTOGRAM_BINS = 8
DX_SPAN_HISTOGRAM_BINS = 6

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


# ------------------------------------------------- context step (N2-a')


def _unique_rows_in_order(rows: list[list[float]]) -> list[list[float]]:
    """Drop repeated label lines, as Ultralytics does when it caches labels."""
    seen: set[tuple[float, ...]] = set()
    unique = []
    for row in rows:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


class FolderCropLoader:
    """Reads neighbour crops from the folder of the centre image.

    Images with ``cv2.IMREAD_GRAYSCALE`` (as the one-channel dataset does),
    labels from the Ultralytics label path (``/images/`` -> ``/labels/``,
    ``.txt``) as xyxy px of a ``width_px`` x ``height_px`` crop.

    Args:
        image_dir: Folder of the centre image.
        width_px: Crop width.
        height_px: Crop height.
    """

    def __init__(self, image_dir: Path, width_px: int, height_px: int) -> None:
        self.image_dir = Path(image_dir)
        self.width_px = width_px
        self.height_px = height_px

    def image(self, file_name: str) -> np.ndarray:
        """Grayscale uint8 image of a crop.

        Raises:
            FileNotFoundError: If OpenCV cannot read the file.
        """
        path = self.image_dir / file_name
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"cannot read context crop {path}")
        return image

    def labels(self, file_name: str) -> tuple[np.ndarray, np.ndarray]:
        """``(boxes [N, 4] xyxy px, classes [N])``; no label file = no objects."""
        from ultralytics.data.utils import img2label_paths  # lazy (ADR-001)

        label_path = Path(img2label_paths([str(self.image_dir / file_name)])[0])
        rows = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == YOLO_LABEL_FIELDS:
                    rows.append([float(value) for value in parts])
        table = np.array(_unique_rows_in_order(rows), dtype=np.float64).reshape(-1, 5)
        scale = np.array([self.width_px, self.height_px])
        centre = table[:, 1:3] * scale
        half_size = table[:, 3:5] * scale / 2
        boxes = np.concatenate([centre - half_size, centre + half_size], axis=1)
        return boxes, table[:, 0].astype(np.int64)


def _set_boxes(
    labels: dict[str, Any],
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    height_px: int,
    width_px: int,
) -> None:
    """Replace all labels by ``boxes_xyxy``/``classes``, keeping the box format.

    Unlike :func:`_replace_boxes` the number of boxes may grow (labels merged
    from neighbour crops).

    Raises:
        ValueError: If the instances carry segments or keypoints.
    """
    instances = labels["instances"]
    if len(instances.segments) or instances.keypoints is not None:
        raise ValueError("geometric GPR steps support boxes only (no segments)")
    # Ultralytics 8.4.192 has no public getter for the box format.
    bbox_format = instances._bboxes.format
    was_normalized = instances.normalized
    box_dtype = instances.bboxes.dtype
    rebuilt = instances[np.zeros(len(instances), dtype=bool)]
    rebuilt.convert_bbox(format="xyxy")
    rebuilt.denormalize(width_px, height_px)
    rebuilt.update(bboxes=np.asarray(boxes_xyxy, dtype=box_dtype).reshape(-1, 4))
    if was_normalized:
        rebuilt.normalize(width_px, height_px)
    rebuilt.convert_bbox(format=bbox_format)
    labels["instances"] = rebuilt
    labels["cls"] = np.asarray(classes).astype(labels["cls"].dtype).reshape(-1, 1)


def _histogram(low: float, high: float, bins: int) -> dict[str, Any]:
    return {"edges": aug_stats.histogram_edges(low, high, bins), "counts": [0] * bins}


def _count_in(histogram: dict[str, Any], value: float) -> None:
    histogram["counts"][aug_stats.histogram_bin(value, histogram["edges"])] += 1


def _new_band_k32_stats() -> dict[str, Any]:
    """Counts of the N2-a″ rule K32 (plan m31_n2a3 §3-2), one process."""
    return {
        "dropped": 0,
        "by_origin": {name: 0 for name in aug_stats.BAND_K32_ORIGIN_NAMES},
        "width_bins": {name: 0 for name in aug_stats.BAND_K32_WIDTH_BIN_NAMES},
    }


def _count_band_k32(stats: dict[str, Any], record: ContextRecord) -> None:
    for width_px, origin in record.band_k32_dropped:
        stats["dropped"] += 1
        stats["by_origin"][aug_stats.BAND_K32_ORIGIN_NAMES[origin]] += 1
        width_bin = aug_stats.band_k32_width_bin(width_px)
        stats["width_bins"][width_bin] = stats["width_bins"].get(width_bin, 0) + 1


class YoloContextScale(_GeometricStep):
    """Ultralytics step: N2-a' context scale (``augment_context_scale``), index 1.

    With the N2-a″ rule K32 on (``band_min_clipped_width_px > 0``) the stats
    file also has a ``band_k32`` block (dropped boxes by origin and width).

    Args:
        config: The ``gpr_context`` settings.
        index: Context index of this run (start-up leak checks already done).
        seed: Base seed, normally the training ``seed``.
        keep_rule: Box keep rule (``gpr_geometric.keep_rule``).
    """

    stats_name = "context_scale"
    stream_id = CONTEXT_STREAM_ID

    def __init__(
        self,
        config: ContextScaleConfig,
        index: ContextIndex,
        seed: int,
        keep_rule: BoxKeepRule | None = None,
    ) -> None:
        self.index = index
        self.keep_rule = keep_rule if keep_rule is not None else BoxKeepRule()
        super().__init__(config, seed)  # type: ignore[arg-type]

    @property
    def _is_k32_on(self) -> bool:
        return self.config.band_min_clipped_width_px > 0

    def _new_stats(self) -> dict[str, Any]:
        config = self.config
        max_shift_px = config.max_shift_px
        stats: dict[str, Any] = {
            "calls": 0,
            "context_scale": aug_stats.StageStats(),
            "counters": {
                "mode": Counter(),
                "source": Counter(),
                "used_neighbors": Counter(),
                "chain_stops": Counter(),
            },
            "histograms": {
                "s_context": _histogram(*config.context_scale, S_HISTOGRAM_BINS),
                "s_plain": _histogram(*config.plain_scale, S_HISTOGRAM_BINS),
                "dx_px": _histogram(-max_shift_px, max_shift_px, DX_HISTOGRAM_BINS),
                "dx_span_px": _histogram(0, 2 * max_shift_px, DX_SPAN_HISTOGRAM_BINS),
            },
        }
        if self._is_k32_on:
            stats[aug_stats.BAND_K32_KEY] = _new_band_k32_stats()
        return stats

    def _stats_document(self) -> dict[str, Any]:
        stats = self._stats
        document = {
            "step": self.stats_name,
            "pid": self._process_id,
            "calls": stats["calls"],
            "stages": {"context_scale": stats["context_scale"].to_dict()},
            "counters": {
                name: dict(sorted(counts.items()))
                for name, counts in stats["counters"].items()
            },
            "histograms": copy.deepcopy(stats["histograms"]),
        }
        # Only with K32 on, so N2-a' stats files keep exactly their old keys.
        if self._is_k32_on:
            document[aug_stats.BAND_K32_KEY] = copy.deepcopy(
                stats[aug_stats.BAND_K32_KEY]
            )
        return document

    def _record(
        self, classes: np.ndarray, keep_mask: np.ndarray, record: ContextRecord
    ) -> None:
        stats = self._stats
        stats["calls"] += 1
        stats["context_scale"].record(True, classes, keep_mask)
        window = record.window
        counters = stats["counters"]
        counters["mode"][str(window.mode)] += 1
        counters["source"][str(record.source)] += 1
        counters["used_neighbors"][str(record.used_neighbors)] += 1
        for reason in record.rejections:
            counters["chain_stops"][str(reason)] += 1
        histograms = stats["histograms"]
        if window.mode is ContextMode.CONTEXT:
            _count_in(histograms["s_context"], window.s)
            _count_in(histograms["dx_px"], window.dx_px)
            _count_in(histograms["dx_span_px"], window.dx_max_px - window.dx_min_px)
        else:
            _count_in(histograms["s_plain"], window.s)
        if self._is_k32_on:
            _count_band_k32(stats[aug_stats.BAND_K32_KEY], record)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Scale ``labels["img"]`` with context; merge and filter its labels.

        Raises:
            ValueError: If the image is not one-channel, or on a context leak.
        """
        image = labels["img"]
        gray = _gray_plane(image, type(self).__name__)
        height_px, width_px = gray.shape
        boxes = _boxes_xyxy_px(labels["instances"], height_px, width_px)
        image_path = Path(labels["im_file"])
        loader = FolderCropLoader(image_path.parent, width_px, height_px)
        result, new_boxes, new_classes, keep_mask, record = augment_context_scale(
            image_path.name,
            CropContent(gray, boxes, _class_ids(labels)),
            self.index,
            loader,
            self.config,
            self.keep_rule,
            self._rng(),
        )
        self._record(new_classes, keep_mask, record)
        labels["img"] = result.reshape(image.shape)
        _set_boxes(
            labels, new_boxes[keep_mask], new_classes[keep_mask], height_px, width_px
        )
        self._count_call()
        return labels


def attach_context_augment(
    dataset: Any, context_step: YoloContextScale, shift_fill_step: YoloGeometricAugment
) -> None:
    """Insert the N2-a' steps once: context scale at index 1, shift/fill before Format.

    Call after :func:`attach_gpr_augment`.

    Raises:
        ValueError: If ``YoloGprAugment`` is not the first step, the N2-a scale
            step is present (double scaling), or there is no ``Format`` step.
    """
    transforms = dataset.transforms
    steps = transforms.transforms
    if not steps or not isinstance(steps[0], YoloGprAugment):
        raise ValueError("attach_gpr_augment must run before attach_context_augment")
    if any(isinstance(step, YoloIsotropicScale) for step in steps):
        raise ValueError("YoloIsotropicScale and YoloContextScale must not both run")
    if not any(isinstance(step, YoloContextScale) for step in steps):
        transforms.insert(SCALE_STEP_INDEX, context_step)
    if not any(isinstance(step, YoloGeometricAugment) for step in steps):
        transforms.insert(_format_step_index(transforms.transforms), shift_fill_step)


def _image_names(entry: str | Path, dataset_root: Path) -> list[str]:
    """Image file names of one data-yaml entry (a list file or an image folder)."""
    path = Path(entry)
    if not path.is_absolute():
        path = dataset_root / path
    if path.is_dir():
        return sorted(
            item.name
            for item in path.rglob("*")
            if item.suffix.lower() in IMAGE_SUFFIXES
        )
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return [Path(line.strip()).name for line in lines if line.strip()]


def _split_image_names(data: Mapping[str, Any], split: str, root: Path) -> list[str]:
    entries = data.get(split) or []
    if not isinstance(entries, list):
        entries = [entries]
    names: list[str] = []
    for entry in entries:
        names.extend(_image_names(entry, root))
    return names


def context_index_for_data_yaml(index_dir: Path, data_yaml: Path) -> ContextIndex:
    """Build the run's :class:`ContextIndex` from an Ultralytics data yaml.

    The ``train`` lists give the context candidates, the ``val`` list the
    folds no neighbour may come from (plan section 4).

    Raises:
        ValueError: On a start-up leak assertion (see :class:`ContextIndex`).
    """
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(data.get("path") or Path(data_yaml).parent)
    if not root.is_absolute():
        root = (Path(data_yaml).parent / root).resolve()
    return ContextIndex.from_csv(
        index_dir,
        _split_image_names(data, "train", root),
        _split_image_names(data, "val", root),
    )


def check_context_names(names: Mapping[Any, Any]) -> None:
    """The band label merge needs ``pipe_band`` at its project class id.

    Raises:
        ValueError: If class ``PIPE_BAND_CLASS_ID`` is not ``pipe_band``.
    """
    name = dict(names).get(PIPE_BAND_CLASS_ID)
    if name != PIPE_BAND_NAME:
        raise ValueError(
            f"class {PIPE_BAND_CLASS_ID} must be {PIPE_BAND_NAME!r} for the context "
            f"label merge, got {name!r}"
        )


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
    overrides: Mapping[str, Any],
    geometric: GeometricAugmentConfig | None,
    context: ContextScaleConfig | None = None,
) -> None:
    """With a project scale on, the Ultralytics ``scale`` must be exactly 0.

    A missing ``scale`` key counts as non-zero (Ultralytics default 0.5). The
    project scale is ``gpr_geometric.scale`` (N2-a) or ``gpr_context`` (N2-a').

    Raises:
        ValueError: If two scale stages would run (factors would multiply).
    """
    is_geometric_scale_on = geometric is not None and geometric.scale.probability != 0
    if context is not None and is_geometric_scale_on:
        raise ValueError(
            "gpr_context and gpr_geometric.scale are both on (double scaling; "
            "docs/experiments/m31_n2a2_retrain_plan.md section 7-1)"
        )
    if not is_geometric_scale_on and context is None:
        return
    if overrides.get("scale") == 0:
        return
    if context is not None:
        raise ValueError(
            f"Ultralytics scale={overrides.get('scale')!r} must be 0 when "
            "gpr_context is on (double scaling; "
            "docs/experiments/m31_n2a2_retrain_plan.md section 7-1)"
        )
    raise ValueError(
        f"Ultralytics scale={overrides.get('scale')!r} must be 0 when "
        "gpr_geometric.scale is on (double scaling; "
        "docs/experiments/m31_n2a_retrain_plan.md section 1-1-b)"
    )


def check_augmentation_policy(
    overrides: Mapping[str, Any],
    geometric: GeometricAugmentConfig | None = None,
    context: ContextScaleConfig | None = None,
) -> list[str]:
    """Validate Ultralytics augmentation arguments against the project policy.

    Args:
        overrides: Ultralytics train arguments.
        geometric: N2-a geometric settings, or ``None`` when they are off.
        context: N2-a' context settings, or ``None`` when they are off.

    Returns:
        Warnings for ablation-only settings and for ``hsv_v``, which Ultralytics
        8.4.192 silently skips on one-channel images (``RandomHSV`` only
        touches 3-channel input).

    Raises:
        ValueError: If a forbidden augmentation is non-zero or missing, or the
            project scale is on while the Ultralytics ``scale`` is not 0.
    """
    check_single_scale_stage(overrides, geometric, context)
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


def _stats_header(
    trainer: Any,
    seed: int,
    geometric: GeometricAugmentConfig,
    context: tuple[ContextScaleConfig, ContextIndex] | None = None,
) -> dict:
    names = trainer.data.get("names", {}) if isinstance(trainer.data, dict) else {}
    header = {
        "seed": seed,
        "class_names": {str(key): str(value) for key, value in dict(names).items()},
        "gpr_geometric": geometric_settings(geometric),
    }
    if context is not None:
        context_config, context_index = context
        header["gpr_context"] = _plain(context_config)
        header["gpr_context_index"] = context_index.summary()
    return header


def _check_context_arguments(
    geometric_config: GeometricAugmentConfig | None,
    context_config: ContextScaleConfig | None,
    context_index: ContextIndex | None,
) -> None:
    if (context_config is None) != (context_index is None):
        raise ValueError("context_config and context_index must be given together")
    if context_config is not None and geometric_config is None:
        raise ValueError(
            "gpr_context needs the gpr_geometric block (shift, edge fill, keep rule)"
        )


def check_context_without_gpr_noise(
    augment_config: AugmentConfig, context_config: ContextScaleConfig | None
) -> None:
    """With ``gpr_context`` on, every GPR augmentation intensity must be 0.

    The GPR noise falls on the centre crop only, so it would blur the
    brightness reference of the stitched neighbours (plan v1.0 section 7-5 (3)).

    Raises:
        ValueError: If ``context_config`` is set and an intensity is > 0.
    """
    if context_config is None:
        return
    intensities = {
        "gaussian_noise.sigma_max": augment_config.gaussian_noise.sigma_max,
        "contrast_gamma.contrast_max_delta": (
            augment_config.contrast_gamma.contrast_max_delta
        ),
        "contrast_gamma.gamma_max_delta": augment_config.contrast_gamma.gamma_max_delta,
        "ringing.amplitude_max": augment_config.ringing.amplitude_max,
        "trace_dropout.fraction_max": augment_config.trace_dropout.fraction_max,
    }
    active = sorted(name for name, value in intensities.items() if value > 0)
    if active:
        raise ValueError(
            f"gpr_context needs the GPR augmentation strength 0, got {active} > 0 "
            "(docs/experiments/m31_n2a2_retrain_plan.md section 7-5)"
        )


def _project_steps(
    augment_seed: int,
    geometric_config: GeometricAugmentConfig | None,
    context: tuple[ContextScaleConfig, ContextIndex] | None,
) -> tuple[Any, YoloGeometricAugment] | None:
    """The two geometric steps (index-1 scale step, shift/fill) or ``None``."""
    if geometric_config is None:
        return None
    shift_fill_step = YoloGeometricAugment(geometric_config, augment_seed)
    if context is None:
        return YoloIsotropicScale(geometric_config, augment_seed), shift_fill_step
    context_config, context_index = context
    context_step = YoloContextScale(
        context_config, context_index, augment_seed, geometric_config.keep_rule
    )
    return context_step, shift_fill_step


def build_gpr_trainer(
    overrides: Mapping[str, Any],
    augment_config: AugmentConfig,
    augment_seed: int,
    geometric_config: GeometricAugmentConfig | None = None,
    context_config: ContextScaleConfig | None = None,
    context_index: ContextIndex | None = None,
) -> Any:
    """Create an Ultralytics ``DetectionTrainer`` that applies the GPR augmentations.

    Args:
        overrides: Ultralytics train arguments (``model``, ``data``, ``imgsz``,
            ``epochs``, augmentation hyperparameters, ...).
        augment_config: Settings of the four GPR augmentations.
        augment_seed: Base seed of the augmentation generators.
        geometric_config: N2-a geometric settings; ``None`` (default) inserts no
            geometric step, which reproduces the M1-10 transforms exactly.
        context_config: N2-a' context settings; with ``context_index`` they
            put :class:`YoloContextScale` at index 1 instead of
            :class:`YoloIsotropicScale` (needs ``geometric_config`` with its
            scale off). ``None`` (default) keeps the N2-a transforms.
        context_index: The run's context index (built from the same data yaml).

    Returns:
        The trainer; call ``.train()`` to start. With a geometric config it
        writes ``aug_stats.json`` into the run folder after every epoch.

    Raises:
        ValueError: If two scale stages would run, the Ultralytics ``scale`` is
            not 0 with a project scale on, or the context arguments are
            incomplete.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer  # lazy (ADR-001)

    _check_context_arguments(geometric_config, context_config, context_index)
    check_context_without_gpr_noise(augment_config, context_config)
    check_single_scale_stage(overrides, geometric_config, context_config)
    augment = YoloGprAugment(augment_config, augment_seed)
    context = None
    if context_config is not None and context_index is not None:
        context = (context_config, context_index)
    geometric_steps = _project_steps(augment_seed, geometric_config, context)

    class GprDetectionTrainer(DetectionTrainer):
        """DetectionTrainer whose train dataset starts with ``YoloGprAugment``."""

        def _attach_project_steps(self, dataset: Any) -> None:
            attach_gpr_augment(dataset, augment)
            if geometric_steps is None:
                return
            # Second guard on the resolved arguments the dataset was built with.
            check_single_scale_stage(vars(self.args), geometric_config, context_config)
            stats_dir = Path(self.save_dir) / aug_stats.WORKER_STATS_DIR_NAME
            stats_dir.mkdir(parents=True, exist_ok=True)
            for step in geometric_steps:
                step.stats_dir = stats_dir
            if context is None:
                attach_geometric_augment(dataset, *geometric_steps)
                return
            check_context_names(dataset.data.get("names", {}))
            attach_context_augment(dataset, *geometric_steps)

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
                header=_stats_header(
                    active_trainer, augment_seed, geometric_config, context
                ),
                flushable_steps=geometric_steps,
            )

        trainer.add_callback("on_train_epoch_end", write_aug_stats)
    return trainer
