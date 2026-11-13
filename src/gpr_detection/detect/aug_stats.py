"""Label counts of the geometric training augmentations (``aug_stats.json``).

Plan ``docs/experiments/m31_n2a_retrain_plan.md`` section 5-2: how many images
each stage touched, the drawn scale factors (10-bin histogram), the band edges
and, per class, how many boxes a stage kept or dropped. These are augmentation
counts, not detection metrics, so the smoke run may open them.

DataLoader workers are separate processes, so every process keeps its own
counts and writes them to ``<run>/aug_stats_workers/<step>_pid<pid>.json``
(every :data:`FLUSH_EVERY_CALLS` images or :data:`FLUSH_EVERY_S` seconds). The
main process sums those files into ``<run>/aug_stats.json`` at the end of each
epoch. Counts of worker processes therefore lag by fewer than
``FLUSH_EVERY_CALLS`` images per worker; main-process steps are flushed first,
so ``workers=0`` counts are exact.

No Ultralytics import here.
"""

import json
import logging
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

AUG_STATS_FILE_NAME = "aug_stats.json"
WORKER_STATS_DIR_NAME = "aug_stats_workers"
AUG_STATS_SCHEMA = "gpr_aug_stats/1"
FLUSH_EVERY_CALLS = 64
FLUSH_EVERY_S = 10.0
SCALE_HISTOGRAM_BINS = 10
NOT_A_METRIC_NOTE = (
    "augmentation label counts (not a detection metric); worker counts lag by "
    f"< {FLUSH_EVERY_CALLS} images per worker process"
)


@dataclass
class StageStats:
    """Counts of one augmentation stage in one process."""

    images: int = 0
    applied: int = 0
    boxes_kept_by_class: Counter[str] = field(default_factory=Counter)
    boxes_dropped_by_class: Counter[str] = field(default_factory=Counter)

    def record(
        self,
        is_applied: bool,
        classes: np.ndarray,
        keep_mask: np.ndarray | None,
        considered_mask: np.ndarray | None = None,
    ) -> None:
        """Count one image.

        Args:
            is_applied: Whether the stage ran on this image.
            classes: ``[N]`` class ids of the boxes entering the image's steps.
            keep_mask: ``[N]`` keep mask of this stage (``None`` if skipped).
            considered_mask: ``[N]`` boxes still alive when the stage ran
                (``None`` = all); dropped-before boxes are not counted again.
        """
        self.images += 1
        if not is_applied or keep_mask is None:
            return
        self.applied += 1
        alive = np.ones(len(classes), dtype=bool)
        if considered_mask is not None:
            alive = considered_mask
        for class_id, is_alive, is_kept in zip(classes, alive, keep_mask, strict=True):
            if not is_alive:
                continue
            target = (
                self.boxes_kept_by_class if is_kept else self.boxes_dropped_by_class
            )
            target[str(int(class_id))] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "applied": self.applied,
            "boxes_kept_by_class": dict(sorted(self.boxes_kept_by_class.items())),
            "boxes_dropped_by_class": dict(sorted(self.boxes_dropped_by_class.items())),
        }


def scale_histogram_edges(scale_min: float, scale_max: float) -> list[float]:
    """Edges of the 10 equal bins over ``[scale_min, scale_max]``."""
    edges = np.linspace(scale_min, scale_max, SCALE_HISTOGRAM_BINS + 1)
    return [float(edge) for edge in edges]


def scale_histogram_bin(s: float, edges: list[float]) -> int:
    """Bin index of ``s``; the top edge belongs to the last bin."""
    index = int(np.searchsorted(edges, s, side="right")) - 1
    return min(max(index, 0), SCALE_HISTOGRAM_BINS - 1)


def write_json_atomically(path: Path, document: Mapping[str, Any]) -> None:
    """Write JSON through a temporary file so readers never see half a file."""
    temporary = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _add_stage(total: dict[str, Any], stage: Mapping[str, Any]) -> None:
    total["images"] += stage["images"]
    total["applied"] += stage["applied"]
    for key in ("boxes_kept_by_class", "boxes_dropped_by_class"):
        total[key].update(stage[key])


def _empty_stage_total() -> dict[str, Any]:
    return {
        "images": 0,
        "applied": 0,
        "boxes_kept_by_class": Counter(),
        "boxes_dropped_by_class": Counter(),
    }


def _read_worker_files(worker_dir: Path) -> list[dict[str, Any]]:
    documents = []
    for path in sorted(worker_dir.glob("*_pid*.json")):
        try:
            documents.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            # A worker may be replacing the file right now; next epoch reads it.
            LOGGER.warning("skipping augmentation stats file %s: %r", path, error)
    return documents


def aggregate_worker_stats(worker_dir: Path) -> dict[str, Any]:
    """Sum every per-process stats file under ``worker_dir``.

    Returns:
        ``{"worker_files", "stages", "scale_histogram", "edge_sides"}``.
    """
    documents = _read_worker_files(worker_dir)
    stages: dict[str, dict[str, Any]] = {}
    histogram_edges: list[float] | None = None
    histogram_counts = np.zeros(SCALE_HISTOGRAM_BINS, dtype=int)
    edge_sides: Counter[str] = Counter()
    for document in documents:
        for name, stage in document["stages"].items():
            _add_stage(stages.setdefault(name, _empty_stage_total()), stage)
        if "scale_histogram" in document:
            histogram_edges = document["scale_histogram"]["edges"]
            histogram_counts += np.asarray(document["scale_histogram"]["counts"])
        edge_sides.update(document.get("edge_sides", {}))
    for stage in stages.values():
        for key in ("boxes_kept_by_class", "boxes_dropped_by_class"):
            stage[key] = dict(sorted(stage[key].items()))
    return {
        "worker_files": len(documents),
        "stages": stages,
        "scale_histogram": {
            "edges": histogram_edges,
            "counts": [int(count) for count in histogram_counts],
        },
        "edge_sides": dict(sorted(edge_sides.items())),
    }


def write_run_stats(
    run_dir: Path,
    epochs_completed: int,
    header: Mapping[str, Any],
    flushable_steps: Iterable[Any] = (),
) -> Path:
    """Flush main-process steps, sum all process files, write ``aug_stats.json``.

    Args:
        run_dir: Training run folder (``trainer.save_dir``).
        epochs_completed: Epochs finished so far (1-based).
        header: Fixed fields to copy in (seed, settings, class names).
        flushable_steps: Steps living in this process (``flush_stats()``).

    Returns:
        Path of the written file.
    """
    for step in flushable_steps:
        step.flush_stats()
    document = {
        "schema": AUG_STATS_SCHEMA,
        "note": NOT_A_METRIC_NOTE,
        "epochs_completed": epochs_completed,
        **header,
        **aggregate_worker_stats(run_dir / WORKER_STATS_DIR_NAME),
    }
    out_path = run_dir / AUG_STATS_FILE_NAME
    write_json_atomically(out_path, document)
    return out_path
