"""M3-2 S5 (stage C, measurements M3-M6): recipe runner on real lines.

Spec: ``docs/processing/m32_image_quality_plan.md`` v0.2.3 (sections 3-2, 3-3,
4-3, 4-4, 5-1, 5-3 S5) and ``docs/experiments/m32_cv_review.md`` section 4-2
and appendix B (the cv requirements).

Stages (plan 4-4, each a separate call)

1. ``--stage uninjected`` (P5a): every recipe (r0, r5, r6) x fold (N2-a'
   ``last.pt``, sha256 asserted) on the stage-A backgrounds (4TU 150, IDS 60)
   and the K bundle (27). Writes the merged L0 boxes (any class, zone (c')),
   M4a-M4d, the ``pipe_hyperbola`` box count per conf level (C-5), M5 flip
   pairs and the M6 apex hits. First a separate gate step runs r0 x K x folds
   at batch size 1 and must reproduce D3 exactly (amendment 5, [CVR] C-4);
   the measurement then runs every recipe on the batched path and the
   batch-1 / batched difference on r0 x K is reported only. D3 values:
   90/205, any fold 30/41, fold 0 15/41.
2. ``--stage plan``: union of the P5a boxes -> zone (c') -> one seeded plan
   per background line (``gpr_detection.eval.inject``) -> injection table,
   exclusions, cost table (``long_layer_runs`` included) and the F2 record
   draft. The stratum level ``l_W`` (median width nearest 11.36 P), the path
   ``primary_stratum`` ("in" with >= 400 in-distribution positions, else
   "in_or_out") and ``bottom_rule`` are chosen here and fixed at F2.
   ``--provisional`` skips P5a (zone (c') empty) for cost estimates;
   ``--report-all-levels`` adds the counts of every level (report only).
3. ``--stage injected`` (P5b): the table fixed at F2 (sha256 asserted through
   the tracked F2 record) -> inject -> recipe L1 -> batched inference on the
   tiles around the injections only -> R2 + V1/V2 merge -> L0 boxes ->
   one-to-one match (+-3 traces, +-0.5 ``P_line``, a box hits only the apex
   nearest its centre). The hit is also written for every conf level of the
   C-5 grid.
4. ``--smoke``: r0 x fold 0 x three 4TU lines (short, middle, long) through
   P5a and P5b paths in a separate folder; measures time and VRAM and writes
   ``smoke_estimate.json`` (cost x measured rates, F1 time record). Not for
   judgement.

All inference runs the same batched path (``predict_batch`` when the detector
has it). The L1 of a realization is built once per recipe and shared by the
five folds.

Blind output: metric files only (``<out-root>/<stage>/<recipe>/fold<k>/*.csv``
and the stage ``summary.json``). Stdout carries line counts, tile counts,
elapsed seconds and paths; its last line is the JSON
``{"status", "stage", "out_dir", "n_lines", "n_tiles", "n_injections",
"elapsed_s"}``. Exit codes: 0 done, 1 identity gate failed or sample short,
2 input refused (sha, tree, output exists), 3 internal error.
"""

import argparse
import hashlib
import json
import logging
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common
import m32_aihub_perturb as stage_b
import m32_pipe_responses as responses_io
import m32_recipe_inputs as inputs
import m32_recipe_match as match

from gpr_detection.detect.base import DetectorOutput
from gpr_detection.eval import inject
from gpr_detection.io.classes import DetectionClass
from gpr_detection.io.fourtu import load_fourtu_radargram
from gpr_detection.io.guangzhou import load_guangzhou_radargram
from gpr_detection.io.paths import DatasetName, load_dataset_paths
from gpr_detection.io.radargram import GridBox, Radargram
from gpr_detection.pipeline import (
    DEFAULT_MERGE_CONFIG,
    LineBox,
    image_box_to_grid_box,
    is_tile_truncated,
    line_boxes_to_grid,
    merge_line_boxes,
    model_box_to_image_box,
    tile_box_to_line_box,
)
from gpr_detection.preprocess.filters import prepare_fitting_branch
from gpr_detection.preprocess.timezero import ZeroResult, find_time_zero
from gpr_detection.preprocess.to_image import DetectionImage, make_detection_image

REPO = inputs.REPO
DEFAULT_OUT_ROOT = REPO / "runs" / "eval" / "m32_recipe"
SMOKE_OUT_ROOT = REPO / "runs" / "eval" / "m32_recipe_smoke"
F2_RECORD = REPO / "analysis" / "detection" / "m32_recipe_f2.json"
SCHEMA = "m32_recipe_eval/1"

CONFIDENCE = 0.25
NMS_IOU = 0.7
INPUT_PX = 256
DEFAULT_BATCH_SIZE = 64
MAX_PLAN_WORKERS = 4
SAMPLE_MINIMUM = 400
PRIMARY_IN = "in"
PRIMARY_IN_OR_OUT = "in_or_out"
BOTTOM_POINT = "point"
BOTTOM_UPPER = "upper_bound"
LEVELS = tuple(inject.LEVEL_AMPLITUDES)
LEVEL_W_TIE = -20.0
F2_DECISIONS = ("level_w_db", "primary_stratum", "bottom_rule", "pi_bottom")
"""Plan-stage choices the F2 record fixes (amendment 5)."""
SAMPLE_TARGET = 600
CONTRIBUTING_LINES_MINIMUM = 60
BOTTOM_TARGET = 100
SMOKE_RECIPE = inputs.R0
SMOKE_FOLD = 0
SMOKE_N_LINES = 3
ORIGINAL = "original"
FLIPPED = "flipped"
EXIT_CODES = {
    "ok": 0,
    "gate_failed": 1,
    "sample_short": 1,
    "input_error": 2,
    "error": 3,
}

logger = logging.getLogger("m32_recipe_eval")


class Stage(StrEnum):
    """Run stages of plan 4-4."""

    UNINJECTED = "uninjected"
    PLAN = "plan"
    INJECTED = "injected"


BOX_COLUMNS = (
    "recipe",
    "fold",
    "line_id",
    "dataset",
    "role",
    "orientation",
    "class_id",
    "class_name",
    "score",
    "x1",
    "y1",
    "x2",
    "y2",
    "i0",
    "i1",
    "k0",
    "k1",
    "in_grid",
    "n_tiles_seen",
    "tile_truncated",
    "tile_clipped",
    "m4",
    "m4c",
    "m4c_d3_8px",
)
CONF_COUNT_COLUMNS = (
    "recipe",
    "fold",
    "line_id",
    "dataset",
    "role",
    *(f"c{level:.2f}" for level in match.conf_grid()),
)
APEX_COLUMNS = ("recipe", "fold", "blind_id", "line_id", "trace", "sample", "hit")
FLIP_COLUMNS = (
    "recipe",
    "fold",
    "line_id",
    "box_index",
    "class_id",
    "flip_box_index",
    "iou",
    "consistent",
)
LINE_COLUMNS = (
    "line_id",
    "dataset",
    "role",
    "zero_ok",
    "t_zero_ns",
    "period_samples",
    "content_height_px",
    "n_tiles",
    "width_px",
    "flags",
)
MATCH_COLUMNS = (
    "recipe",
    "fold",
    "injection_id",
    "realization_id",
    "line_id",
    "dataset",
    "response_id",
    "model_group",
    "snr_db",
    "polarity",
    "k",
    "in_distribution",
    "bottom_stratum",
    "in_horizontal_box_zone",
    "width_class",
    "height_class",
    "width_periods",
    "height_periods",
    "box_clipped",
    "period_ringing",
    "n_clipped_cells",
    "max_clipped_injection_ratio",
    "hit",
    "hit_any_class",
    "n_boxes",
    "score",
    "iou_best",
    "iou_visible_box",
    "hit_mask",
    "period_changed",
)
REJECTION_COLUMNS = ("injection_id", "realization_id", "line_id", "reason")
POSITION_COLUMNS = (
    "line_id",
    "dataset",
    "group",
    "response_id",
    "model_group",
    "apex_trace",
    "bottom_stratum",
    "period_ringing",
    *(
        f"{kind}_m{round(-level)}"  # level_tag(), defined below
        for level in LEVELS
        for kind in ("width", "height", "aperture")
    ),
    "width_class",
    "height_class",
    "in_distribution",
)
STAGE_FILES: dict[Stage, dict[str, tuple[str, ...]]] = {
    Stage.UNINJECTED: {
        "boxes": BOX_COLUMNS,
        "conf_counts": CONF_COUNT_COLUMNS,
        "apex_hits": APEX_COLUMNS,
        "flip_pairs": FLIP_COLUMNS,
    },
    Stage.INJECTED: {"inj_matches": MATCH_COLUMNS},
}


class GateFailedError(Exception):
    """The r0 M6 counts did not reproduce D3 (details in files only)."""


# ------------------------------------------------------------- run setup


@dataclass(frozen=True)
class RunSettings:
    """Command-line choices that shape a run."""

    stage: Stage
    out_root: Path
    device: str
    batch_size: int
    recipes: tuple[str, ...]
    folds: tuple[int, ...]
    flip: bool
    limit_lines: int | None
    smoke: bool
    provisional: bool
    resume: bool
    overwrite: bool
    allow_dirty: bool
    run_root: Path
    f2_record: Path
    workers: int = 1
    cost_table: Path | None = None
    report_all_levels: bool = False

    @property
    def stage_dir(self) -> Path:
        """Output folder of the stage."""
        name = self.stage.value
        if self.stage is Stage.PLAN and self.provisional:
            name = "plan_provisional"
        return self.out_root / name

    @property
    def is_full(self) -> bool:
        """All recipes, folds and lines: the only runs a verdict may use."""
        return (
            not self.smoke
            and self.limit_lines is None
            and self.recipes == inputs.STAGE_C_RECIPES
            and self.folds == inputs.FOLDS
        )


@dataclass
class StageCounts:
    """What the closing stdout line reports (no metric values)."""

    n_lines: int = 0
    n_tiles: int = 0
    n_injections: int = 0


@dataclass
class PhaseTimer:
    """Wall seconds per named phase (smoke time estimate)."""

    seconds: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Add the time spent inside the block to ``seconds[name]``."""
        begin = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - begin
            self.seconds[name] = self.seconds.get(name, 0.0) + elapsed


@dataclass(frozen=True)
class LineData:
    """One loaded line and its recipe-independent zero."""

    ref: inputs.LineRef
    radargram: Radargram
    zero: ZeroResult


_DATASET_ROOTS: dict[str, Path] = {}


def _dataset_root(dataset: str) -> Path:
    if not _DATASET_ROOTS:
        paths = load_dataset_paths()
        _DATASET_ROOTS[inputs.DATASET_4TU] = paths.root(DatasetName.FOURTU)
        _DATASET_ROOTS[inputs.DATASET_IDS] = paths.root(DatasetName.GUANGZHOU)
    return _DATASET_ROOTS[dataset]


def load_radargram(ref: inputs.LineRef) -> Radargram:
    """Read one line from the read-only dataset packages."""
    root = _dataset_root(ref.dataset)
    if ref.dataset == inputs.DATASET_4TU:
        activity_id, path = ref.line_id.split("/")
        return load_fourtu_radargram(root, activity_id, path)
    return load_guangzhou_radargram(root, ref.line_id)


def load_line(ref: inputs.LineRef) -> LineData:
    """Line and its ``select_zero`` default-rule zero (M4d, plan 3-3)."""
    radargram = load_radargram(ref)
    branch = prepare_fitting_branch(radargram)
    return LineData(ref, radargram, find_time_zero(radargram, branch.filtered))


# ------------------------------------------------------------- inference


def load_detectors(settings: RunSettings) -> dict[int, object]:
    """N2-a' models of the selected folds (weights sha asserted before)."""
    # Lazy: ultralytics stays inside gpr_detection.detect (AGPL boundary).
    from gpr_detection.detect.yolo import load_yolo_detector
    from gpr_detection.detect.yolo_logging import quiet_ultralytics_logger

    quiet_ultralytics_logger()
    return {
        fold: load_yolo_detector(
            weights_path(settings.run_root, fold),
            input_size=INPUT_PX,
            confidence_threshold=CONFIDENCE,
            iou_threshold=NMS_IOU,
            device=settings.device,
        )
        for fold in settings.folds
    }


def weights_path(run_root: Path, fold: int) -> Path:
    """``runs/detect/<N2-a' prefix><fold>/weights/last.pt``."""
    return run_root / f"{stage_b.RUN_PREFIX}{fold}" / "weights" / "last.pt"


def assert_weights(run_root: Path, folds: Iterable[int]) -> dict[str, str]:
    """sha256 of every fold's ``last.pt`` against the N2-a' baseline."""
    return {
        str(fold): inputs.checked_sha256(
            weights_path(run_root, fold),
            stage_b.EXPECTED_WEIGHTS_SHA256[fold],
            f"fold {fold} weights",
        )
        for fold in folds
    }


def predict_tiles(
    detector: object, tiles: Sequence[np.ndarray], batch_size: int
) -> list[DetectorOutput]:
    """Batched inference when the detector offers it, else one call per tile."""
    batch_call = getattr(detector, "predict_batch", None)
    if batch_call is None or batch_size <= 1:
        return [detector.predict(tile) for tile in tiles]  # type: ignore[attr-defined]
    outputs: list[DetectorOutput] = []
    for start in range(0, len(tiles), batch_size):
        outputs.extend(batch_call(list(tiles[start : start + batch_size])))
    return outputs


def tile_line_boxes(
    image: DetectionImage, tile_index: int, output: DetectorOutput
) -> list[LineBox]:
    """``detect_tiles`` for one tile: line L1 boxes with truncation marks."""
    tile = image.tiles[tile_index]
    boxes = []
    for detection in output.detections:
        line_box = tile_box_to_line_box(
            model_box_to_image_box(detection.box, output.transform), tile
        )
        boxes.append(
            LineBox(
                class_id=detection.class_id,
                class_name=detection.class_name,
                score=detection.score,
                box=line_box,
                tile_index=tile.index,
                tile_truncated=is_tile_truncated(
                    line_box, tile, DEFAULT_MERGE_CONFIG.edge_px
                ),
            )
        )
    return boxes


@dataclass(frozen=True)
class ImageJob:
    """One L1 image and the tiles of it that are inferred."""

    recipe: str
    orientation: str
    image: DetectionImage
    tile_indices: tuple[int, ...]
    tag: int = 0


def run_jobs(
    jobs: Sequence[ImageJob], detectors: Mapping[int, object], batch_size: int
) -> dict[tuple[int, int], list[LineBox]]:
    """Tile boxes of every job and fold; all tiles of all jobs share batches."""
    tiles = [(j, t) for j, job in enumerate(jobs) for t in job.tile_indices]
    pixels = [jobs[j].image.tile_image(t) for j, t in tiles]
    results: dict[tuple[int, int], list[LineBox]] = {}
    for fold, detector in detectors.items():
        for j in range(len(jobs)):
            results[(j, fold)] = []
        outputs = predict_tiles(detector, pixels, batch_size)
        for (j, t), output in zip(tiles, outputs, strict=True):
            results[(j, fold)].extend(tile_line_boxes(jobs[j].image, t, output))
    return results


def merge_with_grid(
    image: DetectionImage, tile_boxes: Sequence[LineBox]
) -> list[tuple[LineBox, GridBox | None]]:
    """R2 + V1/V2 merge (``MergeConfig()``) and the one outward L0 rounding.

    A box lying wholly in the bottom padding has no L0 box (``None``); it is
    kept because it still is an M4a box (plan 3-3).
    """
    merged = merge_line_boxes(tile_boxes, image, DEFAULT_MERGE_CONFIG)
    return [(box, image_box_to_grid_box(box.box, image.grid_map)) for box in merged]


def merge_to_grid(
    image: DetectionImage, tile_boxes: Sequence[LineBox]
) -> list[tuple[LineBox, GridBox]]:
    """Merged boxes that reach the L0 grid (``line_boxes_to_grid``)."""
    return line_boxes_to_grid(merge_line_boxes(tile_boxes, image), image)


def scored(
    pairs: Iterable[tuple[LineBox, GridBox | None]],
) -> list[match.ScoredBox]:
    """Merged boxes with an L0 box as ``ScoredBox`` values (match rules)."""
    return [
        match.ScoredBox(box.class_id, box.score, grid)
        for box, grid in pairs
        if grid is not None
    ]


def kept_boxes(boxes: Sequence[LineBox], keep: np.ndarray) -> list[LineBox]:
    """Tile boxes whose keep flag is set (conf filter before the merge)."""
    return [box for box, is_kept in zip(boxes, keep, strict=True) if is_kept]


# ------------------------------------------------------- output helpers


def _write_json(path: Path, document: Mapping) -> None:
    common.write_json(path, dict(document))


def _write_part(path: Path, document: Mapping) -> None:
    """Write a per-line part atomically, so a stopped run resumes cleanly."""
    temporary = path.with_suffix(".tmp")
    _write_json(temporary, document)
    temporary.replace(path)


def _float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def assemble_stage_files(settings: RunSettings, parts: Sequence[Mapping]) -> None:
    """Concatenate the line parts into ``<stage>/<recipe>/fold<k>/<name>.csv``."""
    for name, columns in STAGE_FILES[settings.stage].items():
        for recipe in settings.recipes:
            for fold in settings.folds:
                rows = [
                    [row[column] for column in columns]
                    for part in parts
                    for row in part.get(name, ())
                    if row["recipe"] == recipe and row["fold"] == fold
                ]
                path = settings.stage_dir / recipe / f"fold{fold}" / f"{name}.csv"
                common.write_csv(path, columns, rows)


def _write_rows(path: Path, columns: Sequence[str], rows: Iterable[Mapping]) -> None:
    common.write_csv(path, columns, ([row[c] for c in columns] for row in rows))


# ======================================================== P5a: uninjected


def flipped_radargram(radargram: Radargram) -> Radargram:
    """Left-right mirror of a line (M5: the whole path runs on it)."""
    return replace(radargram, data=np.ascontiguousarray(radargram.data[:, ::-1]))


def content_range(image: DetectionImage, zero: ZeroResult) -> match.ContentRange:
    """M4 rows of one line image from its ``ImageGridMap`` (not pixel values)."""
    grid = image.grid_map
    y_bot = grid.n_samples / grid.samples_per_px
    if abs(y_bot - image.content_height_px) > 1e-6:
        raise RuntimeError(
            f"content bottom {y_bot} != H_c {image.content_height_px} (grid map)"
        )
    zero_px = None
    if zero.k_zero is not None and zero.t_zero_ns is not None:
        zero_px = zero.k_zero / grid.samples_per_px
    return match.ContentRange(
        y_top_px=0.0,
        y_bot_px=y_bot,
        period_l1_px=image.period_samples / grid.samples_per_px,
        zero_px=zero_px,
    )


def check_recipe_invariance(images: Mapping[str, DetectionImage]) -> None:
    """S6 (f): period, grid map and content range do not depend on the recipe."""
    reference = (
        images[inputs.R0] if inputs.R0 in images else next(iter(images.values()))
    )
    for recipe, image in images.items():
        same = (
            image.grid_map == reference.grid_map
            and image.content_height_px == reference.content_height_px
            and image.period_samples == reference.period_samples
            and len(image.tiles) == len(reference.tiles)
        )
        if not same:
            raise RuntimeError(f"recipe {recipe} changed the line geometry (S6 (f))")


def uninjected_jobs(
    line: LineData, settings: RunSettings
) -> tuple[list[ImageJob], dict[str, DetectionImage]]:
    """L1 images of every recipe (and its mirror) with all their tiles."""
    jobs, originals = [], {}
    mirrored = flipped_radargram(line.radargram) if settings.flip else None
    for recipe in settings.recipes:
        config = inputs.recipe_config(recipe)
        image = make_detection_image(line.radargram, config)
        originals[recipe] = image
        jobs.append(ImageJob(recipe, ORIGINAL, image, tuple(range(len(image.tiles)))))
        if mirrored is not None:
            flipped = make_detection_image(mirrored, config)
            jobs.append(
                ImageJob(recipe, FLIPPED, flipped, tuple(range(len(flipped.tiles))))
            )
    check_recipe_invariance(originals)
    return jobs, originals


def box_row(
    line: LineData,
    job: ImageJob,
    fold: int,
    pair: tuple[LineBox, GridBox | None],
    content: match.ContentRange,
) -> dict:
    """One merged box with its M4 classification (L1 rows of the content range).

    Mirrored-run boxes (M5 only) carry no M4 fields; their L0 box is in the
    mirrored orientation. A box wholly in the padding has empty L0 fields.
    """
    box, grid = pair
    is_original = job.orientation == ORIGINAL
    category = match.m4_category(box.box.y1, box.box.y2, content)
    return {
        "recipe": job.recipe,
        "fold": fold,
        "line_id": line.ref.line_id,
        "dataset": line.ref.dataset,
        "role": line.ref.role,
        "orientation": job.orientation,
        "class_id": box.class_id,
        "class_name": box.class_name,
        "score": _float(box.score, 6),
        "x1": _float(box.box.x1),
        "y1": _float(box.box.y1),
        "x2": _float(box.box.x2),
        "y2": _float(box.box.y2),
        **_grid_fields(grid),
        "n_tiles_seen": box.n_tiles_seen,
        "tile_truncated": box.tile_truncated,
        "tile_clipped": box.tile_clipped,
        "m4": category.value if is_original and category is not None else "",
        "m4c": is_original
        and match.is_m4c(box.class_id, box.box.y1, box.box.y2, content),
        "m4c_d3_8px": is_original
        and match.is_m4c(
            box.class_id, box.box.y1, box.box.y2, content, match.M4C_D3_WINDOW_PX
        ),
    }


def _grid_fields(grid: GridBox | None) -> dict:
    if grid is None:
        return {"i0": "", "i1": "", "k0": "", "k1": "", "in_grid": False}
    return {"i0": grid.i0, "i1": grid.i1, "k0": grid.k0, "k1": grid.k1, "in_grid": True}


def conf_count_row(
    line: LineData, recipe: str, fold: int, image: DetectionImage, boxes: list[LineBox]
) -> dict:
    """``pipe_hyperbola`` merged boxes at every C-5 conf level (filter, merge)."""

    def count(keep: np.ndarray) -> int:
        merged = merge_line_boxes(kept_boxes(boxes, keep), image, DEFAULT_MERGE_CONFIG)
        return sum(box.class_id == DetectionClass.PIPE_HYPERBOLA for box in merged)

    counts = match.evaluate_on_conf_grid([b.score for b in boxes], count)
    row = {
        "recipe": recipe,
        "fold": fold,
        "line_id": line.ref.line_id,
        "dataset": line.ref.dataset,
        "role": line.ref.role,
    }
    for level, value in zip(match.conf_grid(), counts, strict=True):
        row[f"c{level:.2f}"] = value
    return row


def apex_rows(
    line: LineData,
    recipe: str,
    fold: int,
    boxes: Sequence[match.ScoredBox],
    period_samples: float,
    apexes: Sequence[inputs.KnownApex],
) -> list[dict]:
    """M6 hits of the K apexes on this line (D3 rule)."""
    return [
        {
            "recipe": recipe,
            "fold": fold,
            "blind_id": apex.blind_id,
            "line_id": apex.line_id,
            "trace": apex.trace,
            "sample": apex.sample,
            "hit": match.apex_hit_d3(apex.trace, apex.sample, boxes, period_samples),
        }
        for apex in apexes
        if apex.line_id == line.ref.line_id
    ]


def flip_rows(
    line: LineData,
    recipe: str,
    fold: int,
    original: Sequence[match.ScoredBox],
    flipped: Sequence[match.ScoredBox],
) -> list[dict]:
    """M5: original boxes against the mirrored run, mirrored back."""
    n_traces = line.radargram.n_traces
    back = [
        match.ScoredBox(
            box.class_id, box.score, match.flip_grid_box(box.grid, n_traces)
        )
        for box in flipped
    ]
    return [
        {
            "recipe": recipe,
            "fold": fold,
            "line_id": line.ref.line_id,
            "box_index": index,
            "class_id": original[index].class_id,
            "flip_box_index": "" if other is None else other,
            "iou": _float(value),
            "consistent": other is not None,
        }
        for index, other, value in match.flip_pairs(original, back)
    ]


def line_row(line: LineData, image: DetectionImage) -> dict:
    """Recipe-independent geometry and zero of one line."""
    zero = line.zero
    return {
        "line_id": line.ref.line_id,
        "dataset": line.ref.dataset,
        "role": line.ref.role,
        "zero_ok": zero.t_zero_ns is not None,
        "t_zero_ns": "" if zero.t_zero_ns is None else _float(zero.t_zero_ns),
        "period_samples": _float(image.period_samples),
        "content_height_px": image.content_height_px,
        "n_tiles": len(image.tiles),
        "width_px": image.width_px,
        "flags": ";".join(image.flags),
    }


def evaluate_uninjected_line(
    line: LineData,
    detectors: Mapping[int, object],
    settings: RunSettings,
    apexes: Sequence[inputs.KnownApex],
    timer: PhaseTimer | None = None,
) -> dict:
    """All P5a rows of one line (every recipe, fold and orientation)."""
    timer = timer or PhaseTimer()
    with timer.phase("l1"):
        jobs, originals = uninjected_jobs(line, settings)
    with timer.phase("infer"):
        detections = run_jobs(jobs, detectors, settings.batch_size)
    with timer.phase("post"):
        part = _uninjected_rows(line, jobs, originals, detections, settings, apexes)
    part["n_tiles"] = sum(len(job.tile_indices) for job in jobs)
    return part


def _uninjected_rows(
    line: LineData,
    jobs: Sequence[ImageJob],
    originals: Mapping[str, DetectionImage],
    detections: Mapping[tuple[int, int], list[LineBox]],
    settings: RunSettings,
    apexes: Sequence[inputs.KnownApex],
) -> dict:
    part: dict[str, list] = {name: [] for name in STAGE_FILES[Stage.UNINJECTED]}
    reference = originals[settings.recipes[0]]
    part["line"] = [line_row(line, reference)]
    content = content_range(reference, line.zero)
    merged = {
        key: merge_with_grid(jobs[key[0]].image, boxes)
        for key, boxes in detections.items()
    }
    mirror_of = {
        job.recipe: j for j, job in enumerate(jobs) if job.orientation == FLIPPED
    }
    for (j, fold), pairs in merged.items():
        job = jobs[j]
        part["boxes"] += [box_row(line, job, fold, pair, content) for pair in pairs]
        if job.orientation == FLIPPED:
            continue
        part["conf_counts"].append(
            conf_count_row(line, job.recipe, fold, job.image, detections[(j, fold)])
        )
        part["apex_hits"] += apex_rows(
            line, job.recipe, fold, scored(pairs), job.image.period_samples, apexes
        )
        if job.recipe in mirror_of:
            mirrored = scored(merged[(mirror_of[job.recipe], fold)])
            part["flip_pairs"] += flip_rows(
                line, job.recipe, fold, scored(pairs), mirrored
            )
    return part


def identity_gate(apex_rows_all: Iterable[Mapping]) -> dict:
    """D3 reproduction of r0 (plan C-8): sum 90, any fold 30, fold 0 15."""
    folds_hit: dict[str, int] = {}
    fold0_hits = 0
    for row in apex_rows_all:
        if row["recipe"] != inputs.R0:
            continue
        hit = bool(row["hit"])
        folds_hit[row["blind_id"]] = folds_hit.get(row["blind_id"], 0) + int(hit)
        fold0_hits += int(hit and row["fold"] == 0)
    observed = {
        "sum_hits": sum(folds_hit.values()),
        "any_fold": sum(value > 0 for value in folds_hit.values()),
        "fold0": fold0_hits,
        "n_apexes": len(folds_hit),
    }
    expected = {
        "sum_hits": inputs.GATE_SUM_HITS,
        "any_fold": inputs.GATE_ANY_FOLD,
        "fold0": inputs.GATE_FOLD0,
        "n_apexes": inputs.N_K_APEXES,
    }
    return {"observed": observed, "expected": expected, "pass": observed == expected}


# ============================================================== plan stage


def uninjected_exclusions(stage_dir: Path) -> dict[str, list[GridBox]]:
    """Zone (c'): every original-orientation P5a box, all recipes and folds."""
    boxes: dict[str, list[GridBox]] = {}
    for path in sorted(stage_dir.glob("*/fold*/boxes.csv")):
        for row in inputs.read_csv(path):
            if row["orientation"] != ORIGINAL or row["in_grid"] != "True":
                continue
            box = GridBox(
                int(row["i0"]), int(row["i1"]), int(row["k0"]), int(row["k1"])
            )
            boxes.setdefault(row["line_id"], []).append(box)
    return boxes


def require_complete_uninjected(settings: RunSettings) -> Path:
    """The P5a folder of the same out-root, finished, gated and full."""
    stage_dir = settings.out_root / Stage.UNINJECTED.value
    summary_path = stage_dir / "summary.json"
    if not summary_path.is_file():
        raise inputs.InputError(f"P5a summary missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "ok" or not summary.get("full_run"):
        raise inputs.InputError("P5a is not a finished full run with a passed gate")
    return stage_dir


@dataclass(frozen=True)
class LinePlanResult:
    """Plan stage result of one background line."""

    specs: list[inject.InjectionSpec]
    cost: inject.LineCost
    positions: list[dict]
    report: dict
    candidates: dict[str, tuple[int, int]] = field(default_factory=dict)


def usable_patches(
    context: inject.LineContext, responses: Mapping[str, inject.PipeResponse]
) -> tuple[list[inject.ResponsePatch], dict[str, str]]:
    """Main-``k`` patches of every response, and the reasons of the others."""
    patches, failures = [], {}
    for model_id, response in sorted(responses.items()):
        patch = inject.build_response_patch(response, context, inject.K_MAIN)
        if isinstance(patch, inject.RejectReason):
            failures[model_id] = patch.value
        else:
            patches.append(patch)
    return patches, failures


def level_tag(level_db: float) -> str:
    """Column tag of a stratum level: -20.0 -> ``"m20"``."""
    return f"m{round(-level_db)}"


def position_row(
    line: inputs.LineRef,
    context: inject.LineContext,
    patch: inject.ResponsePatch,
    position: inject.PlannedPosition,
    groups: Mapping[str, responses_io.ModelGroup],
) -> dict:
    """One planned position with the stratum box sizes of every level.

    The sizes are those ``inject`` writes into the truth; the stratum itself
    needs ``l_W`` and is added by :func:`classify_positions`.
    """
    row: dict = {
        "line_id": line.line_id,
        "dataset": line.dataset,
        "group": position.group,
        "response_id": position.response_id,
        "model_group": groups[position.response_id].value,
        "apex_trace": repr(float(position.apex_trace)),
        "bottom_stratum": inject.is_bottom_stratum(context, patch),
        "period_ringing": patch.period_ringing,
    }
    for level_db, box in patch.level_boxes.items():
        tag = level_tag(level_db)
        row[f"width_{tag}"] = box.width_periods
        row[f"height_{tag}"] = box.height_periods
        row[f"aperture_{tag}"] = box.aperture_limited
    return row


def _level_width_class(row: Mapping, level_db: float) -> inject.WidthClass:
    tag = level_tag(level_db)
    return inject.width_class(row[f"width_{tag}"], bool(row[f"aperture_{tag}"]))


def censored_median(widths: Sequence[float], limited: Sequence[bool]) -> float:
    """Median with aperture-limited widths (lower bounds) ranked as the widest.

    Plan 4-2 (amendment 6): the median is +inf only when **more than half**
    of the positions are aperture-limited. Otherwise it is the rank median
    with the limited widths placed above every finite width; with an even
    count exactly half limited, the two middle ranks are the largest finite
    width and an infinite one, and the finite one is taken (``np.median``
    would average them to +inf).
    """
    if not widths or len(widths) != len(limited):
        raise ValueError("need one aperture flag per width (and at least one)")
    n_limited = sum(bool(flag) for flag in limited)
    if 2 * n_limited > len(widths):
        return math.inf
    finite = sorted(
        w for w, is_limited in zip(widths, limited, strict=True) if not is_limited
    )
    ranked = finite + [math.inf] * n_limited
    middle = len(ranked) // 2
    if len(ranked) % 2:
        return float(ranked[middle])
    lower, upper = ranked[middle - 1], ranked[middle]
    if math.isinf(upper):
        return float(lower)
    return float((lower + upper) / 2)


def level_width_report(positions: Sequence[Mapping]) -> dict[str, dict]:
    """Per level: censored median width and the aperture-limited share."""
    report = {}
    for level_db in LEVELS:
        tag = level_tag(level_db)
        widths = [row[f"width_{tag}"] for row in positions]
        limited = [bool(row[f"aperture_{tag}"]) for row in positions]
        if not widths:
            continue
        median = censored_median(widths, limited)
        report[tag] = {
            "censored_median": None if math.isinf(median) else median,
            "aperture_limited_share": sum(limited) / len(limited),
            "candidate": not math.isinf(median),
        }
    return report


def select_level_w(positions: Sequence[Mapping]) -> tuple[float, dict[str, dict]]:
    """``l_W``: level whose censored median width is nearest 11.36 P (log).

    One row per position (all SNR steps and both polarities share it).
    Aperture-limited widths enter the median as infinity (amendment 6); a
    level whose median is infinite (over half limited) is no candidate. Ties
    go to -20 dB, then to the lower level; with no candidate -20 dB is kept.

    Returns:
        ``(l_W, level report)`` (see :func:`level_width_report`).
    """
    report = level_width_report(positions)
    medians = {
        level_db: report[level_tag(level_db)]["censored_median"]
        for level_db in LEVELS
        if level_tag(level_db) in report and report[level_tag(level_db)]["candidate"]
    }
    if not medians:
        return LEVEL_W_TIE, report

    def distance(level_db: float) -> tuple[float, float]:
        gap = abs(math.log(medians[level_db] / inject.AIHUB_WIDTH_MEDIAN_PERIODS))
        return round(gap, 12), abs(level_db - LEVEL_W_TIE)

    return min(medians, key=distance), report


def natural_bottom_share(
    results: Sequence["LinePlanResult"],
    groups: Mapping[str, responses_io.ModelGroup],
) -> dict[str, float | None]:
    """``pi_bottom`` per ``dataset/model group`` cell: bottom share of the
    allowed candidates before any priority ([CVR] appendix D-3)."""
    tallies: dict[str, list[int]] = {}
    for result in results:
        dataset = result.report["dataset"]
        for response_id, (bottom, total) in result.candidates.items():
            cell = f"{dataset}/{groups[response_id].value}"
            tally = tallies.setdefault(cell, [0, 0])
            tally[0] += bottom
            tally[1] += total
    return {
        cell: (bottom / total if total else None)
        for cell, (bottom, total) in sorted(tallies.items())
    }


def classify_positions(positions: Sequence[dict], level_w_db: float) -> None:
    """Add ``in_distribution`` and the width / height reasons at ``l_W``."""
    tag = level_tag(level_w_db)
    for row in positions:
        width = _level_width_class(row, level_w_db)
        row["width_class"] = width.value
        row["height_class"] = inject.height_class(row[f"height_{tag}"]).value
        row["in_distribution"] = width is inject.WidthClass.IN


def level_summary(positions: Sequence[Mapping], level_db: float) -> dict:
    """In-distribution count and four-way reasons of one level (positions)."""
    tag = level_tag(level_db)
    widths = [_level_width_class(row, level_db) for row in positions]
    heights = [inject.height_class(row[f"height_{tag}"]) for row in positions]
    return {
        "in_distribution": sum(w is inject.WidthClass.IN for w in widths),
        "width_over": sum(w is inject.WidthClass.WIDTH_OVER for w in widths),
        "width_over_lower_bound": sum(
            w is inject.WidthClass.WIDTH_OVER_LOWER_BOUND for w in widths
        ),
        "width_under": sum(w is inject.WidthClass.WIDTH_UNDER for w in widths),
        "height_over": sum(h is inject.HeightClass.HEIGHT_OVER for h in heights),
        "height_under": sum(h is inject.HeightClass.HEIGHT_UNDER for h in heights),
        "aperture_limited": sum(bool(row[f"aperture_{tag}"]) for row in positions),
    }


def _report_k_period_failures(
    context: inject.LineContext, responses: Mapping[str, inject.PipeResponse]
) -> int:
    """Models failing the period check at the report ``k`` 0.8 (count only)."""
    return sum(
        inject.build_response_patch(response, context, inject.K_REPORT)
        == inject.RejectReason.PERIOD_CHECK
        for response in responses.values()
    )


def plan_line(
    line: LineData,
    exclusions: inject.LineExclusions,
    library: responses_io.ResponseLibrary,
) -> LinePlanResult:
    """Seeded positions, table rows and cost of one background line."""
    context = inject.prepare_line_context(line.radargram, exclusions)
    responses = library.for_line(line.ref.dataset, line.ref.time_window_ns)
    patches, failures = usable_patches(context, responses)
    plan = inject.plan_line_positions(context, patches)
    conditions = inject.default_conditions()
    by_id = {patch.response.response_id: patch for patch in patches}
    positions = [
        position_row(line.ref, context, by_id[p.response_id], p, library.groups)
        for p in plan.positions
    ]
    report = {
        "line_id": line.ref.line_id,
        "dataset": line.ref.dataset,
        "n_responses": len(responses),
        "patch_failures": failures,
        "n_period_ringing": sum(patch.period_ringing for patch in patches),
        "period_check_failed_k13": sum(
            reason == inject.RejectReason.PERIOD_CHECK.value
            for reason in failures.values()
        ),
        "period_check_failed_k08": _report_k_period_failures(context, responses),
        "n_candidates": plan.n_candidates,
        "n_dropped": plan.n_dropped,
        "n_positions": len(plan.positions),
        "n_realizations": plan.n_realizations,
    }
    return LinePlanResult(
        specs=inject.build_injection_specs(
            line.ref.line_id, plan.positions, conditions
        ),
        cost=inject.line_cost(context, plan, by_id, len(conditions)),
        positions=positions,
        report=report,
        candidates=dict(plan.candidates_by_response),
    )


def line_exclusions(
    line_id: str,
    boxes: Mapping[str, Sequence[GridBox]],
    candidates: inputs.CandidateApexes,
) -> inject.LineExclusions:
    """Zones (c') and (d) of one line."""
    return inject.LineExclusions(
        uninjected_boxes=tuple(boxes.get(line_id, ())),
        k_apex_traces=candidates.k_traces(line_id),
        auto_candidate_apex_traces=candidates.auto_traces(line_id),
    )


def sample_report(positions: Sequence[Mapping], reports: Sequence[Mapping]) -> dict:
    """Plan 3-2 sample rules after amendment 5: path, bottom rule, lines.

    ``primary_stratum`` is ``"in"`` with at least 400 in-distribution
    positions, else ``"in_or_out"`` (user decision 2026-11-14); the run stops
    (``sample_ok`` false) only below 60 contributing lines. Fewer than 100
    bottom-stratum positions switch C-6 to ``bottom_rule="upper_bound"``.
    """
    main = [p for p in positions if p["in_distribution"]]
    per_line: dict[str, int] = {}
    for position in positions:
        per_line[position["line_id"]] = per_line.get(position["line_id"], 0) + 1
    reasons: dict[str, int] = {}
    for report in reports:
        for reason in report["patch_failures"].values():
            reasons[reason] = reasons.get(reason, 0) + 1
    report = {
        "positions_per_snr_step": len(positions),
        "in_distribution_per_snr_step": len(main),
        "out_of_distribution_per_snr_step": len(positions) - len(main),
        "bottom_stratum": sum(bool(p["bottom_stratum"]) for p in positions),
        "contributing_lines": len(per_line),
        "max_positions_per_line": max(per_line.values(), default=0),
        "by_dataset_group": count_by(positions, ("dataset", "model_group")),
        "patch_failure_reasons": reasons,
        "n_dropped": sum(r["n_dropped"] for r in reports),
        "period_ringing_positions": sum(bool(p["period_ringing"]) for p in positions),
        "period_ringing_patches": sum(r["n_period_ringing"] for r in reports),
        "rules": {
            "minimum": SAMPLE_MINIMUM,
            "target": SAMPLE_TARGET,
            "contributing_lines_minimum": CONTRIBUTING_LINES_MINIMUM,
            "bottom_target": BOTTOM_TARGET,
        },
    }
    report["primary_stratum"] = (
        PRIMARY_IN if len(main) >= SAMPLE_MINIMUM else PRIMARY_IN_OR_OUT
    )
    report["bottom_rule"] = (
        BOTTOM_POINT if report["bottom_stratum"] >= BOTTOM_TARGET else BOTTOM_UPPER
    )
    report["sample_ok"] = report["contributing_lines"] >= CONTRIBUTING_LINES_MINIMUM
    return report


def count_by(rows: Iterable[Mapping], keys: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = "/".join(str(row[key]) for key in keys)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


# ======================================================== P5b: injected


@dataclass(frozen=True)
class F2Inputs:
    """The table, exclusions and stratum choices fixed at F2.

    Attributes:
        specs: Injection table rows.
        exclusions: Zone (c') boxes per line.
        sha256: Record, table and exclusions hashes.
        decisions: ``level_w_db``, ``primary_stratum`` and ``bottom_rule``.
    """

    specs: list[inject.InjectionSpec]
    exclusions: dict[str, tuple[GridBox, ...]]
    sha256: dict[str, str]
    decisions: dict


def read_f2_inputs(settings: RunSettings) -> F2Inputs:
    """Read the F2 record (tracked) and assert the table and exclusions."""
    if not settings.f2_record.is_file():
        raise inputs.InputError(f"F2 record missing: {settings.f2_record}")
    record = json.loads(settings.f2_record.read_text(encoding="utf-8"))
    plan_dir = settings.out_root / "plan"
    table_sha = inputs.checked_sha256(
        plan_dir / "injection_table.csv",
        record["injection_table_sha256"],
        "injection table",
    )
    exclusions_sha = inputs.checked_sha256(
        plan_dir / "exclusions.csv", record["exclusions_sha256"], "exclusions"
    )
    specs, _ = inject.read_injection_table(plan_dir / "injection_table.csv")
    missing = [key for key in F2_DECISIONS if key not in record]
    if missing:
        raise inputs.InputError(f"F2 record lacks {missing}")
    return F2Inputs(
        specs=specs,
        exclusions=inputs.read_exclusions(plan_dir / "exclusions.csv"),
        sha256={
            "f2_record": inputs.sha256_file(settings.f2_record),
            "injection_table": table_sha,
            "exclusions": exclusions_sha,
        },
        decisions={key: record[key] for key in F2_DECISIONS},
    )


def group_specs(
    specs: Sequence[inject.InjectionSpec],
) -> dict[str, dict[str, list[inject.InjectionSpec]]]:
    """``line_id -> realization_id -> rows`` in table order."""
    grouped: dict[str, dict[str, list[inject.InjectionSpec]]] = {}
    for spec in specs:
        grouped.setdefault(spec.line_id, {}).setdefault(spec.realization_id, []).append(
            spec
        )
    return grouped


def touched_tiles(
    image: DetectionImage, truths: Sequence[inject.InjectionTruth]
) -> tuple[int, ...]:
    """Tiles overlapping a truth wing box grown by one tile stride ([CVR] B-3)."""
    stride_px = image.tile_px - image.overlap_px
    traces_per_px = image.grid_map.traces_per_px
    touched = set()
    for truth in truths:
        start_px = truth.wing_box.i0 / traces_per_px - stride_px
        stop_px = (truth.wing_box.i1 + 1) / traces_per_px + stride_px
        for tile in image.tiles:
            if tile.x0_px < stop_px and start_px < tile.x0_px + tile.content_width_px:
                touched.add(tile.index)
    return tuple(sorted(touched))


@dataclass(frozen=True)
class MatchContext:
    """What every match row of one line shares."""

    line: LineData
    context: inject.LineContext
    groups: Mapping[str, responses_io.ModelGroup]
    level_w_db: float


def match_rows(
    ctx: MatchContext,
    job: ImageJob,
    fold: int,
    result: inject.InjectionResult,
    tile_boxes: Sequence[LineBox],
) -> list[dict]:
    """Match rows of one realization, recipe and fold (one-to-one, conf grid)."""
    truths = result.truths
    apexes = [
        match.TruthApex(t.injection_id, t.apex_trace, t.apex_sample) for t in truths
    ]
    period_traces, period_samples = (
        ctx.context.period_traces,
        ctx.context.period_samples,
    )

    def outcomes(keep: np.ndarray) -> list[match.MatchOutcome]:
        pairs = merge_to_grid(job.image, kept_boxes(tile_boxes, keep))
        return match.match_one_to_one(
            apexes, scored(pairs), period_traces, period_samples
        )

    by_level = match.evaluate_on_conf_grid([b.score for b in tile_boxes], outcomes)
    base_boxes = scored(merge_to_grid(job.image, tile_boxes))
    any_class = match.match_one_to_one(
        apexes, base_boxes, period_traces, period_samples, class_id=None
    )
    period_changed = job.image.period_samples != ctx.context.period_samples
    rows = []
    for index, truth in enumerate(truths):
        base = by_level[0][index]
        rows.append(
            _match_row(ctx, job, fold, result, truth, base, base_boxes)
            | {
                "hit_any_class": any_class[index].hit,
                "hit_mask": match.hit_mask_text([lv[index].hit for lv in by_level]),
                "period_changed": period_changed,
            }
        )
    return rows


def _match_row(
    ctx: MatchContext,
    job: ImageJob,
    fold: int,
    result: inject.InjectionResult,
    truth: inject.InjectionTruth,
    base: match.MatchOutcome,
    base_boxes: Sequence[match.ScoredBox],
) -> dict:
    score = (
        "" if base.box_index is None else _float(base_boxes[base.box_index].score, 6)
    )
    level_w = ctx.level_w_db
    height_at_level = truth.height_periods_by_level[level_w]
    return {
        "recipe": job.recipe,
        "fold": fold,
        "injection_id": truth.injection_id,
        "realization_id": result.realization_id,
        "line_id": ctx.line.ref.line_id,
        "dataset": ctx.line.ref.dataset,
        "response_id": truth.response_id,
        "model_group": ctx.groups[truth.response_id].value,
        "snr_db": truth.snr_db,
        "polarity": truth.polarity,
        "k": truth.k,
        "in_distribution": truth.in_distribution(level_w),
        "bottom_stratum": truth.bottom_stratum,
        "in_horizontal_box_zone": truth.in_horizontal_box_zone,
        "width_class": truth.width_class(level_w).value,
        "height_class": inject.height_class(height_at_level).value,
        "width_periods": _float(truth.width_periods),
        "height_periods": _float(truth.height_periods),
        "box_clipped": truth.box_clipped,
        "period_ringing": truth.period_ringing,
        "n_clipped_cells": truth.n_clipped_cells,
        "max_clipped_injection_ratio": _float(truth.max_clipped_injection_ratio, 6),
        "hit": base.hit,
        "n_boxes": base.n_boxes,
        "score": score,
        "iou_best": _float(
            match.best_iou(truth.box, base_boxes, DetectionClass.PIPE_HYPERBOLA)
        ),
        "iou_visible_box": _float(
            match.best_iou(truth.visible_box, base_boxes, DetectionClass.PIPE_HYPERBOLA)
        ),
    }


def with_condition(rejections: Iterable[Mapping]) -> list[dict]:
    """Rejection rows with SNR and k read from the realization id."""
    rows = []
    for row in rejections:
        parts = row["realization_id"].rsplit(":", 4)
        fields = {"snr_db": "", "k": ""}
        if len(parts) == 5:  # <line>:g<group>:snr<dB>:k<k>:p<+-1>
            fields = {
                "snr_db": parts[2].removeprefix("snr"),
                "k": parts[3].removeprefix("k"),
            }
        rows.append({**row, **fields})
    return rows


def injected_jobs(
    line: LineData, results: Sequence[inject.InjectionResult], recipe: str
) -> list[ImageJob]:
    """Recipe L1 of every realization with accepted injections."""
    config = inputs.recipe_config(recipe)
    jobs = []
    for index, result in enumerate(results):
        if not result.truths:
            continue
        injected = replace(line.radargram, data=result.data.astype(np.float32))
        image = make_detection_image(injected, config)
        jobs.append(
            ImageJob(
                recipe, ORIGINAL, image, touched_tiles(image, result.truths), index
            )
        )
    return jobs


def evaluate_injected_line(
    line: LineData,
    realizations: Mapping[str, Sequence[inject.InjectionSpec]],
    exclusions: inject.LineExclusions,
    library: responses_io.ResponseLibrary,
    detectors: Mapping[int, object],
    settings: RunSettings,
    timer: PhaseTimer | None = None,
    *,
    level_w_db: float,
) -> dict:
    """All P5b rows of one background line (stratum at the F2 level ``l_W``)."""
    timer = timer or PhaseTimer()
    with timer.phase("context"):
        context = inject.prepare_line_context(line.radargram, exclusions)
        responses = library.for_line(line.ref.dataset, line.ref.time_window_ns)
    with timer.phase("inject"):
        results = [
            inject.inject_realization(context, specs, responses)
            for specs in realizations.values()
        ]
    part: dict = {"inj_matches": [], "n_tiles": 0}
    part["rejections"] = [
        {
            "injection_id": injection_id,
            "realization_id": result.realization_id,
            "line_id": line.ref.line_id,
            "reason": reason,
        }
        for result in results
        for injection_id, reason in result.rejected
    ]
    part["n_accepted"] = sum(len(result.truths) for result in results)
    ctx = MatchContext(line, context, library.groups, level_w_db)
    for recipe in settings.recipes:
        with timer.phase("l1"):
            jobs = injected_jobs(line, results, recipe)
        with timer.phase("infer"):
            detections = run_jobs(jobs, detectors, settings.batch_size)
        part["n_tiles"] += sum(len(job.tile_indices) for job in jobs)
        part["n_l1_builds"] = part.get("n_l1_builds", 0) + len(jobs)
        with timer.phase("match"):
            for (j, fold), tile_boxes in detections.items():
                job = jobs[j]
                part["inj_matches"] += match_rows(
                    ctx, job, fold, results[job.tag], tile_boxes
                )
    return part


# ============================================================ stage runner


def tracked_inputs(settings: RunSettings) -> list[Path]:
    """Files a full run reads that git must track unchanged."""
    here = Path(__file__).resolve().parent
    paths = [
        Path(__file__),
        here / "m31_d1_common.py",
        here / "m32_aihub_perturb.py",
        here / "m32_pipe_responses.py",
        here / "m32_recipe_inputs.py",
        here / "m32_recipe_match.py",
        inputs.STAGE_A_4TU,
        inputs.STAGE_A_IDS,
        inputs.APEX_CANDIDATES,
    ]
    if settings.stage is Stage.INJECTED:
        paths.append(settings.f2_record)
    return paths


def git_blob(path: Path) -> str | None:
    """``git hash-object`` of a working-tree file (the code that ran)."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO), "hash-object", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=common.GIT_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def new_run_identity(settings: RunSettings, extra_sha: Mapping[str, str]) -> dict:
    """``run_identity`` of [CVR] 4-2 (git, versions, GPU, inputs, seed)."""
    identity = common.new_run_identity()
    identity["gpu"] = common.gpu_name(settings.device)
    identity["seed"] = inputs.SEED
    identity["recipes"] = list(settings.recipes)
    identity["recipes_sha256"] = inputs.recipes_sha256(settings.recipes)
    identity["code_blobs"] = {
        name: git_blob(REPO / name)
        for name in (
            "src/gpr_detection/preprocess/to_image.py",
            "src/gpr_detection/eval/inject.py",
            "src/gpr_detection/pipeline.py",
            "src/gpr_detection/detect/yolo.py",
        )
    }
    identity["inputs_sha256"] = dict(extra_sha)
    identity["merge_config"] = asdict(DEFAULT_MERGE_CONFIG)
    identity["confidence"] = CONFIDENCE
    identity["batch_size"] = settings.batch_size
    return identity


def prepare_stage_dir(settings: RunSettings) -> None:
    """Refuse to mix runs: new folder, ``--overwrite`` or ``--resume``."""
    stage_dir = settings.stage_dir
    if stage_dir.exists() and any(stage_dir.iterdir()):
        if settings.overwrite:
            shutil.rmtree(stage_dir)
        elif not settings.resume:
            raise inputs.InputError(
                f"output exists (use --overwrite or --resume): {stage_dir}"
            )
    (stage_dir / "parts").mkdir(parents=True, exist_ok=True)


def check_resume_identity(settings: RunSettings, settings_sha: str) -> None:
    """A resumed stage must have the same settings and input hashes."""
    marker = settings.stage_dir / "stage_settings.json"
    if marker.is_file():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("settings_sha256") != settings_sha:
            raise inputs.InputError("resume with other settings or inputs refused")
        return
    _write_json(marker, {"settings_sha256": settings_sha})


def settings_sha256(settings: RunSettings, extra_sha: Mapping[str, str]) -> str:
    """sha256 over the settings that change results and the input hashes."""
    document = {
        "stage": settings.stage.value,
        "recipes": settings.recipes,
        "folds": settings.folds,
        "flip": settings.flip,
        "limit_lines": settings.limit_lines,
        "smoke": settings.smoke,
        "provisional": settings.provisional,
        "batch_size": settings.batch_size,
        "inputs": dict(sorted(extra_sha.items())),
    }
    return hashlib.sha256(
        json.dumps(document, sort_keys=True).encode("utf-8")
    ).hexdigest()


def run_lines(
    settings: RunSettings,
    lines: Sequence[inputs.LineRef],
    evaluate: Callable[[LineData], dict],
    started: float,
) -> list[dict]:
    """Evaluate each line once (parts are reused on ``--resume``)."""
    parts = []
    for number, ref in enumerate(lines, start=1):
        part_path = settings.stage_dir / "parts" / f"{ref.key}.json"
        if part_path.is_file():
            parts.append(json.loads(part_path.read_text(encoding="utf-8")))
            continue
        line_started = time.time()
        part = evaluate(load_line(ref))
        part["seconds"] = round(time.time() - line_started, 2)
        _write_part(part_path, part)
        parts.append(part)
        print(
            f"{settings.stage.value} {number}/{len(lines)} {ref.line_id}: "
            f"{part.get('n_tiles', 0)} tiles, {part['seconds']} s "
            f"(elapsed {time.time() - started:.0f} s)",
            flush=True,
        )
    return parts


def _selected(lines: Sequence[inputs.LineRef], settings: RunSettings) -> list:
    return list(
        lines if settings.limit_lines is None else lines[: settings.limit_lines]
    )


GATE_BATCH_SIZE = 1
TILE_ORDER = (
    "lines in list order; per line the recipes in order, each original then "
    "mirrored, tiles left to right; that sequence is cut into batches of "
    "batch_size, the same for every fold"
)


def gate_detections(
    line: LineData, detectors: Mapping[int, object], batch_size: int
) -> tuple[DetectionImage, dict[int, list[LineBox]]]:
    """r0 tile boxes of one K line per fold with a given batch size."""
    image = make_detection_image(line.radargram, inputs.recipe_config(inputs.R0))
    job = ImageJob(inputs.R0, ORIGINAL, image, tuple(range(len(image.tiles))))
    detections = run_jobs([job], detectors, batch_size)
    return image, {fold: boxes for (_, fold), boxes in detections.items()}


def max_score_difference(
    first: Mapping[int, Sequence[LineBox]], second: Mapping[int, Sequence[LineBox]]
) -> tuple[float, int]:
    """Largest score gap of tile boxes paired by tile and rank; unpaired tiles."""
    largest, unpaired = 0.0, 0
    for fold, boxes in first.items():
        by_tile: dict[int, list[list[float]]] = {}
        for which, group in enumerate((boxes, second[fold])):
            for box in group:
                by_tile.setdefault(box.tile_index, [[], []])[which].append(box.score)
        for scores_a, scores_b in by_tile.values():
            if len(scores_a) != len(scores_b):
                unpaired += 1
                continue
            for a, b in zip(sorted(scores_a), sorted(scores_b), strict=True):
                largest = max(largest, abs(a - b))
    return largest, unpaired


def run_identity_gate(
    settings: RunSettings,
    k_refs: Sequence[inputs.LineRef],
    detectors: Mapping[int, object],
    apexes: Sequence[inputs.KnownApex],
) -> dict:
    """C-8 gate: r0 x K lines x folds at batch size 1 must reproduce D3.

    Also measures the batch-size effect on the same tiles (report only).
    """
    rows: list[dict] = []
    box_counts: dict[str, int] = {}
    largest, unpaired = 0.0, 0
    for ref in k_refs:
        line = load_line(ref)
        image, single = gate_detections(line, detectors, GATE_BATCH_SIZE)
        _, batched = gate_detections(line, detectors, settings.batch_size)
        for fold, boxes in single.items():
            merged = merge_with_grid(image, boxes)
            box_counts[f"{ref.line_id}|{fold}"] = len(merged)
            rows += apex_rows(
                line, inputs.R0, fold, scored(merged), image.period_samples, apexes
            )
        gap, missing = max_score_difference(single, batched)
        largest, unpaired = max(largest, gap), unpaired + missing
    return {
        "batch_size": GATE_BATCH_SIZE,
        "apex_rows": rows,
        "box_counts": box_counts,
        "result": identity_gate(rows),
        "tile_score_max_difference_batch_vs_1": largest,
        "tiles_with_other_box_count_batch_vs_1": unpaired,
    }


def batch_path_report(gate: Mapping, parts: Sequence[Mapping]) -> dict:
    """Measurement (batch) against the gate (batch 1) on r0 x K; report only."""
    single = {
        (row["blind_id"], row["fold"]): bool(row["hit"]) for row in gate["apex_rows"]
    }
    measured = {
        (row["blind_id"], row["fold"]): bool(row["hit"])
        for part in parts
        for row in part["apex_hits"]
        if row["recipe"] == inputs.R0
    }
    shared = single.keys() & measured.keys()
    return {
        "apex_hit_differences": sum(single[key] != measured[key] for key in shared),
        "apex_fold_pairs_compared": len(shared),
        "box_count_difference_conf025": _box_count_difference(gate, parts),
        "tile_score_max_difference": gate["tile_score_max_difference_batch_vs_1"],
        "tiles_with_other_box_count": gate["tiles_with_other_box_count_batch_vs_1"],
        "note": "report only; every recipe runs the same batched path",
    }


def _box_count_difference(gate: Mapping, parts: Sequence[Mapping]) -> int:
    counts = gate["box_counts"]
    measured: dict[str, int] = {}
    for part in parts:
        for row in part["boxes"]:
            if row["recipe"] == inputs.R0 and row["orientation"] == ORIGINAL:
                key = f"{row['line_id']}|{row['fold']}"
                measured[key] = measured.get(key, 0) + 1
    return sum(abs(measured.get(key, 0) - value) for key, value in counts.items())


def _gate_or_load(
    settings: RunSettings,
    lines: Sequence[inputs.LineRef],
    detectors: Mapping[int, object],
    apexes: Sequence[inputs.KnownApex],
) -> dict | None:
    """Run (or on ``--resume`` reread) the batch-1 gate on the selected K lines."""
    path = settings.stage_dir / "gate.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    k_refs = [ref for ref in lines if ref.role == inputs.ROLE_K]
    if inputs.R0 not in settings.recipes or not k_refs:
        return None
    gate = run_identity_gate(settings, k_refs, detectors, apexes)
    _write_json(path, gate)
    return gate


def run_uninjected(settings: RunSettings, started: float) -> tuple[str, StageCounts]:
    """P5a: batch-1 D3 gate first, then boxes, M4, conf counts, M5 and M6."""
    hashes = inputs.assert_list_hashes()
    hashes.update(assert_weights(settings.run_root, settings.folds))
    lines = _selected(inputs.uninjected_lines(), settings)
    apexes = inputs.known_apexes()
    prepare_stage_dir(settings)
    check_resume_identity(settings, settings_sha256(settings, hashes))
    identity = new_run_identity(settings, hashes)
    identity["tile_order"] = TILE_ORDER
    detectors = load_detectors(settings)
    gate = _gate_or_load(settings, lines, detectors, apexes)
    gate_passed = gate is not None and gate["result"]["pass"]
    if settings.is_full and not gate_passed:
        summary = {
            "schema": SCHEMA,
            "stage": Stage.UNINJECTED.value,
            "status": "gate_failed",
            "full_run": True,
            "run_identity": {**identity, "finished_at": common.local_now()},
            "identity_gate": None if gate is None else gate["result"],
        }
        _write_json(settings.stage_dir / "summary.json", summary)
        return "gate_failed", StageCounts()
    parts = run_lines(
        settings,
        lines,
        lambda line: evaluate_uninjected_line(line, detectors, settings, apexes),
        started,
    )
    assemble_stage_files(settings, parts)
    _write_rows(
        settings.stage_dir / "lines.csv",
        LINE_COLUMNS,
        (row for part in parts for row in part["line"]),
    )
    measured_m6 = identity_gate(row for part in parts for row in part["apex_hits"])
    if gate is not None:
        identity["batch_path"] = batch_path_report(gate, parts)
    status = "ok"
    summary = {
        "schema": SCHEMA,
        "stage": Stage.UNINJECTED.value,
        "status": status,
        "full_run": settings.is_full,
        "gated": settings.is_full,
        "run_identity": {**identity, "finished_at": common.local_now()},
        "identity_gate": None if gate is None else gate["result"],
        "measured_r0_m6_report_only": measured_m6,
        "n_lines": len(parts),
        "flip": settings.flip,
        "resources": resources(settings, started),
        "line_seconds": {p["line"][0]["line_id"]: p["seconds"] for p in parts},
    }
    _write_json(settings.stage_dir / "summary.json", summary)
    counts = StageCounts(len(parts), sum(part["n_tiles"] for part in parts))
    return status, counts


def run_plan(settings: RunSettings, started: float) -> tuple[str, StageCounts]:
    """Zone (c') from P5a, seeded plans, injection table, costs, F2 draft."""
    hashes = inputs.assert_list_hashes()
    library = responses_io.load_library()
    hashes["responses"] = responses_io.library_sha256(library)
    if settings.provisional:
        boxes: dict[str, list[GridBox]] = {}
    else:
        boxes = uninjected_exclusions(require_complete_uninjected(settings))
    lines = _selected(inputs.background_lines(), settings)
    candidates = inputs.candidate_apexes()
    prepare_stage_dir(settings)
    jobs = [(ref, line_exclusions(ref.line_id, boxes, candidates)) for ref in lines]
    results = []
    with ProcessPoolExecutor(
        max_workers=settings.workers, initializer=_init_plan_worker
    ) as executor:
        # map keeps the line order, so the table does not depend on workers.
        for number, result in enumerate(executor.map(_plan_worker, jobs), start=1):
            results.append(result)
            print(f"plan {number}/{len(lines)} {result.report['line_id']}", flush=True)
    return _finish_plan(settings, started, results, boxes, hashes)


_WORKER_LIBRARY: list[responses_io.ResponseLibrary] = []


def _init_plan_worker() -> None:
    """Load the responses once per worker process."""
    _WORKER_LIBRARY.append(responses_io.load_library())


def _plan_worker(job: tuple[inputs.LineRef, inject.LineExclusions]) -> LinePlanResult:
    ref, exclusions = job
    return plan_line(load_line(ref), exclusions, _WORKER_LIBRARY[0])


def _model_groups() -> dict[str, responses_io.ModelGroup]:
    return dict(responses_io.MODEL_GROUPS)


def _finish_plan(
    settings: RunSettings,
    started: float,
    results: Sequence[LinePlanResult],
    boxes: Mapping[str, Sequence[GridBox]],
    hashes: dict[str, str],
) -> tuple[str, StageCounts]:
    stage_dir = settings.stage_dir
    specs = [spec for result in results for spec in result.specs]
    positions = [row for result in results for row in result.positions]
    reports = [result.report for result in results]
    level_w_db, level_report = select_level_w(positions)
    pi_bottom = natural_bottom_share(results, _model_groups())
    classify_positions(positions, level_w_db)
    table_sha = inject.write_injection_table(stage_dir / "injection_table.csv", specs)
    cost_sha = inject.write_cost_table(
        stage_dir / "cost_table.csv", [result.cost for result in results]
    )
    exclusions_sha = inputs.write_exclusions(stage_dir / "exclusions.csv", boxes)
    _write_rows(stage_dir / "positions.csv", POSITION_COLUMNS, positions)
    samples = sample_report(positions, reports)
    samples["level_w_db"] = level_w_db
    samples["level_width_report"] = level_report
    samples["pi_bottom"] = pi_bottom
    samples["in_distribution_by_cell"] = count_by(
        [p for p in positions if p["in_distribution"]], ("dataset", "model_group")
    )
    samples["period_check_failed_k13"] = sum(
        r["period_check_failed_k13"] for r in reports
    )
    samples["period_check_failed_k08"] = sum(
        r["period_check_failed_k08"] for r in reports
    )
    samples["at_level_w"] = level_summary(positions, level_w_db)
    if settings.report_all_levels:
        samples["all_levels"] = {
            level_tag(level): level_summary(positions, level) for level in LEVELS
        }
    status = "ok" if samples["sample_ok"] else "sample_short"
    f2 = {
        "level_w_db": level_w_db,
        "pi_bottom": pi_bottom,
        "primary_stratum": samples["primary_stratum"],
        "bottom_rule": samples["bottom_rule"],
        "injection_table_sha256": table_sha,
        "exclusions_sha256": exclusions_sha,
        "cost_table_sha256": cost_sha,
        "positions_sha256": inputs.sha256_file(stage_dir / "positions.csv"),
        "seed": inject.PLAN_SEED,
        "provisional": settings.provisional,
    }
    if not settings.provisional:
        _write_json(stage_dir / "f2_record_draft.json", f2)
    _write_json(
        stage_dir / "plan_report.json",
        {
            "schema": SCHEMA,
            "stage": "plan",
            "status": status,
            "provisional": settings.provisional,
            "run_identity": {
                **common.new_run_identity(),
                "seed": inject.PLAN_SEED,
                "finished_at": common.local_now(),
            },
            "inputs_sha256": hashes,
            "outputs_sha256": f2,
            "samples": samples,
            "lines": reports,
            "seconds": round(time.time() - started, 1),
        },
    )
    return status, StageCounts(len(results), 0, len(specs))


def run_injected(settings: RunSettings, started: float) -> tuple[str, StageCounts]:
    """P5b: inject the F2 table and match every recipe and fold."""
    hashes = inputs.assert_list_hashes()
    hashes.update(assert_weights(settings.run_root, settings.folds))
    f2 = read_f2_inputs(settings)
    hashes.update(f2.sha256)
    library = responses_io.load_library()
    hashes["responses"] = responses_io.library_sha256(library)
    by_line = group_specs(f2.specs)
    refs = {ref.line_id: ref for ref in inputs.background_lines()}
    lines = _selected([refs[line_id] for line_id in by_line], settings)
    candidates = inputs.candidate_apexes()
    prepare_stage_dir(settings)
    check_resume_identity(settings, settings_sha256(settings, hashes))
    identity = new_run_identity(settings, hashes)
    detectors = load_detectors(settings)

    def evaluate(line: LineData) -> dict:
        exclusions = line_exclusions(line.ref.line_id, f2.exclusions, candidates)
        return evaluate_injected_line(
            line,
            by_line[line.ref.line_id],
            exclusions,
            library,
            detectors,
            settings,
            level_w_db=f2.decisions["level_w_db"],
        )

    parts = run_lines(settings, lines, evaluate, started)
    assemble_stage_files(settings, parts)
    rejections = [row for part in parts for row in part["rejections"]]
    _write_rows(settings.stage_dir / "rejections.csv", REJECTION_COLUMNS, rejections)
    summary = {
        "schema": SCHEMA,
        "stage": Stage.INJECTED.value,
        "status": "ok",
        "full_run": settings.is_full,
        "run_identity": {**identity, "finished_at": common.local_now()},
        "n_lines": len(parts),
        "n_accepted": sum(part["n_accepted"] for part in parts),
        **f2.decisions,
        "rejection_reasons": count_by(rejections, ("reason",)),
        "rejection_reasons_by_snr_k": count_by(
            with_condition(rejections), ("reason", "snr_db", "k")
        ),
        "resources": resources(settings, started),
    }
    _write_json(settings.stage_dir / "summary.json", summary)
    n_rows = summary["n_accepted"] + len(rejections)
    return "ok", StageCounts(len(parts), sum(p["n_tiles"] for p in parts), n_rows)


def resources(settings: RunSettings, started: float) -> dict:
    return {
        "seconds_total": round(time.time() - started, 1),
        "peak_vram_reserved_mb": peak_vram_mb(settings.device),
    }


def peak_vram_mb(device: str) -> float | None:
    """Peak reserved CUDA memory of this process (None on CPU)."""
    if device.strip().lower() == "cpu":
        return None
    import torch

    return round(torch.cuda.max_memory_reserved() / 2**20, 1)


# =================================================================== smoke


def smoke_lines() -> list[inputs.LineRef]:
    """Fewest-tile, middle (median of the 2+ tile lines) and most-tile 4TU lines."""
    tiles = {
        row["line"]: int(row["n_tiles"])
        for row in inputs.read_csv(inputs.CONTENT_HEIGHT_LINES)
        if row["dataset"] == inputs.DATASET_4TU
    }
    candidates = [
        ref
        for ref in inputs.background_lines()
        if ref.dataset == inputs.DATASET_4TU and ref.line_id in tiles
    ]
    ordered = sorted(candidates, key=lambda ref: (tiles[ref.line_id], ref.line_id))
    multi_tile = [ref for ref in ordered if tiles[ref.line_id] >= 2]
    middle = (
        multi_tile[len(multi_tile) // 2] if multi_tile else ordered[len(ordered) // 2]
    )
    return [ordered[0], middle, ordered[-1]]


def run_smoke(settings: RunSettings, started: float) -> tuple[str, StageCounts]:
    """Time and VRAM of both paths on three lines; writes ``smoke_estimate.json``."""
    import m32_recipe_smoke as smoke  # keeps the estimate code out of full runs

    return smoke.run(settings, started)


# ===================================================================== CLI


def parse_args(argv: Sequence[str] | None) -> RunSettings:
    """Command line -> ``RunSettings``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=[s.value for s in Stage], default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--provisional", action="store_true", help="plan without P5a")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--device", default="0", help="'0' (GPU) or 'cpu'")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--recipes", default=",".join(inputs.STAGE_C_RECIPES))
    parser.add_argument("--folds", default=",".join(map(str, inputs.FOLDS)))
    parser.add_argument("--no-flip", action="store_true", help="skip M5 (P5a)")
    parser.add_argument("--limit-lines", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--run-root", type=Path, default=common.RUN_ROOT)
    parser.add_argument("--f2-record", type=Path, default=F2_RECORD)
    parser.add_argument(
        "--workers", type=int, default=1, help="plan stage processes (CPU, <= 4)"
    )
    parser.add_argument(
        "--cost-table", type=Path, default=None, help="smoke: plan cost table"
    )
    parser.add_argument(
        "--report-all-levels",
        action="store_true",
        help="plan: stratum counts of all four levels (report only)",
    )
    args = parser.parse_args(argv)
    if args.smoke == (args.stage is not None):
        parser.error("give exactly one of --stage or --smoke")
    recipes = tuple(item.strip() for item in args.recipes.split(",") if item.strip())
    unknown = set(recipes) - set(inputs.STAGE_C_RECIPES)
    if unknown:
        parser.error(f"unknown recipes {sorted(unknown)}")
    return RunSettings(
        stage=Stage(args.stage) if args.stage else Stage.UNINJECTED,
        out_root=args.out_root or (SMOKE_OUT_ROOT if args.smoke else DEFAULT_OUT_ROOT),
        device=args.device,
        batch_size=args.batch_size,
        recipes=(SMOKE_RECIPE,) if args.smoke else recipes,
        folds=(SMOKE_FOLD,) if args.smoke else _int_list(args.folds),
        flip=not args.no_flip,
        limit_lines=args.limit_lines,
        smoke=args.smoke,
        provisional=args.provisional,
        resume=args.resume,
        overwrite=args.overwrite or args.smoke,
        allow_dirty=args.allow_dirty or args.smoke,
        run_root=args.run_root,
        f2_record=args.f2_record,
        workers=max(1, min(args.workers, MAX_PLAN_WORKERS)),
        cost_table=args.cost_table,
        report_all_levels=args.report_all_levels,
    )


def _int_list(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item.strip())


def run(settings: RunSettings, started: float) -> tuple[str, StageCounts]:
    """Dispatch one stage after the clean-tree check of full runs."""
    if not settings.allow_dirty:
        problems = stage_b.tracked_tree_problems(tracked_inputs(settings), REPO)
        if problems:
            raise inputs.InputError(f"tree not clean for a run: {problems}")
    if settings.smoke:
        return run_smoke(settings, started)
    stage_runner = {
        Stage.UNINJECTED: run_uninjected,
        Stage.PLAN: run_plan,
        Stage.INJECTED: run_injected,
    }[settings.stage]
    return stage_runner(settings, started)


def print_final_line(
    status: str, settings: RunSettings | None, counts: StageCounts, started: float
) -> None:
    """Closing stdout JSON line; never carries metric values."""
    stage_name, out_dir = None, None
    if settings is not None and settings.smoke:
        stage_name, out_dir = "smoke", str(settings.out_root)
    elif settings is not None:
        stage_name, out_dir = settings.stage.value, str(settings.stage_dir)
    line = {
        "status": status,
        "stage": stage_name,
        "out_dir": out_dir,
        "n_lines": counts.n_lines,
        "n_tiles": counts.n_tiles,
        "n_injections": counts.n_injections,
        "elapsed_s": round(time.time() - started, 1),
    }
    print(json.dumps(line, ensure_ascii=False), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry; returns the exit code (see module docstring)."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    started = time.time()
    settings = parse_args(argv)
    counts = StageCounts()
    try:
        status, counts = run(settings, started)
    except inputs.InputError as error:
        logger.error("input error: %s", error)
        status = "input_error"
    except Exception:
        logger.exception("internal error")
        status = "error"
    print_final_line(status, settings, counts, started)
    return EXIT_CODES[status]


if __name__ == "__main__":
    sys.exit(main())
