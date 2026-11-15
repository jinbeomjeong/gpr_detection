"""M3-3 R3 pseudo-band runner: ``m32_recipe_eval.py --injectant band``.

Spec: ``docs/experiments/m33_runner_spec.md`` v0.3.6 §3 ([RUN]); physics
``docs/processing/m32_em_review.md`` appendix D ([EMR]). The P4 value is
computed by ``m33_r3_p4``; the common checks (sealed PP, judgment, end tree,
LF sha) are those of ``m32_recipe_eval`` and ``m33_inputs``.

Stages (``--stage`` with ``--injectant band``; folders under ``--out-root``,
default ``runs/eval/m33/r3``)

1. ``uninjected`` -> ``uninjected_ids/``: r0 x folds on the IDS 60 lines,
   every merged box with its PP outcome (``pp`` ``pre`` / ``post``). 4TU
   un-injected boxes come from R2 (``--r2-root``, same box columns).
2. ``plan`` -> ``plan/``: zone (c') from the un-injected boxes (every class,
   PP-off rows), seeded band positions per line (``inject.plan_band_positions``,
   lengths {40, 64, 100, 150, 200, 300}, seed 20261115), the band injection
   table, ``positions.csv`` (truth box L0 and L1, strata, PP exposure),
   ``exclusions.csv``, ``plan_report.json`` and ``c1b_record_draft.json``.
   The C1b commit copies the draft to ``analysis/detection/m33_r3_c1b.json``.
3. ``injected`` -> ``injected/``: the table fixed at C1b (LF sha asserted),
   ``inject_band_realization`` per realization, r0 L1 (built once and shared
   by the folds), batched inference on every tile (the r0 chain has
   whole-line steps, see :func:`inference_tiles`), merge, PP (sealed set),
   then the horizontal-span match, ghost bands, survival ratio, summary
   tables (main cells and the ``bottom`` report) and the P4 value
   (``p4_adoption_input``; ``P4DefectError`` -> exit 1). Phase seconds go
   to ``timings`` in each stage json and, names and seconds only, to stdout.

``--smoke`` runs the three stages on three IDS 400 MHz lines and fold 0 in
``runs/eval/m33/r3_smoke`` and always ends ``not_judgeable`` (exit 1).

Blind: recall, ghost and survival values go to the files only; stdout carries
stages, counts, seconds and paths (``m32_recipe_eval.print_final_line``).
"""

import json
import logging
import math
import statistics
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import m31_d1_common as common
import m32_pipe_responses as responses_io
import m32_recipe_eval as runner
import m32_recipe_inputs as inputs
import m33_inputs
import m33_r3_p4 as p4
import numpy as np

from gpr_detection.detect.base import PixelBox
from gpr_detection.eval import inject
from gpr_detection.io.classes import DetectionClass
from gpr_detection.io.radargram import GridBox, Radargram
from gpr_detection.pipeline import (
    DEFAULT_MERGE_CONFIG,
    PP_SETS,
    ContentRows,
    LineBox,
    PPAction,
    PPConfig,
    PPOutcome,
    PPSetId,
    apply_pp,
    content_rows_for,
    grid_box_to_image_box,
    image_box_to_grid_box,
    merge_line_boxes,
)
from gpr_detection.preprocess.timezero import find_time_zero
from gpr_detection.preprocess.to_image import DetectionImage, make_detection_image

SCHEMA = "m33_r3_band/1"
logger = logging.getLogger("m33_r3_band")
MAIN_ANTENNA_MHZ = 400.0
EXCLUDED_ANTENNA_MHZ = 2000.0
FREQ_MAIN = "main_400"
FREQ_REPORT = "report_other"
FREQ_EXCLUDED = "excluded_2ghz"
PREFERRED_MODELS_BY_ANTENNA: dict[float | None, tuple[str, ...]] = {
    900.0: ("M11", "M14", "M16"),
}
"""IDS antennas whose lines visit some models first ([EMR] D-6)."""
PX_EDGES = (64.0, 192.0)
"""Plan px strata ``< 64 / 64-192 / > 192`` px (report only, [EMR] D-3)."""
PX_STRATA = ("lt64", "64_192", "gt192")
SMOKE_N_LINES = 3
PHASE_UNINJECTED_INFER = "uninjected_infer"
PHASE_UNINJECTED_L1 = "uninjected_l1"
PHASE_PLAN = "plan"
PHASE_INJECT = "injected_inject"
PHASE_L1 = "injected_l1"
PHASE_INFER = "injected_infer"
PHASE_PP = "injected_merge_pp"
PHASE_SURVIVAL = "injected_step4_zero"
"""Step (4) of each realization for the survival ratio, and its zero."""
PHASE_MATCH = "injected_match"
PHASE_P4 = "p4"
"""``timings`` keys (seconds summed over lines; stdout prints names and seconds)."""
PP_PRE = "pre"
PP_POST = "post"
REPORT_ONLY = {
    "lw_strata": ["05_067", "ge067"],
    "backgrounds": [inputs.DATASET_4TU],
    "ids_freq_strata": [FREQ_REPORT],
}
"""Report-only parts of the R3 summary ([RUN] §3-2, last item)."""

BOX_COLUMNS = (
    "fold",
    "line_id",
    "dataset",
    "realization_id",
    "box_index",
    "pp",
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
    "pp_action",
    "pp_category",
    "n_tiles_seen",
    "tile_truncated",
)
"""Merged boxes with their PP outcome; R2 writes the same columns for 4TU."""
POSITION_COLUMNS = (
    "position_id",
    "line_id",
    "dataset",
    "ids_freq_stratum",
    "group",
    "response_id",
    "model_group",
    "band_center_trace",
    "band_length_traces",
    "taper_traces",
    "n_traces",
    "w_eff",
    "l_over_w_eff",
    "lw_stratum",
    "length_px",
    "px_stratum",
    "l_over_n",
    "apex_sample",
    "i0",
    "i1",
    "k0",
    "k1",
    "truth_x1_px",
    "truth_y1_px",
    "truth_x2_px",
    "truth_y2_px",
    "y_bot_px",
    "period_px",
    "zero_px",
    "box_clipped",
    "wing_past_record_end",
    "main_cell",
    "bottom_stratum",
    "in_horizontal_box_zone",
    "p4_scope",
    "pp_exposed_rule",
    "in_pp_c_window",
)
TRUTH_COLUMNS = (
    "injection_id",
    "realization_id",
    "position_id",
    "line_id",
    "dataset",
    "ids_freq_stratum",
    "response_id",
    "model_group",
    "mode",
    "band_length_traces",
    "band_center_trace",
    "length_px",
    "w_eff",
    "l_over_w_eff",
    "lw_stratum",
    "px_stratum",
    "l_over_n",
    "taper_traces",
    "i0",
    "i1",
    "k0",
    "k1",
    "apex_sample",
    "gain",
    "noise_rms",
    "signal_amplitude",
    "n_clipped_cells",
    "max_clipped_injection_ratio",
    "polarity",
    "snr_db",
    "k",
    "box_clipped",
    "wing_past_record_end",
    "main_cell",
    "bottom_stratum",
    "in_horizontal_box_zone",
    "period_ringing",
    "survival_ratio",
)
"""Truth table ([RUN] §3-1 5) plus the survival ratio (§3-1 7, fold-free)."""
MATCH_COLUMNS = (
    "fold",
    "injection_id",
    "realization_id",
    "line_id",
    "dataset",
    "model_group",
    "lw_stratum",
    "px_stratum",
    "snr_db",
    "polarity",
    "main_cell",
    "in_horizontal_box_zone",
    "length_traces",
    "covered_pre",
    "covered_post",
    "recall_pre",
    "recall_post",
    "n_recall_boxes",
    "covered_any_class_pre",
    "covered_any_class_post",
    "ghost_pairs_pre",
    "ghost_pairs_post",
)
GHOST_COLUMNS = (
    "fold",
    "realization_id",
    "line_id",
    "dataset",
    "ghost_n_boxes_pre",
    "ghost_n_boxes_post",
    "ghost_length_traces_pre",
    "ghost_length_traces_post",
)
REJECTION_COLUMNS = ("injection_id", "realization_id", "line_id", "reason")


# ------------------------------------------------------------ line lists


def freq_stratum_of(antenna_mhz: float) -> str:
    """``ids_freq_stratum`` of an IDS line ([EMR] D-6)."""
    if antenna_mhz == MAIN_ANTENNA_MHZ:
        return FREQ_MAIN
    if antenna_mhz == EXCLUDED_ANTENNA_MHZ:
        return FREQ_EXCLUDED
    return FREQ_REPORT


def ids_antennas() -> dict[str, float]:
    """``line_id -> antenna_MHz`` of the IDS 60 stage-A lines."""
    return {
        row["line"]: float(row["antenna_MHz"])
        for row in inputs.read_csv(inputs.STAGE_A_IDS)
    }


def ids_freq_strata() -> dict[str, str]:
    """``line_id -> ids_freq_stratum`` of the IDS 60 stage-A lines."""
    return {
        line_id: freq_stratum_of(antenna_mhz)
        for line_id, antenna_mhz in ids_antennas().items()
    }


def preferred_models(antenna_mhz: float | None) -> tuple[str, ...]:
    """Models a line visits first: the 500 MHz ones on the 900 MHz IDS line.

    Runner spec §3-1 1 and [EMR] D-6: M11, M14 and M16 (500 MHz) are closer
    to the 700 MHz line period than the 400 MHz models.
    """
    return PREFERRED_MODELS_BY_ANTENNA.get(antenna_mhz, ())


def _ids_lines() -> list[inputs.LineRef]:
    return [
        ref for ref in inputs.background_lines() if ref.dataset == inputs.DATASET_IDS
    ]


def smoke_lines(strata: Mapping[str, str]) -> list[inputs.LineRef]:
    """Three IDS 400 MHz lines: fewest, middle and most tiles."""
    main = [ref for ref in _ids_lines() if strata.get(ref.line_id) == FREQ_MAIN]
    tiles = {
        row["line"]: int(row["n_tiles"])
        for row in inputs.read_csv(inputs.CONTENT_HEIGHT_LINES)
        if row["dataset"] == inputs.DATASET_IDS
    }
    ordered = sorted(main, key=lambda ref: (tiles.get(ref.line_id, 0), ref.line_id))
    if len(ordered) <= SMOKE_N_LINES:
        return ordered
    return [ordered[0], ordered[len(ordered) // 2], ordered[-1]]


def uninjected_ids_lines(
    settings: runner.RunSettings, strata: Mapping[str, str]
) -> list[inputs.LineRef]:
    """The IDS 60 lines (2 GHz included: zone (c') and false alarms only)."""
    if settings.smoke:
        return smoke_lines(strata)
    return runner._selected(_ids_lines(), settings)


def band_background_lines(
    settings: runner.RunSettings, strata: Mapping[str, str]
) -> list[inputs.LineRef]:
    """4TU stage A 150 + K 27, IDS 56 (the four 2 GHz lines are not injected)."""
    if settings.smoke:
        return smoke_lines(strata)
    lines: list[inputs.LineRef] = []
    if inputs.DATASET_4TU in settings.backgrounds:
        lines += [
            ref
            for ref in inputs.background_lines()
            if ref.dataset == inputs.DATASET_4TU
        ]
        lines += inputs.k_lines()
    if inputs.DATASET_IDS in settings.backgrounds:
        lines += [
            ref for ref in _ids_lines() if strata.get(ref.line_id) != FREQ_EXCLUDED
        ]
    return runner._selected(lines, settings)


# ---------------------------------------------------------------- timings


def summed_timings(parts: Iterable[Mapping]) -> dict[str, float]:
    """Phase seconds of every line part, summed."""
    total: Counter[str] = Counter()
    for part in parts:
        total.update(part.get("timings", {}))
    return dict(total)


def report_timings(seconds: Mapping[str, float]) -> None:
    """Stdout: phase names and seconds only (no metric value)."""
    for name, value in seconds.items():
        print(f"timing {name}: {value:.1f} s", flush=True)


# ------------------------------------------------------- strata helpers


def px_stratum(length_px: float) -> str:
    """Plan px stratum of a band length (report only)."""
    if length_px < PX_EDGES[0]:
        return PX_STRATA[0]
    if length_px <= PX_EDGES[1]:
        return PX_STRATA[1]
    return PX_STRATA[2]


def sealed_config(context: runner.RunContext) -> tuple[PPSetId, PPConfig]:
    """The C0b set asserted by ``m33_inputs.assert_sealed_pp``."""
    set_id = PPSetId(context.sealed_pp["pp_set_id"])
    return set_id, PP_SETS[set_id]


def content_of(context: inject.LineContext) -> ContentRows:
    """``ContentRows`` of the line r0 image from the injection context.

    The same values ``content_rows_for(image, zero)`` gives for the r0 image
    (``y_bot = n_s / sigma_y``, ``P_ell = P_line / sigma_y``).
    """
    grid = context.grid_map
    zero = context.zero.k_zero
    return ContentRows(
        y_bot_px=grid.n_samples / grid.samples_per_px,
        period_px=context.period_samples / grid.samples_per_px,
        zero_px=None if zero is None else zero / grid.samples_per_px,
    )


# ------------------------------------------------------------ box rows


def _box_fields(box: LineBox, grid: GridBox | None) -> dict:
    fields = {
        "class_id": box.class_id,
        "class_name": box.class_name,
        "score": runner._exact(box.score),
        "x1": runner._exact(box.box.x1),
        "y1": runner._exact(box.box.y1),
        "x2": runner._exact(box.box.x2),
        "y2": runner._exact(box.box.y2),
        "n_tiles_seen": box.n_tiles_seen,
        "tile_truncated": box.tile_truncated,
    }
    if grid is None:
        return fields | {"i0": "", "i1": "", "k0": "", "k1": "", "in_grid": False}
    return fields | {
        "i0": grid.i0,
        "i1": grid.i1,
        "k0": grid.k0,
        "k1": grid.k1,
        "in_grid": True,
    }


def pp_box_rows(
    base: Mapping, image: DetectionImage, outcomes: Sequence[PPOutcome]
) -> list[dict]:
    """``pre`` row of every merged box and ``post`` row of every kept box."""
    rows = []
    grid_map = image.grid_map
    for index, outcome in enumerate(outcomes):
        category = outcome.decision.category
        shared = {
            **base,
            "box_index": index,
            "pp_action": outcome.action.value,
            "pp_category": "" if category is None else category.value,
        }
        original = outcome.original
        rows.append(
            shared
            | {"pp": PP_PRE}
            | _box_fields(original, image_box_to_grid_box(original.box, grid_map))
        )
        if outcome.output is not None:
            output = outcome.output
            rows.append(
                shared
                | {"pp": PP_POST}
                | _box_fields(output, image_box_to_grid_box(output.box, grid_map))
            )
    return rows


def merged_with_pp(
    image: DetectionImage,
    tile_boxes: Sequence[LineBox],
    content: ContentRows,
    config: PPConfig,
) -> tuple[PPOutcome, ...]:
    """R2 + V1/V2 merge (``MergeConfig()``) and the sealed PP on the result."""
    merged = merge_line_boxes(tile_boxes, image, DEFAULT_MERGE_CONFIG)
    return apply_pp(merged, content, config)


def _is_true(value: object) -> bool:
    return value is True or value == "True"


def _grid_of(row: Mapping) -> GridBox | None:
    if not _is_true(row["in_grid"]):
        return None
    return GridBox(int(row["i0"]), int(row["i1"]), int(row["k0"]), int(row["k1"]))


def pp_boxes_from_rows(
    rows: Iterable[Mapping], key_column: str
) -> dict[tuple[str, int], list[p4.PPBox]]:
    """``(key, fold) -> PPBox`` from box rows (csv text or part values).

    Boxes wholly in the padding (no L0 box before PP) cannot meet a truth
    and are left out; they stay in the box files.
    """
    pre: dict[tuple[str, int, int], Mapping] = {}
    post: dict[tuple[str, int, int], Mapping] = {}
    for row in rows:
        key = (str(row[key_column]), int(row["fold"]), int(row["box_index"]))
        (pre if row["pp"] == PP_PRE else post)[key] = row
    boxes: dict[tuple[str, int], list[p4.PPBox]] = {}
    for key, row in sorted(pre.items()):
        before = _grid_of(row)
        if before is None:
            continue
        after_row = post.get(key)
        boxes.setdefault(key[:2], []).append(
            p4.PPBox(
                box_id=str(key[2]),
                class_id=int(row["class_id"]),
                score=float(row["score"]),
                before=before,
                action=p4.PPAction(row["pp_action"]),
                after=None if after_row is None else _grid_of(after_row),
            )
        )
    return boxes


# ======================================================== uninjected (IDS)


def evaluate_uninjected_ids_line(
    line: runner.LineData,
    detectors: Mapping[int, object],
    settings: runner.RunSettings,
    config: PPConfig,
) -> dict:
    """Every merged r0 box of one IDS line per fold, with its PP outcome."""
    timer = runner.PhaseTimer()
    with timer.phase(PHASE_UNINJECTED_L1):
        image = make_detection_image(line.radargram, inputs.recipe_config(inputs.R0))
        content = content_rows_for(image, line.zero)
    job = runner.ImageJob(
        inputs.R0, runner.ORIGINAL, image, tuple(range(len(image.tiles)))
    )
    with timer.phase(PHASE_UNINJECTED_INFER):
        detections = runner.run_jobs([job], detectors, settings.batch_size)
    rows = []
    for (_, fold), tile_boxes in sorted(detections.items()):
        base = {
            "fold": fold,
            "line_id": line.ref.line_id,
            "dataset": line.ref.dataset,
            "realization_id": "",
        }
        outcomes = merged_with_pp(image, tile_boxes, content, config)
        rows += pp_box_rows(base, image, outcomes)
    return {
        "boxes": rows,
        "line": [runner.line_row(line, image)],
        "n_tiles": len(image.tiles),
        "timings": dict(timer.seconds),
    }


def _write_fold_files(
    stage_dir: Path,
    name: str,
    columns: Sequence[str],
    rows: Sequence[Mapping],
    folds: Sequence[int],
) -> None:
    for fold in folds:
        selected = [row for row in rows if int(row["fold"]) == fold]
        runner._write_rows(stage_dir / f"fold{fold}" / f"{name}.csv", columns, selected)


def run_uninjected_ids(
    settings: runner.RunSettings, started: float, context: runner.RunContext
) -> tuple[str, runner.StageCounts]:
    """IDS 60 lines x r0 x folds: boxes for zone (c') and the ghost pairs."""
    _, config = sealed_config(context)
    strata = ids_freq_strata()
    p4.assert_ids_freq_strata(strata)
    hashes = inputs.assert_list_hashes()
    hashes.update(runner.assert_weights(settings.run_root, settings.folds))
    lines = uninjected_ids_lines(settings, strata)
    runner.prepare_stage_dir(settings)
    runner.check_resume_identity(settings, runner.settings_sha256(settings, hashes))
    identity = runner.new_run_identity(settings, hashes, context)
    identity["tile_order"] = runner.TILE_ORDER
    detectors = runner.load_detectors(settings)
    parts = runner.run_lines(
        settings,
        lines,
        lambda line: evaluate_uninjected_ids_line(line, detectors, settings, config),
        started,
    )
    status, judgment = runner.close_run(
        settings, context, identity, m33_inputs.STATUS_OK
    )
    timings = summed_timings(parts)
    report_timings(timings)
    rows = [row for part in parts for row in part["boxes"]]
    _write_fold_files(settings.stage_dir, "boxes", BOX_COLUMNS, rows, settings.folds)
    runner._write_rows(
        settings.stage_dir / "lines.csv",
        runner.LINE_COLUMNS,
        (row for part in parts for row in part["line"]),
    )
    summary = {
        "schema": SCHEMA,
        "stage": runner.BAND_UNINJECTED_DIR,
        "status": status,
        "full_run": settings.is_full,
        "judgment": judgment,
        "run_identity": {**identity, "finished_at": common.local_now()},
        "n_lines": len(parts),
        "folds": list(settings.folds),
        "ids_freq_strata": dict(Counter(strata.values())),
        "timings": timings,
        "resources": runner.resources(settings, started),
    }
    runner._write_json(settings.stage_dir / "summary.json", summary)
    counts = runner.StageCounts(len(parts), sum(part["n_tiles"] for part in parts))
    return status, counts


# ================================================================== plan


def require_upstream(summary_path: Path, settings: runner.RunSettings) -> dict:
    """A finished upstream stage; a full run needs a judgeable one.

    Raises:
        inputs.InputError: For a missing summary, a failed run, or a run that
            is not judgeable under a full run.
    """
    if not summary_path.is_file():
        raise inputs.InputError(f"upstream summary missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    status = summary.get("status")
    if status not in (m33_inputs.STATUS_OK, m33_inputs.STATUS_NOT_JUDGEABLE):
        raise inputs.InputError(f"upstream {summary_path} ended {status!r}")
    judgeable = (summary.get("judgment") or {}).get("judgeable") is True
    if settings.is_full and not judgeable:
        raise inputs.InputError(f"upstream {summary_path} is not judgeable")
    return summary


def _box_rows_of(stage_dir: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(stage_dir.glob("fold*/boxes.csv")):
        rows += inputs.read_csv(path)
    return rows


def uninjected_box_rows(settings: runner.RunSettings) -> list[dict[str, str]]:
    """Box rows of the IDS un-injected run and, with 4TU, of R2."""
    rows: list[dict[str, str]] = []
    if inputs.DATASET_IDS in settings.backgrounds or settings.smoke:
        stage_dir = settings.out_root / runner.BAND_UNINJECTED_DIR
        require_upstream(stage_dir / "summary.json", settings)
        rows += _box_rows_of(stage_dir)
    if inputs.DATASET_4TU in settings.backgrounds and not settings.smoke:
        require_upstream(settings.r2_root / "summary.json", settings)
        rows += [
            row
            for row in _box_rows_of(settings.r2_root)
            if row.get("orientation", runner.ORIGINAL) == runner.ORIGINAL
        ]
    return rows


def zone_boxes(rows: Iterable[Mapping]) -> dict[str, list[GridBox]]:
    """Zone (c'): every PP-off box on the grid, any class, folds pooled."""
    boxes: dict[str, list[GridBox]] = {}
    for row in rows:
        if row.get("pp", PP_PRE) != PP_PRE:
            continue
        grid = _grid_of(row)
        if grid is not None:
            boxes.setdefault(row["line_id"], []).append(grid)
    return boxes


@dataclass(frozen=True)
class BandPlanResult:
    """Plan of one line: table rows, position rows and the line report."""

    specs: list[inject.InjectionSpec]
    positions: list[dict]
    report: dict


def _position_row(
    line: inputs.LineRef,
    context: inject.LineContext,
    band: inject.BandPatch,
    position: inject.PlannedBand,
    index: int,
    groups: Mapping[str, responses_io.ModelGroup],
    freq: str,
    config: PPConfig,
) -> dict:
    n_samples, n_traces = context.background.n_samples, context.background.n_traces
    truth = inject.band_truth_box(band, position.center_trace, n_samples)
    truth_px = grid_box_to_image_box(truth, context.grid_map)
    content = content_of(context)
    w_eff = inject.effective_median_window(context)
    length_px = position.length_traces / context.grid_map.traces_per_px
    box_clipped = band.row1 > n_samples - 1
    wing = inject.band_wing_past_record_end(band, context)
    main_cell = not (box_clipped or wing)
    model_group = groups[position.response_id].value
    rule = p4.exposed_rule(truth_px, content, config)
    return {
        "position_id": inject.band_position_id(line.line_id, position.group, index),
        "line_id": line.line_id,
        "dataset": line.dataset,
        "ids_freq_stratum": freq,
        "group": position.group,
        "response_id": position.response_id,
        "model_group": model_group,
        "band_center_trace": runner._exact(position.center_trace),
        "band_length_traces": position.length_traces,
        "taper_traces": position.taper_traces,
        "n_traces": n_traces,
        "w_eff": w_eff,
        "l_over_w_eff": runner._exact(position.length_traces / w_eff),
        "lw_stratum": p4.lw_stratum(position.length_traces, n_traces),
        "length_px": runner._exact(length_px),
        "px_stratum": px_stratum(length_px),
        "l_over_n": runner._exact(position.length_traces / n_traces),
        "apex_sample": runner._exact(band.apex_sample),
        "i0": truth.i0,
        "i1": truth.i1,
        "k0": truth.k0,
        "k1": truth.k1,
        "truth_x1_px": runner._exact(truth_px.x1),
        "truth_y1_px": runner._exact(truth_px.y1),
        "truth_x2_px": runner._exact(truth_px.x2),
        "truth_y2_px": runner._exact(truth_px.y2),
        "y_bot_px": runner._exact(content.y_bot_px),
        "period_px": runner._exact(content.period_px),
        "zero_px": "" if content.zero_px is None else runner._exact(content.zero_px),
        "box_clipped": box_clipped,
        "wing_past_record_end": wing,
        "main_cell": main_cell,
        "bottom_stratum": inject.is_bottom_stratum(context, band.patch),
        "in_horizontal_box_zone": inject.band_in_horizontal_box_zone(
            context, (truth.i0, truth.i1)
        ),
        "p4_scope": p4.is_p4_scope_position(line.dataset, freq, model_group, main_cell),
        "pp_exposed_rule": "" if rule is None else rule.value,
        "in_pp_c_window": p4.in_band_window(truth_px, content, config),
    }


def plan_band_line(
    line: runner.LineData,
    exclusions: inject.LineExclusions,
    library: responses_io.ResponseLibrary,
    freq: str,
    config: PPConfig,
    first_models: Sequence[str] = (),
) -> BandPlanResult:
    """Seeded band positions, table rows and position rows of one line.

    ``first_models`` are visited before the other models (900 MHz line).
    """
    context = inject.prepare_line_context(line.radargram, exclusions)
    responses = library.for_line(line.ref.dataset, runner.line_time_window_ns(line))
    patches, failures = runner.usable_patches(context, responses)
    plan = inject.plan_band_positions(
        context,
        patches,
        keep_fresnel_margin=line.ref.dataset == inputs.DATASET_4TU,
        preferred_models=first_models,
    )
    by_id = {patch.response.response_id: patch for patch in patches}
    rows = []
    for group in sorted({position.group for position in plan.positions}):
        members = [p for p in plan.positions if p.group == group]
        for index, position in enumerate(members):
            band = inject.build_band_patch(
                by_id[position.response_id], context, position.length_traces
            )
            assert isinstance(band, inject.BandPatch)  # it was a candidate
            rows.append(
                _position_row(
                    line.ref,
                    context,
                    band,
                    position,
                    index,
                    library.groups,
                    freq,
                    config,
                )
            )
    report = {
        "line_id": line.ref.line_id,
        "dataset": line.ref.dataset,
        "ids_freq_stratum": freq,
        "n_responses": len(responses),
        "patch_failures": failures,
        "n_candidates": plan.n_candidates,
        "n_dropped": plan.n_dropped,
        "lengths_skipped": list(plan.lengths_skipped),
        "n_positions": len(plan.positions),
        "n_realizations": plan.n_realizations,
        "preferred_models": list(first_models),
    }
    specs = inject.build_band_specs(
        line.ref.line_id, plan.positions, inject.band_conditions()
    )
    return BandPlanResult(specs, rows, report)


@dataclass(frozen=True)
class PlanJob:
    """One line of the plan stage."""

    ref: inputs.LineRef
    exclusions: inject.LineExclusions
    freq: str
    first_models: tuple[str, ...] = ()


_WORKER_STATE: list[tuple[responses_io.ResponseLibrary, PPConfig]] = []


def _init_band_worker(set_id: str) -> None:
    """Load the responses once per worker process."""
    _WORKER_STATE.append((responses_io.load_library(), PP_SETS[PPSetId(set_id)]))


def _plan_job(
    job: PlanJob, library: responses_io.ResponseLibrary, config: PPConfig
) -> BandPlanResult:
    return plan_band_line(
        runner.load_line(job.ref),
        job.exclusions,
        library,
        job.freq,
        config,
        job.first_models,
    )


def _band_worker(job: PlanJob) -> BandPlanResult:
    library, config = _WORKER_STATE[0]
    return _plan_job(job, library, config)


def _plan_lines(
    settings: runner.RunSettings, jobs: Sequence[PlanJob], set_id: PPSetId
) -> list[BandPlanResult]:
    """Plan every line; in process with one worker, else a process pool."""
    results = []
    if settings.workers == 1:
        library, config = responses_io.load_library(), PP_SETS[set_id]
        for number, job in enumerate(jobs, start=1):
            results.append(_plan_job(job, library, config))
            print(f"plan {number}/{len(jobs)} {job.ref.line_id}", flush=True)
        return results
    with ProcessPoolExecutor(
        max_workers=settings.workers,
        initializer=_init_band_worker,
        initargs=(set_id.value,),
    ) as executor:
        # map keeps the line order, so the table does not depend on workers.
        for number, result in enumerate(executor.map(_band_worker, jobs), start=1):
            results.append(result)
            print(f"plan {number}/{len(jobs)} {result.report['line_id']}", flush=True)
    return results


def band_positions(rows: Iterable[Mapping]) -> list[p4.BandPosition]:
    """P4-scope positions as ``BandPosition`` (csv text or plan values)."""
    positions = []
    for row in rows:
        if not _is_true(row["p4_scope"]):
            continue
        zero = row["zero_px"]
        positions.append(
            p4.BandPosition(
                position_id=row["position_id"],
                line_id=row["line_id"],
                lw_stratum=row["lw_stratum"],
                truth_px=PixelBox(
                    float(row["truth_x1_px"]),
                    float(row["truth_y1_px"]),
                    float(row["truth_x2_px"]),
                    float(row["truth_y2_px"]),
                ),
                content=ContentRows(
                    float(row["y_bot_px"]),
                    float(row["period_px"]),
                    None if zero in ("", None) else float(zero),
                ),
            )
        )
    return positions


def plan_counts(positions: Sequence[Mapping]) -> dict:
    """Detector-free position counts of the plan report ([RUN] §3-1 4)."""
    main = [row for row in positions if _is_true(row["main_cell"])]
    return {
        "positions": len(positions),
        "by_dataset_lw_stratum": runner.count_by(positions, ("dataset", "lw_stratum")),
        "main_cell_by_dataset_lw_stratum": runner.count_by(
            main, ("dataset", "lw_stratum")
        ),
        "by_dataset_group_lw_stratum": runner.count_by(
            positions, ("dataset", "model_group", "lw_stratum")
        ),
        "by_dataset_px_stratum": runner.count_by(positions, ("dataset", "px_stratum")),
        "by_ids_freq_stratum": runner.count_by(
            [row for row in positions if row["dataset"] == inputs.DATASET_IDS],
            ("ids_freq_stratum",),
        ),
        "by_length": runner.count_by(positions, ("dataset", "band_length_traces")),
        "non_main_cell": len(positions) - len(main),
        "in_horizontal_box_zone": sum(
            _is_true(row["in_horizontal_box_zone"]) for row in positions
        ),
        "target_per_stratum": p4.STRATUM_MIN_POSITIONS,
    }


def run_plan(
    settings: runner.RunSettings, started: float, context: runner.RunContext
) -> tuple[str, runner.StageCounts]:
    """Zone (c'), seeded band positions, table, positions, C1b draft."""
    timer = runner.PhaseTimer()
    set_id, _ = sealed_config(context)
    antennas = ids_antennas()
    strata = ids_freq_strata()
    p4.assert_ids_freq_strata(strata)
    hashes = inputs.assert_list_hashes()
    boxes = {} if settings.provisional else zone_boxes(uninjected_box_rows(settings))
    candidates = inputs.candidate_apexes()
    lines = band_background_lines(settings, strata)
    runner.prepare_stage_dir(settings)
    jobs = [
        PlanJob(
            ref,
            runner.line_exclusions(ref.line_id, boxes, candidates),
            strata.get(ref.line_id, ""),
            preferred_models(antennas.get(ref.line_id)),
        )
        for ref in lines
    ]
    with timer.phase(PHASE_PLAN):
        results = _plan_lines(settings, jobs, set_id)
    report_timings(timer.seconds)
    identity = runner.new_run_identity(settings, hashes, context)
    identity["seed"] = inject.BAND_PLAN_SEED
    identity["timings"] = dict(timer.seconds)
    status, judgment = runner.close_run(
        settings, context, identity, m33_inputs.STATUS_OK
    )
    return _finish_plan(settings, started, results, boxes, identity, status, judgment)


def _finish_plan(
    settings: runner.RunSettings,
    started: float,
    results: Sequence[BandPlanResult],
    boxes: Mapping[str, Sequence[GridBox]],
    identity: dict,
    status: str,
    judgment: dict,
) -> tuple[str, runner.StageCounts]:
    stage_dir = settings.stage_dir
    specs = [spec for result in results for spec in result.specs]
    positions = [row for result in results for row in result.positions]
    table_sha = inject.write_injection_table(stage_dir / "injection_table.csv", specs)
    runner._write_rows(stage_dir / "positions.csv", POSITION_COLUMNS, positions)
    exclusions_sha = inputs.write_exclusions(stage_dir / "exclusions.csv", boxes)
    set_id = PPSetId(identity["sealed_pp_assertion"]["pp_set_id"])
    record = {
        "injection_table_sha256": inputs.sha256_file_lf(
            stage_dir / "injection_table.csv"
        ),
        "positions_sha256": inputs.sha256_file_lf(stage_dir / "positions.csv"),
        "exclusions_sha256": inputs.sha256_file_lf(stage_dir / "exclusions.csv"),
        "raw_sha256": {"injection_table": table_sha, "exclusions": exclusions_sha},
        "seed": inject.BAND_PLAN_SEED,
        "pp_set_id": set_id.value,
        "pp_config_sha256": PP_SETS[set_id].sha256(),
        "provisional": settings.provisional,
    }
    if not settings.provisional:
        runner._write_json(stage_dir / "c1b_record_draft.json", record)
    p4_plan = p4.plan_record(band_positions(positions), PP_SETS[set_id])
    runner._write_json(
        stage_dir / "plan_report.json",
        {
            "schema": SCHEMA,
            "stage": "plan",
            "status": status,
            "provisional": settings.provisional,
            "judgment": judgment,
            "run_identity": {**identity, "finished_at": common.local_now()},
            "outputs_sha256": record,
            "positions": plan_counts(positions),
            "p4_plan": p4_plan,
            "timings": identity["timings"],
            "lines": [result.report for result in results],
            "seconds": round(time.time() - started, 1),
        },
    )
    return status, runner.StageCounts(len(results), 0, len(specs))


# ============================================================== injected


@dataclass(frozen=True)
class C1bInputs:
    """Table, positions and exclusions fixed at C1b (LF sha asserted)."""

    specs: list[inject.InjectionSpec]
    positions: list[dict[str, str]]
    exclusions: dict[str, tuple[GridBox, ...]]
    sha256: dict[str, str]


def read_c1b_inputs(settings: runner.RunSettings) -> C1bInputs:
    """Read the C1b record and assert the plan files against it.

    Raises:
        inputs.InputError: For a missing record or file, or another LF sha.
    """
    if not settings.c1b_record.is_file():
        raise inputs.InputError(f"C1b record missing: {settings.c1b_record}")
    record = json.loads(settings.c1b_record.read_text(encoding="utf-8"))
    plan_dir = settings.out_root / "plan"
    checked = {}
    for name, file_name in (
        ("injection_table", "injection_table.csv"),
        ("positions", "positions.csv"),
        ("exclusions", "exclusions.csv"),
    ):
        key = f"{name}_sha256"
        if key not in record:
            raise inputs.InputError(f"C1b record lacks {key}")
        checked[name] = inputs.checked_text_sha256(
            plan_dir / file_name, record[key], name
        )
    specs, _ = inject.read_injection_table(plan_dir / "injection_table.csv")
    checked["c1b_record"] = inputs.sha256_file_lf(settings.c1b_record)
    return C1bInputs(
        specs=specs,
        positions=inputs.read_csv(plan_dir / "positions.csv"),
        exclusions=inputs.read_exclusions(plan_dir / "exclusions.csv"),
        sha256=checked,
    )


def position_id_of(injection_id: str) -> str:
    """``<line>:g<group>:<index>`` from ``<line>:g<group>:snr:k:p:<index>``."""
    realization_id, index = injection_id.rsplit(":", 1)
    return f"{realization_id.rsplit(':', 3)[0]}:{index}"


def inference_tiles(image: DetectionImage) -> tuple[int, ...]:
    """Every tile of an injected realization.

    The r0 chain has whole-line steps besides the step (4) median: the
    period measurement (trace-averaged spectrum -> ``P_line`` -> the L1
    scale), the AGC floor (1e-12 of the line's largest mean power) and the
    grey clip level (99th percentile of ``|a|`` over the line). A band can
    change all three, so no tile is known to equal the un-injected image.
    """
    return tuple(range(len(image.tiles)))


def tile_trace_range(image: DetectionImage, tiles: Sequence[int]) -> tuple[int, int]:
    """L0 traces under the given tiles (``ImageGridMap`` cell edges)."""
    grid = image.grid_map
    chosen = [image.tiles[index] for index in tiles]
    start_px = min(tile.x0_px for tile in chosen)
    stop_px = max(tile.x0_px + tile.content_width_px for tile in chosen)
    first = max(0, math.floor(start_px * grid.traces_per_px) + grid.trace_offset)
    last = min(
        grid.n_traces - 1,
        math.ceil(stop_px * grid.traces_per_px) - 1 + grid.trace_offset,
    )
    return first, last


def survival_ratios(
    context: inject.LineContext,
    result: inject.BandInjectionResult,
    background_section: np.ndarray | None,
) -> list[float | str]:
    """Survival ratio of every band of a realization (one step (4) pass)."""
    if background_section is None:  # amplitude_processed line: no step (4)
        return [""] * len(result.truths)
    return list(
        inject.band_survival_ratios(
            context, result.data, result.truths, background_section
        )
    )


def zero_changed(context: inject.LineContext, injected: Radargram) -> bool:
    """The zero re-measured on the injected record differs from the line's."""
    filtered = inject.linear_steps(injected.data, injected)
    measured = find_time_zero(injected, filtered).k_zero
    return measured != context.zero.k_zero


def truth_row(
    truth: inject.BandTruth,
    result: inject.BandInjectionResult,
    context: inject.LineContext,
    line: inputs.LineRef,
    groups: Mapping[str, responses_io.ModelGroup],
    freq: str,
    survival_ratio: float | str = "",
) -> dict:
    """One accepted band: truth table columns and its survival ratio."""
    box = truth.box
    return {
        "injection_id": truth.injection_id,
        "realization_id": result.realization_id,
        "position_id": position_id_of(truth.injection_id),
        "line_id": line.line_id,
        "dataset": line.dataset,
        "ids_freq_stratum": freq,
        "response_id": truth.response_id,
        "model_group": groups[truth.response_id].value,
        "mode": inject.InjectionMode.BAND.value,
        "band_length_traces": truth.length_traces,
        "band_center_trace": runner._exact(truth.center_trace),
        "length_px": runner._exact(truth.length_px),
        "w_eff": truth.w_eff,
        "l_over_w_eff": runner._exact(truth.l_over_w_eff),
        "lw_stratum": p4.lw_stratum(truth.length_traces, context.background.n_traces),
        "px_stratum": px_stratum(truth.length_px),
        "l_over_n": runner._exact(truth.l_over_n),
        "taper_traces": truth.taper_traces,
        "i0": box.i0,
        "i1": box.i1,
        "k0": box.k0,
        "k1": box.k1,
        "apex_sample": runner._exact(truth.apex_sample),
        "gain": runner._exact(truth.gain),
        "noise_rms": runner._exact(truth.noise_rms),
        "signal_amplitude": runner._exact(truth.signal_amplitude),
        "n_clipped_cells": truth.n_clipped_cells,
        "max_clipped_injection_ratio": runner._exact(truth.max_clipped_injection_ratio),
        "polarity": truth.polarity,
        "snr_db": runner._exact(truth.snr_db),
        "k": runner._exact(truth.k),
        "box_clipped": truth.box_clipped,
        "wing_past_record_end": truth.wing_past_record_end,
        "main_cell": truth.main_cell,
        "bottom_stratum": truth.bottom_stratum,
        "in_horizontal_box_zone": truth.in_horizontal_box_zone,
        "period_ringing": truth.period_ringing,
        "survival_ratio": survival_ratio,
    }


@dataclass(frozen=True)
class BandLineInputs:
    """What one line of the injected stage needs besides the line itself."""

    realizations: Mapping[str, Sequence[inject.InjectionSpec]]
    exclusions: inject.LineExclusions
    library: responses_io.ResponseLibrary
    detectors: Mapping[int, object]
    config: PPConfig
    freq: str


def evaluate_band_line(
    line: runner.LineData, line_inputs: BandLineInputs, settings: runner.RunSettings
) -> dict:
    """Inject, infer, merge and apply PP for every realization of one line.

    Each realization's r0 image is built once and its tiles are inferred by
    every fold in the same batches (``run_jobs``), so no image is rebuilt per
    fold.
    """
    timer = runner.PhaseTimer()
    with timer.phase(PHASE_INJECT):
        context = inject.prepare_line_context(line.radargram, line_inputs.exclusions)
        responses = line_inputs.library.for_line(
            line.ref.dataset, runner.line_time_window_ns(line)
        )
        results = [
            inject.inject_band_realization(context, specs, responses)
            for specs in line_inputs.realizations.values()
        ]
    jobs, realization_rows, survival = [], [], {}
    background_section = None
    if not line.radargram.amplitude_processed:
        with timer.phase(PHASE_SURVIVAL):
            background_section = inject.background_step4_section(context)
    for index, result in enumerate(results):
        if not result.truths:
            continue
        injected = replace(line.radargram, data=result.data.astype(np.float32))
        with timer.phase(PHASE_L1):
            image = make_detection_image(injected, inputs.recipe_config(inputs.R0))
        tiles = inference_tiles(image)
        jobs.append(runner.ImageJob(inputs.R0, runner.ORIGINAL, image, tiles, index))
        with timer.phase(PHASE_SURVIVAL):
            survival[index] = survival_ratios(context, result, background_section)
            changed_zero = zero_changed(context, injected)
        first, last = tile_trace_range(image, tiles)
        realization_rows.append(
            {
                "realization_id": result.realization_id,
                "infer_i0": first,
                "infer_i1": last,
                "period_changed": image.period_samples != context.period_samples,
                "zero_changed": changed_zero,
            }
        )
    with timer.phase(PHASE_INFER):
        detections = runner.run_jobs(jobs, line_inputs.detectors, settings.batch_size)
    boxes = []
    with timer.phase(PHASE_PP):
        for (j, fold), tile_boxes in sorted(detections.items()):
            boxes += _realization_box_rows(
                line, jobs[j], fold, results, tile_boxes, context, line_inputs.config
            )
    return {
        "boxes": boxes,
        "realizations": realization_rows,
        "truths": [
            truth_row(
                truth,
                result,
                context,
                line.ref,
                line_inputs.library.groups,
                line_inputs.freq,
                survival.get(index, [""] * len(result.truths))[number],
            )
            for index, result in enumerate(results)
            for number, truth in enumerate(result.truths)
        ],
        "rejections": [
            {
                "injection_id": injection_id,
                "realization_id": result.realization_id,
                "line_id": line.ref.line_id,
                "reason": reason,
            }
            for result in results
            for injection_id, reason in result.rejected
        ],
        "n_tiles": sum(len(job.tile_indices) for job in jobs),
        "timings": dict(timer.seconds),
    }


def _realization_box_rows(
    line: runner.LineData,
    job: runner.ImageJob,
    fold: int,
    results: Sequence[inject.BandInjectionResult],
    tile_boxes: Sequence[LineBox],
    context: inject.LineContext,
    config: PPConfig,
) -> list[dict]:
    """Merged boxes of one realization and fold with their PP outcome."""
    base = {
        "fold": fold,
        "line_id": line.ref.line_id,
        "dataset": line.ref.dataset,
        "realization_id": results[job.tag].realization_id,
    }
    # PP sees the rows of the image it runs on (the line zero, [PP] §1).
    content = content_rows_for(job.image, context.zero)
    outcomes = merged_with_pp(job.image, tile_boxes, content, config)
    return pp_box_rows(base, job.image, outcomes)


# ------------------------------------------------------------- analysis


def truth_unit(row: Mapping) -> p4.BandTruthUnit:
    """A truth row as the unit the band match works on."""
    return p4.BandTruthUnit(
        injection_id=row["injection_id"],
        realization_id=row["realization_id"],
        position_id=row["position_id"],
        line_id=row["line_id"],
        lw_stratum=row["lw_stratum"],
        snr_db=float(row["snr_db"]),
        polarity=int(row["polarity"]),
        box=GridBox(int(row["i0"]), int(row["i1"]), int(row["k0"]), int(row["k1"])),
        taper_traces=int(row["taper_traces"]),
    )


@dataclass(frozen=True)
class MatchedUnit:
    """One truth x fold: its truth row, the P4 unit result and any-class cover."""

    row: Mapping
    result: p4.UnitResult
    covered_any_class: tuple[int, int]


def _any_class_cover(truth: GridBox, boxes: Sequence[p4.PPBox]) -> tuple[int, int]:
    """Covered traces by boxes of any class over 0.25, before and after PP."""
    confident = [box for box in boxes if box.score > p4.CONF_OPERATING]
    before = p4.covered_traces(
        truth, p4.recall_boxes(truth, [box.before for box in confident])
    )
    after_boxes = [box.after for box in confident if box.after is not None]
    after = p4.covered_traces(truth, p4.recall_boxes(truth, after_boxes))
    return before, after


def match_units(
    truth_rows: Sequence[Mapping],
    injected: Mapping[tuple[str, int], Sequence[p4.PPBox]],
    uninjected: Mapping[tuple[str, int], Sequence[p4.PPBox]],
    folds: Sequence[int],
    ranges: Mapping[str, tuple[int, int]] | None = None,
) -> list[MatchedUnit]:
    """Every truth x fold through the band match of ``m33_r3_p4``.

    ``ranges`` (``realization_id -> inferred traces``) cut the un-injected
    boxes before the ghost pairing (runner spec §3-1 4).
    """
    units = []
    for row in truth_rows:
        unit = truth_unit(row)
        pair_range = (ranges or {}).get(unit.realization_id)
        for fold in folds:
            boxes = injected.get((unit.realization_id, fold), ())
            result = p4.evaluate_unit(
                unit, fold, boxes, uninjected.get((unit.line_id, fold), ()), pair_range
            )
            units.append(MatchedUnit(row, result, _any_class_cover(unit.box, boxes)))
    return units


def inference_ranges(parts: Iterable[Mapping]) -> dict[str, tuple[int, int]]:
    """``realization_id -> (infer_i0, infer_i1)`` from the line parts."""
    return {
        row["realization_id"]: (int(row["infer_i0"]), int(row["infer_i1"]))
        for part in parts
        for row in part.get("realizations", ())
    }


def padding_only_acts(box_rows: Iterable[Mapping]) -> dict:
    """``pp_acted_padding_only``: acted ``pipe_band`` boxes with no L0 box.

    Such boxes lie wholly in the padding below the content and take no part
    in the match; their PP actions are counted apart (report only).
    """
    by_rule: Counter[str] = Counter()
    for row in box_rows:
        if row["pp"] != PP_PRE or _is_true(row["in_grid"]):
            continue
        if int(row["class_id"]) != DetectionClass.PIPE_BAND:
            continue
        if not float(row["score"]) > p4.CONF_OPERATING:
            continue
        if row["pp_action"] != PPAction.NONE.value:
            by_rule[row["pp_action"]] += 1
    return {"total": sum(by_rule.values()), "by_rule": dict(sorted(by_rule.items()))}


def realization_counts(parts: Iterable[Mapping]) -> dict:
    """``n_period_changed`` and ``n_zero_changed`` over the realizations."""
    rows = [row for part in parts for row in part.get("realizations", ())]
    return {
        "n_realizations": len(rows),
        "n_period_changed": sum(_is_true(row["period_changed"]) for row in rows),
        "n_zero_changed": sum(_is_true(row["zero_changed"]) for row in rows),
    }


def match_row(unit: MatchedUnit) -> dict:
    """``inj_matches.csv``: one truth x fold, PP before and after."""
    row, result = unit.row, unit.result
    length = result.truth.length_traces
    return {
        "fold": result.fold,
        "injection_id": row["injection_id"],
        "realization_id": row["realization_id"],
        "line_id": row["line_id"],
        "dataset": row["dataset"],
        "model_group": row["model_group"],
        "lw_stratum": row["lw_stratum"],
        "px_stratum": row["px_stratum"],
        "snr_db": row["snr_db"],
        "polarity": row["polarity"],
        "main_cell": row["main_cell"],
        "in_horizontal_box_zone": row["in_horizontal_box_zone"],
        "length_traces": length,
        "covered_pre": result.covered_before,
        "covered_post": result.covered_after,
        "recall_pre": result.covered_before / length,
        "recall_post": result.covered_after / length,
        "n_recall_boxes": len(result.recall_set),
        "covered_any_class_pre": unit.covered_any_class[0],
        "covered_any_class_post": unit.covered_any_class[1],
        "ghost_pairs_pre": len(result.ghosts_before),
        "ghost_pairs_post": len(result.ghosts_after),
    }


def ghost_rows(units: Sequence[MatchedUnit]) -> list[dict]:
    """``ghosts.csv``: distinct ghost boxes and length per realization x fold."""
    results = [unit.result for unit in units]
    before = p4.realization_ghosts(results, after=False)
    after = p4.realization_ghosts(results, after=True)
    lines = {u.result.truth.realization_id: u.row for u in units}
    rows = []
    for realization_id, fold in sorted(before):
        row = lines[realization_id]
        rows.append(
            {
                "fold": fold,
                "realization_id": realization_id,
                "line_id": row["line_id"],
                "dataset": row["dataset"],
                "ghost_n_boxes_pre": before[(realization_id, fold)]["ghost_n_boxes"],
                "ghost_n_boxes_post": after[(realization_id, fold)]["ghost_n_boxes"],
                "ghost_length_traces_pre": before[(realization_id, fold)][
                    "ghost_length_traces"
                ],
                "ghost_length_traces_post": after[(realization_id, fold)][
                    "ghost_length_traces"
                ],
            }
        )
    return rows


def _fold_stats(values_by_fold: Mapping[int, float]) -> dict:
    values = [values_by_fold[fold] for fold in sorted(values_by_fold)]
    return {
        "mean": statistics.fmean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "by_fold": {str(fold): values_by_fold[fold] for fold in sorted(values_by_fold)},
    }


def _label(row: Mapping, keys: Sequence[str]) -> str:
    return "/".join(str(row[key]) for key in keys)


def recall_table(
    units: Sequence[MatchedUnit],
    keys: Sequence[str],
    covered: Callable[[MatchedUnit, bool], int] | None = None,
) -> dict:
    """Length-weighted recall per key and fold, PP before and after."""

    def default_cover(unit: MatchedUnit, after: bool) -> int:
        result = unit.result
        return result.covered_after if after else result.covered_before

    cover = covered or default_cover
    sums: dict[str, dict[int, list[int]]] = {}
    for unit in units:
        cell = sums.setdefault(_label(unit.row, keys), {})
        tally = cell.setdefault(unit.result.fold, [0, 0, 0, 0])
        tally[0] += cover(unit, False)
        tally[1] += cover(unit, True)
        tally[2] += unit.result.truth.length_traces
        tally[3] += 1
    return {
        label: {
            "pre": _fold_stats({f: t[0] / t[2] for f, t in by_fold.items()}),
            "post": _fold_stats({f: t[1] / t[2] for f, t in by_fold.items()}),
            "n_truth_folds": sum(t[3] for t in by_fold.values()),
        }
        for label, by_fold in sorted(sums.items())
    }


def ghost_table(units: Sequence[MatchedUnit], keys: Sequence[str]) -> dict:
    """Ghost boxes and length per key, summed per fold, PP before and after."""
    groups: dict[str, list[p4.UnitResult]] = {}
    for unit in units:
        groups.setdefault(_label(unit.row, keys), []).append(unit.result)
    table = {}
    for label, results in sorted(groups.items()):
        entry = {}
        for name, after in (("pre", False), ("post", True)):
            per_realization = p4.realization_ghosts(results, after)
            boxes: Counter[int] = Counter()
            length: Counter[int] = Counter()
            for (_, fold), item in per_realization.items():
                boxes[fold] += item["ghost_n_boxes"]
                length[fold] += item["ghost_length_traces"]
            entry[name] = {
                "ghost_n_boxes": _fold_stats(dict(boxes)),
                "ghost_length_traces": _fold_stats(dict(length)),
            }
        table[label] = entry
    return table


def survival_table(truth_rows: Sequence[Mapping]) -> dict:
    """Survival ratio per background x SNR x ``lw_stratum`` (fold-free)."""
    groups: dict[str, list[float]] = {}
    for row in truth_rows:
        value = row["survival_ratio"]
        if value in ("", None):
            continue
        label = _label(row, ("dataset", "snr_db", "lw_stratum"))
        groups.setdefault(label, []).append(float(value))
    return {
        label: {
            "n": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }
        for label, values in sorted(groups.items())
    }


CELL_KEYS = ("dataset", "model_group", "lw_stratum", "snr_db", "polarity")


def cell_tables(units: Sequence[MatchedUnit]) -> dict:
    """Recall and ghost tables of one set of truths (PP before and after)."""
    zone_subset = [u for u in units if _is_true(u.row["in_horizontal_box_zone"])]
    return {
        "recall": recall_table(units, CELL_KEYS),
        "ghosts": ghost_table(units, CELL_KEYS),
        "recall_px_stratum": recall_table(units, ("dataset", "px_stratum")),
        "recall_horizontal_box_zone_subset": recall_table(zone_subset, ("dataset",)),
        "recall_any_class": recall_table(
            units,
            ("dataset", "lw_stratum"),
            lambda unit, after: unit.covered_any_class[int(after)],
        ),
        "n_truth_folds": len(units),
    }


def band_summary_tables(units: Sequence[MatchedUnit], truth_rows: Sequence[Mapping]):
    """The §3-2 summary tables: main-cell truths, and ``bottom`` (report only).

    ``bottom`` holds the same tables for the truths outside the C3-6 main
    cells (runner spec §3-1 4 배치 구현 해석).
    """
    main = [unit for unit in units if _is_true(unit.row["main_cell"])]
    bottom = [unit for unit in units if not _is_true(unit.row["main_cell"])]
    return {
        **cell_tables(main),
        "bottom": cell_tables(bottom) | {"report_only": True},
        "survival_ratio": survival_table(truth_rows),
        "n_truths": len(truth_rows),
        "n_truths_non_main_cell": sum(
            not _is_true(row["main_cell"]) for row in truth_rows
        ),
        "folds": "mean and sample SD over the folds (folds share positions: not a SE)",
    }


# ---------------------------------------------------------------- P4 wiring


def p4_inputs(
    set_id: PPSetId,
    folds: Sequence[int],
    positions: Sequence[Mapping],
    truth_rows: Sequence[Mapping],
    injected: Mapping[tuple[str, int], Sequence[p4.PPBox]],
    uninjected: Mapping[tuple[str, int], Sequence[p4.PPBox]],
    ranges: Mapping[str, tuple[int, int]] | None = None,
) -> p4.P4Inputs:
    """P4-scope positions (plan) and truths (injected) with their boxes."""
    scope_truths = [
        truth_unit(row)
        for row in truth_rows
        if p4.is_p4_scope_position(
            row["dataset"],
            row["ids_freq_stratum"],
            row["model_group"],
            _is_true(row["main_cell"]),
        )
    ]
    realizations = {truth.realization_id for truth in scope_truths}
    scope_positions = band_positions(positions)
    lines = {position.line_id for position in scope_positions}
    return p4.P4Inputs(
        pp_set_id=set_id,
        folds=tuple(folds),
        positions=tuple(scope_positions),
        truths=tuple(scope_truths),
        injected_boxes={k: v for k, v in injected.items() if k[0] in realizations},
        uninjected_boxes={k: v for k, v in uninjected.items() if k[0] in lines},
        inference_ranges={k: v for k, v in (ranges or {}).items() if k in realizations},
    )


def scope_box_rows(
    box_rows: Iterable[Mapping], inputs_p4: p4.P4Inputs, uninjected_rows: Iterable
) -> list[Mapping]:
    """Box rows of the P4 realizations and of the un-injected P4 lines."""
    realizations = {truth.realization_id for truth in inputs_p4.truths}
    lines = {position.line_id for position in inputs_p4.positions}
    rows = [
        row
        for row in box_rows
        if row["realization_id"] in realizations and int(row["fold"]) in inputs_p4.folds
    ]
    rows += [
        row
        for row in uninjected_rows
        if row["line_id"] in lines and int(row["fold"]) in inputs_p4.folds
    ]
    return rows


def evaluate_p4_value(
    inputs_p4: p4.P4Inputs,
    stage_dir: Path,
    strata: Mapping[str, str],
    padding_rows: Iterable[Mapping] = (),
) -> tuple[dict, bool]:
    """``p4_adoption_input`` and whether a P4 defect was found (exit 1)."""
    split = p4.assert_ids_freq_strata(strata)
    try:
        document = p4.run_p4(inputs_p4, stage_dir)
        defect = False
    except p4.P4DefectError as error:
        document, defect = error.document, True
    document["n_lines"] = split[FREQ_MAIN]
    document["ids_line_split"] = split
    document["report_only"] = REPORT_ONLY
    document["pp_acted_padding_only"] = padding_only_acts(padding_rows)
    return document, defect


# ---------------------------------------------------------- injected stage


def load_uninjected(settings: runner.RunSettings) -> tuple[dict, list[dict[str, str]]]:
    """``(line_id, fold) -> PPBox`` of the un-injected runs, and their rows."""
    rows = uninjected_box_rows(settings)
    return pp_boxes_from_rows(rows, "line_id"), rows


def _assemble_injected(
    settings: runner.RunSettings,
    parts: Sequence[Mapping],
    units: Sequence[MatchedUnit],
) -> list[dict]:
    stage_dir, folds = settings.stage_dir, settings.folds
    truths = [row for part in parts for row in part["truths"]]
    boxes = [row for part in parts for row in part["boxes"]]
    rejections = [row for part in parts for row in part["rejections"]]
    runner._write_rows(stage_dir / "truths.csv", TRUTH_COLUMNS, truths)
    runner._write_rows(stage_dir / "rejections.csv", REJECTION_COLUMNS, rejections)
    _write_fold_files(stage_dir, "inj_boxes", BOX_COLUMNS, boxes, folds)
    matches = [match_row(unit) for unit in units]
    _write_fold_files(stage_dir, "inj_matches", MATCH_COLUMNS, matches, folds)
    _write_fold_files(stage_dir, "ghosts", GHOST_COLUMNS, ghost_rows(units), folds)
    return rejections


def run_injected(
    settings: runner.RunSettings, started: float, context: runner.RunContext
) -> tuple[str, runner.StageCounts]:
    """Inject the C1b table, infer, match and compute the P4 value."""
    set_id, config = sealed_config(context)
    strata = ids_freq_strata()
    p4.assert_ids_freq_strata(strata)
    hashes = inputs.assert_list_hashes()
    hashes.update(runner.assert_weights(settings.run_root, settings.folds))
    c1b = read_c1b_inputs(settings)
    hashes.update(c1b.sha256)
    library = responses_io.load_library()
    hashes["responses"] = responses_io.library_sha256(library)
    uninjected, uninjected_rows = load_uninjected(settings)
    by_line = runner.group_specs(c1b.specs)
    refs = {ref.line_id: ref for ref in band_background_lines(settings, strata)}
    lines = [refs[line_id] for line_id in by_line if line_id in refs]
    candidates = inputs.candidate_apexes()
    runner.prepare_stage_dir(settings)
    runner.check_resume_identity(settings, runner.settings_sha256(settings, hashes))
    identity = runner.new_run_identity(settings, hashes, context)
    identity["tile_order"] = runner.TILE_ORDER
    detectors = runner.load_detectors(settings)

    def evaluate(line: runner.LineData) -> dict:
        line_id = line.ref.line_id
        line_inputs = BandLineInputs(
            realizations=by_line[line_id],
            exclusions=runner.line_exclusions(line_id, c1b.exclusions, candidates),
            library=library,
            detectors=detectors,
            config=config,
            freq=strata.get(line_id, ""),
        )
        return evaluate_band_line(line, line_inputs, settings)

    parts = runner.run_lines(settings, lines, evaluate, started)
    status, judgment = runner.close_run(
        settings, context, identity, m33_inputs.STATUS_OK
    )
    timings = summed_timings(parts)
    timer = runner.PhaseTimer(dict(timings))
    truths = [row for part in parts for row in part["truths"]]
    box_rows = [row for part in parts for row in part["boxes"]]
    ranges = inference_ranges(parts)
    with timer.phase(PHASE_MATCH):
        injected = pp_boxes_from_rows(box_rows, "realization_id")
        units = match_units(truths, injected, uninjected, settings.folds, ranges)
        rejections = _assemble_injected(settings, parts, units)
    with timer.phase(PHASE_P4):
        scope = p4_inputs(
            set_id, settings.folds, c1b.positions, truths, injected, uninjected, ranges
        )
        p4_document, defect = evaluate_p4_value(
            scope,
            settings.stage_dir,
            strata,
            scope_box_rows(box_rows, scope, uninjected_rows),
        )
    report_timings(timer.seconds)
    if defect and status != m33_inputs.STATUS_INPUT_ERROR:
        status = runner.P4_DEFECT
    summary = {
        "schema": SCHEMA,
        "stage": "injected",
        "status": status,
        "full_run": settings.is_full,
        "judgment": judgment,
        "run_identity": {**identity, "finished_at": common.local_now()},
        "n_lines": len(parts),
        "n_accepted": len(truths),
        **realization_counts(parts),
        "pp_acted_padding_only": padding_only_acts(box_rows),
        "folds": list(settings.folds),
        "rejection_reasons": runner.count_by(rejections, ("reason",)),
        **band_summary_tables(units, truths),
        "p4_adoption_input": p4_document,
        "timings": dict(timer.seconds),
        "resources": runner.resources(settings, started),
    }
    runner._write_json(settings.stage_dir / "summary.json", summary)
    n_tiles = sum(part["n_tiles"] for part in parts)
    return status, runner.StageCounts(len(parts), n_tiles, len(truths))


# ------------------------------------------------------------- dispatch


STAGE_RUNNERS = {
    "uninjected": run_uninjected_ids,
    "plan": run_plan,
    "injected": run_injected,
}


def run_smoke(
    settings: runner.RunSettings, started: float, context: runner.RunContext
) -> tuple[str, runner.StageCounts]:
    """The three stages on three IDS 400 MHz lines and fold 0 (smoke folder).

    The injected stage reads the plan's own ``c1b_record_draft.json``. A
    finished smoke is ``not_judgeable`` (exit 1); an earlier failing stage
    stops it with that stage's status.
    """
    stage_type = type(settings.stage)
    counts = runner.StageCounts()
    status = m33_inputs.STATUS_OK
    for name in ("uninjected", "plan", "injected"):
        stage_settings = replace(
            settings,
            stage=stage_type(name),
            backgrounds=(inputs.DATASET_IDS,),
            c1b_record=settings.out_root / "plan" / "c1b_record_draft.json",
            overwrite=True,
        )
        # Each stage tracks its own inputs, so each gets its own start tree.
        stage_context = replace(context, start_tree=runner.measure_tree(stage_settings))
        status, counts = STAGE_RUNNERS[name](stage_settings, started, stage_context)
        print(f"smoke {name}: {status}", flush=True)
        if status not in (m33_inputs.STATUS_OK, m33_inputs.STATUS_NOT_JUDGEABLE):
            return status, counts
    estimate_path = settings.out_root / "smoke_estimate.json"
    runner._write_json(estimate_path, smoke_estimate(settings.out_root))
    print(f"smoke estimate: {estimate_path}", flush=True)
    return m33_inputs.STATUS_NOT_JUDGEABLE, counts


# --------------------------------------------------------- time estimate

FULL_FOLDS = 5
FULL_REALIZATIONS_PER_LINE = inject.MAX_REALIZATIONS_PER_LINE * len(
    inject.band_conditions()
)
"""Upper bound: two groups x eight conditions on every line."""


def _parts(stage_dir: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((stage_dir / "parts").glob("*.json"))
    ]


def _rate(seconds: float, units: float) -> float | None:
    return seconds / units if units else None


def _full_line_tiles() -> dict[str, int]:
    """Tiles of every R3 line from the stage-C line table (``n_tiles``)."""
    return {
        f"{row['dataset']}|{row['line']}": int(row["n_tiles"])
        for row in inputs.read_csv(inputs.CONTENT_HEIGHT_LINES)
    }


def _tiles_of(refs: Iterable[inputs.LineRef], tiles: Mapping[str, int]) -> int:
    known = [
        tiles[f"{r.dataset}|{r.line_id}"]
        for r in refs
        if f"{r.dataset}|{r.line_id}" in tiles
    ]
    missing = sum(f"{r.dataset}|{r.line_id}" not in tiles for r in refs)
    median = statistics.median(known) if known else 0
    return int(sum(known) + missing * median)


def _fourtu_lines() -> list[inputs.LineRef]:
    """4TU stage A 150 + K 27 for the estimate (empty when ``data/`` is absent)."""
    try:
        stage_a = [
            r for r in inputs.background_lines() if r.dataset == inputs.DATASET_4TU
        ]
        return stage_a + inputs.k_lines()
    except (inputs.InputError, OSError) as error:
        logger.warning("4TU lines unavailable for the estimate: %s", error)
        return []


def smoke_estimate(out_root: Path) -> dict:
    """Per-unit rates of the smoke and the full R3 time they imply.

    Rates: uninjected L1 per line tile, inference per tile x fold; plan per
    line tile; injected per realization tile (injection, step (4) and zero,
    L1) and per realization tile x fold (inference, merge + PP). The full
    run uses the stage-C table tiles of the IDS 60 lines (uninjected), the
    IDS 56 + 4TU 177 lines (plan, injected), five folds and an upper bound
    of 16 realizations per line. Times only; no metric value.
    """
    uninjected = _parts(out_root / runner.BAND_UNINJECTED_DIR)
    injected = _parts(out_root / "injected")
    plan_report = out_root / "plan" / "plan_report.json"
    plan = json.loads(plan_report.read_text(encoding="utf-8"))
    smoke_folds = 1
    u_tiles = sum(part["n_tiles"] for part in uninjected)
    u_time = summed_timings(uninjected)
    i_time = summed_timings(injected)
    realization_tiles = sum(part["n_tiles"] for part in injected)
    rates = {
        "uninjected_l1_per_tile": _rate(u_time.get(PHASE_UNINJECTED_L1, 0), u_tiles),
        "uninjected_infer_per_tile_fold": _rate(
            u_time.get(PHASE_UNINJECTED_INFER, 0), u_tiles * smoke_folds
        ),
        "plan_per_line_tile": _rate(plan["timings"].get(PHASE_PLAN, 0), u_tiles),
        "injected_cpu_per_realization_tile": _rate(
            sum(i_time.get(k, 0) for k in (PHASE_INJECT, PHASE_SURVIVAL, PHASE_L1)),
            realization_tiles,
        ),
        "injected_gpu_per_realization_tile_fold": _rate(
            sum(i_time.get(k, 0) for k in (PHASE_INFER, PHASE_PP)),
            realization_tiles * smoke_folds,
        ),
    }
    strata = ids_freq_strata()
    tiles = _full_line_tiles()
    ids_all = _ids_lines()
    ids_injected = [r for r in ids_all if strata.get(r.line_id) != FREQ_EXCLUDED]
    fourtu = _fourtu_lines()
    full = {
        "ids_uninjected_tiles": _tiles_of(ids_all, tiles),
        "ids_injected_line_tiles": _tiles_of(ids_injected, tiles),
        "fourtu_line_tiles": _tiles_of(fourtu, tiles),
    }
    estimate = _full_seconds(rates, full)
    return {
        "smoke_timings": {
            "uninjected": u_time,
            "plan": plan["timings"],
            "injected": i_time,
        },
        "smoke_units": {
            "uninjected_tiles": u_tiles,
            "injected_realization_tiles": realization_tiles,
            "folds": smoke_folds,
        },
        "rates_s": rates,
        "full_units": full
        | {
            "folds": FULL_FOLDS,
            "realizations_per_line_upper": FULL_REALIZATIONS_PER_LINE,
        },
        "full_seconds_upper": estimate,
        "note": "upper bound: every line holds 16 realizations; 4TU lines are "
        "shorter, so their plan and step (4) cost per tile is lower",
    }


def _full_seconds(rates: Mapping[str, float | None], full: Mapping[str, int]) -> dict:
    def rate(name: str) -> float:
        return rates.get(name) or 0.0

    line_tiles = full["ids_injected_line_tiles"] + full["fourtu_line_tiles"]
    realization_tiles = FULL_REALIZATIONS_PER_LINE * line_tiles
    stages = {
        "uninjected_ids": full["ids_uninjected_tiles"]
        * (
            rate("uninjected_l1_per_tile")
            + FULL_FOLDS * rate("uninjected_infer_per_tile_fold")
        ),
        "plan_cpu": line_tiles * rate("plan_per_line_tile"),
        "injected_cpu": realization_tiles * rate("injected_cpu_per_realization_tile"),
        "injected_gpu": realization_tiles
        * FULL_FOLDS
        * rate("injected_gpu_per_realization_tile_fold"),
    }
    return stages | {"total": sum(stages.values())}


def run(
    settings: runner.RunSettings, started: float, context: runner.RunContext
) -> tuple[str, runner.StageCounts]:
    """Entry from ``m32_recipe_eval.run`` for ``--injectant band``."""
    if settings.smoke:
        return run_smoke(settings, started, context)
    return STAGE_RUNNERS[settings.stage.value](settings, started, context)
