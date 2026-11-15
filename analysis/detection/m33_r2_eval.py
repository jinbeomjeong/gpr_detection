"""M3-3 R2 4TU detection report: ``m32_recipe_eval.py --stage uninjected --lines``.

Spec: ``docs/experiments/m33_runner_spec.md`` v0.3.6 §2 ([RUN] R2) and §5
(J1 inputs); common checks of §1 through ``m32_recipe_eval`` / ``m33_inputs``.

One run: r0 x folds on the 754 lines of ``--lines``
(``analysis/detection/m33_r2_lines.csv``, built by ``m33_r2_lines.py``), the
mirrored line for M5, batch 64. The C' line path (merge, then the sealed PP
on the merged boxes) gives every box twice (``pp`` ``pre`` = PP off, S0;
``post`` = the C0b set). Before the measurement the batch-1 D3 gate runs on
the K lines (M3-2 C-8, PP-off boxes).

Outputs (``--out-root``, default ``runs/eval/m33/r2``)

* ``fold<k>/boxes.csv``: merged boxes in the R3 box format (``pp``,
  ``box_index``, ``pp_action``, L0 box, ``in_grid``) plus recipe,
  orientation, ``keep_one``, M4 fields; R3 reads its zone (c') and the 4TU
  ghost pairs from it.
* ``fold<k>/apex_hits.csv`` (K apexes, PP before and after, SNR quartile and
  widths of ``analysis/processing/m33_k_apex_snr.csv``),
  ``fold<k>/conf_counts.csv``, ``fold<k>/flip_pairs.csv`` (both with ``pp``).
* ``fold<k>/j1_boxes_{pre,post}.csv`` (``pipe_hyperbola``) and
  ``fold<k>/j1_band_boxes_{pre,post}.csv`` (``pipe_band``): §5 J1 inputs.
* ``lines.csv``, ``gate.json``, ``summary.json`` (720 main and 754
  sensitivity tables, ``pp_actions``, ``pp_a_top_in_content``,
  ``j1_inputs``, ``judgment``, ``timings``).

Blind: values go to the files only; stdout carries lines, tiles, seconds,
phase names and paths.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import m31_d1_common as common
import m32_recipe_eval as runner
import m32_recipe_inputs as inputs
import m32_recipe_match as match
import m33_inputs
import numpy as np

from gpr_detection.io.classes import DetectionClass
from gpr_detection.io.radargram import GridBox
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
    image_box_to_grid_box,
    is_removed_inside_content,
    merge_line_boxes,
)
from gpr_detection.preprocess.to_image import DetectionImage

SCHEMA = "m33_r2_eval/1"
K_APEX_SNR = runner.R2_K_APEX_SNR
PP_PRE = "pre"
PP_POST = "post"
N_LINES_FULL = 754
N_KEEP_ONE = 720
AGREE_IOU = 0.5
"""``n_agree``: same-class L0 box of IoU >= 0.5 in another fold (D3 rule)."""
SCORE_QUANTILES = (0.1, 0.5, 0.9)
SMOKE_N_LINES = 3
PHASE_L1 = "r2_l1"
PHASE_INFER = "r2_infer"
PHASE_POST = "r2_merge_pp_rows"
PHASE_SUMMARY = "r2_summary"
SNR_COLUMNS = (
    "snr_db",
    "snr_quartile",
    "width_vis_P",
    "edge_vis",
    "width_m6_P",
    "width_m20_P",
    "edge_m20",
    "width_in_aihub_p5_p95",
)
"""X6 columns copied onto the apex rows ([RUN] §2-1 5 and 6)."""

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
    "recipe",
    "orientation",
    "role",
    "keep_one",
    "activity_id",
    "path",
    "tile_clipped",
    "m4",
    "m4c",
    "m4c_d3_8px",
)
"""The R3 ``BOX_COLUMNS`` first (same names, same order), then the R2 fields."""
APEX_COLUMNS = (
    "recipe",
    "fold",
    "pp",
    "blind_id",
    "line_id",
    "keep_one",
    "activity_id",
    "two_event",
    "trace",
    "sample",
    "hit",
    *SNR_COLUMNS,
)
CONF_COUNT_COLUMNS = (
    "recipe",
    "fold",
    "pp",
    "line_id",
    "keep_one",
    *(f"c{level:.2f}" for level in match.conf_grid()),
)
FLIP_COLUMNS = (
    "recipe",
    "fold",
    "pp",
    "line_id",
    "keep_one",
    "box_index",
    "class_id",
    "flip_box_index",
    "iou",
    "consistent",
)
J1_COLUMNS = (
    "line_id",
    "activity_id",
    "path",
    "fold",
    "class_id",
    "score",
    "i0",
    "i1",
    "k0",
    "k1",
    "x0_line_px",
    "y1_px",
    "y2_px",
    "pp_action",
    "tile_truncated",
    "n_tiles_seen",
)
"""§5 J1 box columns; ``x0_line_px`` is the left edge ``x1`` in line L1 px."""
LINE_COLUMNS = (
    *runner.LINE_COLUMNS,
    "keep_one",
    "activity_id",
    "path",
    "site",
    "two_event",
    "n_zero_events",
    "y_bot_px",
    "zero_px",
)
FOLD_FILES = {
    "boxes": BOX_COLUMNS,
    "apex_hits": APEX_COLUMNS,
    "conf_counts": CONF_COUNT_COLUMNS,
    "flip_pairs": FLIP_COLUMNS,
    "j1_boxes_pre": J1_COLUMNS,
    "j1_boxes_post": J1_COLUMNS,
    "j1_band_boxes_pre": J1_COLUMNS,
    "j1_band_boxes_post": J1_COLUMNS,
}


# ------------------------------------------------------------- line list


@dataclass(frozen=True)
class R2Line:
    """One line of the R2 list."""

    ref: inputs.LineRef
    keep_one: bool
    activity_id: str
    path: str
    site: str
    n_traces: int


def split_line_id(line_id: str) -> tuple[str, str]:
    """``(activity_id, path)``: ``line_id`` split at the first ``/`` ([RUN] §5)."""
    activity_id, path = line_id.split("/", 1)
    return activity_id, path


def read_r2_lines(path: Path) -> list[R2Line]:
    """The R2 line list (``m33_r2_lines.py`` output).

    Raises:
        inputs.InputError: For a missing file or a ``line_id`` that does not
            split into its ``activity_id`` and ``path`` columns.
    """
    if not path.is_file():
        raise inputs.InputError(f"R2 line list missing: {path}")
    lines = []
    for row in inputs.read_csv(path):
        if split_line_id(row["line_id"]) != (row["activity_id"], row["path"]):
            raise inputs.InputError(f"line_id {row['line_id']} != activity/path")
        lines.append(
            R2Line(
                ref=inputs.LineRef(row["dataset"], row["line_id"], row["role"]),
                keep_one=row["keep_one"] == "True",
                activity_id=row["activity_id"],
                path=row["path"],
                site=row["site"],
                n_traces=int(row["n_traces"]),
            )
        )
    return lines


def check_full_list(lines: Sequence[R2Line]) -> dict:
    """754 unique lines, 720 ``keep_one`` (a full run only).

    Raises:
        inputs.InputError: For other counts.
    """
    n_keep = sum(line.keep_one for line in lines)
    ids = {line.ref.line_id for line in lines}
    if (len(lines), len(ids), n_keep) != (N_LINES_FULL, N_LINES_FULL, N_KEEP_ONE):
        raise inputs.InputError(f"R2 list has {len(lines)} lines, {n_keep} keep_one")
    return {"n_lines": len(lines), "n_keep_one": n_keep}


def smoke_lines(lines: Sequence[R2Line]) -> list[R2Line]:
    """Shortest, middle and longest line (by traces) for the smoke."""
    ordered = sorted(lines, key=lambda line: (line.n_traces, line.ref.line_id))
    if len(ordered) <= SMOKE_N_LINES:
        return list(ordered)
    return [ordered[0], ordered[len(ordered) // 2], ordered[-1]]


def selected_lines(
    lines: Sequence[R2Line], settings: runner.RunSettings
) -> list[R2Line]:
    """``--only-lines`` / ``--limit-lines`` / smoke selection of the list."""
    if settings.only_lines:
        wanted = set(settings.only_lines)
        missing = wanted - {line.ref.line_id for line in lines}
        if missing:
            raise inputs.InputError(f"--only-lines not in the list: {sorted(missing)}")
        return [line for line in lines if line.ref.line_id in wanted]
    if settings.smoke:
        return smoke_lines(lines)
    if settings.limit_lines is not None:
        return list(lines[: settings.limit_lines])
    return list(lines)


def read_k_apex_snr(path: Path = K_APEX_SNR) -> dict[str, dict[str, str]]:
    """``blind_id -> X6 columns`` (SNR quartile and widths, detector-free)."""
    return {
        row["blind_id"]: {column: row[column] for column in SNR_COLUMNS}
        for row in inputs.read_csv(path)
    }


# --------------------------------------------------------------- per box


def _grid(box: LineBox, image: DetectionImage) -> GridBox | None:
    return image_box_to_grid_box(box.box, image.grid_map)


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
        "tile_clipped": box.tile_clipped,
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


def _m4_fields(box: LineBox, content: match.ContentRange, is_original: bool) -> dict:
    if not is_original:
        return {"m4": "", "m4c": False, "m4c_d3_8px": False}
    y1, y2 = box.box.y1, box.box.y2
    category = match.m4_category(y1, y2, content)
    return {
        "m4": "" if category is None else category.value,
        "m4c": match.is_m4c(box.class_id, y1, y2, content),
        "m4c_d3_8px": match.is_m4c(
            box.class_id, y1, y2, content, match.M4C_D3_WINDOW_PX
        ),
    }


@dataclass(frozen=True)
class LineGeometry:
    """Rows PP and M4 look at, for one line image."""

    image: DetectionImage
    pp: ContentRows
    m4: match.ContentRange


def pp_box_rows(
    base: Mapping,
    geometry: LineGeometry,
    outcomes: Sequence[PPOutcome],
    is_original: bool,
) -> list[dict]:
    """``pre`` row of every merged box and ``post`` row of every kept box."""
    rows = []
    for index, outcome in enumerate(outcomes):
        category = outcome.decision.category
        shared = {
            **base,
            "box_index": index,
            "pp_action": outcome.action.value,
            "pp_category": "" if category is None else category.value,
        }
        versions = [(PP_PRE, outcome.original)]
        if outcome.output is not None:
            versions.append((PP_POST, outcome.output))
        for pp, box in versions:
            rows.append(
                shared
                | {"pp": pp}
                | _box_fields(box, _grid(box, geometry.image))
                | _m4_fields(box, geometry.m4, is_original)
            )
    return rows


def scored_boxes(boxes: Iterable[LineBox], image: DetectionImage) -> list:
    """``ScoredBox`` of the boxes that reach the L0 grid."""
    scored = []
    for box in boxes:
        grid = _grid(box, image)
        if grid is not None:
            scored.append(match.ScoredBox(box.class_id, box.score, grid))
    return scored


def pp_versions(outcomes: Sequence[PPOutcome]) -> dict[str, list[LineBox]]:
    """``{"pre": every merged box, "post": the boxes PP hands on}``."""
    return {
        PP_PRE: [outcome.original for outcome in outcomes],
        PP_POST: [o.output for o in outcomes if o.output is not None],
    }


def j1_rows(
    line: R2Line,
    fold: int,
    geometry: LineGeometry,
    outcomes: Sequence[PPOutcome],
) -> tuple[dict[str, list[dict]], Counter[str]]:
    """§5 J1 rows: conf > 0.25 merged boxes on the grid, PP off and on.

    ``post`` leaves out the removed boxes and keeps ``clipped_b`` with its
    clipped bottom. Boxes wholly in the padding have no L0 box and are left
    out (J1 needs L0); they are counted per file (``n_padding_only_dropped``).

    Returns:
        ``(rows per J1 file, padding-only boxes left out per J1 file)``.
    """
    files: dict[str, list[dict]] = {name: [] for name in FOLD_FILES if "j1" in name}
    dropped: Counter[str] = Counter()
    for outcome in outcomes:
        versions = [(PP_PRE, outcome.original)]
        if outcome.output is not None:
            versions.append((PP_POST, outcome.output))
        for pp, box in versions:
            name = _j1_file(box.class_id, pp)
            if name is None or not box.score > runner.CONFIDENCE:
                continue
            grid = _grid(box, geometry.image)
            if grid is None:
                dropped[name] += 1
                continue
            files[name].append(_j1_row(line, fold, box, grid, outcome.action))
    return files, dropped


def _j1_file(class_id: int, pp: str) -> str | None:
    if class_id == DetectionClass.PIPE_HYPERBOLA:
        return f"j1_boxes_{pp}"
    if class_id == DetectionClass.PIPE_BAND:
        return f"j1_band_boxes_{pp}"
    return None


def _j1_row(
    line: R2Line, fold: int, box: LineBox, grid: GridBox, action: PPAction
) -> dict:
    return {
        "line_id": line.ref.line_id,
        "activity_id": line.activity_id,
        "path": line.path,
        "fold": fold,
        "class_id": box.class_id,
        "score": runner._exact(box.score),
        "i0": grid.i0,
        "i1": grid.i1,
        "k0": grid.k0,
        "k1": grid.k1,
        "x0_line_px": runner._exact(box.box.x1),
        "y1_px": runner._exact(box.box.y1),
        "y2_px": runner._exact(box.box.y2),
        "pp_action": action.value,
        "tile_truncated": box.tile_truncated,
        "n_tiles_seen": box.n_tiles_seen,
    }


# --------------------------------------------------------------- per line


@dataclass(frozen=True)
class R2Context:
    """What every line of the run shares."""

    settings: runner.RunSettings
    detectors: Mapping[int, object]
    config: PPConfig
    apexes: Sequence[inputs.KnownApex]
    snr: Mapping[str, Mapping[str, str]]


def line_record(line: R2Line, data: runner.LineData, geometry: LineGeometry) -> dict:
    """``lines.csv``: flags, ``keep_one``, ``two_event``, ``y_bot``, ``y_z``."""
    n_events = len(data.zero.events_ns)
    return runner.line_row(data, geometry.image) | {
        "keep_one": line.keep_one,
        "activity_id": line.activity_id,
        "path": line.path,
        "site": line.site,
        "two_event": n_events >= 2,
        "n_zero_events": n_events,
        "y_bot_px": runner._exact(geometry.pp.y_bot_px),
        "zero_px": ""
        if geometry.pp.zero_px is None
        else runner._exact(geometry.pp.zero_px),
    }


def conf_count_rows(
    line: R2Line,
    fold: int,
    geometry: LineGeometry,
    tile_boxes: Sequence[LineBox],
    config: PPConfig,
) -> list[dict]:
    """``pipe_hyperbola`` merged boxes at each C-5 conf level, PP off and on."""
    rows = []
    scores = [box.score for box in tile_boxes]
    for pp in (PP_PRE, PP_POST):

        def count(keep: np.ndarray, pp: str = pp) -> int:
            kept = runner.kept_boxes(tile_boxes, keep)
            merged = merge_line_boxes(kept, geometry.image, DEFAULT_MERGE_CONFIG)
            if pp == PP_POST:
                outcomes = apply_pp(merged, geometry.pp, config)
                merged = [o.output for o in outcomes if o.output is not None]
            return sum(box.class_id == DetectionClass.PIPE_HYPERBOLA for box in merged)

        counts = match.evaluate_on_conf_grid(scores, count)
        row = {
            "recipe": inputs.R0,
            "fold": fold,
            "pp": pp,
            "line_id": line.ref.line_id,
            "keep_one": line.keep_one,
        }
        for level, value in zip(match.conf_grid(), counts, strict=True):
            row[f"c{level:.2f}"] = value
        rows.append(row)
    return rows


def apex_rows(
    line: R2Line,
    fold: int,
    pp: str,
    boxes: Sequence[match.ScoredBox],
    period_samples: float,
    shared: R2Context,
    two_event: bool,
) -> list[dict]:
    """K apex hits (D3 rule) with the X6 SNR quartile and widths."""
    rows = []
    for apex in shared.apexes:
        if apex.line_id != line.ref.line_id:
            continue
        snr = shared.snr.get(apex.blind_id, dict.fromkeys(SNR_COLUMNS, ""))
        rows.append(
            {
                "recipe": inputs.R0,
                "fold": fold,
                "pp": pp,
                "blind_id": apex.blind_id,
                "line_id": apex.line_id,
                "keep_one": line.keep_one,
                "activity_id": line.activity_id,
                "two_event": two_event,
                "trace": apex.trace,
                "sample": apex.sample,
                "hit": match.apex_hit_d3(
                    apex.trace, apex.sample, boxes, period_samples
                ),
                **snr,
            }
        )
    return rows


def flip_rows(
    line: R2Line,
    data: runner.LineData,
    fold: int,
    pp: str,
    original: Sequence[match.ScoredBox],
    flipped: Sequence[match.ScoredBox],
) -> list[dict]:
    """M5 pairs of one PP version (mirrored boxes mapped back)."""
    n_traces = data.radargram.n_traces
    back = [
        match.ScoredBox(
            box.class_id, box.score, match.flip_grid_box(box.grid, n_traces)
        )
        for box in flipped
    ]
    return [
        {
            "recipe": inputs.R0,
            "fold": fold,
            "pp": pp,
            "line_id": line.ref.line_id,
            "keep_one": line.keep_one,
            "box_index": index,
            "class_id": original[index].class_id,
            "flip_box_index": "" if other is None else other,
            "iou": runner._exact(value),
            "consistent": other is not None,
        }
        for index, other, value in match.flip_pairs(original, back)
    ]


def _geometry(image: DetectionImage, data: runner.LineData) -> LineGeometry:
    return LineGeometry(
        image=image,
        pp=content_rows_for(image, data.zero),
        m4=runner.content_range(image, data.zero),
    )


def evaluate_r2_line(line: R2Line, data: runner.LineData, shared: R2Context) -> dict:
    """Every R2 row of one line (folds, both orientations, PP off and on)."""
    timer = runner.PhaseTimer()
    settings = shared.settings
    with timer.phase(PHASE_L1):
        jobs, originals = runner.uninjected_jobs(data, settings)
    with timer.phase(PHASE_INFER):
        detections = runner.run_jobs(jobs, shared.detectors, settings.batch_size)
    with timer.phase(PHASE_POST):
        part = _line_rows(line, data, jobs, detections, shared)
    part["line"] = [line_record(line, data, _geometry(originals[inputs.R0], data))]
    part["n_tiles"] = sum(len(job.tile_indices) for job in jobs)
    part["timings"] = dict(timer.seconds)
    return part


def _line_rows(
    line: R2Line,
    data: runner.LineData,
    jobs: Sequence[runner.ImageJob],
    detections: Mapping[tuple[int, int], list[LineBox]],
    shared: R2Context,
) -> dict:
    part: dict = {name: [] for name in FOLD_FILES}
    part["pp_actions"] = {}
    part["pp_a_top_in_content"] = {}
    part["j1_padding_dropped"] = {}
    two_event = len(data.zero.events_ns) >= 2
    scored: dict[tuple[str, int, str], list] = {}
    for (j, fold), tile_boxes in sorted(detections.items()):
        job = jobs[j]
        geometry = _geometry(job.image, data)
        is_original = job.orientation == runner.ORIGINAL
        merged = merge_line_boxes(tile_boxes, job.image, DEFAULT_MERGE_CONFIG)
        outcomes = apply_pp(merged, geometry.pp, shared.config)
        base = {
            "fold": fold,
            "line_id": line.ref.line_id,
            "dataset": line.ref.dataset,
            "realization_id": "",
            "recipe": job.recipe,
            "orientation": job.orientation,
            "role": line.ref.role,
            "keep_one": line.keep_one,
            "activity_id": line.activity_id,
            "path": line.path,
        }
        part["boxes"] += pp_box_rows(base, geometry, outcomes, is_original)
        for pp, boxes in pp_versions(outcomes).items():
            scored[(job.orientation, fold, pp)] = scored_boxes(boxes, job.image)
        if not is_original:
            continue
        _add_original_rows(part, line, fold, geometry, tile_boxes, outcomes, shared)
        for pp in (PP_PRE, PP_POST):
            part["apex_hits"] += apex_rows(
                line,
                fold,
                pp,
                scored[(runner.ORIGINAL, fold, pp)],
                job.image.period_samples,
                shared,
                two_event,
            )
    for (orientation, fold, pp), original in scored.items():
        mirrored = scored.get((runner.FLIPPED, fold, pp))
        if orientation == runner.ORIGINAL and mirrored is not None:
            part["flip_pairs"] += flip_rows(line, data, fold, pp, original, mirrored)
    return part


def _add_original_rows(
    part: dict,
    line: R2Line,
    fold: int,
    geometry: LineGeometry,
    tile_boxes: Sequence[LineBox],
    outcomes: Sequence[PPOutcome],
    shared: R2Context,
) -> None:
    """Conf counts, J1 rows and the PP tallies of the original orientation."""
    part["conf_counts"] += conf_count_rows(
        line, fold, geometry, tile_boxes, shared.config
    )
    files, dropped = j1_rows(line, fold, geometry, outcomes)
    for name, rows in files.items():
        part[name] += rows
    part["j1_padding_dropped"][str(fold)] = dict(dropped)
    actions = Counter(o.action.value for o in outcomes if o.action.is_processed)
    part["pp_actions"][str(fold)] = dict(actions)
    part["pp_a_top_in_content"][str(fold)] = sum(
        is_removed_inside_content(o.original.box.y1, o.action, geometry.pp)
        for o in outcomes
    )


# ================================================================ summary


def _keep(rows: Iterable[Mapping], scope: str) -> list[Mapping]:
    """``keep_one`` (720, main) or every line (754, sensitivity)."""
    rows = list(rows)
    if scope == "keep_one_720":
        return [row for row in rows if runner_bool(row["keep_one"])]
    return rows


def runner_bool(value: object) -> bool:
    return value is True or value == "True"


def k_inclusion(apex_rows_all: Sequence[Mapping], pp: str) -> dict:
    """K hits: 5-fold sum over apex x folds, any fold, by activity."""
    rows = [row for row in apex_rows_all if row["pp"] == pp]
    per_apex: dict[str, list[bool]] = {}
    activity_of: dict[str, str] = {}
    for row in rows:
        per_apex.setdefault(row["blind_id"], []).append(runner_bool(row["hit"]))
        activity_of[row["blind_id"]] = row["activity_id"]
    by_activity: dict[str, list[int]] = {}
    for blind_id, hits in per_apex.items():
        tally = by_activity.setdefault(activity_of[blind_id], [0, 0])
        tally[0] += sum(hits)
        tally[1] += len(hits)
    return {
        "sum_hits": sum(sum(hits) for hits in per_apex.values()),
        "apex_folds": sum(len(hits) for hits in per_apex.values()),
        "any_fold": sum(any(hits) for hits in per_apex.values()),
        "n_apexes": len(per_apex),
        "by_activity": {
            a: {"hits": t[0], "apex_folds": t[1]}
            for a, t in sorted(by_activity.items())
        },
    }


def _strata_hits(apex_rows_all: Sequence[Mapping], pp: str, key: str) -> dict:
    tallies: dict[str, list[int]] = {}
    for row in apex_rows_all:
        if row["pp"] != pp:
            continue
        tally = tallies.setdefault(str(row[key]), [0, 0])
        tally[0] += int(runner_bool(row["hit"]))
        tally[1] += 1
    return {k: {"hits": t[0], "apex_folds": t[1]} for k, t in sorted(tallies.items())}


def m4_table(box_rows: Sequence[Mapping], pp: str) -> dict:
    """M4a-d, M4c, D3 8 px counts and the lines with an M4 box."""
    rows = [
        r for r in box_rows if r["pp"] == pp and r["orientation"] == runner.ORIGINAL
    ]
    categories = Counter(row["m4"] for row in rows if row["m4"])
    lines = {row["line_id"] for row in rows if row["m4"]}
    return {
        "by_category": dict(sorted(categories.items())),
        "m4c": sum(runner_bool(row["m4c"]) for row in rows),
        "m4c_d3_8px": sum(runner_bool(row["m4c_d3_8px"]) for row in rows),
        "n_lines_with_m4": len(lines),
    }


def class_table(box_rows: Sequence[Mapping], pp: str) -> dict:
    """Box count and score quantiles per class (original orientation)."""
    scores: dict[str, list[float]] = {}
    for row in box_rows:
        if row["pp"] == pp and row["orientation"] == runner.ORIGINAL:
            scores.setdefault(row["class_name"], []).append(float(row["score"]))
    table = {}
    for name, values in sorted(scores.items()):
        quantiles = np.quantile(np.asarray(values), SCORE_QUANTILES)
        table[name] = {
            "n_boxes": len(values),
            "score_quantiles": {
                f"q{round(q * 100)}": float(v)
                for q, v in zip(SCORE_QUANTILES, quantiles, strict=True)
            },
        }
    return table


def n_agree_table(box_rows: Sequence[Mapping], pp: str, folds: Sequence[int]) -> dict:
    """``n_agree`` (1-5, fold 0 included) of the fold-0 boxes per class."""
    if 0 not in folds:
        return {}
    by_line_fold: dict[tuple[str, int], list[tuple[int, GridBox]]] = {}
    for row in box_rows:
        if row["pp"] != pp or row["orientation"] != runner.ORIGINAL:
            continue
        if not runner_bool(row["in_grid"]):
            continue
        grid = GridBox(int(row["i0"]), int(row["i1"]), int(row["k0"]), int(row["k1"]))
        key = (row["line_id"], int(row["fold"]))
        by_line_fold.setdefault(key, []).append((int(row["class_id"]), grid))
    table: dict[str, Counter[int]] = {}
    for (line_id, fold), boxes in by_line_fold.items():
        if fold != 0:
            continue
        for class_id, grid in boxes:
            agree = 1 + sum(
                any(
                    other_class == class_id and match.grid_iou(grid, other) >= AGREE_IOU
                    for other_class, other in by_line_fold.get(
                        (line_id, other_fold), ()
                    )
                )
                for other_fold in folds
                if other_fold != 0
            )
            name = common.CLASS_NAMES[class_id]
            table.setdefault(name, Counter())[agree] += 1
    return {name: dict(sorted(c.items())) for name, c in sorted(table.items())}


def flip_table(flip_rows_all: Sequence[Mapping], pp: str) -> dict:
    """Mirror consistency per class: consistent pairs / boxes."""
    tallies: dict[str, list[int]] = {}
    for row in flip_rows_all:
        if row["pp"] != pp:
            continue
        name = common.CLASS_NAMES[int(row["class_id"])]
        tally = tallies.setdefault(name, [0, 0])
        tally[0] += int(runner_bool(row["consistent"]))
        tally[1] += 1
    return {
        name: {"consistent": t[0], "n_boxes": t[1], "rate": t[0] / t[1]}
        for name, t in sorted(tallies.items())
    }


def missed_apex_widths(apex_rows_all: Sequence[Mapping], pp: str) -> list[dict]:
    """K apexes missed by at least one fold, with the X6 widths (§2-1 6).

    ``n_folds_missed`` (1-5) counts the folds that missed the apex (cv
    decision 2026-11-16: "놓친 K 꼭짓점" does not pick folds).
    """
    missed: Counter[str] = Counter()
    first: dict[str, Mapping] = {}
    for row in apex_rows_all:
        if row["pp"] != pp:
            continue
        missed[row["blind_id"]] += int(not runner_bool(row["hit"]))
        first.setdefault(row["blind_id"], row)
    return [
        {
            "blind_id": blind_id,
            "line_id": first[blind_id]["line_id"],
            "n_folds_missed": missed[blind_id],
        }
        | {column: first[blind_id][column] for column in SNR_COLUMNS}
        for blind_id in sorted(first)
        if missed[blind_id] > 0
    ]


def scope_tables(parts: Sequence[Mapping], scope: str, folds: Sequence[int]) -> dict:
    """Every §2-2 table of one line scope, PP off and on."""
    boxes = _keep((r for p in parts for r in p["boxes"]), scope)
    apexes = _keep((r for p in parts for r in p["apex_hits"]), scope)
    flips = _keep((r for p in parts for r in p["flip_pairs"]), scope)
    lines = _keep((r for p in parts for r in p["line"]), scope)
    tables = {"n_lines": len(lines)}
    for pp in (PP_PRE, PP_POST):
        tables[pp] = {
            "k_inclusion": k_inclusion(apexes, pp),
            "m4": m4_table(boxes, pp),
            "classes": class_table(boxes, pp),
            "n_agree_fold0": n_agree_table(boxes, pp, folds),
            "flip_consistency": flip_table(flips, pp),
            "k_hits_by_snr_quartile": _strata_hits(apexes, pp, "snr_quartile"),
            "k_hits_by_two_event": _strata_hits(apexes, pp, "two_event"),
            "missed_apex_widths": missed_apex_widths(apexes, pp),
        }
    tables["two_event_lines"] = sum(runner_bool(r["two_event"]) for r in lines)
    return tables


def pp_tallies(parts: Sequence[Mapping], folds: Sequence[int]) -> dict:
    """``pp_actions`` and ``pp_a_top_in_content`` per fold and summed."""
    actions: dict[str, Counter[str]] = {str(f): Counter() for f in folds}
    top: Counter[str] = Counter()
    for part in parts:
        for fold, counts in part["pp_actions"].items():
            actions[fold].update(counts)
        top.update(part["pp_a_top_in_content"])
    total: Counter[str] = Counter()
    for counts in actions.values():
        total.update(counts)
    return {
        "pp_actions": {
            "by_fold": {f: dict(sorted(c.items())) for f, c in actions.items()},
            "total": dict(sorted(total.items())),
        },
        "pp_a_top_in_content": {
            "by_fold": {str(f): top.get(str(f), 0) for f in folds},
            "total": sum(top.values()),
        },
    }


def j1_inputs_record(
    stage_dir: Path,
    folds: Sequence[int],
    sealed_pp: Mapping,
    weights: Mapping[str, str],
    n_lines: int,
    padding_dropped: Mapping[str, Mapping[str, int]] | None = None,
) -> dict:
    """``j1_inputs`` of §5: fold -> paths, sha256 and box counts.

    ``n_padding_only_dropped`` is the ``post`` file count of conf > 0.25
    ``pipe_hyperbola`` boxes left out for lying wholly in the padding;
    ``n_padding_only_dropped_pre`` and the band file counts sit beside it.
    """
    padding_dropped = padding_dropped or {}
    record = {
        "pp_set_id": sealed_pp["pp_set_id"],
        "pp_config_sha256": sealed_pp["pp_config_sha256"],
        "weights_sha256": dict(weights),
        "n_lines": n_lines,
        "folds": {},
        "band_files": {},
    }
    for fold in folds:
        fold_dir = stage_dir / f"fold{fold}"
        entry = {}
        for pp in (PP_POST, PP_PRE):
            path = fold_dir / f"j1_boxes_{pp}.csv"
            entry[f"{pp}_path"] = path.as_posix()
            entry[f"{pp}_sha256"] = inputs.sha256_file(path)
            entry[f"n_boxes_{pp}"] = len(inputs.read_csv(path))
        dropped = padding_dropped.get(str(fold), {})
        entry["n_padding_only_dropped"] = dropped.get("j1_boxes_post", 0)
        entry["n_padding_only_dropped_pre"] = dropped.get("j1_boxes_pre", 0)
        record["folds"][str(fold)] = entry
        record["band_files"][str(fold)] = {
            pp: inputs.sha256_file(fold_dir / f"j1_band_boxes_{pp}.csv")
            for pp in (PP_PRE, PP_POST)
        } | {
            f"n_padding_only_dropped_{pp}": dropped.get(f"j1_band_boxes_{pp}", 0)
            for pp in (PP_PRE, PP_POST)
        }
    return record


def padding_dropped_by_fold(parts: Iterable[Mapping]) -> dict[str, dict[str, int]]:
    """``fold -> J1 file -> padding-only boxes left out``, over the lines."""
    total: dict[str, Counter[str]] = {}
    for part in parts:
        for fold, counts in part.get("j1_padding_dropped", {}).items():
            total.setdefault(fold, Counter()).update(counts)
    return {fold: dict(counts) for fold, counts in total.items()}


# ================================================================== stage


def r2_hashes(settings: runner.RunSettings) -> dict[str, str]:
    """LF sha of the R2 text inputs (the line list is asserted)."""
    hashes = inputs.assert_list_hashes()
    hashes["r2_lines"] = inputs.checked_text_sha256(
        settings.lines_csv, m33_inputs.R2_LINES_SHA256_LF, "R2 line list"
    )
    hashes["k_apex_snr"] = inputs.checked_text_sha256(
        K_APEX_SNR, m33_inputs.K_APEX_SNR_SHA256_LF, "K apex SNR table"
    )
    return hashes


def run_r2(
    settings: runner.RunSettings, started: float, context: runner.RunContext
) -> tuple[str, runner.StageCounts]:
    """R2: D3 gate on the K lines, then every row of the selected lines."""
    config = PP_SETS[PPSetId(context.sealed_pp["pp_set_id"])]
    hashes = r2_hashes(settings)
    weights = runner.assert_weights(settings.run_root, settings.folds)
    hashes.update(weights)
    all_lines = read_r2_lines(settings.lines_csv)
    lines = selected_lines(all_lines, settings)
    if not settings.scope_blockers():
        check_full_list(lines)
    runner.prepare_stage_dir(settings)
    runner.check_resume_identity(settings, runner.settings_sha256(settings, hashes))
    identity = runner.new_run_identity(settings, hashes, context)
    identity["tile_order"] = runner.TILE_ORDER
    detectors = runner.load_detectors(settings)
    apexes = inputs.known_apexes()
    gate = runner._gate_or_load(
        settings, [line.ref for line in lines], detectors, apexes
    )
    if settings.is_full and not (gate is not None and gate["result"]["pass"]):
        return _gate_failed(settings, context, identity, gate)
    shared = R2Context(settings, detectors, config, apexes, read_k_apex_snr())
    by_id = {line.ref.line_id: line for line in lines}
    parts = runner.run_lines(
        settings,
        [line.ref for line in lines],
        lambda data: evaluate_r2_line(by_id[data.ref.line_id], data, shared),
        started,
    )
    status, judgment = runner.close_run(
        settings, context, identity, m33_inputs.STATUS_OK
    )
    _write_outputs(settings, parts)
    summary = _summary(settings, parts, gate, identity, status, judgment, started)
    runner._write_json(settings.stage_dir / "summary.json", summary)
    counts = runner.StageCounts(len(parts), sum(part["n_tiles"] for part in parts))
    return status, counts


def _gate_failed(settings, context, identity, gate) -> tuple[str, runner.StageCounts]:
    status, judgment = runner.close_run(settings, context, identity, "gate_failed")
    runner._write_json(
        settings.stage_dir / "summary.json",
        {
            "schema": SCHEMA,
            "status": status,
            "full_run": True,
            "judgment": judgment,
            "run_identity": {**identity, "finished_at": common.local_now()},
            "identity_gate": None if gate is None else gate["result"],
        },
    )
    return status, runner.StageCounts()


def _write_outputs(settings: runner.RunSettings, parts: Sequence[Mapping]) -> None:
    for name, columns in FOLD_FILES.items():
        rows = [row for part in parts for row in part[name]]
        for fold in settings.folds:
            selected = [row for row in rows if int(row["fold"]) == fold]
            path = settings.stage_dir / f"fold{fold}" / f"{name}.csv"
            runner._write_rows(path, columns, selected)
    runner._write_rows(
        settings.stage_dir / "lines.csv",
        LINE_COLUMNS,
        (row for part in parts for row in part["line"]),
    )


def _summary(
    settings: runner.RunSettings,
    parts: Sequence[Mapping],
    gate: Mapping | None,
    identity: dict,
    status: str,
    judgment: dict,
    started: float,
) -> dict:
    timer = runner.PhaseTimer(_summed_timings(parts))
    with timer.phase(PHASE_SUMMARY):
        tables = {
            scope: scope_tables(parts, scope, settings.folds)
            for scope in ("keep_one_720", "all_754")
        }
        j1 = j1_inputs_record(
            settings.stage_dir,
            settings.folds,
            identity["sealed_pp_assertion"],
            {k: v for k, v in identity["inputs_sha256"].items() if k.isdigit()},
            len(parts),
            padding_dropped_by_fold(parts),
        )
    for name, value in timer.seconds.items():
        print(f"timing {name}: {value:.1f} s", flush=True)
    if gate is not None:
        identity["batch_path"] = runner.batch_path_report(gate, _gate_parts(parts))
    return {
        "schema": SCHEMA,
        "status": status,
        "full_run": settings.is_full,
        "gated": settings.is_full,
        "judgment": judgment,
        "run_identity": {**identity, "finished_at": common.local_now()},
        "identity_gate": None if gate is None else gate["result"],
        "n_lines": len(parts),
        "n_tiles": sum(part["n_tiles"] for part in parts),
        "folds": list(settings.folds),
        "flip": settings.flip,
        "summary_main": "keep_one_720",
        "tables": tables,
        **pp_tallies(parts, settings.folds),
        "j1_inputs": j1,
        "timings": dict(timer.seconds),
        "resources": runner.resources(settings, started),
    }


def _gate_parts(parts: Sequence[Mapping]) -> list[dict]:
    """PP-off original rows in the M3-2 shape ``batch_path_report`` reads."""
    return [
        {
            "apex_hits": [r for r in part["apex_hits"] if r["pp"] == PP_PRE],
            "boxes": [
                r
                for r in part["boxes"]
                if r["pp"] == PP_PRE and r["orientation"] == runner.ORIGINAL
            ],
        }
        for part in parts
    ]


def _summed_timings(parts: Iterable[Mapping]) -> dict[str, float]:
    total: Counter[str] = Counter()
    for part in parts:
        total.update(part.get("timings", {}))
    return dict(total)
