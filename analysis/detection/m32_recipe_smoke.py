"""M3-2 S5 ``--smoke``: time and VRAM of the stage-C paths, and a full-run estimate.

Runs r0 x fold 0 on three stage-A 4TU lines (fewest tiles, the median of the
lines with two or more tiles, most tiles) through the P5a path (with the
mirror run) and the P5b path (a plan with the smoke P5a boxes as zone (c'),
all ten conditions), in its own folder. Nothing here is a judgement input
([CVR] 4-2 "smoke").

The estimate multiplies measured rates by the work of a full run ([CVR]
appendix B-3):

* P5a: tiles per line from ``m32_content_height_lines.csv`` (recipes x
  orientations x folds inferences, recipes x orientations L1 builds, one
  load and zero per line).
* P5b, three work tables, the same rates:
  ``upper``: 2 realizations x 10 conditions per line, every tile inferred;
  ``smoke_scaled``: the upper bound times the smoke plan's work ratio;
  ``cost_table``: a plan stage cost table (``--cost-table``, e.g. the
  provisional plan without zone (c'), itself an upper bound of the real plan).

Also compares batched and one-by-one inference on the tiles of the longest
smoke line, because all stage-C runs use the batched path while D3, the M6
gate reference, ran one tile per call.
"""

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import m31_d1_common as common
import m32_pipe_responses as responses_io
import m32_recipe_eval as runner
import m32_recipe_inputs as inputs
import numpy as np

from gpr_detection.eval import inject

N_CONDITIONS = len(inject.default_conditions())
MAX_REALIZATIONS = inject.MAX_REALIZATIONS_PER_LINE
N_ORIENTATIONS = 2


def tile_counts() -> dict[str, int]:
    """Tiles per line of the r0 image (``m32_content_height_lines.csv``)."""
    return {
        row["line"]: int(row["n_tiles"])
        for row in inputs.read_csv(inputs.CONTENT_HEIGHT_LINES)
    }


def _warm_up(detectors: Mapping[int, object], batch_size: int) -> None:
    """One inference before timing (CUDA context and cuDNN autotune)."""
    blank = np.full((runner.INPUT_PX, runner.INPUT_PX), 128, dtype=np.uint8)
    for detector in detectors.values():
        runner.predict_tiles(detector, [blank] * min(batch_size, 4), batch_size)


def batch_versus_single(
    detector: object, tiles: Sequence[np.ndarray], batch_size: int
) -> dict:
    """Detections of the batched path against one call per tile."""
    batched = runner.predict_tiles(detector, tiles, batch_size)
    single = runner.predict_tiles(detector, tiles, 1)
    max_score, max_corner, count_mismatch = 0.0, 0.0, 0
    for first, second in zip(batched, single, strict=True):
        if len(first.detections) != len(second.detections):
            count_mismatch += 1
            continue
        pairs = zip(
            sorted(first.detections, key=lambda d: -d.score),
            sorted(second.detections, key=lambda d: -d.score),
            strict=True,
        )
        for a, b in pairs:
            max_score = max(max_score, abs(a.score - b.score))
            corners = np.abs(np.subtract(a.box.xyxy, b.box.xyxy))
            max_corner = max(max_corner, float(corners.max()))
    return {
        "n_tiles": len(tiles),
        "tiles_with_other_box_count": count_mismatch,
        "max_abs_score_difference": max_score,
        "max_abs_corner_difference_px": max_corner,
    }


# ------------------------------------------------------------- measuring


def _smoke_p5a(
    ref: inputs.LineRef,
    detectors: Mapping[int, object],
    settings: runner.RunSettings,
    apexes: Sequence[inputs.KnownApex],
) -> tuple[runner.LineData, dict, dict]:
    timer = runner.PhaseTimer()
    with timer.phase("load"):
        line = runner.load_line(ref)
    part = runner.evaluate_uninjected_line(line, detectors, settings, apexes, timer)
    measured = {
        "line_id": ref.line_id,
        "n_traces": line.radargram.n_traces,
        "n_samples": line.radargram.n_samples,
        "n_tiles_line": part["line"][0]["n_tiles"],
        "n_tiles_inferred": part["n_tiles"],
        "seconds": timer.seconds,
    }
    return line, part, measured


def _smoke_p5b(
    line: runner.LineData,
    boxes: Sequence,
    library: responses_io.ResponseLibrary,
    detectors: Mapping[int, object],
    settings: runner.RunSettings,
) -> dict:
    exclusions = runner.line_exclusions(
        line.ref.line_id,
        {line.ref.line_id: list(boxes)},
        inputs.candidate_apexes(),
    )
    timer = runner.PhaseTimer()
    with timer.phase("plan"):
        planned = runner.plan_line(line, exclusions, library)
    realizations = runner.group_specs(planned.specs).get(line.ref.line_id, {})
    level_w_db, _ = runner.select_level_w(planned.positions)
    part = runner.evaluate_injected_line(
        line,
        realizations,
        exclusions,
        library,
        detectors,
        settings,
        timer,
        level_w_db=level_w_db,
    )
    rejections_by_condition = runner.count_by(
        runner.with_condition(part["rejections"]), ("reason", "snr_db", "k")
    )
    accepted_by_condition = runner.count_by(part["inj_matches"], ("snr_db", "k"))
    return {
        "line_id": line.ref.line_id,
        "n_positions": planned.report["n_positions"],
        "n_realizations": planned.cost.n_realizations,
        "realization_conditions": len(realizations),
        "l1_builds": part.get("n_l1_builds", 0),
        "inferred_tiles": part["n_tiles"],
        "planned_inferred_tiles": planned.cost.inferred_tiles,
        "n_accepted": part["n_accepted"],
        "rejection_reasons": runner.count_by(part["rejections"], ("reason",)),
        "rejection_reasons_by_snr_k": rejections_by_condition,
        "accepted_rows_by_snr_k": accepted_by_condition,
        "seconds": timer.seconds,
    }


# -------------------------------------------------------------- estimate


@dataclass(frozen=True)
class Rates:
    """Measured seconds per unit of work (r0, fold 0)."""

    load_per_line_tile: float
    l1_per_line_tile: float
    infer_per_tile: float
    post_per_tile: float
    context_per_line_tile: float
    inject_per_realization: float
    plan_per_line_tile: float


def _ratio(seconds: float, units: float) -> float:
    return seconds / units if units > 0 else 0.0


def _phase(rows: Iterable[Mapping], name: str) -> float:
    return sum(row["seconds"].get(name, 0.0) for row in rows)


def measure_rates(
    p5a: Sequence[Mapping], p5b: Sequence[Mapping], tiles: Mapping[str, int]
) -> Rates:
    """Rates from the smoke rows; L1 and context cost scale with line tiles."""
    a_tiles = sum(row["n_tiles_line"] for row in p5a)
    a_inferred = sum(row["n_tiles_inferred"] for row in p5a)
    b_line_tiles = [tiles.get(row["line_id"], 1) for row in p5b]
    b_build_tiles = sum(
        row["l1_builds"] * n for row, n in zip(p5b, b_line_tiles, strict=True)
    )
    b_inferred = sum(row["inferred_tiles"] for row in p5b)
    return Rates(
        load_per_line_tile=_ratio(_phase(p5a, "load"), a_tiles),
        l1_per_line_tile=_ratio(
            _phase(p5a, "l1") + _phase(p5b, "l1"),
            N_ORIENTATIONS * a_tiles + b_build_tiles,
        ),
        infer_per_tile=_ratio(
            _phase(p5a, "infer") + _phase(p5b, "infer"), a_inferred + b_inferred
        ),
        post_per_tile=_ratio(
            _phase(p5a, "post") + _phase(p5b, "match"), a_inferred + b_inferred
        ),
        context_per_line_tile=_ratio(_phase(p5b, "context"), sum(b_line_tiles)),
        inject_per_realization=_ratio(
            _phase(p5b, "inject"), sum(r["realization_conditions"] for r in p5b)
        ),
        plan_per_line_tile=_ratio(_phase(p5b, "plan"), sum(b_line_tiles)),
    )


def p5a_seconds(
    rates: Rates, tiles: Mapping[str, int], n_recipes: int, n_folds: int
) -> dict:
    """P5a time over the uninjected lines (unknown tile counts: the median)."""
    lines = inputs.uninjected_lines()
    known = [tiles[ref.line_id] for ref in lines if ref.line_id in tiles]
    fallback = statistics.median(known) if known else 1
    line_tiles = sum(tiles.get(ref.line_id, fallback) for ref in lines)
    builds = n_recipes * N_ORIENTATIONS
    per_line_tile = (
        rates.load_per_line_tile
        + builds * rates.l1_per_line_tile
        + builds * n_folds * (rates.infer_per_tile + rates.post_per_tile)
    )
    return {"line_tiles": line_tiles, "seconds": per_line_tile * line_tiles}


@dataclass(frozen=True)
class LineWork:
    """P5b work of one line per recipe and fold (``inject.LineCost`` units)."""

    line_id: str
    l1_builds: int
    inferred_tiles: int


def p5b_seconds(
    rates: Rates,
    work: Iterable[LineWork],
    tiles: Mapping[str, int],
    n_recipes: int,
    n_folds: int,
) -> float:
    """P5b time of a work table (injection once, L1 per recipe, folds share L1)."""
    seconds = 0.0
    for line in work:
        line_tiles = tiles.get(line.line_id, 1)
        seconds += rates.context_per_line_tile * line_tiles
        seconds += line.l1_builds * rates.inject_per_realization
        seconds += n_recipes * line.l1_builds * line_tiles * rates.l1_per_line_tile
        seconds += (
            n_recipes
            * n_folds
            * line.inferred_tiles
            * (rates.infer_per_tile + rates.post_per_tile)
        )
    return seconds


def upper_work(tiles: Mapping[str, int]) -> list[LineWork]:
    """2 realizations x 10 conditions per background line, every tile inferred."""
    builds = MAX_REALIZATIONS * N_CONDITIONS
    return [
        LineWork(ref.line_id, builds, builds * tiles.get(ref.line_id, 1))
        for ref in inputs.background_lines()
    ]


def read_cost_table(path: Path) -> list[LineWork]:
    """``cost_table.csv`` of a plan stage as work rows."""
    return [
        LineWork(row["line_id"], int(row["l1_builds"]), int(row["inferred_tiles"]))
        for row in inputs.read_csv(path)
    ]


def estimate(
    p5a: Sequence[Mapping],
    p5b: Sequence[Mapping],
    tiles: Mapping[str, int],
    cost_table: Path | None = None,
    n_recipes: int = len(inputs.STAGE_C_RECIPES),
    n_folds: int = len(inputs.FOLDS),
) -> dict:
    """Full-run seconds of P5a, the plan stage and P5b (see module docstring)."""
    rates = measure_rates(p5a, p5b, tiles)
    upper = upper_work(tiles)
    smoke_upper = sum(
        MAX_REALIZATIONS * N_CONDITIONS * tiles.get(row["line_id"], 1) for row in p5b
    )
    smoke_ratio = _ratio(sum(row["inferred_tiles"] for row in p5b), smoke_upper)
    upper_s = p5b_seconds(rates, upper, tiles, n_recipes, n_folds)
    p5b_estimate: dict[str, object] = {
        "upper_bound_seconds": upper_s,
        "smoke_work_ratio_to_upper": smoke_ratio,
        "smoke_scaled_seconds": upper_s * smoke_ratio,
        "cost_table": None,
    }
    if cost_table is not None:
        work = read_cost_table(cost_table)
        p5b_estimate["cost_table"] = {
            "path": str(cost_table),
            "sha256": inputs.sha256_file(cost_table),
            "l1_builds_per_recipe": sum(line.l1_builds for line in work),
            "inferred_tiles_per_recipe_fold": sum(line.inferred_tiles for line in work),
            "seconds": p5b_seconds(rates, work, tiles, n_recipes, n_folds),
        }
    background_tiles = sum(tiles.get(r.line_id, 1) for r in inputs.background_lines())
    return {
        "rates": rates.__dict__,
        "p5a": p5a_seconds(rates, tiles, n_recipes, n_folds),
        "plan_stage_single_process_seconds": rates.plan_per_line_tile
        * background_tiles,
        "p5b": p5b_estimate,
        "assumptions": (
            "r5/r6 L1 builds cost as much as r0; rates come from the batched "
            "path (batch 64) on one GPU; per-line Python overhead outside the "
            "timed phases (csv writing) is not counted"
        ),
    }


# ------------------------------------------------------------------ entry


def run(settings: runner.RunSettings, started: float) -> tuple[str, runner.StageCounts]:
    """Smoke entry called by ``m32_recipe_eval --smoke``."""
    hashes = inputs.assert_list_hashes()
    hashes.update(runner.assert_weights(settings.run_root, settings.folds))
    out_dir = settings.out_root
    if out_dir.exists():
        runner.shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    smoke_settings = replace(settings, stage=runner.Stage.UNINJECTED)
    detectors = runner.load_detectors(smoke_settings)
    _warm_up(detectors, settings.batch_size)
    library = responses_io.load_library()
    apexes = inputs.known_apexes()
    refs = runner.smoke_lines()
    p5a_rows, p5b_rows, n_tiles, n_rows = [], [], 0, 0
    batch_check: dict = {}
    for ref in refs:
        line, part, measured = _smoke_p5a(ref, detectors, smoke_settings, apexes)
        p5a_rows.append(measured)
        if ref == refs[-1]:  # the most tiles
            batch_check = _batch_check(line, detectors, settings)
        measured_b = _smoke_p5b(
            line, _original_boxes(part), library, detectors, smoke_settings
        )
        p5b_rows.append(measured_b)
        n_tiles += part["n_tiles"] + measured_b["inferred_tiles"]
        n_rows += measured_b["n_accepted"] + sum(
            measured_b["rejection_reasons"].values()
        )
        print(f"smoke {ref.line_id}: {measured['n_tiles_line']} line tiles", flush=True)
    document = {
        "schema": "m32_recipe_smoke/1",
        "status": "ok",
        "not_for_judgement": True,
        "run_identity": runner.new_run_identity(settings, hashes),
        "lines": [ref.line_id for ref in refs],
        "p5a_measured": p5a_rows,
        "p5b_measured": p5b_rows,
        "batch_versus_single": batch_check,
        "estimate": estimate(p5a_rows, p5b_rows, tile_counts(), settings.cost_table),
        "resources": runner.resources(settings, started),
    }
    common.write_json(out_dir / "smoke_estimate.json", document)
    return "ok", runner.StageCounts(len(refs), n_tiles, n_rows)


def _original_boxes(part: Mapping) -> list:
    return [
        runner.GridBox(row["i0"], row["i1"], row["k0"], row["k1"])
        for row in part["boxes"]
        if row["orientation"] == runner.ORIGINAL and row["in_grid"]
    ]


def _batch_check(
    line: runner.LineData, detectors: Mapping[int, object], settings: runner.RunSettings
) -> dict:
    image = runner.make_detection_image(line.radargram, inputs.recipe_config(inputs.R0))
    tiles = [image.tile_image(index) for index in range(len(image.tiles))]
    detector = next(iter(detectors.values()))
    return batch_versus_single(detector, tiles, settings.batch_size)
