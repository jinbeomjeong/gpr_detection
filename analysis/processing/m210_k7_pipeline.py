"""K7 harness (M2-10 K7, docs/processing/m210_k7_generator.md v1.1 sections 4-4, 5).

Runs the 4TU code path with the four generator variants on the new gprMax K7
lines and writes one row per run and pipe. **No verdict is made here**
(``m210_k7_judge.py`` reads ``runs_k7.csv`` and ``run_info.json``).

Per model (``MODEL_IDS``, the section 4-3 table order = seed index m):

1. ``load_synthetic(id, include_eps=True, include_radius=False,
   include_height=True)``, ``line_id`` ``K7/<id>``, pipe truths from
   ``load_synthetic_pipe_truths``; resampled to 0.098 ns (``resample_to`` of the
   (b) harness, v1.4 rule 2).
2. Noise-free processing (``prepare_depth_inputs``, envelope of the
   background-removed branch): per pipe ``t_a0`` = envelope maximum of the apex
   trace within the json ``envelope_peak_ns`` +-0.5/f_c, and ``A0_pipe`` =
   envelope maximum over apex trace +-3 and ``t_a0`` +-0.5/f_c. ``A0`` = the
   smallest ``A0_pipe`` of the model; K7_BG_L uses the K7_P2_D070_L calibration
   (``CALIBRATION_PAIRS``). ``sigma_u`` = 1.4826 MAD over the search rows of the
   processed unit-noise section (seed 20301199); ``sigma_raw = A0 / (rho
   sigma_u)``.
3. Model x rho {10, 20} x seed 0-39: white noise ``sigma_raw n`` (seed
   ``20301112 + 1000 m + 100 q + s``) on the resampled raw line, one
   ``prepare_depth_inputs``, then every variant (``cur``, ``a1``, ``a1b2``,
   ``a1b1``, ``GENERATOR_VARIANTS`` of the generator) through the wrapper ->
   ``eval_depth_4tu.run_line(apex_qc=True)`` -> QC-passing candidates attached
   to the ``expected_events`` of ``models_k7.json`` (S1g ``attach_event``).
   One noise-free row per model, variant and pipe (``noise_free`` True).

Outputs (``--out-dir``, default ``data/processing/m210_k7``): ``runs_k7.csv``
(``RUN_COLUMNS``), ``run_info.json`` and ``pre_run.txt`` / ``post_run.txt``,
which only this harness writes. Run guards as the (b) v2/v3 harness: seal record
and inputs before the row loop, a failed start guard (sealed file, dirty tree,
HEAD other than the seal, failed git query, failed K7 tests) ends the run at
once; the end guards compare HEAD, the sealed-file records, the inputs and the G0
regressions (4TU csvs byte for byte, S1g ``candidates.csv``, the (b) v3 subset).

Run (one core, BelowNormal; about 40-70 min):
    conda run -n gpr-dev python -I analysis/processing/m210_k7_pipeline.py \
        --protocol k7 --seal-commit <K7 seal commit>
"""

import argparse
import csv
import dataclasses
import datetime
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# Reproducible MKL results across processes, before numpy loads (M2-10); an
# explicit value from the caller wins. Same setting as the (b) and S1g harnesses.
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import numpy as np
import scipy
import skimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "analysis" / "processing"))

import eval_depth_4tu as depth_evaluation  # noqa: E402
import fourtu_apex_candidate_select as generator  # noqa: E402
import m210_s1g_pipeline as s1g  # noqa: E402
import m210b_floor_rules as floor_rules  # noqa: E402
import m210b_s4s6_pipeline as floor_harness  # noqa: E402

from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402
from gpr_detection.io.radargram import Radargram  # noqa: E402
from gpr_detection.io.synthetic import (  # noqa: E402
    SyntheticTruth,
    load_synthetic,
    load_synthetic_pipe_truths,
)
from gpr_detection.preprocess.signals import pick_signal  # noqa: E402

PROTOCOL = "k7"
SPEC_DOC = "docs/processing/m210_k7_generator.md"
HARNESS_PATH = "analysis/processing/m210_k7_pipeline.py"
GENERATOR_PATH = "analysis/processing/fourtu_apex_candidate_select.py"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "synthetic_gprmax_k7"
DEFAULT_MODELS_JSON = PROJECT_ROOT / "analysis" / "synthetic" / "k7" / "models_k7.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processing" / "m210_k7"
RUNS_CSV = "runs_k7.csv"
LINE_ACTIVITY = "K7"
JSON_BATCH = "k7"

MODEL_IDS = (
    "K7_P2_D070",
    "K7_P2_D070_L",
    "K7_P3_D080",
    "K7_P1_D150_S003",
    "K7_P1_D110_S002",
    "K7_P1_D060",
    "K7_PVC_D070",
    "K7_BG_L",
)
"""Section 4-3 table order; the position is the seed model index m (0-7)."""
ROLE_SETS = {
    "defect_a": ("K7_P2_D070", "K7_P2_D070_L"),
    "defect_b": ("K7_P3_D080",),
    "control": ("K7_P1_D060", "K7_P1_D150_S003", "K7_P1_D110_S002", "K7_PVC_D070"),
    "pvc": ("K7_PVC_D070",),
    "bg": ("K7_BG_L",),
    "c_diag": ("K7_P1_D150_S003", "K7_P1_D110_S002"),
}
"""Section 4-3 role sets (asserted against the json ``role_sets`` at start)."""
CALIBRATION_PAIRS = {"K7_BG_L": "K7_P2_D070_L"}
"""Pipe-less line -> model whose noise calibration it uses (section 4-4)."""
VARIANTS = ("cur", "a1", "a1b2", "a1b1")
RHOS = (10, 20)
N_SEEDS = 40
SEED_BASE = 20301112
UNIT_NOISE_SEED = 20301199
MC_SEED = 20261112
MC_RHOS = (10, 20)
MC_DEPTHS_M = (0.6, 1.0, 1.5)
MC_N_REALISATIONS = 200
"""Section 1-3 Monte Carlo of m210_k7_probe.py: 20261112 and
20261113 + n + 1000 rho + floor(100 d), n 0-199 (asserted disjoint)."""
DT_HARNESS_NS = floor_harness.DT_HARNESS_NS
DEPTH_DIGITS = 5

KIND_FLAT_LAYER = "flat_layer"
K7_EVENT_KINDS = (*s1g.EVENT_KINDS, KIND_FLAT_LAYER, "multiple_layer", "pvc_internal")
"""Event kinds of models_k7.json (``expected_events_meta.kinds``): the S1g kinds
plus the boundary and PVC events of the K7 models."""

LOST_NMS = "nms"
LOST_REASONS = {
    floor_harness.LOST_EARLY_LOBE: LOST_NMS,
    floor_harness.LOST_OTHER_NMS: LOST_NMS,
    floor_harness.LOST_CAP6: "cap6",
    floor_harness.LOST_NOT_ACCEPTED: "not_accepted",
    floor_harness.LOST_NO_PEAK: "no_peak",
}
"""(b) harness loss reasons -> section 4-4 names (both NMS kinds are ``nms``)."""

K7_SEALED_FILES = (
    SPEC_DOC,
    "analysis/processing/m210_k7_probe.py",
    "analysis/processing/m210_k7_rank_replay.py",
    "docs/processing/m210_k7_probe.json",
    "docs/processing/m210_k7_rank_replay.json",
    "analysis/synthetic/k7/models_k7.json",
    GENERATOR_PATH,
    HARNESS_PATH,
    "analysis/processing/m210_k7_judge.py",
    "scripts/synthetic/export_dataset.py",
    "tests/test_m210_k7_generator.py",
    "tests/test_m210_k7_pipeline.py",
    "tests/test_m210_k7_judge.py",
    "tests/test_synthetic_export.py",
    "tests/test_m210b_s4s6_pipeline.py",
    "analysis/processing/m210b_floor_rules.py",
    "analysis/processing/m210b_s4s6_pipeline.py",
    "analysis/processing/m210_s1g_pipeline.py",
    "scripts/eval_depth_4tu.py",
    "docs/processing/m29c_qc_constants.json",
)
"""Section 6 sealed files checked against ``--seal-commit`` (the judgment asserts
``run_info.sealed_files`` covers this list). The EM folder ``K7_SEALED_FOLDERS``
(models json, inputs, pilot inputs, run ``.cmd``) is sealed as a whole."""
K7_SEALED_FOLDERS = ("analysis/synthetic/k7",)
PYTEST_OUTCOMES = (
    "passed",
    "failed",
    "skipped",
    "errors",
    "xfailed",
    "xpassed",
    "deselected",
)
K7_TEST_FILES = (
    "tests/test_m210_k7_generator.py",
    "tests/test_m210_k7_pipeline.py",
    "tests/test_m210_k7_judge.py",
)
"""Pure-function, harness and judgment tests run at start (rank 1 if failing)."""

S1G_DATA_ROOT = PROJECT_ROOT / "data" / "synthetic_gprmax_s1g"
S1G_STORED_CANDIDATES = (
    PROJECT_ROOT / "data" / "processing" / "m210_s1g" / "candidates.csv"
)
V3_DATA_ROOT = floor_harness.V3_DEFAULT_DATA_ROOT
V3_STORED_RUNS = floor_harness.V3_DEFAULT_OUT_DIR / floor_harness.V3_RUNS_CSV
V3_REGRESSION_MODEL = "S4N_D14_S038"
V3_REGRESSION_SEEDS = 2
V3_NO_PIPE_CALIBRATION_PAIR = "D14_S038"
V3_KEY_COLUMNS = ("model_id", "rho", "seed", "rule", "delta_ns", "ramp_noise", "dt_ns")

RUN_COLUMNS = (
    "model_id",
    "model_index",
    "rho",
    "seed",
    "seed_index",
    "noise_free",
    "variant",
    "pipe_index",
    "n_pipes",
    "apex_trace",
    "t_a0_ns",
    "pipe_top_depth_m",
    "sigma_raw",
    "t_zero_ns",
    "zero_rule",
    "accepted_apex",
    "peak_on_crest",
    "kept_apex",
    "kept_apex_trace",
    "kept_apex_sample",
    "lost_reason",
    "lost_reason_detail",
    "qc_primary",
    "depth_top_m",
    "depth_error_m",
    "n_qc_primary_pipe",
    "n_peaks",
    "n_window_starts",
    "n_accepted",
    "n_kept",
    "n_qc_passing",
    "n_qc_unexplained",
    "c_item",
    "c_end_offset_traces",
    "c_start_is_local_max",
    "c_end_is_local_max",
    "c_tau2_vertex_offset_traces",
)
COLUMN_DEFINITIONS = {
    "model_index": "m of the seed formula (section 4-3 order, 0-7)",
    "rho": "nominal rho (10, 20); blank on noise-free rows",
    "seed": "20301112 + 1000 m + 100 q + s; blank on noise-free rows",
    "seed_index": "s (0-39); blank on noise-free rows",
    "noise_free": "True for the one noise-free row per model, variant and pipe",
    "variant": "generator variant (cur, a1, a1b2, a1b1)",
    "pipe_index": "json pipe_index (pipes sorted by x); blank on K7_BG_L",
    "n_pipes": "json n_pipes of the export",
    "t_a0_ns": "noise-free t_a0 of the pipe (4 decimals; run_info unrounded)",
    "sigma_raw": "noise standard deviation added to the raw line (0 noise-free)",
    "accepted_apex": "an accepted candidate in the window (trace +-3, t_a0 +-0.5/f_c)",
    "kept_apex": "some kept candidate in the apex window",
    "kept_apex_trace": "first kept (rank order) in the window",
    "lost_reason": "no_peak / not_accepted / cap6 / nms when not kept; else blank",
    "lost_reason_detail": "(b) harness reason (early_lobe_nms / other_nms split)",
    "qc_primary": "a QC-passing candidate attached to this pipe's primary event",
    "depth_top_m": "depth of that candidate (nearest event time, |dx|, then order)",
    "depth_error_m": "depth_top_m - json pipe_top_depth_m",
    "n_qc_primary_pipe": "QC-passing candidates attached to this pipe's primary",
    "n_window_starts": "detection peaks whose crest-moved point is in the window",
    "n_qc_unexplained": "QC-passing candidates attached to no expected event (U)",
    "c_item": "kept / accepted / blank: the window item of the c columns",
    "c_end_offset_traces": "item trace - apex trace (signed)",
    "c_start_is_local_max": "the crest-walk start of that item is an E time local max",
    "c_end_is_local_max": "the item end point is an E time local max",
    "c_tau2_vertex_offset_traces": "tau^2 vertex trace - apex trace; blank on fallback",
}


# ----------------------------------------------------------------- plan and seeds


@dataclass(frozen=True)
class K7Plan:
    """Conditions of one run; tests replace it with a smaller one."""

    model_ids: tuple[str, ...] = MODEL_IDS
    rhos: tuple[int, ...] = RHOS
    n_seeds: int = N_SEEDS
    variants: tuple[str, ...] = VARIANTS
    calibration_pairs: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: dict(CALIBRATION_PAIRS)
    )


DEFAULT_PLAN = K7Plan()


def noise_seed(model_index: int, rho_index: int, seed_index: int) -> int:
    """``20301112 + 1000 m + 100 q + s`` (section 4-4)."""
    return SEED_BASE + 1000 * model_index + 100 * rho_index + seed_index


def k7_seeds(plan: K7Plan = DEFAULT_PLAN) -> set[int]:
    """Every noise seed of the plan (full model list) and the unit-noise seed."""
    seeds = {
        noise_seed(m, q, s)
        for m in range(len(MODEL_IDS))
        for q in (RHOS.index(rho) for rho in plan.rhos)
        for s in range(plan.n_seeds)
    }
    return seeds | {UNIT_NOISE_SEED}


def monte_carlo_seeds() -> frozenset[int]:
    """Seeds of the section 1-3 Monte Carlo (``m210_k7_probe.py`` part C)."""
    seeds = {MC_SEED}
    for depth_m in MC_DEPTHS_M:
        for rho in MC_RHOS:
            for n in range(MC_N_REALISATIONS):
                seeds.add(MC_SEED + 1 + n + 1000 * rho + int(100 * depth_m))
    return frozenset(seeds)


def earlier_seed_sets() -> dict[str, frozenset[int]]:
    """Seed sets the K7 seeds must not touch (section 4-4)."""
    return {
        "batch1": floor_harness.BATCH1_SEEDS,
        "v2": floor_harness.V2_SEEDS,
        "v3": frozenset(floor_harness.plan_seeds(floor_harness.V3_PLAN)),
        "prototype": floor_harness.PROTOTYPE_SEED_SET,
        "k7_monte_carlo": monte_carlo_seeds(),
    }


def seed_record(plan: K7Plan = DEFAULT_PLAN) -> dict[str, object]:
    """Assert that no K7 seed was used before; ``seed_overlap`` = 0.

    Raises:
        floor_harness.SeedOverlapError: A shared seed.
    """
    seeds = k7_seeds(plan)
    earlier = earlier_seed_sets()
    overlap = floor_harness.assert_no_seed_overlap(
        seeds, set(frozenset().union(*earlier.values()))
    )
    return {
        "seed_overlap": overlap,
        "asserted_against": {name: len(values) for name, values in earlier.items()},
        "seed_formula": f"{SEED_BASE} + 1000 m + 100 q + s",
        "model_index_m": {model_id: m for m, model_id in enumerate(MODEL_IDS)},
        "rho_index_q": {str(rho): RHOS.index(rho) for rho in plan.rhos},
        "n_seeds_per_condition": plan.n_seeds,
        "unit_noise_seed": UNIT_NOISE_SEED,
        "n_k7_seeds": len(seeds),
        "k7_seed_range": [min(seeds), max(seeds)],
    }


# ----------------------------------------------------------------- noise-free model


@dataclass(frozen=True)
class PipeTarget:
    """One pipe of a model, scored on the noise-free processing (section 4-4)."""

    pipe_index: int
    target: floor_harness.ApexTarget
    top_depth_m: float
    a0: float
    """Envelope maximum over apex trace +-3 and ``t_a0`` +-0.5/f_c."""


@dataclass(frozen=True)
class CleanModel:
    """Noise-free resampled line of one model and its pipe targets."""

    model_id: str
    radargram: Radargram
    t_zero_ns: float | None
    zero_rule: str | None
    pipes: tuple[PipeTarget, ...]

    @property
    def a0(self) -> float | None:
        """Smallest pipe ``A0`` of the model (``None`` without a pipe)."""
        return min((pipe.a0 for pipe in self.pipes), default=None)


def clean_model(
    model_id: str, radargram: Radargram, truths: Sequence[SyntheticTruth]
) -> CleanModel:
    """Noise-free processing of a resampled line: zero, ``t_a0`` and ``A0`` per pipe."""
    inputs = prepare_depth_inputs(radargram, DepthConfig())
    envelope = pick_signal(inputs.branch.background_removed, "envelope")
    dt, f_c = radargram.dt_ns, radargram.f_c_hz
    pipes = []
    for pipe_index, truth in enumerate(truths):
        apex_trace = truth.apex_trace_index
        t_a0, _ = floor_harness.apex_time_and_early_lobe(
            envelope, apex_trace, truth.apex_envelope_peak_ns, dt, f_c
        )
        a0 = floor_harness.apex_amplitude(envelope, apex_trace, t_a0, dt, f_c)
        target = floor_harness.ApexTarget(apex_trace, t_a0, f_c)
        pipes.append(PipeTarget(pipe_index, target, truth.pipe_top_depth_m, a0))
    zero_rule = inputs.zero.zero_rule
    return CleanModel(
        model_id,
        radargram,
        inputs.zero.t_zero_ns,
        None if zero_rule is None else str(zero_rule),
        tuple(pipes),
    )


@dataclass(frozen=True)
class Calibration:
    """Noise level of one model (section 4-4)."""

    source_model: str
    a0: float
    sigma_u: float
    sigma_raw_by_rho: dict[int, float]


def calibrate(model: CleanModel, rhos: Sequence[int]) -> Calibration:
    """``sigma_raw = A0 / (rho sigma_u)`` from the model's own noise-free line.

    Raises:
        ValueError: A model without a pipe or without a zero.
    """
    if model.a0 is None or model.t_zero_ns is None:
        raise ValueError(f"{model.model_id}: no pipe or no zero to calibrate")
    k_lo, k_hi = floor_rules.search_rows(
        model.t_zero_ns,
        model.radargram.f_c_hz,
        model.radargram.dt_ns,
        model.radargram.n_samples,
    )
    sigma_u = floor_harness.unit_noise_sigma(
        model.radargram, k_lo, k_hi, UNIT_NOISE_SEED
    )
    return Calibration(
        source_model=model.model_id,
        a0=model.a0,
        sigma_u=sigma_u,
        sigma_raw_by_rho={rho: model.a0 / (rho * sigma_u) for rho in rhos},
    )


# ----------------------------------------------------------------- one run


@dataclass(frozen=True)
class K7Event:
    """One ``expected_events`` entry of ``models_k7.json`` (EM, before the runs).

    The S1g schema plus the K7 kinds: ``flat_layer`` has ``x_m`` ``None`` (the
    event lies on every trace), ``multiple_layer`` and ``pvc_internal`` are
    point events; ``ambiguous_with`` lists event ids with nearly the same curve.
    """

    event_id: str
    kind: str
    x_m: float | None
    t_ns: float
    pipe_index: int | None = None
    ambiguous_with: tuple[str, ...] = ()


def parse_k7_events(entries: Sequence[Mapping[str, object]]) -> list[K7Event]:
    """``expected_events[<id>]`` of ``models_k7.json``, json order kept.

    Raises:
        ValueError: An unknown kind, or ``x_m`` null on a kind other than
            ``flat_layer``.
    """
    events = []
    for entry in entries:
        kind = str(entry["kind"])
        if kind not in K7_EVENT_KINDS:
            raise ValueError(f"{entry.get('event_id')}: unknown event kind {kind!r}")
        x_m = entry.get("x_m")
        if x_m is None and kind != KIND_FLAT_LAYER:
            raise ValueError(f"{entry.get('event_id')}: x_m null on a {kind} event")
        pipe_index = entry.get("pipe_index")
        events.append(
            K7Event(
                event_id=str(entry["event_id"]),
                kind=kind,
                x_m=None if x_m is None else float(x_m),
                t_ns=float(entry["t_ns"]),
                pipe_index=None if pipe_index is None else int(pipe_index),
                ambiguous_with=tuple(entry.get("ambiguous_with") or ()),
            )
        )
    return events


def attach_k7_event(
    apex_x_m: float, apex_t_ns: float, events: Sequence[K7Event], f_c_hz: float
) -> K7Event | None:
    """The expected event of a fitted apex, or ``None`` (unexplained, U).

    The S1g rule (``m210_s1g_pipeline.attach_event``): ``|dx| <= 0.10 m`` and
    ``|dt| <= 0.5 / f_c``, a ``primary`` first, then the nearest time, the
    smaller ``|dx|`` and the earlier json entry. A ``flat_layer`` event (``x_m``
    null) is on every trace, so its ``|dx|`` is 0.
    """
    t_limit_ns = s1g.ATTACH_PERIODS * 1e9 / f_c_hz
    best_key, best_event = None, None
    for order, event in enumerate(events):
        dx_m = 0.0 if event.x_m is None else abs(apex_x_m - event.x_m)
        dt_ns = abs(apex_t_ns - event.t_ns)
        is_far = dx_m > s1g.ATTACH_X_M + s1g.ATTACH_TOL
        if is_far or dt_ns > t_limit_ns + s1g.ATTACH_TOL:
            continue
        key = (event.kind != s1g.KIND_PRIMARY, dt_ns, dx_m, order)
        if best_key is None or key < best_key:
            best_key, best_event = key, event
    return best_event


@dataclass(frozen=True)
class AttachedPrimary:
    """A QC-passing candidate attached to a pipe's primary event."""

    depth_top_m: float
    sort_key: tuple[float, float, int]


@dataclass(frozen=True)
class QcSummary:
    """QC-passing candidates of one run, by pipe (section 4-4)."""

    n_passing: int
    n_unexplained: int
    primaries: dict[int, list[AttachedPrimary]]


def qc_summary(
    radargram: Radargram,
    kept: Sequence[Mapping],
    events: Sequence[K7Event],
) -> QcSummary:
    """Depth and QC of the kept candidates, attached like the S1g harness."""
    candidates = depth_evaluation.run_line(
        radargram, s1g._generator_table(kept), apex_qc=True
    )
    axes = s1g._line_axes(radargram)
    passing = [c for c in candidates if s1g._passes_qc(c)]
    n_unexplained = 0
    primaries: dict[int, list[AttachedPrimary]] = {}
    for order, candidate in enumerate(passing):
        apex_x = axes.x_m(candidate.result.apex_trace)
        apex_t = axes.t_ns(candidate.result.apex_sample)
        event = attach_k7_event(apex_x, apex_t, events, radargram.f_c_hz)
        if event is None:
            n_unexplained += 1
            continue
        # pvc_internal also names its pipe; only the crown (primary) is the truth.
        if event.kind != s1g.KIND_PRIMARY or event.pipe_index is None:
            continue
        sort_key = (abs(apex_t - event.t_ns), abs(apex_x - event.x_m), order)
        attached = AttachedPrimary(float(candidate.result.depth_top_m), sort_key)
        primaries.setdefault(event.pipe_index, []).append(attached)
    return QcSummary(len(passing), n_unexplained, primaries)


def _peak_of_item(peaks: Sequence[Mapping], item: Mapping) -> Mapping | None:
    """First accepted detection peak (detection order) whose end point is ``item``."""
    for peak in peaks:
        is_same_end = (
            peak["trace"] == item["trace"]
            and round(peak["sample"], 2) == item["sample"]
        )
        if peak["status"] == "accepted" and is_same_end:
            return peak
    return None


def c_diagnostics(
    line: Mapping, kept: Sequence[Mapping], target: floor_harness.ApexTarget
) -> dict[str, object]:
    """Section 4-4 c columns: the window kept item, else the window accepted item."""
    blank = {
        "c_item": "",
        "c_end_offset_traces": "",
        "c_start_is_local_max": "",
        "c_end_is_local_max": "",
        "c_tau2_vertex_offset_traces": "",
    }
    window_kept = floor_harness._window_members(kept, target)
    accepted = line.get("_accepted_all", [])
    window_accepted = floor_harness._window_members(accepted, target)
    if window_kept:
        source, item = "kept", kept[window_kept[0]]
    elif window_accepted:
        source, item = "accepted", accepted[window_accepted[0]]
    else:
        return blank
    peak = _peak_of_item(line.get("_peaks", []), item)
    vertex = None if peak is None else peak["tau2_vertex_trace"]
    return {
        "c_item": source,
        "c_end_offset_traces": item["trace"] - target.trace,
        "c_start_is_local_max": "" if peak is None else peak["start_is_local_max"],
        "c_end_is_local_max": "" if peak is None else peak["end_is_local_max"],
        "c_tau2_vertex_offset_traces": "" if vertex is None else vertex - target.trace,
    }


def _n_window_starts(line: Mapping, target, dt_ns: float) -> int:
    return sum(
        floor_harness.is_in_apex_window(
            peak["trace"], peak["sample"] * dt_ns, target.trace, target.t_a0_ns,
            target.f_c_hz,
        )
        for peak in line.get("_peaks", [])
    )  # fmt: skip


def _pipe_fields(
    line: Mapping,
    kept: Sequence[Mapping],
    pipe: PipeTarget,
    qc: QcSummary,
    dt_ns: float,
) -> dict[str, object]:
    outcome = floor_harness.apex_outcome(line, kept, dt_ns, pipe.target)
    attached = sorted(qc.primaries.get(pipe.pipe_index, []), key=lambda a: a.sort_key)
    fields: dict[str, object] = {
        "pipe_index": pipe.pipe_index,
        "apex_trace": pipe.target.trace,
        "t_a0_ns": round(pipe.target.t_a0_ns, 4),
        "pipe_top_depth_m": pipe.top_depth_m,
        "accepted_apex": outcome.accepted_apex,
        "peak_on_crest": outcome.peak_on_crest,
        "kept_apex": outcome.kept_apex,
        "kept_apex_trace": "" if outcome.kept_trace is None else outcome.kept_trace,
        "kept_apex_sample": "" if outcome.kept_sample is None else outcome.kept_sample,
        "lost_reason": LOST_REASONS.get(outcome.lost_reason, outcome.lost_reason),
        "lost_reason_detail": outcome.lost_reason,
        "qc_primary": bool(attached),
        "depth_top_m": "",
        "depth_error_m": "",
        "n_qc_primary_pipe": len(attached),
        "n_window_starts": _n_window_starts(line, pipe.target, dt_ns),
    }
    if attached:
        depth = attached[0].depth_top_m
        fields["depth_top_m"] = round(depth, DEPTH_DIGITS)
        fields["depth_error_m"] = round(depth - pipe.top_depth_m, DEPTH_DIGITS)
    fields.update(c_diagnostics(line, kept, pipe.target))
    return fields


def _blank_pipe_fields() -> dict[str, object]:
    names = (
        "pipe_index", "apex_trace", "t_a0_ns", "pipe_top_depth_m", "accepted_apex",
        "peak_on_crest", "kept_apex", "kept_apex_trace", "kept_apex_sample",
        "lost_reason", "lost_reason_detail", "qc_primary", "depth_top_m",
        "depth_error_m", "n_qc_primary_pipe", "n_window_starts", "c_item",
        "c_end_offset_traces", "c_start_is_local_max", "c_end_is_local_max",
        "c_tau2_vertex_offset_traces",
    )  # fmt: skip
    return dict.fromkeys(names, "")


@dataclass(frozen=True)
class RunKey:
    """Identity of one noisy (or noise-free) line of a model."""

    model_id: str
    model_index: int
    rho: int | None
    rho_index: int | None
    seed_index: int | None
    sigma_raw: float

    @property
    def seed(self) -> int | None:
        if self.rho_index is None or self.seed_index is None:
            return None
        return noise_seed(self.model_index, self.rho_index, self.seed_index)


def variant_rows(
    key: RunKey,
    noisy: Radargram,
    inputs,
    model: CleanModel,
    variant: str,
    events: Sequence[K7Event],
) -> list[dict[str, object]]:
    """Rows of one variant on one line: one per pipe, one for a pipe-less line."""
    start_rule, rank_rule, per_line_cap = generator.GENERATOR_VARIANTS[variant]
    line, kept, _ = generator.apex_candidates_for_radargram(
        noisy, inputs, act=LINE_ACTIVITY, path=model.model_id,
        start_rule=start_rule, rank_rule=rank_rule, per_line_cap=per_line_cap,
    )  # fmt: skip
    qc = qc_summary(noisy, kept, events)
    zero_rule = inputs.zero.zero_rule
    common = {
        "model_id": key.model_id,
        "model_index": key.model_index,
        "rho": "" if key.rho is None else key.rho,
        "seed": "" if key.seed is None else key.seed,
        "seed_index": "" if key.seed_index is None else key.seed_index,
        "noise_free": key.rho is None,
        "variant": variant,
        "n_pipes": len(model.pipes),
        "sigma_raw": round(key.sigma_raw, 10),
        "t_zero_ns": floor_harness._blank_or_rounded(inputs.zero.t_zero_ns, 4),
        "zero_rule": "" if zero_rule is None else str(zero_rule),
        "n_peaks": line.get("n_peaks", 0),
        "n_accepted": len(line.get("_accepted_all", [])),
        "n_kept": len(kept),
        "n_qc_passing": qc.n_passing,
        "n_qc_unexplained": qc.n_unexplained,
    }
    if not model.pipes:
        rows = [{**common, **_blank_pipe_fields()}]
    else:
        rows = [
            {**common, **_pipe_fields(line, kept, pipe, qc, noisy.dt_ns)}
            for pipe in model.pipes
        ]
    return [{column: row[column] for column in RUN_COLUMNS} for row in rows]


def line_rows(
    key: RunKey,
    noisy: Radargram,
    model: CleanModel,
    variants: Sequence[str],
    events: Sequence[K7Event],
) -> list[dict[str, object]]:
    """Process one line once and run every variant on it."""
    inputs = prepare_depth_inputs(noisy, DepthConfig())
    rows: list[dict[str, object]] = []
    for variant in variants:
        rows += variant_rows(key, noisy, inputs, model, variant, events)
    return rows


def model_rows(
    model: CleanModel,
    calibration: Calibration,
    plan: K7Plan,
    events: Sequence[K7Event],
) -> list[dict[str, object]]:
    """Noise-free rows, then rho x seed rows of one model (all variants)."""
    model_index = MODEL_IDS.index(model.model_id)
    clean_key = RunKey(model.model_id, model_index, None, None, None, 0.0)
    rows = line_rows(clean_key, model.radargram, model, plan.variants, events)
    shape = model.radargram.data.shape
    for rho in plan.rhos:
        # q is the position in the protocol rho tuple, also for a reduced plan.
        rho_index = RHOS.index(rho)
        sigma_raw = calibration.sigma_raw_by_rho[rho]
        for seed_index in range(plan.n_seeds):
            key = RunKey(
                model.model_id, model_index, rho, rho_index, seed_index, sigma_raw
            )
            noise = sigma_raw * floor_harness.white_noise(shape, key.seed)
            noisy = floor_harness.with_noise(model.radargram, noise)
            rows += line_rows(key, noisy, model, plan.variants, events)
    return rows


# ----------------------------------------------------------------- models


ModelLoader = Callable[[str, Path], tuple[Radargram, list[SyntheticTruth]]]


def load_model(
    model_id: str, data_root: Path
) -> tuple[Radargram, list[SyntheticTruth]]:
    """Exported K7 line (``line_id`` ``K7/<id>``) and its pipe truths."""
    radargram = load_synthetic(
        model_id,
        root=data_root,
        include_eps=True,
        include_radius=False,
        include_height=True,
    )
    radargram = dataclasses.replace(radargram, line_id=f"{LINE_ACTIVITY}/{model_id}")
    return radargram, load_synthetic_pipe_truths(model_id, root=data_root)


def json_models(models_json: Path) -> dict[str, object]:
    """Model ids (json ``batch == "k7"``), role sets and expected events.

    Raises:
        SystemExit: A json without ``expected_events`` for some model.
    """
    meta = json.loads(models_json.read_text(encoding="utf-8"))
    models = [m for m in meta["models"] if m.get("batch") == JSON_BATCH]
    ids = [m["id"] for m in models]
    events = meta.get("expected_events") or {}
    missing = [model_id for model_id in ids if model_id not in events]
    if missing:
        raise SystemExit(f"{models_json}: no expected_events for {missing}")
    # The json carries the events twice (top-level map and per model); the run
    # reads the top-level map, and a per-model copy must not disagree with it.
    differing_copies = [
        m["id"]
        for m in models
        if "expected_events" in m and m["expected_events"] != events[m["id"]]
    ]
    return {
        "model_ids": ids,
        "role_sets": {
            name: list(values)
            for name, values in (meta.get("role_sets") or {}).items()
            if name != "source"
        },
        "n_pipes": {m["id"]: len(m["pipes"]) for m in meta["models"]},
        "events": {model_id: parse_k7_events(events[model_id]) for model_id in ids},
        "event_copies_differing": differing_copies,
    }


def model_set_failures(listing: Mapping, plan: K7Plan) -> list[str]:
    """The json model list and role sets must be the fixed sets (section 4-6)."""
    failures = []
    if sorted(listing["model_ids"]) != sorted(plan.model_ids):
        failures.append(
            f"json k7 models {sorted(listing['model_ids'])} differ from the plan "
            f"{sorted(plan.model_ids)}"
        )
    expected_roles = {name: sorted(values) for name, values in ROLE_SETS.items()}
    json_roles = {name: sorted(values) for name, values in listing["role_sets"].items()}
    if plan.model_ids == MODEL_IDS and json_roles != expected_roles:
        failures.append(f"json role_sets {json_roles} differ from {expected_roles}")
    if listing.get("event_copies_differing"):
        failures.append(
            "per-model expected_events differ from the top-level map: "
            f"{listing['event_copies_differing']}"
        )
    return failures


# ----------------------------------------------------------------- run record


def _git(*args: str) -> str | None:
    return floor_harness._git(*args)


def seal_record(seal: floor_harness.SealSpec) -> dict[str, object]:
    """Seal and working-tree state; failed git queries are listed, not hidden."""
    queries = {
        "seal_commit_full": ("rev-parse", seal.commit),
        "seal_commit_time": ("show", "-s", "--format=%cI", seal.commit),
        "head_commit": ("rev-parse", "HEAD"),
        "harness_blob_working_tree": ("hash-object", "--", HARNESS_PATH),
        "generator_blob_working_tree": ("hash-object", "--", GENERATOR_PATH),
        "status": ("status", "--porcelain"),
    }
    answers = {name: _git(*args) for name, args in queries.items()}
    # A clean tree prints "" for ``git status --porcelain``: only None is a failure.
    git_failures = [name for name, answer in answers.items() if answer is None]
    status = answers.pop("status")
    return {
        "seal_commit": seal.commit,
        **answers,
        "sealed_files": floor_harness.sealed_blobs(seal),
        "dirty_files": [] if status is None else status.splitlines(),
        "git_failures": git_failures,
        "taken_at": _now(),
    }


def environment_record() -> dict[str, object]:
    """Interpreter, library versions, thread variables, QC constants, DepthConfig."""
    return {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "skimage": skimage.__version__,
        "thread_variables": {
            name: os.environ.get(name) for name in floor_harness.THREAD_VARIABLES
        },
        "qc_constants": s1g.qc_constants_record(),
        "depth_config": s1g.depth_config_record(),
    }


def run_k7_tests() -> dict[str, object]:
    """Run the K7 tests (``K7_TEST_FILES``) in a child pytest; rank 1 if failing."""
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        *K7_TEST_FILES,
    ]  # fmt: skip
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=environment, check=False,
    )  # fmt: skip
    output_lines = completed.stdout.strip().splitlines()
    summary = output_lines[-1] if output_lines else ""
    return {
        "command": " ".join(command[1:]),
        "exit_code": completed.returncode,
        "summary": summary,
        "counts": pytest_counts(summary),
        "pass": completed.returncode == 0,
    }


def pytest_counts(summary: str) -> dict[str, int]:
    """Outcome counts of a pytest summary line, e.g. ``{"passed": 86, "skipped": 1}``.

    Recorded for the judgment header only; the pass rule stays exit code 0.
    """
    counts = dict.fromkeys(PYTEST_OUTCOMES, 0)
    for number, outcome in re.findall(r"(\d+) ([a-z]+)", summary):
        name = "errors" if outcome == "error" else outcome
        if name in counts:
            counts[name] += int(number)
    return counts


def s1g_regression(out_dir: Path) -> dict[str, object]:
    """G0: the S1g harness default path rewrites the stored ``candidates.csv``."""
    if not S1G_STORED_CANDIDATES.exists():
        return {"run": False, "reason": f"{S1G_STORED_CANDIDATES} missing"}
    models = json.loads(s1g.DEFAULT_MODELS_JSON.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for model_id in s1g.S1G_MODELS:
        events = s1g.parse_events(models["expected_events"][model_id])
        line = s1g.load_line(model_id, S1G_DATA_ROOT)
        rows += s1g.run_model(model_id, line, events)[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    regenerated = out_dir / "candidates.csv"
    s1g.write_candidates(regenerated, rows)
    stored_rows, new_rows = _csv_rows(S1G_STORED_CANDIDATES), _csv_rows(regenerated)
    # The stored file is the S1g judgment run (90845b4), written before K1 was
    # removed: it has one more column, cluster_mu. Every current column must agree.
    columns = list(s1g.CANDIDATE_COLUMNS)
    stored_columns = list(stored_rows[0]) if stored_rows else []
    stored_values = [[row[c] for c in columns] for row in stored_rows]
    new_values = [[row[c] for c in columns] for row in new_rows]
    return {
        "run": True,
        "stored_sha256": floor_harness._sha256(S1G_STORED_CANDIDATES),
        "regenerated_sha256": floor_harness._sha256(regenerated),
        "stored_only_columns": [c for c in stored_columns if c not in columns],
        "n_rows": len(new_rows),
        "identical": bool(new_rows)
        and set(columns) <= set(stored_columns)
        and stored_values == new_values,
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def v3_regression(out_dir: Path) -> dict[str, object]:
    """G0: the (b) harness default path on a v3 subset (1 model, 2 seeds)."""
    if not (V3_STORED_RUNS.exists() and V3_DATA_ROOT.exists()):
        return {"run": False, "reason": "v3 runs_v3.csv or exports missing"}
    plan = dataclasses.replace(
        floor_harness.V3_PLAN,
        n_seeds=V3_REGRESSION_SEEDS,
        no_pipe_calibration_pair=V3_NO_PIPE_CALIBRATION_PAIR,
    )
    models = floor_harness.ModelSet(V3_DATA_ROOT, plan)
    rows = floor_harness.model_runs(V3_REGRESSION_MODEL, models)
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = out_dir / "subset_v3.csv"
    floor_harness.write_runs(subset, rows, plan.columns)
    stored = {
        tuple(row[c] for c in V3_KEY_COLUMNS): row for row in _csv_rows(V3_STORED_RUNS)
    }
    regenerated = _csv_rows(subset)
    differing = [
        tuple(row[c] for c in V3_KEY_COLUMNS)
        for row in regenerated
        if stored.get(tuple(row[c] for c in V3_KEY_COLUMNS)) != row
    ]
    return {
        "run": True,
        "model": V3_REGRESSION_MODEL,
        "n_seeds": V3_REGRESSION_SEEDS,
        "n_rows": len(regenerated),
        "n_rows_differing": len(differing),
        "first_differing": [list(key) for key in differing[:5]],
        "identical": bool(regenerated) and not differing,
    }


def g0_record(out_dir: Path, activities: str) -> dict[str, object]:
    """Section 4-2 G0: 4TU csvs, the S1g and the (b) v3 default paths."""
    if activities == "none":
        return {"run": False}
    return {
        "run": True,
        "fourtu": floor_harness.wrapper_regression(
            out_dir / "wrapper_regression", activities
        ),
        "s1g": s1g_regression(out_dir / "s1g_regression"),
        "m210b_v3": v3_regression(out_dir / "m210b_v3_regression"),
    }


def g0_failures(record: Mapping) -> list[str]:
    """Every G0 part must have run and be identical."""
    if not record.get("run"):
        return ["G0 regression not run"]
    failures = []
    fourtu = record["fourtu"]
    csv_records = [r for r in fourtu.values() if isinstance(r, dict)]
    if len(csv_records) != 2 or not all(r["byte_identical"] for r in csv_records):
        failures.append("G0 4TU csvs not byte identical")
    if not record["s1g"].get("identical"):
        failures.append(f"G0 S1g candidates.csv: {record['s1g']}")
    if not record["m210b_v3"].get("identical"):
        failures.append(f"G0 (b) v3 subset: {record['m210b_v3']}")
    return failures


# ----------------------------------------------------------------- entry point


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def write_runs(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RUN_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", choices=(PROTOCOL,), required=True)
    parser.add_argument("--seal-commit", required=True, help="the K7 seal commit")
    parser.add_argument(
        "--data-root", "--input-dir", dest="data_root", type=Path,
        default=DEFAULT_DATA_ROOT,
    )  # fmt: skip
    parser.add_argument("--models-json", type=Path, default=DEFAULT_MODELS_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--regression-activities",
        default="all",
        help="G0: 4TU activities (all, none or a comma list); none skips all of G0",
    )
    return parser.parse_args(argv)


def write_start_failure(
    out_dir: Path, start: Mapping, inputs: Mapping, failures: Sequence[str],
    tests: Mapping | None,
) -> None:  # fmt: skip
    """run_info of a run stopped by the start guards (no model was run)."""
    run_info = {
        "protocol": PROTOCOL,
        "started_at": start["taken_at"],
        "seal": start,
        "sealed_files": sorted(start["sealed_files"]),
        "inputs": dict(inputs),
        "tests": tests,
        "environment": environment_record(),
        "guards": {"pass": False, "stage": "start", "failures": list(failures)},
        "n_runs": 0,
    }
    (out_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=1, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    for failure in failures:
        print(f"START GUARD FAILED: {failure}", file=sys.stderr, flush=True)


def calibration_record(calibration: Calibration) -> dict[str, object]:
    return {
        "source_model": calibration.source_model,
        "a0": calibration.a0,
        "sigma_u": calibration.sigma_u,
        "sigma_raw_by_rho": {
            str(k): v for k, v in calibration.sigma_raw_by_rho.items()
        },
    }


def model_record(
    model: CleanModel, calibration: Calibration, n_pipes_json: int
) -> dict:
    return {
        "line_id": model.radargram.line_id,
        "dt_ns": model.radargram.dt_ns,
        "n_samples": model.radargram.n_samples,
        "t_zero_ns": model.t_zero_ns,
        "zero_rule": model.zero_rule,
        "n_pipes": len(model.pipes),
        "n_pipes_json": n_pipes_json,
        "pipes": [
            {
                "pipe_index": pipe.pipe_index,
                "apex_trace": pipe.target.trace,
                "t_a0_ns": pipe.target.t_a0_ns,
                "a0": pipe.a0,
                "top_depth_m": pipe.top_depth_m,
            }
            for pipe in model.pipes
        ],
        "a0_min": model.a0,
        "calibration": calibration_record(calibration),
    }


def clean_models(
    plan: K7Plan, data_root: Path, loader: ModelLoader
) -> tuple[dict[str, CleanModel], dict[str, Calibration]]:
    """Noise-free models and their calibrations (pairs after their sources)."""
    models: dict[str, CleanModel] = {}
    for model_id in plan.model_ids:
        radargram, truths = loader(model_id, data_root)
        resampled = floor_harness.resample_to(radargram, DT_HARNESS_NS)
        models[model_id] = clean_model(model_id, resampled, truths)
    own = {
        model_id: calibrate(model, plan.rhos)
        for model_id, model in models.items()
        if model_id not in plan.calibration_pairs
    }
    calibrations = {
        model_id: own[plan.calibration_pairs.get(model_id, model_id)]
        for model_id in plan.model_ids
    }
    return models, calibrations


def main(
    argv: Sequence[str] | None = None,
    plan: K7Plan = DEFAULT_PLAN,
    loader: ModelLoader = load_model,
    run_tests: Callable[[], dict[str, object]] = run_k7_tests,
) -> int:
    """Run the harness; ``plan``, ``loader`` and ``run_tests`` are replaced by tests.

    Returns:
        0, or 1 when a run guard failed (outputs are written in both cases;
        a start-guard failure writes ``run_info.json`` only).
    """
    args = parse_args(argv)
    seeds = seed_record(plan)
    listing = json_models(args.models_json)
    seal = floor_harness.SealSpec(args.seal_commit, K7_SEALED_FILES, K7_SEALED_FOLDERS)
    start = seal_record(seal)
    print(f"m210_k7_pipeline start {start['taken_at']} protocol {PROTOCOL}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    floor_harness.write_identity_file(args.out_dir / "pre_run.txt", start)
    inputs = floor_harness.input_record(args.data_root, args.models_json)
    start_failures = floor_harness.start_guard_failures(start, head_must_be_seal=True)
    start_failures += model_set_failures(listing, plan)
    tests = None
    if not start_failures:
        tests = run_tests()
        if not tests["pass"]:
            start_failures.append(f"K7 tests failed: {tests['summary']}")
    if start_failures:
        write_start_failure(args.out_dir, start, inputs, start_failures, tests)
        return 1
    models, calibrations = clean_models(plan, args.data_root, loader)
    rows: list[dict[str, object]] = []
    records: dict[str, object] = {}
    started = time.perf_counter()
    for model_id in plan.model_ids:
        new_rows = model_rows(
            models[model_id], calibrations[model_id], plan, listing["events"][model_id]
        )
        rows += new_rows
        records[model_id] = model_record(
            models[model_id], calibrations[model_id], listing["n_pipes"][model_id]
        )
        elapsed = time.perf_counter() - started
        print(f"{model_id}: {len(new_rows)} rows, elapsed {elapsed:.0f} s", flush=True)
    runs_csv = args.out_dir / RUNS_CSV
    write_runs(runs_csv, rows)
    g0 = g0_record(args.out_dir, args.regression_activities)
    inputs_after = floor_harness.input_record(args.data_root, args.models_json)
    changed = sorted(
        name
        for name in set(inputs) | set(inputs_after)
        if inputs.get(name) != inputs_after.get(name)
    )
    end = seal_record(seal)
    floor_harness.write_identity_file(args.out_dir / "post_run.txt", end)
    failures = floor_harness.end_guard_failures(start, end, changed, {"run": False})
    failures += g0_failures(g0)
    run_info = {
        "spec": f"{SPEC_DOC} v1.1 sections 4-4 and 5; no verdict here",
        "protocol": PROTOCOL,
        "started_at": start["taken_at"],
        "seal": start,
        "seal_at_end": end,
        "sealed_files": sorted(start["sealed_files"]),
        "generator_blob": start["generator_blob_working_tree"],
        "seeds": seeds,
        "seed_overlap": seeds["seed_overlap"],
        "plan": {
            "model_ids": list(plan.model_ids),
            "rhos": list(plan.rhos),
            "n_seeds": plan.n_seeds,
            "variants": list(plan.variants),
            "variant_arguments": {
                name: list(generator.GENERATOR_VARIANTS[name]) for name in plan.variants
            },
            "calibration_pairs": dict(plan.calibration_pairs),
            "dt_harness_ns": DT_HARNESS_NS,
            "apex_window": "apex trace +-3, t_a0 +-0.5/f_c",
        },
        "json_role_sets": listing["role_sets"],
        "inputs": inputs,
        "inputs_changed_during_run": changed,
        "mkl_cbwr": os.environ.get("MKL_CBWR"),
        "environment": environment_record(),
        "tests": tests,
        "models": records,
        "columns": dict(COLUMN_DEFINITIONS),
        "n_runs": len(rows),
        "n_runs_without_zero": sum(1 for row in rows if row["t_zero_ns"] == ""),
        "runs_csv": RUNS_CSV,
        "runs_csv_sha256": floor_harness._sha256(runs_csv),
        "g0": g0,
        "guards": {"pass": not failures, "failures": failures},
        "finished_at": _now(),
    }
    (args.out_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=1, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"wrote {runs_csv} and run_info.json", flush=True)
    for failure in failures:
        print(f"GUARD FAILED: {failure}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
