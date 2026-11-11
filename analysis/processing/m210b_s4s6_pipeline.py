"""S4/S6 batch-1 harness (M2-10 (b), m210b_selector_floor.md v1.4 4-2, 8-3).

Runs the selector (generator wrapper ``apex_candidates_for_radargram``) on the
gprMax S4/S6 scan lines for every floor rule and mute shift and writes one row
per run. **No verdict is made here** (gpr-processing-expert computes M0, R and
F1-F6 from ``runs.csv``).

Steps (4-2 rules 1-5, P1-P4 superset):

1. ``load_synthetic(id, include_eps=True, include_radius=False,
   include_height=True)``, ``line_id`` ``S4S6/<id>``.
2. Resample every trace to ``dt`` 0.098 ns with ``np.interp`` (``resample_to``).
3. Noise-free calibration per depth/sigma pair from the S4N control:
   ``A0`` (apex envelope max over apex trace +-3 and the json expected time
   +-0.5/f_c), ``sigma_u`` (1.4826 MAD over the search rows of the processed
   unit-noise section, seed 20261199) and ``sigma_raw = A0 / (rho sigma_u)``.
   Per model (own noise-free processing): ``t_a0``, early lobe ``t_e``,
   ``A0_self`` and ``Lambda(Delta) = L_current(Delta) / A0_self``.
4. Model x rho {10, 20} x seed 0-9 x rule x Delta {-1.0, 0, +1.0, +2.35} ns:
   white noise ``sigma_raw n`` on the resampled raw data, seed
   ``20261120 + 1000 p + 100 q + s``, ``prepare_depth_inputs``, then the wrapper.
   S6P uses the sigma_raw and seeds of the S4N of the same pair; S6P_NP_S010
   uses the D12_S010 sigma_raw with its own pair index p = 3.
5. S6P_NP_S010 with time-ramped noise ``sigma_raw (0.3 + 1.2 t / t_end)``
   (D12_S010 sigma_raw at rho 10, seeds of p = 3, q = 0; F4b).
6. Original gprMax ``dt`` (current, B_full, Delta 0, rho 10; calibrated on the
   original grid the same way; report only).

The envelope is the src path (``pick_signal``, no zero padding, 4-2-6); the
record-end 5 ns are counted separately (``n_false_record_end``).

Outputs (``--out-dir``, default ``data/processing/m210b_s4s6``):

* ``runs.csv``: one row per run, columns ``RUN_COLUMNS`` (fixed).
* ``run_info.json``: seal commit and the seal-commit blob of every sealed file
  (with the working-tree blob), harness commit, input SHA-256, ``MKL_CBWR``,
  calibration and per-model noise-free values, the seed formula, the column
  definitions and the generator wrapper regression (4TU 759 lines regenerated
  into ``<out-dir>/wrapper_regression`` and compared byte for byte).

Protocol v2 (``--protocol v2``, m210b_selector_floor_v2.md W1-W10): models (json
``batch == "v2"``) come from ``analysis/synthetic/s4s6/models_s4s6_v2.json`` and
the exports from ``--input-dir`` (default ``data/synthetic_gprmax_s4s6_v2``);
20 seeds, seed ``20271120 + 1000 p + 100 q + s``, unit-noise seed 20271199
(disjointness from batch 1 is asserted at start), rules current, B_full, B_min,
B_noise, C_none, and six more columns at the row end (``NEAREST_COLUMNS``: the
nearest kept for F5-B and its depth for F5-C); output ``runs_v2.csv``.
The default protocol v1.4 writes the batch-1 ``runs.csv`` unchanged.

Run guards (pre-run audit): the seal record is taken at start and end. A sealed
file that differs from its seal blob, a dirty working tree, an end HEAD other
than the start HEAD, or a wrapper regression that is not byte identical ends the
run with exit code 1 (outputs are still written, ``run_info.guards`` lists why).

Run (one core, BelowNormal):
    conda run -n gpr-dev python -I analysis/processing/m210b_s4s6_pipeline.py
    ... --regression-activities none   (skip the 4TU regeneration, ~3-4 min)
    ... --protocol v2 --seal-commit <v2 seal commit>
"""

import argparse
import csv
import dataclasses
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# Reproducible MKL results across processes, before numpy loads (M2-10); an
# explicit value from the caller wins. Same setting as the S1g harness.
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import numpy as np
import scipy
import skimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "analysis" / "processing"))

import eval_depth_4tu as depth_evaluation  # noqa: E402
import fourtu_apex_candidate_select as generator  # noqa: E402
import m210b_floor_rules as floor_rules  # noqa: E402

from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import (  # noqa: E402
    estimate_depth,
    prepare_depth_inputs,
)
from gpr_detection.depth.picks import ApexHint  # noqa: E402
from gpr_detection.depth.result import DepthStatus  # noqa: E402
from gpr_detection.io.radargram import Radargram, velocity_from_eps  # noqa: E402
from gpr_detection.io.synthetic import load_synthetic  # noqa: E402
from gpr_detection.preprocess.filters import prepare_fitting_branch  # noqa: E402
from gpr_detection.preprocess.signals import pick_signal  # noqa: E402

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "synthetic_gprmax_s4s6"
DEFAULT_MODELS_JSON = (
    PROJECT_ROOT / "analysis" / "synthetic" / "s4s6" / "models_s4s6.json"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processing" / "m210b_s4s6"
LINE_ACTIVITY = "S4S6"

SEAL_COMMIT = "bc9a474"
"""Seal commit of m210b_selector_floor.md v1.4 (2026-11-11)."""
SEALED_FILES = (
    "docs/processing/m210b_selector_floor.md",
    "analysis/processing/m210b_floor_rules.py",
    "analysis/processing/m210b_floor_proto.py",
    "docs/processing/m210b_floor_proto.json",
    "docs/processing/m210b_a0_S6P_D15_S030.json",
    "docs/processing/m210b_a0_S4N_D15_S030.json",
    "analysis/processing/m210b_lambda_pilot.py",
)
SEALED_FOLDER = "analysis/synthetic/s4s6"
"""Every file of this folder at the seal commit is sealed as well (4-2-0)."""

BATCH1_MODELS = (
    "S4N_D15_S010",
    "S6P_D15_S010",
    "S4N_D15_S030",
    "S6P_D15_S030",
    "S4N_D12_S010",
    "S6P_D12_S010",
    "S6P_NP_S010",
    "S6P_D15_S040",
    "S4N_D15_S040",
)
PAIR_INDEX = {"D12_S010": 0, "D15_S010": 1, "D15_S030": 2, "NP_S010": 3, "D15_S040": 4}
"""Seed index p of a depth/sigma pair (shared by S4N and S6P, 4-2 rule 3)."""
NO_PIPE_CALIBRATION_PAIR = "D12_S010"
"""S6P_NP_S010 takes the sigma_raw of this pair (no apex of its own)."""
CONTROL_FAMILY = "S4N"
NO_PIPE_PAIR_PREFIX = "NP_"
JOINT_SUFFIX = re.compile(r"_J\d+$")
"""v2 S6P ids end with the joint-layout tag (``S6P_D14_S040_J2``)."""

DT_HARNESS_NS = 0.098
RHOS = (10, 20)
N_SEEDS = 10
SEED_BASE = 20261120
UNIT_NOISE_SEED = 20261199

PROTOCOL_V1 = "v1.4"
PROTOCOL_V2 = "v2"
V2_MODELS_JSON = (
    PROJECT_ROOT / "analysis" / "synthetic" / "s4s6" / "models_s4s6_v2.json"
)
"""v2 section 7 item 1 (EM); ``--models-json`` overrides it."""
V2_DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "synthetic_gprmax_s4s6_v2"
V2_DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processing" / "m210b_s4s6_v2"
V1_RUNS_CSV = "runs.csv"
V2_RUNS_CSV = "runs_v2.csv"
V2_BATCH = "v2"
"""json ``batch`` of the models that passed the pilot (v2 section 7 item 1)."""
V2_N_SEEDS = 20
V2_SEED_BASE = 20271120
V2_UNIT_NOISE_SEED = 20271199
V2_RULES = ("current", "B_full", "B_min", "B_noise", "C_none")
"""v2 V3 / H3: B_full, B_min judged; B_noise, C_none reported; A_skip1 dropped."""
V2_PAIR_INDEX = {
    "D14_S040": 0,
    "D16_S040": 1,
    "D15_S045": 2,
    "D13_S010": 3,
    "NP_S040": 4,
}
"""v2 H2: pairs P0-P3 -> p 0-3, the no-pipe line -> p 4."""
V2_PAIR_LABELS = {0: "P0", 1: "P1", 2: "P2", 3: "P3", 4: "NP"}
"""json ``pair`` label of each p (v2 section 3), cross-checked on load."""
V2_SEALED_FILES = (
    *SEALED_FILES,
    "docs/processing/m210b_selector_floor_v2.md",
    "docs/processing/m210b_harness_interpretation.md",
    "analysis/synthetic/s4s6/models_s4s6_v2.json",
    "analysis/processing/m210b_s4s6_pipeline.py",
    "tests/test_m210b_s4s6_pipeline.py",
    "analysis/processing/fourtu_apex_candidate_select.py",
    "analysis/processing/m210b_v2_judge.py",
    "tests/test_m210b_v2_judge.py",
    "scripts/synthetic/export_dataset.py",
    "tests/test_synthetic_export.py",
)
"""v2 section 4 sealed files (checked at the v2 seal commit, ``--seal-commit``)."""
V2_SEALED_FOLDERS = (SEALED_FOLDER,)
"""At the v2 seal commit this folder also holds models_s4s6_v2.json, inputs_v2/
and pilot_v2/ (v2 section 4); the v1.4 files in it must be unchanged."""
INTERPRETATION_DOC = "docs/processing/m210b_harness_interpretation.md"
QC_CONSTANTS_JSON = PROJECT_ROOT / "docs" / "processing" / "m29c_qc_constants.json"
PROTOTYPE_SEEDS = range(20261111, 20262821)
"""m210b_floor_proto.py seeds: 20261111 + i up to + 1000 + 100 * 7 + 9 (reported)."""
THREAD_VARIABLES = (
    "MKL_CBWR",
    "MKL_NUM_THREADS",
    "MKL_THREADING_LAYER",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
RAMP_START, RAMP_SLOPE = 0.3, 1.2
"""F4b noise ramp: sigma(t) = sigma_raw (0.3 + 1.2 t / t_end) (ASSUMPTION, 4-2)."""
RAMP_RHO = 10
ORIGINAL_DT_RULES = ("current", "B_full")
ORIGINAL_DT_RHO = 10

APEX_TRACES = 3
APEX_PERIODS = 0.5
EARLY_LOBE_PERIODS = (1.5, 0.5)
"""Early-lobe window [t_a0 - 1.5/f_c, t_a0 - 0.5/f_c) (P1, P2)."""
EARLY_LOBE_REL = 0.3
FALSE_DEEP_MIN_M = 0.30
RECORD_END_NS = 5.0
WINDOW_TOL = 1e-9
"""Float slack so the closed window boundaries are inside."""
CAP_PEAKS = generator.MAX_CANDIDATES

LOST_EARLY_LOBE = "early_lobe_nms"
LOST_CAP6 = "cap6"
LOST_OTHER_NMS = "other_nms"
LOST_NOT_ACCEPTED = "not_accepted"
LOST_NO_PEAK = "no_peak"

RUN_COLUMNS = (
    "model_id",
    "rho",
    "seed",
    "rule",
    "delta_ns",
    "ramp_noise",
    "dt_ns",
    "t_zero_ns",
    "zero_rule",
    "L",
    "F",
    "sigma_n",
    "rho_real",
    "Lambda",
    "argmax_level_periods",
    "accepted_apex",
    "peak_on_crest",
    "kept_apex",
    "kept_apex_trace",
    "kept_apex_sample",
    "lost_reason",
    "floor_held_at_apex",
    "n_peaks",
    "cap300_hit",
    "n_false_deep",
    "n_false_near_surface",
    "n_false_record_end",
    "rho_real_cal",
)
"""``runs.csv`` columns, fixed by 8-3; ``rho_real_cal`` appended by user decision
D1 (2026-11-11, docs/processing/m210b_harness_interpretation.md)."""

COLUMN_DEFINITIONS = {
    "seed": "full seed 20261120 + 1000 p + 100 q + s (run_info protocol: p, q)",
    "rho_real": (
        "A0_self / sigma_real (decision D1; the column for qualification Q); "
        "A0_self = the model's own noise-free apex envelope; blank without a "
        "pipe. sigma_real = 1.4826 MAD of (S_noisy - S_clean) over the Delta-0 "
        "search rows of the noise-free zero"
    ),
    "rho_real_cal": (
        "A0_cal / sigma_real (auxiliary); A0_cal = calibration A0 of the S4N "
        "control of the pair"
    ),
    "sigma_n": "1.4826 MAD of S_noisy over this run's search rows (B_noise sigma)",
    "Lambda": "L of the noise-free processing (current rule, this Delta) / A0_self",
    "L, F": "level maximum and floor of this run (rule, Delta)",
    "argmax_level_periods": "argmax of the unfloored level, periods below the mute",
    "accepted_apex": "an accepted candidate within apex trace +-3, t_a0 +-0.5/f_c",
    "peak_on_crest": "a detection peak whose crest-moved point is in that window",
    "kept_apex": "a kept candidate (NMS, cap 6) in that window",
    "kept_apex_trace, kept_apex_sample": "best-ranked kept candidate in the window",
    "lost_reason": (
        "blank when kept_apex; else the fate of the best-ranked accepted window "
        "candidate: early_lobe_nms (winner within +-3 traces and "
        "[t_a0 - 1.5/f_c, t_a0 - 0.5/f_c)), other_nms, cap6; not_accepted (a "
        "peak_on_crest but no accepted window candidate); no_peak (also for a "
        "failed zero or a flat section). Blank for a no-pipe model"
    ),
    "floor_held_at_apex": "level(k_a) < F, k_a = round(t_a0 / dt)",
    "cap300_hit": "raw detection peaks (before the edge rule) reached 300",
    "n_false_*": (
        "kept candidates outside the apex window: deep = nominal depth "
        "v_meta tau / 2 >= 0.30 m and t < t_end - 5 ns; near_surface = < 0.30 m "
        "and t < t_end - 5 ns; record_end = t >= t_end - 5 ns"
    ),
}

NEAREST_COLUMNS = (
    "kept_nearest_trace",
    "kept_nearest_sample",
    "kept_nearest_t_ns",
    "n_kept_in_window",
    "depth_nearest_m",
    "depth_nearest_status",
)
"""v2 H4 and H6 columns, appended after ``RUN_COLUMNS`` (protocol v2 only)."""
V2_RUN_COLUMNS = (*RUN_COLUMNS, *NEAREST_COLUMNS)
V2_COLUMN_DEFINITIONS = {
    **COLUMN_DEFINITIONS,
    "seed": "full seed 20271120 + 1000 p + 100 q + s (v2 H2; run_info protocol)",
    "kept_nearest_trace, kept_nearest_sample, kept_nearest_t_ns": (
        "kept candidate in the apex window (trace +-3, t_a0 +-0.5/f_c, closed) "
        "with the smallest |t_ns - t_a0|, then the smallest |trace - apex|, then "
        "the smallest trace (v2 1-3, H4); blank when the window has no kept"
    ),
    "n_kept_in_window": "number of kept candidates in that window",
    "depth_nearest_m, depth_nearest_status": (
        "F5-C (report): eval_depth_4tu.candidate_box -> estimate_depth with "
        "apex_hint = the nearest kept, DepthConfig() defaults, metadata eps_r; "
        "depth blank unless the status is ok (H6)"
    ),
}


# ----------------------------------------------------------------- plan


@dataclass(frozen=True)
class HarnessPlan:
    """Conditions of one harness run (``DEFAULT_PLAN`` is the v1.4 4-2 protocol)."""

    rhos: tuple[int, ...] = RHOS
    n_seeds: int = N_SEEDS
    rules: tuple[str, ...] = floor_rules.RULES
    deltas_ns: tuple[float, ...] = (-1.0, 0.0, 1.0, 2.35)
    original_dt_rules: tuple[str, ...] = ORIGINAL_DT_RULES
    seed_base: int = SEED_BASE
    unit_noise_seed: int = UNIT_NOISE_SEED
    pair_index: Mapping[str, int] = dataclasses.field(
        default_factory=lambda: dict(PAIR_INDEX)
    )
    no_pipe_calibration_pair: str = NO_PIPE_CALIBRATION_PAIR
    columns: tuple[str, ...] = RUN_COLUMNS
    """``V2_RUN_COLUMNS`` also computes the nearest kept and its depth (v2)."""

    @property
    def has_nearest_columns(self) -> bool:
        return "kept_nearest_trace" in self.columns


DEFAULT_PLAN = HarnessPlan()
V2_PLAN = HarnessPlan(
    n_seeds=V2_N_SEEDS,
    rules=V2_RULES,
    seed_base=V2_SEED_BASE,
    unit_noise_seed=V2_UNIT_NOISE_SEED,
    pair_index=dict(V2_PAIR_INDEX),
    no_pipe_calibration_pair="",
    columns=V2_RUN_COLUMNS,
)
"""v2 plan; ``no_pipe_calibration_pair`` comes from the v2 json (see
``load_v2_models``)."""


# ----------------------------------------------------------------- small helpers


def resample_to(radargram: Radargram, dt_ns: float) -> Radargram:
    """Linear resampling of every trace to ``t_k = t_0 + k dt_ns`` (4-2 rule 2).

    ``k = 0 .. floor((t_end - t_0) / dt_ns)``; ``t_0`` is the json
    ``time_of_sample0_ns`` (0 for gprMax exports). The data stay float32.
    """
    t_first = float(radargram.metadata.get("time_of_sample0_ns", 0.0))
    t_old = t_first + np.arange(radargram.n_samples) * radargram.dt_ns
    n_new = math.floor((t_old[-1] - t_first) / dt_ns) + 1
    t_new = t_first + np.arange(n_new) * dt_ns
    data = np.column_stack(
        [
            np.interp(t_new, t_old, radargram.data[:, j])
            for j in range(radargram.n_traces)
        ]
    )
    return dataclasses.replace(
        radargram, data=data.astype(np.float32), dt_ns=float(dt_ns)
    )


def noise_seed(
    pair_index: int, rho_index: int, seed_index: int, base: int = SEED_BASE
) -> int:
    """``base + 1000 p + 100 q + s`` (v1.4 rule 3: 20261120; v2 H2: 20271120)."""
    return base + 1000 * pair_index + 100 * rho_index + seed_index


def plan_seeds(plan: HarnessPlan) -> set[int]:
    """Every noise seed a plan can draw, plus its unit-noise seed."""
    seeds = {
        noise_seed(p, q, s, plan.seed_base)
        for p in plan.pair_index.values()
        for q in range(len(plan.rhos))
        for s in range(plan.n_seeds)
    }
    return seeds | {plan.unit_noise_seed}


class SeedOverlapError(ValueError):
    """A v2 seed is also a batch-1 seed (v2 H2)."""


def assert_no_seed_overlap(seeds: set[int], batch1_seeds: set[int]) -> int:
    """Return 0 (the ``seed_overlap`` record) or raise when the sets intersect."""
    overlap = seeds & batch1_seeds
    if overlap:
        raise SeedOverlapError(
            f"{len(overlap)} seeds shared with batch 1: {sorted(overlap)[:5]}"
        )
    return 0


def white_noise(shape: tuple[int, int], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape)


def with_noise(radargram: Radargram, noise: np.ndarray) -> Radargram:
    """``x + noise`` on the raw (unprocessed) data, back to float32."""
    noisy = radargram.data.astype(np.float64) + noise
    return dataclasses.replace(radargram, data=noisy.astype(np.float32))


def ramp_profile(n_samples: int, dt_ns: float) -> np.ndarray:
    """Column ``0.3 + 1.2 t / t_end`` (F4b), ``t_end`` = last sample time."""
    t_ns = np.arange(n_samples) * dt_ns
    return (RAMP_START + RAMP_SLOPE * t_ns / t_ns[-1])[:, None]


def period_ns(f_c_hz: float) -> float:
    return 1e9 / f_c_hz


def is_in_apex_window(
    trace: float, t_ns: float, apex_trace: int, t_a0: float, f_c_hz: float
) -> bool:
    """Apex trace +-3 and ``t_a0`` +-0.5/f_c, boundaries included."""
    half_ns = APEX_PERIODS * period_ns(f_c_hz)
    return (
        abs(trace - apex_trace) <= APEX_TRACES + WINDOW_TOL
        and abs(t_ns - t_a0) <= half_ns + WINDOW_TOL
    )


def is_in_early_lobe_box(
    trace: float, t_ns: float, apex_trace: int, t_a0: float, f_c_hz: float
) -> bool:
    """Apex trace +-3 and ``[t_a0 - 1.5/f_c, t_a0 - 0.5/f_c)`` (P1)."""
    far, near = (periods * period_ns(f_c_hz) for periods in EARLY_LOBE_PERIODS)
    return (
        abs(trace - apex_trace) <= APEX_TRACES + WINDOW_TOL
        and t_a0 - far - WINDOW_TOL <= t_ns < t_a0 - near - WINDOW_TOL
    )


def _blank_or_rounded(value: float | None, digits: int) -> object:
    if value is None or not math.isfinite(value):
        return ""
    return round(float(value), digits)


# ----------------------------------------------------------------- apex outcome


@dataclass(frozen=True)
class ApexTarget:
    """The apex a run is scored against (noise-free processing of the model)."""

    trace: int
    t_a0_ns: float
    f_c_hz: float


@dataclass(frozen=True)
class ApexOutcome:
    accepted_apex: bool
    peak_on_crest: bool
    kept_apex: bool
    kept_trace: int | None
    kept_sample: float | None
    lost_reason: str


def _window_members(rows: Sequence[Mapping], target: ApexTarget) -> list[int]:
    return [
        index
        for index, row in enumerate(rows)
        if is_in_apex_window(
            row["trace"], row["t_ns"], target.trace, target.t_a0_ns, target.f_c_hz
        )
    ]


def lost_reason_of(
    accepted: Sequence[Mapping],
    fates: Sequence[Mapping],
    kept: Sequence[Mapping],
    window_accepted: Sequence[int],
    target: ApexTarget,
) -> str:
    """Why the best-ranked accepted window candidate is not kept (P1)."""
    first = window_accepted[0]
    fate = fates[first]
    if fate["fate"] == generator.FATE_CAP6:
        return LOST_CAP6
    if fate["fate"] != generator.FATE_NMS:
        raise ValueError(f"accepted window candidate {accepted[first]} is kept")
    winner = kept[fate["winner"]]
    is_early_lobe = is_in_early_lobe_box(
        winner["trace"], winner["t_ns"], target.trace, target.t_a0_ns, target.f_c_hz
    )
    return LOST_EARLY_LOBE if is_early_lobe else LOST_OTHER_NMS


def apex_outcome(
    line: Mapping, kept: Sequence[Mapping], dt_ns: float, target: ApexTarget
) -> ApexOutcome:
    """Accepted-stage and kept-stage success of one run (P1)."""
    accepted = line.get("_accepted_all", [])
    fates = line.get("_accepted_fate", [])
    peaks = line.get("_peaks", [])
    window_accepted = _window_members(accepted, target)
    window_kept = _window_members(kept, target)
    peak_on_crest = any(
        is_in_apex_window(
            peak["trace"],
            peak["sample"] * dt_ns,
            target.trace,
            target.t_a0_ns,
            target.f_c_hz,
        )
        for peak in peaks
    )
    if window_kept:
        best = kept[window_kept[0]]
        return ApexOutcome(
            bool(window_accepted),
            peak_on_crest,
            True,
            best["trace"],
            best["sample"],
            "",
        )
    if window_accepted:
        reason = lost_reason_of(accepted, fates, kept, window_accepted, target)
    elif peak_on_crest:
        reason = LOST_NOT_ACCEPTED
    else:
        reason = LOST_NO_PEAK
    return ApexOutcome(bool(window_accepted), peak_on_crest, False, None, None, reason)


@dataclass(frozen=True)
class FalseCounts:
    deep: int
    near_surface: int
    record_end: int


def false_counts(
    kept: Sequence[Mapping],
    target: ApexTarget | None,
    velocity_m_per_ns: float,
    t_end_ns: float,
) -> FalseCounts:
    """Kept candidates outside the apex window by nominal depth and record end (P4)."""
    deep = near_surface = record_end = 0
    for row in kept:
        if target is not None and is_in_apex_window(
            row["trace"], row["t_ns"], target.trace, target.t_a0_ns, target.f_c_hz
        ):
            continue
        if row["t_ns"] >= t_end_ns - RECORD_END_NS:
            record_end += 1
        elif velocity_m_per_ns * row["tau_ns"] / 2.0 >= FALSE_DEEP_MIN_M:
            deep += 1
        else:
            near_surface += 1
    return FalseCounts(deep, near_surface, record_end)


# ----------------------------------------------------------------- noise-free values


def apex_window_rows(t_expected_ns: float, dt_ns: float, f_c_hz: float) -> slice:
    half_ns = APEX_PERIODS * period_ns(f_c_hz)
    k_first = math.ceil((t_expected_ns - half_ns) / dt_ns - WINDOW_TOL)
    k_last = math.floor((t_expected_ns + half_ns) / dt_ns + WINDOW_TOL)
    return slice(k_first, k_last + 1)


def apex_amplitude(
    envelope: np.ndarray,
    apex_trace: int,
    t_expected_ns: float,
    dt_ns: float,
    f_c_hz: float,
) -> float:
    """``A0``: envelope max over apex trace +-3 and the expected time +-0.5/f_c."""
    rows = apex_window_rows(t_expected_ns, dt_ns, f_c_hz)
    traces = slice(max(0, apex_trace - APEX_TRACES), apex_trace + APEX_TRACES + 1)
    return float(envelope[rows, traces].max())


def apex_time_and_early_lobe(
    envelope: np.ndarray,
    apex_trace: int,
    t_expected_ns: float,
    dt_ns: float,
    f_c_hz: float,
) -> tuple[float, float | None]:
    """``t_a0`` and the early lobe ``t_e`` (or ``None``) on the apex trace (P2).

    ``t_a0`` is the envelope maximum within the expected time +-0.5/f_c. ``t_e``
    is the largest local maximum of at least 0.3 E(t_a0) in
    ``[t_a0 - 1.5/f_c, t_a0 - 0.5/f_c)``.
    """
    column = envelope[:, apex_trace]
    rows = apex_window_rows(t_expected_ns, dt_ns, f_c_hz)
    k_apex = rows.start + int(np.argmax(column[rows]))
    t_a0 = k_apex * dt_ns
    best_k, best_value = None, EARLY_LOBE_REL * float(column[k_apex])
    for k in range(1, column.size - 1):
        if not is_in_early_lobe_box(apex_trace, k * dt_ns, apex_trace, t_a0, f_c_hz):
            continue
        is_local_max = column[k - 1] < column[k] >= column[k + 1]
        if is_local_max and column[k] >= best_value:
            best_k, best_value = k, float(column[k])
    return t_a0, None if best_k is None else best_k * dt_ns


def unit_noise_sigma(
    radargram: Radargram, k_lo: int, k_hi: int, seed: int = UNIT_NOISE_SEED
) -> float:
    """``sigma_u``: processed unit white noise (``seed``), search-row MAD.

    Only the processing branch is used (the zero of a pure-noise section is
    meaningless); the search rows come from the control's noise-free zero.
    """
    noise = white_noise(radargram.data.shape, seed)
    unit = dataclasses.replace(radargram, data=noise.astype(np.float32))
    processed = prepare_fitting_branch(unit, DepthConfig().bg_window)
    return floor_rules.noise_sigma(processed.background_removed[k_lo:k_hi])


@dataclass(frozen=True)
class CleanLine:
    """Noise-free processing of one (resampled or original) line."""

    radargram: Radargram
    background_removed: np.ndarray
    t_zero_ns: float | None
    zero_rule: str | None
    target: ApexTarget | None
    a0_self: float | None
    t_early_ns: float | None
    level_max_by_delta: dict[float, float]
    argmax_periods_by_delta: dict[float, float] = dataclasses.field(
        default_factory=dict
    )
    """Noise-free argmax of the unfloored current level, periods below the mute
    (v2 H5 / V5)."""


def clean_line(
    radargram: Radargram, model_meta: Mapping, plan: HarnessPlan
) -> CleanLine:
    """Noise-free values of a line: zero, apex target, ``A0_self``, ``L(Delta)``."""
    inputs = prepare_depth_inputs(radargram, DepthConfig())
    t_zero = inputs.zero.t_zero_ns
    f_c, dt = radargram.f_c_hz, radargram.dt_ns
    envelope = pick_signal(inputs.branch.background_removed, "envelope")
    target, a0_self, t_early = None, None, None
    apex_trace = model_meta.get("apex_trace_index")
    if apex_trace is not None:
        t_expected = float(model_meta["expected_apex_envelope_peak_ns_approx"])
        t_a0, t_early = apex_time_and_early_lobe(
            envelope, apex_trace, t_expected, dt, f_c
        )
        target = ApexTarget(int(apex_trace), t_a0, f_c)
        a0_self = apex_amplitude(envelope, apex_trace, t_expected, dt, f_c)
    levels, argmax_periods = {}, {}
    if t_zero is not None:
        for delta in plan.deltas_ns:
            line, _, _ = generator.apex_candidates_for_radargram(
                radargram, inputs, act=LINE_ACTIVITY, path="clean",
                floor_rule="current", mute_offset_ns=delta,
            )  # fmt: skip
            if "_floor" in line:
                levels[delta] = line["_floor"]["L"]
                k_argmax = int(np.argmax(line["_floor"]["level"]))
                argmax_periods[delta] = k_argmax * dt / period_ns(f_c)
    return CleanLine(
        radargram,
        inputs.branch.background_removed,
        t_zero,
        None if inputs.zero.zero_rule is None else str(inputs.zero.zero_rule),
        target,
        a0_self,
        t_early,
        levels,
        argmax_periods,
    )


@dataclass(frozen=True)
class Calibration:
    """Noise level of one depth/sigma pair on one time grid (4-2 rule 3)."""

    control_id: str
    a0: float
    sigma_u: float
    sigma_raw_by_rho: dict[int, float]


def calibrate(
    control: CleanLine, rhos: Sequence[int], unit_seed: int = UNIT_NOISE_SEED
) -> Calibration:
    if control.a0_self is None or control.t_zero_ns is None:
        raise ValueError(f"{control.radargram.line_id}: no apex or zero to calibrate")
    k_lo, k_hi = floor_rules.search_rows(
        control.t_zero_ns,
        control.radargram.f_c_hz,
        control.radargram.dt_ns,
        control.radargram.n_samples,
    )
    sigma_u = unit_noise_sigma(control.radargram, k_lo, k_hi, unit_seed)
    return Calibration(
        control_id=control.radargram.line_id,
        a0=control.a0_self,
        sigma_u=sigma_u,
        sigma_raw_by_rho={rho: control.a0_self / (rho * sigma_u) for rho in rhos},
    )


# ----------------------------------------------------------------- runs


@dataclass(frozen=True)
class Realisation:
    """One noisy line of a model (all rules and mute shifts run on it)."""

    model_id: str
    rho: int
    seed: int
    ramp_noise: bool
    noise: np.ndarray


@dataclass(frozen=True)
class RealisedRho:
    """Realised apex-to-noise ratios of one noisy line (decision D1).

    ``own`` (``rho_real``, used for qualification Q) has the model's own
    noise-free apex ``A0_self`` as numerator, ``None`` without a pipe;
    ``calibration`` (``rho_real_cal``, auxiliary) has the pair's ``A0_cal``.
    """

    own: float | None
    calibration: float


def _ratio(numerator: float, sigma: float) -> float:
    if math.isnan(sigma):
        return float("nan")
    return numerator / sigma if sigma > 0 else float("inf")


def realised_rho(
    clean: CleanLine, noisy_removed: np.ndarray, a0_cal: float
) -> RealisedRho:
    """``A0 / sigma_real``; sigma_real = 1.4826 MAD of the processed noise only.

    The noise is ``S_noisy - S_clean`` over the Delta-0 search rows of the
    noise-free zero, so clutter does not enter sigma_real.
    """
    sigma_real = float("nan")
    if clean.t_zero_ns is not None:
        k_lo, k_hi = floor_rules.search_rows(
            clean.t_zero_ns,
            clean.radargram.f_c_hz,
            clean.radargram.dt_ns,
            clean.radargram.n_samples,
        )
        noisy_rows = noisy_removed[k_lo:k_hi].astype(np.float64)
        clean_rows = clean.background_removed[k_lo:k_hi].astype(np.float64)
        sigma_real = floor_rules.noise_sigma(noisy_rows - clean_rows)
    own = None if clean.a0_self is None else _ratio(clean.a0_self, sigma_real)
    return RealisedRho(own=own, calibration=_ratio(a0_cal, sigma_real))


def _floor_fields(
    line: Mapping, target: ApexTarget | None, dt_ns: float, f_c: float
) -> dict[str, object]:
    floor = line.get("_floor")
    if floor is None:
        return {"L": "", "F": "", "argmax_level_periods": "", "floor_held_at_apex": ""}
    level = floor["level"]
    k_argmax = int(np.argmax(level))
    held: object = ""
    if target is not None:
        k_apex = round(target.t_a0_ns / dt_ns) - floor["k_lo"]
        if 0 <= k_apex < level.size:
            held = bool(level[k_apex] < floor["F"])
    return {
        "L": _blank_or_rounded(floor["L"], 8),
        "F": _blank_or_rounded(floor["F"], 8),
        "argmax_level_periods": round(k_argmax * dt_ns / period_ns(f_c), 4),
        "floor_held_at_apex": held,
    }


def run_row(
    realisation: Realisation,
    noisy: Radargram,
    inputs,
    clean: CleanLine,
    rule: str,
    delta_ns: float,
    rho_real: RealisedRho,
    columns: Sequence[str] = RUN_COLUMNS,
) -> dict[str, object]:
    """One ``runs.csv`` row: the wrapper with ``rule`` and mute shift ``delta_ns``.

    ``inputs`` is ``prepare_depth_inputs(noisy)``; ``clean`` gives the apex
    target and the noise-free ``Lambda``. ``columns`` with the nearest-kept
    columns (v2) also runs the F5-C depth path.
    """
    dt, f_c = noisy.dt_ns, noisy.f_c_hz
    line, kept, _ = generator.apex_candidates_for_radargram(
        noisy,
        inputs,
        act=LINE_ACTIVITY,
        path=realisation.model_id,
        floor_rule=rule,
        mute_offset_ns=delta_ns,
    )
    target = clean.target
    t_end = (noisy.n_samples - 1) * dt
    velocity = velocity_from_eps(noisy.eps_r_meta)
    counts = false_counts(kept, target, velocity, t_end)
    row: dict[str, object] = {
        "model_id": realisation.model_id,
        "rho": realisation.rho,
        "seed": realisation.seed,
        "rule": rule,
        "delta_ns": delta_ns,
        "ramp_noise": int(realisation.ramp_noise),
        "dt_ns": dt,
        "t_zero_ns": _blank_or_rounded(inputs.zero.t_zero_ns, 4),
        "zero_rule": ""
        if inputs.zero.zero_rule is None
        else str(inputs.zero.zero_rule),
        "sigma_n": "",
        "rho_real": _blank_or_rounded(rho_real.own, 4),
        "rho_real_cal": _blank_or_rounded(rho_real.calibration, 4),
        "Lambda": "",
        "n_peaks": line.get("n_peaks", 0),
        "cap300_hit": line.get("_floor", {}).get("n_raw_peaks", 0) >= CAP_PEAKS,
        "n_false_deep": counts.deep,
        "n_false_near_surface": counts.near_surface,
        "n_false_record_end": counts.record_end,
    }
    row.update(_floor_fields(line, target, dt, f_c))
    floor = line.get("_floor")
    if floor is not None:
        searched = inputs.branch.background_removed[floor["k_lo"] : floor["k_hi"]]
        row["sigma_n"] = _blank_or_rounded(floor_rules.noise_sigma(searched), 8)
    level_max = clean.level_max_by_delta.get(delta_ns)
    if clean.a0_self and level_max is not None:
        # 4 digits in runs.csv (the judged column, v2.1 section 4); run_info
        # keeps the unrounded value (models.<id>.harness.Lambda).
        row["Lambda"] = round(level_max / clean.a0_self, 4)
    row.update(_outcome_fields(line, kept, dt, target))
    if "kept_nearest_trace" in columns:
        row.update(_nearest_fields(noisy, inputs, kept, target))
    return {column: row[column] for column in columns}


def nearest_kept(
    kept: Sequence[Mapping], target: ApexTarget
) -> tuple[Mapping | None, int]:
    """Nearest kept in the apex window and the number of kept there (v2 1-3, H4).

    Order: smallest ``|t_ns - t_a0|``, then smallest ``|trace - apex|``, then the
    smallest trace index. Independent of the rank order of ``kept``.
    """
    in_window = [kept[index] for index in _window_members(kept, target)]
    if not in_window:
        return None, 0

    def distance(row: Mapping) -> tuple[float, int, int]:
        return (
            abs(row["t_ns"] - target.t_a0_ns),
            abs(row["trace"] - target.trace),
            row["trace"],
        )

    return min(in_window, key=distance), len(in_window)


def nearest_depth(noisy: Radargram, inputs, nearest: Mapping) -> tuple[object, str]:
    """F5-C: depth of the evaluation path with the nearest kept as apex hint (H6)."""
    trace, sample = int(nearest["trace"]), float(nearest["sample"])
    box = depth_evaluation.candidate_box(noisy, trace, sample, float(nearest["tau_ns"]))
    result = estimate_depth(
        noisy,
        box,
        DepthConfig(),
        inputs=inputs,
        apex_hint=ApexHint(trace=trace, sample=sample),
    )
    status = result.status.value
    if status != DepthStatus.OK.value or result.depth_top_m is None:
        return "", status
    return round(float(result.depth_top_m), 5), status


def _nearest_fields(noisy, inputs, kept, target) -> dict[str, object]:
    blank = dict.fromkeys(NEAREST_COLUMNS, "")
    if target is None:
        return blank
    nearest, n_in_window = nearest_kept(kept, target)
    blank["n_kept_in_window"] = n_in_window
    if nearest is None:
        return blank
    depth_m, status = nearest_depth(noisy, inputs, nearest)
    return {
        "kept_nearest_trace": nearest["trace"],
        "kept_nearest_sample": nearest["sample"],
        "kept_nearest_t_ns": nearest["t_ns"],
        "n_kept_in_window": n_in_window,
        "depth_nearest_m": depth_m,
        "depth_nearest_status": status,
    }


def _outcome_fields(line, kept, dt_ns, target) -> dict[str, object]:
    if target is None:
        names = ("accepted_apex", "peak_on_crest", "kept_apex", "kept_apex_trace",
                 "kept_apex_sample", "lost_reason")  # fmt: skip
        return dict.fromkeys(names, "")
    outcome = apex_outcome(line, kept, dt_ns, target)
    return {
        "accepted_apex": outcome.accepted_apex,
        "peak_on_crest": outcome.peak_on_crest,
        "kept_apex": outcome.kept_apex,
        "kept_apex_trace": "" if outcome.kept_trace is None else outcome.kept_trace,
        "kept_apex_sample": "" if outcome.kept_sample is None else outcome.kept_sample,
        "lost_reason": outcome.lost_reason,
    }


def run_realisation(
    realisation: Realisation,
    clean: CleanLine,
    a0_cal: float,
    rules: Sequence[str],
    deltas_ns: Sequence[float],
    columns: Sequence[str] = RUN_COLUMNS,
) -> list[dict[str, object]]:
    """Process one noisy line once and run every rule and mute shift on it."""
    noisy = with_noise(clean.radargram, realisation.noise)
    inputs = prepare_depth_inputs(noisy, DepthConfig())
    rho_real = realised_rho(clean, inputs.branch.background_removed, a0_cal)
    return [
        run_row(realisation, noisy, inputs, clean, rule, delta, rho_real, columns)
        for rule in rules
        for delta in deltas_ns
    ]


# ----------------------------------------------------------------- models


def model_pair(model_id: str) -> str:
    """``S6P_D15_S030`` -> ``D15_S030``; ``S6P_D14_S040_J2`` -> ``D14_S040``."""
    return JOINT_SUFFIX.sub("", model_id.split("_", 1)[1])


def calibration_pair(model_id: str, plan: HarnessPlan = DEFAULT_PLAN) -> str:
    """The S4N pair whose calibration a model uses (no-pipe lines: the plan's)."""
    pair = model_pair(model_id)
    is_no_pipe = pair.startswith(NO_PIPE_PAIR_PREFIX)
    return plan.no_pipe_calibration_pair if is_no_pipe else pair


def load_line(model_id: str, data_root: Path) -> Radargram:
    """S4/S6 line as the harness reads it (4-2 rule 1)."""
    radargram = load_synthetic(
        model_id,
        root=data_root,
        include_eps=True,
        include_radius=False,
        include_height=True,
    )
    return dataclasses.replace(radargram, line_id=f"{LINE_ACTIVITY}/{model_id}")


def load_model(model_id: str, data_root: Path) -> tuple[Radargram, dict]:
    """The line and its export json (for ``apex_trace_index`` and the expected time)."""
    meta = json.loads((data_root / f"{model_id}.json").read_text(encoding="utf-8"))
    return load_line(model_id, data_root), meta


GRID_HARNESS = "harness"
GRID_ORIGINAL = "original"


@dataclass
class ModelSet:
    """Loaded lines, noise-free values and calibrations of one run (lazy, cached).

    ``loader`` returns ``(radargram, export json)`` for a model id; tests pass
    a fake one.
    """

    data_root: Path
    plan: HarnessPlan
    loader: Callable[[str, Path], tuple[Radargram, dict]] = load_model
    _lines: dict = dataclasses.field(default_factory=dict)
    _clean: dict = dataclasses.field(default_factory=dict)
    _calibrations: dict = dataclasses.field(default_factory=dict)

    def _line(self, model_id: str, grid: str) -> tuple[Radargram, dict]:
        if model_id not in self._lines:
            original, meta = self.loader(model_id, self.data_root)
            resampled = resample_to(original, DT_HARNESS_NS)
            self._lines[model_id] = (
                {GRID_ORIGINAL: original, GRID_HARNESS: resampled},
                meta,
            )
        grids, meta = self._lines[model_id]
        return grids[grid], meta

    def clean(self, model_id: str, grid: str) -> CleanLine:
        key = (model_id, grid)
        if key not in self._clean:
            radargram, meta = self._line(model_id, grid)
            self._clean[key] = clean_line(radargram, meta, self.plan)
        return self._clean[key]

    def calibration(self, pair: str, grid: str) -> Calibration:
        key = (pair, grid)
        if key not in self._calibrations:
            control = self.clean(control_id(pair), grid)
            self._calibrations[key] = calibrate(
                control, self.plan.rhos, self.plan.unit_noise_seed
            )
        return self._calibrations[key]

    def meta(self, model_id: str) -> dict:
        """The export json of a loaded model."""
        return self._lines[model_id][1]

    def loaded_ids(self) -> list[str]:
        """Every model read so far (including calibration controls)."""
        return sorted(self._lines)


def control_id(pair: str) -> str:
    return f"{CONTROL_FAMILY}_{pair}"


def model_runs(model_id: str, models: ModelSet) -> list[dict[str, object]]:
    """Steps 4-6 for one model."""
    plan = models.plan
    clean = models.clean(model_id, GRID_HARNESS)
    calibration = models.calibration(calibration_pair(model_id, plan), GRID_HARNESS)
    pair_index = plan.pair_index[model_pair(model_id)]
    shape = clean.radargram.data.shape
    rows = []
    for rho_index, rho in enumerate(plan.rhos):
        sigma_raw = calibration.sigma_raw_by_rho[rho]
        for s in range(plan.n_seeds):
            seed = noise_seed(pair_index, rho_index, s, plan.seed_base)
            noise = sigma_raw * white_noise(shape, seed)
            realisation = Realisation(model_id, rho, seed, False, noise)
            rows += run_realisation(
                realisation,
                clean,
                calibration.a0,
                plan.rules,
                plan.deltas_ns,
                plan.columns,
            )
    if clean.target is None:
        rows += ramp_runs(model_id, clean, calibration, pair_index, plan)
    rows += original_dt_runs(model_id, models, pair_index)
    return rows


def ramp_runs(model_id, clean, calibration, pair_index, plan) -> list[dict]:
    """Step 5: time-ramped noise on the no-pipe line (F4b), rho 10, q = 0."""
    rho_index = plan.rhos.index(RAMP_RHO)
    shape = clean.radargram.data.shape
    ramp = ramp_profile(shape[0], clean.radargram.dt_ns)
    rows = []
    for s in range(plan.n_seeds):
        seed = noise_seed(pair_index, rho_index, s, plan.seed_base)
        noise = calibration.sigma_raw_by_rho[RAMP_RHO] * ramp * white_noise(shape, seed)
        realisation = Realisation(model_id, RAMP_RHO, seed, True, noise)
        rows += run_realisation(
            realisation,
            clean,
            calibration.a0,
            plan.rules,
            plan.deltas_ns,
            plan.columns,
        )
    return rows


def original_dt_runs(model_id: str, models: ModelSet, pair_index: int) -> list[dict]:
    """Step 6: the gprMax dt, calibrated on that grid (report only)."""
    plan = models.plan
    clean = models.clean(model_id, GRID_ORIGINAL)
    calibration = models.calibration(calibration_pair(model_id, plan), GRID_ORIGINAL)
    rho_index = plan.rhos.index(ORIGINAL_DT_RHO)
    shape = clean.radargram.data.shape
    rows = []
    for s in range(plan.n_seeds):
        seed = noise_seed(pair_index, rho_index, s, plan.seed_base)
        noise = calibration.sigma_raw_by_rho[ORIGINAL_DT_RHO] * white_noise(shape, seed)
        realisation = Realisation(model_id, ORIGINAL_DT_RHO, seed, False, noise)
        rows += run_realisation(
            realisation,
            clean,
            calibration.a0,
            plan.original_dt_rules,
            (0.0,),
            plan.columns,
        )
    return rows


def model_record(model_id: str, models: ModelSet) -> dict[str, object]:
    """Noise-free values and the calibration used, per grid (run_info)."""
    plan = models.plan
    meta = models.meta(model_id)
    record: dict[str, object] = {
        "pair": model_pair(model_id),
        "pair_index_p": plan.pair_index[model_pair(model_id)],
        "calibration_pair": calibration_pair(model_id, plan),
        "pipe_top_depth_m": meta.get("pipe_top_depth_m"),
    }
    for grid in (GRID_HARNESS, GRID_ORIGINAL):
        clean = models.clean(model_id, grid)
        calibration = models.calibration(calibration_pair(model_id, plan), grid)
        target = clean.target
        record[grid] = {
            "dt_ns": clean.radargram.dt_ns,
            "n_samples": clean.radargram.n_samples,
            "t_end_ns": (clean.radargram.n_samples - 1) * clean.radargram.dt_ns,
            "zero_rule": clean.zero_rule,
            "t_zero_ns": clean.t_zero_ns,
            "apex_trace": None if target is None else target.trace,
            "t_a0_ns": None if target is None else target.t_a0_ns,
            "t_e_ns": clean.t_early_ns,
            "A0_self": clean.a0_self,
            "L_noise_free_current": {
                f"{d:+.2f}": v for d, v in clean.level_max_by_delta.items()
            },
            "Lambda": {
                f"{d:+.2f}": (v / clean.a0_self if clean.a0_self else None)
                for d, v in clean.level_max_by_delta.items()
            },
            "argmax_level_periods_noise_free": {
                f"{d:+.2f}": round(v, 4)
                for d, v in clean.argmax_periods_by_delta.items()
            },
            "calibration": dataclasses.asdict(calibration),
        }
    return record


# ----------------------------------------------------------------- run record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    """Read-only git query in the project; ``None`` when git fails."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"git {' '.join(args)} failed: {error}", file=sys.stderr)
        return None
    return completed.stdout.strip()


@dataclass(frozen=True)
class SealSpec:
    """Seal commit and the sealed files and folders checked against it."""

    commit: str
    files: tuple[str, ...]
    folders: tuple[str, ...]


V1_SEAL = SealSpec(SEAL_COMMIT, SEALED_FILES, (SEALED_FOLDER,))


def v2_seal(commit: str) -> SealSpec:
    """The v2 seal (v2 section 4, H7); the commit is a run argument."""
    return SealSpec(commit, V2_SEALED_FILES, V2_SEALED_FOLDERS)


def sealed_blobs(seal: SealSpec = V1_SEAL) -> dict[str, dict[str, object]]:
    """Seal-commit blob and working-tree blob of every sealed file (4-2-0)."""
    listing = _git("ls-tree", "-r", seal.commit, "--", *seal.files, *seal.folders)
    blobs: dict[str, dict[str, object]] = {}
    for entry in (listing or "").splitlines():
        info, path = entry.split("\t", 1)
        blobs[path] = {"blob_at_seal": info.split()[2]}
    for path in seal.files:
        blobs.setdefault(path, {"blob_at_seal": None})
    for path, record in blobs.items():
        exists = (PROJECT_ROOT / path).exists()
        working = _git("hash-object", "--", path) if exists else None
        record["blob_working_tree"] = working
        record["matches"] = working is not None and working == record["blob_at_seal"]
    return blobs


def seal_record(seal: SealSpec = V1_SEAL) -> dict[str, object]:
    """Seal and working-tree state; failed git queries are listed, not hidden.

    ``git_failures`` names every query that returned nothing (a failed
    ``git status`` must not read as "no dirty files").
    """
    queries = {
        "seal_commit_full": ("rev-parse", seal.commit),
        "seal_commit_time": ("show", "-s", "--format=%cI", seal.commit),
        "head_commit": ("rev-parse", "HEAD"),
        "harness_blob_working_tree": (
            "hash-object", "--", "analysis/processing/m210b_s4s6_pipeline.py",
        ),
        "status": ("status", "--porcelain"),
    }  # fmt: skip
    answers = {name: _git(*args) for name, args in queries.items()}
    # A clean tree prints "" for ``git status --porcelain``: only None is a failure.
    git_failures = [name for name, answer in answers.items() if answer is None]
    status = answers.pop("status")
    return {
        "seal_commit": seal.commit,
        **answers,
        "sealed_files": sealed_blobs(seal),
        "dirty_files": [] if status is None else status.splitlines(),
        "git_failures": git_failures,
        "taken_at": _now(),
    }


def input_record(data_root: Path, models_json: Path) -> dict:
    """SHA-256 of the models json and of every file directly in the data folder.

    The exports (npy, json, manifest) and anything else written there are
    covered, so a file added or changed during the run is seen (pre-run audit).
    """
    files = {"models_json": models_json}
    for path in sorted(p for p in data_root.iterdir() if p.is_file()):
        files[path.name] = path
    return {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in files.items()
    }


def wrapper_regression(out_dir: Path, activities: str) -> dict[str, object]:
    """Rank-1 condition: the wrapper defaults regenerate the 4TU csvs byte for byte.

    The generator's ``main`` runs with its ``OUT`` constant pointed at
    ``out_dir``; the committed csvs are only read. ``activities`` is ``all``,
    ``none`` (not run) or a comma list.
    """
    if activities == "none":
        return {"run": False}
    out_dir.mkdir(parents=True, exist_ok=True)
    original_out, original_argv = generator.OUT, sys.argv
    wanted = "" if activities == "all" else activities
    try:
        generator.OUT = out_dir
        sys.argv = ["fourtu_apex_candidate_select", "--activities", wanted]
        generator.main()
    finally:
        generator.OUT, sys.argv = original_out, original_argv
    record: dict[str, object] = {"run": True, "activities": activities}
    for name in ("fourtu_apex_candidates_v1.csv", "fourtu_apex_candidate_lines_v1.csv"):
        committed, regenerated = original_out / name, out_dir / name
        record[name] = {
            "committed_sha256": _sha256(committed),
            "regenerated_sha256": _sha256(regenerated),
            "byte_identical": committed.read_bytes() == regenerated.read_bytes(),
        }
    return record


def protocol_record(
    plan: HarnessPlan, column_definitions: Mapping[str, str]
) -> dict[str, object]:
    return {
        "dt_harness_ns": DT_HARNESS_NS,
        "plan": dataclasses.asdict(plan),
        "seed_formula": f"{plan.seed_base} + 1000 p + 100 q + s",
        "pair_index_p": dict(plan.pair_index),
        "rho_index_q": {str(rho): q for q, rho in enumerate(plan.rhos)},
        "unit_noise_seed": plan.unit_noise_seed,
        "no_pipe_calibration_pair": plan.no_pipe_calibration_pair,
        "ramp": f"sigma_raw ({RAMP_START} + {RAMP_SLOPE} t / t_end), rho {RAMP_RHO}",
        "original_dt": {"rules": list(plan.original_dt_rules), "rho": ORIGINAL_DT_RHO},
        "apex_window": f"trace +-{APEX_TRACES}, t_a0 +-{APEX_PERIODS}/f_c",
        "early_lobe_window": "[t_a0 - 1.5/f_c, t_a0 - 0.5/f_c), trace +-3",
        "false_deep_min_m": FALSE_DEEP_MIN_M,
        "record_end_ns": RECORD_END_NS,
        "columns": dict(column_definitions),
    }


def environment_record() -> dict[str, object]:
    """Interpreter, library versions and thread variables (pre-run audit)."""
    return {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "skimage": skimage.__version__,
        "thread_variables": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "interpretation_doc": {
            "path": INTERPRETATION_DOC,
            "blob_at_head": _git("rev-parse", f"HEAD:{INTERPRETATION_DOC}"),
            "blob_working_tree": _git("hash-object", "--", INTERPRETATION_DOC),
        },
        "qc_constants": {
            "path": QC_CONSTANTS_JSON.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(QC_CONSTANTS_JSON),
        },
    }


# ----------------------------------------------------------------- guards


def _record_failures(record: Mapping, when: str) -> list[str]:
    """Git failures, sealed-file mismatches and dirt in one seal record."""
    failures = []
    if record.get("git_failures"):
        failures.append(f"git queries failed at {when}: {record['git_failures']}")
    if not record["sealed_files"]:
        failures.append(f"no sealed files listed at the seal commit ({when})")
    mismatched = [p for p, r in record["sealed_files"].items() if not r["matches"]]
    if mismatched:
        failures.append(f"sealed files differ from the seal at {when}: {mismatched}")
    if record["dirty_files"]:
        failures.append(f"dirty working tree at {when}: {record['dirty_files']}")
    return failures


def start_guard_failures(start: Mapping, head_must_be_seal: bool) -> list[str]:
    """Checked before any model runs; a failure ends the run at once (exit 1).

    ``head_must_be_seal`` (v2): the run commit is the seal commit itself.
    """
    failures = _record_failures(start, "start")
    if head_must_be_seal and (
        start.get("seal_commit_full") is None
        or start.get("seal_commit_full") != start.get("head_commit")
    ):
        failures.append(
            f"HEAD {start.get('head_commit')} is not the seal commit "
            f"{start.get('seal_commit_full')}"
        )
    return failures


def end_guard_failures(
    start: Mapping, end: Mapping, inputs_changed: Sequence[str], regression: Mapping
) -> list[str]:
    """Checked after the runs (exit 1 with the outputs written)."""
    failures = _record_failures(end, "end")
    if start["head_commit"] is None or start["head_commit"] != end["head_commit"]:
        failures.append(f"HEAD changed: {start['head_commit']} -> {end['head_commit']}")
    if end["sealed_files"] != start["sealed_files"]:
        failures.append("sealed file records changed during the run")
    if inputs_changed:
        failures.append(f"inputs changed during the run: {list(inputs_changed)}")
    if regression.get("run"):
        differing = [
            name
            for name, record in regression.items()
            if isinstance(record, dict) and not record.get("byte_identical", True)
        ]
        if differing:
            failures.append(f"wrapper regression differs: {differing}")
    return failures


def guard_failures(
    start: Mapping, end: Mapping, inputs_changed: Sequence[str], regression: Mapping
) -> list[str]:
    """Start and end guards together (start without the v2 HEAD rule)."""
    start_failures = start_guard_failures(start, head_must_be_seal=False)
    return start_failures + end_guard_failures(start, end, inputs_changed, regression)


def write_identity_file(path: Path, record: Mapping) -> None:
    """``pre_run.txt`` / ``post_run.txt``: time, HEAD and dirty files (H7)."""
    lines = [
        f"time {record['taken_at']}",
        f"HEAD {record['head_commit']}",
        f"dirty {len(record['dirty_files'])}",
        *record["dirty_files"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------- run setup


@dataclass(frozen=True)
class RunSetup:
    """Everything a protocol fixes before the first run."""

    protocol: str
    plan: HarnessPlan
    model_ids: tuple[str, ...]
    data_root: Path
    models_json: Path
    out_dir: Path
    runs_csv_name: str
    seal: SealSpec
    column_definitions: Mapping[str, str]
    excluded_models: tuple[str, ...] = ()


def load_v2_models(models_json: Path) -> tuple[list[str], list[str], str | None]:
    """Models with ``batch == "v2"`` (json order), the excluded ones, and the
    optional json ``no_pipe_calibration_pair``.

    Raises:
        ValueError: A v2 model whose pair is not one of v2 H2, or an S6P whose
            S4N control is not a v2 model.
    """
    meta = json.loads(models_json.read_text(encoding="utf-8"))
    included = [m["id"] for m in meta["models"] if m.get("batch") == V2_BATCH]
    excluded = [m["id"] for m in meta["models"] if m.get("batch") != V2_BATCH]
    labels = {m["id"]: m.get("pair") for m in meta["models"]}
    for model_id in included:
        if model_pair(model_id) not in V2_PAIR_INDEX:
            raise ValueError(f"{model_id}: pair {model_pair(model_id)} not in v2 H2")
        expected_label = V2_PAIR_LABELS[V2_PAIR_INDEX[model_pair(model_id)]]
        if labels[model_id] not in (None, expected_label):
            raise ValueError(
                f"{model_id}: json pair {labels[model_id]} is not {expected_label}"
            )
        pair = model_pair(model_id)
        needs_control = model_id.startswith("S6P_") and not pair.startswith(
            NO_PIPE_PAIR_PREFIX
        )
        if needs_control and control_id(pair) not in included:
            raise ValueError(
                f"{model_id}: control {control_id(pair)} is not a v2 model"
            )
    return included, excluded, meta.get("no_pipe_calibration_pair")


def v2_setup(args: argparse.Namespace, plan: HarnessPlan | None) -> RunSetup:
    if args.models is not None:
        raise SystemExit("--models is not used with --protocol v2 (json batch v2)")
    if not args.seal_commit:
        raise SystemExit("--protocol v2 needs --seal-commit (the v2 seal commit)")
    models_json = args.models_json or V2_MODELS_JSON
    model_ids, excluded, json_pair = load_v2_models(models_json)
    no_pipe_pair = json_pair or args.no_pipe_calibration_pair
    if json_pair and args.no_pipe_calibration_pair not in (None, json_pair):
        raise SystemExit("--no-pipe-calibration-pair differs from the json")
    has_no_pipe = any(model_pair(m).startswith(NO_PIPE_PAIR_PREFIX) for m in model_ids)
    if has_no_pipe and not no_pipe_pair:
        raise SystemExit(
            "v2 no-pipe line needs a calibration pair: json "
            "no_pipe_calibration_pair or --no-pipe-calibration-pair "
            "(not fixed by the v2 spec)"
        )
    base_plan = V2_PLAN if plan is None else plan
    if no_pipe_pair:
        base_plan = dataclasses.replace(
            base_plan, no_pipe_calibration_pair=no_pipe_pair
        )
    return RunSetup(
        protocol=PROTOCOL_V2,
        plan=base_plan,
        model_ids=tuple(model_ids),
        data_root=args.data_root or V2_DEFAULT_DATA_ROOT,
        models_json=models_json,
        out_dir=args.out_dir or V2_DEFAULT_OUT_DIR,
        runs_csv_name=V2_RUNS_CSV,
        seal=v2_seal(args.seal_commit),
        column_definitions=V2_COLUMN_DEFINITIONS,
        excluded_models=tuple(excluded),
    )


def v1_setup(args: argparse.Namespace, plan: HarnessPlan | None) -> RunSetup:
    if args.seal_commit not in (None, SEAL_COMMIT):
        raise SystemExit(f"protocol v1.4 is sealed at {SEAL_COMMIT}")
    return RunSetup(
        protocol=PROTOCOL_V1,
        plan=DEFAULT_PLAN if plan is None else plan,
        model_ids=tuple((args.models or ",".join(BATCH1_MODELS)).split(",")),
        data_root=args.data_root or DEFAULT_DATA_ROOT,
        models_json=args.models_json or DEFAULT_MODELS_JSON,
        out_dir=args.out_dir or DEFAULT_OUT_DIR,
        runs_csv_name=V1_RUNS_CSV,
        seal=V1_SEAL,
        column_definitions=COLUMN_DEFINITIONS,
    )


def seed_record(plan: HarnessPlan, protocol: str) -> dict[str, object]:
    """v2 H2: assert no shared seed with batch 1; ``seed_overlap`` = 0."""
    if protocol != PROTOCOL_V2:
        return {"seed_overlap": None, "note": "batch-1 protocol, nothing to compare"}
    seeds = plan_seeds(plan)
    overlap = assert_no_seed_overlap(seeds, plan_seeds(DEFAULT_PLAN))
    return {
        "seed_overlap": overlap,
        "batch1_seed_set": (
            "20261120 + 1000 p + 100 q + s, p 0-4, q 0-1, s 0-9; 20261199"
        ),
        "n_v2_seeds": len(seeds),
        "v2_seed_range": [min(seeds), max(seeds)],
        "prototype_seed_overlap": len(seeds & set(PROTOTYPE_SEEDS)),
    }


# ----------------------------------------------------------------- entry point


def write_runs(
    path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--protocol", choices=(PROTOCOL_V1, PROTOCOL_V2), default=PROTOCOL_V1
    )
    parser.add_argument(
        "--data-root",
        "--input-dir",
        dest="data_root",
        type=Path,
        help=f"exported lines (default {DEFAULT_DATA_ROOT}; v2 {V2_DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--models-json",
        type=Path,
        help=f"default {DEFAULT_MODELS_JSON}; v2 {V2_MODELS_JSON}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help=f"default {DEFAULT_OUT_DIR}; v2 {V2_DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--models", help="v1.4 only: comma list (default batch 1); v2 uses the json"
    )
    parser.add_argument(
        "--seal-commit", help=f"seal commit (v1.4: {SEAL_COMMIT}; v2: required)"
    )
    parser.add_argument(
        "--no-pipe-calibration-pair",
        help="v2 only, when the json has no no_pipe_calibration_pair",
    )
    parser.add_argument(
        "--regression-activities",
        default="all",
        help="4TU activities for the wrapper regression: all, none or a comma list",
    )
    return parser.parse_args(argv)


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def write_start_failure(
    setup: RunSetup, start: Mapping, inputs: Mapping, failures: Sequence[str]
) -> None:
    """run_info of a run stopped by the start guards (no model was run)."""
    run_info = {
        "protocol": setup.protocol,
        "started_at": start["taken_at"],
        "seal": start,
        "inputs": dict(inputs),
        "environment": environment_record(),
        "guards": {"pass": False, "stage": "start", "failures": list(failures)},
        "n_runs": 0,
    }
    (setup.out_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=1, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    for failure in failures:
        print(f"START GUARD FAILED: {failure}", file=sys.stderr, flush=True)


def main(
    argv: Sequence[str] | None = None,
    plan: HarnessPlan | None = None,
    loader: Callable[[str, Path], tuple[Radargram, dict]] = load_model,
) -> int:
    """Run the harness; ``plan`` and ``loader`` are replaced by tests only.

    Returns:
        0, or 1 when a run guard failed (outputs are written in both cases).
    """
    args = parse_args(argv)
    setup = (
        v2_setup(args, plan) if args.protocol == PROTOCOL_V2 else v1_setup(args, plan)
    )
    seeds = seed_record(setup.plan, setup.protocol)
    start = seal_record(setup.seal)
    started_at = start["taken_at"]
    print(
        f"m210b_s4s6_pipeline start {started_at} protocol {setup.protocol}", flush=True
    )
    setup.out_dir.mkdir(parents=True, exist_ok=True)
    write_identity_file(setup.out_dir / "pre_run.txt", start)
    inputs = input_record(setup.data_root, setup.models_json)
    start_failures = start_guard_failures(
        start, head_must_be_seal=setup.protocol == PROTOCOL_V2
    )
    if start_failures:
        write_start_failure(setup, start, inputs, start_failures)
        return 1
    models = ModelSet(setup.data_root, setup.plan, loader)
    rows: list[dict[str, object]] = []
    records: dict[str, object] = {}
    started = time.perf_counter()
    for model_id in setup.model_ids:
        model_rows = model_runs(model_id, models)
        rows += model_rows
        records[model_id] = model_record(model_id, models)
        elapsed = time.perf_counter() - started
        print(
            f"{model_id}: {len(model_rows)} runs, elapsed {elapsed:.0f} s", flush=True
        )
    runs_csv = setup.out_dir / setup.runs_csv_name
    write_runs(runs_csv, rows, setup.plan.columns)
    regression = wrapper_regression(
        setup.out_dir / "wrapper_regression", args.regression_activities
    )
    inputs_after = input_record(setup.data_root, setup.models_json)
    changed = sorted(
        name
        for name in set(inputs) | set(inputs_after)
        if inputs.get(name) != inputs_after.get(name)
    )
    end = seal_record(setup.seal)
    write_identity_file(setup.out_dir / "post_run.txt", end)
    failures = end_guard_failures(start, end, changed, regression)
    run_info = {
        "spec": (
            "m210b_selector_floor v1.4 sections 4-2 and 8-3"
            if setup.protocol == PROTOCOL_V1
            else "m210b_selector_floor_v2 section 8-1 (H1-H8) on v1.4"
        )
        + "; no verdict here",
        "protocol": setup.protocol,
        "started_at": started_at,
        "seal": start,
        "seal_at_end": end,
        "seeds": seeds,
        "seed_overlap": seeds["seed_overlap"],
        "models_run": list(setup.model_ids),
        "models_excluded_in_json": list(setup.excluded_models),
        "inputs": inputs,
        "inputs_changed_during_run": changed,
        "mkl_cbwr": os.environ.get("MKL_CBWR"),
        "environment": environment_record(),
        "python": sys.version,
        "numpy": np.__version__,
        "protocol_detail": protocol_record(setup.plan, setup.column_definitions),
        "models": records,
        "n_runs": len(rows),
        "n_runs_without_zero": sum(1 for row in rows if row["t_zero_ns"] == ""),
        "runs_csv": setup.runs_csv_name,
        "runs_csv_sha256": _sha256(runs_csv),
        "wrapper_regression": regression,
        "guards": {"pass": not failures, "failures": failures},
        "finished_at": _now(),
    }
    (setup.out_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=1, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"wrote {runs_csv} and run_info.json", flush=True)
    for failure in failures:
        print(f"GUARD FAILED: {failure}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
