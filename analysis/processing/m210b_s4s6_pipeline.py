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

Run (one core, BelowNormal):
    conda run -n gpr-dev python -I analysis/processing/m210b_s4s6_pipeline.py
    ... --regression-activities none   (skip the 4TU regeneration, ~3-4 min)
"""

import argparse
import csv
import dataclasses
import datetime
import hashlib
import json
import math
import os
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "analysis" / "processing"))

import fourtu_apex_candidate_select as generator  # noqa: E402
import m210b_floor_rules as floor_rules  # noqa: E402

from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402
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

DT_HARNESS_NS = 0.098
RHOS = (10, 20)
SEED_BASE = 20261120
UNIT_NOISE_SEED = 20261199
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


# ----------------------------------------------------------------- plan


@dataclass(frozen=True)
class HarnessPlan:
    """Conditions of one harness run (``DEFAULT_PLAN`` is the 4-2 protocol)."""

    rhos: tuple[int, ...] = RHOS
    n_seeds: int = 10
    rules: tuple[str, ...] = floor_rules.RULES
    deltas_ns: tuple[float, ...] = (-1.0, 0.0, 1.0, 2.35)
    original_dt_rules: tuple[str, ...] = ORIGINAL_DT_RULES


DEFAULT_PLAN = HarnessPlan()


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


def noise_seed(pair_index: int, rho_index: int, seed_index: int) -> int:
    """``20261120 + 1000 p + 100 q + s`` (4-2 rule 3)."""
    return SEED_BASE + 1000 * pair_index + 100 * rho_index + seed_index


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


def unit_noise_sigma(radargram: Radargram, k_lo: int, k_hi: int) -> float:
    """``sigma_u``: processed unit white noise (seed 20261199), search-row MAD.

    Only the processing branch is used (the zero of a pure-noise section is
    meaningless); the search rows come from the control's noise-free zero.
    """
    noise = white_noise(radargram.data.shape, UNIT_NOISE_SEED)
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
    levels = {}
    if t_zero is not None:
        for delta in plan.deltas_ns:
            line, _, _ = generator.apex_candidates_for_radargram(
                radargram, inputs, act=LINE_ACTIVITY, path="clean",
                floor_rule="current", mute_offset_ns=delta,
            )  # fmt: skip
            if "_floor" in line:
                levels[delta] = line["_floor"]["L"]
    return CleanLine(
        radargram,
        inputs.branch.background_removed,
        t_zero,
        None if inputs.zero.zero_rule is None else str(inputs.zero.zero_rule),
        target,
        a0_self,
        t_early,
        levels,
    )


@dataclass(frozen=True)
class Calibration:
    """Noise level of one depth/sigma pair on one time grid (4-2 rule 3)."""

    control_id: str
    a0: float
    sigma_u: float
    sigma_raw_by_rho: dict[int, float]


def calibrate(control: CleanLine, rhos: Sequence[int]) -> Calibration:
    if control.a0_self is None or control.t_zero_ns is None:
        raise ValueError(f"{control.radargram.line_id}: no apex or zero to calibrate")
    k_lo, k_hi = floor_rules.search_rows(
        control.t_zero_ns,
        control.radargram.f_c_hz,
        control.radargram.dt_ns,
        control.radargram.n_samples,
    )
    sigma_u = unit_noise_sigma(control.radargram, k_lo, k_hi)
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
) -> dict[str, object]:
    """One ``runs.csv`` row: the wrapper with ``rule`` and mute shift ``delta_ns``.

    ``inputs`` is ``prepare_depth_inputs(noisy)``; ``clean`` gives the apex
    target and the noise-free ``Lambda``.
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
        row["Lambda"] = round(level_max / clean.a0_self, 4)
    row.update(_outcome_fields(line, kept, dt, target))
    return {column: row[column] for column in RUN_COLUMNS}


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
) -> list[dict[str, object]]:
    """Process one noisy line once and run every rule and mute shift on it."""
    noisy = with_noise(clean.radargram, realisation.noise)
    inputs = prepare_depth_inputs(noisy, DepthConfig())
    rho_real = realised_rho(clean, inputs.branch.background_removed, a0_cal)
    return [
        run_row(realisation, noisy, inputs, clean, rule, delta, rho_real)
        for rule in rules
        for delta in deltas_ns
    ]


# ----------------------------------------------------------------- models


def model_pair(model_id: str) -> str:
    """``S6P_D15_S030`` -> ``D15_S030``."""
    return model_id.split("_", 1)[1]


def calibration_pair(model_id: str) -> str:
    pair = model_pair(model_id)
    return NO_PIPE_CALIBRATION_PAIR if pair == "NP_S010" else pair


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
            self._calibrations[key] = calibrate(control, self.plan.rhos)
        return self._calibrations[key]

    def loaded_ids(self) -> list[str]:
        """Every model read so far (including calibration controls)."""
        return sorted(self._lines)


def control_id(pair: str) -> str:
    return f"{CONTROL_FAMILY}_{pair}"


def model_runs(model_id: str, models: ModelSet) -> list[dict[str, object]]:
    """Steps 4-6 for one model."""
    plan = models.plan
    clean = models.clean(model_id, GRID_HARNESS)
    calibration = models.calibration(calibration_pair(model_id), GRID_HARNESS)
    pair_index = PAIR_INDEX[model_pair(model_id)]
    shape = clean.radargram.data.shape
    rows = []
    for rho_index, rho in enumerate(plan.rhos):
        sigma_raw = calibration.sigma_raw_by_rho[rho]
        for s in range(plan.n_seeds):
            seed = noise_seed(pair_index, rho_index, s)
            noise = sigma_raw * white_noise(shape, seed)
            realisation = Realisation(model_id, rho, seed, False, noise)
            rows += run_realisation(
                realisation, clean, calibration.a0, plan.rules, plan.deltas_ns
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
        seed = noise_seed(pair_index, rho_index, s)
        noise = calibration.sigma_raw_by_rho[RAMP_RHO] * ramp * white_noise(shape, seed)
        realisation = Realisation(model_id, RAMP_RHO, seed, True, noise)
        rows += run_realisation(
            realisation, clean, calibration.a0, plan.rules, plan.deltas_ns
        )
    return rows


def original_dt_runs(model_id: str, models: ModelSet, pair_index: int) -> list[dict]:
    """Step 6: the gprMax dt, calibrated on that grid (report only)."""
    plan = models.plan
    clean = models.clean(model_id, GRID_ORIGINAL)
    calibration = models.calibration(calibration_pair(model_id), GRID_ORIGINAL)
    rho_index = plan.rhos.index(ORIGINAL_DT_RHO)
    shape = clean.radargram.data.shape
    rows = []
    for s in range(plan.n_seeds):
        seed = noise_seed(pair_index, rho_index, s)
        noise = calibration.sigma_raw_by_rho[ORIGINAL_DT_RHO] * white_noise(shape, seed)
        realisation = Realisation(model_id, ORIGINAL_DT_RHO, seed, False, noise)
        rows += run_realisation(
            realisation, clean, calibration.a0, plan.original_dt_rules, (0.0,)
        )
    return rows


def model_record(model_id: str, models: ModelSet) -> dict[str, object]:
    """Noise-free values and the calibration used, per grid (run_info)."""
    record: dict[str, object] = {
        "pair": model_pair(model_id),
        "pair_index_p": PAIR_INDEX[model_pair(model_id)],
        "calibration_pair": calibration_pair(model_id),
    }
    for grid in (GRID_HARNESS, GRID_ORIGINAL):
        clean = models.clean(model_id, grid)
        calibration = models.calibration(calibration_pair(model_id), grid)
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


def sealed_blobs() -> dict[str, dict[str, object]]:
    """Seal-commit blob and working-tree blob of every sealed file (4-2-0)."""
    listing = _git("ls-tree", "-r", SEAL_COMMIT, "--", *SEALED_FILES, SEALED_FOLDER)
    blobs: dict[str, dict[str, object]] = {}
    for entry in (listing or "").splitlines():
        info, path = entry.split("\t", 1)
        blobs[path] = {"blob_at_seal": info.split()[2]}
    for path in SEALED_FILES:
        blobs.setdefault(path, {"blob_at_seal": None})
    for path, record in blobs.items():
        exists = (PROJECT_ROOT / path).exists()
        working = _git("hash-object", "--", path) if exists else None
        record["blob_working_tree"] = working
        record["matches"] = working is not None and working == record["blob_at_seal"]
    return blobs


def seal_record() -> dict[str, object]:
    return {
        "seal_commit": SEAL_COMMIT,
        "seal_commit_full": _git("rev-parse", SEAL_COMMIT),
        "seal_commit_time": _git("show", "-s", "--format=%cI", SEAL_COMMIT),
        "sealed_files": sealed_blobs(),
        "head_commit": _git("rev-parse", "HEAD"),
        "harness_blob_working_tree": _git(
            "hash-object", "--", "analysis/processing/m210b_s4s6_pipeline.py"
        ),
        "dirty_files": (_git("status", "--porcelain") or "").splitlines(),
    }


def input_record(data_root: Path, models_json: Path, model_ids: Sequence[str]) -> dict:
    files = {"models_json": {"path": str(models_json), "sha256": _sha256(models_json)}}
    for model_id in model_ids:
        for suffix in (".npy", ".json"):
            path = data_root / f"{model_id}{suffix}"
            files[f"{model_id}{suffix}"] = {"path": str(path), "sha256": _sha256(path)}
    return files


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


def protocol_record(plan: HarnessPlan) -> dict[str, object]:
    return {
        "dt_harness_ns": DT_HARNESS_NS,
        "plan": dataclasses.asdict(plan),
        "seed_formula": "20261120 + 1000 p + 100 q + s",
        "pair_index_p": PAIR_INDEX,
        "rho_index_q": {str(rho): q for q, rho in enumerate(plan.rhos)},
        "unit_noise_seed": UNIT_NOISE_SEED,
        "no_pipe_calibration_pair": NO_PIPE_CALIBRATION_PAIR,
        "ramp": f"sigma_raw ({RAMP_START} + {RAMP_SLOPE} t / t_end), rho {RAMP_RHO}",
        "original_dt": {"rules": list(plan.original_dt_rules), "rho": ORIGINAL_DT_RHO},
        "apex_window": f"trace +-{APEX_TRACES}, t_a0 +-{APEX_PERIODS}/f_c",
        "early_lobe_window": "[t_a0 - 1.5/f_c, t_a0 - 0.5/f_c), trace +-3",
        "false_deep_min_m": FALSE_DEEP_MIN_M,
        "record_end_ns": RECORD_END_NS,
        "columns": COLUMN_DEFINITIONS,
    }


# ----------------------------------------------------------------- entry point


def write_runs(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RUN_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--models-json", type=Path, default=DEFAULT_MODELS_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--models", default=",".join(BATCH1_MODELS))
    parser.add_argument(
        "--regression-activities",
        default="all",
        help="4TU activities for the wrapper regression: all, none or a comma list",
    )
    return parser.parse_args(argv)


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def main(
    argv: Sequence[str] | None = None,
    plan: HarnessPlan = DEFAULT_PLAN,
    loader: Callable[[str, Path], tuple[Radargram, dict]] = load_model,
) -> int:
    """Run the harness; ``plan`` and ``loader`` are replaced by tests only."""
    args = parse_args(argv)
    started_at = _now()
    print(f"m210b_s4s6_pipeline start {started_at}", flush=True)
    model_ids = args.models.split(",")
    models = ModelSet(args.data_root, plan, loader)
    rows: list[dict[str, object]] = []
    records: dict[str, object] = {}
    started = time.perf_counter()
    for model_id in model_ids:
        model_rows = model_runs(model_id, models)
        rows += model_rows
        records[model_id] = model_record(model_id, models)
        elapsed = time.perf_counter() - started
        print(
            f"{model_id}: {len(model_rows)} runs, elapsed {elapsed:.0f} s", flush=True
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = args.out_dir / "runs.csv"
    write_runs(runs_csv, rows)
    run_info = {
        "spec": "m210b_selector_floor v1.4 sections 4-2 and 8-3; no verdict here",
        "started_at": started_at,
        "seal": seal_record(),
        "inputs": input_record(args.data_root, args.models_json, models.loaded_ids()),
        "mkl_cbwr": os.environ.get("MKL_CBWR"),
        "python": sys.version,
        "numpy": np.__version__,
        "protocol": protocol_record(plan),
        "models": records,
        "n_runs": len(rows),
        "n_runs_without_zero": sum(1 for row in rows if row["t_zero_ns"] == ""),
        "runs_csv_sha256": _sha256(runs_csv),
        "wrapper_regression": wrapper_regression(
            args.out_dir / "wrapper_regression", args.regression_activities
        ),
        "finished_at": _now(),
    }
    (args.out_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=1, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"wrote {runs_csv} and run_info.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
