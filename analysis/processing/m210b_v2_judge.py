"""M2-10 (b) v2 judgment of the selector-floor rules (committed before the results).

Criteria: docs/processing/m210b_selector_floor_v2.md (v2.0, section 8-2 contract) on
top of v1.4 (docs/processing/m210b_selector_floor.md, decision ranks 1-9) and the
pre-result decisions D1, D2, P-F1-P-F3
(docs/processing/m210b_harness_interpretation.md). This script only counts; it
changes no criterion. No 4TU data is read. Once a result exists it is not edited
(v2 8-2); a fix is a separate commit noted in the judgment.

* Rank 1 (asserted in code, any false -> rank 1, the rest "reference only"):
  P-F1 (v2) row and cell counts (20 seeds), the json ``batch == "v2"`` model
  list, F4b rows, zero failures, noise-free L keys, ``seed_overlap`` 0; and the
  identity: seal blobs, the v1.4 spec and floor rules at bc9a474 and the
  interpretation document and generator wrapper at 7e843d6, HEAD and dirty
  before and after, seal time < harness start, ``runs_v2.csv`` SHA-256, harness
  guards, 4TU wrapper regression; and every ``seed`` on 20271120 + 1000 p +
  100 q + s (s 0-19) with the run_info seed formula and unit-noise seed 20271199.
* R (Q, G, C, e) and ranks 2-9 as v1.4 with ``rho_real`` and ``Lambda`` (A0_self)
  and rate thresholds (V2): Q all 20 seeds >= 5, G Lambda(Delta_e) > 10 and
  floor-seed share >= 0.5 and Lambda(Delta_l) <= 10 (V9; the excluded pairs go to
  ``G_excluded_no_escape``, report only), C |gap| <= 0.1, R e >= 0.5.
* Per rule (rank 4/5): F1-F4, F4b as v1.4 and F5-B (sections 1-3, 1-4 S1) for
  B_full and B_min; B_noise and C_none get the same numbers as a report. Blank
  cells are never filled: they fail or make a check "not judgeable" (= fail).
* Reports: ``F5B_by_scope`` (S2, S3 and more), the F5-B symmetry check, F5-C
  depth error, W5 deep-side loss, F6, original-dt runs, M0, P-F3 sensitivity.

Inputs (read only): data/processing/m210b_s4s6_v2/runs_v2.csv, run_info.json,
pre_run.txt, post_run.txt, analysis/synthetic/s4s6/models_s4s6_v2.json.
Output: docs/processing/m210b_v2_judgment.json (counts, decision, identity).

Run: conda run -n gpr-dev python -I analysis/processing/m210b_v2_judge.py
"""

import argparse
import datetime
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "data" / "processing" / "m210b_s4s6_v2"
DEFAULT_MODELS_JSON = ROOT / "analysis" / "synthetic" / "s4s6" / "models_s4s6_v2.json"
DEFAULT_OUT = ROOT / "docs" / "processing" / "m210b_v2_judgment.json"
RUNS_CSV = "runs_v2.csv"
V1_SEAL = "bc9a474"
INTERP_COMMIT = "7e843d6"
V1_DOC = "docs/processing/m210b_selector_floor.md"
INTERP_DOC = "docs/processing/m210b_harness_interpretation.md"
FLOOR_RULES = "analysis/processing/m210b_floor_rules.py"
GENERATOR = "analysis/processing/fourtu_apex_candidate_select.py"
REFERENCE_BLOB_COMMITS = {
    V1_DOC: V1_SEAL,
    INTERP_DOC: INTERP_COMMIT,
    FLOOR_RULES: V1_SEAL,
    GENERATOR: INTERP_COMMIT,
}
"""Files whose v2-seal blob must equal their blob at these commits (rank 1)."""
V2_BATCH = "v2"
SEED_BASE = 20271120
SEED_FORMULA = f"{SEED_BASE} + 1000 p + 100 q + s"
UNIT_NOISE_SEED = 20271199
PAIR_INDEX = {"D14_S040": 0, "D16_S040": 1, "D15_S045": 2, "D13_S010": 3, "NP_S040": 4}
"""v2 H2: p of each depth/sigma pair (P0-P3, NP)."""
JOINT_SUFFIX = re.compile(r"_J\d+$")
LINE_PREFIX = "S4S6/"

JUDGED = ("B_full", "B_min")
"""W4: judged candidates, in the U1 order."""
REPORTED = ("B_noise", "C_none")
CURRENT = "current"
RULES = (CURRENT, *JUDGED, *REPORTED)
DELTAS = (-1.0, 0.0, 1.0, 2.35)
LATE_DELTA = 2.35
PAIRS = ((0.0, LATE_DELTA), (-1.0, LATE_DELTA))
RHOS = (10, 20)
N_SEEDS = 20
DT_HARNESS_NS = 0.098
TOL = 1e-9

RATE_GAP_MAX = 0.1  # C, F1, F2, F3
EFFECT_MIN = 0.5  # R
LAMBDA_MIN = 10.0  # G
RHO_MIN = 5.0  # Q
FLOOR_SHARE_MIN = 0.5  # G (V2: share of seeds)
F4_MARGIN = 0.5  # F4, F4b
S4N_RATE_MIN = 0.5  # rank 2
F5B_TIME_NS = 0.5 * DT_HARNESS_NS  # 0.049 ns
F5B_TRACES = 1
F5B_MIN_PAIRS = 20
DEEP_LOSS_REPORT = 0.2  # W5 (report only)
G_LAMBDA_E_LOW = "lambda_e_le_10"
G_LAMBDA_E_BLANK = "lambda_e_blank"
G_FLOOR_FRACTION_LOW = "floor_frac_lt_0.5"
G_NO_ESCAPE = "lambda_l_gt_10"
"""V9: the late mute does not leave the floor region (``G_excluded_no_escape``)."""
G_LAMBDA_L_BLANK = "lambda_l_blank"
"""Not met, but kept out of ``G_excluded_no_escape``."""


# ----------------------------------------------------------------- loading


def as_bool(series: pd.Series) -> pd.Series:
    """'True' -> True; anything else (False, blank) -> False (P-F1: blank = not met)."""
    return series.astype(str).str.strip() == "True"


def as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", float("nan")), errors="coerce")


def prepare_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Typed helper columns next to the csv strings."""
    runs = runs.copy()
    runs["delta"] = as_float(runs["delta_ns"]).round(4)
    runs["rho_i"] = as_float(runs["rho"]).astype(int)
    runs["dt"] = as_float(runs["dt_ns"])
    runs["ramp"] = runs["ramp_noise"].astype(str).str.strip() == "1"
    for column in ("accepted_apex", "kept_apex", "floor_held_at_apex"):
        runs[column + "_b"] = as_bool(runs[column])
    for column in (
        "rho_real", "rho_real_cal", "Lambda", "n_false_deep", "argmax_level_periods",
        "kept_nearest_trace", "kept_nearest_t_ns", "depth_nearest_m",
    ):  # fmt: skip
        runs[column + "_f"] = as_float(runs[column])
    return runs


def v2_models_from_json(models_json: Path) -> list[str]:
    meta = json.loads(models_json.read_text(encoding="utf-8"))
    return [m["id"] for m in meta["models"] if m.get("batch") == V2_BATCH]


@dataclass(frozen=True)
class ModelRoles:
    """Pipe models, S6P -> S4N controls and the no-pipe lines of a run."""

    pipe: tuple[str, ...]
    s6p_control: dict[str, str]
    no_pipe: tuple[str, ...]

    @property
    def s4n(self) -> tuple[str, ...]:
        return tuple(m for m in self.pipe if m.startswith("S4N_"))


def model_roles(models: Sequence[str], info: Mapping) -> ModelRoles:
    pipe, no_pipe, control = [], [], {}
    for model_id in models:
        harness = info["models"][model_id]["harness"]
        if harness.get("apex_trace") is None:
            no_pipe.append(model_id)
            continue
        pipe.append(model_id)
        if model_id.startswith("S6P_"):
            control_id = harness["calibration"]["control_id"]
            control[model_id] = control_id.removeprefix(LINE_PREFIX)
    return ModelRoles(tuple(pipe), control, tuple(no_pipe))


# ----------------------------------------------------------------- P-F1


def _cells_with_wrong_seeds(rows: pd.DataFrame, keys: Sequence[str]) -> tuple:
    counts = rows.groupby(list(keys))["seed"].agg(["count", "nunique"])
    wrong = [
        str(key)
        for key, value in counts.iterrows()
        if not (value["count"] == N_SEEDS and value["nunique"] == N_SEEDS)
    ]
    return set(counts.index), wrong


def expected_run_count(roles: ModelRoles) -> int:
    per_model = len(RHOS) * len(RULES) * len(DELTAS) * N_SEEDS
    original_dt = 2 * N_SEEDS  # current, B_full at rho 10, Delta 0 (H3)
    ramp = len(RULES) * len(DELTAS) * N_SEEDS
    models = len(roles.pipe) + len(roles.no_pipe)
    return models * (per_model + original_dt) + len(roles.no_pipe) * ramp


def pf1_checks(
    runs: pd.DataFrame, info: Mapping, v2_models: Sequence[str]
) -> tuple[dict, pd.DataFrame, pd.DataFrame, ModelRoles]:
    """P-F1 (v2) assertions; returns the record, judgment rows, F4b rows, roles."""
    out: dict[str, object] = {}
    run_models = sorted(set(runs["model_id"]))
    out["models_json_v2"] = list(v2_models)
    out["models_in_runs"] = run_models
    out["models_match"] = sorted(v2_models) == run_models and sorted(
        v2_models
    ) == sorted(info.get("models_run", []))
    roles = model_roles([m for m in v2_models if m in info["models"]], info)
    is_harness_dt = runs["dt"].sub(DT_HARNESS_NS).abs() < 1e-12
    judgment = runs[is_harness_dt & ~runs["ramp"]]
    ramp = runs[is_harness_dt & runs["ramp"]]
    keys = ("model_id", "rho_i", "rule", "delta")
    got, wrong = _cells_with_wrong_seeds(judgment, keys)
    expected = {
        (m, r, u, d) for m in v2_models for r in RHOS for u in RULES for d in DELTAS
    }
    out["judgment_rows"] = len(judgment)
    out["cells_expected"] = len(expected)
    out["cells_missing"] = sorted(map(str, expected - got))
    out["cells_extra"] = sorted(map(str, got - expected))
    out["cells_not_20_seeds"] = wrong
    ramp_got, ramp_wrong = _cells_with_wrong_seeds(ramp, keys)
    ramp_expected = {
        (m, 10, u, d) for m in roles.no_pipe for u in RULES for d in DELTAS
    }
    out["ramp_cells_missing_or_extra"] = sorted(map(str, ramp_got ^ ramp_expected))
    out["ramp_cells_not_20_seeds"] = ramp_wrong
    out["n_runs"] = len(runs)
    out["n_runs_expected"] = expected_run_count(roles)
    out["n_runs_info"] = info.get("n_runs")
    out["n_runs_without_zero"] = info.get("n_runs_without_zero")
    out["pipe_models_bad_L_or_apex"] = [
        m
        for m in roles.pipe
        if len(info["models"][m]["harness"].get("L_noise_free_current") or {}) != 4
    ]
    out["seed_overlap"] = info.get("seed_overlap")
    out["pass"] = bool(
        out["models_match"]
        and not out["cells_missing"]
        and not out["cells_extra"]
        and not out["cells_not_20_seeds"]
        and not out["ramp_cells_missing_or_extra"]
        and not out["ramp_cells_not_20_seeds"]
        and out["n_runs"] == out["n_runs_expected"] == out["n_runs_info"]
        and out["n_runs_without_zero"] == 0
        and not out["pipe_models_bad_L_or_apex"]
        and out["seed_overlap"] == 0
    )
    return out, judgment, ramp, roles


def model_pair(model_id: str) -> str:
    """``S6P_D14_S040_J2`` -> ``D14_S040`` (same rule as the harness)."""
    return JOINT_SUFFIX.sub("", model_id.split("_", 1)[1])


def seed_checks(runs: pd.DataFrame, info: Mapping) -> dict:
    """Every ``seed`` is 20271120 + 1000 p + 100 q + s with s in 0-19 (rank 1).

    p comes from the model's pair (v2 H2), q from rho (10 -> 0, 20 -> 1). The
    run_info seed formula and unit-noise seed must be the v2 values as well.
    """
    pair_index = runs["model_id"].map(lambda m: PAIR_INDEX.get(model_pair(m)))
    rho_index = runs["rho_i"].map({rho: q for q, rho in enumerate(RHOS)})
    seed = pd.to_numeric(runs["seed"], errors="coerce")
    s = seed - SEED_BASE - 1000 * pair_index - 100 * rho_index
    is_valid = s.notna() & s.between(0, N_SEEDS - 1) & (s % 1 == 0)
    detail = info.get("protocol_detail") or {}
    return {
        "rows_off_formula": int((~is_valid).sum()),
        "models_without_pair_index": sorted(
            set(runs.loc[pair_index.isna(), "model_id"])
        ),
        "run_info_seed_formula": detail.get("seed_formula"),
        "run_info_unit_noise_seed": detail.get("unit_noise_seed"),
        "pass": bool(
            is_valid.all()
            and detail.get("seed_formula") == SEED_FORMULA
            and detail.get("unit_noise_seed") == UNIT_NOISE_SEED
        ),
    }


def _parse_time(text: object) -> datetime.datetime | None:
    try:
        return datetime.datetime.fromisoformat(str(text))
    except ValueError:
        return None


def rank1_identity(
    info: Mapping, runs_sha_now: str, reference_blobs: Mapping[str, str | None]
) -> dict:
    """Every rank-1 identity condition of v2 8-2, asserted in code.

    ``reference_blobs`` maps the v1.4 spec and the interpretation document to
    their blobs at ``bc9a474`` and ``7e843d6`` (``git rev-parse``).
    """
    seal, seal_end = info["seal"], info.get("seal_at_end") or {}
    sealed = seal.get("sealed_files") or {}
    checks: dict[str, bool] = {
        "sealed_files_listed": bool(sealed),
        "sealed_blobs_match_run": bool(sealed)
        and all(record.get("matches") for record in sealed.values()),
    }
    for path, reference in reference_blobs.items():
        recorded = (sealed.get(path) or {}).get("blob_at_seal")
        checks[f"blob_{path}"] = reference is not None and recorded == reference
    checks["head_same_start_end"] = seal.get("head_commit") is not None and seal.get(
        "head_commit"
    ) == seal_end.get("head_commit")
    checks["dirty_0_start"] = seal.get("dirty_files") == []
    checks["dirty_0_end"] = seal_end.get("dirty_files") == []
    seal_time = _parse_time(seal.get("seal_commit_time"))
    start_time = _parse_time(info.get("started_at"))
    checks["seal_before_harness_start"] = (
        seal_time is not None and start_time is not None and seal_time < start_time
    )
    checks["runs_csv_sha256_matches"] = info.get("runs_csv_sha256") == runs_sha_now
    checks["seed_overlap_0"] = info.get("seed_overlap") == 0
    checks["harness_guards_pass"] = bool((info.get("guards") or {}).get("pass"))
    regression = info.get("wrapper_regression") or {}
    csv_records = [r for r in regression.values() if isinstance(r, dict)]
    checks["wrapper_regression_identical"] = (
        bool(regression.get("run"))
        and len(csv_records) == 2
        and all(r.get("byte_identical") is True for r in csv_records)
    )
    return {"checks": checks, "pass": all(checks.values())}


# ----------------------------------------------------------------- rates and R


def cell(
    rows: pd.DataFrame, model: str, rho: int, rule: str, delta: float
) -> pd.DataFrame:
    return rows[
        (rows["model_id"] == model)
        & (rows["rho_i"] == rho)
        & (rows["rule"] == rule)
        & (rows["delta"] == delta)
    ]


def rate(rows: pd.DataFrame, model: str, rho: int, rule: str, delta: float) -> float:
    """Accepted-stage success share over the 20 seeds (blank = not met)."""
    return float(cell(rows, model, rho, rule, delta)["accepted_apex_b"].sum()) / N_SEEDS


def _lambda(info: Mapping, model: str, delta: float, use_cal: bool) -> float:
    """Lambda_cal from run_info (P-F3), or the self value of run_info (M0 report)."""
    harness = info["models"][model]["harness"]
    key = f"{delta:+.2f}"
    if use_cal:
        level = (harness.get("L_noise_free_current") or {}).get(key)
        a0 = harness["calibration"]["a0"]
        return float("nan") if level is None else level / a0
    value = (harness.get("Lambda") or {}).get(key)
    return float("nan") if value is None else float(value)


def lambda_of_cell(rows: pd.DataFrame) -> float:
    """The ``Lambda`` column of one (model, rho, current, Delta) cell (v2 8-2 G).

    It is one noise-free value per Delta; a blank or a disagreeing value gives
    NaN, so G is not met.
    """
    values = rows["Lambda_f"]
    if values.empty or values.isna().any() or values.nunique() != 1:
        return float("nan")
    return float(values.iloc[0])


def g_fail_reasons(lam_e: float, floor_share: float, lam_l: float) -> list[str]:
    """Why a pair is outside G (V9, reason names of v2 8-2); empty when G holds.

    A blank Lambda is not met, with its own reason (``lambda_e_blank``,
    ``lambda_l_blank``).
    """
    reasons = []
    if math.isnan(lam_e):
        reasons.append(G_LAMBDA_E_BLANK)
    elif lam_e <= LAMBDA_MIN:
        reasons.append(G_LAMBDA_E_LOW)
    if floor_share < FLOOR_SHARE_MIN - TOL:
        reasons.append(G_FLOOR_FRACTION_LOW)
    if math.isnan(lam_l):
        reasons.append(G_LAMBDA_L_BLANK)
    elif lam_l > LAMBDA_MIN + TOL:
        reasons.append(G_NO_ESCAPE)
    return reasons


def r_table(
    rows: pd.DataFrame, info: Mapping, roles: ModelRoles, use_cal: bool = False
) -> list[dict]:
    """Q, G, C, e per S6P condition and mute pair (current rule)."""
    rho_column = "rho_real_cal_f" if use_cal else "rho_real_f"
    table = []
    for model, control in roles.s6p_control.items():
        for rho in RHOS:
            for delta_e, delta_l in PAIRS:
                early = cell(rows, model, rho, CURRENT, delta_e)
                late = cell(rows, model, rho, CURRENT, delta_l)
                both = pd.concat([early, late])
                q = bool(
                    len(early) == N_SEEDS
                    and len(late) == N_SEEDS
                    and (both[rho_column] >= RHO_MIN - TOL).all()
                )
                if use_cal:
                    lam = _lambda(info, model, delta_e, use_cal)
                    lam_late = _lambda(info, model, delta_l, use_cal)
                else:
                    lam, lam_late = lambda_of_cell(early), lambda_of_cell(late)
                floor_share = float(early["floor_held_at_apex_b"].sum()) / N_SEEDS
                reasons = g_fail_reasons(lam, floor_share, lam_late)
                # V9 (W9): G also needs the late mute to leave the floor region.
                g = not reasons
                is_no_escape = reasons == [G_NO_ESCAPE]
                c_gap = rate(rows, control, rho, CURRENT, delta_l) - rate(
                    rows, control, rho, CURRENT, delta_e
                )
                c = abs(c_gap) <= RATE_GAP_MAX + TOL
                rate_e = rate(rows, model, rho, CURRENT, delta_e)
                rate_l = rate(rows, model, rho, CURRENT, delta_l)
                e = rate_l - rate_e
                table.append(
                    {
                        "model": model, "control": control, "rho": rho,
                        "delta_e": delta_e, "delta_l": delta_l,
                        "rate_e": rate_e, "rate_l": rate_l, "e": round(e, 6),
                        "Q": q, "rho_min": float(both[rho_column].min()),
                        # Unrounded: the G boundary 10 must stay visible.
                        "Lambda": None if math.isnan(lam) else lam,
                        "floor_share_e": floor_share,
                        "Lambda_l": None if math.isnan(lam_late) else lam_late,
                        "G_no_escape": is_no_escape, "G_fail_reasons": reasons,
                        "G": g,
                        "control_gap": round(c_gap, 6), "C": c,
                        "R_pair": bool(q and g and c and e >= EFFECT_MIN - TOL),
                    }
                )  # fmt: skip
    return table


def decide(pf1_pass: bool, rows: pd.DataFrame, roles: ModelRoles, table) -> dict:
    """Decision ranks 1-3 and 6-9 (v1.4); 4/5 go per rule."""
    if not pf1_pass:
        return {"rank": 1, "verdict": "hold (harness identity, P-F1)"}
    s4n_low = [
        (m, rate(rows, m, 20, CURRENT, 0.0))
        for m in roles.s4n
        if rate(rows, m, 20, CURRENT, 0.0) < S4N_RATE_MIN - TOL
    ]
    if s4n_low:
        return {"rank": 2, "verdict": "hold (S4N rho 20 rate < 0.5)", "detail": s4n_low}
    if not any(r["Q"] for r in table):
        return {"rank": 3, "verdict": "hold (no qualified condition Q)"}
    if any(r["R_pair"] for r in table):
        return {"rank": "4/5", "verdict": "R holds; per-rule F checks"}
    qgc = [r for r in table if r["Q"] and r["G"] and r["C"]]
    if qgc:
        if any(r["e"] > RATE_GAP_MAX + TOL for r in qgc):
            return {"rank": 6, "verdict": "weak reproduction, hold"}
        return {"rank": 7, "verdict": "mechanism refuted"}
    if any(r["Q"] and r["G"] for r in table):
        return {"rank": 8, "verdict": "hold (control C fails)"}
    per_pair = [
        {
            "model": r["model"],
            "rho": r["rho"],
            "delta_e": r["delta_e"],
            "delta_l": r["delta_l"],
            "G_fail_reasons": r["G_fail_reasons"],
        }
        for r in table
        if r["Q"]
    ]
    return {
        "rank": 9,
        "verdict": "reproduction condition not met (not a refutation)",
        "G_fail_by_pair": per_pair,
    }


def g_excluded_no_escape(table: Sequence[Mapping]) -> list[dict]:
    """REPORT ONLY (V9): pairs left out of G because Lambda(Delta_l) > 10, with e."""
    keys = ("model", "rho", "delta_e", "delta_l", "Q", "C", "e", "Lambda", "Lambda_l")
    return [{k: r[k] for k in keys} for r in table if r["G_no_escape"]]


# ----------------------------------------------------------------- F checks


def false_deep_check(
    values_x: pd.Series, values_cur: pd.Series, n_expected: int = N_SEEDS
) -> dict:
    """F4/F4b on one condition: mean(x) <= mean(current) + 0.5 over its seeds.

    A blank count (or a condition without ``n_expected`` rows) is "not judgeable"
    and fails: blanks are never read as 0 (v2 8-2).
    """
    is_complete = (
        len(values_x) == n_expected
        and len(values_cur) == n_expected
        and values_x.notna().all()
        and values_cur.notna().all()
    )
    if not is_complete:
        n_blank = int(values_x.isna().sum() + values_cur.isna().sum())
        return {"judgeable": False, "pass": False, "n_blank": n_blank}
    mean_x, mean_c = float(values_x.mean()), float(values_cur.mean())
    return {
        "judgeable": True,
        "mean_x": round(mean_x, 3),
        "mean_cur": round(mean_c, 3),
        "pass": bool(mean_x <= mean_c + F4_MARGIN + TOL),
    }


def f1_to_f4(rows, ramp, roles, table, x: str) -> dict:
    """F1-F4 and F4b of v1.4 for rule ``x`` against current."""
    f1 = []
    for model in roles.pipe:
        for rho in RHOS:
            base = rate(rows, model, rho, x, 0.0)
            for delta in (-1.0, 1.0, LATE_DELTA):
                gap = rate(rows, model, rho, x, delta) - base
                if abs(gap) > RATE_GAP_MAX + TOL:
                    f1.append(
                        {
                            "model": model,
                            "rho": rho,
                            "delta": delta,
                            "gap": round(gap, 6),
                        }
                    )
    f2 = []
    for row in table:
        if not row["R_pair"]:
            continue
        rate_x = rate(rows, row["model"], row["rho"], x, row["delta_e"])
        rate_ref = rate(rows, row["model"], row["rho"], CURRENT, row["delta_l"])
        if rate_x < rate_ref - RATE_GAP_MAX - TOL:
            f2.append(
                {
                    "model": row["model"],
                    "rho": row["rho"],
                    "delta_e": row["delta_e"],
                    "rate_x_e": rate_x,
                    "rate_cur_l": rate_ref,
                }
            )
    f3 = []
    for model in roles.s4n:
        for rho in RHOS:
            rate_x, rate_c = (
                rate(rows, model, rho, x, 0.0),
                rate(rows, model, rho, CURRENT, 0.0),
            )
            if rate_x < rate_c - RATE_GAP_MAX - TOL:
                f3.append(
                    {"model": model, "rho": rho, "rate_x": rate_x, "rate_cur": rate_c}
                )
    f4 = []
    for model in (*roles.no_pipe, *roles.s4n):
        for rho in RHOS:
            check = false_deep_check(
                cell(rows, model, rho, x, 0.0)["n_false_deep_f"],
                cell(rows, model, rho, CURRENT, 0.0)["n_false_deep_f"],
            )
            if not check["pass"]:
                f4.append({"model": model, "rho": rho, **check})
    f4b = {}
    for delta in DELTAS:
        f4b[f"{delta:+.2f}"] = false_deep_check(
            ramp[(ramp["rule"] == x) & (ramp["delta"] == delta)]["n_false_deep_f"],
            ramp[(ramp["rule"] == CURRENT) & (ramp["delta"] == delta)][
                "n_false_deep_f"
            ],
            N_SEEDS * len(roles.no_pipe),
        )
    return {
        "F1_fail": f1, "F2_fail": f2, "F3_fail": f3, "F4_fail": f4,
        "F4b": f4b, "F4b_fail": [] if f4b["+0.00"]["pass"] else [f4b["+0.00"]],
    }  # fmt: skip


def nearest_errors(rows: pd.DataFrame, info: Mapping, roles: ModelRoles, rule: str):
    """Per (model, rho, Delta, seed): eps_t (ns) and eps_j (traces) of the nearest."""
    part = rows[(rows["rule"] == rule) & rows["model_id"].isin(roles.pipe)]
    part = part[
        part["kept_nearest_t_ns_f"].notna() & part["kept_nearest_trace_f"].notna()
    ]
    harness = {m: info["models"][m]["harness"] for m in roles.pipe}
    t_a0 = part["model_id"].map({m: h["t_a0_ns"] for m, h in harness.items()})
    apex = part["model_id"].map({m: h["apex_trace"] for m, h in harness.items()})
    return pd.DataFrame(
        {
            "model_id": part["model_id"], "rho_i": part["rho_i"],
            "delta": part["delta"], "seed": part["seed"],
            "eps_t": (part["kept_nearest_t_ns_f"] - t_a0).abs(),
            "eps_j": (part["kept_nearest_trace_f"] - apex).abs(),
        }
    )  # fmt: skip


def f5b_counts(pairs: pd.DataFrame) -> dict:
    """F5-B on paired errors (columns eps_t_x, eps_j_x, eps_t_c, eps_j_c; v2 1-3).

    Pass iff worse <= better, median eps_t(x) <= median eps_t(cur) + 0.049 ns and
    at least 20 pairs; fewer pairs is "not judgeable" and counts as a failure.
    """
    worse = (pairs["eps_t_x"] > pairs["eps_t_c"] + F5B_TIME_NS + TOL) | (
        pairs["eps_j_x"] > pairs["eps_j_c"] + F5B_TRACES + TOL
    )
    better = (pairs["eps_t_c"] > pairs["eps_t_x"] + F5B_TIME_NS + TOL) | (
        pairs["eps_j_c"] > pairs["eps_j_x"] + F5B_TRACES + TOL
    )
    n_pairs = len(pairs)
    median_x = float(pairs["eps_t_x"].median()) if n_pairs else float("nan")
    median_c = float(pairs["eps_t_c"].median()) if n_pairs else float("nan")
    is_judgeable = n_pairs >= F5B_MIN_PAIRS
    passes = bool(
        is_judgeable
        and int(worse.sum()) <= int(better.sum())
        and median_x <= median_c + F5B_TIME_NS + TOL
    )
    return {
        "n_pairs": n_pairs,
        "n_worse": int(worse.sum()),
        "n_better": int(better.sum()),
        "n_worse_and_better": int((worse & better).sum()),
        "median_eps_t_x_ns": None if math.isnan(median_x) else round(median_x, 5),
        "median_eps_t_cur_ns": None if math.isnan(median_c) else round(median_c, 5),
        "judgeable": is_judgeable,
        "pass": passes,
    }


def paired_errors(rows, info, roles, x: str, reference: str = CURRENT) -> pd.DataFrame:
    keys = ["model_id", "rho_i", "delta", "seed"]
    own = nearest_errors(rows, info, roles, x)
    ref = nearest_errors(rows, info, roles, reference)
    merged = own.merge(ref, on=keys, suffixes=("_x", "_c"))
    return merged


def nearest_defects(rows: pd.DataFrame, roles: ModelRoles, rule: str) -> int:
    """Pipe judgment rows whose nearest-kept columns are inconsistent or blank.

    A blank ``kept_nearest_*`` is only valid with ``n_kept_in_window == 0``; any
    other blank makes F5-B "not judgeable" (v2 8-2, no filling).
    """
    part = rows[(rows["rule"] == rule) & rows["model_id"].isin(roles.pipe)]
    count = as_float(part["n_kept_in_window"])
    has_nearest = (
        part["kept_nearest_t_ns_f"].notna() & part["kept_nearest_trace_f"].notna()
    )
    has_any = part["kept_nearest_t_ns_f"].notna() | part["kept_nearest_trace_f"].notna()
    is_valid = (count.eq(0) & ~has_any) | (count.ge(1) & has_nearest)
    return int((~is_valid).sum())


def f5b(rows, info, roles, x: str) -> dict:
    """F5-B for ``x`` (judgment scope S1) and the symmetry check (current vs ``x``)."""
    defects = nearest_defects(rows, roles, x) + nearest_defects(rows, roles, CURRENT)
    result = f5b_counts(paired_errors(rows, info, roles, x))
    symmetric = f5b_counts(paired_errors(rows, info, roles, CURRENT, reference=x))
    result["n_defect_rows"] = defects
    if defects:
        result["judgeable"], result["pass"] = False, False
    if result["pass"] and symmetric["pass"]:
        reading = "no position difference"
    elif result["pass"]:
        reading = f"{x} better"
    else:
        reading = f"{x} not non-inferior"
    return {**result, "symmetric_current_vs_x": symmetric, "reading": reading}


def f5b_by_scope(rows, info, roles, table, x: str) -> dict:
    """REPORT ONLY (W8): F5-B counts over other scopes; the judgment uses S1."""
    pairs = paired_errors(rows, info, roles, x)
    r_conditions = {(r["model"], r["rho"]) for r in table if r["R_pair"]}
    g_conditions = {(r["model"], r["rho"]) for r in table if r["G"]}
    condition = list(zip(pairs["model_id"], pairs["rho_i"], strict=True))
    scopes = {
        "S1_judgment_all_pipe_models": [True] * len(pairs),
        "S2_R_conditions": [c in r_conditions for c in condition],
        "S3_G_conditions": [c in g_conditions for c in condition],
        "S4N_only": [c[0].startswith("S4N_") for c in condition],
        "S6P_outside_G": [
            c[0].startswith("S6P_") and c not in g_conditions for c in condition
        ],
    }
    return {
        name: f5b_counts(pairs[pd.Series(mask, index=pairs.index, dtype=bool)])
        for name, mask in scopes.items()
    }


def original_dt_report(runs: pd.DataFrame) -> dict:
    """REPORT ONLY (F6): auxiliary original-grid runs (dt != 0.098), not judged."""
    original = runs[(runs["dt"].sub(DT_HARNESS_NS).abs() > 1e-9) & ~runs["ramp"]]
    report: dict[str, object] = {"rows": len(original)}
    for (model, rule), group in original.groupby(["model_id", "rule"]):
        report[f"{model}|{rule}"] = {
            "rows": len(group),
            "accepted_rate": round(float(group["accepted_apex_b"].mean()), 3),
            "n_peaks_median": float(as_float(group["n_peaks"]).median()),
            "cap300_hit_rows": int((group["cap300_hit"] == "True").sum()),
        }
    return report


def rule_checks(rows, ramp, info, roles, table, x: str) -> dict:
    checks = f1_to_f4(rows, ramp, roles, table, x)
    checks["F5B"] = f5b(rows, info, roles, x)
    checks["pass_all"] = not (
        checks["F1_fail"] or checks["F2_fail"] or checks["F3_fail"]
        or checks["F4_fail"] or checks["F4b_fail"] or not checks["F5B"]["pass"]
    )  # fmt: skip
    checks["rank"] = 4 if checks["pass_all"] else 5
    return checks


def u1_choice(by_rule: Mapping[str, Mapping]) -> str:
    """U1: B_full if rank 4, else B_min if rank 4, else keep current."""
    for x in JUDGED:
        if by_rule.get(x, {}).get("rank") == 4:
            return x
    return CURRENT


# ----------------------------------------------------------------- reports


def f5c_depth_report(rows, info, roles) -> dict:
    """F5-C (report): |depth - true top depth| of the nearest kept, x vs current."""
    truth = {m: info["models"][m].get("pipe_top_depth_m") for m in roles.pipe}
    out = {}
    for x in (*JUDGED, *REPORTED):
        keys = ["model_id", "rho_i", "delta", "seed"]
        columns = [*keys, "depth_nearest_m_f"]
        own = rows[rows["rule"] == x][columns]
        ref = rows[rows["rule"] == CURRENT][columns]
        merged = own.merge(ref, on=keys, suffixes=("_x", "_c")).dropna()
        merged = merged[merged["model_id"].isin(roles.pipe)]
        true_depth = merged["model_id"].map(truth).astype(float)
        err_x = (merged["depth_nearest_m_f_x"] - true_depth).abs()
        err_c = (merged["depth_nearest_m_f_c"] - true_depth).abs()
        out[x] = {
            "n_pairs": len(merged),
            "abs_err_m_x_q10_50_90": _deciles(err_x),
            "abs_err_m_cur_q10_50_90": _deciles(err_c),
        }
    return out


def _deciles(values: pd.Series) -> list[float] | None:
    """10/50/90 % quantiles, or ``None`` for an empty series."""
    if values.empty:
        return None
    return [round(float(v), 4) for v in values.quantile([0.1, 0.5, 0.9])]


def deep_side_report(rows, table, roles) -> list[dict]:
    """W5 (report): S6P outside G, loss of x at Delta 0 vs current at the late mute."""
    outside_g = {
        (r["model"], r["rho"]) for r in table if r["delta_e"] == 0.0 and not r["G"]
    }
    flagged = []
    for model, rho in sorted(outside_g):
        late = rate(rows, model, rho, CURRENT, LATE_DELTA)
        for x in (*JUDGED, *REPORTED):
            loss = late - rate(rows, model, rho, x, 0.0)
            if loss >= DEEP_LOSS_REPORT - TOL:
                flagged.append(
                    {
                        "model": model,
                        "rho": rho,
                        "rule": x,
                        "rate_cur_late": late,
                        "loss": round(loss, 6),
                    }
                )
    return flagged


def f6_report(rows, roles) -> dict:
    pipe_rows = rows[rows["model_id"].isin(roles.pipe)]
    by_rule = pipe_rows.groupby("rule")
    return {
        "kept_rate": {u: round(float(g["kept_apex_b"].mean()), 4) for u, g in by_rule},
        "lost_reason": {
            u: g["lost_reason"].value_counts().to_dict() for u, g in by_rule
        },
        "cap300_hit_rows": {
            u: int((g["cap300_hit"] == "True").sum()) for u, g in rows.groupby("rule")
        },
    }


def m0_report(rows, info, roles) -> dict:
    out = {}
    for model in roles.s6p_control:
        harness = info["models"][model]["harness"]
        noisy = {
            f"{d:+.2f}": [
                round(float(v), 3)
                for v in cell(rows, model, 10, CURRENT, d)[
                    "argmax_level_periods_f"
                ].quantile([0.1, 0.5, 0.9])
            ]
            for d in DELTAS
        }
        out[model] = {
            "Lambda_self": harness.get("Lambda"),
            "Lambda_cal": {f"{d:+.2f}": _lambda(info, model, d, True) for d in DELTAS},
            "argmax_periods_noise_free": harness.get("argmax_level_periods_noise_free"),
            "argmax_periods_noisy_current_rho10_q10_50_90": noisy,
            "t_a0_ns": harness.get("t_a0_ns"),
            "A0_self": harness.get("A0_self"),
            "A0_cal": harness["calibration"]["a0"],
        }
    return out


# ----------------------------------------------------------------- whole judgment


def judge(
    runs: pd.DataFrame,
    info: Mapping,
    v2_models: Sequence[str],
    runs_sha_now: str,
    reference_blobs: Mapping[str, str | None],
) -> dict:
    """All counts, the decision and the per-rule ranks (no I/O).

    A failed rank-1 condition (P-F1 or identity) gives rank 1; the rest of the
    decision is then written under ``reference_only`` (not a verdict).
    """
    runs = prepare_runs(runs)
    pf1, rows, ramp, roles = pf1_checks(runs, info, v2_models)
    ident = rank1_identity(info, runs_sha_now, reference_blobs)
    seeds = seed_checks(runs, info)
    is_rank1_clear = pf1["pass"] and ident["pass"] and seeds["pass"]
    out: dict[str, object] = {
        "P_F1": pf1,
        "rank1_identity": ident,
        "rank1_seeds": seeds,
    }
    table = r_table(rows, info, roles)
    out["R_table"] = table
    out["G_excluded_no_escape"] = g_excluded_no_escape(table)
    out["R"] = any(r["R_pair"] for r in table)
    out["decision"] = decide(is_rank1_clear, rows, roles, table)
    unchecked = decide(True, rows, roles, table)
    per_rule: dict[str, object] = {}
    if unchecked["rank"] == "4/5":
        by_rule = {x: rule_checks(rows, ramp, info, roles, table, x) for x in RULES[1:]}
        per_rule = {
            "F_by_rule": by_rule,
            "rank_by_rule": {x: by_rule[x]["rank"] for x in JUDGED},
            "rank_by_rule_reported": {x: by_rule[x]["rank"] for x in REPORTED},
            "U1": u1_choice(by_rule),
        }
        per_rule["F5B_by_scope"] = {
            x: f5b_by_scope(rows, info, roles, table, x) for x in RULES[1:]
        }
    if is_rank1_clear:
        out.update(per_rule)
        out.setdefault("U1", CURRENT)
    else:
        out["U1"] = CURRENT
        out["reference_only"] = {"decision_if_rank1_clear": unchecked, **per_rule}
    out["original_dt"] = original_dt_report(runs)
    out["rates"] = {
        f"{m}|{rho}|{u}": {f"{d:+.2f}": rate(rows, m, rho, u, d) for d in DELTAS}
        for m in roles.pipe
        for rho in RHOS
        for u in RULES
    }
    out["F5C_depth"] = f5c_depth_report(rows, info, roles)
    out["W5_deep_side"] = deep_side_report(rows, table, roles)
    out["F6"] = f6_report(rows, roles)
    out["M0"] = m0_report(rows, info, roles)
    table_cal = r_table(rows, info, roles, use_cal=True)
    out["P_F3"] = {
        "R_table": table_cal,
        "R": any(r["R_pair"] for r in table_cal),
        "decision": decide(pf1["pass"], rows, roles, table_cal),
    }
    return out


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), *args], check=True, capture_output=True,
            text=True, encoding="utf-8",
        )  # fmt: skip
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"git {' '.join(args)} failed: {error}", file=sys.stderr)
        return None
    return completed.stdout.strip()


def identity(info: Mapping, run_dir: Path, runs_sha_now: str) -> dict:
    seal = info["seal"]

    def read(name: str) -> str | None:
        path = run_dir / name
        return path.read_text(encoding="utf-8").strip() if path.exists() else None

    return {
        "v2_seal_commit": seal["seal_commit_full"],
        "v2_seal_commit_time": seal["seal_commit_time"],
        "sealed_files": {k: v["blob_at_seal"] for k, v in seal["sealed_files"].items()},
        "v1_doc_blob_at_bc9a474": _git("rev-parse", f"{V1_SEAL}:{V1_DOC}"),
        "interp_doc_blob_at_7e843d6": _git(
            "rev-parse", f"{INTERP_COMMIT}:{INTERP_DOC}"
        ),
        "head_commit_run": seal["head_commit"],
        "harness_started": info["started_at"],
        "harness_finished": info["finished_at"],
        "runs_csv_sha256_info": info["runs_csv_sha256"],
        "runs_csv_sha256_now": runs_sha_now,
        "seed_overlap": info.get("seed_overlap"),
        "pre_run": read("pre_run.txt"),
        "post_run": read("post_run.txt"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--models-json", type=Path, default=DEFAULT_MODELS_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs_path = args.run_dir / RUNS_CSV
    runs = pd.read_csv(runs_path, dtype=str, keep_default_na=False)
    info = json.loads((args.run_dir / "run_info.json").read_text(encoding="utf-8"))
    runs_sha_now = hashlib.sha256(runs_path.read_bytes()).hexdigest()
    reference_blobs = {
        path: _git("rev-parse", f"{commit}:{path}")
        for path, commit in REFERENCE_BLOB_COMMITS.items()
    }
    out = judge(
        runs,
        info,
        v2_models_from_json(args.models_json),
        runs_sha_now,
        reference_blobs,
    )
    out = {
        "note": "v2 judgment counts; criteria m210b_selector_floor_v2 section 8-2",
        "identity": identity(info, args.run_dir, runs_sha_now),
        **out,
    }
    args.out.write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    brief = {k: out[k] for k in ("R", "decision", "U1")}
    brief["P_F1_pass"] = out["P_F1"]["pass"]
    brief["rank_by_rule"] = out.get("rank_by_rule")
    print(json.dumps(brief, ensure_ascii=True, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
