"""M2-10 K7 judgment of the generator variants (committed before the results).

Criteria: docs/processing/m210_k7_generator.md v1.1 sections 4-5 (K-R, K-F1 to
K-F6), 4-6 (decision ranks 1-7, order a1 -> a1b2, fixed sets, cell counts,
empty-case rules) and 8 (judgment header). This script only counts; it changes
no criterion. No 4TU data is read. Once a result exists it is not edited; a fix
is a separate commit noted in the judgment.

* Rank 1 (``rank1_reasons``, any entry -> rank 1 for every candidate, the rest is
  reference only): the identity (expected seal ``--expected-seal-commit`` = run
  seal = run HEAD at start and end, dirty 0, no failed harness git query, every
  sealed blob matching, ``run_info.protocol == "k7"``, ``sealed_files`` covering
  ``K7_SEALED_FILES``, this script's own working-tree blob = its blob at the
  seal, spec and generator blobs at the seal, seal time < harness start,
  ``runs_k7.csv`` SHA-256, the models json SHA-256 read now = the one in
  ``run_info.inputs.models_json``, harness guards, K7 tests, harness-format
  ``pre_run.txt`` / ``post_run.txt``), G0 (4TU csvs, S1g and (b) v3 default
  paths), the seeds (overlap 0, every seed on 20301112 + 1000 m + 100 q + s),
  the fixed sets (models, role sets, variants, rho, per condition and pipe the
  seeds 0-39 once each, one noise-free row per pipe (one blank-pipe row on
  K7_BG_L), pipe counts, calibration pair K7_BG_L -> K7_P2_D070_L,
  no blank outcome) and the judged cell counts (K-F1 a1 4 / a1b2 6, K-F2 8,
  K-F3 16; an empty set is not a pass).
* Rates per condition (model x rho, noise rows only): per pipe over the 40
  seeds, then the mean over the pipes; the denominator is always 40 x pipes.
* Ranks per candidate: 2 the targeted K-R fails (a1 -> K-R(a), a1b2 -> K-R(b))
  or K-F4 is not judgeable (both K7_PVC cells, or every cell, below 20 paired
  rows); 3 K-F2; 4 K-F3; 5 K-F4; 6 K-F1; 7 none of these. a1b2 is ranked only
  when a1 is rank 7. The adopted variant is the last candidate at rank 7.
* Reports: K-F5 depth success, K-F6 (kept counts, loss reasons, a1b1 cost,
  success by number of window starts, c diagnostic distributions), the trivially
  passing K-F2 cells and the K-F4 cells without enough paired rows.

Inputs (read only): data/processing/m210_k7/runs_k7.csv, run_info.json,
pre_run.txt, post_run.txt, analysis/synthetic/k7/models_k7.json.
Output: docs/processing/m210_k7_judgment.json.

Run: conda run -n gpr-dev python -I analysis/processing/m210_k7_judge.py \
         --expected-seal-commit <K7 seal commit>
"""

import argparse
import datetime
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "data" / "processing" / "m210_k7"
DEFAULT_MODELS_JSON = ROOT / "analysis" / "synthetic" / "k7" / "models_k7.json"
DEFAULT_OUT = ROOT / "docs" / "processing" / "m210_k7_judgment.json"
RUNS_CSV = "runs_k7.csv"
PROTOCOL = "k7"
JSON_BATCH = "k7"
JUDGE_PATH = "analysis/processing/m210_k7_judge.py"
SPEC_DOC = "docs/processing/m210_k7_generator.md"
GENERATOR_PATH = "analysis/processing/fourtu_apex_candidate_select.py"
FOURTU_SHA256 = {
    "fourtu_apex_candidates_v1.csv": "1afd23cf",
    "fourtu_apex_candidate_lines_v1.csv": "485d4c22",
}
"""Committed 4TU csvs (prefixes of their SHA-256, section 4-2)."""

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
"""Section 4-3 table order = seed index m (same tuple as the harness)."""
ROLE_SETS = {
    "defect_a": ("K7_P2_D070", "K7_P2_D070_L"),
    "defect_b": ("K7_P3_D080",),
    "control": ("K7_P1_D060", "K7_P1_D150_S003", "K7_P1_D110_S002", "K7_PVC_D070"),
    "pvc": ("K7_PVC_D070",),
    "bg": ("K7_BG_L",),
    "c_diag": ("K7_P1_D150_S003", "K7_P1_D110_S002"),
}
CALIBRATION_PAIRS = {"K7_BG_L": "K7_P2_D070_L"}
K7_SEALED_FILES = (
    SPEC_DOC,
    "analysis/processing/m210_k7_probe.py",
    "analysis/processing/m210_k7_rank_replay.py",
    "docs/processing/m210_k7_probe.json",
    "docs/processing/m210_k7_rank_replay.json",
    "analysis/synthetic/k7/models_k7.json",
    GENERATOR_PATH,
    "analysis/processing/m210_k7_pipeline.py",
    JUDGE_PATH,
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
"""Same list as the harness ``K7_SEALED_FILES`` (a test keeps them equal)."""

CURRENT = "cur"
CANDIDATES = ("a1", "a1b2")
REPORTED = ("a1b1",)
VARIANTS = (CURRENT, *CANDIDATES, *REPORTED)
TARGET_MECHANISM = {"a1": "a", "a1b2": "b"}
KR_SCOPE = {"a": "defect_a", "b": "defect_b"}
KF1_SCOPE = {"a1": ("defect_a",), "a1b2": ("defect_a", "defect_b")}
EXPECTED_CELLS = {"K_F1": {"a1": 4, "a1b2": 6}, "K_F2": 8, "K_F3": 16}
RHOS = (10, 20)
KR_RHO = 20
N_SEEDS = 40
SEED_BASE = 20301112
UNIT_NOISE_SEED = 20301199
SEED_FORMULA = f"{SEED_BASE} + 1000 m + 100 q + s"
TOL = 1e-9

RATE_GAP_MAX = 0.1  # K-F1, K-F2 (v2.1 F items)
EFFECT_MIN = 0.5  # K-R (the e of R)
FALSE_MARGIN = 0.5  # K-F3
DEPTH_SHIFT_MAX_M = 0.02  # K-F4 (synthetic agreement 2 cm)
DEPTH_OK_M = 0.10  # K-F5 (synthetic tolerance 10 cm)
MIN_PAIRED_ROWS = 20  # K-F4 (0.5 of 40 seeds)
WINDOW_TRACES = 3  # c report: +-3 traces

RANK_HOLD_IDENTITY = 1
RANK_HOLD_REPRODUCTION = 2
RANK_CONTROL = 3
RANK_FALSE = 4
RANK_DEPTH = 5
RANK_INSUFFICIENT = 6
RANK_ADOPTABLE = 7
NOT_JUDGED = "not judged (a1 is not rank 7)"
BOOL_TEXT = {"True": True, "False": False}


# ----------------------------------------------------------------- table helpers


def as_bool(series: pd.Series) -> pd.Series:
    """``"True"`` / ``"False"`` -> bool; anything else (a blank) -> None."""
    return series.map(lambda text: BOOL_TEXT.get(text))


def as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def prepare_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Typed copy of ``runs_k7.csv`` read with ``dtype=str`` (blanks stay NaN)."""
    out = runs.copy()
    for column in ("noise_free", "accepted_apex", "kept_apex", "qc_primary"):
        out[column] = as_bool(out[column])
    for column in (
        "rho", "seed", "seed_index", "model_index", "n_pipes", "pipe_index",
        "depth_error_m", "n_kept", "n_qc_passing", "n_qc_unexplained",
        "n_window_starts", "c_end_offset_traces", "c_tau2_vertex_offset_traces",
    ):  # fmt: skip
        out[column] = as_float(out[column])
    return out


def noise_rows(runs: pd.DataFrame) -> pd.DataFrame:
    return runs[runs["noise_free"] == False]  # noqa: E712 (None stays out)


def cell(rows: pd.DataFrame, model: str, rho: int, variant: str) -> pd.DataFrame:
    return rows[
        (rows["model_id"] == model)
        & (rows["rho"] == rho)
        & (rows["variant"] == variant)
    ]


def rate(rows: pd.DataFrame, model: str, rho: int, variant: str, column: str) -> float:
    """Mean over the pipes of the per-pipe seed share of ``column`` (NaN if empty).

    A blank value makes the rate NaN (never filled): a NaN fails every check.
    """
    selected = cell(rows, model, rho, variant)
    if selected.empty or selected[column].isna().any():
        return math.nan
    per_pipe = selected.groupby("pipe_index")[column].mean()
    return float(per_pipe.mean())


def per_pipe_rates(rows, model, rho, variant, column) -> dict[str, float]:
    selected = cell(rows, model, rho, variant)
    if selected.empty or selected[column].isna().any():
        return {}
    grouped = selected.groupby("pipe_index")[column].mean()
    return {str(int(k)): float(v) for k, v in grouped.items()}


def unexplained_per_line(rows, model, rho, variant) -> float:
    """Mean QC-passing U per line (one value per line; no QC pass = 0)."""
    selected = cell(rows, model, rho, variant)
    lines = selected.drop_duplicates(subset=["seed"])
    if lines.empty or lines["n_qc_unexplained"].isna().any():
        return math.nan
    return float(lines["n_qc_unexplained"].mean())


def _at_least(value: float, bound: float) -> bool:
    return not math.isnan(value) and value >= bound - TOL


def _at_most(value: float, bound: float) -> bool:
    return not math.isnan(value) and value <= bound + TOL


def models_of(role: str) -> tuple[str, ...]:
    return ROLE_SETS[role]


def pipe_models() -> tuple[str, ...]:
    return tuple(m for m in MODEL_IDS if m not in ROLE_SETS["bg"])


# ----------------------------------------------------------------- rank 1


def _parse_time(text: object) -> datetime.datetime | None:
    try:
        return datetime.datetime.fromisoformat(str(text))
    except ValueError:
        return None


@dataclass(frozen=True)
class ExpectedSeal:
    """The K7 seal the judgment expects (git-resolved) and this script's blobs."""

    commit_full: str | None
    blobs_at_seal: Mapping[str, str | None]
    """Spec, generator and judge blobs at the seal commit."""
    judge_blob_working_tree: str | None


def identity_file_check(text: str | None, head: str | None) -> bool:
    """``pre_run.txt`` / ``post_run.txt`` in the harness format with that HEAD."""
    if text is None:
        return False
    lines = text.splitlines()
    if len(lines) < 3 or not lines[0].startswith("time "):
        return False
    return lines[1] == f"HEAD {head}" and lines[2] == "dirty 0"


def g0_checks(info: Mapping) -> dict[str, bool]:
    g0 = info.get("g0") or {}
    fourtu = g0.get("fourtu") or {}
    checks = {"g0_run": bool(g0.get("run"))}
    for name, prefix in FOURTU_SHA256.items():
        record = fourtu.get(name) or {}
        checks[f"g0_{name}"] = record.get("byte_identical") is True and str(
            record.get("regenerated_sha256", "")
        ).startswith(prefix)
    checks["g0_s1g"] = (g0.get("s1g") or {}).get("identical") is True
    checks["g0_m210b_v3"] = (g0.get("m210b_v3") or {}).get("identical") is True
    return checks


def rank1_identity(
    info: Mapping,
    runs_sha_now: str,
    expected: ExpectedSeal,
    identity_files: Mapping[str, str | None],
    models_json_sha_now: str | None,
) -> dict:
    """Every identity condition of section 8 that is not a published value.

    ``models_json_sha_now`` is the SHA-256 of the models json read by this
    judgment; it must be the one the harness recorded (``inputs.models_json``).
    """
    seal, seal_end = info.get("seal") or {}, info.get("seal_at_end") or {}
    sealed = seal.get("sealed_files") or {}
    head = seal.get("head_commit")
    checks: dict[str, bool] = {
        "protocol_is_k7": info.get("protocol") == PROTOCOL,
        "sealed_files_cover_k7_list": set(K7_SEALED_FILES)
        <= set(info.get("sealed_files") or []),
        "sealed_blobs_match_run": bool(sealed)
        and all(record.get("matches") for record in sealed.values()),
        "sealed_blobs_match_run_end": bool(seal_end.get("sealed_files"))
        and seal_end.get("sealed_files") == sealed,
        "seal_commit_is_expected": expected.commit_full is not None
        and seal.get("seal_commit_full") == expected.commit_full,
        "run_head_is_expected_seal": expected.commit_full is not None
        and head == expected.commit_full,
        "head_same_start_end": head is not None and head == seal_end.get("head_commit"),
        "dirty_0_start": seal.get("dirty_files") == [],
        "dirty_0_end": seal_end.get("dirty_files") == [],
        "harness_git_queries_ok": not seal.get("git_failures")
        and not seal_end.get("git_failures"),
        "judge_blob_is_seal_blob": expected.judge_blob_working_tree is not None
        and expected.judge_blob_working_tree == expected.blobs_at_seal.get(JUDGE_PATH),
        "runs_csv_sha256_matches": info.get("runs_csv_sha256") == runs_sha_now,
        "harness_guards_pass": bool((info.get("guards") or {}).get("pass")),
        "models_json_sha256_matches_run": models_json_sha_now is not None
        and ((info.get("inputs") or {}).get("models_json") or {}).get("sha256")
        == models_json_sha_now,
        "k7_tests_pass": bool((info.get("tests") or {}).get("pass")),
        "pre_run_txt_harness_format": identity_file_check(
            identity_files.get("pre_run.txt"), head
        ),
        "post_run_txt_harness_format": identity_file_check(
            identity_files.get("post_run.txt"), head
        ),
    }
    for path, blob in expected.blobs_at_seal.items():
        recorded = (sealed.get(path) or {}).get("blob_at_seal")
        checks[f"blob_at_seal_{path}"] = blob is not None and recorded == blob
    checks["generator_blob_is_seal_blob"] = info.get(
        "generator_blob"
    ) is not None and info.get("generator_blob") == expected.blobs_at_seal.get(
        GENERATOR_PATH
    )
    seal_time = _parse_time(seal.get("seal_commit_time"))
    start_time = _parse_time(info.get("started_at"))
    checks["seal_before_harness_start"] = (
        seal_time is not None and start_time is not None and seal_time < start_time
    )
    checks.update(g0_checks(info))
    return {"checks": checks, "pass": all(checks.values())}


def seed_checks(runs: pd.DataFrame, info: Mapping) -> dict:
    """Overlap 0 and every noise seed on the K7 formula."""
    seeds = info.get("seeds") or {}
    rows = noise_rows(runs)
    rho_index = {rho: q for q, rho in enumerate(RHOS)}
    expected = (
        SEED_BASE
        + 1000 * rows["model_index"]
        + 100 * rows["rho"].map(rho_index)
        + rows["seed_index"]
    )
    model_index_ok = (
        rows["model_id"]
        .map(lambda m: MODEL_IDS.index(m) if m in MODEL_IDS else -1)
        .astype(float)
        .eq(rows["model_index"])
    )
    checks = {
        "seed_overlap_0": info.get("seed_overlap") == 0,
        "seed_formula_recorded": seeds.get("seed_formula") == SEED_FORMULA,
        "unit_noise_seed_recorded": seeds.get("unit_noise_seed") == UNIT_NOISE_SEED,
        "rows_on_formula": bool(len(rows)) and bool(rows["seed"].eq(expected).all()),
        "model_index_is_table_order": bool(len(rows)) and bool(model_index_ok.all()),
    }
    return {"checks": checks, "pass": all(checks.values())}


def role_set_checks(json_role_sets: Mapping[str, Sequence[str]]) -> dict[str, bool]:
    """The role sets: section 4-3 definition, structure and the json copy."""
    sets = {name: set(values) for name, values in ROLE_SETS.items()}
    partition = ("defect_a", "defect_b", "control", "bg")
    union = set().union(*(sets[name] for name in partition))
    is_disjoint = sum(len(sets[name]) for name in partition) == len(union)
    json_sets = {
        name: sorted(values)
        for name, values in json_role_sets.items()
        if name != "source"
    }
    return {
        "role_sets_json_equal": json_sets
        == {name: sorted(values) for name, values in ROLE_SETS.items()},
        "pvc_in_control": sets["pvc"] <= sets["control"],
        "c_diag_in_control": sets["c_diag"] <= sets["control"],
        "partition_disjoint": is_disjoint,
        "partition_is_8_models": union == set(MODEL_IDS),
    }


def _expected_rows_per_condition(n_pipes: int) -> int:
    """40 seeds x pipes; a pipe-less line has one row per seed."""
    return N_SEEDS * max(n_pipes, 1)


def fixed_set_checks(
    runs: pd.DataFrame, info: Mapping, json_models: Mapping[str, object]
) -> dict:
    """Section 4-6 fixed sets; any false -> rank 1."""
    plan = info.get("plan") or {}
    json_ids = list(json_models["model_ids"])
    json_pipes = json_models["n_pipes"]
    checks: dict[str, bool] = {
        "json_models_are_expected": sorted(json_ids) == sorted(MODEL_IDS),
        "run_models_are_expected": sorted(set(runs["model_id"])) == sorted(MODEL_IDS),
        "plan_models_are_expected": sorted(plan.get("model_ids") or [])
        == sorted(MODEL_IDS),
        "variants_are_expected": sorted(set(runs["variant"])) == sorted(VARIANTS)
        and sorted(plan.get("variants") or []) == sorted(VARIANTS),
        "rhos_are_expected": sorted(set(noise_rows(runs)["rho"])) == list(RHOS)
        and plan.get("rhos") == list(RHOS),
        "n_seeds_is_40": plan.get("n_seeds") == N_SEEDS,
    }
    checks.update(role_set_checks(json_models["role_sets"]))
    rows = noise_rows(runs)
    for model in MODEL_IDS:
        n_pipes = json_pipes.get(model)
        recorded = ((info.get("models") or {}).get(model) or {}).get("n_pipes")
        run_pipes = set(runs.loc[runs["model_id"] == model, "n_pipes"])
        checks[f"n_pipes_{model}"] = (
            n_pipes is not None
            and recorded == n_pipes
            and run_pipes == {float(n_pipes)}
            and (n_pipes == 0) == (model in ROLE_SETS["bg"])
        )
        expected_rows = _expected_rows_per_condition(n_pipes or 0)
        for variant in VARIANTS:
            clean = runs[
                (runs["model_id"] == model)
                & (runs["variant"] == variant)
                & (runs["noise_free"] == True)  # noqa: E712
            ]
            checks[f"noise_free_rows_{model}_{variant}"] = _noise_free_pipes_ok(
                clean, n_pipes or 0
            )
            for rho in RHOS:
                selected = cell(rows, model, rho, variant)
                checks[f"rows_{model}_{rho}_{variant}"] = len(
                    selected
                ) == expected_rows and _seeds_per_pipe_ok(selected, n_pipes or 0)
    checks["calibration_pair"] = calibration_pair_check(info)
    checks["no_blank_outcome"] = no_blank_outcome(runs)
    return {"checks": checks, "pass": all(checks.values())}


def _pipe_labels(rows: pd.DataFrame) -> list[object]:
    """``pipe_index`` per row; a blank (the pipe-less line) becomes ``None``."""
    return [None if math.isnan(v) else int(v) for v in rows["pipe_index"]]


def _expected_pipe_labels(n_pipes: int) -> list[object]:
    """Pipes 0..n-1, or one blank ``pipe_index`` on a line without a pipe."""
    return list(range(n_pipes)) if n_pipes else [None]


def _noise_free_pipes_ok(rows: pd.DataFrame, n_pipes: int) -> bool:
    """Exactly one noise-free row per pipe (one blank-pipe row on K7_BG_L)."""
    labels = _pipe_labels(rows)
    expected = _expected_pipe_labels(n_pipes)
    return len(labels) == len(expected) and set(labels) == set(expected)


def _seeds_per_pipe_ok(rows: pd.DataFrame, n_pipes: int) -> bool:
    """Every pipe of the condition has seeds 0-39 once each (40 rows)."""
    labels = _pipe_labels(rows)
    all_seeds = list(map(float, range(N_SEEDS)))
    for pipe in _expected_pipe_labels(n_pipes):
        seeds = sorted(
            seed
            for seed, label in zip(rows["seed_index"], labels, strict=True)
            if label == pipe
        )
        if seeds != all_seeds:
            return False
    return set(labels) == set(_expected_pipe_labels(n_pipes))


def calibration_pair_check(info: Mapping) -> bool:
    """K7_BG_L uses the K7_P2_D070_L calibration, sigma_raw included."""
    models = info.get("models") or {}
    plan_pairs = (info.get("plan") or {}).get("calibration_pairs")
    if plan_pairs != CALIBRATION_PAIRS:
        return False
    for line, source in CALIBRATION_PAIRS.items():
        line_cal = (models.get(line) or {}).get("calibration") or {}
        source_cal = (models.get(source) or {}).get("calibration") or {}
        if line_cal.get("source_model") != source or not source_cal:
            return False
        if line_cal.get("sigma_raw_by_rho") != source_cal.get("sigma_raw_by_rho"):
            return False
    return True


def no_blank_outcome(runs: pd.DataFrame) -> bool:
    """Pipe rows carry accepted/kept/QC outcomes; every row its U count."""
    pipe_rows = runs[runs["model_id"].isin(pipe_models())]
    outcome = pipe_rows[["accepted_apex", "kept_apex", "qc_primary"]]
    return (
        bool(len(runs))
        and not outcome.isna().any().any()
        and not runs["n_qc_unexplained"].isna().any()
    )


# ----------------------------------------------------------------- K items


def kr_table(rows: pd.DataFrame) -> list[dict]:
    """K-R per defect model: cur gap at rho 20 (acc rate 0 -> gap <= 0)."""
    table = []
    for mechanism, role in KR_SCOPE.items():
        for model in models_of(role):
            accepted = rate(rows, model, KR_RHO, CURRENT, "accepted_apex")
            kept = rate(rows, model, KR_RHO, CURRENT, "kept_apex")
            gap = accepted - kept
            table.append(
                {
                    "mechanism": mechanism,
                    "model": model,
                    "rho": KR_RHO,
                    "accepted_rate_cur": accepted,
                    "kept_rate_cur": kept,
                    "gap": gap,
                    "reproduced": _at_least(gap, EFFECT_MIN),
                }
            )
    return table


def kr_result(table: Sequence[Mapping]) -> dict[str, bool]:
    """A mechanism is reproduced when at least one of its models is."""
    return {
        mechanism: any(r["reproduced"] for r in table if r["mechanism"] == mechanism)
        for mechanism in KR_SCOPE
    }


def kf1(rows: pd.DataFrame, candidate: str) -> dict:
    cells = []
    for role in KF1_SCOPE[candidate]:
        for model in models_of(role):
            for rho in RHOS:
                bound = rate(rows, model, rho, CURRENT, "accepted_apex") - RATE_GAP_MAX
                kept = rate(rows, model, rho, candidate, "kept_apex")
                cells.append(
                    {"model": model, "rho": rho, "kept_rate": kept,
                     "bound": bound, "pass": _at_least(kept, bound)}
                )  # fmt: skip
    return _cells_result(cells, EXPECTED_CELLS["K_F1"][candidate])


def kf2(rows: pd.DataFrame, candidate: str) -> dict:
    cells, trivial = [], []
    for model in models_of("control"):
        for rho in RHOS:
            entry: dict[str, object] = {"model": model, "rho": rho}
            is_pass = True
            for column in ("kept_apex", "qc_primary"):
                current = rate(rows, model, rho, CURRENT, column)
                value = rate(rows, model, rho, candidate, column)
                entry[f"{column}_cur"] = current
                entry[f"{column}_candidate"] = value
                is_pass = is_pass and _at_least(value, current - RATE_GAP_MAX)
                if current == 0:
                    trivial.append({"model": model, "rho": rho, "metric": column})
            entry["pass"] = is_pass
            cells.append(entry)
    result = _cells_result(cells, EXPECTED_CELLS["K_F2"])
    result["trivial_pass_cells"] = trivial
    return result


def kf3(rows: pd.DataFrame, candidate: str) -> dict:
    cells = []
    for model in MODEL_IDS:
        for rho in RHOS:
            current = unexplained_per_line(rows, model, rho, CURRENT)
            value = unexplained_per_line(rows, model, rho, candidate)
            cells.append(
                {"model": model, "rho": rho, "u_cur": current, "u_candidate": value,
                 "pass": _at_most(value, current + FALSE_MARGIN)}
            )  # fmt: skip
    result = _cells_result(cells, EXPECTED_CELLS["K_F3"])
    result["bg_qc_passing"] = {
        f"{rho}|{variant}": float(
            cell(rows, ROLE_SETS["bg"][0], rho, variant)["n_qc_passing"].sum()
        )
        for rho in RHOS
        for variant in VARIANTS
    }
    return result


def _cells_result(cells: list[dict], expected_count: int) -> dict:
    """Pass needs the defined number of cells and every cell passing."""
    count_ok = len(cells) == expected_count
    return {
        "cells": cells,
        "n_cells": len(cells),
        "n_cells_expected": expected_count,
        "cell_count_ok": count_ok,
        "pass": count_ok and all(c["pass"] for c in cells),
    }


def paired_depth_errors(rows: pd.DataFrame, model: str, rho: int, candidate: str):
    """``depth_error_m`` of cur and candidate on rows where both have a primary."""
    keys = ["seed", "pipe_index"]
    current = cell(rows, model, rho, CURRENT)
    other = cell(rows, model, rho, candidate)
    current = current[current["qc_primary"] == True][[*keys, "depth_error_m"]]  # noqa: E712
    other = other[other["qc_primary"] == True][[*keys, "depth_error_m"]]  # noqa: E712
    return current.merge(other, on=keys, suffixes=("_cur", "_candidate"))


def kf4(rows: pd.DataFrame, candidate: str) -> dict:
    cells, not_applicable = [], []
    for model in pipe_models():
        for rho in RHOS:
            paired = paired_depth_errors(rows, model, rho, candidate)
            entry: dict[str, object] = {
                "model": model,
                "rho": rho,
                "n_paired": len(paired),
            }
            if len(paired) < MIN_PAIRED_ROWS:
                not_applicable.append(
                    {"model": model, "rho": rho, "n_paired": len(paired)}
                )
                continue
            median_cur = float(paired["depth_error_m_cur"].median())
            median_candidate = float(paired["depth_error_m_candidate"].median())
            shift = median_candidate - median_cur
            entry["median_cur_m"] = median_cur
            entry["median_candidate_m"] = median_candidate
            entry["shift_m"] = shift
            entry["pass"] = _at_most(abs(shift), DEPTH_SHIFT_MAX_M)
            cells.append(entry)
    pvc_cells = [c for c in not_applicable if c["model"] in ROLE_SETS["pvc"]]
    is_judgeable = bool(cells) and len(pvc_cells) < len(ROLE_SETS["pvc"]) * len(RHOS)
    return {
        "cells": cells,
        "not_applicable_cells": not_applicable,
        "judgeable": is_judgeable,
        "pass": is_judgeable and all(c["pass"] for c in cells),
    }


def kf5(rows: pd.DataFrame) -> dict:
    """K-F5 (report): QC primary within 0.10 m, share of 40 x pipes rows."""
    out = {}
    for model in pipe_models():
        for rho in RHOS:
            for variant in VARIANTS:
                selected = cell(rows, model, rho, variant)
                is_ok = (selected["qc_primary"] == True) & (  # noqa: E712
                    selected["depth_error_m"].abs() <= DEPTH_OK_M + TOL
                )
                out[f"{model}|{rho}|{variant}"] = (
                    float(is_ok.mean()) if len(selected) else math.nan
                )
    return out


def _quantiles(values: pd.Series) -> list[float] | None:
    clean = values.dropna()
    if clean.empty:
        return None
    return [float(clean.quantile(q)) for q in (0.1, 0.5, 0.9)]


def c_report(rows: pd.DataFrame) -> dict:
    """K-F6 c columns per model and variant (report only, L3)."""
    out = {}
    for model in pipe_models():
        for variant in VARIANTS:
            selected = rows[(rows["model_id"] == model) & (rows["variant"] == variant)]
            has_item = selected[selected["c_item"].fillna("") != ""]
            offsets = has_item["c_end_offset_traces"]
            vertex = has_item["c_tau2_vertex_offset_traces"]
            out[f"{model}|{variant}"] = {
                "n_rows": len(selected),
                "n_with_item": len(has_item),
                "end_offset_q10_q50_q90": _quantiles(offsets),
                "end_within_3": _share(offsets.abs() <= WINDOW_TRACES, has_item),
                "start_local_max": _share(
                    has_item["c_start_is_local_max"] == "True", has_item
                ),
                "end_local_max": _share(
                    has_item["c_end_is_local_max"] == "True", has_item
                ),
                "vertex_offset_q10_q50_q90": _quantiles(vertex),
                "vertex_fallback": int(vertex.isna().sum()),
                "vertex_within_3": _share(vertex.abs() <= WINDOW_TRACES, has_item),
            }
    return out


def _share(mask: pd.Series, rows: pd.DataFrame) -> float | None:
    return None if rows.empty else float(mask.mean())


def kf6(rows: pd.DataFrame) -> dict:
    """K-F6 (report): kept counts, loss reasons, a1b1 cost, starts, c columns."""
    lines = rows.drop_duplicates(subset=["model_id", "rho", "seed", "variant"])
    kept = lines.groupby(["model_id", "rho", "variant"])["n_kept"].mean()
    pipe_rows = rows[rows["model_id"].isin(pipe_models())]
    reasons = pipe_rows[pipe_rows["kept_apex"] == False]  # noqa: E712
    reason_counts = reasons.groupby(["model_id", "variant", "lost_reason"]).size()
    a1_total = float(lines.loc[lines["variant"] == "a1", "n_kept"].sum())
    a1b1_total = float(lines.loc[lines["variant"] == "a1b1", "n_kept"].sum())
    by_starts = pipe_rows.groupby(["variant", "n_window_starts"])["kept_apex"].agg(
        ["mean", "size"]
    )
    return {
        "mean_kept": {f"{m}|{int(r)}|{v}": float(x) for (m, r, v), x in kept.items()},
        "lost_reasons": {
            f"{m}|{v}|{r}": int(n) for (m, v, r), n in reason_counts.items()
        },
        "a1b1_cost": {
            "kept_total_a1": a1_total,
            "kept_total_a1b1": a1b1_total,
            "ratio": a1b1_total / a1_total if a1_total else None,
        },
        "kept_rate_by_window_starts": {
            f"{v}|{int(n)}": {"rate": float(row["mean"]), "n_rows": int(row["size"])}
            for (v, n), row in by_starts.iterrows()
        },
        "c_diagnostics": c_report(rows),
    }


# ----------------------------------------------------------------- decision


def candidate_rank(
    candidate: str,
    rank1_reasons: Sequence[str],
    kr: Mapping[str, bool],
    items: Mapping[str, Mapping],
) -> dict:
    """First matching row of the section 4-6 table for one candidate."""
    if rank1_reasons:
        return {"rank": RANK_HOLD_IDENTITY, "rank_reasons": list(rank1_reasons)}
    reasons = []
    mechanism = TARGET_MECHANISM[candidate]
    if not kr[mechanism]:
        reasons.append(f"K-R({mechanism}) not reproduced")
    if not items["K_F4"]["judgeable"]:
        reasons.append("K-F4 not judgeable (K7_PVC cells or every cell < 20 pairs)")
    if reasons:
        return {"rank": RANK_HOLD_REPRODUCTION, "rank_reasons": reasons}
    ordered = (
        ("K_F2", RANK_CONTROL),
        ("K_F3", RANK_FALSE),
        ("K_F4", RANK_DEPTH),
        ("K_F1", RANK_INSUFFICIENT),
    )
    for name, rank in ordered:
        if not items[name]["pass"]:
            return {"rank": rank, "rank_reasons": [f"{name} failed"]}
    return {"rank": RANK_ADOPTABLE, "rank_reasons": []}


def cell_count_reasons(items_by_candidate: Mapping[str, Mapping]) -> list[str]:
    """Judged cell counts that differ from the definition (0 included) -> rank 1."""
    reasons = []
    for candidate, items in items_by_candidate.items():
        for name in ("K_F1", "K_F2", "K_F3"):
            if not items[name]["cell_count_ok"]:
                reasons.append(
                    f"{candidate} {name}: {items[name]['n_cells']} cells, expected "
                    f"{items[name]['n_cells_expected']}"
                )
    return reasons


def decide(ranks: Mapping[str, Mapping]) -> str:
    """The adopted variant: the last candidate at rank 7 (``current`` if none)."""
    adopted = "current"
    for candidate in CANDIDATES:
        if ranks[candidate].get("rank") == RANK_ADOPTABLE:
            adopted = candidate
    return adopted


def judge(
    runs: pd.DataFrame,
    info: Mapping,
    json_models: Mapping[str, object],
    runs_sha_now: str,
    expected: ExpectedSeal,
    identity_files: Mapping[str, str | None],
) -> dict:
    """All counts, ranks and the adopted variant (no I/O)."""
    runs = prepare_runs(runs)
    identity = rank1_identity(
        info, runs_sha_now, expected, identity_files, json_models.get("sha256")
    )
    seeds = seed_checks(runs, info)
    fixed = fixed_set_checks(runs, info, json_models)
    rows = noise_rows(runs)
    table = kr_table(rows)
    kr = kr_result(table)
    items_by_candidate = {
        candidate: {
            "K_F1": kf1(rows, candidate),
            "K_F2": kf2(rows, candidate),
            "K_F3": kf3(rows, candidate),
            "K_F4": kf4(rows, candidate),
        }
        for candidate in CANDIDATES
    }
    rank1_reasons = [
        f"{group}: {name}"
        for group, record in (
            ("identity", identity),
            ("seeds", seeds),
            ("fixed_sets", fixed),
        )
        for name, ok in record["checks"].items()
        if not ok
    ]
    rank1_reasons += cell_count_reasons(items_by_candidate)
    ranks: dict[str, dict] = {}
    ranks["a1"] = candidate_rank("a1", rank1_reasons, kr, items_by_candidate["a1"])
    a1b2 = candidate_rank("a1b2", rank1_reasons, kr, items_by_candidate["a1b2"])
    if ranks["a1"]["rank"] == RANK_ADOPTABLE:
        ranks["a1b2"] = a1b2
    else:
        ranks["a1b2"] = {"rank": NOT_JUDGED, "rank_reasons": [], "reference": a1b2}
    return {
        "rank1_identity": identity,
        "rank1_seeds": seeds,
        "rank1_fixed_sets": fixed,
        "rank1_reasons": rank1_reasons,
        "K_R_table": table,
        "K_R": kr,
        "items": items_by_candidate,
        "ranks": ranks,
        "adopted": decide(ranks),
        "K_F2_trivial_pass_cells": {
            c: items_by_candidate[c]["K_F2"]["trivial_pass_cells"] for c in CANDIDATES
        },
        "K_F4_not_applicable_cells": {
            c: items_by_candidate[c]["K_F4"]["not_applicable_cells"] for c in CANDIDATES
        },
        "rates": {
            f"{m}|{rho}|{v}": {
                "accepted": rate(rows, m, rho, v, "accepted_apex"),
                "kept": rate(rows, m, rho, v, "kept_apex"),
                "qc_primary": rate(rows, m, rho, v, "qc_primary"),
                "kept_per_pipe": per_pipe_rates(rows, m, rho, v, "kept_apex"),
            }
            for m in pipe_models()
            for rho in RHOS
            for v in VARIANTS
        },
        "K_F5": kf5(rows),
        "K_F6": kf6(rows),
        "published": {
            "mkl_cbwr": info.get("mkl_cbwr"),
            "generator_blob": info.get("generator_blob"),
            "g0": info.get("g0"),
        },
    }


# ----------------------------------------------------------------- I/O


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


def resolve_expected_seal(commit: str) -> ExpectedSeal:
    """Full seal commit, blobs at it and this script's working-tree blob."""
    full = _git("rev-parse", "--verify", f"{commit}^{{commit}}")
    blobs = {
        path: None if full is None else _git("rev-parse", f"{full}:{path}")
        for path in (SPEC_DOC, GENERATOR_PATH, JUDGE_PATH)
    }
    return ExpectedSeal(full, blobs, _git("hash-object", "--", JUDGE_PATH))


def read_runs(path: Path) -> pd.DataFrame:
    """``runs_k7.csv`` as text; blanks become NaN and are never filled."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def read_json_models(models_json: Path) -> dict[str, object]:
    raw = models_json.read_bytes()
    meta = json.loads(raw.decode("utf-8"))
    models = [m for m in meta["models"] if m.get("batch") == JSON_BATCH]
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "model_ids": [m["id"] for m in models],
        "n_pipes": {m["id"]: len(m["pipes"]) for m in models},
        "role_sets": meta.get("role_sets") or {},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--expected-seal-commit",
        required=True,
        help="the K7 seal commit this judgment belongs to (rank-1 identity)",
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--models-json", type=Path, default=DEFAULT_MODELS_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs_path = args.run_dir / RUNS_CSV
    runs = read_runs(runs_path)
    info = json.loads((args.run_dir / "run_info.json").read_text(encoding="utf-8"))
    runs_sha_now = hashlib.sha256(runs_path.read_bytes()).hexdigest()
    identity_files = {}
    for name in ("pre_run.txt", "post_run.txt"):
        path = args.run_dir / name
        identity_files[name] = (
            path.read_text(encoding="utf-8") if path.exists() else None
        )
    expected = resolve_expected_seal(args.expected_seal_commit)
    out = judge(
        runs, info, read_json_models(args.models_json), runs_sha_now, expected,
        identity_files,
    )  # fmt: skip
    out = {
        "note": "K7 judgment counts; criteria m210_k7_generator.md v1.1 4-5, 4-6",
        "expected_seal_commit": expected.commit_full,
        "blobs_at_seal": dict(expected.blobs_at_seal),
        "judge_blob_working_tree": expected.judge_blob_working_tree,
        "runs_csv_sha256_now": runs_sha_now,
        "identity_files": identity_files,
        **out,
    }
    args.out.write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    brief = {
        "adopted": out["adopted"],
        "ranks": {k: v["rank"] for k, v in out["ranks"].items()},
        "n_rank1_reasons": len(out["rank1_reasons"]),
    }
    print(json.dumps(brief, ensure_ascii=True, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
