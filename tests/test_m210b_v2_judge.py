"""v2 judgment script (m210b_selector_floor_v2.md 8-2): F5-B boundaries and P-F1 on
small fake ``runs_v2.csv`` tables. No harness output is needed."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JUDGE_PATH = PROJECT_ROOT / "analysis" / "processing" / "m210b_v2_judge.py"
S4N, S6P, NP = "S4N_D14_S040", "S6P_D14_S040_J2", "S6P_NP_S040_J2"
MODELS = (S4N, S6P, NP)
APEX_TRACE, T_A0_NS = 80, 30.0


@pytest.fixture(scope="module")
def judge():
    spec = importlib.util.spec_from_file_location("m210b_v2_judge", JUDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["m210b_v2_judge"] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------- F5-B boundaries


def _pairs(eps_t_x, eps_t_c, eps_j_x=None, eps_j_c=None) -> pd.DataFrame:
    n = len(eps_t_x)
    return pd.DataFrame(
        {
            "eps_t_x": eps_t_x,
            "eps_t_c": eps_t_c,
            "eps_j_x": eps_j_x or [0.0] * n,
            "eps_j_c": eps_j_c or [0.0] * n,
        }
    )


def test_f5b_worse_equal_to_better_passes(judge) -> None:
    eps_c = [0.1] * 20
    eps_x = [0.1] * 20
    eps_x[0] = 0.3  # worse in time
    eps_x[1] = 0.0  # better in time... by 0.1 > 0.049
    result = judge.f5b_counts(_pairs(eps_x, eps_c))
    assert (result["n_worse"], result["n_better"], result["pass"]) == (1, 1, True)


def test_f5b_one_more_worse_fails(judge) -> None:
    eps_c = [0.1] * 20
    eps_x = [0.1] * 20
    eps_x[0] = 0.3
    eps_x[1] = 0.0
    eps_j_x = [0.0] * 20
    eps_j_x[2] = 2.0  # worse in traces (> 1)
    result = judge.f5b_counts(_pairs(eps_x, eps_c, eps_j_x=eps_j_x))
    assert (result["n_worse"], result["n_better"], result["pass"]) == (2, 1, False)


def test_f5b_median_gap_of_exactly_0_049_passes(judge) -> None:
    result = judge.f5b_counts(_pairs([0.149] * 20, [0.1] * 20))
    assert result["n_worse"] == 0
    assert result["median_eps_t_x_ns"] == pytest.approx(0.149)
    assert result["pass"] is True
    above = judge.f5b_counts(_pairs([0.1491] * 20, [0.1] * 20))
    assert above["pass"] is False


def test_f5b_with_19_pairs_is_not_judgeable(judge) -> None:
    result = judge.f5b_counts(_pairs([0.1] * 19, [0.1] * 19))
    assert (result["judgeable"], result["pass"]) == (False, False)
    assert judge.f5b_counts(_pairs([0.1] * 20, [0.1] * 20))["pass"] is True


# ----------------------------------------------------------------- fake run


def _row(model, rho, seed, rule, delta, ramp=0, dt=0.098, **values) -> dict:
    is_pipe = model != NP
    row = {
        "model_id": model,
        "rho": rho,
        "seed": seed,
        "rule": rule,
        "delta_ns": delta,
        "ramp_noise": ramp,
        "dt_ns": dt,
        "t_zero_ns": 3.4,
        "zero_rule": "air",
        "L": 1.0,
        "F": 0.05,
        "sigma_n": 0.01,
        "rho_real": 20.0 if is_pipe else "",
        "rho_real_cal": 20.0,
        "Lambda": 30.0 if model == S6P else (0.5 if is_pipe else ""),
        "argmax_level_periods": 0.0,
        "accepted_apex": True if is_pipe else "",
        "peak_on_crest": True if is_pipe else "",
        "kept_apex": True if is_pipe else "",
        "kept_apex_trace": APEX_TRACE if is_pipe else "",
        "kept_apex_sample": 306.0 if is_pipe else "",
        "lost_reason": "",
        "floor_held_at_apex": model == S6P if is_pipe else "",
        "n_peaks": 50,
        "cap300_hit": False,
        "n_false_deep": 0,
        "n_false_near_surface": 0,
        "n_false_record_end": 0,
        "kept_nearest_trace": APEX_TRACE if is_pipe else "",
        "kept_nearest_sample": 306.0 if is_pipe else "",
        "kept_nearest_t_ns": T_A0_NS + 0.02 if is_pipe else "",
        "n_kept_in_window": 1 if is_pipe else "",
        "depth_nearest_m": 1.4 if is_pipe else "",
        "depth_nearest_status": "ok" if is_pipe else "",
    }
    row.update(values)
    return row


def _seed(judge, p, rho_index, s) -> int:
    return 20271120 + 1000 * p + 100 * rho_index + s


def fake_runs(judge, s6p_late_lambda: float = 5.0) -> pd.DataFrame:
    """A complete v2 table where R holds and every candidate passes.

    ``s6p_late_lambda`` is the S6P ``Lambda`` at the late mute (+2.35 ns).
    """
    rows = []
    pair_index = {S4N: 0, S6P: 0, NP: 4}
    for model in MODELS:
        p = pair_index[model]
        for q, rho in enumerate(judge.RHOS):
            for s in range(judge.N_SEEDS):
                seed = _seed(judge, p, q, s)
                for rule in judge.RULES:
                    for delta in judge.DELTAS:
                        extra = {}
                        # Coupling: current loses the S6P apex unless the mute is late.
                        if model == S6P and rule == "current" and delta != 2.35:
                            extra["accepted_apex"] = False
                        if model == S6P and delta == 2.35:
                            extra["Lambda"] = s6p_late_lambda
                        rows.append(_row(model, rho, seed, rule, delta, **extra))
        for s in range(judge.N_SEEDS):
            seed = _seed(judge, p, 0, s)
            for rule in ("current", "B_full"):
                rows.append(_row(model, 10, seed, rule, 0.0, dt=0.0118))
            if model == NP:
                for rule in judge.RULES:
                    for delta in judge.DELTAS:
                        rows.append(_row(model, 10, seed, rule, delta, ramp=1))
    return pd.DataFrame(rows).astype(str)


V1_DOC = "docs/processing/m210b_selector_floor.md"
INTERP_DOC = "docs/processing/m210b_harness_interpretation.md"
FLOOR_RULES = "analysis/processing/m210b_floor_rules.py"
GENERATOR = "analysis/processing/fourtu_apex_candidate_select.py"
REFERENCE_BLOBS = {
    V1_DOC: "v1blob",
    INTERP_DOC: "interpblob",
    FLOOR_RULES: "rulesblob",
    GENERATOR: "generatorblob",
}
SHA = "sha"


def fake_info(runs: pd.DataFrame, s6p_late_lambda: float = 5.0) -> dict:
    def harness(apex, lam, late=None):
        lams = {f"{d:+.2f}": lam for d in (-1.0, 0.0, 1.0)}
        lams["+2.35"] = lam if late is None else late
        levels = {key: value * 0.04 for key, value in lams.items()}
        return {
            "apex_trace": apex, "t_a0_ns": T_A0_NS if apex else None,
            "L_noise_free_current": levels,
            "Lambda": lams if apex else {k: None for k in levels},
            "argmax_level_periods_noise_free": {k: 0.0 for k in levels},
            "A0_self": 0.04 if apex else None,
            "calibration": {
                "control_id": f"S4S6/{S4N}",
                "a0": 0.04,
                "sigma_u": 0.5,
                "sigma_raw_by_rho": {"10": 0.008, "20": 0.004},
            },
        }  # fmt: skip

    sealed = {
        path: {"blob_at_seal": blob, "matches": True}
        for path, blob in REFERENCE_BLOBS.items()
    }
    sealed |= {
        "docs/processing/m210b_selector_floor_v2.md": {
            "blob_at_seal": "v2",
            "matches": True,
        },
    }
    seal = {
        "seal_commit_full": "v2seal", "seal_commit_time": "2026-11-12T10:00:00+09:00",
        "sealed_files": sealed, "dirty_files": [], "head_commit": "v2seal",
        "git_failures": [],
    }  # fmt: skip
    return {
        "models_run": list(MODELS),
        "models": {
            S4N: {"harness": harness(APEX_TRACE, 0.5), "pipe_top_depth_m": 1.4},
            S6P: {
                "harness": harness(APEX_TRACE, 30.0, late=s6p_late_lambda),
                "pipe_top_depth_m": 1.4,
            },
            NP: {
                "harness": harness(None, 0.0),
                "pipe_top_depth_m": None,
                "calibration_pair": "D14_S040",
            },
        },
        "n_runs": len(runs),
        "n_runs_without_zero": 0,
        "seed_overlap": 0,
        "seal": seal,
        "seal_at_end": {**seal, "dirty_files": []},
        "guards": {"pass": True, "failures": []},
        "wrapper_regression": {
            "run": True,
            "fourtu_apex_candidates_v1.csv": {"byte_identical": True},
            "fourtu_apex_candidate_lines_v1.csv": {"byte_identical": True},
        },
        "runs_csv_sha256": SHA,
        "protocol_detail": {
            "seed_formula": "20271120 + 1000 p + 100 q + s",
            "unit_noise_seed": 20271199,
            "no_pipe_calibration_pair": "D14_S040",
        },
        "started_at": "2026-11-12T11:00:00+09:00",
        "finished_at": "2026-11-12T11:40:00+09:00",
    }


def run_judge(
    judge, runs, info=None, models=MODELS, sha=SHA, blobs=None, expected=None
):
    info = fake_info(runs) if info is None else info
    expected = judge.ExpectedSeal("v2seal", "v2") if expected is None else expected
    blobs = REFERENCE_BLOBS if blobs is None else blobs
    return judge.judge(
        runs, info, list(models), sha, blobs, expected, frozenset(MODELS)
    )


def test_fake_run_passes_pf1_and_reaches_rank_4(judge) -> None:
    runs = fake_runs(judge)
    out = run_judge(judge, runs)
    assert out["P_F1"]["pass"], out["P_F1"]
    assert out["rank1_identity"]["pass"], out["rank1_identity"]
    assert out["R"] is True and out["decision"]["rank"] == "4/5"
    assert out["rank_by_rule"] == {"B_full": 4, "B_min": 4}
    assert out["U1"] == "B_full"
    f5b = out["F_by_rule"]["B_full"]["F5B"]
    assert (
        f5b["n_pairs"] == 2 * 2 * 4 * 20 and f5b["reading"] == "no position difference"
    )
    scopes = out["F5B_by_scope"]["B_full"]
    assert scopes["S1_judgment_all_pipe_models"]["n_pairs"] == f5b["n_pairs"]
    assert scopes["S2_R_conditions"]["n_pairs"] == 2 * 4 * 20  # S6P, two rho
    assert out["original_dt"]["rows"] == 3 * 2 * 20


def test_pf1_violation_with_19_seeds_is_rank_1(judge) -> None:
    runs = fake_runs(judge)
    is_cell = (runs["model_id"] == S4N) & (runs["rule"] == "B_min")
    is_cell &= (runs["delta_ns"] == "1.0") & (runs["rho"] == "20")
    drop = runs[is_cell & (runs["ramp_noise"] == "0")].index[0]
    short = runs.drop(index=drop)
    out = run_judge(judge, short)
    assert out["P_F1"]["pass"] is False
    assert out["P_F1"]["cells_not_20_seeds"]
    assert out["decision"]["rank"] == 1 and out["U1"] == "current"
    assert out["reference_only"]["decision_if_rank1_clear"]["rank"] == "4/5"
    assert "rank_by_rule" not in out


def _break(info: dict, condition: str) -> dict:
    """One rank-1 condition made false."""
    if condition == "sealed_blob":
        info["seal"]["sealed_files"]["docs/processing/m210b_selector_floor_v2.md"][
            "matches"
        ] = False
    elif condition == "v1_doc_blob":
        info["seal"]["sealed_files"][V1_DOC]["blob_at_seal"] = "changed"
    elif condition == "interp_doc_blob":
        del info["seal"]["sealed_files"][INTERP_DOC]
    elif condition == "head":
        info["seal_at_end"]["head_commit"] = "other"
    elif condition == "dirty_start":
        info["seal"]["dirty_files"] = ["?? a.py"]
    elif condition == "dirty_end":
        info["seal_at_end"]["dirty_files"] = [" M a.py"]
    elif condition == "time_order":
        info["started_at"] = "2026-11-12T09:59:59+09:00"
    elif condition == "seed_overlap":
        info["seed_overlap"] = 1
    elif condition == "regression":
        info["wrapper_regression"]["fourtu_apex_candidates_v1.csv"][
            "byte_identical"
        ] = False
    elif condition == "regression_not_run":
        info["wrapper_regression"] = {"run": False}
    elif condition == "guards":
        info["guards"] = {"pass": False, "failures": ["x"]}
    elif condition == "floor_rules_blob":
        info["seal"]["sealed_files"][FLOOR_RULES]["blob_at_seal"] = "changed"
    elif condition == "generator_blob":
        del info["seal"]["sealed_files"][GENERATOR]
    return info


@pytest.mark.parametrize(
    "condition",
    [
        "sealed_blob", "v1_doc_blob", "interp_doc_blob", "head", "dirty_start",
        "dirty_end", "time_order", "seed_overlap", "regression",
        "regression_not_run", "guards", "floor_rules_blob", "generator_blob",
    ],
)  # fmt: skip
def test_every_rank1_condition_is_asserted(judge, condition) -> None:
    runs = fake_runs(judge)
    out = run_judge(judge, runs, info=_break(fake_info(runs), condition))
    assert out["rank1_identity"]["pass"] is False
    assert out["decision"]["rank"] == 1 and out["U1"] == "current"


def test_runs_sha_and_model_list_are_rank1(judge) -> None:
    runs = fake_runs(judge)
    assert run_judge(judge, runs, sha="other")["decision"]["rank"] == 1
    assert run_judge(judge, runs, models=(S4N, S6P))["decision"]["rank"] == 1
    missing_ref = {**REFERENCE_BLOBS, V1_DOC: None}
    assert run_judge(judge, runs, blobs=missing_ref)["decision"]["rank"] == 1


def test_reference_blobs_at_their_commits_are_the_spec_values(judge) -> None:
    blobs = {
        path: judge._git("rev-parse", f"{commit}:{path}")
        for path, commit in judge.REFERENCE_BLOB_COMMITS.items()
    }
    if any(blob is None for blob in blobs.values()):
        pytest.skip("git history with bc9a474 and 7e843d6 is not available")
    assert blobs[FLOOR_RULES].startswith("bad0f95c")
    assert blobs[GENERATOR].startswith("7cc2bf2b")


def test_reference_blobs_cover_rules_and_generator(judge) -> None:
    assert judge.REFERENCE_BLOB_COMMITS == {
        V1_DOC: "bc9a474",
        INTERP_DOC: "7e843d6",
        FLOOR_RULES: "bc9a474",
        GENERATOR: "7e843d6",
    }


@pytest.mark.parametrize("change", ["row_seed", "formula", "unit_seed"])
def test_seeds_off_the_v2_formula_are_rank1(judge, change) -> None:
    runs = fake_runs(judge)
    info = fake_info(runs)
    if change == "row_seed":
        first = runs[runs["model_id"] == S6P].index[0]
        runs.loc[first, "seed"] = str(20261120)  # a batch-1 seed
    elif change == "formula":
        info["protocol_detail"]["seed_formula"] = "20261120 + 1000 p + 100 q + s"
    else:
        info["protocol_detail"]["unit_noise_seed"] = 20261199
    out = run_judge(judge, runs, info=info)
    assert out["rank1_seeds"]["pass"] is False
    assert out["decision"]["rank"] == 1
    if change == "row_seed":
        assert out["rank1_seeds"]["rows_off_formula"] == 1


def test_fake_seeds_follow_the_formula(judge) -> None:
    runs = fake_runs(judge)
    seeds = run_judge(judge, runs)["rank1_seeds"]
    assert seeds == {
        "rows_off_formula": 0,
        "models_without_pair_index": [],
        "run_info_seed_formula": "20271120 + 1000 p + 100 q + s",
        "run_info_unit_noise_seed": 20271199,
        "pass": True,
    }


@pytest.mark.parametrize(
    ("late_lambda", "is_in_g"), [(10.0, True), (10.0 + 1e-6, False), (5.0, True)]
)
def test_g_needs_the_late_mute_to_leave_the_floor_region(
    judge, late_lambda, is_in_g
) -> None:
    runs = fake_runs(judge, s6p_late_lambda=late_lambda)
    info = fake_info(runs, s6p_late_lambda=late_lambda)
    out = run_judge(judge, runs, info=info)
    assert all(row["G"] is is_in_g for row in out["R_table"])
    excluded = out["G_excluded_no_escape"]
    assert (len(excluded) == 0) is is_in_g
    if is_in_g:
        assert out["decision"]["rank"] == "4/5"
        return
    assert len(excluded) == 4  # two rho x two mute pairs
    for row in excluded:
        assert row["e"] == pytest.approx(1.0) and row["Q"] and row["C"]
        assert row["Lambda"] == 30.0 and row["Lambda_l"] > 10
    decision = out["decision"]
    assert decision["rank"] == 9
    assert all(
        pair["G_fail_reasons"] == [judge.G_NO_ESCAPE]
        for pair in decision["G_fail_by_pair"]
    )


def test_g_fail_reasons_name_each_condition(judge) -> None:
    nan = float("nan")
    assert judge.g_fail_reasons(30.0, 1.0, 5.0) == []
    assert judge.g_fail_reasons(10.0, 1.0, 5.0) == ["lambda_e_le_10"]
    assert judge.g_fail_reasons(30.0, 0.45, 5.0) == ["floor_frac_lt_0.5"]
    assert judge.g_fail_reasons(30.0, 1.0, 10.5) == ["lambda_l_gt_10"]
    assert judge.g_fail_reasons(nan, 0.0, nan) == [
        "lambda_e_blank",
        "floor_frac_lt_0.5",
        "lambda_l_blank",
    ]


def test_blank_late_lambda_is_not_met_but_not_reported_as_no_escape(judge) -> None:
    runs = fake_runs(judge)
    is_late = (runs["model_id"] == S6P) & (runs["delta_ns"] == "2.35")
    runs.loc[is_late & (runs["rule"] == "current"), "Lambda"] = ""
    out = run_judge(judge, runs)
    assert all(not r["G"] for r in out["R_table"])
    assert all(r["G_fail_reasons"] == ["lambda_l_blank"] for r in out["R_table"])
    assert out["G_excluded_no_escape"] == []
    assert out["decision"]["rank"] == 9


def test_blank_lambda_column_fails_g(judge) -> None:
    runs = fake_runs(judge)
    is_early = (runs["model_id"] == S6P) & (runs["delta_ns"] == "0.0")
    runs.loc[is_early & (runs["rule"] == "current"), "Lambda"] = ""
    out = run_judge(judge, runs)
    zero_pairs = [r for r in out["R_table"] if r["delta_e"] == 0.0]
    assert all(not r["G"] and r["Lambda"] is None for r in zero_pairs)


def test_blank_rho_real_fails_q(judge) -> None:
    runs = fake_runs(judge)
    is_s6p = runs["model_id"] == S6P
    runs.loc[is_s6p & (runs["rule"] == "current"), "rho_real"] = ""
    out = run_judge(judge, runs)
    assert not any(row["Q"] for row in out["R_table"])
    assert out["decision"]["rank"] == 3


def test_b_full_worse_position_fails_f5b(judge) -> None:
    runs = fake_runs(judge)
    is_b_full = runs["rule"] == "B_full"
    runs.loc[is_b_full, "kept_nearest_t_ns"] = str(T_A0_NS + 0.5)
    out = run_judge(judge, runs)
    assert out["F_by_rule"]["B_full"]["F5B"]["pass"] is False
    assert out["rank_by_rule"]["B_full"] == 5 and out["U1"] == "B_min"


def test_judgment_scope_with_19_pairs_is_rank_5(judge) -> None:
    runs = fake_runs(judge)
    is_pipe_b_full = (runs["rule"] == "B_full") & runs["model_id"].isin([S4N, S6P])
    is_judged = (
        is_pipe_b_full & (runs["dt_ns"] == "0.098") & (runs["ramp_noise"] == "0")
    )
    keep = runs[is_judged].index[:19]
    blank = is_judged & ~runs.index.isin(keep)
    runs.loc[
        blank, ["kept_nearest_trace", "kept_nearest_sample", "kept_nearest_t_ns"]
    ] = ""
    runs.loc[blank, "n_kept_in_window"] = "0"
    out = run_judge(judge, runs)
    f5b = out["F_by_rule"]["B_full"]["F5B"]
    assert (f5b["n_pairs"], f5b["judgeable"], f5b["pass"]) == (19, False, False)
    assert out["rank_by_rule"]["B_full"] == 5


def test_blank_nearest_with_kept_in_window_is_not_judgeable(judge) -> None:
    runs = fake_runs(judge)
    first = runs[(runs["rule"] == "B_min") & (runs["model_id"] == S4N)].index[0]
    runs.loc[first, "kept_nearest_t_ns"] = ""  # n_kept_in_window stays 1
    out = run_judge(judge, runs)
    f5b = out["F_by_rule"]["B_min"]["F5B"]
    assert f5b["n_defect_rows"] == 1 and f5b["judgeable"] is False
    assert out["rank_by_rule"]["B_min"] == 5


@pytest.mark.parametrize("ramp", ["0", "1"])
def test_blank_false_deep_count_is_not_read_as_zero(judge, ramp) -> None:
    runs = fake_runs(judge)
    target = runs[
        (runs["rule"] == "B_full") & (runs["model_id"] == NP)
        & (runs["delta_ns"] == "0.0") & (runs["ramp_noise"] == ramp)
        & (runs["dt_ns"] == "0.098")
    ].index[0]  # fmt: skip
    runs.loc[target, "n_false_deep"] = ""
    out = run_judge(judge, runs)
    checks = out["F_by_rule"]["B_full"]
    failed = checks["F4_fail"] if ramp == "0" else checks["F4b_fail"]
    assert failed and failed[0]["judgeable"] is False
    assert out["rank_by_rule"]["B_full"] == 5


# ----------------------------------------------------------------- pre-run audit


@pytest.mark.parametrize(
    "expected",
    [("other", "v2"), ("v2seal", "otherblob"), (None, None)],
)
def test_expected_seal_commit_and_v2_blob_are_rank1(judge, expected) -> None:
    runs = fake_runs(judge)
    out = run_judge(judge, runs, expected=judge.ExpectedSeal(*expected))
    assert out["rank1_identity"]["pass"] is False
    assert out["decision"]["rank"] == 1


def test_run_head_must_be_the_expected_seal(judge) -> None:
    runs = fake_runs(judge)
    info = fake_info(runs)
    info["seal"]["head_commit"] = info["seal_at_end"]["head_commit"] = "later"
    out = run_judge(judge, runs, info=info)
    checks = out["rank1_identity"]["checks"]
    assert checks["run_head_is_expected_seal"] is False
    assert checks["head_same_start_end"] is True
    assert out["decision"]["rank"] == 1


def test_harness_git_failure_is_rank1(judge) -> None:
    runs = fake_runs(judge)
    info = fake_info(runs)
    info["seal_at_end"]["git_failures"] = ["status"]
    out = run_judge(judge, runs, info=info)
    assert out["rank1_identity"]["checks"]["harness_git_queries_ok"] is False
    assert out["decision"]["rank"] == 1


@pytest.mark.parametrize("column", ["accepted_apex", "floor_held_at_apex"])
def test_blank_judgment_flags_of_pipe_models_are_rank1(judge, column) -> None:
    runs = fake_runs(judge)
    first = runs[(runs["model_id"] == S4N) & (runs["dt_ns"] == "0.098")].index[0]
    runs.loc[first, column] = ""
    out = run_judge(judge, runs)
    assert out["P_F1"]["pipe_judgment_blanks"][column] == 1
    assert out["decision"]["rank"] == 1


def test_empty_expected_model_set_is_rank1(judge) -> None:
    runs = fake_runs(judge)
    out = run_judge(judge, runs.iloc[0:0], models=())
    assert out["P_F1"]["models_json_v2_nonempty"] is False
    assert out["decision"]["rank"] == 1


def test_w10_no_pipe_calibration_pair_is_rank1(judge) -> None:
    runs = fake_runs(judge)
    info = fake_info(runs)
    info["protocol_detail"]["no_pipe_calibration_pair"] = "D16_S040"
    out = run_judge(judge, runs, info=info)
    assert out["P_F1"]["pass"] is False and out["decision"]["rank"] == 1


def test_v2_expected_models_are_the_nine_ids(judge) -> None:
    nine = {
        "S6P_D14_S040_J2", "S4N_D14_S040", "S6P_D16_S040_J3", "S4N_D16_S040",
        "S6P_D15_S045_J4", "S4N_D15_S045", "S6P_D13_S010_J5", "S4N_D13_S010",
        "S6P_NP_S040_J2",
    }  # fmt: skip
    expected = set(judge.V2_EXPECTED_MODELS)
    assert expected == nine


@pytest.mark.parametrize("dropped", [NP, S6P])
def test_dropping_a_model_from_the_json_is_rank1(judge, dropped) -> None:
    runs = fake_runs(judge)
    models = tuple(m for m in MODELS if m != dropped)
    out = run_judge(judge, runs, models=models)
    assert out["P_F1"]["models_match"] is False
    assert out["decision"]["rank"] == 1


def test_dropping_a_model_from_runs_and_run_info_is_rank1(judge) -> None:
    runs = fake_runs(judge)
    runs = runs[runs["model_id"] != NP]
    info = fake_info(runs)
    info["models_run"] = [S4N, S6P]
    out = run_judge(judge, runs, info=info)
    assert out["P_F1"]["models_match"] is False and out["decision"]["rank"] == 1


def test_the_real_model_set_rejects_a_smaller_run(judge) -> None:
    runs = fake_runs(judge)
    expected = judge.ExpectedSeal("v2seal", "v2")
    out = judge.judge(
        runs, fake_info(runs), list(MODELS), SHA, REFERENCE_BLOBS, expected
    )
    assert out["P_F1"]["models_match"] is False and out["decision"]["rank"] == 1


@pytest.mark.parametrize("change", ["pair", "sigma_raw", "control"])
def test_w10_no_pipe_line_uses_the_p0_calibration(judge, change) -> None:
    runs = fake_runs(judge)
    info = fake_info(runs)
    assert run_judge(judge, runs, info=info)["P_F1"]["W10_calibration"]["pass"]
    no_pipe = info["models"][NP]
    if change == "pair":
        no_pipe["calibration_pair"] = "D16_S040"
    elif change == "sigma_raw":
        no_pipe["harness"]["calibration"] = {
            **no_pipe["harness"]["calibration"],
            "sigma_raw_by_rho": {"10": 0.009, "20": 0.004},
        }
    else:
        no_pipe["harness"]["calibration"] = {
            **no_pipe["harness"]["calibration"],
            "control_id": "S4S6/S4N_D16_S040",
        }
    out = run_judge(judge, runs, info=info)
    assert out["P_F1"]["W10_calibration"]["pass"] is False
    assert out["decision"]["rank"] == 1


def test_judge_cli_requires_the_expected_seal_commit(judge) -> None:
    with pytest.raises(SystemExit):
        judge.parse_args([])
    assert (
        judge.parse_args(["--expected-seal-commit", "abc"]).expected_seal_commit
        == "abc"
    )


def test_lambda_rounding_check_reports_a_boundary_difference(judge) -> None:
    # Column 10.0 (4 digits) is in G; the unrounded run_info value 10.00004 is not.
    runs = fake_runs(judge, s6p_late_lambda=10.0)
    info = fake_info(runs, s6p_late_lambda=10.00004)
    out = run_judge(judge, runs, info=info)
    assert all(row["G"] for row in out["R_table"])
    assert out["decision"]["rank"] == "4/5"
    report = out["lambda_rounding_check"]
    assert len(report) == 4
    for entry in report:
        assert entry["Lambda_column"][1] == 10.0
        assert entry["Lambda_unrounded"][1] == pytest.approx(10.00004)
        assert (entry["G_column"], entry["G_unrounded"], entry["differs"]) == (
            True,
            False,
            True,
        )
    far = run_judge(judge, fake_runs(judge))
    assert far["lambda_rounding_check"] == []
