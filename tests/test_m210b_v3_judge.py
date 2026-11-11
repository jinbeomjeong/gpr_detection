"""v3 judgment script (m210b_selector_floor_v3.md sections 3, 5, 6 V3-4, V3-6):
40 seeds, the v3 model set, seeds and W10-form pair, and the F3-S report, on small
fake ``runs_v3.csv`` tables. No harness output is needed."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JUDGE_PATH = PROJECT_ROOT / "analysis" / "processing" / "m210b_v3_judge.py"
S4N, S6P, NP = "S4N_D14_S038", "S6P_D14_S038_J6", "S6P_NP_S038_J6"
MODELS = (S4N, S6P, NP)
PAIR_INDEX = {S4N: 0, S6P: 0, NP: 5}
APEX_TRACE, T_A0_NS = 80, 30.0
SHA = "sha"
V1_DOC = "docs/processing/m210b_selector_floor.md"
V2_DOC = "docs/processing/m210b_selector_floor_v2.md"
V3_DOC = "docs/processing/m210b_selector_floor_v3.md"
INTERP_DOC = "docs/processing/m210b_harness_interpretation.md"
FLOOR_RULES = "analysis/processing/m210b_floor_rules.py"
GENERATOR = "analysis/processing/fourtu_apex_candidate_select.py"
REFERENCE_BLOBS = {
    V1_DOC: "v1blob",
    INTERP_DOC: "interpblob",
    V2_DOC: "v2blob",
    FLOOR_RULES: "rulesblob",
    GENERATOR: "generatorblob",
}


@pytest.fixture(scope="module")
def judge():
    spec = importlib.util.spec_from_file_location("m210b_v3_judge", JUDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["m210b_v3_judge"] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------- fake run


def _row(model, rho, seed, rule, delta, ramp=0, dt=0.098, **values) -> dict:
    is_pipe = model != NP
    is_current = rule == "current"
    row = {
        "model_id": model, "rho": rho, "seed": seed, "rule": rule,
        "delta_ns": delta, "ramp_noise": ramp, "dt_ns": dt, "t_zero_ns": 3.4,
        "zero_rule": "air", "L": 1.0, "F": 0.05, "sigma_n": 0.01,
        "rho_real": 20.0 if is_pipe else "", "rho_real_cal": 20.0,
        "Lambda": 30.0 if model == S6P else (0.5 if is_pipe else ""),
        "argmax_level_periods": 0.0,
        "accepted_apex": True if is_pipe else "",
        "peak_on_crest": True if is_pipe else "",
        "kept_apex": True if is_pipe else "",
        "kept_apex_trace": APEX_TRACE if is_pipe else "",
        "kept_apex_sample": 306.0 if is_pipe else "", "lost_reason": "",
        "floor_held_at_apex": model == S6P if is_pipe else "",
        "n_peaks": 50, "cap300_hit": False, "n_false_deep": 0,
        "n_false_near_surface": 0, "n_false_record_end": 0,
        "kept_nearest_trace": APEX_TRACE if is_pipe else "",
        "kept_nearest_sample": 306.0 if is_pipe else "",
        "kept_nearest_t_ns": T_A0_NS + 0.02 if is_pipe else "",
        "n_kept_in_window": 1 if is_pipe else "",
        "depth_nearest_m": 1.4 if is_pipe else "",
        "depth_nearest_status": "ok" if is_pipe else "",
        # F3-S: current rows only (blank for the other rules by definition).
        "accepted_apex_strong_start": (True if is_pipe else "") if is_current else "",
        "n_strong_starts": 3 if is_current else "",
    }  # fmt: skip
    row.update(values)
    return row


def _seed(p, rho_index, s) -> int:
    return 20281120 + 1000 * p + 100 * rho_index + s


def fake_runs(judge) -> pd.DataFrame:
    """A complete v3 table where R holds and every candidate passes."""
    rows = []
    for model in MODELS:
        p = PAIR_INDEX[model]
        for q, rho in enumerate(judge.RHOS):
            for s in range(judge.N_SEEDS):
                for rule in judge.RULES:
                    for delta in judge.DELTAS:
                        extra = {}
                        # Coupling: current loses the S6P apex unless the mute is late.
                        if model == S6P and rule == "current" and delta != 2.35:
                            extra["accepted_apex"] = False
                        if model == S6P and delta == 2.35:
                            extra["Lambda"] = 5.0
                        rows.append(
                            _row(model, rho, _seed(p, q, s), rule, delta, **extra)
                        )
        for s in range(judge.N_SEEDS):
            seed = _seed(p, 0, s)
            for rule in ("current", "B_full"):
                rows.append(_row(model, 10, seed, rule, 0.0, dt=0.0118))
            if model == NP:
                for rule in judge.RULES:
                    for delta in judge.DELTAS:
                        rows.append(_row(model, 10, seed, rule, delta, ramp=1))
    return pd.DataFrame(rows).astype(str)


def fake_info(runs: pd.DataFrame) -> dict:
    def harness(apex, lam, late=None):
        lams = {f"{d:+.2f}": lam for d in (-1.0, 0.0, 1.0)}
        lams["+2.35"] = lam if late is None else late
        return {
            "apex_trace": apex, "t_a0_ns": T_A0_NS if apex else None,
            "L_noise_free_current": {k: v * 0.04 for k, v in lams.items()},
            "Lambda": lams if apex else dict.fromkeys(lams),
            "argmax_level_periods_noise_free": dict.fromkeys(lams, 0.0),
            "A0_self": 0.04 if apex else None,
            "calibration": {
                "control_id": f"S4S6/{S4N}", "a0": 0.04, "sigma_u": 0.5,
                "sigma_raw_by_rho": {"10": 0.008, "20": 0.004},
            },
        }  # fmt: skip

    sealed = {
        path: {"blob_at_seal": blob, "matches": True}
        for path, blob in REFERENCE_BLOBS.items()
    }
    sealed[V3_DOC] = {"blob_at_seal": "v3", "matches": True}
    seal = {
        "seal_commit_full": "v3seal", "seal_commit_time": "2026-11-12T10:00:00+09:00",
        "sealed_files": sealed, "dirty_files": [], "head_commit": "v3seal",
        "git_failures": [],
    }  # fmt: skip
    return {
        "models_run": list(MODELS),
        "models": {
            S4N: {"harness": harness(APEX_TRACE, 0.5), "pipe_top_depth_m": 1.4},
            S6P: {"harness": harness(APEX_TRACE, 30.0, 5.0), "pipe_top_depth_m": 1.4},
            NP: {
                "harness": harness(None, 0.0),
                "pipe_top_depth_m": None,
                "calibration_pair": "D14_S038",
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
            "seed_formula": "20281120 + 1000 p + 100 q + s",
            "unit_noise_seed": 20281199,
            "no_pipe_calibration_pair": "D14_S038",
        },
        "started_at": "2026-11-12T11:00:00+09:00",
        "finished_at": "2026-11-12T11:40:00+09:00",
    }


FAKE_NO_PIPE = frozenset({NP})
FAKE_RANK2 = frozenset({S4N})


def run_judge(
    judge, runs, info=None, models=MODELS, expected_models=MODELS,
    expected_no_pipe=FAKE_NO_PIPE, expected_rank2=FAKE_RANK2,
):  # fmt: skip
    info = fake_info(runs) if info is None else info
    expected = judge.ExpectedSeal("v3seal", "v3")
    return judge.judge(
        runs, info, list(models), SHA, REFERENCE_BLOBS, expected,
        frozenset(expected_models), frozenset(expected_no_pipe),
        frozenset(expected_rank2),
    )  # fmt: skip


@pytest.fixture(scope="module")
def base_runs(judge) -> pd.DataFrame:
    return fake_runs(judge)


# ----------------------------------------------------------------- constants


def test_v3_constants_follow_the_spec(judge) -> None:
    assert judge.N_SEEDS == 40
    assert judge.RULES == ("current", "B_full", "B_min", "C_none")
    assert (judge.JUDGED, judge.REPORTED) == (("B_full", "B_min"), ("C_none",))
    assert judge.SEED_FORMULA == "20281120 + 1000 p + 100 q + s"
    assert judge.UNIT_NOISE_SEED == 20281199
    assert judge.NO_PIPE_CALIBRATION_PAIR == "D14_S038"
    assert judge.RUNS_CSV == "runs_v3.csv"
    nine = {
        "S6P_D14_S038_J6", "S4N_D14_S038", "S6P_D13_S042_J7", "S4N_D13_S042",
        "S4N_D16_S045", "S4N_D15_S048", "S6P_D13_S012_J8", "S4N_D13_S012",
        "S6P_NP_S038_J6",
    }  # fmt: skip
    expected = set(judge.V3_EXPECTED_MODELS)
    assert expected == nine


def test_reference_blobs_are_the_v2_seal_for_rules_and_wrapper(judge) -> None:
    assert judge.REFERENCE_BLOB_COMMITS[FLOOR_RULES] == "752e783"
    assert judge.REFERENCE_BLOB_COMMITS[GENERATOR] == "752e783"
    assert judge.REFERENCE_BLOB_COMMITS[V2_DOC] == "752e783"
    rules_blob = judge._git("rev-parse", f"752e783:{FLOOR_RULES}")
    generator_blob = judge._git("rev-parse", f"752e783:{GENERATOR}")
    if rules_blob is None or generator_blob is None:
        pytest.skip("git history with 752e783 is not available")
    assert rules_blob.startswith("bad0f95c")
    assert generator_blob.startswith("7cc2bf2b")


# ----------------------------------------------------------------- P-F1, rank 1


def test_fake_run_passes_pf1_and_reaches_rank_4(judge, base_runs) -> None:
    out = run_judge(judge, base_runs)
    assert out["P_F1"]["pass"], out["P_F1"]
    assert out["rank1_identity"]["pass"], out["rank1_identity"]
    assert out["rank1_seeds"]["pass"], out["rank1_seeds"]
    assert out["decision"]["rank"] == "4/5"
    assert out["rank_by_rule"] == {"B_full": 4, "B_min": 4}
    assert out["rank_by_rule_reported"] == {"C_none": 4}
    assert out["U1"] == "B_full"
    assert out["F_by_rule"]["B_full"]["F5B"]["n_pairs"] == 2 * 2 * 4 * 40


def test_pf1_with_39_seeds_is_rank_1(judge, base_runs) -> None:
    is_cell = (base_runs["model_id"] == S4N) & (base_runs["rule"] == "B_min")
    is_cell &= (base_runs["delta_ns"] == "1.0") & (base_runs["rho"] == "20")
    drop = base_runs[is_cell & (base_runs["ramp_noise"] == "0")].index[0]
    short = base_runs.drop(index=drop)
    out = run_judge(judge, short)
    assert out["P_F1"]["cells_not_40_seeds"]
    assert out["decision"]["rank"] == 1 and out["U1"] == "current"


@pytest.mark.parametrize("dropped", [NP, S4N])
def test_missing_expected_model_is_rank_1(judge, base_runs, dropped) -> None:
    models = tuple(m for m in MODELS if m != dropped)
    out = run_judge(judge, base_runs, models=models)
    assert out["P_F1"]["models_match"] is False
    assert out["decision"]["rank"] == 1


def test_the_real_v3_model_set_rejects_a_smaller_run(judge, base_runs) -> None:
    out = run_judge(judge, base_runs, expected_models=judge.V3_EXPECTED_MODELS)
    assert out["P_F1"]["models_match"] is False and out["decision"]["rank"] == 1


@pytest.mark.parametrize(
    "seed",
    [20271120, 20261120, 20281120 + 40],  # v2, batch 1, s = 40
)
def test_seed_off_the_v3_formula_is_rank_1(judge, base_runs, seed) -> None:
    runs = base_runs.copy()
    runs.loc[runs[runs["model_id"] == S6P].index[0], "seed"] = str(seed)
    out = run_judge(judge, runs)
    assert out["rank1_seeds"]["rows_off_formula"] == 1
    assert out["decision"]["rank"] == 1


def test_w10_form_needs_the_r1_pair(judge, base_runs) -> None:
    info = fake_info(base_runs)
    info["models"][NP]["calibration_pair"] = "D13_S042"
    out = run_judge(judge, base_runs, info=info)
    assert out["P_F1"]["W10_calibration"]["pass"] is False
    assert out["decision"]["rank"] == 1


def test_v3_doc_blob_and_rules_blob_are_rank_1(judge, base_runs) -> None:
    info = fake_info(base_runs)
    info["seal"]["sealed_files"][FLOOR_RULES]["blob_at_seal"] = "changed"
    assert run_judge(judge, base_runs, info=info)["decision"]["rank"] == 1
    info = fake_info(base_runs)
    info["seal"]["sealed_files"][V3_DOC]["blob_at_seal"] = "other"
    assert run_judge(judge, base_runs, info=info)["decision"]["rank"] == 1


# ----------------------------------------------------------------- rank 2 (X6)

F_TARGET = "S4N_D16_S045"


def _rank2_rows(control_successes: int, target_successes: int) -> pd.DataFrame:
    """rho 20 current Delta 0 cells of the pair control and one F target."""
    rows = []
    for model, successes in ((S4N, control_successes), (F_TARGET, target_successes)):
        for s in range(40):
            rows.append(
                {"model_id": model, "rho_i": 20, "rule": "current", "delta": 0.0,
                 "accepted_apex_b": s < successes}
            )  # fmt: skip
    return pd.DataFrame(rows)


@pytest.fixture
def rank2_roles(judge):
    return judge.ModelRoles(
        pipe=(S4N, F_TARGET, S6P), s6p_control={S6P: S4N}, no_pipe=()
    )


def test_rank2_ignores_an_f_target_below_half(judge, rank2_roles) -> None:
    rows = _rank2_rows(control_successes=34, target_successes=10)
    decision = judge.decide(True, rows, rank2_roles, table=[])
    assert decision["rank"] != 2  # no qualified pair here, so rank 3
    assert judge.rank2_models(rank2_roles) == (S4N,)
    assert judge.rank2_report(rows, rank2_roles) == {F_TARGET: pytest.approx(0.25)}


def test_rank2_checks_the_pair_control(judge, rank2_roles) -> None:
    rows = _rank2_rows(control_successes=19, target_successes=40)
    decision = judge.decide(True, rows, rank2_roles, table=[])
    assert decision["rank"] == 2
    assert [model for model, _ in decision["detail"]] == [S4N]


def test_rank2_keys_in_the_judgment(judge, base_runs) -> None:
    out = run_judge(judge, base_runs)
    assert out["rank2_checked_models"] == [S4N]
    assert out["rank2_f_targets_reported"] == {}


def test_fixed_role_sets_are_the_spec_values(judge) -> None:
    assert set(judge.V3_EXPECTED_NO_PIPE) == {"S6P_NP_S038_J6"}
    assert set(judge.V3_EXPECTED_RANK2_CONTROLS) == {
        "S4N_D14_S038", "S4N_D13_S042", "S4N_D13_S012",
    }  # fmt: skip


def test_fixed_roles_pass_and_are_reported(judge, base_runs) -> None:
    roles = run_judge(judge, base_runs)["P_F1"]["roles_fixed"]
    assert roles == {
        "no_pipe_expected": [NP], "no_pipe_found": [NP],
        "rank2_controls_expected": [S4N], "rank2_controls_found": [S4N],
        "pass": True,
    }  # fmt: skip


def test_pipe_model_without_apex_record_is_rank_1(judge, base_runs) -> None:
    # A pipe S6P whose apex record is missing would be read as a no-pipe line.
    info = fake_info(base_runs)
    info["models"][S6P]["harness"]["apex_trace"] = None
    out = run_judge(judge, base_runs, info=info)
    roles = out["P_F1"]["roles_fixed"]
    assert roles["no_pipe_found"] == sorted([NP, S6P]) and roles["pass"] is False
    assert out["decision"]["rank"] == 1


def test_shrunken_pair_control_set_is_rank_1(judge, base_runs) -> None:
    info = fake_info(base_runs)
    info["models"][S6P]["harness"]["calibration"] = {
        **info["models"][S6P]["harness"]["calibration"],
        "control_id": "S4S6/S4N_D13_S042",  # not a model of this run
    }
    out = run_judge(judge, base_runs, info=info)
    roles = out["P_F1"]["roles_fixed"]
    assert roles["rank2_controls_found"] == [] and roles["pass"] is False
    assert out["decision"]["rank"] == 1


def test_real_role_sets_reject_the_fake_run(judge, base_runs) -> None:
    out = run_judge(
        judge, base_runs,
        expected_no_pipe=judge.V3_EXPECTED_NO_PIPE,
        expected_rank2=judge.V3_EXPECTED_RANK2_CONTROLS,
    )  # fmt: skip
    assert out["P_F1"]["roles_fixed"]["pass"] is False
    assert out["decision"]["rank"] == 1


# ----------------------------------------------------------------- F3-S


def test_f3s_label_puts_h_a_first(judge) -> None:
    assert judge.f3s_label(["H-C", "H-A", "H-C"]) == "H-A"
    assert judge.f3s_label(["H-C", "H-C"]) == "H-C"
    assert judge.f3s_label([]) == "H-B"


def _set_rate(runs, rule, column, successes, rho="20"):
    """Make ``successes`` of the 40 S4N Delta-0 judgment rows of ``rule`` True."""
    is_cell = (runs["model_id"] == S4N) & (runs["rule"] == rule)
    is_cell &= (runs["delta_ns"] == "0.0") & (runs["rho"] == rho)
    is_cell &= (runs["ramp_noise"] == "0") & (runs["dt_ns"] == "0.098")
    index = runs[is_cell].index
    runs.loc[index, column] = "False"
    runs.loc[index[:successes], column] = "True"


def _condition(out, rule, rho=20):
    conditions = out["F3S_by_condition"][rule]["conditions"]
    return next(c for c in conditions if c["model"] == S4N and c["rho"] == rho)


@pytest.mark.parametrize(
    ("strong_successes", "reading"),
    [(26, "H-C"), (27, "H-A")],  # gap_S = 26/40 - 22/40 = 0.1 (boundary) / 0.125
)
def test_f3s_reading_on_a_failing_f3_condition(
    judge, base_runs, strong_successes, reading
) -> None:
    runs = base_runs.copy()
    _set_rate(runs, "current", "accepted_apex", 34)  # r_cur 0.85
    _set_rate(runs, "B_full", "accepted_apex", 22)  # r_X 0.55: F3 fails
    _set_rate(runs, "current", "accepted_apex_strong_start", strong_successes)
    out = run_judge(judge, runs)
    condition = _condition(out, "B_full")
    assert condition["F3_fails"] is True
    assert condition["r_cur_S"] == pytest.approx(strong_successes / 40)
    assert condition["reading"] == reading
    assert out["F3S_by_condition"]["B_full"]["summary"] == {
        "H-A": int(reading == "H-A"),
        "H-C": int(reading == "H-C"),
    }
    assert out["F3S_by_condition"]["B_full"]["label"] == reading
    # F3-S is a report: the rank comes from F3 alone.
    assert out["rank_by_rule"]["B_full"] == 5


def test_f3s_reads_h_b_when_f3_passes_everywhere(judge, base_runs) -> None:
    out = run_judge(judge, base_runs)
    for rule in ("B_full", "B_min", "C_none"):
        assert out["F3S_by_condition"][rule]["summary"] == "H-B"
        assert out["F3S_by_condition"][rule]["label"] == "H-B"
        assert all(
            c["reading"] is None for c in out["F3S_by_condition"][rule]["conditions"]
        )


def test_f3s_counts_blank_strong_starts_as_not_met(judge, base_runs) -> None:
    runs = base_runs.copy()
    is_current_s4n = (runs["model_id"] == S4N) & (runs["rule"] == "current")
    first = runs[is_current_s4n & (runs["delta_ns"] == "0.0")].index[0]
    runs.loc[first, "accepted_apex_strong_start"] = ""
    out = run_judge(judge, runs)
    assert out["F3S_by_condition"]["blank_strong_start_in_current_s4n_rows"] == 1
    rho = int(runs.loc[first, "rho"])
    assert _condition(out, "B_full", rho)["r_cur_S"] == pytest.approx(39 / 40)
    # Blank F3-S cells are not P-F1 blanks (F3-S is a report, v3 V3-2).
    assert out["P_F1"]["pass"] is True
