"""K7 judgment (m210_k7_generator.md v1.1 4-5, 4-6, 8) on fake run tables.

The base table is built so that both candidates reach rank 7; every other test
changes one thing and checks the rank it must give (ranks 1-7 and their
boundaries, fixed sets, cell counts, empty cases, the a1 -> a1b2 order).
"""

import csv
import hashlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JUDGE_PATH = PROJECT_ROOT / "analysis" / "processing" / "m210_k7_judge.py"
SEAL = "c" * 40
BLOBS = {
    "docs/processing/m210_k7_generator.md": "spec-blob",
    "analysis/processing/fourtu_apex_candidate_select.py": "generator-blob",
    "analysis/processing/m210_k7_judge.py": "judge-blob",
}
N_PIPES = {
    "K7_P2_D070": 2,
    "K7_P2_D070_L": 2,
    "K7_P3_D080": 3,
    "K7_P1_D150_S003": 1,
    "K7_P1_D110_S002": 1,
    "K7_P1_D060": 1,
    "K7_PVC_D070": 1,
    "K7_BG_L": 0,
}
MODELS_JSON_SHA = "e" * 64
COLUMNS = (
    "model_id", "model_index", "rho", "seed", "seed_index", "noise_free", "variant",
    "pipe_index", "n_pipes", "accepted_apex", "kept_apex", "lost_reason",
    "qc_primary", "depth_error_m", "n_kept", "n_qc_passing", "n_qc_unexplained",
    "n_window_starts", "c_item", "c_end_offset_traces", "c_start_is_local_max",
    "c_end_is_local_max", "c_tau2_vertex_offset_traces",
)  # fmt: skip


@pytest.fixture(scope="module")
def judge():
    spec = importlib.util.spec_from_file_location("m210_k7_judge_test", JUDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------- fake tables

Outcome = Callable[[str, str, int, int, int], dict]


def base_outcome(model: str, variant: str, rho: int, seed: int, pipe: int) -> dict:
    """Both mechanisms reproduced by cur; a1 fixes a, a1b2 fixes a and b."""
    is_defect_a = model in ("K7_P2_D070", "K7_P2_D070_L")
    is_defect_b = model == "K7_P3_D080"
    kept = True
    if is_defect_a:
        kept = variant != "cur"
    elif is_defect_b:
        kept = variant in ("a1b2", "a1b1")
    return {
        "accepted_apex": True,
        "kept_apex": kept,
        "qc_primary": kept,
        "depth_error_m": 0.01,
        "n_qc_unexplained": 0,
    }


def _text(value) -> str:
    if value is None:
        return ""
    return str(value)


def make_rows(judge, outcome: Outcome = base_outcome) -> list[dict]:
    rows = []
    for m, model in enumerate(judge.MODEL_IDS):
        n_pipes = N_PIPES[model]
        pipes = list(range(n_pipes)) or [None]
        for variant in judge.VARIANTS:
            for pipe in pipes:
                rows.append(_row(model, m, None, None, variant, pipe, n_pipes, {}))
            for q, rho in enumerate(judge.RHOS):
                for s in range(judge.N_SEEDS):
                    for pipe in pipes:
                        values = (
                            {}
                            if pipe is None
                            else outcome(model, variant, rho, s, pipe)
                        )
                        if pipe is None:
                            values = {"n_qc_unexplained": 0}
                        rows.append(
                            _row(model, m, (q, rho), s, variant, pipe, n_pipes, values)
                        )
    return rows


def _row(model, m, rho_pair, s, variant, pipe, n_pipes, values) -> dict:
    is_noise_free = rho_pair is None
    seed = None if is_noise_free else 20301112 + 1000 * m + 100 * rho_pair[0] + s
    row = dict.fromkeys(COLUMNS, "")
    row.update(
        model_id=model, model_index=str(m), noise_free=str(is_noise_free),
        variant=variant, n_pipes=str(n_pipes), n_kept="3", n_qc_passing="1",
        n_window_starts="2", n_qc_unexplained="0",
    )  # fmt: skip
    if not is_noise_free:
        row.update(rho=str(rho_pair[1]), seed=str(seed), seed_index=str(s))
    if pipe is not None:
        row.update(pipe_index=str(pipe), accepted_apex="True", kept_apex="True",
                   qc_primary="True", depth_error_m="0.01")  # fmt: skip
    for key, value in values.items():
        row[key] = _text(value)
    if row["kept_apex"] == "False":
        row["lost_reason"] = "cap6"
    return row


def write_runs(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_info(judge, runs_sha: str) -> dict:
    sealed = {
        path: {"blob_at_seal": BLOBS.get(path, f"blob-{i}"), "matches": True}
        for i, path in enumerate(judge.K7_SEALED_FILES)
    }
    seal = {
        "seal_commit_full": SEAL,
        "seal_commit_time": "2026-11-12T10:00:00+09:00",
        "head_commit": SEAL,
        "dirty_files": [],
        "git_failures": [],
        "sealed_files": sealed,
    }
    calibration = {"source_model": "K7_P2_D070_L", "sigma_raw_by_rho": {"10": 1.0}}
    models = {
        model: {
            "n_pipes": n,
            "calibration": calibration
            if model in ("K7_BG_L", "K7_P2_D070_L")
            else {"source_model": model, "sigma_raw_by_rho": {"10": 2.0}},
        }
        for model, n in N_PIPES.items()
    }
    fourtu = {
        name: {"byte_identical": True, "regenerated_sha256": prefix + "0" * 56}
        for name, prefix in judge.FOURTU_SHA256.items()
    }
    return {
        "protocol": "k7",
        "started_at": "2026-11-12T12:00:00+09:00",
        "seal": seal,
        "seal_at_end": dict(seal),
        "sealed_files": sorted(sealed),
        "generator_blob": BLOBS[judge.GENERATOR_PATH],
        "seeds": {"seed_formula": judge.SEED_FORMULA, "unit_noise_seed": 20301199},
        "seed_overlap": 0,
        "plan": {
            "model_ids": list(judge.MODEL_IDS),
            "variants": list(judge.VARIANTS),
            "rhos": [10, 20],
            "n_seeds": 40,
            "calibration_pairs": {"K7_BG_L": "K7_P2_D070_L"},
        },
        "models": models,
        "runs_csv_sha256": runs_sha,
        "inputs": {"models_json": {"path": "m.json", "sha256": MODELS_JSON_SHA}},
        "guards": {"pass": True},
        "tests": {"pass": True},
        "mkl_cbwr": "COMPATIBLE",
        "g0": {
            "run": True,
            "fourtu": {"run": True, **fourtu},
            "s1g": {"identical": True},
            "m210b_v3": {"identical": True},
        },
    }


def json_models(judge) -> dict:
    return {
        "sha256": MODELS_JSON_SHA,
        "model_ids": list(judge.MODEL_IDS),
        "n_pipes": dict(N_PIPES),
        "role_sets": {k: list(v) for k, v in judge.ROLE_SETS.items()},
    }


IDENTITY_FILES = {
    "pre_run.txt": f"time 2026-11-12T12:00:00+09:00\nHEAD {SEAL}\ndirty 0\n",
    "post_run.txt": f"time 2026-11-12T13:00:00+09:00\nHEAD {SEAL}\ndirty 0\n",
}


def run_judge(judge, tmp_path, rows=None, edit_info=None, expected=None, files=None):
    rows = make_rows(judge) if rows is None else rows
    path = write_runs(tmp_path / "runs_k7.csv", rows)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    info = make_info(judge, sha)
    if edit_info is not None:
        edit_info(info)
    expected = expected or judge.ExpectedSeal(
        SEAL, dict(BLOBS), BLOBS[judge.JUDGE_PATH]
    )
    return judge.judge(
        judge.read_runs(path), info, json_models(judge), sha, expected,
        files or IDENTITY_FILES,
    )  # fmt: skip


def with_outcome(judge, change: Callable[[dict, str, str, int, int, int], None]):
    def outcome(model, variant, rho, seed, pipe):
        values = base_outcome(model, variant, rho, seed, pipe)
        change(values, model, variant, rho, seed, pipe)
        return values

    return make_rows(judge, outcome)


# ----------------------------------------------------------------- rank 7 base


def test_base_table_adopts_a1b2(judge, tmp_path) -> None:
    out = run_judge(judge, tmp_path)
    assert out["rank1_reasons"] == []
    assert out["ranks"]["a1"]["rank"] == 7
    assert out["ranks"]["a1b2"]["rank"] == 7
    assert out["adopted"] == "a1b2"
    assert out["K_R"] == {"a": True, "b": True}
    assert out["items"]["a1"]["K_F1"]["n_cells"] == 4
    assert out["items"]["a1b2"]["K_F1"]["n_cells"] == 6
    assert out["items"]["a1"]["K_F2"]["n_cells"] == 8
    assert out["items"]["a1"]["K_F3"]["n_cells"] == 16


def test_pipe_mean_rate_uses_every_pipe(judge, tmp_path) -> None:
    def change(values, model, variant, rho, seed, pipe):
        if model == "K7_P2_D070" and variant == "a1" and pipe == 1:
            values["kept_apex"] = seed < 20  # pipe 1: 0.5, pipe 0: 1.0

    out = run_judge(judge, tmp_path, rows=with_outcome(judge, change))
    rates = out["rates"]["K7_P2_D070|10|a1"]
    assert rates["kept"] == pytest.approx(0.75)
    assert rates["kept_per_pipe"] == {"0": 1.0, "1": 0.5}


# ----------------------------------------------------------------- rank 1


@pytest.mark.parametrize(
    "edit",
    [
        lambda info: info.update(protocol="v3"),
        lambda info: info.update(sealed_files=info["sealed_files"][1:]),
        lambda info: info.update(seed_overlap=1),
        lambda info: info["seal"].update(dirty_files=["M x.py"]),
        lambda info: info["seal_at_end"].update(head_commit="other"),
        lambda info: info["seal"].update(git_failures=["status"]),
        lambda info: info.update(started_at="2026-11-12T09:00:00+09:00"),
        lambda info: info["tests"].update({"pass": False}),
        lambda info: info["guards"].update({"pass": False}),
        lambda info: info["g0"]["s1g"].update(identical=False),
        lambda info: info["g0"]["fourtu"]["fourtu_apex_candidates_v1.csv"].update(
            regenerated_sha256="ffff"
        ),
        lambda info: info.update(generator_blob="other"),
        lambda info: info["plan"].update(calibration_pairs={}),
        lambda info: info["models"]["K7_BG_L"]["calibration"].update(source_model="x"),
        lambda info: info.update(runs_csv_sha256="0"),
    ],
)
def test_identity_failures_give_rank_1(judge, tmp_path, edit) -> None:
    out = run_judge(judge, tmp_path, edit_info=edit)
    assert out["rank1_reasons"]
    assert out["ranks"]["a1"]["rank"] == 1
    assert out["ranks"]["a1b2"]["rank"] == judge.NOT_JUDGED
    assert out["adopted"] == "current"


def test_judge_blob_must_be_the_seal_blob(judge, tmp_path) -> None:
    expected = judge.ExpectedSeal(SEAL, dict(BLOBS), "edited-judge")
    out = run_judge(judge, tmp_path, expected=expected)
    assert "identity: judge_blob_is_seal_blob" in out["rank1_reasons"]


def test_identity_files_must_be_harness_format(judge, tmp_path) -> None:
    files = dict(IDENTITY_FILES)
    files["post_run.txt"] = "main session note\n"
    out = run_judge(judge, tmp_path, files=files)
    assert "identity: post_run_txt_harness_format" in out["rank1_reasons"]


def test_missing_model_gives_rank_1(judge, tmp_path) -> None:
    rows = [r for r in make_rows(judge) if r["model_id"] != "K7_P1_D060"]
    out = run_judge(judge, tmp_path, rows=rows)
    assert "fixed_sets: run_models_are_expected" in out["rank1_reasons"]
    assert out["ranks"]["a1"]["rank"] == 1


def test_models_json_changed_since_the_run_gives_rank_1(judge, tmp_path) -> None:
    def edit(info):
        info["inputs"]["models_json"]["sha256"] = "f" * 64

    out = run_judge(judge, tmp_path, edit_info=edit)
    assert "identity: models_json_sha256_matches_run" in out["rank1_reasons"]
    assert out["ranks"]["a1"]["rank"] == 1


def test_models_json_sha_is_read_from_the_file(judge, tmp_path) -> None:
    path = tmp_path / "models_k7.json"
    path.write_text('{"models": [], "role_sets": {}}', encoding="utf-8")
    listing = judge.read_json_models(path)
    assert listing["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_seed_swapped_between_pipes_gives_rank_1(judge, tmp_path) -> None:
    """Same row count, but pipe 1 lacks seed 0 and pipe 0 has it twice."""
    rows = make_rows(judge)
    moved = next(
        r for r in rows
        if r["model_id"] == "K7_P3_D080" and r["variant"] == "a1"
        and r["rho"] == "20" and r["seed_index"] == "0" and r["pipe_index"] == "1"
    )  # fmt: skip
    moved["pipe_index"] = "0"
    out = run_judge(judge, tmp_path, rows=rows)
    assert "fixed_sets: rows_K7_P3_D080_20_a1" in out["rank1_reasons"]


def test_noise_free_row_per_pipe(judge, tmp_path) -> None:
    rows = make_rows(judge)
    clean = next(
        r for r in rows
        if r["model_id"] == "K7_P2_D070" and r["variant"] == "cur"
        and r["noise_free"] == "True" and r["pipe_index"] == "1"
    )  # fmt: skip
    clean["pipe_index"] = "0"
    out = run_judge(judge, tmp_path, rows=rows)
    assert "fixed_sets: noise_free_rows_K7_P2_D070_cur" in out["rank1_reasons"]


def test_bg_line_noise_free_row_has_a_blank_pipe(judge, tmp_path) -> None:
    rows = make_rows(judge)
    clean = next(
        r for r in rows
        if r["model_id"] == "K7_BG_L" and r["variant"] == "a1b2"
        and r["noise_free"] == "True"
    )  # fmt: skip
    clean["pipe_index"] = "0"
    out = run_judge(judge, tmp_path, rows=rows)
    assert "fixed_sets: noise_free_rows_K7_BG_L_a1b2" in out["rank1_reasons"]


def test_missing_seed_row_gives_rank_1(judge, tmp_path) -> None:
    rows = make_rows(judge)
    target = next(
        i for i, r in enumerate(rows)
        if r["model_id"] == "K7_PVC_D070" and r["variant"] == "a1"
        and r["seed_index"] == "39"
    )  # fmt: skip
    del rows[target]
    out = run_judge(judge, tmp_path, rows=rows)
    assert any("rows_K7_PVC_D070" in r for r in out["rank1_reasons"])


def test_seed_off_formula_gives_rank_1(judge, tmp_path) -> None:
    rows = make_rows(judge)
    noisy = next(r for r in rows if r["noise_free"] == "False")
    noisy["seed"] = str(int(noisy["seed"]) + 1)
    out = run_judge(judge, tmp_path, rows=rows)
    assert "seeds: rows_on_formula" in out["rank1_reasons"]


def test_blank_outcome_gives_rank_1(judge, tmp_path) -> None:
    rows = make_rows(judge)
    pipe_row = next(r for r in rows if r["model_id"] == "K7_P1_D060")
    pipe_row["kept_apex"] = ""
    out = run_judge(judge, tmp_path, rows=rows)
    assert "fixed_sets: no_blank_outcome" in out["rank1_reasons"]


def test_role_set_definition_is_checked(judge) -> None:
    good = judge.role_set_checks({k: list(v) for k, v in judge.ROLE_SETS.items()})
    assert all(good.values())
    bad = {k: list(v) for k, v in judge.ROLE_SETS.items()}
    bad["control"] = bad["control"][:-1]
    assert not judge.role_set_checks(bad)["role_sets_json_equal"]


def test_empty_cell_set_is_not_a_pass(judge) -> None:
    empty = judge._cells_result([], 4)
    assert empty["pass"] is False and empty["cell_count_ok"] is False
    items = {"a1": {name: empty for name in ("K_F1", "K_F2", "K_F3")}}
    assert len(judge.cell_count_reasons(items)) == 3


# ----------------------------------------------------------------- ranks 2-6


def test_kr_not_reproduced_gives_rank_2(judge, tmp_path) -> None:
    def change(values, model, variant, rho, seed, pipe):
        if model in ("K7_P2_D070", "K7_P2_D070_L") and variant == "cur" and rho == 20:
            values["kept_apex"] = seed < 21  # gap 19/40 = 0.475 < 0.5

    out = run_judge(judge, tmp_path, rows=with_outcome(judge, change))
    assert out["K_R"]["a"] is False
    assert out["ranks"]["a1"]["rank"] == 2
    assert out["ranks"]["a1b2"]["rank"] == judge.NOT_JUDGED


def test_kr_boundary_gap_exactly_half_reproduces(judge, tmp_path) -> None:
    def change(values, model, variant, rho, seed, pipe):
        if model == "K7_P2_D070" and variant == "cur" and rho == 20:
            values["kept_apex"] = seed < 20  # gap 0.5
        if model == "K7_P2_D070_L" and variant == "cur":
            values["kept_apex"] = True  # this one not reproduced

    out = run_judge(judge, tmp_path, rows=with_outcome(judge, change))
    assert out["K_R"]["a"] is True  # at least one defect_a model
    assert out["ranks"]["a1"]["rank"] == 7


def test_kr_zero_accepted_rate_is_not_reproduced(judge, tmp_path) -> None:
    def change(values, model, variant, rho, seed, pipe):
        if model == "K7_P3_D080" and variant == "cur":
            values["accepted_apex"] = False

    out = run_judge(judge, tmp_path, rows=with_outcome(judge, change))
    assert out["K_R"]["b"] is False
    assert out["ranks"]["a1"]["rank"] == 7
    assert out["ranks"]["a1b2"]["rank"] == 2
    assert out["adopted"] == "a1"


def test_pvc_without_pairs_gives_rank_2(judge, tmp_path) -> None:
    def change(values, model, variant, rho, seed, pipe):
        if model == "K7_PVC_D070" and variant == "cur":
            values["qc_primary"] = seed < 19  # 19 paired rows in both cells

    out = run_judge(judge, tmp_path, rows=with_outcome(judge, change))
    na = {(c["model"], c["rho"]) for c in out["K_F4_not_applicable_cells"]["a1"]}
    assert {("K7_PVC_D070", 10), ("K7_PVC_D070", 20)} <= na
    assert out["ranks"]["a1"]["rank"] == 2


def test_one_pvc_cell_with_pairs_keeps_k_f4_judgeable(judge, tmp_path) -> None:
    def change(values, model, variant, rho, seed, pipe):
        if model == "K7_PVC_D070" and variant == "cur" and rho == 10:
            values["qc_primary"] = seed < 19

    out = run_judge(judge, tmp_path, rows=with_outcome(judge, change))
    assert out["items"]["a1"]["K_F4"]["judgeable"] is True
    assert out["ranks"]["a1"]["rank"] == 7


def test_control_loss_gives_rank_3(judge, tmp_path) -> None:
    def change(values, model, variant, rho, seed, pipe):
        if model == "K7_P1_D150_S003" and variant == "a1" and rho == 10:
            values["kept_apex"] = seed >= 5  # 0.875 < 1.0 - 0.1

    out = run_judge(judge, tmp_path, rows=with_outcome(judge, change))
    assert out["ranks"]["a1"]["rank"] == 3


def test_control_boundary_drop_of_exactly_0_1_passes(judge, tmp_path) -> None:
    def change(values, model, variant, rho, seed, pipe):
        if model == "K7_P1_D150_S003" and variant == "a1" and rho == 10:
            values["kept_apex"] = seed >= 4  # 0.9 = 1.0 - 0.1

    out = run_judge(judge, tmp_path, rows=with_outcome(judge, change))
    assert out["ranks"]["a1"]["rank"] == 7


def test_control_zero_current_is_listed_as_trivial(judge, tmp_path) -> None:
    def change(values, model, variant, rho, seed, pipe):
        if model == "K7_P1_D060" and rho == 10:
            values["qc_primary"] = False if variant == "cur" else values["qc_primary"]

    out = run_judge(judge, tmp_path, rows=with_outcome(judge, change))
    trivial = out["K_F2_trivial_pass_cells"]["a1"]
    assert {"model": "K7_P1_D060", "rho": 10, "metric": "qc_primary"} in trivial


def test_false_candidates_give_rank_4(judge, tmp_path) -> None:
    def change(values, model, variant, rho, seed, pipe):
        if model == "K7_P1_D060" and variant == "a1" and rho == 20:
            values["n_qc_unexplained"] = 1 if seed < 21 else 0  # mean 0.525

    out = run_judge(judge, tmp_path, rows=with_outcome(judge, change))
    assert out["ranks"]["a1"]["rank"] == 4


def test_false_candidates_on_the_bg_line_count(judge, tmp_path) -> None:
    rows = make_rows(judge)
    for row in rows:
        if (
            row["model_id"] == "K7_BG_L"
            and row["variant"] == "a1"
            and row["rho"] == "10"
        ):
            row["n_qc_unexplained"] = "1"
    out = run_judge(judge, tmp_path, rows=rows)
    assert out["ranks"]["a1"]["rank"] == 4


def test_depth_shift_gives_rank_5(judge, tmp_path) -> None:
    def change(values, model, variant, rho, seed, pipe):
        if model == "K7_PVC_D070" and variant == "a1":
            values["depth_error_m"] = 0.0701  # +6 cm: the lower reflection

    out = run_judge(judge, tmp_path, rows=with_outcome(judge, change))
    assert out["ranks"]["a1"]["rank"] == 5


def test_depth_shift_of_exactly_2_cm_passes(judge, tmp_path) -> None:
    def change(values, model, variant, rho, seed, pipe):
        if model == "K7_PVC_D070" and variant == "a1":
            values["depth_error_m"] = 0.03

    out = run_judge(judge, tmp_path, rows=with_outcome(judge, change))
    assert out["ranks"]["a1"]["rank"] == 7


def test_unfixed_defect_gives_rank_6(judge, tmp_path) -> None:
    def change(values, model, variant, rho, seed, pipe):
        if model == "K7_P2_D070_L" and variant == "a1" and rho == 10:
            values["kept_apex"] = seed < 35  # 0.875 < 1.0 - 0.1
            values["qc_primary"] = values["kept_apex"]

    out = run_judge(judge, tmp_path, rows=with_outcome(judge, change))
    assert out["ranks"]["a1"]["rank"] == 6
    assert out["adopted"] == "current"


def test_a1b2_needs_defect_b_fixed(judge, tmp_path) -> None:
    def change(values, model, variant, rho, seed, pipe):
        if model == "K7_P3_D080" and variant == "a1b2":
            values["kept_apex"] = False
            values["qc_primary"] = False

    out = run_judge(judge, tmp_path, rows=with_outcome(judge, change))
    assert out["ranks"]["a1"]["rank"] == 7
    assert out["ranks"]["a1b2"]["rank"] == 6
    assert out["adopted"] == "a1"


def test_a1b2_rank_order_after_a1(judge, tmp_path) -> None:
    def change(values, model, variant, rho, seed, pipe):
        if model == "K7_P1_D110_S002" and variant == "a1b2":
            values["kept_apex"] = False
            values["qc_primary"] = False

    out = run_judge(judge, tmp_path, rows=with_outcome(judge, change))
    assert out["ranks"]["a1b2"]["rank"] == 3
    assert out["adopted"] == "a1"


def test_reports_are_present(judge, tmp_path) -> None:
    out = run_judge(judge, tmp_path)
    assert "K7_PVC_D070|10|a1" in out["K_F5"]
    assert out["K_F6"]["a1b1_cost"]["kept_total_a1"] > 0
    assert "K7_P1_D150_S003|cur" in out["K_F6"]["c_diagnostics"]
