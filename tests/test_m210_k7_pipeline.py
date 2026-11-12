"""K7 harness (m210_k7_generator.md v1.1 4-4, 5, 6-1): seeds, fixed sets and fake
lines end to end. No gprMax output is needed; git and the K7 tests are replaced."""

import csv
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from gpr_detection.io.synthetic import SyntheticTruth
from synthetic_bscan import T_SRC_NS, BScanSpec, cylinder_times, make_radargram

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = PROJECT_ROOT / "analysis" / "processing" / "m210_k7_pipeline.py"
JUDGE_PATH = PROJECT_ROOT / "analysis" / "processing" / "m210_k7_judge.py"
MODELS_JSON = PROJECT_ROOT / "analysis" / "synthetic" / "k7" / "models_k7.json"
F_C_HZ = 500e6
X_FIRST_M = 0.40
FAKE_DT_NS = 0.02
FAKE_PIPES = {
    "K7_P2_D070_L": (55, 105),
    "K7_P1_D060": (80,),
    "K7_BG_L": (),
}
"""Fake models of the small plan: pipe apex traces (depth 0.6 m)."""
FAKE_DEPTH_M = 0.6


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load("m210_k7_pipeline", HARNESS_PATH)


@pytest.fixture(scope="module")
def judge():
    return _load("m210_k7_judge", JUDGE_PATH)


# ----------------------------------------------------------------- seeds


def test_seed_formula(harness) -> None:
    assert harness.noise_seed(0, 0, 0) == 20301112
    assert harness.noise_seed(7, 1, 39) == 20301112 + 7000 + 100 + 39
    assert harness.UNIT_NOISE_SEED == 20301199
    seeds = harness.k7_seeds()
    assert len(seeds) == 8 * 2 * 40 + 1
    assert harness.UNIT_NOISE_SEED in seeds


def test_seeds_are_disjoint_from_every_earlier_set(harness) -> None:
    record = harness.seed_record()
    assert record["seed_overlap"] == 0
    assert set(record["asserted_against"]) == {
        "batch1", "v2", "v3", "prototype", "k7_monte_carlo",
    }  # fmt: skip
    mc = harness.monte_carlo_seeds()
    assert 20261112 in mc
    assert 20261113 + 199 + 20000 + 150 in mc  # n 199, rho 20, d 1.5
    assert 20261113 + 0 + 10000 + 60 in mc  # n 0, rho 10, d 0.6


def test_one_shared_seed_fails(harness, monkeypatch) -> None:
    shared = harness.noise_seed(3, 1, 7)
    monkeypatch.setattr(
        harness, "earlier_seed_sets", lambda: {"planted": frozenset({shared})}
    )
    with pytest.raises(harness.floor_harness.SeedOverlapError):
        harness.seed_record()


# ----------------------------------------------------------------- fixed lists


def test_fixed_lists_match_the_judge_and_the_spec(harness, judge) -> None:
    assert harness.MODEL_IDS == judge.MODEL_IDS
    assert harness.ROLE_SETS == judge.ROLE_SETS
    assert harness.K7_SEALED_FILES == judge.K7_SEALED_FILES
    assert (
        harness.CALIBRATION_PAIRS
        == judge.CALIBRATION_PAIRS
        == {"K7_BG_L": "K7_P2_D070_L"}
    )
    assert harness.VARIANTS == judge.VARIANTS == ("cur", "a1", "a1b2", "a1b1")
    assert set(harness.VARIANTS) == set(harness.generator.GENERATOR_VARIANTS)
    assert len(harness.MODEL_IDS) == 8


def test_em_json_lists_the_fixed_models_and_roles(harness) -> None:
    if not MODELS_JSON.exists():
        pytest.skip("models_k7.json not available")
    meta = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    ids = [m["id"] for m in meta["models"] if m.get("batch") == "k7"]
    assert sorted(ids) == sorted(harness.MODEL_IDS)
    roles = {k: sorted(v) for k, v in meta["role_sets"].items() if k != "source"}
    assert roles == {k: sorted(v) for k, v in harness.ROLE_SETS.items()}


def test_sealed_list_names_existing_files(harness) -> None:
    created_by_this_task = {"tests/test_m210_k7_judge.py"}
    for path in harness.K7_SEALED_FILES:
        if path not in created_by_this_task:
            assert (PROJECT_ROOT / path).exists(), path


# ----------------------------------------------------------------- fake lines


def _fake_line(model_id: str):
    spec = BScanSpec(
        f_c_hz=F_C_HZ, dt_ns=FAKE_DT_NS, dx_m=0.02, n_traces=161, n_samples=1400,
        snr_db=None, air_gap_m=0.001, offset_m=0.0, direct_amplitude=30.0,
        surface_amplitude=-20.0, pipe_amplitude=0.0, eps_r=9.0,
    )  # fmt: skip
    spec.pipe_times_ns = np.zeros(spec.n_traces)
    events, truths = [], []
    for apex in FAKE_PIPES[model_id]:
        times = cylinder_times(spec.x_m, spec.x_m[apex], FAKE_DEPTH_M, spec.velocity, 0)
        events.append((times, 1.0))
        truths.append(_truth(model_id, apex, T_SRC_NS + float(times[apex])))
    spec.extra_events = events
    radargram = make_radargram(spec, line_id=f"K7/{model_id}")
    metadata = {"trace_x_m_first": X_FIRST_M, "time_of_sample0_ns": 0.0}
    return dataclasses.replace(radargram, metadata=metadata), truths


def _truth(model_id: str, apex: int, t_apex_ns: float) -> SyntheticTruth:
    return SyntheticTruth(
        model_id=model_id, pipe_top_depth_m=FAKE_DEPTH_M,
        pipe_center_depth_m=FAKE_DEPTH_M, pipe_radius_m=0.0,
        pipe_center_x_m=X_FIRST_M + apex * 0.02, eps_r=9.0, velocity_m_per_ns=0.1,
        air_gap_m=0.001, antenna_offset_m=0.0, t0_top_ns=12.0,
        apex_trace_index=apex, apex_envelope_peak_ns=t_apex_ns,
        source_peak_ns=T_SRC_NS, direct_wave_peak_ns=None,
        surface_reflection_peak_ns=None, wavelet_lag_ns=None,
        trace_x_m_first=X_FIRST_M, ray_reflection_time_ns=(),
    )  # fmt: skip


def _caching_loader():
    cache = {}

    def loader(model_id, root):
        if model_id not in cache:
            cache[model_id] = _fake_line(model_id)
        return cache[model_id]

    return loader


def _events(model_id: str) -> list[dict]:
    _, truths = _fake_line(model_id)
    return [
        {
            "event_id": f"{model_id}/E{i:02d}", "kind": "primary", "pipe_index": i,
            "x_m": truth.pipe_center_x_m, "t_ns": truth.apex_envelope_peak_ns,
        }
        for i, truth in enumerate(truths)
    ]  # fmt: skip


def _models_json(path: Path, model_ids=tuple(FAKE_PIPES), with_events=True) -> Path:
    meta = {
        "models": [
            {"id": m, "batch": "k7", "pipes": [{}] * len(FAKE_PIPES[m])}
            for m in model_ids
        ],
        "role_sets": {},
    }
    if with_events:
        meta["expected_events"] = {m: _events(m) for m in model_ids}
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


def _clean_seal(seal, head=None):
    commit = seal.commit
    return {
        "seal_commit": commit,
        "seal_commit_full": commit,
        "seal_commit_time": "2026-11-12T10:00:00+09:00",
        "head_commit": commit if head is None else head,
        "harness_blob_working_tree": "h",
        "generator_blob_working_tree": "g",
        "sealed_files": {"docs/x.md": {"blob_at_seal": "a", "matches": True}},
        "dirty_files": [],
        "git_failures": [],
        "taken_at": "2026-11-12T11:00:00+09:00",
    }


def _placeholder_inputs(root: Path) -> None:
    root.mkdir()
    for model_id in FAKE_PIPES:
        for suffix in (".npy", ".json"):
            (root / f"{model_id}{suffix}").write_bytes(b"placeholder")


SMALL_PLAN = {
    "model_ids": tuple(FAKE_PIPES),
    "rhos": (10,),
    "n_seeds": 1,
}


def _passing_tests():
    return {"command": "fake", "exit_code": 0, "summary": "ok", "pass": True}


def _argv(tmp_path: Path, models_json: Path) -> list[str]:
    return [
        "--protocol", "k7", "--seal-commit", "k7seal",
        "--models-json", str(models_json), "--input-dir", str(tmp_path / "data"),
        "--out-dir", str(tmp_path / "out"), "--regression-activities", "none",
    ]  # fmt: skip


def _read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return reader.fieldnames, rows


@pytest.fixture(scope="module")
def small_run(harness, tmp_path_factory):
    """One small end-to-end run (3 fake models, rho 10, 1 seed, 4 variants)."""
    tmp_path = tmp_path_factory.mktemp("k7_run")
    patcher = pytest.MonkeyPatch()
    patcher.setattr(harness, "seal_record", _clean_seal)
    try:
        _placeholder_inputs(tmp_path / "data")
        models_json = _models_json(tmp_path / "models.json")
        plan = dataclasses.replace(harness.DEFAULT_PLAN, **SMALL_PLAN)
        code = harness.main(
            _argv(tmp_path, models_json), plan, _caching_loader(), _passing_tests
        )
    finally:
        patcher.undo()
    return code, tmp_path / "out"


def test_main_writes_runs_and_identity_files(harness, small_run) -> None:
    code, out = small_run
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    # G0 was skipped (--regression-activities none): a sealed run would fail.
    assert code == 1
    assert info["guards"]["failures"] == ["G0 regression not run"]
    fieldnames, rows = _read_rows(out / "runs_k7.csv")
    assert fieldnames == list(harness.RUN_COLUMNS)
    per_line = {"K7_P2_D070_L": 2, "K7_P1_D060": 1, "K7_BG_L": 1}
    assert len(rows) == sum(n * 4 * 2 for n in per_line.values())
    assert {r["variant"] for r in rows} == {"cur", "a1", "a1b2", "a1b1"}
    noisy = [r for r in rows if r["noise_free"] == "False"]
    assert {(r["model_id"], int(r["seed"])) for r in noisy} == {
        ("K7_P2_D070_L", 20301112 + 1000 * 1),
        ("K7_P1_D060", 20301112 + 1000 * 5),
        ("K7_BG_L", 20301112 + 1000 * 7),
    }
    clean = [r for r in rows if r["noise_free"] == "True"]
    assert all(r["seed"] == r["rho"] == "" for r in clean)
    pipe_rows = [r for r in rows if r["model_id"] != "K7_BG_L"]
    assert all(r["accepted_apex"] in {"True", "False"} for r in pipe_rows)
    assert all(r["kept_apex"] in {"True", "False"} for r in pipe_rows)
    bg = [r for r in rows if r["model_id"] == "K7_BG_L"]
    assert all(r["pipe_index"] == "" and r["kept_apex"] == "" for r in bg)
    assert all(r["n_qc_unexplained"] != "" for r in rows)
    for name in ("pre_run.txt", "post_run.txt"):
        lines = (out / name).read_text(encoding="utf-8").splitlines()
        assert lines[1:3] == ["HEAD k7seal", "dirty 0"]
    assert info["protocol"] == "k7" and info["seed_overlap"] == 0
    assert info["sealed_files"] == ["docs/x.md"]
    assert info["tests"]["pass"] is True
    models = info["models"]
    assert models["K7_BG_L"]["calibration"] == models["K7_P2_D070_L"]["calibration"]
    assert models["K7_P2_D070_L"]["a0_min"] == min(
        pipe["a0"] for pipe in models["K7_P2_D070_L"]["pipes"]
    )


def test_noise_free_clean_pipes_are_found(small_run) -> None:
    """The fake pipes are clean hyperbolas: cur keeps them, a1 does not lose them."""
    _, out = small_run
    _, rows = _read_rows(out / "runs_k7.csv")
    clean = [
        r for r in rows if r["noise_free"] == "True" and r["model_id"] != "K7_BG_L"
    ]
    assert clean and all(r["kept_apex"] == "True" for r in clean)
    assert all(r["qc_primary"] == "True" for r in clean if r["variant"] == "cur")


def test_start_guard_stops_before_tests_and_rows(harness, tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "seal_record", lambda s: _clean_seal(s, "later"))
    _placeholder_inputs(tmp_path / "data")
    models_json = _models_json(tmp_path / "models.json")
    plan = dataclasses.replace(harness.DEFAULT_PLAN, **SMALL_PLAN)

    def must_not_run():
        raise AssertionError("tests must not run after a failed start guard")

    code = harness.main(
        _argv(tmp_path, models_json), plan, _caching_loader(), must_not_run
    )
    assert code == 1
    assert not (tmp_path / "out" / "runs_k7.csv").exists()
    info = json.loads((tmp_path / "out" / "run_info.json").read_text(encoding="utf-8"))
    assert info["guards"]["stage"] == "start"
    assert (tmp_path / "out" / "pre_run.txt").exists()


def test_failing_k7_tests_stop_the_run(harness, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(harness, "seal_record", _clean_seal)
    _placeholder_inputs(tmp_path / "data")
    models_json = _models_json(tmp_path / "models.json")
    plan = dataclasses.replace(harness.DEFAULT_PLAN, **SMALL_PLAN)

    def failing():
        return {"command": "x", "exit_code": 1, "summary": "1 failed", "pass": False}

    code = harness.main(_argv(tmp_path, models_json), plan, _caching_loader(), failing)
    assert code == 1
    assert not (tmp_path / "out" / "runs_k7.csv").exists()
    info = json.loads((tmp_path / "out" / "run_info.json").read_text(encoding="utf-8"))
    assert any("K7 tests failed" in f for f in info["guards"]["failures"])


def test_json_model_list_must_be_the_plan(harness, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(harness, "seal_record", _clean_seal)
    _placeholder_inputs(tmp_path / "data")
    models_json = _models_json(tmp_path / "models.json", ("K7_P2_D070_L", "K7_BG_L"))
    plan = dataclasses.replace(harness.DEFAULT_PLAN, **SMALL_PLAN)
    code = harness.main(
        _argv(tmp_path, models_json), plan, _caching_loader(), _passing_tests
    )
    assert code == 1
    info = json.loads((tmp_path / "out" / "run_info.json").read_text(encoding="utf-8"))
    assert any("differ from the plan" in f for f in info["guards"]["failures"])


def test_json_without_expected_events_is_refused(harness, tmp_path) -> None:
    models_json = _models_json(tmp_path / "models.json", with_events=False)
    with pytest.raises(SystemExit, match="no expected_events"):
        harness.json_models(models_json)


def _event(event_id, kind, x_m, t_ns, **extra) -> dict:
    return {"event_id": event_id, "kind": kind, "x_m": x_m, "t_ns": t_ns, **extra}


def test_k7_event_kinds_parse(harness) -> None:
    events = harness.parse_k7_events(
        [
            _event("M/E01", "primary", 2.0, 17.5, pipe_index=0),
            _event("M/E02", "flat_layer", None, 20.5, ambiguous_with=["M/E03"]),
            _event("M/E03", "multiple_layer", 1.5, 20.6),
            _event("M/E04", "pvc_internal", 2.0, 18.9, pipe_index=0),
        ]
    )
    assert [e.kind for e in events] == [
        "primary", "flat_layer", "multiple_layer", "pvc_internal",
    ]  # fmt: skip
    assert events[1].x_m is None and events[1].ambiguous_with == ("M/E03",)
    with pytest.raises(ValueError, match="unknown event kind"):
        harness.parse_k7_events([_event("M/E09", "ghost", 1.0, 3.0)])
    with pytest.raises(ValueError, match="x_m null"):
        harness.parse_k7_events([_event("M/E09", "multiple_layer", None, 3.0)])


def test_flat_layer_attaches_on_every_trace(harness) -> None:
    events = harness.parse_k7_events(
        [
            _event("M/E01", "primary", 2.0, 17.5, pipe_index=0),
            _event("M/E02", "flat_layer", None, 20.5),
            _event("M/E03", "pvc_internal", 2.0, 18.9, pipe_index=0),
        ]
    )
    far_trace = harness.attach_k7_event(0.5, 20.9, events, F_C_HZ)
    assert far_trace is not None and far_trace.event_id == "M/E02"
    assert harness.attach_k7_event(0.5, 21.6, events, F_C_HZ) is None  # > 1 ns
    # A primary wins over a nearer-in-time event inside the box.
    both = harness.attach_k7_event(2.0, 18.4, events, F_C_HZ)
    assert both.event_id == "M/E01"
    pvc = harness.attach_k7_event(2.0, 19.0, events, F_C_HZ)
    assert pvc.kind == "pvc_internal"  # explained, but not a primary


def test_per_model_event_copy_must_match(harness, tmp_path) -> None:
    path = _models_json(tmp_path / "models.json")
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["models"][0]["expected_events"] = []
    path.write_text(json.dumps(meta), encoding="utf-8")
    listing = harness.json_models(path)
    plan = dataclasses.replace(harness.DEFAULT_PLAN, **SMALL_PLAN)
    failures = harness.model_set_failures(listing, plan)
    assert any("per-model expected_events" in f for f in failures)


def test_em_json_events_parse_with_a_primary_per_pipe(harness) -> None:
    """Dry run on the committed models_k7.json: every kind parses, every pipe has
    its primary, the two event copies agree and the fixed sets hold."""
    if not MODELS_JSON.exists():
        pytest.skip("models_k7.json not available")
    listing = harness.json_models(MODELS_JSON)
    assert harness.model_set_failures(listing, harness.DEFAULT_PLAN) == []
    for model_id, events in listing["events"].items():
        primaries = sorted(e.pipe_index for e in events if e.kind == "primary")
        assert primaries == list(range(listing["n_pipes"][model_id])), model_id
        ids = {e.event_id for e in events}
        assert all(set(e.ambiguous_with) <= ids for e in events), model_id


def test_g0_failures_need_every_part(harness) -> None:
    identical = {
        "run": True,
        "fourtu": {
            "run": True,
            "a.csv": {"byte_identical": True},
            "b.csv": {"byte_identical": True},
        },
        "s1g": {"run": True, "identical": True},
        "m210b_v3": {"run": True, "identical": True},
    }
    assert harness.g0_failures(identical) == []
    broken = json.loads(json.dumps(identical))
    broken["fourtu"]["b.csv"]["byte_identical"] = False
    broken["s1g"] = {"run": False, "reason": "missing"}
    assert len(harness.g0_failures(broken)) == 2
    assert harness.g0_failures({"run": False}) == ["G0 regression not run"]


def test_pytest_counts_read_the_summary_line(harness) -> None:
    counts = harness.pytest_counts(
        "90 passed, 2 skipped, 1 deselected, 1 error, 2 failed in 31.20s"
    )
    assert counts == {
        "passed": 90, "failed": 2, "skipped": 2, "errors": 1, "xfailed": 0,
        "xpassed": 0, "deselected": 1,
    }  # fmt: skip
    assert harness.pytest_counts("")["passed"] == 0


def test_run_k7_tests_records_counts(harness, monkeypatch) -> None:
    class Completed:
        returncode = 0
        stdout = "....\n12 passed, 1 skipped in 3.00s\n"

    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **k: Completed())
    record = harness.run_k7_tests()
    assert record["pass"] is True
    assert record["counts"]["passed"] == 12 and record["counts"]["skipped"] == 1
    assert record["summary"] == "12 passed, 1 skipped in 3.00s"


def test_lost_reasons_map_to_the_spec_names(harness) -> None:
    assert set(harness.LOST_REASONS.values()) == {
        "nms", "cap6", "not_accepted", "no_peak",
    }  # fmt: skip


@pytest.mark.data
def test_s1g_and_v3_default_paths_are_unchanged(harness, tmp_path) -> None:
    """G0 (section 4-2): S1g candidates.csv and a (b) v3 subset (1 model, 2 seeds)."""
    if not harness.S1G_STORED_CANDIDATES.exists():
        pytest.skip("stored S1g candidates.csv not available")
    record = harness.s1g_regression(tmp_path / "s1g")
    assert record["identical"] is True and record["stored_only_columns"] == [
        "cluster_mu"
    ]
    v3 = harness.v3_regression(tmp_path / "v3")
    if not v3["run"]:
        pytest.skip(v3["reason"])
    assert v3["identical"] is True and v3["n_rows"] > 0
