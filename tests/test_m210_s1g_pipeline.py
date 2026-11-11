"""S1g harness (spec_eval_4tu_v1 v1.9 section 9-9-1 (c)): event attachment and
one fake line end to end. No gprMax output is needed."""

import csv
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from synthetic_bscan import T_SRC_NS, BScanSpec, cylinder_times, make_radargram

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = PROJECT_ROOT / "analysis" / "processing" / "m210_s1g_pipeline.py"
F_C_HZ = 500e6
DT_NS = 0.098
X_FIRST_M = 0.40
FAKE_ID = "S1G_FAKE"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("m210_s1g_pipeline", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["m210_s1g_pipeline"] = module
    spec.loader.exec_module(module)
    return module


def _event(harness, event_id, kind, x_m, t_ns, pipe_index=None):
    return harness.ExpectedEvent(event_id, kind, x_m, t_ns, pipe_index)


# ----------------------------------------------------------------- attach_event


def test_attach_boundaries_are_inside(harness) -> None:
    events = [_event(harness, "E1", "multiple_surface", 2.0, 20.0)]
    attach = harness.attach_event
    assert attach(2.10, 21.0, events, F_C_HZ) == events[0]  # |dx| 0.10, |dt| 1 ns
    assert attach(1.90, 19.0, events, F_C_HZ) == events[0]
    assert attach(2.10 + 2e-9, 20.0, events, F_C_HZ) is None
    assert attach(2.0, 21.0 + 2e-9, events, F_C_HZ) is None
    assert attach(2.0, 20.0, [], F_C_HZ) is None  # unexplained


def test_attach_prefers_primary_then_time_then_dx_then_json_order(harness) -> None:
    primary = _event(harness, "P", "primary", 2.08, 20.9, pipe_index=1)
    near_time = _event(harness, "M", "multiple_pipe", 2.0, 20.0)
    assert harness.attach_event(2.0, 20.0, [near_time, primary], F_C_HZ) == primary

    far = _event(harness, "A", "multiple_other", 2.0, 20.5)
    close = _event(harness, "B", "multiple_other", 2.09, 20.1)
    assert harness.attach_event(2.0, 20.0, [far, close], F_C_HZ) == close

    wide = _event(harness, "C", "multiple_surface", 2.08, 20.2)
    narrow = _event(harness, "D", "multiple_surface", 2.02, 20.2)
    assert harness.attach_event(2.0, 20.0, [wide, narrow], F_C_HZ) == narrow

    # Exact binary offsets, so |dx| and |dt| tie exactly.
    first = _event(harness, "E", "multiple_surface", 2.0625, 20.25)
    second = _event(harness, "F", "multiple_surface", 1.9375, 20.25)
    assert harness.attach_event(2.0, 20.0, [first, second], F_C_HZ) == first
    assert harness.attach_event(2.0, 20.0, [second, first], F_C_HZ) == second


def test_parse_events_keeps_order_and_rejects_unknown_kinds(harness) -> None:
    entries = [
        {
            "event_id": "X/E01",
            "kind": "primary",
            "pipe_index": 0,
            "x_m": 2,
            "t_ns": 15.5,
        },
        {"event_id": "X/E02", "kind": "multiple_other", "x_m": 2.0, "t_ns": 29.5},
    ]
    events = harness.parse_events(entries)
    assert [(e.event_id, e.pipe_index) for e in events] == [
        ("X/E01", 0),
        ("X/E02", None),
    ]
    with pytest.raises(ValueError, match="kind"):
        harness.parse_events([{**entries[1], "kind": "ghost"}])


# ----------------------------------------------------------------- one fake line


def _fake_line(harness):
    """Air-coupled straight-ray pipe at 0.5 m, 500 MHz, export-like axes."""
    spec = BScanSpec(
        f_c_hz=F_C_HZ,
        dt_ns=DT_NS,
        dx_m=0.02,
        n_traces=161,
        n_samples=512,
        snr_db=20.0,
        air_gap_m=0.001,
        offset_m=0.0,
        direct_amplitude=30.0,
        surface_amplitude=-20.0,
        eps_r=9.0,
    )
    spec.pipe_times_ns = cylinder_times(spec.x_m, spec.x0_m, 0.5, spec.velocity, 0.0)
    radargram = make_radargram(spec, line_id=f"S1G/{FAKE_ID}")
    radargram = dataclasses.replace(
        radargram,
        metadata={"trace_x_m_first": X_FIRST_M, "time_of_sample0_ns": 0.0},
    )
    apex_trace = spec.n_traces // 2
    apex_t_ns = T_SRC_NS + float(spec.pipe_times_ns[apex_trace])
    primary = _event(
        harness,
        f"{FAKE_ID}/E01",
        "primary",
        X_FIRST_M + apex_trace * 0.02,
        apex_t_ns,
        0,
    )
    return radargram, [primary]


def test_run_model_end_to_end_on_a_fake_line(harness) -> None:
    radargram, events = _fake_line(harness)
    rows, info = harness.run_model(FAKE_ID, radargram, events)
    assert rows and all(list(row) == list(harness.CANDIDATE_COLUMNS) for row in rows)
    primaries = [row for row in rows if row["event_kind"] == "primary"]
    assert len(primaries) >= 1
    row = primaries[0]
    assert (row["status"], row["qc_excluded"], row["event_pipe_index"]) == (
        "ok",
        False,
        0,
    )
    assert row["depth_top_m"] == pytest.approx(0.5, abs=0.02)
    assert row["cluster_mu"] == 1
    # Not QC-passing candidates are not attached and carry no cluster.
    for other in rows:
        if other["status"] != "ok" or other["qc_excluded"] is not False:
            assert other["event_kind"] == "" and other["cluster_id"] == ""
    assert info["k1"]["n_c"] == [1] * len(info["k1"]["n_c"])
    assert info["line_id"] == f"S1G/{FAKE_ID}"


def test_run_model_rejects_a_line_id_without_the_s1g_activity(harness) -> None:
    radargram, events = _fake_line(harness)
    with pytest.raises(ValueError, match="line_id"):
        harness.run_model(FAKE_ID, dataclasses.replace(radargram, line_id="X"), events)


def test_main_writes_candidates_and_run_info(harness, tmp_path, monkeypatch) -> None:
    radargram, events = _fake_line(harness)
    data_root = tmp_path / "data"
    data_root.mkdir()
    for suffix in (".npy", ".json"):
        (data_root / f"{FAKE_ID}{suffix}").write_bytes(b"placeholder")
    models_json = tmp_path / "models.json"
    models_json.write_text(
        json.dumps({"expected_events": {FAKE_ID: [dataclasses.asdict(events[0])]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(harness, "load_line", lambda model_id, root: radargram)
    out = tmp_path / "out"
    argv = [
        "--data-root", str(data_root),
        "--models-json", str(models_json),
        "--out-dir", str(out),
        "--models", FAKE_ID,
        "--regression-activities", "none",
    ]  # fmt: skip
    assert harness.main(argv) == 0
    with (out / "candidates.csv").open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == list(harness.CANDIDATE_COLUMNS)
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    assert info["wrapper_regression"] == {"run": False}
    assert info["qc_constants"]["code_constants_differing"] == []
    assert info["depth_config"]["equals_default"] is True
    assert set(info["inputs"]) == {"models_json", f"{FAKE_ID}.npy", f"{FAKE_ID}.json"}
    assert info["lines"][FAKE_ID]["zero_rule"]
    assert info["seal"]["seal_commit"] == "b3e68fa"
