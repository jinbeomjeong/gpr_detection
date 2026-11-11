"""M2-10 (b) pure-function floor tests T1-T8, T10 (m210b_selector_floor.md 4-1, 8-1).

The prototype ``analysis/processing/m210b_floor_proto.py`` is re-run once per
module with its output folder pointed at a temporary directory, and the result
is compared with the sealed ``docs/processing/m210b_floor_proto.json``
(integers and booleans exactly, floats to a relative 1e-9). The T assertions
read the recomputed result. T9 (line peak cap 300) is a separate item (U2).

The ``check_*`` functions take the recomputed summary so the mutation checks of
8-1 can call them on a mutated prototype run.
"""

import importlib.util
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from synthetic_bscan import BScanSpec, cylinder_times, make_radargram

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSING_DIR = PROJECT_ROOT / "analysis" / "processing"
PROTO_PATH = PROCESSING_DIR / "m210b_floor_proto.py"
SEALED_JSON = PROJECT_ROOT / "docs" / "processing" / "m210b_floor_proto.json"
FLOAT_REL_TOL = 1e-9
PROTO_OUTPUT = Path("docs") / "processing" / "m210b_floor_proto.json"

T1_FLIPS = [
    (200, 5),
    (200, 6),
    (200, 8),
    (400, 5),
    (400, 6),
    (400, 8),
    (400, 12),
    (400, 16),
]
"""Current-rule apexes found with the old (late) mute and lost with the new one."""
CANDIDATE_RULES = ("A_skip1", "B_noise", "B_full", "B_min", "C_none")
NOISE_FLOOR_RULES = ("B_noise", "B_full", "B_min")
P2 = "P2_noise_only"
P4 = "P4_residual_A"
P5 = "P5_floor_ratio_new_over_old"
P6 = "P6_noise_free"
P7B = "P7b_nonstationary_with_strong_event_200"


# ----------------------------------------------------------------- loading


def load_module(name: str, path: Path) -> ModuleType:
    """Import an analysis script by path (it puts analysis/processing on sys.path)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_prototype(proto: ModuleType, out_root: Path) -> dict:
    """Run ``proto.main`` with its root at ``out_root``; return the written json.

    The sealed json in the project is never written: ``main`` writes to
    ``<ROOT>/docs/processing``.
    """
    (out_root / PROTO_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    original_root = proto.ROOT
    try:
        proto.ROOT = out_root
        proto.main()
    finally:
        proto.ROOT = original_root
    return json.loads((out_root / PROTO_OUTPUT).read_text(encoding="utf-8"))


def mismatches(expected: object, actual: object, where: str = "") -> list[str]:
    """Paths where ``actual`` differs from ``expected`` (floats to FLOAT_REL_TOL)."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return [f"{where}: keys {sorted(set(expected) ^ set(actual))}"]
        found = []
        for key in expected:
            found += mismatches(expected[key], actual[key], f"{where}/{key}")
        return found
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{where}: length {len(expected)} != {len(actual)}"]
        found = []
        for index, (old, new) in enumerate(zip(expected, actual, strict=True)):
            found += mismatches(old, new, f"{where}[{index}]")
        return found
    is_number = all(
        isinstance(value, int | float) and not isinstance(value, bool)
        for value in (expected, actual)
    )
    has_float = isinstance(expected, float) or isinstance(actual, float)
    if is_number and has_float:
        if math.isclose(float(expected), float(actual), rel_tol=FLOAT_REL_TOL):
            return []
        return [f"{where}: {expected!r} != {actual!r}"]
    return [] if expected == actual else [f"{where}: {expected!r} != {actual!r}"]


@pytest.fixture(scope="module")
def proto() -> Iterator[ModuleType]:
    try:
        yield load_module("m210b_floor_proto", PROTO_PATH)
    finally:
        sys.modules.pop("m210b_floor_proto", None)


@pytest.fixture(scope="module")
def summary(proto: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> dict:
    return run_prototype(proto, tmp_path_factory.mktemp("m210b_proto"))


@pytest.fixture(scope="module")
def sealed() -> dict:
    return json.loads(SEALED_JSON.read_text(encoding="utf-8"))


# ----------------------------------------------------------------- checks


def check_t1(summary: dict) -> None:
    flips = [
        (row["A_s"], row["A_d"])
        for row in summary["P1_coupling"]
        if row["current_old_hit"] and not row["current_new_hit"]
    ]
    assert flips == T1_FLIPS
    reverse = [
        row
        for row in summary["P1_coupling"]
        if row["current_new_hit"] and not row["current_old_hit"]
    ]
    assert reverse == []


def check_t2(summary: dict) -> None:
    for rule in CANDIDATE_RULES:
        assert summary["P1_flip_any"][rule] == 0, rule


def check_t3(summary: dict) -> None:
    noise_only = summary[P2]
    for rule in ("B_full", "B_noise"):
        assert noise_only[rule]["peaks_cap300"]["mean"] == 0, rule
        assert noise_only[rule]["accepted_cap300"]["mean"] == 0, rule
    assert noise_only["B_min"]["accepted_cap300"]["mean"] == 0
    assert noise_only["current"]["accepted_cap300"]["mean"] == pytest.approx(4.5)
    current = noise_only["current"]
    for rule in CANDIDATE_RULES:
        for stage in ("peaks_cap300", "accepted_cap300"):
            assert noise_only[rule][stage]["mean"] <= current[stage]["mean"], rule


def check_t4(summary: dict) -> None:
    rows = {row["A_s"]: row for row in summary[P4]}
    strong = rows[400]
    for rule in ("current", "A_skip1"):
        assert (strong[f"{rule}_new_hit"], strong[f"{rule}_old_hit"]) == (
            False,
            True,
        ), rule
    for rule in (*NOISE_FLOOR_RULES, "C_none"):
        assert strong[f"{rule}_new_hit"] and strong[f"{rule}_old_hit"], rule
    for rule in ("current", *CANDIDATE_RULES):  # 4-1: A_s 200, every rule succeeds
        assert rows[200][f"{rule}_new_hit"] and rows[200][f"{rule}_old_hit"], rule


def check_t5(summary: dict) -> None:
    expected_hits = {2.5: 0, 3: 0, 4: 3, 5: 2, 6: 5, 8: 4, 10: 8, 12: 8}
    for row in summary["P3_sensitivity"]:
        hits = {row[f"{rule}_hit"] for rule in ("current", *CANDIDATE_RULES)}
        assert hits == {expected_hits[row["A_d"]]}, row["A_d"]


def check_t6(summary: dict) -> None:
    ratios = summary[P5]
    assert all(ratio == 1.0 for ratio in ratios["B_full"])
    assert all(ratio == 1.0 for ratio in ratios["B_min"])
    assert all(ratio > 20 for ratio in ratios["current"])


def check_t7(summary: dict, proto: ModuleType) -> None:
    noise_free = summary[P6]
    assert noise_free["B_full"]["apex_hit"] and noise_free["B_min"]["apex_hit"]
    assert not noise_free["current"]["apex_hit"]
    envelope = proto.envelope(proto.section(None, 200.0, 8.0))
    k_lo, k_hi = proto.fr.search_rows(proto.T_NEW, proto.F_C, proto.DT, proto.NS)
    guard = proto.fr.EPS32 * float(envelope[k_lo:k_hi].max())
    for rule in NOISE_FLOOR_RULES:
        assert noise_free[rule]["F"] == pytest.approx(guard, rel=FLOAT_REL_TOL), rule


def check_t10(summary: dict) -> None:
    ramp_with_clutter = summary[P7B]
    b_min_peaks = ramp_with_clutter["B_min"]["peaks_cap300"]["mean"]
    b_full_peaks = ramp_with_clutter["B_full"]["peaks_cap300"]["mean"]
    assert b_min_peaks > 10 * b_full_peaks
    assert ramp_with_clutter["B_full"]["accepted_cap300"]["mean"] == 0


# ----------------------------------------------------------------- tests


def test_prototype_reproduces_the_sealed_json(summary: dict, sealed: dict) -> None:
    assert mismatches(sealed, summary) == []


def test_t1_current_rule_flips_exactly_eight_conditions(summary: dict) -> None:
    check_t1(summary)


def test_t2_candidate_rules_do_not_flip(summary: dict) -> None:
    check_t2(summary)


def test_t3_noise_only_false_candidates(summary: dict) -> None:
    check_t3(summary)


def test_t4_residual_coupling_of_a_skip1(summary: dict) -> None:
    check_t4(summary)


def test_t5_candidate_stage_sensitivity_is_rule_independent(summary: dict) -> None:
    check_t5(summary)


def test_t6_noise_floors_do_not_move_with_the_mute(summary: dict) -> None:
    check_t6(summary)


def test_t7_noise_free_guard(summary: dict, proto: ModuleType) -> None:
    check_t7(summary, proto)


def test_t10_non_stationary_noise_floods_b_min(summary: dict) -> None:
    check_t10(summary)


# ----------------------------------------------------------------- T8 and the wrapper


def _small_line():
    """Air-coupled straight-ray pipe at 0.9 m, 500 MHz, 0.098 ns, 161 traces."""
    spec = BScanSpec(
        f_c_hz=500e6,
        dt_ns=0.098,
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
    spec.pipe_times_ns = cylinder_times(spec.x_m, spec.x0_m, 0.9, spec.velocity, 0.0)
    return make_radargram(spec, line_id="S4S6/T8")


@pytest.fixture(scope="module")
def line_inputs(proto: ModuleType):
    from gpr_detection.depth.estimate import prepare_depth_inputs

    radargram = _small_line()
    return radargram, prepare_depth_inputs(radargram)


def _wrapper_and_rule_peaks(proto, radargram, inputs, rule, mute_offset_ns):
    generator, floor_rules = proto.gen, proto.fr
    line, kept, envelope = generator.apex_candidates_for_radargram(
        radargram,
        inputs,
        act="S4S6",
        path="T8",
        floor_rule=rule,
        mute_offset_ns=mute_offset_ns,
    )
    t_zero = inputs.zero.t_zero_ns
    k_lo, k_hi = floor_rules.search_rows(
        t_zero + mute_offset_ns, radargram.f_c_hz, radargram.dt_ns, envelope.shape[0]
    )
    edge = math.ceil(generator.EDGE_M / radargram.dx_m)
    rule_peaks, diag, _, _ = floor_rules.detect(
        envelope,
        inputs.branch.background_removed,
        k_lo,
        k_hi,
        radargram.f_c_hz,
        radargram.dt_ns,
        edge,
        rule,
        floor_rules.NUM_PEAKS,
    )
    return line, kept, rule_peaks, diag


@pytest.mark.parametrize(
    ("rule", "mute_offset_ns"),
    [("current", 0.0), ("B_full", 0.0), ("B_min", 1.0), ("current", -1.0)],
)
def test_t8_wrapper_peaks_equal_the_rule_detection(
    proto: ModuleType, line_inputs, rule: str, mute_offset_ns: float
) -> None:
    radargram, inputs = line_inputs
    line, _, rule_peaks, diag = _wrapper_and_rule_peaks(
        proto, radargram, inputs, rule, mute_offset_ns
    )
    wrapper_peaks = [(peak["row"], peak["trace0"]) for peak in line["_peaks"]]
    assert wrapper_peaks, "the test line must give detection peaks"
    assert wrapper_peaks == [(k, j) for k, j, _ in rule_peaks]
    assert line["n_peaks"] == len(rule_peaks)
    assert line["_floor"]["F"] == diag["F"]
    assert line["_floor"]["L"] == diag["L"]
    assert line["_floor"]["n_raw_peaks"] == diag["n_raw_peaks"]


def test_wrapper_defaults_are_selector_v1(proto: ModuleType, line_inputs) -> None:
    radargram, inputs = line_inputs
    generator = proto.gen
    line, kept, _ = generator.apex_candidates_for_radargram(
        radargram, inputs, act="S4S6", path="T8"
    )
    explicit, kept_explicit, _ = generator.apex_candidates_for_radargram(
        radargram,
        inputs,
        act="S4S6",
        path="T8",
        floor_rule="current",
        mute_offset_ns=0.0,
    )
    assert kept and kept == kept_explicit
    assert line["_accepted_all"] == explicit["_accepted_all"]
    level = line["_floor"]["level"]
    assert line["_floor"]["F"] == 0.05 * float(level.max())


def test_mute_offset_moves_only_the_mute(proto: ModuleType, line_inputs) -> None:
    radargram, inputs = line_inputs
    t_zero = inputs.zero.t_zero_ns
    line, kept, _ = proto.gen.apex_candidates_for_radargram(
        radargram, inputs, act="S4S6", path="T8", mute_offset_ns=1.0
    )
    expected_k_lo = math.ceil((t_zero + 1.0 + 1e9 / radargram.f_c_hz) / 0.098) + 2
    assert line["_floor"]["k_lo"] == expected_k_lo
    for row in kept:
        assert row["tau_ns"] == pytest.approx(row["t_ns"] - t_zero, abs=1e-3)


def test_accepted_fates_cover_kept_nms_and_cap(proto: ModuleType) -> None:
    dx = 0.02
    accepted = [{"trace": 20 * i, "t_ns": 20.0} for i in range(8)]  # 0.4 m apart
    accepted.insert(1, {"trace": 1, "t_ns": 21.0})  # within 0.3 m and 2 ns of #0
    kept, fates = proto.gen._rank_and_suppress(accepted, dx)
    assert [fate["fate"] for fate in fates] == [
        "kept", "nms", "kept", "kept", "kept", "kept", "kept", "cap6", "cap6",
    ]  # fmt: skip
    assert fates[1]["winner"] == 0
    assert len(kept) == proto.gen.MAX_APEX_CANDIDATES_PER_LINE
    assert kept == [accepted[i] for i in (0, 2, 3, 4, 5, 6)]


def test_wrapper_line_keeps_the_csv_fields(proto: ModuleType, line_inputs) -> None:
    radargram, inputs = line_inputs
    generator = proto.gen
    line, kept, _ = generator.apex_candidates_for_radargram(
        radargram, inputs, act="S4S6", path="T8"
    )
    public = [key for key in line if not key.startswith("_")]
    assert public == generator.LINE_FIELDS
    assert all(list(row) == generator.APEX_CANDIDATE_FIELDS for row in kept)
    assert len(line["_accepted_fate"]) == len(line["_accepted_all"])
