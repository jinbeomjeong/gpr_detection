"""M3-3 R3 change definitions (``m33_change_defs``) and report tool.

Spec: ``docs/processing/m33_r3_zero_change_definition.md`` §5-4 (tests 1-14).
Tests 1-11 build ``ZeroResult`` values by hand. Tests 12-13 run the band
runner offline on the synthetic line of ``tests/test_inject.py`` (stand-in
detectors, as in ``tests/test_m33_r3_band.py``) to make a small R3 folder,
then the report tool on it. Test 14 reads the real R3 plan, the stored
injected parts and the 4TU line (opt-in: ``pytest -m realdata``).
"""

import functools
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "detection"))
import m32_pipe_responses as responses_io
import m32_recipe_eval as runner
import m32_recipe_inputs as inputs
import m33_change_defs as defs
import m33_inputs
import m33_r3_band as band
import m33_r3_change_report as report
import m33_r3_p4 as p4

from gpr_detection.eval import inject
from gpr_detection.pipeline import PP_SETS, PPSetId
from gpr_detection.preprocess.filters import prepare_fitting_branch
from gpr_detection.preprocess.timezero import ZeroResult, find_time_zero

REPO = Path(__file__).resolve().parents[1]
P = 25.0
FINAL_KEYS = {"status", "out_dir", "n_lines", "n_realizations", "elapsed_s"}
RUN_DEPENDENT_KEYS = ("identity", "timings")
BYTE_IDENTICAL_FILES = (
    "realizations.csv",
    "lines.csv",
    "r3_tree_sha256_start.csv",
    "r3_tree_sha256_end.csv",
)
REAL_LINE = "02.5/Path1"
REAL_GROUP_PREFIX = f"{REAL_LINE}:g00:"


def _zero(
    k_zero: float | None,
    rule: str = "air_single_event",
    label: str | None = "merged",
    n_events: int = 1,
) -> ZeroResult:
    return ZeroResult(
        t_zero_ns=None,
        k_zero=k_zero,
        zero_rule=rule,
        zero_event_label=label,  # type: ignore[arg-type]
        pick_mode="envelope",
        events_ns=tuple(float(i) for i in range(n_events)),
        h_est_m=None,
        t_zero_per_trace_ns=None,
        meta_t0={},
        excluded_traces=0,
        flags=(),
    )


def _failed() -> ZeroResult:
    return _zero(None, rule="zero_failed", label=None, n_events=0)


# ------------------------------------------------------------ zero (1-8)


def test_1_same_zero_is_no_change() -> None:
    change = defs.zero_change(_zero(50.0), _zero(50.0), P)
    assert (change.changed, change.exact, change.dk_samples) == (False, False, 0.0)


def test_2_float_leak_is_exact_but_not_a_change() -> None:
    change = defs.zero_change(_zero(50.0), _zero(50.0 + 1e-6), P)
    assert change.exact is True and change.changed is False


def test_3_exactly_the_tolerance_is_no_change() -> None:
    change = defs.zero_change(_zero(50.0), _zero(50.0 + 0.1 * P), P)
    assert change.dk_samples == 0.1 * P
    assert change.exceeds_tolerance is False and change.changed is False


def test_4_just_above_the_tolerance_is_a_change() -> None:
    change = defs.zero_change(_zero(50.0), _zero(50.0 + 0.1 * P + 1e-9), P)
    assert change.exceeds_tolerance is True and change.changed is True


def test_5_rule_name_change_is_reported_only() -> None:
    background = _zero(50.0, rule="air_single_event", n_events=1)
    injected = _zero(50.0 + 0.003 * P, rule="air_first_event", n_events=2)
    change = defs.zero_change(background, injected, P)
    assert change.rule_changed is True and change.label_changed is False
    assert change.changed is False


def test_5b_label_change_is_a_change() -> None:
    injected = _zero(50.0, label="surface_reflection")
    change = defs.zero_change(_zero(50.0), injected, P)
    assert change.label_changed is True and change.changed is True


def test_6_event_count_change_is_reported_only() -> None:
    background = _zero(50.0, rule="air_first_event", n_events=3)
    injected = _zero(50.0 + 0.01 * P, rule="air_first_event", n_events=2)
    change = defs.zero_change(background, injected, P)
    assert change.events_changed is True and change.changed is False


def test_7_one_failed_zero_is_a_change() -> None:
    change = defs.zero_change(_zero(50.0), _failed(), P)
    assert change.failed_changed is True and change.changed is True
    assert change.dk_samples is None and change.dk_over_p is None


def test_8_both_failed_is_no_change() -> None:
    change = defs.zero_change(_failed(), _failed(), P)
    assert change.changed is False and change.failed_changed is False


def test_zero_change_needs_a_positive_period() -> None:
    for period in (0.0, -1.0, None):
        with pytest.raises(ValueError, match="period_samples"):
            defs.zero_change(_zero(50.0), _zero(50.0), period)  # type: ignore[arg-type]


# --------------------------------------------------------- period (9-11)


def test_9_period_tolerance_is_strict() -> None:
    at_boundary = defs.period_change(20.0, 20.0 * 1.05)
    above = defs.period_change(20.0, 20.0 * 1.0500001)
    assert at_boundary.changed is False and at_boundary.exact is True
    assert above.changed is True and above.exceeds_tolerance is True


def test_10_missing_injected_period_is_a_change() -> None:
    change = defs.period_change(20.0, None)
    assert change.failed is True and change.changed is True
    assert change.dp_samples is None and change.dp_rel is None


def test_11_period_tolerance_is_the_inject_same_period_rule() -> None:
    assert defs.PERIOD_CHANGE_TOLERANCE_REL == inject.PERIOD_CHECK_TOLERANCE


def test_c1_code_blobs_copy_the_seal_record() -> None:
    """``C1_CODE_BLOBS`` must equal the rows of ``m33_c1_seal.md`` §1."""
    seal = (REPO / "docs" / "experiments" / "m33_c1_seal.md").read_text(
        encoding="utf-8"
    )
    rows = dict(re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{40})` \|", seal, re.M))
    assert {path: rows.get(path) for path in report.C1_CODE_BLOBS} == (
        report.C1_CODE_BLOBS
    )


# ------------------------------------------- synthetic R3 folder (12, 13)


@functools.cache
def _inject_helpers() -> ModuleType:
    """``tests/test_inject.py`` (synthetic line and response builders)."""
    return _load_test_module("test_inject.py", "change_defs_inject_helpers")


@functools.cache
def _band_tests() -> ModuleType:
    """``tests/test_m33_r3_band.py`` (stand-in detectors)."""
    return _load_test_module("test_m33_r3_band.py", "change_defs_band_tests")


def _load_test_module(file_name: str, module_name: str) -> ModuleType:
    path = Path(__file__).resolve().parent / file_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_radargram():
    helpers = _inject_helpers()
    return helpers._radargram(helpers._background_data())


def load_synthetic_line(ref: inputs.LineRef) -> runner.LineData:
    """``runner.load_line`` on the synthetic line (module level: picklable)."""
    radargram = _synthetic_radargram()
    branch = prepare_fitting_branch(radargram)
    return runner.LineData(ref, radargram, find_time_zero(radargram, branch.filtered))


def _synthetic_ref() -> inputs.LineRef:
    return inputs.LineRef(inputs.DATASET_IDS, _inject_helpers().LINE_ID, "background")


def _patch_offline(patch: pytest.MonkeyPatch, tmp: Path) -> None:
    """The ``offline_band`` stand-ins of ``tests/test_m33_r3_band.py``."""
    helpers = _inject_helpers()
    ref = _synthetic_ref()
    library = responses_io.ResponseLibrary(
        air={},
        ground={"M12": helpers._response()},
        groups={"M12": responses_io.ModelGroup.A0},
        sha256={},
    )
    c0b = tmp / "d0_gate.json"
    if not c0b.exists():
        selection = {
            "pp_set_id": "S3",
            "pp_config_sha256": PP_SETS[PPSetId.S3].sha256(),
        }
        c0b.write_text(json.dumps({"selection": selection}), encoding="utf-8")
    patch.setattr(m33_inputs, "C0B_D0_GATE_JSON", c0b)
    patch.setattr(m33_inputs, "C0B_D0_GATE_SHA256", inputs.sha256_file(c0b))
    patch.setattr(m33_inputs, "assert_r0_identity", lambda: {"stub": "r0"})
    patch.setattr(inputs, "assert_list_hashes", lambda: {})
    patch.setattr(inputs, "background_lines", lambda: [ref])
    patch.setattr(inputs, "candidate_apexes", lambda: inputs.CandidateApexes({}, {}))
    patch.setattr(band, "ids_freq_strata", lambda: {ref.line_id: "main_400"})
    patch.setattr(
        p4,
        "IDS_FREQ_STRATUM_COUNTS",
        {"main_400": 1, "report_other": 0, "excluded_2ghz": 0},
    )
    patch.setattr(responses_io, "load_library", lambda *_: library)
    patch.setattr(runner, "assert_weights", lambda *_: {"0": "w"})
    patch.setattr(runner, "load_radargram", lambda _ref: _synthetic_radargram())
    detectors = {
        "uninjected": _band_tests().EmptyDetector(),
        "injected": _band_tests().BandDetector(),
    }
    patch.setattr(
        runner, "load_detectors", lambda settings: {0: detectors[settings.stage.value]}
    )


def _runner_argv(r3_root: Path, stage: str) -> list[str]:
    argv = ["--injectant", "band", "--stage", stage, "--folds", "0"]
    argv += ["--device", "cpu", "--allow-dirty", "--backgrounds", "gz_ids"]
    argv += ["--out-root", str(r3_root)]
    if stage == "injected":
        argv += ["--c1b-record", str(r3_root / "plan" / "c1b_record_draft.json")]
    return argv


@pytest.fixture(scope="module")
def synthetic_r3(tmp_path_factory) -> Path:
    """A small R3 folder (three runner stages) and its two hash lists."""
    base = tmp_path_factory.mktemp("synthetic_r3")
    r3_root = base / "r3"
    with pytest.MonkeyPatch.context() as patch:
        _patch_offline(patch, base)
        for stage in ("uninjected", "plan", "injected"):
            assert runner.main(_runner_argv(r3_root, stage)) == 1  # not_judgeable
    report.write_result_hash_list(r3_root, base / "result_hashes.csv")
    report.write_hash_rows(
        base / "upstream_hashes.csv", report.hash_rows(r3_root / "uninjected_ids")
    )
    return base


@pytest.fixture
def offline(monkeypatch, synthetic_r3) -> Path:
    _patch_offline(monkeypatch, synthetic_r3)
    return synthetic_r3


def _report_argv(base: Path, out_dir: Path, workers: int) -> list[str]:
    r3_root = base / "r3"
    return [
        "--r3-root",
        str(r3_root),
        "--c1b-record",
        str(r3_root / "plan" / "c1b_record_draft.json"),
        "--result-hash-list",
        str(base / "result_hashes.csv"),
        "--upstream-hash-list",
        str(base / "upstream_hashes.csv"),
        "--backgrounds",
        "gz_ids",
        "--workers",
        str(workers),
        "--out-dir",
        str(out_dir),
    ]


def _run_report(base: Path, out_dir: Path, workers: int, capsys) -> tuple[int, dict]:
    code = report.main(_report_argv(base, out_dir, workers), loader=load_synthetic_line)
    final = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    return code, final


def _summary(out_dir: Path) -> dict:
    return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))


def test_12_report_reproduces_the_synthetic_run(offline, tmp_path, capsys) -> None:
    out_dir = tmp_path / "report"
    code, final = _run_report(offline, out_dir, 1, capsys)

    assert set(final) == FINAL_KEYS
    assert (code, final["status"]) == (1, "partial"), final  # one background only
    summary = _summary(out_dir)
    assert summary["null_test"]["passed"] is True
    assert summary["null_test"]["n_lines"] == final["n_lines"] == 1
    reproduction = summary["reproduction"]
    assert reproduction["reproduced"] is True
    assert reproduction["n_measured"] == reproduction["n_expected"] > 0
    lines = inputs.read_csv(out_dir / "lines.csv")
    assert [row["null_dk_samples"] for row in lines] == ["0.0"]
    assert [row["null_dp_samples"] for row in lines] == ["0.0"]
    rows = inputs.read_csv(out_dir / "realizations.csv")
    assert len(rows) == final["n_realizations"] == reproduction["n_measured"]
    for row in rows:
        assert row["zero_changed_exact"] == row["stored_zero_changed"]
        assert row["period_changed_exact"] == row["stored_period_changed"]
        assert row["infer_i0"] == row["stored_infer_i0"]
    start = (out_dir / "r3_tree_sha256_start.csv").read_bytes()
    assert start == (out_dir / "r3_tree_sha256_end.csv").read_bytes()
    assert summary["end_checks"]["tree_changes"] == {}
    counts = summary["counts"]["all"]
    assert counts["n_realizations"] == len(rows)
    assert counts["n_zero_changed"] <= counts["n_zero_changed_exact"]
    assert summary["counts"]["p4_scope_gz_ids_main_400"] == counts
    assert summary["zero_change_tolerance_p"] == defs.ZERO_CHANGE_TOLERANCE_P


def test_13_output_does_not_depend_on_workers(offline, tmp_path, capsys) -> None:
    one, two = tmp_path / "workers1", tmp_path / "workers2"
    assert _run_report(offline, one, 1, capsys)[0] == 1
    assert _run_report(offline, two, 2, capsys)[0] == 1

    for name in BYTE_IDENTICAL_FILES:
        assert (one / name).read_bytes() == (two / name).read_bytes(), name
    first, second = _summary(one), _summary(two)
    for key in RUN_DEPENDENT_KEYS:
        first.pop(key), second.pop(key)
    assert first == second


def test_an_existing_output_folder_is_refused(offline, tmp_path, capsys) -> None:
    out_dir = tmp_path / "exists"
    out_dir.mkdir()
    code, final = _run_report(offline, out_dir, 1, capsys)
    assert (code, final["status"]) == (2, "input_error")
    assert list(out_dir.iterdir()) == []


def test_a_changed_r3_file_is_refused(offline, tmp_path, capsys) -> None:
    hashes = offline / "result_hashes.csv"
    original = hashes.read_text(encoding="utf-8")
    first_sha = inputs.read_csv(hashes)[0]["sha256"]
    try:
        hashes.write_text(original.replace(first_sha, "0" * 64), encoding="utf-8")
        code, final = _run_report(offline, tmp_path / "report", 1, capsys)
    finally:
        hashes.write_text(original, encoding="utf-8")
    assert (code, final["status"]) == (2, "input_error")
    assert not (tmp_path / "report").exists()


def _checks(reproduced=True, passed=True, tree=None, blobs=(), unmeasured=()):
    return {
        "reproduction": {"reproduced": reproduced},
        "null_test": {"passed": passed},
        "end_checks": {
            "tree_changes": tree or {},
            "code_blob_changes": list(blobs),
            "code_blob_unmeasured": list(unmeasured),
        },
    }


@pytest.mark.parametrize(
    ("checks", "n_rows", "is_full", "status"),
    [
        (_checks(unmeasured=["x.py"], reproduced=False), 5, True, "error"),
        (_checks(tree={"changed": ["a"]}, reproduced=False), 5, True, "tree_changed"),
        (_checks(blobs=["x.py"]), 5, True, "tree_changed"),
        (_checks(passed=False), 5, False, "not_reproduced"),
        (_checks(reproduced=False), 0, True, "not_reproduced"),
        (_checks(), 0, True, "empty"),
        (_checks(), 5, False, "partial"),
        (_checks(), 5, True, "ok"),
    ],
)
def test_status_follows_the_docstring_table(checks, n_rows, is_full, status) -> None:
    settings = report.ReportSettings(limit_lines=None if is_full else 1)
    assert report.decide_status(settings, n_rows, **checks) == status
    assert report.EXIT_CODES[status] == {"ok": 0, "error": 3}.get(status, 1)


def test_hash_list_command_never_overwrites(tmp_path, capsys) -> None:
    r3_root = tmp_path / "r3"
    for stage in report.RESULT_STAGES:
        (r3_root / stage).mkdir(parents=True)
        (r3_root / stage / "a.csv").write_text("x\n", encoding="utf-8")
    target = tmp_path / "hashes.csv"
    argv = ["--r3-root", str(r3_root), "--write-result-hash-list", str(target)]

    assert report.main(argv) == 0
    first = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (first["status"], first["n_files"]) == ("ok", 2)
    written = target.read_bytes()
    assert report.main(argv) == 2
    assert json.loads(capsys.readouterr().out.strip())["status"] == "input_error"
    assert target.read_bytes() == written


# --------------------------------------------------------- real line (14)


@pytest.mark.data
@pytest.mark.realdata
def test_14_real_4tu_zero_leak_is_linear_in_band_amplitude() -> None:
    """4TU ``02.5/Path1`` g00: 8 realizations, the stored flags, a linear leak."""
    report_inputs = report.read_report_inputs(report.ReportSettings())
    realizations = {
        realization_id: specs
        for realization_id, specs in report_inputs.realizations_by_line[
            REAL_LINE
        ].items()
        if realization_id.startswith(REAL_GROUP_PREFIX)
    }
    ref = next(ref for ref in report_inputs.lines if ref.line_id == REAL_LINE)
    job = report.LineJob(
        ref=ref,
        realizations=realizations,
        exclusions=report_inputs.exclusions[REAL_LINE],
        freq="",
        loader=runner.load_line,
    )
    rows = report.evaluate_line(job, report_inputs.library).realization_rows
    report.attach_stored(rows, report_inputs.stored)

    assert len(rows) == 8
    for row in rows:
        assert row["zero_changed_exact"] is True
        assert row["stored_zero_changed"] is True
        assert row["zero_changed"] is False
        assert abs(row["zero_dk_samples"]) <= 1e-3
    by_condition = {(row["polarity"], row["snr_db"]): row for row in rows}
    for polarity in (1, -1):
        dk = [
            abs(by_condition[(polarity, snr)]["zero_dk_samples"])
            for snr in (0, 6, 12, 18)
        ]
        ratios = np.array(dk[1:]) / np.array(dk[:-1])
        assert np.all(np.abs(ratios - 2.0) <= 0.05), ratios
