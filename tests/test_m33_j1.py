"""M3-3 J1: box -> apex candidates (J1-1) and the judgment wrapper (J1-2, J1-3).

Spec: ``docs/processing/m33_j1_preregistration.md`` v0.3.4 §3, §5 J1-4.

The candidate rule runs on synthetic envelopes (no radargram). The judge
runs on a small fake R2 folder; ``eval_depth_4tu.main`` is replaced by a
stand-in that writes summary / matches / clusters through the real
``_evaluate_one`` on three hand-made activities, so the baseline the judge
reproduces is the one the stand-in drew. Git, the 4TU metadata and the line
list are replaced too. No stored result and no ground-truth file is read;
the ``realdata`` tests (opt-in) read one 4TU line.
"""

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis" / "processing"))
import m33_j1_candidates as cand  # noqa: E402
import m33_j1_judge as judge  # noqa: E402
import m33_j1_seal as seal  # noqa: E402

depth_eval = judge.depth_eval

DT_NS = 0.1
F_C_HZ = 500e6
PERIOD_SAMPLES = 20.0
"""``1 / f_c`` in samples at ``DT_NS``."""

BOX_COLUMNS = list(cand.BOX_COLUMNS)
LINES = {("01.1", "Path1"), ("01.1", "Path2"), ("04.1", "Path1"), ("06.1", "Path1")}
GT_DEPTHS = {"01.1": [0.5, 0.9], "04.1": [0.7], "06.1": [1.0, 1.2, 1.4]}
"""Fake ground truth (top) of three sites; 6 truths."""
ESTIMATES = {"01.1": [0.52, 0.95], "04.1": [0.71, 1.5], "06.1": [1.05], "09.9": [0.4]}
"""Fake kept clusters; 09.9 has no ground truth."""
LINES_PER_ACTIVITY = {"01.1": 2, "04.1": 1, "06.1": 1, "09.9": 1}
SECRET = "SECRET_METRIC_0.4242"
"""Printed and logged by the stand-in depth run; must never reach the console."""


# ===================================================== synthetic envelopes


def _ricker(t_ns: np.ndarray) -> np.ndarray:
    arg = (np.pi * F_C_HZ * 1e-9 * t_ns) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def _section(
    apex_sample: float = 150.0,
    apex_trace: int = 50,
    band_sample: float | None = None,
    band_amplitude: float = 0.0,
    n_samples: int = 400,
    n_traces: int = 101,
) -> np.ndarray:
    """A point-diffractor hyperbola (amplitude falling off the apex) and a band."""
    times_ns = np.arange(n_samples) * DT_NS
    apex_ns = apex_sample * DT_NS
    section = np.zeros((n_samples, n_traces))
    for trace in range(n_traces):
        offset_m = (trace - apex_trace) * 0.02
        arrival_ns = math.hypot(apex_ns, 2.0 * offset_m / 0.1)
        section[:, trace] += (apex_ns / arrival_ns) ** 2 * _ricker(
            times_ns - arrival_ns
        )
        if band_sample is not None:
            band = _ricker(times_ns - band_sample * DT_NS)
            section[:, trace] += band_amplitude * band
    return section


def _prep(section: np.ndarray, k_zero: float | None = 10.0) -> cand.LinePrep:
    t_zero = None if k_zero is None else k_zero * DT_NS
    return cand.LinePrep(
        envelope=cand.line_envelope(section),
        dt_ns=DT_NS,
        t_zero_ns=t_zero,
        k_zero=k_zero,
        period_samples=PERIOD_SAMPLES,
    )


def _box(i0: int, i1: int, k0: int, score: float = 0.8, **fields) -> cand.DetectorBox:
    values = {
        "line_id": "01.1/Path1",
        "activity_id": "01.1",
        "path": "Path1",
        "fold": 0,
        "class_id": 0,
        "score": score,
        "score_text": repr(score),
        "i0": i0,
        "i1": i1,
        "k0": k0,
        "k1": k0 + 100,
        "pp_action": "none",
        "tile_truncated": "False",
    }
    return cand.DetectorBox(**(values | fields))


# ================================================= J1-4 (a)-(e): candidates


def test_a_synthetic_hyperbola_box_finds_the_apex():
    prep = _prep(_section(apex_sample=150.0, apex_trace=50))
    candidate = cand.candidate_for_box(_box(30, 70, 147), prep)
    assert isinstance(candidate, cand.ApexCandidate)
    assert abs(candidate.trace - 50) <= 1
    assert abs(candidate.sample - 150.0) <= 0.25 * PERIOD_SAMPLES


def test_window_uses_the_centre_half_of_the_traces():
    window = cand.search_window(_box(30, 70, 147), _prep(_section()))
    assert (window.trace_first, window.trace_last) == (40, 60)
    narrow = cand.search_window(_box(48, 54, 147), _prep(_section()))
    assert (narrow.trace_first, narrow.trace_last) == (48, 54)


def test_e_zero_band_is_left_out_of_the_window():
    section = _section(apex_sample=60.0, band_sample=30.0, band_amplitude=3.0)
    box = _box(30, 70, 20)  # top above the zero band
    guarded = cand.candidate_for_box(box, _prep(section, k_zero=30.3))
    assert guarded.sample > 30.3 + 0.5 * PERIOD_SAMPLES
    assert abs(guarded.sample - 60.0) <= 0.25 * PERIOD_SAMPLES
    # Without a zero the window reaches the band, whose envelope is larger.
    unguarded = cand.candidate_for_box(box, _prep(section, k_zero=None))
    assert abs(unguarded.sample - 30.0) <= 0.25 * PERIOD_SAMPLES


def test_e_window_wholly_above_the_zero_gives_no_candidate():
    prep = _prep(_section(apex_sample=150.0), k_zero=45.0)
    outcome = cand.candidate_for_box(_box(30, 70, 0), prep)  # 0 + 2.5 P <= 45 + 0.5 P
    assert isinstance(outcome, cand.SkippedBox)
    assert outcome.reason == cand.SKIP_WINDOW_ABOVE_ZERO


def test_c_zero_failure_leaves_tau_blank_and_starts_half_a_period_up():
    prep = _prep(_section(apex_sample=150.0), k_zero=None)
    window = cand.search_window(_box(30, 70, 147), prep)
    assert window.row_first == math.ceil(147 - 0.5 * PERIOD_SAMPLES)
    candidate = cand.candidate_for_box(_box(30, 70, 147), prep)
    row = cand.candidate_row(candidate, prep, "j1_fold0_post")
    assert math.isnan(row["tau_ns"])
    assert row["t_ns"] == pytest.approx(candidate.sample * DT_NS)


def test_tau_is_time_after_the_detected_zero():
    prep = _prep(_section(apex_sample=150.0), k_zero=10.0)
    candidate = cand.candidate_for_box(_box(30, 70, 147), prep)
    row = cand.candidate_row(candidate, prep, "j1_fold0_post")
    assert row["tau_ns"] == pytest.approx(candidate.sample * DT_NS - 1.0)


def _apex(trace: int, sample: float, score: float) -> cand.ApexCandidate:
    return cand.ApexCandidate(_box(0, 10, 0, score=score), trace, sample)


def test_b_dedupe_keeps_the_higher_score():
    low, high = _apex(50, 150.0, 0.5), _apex(52, 155.0, 0.9)
    far = _apex(90, 150.0, 0.3)
    kept = cand.deduplicate([low, high, far], PERIOD_SAMPLES)
    assert kept == [high, far]


def test_b_dedupe_tie_keeps_the_smaller_trace_and_sample():
    first, second = _apex(52, 150.0, 0.7), _apex(50, 151.0, 0.7)
    assert cand.deduplicate([first, second], PERIOD_SAMPLES) == [second]


def test_b_dedupe_radius_is_inclusive():
    base = _apex(50, 150.0, 0.9)
    on_edge = _apex(53, 150.0 + 0.5 * PERIOD_SAMPLES, 0.5)
    outside = _apex(54, 150.0, 0.5)
    assert cand.deduplicate([base, on_edge, outside], PERIOD_SAMPLES) == [base, outside]


def _write_box_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows, columns=BOX_COLUMNS).to_csv(
        path, index=False, lineterminator="\n"
    )
    return path


def _box_row(
    line_id: str, i0: int, i1: int, k0: int, score: str, fold: int = 0
) -> dict:
    activity_id, path = line_id.split("/", 1)
    return {
        "line_id": line_id,
        "activity_id": activity_id,
        "path": path,
        "fold": fold,
        "class_id": 0,
        "score": score,
        "i0": i0,
        "i1": i1,
        "k0": k0,
        "k1": k0 + 100,
        "x0_line_px": 1.5,
        "y1_px": 2.5,
        "y2_px": 30.25,
        "pp_action": "none",
        "tile_truncated": "False",
        "n_tiles_seen": 2,
    }


def test_d_build_writes_the_signal_candidate_columns(tmp_path, monkeypatch):
    rows = [
        _box_row("01.1/Path1", 30, 70, 147, "0.61"),
        _box_row("01.1/Path1", 31, 69, 146, "0.93"),  # same apex: dropped
        _box_row("01.1/Path1", 30, 70, 0, "0.5"),  # window above the zero
    ]
    boxes = _write_box_csv(tmp_path / "j1_boxes_post.csv", rows)
    prep = _prep(_section(apex_sample=150.0), k_zero=45.0)
    monkeypatch.setattr(cand, "prepare_line", lambda radargram: prep)
    built = cand.build_candidate_set(
        boxes, 0, "post", LINES, load_line=lambda a, p: None
    )
    assert (built.n_boxes, built.n_before_dedupe, built.n_candidates) == (3, 2, 1)
    assert built.skipped_by_reason() == {cand.SKIP_WINDOW_ABOVE_ZERO: 1}
    out = tmp_path / "candidates.csv"
    cand.write_candidates_csv(built.table, out)
    signal_csv = REPO / "docs" / "processing" / "fourtu_apex_candidates_v1.csv"
    signal_header = signal_csv.read_text(encoding="utf-8").splitlines()[0]
    assert out.read_text(encoding="utf-8").splitlines()[0] == signal_header
    written = pd.read_csv(out, dtype={"activity_id": str, "path": str})
    assert written.loc[0, "source"] == "j1_fold0_post"
    assert written.loc[0, "note"] == "score=0.93;pp_action=none;tile_truncated=False"
    assert written.loc[0, "sample"] == built.table.loc[0, "sample"]  # full precision


def test_build_refuses_boxes_off_the_line_list(tmp_path):
    boxes = _write_box_csv(tmp_path / "b.csv", [_box_row("99.9/Path1", 1, 9, 5, "0.5")])
    with pytest.raises(ValueError, match="line list"):
        cand.build_candidate_set(boxes, 0, "post", LINES, load_line=lambda a, p: None)


def test_box_columns_match_the_r2_runner():
    sys.path.insert(0, str(REPO / "analysis" / "detection"))
    import m33_r2_eval

    assert tuple(m33_r2_eval.J1_COLUMNS) == cand.BOX_COLUMNS


def _real_line_or_skip(activity_id: str, path: str):
    try:
        return cand.load_fourtu_line(activity_id, path)
    except (OSError, KeyError) as error:
        pytest.skip(f"4TU data not available: {error}")


@pytest.mark.realdata
def test_d_real_line_candidates_pass_the_depth_dry_run(tmp_path, capsys):
    radargram = _real_line_or_skip("01.1", "Path1")
    prep = cand.prepare_line(radargram)
    assert prep.k_zero is not None and prep.period_samples > 0
    width = radargram.n_traces // 2
    rows = [_box_row("01.1/Path1", width // 2, width // 2 + width, 0, "0.9")]
    boxes = _write_box_csv(tmp_path / "j1_boxes_post.csv", rows)
    built = cand.build_candidate_set(boxes, 0, "post", LINES)
    for candidate_sample in built.table["sample"]:
        assert candidate_sample > prep.k_zero + 0.5 * prep.period_samples
    out = tmp_path / "candidates.csv"
    cand.write_candidates_csv(built.table, out)
    assert depth_eval.main(["--dry-run", "--candidates", str(out)]) == 0
    capsys.readouterr()


# ================================================== judge: fake R2 world


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(post_sha: dict[int, str], pre_sha: dict[int, str]) -> dict:
    weights = seal.FOLD0_WEIGHTS_SHA256
    folds = {
        str(fold): {
            "post_path": f"fold{fold}/j1_boxes_post.csv",
            "post_sha256": post_sha[fold],
            "pre_path": f"fold{fold}/j1_boxes_pre.csv",
            "pre_sha256": pre_sha[fold],
            "n_boxes_post": 2,
            "n_boxes_pre": 3,
            "n_padding_only_dropped": 0,
            "n_padding_only_dropped_pre": 0,
        }
        for fold in range(5)
    }
    r0 = {f"{d}/{line}": sha for d, line, sha in judge.m33_inputs.R0_IDENTITY_LINES}
    return {
        "status": "ok",
        "full_run": True,
        "n_lines": 754,
        "judgment": {
            "judgeable": True,
            "blockers": [],
            "expected_folds": [0, 1, 2, 3, 4],
            "folds_run": [0, 1, 2, 3, 4],
        },
        "j1_inputs": {
            "pp_set_id": judge.m33_inputs.C0B_PP_SET_ID.value,
            "pp_config_sha256": judge.m33_inputs.C0B_PP_CONFIG_SHA256,
            "weights_sha256": {"0": weights},
            "n_lines": 754,
            "folds": folds,
        },
        "run_identity": {
            "git": {"head": seal.C1_SEAL_COMMIT, "dirty": False},
            "dirty": False,
            "tree_end": {"head": seal.C1_SEAL_COMMIT, "dirty": False},
            "inputs_sha256": {"0": weights},
            "weights_sha256": {},
            "r_checks": {
                "settings": {"settings_sha256": judge.m33_inputs.R_SETTINGS_SHA256},
                "r0_identity_sha256": r0,
            },
        },
        "pp_actions": {"0": {"removed_a": 1}},
    }


def _default_box_rows(fold: int) -> list[dict]:
    return [
        _box_row("01.1/Path1", 30, 70, 147, "0.9", fold),
        _box_row("04.1/Path1", 10, 40, 80, "0.4", fold),
    ]


def _gt_items() -> dict[str, list]:
    return {
        activity: [
            depth_eval.GtItem(depth, "P", 0.0, False, "1", False) for depth in depths
        ]
        for activity, depths in GT_DEPTHS.items()
    }


def _cluster_rows() -> list[dict]:
    rows = []
    for activity, depths in ESTIMATES.items():
        for index, depth in enumerate(depths):
            rows.append(
                {
                    "activity_id": activity,
                    "zero_variant": "detected",
                    "cluster_id": index,
                    "depth_m": depth,
                    "kept_by_support": True,
                }
            )
    rows.append(  # a cluster the support filter dropped, and another zero
        {
            "activity_id": "01.1",
            "zero_variant": "detected",
            "cluster_id": 9,
            "depth_m": 2.0,
            "kept_by_support": False,
        }
    )
    rows.append(
        {
            "activity_id": "06.1",
            "zero_variant": "est",
            "cluster_id": 0,
            "depth_m": 1.0,
            "kept_by_support": True,
        }
    )
    return rows


DRY_RUN_EXIT = {"code": 0}
"""Exit code of the stand-in dry run (a test sets 1 to fail it)."""


def fake_dry_run(argv: list[str]) -> int:
    """Stand-in ``--dry-run``: reads the candidate csv, prints counts only."""
    candidates = pd.read_csv(argv[argv.index("--candidates") + 1], dtype=str)
    print(SECRET)
    print(json.dumps({"candidates": len(candidates)}))
    return DRY_RUN_EXIT["code"]


def fake_eval_main(argv: list[str], tamper=None) -> int:
    """Stand-in for ``eval_depth_4tu.main``: real ``_evaluate_one``, fake inputs."""
    if "--dry-run" in argv:
        return fake_dry_run(argv)
    prefix = argv[argv.index("--prefix") + 1]
    print(SECRET)
    depth_eval.LOGGER.info(SECRET)
    evaluation = depth_eval.Evaluation()
    args = SimpleNamespace(
        baseline_trials=1000, baseline_trials_aux=20, baseline_seed=0
    )
    combination = depth_eval.Combination("exclude_x", "top", "detected", "main")
    depth_eval._evaluate_one(evaluation, ESTIMATES, _gt_items(), combination, args)
    out = judge.OUT_DIR
    summary_rows = copy.deepcopy(evaluation.summary_rows)
    if tamper is not None:
        tamper(summary_rows)
    depth_eval.write_csv(out / f"{prefix}_summary.csv", summary_rows)
    depth_eval.write_csv(out / f"{prefix}_matches.csv", evaluation.match_rows)
    depth_eval.write_csv(out / f"{prefix}_clusters.csv", _cluster_rows())
    candidates = [{"activity_id": "01.1", "status": "ok", "qc_excluded": False}]
    depth_eval.write_csv(out / f"{prefix}_candidates.csv", candidates)
    (out / f"{prefix}.md").write_text("# stand-in\n", encoding="utf-8")
    return 0


def fake_build(boxes_csv, fold, pp, lines, **_):
    rows = []
    for box in cand.read_boxes(boxes_csv):
        rows.append(
            {
                "activity_id": box.activity_id,
                "site": 1,
                "path": box.path,
                "trace": (box.i0 + box.i1) // 2,
                "sample": box.k0 + 0.5,
                "t_ns": 1.0,
                "tau_ns": 0.5,
                "snr": None,
                "n_left": None,
                "n_right": None,
                "v_straight_qc": None,
                "source": cand.source_label(fold, pp),
                "note": "",
            }
        )
    table = pd.DataFrame(rows, columns=list(cand.CANDIDATE_COLUMNS))
    return cand.CandidateSet(table, len(rows), len(rows), (), 0)


class World:
    """A fake R2 folder, output folders and the patched judge collaborators."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.tmp = tmp_path
        self.monkeypatch = monkeypatch
        self.out_dir = tmp_path / "docs"
        self.out_dir.mkdir()
        self.r2_root = tmp_path / "r2"
        self.failed_tree: list[str] = []
        self.eval_tamper = None
        self.box_rows = {fold: _default_box_rows(fold) for fold in range(5)}
        self.summary_patch = None
        self._patch()

    def _patch(self) -> None:
        patch = self.monkeypatch.setattr
        patch(judge, "OUT_DIR", self.out_dir)
        patch(judge, "RUN_LOG_DIR", self.tmp / "log")
        patch(depth_eval, "DEFAULT_OUT_DIR", self.out_dir)
        patch(depth_eval, "main", lambda argv: fake_eval_main(argv, self.eval_tamper))
        patch(judge, "N_GT", sum(len(v) for v in GT_DEPTHS.values()))
        patch(judge, "check_tree", self._fake_tree)
        patch(judge, "check_fourtu_meta", lambda checks: None)
        patch(judge, "is_ancestor", lambda ancestor, head: True)
        patch(judge, "load_line_set", lambda: set(LINES))
        patch(judge, "load_baseline_inputs", self._baseline_inputs)
        patch(seal, "unfilled_names", lambda: [])
        patch(cand, "build_candidate_set", fake_build)

    def _fake_tree(self, checks: judge.Checks) -> dict:
        checks.require("1.tree_clean", "1.tree_clean" not in self.failed_tree)
        return {"head": "c2c2c2", "dirty": False, "problems": [], "code_blobs": {}}

    @staticmethod
    def _baseline_inputs() -> judge.BaselineInputs:
        return judge.BaselineInputs(copy.deepcopy(GT_DEPTHS), dict(LINES_PER_ACTIVITY))

    def write(self) -> None:
        """Write the box files and the summary; seal the fold-0 sha."""
        post_sha, pre_sha = {}, {}
        for fold in range(5):
            folder = self.r2_root / f"fold{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            post = _write_box_csv(folder / "j1_boxes_post.csv", self.box_rows[fold])
            pre_rows = [
                *self.box_rows[fold],
                _box_row("06.1/Path1", 5, 9, 9, "0.3", fold),
            ]
            pre_rows[-1]["pp_action"] = "removed_d"
            pre = _write_box_csv(folder / "j1_boxes_pre.csv", pre_rows)
            post_sha[fold], pre_sha[fold] = _sha(post), _sha(pre)
        summary = _summary(post_sha, pre_sha)
        if self.summary_patch is not None:
            self.summary_patch(summary)
        (self.r2_root / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        self.monkeypatch.setattr(seal, "R2_J1_POST_SHA256", post_sha[0])
        self.monkeypatch.setattr(seal, "R2_J1_PRE_SHA256", pre_sha[0])
        self.post_sha = post_sha[0]

    def argv(self, *extra: str, boxes_sha: str | None = None) -> list[str]:
        sha = self.post_sha if boxes_sha is None else boxes_sha
        return [
            "--boxes",
            str(self.r2_root / "fold0" / "j1_boxes_post.csv"),
            "--boxes-sha256",
            sha,
            "--pp-config-sha256",
            judge.m33_inputs.C0B_PP_CONFIG_SHA256,
            "--prefix",
            "depth_4tu_j1",
            *extra,
        ]

    def run(
        self,
        capsys,
        *extra: str,
        boxes_sha: str | None = None,
        auto_precheck: bool = True,
    ):
        """One judge call; a judgment run gets its precheck first if none exists."""
        self.write()
        is_judgment = not {"--report", "--precheck"} & set(extra)
        if auto_precheck and is_judgment and not judge.precheck_record_path().exists():
            judge.main(self.argv("--precheck", boxes_sha=boxes_sha))
            capsys.readouterr()
        code = judge.main(self.argv(*extra, boxes_sha=boxes_sha))
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def verdict(self) -> dict:
        path = self.out_dir / "depth_4tu_j1_verdict.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def refused_names(self) -> list[str]:
        records = sorted((self.tmp / "log").glob("*_refused_*.json"))
        document = json.loads(records[-1].read_text(encoding="utf-8"))
        return [r["name"] for r in document["assertions"] if not r["passed"]]


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    return World(tmp_path, monkeypatch)


def _last_json(stdout: str) -> dict:
    return json.loads(stdout.strip().splitlines()[-1])


# ------------------------------------------------------------- happy path


def test_judgment_run_writes_the_verdict_and_stays_blind(world, capsys):
    code, out, err = world.run(capsys)
    assert code == 0, err
    last = _last_json(out)
    assert set(last) == {"status", "out", "n_candidates", "elapsed_s"}
    assert last["status"] == "ok" and last["n_candidates"] == 2
    assert SECRET not in out and SECRET not in err
    assert "판정" not in out and "n_pass" not in out
    assert all(line.startswith("phase ") for line in out.strip().splitlines()[:-1])
    assert SECRET in (world.tmp / "log" / "depth_4tu_j1_eval.log").read_text("utf-8")
    verdict = world.verdict()
    assert verdict["status"] == "ok" and not [
        r for r in verdict["assertions"] if not r["passed"]
    ]
    assert verdict["verdict"]["case"] in {case.value for case in judge.VerdictCase}
    assert (world.out_dir / "depth_4tu_j1_verdict.md").is_file()
    assert (world.out_dir / "depth_4tu_j1_candidates_input.csv").is_file()


def test_eval_output_sha_is_recorded_in_verdict_and_report(world, capsys):
    assert world.run(capsys)[0] == 0
    assert world.run(capsys, "--report", "pre")[0] == 0
    for prefix, result in (
        ("depth_4tu_j1", "depth_4tu_j1_verdict.json"),
        ("depth_4tu_j1_pre", "depth_4tu_j1_pre_report.json"),
    ):
        document = json.loads((world.out_dir / result).read_text(encoding="utf-8"))
        recorded = document["eval_outputs_sha256"]
        for suffix in judge.EVAL_OUTPUT_SUFFIXES:
            name = f"{prefix}_{suffix}.csv"
            assert recorded["raw"][name] == _sha(world.out_dir / name)
        assert set(recorded["lf"]) == set(recorded["raw"])


def test_verdict_counts_and_report_items(world, capsys):
    world.run(capsys)
    verdict = world.verdict()
    main = verdict["main_result"]
    assert (main["n_pass"], main["n_pairs"]) == (4, 4)
    reach = verdict["report"]["reachability"]
    assert reach["excess_estimates"] == 1  # 04.1: two estimates, one truth
    assert reach["n_single_gt_activities_with_estimate"] == 1
    assert [row["site"] for row in verdict["report"]["leave_one_site_out"]] == [1, 4, 6]
    assert (
        verdict["report"]["site_bootstrap"]["replicates"] == judge.BOOTSTRAP_REPLICATES
    )


# ------------------------------------------------------ assertions: exit 2


def _set(path: tuple, value):
    def patch(summary: dict) -> None:
        node = summary
        for key in path[:-1]:
            node = node[key]
        if value is _DELETE:
            node.pop(path[-1])
        else:
            node[path[-1]] = value

    return patch


_DELETE = object()

SUMMARY_FAULTS = [
    (("status",), "not_judgeable", "3.status_ok"),
    (("judgment", "judgeable"), False, "3.judgeable"),
    (("full_run",), False, "3.full_run"),
    (("judgment", "blockers"), ["smoke"], "3.no_blockers"),
    (("judgment", "folds_run"), [0], "3.all_folds"),
    (("judgment", "expected_folds"), [0, 1, 2, 3], "3.all_folds"),
    (("j1_inputs", "n_lines"), 720, "3.j1_n_lines_754"),
    (("n_lines",), 720, "3.n_lines_754"),
    (("j1_inputs", "weights_sha256", "0"), "f" * 64, "3.fold0_weights_sha"),
    (("run_identity", "inputs_sha256"), _DELETE, "3.fold0_weights_sha"),
    (("run_identity", "inputs_sha256", "0"), "", "3.fold0_weights_sha"),
    (("run_identity", "r_checks", "r0_identity_sha256"), {}, "3.r0_11_lines_c1"),
    (
        ("run_identity", "r_checks", "settings", "settings_sha256"),
        "0" * 64,
        "3.settings_sha_c1",
    ),
    (("j1_inputs", "pp_config_sha256"), "1" * 64, "3.pp_config_sha_c0b"),
    (("j1_inputs", "pp_set_id"), "S4", "3.pp_set_id_c0b"),
    (("run_identity", "dirty"), True, "3.r2_tree_clean"),
    (("run_identity", "tree_end", "dirty"), True, "3.r2_tree_clean"),
    (("run_identity", "tree_end", "head"), "other", "3.r2_head_after_c1"),
    (("j1_inputs", "folds", "0", "post_sha256"), "2" * 64, "2.post_sha_r2_summary"),
]


def _assert_refused(world: World, code: int, out: str, expected_name: str) -> None:
    assert code == 2
    assert _last_json(out)["status"] == "input_error"
    assert expected_name in world.refused_names()
    assert list(world.out_dir.iterdir()) == []  # nothing under docs/


@pytest.mark.parametrize(("path", "value", "expected"), SUMMARY_FAULTS)
def test_r2_summary_faults_are_refused(world, capsys, path, value, expected):
    world.summary_patch = _set(path, value)
    code, out, _ = world.run(capsys)
    _assert_refused(world, code, out, expected)


def _mutate_first_row(field_name: str, value: str):
    def mutate(rows: list[dict]) -> list[dict]:
        rows = copy.deepcopy(rows)
        rows[0][field_name] = value
        return rows

    return mutate


BOX_FAULTS = [
    ("fold", "1", "4.fold"),
    ("pp_action", "removed_d", "4.no_removed_rows"),
    ("class_id", "1", "4.class_pipe_hyperbola"),
    ("activity_id", "01.2", "4.line_id_split"),
]


@pytest.mark.parametrize(("field_name", "value", "expected"), BOX_FAULTS)
def test_box_file_faults_are_refused(world, capsys, field_name, value, expected):
    world.box_rows[0] = _mutate_first_row(field_name, value)(world.box_rows[0])
    code, out, _ = world.run(capsys)
    _assert_refused(world, code, out, expected)


def test_box_off_the_754_lines_is_refused(world, capsys):
    world.box_rows[0] = [_box_row("99.9/Path1", 1, 9, 5, "0.5")]
    code, out, _ = world.run(capsys)
    _assert_refused(world, code, out, "4.lines_in_754")


def test_cli_sha_and_pp_hash_are_asserted(world, capsys):
    code, out, _ = world.run(capsys, boxes_sha="3" * 64)
    _assert_refused(world, code, out, "2.post_sha_cli")


def test_wrong_pp_hash_argument_is_refused(world, capsys):
    world.write()
    argv = world.argv()
    argv[argv.index("--pp-config-sha256") + 1] = "4" * 64
    code = judge.main(argv)
    _assert_refused(world, code, capsys.readouterr().out, "3.pp_config_sha_c0b")


def test_sealed_post_sha_must_match(world, capsys, monkeypatch):
    world.write()
    monkeypatch.setattr(seal, "R2_J1_POST_SHA256", "5" * 64)
    code = judge.main(world.argv())
    _assert_refused(world, code, capsys.readouterr().out, "2.post_sha_c2")


def test_unfilled_seal_is_refused(world, capsys, monkeypatch):
    world.write()
    monkeypatch.setattr(seal, "unfilled_names", lambda: ["SRC_TREE_ID"])
    code = judge.main(world.argv())
    _assert_refused(world, code, capsys.readouterr().out, "1.c2_seal_filled")


def test_dirty_tree_is_refused(world, capsys):
    world.failed_tree = ["1.tree_clean"]
    code, out, _ = world.run(capsys)
    _assert_refused(world, code, out, "1.tree_clean")


def test_r2_run_before_c1_is_refused(world, capsys, monkeypatch):
    monkeypatch.setattr(judge, "is_ancestor", lambda ancestor, head: False)
    code, out, _ = world.run(capsys)
    _assert_refused(world, code, out, "3.r2_head_after_c1")


def test_other_prefix_is_refused(world, capsys):
    world.write()
    argv = world.argv()
    argv[argv.index("--prefix") + 1] = "depth_4tu_j1b"
    code = judge.main(argv)
    _assert_refused(world, code, capsys.readouterr().out, "args.prefix")


def test_changed_candidate_csv_is_refused(world, capsys, monkeypatch):
    original = judge.write_candidates

    def tampering(*args):
        record = original(*args)
        with record.path.open("a", encoding="utf-8") as handle:
            handle.write("01.1,1,Path1,1,1.0,1.0,1.0,,,,,x,\n")
        return record

    monkeypatch.setattr(judge, "write_candidates", tampering)
    code, _, _ = world.run(capsys)
    assert code == 2
    assert "5.candidates_sha_unchanged" in world.refused_names()


# ------------------------------------------------------- §1-5 3 precheck


def _precheck_record() -> dict:
    return json.loads(judge.precheck_record_path().read_text(encoding="utf-8"))


def test_precheck_runs_the_dry_run_and_writes_no_judgment_output(world, capsys):
    code, out, err = world.run(capsys, "--precheck")
    assert code == 0, err
    last = _last_json(out)
    assert set(last) == {"status", "out", "n_candidates", "elapsed_s"}
    assert last["status"] == "ok" and last["n_candidates"] == 2
    assert SECRET not in out and SECRET not in err
    record = _precheck_record()
    assert (record["status"], record["dry_run_exit"]) == ("ok", 0)
    assert record["dry_run_report"] == {"candidates": 2}
    candidates = judge.precheck_candidates_path()
    assert record["candidates_sha256"] == _sha(candidates)
    assert list(world.out_dir.iterdir()) == []  # nothing under docs/
    # The judgment is not refused by assertion 6 and reproduces the candidates.
    code, _, err = world.run(capsys)
    assert code == 0, err
    judged = world.out_dir / "depth_4tu_j1_candidates_input.csv"
    assert _sha(judged) == record["candidates_sha256"]


def test_judgment_without_a_precheck_is_refused(world, capsys):
    code, out, _ = world.run(capsys, auto_precheck=False)
    _assert_refused(world, code, out, "1-5.3.precheck_dry_run_passed")


def test_failed_dry_run_fails_the_precheck_and_blocks_the_judgment(
    world, capsys, monkeypatch
):
    monkeypatch.setitem(DRY_RUN_EXIT, "code", 1)
    code, out, _ = world.run(capsys, "--precheck")
    assert code != 0 and _last_json(out)["status"] == "precheck_failed"
    assert _precheck_record()["dry_run_exit"] == 1
    code, out, _ = world.run(capsys)
    _assert_refused(world, code, out, "1-5.3.precheck_dry_run_passed")


def test_candidates_that_differ_from_the_precheck_are_refused(world, capsys):
    assert world.run(capsys, "--precheck")[0] == 0
    path = judge.precheck_record_path()
    record = _precheck_record()
    record["candidates_sha256"] = "6" * 64
    path.write_text(json.dumps(record), encoding="utf-8")
    code, out, _ = world.run(capsys)
    _assert_refused(world, code, out, "1-5.3.candidates_sha_equals_precheck")


def test_precheck_on_other_boxes_is_refused(world, capsys):
    assert world.run(capsys, "--precheck")[0] == 0
    path = judge.precheck_record_path()
    record = _precheck_record()
    record["boxes_sha256"] = "7" * 64
    path.write_text(json.dumps(record), encoding="utf-8")
    code, out, _ = world.run(capsys)
    _assert_refused(world, code, out, "1-5.3.precheck_same_boxes")


def test_refused_precheck_leaves_no_passing_record(world, capsys):
    assert world.run(capsys, "--precheck")[0] == 0
    (world.out_dir / "depth_4tu_j1.md").write_text("", encoding="utf-8")
    code, _, _ = world.run(capsys, "--precheck")
    assert code == 2
    assert not judge.precheck_record_path().exists()


def test_head_rule_accepts_docs_only_commits(monkeypatch):
    def fake_git(*args):
        if args[0] == "log":
            return "c2commit"
        return "docs/processing/m33_j1_preregistration.md\ndocs/CHECKLIST.md"

    monkeypatch.setattr(judge, "_git", fake_git)
    checks = judge.Checks()
    judge._check_head_after_c2(checks)
    assert checks.failed() == []


def test_head_rule_refuses_code_commits_after_c2(monkeypatch):
    def fake_git(*args):
        return "c2commit" if args[0] == "log" else "analysis/processing/m33_j1_judge.py"

    monkeypatch.setattr(judge, "_git", fake_git)
    checks = judge.Checks()
    judge._check_head_after_c2(checks)
    assert checks.failed() == ["1.head_is_c2_or_docs_only"]


def _run_with_changes_after_c2(world, capsys, monkeypatch, changed: str) -> int:
    """Judgment run whose tree check uses the real HEAD rule on a fake git."""

    def fake_git(*args):
        return "c2commit" if args[0] == "log" else changed

    def tree_with_head_rule(checks):
        judge._check_head_after_c2(checks)
        return world._fake_tree(checks)

    monkeypatch.setattr(judge, "_git", fake_git)
    monkeypatch.setattr(judge, "check_tree", tree_with_head_rule)
    code, _, _ = world.run(capsys)
    return code


def test_agent_memory_change_after_c2_counts_as_docs(world, capsys, monkeypatch):
    changed = "docs/CHECKLIST.md\n.claude/agent-memory/x.md"
    assert _run_with_changes_after_c2(world, capsys, monkeypatch, changed) == 0


@pytest.mark.parametrize("changed", [".claude/agents/x.md", "analysis/detection/x.py"])
def test_other_changes_after_c2_are_refused(world, capsys, monkeypatch, changed):
    assert _run_with_changes_after_c2(world, capsys, monkeypatch, changed) == 2
    assert "1.head_is_c2_or_docs_only" in world.refused_names()
    assert list(world.out_dir.iterdir()) == []


# --------------------------------------------------------- one run only


@pytest.mark.parametrize(
    "existing", ["depth_4tu_j1_summary.csv", "depth_4tu_j1_pre.md", "depth_4tu_j1.md"]
)
def test_existing_judgment_output_refuses_the_run(world, capsys, existing):
    (world.out_dir / existing).write_text("", encoding="utf-8")
    world.write()
    code = judge.main(world.argv())
    captured = capsys.readouterr()
    assert code == 2
    assert "6.no_depth_4tu_j1_output" in world.refused_names()
    assert sorted(p.name for p in world.out_dir.iterdir()) == [existing]
    assert SECRET not in captured.out


def test_second_judgment_run_is_refused(world, capsys):
    assert world.run(capsys)[0] == 0
    verdict_bytes = (world.out_dir / "depth_4tu_j1_verdict.json").read_bytes()
    code, _, _ = world.run(capsys)
    assert code == 2
    assert (world.out_dir / "depth_4tu_j1_verdict.json").read_bytes() == verdict_bytes


# ------------------------------------------------- §1-1 3 and reproduction


def _set_all_row(column: str, value: str):
    def tamper(rows: list[dict]) -> None:
        for row in rows:
            if row["scope"] == "all":
                row[column] = value

    return tamper


def test_i_pass_b_that_disagrees_with_the_matches_exits_2(world, capsys):
    world.eval_tamper = _set_all_row("pass_b", 0.5)
    code, out, _ = world.run(capsys)
    assert code == 2 and _last_json(out)["status"] == "input_error"
    verdict = world.verdict()
    failed = [r["name"] for r in verdict["assertions"] if not r["passed"]]
    assert failed == ["1-1.3.n_pass_round_pass_b"]
    assert "verdict" not in verdict


def test_k_that_is_not_an_integer_exits_2(world, capsys):
    world.eval_tamper = _set_all_row("baseline_p", 0.0425)
    code, _, _ = world.run(capsys)
    assert code == 2
    failed = [r["name"] for r in world.verdict()["assertions"] if not r["passed"]]
    assert failed == ["1-1.3.k_integer"]


def test_unreproduced_baseline_leaves_null_report_and_exits_3(world, capsys):
    world.eval_tamper = _set_all_row("baseline_pass_b_mean", 0.1234)
    code, out, _ = world.run(capsys)
    assert code == 3 and _last_json(out)["status"] == "report_failed"
    verdict = world.verdict()
    assert verdict["report"] is None
    assert "all.baseline_pass_b_mean" in verdict["report_failure"]
    assert verdict["verdict"]["phrase"]  # the phrase does not depend on the report


# ------------------------------------------------------- J1-3 report runs


def test_report_run_needs_the_verdict_first(world, capsys):
    code, _, _ = world.run(capsys, "--report", "pre")
    assert code == 2
    assert "J1-3.verdict_written_first" in world.refused_names()


def test_report_runs_after_the_verdict_never_touch_it(world, capsys):
    assert world.run(capsys)[0] == 0
    verdict_bytes = (world.out_dir / "depth_4tu_j1_verdict.json").read_bytes()
    for kind, prefix in (("pre", "depth_4tu_j1_pre"), ("fold2", "depth_4tu_j1_f2")):
        code, _, err = world.run(capsys, "--report", kind)
        assert code == 0, err
        report = json.loads(
            (world.out_dir / f"{prefix}_report.json").read_text("utf-8")
        )
        assert "verdict" not in report and report["kind"] == f"report:{kind}"
        assert (world.out_dir / f"{prefix}_candidates_input.csv").is_file()
    assert (world.out_dir / "depth_4tu_j1_verdict.json").read_bytes() == verdict_bytes
    code, _, _ = world.run(capsys, "--report", "pre")
    assert code == 2 and "J1-3.no_report_output" in world.refused_names()


def test_signal_report_reruns_the_signal_candidates(world, capsys, monkeypatch):
    assert world.run(capsys)[0] == 0
    signal = world.tmp / "fourtu_apex_candidates_v1.csv"
    header = ",".join(cand.CANDIDATE_COLUMNS) + "\n"
    signal.write_text(header, encoding="utf-8", newline="\n")
    monkeypatch.setattr(depth_eval, "DEFAULT_APEX_CANDIDATES", signal)
    monkeypatch.setattr(seal, "SIGNAL_CANDIDATES_SHA256_LF", _sha(signal))
    code, _, err = world.run(capsys, "--report", "signal")
    assert code == 0, err
    assert (world.out_dir / "depth_4tu_j1_signal_report.json").is_file()
    assert not (world.out_dir / "depth_4tu_j1_signal_candidates_input.csv").exists()


# ===================================================== verdict phrases


def _result(n_pass: int, k: int) -> judge.MainResult:
    return judge.MainResult(
        n_pass=n_pass,
        k=k,
        n_pairs=150,
        pass_b=n_pass / 388,
        baseline_p=k / 1000,
        baseline_mean=0.45,
        baseline_p025=0.44,
        baseline_p975=0.4625,
        row={},
    )


@pytest.mark.parametrize(
    ("n_pass", "k", "case", "word", "particle"),
    [
        (120, 10, judge.VerdictCase.PASS, "**합격**", "로"),
        (120, 50, judge.VerdictCase.FAIL_BASELINE, "**불합격(기준선)**", "로"),
        (80, 49, judge.VerdictCase.FAIL_TARGET, "**불합격(목표)**", "이"),
        (80, 900, judge.VerdictCase.FAIL_BOTH, "**불합격(둘 다)**", "이"),
    ],
)
def test_four_verdict_cases(n_pass, k, case, word, particle):
    assert judge.decide_case(n_pass, k) is case
    phrase = judge.verdict_phrase(case, _result(n_pass, k))
    assert phrase.startswith(f"판정: {word}")
    candidate_source = "· 후보: 검출기 N2-a′ fold 0, PP(C0b) 후 박스"  # noqa: RUF001 - J1 §2-3 wording
    parenthesis = f"(판정률 B, `main`·관 상단·검출 영점·X 제외 {candidate_source})"
    assert f"{n_pass}/388{parenthesis}{particle} " in phrase
    assert judge.U3_PARENTHESIS not in phrase  # the source sits inside the parenthesis
    assert f"p = {k}/1000" in phrase
    baseline_text = "기준선 판정률 B 평균 0.4500, 2.5–97.5 % 0.4400–0.4625"  # noqa: RUF001 - [U3] wording
    assert baseline_text in phrase
    assert "제186조 준수" not in phrase


def test_verdict_boundaries_are_the_integer_rules():
    assert judge.decide_case(97, 49) is judge.VerdictCase.PASS
    assert judge.decide_case(96, 49) is judge.VerdictCase.FAIL_TARGET
    assert judge.decide_case(97, 50) is judge.VerdictCase.FAIL_BASELINE


@pytest.mark.parametrize(
    ("n_pass", "k", "has_p", "has_target"),
    [
        (120, 36, False, False),
        (120, 37, True, False),
        (120, 64, True, False),
        (120, 65, False, False),
        (88, 10, False, True),
        (106, 10, False, True),
        (107, 10, False, False),
        (87, 10, False, False),
    ],
)
def test_border_markers(n_pass, k, has_p, has_target):
    markers = "".join(judge.border_markers(n_pass, k))
    assert ("p 경계" in markers) is has_p
    assert ("목표 경계" in markers) is has_target


def test_clopper_pearson_contains_the_point_and_0_05_at_the_border():
    low, high = judge.clopper_pearson(49, 1000)
    assert low < 0.049 < high and low < 0.05 < high


# ============================================ J1-4 (f)-(h): report maths


def _draws(n_pass: np.ndarray, n_est: list[int], n_gt: list[int]):
    trials = n_pass.shape[0]
    pairs = np.tile(np.minimum(n_est, n_gt), (trials, 1))
    return depth_eval.BaselineDraws(
        activities=tuple(f"0{i + 1}.1" for i in range(len(n_est))),
        n_gt=np.array(n_gt),
        n_pass=n_pass,
        n_pairs=pairs,
        abs_error_sum=np.zeros(n_pass.shape),
        n_gate_pairs=np.zeros(n_pass.shape, dtype=int),
        n_est=np.array(n_est),
    )


def test_f_reachability_by_hand_when_perfect_cannot_reach_p():
    # n_e = (1, 2, 1), n_g = (1, 1, 2): pairs (1, 1, 1) = 3, E_+ = 1.
    n_pass = np.array([[1, 1, 1]] * 55 + [[1, 0, 1]] * 5)
    reach = judge.reachability(_draws(n_pass, [1, 2, 1], [1, 1, 2]))
    assert (reach["n_pairs"], reach["k_perfect"], reach["t_50"]) == (3, 55, 3)
    assert reach["n_req"] == 4 and reach["n_req_exceeds_n_pairs"]
    assert reach["perfect_observation_cannot_reach_p"]
    assert reach["excess_estimates"] == 1
    assert reach["n_activities_more_estimates_than_gt"] == 1
    assert reach["n_single_gt_activities_with_estimate"] == 2


def test_f_reachability_by_hand_when_p_is_reachable():
    n_pass = np.array([[1, 1, 1]] * 40 + [[1, 0, 0]] * 20)
    reach = judge.reachability(_draws(n_pass, [1, 2, 1], [1, 1, 2]))
    assert (reach["k_perfect"], reach["t_50"], reach["n_req"]) == (40, 1, 2)
    assert not reach["n_req_exceeds_n_pairs"]
    assert not reach["perfect_observation_cannot_reach_p"]
    assert reach["a_req"] == pytest.approx(2 / 3)


def _random_draws():
    gt = {"01.1": [0.5, 0.9], "01.4": [0.6], "04.1": [0.7, 1.1, 1.3], "06.2": [1.0]}
    n_est = {"01.1": 3, "01.4": 1, "04.1": 2, "06.2": 0}
    return depth_eval.draw_baseline(n_est, gt, trials=200, seed=0)


def _assert_same_stats(left, right) -> None:
    for name in left.__dataclass_fields__:
        a, b = getattr(left, name), getattr(right, name)
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b, equal_nan=True), name
        elif isinstance(a, float) and math.isnan(a):
            assert math.isnan(b), name
        else:
            assert a == b, name


def test_g_unit_weights_are_baseline_stats_bit_for_bit():
    draws = _random_draws()
    ones = np.ones(len(draws.activities), dtype=int)
    _assert_same_stats(
        judge.weighted_baseline_stats(draws, ones), depth_eval.baseline_stats(draws)
    )


def test_g_zero_weight_drops_one_site():
    draws = _random_draws()
    weights = np.array([0 if a.startswith("01.") else 1 for a in draws.activities])
    kept = [a for a in draws.activities if not a.startswith("01.")]
    _assert_same_stats(
        judge.weighted_baseline_stats(draws, weights),
        depth_eval.baseline_stats(draws, kept),
    )


def test_h_site_bootstrap_is_deterministic_per_seed():
    draws = _random_draws()
    sites = judge.site_table(draws, {"01.1": 2, "04.1": 1})
    first = judge.site_bootstrap(draws, sites, replicates=300)
    again = judge.site_bootstrap(draws, sites, replicates=300)
    other = judge.site_bootstrap(draws, sites, replicates=300, seed=1)
    assert first == again
    assert first != other


def test_leave_one_site_out_counts():
    draws = _random_draws()
    sites = judge.site_table(draws, {"01.1": 2, "04.1": 1})
    rows = {row["site"]: row for row in judge.leave_one_site_out(draws, sites)}
    assert sites.sites == (1, 4, 6)
    assert (rows[1]["n_pass"], rows[1]["n_gt"]) == (1, 4)
    without_site_1 = (draws.n_pass[:, 2:]).sum(axis=1)
    assert rows[1]["k"] == int(np.sum(without_site_1 >= 1))
