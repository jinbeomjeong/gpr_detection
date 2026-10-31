"""Unit tests of ``scripts/eval_depth_4tu.py`` (spec_eval_4tu_v1 section 9-2, E1-E8).

Analytic B-scans and small in-memory tables only; no 4TU data is read.
"""

import argparse

import numpy as np
import pandas as pd
import pytest

from gpr_detection.depth.estimate import prepare_depth_inputs
from gpr_detection.depth.refraction import reflection_times
from gpr_detection.depth.result import DepthResult, DepthStatus
from gpr_detection.io.radargram import Radargram
from synthetic_bscan import DT_NS, T_SRC_NS, BScanSpec, make_radargram


@pytest.fixture
def script(load_script):
    return load_script("eval_depth_4tu")


AIR_GAP_M, PIPE_DEPTH_M, EPS_R = 0.05, 0.6, 9.0


def _air_coupled_line() -> tuple[BScanSpec, Radargram]:
    """500 MHz air-coupled B-scan with one point pipe (refraction times)."""
    spec = BScanSpec(
        f_c_hz=500e6, air_gap_m=AIR_GAP_M, offset_m=0.1, d_top_m=PIPE_DEPTH_M
    )
    spec.pipe_times_ns = reflection_times(
        spec.x_m,
        offset_m=0.1,
        h=AIR_GAP_M,
        x0_m=spec.x0_m,
        d_top_m=PIPE_DEPTH_M,
        radius_m=0.0,
        v=spec.velocity,
    )
    return spec, make_radargram(spec, eps_r_meta=EPS_R, line_id="01.1/PathT")


def test_e1_true_apex_candidate_gives_the_depth(script) -> None:
    depth = PIPE_DEPTH_M
    spec, radargram = _air_coupled_line()
    apex_trace = spec.n_traces // 2
    apex_sample = (T_SRC_NS + spec.pipe_times_ns[apex_trace]) / DT_NS
    zero = prepare_depth_inputs(radargram).zero
    candidates = pd.DataFrame(
        {
            "trace": [apex_trace],
            "sample": [apex_sample],
            "tau_ns": [apex_sample * DT_NS - zero.t_zero_ns],
            "source": ["auto_v1"],
        }
    )
    (result,) = script.run_line(radargram, candidates)
    assert result.result.status == "ok"
    assert not result.candidate_moved
    assert result.result.depth_top_m == pytest.approx(depth, abs=0.03)
    assert (result.activity_id, result.path) == ("01.1", "PathT")
    row = script.candidate_row(result, {})
    assert row["search_k0"] >= row["box_k0"]
    assert row["apex_minus_search_k0"] > 2  # not a box-top peak (5-7 item 2)


def test_e2_clusters_and_support_filter(script) -> None:
    clusters = script.cluster_depths([0.50, 0.55, 0.62, 1.00], [1, 2, 3, 1], 0.10)
    assert [(round(c.depth_m, 3), c.n_lines) for c in clusters] == [(0.55, 3), (1.0, 1)]
    kept = script.supported(clusters, n_lines_in_activity=3, min_lines=2)
    assert [round(c.depth_m, 3) for c in kept] == [0.55]
    assert script.supported(clusters, n_lines_in_activity=1, min_lines=2) == clusters


def test_e3_hungarian_counts_missed_and_false_estimates(script) -> None:
    match = script.match_activity([0.5, 1.0, 1.6], [0.55, 1.05])
    assert len(match.pairs) == 2
    assert (match.false_est, match.missed) == (1, 0)
    assert sorted(round(e, 3) for _, _, e in match.pairs) == [-0.05, -0.05]
    swapped = script.match_activity([0.55, 1.05], [0.5, 1.0, 1.6])
    assert (swapped.missed, swapped.false_est) == (1, 0)


def test_e4_no_gate(script) -> None:
    match = script.match_activity([3.0], [0.5])
    ((_, _, error),) = match.pairs
    assert error == pytest.approx(2.5)
    summary = script.summarize({"a": match}, {"a": 1})
    assert summary.n_pass == 0


def test_e1b_candidate_without_tau_counts_as_zero_failed(script) -> None:
    _, radargram = _air_coupled_line()
    candidates = pd.DataFrame(
        {"trace": [10], "sample": [100.0], "tau_ns": [np.nan], "source": ["manual"]}
    )
    (result,) = script.run_line(radargram, candidates)
    assert result.result.status == "zero_failed"
    assert "candidate_tau_missing" in result.result.flags
    assert result.result.apex_trace is None  # no box, no fit
    assert result.depth("detected") is None
    assert not result.candidate_moved


def test_e5_baseline_is_reproducible_and_single_gt_range(script) -> None:
    n_est = {"a": 3, "b": 1}
    gt = {"a": [0.5, 1.5], "b": [1.0]}
    first = script.random_baseline(n_est, gt, trials=50, seed=0)
    second = script.random_baseline(n_est, gt, trials=50, seed=0)
    np.testing.assert_array_equal(first.pass_b_values, second.pass_b_values)
    assert first.pass_b_mean == second.pass_b_mean
    assert script.baseline_range([1.0], "activity") == pytest.approx((0.7, 1.3))
    assert script.baseline_range([0.1], "activity") == pytest.approx((0.0, 0.4))
    assert script.baseline_range([0.5, 1.5], "global") == (0.2, 2.0)


@pytest.mark.parametrize("trials", [1000, 200])
def test_e5_variant_main_and_aux_trial_counts_reproduce(script, trials) -> None:
    n_est = {"a": 3, "b": 1}
    gt = {"a": [0.5, 1.5], "b": [1.0]}
    first = script.random_baseline(n_est, gt, trials=trials, seed=0)
    second = script.random_baseline(n_est, gt, trials=trials, seed=0)
    assert len(first.pass_b_values) == trials
    np.testing.assert_array_equal(first.pass_b_values, second.pass_b_values)
    assert (first.mae_mean, first.mae_p025, first.mae_p975) == (
        second.mae_mean,
        second.mae_p025,
        second.mae_p975,
    )
    assert (first.pass_a_mean, first.pass_a_p025, first.pass_a_p975) == (
        second.pass_a_mean,
        second.pass_a_p025,
        second.pass_a_p975,
    )


def test_e5_variant_trial_counts_per_variant(script) -> None:
    args = _args(baseline_trials=1000, baseline_trials_aux=200)
    assert script.baseline_trials_for("main", args) == {
        "activity": 1000,
        "global": 200,
    }
    assert script.baseline_trials_for("a", args) == {"activity": 200}
    assert script.baseline_trials_for("b", args) == {"activity": 200}


def test_e5_variant_site_baseline_comes_from_the_same_trials(script) -> None:
    n_est = {"01.1": 2, "01.2": 1, "02.1": 1}
    gt = {"01.1": [0.5, 1.5], "01.2": [1.0], "02.1": [0.8]}
    draws = script.draw_baseline(n_est, gt, trials=200, seed=0)
    site = script.baseline_stats(draws, ["01.1", "01.2"])
    # Per-trial pass B of the subset recomputed from the stored counts.
    expected = draws.n_pass[:, :2].sum(axis=1) / 3
    np.testing.assert_allclose(site.pass_b_values, expected)
    overall = script.baseline_stats(draws)
    np.testing.assert_allclose(overall.pass_b_values, draws.n_pass.sum(axis=1) / 4)
    # The overall statistics equal the one-call baseline of the same seed.
    one_call = script.random_baseline(n_est, gt, trials=200, seed=0)
    np.testing.assert_array_equal(overall.pass_b_values, one_call.pass_b_values)
    assert script.baseline_stats(draws, ["99.9"]) is None


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    groundtruth = pd.DataFrame(
        {
            "activity_id": ["01.1", "01.1", "01.1", "01.1", "011.7", "011.7"],
            "trench_no": ["1", "1", "1", "1", "1", "2"],
            "seq": ["1", "2", "3", "4", "1", "1"],
            "depth_m": ["1.10", "", "0.80", "1.40", "0.60", "0.70"],
            "code": ["U", "S", "X", "W", "S", "S"],
        }
    )
    metadata = pd.DataFrame(
        {
            "activity_id": ["01.1", "01.1", "011.7"],
            "seq_meta": ["1", "4", "1"],
            "diameter_mm": ["200", "100", "300"],
            "정답표와 순서 일치": ["예", "예", "아니오"],
        }
    )
    return groundtruth, metadata


def test_e6_ground_truth_preparation(script) -> None:
    groundtruth, metadata = _tables()
    gt = script.build_ground_truth(groundtruth, metadata)
    assert [item.depth_m for item in gt["01.1"]] == [1.10, 1.40]  # blank and X dropped
    first = gt["01.1"][0]
    assert first.radius_known and first.radius_m == pytest.approx(0.1)
    assert first.depth_for("center") == pytest.approx(1.0)
    assert all(item.multi_trench for item in gt["011.7"])
    assert len(gt["011.7"]) == 2  # union of both trenches
    assert not any(item.radius_known for item in gt["011.7"])  # order does not match
    with_x = script.build_ground_truth(groundtruth, metadata, include_code_x=True)
    assert [item.code for item in with_x["01.1"]] == ["U", "X", "W"]


def _candidate(script, activity: str, path: str, depth: float, eps_r: float = 9.0):
    return script.CandidateResult(
        activity_id=activity,
        path=path,
        candidate_trace=0,
        candidate_sample=0.0,
        source="auto_v1",
        result=DepthResult(status=DepthStatus.OK, depth_top_m=depth, eps_r_used=eps_r),
        candidate_moved=False,
    )


def _args(**overrides) -> argparse.Namespace:
    values = {
        "cluster_gap_m": 0.10,
        "min_support_lines": 2,
        "baseline_trials": 30,
        "baseline_trials_aux": 10,
        "baseline_seed": 0,
    }
    return argparse.Namespace(**{**values, **overrides})


def _main_rows(rows, scope: str = "all") -> dict[str, dict[str, object]]:
    """Summary rows of (top, detected, main) keyed by gt set."""
    return {
        row["gt_set"]: row
        for row in rows
        if row["scope"] == scope
        and (row["assumption"], row["zero_variant"], row["variant"])
        == ("top", "detected", "main")
    }


def test_e6_variant_gt_sets_and_multi_trench(script) -> None:
    groundtruth, metadata = _tables()
    gt_sets = script.build_ground_truth_sets(groundtruth, metadata)
    n_x_rows = int(
        ((groundtruth["code"] == "X") & (groundtruth["depth_m"] != "")).sum()
    )
    counts = {name: sum(map(len, gt.values())) for name, gt in gt_sets.items()}
    assert counts["include_x"] - counts["exclude_x"] == n_x_rows == 1
    assert all(item.multi_trench for item in gt_sets["exclude_x"]["011.7"])

    candidates = [
        _candidate(script, activity, path, depth)
        for activity, depths in (("01.1", (1.10, 0.80)), ("011.7", (0.60,)))
        for path in ("P1", "P2")
        for depth in depths
    ]
    evaluation = script.evaluate_combinations(
        candidates, gt_sets, {"01.1": 2, "011.7": 2}, _args()
    )
    rows = _main_rows(evaluation.summary_rows)
    assert (rows["exclude_x"]["n_gt"], rows["include_x"]["n_gt"]) == (4, 5)
    assert rows["exclude_x"]["n_est"] == rows["include_x"]["n_est"] == 3
    assert {row["gt_set"] for row in evaluation.match_rows} == {
        "exclude_x",
        "include_x",
    }
    multi = {row["activity_id"]: row["multi_trench"] for row in evaluation.match_rows}
    assert multi == {"01.1": False, "011.7": True}


def test_e5_variant_auxiliary_baselines_report_pass_b_and_p_only(script) -> None:
    groundtruth, metadata = _tables()
    gt_sets = script.build_ground_truth_sets(groundtruth, metadata)
    candidates = [
        _candidate(script, "01.1", path, depth)
        for path in ("P1", "P2")
        for depth in (1.10, 0.80)
    ]
    evaluation = script.evaluate_combinations(candidates, gt_sets, {"01.1": 2}, _args())
    main = _main_rows(evaluation.summary_rows)["exclude_x"]
    for name in ("mae", "pass_a", "pass_b"):
        for stat in ("mean", "p025", "p975"):
            assert main[f"baseline_{name}_{stat}"] != ""
    global_columns = [key for key in main if key.startswith("baseline_global_")]
    assert global_columns == [
        "baseline_global_pass_b_mean",
        "baseline_global_pass_b_p025",
        "baseline_global_pass_b_p975",
        "baseline_global_p",
    ]
    assert all(main[key] != "" for key in global_columns)
    auxiliary = next(
        row
        for row in evaluation.summary_rows
        if row["scope"] == "all" and row["variant"] == "b"
    )
    assert auxiliary["baseline_mae_mean"] != ""
    assert all(auxiliary[key] == "" for key in global_columns)


def test_e8_site11_diagnostic_rescales_then_follows_the_main_path(script) -> None:
    groundtruth = pd.DataFrame(
        {
            "activity_id": ["011.1"],
            "trench_no": ["1"],
            "seq": ["1"],
            "depth_m": ["1.00"],
            "code": ["S"],
        }
    )
    metadata = pd.DataFrame(
        columns=["activity_id", "seq_meta", "diameter_mm", "정답표와 순서 일치"]
    )
    gt_sets = script.build_ground_truth_sets(groundtruth, metadata)
    # eps_used 18 -> factor sqrt(18 / 8) = 1.5; one stray depth on one line only.
    candidates = [
        _candidate(script, "011.1", "P1", 0.66, eps_r=18.0),
        _candidate(script, "011.1", "P2", 0.68, eps_r=18.0),
        _candidate(script, "011.1", "P2", 0.20, eps_r=18.0),
    ]
    lines = {"011.1": 2}
    rows, summaries = script.site11_diagnostic(candidates, gt_sets, lines, _args())

    main_path = [
        row
        for row in rows
        if row["variant"] == "main_path"
        and (row["gt_set"], row["assumption"]) == ("exclude_x", "top")
    ]
    (pair,) = main_path  # stray depth dropped by the support filter
    assert pair["est_depth_eps8_m"] == pytest.approx(1.5 * 0.67, abs=1e-4)
    assert pair["est_depth_m"] == pytest.approx(0.67, abs=1e-4)
    assert pair["error_eps8_m"] == pytest.approx(0.005, abs=1e-4)
    assert pair["within_30cm_eps8"]
    assert {row["variant"] for row in rows} == {"main_path", "unclustered"}
    assert summaries[("exclude_x", "top", "main_path")].n_est == 1
    assert summaries[("exclude_x", "top", "unclustered")].n_est == 3

    # The main metrics stay unscaled and carry no diagnostic variant.
    evaluation = script.evaluate_combinations(candidates, gt_sets, lines, _args())
    assert not {row["variant"] for row in evaluation.summary_rows} & {
        "main_path",
        "unclustered",
    }
    main_pairs = [
        row
        for row in evaluation.match_rows
        if (row["gt_set"], row["assumption"], row["zero_variant"], row["variant"])
        == ("exclude_x", "top", "detected", "main")
    ]
    assert [row["est_depth_m"] for row in main_pairs] == [pytest.approx(0.67)]


def test_e7_pass_rates_include_the_30_cm_boundary(script) -> None:
    match = script.MatchResult(
        pairs=((0.1, 0, 0.1), (0.3, 1, 0.3), (0.31, 2, 0.31)), missed=1, false_est=0
    )
    summary = script.summarize({"a": match}, {"a": 4})
    assert summary.pass_a == pytest.approx(2 / 3)
    assert summary.pass_b == pytest.approx(2 / 4)
