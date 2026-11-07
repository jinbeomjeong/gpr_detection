"""Unit tests of ``scripts/eval_depth_4tu.py``.

spec_eval_4tu_v1 E1-E8, B1-B13, C1-C20 and the v1.6 gate tests G1-G11.

Analytic B-scans and small in-memory tables only; no 4TU data is read.
"""

import argparse
import ast
import dataclasses
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gpr_detection.depth.estimate import prepare_depth_inputs
from gpr_detection.depth.refraction import reflection_times
from gpr_detection.depth.result import DepthResult, DepthStatus
from gpr_detection.io.radargram import Radargram
from synthetic_bscan import (
    DT_NS,
    T_SRC_NS,
    BScanSpec,
    cylinder_times,
    make_radargram,
)


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


E2_DEPTHS, E2_LINES = [0.50, 0.55, 0.62, 1.00], [1, 2, 3, 1]


def test_e2_clusters_and_support_filter(script) -> None:
    # v1.3 complete linkage: 0.50-0.62 is 0.12 wide, so it cannot be one cluster.
    clusters = script.cluster_depths(E2_DEPTHS, E2_LINES, width_m=0.10)
    assert [(round(c.depth_m, 3), c.n_lines) for c in clusters] == [
        (0.525, 2),
        (0.62, 1),
        (1.0, 1),
    ]
    kept = script.supported(clusters, n_lines_in_activity=3, min_lines=2)
    assert [round(c.depth_m, 3) for c in kept] == [0.525]
    assert script.supported(clusters, n_lines_in_activity=1, min_lines=2) == clusters


def test_e2_v12_expectations_hold_for_single_linkage(script) -> None:
    clusters = script.cluster_depths(E2_DEPTHS, E2_LINES, 0.10, linkage="single")
    assert [(round(c.depth_m, 3), c.n_lines) for c in clusters] == [(0.55, 3), (1.0, 1)]
    kept = script.supported(clusters, n_lines_in_activity=3, min_lines=2)
    assert [round(c.depth_m, 3) for c in kept] == [0.55]


# ----------------------------------------------------------------- B1-B13 (spec 9-2-1)

CLUSTER_WIDTH_M = 0.10
WIDTH_BOUND_M = CLUSTER_WIDTH_M + 1e-9
"""W plus the edge tolerance of spec 4-3-1."""
DEPTH_ABS_TOL = 1e-9


def _assert_clusters(clusters, expected) -> None:
    """Compare ``(depth, n_lines, n_apex_candidates[, span_m])`` per cluster.

    Expected tuples of three entries skip ``span_m``. Depths and spans use
    ``abs=1e-9``; counts are exact.
    """
    assert len(clusters) == len(expected), [
        (c.depth_m, c.n_lines, c.n_apex_candidates, c.span_m) for c in clusters
    ]
    for cluster, (depth, n_lines, n_members, *span) in zip(
        clusters, expected, strict=True
    ):
        assert cluster.depth_m == pytest.approx(depth, abs=DEPTH_ABS_TOL)
        assert (cluster.n_lines, cluster.n_apex_candidates) == (n_lines, n_members)
        if span:
            assert cluster.span_m == pytest.approx(span[0], abs=DEPTH_ABS_TOL)


def _signature(clusters, depths, lines) -> list[tuple]:
    """Order-free outcome: depth, counts, span and the (depth, line) multiset."""
    return [
        (
            c.depth_m,
            c.n_lines,
            c.n_apex_candidates,
            c.span_m,
            tuple(sorted((depths[i], str(lines[i])) for i in c.member_indices)),
        )
        for c in clusters
    ]


def _member_depths(cluster, depths) -> list[float]:
    return [depths[i] for i in cluster.member_indices]


def _chain_input() -> tuple[list[float], list[str]]:
    """B1: 21 depths 0.30..0.90 m in 0.03 m steps over 6 lines."""
    depths = [round(0.30 + 0.03 * k, 2) for k in range(21)]
    return depths, [f"L{k % 6 + 1}" for k in range(21)]


def _dense_input() -> tuple[list[float], list[str]]:
    """B1b: 48 depths 0.40..1.34 m in 0.02 m steps, 6 lines x 8 (4TU-like)."""
    depths = [round(0.40 + 0.02 * k, 2) for k in range(48)]
    return depths, [f"L{k % 6 + 1}" for k in range(48)]


def _random_inputs():
    """B12: 300 random activities from ``default_rng(0)`` with one permutation each."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        size = int(rng.integers(1, 60))
        depths = [float(d) for d in np.round(rng.uniform(0.2, 2.0, size), 3)]
        lines = [f"L{int(x)}" for x in rng.integers(1, 8, size)]
        permutation = rng.permutation(size)
        yield depths, lines, permutation


def test_b1_chain_is_reproduced_by_single_and_cut_by_complete(script) -> None:
    depths, lines = _chain_input()
    (chain,) = script.cluster_depths(depths, lines, linkage="single")
    _assert_clusters([chain], [(0.60, 6, 21)])
    assert chain.span_m == pytest.approx(0.60) and chain.span_m > CLUSTER_WIDTH_M
    clusters = script.cluster_depths(depths, lines)
    _assert_clusters(
        clusters,
        [
            (0.345, 4, 4, 0.09),
            (0.465, 4, 4, 0.09),
            (0.585, 4, 4, 0.09),
            (0.705, 4, 4, 0.09),
            (0.795, 2, 2, 0.03),
            (0.87, 3, 3, 0.06),
        ],
    )
    assert all(c.span_m <= WIDTH_BOUND_M for c in clusters)


def test_b1b_dense_row_splits_into_twelve_clusters(script) -> None:
    depths, lines = _dense_input()
    _assert_clusters(
        script.cluster_depths(depths, lines, linkage="single"), [(0.87, 6, 48)]
    )
    clusters = script.cluster_depths(depths, lines)
    expected_depths = [0.43 + 0.08 * k for k in range(12)]
    _assert_clusters(clusters, [(depth, 4, 4, 0.06) for depth in expected_depths])
    assert clusters[-1].depth_m == pytest.approx(1.31, abs=DEPTH_ABS_TOL)


def test_b2_every_member_pair_within_the_width(script) -> None:
    inputs = [_chain_input(), _dense_input()]
    inputs += [(depths, lines) for depths, lines, _ in _random_inputs()]
    for depths, lines in inputs:
        for cluster in script.cluster_depths(depths, lines):
            assert cluster.span_m <= WIDTH_BOUND_M
            members = _member_depths(cluster, depths)
            for first, second in itertools.combinations(members, 2):
                assert abs(first - second) <= WIDTH_BOUND_M


def test_b3_ties_duplicates_and_input_order(script) -> None:
    depths = [0.50, 0.55, 0.60, 0.60, 0.65, 0.70, 0.50]
    lines = ["L1", "L2", "L3", "L4", "L1", "L2", "L3"]
    clusters = script.cluster_depths(depths, lines)
    _assert_clusters(clusters, [(0.50, 3, 3, 0.05), (0.625, 4, 4, 0.10)])
    reference = _signature(clusters, depths, lines)
    rng = np.random.default_rng(0)
    for _ in range(50):
        permutation = rng.permutation(len(depths))
        shuffled_depths = [depths[i] for i in permutation]
        shuffled_lines = [lines[i] for i in permutation]
        shuffled = script.cluster_depths(shuffled_depths, shuffled_lines)
        assert _signature(shuffled, shuffled_depths, shuffled_lines) == reference


@pytest.mark.parametrize(
    ("pair", "n_clusters"),
    [
        ([0.5, 0.6], 1),  # float width 0.09999999999999998
        ([0.7, 0.8], 1),  # float width 0.10000000000000009, inside the tolerance
        ([1.1, 1.2], 1),  # float width 0.09999999999999987
        ([0.5, 0.6 + 2e-9], 2),  # W + 2e-9 m splits
    ],
)
def test_b4_width_boundary_is_included(script, pair, n_clusters) -> None:
    assert len(script.cluster_depths(pair, ["L1", "L2"])) == n_clusters


@pytest.mark.parametrize(
    ("pair", "n_clusters"),
    [
        ([0.0, 0.1], 1),  # float gap exactly 0.1: v1.2 cuts only on ``>``
        ([0.7, 0.8], 2),  # float gap 0.10000000000000009: no edge tolerance
    ],
)
def test_b4_single_linkage_boundary_has_no_tolerance(script, pair, n_clusters) -> None:
    clusters = script.cluster_depths(pair, ["L1", "L2"], linkage="single")
    assert len(clusters) == n_clusters


BRIDGE_PIPE_A = [0.59, 0.60, 0.61, 0.60]
BRIDGE_PIPE_B = [0.72, 0.73, 0.71, 0.72]
BRIDGE_LINES = ["L1", "L2", "L3", "L4", "L1", "L2", "L3", "L4", "L2"]


def test_b5_bridge_joins_the_nearer_pipe(script) -> None:
    depths = [*BRIDGE_PIPE_A, *BRIDGE_PIPE_B, 0.655]
    _assert_clusters(
        script.cluster_depths(depths, BRIDGE_LINES, linkage="single"),
        [(0.655, 4, 9)],
    )
    _assert_clusters(
        script.cluster_depths(depths, BRIDGE_LINES),
        [(0.60, 4, 5, 0.065), (0.72, 4, 4, 0.02)],
    )


def test_b5t_bridge_tie_goes_to_the_shallower_pair(script) -> None:
    depths = [*BRIDGE_PIPE_A, *BRIDGE_PIPE_B, 0.66]  # 0.07 m to both pipes
    _assert_clusters(
        script.cluster_depths(depths, BRIDGE_LINES),
        [(0.60, 4, 5, 0.07), (0.72, 4, 4, 0.02)],
    )


def test_b6_two_pipes_within_the_width_merge(script) -> None:
    depths = [0.60, 0.61, 0.60, 0.66, 0.67, 0.66]
    lines = ["L1", "L2", "L3", "L1", "L2", "L3"]
    clusters = script.cluster_depths(depths, lines)
    _assert_clusters(clusters, [(0.635, 3, 6, 0.07)])
    match = script.match_activity([c.depth_m for c in clusters], [0.60, 0.66])
    assert (len(match.pairs), match.missed) == (1, 1)  # known limitation (4-3 item 5)


def test_b7_even_median_and_repeated_line(script) -> None:
    clusters = script.cluster_depths([0.50, 0.52, 0.55, 0.58], ["L1", "L1", "L2", "L3"])
    _assert_clusters(clusters, [(0.535, 3, 4, 0.08)])


def test_b8_one_pipe_split_keeps_both_halves(script) -> None:
    depths, lines = [0.50, 0.56, 0.62, 0.68], ["L1", "L2", "L3", "L4"]
    clusters = script.cluster_depths(depths, lines)
    _assert_clusters(clusters, [(0.53, 2, 2, 0.06), (0.65, 2, 2, 0.06)])
    kept = script.supported(clusters, n_lines_in_activity=4, min_lines=2)
    assert kept == clusters  # duplicate estimate (4-3-2)
    _assert_clusters(
        script.cluster_depths(depths, lines, linkage="single"), [(0.59, 4, 4)]
    )


def test_b9_support_filter_interaction(script) -> None:
    one_line_depths = _dense_input()[0][:12]
    one_line = script.cluster_depths(one_line_depths, ["L1"] * 12)
    _assert_clusters(one_line, [(0.43, 1, 4), (0.51, 1, 4), (0.59, 1, 4)])
    assert script.supported(one_line, n_lines_in_activity=1, min_lines=2) == one_line

    mixed = script.cluster_depths(
        [0.50, 0.52, 0.80, 1.20, 1.21], ["L1", "L2", "L1", "L1", "L3"]
    )
    _assert_clusters(
        mixed, [(0.51, 2, 2, 0.02), (0.80, 1, 1, 0.0), (1.205, 2, 2, 0.01)]
    )
    kept = script.supported(mixed, n_lines_in_activity=3, min_lines=2)
    assert [c.depth_m for c in kept] == pytest.approx([0.51, 1.205], abs=DEPTH_ABS_TOL)


def test_b10_e2_input(script) -> None:
    _assert_clusters(
        script.cluster_depths(E2_DEPTHS, E2_LINES),
        [(0.525, 2, 2, 0.05), (0.62, 1, 1, 0.0), (1.0, 1, 1, 0.0)],
    )


def test_b11_empty_single_and_invalid_inputs(script) -> None:
    assert script.cluster_depths([], []) == []
    _assert_clusters(script.cluster_depths([0.8], ["L1"]), [(0.8, 1, 1, 0.0)])
    with pytest.raises(ValueError):
        script.cluster_depths([0.5, 0.6], ["L1"])
    with pytest.raises(ValueError):
        script.cluster_depths([0.5, float("nan")], ["L1", "L2"])
    with pytest.raises(ValueError):
        script.cluster_depths([0.5, float("inf")], ["L1", "L2"])
    with pytest.raises(ValueError):
        script.cluster_depths([0.5, 0.6], ["L1", "L2"], width_m=0)
    with pytest.raises(ValueError):
        script.cluster_depths([0.5, 0.6], ["L1", "L2"], linkage="average")


def test_b12_properties_on_random_inputs(script) -> None:
    for depths, lines, permutation in _random_inputs():
        clusters = script.cluster_depths(depths, lines)
        # P1 width bound
        assert all(c.span_m <= WIDTH_BOUND_M for c in clusters)
        # P2 stopping rule: no adjacent pair could still merge
        for shallow, deep in itertools.pairwise(clusters):
            merged_width_m = max(_member_depths(deep, depths)) - min(
                _member_depths(shallow, depths)
            )
            assert merged_width_m > WIDTH_BOUND_M
        # P3 partition of the input
        members = [i for c in clusters for i in c.member_indices]
        assert sum(c.n_apex_candidates for c in clusters) == len(depths)
        assert sorted(members) == list(range(len(depths)))
        # P4 input order does not matter
        shuffled_depths = [depths[i] for i in permutation]
        shuffled_lines = [lines[i] for i in permutation]
        shuffled = script.cluster_depths(shuffled_depths, shuffled_lines)
        assert _signature(shuffled, shuffled_depths, shuffled_lines) == _signature(
            clusters, depths, lines
        )


CLUSTER_CSV_COLUMNS = [
    "activity_id",
    "site",
    "zero_variant",
    "cluster_id",
    "depth_m",
    "n_lines",
    "n_apex_candidates",
    "kept_by_support",
    "span_m",
    "linkage",
    "width_m",
]
"""Column order of the clusters csv (spec 8)."""
B13_LINES_PER_ACTIVITY = {"01.1": 6, "02.1": 1}


@dataclass(frozen=True)
class _DuckCandidate:
    """Minimal candidate: only what ``estimates_by_activity`` reads."""

    activity_id: str
    path: str
    depth_m: float

    def depth(self, zero_variant: str) -> float | None:
        return self.depth_m


def _b13_candidates() -> list[_DuckCandidate]:
    depths, lines = _chain_input()
    candidates = [
        _DuckCandidate("01.1", line, depth)
        for depth, line in zip(depths, lines, strict=True)
    ]
    return [*candidates, _DuckCandidate("02.1", "M1", 0.8)]


def _b13_estimates(script, linkage: str):
    return script.estimates_by_activity(
        _b13_candidates(),
        "detected",
        "main",
        B13_LINES_PER_ACTIVITY,
        width_m=0.10,
        linkage=linkage,
        min_lines=2,
    )


def _line_starting_with(lines: list[str], prefix: str) -> str:
    (line,) = [line for line in lines if line.startswith(prefix)]
    return line


def test_b13_cluster_rows_and_estimate_counts(script) -> None:
    estimates, rows = _b13_estimates(script, "complete")
    assert {a: len(v) for a, v in estimates.items()} == {"01.1": 6, "02.1": 1}
    assert len(rows) == 7
    assert all(list(row) == CLUSTER_CSV_COLUMNS for row in rows)
    assert (rows[0]["span_m"], rows[0]["linkage"], rows[0]["width_m"]) == (
        0.09,
        "complete",
        0.1,
    )
    estimates, rows = _b13_estimates(script, "single")
    assert {a: len(v) for a, v in estimates.items()} == {"01.1": 1, "02.1": 1}
    assert (rows[0]["span_m"], rows[0]["linkage"]) == (0.6, "single")


def test_b13_md_cluster_diagnostics(script) -> None:
    estimates, rows = _b13_estimates(script, "complete")
    args = _args()
    candidates = _b13_candidates()
    comparison = script.compare_with_single_linkage(
        candidates, B13_LINES_PER_ACTIVITY, estimates, frozenset(estimates), args
    )
    lines = script._cluster_diagnostic_lines(rows, args, comparison)
    assert "m: 0개" in _line_starting_with(lines, "- `span_m` > `width_m`")
    assert "7/7" in _line_starting_with(lines, "- 지지 필터 통과")
    assert _line_starting_with(lines, "  - 전체 활동").endswith(
        "2개: complete 7 / single 2 / 차 +5 (활동 수: 늘어남 1, 같음 1, 줄어듦 0)"
    )
    assert _line_starting_with(lines, "  - 정답 활동").endswith(
        "2개: complete 7 / single 2 / 차 +5 (활동 수: 늘어남 1, 같음 1, 줄어듦 0)"
    )

    only_01 = script.compare_with_single_linkage(
        candidates, B13_LINES_PER_ACTIVITY, estimates, frozenset({"01.1"}), args
    )
    lines = script._cluster_diagnostic_lines(rows, args, only_01)
    assert _line_starting_with(lines, "  - 정답 활동").endswith(
        "1개: complete 6 / single 1 / 차 +5 (활동 수: 늘어남 1, 같음 0, 줄어듦 0)"
    )

    single_args = _args(cluster_linkage="single")
    _, single_rows = _b13_estimates(script, "single")
    assert (
        script.compare_with_single_linkage(
            candidates, B13_LINES_PER_ACTIVITY, estimates, frozenset(), single_args
        )
        is None
    )
    lines = script._cluster_diagnostic_lines(single_rows, single_args, None)
    assert "m: 1개" in _line_starting_with(lines, "- `span_m` > `width_m`")
    assert "해당 없음" in _line_starting_with(lines, "- v1.2 대비 추정 수")


def test_b13_cli_defaults_and_rejected_options(script) -> None:
    args = script.parse_args([])
    assert (args.cluster_width_m, args.cluster_linkage) == (0.1, "complete")
    for argv in (["--cluster-gap-m", "0.1"], ["--cluster-linkage", "average"]):
        with pytest.raises(SystemExit) as raised:
            script.parse_args(argv)
        assert raised.value.code == 2


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
    # v1.5: bd and noqc are auxiliary variants, reported only with the QC on.
    assert script.VARIANTS == ("main", "a", "b", "bd", "noqc")
    assert script.baseline_trials_for("bd", args) == {"activity": 200}
    assert script.baseline_trials_for("noqc", args) == {"activity": 200}
    assert script.variants_for(_args(apex_qc="off")) == ("main", "a", "b")


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
        "cluster_width_m": 0.10,
        "cluster_linkage": "complete",
        "min_support_lines": 2,
        "baseline_trials": 30,
        "baseline_trials_aux": 10,
        "baseline_seed": 0,
        "apex_qc": "on",
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
    # The v1.6 gate columns share the prefix; G8 checks them.
    global_columns = [
        key
        for key in main
        if key.startswith("baseline_global_")
        and not key.startswith("baseline_global_gate_")
    ]
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
    gt_sets = _site11_gt_sets(script)
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


def _site11_gt_sets(script):
    """E8 ground truth: one pipe at 1.00 m in activity 011.1."""
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
    return script.build_ground_truth_sets(groundtruth, metadata)


def _c15_candidates(script) -> list:
    """E8 candidates (no QC, kept) plus a curvature ``out`` and a record-end one.

    Without the QC the two extra depths 0.40 (P1) and 0.41 (P2) rescale to
    0.60 / 0.615 m, one cluster supported by both lines.
    """
    e8 = [
        _candidate(script, "011.1", path, depth, eps_r=18.0)
        for path, depth in (("P1", 0.66), ("P2", 0.68), ("P2", 0.20))
    ]
    extra = [
        ("P1", 0.40, _qc(script, "out")),
        ("P2", 0.41, _qc(script, "in", at_record_end=True)),
    ]
    excluded = [
        dataclasses.replace(_candidate(script, "011.1", path, depth, eps_r=18.0), qc=qc)
        for path, depth, qc in extra
    ]
    return [*e8, *excluded]


@pytest.mark.parametrize(
    ("apex_qc", "n_main_path", "n_unclustered"),
    [("on", 1, 3), ("off", 2, 5)],
)
def test_c15_site11_diagnostic_drops_qc_excluded_candidates(
    script, apex_qc, n_main_path, n_unclustered
) -> None:
    gt_sets = _site11_gt_sets(script)
    candidates = _c15_candidates(script)
    lines = {"011.1": 2}
    args = _args(apex_qc=apex_qc)
    rows, summaries = script.site11_diagnostic(candidates, gt_sets, lines, args)
    assert summaries[("exclude_x", "top", "main_path")].n_est == n_main_path
    assert summaries[("exclude_x", "top", "unclustered")].n_est == n_unclustered
    if apex_qc == "off":
        return
    (pair,) = [
        row
        for row in rows
        if row["variant"] == "main_path"
        and (row["gt_set"], row["assumption"]) == ("exclude_x", "top")
    ]
    assert pair["est_depth_m"] == pytest.approx(0.67, abs=1e-4)  # same as E8
    evaluation = script.evaluate_combinations(candidates, gt_sets, lines, args)
    assert evaluation.estimate_counts["main"]["011.1"] == n_main_path


def _c15b_candidates(script) -> list:
    """C15 plus an unknown-curvature 0.20 m on P1 (eps_r 18).

    Kept, it would pair with the E8 stray 0.20 m on P2 (rescaled 0.30 m).
    """
    unknown = dataclasses.replace(
        _candidate(script, "011.1", "P1", 0.20, eps_r=18.0),
        qc=_qc(script, "unknown"),
    )
    return [*_c15_candidates(script), unknown]


@pytest.mark.parametrize(
    ("apex_qc", "n_main_path", "n_unclustered"),
    [("on", 1, 3), ("off", 3, 6)],
)
def test_c15b_site11_diagnostic_drops_unknown_curvature(
    script, apex_qc, n_main_path, n_unclustered
) -> None:
    """C15b (v1.5): the site 11 diagnostic uses the main rule B+D+E."""
    gt_sets = _site11_gt_sets(script)
    candidates = _c15b_candidates(script)
    lines = {"011.1": 2}
    args = _args(apex_qc=apex_qc)
    rows, summaries = script.site11_diagnostic(candidates, gt_sets, lines, args)
    assert summaries[("exclude_x", "top", "main_path")].n_est == n_main_path
    assert summaries[("exclude_x", "top", "unclustered")].n_est == n_unclustered
    if apex_qc == "off":
        return
    (pair,) = [
        row
        for row in rows
        if row["variant"] == "main_path"
        and (row["gt_set"], row["assumption"]) == ("exclude_x", "top")
    ]
    assert pair["est_depth_m"] == pytest.approx(0.67, abs=1e-4)
    evaluation = script.evaluate_combinations(candidates, gt_sets, lines, args)
    counts = evaluation.estimate_counts
    # bd keeps the unknown 0.20 (P1), which supports the 0.20 (P2) stray.
    assert counts["main"]["011.1"] == 1
    assert counts["bd"]["011.1"] == 2
    assert counts["noqc"]["011.1"] == 3
    # Added by the implementer: main != bd here, so the md difference sign shows.
    md_lines = script._qc_diagnostic_lines(candidates, evaluation, args)
    assert _line_starting_with(md_lines, "  - 전체 활동").endswith(
        "1개: main 1 / bd 2 / noqc 3 / main-bd -1 / main-noqc -2"
    )


def test_e7_pass_rates_include_the_30_cm_boundary(script) -> None:
    match = script.MatchResult(
        pairs=((0.1, 0, 0.1), (0.3, 1, 0.3), (0.31, 2, 0.31)), missed=1, false_est=0
    )
    summary = script.summarize({"a": match}, {"a": 4})
    assert summary.pass_a == pytest.approx(2 / 3)
    assert summary.pass_b == pytest.approx(2 / 4)


# ----------------------------------------------------------------- C1-C14 (spec 9-2-2)
#
# Expected values are the prototype output of analysis/processing/m29c_qc_proto.py
# (docs/processing/m29c_qc_ctests.csv) and m29c_qc_pure_checks.py, 2026-11-07.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QC_CONSTANTS_JSON = PROJECT_ROOT / "docs" / "processing" / "m29c_qc_constants.json"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "eval_depth_4tu.py"

QC_F_C_HZ = 500e6
QC_DT_NS = 0.098
QC_RECORD_SAMPLES = 512
QC_LONG_RECORD_SAMPLES = 768
ZERO_BAND_SCALE = 10.0
"""Zero band 20 dB above the pipe (direct 30, surface -20): an assumption (9-2-2)."""
R_ABS_TOL = 0.01
SEAM_SCORE_ABS_TOL = 0.05
PURE_ABS_TOL = 1e-4
QC_FLAG_NAMES = (
    "curvature_v_out",
    "curvature_v_unknown",
    "apex_at_record_end",
    "apex_on_trace_seam",
)


def _qc_spec(eps_r: float = 9.0, n_samples: int = QC_RECORD_SAMPLES) -> BScanSpec:
    """Common generator of 9-2-2: 500 MHz, 401 traces, merged zero band."""
    return BScanSpec(
        f_c_hz=QC_F_C_HZ,
        dt_ns=QC_DT_NS,
        dx_m=0.02,
        n_traces=401,
        snr_db=20.0,
        air_gap_m=0.001,
        offset_m=0.0,
        direct_amplitude=3.0 * ZERO_BAND_SCALE,
        surface_amplitude=-2.0 * ZERO_BAND_SCALE,
        eps_r=eps_r,
        n_samples=n_samples,
    )


def _straight_ray_spec(
    eps_r: float,
    d_top_m: float,
    radius_m: float = 0.0,
    n_samples: int = QC_LONG_RECORD_SAMPLES,
) -> BScanSpec:
    """Straight-ray pipe (effective air gap 0) under the merged zero band."""
    spec = _qc_spec(eps_r, n_samples)
    spec.d_top_m, spec.radius_m = d_top_m, radius_m
    spec.pipe_times_ns = cylinder_times(
        spec.x_m, spec.x0_m, d_top_m, spec.velocity, radius_m
    )
    return spec


def _true_apex(spec: BScanSpec) -> tuple[int, float]:
    trace = spec.n_traces // 2
    return trace, (T_SRC_NS + float(spec.pipe_times_ns[trace])) / spec.dt_ns


def _radargram(spec: BScanSpec) -> Radargram:
    return make_radargram(
        spec, air_gap="air", eps_r_meta=spec.eps_r, line_id="01.1/PathC"
    )


def _candidate_table(radargram: Radargram, trace: int, sample: float) -> pd.DataFrame:
    """One auto candidate; ``tau_ns`` from the detected zero like the 4TU csv."""
    t_zero_ns = prepare_depth_inputs(radargram).zero.t_zero_ns
    return pd.DataFrame(
        {
            "trace": [trace],
            "sample": [sample],
            "tau_ns": [sample * radargram.dt_ns - t_zero_ns],
            "source": ["auto_v1"],
        }
    )


def _run_one(script, radargram: Radargram, trace: int, sample: float, apex_qc=True):
    (candidate,) = script.run_line(
        radargram, _candidate_table(radargram, trace, sample), apex_qc=apex_qc
    )
    return candidate


def _assert_depth_path_untouched(candidate, radargram: Radargram) -> None:
    """C12: meta_fixed depth, no QC flag in ``DepthResult.flags``, air kept."""
    assert candidate.result.v_rule == "meta_fixed"
    assert not set(candidate.result.flags) & set(QC_FLAG_NAMES)
    assert radargram.air_gap == "air"


def _assert_same_depth_result(first: DepthResult, second: DepthResult) -> None:
    """Field-wise equality; ``repr`` makes NaN fields compare equal (C11)."""
    for item in dataclasses.fields(DepthResult):
        assert repr(getattr(first, item.name)) == repr(getattr(second, item.name)), (
            item.name
        )


def test_qc_constants_match_the_locked_json(script) -> None:
    locked = json.loads(QC_CONSTANTS_JSON.read_text(encoding="utf-8"))
    for name in (
        "QC_R_LO",
        "QC_R_HI",
        "QC_R_TOL",
        "QC_UNSTABLE_SIGMA_REL",
        "QC_UNSTABLE_INLIER",
        "QC_BOUND_REL",
        "QC_RADIUS_M",
        "RECORD_END_PERIODS",
        "SEAM_TIME_PERIODS",
        "SEAM_AMP_RATIO",
        "SEAM_NOISE_K",
        "SEAM_ZERO_HALF_PERIODS",
        "SEAM_LAG_HALF_PERIODS",
        "SEAM_REF_HALF_M",
        "SEAM_APEX_HALF_M",
    ):
        assert getattr(script, name) == locked[name], name


# C1: (eps_r, R, d) -> r of the prototype; every case in, no flag, depth +-0.01 m.
C1_EXPECTED_R = {
    (9.0, 0.0): (0.9969, 0.9970, 0.9980, 0.9981, 0.9988, 0.9987),
    (16.0, 0.0): (0.9973, 0.9979, 0.9985, 0.9983, 0.9984, 0.9976),
    (9.0, 0.1): (1.0963, 1.0823, 1.0637, 1.0488, 1.0411, 1.0297),
    (16.0, 0.1): (1.0981, 1.0833, 1.0656, 1.0508, 1.0411, 1.0289),
}
C1_DEPTHS_M = (0.2, 0.3, 0.5, 0.75, 1.0, 1.5)
C1_CASES = [
    (eps_r, radius_m, depth_m, expected_r)
    for (eps_r, radius_m), ratios in C1_EXPECTED_R.items()
    for depth_m, expected_r in zip(C1_DEPTHS_M, ratios, strict=True)
]


@pytest.mark.parametrize(("eps_r", "radius_m", "depth_m", "expected_r"), C1_CASES)
def test_c1_c12_c14_true_apex_is_in(
    script, eps_r, radius_m, depth_m, expected_r
) -> None:
    spec = _straight_ray_spec(eps_r, depth_m, radius_m)
    radargram = _radargram(spec)
    trace, sample = _true_apex(spec)
    candidate = _run_one(script, radargram, trace, sample)
    assert candidate.result.status == "ok"
    assert candidate.result.depth_top_m == pytest.approx(depth_m, abs=0.01)
    curvature = candidate.qc.curvature
    assert (curvature.verdict, curvature.reason) == ("in", "")
    assert curvature.r == pytest.approx(expected_r, abs=R_ABS_TOL)
    assert script.QC_R_LO <= curvature.r <= script.QC_R_HI
    assert candidate.qc.flags == ()
    assert not candidate.qc.excluded
    _assert_depth_path_untouched(candidate, radargram)
    # C14: the QC picks carry the depth result's apex.
    inputs = prepare_depth_inputs(radargram)
    box = script.candidate_box(
        radargram, trace, sample, sample * QC_DT_NS - inputs.zero.t_zero_ns
    )
    picks = script.extract_picks(
        inputs.branch.background_removed,
        box,
        inputs.zero,
        dt_ns=radargram.dt_ns,
        dx_m=radargram.dx_m,
        f_c_hz=radargram.f_c_hz,
        apex_hint=script.ApexHint(trace=trace, sample=sample),
        period_px=inputs.branch.period_px,
    )
    assert (picks.apex.trace, picks.apex.k) == (
        candidate.result.apex_trace,
        candidate.result.apex_sample,
    )


def _refraction_spec(air_gap_m: float, radius_m: float, depth_m: float) -> BScanSpec:
    """E1 generator (amplitudes 3 / -2, offset 0.1 m, refraction times)."""
    spec = BScanSpec(
        f_c_hz=QC_F_C_HZ,
        eps_r=9.0,
        d_top_m=depth_m,
        radius_m=radius_m,
        air_gap_m=air_gap_m,
        offset_m=0.1,
        n_traces=401,
        n_samples=QC_LONG_RECORD_SAMPLES,
        dt_ns=QC_DT_NS,
    )
    spec.pipe_times_ns = reflection_times(
        spec.x_m,
        offset_m=0.1,
        h=air_gap_m,
        x0_m=spec.x0_m,
        d_top_m=depth_m,
        radius_m=radius_m,
        v=spec.velocity,
    )
    return spec


C1B_CASES = [
    (0.02, 0.0, 0.3, "out", 1.8774),
    (0.02, 0.1, 0.3, "out", 1.8876),
    (0.05, 0.0, 0.3, "out", 1.8818),
    (0.05, 0.1, 0.3, "out", 1.8949),
    (0.02, 0.0, 0.6, "in", 1.4767),
    (0.02, 0.1, 0.6, "in", 1.4880),
    (0.05, 0.0, 0.6, "in", 1.4886),
    (0.05, 0.1, 0.6, "in", 1.5054),
    (0.02, 0.0, 1.0, "in", 1.2370),
    (0.02, 0.1, 1.0, "in", 1.2530),
    (0.05, 0.0, 1.0, "in", 1.2573),
    (0.05, 0.1, 1.0, "in", 1.2755),
]


@pytest.mark.parametrize(
    ("air_gap_m", "radius_m", "depth_m", "verdict", "expected_r"), C1B_CASES
)
def test_c1b_air_gap_limit_of_the_upper_band_edge(
    script, air_gap_m, radius_m, depth_m, verdict, expected_r
) -> None:
    spec = _refraction_spec(air_gap_m, radius_m, depth_m)
    radargram = make_radargram(spec, eps_r_meta=9.0, line_id="01.1/PathC1b")
    candidate = _run_one(script, radargram, *_true_apex(spec))
    assert candidate.result.status == "ok"  # the depth does not depend on the QC
    curvature = candidate.qc.curvature
    assert curvature.verdict == verdict
    assert curvature.reason == ("r_above" if verdict == "out" else "")
    assert curvature.r == pytest.approx(expected_r, abs=R_ABS_TOL)


def _layer_line(times_ns: np.ndarray, amplitude: np.ndarray) -> Radargram:
    """C2-C4 background: a pipe at d 1.4 m plus one extra event."""
    spec = _straight_ray_spec(9.0, 1.4, n_samples=QC_RECORD_SAMPLES)
    spec.extra_events.append((times_ns, amplitude))
    return _radargram(spec)


def _x_and_mask(half_width_m: float) -> tuple[np.ndarray, float, np.ndarray]:
    spec = _qc_spec()
    x_m, x0_m = spec.x_m, spec.x0_m
    return x_m, x0_m, (np.abs(x_m - x0_m) <= half_width_m).astype(float)


LAYER_TIME_NS = 9.0
CENTRE_TRACE = 200


# v1.9 T2 (spec_depth 5-3): amplitude 0.5 was 0.9524 with the v1.8 reverse rule
# (wings 5/15); the median reference now follows the layer to 47/46 picks.
@pytest.mark.parametrize(("amplitude", "inlier_frac"), [(2.0, 0.2755), (0.5, 0.3191)])
def test_c2_c12_flat_layer_is_out_at_the_upper_v_bound(
    script, amplitude, inlier_frac
) -> None:
    x_m, _, inside = _x_and_mask(1.0)
    radargram = _layer_line(np.full(len(x_m), LAYER_TIME_NS), amplitude * inside)
    sample = (T_SRC_NS + LAYER_TIME_NS) / QC_DT_NS
    candidate = _run_one(script, radargram, CENTRE_TRACE, sample)
    curvature = candidate.qc.curvature
    # Order: the v bound (row 6) is checked before the inlier rule (row 8).
    assert (curvature.verdict, curvature.reason) == ("out", "v_at_upper_bound")
    assert curvature.r == pytest.approx(3.0001, abs=R_ABS_TOL)
    assert curvature.inlier_frac == pytest.approx(inlier_frac, abs=PURE_ABS_TOL)
    assert candidate.qc.excluded
    assert candidate.qc.flags == ("curvature_v_out",)
    _assert_depth_path_untouched(candidate, radargram)


def test_c2b_wide_arc_is_out_at_the_upper_v_bound(script) -> None:
    x_m, x0_m, inside = _x_and_mask(1.0)
    arc_ns = LAYER_TIME_NS + (2 / 0.1) * ((x_m - x0_m) ** 2 / (2 * 20.0))
    radargram = _layer_line(arc_ns, 2.0 * inside)
    sample = (T_SRC_NS + LAYER_TIME_NS) / QC_DT_NS
    curvature = _run_one(script, radargram, CENTRE_TRACE, sample).qc.curvature
    assert (curvature.verdict, curvature.reason) == ("out", "v_at_upper_bound")


CROSSING_TIME_NS = 10.0


def _crossing_line(slope_ns_per_m: float, cross: bool) -> Radargram:
    """C3 X (two dipping events) or C3b V (one event bent at x0)."""
    spec = _straight_ray_spec(9.0, 1.4, n_samples=QC_RECORD_SAMPLES)
    x_m, x0_m = spec.x_m, spec.x0_m
    inside = 1.5 * (np.abs(x_m - x0_m) <= 1.2).astype(float)
    if cross:
        for sign in (1.0, -1.0):
            times = CROSSING_TIME_NS + sign * slope_ns_per_m * (x_m - x0_m)
            spec.extra_events.append((times, inside))
    else:
        times = CROSSING_TIME_NS + slope_ns_per_m * np.abs(x_m - x0_m)
        spec.extra_events.append((times, inside))
    return _radargram(spec)


@pytest.mark.parametrize(
    ("slope_ns_per_m", "verdict", "reason"),
    [
        # v1.9 T2 (spec_depth 5-3); v1.8 gave unknown insufficient_picks,
        # fit_failed, fit_failed, insufficient_picks for 5, 10, 15, 20 ns/m.
        (5.0, "out", "r_below"),
        (10.0, "unknown", "unstable"),
        (15.0, "unknown", "unstable"),
        (20.0, "unknown", "unstable"),
        (30.0, "unknown", "insufficient_picks"),
    ],
)
def test_c3_x_crossing_is_never_in(script, slope_ns_per_m, verdict, reason) -> None:
    radargram = _crossing_line(slope_ns_per_m, cross=True)
    sample = (T_SRC_NS + CROSSING_TIME_NS) / QC_DT_NS
    curvature = _run_one(script, radargram, CENTRE_TRACE, sample).qc.curvature
    assert (curvature.verdict, curvature.reason) == (verdict, reason)


@pytest.mark.parametrize(
    ("slope_ns_per_m", "expected_r", "inlier_frac"),
    [(10.0, 1.1636, 0.4242), (20.0, 0.7088, 0.2424)],
)
def test_c3b_v_is_unstable(script, slope_ns_per_m, expected_r, inlier_frac) -> None:
    radargram = _crossing_line(slope_ns_per_m, cross=False)
    sample = (T_SRC_NS + CROSSING_TIME_NS) / QC_DT_NS
    curvature = _run_one(script, radargram, CENTRE_TRACE, sample).qc.curvature
    assert (curvature.verdict, curvature.reason) == ("unknown", "unstable")
    assert curvature.r == pytest.approx(expected_r, abs=R_ABS_TOL)
    assert curvature.inlier_frac == pytest.approx(inlier_frac, abs=PURE_ABS_TOL)


def test_c4_one_wing_has_insufficient_picks(script) -> None:
    spec = _straight_ray_spec(9.0, 3.0, n_samples=QC_RECORD_SAMPLES)  # off record
    x_m, x0_m = spec.x_m, spec.x0_m
    wing_ns = cylinder_times(x_m, x0_m, 0.5, spec.velocity, 0.0)
    spec.extra_events.append((wing_ns, (x_m >= x0_m - 1e-9).astype(float)))
    radargram = _radargram(spec)
    sample = (T_SRC_NS + float(wing_ns[CENTRE_TRACE])) / QC_DT_NS
    curvature = _run_one(script, radargram, CENTRE_TRACE, sample).qc.curvature
    assert (curvature.verdict, curvature.reason) == ("unknown", "insufficient_picks")
    assert (curvature.n_left, curvature.n_right) == (2, 47)


def _record_end_spec(margin_ns: float) -> BScanSpec:
    """C5: apex ``margin_ns`` above the last sample of a 512-sample record."""
    apex_time_ns = (QC_RECORD_SAMPLES - 1) * QC_DT_NS - margin_ns
    depth_m = _qc_spec().velocity * (apex_time_ns - T_SRC_NS) / 2
    return _straight_ray_spec(9.0, depth_m, n_samples=QC_RECORD_SAMPLES)


@pytest.mark.parametrize(
    ("margin_ns", "at_record_end", "verdict", "reason"),
    [
        (0.8, True, "in", ""),
        (1.5, True, "in", ""),
        # v1.9 T2 (spec_depth 5-3): v1.8 stopped both wings early (6/2 picks,
        # unknown insufficient_picks); rule D still excludes the candidate.
        (3.0, True, "in", ""),
        (6.0, False, "in", ""),
    ],
)
def test_c5_apex_at_the_record_end(
    script, margin_ns, at_record_end, verdict, reason
) -> None:
    spec = _record_end_spec(margin_ns)
    candidate = _run_one(script, _radargram(spec), *_true_apex(spec))
    assert candidate.result.status == "ok"
    assert candidate.qc.record_end_margin_ns == pytest.approx(margin_ns, abs=1e-6)
    assert candidate.qc.at_record_end is at_record_end
    assert candidate.qc.excluded is at_record_end
    curvature = candidate.qc.curvature
    assert (curvature.verdict, curvature.reason) == (verdict, reason)


@pytest.mark.parametrize("dt_ns", [0.098, 0.117, 0.195])
def test_c5b_record_end_boundary(script, dt_ns) -> None:
    flags = []
    for margin_ns in (5.0 - 2e-9, 5.0, 5.0 + 1e-6):
        sample = QC_RECORD_SAMPLES - 1 - margin_ns / dt_ns
        flags.append(
            script.apex_at_record_end(sample, QC_RECORD_SAMPLES, dt_ns, QC_F_C_HZ)
        )
    assert flags == [True, False, False]


def _ringing(n_samples: int, dt_ns: float) -> np.ndarray:
    """C6 (g): ringing below the zero band only, ``0.5 sin(...) exp(-(t - 3) / 25)``."""
    t_ns = np.arange(n_samples) * dt_ns
    after_source = np.clip(t_ns - T_SRC_NS, 0, None)
    wave = 0.5 * np.sin(2 * np.pi * 0.35 * (t_ns - T_SRC_NS))
    return wave * np.exp(-after_source / 25.0) * (t_ns > T_SRC_NS)


def _with_seam(data: np.ndarray, split: int, kind: str) -> np.ndarray:
    seamed = np.array(data, dtype=np.float64)
    if kind == "gain_2":
        seamed[:, split:] *= 2.0
    elif kind == "shift_3":
        seamed[:, split:] = np.roll(seamed[:, split:], 3, axis=0)
    elif kind == "dead_trace":
        seamed[:, split] = 0.0
    elif kind == "ringing":
        seamed[:, split:] += _ringing(seamed.shape[0], QC_DT_NS)[:, None]
    return seamed.astype(np.float32)


def _seam_line(kind: str | None, offset: int) -> tuple[Radargram, int, float]:
    spec = _straight_ray_spec(9.0, 0.5, n_samples=QC_RECORD_SAMPLES)
    radargram = _radargram(spec)
    trace, sample = _true_apex(spec)
    if kind is not None:
        data = _with_seam(radargram.data, trace + offset, kind)
        radargram = dataclasses.replace(radargram, data=data)
    return radargram, trace, sample


@pytest.mark.parametrize(
    ("kind", "offset", "on_seam", "score"),
    [
        (None, 0, False, 0.072),
        ("gain_2", 2, True, 3.11),
        ("gain_2", 12, False, None),
        ("shift_3", 2, True, 1.48),
        ("shift_3", 12, False, None),
        ("dead_trace", 3, True, math.inf),
        ("ringing", 2, False, None),  # blocks below the zero band are not seen
    ],
)
def test_c6_trace_seams(script, kind, offset, on_seam, score) -> None:
    radargram, trace, sample = _seam_line(kind, offset)
    candidate = _run_one(script, radargram, trace, sample)
    assert candidate.result.status == "ok"
    assert candidate.qc.curvature.verdict == "in"
    assert candidate.qc.on_trace_seam is on_seam
    assert candidate.qc.excluded is on_seam
    if score == math.inf:
        assert candidate.qc.seam_score == math.inf
    elif score is not None:
        assert candidate.qc.seam_score == pytest.approx(score, abs=SEAM_SCORE_ABS_TOL)


N_PURE_TRACES = 401
PURE_SPLIT_INDEX = 199
"""Split j = 200 (between traces 199 and 200) sits at index j - 1."""


def _pure_scores(script, lag_ns: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    return script.seam_scores(lag_ns, amplitude, QC_F_C_HZ, 0.02)


def _step(values: np.ndarray, start: int, add: float = 0.0, factor: float = 1.0):
    stepped = values.copy()
    stepped[start:] = stepped[start:] * factor + add
    return stepped


@pytest.mark.parametrize(
    ("lag_step_ns", "amplitude_factor", "score", "n_seams"),
    [
        (0.21, 1.0, 1.0500, 1),
        (0.19, 1.0, 0.9500, 0),
        (0.0, 1.26, 1.0357, 1),
        (0.0, 1.24, 0.9640, 0),
    ],
)
def test_c6b_seam_score_steps(
    script, lag_step_ns, amplitude_factor, score, n_seams
) -> None:
    lag_ns = _step(np.zeros(N_PURE_TRACES), 200, add=lag_step_ns)
    amplitude = _step(np.ones(N_PURE_TRACES), 200, factor=amplitude_factor)
    scores = _pure_scores(script, lag_ns, amplitude)
    assert len(scores) == N_PURE_TRACES - 1
    assert scores[PURE_SPLIT_INDEX] == pytest.approx(score, abs=PURE_ABS_TOL)
    assert int(np.sum(scores >= 1)) == n_seams
    assert np.all(np.delete(scores, PURE_SPLIT_INDEX) == 0)


def test_c6b_noise_raises_the_seam_limit(script) -> None:
    lag_ns = np.random.default_rng(0).normal(0.0, 0.05, N_PURE_TRACES)
    lag_ns[200:] += 0.21
    scores = _pure_scores(script, lag_ns, np.ones(N_PURE_TRACES))
    assert scores[PURE_SPLIT_INDEX] == pytest.approx(0.4175, abs=PURE_ABS_TOL)
    others = np.delete(scores, [198, 199, 200])
    assert others.max() == pytest.approx(0.7451, abs=PURE_ABS_TOL)
    assert int(np.sum(scores >= 1)) == 0


def test_c6b_zero_amplitude_is_an_infinite_seam(script) -> None:
    amplitude = np.ones(N_PURE_TRACES)
    amplitude[200] = 0.0
    scores = _pure_scores(script, np.zeros(N_PURE_TRACES), amplitude)
    assert scores[199] == scores[200] == math.inf
    assert int(np.sum(scores >= 1)) == 2
    assert np.all(np.delete(scores, [199, 200]) == 0)


def test_c6b_score_exactly_one_is_a_seam(script) -> None:
    """C6c: a 0.2 ns step is exactly ``theta_t`` (score 1.0); ``s_j >= 1`` is a seam.

    Catches a ``>`` mutant (spec 9-2-2 C6c, listed since v1.4.1).
    """
    lag_ns = _step(np.zeros(N_PURE_TRACES), 200, add=0.2)
    scores = _pure_scores(script, lag_ns, np.ones(N_PURE_TRACES))
    assert scores[PURE_SPLIT_INDEX] == 1.0
    line = SimpleNamespace(dt_ns=QC_DT_NS, dx_m=0.02, f_c_hz=QC_F_C_HZ, n_samples=512)
    row = SimpleNamespace(trace=200, sample=100.0)
    no_box = DepthResult(status=DepthStatus.ZERO_FAILED)
    qc = script.candidate_qc(line, None, row, None, no_box, scores)
    assert (qc.seam_score, qc.seam_split, qc.on_trace_seam) == (1.0, 200, True)


def test_c6b_seam_near_window(script) -> None:
    scores = np.zeros(N_PURE_TRACES - 1)
    scores[[194, 195, 205, 206]] = [5.0, 2.0, 3.0, 9.0]  # splits 195, 196, 206, 207
    # Trace 200, J = 5: splits j with 195 <= j - 1 and j <= 205 -> 196..205.
    assert script.seam_near(scores, 200, 0.02) == (2.0, 196)
    assert script.seam_near(np.zeros(0), 0, 0.02) == (None, None)


def _without_zero_band(script, radargram: Radargram, inputs):
    """``inputs`` with ``branch.filtered`` zeroed over ``[k0 - L, k1 + L]`` (C6d)."""
    period_ns = 1e9 / QC_F_C_HZ
    max_lag = math.ceil(script.SEAM_LAG_HALF_PERIODS * period_ns / QC_DT_NS)
    filtered = np.array(inputs.branch.filtered, copy=True)
    k0, k1 = script._zero_band_window(
        inputs.zero.t_zero_ns, QC_DT_NS, period_ns, filtered.shape[0], max_lag
    )
    filtered[k0 - max_lag : k1 + max_lag + 1] = 0.0
    branch = dataclasses.replace(inputs.branch, filtered=filtered)
    return dataclasses.replace(inputs, branch=branch)


def test_c6d_line_without_zero_band_energy_has_no_seam_rule(script) -> None:
    radargram, trace, sample = _seam_line(None, 0)
    inputs = prepare_depth_inputs(radargram)
    assert script._line_seam_scores(radargram, inputs) is not None
    degenerate = _without_zero_band(script, radargram, inputs)
    assert script._line_seam_scores(radargram, degenerate) is None

    row = SimpleNamespace(trace=trace, sample=sample)
    no_box = DepthResult(status=DepthStatus.ZERO_FAILED)
    qc = script.candidate_qc(radargram, degenerate, row, None, no_box, None)
    assert (qc.seam_score, qc.seam_split, qc.on_trace_seam) == (None, None, False)
    columns = script._qc_columns(qc)
    assert (columns["seam_score"], columns["seam_split"]) == ("", "")
    assert columns["apex_on_trace_seam"] is False


def test_c7_hint_without_an_event_is_no_apex(script) -> None:
    radargram, trace, sample = _seam_line(None, 0)
    candidate = _run_one(script, radargram, trace, sample + 5.0 / QC_DT_NS)
    assert candidate.result.status == "fit_failed"
    assert "apex_hint_unmatched" in candidate.result.flags
    curvature = candidate.qc.curvature
    assert (curvature.verdict, curvature.reason) == ("unknown", "no_apex")


def test_c7_candidate_without_tau_is_no_box(script) -> None:
    radargram, trace, _ = _seam_line(None, 0)
    sample = 300.0
    candidates = pd.DataFrame(
        {"trace": [trace], "sample": [sample], "tau_ns": [np.nan], "source": ["m"]}
    )
    (candidate,) = script.run_line(radargram, candidates)
    assert candidate.result.status == "zero_failed"
    curvature = candidate.qc.curvature
    assert (curvature.verdict, curvature.reason) == ("unknown", "no_box")
    assert candidate.qc.flags[0] == "curvature_v_unknown"
    expected_margin_ns = (QC_RECORD_SAMPLES - 1 - sample) * QC_DT_NS
    assert candidate.qc.record_end_margin_ns == pytest.approx(expected_margin_ns)
    row = script.candidate_row(candidate, {})
    assert row["flags"].split()[-1] == "curvature_v_unknown"
    assert row["qc_reason"] == "no_box"


V_META = 0.1
"""Metadata velocity of the fake fits (m/ns)."""


def _fake_fit(
    v: float = 0.1,
    x0_m: float = 0.0,
    inlier_frac: float = 1.0,
    sigma_v: float = 0.001,
    converged: bool = True,
):
    """Fake ``fit_curvature`` result over 51 picks at x -0.5..0.5 m (C7-C9)."""
    fit = SimpleNamespace(
        v_m_per_ns=v,
        x0_m=x0_m,
        t0_ns=10.0,
        sigma_v=sigma_v,
        inlier_frac=inlier_frac,
        converged=converged,
    )
    x_m = np.linspace(-0.5, 0.5, 51)
    picks = SimpleNamespace(
        apex=object(),
        n_picks=51,
        n_left=25,
        n_right=25,
        x_m=x_m,
        tau_ns=10.0 + np.abs(x_m),
        apex_x_m=0.0,
    )
    return SimpleNamespace(status="fit_failed", fit=fit), picks


@pytest.mark.parametrize(
    ("fit_args", "verdict", "reason"),
    [
        ({"converged": False}, "unknown", "fit_failed"),
        ({"x0_m": -0.5}, "unknown", "fit_failed"),
        ({"inlier_frac": 0.49}, "unknown", "unstable"),
        ({"sigma_v": math.nan}, "unknown", "unstable"),
        ({"sigma_v": 0.016}, "unknown", "unstable"),
        ({"sigma_v": 0.0149}, "in", ""),
        ({"v": 0.069}, "out", "r_below"),  # C8 r 0.69
        ({"v": 0.161}, "out", "r_above"),  # C8 r 1.61
        ({"v": 0.2998}, "out", "v_at_upper_bound"),  # C9
        ({"v": 0.2998, "x0_m": 0.5}, "unknown", "fit_failed"),
        ({"v": 0.03}, "out", "v_at_lower_bound"),
        ({"v": 0.2988}, "out", "r_above"),  # 1e-3 below the bound, r 2.988
    ],
)
def test_c7_c8_c9_classify_fake_fits(script, fit_args, verdict, reason) -> None:
    curvature, picks = _fake_fit(**fit_args)
    qc = script.classify_curvature(curvature, picks, V_META)
    assert (qc.verdict, qc.reason) == (verdict, reason)


def test_c7_unstable_limits_are_exclusive(script) -> None:
    """C7c: ``inlier_frac`` 0.5 or ``sigma_v / v_m`` 0.15 is still stable (3-5-2).

    Catches ``<=`` / ``>=`` mutants (spec 9-2-2 C7c, listed since v1.4.1).
    ``v_m`` 0.125 is a power of two, so ``0.15 * 0.125 / 0.125`` is exactly 0.15.
    """
    v_meta = 0.125
    for fit_args in (
        {"v": v_meta, "inlier_frac": 0.5},
        {"v": v_meta, "sigma_v": script.QC_UNSTABLE_SIGMA_REL * v_meta},
    ):
        curvature, picks = _fake_fit(**fit_args)
        qc = script.classify_curvature(curvature, picks, v_meta)
        assert (qc.verdict, qc.reason) == ("in", ""), fit_args


def test_c7d_non_finite_sigma_v_leaves_the_csv_cell_blank(script) -> None:
    curvature, picks = _fake_fit(sigma_v=math.nan)
    curvature_qc = script.classify_curvature(curvature, picks, V_META)
    assert (curvature_qc.verdict, curvature_qc.reason) == ("unknown", "unstable")
    assert curvature_qc.sigma_v_rel is None
    qc = script.CandidateQc(
        curvature=curvature_qc,
        record_end_margin_ns=10.0,
        at_record_end=False,
        seam_score=0.1,
        seam_split=1,
        on_trace_seam=False,
    )
    columns = script._qc_columns(qc)
    assert columns["qc_sigma_v_rel"] == ""
    assert columns["qc_r"] == 1.0


@pytest.mark.parametrize(
    ("ratio", "verdict"),
    [
        (0.7, "in"),
        (1.6, "in"),
        (0.7 - 1e-13, "in"),
        (1.6 + 1e-13, "in"),
        (0.7 - 1e-9, "out"),
        (1.6 + 1e-9, "out"),
    ],
)
def test_c8_ratio_band_edges(script, ratio, verdict) -> None:
    assert script.classify_ratio(ratio) == verdict


@pytest.mark.parametrize("ratio", [math.nan, math.inf])
def test_c8_non_finite_ratio_is_rejected(script, ratio) -> None:
    with pytest.raises(ValueError):
        script.classify_ratio(ratio)


def _qc(script, verdict: str = "in", at_record_end: bool = False):
    reason = {"in": "", "out": "r_above", "unknown": "insufficient_picks"}[verdict]
    return script.CandidateQc(
        curvature=script.CurvatureQc(script.QcVerdict(verdict), reason),
        record_end_margin_ns=10.0,
        at_record_end=at_record_end,
        seam_score=0.1,
        seam_split=1,
        on_trace_seam=False,
    )


@dataclass(frozen=True)
class _DuckQcCandidate:
    """``estimates_by_activity`` input with a QC (C10)."""

    activity_id: str
    path: str
    depth_m: float
    qc: object

    def depth(self, zero_variant: str) -> float | None:
        return self.depth_m


def _c10_candidates(script) -> list[_DuckQcCandidate]:
    return [
        _DuckQcCandidate("01.1", "L1", 0.60, _qc(script, "in")),
        _DuckQcCandidate("01.1", "L2", 0.61, _qc(script, "out")),
        _DuckQcCandidate("01.1", "L3", 0.62, _qc(script, "unknown")),
        _DuckQcCandidate("01.1", "L1", 1.20, _qc(script, "in", at_record_end=True)),
        _DuckQcCandidate("01.1", "L2", 1.21, _qc(script, "in")),
    ]


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        # B+D+E: 0.61 (out), 0.62 (unknown) and 1.20 (record end) are dropped
        # before clustering, so 0.60 (L1) and 1.21 (L2) are each seen on one line.
        ("main", []),
        ("a", [0.60, 1.21]),
        ("b", [0.60, 1.21]),
        # B+D (the v1.4 main expectation): the unknown 0.62 supports 0.60.
        ("bd", [0.61]),
        ("noqc", [0.61, 1.205]),
    ],
)
def test_c10_qc_runs_before_clustering(script, variant, expected) -> None:
    """C10 (v1.5): each variant's QC rule acts before clustering."""
    estimates, _ = script.estimates_by_activity(
        _c10_candidates(script),
        "detected",
        variant,
        {"01.1": 3},
        width_m=0.10,
        linkage="complete",
        min_lines=2,
    )
    # 1.20 is dropped before clustering, so 1.21 loses its second line.
    assert estimates["01.1"] == pytest.approx(expected, abs=DEPTH_ABS_TOL)


def _c11_lines() -> list[tuple[Radargram, int, float]]:
    """C1 (eps 9, R 0, d 0.5), C2 (amplitude 2), C5 (0.8 ns) and C6 (b) lines."""
    straight = _straight_ray_spec(9.0, 0.5)
    x_m, _, inside = _x_and_mask(1.0)
    layer = _layer_line(np.full(len(x_m), LAYER_TIME_NS), 2.0 * inside)
    record_end = _record_end_spec(0.8)
    return [
        (_radargram(straight), *_true_apex(straight)),
        (layer, CENTRE_TRACE, (T_SRC_NS + LAYER_TIME_NS) / QC_DT_NS),
        (_radargram(record_end), *_true_apex(record_end)),
        _seam_line("gain_2", 2),
    ]


DEPTH_COLUMNS = (
    "status",
    "depth_top_m",
    "depth_est_m",
    "depth_act_m",
    "eps_r_used",
    "apex_trace",
    "apex_sample",
)


def test_c11_qc_does_not_change_the_depth(script) -> None:
    for radargram, trace, sample in _c11_lines():
        with_qc = _run_one(script, radargram, trace, sample, apex_qc=True)
        without_qc = _run_one(script, radargram, trace, sample, apex_qc=False)
        assert with_qc.qc is not None and without_qc.qc is None
        _assert_same_depth_result(with_qc.result, without_qc.result)
        row_with = script.candidate_row(with_qc, {})
        row_without = script.candidate_row(without_qc, {})
        for column in DEPTH_COLUMNS:
            assert row_with[column] == row_without[column], column


def test_c11b_a_wrong_qc_fit_cannot_reach_the_depth(script, monkeypatch) -> None:
    spec = _straight_ray_spec(9.0, 0.5)
    radargram = _radargram(spec)
    trace, sample = _true_apex(spec)
    reference = _run_one(script, radargram, trace, sample, apex_qc=False)
    real_fit_curvature = script.fit_curvature

    def fit_at_quarter_light_speed(picks, **kwargs):
        curvature = real_fit_curvature(picks, **kwargs)
        wrong = dataclasses.replace(curvature.fit, v_m_per_ns=0.25)
        return dataclasses.replace(curvature, fit=wrong)

    monkeypatch.setattr(script, "fit_curvature", fit_at_quarter_light_speed)
    candidate = _run_one(script, radargram, trace, sample, apex_qc=True)
    assert (candidate.qc.curvature.verdict, candidate.qc.curvature.reason) == (
        "out",
        "r_above",
    )
    _assert_same_depth_result(candidate.result, reference.result)
    assert candidate.depth("detected") == reference.depth("detected")


def test_c13_fit_curvature_is_called_only_in_curvature_qc() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    callers = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and _called_name(node) == "fit_curvature":
                callers.append(function.name)
    assert callers == ["curvature_qc"]
    module_level_calls = [
        node
        for statement in tree.body
        if not isinstance(statement, ast.FunctionDef | ast.ClassDef)
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and _called_name(node) == "fit_curvature"
    ]
    assert module_level_calls == []


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


CANDIDATE_CSV_COLUMNS = [
    "activity_id",
    "site",
    "path",
    "candidate_trace",
    "candidate_sample",
    "source",
    "status",
    "depth_top_m",
    "depth_est_m",
    "depth_act_m",
    "label_est",
    "label_act",
    "eps_r_used",
    "zero_rule",
    "t_zero_ns",
    "h_est_m",
    "apex_trace",
    "apex_sample",
    "candidate_moved",
    "eps_qc_lo",
    "eps_qc_hi",
    "flags",
    "box_k0",
    "search_k0",
    "apex_minus_search_k0",
    "qc_verdict",
    "qc_reason",
    "qc_r",
    "qc_v_curv_m_per_ns",
    "qc_sigma_v_rel",
    "qc_inlier_frac",
    "qc_n_picks",
    "qc_n_left",
    "qc_n_right",
    "qc_aperture_m",
    "record_end_margin_ns",
    "apex_at_record_end",
    "seam_score",
    "seam_split",
    "apex_on_trace_seam",
    "qc_excluded",
    "stop_left",
    "stop_right",
    "apex_descended",
]
"""Column order of the candidates csv (spec 8 v1.4; tracking: spec_depth v1.9)."""
QC_DEPENDENT_COLUMNS = CANDIDATE_CSV_COLUMNS[25:-1]
"""Blank with ``--apex-qc off``; ``apex_descended`` comes from the depth flags."""


def test_c13_candidate_csv_columns_and_flags(script) -> None:
    radargram, trace, sample = _seam_line("dead_trace", 3)
    with_qc = _run_one(script, radargram, trace, sample)
    row = script.candidate_row(with_qc, {})
    assert list(row) == CANDIDATE_CSV_COLUMNS
    assert row["seam_score"] == "inf"
    assert row["apex_on_trace_seam"] is True and row["qc_excluded"] is True
    assert row["flags"].split()[-1] == "apex_on_trace_seam"
    assert row["stop_left"] in {stop.value for stop in script.TrackStop}
    assert row["stop_right"] in {stop.value for stop in script.TrackStop}
    assert row["apex_descended"] is False
    without_qc = _run_one(script, radargram, trace, sample, apex_qc=False)
    row = script.candidate_row(without_qc, {})
    assert list(row) == CANDIDATE_CSV_COLUMNS
    assert all(row[column] == "" for column in QC_DEPENDENT_COLUMNS)
    assert row["apex_descended"] is False
    assert "apex_on_trace_seam" not in row["flags"].split()


def test_c13_unknown_curvature_is_qc_excluded_without_a_new_column(script) -> None:
    """C13 (b), v1.5: ``qc_excluded`` follows B+D+E; the columns stay the same."""
    spec = _straight_ray_spec(9.0, 0.5)
    candidate = _run_one(script, _radargram(spec), *_true_apex(spec))
    unknown = script.CurvatureQc(script.QcVerdict.UNKNOWN, "insufficient_picks")
    unknown_copy = dataclasses.replace(
        candidate, qc=dataclasses.replace(candidate.qc, curvature=unknown)
    )
    row = script.candidate_row(candidate, {})
    copy_row = script.candidate_row(unknown_copy, {})
    assert row["qc_verdict"] == "in" and row["qc_excluded"] is False
    assert copy_row["qc_verdict"] == "unknown"
    assert copy_row["qc_excluded"] is True
    assert copy_row["flags"].split()[-1] == "curvature_v_unknown"
    assert list(row) == list(copy_row) == CANDIDATE_CSV_COLUMNS


def test_c13f_qc_flags_follow_long_line(script) -> None:
    radargram, trace, sample = _seam_line("dead_trace", 3)
    candidate = _run_one(script, radargram, trace, sample)
    lines = {(candidate.activity_id, candidate.path): {"length_m": 35.0}}
    row = script.candidate_row(candidate, lines)
    assert row["flags"].split()[-2:] == ["long_line", "apex_on_trace_seam"]


def test_c13_cli_apex_qc_option(script) -> None:
    assert script.parse_args([]).apex_qc == "on"
    assert script.parse_args(["--apex-qc", "off"]).apex_qc == "off"
    with pytest.raises(SystemExit) as raised:
        script.parse_args(["--apex-qc", "maybe"])
    assert raised.value.code == 2


def _qc_candidate(script, path: str, depth: float, qc):
    return dataclasses.replace(_candidate(script, "01.1", path, depth), qc=qc)


def _c13_candidates(script) -> list:
    """01.1 on 3 lines: pipe at 1.10 m on all, a stray 0.80 m (out) on L1, L2."""
    return [
        _qc_candidate(script, "L1", 1.10, _qc(script, "in")),
        _qc_candidate(script, "L2", 1.11, _qc(script, "unknown")),
        _qc_candidate(script, "L3", 1.12, _qc(script, "in")),
        _qc_candidate(script, "L1", 0.80, _qc(script, "out")),
        _qc_candidate(script, "L2", 0.81, _qc(script, "out")),
    ]


METRIC_COLUMNS = ("n_est", "n_pairs", "mae_m", "pass_a", "pass_b", "false_est")


def test_c13_apex_qc_off_restores_the_v131_main_result(script) -> None:
    groundtruth, metadata = _tables()
    gt_sets = script.build_ground_truth_sets(groundtruth, metadata)
    candidates = _c13_candidates(script)
    lines = {"01.1": 3}
    on = script.evaluate_combinations(candidates, gt_sets, lines, _args())
    off = script.evaluate_combinations(candidates, gt_sets, lines, _args(apex_qc="off"))
    all_variants = {"main", "a", "b", "bd", "noqc"}
    for rows in (on.summary_rows, on.match_rows):
        assert {row["variant"] for row in rows} == all_variants
    for rows in (off.summary_rows, off.match_rows):
        assert {row["variant"] for row in rows} == {"main", "a", "b"}
    on_main = _main_rows(on.summary_rows)["exclude_x"]
    off_main = _main_rows(off.summary_rows)["exclude_x"]
    on_bd = _overall_top_detected_row(on.summary_rows, "bd")
    on_noqc = _overall_top_detected_row(on.summary_rows, "noqc")
    # main drops the unknown 1.11 (L2) but 1.10 (L1) and 1.12 (L3) still pair up;
    # bd keeps 1.11, so the same pipe is one estimate either way.
    assert on_main["n_est"] == 1 and on_bd["n_est"] == 1
    assert off_main["n_est"] == 2
    for column in METRIC_COLUMNS:
        assert off_main[column] == on_noqc[column], column


def _overall_top_detected_row(rows, variant: str) -> dict[str, object]:
    (row,) = [
        row
        for row in rows
        if (row["scope"], row["gt_set"], row["assumption"], row["zero_variant"])
        == ("all", "exclude_x", "top", "detected")
        and row["variant"] == variant
    ]
    return row


def test_c13_md_qc_diagnostic_lines(script) -> None:
    groundtruth, metadata = _tables()
    gt_sets = script.build_ground_truth_sets(groundtruth, metadata)
    candidates = _c13_candidates(script)
    args = _args()
    evaluation = script.evaluate_combinations(candidates, gt_sets, {"01.1": 3}, args)
    lines = script._qc_diagnostic_lines(candidates, evaluation, args)
    assert "v1.5" in lines[0]
    assert _line_starting_with(lines, "- `ok` 후보 5개").endswith(
        "in 2 / out 2 / unknown 1"
    )
    reasons = _line_starting_with(lines, "- 사유")
    assert "out `r_above` 2" in reasons
    assert "unknown `insufficient_picks` 1" in reasons
    excluded = _line_starting_with(lines, "- `qc_excluded`")
    assert "(주 결과 B+D+E): 3/5" in excluded
    assert "B+D(`bd`): 2/5" in excluded
    assert "E로 더 뺀 후보(곡률 unknown, D 없음): 1" in excluded
    assert lines[lines.index(excluded) + 1] == "  - E로 더 뺀 후보 출처별: `auto_v1` 1"
    tracking = _line_starting_with(lines, "- 추적 진단")
    assert "`apex_descended` 0/5" in tracking
    assert "왼쪽 `none` 5 / 오른쪽 `none` 5" in tracking
    counts = "1개: main 1 / bd 1 / noqc 2 / main-bd +0 / main-noqc -1"
    assert _line_starting_with(lines, "  - 전체 활동").endswith(counts)
    assert _line_starting_with(lines, "  - 정답 활동").endswith(counts)
    assert script.PREREGISTERED_QC_NOTE in _line_starting_with(lines, "- 사전 등록")
    off_lines = script._qc_diagnostic_lines(
        candidates, evaluation, _args(apex_qc="off")
    )
    assert any("apex_qc=off" in line for line in off_lines)
    qc_setting = script._qc_setting(_args())
    assert "B+D+E" in qc_setting and "`bd`" in qc_setting


def _c16_candidates(script) -> list:
    """Four ``ok`` candidates (in / out / two unknown) and one ``fit_failed``."""

    def with_curvature(depth: float, verdict: str, reason: str, r: float | None):
        curvature = script.CurvatureQc(script.QcVerdict(verdict), reason, r=r)
        qc = dataclasses.replace(_qc(script), curvature=curvature)
        return _qc_candidate(script, "L1", depth, qc)

    failed = with_curvature(0.90, "out", "r_below", 0.5)
    return [
        with_curvature(1.00, "in", "", 1.0),
        with_curvature(1.01, "out", "r_above", 1.7),
        with_curvature(1.02, "unknown", "unstable", 5.0),
        with_curvature(1.03, "unknown", "insufficient_picks", None),
        dataclasses.replace(failed, result=DepthResult(status=DepthStatus.FIT_FAILED)),
    ]


def test_c16_md_qc_lines_count_ok_candidates_and_known_verdict_ratios(
    script,
) -> None:
    groundtruth, metadata = _tables()
    gt_sets = script.build_ground_truth_sets(groundtruth, metadata)
    candidates = _c16_candidates(script)
    args = _args()
    evaluation = script.evaluate_combinations(candidates, gt_sets, {"01.1": 3}, args)
    lines = script._qc_diagnostic_lines(candidates, evaluation, args)
    assert _line_starting_with(lines, "- `ok` 후보 4개").endswith(
        "in 1 / out 1 / unknown 2"
    )
    ratio_line = _line_starting_with(lines, "- 알려진 r")
    assert ratio_line.startswith("- 알려진 r 2개")
    assert ratio_line.endswith("[1.0, 1.07, 1.35, 1.63, 1.7]")
    # v1.5: the fit_failed candidate is not counted under either rule.
    excluded = _line_starting_with(lines, "- `qc_excluded`")
    assert "(주 결과 B+D+E): 3/4" in excluded
    assert "B+D(`bd`): 1/4" in excluded
    assert "E로 더 뺀 후보(곡률 unknown, D 없음): 2" in excluded


def test_c14_a_different_box_or_apex_raises(script) -> None:
    spec = _straight_ray_spec(9.0, 0.5)
    radargram = _radargram(spec)
    trace, sample = _true_apex(spec)
    candidate = _run_one(script, radargram, trace, sample)
    inputs = prepare_depth_inputs(radargram)
    box = script.candidate_box(
        radargram, trace, sample, sample * QC_DT_NS - inputs.zero.t_zero_ns
    )
    hint = script.ApexHint(trace=trace, sample=sample)
    other_apex = dataclasses.replace(
        candidate.result, apex_trace=candidate.result.apex_trace + 1
    )
    with pytest.raises(RuntimeError):
        script.curvature_qc(radargram, box, inputs, hint, other_apex)
    # Box starting 5 traces right of the apex: even widened it misses trace 199.
    shifted_box = dataclasses.replace(box, i0=trace + 5)
    with pytest.raises(RuntimeError):
        script.curvature_qc(radargram, shifted_box, inputs, hint, candidate.result)


# ----------------------------------------------------------------- v1.5 (rule E)

UNKNOWN_REASONS = ("no_box", "no_apex", "insufficient_picks", "fit_failed", "unstable")
V144_CLI_KEYS = {
    "root",
    "candidates",
    "lines",
    "out_dir",
    "prefix",
    "activities",
    "cluster_width_m",
    "cluster_linkage",
    "min_support_lines",
    "baseline_trials",
    "baseline_trials_aux",
    "baseline_seed",
    "apex_qc",
    "dry_run",
}
"""``parse_args([])`` attributes of v1.4.4; v1.5 adds no option (spec 9-1)."""
SPEC_PATH = PROJECT_ROOT / "docs" / "processing" / "spec_eval_4tu_v1.md"
PREREGISTERED_ITEM_3_PREFIX = "3. **꼭짓점 후보 QC(C, v1.5"
CURVATURE_FLAGS = {
    "in": (),
    "out": ("curvature_v_out",),
    "unknown": ("curvature_v_unknown",),
}


def _candidate_qc(script, verdict: str, reason: str, record_end: bool, seam: bool):
    return script.CandidateQc(
        curvature=script.CurvatureQc(script.QcVerdict(verdict), reason),
        record_end_margin_ns=10.0,
        at_record_end=record_end,
        seam_score=0.1,
        seam_split=1,
        on_trace_seam=seam,
    )


def test_c17_exclusion_rule_truth_table(script) -> None:
    """C17 (v1.5): B+D+E and B+D differ only for (unknown, no D flag)."""
    reasons = {"in": "", "out": "r_above", "unknown": "insufficient_picks"}
    differing = []
    for verdict, record_end, seam in itertools.product(
        reasons, (False, True), (False, True)
    ):
        qc = _candidate_qc(script, verdict, reasons[verdict], record_end, seam)
        assert qc.excluded == (verdict != "in" or record_end or seam)
        assert qc.excluded_bd == (verdict == "out" or record_end or seam)
        expected_flags = (
            *CURVATURE_FLAGS[verdict],
            *(("apex_at_record_end",) if record_end else ()),
            *(("apex_on_trace_seam",) if seam else ()),
        )
        assert qc.flags == expected_flags
        if qc.excluded != qc.excluded_bd:
            differing.append((verdict, record_end, seam))
    assert differing == [("unknown", False, False)]


@pytest.mark.parametrize("reason", UNKNOWN_REASONS)
def test_c17_every_unknown_reason_is_excluded_by_e_only(script, reason) -> None:
    qc = _candidate_qc(script, "unknown", reason, record_end=False, seam=False)
    assert qc.excluded is True
    assert qc.excluded_bd is False
    assert qc.flags == ("curvature_v_unknown",)


def test_c18_variants_and_cli(script) -> None:
    """C18 (v1.5): ``bd`` is an auxiliary variant; the CLI is unchanged."""
    assert script.VARIANTS == ("main", "a", "b", "bd", "noqc")
    assert script.variants_for(_args()) == script.VARIANTS
    assert script.variants_for(_args(apex_qc="off")) == ("main", "a", "b")
    args = _args()
    assert script.baseline_trials_for("bd", args) == {
        "activity": args.baseline_trials_aux
    }
    assert script.baseline_trials_for("bd", args) == script.baseline_trials_for(
        "noqc", args
    )
    assert set(vars(script.parse_args([]))) == V144_CLI_KEYS
    with pytest.raises(SystemExit) as raised:
        script.parse_args(["--qc-unknown", "keep"])
    assert raised.value.code == 2


def _markdown_lines(script, tmp_path: Path, apex_qc: str) -> list[str]:
    groundtruth, metadata = _tables()
    gt_sets = script.build_ground_truth_sets(groundtruth, metadata)
    candidates = _c13_candidates(script)
    args = script.parse_args(
        ["--baseline-trials", "30", "--baseline-trials-aux", "10", "--apex-qc", apex_qc]
    )
    evaluation = script.evaluate_combinations(candidates, gt_sets, {"01.1": 3}, args)
    path = tmp_path / f"md_{apex_qc}.md"
    script.write_markdown(path, evaluation, candidates, args, elapsed_s=0.0)
    return path.read_text(encoding="utf-8").splitlines()


def _index_starting_with(lines: list[str], prefix: str) -> int:
    (index,) = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    return index


def test_c19_md_auxiliary_tables_follow_the_main_table(script, tmp_path) -> None:
    """C19 (v1.5): main -> bd -> noqc -> X included; no bd/noqc when off."""
    lines = _markdown_lines(script, tmp_path, "on")
    order = [
        _index_starting_with(lines, prefix)
        for prefix in (
            "## 주 결과",
            "### 보조: B+D (`bd`",
            "### 보조: QC 없음 (`noqc`",
            "## 보조: X 포함",
        )
    ]
    assert order == sorted(order)
    assert any("bd: B+D(곡률 unknown 유지)" in line for line in lines)
    off_lines = _markdown_lines(script, tmp_path, "off")
    assert not any(line.startswith("### 보조: B+D") for line in off_lines)
    assert not any(line.startswith("### 보조: QC 없음") for line in off_lines)
    assert not any("bd: B+D(곡률 unknown 유지)" in line for line in off_lines)


def test_c20_preregistered_note_matches_the_spec(script) -> None:
    """C20 (v1.5): ``PREREGISTERED_QC_NOTE`` is the spec 11 item 3 text verbatim."""
    spec_lines = SPEC_PATH.read_text(encoding="utf-8").splitlines()
    (item_3,) = [
        line for line in spec_lines if line.startswith(PREREGISTERED_ITEM_3_PREFIX)
    ]
    note = script.PREREGISTERED_QC_NOTE
    assert note in item_3
    for fragment in ("B+D+E", "14.3", "35.7", "29/41", "126 → 49"):
        assert fragment in note


def _md_table_under(lines: list[str], heading_prefix: str) -> tuple[str, list[str]]:
    """Header line and data rows of the md table under a unique heading."""
    heading = _index_starting_with(lines, heading_prefix)
    header = next(i for i in range(heading + 1, len(lines)) if lines[i].startswith("|"))
    rows = []
    for line in lines[header + 2 :]:  # skip the |---| separator
        if not line.startswith("|"):
            break
        rows.append(line)
    return lines[header], rows


def _md_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def test_c19b_md_auxiliary_tables_hold_their_own_variant(script, tmp_path) -> None:
    """C19b (v1.5.1): (top, detected) ``n_est`` is main 1, bd 1, noqc 2.

    Catches ``write_markdown`` filling the bd table with noqc rows (mutation X3).
    """
    lines = _markdown_lines(script, tmp_path, "on")
    expected_n_est = {"## 주 결과": "1", "### 보조: B+D": "1", "### 보조: QC 없음": "2"}
    for heading, n_est in expected_n_est.items():
        header, rows = _md_table_under(lines, heading)
        assert _md_cells(header)[3] == "n_est", heading
        first_row = _md_cells(rows[0])
        assert first_row[:2] == ["top", "detected"], heading
        assert first_row[3] == n_est, heading


# ----------------------------------------------------------------- G1-G11 (spec 9-2-3)
#
# v1.6 gate auxiliary metrics (spec 5-1). Hand-counted fixtures; G1, G2, G3, G5
# and G7 equal the prototype analysis/processing/m29d_gate_proto.py output.

GATE_LIMIT_WITH_TOL_M = 0.30 + 1e-12
V15_SUMMARY_COLUMNS = [
    "scope",
    "gt_set",
    "assumption",
    "zero_variant",
    "variant",
    "n_activities",
    "n_gt",
    "n_est",
    "n_pairs",
    "mae_m",
    "bias_m",
    "med_abs_m",
    "p90_abs_m",
    "pass_a",
    "pass_b",
    "missed",
    "false_est",
    "baseline_mae_mean",
    "baseline_mae_p025",
    "baseline_mae_p975",
    "baseline_pass_a_mean",
    "baseline_pass_a_p025",
    "baseline_pass_a_p975",
    "baseline_pass_b_mean",
    "baseline_pass_b_p025",
    "baseline_pass_b_p975",
    "baseline_p",
    "baseline_global_pass_b_mean",
    "baseline_global_pass_b_p025",
    "baseline_global_pass_b_p975",
    "baseline_global_p",
]
"""Summary csv columns up to v1.5 (spec 8), in order."""
V16_GATE_COLUMNS = [
    "gate_pairs",
    "gate_recall",
    "gate_precision",
    "gate_f1",
    "gate_mae_m",
    "baseline_gate_recall_mean",
    "baseline_gate_recall_p025",
    "baseline_gate_recall_p975",
    "baseline_gate_precision_mean",
    "baseline_gate_precision_p025",
    "baseline_gate_precision_p975",
    "baseline_gate_f1_mean",
    "baseline_gate_f1_p025",
    "baseline_gate_f1_p975",
    "baseline_gate_p",
    "baseline_gate_pct",
    "baseline_global_gate_f1_mean",
    "baseline_global_gate_f1_p025",
    "baseline_global_gate_f1_p975",
    "baseline_global_gate_p",
]
"""The 20 gate columns appended to every summary row (spec 8, v1.6)."""
GLOBAL_GATE_COLUMNS = V16_GATE_COLUMNS[-4:]
G8_HEAD_ROWS = {
    "main": (
        "all", "exclude_x", "top", "detected", "main", 1, 2, 1, 1,
        0.01, 0.01, 0.01, 0.01, 1.0, 0.5, 1, 0,
        0.0657, 0.0008, 0.1371, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0,
        0.25, 0.0, 0.5, 0.5,
    ),
    "bd": (
        "all", "exclude_x", "top", "detected", "bd", 1, 2, 1, 1,
        0.01, 0.01, 0.01, 0.01, 1.0, 0.5, 1, 0,
        0.0645, 0.0066, 0.1327, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0,
        "", "", "", "",
    ),
    "noqc": (
        "all", "exclude_x", "top", "detected", "noqc", 1, 2, 2, 2,
        0.2925, -0.2925, 0.2925, 0.2945, 1.0, 1.0, 0, 0,
        0.0954, 0.0268, 0.1438, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        "", "", "", "",
    ),
}  # fmt: skip
"""v1.5 columns of the C13 fixture rows, extracted with the HEAD d1cafe8 code
before the v1.6 change (G8 regression: the gate must not move any of them)."""
GATE_MAIN_TABLE_HEADER = (
    "| assumption | zero_variant | n_gt | n_est | pass_b | gate_pairs | gate_recall"
    " | gate_precision | gate_f1 | gate_mae_m | baseline_gate_f1_mean"
    " | baseline_gate_f1_p025 | baseline_gate_f1_p975 | baseline_gate_p"
    " | baseline_gate_pct |"
)
V15_MAIN_TABLE_HEADER = (
    "| assumption | zero_variant | n_gt | n_est | n_pairs | mae_m | bias_m | pass_a"
    " | pass_b | missed | false_est | baseline_pass_b_mean | baseline_pass_b_p025"
    " | baseline_pass_b_p975 | baseline_p |"
)
G5_SEED = 20261107
G5_CASES = 3000
G5_STRICT_CASES = 106
G5_BOUNDARY_CASES = 416


def _n_pass(match) -> int:
    return sum(abs(error) <= GATE_LIMIT_WITH_TOL_M for _, _, error in match.pairs)


def test_g1_gate_adds_a_pair_and_includes_the_float_boundary(script) -> None:
    est, gt = [0.45, 0.85], [0.15, 0.60, 1.20]
    gated = script.match_activity_gated(est, gt)
    assert gated.n_pairs == 2
    first, second = gated.pairs
    assert first[:2] == (0, 0) and second[:2] == (1, 1)
    assert 0.45 - 0.15 > 0.30
    assert first[2] == 0.45 - 0.15  # 0.30000000000000004: above 0.30, inside the gate
    assert first[2] > 0.30
    assert second[2] == pytest.approx(0.25, abs=1e-9)
    assert gated.abs_error_sum == pytest.approx(0.55, abs=1e-9)

    ungated = script.match_activity(est, gt)
    errors = sorted(error for _, _, error in ungated.pairs)
    assert errors == pytest.approx([-0.35, -0.15], abs=1e-9)
    assert _n_pass(ungated) == 1

    summary = script.summarize({"a": ungated}, {"a": 3}, gate_matches={"a": gated})
    assert summary.pass_b == pytest.approx(1 / 3, abs=1e-9)
    assert summary.gate_pairs == 2
    assert summary.gate_recall == pytest.approx(2 / 3, abs=1e-9)
    assert summary.gate_precision == pytest.approx(1.0, abs=1e-9)
    assert summary.gate_f1 == pytest.approx(0.8, abs=1e-9)
    assert summary.gate_mae_m == pytest.approx(0.275, abs=1e-9)


def test_g2_same_pair_count_takes_the_least_abs_error_sum(script) -> None:
    gated = script.match_activity_gated([0.5, 0.6], [0.75, 0.55])
    assert gated.n_pairs == 2
    # 0.5-0.55 and 0.6-0.75 (0.20), not the identity 0.5-0.75, 0.6-0.55 (0.30).
    assert [pair[:2] for pair in gated.pairs] == [(0, 1), (1, 0)]
    assert gated.abs_error_sum == pytest.approx(0.20, abs=1e-9)


def test_g3_micro_pooling_and_the_precision_denominator(script) -> None:
    estimates = {"A": [1.0, 1.1, 1.2], "B": []}
    truths = {"A": [1.05], "B": [0.5, 0.9]}
    matches = {a: script.match_activity(estimates[a], truths[a]) for a in truths}
    gated = {a: script.match_activity_gated(estimates[a], truths[a]) for a in truths}
    summary = script.summarize(matches, {"A": 1, "B": 2}, gated)
    assert summary.gate_pairs == 1
    assert summary.gate_recall == pytest.approx(1 / 3, abs=1e-9)  # mean over acts: 0.5
    assert summary.gate_precision == pytest.approx(1 / 3, abs=1e-9)  # not 1 / pairs
    assert summary.gate_f1 == pytest.approx(2 * 1 / (3 + 3), abs=1e-9)
    assert summary.gate_mae_m == pytest.approx(0.05, abs=1e-9)


def test_g4_empty_activities_undefined_values_and_old_calls(script) -> None:
    for est, gt in (([], [0.5]), ([0.5], []), ([], [])):
        gated = script.match_activity_gated(est, gt)
        assert (gated.n_pairs, gated.abs_error_sum) == (0, 0.0)
        assert (gated.n_est, gated.n_gt) == (len(est), len(gt))

    no_estimate = script.match_activity([], [0.5])
    summary = script.summarize(
        {"a": no_estimate}, {"a": 1}, {"a": script.match_activity_gated([], [0.5])}
    )
    assert summary.gate_pairs == 0
    assert summary.gate_recall == 0.0
    assert summary.gate_precision is None
    assert summary.gate_f1 == 0.0  # defined although the precision is not
    assert summary.gate_mae_m is None

    e4 = script.summarize({"a": script.match_activity([3.0], [0.5])}, {"a": 1})
    e7_match = script.MatchResult(
        pairs=((0.1, 0, 0.1), (0.3, 1, 0.3), (0.31, 2, 0.31)), missed=1, false_est=0
    )
    e7 = script.summarize({"a": e7_match}, {"a": 4})
    for summary in (e4, e7):
        for name in script.GATE_SUMMARY_FIELDS:
            assert getattr(summary, name) is None, name
    assert e4.n_pass == 0
    assert e7.pass_a == pytest.approx(2 / 3)
    assert e7.pass_b == pytest.approx(2 / 4)


def _brute_force_gate(est: list[float], gt: list[float]) -> tuple[int, float]:
    """Lexicographic optimum (max pairs, then min sum |e|) over all matchings."""
    for k in range(min(len(est), len(gt)), 0, -1):
        sums = []
        for rows in itertools.combinations(range(len(est)), k):
            for cols in itertools.permutations(range(len(gt)), k):
                errors = [abs(est[r] - gt[c]) for r, c in zip(rows, cols, strict=True)]
                if all(error <= GATE_LIMIT_WITH_TOL_M for error in errors):
                    sums.append(sum(errors))
        if sums:
            return k, min(sums)
    return 0, 0.0


def _has_boundary_pair(est: list[float], gt: list[float]) -> bool:
    return any(abs(abs(e - g) - 0.30) < 1e-9 for e in est for g in gt)


def test_g5_gate_equals_the_brute_force_optimum(script) -> None:
    rng = np.random.default_rng(G5_SEED)
    strict_cases = boundary_cases = 0
    for case in range(G5_CASES):
        n_e, n_g = rng.integers(0, 5, size=2)
        est = [float(k) * 0.05 for k in rng.integers(0, 41, n_e)]
        gt = [float(k) * 0.05 for k in rng.integers(0, 41, n_g)]
        gated = script.match_activity_gated(est, gt)
        best_pairs, best_sum = _brute_force_gate(est, gt)
        n_pass = _n_pass(script.match_activity(est, gt))
        assert gated.n_pairs == best_pairs, case
        assert gated.abs_error_sum == pytest.approx(best_sum, abs=1e-9), case
        assert n_pass <= gated.n_pairs <= min(n_e, n_g), case
        strict_cases += gated.n_pairs > n_pass
        boundary_cases += _has_boundary_pair(est, gt)
    # Regression values tied to the draw order (the prototype's).
    assert (strict_cases, boundary_cases) == (G5_STRICT_CASES, G5_BOUNDARY_CASES)


def test_g6_baseline_gate_reuses_the_same_draws(script) -> None:
    n_est = {"a": 3, "b": 1}
    gt = {"a": [0.5, 1.5], "b": [1.0]}
    trials = 50
    draws = script.draw_baseline(n_est, gt, trials=trials, seed=0)

    rng = np.random.default_rng(0)
    shape = (trials, len(gt))
    expected = {
        name: np.zeros(shape, dtype=float if name == "abs_error_sum" else int)
        for name in ("n_pass", "n_pairs", "abs_error_sum", "n_gate_pairs")
    }
    for trial in range(trials):
        for column, activity in enumerate(gt):
            low, high = script.baseline_range(gt[activity], "activity")
            depths = list(rng.uniform(low, high, size=n_est[activity]))
            match = script.match_activity(depths, gt[activity])
            errors = np.abs([error for _, _, error in match.pairs])
            expected["n_pass"][trial, column] = _n_pass(match)
            expected["n_pairs"][trial, column] = len(match.pairs)
            expected["abs_error_sum"][trial, column] = float(errors.sum())
            gated = script.match_activity_gated(depths, gt[activity])
            expected["n_gate_pairs"][trial, column] = gated.n_pairs
    for name, values in expected.items():
        np.testing.assert_array_equal(getattr(draws, name), values, err_msg=name)
    np.testing.assert_array_equal(draws.n_est, [3, 1])
    assert np.all(draws.n_gate_pairs >= draws.n_pass)


def _g7_stats(script):
    draws = script.BaselineDraws(
        activities=("a",),
        n_gt=np.array([4]),
        n_pass=np.array([[0], [1], [1], [1], [2]]),
        n_pairs=np.full((5, 1), 3),
        abs_error_sum=np.full((5, 1), 1.0),
        n_gate_pairs=np.array([[0], [1], [1], [2], [3]]),
        n_est=np.array([3]),
    )
    return script.baseline_stats(draws)


def test_g7_baseline_position_p_and_mid_rank_percentile(script) -> None:
    stats = _g7_stats(script)
    assert script._baseline_gate_columns("baseline", stats, 1) == {
        "baseline_gate_recall_mean": 0.35,
        "baseline_gate_recall_p025": 0.025,
        "baseline_gate_recall_p975": 0.725,
        "baseline_gate_precision_mean": 0.4667,
        "baseline_gate_precision_p025": 0.0333,
        "baseline_gate_precision_p975": 0.9667,
        "baseline_gate_f1_mean": 0.4,
        "baseline_gate_f1_p025": 0.0286,
        "baseline_gate_f1_p975": 0.8286,
        "baseline_gate_p": 0.8,  # ">" would give 0.4
        "baseline_gate_pct": 40.0,  # without the mid-rank half: 20.0
    }
    assert script._baseline_gate_columns("baseline_global", stats, 1) == {
        "baseline_global_gate_f1_mean": 0.4,
        "baseline_global_gate_f1_p025": 0.0286,
        "baseline_global_gate_f1_p975": 0.8286,
        "baseline_global_gate_p": 0.8,
    }
    no_observation = script._baseline_gate_columns("baseline", stats, None)
    assert (no_observation["baseline_gate_p"], no_observation["baseline_gate_pct"]) == (
        "",
        "",
    )
    assert no_observation["baseline_gate_f1_mean"] == 0.4


def _g8_evaluations(script):
    groundtruth, metadata = _tables()
    gt_sets = script.build_ground_truth_sets(groundtruth, metadata)
    candidates = _c13_candidates(script)
    lines = {"01.1": 3}
    on = script.evaluate_combinations(candidates, gt_sets, lines, _args())
    off = script.evaluate_combinations(candidates, gt_sets, lines, _args(apex_qc="off"))
    return on, off


def _assert_gate_properties(row) -> None:
    n_gt, n_est, pairs = row["n_gt"], row["n_est"], row["gate_pairs"]
    assert pairs <= min(n_gt, n_est)
    if row["pass_b"] is not None:
        assert row["gate_recall"] >= row["pass_b"]
    if n_est > 0:
        assert row["gate_precision"] == round(pairs / n_est, 4)
    assert row["gate_f1"] == round(2 * pairs / (n_gt + n_est), 4)


def test_g8_summary_columns_values_and_v15_regression(script) -> None:
    assert list(script.GATE_COLUMNS) == V16_GATE_COLUMNS
    on, off = _g8_evaluations(script)
    for row in on.summary_rows + off.summary_rows:
        assert list(row) == V15_SUMMARY_COLUMNS + V16_GATE_COLUMNS
        _assert_gate_properties(row)
        globals_filled = [row[c] != "" for c in GLOBAL_GATE_COLUMNS]
        assert globals_filled == [row["variant"] == "main"] * 4, row["variant"]

    gate_fields = (
        "gate_pairs",
        "gate_recall",
        "gate_precision",
        "gate_f1",
        "gate_mae_m",
    )
    expected_gate = {
        "main": (1, 0.5, 1.0, 0.6667, 0.01),
        "noqc": (2, 1.0, 1.0, 1.0, 0.2925),
    }
    for variant, values in expected_gate.items():
        row = _overall_top_detected_row(on.summary_rows, variant)
        assert tuple(row[c] for c in gate_fields) == values, variant
    for variant, values in G8_HEAD_ROWS.items():
        row = _overall_top_detected_row(on.summary_rows, variant)
        assert tuple(row[c] for c in V15_SUMMARY_COLUMNS) == values, variant


def test_g9_activities_without_ground_truth_stay_out_of_the_gate(script) -> None:
    groundtruth, metadata = _tables()
    gt_sets = script.build_ground_truth_sets(groundtruth, metadata)
    no_gt_candidates = [
        dataclasses.replace(
            _candidate(script, "99.9", path, 0.50), qc=_qc(script, "in")
        )
        for path in ("L1", "L2")
    ]
    evaluation = script.evaluate_combinations(
        _c13_candidates(script) + no_gt_candidates,
        gt_sets,
        {"01.1": 3, "99.9": 2},
        _args(),
    )
    assert evaluation.no_gt_counts["99.9"] == 1
    row = _overall_top_detected_row(evaluation.summary_rows, "main")
    assert (row["n_est"], row["gate_pairs"], row["gate_precision"]) == (1, 1, 1.0)


@pytest.mark.parametrize(("apex_qc", "n_variant_rows"), [("on", 10), ("off", 6)])
def test_g10_md_gate_section(script, tmp_path, apex_qc, n_variant_rows) -> None:
    lines = _markdown_lines(script, tmp_path, apex_qc)
    order = [
        _index_starting_with(lines, prefix)
        for prefix in ("## 보조: X 포함", "## 게이트 보조 지표", "## 현장별")
    ]
    assert order == sorted(order)
    assert "판정·U-3에 쓰지 않음" in lines[order[1]]

    header, rows = _md_table_under(lines, "### 주 결과 (`main`")
    assert header == GATE_MAIN_TABLE_HEADER
    assert len(rows) == 6
    _, variant_rows = _md_table_under(lines, "### 변형별")
    assert len(variant_rows) == n_variant_rows
    main_header, _ = _md_table_under(lines, "## 주 결과")
    assert main_header == V15_MAIN_TABLE_HEADER


def test_g11_property_1_is_checked_at_run_time(script) -> None:
    ungated = script.match_activity([0.45, 0.85], [0.15, 0.60, 1.20])
    no_gate_pairs = script.GateMatch(n_est=2, n_gt=3, pairs=())
    with pytest.raises(RuntimeError, match="act_g11"):
        script.summarize(
            {"act_g11": ungated}, {"act_g11": 3}, {"act_g11": no_gate_pairs}
        )
    with pytest.raises(ValueError, match="activities"):
        script.summarize({"a": ungated}, {"a": 3}, {"b": no_gate_pairs})
