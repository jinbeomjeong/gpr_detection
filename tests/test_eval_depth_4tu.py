"""Unit tests of ``scripts/eval_depth_4tu.py`` (spec_eval_4tu_v1 E1-E8, B1-B13).

Analytic B-scans and small in-memory tables only; no 4TU data is read.
"""

import argparse
import itertools
from dataclasses import dataclass

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
