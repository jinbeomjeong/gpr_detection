"""M3-2 S6 (i): stage-C judge (``analysis/detection/m32_recipe_judge.py``).

Synthetic runner outputs only (no model, no stage-C result). Covers the one
aggregation function, exact (Fraction) threshold boundaries of C-1, C-2, C-6
and C-7, the empty-set guard, bootstrap determinism, the C-5 operating point,
the M6 guard and identity gate, the selection tie rule, the INJ-A flags, the
polarity rule and the blind CLI contract.
"""

import csv
import json
import re
import sys
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "detection"))
import m32_recipe_inputs as inputs
import m32_recipe_judge as judge
import m32_recipe_match as match

FOLDS = inputs.FOLDS
N_PER_CELL = 100
LINES_PER_DATASET = 10
CELLS = [
    ("4tu", "A1"),
    ("4tu", "A0"),
    ("gz_ids", "A0"),
]
METRIC_WORDS = re.compile(r"recall|delta|hits|adopt|keep_r0", re.IGNORECASE)

HitRule = Callable[[str, int, str, str, float, int, int], bool]
"""``(recipe, fold, dataset, group, snr, polarity, index) -> hit``."""


def half_hits(recipe, fold, dataset, group, snr, polarity, index) -> bool:
    return index < N_PER_CELL // 2


def plus(extra: dict[str, int]) -> HitRule:
    """r0 at 50 %, recipe ``r`` at 50 % + ``extra[r]`` hits per 100."""

    def rule(recipe, fold, dataset, group, snr, polarity, index) -> bool:
        return index < N_PER_CELL // 2 + extra.get(recipe, 0)

    return rule


def make_rows(rule: HitRule, *, mask_rule: HitRule | None = None) -> list:
    rows = []
    for recipe in inputs.STAGE_C_RECIPES:
        for fold in FOLDS:
            for dataset, group in CELLS:
                for snr in judge.SNR_STEPS:
                    for polarity in judge.POLARITIES:
                        for index in range(N_PER_CELL):
                            key = (recipe, fold, dataset, group, snr, polarity, index)
                            hit = rule(*key)
                            masked = hit if mask_rule is None else mask_rule(*key)
                            rows.append(
                                judge.MatchRow(
                                    recipe=recipe,
                                    fold=fold,
                                    injection_id=f"{dataset}:{group}:{snr}:{polarity}:{index}",
                                    line_id=f"{dataset}-L{index % LINES_PER_DATASET}",
                                    dataset=dataset,
                                    group=group,
                                    snr_db=snr,
                                    polarity=polarity,
                                    k=judge.K_MAIN,
                                    in_distribution=True,
                                    bottom_stratum=index % 10 == 0,
                                    in_horizontal_box_zone=False,
                                    hit=hit,
                                    hit_any_class=hit,
                                    hit_mask=("1" if masked else "0") * 71,
                                )
                            )
    return rows


def conf_rows(totals: dict[str, int]) -> list[dict[str, str]]:
    rows = []
    for recipe, total in totals.items():
        row = {
            "recipe": recipe,
            "fold": "0",
            "line_id": "x",
            "dataset": "4tu",
            "role": "background",
        }
        for index, level in enumerate(match.conf_grid()):
            row[f"c{level:.2f}"] = str(max(0, total - index))
        rows.append(row)
    return rows


def apex_rows(hits_per_recipe: dict[str, list[list[bool]]]) -> list[dict[str, str]]:
    """``hits[apex][fold]`` per recipe -> ``apex_hits.csv`` rows."""
    rows = []
    for recipe, table in hits_per_recipe.items():
        for apex_index, per_fold in enumerate(table):
            for fold, hit in enumerate(per_fold):
                rows.append(
                    {
                        "recipe": recipe,
                        "fold": str(fold),
                        "blind_id": f"G{apex_index}",
                        "line_id": "01.1/Path1",
                        "trace": "1",
                        "sample": "1.0",
                        "hit": str(hit),
                    }
                )
    return rows


def d3_hits() -> list[list[bool]]:
    """41 apexes: sum 90, any fold 30, fold 0 15 (the D3 values)."""
    table = [[False] * 5 for _ in range(41)]
    for apex in range(15):  # fold 0 hits
        table[apex][0] = True
    for apex in range(30):  # any fold: apexes 15-29 hit by fold 1
        table[apex][1] = True
    remaining = 90 - sum(map(sum, table))
    for apex in range(30):
        for fold in (2, 3, 4):
            if remaining and not table[apex][fold]:
                table[apex][fold] = True
                remaining -= 1
    assert sum(map(sum, table)) == 90
    return table


def gate_record(*, passed: bool = True) -> dict:
    """The P5a batch-1 ``identity_gate`` record (D3 values when passed)."""
    expected = judge.d3_values()
    observed = dict(expected) if passed else {**expected, "sum_hits": 89}
    return {"observed": observed, "expected": expected, "pass": passed}


PASSED_GATE = object()
"""Sentinel: ``run_data`` default, a passed batch-1 gate."""


def run_data(
    rows, *, conf=None, apexes=None, boxes=(), gate=PASSED_GATE
) -> judge.RunData:
    return judge.RunData(
        matches=rows,
        conf_counts=conf_rows(conf or {"r0": 10, "r5": 10, "r6": 10}),
        boxes=list(boxes),
        apex_hits=apex_rows(apexes or dict.fromkeys(inputs.STAGE_C_RECIPES, d3_hits())),
        lines=[],
        sha256={},
        gate_record=gate_record() if gate is PASSED_GATE else gate,
    )


def counts_of(rows):
    return judge.total_counts(judge.count_cells(rows, judge.main_stratum))


# ------------------------------------------------------------ aggregation


def test_fold_recall_averages_polarity_snr_groups_then_datasets() -> None:
    counts = {}
    recalls = {
        ("4tu", "A1"): (1, 10),
        ("4tu", "A0"): (3, 10),
        ("gz_ids", "A0"): (6, 10),
    }
    for fold in FOLDS:
        for (dataset, group), (hits, n) in recalls.items():
            for snr in judge.SNR_STEPS:
                for polarity in judge.POLARITIES:
                    key = (fold, dataset, group, snr, polarity, False)
                    counts[("r0", key)] = (hits, n)

    values = judge.fold_recalls(counts, "r0", judge.MAIN_SCOPE, FOLDS)

    # 4TU = (0.1 + 0.3) / 2 = 0.2, then (0.2 + 0.6) / 2 = 0.4 (not 1/3 of all).
    assert values == [Fraction(2, 5)] * 5


def test_c1_boundary_is_exact_three_percentage_points() -> None:
    ji = judge.prepare(run_data(make_rows(plus({"r5": 3, "r6": 2}))))

    r5 = judge.injection_criteria(ji, "r5", judge.MAIN_SCOPE)["C-1"]
    r6 = judge.injection_criteria(ji, "r6", judge.MAIN_SCOPE)["C-1"]

    assert r5["value"]["exact"] == "3/100" and r5["pass"]
    assert r6["value"]["exact"] == "1/50" and not r6["pass"]


def test_empty_cell_makes_every_injection_criterion_not_evaluable() -> None:
    rows = [
        row
        for row in make_rows(plus({"r5": 10}))
        if not (row.dataset == "gz_ids" and row.snr_db == 18.0 and row.polarity == -1)
    ]
    data = run_data(rows)
    verdict = judge.judge(data)

    criteria = verdict["candidates"]["r5"]["criteria_by_polarity"]["mean"]
    for name in ("C-1", "C-2", "C-3", "C-4", "C-5"):
        assert criteria[name]["pass"] is False, name
        assert criteria[name].get("reason") == "not_evaluable", name
    assert verdict["decision"] == "keep_r0"


def test_zero_injections_never_pass() -> None:
    assert judge.difference({}, "r5", judge.MAIN_SCOPE, FOLDS).mean is None
    assert judge._criterion(None, True, "x")["pass"] is False


# --------------------------------------------------------------- C-2, C-3


def _per_fold_rule(per_fold_extra: dict[int, int]) -> HitRule:
    def rule(recipe, fold, dataset, group, snr, polarity, index) -> bool:
        extra = per_fold_extra[fold] if recipe == "r5" else 0
        return index < N_PER_CELL // 2 + extra

    return rule


@pytest.mark.parametrize(
    ("per_fold", "passes"),
    [
        ({0: 5, 1: 5, 2: 5, 3: 5, 4: -2}, True),  # worst fold exactly -2.0 %p
        ({0: 5, 1: 5, 2: 5, 3: 5, 4: -3}, False),
        ({0: 5, 1: 5, 2: 5, 3: 0, 4: 0}, False),  # 3/5 positive
    ],
)
def test_c2_fold_consistency(per_fold: dict[int, int], passes: bool) -> None:
    ji = judge.prepare(run_data(make_rows(_per_fold_rule(per_fold))))
    assert judge.injection_criteria(ji, "r5", judge.MAIN_SCOPE)["C-2"]["pass"] is passes


def test_bootstrap_is_deterministic_and_positive_for_a_uniform_gain() -> None:
    ji = judge.prepare(run_data(make_rows(plus({"r5": 5}))))
    args = (ji.main_per_line, ji.line_datasets, "r5", judge.MAIN_SCOPE, FOLDS)

    first = judge.bootstrap_lower_bound(*args, replicates=200)
    second = judge.bootstrap_lower_bound(*args, replicates=200)
    other_seed = judge.bootstrap_lower_bound(*args, replicates=200, seed=1)

    assert first == second
    assert first["lower"] > 0 and first["n_undefined"] == 0
    assert first["index"] == 4  # floor(0.025 * 199)
    assert other_seed["replicates"] == 200


def test_bootstrap_lower_bound_is_not_above_zero_without_a_gain() -> None:
    ji = judge.prepare(run_data(make_rows(half_hits)))
    result = judge.bootstrap_lower_bound(
        ji.main_per_line, ji.line_datasets, "r5", judge.MAIN_SCOPE, FOLDS, replicates=50
    )
    assert result["lower"] == 0


def test_undefined_replicates_count_against_the_bound() -> None:
    values = [None, Fraction(1, 2), Fraction(-1, 4)]
    result = judge.order_statistics(values, 3)
    assert result["lower"] is None and result["n_undefined"] == 1
    assert result["upper"] is None  # +inf: no evidence of harm
    defined = judge.order_statistics([Fraction(i) for i in range(41)], 41)
    assert (defined["lower"], defined["upper"]) == (Fraction(1), Fraction(39))


def test_one_line_resample_weights_every_fold_alike() -> None:
    per_line = {
        f"L{index}": {
            ("r0", (fold, "4tu", "A1", 0.0, 1, False)): [index % 3, 3] for fold in FOLDS
        }
        for index in range(8)
    }
    datasets = {f"L{index}": "4tu" for index in range(8)}
    replicates = list(judge.replicate_counts(per_line, datasets, ["4tu"], 25, 7))
    assert len(replicates) == 25
    for counts in replicates:
        per_fold = {
            counts[("r0", (fold, "4tu", "A1", 0.0, 1, False))] for fold in FOLDS
        }
        assert len(per_fold) == 1  # identical lines -> identical fold counts
    again = list(judge.replicate_counts(per_line, datasets, ["4tu"], 25, 7))
    assert replicates == again


# ------------------------------------------------------------- C-4, C-5, C-6


def test_c4_needs_each_dataset_non_negative() -> None:
    def rule(recipe, fold, dataset, group, snr, polarity, index):
        extra = (10 if dataset == "4tu" else -1) if recipe == "r5" else 0
        return index < N_PER_CELL // 2 + extra

    ji = judge.prepare(run_data(make_rows(rule)))
    result = judge.injection_criteria(ji, "r5", judge.MAIN_SCOPE)
    assert result["C-1"]["pass"]
    assert not result["C-4"]["pass"]
    assert result["C-4"]["by_dataset"]["gz_ids"]["exact"] == "-1/100"


def test_operating_point_is_the_first_conf_with_no_more_boxes_than_r0() -> None:
    totals = {"r0": [10] * 71, "r5": [13, 12, 10, 9] + [0] * 67, "r6": [99] * 71}
    assert judge.operating_index(totals, "r5") == 2
    assert judge.operating_index(totals, "r0") == 0
    assert judge.operating_index(totals, "r6") is None


def test_c5_reads_the_hit_mask_at_the_matched_conf() -> None:
    def mask(recipe, fold, dataset, group, snr, polarity, index):
        return index < N_PER_CELL // 2 + (1 if recipe == "r5" else 0)

    rows = make_rows(plus({"r5": 10}), mask_rule=mask)
    ji = judge.prepare(run_data(rows, conf={"r0": 10, "r5": 13, "r6": 30}))

    c5 = judge.injection_criteria(ji, "r5", judge.MAIN_SCOPE)["C-5"]
    assert c5["conf"] == 0.28 and c5["pass"]
    assert c5["value"]["exact"] == "1/100"
    c5_r6 = judge.injection_criteria(ji, "r6", judge.MAIN_SCOPE)["C-5"]
    assert c5_r6["conf"] == 0.45 and not c5_r6["pass"]


def test_c6_bottom_stratum_boundary() -> None:
    def rule(recipe, fold, dataset, group, snr, polarity, index):
        if recipe == "r5" and index % 10 == 0 and index < 30:
            return False  # 3 of the 10 bottom rows of each cell are lost
        return index < N_PER_CELL // 2 + (5 if recipe == "r5" else 0)

    ji = judge.prepare(run_data(make_rows(rule)))
    c6 = judge.injection_criteria(ji, "r5", judge.MAIN_SCOPE)["C-6"]
    # r0 bottom hits 5/10 per cell (index 0-40), r5 3/10 (30, 40, 50): -20 %p.
    assert c6["value"]["exact"] == "-1/5" and not c6["pass"]


def _bottom_rows(recipe: str, hits_per_fold: int) -> list:
    return [
        judge.MatchRow(
            recipe=recipe,
            fold=fold,
            injection_id=f"b{index}",
            line_id="L0",
            dataset="4tu",
            group="A1",
            snr_db=0.0,
            polarity=1,
            k=judge.K_MAIN,
            in_distribution=False,
            bottom_stratum=True,
            in_horizontal_box_zone=False,
            hit=index < hits_per_fold,
            hit_any_class=False,
            hit_mask="0" * 71,
        )
        for fold in FOLDS
        for index in range(100)
    ]


@pytest.mark.parametrize(("r5_hits", "passes"), [(47, True), (46, False)])
def test_c6_threshold_is_exact_minus_three_points(r5_hits: int, passes: bool) -> None:
    rows = _bottom_rows("r0", 50) + _bottom_rows("r5", r5_hits)
    ji = judge.prepare(run_data(rows))
    c6 = judge._c6(ji, "r5", judge.MAIN_SCOPE)
    assert c6["pass"] is passes
    assert c6["n_injections_fold0"] == 100


# ----------------------------------------------------------- C-7, C-8


def _m4_boxes(recipe: str, per_fold: dict[int, int], lines: int = 1) -> list[dict]:
    rows = []
    for fold, count in per_fold.items():
        for index in range(count):
            rows.append(
                {
                    "recipe": recipe,
                    "fold": str(fold),
                    "line_id": f"L{index % lines}",
                    "orientation": "original",
                    "m4": "M4a",
                    "m4c": "False",
                }
            )
    return rows


def test_c7_fold_limit_is_three_or_ten_percent() -> None:
    base = judge.m4_counts(
        _m4_boxes("r0", {0: 40, 1: 10, 2: 0, 3: 0, 4: 0}), "r0", FOLDS
    )
    same_total = judge.m4_counts(
        _m4_boxes("r5", {0: 34, 1: 13, 2: 3, 3: 0, 4: 0}), "r5", FOLDS
    )
    over = judge.m4_counts(
        _m4_boxes("r5", {0: 30, 1: 14, 2: 0, 3: 0, 4: 0}), "r5", FOLDS
    )

    assert judge.c7_check(same_total, base)["pass"]  # fold 1 limit 13, fold 2 limit 3
    assert not judge.c7_check(over, base)["checks"]["every_fold_within_limit"]


def test_c7_line_count_and_m4c_ratio() -> None:
    base = judge.M4Counts({0: 1}, total=1, lines=1, m4c_total=10)
    two_more_lines = judge.M4Counts({0: 1}, total=1, lines=3, m4c_total=11)
    assert judge.c7_check(two_more_lines, base)["checks"] == {
        "total_le_r0": True,
        "lines_le_r0_plus_1": False,
        "every_fold_within_limit": True,
        "m4c_le_110pct": True,
    }
    many_m4c = judge.M4Counts({0: 1}, total=1, lines=1, m4c_total=12)
    assert not judge.c7_check(many_m4c, base)["checks"]["m4c_le_110pct"]


def test_identity_gate_and_m6_guard() -> None:
    table = d3_hits()
    baseline = judge.m6_counts(apex_rows({"r0": table}), "r0")
    assert judge.d3_comparison(baseline)["equal"]
    assert judge.identity_gate(gate_record())["pass"]
    assert not judge.identity_gate(gate_record(passed=False))["pass"]
    assert not judge.identity_gate(None)["pass"]
    # The judge checks the recorded observation itself, not the runner's flag.
    forged = {**gate_record(passed=False), "pass": True}
    assert not judge.identity_gate(forged)["pass"]

    worse = [list(row) for row in table]
    for apex in range(7):  # 7 hits fewer than r0
        worse[apex][0] = False
    candidate = judge.m6_counts(apex_rows({"r5": worse}), "r5")
    assert candidate["sum_hits"] == 83
    assert not judge.c8_check(candidate, baseline)["pass"]

    shifted = [list(row) for row in table]
    shifted[0][0] = False  # one hit fewer: still >= r0 - 6
    assert judge.c8_check(judge.m6_counts(apex_rows({"r5": shifted}), "r5"), baseline)[
        "pass"
    ]


def test_failed_batch1_gate_invalidates_the_verdict() -> None:
    for gate in (gate_record(passed=False), None):
        data = run_data(make_rows(plus({"r5": 10})), gate=gate)
        verdict = judge.judge(data)
        assert verdict["decision"] == "invalid_identity_gate"
        assert verdict["selection"]["chosen"] is None


def test_batch_path_difference_from_d3_is_reported_not_failed() -> None:
    """Plan C-8: batch 64 vs D3 is disclosed; the batch-1 gate decides."""
    batch_r0 = d3_hits()
    batch_r0[0][0] = False  # batch path: one apex-fold hit fewer than D3
    data = run_data(
        make_rows(plus({"r5": 10})),
        apexes={"r0": batch_r0, "r5": d3_hits(), "r6": d3_hits()},
    )
    verdict = judge.judge(data)
    assert verdict["identity_gate"]["pass"]
    assert verdict["decision"] == "adopt_r5"
    report = verdict["c8_batch_path_vs_d3"]
    assert not report["equal"]
    assert report["differences"] == {
        "n_apexes": 0,
        "sum_hits": -1,
        "any_fold": 0,
        "fold0": -1,
    }
    assert report["n_differing_values"] == 2


# ------------------------------------------------- selection, INJ-A, polarity


def _candidate(delta: Fraction, passed: bool = True) -> dict:
    return {"pass": passed, "delta_exact": delta}


def test_tie_below_half_a_point_goes_to_the_smaller_stage_a_distance() -> None:
    tie = judge.select(
        {"r5": _candidate(Fraction(5, 100)), "r6": _candidate(Fraction(546, 10000))}
    )
    assert tie["chosen"] == "r5" and tie["tie_set"] == ["r5", "r6"]
    clear = judge.select(
        {"r5": _candidate(Fraction(5, 100)), "r6": _candidate(Fraction(55, 1000))}
    )
    assert clear["chosen"] == "r6" and clear["tie_set"] == ["r6"]
    none = judge.select({"r5": _candidate(Fraction(1), False)})
    assert none["chosen"] == "r0"


@pytest.mark.parametrize(
    ("a1", "a0", "leaders_a1", "leaders_a0", "expected"),
    [
        (Fraction(5, 100), Fraction(4, 100), {"r5"}, {"r5"}, (False, False, False)),
        (Fraction(5, 100), Fraction(-1, 100), {"r5"}, {"r5"}, (True, False, True)),
        (Fraction(5, 100), Fraction(4, 100), {"r5"}, {"r6"}, (False, True, False)),
        (Fraction(6, 100), Fraction(3, 100), {"r5"}, {"r5"}, (False, False, True)),
        (None, Fraction(3, 100), None, {"r5"}, (True, True, True)),
    ],
)
def test_inj_a_flags(a1, a0, leaders_a1, leaders_a0, expected) -> None:
    flags = judge.a1_a0_flags(a1, a0, leaders_a1, leaders_a0)
    assert tuple(flags.values()) == expected


def test_a1_a0_split_holds_the_adoption() -> None:
    def rule(recipe, fold, dataset, group, snr, polarity, index):
        extra = {"r5": 10, "r6": 0}.get(recipe, 0)
        if group == "A0" and dataset == "4tu" and recipe == "r5":
            extra = 1
        return index < N_PER_CELL // 2 + extra

    verdict = judge.judge(run_data(make_rows(rule)))
    assert verdict["candidates"]["r5"]["pass"]
    assert verdict["inj_a"]["flags"]["iii_gap_ge_3pp"]
    assert verdict["decision"] == "hold_r5_ask_inj_a"


def test_uniform_gain_is_adopted_and_flags_are_recorded() -> None:
    verdict = judge.judge(run_data(make_rows(plus({"r5": 6, "r6": 5}))))
    assert verdict["decision"] == "adopt_r5"
    assert verdict["b0_direction"]["r6"]["delta_positive"] is True
    assert verdict["samples"]["positions_per_snr_main"]["0"] == 3 * N_PER_CELL


def test_each_polarity_rule_needs_both_polarities() -> None:
    def rule(recipe, fold, dataset, group, snr, polarity, index):
        extra = 10 if recipe == "r5" and polarity == 1 else 0
        return index < N_PER_CELL // 2 + extra

    ji = judge.prepare(run_data(make_rows(rule)))
    mean_rule = judge.evaluate_candidate(ji, "r5", "mean")
    each_rule = judge.evaluate_candidate(ji, "r5", "each")
    assert mean_rule["criteria_by_polarity"]["mean"]["C-1"]["pass"]
    assert not each_rule["criteria_by_polarity"]["p-1"]["C-1"]["pass"]
    assert not each_rule["pass"]


# ------------------------------------------------------------------- CLI


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else ["recipe"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_run(root: Path, rows: list) -> None:
    for stage in ("uninjected", "injected"):
        (root / stage).mkdir(parents=True)
        pi = {"4tu/A1": 0.1, "4tu/A0": 0.1, "gz_ids/A0": 0.1}
        summary = {
            "status": "ok",
            "full_run": True,
            **judge.DEFAULT_DECISIONS,
            "pi_bottom": pi,
        }
        if stage == "uninjected":
            summary["identity_gate"] = gate_record()
        (root / stage / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    gate = {"batch_size": 1, "apex_rows": [], "result": gate_record()}
    (root / "uninjected" / "gate.json").write_text(json.dumps(gate), encoding="utf-8")
    data = run_data(rows)
    for recipe in inputs.STAGE_C_RECIPES:
        for fold in FOLDS:
            folder = f"{recipe}/fold{fold}"
            matches = [
                {
                    "recipe": r.recipe,
                    "fold": r.fold,
                    "injection_id": r.injection_id,
                    "line_id": r.line_id,
                    "dataset": r.dataset,
                    "model_group": r.group,
                    "snr_db": r.snr_db,
                    "polarity": r.polarity,
                    "k": r.k,
                    "in_distribution": r.in_distribution,
                    "bottom_stratum": r.bottom_stratum,
                    "in_horizontal_box_zone": r.in_horizontal_box_zone,
                    "hit": r.hit,
                    "hit_any_class": r.hit_any_class,
                    "hit_mask": r.hit_mask,
                }
                for r in rows
                if r.recipe == recipe and r.fold == fold
            ]
            _write_csv(root / "injected" / folder / "inj_matches.csv", matches)
            for name, table in (
                ("conf_counts", data.conf_counts),
                ("apex_hits", data.apex_hits),
            ):
                selected = [
                    row
                    for row in table
                    if row["recipe"] == recipe and row["fold"] == str(fold)
                ]
                _write_csv(root / "uninjected" / folder / f"{name}.csv", selected)
            _write_csv(root / "uninjected" / folder / "boxes.csv", [])
    _write_csv(root / "uninjected" / "lines.csv", [])


def test_cli_writes_only_the_verdict_and_prints_no_metric(tmp_path, capsys) -> None:
    _write_run(tmp_path, make_rows(plus({"r5": 6})))

    code = judge.main(["--run-root", str(tmp_path)])

    lines = capsys.readouterr().out.strip().splitlines()
    final = json.loads(lines[-1])
    assert code == 0 and final["status"] == "ok"
    assert set(final) == {
        "status",
        "verdict",
        "g_in_cells",
        "g_in_not_evaluable",
        "elapsed_s",
    }
    assert not any(METRIC_WORDS.search(line) for line in lines)
    verdict = json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["schema"] == judge.SCHEMA and "decision" in verdict
    assert verdict["identity_gate"]["pass"]
    assert "uninjected/gate.json" in verdict["inputs_sha256"]


def _rewrite_json(path: Path, change: Callable[[dict], None]) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize(
    "corrupt",
    [
        "no_gate_json",
        "no_summary_record",
        "batch_size_64",
        "records_disagree",
    ],
)
def test_cli_refuses_a_missing_or_foreign_batch1_gate(
    tmp_path, capsys, corrupt: str
) -> None:
    _write_run(tmp_path, make_rows(plus({"r5": 6})))
    gate_path = tmp_path / "uninjected" / "gate.json"
    summary_path = tmp_path / "uninjected" / "summary.json"
    if corrupt == "no_gate_json":
        gate_path.unlink()
    elif corrupt == "no_summary_record":
        _rewrite_json(summary_path, lambda doc: doc.pop("identity_gate"))
    elif corrupt == "batch_size_64":
        _rewrite_json(gate_path, lambda doc: doc.update(batch_size=64))
    else:
        _rewrite_json(
            gate_path, lambda doc: doc.update(result=gate_record(passed=False))
        )

    code = judge.main(["--run-root", str(tmp_path)])

    final = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 2 and final["status"] == "input_error"
    assert not (tmp_path / "verdict.json").exists()


def _in_or_out_rows(r5_in: int, r5_out: int) -> list:
    """Out-of-distribution rows from index 50 on; r5 gains per stratum."""

    def rule(recipe, fold, dataset, group, snr, polarity, index):
        if recipe != "r5":
            return index % 50 < 25
        gain = r5_in if index < 50 else r5_out
        return index % 50 < 25 + gain

    rows = make_rows(rule)
    return [
        replace(row, in_distribution=int(row.injection_id.rsplit(":", 1)[1]) < 50)
        for row in rows
    ]


def test_in_or_out_path_uses_every_size_and_the_in_guard() -> None:
    from dataclasses import replace as _replace

    rows = _in_or_out_rows(r5_in=-2, r5_out=6)
    data = _replace(
        run_data(rows),
        decisions={"primary_stratum": "in_or_out", "bottom_rule": "point"},
    )
    ji = judge.prepare(data)
    criteria = judge.injection_criteria(ji, "r5", judge.MAIN_SCOPE)
    # All sizes: (-2 + 6) / 2 per 50 -> +2 %p ... the guard is exactly -4 %p.
    assert criteria["C-1"]["value"]["exact"] == "1/25"
    assert criteria["G-in"]["value"]["exact"] == "-1/25"
    assert not criteria["G-in"]["pass"]
    edge = _replace(
        run_data(_in_or_out_rows(r5_in=-1, r5_out=6)),
        decisions={"primary_stratum": "in_or_out", "bottom_rule": "point"},
    )
    guard = judge.injection_criteria(judge.prepare(edge), "r5", judge.MAIN_SCOPE)
    assert guard["G-in"]["value"]["exact"] == "-1/50" and guard["G-in"]["pass"]
    report = judge.stratum_report(judge.prepare(edge), "r5")
    assert report["in"]["delta"]["exact"] == "-1/50"
    assert report["out"]["delta"]["exact"] == "3/25"


def test_in_path_has_no_guard() -> None:
    ji = judge.prepare(run_data(make_rows(plus({"r5": 3}))))
    assert "G-in" not in judge.injection_criteria(ji, "r5", judge.MAIN_SCOPE)


def test_c6_upper_bound_rule_blocks_only_clear_harm() -> None:
    from dataclasses import replace as _replace

    rows = _bottom_rows("r0", 50) + _bottom_rows("r5", 40)
    harmful = _replace(
        run_data(rows),
        decisions={"primary_stratum": "in", "bottom_rule": "upper_bound"},
    )
    c6 = judge._c6(judge.prepare(harmful), "r5", judge.MAIN_SCOPE)
    assert c6["rule"] == "upper_bound" and not c6["pass"]  # -10 %p on one line
    rows = _bottom_rows("r0", 50) + _bottom_rows("r5", 46)
    mild = _replace(
        run_data(rows),
        decisions={"primary_stratum": "in", "bottom_rule": "upper_bound"},
    )
    c6 = judge._c6(judge.prepare(mild), "r5", judge.MAIN_SCOPE)
    # The point rule would fail at -4 %p; one line gives a degenerate bootstrap
    # whose upper bound is the point, so this still fails.
    assert c6["upper"]["exact"] == "-1/25" and not c6["pass"]
    rows = _bottom_rows("r0", 50) + _bottom_rows("r5", 48)
    small = _replace(
        run_data(rows),
        decisions={"primary_stratum": "in", "bottom_rule": "upper_bound"},
    )
    assert judge._c6(judge.prepare(small), "r5", judge.MAIN_SCOPE)["pass"]


def test_read_decisions_requires_every_choice() -> None:
    chosen = {"primary_stratum": "in_or_out", "bottom_rule": "upper_bound"}
    assert judge.read_decisions({**chosen, "pi_bottom": {"4tu/A1": 0.5}}) == {
        **chosen,
        "pi_bottom": {"4tu/A1": 0.5},
    }
    with pytest.raises(inputs.InputError):
        judge.read_decisions(chosen)  # pi_bottom missing -> exit 2
    with pytest.raises(inputs.InputError):
        judge.read_decisions({"primary_stratum": "in"})
    with pytest.raises(inputs.InputError):
        judge.read_decisions({"primary_stratum": "both", "bottom_rule": "point"})


# ------------------------------------------------ G-in cells, stratification


def _in_or_out(rows, pi=None) -> judge.JudgeInputs:
    decisions = {
        "primary_stratum": "in_or_out",
        "bottom_rule": "point",
        "pi_bottom": pi,
    }
    return judge.prepare(replace(run_data(rows), decisions=decisions))


def _drop_in_distribution(rows, dataset: str, group: str, keep: int) -> list:
    """Mark the cell's rows out of distribution except ``keep`` per condition."""
    changed = []
    for row in rows:
        index = int(row.injection_id.rsplit(":", 1)[1])
        if row.dataset == dataset and row.group == group and index >= keep:
            row = replace(row, in_distribution=False)
        changed.append(row)
    return changed


def test_g_in_reweights_inside_the_dataset_when_one_cell_is_invalid() -> None:
    def rule(recipe, fold, dataset, group, snr, polarity, index):
        extra = {"A1": -4, "A0": 0}[group] if recipe == "r5" and dataset == "4tu" else 0
        return index < N_PER_CELL // 2 + extra

    rows = _drop_in_distribution(make_rows(rule), "4tu", "A0", keep=9)
    guard = judge.injection_criteria(_in_or_out(rows), "r5", judge.MAIN_SCOPE)["G-in"]
    assert guard["excluded_cells"] == ["4tu/A0"]
    assert guard["cells"]["4tu/A0"]["min_n"] == 9
    # 4TU/A1 takes the whole 4TU half: (-4 %p + 0) / 2 = -2 %p, at the limit.
    assert guard["value"]["exact"] == "-1/50" and guard["pass"]
    assert guard["weights"] == {
        "4tu/A1": {"value": 0.5, "exact": "1/2"},
        "gz_ids/A0": {"value": 0.5, "exact": "1/2"},
    }


def test_g_in_fails_when_a_whole_dataset_is_invalid() -> None:
    rows = _drop_in_distribution(make_rows(plus({"r5": 5})), "gz_ids", "A0", keep=5)
    ji = _in_or_out(rows)
    guard = judge.injection_criteria(ji, "r5", judge.MAIN_SCOPE)["G-in"]
    assert guard["reason"] == "not_evaluable" and not guard["pass"]
    verdict = judge.judge(ji.data)
    assert judge.blind_cell_summary(verdict) == {
        "g_in_cells": 2,
        "g_in_not_evaluable": True,
    }


def test_stratified_recall_equals_the_pooled_one_at_the_actual_share() -> None:
    rows = make_rows(plus({"r5": 6}))
    share = Fraction(10, N_PER_CELL)  # index % 10 == 0 marks the bottom stratum
    cells = {"4tu/A1": share, "4tu/A0": share, "gz_ids/A0": share}
    ji = judge.prepare(
        replace(
            run_data(rows),
            decisions={
                "primary_stratum": "in",
                "bottom_rule": "point",
                "pi_bottom": cells,
            },
        )
    )
    stratified = judge.difference(
        ji.main_counts, "r5", judge.MAIN_SCOPE, FOLDS, pi=ji.pi
    )
    pooled = judge.difference(ji.main_counts, "r5", judge.MAIN_SCOPE, FOLDS)
    assert stratified.mean == pooled.mean
    criteria = judge.injection_criteria(ji, "r5", judge.MAIN_SCOPE)
    c1, c2 = criteria["C-1"], criteria["C-2"]
    assert c1["value"] == c1["unstratified"]["value"]
    assert c2["per_fold"] == c2["unstratified"]["per_fold"]
    assert "pass" not in c2["unstratified"]
    other = {cell: Fraction(1, 2) for cell in cells}
    reweighted = judge.difference(
        ji.main_counts, "r5", judge.MAIN_SCOPE, FOLDS, pi=other
    )
    assert reweighted.mean != pooled.mean or reweighted.mean == Fraction(6, 100)


def test_condition_recall_post_stratifies_and_falls_back_on_an_empty_stratum() -> None:
    key = (0, "4tu", "A1", 0.0, 1)
    counts = {("r0", (*key, True)): (1, 4), ("r0", (*key, False)): (6, 8)}
    pi = {"4tu/A1": Fraction(1, 2)}
    assert judge.condition_recall(counts, "r0", key, pi) == Fraction(1, 2)
    assert judge.condition_recall(counts, "r0", key, None) == Fraction(7, 12)
    only_rest = {("r0", (*key, False)): (6, 8)}
    assert judge.condition_recall(only_rest, "r0", key, pi) == Fraction(3, 4)


def test_bootstrap_resamples_lines_once_for_all_folds() -> None:
    """A shared resample differs from independent per-fold resamples."""
    per_line = {}
    for index in range(12):
        cells = {}
        for fold in FOLDS:
            for snr in judge.SNR_STEPS:
                hits = (index + fold) % 4
                cells[("r5", (fold, "4tu", "A1", snr, 1, False))] = [hits, 4]
                cells[("r0", (fold, "4tu", "A1", snr, 1, False))] = [2, 4]
        per_line[f"L{index}"] = cells
    datasets = dict.fromkeys(per_line, "4tu")
    scope = judge.Scope(("4tu",), {"4tu": ("A1",)}, (1,))
    shared = judge.bootstrap_lower_bound(
        per_line, datasets, "r5", scope, FOLDS, replicates=60, seed=3
    )
    split = {
        f"{line}|{fold}": {
            key: value for key, value in cells.items() if key[1][0] == fold
        }
        for line, cells in per_line.items()
        for fold in FOLDS
    }
    independent = judge.bootstrap_lower_bound(
        split, dict.fromkeys(split, "4tu"), "r5", scope, FOLDS, replicates=60, seed=3
    )

    def mean_delta(counts):
        return judge.difference(counts, "r5", scope, FOLDS, (0.0,)).mean

    replicates = list(judge.replicate_counts(per_line, datasets, ["4tu"], 5, 3))
    for counts in replicates:
        r0_n = {counts[("r0", (fold, "4tu", "A1", 0.0, 1, False))][1] for fold in FOLDS}
        assert r0_n == {48}  # every fold drew the same 12 lines
    assert shared["lower"] is not None and independent["lower"] is not None
    assert shared != independent
    assert mean_delta(judge.total_counts(per_line)) is not None


def test_cli_refuses_missing_or_unfinished_runs(tmp_path, capsys) -> None:
    assert judge.main(["--run-root", str(tmp_path)]) == 2
    final = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert final["status"] == "input_error"
