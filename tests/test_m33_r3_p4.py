"""M3-3 R3 P4 (``analysis/detection/m33_r3_p4.py``): runner spec v0.3.6 tests 10-12.

Synthetic positions, truths and PP outcomes only; no model, no stored
result. Each position holds one band per SNR x polarity realization (8) and
every truth is scored on folds 0-4, so one position is 40 truth-folds.
"""

import csv
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "detection"))
import m33_r3_p4 as p4
from m32_recipe_inputs import InputError

from gpr_detection.detect.base import PixelBox
from gpr_detection.io.classes import DetectionClass
from gpr_detection.io.radargram import GridBox
from gpr_detection.pipeline import ContentRows, PPAction, PPRule, PPSetId

FOLDS = (0, 1, 2, 3, 4)
SNRS = (0.0, 6.0, 12.0, 18.0)
POLARITIES = (1, -1)
CONTENT = ContentRows(y_bot_px=200.0, period_px=6.0, zero_px=10.0)
SAFE_PX = PixelBox(10.0, 80.0, 50.0, 100.0)
"""A truth box no PP rule of any set can reach (middle of the content)."""
TRUTH = GridBox(100, 139, 50, 59)
TAPER = 4
GHOST = GridBox(150, 200, 50, 59)
"""Shares the truth rows, lies right of ``[i0 - taper, i1 + taper]``."""
BAND = DetectionClass.PIPE_BAND

BoxEdit = Callable[[str, int, int], list[p4.PPBox] | None]
"""``(realization_id, position index, fold) -> boxes`` or ``None`` (default)."""


def _box(
    box_id: str,
    before: GridBox,
    action: PPAction = PPAction.NONE,
    after: GridBox | str | None = "same",
    score: float = 0.6,
) -> p4.PPBox:
    if after == "same":
        after = before
    return p4.PPBox(box_id, BAND, score, before, action, after)


def _covering(realization_id: str) -> p4.PPBox:
    return _box(f"{realization_id}/r", TRUTH)


def _inputs(
    counts: dict[str, int],
    edit: BoxEdit | None = None,
    uninjected: dict[tuple[str, int], list[p4.PPBox]] | None = None,
    pp_set: PPSetId = PPSetId.S3,
    truth_px: dict[int, PixelBox] | None = None,
    truth_boxes: dict[int, GridBox] | None = None,
) -> p4.P4Inputs:
    """``counts[stratum]`` positions, each on its own line (L0 truth = TRUTH).

    ``truth_boxes`` gives some positions another L0 truth; their default
    covering box follows it.
    """
    positions, truths, boxes = [], [], {}
    index = 0
    for stratum, n_positions in counts.items():
        for _ in range(n_positions):
            line_id = f"line{index}"
            pixel = (truth_px or {}).get(index, SAFE_PX)
            truth_box = (truth_boxes or {}).get(index, TRUTH)
            positions.append(
                p4.BandPosition(f"pos{index}", line_id, stratum, pixel, CONTENT)
            )
            for snr in SNRS:
                for polarity in POLARITIES:
                    realization = f"{line_id}:snr{snr:g}:p{polarity:+d}"
                    truths.append(
                        p4.BandTruthUnit(
                            f"{realization}/b0",
                            realization,
                            f"pos{index}",
                            line_id,
                            stratum,
                            snr,
                            polarity,
                            truth_box,
                            TAPER,
                        )
                    )
                    covering = _box(f"{realization}/r", truth_box)
                    for fold in FOLDS:
                        custom = edit(realization, index, fold) if edit else None
                        boxes[(realization, fold)] = (
                            [covering] if custom is None else custom
                        )
            index += 1
    return p4.P4Inputs(
        pp_set_id=pp_set,
        folds=FOLDS,
        positions=tuple(positions),
        truths=tuple(truths),
        injected_boxes=boxes,
        uninjected_boxes=uninjected or {},
    )


FIRST = "line0:snr0:p+1"
"""First realization of position 0 (SNR 0 dB, polarity +1)."""


def _only(realization: str, fold: int, boxes: list[p4.PPBox]) -> BoxEdit:
    def edit(realization_id: str, _index: int, fold_id: int):
        return boxes if (realization_id, fold_id) == (realization, fold) else None

    return edit


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ------------------------------------------------- 3a: L / W_eff strata


def test_lw_stratum_edges_go_up_and_use_the_literal_033() -> None:
    assert p4.LW_EDGES[0] == 0.33 and p4.LW_EDGES[0] != 1 / 3
    assert p4.lw_stratum(100, 301) == "033_05"  # 0.3322
    assert p4.lw_stratum(64, 199) == "lt033"  # W_eff = N
    assert p4.lw_stratum(40, 89) == "033_05"
    assert p4.lw_stratum(33, 100) == "033_05"  # exactly 0.33
    assert p4.lw_stratum(50, 100) == "05_067"
    assert p4.lw_stratum(67, 100) == "ge067"
    assert p4.lw_stratum(150, 1000) == "033_05"  # W_eff = 301: 0.498


def test_ids_frequency_strata_must_be_53_3_4() -> None:
    strata = {f"m{i}": "main_400" for i in range(53)}
    strata |= {"o1": "report_other", "o2": "report_other", "o3": "report_other"}
    strata |= {f"x{i}": "excluded_2ghz" for i in range(4)}
    assert p4.assert_ids_freq_strata(strata)["main_400"] == 53
    with pytest.raises(InputError):
        p4.assert_ids_freq_strata(dict(strata, m0="report_other"))
    with pytest.raises(InputError):
        p4.assert_ids_freq_strata(dict(strata, extra="other"))


def test_covered_traces_is_the_union_inside_the_span() -> None:
    truth = GridBox(10, 19, 0, 9)
    boxes = [GridBox(5, 12, 0, 9), GridBox(11, 14, 0, 9), GridBox(18, 30, 0, 9)]
    assert p4.covered_traces(truth, boxes) == 3 + 2 + 2  # 10-12, 13-14, 18-19
    assert p4.covered_traces(truth, []) == 0
    assert p4.shares_rows(truth, GridBox(0, 5, 5, 20))  # 5 of 10 rows
    assert not p4.shares_rows(truth, GridBox(0, 5, 6, 20))


# ------------------------------------------------- test 10: small strata


@pytest.mark.parametrize(
    ("counts", "mode"),
    [
        ((30, 30), p4.StratumMode.TWO_STRATA),
        ((29, 40), p4.StratumMode.POOLED),
        ((0, 12), p4.StratumMode.POOLED),
        ((0, 0), p4.StratumMode.NOT_EVALUABLE),
    ],
)
def test_small_strata_rule(counts, mode) -> None:
    assert p4.stratum_mode(dict(zip(p4.P4_STRATA, counts, strict=True))) is mode


def test_not_evaluable_comes_first_whatever_exposure_and_acts() -> None:
    empty = dict.fromkeys(p4.P4_STRATA, 0)
    assert p4.decide_p4_status(empty, 5, 5) is p4.P4Status.NOT_EVALUABLE
    result = p4.evaluate_p4(_inputs({"05_067": 2}))  # long strata only
    document = result.document
    assert document["p4_status"] == "not_evaluable"
    assert document["recall_delta"] is None and document["ghost_delta"] is None
    assert document["recall_delta_pass"] is None


# --------------------------------------- test 11: exposure follows the set


def _exposure(pp_set: PPSetId, pixel: PixelBox) -> dict:
    inputs = _inputs({"lt033": 1}, pp_set=pp_set, truth_px={0: pixel})
    return p4.evaluate_p4(inputs).document


def test_only_the_c_window_is_not_exposure_under_s3() -> None:
    bottom_in_window = PixelBox(10.0, 170.0, 50.0, 190.0)
    s3 = _exposure(PPSetId.S3, bottom_in_window)
    assert s3["pp_exposed_positions"]["total"] == 0
    assert s3["pp_c_window_positions"]["total"] == 1
    assert s3["pp_exposed_by_rule"] == {"a": 0, "b": 0, "d": 0}
    s4 = _exposure(PPSetId.S4, bottom_in_window)
    assert s4["pp_exposed_positions"]["total"] == 1
    assert s4["pp_exposed_by_rule"]["c"] == 1


@pytest.mark.parametrize(
    ("pixel", "rule"),
    [
        (PixelBox(10.0, 195.0, 50.0, 215.0), "a"),  # centre below y_bot
        (PixelBox(10.0, 190.0, 50.0, 210.0), "b"),  # straddles y_bot
        (PixelBox(10.0, 0.0, 50.0, 20.0), "d"),  # centre above y_z + 0.5 P
    ],
)
def test_s3_rules_expose_their_own_regions(pixel, rule) -> None:
    document = _exposure(PPSetId.S3, pixel)
    assert document["pp_exposed_positions"]["by_stratum"]["lt033"] == 1
    assert document["pp_exposed_by_rule"] == {
        name: int(name == rule) for name in ("a", "b", "d")
    }
    assert p4.exposed_rule(pixel, CONTENT, p4.PP_SETS[PPSetId.S3]) is PPRule(rule)
    assert p4.exposed_rule(pixel, CONTENT, p4.PP_SETS[PPSetId.S0]) is None


# ------------------------------------------- test 12: p4_status and checks


def test_no_action_is_no_effect_with_an_exact_zero(tmp_path) -> None:
    document = p4.run_p4(_inputs({"lt033": 2, "033_05": 1}), tmp_path)
    assert document["p4_status"] == "no_effect"
    assert document["p4_structural_zero"] is True
    assert document["recall_delta"] == 0.0 and repr(document["recall_delta"]) == "0.0"
    assert document["recall_delta_pass"] is True
    assert document["stratum_mode"] == "pooled"
    assert document["pp_acted_boxes"]["total"] == 0
    assert document["recall_delta_exposed"] is None
    assert document["ghost_pp_artifacts"]["n_rows"] == 0
    assert document["defects"] == []


def test_removed_pair_of_a_ghost_candidate_is_no_effect(tmp_path) -> None:
    candidate = _box("g", GHOST)
    edit = _only(FIRST, 0, [_covering(FIRST), candidate])
    pair = _box("u", GHOST, PPAction.REMOVED_D, after=None)
    inputs = _inputs({"lt033": 1}, edit, uninjected={("line0", 0): [pair]})

    document = p4.run_p4(inputs, tmp_path)

    assert document["p4_status"] == "no_effect"
    assert document["pp_acted_ghost_set"]["total"] == 1
    assert document["pp_acted_ghost_set"]["g_prime_unique_uninjected"] == 1
    assert document["pp_acted_boxes_all"]["total"] == 1
    assert document["pp_acted_boxes"]["total"] == 0
    assert document["ghost_delta"] == 1
    (row,) = _rows(tmp_path / p4.ARTIFACT_FILE)
    assert (row["box_id"], row["candidate_pp_action"]) == ("g", "none")
    assert (row["pair_box_id"], row["pair_pp_action"]) == ("u", "removed_d")
    assert row["explained"] == "True"
    assert document["ghost_pp_artifacts"]["unexplained_zero"] is True


def test_removed_box_that_pairs_nothing_is_no_effect(tmp_path) -> None:
    lonely = _box("u", GridBox(300, 340, 50, 59), PPAction.REMOVED_D, after=None)
    inputs = _inputs({"lt033": 1}, uninjected={("line0", 0): [lonely]})

    document = p4.run_p4(inputs, tmp_path)

    assert document["p4_status"] == "no_effect"
    assert document["pp_acted_boxes_all"] == {"total": 1, "by_rule": {"removed_d": 1}}
    assert document["pp_acted_ghost_set"]["total"] == 0
    assert document["pp_acted_boxes"]["total"] == 0


def test_a_clip_in_the_recall_set_keeps_the_coverage(tmp_path) -> None:
    clipped = _box("c", GridBox(100, 139, 50, 70), PPAction.CLIPPED_B,
                   after=GridBox(100, 139, 50, 65))  # fmt: skip
    inputs = _inputs({"lt033": 1}, _only(FIRST, 2, [clipped]))

    document = p4.run_p4(inputs, tmp_path)

    assert document["p4_status"] == "no_effect"
    assert document["p4_structural_zero"] is True
    record = document["pp_acted_clip_recall_set"]
    assert record["total"] == 1 and record["coverage_unchanged"] is True


def test_a_removal_in_the_recall_set_is_acted_unexposed(tmp_path) -> None:
    removed = _box("r", TRUTH, PPAction.REMOVED_D, after=None)
    inputs = _inputs({"lt033": 1}, _only(FIRST, 0, [removed]))

    document = p4.run_p4(inputs, tmp_path)

    assert document["p4_status"] == "acted_unexposed"
    assert document["pp_acted_boxes"] == {
        "by_stratum": {"lt033": 1, "033_05": 0},
        "total": 1,
    }
    assert document["pp_acted_by_rule"]["removed_d"] == 1
    assert document["pp_acted_lines"] == 1
    assert document["recall_delta"] < 0 and document["recall_delta_exposed"] is None
    (row,) = _rows(tmp_path / p4.ACTED_FILE)
    assert (row["set"], row["pp_action"], row["fold"]) == ("R", "removed_d", "0")
    assert document["p4_pp_acted"]["n_rows"] == 1


def test_a_clip_that_shrinks_the_coverage_is_a_defect(tmp_path) -> None:
    """Truth bottom placed below ``y_bot`` (artificial): the clip cuts its rows."""
    clipped = _box("c", TRUTH, PPAction.CLIPPED_B, after=GridBox(100, 139, 50, 51))
    inputs = _inputs({"lt033": 1}, _only(FIRST, 0, [clipped]))

    with pytest.raises(p4.P4DefectError) as raised:
        p4.run_p4(inputs, tmp_path)

    document = raised.value.document
    assert document["pp_acted_clip_recall_set"]["coverage_unchanged"] is False
    assert any("PP-b clip changed" in item for item in document["defects"])
    assert (tmp_path / p4.ACTED_FILE).is_file()  # files are written first


def test_an_unexplained_new_ghost_is_a_defect(tmp_path) -> None:
    candidate = _box("g", GHOST)
    edit = _only(FIRST, 0, [_covering(FIRST), candidate])
    moved = _box("u", GHOST, after=GridBox(150, 160, 50, 59))  # changed, no action
    inputs = _inputs({"lt033": 1}, edit, uninjected={("line0", 0): [moved]})

    with pytest.raises(p4.P4DefectError) as raised:
        p4.run_p4(inputs, tmp_path)

    defects = raised.value.document["defects"]
    assert any(item.startswith("unexplained ghost") for item in defects)
    (row,) = _rows(tmp_path / p4.ARTIFACT_FILE)
    assert row["explained"] == "False"
    assert raised.value.document["ghost_pp_artifacts"]["n_unexplained"] == 1


def test_no_effect_with_a_changed_recall_box_is_a_defect(tmp_path) -> None:
    shrunk = _box("r", TRUTH, after=GridBox(100, 119, 50, 59))  # no action
    inputs = _inputs({"lt033": 1}, _only(FIRST, 0, [shrunk]))

    with pytest.raises(p4.P4DefectError) as raised:
        p4.run_p4(inputs, tmp_path)

    document = raised.value.document
    assert document["p4_status"] == "no_effect"
    assert document["p4_structural_zero"] is False
    assert any("no_effect but recall_delta" in item for item in document["defects"])
    assert any("changed without an action" in item for item in document["defects"])


def test_exposure_makes_the_status_evaluated(tmp_path) -> None:
    inputs = _inputs({"lt033": 2}, truth_px={0: PixelBox(10.0, 195.0, 50.0, 215.0)})
    document = p4.run_p4(inputs, tmp_path)
    assert document["p4_status"] == "evaluated"
    assert document["n_exposed_positions"] == 1
    assert document["recall_delta_exposed"] == 0.0


# ----------------------------------------- test 12: the -1.0 %p boundary


def _losing(n_lost: int) -> BoxEdit:
    """Remove the covering box of the first ``n_lost`` truth-folds.

    Position 0 (``lt033``) owns 40 truth-folds; the 41st is position 1.
    """
    order = [(index, snr, polarity, fold)
             for index in (0, 1)
             for snr in SNRS
             for polarity in POLARITIES
             for fold in FOLDS]  # fmt: skip
    lost = set(order[:n_lost])

    def edit(realization_id: str, index: int, fold: int):
        _, snr_text, polarity_text = realization_id.split(":")
        key = (index, float(snr_text[3:]), int(polarity_text[1:]), fold)
        if key not in lost:
            return None
        return [_box("r", TRUTH, PPAction.REMOVED_A, after=None)]

    return edit


@pytest.mark.parametrize(("n_lost", "passes"), [(40, True), (41, False)])
def test_minus_one_point_zero_passes_and_one_more_loss_fails(n_lost, passes) -> None:
    inputs = _inputs({"lt033": 50, "033_05": 50}, _losing(n_lost))

    document = p4.evaluate_p4(inputs).document

    assert document["stratum_mode"] == "two_strata"
    assert document["p4_status"] == "acted_unexposed"
    expected = -1.25 / 50 * n_lost
    assert document["recall_delta"] == pytest.approx(expected, abs=1e-12)
    assert document["recall_delta_pass"] is passes
    assert p4.recall_delta_passes(-1.0) is True
    assert p4.recall_delta_passes(-1.0 - 2e-9) is False


def test_positions_mode_does_not_change_the_counting_model() -> None:
    """Pooled (29 + 40): one realization-fold weighs 1 / (69 x 40)."""
    inputs = _inputs({"lt033": 29, "033_05": 40}, _losing(1))
    document = p4.evaluate_p4(inputs).document
    assert document["stratum_mode"] == "pooled"
    assert document["recall_delta"] == pytest.approx(-100 / (69 * 40), abs=1e-12)


def test_the_boundary_fixture_has_equal_lengths() -> None:
    """Runner spec §7 12 (cv, decision (A)): -1.0 exactly needs equal lengths."""
    inputs = _inputs({"lt033": 50, "033_05": 50}, _losing(40))
    assert {truth.length_traces for truth in inputs.truths} == {TRUTH.n_traces}


def test_a_longer_lost_band_weighs_its_length_over_the_cell_mean() -> None:
    """Decision (A): losing one truth-fold costs -1.25 (L / L_bar) / n %p."""
    long_truth = GridBox(100, 179, 50, 59)  # L 80 among L 40
    inputs = _inputs(
        {"lt033": 50, "033_05": 50},
        lambda realization_id, index, fold: (
            [_box("r", long_truth, PPAction.REMOVED_A, after=None)]
            if index == 0
            else None
        ),
        truth_boxes={0: long_truth},
    )

    document = p4.evaluate_p4(inputs).document

    cell_mean_length = (49 * 40 + 80) / 50  # L_bar of each lt033 cell
    per_truth_fold = -1.25 * (80 / cell_mean_length) / 50
    assert document["recall_delta"] == pytest.approx(40 * per_truth_fold, abs=1e-12)
    assert document["recall_delta_pass"] is False  # -1.96 %p


def test_uninjected_pairs_are_cut_to_the_inferred_traces() -> None:
    """Item 3: a box cut at the end of the inferred range keeps its pair."""
    cut = _box("g", GridBox(150, 199, 50, 59))  # injected run ends at 199
    whole = _box("u", GridBox(150, 400, 50, 59))  # un-injected, whole line
    assert p4.grid_iou(cut.before, whole.before) < p4.GHOST_PAIR_IOU
    uncut = p4.evaluate_unit(_truth(), 0, [cut], [whole])
    clipped = p4.evaluate_unit(_truth(), 0, [cut], [whole], pair_range=(0, 199))
    assert uncut.ghosts_before == {"g"}  # false ghost without the cut
    assert clipped.ghosts_before == frozenset()
    assert p4.clip_to_traces(GridBox(10, 20, 0, 1), 30, 40) is None
    (kept,) = p4.clip_pair_pool([whole], (0, 199))
    assert kept.before == GridBox(150, 199, 50, 59) and kept.box_id == "u"


def _truth(stratum: str = "lt033", box: GridBox = TRUTH, suffix: str = "0"):
    return p4.BandTruthUnit(
        f"R/b{suffix}", "R", f"p{suffix}", "L", stratum, 0.0, 1, box, TAPER
    )


def test_ghost_delta_counts_a_box_once_across_the_two_strata() -> None:
    """Item 5: a ghost of two truths of different strata is one box."""
    first = _truth("lt033")
    second = _truth("033_05", GridBox(300, 339, 50, 59), "1")
    ghost = _box("g", GridBox(150, 200, 50, 59))
    units = [p4.evaluate_unit(truth, 0, [ghost], []) for truth in (first, second)]
    assert all(unit.ghosts_before == {"g"} for unit in units)
    assert p4.ghost_count(units, after=False) == 1


def test_ghosts_count_distinct_boxes_and_union_length_per_realization() -> None:
    """cv 해석 2026-11-16: one box ghosting two truths of a realization is 1."""
    first = p4.BandTruthUnit("R/b0", "R", "p0", "L", "lt033", 0.0, 1, TRUTH, TAPER)
    second = p4.BandTruthUnit(
        "R/b1", "R", "p1", "L", "lt033", 0.0, 1, GridBox(300, 339, 50, 59), TAPER
    )
    ghost = _box("g", GridBox(150, 200, 50, 59))
    partly = _box("h", GridBox(90, 145, 50, 59))  # A: 90-95, 144-145 outside
    units = [
        p4.evaluate_unit(truth, 0, [ghost, partly], []) for truth in (first, second)
    ]
    assert sum(len(unit.ghosts_before) for unit in units) == 4  # (truth, box) pairs
    (record,) = p4.realization_ghosts(units, after=False).values()
    assert record["ghost_n_boxes"] == 2
    # Union of 150-200 (both truths) and 90-145 (outside the second truth).
    assert record["ghost_length_traces"] == 51 + 56
    assert p4.ghost_count(units, after=False) == 2
    assert p4.union_length([(1, 3), (2, 5), (8, 8)]) == 6


def test_the_report_columns_and_fold_list(tmp_path) -> None:
    document = p4.run_p4(_inputs({"lt033": 1}), tmp_path)
    header = (tmp_path / p4.ACTED_FILE).read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(p4.ACTED_COLUMNS)
    assert document["folds"] == list(FOLDS)
    assert document["n_truth_folds_by_stratum"] == {"lt033": 8 * 5}
    assert document["pp_set_id"] == "S3"
    assert document["pp_config_sha256"] == p4.PP_SETS[PPSetId.S3].sha256()
