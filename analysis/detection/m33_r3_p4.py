"""M3-3 R3 P4: PP exposure, acted sets, ``p4_status`` and the PP ghost check.

Spec: ``docs/experiments/m33_runner_spec.md`` v0.3.6 §3-1 4 (exposure,
acted sets R, G and G', ``p4_status``), §3-1 6 (band match, ghost bands,
PP ghost assertion) and §3-2 (``p4_adoption_input``); plan
``docs/experiments/m33_cross_domain_plan.md`` v0.2.8 §3-5 P4.

The functions here take the P4 scope only: ``gz_ids`` lines of
``ids_freq_stratum`` ``main_400``, model group A0 and main-cell positions
(:func:`is_p4_scope_position`, applied by the R3 runner). They keep the two
short length strata ``lt033`` and ``033_05`` themselves.

Coordinates

* Exposure works on the band truth box in line L1 px against the line's
  ``ContentRows`` with the PP conditions of [PP] §2 (original box, category
  order a > b > d, PP-c only without a category). Only the rules of the
  sealed set count; the score condition of PP-c is left out.
* Matching works on L0 (``GridBox``, closed index ranges): a box takes part
  when it is ``pipe_band``, its score is over 0.25 and its rows overlap the
  truth rows by at least half the truth height. The covered length is the
  number of truth traces ``[i0, i1]`` under the union of such boxes.

Every unit is one truth x one fold. R, G and G' take the ``lw_stratum`` of
the truth band; G' counts (truth, G box, un-injected box) triples.

Defects ([RUN] §3-1 4 and 6) are collected and raised as
:class:`P4DefectError` (exit 1) after the two csv files are written:

* a PP outcome that is not what the action says (a box changed without an
  action, a clip that moved anything but the bottom row);
* a PP-b clip that changes the covered length of a truth;
* a ghost that is new after PP although neither it nor its former
  un-injected pair was acted on;
* ``p4_status`` ``no_effect`` with a ``recall_delta`` that is not exactly 0.

Nothing here prints; values go to the returned dictionary and the files.
"""

import csv
import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from statistics import fmean

from m32_recipe_inputs import InputError
from m32_recipe_match import grid_iou
from scipy.stats import beta

from gpr_detection.detect.base import PixelBox
from gpr_detection.eval.inject import MEDIAN_WINDOW_TRACES, band_conditions
from gpr_detection.io.classes import DetectionClass
from gpr_detection.io.radargram import GridBox
from gpr_detection.pipeline import (
    PP_SETS,
    ContentRows,
    PPAction,
    PPConfig,
    PPRule,
    PPSetId,
    pp_category,
)

CONF_OPERATING = 0.25
"""Operating threshold: a box counts when its score is over this value."""
TIME_OVERLAP_FRACTION = 0.5
"""Rows shared with the truth, as a fraction of the truth height."""
GHOST_PAIR_IOU = 0.5
"""An un-injected ``pipe_band`` box with IoU >= this pairs a ghost candidate."""
RECALL_DELTA_MIN_PP = -1.0
"""P4 passes with ``recall_delta >= -1.0 %p`` ([RUN] §3-2, v0.3.6)."""
RECALL_DELTA_TOLERANCE_PP = 1e-9
"""Floating slack of the P4 comparison: exactly -1.0 %p passes."""
STRATUM_MIN_POSITIONS = 30
"""Small-stratum rule: below 30 positions in a stratum the two are pooled."""

LW_EDGES = (0.33, 0.5, 0.67)
"""``L / W_eff`` stratum edges; a value on an edge goes up ([a, b) ranges)."""
LW_STRATA = ("lt033", "033_05", "05_067", "ge067")
P4_STRATA = ("lt033", "033_05")
"""The two short strata of the P4 value (``L / W_eff`` < 0.5)."""

P4_DATASET = "gz_ids"
P4_FREQ_STRATUM = "main_400"
P4_MODEL_GROUP = "A0"
IDS_FREQ_STRATUM_COUNTS = {"main_400": 53, "report_other": 3, "excluded_2ghz": 4}
"""IDS 60 lines by ``ids_freq_stratum`` ([EMR] D-6); asserted per run."""

REMOVED_IN_RECALL_SET = (PPAction.REMOVED_A, PPAction.REMOVED_B, PPAction.REMOVED_D)
"""Actions on R boxes that decide ``p4_status`` (``pp_acted_boxes``)."""

ACTED_COLUMNS = (
    "line_id",
    "fold",
    "realization_id",
    "injection_id",
    "lw_stratum",
    "set",
    "source",
    "box_id",
    "pp_action",
)
ARTIFACT_COLUMNS = (
    "line_id",
    "fold",
    "realization_id",
    "injection_id",
    "lw_stratum",
    "box_id",
    "candidate_pp_action",
    "pair_box_id",
    "pair_pp_action",
    "explained",
)
ACTED_FILE = "p4_pp_acted.csv"
ARTIFACT_FILE = "ghost_pp_artifacts.csv"

SET_R = "R"
SET_G = "G"
SET_G_PRIME = "G'"
SOURCE_INJECTED = "injected"
SOURCE_UNINJECTED = "uninjected"


class P4Status(StrEnum):
    """``p4_status`` values, decided in this order ([RUN] §3-1 4)."""

    NOT_EVALUABLE = "not_evaluable"
    NO_EFFECT = "no_effect"
    ACTED_UNEXPOSED = "acted_unexposed"
    EVALUATED = "evaluated"


class StratumMode(StrEnum):
    """How the two short strata are weighted ([RUN] §3-2 작은 층 처리)."""

    TWO_STRATA = "two_strata"
    POOLED = "pooled"
    NOT_EVALUABLE = "not_evaluable"


class P4DefectError(RuntimeError):
    """An implementation defect found while computing P4 (exit 1).

    Attributes:
        document: ``p4_adoption_input`` as far as it was computed, with the
            defect list, so the runner can still write its summary.
    """

    def __init__(self, message: str, document: dict) -> None:
        super().__init__(message)
        self.document = document


# ------------------------------------------------------------------ inputs


@dataclass(frozen=True)
class BandPosition:
    """One planned band position (shared by its SNR x polarity realizations).

    Attributes:
        position_id: Plan id of the position.
        line_id: Background line.
        lw_stratum: ``L / W_eff`` stratum of the band.
        truth_px: Band truth box in line L1 px (exposure only).
        content: ``y_bot``, ``P_ell`` and ``y_z`` of the line image.
    """

    position_id: str
    line_id: str
    lw_stratum: str
    truth_px: PixelBox
    content: ContentRows


@dataclass(frozen=True)
class BandTruthUnit:
    """One accepted band injection (L0).

    Attributes:
        injection_id: Table id.
        realization_id: Realization that holds the band.
        position_id: Its planned position.
        line_id: Background line.
        lw_stratum: ``L / W_eff`` stratum of the band.
        snr_db: SNR step.
        polarity: +1 or -1.
        box: Traces ``i0..i1`` between the -6 dB points x the -20 dB rows.
        taper_traces: End taper of each side (outside ``box``).
    """

    injection_id: str
    realization_id: str
    position_id: str
    line_id: str
    lw_stratum: str
    snr_db: float
    polarity: int
    box: GridBox
    taper_traces: int

    @property
    def length_traces(self) -> int:
        return self.box.n_traces


@dataclass(frozen=True)
class PPBox:
    """One merged box before and after PP (L0).

    Attributes:
        box_id: Unique within its realization (or line) and fold.
        class_id: Detection class.
        score: Merged score.
        before: L0 box before PP.
        action: ``pp_action`` of the sealed set.
        after: L0 box after PP, ``None`` when removed.
    """

    box_id: str
    class_id: int
    score: float
    before: GridBox
    action: PPAction
    after: GridBox | None


@dataclass(frozen=True)
class P4Inputs:
    """Everything the P4 value needs (P4 scope, every fold).

    Attributes:
        pp_set_id: Sealed set (``m33_inputs.assert_sealed_pp``).
        folds: Folds of the run.
        positions: Planned positions (C1b plan table).
        truths: Accepted band injections.
        injected_boxes: ``(realization_id, fold) -> boxes``.
        uninjected_boxes: ``(line_id, fold) -> boxes`` of the un-injected run.
    """

    pp_set_id: PPSetId
    folds: tuple[int, ...]
    positions: tuple[BandPosition, ...]
    truths: tuple[BandTruthUnit, ...]
    injected_boxes: Mapping[tuple[str, int], Sequence[PPBox]]
    uninjected_boxes: Mapping[tuple[str, int], Sequence[PPBox]]
    inference_ranges: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    """``realization_id -> (first, last)`` inferred traces (ghost pair cut)."""

    @property
    def config(self) -> PPConfig:
        return PP_SETS[self.pp_set_id]


def is_p4_scope_position(
    dataset: str, freq_stratum: str, model_group: str, main_cell: bool
) -> bool:
    """A position the P4 value may use (the runner filters with this)."""
    return (
        dataset == P4_DATASET
        and freq_stratum == P4_FREQ_STRATUM
        and model_group == P4_MODEL_GROUP
        and main_cell
    )


def assert_ids_freq_strata(line_strata: Mapping[str, str]) -> dict[str, int]:
    """IDS lines per ``ids_freq_stratum`` must be 53 / 3 / 4 ([RUN] §3-2).

    Raises:
        InputError: For other counts (exit 2).
    """
    counts = Counter(line_strata.values())
    found = {name: counts.get(name, 0) for name in IDS_FREQ_STRATUM_COUNTS}
    if found != IDS_FREQ_STRATUM_COUNTS or sum(counts.values()) != sum(found.values()):
        raise InputError(f"IDS lines per ids_freq_stratum {dict(counts)}")
    return found


def lw_stratum(length_traces: int, n_traces: int) -> str:
    """``L / W_eff`` stratum, ``W_eff = min(301, N)``; edges go up."""
    ratio = length_traces / min(MEDIAN_WINDOW_TRACES, n_traces)
    for edge, name in zip(LW_EDGES, LW_STRATA, strict=False):
        if ratio < edge:
            return name
    return LW_STRATA[-1]


# --------------------------------------------------------------- exposure


def exposed_rule(
    truth_px: PixelBox, content: ContentRows, config: PPConfig
) -> PPRule | None:
    """Rule of ``config`` whose [PP] §2 condition the truth box meets.

    The category is fixed on the box (a > b > d); a category whose rule is
    not in the set exposes nothing. PP-c is looked at only without a
    category and without its score condition.
    """
    category = pp_category(
        truth_px.y1, truth_px.y2, content, config.edge_px, config.zero_margin_periods
    )
    if category is not None:
        rule = PPRule(category.value)
        return rule if rule in config.rules else None
    if PPRule.C in config.rules and in_band_window(truth_px, content, config):
        return PPRule.C
    return None


def in_band_window(truth_px: PixelBox, content: ContentRows, config: PPConfig) -> bool:
    """The truth bottom lies in the PP-c window ``y2 >= y_bot - w_c``."""
    return truth_px.y2 >= content.y_bot_px - config.band_window_px


@dataclass(frozen=True)
class Exposure:
    """Exposure of the P4 positions (report counts per stratum)."""

    rule_by_position: dict[str, PPRule | None]
    in_window_by_position: dict[str, bool]

    def exposed_ids(self) -> set[str]:
        return {
            position_id
            for position_id, rule in self.rule_by_position.items()
            if rule is not None
        }


def exposure_of(positions: Iterable[BandPosition], config: PPConfig) -> Exposure:
    rules, windows = {}, {}
    for position in positions:
        rules[position.position_id] = exposed_rule(
            position.truth_px, position.content, config
        )
        windows[position.position_id] = in_band_window(
            position.truth_px, position.content, config
        )
    return Exposure(rules, windows)


# ------------------------------------------------------------ L0 matching


def shares_rows(truth: GridBox, box: GridBox) -> bool:
    """Rows shared with the truth are at least half the truth height."""
    rows = min(truth.k1, box.k1) - max(truth.k0, box.k0) + 1
    return rows >= TIME_OVERLAP_FRACTION * truth.n_samples


def overlaps_traces(box: GridBox, first: int, last: int) -> bool:
    return box.i0 <= last and box.i1 >= first


def leaves_traces(box: GridBox, first: int, last: int) -> bool:
    """Some trace of the box lies outside ``[first, last]``."""
    return box.i0 < first or box.i1 > last


def covered_traces(truth: GridBox, boxes: Iterable[GridBox]) -> int:
    """Truth traces ``i0..i1`` under the union of the boxes (integer)."""
    pieces = sorted(
        (max(box.i0, truth.i0), min(box.i1, truth.i1))
        for box in boxes
        if overlaps_traces(box, truth.i0, truth.i1)
    )
    covered, reach = 0, truth.i0 - 1
    for start, stop in pieces:
        start = max(start, reach + 1)
        if stop >= start:
            covered += stop - start + 1
            reach = stop
    return covered


def counts_for_match(box: PPBox) -> bool:
    """``pipe_band`` with a score over the operating threshold."""
    return box.class_id == DetectionClass.PIPE_BAND and box.score > CONF_OPERATING


def can_cover(truth: GridBox, box: GridBox) -> bool:
    """Shared rows and some trace inside the truth span ``[i0, i1]``."""
    return shares_rows(truth, box) and overlaps_traces(box, truth.i0, truth.i1)


def recall_boxes(truth: GridBox, boxes: Iterable[GridBox]) -> list[GridBox]:
    """Boxes that may cover the truth (:func:`can_cover`)."""
    return [box for box in boxes if can_cover(truth, box)]


def is_ghost_candidate(truth: BandTruthUnit, box: GridBox) -> bool:
    """Shared rows and some trace outside the truth span grown by the taper."""
    first = truth.box.i0 - truth.taper_traces
    last = truth.box.i1 + truth.taper_traces
    return shares_rows(truth.box, box) and leaves_traces(box, first, last)


def has_pair(box: GridBox, uninjected: Iterable[GridBox]) -> bool:
    return any(grid_iou(box, other) >= GHOST_PAIR_IOU for other in uninjected)


# ------------------------------------------------------- outcome checking


def outcome_problem(box: PPBox) -> str | None:
    """Why a PP outcome does not match its action (``None`` when it does)."""
    if box.action is PPAction.NONE:
        return None if box.after == box.before else "changed without an action"
    if box.action.is_removed:
        return None if box.after is None else "removed box still present"
    after = box.after
    if after is None:
        return "clipped box missing"
    same_top_and_traces = (after.i0, after.i1, after.k0) == (
        box.before.i0,
        box.before.i1,
        box.before.k0,
    )
    if not same_top_and_traces or after.k1 > box.before.k1:
        return "clip moved more than the bottom row"
    return None


# ---------------------------------------------------------------- per unit


@dataclass(frozen=True)
class UnitResult:
    """One truth x fold: covered lengths and the boxes of each set."""

    truth: BandTruthUnit
    fold: int
    covered_before: int
    covered_after: int
    covered_clip_only: int
    recall_set: tuple[PPBox, ...]
    ghost_set: tuple[PPBox, ...]
    ghost_pairs: tuple[tuple[PPBox, PPBox], ...]
    ghosts_before: frozenset[str]
    ghosts_after: frozenset[str]
    former_pairs: Mapping[str, tuple[PPBox, ...]] = field(default_factory=dict)
    band_boxes: Mapping[str, PPBox] = field(default_factory=dict)
    """Counted ``pipe_band`` boxes of the realization and fold by ``box_id``."""

    @property
    def has_clip_in_recall_set(self) -> bool:
        return any(box.action is PPAction.CLIPPED_B for box in self.recall_set)

    def ghost_boxes(self, after: bool) -> list[tuple[str, GridBox]]:
        """``(box_id, L0 box)`` of the ghosts before or after PP."""
        ids = self.ghosts_after if after else self.ghosts_before
        boxes = []
        for box_id in sorted(ids):
            box = self.band_boxes[box_id]
            coordinates = box.after if after else box.before
            if coordinates is not None:
                boxes.append((box_id, coordinates))
        return boxes


def _after_boxes(boxes: Iterable[PPBox]) -> list[GridBox]:
    return [box.after for box in boxes if box.after is not None]


def _clip_only(box: PPBox) -> GridBox:
    """The box with only a PP-b clip applied (removals kept as before)."""
    if box.action is PPAction.CLIPPED_B and box.after is not None:
        return box.after
    return box.before


def _ghost_ids(
    truth: BandTruthUnit,
    boxes: Iterable[tuple[str, GridBox]],
    uninjected: Sequence[GridBox],
) -> frozenset[str]:
    return frozenset(
        box_id
        for box_id, box in boxes
        if is_ghost_candidate(truth, box) and not has_pair(box, uninjected)
    )


def clip_to_traces(box: GridBox | None, first: int, last: int) -> GridBox | None:
    """The part of ``box`` inside traces ``[first, last]`` (``None`` if empty)."""
    if box is None or box.i1 < first or box.i0 > last:
        return None
    return GridBox(max(box.i0, first), min(box.i1, last), box.k0, box.k1)


def clip_pair_pool(
    boxes: Iterable[PPBox], trace_range: tuple[int, int] | None
) -> list[PPBox]:
    """Un-injected boxes cut to the inferred traces of the injected realization.

    Runner spec §3-1 4 배치 구현 해석: an injected box cut at the end of the
    inferred range would otherwise lose its pair with the whole-line
    un-injected box and count as a false ghost. ``None`` keeps the boxes.
    """
    if trace_range is None:
        return list(boxes)
    first, last = trace_range
    clipped = []
    for box in boxes:
        before = clip_to_traces(box.before, first, last)
        if before is None:
            continue
        after = clip_to_traces(box.after, first, last)
        clipped.append(replace(box, before=before, after=after))
    return clipped


def evaluate_unit(
    truth: BandTruthUnit,
    fold: int,
    injected: Sequence[PPBox],
    uninjected: Sequence[PPBox],
    pair_range: tuple[int, int] | None = None,
) -> UnitResult:
    """Covered lengths before, after and clip-only; R, G and G'; ghosts.

    ``pair_range`` is the trace range inferred on the injected realization;
    the un-injected boxes are cut to it before any ghost pairing.
    """
    bands = [box for box in injected if counts_for_match(box)]
    pairs_pool = clip_pair_pool(
        (box for box in uninjected if counts_for_match(box)), pair_range
    )
    ghost_set = tuple(box for box in bands if is_ghost_candidate(truth, box.before))
    return UnitResult(
        truth=truth,
        fold=fold,
        covered_before=_covered(truth, [box.before for box in bands]),
        covered_after=_covered(truth, _after_boxes(bands)),
        covered_clip_only=_covered(truth, [_clip_only(box) for box in bands]),
        recall_set=tuple(box for box in bands if can_cover(truth.box, box.before)),
        ghost_set=ghost_set,
        ghost_pairs=tuple(
            (candidate, other)
            for candidate in ghost_set
            for other in pairs_pool
            if _paired_before_or_after(candidate, other)
        ),
        ghosts_before=_ghost_ids(
            truth,
            [(box.box_id, box.before) for box in bands],
            [box.before for box in pairs_pool],
        ),
        ghosts_after=_ghost_ids(
            truth,
            [(box.box_id, box.after) for box in bands if box.after is not None],
            _after_boxes(pairs_pool),
        ),
        former_pairs={
            candidate.box_id: _former_pairs(candidate, pairs_pool)
            for candidate in ghost_set
        },
        band_boxes={box.box_id: box for box in bands},
    )


def _covered(truth: BandTruthUnit, boxes: Iterable[GridBox]) -> int:
    return covered_traces(truth.box, recall_boxes(truth.box, boxes))


def _former_pairs(candidate: PPBox, pool: Iterable[PPBox]) -> tuple[PPBox, ...]:
    """Un-injected boxes that paired the candidate before PP (IoU >= 0.5)."""
    return tuple(
        other
        for other in pool
        if grid_iou(candidate.before, other.before) >= GHOST_PAIR_IOU
    )


def _paired_before_or_after(candidate: PPBox, other: PPBox) -> bool:
    """G' rule: IoU >= 0.5 before PP, or after PP when both boxes remain."""
    if grid_iou(candidate.before, other.before) >= GHOST_PAIR_IOU:
        return True
    if candidate.after is None or other.after is None:
        return False
    return grid_iou(candidate.after, other.after) >= GHOST_PAIR_IOU


# -------------------------------------------------------- stratum counting


def _per_stratum(counter: Counter[str]) -> dict:
    by_stratum = {name: counter.get(name, 0) for name in P4_STRATA}
    return {"by_stratum": by_stratum, "total": sum(by_stratum.values())}


def stratum_mode(n_positions_by_stratum: Mapping[str, int]) -> StratumMode:
    """Two strata with >= 30 positions each, pooled below, none at 0 + 0."""
    counts = [n_positions_by_stratum.get(name, 0) for name in P4_STRATA]
    if not any(counts):
        return StratumMode.NOT_EVALUABLE
    if all(count >= STRATUM_MIN_POSITIONS for count in counts):
        return StratumMode.TWO_STRATA
    return StratumMode.POOLED


def decide_p4_status(
    n_positions_by_stratum: Mapping[str, int], n_exposed: int, n_acted: int
) -> P4Status:
    """``p4_status`` in the fixed order of [RUN] §3-1 4."""
    if stratum_mode(n_positions_by_stratum) is StratumMode.NOT_EVALUABLE:
        return P4Status.NOT_EVALUABLE
    if n_exposed == 0:
        return P4Status.NO_EFFECT if n_acted == 0 else P4Status.ACTED_UNEXPOSED
    return P4Status.EVALUATED


def recall_delta_passes(recall_delta_pp: float | None) -> bool | None:
    """P4 rule ``recall_delta >= -1.0 %p - 1e-9`` (``None`` when not evaluable)."""
    if recall_delta_pp is None:
        return None
    return recall_delta_pp >= RECALL_DELTA_MIN_PP - RECALL_DELTA_TOLERANCE_PP


# ------------------------------------------------------------ recall delta


def _cell_recall(units: Sequence[UnitResult], after: bool) -> float:
    """Length-weighted recall of one SNR x polarity cell."""
    covered = sum(u.covered_after if after else u.covered_before for u in units)
    return covered / sum(u.truth.length_traces for u in units)


def _stratum_recall(units: Sequence[UnitResult], after: bool) -> float:
    """SNR x polarity cells weighted equally."""
    cells: dict[tuple[float, int], list[UnitResult]] = {}
    for unit in units:
        cells.setdefault((unit.truth.snr_db, unit.truth.polarity), []).append(unit)
    return fmean(_cell_recall(cells[key], after) for key in sorted(cells))


def _fold_recall(units: Sequence[UnitResult], mode: StratumMode, after: bool) -> float:
    if mode is StratumMode.POOLED:
        return _stratum_recall(units, after)
    strata = [
        [unit for unit in units if unit.truth.lw_stratum == name] for name in P4_STRATA
    ]
    return fmean(_stratum_recall(group, after) for group in strata if group)


def recall_delta_pp(
    units: Sequence[UnitResult], mode: StratumMode, folds: Sequence[int]
) -> float | None:
    """PP after - before horizontal recall in %p, mean over the folds.

    Equal stratum weights (``two_strata``) or one pooled stratum; SNR x
    polarity cells equal inside a stratum; length-weighted inside a cell.
    The before and after values go through the same arithmetic, so equal
    covered lengths give exactly 0.
    """
    before = mean_recall(units, mode, folds, after=False)
    after = mean_recall(units, mode, folds, after=True)
    if before is None or after is None:
        return None
    return 100.0 * (after - before)


def mean_recall(
    units: Sequence[UnitResult], mode: StratumMode, folds: Sequence[int], after: bool
) -> float | None:
    """Horizontal recall (0-1) before or after PP, mean over the folds.

    The weighting of :func:`recall_delta_pp`; ``recall_pre`` reports the
    value before PP.
    """
    if mode is StratumMode.NOT_EVALUABLE or not units:
        return None
    by_fold = {fold: [u for u in units if u.fold == fold] for fold in folds}
    used = [fold for fold in folds if by_fold[fold]]
    return fmean(_fold_recall(by_fold[fold], mode, after=after) for fold in used)


def outside_segments(truth: BandTruthUnit, box: GridBox) -> list[tuple[int, int]]:
    """Closed trace ranges of the box outside ``[i0 - T, i1 + T]``."""
    first = truth.box.i0 - truth.taper_traces
    last = truth.box.i1 + truth.taper_traces
    segments = []
    if box.i0 < first:
        segments.append((box.i0, min(box.i1, first - 1)))
    if box.i1 > last:
        segments.append((max(box.i0, last + 1), box.i1))
    return segments


def union_length(segments: Iterable[tuple[int, int]]) -> int:
    """Traces under the union of closed integer ranges."""
    length, reach = 0, None
    for start, stop in sorted(segments):
        if reach is not None:
            start = max(start, reach + 1)
        if stop >= start:
            length += stop - start + 1
        reach = stop if reach is None else max(reach, stop)
    return length


def realization_ghosts(
    units: Iterable[UnitResult], after: bool
) -> dict[tuple[str, int], dict[str, int]]:
    """``ghost_n_boxes`` and ``ghost_length_traces`` per (realization, fold).

    A box that is a ghost of two truths of one realization counts once; the
    length is the union of every ghost box's part outside its truth's
    ``[i0 - T, i1 + T]`` (runner spec §3-1 6, cv 해석 2026-11-16).
    """
    boxes: dict[tuple[str, int], set[str]] = {}
    segments: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for unit in units:
        key = (unit.truth.realization_id, unit.fold)
        boxes.setdefault(key, set())
        segments.setdefault(key, [])
        for box_id, box in unit.ghost_boxes(after):
            boxes[key].add(box_id)
            segments[key] += outside_segments(unit.truth, box)
    return {
        key: {
            "ghost_n_boxes": len(boxes[key]),
            "ghost_length_traces": union_length(segments[key]),
        }
        for key in boxes
    }


def ghost_count(units: Iterable[UnitResult], after: bool) -> int:
    """Distinct ghost boxes per realization and fold, summed (``ghost_delta``)."""
    per_realization = realization_ghosts(units, after)
    return sum(item["ghost_n_boxes"] for item in per_realization.values())


# ------------------------------------------------------------- acted sets


def acted_rows(units: Iterable[UnitResult]) -> list[dict]:
    """Rows of ``p4_pp_acted.csv``: every acted box of R, G and G'."""
    rows = []
    for unit in units:
        for box in unit.recall_set:
            if box.action is not PPAction.NONE:
                rows.append(_acted_row(unit, SET_R, SOURCE_INJECTED, box))
        for box in unit.ghost_set:
            if box.action is not PPAction.NONE:
                rows.append(_acted_row(unit, SET_G, SOURCE_INJECTED, box))
        for _, other in unit.ghost_pairs:
            if other.action is not PPAction.NONE:
                rows.append(_acted_row(unit, SET_G_PRIME, SOURCE_UNINJECTED, other))
    return rows


def _acted_row(unit: UnitResult, set_name: str, source: str, box: PPBox) -> dict:
    return {
        "line_id": unit.truth.line_id,
        "fold": unit.fold,
        "realization_id": unit.truth.realization_id,
        "injection_id": unit.truth.injection_id,
        "lw_stratum": unit.truth.lw_stratum,
        "set": set_name,
        "source": source,
        "box_id": box.box_id,
        "pp_action": box.action.value,
    }


def recall_set_acts(units: Iterable[UnitResult]) -> dict:
    """``pp_acted_boxes``, ``pp_acted_by_rule`` and ``pp_acted_lines``.

    ``pp_acted_lines`` counts the lines with a removal (a, b, d) in R, once
    across the two strata; ``pp_acted_lines_by_stratum`` per stratum.
    """
    per_stratum: Counter[str] = Counter()
    by_rule: Counter[str] = Counter()
    lines: dict[str, set[str]] = {name: set() for name in P4_STRATA}
    for unit in units:
        for box in unit.recall_set:
            if box.action in REMOVED_IN_RECALL_SET:
                per_stratum[unit.truth.lw_stratum] += 1
                by_rule[box.action.value] += 1
                lines[unit.truth.lw_stratum].add(unit.truth.line_id)
    return {
        "pp_acted_boxes": _per_stratum(per_stratum),
        "pp_acted_by_rule": {a.value: by_rule[a.value] for a in REMOVED_IN_RECALL_SET},
        "pp_acted_lines": len(set().union(*lines.values())),
        "pp_acted_lines_by_stratum": {name: len(lines[name]) for name in P4_STRATA},
    }


def clip_recall_set(units: Iterable[UnitResult]) -> tuple[dict, list[str]]:
    """``pp_acted_clip_recall_set`` and the truths whose coverage a clip changed."""
    per_stratum: Counter[str] = Counter()
    broken = []
    for unit in units:
        n_clipped = sum(box.action is PPAction.CLIPPED_B for box in unit.recall_set)
        per_stratum[unit.truth.lw_stratum] += n_clipped
        if n_clipped and unit.covered_clip_only != unit.covered_before:
            broken.append(f"{unit.truth.injection_id}|fold{unit.fold}")
    record = _per_stratum(per_stratum) | {"coverage_unchanged": not broken}
    return record, broken


def ghost_set_acts(units: Iterable[UnitResult]) -> dict:
    """``pp_acted_ghost_set``: acted G pairs, acted G' triples, unique G' boxes."""
    g_acted: Counter[str] = Counter()
    g_prime_acted: Counter[str] = Counter()
    unique_uninjected = set()
    for unit in units:
        stratum = unit.truth.lw_stratum
        g_acted[stratum] += sum(b.action is not PPAction.NONE for b in unit.ghost_set)
        for _, other in unit.ghost_pairs:
            if other.action is not PPAction.NONE:
                g_prime_acted[stratum] += 1
                unique_uninjected.add((unit.truth.line_id, unit.fold, other.box_id))
    total = g_acted + g_prime_acted
    return _per_stratum(total) | {
        "g": _per_stratum(g_acted),
        "g_prime_pairs": _per_stratum(g_prime_acted),
        "g_prime_unique_uninjected": len(unique_uninjected),
    }


def all_band_acts(inputs: P4Inputs) -> dict:
    """``pp_acted_boxes_all``: every acted P4 ``pipe_band`` box, both runs.

    Boxes of the P4 realizations and of the un-injected runs of the P4
    lines, every fold, counted once each (no stratum: most boxes match no
    truth).
    """
    realizations = {truth.realization_id for truth in _p4_truths(inputs)}
    lines = {position.line_id for position in _p4_positions(inputs)}
    by_rule: Counter[str] = Counter()
    for (realization_id, fold), boxes in inputs.injected_boxes.items():
        if realization_id in realizations and fold in inputs.folds:
            by_rule.update(_acted_band_actions(boxes))
    for (line_id, fold), boxes in inputs.uninjected_boxes.items():
        if line_id in lines and fold in inputs.folds:
            by_rule.update(_acted_band_actions(boxes))
    return {"total": sum(by_rule.values()), "by_rule": dict(sorted(by_rule.items()))}


def _acted_band_actions(boxes: Iterable[PPBox]) -> list[str]:
    return [
        box.action.value
        for box in boxes
        if counts_for_match(box) and box.action is not PPAction.NONE
    ]


# ---------------------------------------------------------- ghost artefacts


def ghost_artifact_rows(units: Iterable[UnitResult]) -> list[dict]:
    """New ghosts after PP with the actions that may explain them."""
    rows = []
    for unit in units:
        by_id = {box.box_id: box for box in unit.ghost_set}
        for box_id in sorted(unit.ghosts_after - unit.ghosts_before):
            candidate = by_id.get(box_id)
            action = PPAction.NONE if candidate is None else candidate.action
            pairs = unit.former_pairs.get(box_id, ()) or (None,)
            for pair in pairs:
                rows.append(_artifact_row(unit, box_id, action, pair))
    return rows


def _artifact_row(
    unit: UnitResult, box_id: str, action: PPAction, pair: PPBox | None
) -> dict:
    pair_action = PPAction.NONE if pair is None else pair.action
    explained = action is not PPAction.NONE or pair_action is not PPAction.NONE
    return {
        "line_id": unit.truth.line_id,
        "fold": unit.fold,
        "realization_id": unit.truth.realization_id,
        "injection_id": unit.truth.injection_id,
        "lw_stratum": unit.truth.lw_stratum,
        "box_id": box_id,
        "candidate_pp_action": action.value,
        "pair_box_id": "" if pair is None else pair.box_id,
        "pair_pp_action": "" if pair is None else pair_action.value,
        "explained": explained,
    }


def unexplained_ghosts(rows: Iterable[Mapping]) -> set[tuple[str, int, str, str]]:
    """New ghosts with no acted candidate and no acted former pair."""
    explained, seen = set(), set()
    for row in rows:
        key = (row["realization_id"], row["fold"], row["injection_id"], row["box_id"])
        seen.add(key)
        if row["explained"]:
            explained.add(key)
    return seen - explained


# ------------------------------------------------------------------ driver


def _p4_positions(inputs: P4Inputs) -> list[BandPosition]:
    return [p for p in inputs.positions if p.lw_stratum in P4_STRATA]


def _p4_truths(inputs: P4Inputs) -> list[BandTruthUnit]:
    return [t for t in inputs.truths if t.lw_stratum in P4_STRATA]


def _p4_units(inputs: P4Inputs) -> list[UnitResult]:
    units = []
    for truth in _p4_truths(inputs):
        for fold in inputs.folds:
            units.append(
                evaluate_unit(
                    truth,
                    fold,
                    inputs.injected_boxes.get((truth.realization_id, fold), ()),
                    inputs.uninjected_boxes.get((truth.line_id, fold), ()),
                    inputs.inference_ranges.get(truth.realization_id),
                )
            )
    return units


def _outcome_defects(inputs: P4Inputs) -> list[str]:
    defects = []
    for source in (inputs.injected_boxes, inputs.uninjected_boxes):
        for (key, fold), boxes in source.items():
            for box in boxes:
                problem = outcome_problem(box)
                if problem:
                    defects.append(
                        f"pp outcome {key}|fold{fold}|{box.box_id}: {problem}"
                    )
    return defects


@dataclass
class P4Result:
    """``p4_adoption_input``, the two csv row lists and the defects found."""

    document: dict
    acted_rows: list[dict]
    artifact_rows: list[dict]
    defects: list[str]


def _exposure_record(
    positions: Sequence[BandPosition], exposure: Exposure, config: PPConfig
) -> dict:
    exposed: Counter[str] = Counter()
    window: Counter[str] = Counter()
    by_rule = {rule.value: 0 for rule in config.rules}
    for position in positions:
        rule = exposure.rule_by_position[position.position_id]
        if rule is not None:
            exposed[position.lw_stratum] += 1
            by_rule[rule.value] += 1
        window[position.lw_stratum] += exposure.in_window_by_position[
            position.position_id
        ]
    return {
        "pp_exposed_positions": _per_stratum(exposed),
        "pp_exposed_by_rule": by_rule,
        "pp_c_window_positions": _per_stratum(window),
    }


def _exposed_delta(
    units: Sequence[UnitResult],
    exposed_ids: set[str],
    mode: StratumMode,
    folds: Sequence[int],
) -> dict:
    """``recall_delta_exposed`` (report only) and its position and line counts."""
    exposed_units = [u for u in units if u.truth.position_id in exposed_ids]
    if not exposed_units:
        return {"recall_delta_exposed": None, "n_exposed_positions": 0, "n_lines": 0}
    exposed_mode = mode
    present = {u.truth.lw_stratum for u in exposed_units}
    if mode is StratumMode.TWO_STRATA and len(present) < len(P4_STRATA):
        exposed_mode = StratumMode.POOLED
    return {
        "recall_delta_exposed": recall_delta_pp(exposed_units, exposed_mode, folds),
        "n_exposed_positions": len({u.truth.position_id for u in exposed_units}),
        "n_exposed_lines": len({u.truth.line_id for u in exposed_units}),
        "exposed_stratum_mode": exposed_mode.value,
    }


def mde_inputs(positions: Sequence[BandPosition]) -> dict:
    """Detector-free counts stats-design needs for ``mde_pp`` ([RUN] §3-2)."""
    per_line: Counter[str] = Counter(position.line_id for position in positions)
    by_stratum_line: dict[str, Counter[str]] = {name: Counter() for name in P4_STRATA}
    for position in positions:
        by_stratum_line[position.lw_stratum][position.line_id] += 1
    return {
        "n_positions_by_stratum": {
            name: sum(counts.values()) for name, counts in by_stratum_line.items()
        },
        "n_lines": len(per_line),
        "positions_per_line": dict(sorted(per_line.items())),
        "positions_per_line_by_stratum": {
            name: dict(sorted(counts.items()))
            for name, counts in by_stratum_line.items()
        },
    }


def plan_record(positions: Sequence[BandPosition], config: PPConfig) -> dict:
    """Detector-free P4 values of the plan report (fixed at C1b).

    Position counts of the two strata, ``stratum_mode``, the exposure counts
    and the ``mde_pp`` inputs; only P4-scope positions are passed in.
    """
    scope = [position for position in positions if position.lw_stratum in P4_STRATA]
    counts = Counter(position.lw_stratum for position in scope)
    n_by_stratum = {name: counts.get(name, 0) for name in P4_STRATA}
    exposure = exposure_of(scope, config)
    return {
        "pp_set_id": config.set_id.value,
        "pp_config_sha256": config.sha256(),
        "n_positions_by_stratum": n_by_stratum,
        "stratum_mode": stratum_mode(n_by_stratum).value,
        **_exposure_record(scope, exposure, config),
        "mde_pp": None,
        "mde_inputs": mde_inputs(scope),
    }


def cell_name(snr_db: float, polarity: int) -> str:
    """``snr<dB>_pos`` / ``snr<dB>_neg``: one SNR x polarity cell."""
    return f"snr{snr_db:g}_{'pos' if polarity > 0 else 'neg'}"


BAND_CELLS = tuple(
    cell_name(condition.snr_db, condition.polarity) for condition in band_conditions()
)
"""The 8 SNR x polarity cells of the R3 table (``inject.band_conditions``)."""


def truth_cell_counts(
    units: Sequence[UnitResult],
    n_positions_by_stratum: Mapping[str, int],
    folds: Sequence[int],
) -> dict:
    """Truth x fold units per stratum x cell x fold, and the empty cells.

    Report only, not a defect: an injection is rejected per realization
    (int16 rule, scale, spacing), and the higher SNR steps are rejected more
    often, so a stratum with positions may lose a whole cell on the normal
    path (verif audit item 5, 2026-11-16). Strata without positions list no
    missing cells (``stratum_mode`` handles them).
    """
    counts: dict[str, dict[str, Counter[int]]] = {
        name: {cell: Counter() for cell in BAND_CELLS} for name in P4_STRATA
    }
    for unit in units:
        cell = cell_name(unit.truth.snr_db, unit.truth.polarity)
        counts[unit.truth.lw_stratum][cell][unit.fold] += 1
    by_cell = {
        name: {
            cell: {str(fold): cells[cell][fold] for fold in folds}
            for cell in BAND_CELLS
        }
        for name, cells in counts.items()
    }
    missing = {
        name: [cell for cell in BAND_CELLS if not sum(counts[name][cell].values())]
        for name in P4_STRATA
        if n_positions_by_stratum.get(name, 0) > 0
    }
    return {
        "by_stratum_cell_fold": by_cell,
        "expected_cells": list(BAND_CELLS),
        "missing_cells_by_stratum": missing,
        "cells_complete": not any(missing.values()),
    }


def recall_set_extent(units: Iterable[UnitResult]) -> dict:
    """``n_r_boxes`` and ``n_r_lines``: the size of R before PP (user 2026-11-16).

    ``n_r_boxes`` counts like ``pp_acted_boxes`` (a box in R of two truths
    counts for each truth x fold unit). ``n_r_lines`` counts the lines with
    at least one R box per stratum; the total is their union.
    """
    per_stratum: Counter[str] = Counter()
    lines: dict[str, set[str]] = {name: set() for name in P4_STRATA}
    for unit in units:
        if unit.recall_set:
            per_stratum[unit.truth.lw_stratum] += len(unit.recall_set)
            lines[unit.truth.lw_stratum].add(unit.truth.line_id)
    return {
        "n_r_boxes": _per_stratum(per_stratum),
        "n_r_lines": {
            "by_stratum": {name: len(lines[name]) for name in P4_STRATA},
            "total": len(set().union(*lines.values())),
        },
    }


CLOPPER_PEARSON_TAIL = 0.025
"""Two-sided 95 %: the upper bound leaves 2.5 % above."""


def clopper_pearson_upper(n_events: int, n_trials: int) -> float | None:
    """Two-sided 95 % Clopper-Pearson upper bound of ``n_events / n_trials``.

    ``1 - 0.025^(1/n)`` for no event, the beta quantile otherwise, ``None``
    for no trial.
    """
    if n_trials == 0:
        return None
    if n_events >= n_trials:
        return 1.0
    if n_events == 0:
        return 1.0 - CLOPPER_PEARSON_TAIL ** (1.0 / n_trials)
    tail = 1.0 - CLOPPER_PEARSON_TAIL
    return float(beta.ppf(tail, n_events + 1, n_trials - n_events))


FEW_R_LINES = 30
"""Below this many R lines the upper bound is reported as wide."""


def action_line_rate_note(n_r_lines: int, upper95: float | None) -> str:
    """Note of ``action_line_rate_upper95`` (user decision 2026-11-16).

    No R line, fewer than 30 R lines (the bound in % with one decimal) or
    enough lines; the wording follows the runner spec.
    """
    if n_r_lines == 0 or upper95 is None:
        return "상한 없음(R 없음)"
    if n_r_lines < FEW_R_LINES:
        return f"R이 있는 측선이 적어 상한이 넓다({100.0 * upper95:.1f} %)"
    return "보고 전용: Clopper-Pearson 양측 95 % 상한"  # ASCII hyphen (RUF001)


def evaluate_p4(inputs: P4Inputs) -> P4Result:
    """``p4_adoption_input`` of one R3 injected run ([RUN] §3-2).

    Defects are collected in ``P4Result.defects``; :func:`run_p4` writes the
    files and then raises.
    """
    config = inputs.config
    positions = _p4_positions(inputs)
    n_positions = Counter(position.lw_stratum for position in positions)
    n_positions_by_stratum = {name: n_positions.get(name, 0) for name in P4_STRATA}
    mode = stratum_mode(n_positions_by_stratum)
    exposure = exposure_of(positions, config)
    exposure_record = _exposure_record(positions, exposure, config)
    units = _p4_units(inputs)
    recall_acts = recall_set_acts(units)
    clip_record, broken_clips = clip_recall_set(units)
    status = decide_p4_status(
        n_positions_by_stratum,
        exposure_record["pp_exposed_positions"]["total"],
        recall_acts["pp_acted_boxes"]["total"],
    )
    delta = recall_delta_pp(units, mode, inputs.folds)
    r_extent = recall_set_extent(units)
    n_r_lines = r_extent["n_r_lines"]["total"]
    upper95 = clopper_pearson_upper(recall_acts["pp_acted_lines"], n_r_lines)
    artifacts = ghost_artifact_rows(units)
    unexplained = unexplained_ghosts(artifacts)
    structural_zero = None
    if status is P4Status.NO_EFFECT:
        structural_zero = delta == 0.0
    defects = _outcome_defects(inputs)
    defects += [f"PP-b clip changed the covered length: {key}" for key in broken_clips]
    defects += [f"unexplained ghost after PP: {key}" for key in sorted(unexplained)]
    if structural_zero is False:
        defects.append(f"no_effect but recall_delta {delta!r} != 0")
    document = {
        "pp_set_id": inputs.pp_set_id.value,
        "pp_config_sha256": config.sha256(),
        "p4_status": status.value,
        "p4_structural_zero": structural_zero,
        "recall_delta": delta,
        "recall_delta_rule": {
            "min_pp": RECALL_DELTA_MIN_PP,
            "tolerance_pp": RECALL_DELTA_TOLERANCE_PP,
        },
        "recall_delta_pass": recall_delta_passes(delta),
        "recall_pre": mean_recall(units, mode, inputs.folds, after=False),
        "ghost_delta": None
        if mode is StratumMode.NOT_EVALUABLE
        else ghost_count(units, after=True) - ghost_count(units, after=False),
        "stratum_mode": mode.value,
        "mde_pp": None,
        "mde_note": "computed by stats-design-expert after the C1b plan",
        "mde_inputs": mde_inputs(positions),
        "n_positions_by_stratum": n_positions_by_stratum,
        "n_lines_with_positions": len({position.line_id for position in positions}),
        "n_truth_folds_by_stratum": dict(
            sorted(Counter(u.truth.lw_stratum for u in units).items())
        ),
        "n_truth_folds_by_cell": truth_cell_counts(
            units, n_positions_by_stratum, inputs.folds
        ),
        **r_extent,
        **exposure_record,
        **recall_acts,
        "action_line_rate_upper95": upper95,
        "action_line_rate_note": action_line_rate_note(n_r_lines, upper95),
        "pp_acted_clip_recall_set": clip_record,
        "pp_acted_ghost_set": ghost_set_acts(units),
        "pp_acted_boxes_all": all_band_acts(inputs),
        **_exposed_delta(units, exposure.exposed_ids(), mode, inputs.folds),
        "folds": list(inputs.folds),
    }
    return P4Result(document, acted_rows(units), artifacts, defects)


def _write_rows(path: Path, columns: Sequence[str], rows: Iterable[Mapping]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows([row[column] for column in columns] for row in rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_p4(inputs: P4Inputs, out_dir: Path) -> dict:
    """Evaluate P4, write the two csv files and return ``p4_adoption_input``.

    Raises:
        P4DefectError: After the files are written, when any defect was found
            (exit 1, the summary must record the failure).
    """
    result = evaluate_p4(inputs)
    acted_path = out_dir / ACTED_FILE
    artifact_path = out_dir / ARTIFACT_FILE
    document = dict(result.document)
    document["p4_pp_acted"] = {
        "path": str(acted_path),
        "sha256": _write_rows(acted_path, ACTED_COLUMNS, result.acted_rows),
        "n_rows": len(result.acted_rows),
    }
    unexplained = unexplained_ghosts(result.artifact_rows)
    document["ghost_pp_artifacts"] = {
        "path": str(artifact_path),
        "sha256": _write_rows(artifact_path, ARTIFACT_COLUMNS, result.artifact_rows),
        "n_rows": len(result.artifact_rows),
        "n_unexplained": len(unexplained),
        "unexplained_zero": not unexplained,
    }
    document["defects"] = list(result.defects)
    if result.defects:
        raise P4DefectError("; ".join(result.defects[:5]), document)
    return document
