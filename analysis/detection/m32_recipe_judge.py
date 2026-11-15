"""M3-2 S5 stage-C judge: plan 4-3 C-1-C-8 and selection (C-9 report-only, unmeasured).

Spec: ``docs/processing/m32_image_quality_plan.md`` v0.2.3 section 4-3 (the
thresholds), 4-3-1 (aggregation), 4-4 (fixed order), 4-8 (stage-B polarity
and B0 records), 4-9 (amendment 9) and ``docs/experiments/m32_cv_review.md``
sections 3-1 and 4-2. C-9 (band) is report-only by user decision and is not
measured by S5; the verdict only records that status.

Reads the runner outputs (``m32_recipe_eval.py``: ``uninjected/`` P5a and
``injected/`` P5b of one out-root, both finished full runs) and writes
``verdict.json`` only. Stdout carries no metric: its last line is
``{"status", "verdict", "g_in_cells", "g_in_not_evaluable", "elapsed_s"}``
(the G-in valid cell count and not-evaluable flag, both ``null`` off the
in-or-out path or after a failed pre-check). Exit codes: 0 verdict written,
1 pre-check failed (``verdict.json`` holds the r0 pre-check only), 2 input
refused, 3 internal error.

Pre-check (plan 3-2 after amendment 9, ``stage_c_precheck``): r0, fold 0,
``k`` 1.3, main cells IDS/A0 and 4TU/A0, in distribution; pooled and
cell-weighted recall >= 50 % at 12 dB and monotone 0 -> 6 -> 12 dB (a later
step at most 1 %p lower).

Order (plan 4-4, 9-7), two calls:

1. ``--precheck-only`` reads ``injected_precheck/`` (``m32_recipe_eval.py
   --stage injected --recipes r0 --folds 0``), asserts the F2 record,
   injection table and exclusions sha256, refuses any file or row outside r0
   fold 0 (exit 2, nothing else is read) and writes ``precheck.json``. Exit 0
   pass, 1 fail. Stdout: ``{"status", "precheck", "elapsed_s"}``.
2. The verdict call (full P5b in ``injected/``) recomputes the pre-check
   from its own r0 fold-0 rows; ``precheck.json`` must exist and hold the
   same values (else exit 2). A failed pre-check stops before any recipe
   comparison is computed.

Amendment 9 (plan 4-9)

* Weighted cells are IDS/A0 1/2 and 4TU/A0 1/2 for C-1-C-5 and G-in; 4TU/A1
  is reported only (``report_only_cell_4tu_a1``). C-6 keeps every group.
* Rows with ``main_cell`` False (truth box clipped or the -20 dB wing past
  the record end - 1 P) leave every main-cell computation and count only in
  C-6 (the bottom stratum, any size).
* The A1/A0 (INJ-A) row is used only when the r0 4TU/A1 recall at 12 dB
  (polarity mean, unstratified, 5-fold mean) is >= 30 %; otherwise it is
  "A1 정보 없음" and never holds the adoption.
* Strong-signal robustness ``R(18 dB) - R(12 dB)`` per recipe and cell is
  reported only; C-1 keeps the four SNR steps.

Identity gate (plan C-8, amendment 5): the gate is the P5a batch-1 run of
r0 x K lines x folds (``uninjected/gate.json`` and the ``identity_gate``
record of ``uninjected/summary.json``); a missing record is refused (exit 2)
and a failed one makes the verdict ``invalid_identity_gate``. The batch-64
r0 M6 values are compared with D3 for the report only; a difference is
disclosed, never a failure.

Aggregation (one function, ``fold_recalls``; [CVR] 4-2 "집계 함수는 하나")

* Cell = (fold, dataset, model group, SNR, polarity) over accepted
  injections with ``k`` 1.3; recall = hits / n with the one-to-one match at
  conf 0.25. The rows are the in-distribution stratum (width only, level
  ``l_W``) when the F2 path is ``primary_stratum="in"``; with
  ``"in_or_out"`` (fewer than 400 in-distribution positions, user decision
  2026-11-14) every size counts and the in-distribution guard
  ``G-in``: ``Delta_in >= -2.0 %p`` (point estimate) is added.
* Per fold: polarity mean (stage B drop <= 5 %p, plan 4-8; with
  ``POLARITY_RULE = "each"`` every criterion runs on each polarity), then SNR
  0/6/12/18 unweighted mean, then the group mean inside a dataset (4TU/A0
  alone since amendment 9), then the mean of 4TU and IDS.
  ``Delta = mean over folds of R(r) - R(r0)``.
* Post-stratification (amendment 7, [CVR] D-3): every (cell, SNR, polarity)
  recall is ``pi R_bottom + (1 - pi) R_rest`` with the F2 ``pi_bottom`` of
  its (dataset, group) cell (one stratum empty: the other one) for C-1-C-5
  and G-in; ``unstratified`` values are reported next to them. G-in uses the
  C-1 cell weights over valid cells (>= 10 in-distribution positions at
  every SNR and polarity), re-weighted inside a dataset; it is
  not evaluable (fail) only when a dataset has no valid cell (with 4TU/A0
  the only 4TU cell, an invalid 4TU/A0 is enough).
* All values are exact ``Fraction`` numbers and thresholds are Fractions, so
  a boundary value (e.g. exactly +3 %p) is decided without rounding.
* A criterion whose input set is empty (a cell with n = 0, no M6 apex, no
  line) is ``not_evaluable`` and never passes (empty-set guard).

Interpretations (reported in ``settings.interpretations``)

* C-3: lines are resampled with replacement within each dataset (4TU, IDS);
  one draw weights every cell of a line, so all five folds and both
  recipes share it; the 95 % lower bound is the ``floor(0.025 (B - 1))``-th
  sorted replicate (upper: the mirrored index); an undefined replicate is
  minus infinity for a lower and plus infinity for an upper bound.
* C-5: the conf grid is 0.25-0.95 in 0.01 steps (``score > conf`` before
  the merge); only confs >= 0.25 are searched, so a candidate with fewer
  boxes than r0 stays at 0.25 (the stricter reading). No grid level -> fail.
* C-6: bottom stratum (apex within 3 ``P_line`` of the record end, any size
  stratum, ``k`` 1.3) pooled over SNR, polarity, dataset and group per fold.
  ``bottom_rule="point"`` (>= 100 positions): point >= -3.0 %p.
  ``"upper_bound"``: fail only if the bootstrap 95 % upper bound < -3.0 %p.
* C-7: all P5a lines (4TU stage A + K, IDS); a box counts once in
  ``M4a + M4b + M4d`` (category order a > b > d); "M4 line" = a line with
  such a box in any fold; the fold limit is ``r0 + max(3, 10 % of r0)``.
* A1/A0: "1st recipe" compares r0, r5, r6 by the 4TU A1 (A0) recall; tied
  leaders form a set, and different sets count as a split. Comparisons use
  unstratified values (plan 4-3-1).
* C-3 bootstrap: only the cells of the scope are resampled; a line with
  report-only cells alone is left out of the line clusters.
"""

import argparse
import csv
import json
import logging
import math
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common
import m32_recipe_inputs as inputs
import m32_recipe_match as match

DEFAULT_RUN_ROOT = inputs.REPO / "runs" / "eval" / "m32_recipe"
DEFAULT_F2_RECORD = inputs.REPO / "analysis" / "detection" / "m32_recipe_f2.json"
SCHEMA = "m32_recipe_judge/1"
PRECHECK_SCHEMA = "m32_recipe_precheck/1"
PRECHECK_STAGE_DIR = "injected_precheck"
"""P5b folder of ``--recipes r0 --folds 0`` (``m32_recipe_eval`` writes it)."""
PRECHECK_FILE = "precheck.json"

SNR_STEPS = (0.0, 6.0, 12.0, 18.0)
POLARITIES = (1, -1)
K_MAIN = 1.3
K_REPORT = 0.8
POLARITY_RULE = "mean"
"""Plan 4-8: stage-B polarity change +0.46 %p (<= 5 %p) -> mean of +-1."""

C1_MIN = Fraction(3, 100)
C2_MIN_POSITIVE_FOLDS = 4
C2_WORST_FOLD_MIN = Fraction(-2, 100)
C6_MIN = Fraction(-3, 100)
C7_LINE_SLACK = 1
C7_FOLD_SLACK = 3
C7_FOLD_SHARE = Fraction(1, 10)
C7_M4C_FACTOR = Fraction(11, 10)
C8_SUM_SLACK = 6
C8_ANY_FOLD_SLACK = 2
GATE_BATCH_SIZE = 1
"""Plan C-8: the gate runs at batch size 1 (``m32_recipe_eval.GATE_BATCH_SIZE``)."""
GATE_KEYS = ("n_apexes", "sum_hits", "any_fold", "fold0")
TIE_MARGIN = Fraction(5, 1000)
IN_GUARD_MIN = Fraction(-2, 100)
"""In-or-out path: in-distribution Delta >= -2.0 %p (amendment 5)."""
PRIMARY_IN = "in"
PRIMARY_IN_OR_OUT = "in_or_out"
BOTTOM_POINT = "point"
BOTTOM_UPPER = "upper_bound"
DECISION_KEYS = ("primary_stratum", "bottom_rule", "pi_bottom")
DEFAULT_DECISIONS = {
    "primary_stratum": PRIMARY_IN,
    "bottom_rule": BOTTOM_POINT,
    "pi_bottom": None,
}
"""``pi_bottom`` None (tests only) means no post-stratification."""
G_IN_MIN_PER_CONDITION = 10
"""G-in: a (dataset, group) cell is valid with >= 10 in-distribution
positions at every (SNR, polarity) ([CVR] appendix D-2)."""
INJ_A_GAP = Fraction(3, 100)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_LOWER_SHARE = 0.025
EXIT_CODES = {"ok": 0, "precheck_failed": 1, "input_error": 2, "error": 3}

GROUPS_BY_DATASET = {inputs.DATASET_4TU: ("A0",), inputs.DATASET_IDS: ("A0",)}
"""Weighted C-1 cells (amendment 9): IDS/A0 1/2, 4TU/A0 1/2."""
REPORT_ONLY_GROUPS = {inputs.DATASET_4TU: ("A1",)}
"""Amendment 9 (9-3): 4TU/A1 leaves every weighted criterion; reported only."""
INJ_A_MIN_A1_RECALL = Fraction(30, 100)
"""INJ-A row only when the r0 4TU/A1 12 dB recall (polarity mean, unstratified,
5-fold mean) is >= 30 % (amendment 9, F1')."""
INJ_A_SNR_DB = 12.0
STRONG_SIGNAL_SNRS = (12.0, 18.0)
"""Strong-signal robustness (amendment 9, 9-1): R(18 dB) - R(12 dB), report."""
PRECHECK_FOLD = 0
PRECHECK_SNR_DB = 12.0
PRECHECK_MIN_RECALL = Fraction(1, 2)
PRECHECK_MONOTONE_SNRS = (0.0, 6.0, 12.0)
"""Amendment 9 (9-1): monotone from 0 to 12 dB only."""
PRECHECK_TOLERANCE = Fraction(1, 100)
"""A later step may be at most 1 %p below an earlier one."""

logger = logging.getLogger("m32_recipe_judge")


# ----------------------------------------------------------------- inputs


@dataclass(frozen=True)
class MatchRow:
    """The judge's view of one ``inj_matches.csv`` row."""

    recipe: str
    fold: int
    injection_id: str
    line_id: str
    dataset: str
    group: str
    snr_db: float
    polarity: int
    k: float
    in_distribution: bool
    bottom_stratum: bool
    in_horizontal_box_zone: bool
    hit: bool
    hit_any_class: bool
    hit_mask: str
    main_cell: bool = True
    """False: box clipped or wing past the record end - 1 P (amendment 9);
    such rows count only in C-6 (bottom stratum)."""


def _flag(text: str) -> bool:
    if text in ("True", "true", "1"):
        return True
    if text in ("False", "false", "0", ""):
        return False
    raise ValueError(f"not a boolean: {text!r}")


def match_row(row: Mapping[str, str]) -> MatchRow:
    """Parse one csv row (raises ``ValueError`` on a malformed value)."""
    return MatchRow(
        recipe=row["recipe"],
        fold=int(row["fold"]),
        injection_id=row["injection_id"],
        line_id=row["line_id"],
        dataset=row["dataset"],
        group=row["model_group"],
        snr_db=float(row["snr_db"]),
        polarity=int(row["polarity"]),
        k=float(row["k"]),
        in_distribution=_flag(row["in_distribution"]),
        bottom_stratum=_flag(row["bottom_stratum"]),
        in_horizontal_box_zone=_flag(row["in_horizontal_box_zone"]),
        hit=_flag(row["hit"]),
        hit_any_class=_flag(row["hit_any_class"]),
        hit_mask=row["hit_mask"],
        main_cell=_flag(row["main_cell"]),
    )


@dataclass(frozen=True)
class RunData:
    """Everything the criteria read."""

    matches: list[MatchRow]
    conf_counts: list[dict[str, str]]
    boxes: list[dict[str, str]]
    apex_hits: list[dict[str, str]]
    lines: list[dict[str, str]]
    sha256: dict[str, str]
    decisions: dict = field(default_factory=lambda: dict(DEFAULT_DECISIONS))
    """F2 choices copied by P5b: ``primary_stratum`` and ``bottom_rule``."""
    gate_record: Mapping | None = None
    """P5a batch-1 gate result (``observed``, ``expected``, ``pass``); ``None``
    counts as a failed gate."""
    batch_path: Mapping | None = None
    """P5a batch vs batch-1 report on r0 x K (C-8 disclosure only)."""


def _stage_rows(
    stage_dir: Path, name: str, recipes: Sequence[str], folds: Sequence[int]
) -> tuple[list[dict[str, str]], dict[str, str]]:
    rows, hashes = [], {}
    for recipe in recipes:
        for fold in folds:
            path = stage_dir / recipe / f"fold{fold}" / f"{name}.csv"
            if not path.is_file():
                raise inputs.InputError(f"missing runner output {path}")
            hashes[path.relative_to(stage_dir.parent).as_posix()] = inputs.sha256_file(
                path
            )
            rows += inputs.read_csv(path)
    return rows, hashes


def _require_final(stage_dir: Path) -> tuple[str, dict]:
    path = stage_dir / "summary.json"
    if not path.is_file():
        raise inputs.InputError(f"missing {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "ok" or not summary.get("full_run"):
        raise inputs.InputError(f"{path}: not a finished full run")
    return inputs.sha256_file(path), summary


def read_decisions(summary: Mapping) -> dict:
    """``primary_stratum`` and ``bottom_rule`` fixed at F2 (P5b summary)."""
    try:
        decisions = {key: summary[key] for key in DECISION_KEYS}
    except KeyError as error:
        raise inputs.InputError(f"P5b summary lacks the F2 choice {error}") from error
    if decisions["primary_stratum"] not in (PRIMARY_IN, PRIMARY_IN_OR_OUT):
        raise inputs.InputError(f"unknown primary_stratum {decisions}")
    if decisions["bottom_rule"] not in (BOTTOM_POINT, BOTTOM_UPPER):
        raise inputs.InputError(f"unknown bottom_rule {decisions}")
    if not isinstance(decisions["pi_bottom"], Mapping):
        raise inputs.InputError("pi_bottom must be a cell -> share mapping")
    return decisions


def read_gate_record(summary: Mapping, gate_path: Path) -> dict:
    """The P5a batch-1 identity gate result (plan C-8).

    Args:
        summary: The P5a ``summary.json`` (its ``identity_gate`` record).
        gate_path: The P5a ``gate.json`` written by the batch-1 gate step.

    Returns:
        The gate result with ``observed``, ``expected`` and ``pass``.

    Raises:
        inputs.InputError: A record is missing or malformed, ``gate.json`` was
            not a batch-1 run, or the two records disagree.
    """
    record = summary.get("identity_gate")
    is_well_formed = (
        isinstance(record, Mapping)
        and {"observed", "expected", "pass"} <= set(record)
        and isinstance(record["observed"], Mapping)
    )
    if not is_well_formed:
        raise inputs.InputError("P5a summary lacks the batch-1 identity gate record")
    if not gate_path.is_file():
        raise inputs.InputError(f"missing batch-1 gate record {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("batch_size") != GATE_BATCH_SIZE:
        raise inputs.InputError(f"{gate_path}: not a batch-size-1 gate run")
    if gate.get("result") != record:
        raise inputs.InputError(f"{gate_path}: gate result differs from the summary")
    return dict(record)


def check_pi_cells(pi_bottom: Mapping | None, matches: Iterable[MatchRow]) -> None:
    """Every weighted (dataset, group) cell with injections needs ``pi_bottom``.

    The report-only 4TU/A1 cell may lack it (its report then pools strata).
    """
    if pi_bottom is None:
        return
    cells = {
        f"{row.dataset}/{row.group}"
        for row in matches
        if row.group in GROUPS_BY_DATASET.get(row.dataset, ())
    }
    missing = sorted(cell for cell in cells if pi_bottom.get(cell) is None)
    if missing:
        raise inputs.InputError(f"pi_bottom missing for cells {missing}")


def load_run(
    run_root: Path,
    recipes: Sequence[str] = inputs.STAGE_C_RECIPES,
    folds: Sequence[int] = inputs.FOLDS,
) -> RunData:
    """Read and check the P5a and P5b outputs of one out-root."""
    uninjected, injected = run_root / "uninjected", run_root / "injected"
    uninjected_sha, uninjected_summary = _require_final(uninjected)
    injected_sha, injected_summary = _require_final(injected)
    gate_path = uninjected / "gate.json"
    gate_record = read_gate_record(uninjected_summary, gate_path)
    hashes = {
        "uninjected/summary.json": uninjected_sha,
        "uninjected/gate.json": inputs.sha256_file(gate_path),
        "injected/summary.json": injected_sha,
    }
    tables: dict[str, list[dict[str, str]]] = {}
    for stage_dir, name in (
        (injected, "inj_matches"),
        (uninjected, "conf_counts"),
        (uninjected, "boxes"),
        (uninjected, "apex_hits"),
    ):
        tables[name], file_hashes = _stage_rows(stage_dir, name, recipes, folds)
        hashes.update(file_hashes)
    lines_path = uninjected / "lines.csv"
    hashes["uninjected/lines.csv"] = inputs.sha256_file(lines_path)
    try:
        matches = [match_row(row) for row in tables["inj_matches"]]
    except (KeyError, ValueError) as error:
        raise inputs.InputError(f"malformed inj_matches: {error!r}") from error
    check_pairing(matches, recipes, folds)
    decisions = read_decisions(injected_summary)
    check_pi_cells(decisions["pi_bottom"], matches)
    return RunData(
        matches=matches,
        conf_counts=tables["conf_counts"],
        boxes=tables["boxes"],
        apex_hits=tables["apex_hits"],
        lines=inputs.read_csv(lines_path),
        sha256=hashes,
        decisions=decisions,
        gate_record=gate_record,
        batch_path=uninjected_summary.get("run_identity", {}).get("batch_path"),
    )


def check_pairing(
    matches: Sequence[MatchRow], recipes: Sequence[str], folds: Sequence[int]
) -> None:
    """Every recipe x fold must score the same injections (paired design)."""
    sets: dict[tuple[str, int], set[str]] = {}
    for row in matches:
        sets.setdefault((row.recipe, row.fold), set()).add(row.injection_id)
    reference = sets.get((recipes[0], folds[0]), set())
    if not reference:
        raise inputs.InputError("no injection rows")
    for recipe in recipes:
        for fold in folds:
            if sets.get((recipe, fold), set()) != reference:
                raise inputs.InputError(f"{recipe} fold {fold}: other injections")


# ------------------------------------------------------------- aggregation


CellKey = tuple[int, str, str, float, int, bool]
"""``(fold, dataset, model group, snr_db, polarity, bottom stratum)``."""

PiBottom = Mapping[str, float | Fraction | None] | None
"""``"dataset/group" -> natural bottom share``; None: no post-stratification."""

CellCounts = Mapping[tuple[str, CellKey], tuple[int, int]]
"""``(recipe, cell) -> (hits, n)``."""


@dataclass(frozen=True)
class Scope:
    """Which datasets, groups and polarities a recall averages over."""

    datasets: tuple[str, ...]
    groups: Mapping[str, tuple[str, ...]]
    polarities: tuple[int, ...]


MAIN_SCOPE = Scope(
    datasets=(inputs.DATASET_4TU, inputs.DATASET_IDS),
    groups=GROUPS_BY_DATASET,
    polarities=POLARITIES,
)


def with_polarity(scope: Scope, polarity: int) -> Scope:
    """The same scope on one polarity only."""
    return Scope(scope.datasets, scope.groups, (polarity,))


def dataset_scope(dataset: str, groups: tuple[str, ...] | None = None) -> Scope:
    """One dataset (optionally one model group)."""
    return Scope(
        (dataset,), {dataset: groups or GROUPS_BY_DATASET[dataset]}, POLARITIES
    )


def _mean(values: Sequence[Fraction]) -> Fraction:
    return sum(values, Fraction(0)) / len(values)


def condition_recall(
    counts: CellCounts,
    recipe: str,
    key: tuple[int, str, str, float, int],
    pi: PiBottom,
) -> Fraction | None:
    """Recall of one (fold, dataset, group, SNR, polarity) condition.

    With ``pi`` the bottom and the other rows are post-stratified,
    ``R = pi R_bottom + (1 - pi) R_rest`` ([CVR] appendix D-3); an empty
    stratum leaves the other one. Without ``pi`` the rows are pooled.
    """
    bottom = counts.get((recipe, (*key, True)), (0, 0))
    rest = counts.get((recipe, (*key, False)), (0, 0))
    share = None if pi is None else pi.get(f"{key[1]}/{key[2]}")
    if share is None or bottom[1] == 0 or rest[1] == 0:
        n = bottom[1] + rest[1]
        return Fraction(bottom[0] + rest[0], n) if n else None
    weight = Fraction(share)
    return weight * Fraction(*bottom) + (1 - weight) * Fraction(*rest)


def fold_recalls(
    counts: CellCounts,
    recipe: str,
    scope: Scope,
    folds: Sequence[int],
    snr_steps: Sequence[float] = SNR_STEPS,
    pi: PiBottom = None,
) -> list[Fraction] | None:
    """Per-fold recall: polarity, SNR, group, dataset unweighted means.

    Returns:
        One Fraction per fold, or ``None`` when any needed condition is empty.
    """
    per_fold = []
    for fold in folds:
        dataset_values = []
        for dataset in scope.datasets:
            group_values = []
            for group in scope.groups[dataset]:
                snr_values = []
                for snr in snr_steps:
                    polarity_values = []
                    for polarity in scope.polarities:
                        value = condition_recall(
                            counts, recipe, (fold, dataset, group, snr, polarity), pi
                        )
                        if value is None:
                            return None
                        polarity_values.append(value)
                    snr_values.append(_mean(polarity_values))
                group_values.append(_mean(snr_values))
            dataset_values.append(_mean(group_values))
        per_fold.append(_mean(dataset_values))
    return per_fold


@dataclass(frozen=True)
class Difference:
    """``R(r) - R(r0)`` per fold and its mean (``None`` if not evaluable)."""

    per_fold: list[Fraction] | None
    mean: Fraction | None


def difference(
    counts: CellCounts,
    recipe: str,
    scope: Scope,
    folds: Sequence[int],
    snr_steps: Sequence[float] = SNR_STEPS,
    pi: PiBottom = None,
) -> Difference:
    """Paired recall difference of ``recipe`` against r0."""
    candidate = fold_recalls(counts, recipe, scope, folds, snr_steps, pi)
    baseline = fold_recalls(counts, inputs.R0, scope, folds, snr_steps, pi)
    if candidate is None or baseline is None:
        return Difference(None, None)
    per_fold = [a - b for a, b in zip(candidate, baseline, strict=True)]
    return Difference(per_fold, _mean(per_fold))


HitSelector = Callable[[MatchRow], bool]


def base_hit(row: MatchRow) -> bool:
    """One-to-one ``pipe_hyperbola`` hit at conf 0.25."""
    return row.hit


def mask_hit(index: int) -> HitSelector:
    """Hit at grid level ``index`` of ``match.conf_grid()``."""

    def select(row: MatchRow) -> bool:
        return row.hit_mask[index] == "1"

    return select


def all_sizes_stratum(row: MatchRow) -> bool:
    """Main-cell rows of every size at ``k`` 1.3 (C-1 on the in-or-out path).

    Rows outside the main cells (amendment 9) count only in C-6.
    """
    return math.isclose(row.k, K_MAIN) and row.main_cell


def main_stratum(row: MatchRow) -> bool:
    """In-distribution main-cell rows at ``k`` 1.3 (C-1 on the ``"in"`` path)."""
    return all_sizes_stratum(row) and row.in_distribution


def out_stratum(row: MatchRow) -> bool:
    """Out-of-distribution main-cell rows at ``k`` 1.3 (stratum report)."""
    return all_sizes_stratum(row) and not row.in_distribution


def primary_filter(primary_stratum: str) -> Callable[[MatchRow], bool]:
    """Rows of the C-1-C-8 main metric for the F2 path (amendment 5)."""
    return main_stratum if primary_stratum == PRIMARY_IN else all_sizes_stratum


def count_cells(
    rows: Iterable[MatchRow],
    keep: Callable[[MatchRow], bool],
    hit_by_recipe: Mapping[str, HitSelector] | None = None,
) -> dict[str, dict[tuple[str, CellKey], list[int]]]:
    """Per-line ``(recipe, cell) -> [hits, n]`` of the kept rows."""
    per_line: dict[str, dict[tuple[str, CellKey], list[int]]] = {}
    hits_of = hit_by_recipe or {}
    for row in rows:
        if not keep(row):
            continue
        condition = (row.fold, row.dataset, row.group, row.snr_db, row.polarity)
        key = (row.recipe, (*condition, row.bottom_stratum))
        tally = per_line.setdefault(row.line_id, {}).setdefault(key, [0, 0])
        tally[0] += int(hits_of.get(row.recipe, base_hit)(row))
        tally[1] += 1
    return per_line


def total_counts(
    per_line: Mapping[str, Mapping[tuple[str, CellKey], Sequence[int]]],
) -> dict[tuple[str, CellKey], tuple[int, int]]:
    """Sum the per-line counts."""
    totals: dict[tuple[str, CellKey], list[int]] = {}
    for cells in per_line.values():
        for key, (hits, n) in cells.items():
            total = totals.setdefault(key, [0, 0])
            total[0] += hits
            total[1] += n
    return {key: (value[0], value[1]) for key, value in totals.items()}


# --------------------------------------------------------------- bootstrap


def draw_line_weights(
    rng: np.random.Generator, strata: Sequence[np.ndarray], n_lines: int
) -> np.ndarray:
    """One resample of lines (with replacement inside each dataset stratum).

    The same weights multiply every cell of a line, i.e. all five folds use
    one resample ([CVR] appendix C-2): folds share the injected objects.
    """
    weights = np.zeros(n_lines, dtype=np.int64)
    for members in strata:
        if members.size:
            drawn = rng.integers(0, members.size, members.size)
            np.add.at(weights, members[drawn], 1)
    return weights


def replicate_counts(
    per_line: Mapping[str, Mapping[tuple[str, CellKey], Sequence[int]]],
    line_datasets: Mapping[str, str],
    datasets: Sequence[str],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = inputs.SEED,
) -> Iterator[dict[tuple[str, CellKey], tuple[int, int]]]:
    """Cell counts of each bootstrap replicate (line clusters, seed fixed)."""
    lines = sorted(per_line)
    keys = sorted({key for cells in per_line.values() for key in cells})
    hits = np.array(
        [[per_line[line].get(k, (0, 0))[0] for k in keys] for line in lines]
    ).reshape(len(lines), len(keys))
    trials = np.array(
        [[per_line[line].get(k, (0, 0))[1] for k in keys] for line in lines]
    ).reshape(len(lines), len(keys))
    strata = [
        np.array(
            [i for i, line in enumerate(lines) if line_datasets[line] == dataset],
            dtype=np.int64,
        )
        for dataset in datasets
    ]
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        weights = draw_line_weights(rng, strata, len(lines))
        yield {
            key: (int(h), int(n))
            for key, h, n in zip(keys, weights @ hits, weights @ trials, strict=True)
        }


def bootstrap_bounds(
    per_line: Mapping[str, Mapping[tuple[str, CellKey], Sequence[int]]],
    line_datasets: Mapping[str, str],
    datasets: Sequence[str],
    evaluate: Callable[[CellCounts], Fraction | None],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = inputs.SEED,
) -> dict:
    """95 % lower and upper order statistics of a bootstrapped statistic."""
    values = [
        evaluate(counts)
        for counts in replicate_counts(
            per_line, line_datasets, datasets, replicates, seed
        )
    ]
    return order_statistics(values, replicates)


def bootstrap_lower_bound(
    per_line: Mapping[str, Mapping[tuple[str, CellKey], Sequence[int]]],
    line_datasets: Mapping[str, str],
    recipe: str,
    scope: Scope,
    folds: Sequence[int],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = inputs.SEED,
    pi: PiBottom = None,
) -> dict:
    """C-3: line-cluster bootstrap of ``Delta`` (see module docstring).

    Each replicate: one line resample for all five folds, then the
    post-stratification and cell weights, then the fold mean.
    """

    def mean_delta(counts: CellCounts) -> Fraction | None:
        return difference(counts, recipe, scope, folds, pi=pi).mean

    return bootstrap_bounds(
        per_line, line_datasets, scope.datasets, mean_delta, replicates, seed
    )


def order_statistics(values: Sequence[Fraction | None], replicates: int) -> dict:
    """2.5 % and 97.5 % order statistics; an undefined replicate is -inf for
    the lower and +inf for the upper bound (both conservative directions)."""
    defined = sorted(value for value in values if value is not None)
    undefined = len(values) - len(defined)
    index = math.floor(BOOTSTRAP_LOWER_SHARE * (replicates - 1))
    upper_index = replicates - 1 - index
    lower = defined[index - undefined] if index >= undefined and defined else None
    upper = defined[upper_index] if upper_index < len(defined) else None
    return {
        "lower": lower,
        "upper": upper,
        "index": index,
        "upper_index": upper_index,
        "replicates": replicates,
        "n_undefined": undefined,
    }


# ---------------------------------------------------------- P5a criteria


def conf_box_totals(rows: Iterable[Mapping[str, str]]) -> dict[str, list[int]]:
    """C-5: ``pipe_hyperbola`` boxes per grid level, summed over lines and folds."""
    levels = match.conf_grid()
    totals: dict[str, list[int]] = {}
    for row in rows:
        values = totals.setdefault(row["recipe"], [0] * len(levels))
        for index, level in enumerate(levels):
            values[index] += int(row[f"c{level:.2f}"])
    return totals


def operating_index(totals: Mapping[str, Sequence[int]], recipe: str) -> int | None:
    """Smallest grid conf whose box total is <= r0's at 0.25 (None if none)."""
    target = totals[inputs.R0][0]
    for index, value in enumerate(totals[recipe]):
        if value <= target:
            return index
    return None


@dataclass(frozen=True)
class M4Counts:
    """C-7 inputs of one recipe."""

    per_fold: dict[int, int]
    total: int
    lines: int
    m4c_total: int


def m4_counts(
    boxes: Iterable[Mapping[str, str]], recipe: str, folds: Sequence[int]
) -> M4Counts:
    """``M4a + M4b + M4d`` per fold, M4 lines and M4c (original orientation)."""
    per_fold = dict.fromkeys(folds, 0)
    lines, m4c = set(), 0
    for row in boxes:
        if row["recipe"] != recipe or row["orientation"] != "original":
            continue
        if row["m4"]:
            per_fold[int(row["fold"])] += 1
            lines.add(row["line_id"])
        m4c += int(_flag(row["m4c"]))
    return M4Counts(per_fold, sum(per_fold.values()), len(lines), m4c)


def c7_check(candidate: M4Counts, baseline: M4Counts) -> dict:
    """C-7 (Fraction comparisons)."""
    fold_limits = {
        fold: baseline.per_fold[fold]
        + max(Fraction(C7_FOLD_SLACK), C7_FOLD_SHARE * baseline.per_fold[fold])
        for fold in baseline.per_fold
    }
    checks = {
        "total_le_r0": candidate.total <= baseline.total,
        "lines_le_r0_plus_1": candidate.lines <= baseline.lines + C7_LINE_SLACK,
        "every_fold_within_limit": all(
            candidate.per_fold[fold] <= limit for fold, limit in fold_limits.items()
        ),
        "m4c_le_110pct": candidate.m4c_total <= C7_M4C_FACTOR * baseline.m4c_total,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "candidate": candidate.__dict__,
        "r0": baseline.__dict__,
        "fold_limits": {str(f): _number(v) for f, v in fold_limits.items()},
    }


def m6_counts(rows: Iterable[Mapping[str, str]], recipe: str) -> dict:
    """M6 sum over folds, apexes hit by any fold, fold 0, per fold."""
    folds_hit: dict[str, int] = {}
    per_fold: dict[int, int] = {}
    for row in rows:
        if row["recipe"] != recipe:
            continue
        hit = _flag(row["hit"])
        folds_hit[row["blind_id"]] = folds_hit.get(row["blind_id"], 0) + int(hit)
        fold = int(row["fold"])
        per_fold[fold] = per_fold.get(fold, 0) + int(hit)
    return {
        "n_apexes": len(folds_hit),
        "sum_hits": sum(folds_hit.values()),
        "any_fold": sum(value > 0 for value in folds_hit.values()),
        "fold0": per_fold.get(0, 0),
        "per_fold": per_fold,
    }


def d3_values() -> dict[str, int]:
    """D3 values of r0 on the K apexes (41 apexes, 90/205, 30/41, fold 0 15/41)."""
    return {
        "n_apexes": inputs.N_K_APEXES,
        "sum_hits": inputs.GATE_SUM_HITS,
        "any_fold": inputs.GATE_ANY_FOLD,
        "fold0": inputs.GATE_FOLD0,
    }


def identity_gate(record: Mapping | None) -> dict:
    """C-8 gate from the P5a batch-1 record (the batch path never decides it).

    The judge does not trust the runner's ``pass`` alone: the recorded
    observation must also equal the D3 values held here.
    """
    if record is None:
        return {"pass": False, "reason": "no_batch1_gate_record"}
    observed = {key: record["observed"].get(key) for key in GATE_KEYS}
    expected = d3_values()
    return {
        "pass": record["pass"] is True and observed == expected,
        "source": "uninjected/gate.json (batch size 1)",
        "batch_size": GATE_BATCH_SIZE,
        "observed": observed,
        "expected": expected,
    }


def d3_comparison(measured: Mapping, batch_path: Mapping | None = None) -> dict:
    """C-8 report: batch-path r0 M6 against D3 (disclosed, never a failure)."""
    expected = d3_values()
    observed = {key: measured[key] for key in GATE_KEYS}
    differences = {key: observed[key] - expected[key] for key in GATE_KEYS}
    return {
        "observed_batch_path": observed,
        "d3": expected,
        "differences": differences,
        "n_differing_values": sum(value != 0 for value in differences.values()),
        "equal": observed == expected,
        "batch_vs_batch1": None if batch_path is None else dict(batch_path),
        "note": "report only (plan C-8): batch vs batch 1 is disclosed, not a failure",
    }


def c8_check(candidate: Mapping, baseline: Mapping) -> dict:
    """C-8: 5-fold sum >= r0 - 6, any fold >= r0 - 2 (fold 0 report only)."""
    if candidate["n_apexes"] != inputs.N_K_APEXES:
        return {"pass": False, "reason": "not_evaluable", "candidate": dict(candidate)}
    checks = {
        "sum_ge_r0_minus_6": candidate["sum_hits"]
        >= baseline["sum_hits"] - C8_SUM_SLACK,
        "any_fold_ge_r0_minus_2": candidate["any_fold"]
        >= baseline["any_fold"] - C8_ANY_FOLD_SLACK,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "candidate": dict(candidate),
        "r0": dict(baseline),
        "fold0_report_only": candidate["fold0"],
    }


# ------------------------------------------------------------- verdict


def _number(value: Fraction | None) -> dict | None:
    if value is None:
        return None
    return {"value": float(value), "exact": f"{value.numerator}/{value.denominator}"}


def _criterion(value: Fraction | None, passed: bool, threshold: str) -> dict:
    if value is None:
        return {"pass": False, "reason": "not_evaluable", "threshold": threshold}
    return {"pass": passed, "value": _number(value), "threshold": threshold}


@dataclass(frozen=True)
class JudgeInputs:
    """Precomputed counts shared by every candidate."""

    data: RunData
    main_per_line: dict
    main_counts: dict
    line_datasets: dict[str, str]
    conf_totals: dict[str, list[int]]
    folds: tuple[int, ...]
    keep: Callable[[MatchRow], bool] = main_stratum
    primary_stratum: str = PRIMARY_IN
    bottom_rule: str = BOTTOM_POINT
    pi: PiBottom = None


def prepare(data: RunData, folds: Sequence[int] = inputs.FOLDS) -> JudgeInputs:
    """Counts of the F2 main stratum and the P5a operating-point totals."""
    keep = primary_filter(data.decisions["primary_stratum"])
    per_line = count_cells(data.matches, keep)
    return JudgeInputs(
        data=data,
        main_per_line=per_line,
        main_counts=total_counts(per_line),
        line_datasets={row.line_id: row.dataset for row in data.matches},
        conf_totals=conf_box_totals(data.conf_counts),
        folds=tuple(folds),
        keep=keep,
        primary_stratum=data.decisions["primary_stratum"],
        bottom_rule=data.decisions["bottom_rule"],
        pi=data.decisions.get("pi_bottom"),
    )


STRATIFIED_CRITERIA = ("C-1", "C-2", "C-3", "C-4", "C-5", "G-in")
"""Post-stratified criteria; each also reports its ``unstratified`` value
(C-2 reads the per-fold values of the stratified C-1)."""


def injection_criteria(ji: JudgeInputs, recipe: str, scope: Scope) -> dict:
    """C-1 to C-6 (and G-in) of one candidate on one polarity scope.

    Pass decisions use the post-stratified values (``pi_bottom`` of F2); the
    criteria of ``STRATIFIED_CRITERIA`` also carry the unstratified value.
    C-2 follows the stratified C-1 per fold; C-6 is the bottom stratum alone.
    """
    criteria = _criteria_with(ji, recipe, scope, ji.pi)
    criteria["C-6"] = _c6(ji, recipe, scope)
    if ji.pi is not None:
        plain = _criteria_with(ji, recipe, scope, None)
        for name in STRATIFIED_CRITERIA:
            if name in criteria:
                criteria[name]["unstratified"] = _without_pass(plain[name])
    return criteria


def _without_pass(criterion: Mapping) -> dict:
    return {key: value for key, value in criterion.items() if key != "pass"}


def _criteria_with(ji: JudgeInputs, recipe: str, scope: Scope, pi: PiBottom) -> dict:
    main = difference(ji.main_counts, recipe, scope, ji.folds, pi=pi)
    criteria: dict[str, dict] = {
        "C-1": _criterion(
            main.mean, main.mean is not None and main.mean >= C1_MIN, ">= +3.0 %p"
        ),
        "C-2": _c2(main),
        "C-3": _c3(ji, recipe, scope, pi),
        "C-4": _c4(ji, recipe, scope, pi),
        "C-5": _c5(ji, recipe, scope, pi),
    }
    if ji.primary_stratum == PRIMARY_IN_OR_OUT:
        criteria["G-in"] = in_distribution_guard(ji, recipe, scope, pi)
    return criteria


def g_in_cells(ji: JudgeInputs, scope: Scope) -> dict[str, dict]:
    """In-distribution positions per (dataset, group) and their validity.

    Positions are counted on r0 of the first fold (every recipe and fold
    scores the same injections); a cell is valid with at least 10 at every
    (SNR, polarity) of the scope ([CVR] appendix D-2).
    """
    counts: dict[tuple[str, str, float, int], int] = {}
    for row in ji.data.matches:
        if row.recipe == inputs.R0 and row.fold == ji.folds[0] and main_stratum(row):
            key = (row.dataset, row.group, row.snr_db, row.polarity)
            counts[key] = counts.get(key, 0) + 1
    cells = {}
    for dataset in scope.datasets:
        for group in scope.groups[dataset]:
            per_condition = [
                counts.get((dataset, group, snr, polarity), 0)
                for snr in SNR_STEPS
                for polarity in scope.polarities
            ]
            cells[f"{dataset}/{group}"] = {
                "dataset": dataset,
                "group": group,
                "min_n": min(per_condition),
                "valid": min(per_condition) >= G_IN_MIN_PER_CONDITION,
            }
    return cells


def valid_cell_scope(cells: Mapping[str, Mapping], scope: Scope) -> Scope | None:
    """Scope of the valid cells; ``None`` when a dataset has none left.

    Dropping a cell re-weights inside its dataset (the datasets keep 1/2).
    """
    groups = {
        dataset: tuple(
            cell["group"]
            for cell in cells.values()
            if cell["dataset"] == dataset and cell["valid"]
        )
        for dataset in scope.datasets
    }
    if any(not members for members in groups.values()):
        return None
    return Scope(scope.datasets, groups, scope.polarities)


def _cell_weights(valid: Scope) -> dict[str, Fraction]:
    share = Fraction(1, len(valid.datasets))
    return {
        f"{dataset}/{group}": share / len(valid.groups[dataset])
        for dataset in valid.datasets
        for group in valid.groups[dataset]
    }


def _cell_delta(
    counts: CellCounts,
    recipe: str,
    cell: Mapping,
    scope: Scope,
    ji: JudgeInputs,
    pi: PiBottom,
) -> dict | None:
    """Delta of one (dataset, group) cell (G-in report)."""
    dataset, group = cell["dataset"], cell["group"]
    one_cell = Scope((dataset,), {dataset: (group,)}, scope.polarities)
    return _number(difference(counts, recipe, one_cell, ji.folds, pi=pi).mean)


def in_distribution_guard(
    ji: JudgeInputs, recipe: str, scope: Scope, pi: PiBottom = None
) -> dict:
    """In-or-out path: in-distribution ``Delta_in >= -2.0 %p`` (point estimate).

    C-1 cell weights over the valid (dataset, group) cells, re-weighted in a
    dataset when a cell is invalid; not evaluable (and failed) only when a
    whole dataset has no valid cell ([CVR] appendix D-2, D-4).
    """
    cells = g_in_cells(ji, scope)
    counts = total_counts(count_cells(ji.data.matches, main_stratum))
    valid = valid_cell_scope(cells, scope)
    pooled = pooled_difference(counts, recipe, ji.folds).mean
    threshold = ">= -2.0 %p (point)"
    report: dict = {
        "cells": {
            name: {**cell, "delta": _cell_delta(counts, recipe, cell, scope, ji, pi)}
            for name, cell in cells.items()
        },
        "excluded_cells": sorted(
            name for name, cell in cells.items() if not cell["valid"]
        ),
        "pooled_check": _number(pooled),
        "threshold": threshold,
    }
    if valid is None:
        return {"pass": False, "reason": "not_evaluable", **report}
    value = difference(counts, recipe, valid, ji.folds, pi=pi).mean
    report["weights"] = {name: _number(w) for name, w in _cell_weights(valid).items()}
    return {
        **_criterion(value, value is not None and value >= IN_GUARD_MIN, threshold),
        **report,
    }


def _c2(main: Difference) -> dict:
    if main.per_fold is None:
        return {"pass": False, "reason": "not_evaluable"}
    positive = sum(value > 0 for value in main.per_fold)
    worst = min(main.per_fold)
    return {
        "pass": positive >= C2_MIN_POSITIVE_FOLDS and worst >= C2_WORST_FOLD_MIN,
        "positive_folds": positive,
        "worst_fold": _number(worst),
        "per_fold": [_number(v) for v in main.per_fold],
        "threshold": ">= 4/5 positive and worst fold >= -2.0 %p",
    }


def _c3(ji: JudgeInputs, recipe: str, scope: Scope, pi: PiBottom = None) -> dict:
    per_line = _scope_lines(ji.main_per_line, scope)
    result = bootstrap_lower_bound(
        per_line, ji.line_datasets, recipe, scope, ji.folds, pi=pi
    )
    lower = result["lower"]
    return {
        **_criterion(lower, lower is not None and lower > 0, "95 % lower bound > 0"),
        "replicates": result["replicates"],
        "order_statistic_index": result["index"],
        "n_undefined": result["n_undefined"],
    }


def _in_scope(cell: CellKey, scope: Scope) -> bool:
    _, dataset, group, _, polarity, _ = cell
    return (
        dataset in scope.datasets
        and group in scope.groups[dataset]
        and polarity in scope.polarities
    )


def _scope_lines(per_line: Mapping, scope: Scope) -> dict:
    """Bootstrap input: cells of the scope only, lines left empty dropped.

    A line with only report-only (4TU/A1) cells carries no weighted data, so
    resampling it would only dilute the line clusters (amendment 9).
    """
    kept = {}
    for line, cells in per_line.items():
        in_scope = {
            key: value for key, value in cells.items() if _in_scope(key[1], scope)
        }
        if in_scope:
            kept[line] = in_scope
    return kept


def _c4(ji: JudgeInputs, recipe: str, scope: Scope, pi: PiBottom = None) -> dict:
    by_dataset: dict[str, Fraction | None] = {}
    for dataset in scope.datasets:
        sub = Scope((dataset,), {dataset: scope.groups[dataset]}, scope.polarities)
        by_dataset[dataset] = difference(
            ji.main_counts, recipe, sub, ji.folds, pi=pi
        ).mean
    values = [value for value in by_dataset.values() if value is not None]
    evaluable = len(values) == len(by_dataset)
    result = {
        "pass": evaluable and all(value >= 0 for value in values),
        "by_dataset": {name: _number(value) for name, value in by_dataset.items()},
        "threshold": ">= 0 on 4TU and on IDS",
    }
    if not evaluable:
        result["reason"] = "not_evaluable"
    return result


def _c5(ji: JudgeInputs, recipe: str, scope: Scope, pi: PiBottom = None) -> dict:
    if recipe not in ji.conf_totals or inputs.R0 not in ji.conf_totals:
        return {"pass": False, "reason": "not_evaluable"}
    index = operating_index(ji.conf_totals, recipe)
    if index is None:
        return {"pass": False, "reason": "no_conf_level_reaches_r0_box_count"}
    hits = {recipe: mask_hit(index), inputs.R0: base_hit}
    counts = total_counts(count_cells(ji.data.matches, ji.keep, hits))
    value = difference(counts, recipe, scope, ji.folds, pi=pi).mean
    return {
        **_criterion(value, value is not None and value > 0, "> 0 at the matched conf"),
        "conf": match.conf_grid()[index],
        "box_total_at_conf": ji.conf_totals[recipe][index],
        "r0_box_total_at_025": ji.conf_totals[inputs.R0][0],
    }


def bottom_stratum_rows(scope: Scope) -> Callable[[MatchRow], bool]:
    """Bottom-stratum rows at ``k`` 1.3 (any size) inside the scope."""

    def keep(row: MatchRow) -> bool:
        return (
            row.bottom_stratum
            and math.isclose(row.k, K_MAIN)
            and row.polarity in scope.polarities
            and row.dataset in scope.datasets
        )

    return keep


def pooled_difference(
    counts: CellCounts, recipe: str, folds: Sequence[int]
) -> Difference:
    """Per-fold recall pooled over all cells, minus r0, and its fold mean."""
    per_fold = []
    for fold in folds:
        candidate = _pooled(counts, recipe, fold)
        baseline = _pooled(counts, inputs.R0, fold)
        if candidate is None or baseline is None:
            return Difference(None, None)
        per_fold.append(candidate - baseline)
    return Difference(per_fold, _mean(per_fold))


def _pooled(counts: CellCounts, recipe: str, fold: int) -> Fraction | None:
    hits = n = 0
    for (row_recipe, cell), (cell_hits, cell_n) in counts.items():
        if row_recipe == recipe and cell[0] == fold:
            hits, n = hits + cell_hits, n + cell_n
    return Fraction(hits, n) if n else None


def bottom_difference(ji: JudgeInputs, recipe: str, scope: Scope) -> Difference:
    """C-6: per-fold recall pooled over the bottom stratum (``k`` 1.3)."""
    counts = total_counts(count_cells(ji.data.matches, bottom_stratum_rows(scope)))
    return pooled_difference(counts, recipe, ji.folds)


def _c6(ji: JudgeInputs, recipe: str, scope: Scope) -> dict:
    """Point rule (>= 100 bottom positions) or bootstrap upper bound rule."""
    keep = bottom_stratum_rows(scope)
    n_bottom = sum(
        1
        for row in ji.data.matches
        if row.recipe == recipe and row.fold == ji.folds[0] and keep(row)
    )
    point = bottom_difference(ji, recipe, scope).mean
    if ji.bottom_rule == BOTTOM_POINT:
        result = _criterion(point, point is not None and point >= C6_MIN, ">= -3.0 %p")
        return {**result, "rule": BOTTOM_POINT, "n_injections_fold0": n_bottom}
    per_line = count_cells(ji.data.matches, keep)

    def pooled_delta(counts: CellCounts) -> Fraction | None:
        return pooled_difference(counts, recipe, ji.folds).mean

    bounds = bootstrap_bounds(per_line, ji.line_datasets, scope.datasets, pooled_delta)
    upper = bounds["upper"]
    # Upper-bound rule: block only on evidence of harm; no data is no evidence.
    passed = upper is None or upper >= C6_MIN
    return {
        "pass": passed,
        "rule": BOTTOM_UPPER,
        "point": _number(point),
        "upper": _number(upper),
        "n_undefined": bounds["n_undefined"],
        "n_injections_fold0": n_bottom,
        "threshold": "fail only if the 95 % upper bound < -3.0 %p",
    }


def guard_criteria(data: RunData, recipe: str, folds: Sequence[int]) -> dict:
    """C-7 (M4) and C-8 (M6) of one candidate."""
    return {
        "C-7": c7_check(
            m4_counts(data.boxes, recipe, folds),
            m4_counts(data.boxes, inputs.R0, folds),
        ),
        "C-8": c8_check(
            m6_counts(data.apex_hits, recipe), m6_counts(data.apex_hits, inputs.R0)
        ),
    }


def polarity_scopes(rule: str) -> dict[str, Scope]:
    """``mean``: one scope over +-1; ``each``: one scope per polarity."""
    if rule == "mean":
        return {"mean": MAIN_SCOPE}
    if rule == "each":
        return {f"p{p:+d}": with_polarity(MAIN_SCOPE, p) for p in POLARITIES}
    raise ValueError(f"polarity rule must be mean or each, got {rule!r}")


def evaluate_candidate(ji: JudgeInputs, recipe: str, rule: str = POLARITY_RULE) -> dict:
    """All criteria of one candidate; ``pass`` needs every C-1-C-8."""
    by_polarity = {
        name: injection_criteria(ji, recipe, scope)
        for name, scope in polarity_scopes(rule).items()
    }
    guards = guard_criteria(ji.data, recipe, ji.folds)
    passed = all(
        criterion["pass"]
        for criteria in by_polarity.values()
        for criterion in criteria.values()
    ) and all(criterion["pass"] for criterion in guards.values())
    main = difference(ji.main_counts, recipe, MAIN_SCOPE, ji.folds, pi=ji.pi)
    return {
        "pass": passed,
        "delta": _number(main.mean),
        "delta_exact": main.mean,
        "criteria_by_polarity": by_polarity,
        "guards": guards,
        "C-9": {"status": "report_only_not_measured_by_s5"},
    }


def select(candidates: Mapping[str, Mapping]) -> dict:
    """Largest Delta among passers; a < 0.5 %p gap is a tie -> smaller ``D_r``."""
    passers = {r: c["delta_exact"] for r, c in candidates.items() if c["pass"]}
    if not passers:
        return {
            "chosen": inputs.R0,
            "reason": "no candidate passed (r0 kept)",
            "tie_set": [],
        }
    best = max(passers.values())
    tie_set = sorted(r for r, value in passers.items() if best - value < TIE_MARGIN)
    chosen = min(tie_set, key=lambda r: (inputs.STAGE_A_DISTANCE[r], r))
    return {
        "chosen": chosen,
        "reason": "largest delta, tie -> smaller D_r",
        "tie_set": tie_set,
    }


def recall_by_group(ji: JudgeInputs, recipe: str, group: str) -> Fraction | None:
    """4TU recall of one model group (fold mean)."""
    scope = dataset_scope(inputs.DATASET_4TU, (group,))
    values = fold_recalls(ji.main_counts, recipe, scope, ji.folds)
    return None if values is None else _mean(values)


def a1_information(ji: JudgeInputs) -> dict:
    """INJ-A precondition: r0 4TU/A1 recall at 12 dB (amendment 9, F1').

    Polarity mean, unstratified, mean of the five folds, on the main rows of
    the A1 cell; below 30 % (or not computable) the A1/A0 row is not used.

    Population (verdict disclosure): ``main_counts``, i.e. the F2 main-metric
    rows - in-distribution main-cell rows on the ``"in"`` path, main-cell
    rows of every size on the ``"in_or_out"`` path - of 4TU/A1 at 12 dB.
    """
    scope = dataset_scope(inputs.DATASET_4TU, REPORT_ONLY_GROUPS[inputs.DATASET_4TU])
    values = fold_recalls(
        ji.main_counts, inputs.R0, scope, ji.folds, snr_steps=(INJ_A_SNR_DB,)
    )
    recall = None if values is None else _mean(values)
    available = recall is not None and recall >= INJ_A_MIN_A1_RECALL
    return {
        "available": available,
        "status": "A1 information available" if available else "A1 정보 없음",
        "r0_a1_recall_12db": _number(recall),
        "threshold": ">= 30 % (polarity mean, unstratified, 5-fold mean)",
    }


def inj_a_flags(ji: JudgeInputs, chosen: str, recipes: Sequence[str]) -> dict:
    """A1/A0 row of plan 4-3: any flag holds the adoption (ask INJ-A).

    The row is used only when ``a1_information`` is available (amendment 9);
    otherwise it is recorded as "A1 정보 없음" and never holds.
    """
    if chosen == inputs.R0:
        return {"applies": False, "hold": False}
    a1_info = a1_information(ji)
    if not a1_info["available"]:
        return {"applies": False, "hold": False, "a1_information": a1_info}
    deltas = {
        group: difference(
            ji.main_counts,
            chosen,
            dataset_scope(inputs.DATASET_4TU, (group,)),
            ji.folds,
        ).mean
        for group in ("A1", "A0")
    }
    leaders = {group: _leaders(ji, recipes, group) for group in ("A1", "A0")}
    flags = a1_a0_flags(deltas["A1"], deltas["A0"], leaders["A1"], leaders["A0"])
    return {
        "applies": True,
        "hold": any(flags.values()),
        "a1_information": a1_info,
        "flags": flags,
        "delta": {group: _number(value) for group, value in deltas.items()},
        "leaders": {
            group: None if value is None else sorted(value)
            for group, value in leaders.items()
        },
    }


def a1_a0_flags(
    delta_a1: Fraction | None,
    delta_a0: Fraction | None,
    leaders_a1: set[str] | None,
    leaders_a0: set[str] | None,
) -> dict[str, bool]:
    """Plan 4-3 A1/A0 conditions (i)-(iii); a value that cannot be computed flags.

    (i) the adopted Delta < 0 in A1 or A0; (ii) the leading recipes differ;
    (iii) ``|Delta_A1 - Delta_A0| >= 3.0 %p``.
    """
    if delta_a1 is None or delta_a0 is None:
        negative, gap = True, True
    else:
        negative = delta_a1 < 0 or delta_a0 < 0
        gap = abs(delta_a1 - delta_a0) >= INJ_A_GAP
    leaders_differ = (
        leaders_a1 is None or leaders_a0 is None or leaders_a1 != leaders_a0
    )
    return {
        "i_negative_in_a_group": negative,
        "ii_leaders_differ": leaders_differ,
        "iii_gap_ge_3pp": gap,
    }


def _leaders(ji: JudgeInputs, recipes: Sequence[str], group: str) -> set[str] | None:
    """Recipes with the highest 4TU recall of ``group`` (``None`` if empty cells)."""
    values: dict[str, Fraction] = {}
    for recipe in recipes:
        value = recall_by_group(ji, recipe, group)
        if value is None:
            return None
        values[recipe] = value
    best = max(values.values())
    return {recipe for recipe, value in values.items() if value == best}


def _out_of_distribution(row: MatchRow) -> bool:
    return out_stratum(row)


def _ricker_like(row: MatchRow) -> bool:
    return math.isclose(row.k, K_REPORT) and row.in_distribution and row.main_cell


def _old_zone_c(row: MatchRow) -> bool:
    return main_stratum(row) and row.in_horizontal_box_zone


def _any_class_hit(row: MatchRow) -> bool:
    return row.hit_any_class


def _report_delta(
    ji: JudgeInputs,
    recipe: str,
    keep: Callable[[MatchRow], bool],
    *,
    hit: HitSelector = base_hit,
    scope: Scope = MAIN_SCOPE,
    snr_steps: Sequence[float] = SNR_STEPS,
) -> dict | None:
    hits = {recipe: hit, inputs.R0: hit}
    counts = total_counts(count_cells(ji.data.matches, keep, hits))
    return _number(difference(counts, recipe, scope, ji.folds, snr_steps).mean)


def stratum_report(ji: JudgeInputs, recipe: str) -> dict:
    """Delta and C-3-style lower bound of the in and out strata (report)."""
    report = {}
    for name, keep in (("in", main_stratum), ("out", out_stratum)):
        per_line = count_cells(ji.data.matches, keep)
        value = difference(total_counts(per_line), recipe, MAIN_SCOPE, ji.folds).mean
        bounds = bootstrap_lower_bound(
            per_line, ji.line_datasets, recipe, MAIN_SCOPE, ji.folds
        )
        report[name] = {
            "delta": _number(value),
            "lower": _number(bounds["lower"]),
            "n_undefined": bounds["n_undefined"],
        }
    return report


def report_only(ji: JudgeInputs, recipe: str) -> dict:
    """Report items of plan 3-2 and [CVR] (never part of a criterion)."""
    return {
        "strata": stratum_report(ji, recipe),
        "delta_by_snr": {
            f"{snr:g}": _report_delta(ji, recipe, ji.keep, snr_steps=(snr,))
            for snr in SNR_STEPS
        },
        "delta_by_polarity": {
            f"{polarity:+d}": _report_delta(
                ji, recipe, ji.keep, scope=with_polarity(MAIN_SCOPE, polarity)
            )
            for polarity in POLARITIES
        },
        "delta_out_of_distribution": _report_delta(ji, recipe, _out_of_distribution),
        "delta_k08_12db": _report_delta(ji, recipe, _ricker_like, snr_steps=(12.0,)),
        "delta_class_agnostic": _report_delta(ji, recipe, ji.keep, hit=_any_class_hit),
        "delta_old_horizontal_zone_c_subset": _report_delta(ji, recipe, _old_zone_c),
        "report_only_cell_4tu_a1": report_only_cell(ji, recipe),
    }


A1_SCOPE = Scope(
    datasets=(inputs.DATASET_4TU,),
    groups=REPORT_ONLY_GROUPS,
    polarities=POLARITIES,
)


def report_only_cell(ji: JudgeInputs, recipe: str) -> dict:
    """4TU/A1 Delta on the main rows (amendment 9: out of every weighted cell)."""
    stratified = difference(ji.main_counts, recipe, A1_SCOPE, ji.folds, pi=ji.pi)
    plain = difference(ji.main_counts, recipe, A1_SCOPE, ji.folds)
    return {
        "delta": _number(stratified.mean),
        "delta_unstratified": _number(plain.mean),
    }


REPORT_CELLS = {
    "gz_ids/A0": dataset_scope(inputs.DATASET_IDS, ("A0",)),
    "4tu/A0": dataset_scope(inputs.DATASET_4TU, ("A0",)),
    "4tu/A1": A1_SCOPE,
}


def strong_signal_robustness(ji: JudgeInputs, recipes: Sequence[str]) -> dict:
    """Per recipe and cell: ``R(18 dB) - R(12 dB)`` (amendment 9, report only).

    Polarity mean, unstratified, mean of the five folds, on the main rows.
    C-1 keeps all four SNR steps; this column never enters a criterion.
    """
    low_db, high_db = STRONG_SIGNAL_SNRS
    report: dict[str, dict] = {}
    for recipe in recipes:
        per_cell: dict[str, dict] = {}
        for name, scope in REPORT_CELLS.items():
            low = fold_recalls(ji.main_counts, recipe, scope, ji.folds, (low_db,))
            high = fold_recalls(ji.main_counts, recipe, scope, ji.folds, (high_db,))
            if low is None or high is None:
                per_cell[name] = {"reason": "not_evaluable"}
                continue
            per_cell[name] = {
                "recall_12db": _number(_mean(low)),
                "recall_18db": _number(_mean(high)),
                "difference_18_minus_12": _number(_mean(high) - _mean(low)),
            }
        report[recipe] = per_cell
    return report


def sample_counts(ji: JudgeInputs) -> dict:
    """Injections per SNR step (main stratum, one recipe and fold, pairs as one)."""
    reference = [
        row
        for row in ji.data.matches
        if row.recipe == inputs.R0 and row.fold == ji.folds[0] and ji.keep(row)
    ]
    per_snr = {
        f"{s:g}": sum(r.snr_db == s for r in reference) // len(POLARITIES)
        for s in SNR_STEPS
    }
    in_distribution = [row for row in reference if row.in_distribution]
    return {
        "primary_stratum": ji.primary_stratum,
        "bottom_rule": ji.bottom_rule,
        "in_distribution_positions_fold0": len(in_distribution)
        // (len(POLARITIES) * len(SNR_STEPS)),
        "positions_per_snr_main": per_snr,
        "contributing_lines": len({r.line_id for r in reference}),
        "bottom_stratum_positions": sum(
            r.bottom_stratum
            for r in ji.data.matches
            if r.recipe == inputs.R0
            and r.fold == ji.folds[0]
            and math.isclose(r.k, K_MAIN)
        )
        // (len(POLARITIES) * len(SNR_STEPS)),
        "main_cell_excluded_fold0": sum(
            not r.main_cell
            for r in ji.data.matches
            if r.recipe == inputs.R0
            and r.fold == ji.folds[0]
            and math.isclose(r.k, K_MAIN)
        )
        // (len(POLARITIES) * len(SNR_STEPS)),
    }


# --------------------------------------------------------------- pre-check


@dataclass(frozen=True)
class PrecheckCell:
    """One weighted cell of the pre-check and its C-1 weight."""

    dataset: str
    group: str
    weight: Fraction


PRECHECK_CELLS = (
    PrecheckCell(inputs.DATASET_IDS, "A0", Fraction(1, 2)),
    PrecheckCell(inputs.DATASET_4TU, "A0", Fraction(1, 2)),
)
"""Amendment 9 cell weights (the cv ad-hoc script used 1/2, 1/4, 1/4)."""


def precheck_row(row: MatchRow) -> bool:
    """r0, fold 0, ``k`` 1.3, main cell, in distribution, weighted cell."""
    return (
        row.recipe == inputs.R0
        and row.fold == PRECHECK_FOLD
        and main_stratum(row)
        and row.group in GROUPS_BY_DATASET.get(row.dataset, ())
    )


def _is_monotone(values: Sequence[Fraction]) -> bool:
    """No later step more than ``PRECHECK_TOLERANCE`` below any earlier one."""
    return all(
        later >= earlier - PRECHECK_TOLERANCE
        for i, earlier in enumerate(values)
        for later in values[i + 1 :]
    )


def _precheck_recalls(rows: Sequence[MatchRow]) -> tuple[dict, dict]:
    """Pooled and cell-weighted (polarity-mean) r0 recall per SNR step."""
    tallies: dict[tuple[float, str, str, int], list[int]] = {}
    for row in rows:
        key = (row.snr_db, row.dataset, row.group, row.polarity)
        tally = tallies.setdefault(key, [0, 0])
        tally[0] += int(row.hit)
        tally[1] += 1
    pooled: dict[float, Fraction | None] = {}
    weighted: dict[float, Fraction | None] = {}
    for snr in SNR_STEPS:
        hits = sum(v[0] for k, v in tallies.items() if k[0] == snr)
        n = sum(v[1] for k, v in tallies.items() if k[0] == snr)
        pooled[snr] = Fraction(hits, n) if n else None
        weighted[snr] = _weighted_precheck_recall(tallies, snr)
    return pooled, weighted


def _weighted_precheck_recall(
    tallies: Mapping[tuple[float, str, str, int], Sequence[int]], snr: float
) -> Fraction | None:
    total = Fraction(0)
    for cell in PRECHECK_CELLS:
        recalls = []
        for polarity in POLARITIES:
            hits, n = tallies.get((snr, cell.dataset, cell.group, polarity), (0, 0))
            if n == 0:
                return None
            recalls.append(Fraction(hits, n))
        total += cell.weight * _mean(recalls)
    return total


def _precheck_conditions(recall: Mapping[float, Fraction | None]) -> dict[str, bool]:
    ladder = [recall[snr] for snr in PRECHECK_MONOTONE_SNRS]
    at_12 = recall[PRECHECK_SNR_DB]
    return {
        "recall_12db_ge_50pct": at_12 is not None and at_12 >= PRECHECK_MIN_RECALL,
        "monotone_0_to_12db": all(v is not None for v in ladder)
        and _is_monotone([v for v in ladder if v is not None]),
    }


def stage_c_precheck(matches: Iterable[MatchRow]) -> dict:
    """Plan 3-2 pre-check after amendment 9 (r0 only, not a judgement).

    Rows: r0, fold 0, ``k`` 1.3, main cells (IDS/A0, 4TU/A0), in
    distribution. Both the pooled recall and the cell-weighted one (weights
    1/2, 1/2, polarity mean, unstratified) must reach 50 % at 12 dB and be
    monotone from 0 to 12 dB within 1 %p, as in
    ``analysis/detection/m32_stageC_precheck.py`` with the amendment-9 cells
    and range. 18 dB is reported, never tested. An empty cell fails.
    """
    rows = [row for row in matches if precheck_row(row)]
    pooled, weighted = _precheck_recalls(rows)
    conditions = {
        f"{name}_{form}": passed
        for form, recall in (("pooled", pooled), ("weighted", weighted))
        for name, passed in _precheck_conditions(recall).items()
    }
    return {
        "pass": all(conditions.values()),
        "conditions": conditions,
        "n_rows_per_snr": {
            f"{snr:g}": sum(row.snr_db == snr for row in rows) for snr in SNR_STEPS
        },
        "recall_pooled": {f"{s:g}": _number(v) for s, v in pooled.items()},
        "recall_cell_weighted": {f"{s:g}": _number(v) for s, v in weighted.items()},
        "cell_weights": {
            f"{cell.dataset}/{cell.group}": _number(cell.weight)
            for cell in PRECHECK_CELLS
        },
        "monotone_snrs": list(PRECHECK_MONOTONE_SNRS),
        "tolerance": _number(PRECHECK_TOLERANCE),
    }


@dataclass(frozen=True)
class PrecheckRun:
    """The r0 fold-0 rows of the P5b pre-check subset and the hashes read."""

    matches: list[MatchRow]
    sha256: dict[str, str]


def _assert_f2_inputs(run_root: Path, f2_record: Path, summary: Mapping) -> dict:
    """F2 record, injection table and exclusions as fixed at F2 (sha256).

    The P5b summary must have run on the same three files.
    """
    if not f2_record.is_file():
        raise inputs.InputError(f"F2 record missing: {f2_record}")
    record = json.loads(f2_record.read_text(encoding="utf-8"))
    plan_dir = run_root / "plan"
    try:
        hashes = {
            "f2_record": inputs.sha256_file(f2_record),
            "injection_table": inputs.checked_sha256(
                plan_dir / "injection_table.csv",
                record["injection_table_sha256"],
                "injection table",
            ),
            "exclusions": inputs.checked_sha256(
                plan_dir / "exclusions.csv", record["exclusions_sha256"], "exclusions"
            ),
        }
    except KeyError as error:
        raise inputs.InputError(f"F2 record lacks {error}") from error
    used = summary.get("run_identity", {}).get("inputs_sha256", {})
    differing = sorted(key for key, value in hashes.items() if used.get(key) != value)
    if differing:
        raise inputs.InputError(f"P5b ran on other F2 inputs: {differing}")
    return hashes


def _precheck_rows(path: Path) -> list[MatchRow]:
    """Rows of the pre-check csv; any non-r0 or non-fold-0 row is refused.

    Each row is checked before it is parsed, so no comparison value of another
    recipe is ever held in memory (blind rule).
    """
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("recipe") != inputs.R0 or raw.get("fold") != str(PRECHECK_FOLD):
                raise inputs.InputError(f"{path}: a row outside r0 fold 0")
            try:
                rows.append(match_row(raw))
            except (KeyError, ValueError) as error:
                raise inputs.InputError(f"malformed inj_matches: {error!r}") from error
    if not rows:
        raise inputs.InputError(f"{path}: no injection rows")
    return rows


def load_precheck_run(run_root: Path, f2_record: Path) -> PrecheckRun:
    """Read ``injected_precheck/`` (P5b with ``--recipes r0 --folds 0``).

    Raises:
        inputs.InputError: Unfinished or other subset run, any file or row
            outside r0 fold 0, or F2 inputs that differ from the record.
    """
    stage_dir = run_root / PRECHECK_STAGE_DIR
    summary_path = stage_dir / "summary.json"
    if not summary_path.is_file():
        raise inputs.InputError(f"missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    is_subset = (
        summary.get("status") == "ok"
        and summary.get("precheck_subset") is True
        and summary.get("recipes") == [inputs.R0]
        and summary.get("folds") == [PRECHECK_FOLD]
    )
    if not is_subset:
        raise inputs.InputError(f"{summary_path}: not a finished r0 fold-0 subset")
    expected = stage_dir / inputs.R0 / f"fold{PRECHECK_FOLD}" / "inj_matches.csv"
    found = sorted(stage_dir.glob("*/fold*/inj_matches.csv"))
    if found != [expected]:
        raise inputs.InputError(f"pre-check folder holds other results: {found}")
    hashes = _assert_f2_inputs(run_root, f2_record, summary)
    hashes[f"{PRECHECK_STAGE_DIR}/summary.json"] = inputs.sha256_file(summary_path)
    hashes[expected.relative_to(run_root).as_posix()] = inputs.sha256_file(expected)
    return PrecheckRun(matches=_precheck_rows(expected), sha256=hashes)


def _json_round_trip(document: Mapping) -> dict:
    return json.loads(json.dumps(document))


def check_recorded_precheck(run_root: Path, precheck: Mapping) -> str:
    """The full run's r0 fold-0 pre-check must equal ``precheck.json``.

    Returns:
        The sha256 of ``precheck.json``.

    Raises:
        inputs.InputError: The file is missing (the pre-check must come first,
            plan 4-4) or its values differ from the full run's.
    """
    path = run_root / PRECHECK_FILE
    if not path.is_file():
        raise inputs.InputError(f"missing {path}: run --precheck-only first")
    recorded = json.loads(path.read_text(encoding="utf-8")).get("precheck")
    if recorded != _json_round_trip(precheck):
        raise inputs.InputError(
            "full-run r0 fold-0 pre-check differs from precheck.json"
        )
    return inputs.sha256_file(path)


def judge(data: RunData, rule: str = POLARITY_RULE) -> dict:
    """The verdict document (no file I/O).

    Callers run :func:`stage_c_precheck` first; ``main`` stops on a failed
    pre-check before any recipe comparison is computed.
    """
    ji = prepare(data)
    gate = identity_gate(data.gate_record)
    batch_report = d3_comparison(m6_counts(data.apex_hits, inputs.R0), data.batch_path)
    candidates = [r for r in inputs.STAGE_C_RECIPES if r != inputs.R0]
    results = {recipe: evaluate_candidate(ji, recipe, rule) for recipe in candidates}
    selection = (
        select(results)
        if gate["pass"]
        else {"chosen": None, "reason": "identity gate failed"}
    )
    chosen = selection["chosen"]
    inj_a = (
        inj_a_flags(ji, chosen, inputs.STAGE_C_RECIPES)
        if chosen
        else {"applies": False, "hold": False}
    )
    decision = _decision(gate, chosen, inj_a)
    b0_direction = b0_direction_flags(results)
    for result in results.values():
        result.pop("delta_exact")
    return {
        "schema": SCHEMA,
        "decision": decision,
        "identity_gate": gate,
        "c8_batch_path_vs_d3": batch_report,
        "polarity_rule": rule,
        "candidates": results,
        "selection": selection,
        "inj_a": inj_a,
        "b0_direction": b0_direction,
        "report_only": {recipe: report_only(ji, recipe) for recipe in candidates},
        "strong_signal_robustness": strong_signal_robustness(
            ji, inputs.STAGE_C_RECIPES
        ),
        "samples": sample_counts(ji),
    }


def b0_direction_flags(results: Mapping[str, Mapping]) -> dict:
    """Plan 4-8: a B1-B3 recipe with Delta > 0 runs against the stage-B B0 mimic.

    The flag goes into the judgement record; it never changes the decision.
    """
    flags = {}
    for recipe, result in results.items():
        if recipe not in inputs.BACKGROUND_REMOVAL_WEAKENED:
            continue
        delta = result["delta_exact"]
        flags[recipe] = {
            "delta_positive": delta is not None and delta > 0,
            "note": "weaker background removal than B0; stage B: B0 mimic "
            "did not lower AIHub hyperbola recall",
        }
    return flags


def _decision(gate: Mapping, chosen: str | None, inj_a: Mapping) -> str:
    if not gate["pass"]:
        return "invalid_identity_gate"
    if chosen == inputs.R0:
        return "keep_r0"
    if inj_a.get("hold"):
        return f"hold_{chosen}_ask_inj_a"
    return f"adopt_{chosen}"


# ===================================================================== CLI


def blind_cell_summary(verdict: Mapping) -> dict:
    """Stdout-safe G-in facts: valid cell count and not-evaluable flag only."""
    for candidate in verdict["candidates"].values():
        for criteria in candidate["criteria_by_polarity"].values():
            guard = criteria.get("G-in")
            if guard is not None:
                n_valid = sum(cell["valid"] for cell in guard["cells"].values())
                return {
                    "g_in_cells": n_valid,
                    "g_in_not_evaluable": guard.get("reason") == "not_evaluable",
                }
    return {"g_in_cells": None, "g_in_not_evaluable": None}


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default <run-root>/verdict.json (precheck.json with --precheck-only)",
    )
    parser.add_argument(
        "--precheck-only",
        action="store_true",
        help="read injected_precheck/ (P5b r0 fold 0) and write precheck.json",
    )
    parser.add_argument("--f2-record", type=Path, default=DEFAULT_F2_RECORD)
    return parser.parse_args(argv)


def run_precheck_only(args: argparse.Namespace, out: Path) -> str:
    """Pre-check of the r0 fold-0 subset; returns ``ok`` or ``precheck_failed``."""
    run = load_precheck_run(args.run_root, args.f2_record)
    precheck = stage_c_precheck(run.matches)
    document = {
        "schema": PRECHECK_SCHEMA,
        "precheck": precheck,
        "inputs_sha256": run.sha256,
        "run_identity": common.new_run_identity(),
    }
    common.write_json(out, document)
    return "ok" if precheck["pass"] else "precheck_failed"


def run_verdict(args: argparse.Namespace, out: Path) -> tuple[str, dict]:
    """Full judgement; returns the status and the stdout-safe G-in facts."""
    data = load_run(args.run_root)
    precheck = stage_c_precheck(data.matches)
    precheck_sha = check_recorded_precheck(args.run_root, precheck)
    if precheck["pass"]:
        verdict = {**judge(data), "precheck": precheck}
        status = "ok"
    else:
        # Plan 3-2 / 4-9 (9-7): stop before any recipe comparison exists.
        verdict = {
            "schema": SCHEMA,
            "decision": "precheck_failed",
            "precheck": precheck,
        }
        status = "precheck_failed"
    verdict["inputs_sha256"] = {**data.sha256, PRECHECK_FILE: precheck_sha}
    verdict["run_identity"] = common.new_run_identity()
    verdict["settings"] = {"interpretations": __doc__.split("Interpretations")[1]}
    common.write_json(out, verdict)
    cells = blind_cell_summary(verdict) if status == "ok" else dict(NO_CELLS)
    return status, cells


NO_CELLS = {"g_in_cells": None, "g_in_not_evaluable": None}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry; returns the exit code (see module docstring)."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    started = time.time()
    args = parse_args(argv)
    default_name = PRECHECK_FILE if args.precheck_only else "verdict.json"
    out = args.out or args.run_root / default_name
    cells: dict = dict(NO_CELLS)
    try:
        if args.precheck_only:
            status = run_precheck_only(args, out)
        else:
            status, cells = run_verdict(args, out)
    except inputs.InputError as error:
        logger.error("input error: %s", error)
        status = "input_error"
    except Exception:
        logger.exception("internal error")
        status = "error"
    if args.precheck_only:
        line: dict = {"status": status, "precheck": str(out)}
    else:
        line = {"status": status, "verdict": str(out), **cells}
    line["elapsed_s"] = round(time.time() - started, 1)
    print(json.dumps(line, ensure_ascii=False), flush=True)
    return EXIT_CODES[status]


if __name__ == "__main__":
    sys.exit(main())
