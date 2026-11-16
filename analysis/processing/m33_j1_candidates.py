"""M3-3 J1-1: one apex candidate per detector box (J1 pre-registration §3).

Spec: ``docs/processing/m33_j1_preregistration.md`` v0.3.4 §3 (box -> apex
candidate rule) and §5 J1-1. The R2 runner wrote the boxes
(``runs/eval/m33/r2/fold<k>/j1_boxes_{post,pre}.csv``, ``m33_runner_spec.md``
§5): only ``pipe_hyperbola`` boxes with conf > 0.25 and an L0 box, PP already
applied. Every row is used; nothing is filtered again here.

Per line, once (the depth path's own values):

* section: fitting branch steps (1)-(4), background removed (median 301);
  envelope ``e``: ``|hilbert|`` along time with 2N zero padding;
* zero: ``prepare_depth_inputs(radargram, DepthConfig()).zero`` (the detected
  zero of ``scripts/eval_depth_4tu.py``), ``t_z`` in ns and ``k_z`` in L0
  samples;
* ``P_line``: ``make_detection_image(radargram).period_samples`` (r0).

Per box (L0 ``i0, i1, k0, k1``):

1. search window rows ``[k_lo, k0 + 2.5 P]`` with
   ``k_lo = max(k0 - 0.5 P, k_z + 0.5 P)`` (``k0 - 0.5 P`` without a zero),
   clipped to the record; traces: the centre half
   ``[i0 + w // 4, i1 - w // 4]`` (``w = i1 - i0 + 1``; all traces when
   ``w < 8``). A window that ends above ``k_z + 0.5 P`` gives no candidate
   (``window_above_zero``).
2. apex: the largest ``e`` in the window; its row is refined by a three-point
   parabola inside the window column.
3. ``t_ns = k * dt``, ``tau_ns = t_ns - t_z`` (blank when the zero failed).
4. on one line, candidates within +-3 traces and +-0.5 ``P_line`` keep only
   the higher box score (equal scores: the smaller ``(trace, sample)``),
   greedy in descending score order.
5. ``source`` = ``j1_fold<k>_<pp>``; ``note`` = box score, ``pp_action``,
   ``tile_truncated``.

The ground-truth files are never read here (allowed inputs: the box csv, the
line list, the 4TU radargrams and ``src``). Deterministic, no random numbers.
The judge (``m33_j1_judge.py``) calls :func:`build_candidate_set`; this module
has no command line of its own, so it can never write a judgment-prefix file
by itself.
"""

import csv
import math
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert

from gpr_detection.depth.config import DepthConfig
from gpr_detection.depth.estimate import prepare_depth_inputs
from gpr_detection.io.classes import DetectionClass
from gpr_detection.io.fourtu import T0Choice, load_fourtu_radargram, site_of_activity
from gpr_detection.io.paths import load_dataset_paths
from gpr_detection.io.radargram import Radargram
from gpr_detection.preprocess.signals import parabolic_peak
from gpr_detection.preprocess.to_image import make_detection_image

CANDIDATE_COLUMNS = (
    "activity_id",
    "site",
    "path",
    "trace",
    "sample",
    "t_ns",
    "tau_ns",
    "snr",
    "n_left",
    "n_right",
    "v_straight_qc",
    "source",
    "note",
)
"""Columns of ``docs/processing/fourtu_apex_candidates_v1.csv`` (J1 §1-1 1)."""

BOX_COLUMNS = (
    "line_id",
    "activity_id",
    "path",
    "fold",
    "class_id",
    "score",
    "i0",
    "i1",
    "k0",
    "k1",
    "x0_line_px",
    "y1_px",
    "y2_px",
    "pp_action",
    "tile_truncated",
    "n_tiles_seen",
)
"""R2 ``J1_COLUMNS`` (``m33_r2_eval.py``, [RS] §5)."""

WINDOW_UP_PERIODS = 0.5
"""Rows searched above the box top ``k0`` (§3 step 1)."""
WINDOW_DOWN_PERIODS = 2.5
"""Rows searched below the box top ``k0``."""
ZERO_MARGIN_PERIODS = 0.5
"""The window starts this far below the zero ``k_z`` (the PP-d margin)."""
CENTRE_TRACE_DIVISOR = 4
"""The window drops ``w // 4`` traces on each side of the box."""
MIN_WIDTH_FOR_CENTRE = 8
"""Boxes narrower than this many traces use all their traces."""
DEDUPE_TRACES = 3
DEDUPE_PERIODS = 0.5
"""Two candidates within +-3 traces and +-0.5 ``P_line`` are one apex (M3-2)."""

SKIP_WINDOW_ABOVE_ZERO = "window_above_zero"
"""The whole window lies above ``k_z + 0.5 P`` (§3 step 1, v0.3)."""
SKIP_WINDOW_EMPTY = "window_empty"
"""No record row is left after clipping (outside the §3 cases; reported)."""


@dataclass(frozen=True)
class DetectorBox:
    """One row of a ``j1_boxes_*.csv`` file (L0 grid box)."""

    line_id: str
    activity_id: str
    path: str
    fold: int
    class_id: int
    score: float
    score_text: str
    i0: int
    i1: int
    k0: int
    k1: int
    pp_action: str
    tile_truncated: str

    @property
    def line(self) -> tuple[str, str]:
        return self.activity_id, self.path


@dataclass(frozen=True)
class LinePrep:
    """Per-line values the candidate rule reads (§3 측선 준비).

    Attributes:
        envelope: ``(samples, traces)`` envelope of the background-removed
            fitting-branch section.
        dt_ns: Sample interval.
        t_zero_ns: Detected zero ``t_z``; ``None`` when the zero failed.
        k_zero: The same zero in L0 samples; ``None`` when it failed.
        period_samples: ``P_line`` of the r0 detection image.
    """

    envelope: np.ndarray
    dt_ns: float
    t_zero_ns: float | None
    k_zero: float | None
    period_samples: float

    @property
    def n_samples(self) -> int:
        return int(self.envelope.shape[0])

    @property
    def n_traces(self) -> int:
        return int(self.envelope.shape[1])


@dataclass(frozen=True)
class SearchWindow:
    """Inclusive L0 rows and traces of one box's apex search."""

    row_first: int
    row_last: int
    trace_first: int
    trace_last: int


@dataclass(frozen=True)
class ApexCandidate:
    """The apex found for one box (before duplicates are removed)."""

    box: DetectorBox
    trace: int
    sample: float


@dataclass(frozen=True)
class SkippedBox:
    """A box that gives no candidate, and why."""

    box: DetectorBox
    reason: str


@dataclass(frozen=True)
class CandidateSet:
    """Everything one candidate build produces.

    Attributes:
        table: Candidate rows (``CANDIDATE_COLUMNS``), sorted by line, trace
            and sample.
        n_boxes: Rows of the box file.
        n_before_dedupe: Candidates before step 4.
        skipped: Boxes without a candidate, with the reason.
        n_zero_failed: Kept candidates whose ``tau_ns`` is blank.
    """

    table: pd.DataFrame
    n_boxes: int
    n_before_dedupe: int
    skipped: tuple[SkippedBox, ...]
    n_zero_failed: int

    @property
    def n_candidates(self) -> int:
        return len(self.table)

    def skipped_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for skipped in self.skipped:
            counts[skipped.reason] = counts.get(skipped.reason, 0) + 1
        return counts


# ------------------------------------------------------------------ boxes


def _box_from_row(row: dict[str, str]) -> DetectorBox:
    return DetectorBox(
        line_id=row["line_id"],
        activity_id=row["activity_id"],
        path=row["path"],
        fold=int(row["fold"]),
        class_id=int(row["class_id"]),
        score=float(row["score"]),
        score_text=row["score"],
        i0=int(row["i0"]),
        i1=int(row["i1"]),
        k0=int(row["k0"]),
        k1=int(row["k1"]),
        pp_action=row["pp_action"],
        tile_truncated=row["tile_truncated"],
    )


def read_box_rows(path: Path) -> list[dict[str, str]]:
    """Rows of a ``j1_boxes_*.csv`` file as text.

    Raises:
        ValueError: When the header is not the R2 J1 header.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != BOX_COLUMNS:
            raise ValueError(f"{path.name}: header {reader.fieldnames} != J1 columns")
        return list(reader)


def read_boxes(path: Path) -> list[DetectorBox]:
    """The boxes of a ``j1_boxes_*.csv`` file (empty L0 cells raise)."""
    return [_box_from_row(row) for row in read_box_rows(path)]


# ------------------------------------------------------------------- line


def line_envelope(section: np.ndarray) -> np.ndarray:
    """``|hilbert|`` along time (axis 0) with 2N zero padding (no wrap-around)."""
    n_samples = section.shape[0]
    values = np.asarray(section, dtype=np.float64)
    return np.abs(hilbert(values, N=2 * n_samples, axis=0)[:n_samples])


def prepare_line(radargram: Radargram) -> LinePrep:
    """Envelope, detected zero and ``P_line`` of one line (§3 측선 준비)."""
    if radargram.dt_ns is None:
        raise ValueError(f"{radargram.line_id}: J1 needs a time scale")
    inputs = prepare_depth_inputs(radargram, DepthConfig())
    period_samples = make_detection_image(radargram).period_samples
    return LinePrep(
        envelope=line_envelope(inputs.branch.background_removed),
        dt_ns=radargram.dt_ns,
        t_zero_ns=inputs.zero.t_zero_ns,
        k_zero=inputs.zero.k_zero,
        period_samples=period_samples,
    )


def load_fourtu_line(activity_id: str, path: str) -> Radargram:
    """The 4TU line the depth path loads (``T0Choice.BOTH``, read-only)."""
    root = load_dataset_paths().fourtu
    return load_fourtu_radargram(root, activity_id, path, T0Choice.BOTH)


# ------------------------------------------------------------- one box


def _trace_range(box: DetectorBox, n_traces: int) -> tuple[int, int]:
    width = box.i1 - box.i0 + 1
    margin = 0 if width < MIN_WIDTH_FOR_CENTRE else width // CENTRE_TRACE_DIVISOR
    return max(box.i0 + margin, 0), min(box.i1 - margin, n_traces - 1)


def search_window(box: DetectorBox, prep: LinePrep) -> SearchWindow | str:
    """Step 1: the window of ``box``, or the reason it is empty."""
    period = prep.period_samples
    row_low = box.k0 - WINDOW_UP_PERIODS * period
    row_high = box.k0 + WINDOW_DOWN_PERIODS * period
    if prep.k_zero is not None:
        zero_limit = prep.k_zero + ZERO_MARGIN_PERIODS * period
        if row_high <= zero_limit:
            return SKIP_WINDOW_ABOVE_ZERO
        row_low = max(row_low, zero_limit)
    row_first = math.ceil(max(row_low, 0.0))
    row_last = math.floor(min(row_high, prep.n_samples - 1))
    trace_first, trace_last = _trace_range(box, prep.n_traces)
    if row_first > row_last or trace_first > trace_last:
        return SKIP_WINDOW_EMPTY
    return SearchWindow(row_first, row_last, trace_first, trace_last)


def apex_in_window(prep: LinePrep, window: SearchWindow) -> tuple[int, float]:
    """Step 2: ``(trace, fractional sample)`` of the envelope maximum.

    Ties go to the earliest row, then the smallest trace (row-major argmax).
    The parabola uses the window column only, so the refined row never leaves
    the window.
    """
    rows = slice(window.row_first, window.row_last + 1)
    traces = slice(window.trace_first, window.trace_last + 1)
    patch = prep.envelope[rows, traces]
    row, column = np.unravel_index(int(np.argmax(patch)), patch.shape)
    refined_row, _ = parabolic_peak(patch[:, column], int(row))
    return window.trace_first + int(column), window.row_first + refined_row


def candidate_for_box(box: DetectorBox, prep: LinePrep) -> ApexCandidate | SkippedBox:
    """Steps 1-2 for one box."""
    window = search_window(box, prep)
    if isinstance(window, str):
        return SkippedBox(box, window)
    trace, sample = apex_in_window(prep, window)
    return ApexCandidate(box, trace, sample)


def deduplicate(
    candidates: Sequence[ApexCandidate], period_samples: float
) -> list[ApexCandidate]:
    """Step 4 on one line: greedy by descending score, then (trace, sample)."""
    sample_radius = DEDUPE_PERIODS * period_samples
    ordered = sorted(candidates, key=lambda c: (-c.box.score, c.trace, c.sample))
    kept: list[ApexCandidate] = []
    for candidate in ordered:
        is_duplicate = any(
            abs(candidate.trace - other.trace) <= DEDUPE_TRACES
            and abs(candidate.sample - other.sample) <= sample_radius
            for other in kept
        )
        if not is_duplicate:
            kept.append(candidate)
    return kept


# ------------------------------------------------------------------ rows


def source_label(fold: int, pp: str) -> str:
    """``source`` column of step 5, e.g. ``j1_fold0_post``."""
    return f"j1_fold{fold}_{pp}"


def _note(box: DetectorBox) -> str:
    return (
        f"score={box.score_text};pp_action={box.pp_action};"
        f"tile_truncated={box.tile_truncated}"
    )


def candidate_row(candidate: ApexCandidate, prep: LinePrep, source: str) -> dict:
    """Step 3 and 5: one row in ``CANDIDATE_COLUMNS``."""
    t_ns = candidate.sample * prep.dt_ns
    tau_ns = math.nan if prep.t_zero_ns is None else t_ns - prep.t_zero_ns
    box = candidate.box
    return {
        "activity_id": box.activity_id,
        "site": site_of_activity(box.activity_id),
        "path": box.path,
        "trace": candidate.trace,
        "sample": candidate.sample,
        "t_ns": t_ns,
        "tau_ns": tau_ns,
        "snr": None,
        "n_left": None,
        "n_right": None,
        "v_straight_qc": None,
        "source": source,
        "note": _note(box),
    }


# ------------------------------------------------------------------ build


@dataclass(frozen=True)
class _LineResult:
    rows: list[dict]
    n_before_dedupe: int
    skipped: list[SkippedBox]


def _line_candidates(
    boxes: Sequence[DetectorBox], prep: LinePrep, source: str
) -> _LineResult:
    found: list[ApexCandidate] = []
    skipped: list[SkippedBox] = []
    for box in boxes:
        outcome = candidate_for_box(box, prep)
        if isinstance(outcome, SkippedBox):
            skipped.append(outcome)
        else:
            found.append(outcome)
    kept = deduplicate(found, prep.period_samples)
    rows = [candidate_row(candidate, prep, source) for candidate in kept]
    return _LineResult(rows, len(found), skipped)


def _boxes_by_line(
    boxes: Iterable[DetectorBox], lines: Collection[tuple[str, str]]
) -> dict[tuple[str, str], list[DetectorBox]]:
    grouped: dict[tuple[str, str], list[DetectorBox]] = {}
    for box in boxes:
        if box.line not in lines:
            raise ValueError(f"box on {box.line_id}, which is not in the line list")
        if box.class_id != DetectionClass.PIPE_HYPERBOLA:
            raise ValueError(f"box of class {box.class_id} on {box.line_id}")
        grouped.setdefault(box.line, []).append(box)
    return grouped


def _sorted_table(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=list(CANDIDATE_COLUMNS))
    if table.empty:
        return table
    order_keys = ["activity_id", "path", "trace", "sample"]
    return table.sort_values(order_keys, kind="stable").reset_index(drop=True)


def build_candidate_set(
    boxes_csv: Path,
    fold: int,
    pp: str,
    lines: Collection[tuple[str, str]],
    *,
    load_line: Callable[[str, str], Radargram] = load_fourtu_line,
) -> CandidateSet:
    """J1-1 with its counts: every box of ``boxes_csv`` -> candidates (§3).

    Args:
        boxes_csv: R2 ``j1_boxes_{post,pre}.csv`` of one fold.
        fold: The fold of the file (``source`` label).
        pp: ``post`` or ``pre`` (``source`` label).
        lines: The ``(activity_id, path)`` lines a box may lie on (754).
        load_line: ``(activity_id, path) -> Radargram`` (default: 4TU).

    Raises:
        ValueError: A box outside ``lines``, of another class, or a file with
            another header.
    """
    boxes = read_boxes(boxes_csv)
    source = source_label(fold, pp)
    rows: list[dict] = []
    skipped: list[SkippedBox] = []
    n_before_dedupe = 0
    for (activity_id, path), line_boxes in sorted(_boxes_by_line(boxes, lines).items()):
        prep = prepare_line(load_line(activity_id, path))
        result = _line_candidates(line_boxes, prep, source)
        rows += result.rows
        skipped += result.skipped
        n_before_dedupe += result.n_before_dedupe
    table = _sorted_table(rows)
    return CandidateSet(
        table=table,
        n_boxes=len(boxes),
        n_before_dedupe=n_before_dedupe,
        skipped=tuple(skipped),
        n_zero_failed=int(table["tau_ns"].isna().sum()) if len(table) else 0,
    )


def build_j1_candidates(
    boxes_csv: Path, fold: int, pp: str, lines: Collection[tuple[str, str]]
) -> pd.DataFrame:
    """J1-1 (§5): the candidate table of one box file.

    See :func:`build_candidate_set` for the counts that come with it.
    """
    return build_candidate_set(boxes_csv, fold, pp, lines).table


def write_candidates_csv(table: pd.DataFrame, path: Path) -> None:
    """Write the candidate table (LF, UTF-8, full float precision, NaN blank)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        path, index=False, columns=list(CANDIDATE_COLUMNS), lineterminator="\n"
    )
