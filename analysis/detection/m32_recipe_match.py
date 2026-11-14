"""M3-2 S5: matching and box rules of the recipe runner (pure functions).

Sources: plan ``docs/processing/m32_image_quality_plan.md`` v0.2.3 sections
3-2 (match rule, one-to-one), 3-3 (M4a-M4d), 4-3 (C-5 operating point, C-8
M6) and ``docs/experiments/m32_cv_review.md`` section 2-4 and appendix B-4.

Everything here works on L0 ``GridBox`` values or on line L1 boxes with the
line ``ImageGridMap`` content range, never on pixel values, so the runner
(``m32_recipe_eval.py``) and the tests share one definition.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from gpr_detection.io.classes import DetectionClass
from gpr_detection.io.radargram import GridBox

MATCH_TRACE_MARGIN = 3
MATCH_SAMPLE_PERIODS = 0.5
"""Apex match range: box grown by +-3 traces and +-0.5 ``P_line`` (plan 3-2)."""

EDGE_PX = 2.0
"""M4b straddle margin, the ``MergeConfig.edge_px`` value ([CVR] 2-4)."""
M4C_WINDOW_PERIODS = 3.0
M4C_WINDOW_PX = 18.0
"""M4c fixed window: 3 P at ``P*`` 6 px for every recipe ([CVR] 2-4)."""
M4C_D3_WINDOW_PX = 8.0
"""The D3 8 px band window, reported next to M4c for continuity."""
M4D_ZERO_MARGIN_PERIODS = 0.5

CONF_BASE = 0.25
CONF_GRID_STEP = 0.01
CONF_GRID_MAX = 0.95
"""C-5 operating-point grid: 0.25, 0.26, ..., 0.95 (strict ``score > conf``)."""

FLIP_IOU = 0.5


class M4Category(StrEnum):
    """Definition false positives of plan 3-3 (one primary category per box)."""

    A = "M4a"
    B = "M4b"
    D = "M4d"


@dataclass(frozen=True)
class TruthApex:
    """An injected apex in L0 (fractional trace and sample)."""

    key: str
    apex_trace: float
    apex_sample: float


@dataclass(frozen=True)
class ScoredBox:
    """A merged line box in L0 with its class and score."""

    class_id: int
    score: float
    grid: GridBox


@dataclass(frozen=True)
class MatchOutcome:
    """Result of the one-to-one match for one apex.

    Attributes:
        hit: A box was assigned to the apex.
        box_index: Highest-score box assigned to the apex, else None.
        n_boxes: Boxes assigned to the apex (counted once as a hit).
    """

    hit: bool
    box_index: int | None
    n_boxes: int


NO_MATCH = MatchOutcome(hit=False, box_index=None, n_boxes=0)


# -------------------------------------------------------------- matching


def box_contains_apex(box: GridBox, apex: TruthApex, period_samples: float) -> bool:
    """Apex inside the box grown by +-3 traces and +-0.5 ``P_line`` samples."""
    sample_margin = MATCH_SAMPLE_PERIODS * period_samples
    inside_traces = (
        box.i0 - MATCH_TRACE_MARGIN <= apex.apex_trace <= box.i1 + MATCH_TRACE_MARGIN
    )
    inside_samples = (
        box.k0 - sample_margin <= apex.apex_sample <= box.k1 + sample_margin
    )
    return inside_traces and inside_samples


def _box_centre(box: GridBox) -> tuple[float, float]:
    """Continuous centre of a closed index box (cell ``j`` covers [j, j + 1))."""
    return (box.i0 + box.i1 + 1) / 2.0, (box.k0 + box.k1 + 1) / 2.0


def _distance_in_periods(
    box: GridBox, apex: TruthApex, period_traces: float, period_samples: float
) -> float:
    """Box-centre to apex distance with both axes in periods (L1 is isotropic)."""
    centre_trace, centre_sample = _box_centre(box)
    trace_offset = (apex.apex_trace - centre_trace) / period_traces
    sample_offset = (apex.apex_sample - centre_sample) / period_samples
    return math.hypot(trace_offset, sample_offset)


def match_one_to_one(
    apexes: Sequence[TruthApex],
    boxes: Sequence[ScoredBox],
    period_traces: float,
    period_samples: float,
    class_id: int | None = DetectionClass.PIPE_HYPERBOLA,
) -> list[MatchOutcome]:
    """One-to-one apex match of plan 3-2 ([CVR] appendix B-4).

    Each box (of ``class_id``; ``None`` means any class) that contains some
    apexes is assigned to the one nearest its centre (ties: the earlier
    apex). An apex is hit when at least one box was assigned to it; several
    boxes on one apex count as one hit.

    Returns:
        One outcome per apex, in input order.
    """
    assigned: list[list[int]] = [[] for _ in apexes]
    for box_index, box in enumerate(boxes):
        if class_id is not None and box.class_id != class_id:
            continue
        contained = [
            apex_index
            for apex_index, apex in enumerate(apexes)
            if box_contains_apex(box.grid, apex, period_samples)
        ]
        if not contained:
            continue
        nearest = min(
            contained,
            key=lambda index: (
                _distance_in_periods(
                    box.grid, apexes[index], period_traces, period_samples
                ),
                index,
            ),
        )
        assigned[nearest].append(box_index)
    outcomes = []
    for box_indices in assigned:
        if not box_indices:
            outcomes.append(NO_MATCH)
            continue
        best = max(box_indices, key=lambda index: (boxes[index].score, -index))
        outcomes.append(MatchOutcome(True, best, len(box_indices)))
    return outcomes


def grid_iou(a: GridBox, b: GridBox) -> float:
    """IoU of two closed index boxes counted in cells."""
    traces = min(a.i1, b.i1) - max(a.i0, b.i0) + 1
    samples = min(a.k1, b.k1) - max(a.k0, b.k0) + 1
    if traces <= 0 or samples <= 0:
        return 0.0
    intersection = traces * samples
    union = a.n_traces * a.n_samples + b.n_traces * b.n_samples - intersection
    return intersection / union


def best_iou(truth: GridBox, boxes: Iterable[ScoredBox], class_id: int) -> float:
    """Largest IoU between ``truth`` and a box of ``class_id`` (report only)."""
    values = [grid_iou(truth, box.grid) for box in boxes if box.class_id == class_id]
    return max(values, default=0.0)


# ------------------------------------------------------------ M6 (D3 rule)


def apex_hit_d3(
    apex_trace: float,
    apex_sample: float,
    boxes: Iterable[ScoredBox],
    period_samples: float,
) -> bool:
    """M6 hit exactly as D3 ``_apex_hit``: any ``pipe_hyperbola`` box contains it.

    Not one-to-one: the guard and its identity gate reproduce D3 (plan C-8).
    """
    apex = TruthApex("", apex_trace, apex_sample)
    return any(
        box.class_id == DetectionClass.PIPE_HYPERBOLA
        and box_contains_apex(box.grid, apex, period_samples)
        for box in boxes
    )


# ------------------------------------------------------------------- M4


@dataclass(frozen=True)
class ContentRange:
    """Line L1 content rows and the zero row, from the ``ImageGridMap``.

    Attributes:
        y_top_px: First content row (0 for ``DetectionImage``).
        y_bot_px: Content bottom ``H_c`` (rows below are padding).
        period_l1_px: ``P_L1``.
        zero_px: ``t_z / dt / samples_per_px``; ``None`` when the zero failed.
    """

    y_top_px: float
    y_bot_px: float
    period_l1_px: float
    zero_px: float | None


def m4_category(y1: float, y2: float, content: ContentRange) -> M4Category | None:
    """Primary M4 category of one box (any class); ``None`` if none applies.

    Order a > b > d, so a box is counted once in ``M4a + M4b + M4d``.
    """
    centre = 0.5 * (y1 + y2)
    if centre > content.y_bot_px:
        return M4Category.A
    if y1 < content.y_bot_px - EDGE_PX and y2 > content.y_bot_px + EDGE_PX:
        return M4Category.B
    if content.zero_px is not None:
        zero_limit = content.zero_px + M4D_ZERO_MARGIN_PERIODS * content.period_l1_px
        if centre < zero_limit:
            return M4Category.D
    return None


def is_m4c(
    class_id: int,
    y1: float,
    y2: float,
    content: ContentRange,
    window_px: float = M4C_WINDOW_PX,
) -> bool:
    """M4c (suspect): ``pipe_band`` near the content bottom, not M4a or M4b."""
    if class_id != DetectionClass.PIPE_BAND:
        return False
    if m4_category(y1, y2, content) in (M4Category.A, M4Category.B):
        return False
    return y2 >= content.y_bot_px - window_px


# ------------------------------------------------------ C-5 operating point


def conf_grid() -> tuple[float, ...]:
    """0.25, 0.26, ..., 0.95 as exactly rounded values."""
    n_steps = round((CONF_GRID_MAX - CONF_BASE) / CONF_GRID_STEP)
    return tuple(round(CONF_BASE + i * CONF_GRID_STEP, 2) for i in range(n_steps + 1))


def evaluate_on_conf_grid[T](
    scores: Sequence[float],
    evaluate: Callable[[np.ndarray], T],
    grid: Sequence[float] | None = None,
) -> list[T]:
    """``evaluate(keep)`` for every grid conf, keeping ``score > conf``.

    The kept set only changes when the conf passes a score, so ``evaluate``
    runs once per distinct kept set.

    Args:
        scores: Scores of the pre-merge tile boxes.
        evaluate: Called with the boolean keep mask.
        grid: Conf levels (default ``conf_grid()``).
    """
    levels = conf_grid() if grid is None else tuple(grid)
    values = np.asarray(scores, dtype=float)
    cache: dict[bytes, T] = {}
    results = []
    for level in levels:
        keep = values > level
        signature = keep.tobytes()
        if signature not in cache:
            cache[signature] = evaluate(keep)
        results.append(cache[signature])
    return results


def hit_mask_text(hits: Sequence[bool]) -> str:
    """``"1"``/``"0"`` per grid level (compact csv cell)."""
    return "".join("1" if hit else "0" for hit in hits)


# ------------------------------------------------------------------ M5 flip


def flip_grid_box(box: GridBox, n_traces: int) -> GridBox:
    """Mirror an L0 box left-right on a line of ``n_traces``."""
    return GridBox(
        i0=n_traces - 1 - box.i1, i1=n_traces - 1 - box.i0, k0=box.k0, k1=box.k1
    )


def flip_pairs(
    original: Sequence[ScoredBox], flipped_back: Sequence[ScoredBox]
) -> list[tuple[int, int | None, float]]:
    """M5 pairs: each original box and its best same-class flipped box.

    ``flipped_back`` boxes are already mirrored back to the original
    orientation. Greedy by original score, a flipped box is used once.

    Returns:
        ``(original index, flipped index or None, IoU)``; consistent when
        IoU >= 0.5.
    """
    order = sorted(range(len(original)), key=lambda i: (-original[i].score, i))
    used: set[int] = set()
    pairs = []
    for index in order:
        box = original[index]
        best_index, best_value = None, 0.0
        for other_index, other in enumerate(flipped_back):
            if other_index in used or other.class_id != box.class_id:
                continue
            value = grid_iou(box.grid, other.grid)
            if value > best_value:
                best_index, best_value = other_index, value
        if best_index is not None and best_value >= FLIP_IOU:
            used.add(best_index)
        else:
            best_index = None
        pairs.append((index, best_index, best_value))
    return sorted(pairs)
