"""Box-split-insensitive metrics for ``pipe_band`` and a per-subset AP50.

The IoU-based AP of ``pipe_band`` depends on how a session's labellers cut a
long horizontal band into boxes (M1-10 baseline section 4-2). The metrics here
measure how much of the band's horizontal extent is found instead, so one long
box and several adjacent pieces of the same band give the same value.

Definitions (U-3 decision, 2026-11-01). All boxes are ``x1, y1, x2, y2`` in the
same pixel frame; only ``pipe_band`` ground truth (GT) and predictions are
passed in.

* **Vertical match**: a prediction may cover a GT box only where their vertical
  overlap is at least ``min_vertical_fraction`` (0.5) of the GT height.
* **Layers**: boxes of one kind (GT or predictions) whose vertical overlap is at
  least ``min_vertical_fraction`` of the *smaller* height are joined into one
  layer (transitively). Pieces of one band share a layer; bands stacked at
  different depths are separate layers.
* **GT length** = sum over GT layers of the length of the union of the layer's
  x-intervals. **Covered GT length** = the same sum of the union of the parts of
  each GT x-interval that the x-intervals of its vertically matched predictions
  cover. Recall = covered / GT length.
* **Prediction length** and **matched prediction length** are defined the same
  way with the roles swapped (the vertical match still uses the GT height).
  Precision = matched / prediction length.
* **Fragment count** of a GT box = number of predictions that are vertically
  matched to it and overlap its x-interval with positive length.

:func:`true_positives_at_iou` and :func:`average_precision_50` compute AP50 from
saved predictions for any subset of images (for example one session). They are
written from the standard definitions (greedy IoU assignment of PASCAL VOC and
COCO; precision envelope sampled at the 101 COCO recall levels, see
:func:`average_precision_50`). Their per-class AP50 equals the value of the
Ultralytics validator, which is checked by tests, not used as a source.
:func:`greedy_assignment` is the one-to-one matching behind the AP and is
public so that analysis scripts build their confusion matrices with it. Only
numpy is used, so this module works in every project env (ADR-001).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_MIN_VERTICAL_FRACTION = 0.5
"""Vertical overlap needed, as a fraction of the GT height (U-3 decision)."""
AP_IOU_THRESHOLD = 0.5
_AP_RECALL_LEVELS = np.linspace(0, 1, 101)
"""The 101 recall levels 0, 0.01, ..., 1 of the COCO evaluation."""
_EMPTY_INTERVALS = np.zeros((0, 2))


@dataclass(frozen=True)
class BandCoverage:
    """Lengths (pixels) behind the band recall and precision of one or more images."""

    gt_length_px: float
    covered_gt_length_px: float
    prediction_length_px: float
    matched_prediction_length_px: float

    @property
    def recall(self) -> float | None:
        """Covered GT length / GT length; ``None`` when there is no GT band."""
        if self.gt_length_px <= 0:
            return None
        return self.covered_gt_length_px / self.gt_length_px

    @property
    def precision(self) -> float | None:
        """Matched / prediction length; ``None`` when there is no prediction."""
        if self.prediction_length_px <= 0:
            return None
        return self.matched_prediction_length_px / self.prediction_length_px

    def __add__(self, other: "BandCoverage") -> "BandCoverage":
        return BandCoverage(
            self.gt_length_px + other.gt_length_px,
            self.covered_gt_length_px + other.covered_gt_length_px,
            self.prediction_length_px + other.prediction_length_px,
            self.matched_prediction_length_px + other.matched_prediction_length_px,
        )


EMPTY_COVERAGE = BandCoverage(0.0, 0.0, 0.0, 0.0)


def _as_boxes(boxes: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def merge_intervals(intervals: np.ndarray) -> np.ndarray:
    """Union of 1-D intervals as sorted, disjoint ``(start, end)`` rows.

    Touching intervals (``end == next start``) are merged; empty intervals
    (``end <= start``) are dropped.
    """
    intervals = np.asarray(intervals, dtype=float).reshape(-1, 2)
    intervals = intervals[intervals[:, 1] > intervals[:, 0]]
    if len(intervals) == 0:
        return _EMPTY_INTERVALS.copy()
    intervals = intervals[np.argsort(intervals[:, 0], kind="stable")]
    merged = [intervals[0].copy()]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append(np.array([start, end]))
    return np.array(merged)


def total_length(intervals: np.ndarray) -> float:
    """Length of the union of ``intervals``."""
    merged = merge_intervals(intervals)
    return float((merged[:, 1] - merged[:, 0]).sum())


def clip_intervals(intervals: np.ndarray, start: float, end: float) -> np.ndarray:
    """Parts of ``intervals`` that lie inside ``[start, end]`` (possibly empty rows)."""
    intervals = np.asarray(intervals, dtype=float).reshape(-1, 2)
    return np.column_stack(
        [np.maximum(intervals[:, 0], start), np.minimum(intervals[:, 1], end)]
    )


def vertical_overlap(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vertical overlap length (pixels, >= 0) between every box of a and of b."""
    boxes_a, boxes_b = _as_boxes(boxes_a), _as_boxes(boxes_b)
    top = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    bottom = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    return np.clip(bottom - top, 0, None)


def horizontal_overlap(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Horizontal overlap length (pixels, >= 0) between every box of a and of b."""
    boxes_a, boxes_b = _as_boxes(boxes_a), _as_boxes(boxes_b)
    left = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    right = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    return np.clip(right - left, 0, None)


def vertical_match(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    min_vertical_fraction: float = DEFAULT_MIN_VERTICAL_FRACTION,
) -> np.ndarray:
    """``(n_gt, n_pred)`` bool: vertical overlap >= fraction x GT height."""
    gt_boxes = _as_boxes(gt_boxes)
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    overlap = vertical_overlap(gt_boxes, pred_boxes)
    return overlap >= min_vertical_fraction * gt_heights[:, None]


def group_layers(
    boxes: np.ndarray, min_vertical_fraction: float = DEFAULT_MIN_VERTICAL_FRACTION
) -> list[np.ndarray]:
    """Indices of boxes joined into vertical layers (see the module docstring).

    Two boxes are linked when their vertical overlap is at least the fraction
    of the smaller height; layers are the connected groups of links.
    """
    boxes = _as_boxes(boxes)
    heights = boxes[:, 3] - boxes[:, 1]
    smaller_height = np.minimum(heights[:, None], heights[None, :])
    is_linked = vertical_overlap(boxes, boxes) >= min_vertical_fraction * smaller_height
    layer_of = np.full(len(boxes), -1)
    n_layers = 0
    for seed in range(len(boxes)):
        if layer_of[seed] >= 0:
            continue
        stack = [seed]
        layer_of[seed] = n_layers
        while stack:
            current = stack.pop()
            for neighbour in np.flatnonzero(is_linked[current] & (layer_of < 0)):
                layer_of[neighbour] = n_layers
                stack.append(neighbour)
        n_layers += 1
    return [np.flatnonzero(layer_of == layer) for layer in range(n_layers)]


def _layered_lengths(
    boxes: np.ndarray, partners: np.ndarray, is_partner: np.ndarray, layers
) -> tuple[float, float]:
    """(own length, length covered by partners) summed over the layers of ``boxes``.

    ``is_partner[i, j]`` says whether partner box ``j`` may cover box ``i``.
    """
    own_length = 0.0
    covered_length = 0.0
    for layer in layers:
        own_length += total_length(boxes[layer][:, [0, 2]])
        covered_pieces = []
        for index in layer:
            partner_intervals = merge_intervals(partners[is_partner[index]][:, [0, 2]])
            covered_pieces.append(
                clip_intervals(partner_intervals, boxes[index, 0], boxes[index, 2])
            )
        if covered_pieces:
            covered_length += total_length(np.concatenate(covered_pieces))
    return own_length, covered_length


def band_coverage(
    gt_boxes: np.ndarray | Sequence[Sequence[float]],
    pred_boxes: np.ndarray | Sequence[Sequence[float]],
    min_vertical_fraction: float = DEFAULT_MIN_VERTICAL_FRACTION,
) -> BandCoverage:
    """Horizontal-extent recall and precision lengths of one image.

    Args:
        gt_boxes: ``(n, 4)`` ``pipe_band`` GT boxes, ``x1, y1, x2, y2`` pixels.
        pred_boxes: ``(m, 4)`` ``pipe_band`` predictions (already filtered by score).
        min_vertical_fraction: Vertical overlap needed, as a fraction of the GT
            height, for a prediction to cover a GT box.

    Returns:
        The four lengths; add the results of several images for the
        length-weighted totals.
    """
    gt_boxes, pred_boxes = _as_boxes(gt_boxes), _as_boxes(pred_boxes)
    is_match = vertical_match(gt_boxes, pred_boxes, min_vertical_fraction)
    gt_length, covered_gt = _layered_lengths(
        gt_boxes,
        pred_boxes,
        is_match,
        group_layers(gt_boxes, min_vertical_fraction),
    )
    pred_length, matched_pred = _layered_lengths(
        pred_boxes,
        gt_boxes,
        is_match.T,
        group_layers(pred_boxes, min_vertical_fraction),
    )
    return BandCoverage(gt_length, covered_gt, pred_length, matched_pred)


def fragment_counts(
    gt_boxes: np.ndarray | Sequence[Sequence[float]],
    pred_boxes: np.ndarray | Sequence[Sequence[float]],
    min_vertical_fraction: float = DEFAULT_MIN_VERTICAL_FRACTION,
) -> np.ndarray:
    """Per GT box: predictions vertically matched to it that overlap it in x."""
    gt_boxes, pred_boxes = _as_boxes(gt_boxes), _as_boxes(pred_boxes)
    is_match = vertical_match(gt_boxes, pred_boxes, min_vertical_fraction)
    overlaps_in_x = horizontal_overlap(gt_boxes, pred_boxes) > 0
    return (is_match & overlaps_in_x).sum(axis=1)


def merged_band_fragment_counts(
    gt_boxes: np.ndarray | Sequence[Sequence[float]],
    pred_boxes: np.ndarray | Sequence[Sequence[float]],
    min_vertical_fraction: float = DEFAULT_MIN_VERTICAL_FRACTION,
) -> np.ndarray:
    """Fragment count per merged GT band instead of per GT box.

    A merged band is one connected x-segment of a GT layer, so a band labelled
    as several touching pieces counts once. A prediction counts for the band
    when it is a fragment (see :func:`fragment_counts`) of any of its pieces.
    """
    gt_boxes, pred_boxes = _as_boxes(gt_boxes), _as_boxes(pred_boxes)
    is_fragment = vertical_match(gt_boxes, pred_boxes, min_vertical_fraction) & (
        horizontal_overlap(gt_boxes, pred_boxes) > 0
    )
    counts = []
    for layer in group_layers(gt_boxes, min_vertical_fraction):
        for start, end in merge_intervals(gt_boxes[layer][:, [0, 2]]):
            inside = layer[(gt_boxes[layer, 0] >= start) & (gt_boxes[layer, 2] <= end)]
            counts.append(int(is_fragment[inside].any(axis=0).sum()))
    return np.array(counts, dtype=int)


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU (intersection area / union area) between every box of a and of b.

    Boxes are ``x1, y1, x2, y2``. Pairs whose union is empty (two degenerate
    boxes) get IoU 0.
    """
    boxes_a, boxes_b = _as_boxes(boxes_a), _as_boxes(boxes_b)
    intersection = horizontal_overlap(boxes_a, boxes_b) * vertical_overlap(
        boxes_a, boxes_b
    )
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    iou = np.zeros_like(union)
    np.divide(intersection, union, out=iou, where=union > 0)
    return iou


UNASSIGNED = -1
"""Column index :func:`greedy_assignment` gives a row that took no GT box."""


def _is_match(iou: float, iou_threshold: float, is_threshold_inclusive: bool) -> bool:
    if is_threshold_inclusive:
        return iou >= iou_threshold
    return iou > iou_threshold


def greedy_assignment(
    iou: np.ndarray, iou_threshold: float, *, is_threshold_inclusive: bool = True
) -> np.ndarray:
    """One-to-one greedy assignment of predictions (rows) to GT boxes (columns).

    Rows are visited top to bottom, so pass them in descending score. Each row
    takes the still-unassigned GT column with the highest IoU (lowest column
    on ties) when that IoU passes the threshold; otherwise it takes nothing and
    the column stays free for later rows.

    Args:
        iou: ``(n_pred, n_gt)`` IoU matrix, rows in visiting order.
        iou_threshold: IoU a pair needs to be assigned.
        is_threshold_inclusive: ``True`` matches at IoU >= threshold (PASCAL
            VOC, COCO), ``False`` only above it.

    Returns:
        ``(n_pred,)`` int array: assigned GT column per row, or
        :data:`UNASSIGNED`.
    """
    iou = np.asarray(iou, dtype=float)
    assigned_column = np.full(iou.shape[0], UNASSIGNED, dtype=int)
    if iou.shape[1] == 0:
        return assigned_column
    is_gt_taken = np.zeros(iou.shape[1], dtype=bool)
    for row, row_iou in enumerate(iou):
        free_iou = np.where(is_gt_taken, -np.inf, row_iou)
        best_column = int(free_iou.argmax())
        if _is_match(free_iou[best_column], iou_threshold, is_threshold_inclusive):
            assigned_column[row] = best_column
            is_gt_taken[best_column] = True
    return assigned_column


def true_positives_at_iou(
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    iou_threshold: float = AP_IOU_THRESHOLD,
) -> np.ndarray:
    """True-positive flag per prediction of one image (greedy IoU assignment).

    The assignment of the PASCAL VOC and COCO detection evaluations, done per
    class: predictions, in descending score order, each take the unassigned GT
    box of their class with the highest IoU, and count as true positives when
    that IoU is at least ``iou_threshold`` (> 0). Every GT box is taken at
    most once.

    Args:
        gt_boxes: ``(n, 4)`` GT boxes, ``x1, y1, x2, y2``.
        gt_classes: ``(n,)`` GT class ids.
        pred_boxes: ``(m, 4)`` predictions, sorted by descending score (the
            order the detector's NMS returns).
        pred_classes: ``(m,)`` predicted class ids, same order.
        iou_threshold: Lowest IoU that makes a match.

    Returns:
        ``(m,)`` bool array.
    """
    gt_boxes, pred_boxes = _as_boxes(gt_boxes), _as_boxes(pred_boxes)
    gt_classes, pred_classes = np.asarray(gt_classes), np.asarray(pred_classes)
    is_true_positive = np.zeros(len(pred_classes), dtype=bool)
    for class_id in np.intersect1d(gt_classes, pred_classes):
        gt_of_class = np.flatnonzero(gt_classes == class_id)
        predictions_of_class = np.flatnonzero(pred_classes == class_id)
        iou = box_iou(pred_boxes[predictions_of_class], gt_boxes[gt_of_class])
        assigned_gt = greedy_assignment(iou, iou_threshold)
        is_true_positive[predictions_of_class] = assigned_gt != UNASSIGNED
    return is_true_positive


def _precision_envelope(precision: np.ndarray) -> np.ndarray:
    """Highest precision at the same or a later rank (a non-increasing curve)."""
    return np.maximum.accumulate(precision[::-1])[::-1]


def average_precision_50(
    is_correct: np.ndarray,
    scores: np.ndarray,
    pred_classes: np.ndarray,
    n_gt: int,
    class_id: int,
) -> float | None:
    """AP of one class from ranked true-positive flags (AP50 at IoU 0.5 flags).

    Definition. Rank the predictions of ``class_id`` by descending score; at
    rank k, recall = TP(k) / ``n_gt`` and precision = TP(k) / k. Close the
    precision-recall curve with the points (recall 0, precision 1) in front and
    (last recall, 0), (1, 0) behind, replace each precision by the precision
    envelope (highest precision at equal or higher recall, PASCAL VOC 2010+ and
    COCO), sample it with linear interpolation at the 101 COCO recall levels
    0, 0.01, ..., 1, and integrate over recall with the trapezoidal rule.

    The COCO API (``COCOeval``) instead reads the envelope at the first rank that
    reaches each recall level and averages the 101 values, so its AP differs
    slightly. The per-class AP50 of this definition equals that of the Ultralytics
    8.4.192 validator; the equivalence is checked on the ten M1-10 evaluations
    (difference 0.0) and in ``tests/test_band_metrics.py``.

    Args:
        is_correct: True-positive flag per prediction (all classes, concatenated
            over images in validation order).
        scores: Score per prediction, same order.
        pred_classes: Class per prediction, same order.
        n_gt: Number of GT boxes of ``class_id`` in the evaluated images.
        class_id: Class to score.

    Returns:
        AP in [0, 1); ``0.0`` when the class has GT but no prediction, ``None``
        when it has no GT.
    """
    if n_gt == 0:
        return None
    # One ranking of all classes with numpy's default sort, so that tied scores
    # keep the order the validator's AP sees (this decides AP on ties).
    ranking = np.argsort(-np.asarray(scores, dtype=float))
    is_of_class = np.asarray(pred_classes)[ranking] == class_id
    hits = np.asarray(is_correct)[ranking][is_of_class].astype(float)
    if len(hits) == 0:
        return 0.0
    true_positives = hits.cumsum()
    recall = true_positives / n_gt
    precision = true_positives / np.arange(1, len(hits) + 1)
    recall_curve = np.hstack([0.0, recall, recall[-1], 1.0])
    precision_curve = np.hstack([1.0, precision, 0.0, 0.0])
    sampled_precision = np.interp(
        _AP_RECALL_LEVELS, recall_curve, _precision_envelope(precision_curve)
    )
    return float(np.trapezoid(sampled_precision, _AP_RECALL_LEVELS))
