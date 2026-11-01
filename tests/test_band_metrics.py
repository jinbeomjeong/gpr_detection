"""Tests of the split-insensitive pipe_band metrics and the per-subset AP50."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpr_detection.detect import band_metrics
from gpr_detection.detect.band_metrics import (
    AP_IOU_THRESHOLD,
    EMPTY_COVERAGE,
    UNASSIGNED,
    average_precision_50,
    band_coverage,
    box_iou,
    fragment_counts,
    greedy_assignment,
    group_layers,
    merge_intervals,
    merged_band_fragment_counts,
    true_positives_at_iou,
)

ONE_BAND = [[0.0, 10.0, 100.0, 20.0]]
ONE_BAND_IN_TWO_PIECES = [[0.0, 10.0, 50.0, 20.0], [50.0, 10.0, 100.0, 20.0]]
ONE_BAND_IN_OVERLAPPING_PIECES = [[0.0, 11.0, 60.0, 20.0], [40.0, 10.0, 100.0, 19.0]]


def _recall_precision(gt, pred) -> tuple[float | None, float | None]:
    coverage = band_coverage(gt, pred)
    return coverage.recall, coverage.precision


def test_merge_intervals_joins_touching_and_drops_empty() -> None:
    merged = merge_intervals([[5, 8], [0, 5], [10, 10], [12, 15], [14, 20]])
    assert merged.tolist() == [[0, 8], [12, 20]]


@pytest.mark.parametrize("gt", [ONE_BAND, ONE_BAND_IN_TWO_PIECES])
@pytest.mark.parametrize(
    "pred",
    [
        [[20.0, 10.0, 80.0, 20.0]],
        [[20.0, 10.0, 50.0, 20.0], [50.0, 10.0, 80.0, 20.0]],
        [[20.0, 10.0, 60.0, 20.0], [40.0, 11.0, 80.0, 19.0]],
    ],
)
def test_split_and_merged_boxes_give_the_same_values(gt, pred) -> None:
    recall, precision = _recall_precision(gt, pred)
    assert recall == pytest.approx(0.6)
    assert precision == pytest.approx(1.0)


def test_overlapping_gt_pieces_are_not_counted_twice() -> None:
    pred = [[0.0, 10.0, 50.0, 20.0]]
    assert _recall_precision(ONE_BAND_IN_OVERLAPPING_PIECES, pred) == (
        pytest.approx(0.5),
        pytest.approx(1.0),
    )


def test_prediction_beyond_the_band_lowers_precision() -> None:
    recall, precision = _recall_precision(ONE_BAND, [[50.0, 10.0, 150.0, 20.0]])
    assert recall == pytest.approx(0.5)
    assert precision == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("pred_top", "is_covered"),
    [(15.0, True), (15.001, False)],
)
def test_vertical_overlap_boundary_is_half_the_gt_height(pred_top, is_covered) -> None:
    # GT height 10 px (y 10-20); the prediction overlaps 20 - pred_top px.
    recall, precision = _recall_precision(ONE_BAND, [[0.0, pred_top, 100.0, 40.0]])
    assert recall == (1.0 if is_covered else 0.0)
    assert precision == (1.0 if is_covered else 0.0)


def test_vertical_rule_uses_the_gt_height_not_the_prediction_height() -> None:
    # A tall prediction holding the whole GT band covers it (overlap = GT height).
    recall, _ = _recall_precision(ONE_BAND, [[0.0, 0.0, 100.0, 200.0]])
    assert recall == pytest.approx(1.0)


def test_empty_gt_gives_no_recall_and_zero_precision() -> None:
    coverage = band_coverage(np.zeros((0, 4)), [[0.0, 0.0, 30.0, 5.0]])
    assert coverage.recall is None
    assert coverage.precision == 0.0
    assert coverage.prediction_length_px == pytest.approx(30.0)


def test_empty_prediction_gives_zero_recall_and_no_precision() -> None:
    coverage = band_coverage(ONE_BAND, [])
    assert coverage.recall == 0.0
    assert coverage.precision is None


def test_empty_image_adds_nothing() -> None:
    assert band_coverage([], []) == EMPTY_COVERAGE
    assert EMPTY_COVERAGE.recall is None
    assert EMPTY_COVERAGE.precision is None


STACKED_BANDS = [[0.0, 10.0, 100.0, 20.0], [0.0, 50.0, 100.0, 60.0]]


def test_stacked_bands_are_separate_layers() -> None:
    assert [layer.tolist() for layer in group_layers(STACKED_BANDS)] == [[0], [1]]
    recall, precision = _recall_precision(STACKED_BANDS, [[0.0, 10.0, 100.0, 20.0]])
    assert recall == pytest.approx(0.5)  # the deeper band is missed
    assert precision == pytest.approx(1.0)


def test_prediction_spanning_two_stacked_bands_covers_both() -> None:
    recall, precision = _recall_precision(STACKED_BANDS, [[0.0, 5.0, 100.0, 65.0]])
    assert recall == pytest.approx(1.0)
    assert precision == pytest.approx(1.0)


def test_duplicate_predictions_on_one_band_do_not_lower_precision() -> None:
    pred = [[0.0, 10.0, 100.0, 20.0], [0.0, 11.0, 100.0, 21.0]]
    assert _recall_precision(ONE_BAND, pred) == (1.0, 1.0)


def test_prediction_at_another_depth_is_its_own_layer() -> None:
    pred = [[0.0, 10.0, 100.0, 20.0], [0.0, 80.0, 100.0, 90.0]]
    assert _recall_precision(ONE_BAND, pred) == (1.0, 0.5)


def test_coverage_adds_up_length_weighted() -> None:
    total = band_coverage(ONE_BAND, [[0.0, 10.0, 100.0, 20.0]]) + band_coverage(
        [[0.0, 10.0, 300.0, 20.0]], []
    )
    assert total.recall == pytest.approx(100 / 400)
    assert total.precision == pytest.approx(1.0)


def test_fragment_counts_per_gt_box_and_per_merged_band() -> None:
    pred = [
        [0.0, 10.0, 30.0, 20.0],
        [35.0, 10.0, 70.0, 20.0],
        [75.0, 10.0, 100.0, 20.0],
        [0.0, 80.0, 100.0, 90.0],  # another depth: not a fragment
    ]
    assert fragment_counts(ONE_BAND, pred).tolist() == [3]
    assert fragment_counts(ONE_BAND_IN_TWO_PIECES, pred).tolist() == [2, 2]
    assert merged_band_fragment_counts(ONE_BAND_IN_TWO_PIECES, pred).tolist() == [3]
    assert merged_band_fragment_counts(STACKED_BANDS, pred).tolist() == [3, 0]


def test_touching_prediction_is_not_a_fragment() -> None:
    assert fragment_counts(ONE_BAND, [[100.0, 10.0, 120.0, 20.0]]).tolist() == [0]


def test_true_positives_follow_score_order_and_class() -> None:
    gt_boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    pred_boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 9.0]] * 2)
    pred_classes = np.array([1, 1, 0, 1])
    is_correct = true_positives_at_iou(
        gt_boxes, np.array([1]), pred_boxes, pred_classes
    )
    assert is_correct.tolist() == [True, False, False, False]


def test_prediction_takes_the_best_unassigned_gt() -> None:
    gt_boxes = np.array([[0.0, 0.0, 10.0, 10.0], [2.0, 0.0, 12.0, 10.0]])
    # The first prediction fits GT 1 best; the second then gets GT 0.
    pred_boxes = np.array([[2.0, 0.0, 12.0, 10.0], [1.0, 0.0, 11.0, 10.0]])
    is_correct = true_positives_at_iou(gt_boxes, [0, 0], pred_boxes, [0, 0])
    assert is_correct.tolist() == [True, True]


def test_iou_exactly_at_the_threshold_is_a_match() -> None:
    # IoU = 50 / 100 = 0.5 exactly; COCO counts IoU >= threshold.
    is_correct = true_positives_at_iou(
        [[0.0, 0.0, 10.0, 10.0]], [0], [[0.0, 0.0, 10.0, 5.0]], [0]
    )
    assert is_correct.tolist() == [True]


def test_strict_assignment_leaves_iou_at_the_threshold_unassigned() -> None:
    iou = np.array([[0.5, 0.2], [0.6, 0.5]])
    inclusive = greedy_assignment(iou, 0.5)
    strict = greedy_assignment(iou, 0.5, is_threshold_inclusive=False)
    assert inclusive.tolist() == [0, 1]
    assert strict.tolist() == [UNASSIGNED, 0]


def test_assignment_without_gt_assigns_nothing() -> None:
    assert greedy_assignment(np.zeros((3, 0)), 0.5).tolist() == [UNASSIGNED] * 3


def test_box_iou_of_degenerate_boxes_is_zero() -> None:
    iou = box_iou([[5.0, 5.0, 5.0, 5.0]], [[5.0, 5.0, 5.0, 5.0], [0.0, 0.0, 2.0, 2.0]])
    assert iou.tolist() == [[0.0, 0.0]]


def test_average_precision_of_a_perfect_and_an_empty_ranking() -> None:
    # A perfect ranking gives 0.995, not 1: the closing point (recall 1,
    # precision 0) makes the last trapezoid (recall 0.99 to 1) a triangle.
    assert average_precision_50([True, True], [0.9, 0.8], [1, 1], 2, 1) == (
        pytest.approx(0.995)
    )
    assert average_precision_50([], [], [], 3, 1) == 0.0
    assert average_precision_50([True], [0.9], [1], 0, 1) is None


def test_average_precision_follows_the_envelope_definition() -> None:
    # Two of three GT found; ranking TP, FP, TP. The envelope is precision 1 up
    # to recall 1/3, then 2/3 up to recall 2/3, then 0.
    value = average_precision_50([True, False, True], [0.9, 0.8, 0.7], [1, 1, 1], 3, 1)
    recall_levels = np.linspace(0, 1, 101)
    expected_curve = np.interp(
        recall_levels,
        [0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1],
        [1, 1, 2 / 3, 2 / 3, 0, 0],
    )
    assert value == pytest.approx(np.trapezoid(expected_curve, recall_levels))


def test_average_precision_ignores_other_classes() -> None:
    with_other_class = average_precision_50(
        [True, False, True], [0.9, 0.8, 0.7], [1, 0, 1], 2, 1
    )
    alone = average_precision_50([True, True], [0.9, 0.7], [1, 1], 2, 1)
    assert with_other_class == alone


# --- equivalence with the Ultralytics validator (gpr-dev only) ------------------------

EQUIVALENCE_SEED = 20261101
EQUIVALENCE_IMAGES = 60
EQUIVALENCE_CLASSES = 4


def _random_image(rng: np.random.Generator):
    """GT boxes plus jittered copies and stray predictions, scores sorted."""
    n_gt = int(rng.integers(0, 6))
    corners = rng.uniform(0, 200, size=(n_gt, 2))
    gt_boxes = np.hstack([corners, corners + rng.uniform(5, 60, size=(n_gt, 2))])
    gt_classes = rng.integers(0, EQUIVALENCE_CLASSES, size=n_gt)
    copies = np.repeat(np.arange(n_gt), rng.integers(0, 3, size=n_gt))
    jitter = rng.normal(0, 4, size=(len(copies), 4))
    n_stray = int(rng.integers(0, 4))
    stray_corners = rng.uniform(0, 200, size=(n_stray, 2))
    stray = np.hstack([stray_corners, stray_corners + rng.uniform(5, 60, (n_stray, 2))])
    pred_boxes = np.vstack([gt_boxes[copies] + jitter, stray]).reshape(-1, 4)
    pred_boxes[:, 2:] = np.maximum(pred_boxes[:, 2:], pred_boxes[:, :2] + 1)
    pred_classes = np.concatenate(
        [gt_classes[copies], rng.integers(0, EQUIVALENCE_CLASSES, size=n_stray)]
    )
    # Two decimals make tied scores, which the AP must order as the validator.
    scores = np.round(rng.uniform(0.01, 1.0, size=len(pred_classes)), 2)
    order = np.argsort(-scores, kind="stable")
    return gt_boxes, gt_classes, pred_boxes[order], pred_classes[order], scores[order]


def test_ap50_equals_the_ultralytics_validator() -> None:
    """Value check only; IoU exactly at 0.5 is excluded by the continuous boxes.

    Ultralytics adds 1e-7 to the IoU union, so a pair at exactly IoU 0.5 is no
    match there and a match here (see the threshold test above).
    """
    torch = pytest.importorskip("torch")
    ultralytics_metrics = pytest.importorskip("ultralytics.utils.metrics")
    from ultralytics.engine.validator import BaseValidator

    validator = SimpleNamespace(iouv=torch.tensor([AP_IOU_THRESHOLD]))
    rng = np.random.default_rng(EQUIVALENCE_SEED)
    ours, reference, scores, classes, gt_classes = [], [], [], [], []
    for _ in range(EQUIVALENCE_IMAGES):
        gt_boxes, gt_cls, pred_boxes, pred_cls, pred_scores = _random_image(rng)
        ours.append(true_positives_at_iou(gt_boxes, gt_cls, pred_boxes, pred_cls))
        if len(gt_cls) and len(pred_cls):
            iou = ultralytics_metrics.box_iou(
                torch.tensor(gt_boxes), torch.tensor(pred_boxes)
            )
            matched = BaseValidator.match_predictions(
                validator, torch.tensor(pred_cls), torch.tensor(gt_cls), iou
            )
            reference.append(matched[:, 0].numpy())
        else:
            reference.append(np.zeros(len(pred_cls), dtype=bool))
        scores.append(pred_scores.astype(np.float32))
        classes.append(pred_cls)
        gt_classes.append(gt_cls)
    is_correct, reference_correct = np.concatenate(ours), np.concatenate(reference)
    assert is_correct.tolist() == reference_correct.tolist()

    all_scores, all_classes = np.concatenate(scores), np.concatenate(classes)
    all_gt_classes = np.concatenate(gt_classes)
    result = ultralytics_metrics.ap_per_class(
        reference_correct[:, None], all_scores, all_classes, all_gt_classes
    )
    reference_ap = dict(zip(result[6].tolist(), result[5][:, 0].tolist(), strict=True))
    for class_id, expected in reference_ap.items():
        n_gt = int((all_gt_classes == class_id).sum())
        value = average_precision_50(
            is_correct, all_scores, all_classes, n_gt, class_id
        )
        assert value == expected


def test_module_imports_numpy_only() -> None:
    tree = ast.parse(Path(band_metrics.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"collections", "dataclasses", "numpy"}
