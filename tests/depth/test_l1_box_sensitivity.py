"""Box-top sensitivity of the depth to L1 quantisation (spec_detection_image_v1 T10).

Report test (opt-in, ``-m synthetic``): for the 20 gprMax models the M2-5
supply box goes to the line L1 of ``make_detection_image`` and back to L0
after (i) outward quantisation to whole L1 px and (ii) moving the box top by
+-1 / +-2 L1 px. Every depth change and flag set is written to
``data/processing/l1_box_sensitivity.csv`` (git-ignored). Criterion (i)
``|delta d| <= 1 cm`` is an assumption: exceedances are reported as warnings,
not failures; (ii) has no criterion.
"""

import csv
import math
import warnings
from pathlib import Path

import pytest

from gpr_detection.depth.estimate import estimate_depth, prepare_depth_inputs
from gpr_detection.detect.base import PixelBox
from gpr_detection.io.paths import DEFAULT_SYNTHETIC_ROOT
from gpr_detection.io.synthetic import (
    list_synthetic_models,
    load_synthetic,
    load_synthetic_truth,
)
from gpr_detection.pipeline import grid_box_to_image_box, image_box_to_grid_box
from gpr_detection.preprocess.to_image import make_detection_image

pytestmark = pytest.mark.synthetic

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_CSV = PROJECT_ROOT / "data" / "processing" / "l1_box_sensitivity.csv"
QUANTISATION_LIMIT_M = 0.01
TOP_SHIFTS_PX = (-2, -1, 1, 2)
CSV_FIELDS = (
    "model_id",
    "variant",
    "i0",
    "i1",
    "k0",
    "k1",
    "status",
    "depth_top_m",
    "delta_depth_m",
    "flags",
)


def _outward_px(box: PixelBox) -> PixelBox:
    return PixelBox(
        math.floor(box.x1), math.floor(box.y1), math.ceil(box.x2), math.ceil(box.y2)
    )


def _shift_top(box: PixelBox, shift_px: float) -> PixelBox:
    return PixelBox(box.x1, min(box.y1 + shift_px, box.y2), box.x2, box.y2)


def _row(model_id, variant, grid_box, result, base_depth) -> dict[str, object]:
    delta = None
    if result.depth_top_m is not None and base_depth is not None:
        delta = result.depth_top_m - base_depth
    return {
        "model_id": model_id,
        "variant": variant,
        "i0": grid_box.i0,
        "i1": grid_box.i1,
        "k0": grid_box.k0,
        "k1": grid_box.k1,
        "status": str(result.status),
        "depth_top_m": result.depth_top_m,
        "delta_depth_m": delta,
        "flags": ";".join(result.flags),
    }


def test_t10_box_top_sensitivity_report(load_script) -> None:
    if not DEFAULT_SYNTHETIC_ROOT.is_dir():
        pytest.skip(f"synthetic set not found at {DEFAULT_SYNTHETIC_ROOT}")
    supply_box = load_script("eval_depth_synthetic").supply_box
    rows = []
    exceeded = []
    for model_id in list_synthetic_models(DEFAULT_SYNTHETIC_ROOT):
        radargram = load_synthetic(model_id, root=DEFAULT_SYNTHETIC_ROOT)
        truth = load_synthetic_truth(model_id, root=DEFAULT_SYNTHETIC_ROOT)
        grid_map = make_detection_image(radargram).grid_map
        inputs = prepare_depth_inputs(radargram)
        supplied = supply_box(truth, radargram)
        base = estimate_depth(radargram, supplied, inputs=inputs)
        rows.append(_row(model_id, "supplied", supplied, base, base.depth_top_m))

        line_box = grid_box_to_image_box(supplied, grid_map)
        variants = {"quantised": _outward_px(line_box)}
        for shift in TOP_SHIFTS_PX:
            variants[f"top{shift:+d}px"] = _shift_top(line_box, shift)
        for variant, box in variants.items():
            grid_box = image_box_to_grid_box(box, grid_map)
            assert grid_box is not None
            result = estimate_depth(radargram, grid_box, inputs=inputs)
            row = _row(model_id, variant, grid_box, result, base.depth_top_m)
            rows.append(row)
            delta = row["delta_depth_m"]
            if variant == "quantised" and (
                delta is None or abs(delta) > QUANTISATION_LIMIT_M
            ):
                exceeded.append((model_id, delta, result.status))

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if exceeded:
        warnings.warn(
            f"T10 (i) |delta d| > {QUANTISATION_LIMIT_M} m or no depth: {exceeded}",
            stacklevel=1,
        )
    assert len(rows) == len(list_synthetic_models(DEFAULT_SYNTHETIC_ROOT)) * 6
