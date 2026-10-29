"""Joins the detection and depth branches; owns the shared coordinate mapping.

The inverse mapping from model input (L2) through the detection image (L1) to
the radargram grid (L0) is implemented **only here**
(``docs/architecture/README.md`` section 2-3, user decision 2026-10-29):

* L2 -> L1: ``u = (x - pad_x) / scale_x`` with the backend's ``InputTransform``
* L1 -> L0: ``i = trace_offset + u * traces_per_px`` with the ``ImageGridMap``,
  rounded outward to whole traces/samples and clipped to the grid

Pixel convention: continuous coordinates, cell ``j`` covers ``[j, j + 1)``. A
continuous trace span ``[t1, t2]`` therefore covers the closed index range
``floor(t1) .. ceil(t2) - 1``.
"""

import math

from gpr_detection.detect.base import (
    Detection,
    DetectorOutput,
    InputTransform,
    PixelBox,
)
from gpr_detection.io.radargram import GridBox
from gpr_detection.preprocess.to_image import ImageGridMap

GRID_ROUNDING_TOLERANCE = 1e-6
"""Grid units ignored when rounding outward, so float noise such as 9.9999999
does not widen a box by a whole trace."""


def _outward_index_range(
    start: float, stop: float, grid_size: int
) -> tuple[int, int] | None:
    """Closed index range covered by ``[start, stop]``, clipped to the grid."""
    first = math.floor(start + GRID_ROUNDING_TOLERANCE)
    last = math.ceil(stop - GRID_ROUNDING_TOLERANCE) - 1
    last = max(last, first)  # a zero-width box still covers one cell
    first = max(first, 0)
    last = min(last, grid_size - 1)
    if first > last:
        return None
    return first, last


def model_box_to_image_box(box: PixelBox, transform: InputTransform) -> PixelBox:
    """Undo the backend resize/padding: L2 box -> L1 box (not clipped)."""
    return PixelBox(
        x1=(box.x1 - transform.pad_x) / transform.scale_x,
        y1=(box.y1 - transform.pad_y) / transform.scale_y,
        x2=(box.x2 - transform.pad_x) / transform.scale_x,
        y2=(box.y2 - transform.pad_y) / transform.scale_y,
    )


def image_box_to_grid_box(box: PixelBox, grid_map: ImageGridMap) -> GridBox | None:
    """Map an L1 box to the radargram grid (L0), rounding outward.

    Returns:
        The covering ``GridBox`` clipped to the radargram, or None when the box
        lies completely outside the grid (e.g. inside letterbox padding).
    """
    trace_start = grid_map.trace_offset + box.x1 * grid_map.traces_per_px
    trace_stop = grid_map.trace_offset + box.x2 * grid_map.traces_per_px
    sample_start = grid_map.sample_offset + box.y1 * grid_map.samples_per_px
    sample_stop = grid_map.sample_offset + box.y2 * grid_map.samples_per_px
    traces = _outward_index_range(trace_start, trace_stop, grid_map.n_traces)
    samples = _outward_index_range(sample_start, sample_stop, grid_map.n_samples)
    if traces is None or samples is None:
        return None
    return GridBox(i0=traces[0], i1=traces[1], k0=samples[0], k1=samples[1])


def model_box_to_grid_box(
    box: PixelBox, transform: InputTransform, grid_map: ImageGridMap
) -> GridBox | None:
    """Map a model-input box (L2) all the way to the radargram grid (L0).

    Returns:
        The covering ``GridBox``, or None when the box misses the grid.
    """
    return image_box_to_grid_box(model_box_to_image_box(box, transform), grid_map)


def detections_to_grid_boxes(
    output: DetectorOutput, grid_map: ImageGridMap
) -> list[tuple[Detection, GridBox]]:
    """Pair every detection with its L0 box; detections outside the grid are dropped.

    Args:
        output: Backend result (L2 boxes and the transform behind them).
        grid_map: Map of the detection image that was passed to the backend.
    """
    pairs = []
    for detection in output.detections:
        grid_box = model_box_to_grid_box(detection.box, output.transform, grid_map)
        if grid_box is not None:
            pairs.append((detection, grid_box))
    return pairs


def grid_box_to_image_box(grid_box: GridBox, grid_map: ImageGridMap) -> PixelBox:
    """Forward map L0 -> L1 (cell edges), e.g. to draw or crop a known box."""
    return PixelBox(
        x1=(grid_box.i0 - grid_map.trace_offset) / grid_map.traces_per_px,
        y1=(grid_box.k0 - grid_map.sample_offset) / grid_map.samples_per_px,
        x2=(grid_box.i1 + 1 - grid_map.trace_offset) / grid_map.traces_per_px,
        y2=(grid_box.k1 + 1 - grid_map.sample_offset) / grid_map.samples_per_px,
    )
