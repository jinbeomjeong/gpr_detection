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

M3-1 (spec_detection_image_v1 sections 5-6) adds the scaled-input path:

* tiles of a line L1 image -> boxes in continuous line L1 coordinates
  (:func:`detect_tiles`) -> duplicate merge across tiles (R2 + physics veto,
  band x arbitration, :func:`merge_line_boxes`) -> one outward rounding to L0
  (:func:`line_boxes_to_grid`); :func:`detect_line` runs all of it;
* the per-line depth call order of ADR-004 decision 7
  (:func:`estimate_line_depths`, :func:`estimate_survey_depths`).
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from gpr_detection.depth.band import band_exclude_traces, estimate_band_depths
from gpr_detection.depth.config import (
    DEFAULT_BAND_CONFIG,
    DEFAULT_DEPTH_CONFIG,
    BandConfig,
    DepthConfig,
)
from gpr_detection.depth.estimate import (
    DepthInputs,
    estimate_depth,
    prepare_depth_inputs,
)
from gpr_detection.depth.result import DepthResult
from gpr_detection.depth.velocity import VelocityEstimate, velocity_estimate_from
from gpr_detection.detect.band_metrics import group_layers
from gpr_detection.detect.base import (
    Detection,
    Detector,
    DetectorOutput,
    InputTransform,
    PixelBox,
)
from gpr_detection.io.classes import DetectionClass
from gpr_detection.io.radargram import GridBox, Radargram
from gpr_detection.preprocess.to_image import (
    DEFAULT_DETECTION_IMAGE_CONFIG,
    DetectionImage,
    DetectionImageConfig,
    ImageGridMap,
    LineTile,
    make_detection_image,
)

GRID_ROUNDING_TOLERANCE = 1e-6
"""Grid units ignored when rounding outward, so float noise such as 9.9999999
does not widen a box by a whole trace."""

TILE_TRUNCATED = "tile_truncated"
TILE_CLIPPED = "tile_clipped"
"""Depth-box flags set by the tile merge (spec_detection_image_v1 section 6)."""


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


# ======================================================= tile merge (section 5)


@dataclass(frozen=True)
class LineBox:
    """A detection in continuous line L1 coordinates (spec section 5-1).

    Attributes:
        class_id: ``DetectionClass`` value.
        class_name: Class name from the backend.
        score: Confidence.
        box: Line L1 box before any rounding.
        tile_index: Tile that produced the box.
        tile_truncated: Left or right edge within ``edge_px`` of an inner edge
            of its tile.
        tile_clipped: ``pipe_band`` trimmed by the x arbitration.
        n_tiles_seen: Distinct tiles in the merged group.
        merged_from_tiles: Sorted tile indices of the merged group.
    """

    class_id: int
    class_name: str
    score: float
    box: PixelBox
    tile_index: int
    tile_truncated: bool = False
    tile_clipped: bool = False
    n_tiles_seen: int = 1
    merged_from_tiles: tuple[int, ...] = ()

    @property
    def center_x(self) -> float:
        """Horizontal box centre in line L1 px."""
        return 0.5 * (self.box.x1 + self.box.x2)

    @property
    def tiles_seen(self) -> tuple[int, ...]:
        """``merged_from_tiles``, or the own tile for a box never merged."""
        return self.merged_from_tiles or (self.tile_index,)


@dataclass(frozen=True)
class MergeConfig:
    """Tile-merge settings ([CV] section 5 R2, [SP] section 5-3 V1/V2).

    Attributes:
        iou_min: R2 duplicate when IoU reaches this.
        iomin_truncated: R2 duplicate when one box is truncated and
            intersection / smaller area reaches this.
        edge_px: Truncation distance ``delta`` to an inner tile edge (L1 px).
        band_layer_overlap: Band boxes share a layer when their vertical
            overlap reaches this fraction of the smaller height.
        physics_veto: Apply V1/V2 to ``pipe_hyperbola`` pairs.
        veto_period_factor: V1: top difference above this many ``P_L1``.
        veto_min_center_px: V2: lower bound of the centre-distance limit.
    """

    iou_min: float = 0.7
    iomin_truncated: float = 0.8
    edge_px: float = 2.0
    band_layer_overlap: float = 0.5
    physics_veto: bool = True
    veto_period_factor: float = 1.0
    veto_min_center_px: float = 2.0


DEFAULT_MERGE_CONFIG = MergeConfig()


@dataclass(frozen=True, eq=False)
class LineDetections:
    """Merged detections of one line (spec section 5-2).

    Attributes:
        image: The line L1 image the detector saw.
        boxes: Merged boxes with their L0 box, in merge output order.
        tile_seam_box: Output boxes still ``tile_truncated`` after the merge.
    """

    image: DetectionImage
    boxes: tuple[tuple[LineBox, GridBox], ...]
    tile_seam_box: int


def _area(box: PixelBox) -> float:
    return (box.x2 - box.x1) * (box.y2 - box.y1)


def _intersection_area(a: PixelBox, b: PixelBox) -> float:
    width = min(a.x2, b.x2) - max(a.x1, b.x1)
    height = min(a.y2, b.y2) - max(a.y1, b.y1)
    if width <= 0 or height <= 0:
        return 0.0
    return width * height


def _iou(a: PixelBox, b: PixelBox) -> float:
    intersection = _intersection_area(a, b)
    union = _area(a) + _area(b) - intersection
    return intersection / union if union > 0 else 0.0


def _iomin(a: PixelBox, b: PixelBox) -> float:
    smaller = min(_area(a), _area(b))
    return _intersection_area(a, b) / smaller if smaller > 0 else 0.0


def _is_r2_duplicate(a: LineBox, b: LineBox, config: MergeConfig) -> bool:
    """[CV] R2: IoU >= 0.7, or IoMin >= 0.8 when one box is truncated."""
    if _iou(a.box, b.box) >= config.iou_min:
        return True
    is_truncated_pair = a.tile_truncated or b.tile_truncated
    return is_truncated_pair and _iomin(a.box, b.box) >= config.iomin_truncated


def _fresnel_radius_px(a: LineBox, b: LineBox, image: DetectionImage) -> float:
    """First Fresnel radius ``sqrt(lambda d / 2)`` in line L1 px (V2).

    ``d`` is the depth of the mean box top without subtracting the zero, so
    the radius errs on the large side.
    """
    depth_per_sample = image.depth_m_per_sample
    wavelength_m = 2.0 * depth_per_sample * image.period_samples
    mean_top_px = 0.5 * (a.box.y1 + b.box.y1)
    depth_m = mean_top_px * image.grid_map.samples_per_px * depth_per_sample
    radius_m = math.sqrt(wavelength_m * depth_m / 2.0)
    return radius_m / (image.grid_map.traces_per_px * image.dx_m)


def tile_box_to_line_box(box: PixelBox, tile: LineTile) -> PixelBox:
    """Tile L1 -> line L1: add the integer tile origin to ``x``."""
    return PixelBox(
        x1=box.x1 + tile.x0_px, y1=box.y1, x2=box.x2 + tile.x0_px, y2=box.y2
    )


def is_tile_truncated(box: PixelBox, tile: LineTile, edge_px: float) -> bool:
    """True when a side of a line L1 box lies within ``edge_px`` of an inner edge.

    Line ends and the padding edge of a short line are not inner edges. Top
    and bottom are not checked (no vertical tiling in M3-1).
    """
    tile_right_px = tile.x0_px + tile.content_width_px
    is_left_cut = tile.inner_left and box.x1 - tile.x0_px <= edge_px
    is_right_cut = tile.inner_right and tile_right_px - box.x2 <= edge_px
    return is_left_cut or is_right_cut


def detect_tiles(
    image: DetectionImage, detector: Detector, edge_px: float = 2.0
) -> list[LineBox]:
    """Run the detector on every tile; boxes in line L1 with truncation marks."""
    line_boxes = []
    for tile in image.tiles:
        output = detector.predict(image.tile_image(tile.index))
        for detection in output.detections:
            tile_box = model_box_to_image_box(detection.box, output.transform)
            line_box = tile_box_to_line_box(tile_box, tile)
            line_boxes.append(
                LineBox(
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    score=detection.score,
                    box=line_box,
                    tile_index=tile.index,
                    tile_truncated=is_tile_truncated(line_box, tile, edge_px),
                )
            )
    return line_boxes


def physics_veto(
    a: LineBox, b: LineBox, image: DetectionImage, config: MergeConfig
) -> bool:
    """Physics veto of a ``pipe_hyperbola`` pair ([SP] section 5-3 item 3-2).

    * V1: box tops differ by more than ``veto_period_factor * P_L1``
      (vertically separate reflectors).
    * V2: neither box is truncated and the centres are farther apart than
      ``max(r_F / (sigma_x dx), veto_min_center_px)``.

    Returns:
        True when the pair must not be merged.
    """
    top_difference_px = abs(a.box.y1 - b.box.y1)
    if top_difference_px > config.veto_period_factor * image.period_l1_px:
        return True
    if a.tile_truncated or b.tile_truncated:
        return False  # a truncated box has a shifted centre
    centre_limit_px = max(_fresnel_radius_px(a, b, image), config.veto_min_center_px)
    return abs(a.center_x - b.center_x) > centre_limit_px


def _representative_key(
    position: int, boxes: Sequence[LineBox], image: DetectionImage
) -> tuple[bool, bool, float, int, int]:
    """Sort key of [CV] 5-4 item 5 / [SP] 5-3 item 3-3 (smallest wins)."""
    box = boxes[position]
    centre_is_owned = image.tiles[box.tile_index].owns(box.center_x)
    return (
        box.tile_truncated,
        not centre_is_owned,
        -box.score,
        box.tile_index,
        position,
    )


def _merged_representative(
    representative: LineBox, group: Iterable[LineBox]
) -> LineBox:
    tiles = sorted({tile for box in group for tile in box.tiles_seen})
    return replace(
        representative, n_tiles_seen=len(tiles), merged_from_tiles=tuple(tiles)
    )


def _connected_components(
    members: Sequence[int], edges: dict[int, set[int]]
) -> list[list[int]]:
    """Components of the duplicate graph restricted to ``members`` (sorted)."""
    allowed = set(members)
    seen: set[int] = set()
    components = []
    for seed in members:
        if seed in seen:
            continue
        component = []
        stack = [seed]
        seen.add(seed)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in edges.get(current, ()):
                if neighbour in allowed and neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    return components


def _duplicate_edges(
    boxes: Sequence[LineBox],
    members: Sequence[int],
    image: DetectionImage,
    config: MergeConfig,
    use_veto: bool,
) -> dict[int, set[int]]:
    """R2 duplicate pairs from different tiles, minus vetoed pairs."""
    edges: dict[int, set[int]] = {position: set() for position in members}
    for n, i in enumerate(members):
        for j in members[n + 1 :]:
            a, b = boxes[i], boxes[j]
            if a.tile_index == b.tile_index or not _is_r2_duplicate(a, b, config):
                continue
            if use_veto and physics_veto(a, b, image, config):
                continue
            edges[i].add(j)
            edges[j].add(i)
    return edges


def _merge_same_class(
    boxes: Sequence[LineBox],
    members: Sequence[int],
    image: DetectionImage,
    config: MergeConfig,
    use_veto: bool,
) -> list[LineBox]:
    """R2 groups, one representative each, with the veto re-check (3-3, 3-4)."""
    edges = _duplicate_edges(boxes, members, image, config, use_veto)
    merged = []
    pending = [list(members)]
    while pending:
        for component in _connected_components(pending.pop(), edges):
            best = min(component, key=lambda p: _representative_key(p, boxes, image))
            vetoed = []
            if use_veto:
                vetoed = [
                    p
                    for p in component
                    if p != best and physics_veto(boxes[best], boxes[p], image, config)
                ]
            kept = [boxes[p] for p in component if p not in vetoed]
            merged.append(_merged_representative(boxes[best], kept))
            if vetoed:
                pending.append(vetoed)
    return merged


def _subtract_intervals(
    interval: tuple[float, float], removed: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Parts of ``interval`` outside every ``removed`` interval (non-empty only)."""
    pieces = [interval]
    for cut_low, cut_high in sorted(removed):
        remaining = []
        for low, high in pieces:
            if cut_high <= low or cut_low >= high:
                remaining.append((low, high))
                continue
            if cut_low > low:
                remaining.append((low, cut_low))
            if cut_high < high:
                remaining.append((cut_high, high))
        pieces = remaining
    return [(low, high) for low, high in pieces if high > low]


def _covered_by_other_owners(
    band: LineBox, layer: Sequence[LineBox], image: DetectionImage
) -> list[tuple[float, float]]:
    """x ranges in another tile's owner interval that the same tile's box covers."""
    covered = []
    for other in layer:
        if other.tile_index == band.tile_index:
            continue
        owner_low, owner_high = image.tiles[other.tile_index].owner_px
        low = max(owner_low, other.box.x1, band.box.x1)
        high = min(owner_high, other.box.x2, band.box.x2)
        if high > low:
            covered.append((low, high))
    return covered


def _clip_band(
    band: LineBox, layer: Sequence[LineBox], image: DetectionImage, edge_px: float
) -> list[LineBox]:
    """[SP] 5-3 item 4-2: drop x already covered by its owner tile; y unchanged."""
    removed = _covered_by_other_owners(band, layer, image)
    if not removed:
        return [replace(band, merged_from_tiles=band.tiles_seen)]
    tile = image.tiles[band.tile_index]
    fragments = []
    for low, high in _subtract_intervals((band.box.x1, band.box.x2), removed):
        box = PixelBox(x1=low, y1=band.box.y1, x2=high, y2=band.box.y2)
        fragments.append(
            replace(
                band,
                box=box,
                tile_clipped=True,
                # A cut edge sits mid-overlap, not at a seam; only an uncut
                # seam edge (neighbour missed the band) stays truncated.
                tile_truncated=is_tile_truncated(box, tile, edge_px),
                merged_from_tiles=band.tiles_seen,
            )
        )
    return fragments


def _arbitrate_bands(
    bands: Sequence[LineBox], image: DetectionImage, config: MergeConfig
) -> list[LineBox]:
    """``pipe_band``: per layer, trim trace ranges only (``k0``, ``k1`` kept)."""
    if not bands:
        return []
    corners = np.array([band.box.xyxy for band in bands])
    fragments = []
    for layer_indices in group_layers(corners, config.band_layer_overlap):
        layer = [bands[i] for i in layer_indices]
        for band in layer:
            fragments.extend(_clip_band(band, layer, image, config.edge_px))
    return fragments


def _output_order(box: LineBox) -> tuple[int, float, float, int]:
    return (box.class_id, box.box.x1, box.box.y1, box.tile_index)


def merge_line_boxes(
    boxes: Sequence[LineBox],
    image: DetectionImage,
    config: MergeConfig = DEFAULT_MERGE_CONFIG,
) -> tuple[LineBox, ...]:
    """Merge tile duplicates of one line ([SP] section 5-3).

    * ``pipe_hyperbola``, ``cavity``, ``manhole``: [CV] R2 groups of boxes from
      different tiles, one representative per group, coordinates and score of
      the representative unchanged. ``pipe_hyperbola`` also gets the physics
      veto V1/V2 and its re-check against the representative.
    * ``pipe_band``: x arbitration per layer; ``y1``/``y2`` never change.

    Returns:
        Merged boxes ordered by ``(class_id, x1, y1, tile_index)``.
    """
    boxes = list(boxes)
    merged: list[LineBox] = []
    for class_id in sorted({box.class_id for box in boxes}):
        members = [i for i, box in enumerate(boxes) if box.class_id == class_id]
        if class_id == DetectionClass.PIPE_BAND:
            merged.extend(_arbitrate_bands([boxes[i] for i in members], image, config))
            continue
        use_veto = config.physics_veto and class_id == DetectionClass.PIPE_HYPERBOLA
        merged.extend(_merge_same_class(boxes, members, image, config, use_veto))
    return tuple(sorted(merged, key=_output_order))


def line_boxes_to_grid(
    boxes: Sequence[LineBox], image: DetectionImage
) -> list[tuple[LineBox, GridBox]]:
    """Round merged line L1 boxes outward to L0, once; boxes off the grid drop."""
    pairs = []
    for line_box in boxes:
        grid_box = image_box_to_grid_box(line_box.box, image.grid_map)
        if grid_box is not None:
            pairs.append((line_box, grid_box))
    return pairs


def detect_line(
    radargram: Radargram,
    detector: Detector,
    image_config: DetectionImageConfig = DEFAULT_DETECTION_IMAGE_CONFIG,
    merge_config: MergeConfig = DEFAULT_MERGE_CONFIG,
) -> LineDetections:
    """Detect on one scaled radargram: image -> tiles -> merge -> L0."""
    image = make_detection_image(radargram, image_config)
    tile_boxes = detect_tiles(image, detector, merge_config.edge_px)
    merged = merge_line_boxes(tile_boxes, image, merge_config)
    pairs = tuple(line_boxes_to_grid(merged, image))
    tile_seam_box = sum(1 for line_box, _ in pairs if line_box.tile_truncated)
    return LineDetections(image=image, boxes=pairs, tile_seam_box=tile_seam_box)


# ================================================= per-line depth (section 6)


@dataclass(frozen=True)
class DepthBox:
    """One L0 box handed to the depth branch (spec section 6-1).

    Attributes:
        class_id: ``DetectionClass`` value.
        grid_box: Box on the radargram grid (L0).
        flags: ``"tile_truncated"``, ``"tile_clipped"``.
    """

    class_id: int
    grid_box: GridBox
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class LineDepths:
    """Depth results of one line (spec section 6-1).

    Attributes:
        line_id: Line id used for the velocity pool.
        inputs: Preprocessing and the single zero of the line.
        exclude_traces: Band traces left out of the zero (bool ``[N]``) or None.
        hyperbola: ``(box, result)`` per ``pipe_hyperbola`` box, input order.
        pool: Velocities of this line's hyperbolas.
        band: ``estimate_band_depths`` results in its order.
        skipped: ``cavity`` and ``manhole`` boxes (no depth, ADR-002 decision 8).
    """

    line_id: str
    inputs: DepthInputs
    exclude_traces: np.ndarray | None
    hyperbola: tuple[tuple[DepthBox, DepthResult], ...]
    pool: tuple[VelocityEstimate, ...]
    band: tuple[DepthResult, ...]
    skipped: tuple[DepthBox, ...]


@dataclass(frozen=True)
class _HyperbolaStage:
    """Steps 1-4 of one line, kept for the survey-wide band step."""

    radargram: Radargram
    line_id: str
    inputs: DepthInputs
    exclude_traces: np.ndarray | None
    hyperbola: tuple[tuple[DepthBox, DepthResult], ...]
    pool: tuple[VelocityEstimate, ...]
    band_boxes: tuple[DepthBox, ...]
    skipped: tuple[DepthBox, ...]


def depth_boxes_from(detections: LineDetections) -> list[DepthBox]:
    """Depth boxes of merged detections, with the tile-merge flags."""
    depth_boxes = []
    for line_box, grid_box in detections.boxes:
        flags = []
        if line_box.tile_truncated:
            flags.append(TILE_TRUNCATED)
        if line_box.tile_clipped:
            flags.append(TILE_CLIPPED)
        depth_boxes.append(DepthBox(line_box.class_id, grid_box, tuple(flags)))
    return depth_boxes


def _require_same_pick_mode(depth_config: DepthConfig, band_config: BandConfig) -> None:
    if depth_config.pick_mode != band_config.pick_mode:
        raise ValueError(
            f"pick_mode differs: depth {depth_config.pick_mode!r}, "
            f"band {band_config.pick_mode!r} (zero and apex need one phase)"
        )


def _with_flag(result: DepthResult, flag: str) -> DepthResult:
    if flag in result.flags:
        return result
    return replace(result, flags=(*result.flags, flag))


def _split_by_class(
    boxes: Sequence[DepthBox],
) -> tuple[list[DepthBox], list[DepthBox], list[DepthBox]]:
    """Step 1: ``(hyperbola, band, skipped)``, each in input order."""
    hyperbola, band, skipped = [], [], []
    for box in boxes:
        if box.class_id == DetectionClass.PIPE_HYPERBOLA:
            hyperbola.append(box)
        elif box.class_id == DetectionClass.PIPE_BAND:
            band.append(box)
        else:
            skipped.append(box)
    return hyperbola, band, skipped


def _hyperbola_stage(
    radargram: Radargram,
    boxes: Sequence[DepthBox],
    depth_config: DepthConfig,
    band_config: BandConfig,
    line_id: str,
    survey_key: str | None,
) -> _HyperbolaStage:
    """Steps 1-4: classes, one zero (band traces excluded), hyperbolas, pool."""
    hyperbola_boxes, band_boxes, skipped = _split_by_class(boxes)
    exclude_traces = None
    if band_boxes:
        exclude_traces = band_exclude_traces(
            [box.grid_box for box in band_boxes],
            radargram.n_traces,
            radargram.dx_m,
            band_config,
        )
    inputs = prepare_depth_inputs(
        radargram, depth_config, exclude_traces=exclude_traces
    )
    hyperbola = []
    for box in hyperbola_boxes:
        result = estimate_depth(radargram, box.grid_box, depth_config, inputs=inputs)
        if TILE_TRUNCATED in box.flags:
            result = _with_flag(result, TILE_TRUNCATED)
        hyperbola.append((box, result))
    pool = []
    for j, (_, result) in enumerate(hyperbola):
        estimate = velocity_estimate_from(
            result, line_id=line_id, source_index=j, survey_key=survey_key
        )
        if estimate is not None:
            pool.append(estimate)
    return _HyperbolaStage(
        radargram=radargram,
        line_id=line_id,
        inputs=inputs,
        exclude_traces=exclude_traces,
        hyperbola=tuple(hyperbola),
        pool=tuple(pool),
        band_boxes=tuple(band_boxes),
        skipped=tuple(skipped),
    )


def _band_stage(
    stage: _HyperbolaStage,
    band_config: BandConfig,
    survey_pool: Sequence[VelocityEstimate],
) -> LineDepths:
    """Steps 5-6: band depths with the line pool, then the result record."""
    band_results: list[DepthResult] = []
    if stage.band_boxes:
        band_results = estimate_band_depths(
            stage.radargram,
            [box.grid_box for box in stage.band_boxes],
            stage.inputs.zero,
            stage.pool,
            survey_pool,
            band_config,
            branch=stage.inputs.branch,
        )
    flagged = []
    for result in band_results:
        members = () if result.band is None else result.band.member_indices
        if any(TILE_CLIPPED in stage.band_boxes[m].flags for m in members):
            result = _with_flag(result, TILE_CLIPPED)
        flagged.append(result)
    return LineDepths(
        line_id=stage.line_id,
        inputs=stage.inputs,
        exclude_traces=stage.exclude_traces,
        hyperbola=stage.hyperbola,
        pool=stage.pool,
        band=tuple(flagged),
        skipped=stage.skipped,
    )


def estimate_line_depths(
    radargram: Radargram,
    boxes: Sequence[DepthBox],
    depth_config: DepthConfig = DEFAULT_DEPTH_CONFIG,
    band_config: BandConfig = DEFAULT_BAND_CONFIG,
    *,
    line_id: str | None = None,
    survey_key: str | None = None,
    survey_pool: Sequence[VelocityEstimate] = (),
) -> LineDepths:
    """Depths of every box of one line in the ADR-004 decision 7 order.

    ``band_exclude_traces`` -> ``prepare_depth_inputs`` (one zero) ->
    ``estimate_depth`` per hyperbola -> ``velocity_estimate_from`` ->
    ``estimate_band_depths``. ``cavity`` and ``manhole`` get no depth.

    Args:
        radargram: The line.
        boxes: L0 boxes with class and tile-merge flags.
        depth_config: Hyperbola settings.
        band_config: Band settings (same pick mode).
        line_id: Pool line id; defaults to ``radargram.line_id`` (the band
            only borrows pool velocities whose line id equals that one).
        survey_key: Survey unit written into the pool entries.
        survey_pool: Velocities of the survey unit (band rule 4).

    Raises:
        ValueError: If the two configs use different pick modes.
    """
    _require_same_pick_mode(depth_config, band_config)
    stage = _hyperbola_stage(
        radargram,
        boxes,
        depth_config,
        band_config,
        radargram.line_id if line_id is None else line_id,
        survey_key,
    )
    return _band_stage(stage, band_config, survey_pool)


def estimate_survey_depths(
    lines: Sequence[tuple[Radargram, Sequence[DepthBox]]],
    depth_config: DepthConfig = DEFAULT_DEPTH_CONFIG,
    band_config: BandConfig = DEFAULT_BAND_CONFIG,
    *,
    survey_key: str,
) -> list[LineDepths]:
    """Two-stage depths of a ground-coupled survey unit (spec section 6-2).

    Stage 1 runs steps 0-4 on every line; stage 2 runs the band step of every
    line with the survey pool = all line pools in line order. The zero of a
    line is computed once.

    Raises:
        ValueError: If the two configs use different pick modes.
    """
    _require_same_pick_mode(depth_config, band_config)
    stages = [
        _hyperbola_stage(
            radargram, boxes, depth_config, band_config, radargram.line_id, survey_key
        )
        for radargram, boxes in lines
    ]
    survey_pool = [estimate for stage in stages for estimate in stage.pool]
    return [_band_stage(stage, band_config, survey_pool) for stage in stages]
