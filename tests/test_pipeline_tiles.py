"""Tile merge and exact round trip (spec_detection_image_v1 v0.3.4 section 8-2).

IDs follow [SP] (``docs/processing/spec_detection_image_v1.md``); the [CV] IDs
(``docs/experiments/m31_detection_input_tiling.md`` v0.2 C1-C5) are given in
each docstring. Merge tests build ``LineBox`` lists directly on a 4TU-like
fixture image (sigma_x 1.5445, sigma_y 6.32, P_line 25.61 samples -> P_L1 4.05
px, delta_z 0.004897, dx 0.02, W1 1000, O fixed at 128 -> 7 tiles).
"""

import itertools
import math

import numpy as np
import pytest

from gpr_detection import pipeline
from gpr_detection.depth.band import candidate_pairs, estimate_band_depths
from gpr_detection.detect.base import (
    Detection,
    DetectorOutput,
    PixelBox,
    letterbox,
)
from gpr_detection.io.radargram import GridBox
from gpr_detection.pipeline import (
    LineBox,
    MergeConfig,
    detect_line,
    detect_tiles,
    grid_box_to_image_box,
    image_box_to_grid_box,
    line_boxes_to_grid,
    merge_line_boxes,
    model_box_to_image_box,
)
from gpr_detection.preprocess.timezero import given_zero
from gpr_detection.preprocess.to_image import (
    M3_2_ASSUMED,
    DetectionImage,
    DetectionImageConfig,
    ImageGridMap,
    line_scale,
    overlap_for,
    tile_layout,
)
from synthetic_bscan import T_SRC_NS, BScanSpec, make_radargram

SEED = 20261113
TILE_PX = 256
CLASS_NAMES = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")
HYPERBOLA, BAND, CAVITY = 0, 1, 2

FIXTURE_TRACES_PER_PX = 1.5445
FIXTURE_SAMPLES_PER_PX = 6.32
FIXTURE_PERIOD_SAMPLES = 25.61
FIXTURE_DEPTH_PER_SAMPLE = 0.004897
FIXTURE_DX_M = 0.02
FIXTURE_DT_NS = 0.098
FIXTURE_N_SAMPLES = 512
FIXTURE_OVERLAP_PX = 128

V_EPS9 = 0.299792458 / 3.0
ROUND_TRIP_CASES = {
    "4tu_01.1_path1": (512, 304, 0.02, 2.51 / 0.097, 0.097 * V_EPS9 / 2, 4.06),
    "4tu_09.1_path2": (512, 12186, 0.02, 2.51 / 0.195, 0.195 * V_EPS9 / 2, 4.06),
    "4tu_09.1_path2_p8": (512, 12186, 0.02, 2.51 / 0.195, 0.195 * V_EPS9 / 2, 8.0),
    "gz_1030_30ns": (512, 710, 0.024, 2.93 / (30 / 512), (30 / 512) * 0.10 / 2, 4.06),
    "gprsd_cementroad2_1": (321, 9411, 0.03, 51.2, 0.012, 4.06),
    "4tu_03.9_path5": (512, 9, 0.02, 2.51 / 0.098, 0.098 * V_EPS9 / 2, 4.06),
}


def fixture_image(
    width_px: int = 1000, overlap_px: int = FIXTURE_OVERLAP_PX
) -> DetectionImage:
    """4TU-like line L1 image of the T8 fixture (no radargram behind it)."""
    grid_map = ImageGridMap(
        trace_offset=0,
        sample_offset=0,
        traces_per_px=FIXTURE_TRACES_PER_PX,
        samples_per_px=FIXTURE_SAMPLES_PER_PX,
        n_traces=round(width_px * FIXTURE_TRACES_PER_PX),
        n_samples=FIXTURE_N_SAMPLES,
    )
    rng = np.random.default_rng(SEED)
    return DetectionImage(
        line_image=rng.integers(0, 256, (TILE_PX, width_px), dtype=np.uint8),
        grid_map=grid_map,
        tiles=tile_layout(width_px, TILE_PX, overlap_px),
        content_height_px=81,
        period_samples=FIXTURE_PERIOD_SAMPLES,
        period_source="measured",
        depth_m_per_sample=FIXTURE_DEPTH_PER_SAMPLE,
        dx_m=FIXTURE_DX_M,
        v_scale_m_per_ns=2 * FIXTURE_DEPTH_PER_SAMPLE / FIXTURE_DT_NS,
        v_scale_source="eps_r_meta",
        p_target_px=4.06,
        overlap_px=overlap_px,
        overlap_phys_px=65,
        compression=1.0,
        flags=(),
        assumed_params=M3_2_ASSUMED,
    )


def line_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    tile: int,
    *,
    class_id: int = HYPERBOLA,
    score: float = 0.9,
    truncated: bool = False,
) -> LineBox:
    return LineBox(
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        score=score,
        box=PixelBox(x1, y1, x2, y2),
        tile_index=tile,
        tile_truncated=truncated,
    )


class VisiblePartDetector:
    """Fake backend: for each tile, the part of known line boxes inside it.

    Tiles are identified by call order (``detect_tiles`` walks them left to
    right); the received image is checked against ``tile_image``.
    """

    def __init__(
        self,
        image: DetectionImage,
        boxes: list[tuple[int, float, PixelBox]],
        only_tiles: set[int] | None = None,
    ) -> None:
        self._image = image
        self._boxes = boxes
        self._only_tiles = only_tiles
        self._calls = 0

    @property
    def class_names(self) -> tuple[str, ...]:
        return CLASS_NAMES

    def predict(self, image: np.ndarray) -> DetectorOutput:
        tile = self._image.tiles[self._calls]
        self._calls += 1
        np.testing.assert_array_equal(image, self._image.tile_image(tile.index))
        _, transform = letterbox(image, TILE_PX)
        detections = []
        if self._only_tiles is None or tile.index in self._only_tiles:
            right = tile.x0_px + tile.content_width_px
            for class_id, score, box in self._boxes:
                low, high = max(box.x1, tile.x0_px), min(box.x2, right)
                if high <= low:
                    continue
                tile_box = PixelBox(low - tile.x0_px, box.y1, high - tile.x0_px, box.y2)
                detections.append(
                    Detection(
                        class_id,
                        CLASS_NAMES[class_id],
                        score,
                        transform.apply(tile_box),
                    )
                )
        return DetectorOutput(tuple(detections), transform)


def _use_image(monkeypatch: pytest.MonkeyPatch, image: DetectionImage) -> None:
    monkeypatch.setattr(pipeline, "make_detection_image", lambda *a, **k: image)


# ------------------------------------------------------------------ T1 (C2)


@pytest.mark.parametrize("case", sorted(ROUND_TRIP_CASES))
@pytest.mark.parametrize("overlap_cv", [128, 192])
def test_t1_exact_round_trip_through_every_containing_tile(
    case: str, overlap_cv: int
) -> None:
    """T1 ([CV] C2, QA-8): L0 -> line L1 -> tile L1 -> L2 -> back gives the box."""
    n_s, n_traces, dx_m, period, depth_per_sample, p_star = ROUND_TRIP_CASES[case]
    scale = line_scale(n_s, n_traces, dx_m, period, depth_per_sample, p_star)
    config = DetectionImageConfig(p_target_px=p_star, overlap_px=overlap_cv)
    overlap, _, _ = overlap_for(scale.traces_per_px, dx_m, p_star, config)
    tiles = tile_layout(scale.width_px, TILE_PX, overlap)
    grid_map = ImageGridMap(
        0, 0, scale.traces_per_px, scale.samples_per_px, n_traces, n_s
    )
    _, identity = letterbox(np.zeros((TILE_PX, TILE_PX), np.uint8), TILE_PX)

    rng = np.random.default_rng(SEED)
    max_width_traces = max(1, math.floor(overlap * scale.traces_per_px))
    n_crossing = 0
    for _ in range(200):
        width = int(rng.integers(1, min(max_width_traces, n_traces) + 1))
        i0 = int(rng.integers(0, n_traces - width + 1))
        k0 = int(rng.integers(0, n_s))
        k1 = int(rng.integers(k0, n_s))
        original = GridBox(i0, i0 + width - 1, k0, k1)
        line_l1 = grid_box_to_image_box(original, grid_map)
        containing = [
            tile
            for tile in tiles
            if tile.x0_px - 1e-9 <= line_l1.x1
            and line_l1.x2 <= tile.x0_px + tile.content_width_px + 1e-9
        ]
        assert containing, (case, original)
        n_crossing += len(containing) > 1
        for tile in containing:
            tile_l1 = PixelBox(
                line_l1.x1 - tile.x0_px, line_l1.y1, line_l1.x2 - tile.x0_px, line_l1.y2
            )
            model = identity.apply(tile_l1)
            back = model_box_to_image_box(model, identity)
            assert (
                max(abs(a - b) for a, b in zip(back.xyxy, tile_l1.xyxy, strict=True))
                <= 0.5
            )
            back_line = pipeline.tile_box_to_line_box(back, tile)
            assert image_box_to_grid_box(back_line, grid_map) == original
    if len(tiles) > 1:
        assert n_crossing > 0


# ----------------------------------------------------------- T8a-T8e (C3a-e)


def test_t8a_same_hyperbola_from_two_tiles_keeps_owner_box() -> None:
    """T8a ([CV] C3a): jittered copies -> one box, the owner tile's."""
    image = fixture_image()
    owner_copy = line_box(180, 30, 240, 60, tile=1, score=0.80)
    other_copy = line_box(181, 31, 241, 61, tile=0, score=0.95)
    (merged,) = merge_line_boxes([other_copy, owner_copy], image)
    assert merged.box == owner_copy.box
    assert merged.tile_index == 1
    assert merged.merged_from_tiles == (0, 1)


@pytest.mark.parametrize("visible", [0.3, 0.5, 0.7, 0.9])
def test_t8b_full_and_truncated_merge_to_full(visible: float) -> None:
    """T8b ([CV] C3b): full + truncated -> the full box, not truncated."""
    image = fixture_image()
    width = 60.0
    seam = 256.0
    x1 = seam - visible * width
    full = line_box(x1, 30, x1 + width, 60, tile=1, score=0.5)
    cut = line_box(x1, 30, seam, 60, tile=0, score=0.9, truncated=True)
    (merged,) = merge_line_boxes([cut, full], image)
    assert merged.box == full.box
    assert merged.tile_truncated is False
    assert merged.n_tiles_seen == 2


def test_t8c_unpaired_truncated_box_is_kept_and_counted(monkeypatch) -> None:
    """T8c ([CV] C3c): only the truncated part -> kept, tile_seam_box 1."""
    image = fixture_image()
    _use_image(monkeypatch, image)
    detector = VisiblePartDetector(
        image, [(HYPERBOLA, 0.9, PixelBox(230, 30, 290, 60))], only_tiles={0}
    )
    detections = detect_line(object(), detector)
    ((merged, _),) = detections.boxes
    assert merged.tile_truncated is True
    assert merged.box == PixelBox(230, 30, 256, 60)
    assert detections.tile_seam_box == 1


def test_t8d_two_close_pipes_stay_two() -> None:
    """T8d ([CV] C3d): width-60 boxes 20 px apart (IoU 0.5) -> 2."""
    image = fixture_image()
    boxes = [line_box(150, 30, 210, 60, tile=0), line_box(170, 30, 230, 60, tile=1)]
    assert len(merge_line_boxes(boxes, image)) == 2


def test_t8e_different_classes_are_never_merged() -> None:
    """T8e ([CV] C3e): overlapping pipe_hyperbola + cavity -> both kept."""
    image = fixture_image()
    boxes = [
        line_box(150, 30, 210, 60, tile=0),
        line_box(150, 30, 210, 60, tile=1, class_id=CAVITY),
    ]
    merged = merge_line_boxes(boxes, image)
    assert sorted(box.class_id for box in merged) == [HYPERBOLA, CAVITY]


# ------------------------------------------------------------ T8f-T8i (C3j)


def test_t8f_v1_vetoes_vertically_separate_reflectors() -> None:
    """T8f ([CV] C3j-1): top difference 1.5 P_L1 (IoU 0.82) -> 2; R2 alone -> 1."""
    image = fixture_image()
    shift = 1.5 * image.period_l1_px
    boxes = [
        line_box(140, 30, 190, 90, tile=0),
        line_box(140, 30 + shift, 190, 90 + shift, tile=1),
    ]
    assert pipeline._iou(boxes[0].box, boxes[1].box) == pytest.approx(0.82, abs=0.01)
    assert len(merge_line_boxes(boxes, image)) == 2
    no_veto = MergeConfig(physics_veto=False)
    assert len(merge_line_boxes(boxes, image, no_veto)) == 1

    short = [
        line_box(140, 30, 190, 55, tile=0),
        line_box(140, 30 + shift, 190, 55 + shift, tile=1),
    ]
    assert pipeline._iou(short[0].box, short[1].box) == pytest.approx(0.61, abs=0.01)
    assert len(merge_line_boxes(short, image)) == 2
    assert len(merge_line_boxes(short, image, no_veto)) == 2


def test_t8g_v1_boundary_half_period_merges() -> None:
    """T8g ([CV] C3j-2): top difference 0.5 P_L1 = 2.0 px -> R2, one box."""
    image = fixture_image()
    boxes = [
        line_box(140, 30, 190, 90, tile=0),
        line_box(140, 32.0, 190, 92.0, tile=1),
    ]
    assert len(merge_line_boxes(boxes, image)) == 1


@pytest.mark.parametrize(("gap_px", "expected"), [(20.0, 2), (8.0, 1)])
def test_t8h_v2_vetoes_pairs_beyond_fresnel_radius(gap_px: float, expected: int):
    """T8h ([CV] C3j-3): y1 40 px -> r_F ~ 12.7 px; gap 20 -> 2, gap 8 -> 1."""
    image = fixture_image()
    a = line_box(130, 40, 250, 80, tile=0)
    b = line_box(130 + gap_px, 40, 250 + gap_px, 80, tile=1)
    assert pipeline._fresnel_radius_px(a, b, image) == pytest.approx(12.7, abs=0.1)
    assert pipeline._iou(a.box, b.box) >= 0.7
    assert len(merge_line_boxes([a, b], image)) == expected


def test_t8i_chain_is_split_by_the_veto_recheck() -> None:
    """T8i ([CV] C3j-4): A-B and B-C merge, A-C vetoed -> A(+B) and C.

    Same x for all three boxes, so B (tile 1) cannot have its centre in its
    own owner interval as the spec text says; the expected outcome does not
    depend on it (A wins on score among the owned boxes).
    """
    image = fixture_image()
    a = line_box(140, 40.0, 190, 100.0, tile=0, score=0.9)
    b = line_box(140, 42.5, 190, 102.5, tile=1, score=0.8)
    c = line_box(140, 45.0, 190, 105.0, tile=0, score=0.7)
    merged = merge_line_boxes([a, b, c], image)
    assert len(merged) == 2
    first, second = sorted(merged, key=lambda box: box.box.y1)
    assert first.box == a.box
    assert first.n_tiles_seen == 2
    assert first.merged_from_tiles == (0, 1)
    assert second.box == c.box
    assert second.n_tiles_seen == 1


# ----------------------------------------------------------------- T8j (C3i)


def test_t8j_rounding_happens_once_after_the_merge() -> None:
    """T8j ([CV] C3i): L0 box = one outward rounding of the representative."""
    image = fixture_image()
    rng = np.random.default_rng(SEED)
    boxes = []
    for _ in range(30):
        x1 = float(rng.uniform(130, 190))
        y1 = float(rng.uniform(5, 40))
        width, height = float(rng.uniform(20, 60)), float(rng.uniform(10, 35))
        boxes.append(line_box(x1, y1, x1 + width, y1 + height, tile=0))
        boxes.append(line_box(x1 + 0.3, y1 + 0.2, x1 + width + 0.3, y1 + height, 1))
    sigma_x = image.grid_map.traces_per_px
    sigma_y = image.grid_map.samples_per_px
    for merged, grid_box in line_boxes_to_grid(merge_line_boxes(boxes, image), image):
        assert grid_box == image_box_to_grid_box(merged.box, image.grid_map)
        assert grid_box.i0 <= merged.box.x1 * sigma_x + 1e-6
        assert grid_box.i1 + 1 >= merged.box.x2 * sigma_x - 1e-6
        assert grid_box.k0 <= merged.box.y1 * sigma_y + 1e-6
        assert grid_box.k1 + 1 >= merged.box.y2 * sigma_y - 1e-6


# ------------------------------------------------------------- T8k, T8l (C3f-h)

BAND_X_PX = (100.0, 620.0)
BAND_TAU_NS = 12.0
SECOND_BAND_DELAY_NS = 6.0
BAND_TOP_JITTER_PX = {0: 0.0, 1: -2.0, 2: -1.0, 3: 1.0, 4: 2.0}
"""Tops differ by 0/+-1/+-2 px; neighbouring tiles differ by 1-2 px ([SP] 5-3 4-4)."""
BAND_PERIOD_NS = 2.5


def _band_y_px(tau_ns: float) -> tuple[float, float]:
    """Band box rows: 0.4 period above to 0.8 period below the band (L1 px)."""
    px_per_ns = 1.0 / (FIXTURE_DT_NS * FIXTURE_SAMPLES_PER_PX)
    record_ns = T_SRC_NS + tau_ns
    return (
        (record_ns - 0.4 * BAND_PERIOD_NS) * px_per_ns,
        (record_ns + 0.8 * BAND_PERIOD_NS) * px_per_ns,
    )


def _tiled_band_boxes(image: DetectionImage) -> list[LineBox]:
    """Two bands seen by tiles 0-4, tops jittered per tile."""
    boxes = []
    for tau_ns in (BAND_TAU_NS, BAND_TAU_NS + SECOND_BAND_DELAY_NS):
        top, bottom = _band_y_px(tau_ns)
        for tile_index, jitter in BAND_TOP_JITTER_PX.items():
            tile = image.tiles[tile_index]
            low = max(BAND_X_PX[0], tile.x0_px)
            high = min(BAND_X_PX[1], tile.x0_px + tile.content_width_px)
            truncated = pipeline.is_tile_truncated(
                PixelBox(low, top, high, bottom), tile, 2.0
            )
            boxes.append(
                line_box(
                    low,
                    top + jitter,
                    high,
                    bottom + jitter,
                    tile_index,
                    class_id=BAND,
                    truncated=truncated,
                )
            )
    return boxes


def _band_radargram(image: DetectionImage):
    """Synthetic line on the fixture grid with the two flat bands (amplitude 1)."""
    n_traces = image.grid_map.n_traces
    first = math.floor(BAND_X_PX[0] * FIXTURE_TRACES_PER_PX)
    last = math.ceil(BAND_X_PX[1] * FIXTURE_TRACES_PER_PX)
    amplitude = np.zeros(n_traces)
    amplitude[first:last] = 1.0
    spec = BScanSpec(
        n_traces=n_traces,
        n_samples=FIXTURE_N_SAMPLES,
        dt_ns=FIXTURE_DT_NS,
        dx_m=FIXTURE_DX_M,
        pipe_times_ns=np.full(n_traces, 1000.0),
    )
    spec.extra_events = [
        (np.full(n_traces, BAND_TAU_NS), amplitude),
        (np.full(n_traces, BAND_TAU_NS + SECOND_BAND_DELAY_NS), amplitude),
    ]
    return make_radargram(spec)


def test_t8k_band_fragments_tile_the_band_and_keep_rows() -> None:
    """T8k ([CV] C3f): 2 layers; touching fragments, union kept, y unchanged."""
    image = fixture_image()
    tiled = _tiled_band_boxes(image)
    merged = merge_line_boxes(tiled, image)

    input_rows = {(box.tile_index, box.box.y1, box.box.y2) for box in tiled}
    assert all((b.tile_index, b.box.y1, b.box.y2) in input_rows for b in merged)
    layers = [
        [b for b in merged if b.box.y1 < _band_y_px(BAND_TAU_NS)[1]],
        [b for b in merged if b.box.y1 >= _band_y_px(BAND_TAU_NS)[1]],
    ]
    for layer in layers:
        spans = sorted((b.box.x1, b.box.x2) for b in layer)
        for (_, right), (left, _) in itertools.pairwise(spans):
            assert left == right  # touching, no overlap, no gap
        assert spans[0][0] == BAND_X_PX[0]
        assert spans[-1][1] == BAND_X_PX[1]
        assert all(b.tile_clipped for b in layer)
        assert not any(b.tile_truncated for b in layer)
    assert sum(len(layer) for layer in layers) == len(merged)

    # Depth: the first layer's fragments form one band result at 0.6 m.
    radargram = _band_radargram(image)
    pairs = line_boxes_to_grid(merged, image)
    grid_boxes = [grid_box for _, grid_box in pairs]
    first_layer = {i for i, (b, _) in enumerate(pairs) if b in layers[0]}
    zero = given_zero(T_SRC_NS, FIXTURE_DT_NS, pick_mode="envelope")
    results = estimate_band_depths(radargram, grid_boxes, zero)
    (first_result,) = [r for r in results if set(r.band.member_indices) & first_layer]
    assert set(first_result.band.member_indices) == first_layer
    true_depth_m = V_EPS9 * BAND_TAU_NS / 2
    assert first_result.depth_top_m == pytest.approx(true_depth_m, abs=0.02)


T8K2_TOP_PX = 20.0
T8K2_CASES = [
    # (box height px, top difference px, grouped)
    (16.0, 3.0, True),
    (16.0, 6.0, True),
    (16.0, 9.0, False),
    (5.0, 2.0, True),
    (5.0, 3.0, False),
]


def _single_band_radargram(
    image: DetectionImage, record_ns: float, x_px: tuple[float, float]
):
    """Fixture-grid line with one flat band (amplitude 1) at ``record_ns``.

    The band only spans ``x_px`` so the masked background (traces outside the
    band boxes) does not remove it.
    """
    n_traces = image.grid_map.n_traces
    amplitude = np.zeros(n_traces)
    first = math.floor(x_px[0] * FIXTURE_TRACES_PER_PX)
    amplitude[first : math.ceil(x_px[1] * FIXTURE_TRACES_PER_PX)] = 1.0
    spec = BScanSpec(
        n_traces=n_traces,
        n_samples=FIXTURE_N_SAMPLES,
        dt_ns=FIXTURE_DT_NS,
        dx_m=FIXTURE_DX_M,
        pipe_times_ns=np.full(n_traces, 1000.0),
    )
    spec.extra_events = [(np.full(n_traces, record_ns - T_SRC_NS), amplitude)]
    return make_radargram(spec)


@pytest.mark.parametrize(("height_px", "top_difference_px", "grouped"), T8K2_CASES)
def test_t8k2_band_grouping_limit_is_half_the_box_height(
    height_px: float, top_difference_px: float, grouped: bool
) -> None:
    """T8k-2 (v0.3.5): touching fragments group while the top difference is
    at most half the box height; y1/y2 always stay as detected."""
    image = fixture_image()
    upper = line_box(100, T8K2_TOP_PX, 240, T8K2_TOP_PX + height_px, 0, class_id=BAND)
    lower_top = T8K2_TOP_PX + top_difference_px
    lower = line_box(150, lower_top, 300, lower_top + height_px, 1, class_id=BAND)

    merged = merge_line_boxes([upper, lower], image)

    assert len(merged) == 2
    rows = {(b.tile_index, b.box.y1, b.box.y2) for b in merged}
    assert rows == {
        (0, upper.box.y1, upper.box.y2),
        (1, lower.box.y1, lower.box.y2),
    }
    left, right = sorted(merged, key=lambda b: b.box.x1)
    if top_difference_px <= height_px / 2:
        assert left.box.x2 == right.box.x1  # one layer: arbitrated, touching
    else:
        assert (left.box, right.box) == (upper.box, lower.box)  # two layers

    # The band lies inside both boxes' rows: halfway through their overlap.
    event_px = 0.5 * (lower_top + T8K2_TOP_PX + height_px)
    record_ns = event_px * FIXTURE_SAMPLES_PER_PX * FIXTURE_DT_NS
    radargram = _single_band_radargram(image, record_ns, (upper.box.x1, lower.box.x2))
    grid_boxes = [grid for _, grid in line_boxes_to_grid(merged, image)]
    is_candidate = bool(candidate_pairs(grid_boxes, FIXTURE_DX_M))
    zero = given_zero(T_SRC_NS, FIXTURE_DT_NS, pick_mode="envelope")
    results = estimate_band_depths(radargram, grid_boxes, zero)

    assert is_candidate == grouped
    assert len(results) == (1 if grouped else 2)


def test_t8l_one_sided_band_order_and_idempotence() -> None:
    """T8l ([CV] C3g, C3h): one tile only keeps its length; order; merge twice."""
    image = fixture_image()
    top, bottom = _band_y_px(BAND_TAU_NS)
    alone = line_box(100, top, 240, bottom, tile=0, class_id=BAND)
    (kept,) = merge_line_boxes([alone], image)
    assert kept.box == alone.box
    assert kept.tile_clipped is False

    boxes = [
        *_tiled_band_boxes(image),
        line_box(180, 30, 240, 60, tile=1),
        line_box(181, 31, 241, 61, tile=0),
    ]
    reference = merge_line_boxes(boxes, image)
    rng = np.random.default_rng(SEED)
    for _ in range(5):
        shuffled = [boxes[i] for i in rng.permutation(len(boxes))]
        assert merge_line_boxes(shuffled, image) == reference
    assert merge_line_boxes(reference, image) == reference


# ------------------------------------------------------------------ T8m (C5)


def _seam_rectangles(image: DetectionImage, count: int) -> list[PixelBox]:
    """Non-overlapping rectangles, width <= 120 px, in three row bands."""
    rng = np.random.default_rng(SEED)
    rows = [(2.0, 22.0), (28.0, 50.0), (56.0, 78.0)]
    rectangles = []
    for top, bottom in rows:
        x = float(rng.uniform(0, 20))
        while True:
            width = float(rng.uniform(10, 120))
            if x + width > image.width_px:
                break
            y1 = float(rng.uniform(top, top + 5))
            y2 = float(rng.uniform(bottom - 5, bottom))
            rectangles.append(PixelBox(x, y1, x + width, y2))
            x += width + float(rng.uniform(5, 60))
    order = rng.permutation(len(rectangles))[:count]
    return [rectangles[i] for i in sorted(order)]


def test_t8m_end_to_end_fake_detector(monkeypatch) -> None:
    """T8m ([CV] C5): 50 rectangles -> exactly 50 boxes, tile_seam_box 0."""
    image = fixture_image(width_px=2250)
    rectangles = _seam_rectangles(image, 50)
    seams = {seam for tile in image.tiles for seam in tile.seams_px}
    assert any(r.x1 < seam < r.x2 for r in rectangles for seam in seams)
    _use_image(monkeypatch, image)
    detector = VisiblePartDetector(image, [(HYPERBOLA, 0.9, r) for r in rectangles])

    detections = detect_line(object(), detector)

    assert detections.tile_seam_box == 0
    assert len(detections.boxes) == len(rectangles)
    got = sorted((b.box.x1, grid) for b, grid in detections.boxes)
    expected = sorted(rectangles, key=lambda r: r.x1)
    sigma_x = image.grid_map.traces_per_px
    sigma_y = image.grid_map.samples_per_px
    for (_, grid_box), rectangle in zip(got, expected, strict=True):
        assert grid_box == image_box_to_grid_box(rectangle, image.grid_map)
        assert 0 <= rectangle.x1 * sigma_x - grid_box.i0 < 1
        assert 0 <= grid_box.i1 + 1 - rectangle.x2 * sigma_x < 1
        assert 0 <= rectangle.y1 * sigma_y - grid_box.k0 < 1
        assert 0 <= grid_box.k1 + 1 - rectangle.y2 * sigma_y < 1


# ------------------------------------------------------------------ T8n (C4)


@pytest.mark.parametrize("width_px", [6, 197])
def test_t8n_single_tile_equals_unmerged(monkeypatch, width_px: int) -> None:
    """T8n ([CV] C4): one tile -> detect_line == detect_tiles + line_boxes_to_grid."""
    image = fixture_image(width_px=width_px)
    rng = np.random.default_rng(SEED)
    boxes = []
    for _ in range(8):
        x1 = float(rng.uniform(0, width_px * 0.8))
        x2 = float(rng.uniform(x1, width_px))
        y1 = float(rng.uniform(0, 60))
        class_id = int(rng.integers(0, 4))
        boxes.append(
            (class_id, float(rng.uniform(0.3, 1)), PixelBox(x1, y1, x2, y1 + 15))
        )
    boxes.append((HYPERBOLA, 0.5, boxes[0][2]))  # same-tile duplicate is not merged
    _use_image(monkeypatch, image)

    detections = detect_line(object(), VisiblePartDetector(image, boxes))
    unmerged = detect_tiles(image, VisiblePartDetector(image, boxes))
    unmerged.sort(key=pipeline._output_order)
    reference = line_boxes_to_grid(unmerged, image)

    assert [(b.score, g) for b, g in detections.boxes] == [
        (b.score, g) for b, g in reference
    ]
    assert detections.tile_seam_box == 0


def test_detect_tiles_marks_seam_boxes() -> None:
    image = fixture_image()
    detector = VisiblePartDetector(
        image, [(HYPERBOLA, 0.9, PixelBox(230, 30, 290, 60))]
    )
    boxes = detect_tiles(image, detector)
    truncated = {box.tile_index: box.tile_truncated for box in boxes}
    # Tile 0 [0, 256) cuts it on the right, tile 2 [256, 512) on the left.
    assert truncated == {0: True, 1: False, 2: True}
