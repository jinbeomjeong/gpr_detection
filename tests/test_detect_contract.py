"""Tests for the detector contract and the shared coordinate mapping (M1-8).

Covers letterbox -> ``InputTransform``, the pipeline inverse L2 -> L1 -> L0
(round trip <= 0.5 px, architecture README section 2-3), ``GridBox`` and
``ImageGridMap``, and the YOLO wrapper (stand-in model, plus a tiny real
one-channel model when ultralytics is installed).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gpr_detection.detect.base import (
    LETTERBOX_PAD_VALUE,
    Detection,
    Detector,
    DetectorOutput,
    InputTransform,
    PixelBox,
    letterbox,
)
from gpr_detection.detect.yolo import YoloDetector, load_yolo_detector
from gpr_detection.io.radargram import GridBox
from gpr_detection.pipeline import (
    detections_to_grid_boxes,
    grid_box_to_image_box,
    image_box_to_grid_box,
    model_box_to_grid_box,
    model_box_to_image_box,
)
from gpr_detection.preprocess.to_image import ImageGridMap

INPUT_SIZE_PX = 256
ROUND_TRIP_TOLERANCE_PX = 0.5
CLASS_NAMES = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")
# (height, width) of L1 images: AIHub square, Morocco, 4TU-like tall/wide strips.
L1_SHAPES = [(256, 256), (224, 224), (512, 300), (512, 1400), (180, 700), (97, 61)]


def _random_boxes(height: int, width: int, count: int, seed: int) -> list[PixelBox]:
    rng = np.random.default_rng(seed)
    boxes = []
    for _ in range(count):
        x1, x2 = np.sort(rng.uniform(0, width, size=2))
        y1, y2 = np.sort(rng.uniform(0, height, size=2))
        boxes.append(PixelBox(float(x1), float(y1), float(x2), float(y2)))
    return boxes


def _max_corner_error(first: PixelBox, second: PixelBox) -> float:
    return float(np.max(np.abs(np.subtract(first.xyxy, second.xyxy))))


# --- letterbox / InputTransform -------------------------------------------------


def test_letterbox_of_input_sized_image_is_identity() -> None:
    image = np.random.default_rng(0).integers(0, 256, (256, 256), dtype=np.uint8)

    padded, transform = letterbox(image, INPUT_SIZE_PX)

    np.testing.assert_array_equal(padded, image)
    assert transform == InputTransform(1.0, 1.0, 0.0, 0.0, 256, 256)


@pytest.mark.parametrize(("height", "width"), L1_SHAPES)
def test_letterbox_reports_exact_scale_and_padding(height: int, width: int) -> None:
    image = np.full((height, width), 200, dtype=np.uint8)

    padded, transform = letterbox(image, INPUT_SIZE_PX)

    assert padded.shape == (INPUT_SIZE_PX, INPUT_SIZE_PX)
    content_w = round(width * transform.scale_x)
    content_h = round(height * transform.scale_y)
    left, top = int(transform.pad_x), int(transform.pad_y)
    assert max(content_w, content_h) == INPUT_SIZE_PX
    assert np.all(padded[top : top + content_h, left : left + content_w] == 200)
    assert np.count_nonzero(padded == LETTERBOX_PAD_VALUE) == (
        INPUT_SIZE_PX * INPUT_SIZE_PX - content_w * content_h
    )


def test_letterbox_keeps_single_channel_axis() -> None:
    image = np.zeros((100, 50, 1), dtype=np.uint8)
    padded, _ = letterbox(image, INPUT_SIZE_PX)
    assert padded.shape == (INPUT_SIZE_PX, INPUT_SIZE_PX, 1)


def test_letterbox_rejects_empty_image() -> None:
    with pytest.raises(ValueError, match="letterbox"):
        letterbox(np.zeros((0, 10), dtype=np.uint8), INPUT_SIZE_PX)


@pytest.mark.parametrize(
    "build",
    [
        lambda: InputTransform(0.0, 1.0, 0.0, 0.0, 256, 256),
        lambda: InputTransform(1.0, 1.0, 0.0, 0.0, 0, 256),
        lambda: PixelBox(5.0, 0.0, 4.0, 1.0),
    ],
)
def test_invalid_transform_or_box_is_rejected(build) -> None:
    with pytest.raises(ValueError):
        build()


# --- L1 -> L2 -> L1 round trip (<= 0.5 px) -----------------------------------------


@pytest.mark.parametrize(("height", "width"), L1_SHAPES)
def test_box_round_trip_l1_l2_l1_within_half_pixel(height: int, width: int) -> None:
    _, transform = letterbox(np.zeros((height, width), np.uint8), INPUT_SIZE_PX)

    for box in _random_boxes(height, width, count=50, seed=height * width):
        recovered = model_box_to_image_box(transform.apply(box), transform)
        assert _max_corner_error(box, recovered) <= ROUND_TRIP_TOLERANCE_PX


@pytest.mark.parametrize(("height", "width"), [(256, 256), (512, 300), (180, 700)])
def test_drawn_rectangle_round_trip_within_half_pixel(height: int, width: int) -> None:
    """Find a bright rectangle in the letterboxed pixels and map it back to L1.

    Checks that ``InputTransform`` describes where the pixels really went, not
    only that the forward and inverse formulas agree.
    """
    box = PixelBox(x1=width * 0.25, y1=height * 0.5, x2=width * 0.75, y2=height * 0.75)
    image = np.zeros((height, width), dtype=np.uint8)
    image[round(box.y1) : round(box.y2), round(box.x1) : round(box.x2)] = 255
    drawn = PixelBox(round(box.x1), round(box.y1), round(box.x2), round(box.y2))

    padded, transform = letterbox(image, INPUT_SIZE_PX)

    bright_rows = np.flatnonzero((padded >= 128).any(axis=1))
    bright_cols = np.flatnonzero((padded >= 128).any(axis=0))
    found_l2 = PixelBox(
        float(bright_cols[0]),
        float(bright_rows[0]),
        float(bright_cols[-1] + 1),
        float(bright_rows[-1] + 1),
    )
    recovered = model_box_to_image_box(found_l2, transform)
    max_error_l1_px = _max_corner_error(drawn, recovered)
    # One L2 pixel of edge quantisation is worth 1/scale L1 pixels.
    allowed_px = max(ROUND_TRIP_TOLERANCE_PX, 0.5 / transform.scale_x)
    assert max_error_l1_px <= allowed_px


# --- L0 grid ----------------------------------------------------------------------


def test_grid_box_validation_and_slices() -> None:
    box = GridBox(i0=3, i1=5, k0=10, k1=10)
    assert (box.n_traces, box.n_samples) == (3, 1)
    data = np.arange(20 * 8).reshape(20, 8)
    assert data[box.sample_slice, box.trace_slice].shape == (1, 3)
    with pytest.raises(ValueError):
        GridBox(i0=5, i1=4, k0=0, k1=0)
    with pytest.raises(ValueError):
        GridBox(i0=-1, i1=4, k0=0, k1=0)


@pytest.mark.parametrize(
    "build",
    [
        lambda: ImageGridMap(0, 0, 0.0, 1.0, 10, 10),
        lambda: ImageGridMap(0, 0, 1.0, 1.0, 0, 10),
        lambda: ImageGridMap(10, 0, 1.0, 1.0, 10, 10),
    ],
)
def test_invalid_grid_map_is_rejected(build) -> None:
    with pytest.raises(ValueError):
        build()


def test_identity_grid_map_rounds_box_outward() -> None:
    grid_map = ImageGridMap.identity(width_px=256, height_px=256)

    grid_box = image_box_to_grid_box(PixelBox(10.2, 20.0, 19.5, 30.0), grid_map)

    assert grid_box == GridBox(i0=10, i1=19, k0=20, k1=29)


def test_grid_box_is_clipped_and_outside_box_is_none() -> None:
    grid_map = ImageGridMap.identity(width_px=100, height_px=50)

    clipped = image_box_to_grid_box(PixelBox(-5.0, -3.0, 120.0, 70.0), grid_map)
    outside = image_box_to_grid_box(PixelBox(110.0, 0.0, 130.0, 10.0), grid_map)

    assert clipped == GridBox(i0=0, i1=99, k0=0, k1=49)
    assert outside is None


def test_zero_width_box_covers_one_cell() -> None:
    grid_map = ImageGridMap.identity(width_px=100, height_px=50)
    grid_box = image_box_to_grid_box(PixelBox(7.0, 7.0, 7.0, 7.0), grid_map)
    assert grid_box == GridBox(i0=7, i1=7, k0=7, k1=7)


def test_grid_box_round_trip_l0_l1_l2_l0_is_exact() -> None:
    # A tile of a 4TU-like profile: offset origin, different axis factors.
    grid_map = ImageGridMap(
        trace_offset=1200,
        sample_offset=40,
        traces_per_px=2.5,
        samples_per_px=1.75,
        n_traces=5000,
        n_samples=512,
    )
    image_height = round((512 - 40) / 1.75)
    image_width = 700
    _, transform = letterbox(
        np.zeros((image_height, image_width), np.uint8), INPUT_SIZE_PX
    )
    rng = np.random.default_rng(3)
    for _ in range(200):
        i0 = int(rng.integers(1200, 2900))
        k0 = int(rng.integers(40, 450))
        grid_box = GridBox(
            i0=i0,
            i1=i0 + int(rng.integers(0, 50)),
            k0=k0,
            k1=k0 + int(rng.integers(0, 50)),
        )
        model_box = transform.apply(grid_box_to_image_box(grid_box, grid_map))
        assert model_box_to_grid_box(model_box, transform, grid_map) == grid_box


def test_detections_to_grid_boxes_drops_boxes_in_padding() -> None:
    _, transform = letterbox(np.zeros((128, 256), np.uint8), INPUT_SIZE_PX)
    grid_map = ImageGridMap.identity(width_px=256, height_px=128)
    inside = Detection(0, "pipe_hyperbola", 0.9, PixelBox(10.0, 70.0, 50.0, 100.0))
    in_padding = Detection(1, "pipe_band", 0.8, PixelBox(10.0, 2.0, 50.0, 30.0))
    output = DetectorOutput(detections=(inside, in_padding), transform=transform)

    pairs = detections_to_grid_boxes(output, grid_map)

    assert pairs == [(inside, GridBox(i0=10, i1=49, k0=6, k1=35))]


# --- YOLO wrapper with a stand-in model ---------------------------------------------


class _FakeYolo:
    """Records the predict call and returns fixed L2 boxes."""

    def __init__(self) -> None:
        self.names = dict(enumerate(CLASS_NAMES))
        self.calls: list[dict] = []

    def predict(self, source, **kwargs):
        self.calls.append({"source": source, **kwargs})
        boxes = SimpleNamespace(
            xyxy=np.array([[10.0, 20.0, 30.0, 40.0], [0.0, 0.0, 256.0, 256.0]]),
            conf=np.array([0.9, 0.4]),
            cls=np.array([1.0, 3.0]),
        )
        return [SimpleNamespace(boxes=boxes)]


def test_yolo_detector_returns_l2_boxes_and_letterbox_transform() -> None:
    model = _FakeYolo()
    detector = YoloDetector(model, input_channels=1, device="cpu")

    output = detector.predict(np.zeros((128, 256), dtype=np.uint8))

    assert isinstance(detector, Detector)
    assert detector.class_names == CLASS_NAMES
    call = model.calls[0]
    assert tuple(call["source"].shape) == (1, 1, INPUT_SIZE_PX, INPUT_SIZE_PX)
    assert call["imgsz"] == INPUT_SIZE_PX
    assert output.transform == InputTransform(1.0, 1.0, 0.0, 64.0, 256, 256)
    first = output.detections[0]
    assert (first.class_id, first.class_name) == (1, "pipe_band")
    assert first.score == pytest.approx(0.9)
    assert first.box == PixelBox(10.0, 20.0, 30.0, 40.0)


def test_yolo_detector_replicates_gray_for_three_channel_weights() -> None:
    model = _FakeYolo()
    detector = YoloDetector(model, input_channels=3)
    color = np.zeros((256, 256, 3), dtype=np.uint8)

    detector.predict(color)

    source = model.calls[0]["source"]
    assert tuple(source.shape) == (1, 3, INPUT_SIZE_PX, INPUT_SIZE_PX)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((16, 16), dtype=np.float32),
        np.zeros((16, 16, 2), dtype=np.uint8),
    ],
)
def test_yolo_detector_rejects_unsupported_images(image) -> None:
    detector = YoloDetector(_FakeYolo(), input_channels=1)
    with pytest.raises(ValueError):
        detector.predict(image)


def test_yolo_detector_accepts_single_channel_axis() -> None:
    model = _FakeYolo()
    YoloDetector(model, input_channels=1).predict(np.zeros((64, 64, 1), np.uint8))
    assert tuple(model.calls[0]["source"].shape) == (1, 1, 256, 256)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"input_size": 250}, "multiple"), ({"input_channels": 2}, "1 or 3")],
)
def test_yolo_detector_rejects_bad_settings(kwargs, message) -> None:
    settings = {"input_channels": 1, **kwargs}
    with pytest.raises(ValueError, match=message):
        YoloDetector(_FakeYolo(), **settings)


def test_load_yolo_detector_needs_existing_weights(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yolo_detector(tmp_path / "missing.pt")


# --- YOLO wrapper with a tiny real one-channel model ---------------------------------


@pytest.fixture(scope="module")
def tiny_one_channel_weights(tmp_path_factory):
    pytest.importorskip("ultralytics")
    import torch
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel("yolo11n.yaml", ch=1, nc=len(CLASS_NAMES), verbose=False)
    model.names = dict(enumerate(CLASS_NAMES))
    path = tmp_path_factory.mktemp("weights") / "tiny_ch1.pt"
    torch.save({"model": model, "train_args": {}}, path)
    return path


def test_real_one_channel_yolo_reports_boxes_inside_model_input(
    tiny_one_channel_weights,
) -> None:
    detector = load_yolo_detector(
        tiny_one_channel_weights, confidence_threshold=0.0001, device="cpu"
    )
    image = np.random.default_rng(0).integers(0, 256, (180, 300), dtype=np.uint8)

    output = detector.predict(image)

    assert detector.class_names == CLASS_NAMES
    assert output.transform.input_w == INPUT_SIZE_PX
    assert output.detections, "untrained model at conf 1e-4 should return boxes"
    corners = np.array([detection.box.xyxy for detection in output.detections])
    assert corners.min() >= 0.0
    assert corners.max() <= INPUT_SIZE_PX
