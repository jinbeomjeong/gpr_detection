"""Per-line depth call order (spec_detection_image_v1 v0.3.4 section 8-3, TD1-TD6, T9).

Synthetic line: ``BScanSpec(eps_r=9, d_top_m=0.6, n_traces=401, air_gap_m=0)``
(ground-coupled, curvature velocity), optionally with a flat band as an extra
event. The hyperbola box follows the M2-5 supply rule (+-0.8 m, 1.5 periods).
Spies wrap the depth functions as ``pipeline`` sees them.
"""

import math
from collections.abc import Callable

import numpy as np
import pytest

from gpr_detection import pipeline
from gpr_detection.depth.band import band_exclude_traces
from gpr_detection.depth.config import (
    DEFAULT_BAND_CONFIG,
    DEFAULT_DEPTH_CONFIG,
    BandConfig,
)
from gpr_detection.depth.estimate import estimate_depth, prepare_depth_inputs
from gpr_detection.detect.base import PixelBox
from gpr_detection.io.classes import DetectionClass
from gpr_detection.io.radargram import GridBox
from gpr_detection.pipeline import (
    DepthBox,
    LineBox,
    LineDetections,
    estimate_line_depths,
    estimate_survey_depths,
)
from gpr_detection.preprocess.filters import prepare_fitting_branch
from gpr_detection.preprocess.timezero import find_time_zero
from gpr_detection.preprocess.to_image import DetectionImageConfig, make_detection_image
from synthetic_bscan import (
    DT_NS,
    F_C_HZ,
    T_SRC_NS,
    BScanSpec,
    cylinder_times,
    make_radargram,
)

N_TRACES = 401
PERIOD_NS = 1e9 / F_C_HZ
BOX_HALF_WIDTH_M = 0.8
BOX_MARGIN_PERIODS = 1.5
BAND_TAU_NS = 6.0
BAND_AMPLITUDE = 2.0
BAND_TRACES = slice(0, 200)

HYPERBOLA = DetectionClass.PIPE_HYPERBOLA
BAND = DetectionClass.PIPE_BAND


def _spec(band_traces: slice | None = None) -> BScanSpec:
    spec = BScanSpec(eps_r=9, d_top_m=0.6, n_traces=N_TRACES, air_gap_m=0)
    if band_traces is not None:
        amplitude = np.zeros(N_TRACES)
        amplitude[band_traces] = BAND_AMPLITUDE
        spec.extra_events = [(np.full(N_TRACES, BAND_TAU_NS), amplitude)]
    return spec


def _hyperbola_box(spec: BScanSpec) -> GridBox:
    """M2-5 supply rule on the analytic profile (apex at the centre trace)."""
    apex = spec.n_traces // 2
    half_width = math.ceil(BOX_HALF_WIDTH_M / spec.dx_m - 1e-9)
    i0, i1 = apex - half_width, apex + half_width
    times = T_SRC_NS + cylinder_times(
        spec.x_m, spec.x0_m, spec.d_top_m, spec.velocity, spec.radius_m
    )
    margin_ns = BOX_MARGIN_PERIODS * PERIOD_NS
    return GridBox(
        i0=i0,
        i1=i1,
        k0=math.floor((times[apex] - margin_ns) / DT_NS),
        k1=math.ceil((times[i0 : i1 + 1].max() + margin_ns) / DT_NS),
    )


def _band_box(i0: int, i1: int) -> GridBox:
    """0.4 period above to 0.8 period below the flat band."""
    record_ns = T_SRC_NS + BAND_TAU_NS
    return GridBox(
        i0=i0,
        i1=i1,
        k0=math.floor((record_ns - 0.4 * PERIOD_NS) / DT_NS),
        k1=math.ceil((record_ns + 0.8 * PERIOD_NS) / DT_NS),
    )


class CallLog:
    """Records ``(name, args, kwargs)`` of wrapped pipeline functions in order."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        for name in names:
            monkeypatch.setattr(
                pipeline, name, self._wrap(name, getattr(pipeline, name))
            )

    def _wrap(self, name: str, function: Callable) -> Callable:
        def spy(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return function(*args, **kwargs)

        return spy

    def names(self) -> list[str]:
        return [name for name, _, _ in self.calls]

    def of(self, name: str) -> list[tuple[tuple, dict]]:
        return [(args, kwargs) for n, args, kwargs in self.calls if n == name]


DEPTH_FUNCTIONS = (
    "prepare_depth_inputs",
    "estimate_depth",
    "velocity_estimate_from",
    "estimate_band_depths",
)


@pytest.fixture(scope="module")
def band_line():
    spec = _spec(BAND_TRACES)
    radargram = make_radargram(spec)
    boxes = [
        DepthBox(BAND, _band_box(0, 199)),
        DepthBox(HYPERBOLA, _hyperbola_box(spec)),
    ]
    return radargram, boxes


def test_td1_band_traces_are_excluded_from_the_one_zero(monkeypatch, band_line):
    """TD1: one prepare call with the band mask; every result shares its zero."""
    radargram, boxes = band_line
    log = CallLog(monkeypatch, "prepare_depth_inputs")
    depths = estimate_line_depths(radargram, boxes)

    ((_, kwargs),) = log.of("prepare_depth_inputs")
    expected_mask = band_exclude_traces(
        [boxes[0].grid_box], N_TRACES, radargram.dx_m, DEFAULT_BAND_CONFIG
    )
    np.testing.assert_array_equal(kwargs["exclude_traces"], expected_mask)
    np.testing.assert_array_equal(depths.exclude_traces, expected_mask)

    config = DEFAULT_DEPTH_CONFIG
    direct_zero = find_time_zero(
        radargram,
        prepare_fitting_branch(radargram, config.bg_window).filtered,
        pick_mode=config.pick_mode,
        exclude_traces=expected_mask,
        config=config.zero_config(),
    )
    assert depths.inputs.zero.t_zero_ns == direct_zero.t_zero_ns
    ((_, hyperbola_result),) = depths.hyperbola
    (band_result,) = depths.band
    assert hyperbola_result.t_zero_ns == direct_zero.t_zero_ns
    assert band_result.t_zero_ns == direct_zero.t_zero_ns


def test_td2_full_line_band_keeps_the_unmasked_zero():
    """TD2: a band over every trace -> zero_band_overlap, same zero as no mask."""
    spec = _spec(slice(0, N_TRACES))
    radargram = make_radargram(spec)
    boxes = [
        DepthBox(BAND, _band_box(0, N_TRACES - 1)),
        DepthBox(HYPERBOLA, _hyperbola_box(spec)),
    ]
    depths = estimate_line_depths(radargram, boxes)
    assert "zero_band_overlap" in depths.inputs.zero.flags
    unmasked = prepare_depth_inputs(radargram, DEFAULT_DEPTH_CONFIG)
    assert depths.inputs.zero.t_zero_ns == unmasked.zero.t_zero_ns


def test_td3_no_band_equals_direct_calls(monkeypatch):
    """TD3: without a band the mask is None and results equal the direct calls."""
    spec = _spec()
    radargram = make_radargram(spec)
    box = _hyperbola_box(spec)
    log = CallLog(monkeypatch, "prepare_depth_inputs")
    depths = estimate_line_depths(radargram, [DepthBox(HYPERBOLA, box)])

    ((_, kwargs),) = log.of("prepare_depth_inputs")
    assert kwargs["exclude_traces"] is None
    assert depths.exclude_traces is None
    inputs = prepare_depth_inputs(radargram, DEFAULT_DEPTH_CONFIG)
    expected = estimate_depth(radargram, box, DEFAULT_DEPTH_CONFIG, inputs=inputs)
    ((_, result),) = depths.hyperbola
    assert result == expected
    assert depths.band == ()


def test_td4_call_order_and_the_velocity_pool(monkeypatch, band_line):
    """TD4: prepare -> estimate_depth -> velocity_estimate_from -> band, pool passed."""
    radargram, boxes = band_line
    log = CallLog(monkeypatch, *DEPTH_FUNCTIONS)
    depths = estimate_line_depths(radargram, boxes)

    assert log.names() == list(DEPTH_FUNCTIONS)
    assert depths.pool and depths.pool[0].eligible, "fixture: hyperbola not eligible"
    ((args, kwargs),) = log.of("estimate_band_depths")
    _, band_boxes, zero, pool, survey_pool, _ = args
    assert band_boxes == [boxes[0].grid_box]
    assert zero is depths.inputs.zero
    assert tuple(pool) == depths.pool
    assert tuple(survey_pool) == ()
    assert kwargs["branch"] is depths.inputs.branch
    (band_result,) = depths.band
    ((_, hyperbola_result),) = depths.hyperbola
    assert band_result.v_rule == "neighbor_hyperbola"
    assert band_result.v_m_per_ns == hyperbola_result.v_m_per_ns


def test_td5_classes_and_tile_flags(monkeypatch, band_line):
    """TD5: cavity/manhole skipped; tile_truncated / tile_clipped carried over."""
    radargram, boxes = band_line
    band_box, hyperbola_box = boxes
    cavity = DepthBox(DetectionClass.CAVITY, GridBox(300, 340, 100, 200))
    manhole = DepthBox(DetectionClass.MANHOLE, GridBox(20, 60, 300, 400))
    line_boxes = [
        cavity,
        DepthBox(HYPERBOLA, hyperbola_box.grid_box, ("tile_truncated",)),
        DepthBox(BAND, band_box.grid_box, ("tile_clipped",)),
        manhole,
    ]
    log = CallLog(monkeypatch, "estimate_depth", "estimate_band_depths")
    depths = estimate_line_depths(radargram, line_boxes)

    assert depths.skipped == (cavity, manhole)
    (depth_call,) = log.of("estimate_depth")
    assert depth_call[0][1] == hyperbola_box.grid_box
    ((band_args, _),) = log.of("estimate_band_depths")
    assert band_args[1] == [band_box.grid_box]
    ((_, hyperbola_result),) = depths.hyperbola
    assert hyperbola_result.flags[-1] == "tile_truncated"
    (band_result,) = depths.band
    assert "tile_clipped" in band_result.flags


def test_td6_pick_mode_mismatch_fails_before_any_call(monkeypatch, band_line):
    """TD6 (first part): different pick modes -> ValueError, nothing called."""
    radargram, boxes = band_line
    log = CallLog(monkeypatch, *DEPTH_FUNCTIONS)
    mismatch = BandConfig(pick_mode="signed_max")
    with pytest.raises(ValueError, match="pick_mode"):
        estimate_line_depths(radargram, boxes, band_config=mismatch)
    with pytest.raises(ValueError, match="pick_mode"):
        estimate_survey_depths(
            [(radargram, boxes)], band_config=mismatch, survey_key="road"
        )
    assert log.calls == []


def test_td6_survey_pool_spans_lines_and_zero_is_reused(monkeypatch, band_line):
    """TD6: one prepare per line; each band call gets both lines' pools."""
    radargram, boxes = band_line
    second = make_radargram(_spec(BAND_TRACES), line_id="unit/line2")
    log = CallLog(monkeypatch, "prepare_depth_inputs", "estimate_band_depths")

    depths = estimate_survey_depths(
        [(radargram, boxes), (second, boxes)], survey_key="road"
    )

    assert len(log.of("prepare_depth_inputs")) == 2
    expected_survey_pool = (*depths[0].pool, *depths[1].pool)
    assert len(expected_survey_pool) == 2
    band_calls = log.of("estimate_band_depths")
    assert len(band_calls) == 2
    for args, _ in band_calls:
        assert tuple(args[4]) == expected_survey_pool
    assert [d.line_id for d in depths] == ["unit/line", "unit/line2"]
    assert all(e.survey_key == "road" for e in expected_survey_pool)


def test_depth_boxes_from_carries_flags() -> None:
    clipped = LineBox(1, "pipe_band", 0.9, PixelBox(0, 0, 5, 5), 0, tile_clipped=True)
    cut = LineBox(
        0, "pipe_hyperbola", 0.9, PixelBox(0, 0, 5, 5), 0, tile_truncated=True
    )
    detections = LineDetections(
        image=None,  # not used by depth_boxes_from
        boxes=((clipped, GridBox(0, 1, 0, 1)), (cut, GridBox(2, 3, 0, 1))),
        tile_seam_box=1,
    )
    assert pipeline.depth_boxes_from(detections) == [
        DepthBox(1, GridBox(0, 1, 0, 1), ("tile_clipped",)),
        DepthBox(0, GridBox(2, 3, 0, 1), ("tile_truncated",)),
    ]


# ----------------------------------------------------------------------- T9

T9_CONFIGS = [
    DetectionImageConfig(p_target_px=p_star, overlap_px=overlap)
    for p_star in (4.06, 6.0, 8.0)
    for overlap in (128, 192)
]


def test_t9_depth_does_not_depend_on_the_detection_image(band_line):
    """T9: same L0 boxes -> identical results for every DetectionImageConfig."""
    radargram, boxes = band_line
    data_before = radargram.data.copy()
    reference = estimate_line_depths(radargram, boxes)
    for config in T9_CONFIGS:
        make_detection_image(radargram, config)
        depths = estimate_line_depths(radargram, boxes)
        assert depths.hyperbola == reference.hyperbola
        assert depths.band == reference.band
        assert depths.pool == reference.pool
    np.testing.assert_array_equal(radargram.data, data_before)
