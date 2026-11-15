"""Real-background injection generator (M3-2 S4; S6 tests (g), (h) and (j)).

Band mode (M3-3 R3, [EMR] D-5 tests (a)-(f)) is at the end of the file.

(g) truth boxes are L0 cells that survive L0 -> L1 -> L0 through the
``pipeline`` coordinate functions exactly (QA-8, ADR-004 decision 8).
(h) the waveform replacement filter passes the period check (+-5 %) and keeps
the mirror symmetry of the response.
(j) the strong-layer zone (b') of plan v0.2.3: few false alarms on decaying
noise, a flat reflector is flagged and excluded only within 1 P, and the
attenuation trend alone flags nothing.

The background is an air-coupled 4TU-like line (flat band + laterally random
Ricker texture); the pipe response is an analytic symmetric hyperbola on a
gprMax-like grid. No dataset files are read.
"""

import hashlib
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import convolve1d
from scipy.signal import hilbert

from gpr_detection.eval.inject import (
    EARLY_ENERGY_KEY,
    EXTENDED_INJECTION_TABLE_COLUMNS,
    INJECTION_TABLE_COLUMNS,
    K_MAIN,
    K_REPORT,
    MAX_REALIZATIONS_PER_LINE,
    MIRROR_DIFF_KEY,
    BandInjectionResult,
    BandPatch,
    BandSurvivalSpan,
    InjectionMode,
    InjectionSpec,
    InjectionTruth,
    LineExclusions,
    PipeResponse,
    RejectReason,
    ResponsePatch,
    band_spacing_ok,
    band_survival_from_sections,
    band_survival_ratio,
    band_taper_traces,
    box_size_in_periods,
    build_band_patch,
    build_injection_specs,
    build_response_patch,
    default_conditions,
    edge_taper_weights,
    inject_band_realization,
    inject_realization,
    is_in_distribution,
    line_cost,
    long_strong_layer_runs,
    make_pseudo_band,
    period_check_passes,
    plan_line_positions,
    prepare_line_context,
    read_injection_table,
    realization_spacing_ok,
    strong_layer_samples,
    write_cost_table,
    write_injection_table,
)
from gpr_detection.io.radargram import GridBox, Radargram
from gpr_detection.pipeline import grid_box_to_image_box, image_box_to_grid_box
from gpr_detection.preprocess.filters import bandpass, dewow, remove_dc

LINE_DT_NS = 0.098
LINE_DX_M = 0.02
F_C_HZ = 400e6
LINE_SAMPLES = 512
LINE_TRACES = 235
BAND_TIME_NS = 5.0
BAND_AMPLITUDE = 10000.0
TEXTURE_SCALE = 100.0

SYN_DT_NS = 0.0118
SYN_SAMPLES = 3224  # 38 ns window
SYN_TRACES = 111
SYN_APEX_TRACE = 55.0
SYN_REFERENCE_NS = 4.0
SYN_APEX_NS = 16.0
SOIL_VELOCITY_M_PER_NS = 0.1
BEAM_WIDTH_M = 0.4
"""Gaussian lateral amplitude so the -20 dB wing ends inside the aperture."""

LINE_ID = "01.1/Path3"
APEX_TRACE = 117.0  # offset 62 traces from the synthetic apex


def _ricker(times_ns: np.ndarray, f_c_hz: float) -> np.ndarray:
    arg = (np.pi * f_c_hz * 1e-9 * times_ns) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def _background_data(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    times_ns = np.arange(LINE_SAMPLES) * LINE_DT_NS
    band = BAND_AMPLITUDE * _ricker(times_ns - BAND_TIME_NS, F_C_HZ)
    wavelet = _ricker(np.arange(-40, 41) * LINE_DT_NS, F_C_HZ)
    spikes = rng.normal(0.0, 1.0, (LINE_SAMPLES, LINE_TRACES))
    texture = TEXTURE_SCALE * convolve1d(spikes, wavelet, axis=0)
    return band[:, None] + texture


def _radargram(data: np.ndarray, air_gap: str = "air") -> Radargram:
    return Radargram(
        data=data.astype(np.float32),
        dt_ns=LINE_DT_NS,
        dx_m=LINE_DX_M,
        air_gap=air_gap,  # type: ignore[arg-type]
        amplitude_processed=False,
        dataset="4tu",
        line_id=LINE_ID,
        f_c_hz=F_C_HZ,
    )


def _response(
    response_id: str = "M12",
    apex_ns: float = SYN_APEX_NS,
    beam_width_m: float = BEAM_WIDTH_M,
    **changes: object,
) -> PipeResponse:
    times_ns = np.arange(SYN_SAMPLES) * SYN_DT_NS
    offsets_m = (np.arange(SYN_TRACES) - SYN_APEX_TRACE) * LINE_DX_M
    depth_m = SOIL_VELOCITY_M_PER_NS * (apex_ns - SYN_REFERENCE_NS) / 2.0
    arrivals_ns = SYN_REFERENCE_NS + 2.0 / SOIL_VELOCITY_M_PER_NS * np.sqrt(
        depth_m**2 + offsets_m**2
    )
    beam = np.exp(-((offsets_m / beam_width_m) ** 2))
    data = beam[None, :] * _ricker(times_ns[:, None] - arrivals_ns[None, :], F_C_HZ)
    fields: dict[str, object] = {
        "response_id": response_id,
        "data": data,
        "dt_ns": SYN_DT_NS,
        "dx_m": LINE_DX_M,
        "apex_trace": SYN_APEX_TRACE,
        "apex_time_ns": apex_ns,
        "reference_time_ns": SYN_REFERENCE_NS,
        "coupling": "air",
        "air_gap_m": 0.1,
        "eps_r": 9.0,
        "f_c_hz": F_C_HZ,
        "top_depth_m": depth_m,
        "od_m": 0.2,
        "material": "pec",
        "quality": {MIRROR_DIFF_KEY: 0.0, EARLY_ENERGY_KEY: 0.0},
    }
    fields.update(changes)
    return PipeResponse(**fields)  # type: ignore[arg-type]


def _spec(
    apex_trace: float = APEX_TRACE,
    *,
    index: int = 0,
    realization_id: str = "r",
    polarity: int = 1,
    snr_db: float = 12.0,
    response_id: str = "M12",
    k: float = K_MAIN,
) -> InjectionSpec:
    return InjectionSpec(
        injection_id=f"{realization_id}:{index:02d}",
        realization_id=realization_id,
        line_id=LINE_ID,
        apex_trace=apex_trace,
        response_id=response_id,
        snr_db=snr_db,
        polarity=polarity,
        k=k,
        seed=0,
    )


@pytest.fixture(scope="module")
def background() -> Radargram:
    return _radargram(_background_data())


@pytest.fixture(scope="module")
def context(background):
    return prepare_line_context(background)


@pytest.fixture(scope="module")
def response() -> PipeResponse:
    return _response()


@pytest.fixture(scope="module")
def patch(response, context) -> ResponsePatch:
    built = build_response_patch(response, context)
    assert isinstance(built, ResponsePatch), built
    return built


@pytest.fixture(scope="module")
def injected(context, response):
    return inject_realization(context, [_spec()], {"M12": response})


def _linear_steps(section: np.ndarray) -> np.ndarray:
    filtered = dewow(remove_dc(section), LINE_DT_NS, F_C_HZ)
    return bandpass(filtered, LINE_DT_NS, F_C_HZ)


def _padded_envelope(section: np.ndarray) -> np.ndarray:
    n_samples = section.shape[0]
    return np.abs(hilbert(section, N=2 * n_samples, axis=0)[:n_samples])


# ------------------------------------------------------------- (g) truth boxes


def test_g_truth_boxes_round_trip_exactly_through_pipeline(context, injected):
    (truth,) = injected.truths
    for box in (truth.box, truth.wing_box, truth.core_box):
        line_box = grid_box_to_image_box(box, context.grid_map)
        assert image_box_to_grid_box(line_box, context.grid_map) == box


def test_g_planned_truth_boxes_round_trip_exactly(context, response, patch):
    plan = plan_line_positions(context, [patch], max_injections=6)
    specs = build_injection_specs(LINE_ID, plan.positions, default_conditions()[:1])
    by_realization: dict[str, list[InjectionSpec]] = {}
    for spec in specs:
        by_realization.setdefault(spec.realization_id, []).append(spec)
    truths: list[InjectionTruth] = []
    for realization_specs in by_realization.values():
        result = inject_realization(context, realization_specs, {"M12": response})
        truths.extend(result.truths)
    assert truths
    for truth in truths:
        line_box = grid_box_to_image_box(truth.box, context.grid_map)
        assert image_box_to_grid_box(line_box, context.grid_map) == truth.box


def test_g_box_size_in_periods_is_the_pipeline_forward_map(context, injected):
    (truth,) = injected.truths
    line_box = grid_box_to_image_box(truth.wing_box, context.grid_map)
    period_l1_px = context.period_samples / context.grid_map.samples_per_px
    width, height = box_size_in_periods(
        truth.wing_box, context.grid_map, context.period_samples
    )
    assert width == pytest.approx((line_box.x2 - line_box.x1) / period_l1_px)
    assert height == pytest.approx((line_box.y2 - line_box.y1) / period_l1_px)
    assert (truth.width_periods, truth.height_periods) == (width, height)


def test_truth_apex_follows_the_time_alignment(context, injected):
    (truth,) = injected.truths
    t_zero_ns = context.zero.t_zero_ns
    expected_ns = t_zero_ns + (SYN_APEX_NS - SYN_REFERENCE_NS)
    assert truth.apex_sample * LINE_DT_NS == pytest.approx(expected_ns)
    assert truth.box.i0 <= truth.apex_trace <= truth.box.i1
    assert truth.box.k0 <= truth.apex_sample <= truth.box.k1


def test_truth_box_excludes_the_aperture_taper(patch, injected):
    (truth,) = injected.truths
    taper_traces = round(0.2 / LINE_DX_M)
    offset = round(APEX_TRACE - SYN_APEX_TRACE)
    assert truth.box.i0 >= offset + taper_traces
    assert truth.box.i1 <= offset + SYN_TRACES - 1 - taper_traces
    assert truth.wing_box.i0 <= truth.box.i0
    assert truth.wing_box.i1 >= truth.box.i1


# ----------------------------------------------------- (h) waveform filter


@pytest.mark.parametrize("k", [K_MAIN, K_REPORT])
def test_h_filtered_period_is_within_5_percent(response, context, k):
    built = build_response_patch(response, context, k)
    assert isinstance(built, ResponsePatch), built
    assert built.period_samples == pytest.approx(context.period_samples, rel=0.05)


@pytest.mark.parametrize("k", [K_MAIN, K_REPORT])
def test_h_filter_keeps_the_mirror_symmetry(response, context, k):
    built = build_response_patch(response, context, k)
    assert isinstance(built, ResponsePatch)
    apex = round(SYN_APEX_TRACE)
    left = built.field[:, apex::-1]
    right = built.field[:, apex : apex + left.shape[1]]
    assert np.max(np.abs(left - right)) <= 1e-9 * np.max(np.abs(built.field))


def test_h_period_check_tolerance():
    assert period_check_passes(25.0 * 1.049, 25.0)
    assert period_check_passes(25.0 * 0.951, 25.0)
    assert not period_check_passes(25.0 * 1.06, 25.0)
    assert not period_check_passes(None, 25.0)


def test_gabor_width_changes_the_injected_wavelet_length(response, context):
    long_patch = build_response_patch(response, context, K_MAIN)
    short_patch = build_response_patch(response, context, K_REPORT)
    assert isinstance(long_patch, ResponsePatch)
    assert isinstance(short_patch, ResponsePatch)

    def fwhm_samples(field: np.ndarray) -> int:
        envelope = np.abs(hilbert(field[:, round(SYN_APEX_TRACE)]))
        return int(np.count_nonzero(envelope >= 0.5 * envelope.max()))

    assert fwhm_samples(long_patch.field) > fwhm_samples(short_patch.field)


# ------------------------------------------------------------ SNR, polarity


def test_snr_scale_matches_the_ladder_definition(background, injected):
    (truth,) = injected.truths
    assert abs(truth.gain) * truth.signal_amplitude / truth.noise_rms == pytest.approx(
        10.0 ** (12.0 / 20.0)
    )
    injected_only = injected.data - background.data.astype(np.float64)
    envelope = _padded_envelope(_linear_steps(injected_only))
    apex_column = round(truth.apex_trace)
    assert envelope[:, apex_column].max() == pytest.approx(
        abs(truth.gain) * truth.signal_amplitude, rel=1e-6
    )


def test_polarity_pair_is_an_exact_negation(background, context, response):
    plus = inject_realization(context, [_spec(polarity=1)], {"M12": response})
    minus = inject_realization(
        context, [_spec(polarity=-1, realization_id="m")], {"M12": response}
    )
    (plus_truth,) = plus.truths
    (minus_truth,) = minus.truths
    assert plus_truth.gain == -minus_truth.gain
    assert plus_truth.box == minus_truth.box
    raw = background.data.astype(np.float64)
    added = plus.data - raw
    # Equal up to the rounding of adding the field to the int16-valued raw line.
    tolerance = 1e-12 * np.max(np.abs(added)) + 1e-9
    np.testing.assert_allclose(added, -(minus.data - raw), rtol=0, atol=tolerance)


def test_injection_leaves_the_background_untouched(background, context, response):
    before = background.data.copy()
    inject_realization(context, [_spec()], {"M12": response})
    np.testing.assert_array_equal(background.data, before)


# ---------------------------------------------------------- rejection rules


def _single_rejection(context, specs, responses) -> str:
    result = inject_realization(context, specs, responses)
    assert result.truths == ()
    ((_, reason),) = result.rejected
    return reason


@pytest.mark.parametrize(
    ("quality", "reason"),
    [
        ({MIRROR_DIFF_KEY: 2e-3, EARLY_ENERGY_KEY: 0.0}, RejectReason.MIRROR_DIFF),
        ({MIRROR_DIFF_KEY: 0.0, EARLY_ENERGY_KEY: 0.02}, RejectReason.EARLY_ENERGY),
        ({MIRROR_DIFF_KEY: 0.0}, RejectReason.QUALITY_MISSING),
    ],
)
def test_quality_gate(context, quality, reason):
    bad = _response(quality=quality)
    assert _single_rejection(context, [_spec()], {"M12": bad}) == reason


def test_coupling_must_match(context):
    ground = _response(coupling="ground")
    reason = _single_rejection(context, [_spec()], {"M12": ground})
    assert reason == RejectReason.COUPLING_MISMATCH


def test_unknown_response_and_off_grid_apex(context, response):
    responses = {"M12": response}
    assert (
        _single_rejection(context, [_spec(response_id="M99")], responses)
        == RejectReason.UNKNOWN_RESPONSE
    )
    assert (
        _single_rejection(context, [_spec(APEX_TRACE + 0.5)], responses)
        == RejectReason.OFF_GRID_APEX
    )
    assert (
        _single_rejection(context, [_spec(SYN_APEX_TRACE - 1.0)], responses)
        == RejectReason.APERTURE_OUTSIDE_LINE
    )


def test_band_zone_rejects_shallow_apexes(context):
    shallow = _response(apex_ns=SYN_REFERENCE_NS + 3.0)
    reason = _single_rejection(context, [_spec()], {"M12": shallow})
    assert reason == RejectReason.BAND_ZONE


def test_saturation_zone_uses_the_last_saturated_row():
    data = _background_data()
    data[160, 5] = 32767.0
    saturated_context = prepare_line_context(_radargram(data))
    assert saturated_context.last_saturated_sample == 160
    reason = _single_rejection(saturated_context, [_spec()], {"M12": _response()})
    assert reason == RejectReason.SATURATION_ZONE


def test_strong_flat_layer_near_the_apex_is_excluded(patch):
    data = _background_data()
    layer_row = round(patch.apex_sample) + 10
    times_ns = (np.arange(LINE_SAMPLES) - layer_row) * LINE_DT_NS
    data += 20.0 * TEXTURE_SCALE * _ricker(times_ns, F_C_HZ)[:, None]
    layered = prepare_line_context(_radargram(data))
    reason = _single_rejection(layered, [_spec()], {"M12": _response()})
    assert reason == RejectReason.STRONG_LAYER_ZONE


def test_int16_clipping_inside_the_truth_box_rejects(patch):
    """Rule (i): a clipped cell in the truth box rejects the injection."""
    data = _background_data()
    offset = round(APEX_TRACE - SYN_APEX_TRACE)
    wing_column = offset + round(SYN_APEX_TRACE + patch.half_width_6db) + 3
    data[:, wing_column : wing_column + 4] = 32766.6
    near_limit = prepare_line_context(_radargram(data))
    assert near_limit.last_saturated_sample is None
    spec = _spec(snr_db=18.0)
    reason = _single_rejection(near_limit, [spec], {"M12": _response()})
    assert reason == RejectReason.INT16_OVERFLOW


def _saturated_above_apex(patch, periods_above: float) -> np.ndarray:
    """Background with a saturated row ``periods_above`` P above the apex."""
    data = _background_data()
    row = round(patch.apex_sample - periods_above * patch.period_samples)
    data[row, :] = 32767.0
    data[row + 1, :] = -32768.0
    return data


@pytest.mark.parametrize("snr_db", [0.0, 18.0])
def test_int16_clipping_far_above_the_apex_is_kept_at_every_snr(patch, snr_db):
    """Small clipped cells outside the truth box and SNR window do not reject."""
    clipped_context = prepare_line_context(
        _radargram(_saturated_above_apex(patch, 2.5))
    )
    responses = {"M12": _response()}
    result = inject_realization(clipped_context, [_spec(snr_db=snr_db)], responses)
    (truth,) = result.truths
    assert truth.n_clipped_cells > 0
    assert 0 < truth.max_clipped_injection_ratio <= 0.05
    assert result.data.max() <= 32767 and result.data.min() >= -32768


def test_clip_rules_protected_box_and_ratio(patch):
    from gpr_detection.eval import inject

    offset = 10
    n_rows = LINE_SAMPLES
    data = np.zeros((n_rows, LINE_TRACES))
    apex_row = round(patch.apex_sample)
    column = round(SYN_APEX_TRACE)
    gain = 1.0 / abs(patch.field[apex_row, column])
    data[apex_row, offset + column] = 32767.0 * np.sign(patch.field[apex_row, column])
    summed, outcome = inject._clip_sum(data, patch, offset, gain, ())
    assert outcome.n_clipped_cells == 1
    assert outcome.max_ratio == pytest.approx(
        abs(patch.field[apex_row, column]) / patch.signal_amplitude
    )
    assert outcome.rejects  # the apex carries far more than 5 %
    assert np.abs(summed).max() <= 32768
    box = GridBox(offset + column, offset + column, apex_row, apex_row)
    _, protected = inject._clip_sum(data, patch, offset, gain, (box,))
    assert protected.protected_hit
    small = inject.ClipOutcome(n_clipped_cells=3, max_ratio=0.05, protected_hit=False)
    assert not small.rejects


def _box_context(background, box: GridBox):
    return prepare_line_context(background, LineExclusions(uninjected_boxes=(box,)))


def test_zone_c_blocks_a_box_touching_the_core(background, response, patch):
    apex_row = round(patch.apex_sample)
    beside_core = GridBox(i0=100, i1=102, k0=apex_row, k1=apex_row + 5)
    assert (
        _single_rejection(
            _box_context(background, beside_core), [_spec()], {"M12": response}
        )
        == RejectReason.UNINJECTED_BOX_ZONE
    )


def test_zone_c_lets_a_box_far_above_the_apex_pass(background, response, patch):
    """Same traces as the apex, far above: the old horizontal rule blocked it."""
    above = GridBox(i0=110, i1=112, k0=0, k1=10)
    result = inject_realization(
        _box_context(background, above), [_spec()], {"M12": response}
    )
    (truth,) = result.truths
    assert truth.in_horizontal_box_zone
    period = patch.period_samples
    assert above.k1 + period < patch.core_bbox[2]


def test_zone_c_grows_boxes_by_one_period_in_samples(background, response, patch):
    core_top = patch.core_bbox[2]
    period = round(patch.period_samples)
    touching = GridBox(i0=110, i1=112, k0=0, k1=core_top - period + 1)
    clear = GridBox(i0=110, i1=112, k0=0, k1=core_top - period - 2)
    responses = {"M12": response}
    assert (
        _single_rejection(_box_context(background, touching), [_spec()], responses)
        == RejectReason.UNINJECTED_BOX_ZONE
    )
    result = inject_realization(_box_context(background, clear), [_spec()], responses)
    assert len(result.truths) == 1


def test_candidate_zone_keeps_the_k_radius(background, response, patch):
    near = APEX_TRACE + patch.half_width_20db - 1.0
    candidate_context = prepare_line_context(
        background, LineExclusions(k_apex_traces=(near,))
    )
    assert (
        _single_rejection(candidate_context, [_spec()], {"M12": response})
        == RejectReason.CANDIDATE_APEX_ZONE_K
    )


def test_auto_candidates_block_only_the_core_plus_one_period(
    background, response, patch, context
):
    radius = patch.half_width_6db + context.period_traces
    assert radius < patch.half_width_20db
    inside = APEX_TRACE + radius - 0.5
    blocked = prepare_line_context(
        background, LineExclusions(auto_candidate_apex_traces=(inside,))
    )
    assert (
        _single_rejection(blocked, [_spec()], {"M12": response})
        == RejectReason.CANDIDATE_APEX_ZONE_AUTO
    )
    outside = APEX_TRACE + radius + 0.5  # still inside the -20 dB half width
    allowed = prepare_line_context(
        background, LineExclusions(auto_candidate_apex_traces=(outside,))
    )
    assert len(inject_realization(allowed, [_spec()], {"M12": response}).truths) == 1


def test_realization_spacing_is_width_plus_one_period(context, response, patch):
    """(e'): a second apex needs W_-20 + 1 P; just below it is rejected."""
    minimum = patch.expected_width + context.period_traces
    second = SYN_APEX_TRACE + np.floor(minimum)
    if second - SYN_APEX_TRACE >= minimum:
        second -= 1.0
    close = [_spec(SYN_APEX_TRACE), _spec(second, index=1)]
    result = inject_realization(context, close, {"M12": response})
    assert [truth.injection_id for truth in result.truths] == ["r:00"]
    assert result.rejected == (("r:01", RejectReason.SPACING.value),)
    far = [_spec(SYN_APEX_TRACE), _spec(SYN_APEX_TRACE + np.ceil(minimum), index=1)]
    assert len(inject_realization(context, far, {"M12": response}).truths) == 2


@pytest.mark.parametrize("period_traces", [0.0, 5.86])
def test_realization_spacing_with_mixed_widths_adds_both_half_widths(period_traces):
    """(e') with W 40 and W 80: the gap is 20 + 40 + 1 P, not W_max + 1 P."""
    boundary = 20.0 + 40.0 + period_traces
    placed = [(100.0, 40.0)]
    assert realization_spacing_ok(100.0 + boundary, 20.0, placed, period_traces)
    assert realization_spacing_ok(100.0 - boundary - 1.0, 20.0, placed, period_traces)
    assert not realization_spacing_ok(
        100.0 + boundary - 0.01, 20.0, placed, period_traces
    )
    assert not realization_spacing_ok(
        100.0 - boundary + 0.01, 20.0, placed, period_traces
    )


def test_realization_spacing_with_two_models_on_a_line(context):
    """Injected narrow and wide models follow hw_a + hw_b + 1 P."""
    narrow = build_response_patch(_response("N", beam_width_m=0.25), context)
    wide = build_response_patch(_response("W"), context)
    assert isinstance(narrow, ResponsePatch)
    assert isinstance(wide, ResponsePatch)
    assert narrow.half_width_20db < wide.half_width_20db
    minimum = narrow.half_width_20db + wide.half_width_20db + context.period_traces
    responses = {"N": narrow.response, "W": wide.response}
    below = SYN_APEX_TRACE + np.ceil(minimum) - 1.0
    above = SYN_APEX_TRACE + np.ceil(minimum)
    for second, accepted in ((below, 1), (above, 2)):
        specs = [
            _spec(SYN_APEX_TRACE, response_id="W"),
            _spec(second, index=1, response_id="N"),
        ]
        result = inject_realization(context, specs, responses)
        assert len(result.truths) == accepted, (second, result.rejected)
        if accepted == 1:
            assert result.rejected == (("r:01", RejectReason.SPACING.value),)


def _ringing_layer(layer_row: int, fwhm_periods: float, period: float) -> np.ndarray:
    """Flat Gabor layer: carrier ``f_c``, envelope FWHM ``fwhm_periods`` P."""
    times_ns = (np.arange(LINE_SAMPLES) - layer_row) * LINE_DT_NS
    sigma_ns = fwhm_periods * period * LINE_DT_NS / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    envelope = np.exp(-0.5 * (times_ns / sigma_ns) ** 2)
    return envelope * np.cos(2.0 * np.pi * F_C_HZ * 1e-9 * times_ns)


@pytest.mark.parametrize(
    ("fwhm_periods", "expected_runs"), [(1.0, 0), (2.5, 1), (4.0, 1)]
)
def test_long_layer_diagnostic_counts_layers_wider_than_2p(
    patch, fwhm_periods, expected_runs
):
    """Decision (a): long layers are counted but never block an injection."""
    base = _decaying_noise_data(0)
    base_context = prepare_line_context(_radargram(base))
    period = base_context.period_samples
    layer_row = round(_apex_sample_on(base_context))
    layer = 20.0 * DECAY_NOISE_RMS * _ringing_layer(layer_row, fwhm_periods, period)
    layered = prepare_line_context(_radargram(base + layer[:, None]))
    assert long_strong_layer_runs(layered) == expected_runs
    if expected_runs:
        result = inject_realization(layered, [_spec()], {"M12": _response()})
        reasons = [reason for _, reason in result.rejected]
        assert RejectReason.STRONG_LAYER_ZONE not in reasons


def test_long_layer_diagnostic_is_zero_on_noise():
    noise_context = prepare_line_context(_radargram(_decaying_noise_data(1)))
    assert long_strong_layer_runs(noise_context) == 0


def test_line_cost_reports_the_long_layer_count(context, patch):
    plan = plan_line_positions(context, [patch])
    cost = line_cost(context, plan, {"M12": patch}, n_conditions=10)
    assert cost.long_layer_runs == long_strong_layer_runs(context)


def test_rows_of_other_realizations_or_lines_are_refused(context, response):
    with pytest.raises(ValueError, match="one realization_id"):
        inject_realization(
            context, [_spec(), _spec(realization_id="x")], {"M12": response}
        )
    with pytest.raises(ValueError, match="other lines"):
        inject_realization(context, [replace(_spec(), line_id="x")], {"M12": response})


def test_distribution_is_width_only_at_the_given_level(context, injected):
    from gpr_detection.eval.inject import (
        HeightClass,
        WidthClass,
        height_class,
        width_class,
    )

    (truth,) = injected.truths
    for level in (-6.0, -10.0, -12.0, -20.0):
        assert truth.in_distribution(level) == is_in_distribution(
            truth.width_periods_by_level, truth.aperture_limited_by_level, level
        )
    assert not truth.bottom_stratum
    assert width_class(7.79, False) is WidthClass.IN
    assert width_class(14.67, False) is WidthClass.IN
    assert width_class(7.78, False) is WidthClass.WIDTH_UNDER
    assert width_class(14.68, False) is WidthClass.WIDTH_OVER
    assert width_class(10.0, True) is WidthClass.WIDTH_OVER_LOWER_BOUND
    # The height never filters; it is a report-only reason.
    assert is_in_distribution({-20.0: 10.0}, {-20.0: False}, -20.0)
    assert height_class(5.21) is HeightClass.HEIGHT_UNDER
    assert height_class(8.36) is HeightClass.HEIGHT_OVER
    assert height_class(6.0) is HeightClass.IN
    with pytest.raises(TypeError):
        is_in_distribution({-20.0: 10.0}, {-20.0: False})  # type: ignore[call-arg]


def test_level_boxes_grow_towards_lower_levels(patch):
    levels = patch.level_boxes
    assert list(levels) == [-6.0, -10.0, -12.0, -20.0]
    assert levels[-20.0].bbox == patch.wing_bbox
    widths = [levels[level].width_periods for level in levels]
    assert widths == sorted(widths)
    assert not levels[-6.0].aperture_limited


def test_aperture_limited_when_the_component_reaches_the_taper(context):
    wide = _response(beam_width_m=10.0)  # no lateral decay inside the aperture
    patch = build_response_patch(wide, context)
    assert isinstance(patch, ResponsePatch), patch
    assert patch.level_boxes[-20.0].aperture_limited


def test_visible_box_follows_the_snr(context, response):
    low = inject_realization(context, [_spec(snr_db=0.0)], {"M12": response})
    high = inject_realization(context, [_spec(snr_db=18.0)], {"M12": response})
    (low_truth,), (high_truth,) = low.truths, high.truths
    assert low_truth.visible_box.n_traces <= high_truth.visible_box.n_traces
    assert high_truth.visible_box.n_traces <= high_truth.wing_box.n_traces
    from gpr_detection.eval.inject import visible_amplitude

    assert visible_amplitude(0.0) == 0.5
    assert visible_amplitude(6.0) == pytest.approx(10 ** (-6 / 20))
    assert visible_amplitude(30.0) == 0.1


def test_first_reflection_mask_ends_one_period_before_the_multiple(response, context):
    from gpr_detection.eval.inject import first_reflection_mask

    apex_sample = 150.0
    mask = first_reflection_mask(
        np.ones((400, 5), bool), response, context, apex_sample
    )
    tau_m_ns = response.apex_time_ns - response.time_origin_ns
    last_row = apex_sample + (tau_m_ns - context.period_ns) / context.dt_ns
    first_row = apex_sample - 2.0 * context.period_samples
    rows = np.flatnonzero(mask[:, 0])
    assert rows[-1] == int(np.floor(last_row))
    assert rows[0] == int(np.ceil(first_row))
    partial = np.zeros((400, 5), bool)
    partial[140:150, 2] = True
    assert first_reflection_mask(partial, response, context, apex_sample).sum() == 10


@pytest.mark.parametrize(("material", "ringing"), [("pvc_water", True), ("pec", None)])
def test_water_filled_pvc_is_kept_with_a_ringing_flag(
    monkeypatch, context, material, ringing
):
    from gpr_detection.eval import inject

    monkeypatch.setattr(inject, "period_check_passes", lambda *_: False)
    patch = build_response_patch(_response(material=material), context)
    if ringing is None:
        assert patch == RejectReason.PERIOD_CHECK
        return
    assert isinstance(patch, ResponsePatch) and patch.period_ringing
    result = inject_realization(
        context, [_spec()], {"M12": _response(material=material)}
    )
    assert result.truths[0].period_ringing
    assert inject.is_water_filled_pvc("PVC_water") and not inject.is_water_filled_pvc(
        "pvc"
    )


def test_one_bottom_candidate_comes_first_then_the_seeded_order():
    from gpr_detection.eval import inject

    flags = [False, True, False, True, False, True]
    first = inject._bottom_first_order(flags, np.random.default_rng(1))
    again = inject._bottom_first_order(flags, np.random.default_rng(1))
    permutation = [int(i) for i in np.random.default_rng(1).permutation(6)]
    assert first == again
    leader = next(index for index in permutation if flags[index])
    assert first[0] == leader and inject.BOTTOM_PRIORITY_PER_LINE == 1
    assert first[1:] == [index for index in permutation if index != leader]


def test_plan_counts_the_natural_bottom_share_of_candidates(context, patch):
    plan = plan_line_positions(context, [patch])
    bottom, total = plan.candidates_by_response["M12"]
    assert total == plan.n_candidates and bottom in (0, total)


def _with_wing_end(patch: ResponsePatch, last_row: int) -> ResponsePatch:
    col0, col1, row0, _ = patch.wing_bbox
    return replace(patch, wing_bbox=(col0, col1, row0, last_row))


def test_wing_past_record_end_boundary_is_record_end_minus_one_period(context, patch):
    from gpr_detection.eval import inject

    limit = context.background.n_samples - 1 - context.period_samples
    on_limit = _with_wing_end(patch, math.floor(limit))
    past_limit = _with_wing_end(patch, math.floor(limit) + 1)
    assert not inject.is_wing_past_record_end(context, on_limit)
    assert inject.is_wing_past_record_end(context, past_limit)
    assert inject.is_main_cell(context, on_limit) == (
        not inject.is_box_clipped(context, on_limit)
    )
    assert not inject.is_main_cell(context, past_limit)


def test_box_clipped_patch_matches_the_truth_flag(context, patch, injected):
    from gpr_detection.eval import inject

    (truth,) = injected.truths
    assert inject.is_box_clipped(context, patch) == truth.box_clipped
    assert truth.wing_past_record_end == inject.is_wing_past_record_end(context, patch)
    assert truth.main_cell == inject.is_main_cell(context, patch)
    assert not replace(truth, wing_past_record_end=True).main_cell
    assert not replace(truth, box_clipped=True).main_cell


def test_non_main_cell_models_leave_the_pi_candidate_set(context, patch):
    limit = context.background.n_samples - 1 - context.period_samples
    cut = _with_wing_end(patch, math.floor(limit) + 1)
    plan = plan_line_positions(context, [cut])
    assert plan.n_candidates > 0 and plan.positions  # still planned (C-6 report)
    assert "M12" not in plan.candidates_by_response


def test_period_check_ignores_components_more_than_2p_before_the_apex(
    response, context, patch
):
    from gpr_detection.eval import inject

    wing = np.ones(patch.field.shape, dtype=bool)
    reference, _ = inject._first_reflection_period(
        patch.field, wing, response, context, patch.apex_sample
    )
    first_row = math.floor(patch.apex_sample - 2.0 * context.period_samples) - 1
    rows = np.arange(first_row)[:, None]
    precursor = patch.field.copy()
    precursor[:first_row] += (
        5.0
        * np.abs(patch.field).max()
        * np.sin(2 * np.pi * rows / (0.5 * context.period_samples))
    )
    measured, _ = inject._first_reflection_period(
        precursor, wing, response, context, patch.apex_sample
    )
    assert measured == reference
    mask = inject.first_reflection_mask(wing, response, context, patch.apex_sample)
    assert not mask[: first_row + 1].any()


# ---------------------------------------------------------- plan and table


def test_plan_is_seeded_spaced_and_capped(context, patch):
    first = plan_line_positions(context, [patch])
    assert first == plan_line_positions(context, [patch])
    assert plan_line_positions(context, [patch], seed=1) != first
    positions = first.positions
    assert 0 < len(positions) <= 15
    assert first.n_realizations <= MAX_REALIZATIONS_PER_LINE
    assert first.n_candidates > len(positions)
    all_traces = sorted(position.apex_trace for position in positions)
    assert np.all(np.diff(all_traces) >= patch.half_width_20db)
    realization_gap = patch.expected_width + context.period_traces
    for group in {position.group for position in positions}:
        traces = sorted(p.apex_trace for p in positions if p.group == group)
        assert np.all(np.diff(traces) >= realization_gap)


def test_plan_counts_positions_without_a_realization(context, patch):
    """235 traces hold 2 positions per realization: the rest is dropped and counted."""
    plan = plan_line_positions(context, [patch])
    assert plan.n_realizations == MAX_REALIZATIONS_PER_LINE
    assert len(plan.positions) <= 2 * MAX_REALIZATIONS_PER_LINE
    assert plan.n_dropped > 0


def test_specs_pair_every_position_with_every_condition(context, patch):
    plan = plan_line_positions(context, [patch], max_injections=4)
    conditions = default_conditions()
    specs = build_injection_specs(LINE_ID, plan.positions, conditions)
    assert len(conditions) == 10
    assert len(specs) == len(plan.positions) * len(conditions)
    assert len({spec.injection_id for spec in specs}) == len(specs)
    assert len({spec.realization_id for spec in specs}) == plan.n_realizations * 10


def test_cost_counts_l1_builds_and_touched_tiles(tmp_path: Path, context, patch):
    plan = plan_line_positions(context, [patch])
    cost = line_cost(context, plan, {"M12": patch}, n_conditions=10)
    assert cost.n_positions == len(plan.positions)
    assert cost.l1_builds == plan.n_realizations * 10
    assert 0 < cost.inferred_tiles <= len(context.tiles) * plan.n_realizations * 10
    table = tmp_path / "cost.csv"
    sha = write_cost_table(table, [cost])
    assert sha == hashlib.sha256(table.read_bytes()).hexdigest()
    lines = table.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("line_id,n_positions")
    assert lines[1].startswith(f"{LINE_ID},{cost.n_positions},")


def test_table_round_trip_and_sha(tmp_path: Path, context, patch):
    plan = plan_line_positions(context, [patch], max_injections=3)
    specs = build_injection_specs(LINE_ID, plan.positions, default_conditions())
    table = tmp_path / "injections.csv"
    sha = write_injection_table(table, specs)
    assert sha == hashlib.sha256(table.read_bytes()).hexdigest()
    read_specs, read_sha = read_injection_table(table)
    assert read_specs == specs
    assert read_sha == sha
    with pytest.raises(ValueError, match="unique"):
        write_injection_table(tmp_path / "dup.csv", [specs[0], specs[0]])


# --------------------------------------------- (j) strong layer (plan v0.2.3)

DECAY_NS = 15.0
DECAY_NOISE_RMS = 300.0
WINDOW_TRACES = 31


def _decaying_noise_data(seed: int) -> np.ndarray:
    """Band + Gaussian noise (Ricker-coloured, unit RMS) x exp(-t / 15 ns)."""
    rng = np.random.default_rng(seed)
    times_ns = np.arange(LINE_SAMPLES) * LINE_DT_NS
    band = BAND_AMPLITUDE * _ricker(times_ns - BAND_TIME_NS, F_C_HZ)
    wavelet = _ricker(np.arange(-40, 41) * LINE_DT_NS, F_C_HZ)
    coloured = convolve1d(rng.normal(size=(LINE_SAMPLES, LINE_TRACES)), wavelet, axis=0)
    coloured /= coloured.std()
    decay = np.exp(-times_ns / DECAY_NS)
    return band[:, None] + DECAY_NOISE_RMS * decay[:, None] * coloured


def _with_flat_layer(data: np.ndarray, layer_row: int, period: float) -> np.ndarray:
    """Add a flat 1 P reflector of 10 x the local per-trace RMS at ``layer_row``."""
    half = round(period)
    local = data[layer_row - half : layer_row + half + 1, :]
    local_rms = float(np.sqrt(np.mean((local - local.mean()) ** 2)))
    times_ns = (np.arange(LINE_SAMPLES) - layer_row) * LINE_DT_NS
    return data + 10.0 * local_rms * _ricker(times_ns, F_C_HZ)[:, None]


def _region_start(context) -> int:
    return int(np.ceil(context.zero.k_zero + 2.0 * context.period_samples))


def _windows() -> list[tuple[int, int]]:
    starts = range(0, LINE_TRACES - WINDOW_TRACES + 1, WINDOW_TRACES)
    return [(start, start + WINDOW_TRACES - 1) for start in starts]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_j_i_decaying_noise_has_at_most_1_percent_false_alarms(seed):
    noise_context = prepare_line_context(_radargram(_decaying_noise_data(seed)))
    region_rows = LINE_SAMPLES - _region_start(noise_context)
    flagged = sum(
        len(strong_layer_samples(noise_context, trace0, trace1))
        for trace0, trace1 in _windows()
    )
    assert flagged <= 0.01 * region_rows * len(_windows())


@pytest.mark.parametrize(
    "offset_periods",
    [-2.0, -1.4, -1.1, -0.9, -0.4, 0.0, 0.5, 0.95, 1.05, 1.3, 2.5],
)
def test_j_ii_flat_layer_is_flagged_and_excludes_only_within_1p(offset_periods):
    """Layer at apex + offset: flags stay within 1 P; rejected iff |offset| <= 1 P."""
    base = _decaying_noise_data(0)
    base_context = prepare_line_context(_radargram(base))
    period = base_context.period_samples
    layer_row = round(_apex_sample_on(base_context) + offset_periods * period)
    layered = prepare_line_context(
        _radargram(_with_flat_layer(base, layer_row, period))
    )
    flagged = strong_layer_samples(layered, 100, 130)
    assert flagged.size > 0
    assert np.all(np.abs(flagged - layer_row) <= layered.period_samples)
    result = inject_realization(layered, [_spec()], {"M12": _response()})
    rejected_by_layer = any(
        reason == RejectReason.STRONG_LAYER_ZONE for _, reason in result.rejected
    )
    assert rejected_by_layer == (abs(offset_periods) <= 1.0)


def _apex_sample_on(context) -> float:
    built = build_response_patch(_response(), context)
    assert isinstance(built, ResponsePatch)
    return built.apex_sample


def test_j_iii_attenuation_alone_does_not_flag_the_shallow_end():
    """Regression: the global-median rule flagged the top of the range."""
    trend_context = prepare_line_context(_radargram(_decaying_noise_data(3)))
    start = _region_start(trend_context)
    shallow_end = start + round(2.0 * trend_context.period_samples)
    for trace0, trace1 in _windows():
        flagged = strong_layer_samples(trend_context, trace0, trace1)
        assert not np.any(flagged < shallow_end)


def test_default_time_origin_is_the_ricker_transmit_peak():
    assert _response().time_origin_ns == pytest.approx(np.sqrt(2.0) / F_C_HZ * 1e9)
    assert _response(source_peak_ns=1.0).time_origin_ns == 1.0


# ----------------------------------------------------------- pseudo band


def test_pseudo_band_repeats_the_apex_trace_with_tapered_ends():
    apex_values = np.sin(np.linspace(0.0, 6.0, 50))
    background_trace = np.linspace(0.0, 1.0, 50)
    band = make_pseudo_band(apex_values, background_trace, 60, LINE_DX_M)
    assert band.shape == (50, 60)
    np.testing.assert_allclose(band[:, 30], apex_values + background_trace)
    edge_ratio = (band[:, 0] - background_trace) / np.where(
        apex_values == 0, 1, apex_values
    )
    assert np.all(np.abs(edge_ratio) < 0.05)


# ------------------------------------------ band mode (M3-3 R3, [EMR] D-5)
#
# (a) flat columns = hyperbola apex column bit for bit, (b) flat columns equal
# each other, (c) no ``_bg`` is added, (d) Monte Carlo survival references of
# [EMR] D-3, (e) QA-8 box round trip, (f) the hyperbola tests above.

BAND_LENGTH = 40
BAND_CENTER = 117.5
"""40 + 14 taper columns: an even column count needs a half-trace centre."""


def _band_spec(
    center: float = BAND_CENTER,
    *,
    length: int = BAND_LENGTH,
    index: int = 0,
    realization_id: str = "b",
    polarity: int = 1,
    snr_db: float = 12.0,
) -> InjectionSpec:
    return InjectionSpec(
        injection_id=f"{realization_id}:{index:02d}",
        realization_id=realization_id,
        line_id=LINE_ID,
        apex_trace=center,
        response_id="M12",
        snr_db=snr_db,
        polarity=polarity,
        k=K_MAIN,
        seed=0,
        mode=InjectionMode.BAND,
        band_length_traces=length,
        band_center_trace=center,
    )


@pytest.fixture(scope="module")
def band(patch, context) -> BandPatch:
    built = build_band_patch(patch, context, BAND_LENGTH)
    assert isinstance(built, BandPatch), built
    return built


@pytest.fixture(scope="module")
def band_injected(context, response) -> BandInjectionResult:
    return inject_band_realization(context, [_band_spec()], {"M12": response})


def test_band_a_flat_columns_are_the_hyperbola_apex_column_bitwise(patch, band):
    apex_column = patch.field[:, round(SYN_APEX_TRACE)]
    flat_first, flat_last = band.flat_columns
    for column in range(flat_first, flat_last + 1):
        assert np.array_equal(band.field[:, column], apex_column)
    assert band.signal_amplitude == patch.signal_amplitude
    assert band.apex_sample == patch.apex_sample


def test_band_b_flat_columns_equal_each_other_and_tapers_are_lower(band):
    flat_first, flat_last = band.flat_columns
    flat = band.field[:, flat_first : flat_last + 1]
    assert np.array_equal(flat, np.repeat(flat[:, :1], flat.shape[1], axis=1))
    peak_row = np.argmax(np.abs(band.field[:, flat_first]))
    edges = np.abs(band.field[peak_row, :flat_first])
    assert np.all(edges < np.abs(band.field[peak_row, flat_first]))


def test_band_length_is_between_the_minus_6db_taper_points(band):
    weights = edge_taper_weights(band.n_columns, band.taper_traces)
    first, last = band.box_columns
    assert last - first + 1 == BAND_LENGTH == band.length_traces
    assert weights[first] > 0.5 > weights[first - 1]
    assert weights[last] > 0.5 > weights[last + 1]
    assert band.n_columns == BAND_LENGTH + band.taper_traces


def test_band_c_background_trace_is_not_added(context, background, band_injected):
    (truth,) = band_injected.truths
    added = band_injected.data - background.data.astype(np.float64)
    apex_amplitude = abs(truth.gain) * truth.signal_amplitude
    band_limit = math.floor(context.zero.k_zero + 2.0 * context.period_samples)
    assert np.abs(added[:band_limit]).max() <= 0.02 * apex_amplitude
    assert np.abs(added[:5]).max() <= 1e-3 * apex_amplitude
    outside = np.ones(added.shape[1], dtype=bool)
    offset = round(BAND_CENTER - (BAND_LENGTH + truth.taper_traces - 1) / 2)
    outside[offset : offset + BAND_LENGTH + truth.taper_traces] = False
    assert not added[:, outside].any()


def test_band_taper_is_the_fresnel_radius_rounded_to_even_traces(
    context, response, band
):
    wavelength_m = 0.1 * context.period_ns  # no eps_r metadata: 0.1 m/ns
    fresnel_m = math.sqrt(wavelength_m * response.top_depth_m / 2.0)
    assert fresnel_m > 0.2
    expected = 2 * round(fresnel_m / LINE_DX_M / 2.0)
    assert band_taper_traces(response, context) == expected == band.taper_traces
    shallow = _response(apex_ns=SYN_REFERENCE_NS + 4.0)  # d 0.2 m: r_F < 0.2 m
    assert band_taper_traces(shallow, context) == round(0.2 / LINE_DX_M)


def test_band_shorter_than_its_taper_is_rejected(patch, context, band):
    short = build_band_patch(patch, context, band.taper_traces)
    assert short == RejectReason.BAND_SHORTER_THAN_TAPER
    with pytest.raises(ValueError, match="even"):
        build_band_patch(patch, context, BAND_LENGTH, taper_traces=13)


def test_band_snr_uses_sigma_over_the_flat_traces(context, background, band_injected):
    (truth,) = band_injected.truths
    ratio = abs(truth.gain) * truth.signal_amplitude / truth.noise_rms
    assert ratio == pytest.approx(10.0 ** (12.0 / 20.0))
    period = context.period_samples
    rows = slice(
        math.floor(truth.apex_sample - period),
        math.ceil(truth.apex_sample + period) + 1,
    )
    flat = slice(
        truth.box.i0 + truth.taper_traces // 2,
        truth.box.i1 - truth.taper_traces // 2 + 1,
    )
    expected = np.sqrt(np.mean(context.background_removed[rows, flat] ** 2))
    assert truth.noise_rms == pytest.approx(expected)
    injected_only = band_injected.data - background.data.astype(np.float64)
    envelope = _padded_envelope(_linear_steps(injected_only))
    centre = round(truth.center_trace)
    assert envelope[:, centre].max() == pytest.approx(
        abs(truth.gain) * truth.signal_amplitude, rel=1e-6
    )


def test_band_polarity_pair_is_an_exact_negation(background, context, response):
    responses = {"M12": response}
    plus = inject_band_realization(context, [_band_spec(polarity=1)], responses)
    minus = inject_band_realization(
        context, [_band_spec(polarity=-1, realization_id="m")], responses
    )
    (plus_truth,), (minus_truth,) = plus.truths, minus.truths
    assert plus_truth.gain == -minus_truth.gain
    assert plus_truth.box == minus_truth.box
    raw = background.data.astype(np.float64)
    added = plus.data - raw
    tolerance = 1e-12 * np.max(np.abs(added)) + 1e-9
    np.testing.assert_allclose(added, -(minus.data - raw), rtol=0, atol=tolerance)


def test_band_truth_records_lengths_and_windows(context, band, band_injected):
    (truth,) = band_injected.truths
    assert truth.length_traces == BAND_LENGTH
    assert truth.box.n_traces == BAND_LENGTH
    assert truth.center_trace == (truth.box.i0 + truth.box.i1) / 2.0 == BAND_CENTER
    assert truth.length_px == BAND_LENGTH / context.grid_map.traces_per_px
    assert truth.w_eff == LINE_TRACES  # the line is shorter than W = 301
    assert truth.l_over_w_eff == truth.l_over_n == BAND_LENGTH / LINE_TRACES
    assert (truth.box.k0, truth.box.k1) == (band.row0, band.row1)
    assert truth.box.k0 <= truth.apex_sample <= truth.box.k1
    assert truth.main_cell and not truth.bottom_stratum
    assert truth.n_clipped_cells == 0 and truth.max_clipped_injection_ratio == 0.0


def test_band_vertical_box_is_the_apex_column_minus_20db_run(patch, band):
    envelope = _padded_envelope(_linear_steps(band.field))
    column = envelope[:, band.flat_columns[0]]
    peak = column.max()
    assert column[band.row0] >= 0.1 * peak and column[band.row1] >= 0.1 * peak
    assert column[band.row0 - 1] < 0.1 * peak and column[band.row1 + 1] < 0.1 * peak


def test_band_e_truth_boxes_round_trip_exactly_through_pipeline(context, response):
    responses = {"M12": response}
    for center, length in ((BAND_CENTER, 40), (117.0, 41), (110.5, 64)):
        spec = _band_spec(center, length=length, realization_id=f"e{length}")
        (truth,) = inject_band_realization(context, [spec], responses).truths
        line_box = grid_box_to_image_box(truth.box, context.grid_map)
        assert image_box_to_grid_box(line_box, context.grid_map) == truth.box


def test_band_centre_off_the_column_grid_is_rejected(context, response):
    spec = _band_spec(BAND_CENTER + 0.5)
    reason = _single_band_rejection(context, [spec], {"M12": response})
    assert reason == RejectReason.OFF_GRID_APEX


def _single_band_rejection(context, specs, responses) -> str:
    result = inject_band_realization(context, specs, responses)
    assert result.truths == ()
    ((_, reason),) = result.rejected
    return reason


def test_band_zone_c_i_box_more_than_2p_above_does_not_block(
    background, context, response, band_injected
):
    """(c') is 2-D: same traces, more than 2 P shallower -> not blocked."""
    (truth,) = band_injected.truths
    period = context.period_samples
    last_row = math.floor(truth.box.k0 - 2.0 * period) - 1
    above = GridBox(i0=truth.box.i0, i1=truth.box.i1, k0=0, k1=last_row)
    result = inject_band_realization(
        _box_context(background, above), [_band_spec()], {"M12": response}
    )
    (kept,) = result.truths
    assert kept.box == truth.box
    assert kept.in_horizontal_box_zone  # the old horizontal rule blocked it
    assert not truth.in_horizontal_box_zone


@pytest.mark.parametrize("where", ["inside", "beside"])
def test_band_zone_c_ii_box_at_the_band_depth_blocks(
    background, context, response, band_injected, where
):
    """(c') blocks a box whose depth overlaps the truth box (inside or +- 1 P)."""
    (truth,) = band_injected.truths
    rows = {"k0": round(truth.apex_sample), "k1": round(truth.apex_sample) + 3}
    if where == "inside":
        box = GridBox(i0=truth.box.i0 + 5, i1=truth.box.i0 + 7, **rows)
    else:
        start = truth.box.i1 + math.floor(context.period_traces)
        box = GridBox(i0=start, i1=start + 2, **rows)
    assert (
        _single_band_rejection(
            _box_context(background, box), [_band_spec()], {"M12": response}
        )
        == RejectReason.UNINJECTED_BOX_ZONE
    )


def test_band_zone_c_grows_boxes_by_one_period_in_samples(
    background, context, response, band_injected
):
    (truth,) = band_injected.truths
    period = context.period_samples
    touching_k1 = math.ceil(truth.box.k0 - period)
    touching = GridBox(i0=truth.box.i0, i1=truth.box.i1, k0=0, k1=touching_k1)
    assert (
        _single_band_rejection(
            _box_context(background, touching), [_band_spec()], {"M12": response}
        )
        == RejectReason.UNINJECTED_BOX_ZONE
    )
    clear = replace(touching, k1=math.floor(truth.box.k0 - period) - 1)
    result = inject_band_realization(
        _box_context(background, clear), [_band_spec()], {"M12": response}
    )
    assert len(result.truths) == 1


def test_band_zone_d_measures_the_k_radius_from_the_span_ends(
    background, response, band, band_injected
):
    (truth,) = band_injected.truths
    near = truth.box.i1 + band.patch.half_width_20db - 0.5
    blocked = prepare_line_context(background, LineExclusions(k_apex_traces=(near,)))
    assert (
        _single_band_rejection(blocked, [_band_spec()], {"M12": response})
        == RejectReason.CANDIDATE_APEX_ZONE_K
    )
    far = truth.box.i1 + band.patch.half_width_20db + 0.5
    allowed = prepare_line_context(background, LineExclusions(k_apex_traces=(far,)))
    result = inject_band_realization(allowed, [_band_spec()], {"M12": response})
    assert len(result.truths) == 1


def test_band_depth_zones_follow_the_hyperbola_rules(context, patch):
    shallow = _response(apex_ns=SYN_REFERENCE_NS + 3.0)
    spec = _band_spec()
    assert (
        _single_band_rejection(context, [spec], {"M12": shallow})
        == RejectReason.BAND_ZONE
    )
    layer_data = _background_data()
    layer_row = round(patch.apex_sample) + 10
    times_ns = (np.arange(LINE_SAMPLES) - layer_row) * LINE_DT_NS
    layer_data += 20.0 * TEXTURE_SCALE * _ricker(times_ns, F_C_HZ)[:, None]
    layered = prepare_line_context(_radargram(layer_data))
    assert (
        _single_band_rejection(layered, [spec], {"M12": _response()})
        == RejectReason.STRONG_LAYER_ZONE
    )


def test_band_spacing_is_half_lengths_plus_w_eff(context, response):
    assert band_spacing_ok(400.0, 40, [(100.0, 60)], 250)  # 300 = 50 + 250
    assert not band_spacing_ok(399.9, 40, [(100.0, 60)], 250)
    assert band_spacing_ok(100.0, 40, [], 250)
    specs = [_band_spec(80.5), _band_spec(BAND_CENTER + 40.0, index=1)]
    result = inject_band_realization(context, specs, {"M12": response})
    assert [truth.injection_id for truth in result.truths] == ["b:00"]
    assert result.rejected == (("b:01", RejectReason.BAND_SPACING.value),)


def test_band_taper_uses_the_line_eps_r_when_known(background, response):
    eps_context = prepare_line_context(replace(background, eps_r_meta=4.0))
    velocity = 0.299792458 / 2.0
    fresnel_m = math.sqrt(velocity * eps_context.period_ns * response.top_depth_m / 2)
    expected = 2 * round(fresnel_m / LINE_DX_M / 2.0)
    assert band_taper_traces(response, eps_context) == expected


def test_band_zone_d_auto_candidates_and_line_ends(
    background, context, response, band, band_injected
):
    (truth,) = band_injected.truths
    radius = band.patch.half_width_6db + context.period_traces
    near = truth.box.i0 - radius + 0.5
    blocked = prepare_line_context(
        background, LineExclusions(auto_candidate_apex_traces=(near,))
    )
    assert (
        _single_band_rejection(blocked, [_band_spec()], {"M12": response})
        == RejectReason.CANDIDATE_APEX_ZONE_AUTO
    )
    past_end = LINE_TRACES - 10.5
    assert (
        _single_band_rejection(context, [_band_spec(past_end)], {"M12": response})
        == RejectReason.APERTURE_OUTSIDE_LINE
    )


def test_band_int16_clipping_inside_the_truth_box_rejects():
    data = _background_data()
    data[:, round(BAND_CENTER) : round(BAND_CENTER) + 3] = 32766.6
    near_limit = prepare_line_context(_radargram(data))
    spec = _band_spec(snr_db=18.0)
    reason = _single_band_rejection(near_limit, [spec], {"M12": _response()})
    assert reason == RejectReason.INT16_OVERFLOW


def test_band_int16_far_above_the_apex_is_kept(patch):
    clipped_context = prepare_line_context(
        _radargram(_saturated_above_apex(patch, 2.5))
    )
    result = inject_band_realization(
        clipped_context, [_band_spec(snr_db=18.0)], {"M12": _response()}
    )
    (truth,) = result.truths
    assert truth.n_clipped_cells > 0
    assert 0 < truth.max_clipped_injection_ratio <= 0.05
    assert result.data.max() <= 32767 and result.data.min() >= -32768


def test_band_and_hyperbola_rows_never_share_a_realization(context, response):
    responses = {"M12": response}
    band_row = _band_spec(realization_id="r", index=1)
    with pytest.raises(ValueError, match="hyperbola realization"):
        inject_realization(context, [_spec(), band_row], responses)
    with pytest.raises(ValueError, match="band realization"):
        inject_band_realization(context, [band_row, _spec()], responses)


def test_band_spec_fields_are_checked():
    with pytest.raises(ValueError, match="length and centre"):
        replace(_band_spec(), band_length_traces=None)
    with pytest.raises(ValueError, match="must equal apex_trace"):
        replace(_band_spec(), band_center_trace=BAND_CENTER + 1.0)
    with pytest.raises(ValueError, match="hyperbola row"):
        replace(_spec(), band_length_traces=40)
    as_text = replace(_band_spec(), mode="band")  # type: ignore[arg-type]
    assert as_text.mode is InjectionMode.BAND


def test_band_table_round_trip_and_legacy_header(tmp_path: Path):
    hyperbola_rows = [_spec(index=0), _spec(index=1, polarity=-1)]
    legacy = tmp_path / "legacy.csv"
    write_injection_table(legacy, hyperbola_rows)
    header = legacy.read_text(encoding="utf-8").splitlines()[0]
    assert tuple(header.split(",")) == INJECTION_TABLE_COLUMNS
    assert read_injection_table(legacy)[0] == hyperbola_rows
    mixed = [*hyperbola_rows, _band_spec(), _band_spec(117.0, length=41, index=1)]
    table = tmp_path / "bands.csv"
    sha = write_injection_table(table, mixed)
    lines = table.read_text(encoding="utf-8").splitlines()
    assert tuple(lines[0].split(",")) == EXTENDED_INJECTION_TABLE_COLUMNS
    assert lines[1].endswith(",hyperbola,,")
    assert lines[3].endswith(",band,40,117.5")
    read_specs, read_sha = read_injection_table(table)
    assert read_specs == mixed and read_sha == sha


def test_band_survival_ratio_of_a_short_band_is_high(context, band_injected):
    """L 40 on a 235-trace line (global median): [EMR] D-3 gives 0.93 at 12 dB."""
    (truth,) = band_injected.truths
    survival = band_survival_ratio(context, band_injected.data, truth)
    assert 0.8 <= survival <= 1.05


MC_TRACES = 3000
MC_ROWS = 96
MC_PERIOD = 25.0
MC_TAPER = 14
MC_SEEDS = 5


def _mc_unit_column() -> np.ndarray:
    """Gabor column (carrier 1 / P, FWHM 1.3 P) with envelope peak 1."""
    rows = np.arange(MC_ROWS) - MC_ROWS / 2
    sigma = 1.3 * MC_PERIOD / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    column = np.cos(2 * np.pi * rows / MC_PERIOD) * np.exp(-0.5 * (rows / sigma) ** 2)
    return column / _padded_envelope(column[:, None]).max()


def _mc_survival(length: int, snr_db: float) -> float:
    """[EMR] D-3 Monte Carlo through ``band_survival_from_sections``.

    White noise (sigma 1) on a 3,000-trace line, moving median W 301, band of
    ``L`` traces between the -6 dB taper points, mean over 5 seeds.
    """
    amplitude = 10.0 ** (snr_db / 20.0)
    n_columns = length + MC_TAPER
    first = MC_TRACES // 2 - n_columns // 2
    band_rows = amplitude * _mc_unit_column()[:, None]
    band_field = band_rows * edge_taper_weights(n_columns, MC_TAPER)[None, :]
    span = BandSurvivalSpan(
        apex_sample=MC_ROWS / 2,
        period_samples=MC_PERIOD,
        trace0=first + MC_TAPER // 2,
        trace1=first + MC_TAPER // 2 + length - 1,
        reference_amplitude=amplitude,
    )
    ratios = []
    for seed in range(MC_SEEDS):
        noise = np.random.default_rng(seed).normal(size=(MC_ROWS, MC_TRACES))
        injected = noise.copy()
        injected[:, first : first + n_columns] += band_field
        ratios.append(band_survival_from_sections(injected, noise, span, 301))
    return float(np.mean(ratios))


def test_band_d_monte_carlo_short_band_survives():
    assert _mc_survival(40, 18.0) >= 0.85


@pytest.mark.parametrize("snr_db", [0.0, 18.0])
def test_band_d_monte_carlo_long_band_is_erased(snr_db):
    assert _mc_survival(400, snr_db) <= 0.05
