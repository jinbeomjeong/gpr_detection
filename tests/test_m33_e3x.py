"""Tests of the M3-3 E3x tools (plan ``docs/processing/m33_e3x_plan.md`` A-5).

Synthetic lines and responses only (``tests/test_inject.py`` builders); no
dataset, detector output or GPU. Ultralytics is not imported (AGPL boundary):
NMS identity is precheck 6's job on the real model.
"""

import importlib.util
import json
import logging
import math
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "detection"))
import m32_recipe_inputs as inputs
import m33_e3x as e3x
import m33_e3x_judge as judge
import m33_e3x_place as place
import m33_e3x_pool as pool
import m33_e3x_seal2_r1_draft as seal_draft

from gpr_detection.detect.base import PixelBox
from gpr_detection.eval import inject
from gpr_detection.io.classes import DetectionClass
from gpr_detection.io.radargram import GridBox
from gpr_detection.pipeline import (
    DEFAULT_MERGE_CONFIG,
    PP_SETS,
    LineBox,
    PPSetId,
    merge_line_boxes,
)
from gpr_detection.preprocess.to_image import make_detection_image

REPO = Path(__file__).resolve().parents[1]
PROCESSING = REPO / "analysis" / "processing"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helpers() -> ModuleType:
    """``tests/test_inject.py`` (synthetic line and response builders)."""
    return _load(Path(__file__).resolve().parent / "test_inject.py", "e3x_helpers")


@pytest.fixture(scope="module")
def context(helpers) -> inject.LineContext:
    return inject.prepare_line_context(helpers._radargram(helpers._background_data()))


@pytest.fixture(scope="module")
def long_context(helpers) -> inject.LineContext:
    """1,880 traces: room for two band positions (W_eff 301)."""
    data = np.tile(helpers._background_data(), (1, 8))
    return inject.prepare_line_context(helpers._radargram(data))


@pytest.fixture(scope="module")
def patch(helpers, context) -> inject.ResponsePatch:
    built = inject.build_response_patch(helpers._response(), context, inject.K_MAIN)
    assert isinstance(built, inject.ResponsePatch)
    return built


# ------------------------------------------------------------- 1-3 shifts


def test_1_shifted_response_moves_the_patch_by_whole_samples(helpers, context, patch):
    response = helpers._response()
    same = inject.build_response_patch(
        place.shifted_response(response, 0.0), context, inject.K_MAIN
    )
    assert isinstance(same, inject.ResponsePatch)
    assert np.array_equal(same.field, patch.field)
    n = 17
    moved = inject.build_response_patch(
        place.shifted_response(response, n * context.dt_ns), context, inject.K_MAIN
    )
    assert isinstance(moved, inject.ResponsePatch)
    assert moved.response.top_depth_m == response.top_depth_m
    assert abs(moved.apex_sample - patch.apex_sample - n) < 1e-9
    rows = min(patch.field.shape[0], moved.field.shape[0] - n)
    gap = np.abs(moved.field[n : n + rows] - patch.field[:rows]).max()
    assert gap <= 1e-9 * np.abs(patch.field).max()


SHALLOW_LIFT_SAMPLES = 128
"""Synthetic shallow native band: top at sample 18 (0.75 P), re-measured step -1."""


def _shallow_response(helpers, context) -> inject.PipeResponse:
    return place.shifted_response(
        helpers._response(), -SHALLOW_LIFT_SAMPLES * context.dt_ns
    )


def _shift_inputs(context, response):
    taper = place.position_taper_traces(response, context)
    length = place.band_length_traces(context, 13)
    native = inject.build_response_patch(response, context, inject.K_MAIN)
    return taper, length, place.native_row0(context, native, length, taper)


def _counting(monkeypatch) -> list[int]:
    """Count the shifted builds (``_shifted_top`` calls) and their shifts."""
    calls: list[int] = []
    real = place._shifted_top

    def counted(context, response, k, shift, length, taper):
        calls.append(shift)
        return real(context, response, k, shift, length, taper)

    monkeypatch.setattr(place, "_shifted_top", counted)
    return calls


def test_2a_corrected_top_lands_on_every_row(helpers, context):
    response = helpers._response()
    taper, length, _ = _shift_inputs(context, response)
    for row in place.TARGET_ROWS_PX:
        out = place.band_shift_for(context, response, inject.K_MAIN, row, taper, length)
        assert isinstance(out, tuple), (row, out)
        shift, band = out
        assert abs(band.row0 - place.target_sample(context, row)) <= 1.0
        assert shift.shift_periods <= place.SHIFT_LIMIT_P


def test_2b_shallow_native_top_after_correction(helpers, context):
    response = _shallow_response(helpers, context)
    taper, length, row0 = _shift_inputs(context, response)
    assert row0 <= context.period_samples  # within 1 P of the record start
    out = place.band_shift_for(context, response, inject.K_MAIN, 47.0, taper, length)
    shift, band = out
    assert shift.shift_step_samples != 0
    assert abs(band.row0 - place.target_sample(context, 47.0)) <= 1.0


@pytest.mark.data
def test_2b_real_shallow_line_m01():
    """[PDD] case: IDS ``pipe/1030_190118AC/LID10001`` M01 (top 0.80 P, step +1)."""
    import m32_recipe_eval as runner

    line_id = "pipe/1030_190118AC/LID10001"
    windows = {
        r["line"]: float(r["time_window_ns"])
        for r in inputs.read_csv(inputs.STAGE_A_IDS)
    }
    ref = inputs.LineRef(inputs.DATASET_IDS, line_id, "background", windows[line_id])
    try:
        line = runner.load_line(ref)
        library = e3x.load_hyperbola_library()
    except (OSError, ValueError, KeyError) as error:
        pytest.skip(f"dataset not readable: {error}")
    context = inject.prepare_line_context(line.radargram)
    response = e3x.line_responses(library, line, e3x.HYPERBOLA_MODELS)["M01"]
    taper, length, _ = _shift_inputs(context, response)
    shift, band = place.band_shift_for(
        context, response, inject.K_MAIN, 28.0, taper, length
    )
    assert shift.shift_step_samples == 1 and shift.shift_correction_samples == -1
    assert abs(band.row0 - place.target_sample(context, 28.0)) <= 1.0


def test_2c_correction_path(helpers, context, monkeypatch):
    calls = _counting(monkeypatch)
    response = helpers._response()
    taper, length, row0 = _shift_inputs(context, response)
    row = place.row_shift_for(
        context, response, inject.K_MAIN, 47.0, row0, taper, length
    )
    assert row.shift_step_samples == 0 and row.shift_samples == row.shift_samples_first
    assert len(calls) == 1  # no correction: one build
    calls.clear()
    shallow = _shallow_response(helpers, context)
    taper, length, row0 = _shift_inputs(context, shallow)
    row = place.row_shift_for(
        context, shallow, inject.K_MAIN, 47.0, row0, taper, length
    )
    k_target = place.target_sample(context, 47.0)
    first = round(k_target - row0)
    first_top = row0 + first + row.shift_step_samples
    assert row.shift_samples_first == first
    assert row.shift_samples == first + round(k_target - first_top)
    assert row.shift_correction_samples == -row.shift_step_samples != 0
    assert calls == [first, row.shift_samples]  # built twice, corrected once


def test_2d_limit_uses_the_corrected_shift(helpers, context, monkeypatch):
    assert place.within_shift_limit(24, 2.0)  # exactly 12 P: allowed
    assert not place.within_shift_limit(25, 2.0)
    response = helpers._response()
    taper, length, row0 = _shift_inputs(context, response)
    patch = inject.build_response_patch(response, context, inject.K_MAIN)
    limit = math.floor(place.SHIFT_LIMIT_P * context.period_samples)
    target_row = place.row_px_of(context, row0 + limit)

    def fake(context_, response_, k, shift, length_, taper_):
        # The re-measured top lies 3 samples high: n2 = n1 + 3 > 12 P.
        return place._ShiftedTop(response_, patch, row0 + shift - 3)

    monkeypatch.setattr(place, "_shifted_top", fake)
    out = place.row_shift_for(
        context, response, inject.K_MAIN, target_row, row0, taper, length
    )
    assert out is place.ShiftReject.SHIFT_LIMIT  # n1 inside, only n2 outside


def test_2e_a_two_sample_miss_is_a_mismatch(helpers, context):
    response = helpers._response()
    taper, length, _ = _shift_inputs(context, response)
    row, band = place.band_shift_for(
        context, response, inject.K_MAIN, 47.0, taper, length
    )
    samples_per_px = context.grid_map.samples_per_px
    gap = band.row0 - place.target_sample(context, 47.0)
    for offset, expected_ok in ((1.0, True), (2.0, False)):
        moved = replace(row, target_row_px=47.0 + (gap + offset) / samples_per_px)
        out = place.shifted_band(context, moved, length, taper)
        assert isinstance(out, inject.BandPatch) is expected_ok, offset
    assert out is place.ShiftReject.SHIFT_MISMATCH


def test_3_one_taper_per_position_and_depth_unchanged(helpers, long_context):
    response = helpers._response()
    plan = place.plan_band_positions_e3x(long_context, {"M12": response})
    assert plan.positions and not plan.n_shift_mismatch
    expected = inject.band_taper_traces(response, long_context)
    for position in plan.positions:
        assert position.taper_traces == expected
        assert place.offsets_parallel([arm.shift for arm in position.arms])
        for arm in position.arms:
            assert arm.band.taper_traces == expected
            assert arm.shift.response.top_depth_m == response.top_depth_m


def test_3b_arm_parallel_assertion_and_short_centre(helpers, long_context, monkeypatch):
    real = place.row_shift_for

    def one_arm_off(context, response, k, row_px, *rest):
        row = real(context, response, k, row_px, *rest)
        if isinstance(row, place.RowShift) and row_px == 68.0:
            return replace(row, shifted_row0=row.shifted_row0 + 1)
        return row

    monkeypatch.setattr(place, "row_shift_for", one_arm_off)
    plan = place.plan_band_positions_e3x(long_context, {"M12": helpers._response()})
    assert not plan.positions
    assert plan.model_reasons["M12"] == "arm_offsets_not_parallel"
    assert plan.n_shift_mismatch > 0
    assert not any(item["arm_offsets_parallel"] for item in plan.shift_mismatches)
    assert place.short_arm_center(100.0, 90, 50) == 100.0  # same parity
    assert place.short_arm_center(100.0, 91, 50) == 99.5  # parity differs


# -------------------------------------------------------------- 4 texture


def test_4_texture_field_amplitude_and_jitter():
    flat = (14, 76)
    w = place.texture_field(91, flat, 5.86, 3.0, place.texture_rng("p"))
    again = place.texture_field(91, flat, 5.86, 3.0, place.texture_rng("p"))
    assert np.array_equal(w, again)
    flat_w = w[flat[0] : flat[1] + 1]
    assert abs(flat_w.mean()) <= 1e-12 and abs(flat_w.std() - 1) <= 1e-12
    m = place.amplitude_texture(w, flat, 2.5)
    assert abs(np.mean(m[flat[0] : flat[1] + 1] ** 2) - 1) <= 1e-12
    # Generator unit: >= 1,000 columns, autocorrelation SD sqrt(2) l_c (+-10 %).
    period_traces, corr = 4.0, 3.0
    long = place.texture_field(
        60_000, (0, 59_999), period_traces, corr, np.random.default_rng(1)
    )
    sigma = corr * period_traces
    lags = np.arange(1, int(2 * sigma))
    rho = np.array([np.mean(long[:-k] * long[k:]) for k in lags])
    estimated = np.median(lags / np.sqrt(-2 * np.log(rho)))
    assert abs(estimated / (math.sqrt(2) * sigma) - 1) <= 0.10
    # Jitter: integer delay = np.roll; energy kept on a band-limited column.
    rng = np.random.default_rng(2)
    column = rng.standard_normal(256)
    delays = np.full(5, 3.0)
    shifted = place.time_jitter(column, delays / (0.5 * 6.0), 0.5, 6.0)
    assert np.abs(shifted[:, 0] - np.roll(column, 3)).max() <= 1e-9
    t = np.arange(512)
    smooth = np.exp(-(((t - 256) / 30.0) ** 2)) * np.cos(2 * np.pi * t / 25.0)
    moved = place.time_jitter(smooth, w[:5], 0.2, 25.0)
    energy = np.sum(moved**2, axis=0) / np.sum(smooth**2)
    assert np.abs(energy - 1).max() <= 1e-9


def test_4b_texture_check_passes_on_a_real_band(helpers, long_context):
    plan = place.plan_band_positions_e3x(long_context, {"M12": helpers._response()})
    arm = plan.positions[0].arm_of(47.0, place.LONG_ARM)
    band = arm.band
    w = place.texture_field(
        band.n_columns,
        band.flat_columns,
        long_context.period_traces,
        3.0,
        place.texture_rng("x"),
    )
    m = place.amplitude_texture(w, band.flat_columns, 2.5)
    check = place.texture_check(band, w, m, long_context.period_samples)
    assert check.passes(), check
    column = place.apex_column(band)
    assert check.jitter_nyquist_fraction == place.nyquist_fraction(column)
    assert set(check.jitter_energy_identity) == set(place.JITTER_PERIODS)
    record = check.as_record()
    assert record["jitter_wrap_rel@0.2"] == check.jitter_wrap_rel[0.2]
    assert 0 < record["jitter_start_step_rel"] <= 1


NYQUIST_FIXTURE_ROWS = 64


def _nyquist_column(amplitude: float, n_rows: int = NYQUIST_FIXTURE_ROWS):
    """Low-frequency pulse + ``a (-1)^k``: ``r_N`` close to ``a^2 n / E``."""
    k = np.arange(n_rows)
    pulse = np.exp(-(((k - n_rows / 2) / 6.0) ** 2)) * np.cos(2 * np.pi * k / 16.0)
    return pulse + amplitude * (-1.0) ** k


def test_4c_time_jitter_energy_identity_wrap_and_step():
    # delta = 0 gives the column back.
    column = _nyquist_column(1e-3)
    still = place.time_jitter(column, np.zeros(3), 0.2, 10.0)
    assert np.abs(still - column[:, None]).max() <= 1e-12
    # Odd length: no Nyquist bin, the energy is kept for every delay.
    odd = np.random.default_rng(3).standard_normal(63)
    moved = place.time_jitter(odd, np.linspace(-2.0, 2.0, 9), 1.0, 1.0)
    energy = np.sum(moved**2, axis=0) / np.sum(odd**2)
    assert place.nyquist_fraction(odd) == 0.0
    assert np.abs(energy - 1).max() <= 1e-12
    # Even length: E'/E = 1 - r_N sin^2(pi delta); delta = 0.5 loses r_N.
    delays = np.array([0.0, 0.25, 0.5, 0.75, 1.3, -0.4])
    r_n = place.nyquist_fraction(column)
    hand = 1e-3**2 * NYQUIST_FIXTURE_ROWS / np.sum(column**2)
    assert abs(r_n / hand - 1) <= 1e-6
    shifted = place.time_jitter(column, delays, 1.0, 1.0)
    ratio = np.sum(shifted**2, axis=0) / np.sum(column**2)
    assert np.abs(ratio - 1 + r_n * np.sin(np.pi * delays) ** 2).max() <= 1e-12
    assert abs((1 - ratio[2]) - r_n) <= 1e-12
    # Start step: r_N has the order of the boundary term (s/2)^2 / (n E).
    t = np.arange(512)
    stepped = np.exp(-t / 20.0) * np.cos(2 * np.pi * t / 40.0)
    approx = (stepped[0] / 2) ** 2 / (t.size * np.sum(stepped**2))
    assert 0.5 <= place.nyquist_fraction(stepped) / approx <= 2.0
    assert place.start_step_fraction(stepped) == 1.0
    # The linear (4n zero pad) shift has no wrap: an integer delay moves the
    # column down and leaves zeros above it.
    linear = place.linear_time_jitter(stepped, np.array([3.0]), 1.0, 1.0)[:, 0]
    assert np.abs(linear[:3]).max() <= 1e-12
    assert np.abs(linear[3:] - stepped[:-3]).max() <= 1e-9


def _band_fixture(helpers, long_context):
    plan = place.plan_band_positions_e3x(long_context, {"M12": helpers._response()})
    band = plan.positions[0].arm_of(47.0, place.LONG_ARM).band
    w = place.texture_field(
        band.n_columns,
        band.flat_columns,
        long_context.period_traces,
        3.0,
        place.texture_rng("x"),
    )
    return band, w, place.amplitude_texture(w, band.flat_columns, 2.5)


def test_4d_texture_check_nyquist_bound_and_b1_mock(helpers, long_context, monkeypatch):
    band, w, m = _band_fixture(helpers, long_context)
    period = long_context.period_samples
    columns = {"over": _nyquist_column(1e-3), "under": _nyquist_column(1e-4)}
    monkeypatch.setattr(place, "apex_column", lambda _band: columns["over"])
    over = place.texture_check(band, w, m, period)
    assert over.jitter_nyquist_fraction > place.JITTER_NYQUIST_FRACTION_MAX
    assert max(over.jitter_energy_identity.values()) <= 1e-12
    assert not over.passes()
    monkeypatch.setattr(place, "apex_column", lambda _band: columns["under"])
    under = place.texture_check(band, w, m, period)
    assert 1e-8 < under.jitter_nyquist_fraction <= place.JITTER_NYQUIST_FRACTION_MAX
    assert under.passes(), under
    # A shift that drops the Nyquist bin (b1-type defect) loses r_N on every
    # column whatever the delay: the identity catches it, the r_N bound not.
    sealed = place.time_jitter

    def nyquist_dropped(column, w_, sigma, period_):
        spectrum = np.fft.rfft(column)
        spectrum[-1] = 0.0
        flat = np.fft.irfft(spectrum, n=column.shape[0])
        return sealed(flat, w_, sigma, period_)

    monkeypatch.setattr(place, "time_jitter", nyquist_dropped)
    broken = place.texture_check(band, w, m, period)
    assert broken.jitter_nyquist_fraction == under.jitter_nyquist_fraction
    assert max(broken.jitter_energy_identity.values()) > 1e-12
    assert not broken.passes()


# ------------------------------------------------------- 5 end treatment


def _spec(
    helpers, apex: float | None = None, snr: float = 18.0, rid: str = "r"
) -> inject.InjectionSpec:
    apex = helpers.APEX_TRACE if apex is None else apex
    return inject.InjectionSpec(
        f"{rid}:00", rid, helpers.LINE_ID, apex, "M12", snr, 1, inject.K_MAIN, 0
    )


def test_5_off_arm_equals_on_outside_the_tapers_and_y0(helpers, context, patch):
    response = helpers._response()
    spec = _spec(helpers)
    on = inject.inject_realization(context, [spec], {"M12": response})
    assert on.truths
    off = place.inject_untapered(context, on, {("M12", inject.K_MAIN): patch})
    assert off.accepted == (spec.injection_id,)
    columns = place.aperture_taper_columns(
        [(patch, spec.apex_trace)], context.background.n_traces
    )
    assert columns.sum() == 2 * patch.n_aperture_taper
    assert place.off_equals_on_outside_tapers(on.data, off.data, columns)
    assert not np.array_equal(on.data, off.data)
    y0 = _load(PROCESSING / "m33_y0_diag_infer.py", "y0_copy_check")
    original = y0.inject_without_taper(context, response, spec)
    assert np.array_equal(original.radargram.data, off.data.astype(np.float32))
    assert original.truth.gain == on.truths[0].gain
    assert np.array_equal(y0.untapered_field(patch), place.untapered_field(patch))


def test_5b_r_edge_is_the_on_off_difference_field(patch):
    n_rows, n_cols = 600, patch.response.n_traces
    n_taper = patch.n_aperture_taper
    t = np.arange(n_rows)

    def burst(centre: float) -> np.ndarray:
        return np.exp(-(((t - centre) / 40.0) ** 2)) * np.cos(2 * np.pi * t / 25.0)

    untapered = np.zeros((n_rows, n_cols))
    apex = round(patch.response.apex_trace)
    untapered[:, apex] = burst(200.0)
    # Edge wing several P later, amplitude falling outward inside the taper.
    amplitudes = 0.05 + 0.01 * np.arange(n_taper)
    for column, amplitude in enumerate(amplitudes):
        untapered[:, column] = amplitude * burst(330.0)
    weights = inject.edge_taper_weights(n_cols, n_taper)
    fake = replace(patch, field=untapered * weights[None, :], apex_sample=200.0)
    context = SimpleNamespace(
        background=SimpleNamespace(amplitude_processed=True), period_samples=25.0
    )
    difference = place.untapered_field(fake) - fake.field
    assert not difference[:, n_taper:].any()  # zero outside the taper columns
    expected = 20 * math.log10(float(np.max(amplitudes * (1 - weights[:n_taper]))))
    v03 = 20 * math.log10(float(amplitudes.max()))  # innermost taper column
    measured = place.r_edge_db(fake, context)
    assert abs(measured - expected) < 0.01
    assert abs(measured - v03) > 1.0


def test_5c_inject_bands_equals_the_library_on_plain_bands(helpers, long_context):
    response = helpers._response()
    patch = inject.build_response_patch(response, long_context, inject.K_MAIN)
    band = inject.build_band_patch(patch, long_context, 100)
    center = 400 + (band.n_columns - 1) / 2
    spec = inject.InjectionSpec(
        "b:00",
        "b",
        helpers.LINE_ID,
        center,
        "M12",
        18.0,
        -1,
        inject.K_MAIN,
        0,
        mode=inject.InjectionMode.BAND,
        band_length_traces=100,
        band_center_trace=center,
    )
    library = inject.inject_band_realization(long_context, [spec], {"M12": response})
    mine = place.inject_bands(long_context, [(spec, band)])
    assert np.array_equal(library.data, mine.data)
    assert library.truths == mine.truths and library.rejected == mine.rejected


def test_5d_band_contrast_equals_the_r3_script():
    contrast = _load(PROCESSING / "m33_r3_band_contrast.py", "contrast_copy_check")
    q = np.random.default_rng(3).normal(size=(120, 300))
    q[40:50, 100:180] += 3.0
    for box, gap in (((100.2, 40.1, 180.7, 50.0), 7), ((10.0, 2.0, 30.0, 5.0), 0)):
        assert place.band_contrast(q, box, 110, gap) == contrast.metrics(
            q, *box, 110, gap
        )
    assert place.band_contrast(q, (0.0, 0.0, 2.0, 1.0), 110) is None


# -------------------------------------------------------- 6 round-robin


def _candidates() -> list[place.HyperbolaCandidate]:
    found = []
    for line in range(4):
        for model in ("A", "B", "C"):
            for apex in range(20, 400, 15):
                found.append(
                    place.HyperbolaCandidate(
                        f"L{line}", "4tu", model, float(apex), 20.0, 5.0
                    )
                )
    return found


def test_6_round_robin_is_seeded_capped_and_spaced():
    targets = {("4tu", "A"): 4, ("4tu", "B"): 4, ("4tu", "C"): 8}
    plan = place.plan_hyperbola_positions(_candidates(), targets)
    assert plan == place.plan_hyperbola_positions(_candidates(), targets)
    by_line: dict[str, list[place.PlannedHyperbola]] = {}
    for position in plan.positions:
        by_line.setdefault(position.line_id, []).append(position)
    counts = {m: sum(p.model == m for p in plan.positions) for m in "ABC"}
    assert counts == {"A": 4, "B": 4, "C": 8} and not plan.shortfall
    for members in by_line.values():
        assert len(members) <= 4
        for group in (0, 1):
            in_group = [p for p in members if p.group == group]
            assert len(in_group) <= 2
            for i, a in enumerate(in_group):
                for b in in_group[i + 1 :]:
                    assert abs(a.apex_trace - b.apex_trace) >= 40.0 + 5.0
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                assert abs(a.apex_trace - b.apex_trace) >= 20.0
    # C needs 8 over 4 lines: second positions only once every line has one.
    c_lines = [p.line_id for p in plan.positions if p.model == "C"]
    first_pass = [p for p in plan.positions if p.model == "C" and not p.second_on_line]
    assert len(set(c_lines)) == 4 and len(first_pass) == 4
    shortfall = place.plan_hyperbola_positions(_candidates(), {("4tu", "A"): 99})
    assert shortfall.shortfall[("4tu", "A")] > 0


# ------------------------------------------------------- 7 band positions


def test_7_band_arms_share_model_centre_taper_and_second_spacing(helpers, long_context):
    plan = place.plan_band_positions_e3x(long_context, {"M12": helpers._response()})
    assert len(plan.positions) == 2
    w_eff = inject.effective_median_window(long_context)
    for position in plan.positions:
        longs = {a.center_trace for a in position.arms if a.arm == place.LONG_ARM}
        assert longs == {position.center_trace}
        for arm in position.arms:
            assert arm.band.patch.response.response_id == position.model
            assert abs(arm.center_trace - position.center_trace) <= 0.5
    first, second = plan.positions
    length = plan.lengths[place.LONG_ARM]
    assert abs(first.center_trace - second.center_trace) >= length + w_eff
    assert {a.row_px for a in first.arms if a.arm == place.SHORT_ARM} == {47.0, 68.0}


# ------------------------------------------------------ 8-9 decisions


def _interval(low, high, estimate=None, degenerate=False) -> judge.Interval:
    estimate = (low + high) / 2 if estimate is None else estimate
    return judge.Interval(
        estimate,
        low,
        high,
        "satterthwaite_t",
        judge.CI_LEVEL,
        5.0,
        {},
        {},
        {},
        {},
        low,
        high,
        degenerate,
    )


def test_8_decision_table_boundaries():
    d = judge.DELTA
    assert judge.directional_verdict(_interval(0.0, 0.1), 0.5, None).decision is (
        judge.Decision.INCONCLUSIVE
    )
    assert judge.directional_verdict(_interval(1e-12, 0.1), 0.5, None).decision is (
        judge.Decision.CONFIRMED
    )
    assert judge.directional_verdict(_interval(-d, 0.01), 0.5, None).decision is (
        judge.Decision.INCONCLUSIVE
    )
    assert judge.directional_verdict(_interval(-0.01, d), 0.5, None).decision is (
        judge.Decision.INCONCLUSIVE
    )
    assert judge.directional_verdict(_interval(-0.01, 0.01), 0.5, None).decision is (
        judge.Decision.NEGLIGIBLE
    )
    two = judge.directional_verdict(_interval(-0.2, -0.1), 0.5, None, two_sided=True)
    assert two.decision is judge.Decision.CONFIRMED_NEGATIVE
    small = judge.directional_verdict(_interval(0.01, 0.07, 0.04), 0.5, None)
    assert small.small_effect and judge.WORDING_SMALL in small.wording
    # single_background: a background under 15 planned positions leaves.
    assert judge.kept_backgrounds({"4tu": 40, "gz_ids": 14}) == (("4tu",), True)
    assert judge.kept_backgrounds({"4tu": 40, "gz_ids": 15}) == (
        ("4tu", "gz_ids"),
        False,
    )
    # [STV] additional review: degenerate, floor, nu floor, control not_run.
    flat = _interval(0.03, 0.03, 0.03, degenerate=True)
    assert judge.directional_verdict(flat, 0.5, None).reason == "degenerate_ci"
    zero = _interval(0.0, 0.0, 0.0, degenerate=True)
    assert judge.directional_verdict(zero, 0.05, None).reason == "floor"
    assert (
        judge.directional_verdict(zero, 0.10, None).decision
        is judge.Decision.NEGLIGIBLE
    )
    _, _, _, _, nu_g, _, _ = judge.satterthwaite(
        0.0, {"a": np.array([0.0, 1.0, 2.0])}, {"a": 2}, {"a": 1.5}, judge.TAIL
    )
    assert nu_g == {"a": 1.0}
    single = judge.LinearResult(None, "single_cluster_background", 40, 20, (), {}, 0.5)
    assert judge.negative_control_outcome(single) is judge.CheckOutcome.NOT_RUN


def _result(interval, n=40, clusters=20, models=("A", "B", "C", "D", "E"), reason=None):
    return judge.LinearResult(interval, reason, n, clusters, models, {}, 0.5)


def test_8b_hh1_label_rules():
    good = _result(_interval(0.1, 0.3))
    fail = _result(_interval(0.06, 0.1, 0.08), n=30)
    assert judge.negative_control_outcome(fail) is judge.CheckOutcome.FAIL
    verdict = judge.hh1_verdict(good, judge.CheckOutcome.FAIL, None)
    assert (verdict.decision, verdict.reason) == (
        judge.Decision.NOT_EVALUABLE,
        "negative_control_failed",
    )
    flagged = judge.hh1_verdict(good, judge.CheckOutcome.FLAG, None)
    assert flagged.decision is judge.Decision.CONFIRMED_POSITIVE
    assert judge.WORDING_NEG_FLAG in flagged.wording
    not_run = judge.hh1_verdict(good, judge.CheckOutcome.NOT_RUN, None)
    assert judge.WORDING_NEG_NOT_RUN in not_run.wording
    few = judge.hh1_verdict(
        _result(_interval(0.1, 0.3), n=29), judge.CheckOutcome.PASS, None
    )
    assert few.decision is judge.Decision.NOT_EVALUABLE  # no other estimand
    models = judge.hh1_verdict(
        _result(_interval(0.1, 0.3), models=("A", "B", "C", "D")),
        judge.CheckOutcome.PASS,
        None,
    )
    assert models.reason == "models_below_minimum"
    rows = [
        {
            "accepted": "True",
            "c_lat": str(v),
            "length_arm": "long",
            "texture": "none",
            "recipe": "r0",
            "snr_db": "12.0",
            "bin_target_px": str(row),
            "realization_id": f"r{row}{i}",
            "position_id": "p",
        }
        for row, v in ((28.0, -2.0), (47.0, 2.0), (68.0, 1.5))
        for i in range(3)
    ]
    assert judge.contrast_flag(rows)["flag"]
    assert not judge.contrast_flag([r | {"c_lat": "0.0"} for r in rows])["flag"]


def _rows(values, clusters, models=None, background="4tu"):
    models = models or ["A"] * len(values)
    return [
        judge.PositionValue(f"p{i}", background, c, m, v, 0.5)
        for i, (v, c, m) in enumerate(zip(values, clusters, models, strict=True))
    ]


def test_9_bootstrap_is_seeded_ordered_and_has_edge_cases():
    rows = _rows([0.1, 0.3, 0.2, 0.0, 0.4, 0.1], ["a", "a", "b", "c", "c", "d"])
    first = judge.linear_result(rows, judge.Weighting.MODEL, judge.Bootstrap(5, 400))
    second = judge.linear_result(rows, judge.Weighting.MODEL, judge.Bootstrap(5, 400))
    assert first.interval == second.interval
    both = rows + _rows([0.2, 0.1], ["z1", "z2"], background="gz_ids")
    bootstrap = judge.Bootstrap(7, 50)
    design = judge.ClusterDesign.of(both, judge.BACKGROUNDS)
    replicates = judge.draw_replicates(design, bootstrap)
    rng = np.random.default_rng(7)
    fourtu = rng.integers(0, 4, size=(50, 4))
    ids = rng.integers(0, 2, size=(50, 2))
    assert np.array_equal(replicates.full[:, 0], (fourtu == 0).sum(axis=1))
    assert np.array_equal(replicates.full[:, 6], (ids == 0).sum(axis=1))
    lone = _rows([0.1, 0.2], ["a", "a"])
    result = judge.linear_result(lone, judge.Weighting.MODEL, judge.Bootstrap(1, 50))
    assert result.not_evaluable_reason == "single_cluster_background"
    same = _rows([0.2] * 6, ["a", "a", "b", "c", "c", "d"])
    flat = judge.linear_result(same, judge.Weighting.MODEL, judge.Bootstrap(1, 50))
    assert flat.interval.degenerate and flat.interval.low == flat.interval.high == 0.2
    assert judge.CI_LEVEL == 1 - 0.05 / 6
    assert judge.TAIL == 0.05 / 12


# ---------------------------------------------- 10 merge-order fixture


def test_10_operating_cut_before_the_merge_equals_the_direct_path(long_context):
    image = make_detection_image(long_context.background, inputs.recipe_config("r0"))
    assert len(image.tiles) >= 2
    box = PixelBox(100.0, 30.0, 140.0, 60.0)
    strong = LineBox(0, "pipe_hyperbola", 0.9, box, 0, tile_truncated=True)
    weak = LineBox(0, "pipe_hyperbola", 0.1, box, 1, tile_truncated=False)
    tile_boxes = [strong, weak]
    direct = merge_line_boxes([strong], image, DEFAULT_MERGE_CONFIG)
    before = merge_line_boxes(
        e3x.operating_boxes(tile_boxes), image, DEFAULT_MERGE_CONFIG
    )
    after = [
        b
        for b in merge_line_boxes(tile_boxes, image, DEFAULT_MERGE_CONFIG)
        if b.score > e3x.CONF_OPERATING
    ]
    assert before == direct and before
    assert after != direct  # the weak, untruncated box represented the pair


# ------------------------------------------------------ 11-12 workers, stdout


def _double(value: int) -> int:
    return 2 * value


def _no_setup() -> None:
    return None


def test_11_workers_do_not_change_the_output(tmp_path):
    jobs = list(range(7))
    one = e3x.map_ordered(1, _double, jobs, _no_setup, "t")
    two = e3x.map_ordered(2, _double, jobs, _no_setup, "t")
    assert one == two == [2 * j for j in jobs]
    rows = [{"a": 0.1 + v, "b": v} for v in one]
    e3x.write_rows(tmp_path / "w1.csv", ("a", "b"), rows)
    e3x.write_rows(
        tmp_path / "w2.csv", ("a", "b"), [{"a": 0.1 + v, "b": v} for v in two]
    )
    assert (tmp_path / "w1.csv").read_bytes() == (tmp_path / "w2.csv").read_bytes()


def test_12_stdout_is_blind_and_nms_warnings_are_counted(capsys):
    logger = logging.getLogger("e3x-test-nms")
    counter = e3x.WarningCounter()
    logger.addHandler(counter)
    logger.warning("NMS time limit 5.200s exceeded")
    logger.warning("something else")
    assert counter.take() == 1 and counter.take() == 0
    empty = e3x.LineDetections({}, {0: 2})
    line = SimpleNamespace(entries=(), defects={}, plan=None)
    _, defects = e3x.line_records(line, empty, (0,), PP_SETS[PPSetId.S3])
    assert defects["nms_time_limit_warning"] == 2
    capsys.readouterr()
    e3x.print_final_line("ok", None, e3x.Counts(1, 2, 3, 4), 0.0)
    final = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert tuple(final) == e3x.FINAL_KEYS
    assert not e3x.METRIC_WORDS.search(json.dumps(final))


# --------------------------------------------------------- 13 best score


def test_13_best_scores_use_the_pre_merge_boxes(long_context):
    image = make_detection_image(long_context.background, inputs.recipe_config("r0"))
    grid = image.grid_map
    apex = match_apex = e3x.match.TruthApex("x", 100.0, 120.0)
    period = long_context.period_samples
    inside = PixelBox(
        95 / grid.traces_per_px,
        110 / grid.samples_per_px,
        105 / grid.traces_per_px,
        130 / grid.samples_per_px,
    )
    far = PixelBox(
        500 / grid.traces_per_px,
        110 / grid.samples_per_px,
        520 / grid.traces_per_px,
        130 / grid.samples_per_px,
    )
    boxes = [
        LineBox(0, "pipe_hyperbola", 0.2, inside, 0),
        LineBox(1, "pipe_band", 0.7, inside, 0),
        LineBox(0, "pipe_hyperbola", 0.9, far, 0),
    ]
    best = e3x.best_scores(
        boxes,
        image,
        e3x.apex_accepts(apex, period),
        DetectionClass.PIPE_HYPERBOLA,
        "hyp",
    )
    assert best["best_score_001_hyp"] == 0.2 and not best["best_score_001_hyp_no_box"]
    assert best["best_score_001_any"] == 0.7
    none = e3x.best_scores(
        [],
        image,
        e3x.apex_accepts(match_apex, period),
        DetectionClass.PIPE_HYPERBOLA,
        "hyp",
    )
    assert none["best_score_001_hyp"] == "" and none["best_score_001_hyp_no_box"]
    truth = GridBox(200, 260, 100, 130)
    beside = LineBox(
        1,
        "pipe_band",
        0.8,
        PixelBox(
            300 / grid.traces_per_px,
            100 / grid.samples_per_px,
            340 / grid.traces_per_px,
            130 / grid.samples_per_px,
        ),
        0,
    )
    band = e3x.best_scores(
        [beside], image, e3x.band_accepts(truth), DetectionClass.PIPE_BAND, "band"
    )
    assert band["best_score_001_band_no_box"]  # same rows, no trace in [i0, i1]
    fold = e3x.FoldBoxes((), (), (), 0, 0)
    row = {"hit": True, "best_score_001_hyp": 0.2}
    assert (
        e3x._fold_defects(e3x.KIND_HYPERBOLA, [row], fold)["hit_without_operating_box"]
        == 1
    )
    row = {"hit": True, "best_score_001_hyp": 0.3}
    assert not e3x._fold_defects(e3x.KIND_HYPERBOLA, [row], fold)[
        "hit_without_operating_box"
    ]


def test_13b_merge_keeps_tile_boxes_and_scores(long_context):
    """(i) hyperbola duplicates merge into one input box; (ii) band pieces are
    x-cuts of an input box (same tile, rows and score) - the basis of the
    "hit => best pre-merge score > 0.25" assertion (cv, plan v0.4 A-5 13)."""
    image = make_detection_image(long_context.background, inputs.recipe_config("r0"))
    duplicates = [
        LineBox(
            0,
            "pipe_hyperbola",
            0.9,
            PixelBox(100.0, 30.0, 140.0, 60.0),
            0,
            tile_truncated=True,
        ),
        LineBox(0, "pipe_hyperbola", 0.4, PixelBox(101.0, 30.5, 140.5, 60.0), 1),
        LineBox(0, "pipe_hyperbola", 0.7, PixelBox(99.0, 29.5, 139.0, 59.0), 2),
    ]
    merged = merge_line_boxes(duplicates, image, DEFAULT_MERGE_CONFIG)
    inputs_seen = {(b.box, b.score) for b in duplicates}
    assert merged and all((b.box, b.score) in inputs_seen for b in merged)
    bands = [
        LineBox(1, "pipe_band", 0.8, PixelBox(20.0, 40.0, 200.0, 48.0), 0),
        LineBox(1, "pipe_band", 0.6, PixelBox(90.0, 41.0, 300.0, 49.0), 1),
        LineBox(1, "pipe_band", 0.5, PixelBox(150.0, 40.5, 330.0, 48.5), 2),
    ]
    pieces = merge_line_boxes(bands, image, DEFAULT_MERGE_CONFIG)
    assert pieces
    for piece in pieces:
        assert any(
            b.tile_index == piece.tile_index
            and b.score == piece.score
            and (b.box.y1, b.box.y2) == (piece.box.y1, piece.box.y2)
            and b.box.x1 <= piece.box.x1
            and piece.box.x2 <= b.box.x2
            for b in bands
        ), piece


def test_plan_report_counts_short_offset_positions_and_limit_in_product_form():
    arms = [
        {"position_id": "a", "length_arm": "short", "short_center_offset_traces": -0.5},
        {"position_id": "a", "length_arm": "short", "short_center_offset_traces": -0.5},
        {"position_id": "a", "length_arm": "long", "short_center_offset_traces": 0.0},
        {"position_id": "b", "length_arm": "short", "short_center_offset_traces": 0.0},
    ]
    assert e3x.short_offset_positions(arms) == 1  # two short rows, one position

    def plan_info(shift: int) -> dict:
        arm = {
            "position_id": "p",
            "shift_samples": shift,
            "shift_P": shift / 2.0,
            "shift_gap_samples": 0.0,
            "taper_traces": 14,
            "k0": 100 + shift,
        }
        return {
            "n_shift_mismatch": 0,
            "arms": [arm],
            "positions": [{"position_id": "p", "period_bg_samples": 2.0}],
        }

    assert e3x.shift_check(plan_info(24))  # exactly 12 P is allowed
    assert not e3x.shift_check(plan_info(25))


# ------------------------------------------------------------- 14 pool


def test_14_fourtu_sample_rule():
    rows = inputs.read_csv(pool.m33_inputs.R2_LINES_CSV)
    candidates = pool.fourtu_candidates(rows)
    sample = pool.draw_fourtu_sample(candidates)
    assert sample == pool.draw_fourtu_sample(candidates)
    r3 = pool.r3_line_ids(rows)
    assert not {c.line_id for c in sample} & r3
    per_activity: dict[str, int] = {}
    for c in sample:
        per_activity[c.activity_id] = per_activity.get(c.activity_id, 0) + 1
    assert max(per_activity.values()) <= 3
    assert sum(c.is_long_window for c in sample) >= 12
    unshared = [c for c in candidates if not c.shares_r3_activity]
    capped: dict[str, int] = {}
    for c in unshared:
        capped[c.activity_id] = min(3, capped.get(c.activity_id, 0) + 1)
    assert sum(not c.shares_r3_activity for c in sample) == sum(capped.values())
    built, counts = pool.build_pool(rows)
    assert counts["n_fourtu"] == 100 and counts["n_ids"] == 53
    assert sum(row["role"] == pool.ROLE_PRECHECK for row in built) == 2


# ------------------------------------------------------- 15-19 statistics


def test_15_satterthwaite_nu_balanced_and_two_backgrounds():
    rng = np.random.default_rng(4)
    values = rng.normal(size=8)
    _, _, nu, _, _, _, _ = judge.satterthwaite(
        0.0, {"a": values}, {"a": 6}, {"a": judge.kish_clusters([3] * 6)}, judge.TAIL
    )
    assert abs(nu - 5.0) <= 1e-9 * 5.0
    a, b = rng.normal(size=50), 2 * rng.normal(size=50)
    _, _, nu, v_by_background, nu_g, _, _ = judge.satterthwaite(
        0.0, {"a": a, "b": b}, {"a": 10, "b": 4}, {"a": 9.0, "b": 3.0}, judge.TAIL
    )
    v_a = 10 / 9 * np.var(a, ddof=1)
    v_b = 4 / 3 * np.var(b, ddof=1)
    expected = (v_a + v_b) ** 2 / (v_a**2 / 8.0 + v_b**2 / 2.0)
    assert abs(nu - expected) <= 1e-9 * expected
    assert nu_g == {"a": 8.0, "b": 2.0}
    assert v_by_background == {
        "a": 10 / 9 * np.var(a, ddof=1),
        "b": 4 / 3 * np.var(b, ddof=1),
    }


def test_16_basic_equals_percentile_for_a_symmetric_distribution():
    replicates = 0.6 + np.concatenate([np.linspace(-0.2, 0.2, 401)])
    assert np.allclose(
        judge.basic_interval(0.6, replicates),
        judge.percentile_interval(replicates),
        rtol=0,
        atol=1e-12,
    )


def test_17_floor_boundary():
    inside = _interval(-0.01, 0.01)
    assert (
        judge.directional_verdict(inside, 0.10, None).decision
        is judge.Decision.NEGLIGIBLE
    )
    below = judge.directional_verdict(inside, 0.10 - 1e-12, None)
    assert (below.decision, below.reason) == (judge.Decision.INCONCLUSIVE, "floor")


def test_18_negative_control_boundaries():
    def outcome(estimate, low, high):
        return judge.negative_control_outcome(_result(_interval(low, high, estimate)))

    assert outcome(0.05, 0.01, 0.09) is judge.CheckOutcome.FAIL
    assert outcome(0.05 - 1e-12, 0.01, 0.09) is judge.CheckOutcome.FLAG
    assert outcome(0.02, 0.001, 0.04) is judge.CheckOutcome.FLAG
    assert outcome(0.02 - 1e-12, 0.001, 0.04) is judge.CheckOutcome.PASS
    assert outcome(0.3, -0.01, 0.6) is judge.CheckOutcome.PASS
    assert outcome(-0.06, -0.09, -0.01) is judge.CheckOutcome.FAIL
    few = judge.negative_control_outcome(_result(_interval(0.1, 0.2), n=9))
    assert few is judge.CheckOutcome.NOT_RUN
    few_clusters = judge.negative_control_outcome(
        _result(_interval(0.1, 0.2), clusters=4)
    )
    assert few_clusters is judge.CheckOutcome.NOT_RUN


def test_19_a_model_missing_from_a_replicate_reweights():
    values = np.array([0.2, 0.4, 0.9, 0.1])
    groups = ["A", "A", "B", "C"]
    weights = np.array([[1, 1, 1, 1], [2, 0, 0, 1]], dtype=float)
    means = judge.equal_weight_mean(weights, values, groups)
    assert abs(means[0] - (0.3 + 0.9 + 0.1) / 3) < 1e-15
    assert abs(means[1] - (0.2 + 0.1) / 2) < 1e-15  # B is gone in replicate 2
    backgrounds = ["4tu"] * 6
    models = ["A", "A", "B", "B", "C", "C"]
    y = np.array([0.1, 0.3, 0.8, 0.6, 0.0, 0.2])
    full = judge.decompose(backgrounds, models, y)
    dropped = judge.decompose(backgrounds, models, y, np.array([1, 1, 0, 0, 1, 1.0]))
    assert (full.df_model, dropped.df_model) == (2, 1)
    assert (full.df_res, dropped.df_res) == (3, 2)


def test_19b_eta_squared_against_a_hand_computation():
    backgrounds = ["4tu"] * 4 + ["gz_ids"] * 4
    models = ["A", "B", "A", "B", "A", "B", "A", "B"]
    y = np.array([0.0, 1.0, 0.2, 0.8, 0.1, 0.9, 0.3, 0.7])
    result = judge.decompose(backgrounds, models, y)
    # Balanced: Type II SS_model = 8 * (model mean - grand mean)^2.
    grand = y.mean()
    ss_model = 4 * ((y[0::2].mean() - grand) ** 2 + (y[1::2].mean() - grand) ** 2)
    assert abs(result.ss_model - ss_model) < 1e-12
    assert abs(result.eta2_p - ss_model / (ss_model + result.ss_res)) < 1e-15


# ------------------------------------------------------------ 20 strata


def test_20_position_strata_from_r_edge():
    assert place.edge_snr_set(-12.0) == (12.0, 18.0)  # r + 12 = 0: in
    assert place.edge_snr_set(-12.0 - 1e-9) == (18.0,)
    assert place.edge_stratum(-18.0) is place.EdgeStratum.MANIPULATED  # r + 18 = 0
    assert place.edge_stratum(-24.0) is place.EdgeStratum.GRAY  # r + 18 = -6
    assert place.edge_stratum(-24.0 - 1e-9) is place.EdgeStratum.NEGATIVE_CONTROL
    # The same model on two lines can fall in different strata.
    rows = [
        {
            "position_id": "a",
            "dataset": "4tu",
            "cluster_id": "c1",
            "model": "M15",
            "r_edge_db": "-19.0",
            "edge_stratum": place.edge_stratum(-19.0).value,
            "shares_r3_activity": "False",
        },
        {
            "position_id": "b",
            "dataset": "4tu",
            "cluster_id": "c2",
            "model": "M15",
            "r_edge_db": "-25.0",
            "edge_stratum": place.edge_stratum(-25.0).value,
            "shares_r3_activity": "False",
        },
    ]
    positions = judge.hyperbola_positions(rows)
    assert (
        positions["a"].stratum == "gray"
        and positions["b"].stratum == "negative_control"
    )


# ------------------------------------------------ judge on unit tables


def _hyperbola_record(
    pid,
    model,
    cluster,
    taper,
    snr,
    pol,
    fold,
    hit,
    stratum="manipulated",
    r_edge="-5.0",
    background="4tu",
) -> dict:
    return {
        "position_id": pid,
        "model": model,
        "cluster_id": cluster,
        "dataset": background,
        "taper": taper,
        "snr_db": str(snr),
        "polarity": str(pol),
        "fold": str(fold),
        "hit": str(hit),
        "accepted": "True",
        "edge_stratum": stratum,
        "r_edge_db": r_edge,
        "shares_r3_activity": "False",
        "reject_reason": "",
        "hit_s3": str(hit),
        "n_edge_boxes": "0",
        "period_inj": "24.0",
        "best_score_001_hyp": "0.5",
    }


def test_judge_units_m_star_and_hh1_rows():
    records = []
    for i in range(3):
        for taper, hit in (("on", True), ("off", i == 0)):
            for snr in (12.0, 18.0):
                for pol in (1, -1):
                    for fold in judge.FOLDS:
                        records.append(
                            _hyperbola_record(
                                f"p{i}", "M03", f"c{i}", taper, snr, pol, fold, hit
                            )
                        )
    units = judge.fold_means(records, judge.hyperbola_key, "hit")
    positions = judge.hyperbola_positions(records)
    assert judge.m_star_models(units, positions) == ("M03",)
    rows = judge.hh1_rows(units, positions, ("M03",))
    assert [r.difference for r in rows] == [0.0, 1.0, 1.0]
    missing = [
        r for r in records if not (r["position_id"] == "p1" and r["fold"] == "4")
    ]
    units = judge.fold_means(missing, judge.hyperbola_key, "hit")
    assert [r.position_id for r in judge.hh1_rows(units, positions, ("M03",))] == [
        "p0",
        "p2",
    ]


def test_score_bins():
    assert [judge.score_bin(v) for v in ("", 0.01, 0.05, 0.25, 0.3)] == [
        "no_box",
        "le_0.01",
        "0.01_0.1",
        "0.1_0.25",
        "0.25_1",
    ]


def test_runner_settings_and_pre_seal_problems(tmp_path):
    settings = e3x.parse_args(["--stage", "plan", "--out-root", str(tmp_path)])
    assert settings.stage_dir == tmp_path / "plan" and settings.folds_run is None
    assert not settings.scope_blockers()
    partial = e3x.parse_args(["--stage", "injected", "--folds", "0"])
    assert partial.scope_blockers() == ["partial_folds"]
    tolerated, rest = e3x.pre_seal2_problems(
        ["not tracked: analysis/detection/m33_e3x.py", "modified: M docs/x.md"]
    )
    assert tolerated == ["not tracked: analysis/detection/m33_e3x.py"]
    assert rest == ["modified: M docs/x.md"]
    with pytest.raises(SystemExit):
        e3x.parse_args(["--stage", "injected", "--overwrite"])
    assert re.search(e3x.METRIC_WORDS, "recall") and not e3x.METRIC_WORDS.search(
        "tiles"
    )


SEAL2_R1 = REPO / "analysis" / "detection" / "m33_e3x_seal2_r1.json"


def test_seal2_r1_record_supersedes_seal2_and_keeps_the_plan():
    old = json.loads(seal_draft.SEAL2.read_text("utf-8"))
    record = json.loads(SEAL2_R1.read_text("utf-8"))
    assert e3x.SEAL2_RECORD == SEAL2_R1
    assert set(old) | {"supersedes", "plan_code_blobs", "plan_reuse"} == set(record)
    assert record["supersedes"] == {
        "path": "analysis/detection/m33_e3x_seal2.json",
        "sha256_lf": inputs.sha256_file_lf(seal_draft.SEAL2),
        "commit": "729d19ced8674039efac0cfe906fcb41f45c31e3",
    }
    for key in (
        "pool_csv_sha256_lf",
        "plan_files_sha256_lf",
        "plan_report_sha256_lf",
        "src_tree_id",
        "plan_head",
        "plan_run_id",
        "seeds",
    ):
        assert record[key] == old[key], key
    assert record["plan_code_blobs"] == old["code_blobs"]
    assert set(record["code_blobs"]) == set(old["code_blobs"])
    assert record["code_blobs"] != old["code_blobs"]
    reuse = record["plan_reuse"]
    assert set(reuse["reachable_changed"]) <= set(seal_draft.REACHABLE_NOTES)


def test_seal2_r1_reachability_graph_on_a_small_module():
    old = seal_draft.top_level_definitions(
        "A = 1\ndef plan():\n    return helper()\ndef helper():\n    return A\n"
        "def other():\n    return 2\n"
    )
    new = seal_draft.top_level_definitions(
        "A = 2\ndef plan():\n    return helper()\ndef helper():\n    return A\n"
        "def other():\n    return 3\ndef added():\n    return place.x\n"
    )
    assert seal_draft.changed_names(old, new) == ["A", "added", "other"]
    graph = seal_draft.reference_graph({"m33_e3x": new, "m33_e3x_place": {}})
    reach = seal_draft.reachable(graph, ["m33_e3x:plan"])
    assert reach == {"m33_e3x:plan", "m33_e3x:helper", "m33_e3x:A"}
    source = "def plan():\n    return helper()\ndef helper():\n    return 1\n"
    assert seal_draft.references(source, "helper", None) == [{"line": 2, "in": "plan"}]


# ------------------------------------------------------ offline end to end


class TileDetector:
    """One ``pipe_hyperbola`` and one ``pipe_band`` box over every tile."""

    class_names = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")

    def __init__(self, confidence: float = 0.001) -> None:
        self.confidence = confidence

    def predict(self, image: np.ndarray):
        from gpr_detection.detect.base import Detection, DetectorOutput, InputTransform

        size = image.shape[0]
        boxes = (
            Detection(0, "pipe_hyperbola", 0.6, PixelBox(0, 0, size, size)),
            Detection(1, "pipe_band", 0.7, PixelBox(0, 0, size, size)),
            Detection(0, "pipe_hyperbola", 0.01, PixelBox(0, 0, 8, 8)),
        )
        kept = tuple(box for box in boxes if box.score > self.confidence)
        return DetectorOutput(kept, InputTransform(1, 1, 0, 0, size, size))


@pytest.fixture
def offline(monkeypatch, helpers, tmp_path):
    """One synthetic 4TU stand-in line, two stand-in models, no stored input."""
    import m32_pipe_responses as responses_io

    radargram = helpers._radargram(helpers._background_data())
    library = responses_io.ResponseLibrary(
        air={"M12": helpers._response("M12"), "M03": helpers._response("M03")},
        ground={},
        groups={"M12": responses_io.ModelGroup.A1, "M03": responses_io.ModelGroup.A0},
        sha256={},
    )
    ref = inputs.LineRef(inputs.DATASET_4TU, helpers.LINE_ID, "background")
    line = e3x.PoolLine(ref, "01.1", False, "background")
    zero = inject.prepare_line_context(radargram).zero
    monkeypatch.setattr(e3x, "zone_c_boxes", lambda settings: ({}, {}))
    monkeypatch.setattr(e3x, "pool_input_hashes", lambda settings: {})
    monkeypatch.setattr(e3x, "background_pool", lambda settings: [line])
    monkeypatch.setattr(
        inputs, "candidate_apexes", lambda: inputs.CandidateApexes({}, {})
    )
    monkeypatch.setattr(
        e3x.runner, "load_line", lambda r: e3x.runner.LineData(r, radargram, zero)
    )
    monkeypatch.setattr(responses_io, "load_library", lambda *_: library)
    tree = {
        "head": "h",
        "dirty": False,
        "problems": [],
        "code_blobs": {"analysis/detection/m33_e3x.py": "blob"},
        "src_tree_id": "s",
    }
    monkeypatch.setattr(e3x, "measure_tree", lambda settings: dict(tree))
    pool_csv = tmp_path / "pool.csv"
    pool_csv.write_text("line_id\n", encoding="utf-8")
    argv = ["--stage", "plan", "--out-root", str(tmp_path), "--pool", str(pool_csv)]
    settings = e3x.parse_args(argv)
    context = e3x.Context(tree, {"pp_set_id": "S3"}, {})
    return SimpleNamespace(
        settings=settings, context=context, line=line, library=library
    )


def test_offline_plan_realizations_records_and_judge(offline, capsys):
    status, counts = e3x.run_plan(offline.settings, 0.0, offline.context)
    assert status == "ok" and counts.n_positions > 0
    plan_dir = offline.settings.stage_dir
    for name in (*e3x.PLAN_FILES, "plan_report.json", "seal2_record_draft.json"):
        assert (plan_dir / name).is_file(), name
    stdout = capsys.readouterr().out
    assert not e3x.METRIC_WORDS.search(stdout)
    plans = e3x.read_plan(plan_dir, [offline.line], inputs.CandidateApexes({}, {}))
    built = e3x.build_line(plans[0], offline.library, offline.library)
    assert built.zero_test and all(built.off_on_equal.values())
    assert all(built.texture_checks.values()) and not built.defects["shift_mismatch"]
    detections = e3x.infer_entries(
        built.entries, {0: TileDetector(), 1: TileDetector()}, None
    )
    records, defects = e3x.line_records(built, detections, (0, 1), PP_SETS[PPSetId.S3])
    assert sum(defects.values()) == 0, defects
    kinds = {entry.kind for entry in built.entries}
    assert kinds == {e3x.KIND_HYPERBOLA, e3x.KIND_BAND}
    recipes = {entry.recipe for entry in built.entries}
    assert recipes == {e3x.RECIPE_R0, e3x.RECIPE_P42}
    accepted = [r for r in records[e3x.KIND_BAND] if r["accepted"]]
    assert accepted and all(r["coverage"] == 1.0 for r in accepted)
    assert all(r["best_score_001_band"] == 0.7 for r in accepted)
    hits = [r for r in records[e3x.KIND_HYPERBOLA] if r["accepted"]]
    # One whole-tile box is assigned to the nearest apex only (one-to-one).
    assert any(r["hit"] for r in hits) and any(r["hit_contain"] for r in hits)
    part = {"records": records}
    hashes = e3x.write_record_files(offline.settings.out_root, [part], (0, 1))
    assert set(hashes) >= {"hyperbola_records.csv", "band_records.csv", "truths.csv"}
    hyperbola = inputs.read_csv(offline.settings.out_root / "hyperbola_records.csv")
    band = inputs.read_csv(offline.settings.out_root / "band_records.csv")
    planned = judge.planned_band_counts(plan_dir)
    data = judge.load_data(hyperbola, band, planned)
    folds = (0, 1)
    data = replace(
        data,
        hyperbola_units=judge.fold_means(hyperbola, judge.hyperbola_key, "hit", folds),
        band_units=judge.fold_means(band, judge.band_key, "coverage", folds),
    )
    bootstrap = judge.Bootstrap(judge.BOOT_SEED, 50)
    results, tables = judge.judge_hypotheses(data, bootstrap)
    assert set(results) == set(judge.HYPOTHESES)
    assert all(r["decision"] == "not_evaluable" for r in results.values())
    report = judge.report_band(data, bootstrap)
    assert set(report) == {"R-a", "R-b", "R-c"}
    assert judge.model_table(data)
    judge.write_units(offline.settings.out_root, tables)
    m_star = results["HH1"]["m_star"]
    assert judge.report_best_scores(data, None)["rd_baseline"] is None
    assert "hh1_all_units" in judge.report_hyperbola(data, m_star, bootstrap)
    assert "r3_unshared_4tu" in judge.report_sensitivity(data, m_star, bootstrap)
    estimates = {name: entry.get("estimate") for name, entry in results.items()}
    assert set(judge.fold_agreement(data, m_star, estimates)) >= {"HH1", "HB1"}
    uninjected = e3x.uninjected_line(
        plans[0],
        {0: TileDetector(), 1: TileDetector()},
        None,
        replace(offline.settings, folds=(0, 1)),
        PP_SETS[PPSetId.S3],
    )
    kinds = {row["kind"] for row in uninjected["rows"]}
    assert kinds == {e3x.KIND_HYPERBOLA, e3x.KIND_BAND}
    baseline = [{k: str(v) for k, v in row.items()} for row in uninjected["rows"]]
    rd = judge.report_best_scores(data, baseline)
    assert "baseline" in rd["band"] or rd["band"]["n_missed_unit_folds"] == 0
    agreement = e3x.agreement_counts([uninjected], {})
    assert agreement["same"] == 0 and agreement["only_e3x"] > 0


def test_offline_seal2_injected_stage_and_judge_run(offline, monkeypatch, tmp_path):
    assert e3x.run_plan(offline.settings, 0.0, offline.context)[0] == "ok"
    plan_dir = offline.settings.stage_dir
    monkeypatch.setattr(e3x, "SEALED_PLAN_DIR", plan_dir)
    record = tmp_path / "seal2.json"
    record.write_bytes((plan_dir / "seal2_record_draft.json").read_bytes())
    argv = [
        "--stage",
        "injected",
        "--out-root",
        str(offline.settings.out_root),
        "--pool",
        str(offline.settings.pool_csv),
        "--seal-record",
        str(record),
        "--device",
        "cpu",
    ]
    settings = e3x.parse_args(argv)
    assert e3x.assert_seal2(settings, offline.context)["result"] == "pass"
    monkeypatch.setattr(e3x.runner, "assert_weights", lambda *_: {})
    monkeypatch.setattr(
        e3x, "load_detectors", lambda s, c: dict.fromkeys(s.folds, TileDetector(c))
    )
    monkeypatch.setattr(e3x, "attach_nms_counter", e3x.WarningCounter)
    status, counts = e3x.run_injected(settings, 0.0, offline.context)
    assert status == "ok" and counts.n_inferences > 0
    summary = json.loads((settings.stage_dir / "summary.json").read_text("utf-8"))
    assert summary["judgment"]["judgeable"] and not summary["defects"].get("zero_test")
    progress = json.loads((settings.stage_dir / "progress.json").read_text("utf-8"))
    assert set(progress) == {
        "stage",
        "done",
        "total",
        "unit",
        "started_at",
        "updated_at",
    }
    assert progress["stage"] == "injected" and progress["done"] == progress["total"]
    assert not (settings.stage_dir / "progress.json.tmp").exists()
    monkeypatch.setattr(
        judge.m33_inputs,
        "git_blobs",
        lambda paths: {"analysis/detection/m33_e3x.py": "blob"},
    )
    args = SimpleNamespace(
        injected=settings.stage_dir,
        uninjected=tmp_path / "none",
        plan=plan_dir,
        seal_record=record,
        out=tmp_path / "judgment",
        n_boot=40,
    )
    document = judge.run(args)
    assert set(document["hypotheses"]) == set(judge.HYPOTHESES)
    assert document["constants"]["CI_LEVEL"] == 1 - 0.05 / 6
    assert not document["n_boot_is_sealed"]
    assert (tmp_path / "judgment" / "units_HB1.csv").is_file()
    with pytest.raises(inputs.InputError):
        judge.run(args)  # the output folder exists
    # v5-3: an un-injected run of another seal record is refused.
    other = tmp_path / "uninjected001"
    other.mkdir()
    foreign = {"run_identity": {"seal2": {"seal_record_sha256_lf": "0" * 64}}}
    (other / "summary.json").write_text(json.dumps(foreign), encoding="utf-8")
    with pytest.raises(inputs.InputError, match="un-injected run"):
        judge.run(replace_namespace(args, uninjected=other, out=tmp_path / "j2"))
    # An un-injected run of the same seal record and blobs is accepted.
    own = json.loads((settings.stage_dir / "summary.json").read_text("utf-8"))
    (other / "summary.json").write_text(json.dumps(own), encoding="utf-8")
    judge.assert_run_identities(own, other, record)
    (plan_dir / "band_arms.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(inputs.InputError):
        e3x.assert_seal2(settings, offline.context)


def replace_namespace(namespace: SimpleNamespace, **changes) -> SimpleNamespace:
    return SimpleNamespace(**{**vars(namespace), **changes})


def test_first_try_outputs_are_refused(offline, tmp_path, capsys):
    first_try = e3x.parse_args(["--stage", "injected", "--device", "cpu"])
    assert first_try.stage_dir == e3x.DEFAULT_OUT_ROOT / "injected"
    for run_stage in (e3x.run_injected, e3x.run_uninjected001):
        with pytest.raises(inputs.InputError, match="first-try output refused"):
            run_stage(first_try, 0.0, offline.context)
    retry = e3x.parse_args(
        ["--stage", "injected", "--out-root", str(e3x.RETRY_OUT_ROOT)]
    )
    assert not e3x.is_first_try_path(retry.stage_dir)
    e3x.assert_retry_output(retry)
    argv = [
        "--injected",
        str(e3x.DEFAULT_OUT_ROOT / "injected"),
        "--uninjected",
        str(tmp_path / "uninjected001"),
        "--out",
        str(tmp_path / "judgment"),
    ]
    assert judge.main(argv) == 2
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == (
        "input_error"
    )
    assert not (tmp_path / "judgment").exists()
    injected = tmp_path / "injected"
    with pytest.raises(inputs.InputError, match="first-try folder"):
        judge.assert_retry_paths(
            injected, e3x.DEFAULT_OUT_ROOT / "uninjected001", tmp_path / "judgment"
        )
    with pytest.raises(inputs.InputError, match="first-try folder"):
        judge.assert_retry_paths(
            injected, tmp_path / "uninjected001", e3x.DEFAULT_OUT_ROOT / "judgment"
        )
    judge.assert_retry_paths(
        injected, tmp_path / "uninjected001", tmp_path / "judgment"
    )


class InProcessPool:
    """``ProcessPoolExecutor`` stand-in: monkeypatches do not reach children."""

    def __init__(self, max_workers: int, initializer) -> None:
        initializer()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def map(self, function, jobs):
        return map(function, jobs)

    def submit(self, function, *args):
        from concurrent.futures import Future

        future = Future()
        future.set_result(function(*args))
        return future


@pytest.fixture
def precheck_run(offline, monkeypatch, tmp_path):
    """Offline precheck: the plan line as the main line, two precheck lines."""
    assert e3x.run_plan(offline.settings, 0.0, offline.context)[0] == "ok"
    monkeypatch.setattr(e3x, "SEALED_PLAN_DIR", offline.settings.stage_dir)
    record = tmp_path / "seal2.json"
    record.write_bytes(
        (offline.settings.stage_dir / "seal2_record_draft.json").read_bytes()
    )
    pool_csv = offline.settings.pool_csv
    rows = [
        {
            "line_id": line_id,
            "dataset": "4tu",
            "cluster_id": "01.1",
            "n_traces": 235,
            "dt_ns": 0.098,
            "time_window_ns": 50.0,
            "shares_r3_activity": True,
            "role": "precheck",
        }
        for line_id in ("01.1/Path4", "01.1/Path5")  # Path3 is the main line
    ]
    pool.write_pool(pool_csv, rows)
    record_data = json.loads(record.read_text("utf-8"))
    record_data["pool_csv_sha256_lf"] = inputs.sha256_file_lf(pool_csv)
    record.write_text(json.dumps(record_data), encoding="utf-8")
    base = offline.context
    loader = e3x.runner.load_line

    def load(ref):
        line = loader(ref)
        return replace(line, radargram=replace(line.radargram, line_id=ref.line_id))

    monkeypatch.setattr(e3x.runner, "load_line", load)
    monkeypatch.setattr(e3x.runner, "assert_weights", lambda *_: {})
    monkeypatch.setattr(
        e3x, "load_detectors", lambda s, c: dict.fromkeys(s.folds, TileDetector(c))
    )
    monkeypatch.setattr(e3x, "attach_nms_counter", e3x.WarningCounter)
    monkeypatch.setattr(e3x, "ProcessPoolExecutor", InProcessPool)
    argv = [
        "--precheck",
        "--workers",
        "2",
        "--out-root",
        str(offline.settings.out_root),
        "--pool",
        str(pool_csv),
        "--seal-record",
        str(record),
        "--device",
        "cpu",
        "--precheck-dir",
        str(tmp_path / "pre"),
    ]
    return SimpleNamespace(settings=e3x.parse_args(argv), context=base)


def _csv_rows(path: Path) -> dict[str, dict[str, str]]:
    return {row["line_id"]: row for row in inputs.read_csv(path)}


def test_offline_precheck_runs_every_check(precheck_run, offline, capsys):
    settings = precheck_run.settings
    status, _ = e3x.run_precheck(settings, 0.0, precheck_run.context)
    pre = settings.stage_dir
    document = json.loads((pre / "precheck.json").read_text("utf-8"))
    cpu = json.loads((pre / e3x.CPU_ASSERTIONS_JSON).read_text("utf-8"))
    assert status == "ok", (document["checks"], cpu["guard_problems"])
    assert all(document["checks"].values()) and document["not_run"] == []
    assert list(document["checks"])[:2] == ["1_identity", "10_cpu_assertions"]
    assert set(document["checks"]) == {
        "1_identity",
        "10_cpu_assertions",
        *e3x.GPU_PRECHECK_CHECKS,
    }
    stdout = capsys.readouterr().out
    assert not e3x.METRIC_WORDS.search(stdout)
    summary = json.loads(
        next(line for line in stdout.splitlines() if "10_cpu_assertions" in line)
    )
    assert set(summary) == {
        "precheck",
        "status",
        "n_lines",
        "n_checks",
        "n_failed",
        "elapsed_s",
    }
    assert summary["n_lines"] == 3 and summary["n_failed"] == 0
    assert cpu["passed"] and not cpu["guard_problems"] and not cpu["failures"]
    for key, counts in cpu["assertions"].items():
        assert counts["n_checked"] == counts["n_expected"] > 0, key
    assert cpu["run_identity"]["started_at"] == e3x._local_time(0.0)
    assert cpu["texture_positions"] and cpu["texture_distribution"]
    # Same counts as the full build_line on the two precheck lines (A-5 4).
    rows = _csv_rows(pre / e3x.CPU_ASSERTIONS_LINES)
    lines = e3x.precheck_pool(settings)
    plans = e3x.read_plan(pre / "plan", lines, inputs.CandidateApexes({}, {}))
    for plan in plans:
        full = e3x.build_line(plan, offline.library, offline.library)
        row = rows[plan.line.line_id]
        assert row["plan_source"] == "precheck"
        for key in e3x.CPU_ASSERTIONS:
            assert int(row[f"{key}_failed"]) == full.defects[key] == 0, key
            assert int(row[f"{key}_checked"]) == full.n_checked[key], key
            assert row[f"{key}_checked"] == row[f"{key}_expected"], key
    progress = json.loads((pre / "progress.json").read_text("utf-8"))
    assert progress["stage"] == "precheck_6" and progress["done"] == progress["total"]
    with pytest.raises(inputs.InputError):
        e3x.run_precheck(settings, 0.0, precheck_run.context)  # the folder exists


def test_offline_precheck10_failure_skips_the_gpu_checks(precheck_run, monkeypatch):
    def no_detector(*_):
        raise AssertionError("a GPU check ran after check 10 failed")

    monkeypatch.setattr(e3x, "load_detectors", no_detector)
    monkeypatch.setattr(e3x, "zero_test_passes", lambda context: False)
    settings = precheck_run.settings
    status, counts = e3x.run_precheck(settings, 0.0, precheck_run.context)
    assert status == "precheck_failed" and counts.n_inferences == 0
    pre = settings.stage_dir
    document = json.loads((pre / "precheck.json").read_text("utf-8"))
    assert document["checks"] == {"1_identity": True, "10_cpu_assertions": False}
    assert document["not_run"] == list(e3x.GPU_PRECHECK_CHECKS)
    assert not (pre / "w1").exists()
    cpu = json.loads((pre / e3x.CPU_ASSERTIONS_JSON).read_text("utf-8"))
    assert cpu["assertions"]["zero_test"]["n_failed"] == 3
    assert {f["assertion"] for f in cpu["failures"]} == {"zero_test"}


def test_offline_precheck1_failure_stops_before_every_other_check(
    precheck_run, monkeypatch
):
    def seal_differs(*_):
        raise inputs.InputError("seal-2 record differs: ['code_blobs']")

    def must_not_run(*_):
        raise AssertionError("a check ran after check 1 failed")

    monkeypatch.setattr(e3x, "assert_seal2", seal_differs)
    for name in ("_precheck_plan", "cpu_assertion_jobs", "load_detectors"):
        monkeypatch.setattr(e3x, name, must_not_run)
    settings = precheck_run.settings
    status, counts = e3x.run_precheck(settings, 0.0, precheck_run.context)
    assert status == "precheck_failed" and counts == e3x.Counts()
    pre = settings.stage_dir
    document = json.loads((pre / "precheck.json").read_text("utf-8"))
    assert document["checks"] == {"1_identity": False}
    assert document["not_run"] == ["10_cpu_assertions", *e3x.GPU_PRECHECK_CHECKS]
    assert document["run_identity"]["started_at"] == e3x._local_time(0.0)
    assert not (pre / e3x.CPU_ASSERTIONS_JSON).exists()
    assert not (pre / "plan").exists() and not (pre / "w1").exists()


def test_offline_precheck10_equals_build_line_and_guards_counts(offline):
    assert e3x.run_plan(offline.settings, 0.0, offline.context)[0] == "ok"
    plan_dir = offline.settings.stage_dir
    plans = e3x.read_plan(plan_dir, [offline.line], inputs.CandidateApexes({}, {}))
    full = e3x.build_line(plans[0], offline.library, offline.library)
    cpu = e3x.build_line(plans[0], offline.library, offline.library, images=False)
    assert cpu.entries == () and full.entries
    assert cpu.defects == full.defects
    assert cpu.n_checked == full.n_checked
    assert cpu.texture_checks == full.texture_checks
    assert cpu.texture_details == full.texture_details
    expected = e3x.planned_check_counts(plan_dir, [offline.line.line_id])
    assert expected[offline.line.line_id] == dict(full.n_checked)
    assert all(full.n_checked[key] > 0 for key in e3x.CPU_ASSERTIONS)
    e3x._init_line_worker()
    result = e3x._cpu_check_worker((plans[0], "main"))
    assert result.n_failed == {key: full.defects[key] for key in e3x.CPU_ASSERTIONS}
    assert not e3x.cpu_assertion_guard([result], expected)
    # The guard: a line with no checks, a missing line, a wrong count.
    empty = replace(result, n_checked=dict.fromkeys(e3x.CPU_ASSERTIONS, 0))
    zero_plan = {result.line_id: dict.fromkeys(e3x.CPU_ASSERTIONS, 0)}
    problems = e3x.cpu_assertion_guard([empty], zero_plan)
    assert len(problems) == len(e3x.CPU_ASSERTIONS)
    assert all(problem.endswith("no check") for problem in problems)
    assert e3x.cpu_assertion_guard([], expected)
    more = {**expected, "other/line": expected[result.line_id]}
    assert e3x.cpu_assertion_guard([result], more)
    off_by_one = replace(result, n_checked={**result.n_checked, "texture_check": 0})
    assert e3x.cpu_assertion_guard([off_by_one], expected)
