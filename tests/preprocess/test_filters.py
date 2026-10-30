"""Unit tests of the fitting-branch filters (spec_depth_v1 section 3, M2-1)."""

import numpy as np
import pytest

from gpr_detection.preprocess.filters import (
    agc,
    bandpass,
    bandpass_corners_hz,
    bandpass_px,
    dewow,
    dewow_sigma_samples,
    estimate_period_px,
    prepare_fitting_branch,
    remove_background,
    remove_dc,
)
from gpr_detection.preprocess.signals import parabolic_peak, pick_signal
from synthetic_bscan import DT_NS, F_C_HZ, BScanSpec, make_radargram, ricker

N_SAMPLES = 800
TIME_NS = np.arange(N_SAMPLES) * DT_NS


def _envelope_peak_ns(trace: np.ndarray) -> float:
    envelope = pick_signal(trace, "envelope")
    index, _ = parabolic_peak(envelope, int(np.argmax(envelope)))
    return index * DT_NS


def test_remove_dc_zeroes_each_trace_mean() -> None:
    data = np.column_stack([np.full(10, 5.0), np.arange(10.0)])
    np.testing.assert_allclose(remove_dc(data).mean(axis=0), 0.0, atol=1e-12)


def test_dewow_sigma_is_one_period() -> None:
    assert dewow_sigma_samples(0.05, 500e6) == pytest.approx(40.0)  # 2 ns / 0.05
    assert dewow_sigma_samples(0.1, None) == pytest.approx(20.0)  # default 2 ns
    assert dewow_sigma_samples(0.05, 400e6) == pytest.approx(50.0)  # 2.5 ns


def test_dewow_removes_slow_wow_and_keeps_pulse_time() -> None:
    pulse = ricker(TIME_NS - 15.0, F_C_HZ)
    wow = 0.8 * np.exp(-TIME_NS / 20.0)
    filtered = dewow((pulse + wow)[:, None], DT_NS, F_C_HZ)[:, 0]
    late = TIME_NS > 30.0
    assert np.abs(filtered[late]).max() < 0.05
    assert _envelope_peak_ns(filtered) == pytest.approx(15.0, abs=0.02)


def test_bandpass_corners_follow_spec_and_nyquist_cap() -> None:
    assert bandpass_corners_hz(0.05, 500e6) == pytest.approx((125e6, 1e9))
    # 0.45 / 0.5 ns = 0.9 GHz caps 2 f_c = 1 GHz
    assert bandpass_corners_hz(0.5, 500e6) == pytest.approx((125e6, 0.9e9))


def test_bandpass_is_zero_phase() -> None:
    pulse = ricker(TIME_NS - 12.345, F_C_HZ)
    rng = np.random.default_rng(0)
    noisy = pulse + 0.3 * np.sin(2 * np.pi * 0.02 * TIME_NS) + rng.normal(0, 0.02, 800)
    filtered = bandpass(noisy[:, None], DT_NS, F_C_HZ)[:, 0]
    assert _envelope_peak_ns(filtered) == pytest.approx(12.345, abs=0.02)


def test_bandpass_rejects_empty_band() -> None:
    with pytest.raises(ValueError, match="empty pass band"):
        bandpass(np.zeros((10, 2)), dt_ns=5.0, f_c_hz=500e6)


def test_median_background_removes_flat_event_and_keeps_hyperbola() -> None:
    spec = BScanSpec(snr_db=None)
    radargram = make_radargram(spec)
    removed = remove_background(radargram.data, window_traces=301)
    direct_rows = slice(int(3.0 / DT_NS), int(4.5 / DT_NS))
    assert np.abs(removed[direct_rows]).max() < 0.05 * spec.direct_amplitude
    apex_trace = spec.n_traces // 2
    assert np.abs(removed[:, apex_trace]).max() > 0.9 * spec.pipe_amplitude


def test_median_background_window_shifts_inward_at_profile_ends() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(20, 50))
    removed = remove_background(data, window_traces=11)
    first_window_median = np.median(data[:, :11], axis=1)
    np.testing.assert_allclose(data[:, 0] - removed[:, 0], first_window_median)
    centre_median = np.median(data[:, 20:31], axis=1)
    np.testing.assert_allclose(data[:, 25] - removed[:, 25], centre_median)
    last_window_median = np.median(data[:, -11:], axis=1)
    np.testing.assert_allclose(data[:, -1] - removed[:, -1], last_window_median)


def test_even_background_window_is_raised_to_odd() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(20, 50))
    np.testing.assert_array_equal(
        remove_background(data, window_traces=10),
        remove_background(data, window_traces=11),
    )


def test_agc_equalises_early_and_late_amplitudes() -> None:
    trace = ricker(TIME_NS - 5.0, F_C_HZ) + 0.01 * ricker(TIME_NS - 30.0, F_C_HZ)
    gained = agc(trace[:, None], DT_NS, F_C_HZ)[:, 0]
    early = np.abs(gained[TIME_NS < 10]).max()
    late = np.abs(gained[(TIME_NS > 25) & (TIME_NS < 35)]).max()
    assert late / early > 0.5


def test_prepare_fitting_branch_runs_steps_in_order() -> None:
    branch = prepare_fitting_branch(make_radargram(BScanSpec()))
    assert branch.steps == ("dc", "dewow", "bandpass", "background_median")
    assert branch.filtered.shape == branch.background_removed.shape


def test_prepare_fitting_branch_uses_a_pixel_bandpass_without_scale() -> None:
    radargram = make_radargram(BScanSpec(), dt_ns=None, dx_m=None, f_c_hz=None)
    branch = prepare_fitting_branch(radargram)
    assert branch.steps == ("dc", "bandpass_px", "background_median")
    assert branch.flags == ()


def test_prepare_fitting_branch_skips_the_pixel_bandpass_out_of_range() -> None:
    long_wave = np.sin(2 * np.pi * np.arange(800) / 400.0)  # 400 px period
    data = np.tile(long_wave[:, None], (1, 10)).astype(np.float32)
    radargram = make_radargram(
        BScanSpec(), data=data, dt_ns=None, dx_m=None, f_c_hz=None
    )
    branch = prepare_fitting_branch(radargram)
    assert branch.steps == ("dc", "background_median")
    assert branch.flags == ("bandpass_px_skipped",)


def test_estimate_period_px_finds_the_ricker_period() -> None:
    trace = ricker(TIME_NS - 15.0, F_C_HZ)  # 2.5 ns period = 50 samples
    data = remove_dc(np.tile(trace[:, None], (1, 5)))
    assert estimate_period_px(data) == pytest.approx(50.0, rel=0.05)


def test_estimate_period_px_rejects_out_of_range_or_flat_data() -> None:
    short = np.tile(np.cos(np.pi * np.arange(64))[:, None], (1, 3))  # 2 px period
    assert estimate_period_px(short) is None
    assert estimate_period_px(np.zeros((64, 3))) is None


def test_bandpass_px_is_zero_phase() -> None:
    pulse = ricker(TIME_NS - 12.345, F_C_HZ)
    filtered = bandpass_px(pulse[:, None], period_px=50.0)[:, 0]
    assert _envelope_peak_ns(filtered) == pytest.approx(12.345, abs=0.02)


def test_prepare_fitting_branch_keeps_upstream_processed_data() -> None:
    radargram = make_radargram(BScanSpec(), amplitude_processed=True)
    branch = prepare_fitting_branch(radargram)
    assert branch.steps == ("upstream",)
    np.testing.assert_array_equal(branch.filtered, radargram.data)
