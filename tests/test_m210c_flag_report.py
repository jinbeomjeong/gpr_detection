"""M2-10 (c) P3 flag report (m210_c_qc.md v1.1 C-7): seeds, follow slope, rows."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from gpr_detection.io.radargram import Radargram
from synthetic_bscan import cylinder_times, ricker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "analysis" / "processing" / "m210c_flag_report.py"
F_C_HZ = 500e6
DT_NS = 0.05
DX_M = 0.02
N_TRACES = 161
N_SAMPLES = 600
STEP_NS = 0.5


@pytest.fixture(scope="module")
def report():
    spec = importlib.util.spec_from_file_location("m210c_flag_report", REPORT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _delay() -> np.ndarray:
    return np.where(np.arange(N_TRACES) >= 80, STEP_NS, 0.0)


def _line() -> Radargram:
    """Band and a pipe that follow a 0.5 ns height step (full follow)."""
    x_m = np.arange(N_TRACES) * DX_M
    t_ns = np.arange(N_SAMPLES)[:, None] * DT_NS
    delay = _delay()
    data = 30.0 * ricker(t_ns - (3.0 + delay)[None, :], F_C_HZ)
    times = cylinder_times(x_m, x_m[40], 0.6, 0.1, 0.0)
    data += ricker(t_ns - (3.0 + times + delay)[None, :], F_C_HZ)
    return Radargram(
        data=data.astype(np.float32), dt_ns=DT_NS, dx_m=DX_M, air_gap="air",
        amplitude_processed=False, dataset="unit", line_id="C/Z_HSTEP",
        f_c_hz=F_C_HZ, eps_r_meta=9.0, eps_r_meta_source="synthetic_truth",
    )  # fmt: skip


def test_seeds_have_no_overlap(report) -> None:
    record = report.seed_record()
    assert record["seed_overlap"] == 0
    assert record["n_seeds"] == 4 * 2 * 20 + 1
    assert report.noise_seed(3, 1, 19) == 20311112 + 3000 + 100 + 19
    assert {"k7_harness", "a_eps_bias", "v3"} <= set(record["checked_against"])


def test_follow_slope(report) -> None:
    expected = np.linspace(-0.3, 0.3, 11)
    shifts = 0.4 * expected
    shifts[3] = np.nan
    assert report.follow_slope(shifts, expected) == pytest.approx(0.4)
    assert report.follow_slope(shifts, None) is None
    assert report.follow_slope(shifts, np.zeros(11)) is None


def test_line_row_on_a_fully_following_band(report) -> None:
    expected = _delay() - np.mean(_delay())
    model = report.ReportModel(
        model_id="Z_HSTEP", role="effect", radargram=_line(),
        sigma_raw_by_rho={10: 0.0, 20: 0.0},
        expected_geometry_ns=expected, expected_band_peak_ns=expected,
    )  # fmt: skip
    row = report.line_row(model, model.radargram, None, None, None)
    assert tuple(row) == report.ROW_COLUMNS
    assert row["zero_shift_large"] is True and row["zero_tracks_surface"] is True
    assert row["follow_slope_geometry"] == pytest.approx(1.0, abs=0.02)
    assert row["band_shift_n_valid"] == N_TRACES
    summary = report.summarise([row])
    rates = summary["Z_HSTEP|noise_free"]["flag_rates"]
    assert rates == {"zero_shift_large": 1.0, "zero_tracks_surface": 1.0,
                     "zero_shift_not_surface": 0.0}  # fmt: skip
