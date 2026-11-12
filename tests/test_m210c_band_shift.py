"""M2-10 (c) P3 band-shift QC and P5 header flag (m210_c_qc.md v1.1 section 5).

T1-T6 of section 5 on analytic Ricker sections, and the G0 replays of section
4-3 that need the stored data (opt-in ``data`` marker: the M2-9 eval candidate
rows of four activities and a K7 harness subset). The QC sets flags and fields
only: no depth is ever changed.
"""

import csv
import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from gpr_detection.depth.config import DepthConfig
from gpr_detection.depth.estimate import (
    EPS_META_EQUALS_HEADER_DEFAULT,
    estimate_depth,
    prepare_depth_inputs,
)
from gpr_detection.depth.picks import ApexHint
from gpr_detection.io.radargram import GridBox, Radargram
from gpr_detection.preprocess.timezero import (
    ZERO_SHIFT_LARGE,
    ZERO_SHIFT_NOT_SURFACE,
    ZERO_TRACKS_SURFACE,
    ZeroConfig,
    band_shift_flags,
    band_shift_qc,
    find_time_zero,
)
from synthetic_bscan import cylinder_times, ricker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
F_C_HZ = 500e6
DT_NS = 0.05
DX_M = 0.02
N_TRACES = 60
N_SAMPLES = 600
T_ZERO_NS = 5.0
SHIFT_TOL_NS = 0.01
V_M_PER_NS = 0.299792458 / 3.0
BAND_FLAGS = {ZERO_SHIFT_LARGE, ZERO_TRACKS_SURFACE, ZERO_SHIFT_NOT_SURFACE}
NEW_ZERO_FIELDS = ("band_shift_std_ns", "band_coherence", "band_shift_n_valid")


def _band_section(shifts_ns: np.ndarray) -> np.ndarray:
    """One Ricker band per trace at ``T_ZERO_NS + shift``, no noise."""
    t_ns = np.arange(N_SAMPLES)[:, None] * DT_NS
    return ricker(t_ns - (T_ZERO_NS + shifts_ns)[None, :], F_C_HZ)


def _step(step_ns: float) -> np.ndarray:
    """Half the traces at ``-step / 2``, the other half at ``+step / 2``."""
    return np.where(np.arange(N_TRACES) < N_TRACES // 2, -step_ns / 2, step_ns / 2)


def _qc(shifts_ns: np.ndarray, mask: np.ndarray | None = None):
    mask = np.ones(N_TRACES, dtype=bool) if mask is None else mask
    return band_shift_qc(_band_section(shifts_ns), DT_NS, F_C_HZ, T_ZERO_NS, mask)


# ----------------------------------------------------------------- T1-T3


@pytest.mark.parametrize(
    ("step_ns", "flags"),
    [
        (0.38, ()),  # sigma_s 0.19 <= 0.2
        (0.44, (ZERO_SHIFT_LARGE, ZERO_TRACKS_SURFACE)),  # sigma_s 0.22 > 0.2
    ],
)
def test_t1_step_shifts_and_flags_at_the_drift_threshold(step_ns, flags) -> None:
    shift = _qc(_step(step_ns))
    np.testing.assert_allclose(shift.shifts_ns, _step(step_ns), atol=SHIFT_TOL_NS)
    assert shift.std_ns == pytest.approx(step_ns / 2, abs=SHIFT_TOL_NS)
    assert shift.n_valid == N_TRACES
    assert shift.flags == flags
    assert shift.c1 >= shift.c0


def test_t2_large_shift_that_does_not_stack_better_is_not_surface() -> None:
    """``C1 <= C0`` -> ``zero_shift_not_surface``.

    Every lag maximises the trace's correlation with the reference, so on
    analytic sections the shifted stack is (nearly) always more coherent; no
    mixed-band fixture with ``C1 <= C0`` was found (stationary + moving bands,
    inverted polarity, noise). The branch is tested on the decision function.
    """
    not_surface = (ZERO_SHIFT_LARGE, ZERO_SHIFT_NOT_SURFACE)
    assert band_shift_flags(0.3, 0.8, 0.8, 0.2) == not_surface
    assert band_shift_flags(0.3, 0.8, 0.7, 0.2) == not_surface
    assert band_shift_flags(0.3, 0.8, float("nan"), 0.2) == not_surface
    assert band_shift_flags(0.3, 0.7, 0.8, 0.2) == (
        ZERO_SHIFT_LARGE,
        ZERO_TRACKS_SURFACE,
    )
    assert band_shift_flags(0.2, 0.7, 0.8, 0.2) == ()


def test_t3_shift_beyond_half_a_period_is_left_out() -> None:
    shifts = np.zeros(N_TRACES)
    shifts[5] = 1.3  # > 0.5 / f_c = 1.0 ns: the best lag sits on the search end
    shift = _qc(shifts)
    assert np.isnan(shift.shifts_ns[5])
    assert np.isfinite(np.delete(shift.shifts_ns, 5)).all()
    assert shift.n_valid == N_TRACES - 1
    assert shift.std_ns == pytest.approx(0.0, abs=1e-6)
    assert shift.flags == ()


def test_excluded_traces_are_not_counted() -> None:
    mask = np.ones(N_TRACES, dtype=bool)
    mask[:10] = False
    shift = _qc(_step(0.44), mask)
    assert np.isnan(shift.shifts_ns[:10]).all()
    assert shift.n_valid == N_TRACES - 10


# ----------------------------------------------------------------- lines


STEP_TRACE = 80
STEP_NS = 0.5
PIPES = (40, 120)
PIPE_DEPTH_M = 0.6
BAND_NS = 3.0


def _line(step_ns: float) -> Radargram:
    """Merged band and two pipes, all delayed ``step_ns`` right of trace 80."""
    n_traces = 161
    x_m = np.arange(n_traces) * DX_M
    delay = np.where(np.arange(n_traces) >= STEP_TRACE, step_ns, 0.0)
    t_ns = np.arange(N_SAMPLES)[:, None] * DT_NS
    data = 30.0 * ricker(t_ns - (BAND_NS + delay)[None, :], F_C_HZ)
    for apex in PIPES:
        times = cylinder_times(x_m, x_m[apex], PIPE_DEPTH_M, V_M_PER_NS, 0.0)
        data += ricker(t_ns - (BAND_NS + times + delay)[None, :], F_C_HZ)
    return Radargram(
        data=data.astype(np.float32), dt_ns=DT_NS, dx_m=DX_M, air_gap="air",
        amplitude_processed=False, dataset="unit", line_id="unit/height_step",
        f_c_hz=F_C_HZ, eps_r_meta=9.0, eps_r_meta_source="synthetic_truth",
    )  # fmt: skip


def _depths(radargram: Radargram, config: DepthConfig):
    inputs = prepare_depth_inputs(radargram, config)
    results = []
    for apex in PIPES:
        delay = STEP_NS if apex >= STEP_TRACE else 0.0
        k_apex = (BAND_NS + 2 * PIPE_DEPTH_M / V_M_PER_NS + delay) / DT_NS
        box = GridBox(apex - 25, apex + 25, int(k_apex) - 20, int(k_apex) + 30)
        hint = ApexHint(trace=apex, sample=k_apex)
        results.append(
            estimate_depth(radargram, box, config, inputs=inputs, apex_hint=hint)
        )
    return inputs.zero, results


def test_height_step_line_is_flagged_and_the_depth_is_unchanged() -> None:
    radargram = _line(STEP_NS)
    zero_off, off = _depths(radargram, DepthConfig())
    zero_on, on = _depths(radargram, DepthConfig(band_shift_qc=True))
    assert {ZERO_SHIFT_LARGE, ZERO_TRACKS_SURFACE} <= set(zero_on.flags)
    assert zero_on.band_shift_std_ns == pytest.approx(STEP_NS / 2, abs=SHIFT_TOL_NS)
    assert zero_on.band_shift_n_valid == radargram.n_traces
    assert zero_on.t_zero_per_trace_ns is None  # never filled (v1.1)
    assert zero_on.t_zero_ns == zero_off.t_zero_ns
    for old, new in zip(off, on, strict=True):
        assert dataclasses.replace(new, flags=old.flags) == old
        assert [f for f in new.flags if f not in old.flags] == [
            ZERO_SHIFT_LARGE,
            ZERO_TRACKS_SURFACE,
        ]


# ----------------------------------------------------------------- T4, T5


def test_t4_no_qc_on_processed_data_failed_zero_or_without_scale() -> None:
    radargram = _line(STEP_NS)
    config = ZeroConfig(band_shift_qc=True)
    cases = {
        "processed": dataclasses.replace(radargram, amplitude_processed=True),
        "no_scale": dataclasses.replace(radargram, dt_ns=None),
        "zero_failed": dataclasses.replace(
            radargram, data=np.zeros_like(radargram.data)
        ),
    }
    for name, line in cases.items():
        zero = find_time_zero(line, line.data, config=config)
        assert all(getattr(zero, field) is None for field in NEW_ZERO_FIELDS), name
        assert not BAND_FLAGS & set(zero.flags), name
    failed = find_time_zero(
        cases["zero_failed"], cases["zero_failed"].data, config=config
    )
    assert failed.failed


def test_t5_default_runs_no_qc_and_keeps_every_value() -> None:
    radargram = _line(STEP_NS)
    zero, results = _depths(radargram, DepthConfig())
    assert all(getattr(zero, field) is None for field in NEW_ZERO_FIELDS)
    assert not BAND_FLAGS & set(zero.flags)
    explicit_zero, explicit = _depths(
        radargram, DepthConfig(band_shift_qc=False, flag_header_eps=False)
    )
    assert explicit == results
    assert explicit_zero.flags == zero.flags


def test_t5_still_band_on_gives_identical_depths() -> None:
    radargram = _line(0.0)
    zero_off, off = _depths(radargram, DepthConfig())
    zero_on, on = _depths(radargram, DepthConfig(band_shift_qc=True))
    assert zero_on.band_shift_std_ns == pytest.approx(0.0, abs=1e-6)
    assert zero_on.flags == zero_off.flags
    assert on == off


# ----------------------------------------------------------------- T6


@pytest.mark.parametrize(
    ("eps_header", "is_flagged"),
    [(9.0, True), (7.0, False), (None, False)],
)
def test_t6_header_eps_flag_only_when_equal(eps_header, is_flagged) -> None:
    radargram = dataclasses.replace(_line(0.0), eps_r_header=eps_header)
    _, plain = _depths(radargram, DepthConfig())
    _, flagged = _depths(radargram, DepthConfig(flag_header_eps=True))
    for old, new in zip(plain, flagged, strict=True):
        extra = [f for f in new.flags if f not in old.flags]
        assert extra == ([EPS_META_EQUALS_HEADER_DEFAULT] if is_flagged else [])
        assert dataclasses.replace(new, flags=old.flags) == old


@pytest.mark.data
def test_t6_fourtu_loader_fills_the_header_eps() -> None:
    from gpr_detection.io.fourtu import load_fourtu_radargram
    from gpr_detection.io.paths import DatasetName, load_dataset_paths

    root = load_dataset_paths().root(DatasetName.FOURTU)
    if not root.is_dir():
        pytest.skip("4TU data not available")
    radargram = load_fourtu_radargram(root, "01.1", "Path1")
    assert radargram.eps_r_header == float(radargram.metadata["eps_r_header"])
    assert radargram.eps_r_meta == float(radargram.metadata["eps_r_used"])


# ----------------------------------------------------------------- eval --diag-out


@pytest.fixture(scope="module")
def evaluation():
    return _load("eval_depth_4tu_diag", PROJECT_ROOT / "scripts" / "eval_depth_4tu.py")


def test_diag_options_need_a_diag_folder(evaluation) -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--band-shift-qc"])
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--flag-header-eps"])
    args = evaluation.parse_args(["--band-shift-qc", "--diag-out", "x"])
    assert args.band_shift_qc and args.diag_out == Path("x")


def test_line_diagnostic_row_has_the_section_7_columns(evaluation) -> None:
    radargram = dataclasses.replace(
        _line(STEP_NS), line_id="011.14/Profile_1", eps_r_header=9.0
    )
    config = DepthConfig(band_shift_qc=True, flag_header_eps=True)
    row = evaluation.line_diagnostic_row(radargram, config)
    assert tuple(row) == evaluation.LINE_DIAG_COLUMNS
    assert row["site"] == 11 and row["path"] == "Profile_1"
    assert row[ZERO_SHIFT_LARGE] is True and row[ZERO_TRACKS_SURFACE] is True
    assert row[ZERO_SHIFT_NOT_SURFACE] is False
    assert row[EPS_META_EQUALS_HEADER_DEFAULT] is True
    assert row["band_shift_n_valid"] == radargram.n_traces
    assert row["pipe_time_std_ref_ns"] == pytest.approx(
        2.5 * row["band_shift_std_ns"], abs=1e-3
    )
    plain = evaluation.line_diagnostic_row(radargram, DepthConfig())
    assert plain["band_shift_std_ns"] == "" and plain[ZERO_SHIFT_LARGE] is False
    assert plain[EPS_META_EQUALS_HEADER_DEFAULT] is False


@pytest.mark.data
def test_diag_out_writes_only_the_line_csv(evaluation, tmp_path) -> None:
    from gpr_detection.io.paths import DatasetName, load_dataset_paths

    if not load_dataset_paths().root(DatasetName.FOURTU).is_dir():
        pytest.skip("4TU data not available")
    out = tmp_path / "diag"
    default_out = tmp_path / "x"
    argv = [
        "--band-shift-qc", "--flag-header-eps", "--diag-out", str(out),
        "--activities", "01.1", "--prefix", "t", "--out-dir", str(default_out),
    ]  # fmt: skip
    assert evaluation.main(argv) == 0
    assert sorted(p.name for p in out.iterdir()) == ["t_line_diag.csv"]
    assert not default_out.exists()
    with (out / "t_line_diag.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and {r["activity_id"] for r in rows} == {"01.1"}
    assert all(r["band_shift_std_ns"] != "" for r in rows)


# ----------------------------------------------------------------- G0 (data)


G0_EVAL_ACTIVITIES = ("01.1", "04.1", "011.2", "011.14")
"""Four activities (011.14: a zero_window_drift site-11 activity)."""
M29_ARGS = (
    "--cluster-width-m", "0.10", "--cluster-linkage", "complete",
    "--min-support-lines", "2", "--baseline-trials", "20",
    "--baseline-trials-aux", "10", "--baseline-seed", "0", "--apex-qc", "on",
)  # fmt: skip
"""The depth_4tu_v2 arguments; fewer baseline trials (candidate rows do not
depend on them)."""


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rows(path: Path, activities: set[str]) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        table = list(csv.reader(handle))
    return [table[0], *(row for row in table[1:] if row[0] in activities)]


@pytest.mark.data
def test_g0_m29_eval_candidate_rows_are_unchanged(tmp_path) -> None:
    committed = PROJECT_ROOT / "docs" / "processing" / "depth_4tu_v2_candidates.csv"
    evaluation = _load(
        "eval_depth_4tu_g0", PROJECT_ROOT / "scripts" / "eval_depth_4tu.py"
    )
    from gpr_detection.io.paths import DatasetName, load_dataset_paths

    if not load_dataset_paths().root(DatasetName.FOURTU).is_dir():
        pytest.skip("4TU data not available")
    evaluation.main(
        [*M29_ARGS, "--out-dir", str(tmp_path), "--prefix", "g0",
         "--activities", ",".join(G0_EVAL_ACTIVITIES)]
    )  # fmt: skip
    wanted = set(G0_EVAL_ACTIVITIES)
    regenerated = _rows(tmp_path / "g0_candidates.csv", wanted)
    assert len(regenerated) > 1
    assert regenerated == _rows(committed, wanted)


@pytest.mark.data
def test_g0_k7_harness_subset_is_unchanged(tmp_path) -> None:
    """K7 harness default path, model K7_P1_D060, 2 seeds, ``cur`` vs runs_k7.csv."""
    k7 = _load(
        "m210_k7_pipeline_g0",
        PROJECT_ROOT / "analysis" / "processing" / "m210_k7_pipeline.py",
    )
    stored = PROJECT_ROOT / "data" / "processing" / "m210_k7" / "runs_k7.csv"
    if not stored.exists():
        pytest.skip("runs_k7.csv not available")
    model = "K7_P1_D060"
    plan = k7.K7Plan(model_ids=(model,), n_seeds=2, variants=("cur",),
                     calibration_pairs={})  # fmt: skip
    events = k7.json_models(k7.DEFAULT_MODELS_JSON)["events"][model]
    models, calibrations = k7.clean_models(plan, k7.DEFAULT_DATA_ROOT, k7.load_model)
    rows = k7.model_rows(models[model], calibrations[model], plan, events)
    subset = tmp_path / "subset.csv"
    k7.write_runs(subset, rows)
    keys = ("model_id", "rho", "seed", "noise_free", "variant", "pipe_index")
    stored_rows = {tuple(r[c] for c in keys): r for r in k7._csv_rows(stored)}
    regenerated = k7._csv_rows(subset)
    assert len(regenerated) == 2 * 2 + 1
    for row in regenerated:
        assert stored_rows[tuple(row[c] for c in keys)] == row
