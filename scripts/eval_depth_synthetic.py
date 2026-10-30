"""Depth regression on the 20 gprMax synthetic models (CHECKLIST M2-5, spec 11-2).

For every model the script loads survey metadata only (``load_synthetic``),
supplies a box from the ground truth (spec 11-2 "box supply rule"), runs
``estimate_depth`` with the detected zero and judges R1-R13. Ground-coupled
models also run with the tabulated direct-wave shift (R9) and with the radius
withheld (R11). Truth is read separately (``load_synthetic_truth``) and only
used for the box and for judging.

Outputs (``--out-dir``, default ``docs/processing``):
``<stem>.csv`` one row per model, ``<stem>_checks.csv`` one row per check,
``<stem>.md`` summary and ``<stem>_*.png`` figures.

The box helper lives here, not in ``src``, because it depends on the truth.
``tests/depth/test_regression_synthetic.py`` imports this file.

Example:
    conda run -n gpr-dev python -I scripts/eval_depth_synthetic.py
"""

import argparse
import csv
import logging
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from gpr_detection.depth.config import DepthConfig
from gpr_detection.depth.estimate import (
    DepthInputs,
    estimate_depth,
    prepare_depth_inputs,
)
from gpr_detection.depth.hyperbola import fit_curvature, fixed_velocity_apex
from gpr_detection.depth.picks import HyperbolaPicks, extract_picks
from gpr_detection.depth.refraction import (
    RefractionQC,
    refraction_qc,
    surface_reflection_time,
)
from gpr_detection.depth.result import DepthResult
from gpr_detection.io.paths import DEFAULT_SYNTHETIC_ROOT
from gpr_detection.io.radargram import GridBox, Radargram
from gpr_detection.io.synthetic import (
    SyntheticTruth,
    list_synthetic_models,
    load_synthetic,
    load_synthetic_background,
    load_synthetic_truth,
)
from gpr_detection.preprocess.filters import prepare_fitting_branch
from gpr_detection.preprocess.signals import parabolic_peak, pick_signal
from gpr_detection.preprocess.timezero import GroundZeroShift

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "processing"
DEFAULT_STEM = "depth_synthetic_v1"

BOX_HALF_WIDTH_M = 0.8
BOX_TIME_MARGIN_PERIODS = 1.5
"""Box supply rule (spec 11-2): +-0.8 m around the apex, 1.5 periods in time."""

SINGLE_EVENT_GAP_M = 0.1
TWO_EVENT_GAP_M = 0.3
"""Air-gap groups of the synthetic set: h 0.1 (R3, R7) and h 0.3 (M20)."""

BACKGROUND_APEX_COLUMN = 1
"""``_bg.npy`` column at json x 1.5 m = trace 55, the apex trace (spec 11-2 v1.6)."""

RAY_QC_EPS_TOLERANCE = 0.1
RAY_QC_DEPTH_TOLERANCE_M = 0.01
DATA_QC_EPS_REL_TOLERANCE = 0.20
DATA_QC_DEPTH_TOLERANCE_M = 0.05
"""R13a-1 (exact ray picks) and R13a-2 (measured picks) criteria, spec v1.6."""

MATCH_V_REL = 0.05
MATCH_DEPTH_M = 0.02
ACCEPT_DEPTH_M = 0.10
"""Verification grades: match <= 5 % and 2 cm, acceptable match d <= 10 cm."""

LOGGER = logging.getLogger("eval_depth_synthetic")


# ----------------------------------------------------------------- box supply


def supply_box(truth: SyntheticTruth, radargram: Radargram) -> GridBox:
    """Ground-truth box of spec 11-2 (index-based, clipped to the record).

    Traces ``i_a +- ceil(0.8 m / dx)``; top ``floor((t_apex - 1.5/f_c) / dt)``;
    bottom ``ceil((max(t_src + tau_ray + lag) + 1.5/f_c) / dt)`` over the box
    traces.
    """
    assert radargram.dt_ns is not None and radargram.dx_m is not None
    assert radargram.f_c_hz is not None
    dt_ns, period_ns = radargram.dt_ns, 1e9 / radargram.f_c_hz
    margin_ns = BOX_TIME_MARGIN_PERIODS * period_ns
    half_width = math.ceil(BOX_HALF_WIDTH_M / radargram.dx_m - 1e-9)
    i0 = max(0, truth.apex_trace_index - half_width)
    i1 = min(radargram.n_traces - 1, truth.apex_trace_index + half_width)
    lag_ns = truth.wavelet_lag_ns or 0.0
    ray_ns = np.asarray(truth.ray_reflection_time_ns[i0 : i1 + 1])
    latest_ns = float(np.max(truth.source_peak_ns + ray_ns + lag_ns))
    k0 = max(0, math.floor((truth.apex_envelope_peak_ns - margin_ns) / dt_ns))
    k1 = min(radargram.n_samples - 1, math.ceil((latest_ns + margin_ns) / dt_ns))
    return GridBox(i0=i0, i1=i1, k0=k0, k1=k1)


# ----------------------------------------------------------------- per model


@dataclass(frozen=True)
class ModelRun:
    """All results of one model.

    Attributes:
        truth: Ground truth.
        box: Supplied box.
        inputs: Preprocessing and detected zero (default config).
        main: ``estimate_depth`` with eps_r and radius from the json.
        shifted: Ground only: run with ``ground_zero_shift="synthetic_table"`` (R9).
        radius_free: Ground only: run without the radius (R11).
        picks: Picks of the main run (for R12/R13).
        ideal_apex_ns: R5 reference: apex-trace envelope peak of the ideal
            background-free data ``D - bg`` after steps (1)-(3) (spec v1.6).
        background_lateral_rel: ``max|bg_side - bg_middle| / max|bg_middle|``
            of the no-pipe run (checks that one bg column fits every trace).
        refraction_true_h: M20 only: QC on the measured picks with the json h
            (R13a-2).
        refraction_ray_picks: M20 only: QC on the json ray times (R13a-1).
        straight_ray_refused: M20 only: ``fit_curvature`` raised ``ValueError``.
        seconds: Wall time of this model.
    """

    truth: SyntheticTruth
    box: GridBox
    inputs: DepthInputs
    main: DepthResult
    shifted: DepthResult | None
    radius_free: DepthResult | None
    picks: HyperbolaPicks
    ideal_apex_ns: float | None
    background_lateral_rel: float
    refraction_true_h: RefractionQC | None
    refraction_ray_picks: RefractionQC | None
    straight_ray_refused: bool | None
    seconds: float

    @property
    def model_id(self) -> str:
        return self.truth.model_id

    @property
    def is_ground(self) -> bool:
        return self.truth.air_gap_m == 0


def _picks_of(
    radargram: Radargram, box: GridBox, inputs: DepthInputs
) -> HyperbolaPicks:
    return extract_picks(
        inputs.branch.background_removed,
        box,
        inputs.zero,
        dt_ns=radargram.dt_ns,
        dx_m=radargram.dx_m,
        f_c_hz=radargram.f_c_hz,
    )


def _straight_ray_refused(picks: HyperbolaPicks) -> bool:
    try:
        fit_curvature(picks, radius_m=0.0, air_gap="air")
    except ValueError:
        return True
    return False


def _refraction_with_true_h(
    radargram: Radargram, picks: HyperbolaPicks, truth: SyntheticTruth
) -> RefractionQC:
    """R13a-2: the QC on the measured picks with the json antenna height."""
    vertex = fixed_velocity_apex(picks)
    return refraction_qc(
        picks.x_m,
        picks.tau_ns,
        x0_m=vertex.x0_m,
        tau_apex_ns=vertex.t0_ns,
        h_m=truth.air_gap_m,
        offset_m=radargram.antenna_offset_m or 0.0,
        radius_m=radargram.pipe_radius_m or 0.0,
    )


def _refraction_on_ray_times(
    radargram: Radargram, truth: SyntheticTruth
) -> RefractionQC:
    """R13a-1: the QC algorithm on exact json ray times (all traces).

    ``tau = ray_time - 2 sqrt(h^2 + s^2/4) / c``: both are geometric times, so
    the source delay and wavelet lag cancel. The surface term is computed with
    the same formula as json ``surface_reflection_geom_ns``.
    """
    assert radargram.dx_m is not None
    ray_ns = np.asarray(truth.ray_reflection_time_ns)
    tau_ns = ray_ns - surface_reflection_time(truth.air_gap_m, truth.antenna_offset_m)
    x_m = np.arange(len(ray_ns)) * radargram.dx_m
    return refraction_qc(
        x_m,
        tau_ns,
        x0_m=truth.apex_trace_index * radargram.dx_m,
        tau_apex_ns=float(tau_ns[truth.apex_trace_index]),
        h_m=truth.air_gap_m,
        offset_m=truth.antenna_offset_m,
        radius_m=truth.pipe_radius_m,
    )


def _ideal_section(radargram: Radargram, background: np.ndarray) -> np.ndarray:
    """``D - bg`` (middle no-pipe trace) after the fitting steps (1)-(3)."""
    middle = background[:, [BACKGROUND_APEX_COLUMN]]
    ideal = replace(radargram, data=(radargram.data - middle).astype(np.float32))
    return prepare_fitting_branch(ideal).filtered


def _first_envelope_peak_ns(
    trace: np.ndarray, box: GridBox, dt_ns: float, candidate_rel: float
) -> float | None:
    """First envelope peak >= ``candidate_rel`` x box maximum (candidate rule)."""
    envelope = pick_signal(trace, "envelope")
    window = envelope[box.sample_slice]
    peaks, _ = find_peaks(window, height=candidate_rel * float(window.max()))
    if len(peaks) == 0:
        return None
    k_local, _ = parabolic_peak(window, int(peaks[0]))
    return (box.k0 + k_local) * dt_ns


def _ideal_apex_and_lateral(
    radargram: Radargram, truth: SyntheticTruth, box: GridBox, root: Path
) -> tuple[float | None, float]:
    """R5 reference time and the lateral spread of the no-pipe background."""
    background = load_synthetic_background(truth.model_id, root=root).astype(float)
    middle = background[:, BACKGROUND_APEX_COLUMN]
    sides = np.delete(background, BACKGROUND_APEX_COLUMN, axis=1)
    lateral_rel = float(np.abs(sides - middle[:, None]).max() / np.abs(middle).max())
    ideal = _ideal_section(radargram, background)
    assert radargram.dt_ns is not None
    reference_ns = _first_envelope_peak_ns(
        ideal[:, truth.apex_trace_index],
        box,
        radargram.dt_ns,
        DepthConfig().candidate_rel,
    )
    return reference_ns, lateral_rel


def evaluate_model(model_id: str, root: Path = DEFAULT_SYNTHETIC_ROOT) -> ModelRun:
    """Run every estimate one model needs for R1-R13."""
    started = time.perf_counter()
    truth = load_synthetic_truth(model_id, root=root)
    radargram = load_synthetic(model_id, root=root)
    box = supply_box(truth, radargram)
    inputs = prepare_depth_inputs(radargram)
    main = estimate_depth(radargram, box, inputs=inputs)
    picks = _picks_of(radargram, box, inputs)
    ideal_apex_ns, lateral_rel = _ideal_apex_and_lateral(radargram, truth, box, root)

    shifted = radius_free = None
    refraction_true_h = refraction_ray_picks = straight_ray_refused = None
    if truth.air_gap_m == 0:
        shift_config = DepthConfig(ground_zero_shift=GroundZeroShift.SYNTHETIC_TABLE)
        shifted = estimate_depth(radargram, box, shift_config)
        no_radius = replace(radargram, pipe_radius_m=None)
        radius_free = estimate_depth(no_radius, box, inputs=inputs)
    elif main.zero_rule == "air_surface_late":
        refraction_true_h = _refraction_with_true_h(radargram, picks, truth)
        refraction_ray_picks = _refraction_on_ray_times(radargram, truth)
        straight_ray_refused = _straight_ray_refused(picks)
    return ModelRun(
        truth=truth,
        box=box,
        inputs=inputs,
        main=main,
        shifted=shifted,
        radius_free=radius_free,
        picks=picks,
        ideal_apex_ns=ideal_apex_ns,
        background_lateral_rel=lateral_rel,
        refraction_true_h=refraction_true_h,
        refraction_ray_picks=refraction_ray_picks,
        straight_ray_refused=straight_ray_refused,
        seconds=time.perf_counter() - started,
    )


def evaluate_all(root: Path = DEFAULT_SYNTHETIC_ROOT) -> list[ModelRun]:
    """Evaluate every model in ``manifest.csv`` (single process)."""
    runs = []
    for model_id in list_synthetic_models(root):
        run = evaluate_model(model_id, root)
        LOGGER.info("%s %s %.1f s", model_id, run.main.status, run.seconds)
        runs.append(run)
    return runs


# ----------------------------------------------------------------- checks


@dataclass(frozen=True)
class Check:
    """One regression check of one model."""

    check_id: str
    model_id: str
    measured: str
    criterion: str
    passed: bool
    grade: str = ""


def _depth_error_m(result: DepthResult | None, truth: SyntheticTruth) -> float | None:
    if result is None or result.depth_top_m is None:
        return None
    return result.depth_top_m - truth.pipe_top_depth_m


def _alt_meta_error_m(
    result: DepthResult | None, truth: SyntheticTruth
) -> float | None:
    if result is None or "meta_fixed" not in result.alt_depths:
        return None
    return result.alt_depths["meta_fixed"].depth_top_m - truth.pipe_top_depth_m


def _fmt(value: float | None, unit: str = "", scale: float = 1.0) -> str:
    return "None" if value is None else f"{value * scale:+.3f}{unit}"


def _zero_checks(run: ModelRun) -> list[Check]:
    """R1-R3: detected zero events vs the json zero candidates."""
    zero, truth = run.inputs.zero, run.truth
    checks = []
    first_event = zero.events_ns[0] if zero.events_ns else None
    if run.is_ground:
        reference = truth.direct_wave_peak_ns
        diff = None if None in (first_event, reference) else first_event - reference
        passed = diff is not None and abs(diff) <= 0.05
        checks.append(
            Check("R1", run.model_id, _fmt(diff, " ns"), "|dt| <= 0.05 ns", passed)
        )
    elif truth.air_gap_m == SINGLE_EVENT_GAP_M:
        reference = truth.direct_wave_peak_ns
        diff = (
            None if None in (zero.t_zero_ns, reference) else zero.t_zero_ns - reference
        )
        passed = (
            zero.zero_rule == "air_single_event"
            and diff is not None
            and abs(diff) <= 0.05
        )
        measured = f"{zero.zero_rule}, {_fmt(diff, ' ns')}"
        checks.append(
            Check(
                "R3",
                run.model_id,
                measured,
                "air_single_event, |dt| <= 0.05 ns",
                passed,
            )
        )
    elif truth.air_gap_m == TWO_EVENT_GAP_M:
        reference = truth.surface_reflection_peak_ns
        diff = (
            None if None in (zero.t_zero_ns, reference) else zero.t_zero_ns - reference
        )
        h_error = None if zero.h_est_m is None else zero.h_est_m - truth.air_gap_m
        passed = (
            len(zero.events_ns) == 2
            and diff is not None
            and abs(diff) <= 0.35
            and h_error is not None
            and abs(h_error) <= 0.07
        )
        measured = (
            f"{len(zero.events_ns)} events, t_surf {_fmt(diff, ' ns')}, "
            f"h_est {_fmt(h_error, ' m')}"
        )
        checks.append(
            Check(
                "R2",
                run.model_id,
                measured,
                "2 events, |dt| <= 0.35 ns, |dh| <= 0.07 m",
                passed,
            )
        )
    return checks


def _apex_checks(run: ModelRun) -> list[Check]:
    """R4: candidate trace vs json apex; R5 (v1.6): its time vs ideal ``D - bg``."""
    main, truth = run.main, run.truth
    trace_diff = (
        None if main.apex_trace is None else main.apex_trace - truth.apex_trace_index
    )
    time_diff = _r5_diff_ns(run)
    return [
        Check(
            "R4",
            run.model_id,
            "None" if trace_diff is None else f"{trace_diff:+d} traces",
            "|di| <= 3",
            trace_diff is not None and abs(trace_diff) <= 3,
        ),
        Check(
            "R5",
            run.model_id,
            _fmt(time_diff, " ns"),
            "|t_candidate - t_ideal(D-bg, (1)-(3))| <= 0.05 ns",
            time_diff is not None and abs(time_diff) <= 0.05,
        ),
    ]


def load_dt(run: ModelRun) -> float:
    """Sample interval of the model (from its picks' scale)."""
    assert run.picks.scale.dt_ns is not None
    return run.picks.scale.dt_ns


def _candidate_time_ns(run: ModelRun) -> float | None:
    """Apex candidate time ``t_a`` (first envelope peak, median background)."""
    if run.main.apex_sample is None:
        return None
    return run.main.apex_sample * load_dt(run)


def _r5_diff_ns(run: ModelRun) -> float | None:
    candidate_ns = _candidate_time_ns(run)
    if candidate_ns is None or run.ideal_apex_ns is None:
        return None
    return candidate_ns - run.ideal_apex_ns


def _filter_shift_ns(run: ModelRun) -> float | None:
    """Recorded only (v1.6): ideal apex after (1)-(3) minus json envelope peak."""
    if run.ideal_apex_ns is None:
        return None
    return run.ideal_apex_ns - run.truth.apex_envelope_peak_ns


def _grade(depth_error_m: float | None, v_rel_error: float | None) -> str:
    """Verification grade: match / acceptable match / mismatch."""
    if depth_error_m is None:
        return "no depth"
    v_ok = v_rel_error is None or abs(v_rel_error) <= MATCH_V_REL
    if abs(depth_error_m) <= MATCH_DEPTH_M and v_ok:
        return "match"
    if abs(depth_error_m) <= ACCEPT_DEPTH_M:
        return "acceptable match"
    return "mismatch"


def _v_rel_error(result: DepthResult | None, truth: SyntheticTruth) -> float | None:
    if result is None or result.v_m_per_ns is None:
        return None
    return result.v_m_per_ns / truth.velocity_m_per_ns - 1.0


def _ground_depth_checks(run: ModelRun) -> list[Check]:
    """R8-R11 for ground-coupled models."""
    truth, main = run.truth, run.main
    meta_error = _alt_meta_error_m(main, truth)
    shifted_error = _alt_meta_error_m(run.shifted, truth)
    curvature_error = _depth_error_m(main, truth)
    v_error = _v_rel_error(main, truth)
    radius_free_v = _v_rel_error(run.radius_free, truth)
    shifted_rule = None if run.shifted is None else run.shifted.zero_rule
    is_curvature = main.v_rule == "curvature_R_fixed"
    r10_passed = (
        is_curvature
        and v_error is not None
        and abs(v_error) <= 0.06
        and curvature_error is not None
        and abs(curvature_error) <= 0.05
    )
    return [
        Check(
            "R8",
            run.model_id,
            _fmt(meta_error, " cm", 100),
            "-5 <= dd < 0 cm (meta_fixed, ground_direct_peak)",
            meta_error is not None and -0.05 <= meta_error < 0,
        ),
        Check(
            "R9",
            run.model_id,
            f"{shifted_rule}, {_fmt(shifted_error, ' cm', 100)}",
            "|dd| <= 2 cm (meta_fixed, ground_direct_shifted)",
            shifted_rule == "ground_direct_shifted"
            and shifted_error is not None
            and abs(shifted_error) <= 0.02,
        ),
        Check(
            "R10",
            run.model_id,
            f"{main.v_rule}, v {_fmt(v_error, ' %', 100)}, "
            f"d {_fmt(curvature_error, ' cm', 100)}",
            "|dv| <= 6 %, |dd| <= 5 cm (curvature_R_fixed)",
            r10_passed,
            _grade(curvature_error, v_error) if is_curvature else "",
        ),
        Check(
            "R11",
            run.model_id,
            f"{None if run.radius_free is None else run.radius_free.v_rule}, "
            f"v'/v - 1 {_fmt(radius_free_v, ' %', 100)}",
            "v'/v > 1 (R = 0 curvature)",
            radius_free_v is not None and radius_free_v > 0,
        ),
    ]


def _air_depth_checks(run: ModelRun) -> list[Check]:
    """R6, R7, R12, R13 for air-coupled models."""
    truth, main = run.truth, run.main
    error = _depth_error_m(main, truth)
    is_meta = main.v_rule == "meta_fixed"
    if truth.air_gap_m == SINGLE_EVENT_GAP_M:
        return [
            Check(
                "R7",
                run.model_id,
                f"{main.v_rule}, {_fmt(error, ' cm', 100)}",
                "0 < dd <= 5 cm (meta_fixed)",
                is_meta and error is not None and 0 < error <= 0.05,
                _grade(error, None),
            )
        ]
    checks = [
        Check(
            "R6",
            run.model_id,
            f"{main.v_rule}, {_fmt(error, ' cm', 100)}",
            "|dd| <= 2 cm (meta_fixed)",
            is_meta and error is not None and abs(error) <= 0.02,
            _grade(error, None),
        ),
        Check(
            "R12",
            run.model_id,
            f"ValueError raised: {run.straight_ray_refused}",
            "straight-ray curvature request raises ValueError",
            bool(run.straight_ray_refused),
        ),
    ]
    checks.extend(_refraction_checks(run))
    return checks


def _qc_text(qc: RefractionQC | None, truth: SyntheticTruth) -> str:
    if qc is None:
        return "None"
    eps_rel = qc.eps_fit / truth.eps_r - 1.0
    depth_error = qc.d_top_m - truth.pipe_top_depth_m
    return (
        f"eps {qc.eps_fit:.3f} ({_fmt(eps_rel, ' %', 100)}), "
        f"d {_fmt(depth_error, ' cm', 100)}"
    )


def _qc_within(
    qc: RefractionQC | None, truth: SyntheticTruth, eps_abs: float, depth_m: float
) -> bool:
    return (
        qc is not None
        and abs(qc.eps_fit - truth.eps_r) <= eps_abs
        and abs(qc.d_top_m - truth.pipe_top_depth_m) <= depth_m
    )


def _refraction_checks(run: ModelRun) -> list[Check]:
    """R13a-1 (algorithm, ray picks), R13a-2 (data, measured picks), R13b."""
    truth, main = run.truth, run.main
    ray_qc, data_qc = run.refraction_ray_picks, run.refraction_true_h
    eps_range = main.eps_r_refraction_qc_range
    range_passed = (
        eps_range is not None
        and eps_range[0] <= truth.eps_r <= eps_range[1]
        and "meta_eps_suspect" not in main.flags
    )
    range_text = (
        "None" if eps_range is None else f"[{eps_range[0]:.2f}, {eps_range[1]:.2f}]"
    )
    return [
        Check(
            "R13a-1",
            run.model_id,
            _qc_text(ray_qc, truth),
            "json ray picks, json h/s/R: eps +- 0.1, |dd| <= 1 cm",
            _qc_within(ray_qc, truth, RAY_QC_EPS_TOLERANCE, RAY_QC_DEPTH_TOLERANCE_M),
        ),
        Check(
            "R13a-2",
            run.model_id,
            _qc_text(data_qc, truth) + " (wing delay bias = eps rel. error)",
            "measured picks, json h/s/R: |eps/eps_true - 1| <= 20 %, |dd| <= 5 cm",
            _qc_within(
                data_qc,
                truth,
                DATA_QC_EPS_REL_TOLERANCE * truth.eps_r,
                DATA_QC_DEPTH_TOLERANCE_M,
            ),
        ),
        Check(
            "R13b",
            run.model_id,
            f"range {range_text}, suspect {'meta_eps_suspect' in main.flags}",
            "h_est path: range contains 9, no meta_eps_suspect",
            range_passed,
        ),
    ]


def regression_checks(runs: Sequence[ModelRun]) -> list[Check]:
    """R1-R13 over all models, in model order."""
    checks: list[Check] = []
    for run in runs:
        checks.extend(_zero_checks(run))
        checks.extend(_apex_checks(run))
        if run.is_ground:
            checks.extend(_ground_depth_checks(run))
        else:
            checks.extend(_air_depth_checks(run))
    return checks


# ----------------------------------------------------------------- tables


def _round(value: float | None, digits: int = 4) -> float | str:
    return "" if value is None else round(float(value), digits)


def model_row(run: ModelRun) -> dict[str, object]:
    """One csv row per model (lengths in m unless the name says cm)."""
    truth, main = run.truth, run.main
    error = _depth_error_m(main, truth)
    v_error = _v_rel_error(main, truth)
    x0_json = None if main.x0_m is None else main.x0_m + truth.trace_x_m_first
    qc_range = main.eps_r_refraction_qc_range
    return {
        "model_id": run.model_id,
        "pipe_material": truth.pipe_material or "",
        "air_gap_m": truth.air_gap_m,
        "eps_r": truth.eps_r,
        "pipe_radius_m": truth.pipe_radius_m,
        "d_true_m": truth.pipe_top_depth_m,
        "d_est_m": _round(main.depth_top_m),
        "d_err_cm": _round(None if error is None else error * 100, 2),
        "grade": _grade(error, v_error if main.v_rule == "curvature_R_fixed" else None),
        "status": main.status.value,
        "v_rule": main.v_rule or "",
        "v_err_pct": _round(None if v_error is None else v_error * 100, 2),
        "sigma_d_cm": _round(
            None if main.sigma_depth_m is None else main.sigma_depth_m * 100, 2
        ),
        "zero_rule": main.zero_rule or "",
        "t_zero_ns": _round(main.t_zero_ns),
        "h_est_m": _round(main.h_est_m),
        "meta_alt_err_cm": _round(_scaled(_alt_meta_error_m(main, truth)), 2),
        "shifted_meta_err_cm": _round(
            _scaled(_alt_meta_error_m(run.shifted, truth)), 2
        ),
        "radius_free_v_err_pct": _round(
            _scaled(_v_rel_error(run.radius_free, truth)), 2
        ),
        "apex_trace_true": truth.apex_trace_index,
        "apex_trace_est": "" if main.apex_trace is None else main.apex_trace,
        "t_candidate_ns": _round(_candidate_time_ns(run)),
        "t_ideal_apex_ns": _round(run.ideal_apex_ns),
        "r5_diff_ns": _round(_r5_diff_ns(run)),
        "filter_shift_ns": _round(_filter_shift_ns(run)),
        "bg_lateral_rel": f"{run.background_lateral_rel:.2e}",
        "x0_err_cm": _round(
            None if x0_json is None else (x0_json - truth.pipe_center_x_m) * 100, 2
        ),
        "n_picks": main.n_picks,
        "eps_qc": _round(main.eps_r_refraction_qc, 3),
        "eps_qc_lo": "" if qc_range is None else round(qc_range[0], 3),
        "eps_qc_hi": "" if qc_range is None else round(qc_range[1], 3),
        "box": f"{run.box.i0}-{run.box.i1}/{run.box.k0}-{run.box.k1}",
        "searched_k0": run.picks.box.k0,
        "flags": " ".join(main.flags),
        "seconds": round(run.seconds, 2),
    }


def _scaled(value: float | None) -> float | None:
    return None if value is None else value * 100


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_row(check: Check) -> dict[str, object]:
    return {
        "check": check.check_id,
        "model_id": check.model_id,
        "measured": check.measured,
        "criterion": check.criterion,
        "passed": check.passed,
        "grade": check.grade,
    }


# ----------------------------------------------------------------- figures


SERIES_COLORS = ("#2a78d6", "#eb6834")
"""Categorical slots 1-2 of the validated default palette (dataviz skill)."""


def plot_depth_errors(runs: Sequence[ModelRun], path: Path) -> None:
    """Main-result and meta_fixed-alternative depth error per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [run.model_id for run in runs]
    positions = np.arange(len(runs))
    main_cm = [_scaled(_depth_error_m(run.main, run.truth)) for run in runs]
    meta_cm = [_scaled(_alt_meta_error_m(run.main, run.truth)) for run in runs]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for limit, style in ((2, "--"), (5, ":")):
        for sign in (-1, 1):
            ax.axhline(sign * limit, color="#9a9a9a", lw=0.8, ls=style)
    ax.axhline(0, color="#52514e", lw=0.8)
    series = (
        ("main result (curvature or meta_fixed)", main_cm, -0.15),
        ("meta_fixed alternative (ground)", meta_cm, 0.15),
    )
    for (name, values, shift), color in zip(series, SERIES_COLORS, strict=True):
        shown = [
            (p + shift, v)
            for p, v in zip(positions, values, strict=True)
            if v is not None
        ]
        if shown:
            xs, ys = zip(*shown, strict=True)
            ax.plot(xs, ys, "o", ms=6, color=color, label=name)
    ax.set_xticks(positions, labels, fontsize=8)
    ax.set_ylabel("depth error d - d_top (cm)")
    ax.set_title("Synthetic depth error per model (lines: +-2 cm, +-5 cm)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_velocity_ratios(runs: Sequence[ModelRun], path: Path) -> None:
    """R10 (R = json) and R11 (R = 0) curvature v / v_true for ground models."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ground = [run for run in runs if run.is_ground]
    positions = np.arange(len(ground))
    fitted = [_scaled(_v_rel_error(run.main, run.truth)) for run in ground]
    radius_free = [_scaled(_v_rel_error(run.radius_free, run.truth)) for run in ground]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for limit in (-6, 6):
        ax.axhline(limit, color="#9a9a9a", lw=0.8, ls="--")
    ax.axhline(0, color="#52514e", lw=0.8)
    series = (("R = json (R10)", fitted, -0.15), ("R = 0 (R11)", radius_free, 0.15))
    for (name, values, shift), color in zip(series, SERIES_COLORS, strict=True):
        shown = [
            (p + shift, v)
            for p, v in zip(positions, values, strict=True)
            if v is not None
        ]
        if shown:
            xs, ys = zip(*shown, strict=True)
            ax.plot(xs, ys, "o", ms=6, color=color, label=name)
    ax.set_xticks(positions, [run.model_id for run in ground], fontsize=8)
    ax.set_ylabel("v / v_true - 1 (%)")
    ax.set_title(
        "Curvature velocity error, ground-coupled models (lines: +-6 %)", fontsize=10
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------- markdown


def _markdown_table(
    rows: Sequence[dict[str, object]], columns: Sequence[str]
) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        # Criteria such as "|dt| <= 0.05 ns" contain pipes; escape them for markdown.
        cells = [str(row[column]).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _check_summary(checks: Sequence[Check]) -> list[dict[str, object]]:
    summary = []
    for check_id in dict.fromkeys(check.check_id for check in checks):
        group = [check for check in checks if check.check_id == check_id]
        failed = [check.model_id for check in group if not check.passed]
        summary.append(
            {
                "check": check_id,
                "criterion": group[0].criterion,
                "passed": f"{len(group) - len(failed)}/{len(group)}",
                "verdict": "PASS" if not failed else "FAIL",
                "failed models": ", ".join(failed) or "-",
            }
        )
    return summary


def _assumption_lines(runs: Sequence[ModelRun]) -> list[str]:
    """Assumptions measured here for the first time (spec 11-2, 7 2a, box rule)."""
    lines = ["## 실측으로 처음 확인한 가정", ""]
    two_event = [run for run in runs if run.main.h_est_m is not None]
    for run in two_event:
        h_error = run.main.h_est_m - run.truth.air_gap_m
        lines.append(
            f"- h_est - h ({run.model_id}): {h_error:+.3f} m "
            f"(Delta h 0.06 m covers it: {0 <= h_error <= 0.06})"
        )
    for run in runs:
        if run.refraction_true_h is not None:
            qc_range = run.main.eps_r_refraction_qc_range
            lines.append(
                f"- {run.model_id} eps_r interval (h_est-0.06, h_est): {qc_range}"
            )
            lines.append(
                f"- {run.model_id} R13a-1 (json ray picks): "
                f"{_qc_text(run.refraction_ray_picks, run.truth)}"
            )
            lines.append(
                f"- {run.model_id} R13a-2 (measured picks, json h): "
                f"{_qc_text(run.refraction_true_h, run.truth)}; the eps relative "
                "error is the wing delay bias of the gprMax envelope picks"
            )
            error = _depth_error_m(run.main, run.truth)
            lines.append(
                f"- {run.model_id} R6 depth error {_fmt(error, ' cm', 100)} "
                "(spec v1.6 reference -1.49 cm)"
            )
    lateral = max(run.background_lateral_rel for run in runs)
    lines.append(
        f"- no-pipe background lateral spread (max over models): {lateral:.2e} "
        "of its peak; one middle column is used for every trace (R5 reference)"
    )
    shifts = [s for s in (_filter_shift_ns(run) for run in runs) if s is not None]
    lines.append(
        f"- filter shift (ideal (1)-(3) apex - json envelope peak): "
        f"{min(shifts):+.3f} to {max(shifts):+.3f} ns (recorded only)"
    )
    edge = [run.model_id for run in runs if "apex_at_box_edge" in run.main.flags]
    cut = [run.model_id for run in runs if run.picks.box.k0 > run.box.k0]
    lines.append(f"- box 0.8 m: apex_at_box_edge in {edge or 'no model'}")
    lines.append(f"- box 1.5 periods: top cut by the mute in {cut or 'no model'}")
    picks = [run.main.n_picks for run in runs]
    lines.append(f"- picks per model: min {min(picks)}, max {max(picks)}")
    return lines


def write_markdown(
    path: Path,
    runs: Sequence[ModelRun],
    checks: Sequence[Check],
    model_rows: Sequence[dict[str, object]],
    figure_names: Sequence[str],
    elapsed_s: float,
) -> None:
    from gpr_detection.run_info import git_state, local_timestamp

    git = git_state(PROJECT_ROOT)
    fastest_s = min(run.seconds for run in runs)
    slowest_s = max(run.seconds for run in runs)
    columns = (
        "model_id",
        "pipe_material",
        "air_gap_m",
        "eps_r",
        "d_true_m",
        "d_est_m",
        "d_err_cm",
        "grade",
        "status",
        "v_rule",
        "v_err_pct",
        "zero_rule",
        "flags",
    )
    lines = [
        "# gprMax 합성 깊이 회귀 v1 (M2-5)",
        "",
        f"- 생성: `scripts/eval_depth_synthetic.py`, {local_timestamp()}, "
        f"commit {git['commit']} (dirty {git['dirty']})",
        "- 명세: `docs/processing/spec_depth_v1.md` §11-2 (R1-R13, 박스 공급 규칙)",
        f"- 실행 시간: 전체 {elapsed_s:.1f} s, "
        f"모델별 {fastest_s:.1f}-{slowest_s:.1f} s (단일 프로세스)",
        "- 판정 등급: match ≤5 %·2 cm, acceptable match d ≤10 cm (공기결합은 d만)",
        "",
        "## R1-R13 판정",
        "",
        *_markdown_table(
            _check_summary(checks),
            ("check", "criterion", "passed", "verdict", "failed models"),
        ),
        "",
        "## 모델별 결과",
        "",
        *_markdown_table(model_rows, columns),
        "",
        "## 검사별 측정값",
        "",
        *_markdown_table(
            [check_row(check) for check in checks],
            ("check", "model_id", "measured", "criterion", "passed", "grade"),
        ),
        "",
        *_assumption_lines(runs),
        "",
        "## 그림",
        "",
        *[f"![{name}]({name})" for name in figure_names],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------- entry point


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_SYNTHETIC_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate all models, write csv/md/png; exit 1 when any check failed."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    started = time.perf_counter()
    runs = evaluate_all(args.root)
    checks = regression_checks(runs)
    elapsed_s = time.perf_counter() - started

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_rows = [model_row(run) for run in runs]
    write_csv(args.out_dir / f"{args.stem}.csv", model_rows)
    write_csv(args.out_dir / f"{args.stem}_checks.csv", [check_row(c) for c in checks])
    figure_names = [f"{args.stem}_depth_error.png", f"{args.stem}_velocity.png"]
    plot_depth_errors(runs, args.out_dir / figure_names[0])
    plot_velocity_ratios(runs, args.out_dir / figure_names[1])
    write_markdown(
        args.out_dir / f"{args.stem}.md",
        runs,
        checks,
        model_rows,
        figure_names,
        elapsed_s,
    )
    failed = [check for check in checks if not check.passed]
    LOGGER.info(
        "checks: %d passed, %d failed, %.1f s",
        len(checks) - len(failed),
        len(failed),
        elapsed_s,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
