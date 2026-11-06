"""M2-9 C step 2: prototype of the apex-candidate QC (rule B + D) for spec_eval_4tu_v1 v1.4.

Synthetic inputs only (no 4TU radargram, no ground truth, no M2-8 result).

Parts
-----
E  record-end zone: how far the record end disturbs the pick envelope of the
   fitting branch (DC, Gaussian dewow, zero-phase band-pass, median background,
   Hilbert envelope). A 1024-sample record is processed whole and truncated to
   512 samples; the difference at the same samples is the truncation artefact.
S  trace-seam statistic: adjacent-trace difference ratio on row-normalised
   ``branch.filtered``; null distribution on continuous synthetic lines and
   response to injected seams (gain step, time shift, block background, dead
   trace).
C  expected values of the reproduction tests C1-C14 (curvature QC through the
   real depth path: candidate_box -> estimate_depth(apex_hint) -> extract_picks ->
   fit_curvature(R = 0, air_gap = "ground", QC only) -> classification).

Outputs (docs/processing): m29c_qc_edge.csv, m29c_qc_seam_null.csv,
m29c_qc_seam_power.csv, m29c_qc_ctests.csv, m29c_qc_constants.json.

Seeds: numpy default_rng(0) for part E noise, default_rng(1) for part S lines,
tests/synthetic_bscan.py noise seed 0 for part C.

Run: conda run -n gpr-dev python -I analysis/processing/m29c_qc_proto.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "tests"))

from eval_depth_4tu import BOX_HALF_WIDTH_M, candidate_box  # noqa: E402
from synthetic_bscan import T_SRC_NS, BScanSpec, cylinder_times, make_radargram  # noqa: E402

from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import estimate_depth, prepare_depth_inputs  # noqa: E402
from gpr_detection.depth.hyperbola import (  # noqa: E402
    V_MAX_M_PER_NS,
    V_MIN_M_PER_NS,
    fit_curvature,
)
from gpr_detection.depth.picks import ApexHint, extract_picks, mute_sample  # noqa: E402
from gpr_detection.depth.refraction import reflection_times  # noqa: E402
from gpr_detection.io.radargram import Radargram, velocity_from_eps  # noqa: E402
from gpr_detection.preprocess.filters import prepare_fitting_branch  # noqa: E402
from gpr_detection.preprocess.signals import pick_signal  # noqa: E402

DOCS = PROJECT / "docs" / "processing"
F_C_HZ = 500e6
PERIOD_NS = 1e9 / F_C_HZ
DX_M = 0.02
FOURTU_DTS = (0.098, 0.117, 0.195)
N_RECORD = 512

# ---- QC constants (spec_eval_4tu_v1 v1.4 section 3-5) --------------------------------
QC_R_LO = 0.7
QC_R_HI = 1.6
QC_R_TOL = 1e-12
QC_UNSTABLE_SIGMA_REL = 0.15
QC_UNSTABLE_INLIER = 0.5
QC_BOUND_REL = 1e-4
QC_RADIUS_M = 0.0
MAIN_LOBE_PERIODS = 0.6
EDGE_REL_ERROR = 0.10
SEAM_APEX_HALF_M = 0.10

# Filled from parts E and S (rounded up as stated in the spec), then locked.
RECORD_END_PERIODS: float | None = None
SEAM_RATIO: float | None = None
SEAM_LAG_NS = 0.25
SEAM_RATIO_FIXED = 3.0  # provisional; final value set from the null maximum (spec 3-5-4)


# =============================================================================== part E
def _radargram(data: np.ndarray, dt_ns: float, air: bool = True) -> Radargram:
    return Radargram(
        data=data.astype(np.float32),
        dt_ns=dt_ns,
        dx_m=DX_M,
        air_gap="air" if air else "ground",
        amplitude_processed=False,
        dataset="unit",
        line_id="unit/line",
        f_c_hz=F_C_HZ,
        eps_r_meta=9.0,
        eps_r_meta_source="synthetic_truth",
        antenna_offset_m=0.1,
    )


def _envelope(data: np.ndarray, dt_ns: float) -> np.ndarray:
    branch = prepare_fitting_branch(_radargram(data, dt_ns), DepthConfig().bg_window)
    return pick_signal(branch.background_removed, "envelope")


def _ricker_colour(noise: np.ndarray, dt_ns: float) -> np.ndarray:
    t = (np.arange(-64, 65) * dt_ns)[:, None]
    arg = (np.pi * F_C_HZ * 1e-9 * t) ** 2
    kernel = ((1 - 2 * arg) * np.exp(-arg))[:, 0]
    from scipy.signal import fftconvolve

    return fftconvolve(noise, kernel[:, None], mode="same", axes=0)


def edge_zone() -> pd.DataFrame:
    """Envelope RMS elevation near the record end for stationary inputs.

    ``ratio(k) = RMS_i E(k, i) / median_k' RMS_i E(k', i)`` over the middle half of
    the record. The edge zone is the run from the end where ``|ratio - 1| > 0.1``.
    ``trunc_*`` columns: error of the 512-sample envelope against the same data
    processed as a 1024-sample record (information only).
    """
    rng = np.random.default_rng(0)
    rows = []
    n_long, n_traces = 2 * N_RECORD, 1200
    for dt in FOURTU_DTS:
        white = rng.normal(size=(n_long, n_traces))
        cases = {"white": white, "ricker_coloured": _ricker_colour(white, dt)}
        for name, full in cases.items():
            env_cut = _envelope(full[:N_RECORD], dt)
            rms = np.sqrt(np.mean(env_cut**2, axis=1))
            ref = float(np.median(rms[N_RECORD // 4 : 3 * N_RECORD // 4]))
            ratio = rms / ref
            dev = np.abs(ratio - 1.0) > EDGE_REL_ERROR
            n_edge = 0
            for k in range(N_RECORD - 1, N_RECORD // 2, -1):
                if dev[k]:
                    n_edge = N_RECORD - k
            n_start = 0
            for k in range(0, N_RECORD // 2):
                if dev[k]:
                    n_start = k + 1
            env_full = _envelope(full, dt)[:N_RECORD]
            trunc = np.sqrt(np.mean((env_cut - env_full) ** 2, axis=1)) / np.sqrt(
                np.mean(env_full**2, axis=1)
            )
            rows.append(
                dict(
                    dt_ns=dt,
                    case=name,
                    edge_samples=n_edge,
                    edge_ns=n_edge * dt,
                    edge_periods=n_edge * dt / PERIOD_NS,
                    start_samples=n_start,
                    ratio_last=float(ratio[-1]),
                    ratio_at_0p5=float(ratio[N_RECORD - 1 - int(round(0.5 * PERIOD_NS / dt))]),
                    ratio_at_1p=float(ratio[N_RECORD - 1 - int(round(PERIOD_NS / dt))]),
                    ratio_max_end_zone=float(ratio[N_RECORD // 2 :].max()),
                    ratio_min_end_zone=float(ratio[N_RECORD // 2 :].min()),
                    trunc_at_1p=float(trunc[N_RECORD - 1 - int(round(PERIOD_NS / dt))]),
                    trunc_at_2p=float(trunc[N_RECORD - 1 - int(round(2 * PERIOD_NS / dt))]),
                )
            )
    return pd.DataFrame(rows)


# =============================================================================== part S
SEAM_TIME_PERIODS = 0.1   # theta_t = 0.1 / f_c (0.2 ns at 500 MHz)
SEAM_AMP_RATIO = 1.25     # theta_A = ln(1.25)
SEAM_NOISE_K = 5.0        # same k as the apex-hint noise floor (spec_depth 5-7)
SEAM_ZERO_HALF_PERIODS = 1.0


def zero_band_track(filtered: np.ndarray, t_zero_ns: float, dt_ns: float):
    """Per-trace zero-band lag (ns) and amplitude against the line's median waveform.

    Window t_z +- 1 period of the fitting-branch ``filtered`` section (signed). The
    template is the median trace over the line in that window. Each trace is matched
    by zero-mean-free normalised cross-correlation for lags within +-0.5 period
    (parabolic refinement); the amplitude is the least-squares gain at that lag.
    """
    half = SEAM_ZERO_HALF_PERIODS * PERIOD_NS
    max_lag = int(math.ceil(0.5 * PERIOD_NS / dt_ns))
    k0 = int(math.floor((t_zero_ns - half) / dt_ns))
    k1 = int(math.ceil((t_zero_ns + half) / dt_ns))
    k0 = max(max_lag, k0)
    k1 = min(filtered.shape[0] - 1 - max_lag, k1)
    data = np.asarray(filtered, dtype=float)
    tmpl = np.median(data[k0 : k1 + 1], axis=1)
    tn = float(tmpl @ tmpl)
    n_tr = data.shape[1]
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.empty((len(lags), n_tr))
    for i, lag in enumerate(lags):
        seg = data[k0 + lag : k1 + 1 + lag]
        cc[i] = tmpl @ seg
    best = np.argmax(cc, axis=0)
    frac = np.zeros(n_tr)
    inner = (best > 0) & (best < len(lags) - 1)
    cols = np.nonzero(inner)[0]
    y0, y1, y2 = cc[best[cols] - 1, cols], cc[best[cols], cols], cc[best[cols] + 1, cols]
    den = y0 - 2 * y1 + y2
    ok = den < 0
    frac[cols[ok]] = 0.5 * (y0[ok] - y2[ok]) / den[ok]
    lag_ns = (lags[best] + frac) * dt_ns
    amp = cc[best, np.arange(n_tr)] / tn if tn > 0 else np.zeros(n_tr)
    return lag_ns, np.abs(amp)

def _robust_sigma(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


def seam_ratios(filtered: np.ndarray, t_zero_ns: float, dt_ns: float, ref_half: int):
    """Zero-band jumps per split j (index j-1): time jump (ns) and log-amplitude jump.

    Returns (dt_jump, la_jump, dt_limit, la_limit); a split is a seam when
    dt_jump >= dt_limit or la_jump >= la_limit, where each limit is
    max(physical threshold, k * robust sigma of the signed jumps over +-K splits).
    """
    t, a = zero_band_track(filtered, t_zero_ns, dt_ns)
    return seam_from_track(t, a, ref_half)


def seam_from_track(t: np.ndarray, a: np.ndarray, ref_half: int):
    """Pure part of :func:`seam_ratios` (spec 3-5-4 items 4-6)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        la = np.log(a)
    dts = np.diff(t)
    das = np.diff(la)
    das[~np.isfinite(das)] = np.inf
    n = len(dts)
    dt_lim = np.full(n, np.nan)
    la_lim = np.full(n, np.nan)
    theta_t = SEAM_TIME_PERIODS * PERIOD_NS
    theta_a = math.log(SEAM_AMP_RATIO)
    for j in range(n):
        lo, hi = max(0, j - ref_half), min(n, j + ref_half + 1)
        ref_t = np.concatenate([dts[lo:j], dts[j + 1 : hi]])
        ref_a = np.concatenate([das[lo:j], das[j + 1 : hi]])
        dt_lim[j] = max(theta_t, SEAM_NOISE_K * _robust_sigma(ref_t))
        la_lim[j] = max(theta_a, SEAM_NOISE_K * _robust_sigma(ref_a))
    return np.abs(dts), np.abs(das), dt_lim, la_lim


def seam_score(filtered, t_zero_ns, dt_ns, ref_half):
    """max(dt_jump / dt_limit, la_jump / la_limit) per split; >= 1 means seam."""
    dj, aj, dl, al = seam_ratios(filtered, t_zero_ns, dt_ns, ref_half)
    with np.errstate(invalid="ignore"):
        return np.maximum(dj / dl, aj / al), dj, aj

def record_end_samples(dt_ns: float) -> int:
    assert RECORD_END_PERIODS is not None
    return math.ceil(RECORD_END_PERIODS * PERIOD_NS / dt_ns - 1e-9)


def seam_rows(radargram: Radargram, inputs) -> tuple[int, int]:
    k_lo = mute_sample(inputs.zero.t_zero_ns, PERIOD_NS, radargram.dt_ns)
    k_hi = radargram.n_samples - record_end_samples(radargram.dt_ns)
    return k_lo, k_hi


def ref_half_traces(dx_m: float) -> int:
    return math.ceil(BOX_HALF_WIDTH_M / dx_m - 1e-9)


def apex_half_traces(dx_m: float) -> int:
    return math.ceil(SEAM_APEX_HALF_M / dx_m - 1e-9)


def seam_near(ratios: np.ndarray, trace: int, half: int) -> tuple[float, int | None]:
    """Max Q over splits j (between j-1 and j) with j-1 >= trace-half and j <= trace+half."""
    best, best_j = -np.inf, None
    for j in range(max(1, trace - half + 1), trace + half + 1):
        if j - 1 >= len(ratios):
            break
        value = ratios[j - 1]
        if not np.isnan(value) and value > best:
            best, best_j = float(value), j
    return best, best_j


ZERO_BAND_SCALE = 10.0  # direct 30 / surface -20: zero band 20 dB above the pipe (assumption)


def _air_spec(n_traces=401, n_samples=N_RECORD, dt=0.098, eps=9.0, snr=20.0):
    return BScanSpec(
        direct_amplitude=3.0 * ZERO_BAND_SCALE,
        surface_amplitude=-2.0 * ZERO_BAND_SCALE,
        f_c_hz=F_C_HZ,
        eps_r=eps,
        d_top_m=0.6,
        air_gap_m=0.001,
        offset_m=0.0,
        n_traces=n_traces,
        n_samples=n_samples,
        dt_ns=dt,
        snr_db=snr,
    )


def _ringing(n_samples: int, dt: float, amp: float, f_ghz: float, decay_ns: float):
    t = np.arange(n_samples) * dt
    return amp * np.sin(2 * np.pi * f_ghz * (t - T_SRC_NS)) * np.exp(-np.clip(t - T_SRC_NS, 0, None) / decay_ns) * (t > T_SRC_NS)


def _continuous_line(rng, dt, eps, snr) -> BScanSpec:
    """Shallow steep pipe, deeper pipe, finite flat layer, dipping layer, noise."""
    spec = _air_spec(dt=dt, eps=eps, snr=snr)
    x = spec.x_m
    v = spec.velocity
    d1, d2 = rng.uniform(0.1, 0.4), rng.uniform(0.6, 1.2)
    x1, x2 = rng.uniform(1.5, 3.0), rng.uniform(5.0, 6.5)
    spec.d_top_m = d1
    spec.pipe_times_ns = cylinder_times(x, x1, d1, v, 0.0)
    spec.extra_events.append((cylinder_times(x, x2, d2, v, 0.05), rng.uniform(0.5, 2.0)))
    layer_t = rng.uniform(6.0, 12.0)
    inside = (np.abs(x - rng.uniform(2.0, 6.0)) <= rng.uniform(0.5, 1.5)).astype(float)
    spec.extra_events.append((np.full(len(x), layer_t), rng.uniform(0.5, 2.0) * inside))
    slope = rng.uniform(-8.0, 8.0)  # ns/m
    spec.extra_events.append((14.0 + slope * (x - x.mean()), 0.7))
    return spec


def _with_seam(data: np.ndarray, split: int, kind: str, dt: float) -> np.ndarray:
    out = np.array(data, dtype=np.float64)
    if kind.startswith("gain"):
        out[:, split:] *= float(kind.split("_")[1])
    elif kind.startswith("shift"):
        n = int(kind.split("_")[1])
        out[:, split:] = np.roll(out[:, split:], n, axis=0)
    elif kind.startswith("blockshift"):
        n = int(kind.split("_")[1])
        out[:, split : split + 20] = np.roll(out[:, split : split + 20], n, axis=0)
    elif kind == "block_background":
        out[:, split:] += _ringing(out.shape[0], dt, 0.5, 0.35, 25.0)[:, None]
    elif kind == "dead_trace":
        out[:, split] = 0.0
    return out.astype(np.float32)


def _line_inputs(spec, eps, ring):
    rg = make_radargram(spec, air_gap="air", eps_r_meta=eps)
    if ring is not None:
        rg = replace(rg, data=(rg.data + ring[:, None]).astype(np.float32))
    return rg, prepare_depth_inputs(rg)


def seam_study() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(1)
    null_rows, power_rows = [], []
    kinds = [
        "gain_1.25", "gain_1.5", "gain_2", "shift_1", "shift_2", "shift_3", "shift_5",
        "blockshift_5", "block_background", "dead_trace",
    ]
    half = ref_half_traces(DX_M)
    for dt in (0.098, 0.195):
        for eps in (9.0, 16.0):
            for snr in (20.0, 6.0):
                for ring_amp in (0.0, 1.0):
                    for rep in range(4):
                        spec = _continuous_line(rng, dt, eps, snr)
                        ring = None if ring_amp == 0 else _ringing(
                            spec.n_samples, dt, ring_amp, rng.uniform(0.3, 0.6), rng.uniform(5, 20))
                        rg, inputs = _line_inputs(spec, eps, ring)
                        score, dj, aj = seam_score(inputs.branch.filtered, inputs.zero.t_zero_ns, dt, half)
                        null_rows.append(
                            dict(dt_ns=dt, eps_r=eps, snr_db=snr, ring=ring_amp, rep=rep,
                                 score_max=float(np.nanmax(score)), n_seams=int(np.sum(score >= 1)),
                                 dt_jump_max=float(dj.max()), la_jump_max=float(aj.max()))
                        )
                        if rep > 0:
                            continue
                        split = int(rng.integers(120, 260))
                        for kind in kinds:
                            data = _with_seam(rg.data, split, kind, dt)
                            rg2 = replace(rg, data=data)
                            inputs2 = prepare_depth_inputs(rg2)
                            s2, dj2, aj2 = seam_score(inputs2.branch.filtered, inputs2.zero.t_zero_ns, dt, half)
                            at, _ = seam_near(s2, split, 1)
                            mask = np.ones(len(s2), bool)
                            mask[max(0, split - 3) : split + 23] = False
                            power_rows.append(
                                dict(dt_ns=dt, eps_r=eps, snr_db=snr, ring=ring_amp, kind=kind,
                                     split=split, score_at_seam=at,
                                     detected=bool(at >= 1),
                                     false_elsewhere=int(np.sum(s2[mask] >= 1)))
                            )
    return pd.DataFrame(null_rows), pd.DataFrame(power_rows)

# =============================================================================== part C
class Verdict:
    IN, OUT, UNKNOWN = "in", "out", "unknown"


def classify_ratio(r: float) -> str:
    if not math.isfinite(r):
        raise ValueError("r must be finite")
    return Verdict.IN if QC_R_LO - QC_R_TOL <= r <= QC_R_HI + QC_R_TOL else Verdict.OUT


@dataclass
class QcOut:
    verdict: str
    reason: str
    r: float = math.nan
    v_curv: float = math.nan
    sigma_v: float = math.nan
    inlier_frac: float = math.nan
    n_picks: int = 0
    n_left: int = 0
    n_right: int = 0
    aperture_m: float = math.nan


def classify_curvature(curv, picks, v_meta: float) -> QcOut:
    base = dict(n_picks=picks.n_picks, n_left=picks.n_left, n_right=picks.n_right)
    if picks.apex is None:
        return QcOut(Verdict.UNKNOWN, "no_apex", **base)
    if curv.status == "insufficient_picks":
        return QcOut(Verdict.UNKNOWN, "insufficient_picks", **base)
    fit = curv.fit
    if fit is None or not fit.converged:
        return QcOut(Verdict.UNKNOWN, "fit_failed", **base)
    x, tau = picks.x_m, picks.tau_ns
    vals = dict(r=fit.v_m_per_ns / v_meta, v_curv=fit.v_m_per_ns, sigma_v=fit.sigma_v,
                inlier_frac=fit.inlier_frac,
                aperture_m=float(np.max(np.abs(x - picks.apex_x_m))), **base)

    def at(value, lo, hi):
        return min(value - lo, hi - value) <= QC_BOUND_REL * (hi - lo)

    xb = at(fit.x0_m, float(x.min()), float(x.max()))
    tb = at(fit.t0_ns, 0.0, float(tau.max()))
    vhi = V_MAX_M_PER_NS - fit.v_m_per_ns <= QC_BOUND_REL * (V_MAX_M_PER_NS - V_MIN_M_PER_NS)
    vlo = fit.v_m_per_ns - V_MIN_M_PER_NS <= QC_BOUND_REL * (V_MAX_M_PER_NS - V_MIN_M_PER_NS)
    if xb or tb:
        return QcOut(Verdict.UNKNOWN, "fit_failed", **vals)
    if vhi:
        return QcOut(Verdict.OUT, "v_at_upper_bound", **vals)
    if vlo:
        return QcOut(Verdict.OUT, "v_at_lower_bound", **vals)
    if fit.inlier_frac < QC_UNSTABLE_INLIER:
        return QcOut(Verdict.UNKNOWN, "unstable", **vals)
    if not math.isfinite(fit.sigma_v) or fit.sigma_v / v_meta > QC_UNSTABLE_SIGMA_REL:
        return QcOut(Verdict.UNKNOWN, "unstable", **vals)
    verdict = classify_ratio(vals["r"])
    reason = "" if verdict == Verdict.IN else ("r_below" if vals["r"] < 1 else "r_above")
    return QcOut(verdict, reason, **vals)


def run_qc(rg: Radargram, trace: int, sample: float, inputs=None):
    config = DepthConfig()
    inputs = inputs or prepare_depth_inputs(rg, config)
    tau = sample * rg.dt_ns - inputs.zero.t_zero_ns
    box = candidate_box(rg, trace, sample, tau)
    hint = ApexHint(trace=trace, sample=sample)
    result = estimate_depth(rg, box, config, inputs=inputs, apex_hint=hint)
    picks = extract_picks(
        inputs.branch.background_removed, box, inputs.zero, dt_ns=rg.dt_ns, dx_m=rg.dx_m,
        f_c_hz=rg.f_c_hz, config=config, apex_hint=hint, period_px=inputs.branch.period_px,
    )
    if picks.apex is not None:
        assert (picks.apex.trace, picks.apex.k) == (result.apex_trace, result.apex_sample)
    v_meta = velocity_from_eps(rg.eps_r_meta)
    curv = fit_curvature(picks, radius_m=QC_RADIUS_M, air_gap="ground", v_fallback=v_meta,
                         config=config)
    qc = classify_curvature(curv, picks, v_meta)
    margin = (rg.n_samples - 1 - sample) * rg.dt_ns
    record_end = margin < RECORD_END_PERIODS * PERIOD_NS
    k_lo, k_hi = seam_rows(rg, inputs)
    score, _, _ = seam_score(inputs.branch.filtered, inputs.zero.t_zero_ns, rg.dt_ns, ref_half_traces(rg.dx_m))
    seam_max, seam_j = seam_near(score, trace, apex_half_traces(rg.dx_m))
    seam_lag = math.nan
    seam = bool(seam_max >= 1.0)
    return result, qc, dict(margin_ns=margin, record_end=record_end, seam_max=seam_max, seam_lag=seam_lag,
                            seam_j=seam_j, seam=seam, t_zero_ns=inputs.zero.t_zero_ns,
                            zero_rule=inputs.zero.zero_rule)


def _straight_line(eps, d, radius, n_samples=768, snr=20.0, dt=0.098):
    spec = _air_spec(n_samples=n_samples, dt=dt, eps=eps, snr=snr)
    spec.d_top_m, spec.radius_m = d, radius
    spec.pipe_times_ns = cylinder_times(spec.x_m, spec.x0_m, d, spec.velocity, radius)
    return spec


def _refraction_line(eps, d, radius, h=0.05, n_samples=768, dt=0.098):
    spec = BScanSpec(f_c_hz=F_C_HZ, eps_r=eps, d_top_m=d, radius_m=radius, air_gap_m=h,
                     offset_m=0.1, n_traces=401, n_samples=n_samples, dt_ns=dt)
    spec.pipe_times_ns = reflection_times(spec.x_m, offset_m=0.1, h=h, x0_m=spec.x0_m,
                                          d_top_m=d, radius_m=radius, v=spec.velocity)
    return spec


def _apex(spec: BScanSpec) -> tuple[int, float]:
    trace = spec.n_traces // 2
    return trace, (T_SRC_NS + float(spec.pipe_times_ns[trace])) / spec.dt_ns


def _record(test, case, rg, trace, sample, extra=None, inputs=None):
    result, qc, d = run_qc(rg, trace, sample, inputs)
    row = dict(test=test, case=case, status=str(result.status), depth_m=result.depth_top_m,
               v_rule=result.v_rule, verdict=qc.verdict, reason=qc.reason, r=qc.r,
               v_curv=qc.v_curv, sigma_v_rel=qc.sigma_v / velocity_from_eps(rg.eps_r_meta),
               inlier=qc.inlier_frac, n_picks=qc.n_picks, n_left=qc.n_left,
               n_right=qc.n_right, aperture_m=qc.aperture_m, **d)
    row.update(extra or {})
    return row


def ctests() -> pd.DataFrame:
    rows = []
    # C1 straight ray (effective air gap 0), merged single band, offset 0
    for eps in (9.0, 16.0):
        for radius in (0.0, 0.1):
            for d in (0.2, 0.3, 0.5, 0.75, 1.0, 1.5):
                spec = _straight_line(eps, d, radius)
                rg = make_radargram(spec, air_gap="air", eps_r_meta=eps, line_id="01.1/PathC1")
                tr, sa = _apex(spec)
                rows.append(_record("C1", f"eps{eps:g}_R{radius:g}_d{d:g}", rg, tr, sa,
                                    dict(eps_r=eps, radius_m=radius, d_true=d)))
    # C1b refraction upper bound (h 0.02 / 0.05, offset 0.1)
    for h in (0.02, 0.05):
        for radius in (0.0, 0.1):
            for d in (0.3, 0.6, 1.0):
                spec = _refraction_line(9.0, d, radius, h=h)
                rg = make_radargram(spec, eps_r_meta=9.0, line_id="01.1/PathC1b")
                tr, sa = _apex(spec)
                rows.append(_record("C1b", f"h{h:g}_R{radius:g}_d{d:g}", rg, tr, sa,
                                    dict(eps_r=9.0, radius_m=radius, d_true=d, h=h)))
    # C2 finite flat layer (+-1.0 m, 401 traces), pipe far below
    for amp, tag in ((2.0, "amp2"), (0.5, "amp0.5")):
        spec = _straight_line(9.0, 1.4, 0.0, n_samples=N_RECORD)
        x = spec.x_m
        inside = (np.abs(x - spec.x0_m) <= 1.0).astype(float)
        layer_t = 9.0
        spec.extra_events.append((np.full(len(x), layer_t), amp * inside))
        rg = make_radargram(spec, air_gap="air", eps_r_meta=9.0, line_id="01.1/PathC2")
        rows.append(_record("C2", f"flat_layer_{tag}", rg, spec.n_traces // 2,
                            (T_SRC_NS + layer_t) / spec.dt_ns))
    # C2b gently curved layer bump (very wide arc): radius of curvature 20 m
    spec = _straight_line(9.0, 1.4, 0.0, n_samples=N_RECORD)
    x = spec.x_m
    inside = (np.abs(x - spec.x0_m) <= 1.0).astype(float)
    arc = 9.0 + (2 / 0.1) * ((x - spec.x0_m) ** 2 / (2 * 20.0))
    spec.extra_events.append((arc, 2.0 * inside))
    rg = make_radargram(spec, air_gap="air", eps_r_meta=9.0, line_id="01.1/PathC2b")
    rows.append(_record("C2b", "wide_arc", rg, spec.n_traces // 2, (T_SRC_NS + 9.0) / spec.dt_ns))
    # C3 V crossing (two straight dipping events crossing at x0)
    for p in (5.0, 10.0, 15.0, 20.0, 30.0):
        spec = _straight_line(9.0, 1.4, 0.0, n_samples=N_RECORD)
        x = spec.x_m
        inside = (np.abs(x - spec.x0_m) <= 1.2).astype(float)
        tc = 10.0
        spec.extra_events.append((tc + p * (x - spec.x0_m), 1.5 * inside))
        spec.extra_events.append((tc - p * (x - spec.x0_m), 1.5 * inside))
        rg = make_radargram(spec, air_gap="air", eps_r_meta=9.0, line_id="01.1/PathC3")
        rows.append(_record("C3", f"X_p{p:g}", rg, spec.n_traces // 2,
                            (T_SRC_NS + tc) / spec.dt_ns, dict(slope_ns_per_m=p)))
    # C3b V only (two limbs meeting at x0, going down)
    for p in (10.0, 20.0):
        spec = _straight_line(9.0, 1.4, 0.0, n_samples=N_RECORD)
        x = spec.x_m
        inside = (np.abs(x - spec.x0_m) <= 1.2).astype(float)
        tc = 10.0
        spec.extra_events.append((tc + p * np.abs(x - spec.x0_m), 1.5 * inside))
        rg = make_radargram(spec, air_gap="air", eps_r_meta=9.0, line_id="01.1/PathC3b")
        rows.append(_record("C3b", f"V_p{p:g}", rg, spec.n_traces // 2,
                            (T_SRC_NS + tc) / spec.dt_ns, dict(slope_ns_per_m=p)))
    # C4 one wing only (right side of a d 0.5 m hyperbola)
    spec = _straight_line(9.0, 3.0, 0.0, n_samples=N_RECORD)  # main pipe outside record
    x = spec.x_m
    half = cylinder_times(x, spec.x0_m, 0.5, spec.velocity, 0.0)
    mask = (x >= spec.x0_m - 1e-9).astype(float)
    spec.extra_events.append((half, mask))
    rg = make_radargram(spec, air_gap="air", eps_r_meta=9.0, line_id="01.1/PathC4")
    rows.append(_record("C4", "one_wing", rg, spec.n_traces // 2,
                        (T_SRC_NS + float(half[spec.n_traces // 2])) / spec.dt_ns))
    # C5 record end: apex near the end of a 512-sample record (dt 0.098)
    for margin_ns in (0.8, 1.5, 3.0, 6.0):
        t_apex = (N_RECORD - 1) * 0.098 - margin_ns
        d = velocity_from_eps(9.0) * (t_apex - T_SRC_NS) / 2
        spec = _straight_line(9.0, d, 0.0, n_samples=N_RECORD)
        rg = make_radargram(spec, air_gap="air", eps_r_meta=9.0, line_id="01.1/PathC5")
        tr, sa = _apex(spec)
        rows.append(_record("C5", f"margin{margin_ns:g}ns", rg, tr, sa, dict(d_true=d)))
    # C6 trace seams on a C1 line (eps 9, d 0.5, R 0, 512 samples)
    base = _straight_line(9.0, 0.5, 0.0, n_samples=N_RECORD)
    rg0 = make_radargram(base, air_gap="air", eps_r_meta=9.0, line_id="01.1/PathC6")
    tr, sa = _apex(base)
    rows.append(_record("C6", "no_seam", rg0, tr, sa))
    for kind, offset in (("gain_2", 2), ("gain_2", 12), ("shift_3", 2), ("shift_3", 12),
                         ("dead_trace", 3), ("block_background", 2)):
        data = _with_seam(rg0.data, tr + offset, kind, 0.098)
        rg = replace(rg0, data=data)
        rows.append(_record("C6", f"{kind}_at+{offset}", rg, tr, sa, dict(seam_offset=offset)))
    # C7 unknown reasons on real paths: hint 5 ns below the apex (nothing there)
    rows.append(_record("C7", "hint_unmatched_+5ns", rg0, tr, sa + 5.0 / 0.098))
    return pd.DataFrame(rows)


def main() -> None:
    global RECORD_END_PERIODS, SEAM_RATIO
    edge = edge_zone()
    edge.to_csv(DOCS / "m29c_qc_edge.csv", index=False, float_format="%.4f")
    t_edge = float(edge["edge_ns"].max())
    t_end = MAIN_LOBE_PERIODS * PERIOD_NS + t_edge
    RECORD_END_PERIODS = math.ceil(t_end / PERIOD_NS * 2 - 1e-9) / 2  # up to 0.5 period
    print('edge zone periods', RECORD_END_PERIODS)
    print("edge", edge.to_string(), "\nT_edge", t_edge, "T_end", t_end, "periods", RECORD_END_PERIODS)

    null, power = seam_study()
    null.to_csv(DOCS / "m29c_qc_seam_null.csv", index=False, float_format="%.4f")
    power.to_csv(DOCS / "m29c_qc_seam_power.csv", index=False, float_format="%.4f")
    null_max = float(null["score_max"].max())
    SEAM_RATIO = 1.0
    with pd.option_context("display.width", 250):
        print("null", null.groupby(["dt_ns", "snr_db", "ring"])[["score_max", "n_seams", "dt_jump_max", "la_jump_max"]].max().to_string())
        print("null max score", null_max)
        print(power.groupby(["kind", "snr_db"])[["score_at_seam", "detected", "false_elsewhere"]].agg(["min", "mean", "max"]).to_string())
    const = dict(
        created=datetime.now().astimezone().isoformat(timespec="seconds"),
        QC_R_LO=QC_R_LO, QC_R_HI=QC_R_HI, QC_R_TOL=QC_R_TOL,
        QC_UNSTABLE_SIGMA_REL=QC_UNSTABLE_SIGMA_REL, QC_UNSTABLE_INLIER=QC_UNSTABLE_INLIER,
        QC_BOUND_REL=QC_BOUND_REL, QC_RADIUS_M=QC_RADIUS_M,
        MAIN_LOBE_PERIODS=MAIN_LOBE_PERIODS, EDGE_REL_ERROR=EDGE_REL_ERROR,
        T_EDGE_NS=t_edge, RECORD_END_PERIODS=RECORD_END_PERIODS,
        SEAM_NULL_SCORE_MAX=null_max, SEAM_TIME_PERIODS=SEAM_TIME_PERIODS,
        SEAM_AMP_RATIO=SEAM_AMP_RATIO, SEAM_NOISE_K=SEAM_NOISE_K,
        SEAM_ZERO_HALF_PERIODS=SEAM_ZERO_HALF_PERIODS, SEAM_LAG_HALF_PERIODS=0.5,
        SEAM_REF_HALF_M=BOX_HALF_WIDTH_M, SEAM_APEX_HALF_M=SEAM_APEX_HALF_M,
        V_BOUND_RULE="v at the v bound with x0, t0 interior -> out (before the inlier/sigma checks)",
        SYNTHETIC_ZERO_BAND_SCALE=ZERO_BAND_SCALE,
    )
    path = DOCS / "m29c_qc_constants.json"
    if path.exists():
        locked = json.loads(path.read_text(encoding="utf-8"))
        same = {k: v for k, v in locked.items() if k != "created"} == {
            k: v for k, v in const.items() if k != "created"
        }
        if not same:
            raise SystemExit("constants differ from the locked file; do not overwrite the lock")
        text = path.read_text(encoding="utf-8")  # keep the lock (and its hash) unchanged
    else:
        text = json.dumps(const, indent=1)
        path.write_text(text, encoding="utf-8")
    print("constants sha256", hashlib.sha256(text.encode()).hexdigest())

    ct = ctests()
    ct.to_csv(DOCS / "m29c_qc_ctests.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(ct[["test", "case", "status", "depth_m", "verdict", "reason", "r", "sigma_v_rel",
                  "inlier", "n_left", "n_right", "aperture_m", "margin_ns", "record_end",
                  "seam_max", "seam_lag", "seam", "zero_rule"]].to_string())


if __name__ == "__main__":
    main()
