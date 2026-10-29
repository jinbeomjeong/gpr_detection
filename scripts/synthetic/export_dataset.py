"""Export the synthetic gprMax B-scans to npy + json ground truth and run timing checks.

Inputs : data/synthetic_gprmax/_raw/<id>/<id>_merged.h5, <id>_bg_merged.h5, optional <id>_fs.h5
Outputs: data/synthetic_gprmax/<id>.npy      float32 [samples, traces], raw Ez (no processing)
         data/synthetic_gprmax/<id>_bg.npy   float32 [samples, 3], same model without pipe at x = 0.4/1.5/2.6 m
         data/synthetic_gprmax/<id>_fs.npy   float32 [samples, 1], antennas in free space (air-gap models only)
         data/synthetic_gprmax/<id>.json     ground truth, time-zero candidates, measured apex timing, checks
         data/synthetic_gprmax/manifest.csv

Run: conda run -n gprmax-acceleration python -I scripts/synthetic/export_dataset.py
"""

import csv
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import hilbert

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ray_geometry import (  # noqa: E402
    C0_M_PER_NS,
    direct_air_time,
    surface_reflection_time,
    trace_reflection_times,
)

PROJECT = HERE.parent.parent
OUT = PROJECT / "data" / "synthetic_gprmax"
RAW = OUT / "_raw"


def snap(value, dx):
    return round(round(value / dx) * dx, 6)


def read_ez(path):
    with h5py.File(path, "r") as f:
        ez = np.asarray(f["rxs/rx1/Ez"])
        dt = float(f.attrs["dt"])
        iterations = int(f.attrs["Iterations"])
    if ez.ndim == 1:
        ez = ez[:, None]
    return ez, dt, iterations


def refine_peak(env, k):
    """Parabolic sub-sample refinement of a local maximum at index k (returns fractional index)."""
    if 0 < k < len(env) - 1:
        a, b, c = env[k - 1], env[k], env[k + 1]
        den = a - 2 * b + c
        if den < 0:
            return k + 0.5 * (a - c) / den
    return float(k)


def envelope_pick(trace, t, t_lo, t_hi):
    """Envelope maximum of a trace in [t_lo, t_hi] (ns). Returns (time, envelope value, index, at_edge)."""
    env = np.abs(hilbert(trace))
    sel = np.where((t >= t_lo) & (t <= t_hi))[0]
    k = int(sel[np.argmax(env[sel])])
    kf = refine_peak(env, k)
    dt = t[1] - t[0]
    at_edge = k in (sel[0], sel[-1])
    return float(t[0] + kf * dt), float(env[k]), k, at_edge


def export_model(spec):
    mid = spec["model_id"]
    raw_dir = RAW / mid
    data, dt_s, iters = read_ez(raw_dir / f"{mid}_merged.h5")
    bg, dt_bg, iters_bg = read_ez(raw_dir / f"{mid}_bg_merged.h5")
    if not (math.isclose(dt_s, dt_bg, rel_tol=0, abs_tol=0) and iters == iters_bg):
        raise ValueError(f"{mid}: dt/Iterations differ between pipe and background runs")
    if data.shape != (iters, spec["n_traces"]) or bg.shape != (iters, 3):
        raise ValueError(f"{mid}: unexpected shapes {data.shape} {bg.shape}")
    fs = None
    fs_path = raw_dir / f"{mid}_fs.h5"
    if spec.get("fs_reference") and fs_path.exists():
        fs, dt_fs, iters_fs = read_ez(fs_path)
        if dt_fs != dt_s or iters_fs != iters:
            raise ValueError(f"{mid}: free-space reference dt/Iterations differ")

    dt_ns = dt_s * 1e9
    t = np.arange(iters) * dt_ns  # gprMax Ez receiver sample k is at k*dt (E-field offset 0)
    f_ghz = spec["center_freq_hz"] * 1e-9
    v = spec["velocity_m_per_ns"]
    dx = spec["dx_cell_m"]
    s = spec["antenna_offset_m"]
    h = round(spec["y_antenna_m"] - spec["y_surface_m"], 6)
    radius = spec["pipe_radius_m"]
    d_top = spec["pipe_top_depth_m"]
    x0 = spec["pipe_center_x_m"]
    n = spec["n_traces"]

    tx = snap(spec["scan_start_m"] - s / 2, dx) + np.arange(n) * spec["trace_step_m"]
    rx = snap(spec["scan_start_m"] + s / 2, dx) + np.arange(n) * spec["trace_step_m"]
    x_mid = 0.5 * (tx + rx)
    s_eff = round(float(rx[0] - tx[0]), 6)

    t_src = math.sqrt(2.0) / f_ghz  # Ricker current peak (ns)
    t_ray = trace_reflection_times(x_mid, s_eff, h, x0, d_top, radius, v)

    # Pipe-only response: subtract the no-pipe run (middle background trace, lateral invariance checked below).
    residual = data.astype(np.float64) - bg[:, 1:2].astype(np.float64)

    ia = int(np.argmin(np.abs(x_mid - x0)))
    t_expect = t_src + t_ray[ia]
    t_apex, env_apex, k_apex, edge = envelope_pick(residual[:, ia], t, t_expect - 0.6 / f_ghz, t_expect + 0.6 / f_ghz)
    win = (t >= t_expect - 0.6 / f_ghz) & (t <= t_expect + 0.6 / f_ghz)
    idx = np.where(win)[0]
    k_signed = int(idx[np.argmax(np.abs(residual[idx, ia]))])
    t_apex_signed = float(t[0] + refine_peak(np.abs(residual[:, ia]), k_signed) * dt_ns)
    polarity = int(np.sign(residual[k_signed, ia]))
    # Onset: first sample before the envelope peak where |residual| exceeds 5 % of the envelope peak.
    seg = np.abs(residual[: k_apex + 1, ia])
    above = np.where(seg > 0.05 * env_apex)[0]
    t_apex_onset5 = float(t[above[0]]) if above.size else None
    lag = t_apex - t_expect

    # Per-trace picks for a velocity check: residual time vs ray time should be flat if the grid velocity is right.
    picks = [None] * n
    rows = []
    for i in range(n):
        if abs(x_mid[i] - x0) > 1.0:
            continue
        te = t_src + lag + t_ray[i]
        if te + 0.35 / f_ghz > t[-1]:
            continue
        tp, ep, _, at_edge = envelope_pick(residual[:, i], t, te - 0.35 / f_ghz, te + 0.35 / f_ghz)
        if ep >= 0.2 * env_apex and not at_edge:
            picks[i] = round(tp, 4)
            rows.append((t_ray[i], tp - t_src - t_ray[i]))
    rows = np.array(rows)
    slope, intercept = np.polyfit(rows[:, 0], rows[:, 1], 1)
    resid_rms = float(np.sqrt(np.mean((rows[:, 1] - (intercept + slope * rows[:, 0])) ** 2)))
    spread = float(np.max(rows[:, 1]) - np.min(rows[:, 1]))

    # Time-zero candidates.
    env_bg = np.abs(hilbert(bg[:, 1].astype(np.float64)))
    k_dir = int(np.argmax(env_bg))
    tz_direct = float(t[0] + refine_peak(env_bg, k_dir) * dt_ns)
    t_surf_geom = surface_reflection_time(s_eff, h) if h > 0 else None
    tz_surface_meas = None
    if fs is not None:
        surf = bg[:, 1].astype(np.float64) - fs[:, 0].astype(np.float64)
        te_s = t_src + t_surf_geom
        tz_surface_meas, _, _, _ = envelope_pick(surf, t, te_s - 0.5 / f_ghz, te_s + 1.0 / f_ghz)
    tz_surface = tz_surface_meas if tz_surface_meas is not None else (tz_direct if h == 0 else None)
    tz = {"source_peak": t_src, "direct_wave_peak": tz_direct, "surface_reflection_peak": tz_surface}
    t0_eff = {k: (t_apex - val if val is not None else None) for k, val in tz.items()}
    d_eff = {k: (v * val / 2 if val is not None else None) for k, val in t0_eff.items()}

    # Artefact checks (all relative to the apex reflection envelope).
    lateral = [float(np.max(np.abs(bg[:, j].astype(np.float64) - bg[:, 1])) / env_apex) for j in (0, 2)]
    pre = 0.0
    for i in range(n):
        cut = t_src + t_ray[i] - 1.2 / f_ghz
        m = t < cut
        if m.any():
            pre = max(pre, float(np.max(np.abs(residual[m, i]))))
    pre_rel = pre / env_apex
    pml_y = spec["pml_cells"] * dx
    depth_pml = spec["y_surface_m"] - pml_y
    t_bottom = t_src + 2 * h / C0_M_PER_NS + 2 * math.hypot(depth_pml, s_eff / 2) / v
    bottom_rel = None
    if t_bottom + 0.5 / f_ghz < t[-1]:
        m = (t >= t_bottom - 0.5 / f_ghz) & (t <= t_bottom + 0.5 / f_ghz)
        bottom_rel = float(np.max(env_bg[m]) / env_apex)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{mid}.npy", data.astype(np.float32))
    np.save(OUT / f"{mid}_bg.npy", bg.astype(np.float32))
    if fs is not None:
        np.save(OUT / f"{mid}_fs.npy", fs.astype(np.float32))

    t0_top = 2 * d_top / v
    t0_center = 2 * (d_top + radius) / v
    meta = {
        "model_id": mid,
        "array_layout": "[samples, traces], float32, raw Ez (V/m) from gprMax 4.0.1 2D TMz, no gain/filter/time-zero shift",
        "dt_ns": dt_ns,
        "n_samples": iters,
        "n_traces": n,
        "dx_m": spec["trace_step_m"],
        "time_of_sample0_ns": 0.0,
        "trace_x_m_first": float(x_mid[0]),
        "trace_x_m": "x_m[i] = trace_x_m_first + i * dx_m (Tx/Rx midpoint)",
        "eps_r": spec["eps_r"],
        "soil_sigma_S_per_m": spec["soil_sigma_S_per_m"],
        "velocity_m_per_ns": v,
        "pipe_material": spec["pipe_kind"],
        "pipe_wall_m": spec["pipe_wall_m"],
        "pipe_center_x_m": x0,
        "pipe_top_depth_m": d_top,
        "pipe_center_depth_m": d_top + radius,
        "pipe_radius_m": radius,
        "center_freq_hz": spec["center_freq_hz"],
        "wavelet": "gprMax ricker (current of z-directed Hertzian dipole), peak at sqrt(2)/f",
        "air_gap_m": h,
        "antenna_offset_m": s_eff,
        "fdtd_cell_m": dx,
        "pml_cells": spec["pml_cells"],
        "time_window_ns": spec["time_window_ns"],
        "t0_definition": (
            "t0 = two-way vertical travel time in soil from the ground surface to the pipe top, "
            "zero offset: t0_top_ns = 2*d_top/v. Then d_top = v*t0/2. "
            "Times in the array are simulation time (sample k at k*dt_ns); subtract a time-zero "
            "(see time_zero_candidates_ns) before using the hyperbola formula. With a pipe of radius R "
            "the exact zero-offset curve is t(x) = (2/v)*(sqrt((d_top+R)^2+(x-x0)^2) - R)."
        ),
        "t0_top_ns": t0_top,
        "t0_center_ns": t0_center,
        "air_two_way_ns": 2 * h / C0_M_PER_NS,
        "ray_reflection_time_ns": [round(float(x), 5) for x in t_ray],
        "ray_reflection_time_definition": (
            "geometric-optics travel time Tx -> specular point on pipe top -> Rx for each trace, "
            "including antenna offset and refraction through the air gap; excludes the source delay "
            "and the wavelet lag. Data time of the reflection envelope peak ~= source_peak + wavelet_lag + this."
        ),
        "time_zero_candidates_ns": {
            "source_peak": t_src,
            "direct_wave_peak": tz_direct,
            "surface_reflection_peak": tz_surface,
            "notes": (
                "source_peak = sqrt(2)/f (Ricker current peak, exact). direct_wave_peak = envelope maximum of "
                "the no-pipe trace (direct coupling; for h=0 it also contains the ground wave and the surface "
                "reflection). surface_reflection_peak = envelope maximum of (no-pipe - free-space) for h>0, "
                "equal to direct_wave_peak for h=0."
            ),
        },
        "surface_reflection_geom_ns": t_surf_geom,
        "direct_air_geom_ns": direct_air_time(s_eff),
        "apex": {
            "trace_index": ia,
            "x_m": float(x_mid[ia]),
            "ray_time_ns": float(t_ray[ia]),
            "expected_envelope_peak_ns": t_expect,
            "envelope_peak_ns": t_apex,
            "envelope_peak_at_window_edge": bool(edge),
            "signed_peak_ns": t_apex_signed,
            "signed_peak_polarity": polarity,
            "onset_5pct_ns": t_apex_onset5,
            "wavelet_lag_ns": lag,
            "envelope_peak_value": env_apex,
        },
        "t0_effective_ns": t0_eff,
        "depth_from_t0_effective_m": d_eff,
        "velocity_check": {
            "n_picks": int(len(rows)),
            "slope_residual_vs_raytime": float(slope),
            "apparent_v_ratio_from_pick_slope": float(1.0 / (1.0 + slope)),
            "intercept_ns": float(intercept),
            "rms_after_linear_ns": resid_rms,
            "residual_spread_ns": spread,
            "max_abs_offset_from_apex_m": float(max(abs(x_mid[i] - x0) for i in range(n) if picks[i] is not None)),
        },
        "trace_envelope_pick_ns": picks,
        "checks": {
            "bg_lateral_diff_rel_x0.4_x2.6": lateral,
            "pre_arrival_residual_rel": pre_rel,
            "bottom_pml_window_ns": t_bottom,
            "bottom_pml_window_envelope_rel": bottom_rel,
        },
        "files": {
            "bscan": f"{mid}.npy",
            "background": f"{mid}_bg.npy",
            "free_space": f"{mid}_fs.npy" if fs is not None else None,
            "gprmax_input": f"scripts/synthetic/inputs/{mid}.in",
        },
    }
    (OUT / f"{mid}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    return meta


def main():
    specs = json.loads((HERE / "model_specs.json").read_text(encoding="utf-8"))
    fields = [
        "model_id", "npy", "json", "pipe_material", "pipe_outer_diameter_m", "pipe_radius_m", "pipe_top_depth_m",
        "pipe_center_depth_m", "pipe_center_x_m", "eps_r", "soil_sigma_S_per_m", "velocity_m_per_ns",
        "center_freq_mhz", "air_gap_m", "antenna_offset_m", "n_samples", "n_traces", "dt_ns", "dx_m",
        "time_window_ns", "fdtd_cell_m", "t0_top_ns", "t0_center_ns", "tz_source_ns", "tz_direct_ns",
        "tz_surface_ns", "apex_envelope_peak_ns", "wavelet_lag_ns", "apparent_v_ratio_pick_slope",
    ]
    rows = []
    for spec in specs:
        m = export_model(spec)
        tzc = m["time_zero_candidates_ns"]
        rows.append({
            "model_id": m["model_id"], "npy": m["files"]["bscan"], "json": f"{m['model_id']}.json",
            "pipe_material": m["pipe_material"], "pipe_outer_diameter_m": 2 * m["pipe_radius_m"],
            "pipe_radius_m": m["pipe_radius_m"], "pipe_top_depth_m": m["pipe_top_depth_m"],
            "pipe_center_depth_m": round(m["pipe_center_depth_m"], 6), "pipe_center_x_m": m["pipe_center_x_m"],
            "eps_r": m["eps_r"], "soil_sigma_S_per_m": m["soil_sigma_S_per_m"],
            "velocity_m_per_ns": round(m["velocity_m_per_ns"], 6), "center_freq_mhz": m["center_freq_hz"] / 1e6,
            "air_gap_m": m["air_gap_m"], "antenna_offset_m": m["antenna_offset_m"], "n_samples": m["n_samples"],
            "n_traces": m["n_traces"], "dt_ns": round(m["dt_ns"], 8), "dx_m": m["dx_m"],
            "time_window_ns": m["time_window_ns"], "fdtd_cell_m": m["fdtd_cell_m"],
            "t0_top_ns": round(m["t0_top_ns"], 4), "t0_center_ns": round(m["t0_center_ns"], 4),
            "tz_source_ns": round(tzc["source_peak"], 4), "tz_direct_ns": round(tzc["direct_wave_peak"], 4),
            "tz_surface_ns": None if tzc["surface_reflection_peak"] is None else round(tzc["surface_reflection_peak"], 4),
            "apex_envelope_peak_ns": round(m["apex"]["envelope_peak_ns"], 4),
            "wavelet_lag_ns": round(m["apex"]["wavelet_lag_ns"], 4),
            "apparent_v_ratio_pick_slope": round(m["velocity_check"]["apparent_v_ratio_from_pick_slope"], 5),
        })
        a, vc, ck = m["apex"], m["velocity_check"], m["checks"]
        print(f"{m['model_id']} shape=({m['n_samples']},{m['n_traces']}) dt={m['dt_ns']:.6f} "
              f"apex={a['envelope_peak_ns']:.3f} exp={a['expected_envelope_peak_ns']:.3f} lag={a['wavelet_lag_ns']:+.3f} "
              f"edge={a['envelope_peak_at_window_edge']} pol={a['signed_peak_polarity']:+d} "
              f"vg/vt={vc['apparent_v_ratio_from_pick_slope']:.4f} n={vc['n_picks']} rms={vc['rms_after_linear_ns']:.3f} "
              f"lat={max(ck['bg_lateral_diff_rel_x0.4_x2.6']):.2e} pre={ck['pre_arrival_residual_rel']:.2e} "
              f"bot={ck['bottom_pml_window_envelope_rel']}", flush=True)
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
