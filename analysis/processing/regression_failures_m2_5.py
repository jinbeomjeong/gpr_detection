"""Decompose the two M2-5 regression failures (R5 M15, R13a M20).

R5 M15: apex envelope-peak time of the apex trace under each processing step,
vs json ``apex.envelope_peak_ns`` (raw no-pipe-subtracted trace) and
``expected_envelope_peak_ns`` (source peak + ray time).

R13a M20: refraction QC with the json antenna height (0.3 m), s 0.1 m, R 0.1 m
on picks from different backgrounds / filters / zeros, to separate the zero
delay, the processed wing picks and the background method.

Read-only: imports src and scripts/eval_depth_synthetic.supply_box.
Single process; QC on a narrowed eps grid 6.5-10.5 (same algorithm).
Run:  conda run -n gpr-dev python -I analysis/processing/regression_failures_m2_5.py
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, hilbert

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_depth_synthetic import supply_box  # noqa: E402
from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402
from gpr_detection.depth.hyperbola import fixed_velocity_apex  # noqa: E402
from gpr_detection.depth.picks import extract_picks  # noqa: E402
from gpr_detection.depth.refraction import refraction_qc  # noqa: E402
from gpr_detection.io.synthetic import (  # noqa: E402
    load_synthetic,
    load_synthetic_background,
    load_synthetic_truth,
)
from gpr_detection.preprocess.filters import (  # noqa: E402
    bandpass,
    dewow,
    prepare_fitting_branch,
    remove_dc,
)
from gpr_detection.preprocess.signals import parabolic_peak  # noqa: E402
from gpr_detection.preprocess.timezero import given_zero  # noqa: E402

SYN = ROOT / "data" / "synthetic_gprmax"


def env_peak_near(trace, dt, t_ref, half_ns=0.6):
    e = np.abs(hilbert(trace))
    pk, _ = find_peaks(e)
    pk = pk[np.abs(pk * dt - t_ref) <= half_ns]
    k = int(pk[np.argmax(e[pk])])
    return parabolic_peak(e, k)[0] * dt


def r5_m15():
    mid = "M15"
    truth = load_synthetic_truth(mid)
    meta = json.loads((SYN / f"{mid}.json").read_text())
    rg = load_synthetic(mid)
    dt, fc = rg.dt_ns, rg.f_c_hz
    d = rg.data.astype(np.float64)
    bg = load_synthetic_background(mid).astype(np.float64)
    j = truth.apex_trace_index
    t_json = truth.apex_envelope_peak_ns
    t_exp = meta["apex"]["expected_envelope_peak_ns"]
    ideal = d - bg[:, [1]]
    rows = {
        "raw D-bg": ideal[:, j],
        "D-bg + DC": remove_dc(ideal)[:, j],
        "D-bg + DC + dewow": dewow(remove_dc(ideal), dt, fc)[:, j],
        "D-bg + DC + bandpass": bandpass(remove_dc(ideal), dt, fc)[:, j],
        "D-bg + (1)-(3)": bandpass(dewow(remove_dc(ideal), dt, fc), dt, fc)[:, j],
        "median bg, no filter": (d - np.median(d, axis=1, keepdims=True))[:, j],
        "current: (1)-(3) + median bg": prepare_depth_inputs(rg).branch.background_removed[:, j],
    }
    print(f"[R5 M15] json envelope {t_json:.3f}, expected (src+ray) {t_exp:.3f}, lag {t_json - t_exp:+.3f} ns")
    for name, tr in rows.items():
        t = env_peak_near(tr, dt, t_json)
        print(f"  {name:32s} {t:.3f}  vs json {t - t_json:+.3f}  vs expected {t - t_exp:+.3f}", flush=True)


def r13a_m20():
    mid = "M20"
    truth = load_synthetic_truth(mid)
    rg = load_synthetic(mid)
    dt = rg.dt_ns
    box = supply_box(truth, rg)
    inputs = prepare_depth_inputs(rg)
    bg = load_synthetic_background(mid).astype(np.float64)
    ideal = rg.data.astype(np.float64) - bg[:, [1]]
    ideal_filtered = prepare_fitting_branch(replace(rg, data=ideal.astype(np.float32))).filtered
    sections = {
        "median bg + (1)-(3) [current]": inputs.branch.background_removed,
        "ideal bg + (1)-(3)": ideal_filtered,
        "ideal bg, raw": ideal,
    }
    zeros = {
        "detected": inputs.zero,
        "json surface": given_zero(truth.surface_reflection_peak_ns, dt),
    }
    cfg = DepthConfig(refraction_eps_min=6.5, refraction_eps_max=10.5)
    print(f"[R13a M20] detected zero {inputs.zero.t_zero_ns:.3f}, json surface "
          f"{truth.surface_reflection_peak_ns:.3f}", flush=True)
    for sname, sec in sections.items():
        for zname, zero in zeros.items():
            if sname != "median bg + (1)-(3) [current]" and zname == "detected":
                continue
            picks = extract_picks(sec, box, zero, dt_ns=dt, dx_m=rg.dx_m, f_c_hz=rg.f_c_hz)
            vertex = fixed_velocity_apex(picks)
            for thin in (1, 2):
                x, tau = picks.x_m[::thin], picks.tau_ns[::thin]
                qc = refraction_qc(x, tau, x0_m=vertex.x0_m, tau_apex_ns=vertex.t0_ns, h_m=0.3,
                                   offset_m=0.1, radius_m=0.1, config=cfg)
                print(f"  {sname:30s} zero={zname:12s} thin={thin} n={len(x):3d} "
                      f"apex t0 {vertex.t0_ns:.3f}: eps {qc.eps_fit:.2f}, "
                      f"d {100 * (qc.d_top_m - truth.pipe_top_depth_m):+.2f} cm, rms {qc.rms_ns:.3f}",
                      flush=True)


def r13a_wing_lag():
    """M20: picks (ideal bg, raw, json zero) minus the json ray time, along the wing.

    Also checks that ``reflection_times`` reproduces ``ray_reflection_time_ns``.
    """
    from gpr_detection.depth.refraction import reflection_times
    from gpr_detection.io.radargram import velocity_from_eps

    mid = "M20"
    truth = load_synthetic_truth(mid)
    meta = json.loads((SYN / f"{mid}.json").read_text())
    rg = load_synthetic(mid)
    dt = rg.dt_ns
    box = supply_box(truth, rg)
    bg = load_synthetic_background(mid).astype(np.float64)
    ideal = rg.data.astype(np.float64) - bg[:, [1]]
    zero = given_zero(truth.surface_reflection_peak_ns, dt)
    picks = extract_picks(ideal, box, zero, dt_ns=dt, dx_m=rg.dx_m, f_c_hz=rg.f_c_hz)
    ray = np.asarray(truth.ray_reflection_time_ns)
    x_json = truth.trace_x_m_first + np.arange(rg.n_traces) * rg.dx_m
    model = reflection_times(x_json, offset_m=0.1, h=0.3, x0_m=truth.pipe_center_x_m,
                             d_top_m=0.6, radius_m=0.1, v=velocity_from_eps(9.0))
    print(f"[R13a lag] model vs json ray time: max |diff| {np.max(np.abs(model - ray)):.4f} ns")
    t_surf_geom = meta["surface_reflection_geom_ns"]
    resid = picks.tau_ns - (ray[picks.trace_idx] - t_surf_geom)
    off = np.abs(picks.trace_idx - truth.apex_trace_index) * rg.dx_m
    for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)):
        sel = (off >= lo) & (off < hi)
        if sel.any():
            print(f"  |x-x0| {lo:.1f}-{hi:.1f} m: pick - ray (rel. surface) {resid[sel].mean():+.3f} ns "
                  f"(n {sel.sum()})", flush=True)


if __name__ == "__main__":
    if "--lag-only" in sys.argv:
        r13a_wing_lag()
        sys.exit(0)
    r5_m15()
    if "--r5-only" not in sys.argv:
        r13a_m20()
        r13a_wing_lag()
