"""Why is the air-coupled surface-reflection zero late after preprocessing?

Decomposes the shift of the `air_surface_late` event (spec_depth_v1 section 4)
by processing stage, by event composition (surface only / direct only / both)
and by pick mode, for

* the analytic Ricker test profile of tests/preprocess/test_timezero.py
  (500 MHz, h 0.3 m, s 0.1 m, surface amplitude -2, direct +3), and
* gprMax M20 (h 0.3 m): fs = free space (direct only), bg - fs = surface only,
  bg = no-pipe trace (both), full = M20.npy mean trace (what the code sees).

Candidate processing changes are also checked for the Z1 ghost-lobe case.

Read-only: imports src/ and tests/ helpers, never modifies them.
Single process, a few seconds. No random numbers except the test generator
(seed 0, fixed inside tests/synthetic_bscan.py).

Run:  conda run -n gpr-dev python -I analysis/processing/surface_zero_shift.py
Out:  docs/processing/surface_zero_shift.csv, surface_zero_shift_z1.csv,
      surface_zero_shift_hsweep.csv, surface_zero_shift.png
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from gpr_detection.preprocess.filters import remove_dc  # noqa: E402
from gpr_detection.preprocess.signals import parabolic_peak  # noqa: E402
from gpr_detection.preprocess.timezero import detect_zero_events  # noqa: E402
from synthetic_bscan import (  # noqa: E402
    T_SRC_NS,
    BScanSpec,
    make_bscan,
    ricker,
)

C0 = 0.299792458
SYN = ROOT / "data" / "synthetic_gprmax"
OUT = ROOT / "docs" / "processing"
V_EPS9 = C0 / 3.0


# --------------------------------------------------------------------- filters
def gauss_dewow(x, dt, fc, k_sigma=1.0):
    sigma = k_sigma * 1e9 / fc / dt
    return x - gaussian_filter1d(x, sigma=sigma, axis=0, mode="reflect", truncate=4.0)


def box_dewow(x, dt, fc, k_window=3.0):
    n = max(1, round(k_window * 1e9 / fc / dt))
    n += 1 - n % 2
    return x - uniform_filter1d(x, size=n, axis=0, mode="reflect")


def bp(x, dt, fc, low_frac=0.25, high_frac=2.0, order=4):
    high = min(high_frac * fc, 0.45 / dt * 1e9)
    sos = butter(order, [low_frac * fc, high], btype="bandpass", fs=1e9 / dt, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def chain(*steps):
    def run(x, dt, fc):
        y = np.asarray(x, dtype=np.float64)
        for step in steps:
            y = step(y, dt, fc)
        return y

    return run


def dc(x, dt, fc):
    return remove_dc(x)


VARIANTS = {
    "raw": chain(),
    "dc": chain(dc),
    "dc+gauss1": chain(dc, gauss_dewow),
    "dc+gauss1+bp (current)": chain(dc, gauss_dewow, bp),
    "dc+bp (no dewow)": chain(dc, bp),
    "dc+box3 (no bp)": chain(dc, box_dewow),
    "dc+box3+bp (v1)": chain(dc, box_dewow, bp),
    "dc+gauss2+bp": chain(dc, lambda x, dt, fc: gauss_dewow(x, dt, fc, 2.0), bp),
    "dc+gauss0.5+bp": chain(dc, lambda x, dt, fc: gauss_dewow(x, dt, fc, 0.5), bp),
    "dc+gauss1+bpL0.125": chain(dc, gauss_dewow, lambda x, dt, fc: bp(x, dt, fc, 0.125)),
    "dc+gauss1+bpL0.5": chain(dc, gauss_dewow, lambda x, dt, fc: bp(x, dt, fc, 0.5)),
    "dc+gauss1+bp order2": chain(dc, gauss_dewow, lambda x, dt, fc: bp(x, dt, fc, order=2)),
}


# --------------------------------------------------------------------- picking
def pick_near(trace, dt, t_ref, mode, half_ns=0.6):
    """Local max of the pick signal nearest to t_ref (fractional, ns)."""
    if mode == "envelope":
        s = np.abs(hilbert(trace))
    elif mode == "signed_min":
        s = -trace
    else:
        s = trace.copy()
    pk, _ = find_peaks(s)
    pk = pk[np.abs(pk * dt - t_ref) <= half_ns]
    if len(pk) == 0:
        return np.nan, np.nan
    k = int(pk[np.argmin(np.abs(pk * dt - t_ref))])
    kk, val = parabolic_peak(s, k)
    return kk * dt, val


def env_at(trace, dt, t):
    e = np.abs(hilbert(trace))
    return float(np.interp(t / dt, np.arange(len(e)), e))


# --------------------------------------------------------------------- inputs
def analytic_inputs():
    """Clean single traces of the test profile + the test's full mean trace."""
    spec = BScanSpec(f_c_hz=500e6, air_gap_m=0.3, offset_m=0.1, d_top_m=1.0)
    dt, fc, n = spec.dt_ns, spec.f_c_hz, spec.n_samples
    t = np.arange(n) * dt
    t_dir = T_SRC_NS + spec.offset_m / C0
    t_surf = T_SRC_NS + 2 * np.hypot(spec.air_gap_m, spec.offset_m / 2) / C0
    direct = spec.direct_amplitude * ricker(t - t_dir, fc)
    surface = spec.surface_amplitude * ricker(t - t_surf, fc)
    full_mean = make_bscan(spec).astype(np.float64)  # pipe + noise (seed 0)
    comps = {
        "surface_only": surface[:, None],
        "direct_only": direct[:, None],
        "both": (direct + surface)[:, None],
        "test_full": full_mean,  # mean taken after processing, as in the code
    }
    return dict(name="analytic", dt=dt, fc=fc, t_surf=t_surf, t_dir=t_dir, comps=comps,
                offset=spec.offset_m)


def m20_inputs():
    meta = json.loads((SYN / "M20.json").read_text())
    dt, fc = meta["dt_ns"], meta["center_freq_hz"]
    bg = np.load(SYN / "M20_bg.npy").astype(np.float64)
    fs = np.load(SYN / "M20_fs.npy").astype(np.float64)
    full = np.load(SYN / "M20.npy").astype(np.float64)
    bgc, fsc = bg[:, [1]] if bg.ndim == 2 and bg.shape[1] > 1 else bg.reshape(-1, 1), fs[:, [0]] if fs.ndim == 2 else fs.reshape(-1, 1)
    tz = meta["time_zero_candidates_ns"]
    comps = {
        "surface_only": bgc - fsc,
        "direct_only": fsc,
        "both": bgc,
        "test_full": full,
    }
    return dict(name="M20", dt=dt, fc=fc, t_surf=tz["surface_reflection_peak"],
                t_dir=tz["direct_wave_peak"], comps=comps, offset=meta["antenna_offset_m"])


# --------------------------------------------------------------------- analysis
def analyse(case, rows):
    dt, fc, t_s = case["dt"], case["fc"], case["t_surf"]
    for vname, fn in VARIANTS.items():
        proc = {c: fn(a, dt, fc).mean(axis=1) for c, a in case["comps"].items()}
        e_s_peak = pick_near(proc["surface_only"], dt, t_s, "envelope")[1]
        tail = env_at(proc["direct_only"], dt, t_s) / e_s_peak if e_s_peak else np.nan
        for comp, tr in proc.items():
            for mode in ("envelope", "signed_min"):
                tp, _ = pick_near(tr, dt, t_s, mode)
                rows.append(dict(case=case["name"], variant=vname, composition=comp,
                                 pick_mode=mode, t_pick_ns=round(tp, 4),
                                 shift_ns=round(tp - t_s, 4),
                                 depth_cm_eps9=round(-V_EPS9 * (tp - t_s) / 2 * 100, 2),
                                 direct_tail_ratio=round(tail, 3)))
        # what the code would do on the full profile (events, late event, h_est)
        filt = fn(case["comps"]["test_full"], dt, fc)
        ev = detect_zero_events(filt, dt, fc, antenna_offset_m=case["offset"])
        late = ev.events_k[-1] * dt if ev.n_events else np.nan
        rows.append(dict(case=case["name"], variant=vname, composition="code_detect_zero_events",
                         pick_mode="envelope", t_pick_ns=round(late, 4),
                         shift_ns=round(late - t_s, 4),
                         depth_cm_eps9=round(-V_EPS9 * (late - t_s) / 2 * 100, 2),
                         direct_tail_ratio=f"n_events={ev.n_events};h_est={ev.h_est_m and round(ev.h_est_m, 3)}"))


def z1_check():
    """Z1 trace (spec 11-1) through every variant: number of envelope events."""
    fc, dt, n, te = 500e6, 0.098, 512, 5.0
    t = np.arange(n) * dt
    after = t >= te
    tail = np.where(after, 0.6 * np.exp(-(t - te) / 1.5 * after), 0.0)
    data = np.tile((ricker(t - te, fc) + tail)[:, None], (1, 20))
    out = []
    for vname, fn in VARIANTS.items():
        filt = fn(data, dt, fc)
        ev = detect_zero_events(filt, dt, fc)
        env = np.abs(hilbert(filt.mean(axis=1)))
        k0 = int(round(te / dt))
        far = np.abs(t - te) > 1.0
        ghost = float(env[far & (t < te + 6)].max() / env[k0 - 3:k0 + 4].max())
        ring = (np.abs(t - te) >= 2.5) & (np.abs(t - te) <= 3.5)
        ghost3 = float(env[ring].max() / env[k0 - 3:k0 + 4].max())
        out.append(dict(variant=vname, n_events=ev.n_events,
                        events_ns=";".join(f"{k * dt:.3f}" for k in ev.events_k),
                        max_side_envelope_rel=round(ghost, 3),
                        envelope_rel_at_pm3ns=round(ghost3, 3)))
    return out


def h_sweep():
    """Interference bias of the late event vs air gap and surface amplitude (analytic, clean)."""
    fc, dt, n, s = 500e6, 0.05, 800, 0.1
    t = np.arange(n) * dt
    fn = VARIANTS["dc+gauss1+bp (current)"]
    out = []
    for h in (0.22, 0.25, 0.3, 0.35, 0.4, 0.5):
        for a_s in (-1.0, -2.0, -3.0):
            t_dir = T_SRC_NS + s / C0
            t_srf = T_SRC_NS + 2 * np.hypot(h, s / 2) / C0
            tr = 3.0 * ricker(t - t_dir, fc) + a_s * ricker(t - t_srf, fc)
            data = np.tile(tr[:, None], (1, 5))
            res = {}
            for name in ("raw", "dc+gauss1+bp (current)"):
                ev = detect_zero_events(VARIANTS[name](data, dt, fc), dt, fc, antenna_offset_m=s)
                late = ev.events_k[-1] * dt if ev.n_events >= 2 else np.nan
                res[name] = (ev.n_events, late - t_srf)
            out.append(dict(h_m=h, surface_amp=a_s, gap_ns=round(t_srf - t_dir, 3),
                            n_raw=res["raw"][0], shift_raw_ns=round(res["raw"][1], 3),
                            n_current=res["dc+gauss1+bp (current)"][0],
                            shift_current_ns=round(res["dc+gauss1+bp (current)"][1], 3)))
    del fn
    return out


def figure(case_a, case_m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, case in zip(axes, (case_a, case_m)):
        dt, fc, ts = case["dt"], case["fc"], case["t_surf"]
        fn = VARIANTS["dc+gauss1+bp (current)"]
        t = np.arange(case["comps"]["both"].shape[0]) * dt
        sel = (t > case["t_dir"] - 1.5) & (t < ts + 2.5)
        styles = {"surface_only": ("#1f77b4", "-"), "direct_only": ("#7f7f7f", "--"),
                  "both": ("#d62728", "-")}
        for comp, (col, ls) in styles.items():
            e = np.abs(hilbert(fn(case["comps"][comp], dt, fc).mean(axis=1)))
            ax.plot(t[sel], e[sel], color=col, ls=ls, lw=1.6, label=comp.replace("_", " "))
        ax.axvline(ts, color="k", lw=0.8, ls=":")
        tp, _ = pick_near(fn(case["comps"]["both"], dt, fc).mean(axis=1), dt, ts, "envelope")
        ax.axvline(tp, color="#d62728", lw=0.8, ls=":")
        ax.set_title(f"{case['name']}: envelope after DC+Gauss dewow+BP\n"
                     f"surface ref {ts:.3f} ns, both-pick {tp:.3f} ns ({tp - ts:+.3f})", fontsize=9)
        ax.set_xlabel("time (ns)")
        ax.set_ylabel("envelope (a.u.)")
        ax.legend(fontsize=8, frameon=False)
    fig.savefig(OUT / "surface_zero_shift.png", dpi=130)


def main():
    rows = []
    ca, cm = analytic_inputs(), m20_inputs()
    for case in (ca, cm):
        analyse(case, rows)
    with open(OUT / "surface_zero_shift.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    z1 = z1_check()
    with open(OUT / "surface_zero_shift_z1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(z1[0]))
        w.writeheader()
        w.writerows(z1)
    figure(ca, cm)

    for case in ("analytic", "M20"):
        print(f"\n== {case} (shift ns vs reference; envelope / signed_min) ==")
        print(f"{'variant':26s} {'surf_only':>18s} {'both':>18s} {'full':>18s} {'code':>8s} tail  code-info")
        for v in VARIANTS:
            def g(comp, mode):
                r = [x for x in rows if x["case"] == case and x["variant"] == v
                     and x["composition"] == comp and x["pick_mode"] == mode][0]
                return r
            parts = []
            for comp in ("surface_only", "both", "test_full"):
                parts.append(f"{g(comp, 'envelope')['shift_ns']:+.3f}/{g(comp, 'signed_min')['shift_ns']:+.3f}")
            code = g("code_detect_zero_events", "envelope")
            print(f"{v:26s} {parts[0]:>18s} {parts[1]:>18s} {parts[2]:>18s} {code['shift_ns']:+8.3f} "
                  f"{g('both', 'envelope')['direct_tail_ratio']:.3f} {code['direct_tail_ratio']}")
    print("\n== Z1 ==")
    for r in z1:
        print(r)
    hs = h_sweep()
    with open(OUT / "surface_zero_shift_hsweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hs[0]))
        w.writeheader()
        w.writerows(hs)
    print("\n== h / amplitude sweep (analytic, clean, late event shift ns) ==")
    for r in hs:
        print(r)


if __name__ == "__main__":
    main()
