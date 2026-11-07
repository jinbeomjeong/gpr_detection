"""M2-9 doc v2 -- physics checks for depth_physics.md section 7-2 items E1, E2, E4.

Analytic / ray-theory only (no gprMax run, no 4TU data, no ground truth).

E1  Air-gap refraction vs straight-ray hyperbola curvature:
    (a) forward: truth (eps 9, h, s, R, d) -> straight-ray eps' (fourtu_explore.straight_fit form)
    (b) inverse: "observed" straight-ray hyperbola (eps 8.69, h=0) fitted with air-gap models
        (h_m, s_m, R_m) -> eps needed; does s>0 / R>0 raise it (stronger h bound)?
    (c) shape residual: best RMS of an air-gap model fitted to an h=0 hyperbola (eps, d free)
    (d) trace-spacing (x scale) error that would hide h = 0.1 m
E2  Two-event separation limit and envelope widening for wavelets matched to the measured
    4TU apex-reflection envelope FWHM (2.9 / 3.8 ns) vs the nominal 500 MHz Ricker.
E4  Zero bias of the merged envelope peak relative to the surface-reflection peak.

Run (single process, BLAS 1 thread; E1 took 29 min on a saturated CPU, 2026-11-07):
    conda run -n gpr-dev python -I analysis/processing/m29_em_band_physics.py
    conda run -n gpr-dev python -I analysis/processing/m29_em_band_physics.py --reuse-e1   # figure / E2-E4 only
Outputs: analysis/processing/m29_em_band_physics.json, docs/research/m29_em_band_physics.png
"""

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "analysis" / "processing" / "m29_em_band_physics.json"
OUT_PNG = ROOT / "docs" / "research" / "m29_em_band_physics.png"
C0 = 0.299792458  # m/ns


# ----------------------------------------------------------------------------- ray geometry
def leg_time(xa, h, px, pz, v):
    """Fermat air->soil time (ns), antenna (xa, h), soil point (px, pz<=0); broadcasts."""
    xa, px, pz = np.broadcast_arrays(np.asarray(xa, float), np.asarray(px, float), np.asarray(pz, float))
    depth = np.abs(pz)
    if h <= 1e-12:
        return np.hypot(px - xa, depth) / v
    lo = np.minimum(xa, px)
    hi = np.maximum(xa, px)
    for _ in range(60):
        a = 0.5 * (lo + hi)
        g = (a - xa) / (C0 * np.hypot(a - xa, h)) + (a - px) / (v * np.hypot(a - px, depth))
        hi = np.where(g > 0, a, hi)
        lo = np.where(g > 0, lo, a)
    a = 0.5 * (lo + hi)
    return np.hypot(a - xa, h) / C0 + np.hypot(a - px, depth) / v


def refl_times(x, s, h, d_top, R, v):
    """Common-offset (Tx x-s/2, Rx x+s/2) top reflection of a cylinder (radius R), ns."""
    x = np.asarray(x, float)[:, None]
    zc = -(d_top + R)
    if R <= 1e-9:
        return (leg_time(x - s / 2, h, 0.0, -d_top, v) + leg_time(x + s / 2, h, 0.0, -d_top, v))[:, 0]
    th = np.linspace(0.0, np.pi, 361)[None, :]
    t = leg_time(x - s / 2, h, R * np.cos(th), zc + R * np.sin(th), v) + \
        leg_time(x + s / 2, h, R * np.cos(th), zc + R * np.sin(th), v)
    k = np.argmin(t, axis=1)
    step = np.pi / 360
    th0 = np.clip(th[0, k] - step, 0, np.pi)[:, None]
    fine = th0 + np.linspace(0, 2 * step, 81)[None, :]
    tf = leg_time(x - s / 2, h, R * np.cos(fine), zc + R * np.sin(fine), v) + \
        leg_time(x + s / 2, h, R * np.cos(fine), zc + R * np.sin(fine), v)
    return tf.min(axis=1)


def surf_time(s, h):
    return 2.0 * np.hypot(h, s / 2) / C0


def tau_model(x, eps, d_top, h, s, R):
    """Time after the surface reflection (zero = surface reflection peak), ns."""
    v = C0 / np.sqrt(eps)
    return refl_times(x, s, h, d_top, R, v) - surf_time(s, h)


def straight_fit(x, tau):
    """Same form as analysis/processing/fourtu_explore.py straight_fit (x0, t0, v), no robust loss."""
    k = int(np.argmin(tau))
    res = least_squares(lambda p: 2 / p[2] * np.sqrt((p[2] * p[1] / 2) ** 2 + (x - p[0]) ** 2) - tau,
                        x0=[x[k], tau[k], 0.1], bounds=([x.min(), 0.0, 0.02], [x.max(), 200.0, 0.6]))
    v = float(res.x[2])
    return (C0 / v) ** 2, float(np.sqrt(np.mean(res.fun ** 2)))


def airgap_fit(x, tau, h, s, R, eps_max=200.0):
    """Fit eps and d_top (x0 = 0 fixed) with an air-gap model; returns eps, d, rms."""
    k = int(np.argmin(tau))
    best = None
    for e0 in (6.0, 15.0, 40.0):
        v0 = C0 / np.sqrt(e0)
        d0 = max(v0 * tau[k] / 2, 0.05)
        try:
            res = least_squares(lambda p: tau_model(x, p[0], p[1], h, s, R) - tau, x0=[e0, d0],
                                bounds=([1.5, 0.02], [eps_max, 5.0]), x_scale=[5.0, 0.1])
        except ValueError:
            continue
        rms = float(np.sqrt(np.mean(res.fun ** 2)))
        if best is None or rms < best[2]:
            best = (float(res.x[0]), float(res.x[1]), rms)
    return best


def e1(results):
    out = {}
    # (a) forward
    rows = []
    for a_max in (1.0, 0.4):
        x = np.arange(-a_max, a_max + 1e-9, 0.02)
        for d in (0.3, 0.5, 0.8):
            for h in (0.0, 0.05, 0.1, 0.2, 0.3):
                for s, R in ((0.0, 0.0), (0.15, 0.0), (0.0, 0.1), (0.0, 0.25), (0.15, 0.25)):
                    tau = tau_model(x, 9.0, d, h, s, R)
                    e_s, rms = straight_fit(x, tau)
                    rows.append(dict(a_max=a_max, d=d, h=h, s=s, R=R, eps_straight=round(e_s, 3),
                                     rms_ns=round(rms, 4)))
    out["forward_eps9"] = rows
    # (b) inverse: observed = straight hyperbola eps 8.69 (h=0, s=0, R=0)
    rows = []
    for a_max in (1.0, 0.4):
        x = np.arange(-a_max, a_max + 1e-9, 0.02)
        for d in (0.3, 0.5, 0.8):
            tau = tau_model(x, 8.69, d, 0.0, 0.0, 0.0)
            for h in (0.0, 0.05, 0.1):
                for s, R in ((0.0, 0.0), (0.15, 0.0), (0.0, 0.1), (0.0, 0.25), (0.15, 0.25)):
                    e_f, d_f, rms = airgap_fit(x, tau, h, s, R)
                    rows.append(dict(a_max=a_max, d_obs=d, h_model=h, s_model=s, R_model=R,
                                     eps_fit=round(e_f, 2), d_fit=round(d_f, 3), rms_ns=round(rms, 4)))
    out["inverse_obs_eps8.69"] = rows
    # (c) shape residual vs model h (s=0, R=0)
    rows = []
    for a_max in (1.0, 0.4):
        x = np.arange(-a_max, a_max + 1e-9, 0.02)
        for d in (0.3, 0.5, 0.8):
            tau = tau_model(x, 8.69, d, 0.0, 0.0, 0.0)
            for h in (0.05, 0.1, 0.2, 0.3):
                e_f, d_f, rms = airgap_fit(x, tau, h, 0.0, 0.0, eps_max=400.0)
                rows.append(dict(a_max=a_max, d_obs=d, h_model=h, eps_fit=round(e_f, 2), rms_ns=round(rms, 4)))
    out["shape_rms"] = rows
    # (d) x-scale factor k = dx_assumed/dx_true that makes truth (eps 9, h 0.1) read eps' 8.69
    rows = []
    for d in (0.3, 0.5, 0.8):
        x = np.arange(-1.0, 1.0 + 1e-9, 0.02)
        e_s, _ = straight_fit(x, tau_model(x, 9.0, d, 0.1, 0.0, 0.0))
        rows.append(dict(d=d, eps_straight_true_x=round(e_s, 3), k_needed=round(float(np.sqrt(e_s / 8.69)), 3)))
    out["xscale"] = rows
    results["E1"] = out


# ----------------------------------------------------------------------------- wavelets
DT = 0.01  # ns
T = np.arange(0, 80, DT)
FC_PROC = 0.5  # GHz, processing centre frequency (4TU)


def ricker(t, f):
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def prep(trace):
    """fourtu_explore.prep: DC, dewow (3/fc box), zero-phase Butterworth 0.25fc-2fc (fc 500 MHz)."""
    xx = trace - trace.mean()
    w = int(round(3.0 / FC_PROC / DT)) | 1
    xx = xx - uniform_filter1d(xx, w, mode="nearest")
    sos = butter(4, [0.25 * FC_PROC, 2 * FC_PROC], btype="band", fs=1.0 / DT, output="sos")
    return sosfiltfilt(sos, xx)


def env(trace):
    return np.abs(hilbert(trace))


def fwhm_ns(e):
    k = int(np.argmax(e))
    half = e[k] / 2
    lo = k
    while lo > 0 and e[lo] > half:
        lo -= 1
    hi = k
    while hi < len(e) - 1 and e[hi] > half:
        hi += 1
    # linear interpolation at both crossings
    tl = T[lo] + (half - e[lo]) / (e[lo + 1] - e[lo]) * DT
    th = T[hi - 1] + (half - e[hi - 1]) / (e[hi] - e[hi - 1]) * DT
    return th - tl


def n_events(e):
    m = e.max()
    pk, _ = find_peaks(e, height=0.1 * m, prominence=0.05 * m)
    return pk


def wav(f, t0=20.0):
    return lambda dly=0.0: ricker(T - t0 - dly, f)


def match_f(target):
    fs = np.arange(0.15, 0.6, 0.0025)
    fw = np.array([fwhm_ns(env(prep(ricker(T - 20, f)))) for f in fs])
    return float(fs[int(np.argmin(np.abs(fw - target)))])


def e2_e4(results):
    out = {}
    wavelets = {"ricker500": 0.5, "fwhm2.9": match_f(2.9), "fwhm3.8": match_f(3.8)}
    out["wavelet_f_GHz"] = {k: round(v, 4) for k, v in wavelets.items()}
    out["wavelet_fwhm_ns"] = {k: round(fwhm_ns(env(prep(ricker(T - 20, f)))), 3) for k, f in wavelets.items()}
    deltas = np.round(np.arange(0.0, 5.0001, 0.01), 3)
    ratios = (-0.67, -1.5, -3.0, 0.67)
    sep, widen, bias = {}, {}, {}
    curves = {}
    for wn, f in wavelets.items():
        w = wav(f)
        e_sr = env(prep(w(0.0)))
        t_sr_alone = T[int(np.argmax(e_sr))]
        fw_sr = fwhm_ns(e_sr)
        for a in ratios:
            ne, fwd, zb = [], [], []
            for dl in deltas:
                tr = prep(w(0.0) + a * w(dl))  # direct coupling at 0, surface reflection at dl
                e = env(tr)
                pk = n_events(e)
                ne.append(len(pk))
                fwd.append(fwhm_ns(e) - fw_sr)
                # zero = strongest envelope peak if single, else late; bias vs SR-alone peak (+dl)
                k = int(np.argmax(e))
                kk = k
                if 0 < k < len(e) - 1:
                    y0, y1, y2 = e[k - 1], e[k], e[k + 1]
                    den = y0 - 2 * y1 + y2
                    kk = k + (0.5 * (y0 - y2) / den if den != 0 else 0.0)
                zb.append(float(kk * DT - (t_sr_alone + dl)))
            ne = np.array(ne)
            two = np.where(ne >= 2)[0]
            first2 = float(deltas[two[0]]) if len(two) else None
            # stable separation: smallest delta after which every larger delta has >= 2 events
            stable = None
            for i in range(len(deltas)):
                if np.all(ne[i:] >= 2):
                    stable = float(deltas[i])
                    break
            key = f"{wn}|a={a}"
            sep[key] = dict(first_two_ns=first2, stable_two_ns=stable)
            curves[key] = dict(n=ne.tolist(), widen=np.round(fwd, 4).tolist(), zbias=np.round(zb, 4).tolist())
            fwd = np.array(fwd)
            widen[key] = {f"delta_for_+{x}ns": (float(deltas[np.where(fwd >= x)[0][0]]) if np.any(fwd >= x) else None)
                          for x in (0.4, 1.4)}
            widen[key]["widen_at_h0.1_s0"] = round(float(np.interp(2 * 0.1 / C0, deltas, fwd)), 3)
            b = {}
            v = C0 / 3.0
            for h in (0.02, 0.05, 0.1):
                for s in (0.0, 0.15):
                    dl = (2 * np.hypot(h, s / 2) - s) / C0
                    zt = float(np.interp(dl, deltas, zb))
                    b[f"h{h}_s{s}"] = dict(delta_ns=round(dl, 3), zero_minus_sr_ns=round(zt, 3),
                                           depth_bias_cm_eps9=round(-v * zt / 2 * 100, 2),
                                           h_over_sqrt_eps_cm=round(h / 3.0 * 100, 2))
            bias[key] = b
    # separation in h for s = 0, 0.1, 0.2, 0.3 using stable delta
    hsep = {}
    for key, sv in sep.items():
        dl = sv["stable_two_ns"]
        if dl is None:
            continue
        hsep[key] = {f"s{s}": round(float(np.sqrt(((C0 * dl + s) / 2) ** 2 - s ** 2 / 4)), 3) for s in (0.0, 0.1, 0.2, 0.3)}
    out.update(separation=sep, h_sep_m=hsep, widening=widen, zero_bias=bias)
    results["E2_E4"] = out
    return deltas, curves


def figure(results, deltas, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    # panel 1: E1 inverse, eps needed vs (s, R) model at h = 0 / 0.05 / 0.1; failed fits (eps at bound) on top
    rows = results["E1"]["inverse_obs_eps8.69"]
    combos = [(0.0, 0.0), (0.15, 0.0), (0.0, 0.1), (0.0, 0.25), (0.15, 0.25)]
    marks = {0.0: "o", 0.05: "s", 0.1: "^"}
    cols = {(1.0, 0.3): "tab:blue", (1.0, 0.5): "tab:orange", (1.0, 0.8): "tab:green",
            (0.4, 0.3): "tab:cyan", (0.4, 0.5): "tab:red", (0.4, 0.8): "tab:purple"}
    for (am, d), c in cols.items():
        for h, mk in marks.items():
            xs, ys = [], []
            for i, (s, R) in enumerate(combos):
                r = [q for q in rows if q["a_max"] == am and q["d_obs"] == d and q["h_model"] == h
                     and q["s_model"] == s and q["R_model"] == R][0]
                xs.append(i + {0.0: -0.2, 0.05: 0.0, 0.1: 0.2}[h])
                ys.append(min(r["eps_fit"], 150.0))
            ax[0].plot(xs, ys, mk, color=c, ms=4, mfc="none" if h > 0 else c,
                       label=(f"|x|<={am} m, d {d} m" if h == 0.0 else None))
    ax[0].axhspan(120, 200, color="0.9", zorder=0)
    ax[0].text(0.0, 160, "fit fails: eps at bound (>=200)", fontsize=7)
    ax[0].set_yscale("log")
    ax[0].set_ylim(5, 220)
    ax[0].set_xticks(range(len(combos)))
    ax[0].set_xticklabels([f"s {s}\nR {R}" for s, R in combos], fontsize=7)
    ax[0].set_xlabel("model s / R (m); o h=0, square h=0.05, triangle h=0.1")
    ax[0].set_ylabel("eps_r needed to fit (log)")
    ax[0].set_title("E1: air-gap model fitted to straight hyperbola eps 8.69", fontsize=10)
    ax[0].axhline(8.69, color="k", lw=0.6, ls=":")
    ax[0].legend(fontsize=6.5, loc="center right")
    # panel 2: E2 widening vs delta
    for key, c in (("ricker500|a=-1.5", "tab:gray"), ("fwhm2.9|a=-1.5", "tab:blue"), ("fwhm3.8|a=-1.5", "tab:red"),
                   ("fwhm2.9|a=-0.67", "tab:cyan"), ("fwhm2.9|a=-3.0", "tab:purple")):
        w = np.array(curves[key]["widen"], float)
        n = np.array(curves[key]["n"])
        w1 = np.where(n == 1, w, np.nan)
        ax[1].plot(deltas, w1, color=c, lw=1.2, label=key)
    ax[1].axhspan(0.4, 1.4, color="0.85", zorder=0, label="4TU band - apex FWHM (0.4-1.4 ns)")
    ax[1].axvline(2 * 0.1 / C0, color="k", lw=0.6, ls=":", label="h 0.1 m, s 0")
    ax[1].set_xlim(0, 3.5)
    ax[1].set_xlabel("surface reflection delay after direct coupling (ns)")
    ax[1].set_ylabel("envelope FWHM increase (ns), single event only")
    ax[1].set_title("E2: widening of the merged band")
    ax[1].legend(fontsize=6.5)
    # panel 3: E4 zero bias (depth, eps 9) vs delta
    v = C0 / 3.0
    for key, c in (("fwhm2.9|a=-0.67", "tab:cyan"), ("fwhm2.9|a=-1.5", "tab:blue"), ("fwhm2.9|a=-3.0", "tab:purple"),
                   ("ricker500|a=-0.67", "tab:gray")):
        z = np.array(curves[key]["zbias"], float)
        n = np.array(curves[key]["n"])
        ax[2].plot(deltas, np.where(n == 1, -v * z / 2 * 100, np.nan), color=c, lw=1.2, label=key)
    ax[2].plot(deltas, v * deltas / 2 * 100, "k--", lw=0.8, label="h/sqrt(eps) (s=0)")
    ax[2].axvline(2 * 0.1 / C0, color="k", lw=0.6, ls=":", label="h 0.1 m, s 0")
    ax[2].axhline(0, color="0.5", lw=0.5)
    ax[2].set_xlim(0, 1.6)
    ax[2].set_ylim(-2, 8)
    ax[2].set_xlabel("surface reflection delay after direct coupling (ns)")
    ax[2].set_ylabel("depth bias of merged-peak zero (cm, eps 9)")
    ax[2].set_title("E4: merged-peak zero bias (+ = deeper)")
    ax[2].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)


def main():
    import sys
    t0 = time.time()
    results = {"note": "analytic/ray only; no 4TU data, no ground truth, no gprMax"}
    if "--reuse-e1" in sys.argv and OUT_JSON.exists():
        # E1 is slow (~30 min on a loaded CPU); reuse it and recompute the fast wavelet part + figure
        results["E1"] = json.loads(OUT_JSON.read_text(encoding="utf-8"))["E1"]
    else:
        e1(results)
    deltas, curves = e2_e4(results)
    figure(results, deltas, curves)
    results["elapsed_s"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(OUT_JSON), "png": str(OUT_PNG), "elapsed_s": results["elapsed_s"]}))


if __name__ == "__main__":
    main()
