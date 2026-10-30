"""Evidence for the pipe_band depth spec (checklist M2-6).

Part A  Pseudo-band B-scans from the 2D gprMax set (20 models).
        A pipe parallel to the survey line and directly below it gives, on every trace,
        the same zero-offset response as the apex trace of a perpendicular crossing.
        We tile the apex trace (json apex.trace_index) over a band region and fill the
        remaining traces with the no-pipe model trace (<id>_bg.npy), add white noise
        (seed 0), run the spec_depth_v1 section 3 fitting branch (DC, dewow, bandpass),
        then compare background-removal variants and band-top picking rules.
Part B  Analytic geometry tables: lateral offset bias, oblique-crossing curvature
        bias, band-likeness threshold angle, pipe grade.
Part C  Oblique small-angle band: recover the crossing angle from the quadratic
        coefficient of the band-top times with v known (straight ray, R fixed).

Single process, no GPU, a few tens of seconds.
Run:  conda run -n gpr-dev python -I analysis/processing/pipe_band_check.py
Out:  docs/processing/pipe_band_check_synthetic.csv
      docs/processing/pipe_band_check_geometry.csv
      docs/processing/pipe_band_check_oblique.csv
      docs/processing/pipe_band_check.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data" / "synthetic_gprmax"
OUT = ROOT / "docs" / "processing"
C0 = 0.299792458  # m/ns
SEED = 0
N_TR = 160  # traces in the pseudo-band section
NOISE_REL = 0.05  # noise std relative to max |pipe-only apex trace|
H_MAX_M = 0.5
MODELS = [f"M{i:02d}" for i in range(1, 21)]


# ----------------------------------------------------------------------------- helpers
def envelope(a: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.abs(hilbert(a, axis=axis))


def parabolic_peak(y: np.ndarray, k: int, dt: float) -> tuple[float, float]:
    if 0 < k < len(y) - 1:
        y0, y1, y2 = y[k - 1], y[k], y[k + 1]
        den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        return (k + delta) * dt, float(y1 - 0.25 * (y0 - y2) * delta)
    return k * dt, float(y[k])


def onset_time(e: np.ndarray, k_peak: int, dt: float, frac: float = 0.5) -> float:
    """Last crossing of frac*peak before the peak (linear interpolation), ns."""
    level = frac * e[k_peak]
    k = k_peak
    while k > 0 and e[k] >= level:
        k -= 1
    if k == 0:
        return 0.0
    # e[k] < level <= e[k+1]
    return (k + (level - e[k]) / (e[k + 1] - e[k])) * dt


def preprocess(D: np.ndarray, dt: float, f_hz: float) -> np.ndarray:
    """spec_depth_v1 section 3 steps (1)-(3): DC, dewow (3/f_c), Butterworth 4 zero-phase."""
    fc = f_hz / 1e9  # GHz
    a = D - D.mean(axis=0, keepdims=True)
    w = int(round(3.0 / fc / dt)) | 1
    a = a - uniform_filter1d(a, size=w, axis=0, mode="nearest")
    fs = 1.0 / dt
    hi = min(2 * fc, 0.45 * fs)
    sos = butter(4, [0.25 * fc, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, a, axis=0)


def zero_events(A: np.ndarray, dt: float, f_hz: float) -> list[tuple[float, float]]:
    """spec section 4-1 on the mean trace (before background removal)."""
    m = A.mean(axis=1)
    e = envelope(m)
    n25 = int(0.25 * len(e))
    t_max = int(np.argmax(e[:n25])) * dt
    t_end = t_max + 2 * H_MAX_M / C0 + 0.5e9 / f_hz
    k_end = min(int(t_end / dt) + 1, len(e))
    ee = e[:k_end]
    pk, _ = find_peaks(ee, height=0.1 * ee.max(), prominence=0.05 * ee.max())
    return [parabolic_peak(ee, int(k), dt) + (int(k),) for k in pk], e


def select_zero(events, air_gap: float):
    if air_gap > 0 and len(events) >= 2:
        t, _, k = events[-1]
        return t, k, "air_surface_late"
    t, _, k = events[0]
    return t, k, "air_single_event" if air_gap > 0 else "ground_direct_peak"


def band_picks(R: np.ndarray, dt: float, cols: np.ndarray, k_lo: int, k_hi: int,
               k_box0: int, k_box1: int):
    """First envelope peak >= 0.3 * E_box_max per trace inside [k_lo, k_hi]."""
    E = envelope(R[:, cols])
    e_ref = E[k_box0:k_box1 + 1].max()
    t_pk, t_on, amp = [], [], []
    for j in range(E.shape[1]):
        seg = E[k_lo:k_hi + 1, j]
        pk, _ = find_peaks(seg, height=0.3 * e_ref)
        if len(pk) == 0:
            continue
        k = k_lo + int(pk[0])
        t, a = parabolic_peak(E[:, j], k, dt)
        t_pk.append(t)
        t_on.append(onset_time(E[:, j], k, dt))
        amp.append(a)
    return np.array(t_pk), np.array(t_on), np.array(amp), e_ref


# ----------------------------------------------------------------------------- part A
def part_a():
    rng = np.random.default_rng(SEED)
    rows = []
    fig_payload = None
    for mid in MODELS:
        meta = json.loads((SYN / f"{mid}.json").read_text(encoding="utf-8"))
        D = np.load(SYN / f"{mid}.npy").astype(np.float64)
        bg = np.load(SYN / f"{mid}_bg.npy").astype(np.float64)
        dt, f = meta["dt_ns"], meta["center_freq_hz"]
        fc = f / 1e9
        v, d_top, h = meta["velocity_m_per_ns"], meta["pipe_top_depth_m"], meta["air_gap_m"]
        j_a = int(meta["apex"]["trace_index"])
        t_apex_json = float(meta["apex"]["envelope_peak_ns"])
        pipe_tr = D[:, j_a]
        bg_tr = bg[:, min(1, bg.shape[1] - 1)]
        amp_ref = np.abs(pipe_tr - bg_tr).max()

        for cover in (0.4, 1.0):
            n_band = int(round(cover * N_TR))
            i0 = (N_TR - n_band) // 2
            i1 = i0 + n_band - 1
            S = np.repeat(bg_tr[:, None], N_TR, axis=1)
            S[:, i0:i1 + 1] = pipe_tr[:, None]
            S = S + rng.normal(0.0, NOISE_REL * amp_ref, S.shape)
            A = preprocess(S, dt, f)

            # zero from the mean trace before background removal (spec 4-1/4-2)
            events, e_mean = zero_events(A, dt, f)
            t_z, k_z, zrule = select_zero(events, h)
            t_z_on = onset_time(e_mean, k_z, dt)

            # emulated detection box hugging the visible band
            k_box0 = int((t_apex_json - 0.4 / fc) / dt)
            k_box1 = int((t_apex_json + 0.8 / fc) / dt)
            k_mute = int((t_z + 1.0 / fc) / dt)
            k_lo = max(k_box0 - int(1.0 / fc / dt), k_mute)
            k_hi = min(k_box1 + int(0.5 / fc / dt), A.shape[0] - 1)
            cols = np.arange(i0, i1 + 1)

            variants = {"none": A}
            variants["median_all"] = A - np.median(A, axis=1, keepdims=True)
            margin = 5
            outside = np.r_[0:max(i0 - margin, 0), min(i1 + margin + 1, N_TR):N_TR]
            if len(outside) >= 30:
                variants["masked_median"] = A - np.median(A[:, outside], axis=1, keepdims=True)
            else:
                variants["masked_median(fallback none)"] = A

            amp_none = None
            for vname, R in variants.items():
                t_pk, t_on, amp, _ = band_picks(R, dt, cols, k_lo, k_hi, k_box0, k_box1)
                med_amp = float(np.median(amp)) if len(amp) else 0.0
                if vname == "none":
                    amp_none = med_amp
                n_ok = len(t_pk)
                if n_ok:
                    tb = float(np.median(t_pk))
                    tb_on = float(np.median(t_on))
                    mad = float(np.median(np.abs(t_pk - tb)))
                    d_pk = v * (tb - t_z) / 2
                    d_mix = v * (tb_on - t_z) / 2  # onset band, peak zero (forbidden mix)
                    d_on = v * (tb_on - t_z_on) / 2  # onset both
                    d_apexrule = v * (t_apex_json - t_z) / 2
                else:
                    tb = tb_on = mad = d_pk = d_mix = d_on = d_apexrule = float("nan")
                rows.append({
                    "model": mid, "eps_r": meta["eps_r"], "f_mhz": f / 1e6, "air_gap_m": h,
                    "material": meta["pipe_material"], "R_m": meta["pipe_radius_m"], "d_top_m": d_top,
                    "coverage": cover, "bg": vname, "zero_rule": zrule, "t_zero_ns": round(t_z, 4),
                    "n_pick_frac": round(n_ok / n_band, 3),
                    "amp_ratio_vs_none": round(med_amp / amp_none, 3) if amp_none else "",
                    "t_band_ns": round(tb, 4), "t_apex_json_ns": round(t_apex_json, 4),
                    "t_band_minus_apex_ns": round(tb - t_apex_json, 4), "pick_mad_ns": round(mad, 4),
                    "d_band_m": round(d_pk, 4), "d_err_cm": round((d_pk - d_top) * 100, 2),
                    "d_minus_apexrule_cm": round((d_pk - d_apexrule) * 100, 2),
                    "d_onset_mix_err_cm": round((d_mix - d_top) * 100, 2),
                    "d_onset_both_err_cm": round((d_on - d_top) * 100, 2),
                })
            if mid == "M02" and cover == 1.0:
                fig_payload = dict(dt=dt, variants={k: envelope(vv) for k, vv in variants.items()},
                                   t_apex=t_apex_json, t_z=t_z)
            if mid == "M02" and cover == 0.4:
                fig_payload_04 = dict(dt=dt, variants={k: envelope(vv) for k, vv in variants.items()},
                                      t_apex=t_apex_json, t_z=t_z)
    return rows, fig_payload_04, fig_payload


# ----------------------------------------------------------------------------- part B
def part_b():
    rows = []
    # lateral offset of the survey line from the pipe axis
    for d in (0.3, 0.6, 1.0, 1.5):
        for R in (0.025, 0.1):
            for y in (0.05, 0.1, 0.2, 0.3, 0.5):
                d_app = np.sqrt((d + R) ** 2 + y ** 2) - R
                rows.append({"table": "lateral_offset", "d_top_m": d, "R_m": R, "y_m": y,
                             "d_apparent_m": round(d_app, 4), "bias_cm": round((d_app - d) * 100, 2)})
    # oblique crossing: curvature fit with straight ray gives v' = v / sin(theta)
    for th in (90, 80, 70, 60, 45, 30):
        s = np.sin(np.radians(th))
        rows.append({"table": "oblique_curvature", "theta_deg": th, "v_ratio": round(1 / s, 4),
                     "d_ratio": round(1 / s, 4)})
    # band-likeness: max time sag over half length a stays below 0.5/f_c (500 MHz -> 1.0 ns)
    v = 0.1
    for f_mhz in (400, 500):
        thr = 0.5 / (f_mhz / 1000)
        for d in (0.3, 0.6, 1.0):
            for a in (0.5, 1.0, 2.0):
                # 2/v (sqrt(d^2 + a^2 s^2) - d) = thr  ->  s
                q = (thr * v / 2 + d) ** 2 - d ** 2
                s = min(np.sqrt(q) / a, 1.0)
                rows.append({"table": "band_threshold", "f_mhz": f_mhz, "sag_ns": thr, "v": v,
                             "d_top_m": d, "half_length_m": a,
                             "theta_max_deg": round(float(np.degrees(np.arcsin(s))), 2)})
    # pipe grade along the line: zero-offset time gives d*cos(alpha) (dipping reflector)
    for g in (0.01, 0.02, 0.05, 0.10, 0.20):
        al = np.arctan(g)
        rows.append({"table": "grade", "grade": g, "alpha_deg": round(float(np.degrees(al)), 3),
                     "cos_alpha": round(float(np.cos(al)), 5),
                     "bias_at_1m_cm": round((np.cos(al) - 1) * 100, 3),
                     "time_slope_ns_per_m_v0.1": round(2 * np.sin(al) / 0.1, 3)})
    return rows


# ----------------------------------------------------------------------------- part C
def part_c():
    """Quadratic fit of band-top times (straight ray, R fixed) -> crossing angle."""
    rng = np.random.default_rng(SEED)
    rows = []
    v, R = 0.1, 0.05
    for d in (0.6, 1.0):
        for th in (3, 5, 10, 15, 20, 30):
            s = np.sin(np.radians(th))
            x = np.arange(-1.0, 1.0 + 1e-9, 0.02)
            tau = (2 / v) * (np.sqrt((d + R) ** 2 + (x * s) ** 2) - R)
            tau_n = tau + rng.normal(0, 0.05, x.size)  # 0.05 ns pick noise
            for half in (0.5, 1.0):
                m = np.abs(x) <= half
                p2, p1, p0 = np.polyfit(x[m], tau_n[m], 2)
                cov = np.polyfit(x[m], tau_n[m], 2, cov=True)[1]
                sig_c = float(np.sqrt(cov[0, 0]))
                tau0 = p0 - p1 ** 2 / (4 * p2) if p2 > 0 else float(np.min(tau_n[m]))
                # tau ~ tau0 + c x^2 with c = 2 s^2 (d+R)... use exact small-x: c = 2 s^2/(v^2 (tau0 + 2R/v))
                tau_c = tau0 + 2 * R / v  # centre-referenced time
                s_est = np.sqrt(max(p2, 0) * v ** 2 * tau_c / 2)
                th_est = float(np.degrees(np.arcsin(min(s_est, 1.0))))
                d_vertex = v * tau0 / 2
                d_median = v * float(np.median(tau_n[m])) / 2
                rows.append({"d_top_m": d, "R_m": R, "theta_deg": th, "half_window_m": half,
                             "c_ns_per_m2": round(p2, 4), "c_over_sigma": round(p2 / sig_c, 2),
                             "theta_est_deg": round(th_est, 2),
                             "d_vertex_err_cm": round((d_vertex - d) * 100, 2),
                             "d_median_err_cm": round((d_median - d) * 100, 2)})
    return rows


def write_csv(path: Path, rows: list[dict]):
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def make_figure(p04, p10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharey=True)
    for r, (p, lab) in enumerate(((p04, "band 40 % of traces"), (p10, "band 100 % of traces"))):
        dt = p["dt"]
        for c, (name, E) in enumerate(p["variants"].items()):
            ax = axes[r, c]
            t_max = p["t_apex"] + 6
            k_max = int(t_max / dt)
            img = E[:k_max] / E[:k_max][int((p["t_z"] + 2.5) / dt):].max()
            ax.imshow(img, aspect="auto", cmap="Greys", vmin=0, vmax=1,
                      extent=[0, E.shape[1], k_max * dt, 0])
            ax.axhline(p["t_apex"], color="#c4561d", lw=0.9, ls="--")
            ax.set_title(f"M02, {lab}\nbackground: {name}", fontsize=9)
            ax.set_xlabel("trace")
            if c == 0:
                ax.set_ylabel("time (ns, simulation clock)")
    fig.suptitle("Envelope after background removal (dashed: true band-top envelope peak)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "pipe_band_check.png", dpi=120)


def main():
    rows_a, p04, p10 = part_a()
    write_csv(OUT / "pipe_band_check_synthetic.csv", rows_a)
    rows_b = part_b()
    write_csv(OUT / "pipe_band_check_geometry.csv", rows_b)
    rows_c = part_c()
    write_csv(OUT / "pipe_band_check_oblique.csv", rows_c)
    make_figure(p04, p10)

    print("== Part A (pseudo-band, seed 0) ==")
    hdr = ["model", "air_gap_m", "coverage", "bg", "zero_rule", "n_pick_frac", "amp_ratio_vs_none",
           "t_band_minus_apex_ns", "pick_mad_ns", "d_err_cm", "d_minus_apexrule_cm",
           "d_onset_mix_err_cm", "d_onset_both_err_cm"]
    print(" | ".join(hdr))
    for r in rows_a:
        print(" | ".join(str(r[k]) for k in hdr))
    print("\n== Part C (oblique) ==")
    for r in rows_c:
        print(r)


if __name__ == "__main__":
    main()
