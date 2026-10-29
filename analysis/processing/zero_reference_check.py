"""Time-zero reference check on gprMax synthetics M02 (ground-coupled) and M20 (air gap 0.3 m).

Questions answered with numbers:
  1. Which time-zero (source / direct-wave / surface-reflection / first-break) makes
     d = v * (t_apex - t_zero) / 2 match the true pipe-top depth, for h = 0 and h > 0?
  2. Can the zero events be found blind from the data (mean trace, before background removal)?
  3. Does curvature-based velocity work for h = 0 and fail for h > 0, and does a
     refraction (air-gap) model fix it?

Single process, light computation (CPU budget: a few seconds).
Run:  conda run -n gpr-dev python -I analysis/processing/zero_reference_check.py
Outputs: docs/processing/zero_reference_check.csv, zero_reference_curvature.csv,
         zero_reference_airgap_models.csv, zero_reference_check.png
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, hilbert
from sklearn.linear_model import RANSACRegressor, LinearRegression

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data" / "synthetic_gprmax"
OUT = ROOT / "docs" / "processing"
sys.path.insert(0, str(ROOT / "scripts" / "synthetic"))
from ray_geometry import reflection_time, surface_reflection_time  # noqa: E402  (read-only reuse)

C0 = 0.299792458  # m/ns
SEED = 0
H_MAX_M = 0.5  # assumed max antenna height when searching the surface reflection
MODELS = ["M02", "M20"]


# ----------------------------------------------------------------------------- helpers
def envelope(a: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.abs(hilbert(a, axis=axis))


def parabolic_peak(y: np.ndarray, k: int, dt: float) -> tuple[float, float]:
    """Sub-sample peak time (ns) and value around integer index k."""
    if 0 < k < len(y) - 1:
        y0, y1, y2 = y[k - 1], y[k], y[k + 1]
        den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        return (k + delta) * dt, float(y1 - 0.25 * (y0 - y2) * delta)
    return k * dt, float(y[k])


def blind_zero_events(D: np.ndarray, dt: float, f_hz: float) -> dict:
    """Find direct-wave and late (surface-reflection) events on the raw mean trace."""
    m = D.mean(axis=1)
    m = m - m[: max(int(0.5 / dt), 1)].mean()  # DC from the pre-trigger part
    env = envelope(m)
    t_end = int(10.0 / dt)
    e = env[:t_end]
    pk, _ = find_peaks(e, height=0.1 * e.max(), prominence=0.05 * e.max())
    times = [parabolic_peak(e, int(k), dt) for k in pk]
    t_first = times[0][0]
    t_late_max = t_first + 2 * H_MAX_M / C0 + 0.5e9 / f_hz
    late = [t for t in times if t[0] <= t_late_max]
    t_late = late[-1][0]
    # 4TU-like rule: sample of max |mean amplitude| in the early window
    t_absmax = float(np.argmax(np.abs(m[:t_end]))) * dt
    # first break: first sample where envelope exceeds 5 % of the early max
    t_fb = float(np.argmax(e > 0.05 * e.max())) * dt
    return {
        "peaks_ns": [round(t, 4) for t, _ in times],
        "peaks_rel_amp": [round(a / e.max(), 3) for _, a in times],
        "direct_ns": t_first,
        "late_ns": t_late,
        "absmax_mean_ns": t_absmax,
        "first_break_ns": t_fb,
    }


def apex_pick(R: np.ndarray, dt: float, t_mute: float) -> tuple[int, float, float]:
    """Apex = earliest FIRST envelope peak (>= 0.3 of the section maximum) after the mute time.

    Using the first qualifying peak, not the per-trace global maximum, keeps the pipe-top
    reflection when a later event (water-filled pipe bottom, internal multiple) is stronger.
    """
    env = envelope(R)
    k0 = int(t_mute / dt)
    ref = env[k0:].max()
    k_first = np.full(R.shape[1], np.iinfo(np.int64).max)
    for j in range(R.shape[1]):
        pk, _ = find_peaks(env[k0:, j], height=0.3 * ref)
        if len(pk):
            k_first[j] = pk[0] + k0
    j = int(np.argmin(k_first))
    t, a = parabolic_peak(env[:, j], int(k_first[j]), dt)
    return j, t, a


def track_picks(R: np.ndarray, dt: float, j0: int, t0: float, x: np.ndarray, half_aperture: float):
    env = envelope(R)
    a0 = parabolic_peak(env[:, j0], int(round(t0 / dt)), dt)[1]
    picks = {j0: t0}
    for step in (-1, 1):
        prev = t0
        j = j0 + step
        while 0 <= j < R.shape[1] and abs(x[j] - x[j0]) <= half_aperture:
            lo, hi = int((prev - 0.3) / dt), int((prev + 1.2) / dt)
            k = lo + int(np.argmax(env[lo:hi, j]))
            t, a = parabolic_peak(env[:, j], k, dt)
            if a < 0.2 * a0:
                break
            picks[j] = t
            prev = t
            j += step
    idx = np.array(sorted(picks))
    return idx, np.array([picks[i] for i in idx])


def fit_hyperbola(x: np.ndarray, t: np.ndarray, radius: float, v_init: float):
    """RANSAC on (t/2)^2 = a + b x + c x^2, then robust least_squares on (x0, t0, v)."""
    y = (t / 2) ** 2
    X = np.column_stack([x, x**2])
    thr = float(np.median(t) / 2 * 0.3)  # ~0.3 ns in t
    rs = RANSACRegressor(LinearRegression(), residual_threshold=thr, random_state=SEED)
    rs.fit(X, y)
    a = rs.estimator_.intercept_
    b, c = rs.estimator_.coef_
    if c > 0:
        v_r = 1 / np.sqrt(c)
        x0_r = -b / (2 * c)
        t0_r = 2 * np.sqrt(max(a - x0_r**2 * c, 1e-6))
    else:
        v_r, x0_r, t0_r = v_init, float(x[np.argmin(t)]), float(t.min())

    def model(p, xx):
        x0, t0, v = p
        d = v * t0 / 2
        return (2 / v) * (np.sqrt((d + radius) ** 2 + (xx - x0) ** 2) - radius)

    res = least_squares(
        lambda p: model(p, x) - t,
        x0=[x0_r, t0_r, v_r],
        loss="soft_l1",
        f_scale=0.1,
        bounds=([x.min(), 0.0, 0.03], [x.max(), 200.0, C0]),
    )
    x0, t0, v = res.x
    rms = float(np.sqrt(np.mean(res.fun**2)))
    return {"x0": x0, "t0": t0, "v": v, "d_top": v * t0 / 2, "rms_ns": rms,
            "ransac": {"x0": x0_r, "t0": t0_r, "v": v_r}, "inliers": int(rs.inlier_mask_.sum())}


def refraction_grid(x, t_rel, x0, h, s, radius, t_apex_rel, eps_grid):
    """For each eps_r: solve d_top so the apex matches, then RMS of all picks.

    Model time relative to the surface-reflection zero:
      t_rel(x) = T_refl(x; h, d, v, R, s) - T_surf(h, s)
    """
    t_surf = surface_reflection_time(s, h)
    out = []
    for eps in eps_grid:
        v = C0 / np.sqrt(eps)

        def apex_time(d):
            tt, _ = reflection_time(x0 - s / 2, x0 + s / 2, h, x0, -(d + radius), radius, v, n_theta=721)
            return tt - t_surf

        lo, hi = 0.01, 5.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if apex_time(mid) > t_apex_rel:
                hi = mid
            else:
                lo = mid
        d = 0.5 * (lo + hi)
        pred = np.array([
            reflection_time(xi - s / 2, xi + s / 2, h, x0, -(d + radius), radius, v, n_theta=721)[0]
            for xi in x
        ]) - t_surf
        out.append((eps, v, d, float(np.sqrt(np.mean((pred - t_rel) ** 2)))))
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, curv_rows, fig_data = [], [], {}

    for mid in MODELS:
        meta = json.loads((SYN / f"{mid}.json").read_text(encoding="utf-8"))
        D = np.load(SYN / f"{mid}.npy").astype(np.float64)
        bg = np.load(SYN / f"{mid}_bg.npy").astype(np.float64)
        dt, dx = meta["dt_ns"], meta["dx_m"]
        v_true, d_top, R = meta["velocity_m_per_ns"], meta["pipe_top_depth_m"], meta["pipe_radius_m"]
        h, s, f = meta["air_gap_m"], meta["antenna_offset_m"], meta["center_freq_hz"]
        x = meta["trace_x_m_first"] + np.arange(D.shape[1]) * dx
        tz = meta["time_zero_candidates_ns"]

        # --- zero events, blind, raw data (before background removal)
        ev = blind_zero_events(D, dt, f)
        # reference surface reflection from (no-pipe - free-space), only if fs exists
        tz_surf_fs = None
        if (SYN / f"{mid}_fs.npy").exists():
            fs = np.load(SYN / f"{mid}_fs.npy").astype(np.float64)[:, 0]
            e = envelope(bg[:, 1] - fs)
            k = int(np.argmax(e[: int(10 / dt)]))
            tz_surf_fs = parabolic_peak(e, k, dt)[0]

        # --- apex time with three background-removal variants
        t_mute = ev["late_ns"] + 1.0e9 / f
        apex = {}
        for name, Rb in (
            ("ideal_bg", D - bg[:, [1]]),
            ("mean_trace", D - D.mean(axis=1, keepdims=True)),
            ("median_trace", D - np.median(D, axis=1, keepdims=True)),
        ):
            j, t_a, _ = apex_pick(Rb, dt, t_mute)
            apex[name] = (j, t_a)
        j_a, t_apex = apex["ideal_bg"]

        zeros = {
            "source_peak(json, ideal)": tz["source_peak"],
            "direct_wave(blind first peak)": ev["direct_ns"],
            "late_event(blind, = surface refl. if h>0)": ev["late_ns"],
            "absmax_mean_trace(4TU t0_sample_est rule)": ev["absmax_mean_ns"],
            "first_break_5pct": ev["first_break_ns"],
        }
        if tz_surf_fs is not None:
            zeros["surface_refl(bg-fs reference)"] = tz_surf_fs
        for zname, z in zeros.items():
            for aname, (j, ta) in apex.items():
                t_rel = ta - z
                d = v_true * t_rel / 2
                d_off = float(np.sqrt(max(d**2 - (s / 2) ** 2, 0.0))) if h == 0 else d
                rows.append({
                    "model": mid, "air_gap_m": h, "zero": zname, "zero_ns": round(z, 4),
                    "apex_bg": aname, "apex_trace": j, "apex_ns": round(ta, 4),
                    "t_rel_ns": round(t_rel, 4), "t0_top_json_ns": round(meta["t0_top_ns"], 4),
                    "t_rel_minus_t0top_ns": round(t_rel - meta["t0_top_ns"], 4),
                    "d_m": round(d, 4), "d_err_cm": round((d - d_top) * 100, 2),
                    "d_offsetcorr_m": round(d_off, 4), "d_offsetcorr_err_cm": round((d_off - d_top) * 100, 2),
                })

        # --- curvature fits (ideal background, |x - x_apex| <= 1.0 m)
        Rb = D - bg[:, [1]]
        idx, tp = track_picks(Rb, dt, j_a, t_apex, x, 1.0)
        xp = x[idx]
        zero_used = ev["late_ns"] if h > 0 else ev["direct_ns"]
        zname = "late_event(surface refl.)" if h > 0 else "direct_wave"
        t_rel = tp - zero_used
        for radius in (0.0, R):
            fr = fit_hyperbola(xp, t_rel, radius, v_true)
            curv_rows.append({
                "model": mid, "air_gap_m": h, "zero": zname, "fit": f"straight-ray R={radius:g} free(x0,t0,v)",
                "n_picks": len(xp), "v_fit": round(fr["v"], 5), "v_ratio": round(fr["v"] / v_true, 4),
                "eps_fit": round((C0 / fr["v"]) ** 2, 3), "d_top_m": round(fr["d_top"], 4),
                "d_err_cm": round((fr["d_top"] - d_top) * 100, 2), "rms_ns": round(fr["rms_ns"], 4),
                "x0_m": round(fr["x0"], 4),
            })
        # fixed v (meta eps) -> depth from apex only
        d_fix = v_true * (t_apex - zero_used) / 2
        curv_rows.append({
            "model": mid, "air_gap_m": h, "zero": zname, "fit": "v fixed (meta eps), apex only",
            "n_picks": 1, "v_fit": round(v_true, 5), "v_ratio": 1.0, "eps_fit": meta["eps_r"],
            "d_top_m": round(d_fix, 4), "d_err_cm": round((d_fix - d_top) * 100, 2), "rms_ns": "",
            "x0_m": round(x[j_a], 4),
        })
        if h > 0:
            # antenna height from the direct/late gap (offset s assumed known = 0.1 m)
            gap = ev["late_ns"] - ev["direct_ns"]
            h_est = float(np.sqrt(((gap + s / C0) * C0 / 2) ** 2 - (s / 2) ** 2))
            sub = slice(None, None, 2)  # every 2nd pick keeps the cost small
            grid = refraction_grid(xp[sub], t_rel[sub], x[j_a], h_est, s, R, t_apex - zero_used,
                                   np.arange(6.0, 14.01, 0.25))
            eps_b, v_b, d_b, rms_b = min(grid, key=lambda r: r[3])
            curv_rows.append({
                "model": mid, "air_gap_m": h, "zero": zname,
                "fit": f"refraction grid eps 6-14 (h_est={h_est:.3f} m, R={R:g}, s={s:g})",
                "n_picks": len(xp[sub]), "v_fit": round(v_b, 5), "v_ratio": round(v_b / v_true, 4),
                "eps_fit": eps_b, "d_top_m": round(d_b, 4), "d_err_cm": round((d_b - d_top) * 100, 2),
                "rms_ns": round(rms_b, 4), "x0_m": round(x[j_a], 4),
            })
            print(f"[{mid}] h_est from direct/late gap {gap:.3f} ns -> {h_est:.3f} m (true {h})")

        print(f"[{mid}] blind early peaks (ns): {ev['peaks_ns']} rel.amp {ev['peaks_rel_amp']}")
        print(f"[{mid}] json zeros: {tz}; bg-fs surface ref: {tz_surf_fs}")
        fig_data[mid] = dict(meta=meta, D=D, bg=bg, ev=ev, zeros=zeros, t_apex=t_apex, j=j_a,
                             tz_surf_fs=tz_surf_fs)

    # --- extra (light): blind zero events on every air-gap model, apex depth with v = meta
    print("\n== air-gap models: blind late event vs json surface reflection ==")
    print("| model | h (m) | f (MHz) | blind peaks (ns) | late (ns) | json surf (ns) | late-json (ns) | h_est (m) | d late (m) | err (cm) | d json-surf err (cm) | apex trace |")
    extra_rows = []
    for mid in ["M12", "M13", "M15", "M18", "M20"]:
        meta = json.loads((SYN / f"{mid}.json").read_text(encoding="utf-8"))
        D = np.load(SYN / f"{mid}.npy").astype(np.float64)
        bg = np.load(SYN / f"{mid}_bg.npy").astype(np.float64)
        dt, f, s = meta["dt_ns"], meta["center_freq_hz"], meta["antenna_offset_m"]
        v, d_top = meta["velocity_m_per_ns"], meta["pipe_top_depth_m"]
        ev = blind_zero_events(D, dt, f)
        tzs = meta["time_zero_candidates_ns"]["surface_reflection_peak"]
        j, ta, _ = apex_pick(D - np.median(D, axis=1, keepdims=True), dt, ev["late_ns"] + 1.0e9 / f)
        n_ev = len(ev["peaks_ns"])
        gap = ev["late_ns"] - ev["direct_ns"]
        h_est = float(np.sqrt(((gap + s / C0) * C0 / 2) ** 2 - (s / 2) ** 2)) if n_ev > 1 else float("nan")
        d_late = v * (ta - ev["late_ns"]) / 2
        d_js = v * (ta - tzs) / 2
        r = dict(model=mid, h=meta["air_gap_m"], f_mhz=f / 1e6, peaks=ev["peaks_ns"], late=round(ev["late_ns"], 4),
                 json_surf=round(tzs, 4), late_minus_json=round(ev["late_ns"] - tzs, 4), h_est=round(h_est, 3),
                 d_late=round(d_late, 4), err_cm=round((d_late - d_top) * 100, 2),
                 err_json_cm=round((d_js - d_top) * 100, 2), apex_trace=j)
        extra_rows.append(r)
        print("| " + " | ".join(str(val) for val in r.values()) + " |")
    with open(OUT / "zero_reference_airgap_models.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(extra_rows[0]))
        w.writeheader()
        w.writerows(extra_rows)

    # --- write tables
    with open(OUT / "zero_reference_check.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "zero_reference_curvature.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curv_rows[0]))
        w.writeheader()
        w.writerows(curv_rows)

    print("\n== apex depth vs zero (v = meta, apex bg = ideal_bg) ==")
    print("| model | zero | zero (ns) | t_apex-zero (ns) | t0_top json (ns) | diff (ns) | d (m) | err (cm) | offset-corr err (cm) |")
    for r in rows:
        if r["apex_bg"] == "ideal_bg":
            print(f"| {r['model']} | {r['zero']} | {r['zero_ns']} | {r['t_rel_ns']} | {r['t0_top_json_ns']} | "
                  f"{r['t_rel_minus_t0top_ns']} | {r['d_m']} | {r['d_err_cm']} | {r['d_offsetcorr_err_cm']} |")
    print("\n== apex time by background-removal variant ==")
    for r in rows:
        if r["zero"].startswith("source"):
            print(f"| {r['model']} | {r['apex_bg']} | trace {r['apex_trace']} | {r['apex_ns']} ns |")
    print("\n== curvature fits ==")
    for r in curv_rows:
        print(" | ".join(str(v) for v in r.values()))

    # --- figure: apex trace envelopes with zero markers
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"source": "#6b6b6b", "direct": "#1f6fb2", "late": "#c4561d", "first": "#7a52a8", "fs": "#2a8a4a"}
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for ax, mid in zip(axes, MODELS):
        g = fig_data[mid]
        dt = g["meta"]["dt_ns"]
        t = np.arange(g["D"].shape[0]) * dt
        raw = g["D"][:, g["j"]]
        pipe = raw - g["bg"][:, 1]
        e_raw = envelope(raw)
        e_pipe = envelope(pipe)
        ax.plot(t, e_raw / e_raw.max(), color="#9a9a9a", lw=1.2, label="raw apex trace envelope")
        ax.plot(t, e_pipe / e_pipe.max(), color="#222222", lw=1.6, label="pipe-only envelope (raw - no-pipe)")
        marks = [
            ("source", g["meta"]["time_zero_candidates_ns"]["source_peak"], "source peak"),
            ("first", g["ev"]["first_break_ns"], "first break"),
            ("direct", g["ev"]["direct_ns"], "direct wave"),
            ("late", g["ev"]["late_ns"], "late event"),
        ]
        if g["tz_surf_fs"] is not None:
            marks.append(("fs", g["tz_surf_fs"], "surface refl. (no-pipe - free-space)"))
        for i, (key, tm, lab) in enumerate(marks):
            ax.axvline(tm, color=colors[key], lw=1.4, ls="--")
            ax.text(tm + 0.1, 0.92 - 0.12 * i, f"{lab} {tm:.2f}", color="#222222", fontsize=8)
        ax.axvline(g["t_apex"], color="#222222", lw=1.0, ls=":")
        ax.text(g["t_apex"] + 0.1, 0.5, f"apex {g['t_apex']:.2f} ns", fontsize=8)
        ax.set_title(f"{mid}: h = {g['meta']['air_gap_m']} m, eps_r {g['meta']['eps_r']}, "
                     f"d_top {g['meta']['pipe_top_depth_m']} m, trace {g['j']}", fontsize=10)
        ax.set_ylabel("normalised envelope")
        ax.set_xlim(0, 22)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), fontsize=8, frameon=False)
    axes[-1].set_xlabel("time (ns, simulation clock)")
    fig.tight_layout()
    fig.savefig(OUT / "zero_reference_check.png", dpi=130)
    print(f"\nwrote {OUT / 'zero_reference_check.csv'}, {OUT / 'zero_reference_curvature.csv'}, "
          f"{OUT / 'zero_reference_check.png'}")


if __name__ == "__main__":
    main()
