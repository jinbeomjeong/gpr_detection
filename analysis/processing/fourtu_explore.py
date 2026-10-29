"""4TU field exploration (checklist M2-7) under the rules of docs/processing/spec_depth_v1.md.

Activities: 01.1, 04.1, 010.14, 011.14 (C:\\workspace\\GPR_Data is read-only).

Questions answered with numbers:
  1. Zero events per line: are direct wave and surface reflection separate envelope peaks
     (spec 4-1)?  Fraction of `air_single_event`.  Air gap h from the event gap.
     Extra tests: (a) per-trace lobe timing (are the meta t0 samples two lobes of ONE wavelet?),
     (b) surface-tracking shift test: align traces by k * s_j (s_j = per-trace shift of the
     zero band) and measure stack coherence for k = 0 (stationary: direct coupling),
     k = 1 (moves with the surface: surface reflection / sub-surface), k = 2 (surface multiple).
  2. Label `t0_sample_est` and `t0_sample_act_median` (direct / surface / lobe of merged band).
     Whole-index (759 lines, CSV only) distribution of est - act.
  3. Semi-automatic apexes (hand seeds below) -> refraction-model QC: eps_r(h) for h grid,
     compare with eps_r_used.  Straight-ray (h = 0) eps is reported for comparison only.
  4. Apex depth (meta v, zero = band envelope peak) vs ground-truth depth set,
     top and centre hypotheses (exploration, not a verdict).

Single process, light computation.  No random numbers are used (deterministic).
Run:  conda run -n gpr-dev python -I analysis/processing/fourtu_explore.py
Outputs (docs/processing/): fourtu_explore_zero.csv, fourtu_explore_lobes.csv,
  fourtu_explore_shift.csv, fourtu_explore_meta_t0_all.csv, fourtu_explore_apex.csv,
  fourtu_explore_eps.csv, fourtu_explore_depth_match.csv, fourtu_explore_*.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015")
NPY = BASE / "05_radargrams_npy"
META = BASE / "01_메타데이터"
OUT = ROOT / "docs" / "processing"

ACTS = ["01.1", "04.1", "010.14", "011.14"]
C0 = 0.299792458  # m/ns
FC_HZ = 500e6
H_MAX_M = 0.5  # spec 4-1 assumption
W_BG = 301  # spec 3 (4)
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2 = "#0b0b0b", "#52514e"

# Hand-chosen apex seeds (trace index, sample index) read off fourtu_explore_bscans_<act>.png.
# Filled after the first quick-look pass; the refinement below snaps to the local envelope max.
SEEDS: dict[tuple[str, str], list[tuple[int, int]]] = {
    ("01.1", "Path1"): [(105, 175), (160, 125), (15, 285)],
    ("01.1", "Path2"): [(70, 180), (140, 150)],
    ("01.1", "Path3"): [(190, 145), (95, 190)],
    ("01.1", "Path4"): [(200, 150), (85, 255)],
    ("01.1", "Path5"): [(225, 130), (100, 255), (185, 110)],
    ("01.1", "Path6"): [(245, 145), (95, 245)],
    ("01.1", "Path7"): [(85, 180), (280, 140), (175, 75)],
    ("01.1", "Path8"): [(85, 175), (180, 80)],
    ("01.1", "Path9"): [(90, 170)],
    ("01.1", "Path10"): [(85, 160)],
    ("01.1", "Path11"): [(105, 150), (185, 85)],
    ("01.1", "Path12"): [(70, 165)],
    ("04.1", "Path1"): [(85, 245)],  # low confidence: gentle hump, layered section
    ("04.1", "Path3"): [(80, 228)],  # low confidence
    ("010.14", "Path2"): [(175, 135), (470, 165), (585, 170), (330, 190)],
    ("010.14", "Path3"): [(565, 185)],
    ("010.14", "Path4"): [(440, 140), (605, 150)],
    ("011.14", "Profile_6161"): [(165, 120), (200, 105)],
    ("011.14", "Profile_6163"): [(200, 110), (380, 125)],
    ("011.14", "Profile_6164"): [(260, 115), (140, 95)],
}
H_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)  # m, antenna height hypotheses for the refraction QC


# ----------------------------------------------------------------------------- io / preprocessing
def line_names(act: str) -> list[str]:
    names = [p.stem for p in (NPY / act).glob("*.npy")]
    return sorted(names, key=lambda s: (len(s), s))


def load(act: str, name: str):
    meta = json.loads((NPY / act / f"{name}.json").read_text(encoding="utf-8"))
    D = np.load(NPY / act / f"{name}.npy").astype(np.float64)
    return D, meta


def prep(D: np.ndarray, dt: float, fc: float = FC_HZ) -> np.ndarray:
    """spec 3 steps (1)-(3): DC, dewow (centred 3/fc window), zero-phase Butterworth 0.25fc-2fc."""
    X = D - D.mean(axis=0, keepdims=True)
    w = int(round(3.0 / (fc * 1e-9) / dt)) | 1
    X = X - uniform_filter1d(X, w, axis=0, mode="nearest")
    fs = 1.0 / (dt * 1e-9)
    hi = min(2 * fc, 0.45 * fs)
    sos = butter(4, [0.25 * fc, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, X, axis=0)


def median_bg(X: np.ndarray, window: int = W_BG) -> np.ndarray:
    """Moving median-trace background removal (spec 3 (4)); whole-line median if shorter."""
    n = X.shape[1]
    if n <= window:
        return X - np.median(X, axis=1, keepdims=True)
    out = np.empty_like(X)
    half = window // 2
    step = 25  # median recomputed every 25 traces (cost control; bg varies slowly)
    for j0 in range(0, n, step):
        c = min(max(j0 + step // 2, half), n - half - 1)
        bg = np.median(X[:, c - half:c + half + 1], axis=1)
        out[:, j0:j0 + step] = X[:, j0:j0 + step] - bg[:, None]
    return out


def envelope(a: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.abs(hilbert(a, axis=axis))


def parabolic(y: np.ndarray, k: int) -> tuple[float, float]:
    if 0 < k < len(y) - 1:
        y0, y1, y2 = y[k - 1], y[k], y[k + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        d = float(np.clip(d, -0.5, 0.5))
        return k + d, float(y1 - 0.25 * (y0 - y2) * d)
    return float(k), float(y[k])


def fwhm(e: np.ndarray, k: int) -> float:
    half = e[k] / 2
    lo = k
    while lo > 0 and e[lo] > half:
        lo -= 1
    hi = k
    while hi < len(e) - 1 and e[hi] > half:
        hi += 1
    return float(hi - lo)


# ----------------------------------------------------------------------------- 1. zero events
def zero_events(X: np.ndarray, dt: float) -> dict:
    """spec 4-1 on one mean trace (already window-averaged)."""
    m = X.mean(axis=1)
    e = envelope(m)
    n25 = int(0.25 * len(e))
    k_max = int(np.argmax(e[:n25]))
    t_end = k_max * dt + 2 * H_MAX_M / C0 + 0.5e9 / FC_HZ
    k_end = min(int(t_end / dt) + 1, len(e))
    ee = e[:k_end]
    pk, _ = find_peaks(ee, height=0.1 * ee.max(), prominence=0.05 * ee.max())
    ks = [parabolic(ee, int(k))[0] for k in pk]
    out = {"n_events": len(ks), "events_sample": [round(k, 2) for k in ks],
           "fwhm_ns": fwhm(e, k_max) * dt, "k_env": parabolic(e, k_max)[0]}
    if len(ks) >= 2:
        out["t_dir_sample"], out["t_surf_sample"] = ks[0], ks[-1]
        out["h_est_m"] = C0 * (ks[-1] - ks[0]) * dt / 2
        out["rule"] = "air_surface_late"
    elif len(ks) == 1:
        out["t_dir_sample"] = out["t_surf_sample"] = ks[0]
        out["h_est_m"] = float("nan")
        out["rule"] = "air_single_event"
    else:
        out["rule"] = "zero_failed"
    return out


def lobe_stats(X: np.ndarray, dt: float, k_env: float) -> list[dict]:
    """Signed lobes of the mean trace around the zero band; per-trace timing and amplitude."""
    m = X.mean(axis=1)
    lo = max(int(k_env - 3.0 / dt), 1)
    hi = min(int(k_env + 3.0 / dt), len(m) - 2)
    seg = m[lo:hi]
    ref = np.abs(seg).max()
    lobes = []
    for sgn in (1, -1):
        pk, _ = find_peaks(sgn * seg, height=0.2 * ref)
        for k in pk:
            lobes.append((lo + int(k), sgn))
    lobes.sort()
    half = max(int(round(0.5 / dt)), 2)
    rows = []
    for k_m, sgn in lobes:
        seg2 = sgn * X[k_m - half:k_m + half + 1]
        kk = np.argmax(seg2, axis=0)
        t = np.array([parabolic(seg2[:, j], int(kk[j]))[0] for j in range(X.shape[1])]) + k_m - half
        a = seg2[kk, np.arange(X.shape[1])]
        rows.append({"k_mean": k_m, "sign": "+" if sgn > 0 else "-",
                     "rel_amp_mean": round(float(sgn * m[k_m] / ref), 3),
                     "t_med_sample": float(np.median(t)), "t_std_ns": float(np.std(t) * dt),
                     "amp_cv": float(np.std(a) / max(np.mean(a), 1e-9)), "_t": t})
    return rows


def band_shift(X: np.ndarray, dt: float, k_env: float) -> np.ndarray:
    """Per-trace time shift (samples) of the zero band by cross-correlation with the mean trace."""
    w = int(round(1.5 / dt))
    lo, hi = int(k_env) - w, int(k_env) + w + 1
    ref = X[lo:hi].mean(axis=1)
    maxlag = int(round(1.0 / dt))
    s = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        tr = X[lo - maxlag:hi + maxlag, j]
        cc = np.array([np.dot(ref, tr[L:L + len(ref)]) for L in range(2 * maxlag + 1)])
        k = int(np.argmax(cc))
        s[j] = parabolic(cc, k)[0] - maxlag
    return s - np.median(s)


def shift_coherence(X: np.ndarray, s: np.ndarray, ks=(0.0, 1.0, 2.0), win: int = 9) -> dict:
    """Stack coherence C_k(t) = sum_win(stack^2) / mean_j sum_win(trace^2) after aligning by k*s_j."""
    n, N = X.shape
    t = np.arange(n, dtype=float)
    out = {}
    for k in ks:
        A = np.empty_like(X)
        for j in range(N):
            A[:, j] = np.interp(t + k * s[j], t, X[:, j])
        st = A.mean(axis=1)
        num = uniform_filter1d(st**2, win)
        den = uniform_filter1d((A**2).mean(axis=1), win) + 1e-12
        out[k] = num / den
    return out


# ----------------------------------------------------------------------------- 3. refraction QC
def leg_time_vec(dx: np.ndarray, h: float, depth: float, v: float) -> np.ndarray:
    """Fermat air->soil time (ns) from antenna at height h to a soil point at horizontal
    offset dx >= 0 and depth `depth` (bisection on the crossing point)."""
    dx = np.abs(np.asarray(dx, float))
    if h <= 1e-9:
        return np.hypot(dx, depth) / v
    lo = np.zeros_like(dx)
    hi = dx.copy()
    for _ in range(40):
        a = 0.5 * (lo + hi)
        g = a / (C0 * np.hypot(a, h)) - (dx - a) / (v * np.hypot(dx - a, depth))
        hi = np.where(g > 0, a, hi)
        lo = np.where(g > 0, lo, a)
    a = 0.5 * (lo + hi)
    return np.hypot(a, h) / C0 + np.hypot(dx - a, depth) / v


def model_tau(x: np.ndarray, x0: float, tau_a: float, eps: float, h: float, R: float) -> np.ndarray:
    """Monostatic (s = 0) top reflection of a cylinder of radius R, time relative to the
    surface reflection 2h/c.  Depth tied to the apex: d_top = v * tau_a / 2.
    Normal-incidence ray passes through the centre: t = 2 (T_leg(centre) - R / v)."""
    v = C0 / np.sqrt(eps)
    d = v * tau_a / 2
    return 2 * (leg_time_vec(x - x0, h, d + R, v) - R / v) - 2 * h / C0


def eps_scan(x, tau, x0, tau_a, h, R, grid=np.arange(2.0, 40.01, 0.25)):
    rms = np.array([np.sqrt(np.mean((model_tau(x, x0, tau_a, e, h, R) - tau) ** 2)) for e in grid])
    k = int(np.argmin(rms))
    if 0 < k < len(grid) - 1:
        r = minimize_scalar(lambda e: np.sqrt(np.mean((model_tau(x, x0, tau_a, e, h, R) - tau) ** 2)),
                            bounds=(grid[k - 1], grid[k + 1]), method="bounded")
        return float(r.x), float(r.fun), False
    return float(grid[k]), float(rms[k]), True  # at grid edge


# ----------------------------------------------------------------------------- apex picks
def refine_seed(E: np.ndarray, j: int, k: int, dj: int = 8, dk: int = 10) -> tuple[int, float]:
    sub = E[max(k - dk, 0):k + dk + 1, max(j - dj, 0):j + dj + 1]
    kk, jj = np.unravel_index(int(np.argmax(sub)), sub.shape)
    j2, k2 = max(j - dj, 0) + jj, max(k - dk, 0) + kk
    return int(j2), parabolic(E[:, j2], int(k2))[0]


def track(E: np.ndarray, j0: int, k0: float, dt: float, dx: float, a_max: float = 1.0,
          v_min: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """spec 5-3 tracking on the envelope (samples)."""
    a0 = E[int(round(k0)), j0]
    picks = {j0: k0}
    for step in (-1, 1):
        prev = k0
        j = j0 + step
        while 0 <= j < E.shape[1] and abs(j - j0) * dx <= a_max:
            lo = max(int((prev * dt - 0.3) / dt), 0)
            hi = int((prev * dt + 2 * dx / v_min + 0.3) / dt) + 1
            kk = lo + int(np.argmax(E[lo:hi, j]))
            kp, a = parabolic(E[:, j], kk)
            if a < 0.2 * a0 or (kp - prev) * dt < -0.1:
                break
            picks[j] = kp
            prev = kp
            j += step
    idx = np.array(sorted(picks))
    return idx, np.array([picks[i] for i in idx])


def local_apex(xp, tp, x_a, half=0.10):
    """spec 6-4: quadratic on |x - x_a| <= 0.10 m."""
    sel = np.abs(xp - x_a) <= half + 1e-9
    if sel.sum() >= 5:
        p2, p1, p0 = np.polyfit(xp[sel] - x_a, tp[sel], 2)
        if p2 > 0:
            return x_a - p1 / (2 * p2), p0 - p1**2 / (4 * p2)
    k = int(np.argmin(tp))
    return float(xp[k]), float(tp[k])


def straight_fit(x, tau):
    """Point-scatter straight-ray fit (x0, t0, v) -- informational only on air-coupled data."""
    from scipy.optimize import least_squares
    k = int(np.argmin(tau))
    res = least_squares(lambda p: 2 / p[2] * np.sqrt((p[2] * p[1] / 2) ** 2 + (x - p[0]) ** 2) - tau,
                        x0=[x[k], tau[k], 0.1], loss="soft_l1", f_scale=0.1,
                        bounds=([x.min(), 0.0, 0.02], [x.max(), 200.0, 0.6]))
    return float(res.x[2]), float(np.sqrt(np.mean(res.fun ** 2)))


# ----------------------------------------------------------------------------- helpers
def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def nanf(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


# ----------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gt = [r for r in read_csv(META / "utilities_groundtruth.csv") if r["activity_id"] in ACTS]
    ulist = [r for r in read_csv(META / "utilities_metadata_list.csv") if r["activity_id"] in ACTS]
    index = read_csv(META / "radargrams_index.csv")

    # ---- 2b. whole-index meta t0 relation (CSV only)
    all_rows = []
    for r in index:
        est, act, dt = int(r["t0_sample_est"]), int(r["t0_sample_act_median"]), float(r["dt_ns"])
        all_rows.append({"activity_id": r["activity_id"], "path": r["path"], "dt_ns": dt,
                         "est": est, "act": act, "est_minus_act": est - act,
                         "est_minus_act_ns": (est - act) * dt})
    write_csv(OUT / "fourtu_explore_meta_t0_all.csv", all_rows)
    diffs = np.array([r["est_minus_act_ns"] for r in all_rows])
    nz = diffs[np.abs(diffs) > 0.15]
    print(f"[index] lines {len(diffs)}, est==act (|diff|<=0.15 ns) {np.mean(np.abs(diffs) <= 0.15):.3f}; "
          f"nonzero n={len(nz)} median |diff| {np.median(np.abs(nz)):.3f} ns, "
          f"p10/p90 {np.percentile(np.abs(nz), [10, 90]).round(3)} ns")
    hist, edges = np.histogram(np.abs(nz), bins=np.arange(0, 6.01, 0.25))
    print("[index] |est-act| ns histogram (nonzero):", {f"{edges[i]:.2f}": int(h) for i, h in enumerate(hist) if h})

    zero_rows, lobe_rows, shift_rows = [], [], []
    store = {}
    for act in ACTS:
        for name in line_names(act):
            D, m = load(act, name)
            dt, dx = m["dt_ns"], m["trace_dx_m"]
            X = prep(D, dt)
            N = X.shape[1]
            # windows of W_BG traces (spec 4-1)
            nwin = max(1, int(round(N / W_BG)))
            edges_w = np.linspace(0, N, nwin + 1).astype(int)
            wins = [zero_events(X[:, a:b], dt) for a, b in zip(edges_w[:-1], edges_w[1:])]
            ev = zero_events(X, dt)
            k_env = ev["k_env"]
            lobes = lobe_stats(X, dt, k_env)
            s = band_shift(X, dt, k_env)
            coh = shift_coherence(X, s)
            sat = float(np.mean(np.abs(D) >= 32767))
            # label meta t0 against lobes / envelope
            def label(k_meta: int) -> str:
                near = min(lobes, key=lambda L: abs(L["t_med_sample"] - k_meta)) if lobes else None
                tag = "single_band" if ev["n_events"] == 1 else "multi"
                if near is not None and abs(near["t_med_sample"] - k_meta) * dt <= 0.3:
                    return f"{tag}:lobe{near['sign']}@{near['t_med_sample']:.1f}"
                return f"{tag}:unclassified"
            zr = {
                "activity": act, "line": name, "n_traces": N, "dt_ns": dt, "eps_r_used": m["eps_r_used"],
                "sat_frac": sat, "n_win": nwin,
                "n_events_line": ev["n_events"], "events_sample": " ".join(map(str, ev["events_sample"])),
                "n_single_win": sum(w["n_events"] == 1 for w in wins),
                "k_env_sample": k_env, "t_env_ns": k_env * dt, "env_fwhm_ns": ev["fwhm_ns"],
                "k_env_win_std_ns": float(np.std([w["k_env"] for w in wins]) * dt),
                "h_est_m": ev.get("h_est_m", float("nan")), "rule": ev["rule"],
                "t0_est": m["t0_sample_est"], "t0_act": m["t0_sample_act_median"],
                "t0_est_label": label(m["t0_sample_est"]), "t0_act_label": label(m["t0_sample_act_median"]),
                "t0_est_minus_env_ns": (m["t0_sample_est"] - k_env) * dt,
                "t0_act_minus_env_ns": (m["t0_sample_act_median"] - k_env) * dt,
                "band_shift_std_ns": float(np.std(s) * dt),
                "band_shift_p95_ns": float(np.percentile(np.abs(s), 95) * dt),
            }
            zero_rows.append(zr)
            for L in lobes:
                lobe_rows.append({"activity": act, "line": name, "sign": L["sign"], "k_mean": L["k_mean"],
                                  "rel_amp_mean": L["rel_amp_mean"], "t_med_sample": L["t_med_sample"],
                                  "t_med_minus_env_ns": (L["t_med_sample"] - k_env) * dt,
                                  "t_std_ns": L["t_std_ns"], "amp_cv": L["amp_cv"]})
            # lobe pair rigidity: gap std between the two strongest lobes
            if len(lobes) >= 2:
                two = sorted(lobes, key=lambda L: -abs(L["rel_amp_mean"]))[:2]
                two.sort(key=lambda L: L["k_mean"])
                gap = (two[1]["_t"] - two[0]["_t"]) * dt
                c = float(np.corrcoef(two[0]["_t"], two[1]["_t"])[0, 1])
                zr["main_lobes"] = f"{two[0]['sign']}{two[0]['t_med_sample']:.1f}/{two[1]['sign']}{two[1]['t_med_sample']:.1f}"
                zr["main_gap_ns"] = float(np.median(gap))
                zr["main_gap_std_ns"] = float(np.std(gap))
                zr["main_lobe_corr"] = c
            # shift coherence per lobe window and best k after the band
            for L in lobes:
                k = int(round(L["t_med_sample"]))
                shift_rows.append({"activity": act, "line": name, "window": f"lobe{L['sign']}@{k}",
                                   "t_rel_env_ns": (k - k_env) * dt,
                                   "C0": coh[0.0][k], "C1": coh[1.0][k], "C2": coh[2.0][k],
                                   "best_k": max((0.0, 1.0, 2.0), key=lambda q: coh[q][k])})
            # strongest k=2-dominant window between band + 1.5 ns and band + 8 ns
            lo, hi = int(k_env + 1.5 / dt), min(int(k_env + 8.0 / dt), X.shape[0] - 1)
            ratio = coh[2.0][lo:hi] / (coh[1.0][lo:hi] + 1e-12)
            score = coh[2.0][lo:hi] * (ratio > 1.0)
            kk = lo + int(np.argmax(score))
            shift_rows.append({"activity": act, "line": name, "window": f"k2max@{kk}",
                               "t_rel_env_ns": (kk - k_env) * dt,
                               "C0": coh[0.0][kk], "C1": coh[1.0][kk], "C2": coh[2.0][kk],
                               "best_k": max((0.0, 1.0, 2.0), key=lambda q: coh[q][kk])})
            store[(act, name)] = dict(X=X, D=D, m=m, ev=ev, lobes=lobes, s=s, coh=coh)
            print(f"[zero] {act:6s} {name:13s} N={N:5d} events={ev['events_sample']} fwhm={ev['fwhm_ns']:.2f} ns "
                  f"single_win {zr['n_single_win']}/{nwin} est={m['t0_sample_est']}({zr['t0_est_label']}) "
                  f"act={m['t0_sample_act_median']}({zr['t0_act_label']}) shift_std={zr['band_shift_std_ns']:.3f} ns "
                  f"lobes={[(L['sign'], round(L['t_med_sample'], 1), round(L['t_std_ns'], 3)) for L in lobes]} "
                  f"gap_std={zr.get('main_gap_std_ns', float('nan')):.3f} corr={zr.get('main_lobe_corr', float('nan')):.2f}")
    # equalise keys for csv
    keys = []
    for r in zero_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    zero_rows = [{k: r.get(k, "") for k in keys} for r in zero_rows]
    write_csv(OUT / "fourtu_explore_zero.csv", zero_rows)
    write_csv(OUT / "fourtu_explore_lobes.csv", lobe_rows)
    write_csv(OUT / "fourtu_explore_shift.csv", shift_rows)
    for r in shift_rows:
        print(f"[shift] {r['activity']:6s} {r['line']:13s} {r['window']:14s} dt_env={r['t_rel_env_ns']:+.2f} ns "
              f"C0={r['C0']:.3f} C1={r['C1']:.3f} C2={r['C2']:.3f} best k={r['best_k']}")

    # ---- figures for part 1
    fig_zero(store)
    fig_quicklook(store)

    # ---- 3/4. apexes
    if SEEDS:
        apex_part(store, gt, ulist)


# ----------------------------------------------------------------------------- figures
def fig_zero(store: dict) -> None:
    reps = [("01.1", "Path1"), ("01.1", "Path4"), ("04.1", "Path2"), ("010.14", "Path2"), ("011.14", "Profile_6161")]
    fig, axes = plt.subplots(3, len(reps), figsize=(3.6 * len(reps), 10.5),
                             gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]})
    for c, key in enumerate(reps):
        g = store[key]
        X, m, ev, dt = g["X"], g["m"], g["ev"], g["m"]["dt_ns"]
        k_env = ev["k_env"]
        n = int(k_env + 6.0 / dt)
        t = np.arange(n) * dt
        ax = axes[0, c]
        lim = np.percentile(np.abs(X[:n]), 99)
        ax.imshow(X[:n], aspect="auto", cmap="gray", vmin=-lim, vmax=lim,
                  extent=[0, X.shape[1] * m["trace_dx_m"], n * dt, 0])
        ax.axhline(m["t0_sample_est"] * dt, color=COL[1], lw=1.2, ls="--", label="t0_sample_est")
        ax.axhline(m["t0_sample_act_median"] * dt, color=COL[2], lw=1.2, ls=":", label="t0_sample_act_median")
        ax.axhline(k_env * dt, color=COL[0], lw=1.0, label="band envelope peak")
        ax.set_title(f"{key[0]} {key[1]}", fontsize=10, color=INK)
        ax.set_xlabel("distance along line (m)", fontsize=8)
        if c == 0:
            ax.set_ylabel("time (ns, record clock)")
            ax.legend(fontsize=7, loc="lower right", frameon=True)
        ax = axes[1, c]
        mt = X[:n].mean(axis=1)
        sc = np.abs(mt).max()
        ax.plot(mt / sc, t, color=COL[0], lw=1.6, label="mean trace (signed)")
        ax.plot(envelope(X.mean(axis=1))[:n] / sc, t, color=INK2, lw=1.2, ls="-", label="envelope")
        ax.axhline(m["t0_sample_est"] * dt, color=COL[1], lw=1.2, ls="--")
        ax.axhline(m["t0_sample_act_median"] * dt, color=COL[2], lw=1.2, ls=":")
        for L in g["lobes"]:
            ax.plot([L["rel_amp_mean"]], [L["t_med_sample"] * dt], "o", ms=6, color=COL[3])
        ax.invert_yaxis()
        ax.set_xlabel("normalised amplitude", fontsize=8)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        if c == 0:
            ax.set_ylabel("time (ns)")
            ax.legend(fontsize=7, loc="lower right", frameon=False)
        ax = axes[2, c]
        for i, k in enumerate((0.0, 1.0, 2.0)):
            ax.plot(g["coh"][k][:n], t, color=COL[[6, 0, 1][i]], lw=1.5, label=f"aligned by {k:g}·s_j")
        ax.invert_yaxis()
        ax.set_xlabel("stack coherence", fontsize=8)
        ax.set_title(f"band shift std {np.std(g['s']) * dt:.3f} ns", fontsize=8, color=INK2)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        if c == 0:
            ax.set_ylabel("time (ns)")
            ax.legend(fontsize=7, loc="lower right", frameon=False)
    fig.suptitle("4TU zero band: B-scan (bandpassed, no gain), mean trace and lobes, shift-coherence test",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fourtu_explore_zero.png", dpi=110)
    plt.close(fig)


def display_gain(R: np.ndarray, dt: float, k_env: float) -> np.ndarray:
    """Display-only gain (t^1.5 after the band) -- never used for picking."""
    t = np.clip((np.arange(R.shape[0]) - k_env) * dt, 0, None)
    G = R * (1 + t[:, None]) ** 1.5
    return G


def fig_quicklook(store: dict) -> None:
    for act in ACTS:
        keys = [k for k in store if k[0] == act]
        if act == "010.14":
            keys = [k for k in keys if k[1] != "Path1"]  # Path1 is a 67 m zig-zag multi-line
        fig, axes = plt.subplots(len(keys), 1, figsize=(16, 3.4 * len(keys)))
        axes = np.atleast_1d(axes)
        for ax, key in zip(axes, keys):
            g = store[key]
            dt = g["m"]["dt_ns"]
            R = median_bg(g["X"])
            g["R"] = R
            n = min(R.shape[0], int(g["ev"]["k_env"] + (32.0 if dt < 0.15 else 26.0) / dt))
            G = display_gain(R, dt, g["ev"]["k_env"])[:n]
            lim = np.percentile(np.abs(G[int(g["ev"]["k_env"] + 1.0 / dt):]), 99)
            ax.imshow(G, aspect="auto", cmap="gray", vmin=-lim, vmax=lim)
            ax.set_xticks(np.arange(0, R.shape[1], 25), minor=True)
            ax.set_xticks(np.arange(0, R.shape[1], 50))
            ax.set_yticks(np.arange(0, n, 10), minor=True)
            ax.set_yticks(np.arange(0, n, 50))
            ax.grid(which="both", color="#ff9b6b", alpha=0.25, lw=0.5)
            ax.set_title(f"{key[0]} {key[1]} (median bg, display gain t^1.5) — x: trace, y: sample", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"fourtu_explore_bscans_{act.replace('.', '_')}.png", dpi=70)
        plt.close(fig)


def gt_table(gt, ulist):
    """GT depths per activity with radius when the metadata order matches the GT table."""
    out = {}
    for act in ACTS:
        g = [r for r in gt if r["activity_id"] == act and r["depth_m"] not in ("", None)]
        u = [r for r in ulist if r["activity_id"] == act]
        order_ok = bool(u) and all(r["정답표와 순서 일치"] == "예" for r in u) and len(u) == len(g)
        rows = []
        for i, r in enumerate(g):
            dia = nanf(u[i]["diameter_mm"]) if order_ok else float("nan")
            rows.append({"seq": int(r["seq"]), "code": r["code"], "depth_m": float(r["depth_m"]),
                         "R_m": dia / 2000 if np.isfinite(dia) else float("nan")})
        out[act] = rows
    return out


def apex_part(store, gt, ulist) -> None:
    gtt = gt_table(gt, ulist)
    apex_rows, eps_rows, examples = [], [], []
    for (act, name), seeds in SEEDS.items():
        g = store[(act, name)]
        m, X, dt, dx = g["m"], g["X"], g["m"]["dt_ns"], g["m"]["trace_dx_m"]
        eps_meta = float(m["eps_r_used"])
        v_meta = C0 / np.sqrt(eps_meta)
        R = g.get("R")
        if R is None:
            R = median_bg(X)
        E = envelope(R)
        for (js, ks) in seeds:
            j_a, k_a = refine_seed(E, js, ks, dj=12, dk=15)
            # local zero: band envelope peak of the mean trace within +-W_BG/2 traces (spec 4-1)
            a, b = max(j_a - W_BG // 2, 0), min(j_a + W_BG // 2 + 1, X.shape[1])
            k_z = zero_events(X[:, a:b], dt)["k_env"]
            idx, kp = track(E, j_a, k_a, dt, dx)
            x = idx * dx
            tau = (kp - k_z) * dt
            n_l, n_r = int((idx < j_a).sum()), int((idx > j_a).sum())
            ok = len(idx) >= 9 and n_l >= 3 and n_r >= 3
            x0, tau_a = local_apex(x, tau, j_a * dx)
            d_top = v_meta * tau_a / 2
            row = {"activity": act, "line": name, "seed": f"{js}/{ks}", "apex_trace": j_a,
                   "apex_sample": k_a, "zero_sample_env": k_z, "x0_m": x0, "tau_a_ns": tau_a,
                   "apex_env_fwhm_ns": fwhm(E[:, j_a], int(round(k_a))) * dt,
                   "band_env_fwhm_ns": g["ev"]["fwhm_ns"],
                   "n_picks": len(idx), "n_left": n_l, "n_right": n_r, "fit_ok": ok,
                   "eps_meta": eps_meta, "d_top_meta_m": d_top,
                   "d_top_zero_est_m": v_meta * (tau_a + (k_z - m["t0_sample_est"]) * dt) / 2,
                   "d_top_zero_act_m": v_meta * (tau_a + (k_z - m["t0_sample_act_median"]) * dt) / 2}
            if ok:
                v_s, rms_s = straight_fit(x, tau)
                row["eps_straight"] = (C0 / v_s) ** 2
                row["rms_straight_ns"] = rms_s
                epss = []
                for h in H_GRID:
                    e_fit, rms_h, edge = eps_scan(x, tau, x0, tau_a, h, 0.0)
                    rms_meta = float(np.sqrt(np.mean((model_tau(x, x0, tau_a, eps_meta, h, 0.0) - tau) ** 2)))
                    epss.append(e_fit)
                    eps_rows.append({"activity": act, "line": name, "apex_trace": j_a, "tau_a_ns": tau_a,
                                     "h_m": h, "eps_fit": e_fit, "rms_fit_ns": rms_h, "at_grid_edge": edge,
                                     "rms_at_eps_meta_ns": rms_meta, "eps_meta": eps_meta,
                                     "rel_dev": (e_fit - eps_meta) / eps_meta})
                    row[f"eps_h{h:g}"] = e_fit
                    row[f"rms_h{h:g}"] = rms_h
                # h at which the refraction fit reproduces eps_meta (linear interpolation)
                eh = np.array(epss)
                hh = np.array(H_GRID)
                h_imp = float("nan")
                for i in range(len(hh) - 1):
                    if (eh[i] - eps_meta) * (eh[i + 1] - eps_meta) <= 0 and eh[i] != eh[i + 1]:
                        h_imp = hh[i] + (eps_meta - eh[i]) / (eh[i + 1] - eh[i]) * (hh[i + 1] - hh[i])
                        break
                row["h_implied_by_eps_meta_m"] = h_imp
                examples.append((act, name, j_a, x, tau, x0, tau_a, eps_meta, row))
            apex_rows.append(row)
            print(f"[apex] {act:6s} {name:13s} seed {js}/{ks} -> trace {j_a} sample {k_a:.1f} zero {k_z:.1f} "
                  f"tau_a {tau_a:.2f} ns d_top(meta) {d_top:.3f} m picks {len(idx)} ({n_l}/{n_r}) "
                  + (f"eps straight {row['eps_straight']:.2f} | eps(h) "
                     + " ".join(f"{row[f'eps_h{h:g}']:.1f}" for h in H_GRID)
                     + f" | h_implied {row['h_implied_by_eps_meta_m']:.2f}" if ok else "fit skipped"))
    keys = []
    for r in apex_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    apex_rows = [{k: r.get(k, "") for k in keys} for r in apex_rows]
    write_csv(OUT / "fourtu_explore_apex.csv", apex_rows)
    write_csv(OUT / "fourtu_explore_eps.csv", eps_rows)

    # ---- 4. depth correspondence (per line Hungarian, two GT reference hypotheses)
    match_rows = []
    for act in ACTS:
        G = gtt[act]
        if not G:
            continue
        for name in sorted({r["line"] for r in apex_rows if r["activity"] == act}):
            P = [r for r in apex_rows if r["activity"] == act and r["line"] == name]
            for hyp in ("top", "center"):
                gd = np.array([gg["depth_m"] - (gg["R_m"] if hyp == "center" and np.isfinite(gg["R_m"]) else 0.0)
                               for gg in G])
                for zero_key in ("d_top_meta_m", "d_top_zero_est_m", "d_top_zero_act_m"):
                    pd = np.array([float(p[zero_key]) for p in P])
                    cost = np.abs(pd[:, None] - gd[None, :])
                    ri, ci = linear_sum_assignment(cost)
                    for i, c in zip(ri, ci):
                        match_rows.append({"activity": act, "line": name, "gt_ref": hyp, "zero": zero_key,
                                           "apex_trace": P[i]["apex_trace"], "d_pred_m": pd[i],
                                           "gt_seq": G[c]["seq"], "gt_code": G[c]["code"],
                                           "gt_depth_m": G[c]["depth_m"], "gt_R_m": G[c]["R_m"],
                                           "gt_top_used_m": gd[c], "err_m": pd[i] - gd[c],
                                           "within_30cm": bool(abs(pd[i] - gd[c]) <= 0.30)})
    write_csv(OUT / "fourtu_explore_depth_match.csv", match_rows)
    print("\n[match] activity | gt_ref | zero | n | MAE (m) | mean err (m) | within 0.30 m")
    for act in ACTS:
        for hyp in ("top", "center"):
            for zk in ("d_top_meta_m", "d_top_zero_est_m", "d_top_zero_act_m"):
                rr = [r for r in match_rows if r["activity"] == act and r["gt_ref"] == hyp and r["zero"] == zk]
                if rr:
                    e = np.array([r["err_m"] for r in rr])
                    print(f"[match] {act} | {hyp} | {zk} | {len(e)} | {np.mean(np.abs(e)):.3f} | {np.mean(e):+.3f} | "
                          f"{sum(r['within_30cm'] for r in rr)}/{len(rr)}")
    # apex depths vs GT list (unmatched view)
    for act in ACTS:
        ds = sorted(float(r["d_top_meta_m"]) for r in apex_rows if r["activity"] == act)
        print(f"[depths] {act}: apex d_top(meta) {np.round(ds, 2).tolist()} | GT {[gg['depth_m'] for gg in gtt[act]]} "
              f"R {[gg['R_m'] for gg in gtt[act]]}")

    # quality-filtered refraction QC summary (both wings >= 8 picks, straight-ray RMS <= 0.3 ns)
    print("\n[qc] activity | n good | eps_straight median (IQR) | eps(h=0) median | eps(h=0.1) median | "
          "h_implied median (m) | apex FWHM median (ns) | band FWHM (ns)")
    for act in ACTS:
        rr = [r for r in apex_rows if r["activity"] == act and r["fit_ok"] is True
              and r["n_left"] >= 8 and r["n_right"] >= 8 and float(r["rms_straight_ns"]) <= 0.3]
        allr = [r for r in apex_rows if r["activity"] == act]
        fw = np.median([float(r["apex_env_fwhm_ns"]) for r in allr]) if allr else float("nan")
        bw = np.median([float(r["band_env_fwhm_ns"]) for r in allr]) if allr else float("nan")
        if rr:
            es = np.array([float(r["eps_straight"]) for r in rr])
            e0 = np.array([float(r["eps_h0"]) for r in rr])
            e1 = np.array([float(r["eps_h0.1"]) for r in rr])
            hi = np.array([float(r["h_implied_by_eps_meta_m"]) for r in rr])
            print(f"[qc] {act} | {len(rr)} | {np.median(es):.2f} ({np.percentile(es, 25):.2f}-{np.percentile(es, 75):.2f}) | "
                  f"{np.median(e0):.2f} | {np.median(e1):.2f} | {np.nanmedian(hi) if np.isfinite(hi).any() else float('nan'):.3f} "
                  f"(finite {np.isfinite(hi).sum()}) | {fw:.2f} | {bw:.2f}")
        else:
            print(f"[qc] {act} | 0 | - | - | - | - | {fw:.2f} | {bw:.2f}")

    fig_eps(apex_rows, gtt)
    fig_examples(examples)
    fig_depth(apex_rows, gtt)


def fig_eps(apex_rows, gtt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for i, act in enumerate(ACTS):
        rr = [r for r in apex_rows if r["activity"] == act and r["fit_ok"] is True]
        for k, r in enumerate(rr):
            ax.plot(H_GRID, [r[f"eps_h{h:g}"] for h in H_GRID], "-o", color=COL[i], ms=3, lw=1.0, alpha=0.7,
                    label=act if k == 0 else None)
        if rr:
            ax.axhline(rr[0]["eps_meta"], color=COL[i], lw=1.4, ls="--")
    ax.set_xlabel("assumed antenna height h (m), s = 0")
    ax.set_ylabel("eps_r from refraction-model fit")
    ax.set_title("eps_r(h) per apex; dashed = eps_r_used (meta)", fontsize=10)
    ax.set_ylim(0, 40)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    for i, act in enumerate(ACTS):
        rr = [r for r in apex_rows if r["activity"] == act and r["fit_ok"] is True]
        for k, r in enumerate(rr):
            ax.plot(H_GRID, [r[f"rms_h{h:g}"] for h in H_GRID], "-o", color=COL[i], ms=3, lw=1.0, alpha=0.7,
                    label=act if k == 0 else None)
    ax.set_xlabel("assumed antenna height h (m)")
    ax.set_ylabel("pick RMS at best eps (ns)")
    ax.set_title("fit RMS vs h (flat = h not identifiable from one hyperbola)", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fourtu_explore_eps.png", dpi=110)
    plt.close(fig)


def fig_examples(examples) -> None:
    pick = examples[:: max(1, len(examples) // 4)][:4]
    fig, axes = plt.subplots(1, len(pick), figsize=(4.2 * len(pick), 4.2))
    for ax, (act, name, j_a, x, tau, x0, tau_a, eps_meta, row) in zip(np.atleast_1d(axes), pick):
        xx = np.linspace(x.min(), x.max(), 200)
        ax.plot(x - x0, tau, "o", ms=3, color=INK2, label="envelope picks")
        for i, (h, e, lab) in enumerate([(0.0, row["eps_straight"], "straight ray, free v"),
                                         (0.0, eps_meta, "straight ray, eps_meta"),
                                         (0.3, eps_meta, "refraction h 0.3, eps_meta"),
                                         (0.3, row["eps_h0.3"], "refraction h 0.3, best eps")]):
            ax.plot(xx - x0, model_tau(xx, x0, tau_a, e, h, 0.0), color=COL[i], lw=1.5,
                    label=f"{lab} ({e:.1f})")
        ax.invert_yaxis()
        ax.set_xlabel("x - x0 (m)")
        ax.set_ylabel("tau = t - t_zero (ns)")
        ax.set_title(f"{act} {name} trace {j_a}", fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False, loc="lower center")
    fig.tight_layout()
    fig.savefig(OUT / "fourtu_explore_fit_examples.png", dpi=110)
    plt.close(fig)


def fig_depth(apex_rows, gtt) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, act in enumerate(ACTS):
        G = gtt[act]
        gd = [gg["depth_m"] for gg in G]
        gc = [gg["depth_m"] - gg["R_m"] for gg in G if np.isfinite(gg["R_m"])]
        ax.plot([i - 0.25] * len(gd), gd, "_", ms=18, mew=2, color=INK, label="GT depth_m" if i == 0 else None)
        if gc:
            ax.plot([i - 0.25] * len(gc), gc, "_", ms=12, mew=2, color=COL[7],
                    label="GT top if depth_m = centre (known R)" if i == 0 else None)
        pd = [float(r["d_top_meta_m"]) for r in apex_rows if r["activity"] == act]
        jit = np.linspace(-0.08, 0.08, len(pd)) if pd else []
        ax.plot(i + 0.2 + np.asarray(jit), pd, "o", ms=6, color=COL[i], mec="#fcfcfb", mew=1.0,
                label=None)
    ax.set_xticks(range(len(ACTS)))
    ax.set_xticklabels([f"{a}\n(left: GT, right: apex d_top, meta v)" for a in ACTS], fontsize=8)
    ax.set_ylabel("depth (m)")
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Apex depth (zero = band envelope peak, v = c/sqrt(eps_r_used)) vs ground-truth depth set",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fourtu_explore_depth.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
