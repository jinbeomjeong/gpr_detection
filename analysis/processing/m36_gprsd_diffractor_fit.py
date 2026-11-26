"""M3-6 D2 / D2s / D1b: point-diffractor fits outside the label windows.

Region (Q3): every pick of a diffractor must lie outside pipeline mark +-400 traces.

Units: tau in samples, u in traces. Model (ground coupled, point scatterer R = 0):
    tau(u) = tau_z + sqrt(h^2 + ((u - u0) g)^2),   g = dx / dz  [samples per trace],
    tau_a = tau_z + h (apex), dz = dx / g [m per sample].
With a fixed radius R (m):  tau(u) = tau_z + sqrt((h + R/dz)^2 + ((u - u0) g)^2) - R/dz.
The radius is never a free parameter.

Steps per line
  1. section = stored amplitude - moving median (101 traces) (horizontal coherent part removed)
  2. pick section = zero-phase Butterworth band-pass, periods 8-40 samples (excludes the
     ~51-sample component, D1a); envelope = |hilbert|
  3. candidates: 2-D local maxima of the envelope (21 samples x 41 traces), sample 20-250,
     envelope >= 5 x line median envelope (samples 20-250), top 500 per line
  4. tracking (signed extremum of the apex polarity, both sides, <= 70 traces, search window
     around a linear prediction, floor 0.2 x apex amplitude, stop after 2 misses)
  5. own RANSAC on (tau - tau_z)^2 = c0 + c1 u + c2 u^2 (seeded) -> least_squares soft_l1
     (f_scale 1 sample) for (u0, h, g) with tau_z fixed at 0 (QC fit)
  6. QC (decided before the run): converged and not at bounds; >= 15 picks each side of u0;
     RMS residual <= 0.25 P_ref; inlier (|res| <= 2) fraction >= 0.5; sigma_dz/dz <= 0.15;
     dz in the physical band [0.0070, 0.0262] m (eps_r 3-30 under dt A or B).
     De-duplicate per line: |du0| <= 15 and |dtau_a| <= 2 P_ref -> keep the larger apex envelope.
  7. D2: common tau_z by profiling: for each tau_z on a grid each diffractor is refitted
     (u0, h, g free) and its mean soft_l1 cost is summed (equal weight per diffractor).
     Road-cluster bootstrap (seed 20261126) of the argmin. dz at the estimate = median.
  8. D2s: dz at tau_z in {0, D3b candidate, D2 estimate}; tau_z profile for R in {0, 0.05, 0.15} m;
     depth-trend variant (Theil-Sen slope of dz_i vs h_i is zero) - report only.
  9. D1b: carrier period at the apex A-scan (section of step 1, mean of 3 traces): envelope
     gate +-1.5 P_ref around the envelope peak; gated spectrum peak (Hann, x8 pad; primary)
     and mean zero-crossing interval x 2 (secondary).
 10. Request (c) of D6: v = 2 dz / dt for dt A and B, report only.

Usage: python -I m36_gprsd_diffractor_fit.py <P_ref samples> <tau_z D3b candidate samples> [plan|strict]
  plan   -> m36_gprsd_diffractor_fit.json (QC of the plan, step 6)
  strict -> m36_gprsd_diffractor_fit_strict.json (strict_check added after the plan run, disclosed)
Output: analysis/processing/<name> public summary (no diffractors/picks) + data/derived/m36_gprsd/<name> full
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from scipy import ndimage, optimize, signal, stats  # noqa: E402

from m36_gprsd_common import (BG_WINDOW, C_M_PER_NS as C_LIGHT, DT_A, DT_B, DX_M, DZ_META, ROADS, SEED,  # noqa: E402
                              bandpass, excluded_mask, line_ids, load_line, pipe_marks,
                              qstats, road_bootstrap, road_of, write_json_split)

PICK_BAND = (8.0, 40.0)
S_MIN, S_MAX = 20, 250
EDGE = 80
TOP_N = 500
ENV_K = 5.0
MAX_SIDE = 70
FLOOR = 0.2
MIN_SIDE = 15
G_BOUNDS = (DX_M / 0.06, DX_M / 0.003)
DZ_PHYS = (0.0070, 0.0262)
TAUZ_GRID = np.arange(-80.0, 40.0 + 0.5, 1.0)
R_LIST = (0.0, 0.05, 0.15)
F_SCALE = 1.0
N_BOOT = 2000


def section_bg_removed(a: np.ndarray, chunk: int = 3000) -> np.ndarray:
    n = a.shape[1]
    out = np.empty_like(a)
    h = BG_WINDOW // 2
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        m0, m1 = max(0, c0 - h), min(n, c1 + h)
        bg = ndimage.median_filter(a[:, m0:m1], size=(1, BG_WINDOW), mode="nearest")
        out[:, c0:c1] = a[:, c0:c1] - bg[:, c0 - m0:c1 - m0]
    return out


def find_candidates(env: np.ndarray, excl: np.ndarray) -> list[tuple[int, int, float]]:
    mx = ndimage.maximum_filter(env, size=(21, 41), mode="nearest")
    thr = ENV_K * np.median(env[S_MIN:S_MAX])
    k, u = np.nonzero((env == mx) & (env >= thr))
    keep = (k >= S_MIN) & (k <= S_MAX) & (u >= EDGE) & (u < env.shape[1] - EDGE)
    k, u = k[keep], u[keep]
    keep = ~excl[u]
    k, u = k[keep], u[keep]
    e = env[k, u]
    order = np.argsort(-e)[:TOP_N]
    return [(int(k[i]), int(u[i]), float(e[i])) for i in order]


def parabolic(y: np.ndarray, i: int) -> float:
    if 0 < i < len(y) - 1:
        d = y[i - 1] - 2 * y[i] + y[i + 1]
        if d != 0:
            return i + 0.5 * (y[i - 1] - y[i + 1]) / d
    return float(i)


def local_extrema(col: np.ndarray, lo: int, hi: int) -> list[int]:
    lo, hi = max(1, lo), min(len(col) - 2, hi)
    return [i for i in range(lo, hi + 1) if col[i] >= col[i - 1] and col[i] >= col[i + 1] and col[i] > 0]


def track(b: np.ndarray, k0: int, u0: int, p_ref: float):
    n_s, n_t = b.shape
    half = int(round(p_ref / 2))
    seg = np.abs(b[max(0, k0 - half):k0 + half + 1, u0])
    ka = max(0, k0 - half) + int(np.argmax(seg))
    s = 1.0 if b[ka, u0] >= 0 else -1.0
    amp0 = s * b[ka, u0]
    col = s * b[:, u0]
    picks = {u0: parabolic(col, ka)}
    for direction in (-1, 1):
        prev, prevprev = picks[u0], None
        miss = 0
        u = u0
        for _ in range(MAX_SIDE):
            u += direction
            if u < 0 or u >= n_t:
                break
            slope = 0.0 if prevprev is None else float(np.clip(prev - prevprev, 0.0, 4.0))
            pred = prev + slope * (1 + miss)
            lo = int(np.floor(max(pred - 2.0, prev - 1.0)))
            hi = int(np.ceil(min(pred + 3.0, prev + 5.0 * (1 + miss))))
            col = s * b[:, u]
            cand = [i for i in local_extrema(col, lo, hi) if col[i] >= FLOOR * amp0]
            if not cand:
                miss += 1
                if miss >= 2:
                    break
                continue
            i = min(cand, key=lambda j: abs(j - pred))
            t = parabolic(col, i)
            if t >= n_s - 3:
                break
            prevprev, prev = (prev if miss == 0 else prev - slope * miss), t
            picks[u] = t
            miss = 0
    us = np.array(sorted(picks))
    ts = np.array([picks[x] for x in us])
    return us.astype(float), ts, s, float(amp0), ka


def model(p, u, tau_z, r_samp):
    u0, h, g = p
    return tau_z + np.sqrt((h + r_samp) ** 2 + ((u - u0) * g) ** 2) - r_samp


def ransac_init(u, t, tau_z, rng, n_iter=200):
    y = (t - tau_z)
    best, best_n = None, -1
    n = len(u)
    if n < 3:
        return None
    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        try:
            c2, c1, c0 = np.polyfit(u[idx], y[idx] ** 2, 2)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not c2 > 0:
            continue
        uu0 = -c1 / (2 * c2)
        h2 = c0 - c2 * uu0 ** 2
        if not h2 > 0:
            continue
        pred = np.sqrt(h2 + c2 * (u - uu0) ** 2)
        ni = int((np.abs(pred - y) <= 2.0).sum())
        if ni > best_n:
            best_n, best = ni, (uu0, np.sqrt(h2), np.sqrt(c2))
    return best


def fit_fixed(u, t, tau_z, r_m, p_init):
    """least_squares soft_l1 for (u0, h, g) at fixed tau_z and radius r_m (m)."""
    lo = [u.min(), 0.5, G_BOUNDS[0]]
    hi = [u.max(), 400.0, G_BOUNDS[1]]
    x0 = np.clip(np.asarray(p_init, float), np.array(lo) + 1e-6, np.array(hi) - 1e-6)

    def fun(p):
        r_samp = r_m * p[2] / DX_M  # R / dz in samples, dz = dx / g
        return model(p, u, tau_z, r_samp) - t

    res = optimize.least_squares(fun, x0, bounds=(lo, hi), loss="soft_l1", f_scale=F_SCALE)
    return res, lo, hi


def qc_fit(u, t, p_ref, rng):
    init = ransac_init(u, t, 0.0, rng)
    if init is None:
        return None, "ransac_failed"
    res, lo, hi = fit_fixed(u, t, 0.0, 0.0, init)
    u0, h, g = res.x
    r = model(res.x, u, 0.0, 0.0) - t
    info = {"u0": float(u0), "h": float(h), "g": float(g), "dz": DX_M / float(g),
            "rms": float(np.sqrt(np.mean(r ** 2))), "inlier_frac": float(np.mean(np.abs(r) <= 2.0)),
            "n_left": int((u < u0).sum()), "n_right": int((u > u0).sum()), "n_picks": int(len(u))}
    try:
        J = res.jac
        dof = max(1, len(u) - 3)
        s2 = float((r[np.abs(r) <= 2.0] ** 2).sum()) / dof
        cov = np.linalg.inv(J.T @ J) * s2
        sg = float(np.sqrt(cov[2, 2]))
        info["sigma_dz_rel"] = sg / float(g)
    except np.linalg.LinAlgError:
        info["sigma_dz_rel"] = float("inf")
    if not res.success:
        return info, "not_converged"
    span = (hi[2] - lo[2]) * 1e-3
    if g - lo[2] <= span or hi[2] - g <= span:
        return info, "g_at_bound"
    if u0 - u.min() < 1 or u.max() - u0 < 1:
        return info, "u0_at_edge"
    if info["n_left"] < MIN_SIDE or info["n_right"] < MIN_SIDE:
        return info, "short_wing"
    if info["rms"] > 0.25 * p_ref:
        return info, "rms"
    if info["inlier_frac"] < 0.5:
        return info, "inliers"
    if not info["sigma_dz_rel"] <= 0.15:
        return info, "unstable"
    if not DZ_PHYS[0] <= info["dz"] <= DZ_PHYS[1]:
        return info, "dz_out_of_band"
    return info, "pass"


def apex_period(sec: np.ndarray, u0: float, tau_a: float, p_ref: float) -> dict:
    j = int(round(u0))
    tr = sec[:, max(0, j - 1):j + 2].mean(axis=1)
    env = np.abs(signal.hilbert(tr))
    half = int(round(p_ref / 2))
    k0 = int(round(tau_a))
    lo = max(0, k0 - half)
    kp = lo + int(np.argmax(env[lo:k0 + half + 1]))
    g0, g1 = max(0, int(round(kp - 1.5 * p_ref))), min(len(tr), int(round(kp + 1.5 * p_ref)) + 1)
    seg = tr[g0:g1] - tr[g0:g1].mean()
    out = {"env_peak_sample": kp, "gate": [g0, g1]}
    if len(seg) < 8:
        return out
    w = seg * np.hanning(len(seg))
    nfft = 512
    sp = np.abs(np.fft.rfft(w, nfft))
    f = np.fft.rfftfreq(nfft)
    ok = (f >= 1 / 80.0) & (f <= 1 / 4.0)
    loc, _ = signal.find_peaks(sp)
    loc = loc[ok[loc]]
    if loc.size:
        out["gated_spec_period"] = float(1.0 / f[loc[np.argmax(sp[loc])]])
    zc = []
    for i in range(len(seg) - 1):
        if seg[i] == 0 or seg[i] * seg[i + 1] < 0:
            zc.append(i + (seg[i] / (seg[i] - seg[i + 1]) if seg[i] != seg[i + 1] else 0.0))
    if len(zc) >= 3:
        out["zc_period"] = float(2.0 * np.mean(np.diff(zc)))
        out["n_zero_crossings"] = len(zc)
    return out


def profile_curves(diffs, r_m):
    """Per-diffractor mean soft_l1 cost on TAUZ_GRID (inf where tau_z >= tau_a - 0.5)."""
    curves = np.full((len(diffs), len(TAUZ_GRID)), np.inf)
    dz_tab = np.full((len(diffs), len(TAUZ_GRID)), np.nan)
    h_tab = np.full((len(diffs), len(TAUZ_GRID)), np.nan)
    for i, d in enumerate(diffs):
        u, t = d["_u"], d["_t"]
        # start at tau_z = 0 from the QC fit and walk outwards (warm start)
        i0 = int(np.argmin(np.abs(TAUZ_GRID)))
        for order in (range(i0, len(TAUZ_GRID)), range(i0 - 1, -1, -1)):
            p = np.array([d["u0"], d["h"], d["g"]])
            for j in order:
                tz = TAUZ_GRID[j]
                tau_a = d["u0_tau_a"]
                if tz >= tau_a - 0.5:
                    break
                p[1] = max(0.6, tau_a - tz)
                res, _, _ = fit_fixed(u, t, tz, r_m, p)
                rr = res.fun / F_SCALE
                curves[i, j] = float(np.mean(2.0 * (np.sqrt(1.0 + rr ** 2) - 1.0)))
                dz_tab[i, j] = DX_M / res.x[2]
                h_tab[i, j] = res.x[1]
                p = res.x.copy()
    return curves, dz_tab, h_tab


def argmin_tauz(curves, rows=None):
    c = curves if rows is None else curves[rows]
    tot = c.sum(axis=0)
    if not np.isfinite(tot).any():
        return float("nan"), tot
    j = int(np.nanargmin(np.where(np.isfinite(tot), tot, np.nan)))
    # sub-grid parabola
    x = float(TAUZ_GRID[j])
    if 0 < j < len(tot) - 1 and np.isfinite(tot[j - 1]) and np.isfinite(tot[j + 1]):
        d = tot[j - 1] - 2 * tot[j] + tot[j + 1]
        if d > 0:
            x += 0.5 * (tot[j - 1] - tot[j + 1]) / d
    return x, tot


def boot_tauz(curves, roads, rng, n_boot=N_BOOT):
    road_names = sorted(set(roads))
    idx = {r: np.flatnonzero(np.array(roads) == r) for r in road_names}
    est, _ = argmin_tauz(curves)
    bs = np.empty(n_boot)
    bs_in = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(road_names), len(road_names))
        rows = np.concatenate([idx[road_names[i]] for i in pick])
        bs[b], _ = argmin_tauz(curves, rows)
        rows2 = np.concatenate([rng.choice(idx[r], len(idx[r]), replace=True) for r in road_names])
        bs_in[b], _ = argmin_tauz(curves, rows2)
    out = {}
    for name, arr in (("road_cluster", bs), ("diffractor_within_road", bs_in)):
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        out[name] = {"ci95": [float(lo), float(hi)], "half_width": float((hi - lo) / 2),
                     "n_boot": n_boot, "seed": SEED}
    return est, out


STRICT_RMS = 1.5          # samples
STRICT_MAX_ABS = 3.0      # samples (no lobe / event jumps)
STRICT_INLIER = 0.9
STRICT_STEP_SLACK = 1.5   # samples per trace above the fitted asymptote slope g


def strict_check(u, t, info) -> str:
    """Strict QC (added after the plan-QC run showed lobe-jumping picks; disclosed).

    Every pick within 3 samples of the tau_z = 0 fit, RMS <= 1.5 samples, >= 90 % within 2
    samples, picks never rise by more than 0.5 sample and never fall faster than g + 1.5 per
    trace going outward from the apex (beyond 2 traces), dz <= c dt_A / 2 (v <= c; the plan band
    [0.0070, 0.0262] is NOT applied here because it truncated the distribution).
    """
    r = model((info["u0"], info["h"], info["g"]), u, 0.0, 0.0) - t
    if np.sqrt(np.mean(r ** 2)) > STRICT_RMS:
        return "strict_rms"
    if np.max(np.abs(r)) > STRICT_MAX_ABS:
        return "strict_max_abs"
    if np.mean(np.abs(r) <= 2.0) < STRICT_INLIER:
        return "strict_inliers"
    o = np.argsort(u)
    uu, tt = u[o], t[o]
    for side in (1, -1):
        sel = (uu - info["u0"]) * side > 2
        us_, ts_ = uu[sel], tt[sel]
        if side == -1:
            us_, ts_ = us_[::-1], ts_[::-1]
        steps = np.diff(ts_) / np.maximum(1.0, np.abs(np.diff(us_)))
        if steps.size and (steps.min() < -0.5 or steps.max() > info["g"] + STRICT_STEP_SLACK):
            return "strict_step"
    if info["dz"] > C_LIGHT * DT_A / 2:
        return "strict_v_above_c"
    return "pass"


def main() -> None:
    p_ref = float(sys.argv[1])
    tauz_d3b = float(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "plan"
    assert mode in ("plan", "strict")
    out_name = "m36_gprsd_diffractor_fit.json" if mode == "plan" else "m36_gprsd_diffractor_fit_strict.json"
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    marks = pipe_marks()
    diffs, reasons, n_cand = [], {}, {}
    rejected_log = []
    for fid in line_ids():
        a = load_line(fid)
        n = a.shape[1]
        excl = excluded_mask(fid, n, marks)
        sec = section_bg_removed(a)
        b = bandpass(sec, *PICK_BAND)
        env = np.abs(signal.hilbert(b, axis=0))
        cands = find_candidates(env, excl)
        n_cand[fid] = len(cands)
        kept = []
        for (k0, u0, e0) in cands:
            us, ts, s, amp0, ka = track(b, k0, u0, p_ref)
            if excl[us.astype(int)].any():
                reasons["touches_exclusion"] = reasons.get("touches_exclusion", 0) + 1
                continue
            info, why = qc_fit(us, ts, p_ref, rng)
            if mode == "strict" and why in ("pass", "dz_out_of_band"):
                why = strict_check(us, ts, info)
            reasons[why] = reasons.get(why, 0) + 1
            if info is not None and why in ("dz_out_of_band", "unstable", "inliers", "rms", "pass"):
                rejected_log.append({"reason": why, "dz": info["dz"], "rms": info["rms"],
                                     "n_left": info["n_left"], "n_right": info["n_right"],
                                     "tau_a": info["h"], "road": road_of(fid)})
            if why != "pass":
                continue
            info.update({"file_id": fid, "road": road_of(fid), "cand_sample": k0, "cand_trace": u0,
                         "apex_env": e0, "polarity": int(s), "tau_a": info["h"],
                         "u0_tau_a": info["h"], "_u": us, "_t": ts})
            kept.append(info)
        # de-duplicate
        kept.sort(key=lambda d: -d["apex_env"])
        final = []
        for d in kept:
            if any(abs(d["u0"] - f["u0"]) <= 15 and abs(d["tau_a"] - f["tau_a"]) <= 2 * p_ref for f in final):
                reasons["duplicate"] = reasons.get("duplicate", 0) + 1
                continue
            final.append(d)
        for d in final:
            d.update(apex_period(sec, d["u0"], d["tau_a"], p_ref))
            d["apex_period_raw_stored"] = apex_period(a, d["u0"], d["tau_a"], p_ref).get("gated_spec_period")
        diffs.extend(final)
        print(json.dumps({"line": fid, "candidates": len(cands), "pass": len(final),
                          "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    roads = [d["road"] for d in diffs]
    by_road_n = {r: roads.count(r) for r in ROADS}
    out = {"diagnostic": "D2/D2s/D1b", "qc_mode": mode, "p_ref_samples": p_ref, "pick_band_periods": PICK_BAND,
           "seed": SEED, "candidates_per_line": n_cand, "qc_reasons": reasons,
           "n_pass": len(diffs), "n_pass_by_road": by_road_n,
           "n_roads_with_pass": sum(1 for r in ROADS if by_road_n[r] > 0),
           "dz_phys_band": DZ_PHYS, "tauz_grid": [float(TAUZ_GRID[0]), float(TAUZ_GRID[-1]), 1.0]}
    if len(diffs) < 3:
        p, pf = write_json_split(out_name, out)
        print(json.dumps({"status": "too_few", "n_pass": len(diffs), "out": str(p), "out_full": str(pf)}))
        return

    # D1b
    vb = {r: np.array([d["gated_spec_period"] for d in diffs if d["road"] == r and "gated_spec_period" in d])
          for r in ROADS}
    vz = {r: np.array([d["zc_period"] for d in diffs if d["road"] == r and "zc_period" in d]) for r in ROADS}
    out["D1b"] = {
        "gated_spec_period": qstats([d.get("gated_spec_period") for d in diffs]),
        "gated_spec_period_by_road": {r: qstats(vb[r]) for r in ROADS},
        "gated_spec_period_bootstrap": road_bootstrap(vb),
        "zc_period": qstats([d.get("zc_period") for d in diffs]),
        "zc_period_bootstrap": road_bootstrap(vz),
        "gated_spec_period_stored_amplitude": qstats([d.get("apex_period_raw_stored") for d in diffs]),
    }

    # D2 / D2s profiles
    prof = {}
    for r_m in R_LIST:
        curves, dz_tab, h_tab = profile_curves(diffs, r_m)
        est, ci = boot_tauz(curves, roads, np.random.default_rng(SEED))
        per_road = {}
        for rd in ROADS:
            rows = np.flatnonzero(np.array(roads) == rd)
            if rows.size:
                per_road[rd] = argmin_tauz(curves, rows)[0]
        _, tot = argmin_tauz(curves)
        prof[r_m] = {"curves": curves, "dz": dz_tab, "h": h_tab, "est": est, "ci": ci,
                     "per_road": per_road, "total": tot}
    p0 = prof[0.0]
    tz_hat = p0["est"]

    def dz_at(tz):
        j = int(np.argmin(np.abs(TAUZ_GRID - tz)))
        col = p0["dz"][:, j]
        okr = {r: col[(np.array(roads) == r) & np.isfinite(col)] for r in ROADS}
        return {"tau_z_grid": float(TAUZ_GRID[j]), "n_feasible": int(np.isfinite(col).sum()),
                "dz": qstats(col, 5), "dz_bootstrap": road_bootstrap({k: v for k, v in okr.items()}),
                "dz_by_road": {r: qstats(okr[r], 5) for r in ROADS}}

    out["D2"] = {
        "tau_z_hat": tz_hat, "tau_z_ci": p0["ci"], "tau_z_per_road": p0["per_road"],
        "min_tau_a": float(min(d["tau_a"] for d in diffs)),
        "dz_at_tau_z_hat": dz_at(tz_hat),
        "profile_total_cost": [None if not np.isfinite(v) else round(float(v), 4) for v in p0["total"]],
    }
    # Theil-Sen trend variant
    slopes = []
    for j, tz in enumerate(TAUZ_GRID):
        col, hh = p0["dz"][:, j], p0["h"][:, j]
        ok = np.isfinite(col)
        if ok.sum() >= 10:
            sl = stats.theilslopes(col[ok], hh[ok])[0]
            slopes.append((float(tz), float(sl)))
    zero_cross = None
    for (x1, s1), (x2, s2) in zip(slopes[:-1], slopes[1:]):
        if s1 == 0 or s1 * s2 < 0:
            zero_cross = x1 + (x2 - x1) * s1 / (s1 - s2) if s1 != s2 else x1
            break
    # QC-band truncation check (added after the first run, report only): fits that passed every
    # QC except the physical dz band, split by side, and the median with the band relaxed to v < c
    oob = [r for r in rejected_log if r["reason"] == "dz_out_of_band"]
    dz_c_a = C_LIGHT * DT_A / 2
    relaxed = [r["dz"] for r in rejected_log if r["reason"] == "pass"] + \
              [r["dz"] for r in oob if r["dz"] <= dz_c_a]
    out["qc_band_check"] = {
        "out_of_band_below": qstats([r["dz"] for r in oob if r["dz"] < DZ_PHYS[0]], 5),
        "out_of_band_above": qstats([r["dz"] for r in oob if r["dz"] > DZ_PHYS[1]], 5),
        "out_of_band_above_short_wing_or_dup_not_checked": "these fits passed wing/rms/inlier/sigma QC; "
                                                           "no de-duplication applied",
        "relaxed_band_upper_dz_(v=c_at_dtA)": dz_c_a,
        "relaxed_dz_tau_z_0": qstats(relaxed, 5),
        "pass_dz_tau_z_0": qstats([r["dz"] for r in rejected_log if r["reason"] == "pass"], 5),
        "other_rejects": {k: qstats([r["dz"] for r in rejected_log if r["reason"] == k], 5)
                          for k in ("unstable", "inliers", "rms")},
    }
    lo_ci, hi_ci = p0["ci"]["road_cluster"]["ci95"]
    out["D2s"] = {
        "dz_at_tau_z": {"0": dz_at(0.0), "d3b_candidate": dz_at(tauz_d3b), "d2_hat": dz_at(tz_hat),
                        "tau_z_ci_low": dz_at(lo_ci), "tau_z_ci_high": dz_at(hi_ci),
                        "minus_40": dz_at(-40.0)},
        "d3b_candidate_tau_z": tauz_d3b,
        "tau_z_by_radius_m": {str(r): {"est": prof[r]["est"], "ci": prof[r]["ci"],
                                       "per_road": prof[r]["per_road"]} for r in R_LIST},
        "trend_zero_tau_z": zero_cross,
        "trend_slopes_dz_per_sample": [(x, round(s, 8)) for x, s in slopes[::5]],
    }
    # (c) velocity under dt A/B at tau_z = 0 and at the estimate
    dz0 = np.array([d["dz"] for d in diffs])
    out["v_check_c"] = {
        "note": "report only; v = 2 dz / dt; physical road range 0.08-0.13 m/ns (D6 request)",
        "tau_z_0": {"v_hypA": qstats(2 * dz0 / DT_A), "v_hypB": qstats(2 * dz0 / DT_B),
                    "share_in_0.08_0.13_A": float(np.mean((2 * dz0 / DT_A >= 0.08) & (2 * dz0 / DT_A <= 0.13))),
                    "share_in_0.08_0.13_B": float(np.mean((2 * dz0 / DT_B >= 0.08) & (2 * dz0 / DT_B <= 0.13)))},
        "dz_meta": DZ_META, "v_meta_A": 2 * DZ_META / DT_A, "v_meta_B": 2 * DZ_META / DT_B,
    }
    out["qc_fit_tau_z_0"] = {"dz": qstats(dz0, 5), "tau_a": qstats([d["tau_a"] for d in diffs]),
                             "half_width_traces": qstats([min(d["n_left"], d["n_right"]) for d in diffs]),
                             "rms": qstats([d["rms"] for d in diffs])}
    # diffractor table (outside label windows only), picks for figures; both withheld from the
    # public copy (PRIVATE_KEYS in m36_gprsd_common) and kept in the local full file only
    out["diffractors"] = [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()
                           if not k.startswith("_") and k != "u0_tau_a"} for d in diffs]
    out["picks"] = [{"file_id": d["file_id"], "u": d["_u"].tolist(), "t": [round(x, 3) for x in d["_t"]]}
                    for d in diffs]
    out["profile_dz_median_by_tauz"] = [
        (float(tz), None if not np.isfinite(p0["dz"][:, j]).any() else float(np.nanmedian(p0["dz"][:, j])))
        for j, tz in enumerate(TAUZ_GRID)]
    out["profile_total_by_radius"] = {str(r): [None if not np.isfinite(v) else round(float(v), 4)
                                               for v in prof[r]["total"]] for r in R_LIST}
    out["elapsed_s"] = round(time.time() - t0, 1)
    p, pf = write_json_split(out_name, out)
    print(json.dumps({"status": "ok", "n_pass": len(diffs), "n_roads": out["n_roads_with_pass"],
                      "out": str(p), "out_full": str(pf)}))


if __name__ == "__main__":
    main()
