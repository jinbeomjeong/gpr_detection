"""M2-9 C follow-up: where and why does pick tracking of true 4TU apexes stop? (diagnosis)

Review labels (docs/processing/m29c_apex_review.csv, type T) are used ONLY to
choose the cases to diagnose: the 15 true apexes whose curvature QC is
``unknown`` and, as a comparison group, the 26 true apexes with a known
curvature. No threshold of any new rule is chosen here (user decision
2026-11-07: thresholds come from synthetic reproduction and physics).
Ground-truth depth tables and M2-8 matching results are NOT read.

Per case (same path as scripts/eval_depth_4tu.py, DepthConfig defaults):
  box = candidate_box, picks = extract_picks(apex_hint = candidate),
  QC = eval_depth_4tu.curvature_qc (spec_eval_4tu v1.4 3-5).
The tracker of spec_depth 5-3 (picks._track_one_side) is replayed step by
step with the same limits; the replay must reproduce the picks exactly
(asserted). For each side it records the stop rule that fired
(box_edge / a_max / amplitude / reverse / window) and the state at the
failing step: amplitude ratio E/E(t_a), time step dtau, whether the window
argmax sat on the window top (k == lo) or bottom (k == hi).

Probes after the stop (diagnostic only, never used by the depth path):
  * continuation: keep walking 0.20 m past the stop with the stop rules off
    (same window); net time change and how many of those steps would have
    passed both rules -> one-step jitter vs a lasting trend.
  * meta-v wing: envelope maximum within +-0.5 period of the straight-ray
    meta-v hyperbola through the apex candidate at |x - x_a| = 0.1..0.8 m,
    relative to E(t_a) and to the noise sigma of picks.noise_sigma.
  * apex offset: v fixed to v_meta, (x0, t0) least-squares on the baseline
    picks (soft-L1) -> x0 - x_a. Also the local minimum-time trace of the
    envelope ridge within +-0.20 m (window +-0.5 period around t_a).

Outputs
  docs/processing/m29c_track_cases.csv      one row per case and side
  docs/processing/m29c_track_summary.json   counts by stop rule / group
  docs/processing/m29c_track_unknown.png    montage of the 15 unknown T
  docs/processing/m29c_track_known.png      montage of 8 known T (comparison)
  data/processing/m29c_track_gallery/<blind_id>.png  per-case panels (git-excluded)

Run: conda run -n gpr-dev python -I analysis/processing/m29c_track_diagnose.py

Code version: the replay and the counterfactuals (``track_reverse_ns``) are
the **v1.8** tracker (commit f179e23). spec_depth v1.9 replaced that rule, so
this script stops on the v1.9 code; to reproduce the committed outputs import
the v1.8 package (``git worktree add <dir> f179e23``, ``PYTHONPATH=<dir>\\src``,
without ``-I``).
"""

import json
import math
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))

from dataclasses import replace  # noqa: E402

from eval_depth_4tu import candidate_box, classify_curvature, curvature_qc  # noqa: E402

from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import estimate_depth, prepare_depth_inputs  # noqa: E402
from gpr_detection.depth.hyperbola import fit_curvature, hyperbola_time  # noqa: E402
from gpr_detection.depth.picks import (  # noqa: E402
    ApexHint,
    extract_picks,
    noise_sigma,
)
from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.radargram import velocity_from_eps  # noqa: E402
from gpr_detection.preprocess.filters import agc  # noqa: E402
from gpr_detection.preprocess.signals import parabolic_peak, pick_signal  # noqa: E402

V18_COMMIT = "f179e23"
if "track_reverse_ns" not in {f.name for f in fields(DepthConfig)}:
    raise SystemExit(
        "m29c_track_diagnose.py replays the v1.8 tracker; this gpr_detection is "
        f"v1.9. Import the package of commit {V18_COMMIT} (see docstring)."
    )

BASE = Path(r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015")
DOCS = PROJECT / "docs" / "processing"
GALLERY = PROJECT / "data" / "processing" / "m29c_track_gallery"
REVIEW = DOCS / "m29c_apex_review.csv"

PROBE_CONTINUE_M = 0.20
WING_OFFSETS_M = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
LOCAL_APEX_HALF_M = 0.20
KNOWN_MONTAGE = 8


@dataclass
class Limits:
    back: float
    forward: float
    reverse: float
    stop_amp: float
    max_steps: int


def replay_side(signal, cand_trace, cand_k, cand_amp, step, i_lo, i_hi, lim, dt):
    """Replay of picks._track_one_side with a record of the stop step."""
    picks = []
    prev = cand_k
    n = signal.shape[0]
    for dist in range(1, lim.max_steps + 1):
        tr = cand_trace + step * dist
        if not i_lo <= tr <= i_hi:
            return picks, {"reason": "box_edge", "dist": dist}
        lo = max(0, math.floor(prev - lim.back))
        hi = min(n - 1, math.ceil(prev + lim.forward))
        if hi <= lo:
            return picks, {"reason": "window", "dist": dist}
        col = signal[:, tr]
        kp = lo + int(np.argmax(col[lo : hi + 1]))
        k, a = parabolic_peak(col, kp)
        info = {
            "dist": dist,
            "trace_local": tr,
            "k": k,
            "prev_k": prev,
            "amp_ratio": a / cand_amp,
            "dtau_ns": (k - prev) * dt,
            "at_top": kp == lo,
            "at_bottom": kp == hi,
        }
        if a < lim.stop_amp:
            return picks, {"reason": "amplitude", **info}
        if k < prev - lim.reverse:
            return picks, {"reason": "reverse", **info}
        picks.append((tr, k, a))
        prev = k
    return picks, {"reason": "a_max", "dist": lim.max_steps + 1}


def continue_probe(signal, start_trace, prev_k, step, i_lo, i_hi, lim, n_steps, dt, cand_amp):
    """Walk on past a stop with the stop rules off (same window)."""
    ks, amps, passed = [], [], 0
    prev = prev_k
    n = signal.shape[0]
    for j in range(n_steps):
        tr = start_trace + step * j
        if not i_lo <= tr <= i_hi:
            break
        lo = max(0, math.floor(prev - lim.back))
        hi = min(n - 1, math.ceil(prev + lim.forward))
        col = signal[:, tr]
        kp = lo + int(np.argmax(col[lo : hi + 1]))
        k, a = parabolic_peak(col, kp)
        ok = a >= lim.stop_amp and k >= prev - lim.reverse
        passed += int(ok)
        ks.append(k)
        amps.append(a / cand_amp)
        prev = k
    if not ks:
        return {"cont_steps": 0}
    return {
        "cont_steps": len(ks),
        "cont_pass": passed,
        "cont_net_dtau_ns": (ks[-1] - prev_k) * dt,
        "cont_min_dtau_ns": (min(ks) - prev_k) * dt,
        "cont_amp_med": float(np.median(amps)),
    }


def wing_profile(signal, apex_trace_local, apex_tau, t_zero, v_m, dx, dt, period, step):
    """Envelope max near the meta-v straight-ray wing at fixed offsets."""
    out = {}
    d = v_m * apex_tau / 2.0
    n_s, n_tr = signal.shape
    for a in WING_OFFSETS_M:
        tr = apex_trace_local + step * round(a / dx)
        if not 0 <= tr < n_tr:
            out[a] = math.nan
            continue
        tau = 2.0 / v_m * math.hypot(d, a)
        k_c = (tau + t_zero) / dt
        lo = max(0, math.floor(k_c - 0.5 * period / dt))
        hi = min(n_s - 1, math.ceil(k_c + 0.5 * period / dt))
        out[a] = float(signal[lo : hi + 1, tr].max()) if hi > lo else math.nan
    return out


def fixed_v_apex(x, tau, x_a, tau_a, v_m):
    """(x0, t0) with v fixed, soft-L1: where the picks say the apex is."""
    if len(x) < 4:
        return math.nan, math.nan
    sol = least_squares(
        lambda p: hyperbola_time(x, p[0], p[1], v_m, 0.0) - tau,
        [x_a, tau_a],
        loss="soft_l1",
        f_scale=0.1,
    )
    return float(sol.x[0]), float(sol.x[1])


def local_ridge_apex(signal, apex_trace_local, apex_k, dx, dt, period):
    """Trace of the earliest local envelope peak within +-0.5 period, +-0.2 m."""
    half = round(LOCAL_APEX_HALF_M / dx)
    n_s, n_tr = signal.shape
    lo = max(1, math.floor(apex_k - 0.5 * period / dt))
    hi = min(n_s - 2, math.ceil(apex_k + 0.5 * period / dt))
    best = None
    for tr in range(max(0, apex_trace_local - half), min(n_tr, apex_trace_local + half + 1)):
        col = signal[:, tr]
        kp = lo + int(np.argmax(col[lo : hi + 1]))
        if kp in (lo, hi):
            continue
        k, _ = parabolic_peak(col, kp)
        if best is None or k < best[1]:
            best = (tr, k)
    if best is None:
        return math.nan, math.nan
    return (best[0] - apex_trace_local) * dx, (best[1] - apex_k) * dt


def qc_of(rg, inputs, box, hint, config):
    """Picks + QC verdict for a counterfactual config / hint (diagnostic only)."""
    picks = extract_picks(
        inputs.branch.background_removed,
        box,
        inputs.zero,
        dt_ns=rg.dt_ns,
        dx_m=rg.dx_m,
        f_c_hz=rg.f_c_hz,
        config=config,
        apex_hint=hint,
        period_px=inputs.branch.period_px,
    )
    v_m = velocity_from_eps(rg.eps_r_meta)
    curv = fit_curvature(picks, radius_m=0.0, air_gap="ground", v_fallback=v_m, config=config)
    qc = classify_curvature(curv, picks, v_m)
    return picks, qc


def recentred(rg, inputs, box, hint, config, n_iter=3):
    """CF1: hint moved onto the found apex until it stops moving (diagnostic)."""
    picks, qc = qc_of(rg, inputs, box, hint, config)
    for _ in range(n_iter):
        if picks.apex is None:
            break
        new = ApexHint(trace=picks.apex.trace, sample=picks.apex.k)
        if (new.trace, new.sample) == (hint.trace, hint.sample):
            break
        hint = new
        picks, qc = qc_of(rg, inputs, box, hint, config)
    return picks, qc


def hint_top_stats(signal, searched, hint, config, dt, period):
    """Where the hint window top sits relative to the event (diagnostic)."""
    half_s = config.hint_periods * period / dt
    k_top = max(searched.k0, math.ceil(hint.sample - half_s))
    j_h = hint.trace - searched.i0
    above = 0
    n = 0
    for tr in range(max(0, j_h - config.hint_traces), min(signal.shape[1], j_h + config.hint_traces + 1)):
        lo = max(0, math.floor(hint.sample - 2.0 * half_s - half_s))
        hi = min(signal.shape[0] - 1, math.ceil(hint.sample + half_s))
        kp = lo + int(np.argmax(signal[lo : hi + 1, tr]))
        n += 1
        above += int(kp < k_top)
    return k_top, above, n


def diagnose(rg, inputs, row, config):
    trace, sample = round(float(row.trace)), float(row.sample)
    box = candidate_box(rg, trace, sample, float(row.tau_ns))
    hint = ApexHint(trace=trace, sample=sample)
    result = estimate_depth(rg, box, config, inputs=inputs, apex_hint=hint)
    qc = curvature_qc(rg, box, inputs, hint, result, config)
    section = inputs.branch.background_removed
    picks = extract_picks(
        section,
        box,
        inputs.zero,
        dt_ns=rg.dt_ns,
        dx_m=rg.dx_m,
        f_c_hz=rg.f_c_hz,
        config=config,
        apex_hint=hint,
        period_px=inputs.branch.period_px,
    )
    dt, dx = rg.dt_ns, rg.dx_m
    period = 1e9 / rg.f_c_hz
    v_m = velocity_from_eps(rg.eps_r_meta)
    base = {
        "blind_id": row.blind_id,
        "activity_id": row.activity_id,
        "site": int(row.site),
        "path": row.path,
        "source": row.source,
        "hint_trace": trace,
        "hint_sample": sample,
        "d_cand_m": row.d_cand_m,
        "confidence": row.confidence,
        "qc_verdict": str(qc.verdict.value if hasattr(qc.verdict, "value") else qc.verdict),
        "qc_reason": qc.reason,
        "qc_r": qc.r,
        "n_picks": picks.n_picks,
        "n_left": picks.n_left,
        "n_right": picks.n_right,
        "flags": ";".join(picks.flags),
        "dt_ns": dt,
        "v_meta": v_m,
        "box_i0": box.i0,
        "box_i1": box.i1,
        "box_k0": box.k0,
        "box_k1": box.k1,
    }
    if picks.apex is None:
        base["side"] = "none"
        base["stop_reason"] = "no_apex_candidate"
        return [base], None
    searched = picks.box
    signal = pick_signal(section[:, searched.trace_slice], config.pick_mode)
    cand_trace = picks.apex.trace - searched.i0
    lim = Limits(
        back=config.track_back_ns / dt,
        forward=(2.0 * dx / config.v_min + config.track_back_ns) / dt,
        reverse=config.track_reverse_ns / dt,
        stop_amp=config.track_stop_rel * picks.apex.amplitude,
        max_steps=math.floor(config.a_max_m / dx + 1e-9),
    )
    sigma_n = noise_sigma(section, searched)
    t_zero = inputs.zero.t_zero_ns
    apex_tau = picks.apex.k * dt - t_zero
    x0_fix, t0_fix = fixed_v_apex(picks.x_m, picks.tau_ns, picks.apex_x_m, apex_tau, v_m)
    ridge_dx, ridge_dt = local_ridge_apex(signal, cand_trace, picks.apex.k, dx, dt, period)
    base.update(
        apex_trace=picks.apex.trace,
        apex_k=picks.apex.k,
        apex_moved_traces=picks.apex.trace - trace,
        apex_moved_ns=(picks.apex.k - sample) * dt,
        apex_amp=picks.apex.amplitude,
        apex_amp_over_sigma=picks.apex.amplitude / sigma_n if sigma_n > 0 else math.nan,
        sigma_n=sigma_n,
        apex_tau_ns=apex_tau,
        x0_fixv_minus_xa_m=x0_fix - picks.apex_x_m,
        t0_fixv_minus_taua_ns=t0_fix - apex_tau,
        ridge_apex_dx_m=ridge_dx,
        ridge_apex_dt_ns=ridge_dt,
        searched_k0=searched.k0,
        apex_minus_k0=picks.apex.k - searched.k0,
    )
    k_top_h, n_above, n_h = hint_top_stats(signal, searched, hint, config, dt, period)
    base.update(
        hint_top_k=k_top_h,
        apex_minus_hint_top=picks.apex.k - k_top_h,
        hint_traces_peak_above_top=n_above,
        hint_traces_n=n_h,
    )
    cfs = {
        "cf_recentre": recentred(rg, inputs, box, hint, config),
        "cf_rev03": qc_of(rg, inputs, box, hint, replace(config, track_reverse_ns=0.3)),
        "cf_both": recentred(rg, inputs, box, hint, replace(config, track_reverse_ns=0.3)),
    }
    for name, (cp, cq) in cfs.items():
        base[f"{name}_n_left"] = cp.n_left
        base[f"{name}_n_right"] = cp.n_right
        base[f"{name}_verdict"] = str(cq.verdict)
        base[f"{name}_reason"] = cq.reason
        base[f"{name}_r"] = cq.r
        if cp.apex is not None:
            base[f"{name}_apex_dtr"] = cp.apex.trace - picks.apex.trace
            base[f"{name}_apex_dns"] = (cp.apex.k - picks.apex.k) * dt
    rows, sides = [], {}
    for name, step in (("left", -1), ("right", +1)):
        side_picks, stop = replay_side(
            signal, cand_trace, picks.apex.k, picks.apex.amplitude, step,
            0, searched.n_traces - 1, lim, dt,
        )
        expected = picks.n_left if step < 0 else picks.n_right
        assert len(side_picks) == expected, (row.blind_id, name, len(side_picks), expected)
        rec = dict(base)
        rec["side"] = name
        rec["n_side"] = len(side_picks)
        rec["stop_reason"] = stop["reason"]
        rec["stop_dist_traces"] = stop["dist"]
        rec["stop_offset_m"] = stop["dist"] * dx
        for key in ("amp_ratio", "dtau_ns", "at_top", "at_bottom"):
            rec[f"stop_{key}"] = stop.get(key, math.nan)
        if stop["reason"] in ("amplitude", "reverse"):
            cont = continue_probe(
                signal, stop["trace_local"], stop["prev_k"], step, 0, searched.n_traces - 1,
                lim, round(PROBE_CONTINUE_M / dx), dt, picks.apex.amplitude,
            )
            rec.update(cont)
            # local time slope before the stop (last 3 picks), ns per trace
            if len(side_picks) >= 3:
                ks = [p[1] for p in side_picks[-3:]]
                rec["slope_before_ns_per_tr"] = (ks[-1] - ks[0]) / 2 * dt
        wp = wing_profile(signal, cand_trace, apex_tau, t_zero, v_m, dx, dt, period, step)
        for a, e in wp.items():
            rec[f"wing_E_rel_{a:.1f}"] = e / picks.apex.amplitude
            rec[f"wing_E_snr_{a:.1f}"] = e / sigma_n if sigma_n > 0 else math.nan
        if side_picks:
            amps = np.array([p[2] for p in side_picks]) / picks.apex.amplitude
            rec["side_amp_min"] = float(amps.min())
            rec["side_amp_last"] = float(amps[-1])
        rows.append(rec)
        sides[name] = (side_picks, stop)
    figinfo = {
        "signal": signal,
        "searched": searched,
        "cand_trace": cand_trace,
        "apex_k": picks.apex.k,
        "apex_amp": picks.apex.amplitude,
        "sides": sides,
        "t_zero": t_zero,
        "apex_tau": apex_tau,
        "v_m": v_m,
        "x0_fix": x0_fix - picks.apex_x_m,
        "t0_fix": t0_fix,
        "sigma_n": sigma_n,
        "stop_amp": lim.stop_amp,
    }
    return rows, figinfo


def draw_case(axes, rg, inputs, row, info, recs):
    ax_img, ax_amp = axes
    dt, dx = rg.dt_ns, rg.dx_m
    title = f"{row.blind_id} {row.activity_id} {row.path} tr{int(row.trace)} | {recs[0]['qc_reason'] or recs[0]['qc_verdict']}"
    if info is None:
        ax_img.set_title(title + " | no apex", fontsize=7)
        ax_amp.axis("off")
        return
    searched = info["searched"]
    i_a = searched.i0 + info["cand_trace"]
    img = agc(inputs.branch.background_removed, dt, rg.f_c_hz)
    half = round(1.0 / dx)
    t_a = info["apex_k"] * dt
    k0 = max(int((t_a - 5) / dt), 0)
    k1 = min(int((t_a + 14) / dt), img.shape[0] - 1)
    i0, i1 = max(i_a - half, 0), min(i_a + half, img.shape[1] - 1)
    sub = img[k0 : k1 + 1, i0 : i1 + 1]
    lim = float(np.percentile(np.abs(sub), 99)) or 1.0
    ext = ((i0 - i_a) * dx, (i1 - i_a) * dx, k1 * dt, k0 * dt)
    ax_img.imshow(sub, cmap="gray", vmin=-lim, vmax=lim, aspect="auto", extent=ext)
    colors = {"left": "#2a78d6", "right": "#eb6834"}
    for name, (sp, stop) in info["sides"].items():
        if sp:
            xs = [(searched.i0 + p[0] - i_a) * dx for p in sp]
            ts = [p[1] * dt for p in sp]
            ax_img.plot(xs, ts, ".", ms=2.5, color=colors[name])
        if "k" in stop:
            xs = (searched.i0 + stop["trace_local"] - i_a) * dx
            ax_img.plot([xs], [stop["k"] * dt], "x", ms=7, mew=1.6, color=colors[name])
    xx = np.linspace(-1.0, 1.0, 201)
    tt = hyperbola_time(xx, 0.0, info["apex_tau"], info["v_m"], 0.0) + info["t_zero"]
    ax_img.plot(xx, tt, ls="--", lw=0.8, color="#3fb08a")
    ax_img.plot([0], [t_a], marker="o", mfc="none", color="#e34948", ms=7)
    bx = ((searched.i0 - i_a) * dx, (searched.i1 - i_a) * dx)
    for b in bx:
        ax_img.axvline(b, color="#999999", lw=0.6, ls=":")
    ax_img.set_xlim(-1.0, 1.0)
    ax_img.set_title(title, fontsize=7)
    ax_img.tick_params(labelsize=6)
    ax_img.set_xlabel("x - x_a (m)", fontsize=6)
    ax_img.set_ylabel("t (ns)", fontsize=6)
    # amplitude along picks and along the meta-v wing
    for name, (sp, stop) in info["sides"].items():
        sgn = -1 if name == "left" else 1
        if sp:
            ax_amp.plot(
                [(searched.i0 + p[0] - i_a) * dx for p in sp],
                [p[2] / info["apex_amp"] for p in sp],
                ".", ms=2.5, color=colors[name],
            )
        if "amp_ratio" in stop:
            ax_amp.plot([sgn * stop["dist"] * dx], [stop["amp_ratio"]], "x", ms=7, mew=1.6, color=colors[name])
        rec = next(r for r in recs if r["side"] == name)
        ax_amp.plot(
            [sgn * a for a in WING_OFFSETS_M],
            [rec[f"wing_E_rel_{a:.1f}"] for a in WING_OFFSETS_M],
            "s", mfc="none", ms=4, color=colors[name],
        )
    ax_amp.axhline(0.2, color="#555555", lw=0.8, ls="--")
    ax_amp.axhline(5 * info["sigma_n"] / info["apex_amp"], color="#c9a227", lw=0.8, ls=":")
    ax_amp.set_xlim(-1.0, 1.0)
    ax_amp.set_ylim(0, 1.6)
    ax_amp.tick_params(labelsize=6)
    ax_amp.set_xlabel("x - x_a (m)", fontsize=6)
    ax_amp.set_ylabel("E / E(t_a)", fontsize=6)


def main() -> None:
    config = DepthConfig()
    review = pd.read_csv(REVIEW, dtype={"activity_id": str})
    cases = review[review["type"] == "T"].copy()
    GALLERY.mkdir(parents=True, exist_ok=True)
    rows, infos, loaded = [], {}, {}
    for (act, path), group in cases.groupby(["activity_id", "path"], sort=True):
        rg = load_fourtu_radargram(BASE, act, path)
        inputs = prepare_depth_inputs(rg, config)
        loaded[(act, path)] = (rg, inputs)
        for row in group.itertuples():
            recs, info = diagnose(rg, inputs, row, config)
            for r in recs:
                r["review_curv_flag"] = row.curv_flag
                r["review_unknown_reason"] = row.unknown_reason if isinstance(row.unknown_reason, str) else ""
            rows.extend(recs)
            infos[row.blind_id] = (row, info, recs)
            fig, axes = plt.subplots(2, 1, figsize=(5.0, 6.0), height_ratios=[2.2, 1], constrained_layout=True)
            draw_case(axes, rg, inputs, row, info, recs)
            fig.savefig(GALLERY / f"{row.blind_id}.png", dpi=120)
            plt.close(fig)
    out = pd.DataFrame(rows)
    out["group"] = np.where(out["qc_verdict"] == "unknown", "T_unknown", "T_known")
    out.to_csv(DOCS / "m29c_track_cases.csv", index=False, float_format="%.4f")

    # Montages
    unk = [b for b, (r, i, recs) in infos.items() if recs[0]["qc_verdict"] == "unknown"]
    known = sorted(b for b in infos if b not in unk)
    rng = np.random.default_rng(20261107)
    known_pick = sorted(rng.choice(known, size=min(KNOWN_MONTAGE, len(known)), replace=False))
    for name, ids, ncol in (("unknown", sorted(unk), 5), ("known", known_pick, 4)):
        nrow = math.ceil(len(ids) / ncol)
        fig = plt.figure(figsize=(3.4 * ncol, 4.6 * nrow), constrained_layout=True)
        sub = fig.subfigures(nrow, ncol)
        for sf, b in zip(np.ravel(sub), ids, strict=False):
            row, info, recs = infos[b]
            axes = sf.subplots(2, 1, height_ratios=[2.2, 1])
            rg, inputs = loaded[(row.activity_id, row.path)]
            draw_case(axes, rg, inputs, row, info, recs)
        fig.suptitle(
            f"M2-9 C tracking diagnosis, true apexes with curvature {name} (4TU, CC0 1.0)\n"
            "dots: picks (blue left, orange right); x: failing step; green dashed: meta-v;\n"
            "squares: envelope near meta-v wing; dashed 0.2: stop rule; dotted: 5 sigma_n",
            fontsize=9,
        )
        fig.savefig(DOCS / f"m29c_track_{name}.png", dpi=100)
        plt.close(fig)

    sides = out[out["side"].isin(["left", "right"])]
    summary = {
        "n_cases": int(cases.shape[0]),
        "known_montage_seed": 20261107,
        "known_montage_ids": [str(b) for b in known_pick],
        "stop_reason_by_group": pd.crosstab(sides["group"], sides["stop_reason"]).to_dict("index"),
        "qc_reason_by_group": {
            f"{g}|{r}": int(n)
            for (g, r), n in out.drop_duplicates("blind_id")
            .groupby("group")["qc_reason"]
            .value_counts()
            .items()
        },
    }
    (DOCS / "m29c_track_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    cols = [
        "blind_id", "group", "qc_reason", "side", "n_side", "stop_reason", "stop_dist_traces",
        "stop_amp_ratio", "stop_dtau_ns", "stop_at_top", "cont_pass", "cont_net_dtau_ns",
        "x0_fixv_minus_xa_m", "ridge_apex_dx_m", "apex_moved_traces", "apex_moved_ns",
    ]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print(sides[sides["group"] == "T_unknown"][cols].to_string(index=False))
    print(json.dumps(summary["stop_reason_by_group"], default=str))
    one = out.drop_duplicates("blind_id")
    cf_cols = [
        "blind_id", "group", "qc_reason", "n_left", "n_right", "apex_moved_traces",
        "apex_moved_ns", "apex_minus_hint_top", "hint_traces_peak_above_top",
        "cf_recentre_n_left", "cf_recentre_n_right", "cf_recentre_verdict", "cf_recentre_r",
        "cf_recentre_apex_dtr", "cf_recentre_apex_dns",
        "cf_rev03_n_left", "cf_rev03_n_right", "cf_rev03_verdict",
        "cf_both_n_left", "cf_both_n_right", "cf_both_verdict", "cf_both_r",
    ]
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.max_rows", 200):
        print(one[cf_cols].sort_values(["group", "blind_id"]).to_string(index=False))
    for name in ("cf_recentre", "cf_rev03", "cf_both"):
        summary[f"{name}_verdict_by_group"] = pd.crosstab(one["group"], one[f"{name}_verdict"]).to_dict("index")
    (DOCS / "m29c_track_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
