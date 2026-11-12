"""M2-10 K7 generator-defect probe (development evidence, already-seen data, no judgement).

Three read-only parts, none of which changes the generator
(``fourtu_apex_candidate_select.py`` is imported unchanged):

A. S1g (gprMax, already seen in the K1 judgement): for each primary apex of
   S1G_C1 / D060 / D100, the accepted items near it with the *raw* envelope E at
   the item, the detection SNR, wings, rank and fate; the processed envelope's
   local maxima on the apex trace; and the fates under offline replays of the
   ranking step (current, K7-a1 lobe merge, K7-a2 NMS representative by raw E,
   K7-b1 no per-line cap). The replays only reorder ``line["_accepted_all"]``.

B. 4TU (already seen, exploration only, NOT a tuning input): replay of the
   ranking step on ``data/processing/m210d_generator_items.csv`` (550 lines with
   ground truth). Counts of the early-lobe NMS pattern, cap losses, the kept count
   without the cap, and the crest-move distance of kept items.

C. Pure-function Monte Carlo of the crest move on a flat deep apex: earliest
   point (current ``crest_apex``) vs the tau^2-quadratic vertex of the crest
   points (K7-c1 prototype). Synthetic Ricker hyperbola, band-passed white noise,
   seeds 20261112 + n.

D. Prototype variants run through the unchanged generator with ``crest_apex`` /
   the per-line cap swapped in-process (a1 start snap, b1 no cap, c1 tau^2 vertex):
   S1g kept-in-window per pipe, and the 4TU kept-set change on the m210d lines
   (``K7_PROBE_LINES=n`` limits the 4TU lines).

Output: ``docs/processing/m210_k7_probe.json``.

Run (single process, BelowNormal, one core):
    conda run -n gpr-dev python -I analysis/processing/m210_k7_probe.py
"""

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "processing"))
sys.path.insert(0, str(ROOT / "scripts"))

import fourtu_apex_candidate_select as gen  # noqa: E402
import m210_s1g_pipeline as s1g  # noqa: E402

from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402
from gpr_detection.preprocess.signals import parabolic_peak  # noqa: E402

OUT = ROOT / "docs" / "processing" / "m210_k7_probe.json"
S1G_ROOT = ROOT / "data" / "synthetic_gprmax_s1g"
ITEMS = ROOT / "data" / "processing" / "m210d_generator_items.csv"
LINES = ROOT / "data" / "processing" / "m210d_generator_lines.csv"
ID_COLUMNS = {"activity_id": str, "path": str, "line_id": str}
F_C_4TU = 500e6
SEED = 20261112


# ----------------------------------------------------------------- ranking replays


def rank_key(s):
    return (-min(s["n_left"], s["n_right"]), -s["snr"])


def replay(items, dx, dt, f_c, mode, cap=gen.MAX_APEX_CANDIDATES_PER_LINE):
    """Fates of ``items`` (already in rank order) under a ranking variant.

    mode:
      "current": the generator's ``_rank_and_suppress``.
      "a1": lobe merge - when s is NMS-suppressed by kept o, o lies within +-3
            traces and in the early-lobe box of s ([t_s - 1.5/f_c, t_s - 0.5/f_c)),
            and E(s) > E(o), s takes o's slot (o becomes suppressed).
      "a2": NMS representative by raw E - when s is suppressed by kept o and
            E(s) > E(o), s takes o's slot (no lobe-box condition).
    ``cap=None`` removes the per-line cap (K7-b1).
    Returns the kept list (items) and a list of fates parallel to ``items``.
    """
    period = 1e9 / f_c
    kept, fate = [], [None] * len(items)
    for i, s in enumerate(items):
        if cap is not None and len(kept) >= cap:
            fate[i] = "cap6"
            continue
        win = next(
            (
                n
                for n, o in enumerate(kept)
                if abs(s["trace"] - o["trace"]) * dx <= gen.NMS_M
                and abs(s["t_ns"] - o["t_ns"]) <= gen.NMS_NS
            ),
            None,
        )
        if win is None:
            kept.append(s)
            fate[i] = "kept"
            continue
        o = kept[win]
        swap = False
        if mode in ("a1", "a2") and s.get("E") is not None and o.get("E") is not None:
            if s["E"] > o["E"]:
                if mode == "a2":
                    swap = True
                else:
                    in_box = (
                        abs(s["trace"] - o["trace"]) <= 3
                        and s["t_ns"] - 1.5 * period <= o["t_ns"] < s["t_ns"] - 0.5 * period
                    )
                    swap = in_box
        if swap:
            kept[win] = s
            fate[i] = "kept"
            j = items.index(o)
            fate[j] = "nms"
        else:
            fate[i] = "nms"
    return kept, fate


# ----------------------------------------------------------------------- part A


def local_maxima(col, k_lo, k_hi):
    seg = col[k_lo:k_hi]
    out = []
    for i in range(1, len(seg) - 1):
        if seg[i] >= seg[i - 1] and seg[i] > seg[i + 1]:
            out.append(k_lo + i)
    return out


def part_a():
    result = {}
    for model in ("S1G_C1", "S1G_D060", "S1G_D100"):
        rg = s1g.load_line(model, S1G_ROOT)
        meta = json.loads((S1G_ROOT / f"{model}.json").read_text(encoding="utf-8"))
        inputs = prepare_depth_inputs(rg, DepthConfig())
        line, kept, E = gen.apex_candidates_for_radargram(
            rg, inputs, act="S1G", path=model
        )
        dt, dx = rg.dt_ns, rg.dx_m
        f_c = rg.f_c_hz
        acc = [dict(s) for s in line["_accepted_all"]]
        for s in acc:
            s["E"] = float(E[int(round(s["sample"])), s["trace"]])
        # Sanity: offline current replay == generator fates.
        _, fate_cur = replay(acc, dx, dt, f_c, "current")
        gen_fates = [f["fate"] for f in line["_accepted_fate"]]
        same = fate_cur == gen_fates
        pipes = meta.get("pipes") or [
            {"apex": meta["apex"], "apex_trace_index": meta["apex"]["trace_index"]}
        ]
        variants = {}
        for name, mode, cap in (
            ("current", "current", gen.MAX_APEX_CANDIDATES_PER_LINE),
            ("a1", "a1", gen.MAX_APEX_CANDIDATES_PER_LINE),
            ("a2", "a2", gen.MAX_APEX_CANDIDATES_PER_LINE),
            ("b1", "current", None),
            ("a1_b1", "a1", None),
        ):
            kept_v, _ = replay(acc, dx, dt, f_c, mode, cap)
            per_pipe = []
            for p in pipes:
                ja = p["apex_trace_index"]
                ta = p["apex"]["envelope_peak_ns"]
                hit = [
                    s
                    for s in kept_v
                    if abs(s["trace"] - ja) <= 3 and abs(s["t_ns"] - ta) <= 0.5e9 / f_c
                ]
                per_pipe.append(
                    {
                        "apex_trace": ja,
                        "t_apex_ns": round(ta, 3),
                        "kept_in_window": len(hit),
                        "best": None
                        if not hit
                        else {"trace": hit[0]["trace"], "t_ns": hit[0]["t_ns"]},
                    }
                )
            variants[name] = {"n_kept": len(kept_v), "pipes": per_pipe}
        pipe_rows = []
        for p in pipes:
            ja = p["apex_trace_index"]
            ta = p["apex"]["envelope_peak_ns"]
            near = []
            for rank, (s, f) in enumerate(zip(acc, gen_fates)):
                if abs(s["trace"] - ja) <= 15 and abs(s["t_ns"] - ta) <= 2.5:
                    rec = next(
                        p
                        for p in line["_peaks"]
                        if p["status"] == "accepted"
                        and p["trace"] == s["trace"]
                        and round(p["sample"], 2) == s["sample"]
                    )
                    best, _ = crest_walk_detail(E, rec["trace0"], rec["sample0"], dt, dx)
                    col_k = int(round(s["sample"]))
                    colv = E[:, s["trace"]]
                    near.append(
                        {
                            "rank": rank,
                            "trace": s["trace"],
                            "t_ns": s["t_ns"],
                            "dt_from_apex_ns": round(s["t_ns"] - ta, 3),
                            "snr": s["snr"],
                            "E_raw": round(s["E"], 6),
                            "level_implied": round(s["E"] / s["snr"], 6)
                            if s["snr"]
                            else None,
                            "n_left": s["n_left"],
                            "n_right": s["n_right"],
                            "fate": f,
                            "start_trace": rec["trace0"],
                            "start_t_ns": round(rec["sample0"] * dt, 3),
                            "crest_end_edge": best["edge"],
                            "end_is_local_max": bool(colv[col_k] >= colv[col_k - 1] and colv[col_k] >= colv[col_k + 1]),
                        }
                    )
            col = E[:, ja]
            k_lo = int(math.floor((ta - 3.0) / dt))
            k_hi = int(math.ceil((ta + 1.5) / dt))
            maxima = []
            for k in local_maxima(col, k_lo, k_hi):
                kk, a = parabolic_peak(col, k)
                maxima.append({"t_ns": round(kk * dt, 3), "E": round(a, 6)})
            emax = max(m["E"] for m in maxima) if maxima else None
            for m in maxima:
                m["rel"] = round(m["E"] / emax, 3) if emax else None
            pipe_rows.append(
                {
                    "apex_trace": ja,
                    "t_apex_ns": round(ta, 3),
                    "accepted_near": near,
                    "apex_trace_envelope_maxima": [m for m in maxima if m["rel"] >= 0.05],
                }
            )
        result[model] = {
            "dt_ns": dt,
            "n_accepted": len(acc),
            "n_kept": len(kept),
            "replay_current_equals_generator": same,
            "pipes": pipe_rows,
            "variants": variants,
        }
    return result


# ----------------------------------------------------------------------- part B


def part_b():
    items = pd.read_csv(ITEMS, dtype=ID_COLUMNS)
    lines = pd.read_csv(LINES, dtype=ID_COLUMNS)
    dt_of = {(r.activity_id, r.path): float(r.dt_ns) for r in lines.itertuples()}
    acc = items[items["stage"].isin(["kept", "accepted"])]
    dx = 0.02
    period = 1e9 / F_C_4TU
    out = {
        "n_lines": 0,
        "replay_matches_kept": 0,
        "n_accepted": 0,
        "n_kept_current": 0,
        "n_kept_no_cap": 0,
        "n_cap6": 0,
        "lines_cap_active": 0,
        "n_nms": 0,
        "n_nms_early_lobe_winner": 0,
        "lines_with_early_lobe_nms": 0,
    }
    moves = []
    for (act, path), g in acc.groupby(["activity_id", "path"], sort=False):
        dt = dt_of[(act, path)]
        rows = []
        for r in g.itertuples():
            rows.append(
                {
                    "trace": int(r.trace),
                    "t_ns": round(float(r.sample) * dt, 3),
                    "snr": float(r.snr),
                    "n_left": int(r.n_left),
                    "n_right": int(r.n_right),
                    "stage": r.stage,
                    "move": abs(int(r.trace) - int(r.peak_trace)),
                }
            )
        rows.sort(key=rank_key)  # stable: csv order inside ties
        kept, fate = replay(rows, dx, dt, F_C_4TU, "current")
        kept_nc, _ = replay(rows, dx, dt, F_C_4TU, "current", cap=None)
        out["n_lines"] += 1
        replay_kept = sorted((s["trace"], s["t_ns"]) for s in kept)
        file_kept = sorted((s["trace"], s["t_ns"]) for s in rows if s["stage"] == "kept")
        out["replay_matches_kept"] += int(replay_kept == file_kept)
        out["n_accepted"] += len(rows)
        out["n_kept_current"] += len(kept)
        out["n_kept_no_cap"] += len(kept_nc)
        out["n_cap6"] += fate.count("cap6")
        out["lines_cap_active"] += int("cap6" in fate)
        el = 0
        for s, f in zip(rows, fate):
            if f != "nms":
                continue
            out["n_nms"] += 1
            o = next(
                o
                for o in kept
                if abs(s["trace"] - o["trace"]) * dx <= gen.NMS_M
                and abs(s["t_ns"] - o["t_ns"]) <= gen.NMS_NS
            )
            if (
                abs(s["trace"] - o["trace"]) <= 3
                and s["t_ns"] - 1.5 * period <= o["t_ns"] < s["t_ns"] - 0.5 * period
            ):
                el += 1
        out["n_nms_early_lobe_winner"] += el
        out["lines_with_early_lobe_nms"] += int(el > 0)
        moves.extend(s["move"] for s in rows if s["stage"] == "kept")
    m = np.array(moves)
    out["kept_crest_move_traces"] = {
        "n": int(m.size),
        "median": float(np.median(m)),
        "p90": float(np.percentile(m, 90)),
        "frac_ge4": round(float(np.mean(m >= 4)), 4),
        "frac_ge10": round(float(np.mean(m >= 10)), 4),
    }
    return out


# ----------------------------------------------------------------------- part C


def ricker(t, f):
    a = (math.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def crest_points(E, j0, k0, dt, dx):
    """Points visited by the generator's crest walk (same rules as ``crest_apex``)."""
    e0 = float(E[int(round(k0)), j0])
    pts = [(j0, k0)]
    for step in (-1, 1):
        prev = k0
        for dist in range(1, int(0.5 / dx + 1e-9) + 1):
            j = j0 + step * dist
            if not 0 <= j < E.shape[1]:
                break
            lo = max(0, math.floor(prev - 1.0 / dt))
            hi = min(E.shape[0] - 1, math.ceil(prev + 1.0 / dt))
            col = E[:, j]
            k, a = parabolic_peak(col, lo + int(np.argmax(col[lo : hi + 1])))
            if a < gen.STOP_REL * e0:
                break
            pts.append((j, k))
            prev = k
    return pts


def crest_walk_detail(E, j0, k0, dt, dx):
    """Crest walk of ``crest_apex`` with, per visited point, whether the window
    argmax sat on the window edge (``lo``/``hi``, not the array end), and the
    returned earliest point (same tie rule: strictly earlier replaces)."""
    e0 = float(E[int(round(k0)), j0])
    best = {"trace": j0, "k": k0, "edge": "start"}
    visited = []
    for step in (-1, 1):
        prev = k0
        for dist in range(1, int(0.5 / dx + 1e-9) + 1):
            j = j0 + step * dist
            if not 0 <= j < E.shape[1]:
                break
            lo = max(0, math.floor(prev - 1.0 / dt))
            hi = min(E.shape[0] - 1, math.ceil(prev + 1.0 / dt))
            col = E[:, j]
            idx = lo + int(np.argmax(col[lo : hi + 1]))
            k, a = parabolic_peak(col, idx)
            if a < gen.STOP_REL * e0:
                break
            edge = "lo" if idx == lo and lo > 0 else "hi" if idx == hi else ""
            visited.append({"trace": j, "k": k, "edge": edge})
            if k < best["k"]:
                best = {"trace": j, "k": k, "edge": edge}
            prev = k
    return best, visited


_ORIG_CREST = gen.crest_apex
_ORIG_CAP = gen.MAX_APEX_CANDIDATES_PER_LINE
_STATE = {"t_z": None, "snap": False, "vertex": False}


def snap_up(col, k0):
    """K7-a1 prototype: climb the un-normalised envelope from the start sample to
    its local maximum (strictly increasing steps, so it stops)."""
    k = int(round(k0))
    n = len(col)
    while True:
        up = k + 1 < n and col[k + 1] > col[k]
        down = k >= 1 and col[k - 1] > col[k]
        if up and (not down or col[k + 1] >= col[k - 1]):
            k += 1
        elif down:
            k -= 1
        else:
            return k


def patched_crest(E, j0, k0, dt, dx):
    if _STATE["snap"]:
        k0, _ = parabolic_peak(E[:, j0], snap_up(E[:, j0], k0))
    if not _STATE["vertex"]:
        return _ORIG_CREST(E, j0, k0, dt, dx)
    pts = crest_points(E, j0, k0, dt, dx)
    j_e, k_e = min(pts, key=lambda p: p[1])
    jv = vertex_tau2(pts, j_e, dt, dx, _STATE["t_z"])
    if jv is None or jv == j_e:
        return j_e, k_e
    at = [p for p in pts if p[0] == jv]
    return (jv, at[0][1]) if at else (j_e, k_e)


VARIANTS = {
    "cur": (False, False, False),
    "a1": (True, False, False),
    "b1": (False, True, False),
    "c1": (False, False, True),
    "a1b1": (True, True, False),
    "a1c1": (True, False, True),
    "a1b1c1": (True, True, True),
}


def run_variant(rg, inputs, act, path, name):
    snap, nocap, vertex = VARIANTS[name]
    _STATE.update(snap=snap, vertex=vertex, t_z=inputs.zero.t_zero_ns)
    gen.crest_apex = patched_crest
    gen.MAX_APEX_CANDIDATES_PER_LINE = 10**9 if nocap else _ORIG_CAP
    try:
        return gen.apex_candidates_for_radargram(rg, inputs, act=act, path=path)
    finally:
        gen.crest_apex = _ORIG_CREST
        gen.MAX_APEX_CANDIDATES_PER_LINE = _ORIG_CAP


def part_d_s1g():
    out = {}
    for model in ("S1G_C1", "S1G_D060", "S1G_D100", "S1G_BGL"):
        rg = s1g.load_line(model, S1G_ROOT)
        meta = json.loads((S1G_ROOT / f"{model}.json").read_text(encoding="utf-8"))
        inputs = prepare_depth_inputs(rg, DepthConfig())
        pipes = meta.get("pipes") or (
            [{"apex": meta["apex"], "apex_trace_index": meta["apex"]["trace_index"]}]
            if meta.get("apex")
            else []
        )
        res = {}
        for name in VARIANTS:
            line, kept, _ = run_variant(rg, inputs, "S1G", model, name)
            per = []
            for p in pipes:
                ja, ta = p["apex_trace_index"], p["apex"]["envelope_peak_ns"]
                win = lambda s: abs(s["trace"] - ja) <= 3 and abs(s["t_ns"] - ta) <= 1.0  # noqa: E731
                per.append(
                    {
                        "apex_trace": ja,
                        "kept_in_window": sum(win(s) for s in kept),
                        "accepted_in_window": sum(win(s) for s in line.get("_accepted_all", [])),
                    }
                )
            res[name] = {
                "n_kept": len(kept),
                "n_accepted": len(line.get("_accepted_all", [])),
                "pipes": per,
            }
        out[model] = res
    return out


def part_d_4tu():
    """4TU (already seen, exploration only): kept sets under each variant."""
    lines = pd.read_csv(LINES, dtype=ID_COLUMNS)
    limit = int(os.environ.get("K7_PROBE_LINES", "0"))
    if limit:
        lines = lines.head(limit)
    names = list(VARIANTS)
    tot = {n: {"n_kept": 0, "lines_changed": 0, "added": 0, "removed": 0} for n in names}
    fp = {"n_kept": 0, "end_not_local_max": 0}
    n_lines = 0
    for r in lines.itertuples():
        rg = gen.load_fourtu_radargram(gen.BASE, r.activity_id, r.path)
        inputs = gen.prepare_depth_inputs(rg)
        if inputs.zero.t_zero_ns is None:
            continue
        n_lines += 1
        keys = {}
        for name in names:
            line, kept, E = run_variant(rg, inputs, r.activity_id, r.path, name)
            keys[name] = {(s["trace"], s["sample"]) for s in kept}
            if name == "cur" and E is not None:
                for s in kept:
                    col = E[:, s["trace"]]
                    k = int(round(s["sample"]))
                    fp["n_kept"] += 1
                    fp["end_not_local_max"] += int(
                        not (col[k] >= col[k - 1] and col[k] >= col[k + 1])
                    )
        dt = rg.dt_ns

        def near(p, q):
            return abs(p[0] - q[0]) <= 3 and abs(p[1] - q[1]) * dt <= 1.0

        for name in names:
            a, b = keys["cur"], keys[name]
            tot[name]["n_kept"] += len(b)
            tot[name]["lines_changed"] += int(a != b)
            tot[name]["added"] += len(b - a)
            tot[name]["removed"] += len(a - b)
            tot[name].setdefault("new_far", 0)
            tot[name].setdefault("lost_far", 0)
            tot[name]["new_far"] += sum(not any(near(q, p) for p in a) for q in b)
            tot[name]["lost_far"] += sum(not any(near(p, q) for q in b) for p in a)
    return {
        "n_lines": n_lines,
        "variants": tot,
        "cur_kept_end_fingerprint": fp,
        "note": "added/removed = exact (trace, sample) keys; new_far/lost_far = no counterpart within +-3 traces and +-1 ns",
    }

def vertex_tau2(pts, j_e, dt, dx, t_z):
    """K7-c1 prototype: tau^2 = a + b xi + c xi^2 on crest points |xi| <= 0.30 m of j_e."""
    xi = np.array([(p[0] - j_e) * dx for p in pts])
    tau = np.array([p[1] * dt - t_z for p in pts])
    sel = np.abs(xi) <= 0.30 + 1e-9
    if sel.sum() < 5 or np.any(tau[sel] <= 0):
        return None
    c2, c1, _ = np.polyfit(xi[sel], tau[sel] ** 2, 2)
    if c2 <= 0:
        return None
    x0 = -c1 / (2 * c2)
    if abs(x0) > 0.30 + 1e-9:
        return None
    return j_e + int(round(x0 / dx))


def part_c():
    dt, dx, f = 0.098, 0.02, 500e6
    n_t, n_x = 512, 161
    t_lag = 3.0  # ns, arbitrary record offset (plays the zero)
    sos = butter(4, [0.25 * f, 2.0 * f], btype="band", fs=1e9 / dt, output="sos")
    t = np.arange(n_t) * dt
    res = {}
    for depth in (0.6, 1.0, 1.5):
        v = 0.1
        ja = 80
        x = (np.arange(n_x) - ja) * dx
        tt = t_lag + 2.0 / v * np.sqrt(depth**2 + x**2)
        amp = depth / np.sqrt(depth**2 + x**2)
        clean = amp[None, :] * ricker((t[:, None] - tt[None, :]) * 1e-9, f)
        e_clean = np.abs(hilbert(clean, axis=0))
        a0 = float(e_clean[:, ja].max())
        rng0 = np.random.default_rng(SEED)
        pure = sosfiltfilt(sos, rng0.standard_normal((n_t, n_x)), axis=0)
        e_pure = np.abs(hilbert(pure, axis=0))
        sig_u = 1.4826 * float(np.median(np.abs(e_pure - np.median(e_pure))))
        for rho in (10, 20):
            sigma = a0 / (rho * sig_u)
            off_cur, off_vtx, n_none = [], [], 0
            for n in range(200):
                rng = np.random.default_rng(SEED + 1 + n + 1000 * rho + int(100 * depth))
                noise = sosfiltfilt(sos, rng.standard_normal((n_t, n_x)), axis=0) * sigma
                E = np.abs(hilbert(clean + noise, axis=0))
                j0 = ja + int(rng.integers(-10, 11))
                k_guess = int(round(tt[j0] / dt))
                lo, hi = k_guess - 10, k_guess + 10
                k0, _ = parabolic_peak(E[:, j0], lo + int(np.argmax(E[lo:hi, j0])))
                pts = crest_points(E, j0, k0, dt, dx)
                j_e, _ = min(pts, key=lambda p: p[1])
                off_cur.append(j_e - ja)
                jv = vertex_tau2(pts, j_e, dt, dx, t_lag)
                if jv is None:
                    n_none += 1
                    jv = j_e
                off_vtx.append(jv - ja)
            oc, ov = np.abs(off_cur), np.abs(off_vtx)
            res[f"d{depth}_rho{rho}"] = {
                "n": 200,
                "earliest_within3": round(float(np.mean(oc <= 3)), 3),
                "earliest_median_abs": float(np.median(oc)),
                "earliest_p90_abs": float(np.percentile(oc, 90)),
                "vertex_within3": round(float(np.mean(ov <= 3)), 3),
                "vertex_median_abs": float(np.median(ov)),
                "vertex_p90_abs": float(np.percentile(ov, 90)),
                "vertex_fallback": n_none,
                "dt_at_3_traces_ns": round(
                    float(2.0 / v * (math.sqrt(depth**2 + (3 * dx) ** 2) - depth)), 4
                ),
            }
    return res


def main():
    out = {
        "seed": SEED,
        "note": "already-seen data / development evidence; not a judgement input",
        "A_s1g": part_a(),
        "B_4tu_replay": part_b(),
        "C_crest_monte_carlo": part_c(),
        "D_variants_s1g": part_d_s1g(),
        "D_variants_4tu": part_d_4tu(),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "ok": True}, ensure_ascii=True))


if __name__ == "__main__":
    main()
