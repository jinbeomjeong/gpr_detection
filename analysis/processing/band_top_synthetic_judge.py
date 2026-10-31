"""Quantitative Q1-Q3 judgement of the band-top rule on the gprMax band-top set (band_box_convention_depth.md 7).

Input: data/synthetic_gprmax_band_top/export/<id>.json / .npy (apex trace) / _bg.npy (layer, no pipe),
homogeneous backgrounds h5/K0f.h5 (Ricker, 5089 samples) and h5/K0r.h5 (ring, 6785 samples).
Sets: v1 grid L_* (60), wet bed W_* (20), ring RL_* (28) + RK_* (2), steel plate PL_* (15, bias only:
their t_top_true_ns is a plate-edge diffraction, not the pipe top).

Pseudo band ([B] 9-2): 200 traces, dx 0.02 m, band = apex trace on traces 40-160, noise
sigma = 0.05 * max|pipe-only| (PL: max|total|), seed 0 per model. Every trace is zero-padded to twice
its length so the project's Hilbert envelope (pick_signal) is a zero-padded one (circular leakage of
ring tails, band_top_synthetic_v1.md 9-1).
Background paths: masked (outside = layer-only traces -> masked median removes the layer),
direct (outside = homogeneous -> only the direct wave is removed, layer local to the band),
none (no background removal).
Boxes: [t_top - 0.4P + delta, t_top + 0.8P]. Q1/Q2: one box (40-160), delta in {-1.5..+1}P.
Q3: fragments 40-96 (delta 0) and 103-160 (delta2), compared as
  (B) union window (current merged group) vs (C) per-fragment windows (S3) on member traces,
plus the real v1.4 merge decision (group_band_boxes with fragment_picks).
Correct pick: |t_pick - t_top_true| <= 0.25 P. Layer pick: within 0.25 P of a layer_times_ns entry.
Representative: fit_band_profile tau_ref + t_zero (direct-wave peak), error in ns and in cm via v_eff.

Depth code is called read-only (public band.py functions). Output: analysis/processing/
band_top_synthetic_judge.json (aggregates) and data/audit/band_top_judge_rows.csv (cells).
Example: conda run -n gpr-dev python -I analysis/processing/band_top_synthetic_judge.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "synthetic_gprmax_band_top"
OUT_JSON = ROOT / "analysis" / "processing" / "band_top_synthetic_judge.json"
OUT_ROWS = ROOT / "data" / "audit" / "band_top_judge_rows.csv"
N_TRACES, BAND = 200, (40, 160)
FRAGS = ((40, 96), (103, 160))
DX = 0.02
DELTAS = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0)
DELTA2 = (-1.5, -1.0, -0.5, 0.5, 1.0)
TOL_P = 0.25
DECIMATE = 6  # dt 0.0354 ns, 57 samples per 2 ns period (overridden by --decimate)
PIPE_DELAY_NS = {("ricker", "pec"): 2.87, ("ricker", "pvc"): 3.06, ("ring", "pec"): 3.31, ("ring", "pvc"): 3.60}


def homogeneous(name: str) -> np.ndarray:
    with h5py.File(SRC / "h5" / f"{name}.h5", "r") as handle:
        return np.asarray(handle["rxs/rx1/Ez"][()], dtype=np.float64)


def ids() -> dict[str, list[str]]:
    names = sorted(p.stem for p in (SRC / "export").glob("*.json") if not p.stem.startswith(("summary", "analysis")))
    return {
        "v1_grid": [n for n in names if n.startswith("L_")],
        "wet_bed": [n for n in names if n.startswith("W_")],
        "ring": [n for n in names if n.startswith(("RL_", "RK_"))],
        "plate": [n for n in names if n.startswith("PL_")],
    }


def section(band: np.ndarray, outside: np.ndarray, sigma: float, rng) -> np.ndarray:
    n = len(band)
    sec = np.repeat(outside[:n, None], N_TRACES, axis=1)
    sec[:, BAND[0]: BAND[1] + 1] = band[:, None]
    sec = sec + rng.normal(0.0, sigma, sec.shape)
    return np.vstack([sec, np.zeros_like(sec)])  # zero padding -> zero-padded Hilbert downstream


def classify(t: float | None, t_true: float, layers: list[float], period: float) -> str:
    if t is None or not np.isfinite(t):
        return "none"
    if abs(t - t_true) <= TOL_P * period:
        return "top"
    if any(abs(t - lt) <= TOL_P * period for lt in layers):
        return "layer"
    return "other"


def per_trace(picks, dt, traces, t_true, layers, period):
    lookup = {int(i): float(k) * dt for i, k in zip(picks.trace_idx, picks.k_pick)}
    return {t: classify(lookup.get(t), t_true, layers, period) for t in traces}, lookup


def representative(picks, dt, t_zero, cfg):
    from gpr_detection.depth.band import fit_band_profile
    if picks.n_picks < 5:
        return None, None
    fit = fit_band_profile(picks, DX, None, 0.0, cfg, dt_ns=dt, f_c_hz=5e8, k_zero=t_zero / dt)
    return fit.tau_ref + t_zero, fit.model


def concat_picks(base, parts):
    return replace(base,
                   trace_idx=np.concatenate([p.trace_idx for p in parts]),
                   k_pick=np.concatenate([p.k_pick for p in parts]),
                   t_pick_ns=None if base.t_pick_ns is None else np.concatenate([p.t_pick_ns for p in parts]),
                   amp=np.concatenate([p.amp for p in parts]),
                   n_member_picks=sum(p.n_member_picks for p in parts))


def judge_model(model_id: str, set_name: str, backgrounds: dict[str, np.ndarray]):
    from gpr_detection.depth.band import (extract_band_picks, group_band_boxes, masked_background)
    from gpr_detection.depth.config import BandConfig
    from gpr_detection.io.radargram import GridBox, Radargram
    from gpr_detection.preprocess.filters import prepare_fitting_branch

    meta = json.loads((SRC / "export" / f"{model_id}.json").read_text(encoding="utf-8"))
    dt, period = float(meta["dt_ns"]), float(meta["period_ns"])
    total = np.load(SRC / "export" / f"{model_id}.npy")[:, 0].astype(np.float64)
    layer_bg = np.load(SRC / "export" / f"{model_id}_bg.npy")[:, 0].astype(np.float64)
    pipe_only = np.load(SRC / "export" / f"{model_id}_pipeonly.npy")[:, 0].astype(np.float64)
    wavelet = meta.get("wavelet", "ricker")
    homog = backgrounds["ring" if wavelet == "ring" else "ricker"][: len(total)]
    # gprMax dt (~0.0059 ns) is ~340 samples per period; keep every DECIMATE-th sample (content < 2 GHz,
    # Nyquist after decimation > 10 GHz, so no anti-alias filter is needed)
    total, layer_bg, pipe_only, homog = (x[::DECIMATE] for x in (total, layer_bg, pipe_only, homog))
    dt *= DECIMATE
    plate = (meta.get("design") or {}).get("strong_kind") == "steel_plate"
    t_zero = float(meta["time_zero_candidates_ns"]["direct_wave_peak"])
    layers = [float(x) for x in meta.get("layer_times_ns") or []]
    if plate:
        t_true = float(meta["tau_top_vertical_ns"]) + PIPE_DELAY_NS[(wavelet, meta["pipe_material"])]
        box_ref = layers[0] if layers else t_true
    else:
        t_true = float(meta["t_top_true_ns"])
        box_ref = t_true
    sigma = 0.05 * float(np.max(np.abs(total if plate else pipe_only)))
    cfg = BandConfig()
    out = []
    sections = {"masked": section(total, layer_bg, sigma, np.random.default_rng(0)),
                "direct": section(total, homog, sigma, np.random.default_rng(0))}
    sections["none"] = sections["direct"]
    filtered = {}
    for key in ("masked", "direct"):
        rg = Radargram(data=sections[key].astype(np.float32), dt_ns=dt, dx_m=DX, air_gap="ground", f_c_hz=5e8,
                       amplitude_processed=False, dataset="synthetic", line_id=model_id)
        filtered[key] = prepare_fitting_branch(rg).filtered
    filtered["none"] = filtered["direct"]
    k_mute = int(math.ceil((t_zero + period) / dt))
    v_eff = float(meta.get("v_eff_to_top_m_per_ns") or 0.0999)
    dl = [abs(x) for x in (meta.get("layer_dt_measured_P") or [])]
    common = {"set": set_name, "model": model_id, "wavelet": wavelet, "pipe": meta["pipe_material"],
              "side": (meta.get("design") or {}).get("layer_side"), "ratio": meta.get("ratio_layer_to_pipe_measured"),
              "dl_min_P": min(dl) if dl else None, "plate": plate,
              "plate_top_m": (meta.get("design") or {}).get("plate_top_depth_m")}

    def box(i0, i1, delta):
        k0 = int(math.floor((box_ref - 0.4 * period + delta * period) / dt))
        k1 = int(math.ceil((box_ref + 0.8 * period) / dt))
        return GridBox(i0=i0, i1=i1, k0=k0, k1=k1)

    def a_bg(path, boxes):
        a = filtered[path]
        if path == "none":
            return a, "none"
        bg, mode = masked_background(a, boxes, (BAND[0] + BAND[1]) // 2, DX, cfg)
        return (a, mode) if bg is None else (a - bg[:, None], mode)

    def picks_of(a, mode, bx, members, idx):
        traces = np.array(sorted({t for b in members for t in range(b.i0, b.i1 + 1)}))
        return extract_band_picks(a, bx, k_mute, dt, 5e8, cfg, k_zero=t_zero / dt, member_traces=traces,
                                  member_indices=idx, bg_mode=mode)

    for path in ("masked", "direct", "none"):
        # Q1/Q2: one box
        for delta in DELTAS:
            bx = box(*BAND, delta)
            a, mode = a_bg(path, [bx])
            p = picks_of(a, mode, bx, [bx], (0,))
            cls, _ = per_trace(p, dt, range(BAND[0], BAND[1] + 1), t_true, layers, period)
            t_ref, model = representative(p, dt, t_zero, cfg)
            counts = {c: sum(v == c for v in cls.values()) for c in ("top", "layer", "other", "none")}
            out.append(common | {"q": "Q2", "path": path, "bg_mode": mode, "delta": delta, "delta2": None,
                                 "n_traces": len(cls), **{f"n_{c}": counts[c] for c in counts},
                                 "t_ref_err_P": None if t_ref is None else (t_ref - t_true) / period,
                                 "d_ref_err_cm": None if t_ref is None else 100 * v_eff * (t_ref - t_true) / 2,
                                 "ref_class": classify(t_ref, t_true, layers, period), "fit_model": model})
        if plate:
            continue
        # Q3: two fragments
        for delta2 in DELTA2:
            f1, f2 = box(*FRAGS[0], 0.0), box(*FRAGS[1], delta2)
            a, mode = a_bg(path, [f1, f2])
            p1 = picks_of(a, mode, f1, [f1], (0,))
            p2 = picks_of(a, mode, f2, [f2], (1,))
            union = GridBox(i0=f1.i0, i1=f2.i1, k0=min(f1.k0, f2.k0), k1=max(f1.k1, f2.k1))
            pu = picks_of(a, mode, union, [f1, f2], (0, 1))
            member = [t for f in FRAGS for t in range(f[0], f[1] + 1)]
            cls_b, _ = per_trace(pu, dt, member, t_true, layers, period)
            pc = concat_picks(p1, [p1, p2])
            cls_c, _ = per_trace(pc, dt, member, t_true, layers, period)
            tol = max(0.5 * period / dt, 3.0)
            groups = group_band_boxes([f1, f2], DX, cfg, fragment_picks=[p1, p2], tolerance_samples=tol)
            merged = len(groups) == 1
            t_b, _ = representative(pu, dt, t_zero, cfg)
            t_c, _ = representative(pc, dt, t_zero, cfg)
            t_a = [representative(p, dt, t_zero, cfg)[0] for p in (p1, p2)]
            # moved traces: those whose window differs between (B) and (C)
            moved = list(range(FRAGS[1][0], FRAGS[1][1] + 1)) if delta2 > 0 else list(range(FRAGS[0][0], FRAGS[0][1] + 1))
            out.append(common | {
                "q": "Q3", "path": path, "bg_mode": mode, "delta": 0.0, "delta2": delta2, "n_traces": len(member),
                "top_B": sum(v == "top" for v in cls_b.values()), "top_C": sum(v == "top" for v in cls_c.values()),
                "layer_B": sum(v == "layer" for v in cls_b.values()), "layer_C": sum(v == "layer" for v in cls_c.values()),
                "moved_top_B": sum(cls_b[t] == "top" for t in moved), "moved_top_C": sum(cls_c[t] == "top" for t in moved),
                "n_moved": len(moved), "v14_merged": merged,
                "B_ref_err_P": None if t_b is None else (t_b - t_true) / period,
                "C_ref_err_P": None if t_c is None else (t_c - t_true) / period,
                "A_ref_err_P": json.dumps([None if t is None else round((t - t_true) / period, 3) for t in t_a]),
                "B_d_err_cm": None if t_b is None else 100 * v_eff * (t_b - t_true) / 2,
                "C_d_err_cm": None if t_c is None else 100 * v_eff * (t_c - t_true) / 2})
    return out


def ratio_bin(r):
    if r is None:
        return "none"
    return "<0.5" if r < 0.5 else ("0.5-1.5" if r < 1.5 else ("1.5-3" if r < 3 else ">=3"))


def dl_bin(d):
    if d is None:
        return "no_layer"
    return "<=1P" if d <= 1.0 else ("1-2P" if d <= 2.0 else ">2P")


def aggregate(rows):
    agg: dict = {}
    q2 = [r for r in rows if r["q"] == "Q2" and not r["plate"]]
    q3 = [r for r in rows if r["q"] == "Q3"]
    # Q1: delta 0, layers > 1P away (or no layer)
    q1 = {}
    for s in sorted({r["set"] for r in q2}):
        for path in ("masked", "direct", "none"):
            sel = [r for r in q2 if r["set"] == s and r["path"] == path and r["delta"] == 0.0
                   and (r["dl_min_P"] is None or r["dl_min_P"] > 1.0)]
            if not sel:
                continue
            n = sum(r["n_traces"] for r in sel)
            d = [abs(r["d_ref_err_cm"]) for r in sel if r["d_ref_err_cm"] is not None]
            q1[f"{s}|{path}"] = {"cells": len(sel), "trace_top_rate": round(sum(r["n_top"] for r in sel) / n, 4),
                                 "ref_abs_d_cm_median": round(float(np.median(d)), 2) if d else None,
                                 "ref_abs_d_cm_max": round(float(np.max(d)), 2) if d else None,
                                 "ref_le_2cm": round(float(np.mean(np.array(d) <= 2.0)), 4) if d else None,
                                 "pass": (sum(r["n_top"] for r in sel) / n >= 0.95) and bool(d) and max(d) <= 2.0}
    agg["Q1"] = q1
    # Q2: correct rate vs delta by set/path/side/ratio/dl
    q2agg = defaultdict(dict)
    for r in q2:
        key = f"{r['set']}|{r['path']}|{r['side']}|{ratio_bin(r['ratio'])}|{dl_bin(r['dl_min_P'])}"
        cell = q2agg[key].setdefault(str(r["delta"]), [0, 0, 0])
        cell[0] += r["n_top"]
        cell[1] += r["n_traces"]
        cell[2] += 1
    agg["Q2"] = {k: {d: {"top_rate": round(v[0] / v[1], 3), "cells": v[2]} for d, v in sorted(dd.items(), key=lambda x: float(x[0]))}
                 for k, dd in sorted(q2agg.items())}
    # Q2 summary per set/path: delta range keeping >= 90 %
    q2s = {}
    for s in sorted({r["set"] for r in q2}):
        for path in ("masked", "direct", "none"):
            per = {}
            for d in DELTAS:
                sel = [r for r in q2 if r["set"] == s and r["path"] == path and r["delta"] == d]
                per[str(d)] = round(sum(r["n_top"] for r in sel) / max(1, sum(r["n_traces"] for r in sel)), 3)
            q2s[f"{s}|{path}"] = per
    agg["Q2_summary"] = q2s
    # Q3
    q3s = {}
    for s in sorted({r["set"] for r in q3}):
        for path in ("masked", "direct", "none"):
            sel = [r for r in q3 if r["set"] == s and r["path"] == path]
            if not sel:
                continue
            diffs = [(r["top_C"] - r["top_B"]) / r["n_traces"] for r in sel]
            moved = [(r["moved_top_C"] - r["moved_top_B"]) / r["n_moved"] for r in sel]
            q3s[f"{s}|{path}"] = {
                "cells": len(sel),
                "B_top_rate": round(sum(r["top_B"] for r in sel) / sum(r["n_traces"] for r in sel), 4),
                "C_top_rate": round(sum(r["top_C"] for r in sel) / sum(r["n_traces"] for r in sel), 4),
                "C_minus_B_mean_pp": round(100 * float(np.mean(diffs)), 2),
                "C_minus_B_min_pp": round(100 * float(np.min(diffs)), 2),
                "C_minus_B_max_pp": round(100 * float(np.max(diffs)), 2),
                "cells_C_worse_5pp": int(sum(d < -0.05 for d in diffs)),
                "cells_C_better_10pp": int(sum(d > 0.10 for d in diffs)),
                "moved_C_minus_B_mean_pp": round(100 * float(np.mean(moved)), 2),
                "by_delta2_mean_pp": {str(d2): round(100 * float(np.mean([(r["top_C"] - r["top_B"]) / r["n_traces"]
                                                                          for r in sel if r["delta2"] == d2])), 2)
                                      for d2 in DELTA2},
                "v14_merge_rate": round(float(np.mean([r["v14_merged"] for r in sel])), 3),
                "B_ref_abs_P_median": round(float(np.median([abs(r["B_ref_err_P"]) for r in sel if r["B_ref_err_P"] is not None])), 3),
                "C_ref_abs_P_median": round(float(np.median([abs(r["C_ref_err_P"]) for r in sel if r["C_ref_err_P"] is not None])), 3),
                "B_ref_le_0.25P": round(float(np.mean([abs(r["B_ref_err_P"]) <= 0.25 for r in sel if r["B_ref_err_P"] is not None])), 3),
                "C_ref_le_0.25P": round(float(np.mean([abs(r["C_ref_err_P"]) <= 0.25 for r in sel if r["C_ref_err_P"] is not None])), 3),
                "pass_S3": float(np.mean(diffs)) >= 0.10 and float(np.min(diffs)) >= -0.05,
            }
    agg["Q3"] = q3s
    # Q4-like bias for layer within 1P (delta 0, masked excluded)
    q4 = {}
    for s in sorted({r["set"] for r in q2}):
        for path in ("direct", "none"):
            sel = [r for r in q2 if r["set"] == s and r["path"] == path and r["delta"] == 0.0
                   and r["dl_min_P"] is not None and r["dl_min_P"] <= 1.0 and r["t_ref_err_P"] is not None]
            for side in ("above", "below"):
                ss = [r["t_ref_err_P"] for r in sel if r["side"] == side]
                if ss:
                    q4[f"{s}|{path}|{side}"] = {"n": len(ss), "min_P": round(min(ss), 3), "max_P": round(max(ss), 3)}
    agg["Q4_bias_delta0"] = q4
    # plate
    pl = [r for r in rows if r["plate"] and r["q"] == "Q2"]
    agg["plate"] = {}
    for path in ("masked", "direct", "none"):
        sel = [r for r in pl if r["path"] == path and r["delta"] == 0.0]
        agg["plate"][path] = {
            "cells": len(sel), "ref_class": dict(zip(*np.unique([r["ref_class"] for r in sel], return_counts=True))) if sel else {},
            "d_err_vs_pipe_cm": [round(r["d_ref_err_cm"], 1) if r["d_ref_err_cm"] is not None else None for r in sel],
            "plate_above_pipe_m": [round(0.6 - r["plate_top_m"], 3) if r["plate_top_m"] else None for r in sel],
            "models": [r["model"] for r in sel]}
        agg["plate"][path]["ref_class"] = {str(k): int(v) for k, v in agg["plate"][path]["ref_class"].items()}
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--decimate", type=int, default=6)
    ap.add_argument("--only", default="", help="comma list of set names")
    a = ap.parse_args()
    global DECIMATE
    DECIMATE = a.decimate
    backgrounds = {"ricker": homogeneous("K0f"), "ring": homogeneous("K0r")}
    start = time.perf_counter()
    rows = []
    only = {s for s in a.only.split(",") if s}
    for set_name, names in ids().items():
        if only and set_name not in only:
            continue
        for name in names[: a.limit or None]:
            rows.extend(judge_model(name, set_name, backgrounds))
    agg = aggregate(rows)
    agg["meta"] = {"seconds": round(time.perf_counter() - start, 1), "seed": 0, "tol_P": TOL_P,
                   "noise": "0.05*max|pipe-only| (plate: |total|), per decimated sample",
                   "hilbert": "zero-padded (trace doubled)", "decimate": DECIMATE,
                   "n_rows": len(rows)}
    OUT_JSON.write_text(json.dumps(agg, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    OUT_ROWS.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with OUT_ROWS.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"status": "ok", "rows": len(rows), "seconds": agg["meta"]["seconds"], "json": str(OUT_JSON)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
