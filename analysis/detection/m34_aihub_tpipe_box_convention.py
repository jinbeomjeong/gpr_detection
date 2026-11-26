"""M3-4 box convention: measure where AIHub tpipe (pipe_hyperbola) label boxes sit relative to the hyperbola.

Purpose (plan docs/processing/m36_gprsd_plan.md section 5-3 item 2, order 1c): turn the AIHub label habit into
a written box convention for the GPR-SD labels. Exploration only, no verdict.

Data: train_pool only (subset == trainpool, folds 0-4); the test sessions are never read (asserted).
One image per crop unit (crop_key): gain 1.0 if present, else 0.5, else 0.25. Label files of the gain copies
of one crop are checked to be identical (boxes are the same across gains).

Per pipe_hyperbola box (class 0):
* geometry in px and in carrier periods P (P_px of the crop from the pixel band-pass estimator,
  prepare_fitting_branch, as in analysis/processing/aihub_wavelet_length.py);
* apex: section = band-passed crop minus its row median (horizontal background); envelope |hilbert| per column.
  Search rows [y0 - 2P, y1]. Reference level E = max envelope inside the box centre band (box centre +- 10 % width).
  For every column of the centre band the first local envelope maximum >= 0.5 E is the "first strong lobe" of
  that column; the apex is the shallowest of them (row, column). Envelope peak row refined by a parabola.
* main lobe top: first row above the apex peak where the apex-column envelope falls below 0.5 (and 0.1) of the
  apex peak ("lobe_top_05", "onset_01").
* margins (P): top_vs_peak = (apex_peak - y0)/P   ([DI] section 7 (a): >= 0.6 P required)
                top_vs_lobe05 = (lobe_top_05 - y0)/P, top_vs_onset01 = (onset_01 - y0)/P
* bottom: bottom_vs_peak = (y1 - apex_peak)/P; ring end below the apex where envelope < 0.3 / 0.1 of the apex
  peak (bottom_vs_ring03 = (y1 - ring03_end)/P); half cycles between lobe_top_05 and y1 = zero crossings of the
  apex-column section in that interval.
* wings: from the apex column outwards, the envelope ridge is tracked column by column (search window
  [prev - 0.5 P, prev + 1.5 P], ridge must stay >= 0.15 of the apex peak or tracking stops). Reported:
  descent at the box edge (row_at_edge - apex_peak)/P, whether tracking reached the edge, and the ridge
  amplitude at the edge relative to the apex peak; also the ridge continued 16 px beyond the box edge
  (if inside the crop) to see if the label cut a still-strong wing.
* truncation: box touching the crop border; overlap: pairs of tpipe boxes in one crop (IoU, intersection over
  the smaller box, apex row gap in P).

Output: analysis/detection/m34_aihub_tpipe_box_convention.json (aggregates, no file names),
per-box csv and a montage (AIHub pixels -> git-ignored data/analysis/m34_box_convention/).
stdout last line JSON.
Example: conda run -n gpr-dev python -I analysis/detection/m34_aihub_tpipe_box_convention.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[2]
SPLIT_CSV = ROOT / "data" / "splits" / "aihub_yolo_v1.csv"
DATASET = ROOT / "data" / "aihub_yolo_v1"
OUT_JSON = ROOT / "analysis" / "detection" / "m34_aihub_tpipe_box_convention.json"
OUT_DIR = ROOT / "data" / "analysis" / "m34_box_convention"
SEED = 0
GAIN_ORDER = ("1.0", "0.5", "0.25")


def envelope_cols(sec: np.ndarray) -> np.ndarray:
    n = sec.shape[0]
    pad = np.concatenate([sec, np.zeros_like(sec)], axis=0)
    return np.abs(hilbert(pad, axis=0))[:n]


def parse_boxes(label: Path, w: int, h: int):
    out = []
    for line in label.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if not p:
            continue
        cls = int(p[0])
        xc, yc, bw, bh = (float(v) for v in p[1:5])
        out.append((cls, (xc - bw / 2) * w, (xc + bw / 2) * w, (yc - bh / 2) * h, (yc + bh / 2) * h))
    return out


def parabolic(y: np.ndarray, i: int) -> float:
    if 0 < i < len(y) - 1:
        a, b, c = y[i - 1], y[i], y[i + 1]
        den = a - 2 * b + c
        if den < 0:
            return i + 0.5 * (a - c) / den
    return float(i)


def first_strong_peak(env: np.ndarray, lo: int, hi: int, thr: float):
    for r in range(max(lo, 1), min(hi, len(env) - 2) + 1):
        if env[r] >= thr and env[r] >= env[r - 1] and env[r] >= env[r + 1]:
            return r
    return None


def crossing_above(env: np.ndarray, peak: int, level: float):
    thr = level * env[peak]
    r = peak
    while r > 0 and env[r - 1] >= thr:
        r -= 1
    if r == 0:
        return None
    return r - (env[r] - thr) / max(env[r] - env[r - 1], 1e-12)


def crossing_below(env: np.ndarray, peak: int, level: float):
    """Last row of the ring below the apex: env stays >= level*peak from peak downward, allowing gaps < P/2."""
    thr = level * env[peak]
    r = peak
    n = len(env)
    while r < n - 1 and env[r + 1] >= thr:
        r += 1
    if r == n - 1:
        return None
    return r + (env[r] - thr) / max(env[r] - env[r + 1], 1e-12)


def track_ridge(env2d: np.ndarray, apex_row: int, apex_col: int, step: int, stop_col: int, P: float,
                peak_amp: float, extra: int):
    """Follow the envelope ridge from the apex column to stop_col (+extra beyond). Returns dict."""
    h, w = env2d.shape
    row = apex_row
    col = apex_col
    reached = None
    edge_row = None
    edge_amp = None
    beyond_amp = None
    beyond_row = None
    path = [(col, row)]
    target = stop_col + step * extra
    while True:
        nc = col + step
        if nc < 0 or nc >= w:
            break
        lo = max(0, int(math.floor(row - 0.5 * P)))
        hi = min(h - 1, int(math.ceil(row + 1.5 * P)))
        seg = env2d[lo:hi + 1, nc]
        r = lo + int(np.argmax(seg))
        amp = float(env2d[r, nc])
        if amp < 0.15 * peak_amp:
            break
        col, row = nc, r
        path.append((col, row))
        if col == stop_col:
            reached = True
            edge_row, edge_amp = row, amp
        if col == target:
            beyond_row, beyond_amp = row, amp
            break
    if reached is None:
        reached = False
    return {"reached": reached, "edge_row": edge_row, "edge_amp": edge_amp, "beyond_row": beyond_row,
            "beyond_amp": beyond_amp, "last_col": col, "last_row": row, "path": path}


def q(v, ps=(5, 10, 25, 50, 75, 90, 95)):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return None
    a = np.asarray(v, dtype=float)
    return {f"p{p}": round(float(np.percentile(a, p)), 3) for p in ps} | {"n": int(a.size),
                                                                          "mean": round(float(a.mean()), 3)}


def frac(v, cond):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return round(sum(1 for x in v if cond(x)) / len(v), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", type=int, default=24)
    ap.add_argument("--bg", choices=("row", "local"), default="local")
    ap.add_argument("--above", type=float, default=1.5, help="search starts this many P above the box top")
    ap.add_argument("--search-frac", type=float, default=0.6, help="first-lobe search ends at y0 + frac * h")
    ap.add_argument("--tag", default="")
    ap.add_argument("--apex", choices=("first", "arc"), default="arc")
    ap.add_argument("--arc-drop", type=float, default=0.75)
    args = ap.parse_args()
    global OUT_JSON, OUT_DIR
    if args.tag:
        OUT_JSON = OUT_JSON.with_name(f"{OUT_JSON.stem}_{args.tag}.json")
        OUT_DIR = OUT_DIR / args.tag

    import cv2
    import pandas as pd
    from gpr_detection.io._common import read_grayscale_file
    from gpr_detection.io.radargram import Radargram
    from gpr_detection.preprocess.filters import prepare_fitting_branch

    split = pd.read_csv(SPLIT_CSV, dtype=str, keep_default_na=False)
    pool = split[(split["subset"] == "trainpool") & split["fold"].isin({"0", "1", "2", "3", "4"})]
    assert (pool["subset"] != "test").all()
    test_sessions = set(split.loc[split["subset"] == "test", "session"])
    assert not (set(pool["session"]) & test_sessions), "test session in train_pool"
    has0 = pool["classes"].str.split(";").apply(lambda c: "0" in c)
    pool0 = pool[has0]

    # one image per crop unit, label identity across gains
    label_mismatch = 0
    chosen = []
    for key, grp in pool0.groupby("crop_key"):
        texts = {g: (DATASET / "labels" / f"{Path(f).stem}.txt").read_text(encoding="utf-8").strip()
                 for g, f in zip(grp["gain"], grp["file_name"])}
        if len(set(texts.values())) > 1:
            label_mismatch += 1
        for g in GAIN_ORDER:
            sel = grp[grp["gain"] == g]
            if len(sel):
                chosen.append((key, sel.iloc[0]["file_name"], g, sel.iloc[0]["session"], sel.iloc[0]["fold"]))
                break

    rows = []
    pairs = []
    sizes = {}
    montage_items = []
    for key, fname, gain, session, fold in chosen:
        img = read_grayscale_file(DATASET / "images" / fname)
        h, w = img.shape
        sizes[(w, h)] = sizes.get((w, h), 0) + 1
        boxes = [b for b in parse_boxes(DATASET / "labels" / f"{Path(fname).stem}.txt", w, h) if b[0] == 0]
        rg = Radargram(data=img.astype(np.float32), dt_ns=None, dx_m=None, air_gap="unknown",
                       amplitude_processed=False, dataset="aihub", line_id=Path(fname).stem)
        br = prepare_fitting_branch(rg)
        P = float(br.period_px) if br.period_px is not None else None
        P_used = P if P is not None else 4.06
        sec0 = br.filtered.astype(np.float64)
        if args.bg == "row":
            sec = sec0 - np.median(sec0, axis=1, keepdims=True)
            env2d = envelope_cols(sec)
        box_apex = []
        for bi, (_, x0f, x1f, y0f, y1f) in enumerate(boxes):
            if args.bg == "local":
                # horizontal moving median, window 2 * box width + 1 (flat layers removed, the apex lobe is
                # narrower than half the window so it survives)
                from scipy.ndimage import median_filter
                win = int(2 * (x1f - x0f)) | 1
                sec = sec0 - median_filter(sec0, size=(1, win), mode="reflect")
                env2d = envelope_cols(sec)
            rec = {"crop_key": key, "fold": fold, "gain": gain, "W": w, "H": h, "P_px": P,
                   "x0": x0f, "x1": x1f, "y0": y0f, "y1": y1f, "w_px": x1f - x0f, "h_px": y1f - y0f,
                   "n_hyp_in_crop": len(boxes),
                   "touch_left": x0f <= 0.5, "touch_right": x1f >= w - 0.5,
                   "touch_top": y0f <= 0.5, "touch_bottom": y1f >= h - 0.5}
            rec["w_P"] = rec["w_px"] / P_used
            rec["h_P"] = rec["h_px"] / P_used
            x0, x1 = max(0, int(math.floor(x0f))), min(w - 1, int(math.ceil(x1f)) - 1)
            y0, y1 = max(0, int(math.floor(y0f))), min(h - 1, int(math.ceil(y1f)) - 1)
            xc = 0.5 * (x0f + x1f)
            half = max(2, int(round(0.10 * (x1f - x0f))))
            cols = range(max(0, int(round(xc)) - half), min(w - 1, int(round(xc)) + half) + 1)
            lo = max(1, int(math.floor(y0f - args.above * P_used)))
            hi = min(h - 2, y1)
            E = float(env2d[lo:hi + 1, list(cols)].max())
            hi = min(hi, int(y0f + args.search_frac * (y1f - y0f)))
            best = None
            if args.apex == "first":
                for c in cols:
                    r = first_strong_peak(env2d[:, c], lo, hi, 0.5 * E)
                    if r is not None and (best is None or r < best[0]):
                        best = (r, c)
            else:
                # arc mode: candidates = local envelope maxima >= 0.3 E in the centre band, shallowest first;
                # accept the first whose ridge descends >= arc_drop P on BOTH sides at 0.4 box width
                # (per-column step limited to [-0.3 P, +0.6 P], ridge >= 0.15 of the candidate peak).
                cands = []
                for c in cols:
                    e = env2d[:, c]
                    for rr in range(max(lo, 1), min(hi, h - 2) + 1):
                        if e[rr] >= 0.3 * E and e[rr] >= e[rr - 1] and e[rr] >= e[rr + 1]:
                            cands.append((rr, c))
                cands.sort()
                reach = max(3, int(round(0.4 * (x1f - x0f))))
                for rr, c in cands:
                    amp0 = env2d[rr, c]
                    ok_sides = 0
                    for step in (-1, 1):
                        row, colx, good = rr, c, True
                        for _ in range(reach):
                            nc = colx + step
                            if nc < 0 or nc >= w:
                                good = False
                                break
                            a_ = max(0, int(math.floor(row - 0.3 * P_used)))
                            b_ = min(h - 1, int(math.ceil(row + 0.6 * P_used)))
                            seg = env2d[a_:b_ + 1, nc]
                            row = a_ + int(np.argmax(seg))
                            colx = nc
                            if env2d[row, nc] < 0.15 * amp0:
                                good = False
                                break
                        if good and (row - rr) >= args.arc_drop * P_used:
                            ok_sides += 1
                    if ok_sides == 2:
                        best = (rr, c)
                        break
            if best is None:
                rec["apex_ok"] = False
                rows.append(rec)
                box_apex.append(None)
                continue
            r, c = best
            env = env2d[:, c]
            peak = parabolic(env, r)
            rec.update({"apex_ok": True, "apex_row": peak, "apex_col": c,
                        "apex_col_off_w": (c - xc) / max(x1f - x0f, 1e-9),
                        "apex_in_box": (y0f <= peak <= y1f)})
            lt05 = crossing_above(env, r, 0.5)
            on01 = crossing_above(env, r, 0.1)
            rec["top_vs_peak_P"] = (peak - y0f) / P_used
            rec["top_vs_lobe05_P"] = None if lt05 is None else (lt05 - y0f) / P_used
            rec["top_vs_onset01_P"] = None if on01 is None else (on01 - y0f) / P_used
            rec["lobe05_half_above_P"] = None if lt05 is None else (peak - lt05) / P_used
            rec["bottom_vs_peak_P"] = (y1f - peak) / P_used
            r03 = crossing_below(env, r, 0.3)
            r01 = crossing_below(env, r, 0.1)
            rec["bottom_vs_ring03_P"] = None if r03 is None else (y1f - r03) / P_used
            rec["bottom_vs_ring01_P"] = None if r01 is None else (y1f - r01) / P_used
            if lt05 is not None:
                a, b = int(math.ceil(lt05)), int(min(h - 1, math.floor(y1f)))
                s = sec[a:b + 1, c]
                rec["half_cycles_lobe_to_bottom"] = int(np.sum(np.signbit(s[1:]) != np.signbit(s[:-1])))
            raw = img[max(0, r - int(P_used)):min(h, r + int(P_used) + 1), c]
            rec["apex_clipped"] = bool(np.any(raw <= 0) or np.any(raw >= 255))
            pk_amp = float(env[r])
            for side, step, stop in (("L", -1, x0), ("R", 1, x1)):
                t = track_ridge(env2d, r, c, step, stop, P_used, pk_amp, extra=16)
                rec[f"wing{side}_reached_edge"] = t["reached"]
                if t["reached"]:
                    rec[f"wing{side}_descent_P"] = (t["edge_row"] - peak) / P_used
                    rec[f"wing{side}_edge_amp_rel"] = t["edge_amp"] / pk_amp
                    rec[f"wing{side}_bottom_minus_edge_P"] = (y1f - t["edge_row"]) / P_used
                    rec[f"wing{side}_beyond_amp_rel"] = (None if t["beyond_amp"] is None
                                                         else t["beyond_amp"] / pk_amp)
                    rec[f"wing{side}_beyond_descent_P"] = (None if t["beyond_row"] is None
                                                           else (t["beyond_row"] - peak) / P_used)
                else:
                    rec[f"wing{side}_stop_frac"] = abs(t["last_col"] - c) / max(abs(stop - c), 1)
                    rec[f"wing{side}_stop_descent_P"] = (t["last_row"] - peak) / P_used
                rec[f"_path{side}"] = t["path"]
            rows.append(rec)
            box_apex.append(peak)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ix = max(0.0, min(a[2], b[2]) - max(a[1], b[1]))
                iy = max(0.0, min(a[4], b[4]) - max(a[3], b[3]))
                inter = ix * iy
                aa = (a[2] - a[1]) * (a[4] - a[3])
                bb = (b[2] - b[1]) * (b[4] - b[3])
                gap = (None if box_apex[i] is None or box_apex[j] is None
                       else abs(box_apex[i] - box_apex[j]) / P_used)
                xgap = abs(0.5 * (a[1] + a[2]) - 0.5 * (b[1] + b[2])) / (0.5 * ((a[2] - a[1]) + (b[2] - b[1])))
                pairs.append({"iou": inter / (aa + bb - inter), "ios": inter / min(aa, bb),
                              "apex_gap_P": gap, "center_dx_over_mean_w": xgap})
        montage_items.append((fname, img, [r_ for r_ in rows if r_["crop_key"] == key]))

    ok = [r for r in rows if r.get("apex_ok")]
    okc = [r for r in ok if not r["apex_clipped"]]

    def col(rs, k):
        return [r.get(k) for r in rs]

    def wing_stats(rs):
        d = {}
        for s in ("L", "R"):
            d[s] = {"reached_edge_frac": frac(col(rs, f"wing{s}_reached_edge"), bool),
                    "descent_P": q(col(rs, f"wing{s}_descent_P")),
                    "edge_amp_rel": q(col(rs, f"wing{s}_edge_amp_rel")),
                    "bottom_minus_edge_P": q(col(rs, f"wing{s}_bottom_minus_edge_P")),
                    "beyond16_amp_rel": q(col(rs, f"wing{s}_beyond_amp_rel")),
                    "beyond16_descent_P": q(col(rs, f"wing{s}_beyond_descent_P")),
                    "stop_frac_if_not_reached": q(col(rs, f"wing{s}_stop_frac")),
                    "stop_descent_P_if_not_reached": q(col(rs, f"wing{s}_stop_descent_P"))}
        both = [min(a, b) for a, b in zip(col(rs, "wingL_descent_P"), col(rs, "wingR_descent_P"))
                if a is not None and b is not None]
        d["min_descent_both_reached_P"] = q(both)
        return d

    out = {
        "seed": SEED,
        "config": {"bg": args.bg, "above_P": args.above, "search_frac": args.search_frac,
                   "apex": args.apex, "arc_drop_P": args.arc_drop},
        "source": {"split_csv": "data/splits/aihub_yolo_v1.csv", "subset": "trainpool folds 0-4",
                   "test_read": False, "crop_units_with_tpipe": len(chosen),
                   "gain_used": {g: sum(1 for c in chosen if c[2] == g) for g in GAIN_ORDER},
                   "label_mismatch_across_gains": label_mismatch,
                   "sessions": len({c[3] for c in chosen}),
                   "image_sizes": {f"{k[0]}x{k[1]}": v for k, v in sizes.items()}},
        "boxes": {"n": len(rows), "per_crop": q(col(rows, "n_hyp_in_crop")),
                  "w_px": q(col(rows, "w_px")), "h_px": q(col(rows, "h_px")),
                  "w_P": q(col(rows, "w_P")), "h_P": q(col(rows, "h_P")),
                  "w_over_h": q([r["w_px"] / r["h_px"] for r in rows]),
                  "P_px": q(col(rows, "P_px")), "P_missing": sum(1 for r in rows if r["P_px"] is None),
                  "touch": {k: sum(1 for r in rows if r[k]) for k in
                            ("touch_left", "touch_right", "touch_top", "touch_bottom")},
                  "y0_px": q(col(rows, "y0")), "y1_px": q(col(rows, "y1"))},
        "apex": {"found": len(ok), "not_found": len(rows) - len(ok),
                 "clipped": sum(1 for r in ok if r["apex_clipped"]),
                 "in_box_frac": frac(col(ok, "apex_in_box"), bool),
                 "col_offset_over_w": q(col(ok, "apex_col_off_w")),
                 "lobe05_half_above_P": q(col(okc, "lobe05_half_above_P"))},
        "top_margin": {
            "all": {"top_vs_peak_P": q(col(ok, "top_vs_peak_P")),
                    "frac_peak_ge_0.6P": frac(col(ok, "top_vs_peak_P"), lambda x: x >= 0.6),
                    "frac_peak_ge_1.2P": frac(col(ok, "top_vs_peak_P"), lambda x: x >= 1.2),
                    "frac_peak_lt_0": frac(col(ok, "top_vs_peak_P"), lambda x: x < 0),
                    "top_vs_lobe05_P": q(col(ok, "top_vs_lobe05_P")),
                    "frac_lobe05_ge_0": frac(col(ok, "top_vs_lobe05_P"), lambda x: x >= 0),
                    "frac_lobe05_ge_0.6P": frac(col(ok, "top_vs_lobe05_P"), lambda x: x >= 0.6),
                    "top_vs_onset01_P": q(col(ok, "top_vs_onset01_P")),
                    "frac_onset01_ge_0": frac(col(ok, "top_vs_onset01_P"), lambda x: x >= 0)},
            "unclipped": {"top_vs_peak_P": q(col(okc, "top_vs_peak_P")),
                          "frac_peak_ge_0.6P": frac(col(okc, "top_vs_peak_P"), lambda x: x >= 0.6),
                          "top_vs_lobe05_P": q(col(okc, "top_vs_lobe05_P")),
                          "top_vs_onset01_P": q(col(okc, "top_vs_onset01_P"))}},
        "bottom": {"bottom_vs_peak_P": q(col(ok, "bottom_vs_peak_P")),
                   "bottom_vs_ring03_P": q(col(okc, "bottom_vs_ring03_P")),
                   "bottom_vs_ring01_P": q(col(okc, "bottom_vs_ring01_P")),
                   "frac_bottom_below_ring03": frac(col(okc, "bottom_vs_ring03_P"), lambda x: x >= 0),
                   "half_cycles_lobe_to_bottom": q(col(ok, "half_cycles_lobe_to_bottom"))},
        "wings": wing_stats(ok),
        "pairs": {"n": len(pairs), "iou": q([p["iou"] for p in pairs]), "ios": q([p["ios"] for p in pairs]),
                  "frac_overlap": frac([p["iou"] for p in pairs], lambda x: x > 0),
                  "frac_ios_ge_0.5": frac([p["ios"] for p in pairs], lambda x: x >= 0.5),
                  "apex_gap_P": q([p["apex_gap_P"] for p in pairs]),
                  "center_dx_over_mean_w": q([p["center_dx_over_mean_w"] for p in pairs])},
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = [k for k in rows[0].keys() if not k.startswith("_")]
    allk = sorted({k for r in rows for k in r if not k.startswith("_")}, key=lambda k: (k not in keys, k))
    pd.DataFrame([{k: r.get(k) for k in allk} for r in rows]).to_csv(OUT_DIR / "per_box.csv", index=False)

    # montage: random crops, box (green), apex (red), lobe05 top (yellow line), ridge (cyan)
    rng = np.random.default_rng(SEED)
    idx = sorted(rng.choice(len(montage_items), size=min(args.montage, len(montage_items)), replace=False))
    tiles = []
    for i in idx:
        _, img, recs = montage_items[i]
        z = 2
        im = cv2.cvtColor(cv2.resize(img, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        for r_ in recs:
            cv2.rectangle(im, (int(r_["x0"] * z), int(r_["y0"] * z)), (int(r_["x1"] * z), int(r_["y1"] * z)),
                          (0, 255, 0), 1)
            if r_.get("apex_ok"):
                for s in ("L", "R"):
                    for (cc, rr) in r_[f"_path{s}"]:
                        cv2.circle(im, (cc * z, rr * z), 1, (255, 255, 0), -1)
                cv2.circle(im, (r_["apex_col"] * z, int(r_["apex_row"] * z)), 3, (0, 0, 255), -1)
                y = r_["y0"] + r_["top_vs_peak_P"] * 0  # box top
                yp = r_["apex_row"] - 0.6 * (r_["P_px"] or 4.06)
                cv2.line(im, (int(r_["x0"] * z), int(yp * z)), (int(r_["x1"] * z), int(yp * z)), (0, 255, 255), 1)
        tiles.append(im)
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0] + 4, 0, tw - t.shape[1] + 4, cv2.BORDER_CONSTANT,
                                value=(40, 40, 40)) for t in tiles]
    ncol = 6
    grid_rows = [np.hstack(tiles[k:k + ncol] + [np.zeros_like(tiles[0])] * (ncol - len(tiles[k:k + ncol])))
                 for k in range(0, len(tiles), ncol)]
    cv2.imwrite(str(OUT_DIR / "montage.png"), np.vstack(grid_rows))

    print(json.dumps({"status": "ok", "boxes": len(rows), "apex_found": len(ok), "pairs": len(pairs),
                      "out": str(OUT_JSON)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
