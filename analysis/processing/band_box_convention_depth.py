"""pipe_band box convention (v1 split boxes vs v2 merged boxes) judged by the band depth path.

Question (docs/processing/band_box_convention_depth.md): does the band depth rule
(src/gpr_detection/depth/band.py, spec_band_depth_v1) give the same band-top row when the band
is given as split boxes (v1 AIHub labels, v1-trained detector) or as merged boxes (v2 auto-32
labels, v2-trained detector)?

Data: the fold-0 crop list of docs/experiments/band_merge_v2_compare.md (1,298 crops, one gain copy
per crop), restricted to crops with a v1 pipe_band GT box. AIHub has no dt/dx, so everything is
in px (rows = samples, columns = traces) and every result is ``no_scale`` (QA-4).

Sources (box sets fed to the same depth path, per crop):
  gt_v1, gt_v2                     ground-truth boxes, merge_gap_px 8 set explicitly
  gt_v1_G32                        v1 GT with merge_gap_px 32 (depth-stage grouping, edge check kept)
  pred_<v1|v2>_s<0|1>[_G32]        detector boxes from band_convention_predict.py (conf 0.25)
The fragment check is whatever band.py implements: v1.3 facing ends (first run,
band_box_convention_depth.json) or v1.4 S1 overlap rule (``--suffix _v14``). Since spec v1.4 the
BandConfig default is merge_gap_px 32, so the *_G32 sources are the v1.4 defaults.

The band objects of reference are the v2 GT boxes (each one is the union of one or more v1 boxes).
The depth path is called exactly as the pipeline would (band_exclude_traces -> find_time_zero ->
estimate_band_depths). The private helpers ``_context``/``_merged_groups``/``_evaluate_group``/
``_picks_for``/``_edge_picks`` of band.py are called read-only to get the per-trace picks and fits
behind each result (same code path, nothing re-implemented).

Outputs:
  analysis/processing/band_box_convention_depth.json   aggregates only (public)
  docs/processing/band_box_convention_depth.png        distributions, no AIHub imagery (public)
  data/audit/band_box_convention_rows.csv              per band x source rows (local: file names)
  data/audit/band_convention_figures/*.png             AIHub crop examples (local only)
stdout last line: JSON summary. No random numbers except the example pick (seed 0).

Example:
    conda run -n gpr-dev python -I analysis/processing/band_box_convention_depth.py --workers 6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V1_DIR = ROOT / "data" / "aihub_yolo_v1"
V2_DIR = ROOT / "data" / "aihub_yolo_v2_auto32"
PRED_CSV = ROOT / "data" / "audit" / "band_convention_preds.csv"
LIST_TXT = (ROOT / "runs" / "detect" / "band_v2_eval" / "yolo11m_bandv2_aug0_20261031_fold0_s0__gt_v2"
            / "eval" / "last_crop" / "eval_images.txt")
OUT_JSON = ROOT / "analysis" / "processing" / "band_box_convention_depth.json"
OUT_PNG = ROOT / "docs" / "processing" / "band_box_convention_depth.png"
OUT_ROWS = ROOT / "data" / "audit" / "band_box_convention_rows.csv"
OUT_FIG_LOCAL = ROOT / "data" / "audit" / "band_convention_figures"
PIPE_BAND = 1
PRED_RUNS = ("v1_s0", "v1_s1", "v2_s0", "v2_s1")
EDGE_TRACES = 5


# ----------------------------------------------------------------- boxes

def yolo_band_boxes(label_path: Path, width: int, height: int):
    from gpr_detection.io.radargram import GridBox
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or int(parts[0]) != PIPE_BAND:
            continue
        xc, yc, w, h = (float(v) for v in parts[1:5])
        boxes.append(xyxy_box((xc - w / 2) * width, (yc - h / 2) * height,
                              (xc + w / 2) * width, (yc + h / 2) * height, width, height))
    return [b for b in boxes if b is not None]


def xyxy_box(x1, y1, x2, y2, width, height):
    """Pixel j covers [j, j+1) -> closed L0 index box (same rule as scripts/check_band_picks_aihub.py)."""
    from gpr_detection.io.radargram import GridBox
    i0 = max(0, math.floor(x1))
    i1 = min(width - 1, math.ceil(x2) - 1)
    k0 = max(0, math.floor(y1))
    k1 = min(height - 1, math.ceil(y2) - 1)
    if i1 < i0 or k1 < k0:
        return None
    return GridBox(i0=i0, i1=i1, k0=k0, k1=k1)


def y_overlap_ok(a, b, frac=0.5):
    overlap = min(a.k1, b.k1) - max(a.k0, b.k0) + 1
    return overlap >= frac * min(a.k1 - a.k0 + 1, b.k1 - b.k0 + 1)


def x_overlap(a, b):
    return max(0, min(a.i1, b.i1) - max(a.i0, b.i0) + 1)


def contains(outer, inner, tol=1):
    return (inner.i0 >= outer.i0 - tol and inner.i1 <= outer.i1 + tol
            and inner.k0 >= outer.k0 - tol and inner.k1 <= outer.k1 + tol)


# ----------------------------------------------------------------- one source

def run_source(radargram, branch, boxes, cfg):
    """Pipeline call + per-group picks/fit (read-only use of band.py internals)."""
    from gpr_detection.depth import band as B
    from gpr_detection.preprocess.timezero import find_time_zero

    width = radargram.data.shape[1]
    excluded = B.band_exclude_traces(boxes, width, None, cfg)
    zero = find_time_zero(radargram, branch.filtered, exclude_traces=excluded)
    results = B.estimate_band_depths(radargram, boxes, zero, cfg=cfg, branch=branch)
    if not boxes:
        return {"zero": zero, "groups": []}
    context = B._context(radargram, zero, branch)
    groups = B._merged_groups(context, boxes, cfg)
    assert len(groups) == len(results), (len(groups), len(results))
    k_ref = 0.0 if zero.k_zero is None else float(zero.k_zero)
    out = []
    for members, result in zip(groups, results):
        group = B._evaluate_group(context, boxes, members, cfg)
        picks, fit = group.picks, group.fit
        if result.band is not None:
            assert result.band.n_picks == picks.n_picks
        residual = None
        if fit is not None and picks.n_picks:
            model = np.array([fit.tau_at(float(t)) for t in picks.trace_idx])
            residual = (picks.k_pick - k_ref) - model
        member_boxes = [boxes[i] for i in members]
        ends = sorted({e for b in member_boxes for e in (b.i0, b.i1)})
        out.append({
            "members": members,
            "union": picks.box,
            "member_boxes": member_boxes,
            "outer_ends": (picks.box.i0, picks.box.i1),
            "fragment_ends": ends,
            "status": group.status.value,
            "has_fit": fit is not None,
            "ok": fit is not None and group.status.value == "ok",
            "ref_sample": None if result.band is None else float(result.band.ref_sample),
            "model": None if result.band is None else result.band.model,
            "coverage": float(picks.coverage),
            "n_picks": int(picks.n_picks),
            "sigma_res": None if fit is None else float(fit.sigma_res),
            "confidence": None if result.band is None else float(result.band.confidence),
            "bg_mode": picks.bg_mode,
            "flags": tuple(result.flags),
            "trace_idx": picks.trace_idx.copy(),
            "k_pick": picks.k_pick.copy(),
            "residual": residual,
            "edge_picks": B._edge_picks(picks),
        })
    return {"zero": zero, "groups": out}


def single_fragment_edges(radargram, branch, boxes, cfg):
    """Single-box picks: facing-end medians (v1.3 check) and the BandPicks (v1.4 check)."""
    from gpr_detection.depth import band as B
    from gpr_detection.preprocess.timezero import find_time_zero
    excluded = B.band_exclude_traces(boxes, radargram.data.shape[1], None, cfg)
    zero = find_time_zero(radargram, branch.filtered, exclude_traces=excluded)
    context = B._context(radargram, zero, branch)
    picks = [B._picks_for(context, boxes, (i,), cfg) for i in range(len(boxes))]
    return [B._edge_picks(p) for p in picks], picks


def pair_difference_v14(a, b, boxes, picks):
    """Spec v1.4 section 3-4 item 2 difference (None when band.py predates v1.4)."""
    from gpr_detection.depth import band as B
    fn = getattr(B, "pair_pick_difference", None)
    return None if fn is None else float(fn(a, b, boxes, picks))


# ----------------------------------------------------------------- one crop

def evaluate_crop(job):
    file_name, pred_boxes_xyxy = job
    from gpr_detection.depth.band import pixel_limits
    from gpr_detection.depth.config import BandConfig
    from gpr_detection.io._common import read_grayscale_file
    from gpr_detection.io.radargram import Radargram
    from gpr_detection.preprocess.filters import prepare_fitting_branch

    image = read_grayscale_file(V1_DIR / "images" / file_name)
    height, width = image.shape
    stem = Path(file_name).stem
    radargram = Radargram(data=image.astype(np.float32), dt_ns=None, dx_m=None, air_gap="unknown",
                          amplitude_processed=False, dataset="aihub", line_id=stem)
    branch = prepare_fitting_branch(radargram)
    period = branch.period_px
    limits = pixel_limits(period)
    # explicit gaps: BandConfig() defaults to 32 px since spec v1.4 (was 8)
    cfg8 = replace(BandConfig(), merge_gap_px=8)
    cfg32 = replace(cfg8, merge_gap_px=32)

    gt_v1 = yolo_band_boxes(V1_DIR / "labels" / f"{stem}.txt", width, height)
    gt_v2 = yolo_band_boxes(V2_DIR / "labels" / f"{stem}.txt", width, height)
    sources = {"gt_v1": (gt_v1, cfg8), "gt_v1_G32": (gt_v1, cfg32), "gt_v2": (gt_v2, cfg8)}
    for run in PRED_RUNS:
        boxes = [b for b in (xyxy_box(*xy, width, height) for xy in pred_boxes_xyxy.get(run, [])) if b]
        sources[f"pred_{run}"] = (boxes, cfg8)
        sources[f"pred_{run}_G32"] = (boxes, cfg32)
    runs = {name: run_source(radargram, branch, boxes, cfg) for name, (boxes, cfg) in sources.items()}

    # v1 -> v2 membership
    member_of = []
    for b1 in gt_v1:
        owners = [j for j, b2 in enumerate(gt_v2) if contains(b2, b1)]
        member_of.append(owners[0] if owners else -1)
    v1_members = defaultdict(list)
    for i, j in enumerate(member_of):
        v1_members[j].append(i)

    # facing-end check of the v1 fragments merged by v2
    pair_rows = []
    if any(len(v) > 1 for k, v in v1_members.items() if k >= 0):
        edges, frag_picks = single_fragment_edges(radargram, branch, gt_v1, cfg8)
        for j, idx in v1_members.items():
            if j < 0 or len(idx) < 2:
                continue
            idx = sorted(idx, key=lambda i: (gt_v1[i].i0, i))
            for a, b in zip(idx, idx[1:]):
                gap = gt_v1[b].i0 - gt_v1[a].i1 - 1
                diff = edges[a][1] - edges[b][0]
                pair_rows.append({"band": j, "gap": gap, "end_diff_px": float(diff),
                                  "pair_diff_v14_px": pair_difference_v14(a, b, gt_v1, frag_picks),
                                  "tol_px": float(limits.merge_tolerance), "P": period})

    band_rows, trace_rows, fp_counts = [], [], {}
    for name, res in runs.items():
        groups = res["groups"]
        matched_any = [False] * len(groups)
        for j, bv2 in enumerate(gt_v2):
            matches = []
            for g_i, g in enumerate(groups):
                ov = x_overlap(g["union"], bv2)
                if ov > 0 and y_overlap_ok(g["union"], bv2):
                    matches.append((g, ov))
                    matched_any[g_i] = True
            fit = [(g, ov) for g, ov in matches if g["has_fit"] and g["ref_sample"] is not None]
            ok = [(g, ov) for g, ov in fit if g["ok"]]
            width_b = bv2.i1 - bv2.i0 + 1

            def cov(items):
                covered = np.zeros(width_b, bool)
                for g, _ in items:
                    lo, hi = max(g["union"].i0, bv2.i0), min(g["union"].i1, bv2.i1)
                    if hi >= lo:
                        covered[lo - bv2.i0: hi - bv2.i0 + 1] = True
                return float(covered.mean())

            def wmean(items):
                if not items:
                    return None
                w = np.array([ov for _, ov in items], float)
                r = np.array([g["ref_sample"] for g, _ in items], float)
                return float((w * r).sum() / w.sum())

            def spread(items):
                if len(items) < 2:
                    return 0.0 if items else None
                r = [g["ref_sample"] for g, _ in items]
                return float(max(r) - min(r))

            band_rows.append({
                "file_name": file_name, "band": j, "source": name, "P": period,
                "n_v1_members": len(v1_members.get(j, [])), "band_width": width_b,
                "band_height": bv2.k1 - bv2.k0 + 1,
                "n_matched": len(matches), "n_fit": len(fit), "n_ok": len(ok),
                "cov_fit": cov(fit), "cov_ok": cov(ok),
                "ref_fit": wmean(fit), "ref_ok": wmean(ok),
                "spread_fit": spread(fit), "spread_ok": spread(ok),
                "inconsistent_any": any("inconsistent_picks" in g["flags"] for g, _ in matches),
            })
        fp_counts[name] = int(sum(1 for m, g in zip(matched_any, groups) if not m))

    # per-trace picks gt_v1 vs gt_v2 (traces of v1 member boxes) and gap traces of v2 merged boxes
    def trace_map(groups):
        m = {}
        for g in groups:
            for t, k in zip(g["trace_idx"], g["k_pick"]):
                m[int(t)] = float(k)
        return m

    def resid_map(groups):
        m = {}
        for g in groups:
            if g["residual"] is None:
                continue
            for t, r in zip(g["trace_idx"], g["residual"]):
                m[int(t)] = float(r)
        return m

    pick1 = trace_map(runs["gt_v1"]["groups"])
    pick2 = trace_map(runs["gt_v2"]["groups"])
    res2 = resid_map(runs["gt_v2"]["groups"])
    for j, bv2 in enumerate(gt_v2):
        idx = v1_members.get(j, [])
        member_traces = set()
        for i in idx:
            member_traces.update(range(gt_v1[i].i0, gt_v1[i].i1 + 1))
        for t in range(bv2.i0, bv2.i1 + 1):
            in_member = t in member_traces
            k0_v1 = min((gt_v1[i].k0 for i in idx if gt_v1[i].i0 <= t <= gt_v1[i].i1), default=None)
            trace_rows.append({
                "band": j, "merged": len(idx) > 1, "in_member": in_member, "P": period,
                "k_v1": pick1.get(t) if in_member else None, "k_v2": pick2.get(t),
                "r_v2": res2.get(t), "k0_v1": k0_v1, "k0_v2": bv2.k0,
            })

    # edge vs interior residuals per source
    edge_rows = []
    for name, res in runs.items():
        for g in res["groups"]:
            if g["residual"] is None:
                continue
            for t, r in zip(g["trace_idx"], g["residual"]):
                d_outer = min(abs(int(t) - g["outer_ends"][0]), abs(int(t) - g["outer_ends"][1]))
                d_frag = min(abs(int(t) - e) for e in g["fragment_ends"])
                edge_rows.append((name, d_outer < EDGE_TRACES, d_frag < EDGE_TRACES, abs(float(r)), period))

    # result-level rows
    result_rows = []
    for name, res in runs.items():
        for g in res["groups"]:
            result_rows.append({
                "source": name, "n_members": len(g["members"]), "width": g["union"].i1 - g["union"].i0 + 1,
                "height": g["union"].k1 - g["union"].k0 + 1, "status": g["status"], "has_fit": g["has_fit"],
                "ok": g["ok"], "model": g["model"], "coverage": g["coverage"], "n_picks": g["n_picks"],
                "sigma_res_P": None if g["sigma_res"] is None or not period else g["sigma_res"] / period,
                "confidence": g["confidence"], "bg_mode": g["bg_mode"], "flags": " ".join(g["flags"]),
            })
    n_input = {name: len(boxes) for name, (boxes, _) in sources.items()}
    return {"file_name": file_name, "P": period, "band_rows": band_rows, "trace_rows": trace_rows,
            "edge_rows": edge_rows, "result_rows": result_rows, "pair_rows": pair_rows,
            "fp_counts": fp_counts, "n_input": n_input, "n_v1": len(gt_v1), "n_v2": len(gt_v2),
            "unowned_v1": int(sum(1 for j in member_of if j < 0))}


# ----------------------------------------------------------------- aggregate

def q(values, ps=(0, 10, 25, 50, 75, 90, 100)):
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if v.size == 0:
        return None
    return {f"p{p}": round(float(np.percentile(v, p)), 3) for p in ps} | {"n": int(v.size),
                                                                         "mean": round(float(v.mean()), 3)}


def frac(flags):
    flags = list(flags)
    return None if not flags else round(sum(bool(f) for f in flags) / len(flags), 4)


def aggregate(crops):
    sources = list(crops[0]["n_input"].keys())
    agg = {"n_crops": len(crops), "n_v1_boxes": sum(c["n_v1"] for c in crops),
           "n_v2_boxes": sum(c["n_v2"] for c in crops), "unowned_v1": sum(c["unowned_v1"] for c in crops),
           "P_px": q([c["P"] for c in crops])}
    # result level
    res = defaultdict(list)
    for c in crops:
        for r in c["result_rows"]:
            res[r["source"]].append(r)
    agg["results"] = {}
    for s in sources:
        rows = res[s]
        flags = Counter(f for r in rows for f in r["flags"].split())
        agg["results"][s] = {
            "n_input_boxes": sum(c["n_input"][s] for c in crops),
            "n_results": len(rows),
            "n_fit": sum(r["has_fit"] for r in rows),
            "n_ok": sum(r["ok"] for r in rows),
            "fit_rate": frac(r["has_fit"] for r in rows), "ok_rate": frac(r["ok"] for r in rows),
            "merged_results": sum(r["n_members"] > 1 for r in rows),
            "width": q([r["width"] for r in rows]), "n_picks": q([r["n_picks"] for r in rows]),
            "coverage": q([r["coverage"] for r in rows]),
            "sigma_res_P": q([r["sigma_res_P"] for r in rows]),
            "confidence": q([r["confidence"] for r in rows]),
            "models": dict(Counter(r["model"] for r in rows if r["model"])),
            "bg_mode": dict(Counter(r["bg_mode"] for r in rows)),
            "flags": {k: flags[k] for k in ("inconsistent_picks", "insufficient_picks", "band_concave",
                                            "top_at_window_edge", "fragment_of_group", "bg_none",
                                            "touches_profile_edge_left", "touches_profile_edge_right")},
            "unmatched_results": sum(c["fp_counts"][s] for c in crops),
        }
    # band level (v2 GT bands as reference objects)
    bands = defaultdict(list)
    for c in crops:
        for r in c["band_rows"]:
            bands[r["source"]].append(r)
    ref = {(r["file_name"], r["band"]): r for r in bands["gt_v2"]}
    ref1 = {(r["file_name"], r["band"]): r for r in bands["gt_v1"]}
    agg["bands"] = {}
    for s in sources:
        out = {}
        for subset, keep in (("all", lambda r: True), ("merged", lambda r: r["n_v1_members"] > 1),
                             ("single", lambda r: r["n_v1_members"] <= 1)):
            rows = [r for r in bands[s] if keep(r)]
            if not rows:
                continue
            d2 = [(r["ref_fit"] - ref[(r["file_name"], r["band"])]["ref_fit"]) / r["P"]
                  for r in rows if r["ref_fit"] is not None and ref[(r["file_name"], r["band"])]["ref_fit"]
                  is not None and r["P"]]
            d1 = [(r["ref_fit"] - ref1[(r["file_name"], r["band"])]["ref_fit"]) / r["P"]
                  for r in rows if r["ref_fit"] is not None and ref1[(r["file_name"], r["band"])]["ref_fit"]
                  is not None and r["P"]]
            d2px = [(r["ref_fit"] - ref[(r["file_name"], r["band"])]["ref_fit"])
                    for r in rows if r["ref_fit"] is not None and ref[(r["file_name"], r["band"])]["ref_fit"]
                    is not None]
            out[subset] = {
                "n_bands": len(rows),
                "has_fit": frac(r["n_fit"] > 0 for r in rows),
                "has_ok": frac(r["n_ok"] > 0 for r in rows),
                "n_fit_per_band": q([r["n_fit"] for r in rows]),
                "multi_result_bands": frac(r["n_fit"] > 1 for r in rows),
                "cov_fit": q([r["cov_fit"] for r in rows]), "cov_ok": q([r["cov_ok"] for r in rows]),
                "spread_fit_P": q([r["spread_fit"] / r["P"] for r in rows
                                   if r["n_fit"] > 1 and r["P"]]),
                "spread_gt_0.5P": frac(r["spread_fit"] / r["P"] > 0.5 for r in rows if r["n_fit"] > 1 and r["P"]),
                "abs_delta_vs_gt_v2_P": q(np.abs(d2).tolist()),
                "delta_vs_gt_v2_P": q(d2),
                "abs_delta_vs_gt_v2_px": q(np.abs(d2px).tolist()),
                "delta_gt_v2_gt_0.25P": frac(abs(x) > 0.25 for x in d2),
                "delta_gt_v2_gt_0.5P": frac(abs(x) > 0.5 for x in d2),
                "delta_gt_v2_gt_1P": frac(abs(x) > 1.0 for x in d2),
                "abs_delta_vs_gt_v1_P": q(np.abs(d1).tolist()),
                "delta_gt_v1_gt_0.5P": frac(abs(x) > 0.5 for x in d1),
                "inconsistent_any": frac(r["inconsistent_any"] for r in rows if r["n_matched"]),
            }
        agg["bands"][s] = out
    # per-trace pick agreement gt_v1 vs gt_v2
    tr = [t for c in crops for t in c["trace_rows"]]
    agg["trace_picks"] = {}
    for subset, keep in (("all", lambda t: True), ("merged", lambda t: t["merged"]),
                         ("single", lambda t: not t["merged"])):
        both = [t for t in tr if keep(t) and t["in_member"] and t["k_v1"] is not None and t["k_v2"] is not None]
        dk = np.array([t["k_v2"] - t["k_v1"] for t in both], float)
        dkP = np.array([(t["k_v2"] - t["k_v1"]) / t["P"] for t in both if t["P"]], float)
        member = [t for t in tr if keep(t) and t["in_member"]]
        agg["trace_picks"][subset] = {
            "member_traces": len(member),
            "picked_v1": frac(t["k_v1"] is not None for t in member),
            "picked_v2": frac(t["k_v2"] is not None for t in member),
            "both": len(both),
            "abs_dk_px": q(np.abs(dk).tolist()),
            "same_le_0.5px": frac(np.abs(dk) <= 0.5),
            "gt_0.25P": frac(np.abs(dkP) > 0.25), "gt_0.5P": frac(np.abs(dkP) > 0.5),
            "gt_0.75P": frac(np.abs(dkP) > 0.75),
            "v2_earlier_gt_0.5P": frac(dkP < -0.5), "v2_later_gt_0.5P": frac(dkP > 0.5),
        }
    # dependence of the pick on the box top edge (k0)
    both_m = [t for t in tr if t["merged"] and t["in_member"] and t["k_v1"] is not None
              and t["k_v2"] is not None and t["P"] and t["k0_v1"] is not None]
    top = {}
    for name, keep in (("dk0_eq_0", lambda d: d == 0), ("dk0_m1_to_m4", lambda d: -4 <= d <= -1),
                       ("dk0_le_m5", lambda d: d <= -5)):
        sel = [t for t in both_m if keep(t["k0_v2"] - t["k0_v1"])]
        dkP = [(t["k_v2"] - t["k_v1"]) / t["P"] for t in sel]
        top[name] = {"n": len(sel), "dk_P": q(dkP, (10, 50, 90)), "abs_gt_0.5P": frac(abs(x) > 0.5 for x in dkP),
                     "earlier_gt_0.5P": frac(x < -0.5 for x in dkP)}
    all_v1 = [t for t in tr if t["in_member"] and t["k_v1"] is not None and t["P"] and t["k0_v1"] is not None]
    top["pick_minus_k0_v1_P"] = q([(t["k_v1"] - t["k0_v1"]) / t["P"] for t in all_v1])
    top["pick_minus_k0_v2_P"] = q([(t["k_v2"] - t["k0_v2"]) / t["P"] for t in tr if t["k_v2"] is not None and t["P"]])
    agg["box_top_dependence"] = top
    gap = [t for t in tr if t["merged"] and not t["in_member"]]
    mem_merged = [t for t in tr if t["merged"] and t["in_member"]]
    agg["gap_traces"] = {
        "n": len(gap), "picked_v2": frac(t["k_v2"] is not None for t in gap),
        "abs_r_v2_P": q([abs(t["r_v2"]) / t["P"] for t in gap if t["r_v2"] is not None and t["P"]]),
        "member_abs_r_v2_P": q([abs(t["r_v2"]) / t["P"] for t in mem_merged if t["r_v2"] is not None and t["P"]]),
    }
    # edge effect
    edge = defaultdict(lambda: {"outer": [], "frag": [], "interior": []})
    for c in crops:
        for name, is_outer, is_frag, r, P in c["edge_rows"]:
            if not P:
                continue
            key = "outer" if is_outer else ("frag" if is_frag else "interior")
            edge[name][key].append(r / P)
    agg["edge_residual_P"] = {
        s: {k: q(v, (50, 90)) for k, v in edge[s].items()} | {
            "edge_share": frac([1] * (len(edge[s]["outer"]) + len(edge[s]["frag"])) + [0] * len(edge[s]["interior"]))
        } for s in sources}
    # v2 merged pairs: facing-end consistency
    pairs = [p for c in crops for p in c["pair_rows"]]
    agg["merged_pairs"] = {}
    for name, lo, hi in (("all", -10**6, 10**6), ("gap_le8", -10**6, 8), ("gap_9_16", 9, 16), ("gap_17_32", 17, 32)):
        sel = [p for p in pairs if lo <= p["gap"] <= hi]
        valid = [p for p in sel if not math.isnan(p["end_diff_px"])]
        agg["merged_pairs"][name] = {
            "n": len(sel), "n_valid": len(valid),
            "abs_end_diff_P": q([abs(p["end_diff_px"]) / p["P"] for p in valid if p["P"]]),
            "exceeds_tol": frac(abs(p["end_diff_px"]) > p["tol_px"] for p in valid),
            "exceeds_0.5P": frac(abs(p["end_diff_px"]) / p["P"] > 0.5 for p in valid if p["P"]),
            "exceeds_1P": frac(abs(p["end_diff_px"]) / p["P"] > 1.0 for p in valid if p["P"]),
            "exceeds_tol_v14": frac(abs(p["pair_diff_v14_px"]) > p["tol_px"] for p in sel
                                    if p.get("pair_diff_v14_px") is not None
                                    and not math.isnan(p["pair_diff_v14_px"])),
        }
    return agg


# ----------------------------------------------------------------- figures

def public_figure(crops, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    ink, muted, c1, c2, c3 = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), facecolor="#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=muted)
    # (a) per-trace pick difference gt_v2 - gt_v1 (units of P), merged bands
    tr = [t for c in crops for t in c["trace_rows"]]
    bins = [("병합 없는 띠", lambda t: not t["merged"]),
            ("병합 띠, 위 끝 같음", lambda t: t["merged"] and t["k0_v2"] - t["k0_v1"] == 0),
            ("병합 띠, 위 끝 1–4 px 위", lambda t: t["merged"] and -4 <= t["k0_v2"] - t["k0_v1"] <= -1),
            ("병합 띠, 위 끝 ≥5 px 위", lambda t: t["merged"] and t["k0_v2"] - t["k0_v1"] <= -5)]
    labels, values, ns = [], [], []
    for lab, keep in bins:
        sel = [t for t in tr if t["in_member"] and t["k_v1"] is not None and t["k_v2"] is not None
               and t["P"] and t["k0_v1"] is not None and keep(t)]
        d = np.array([(t["k_v2"] - t["k_v1"]) / t["P"] for t in sel])
        labels.append(lab)
        values.append(100.0 * float(np.mean(np.abs(d) > 0.5)) if len(d) else 0.0)
        ns.append(len(d))
    y = np.arange(len(labels))
    axes[0].barh(y, values, color=c1, height=0.55)
    for yi, v, n in zip(y, values, ns):
        axes[0].text(v + 0.8, yi, f"{v:.1f} % (n={n})", va="center", fontsize=8, color=ink)
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(values) * 1.45 + 1)
    axes[0].set_xlabel("v1↔v2 정답 픽이 0.5 P 넘게 다른 트레이스 비율 [%]", color=ink)
    axes[0].set_title("(a) 합집합 박스 위 끝이 올라가면 픽이 바뀐다", color=ink, fontsize=10, loc="left")
    # (b) band-level |delta ref| vs gt_v2, sources
    bands = defaultdict(list)
    for c in crops:
        for r in c["band_rows"]:
            bands[r["source"]].append(r)
    ref = {(r["file_name"], r["band"]): r["ref_fit"] for r in bands["gt_v2"]}
    names = [("gt_v1", "v1 정답 (G 8)"), ("gt_v1_G32", "v1 정답 (G 32)"), ("pred_v1_s0", "v1 모델 s0"),
             ("pred_v1_s0_G32", "v1 모델 s0 (G 32)"), ("pred_v2_s0", "v2 모델 s0")]
    data = []
    for s, _ in names:
        vals = [abs(r["ref_fit"] - ref[(r["file_name"], r["band"])]) / r["P"] for r in bands[s]
                if r["n_v1_members"] > 1 and r["ref_fit"] is not None and ref[(r["file_name"], r["band"])] is not None
                and r["P"]]
        data.append(vals)
    bp = axes[1].boxplot(data, orientation="horizontal", showfliers=False, patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cde2fb")
        patch.set_edgecolor(c1)
    for med in bp["medians"]:
        med.set_color(ink)
    axes[1].set_yticks(range(1, len(names) + 1), [f"{lab} (n={len(d)})" for (_, lab), d in zip(names, data)],
                       fontsize=8)
    axes[1].axvline(0.5, color=muted, linestyle="--", linewidth=1)
    axes[1].set_xlabel("|대표 행 − v2 정답 대표 행| [주기 P], v2 병합 띠만", color=ink)
    axes[1].set_title("(b) 띠 하나의 대표 깊이(px) 차", color=ink, fontsize=10, loc="left")
    # (c) facing-end difference of v1 fragments merged by v2, by gap
    pairs = [p for c in crops for p in c["pair_rows"] if not math.isnan(p["end_diff_px"]) and p["P"]]
    for (lo, hi), color, lab in (((-999, 8), c1, "간격 ≤ 8 px"), ((9, 32), c2, "간격 9–32 px")):
        d = [abs(p["end_diff_px"]) / p["P"] for p in pairs if lo <= p["gap"] <= hi]
        axes[2].hist(np.clip(d, 0, 3), bins=np.linspace(0, 3, 31), color=color, alpha=0.75, density=True,
                     label=f"{lab} (n={len(d)})")
    axes[2].axvline(0.5, color=muted, linestyle="--", linewidth=1)
    axes[2].set_xlabel("v2가 병합한 v1 조각의 맞닿는 끝 픽 차 [주기 P]", color=ink)
    axes[2].set_title("(c) 병합 쌍의 같은 사건 여부", color=ink, fontsize=10, loc="left")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("pipe_band 박스 규약과 띠 깊이 경로 — AIHub fold 0 crop (스케일 없음, px·주기 단위)",
                 color=ink, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="#fcfcfb")
    plt.close(fig)


def local_gallery(crops, out_dir, n=8):
    """AIHub crops with v1/v2 GT boxes and per-trace picks. LOCAL ONLY (data/ is git-ignored)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gpr_detection.io._common import read_grayscale_file
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for c in crops:
        for p in c["pair_rows"]:
            if not math.isnan(p["end_diff_px"]) and p["P"]:
                cases.append((abs(p["end_diff_px"]) / p["P"], c["file_name"], p["band"], p["gap"]))
    cases.sort(reverse=True)
    rng = np.random.default_rng(0)
    worst = cases[:n]
    rest = [cases[i] for i in rng.choice(len(cases), size=min(n, len(cases)), replace=False)] if cases else []
    for tag, sel in (("worst_end_diff", worst), ("random", rest)):
        fig, axes = plt.subplots(2, max(1, len(sel) // 2), figsize=(3 * max(1, len(sel) // 2), 6.4), squeeze=False)
        for ax, (score, fname, band, gap) in zip(axes.ravel(), sel):
            img = read_grayscale_file(V1_DIR / "images" / fname)
            h, w = img.shape
            ax.imshow(img, cmap="gray", aspect="auto")
            for d, col, ls in ((V1_DIR, "#2a78d6", "-"), (V2_DIR, "#eb6834", "--")):
                for b in yolo_band_boxes(d / "labels" / f"{Path(fname).stem}.txt", w, h):
                    ax.add_patch(plt.Rectangle((b.i0, b.k0), b.i1 - b.i0 + 1, b.k1 - b.k0 + 1, fill=False,
                                               edgecolor=col, linestyle=ls, linewidth=1.2))
            ax.set_title(f"끝 차 {score:.2f}P, 간격 {gap}px", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"[로컬 전용 - AIHub 원본] 파랑 실선 v1 정답, 주황 점선 v2 정답 ({tag})", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"band_convention_{tag}.png", dpi=120)
        plt.close(fig)


# ----------------------------------------------------------------- main

def _has_v14() -> bool:
    from gpr_detection.depth import band as B
    return hasattr(B, "pair_pick_difference")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--suffix", default="", help="output name suffix, e.g. _v14 (keeps earlier outputs)")
    a = ap.parse_args()
    out_json = OUT_JSON.with_name(OUT_JSON.stem + a.suffix + OUT_JSON.suffix)
    out_rows = OUT_ROWS.with_name(OUT_ROWS.stem + a.suffix + OUT_ROWS.suffix)
    out_png = OUT_PNG.with_name(OUT_PNG.stem + a.suffix + OUT_PNG.suffix)
    out_fig_local = OUT_FIG_LOCAL.with_name(OUT_FIG_LOCAL.name + a.suffix)
    try:
        names = [Path(p.strip()).name for p in LIST_TXT.read_text(encoding="utf-8").splitlines() if p.strip()]
        preds = defaultdict(lambda: defaultdict(list))
        with PRED_CSV.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                preds[row["file_name"]][row["run"]].append(
                    tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2")))
    except OSError as exc:
        print(json.dumps({"status": "input_error", "error": repr(exc)}))
        return 2
    jobs = []
    for name in names:
        label = V1_DIR / "labels" / f"{Path(name).stem}.txt"
        if label.exists() and any(l.split()[:1] == ["1"] for l in label.read_text(encoding="utf-8").splitlines()):
            jobs.append((name, {k: v for k, v in preds.get(name, {}).items()}))
    if a.limit:
        jobs = jobs[: a.limit]
    start = time.perf_counter()
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            crops = list(pool.map(evaluate_crop, jobs, chunksize=4))
    else:
        crops = [evaluate_crop(j) for j in jobs]
    elapsed = time.perf_counter() - start
    agg = aggregate(crops)
    agg["meta"] = {"seconds": round(elapsed, 1), "workers": a.workers, "crop_list": str(LIST_TXT.relative_to(ROOT)),
                   "pred_csv": "data/audit/band_convention_preds.csv (local)", "edge_traces": EDGE_TRACES,
                   "seed": 0, "suffix": a.suffix,
                   "band_py_has_pair_pick_difference": pair_difference_v14 is not None and _has_v14()}
    out_json.write_text(json.dumps(agg, ensure_ascii=False, indent=1), encoding="utf-8")
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for c in crops for r in c["band_rows"]]
    with out_rows.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not a.no_figures:
        public_figure(crops, out_png)
        local_gallery(crops, out_fig_local)
    print(json.dumps({"status": "ok", "crops": len(crops), "seconds": round(elapsed, 1),
                      "json": str(out_json), "png": None if a.no_figures else str(out_png)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
