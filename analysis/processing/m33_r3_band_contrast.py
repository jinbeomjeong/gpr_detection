"""M3-3 pre-close item 2: band contrast in the detector input (L1 grey).

Question: is an R3 pseudo band at 0-18 dB as visible in the r0 L1 image as
the AIHub ``pipe_band`` (lpipe) labels the detector learned, and as the bands
the detector itself finds on the un-injected IDS lines?

Same metric in all three populations, measured on the grey image the detector
sees (q = grey - median grey of the content):

- ``c_vert_db``: 20 log10 RMS(q, box rows x central half of the box columns)
  / RMS(q, the same columns, 2 box heights above and below the box).
- ``c_lat_db``: same numerator / RMS(q, the box rows, columns beside the box
  (up to one box width each side, outside the band taper)).
- ``coh``: horizontal coherence inside the box = mean over box rows of
  (row mean of q)^2 / mean(q^2) (1 = perfectly flat band, ~1/n_cols = texture).

Populations
- AIHub: train-pool lpipe labels (class 1), gain 1.0 copies, label boxes.
  AIHub test is not opened.
- IDS-detected: conf > 0.25 ``pipe_band`` boxes of the un-injected r0 runs
  (``runs/eval/m33/r3/uninjected_ids``) on the 53 ``main_400`` lines.
- R3 P4 truths: the injected realizations rebuilt exactly as the change report
  does (``m33_r3_change_report.read_report_inputs``: plan, C1b record and
  responses hash-checked against the stored injected run), r0 image, truth box
  mapped to L1 through the image's own grid map. The background image at the
  same box is measured too (``bg``).

Read only; writes ``analysis/processing/m33_r3_band_contrast.json``.
Run: conda run -n gpr-dev python -I -B analysis/processing/m33_r3_band_contrast.py [--limit-lines N]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "detection"))
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

import m32_recipe_eval as runner  # noqa: E402
import m32_recipe_inputs as inputs  # noqa: E402
import m33_r3_change_report as report  # noqa: E402
from gpr_detection.eval import inject  # noqa: E402
from gpr_detection.preprocess.to_image import make_detection_image  # noqa: E402

R3 = ROOT / "runs" / "eval" / "m33" / "r3"
AIHUB = ROOT / "data" / "aihub_yolo_v1"
MANIFEST = ROOT / "data" / "splits" / "aihub_yolo_v1.csv"
UNITS = ROOT / "analysis" / "processing" / "m33_r3_recall_diag_units.csv"
OUT = ROOT / "analysis" / "processing" / "m33_r3_band_contrast.json"
SEED = 20261120
N_AIHUB_IMAGES = 600


def lower_priority() -> None:
    try:
        import ctypes

        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, 0x00004000)
    except Exception:  # pragma: no cover
        pass


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a, dtype=np.float64)))) if a.size else float("nan")


def db(x: float) -> float:
    return 20 * math.log10(x) if x > 0 and math.isfinite(x) else float("nan")


def metrics(q: np.ndarray, x1, y1, x2, y2, h_limit: int, side_gap: int = 0) -> dict | None:
    """Box ``[x1, x2) x [y1, y2)`` in px on the centred grey ``q`` (rows < h_limit)."""
    H, W = q.shape
    h_limit = min(h_limit, H)
    x1, x2 = max(0, int(math.floor(x1))), min(W, int(math.ceil(x2)))
    y1, y2 = max(0, int(math.floor(y1))), min(h_limit, int(math.ceil(y2)))
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 2:
        return None
    c0, c1 = x1 + bw // 4, x2 - bw // 4
    inside = q[y1:y2, c0:c1]
    above = q[max(0, y1 - 2 * bh) : y1, c0:c1]
    below = q[y2 : min(h_limit, y2 + 2 * bh), c0:c1]
    vert = np.concatenate([above.ravel(), below.ravel()])
    left = q[y1:y2, max(0, x1 - side_gap - bw) : max(0, x1 - side_gap)]
    right = q[y1:y2, min(W, x2 + side_gap) : min(W, x2 + side_gap + bw)]
    lat = np.concatenate([left.ravel(), right.ravel()])
    box = q[y1:y2, x1:x2].astype(np.float64)
    coh = float(np.mean(box.mean(axis=1) ** 2) / max(np.mean(box**2), 1e-12))
    r_in = rms(inside)
    return {
        "c_vert_db": db(r_in / rms(vert)) if vert.size >= 8 else float("nan"),
        "c_lat_db": db(r_in / rms(lat)) if lat.size >= 8 else float("nan"),
        "coh": coh,
        "rms_in": r_in,
        "w_px": bw,
        "h_px": bh,
    }


def quant(values) -> dict:
    v = np.array([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0}
    qs = np.percentile(v, [5, 25, 50, 75, 95])
    return {"n": int(v.size), **{f"p{p}": round(float(x), 3) for p, x in zip((5, 25, 50, 75, 95), qs)}}


def summarize(rows) -> dict:
    return {k: quant([r[k] for r in rows]) for k in ("c_vert_db", "c_lat_db", "coh", "h_px", "w_px")}


# ------------------------------------------------------------------ AIHub


def aihub_rows() -> list[dict]:
    rows = [
        r
        for r in read(MANIFEST)
        if r["subset"] == "trainpool" and r["gain"] == "1.0" and "1" in r["classes"].split(";")
    ]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    out = []
    for r in rows[:N_AIHUB_IMAGES]:
        stem = Path(r["file_name"]).stem
        label = AIHUB / "labels" / f"{stem}.txt"
        if not label.is_file():
            continue
        boxes = []
        for line in label.read_text(encoding="utf-8").split("\n"):
            parts = line.split()
            if len(parts) == 5:
                boxes.append((int(parts[0]), *map(float, parts[1:])))
        bands = [b for b in boxes if b[0] == 1]
        if not bands:
            continue
        img = cv2.imread(str(AIHUB / "images" / r["file_name"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape
        q = img.astype(np.float64) - float(np.median(img))
        for _, xc, yc, w, h in bands:
            m = metrics(q, (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H, H)
            if m:
                m["session"] = r["session"]
                out.append(m)
    return out


# -------------------------------------------------------------------- R3


def unit_hits() -> dict[str, float]:
    """injection_id -> mean over folds of covered / L (pipe_band, 50 % rule)."""
    acc = defaultdict(list)
    for u in read(UNITS):
        acc[u["injection_id"]].append(int(u["cov_band50"]) / int(u["L"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def r3_rows(limit_lines: int | None) -> tuple[list[dict], list[dict], list[dict], dict]:
    truths = read(R3 / "injected" / "truths.csv")
    scope = {
        t["injection_id"]: t
        for t in truths
        if t["dataset"] == "gz_ids"
        and t["ids_freq_stratum"] == "main_400"
        and t["model_group"] == "A0"
        and t["main_cell"] == "True"
        and t["lw_stratum"] in ("lt033", "033_05")
    }
    hits = unit_hits()
    by_realization = defaultdict(list)
    for t in scope.values():
        by_realization[t["realization_id"]].append(t)
    settings = report.ReportSettings(backgrounds=("gz_ids",))
    rin = report.read_report_inputs(settings)
    jobs = [j for j in report.line_jobs(rin, runner.load_line) if j.freq == "main_400"]
    if limit_lines:
        jobs = jobs[:limit_lines]
    uninj = defaultdict(list)
    for fold in range(5):
        for b in read(R3 / "uninjected_ids" / f"fold{fold}" / "boxes.csv"):
            if b["pp"] == "pre" and b["class_name"] == "pipe_band" and float(b["score"]) > 0.25:
                uninj[b["line_id"]].append(b)
    inj_rows, bg_rows, det_rows = [], [], []
    timing = {"lines": 0, "realizations": 0, "seconds": 0.0}
    started = time.perf_counter()
    for job in jobs:
        line = job.loader(job.ref)
        radargram = line.radargram
        context = inject.prepare_line_context(radargram, job.exclusions)
        responses = rin.library.for_line(job.ref.dataset, runner.line_time_window_ns(line))
        bg32 = replace(radargram, data=np.asarray(radargram.data).astype(np.float32))
        bg_img = make_detection_image(bg32, inputs.recipe_config(inputs.R0))
        bg_q = bg_img.line_image.astype(np.float64) - float(np.median(bg_img.line_image[: bg_img.content_height_px]))
        # detector's own bands on this line (all folds, near-duplicates kept)
        seen = set()
        for b in uninj.get(job.ref.line_id, []):
            key = tuple(round(float(b[k]) / 4) for k in ("x1", "y1", "x2", "y2"))
            if key in seen:
                continue
            seen.add(key)
            m = metrics(bg_q, float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]), bg_img.content_height_px)
            if m:
                m["line_id"] = job.ref.line_id
                det_rows.append(m)
        bg_done = set()
        for rid, specs in job.realizations.items():
            ids_here = by_realization.get(rid, [])
            if not ids_here:
                continue
            result = inject.inject_band_realization(context, specs, responses)
            if not result.truths:
                continue
            injected = replace(context.background, data=result.data.astype(np.float32))
            img = make_detection_image(injected, inputs.recipe_config(inputs.R0))
            q = img.line_image.astype(np.float64) - float(np.median(img.line_image[: img.content_height_px]))
            gm = img.grid_map
            timing["realizations"] += 1
            for t in ids_here:
                i0, i1, k0, k1 = (int(t[k]) for k in ("i0", "i1", "k0", "k1"))
                taper_px = int(t["taper_traces"]) / gm.traces_per_px
                box = (i0 / gm.traces_per_px, k0 / gm.samples_per_px, (i1 + 1) / gm.traces_per_px, (k1 + 1) / gm.samples_per_px)
                m = metrics(q, *box, img.content_height_px, side_gap=int(math.ceil(taper_px)))
                if not m:
                    continue
                m.update(
                    {
                        "injection_id": t["injection_id"],
                        "line_id": t["line_id"],
                        "snr": float(t["snr_db"]),
                        "pol": int(t["polarity"]),
                        "L": int(t["band_length_traces"]),
                        "recall": hits.get(t["injection_id"]),
                        "period_ratio": img.period_samples / context.period_samples,
                    }
                )
                inj_rows.append(m)
                if t["position_id"] not in bg_done:
                    gb = bg_img.grid_map
                    bbox = (i0 / gb.traces_per_px, k0 / gb.samples_per_px, (i1 + 1) / gb.traces_per_px, (k1 + 1) / gb.samples_per_px)
                    mb = metrics(bg_q, *bbox, bg_img.content_height_px, side_gap=int(math.ceil(taper_px)))
                    if mb:
                        mb["line_id"] = t["line_id"]
                        bg_rows.append(mb)
                    bg_done.add(t["position_id"])
        timing["lines"] += 1
    timing["seconds"] = round(time.perf_counter() - started, 1)
    return inj_rows, bg_rows, det_rows, timing


def main() -> None:
    lower_priority()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-lines", type=int, default=None)
    args = parser.parse_args()
    t0 = time.perf_counter()
    inj, bg, det, timing = r3_rows(args.limit_lines)
    ai = aihub_rows()
    out = {
        "seed_aihub_sample": SEED,
        "timing_r3": timing,
        "aihub_lpipe_trainpool_g1": summarize(ai) | {"n_sessions": len({r["session"] for r in ai})},
        "ids_detected_band_uninjected": summarize(det) | {"n_lines": len({r["line_id"] for r in det})},
        "p4_background_at_truth_box": summarize(bg),
        "p4_injected_by_snr": {str(s): summarize([r for r in inj if r["snr"] == s]) for s in (0.0, 6.0, 12.0, 18.0)},
        "p4_injected_by_snr_hit": {
            f"{s}|{'hit' if h else 'miss'}": summarize([r for r in inj if r["snr"] == s and ((r["recall"] or 0) > 0) == h])
            for s in (12.0, 18.0)
            for h in (True, False)
        },
        "p4_injected_by_L_snr18": {str(L): summarize([r for r in inj if r["snr"] == 18.0 and r["L"] == L]) for L in (40, 64, 100, 150)},
        "p4_recall_by_cvert_bin": {},
        "limit_lines": args.limit_lines,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    edges = (-1e9, 3, 6, 9, 12, 1e9)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in inj if lo <= r["c_vert_db"] < hi and r["recall"] is not None]
        out["p4_recall_by_cvert_bin"][f"[{lo},{hi})"] = {
            "n": len(sel),
            "mean_recall": round(sum(r["recall"] for r in sel) / len(sel), 4) if sel else None,
            "frac_any_hit": round(sum(1 for r in sel if r["recall"] > 0) / len(sel), 4) if sel else None,
        }
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "n_inj": len(inj), "n_bg": len(bg), "n_det": len(det), "n_aihub": len(ai), "elapsed_s": out["elapsed_s"], "out": str(OUT)}))


if __name__ == "__main__":
    main()
