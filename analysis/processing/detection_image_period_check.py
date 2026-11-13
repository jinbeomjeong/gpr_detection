"""Vertical carrier period of real lines, measured like the AIHub P_px (spec_depth v1.3 section 3).

AIHub P_px (median 4.06 px) comes from the trace-averaged |rfft| of DC-removed
256x256 crops. For the L1 scale rule we need the same quantity in samples for
4TU / Guangzhou / GPR-SD, and whether it depends on which section is used
(raw DC-removed, after background removal, or a window around a pipe apex).

Inputs (read-only): 4TU true apexes of the blinded review (type T), the first
eight Guangzhou IDS 400 MHz files of the 30 ns and 50 ns groups (sorted file_id),
GPR-SD pipeline marks. No random numbers.
Output: docs/processing/detection_image_period_check_v1.json
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\workspace\gpr_detection")
sys.path.insert(0, str(ROOT / "src"))

from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.guangzhou import load_guangzhou_radargram  # noqa: E402
from gpr_detection.preprocess.filters import (  # noqa: E402
    dominant_period_px,
    prepare_fitting_branch,
    remove_dc,
)

GPR = Path(r"C:\workspace\GPR_Data")
FOURTU = GPR / "4TU_네덜란드_매설관GPR_가공본_261015"
GZ = GPR / "광저우대_중국_관로GPR_가공본_261016"
SD = GPR / "GPR-SD_중국USTC_도로GPR_가공본_261016"
OUT = ROOT / "docs" / "processing" / "detection_image_period_check_v1.json"


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def padded_period(section, pad=8):
    """Dominant period (samples) of the trace-averaged zero-padded |rfft| of a DC-removed window."""
    a = np.asarray(section, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    n = a.shape[0]
    nfft = 1 << int(np.ceil(np.log2(pad * n)))
    spec = np.abs(np.fft.rfft(a, nfft, axis=0)).mean(axis=1)
    f = np.fft.rfftfreq(nfft, d=1.0)
    ok = f >= 1.0 / 128.0
    return 1.0 / float(f[ok][np.argmax(spec[ok])])


def q(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"p10": round(float(np.percentile(a, 10)), 2), "p50": round(float(np.median(a)), 2),
            "p90": round(float(np.percentile(a, 90)), 2), "n": int(a.size)}


def fourtu():
    rows = [r for r in read_csv(ROOT / "docs" / "processing" / "m29c_apex_review.csv") if r["type"] == "T"]
    per_line = defaultdict(list)
    for r in rows:
        per_line[(r["activity_id"], r["path"])].append((int(r["trace"]), float(r["sample"])))
    res = {"raw_whole": [], "bg_whole": [], "bg_apex": [], "dt": []}
    for (act, path), apexes in sorted(per_line.items()):
        rg = load_fourtu_radargram(FOURTU, act, path)
        br = prepare_fitting_branch(rg)
        res["dt"].append(rg.dt_ns)
        res["raw_whole"].append(dominant_period_px(remove_dc(rg.data)) * rg.dt_ns)
        res["bg_whole"].append(dominant_period_px(br.background_removed) * rg.dt_ns)
        for i, k in apexes:
            k = int(round(k))
            i0, i1 = max(0, i - 10), min(rg.data.shape[1], i + 11)
            k0, k1 = max(0, k - int(3.0 / rg.dt_ns)), min(rg.data.shape[0], k + int(6.0 / rg.dt_ns))
            res["bg_apex"].append(padded_period(br.background_removed[k0:k1, i0:i1]) * rg.dt_ns)
    return {
        "n_lines": len(per_line), "n_apex": len(rows),
        "period_ns": {k: q(v) for k, v in res.items() if k != "dt"},
        "note": "raw_whole = DC-removed raw (AIHub estimator); bg_whole = after (1)-(4); bg_apex = apex window -3..+6 ns, +-10 traces, zero-padded x8",
    }


def guangzhou():
    rows = read_csv(GZ / "01_메타데이터" / "radargrams_index.csv")
    out = {}
    for tw in ("30.0", "50.0"):
        sel = sorted(
            (r for r in rows if r["format"] == "IDS .dt" and r["antenna_MHz"] == "400.0"
             and r["n_samples"] == "512" and r["time_window_ns"] == tw),
            key=lambda r: r["file_id"],
        )[:8]
        raw, bg, hdr = [], [], []
        for r in sel:
            rg = load_guangzhou_radargram(GZ, r["file_id"])
            br = prepare_fitting_branch(rg)
            raw.append(dominant_period_px(remove_dc(rg.data)) * rg.dt_ns)
            bg.append(padded_period(br.background_removed) * rg.dt_ns)
            hdr.append(1e3 / float(r["spectrum_peak_MHz"]))
        out[tw] = {"files": [r["file_id"] for r in sel], "raw_whole_ns": q(raw),
                   "bg_whole_padded_ns": q(bg), "header_peak_period_ns": q(hdr),
                   "dt_ns": float(sel[0]["dt_ns"])}
    return out


def gprsd():
    marks = [m for m in read_csv(SD / "01_메타데이터" / "marks.csv") if m["type"] == "pipeline"]
    whole, win = [], []
    seen = set()
    for m in marks:
        fid = m["files"].split(" | ")[0]
        a = np.load(SD / "05_radargrams_npy" / f"{fid}.npy", mmap_mode="r")
        if fid not in seen:
            seen.add(fid)
            whole.append(padded_period(np.asarray(a[:, 1000:4000:3])))
        t = int(m["location_trace"])
        i0, i1 = max(0, t - 100), min(a.shape[1], t + 101)
        win.append(padded_period(np.asarray(a[20:200, i0:i1])))
    return {"n_marks": len(marks), "whole_period_samples": q(whole), "mark_window_period_samples": q(win),
            "note": "samples (dt-free); window = samples 20-200, mark +-100 traces, first file of the road"}


def main():
    out = {"fourtu": fourtu(), "guangzhou": guangzhou(), "gprsd": gprsd()}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
