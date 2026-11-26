"""M3-4 box convention: box-aligned mean envelope profiles of AIHub tpipe boxes (train_pool only).

Per-box apex picking is unreliable in the shallow clutter of AIHub crops, so this script measures the label
habit without picking: profiles are aligned on the label box edges and averaged over boxes.

Section: pixel band-pass (prepare_fitting_branch.filtered) minus a horizontal moving median of 2 * box width + 1
(flat layers removed), envelope |hilbert| per column, normalised by the max envelope inside the box.
* vertical profile at the box centre band (centre +- 10 % width, mean over columns) on a grid of offsets in P
  from the box top y0 (-4 P .. +4 P) and from the box bottom y1 (-4 P .. +4 P);
  "first rise": first offset from y0 - 3 P downward where the profile reaches 0.5 (per box, also reported).
* horizontal profile: for column offset u = (x - xc) / (w / 2) in [-2.5, 2.5], max envelope in rows
  [y0, y1] ("in-band") and in rows [y0, y1 + h] ("deeper"), mean over boxes.
Same crop selection as m34_aihub_tpipe_box_convention.py (one image per crop unit, gain 1.0 > 0.5 > 0.25).
Output: analysis/detection/m34_aihub_box_profiles.json. stdout last line JSON.
Example: conda run -n gpr-dev python -I analysis/detection/m34_aihub_box_profiles.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[2]
SPLIT_CSV = ROOT / "data" / "splits" / "aihub_yolo_v1.csv"
DATASET = ROOT / "data" / "aihub_yolo_v1"
OUT_JSON = ROOT / "analysis" / "detection" / "m34_aihub_box_profiles.json"
GAIN_ORDER = ("1.0", "0.5", "0.25")
VGRID = np.round(np.arange(-4.0, 4.0001, 0.2), 2)
UGRID = np.round(np.arange(-2.5, 2.5001, 0.1), 2)


def env_cols(sec):
    n = sec.shape[0]
    return np.abs(hilbert(np.concatenate([sec, np.zeros_like(sec)], axis=0), axis=0))[:n]


def q(v):
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return None
    return {f"p{p}": round(float(np.percentile(a, p)), 3) for p in (5, 10, 25, 50, 75, 90, 95)} | {
        "n": int(a.size)}


def main() -> int:
    import pandas as pd
    from gpr_detection.io._common import read_grayscale_file
    from gpr_detection.io.radargram import Radargram
    from gpr_detection.preprocess.filters import prepare_fitting_branch

    split = pd.read_csv(SPLIT_CSV, dtype=str, keep_default_na=False)
    pool = split[(split["subset"] == "trainpool") & split["fold"].isin({"0", "1", "2", "3", "4"})]
    test_sessions = set(split.loc[split["subset"] == "test", "session"])
    assert not (set(pool["session"]) & test_sessions)
    pool0 = pool[pool["classes"].str.split(";").apply(lambda c: "0" in c)]
    chosen = []
    for key, grp in pool0.groupby("crop_key"):
        for g in GAIN_ORDER:
            sel = grp[grp["gain"] == g]
            if len(sel):
                chosen.append(sel.iloc[0]["file_name"])
                break

    top_prof, bot_prof, hin, hdeep, rise_top, rise_from_bottom, last_bottom = [], [], [], [], [], [], []
    top_peak, first_lobe = [], []
    for fname in chosen:
        img = read_grayscale_file(DATASET / "images" / fname)
        h, w = img.shape
        rg = Radargram(data=img.astype(np.float32), dt_ns=None, dx_m=None, air_gap="unknown",
                       amplitude_processed=False, dataset="aihub", line_id=Path(fname).stem)
        br = prepare_fitting_branch(rg)
        P = float(br.period_px) if br.period_px is not None else 4.06
        sec0 = br.filtered.astype(np.float64)
        for line in (DATASET / "labels" / f"{Path(fname).stem}.txt").read_text(encoding="utf-8").splitlines():
            p = line.split()
            if not p or int(p[0]) != 0:
                continue
            xc, yc, bw, bh = (float(v) for v in p[1:5])
            x0, x1, y0, y1 = (xc - bw / 2) * w, (xc + bw / 2) * w, (yc - bh / 2) * h, (yc + bh / 2) * h
            bwp, bhp = x1 - x0, y1 - y0
            win = int(2 * bwp) | 1
            env = env_cols(sec0 - median_filter(sec0, size=(1, win), mode="reflect"))
            r0, r1 = int(math.floor(y0)), int(math.ceil(y1))
            c0, c1 = int(math.floor(x0)), int(math.ceil(x1))
            norm = float(env[max(0, r0):min(h, r1), max(0, c0):min(w, c1)].max())
            half = max(2, int(round(0.10 * bwp)))
            cc = int(round(0.5 * (x0 + x1)))
            centre = env[:, max(0, cc - half):min(w, cc + half + 1)].mean(axis=1) / norm
            rows = np.arange(h)
            tp = np.interp(y0 + VGRID * P, rows, centre, left=np.nan, right=np.nan)
            bp = np.interp(y1 + VGRID * P, rows, centre, left=np.nan, right=np.nan)
            top_prof.append(tp)
            bot_prof.append(bp)
            # first rise >= 0.5 scanning down from y0 - 3 P
            fine = np.arange(-3.0, (y1 - y0) / P + 1e-9, 0.05)
            vals = np.interp(y0 + fine * P, rows, centre, left=np.nan, right=np.nan)
            idx = np.flatnonzero(vals >= 0.5)
            rise_top.append(float(fine[idx[0]]) if idx.size else np.nan)
            # strongest centre-band lobe near the box top: argmax in [y0 - 1 P, y0 + 2.5 P]
            near = np.arange(-1.0, 2.5001, 0.05)
            vn = np.interp(y0 + near * P, rows, centre, left=np.nan, right=np.nan)
            if np.isfinite(vn).any():
                top_peak.append(float(near[int(np.nanargmax(vn))]))
                # first local maximum >= 0.7 of that window max (shallowest strong lobe)
                m = np.nanmax(vn)
                fl = np.nan
                for i in range(1, len(vn) - 1):
                    if vn[i] >= 0.7 * m and vn[i] >= vn[i - 1] and vn[i] >= vn[i + 1]:
                        fl = float(near[i])
                        break
                first_lobe.append(fl)
            # last row (from the bottom up) with profile >= 0.3 inside [y0, y1 + 3 P], offset to y1 in P
            fine_b = np.arange(0.0, (y1 - y0) / P + 3.0, 0.05)
            vb = np.interp(y0 + fine_b * P, rows, centre, left=np.nan, right=np.nan)
            ib = np.flatnonzero(vb >= 0.3)
            last_bottom.append(float(fine_b[ib[-1]] - (y1 - y0) / P) if ib.size else np.nan)
            cols = np.arange(w)
            colmax_in = env[max(0, r0):min(h, r1), :].max(axis=0) / norm
            colmax_deep = env[max(0, r0):min(h, int(math.ceil(y1 + bhp))), :].max(axis=0) / norm
            xs = 0.5 * (x0 + x1) + UGRID * bwp / 2
            hin.append(np.interp(xs, cols, colmax_in, left=np.nan, right=np.nan))
            hdeep.append(np.interp(xs, cols, colmax_deep, left=np.nan, right=np.nan))

    def mean_prof(arr, grid):
        a = np.vstack(arr)
        return {f"{g:+.1f}": [round(float(np.nanmean(a[:, i])), 3), round(float(np.nanmedian(a[:, i])), 3)]
                for i, g in enumerate(grid)}

    out = {
        "source": {"subset": "trainpool folds 0-4", "test_read": False, "boxes": len(top_prof),
                   "crop_units": len(chosen)},
        "note": "values [mean, median] of the box-normalised centre-band envelope; offsets in P",
        "vertical_from_top_P": mean_prof(top_prof, VGRID),
        "vertical_from_bottom_P": mean_prof(bot_prof, VGRID),
        "first_rise_0.5_from_top_P": q(rise_top),
        "frac_first_rise_above_top": round(float(np.mean(np.asarray(rise_top) < 0)), 4),
        "frac_first_rise_ge_0.6P_below_top": round(float(np.mean(np.asarray(rise_top) >= 0.6)), 4),
        "last_0.3_minus_bottom_P": q(last_bottom),
        "top_window_peak_below_top_P": q(top_peak),
        "first_strong_lobe_below_top_P": q(first_lobe),
        "frac_first_lobe_ge_0.6P": round(float(np.nanmean(np.asarray(first_lobe) >= 0.6)), 4),
        "frac_first_lobe_lt_0": round(float(np.nanmean(np.asarray(first_lobe) < 0)), 4),
        "horizontal_in_band_u": mean_prof(hin, UGRID),
        "horizontal_deeper_u": mean_prof(hdeep, UGRID),
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"status": "ok", "boxes": len(top_prof), "out": str(OUT_JSON)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
