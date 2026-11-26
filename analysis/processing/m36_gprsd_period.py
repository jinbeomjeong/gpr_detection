"""M3-6 D1a: structure of the period spectrum outside the label windows.

Grid windows: centre every 1,000 traces, +-150 traces, kept only if no trace lies within
+-400 traces of a pipeline mark (label window +-300 plus margin +-100, plan Q3).
Per window (stored amplitude, samples 20-200 for the spectrum, same range as the pilot):
  * trace-mean |rfft| (x8 zero padding) before and after subtracting the moving median
    (101 traces) -> overall peak, short peak (< 30 samples), long peak (>= 30 samples);
  * horizontal coherence: energy share of the moving-median (coherent) part in the
    low band (periods 35-80 samples) and in the carrier band (10-20 samples);
  * depth profile: RMS per sample of the low band and the carrier band.
No random numbers. Output: analysis/processing/m36_gprsd_period.json
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from m36_gprsd_common import (BG_WINDOW, ROADS, bandpass, excluded_mask, line_ids,  # noqa: E402
                              load_line, mean_spectrum, pipe_marks, qstats, road_of,
                              spectrum_peaks, write_json)

HALF = 150
STEP = 1000
S0, S1 = 20, 200
LOW_BAND = (35.0, 80.0)
CAR_BAND = (10.0, 20.0)


def main() -> None:
    t_start = time.time()
    marks = pipe_marks()
    rows = []
    prof = {r: {"low_raw": [], "car_raw": [], "low_res": [], "car_res": []} for r in ROADS}
    spec_acc = {r: {"raw": [], "res": []} for r in ROADS}
    f_ref = None
    for fid in line_ids():
        a = load_line(fid)
        n = a.shape[1]
        ex = excluded_mask(fid, n, marks)
        for c in range(STEP, n - HALF, STEP):
            i0, i1 = c - HALF, c + HALF + 1
            if ex[i0:i1].any():
                continue
            m0, m1 = max(0, i0 - BG_WINDOW // 2), min(n, i1 + BG_WINDOW // 2)
            big = a[:, m0:m1]
            from scipy import ndimage
            bgb = ndimage.median_filter(big, size=(1, BG_WINDOW), mode="nearest")
            raw = big[:, i0 - m0:i1 - m0]
            bg = bgb[:, i0 - m0:i1 - m0]
            res = raw - bg
            f, s_raw = mean_spectrum(raw[S0:S1])
            _, s_res = mean_spectrum(res[S0:S1])
            f_ref = f
            pk_raw = spectrum_peaks(f, s_raw)
            pk_res = spectrum_peaks(f, s_res)
            low_raw = bandpass(raw, *LOW_BAND)
            low_bg = bandpass(bg, *LOW_BAND)
            car_raw = bandpass(raw, *CAR_BAND)
            car_bg = bandpass(bg, *CAR_BAND)
            low_res = low_raw - low_bg
            car_res = car_raw - car_bg
            sl = slice(S0, S1)
            e = lambda x: float((x[sl] ** 2).sum())  # noqa: E731
            road = road_of(fid)
            rows.append({
                "file_id": fid, "road": road, "centre_trace": c,
                "raw": {k: (round(v, 4) if np.isfinite(v) else None) for k, v in pk_raw.items()},
                "bg_removed": {k: (round(v, 4) if np.isfinite(v) else None) for k, v in pk_res.items()},
                "low_band_coherent_share": round(e(low_bg) / e(low_raw), 4),
                "carrier_band_coherent_share": round(e(car_bg) / e(car_raw), 4),
                "low_over_carrier_energy_raw": round(e(low_raw) / e(car_raw), 4),
                "low_over_carrier_energy_bg_removed": round(e(low_res) / e(car_res), 4),
                "low_band_rms_centroid_sample_raw": round(float(
                    (np.arange(321) * (low_raw ** 2).mean(axis=1)).sum() / (low_raw ** 2).mean(axis=1).sum()), 1),
            })
            prof[road]["low_raw"].append(np.sqrt((low_raw ** 2).mean(axis=1)))
            prof[road]["car_raw"].append(np.sqrt((car_raw ** 2).mean(axis=1)))
            prof[road]["low_res"].append(np.sqrt((low_res ** 2).mean(axis=1)))
            prof[road]["car_res"].append(np.sqrt((car_res ** 2).mean(axis=1)))
            spec_acc[road]["raw"].append(s_raw / s_raw.max())
            spec_acc[road]["res"].append(s_res / s_res.max())

    def summ(sel):
        keys = {}
        for stage in ("raw", "bg_removed"):
            for k in ("peak_period", "short_peak_period", "short_peak_rel_amp", "long_peak_period",
                      "long_peak_rel_amp", "short_over_long_amp", "power_share_period_lt20",
                      "centroid_period"):
                keys[f"{stage}.{k}"] = qstats([r[stage][k] for r in sel])
        for k in ("low_band_coherent_share", "carrier_band_coherent_share",
                  "low_over_carrier_energy_raw", "low_over_carrier_energy_bg_removed",
                  "low_band_rms_centroid_sample_raw"):
            keys[k] = qstats([r[k] for r in sel])
        return keys

    by_road = {r: summ([x for x in rows if x["road"] == r]) for r in ROADS}
    by_road_n = {r: sum(1 for x in rows if x["road"] == r) for r in ROADS}
    road_medians = {r: by_road[r]["bg_removed.short_peak_period"]["p50"] for r in ROADS}
    per = 1.0 / f_ref[f_ref >= 1.0 / 128.0]
    keep = (per >= 4) & (per <= 128)
    spectra = {}
    for r in ROADS:
        ok = f_ref >= 1.0 / 128.0
        spectra[r] = {
            "raw_mean_norm": [round(float(v), 4) for v in np.mean(spec_acc[r]["raw"], axis=0)[ok][keep]],
            "bg_removed_mean_norm": [round(float(v), 4) for v in np.mean(spec_acc[r]["res"], axis=0)[ok][keep]],
        }
    profiles = {r: {k: [round(float(v), 2) for v in np.mean(prof[r][k], axis=0)] for k in prof[r]}
                for r in ROADS}
    out = {
        "diagnostic": "D1a", "window_half_traces": HALF, "grid_step_traces": STEP,
        "exclusion": "pipeline mark +-400 traces (label +-300 + margin 100)",
        "spectrum_samples": [S0, S1], "bg_window_traces": BG_WINDOW,
        "low_band_periods": LOW_BAND, "carrier_band_periods": CAR_BAND,
        "n_windows": len(rows), "n_windows_by_road": by_road_n,
        "pooled": summ(rows), "by_road": by_road,
        "road_median_short_peak_bg_removed": road_medians,
        "spectrum_period_axis": [round(float(v), 3) for v in per[keep]],
        "spectra_by_road": spectra, "depth_profiles_by_road": profiles,
        "windows": rows, "elapsed_s": round(time.time() - t_start, 1),
        "note": "stored (MATGPR-processed) amplitude; periods in samples; no random numbers",
    }
    p = write_json("m36_gprsd_period.json", out)
    print(json.dumps({"status": "ok", "n_windows": len(rows), "out": str(p)}))


if __name__ == "__main__":
    main()
