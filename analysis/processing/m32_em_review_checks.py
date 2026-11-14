"""M3-2 EM review checks (read-only evidence for docs/processing/m32_em_review.md).

Part S: existing gprMax pipe-only responses (full - _bg) - dominant period, apex
envelope width, aperture-edge amplitude, mirror symmetry, flank spreading exponent,
latest significant energy vs time window.
Part G: Guangzhou MALA lines - dominant period over the whole trace and early / late
thirds, and the share of spectral amplitude below 150 MHz.
Part T: 4TU raw saturation (|D| >= 32767 after an int32 cast) per line and the Guangzhou IDS
400 MHz index `saturated_ratio`.

The period estimator reproduces gpr_detection.preprocess.to_image.measure_period_samples
(trace-averaged |rfft| of DC-removed traces, zero padded x8, peak in [n_s/4, 4] samples).
Nothing is written except the JSON given by --out. Input data are only read.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYNTH_DIRS = [
    "synthetic_gprmax",
    "synthetic_gprmax_k7",
    "synthetic_gprmax_s1g",
    "synthetic_gprmax_s4s6",
    "synthetic_gprmax_s4s6_v2",
    "synthetic_gprmax_s4s6_v3",
]
GZ_ROOT = r"C:\workspace\GPR_Data\광저우대_중국_관로GPR_가공본_261016\05_radargrams_npy\pipe"


def measure_period_samples(section, pad_factor=8, min_period=4.0):
    values = np.asarray(section, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    n = values.shape[0]
    values = values - values.mean(axis=0, keepdims=True)
    n_fft = 1 << math.ceil(math.log2(pad_factor * n))
    spec = np.abs(np.fft.rfft(values, n_fft, axis=0)).mean(axis=1)
    freq = np.fft.rfftfreq(n_fft, d=1.0)
    band = np.flatnonzero((freq >= 1.0 / (n / 4.0)) & (freq <= 1.0 / min_period))
    if len(band) == 0 or not spec[band].max() > 0:
        return None
    k = int(np.argmax(spec[band]))
    if k in (0, len(band) - 1):
        return None
    return 1.0 / float(freq[band[k]])


def envelope(trace):
    """Hilbert envelope with 2N zero padding (circular leakage guard)."""
    n = len(trace)
    m = 1 << math.ceil(math.log2(2 * n))
    spec = np.fft.fft(trace, m)
    h = np.zeros(m)
    h[0] = 1.0
    h[1 : m // 2] = 2.0
    h[m // 2] = 1.0
    return np.abs(np.fft.ifft(spec * h))[:n]


def fwhm(env, dt):
    k = int(np.argmax(env))
    half = env[k] / 2.0
    lo = k
    while lo > 0 and env[lo] > half:
        lo -= 1
    hi = k
    while hi < len(env) - 1 and env[hi] > half:
        hi += 1
    return (hi - lo) * dt


def synth_part():
    rows = []
    for sub in SYNTH_DIRS:
        for jpath in sorted(glob.glob(os.path.join(ROOT, "data", sub, "*.json"))):
            base = jpath[:-5]
            mid = os.path.basename(base)
            if mid.endswith(("_bg", "_fs")):
                continue
            if not (os.path.exists(base + ".npy") and os.path.exists(base + "_bg.npy")):
                continue
            meta = json.load(open(jpath, encoding="utf-8"))
            if meta.get("pipe_material") in (None, "") or meta.get("n_pipes", 1) not in (None, 1):
                continue
            full = np.load(base + ".npy").astype(np.float64)
            bg = np.load(base + "_bg.npy").astype(np.float64)
            if bg.shape[1] != full.shape[1]:
                bg = np.repeat(bg[:, :1], full.shape[1], axis=1)
            d = full - bg
            dt = float(meta["dt_ns"])
            dx = float(meta["dx_m"])
            x = float(meta["trace_x_m_first"]) + dx * np.arange(d.shape[1])
            xc = meta["pipe_center_x_m"]
            xc = float(xc[0] if isinstance(xc, list) else xc)
            ia = int(np.argmin(np.abs(x - xc)))
            amax = float(np.abs(d[:, ia]).max())
            p_pipe = measure_period_samples(d)
            p_bg = measure_period_samples(bg[:, :1])
            env_a = envelope(d[:, ia])
            # mirror symmetry about the apex trace
            worst = 0.0
            for i in range(d.shape[1]):
                j = 2 * ia - i
                if 0 <= j < d.shape[1] and j != i:
                    worst = max(worst, float(np.abs(d[:, i] - d[:, j]).max()) / amax)
            # flank spreading exponent: envelope peak vs time after source peak
            # source (Ricker) peak at sqrt(2)/f, see the "wavelet" metadata field
            tz = math.sqrt(2) / (float(meta["center_freq_hz"]) * 1e-9)
            amps, times = [], []
            for i in range(d.shape[1]):
                if abs(x[i] - xc) > 1.0:
                    continue
                e = envelope(d[:, i])
                k = int(np.argmax(e))
                amps.append(e[k])
                times.append(k * dt - tz)
            amps = np.asarray(amps)
            times = np.asarray(times)
            ok = times > 0
            slope = float(np.polyfit(np.log(times[ok]), np.log(amps[ok]), 1)[0]) if ok.sum() > 5 else None
            env_all = np.array([envelope(d[:, i]) for i in (0, ia, d.shape[1] - 1)])
            gmax = float(np.abs(d).max())
            last_sig = None
            col_env = np.abs(d).max(axis=1)
            idx = np.flatnonzero(col_env > 1e-2 * gmax)
            if len(idx):
                last_sig = float(idx[-1] * dt)
            rows.append(
                {
                    "set": sub,
                    "model": mid,
                    "material": meta.get("pipe_material"),
                    "od_m": 2 * float(meta["pipe_radius_m"]),
                    "top_m": float(meta["pipe_top_depth_m"]),
                    "eps_r": float(meta["eps_r"]),
                    "sigma": float(meta["soil_sigma_S_per_m"]),
                    "f_mhz": float(meta["center_freq_hz"]) / 1e6,
                    "air_gap_m": float(meta["air_gap_m"]),
                    "n_traces": int(d.shape[1]),
                    "aperture_m": float(x[-1] - x[0]),
                    "window_ns": float(meta["time_window_ns"]),
                    "dt_ns": dt,
                    "period_pipe_ns": None if p_pipe is None else p_pipe * dt,
                    "period_bg_ns": None if p_bg is None else p_bg * dt,
                    "apex_env_fwhm_ns": fwhm(env_a, dt),
                    "edge_over_apex_peak": float(max(env_all[0].max(), env_all[2].max()) / env_all[1].max()),
                    "mirror_maxdiff_rel": worst,
                    "flank_spreading_exponent": slope,
                    "last_1pct_energy_ns": last_sig,
                }
            )
    return rows


def mala_part():
    rows = []
    for npy in sorted(glob.glob(os.path.join(GZ_ROOT, "MALA_*", "*.npy"))):
        meta = json.load(open(npy[:-4] + ".json", encoding="utf-8"))
        a = np.load(npy).astype(np.float64)
        dt = float(meta["dt_ns"])
        n = a.shape[0]
        third = n // 3
        p_all = measure_period_samples(a)
        p_early = measure_period_samples(a[:third])
        p_late = measure_period_samples(a[-third:])
        v = a - a.mean(axis=0, keepdims=True)
        n_fft = 1 << math.ceil(math.log2(8 * n))
        spec = np.abs(np.fft.rfft(v, n_fft, axis=0)).mean(axis=1)
        f_mhz = np.fft.rfftfreq(n_fft, d=dt) * 1e3
        share_low = float(spec[f_mhz < 150].sum() / spec.sum())
        rms_early = float(np.sqrt((v[:third] ** 2).mean()))
        rms_late = float(np.sqrt((v[-third:] ** 2).mean()))
        rows.append(
            {
                "file_id": meta["file_id"],
                "folder": meta["mission"],
                "n_samples": n,
                "n_traces": int(a.shape[1]),
                "dt_ns": dt,
                "window_ns": meta["time_window_ns"],
                "dx_m": meta["trace_dx_m"],
                "spectrum_peak_MHz_index": meta.get("spectrum_peak_MHz"),
                "period_all_samples": p_all,
                "period_early_samples": p_early,
                "period_late_samples": p_late,
                "f_all_MHz": None if p_all is None else 1e3 / (p_all * dt),
                "f_early_MHz": None if p_early is None else 1e3 / (p_early * dt),
                "f_late_MHz": None if p_late is None else 1e3 / (p_late * dt),
                "spec_share_below_150MHz": share_low,
                "rms_late_over_early": rms_late / rms_early if rms_early > 0 else None,
            }
        )
    return rows


T4_ROOT = r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015\05_radargrams_npy"
GZ_INDEX = r"C:\workspace\GPR_Data\광저우대_중국_관로GPR_가공본_261016\01_메타데이터\radargrams_index.csv"


def saturation_part():
    """4TU raw int16 saturation (|D| >= 32767 after int32 cast) and Guangzhou IDS 400 MHz index ratio.

    Per-line ratio = saturated samples / all samples of that line. Quantiles are reported over two
    denominators: all lines, and only lines with at least one saturated sample.
    """
    ratios, last_rows = [], []
    for npy in sorted(glob.glob(os.path.join(T4_ROOT, "*", "*.npy"))):
        sat = np.abs(np.load(npy).astype(np.int32)) >= 32767
        ratios.append(float(sat.mean()))
        rows_sat = np.flatnonzero(sat.any(axis=1))
        if len(rows_sat):
            last_rows.append(int(rows_sat[-1]))
    ratios = np.asarray(ratios)
    pos = ratios[ratios > 0]
    last_rows = np.asarray(last_rows)

    def pct(v):
        return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    with open(GZ_INDEX, encoding="utf-8-sig") as fh:
        idx = list(csv.DictReader(fh))
    ids = np.array(
        [
            float(r["saturated_ratio"])
            for r in idx
            if r["target"] == "pipe"
            and r["format"].startswith("IDS")
            and r["antenna_MHz"]
            and float(r["antenna_MHz"]) == 400
            and r["saturated_ratio"]
        ]
    )
    return {
        "fourtu": {
            "n_lines": int(len(ratios)),
            "n_lines_saturated": int(len(pos)),
            "line_ratio_all_lines": pct(ratios),
            "line_ratio_saturated_lines_only": pct(pos),
            "last_saturated_sample_index_saturated_lines": pct(last_rows),
            "n_lines_saturated_after_sample_100": int((last_rows > 100).sum()),
        },
        "guangzhou_ids_400MHz_index": {
            "n_lines": int(len(ids)),
            "n_lines_saturated": int((ids > 0).sum()),
            "line_ratio_all_lines": pct(ids),
            "line_ratio_saturated_lines_only": pct(ids[ids > 0]),
        },
    }


def summarise_mala(rows):
    out = {}
    for folder in sorted({r["folder"] for r in rows}):
        rs = [r for r in rows if r["folder"] == folder]

        def q(key):
            v = np.array([r[key] for r in rs if r[key] is not None], dtype=float)
            if len(v) == 0:
                return None
            return [float(np.min(v)), float(np.median(v)), float(np.max(v)), int(len(v))]

        out[folder] = {
            "n_files": len(rs),
            "window_ns": sorted({r["window_ns"] for r in rs}),
            "n_samples": sorted({r["n_samples"] for r in rs}),
            "period_all_samples_min_med_max_n": q("period_all_samples"),
            "f_all_MHz": q("f_all_MHz"),
            "f_early_MHz": q("f_early_MHz"),
            "f_late_MHz": q("f_late_MHz"),
            "spec_share_below_150MHz": q("spec_share_below_150MHz"),
            "rms_late_over_early": q("rms_late_over_early"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    synth = synth_part()
    mala = mala_part()
    sat = saturation_part()
    result = {"synthetic": synth, "mala_files": mala, "mala_by_folder": summarise_mala(mala), "saturation": sat}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"n_synth": len(synth), "n_mala": len(mala), "out": args.out}))


if __name__ == "__main__":
    main()
