"""AIHub equipment wavelet: envelope length of isolated reflections vs the Ricker and ring gprMax wavelets.

Question (user decision 2026-10-31): is the AIHub radar wavelet closer to the Ricker reflection used in the
v1 band-top grid or to the ringing wavelet of the RL/RK set (envelope >= 0.3 for 3.2 P, >= 0.1 for 4.5 P)?

AIHub: train_pool (folds 0-4) gain-1.0 crops of data/aihub_yolo_v1 (test stays sealed), seed 0,
up to --count crops with pipe_hyperbola boxes and --count crops with pipe_band boxes.
Per crop: DC removal + pixel band-pass (prepare_fitting_branch, no scale) gives the section and P_px.
Hyperbola apex: horizontal background removed (row median of the whole crop), box centre column; band:
no background removal (it would erase the band), box centre column. In the box rows (+-1 P) the
strongest envelope peak is taken; the widths of the contiguous run around it with envelope >= 0.3 and
>= 0.1 of that peak are measured in units of P_px (linear interpolation at the crossings).
Columns whose event region touches 0 or 255 (clipped JPG) are skipped.
Envelope: |hilbert| with zero padding to twice the length.

Synthetic references, same measurement on the pipe-only apex trace (no layer near the pipe):
Ricker RK-free grid model (L_PEC_B30_R03, L_PVC_B30_R03) and ring RK_PEC, RK_PVC; P = 2 ns.

Output: analysis/processing/aihub_wavelet_length.json (aggregates, no file names). stdout last line JSON.
Example: conda run -n gpr-dev python -I analysis/processing/aihub_wavelet_length.py --count 150
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
SYN = ROOT / "data" / "synthetic_gprmax_band_top" / "export"
OUT = ROOT / "analysis" / "processing" / "aihub_wavelet_length.json"
SEED = 0


def envelope(x: np.ndarray) -> np.ndarray:
    n = len(x)
    return np.abs(hilbert(np.concatenate([x, np.zeros(n)])))[:n]


def run_width(env: np.ndarray, peak: int, level: float) -> float | None:
    """Width (samples) of the contiguous run around ``peak`` with env >= level * env[peak]."""
    thr = level * env[peak]
    lo = peak
    while lo > 0 and env[lo - 1] >= thr:
        lo -= 1
    hi = peak
    while hi < len(env) - 1 and env[hi + 1] >= thr:
        hi += 1
    if lo == 0 or hi == len(env) - 1:
        return None  # run touches the record end: not isolated
    left = lo - (env[lo] - thr) / max(env[lo] - env[lo - 1], 1e-12)
    right = hi + (env[hi] - thr) / max(env[hi] - env[hi + 1], 1e-12)
    return float(right - left)


def boxes_of(label: Path, cls: int, w: int, h: int):
    out = []
    for line in label.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if p and int(p[0]) == cls:
            xc, yc, bw, bh = (float(v) for v in p[1:5])
            out.append((max(0, math.floor((xc - bw / 2) * w)), min(w - 1, math.ceil((xc + bw / 2) * w) - 1),
                        max(0, math.floor((yc - bh / 2) * h)), min(h - 1, math.ceil((yc + bh / 2) * h) - 1)))
    return out


def measure_aihub(count: int) -> dict:
    import pandas as pd
    from gpr_detection.io._common import read_grayscale_file
    from gpr_detection.io.radargram import Radargram
    from gpr_detection.preprocess.filters import prepare_fitting_branch

    split = pd.read_csv(SPLIT_CSV, dtype=str, keep_default_na=False)
    base = split[(split["subset"] == "trainpool") & split["fold"].isin({"0", "1", "2", "3", "4"}) & (split["gain"] == "1.0")]
    result = {}
    for cls, name in ((0, "hyperbola_apex"), (1, "band")):
        has = base["classes"].str.split(";").apply(lambda c: str(cls) in c)
        names = sorted(base[has]["file_name"])
        rng = np.random.default_rng(SEED)
        chosen = [names[i] for i in sorted(rng.choice(len(names), size=min(count, len(names)), replace=False))]
        w03, w01, periods, skipped = [], [], [], {"clipped": 0, "edge": 0, "no_period": 0}
        for fname in chosen:
            img = read_grayscale_file(DATASET / "images" / fname)
            h, w = img.shape
            rg = Radargram(data=img.astype(np.float32), dt_ns=None, dx_m=None, air_gap="unknown",
                           amplitude_processed=False, dataset="aihub", line_id=Path(fname).stem)
            br = prepare_fitting_branch(rg)
            if br.period_px is None:
                skipped["no_period"] += 1
                continue
            period = float(br.period_px)
            sec = br.filtered.astype(np.float64)
            if cls == 0:
                sec = sec - np.median(sec, axis=1, keepdims=True)
            for i0, i1, k0, k1 in boxes_of(DATASET / "labels" / f"{Path(fname).stem}.txt", cls, w, h):
                col = (i0 + i1) // 2
                env = envelope(sec[:, col])
                lo, hi = max(0, k0 - int(round(period))), min(h - 1, k1 + int(round(period)))
                peak = lo + int(np.argmax(env[lo: hi + 1]))
                a = run_width(env, peak, 0.3)
                b = run_width(env, peak, 0.1)
                if a is None or b is None:
                    skipped["edge"] += 1
                    continue
                left = max(0, int(peak - b / 2 - 2))
                right = min(h, int(peak + b / 2 + 3))
                raw = img[left:right, col]
                if np.any(raw <= 0) or np.any(raw >= 255):
                    skipped["clipped"] += 1
                    continue
                w03.append(a / period)
                w01.append(b / period)
                periods.append(period)
        q = lambda v: {f"p{p}": round(float(np.percentile(v, p)), 3) for p in (10, 25, 50, 75, 90)} | {"n": len(v)}
        result[name] = {"crops": len(chosen), "w03_P": q(w03) if w03 else None, "w01_P": q(w01) if w01 else None,
                        "P_px": q(periods) if periods else None, "skipped": skipped}
    return result


def spectral_metrics(columns: np.ndarray, dt: float, f_lo: float, f_hi: float) -> dict:
    """Statistical wavelet length (white-reflectivity assumption).

    ``columns`` is ``[samples, n]``; each column is detrended (moving average over 8 / f_lo... removed),
    Hann-tapered and zero-padded. Mean amplitude spectrum -> peak f_p in [f_lo, f_hi] and relative
    half-amplitude bandwidth; mean autocorrelation -> envelope widths at 0.3 / 0.1 in periods 1 / f_p.
    """
    n = columns.shape[0]
    win = np.hanning(n)[:, None]
    x = columns - columns.mean(axis=0, keepdims=True)
    x = x * win
    nfft = 1 << int(math.ceil(math.log2(8 * n)))
    spec = np.abs(np.fft.rfft(x, nfft, axis=0)).mean(axis=1)
    freqs = np.fft.rfftfreq(nfft, dt)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    ip = int(np.flatnonzero(band)[np.argmax(spec[band])])
    fp, half = float(freqs[ip]), 0.5 * spec[ip]
    lo = ip
    while lo > 0 and spec[lo - 1] >= half:
        lo -= 1
    hi = ip
    while hi < len(spec) - 1 and spec[hi + 1] >= half:
        hi += 1
    rel_bw = float((freqs[hi] - freqs[lo]) / fp)
    # autocorrelation of the band-limited traces (power spectrum -> inverse FFT)
    power = (np.abs(np.fft.rfft(x, nfft, axis=0)) ** 2).mean(axis=1)
    ac = np.fft.irfft(power, nfft)[: n]
    ac_full = np.concatenate([ac[:0:-1], ac])
    env = envelope(ac_full)
    peak = int(np.argmax(env))
    period = 1.0 / fp / dt  # samples
    w03 = run_width(env, peak, 0.3)
    w01 = run_width(env, peak, 0.1)
    return {"f_peak": round(fp, 4), "P_samples": round(period, 3), "rel_bw_half_amp": round(rel_bw, 3),
            "ac_w03_P": None if w03 is None else round(w03 / period, 3),
            "ac_w01_P": None if w01 is None else round(w01 / period, 3)}


def spectral_aihub(count: int) -> dict:
    import pandas as pd
    from gpr_detection.io._common import read_grayscale_file

    split = pd.read_csv(SPLIT_CSV, dtype=str, keep_default_na=False)
    base = split[(split["subset"] == "trainpool") & split["fold"].isin({"0", "1", "2", "3", "4"}) & (split["gain"] == "1.0")]
    names = sorted(base["file_name"])
    rng = np.random.default_rng(SEED)
    chosen = [names[i] for i in sorted(rng.choice(len(names), size=min(count, len(names)), replace=False))]
    rows = []
    for fname in chosen:
        img = read_grayscale_file(DATASET / "images" / fname).astype(np.float64)
        # skip the top 20 rows (surface / direct wave, often clipped); 1 px = 1 sample
        cols = img[20:, :]
        cols = cols - np.apply_along_axis(lambda c: np.convolve(c, np.ones(17) / 17, mode="same"), 0, cols)
        rows.append(spectral_metrics(cols, 1.0, 1 / 8.0, 1 / 2.2))
    keys = ("f_peak", "P_samples", "rel_bw_half_amp", "ac_w03_P", "ac_w01_P")

    def stats(sel):
        return {k: {f"p{p}": round(float(np.percentile([r[k] for r in sel if r[k] is not None], p)), 3)
                    for p in (10, 25, 50, 75, 90)} for k in keys}

    # carrier-dominated crops: spectral peak at P 3-6 px (bandpass_px range of AIHub, 2.44-4.83 px observed)
    carrier = [r for r in rows if 3.0 <= r["P_samples"] <= 6.0]
    return {"crops": len(chosen), **stats(rows), "carrier_dominated": {"crops": len(carrier), **stats(carrier)}}


def spectral_synthetic() -> dict:
    out = {}
    for mid in ("L_PEC_B30_R03", "RK_PEC", "L_PVC_B30_R03", "RK_PVC"):
        meta = json.loads((SYN / f"{mid}.json").read_text(encoding="utf-8"))
        dt = float(meta["dt_ns"])
        x = np.load(SYN / f"{mid}_pipeonly.npy")[:, 0].astype(np.float64)[::6]
        width = int(round(8.0 / (dt * 6))) | 1  # same high-pass as AIHub: moving average over ~4 P
        x = x - np.convolve(x, np.ones(width) / width, mode="same")
        out[mid] = {"wavelet": meta.get("wavelet", "ricker"), **spectral_metrics(x[:, None], dt * 6, 0.125, 1.0)}
    return out


def measure_synthetic() -> dict:
    from scipy.signal import butter, sosfiltfilt
    out = {}
    for mid in ("L_PEC_B30_R03", "L_PVC_B30_R03", "RK_PEC", "RK_PVC"):
        meta = json.loads((SYN / f"{mid}.json").read_text(encoding="utf-8"))
        dt, period = float(meta["dt_ns"]), float(meta["period_ns"])
        x = np.load(SYN / f"{mid}_pipeonly.npy")[:, 0].astype(np.float64)
        sos = butter(4, [0.25 * 0.5, 2.0 * 0.5], btype="band", fs=1.0 / dt, output="sos")  # GHz, 0.25-2 f_c
        x = sosfiltfilt(sos, x)
        env = envelope(x)
        peak = int(np.argmax(env))
        out[mid] = {"wavelet": meta.get("wavelet", "ricker"),
                    "w03_P": round(run_width(env, peak, 0.3) * dt / period, 3),
                    "w01_P": round(run_width(env, peak, 0.1) * dt / period, 3)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--count", type=int, default=150)
    a = ap.parse_args()
    syn = measure_synthetic()
    aih = measure_aihub(a.count)
    ref = {"ricker": [np.mean([v[k] for v in syn.values() if v["wavelet"] == "ricker"]) for k in ("w03_P", "w01_P")],
           "ring": [np.mean([v[k] for v in syn.values() if v["wavelet"] == "ring"]) for k in ("w03_P", "w01_P")]}
    verdict = {}
    for name, m in aih.items():
        if not m["w03_P"]:
            continue
        obs = [m["w03_P"]["p50"], m["w01_P"]["p50"]]
        dist = {k: float(np.sqrt(sum(math.log(o / r) ** 2 for o, r in zip(obs, v)))) for k, v in ref.items()}
        verdict[name] = {"median_w03_w01_P": obs, "log_distance": {k: round(d, 3) for k, d in dist.items()},
                         "closer_to": min(dist, key=dist.get)}
    s_syn = spectral_synthetic()
    s_aih = spectral_aihub(a.count)
    s_ref = {w: {k: float(np.mean([v[k] for v in s_syn.values() if v["wavelet"] == w]))
                 for k in ("rel_bw_half_amp", "ac_w03_P", "ac_w01_P")} for w in ("ricker", "ring")}
    obs = {k: s_aih["carrier_dominated"][k]["p50"] for k in ("rel_bw_half_amp", "ac_w03_P", "ac_w01_P")}
    s_dist = {w: round(float(np.sqrt(sum(math.log(obs[k] / r[k]) ** 2 for k in obs))), 3) for w, r in s_ref.items()}
    spectral = {"synthetic": s_syn, "reference_mean": {w: {k: round(x, 3) for k, x in r.items()} for w, r in s_ref.items()},
                "aihub": s_aih, "log_distance": s_dist, "closer_to": min(s_dist, key=s_dist.get)}
    report = {"synthetic_reference": syn, "reference_mean": {k: [round(x, 3) for x in v] for k, v in ref.items()},
              "aihub": aih, "verdict": verdict, "spectral": spectral, "seed": SEED, "count": a.count}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"status": "ok", "verdict": verdict, "reference_mean": report["reference_mean"],
                      "spectral": {"reference_mean": spectral["reference_mean"], "aihub_median": obs,
                                   "log_distance": s_dist, "closer_to": spectral["closer_to"]}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
