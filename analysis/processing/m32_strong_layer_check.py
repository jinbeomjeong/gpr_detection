"""M3-2 S4 review (exploratory, not a verdict): what zone (b') "strong layer" flags on real backgrounds.

Compares, on 8 4TU + 4 Guangzhou IDS lines of the stage A lists (seed 20261114), apex windows of
+-0.3 m (a typical -6 dB half-width) every 0.6 m:
  current  (v0.2.2 정의(탐색 당시)): envelope of the mean (1)-(3) trace over the
           window > 3 x its median over the rows deeper than band + 2 P;
  proposed coherent fraction R_k = env(mean trace)_k^2 / mean_traces env(trace)_k^2, both smoothed over
           1 P; flagged where R_k >= R_MIN (scale- and decay-free; pure noise gives ~1/N).
Reported per rule: share of the injectable depth range (band + 2 P .. record end - 1 P) that is excluded
(+-1 P around flagged rows), depth (P below band + 2 P) of the shallowest flagged row, and the same on a
noise control (the line's (1)-(3) section with each trace circularly shifted by a random lag: keeps the
per-trace spectrum and depth decay, destroys lateral coherence).
This script counts the excluded share as flagged rows +-1 P (an upper bound). The
implemented zone (b') excludes peak rows +-1 P only (``inject.strong_layer_peaks``: the
envelope peak of each flagged run).
Output: analysis/processing/m32_strong_layer_check.json. Run:
conda run -n gpr-dev python -I analysis/processing/m32_strong_layer_check.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.guangzhou import load_guangzhou_radargram  # noqa: E402
from gpr_detection.io.paths import DatasetName, load_dataset_paths  # noqa: E402
from gpr_detection.preprocess.filters import prepare_fitting_branch  # noqa: E402
from gpr_detection.preprocess.timezero import find_time_zero  # noqa: E402
from gpr_detection.preprocess.to_image import make_detection_image  # noqa: E402

SEED = 20261114
HALF_WIDTH_M = 0.3
STEP_M = 0.6
FACTOR = 3.0
R_MIN = 0.5
OUT = ROOT / "analysis" / "processing" / "m32_strong_layer_check.json"
HERE = ROOT / "analysis" / "processing"


def env(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.abs(hilbert(x, axis=axis))


def current_rule(filtered: np.ndarray, start: int, t0: int, t1: int) -> np.ndarray:
    region = env(filtered[:, t0 : t1 + 1].mean(axis=1))[start:]
    return start + np.flatnonzero(region > FACTOR * float(np.median(region)))


def coherent_rule(filtered: np.ndarray, start: int, t0: int, t1: int, period: float) -> np.ndarray:
    block = filtered[:, t0 : t1 + 1]
    win = max(1, int(round(period)))
    num = uniform_filter1d(env(block.mean(axis=1)) ** 2, win)
    den = uniform_filter1d((env(block) ** 2).mean(axis=1), win)
    r = num / np.maximum(den, 1e-30)
    return start + np.flatnonzero(r[start:] >= R_MIN), r


def local_rule(filtered: np.ndarray, start: int, t0: int, t1: int, period: float, factor: float) -> np.ndarray:
    """Depth-localised layer: mean-trace envelope above factor x its moving median over +-3 P."""
    from scipy.ndimage import median_filter

    e = env(filtered[:, t0 : t1 + 1].mean(axis=1))
    base = median_filter(e, size=max(3, int(round(6 * period)) | 1), mode="nearest")
    return start + np.flatnonzero(e[start:] > factor * np.maximum(base[start:], 1e-30))


def excluded_share(flagged: np.ndarray, start: int, end: int, period: float) -> float:
    if end <= start:
        return float("nan")
    mask = np.zeros(end - start, dtype=bool)
    for k in flagged:
        lo, hi = int(k - period) - start, int(k + period) - start + 1
        mask[max(lo, 0) : max(min(hi, end - start), 0)] = True
    return float(mask.mean())


def analyse(rg, rng) -> dict[str, object]:
    branch = prepare_fitting_branch(rg)
    filtered = np.asarray(branch.filtered, dtype=np.float64)
    zero = find_time_zero(rg, filtered)
    period = float(make_detection_image(rg).period_samples)
    if zero.k_zero is None:
        return {"line": rg.line_id, "skipped": "no zero"}
    n_s, n_tr = filtered.shape
    start = math.ceil(zero.k_zero + 2.0 * period)
    end = int(n_s - period)
    half = max(1, int(round(HALF_WIDTH_M / rg.dx_m)))
    step = max(1, int(round(STEP_M / rg.dx_m)))
    shifted = np.stack([np.roll(filtered[:, j], int(rng.integers(n_s))) for j in range(n_tr)], axis=1)
    out = {"line": rg.line_id, "period": period, "region_P": (end - start) / period, "windows": 0,
           "current": [], "coherent": [], "current_noise": [], "coherent_noise": [],
           "current_shallowest_P": [], "coherent_r_max": []}
    for c in range(half, n_tr - half, step):
        t0, t1 = c - half, c + half
        f_cur = current_rule(filtered, start, t0, t1)
        f_coh, r = coherent_rule(filtered, start, t0, t1, period)
        out["current"].append(excluded_share(f_cur, start, end, period))
        out["coherent"].append(excluded_share(f_coh, start, end, period))
        out["current_noise"].append(excluded_share(current_rule(shifted, start, t0, t1), start, end, period))
        out["coherent_noise"].append(excluded_share(coherent_rule(shifted, start, t0, t1, period)[0],
                                                    start, end, period))
        for fac in (3.0, 4.0):
            f_loc = local_rule(filtered, start, t0, t1, period, fac)
            out.setdefault(f"local{fac:g}", []).append(excluded_share(f_loc, start, end, period))
            out.setdefault(f"local{fac:g}_noise", []).append(
                excluded_share(local_rule(shifted, start, t0, t1, period, fac), start, end, period))
        if f_cur.size:
            out["current_shallowest_P"].append(float((f_cur.min() - start) / period))
        out["coherent_r_max"].append(float(r[start:end].max()) if end > start else float("nan"))
        out["windows"] += 1
    for key in ("current", "coherent", "current_noise", "coherent_noise",
                "local3", "local3_noise", "local4", "local4_noise"):
        out[key] = float(np.nanmean(out[key])) if out[key] else None
    out["current_shallowest_P"] = (float(np.median(out["current_shallowest_P"]))
                                   if out["current_shallowest_P"] else None)
    out["coherent_r_max"] = float(np.nanmedian(out["coherent_r_max"])) if out["coherent_r_max"] else None
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    paths = load_dataset_paths()
    l4 = [r["line"] for r in csv.DictReader(open(HERE / "m32_stageA_lines_4tu.csv", encoding="utf-8"))]
    li = [r["line"] for r in csv.DictReader(open(HERE / "m32_stageA_lines_ids.csv", encoding="utf-8"))]
    pick4 = [l4[i] for i in sorted(rng.choice(len(l4), 8, replace=False))]
    picki = [li[i] for i in sorted(rng.choice(len(li), 4, replace=False))]
    results = []
    for fid in pick4:
        act, path = fid.split("/")
        results.append({"dataset": "4tu"} | analyse(load_fourtu_radargram(paths.root(DatasetName.FOURTU), act, path), rng))
        print(f"4tu {fid} done", flush=True)
    for fid in picki:
        results.append({"dataset": "gz_ids"} | analyse(load_guangzhou_radargram(paths.root(DatasetName.GUANGZHOU), fid), rng))
        print(f"ids {fid} done", flush=True)
    OUT.write_text(json.dumps({"seed": SEED, "half_width_m": HALF_WIDTH_M, "factor": FACTOR, "r_min": R_MIN,
                               "lines": results}, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "n": len(results)}))


if __name__ == "__main__":
    main()
