"""M3-2 R1 diagnosis (S2, report only): Guangzhou MALA period rules and the folder 07011 spectrum.

Plan: docs/processing/m32_image_quality_plan.md v0.2.1 section 2-4 (diagnosis 1 and 2). Defaults are not changed.
Diagnosis 1 compares, for all 75 MALA lines (stored processed amplitudes, DC removed):
  R-a  current rule: DetectionImage period (make_detection_image, P* 6) and compression c;
  EM   rule of docs/processing/m32_em_review.md section 3-3:
       R-b early window  -> samples before the last time the trace-mean envelope exceeds 3x its late-third median
                            (at least n_s / 3 samples);
       R-c folder median -> f_med = median over the folder of the R-b frequencies;
       band limit        -> peak of the early-window spectrum searched in [0.6, 1.5] f_med;
       +-30 %            -> if the band-limited period differs from 1 / f_med by more than 30 %, use 1 / f_med (flag).
  For each rule: period (samples, MHz), H_c = round(n_s P* / P), compression count and c > 1.35 count.
Diagnosis 2: per folder, the folder-mean normalised amplitude spectrum; peak amplitude in 180-250 MHz vs 380-470 MHz
  (A_213 / A_427), and whether the 180-250 MHz band holds a local maximum. 07011 is the folder in question.
Spectra: trace-mean |rfft| with 8x zero padding (same estimator as measure_period_samples).
Output: analysis/processing/m32_mala_period_diag.json. Read-only, no detector, no depth.
Run: conda run -n gpr-dev python -I analysis/processing/m32_mala_period_diag.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpr_detection.io.guangzhou import load_guangzhou_radargram  # noqa: E402
from gpr_detection.io.paths import DatasetName, load_dataset_paths  # noqa: E402
from gpr_detection.preprocess.to_image import make_detection_image  # noqa: E402

P_STAR = 6.0
TILE = 256
COMPRESS_MAX = 1.35
OUT = ROOT / "analysis" / "processing" / "m32_mala_period_diag.json"


def spectrum(a: np.ndarray, pad: int = 8) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    n = a.shape[0]
    nfft = 1 << int(np.ceil(np.log2(pad * n)))
    spec = np.abs(np.fft.rfft(a, nfft, axis=0)).mean(axis=1)
    return np.fft.rfftfreq(nfft, d=1.0), spec


def peak_period(f: np.ndarray, spec: np.ndarray, lo: float, hi: float) -> float | None:
    ok = (f >= lo) & (f <= hi)
    if not ok.any():
        return None
    i = int(np.argmax(np.where(ok, spec, -np.inf)))
    if i in (int(np.flatnonzero(ok)[0]), int(np.flatnonzero(ok)[-1])):
        return None
    return 1.0 / float(f[i])


def early_end(a: np.ndarray) -> int:
    env = np.abs(hilbert(a - a.mean(axis=0, keepdims=True), axis=0)).mean(axis=1)
    n = len(env)
    floor = float(np.median(env[-(n // 3):]))
    above = np.flatnonzero(env > 3.0 * floor)
    end = int(above[-1]) + 1 if above.size else n
    return max(end, n // 3)


def h_c(n_s: int, p: float) -> int:
    return int(round(n_s * P_STAR / p))


def comp(n_s: int, p: float) -> float:
    return max(1.0, n_s * P_STAR / (TILE * p)) if h_c(n_s, p) > TILE else 1.0


def main() -> None:
    root = load_dataset_paths().root(DatasetName.GUANGZHOU)
    index = [r for r in csv.DictReader(open(root / "01_메타데이터" / "radargrams_index.csv", encoding="utf-8-sig"))
             if r["format"] == "MALA .rd3"]
    lines = []
    folder_spec: dict[str, list] = defaultdict(list)
    for r in index:
        fid = r["file_id"]
        folder = fid.split("/")[1]
        rg = load_guangzhou_radargram(root, fid)
        a = np.asarray(rg.data, dtype=np.float64)
        n_s, dt = a.shape[0], float(rg.dt_ns)
        img = make_detection_image(rg)
        end = early_end(a)
        f, spec_e = spectrum(a[:end])
        p_b = peak_period(f, spec_e, 1.0 / (end / 4.0), 1.0 / 4.0)
        f_all, spec_all = spectrum(a)
        mhz = f_all / dt * 1000.0
        folder_spec[folder].append((mhz, spec_all / spec_all.max()))
        lines.append({"line": fid, "folder": folder, "n_s": n_s, "dt_ns": dt, "early_end": end,
                      "P_a": img.period_samples, "c_a": img.compression, "P_b": p_b,
                      "_f": f, "_spec": spec_e})
    # folder median and band-limited, +-30 % rule
    by_folder: dict[str, list] = defaultdict(list)
    for ln in lines:
        by_folder[ln["folder"]].append(ln)
    folders = {}
    for folder, group in sorted(by_folder.items()):
        fb = [1.0 / g["P_b"] for g in group if g["P_b"]]
        f_med = float(np.median(fb))
        for g in group:
            p_bl = peak_period(g["_f"], g["_spec"], 0.6 * f_med, 1.5 * f_med)
            p_med = 1.0 / f_med
            if p_bl is None or abs(p_bl / p_med - 1.0) > 0.30:
                g["P_em"], g["em_replaced"] = p_med, True
            else:
                g["P_em"], g["em_replaced"] = p_bl, False
        # diagnosis 2
        mhz = folder_spec[folder][0][0]
        mean_spec = np.mean([s for m, s in folder_spec[folder] if len(m) == len(mhz)], axis=0)
        band1 = (mhz >= 180) & (mhz <= 250)
        band2 = (mhz >= 380) & (mhz <= 470)
        i1 = int(np.argmax(np.where(band1, mean_spec, -np.inf)))
        i2 = int(np.argmax(np.where(band2, mean_spec, -np.inf)))
        local_max_1 = bool(0 < i1 < len(mean_spec) - 1 and band1[i1 - 1] and band1[i1 + 1]
                           and mean_spec[i1] >= mean_spec[i1 - 1] and mean_spec[i1] >= mean_spec[i1 + 1])
        i_all = int(np.argmax(np.where(mhz > 20, mean_spec, -np.inf)))
        folders[folder] = {
            "n_lines": len(group), "f_med_MHz_Rb": f_med / group[0]["dt_ns"] * 1000.0,
            "dominant_MHz_folder_mean": float(mhz[i_all]),
            "A_180_250": float(mean_spec[i1]), "f_180_250_MHz": float(mhz[i1]),
            "A_380_470": float(mean_spec[i2]), "f_380_470_MHz": float(mhz[i2]),
            "ratio_A213_over_A427": float(mean_spec[i1] / mean_spec[i2]),
            "local_max_in_180_250": local_max_1,
            "n_replaced_pm30": sum(g["em_replaced"] for g in group),
        }
    summary = {}
    for rule, key in (("R-a", "P_a"), ("EM", "P_em")):
        ps = [ln[key] for ln in lines]
        cs = [ln["c_a"] if rule == "R-a" else comp(ln["n_s"], ln[key]) for ln in lines]
        mhz = [1000.0 / (p * ln["dt_ns"]) for p, ln in zip(ps, lines)]
        summary[rule] = {
            "period_samples_min_med_max": [float(np.min(ps)), float(np.median(ps)), float(np.max(ps))],
            "MHz_min_med_max": [float(np.min(mhz)), float(np.median(mhz)), float(np.max(mhz))],
            "n_compressed": int(sum(c > 1.0 for c in cs)),
            "n_scale_out_of_range": int(sum(c > COMPRESS_MAX for c in cs)),
            "c_min_max_of_compressed": [float(min([c for c in cs if c > 1.0], default=1.0)),
                                        float(max(cs))],
            "H_c_min_med_max": [int(min(h_c(ln["n_s"], p) for p, ln in zip(ps, lines))),
                                int(np.median([h_c(ln["n_s"], p) for p, ln in zip(ps, lines)])),
                                int(max(h_c(ln["n_s"], p) for p, ln in zip(ps, lines)))],
        }
    per_line = [{k: v for k, v in ln.items() if not k.startswith("_")} for ln in lines]
    result = {"n_lines": len(lines), "rules": summary, "folders": folders, "lines": per_line,
              "note": "report only (PQ3 (a)); defaults unchanged"}
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "n_lines": len(lines)}))


if __name__ == "__main__":
    main()
