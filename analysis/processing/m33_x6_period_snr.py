"""M3-3 X6 (report only, no detector, no ground truth): Morocco L1 period and 4TU known-apex SNR proxy.

Part M  Morocco eval set (data/morocco_eval_v1, 233 images; manifest `subset` column only, labels not read).
        Period per image with the estimator that produced AIHub P_A 4.06 px (analysis/processing/
        aihub_wavelet_length.py: whole crop, scale-less Radargram, prepare_fitting_branch -> period_px, i.e.
        DC removal + filters.estimate_period_px). Reported in native px (224 x 224) and in model-input px
        (letterbox to 256: x 256 / max(H, W)), against P_A 4.06 x [0.55, 1.5] = [2.23, 6.09] px.
Part S  4TU known apexes (data/processing/d3/known_apexes.csv, columns blind_id, line, trace, sample only).
        SNR proxy, same quantities as the M3-2 injection SNR but from the real section:
          A     = envelope peak of the fitting-branch (1)-(4) section on the apex trace, apex sample +- 0.5 P;
          sigma = robust RMS (1.4826 x MAD) of the same (1)-(4) section over rows apex +- 1 P and traces whose
                  lateral distance from the apex is 1.0-3.0 m (both sides; whole line outside +- 1.0 m when
                  fewer than 10 such traces);
          SNR   = 20 log10(A / sigma).  P = make_detection_image(r0).period_samples.
Output: analysis/processing/m33_x6_period_snr.json. Run:
conda run -n gpr-dev python -I analysis/processing/m33_x6_period_snr.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.paths import DatasetName, load_dataset_paths  # noqa: E402
from gpr_detection.io.radargram import Radargram  # noqa: E402
from gpr_detection.preprocess.filters import prepare_fitting_branch  # noqa: E402
from gpr_detection.preprocess.to_image import make_detection_image  # noqa: E402

MOROCCO = ROOT / "data" / "morocco_eval_v1"
MANIFEST = ROOT / "data" / "splits" / "morocco_eval_v1.csv"
APEXES = ROOT / "data" / "processing" / "d3" / "known_apexes.csv"
OUT = ROOT / "analysis" / "processing" / "m33_x6_period_snr.json"
P_A = 4.06
SCALE_RANGE = (0.55, 1.5)
EXCL_M, REF_M = 1.0, 3.0


def pct(values, qs=(5, 25, 50, 75, 95)):
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return None
    return {f"p{q}": round(float(np.percentile(arr, q)), 3) for q in qs} | {"n": int(arr.size)}


def morocco() -> dict:
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
    lo, hi = P_A * SCALE_RANGE[0], P_A * SCALE_RANGE[1]
    by_subset: dict[str, list] = {}
    per_image = []
    for r in rows:
        img = np.asarray(Image.open(MOROCCO / "images" / r["file_name"]).convert("L"), dtype=np.float32)
        rg = Radargram(data=img, dt_ns=None, dx_m=None, air_gap="unknown",
                       amplitude_processed=False, dataset="morocco", line_id=r["image_id"])
        period = prepare_fitting_branch(rg).period_px
        scale = 256.0 / max(img.shape)
        p_in = None if period is None else float(period) * scale
        per_image.append({"image_id": r["image_id"], "subset": r["subset"],
                          "anisotropic_stretch": r["anisotropic_stretch"], "P_native_px": period,
                          "P_input_px": p_in, "outside": None if p_in is None else not (lo <= p_in <= hi)})
        by_subset.setdefault(r["subset"], []).append(per_image[-1])
    out = {"P_A": P_A, "input_range_px": [round(lo, 3), round(hi, 3)], "subsets": {}}
    for name, items in {**by_subset, "all": per_image}.items():
        p_in = [x["P_input_px"] for x in items]
        out["subsets"][name] = {
            "n": len(items), "n_no_period": sum(v is None for v in p_in),
            "P_native_px": pct([x["P_native_px"] for x in items]),
            "P_input_px": pct(p_in),
            "ratio_to_P_A": pct([None if v is None else v / P_A for v in p_in]),
            "n_outside_range": sum(bool(x["outside"]) for x in items),
        }
    return out


def env(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    return np.abs(hilbert(np.concatenate([x, np.zeros_like(x)])))[:n]


def snr_proxy() -> dict:
    root = load_dataset_paths().root(DatasetName.FOURTU)
    apexes = [{k: r[k] for k in ("blind_id", "line", "trace", "sample")}
              for r in csv.DictReader(open(APEXES, encoding="utf-8"))]
    cache: dict[str, tuple] = {}
    per_apex = []
    for a in apexes:
        if a["line"] not in cache:
            act, path = a["line"].split("/")
            rg = load_fourtu_radargram(root, act, path)
            sec = prepare_fitting_branch(rg).background_removed
            cache = {a["line"]: (rg, sec, float(make_detection_image(rg).period_samples))}
        rg, sec, p = cache[a["line"]]
        tr, k = int(round(float(a["trace"]))), float(a["sample"])
        n_s, n_tr = sec.shape
        e = env(np.asarray(sec[:, tr], dtype=np.float64))
        a0, a1 = max(0, int(math.floor(k - 0.5 * p))), min(n_s, int(math.ceil(k + 0.5 * p)) + 1)
        amp = float(e[a0:a1].max())
        r0, r1 = max(0, int(math.floor(k - p))), min(n_s, int(math.ceil(k + p)) + 1)
        dist = np.abs(np.arange(n_tr) - tr) * rg.dx_m
        ref = (dist >= EXCL_M) & (dist <= REF_M)
        fallback = int(ref.sum()) < 10
        if fallback:
            ref = dist >= EXCL_M
        vals = np.asarray(sec[r0:r1][:, ref], dtype=np.float64).ravel()
        sigma = float(1.4826 * np.median(np.abs(vals - np.median(vals)))) if vals.size else float("nan")
        snr = 20 * math.log10(amp / sigma) if sigma > 0 and amp > 0 else None
        per_apex.append({"blind_id": a["blind_id"], "line": a["line"], "P_line": round(p, 3),
                         "snr_db": None if snr is None else round(snr, 2),
                         "ref_traces": int(ref.sum()), "ref_fallback": fallback})
        print(f"{a['blind_id']} snr={per_apex[-1]['snr_db']}", flush=True)
    snrs = [x["snr_db"] for x in per_apex]
    q = np.percentile([s for s in snrs if s is not None], [25, 50, 75])
    return {"definition": "A = env peak of (1)-(4) on apex trace within apex +-0.5 P; sigma = 1.4826 MAD of "
                          "(1)-(4) over rows apex +-1 P, traces 1.0-3.0 m from apex",
            "quartile_edges_db": [round(float(x), 2) for x in q], "summary": pct(snrs),
            "n_fallback": sum(x["ref_fallback"] for x in per_apex),
            "share_by_ladder": {
                "below_3dB": sum(1 for s in snrs if s is not None and s < 3),
                "3_9dB": sum(1 for s in snrs if s is not None and 3 <= s < 9),
                "9_15dB": sum(1 for s in snrs if s is not None and 9 <= s < 15),
                "15dB_up": sum(1 for s in snrs if s is not None and s >= 15)},
            "apexes": per_apex}


def main() -> None:
    result = {"morocco": morocco(), "fourtu_k_snr": snr_proxy()}
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(OUT)}))


if __name__ == "__main__":
    main()
