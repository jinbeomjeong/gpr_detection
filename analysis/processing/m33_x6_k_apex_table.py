"""M3-3 X6 table for the R2 runner (report only, no detector, no ground truth): K apex SNR quartile and
L1 -6 dB width.

Input: analysis/processing/m33_x6_period_snr.json (SNR proxy per K apex, same run) and the K apex positions
(data/processing/d3/known_apexes.csv: blind_id, line, trace, sample only).
Width (L1 periods, -6 dB, same level as the M3-2 in-distribution width l_W):
  on the fitting-branch (1)-(4) section, start at the apex (envelope peak within apex +- 0.5 P on the apex trace);
  step one trace at a time to each side and take the envelope maximum in rows [prev - 0.3 P, prev + 1.0 P]
  (the hyperbola only moves later away from the apex); stop before the first trace whose maximum is below
  0.5 x the apex envelope (-6 dB), or at the line edge (flag edge_left / edge_right, width is then a lower bound).
  width_traces = left + right + 1; width_P = width_traces / traces_per_px / (P_line / samples_per_px) with the r0
  DetectionImage grid map (L1 px per trace and per sample) -> comparable to AIHub 7.79-14.67 P.
Also reported (same tracker, other stop levels): width_m20_P (0.1, -20 dB) and width_vis_P ("visible" level
max(10^(-SNR/20), 0.1), the M3-2 IoU-report box level of [EMR] B-1(b): where the wing sinks into the noise).
Quartile edges come from the SNR json (p25 / p50 / p75 of the 41 apexes): q1 < p25 <= q2 < p50 <= q3 < p75 <= q4.
Output: analysis/processing/m33_k_apex_snr.csv. Run:
conda run -n gpr-dev python -I analysis/processing/m33_x6_k_apex_table.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.paths import DatasetName, load_dataset_paths  # noqa: E402
from gpr_detection.preprocess.filters import prepare_fitting_branch  # noqa: E402
from gpr_detection.preprocess.to_image import make_detection_image  # noqa: E402

SNR_JSON = ROOT / "analysis" / "processing" / "m33_x6_period_snr.json"
APEXES = ROOT / "data" / "processing" / "d3" / "known_apexes.csv"
OUT = ROOT / "analysis" / "processing" / "m33_k_apex_snr.csv"
LEVEL = 0.5


def track(env: np.ndarray, tr0: int, k0: int, p: float, step: int, amp: float,
          level: float = LEVEL) -> tuple[int, bool]:
    n_s, n_tr = env.shape
    k, count, tr = k0, 0, tr0
    while True:
        tr += step
        if tr < 0 or tr >= n_tr:
            return count, True
        lo, hi = max(0, int(math.floor(k - 0.3 * p))), min(n_s, int(math.ceil(k + 1.0 * p)) + 1)
        seg = env[lo:hi, tr]
        j = int(np.argmax(seg))
        if seg[j] < level * amp:
            return count, False
        k = lo + j
        count += 1


def gm_scale(img) -> float:
    return float(img.grid_map.traces_per_px)


def main() -> None:
    snr = json.loads(SNR_JSON.read_text(encoding="utf-8"))["fourtu_k_snr"]
    q25, q50, q75 = snr["quartile_edges_db"]
    snr_by_id = {a["blind_id"]: a for a in snr["apexes"]}
    root = load_dataset_paths().root(DatasetName.FOURTU)
    apexes = [{k: r[k] for k in ("blind_id", "line", "trace", "sample")}
              for r in csv.DictReader(open(APEXES, encoding="utf-8"))]
    rows, cache = [], {}
    for a in apexes:
        if a["line"] not in cache:
            act, path = a["line"].split("/")
            rg = load_fourtu_radargram(root, act, path)
            sec = np.asarray(prepare_fitting_branch(rg).background_removed, dtype=np.float64)
            n = sec.shape[0]
            env = np.abs(hilbert(np.concatenate([sec, np.zeros_like(sec)], axis=0), axis=0))[:n]
            img = make_detection_image(rg)
            cache = {a["line"]: (env, img)}
        env, img = cache[a["line"]]
        p = float(img.period_samples)
        tr, k = int(round(float(a["trace"]))), float(a["sample"])
        lo, hi = max(0, int(math.floor(k - 0.5 * p))), min(env.shape[0], int(math.ceil(k + 0.5 * p)) + 1)
        k_peak = lo + int(np.argmax(env[lo:hi, tr]))
        amp = float(env[k_peak, tr])
        left, edge_l = track(env, tr, k_peak, p, -1, amp)
        right, edge_r = track(env, tr, k_peak, p, +1, amp)
        width_tr = left + right + 1
        gm = img.grid_map
        p_l1 = p / gm.samples_per_px
        width_p = width_tr / gm.traces_per_px / p_l1
        s = snr_by_id[a["blind_id"]]["snr_db"]
        quart = None if s is None else (1 if s < q25 else 2 if s < q50 else 3 if s < q75 else 4)
        extra = {}
        vis_level = 0.5 if s is None else max(10 ** (-s / 20.0), 0.1)
        for tag, level in (("m20", 0.1), ("vis", vis_level)):
            lft, el = track(env, tr, k_peak, p, -1, amp, level)
            rgt, er = track(env, tr, k_peak, p, +1, amp, level)
            wp = (lft + rgt + 1) / gm_scale(img) / (p / img.grid_map.samples_per_px)
            extra[f"width_{tag}_P"] = round(wp, 3)
            extra[f"edge_{tag}"] = el or er
        extra["vis_level"] = round(vis_level, 4)
        rows.append({"blind_id": a["blind_id"], "line": a["line"], "trace": a["trace"], "sample": a["sample"],
                     "snr_db": s, "snr_quartile": quart, "width_m6_traces": width_tr,
                     "width_m6_P": round(width_p, 3), "left_traces": left, "right_traces": right,
                     "edge_limited": edge_l or edge_r, "P_line": round(p, 3),
                     "width_in_aihub_p5_p95": 7.79 <= width_p <= 14.67, **extra})
        print(f"{a['blind_id']} W={width_p:.2f}P", flush=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"out": str(OUT), "n": len(rows)}))


if __name__ == "__main__":
    main()
