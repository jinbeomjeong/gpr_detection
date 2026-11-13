"""Which Guangzhou pipe lines exceed 256 px content height in C', and by how much.

Also: measured dominant period of a few MALA lines (stored amplitudes, DC removed,
same estimator as detection_image_period_check.py) and O_phys under two rules.
Read-only, metadata + 6 small npy files.
"""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

GZ = Path(r"C:\workspace\GPR_Data\광저우대_중국_관로GPR_가공본_261016")
FOURTU = Path(r"C:\workspace\gpr_detection\docs\processing\fourtu_lines_v1.csv")


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def padded_period(a, pad=8):
    a = np.asarray(a, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    n = a.shape[0]
    nfft = 1 << int(np.ceil(np.log2(pad * n)))
    spec = np.abs(np.fft.rfft(a, nfft, axis=0)).mean(axis=1)
    f = np.fft.rfftfreq(nfft, d=1.0)
    ok = (f >= 4.0 / n) & (f <= 0.25)
    return 1.0 / float(f[ok][np.argmax(spec[ok])])


def o_phys(m_per_px, p_star, rule):
    w = 2.0 / m_per_px
    if rule == "period_cap":
        w = min(w, 20.0 * p_star)
    return max(64, math.ceil(w - 1e-9))


out = {}
gz = [r for r in rows(GZ / "01_메타데이터" / "radargrams_index.csv") if r["target"] == "pipe"]

# measured MALA periods (first file of each MALA folder)
mala_p = {}
for r in gz:
    if r["format"] != "MALA .rd3":
        continue
    folder = r["file_id"].split("/")[1]
    if folder in mala_p:
        continue
    a = np.load(GZ / "05_radargrams_npy" / f"{r['file_id']}.npy", mmap_mode="r")
    mala_p[folder] = {"period_samples": round(padded_period(np.asarray(a)), 2),
                      "dt_ns": float(r["dt_ns"]), "n_samples": int(r["n_samples"])}
out["mala_measured_period"] = mala_p

for p_star in (4.06, 6.0):
    over = Counter()
    cvals = defaultdict(list)
    ophys = defaultdict(list)
    for r in gz:
        dt = float(r["dt_ns"])
        ns = int(r["n_samples"])
        dx = float(r["trace_dx_m"])
        f = float(r["antenna_MHz"]) if r["antenna_MHz"] else None
        tw = round(float(r["time_window_ns"]))
        if f is None:
            folder = r["file_id"].split("/")[1]
            per = mala_p[folder]["period_samples"]
            grp = "MALA"
        elif f == 400 and tw in (30, 50):
            per = {30: 2.93, 50: 2.11}[tw] / dt
            grp = f"400MHz_{tw}ns"
        else:
            per = (1000.0 / f) / dt
            grp = f"{int(f)}MHz"
        s_y = per / p_star
        h = round(ns / s_y)
        dz = 0.10 * dt / 2
        s_x = s_y * dz / dx
        mpp = s_x * dx
        if h > 256:
            over[grp] += 1
            cvals[grp].append(round(h / 256, 2))
        ophys[grp].append((o_phys(mpp, p_star, "metres"), o_phys(mpp, p_star, "period_cap")))
    out[f"P{p_star}"] = {
        "height_gt_256_by_group": dict(over),
        "compression_c_by_group": {g: [min(v), max(v)] for g, v in cvals.items()},
        "o_phys_metres_max_by_group": {g: max(x[0] for x in v) for g, v in ophys.items()},
        "o_phys_periodcap_max_by_group": {g: max(x[1] for x in v) for g, v in ophys.items()},
    }

# 4TU per-line eps
for p_star in (4.06, 6.0):
    om, oc = [], []
    for r in rows(FOURTU):
        dt = float(r["dt_ns"])
        v = 0.299792458 / math.sqrt(float(r["eps_r_used"]))
        s_y = (2.51 / dt) / p_star
        mpp = s_y * v * dt / 2
        om.append(o_phys(mpp, p_star, "metres"))
        oc.append(o_phys(mpp, p_star, "period_cap"))
    out[f"4TU_P{p_star}"] = {"o_phys_metres_min_max": [min(om), max(om)],
                              "o_phys_periodcap_min_max": [min(oc), max(oc)]}
print(json.dumps(out, ensure_ascii=False, indent=1))
