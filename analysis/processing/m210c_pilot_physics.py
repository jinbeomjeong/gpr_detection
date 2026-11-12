"""M2-10 (c) pilot physics check (before the seal; `cur`-side physics only).

Reads the EM pilot h5 of Z_HSTEP / Z_HWAVE (raw Ez) and reports, without the
P3 correction (no `cand` quantity, spec m210_c_qc.md section 4-4):
* the envelope peak time of the merged early band per trace (0.25-2 f_c band-pass,
  Hilbert envelope, early window) and its spread (std over traces),
* the pipe apex envelope peak time at the two apex traces,
* the pipe time difference high_gap - low_gap against the geometric 2 dh / c,
* the noise-free pair depth difference without correction, v * dt_pipe / 2.

Output: docs/processing/m210c_pilot_physics.json (LF, relative paths).
Run: conda run -n gpr-dev python -I analysis/processing/m210c_pilot_physics.py
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import h5py
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "data" / "synthetic_gprmax_c_qc" / "pilot"
MODELS = ROOT / "analysis" / "synthetic" / "c_qc" / "models_c_qc.json"
OUT = ROOT / "docs" / "processing" / "m210c_pilot_physics.json"
C = 0.299792458  # m/ns
F_C = 500e6


def parabolic(y, i):
    if 0 < i < len(y) - 1:
        a, b, c = y[i - 1], y[i], y[i + 1]
        den = a - 2 * b + c
        if den != 0:
            return i + 0.5 * (a - c) / den
    return float(i)


def main() -> None:
    models = {m["id"]: m for m in json.loads(MODELS.read_text(encoding="utf-8"))["models"]}
    out = {"note": "pilot h5, raw Ez, no P3 correction computed (cur-side physics only)"}
    for mid in ("Z_HSTEP", "Z_HWAVE"):
        with h5py.File(PILOT / f"{mid}_merged.h5", "r") as f:
            ez = np.asarray(f["rxs/rx1/Ez"], dtype=float)
            dt_ns = float(f.attrs["dt"]) * 1e9
        sos = butter(4, [0.25 * F_C, 2.0 * F_C], btype="band", fs=1e9 / dt_ns, output="sos")
        filt = sosfiltfilt(sos, ez, axis=0)
        env = np.abs(hilbert(filt, axis=0))
        n_early = int(round(6.0 / dt_ns))
        band_t = np.array([parabolic(env[:n_early, j], int(np.argmax(env[:n_early, j]))) * dt_ns
                           for j in range(env.shape[1])])
        m = models[mid]
        h = np.array(m["antenna_height_m"], dtype=float)
        slope = float(np.polyfit(h, band_t, 1)[0])
        lo_k, hi_k = int(round(12.0 / dt_ns)), int(round(20.0 / dt_ns))
        pipes = {}
        for p in m["pipes"]:
            j = p["apex_trace_index"]
            k = lo_k + int(np.argmax(env[lo_k:hi_k, j]))
            pipes[p["side"]] = {
                "trace": j, "h_m": p["antenna_height_at_apex_m"],
                "t_pipe_ns": round(parabolic(env[:, j], k) * dt_ns, 4),
                "t_band_ns": round(float(band_t[j]), 4),
            }
        dh = pipes["high_gap"]["h_m"] - pipes["low_gap"]["h_m"]
        dt_pipe = pipes["high_gap"]["t_pipe_ns"] - pipes["low_gap"]["t_pipe_ns"]
        dt_band = pipes["high_gap"]["t_band_ns"] - pipes["low_gap"]["t_band_ns"]
        v = C / 3.0
        out[mid] = {
            "dt_ns": dt_ns,
            "band_peak_std_ns": round(float(np.std(band_t - band_t.mean())), 4),
            "band_peak_slope_ns_per_m": round(slope, 3),
            "geometric_slope_ns_per_m": round(2 / C, 3),
            "band_fraction_of_geometric": round(slope / (2 / C), 3),
            "pipes": pipes,
            "dh_m": round(dh, 4),
            "geometric_2dh_over_c_ns": round(2 * dh / C, 4),
            "pipe_dt_high_minus_low_ns": round(dt_pipe, 4),
            "band_dt_high_minus_low_ns": round(dt_band, 4),
            "pair_depth_diff_uncorrected_cm": round(100 * v * dt_pipe / 2, 2),
        }
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"out": "docs/processing/m210c_pilot_physics.json"}))


if __name__ == "__main__":
    main()
