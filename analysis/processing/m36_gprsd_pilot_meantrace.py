"""M3-6 planning pilot 2 (exploratory): residual horizontal (mean-trace) structure near the record top.

The stored GPR-SD amplitude has had the "ground echo" removed by MATGPR. If removal was a global
mean-trace subtraction, the whole-line mean trace is ~0; a local-window mean (500 traces) then shows
what horizontal energy is left and where (sample index). A residual wavelet near the top is a
time-zero clue (direct/surface wave remnant), not a verdict.

Inputs (read-only): GPR-SD 05_radargrams_npy. Output: analysis/processing/m36_gprsd_pilot_meantrace.json
No random numbers. Units: samples (dt-free).
"""

import json
from pathlib import Path

import numpy as np

SD = Path(r"C:\workspace\GPR_Data\GPR-SD_중국USTC_도로GPR_가공본_261016") / "05_radargrams_npy"
OUT = Path(r"C:\workspace\gpr_detection\analysis\processing\m36_gprsd_pilot_meantrace.json")
WIN = 500


def main():
    out = {}
    for p in sorted(SD.glob("*.npy")):
        a = np.load(p, mmap_mode="r")
        n = a.shape[1]
        whole = np.asarray(a, dtype=np.float64).mean(axis=1)
        rms = float(np.sqrt(np.mean(np.asarray(a[:, ::7], dtype=np.float64) ** 2)))
        # local mean traces of non-overlapping 500-trace blocks
        k = n // WIN
        blk = np.stack([np.asarray(a[:, i * WIN:(i + 1) * WIN], dtype=np.float64).mean(axis=1) for i in range(k)], axis=1)
        absblk = np.abs(blk)
        prof = np.median(absblk, axis=1)  # median over blocks of |local mean| per sample
        # top-40-sample wavelet of the whole-line mean trace
        out[p.stem] = {
            "rms_all": round(rms, 1),
            "whole_mean_trace_0_40": [round(float(x), 1) for x in whole[:41]],
            "whole_mean_trace_absmax_sample": int(np.argmax(np.abs(whole))),
            "local_mean_absmedian_by_sample_0_60": [round(float(x), 1) for x in prof[:61]],
            "local_mean_absmedian_argmax_sample": int(np.argmax(prof)),
            "local_mean_absmedian_over_rms_max": round(float(prof.max() / rms), 3),
            "n_blocks": int(k),
        }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"status": "ok", "n_lines": len(out), "out": str(OUT)}))


if __name__ == "__main__":
    main()
