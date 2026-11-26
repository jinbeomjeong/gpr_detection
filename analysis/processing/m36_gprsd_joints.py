"""M3-6 D4 (report only): slab-joint periodicity as a weak check of the trace spacing.

Per line: section = stored - moving median (101 traces); band-pass periods 8-40 samples;
envelope summed over samples 5-60 (slab and joints) -> along-line series s(u); s is
normalised by its moving median (401 traces) and de-meaned. Normalised autocorrelation for
lags 40-600 traces; the highest local maximum in lags 100-300 (3-9 m at dx 0.03) and in the
whole range are reported, plus the same for asphalt roads (control: no transverse joints).
Joint spacing reference: JTG D40-2011 s5.1.3, 4-6 m for ordinary cement concrete slabs
(docs/research/gprsd_source_specs.md [10]) -> 133-200 traces if dx = 0.03 m.
No random numbers. Output: analysis/processing/m36_gprsd_joints.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from scipy import ndimage, signal  # noqa: E402

from m36_gprsd_common import DX_M, bandpass, line_ids, load_line, road_of, write_json  # noqa: E402
from m36_gprsd_diffractor_fit import section_bg_removed  # noqa: E402

LAG_MIN, LAG_MAX = 40, 600


def main() -> None:
    rows = {}
    for fid in line_ids():
        a = load_line(fid)
        sec = section_bg_removed(a)
        env = np.abs(signal.hilbert(bandpass(sec, 8.0, 40.0), axis=0))
        s = env[5:61].sum(axis=0)
        med = ndimage.median_filter(s, size=401, mode="nearest")
        n_zero = int((med <= 0).sum())
        s = np.where(med > 0, s / np.where(med > 0, med, 1.0) - 1.0, 0.0)
        s = s - s.mean()
        n = len(s)
        nfft = 1 << int(np.ceil(np.log2(2 * n)))
        F = np.fft.rfft(s, nfft)
        ac = np.fft.irfft(np.abs(F) ** 2, nfft)[:LAG_MAX + 1]
        ac = ac / ac[0]
        loc, _ = signal.find_peaks(ac[LAG_MIN:LAG_MAX + 1])
        loc = loc + LAG_MIN
        def best(lo, hi):
            sel = loc[(loc >= lo) & (loc <= hi)]
            if not sel.size:
                return None
            j = int(sel[np.argmax(ac[sel])])
            return {"lag_traces": j, "lag_m_at_dx_0.03": round(j * DX_M, 3), "ac": round(float(ac[j]), 4)}
        rows[fid] = {"road": road_of(fid), "n_traces_zero_median": n_zero, "best_lag_100_300": best(100, 300),
                     "best_lag_all": best(LAG_MIN, LAG_MAX),
                     "ac_at_lags": {str(L): round(float(ac[L]), 4) for L in (100, 133, 150, 167, 200, 250, 300)},
                     "top5_peaks": [(int(j), round(float(ac[j]), 4)) for j in loc[np.argsort(-ac[loc])][:5]]}
        print(json.dumps({"line": fid, "best": rows[fid]["best_lag_100_300"]}), flush=True)
    out = {"diagnostic": "D4 (report only)", "lag_range": [LAG_MIN, LAG_MAX],
           "expected_traces_for_4_6_m_at_dx_0.03": [133, 200], "lines": rows}
    p = write_json("m36_gprsd_joints.json", out)
    print(json.dumps({"status": "ok", "out": str(p)}))


if __name__ == "__main__":
    main()
