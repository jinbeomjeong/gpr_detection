"""M3-6 D2 self-check: tracker + QC fit on synthetic point-diffraction hyperbolas.

Ricker wavelet (period ~20 samples), amplitude ~ h/tau, white noise (amplitude SNR 3 and 10),
g in {1.2, 1.5, 2.5, 3.5} samples/trace (dz 0.025-0.0086 m at dx 0.03), apex h in {40, 80, 150}
samples, tau_z = 0. Same band-pass, track() and qc_fit() as m36_gprsd_diffractor_fit.py, with and
without the moving-median (101 traces) background removal used on the real data.
Seed 20261126. Output: analysis/processing/m36_gprsd_synth_check.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

import m36_gprsd_diffractor_fit as m  # noqa: E402
from m36_gprsd_common import bandpass, qstats, write_json  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(20261126)
    P = 20.0
    rows = []
    for g in (1.2, 1.5, 2.5, 3.5):
        for h in (40.0, 80.0, 150.0):
            for snr in (3.0, 10.0):
                n_s, n_t, u0 = 321, 301, 150.3
                k = np.arange(n_s)[:, None]
                u = np.arange(n_t)[None, :]
                tau = np.sqrt(h ** 2 + ((u - u0) * g) ** 2)
                x = (k - tau) / P
                wav = (1 - 2 * (np.pi * x) ** 2) * np.exp(-(np.pi * x) ** 2)
                img = (h / tau) * wav + rng.normal(0, 1.0 / snr, (n_s, n_t))
                for bgmode in ("none", "median101"):
                    src = img if bgmode == "none" else m.section_bg_removed(img)
                    b = bandpass(src, *m.PICK_BAND)
                    env = np.abs(signal.hilbert(b, axis=0))
                    k0 = int(np.argmax(env[:, 150]))
                    us, ts, s, a0, ka = m.track(b, k0, 150, P)
                    info, why = m.qc_fit(us, ts, P, rng)
                    rows.append({"g": g, "h": h, "snr": snr, "bg": bgmode, "why": why,
                                 "g_err_rel": None if info is None else round(info["g"] / g - 1, 4),
                                 "h_fit_minus_h": None if info is None else round(info["h"] - h, 2),
                                 "n_left": None if info is None else info["n_left"],
                                 "n_right": None if info is None else info["n_right"]})
    summ = {bg: {"n": sum(r["bg"] == bg for r in rows),
                 "n_pass": sum(r["bg"] == bg and r["why"] == "pass" for r in rows),
                 "g_err_rel_pass": qstats([r["g_err_rel"] for r in rows if r["bg"] == bg and r["why"] == "pass"], 4),
                 "abs_g_err_rel_max_pass": max(abs(r["g_err_rel"]) for r in rows
                                               if r["bg"] == bg and r["why"] == "pass")}
            for bg in ("none", "median101")}
    write_json("m36_gprsd_synth_check.json", {"seed": 20261126, "summary": summ, "rows": rows})
    print(json.dumps({"status": "ok", "summary": summ}))


if __name__ == "__main__":
    main()
