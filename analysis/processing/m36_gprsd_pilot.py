"""M3-6 planning pilot (exploratory, not a verdict): GPR-SD carrier period and mean-trace structure.

Purpose: give the M3-6 plan concrete numbers before committing to diagnostics.
  1. Spectrum of the stored amplitude in pipe-mark windows (all 17 marks x 3 lines):
     peak period, spectral centroid period, power share at periods < 20 samples.
  2. Trace autocorrelation near each mark: lag of first zero crossing and first minimum
     (quarter / half carrier period if the data were a simple wavelet).
  3. Whole-line mean trace and RMS-vs-sample profile (samples 0-320) for residual
     direct/surface wave and gain shape.
  4. PNG renders of a few mark windows (to scratchpad, for visual inspection only).

Inputs (read-only): GPR-SD 05_radargrams_npy, 01_메타데이터/marks.csv.
Output: analysis/processing/m36_gprsd_pilot.json public (no "windows" = pipe-mark window positions)
  + data/derived/m36_gprsd/m36_gprsd_pilot.json full (git-ignored; split 2026-11-26, user decision)
  (+ PNGs to the folder given by argv[1] if any).
No random numbers. Units: samples and traces (dt-free); ns only via hypotheses A/B where stated.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m36_gprsd_common import write_json_split  # noqa: E402

SD =Path(r"C:\workspace\GPR_Data\GPR-SD_중국USTC_도로GPR_가공본_261016")
OUT = Path(r"C:\workspace\gpr_detection\analysis\processing\m36_gprsd_pilot.json")
DT_A, DT_B = 97.0 / 321, 97.0 / 380
HALF = 300  # traces on each side of a mark


def read_marks():
    with open(SD / "01_메타데이터" / "marks.csv", encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r["type"] == "pipeline"]


def spectrum_stats(win):
    """win: [samples, traces] float64. Trace-averaged |rfft| of DC-removed traces, zero-padded x8."""
    a = win - win.mean(axis=0, keepdims=True)
    n = a.shape[0]
    nfft = 1 << int(np.ceil(np.log2(8 * n)))
    spec = np.abs(np.fft.rfft(a, nfft, axis=0)).mean(axis=1)
    f = np.fft.rfftfreq(nfft, d=1.0)  # cycles / sample
    ok = f >= 1.0 / 128.0
    p = spec[ok] ** 2
    fk = f[ok]
    peak = 1.0 / fk[np.argmax(spec[ok])]
    centroid = 1.0 / float((fk * p).sum() / p.sum())
    hi = float(p[fk > 1.0 / 20.0].sum() / p.sum())
    # secondary peak at periods < 30 samples
    sel = fk > 1.0 / 30.0
    peak_short = 1.0 / fk[sel][np.argmax(spec[ok][sel])]
    ratio_short = float(spec[ok][sel].max() / spec[ok].max())
    return float(peak), centroid, hi, float(peak_short), ratio_short


def autocorr_lags(win):
    a = win - win.mean(axis=0, keepdims=True)
    n = a.shape[0]
    ac = np.zeros(n)
    for j in range(a.shape[1]):
        x = a[:, j]
        c = np.correlate(x, x, mode="full")[n - 1:]
        if c[0] > 0:
            ac += c / c[0]
    ac /= a.shape[1]
    zc = int(np.argmax(ac < 0)) if np.any(ac < 0) else -1
    # first local minimum after zero crossing
    mn = -1
    if zc > 0:
        seg = ac[zc:]
        d = np.diff(seg)
        idx = np.where((d[:-1] < 0) & (d[1:] >= 0))[0]
        if idx.size:
            mn = int(zc + idx[0] + 1)
    return zc, mn, float(ac[mn]) if mn > 0 else None


def q(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"p10": round(float(np.percentile(a, 10)), 2), "p50": round(float(np.median(a)), 2),
            "p90": round(float(np.percentile(a, 90)), 2), "n": int(a.size)}


def main():
    png_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    marks = read_marks()
    rows = []
    for m in marks:
        for fid in m["files"].split(" | "):
            a = np.load(SD / "05_radargrams_npy" / f"{fid}.npy", mmap_mode="r")
            t = int(m["location_trace"])
            i0, i1 = max(0, t - HALF), min(a.shape[1], t + HALF + 1)
            w = np.asarray(a[:, i0:i1], dtype=np.float64)
            pk, cen, hi, pks, rs = spectrum_stats(w[20:200])
            zc, mn, acmin = autocorr_lags(w[20:200])
            rows.append({"xlsx_row": m["xlsx_row"], "file": fid, "trace": t, "n_tr": i1 - i0,
                         "peak_period": round(pk, 2), "centroid_period": round(cen, 2),
                         "power_share_period_lt20": round(hi, 4),
                         "short_peak_period": round(pks, 2), "short_peak_rel_amp": round(rs, 3),
                         "ac_zero_lag": zc, "ac_min_lag": mn,
                         "ac_min_val": None if acmin is None else round(acmin, 3)})
            if png_dir is not None and fid.endswith("_1"):
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                clip = np.percentile(np.abs(w), 99)
                fig, ax = plt.subplots(figsize=(12, 4.5), dpi=110)
                ax.imshow(w, cmap="gray", vmin=-clip, vmax=clip, aspect="auto",
                          extent=(i0 - 0.5, i1 - 0.5, w.shape[0] - 0.5, -0.5))
                ax.axvline(t, color="r", lw=0.6, ls="--")
                ax.set_xlabel("trace")
                ax.set_ylabel("sample")
                ax.set_title(f"{fid} mark row {m['xlsx_row']} trace {t} (stored amplitude, p99 clip)")
                fig.tight_layout()
                fig.savefig(png_dir / f"{fid}_row{m['xlsx_row']}.png")
                plt.close(fig)

    # whole-line mean trace and RMS profile
    lines = {}
    for p in sorted((SD / "05_radargrams_npy").glob("*.npy")):
        a = np.load(p, mmap_mode="r")
        sub = np.asarray(a[:, ::5], dtype=np.float64)
        mt = sub.mean(axis=1)
        rms = np.sqrt((sub ** 2).mean(axis=1))
        lines[p.stem] = {
            "mean_trace_absmax": round(float(np.abs(mt).max()), 2),
            "mean_trace_absmax_sample": int(np.argmax(np.abs(mt))),
            "rms_by_sample_every10": [round(float(x), 1) for x in rms[::10]],
            "rms_first10": [round(float(x), 1) for x in rms[:10]],
            "rms_argmax_sample": int(np.argmax(rms)),
            "whole_peak_period": round(spectrum_stats(sub[20:200])[0], 2),
            "whole_centroid_period": round(spectrum_stats(sub[20:200])[1], 2),
            "whole_power_share_period_lt20": round(spectrum_stats(sub[20:200])[2], 4),
        }

    summ = {k: q([r[k] for r in rows]) for k in
            ("peak_period", "centroid_period", "power_share_period_lt20", "short_peak_period",
             "short_peak_rel_amp", "ac_zero_lag", "ac_min_lag")}
    out = {"n_windows": len(rows), "half_window_traces": HALF, "sample_range_spectrum": [20, 200],
           "summary": summ,
           "summary_ns_hypA": {k: {kk: (round(vv * DT_A, 2) if kk != "n" else vv) for kk, vv in summ[k].items()}
                               for k in ("peak_period", "centroid_period")},
           "summary_ns_hypB": {k: {kk: (round(vv * DT_B, 2) if kk != "n" else vv) for kk, vv in summ[k].items()}
                               for k in ("peak_period", "centroid_period")},
           "windows": rows, "lines": lines,
           "note": "exploratory; stored (MATGPR-processed) amplitude, no further filtering; periods in samples"}
    pub, full = write_json_split(OUT.name, out)  # same serialisation as before (indent=1)
    print(json.dumps({"status": "ok", "n_windows": len(rows), "out": str(pub), "out_full": str(full)}))


if __name__ == "__main__":
    main()
