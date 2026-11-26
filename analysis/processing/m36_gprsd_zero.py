"""M3-6 D3a / D3b (+ D6 local request (b)): record start, CR2 residual wavelet, late-time gain trend.

D3a  per line: RMS per sample (every 5th trace) for samples 0-40, whole-line mean trace and its
     envelope (|hilbert|) in samples 0-60, ratio of sample-0 RMS to the line RMS.
D3b  CementRoad2 lines (and, for comparison, all lines): mean trace of every 500-trace block;
     envelope peak sample in samples 3-45 and the signed extremum nearest to it; spread over
     blocks. A wavelet fixed to the antenna (direct-wave remnant) keeps a constant sample; a
     slab-bottom reflection follows slab thickness.
(b)  late-time gain trend: log RMS per sample over samples 200-320 (raw and after moving-median
     removal, stride-5 approximation: 21 subsampled traces ~ 105 traces); slope of the last
     40 samples vs samples 200-280, and the last-10-sample change. A record cut at the late end
     keeps rising to the last row; a complete record often ends with the gain plateau / taper.
No random numbers. Output: analysis/processing/m36_gprsd_zero.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from scipy import ndimage, signal  # noqa: E402

from m36_gprsd_common import line_ids, load_line, qstats, road_of, write_json  # noqa: E402

BLOCK = 500


def main() -> None:
    lines = {}
    blocks_all = {}
    for fid in line_ids():
        a = load_line(fid)
        n = a.shape[1]
        sub = a[:, ::5]
        rms = np.sqrt((sub ** 2).mean(axis=1))
        res = sub - ndimage.median_filter(sub, size=(1, 21), mode="nearest")
        rms_res = np.sqrt((res ** 2).mean(axis=1))
        mt = a.mean(axis=1)
        env = np.abs(signal.hilbert(mt))
        line_rms = float(np.sqrt((a ** 2).mean()))
        # late-time trend
        def tail(r):
            lr = np.log(r)
            s_mid = np.polyfit(np.arange(200, 281), lr[200:281], 1)[0]
            s_end = np.polyfit(np.arange(281, 321), lr[281:321], 1)[0]
            return {"slope_log_rms_200_280": round(float(s_mid), 5),
                    "slope_log_rms_281_320": round(float(s_end), 5),
                    "ratio_rms_320_to_310": round(float(r[320] / r[310]), 4),
                    "ratio_rms_316_320_to_300_304": round(float(r[316:321].mean() / r[300:305].mean()), 4),
                    "argmax_sample_200_320": int(200 + np.argmax(r[200:321]))}
        # block mean traces
        blk = []
        ref = mt[:46] - mt[:46].mean()
        for b0 in range(0, n - BLOCK + 1, BLOCK):
            m = a[:, b0:b0 + BLOCK].mean(axis=1)
            e = np.abs(signal.hilbert(m))
            kp = 3 + int(np.argmax(e[3:46]))
            w = m[max(0, kp - 8):kp + 9]
            ks = max(0, kp - 8) + int(np.argmax(np.abs(w)))
            # projection on the line-mean wavelet (samples 0-45) and best lag within +-8 samples
            best = (0, -2.0, 0.0)
            for lag in range(-8, 9):
                lo, hi = max(0, lag), min(46, 46 + lag)
                x = m[lo:hi] - m[lo:hi].mean()
                y = ref[lo - lag:hi - lag]
                den = np.sqrt((x ** 2).sum() * (y ** 2).sum())
                cc = float((x * y).sum() / den) if den > 0 else 0.0
                if cc > best[1]:
                    best = (lag, cc, float((x * y).sum() / (y ** 2).sum()) if (y ** 2).sum() > 0 else 0.0)
            blk.append({"block0": b0, "ref_lag": best[0], "ref_corr": round(best[1], 3),
                        "ref_gain": round(best[2], 3),
                        "env_peak_sample": kp, "env_peak": round(float(e[kp]), 1),
                        "signed_ext_sample": ks, "signed_ext_value": round(float(m[ks]), 1),
                        "block_rms": round(float(np.sqrt((a[:, b0:b0 + BLOCK] ** 2).mean())), 1)})
        blocks_all[fid] = blk
        ep = np.array([b["env_peak_sample"] for b in blk])
        ev = np.array([b["env_peak"] for b in blk])
        br = np.array([b["block_rms"] for b in blk])
        lines[fid] = {
            "road": road_of(fid), "n_traces": n, "line_rms": round(line_rms, 1),
            "rms_samples_0_40": [round(float(x), 1) for x in rms[:41]],
            "rms0_over_line_rms": round(float(rms[0] / line_rms), 3),
            "rms_0_4_over_line_rms": round(float(rms[:5].mean() / line_rms), 3),
            "mean_trace_0_60": [round(float(x), 1) for x in mt[:61]],
            "mean_trace_env_0_60": [round(float(x), 1) for x in env[:61]],
            "mean_trace_env_peak_sample_3_45": int(3 + np.argmax(env[3:46])),
            "mean_trace_env_peak_over_line_rms": round(float(env[3:46].max() / line_rms), 3),
            "blocks": {"n": len(blk), "env_peak_sample": qstats(ep, 2),
                       "env_peak_sample_sd": round(float(ep.std()), 2),
                       "env_peak_over_block_rms": qstats(ev / br, 3),
                       "signed_ext_sample": qstats([b["signed_ext_sample"] for b in blk], 2),
                       "signed_ext_sign_share_positive": round(float(np.mean([b["signed_ext_value"] > 0 for b in blk])), 3)},
            "tail_raw": tail(rms), "tail_bg_removed": tail(rms_res),
            "rms_profile_raw_every5": [round(float(x), 1) for x in rms[::5]],
            "rms_profile_bg_removed_every5": [round(float(x), 1) for x in rms_res[::5]],
            "rms_last_12_raw": [round(float(x), 1) for x in rms[309:321]],
            "rms_last_12_bg_removed": [round(float(x), 1) for x in rms_res[309:321]],
        }
    cr2 = [lines[f] for f in lines if lines[f]["road"] == "CementRoad2"]
    cr2_blocks = np.array([b["env_peak_sample"] for f in blocks_all if road_of(f) == "CementRoad2"
                           for b in blocks_all[f]])
    cr2_strong = np.array([b["env_peak_sample"] for f in blocks_all if road_of(f) == "CementRoad2"
                           for b in blocks_all[f] if b["env_peak"] / b["block_rms"] >= 0.15])
    out = {
        "diagnostic": "D3a/D3b/(b)", "block_traces": BLOCK,
        "D3b_CR2": {
            "line_mean_env_peak_sample": [x["mean_trace_env_peak_sample_3_45"] for x in cr2],
            "block_env_peak_sample": qstats(cr2_blocks, 2),
            "block_env_peak_sample_sd": round(float(cr2_blocks.std()), 2),
            "block_env_peak_sample_strong_(env/rms>=0.15)": qstats(cr2_strong, 2),
            "candidate_tau_z_samples": float(np.median(cr2_blocks)),
            "line_mean_signed_peak_sample": [int(np.argmax(np.asarray(x["mean_trace_0_60"][3:46]))) + 3 for x in cr2],
            "block_ref_lag": qstats([b["ref_lag"] for f in blocks_all if road_of(f) == "CementRoad2"
                                     for b in blocks_all[f]], 2),
            "block_ref_corr": qstats([b["ref_corr"] for f in blocks_all if road_of(f) == "CementRoad2"
                                      for b in blocks_all[f]], 3),
            "block_ref_gain": qstats([b["ref_gain"] for f in blocks_all if road_of(f) == "CementRoad2"
                                      for b in blocks_all[f]], 3),
            "share_blocks_corr_ge_0.5_and_abs_lag_le_2": round(float(np.mean(
                [(b["ref_corr"] >= 0.5 and abs(b["ref_lag"]) <= 2) for f in blocks_all
                 if road_of(f) == "CementRoad2" for b in blocks_all[f]])), 3),
        },
        "block_ref_by_road_(own_line_mean_reference)": {
            r: {"lag": qstats([b["ref_lag"] for f in blocks_all if road_of(f) == r for b in blocks_all[f]], 2),
                "corr": qstats([b["ref_corr"] for f in blocks_all if road_of(f) == r for b in blocks_all[f]], 3)}
            for r in ("AsphaltRoad1", "AsphaltRoad2", "CementRoad1", "CementRoad2")},
        "lines": lines,
        "blocks": blocks_all,
    }
    p = write_json("m36_gprsd_zero.json", out)
    print(json.dumps({"status": "ok", "cr2_candidate": out["D3b_CR2"]["candidate_tau_z_samples"], "out": str(p)}))


if __name__ == "__main__":
    main()
