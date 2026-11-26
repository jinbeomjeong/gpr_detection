"""M3-6 D5: L1 detection-image scale for GPR-SD, r0 (measured ~51) vs the M3-6a values.

r0 : P_line = measure_period_samples(remove_dc(stored)) per line (the r0 code path for
     amplitude_processed data), dz = 0.012 (depth_per_sample_m_rough).
new: P = argv[1] (dataset-fixed period, samples), dz = argv[2] (m/sample).
Both: line_scale(n_s, N, dx 0.03, P, dz, P* 6.0) from src (read-only import), overlap_for,
tile count. Hyperbola width in periods = w dx / (dz P) (lambda/2 = dz P in metres) for
(i) the analytic a = 0.35/0.6/0.8 m (spec_detection_image_v1 s2-3 model) and (ii) the tracked
extent of the strict-QC D2 diffractors (local full file
data/derived/m36_gprsd/m36_gprsd_diffractor_fit_strict.json, outside the label windows, lower
bound of the visible width).
AIHub tpipe reference: 11.4 P (p50), p10-p90 37-58 px at P_px 4.06 -> 9.1-14.3 P.
Usage: python -I m36_gprsd_l1.py <P_new> <dz_new>
Output: analysis/processing/m36_gprsd_l1.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, r"C:\workspace\gpr_detection\src")
import numpy as np  # noqa: E402

from gpr_detection.preprocess.filters import remove_dc  # noqa: E402
from gpr_detection.preprocess.to_image import (DEFAULT_DETECTION_IMAGE_CONFIG,  # noqa: E402
                                               line_scale, measure_period_samples, overlap_for,
                                               tile_layout)
from m36_gprsd_common import (DX_M, DZ_META, line_ids, load_full_json, load_line, qstats,  # noqa: E402
                              write_json)

P_STAR = DEFAULT_DETECTION_IMAGE_CONFIG.p_target_px
L = DEFAULT_DETECTION_IMAGE_CONFIG.tile_px


def scale_row(n_s, n_t, p, dz):
    ls = line_scale(n_s, n_t, DX_M, p, dz, P_STAR, L)
    o, o_phys, capped = overlap_for(ls.traces_per_px, DX_M, P_STAR, DEFAULT_DETECTION_IMAGE_CONFIG)
    tiles = tile_layout(ls.width_px, L, o)
    return {"P_samples": round(p, 3), "dz_m": dz, "s_y": round(ls.s_y_nominal, 4), "s_x": round(ls.s_x_nominal, 4),
            "H_c_px": ls.content_height_px, "W_1_px": ls.width_px, "compression": round(ls.compression, 4),
            "overlap_px": o, "overlap_phys_px": o_phys, "period_capped": capped, "n_tiles": len(tiles),
            "a0.6m_box_width_px": round(2 * 0.6 / DX_M / ls.traces_per_px, 1)}


def main() -> None:
    # scenarios: name:P:dz (argv[1:])
    scen = {}
    for s in sys.argv[1:]:
        name, p, dz = s.split(":")
        scen[name] = (float(p), float(dz))
    # per-diffractor picks are in the local full file only (data/derived/m36_gprsd/, git-ignored)
    diff = load_full_json("m36_gprsd_diffractor_fit_strict.json")
    rows = []
    for fid in line_ids():
        a = load_line(fid)
        p_r0 = measure_period_samples(remove_dc(a))
        row = {"file_id": fid, "n_traces": a.shape[1],
               "r0": scale_row(a.shape[0], a.shape[1], p_r0 if p_r0 else 51.2, DZ_META)
               | {"period_source": "measured" if p_r0 else "dataset_default"}}
        for name, (p, dz) in scen.items():
            row[name] = scale_row(a.shape[0], a.shape[1], p, dz)
        rows.append(row)
    widths_tr = np.array([max(p["u"]) - min(p["u"]) for p in diff["picks"]], float)
    half_min_tr = np.array([min(d["n_left"], d["n_right"]) for d in diff["diffractors"]], float)

    def per_p(p, dz):
        return {"analytic_a_m": {str(a_m): round(2 * a_m / (dz * p), 2) for a_m in (0.35, 0.6, 0.8)},
                "d2_tracked_full_width_P": qstats(widths_tr * DX_M / (dz * p), 2),
                "d2_tracked_2x_min_half_width_P": qstats(2 * half_min_tr * DX_M / (dz * p), 2)}
    names = ["r0", *scen]
    out = {"diagnostic": "D5", "p_target_px": P_STAR, "tile_px": L, "dx_m": DX_M,
           "scenarios": {k: {"P_samples": v[0], "dz_m": v[1]} for k, v in scen.items()},
           "lines": rows,
           "r0_measured_period": qstats([r["r0"]["P_samples"] for r in rows], 3),
           "H_c": {k: sorted({r[k]["H_c_px"] for r in rows}) for k in names},
           "s_x": {k: qstats([r[k]["s_x"] for r in rows], 4) for k in names},
           "n_tiles_total": {k: sum(r[k]["n_tiles"] for r in rows) for k in names},
           "a0.6m_box_width_px": {k: qstats([r[k]["a0.6m_box_width_px"] for r in rows], 1) for k in names},
           "hyperbola_width_periods": {"r0_(P_51.2,dz_0.012)": per_p(51.2, DZ_META),
                                        **{k: per_p(*v) for k, v in scen.items()}},
           "aihub_tpipe_width_periods": {"p50": 11.4, "p10_p90_approx": [9.1, 14.3],
                                          "source": "spec_detection_image_v1 s2-3 (37-58 px / 4.06 px)"}}
    pth = write_json("m36_gprsd_l1.json", out)
    print(json.dumps({"status": "ok", "out": str(pth)}))


if __name__ == "__main__":
    main()
