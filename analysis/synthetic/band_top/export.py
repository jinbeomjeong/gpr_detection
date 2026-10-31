"""Export band-top experiment outputs: npy traces, per-model json truth, calibration amplitudes.

  C:\\ProgramData\\miniforge3\\envs\\python-dev\\python.exe -I analysis/synthetic/band_top/export.py --stage calib|grid|conf3d

Truth (requirement doc section 7-2 Q1): t_top_true_ns = first envelope peak (>= 0.3 x max) of the
pipe-only difference trace (with pipe - without pipe, same layers, same grid). Layer times: envelope
peaks of (layered background - homogeneous background). Envelope = |hilbert(zero-padded trace)| on raw Ez/Ex/Ey,
no filter, no gain. Peak times use parabolic interpolation.
"""

import argparse
import json
import math
import os

os.environ.setdefault("MKL_THREADING_LAYER", "TBB")
from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import find_peaks, hilbert  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent.parent / "data" / "synthetic_gprmax_band_top"
H5 = DATA / "h5"
OUT = DATA / "export"
P_NS = 2.0
THR = 0.3


def load(mid):
    with h5py.File(H5 / f"{mid}.h5", "r") as f:
        dt_ns = float(f.attrs["dt"]) * 1e9
        it = int(f.attrs["Iterations"])
        grp = f["rxs/rx1"]
        comp = [k for k in grp.keys() if k in ("Ex", "Ey", "Ez")]
        assert len(comp) == 1, comp
        tr = np.asarray(grp[comp[0]][()], dtype=np.float64)
    assert tr.shape == (it,), (mid, tr.shape)
    return tr, dt_ns, comp[0]


def env(tr):
    # Zero-pad to twice the length: the FFT Hilbert transform is circular, and late-time energy near the end
    # of the window otherwise leaks envelope into the first samples.
    n = tr.size
    return np.abs(hilbert(np.concatenate([tr, np.zeros(n)])))[:n]


def interp_peak(e, k, dt):
    if 0 < k < len(e) - 1:
        a, b, c = e[k - 1], e[k], e[k + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
        return (k + off) * dt, b - 0.25 * (a - c) * off
    return k * dt, e[k]


def peaks(tr, dt, thr=THR, t_min=None, t_max=None):
    e = env(tr)
    lo = 0 if t_min is None else max(0, int(t_min / dt))
    hi = len(e) if t_max is None else min(len(e), int(t_max / dt) + 1)
    seg = e[lo:hi]
    ref = seg.max()
    idx, _ = find_peaks(seg, height=thr * ref)
    out = [interp_peak(e, lo + int(k), dt) for k in idx]
    return [{"t_ns": float(t), "amp": float(a), "rel": float(a / ref)} for t, a in out], float(ref)


def first_peak(tr, dt, **kw):
    pk, ref = peaks(tr, dt, **kw)
    return pk[0], ref, pk


def stage_calib():
    k0, dt, _ = load("K0")
    res = {"dt_ns": dt}
    for key, mid in (("pec", "KPEC"), ("pvc", "KPVC")):
        tr, _, _ = load(mid)
        p, ref, allp = first_peak(tr - k0, dt)
        res.setdefault("A_pipe", {})[key] = ref
        res.setdefault("t_pipe_first_peak_ns", {})[key] = p["t_ns"]
        res.setdefault("pipe_peaks", {})[key] = allp
    tr, _, _ = load("KL")
    p, ref, _ = first_peak(tr - k0, dt, t_min=6.0)
    res.update({"A_layer_ref": ref, "t_layer_ref_ns": p["t_ns"], "d_ref": 0.6,
                "r_ref": (3.0 - math.sqrt(12.0)) / (3.0 + math.sqrt(12.0))})
    kph, _, _ = load("KPECh")
    k0h, _, _ = load("K0h")
    p, ref, _ = first_peak(kph - k0h, dt)
    res["A_pipe_h01_pec"] = ref
    res["t_pipe_first_peak_h01_pec_ns"] = p["t_ns"]
    (DATA / "calib_amplitudes.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("A_pipe", "A_layer_ref", "t_pipe_first_peak_ns", "t_layer_ref_ns")}))


def stage_models(stage):
    meta = json.loads((HERE / f"models_{stage}.json").read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in meta["models"]}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    cache = {}

    def get(mid):
        if mid not in cache:
            cache[mid] = load(mid)
        return cache[mid]

    for m in meta["models"]:
        if not m["pipe"]:
            continue
        mid = m["id"]
        tot, dt, comp = get(mid)
        bg, _, _ = get(m["bg_id"])
        hb, _, _ = get(m["homog_bg_id"])
        pipe_only = tot - bg
        layer_only = bg - hb
        ptop, a_pipe, pipe_pk = first_peak(pipe_only, dt)
        e_po = env(pipe_only)
        onset10 = float(np.argmax(e_po >= 0.1 * e_po.max()) * dt)
        lay_pk, a_layer = peaks(layer_only, dt, t_min=4.0) if m["layer_interfaces_m"] else ([], 0.0)
        # Direct wave (time zero candidate): envelope max of the background before 6 ns.
        e_bg = env(bg)
        kdz = int(np.argmax(e_bg[: int(6.0 / dt)]))
        t_direct, _ = interp_peak(e_bg, kdz, dt)
        # Preview for Q4: envelope peaks of (total - homogeneous background) = pipe + layers, near the top.
        band = tot - hb
        band_pk, band_ref = peaks(band, dt, thr=0.1, t_min=ptop["t_ns"] - 4 * P_NS, t_max=ptop["t_ns"] + 3 * P_NS)
        names = {}
        for suffix, arr in (("", tot), ("_bg", bg), ("_pipeonly", pipe_only), ("_layeronly", layer_only)):
            fn = f"{mid}{suffix}.npy"
            np.save(OUT / fn, arr.astype(np.float32).reshape(-1, 1))
            names[suffix or "total"] = fn
        rec = {
            "model_id": mid, "dim": m["dim"], "component": comp,
            "array_layout": "[samples, 1] float32 raw field (V/m), gprMax 4.0.1, apex trace; no gain/filter/time-zero",
            "dt_ns": dt, "n_samples": int(tot.size), "time_of_sample0_ns": 0.0,
            "pseudo_band_dx_m_suggested": 0.02,
            "pipe_material": m["pipe"], "pipe_outer_diameter_m": m["pipe_outer_diameter_m"],
            "pipe_top_depth_m": m["pipe_top_depth_m"], "air_gap_m": m["h_m"], "antenna_offset_m": 0.1,
            "center_freq_hz": 5e8, "period_ns": P_NS,
            "eps_r": m["layer_eps_r"][0], "layer_interfaces_m": m["layer_interfaces_m"],
            "layer_eps_r": m["layer_eps_r"], "eps_eff_to_top": m["eps_eff_to_top"],
            "v_eff_to_top_m_per_ns": m["v_eff_to_top_m_per_ns"],
            "tau_top_vertical_ns": m["tau_top_vertical_ns"], "layer_tau_vertical_ns": m["layer_tau_vertical_ns"],
            "t_top_true_ns": ptop["t_ns"],
            "t_top_true_definition": "first envelope peak (>= 0.3 x max) of pipe-only = with pipe - without pipe",
            "pipe_only_peaks": pipe_pk, "pipe_only_env_max": a_pipe,
            "layer_times_ns": [p["t_ns"] for p in lay_pk], "layer_peaks": lay_pk, "layer_only_env_max": a_layer,
            "ratio_layer_to_pipe_measured": (a_layer / a_pipe) if a_layer else None,
            "layer_dt_measured_P": [(p["t_ns"] - ptop["t_ns"]) / P_NS for p in lay_pk],
            "time_zero_candidates_ns": {"source_peak": (math.sqrt(2) / 5e8 * 1e9 if m.get("wavelet") != "ring"
                                                        else 3.3), "direct_wave_peak": t_direct},
            "wavelet": m.get("wavelet", "ricker"),
            "pipe_only_onset10_ns": onset10,
            "pipe_only_onset10_definition": "first time the pipe-only envelope reaches 0.1 x its max (visible start)",
            "apex": {"trace_index": 0, "envelope_peak_ns": ptop["t_ns"]},
            "band_preview_peaks_thr0.1": band_pk, "band_preview_ref": band_ref,
            "design": {k: m.get(k) for k in ("layer_side", "layer_dt_target_P", "ratio_target", "r_design",
                                             "eps2_capped", "dipole", "pair_2d", "role", "strong_kind", "bed_eps_r",
                                             "bed_thickness_m", "plate_top_depth_m", "plate_width_m",
                                             "plate_thickness_m", "plate_note", "geometry_of", "wavelet")},
            "files": names | {"gprmax_input": f"analysis/synthetic/band_top/inputs/{mid}.in",
                              "h5": f"data/synthetic_gprmax_band_top/h5/{mid}.h5",
                              "h5_bg": f"data/synthetic_gprmax_band_top/h5/{m['bg_id']}.h5"},
        }
        (OUT / f"{mid}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        rows.append({"id": mid, "pipe": m["pipe"], "side": m.get("layer_side"), "dt_P": m.get("layer_dt_target_P"),
                     "ratio_t": m.get("ratio_target"), "ratio_m": rec["ratio_layer_to_pipe_measured"],
                     "t_top": ptop["t_ns"], "tau_v": m["tau_top_vertical_ns"], "t_dir": t_direct,
                     "lay_dt_P": rec["layer_dt_measured_P"], "n_pipe_pk": len(pipe_pk)})
    (OUT / f"summary_{stage}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(json.dumps({"stage": stage, "exported": len(rows)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="calib, or the name of any models_<stage>.json")
    args = ap.parse_args()
    if args.stage == "calib":
        stage_calib()
    else:
        stage_models(args.stage)


if __name__ == "__main__":
    main()
