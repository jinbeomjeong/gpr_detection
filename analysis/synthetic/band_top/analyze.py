"""Physics summary and pick-rule preview for the band-top experiment (no src/ import).

  C:\\ProgramData\\miniforge3\\envs\\python-dev\\python.exe -I analysis/synthetic/band_top/analyze.py

Preview only: the depth-path verdict (pseudo-band, noise, bandpass, masked median, grouping) is done by
gpr-processing-expert. Here the current band pick rule is emulated on the clean apex A-scan:
box [t_top - 0.4P + delta, t_top + 0.8P], window [k0 - 1P, k1 + 0.5P] (f_c known), threshold
0.3 x envelope max inside the box, first envelope peak in the window. Two background cases:
  bg_none   : raw trace (layers stay, as when the band covers the whole profile, AIHub bg_none 37 %)
  bg_masked : trace - same-model background (what masked median gives when the layer extends beyond the band)
"""

import json
import os

os.environ.setdefault("MKL_THREADING_LAYER", "TBB")
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import find_peaks, hilbert  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent.parent
EXP = PROJECT / "data" / "synthetic_gprmax_band_top" / "export"
P = 2.0
DELTAS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]


def pick(tr, dt, t_top, delta):
    n = tr.size
    e = np.abs(hilbert(np.concatenate([tr, np.zeros(n)])))[:n]  # zero-padded (non-circular) envelope
    k0 = int(round((t_top - 0.4 * P + delta * P) / dt))
    k1 = int(round((t_top + 0.8 * P) / dt))
    if k1 <= k0:
        return None
    e_box = e[k0:k1 + 1].max()
    lo, hi = max(0, k0 - int(round(P / dt))), min(len(e) - 1, k1 + int(round(0.5 * P / dt)))
    seg = e[lo:hi + 1]
    idx, _ = find_peaks(seg, height=0.3 * e_box)
    if len(idx) == 0:
        return None
    k = lo + int(idx[0])
    a, b, c = e[k - 1], e[k], e[k + 1]
    den = a - 2 * b + c
    off = 0.5 * (a - c) / den if den != 0 else 0.0
    return (k + off) * dt


def classify(err_p, t_pick, layers):
    if abs(err_p) <= 0.25:
        return "top"
    if any(abs(t_pick - tl) <= 0.25 * P for tl in layers):
        return "layer"
    return "other"


SETS = {"L_": "grid (single interface, ricker)", "W_": "strong: tuned wet bed (ricker)",
        "PL_": "strong: steel protection plate (ricker)", "RL_": "ringing wavelet (grid geometry)"}
CMAP = {"top": 0, "layer": 1, "other": 2, "none": 3}


def set_of(mid):
    return next((k for k in SETS if mid.startswith(k)), None)


def row_label(x):
    lab = f"{x['side'][0].upper()} {x['dt_P']:.1f}P"
    if x["id"].startswith("PL_"):
        lab = f"w{x['plate_width_m']:.1f} " + lab
    return lab + (f" r{x['ratio']:.2f}" if x["ratio"] is not None else "")


def figure(rows, prefix, out, mode="bg_none"):
    from matplotlib.colors import ListedColormap
    cm = ListedColormap(["#2a9d8f", "#e76f51", "#8d99ae", "#ffffff"])
    sel_all = [x for x in rows if x["id"].startswith(prefix)]
    pipes = [p for p in ("pec", "pvc") if any(x["pipe"] == p for x in sel_all)]
    h = max(3.0, 0.24 * max(len([x for x in sel_all if x["pipe"] == p]) for p in pipes) + 1.5)
    fig, axes = plt.subplots(1, len(pipes), figsize=(5.6 * len(pipes), h), constrained_layout=True, squeeze=False)
    for ax, pk in zip(axes[0], pipes):
        sel = sorted([x for x in sel_all if x["pipe"] == pk],
                     key=lambda x: (x.get("plate_width_m") or 0, x["side"] != "above",
                                    -x["dt_P"] if x["side"] == "above" else x["dt_P"], x["ratio_target"] or 0))
        mat = np.array([[CMAP[x["picks"][f"{mode}|{d}"]["cls"]] for d in DELTAS] for x in sel])
        ax.imshow(mat, cmap=cm, vmin=0, vmax=3, aspect="auto")
        for i, x in enumerate(sel):
            for j, d in enumerate(DELTAS):
                e = x["picks"][f"{mode}|{d}"]["err_P"]
                if e is not None:
                    ax.text(j, i, f"{e:+.2f}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(len(DELTAS)), [f"{d:+.1f}" for d in DELTAS])
        ax.set_yticks(range(len(sel)), [row_label(x) for x in sel], fontsize=6)
        ax.set_xlabel("box top offset delta (P = 2 ns, negative = up)")
        ax.set_title(f"{pk.upper()}: pick error (P), {mode}", fontsize=9)
    fig.suptitle(f"{SETS[prefix]} - current band pick rule on clean apex A-scan "
                 "(green = pipe top within 0.25 P, red = layer, grey = other)", fontsize=8)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(EXP.glob("*.json"))
            if not p.name.startswith(("summary", "analysis"))]
    rows = []
    for r in recs:
        tot = np.load(EXP / r["files"]["total"])[:, 0].astype(np.float64)
        bg = np.load(EXP / r["files"]["_bg"])[:, 0].astype(np.float64)
        lay = np.load(EXP / r["files"]["_layeronly"])[:, 0].astype(np.float64)
        homog = bg - lay
        dt, tt = r["dt_ns"], r["t_top_true_ns"]
        d = r["design"]
        local = d.get("strong_kind") == "steel_plate"
        onset10 = r.get("pipe_only_onset10_ns")
        row = {"id": r["model_id"], "set": set_of(r["model_id"]), "dim": r["dim"], "pipe": r["pipe_material"],
               "side": d.get("layer_side"), "dt_P": d.get("layer_dt_target_P"),
               "ratio_target": d.get("ratio_target"), "ratio": r["ratio_layer_to_pipe_measured"],
               "plate_width_m": d.get("plate_width_m"), "wavelet": r.get("wavelet", "ricker"),
               "layer_dt_meas_P": r["layer_dt_measured_P"], "t_top": tt,
               "t_top_minus_tau_v": tt - r["tau_top_vertical_ns"],
               "onset10_minus_top_P": None if onset10 is None else (onset10 - tt) / P,
               "delta_at_onset10": None if onset10 is None else (onset10 - tt) / P + 0.4,
               "t_top_minus_direct": tt - r["time_zero_candidates_ns"]["direct_wave_peak"],
               "depth_err_direct_zero_cm": 100 * (r["v_eff_to_top_m_per_ns"] *
                                                 (tt - r["time_zero_candidates_ns"]["direct_wave_peak"]) / 2
                                                 - r["pipe_top_depth_m"]),
               "n_pipe_peaks": len(r["pipe_only_peaks"]),
               "bg_masked_meaning": ("outside traces have no plate: masked median = homogeneous background"
                                     if local else "layer extends beyond the band: masked median = layered bg"),
               "picks": {}}
        masked = (tot - homog) if local else (tot - bg)
        deltas = list(DELTAS) + ([round(row["delta_at_onset10"], 3)] if onset10 is not None else [])
        # bg_direct: direct wave / antenna ringing removed (flat across the profile) but the layer kept, i.e. a layer
        # that exists only inside the band (masked median with layer-free outside traces).
        for mode, tr in (("bg_none", tot), ("bg_masked", masked), ("bg_direct", tot - homog)):
            for dd in deltas:
                key = f"{mode}|{dd}" if dd in DELTAS else f"{mode}|onset10"
                tp = pick(tr, dt, tt, dd)
                if tp is None:
                    row["picks"][key] = {"err_P": None, "cls": "none"}
                    continue
                ep = (tp - tt) / P
                row["picks"][key] = {"err_P": ep, "cls": classify(ep, tp, r["layer_times_ns"])}
        rows.append(row)
    (EXP / "analysis_preview.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    agg = {}
    for prefix in SETS:
        grid = [x for x in rows if x["id"].startswith(prefix)]
        if not grid:
            continue
        for mode in ("bg_none", "bg_masked", "bg_direct"):
            for dd in [*DELTAS, "onset10"]:
                for sel_name, sel in (("all", grid), ("far>1P", [x for x in grid if x["dt_P"] > 1.0]),
                                      ("near<=1P", [x for x in grid if x["dt_P"] <= 1.0]),
                                      ("above", [x for x in grid if x["side"] == "above"]),
                                      ("below", [x for x in grid if x["side"] == "below"])):
                    cls = [x["picks"][f"{mode}|{dd}"]["cls"] for x in sel if f"{mode}|{dd}" in x["picks"]]
                    if cls:
                        agg[f"{prefix}|{mode}|{dd}|{sel_name}"] = {c: cls.count(c) for c in CMAP} | {"n": len(cls)}
    (EXP / "analysis_preview_agg.json").write_text(json.dumps(agg, indent=1), encoding="utf-8")

    docs = PROJECT / "docs" / "processing"
    outs = {"L_": "band_top_synthetic_preview.png", "W_": "band_top_synthetic_preview_wetbed.png",
            "PL_": "band_top_synthetic_preview_plate.png", "RL_": "band_top_synthetic_preview_ring.png"}
    made = []
    for prefix, name in outs.items():
        if any(x["id"].startswith(prefix) for x in rows):
            figure(rows, prefix, docs / name)
            made.append(name)
    for prefix, name in (("RL_", "band_top_synthetic_preview_ring_bgdirect.png"),
                         ("W_", "band_top_synthetic_preview_wetbed_bgdirect.png")):
        if any(x["id"].startswith(prefix) for x in rows):
            figure(rows, prefix, docs / name, mode="bg_direct")
            made.append(name)
    print(json.dumps({"rows": len(rows), "figures": made}))




if __name__ == "__main__":
    main()

