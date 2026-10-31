"""Generate gprMax inputs for the pipe_band "which reflection is the pipe top" experiment (D3).

Requirement: docs/processing/band_box_convention_depth.md section 7.
Main set: 2D TMz layered soil + pipe, apex trace only (pseudo-band is built downstream by repeating
it, [B] section 9-2). The 2D out-of-plane axis is the pipe axis and also the survey-line direction, so
a 2D A-scan above the pipe equals every trace of a pipe lying parallel to the line (theta = 0) with a
line source. 3D confirmation: the same layered configuration with a point dipole (x and y polarised).

Stages
  calib : homogeneous references + one layer-only reference (amplitude calibration)
  grid  : layered grid; needs calib_amplitudes.json (written by export.py --stage calib)
  conf3d: 3D confirmation models; needs grid_design.json (written by the grid stage)

Run (python-dev, no gprMax import):
  C:\\ProgramData\\miniforge3\\envs\\python-dev\\python.exe -I analysis/synthetic/band_top/make_inputs.py --stage calib
"""

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
DATA = HERE.parent.parent.parent / "data" / "synthetic_gprmax_band_top"

C0 = 0.299792458  # m/ns
DX = 0.005
DXF = 0.0025  # 2D grid stage: eps up to 40 needs >= 10 cells per wavelength at 3 f_c
FC = 5e8
P_NS = 1e9 / FC  # carrier period, 2 ns
EPS1 = 9.0
SIGMA = 0.001
OFFSET = 0.1
D_TOP = 0.6

# 2D grid (same conventions as scripts/synthetic M-series)
D2 = {"lx": 3.0, "ly": 1.45, "ys": 1.2, "pml": 20, "tw_ns": 30.0, "xc": 1.5}
# 3D confirmation grid
D3 = {"lx": 1.2, "ly": 1.2, "lz": 1.2, "air": 0.15, "pml": 10, "tw_ns": 24.0}

PIPES = {
    "pec": {"kind": "pec", "od": 0.1},
    "pvc": {"kind": "pvc_air", "od": 0.2, "wall": 0.01, "eps": 3.0},
}
DT_P = [0.5, 1.0, 1.5, 2.0, 3.0]
RATIOS = [0.3, 1.0, 3.0]
EPS2_MAX = 40.0  # wettest realistic soil layer (saturated clay); higher ratios are capped

# Strong-layer set (decision 2026-10-31): tuned wet thin bed and steel protection plate.
BED_EPS = 49.0  # (sigma = soil 0.001) saturated clay lens / perched water; 2.5 mm cells keep >= 10 cells per wavelength at 3 f_c
BED_THICK = 0.0225  # ~ quarter wavelength in the bed at f_c: 0.2998/7/0.5/4 = 0.0214 m -> 9 cells
PLATE_THICK = 0.005  # 2 cells of PEC (steel plate 4-6 mm)
PLATE_WIDTHS = [0.4, 0.2]

# Ringing wavelet (decision 2026-10-31): current I(t) = g'(t)/omega,
# g(t) = e(x) sin(omega (t - ts)), e(x) = x^n exp(n (1 - x)), x = (t - ts)/tp (0 for t < ts).
# n = 2, tp = 2.8 ns: envelope >= 0.3 of max over 2.29 tp = 6.4 ns (3.2 P), >= 0.1 over 3.2 tp = 9.0 ns (4.5 P),
# matching AIHub band boxes of 4-5 carrier periods (box height 15-21 px at P_px ~4.3 px).
RING_FILE = "ring_500MHz.txt"
RING_N = 2
RING_TP_NS = 2.8
RING_TS_NS = 0.5
RING_TW_NS = 40.0


def v_of(eps):
    return C0 / math.sqrt(eps)


def refl(e1, e2):
    return (math.sqrt(e1) - math.sqrt(e2)) / (math.sqrt(e1) + math.sqrt(e2))


def eps2_for_r(abs_r, e1=EPS1):
    """eps below the interface for |r| with eps2 > eps1 (wetter below)."""
    abs_r = min(abs_r, 0.95)
    return e1 * ((1 + abs_r) / (1 - abs_r)) ** 2


def layer_depth(side, dt_p, eps2):
    """Interface depth so that |t_layer - t_top| = dt_p periods (vertical, zero offset)."""
    dt_ns = dt_p * P_NS
    if side == "above":  # pipe sits in eps2; the time between the two reflections is spent in eps2
        return D_TOP - dt_ns * v_of(eps2) / 2.0
    return D_TOP + dt_ns * v_of(EPS1) / 2.0


def two_way_ns(depth, interfaces):
    """Vertical two-way time from the surface to `depth`; interfaces = [(z_depth, eps_below)] sorted."""
    t, z_prev, eps = 0.0, 0.0, EPS1
    for z_i, e_below in interfaces:
        if z_i >= depth:
            break
        t += 2 * (z_i - z_prev) / v_of(eps)
        z_prev, eps = z_i, e_below
    return t + 2 * (depth - z_prev) / v_of(eps)


# ---------------------------------------------------------------- 2D
def in_2d(mid, title, pipe, interfaces, h=0.0, dx=DX, wave="ricker", tw=None, extra=()):
    """extra: additional geometry lines placed after the soil/layer boxes and before the pipe."""
    g = dict(D2, pml=D2["pml"] * round(DX / dx))
    if tw is not None:
        g["tw_ns"] = tw
    ys, xc = g["ys"], g["xc"]
    ya = ys + h
    lines = [f"#title: {mid} {title}",
             "#domain_mode: TM",
             f"#domain: {g['lx']} {g['ly']} inf",
             f"#dx_dy_dz: {dx} {dx} {dx}",
             f"#time_window: {g['tw_ns']:.3f}e-9",
             f"#pml_cells: {g['pml']} {g['pml']} 0 {g['pml']} {g['pml']} 0",
             "",
             f"#material: {EPS1} {SIGMA} 1 0 soil"]
    for i, (_, e2) in enumerate(interfaces):
        lines.append(f"#material: {e2:.4f} {SIGMA} 1 0 soil{i + 2}")
    if pipe and pipe["kind"] == "pvc_air":
        lines.append(f"#material: {pipe['eps']} 0 1 0 pvc")
    wave_line = (f"#waveform: ricker 1 {FC:.0e} src_wave" if wave == "ricker"
                 else f"#excitation_file: {RING_FILE} linear 0")
    src_id = "src_wave" if wave == "ricker" else "ring_wave"
    lines += ["", wave_line,
              f"#hertzian_dipole: z {xc - OFFSET / 2:.3f} {ya:.3f} inf {src_id}",
              f"#rx: {xc + OFFSET / 2:.3f} {ya:.3f} inf rx1 Ez", "",
              f"#box: 0 0 0 {g['lx']} {ys:.3f} inf soil"]
    for i, (z_i, _) in enumerate(interfaces):
        lines.append(f"#box: 0 0 0 {g['lx']} {ys - z_i:.4f} inf soil{i + 2}")
    lines += list(extra)
    if pipe:
        r = pipe["od"] / 2
        yc = ys - D_TOP - r
        if pipe["kind"] == "pec":
            lines.append(f"#cylinder: {xc} {yc:.4f} 0 {xc} {yc:.4f} inf {r:.4f} pec")
        else:
            lines.append(f"#cylinder: {xc} {yc:.4f} 0 {xc} {yc:.4f} inf {r:.4f} pvc")
            lines.append(f"#cylinder: {xc} {yc:.4f} 0 {xc} {yc:.4f} inf {r - pipe['wall']:.4f} free_space")
    return "\n".join(lines) + "\n"


def rec(mid, dim, pipe_key, interfaces, h, bg, homog_bg, role, extra=None):
    r = {"id": mid, "dim": dim, "pipe": pipe_key, "h_m": h,
         "layer_interfaces_m": [z for z, _ in interfaces],
         "layer_eps_r": [EPS1] + [e for _, e in interfaces],
         "bg_id": bg, "homog_bg_id": homog_bg, "role": role,
         "pipe_top_depth_m": D_TOP if pipe_key else None,
         "tau_top_vertical_ns": two_way_ns(D_TOP, interfaces) if pipe_key else None,
         "layer_tau_vertical_ns": [two_way_ns(z, interfaces) for z, _ in interfaces]}
    if pipe_key:
        r["pipe_outer_diameter_m"] = PIPES[pipe_key]["od"]
        # Effective (time-average) velocity from the surface to the pipe top: d = v_eff * tau / 2.
        r["v_eff_to_top_m_per_ns"] = 2 * D_TOP / r["tau_top_vertical_ns"]
        r["eps_eff_to_top"] = (C0 / r["v_eff_to_top_m_per_ns"]) ** 2
    if extra:
        r.update(extra)
    return r


def stage_calib():
    out = []
    jobs = [("K0", "homogeneous background h0", None, [], 0.0, None, None, "homog_bg"),
            ("K0h", "homogeneous background h0.1", None, [], 0.1, None, None, "homog_bg"),
            ("KPEC", "PEC OD0.1 homogeneous h0", "pec", [], 0.0, "K0", "K0", "pipe_homog"),
            ("KPVC", "PVC air OD0.2 homogeneous h0", "pvc", [], 0.0, "K0", "K0", "pipe_homog"),
            ("KPECh", "PEC OD0.1 homogeneous h0.1", "pec", [], 0.1, "K0h", "K0h", "pipe_homog"),
            ("KL", "layer only eps9->12 at 0.6 m h0", None, [(0.6, 12.0)], 0.0, None, "K0", "layer_ref")]
    for mid, title, pk, itf, h, bg, hb, role in jobs:
        (INPUTS / f"{mid}.in").write_text(in_2d(mid, title, PIPES.get(pk), itf, h), encoding="utf-8", newline="\n")
        out.append(rec(mid, 2, pk, itf, h, bg, hb, role))
    return out


def stage_grid(cal):
    """cal: {'A_pipe': {'pec':..,'pvc':..}, 'A_layer_ref': .., 'r_ref': .., 'd_ref': ..} (envelope peaks)."""
    out, design = [], []
    for mid, h in (("K0f", 0.0), ("K0hf", 0.1)):
        (INPUTS / f"{mid}.in").write_text(in_2d(mid, f"homogeneous background h{h} fine grid", None, [], h, dx=DXF),
                                          encoding="utf-8", newline="\n")
        out.append(rec(mid, 2, None, [], h, None, None, "homog_bg"))
    for pk in PIPES:
        for side in ("above", "below"):
            for dtp in DT_P:
                for ratio in RATIOS:
                    # 2D line source: planar reflection amplitude ~ |r| / sqrt(depth) (far-field image source).
                    # First guess with the reference depth, then one fixed-point pass with the actual depth.
                    abs_r = ratio * cal["A_pipe"][pk] / cal["A_layer_ref"] * abs(cal["r_ref"])
                    for _ in range(3):
                        e2 = eps2_for_r(abs_r)
                        z = layer_depth(side, dtp, e2)
                        abs_r = ratio * cal["A_pipe"][pk] / cal["A_layer_ref"] * abs(cal["r_ref"]) * math.sqrt(
                            z / cal["d_ref"])
                    e2 = min(eps2_for_r(abs_r), EPS2_MAX)
                    z = layer_depth(side, dtp, e2)
                    tag = f"{pk.upper()}_{side[0].upper()}{int(dtp * 10):02d}_R{int(ratio * 10):02d}"
                    mid, bid = f"L_{tag}", f"G_{tag}"
                    itf = [(round(z, 4), round(e2, 4))]
                    title = f"{pk} d_top {D_TOP} layer {side} dt {dtp}P ratio {ratio} eps2 {e2:.2f} z {z:.3f}"
                    (INPUTS / f"{mid}.in").write_text(in_2d(mid, title, PIPES[pk], itf, dx=DXF), encoding="utf-8",
                                                      newline="\n")
                    (INPUTS / f"{bid}.in").write_text(in_2d(bid, "background " + title, None, itf, dx=DXF),
                                                      encoding="utf-8", newline="\n")
                    extra = {"layer_side": side, "layer_dt_target_P": dtp, "ratio_target": ratio,
                             "r_design": -abs_r, "eps2_capped": e2 >= EPS2_MAX}
                    out.append(rec(mid, 2, pk, itf, 0.0, bid, "K0f", "grid", extra))
                    out.append(rec(bid, 2, None, itf, 0.0, None, "K0f", "grid_bg", extra))
                    design.append({"id": mid, "pipe": pk, "side": side, "dt_P": dtp, "ratio": ratio,
                                   "eps2": e2, "z": z})
    # Stack: three interfaces at -2P, -1P (above the top) and +1P (below), alternating 9 -> e_a -> 9 -> e_a,
    # each designed for a layer/pipe ratio of about 0.5. The pipe sits in eps 9.
    for pk in PIPES:
        abs_r = 0.5 * cal["A_pipe"][pk] / cal["A_layer_ref"] * abs(cal["r_ref"])
        e_a = min(eps2_for_r(abs_r), EPS2_MAX)
        z2 = D_TOP - 1.0 * P_NS * v_of(EPS1) / 2
        z1 = z2 - 1.0 * P_NS * v_of(e_a) / 2
        z3 = D_TOP + 1.0 * P_NS * v_of(EPS1) / 2
        itf = [(round(z1, 4), round(e_a, 4)), (round(z2, 4), EPS1), (round(z3, 4), round(e_a, 4))]
        mid, bid = f"S_{pk.upper()}", f"GS_{pk.upper()}"
        title = f"{pk} stack -2P -1P +1P"
        (INPUTS / f"{mid}.in").write_text(in_2d(mid, title, PIPES[pk], itf, dx=DXF), encoding="utf-8", newline="\n")
        (INPUTS / f"{bid}.in").write_text(in_2d(bid, "background " + title, None, itf, dx=DXF), encoding="utf-8",
                                          newline="\n")
        out.append(rec(mid, 2, pk, itf, 0.0, bid, "K0f", "stack"))
        out.append(rec(bid, 2, None, itf, 0.0, None, "K0f", "stack_bg"))
    # Air-coupled h = 0.1: PEC, layer 1P above / below, ratio target 1 (eps2 taken from the h0 design).
    for side in ("above", "below"):
        d = next(x for x in design if x["pipe"] == "pec" and x["side"] == side and x["dt_P"] == 1.0
                 and x["ratio"] == 1.0)
        itf = [(round(d["z"], 4), round(d["eps2"], 4))]
        mid, bid = f"H_PEC_{side[0].upper()}10", f"GH_PEC_{side[0].upper()}10"
        title = f"pec h0.1 layer {side} 1P eps2 {d['eps2']:.2f}"
        (INPUTS / f"{mid}.in").write_text(in_2d(mid, title, PIPES["pec"], itf, 0.1, dx=DXF), encoding="utf-8", newline="\n")
        (INPUTS / f"{bid}.in").write_text(in_2d(bid, "background " + title, None, itf, 0.1, dx=DXF), encoding="utf-8",
                                          newline="\n")
        out.append(rec(mid, 2, "pec", itf, 0.1, bid, "K0hf", "air", {"layer_side": side, "layer_dt_target_P": 1.0}))
        out.append(rec(bid, 2, None, itf, 0.1, None, "K0hf", "air_bg"))
    # Same cell as the 3D confirmation (5 mm) to separate grid effects from 2D/3D effects.
    d = next(x for x in design if x["id"] == "L_PEC_A10_R03")
    itf = [(round(d["z"], 4), round(d["eps2"], 4))]
    for mid, pipe, bg in (("C5_PEC_A10_R03", PIPES["pec"], "C5G_PEC_A10_R03"), ("C5G_PEC_A10_R03", None, None)):
        (INPUTS / f"{mid}.in").write_text(in_2d(mid, "5 mm twin of L_PEC_A10_R03", pipe, itf), encoding="utf-8",
                                          newline="\n")
        out.append(rec(mid, 2, "pec" if pipe else None, itf, 0.0, bg, "K0", "grid_c5", {"pair_2d": d["id"]}))
    (HERE / "grid_design.json").write_text(json.dumps(design, indent=1), encoding="utf-8")
    return out


def write_ring_waveform():
    import numpy as np  # local: only this stage needs numpy

    t = np.arange(0.0, RING_TW_NS + 1e-9, 0.001)  # ns, 1 ps
    omega = 2 * math.pi * FC * 1e-9
    x = np.clip((t - RING_TS_NS) / RING_TP_NS, 0.0, None)
    env = np.where(x > 0, x ** RING_N * np.exp(RING_N * (1 - x)), 0.0)
    g = env * np.sin(omega * (t - RING_TS_NS))
    cur = np.gradient(g, t) / omega
    cur /= np.max(np.abs(cur))
    with open(INPUTS / RING_FILE, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("time ring_wave\n")
        for ti, ci in zip(t, cur):
            fh.write(f"{ti * 1e-9:.6e} {ci:.8e}\n")
    above = lambda thr: float(np.ptp(t[env >= thr]))  # noqa: E731
    return {"env_ge_0.3_ns": above(0.3), "env_ge_0.1_ns": above(0.1), "env_peak_ns": RING_TS_NS + RING_TP_NS,
            "net_charge": float(np.trapezoid(cur, t))}


def stage_strong():
    """Strong reflectors: tuned wet thin bed (laterally infinite) and steel protection plate (local)."""
    out = []
    for pk in PIPES:
        for side in ("above", "below"):
            for dtp in DT_P:
                if side == "above":
                    # bed top -> pipe top = 2 T / v_bed + 2 (d_top - z_bot) / v_soil = dtp P
                    gap_ns = dtp * P_NS - 2 * BED_THICK / v_of(BED_EPS)
                    z_bot = min(D_TOP - gap_ns * v_of(EPS1) / 2, D_TOP - 0.005)
                    z_top = z_bot - BED_THICK
                else:
                    z_top = D_TOP + dtp * P_NS * v_of(EPS1) / 2
                    z_bot = z_top + BED_THICK
                itf = [(round(z_top, 4), BED_EPS), (round(z_bot, 4), EPS1)]
                tag = f"{pk.upper()}_{side[0].upper()}{int(dtp * 10):02d}"
                mid, bid = f"W_{tag}", f"GW_{tag}"
                title = f"{pk} tuned wet bed eps {BED_EPS} T {BED_THICK} top {z_top:.4f} {side} {dtp}P"
                for m, p in ((mid, PIPES[pk]), (bid, None)):
                    (INPUTS / f"{m}.in").write_text(in_2d(m, ("" if p else "background ") + title, p, itf, dx=DXF),
                                                    encoding="utf-8", newline="\n")
                extra = {"layer_side": side, "layer_dt_target_P": dtp, "strong_kind": "tuned_wet_bed",
                         "bed_eps_r": BED_EPS, "bed_thickness_m": BED_THICK}
                out.append(rec(mid, 2, pk, itf, 0.0, bid, "K0f", "strong", extra))
                out.append(rec(bid, 2, None, itf, 0.0, None, "K0f", "strong_bg", extra))
    g = D2
    for w in PLATE_WIDTHS:
        for pk in (PIPES if w == PLATE_WIDTHS[0] else ["pec"]):
            for dtp in DT_P:
                z_top = D_TOP - dtp * P_NS * v_of(EPS1) / 2  # plate top; pipe sits in eps 9
                y1 = g["ys"] - z_top
                plate = [f"#box: {g['xc'] - w / 2:.4f} {y1 - PLATE_THICK:.4f} 0 {g['xc'] + w / 2:.4f} {y1:.4f} inf pec"]
                tag = f"{pk.upper()}_W{int(w * 100):02d}_A{int(dtp * 10):02d}"
                mid, bid = f"PL_{tag}", f"GPL_{tag}"
                title = f"{pk} steel protection plate width {w} m top {z_top:.4f} above {dtp}P"
                for m, p in ((mid, PIPES[pk]), (bid, None)):
                    (INPUTS / f"{m}.in").write_text(in_2d(m, ("" if p else "background ") + title, p, [], dx=DXF,
                                                          extra=plate), encoding="utf-8", newline="\n")
                extra = {"layer_side": "above", "layer_dt_target_P": dtp, "strong_kind": "steel_plate",
                         "plate_top_depth_m": round(z_top, 4), "plate_width_m": w, "plate_thickness_m": PLATE_THICK,
                         "plate_note": "PEC strip across the pipe, centred over it, invariant along the pipe/line; "
                                       "laterally local, so masked median does not remove it"}
                r_p = rec(mid, 2, pk, [], 0.0, bid, "K0f", "strong", extra)
                r_p["layer_interfaces_m"] = [round(z_top, 4)]
                r_p["layer_eps_r"] = [EPS1, "pec_plate"]
                r_p["layer_tau_vertical_ns"] = [two_way_ns(z_top, [])]
                out.append(r_p)
                out.append(rec(bid, 2, None, [], 0.0, None, "K0f", "strong_bg", extra))
    return out


def stage_ring():
    info = write_ring_waveform()
    design = {x["id"]: x for x in json.loads((HERE / "grid_design.json").read_text(encoding="utf-8"))}
    out = []
    mid = "K0r"
    (INPUTS / f"{mid}.in").write_text(in_2d(mid, "homogeneous background ringing", None, [], dx=DXF, wave="ring",
                                            tw=RING_TW_NS), encoding="utf-8", newline="\n")
    out.append(rec(mid, 2, None, [], 0.0, None, None, "homog_bg", {"wavelet": "ring"}))
    for pk in PIPES:
        mid = f"RK_{pk.upper()}"
        (INPUTS / f"{mid}.in").write_text(in_2d(mid, f"{pk} homogeneous ringing", PIPES[pk], [], dx=DXF, wave="ring",
                                                tw=RING_TW_NS), encoding="utf-8", newline="\n")
        out.append(rec(mid, 2, pk, [], 0.0, "K0r", "K0r", "ring", {"wavelet": "ring"}))
    cells = [(pk, side, dtp, 1.0) for pk in PIPES for side in ("above", "below") for dtp in DT_P]
    cells += [(pk, side, dtp, 0.3) for pk in PIPES for side in ("above", "below") for dtp in (1.0, 2.0)]
    for pk, side, dtp, ratio in cells:
        src = f"L_{pk.upper()}_{side[0].upper()}{int(dtp * 10):02d}_R{int(ratio * 10):02d}"
        d = design[src]
        itf = [(round(d["z"], 4), round(d["eps2"], 4))]
        mid, bid = "R" + src, "RG" + src[1:]
        title = f"ringing wavelet, geometry of {src}"
        for m, p in ((mid, PIPES[pk]), (bid, None)):
            (INPUTS / f"{m}.in").write_text(in_2d(m, ("" if p else "background ") + title, p, itf, dx=DXF,
                                                  wave="ring", tw=RING_TW_NS), encoding="utf-8", newline="\n")
        extra = {"layer_side": side, "layer_dt_target_P": dtp, "ratio_target": ratio, "wavelet": "ring",
                 "geometry_of": src}
        out.append(rec(mid, 2, pk, itf, 0.0, bid, "K0r", "ring", extra))
        out.append(rec(bid, 2, None, itf, 0.0, None, "K0r", "ring_bg", extra))
    (HERE / "ring_waveform_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info))
    return out


# ---------------------------------------------------------------- 3D
# Fine 3D grid for the eps-49 wet bed (5 mm would give 5.7 cells per wavelength at 3 f_c).
D3F = {"lx": 0.8, "ly": 1.0, "lz": 1.0, "air": 0.15, "pml": 20, "tw_ns": 20.0, "dx": 0.0025}


def in_3d(mid, title, pipe, interfaces, pol, g=None, extra=()):
    """3D model: line and pipe axis along x, ground surface at z = lz - air. extra(zs, yc, g) -> lines."""
    g = dict(D3, dx=DX) if g is None else g
    dx = g["dx"]
    zs = g["lz"] - g["air"]
    yc = g["ly"] / 2
    xm = g["lx"] / 2
    comp = {"x": "Ex", "y": "Ey"}[pol]
    lines = [f"#title: {mid} {title}",
             f"#domain: {g['lx']} {g['ly']} {g['lz']}",
             f"#dx_dy_dz: {dx} {dx} {dx}",
             f"#time_window: {g['tw_ns']:.3f}e-9",
             f"#pml_cells: {g['pml']}",
             "",
             f"#material: {EPS1} {SIGMA} 1 0 soil"]
    for i, (_, e2) in enumerate(interfaces):
        lines.append(f"#material: {e2:.4f} {SIGMA} 1 0 soil{i + 2}")
    if pipe and pipe["kind"] == "pvc_air":
        lines.append(f"#material: {pipe['eps']} 0 1 0 pvc")
    lines += ["", f"#waveform: ricker 1 {FC:.0e} src_wave",
              f"#hertzian_dipole: {pol} {xm - OFFSET / 2:.4f} {yc:.4f} {zs:.4f} src_wave",
              f"#rx: {xm + OFFSET / 2:.4f} {yc:.4f} {zs:.4f} rx1 {comp}", "",
              f"#box: 0 0 0 {g['lx']} {g['ly']} {zs:.4f} soil"]
    for i, (z_i, _) in enumerate(interfaces):
        lines.append(f"#box: 0 0 0 {g['lx']} {g['ly']} {zs - z_i:.4f} soil{i + 2}")
    lines += list(extra(zs, yc, g)) if extra else []
    if pipe:
        r = pipe["od"] / 2
        zc = zs - D_TOP - r
        mat = "pec" if pipe["kind"] == "pec" else "pvc"
        lines.append(f"#cylinder: 0 {yc:.4f} {zc:.4f} {g['lx']} {yc:.4f} {zc:.4f} {r:.4f} {mat}")
        if pipe["kind"] == "pvc_air":
            lines.append(f"#cylinder: 0 {yc:.4f} {zc:.4f} {g['lx']} {yc:.4f} {zc:.4f} {r - pipe['wall']:.4f} free_space")
    return "\n".join(lines) + "\n"


def bed_interfaces(side, dtp):
    """Same tuned-bed geometry as stage_strong."""
    if side == "above":
        gap_ns = dtp * P_NS - 2 * BED_THICK / v_of(BED_EPS)
        z_bot = min(D_TOP - gap_ns * v_of(EPS1) / 2, D_TOP - 0.005)
        z_top = z_bot - BED_THICK
    else:
        z_top = D_TOP + dtp * P_NS * v_of(EPS1) / 2
        z_bot = z_top + BED_THICK
    return [(round(z_top, 4), BED_EPS), (round(z_bot, 4), EPS1)]


def stage_conf3d_strong(args):
    """3D confirmation of the strong-layer set: tuned wet bed (fine grid) and steel plate (5 mm grid, 2D twins).

    Plate geometry comes from arguments so a revised specification only needs a rerun:
      --plate-top-above-m (distance pipe top -> plate top), --plate-widths, --plate-thick, --pipes, --pols.
    """
    out = []
    pols = args.pols
    # (a) Wet bed, PEC, 1 P above (2D: W_PEC_A10), fine grid.
    g = D3F
    itf = bed_interfaces("above", 1.0)
    for pol in ([] if args.no_bed else pols):
        jobs = [(f"W3_PEC_A10_{pol}", PIPES["pec"], f"W3G_A10_{pol}", f"W3G0_{pol}", "conf3d_strong"),
                (f"W3G_A10_{pol}", None, None, f"W3G0_{pol}", "conf3d_strong_bg"),
                (f"W3G0_{pol}", None, None, None, "conf3d_homog_bg")]
        for mid, pipe, bg, hb, role in jobs:
            it = [] if mid.startswith("W3G0") else itf
            (INPUTS / f"{mid}.in").write_text(in_3d(mid, f"3D wet bed eps {BED_EPS} 1P above, dipole {pol}", pipe,
                                                    it, pol, g), encoding="utf-8", newline="\n")
            out.append(rec(mid, 3, "pec" if pipe else None, it, 0.0, bg, hb, role,
                           {"dipole": pol, "pair_2d": "W_PEC_A10", "strong_kind": "tuned_wet_bed",
                            "bed_eps_r": BED_EPS, "bed_thickness_m": BED_THICK, "layer_side": "above",
                            "layer_dt_target_P": 1.0, "grid": "3D 2.5 mm"}))
    # (b) Steel plate: 3D 5 mm (soil only, >= 13 cells per wavelength) + 2D 2.5 mm twins with identical dims.
    # Width: --plate-width-rule fixed (every --plate-widths for every pipe) or spec (A = D + 0.1 m, KGS, see
    # docs/research/buried_pipe_protective_plate.md). Channel section: --plate-flange-m > 0 adds two flanges bent
    # 90 deg downward at both edges (the 90-135 deg range of the code drawing is approximated by 90 deg).
    z_top = round(D_TOP - args.plate_top_above_m, 4)
    th = args.plate_thick
    fl = args.plate_flange_m
    gtag = f"G{round(args.plate_top_above_m * 100):02d}" + (f"C{round(fl * 100):02d}" if fl > 0 else "")
    if args.plate_width_rule in ("spec", "spec200"):
        add = 0.1 if args.plate_width_rule == "spec" else 0.2  # KGS FS451 D+100 / KOGAS-GSM-2121 D+200
        combos = [(round(PIPES[pk]["od"] + add, 4), [pk]) for pk in args.pipes]
    else:
        combos = [(w, list(args.pipes)) for w in args.plate_widths]
    for w, pipe_list in combos:
        wtag = f"W{round(w * 100):02d}"
        meta = {"strong_kind": "steel_plate", "plate_top_depth_m": z_top, "plate_width_m": w,
                "plate_thickness_m": th, "plate_top_above_pipe_top_m": args.plate_top_above_m,
                "plate_flange_m": fl, "plate_section": "channel_90deg_down" if fl > 0 else "flat",
                "layer_side": "above"}

        def plate3(zs, yc, gg, w=w):
            z1 = zs - z_top
            lines = [f"#box: 0 {yc - w / 2:.4f} {z1 - th:.4f} {gg['lx']} {yc + w / 2:.4f} {z1:.4f} pec"]
            if fl > 0:
                for y0 in (yc - w / 2, yc + w / 2 - th):
                    lines.append(f"#box: 0 {y0:.4f} {z1 - fl:.4f} {gg['lx']} {y0 + th:.4f} {z1:.4f} pec")
            return lines

        y1 = D2["ys"] - z_top
        plate2 = [f"#box: {D2['xc'] - w / 2:.4f} {y1 - th:.4f} 0 {D2['xc'] + w / 2:.4f} {y1:.4f} inf pec"]
        if fl > 0:
            for x0 in (D2["xc"] - w / 2, D2["xc"] + w / 2 - th):
                plate2.append(f"#box: {x0:.4f} {y1 - fl:.4f} 0 {x0 + th:.4f} {y1:.4f} inf pec")
        for pol in pols:
            bid = f"P3G_{wtag}_{gtag}_{pol}"
            (INPUTS / f"{bid}.in").write_text(in_3d(bid, f"3D plate only {wtag} {gtag} dipole {pol}", None, [], pol,
                                                    extra=plate3), encoding="utf-8", newline="\n")
            out.append(rec(bid, 3, None, [], 0.0, None, f"TG0_{pol}", "conf3d_strong_bg", dict(meta, dipole=pol)))
        bid2 = f"P2G_{wtag}_{gtag}"
        (INPUTS / f"{bid2}.in").write_text(in_2d(bid2, f"2D plate only {wtag} {gtag}", None, [], dx=DXF,
                                                 extra=plate2), encoding="utf-8", newline="\n")
        out.append(rec(bid2, 2, None, [], 0.0, None, "K0f", "conf3d_strong_bg", meta))
        for pk in pipe_list:
            for pol in pols:
                mid = f"P3_{pk.upper()}_{wtag}_{gtag}_{pol}"
                (INPUTS / f"{mid}.in").write_text(in_3d(mid, f"3D {pk} under plate {wtag} {gtag} dipole {pol}",
                                                        PIPES[pk], [], pol, extra=plate3), encoding="utf-8",
                                                  newline="\n")
                r_p = rec(mid, 3, pk, [], 0.0, f"P3G_{wtag}_{gtag}_{pol}", f"TG0_{pol}", "conf3d_strong",
                          dict(meta, dipole=pol, pair_2d=f"P2_{pk.upper()}_{wtag}_{gtag}",
                               pipe_unshadowed=f"P3K_{pk.upper()}_{pol}"))
                r_p["layer_interfaces_m"], r_p["layer_eps_r"] = [z_top], [EPS1, "pec_plate"]
                r_p["layer_tau_vertical_ns"] = [two_way_ns(z_top, [])]
                out.append(r_p)
            mid2 = f"P2_{pk.upper()}_{wtag}_{gtag}"
            (INPUTS / f"{mid2}.in").write_text(in_2d(mid2, f"2D {pk} under plate {wtag} {gtag}", PIPES[pk], [],
                                                     dx=DXF, extra=plate2), encoding="utf-8", newline="\n")
            r2 = rec(mid2, 2, pk, [], 0.0, bid2, "K0f", "conf3d_strong",
                     dict(meta, pipe_unshadowed=f"P2K_{pk.upper()}"))
            r2["layer_interfaces_m"], r2["layer_eps_r"] = [z_top], [EPS1, "pec_plate"]
            r2["layer_tau_vertical_ns"] = [two_way_ns(z_top, [])]
            out.append(r2)
    # Unshadowed pipe references (same grids): 3D 5 mm and 2D 2.5 mm.
    for pk in args.pipes:
        for pol in pols:
            mid = f"P3K_{pk.upper()}_{pol}"
            (INPUTS / f"{mid}.in").write_text(in_3d(mid, f"3D {pk} homogeneous dipole {pol}", PIPES[pk], [], pol),
                                              encoding="utf-8", newline="\n")
            out.append(rec(mid, 3, pk, [], 0.0, f"TG0_{pol}", f"TG0_{pol}", "pipe_homog", {"dipole": pol}))
        mid = f"P2K_{pk.upper()}"
        (INPUTS / f"{mid}.in").write_text(in_2d(mid, f"2D {pk} homogeneous fine grid", PIPES[pk], [], dx=DXF),
                                          encoding="utf-8", newline="\n")
        out.append(rec(mid, 2, pk, [], 0.0, "K0f", "K0f", "pipe_homog"))
    return out


def stage_conf3d():
    design = json.loads((HERE / "grid_design.json").read_text(encoding="utf-8"))
    d = next(x for x in design if x["id"] == "L_PEC_A10_R03")  # eps2 12.7: 5 mm cells still >= 10 per wavelength
    itf = [(round(d["z"], 4), round(d["eps2"], 4))]
    out = []
    for pol in ("x", "y"):
        jobs = [(f"T_PEC_A10_{pol}", "3D PEC layer above 1P", PIPES["pec"], itf, f"TG_A10_{pol}", f"TG0_{pol}", "conf3d"),
                (f"TG_A10_{pol}", "3D background layer above 1P", None, itf, None, f"TG0_{pol}", "conf3d_bg"),
                (f"TG0_{pol}", "3D homogeneous background", None, [], None, None, "conf3d_homog_bg")]
        for mid, title, pipe, it, bg, hb, role in jobs:
            (INPUTS / f"{mid}.in").write_text(in_3d(mid, f"{title} dipole {pol}", pipe, it, pol), encoding="utf-8",
                                              newline="\n")
            out.append(rec(mid, 3, "pec" if pipe else None, it, 0.0, bg, hb, role,
                           {"dipole": pol, "pair_2d": d["id"], "pair_2d_5mm": "C5_PEC_A10_R03"}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["calib", "grid", "conf3d", "strong", "ring", "conf3d_strong"], required=True)
    ap.add_argument("--plate-top-above-m", type=float, default=0.30,
                    help="conf3d_strong: distance from pipe top up to plate top (m); assumed code value, to be checked")
    ap.add_argument("--plate-widths", type=float, nargs="+", default=[0.4, 0.2])
    ap.add_argument("--plate-thick", type=float, default=0.005, help="plate thickness (m), multiple of 5 mm")
    ap.add_argument("--pipes", nargs="+", default=["pec", "pvc"], choices=list(PIPES))
    ap.add_argument("--pols", nargs="+", default=["x", "y"], choices=["x", "y"])
    ap.add_argument("--plate-width-rule", choices=["fixed", "spec", "spec200"], default="fixed",
                    help="spec: A = OD + 0.1 m (KGS FS451); spec200: A = OD + 0.2 m (KOGAS-GSM-2121); docs/research/buried_pipe_protective_plate.md")
    ap.add_argument("--plate-flange-m", type=float, default=0.0,
                    help="channel section: flange height bent 90 deg down at both edges (m)")
    ap.add_argument("--no-bed", action="store_true", help="conf3d_strong: skip the wet-bed 3D models")
    ap.add_argument("--models-tag", default="",
                    help="write models_<stage>_<tag>.json; run/export with --stage <stage>_<tag>")
    args = ap.parse_args()
    INPUTS.mkdir(parents=True, exist_ok=True)
    if args.stage == "calib":
        recs = stage_calib()
    elif args.stage == "grid":
        cal = json.loads((DATA / "calib_amplitudes.json").read_text(encoding="utf-8"))
        recs = stage_grid(cal)
    elif args.stage == "strong":
        recs = stage_strong()
    elif args.stage == "ring":
        recs = stage_ring()
    elif args.stage == "conf3d_strong":
        recs = stage_conf3d_strong(args)
    else:
        recs = stage_conf3d()
    common = {"dx_m": DX, "fc_hz": FC, "period_ns": P_NS, "eps_soil": EPS1, "sigma_soil": SIGMA,
              "antenna_offset_m": OFFSET, "pipe_top_depth_m": D_TOP, "grid2d": D2, "grid3d": D3, "grid3d_fine": D3F,
              "pipes": PIPES, "wavelet": "gprMax ricker current, Hertzian dipole, peak at sqrt(2)/f"}
    stage_name = args.stage + (f"_{args.models_tag}" if args.models_tag else "")
    (HERE / f"models_{stage_name}.json").write_text(json.dumps({"common": common, "models": recs}, indent=1),
                                                    encoding="utf-8")
    print(json.dumps({"stage": stage_name, "models": len(recs)}))


if __name__ == "__main__":
    main()
