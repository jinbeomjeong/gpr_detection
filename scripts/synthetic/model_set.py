"""Synthetic gprMax (2D TMz) B-scan model set with known pipe depth.

Writes one gprMax input per model (pipe B-scan) and one background input per
model (same model without the pipe, 3 traces) into ``scripts/synthetic/inputs``
and a machine-readable ``model_specs.json`` used by the run and export scripts.

Coordinate convention (gprMax): x horizontal, y vertical (up). The ground
surface is at y = y_surf. Depth below the surface is d = y_surf - y.
Trace position x_m is the midpoint between transmitter and receiver
(common-offset, separation ``antenna_offset_m``).

Run:  conda run -n gprmax-acceleration python -I scripts/synthetic/model_set.py
"""

import json
import math
from pathlib import Path

C0 = 299792458.0  # m/s

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "inputs"

# Common layout (metres)
DOMAIN_X = 3.0
PIPE_X = 1.5
SCAN_START = 0.4
SCAN_STOP = 2.6
TRACE_STEP = 0.02
ANTENNA_OFFSET = 0.10  # Tx-Rx separation; Tx at x_m - s/2, Rx at x_m + s/2
PML_CELLS = 20
BELOW_PIPE_MARGIN = 0.30  # soil between pipe bottom and bottom PML
ABOVE_ANTENNA_MARGIN = 0.15  # air between antenna and top PML
BASE_SIGMA = 0.001  # S/m, baseline soil conductivity
PVC_EPS = 3.0
WATER_EPS = 80.0
WATER_SIGMA = 0.01

# id, pipe kind, outer diameter D, top depth d_top, soil eps_r, f_c, air gap h, soil sigma
# kinds: pec | pvc_air | pvc_water
MODELS = [
    ("M01", "pec", 0.20, 0.30, 9.0, 400e6, 0.0, BASE_SIGMA),
    ("M02", "pec", 0.20, 0.60, 9.0, 400e6, 0.0, BASE_SIGMA),
    ("M03", "pec", 0.20, 1.00, 9.0, 400e6, 0.0, BASE_SIGMA),
    ("M04", "pec", 0.20, 1.50, 9.0, 400e6, 0.0, BASE_SIGMA),
    ("M05", "pec", 0.05, 0.60, 9.0, 400e6, 0.0, BASE_SIGMA),
    ("M06", "pec", 0.40, 0.60, 9.0, 400e6, 0.0, BASE_SIGMA),
    ("M07", "pvc_air", 0.20, 0.60, 9.0, 400e6, 0.0, BASE_SIGMA),
    ("M08", "pvc_water", 0.20, 0.60, 9.0, 400e6, 0.0, BASE_SIGMA),
    ("M09", "pec", 0.20, 0.60, 4.0, 400e6, 0.0, BASE_SIGMA),
    ("M10", "pec", 0.20, 0.60, 16.0, 400e6, 0.0, BASE_SIGMA),
    ("M11", "pec", 0.20, 0.60, 9.0, 500e6, 0.0, BASE_SIGMA),
    ("M12", "pec", 0.20, 0.60, 9.0, 400e6, 0.10, BASE_SIGMA),
    ("M13", "pvc_air", 0.40, 1.00, 4.0, 500e6, 0.10, BASE_SIGMA),
    ("M14", "pvc_water", 0.05, 0.30, 16.0, 500e6, 0.0, BASE_SIGMA),
    ("M15", "pec", 0.40, 1.50, 16.0, 400e6, 0.10, BASE_SIGMA),
    ("M16", "pec", 0.05, 1.00, 4.0, 500e6, 0.0, BASE_SIGMA),
    ("M17", "pvc_air", 0.20, 0.30, 16.0, 400e6, 0.0, BASE_SIGMA),
    ("M18", "pvc_water", 0.40, 1.00, 9.0, 400e6, 0.10, BASE_SIGMA),
    ("M19", "pec", 0.20, 1.00, 16.0, 400e6, 0.0, 0.01),
    ("M20", "pec", 0.20, 0.60, 9.0, 500e6, 0.30, BASE_SIGMA),
]


def cell_size(kind):
    """Spatial step: 5 mm generally, 2.5 mm when water (eps_r 80) is present."""
    return 0.0025 if kind == "pvc_water" else 0.005


def snap(value, dx):
    return round(round(value / dx) * dx, 6)


def wall_thickness(diameter):
    return 0.01 if diameter >= 0.2 else 0.005


def ricker_peak_time(freq):
    """gprMax ricker: w(t) = -(2 zeta tau^2 - 1) exp(-zeta tau^2), tau = t - chi, chi = sqrt(2)/f."""
    return math.sqrt(2.0) / freq


def build_spec(row):
    mid, kind, diameter, d_top, eps, freq, air, sigma = row
    dx = cell_size(kind)
    pml = PML_CELLS * dx
    radius = diameter / 2.0
    v = C0 / math.sqrt(eps)
    y_surf = snap(pml + d_top + diameter + BELOW_PIPE_MARGIN, dx)
    y_ant = snap(y_surf + air, dx)
    domain_y = snap(y_ant + ABOVE_ANTENNA_MARGIN + pml, dx)
    y_center = y_surf - d_top - radius
    n_traces = int(round((SCAN_STOP - SCAN_START) / TRACE_STEP)) + 1

    # Time window: farthest-trace top echo, apex echo from pipe bottom (penetrable pipes),
    # plus source delay, wavelet duration and margin.
    half_aperture = max(PIPE_X - SCAN_START, SCAN_STOP - PIPE_X) + ANTENNA_OFFSET / 2
    t_far_top = 2 * air / C0 + 2 * (math.hypot(d_top + radius, half_aperture) - radius) / v
    t_apex_bottom = 2 * air / C0 + 2 * d_top / v
    if kind == "pvc_air":
        t_apex_bottom += 2 * diameter / C0
    elif kind == "pvc_water":
        t_apex_bottom += 2 * diameter * math.sqrt(WATER_EPS) / C0
    t_needed = max(t_far_top, t_apex_bottom)
    time_window = ricker_peak_time(freq) + t_needed + 2.0 / freq + 3e-9
    time_window = math.ceil(time_window * 1e9) * 1e-9

    return {
        "model_id": mid,
        "pipe_kind": kind,
        "pipe_outer_diameter_m": diameter,
        "pipe_radius_m": radius,
        "pipe_wall_m": None if kind == "pec" else wall_thickness(diameter),
        "pipe_top_depth_m": d_top,
        "pipe_center_depth_m": d_top + radius,
        "pipe_center_x_m": PIPE_X,
        "eps_r": eps,
        "soil_sigma_S_per_m": sigma,
        "velocity_m_per_ns": v * 1e-9,
        "center_freq_hz": freq,
        "air_gap_m": air,
        "antenna_offset_m": ANTENNA_OFFSET,
        "dx_cell_m": dx,
        "pml_cells": PML_CELLS,
        "domain_x_m": DOMAIN_X,
        "domain_y_m": domain_y,
        "y_surface_m": y_surf,
        "y_antenna_m": y_ant,
        "y_pipe_center_m": round(y_center, 6),
        "scan_start_m": SCAN_START,
        "trace_step_m": TRACE_STEP,
        "n_traces": n_traces,
        "time_window_ns": round(time_window * 1e9, 3),
        "source_peak_time_ns": ricker_peak_time(freq) * 1e9,
    }


def input_text(spec, with_pipe, bg_positions=None, with_soil=True):
    dx = spec["dx_cell_m"]
    eps = spec["eps_r"]
    label = "pipe" if with_pipe else ("background" if with_soil else "free-space reference")
    lines = [
        f"#title: {spec['model_id']} {label} "
        f"{spec['pipe_kind']} D={spec['pipe_outer_diameter_m']} d_top={spec['pipe_top_depth_m']} "
        f"eps={eps} f={spec['center_freq_hz'] / 1e6:g}MHz h={spec['air_gap_m']}",
        "#domain_mode: TM",
        f"#domain: {spec['domain_x_m']} {spec['domain_y_m']} inf",
        f"#dx_dy_dz: {dx} {dx} {dx}",
        f"#time_window: {spec['time_window_ns']:.3f}e-9",
        # 2D: no PML along the invariant z axis (x0 y0 z0 xmax ymax zmax).
        f"#pml_cells: {spec['pml_cells']} {spec['pml_cells']} 0 {spec['pml_cells']} {spec['pml_cells']} 0",
        "",
        f"#material: {eps} {spec['soil_sigma_S_per_m']} 1 0 soil",
    ]
    # Declare only materials present in the model so gprMax's resolution check reports the real limit.
    if with_pipe and spec["pipe_kind"] != "pec":
        lines.append(f"#material: {PVC_EPS} 0 1 0 pvc")
    if with_pipe and spec["pipe_kind"] == "pvc_water":
        lines.append(f"#material: {WATER_EPS} {WATER_SIGMA} 1 0 water")
    lines += [
        "",
        f"#waveform: ricker 1 {spec['center_freq_hz']:g} src_wave",
    ]
    half = spec["antenna_offset_m"] / 2
    if with_pipe:
        x_first = spec["scan_start_m"]
        step = spec["trace_step_m"]
    else:
        x_first = bg_positions[0]
        step = bg_positions[1] - bg_positions[0]
    tx = snap(x_first - half, dx)
    rx = snap(x_first + half, dx)
    ya = spec["y_antenna_m"]
    lines += [
        f"#hertzian_dipole: z {tx} {ya} inf src_wave",
        f"#rx: {rx} {ya} inf rx1 Ez",
        f"#src_steps: {step} 0 0",
        f"#rx_steps: {step} 0 0",
        "",
    ]
    if with_soil:
        lines.append(f"#box: 0 0 0 {spec['domain_x_m']} {spec['y_surface_m']} inf soil")
    if with_pipe:
        xc = spec["pipe_center_x_m"]
        yc = spec["y_pipe_center_m"]
        r = spec["pipe_radius_m"]
        kind = spec["pipe_kind"]
        if kind == "pec":
            lines.append(f"#cylinder: {xc} {yc} 0 {xc} {yc} inf {r} pec")
        else:
            fill = "free_space" if kind == "pvc_air" else "water"
            r_in = round(r - spec["pipe_wall_m"], 6)
            lines.append(f"#cylinder: {xc} {yc} 0 {xc} {yc} inf {r} pvc")
            lines.append(f"#cylinder: {xc} {yc} 0 {xc} {yc} inf {r_in} {fill}")
    return "\n".join(lines) + "\n"


BG_POSITIONS = [SCAN_START, PIPE_X, SCAN_STOP]  # equally spaced (1.1 m)


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = []
    for row in MODELS:
        spec = build_spec(row)
        (INPUT_DIR / f"{spec['model_id']}.in").write_text(input_text(spec, True), encoding="utf-8", newline="\n")
        (INPUT_DIR / f"{spec['model_id']}_bg.in").write_text(
            input_text(spec, False, BG_POSITIONS), encoding="utf-8", newline="\n"
        )
        spec["bg_trace_positions_m"] = BG_POSITIONS
        # Free-space reference (no soil, 1 trace at the pipe x) for air-gap models: background minus this
        # isolates the ground-surface reflection, giving a measured surface-reflection time-zero.
        spec["fs_reference"] = spec["air_gap_m"] > 0
        if spec["fs_reference"]:
            (INPUT_DIR / f"{spec['model_id']}_fs.in").write_text(
                input_text(spec, False, [PIPE_X, SCAN_STOP], with_soil=False), encoding="utf-8", newline="\n"
            )
        specs.append(spec)
        cells = round(spec["domain_x_m"] / spec["dx_cell_m"]) * round(spec["domain_y_m"] / spec["dx_cell_m"])
        dt = 1.0 / (C0 * math.sqrt(2.0) / spec["dx_cell_m"])
        iters = math.ceil(spec["time_window_ns"] * 1e-9 / dt) + 1
        spec["est_cells"] = cells
        spec["est_dt_ps_cfl"] = dt * 1e12
        spec["est_iterations"] = iters
        print(
            f"{spec['model_id']} {spec['pipe_kind']:9s} dx={spec['dx_cell_m']} Y={spec['domain_y_m']:.3f} "
            f"tw={spec['time_window_ns']:.0f}ns cells={cells} iters={iters} "
            f"updates/trace={cells * iters:.3e}"
        )
    (HERE / "model_specs.json").write_text(json.dumps(specs, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
