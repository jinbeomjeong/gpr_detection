"""Export synthetic gprMax B-scans to npy + json ground truth and run timing checks.

Two model sets are supported.

M01-M20 (default, one pipe each), described by ``scripts/synthetic/model_specs.json``:

* inputs: ``data/synthetic_gprmax/_raw/<id>/<id>_merged.h5``, ``<id>_bg_merged.h5``,
  optional ``<id>_fs.h5``
* ``data/synthetic_gprmax/<id>.npy``: float32 [samples, traces], raw Ez (no processing)
* ``<id>_bg.npy``: float32 [samples, 3], same model without pipe at x = 0.4/1.5/2.6 m
* ``<id>_fs.npy``: float32 [samples, 1], antennas in free space (air-gap models only)
* ``<id>.json``: ground truth, time-zero candidates, measured apex timing, checks
* ``manifest.csv``

S1g (M2-10, scan lines with 0, 1 or 2 pipes; S1G_BGL is the no-pipe background
line), described by ``analysis/synthetic/s1g/models_s1g.json``:

* inputs: ``data/synthetic_gprmax_s1g/h5/<id>_merged.h5`` and the shared
  ``S1G_bg_merged.h5`` / ``S1G_fs.h5``
* outputs in ``data/synthetic_gprmax_s1g``: the same file set as above
* the json keeps every single-pipe key (they describe ``pipes[0]``, the pipe with the
  smallest x) and adds ``n_pipes`` and ``pipes`` (per pipe: ray times, apex trace and
  timing, top depth, picks); ``manifest.csv`` gains ``n_pipes`` and joins per-pipe
  values with ';'.

Specs without a ``pipes`` list are exported exactly as before (same bytes, no
``pipes`` key).

S4/S6 (M2-10 (b), scan lines with 1 pipe or none, shallow clutter ``none``,
``paving`` or ``stones``), described by ``analysis/synthetic/s4s6/models_s4s6.json``:

* inputs: ``data/synthetic_gprmax_s4s6/h5/<id>_merged.h5``; the 3-trace
  background of the same soil sigma (``BG_PF_S<sigma>`` under paving,
  ``BG_N_S<sigma>`` otherwise, ``<ref>_merged.h5``) and the shared ``FS.h5``
* outputs in ``data/synthetic_gprmax_s4s6``: the same file set and json schema
  as S1g (``n_pipes`` 1 or 0), plus ``expected_apex_envelope_peak_ns_approx``,
  ``apex_trace_index`` (both null without a pipe) and ``s4s6`` (family, batch,
  role, clutter and its ``paving`` / ``stones`` block, background reference)

Run:
    conda run -n gprmax-acceleration python -I scripts/synthetic/export_dataset.py
    conda run -n gprmax-acceleration python -I scripts/synthetic/export_dataset.py ^
        --s1g analysis/synthetic/s1g/models_s1g.json [--models S1G_D060 ...]
    conda run -n gprmax-acceleration python -I scripts/synthetic/export_dataset.py ^
        --s4s6 analysis/synthetic/s4s6/models_s4s6.json [--models S4N_D15_S010 ...]
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import hilbert

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ray_geometry import (  # noqa: E402
    C0_M_PER_NS,
    direct_air_time,
    surface_reflection_time,
    trace_reflection_times,
)

PROJECT = HERE.parent.parent
OUT = PROJECT / "data" / "synthetic_gprmax"
RAW = OUT / "_raw"
S1G_OUT = PROJECT / "data" / "synthetic_gprmax_s1g"
S4S6_OUT = PROJECT / "data" / "synthetic_gprmax_s4s6"

# Search windows, in periods of the centre frequency (1/f).
APEX_WINDOW_PERIODS = 0.6
PICK_WINDOW_PERIODS = 0.35
# An echo is taken to start this long before its envelope peak (pre-arrival check).
PRE_ARRIVAL_MARGIN_PERIODS = 1.2
# Per-trace velocity picks: only within this lateral distance of the pipe, and only
# when the envelope reaches this fraction of the apex envelope.
PICK_MAX_OFFSET_M = 1.0
PICK_MIN_ENVELOPE_RATIO = 0.2
# Multi-pipe models: a trace is skipped for a pipe's picks when another pipe's echo
# peak is expected closer than this (pick window + echo lead). With the apex window
# (0.6 periods) instead of the echo lead, fake two-pipe data at 500 MHz still showed
# envelope picks pulled by up to 0.07 ns near the midpoint. For close pipes this can
# drop the apex trace itself from the picks; the apex timing is measured separately.
NEIGHBOUR_SEPARATION_PERIODS = PICK_WINDOW_PERIODS + PRE_ARRIVAL_MARGIN_PERIODS
ONSET_FRACTION = 0.05

# S1G_bg.in: Tx at 0.35 m, step 1.6 m, offset 0.10 m -> midpoints 0.4 / 2.0 / 3.6 m.
S1G_BG_TRACE_X_M = [0.4, 2.0, 3.6]
# models_s4s6.json references BG_*: "3 traces x 0.4/2.0/3.6".
S4S6_BG_TRACE_X_M = [0.4, 2.0, 3.6]
S4S6_FREE_SPACE_ID = "FS"
S4S6_BACKGROUND_PREFIX = {"none": "BG_N", "stones": "BG_N", "paving": "BG_PF"}
"""Background reference family per clutter (flat paving without joints for S6P)."""
S4S6_CLUTTER_BLOCKS = ("paving", "stones")

MANIFEST_FIELDS = [
    "model_id",
    "npy",
    "json",
    "pipe_material",
    "pipe_outer_diameter_m",
    "pipe_radius_m",
    "pipe_top_depth_m",
    "pipe_center_depth_m",
    "pipe_center_x_m",
    "eps_r",
    "soil_sigma_S_per_m",
    "velocity_m_per_ns",
    "center_freq_mhz",
    "air_gap_m",
    "antenna_offset_m",
    "n_samples",
    "n_traces",
    "dt_ns",
    "dx_m",
    "time_window_ns",
    "fdtd_cell_m",
    "t0_top_ns",
    "t0_center_ns",
    "tz_source_ns",
    "tz_direct_ns",
    "tz_surface_ns",
    "apex_envelope_peak_ns",
    "wavelet_lag_ns",
    "apparent_v_ratio_pick_slope",
]
MULTI_PIPE_MANIFEST_FIELDS = [*MANIFEST_FIELDS, "n_pipes"]
PIPE_LIST_SEPARATOR = ";"

PIPE_KEYS_NOTE = (
    "Top-level pipe_* keys, ray_reflection_time_ns, apex, t0_effective_ns, "
    "depth_from_t0_effective_m, velocity_check and trace_envelope_pick_ns describe "
    "pipes[0] (smallest pipe_center_x_m) so single-pipe readers keep working; use "
    "pipes for every pipe. Each pipe's ray times treat that pipe as the only "
    "scatterer. Velocity picks of a pipe use only traces nearer to it than to any "
    "other pipe and where every other pipe's expected echo peak is at least "
    "(0.35 + 1.2)/f away in time. checks are relative to the smallest apex "
    "envelope over the pipes."
)
BACKGROUND_LINE_NOTE = (
    "Background line (no pipe, n_pipes 0): every pipe key is null. checks are "
    "relative to the envelope peak of the middle no-pipe trace (time-zero band); "
    "pre_arrival_residual_rel is null and line_residual_max_rel is max|line - "
    "middle no-pipe trace| over the whole line."
)


@dataclass(frozen=True)
class Pipe:
    """One buried pipe of a model."""

    center_x_m: float
    top_depth_m: float
    radius_m: float


@dataclass(frozen=True)
class SourceFiles:
    """gprMax output files of one model."""

    bscan_h5: Path
    background_h5: Path
    free_space_h5: Path | None


@dataclass(frozen=True)
class ScanGeometry:
    """Time axis, trace positions and propagation constants shared by every pipe."""

    t_ns: np.ndarray
    dt_ns: float
    x_mid_m: np.ndarray
    offset_m: float
    air_gap_m: float
    velocity_m_per_ns: float
    f_ghz: float
    source_peak_ns: float


def snap(value, dx):
    return round(round(value / dx) * dx, 6)


def read_ez(path):
    with h5py.File(path, "r") as f:
        ez = np.asarray(f["rxs/rx1/Ez"])
        dt = float(f.attrs["dt"])
        iterations = int(f.attrs["Iterations"])
    if ez.ndim == 1:
        ez = ez[:, None]
    return ez, dt, iterations


def refine_peak(env, k):
    """Parabolic sub-sample refinement of a local maximum at index k (fractional)."""
    if 0 < k < len(env) - 1:
        a, b, c = env[k - 1], env[k], env[k + 1]
        den = a - 2 * b + c
        if den < 0:
            return k + 0.5 * (a - c) / den
    return float(k)


def envelope_pick(trace, t, t_lo, t_hi):
    """Envelope maximum of a trace in [t_lo, t_hi] (ns).

    Returns (time, envelope value, index, at_edge).
    """
    env = np.abs(hilbert(trace))
    sel = np.where((t >= t_lo) & (t <= t_hi))[0]
    k = int(sel[np.argmax(env[sel])])
    kf = refine_peak(env, k)
    dt = t[1] - t[0]
    at_edge = k in (sel[0], sel[-1])
    return float(t[0] + kf * dt), float(env[k]), k, at_edge


def pipes_from_spec(spec: dict) -> list[Pipe]:
    """Pipes of a model sorted by x: its ``pipes`` list (may be empty for a background
    line) or its single-pipe keys."""
    if "pipes" not in spec:
        return [
            Pipe(
                spec["pipe_center_x_m"], spec["pipe_top_depth_m"], spec["pipe_radius_m"]
            )
        ]
    pipes = [
        Pipe(p["center_x_m"], p["top_depth_m"], p["radius_m"]) for p in spec["pipes"]
    ]
    return sorted(pipes, key=lambda pipe: pipe.center_x_m)


def legacy_source_files(spec: dict, raw_dir: Path | None = None) -> SourceFiles:
    """File layout of the M01-M20 set: ``<raw_dir>/<id>/<id>_merged.h5`` etc."""
    model_id = spec["model_id"]
    model_dir = (RAW if raw_dir is None else raw_dir) / model_id
    fs_path = model_dir / f"{model_id}_fs.h5"
    has_free_space = bool(spec.get("fs_reference")) and fs_path.exists()
    return SourceFiles(
        bscan_h5=model_dir / f"{model_id}_merged.h5",
        background_h5=model_dir / f"{model_id}_bg_merged.h5",
        free_space_h5=fs_path if has_free_space else None,
    )


def _s1g_spec(common: dict, model: dict) -> dict:
    pipe_xs = model["pipe_center_x_m"]
    has_pipes = bool(pipe_xs)
    return {
        "model_id": model["id"],
        # A background line (no pipe) gets null pipe keys, not the common pipe values.
        "pipe_kind": common["pipe_material"] if has_pipes else None,
        "pipe_wall_m": common.get("pipe_wall_m"),
        "pipe_radius_m": common["pipe_radius_m"] if has_pipes else None,
        "pipe_top_depth_m": common["pipe_top_depth_m"] if has_pipes else None,
        "pipe_center_x_m": pipe_xs[0] if has_pipes else None,
        "pipes": [
            {
                "center_x_m": x,
                "top_depth_m": common["pipe_top_depth_m"],
                "radius_m": common["pipe_radius_m"],
            }
            for x in pipe_xs
        ],
        "expected_apex_trace_index": model.get("apex_trace_index"),
        "eps_r": common["eps_r"],
        "soil_sigma_S_per_m": common["soil_sigma_S_per_m"],
        "velocity_m_per_ns": common["velocity_m_per_ns"],
        "center_freq_hz": common["center_freq_hz"],
        "antenna_offset_m": common["antenna_offset_m"],
        "dx_cell_m": common["fdtd_cell_m"],
        "pml_cells": common["pml_cells"],
        "y_surface_m": common["y_surface_m"],
        "y_antenna_m": common["y_antenna_m"],
        "time_window_ns": common["time_window_ns"],
        "scan_start_m": model["scan_start_m"],
        "trace_step_m": model["trace_step_m"],
        "n_traces": model["n_traces"],
        "bg_trace_positions_m": S1G_BG_TRACE_X_M,
        "fs_reference": True,
        "gprmax_input": model["input"],
    }


def load_s1g_specs(
    models_json: Path, h5_dir: Path | None = None
) -> list[tuple[dict, SourceFiles]]:
    """Specs and source files of the S1g scan lines (with 0, 1 or 2 pipes).

    Scan lines are the entries with ``scan_start_m``; the 3-trace background
    (``*_bg``) and the free-space reference (``*_fs``) are shared inputs.

    Args:
        models_json: ``analysis/synthetic/s1g/models_s1g.json``.
        h5_dir: Folder of the gprMax outputs; default is the folder named by each
            model's ``output``.

    Raises:
        ValueError: If the json has no background or free-space entry.
    """
    meta = json.loads(models_json.read_text(encoding="utf-8"))
    line_models = [m for m in meta["models"] if "scan_start_m" in m]
    shared = {m["id"]: m["output"] for m in meta["models"] if "scan_start_m" not in m}
    background = next((out for mid, out in shared.items() if mid.endswith("_bg")), None)
    free_space = next((out for mid, out in shared.items() if mid.endswith("_fs")), None)
    if background is None or free_space is None:
        raise ValueError(
            f"{models_json}: background (_bg) and free-space (_fs) entries are required"
        )

    def locate(relative_output: str) -> Path:
        path = PROJECT / relative_output
        return path if h5_dir is None else h5_dir / path.name

    fs_path = locate(free_space)
    specs = []
    for model in line_models:
        sources = SourceFiles(
            bscan_h5=locate(model["output"]),
            background_h5=locate(background),
            free_space_h5=fs_path if fs_path.exists() else None,
        )
        specs.append((_s1g_spec(meta["common"], model), sources))
    return specs


def _sigma_tag(sigma_s_per_m: float) -> str:
    """``S010`` for 0.01 S/m: the soil tag of the S4/S6 model and reference ids."""
    return f"S{round(sigma_s_per_m * 1000):03d}"


def s4s6_background_id(model: dict) -> str:
    """3-trace background reference of an S4/S6 model, e.g. ``BG_PF_S010``.

    Raises:
        ValueError: An unknown clutter, or a model id whose soil tag does not
            match its ``soil_sigma_S_per_m``.
    """
    clutter = model["clutter"]
    if clutter not in S4S6_BACKGROUND_PREFIX:
        raise ValueError(f"{model['id']}: unknown clutter {clutter!r}")
    tag = _sigma_tag(model["soil_sigma_S_per_m"])
    if not model["id"].endswith(f"_{tag}"):
        raise ValueError(f"{model['id']}: soil tag differs from sigma tag {tag}")
    return f"{S4S6_BACKGROUND_PREFIX[clutter]}_{tag}"


def _s4s6_extra_meta(meta: dict, model: dict, background_id: str) -> dict:
    """Json keys added after the S1g schema (harness inputs and the clutter)."""
    has_pipe = model.get("pipe_material") is not None
    clutter = model["clutter"]
    return {
        "expected_apex_envelope_peak_ns_approx": (
            model["expected_apex_envelope_peak_ns_approx"] if has_pipe else None
        ),
        "apex_trace_index": model["apex_trace_index"] if has_pipe else None,
        "s4s6": {
            "family": model["family"],
            "batch": model["batch"],
            "role": model["role"],
            "clutter": clutter,
            "clutter_meta": meta[clutter] if clutter in S4S6_CLUTTER_BLOCKS else None,
            "background_reference": background_id,
            "two_way_vertical_ns_surface_to_top": model.get(
                "two_way_vertical_ns_surface_to_top"
            ),
            "soil_two_way_attenuation_db_approx": model.get(
                "soil_two_way_attenuation_db_approx"
            ),
        },
    }


def _s4s6_spec(meta: dict, model: dict, background_id: str) -> dict:
    common = meta["common"]
    has_pipe = model.get("pipe_material") is not None
    eps_r = common["eps_r_soil"]
    pipes = []
    if has_pipe:
        pipes = [
            {
                "center_x_m": model["pipe_center_x_m"],
                "top_depth_m": model["pipe_top_depth_m"],
                "radius_m": model["pipe_radius_m"],
            }
        ]
    return {
        "model_id": model["id"],
        "pipe_kind": model["pipe_material"] if has_pipe else None,
        "pipe_wall_m": None,
        "pipe_radius_m": model["pipe_radius_m"] if has_pipe else None,
        "pipe_top_depth_m": model["pipe_top_depth_m"] if has_pipe else None,
        "pipe_center_x_m": model["pipe_center_x_m"] if has_pipe else None,
        "pipes": pipes,
        "expected_apex_trace_index": [model["apex_trace_index"]] if has_pipe else [],
        "eps_r": eps_r,
        "soil_sigma_S_per_m": model["soil_sigma_S_per_m"],
        # Soil velocity; the paving layer of S6 is not in the ray times (it shifts
        # the S6P apex about 0.7 ns earlier, inside the apex window).
        "velocity_m_per_ns": C0_M_PER_NS / math.sqrt(eps_r),
        "center_freq_hz": common["center_freq_hz"],
        "antenna_offset_m": common["antenna_offset_m"],
        "dx_cell_m": common["fdtd_cell_m"],
        "pml_cells": common["pml_cells"],
        "y_surface_m": common["y_surface_m"],
        "y_antenna_m": common["y_antenna_m"],
        "time_window_ns": common["time_window_ns"],
        "scan_start_m": model["scan_start_m"],
        "trace_step_m": model["trace_step_m"],
        "n_traces": model["n_traces"],
        "bg_trace_positions_m": S4S6_BG_TRACE_X_M,
        "fs_reference": True,
        "gprmax_input": model["input"],
        "extra_meta": _s4s6_extra_meta(meta, model, background_id),
    }


def load_s4s6_specs(
    models_json: Path, h5_dir: Path | None = None
) -> list[tuple[dict, SourceFiles]]:
    """Specs and source files of the S4/S6 scan lines (1 pipe or none).

    Args:
        models_json: ``analysis/synthetic/s4s6/models_s4s6.json``.
        h5_dir: Folder of the gprMax outputs (models, ``BG_*`` references and
            ``FS``); default is the folder of each model's ``output``.

    Raises:
        ValueError: If a model's background reference is not listed in
            ``references`` (see ``s4s6_background_id``).
    """
    meta = json.loads(models_json.read_text(encoding="utf-8"))
    reference_traces = {ref["id"]: ref["n_traces"] for ref in meta["references"]}

    def reference_h5(folder: Path, reference_id: str) -> Path:
        # run_s4s6.cmd: <name>_merged.h5 for n > 1 traces, <name>.h5 for one.
        is_merged = reference_traces[reference_id] > 1
        name = f"{reference_id}_merged.h5" if is_merged else f"{reference_id}.h5"
        return folder / name

    specs = []
    for model in meta["models"]:
        background_id = s4s6_background_id(model)
        if background_id not in reference_traces:
            raise ValueError(f"{model['id']}: reference {background_id} not listed")
        bscan_h5 = PROJECT / model["output"]
        folder = bscan_h5.parent if h5_dir is None else h5_dir
        fs_path = reference_h5(folder, S4S6_FREE_SPACE_ID)
        sources = SourceFiles(
            bscan_h5=folder / bscan_h5.name,
            background_h5=reference_h5(folder, background_id),
            free_space_h5=fs_path if fs_path.exists() else None,
        )
        specs.append((_s4s6_spec(meta, model, background_id), sources))
    return specs


def _read_model_arrays(spec, sources):
    """B-scan, background and optional free-space traces on a common time axis."""
    mid = spec["model_id"]
    data, dt_s, iters = read_ez(sources.bscan_h5)
    bg, dt_bg, iters_bg = read_ez(sources.background_h5)
    if not (math.isclose(dt_s, dt_bg, rel_tol=0, abs_tol=0) and iters == iters_bg):
        raise ValueError(
            f"{mid}: dt/Iterations differ between pipe and background runs"
        )
    if data.shape != (iters, spec["n_traces"]) or bg.shape != (iters, 3):
        raise ValueError(f"{mid}: unexpected shapes {data.shape} {bg.shape}")
    fs = None
    if sources.free_space_h5 is not None:
        fs, dt_fs, iters_fs = read_ez(sources.free_space_h5)
        if dt_fs != dt_s or iters_fs != iters:
            raise ValueError(f"{mid}: free-space reference dt/Iterations differ")
    return data, bg, fs, dt_s, iters


def _scan_geometry(spec, dt_s, iters):
    dt_ns = dt_s * 1e9
    dx = spec["dx_cell_m"]
    s = spec["antenna_offset_m"]
    n = spec["n_traces"]
    tx = snap(spec["scan_start_m"] - s / 2, dx) + np.arange(n) * spec["trace_step_m"]
    rx = snap(spec["scan_start_m"] + s / 2, dx) + np.arange(n) * spec["trace_step_m"]
    f_ghz = spec["center_freq_hz"] * 1e-9
    return ScanGeometry(
        # gprMax Ez receiver sample k is at k*dt (E-field offset 0).
        t_ns=np.arange(iters) * dt_ns,
        dt_ns=dt_ns,
        x_mid_m=0.5 * (tx + rx),
        offset_m=round(float(rx[0] - tx[0]), 6),
        air_gap_m=round(spec["y_antenna_m"] - spec["y_surface_m"], 6),
        velocity_m_per_ns=spec["velocity_m_per_ns"],
        f_ghz=f_ghz,
        source_peak_ns=math.sqrt(2.0) / f_ghz,  # Ricker current peak (ns)
    )


def _pipe_ray_times(pipe, geometry):
    return trace_reflection_times(
        geometry.x_mid_m,
        geometry.offset_m,
        geometry.air_gap_m,
        pipe.center_x_m,
        pipe.top_depth_m,
        pipe.radius_m,
        geometry.velocity_m_per_ns,
    )


def _measure_apex(residual, geometry, x0, t_ray):
    """Envelope, signed and onset timing of the echo at the trace nearest the pipe."""
    t, f_ghz = geometry.t_ns, geometry.f_ghz
    ia = int(np.argmin(np.abs(geometry.x_mid_m - x0)))
    t_expect = geometry.source_peak_ns + t_ray[ia]
    half_window = APEX_WINDOW_PERIODS / f_ghz
    t_apex, env_apex, k_apex, edge = envelope_pick(
        residual[:, ia], t, t_expect - half_window, t_expect + half_window
    )
    idx = np.where((t >= t_expect - half_window) & (t <= t_expect + half_window))[0]
    k_signed = int(idx[np.argmax(np.abs(residual[idx, ia]))])
    t_apex_signed = float(
        t[0] + refine_peak(np.abs(residual[:, ia]), k_signed) * geometry.dt_ns
    )
    # Onset: first sample before the envelope peak where |residual| exceeds 5 % of it.
    seg = np.abs(residual[: k_apex + 1, ia])
    above = np.where(seg > ONSET_FRACTION * env_apex)[0]
    return {
        "trace_index": ia,
        "x_m": float(geometry.x_mid_m[ia]),
        "ray_time_ns": float(t_ray[ia]),
        "expected_envelope_peak_ns": t_expect,
        "envelope_peak_ns": t_apex,
        "envelope_peak_at_window_edge": bool(edge),
        "signed_peak_ns": t_apex_signed,
        "signed_peak_polarity": int(np.sign(residual[k_signed, ia])),
        "onset_5pct_ns": float(t[above[0]]) if above.size else None,
        "wavelet_lag_ns": t_apex - t_expect,
        "envelope_peak_value": env_apex,
    }


def _trace_is_free_of_neighbours(i, pipe, t_ray, neighbours, geometry):
    """True when trace i is nearer to this pipe than to any neighbour pipe and no
    neighbour echo peak is expected within ``NEIGHBOUR_SEPARATION_PERIODS``."""
    x = geometry.x_mid_m[i]
    separation_ns = NEIGHBOUR_SEPARATION_PERIODS / geometry.f_ghz
    for other, other_ray in neighbours:
        if abs(x - other.center_x_m) <= abs(x - pipe.center_x_m):
            return False
        if abs(other_ray[i] - t_ray[i]) < separation_ns:
            return False
    return True


def _velocity_picks(residual, geometry, pipe, t_ray, apex, neighbours):
    """Per-trace envelope picks (residual vs ray time is flat if v is right)."""
    t, f_ghz = geometry.t_ns, geometry.f_ghz
    x0 = pipe.center_x_m
    half_window = PICK_WINDOW_PERIODS / f_ghz
    picks = [None] * len(t_ray)
    rows = []
    for i in range(len(t_ray)):
        if abs(geometry.x_mid_m[i] - x0) > PICK_MAX_OFFSET_M:
            continue
        if not _trace_is_free_of_neighbours(i, pipe, t_ray, neighbours, geometry):
            continue
        t_expected = geometry.source_peak_ns + apex["wavelet_lag_ns"] + t_ray[i]
        if t_expected + half_window > t[-1]:
            continue
        tp, ep, _, at_edge = envelope_pick(
            residual[:, i], t, t_expected - half_window, t_expected + half_window
        )
        if ep >= PICK_MIN_ENVELOPE_RATIO * apex["envelope_peak_value"] and not at_edge:
            picks[i] = round(tp, 4)
            rows.append((t_ray[i], tp - geometry.source_peak_ns - t_ray[i]))
    return picks, np.array(rows)


def _velocity_check(picks, rows, geometry, x0, label):
    if len(rows) == 0:
        raise ValueError(f"{label}: no usable per-trace picks for the velocity check")
    slope, intercept = np.polyfit(rows[:, 0], rows[:, 1], 1)
    resid_rms = float(
        np.sqrt(np.mean((rows[:, 1] - (intercept + slope * rows[:, 0])) ** 2))
    )
    picked_offsets = [
        abs(geometry.x_mid_m[i] - x0) for i in range(len(picks)) if picks[i] is not None
    ]
    return {
        "n_picks": len(rows),
        "slope_residual_vs_raytime": float(slope),
        "apparent_v_ratio_from_pick_slope": float(1.0 / (1.0 + slope)),
        "intercept_ns": float(intercept),
        "rms_after_linear_ns": resid_rms,
        "residual_spread_ns": float(np.max(rows[:, 1]) - np.min(rows[:, 1])),
        "max_abs_offset_from_apex_m": float(max(picked_offsets)),
    }


def _analyse_pipe(residual, geometry, pipe, t_ray, neighbours, label):
    apex = _measure_apex(residual, geometry, pipe.center_x_m, t_ray)
    picks, rows = _velocity_picks(residual, geometry, pipe, t_ray, apex, neighbours)
    return {
        "ray_time_ns": t_ray,
        "apex": apex,
        "velocity_check": _velocity_check(
            picks, rows, geometry, pipe.center_x_m, label
        ),
        "picks": picks,
    }


def _time_zero_candidates(bg, fs, geometry):
    """Time-zero candidates (source, direct wave, surface reflection; ns).

    Also returns the geometric surface-reflection time and the no-pipe envelope.
    """
    t, f_ghz, h = geometry.t_ns, geometry.f_ghz, geometry.air_gap_m
    env_bg = np.abs(hilbert(bg[:, 1].astype(np.float64)))
    k_dir = int(np.argmax(env_bg))
    tz_direct = float(t[0] + refine_peak(env_bg, k_dir) * geometry.dt_ns)
    t_surf_geom = surface_reflection_time(geometry.offset_m, h) if h > 0 else None
    tz_surface_meas = None
    if fs is not None:
        surf = bg[:, 1].astype(np.float64) - fs[:, 0].astype(np.float64)
        te_s = geometry.source_peak_ns + t_surf_geom
        tz_surface_meas, _, _, _ = envelope_pick(
            surf, t, te_s - 0.5 / f_ghz, te_s + 1.0 / f_ghz
        )
    tz_surface = (
        tz_surface_meas
        if tz_surface_meas is not None
        else (tz_direct if h == 0 else None)
    )
    tz = {
        "source_peak": geometry.source_peak_ns,
        "direct_wave_peak": tz_direct,
        "surface_reflection_peak": tz_surface,
    }
    return tz, t_surf_geom, env_bg


def _effective_t0_and_depth(tz, t_apex, v):
    t0_eff = {k: (t_apex - val if val is not None else None) for k, val in tz.items()}
    d_eff = {k: (v * val / 2 if val is not None else None) for k, val in t0_eff.items()}
    return t0_eff, d_eff


def _artefact_checks(
    spec, bg, residual, env_bg, geometry, earliest_ray_ns, env_reference
):
    """Artefact levels relative to ``env_reference`` (apex echo envelope).

    ``earliest_ray_ns`` is None for a background line (no pipe): the pre-arrival
    level is then undefined and the whole-line residual level is reported instead.
    """
    t, f_ghz, h = geometry.t_ns, geometry.f_ghz, geometry.air_gap_m
    lateral = [
        float(np.max(np.abs(bg[:, j].astype(np.float64) - bg[:, 1])) / env_reference)
        for j in (0, 2)
    ]
    pre_rel = None
    if earliest_ray_ns is not None:
        pre = 0.0
        for i in range(residual.shape[1]):
            m = (
                t
                < geometry.source_peak_ns
                + earliest_ray_ns[i]
                - PRE_ARRIVAL_MARGIN_PERIODS / f_ghz
            )
            if m.any():
                pre = max(pre, float(np.max(np.abs(residual[m, i]))))
        pre_rel = pre / env_reference
    depth_pml = spec["y_surface_m"] - spec["pml_cells"] * spec["dx_cell_m"]
    t_bottom = (
        geometry.source_peak_ns
        + 2 * h / C0_M_PER_NS
        + 2 * math.hypot(depth_pml, geometry.offset_m / 2) / geometry.velocity_m_per_ns
    )
    bottom_rel = None
    if t_bottom + 0.5 / f_ghz < t[-1]:
        m = (t >= t_bottom - 0.5 / f_ghz) & (t <= t_bottom + 0.5 / f_ghz)
        bottom_rel = float(np.max(env_bg[m]) / env_reference)
    first_bg_x, _, last_bg_x = spec["bg_trace_positions_m"]
    checks = {
        f"bg_lateral_diff_rel_x{first_bg_x:g}_x{last_bg_x:g}": lateral,
        "pre_arrival_residual_rel": pre_rel,
        "bottom_pml_window_ns": t_bottom,
        "bottom_pml_window_envelope_rel": bottom_rel,
    }
    if earliest_ray_ns is None:
        # Background line minus the 3-trace background: FDTD lateral residual only.
        checks["line_residual_max_rel"] = (
            float(np.max(np.abs(residual))) / env_reference
        )
    return checks


def _check_apex_traces(spec, results):
    expected = spec.get("expected_apex_trace_index")
    found = [r["apex"]["trace_index"] for r in results]
    if expected is not None and list(expected) != found:
        raise ValueError(
            f"{spec['model_id']}: apex traces {found} differ from the spec {expected}"
        )


def _ray_list(t_ray):
    return [round(float(x), 5) for x in t_ray]


def _pipe_record(index, analysed_pipe, spec):
    pipe = _pipe_fields(analysed_pipe, spec["velocity_m_per_ns"])
    return {
        "pipe_index": index,
        "pipe_material": spec["pipe_kind"],
        "pipe_center_x_m": pipe["center_x"],
        "pipe_top_depth_m": pipe["top"],
        "pipe_center_depth_m": pipe["center_depth"],
        "pipe_radius_m": pipe["radius"],
        "t0_top_ns": pipe["t0_top"],
        "t0_center_ns": pipe["t0_center"],
        "apex_trace_index": pipe["apex"]["trace_index"],
        "ray_reflection_time_ns": pipe["ray"],
        "apex": pipe["apex"],
        "t0_effective_ns": pipe["t0_eff"],
        "depth_from_t0_effective_m": pipe["d_eff"],
        "velocity_check": pipe["velocity_check"],
        "trace_envelope_pick_ns": pipe["picks"],
    }


def _save_arrays(out_dir, mid, data, bg, fs):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{mid}.npy", data.astype(np.float32))
    np.save(out_dir / f"{mid}_bg.npy", bg.astype(np.float32))
    if fs is not None:
        np.save(out_dir / f"{mid}_fs.npy", fs.astype(np.float32))


def export_model(
    spec: dict, sources: SourceFiles | None = None, out_dir: Path | None = None
) -> dict:
    """Export one model's arrays and ground-truth json; return the json content.

    Args:
        spec: Model spec (``model_specs.json`` entry, or a ``load_s1g_specs`` spec
            with ``pipes``).
        sources: gprMax output files; default is the M01-M20 layout under ``RAW``.
        out_dir: Output folder; default ``OUT``.

    Raises:
        ValueError: If the runs disagree in time axis or shape, a pipe has no usable
            velocity picks, or the apex traces differ from
            ``expected_apex_trace_index``.
    """
    sources = legacy_source_files(spec) if sources is None else sources
    out_dir = OUT if out_dir is None else out_dir
    mid = spec["model_id"]
    data, bg, fs, dt_s, iters = _read_model_arrays(spec, sources)
    geometry = _scan_geometry(spec, dt_s, iters)
    pipes = pipes_from_spec(spec)
    ray_times = [_pipe_ray_times(pipe, geometry) for pipe in pipes]

    # Pipe-only response: subtract the no-pipe run (middle background trace; its
    # lateral invariance is checked in _artefact_checks).
    residual = data.astype(np.float64) - bg[:, 1:2].astype(np.float64)
    results = []
    for k, (pipe, t_ray) in enumerate(zip(pipes, ray_times, strict=True)):
        neighbours = [
            (p, r)
            for j, (p, r) in enumerate(zip(pipes, ray_times, strict=True))
            if j != k
        ]
        results.append(
            _analyse_pipe(
                residual, geometry, pipe, t_ray, neighbours, f"{mid} pipe {k}"
            )
        )
    _check_apex_traces(spec, results)

    tz, t_surf_geom, env_bg = _time_zero_candidates(bg, fs, geometry)
    effective = [
        _effective_t0_and_depth(
            tz, r["apex"]["envelope_peak_ns"], geometry.velocity_m_per_ns
        )
        for r in results
    ]
    if pipes:
        earliest_ray_ns = np.min(np.vstack(ray_times), axis=0)
        env_reference = min(r["apex"]["envelope_peak_value"] for r in results)
    else:
        # Background line: no echo to normalise by; use the time-zero band peak.
        earliest_ray_ns = None
        env_reference = float(np.max(env_bg))
    checks = _artefact_checks(
        spec, bg, residual, env_bg, geometry, earliest_ray_ns, env_reference
    )
    _save_arrays(out_dir, mid, data, bg, fs)

    first_pipe = (pipes[0], results[0], effective[0]) if pipes else None
    meta = _single_pipe_meta(
        spec, geometry, iters, first_pipe, tz, t_surf_geom, checks, fs
    )
    if "pipes" in spec:
        meta["n_pipes"] = len(pipes)
        analysed = zip(pipes, results, effective, strict=True)
        meta["pipes"] = [
            _pipe_record(k, analysed_pipe, spec)
            for k, analysed_pipe in enumerate(analysed)
        ]
        meta["pipe_keys_note"] = PIPE_KEYS_NOTE if pipes else BACKGROUND_LINE_NOTE
    meta.update(spec.get("extra_meta", {}))
    (out_dir / f"{mid}.json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8", newline="\n"
    )
    return meta


def _pipe_fields(analysed_pipe, v):
    """Json values of one analysed pipe ``(pipe, result, (t0_eff, d_eff))``.

    All None when ``analysed_pipe`` is None (background line, no pipe).
    """
    if analysed_pipe is None:
        names = (
            "center_x", "top", "center_depth", "radius", "t0_top", "t0_center",
            "ray", "apex", "t0_eff", "d_eff", "velocity_check", "picks",
        )  # fmt: skip
        return dict.fromkeys(names)
    pipe, result, (t0_eff, d_eff) = analysed_pipe
    return {
        "center_x": pipe.center_x_m,
        "top": pipe.top_depth_m,
        "center_depth": pipe.top_depth_m + pipe.radius_m,
        "radius": pipe.radius_m,
        "t0_top": 2 * pipe.top_depth_m / v,
        "t0_center": 2 * (pipe.top_depth_m + pipe.radius_m) / v,
        "ray": _ray_list(result["ray_time_ns"]),
        "apex": result["apex"],
        "t0_eff": t0_eff,
        "d_eff": d_eff,
        "velocity_check": result["velocity_check"],
        "picks": result["picks"],
    }


def _single_pipe_meta(spec, geometry, iters, first_pipe, tz, t_surf_geom, checks, fs):
    """Json content in the single-pipe schema.

    ``first_pipe`` is (pipe, analysis result, (t0_eff, d_eff)) of pipes[0], or None
    for a background line. Key order and string texts are fixed: the M01-M20 files
    must not change.
    """
    mid = spec["model_id"]
    v = spec["velocity_m_per_ns"]
    h = geometry.air_gap_m
    pipe = _pipe_fields(first_pipe, v)
    return {
        "model_id": mid,
        "array_layout": (
            "[samples, traces], float32, raw Ez (V/m) from gprMax 4.0.1 2D TMz, "
            "no gain/filter/time-zero shift"
        ),
        "dt_ns": geometry.dt_ns,
        "n_samples": iters,
        "n_traces": spec["n_traces"],
        "dx_m": spec["trace_step_m"],
        "time_of_sample0_ns": 0.0,
        "trace_x_m_first": float(geometry.x_mid_m[0]),
        "trace_x_m": "x_m[i] = trace_x_m_first + i * dx_m (Tx/Rx midpoint)",
        "eps_r": spec["eps_r"],
        "soil_sigma_S_per_m": spec["soil_sigma_S_per_m"],
        "velocity_m_per_ns": v,
        "pipe_material": spec["pipe_kind"],
        "pipe_wall_m": spec["pipe_wall_m"],
        "pipe_center_x_m": pipe["center_x"],
        "pipe_top_depth_m": pipe["top"],
        "pipe_center_depth_m": pipe["center_depth"],
        "pipe_radius_m": pipe["radius"],
        "center_freq_hz": spec["center_freq_hz"],
        "wavelet": (
            "gprMax ricker (current of z-directed Hertzian dipole), peak at sqrt(2)/f"
        ),
        "air_gap_m": h,
        "antenna_offset_m": geometry.offset_m,
        "fdtd_cell_m": spec["dx_cell_m"],
        "pml_cells": spec["pml_cells"],
        "time_window_ns": spec["time_window_ns"],
        "t0_definition": (
            "t0 = two-way vertical travel time in soil from the ground surface to "
            "the pipe top, zero offset: t0_top_ns = 2*d_top/v. Then d_top = v*t0/2. "
            "Times in the array are simulation time (sample k at k*dt_ns); "
            "subtract a time-zero (see time_zero_candidates_ns) before using the "
            "hyperbola formula. With a pipe of radius R the exact zero-offset "
            "curve is t(x) = (2/v)*(sqrt((d_top+R)^2+(x-x0)^2) - R)."
        ),
        "t0_top_ns": pipe["t0_top"],
        "t0_center_ns": pipe["t0_center"],
        "air_two_way_ns": 2 * h / C0_M_PER_NS,
        "ray_reflection_time_ns": pipe["ray"],
        "ray_reflection_time_definition": (
            "geometric-optics travel time Tx -> specular point on pipe top -> Rx "
            "for each trace, including antenna offset and refraction through the "
            "air gap; excludes the source delay and the wavelet lag. Data time of "
            "the reflection envelope peak ~= source_peak + wavelet_lag + this."
        ),
        "time_zero_candidates_ns": {
            **tz,
            "notes": (
                "source_peak = sqrt(2)/f (Ricker current peak, exact). "
                "direct_wave_peak = envelope maximum of the no-pipe trace (direct "
                "coupling; for h=0 it also contains the ground wave and the surface "
                "reflection). surface_reflection_peak = envelope maximum of "
                "(no-pipe - free-space) for h>0, equal to direct_wave_peak for h=0."
            ),
        },
        "surface_reflection_geom_ns": t_surf_geom,
        "direct_air_geom_ns": direct_air_time(geometry.offset_m),
        "apex": pipe["apex"],
        "t0_effective_ns": pipe["t0_eff"],
        "depth_from_t0_effective_m": pipe["d_eff"],
        "velocity_check": pipe["velocity_check"],
        "trace_envelope_pick_ns": pipe["picks"],
        "checks": checks,
        "files": {
            "bscan": f"{mid}.npy",
            "background": f"{mid}_bg.npy",
            "free_space": f"{mid}_fs.npy" if fs is not None else None,
            "gprmax_input": spec.get(
                "gprmax_input", f"scripts/synthetic/inputs/{mid}.in"
            ),
        },
    }


def _per_pipe_value(meta, read_value):
    """Single-pipe json: the scalar. With a pipe list: one value per pipe joined by ';'.

    ``read_value`` works on both the top level and a ``pipes`` entry, which share the
    keys ``pipe_center_x_m``, ``apex`` and ``velocity_check``.
    """
    if "pipes" not in meta:
        return read_value(meta)
    return PIPE_LIST_SEPARATOR.join(str(read_value(pipe)) for pipe in meta["pipes"])


def _apex_peak_ns(record):
    return round(record["apex"]["envelope_peak_ns"], 4)


def _wavelet_lag_ns(record):
    return round(record["apex"]["wavelet_lag_ns"], 4)


def _pick_slope_v_ratio(record):
    return round(record["velocity_check"]["apparent_v_ratio_from_pick_slope"], 5)


def _rounded_or_none(value, digits):
    return None if value is None else round(value, digits)


def _doubled_or_none(value):
    return None if value is None else 2 * value


def manifest_row(m: dict) -> dict:
    """One manifest.csv row of an exported model json."""
    tzc = m["time_zero_candidates_ns"]
    row = {
        "model_id": m["model_id"],
        "npy": m["files"]["bscan"],
        "json": f"{m['model_id']}.json",
        "pipe_material": m["pipe_material"],
        "pipe_outer_diameter_m": _doubled_or_none(m["pipe_radius_m"]),
        "pipe_radius_m": m["pipe_radius_m"],
        "pipe_top_depth_m": m["pipe_top_depth_m"],
        "pipe_center_depth_m": _rounded_or_none(m["pipe_center_depth_m"], 6),
        "pipe_center_x_m": _per_pipe_value(m, lambda record: record["pipe_center_x_m"]),
        "eps_r": m["eps_r"],
        "soil_sigma_S_per_m": m["soil_sigma_S_per_m"],
        "velocity_m_per_ns": round(m["velocity_m_per_ns"], 6),
        "center_freq_mhz": m["center_freq_hz"] / 1e6,
        "air_gap_m": m["air_gap_m"],
        "antenna_offset_m": m["antenna_offset_m"],
        "n_samples": m["n_samples"],
        "n_traces": m["n_traces"],
        "dt_ns": round(m["dt_ns"], 8),
        "dx_m": m["dx_m"],
        "time_window_ns": m["time_window_ns"],
        "fdtd_cell_m": m["fdtd_cell_m"],
        "t0_top_ns": _rounded_or_none(m["t0_top_ns"], 4),
        "t0_center_ns": _rounded_or_none(m["t0_center_ns"], 4),
        "tz_source_ns": round(tzc["source_peak"], 4),
        "tz_direct_ns": round(tzc["direct_wave_peak"], 4),
        "tz_surface_ns": _rounded_or_none(tzc["surface_reflection_peak"], 4),
        "apex_envelope_peak_ns": _per_pipe_value(m, _apex_peak_ns),
        "wavelet_lag_ns": _per_pipe_value(m, _wavelet_lag_ns),
        "apparent_v_ratio_pick_slope": _per_pipe_value(m, _pick_slope_v_ratio),
    }
    if "pipes" in m:
        row["n_pipes"] = m["n_pipes"]
    return row


def _summary_line(m):
    a, vc, ck = m["apex"], m["velocity_check"], m["checks"]
    lateral_key = next(key for key in ck if key.startswith("bg_lateral_diff_rel_"))
    if a is None:  # background line, no pipe
        return (
            f"{m['model_id']} shape=({m['n_samples']},{m['n_traces']}) "
            f"dt={m['dt_ns']:.6f} no pipe lat={max(ck[lateral_key]):.2e} "
            f"line_residual={ck['line_residual_max_rel']:.2e} "
            f"bot={ck['bottom_pml_window_envelope_rel']}"
        )
    line = (
        f"{m['model_id']} shape=({m['n_samples']},{m['n_traces']}) "
        f"dt={m['dt_ns']:.6f} apex={a['envelope_peak_ns']:.3f} "
        f"exp={a['expected_envelope_peak_ns']:.3f} lag={a['wavelet_lag_ns']:+.3f} "
        f"edge={a['envelope_peak_at_window_edge']} pol={a['signed_peak_polarity']:+d} "
        f"vg/vt={vc['apparent_v_ratio_from_pick_slope']:.4f} n={vc['n_picks']} "
        f"rms={vc['rms_after_linear_ns']:.3f} lat={max(ck[lateral_key]):.2e} "
        f"pre={ck['pre_arrival_residual_rel']:.2e} "
        f"bot={ck['bottom_pml_window_envelope_rel']}"
    )
    for p in m.get("pipes", []):
        pipe_apex, pipe_vc = p["apex"], p["velocity_check"]
        line += (
            f"\n  pipe {p['pipe_index']} x0={p['pipe_center_x_m']} "
            f"apex_trace={p['apex_trace_index']} "
            f"apex={pipe_apex['envelope_peak_ns']:.3f} "
            f"lag={pipe_apex['wavelet_lag_ns']:+.3f} "
            f"vg/vt={pipe_vc['apparent_v_ratio_from_pick_slope']:.4f} "
            f"n={pipe_vc['n_picks']}"
        )
    return line


def write_manifest(out_dir: Path, metas: list[dict]) -> None:
    """Write ``manifest.csv`` for the exported models (``n_pipes`` with pipe lists)."""
    rows = [manifest_row(m) for m in metas]
    has_pipe_lists = any("pipes" in m for m in metas)
    fields = MULTI_PIPE_MANIFEST_FIELDS if has_pipe_lists else MANIFEST_FIELDS
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--s1g",
        type=Path,
        help="S1g models json; default exports scripts/synthetic/model_specs.json",
    )
    parser.add_argument(
        "--s4s6",
        type=Path,
        help="S4/S6 models json (analysis/synthetic/s4s6/models_s4s6.json)",
    )
    parser.add_argument(
        "--h5-dir", type=Path, help="S1g, S4/S6 only: folder of the gprMax h5 outputs"
    )
    parser.add_argument(
        "--raw-dir", type=Path, help=f"M01-M20 only: raw h5 folder (default {RAW})"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help=(
            f"output folder (default {OUT}, {S1G_OUT} with --s1g, "
            f"{S4S6_OUT} with --s4s6)"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="export only these model ids; manifest.csv then lists only these",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.s1g is not None and args.s4s6 is not None:
        raise SystemExit("--s1g and --s4s6 are exclusive")
    if args.s4s6 is not None:
        jobs = load_s4s6_specs(args.s4s6, args.h5_dir)
        out_dir = S4S6_OUT if args.out_dir is None else args.out_dir
    elif args.s1g is not None:
        jobs = load_s1g_specs(args.s1g, args.h5_dir)
        out_dir = S1G_OUT if args.out_dir is None else args.out_dir
    else:
        specs = json.loads((HERE / "model_specs.json").read_text(encoding="utf-8"))
        jobs = [(spec, legacy_source_files(spec, args.raw_dir)) for spec in specs]
        out_dir = OUT if args.out_dir is None else args.out_dir
    if args.models:
        unknown = set(args.models) - {spec["model_id"] for spec, _ in jobs}
        if unknown:
            raise SystemExit(f"unknown model ids: {sorted(unknown)}")
        jobs = [
            (spec, sources) for spec, sources in jobs if spec["model_id"] in args.models
        ]
    metas = []
    for spec, sources in jobs:
        meta = export_model(spec, sources, out_dir)
        print(_summary_line(meta), flush=True)
        metas.append(meta)
    write_manifest(out_dir, metas)


if __name__ == "__main__":
    main()
