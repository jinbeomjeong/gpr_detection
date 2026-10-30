"""gprMax synthetic pipe models (``data/synthetic_gprmax``, spec_depth_v1 section 1-1).

Facts used here (``scripts/synthetic/export_dataset.py`` and the json files):

* ``<id>.npy`` is float32 ``[n_samples, n_traces]`` raw Ez from gprMax 2-D TMz,
  no gain, filter or time-zero shift; sample ``k`` is at ``k * dt_ns``.
* ``<id>_bg.npy`` is float32 ``[n_samples, 3]``: the **same model without the
  pipe**, traces at x = 0.4 / 1.5 / 2.6 m. It holds the direct coupling, ground
  wave and surface reflection, i.e. an ideal background for analysis only.
* ``<id>.json`` mixes antenna/grid settings with ground truth (depth, velocity,
  zero and apex times). Only the settings a real survey would record go into
  the :class:`Radargram` (``load_synthetic``); the truth is a separate typed
  object (``load_synthetic_truth``) so it cannot leak into depth estimation.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gpr_detection.io._common import read_csv_as_text, read_json
from gpr_detection.io.paths import DEFAULT_SYNTHETIC_ROOT
from gpr_detection.io.radargram import Radargram

DATASET_KEY = "synthetic_gprmax"
EPS_R_SOURCE = "synthetic_truth"
MANIFEST_CSV = "manifest.csv"
BACKGROUND_SUFFIX = "_bg"

_REQUIRED_TOP_LEVEL = (
    "dt_ns",
    "dx_m",
    "n_samples",
    "n_traces",
    "center_freq_hz",
    "eps_r",
    "velocity_m_per_ns",
    "pipe_top_depth_m",
    "pipe_center_depth_m",
    "pipe_radius_m",
    "pipe_center_x_m",
    "air_gap_m",
    "antenna_offset_m",
    "t0_top_ns",
    "trace_x_m_first",
    "ray_reflection_time_ns",
    "apex",
    "time_zero_candidates_ns",
)
_REQUIRED_APEX = ("trace_index", "envelope_peak_ns")
_REQUIRED_ZERO = ("source_peak",)


@dataclass(frozen=True)
class SyntheticTruth:
    """Ground truth of one synthetic model (for regression tests, spec 11-2).

    Attributes:
        model_id: e.g. ``"M02"``.
        pipe_top_depth_m: Depth of the pipe top below the ground surface.
        pipe_center_depth_m: ``pipe_top_depth_m + pipe_radius_m``.
        pipe_radius_m: Outer radius.
        pipe_center_x_m: Pipe position on the json x axis (first trace at
            ``trace_x_m_first``, not 0).
        eps_r: Soil relative permittivity.
        velocity_m_per_ns: ``c / sqrt(eps_r)``.
        air_gap_m: Antenna height ``h`` (0 = ground-coupled).
        antenna_offset_m: Tx-Rx separation ``s``.
        t0_top_ns: Two-way soil time to the pipe top, ``2 d_top / v``.
        apex_trace_index: Trace of the hyperbola apex.
        apex_envelope_peak_ns: Envelope peak time of the apex (simulation clock).
        source_peak_ns: Transmit (Ricker current) peak time.
        direct_wave_peak_ns: Envelope peak of the no-pipe trace, if given.
        surface_reflection_peak_ns: Surface-reflection peak (equals the direct
            wave peak for h = 0), if given.
        wavelet_lag_ns: Envelope peak lag behind the ray time at the apex.
        trace_x_m_first: x of trace 0 on the json axis.
        ray_reflection_time_ns: Geometric-optics time per trace (source delay
            and wavelet lag excluded).
        pipe_material: e.g. ``"pec"``, ``"pvc_water"``; ``None`` if absent.
    """

    model_id: str
    pipe_top_depth_m: float
    pipe_center_depth_m: float
    pipe_radius_m: float
    pipe_center_x_m: float
    eps_r: float
    velocity_m_per_ns: float
    air_gap_m: float
    antenna_offset_m: float
    t0_top_ns: float
    apex_trace_index: int
    apex_envelope_peak_ns: float
    source_peak_ns: float
    direct_wave_peak_ns: float | None
    surface_reflection_peak_ns: float | None
    wavelet_lag_ns: float | None
    trace_x_m_first: float
    ray_reflection_time_ns: tuple[float, ...]
    pipe_material: str | None = None


def _model_paths(root: Path, model_id: str) -> tuple[Path, Path]:
    return root / f"{model_id}.npy", root / f"{model_id}.json"


def _missing_fields(record: Mapping[str, Any], fields: tuple[str, ...]) -> list[str]:
    return [name for name in fields if record.get(name) is None]


def _read_checked_json(json_path: Path) -> dict[str, Any]:
    """Read a model json and check every field this module relies on.

    Raises:
        ValueError: If a required field is missing or null.
    """
    meta = read_json(json_path)
    missing = _missing_fields(meta, _REQUIRED_TOP_LEVEL)
    missing += [
        f"apex.{name}"
        for name in _missing_fields(meta.get("apex") or {}, _REQUIRED_APEX)
    ]
    zero_candidates = meta.get("time_zero_candidates_ns") or {}
    missing += [
        f"time_zero_candidates_ns.{name}"
        for name in _missing_fields(zero_candidates, _REQUIRED_ZERO)
    ]
    if missing:
        raise ValueError(f"{json_path}: missing required fields {missing}")
    return meta


def list_synthetic_models(root: Path = DEFAULT_SYNTHETIC_ROOT) -> list[str]:
    """Return the model ids (``M01``...) listed in ``manifest.csv``."""
    return read_csv_as_text(root / MANIFEST_CSV)["model_id"].tolist()


def load_synthetic(
    model_id: str,
    *,
    root: Path = DEFAULT_SYNTHETIC_ROOT,
    include_eps: bool = True,
    include_radius: bool = True,
) -> Radargram:
    """Load one model as a :class:`Radargram` with survey-level metadata only.

    Filled: ``dt_ns``, ``dx_m`` (trace ``i`` at ``i * dx``; the json axis starts
    at ``trace_x_m_first``), ``f_c_hz``, ``air_gap`` (``"air"`` when
    ``air_gap_m > 0``), ``antenna_offset_m`` and optionally ``eps_r_meta``
    (source ``"synthetic_truth"``) and ``pipe_radius_m``. Zero candidates,
    apex times and depths are **not** passed on (ground truth).

    Args:
        model_id: e.g. ``"M02"``.
        root: Folder holding the ``.npy``/``.json`` pairs.
        include_eps: Pass the true eps_r as ``eps_r_meta``; False tests the
            no-metadata velocity rules.
        include_radius: Pass the true radius as ``pipe_radius_m``.

    Raises:
        ValueError: If the array shape disagrees with the json, or a required
            json field is missing.
    """
    npy_path, json_path = _model_paths(root, model_id)
    meta = _read_checked_json(json_path)
    raw = np.load(npy_path)
    expected_shape = (meta["n_samples"], meta["n_traces"])
    if raw.shape != expected_shape:
        raise ValueError(f"{npy_path} shape {raw.shape} != json {expected_shape}")

    return Radargram(
        data=raw.astype(np.float32, copy=False),
        dt_ns=float(meta["dt_ns"]),
        dx_m=float(meta["dx_m"]),
        air_gap="air" if float(meta["air_gap_m"]) > 0 else "ground",
        amplitude_processed=False,
        dataset=DATASET_KEY,
        line_id=model_id,
        f_c_hz=float(meta["center_freq_hz"]),
        eps_r_meta=float(meta["eps_r"]) if include_eps else None,
        eps_r_meta_source=EPS_R_SOURCE if include_eps else "",
        antenna_offset_m=float(meta["antenna_offset_m"]),
        pipe_radius_m=float(meta["pipe_radius_m"]) if include_radius else None,
        metadata={
            "model_id": model_id,
            "trace_x_m_first": float(meta["trace_x_m_first"]),
            "time_of_sample0_ns": float(meta.get("time_of_sample0_ns", 0.0)),
        },
    )


def load_synthetic_background(
    model_id: str, *, root: Path = DEFAULT_SYNTHETIC_ROOT
) -> np.ndarray:
    """Load ``<id>_bg.npy``: the same model without the pipe (analysis only).

    Returns:
        float32 ``[n_samples, 3]``, traces at x = 0.4 / 1.5 / 2.6 m on the json
        axis (first, middle, last trace position).
    """
    return np.load(root / f"{model_id}{BACKGROUND_SUFFIX}.npy").astype(
        np.float32, copy=False
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def load_synthetic_truth(
    model_id: str, *, root: Path = DEFAULT_SYNTHETIC_ROOT
) -> SyntheticTruth:
    """Read the ground truth of one model from its json.

    Raises:
        ValueError: If a required field is missing or null.
    """
    meta = _read_checked_json(root / f"{model_id}.json")
    apex = meta["apex"]
    zero_candidates = meta["time_zero_candidates_ns"]
    return SyntheticTruth(
        model_id=model_id,
        pipe_top_depth_m=float(meta["pipe_top_depth_m"]),
        pipe_center_depth_m=float(meta["pipe_center_depth_m"]),
        pipe_radius_m=float(meta["pipe_radius_m"]),
        pipe_center_x_m=float(meta["pipe_center_x_m"]),
        eps_r=float(meta["eps_r"]),
        velocity_m_per_ns=float(meta["velocity_m_per_ns"]),
        air_gap_m=float(meta["air_gap_m"]),
        antenna_offset_m=float(meta["antenna_offset_m"]),
        t0_top_ns=float(meta["t0_top_ns"]),
        apex_trace_index=int(apex["trace_index"]),
        apex_envelope_peak_ns=float(apex["envelope_peak_ns"]),
        source_peak_ns=float(zero_candidates["source_peak"]),
        direct_wave_peak_ns=_optional_float(zero_candidates.get("direct_wave_peak")),
        surface_reflection_peak_ns=_optional_float(
            zero_candidates.get("surface_reflection_peak")
        ),
        wavelet_lag_ns=_optional_float(apex.get("wavelet_lag_ns")),
        trace_x_m_first=float(meta["trace_x_m_first"]),
        ray_reflection_time_ns=tuple(float(t) for t in meta["ray_reflection_time_ns"]),
        pipe_material=meta.get("pipe_material"),
    )
