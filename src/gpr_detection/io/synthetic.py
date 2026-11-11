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
* Multi-pipe exports (S1g, M2-10) add ``n_pipes`` and a ``pipes`` list; the
  top-level pipe keys then describe ``pipes[0]``. A background line has
  ``n_pipes == 0`` and null pipe keys: it loads as a radargram, has no pipe
  truth (``load_synthetic_pipe_truths`` returns ``[]``). Jsons without
  ``n_pipes`` (M01-M20) are read exactly as before.
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
# Survey settings a background line (n_pipes == 0) still must carry.
_REQUIRED_WITHOUT_PIPE = (
    "dt_ns",
    "dx_m",
    "n_samples",
    "n_traces",
    "center_freq_hz",
    "eps_r",
    "velocity_m_per_ns",
    "air_gap_m",
    "antenna_offset_m",
    "trace_x_m_first",
    "time_zero_candidates_ns",
)
_REQUIRED_APEX = ("trace_index", "envelope_peak_ns")
_REQUIRED_ZERO = ("source_peak",)
_REQUIRED_PIPE_ENTRY = (
    "pipe_top_depth_m",
    "pipe_center_depth_m",
    "pipe_radius_m",
    "pipe_center_x_m",
    "t0_top_ns",
    "ray_reflection_time_ns",
    "apex",
)


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


def _is_background_line(meta: Mapping[str, Any]) -> bool:
    """True for an export without any pipe (``n_pipes == 0``)."""
    return meta.get("n_pipes") == 0


def _read_checked_json(json_path: Path) -> dict[str, Any]:
    """Read a model json and check every field this module relies on.

    A background line (``n_pipes == 0``) is checked for survey settings only.

    Raises:
        ValueError: If a required field is missing or null.
    """
    meta = read_json(json_path)
    if _is_background_line(meta):
        missing = _missing_fields(meta, _REQUIRED_WITHOUT_PIPE)
    else:
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
    include_height: bool = True,
) -> Radargram:
    """Load one model as a :class:`Radargram` with survey-level metadata only.

    Filled: ``dt_ns``, ``dx_m`` (trace ``i`` at ``i * dx``; the json axis starts
    at ``trace_x_m_first``), ``f_c_hz``, ``air_gap`` (``"air"`` when
    ``air_gap_m > 0``), ``antenna_offset_m`` and optionally ``eps_r_meta``
    (source ``"synthetic_truth"``), ``pipe_radius_m`` and ``antenna_height_m``
    (json ``air_gap_m`` of air-coupled models: the antenna mounting height is
    survey metadata, spec_depth_v1 v2.1 section 1-1). Zero candidates, apex
    times and depths are **not** passed on (ground truth).

    Args:
        model_id: e.g. ``"M02"``.
        root: Folder holding the ``.npy``/``.json`` pairs.
        include_eps: Pass the true eps_r as ``eps_r_meta``; False tests the
            no-metadata velocity rules.
        include_radius: Pass the true radius as ``pipe_radius_m`` (None for a
            background line, ``n_pipes == 0``).
        include_height: Pass ``air_gap_m > 0`` as ``antenna_height_m``; False
            tests the unknown-height zero rule (regression R14).

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

    air_gap_m = float(meta["air_gap_m"])
    return Radargram(
        data=raw.astype(np.float32, copy=False),
        dt_ns=float(meta["dt_ns"]),
        dx_m=float(meta["dx_m"]),
        air_gap="air" if air_gap_m > 0 else "ground",
        amplitude_processed=False,
        dataset=DATASET_KEY,
        line_id=model_id,
        f_c_hz=float(meta["center_freq_hz"]),
        eps_r_meta=float(meta["eps_r"]) if include_eps else None,
        eps_r_meta_source=EPS_R_SOURCE if include_eps else "",
        antenna_offset_m=float(meta["antenna_offset_m"]),
        antenna_height_m=air_gap_m if include_height and air_gap_m > 0 else None,
        # A background line has no pipe, so its radius stays None.
        pipe_radius_m=(
            _optional_float(meta["pipe_radius_m"]) if include_radius else None
        ),
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


def _truth_from(
    model_id: str, meta: Mapping[str, Any], pipe: Mapping[str, Any]
) -> SyntheticTruth:
    """Build the truth of one pipe: pipe keys from ``pipe``, the rest from ``meta``.

    ``pipe`` is the json top level for a single-pipe export or one ``pipes``
    entry; both use the same key names.
    """
    apex = pipe["apex"]
    zero_candidates = meta["time_zero_candidates_ns"]
    return SyntheticTruth(
        model_id=model_id,
        pipe_top_depth_m=float(pipe["pipe_top_depth_m"]),
        pipe_center_depth_m=float(pipe["pipe_center_depth_m"]),
        pipe_radius_m=float(pipe["pipe_radius_m"]),
        pipe_center_x_m=float(pipe["pipe_center_x_m"]),
        eps_r=float(meta["eps_r"]),
        velocity_m_per_ns=float(meta["velocity_m_per_ns"]),
        air_gap_m=float(meta["air_gap_m"]),
        antenna_offset_m=float(meta["antenna_offset_m"]),
        t0_top_ns=float(pipe["t0_top_ns"]),
        apex_trace_index=int(apex["trace_index"]),
        apex_envelope_peak_ns=float(apex["envelope_peak_ns"]),
        source_peak_ns=float(zero_candidates["source_peak"]),
        direct_wave_peak_ns=_optional_float(zero_candidates.get("direct_wave_peak")),
        surface_reflection_peak_ns=_optional_float(
            zero_candidates.get("surface_reflection_peak")
        ),
        wavelet_lag_ns=_optional_float(apex.get("wavelet_lag_ns")),
        trace_x_m_first=float(meta["trace_x_m_first"]),
        ray_reflection_time_ns=tuple(float(t) for t in pipe["ray_reflection_time_ns"]),
        pipe_material=pipe.get("pipe_material"),
    )


def load_synthetic_truth(
    model_id: str, *, root: Path = DEFAULT_SYNTHETIC_ROOT
) -> SyntheticTruth:
    """Read the ground truth of one model from its json.

    For a multi-pipe export this is the truth of ``pipes[0]`` (the pipe with
    the smallest x, mirrored in the top-level keys); use
    :func:`load_synthetic_pipe_truths` for every pipe.

    Raises:
        ValueError: If a required field is missing or null, or the model is a
            background line (``n_pipes == 0``, no pipe truth).
    """
    meta = _read_checked_json(root / f"{model_id}.json")
    if _is_background_line(meta):
        raise ValueError(
            f"{model_id} is a background line (n_pipes == 0) and has no pipe truth;"
            " use load_synthetic_pipe_truths"
        )
    return _truth_from(model_id, meta, meta)


def load_synthetic_pipe_truths(
    model_id: str, *, root: Path = DEFAULT_SYNTHETIC_ROOT
) -> list[SyntheticTruth]:
    """Read the ground truth of every pipe of one model, sorted by pipe x.

    Returns:
        ``[]`` for a background line (``n_pipes == 0``); one entry per
        ``pipes`` element for a multi-pipe export; for a json without
        ``pipes`` (M01-M20) the single :func:`load_synthetic_truth` result.

    Raises:
        ValueError: If a required field (top level or pipe entry) is missing
            or null.
    """
    json_path = root / f"{model_id}.json"
    meta = _read_checked_json(json_path)
    if "pipes" not in meta:
        return [_truth_from(model_id, meta, meta)]
    truths = []
    for index, pipe in enumerate(meta["pipes"]):
        missing = _missing_fields(pipe, _REQUIRED_PIPE_ENTRY)
        missing += [
            f"apex.{name}"
            for name in _missing_fields(pipe.get("apex") or {}, _REQUIRED_APEX)
        ]
        if missing:
            raise ValueError(f"{json_path}: pipes[{index}] misses fields {missing}")
        truths.append(_truth_from(model_id, meta, pipe))
    return truths
