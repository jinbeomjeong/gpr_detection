"""M3-2 S5: gprMax pipe-only responses for the real-background injection (M3).

Sources: plan ``docs/processing/m32_image_quality_plan.md`` v0.2.3 section 3-2
("관 단독 응답") and the EM review ``docs/processing/m32_em_review.md``
section 2-2 ([EMR]). This loader stays in ``analysis/`` because
``gpr_detection.eval`` may not read data files (architecture rule 8).

Model groups (plan 3-2, [EMR] 2-2)

* ``A1`` air gap h 0.1 m: M12, M15, M18 (400 MHz) and the four single-pipe K7
  models (500 MHz). 4TU only.
* ``A0`` ground coupled h 0: M01-M11, M14, M16, M17, M19. 4TU (the h -> 0
  limit) and Guangzhou IDS.
* Excluded: M20 (h 0.3), S4N, S6P, Z_* (other folders, never read). The
  auxiliary models (M13, S1G, K7_P2/P3) are report-only in the plan and are not
  loaded here.

A response is built for the coupling of the line it is injected into, so the
A0 models get ``coupling="air"`` on 4TU. For a ground-coupled model the
``_bg`` band (direct wave and surface reflection, one event) is the event the
4TU zero ``t_z`` sits on, so both alignments use the same measured time:

* ``reference_time_ns``: envelope peak of the ``_bg`` trace after steps
  (1)-(3) (air: band peak; ground: direct-wave peak, plan 3-2 "시간 맞춤").
* ``apex_time_ns``: envelope peak of the apex trace of ``full - _bg`` after
  steps (1)-(3).
* ``source_peak_ns``: json ``time_zero_candidates_ns.source_peak`` (the
  Ricker transmit peak, [EMR] appendix A, L1).

Quality values ([EMR] 1-6), checked by ``gpr_detection.eval.inject``:

* ``mirror_diff``: ``max_i max_t |d_i - d_mirror(i)| / max_t |d_apex|`` on the
  raw pipe-only field, as in ``analysis/processing/m32_em_review_checks.py``.
* ``early_energy_ratio``: ``(e_early / e_apex)^2``, where ``e_early`` is the
  largest envelope (after (1)-(3), any trace) earlier than ``apex_time - 1 P``
  and ``e_apex`` the apex envelope peak, ``P = 1 / f_c``. The plan says
  "energy"; read as an amplitude ratio the band-passed wavelet's own envelope
  (about 3 % at 1 P before its peak) would reject every model.

IDS depth limit ([EMR] 2-2): lines whose time window is at most 30 ns use only
models with a pipe top at most 1.0 m deep.
"""

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

from gpr_detection.eval.inject import (
    EARLY_ENERGY_KEY,
    MIRROR_DIFF_KEY,
    Coupling,
    PipeResponse,
)
from gpr_detection.preprocess.filters import bandpass, dewow, remove_dc

REPO = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = REPO / "data" / "synthetic_gprmax"
SYNTHETIC_K7_DIR = REPO / "data" / "synthetic_gprmax_k7"

NS_PER_S = 1e9
IDS_SHORT_WINDOW_NS = 30.0
"""IDS lines with a time window up to this use only shallow models ([EMR] 2-2)."""
IDS_SHORT_WINDOW_MAX_TOP_M = 1.0
APEX_SEARCH_PERIODS = 0.5
"""The apex peak is searched within +-0.5 P of the json ray-time prediction."""


class ModelGroup(StrEnum):
    """Response bundle of plan 3-2; 4TU reports the two bundles separately."""

    A1 = "A1"
    A0 = "A0"


A1_MODELS = (
    "M12",
    "M15",
    "M18",
    "K7_P1_D060",
    "K7_P1_D110_S002",
    "K7_P1_D150_S003",
    "K7_PVC_D070",
)
A0_MODELS = (
    "M01",
    "M02",
    "M03",
    "M04",
    "M05",
    "M06",
    "M07",
    "M08",
    "M09",
    "M10",
    "M11",
    "M14",
    "M16",
    "M17",
    "M19",
)
EXCLUDED_MODELS = ("M20",)
"""h 0.3 m, a height excluded for 4TU ([EMR] 2-1)."""

MODEL_GROUPS: dict[str, ModelGroup] = {
    **dict.fromkeys(A1_MODELS, ModelGroup.A1),
    **dict.fromkeys(A0_MODELS, ModelGroup.A0),
}


@dataclass(frozen=True)
class ModelFiles:
    """The three files of one model."""

    model_id: str
    bscan: Path
    background: Path
    metadata: Path


@dataclass(frozen=True, eq=False)
class RawModel:
    """One model as read from disk, before it is tied to a line coupling.

    Attributes:
        model_id: e.g. ``"M12"``.
        group: A1 or A0.
        pipe_field: float64 ``full - bg[:, [0]]``.
        background_trace: float64 first ``_bg`` trace.
        metadata: The json document.
        sha256: File name -> sha256 of the three files.
    """

    model_id: str
    group: ModelGroup
    pipe_field: np.ndarray
    background_trace: np.ndarray
    metadata: Mapping[str, object]
    sha256: Mapping[str, str]


def model_files(model_id: str) -> ModelFiles:
    """Paths of a model's ``.npy``, ``_bg.npy`` and ``.json``.

    Raises:
        KeyError: If ``model_id`` is not in A1 or A0.
    """
    if model_id not in MODEL_GROUPS:
        raise KeyError(f"{model_id} is not an A1/A0 injection model")
    folder = SYNTHETIC_K7_DIR if model_id.startswith("K7_") else SYNTHETIC_DIR
    return ModelFiles(
        model_id=model_id,
        bscan=folder / f"{model_id}.npy",
        background=folder / f"{model_id}_bg.npy",
        metadata=folder / f"{model_id}.json",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw_model(model_id: str) -> RawModel:
    """Read one model (h5 files are never used).

    Raises:
        FileNotFoundError: If one of the three files is missing.
        ValueError: If the arrays disagree with each other or with the json.
    """
    files = model_files(model_id)
    for path in (files.bscan, files.background, files.metadata):
        if not path.is_file():
            raise FileNotFoundError(f"{model_id}: missing {path}")
    metadata = json.loads(files.metadata.read_text(encoding="utf-8"))
    full = np.load(files.bscan, allow_pickle=False).astype(np.float64)
    background = np.load(files.background, allow_pickle=False).astype(np.float64)
    if background.ndim == 1:
        background = background[:, None]
    if full.ndim != 2 or background.shape[0] != full.shape[0]:
        raise ValueError(
            f"{model_id}: bscan {full.shape} and background {background.shape} differ"
        )
    if full.shape != (metadata["n_samples"], metadata["n_traces"]):
        raise ValueError(f"{model_id}: bscan {full.shape} disagrees with the json")
    return RawModel(
        model_id=model_id,
        group=MODEL_GROUPS[model_id],
        pipe_field=full - background[:, [0]],
        background_trace=background[:, 0].copy(),
        metadata=metadata,
        sha256={
            path.name: _sha256(path)
            for path in (files.bscan, files.background, files.metadata)
        },
    )


# ------------------------------------------------------------ measurements


def _linear_steps(section: np.ndarray, dt_ns: float, f_c_hz: float) -> np.ndarray:
    """Steps (1)-(3) (DC, dewow, band-pass) on the synthetic grid."""
    values = np.asarray(section, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return bandpass(dewow(remove_dc(values), dt_ns, f_c_hz), dt_ns, f_c_hz)


def _envelope(section: np.ndarray) -> np.ndarray:
    """Hilbert envelope along time, zero padded to 2 N (no wrap-around)."""
    n_samples = section.shape[0]
    return np.abs(hilbert(section, N=2 * n_samples, axis=0)[:n_samples])


def apex_trace_of(metadata: Mapping[str, object]) -> float:
    """Mirror axis in traces: ``(pipe_center_x - trace_x_first) / dx``."""
    centre = metadata["pipe_center_x_m"]
    if isinstance(centre, list):
        centre = centre[0]
    first = float(metadata["trace_x_m_first"])  # type: ignore[arg-type]
    return (float(centre) - first) / float(metadata["dx_m"])  # type: ignore[arg-type]


def mirror_difference(pipe_field: np.ndarray, apex_trace: float) -> float:
    """Largest mirror difference relative to the apex trace peak ([EMR] 1-6)."""
    apex_column = round(apex_trace)
    apex_peak = float(np.abs(pipe_field[:, apex_column]).max())
    if not apex_peak > 0:
        return math.inf
    worst = 0.0
    n_traces = pipe_field.shape[1]
    for column in range(n_traces):
        mirrored = 2 * apex_column - column
        if 0 <= mirrored < n_traces and mirrored != column:
            difference = np.abs(pipe_field[:, column] - pipe_field[:, mirrored]).max()
            worst = max(worst, float(difference) / apex_peak)
    return worst


@dataclass(frozen=True)
class ResponseTimes:
    """Times measured on the synthetic grid after steps (1)-(3)."""

    apex_time_ns: float
    reference_time_ns: float
    early_energy_ratio: float


def _expected_apex_ns(metadata: Mapping[str, object]) -> float:
    apex = metadata["apex"]
    assert isinstance(apex, Mapping)
    return float(apex["expected_envelope_peak_ns"])


def measure_times(raw: RawModel) -> ResponseTimes:
    """Apex and band times and the early-energy ratio of one model.

    The apex is the envelope peak within ``+-0.5 P`` of the json
    ``apex.expected_envelope_peak_ns`` (ray time + source peak): water-filled
    PVC pipes (M08, M14, M18) ring inside the pipe and a later lobe is
    stronger than the top reflection.
    """
    dt_ns = float(raw.metadata["dt_ns"])  # type: ignore[arg-type]
    f_c_hz = float(raw.metadata["center_freq_hz"])  # type: ignore[arg-type]
    apex_column = round(apex_trace_of(raw.metadata))
    period_rows = NS_PER_S / f_c_hz / dt_ns
    pipe_envelope = _envelope(_linear_steps(raw.pipe_field, dt_ns, f_c_hz))
    expected_row = _expected_apex_ns(raw.metadata) / dt_ns
    half_window = APEX_SEARCH_PERIODS * period_rows
    first = max(0, math.floor(expected_row - half_window))
    last = min(pipe_envelope.shape[0] - 1, math.ceil(expected_row + half_window))
    apex_row = first + int(np.argmax(pipe_envelope[first : last + 1, apex_column]))
    apex_peak = float(pipe_envelope[apex_row, apex_column])
    background_envelope = _envelope(_linear_steps(raw.background_trace, dt_ns, f_c_hz))[
        :, 0
    ]
    early_rows = max(0, math.floor(apex_row - period_rows))
    early_peak = float(pipe_envelope[:early_rows].max()) if early_rows else 0.0
    return ResponseTimes(
        apex_time_ns=apex_row * dt_ns,
        reference_time_ns=int(np.argmax(background_envelope)) * dt_ns,
        early_energy_ratio=_energy_ratio(early_peak, apex_peak),
    )


def _energy_ratio(early_peak: float, apex_peak: float) -> float:
    """``(early / apex)^2``: envelope amplitudes compared as energies."""
    if not apex_peak > 0:
        return math.inf
    return (early_peak / apex_peak) ** 2


def _source_peak_ns(metadata: Mapping[str, object]) -> float | None:
    candidates = metadata.get("time_zero_candidates_ns")
    if not isinstance(candidates, Mapping):
        return None
    value = candidates.get("source_peak")
    return None if value is None else float(value)


def build_response(
    raw: RawModel, times: ResponseTimes, coupling: Coupling
) -> PipeResponse:
    """``PipeResponse`` of ``raw`` for a line of the given coupling."""
    metadata = raw.metadata
    radius = metadata["pipe_radius_m"]
    return PipeResponse(
        response_id=raw.model_id,
        data=raw.pipe_field,
        dt_ns=float(metadata["dt_ns"]),  # type: ignore[arg-type]
        dx_m=float(metadata["dx_m"]),  # type: ignore[arg-type]
        apex_trace=apex_trace_of(metadata),
        apex_time_ns=times.apex_time_ns,
        reference_time_ns=times.reference_time_ns,
        coupling=coupling,
        air_gap_m=float(metadata["air_gap_m"]),  # type: ignore[arg-type]
        eps_r=float(metadata["eps_r"]),  # type: ignore[arg-type]
        f_c_hz=float(metadata["center_freq_hz"]),  # type: ignore[arg-type]
        top_depth_m=float(metadata["pipe_top_depth_m"]),  # type: ignore[arg-type]
        od_m=2.0 * float(radius),  # type: ignore[arg-type]
        material=str(metadata["pipe_material"]),
        quality={
            MIRROR_DIFF_KEY: mirror_difference(raw.pipe_field, apex_trace_of(metadata)),
            EARLY_ENERGY_KEY: times.early_energy_ratio,
        },
        source_peak_ns=_source_peak_ns(metadata),
    )


# ---------------------------------------------------------------- library


@dataclass(frozen=True, eq=False)
class ResponseLibrary:
    """All A1/A0 responses, prepared for both line couplings.

    Attributes:
        air: Responses for 4TU (A1 and A0, ``coupling="air"``).
        ground: Responses for IDS (A0 only, ``coupling="ground"``).
        groups: ``model_id`` -> A1/A0.
        sha256: ``model_id`` -> file name -> sha256.
    """

    air: Mapping[str, PipeResponse]
    ground: Mapping[str, PipeResponse]
    groups: Mapping[str, ModelGroup]
    sha256: Mapping[str, Mapping[str, str]]

    def for_line(self, dataset: str, time_window_ns: float | None) -> dict:
        """Responses usable on a line of ``dataset`` (``"4tu"`` or ``"gz_ids"``)."""
        if dataset == "4tu":
            return dict(self.air)
        if dataset != "gz_ids":
            raise ValueError(f"no injection responses for dataset {dataset!r}")
        return select_ids_responses(self.ground, time_window_ns)


def select_ids_responses(
    responses: Mapping[str, PipeResponse], time_window_ns: float | None
) -> dict[str, PipeResponse]:
    """A0 responses for one IDS line; short windows keep shallow pipes only."""
    is_short = time_window_ns is not None and time_window_ns <= IDS_SHORT_WINDOW_NS
    return {
        model_id: response
        for model_id, response in responses.items()
        if not is_short or response.top_depth_m <= IDS_SHORT_WINDOW_MAX_TOP_M
    }


def load_library(model_ids: Iterable[str] | None = None) -> ResponseLibrary:
    """Read every A1/A0 model once and build the air and ground responses."""
    selected: Sequence[str] = (
        tuple(model_ids) if model_ids is not None else (*A1_MODELS, *A0_MODELS)
    )
    air: dict[str, PipeResponse] = {}
    ground: dict[str, PipeResponse] = {}
    hashes: dict[str, Mapping[str, str]] = {}
    for model_id in selected:
        raw = read_raw_model(model_id)
        times = measure_times(raw)
        air[model_id] = build_response(raw, times, "air")
        if raw.group is ModelGroup.A0:
            ground[model_id] = build_response(raw, times, "ground")
        hashes[model_id] = raw.sha256
    return ResponseLibrary(
        air=air,
        ground=ground,
        groups={model_id: MODEL_GROUPS[model_id] for model_id in selected},
        sha256=hashes,
    )


def library_sha256(library: ResponseLibrary) -> str:
    """One sha256 over the sorted model file hashes (run identity)."""
    digest = hashlib.sha256()
    for model_id in sorted(library.sha256):
        for name, value in sorted(library.sha256[model_id].items()):
            digest.update(f"{model_id}\t{name}\t{value}\n".encode())
    return digest.hexdigest()
