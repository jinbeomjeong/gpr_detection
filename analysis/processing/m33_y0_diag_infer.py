"""M3-3 Y0 diagnosis: GPU inference experiments E0-E4 (H2 against H3).

Spec: ``docs/processing/m33_y0_detection_diag.md`` section 6, written by
gpr-processing-expert, who also reads the results. Diagnostic only: there is
no pre-registration and no judgment; the seed and every variant are fixed in
this file.

Every image is inferred with the N2-a' ``last.pt`` of folds 0-4 at conf 0.001
(NMS IoU 0.7, batches of 64, the D0 / M3-2 batched path). Per image and fold
the record keeps the highest ``pipe_hyperbola`` score among the merged line
boxes that contain the truth apex (the D0 / M3-2 rule: box grown by +-3
traces and +-0.5 ``P_line``), so a reaction below 0.25 becomes visible.

Experiments (gprMax models M01-M20):

* E0 ``r0``: the stand-alone record through ``make_detection_image``, i.e.
  the D0 Y0 imaging with the D0 apex.
* E1 ``G4`` (gain none) and ``G2_b0.5`` (capped AGC, beta 0.5): the E0
  record with only the gain step changed (``DetectionImageConfig`` recipe
  options of M3-2 S1); period, scale and tiles stay those of E0.
* E2 ``snr<s>``: pipe-only response (full minus the middle ``_bg`` trace, as
  in ``m33_y0_detect_diag``) plus white noise. SNR is the apex envelope peak
  over the noise RMS, both after the linear steps (DC, dewow, band-pass), at
  6, 12, 18 and 30 dB. One noise field per model (seed 20261115 and the
  model id) is scaled per SNR. r0 with the E0 period.
* E3 ``snr<s>``: the gprMax pipe-only response injected into the first 4TU
  background position of the model in the M3-2 stage-C injection table
  (``inject.inject_realization``, polarity +1, k 1.3, the E2 SNR steps); the
  same gain is then applied to the field with the aperture taper divided out.
  r0. Only the tiles around the apex are inferred.
* E4 ``centred``: the E0 tile with the content moved to the middle (side
  padding split left and right, bottom padding unchanged); boxes are moved
  back before the merge and the match.

Outputs (``--out-root``, default ``runs/eval/m33_y0_diag_infer``):
``records.csv`` (one row per image and fold, scores and coordinates at full
precision) and ``diag_infer.json`` (settings, run identity, images with their
details, skipped images, counts per variant, E0 against the D0 Y0 hits).
The D0 and D0' result folders are refused as output folders.

Stdout carries progress; the last line is the JSON ``{"status", "out_dir",
"n_images", "n_records", "n_tiles", "elapsed_s"}``. Exit codes: 0 done,
2 input refused, 3 internal error.
"""

import argparse
import dataclasses
import fnmatch
import json
import logging
import math
import os
import sys
import time
import zlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import hilbert

# -I leaves the script folder off sys.path; the sibling analysis modules are
# imported from here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detection"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common
import m32_pipe_responses as responses_io
import m32_recipe_eval as stage_c
import m32_recipe_inputs as stage_c_inputs
import m33_d0_gate as d0
import m33_y0_detect_diag as y0_diag

from gpr_detection.detect.base import PixelBox
from gpr_detection.eval import inject
from gpr_detection.io.classes import DetectionClass
from gpr_detection.io.paths import DatasetName, load_dataset_paths
from gpr_detection.io.radargram import GridBox, Radargram
from gpr_detection.io.synthetic import load_synthetic, load_synthetic_truth
from gpr_detection.pipeline import LineBox
from gpr_detection.preprocess.filters import prepare_fitting_branch
from gpr_detection.preprocess.to_image import (
    DetectionImage,
    DetectionImageConfig,
    GainMode,
    make_detection_image,
)

REPO = d0.REPO
SCHEMA = "m33_y0_diag_infer/1"
SPEC = "docs/processing/m33_y0_detection_diag.md section 6"
DEFAULT_OUT_ROOT = REPO / "runs" / "eval" / "m33_y0_diag_infer"
PROTECTED_OUT_ROOTS = (
    REPO / "runs" / "eval" / "m33_d0",
    REPO / "runs" / "eval" / "m33_d0r",
)
"""D0 (sealed by C0b) and D0' results: read only, never an output folder."""
D0_APEX_CSV = REPO / "runs" / "eval" / "m33_d0" / "synthetic_apex.csv"
RUN_ROOT = d0.RUN_ROOT
SYNTHETIC_ROOT = d0.SYNTHETIC_ROOT
STAGE_C_TABLE = (
    REPO / "runs" / "eval" / "m32_recipe_v2" / "plan" / "injection_table.csv"
)
STAGE_C_F2_RECORD = REPO / "analysis" / "detection" / "m32_recipe_f2_v2.json"
"""F2 record of the stage-C run whose table E3 reads (``injection_table_sha256``)."""

SEED = 20261115
CONFIDENCE = 0.001
NMS_IOU = d0.NMS_IOU
INPUT_PX = d0.CANVAS_PX
BATCH_SIZE = d0.BATCH_SIZE
CONF_LINE = d0.CONF_LINE
"""D0 line threshold: an E0 best score above it should be a D0 Y0 hit."""
SNR_STEPS_DB = (6.0, 12.0, 18.0, 30.0)
MODELS = d0.SYNTHETIC_MODELS
FOLDS = d0.EXPECTED_FOLDS
E3_POSITIONS_PER_MODEL = 1
E3_POLARITY = 1
E3_K = inject.K_MAIN
FOURTU = stage_c_inputs.DATASET_4TU

GAIN_RECIPES = {
    "G4": DetectionImageConfig(gain_mode=GainMode.NONE, recipe_id="G4"),
    "G2_b0.5": DetectionImageConfig(
        gain_mode=GainMode.AGC_CAPPED, agc_floor_beta=0.5, recipe_id="G2_b0.5"
    ),
}
"""E1: r0 with only the gain step changed (plan section 2-3 G4, G2)."""

FORBIDDEN_PATTERNS = (
    *(
        pattern
        for pattern in d0.FORBIDDEN_PATTERNS
        if pattern != "runs/eval/m32_recipe*"
    ),
    "data/aihub_yolo_v1/*",
)
"""The D0 forbidden list without the stage-C folder (E3 reads its table), plus
every AIHub crop, so no AIHub (test) image can be opened."""

STATUS_INPUT_ERROR = "input_error"
EXIT_CODES = {"ok": 0, STATUS_INPUT_ERROR: 2, "error": 3}
INPUT_ERRORS = (common.InputError, stage_c_inputs.InputError)

RECORD_HEADER = (
    "fold",
    "image_id",
    "experiment",
    "variant",
    "model",
    "line_id",
    "snr_db",
    "apex_trace",
    "apex_sample",
    "period_samples",
    "n_tiles",
    "n_line_boxes",
    "n_apex_hyperbola_boxes",
    "best_apex_hyperbola_score",
    "best_apex_any_class_score",
    "best_apex_any_class",
)

logger = logging.getLogger("m33_y0_diag_infer")


class Experiment(StrEnum):
    """Input variants of spec section 6."""

    E0 = "E0"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"


# ------------------------------------------------------------------ types


@dataclass(frozen=True, eq=False)
class SyntheticModel:
    """One gprMax model: its record, pipe-only field and truth apex (L0).

    Attributes:
        model_id: ``M01``-``M20``.
        radargram: The stand-alone record (the D0 Y0 input).
        pipe_only: float64 record minus the middle ``_bg`` trace.
        apex_trace: Truth apex trace (json ``apex.trace_index``).
        apex_sample: Truth apex sample (json envelope peak on the record axis).
    """

    model_id: str
    radargram: Radargram
    pipe_only: np.ndarray
    apex_trace: float
    apex_sample: float


@dataclass(frozen=True, eq=False)
class DiagImage:
    """One detector input line and the truth apex its records are matched to.

    Attributes:
        experiment: E0-E4.
        variant: ``r0``, ``G4``, ``G2_b0.5``, ``snr<s>`` or ``centred``.
        model_id: gprMax model.
        line_id: The model id, or the 4TU background line for E3.
        image: Line L1 image (E4: the E0 image, moved only at inference).
        apex_trace: Truth apex trace (L0).
        apex_sample: Truth apex sample (L0).
        content_shift_px: E4: columns the content moves right inside the tile.
        snr_db: E2 and E3 SNR step, else None.
        details: Per-image values written to the json (gain, noise, clipping).
    """

    experiment: Experiment
    variant: str
    model_id: str
    line_id: str
    image: DetectionImage
    apex_trace: float
    apex_sample: float
    content_shift_px: int = 0
    snr_db: float | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def image_id(self) -> str:
        """Unique key of the image (records and json)."""
        return f"{self.experiment}/{self.variant}/{self.model_id}/{self.line_id}"

    @property
    def tile_indices(self) -> tuple[int, ...]:
        """Tiles whose columns reach the apex +- 3 traces (the match margin)."""
        traces_per_px = self.image.grid_map.traces_per_px
        margin = d0.MATCH_TRACE_MARGIN
        first_px = (self.apex_trace - margin) / traces_per_px
        end_px = (self.apex_trace + margin + 1) / traces_per_px
        return tuple(
            tile.index
            for tile in self.image.tiles
            if tile.x0_px < end_px and first_px < tile.x0_px + tile.content_width_px
        )


@dataclass(frozen=True)
class StageCPosition:
    """A stage-C injection row used as an E3 position (L0 apex on a 4TU line)."""

    spec: inject.InjectionSpec

    @property
    def model_id(self) -> str:
        return self.spec.response_id

    @property
    def line_id(self) -> str:
        return self.spec.line_id


@dataclass(frozen=True, eq=False)
class InjectedLine:
    """A 4TU line with one untapered pipe response (E3).

    Attributes:
        radargram: Background plus ``gain`` x untapered field, int16 clipped.
        truth: Stage-C truth of the tapered injection (gain, apex, noise).
        n_clipped_cells: Cells of the untapered sum cut to the int16 range.
        n_aperture_taper: Taper columns at each aperture end that were undone.
    """

    radargram: Radargram
    truth: inject.InjectionTruth
    n_clipped_cells: int
    n_aperture_taper: int


@dataclass
class ImageSet:
    """Built images and the ones that could not be built (with the reason)."""

    images: list[DiagImage] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def add(self, other: "ImageSet") -> None:
        self.images.extend(other.images)
        self.skipped.extend(other.skipped)

    def skip(
        self, experiment: Experiment, model_id: str, line_id: str, reason: str
    ) -> None:
        self.skipped.append(
            {
                "experiment": experiment.value,
                "model": model_id,
                "line_id": line_id,
                "reason": reason,
            }
        )


@dataclass(frozen=True)
class ApexScores:
    """What the merged boxes of one image and fold say about the apex.

    Attributes:
        n_line_boxes: Merged boxes that reach the L0 grid (any class).
        n_apex_hyperbola_boxes: ``pipe_hyperbola`` boxes containing the apex.
        best_apex_hyperbola_score: Highest of them, None when there is none.
        best_apex_any_class_score: Highest score of any class containing it.
        best_apex_any_class: Class of that box.
    """

    n_line_boxes: int
    n_apex_hyperbola_boxes: int
    best_apex_hyperbola_score: float | None
    best_apex_any_class_score: float | None
    best_apex_any_class: str | None


@dataclass(frozen=True)
class TileJob:
    """One inferred tile: image position in the run list and tile index."""

    image_index: int
    tile_index: int


@dataclass(frozen=True)
class RunSettings:
    """Command-line choices of one run."""

    out_root: Path
    folds: tuple[int, ...]
    models: tuple[str, ...]
    experiments: tuple[Experiment, ...]
    device: str
    e3_positions_per_model: int
    overwrite: bool
    images_only: bool


# ------------------------------------------------------------ input guard


def _folder_key(path: Path) -> str:
    # normcase: Windows paths that differ only in case name the same folder.
    return os.path.normcase(str(Path(path).resolve()))


def is_inside(path: Path, root: Path) -> bool:
    """``path`` is ``root`` or lies below it (case-insensitive on Windows)."""
    path_key, root_key = _folder_key(path), _folder_key(root)
    return path_key == root_key or path_key.startswith(root_key + os.sep)


@dataclass
class InputGuard:
    """Refuses forbidden inputs; outside the repository only the listed roots.

    Attributes:
        external_roots: Dataset folders outside the repository that may be
            read (the 4TU package for E3).
        opened: Paths checked so far (``run_identity``).
    """

    external_roots: tuple[Path, ...] = ()
    opened: list[str] = field(default_factory=list)

    def check(self, path: Path) -> Path:
        """Return ``path`` resolved when it may be read.

        Raises:
            common.InputError: For a forbidden path or one outside the
                repository and the external roots (exit 2).
        """
        resolved = Path(path).resolve()
        if is_inside(resolved, REPO):
            name = resolved.relative_to(REPO.resolve()).as_posix().lower()
            if any(
                fnmatch.fnmatchcase(name, pattern) for pattern in FORBIDDEN_PATTERNS
            ):
                raise common.InputError(f"forbidden input: {name}")
        elif any(is_inside(resolved, root) for root in self.external_roots):
            name = str(resolved)
        else:
            raise common.InputError(f"input outside the allowed roots: {resolved}")
        if name not in self.opened:
            self.opened.append(name)
        return resolved


def check_out_root(out_root: Path, overwrite: bool) -> None:
    """Refuse the D0 / D0' folders (and anything below them) and existing output.

    Raises:
        common.InputError: For a protected folder, or an existing one without
            ``--overwrite`` (files are then rewritten, nothing is deleted).
    """
    for protected in PROTECTED_OUT_ROOTS:
        if is_inside(out_root, protected):
            raise common.InputError(f"{out_root} is inside the read-only {protected}")
    if out_root.exists() and not overwrite:
        raise common.InputError(f"output exists (use --overwrite): {out_root}")


# --------------------------------------------------------- image building


def load_synthetic_model(model_id: str, guard: InputGuard) -> SyntheticModel:
    """Record, pipe-only field and apex of one model, as D0 reads them."""
    for suffix in (".npy", "_bg.npy", ".json"):
        guard.check(SYNTHETIC_ROOT / f"{model_id}{suffix}")
    radargram = load_synthetic(model_id, root=SYNTHETIC_ROOT)
    truth = load_synthetic_truth(model_id, root=SYNTHETIC_ROOT)
    dt_ns = radargram.dt_ns
    if dt_ns is None:
        raise common.InputError(f"{model_id}: synthetic record without dt_ns")
    # Same apex as d0.synthetic_lines: the json envelope peak on the record axis.
    origin_ns = float(radargram.metadata.get("time_of_sample0_ns", 0.0))
    return SyntheticModel(
        model_id=model_id,
        radargram=radargram,
        pipe_only=y0_diag.pipe_only(radargram, model_id),
        apex_trace=float(truth.apex_trace_index),
        apex_sample=(truth.apex_envelope_peak_ns - origin_ns) / dt_ns,
    )


def e0_image(model: SyntheticModel) -> DiagImage:
    """The D0 Y0 image: ``make_detection_image`` with the r0 defaults."""
    return DiagImage(
        Experiment.E0,
        "r0",
        model.model_id,
        model.model_id,
        make_detection_image(model.radargram),
        model.apex_trace,
        model.apex_sample,
    )


def e1_images(model: SyntheticModel) -> list[DiagImage]:
    """The E0 record through the G4 and G2 (beta 0.5) gain recipes."""
    images = []
    for variant, config in GAIN_RECIPES.items():
        image = make_detection_image(model.radargram, config)
        images.append(
            DiagImage(
                Experiment.E1,
                variant,
                model.model_id,
                model.model_id,
                image,
                model.apex_trace,
                model.apex_sample,
            )
        )
    return images


def unit_noise(model_id: str, shape: tuple[int, int]) -> np.ndarray:
    """Standard normal field of one model (seed 20261115 and the model id)."""
    model_key = zlib.crc32(model_id.encode("utf-8"))
    return np.random.default_rng([SEED, model_key]).standard_normal(shape)


def linear_steps(radargram: Radargram, data: np.ndarray) -> np.ndarray:
    """Steps (1)-(3) (DC, dewow, band-pass) of ``data`` on the record's axis."""
    record = dataclasses.replace(radargram, data=np.asarray(data, dtype=np.float32))
    return prepare_fitting_branch(record).filtered


def apex_envelope_peak(
    section: np.ndarray, apex_trace: float, apex_sample: float, period_samples: float
) -> float:
    """Envelope maximum of the apex column within apex +- 1 P (stage-C ``A``)."""
    n_samples = section.shape[0]
    column = np.asarray(section[:, round(apex_trace)], dtype=np.float64)
    # Padded to 2 N like inject._envelope, so the record end does not wrap.
    envelope = np.abs(hilbert(column, N=2 * n_samples)[:n_samples])
    first = max(0, math.floor(apex_sample - period_samples))
    last = min(n_samples - 1, math.ceil(apex_sample + period_samples))
    return float(envelope[first : last + 1].max())


def noise_scale(signal_amplitude: float, unit_noise_rms: float, snr_db: float) -> float:
    """Factor on the unit noise that gives ``snr_db`` after the linear steps."""
    return signal_amplitude / (10.0 ** (snr_db / 20.0) * unit_noise_rms)


def e2_images(model: SyntheticModel, period_samples: float) -> list[DiagImage]:
    """Pipe-only response plus scaled white noise, r0 with the E0 period.

    The linear steps are linear, so the band-passed noise RMS is the scale
    times the band-passed unit-noise RMS and the SNR is exact.
    """
    filtered_pipe = linear_steps(model.radargram, model.pipe_only)
    signal = apex_envelope_peak(
        filtered_pipe, model.apex_trace, model.apex_sample, period_samples
    )
    noise = unit_noise(model.model_id, model.pipe_only.shape)
    unit_rms = float(np.sqrt(np.mean(linear_steps(model.radargram, noise) ** 2)))
    images = []
    for snr_db in SNR_STEPS_DB:
        scale = noise_scale(signal, unit_rms, snr_db)
        noisy = (model.pipe_only + scale * noise).astype(np.float32)
        record = dataclasses.replace(model.radargram, data=noisy)
        image = make_detection_image(record, period_samples_override=period_samples)
        details = {
            "signal_amplitude": signal,
            "unit_noise_rms": unit_rms,
            "noise_scale": scale,
        }
        images.append(
            DiagImage(
                Experiment.E2,
                f"snr{snr_db:g}",
                model.model_id,
                model.model_id,
                image,
                model.apex_trace,
                model.apex_sample,
                snr_db=snr_db,
                details=details,
            )
        )
    return images


def e4_image(e0: DiagImage) -> DiagImage | None:
    """E0 with the content centred in its tile; None for a line wider than a tile."""
    if len(e0.image.tiles) != 1:
        return None
    shift_px = (e0.image.tile_px - e0.image.width_px) // 2
    return dataclasses.replace(
        e0,
        experiment=Experiment.E4,
        variant="centred",
        content_shift_px=shift_px,
        details={"content_shift_px": shift_px},
    )


def synthetic_model_images(
    model: SyntheticModel, experiments: Sequence[Experiment]
) -> ImageSet:
    """E0, E1, E2 and E4 images of one model (those in ``experiments``)."""
    e0 = e0_image(model)
    found = ImageSet()
    if Experiment.E0 in experiments:
        found.images.append(e0)
    if Experiment.E1 in experiments:
        found.images.extend(e1_images(model))
    if Experiment.E2 in experiments:
        found.images.extend(e2_images(model, e0.image.period_samples))
    if Experiment.E4 in experiments:
        centred = e4_image(e0)
        if centred is None:
            found.skip(Experiment.E4, model.model_id, model.model_id, "multi_tile_line")
        else:
            found.images.append(centred)
    return found


def stage_c_positions(
    specs: Sequence[inject.InjectionSpec],
    fourtu_line_ids: set[str],
    models: Sequence[str],
    per_model: int,
) -> list[StageCPosition]:
    """First ``per_model`` 4TU positions of each model, in table order.

    Only hyperbola rows with polarity +1 and k 1.3 count; every (line, apex)
    pair is taken once (the table repeats it per SNR and polarity).
    """
    chosen: dict[str, list[StageCPosition]] = {model: [] for model in models}
    seen: set[tuple[str, str, float]] = set()
    for spec in specs:
        picked = chosen.get(spec.response_id)
        if picked is None or len(picked) >= per_model:
            continue
        if spec.line_id not in fourtu_line_ids:
            continue
        if spec.mode is not inject.InjectionMode.HYPERBOLA:
            continue
        if spec.polarity != E3_POLARITY or spec.k != E3_K:
            continue
        key = (spec.response_id, spec.line_id, spec.apex_trace)
        if key not in seen:
            seen.add(key)
            picked.append(StageCPosition(spec))
    return [position for model in models for position in chosen[model]]


def untapered_field(patch: inject.ResponsePatch) -> np.ndarray:
    """``patch.field`` with the aperture taper divided out (window-end taper kept)."""
    weights = inject.edge_taper_weights(patch.response.n_traces, patch.n_aperture_taper)
    return patch.field / weights[None, :]


def inject_without_taper(
    context: inject.LineContext,
    response: inject.PipeResponse,
    spec: inject.InjectionSpec,
) -> InjectedLine | str:
    """Stage-C injection of one row, then the same gain on the untapered field.

    ``inject_realization`` decides acceptance, gain (SNR) and the truth with
    the tapered field, exactly as in stage C; the line is then rebuilt from
    the background with that gain on the untapered field.

    Returns:
        The injected line, or the stage-C rejection reason.
    """
    result = inject.inject_realization(
        context, [spec], {response.response_id: response}
    )
    if result.rejected:
        return result.rejected[0][1]
    truth = result.truths[0]
    patch = inject.build_response_patch(response, context, spec.k)
    if not isinstance(patch, inject.ResponsePatch):
        raise RuntimeError(f"{spec.injection_id}: patch rejected after injection")
    data = np.asarray(context.background.data, dtype=np.float64).copy()
    offset = round(truth.apex_trace - response.apex_trace)
    columns = slice(offset, offset + response.n_traces)
    summed = data[:, columns] + truth.gain * untapered_field(patch)[: data.shape[0]]
    is_clipped = (summed < inject.INT16_MIN) | (summed > inject.INT16_MAX)
    data[:, columns] = np.clip(summed, inject.INT16_MIN, inject.INT16_MAX)
    return InjectedLine(
        radargram=dataclasses.replace(context.background, data=data.astype(np.float32)),
        truth=truth,
        n_clipped_cells=int(np.count_nonzero(is_clipped)),
        n_aperture_taper=patch.n_aperture_taper,
    )


def e3_images(
    position: StageCPosition,
    context: inject.LineContext,
    response: inject.PipeResponse,
) -> ImageSet:
    """The untapered injection of one position at every SNR step, r0."""
    found = ImageSet()
    for snr_db in SNR_STEPS_DB:
        realization_id = f"{position.line_id}:e3:snr{snr_db:g}"
        spec = dataclasses.replace(
            position.spec,
            snr_db=snr_db,
            realization_id=realization_id,
            injection_id=f"{realization_id}:{position.model_id}",
        )
        outcome = inject_without_taper(context, response, spec)
        if isinstance(outcome, str):
            found.skip(Experiment.E3, position.model_id, position.line_id, outcome)
            continue
        truth = outcome.truth
        details = {
            "gain": truth.gain,
            "signal_amplitude": truth.signal_amplitude,
            "noise_rms": truth.noise_rms,
            "n_clipped_cells_untapered": outcome.n_clipped_cells,
            "n_clipped_cells_stage_c": truth.n_clipped_cells,
            "n_aperture_taper_columns": outcome.n_aperture_taper,
            "background_period_samples": context.period_samples,
        }
        found.images.append(
            DiagImage(
                Experiment.E3,
                f"snr{snr_db:g}",
                position.model_id,
                position.line_id,
                make_detection_image(outcome.radargram),
                truth.apex_trace,
                truth.apex_sample,
                snr_db=snr_db,
                details=details,
            )
        )
    return found


def read_stage_c_positions(
    models: Sequence[str], per_model: int, guard: InputGuard
) -> list[StageCPosition]:
    """E3 positions from the F2-fixed stage-C table and the stage-A 4TU list.

    Raises:
        common.InputError: When the table is not the F2 one.
        stage_c_inputs.InputError: When the 4TU list is not the plan one.
    """
    specs, table_sha256 = inject.read_injection_table(guard.check(STAGE_C_TABLE))
    f2_record = json.loads(guard.check(STAGE_C_F2_RECORD).read_text(encoding="utf-8"))
    if table_sha256 != f2_record["injection_table_sha256"]:
        raise common.InputError(f"stage-C table sha256 {table_sha256} is not F2")
    lines_csv = guard.check(stage_c_inputs.STAGE_A_4TU)
    stage_c_inputs.checked_sha256(
        lines_csv, stage_c_inputs.EXPECTED_SHA256["stage_a_4tu"], "stage-A 4TU list"
    )
    fourtu_line_ids = {row["line"] for row in stage_c_inputs.read_csv(lines_csv)}
    return stage_c_positions(specs, fourtu_line_ids, models, per_model)


def build_e3_images(
    models: Sequence[str], per_model: int, guard: InputGuard
) -> ImageSet:
    """E3 images of every model with a stage-C 4TU position."""
    found = ImageSet()
    positions = read_stage_c_positions(models, per_model, guard)
    placed_models = {position.model_id for position in positions}
    for model_id in models:
        if model_id not in placed_models:
            found.skip(Experiment.E3, model_id, "", "no_stage_c_4tu_position")
    if not positions:
        return found
    for model_id in sorted(placed_models):
        files = responses_io.model_files(model_id)
        for path in (files.bscan, files.background, files.metadata):
            guard.check(path)
    library = responses_io.load_library(sorted(placed_models))
    for position in positions:
        line = stage_c_inputs.LineRef(FOURTU, position.line_id, "background")
        context = inject.prepare_line_context(stage_c.load_radargram(line))
        found.add(e3_images(position, context, library.air[position.model_id]))
    return found


def build_images(settings: RunSettings, guard: InputGuard) -> ImageSet:
    """Every image of the selected experiments and models (CPU)."""
    found = ImageSet()
    synthetic_experiments = [e for e in settings.experiments if e is not Experiment.E3]
    if synthetic_experiments:
        for model_id in settings.models:
            model = load_synthetic_model(model_id, guard)
            found.add(synthetic_model_images(model, synthetic_experiments))
    if Experiment.E3 in settings.experiments:
        found.add(
            build_e3_images(settings.models, settings.e3_positions_per_model, guard)
        )
    return found


# -------------------------------------------------------------- inference


def load_fold_detector(fold: int, device: str) -> object:
    """N2-a' ``last.pt`` of ``fold`` at conf 0.001 (weights asserted before)."""
    # Lazy: ultralytics stays inside gpr_detection.detect (AGPL boundary).
    from gpr_detection.detect.yolo import load_yolo_detector
    from gpr_detection.detect.yolo_logging import quiet_ultralytics_logger

    quiet_ultralytics_logger()
    return load_yolo_detector(
        stage_c.weights_path(RUN_ROOT, fold),
        input_size=INPUT_PX,
        confidence_threshold=CONFIDENCE,
        iou_threshold=NMS_IOU,
        device=device,
    )


def tile_jobs(images: Sequence[DiagImage]) -> list[TileJob]:
    """Every inferred tile of every image, in image order."""
    return [
        TileJob(index, tile_index)
        for index, item in enumerate(images)
        for tile_index in item.tile_indices
    ]


def tile_pixels(item: DiagImage, tile_index: int) -> np.ndarray:
    """The tile the detector sees; E4 moves the content right by its shift."""
    pixels = item.image.tile_image(tile_index)
    shift_px = item.content_shift_px
    if shift_px == 0:
        return pixels
    width_px = item.image.tiles[tile_index].content_width_px
    moved = np.full_like(pixels, item.image.pad_gray)
    moved[:, shift_px : shift_px + width_px] = pixels[:, :width_px]
    return moved


def shifted_back(box: LineBox, shift_px: int) -> LineBox:
    """A box of a moved tile in the line L1 coordinates of the unmoved content."""
    if shift_px == 0:
        return box
    moved = box.box
    original = PixelBox(
        x1=moved.x1 - shift_px, y1=moved.y1, x2=moved.x2 - shift_px, y2=moved.y2
    )
    return dataclasses.replace(box, box=original)


def contains_apex(grid_box: GridBox, item: DiagImage) -> bool:
    """D0 / M3-2 containment (+-3 traces, +-0.5 ``P_line``) for any class."""
    # d0.apex_hit applies the rule to pipe_hyperbola boxes only, so the box is
    # passed under that class to test the geometry alone.
    as_hyperbola = [(int(DetectionClass.PIPE_HYPERBOLA), grid_box)]
    return d0.apex_hit(
        as_hyperbola, item.apex_trace, item.apex_sample, item.image.period_samples
    )


def apex_scores(item: DiagImage, tile_boxes: Sequence[LineBox]) -> ApexScores:
    """Merge the tile boxes (R2 + V1/V2) and score the ones containing the apex."""
    pairs = stage_c.merge_to_grid(item.image, tile_boxes)
    hyperbola_scores = []
    best_any: LineBox | None = None
    for box, grid_box in pairs:
        if not contains_apex(grid_box, item):
            continue
        if box.class_id == DetectionClass.PIPE_HYPERBOLA:
            hyperbola_scores.append(float(box.score))
        if best_any is None or box.score > best_any.score:
            best_any = box
    return ApexScores(
        n_line_boxes=len(pairs),
        n_apex_hyperbola_boxes=len(hyperbola_scores),
        best_apex_hyperbola_score=max(hyperbola_scores, default=None),
        best_apex_any_class_score=None if best_any is None else float(best_any.score),
        best_apex_any_class=(
            None if best_any is None else common.CLASS_NAMES[best_any.class_id]
        ),
    )


def record_row(fold: int, item: DiagImage, scores: ApexScores) -> dict[str, Any]:
    """One ``records.csv`` row; floats stay Python floats (full precision)."""
    return {
        "fold": fold,
        "image_id": item.image_id,
        "experiment": item.experiment.value,
        "variant": item.variant,
        "model": item.model_id,
        "line_id": item.line_id,
        "snr_db": item.snr_db,
        "apex_trace": float(item.apex_trace),
        "apex_sample": float(item.apex_sample),
        "period_samples": float(item.image.period_samples),
        "n_tiles": len(item.tile_indices),
        **dataclasses.asdict(scores),
    }


def score_fold(
    detector: object,
    images: Sequence[DiagImage],
    jobs: Sequence[TileJob],
    pixels: Sequence[np.ndarray],
    fold: int,
) -> list[dict[str, Any]]:
    """Infer every tile of one fold in shared batches and score each image."""
    outputs = stage_c.predict_tiles(detector, pixels, BATCH_SIZE)
    boxes: list[list[LineBox]] = [[] for _ in images]
    for job, output in zip(jobs, outputs, strict=True):
        item = images[job.image_index]
        for box in stage_c.tile_line_boxes(item.image, job.tile_index, output):
            boxes[job.image_index].append(shifted_back(box, item.content_shift_px))
    return [
        record_row(fold, item, apex_scores(item, found))
        for item, found in zip(images, boxes, strict=True)
    ]


# ----------------------------------------------------------------- report


def _json_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def image_entry(item: DiagImage) -> dict[str, Any]:
    """The json description of one image (fold independent)."""
    image = item.image
    return {
        "image_id": item.image_id,
        "experiment": item.experiment.value,
        "variant": item.variant,
        "model": item.model_id,
        "line_id": item.line_id,
        "snr_db": item.snr_db,
        "apex_trace": float(item.apex_trace),
        "apex_sample": float(item.apex_sample),
        "period_samples": float(image.period_samples),
        "period_source": str(image.period_source),
        "recipe_id": image.recipe_id,
        "width_px": image.width_px,
        "content_height_px": image.content_height_px,
        "n_line_tiles": len(image.tiles),
        "tile_indices": list(item.tile_indices),
        "content_shift_px": item.content_shift_px,
        "flags": [str(flag) for flag in image.flags],
        "details": {key: _json_value(value) for key, value in item.details.items()},
    }


def variant_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Per ``experiment/variant``: records, with an apex box, above conf 0.25."""
    counts: dict[str, Counter] = {}
    for record in records:
        key = f"{record['experiment']}/{record['variant']}"
        cell = counts.setdefault(key, Counter())
        best = record["best_apex_hyperbola_score"]
        cell["records"] += 1
        cell["apex_hyperbola_box"] += int(best is not None)
        cell["above_conf_line"] += int(best is not None and best > CONF_LINE)
    return {key: dict(cell) for key, cell in sorted(counts.items())}


def compare_e0_with_d0(
    records: Sequence[Mapping[str, Any]], guard: InputGuard
) -> dict[str, Any]:
    """E0 best score above 0.25 against the D0 Y0 ``hit_S0`` (same model, fold).

    Report only: D0 ran conf 0.25 in other batches, so a score near 0.25 may
    fall on the other side (batch-dependent float differences).
    """
    if not D0_APEX_CSV.is_file():
        return {"available": False}
    rows = stage_c_inputs.read_csv(guard.check(D0_APEX_CSV))
    d0_hits = {
        (int(row["fold"]), row["model"]): row["hit_S0"] == "1"
        for row in rows
        if row["condition"] == "Y0"
    }
    compared = 0
    mismatches = []
    for record in records:
        key = (record["fold"], record["model"])
        if record["experiment"] != Experiment.E0.value or key not in d0_hits:
            continue
        compared += 1
        best = record["best_apex_hyperbola_score"]
        e0_hit = best is not None and best > CONF_LINE
        if e0_hit != d0_hits[key]:
            mismatches.append(
                {
                    "fold": key[0],
                    "model": key[1],
                    "d0_hit": d0_hits[key],
                    "e0_best_score": best,
                }
            )
    return {
        "available": True,
        "rule": f"E0 best_apex_hyperbola_score > {CONF_LINE} vs D0 Y0 hit_S0",
        "n_compared": compared,
        "mismatches": mismatches,
    }


def settings_record(settings: RunSettings) -> dict[str, Any]:
    """Settings and fixed constants of the run (json)."""
    return {
        "spec": SPEC,
        "seed": SEED,
        "confidence": CONFIDENCE,
        "nms_iou": NMS_IOU,
        "input_px": INPUT_PX,
        "batch_size": BATCH_SIZE,
        "match_trace_margin": d0.MATCH_TRACE_MARGIN,
        "match_sample_periods": d0.MATCH_SAMPLE_PERIODS,
        "snr_steps_db": list(SNR_STEPS_DB),
        "gain_recipes": {
            name: {
                "gain_mode": str(config.gain_mode),
                "agc_window_periods": config.agc_window_periods,
                "agc_floor_beta": config.agc_floor_beta,
            }
            for name, config in GAIN_RECIPES.items()
        },
        "e3_table": STAGE_C_TABLE.relative_to(REPO).as_posix(),
        "e3_polarity": E3_POLARITY,
        "e3_k": E3_K,
        "e3_positions_per_model": settings.e3_positions_per_model,
        "folds": list(settings.folds),
        "models": list(settings.models),
        "experiments": [str(experiment) for experiment in settings.experiments],
        "device": settings.device,
        "images_only": settings.images_only,
    }


def code_paths() -> list[Path]:
    """Code whose git blobs identify the run."""
    return [
        Path(__file__).resolve(),
        Path(d0.__file__).resolve(),
        Path(stage_c.__file__).resolve(),
        Path(responses_io.__file__).resolve(),
        Path(y0_diag.__file__).resolve(),
        REPO / "src" / "gpr_detection" / "preprocess" / "to_image.py",
        REPO / "src" / "gpr_detection" / "eval" / "inject.py",
        REPO / "src" / "gpr_detection" / "pipeline.py",
    ]


def new_run_identity(
    settings: RunSettings, weights_sha256: Mapping[str, str]
) -> dict[str, Any]:
    """Git state, versions, GPU, weights and code blobs of the run."""
    identity = common.new_run_identity()
    for unused in ("eval_list_sha256", "g1_inputs_sha256"):
        del identity[unused]
    identity["gpu"] = None if settings.images_only else common.gpu_name(settings.device)
    identity["weights_sha256"] = dict(weights_sha256)
    identity["code_blobs"] = d0.git_blobs(code_paths())
    identity["synthetic_sha256"] = {
        path.name: common.sha256_file(path)
        for model_id in settings.models
        for path in (
            SYNTHETIC_ROOT / f"{model_id}{suffix}"
            for suffix in (".npy", "_bg.npy", ".json")
        )
    }
    return identity


def write_outputs(
    out_root: Path, records: Sequence[Mapping[str, Any]], document: Mapping
) -> None:
    rows = ([record[name] for name in RECORD_HEADER] for record in records)
    common.write_csv(out_root / "records.csv", RECORD_HEADER, rows)
    common.write_json(out_root / "diag_infer.json", dict(document))


# --------------------------------------------------------------------- run


def run(
    settings: RunSettings,
    guard: InputGuard,
    detector_for_fold: Callable[[int], object],
    weights_sha256: Mapping[str, str],
) -> dict[str, int]:
    """Build the images, infer every fold, write the outputs.

    Returns:
        ``{"n_images", "n_records", "n_tiles"}`` for the closing line
        (``n_tiles`` per fold).
    """
    started = time.time()
    identity = new_run_identity(settings, weights_sha256)
    image_set = build_images(settings, guard)
    images = image_set.images
    jobs = tile_jobs(images)
    seconds_build = time.time() - started
    print(
        f"images: {len(images)} ({len(jobs)} tiles per fold, "
        f"{len(image_set.skipped)} skipped, {seconds_build:.0f} s)",
        flush=True,
    )
    records: list[dict[str, Any]] = []
    if not settings.images_only:
        pixels = [tile_pixels(images[job.image_index], job.tile_index) for job in jobs]
        for fold in settings.folds:
            records += score_fold(detector_for_fold(fold), images, jobs, pixels, fold)
            elapsed = time.time() - started
            print(f"fold {fold}: {len(jobs)} tiles ({elapsed:.0f} s)", flush=True)
    identity["opened_paths"] = list(guard.opened)
    identity["finished_at"] = common.local_now()
    document = {
        "schema": SCHEMA,
        "status": "ok",
        "settings": settings_record(settings),
        "run_identity": identity,
        "images": [image_entry(item) for item in images],
        "skipped": image_set.skipped,
        "variant_counts": variant_counts(records),
        "e0_vs_d0": compare_e0_with_d0(records, guard),
        "resources": {
            "seconds_build": round(seconds_build, 1),
            "seconds_total": round(time.time() - started, 1),
            "peak_vram_reserved_mb": (
                None if settings.images_only else stage_c.peak_vram_mb(settings.device)
            ),
        },
    }
    write_outputs(settings.out_root, records, document)
    print(f"wrote {settings.out_root}", flush=True)
    return {"n_images": len(images), "n_records": len(records), "n_tiles": len(jobs)}


def _choices(text: str, allowed: Sequence[str], what: str) -> tuple[str, ...]:
    """Comma list of ``allowed`` values; duplicates and unknown values refused."""
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = sorted(set(values) - set(allowed))
    if not values or unknown or len(set(values)) != len(values):
        raise common.InputError(f"bad --{what} {text!r} (unknown: {unknown})")
    return values


def parse_args(argv: Sequence[str] | None) -> RunSettings:
    """Command line -> settings.

    Raises:
        common.InputError: For unknown or repeated folds, models or experiments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--folds", default=",".join(map(str, FOLDS)))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--experiments", default=",".join(Experiment))
    parser.add_argument("--device", default="0", help="'0' (GPU) or 'cpu'")
    parser.add_argument(
        "--e3-positions",
        type=int,
        default=E3_POSITIONS_PER_MODEL,
        help="stage-C 4TU positions per model for E3 (table order)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="build and list the images (CPU, no weights, no detector)",
    )
    args = parser.parse_args(argv)
    if args.e3_positions < 1:
        raise common.InputError("--e3-positions must be at least 1")
    experiments = _choices(args.experiments, tuple(Experiment), "experiments")
    return RunSettings(
        out_root=args.out_root,
        folds=d0.parse_folds(args.folds),
        models=_choices(args.models, MODELS, "models"),
        experiments=tuple(Experiment(name) for name in experiments),
        device=args.device,
        e3_positions_per_model=args.e3_positions,
        overwrite=args.overwrite,
        images_only=args.images_only,
    )


def _prepare(settings: RunSettings) -> tuple[InputGuard, dict[str, str]]:
    """Input guard (4TU root for E3) and the asserted weights of every fold."""
    external_roots: tuple[Path, ...] = ()
    if Experiment.E3 in settings.experiments:
        external_roots = (load_dataset_paths().root(DatasetName.FOURTU),)
    guard = InputGuard(external_roots=external_roots)
    if settings.images_only:
        return guard, {}
    for fold in settings.folds:
        guard.check(stage_c.weights_path(RUN_ROOT, fold))
    return guard, stage_c.assert_weights(RUN_ROOT, settings.folds)


def print_final_line(
    status: str, out_root: Path | None, counts: Mapping, started: float
) -> None:
    """Closing stdout JSON line (counts and paths only)."""
    line = {
        "status": status,
        "out_dir": None if out_root is None else str(out_root),
        "n_images": counts.get("n_images"),
        "n_records": counts.get("n_records"),
        "n_tiles": counts.get("n_tiles"),
        "elapsed_s": round(time.time() - started, 1),
    }
    print(json.dumps(line, ensure_ascii=False), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry; returns the exit code (0 done, 2 refused, 3 error)."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    started = time.time()
    out_root = None
    counts: dict[str, int] = {}
    try:
        settings = parse_args(argv)
        out_root = settings.out_root
        check_out_root(settings.out_root, settings.overwrite)
        y0_diag.below_normal()
        guard, weights_sha256 = _prepare(settings)
        detector_for_fold = partial(load_fold_detector, device=settings.device)
        counts = run(settings, guard, detector_for_fold, weights_sha256)
        status = "ok"
    except INPUT_ERRORS as error:
        logger.error("input error: %s", error)
        status = STATUS_INPUT_ERROR
    except Exception:
        logger.exception("internal error")
        status = "error"
    print_final_line(status, out_root, counts, started)
    return EXIT_CODES[status]


if __name__ == "__main__":
    sys.exit(main())
