"""M3-3 pre-close item 3 (E3x) runner: plan, un-injected 0.001 and injected stages.

Spec: ``docs/processing/m33_e3x_plan.md`` v0.5 (seal 1 was v0.3 at 72b0140;
seal 2 was v0.4 at 729d19c; v0.5 takes effect at seal 2 r1) §2, §3-1, §5 and
appendix A; cv conditions ``docs/experiments/m33_e3x_cv_review.md``; EM
conditions ``docs/processing/m33_e3x_em_review.md``; v0.5 review
``docs/processing/m33_e3x_texture_defect_review.md`` §10. Common M3-3 runner
rules (full precision, sealed PP, judgeability, end tree, LF sha) come from
``m33_inputs`` and ``m32_recipe_eval``.

v0.5 (after the injected first try ended ``e3x_defect``): the texture check
uses the length-n DFT energy identity (``m33_e3x_place.texture_check``); the
precheck runs check 10 (the five CPU assertions on every pool line) right
after check 1 and before the GPU checks 2-9; the detector stages write only
to a retry root (``runs/eval/m33/e3x_retry``, then ``e3x_retry<k>``); the
seal record is ``m33_e3x_seal2_r1.json`` and the plan tables are still
read from ``runs/eval/m33/e3x/plan`` (``SEALED_PLAN_DIR``, seal 2 sha).

Stages (folders under ``--out-root``, default ``runs/eval/m33/e3x``)

1. ``--stage plan`` (CPU, no detector) -> ``plan/``: zone (c) from the R2
   (4TU, ``original``, PP off) and R3 (IDS) un-injected boxes; per pool line
   the allowed hyperbola (model, apex) pairs with their ``r_edge(p)`` and the
   band positions (pure vertical shift, one taper per position); then the
   round-robin hyperbola choice over all lines. Writes
   ``hyperbola_positions.csv``, ``band_positions.csv``, ``band_arms.csv``,
   ``lines.csv``, ``exclusions.csv``, ``plan_report.json`` (counts only) and
   ``seal2_record_draft.json``. Seal 2 copies the draft to
   ``analysis/detection/m33_e3x_seal2.json`` (tracked).
2. ``--stage uninjected001`` (GPU) -> ``uninjected001/``: conf 0.001 on the
   153 background lines un-injected; best pre-merge scores at every planned
   truth (R-d baseline, report only).
3. ``--stage injected --workers N`` (GPU + N CPU workers) -> ``injected/``:
   every realization of the plan (hyperbola taper on / off x SNR 6-30 x
   polarity; band rows x lengths x SNR 12-24 x polarity, three textures,
   ``e3x_p42`` re-imaging), one conf 0.001 inference in shared batches of 64,
   the operating boxes cut **before** the merge (``score > 0.25``), merge, S0
   (judgment) and S3 (report) from the same merged boxes.
4. ``--precheck --workers N`` (GPU) -> ``runs/eval/m33/e3x_precheck_r1/``
   (``--precheck-dir`` for ``_r<k>``; refused if the folder exists): §5-2
   check 1, then check 10 (CPU, N workers, every pool line, writes
   ``cpu_assertions.json`` and ``cpu_assertions_lines.csv``), then checks
   3 and 2, 4-9 on the two precheck lines. A failed check 1 stops there
   (exit 1); a failed check 10 skips 3 and 2, 4-9.

Detector stages (2, 3) refuse an output folder under the first-try root
``runs/eval/m33/e3x`` (exit 2, nothing read there); a v0.5 run passes
``--out-root runs/eval/m33/e3x_retry`` (``e3x_retry<k>`` for a later try).

Progress file (blind exception, value free): ``progress.json`` in the stage
folder (precheck: the precheck folder) with only ``stage``, ``done``,
``total``, ``unit``, ``started_at`` and ``updated_at``; ``total`` is the
planned line count. It is rewritten atomically after every line of check 10,
of the precheck GPU passes, of ``uninjected001`` and of ``injected``.

Pre-seal-2 plan: the plan runs before the seal-2 commit, while the E3x code
and the pool list are not tracked yet. For the plan stage only, ``not
tracked`` problems on exactly those files (``PRE_SEAL2_FILES``) neither stop
the start nor block judgeability; they are kept in
``run_identity.pre_seal2_untracked``. Every later stage needs them tracked.

Blind output (§5-3): recall, hits, coverage, scores, contrast and estimates
go to files only. Stdout carries stage, counts, seconds, paths and the NMS
warning and max-det counters; its last line is ``{"status", "out_dir",
"n_lines", "n_positions", "n_realizations", "n_inferences", "elapsed_s"}``
(``n_inferences``: images inferred, each by every fold).

Exit codes and ``status`` (the §5-4 failure table)

====  ===================  ==================================================
exit  status               meaning
====  ===================  ==================================================
0     ``ok``               finished, judgeable (plan: also pre-seal-2)
1     ``not_judgeable``    finished with a blocker (diagnostic arguments,
                           dirty tree at the start)
1     ``e3x_defect``       an implementation assertion failed (zero test,
                           ``shift_mismatch``, end-treatment or texture
                           check, hit without a 0.25 box, NMS time limit,
                           a tile with 300 boxes over 0.25, coverage)
1     ``precheck_failed``  ``--precheck``: a §5-2 check failed (check 10
                           included: an assertion or its count guard)
2     ``input_error``      input refused (sha, seal record, tree, sealed PP,
                           folds, output exists, an output folder under the
                           first-try root) or the tree changed
3     ``error``            internal error (GPU memory included)
====  ===================  ==================================================

Diagnostic arguments (never in a formal run, they make it not judgeable or
are refused there): ``--folds``, ``--only-lines``, ``--allow-dirty``,
``--overwrite`` (plan only, before seal 2), ``--out-root`` (in a formal run
only the retry folder ``runs/eval/m33/e3x_retry`` or ``e3x_retry<k>``),
``--precheck-dir`` (in a formal run only ``e3x_precheck_r<k>``),
``--run-root``, ``--r2-root``, ``--r3-root``, ``--seal-record``.
"""

import argparse
import contextlib
import hashlib
import io
import json
import logging
import math
import re
import shutil
import statistics
import sys
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from itertools import islice
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common
import m32_pipe_responses as responses_io
import m32_recipe_eval as runner
import m32_recipe_inputs as inputs
import m32_recipe_match as match
import m33_change_defs as change_defs
import m33_e3x_place as place
import m33_e3x_pool as pool
import m33_inputs
import m33_r3_band as band_runner
import m33_r3_p4 as p4

from gpr_detection.eval import inject
from gpr_detection.io.classes import DetectionClass
from gpr_detection.io.radargram import GridBox, Radargram
from gpr_detection.pipeline import (
    DEFAULT_MERGE_CONFIG,
    PP_SETS,
    LineBox,
    PPConfig,
    PPSetId,
    apply_pp,
    content_rows_for,
    image_box_to_grid_box,
    kept_after_pp,
    merge_line_boxes,
)
from gpr_detection.preprocess.timezero import ZeroResult, find_time_zero
from gpr_detection.preprocess.to_image import DetectionImage, make_detection_image

REPO = inputs.REPO
SCHEMA = "m33_e3x/1"
DEFAULT_OUT_ROOT = REPO / "runs" / "eval" / "m33" / "e3x"
RETRY_OUT_ROOT = REPO / "runs" / "eval" / "m33" / "e3x_retry"
DEFAULT_PRECHECK_DIR = REPO / "runs" / "eval" / "m33" / "e3x_precheck_r1"
"""v0.5 precheck folder (a new ``_r<k>`` per try; the first try's is kept)."""
SEAL2_RECORD = REPO / "analysis" / "detection" / "m33_e3x_seal2_r1.json"
"""Seal 2 r1 record (v0.5); ``m33_e3x_seal2.json`` stays as the first try's."""
R3_UNINJECTED_IDS = m33_inputs.R3_OUT_ROOT / runner.BAND_UNINJECTED_DIR
PLAN_DIR = "plan"
SEALED_PLAN_DIR = DEFAULT_OUT_ROOT / PLAN_DIR
"""Plan tables of seal 2, reused by seal 2 r1 (user decision v5-5)."""
PROGRESS_FILE = "progress.json"
PROGRESS_WRITE_ATTEMPTS = 3
PROGRESS_RETRY_S = 0.05

CONF_AP = m33_inputs.CONF_AP
CONF_OPERATING = m33_inputs.CONF_OPERATING
NMS_IOU = m33_inputs.NMS_IOU
INPUT_PX = runner.INPUT_PX
BATCH_SIZE = runner.DEFAULT_BATCH_SIZE
MAX_DET = 300
"""Ultralytics ``max_det`` of both conf paths (``YoloDetector`` passes none)."""
NMS_WARNING = re.compile(r"NMS time limit")
FOLDS = m33_inputs.EXPECTED_FOLDS
MAX_WORKERS = 8

HYPERBOLA_SNR_DB = (6.0, 12.0, 18.0, 30.0)
BAND_SNR_DB = (12.0, 18.0, 24.0)
TEXTURE_SNR_DB = (12.0, 18.0)
P42_SNR_DB = 18.0
POLARITIES = inject.POLARITIES
TAPER_ON = "on"
TAPER_OFF = "off"
TEXTURE_NONE = "none"
TEXTURE_AMPLITUDE = "amp"
TEXTURES = (TEXTURE_AMPLITUDE, "jitter0.1", "jitter0.2")
TEXTURE_JITTER = {"jitter0.1": 0.1, "jitter0.2": 0.2}
RECIPE_R0 = inputs.R0
RECIPE_P42 = "e3x_p42"
P42_TARGET_PX = 4.2

HYPERBOLA_TARGETS = {
    (inputs.DATASET_4TU, responses_io.ModelGroup.A0): 4,
    (inputs.DATASET_IDS, responses_io.ModelGroup.A0): 4,
    (inputs.DATASET_4TU, responses_io.ModelGroup.A1): 8,
}
"""Positions per model and background (plan §2-2: A0 4 + 4, A1 8 on 4TU)."""
HYPERBOLA_MODELS = (*responses_io.A0_MODELS, "M12", "M15", "M18")
"""A0 15 + A1 M12, M15, M18; K7, M13 and M20 are left out (plan §2-2)."""
BAND_MODELS: tuple[str, ...] | None = None
"""Band models: every library model valid on the line (``for_line``, §2-4)."""

PRE_SEAL2_FILES = (
    "analysis/detection/m33_e3x.py",
    "analysis/detection/m33_e3x_place.py",
    "analysis/detection/m33_e3x_pool.py",
    "analysis/detection/m33_e3x_judge.py",
    "analysis/detection/m33_e3x_pool.csv",
)
"""Files the plan may find untracked: it runs before the seal-2 commit."""

STATUS_OK = m33_inputs.STATUS_OK
STATUS_NOT_JUDGEABLE = m33_inputs.STATUS_NOT_JUDGEABLE
STATUS_INPUT_ERROR = m33_inputs.STATUS_INPUT_ERROR
STATUS_DEFECT = "e3x_defect"
STATUS_PRECHECK_FAILED = "precheck_failed"
STATUS_ERROR = "error"
EXIT_CODES = {
    STATUS_OK: 0,
    STATUS_NOT_JUDGEABLE: 1,
    STATUS_DEFECT: 1,
    STATUS_PRECHECK_FAILED: 1,
    STATUS_INPUT_ERROR: 2,
    STATUS_ERROR: 3,
}
FINAL_KEYS = (
    "status",
    "out_dir",
    "n_lines",
    "n_positions",
    "n_realizations",
    "n_inferences",
    "elapsed_s",
)
METRIC_WORDS = re.compile(
    r"recall|hit|cover|score|coh|c_lat|c_vert|estimate|delta|ci_|survival",
    re.IGNORECASE,
)
"""Words stdout must not carry (§5-3); the precheck asserts it (check 8)."""

CPU_ASSERTIONS = (
    "shift_mismatch",
    "off_differs_from_on",
    "r_edge_differs_from_plan",
    "texture_check",
    "zero_test",
)
"""The ``build_line`` defect keys precheck 10 counts on every line (§5-2 10)."""
PLAN_SOURCE_MAIN = "main"
PLAN_SOURCE_PRECHECK = "precheck"
CPU_ASSERTIONS_JSON = "cpu_assertions.json"
CPU_ASSERTIONS_LINES = "cpu_assertions_lines.csv"
GPU_PRECHECK_CHECKS = (
    "2_zero_test",
    "3_shift",
    "4_off_on_equal",
    "5_textures",
    "6_conf_paths_equal",
    "7_workers_bytes_equal",
    "8_stdout_blind",
    "9_nms_and_maxdet",
)
"""Checks 2-9: not run when check 10 fails (plan §5-2, v0.5)."""
RECORD_PERCENTILE = 90

logger = logging.getLogger("m33_e3x")


class Stage(StrEnum):
    """Run stages of plan §5-1."""

    PLAN = "plan"
    UNINJECTED001 = "uninjected001"
    INJECTED = "injected"


class E3xDefectError(RuntimeError):
    """An implementation assertion of §5-4 failed (``e3x_defect``, exit 1)."""


# ================================================================ settings


@dataclass(frozen=True)
class Settings:
    """Command-line choices of one run."""

    stage: Stage | None
    precheck: bool
    out_root: Path
    precheck_dir: Path
    device: str
    workers: int
    resume: bool
    overwrite: bool
    allow_dirty: bool
    folds: tuple[int, ...]
    only_lines: tuple[str, ...]
    run_root: Path
    r2_root: Path
    r3_root: Path
    pool_csv: Path
    seal_record: Path
    batch_size: int = BATCH_SIZE

    @property
    def stage_dir(self) -> Path:
        """Output folder of the stage (the precheck folder for ``--precheck``)."""
        if self.precheck:
            return self.precheck_dir
        assert self.stage is not None
        return self.out_root / self.stage.value

    @property
    def runs_models(self) -> bool:
        """Every stage but the plan runs the detector."""
        return self.precheck or self.stage is not Stage.PLAN

    def scope_blockers(self) -> list[str]:
        """Why the run scope is not a judgment input (empty for a full run)."""
        blockers = []
        if self.only_lines:
            blockers.append(m33_inputs.BLOCKER_LINE_LIMIT)
        if self.runs_models and not m33_inputs.has_all_folds(self.folds):
            blockers.append(m33_inputs.BLOCKER_PARTIAL_FOLDS)
        return blockers

    @property
    def folds_run(self) -> tuple[int, ...] | None:
        """Folds of a stage that runs models (``None`` for the plan)."""
        return self.folds if self.runs_models else None


@dataclass(frozen=True)
class Context:
    """Checks made before any input is read (start tree, sealed PP, R checks)."""

    start_tree: dict
    sealed_pp: dict
    r_checks: dict
    pre_seal2_untracked: tuple[str, ...] = ()


@dataclass
class Counts:
    """What the closing stdout line reports (no metric value)."""

    n_lines: int = 0
    n_positions: int = 0
    n_realizations: int = 0
    n_inferences: int = 0


def is_first_try_path(path: Path) -> bool:
    """``path`` is the first-try root ``runs/eval/m33/e3x`` or lies inside it."""
    return path.resolve().is_relative_to(DEFAULT_OUT_ROOT.resolve())


def assert_retry_output(settings: Settings) -> None:
    """Refuse a detector stage writing under the first-try root (v0.5, §5-4).

    The first try's ``uninjected001/`` and ``injected/`` are kept unopened;
    a retry writes to ``runs/eval/m33/e3x_retry`` (``e3x_retry<k>``).
    Checked before any input is read or any folder is made.

    Raises:
        inputs.InputError: For an output folder under the first-try root.
    """
    if is_first_try_path(settings.stage_dir):
        raise inputs.InputError(
            f"first-try output refused: {settings.stage_dir} "
            f"(use --out-root {RETRY_OUT_ROOT})"
        )


def _local_time(epoch_s: float) -> str:
    """Local time of an epoch, ISO 8601 with the UTC offset (as ``local_now``)."""
    # From an aware UTC time: a naive one goes through mktime, which Windows
    # refuses near the epoch.
    moment = datetime.fromtimestamp(epoch_s, tz=UTC).astimezone()
    return moment.isoformat(timespec="seconds")


@dataclass
class Progress:
    """Value-free progress file of a long stage (agent rule 12; plan §5-3 blind).

    The file holds only ``stage``, ``done``, ``total``, ``unit``,
    ``started_at`` and ``updated_at``. ``total`` is the planned count fixed
    before the run (plan lines); no judged input, metric or per-class count
    goes in. Each update replaces the file atomically (temporary file, then
    ``Path.replace``); a failed write is logged and the run goes on.
    """

    path: Path
    stage: str
    total: int
    unit: str = "lines"
    started_at: str = field(default_factory=common.local_now)

    def update(self, done: int) -> None:
        """Rewrite the file with ``done`` of ``total``."""
        document = {
            "stage": self.stage,
            "done": done,
            "total": self.total,
            "unit": self.unit,
            "started_at": self.started_at,
            "updated_at": common.local_now(),
        }
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        failure: OSError | None = None
        for _ in range(PROGRESS_WRITE_ATTEMPTS):
            try:
                common.write_json(temporary, document)
                temporary.replace(self.path)
                return
            except OSError as error:  # a reader may hold the file (Windows)
                failure = error
                time.sleep(PROGRESS_RETRY_S)
        logger.warning("progress file not updated: %s", type(failure).__name__)


# ============================================================ code and tree


SRC_CODE_FILES = runner.SRC_CODE_FILES


def code_paths() -> list[Path]:
    """E3x code and the reused modules: blobs at the start and the end."""
    here = Path(__file__).resolve().parent
    names = (
        "m33_e3x.py",
        "m33_e3x_place.py",
        "m33_e3x_pool.py",
        "m33_e3x_judge.py",
        "m31_d1_common.py",
        "m32_aihub_perturb.py",
        "m32_pipe_responses.py",
        "m32_recipe_eval.py",
        "m32_recipe_inputs.py",
        "m32_recipe_match.py",
        "m33_change_defs.py",
        "m33_inputs.py",
        "m33_r3_band.py",
        "m33_r3_p4.py",
    )
    return [here / name for name in names] + [REPO / name for name in SRC_CODE_FILES]


def tracked_inputs(settings: Settings) -> list[Path]:
    """Files git must track unchanged for this stage."""
    paths = [
        *code_paths(),
        settings.pool_csv,
        inputs.STAGE_A_4TU,
        inputs.STAGE_A_IDS,
        inputs.APEX_CANDIDATES,
        m33_inputs.R2_LINES_CSV,
        pool.FOURTU_LINES,
    ]
    if settings.precheck or settings.stage is not Stage.PLAN:
        paths.append(settings.seal_record)
    return paths


def measure_tree(settings: Settings) -> dict:
    """``m33_inputs.tree_state`` over this runner's inputs and code."""
    return m33_inputs.tree_state(tracked_inputs(settings), code_paths())


def pre_seal2_problems(problems: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split tree problems into the tolerated pre-seal-2 ones and the rest."""
    allowed = {f"not tracked: {name}" for name in PRE_SEAL2_FILES}
    tolerated = [problem for problem in problems if problem in allowed]
    rest = [problem for problem in problems if problem not in allowed]
    return tolerated, rest


def effective_tree(tree: Mapping, tolerated: Sequence[str]) -> dict:
    """The tree with the tolerated problems removed (judgment of the plan)."""
    rest = [problem for problem in tree["problems"] if problem not in tolerated]
    return {**tree, "problems": rest, "dirty": bool(rest)}


def start_context(settings: Settings) -> Context:
    """Fold ids, start tree, sealed PP and the R checks, before any input.

    Raises:
        inputs.InputError: For refused folds, a dirty tree without
            ``--allow-dirty``, another PP, settings, ``inject.py`` or r0.
    """
    m33_inputs.check_folds(settings.folds)
    tree = measure_tree(settings)
    tolerated: list[str] = []
    if settings.stage is Stage.PLAN and not settings.precheck:
        tolerated, _ = pre_seal2_problems(tree["problems"])
    effective = effective_tree(tree, tolerated)
    if effective["problems"] and not settings.allow_dirty:
        raise inputs.InputError(f"tree not clean: {effective['problems']}")
    sealed = m33_inputs.assert_sealed_pp(tree["code_blobs"])
    checks = {
        "settings": m33_inputs.assert_settings(CONF_OPERATING, CONF_AP, NMS_IOU),
        "inject_blob": m33_inputs.assert_inject_blob(tree["code_blobs"]),
        "r0_identity_sha256": m33_inputs.assert_r0_identity(),
    }
    return Context(effective, sealed, checks, tuple(tolerated))


def close_run(
    settings: Settings, context: Context, identity: dict, base_status: str
) -> tuple[str, dict]:
    """Measure the tree again before the results are written ([RUN] §1)."""
    end = effective_tree(measure_tree(settings), context.pre_seal2_untracked)
    identity["tree_end"] = end
    judgment = m33_inputs.judgment_record(
        settings.folds_run, settings.scope_blockers(), context.start_tree, end
    )
    if judgment["tree_changes"]:
        logger.error("tree changed during the run: %s", judgment["tree_changes"])
    return m33_inputs.final_status(base_status, judgment), judgment


def sealed_config(context: Context) -> tuple[PPSetId, PPConfig]:
    """The C0b set asserted at the start (S3; judgment uses S0)."""
    set_id = PPSetId(context.sealed_pp["pp_set_id"])
    return set_id, PP_SETS[set_id]


def new_identity(settings: Settings, context: Context, hashes: Mapping) -> dict:
    """``run_identity``: git, versions, GPU, seeds, inputs, tree, checks."""
    identity = common.new_run_identity()
    if settings.runs_models:
        identity["gpu"] = common.gpu_name(settings.device)
    identity.update(
        {
            "schema": SCHEMA,
            "seeds": {
                "pool": pool.POOL_SEED,
                "plan": place.PLAN_SEED_E3X,
                "texture": place.TEXTURE_SEED,
            },
            "inputs_sha256": dict(hashes),
            "merge_config": asdict(DEFAULT_MERGE_CONFIG),
            "conf_ap": CONF_AP,
            "conf_operating": CONF_OPERATING,
            "nms_iou": NMS_IOU,
            "batch_size": settings.batch_size,
            "max_det": MAX_DET,
            "workers": settings.workers,
            "dirty": context.start_tree["dirty"],
            "tree_problems": context.start_tree["problems"],
            "pre_seal2_untracked": list(context.pre_seal2_untracked),
            "allow_dirty": settings.allow_dirty,
            "code_blobs": context.start_tree["code_blobs"],
            "src_tree_id": context.start_tree["src_tree_id"],
            "head": context.start_tree["head"],
            "sealed_pp_assertion": context.sealed_pp,
            "r_checks": context.r_checks,
        }
    )
    return identity


def prepare_stage_dir(settings: Settings) -> None:
    """New folder, ``--overwrite`` (plan only) or ``--resume``; else refuse."""
    stage_dir = settings.stage_dir
    if stage_dir.exists() and any(stage_dir.iterdir()):
        if settings.overwrite and settings.stage is Stage.PLAN:
            shutil.rmtree(stage_dir)
        elif not settings.resume or settings.precheck:
            raise inputs.InputError(f"output exists: {stage_dir}")
    (stage_dir / "parts").mkdir(parents=True, exist_ok=True)


def settings_sha256(settings: Settings, hashes: Mapping[str, str]) -> str:
    """sha256 over the settings that change results and the input hashes."""
    document = {
        "stage": None if settings.stage is None else settings.stage.value,
        "precheck": settings.precheck,
        "folds": list(settings.folds),
        "only_lines": list(settings.only_lines),
        "batch_size": settings.batch_size,
        "inputs": dict(sorted(hashes.items())),
    }
    text = json.dumps(document, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_resume(settings: Settings, hashes: Mapping[str, str], tree: Mapping) -> None:
    """``runner.check_resume_identity`` (same settings, HEAD, blobs, src tree)."""
    runner.check_resume_identity(settings, settings_sha256(settings, hashes), tree)  # type: ignore[arg-type]


def write_json(path: Path, document: Mapping) -> None:
    common.write_json(path, dict(document))


def write_rows(path: Path, columns: Sequence[str], rows: Iterable[Mapping]) -> None:
    """LF csv, floats at full precision (``str(float)`` is ``repr``)."""
    common.write_csv(
        path, columns, ([_cell(row.get(c, "")) for c in columns] for row in rows)
    )


def _cell(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if value is None:
        return ""
    return value


def _exact(value: float | None) -> float | str:
    return "" if value is None else float(value)


# ================================================================== inputs


@dataclass(frozen=True)
class PoolLine:
    """One pool row as the runner uses it."""

    ref: inputs.LineRef
    cluster_id: str
    shares_r3_activity: bool
    role: str

    @property
    def line_id(self) -> str:
        return self.ref.line_id

    @property
    def dataset(self) -> str:
        return self.ref.dataset


def read_pool_lines(path: Path) -> list[PoolLine]:
    """Pool rows as ``PoolLine`` values (IDS keep their stage-A time window)."""
    lines = []
    for row in pool.read_pool(path):
        window = None
        if row["dataset"] == inputs.DATASET_IDS:
            window = float(row["time_window_ns"])
        lines.append(
            PoolLine(
                ref=inputs.LineRef(row["dataset"], row["line_id"], row["role"], window),
                cluster_id=row["cluster_id"],
                shares_r3_activity=row["shares_r3_activity"] == "True",
                role=row["role"],
            )
        )
    return lines


def background_pool(settings: Settings) -> list[PoolLine]:
    """The 153 background lines (``--only-lines`` applied)."""
    lines = [
        line
        for line in read_pool_lines(settings.pool_csv)
        if line.role == pool.ROLE_BACKGROUND
    ]
    if settings.only_lines:
        wanted = set(settings.only_lines)
        missing = wanted - {line.line_id for line in lines}
        if missing:
            raise inputs.InputError(f"--only-lines not in the pool: {sorted(missing)}")
        lines = [line for line in lines if line.line_id in wanted]
    return lines


def precheck_pool(settings: Settings) -> list[PoolLine]:
    """The two precheck lines of the pool (§2-1)."""
    lines = [
        line
        for line in read_pool_lines(settings.pool_csv)
        if line.role == pool.ROLE_PRECHECK
    ]
    if len(lines) != 2:
        raise inputs.InputError(f"pool has {len(lines)} precheck lines, not 2")
    return lines


def _upstream_summary(path: Path) -> dict:
    """A finished, judgeable un-injected run (R2 or R3, judged before)."""
    if not path.is_file():
        raise inputs.InputError(f"upstream summary missing: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    judgeable = (summary.get("judgment") or {}).get("judgeable") is True
    if summary.get("status") != STATUS_OK or not judgeable:
        raise inputs.InputError(f"upstream {path} is not a judgeable ok run")
    return summary


def zone_c_boxes(settings: Settings) -> tuple[dict[str, list[GridBox]], dict]:
    """Zone (c) boxes: R2 4TU (``original``, PP off) and R3 IDS, every class.

    The R3 input of plan §2-1 (``m33_r3_band.py`` 644: folds 0-4 pooled).
    """
    rows: list[dict[str, str]] = []
    hashes = {}
    for name, root in (
        ("r2", settings.r2_root),
        ("r3_uninjected_ids", settings.r3_root),
    ):
        summary_path = root / "summary.json"
        _upstream_summary(summary_path)
        hashes[f"{name}_summary"] = inputs.sha256_file(summary_path)
        for path in sorted(root.glob("fold*/boxes.csv")):
            hashes[f"{name}_{path.parent.name}_boxes"] = inputs.sha256_file_lf(path)
            rows += [
                row
                for row in inputs.read_csv(path)
                if row.get("orientation", runner.ORIGINAL) == runner.ORIGINAL
            ]
    return band_runner.zone_boxes(rows), hashes


def load_hyperbola_library() -> responses_io.ResponseLibrary:
    """The 18 E3x hyperbola models (A0 15 + M12, M15, M18)."""
    return responses_io.load_library(HYPERBOLA_MODELS)


def line_responses(
    library: responses_io.ResponseLibrary,
    line: runner.LineData,
    models: Sequence[str] | None = None,
) -> dict[str, inject.PipeResponse]:
    """Models usable on a line (``for_line``), optionally restricted."""
    responses = library.for_line(line.ref.dataset, runner.line_time_window_ns(line))
    if models is None:
        return dict(sorted(responses.items()))
    return {name: responses[name] for name in sorted(responses) if name in models}


# ============================================================ seal 2 record


PLAN_FILES = (
    "hyperbola_positions.csv",
    "band_positions.csv",
    "band_arms.csv",
    "lines.csv",
    "exclusions.csv",
)


def plan_file_hashes(plan_dir: Path) -> dict[str, str]:
    """LF sha of the plan tables (``plan_report.json`` apart)."""
    return {name: inputs.sha256_file_lf(plan_dir / name) for name in PLAN_FILES}


def seal2_draft(
    plan_dir: Path, settings: Settings, context: Context, identity: Mapping
) -> dict:
    """The record seal 2 copies to ``analysis/detection/m33_e3x_seal2.json``."""
    return {
        "schema": "m33_e3x_seal2/1",
        "pool_csv_sha256_lf": inputs.sha256_file_lf(settings.pool_csv),
        "plan_files_sha256_lf": plan_file_hashes(plan_dir),
        "plan_report_sha256_lf": inputs.sha256_file_lf(plan_dir / "plan_report.json"),
        "code_blobs": context.start_tree["code_blobs"],
        "src_tree_id": context.start_tree["src_tree_id"],
        "plan_head": context.start_tree["head"],
        "plan_run_id": identity["run_id"],
        "seeds": identity["seeds"],
    }


def assert_seal2(settings: Settings, context: Context) -> dict:
    """The tracked seal-2 record against the pool, the plan, the code and ``src``.

    v0.5: the record is seal 2 r1 (``SEAL2_RECORD``) and the plan tables are
    read from ``SEALED_PLAN_DIR`` whatever ``--out-root`` is (a retry root
    has no plan of its own).

    Raises:
        inputs.InputError: For a missing record or any other value (exit 2).
    """
    path = settings.seal_record
    if not path.is_file():
        raise inputs.InputError(f"seal-2 record missing: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    plan_dir = SEALED_PLAN_DIR
    problems = []
    if inputs.sha256_file_lf(settings.pool_csv) != record.get("pool_csv_sha256_lf"):
        problems.append("pool_csv")
    if plan_file_hashes(plan_dir) != record.get("plan_files_sha256_lf"):
        problems.append("plan_files")
    report = plan_dir / "plan_report.json"
    if inputs.sha256_file_lf(report) != record.get("plan_report_sha256_lf"):
        problems.append("plan_report")
    blobs = context.start_tree["code_blobs"]
    recorded = record.get("code_blobs") or {}
    changed = sorted(key for key, value in recorded.items() if blobs.get(key) != value)
    if changed or not recorded:
        problems.append(f"code_blobs {changed}")
    if context.start_tree["src_tree_id"] != record.get("src_tree_id"):
        problems.append("src_tree_id")
    if problems:
        raise inputs.InputError(f"seal-2 record differs: {problems}")
    plan_report = json.loads(report.read_text(encoding="utf-8"))
    if plan_report.get("status") != STATUS_OK:
        raise inputs.InputError(f"plan ended {plan_report.get('status')!r}")
    return {
        "seal_record_sha256_lf": inputs.sha256_file_lf(path),
        "plan_files_sha256_lf": record["plan_files_sha256_lf"],
        "result": "pass",
    }


# =================================================================== plan


@dataclass(frozen=True)
class LineJob:
    """One line of a stage and the zones it needs."""

    line: PoolLine
    exclusions: inject.LineExclusions


@dataclass(frozen=True)
class LinePlanPart:
    """Plan of one line before the global hyperbola choice."""

    line_row: dict
    candidates: tuple[place.HyperbolaCandidate, ...]
    hyperbola_models: Mapping[str, dict]
    band_positions: tuple[dict, ...]
    band_arms: tuple[dict, ...]
    band_report: dict
    n_shift_mismatch: int


def line_row(job: LineJob, line: runner.LineData, context: inject.LineContext) -> dict:
    """``lines.csv`` row: geometry and zero of the background line."""
    background = context.background
    return {
        "line_id": job.line.line_id,
        "dataset": job.line.dataset,
        "cluster_id": job.line.cluster_id,
        "shares_r3_activity": job.line.shares_r3_activity,
        "role": job.line.role,
        "n_traces": background.n_traces,
        "n_samples": background.n_samples,
        "dt_ns": float(background.dt_ns or math.nan),
        "dx_m": float(background.dx_m or math.nan),
        "time_window_ns": _exact(runner.line_time_window_ns(line)),
        "period_samples": float(context.period_samples),
        "period_traces": float(context.period_traces),
        "samples_per_px": float(context.grid_map.samples_per_px),
        "traces_per_px": float(context.grid_map.traces_per_px),
        "k_zero": _exact(context.zero.k_zero),
        "t_zero_ns": _exact(context.zero.t_zero_ns),
        "w_eff": inject.effective_median_window(context),
    }


def hyperbola_candidates(
    job: LineJob,
    context: inject.LineContext,
    responses: Mapping[str, inject.PipeResponse],
) -> tuple[list[place.HyperbolaCandidate], dict[str, dict]]:
    """Allowed (model, apex) pairs and per-model ``r_edge`` of one line.

    A model counts when its S4 patch (k 1.3, 2.5-D on) is built and lies in
    a C3-6 main cell; its apexes are the line offsets passing zones (a)-(d)
    (``inject.placement_failure``, the stage-C rule).
    """
    candidates, models = [], {}
    n_traces = context.background.n_traces
    for model, response in responses.items():
        patch = inject.build_response_patch(response, context, inject.K_MAIN)
        if isinstance(patch, inject.RejectReason):
            models[model] = {"reason": patch.value}
            continue
        if not inject.is_main_cell(context, patch):
            models[model] = {"reason": "not_main_cell"}
            continue
        apexes = [
            offset + response.apex_trace
            for offset in range(max(0, n_traces - response.n_traces + 1))
            if inject.placement_failure(context, patch, offset + response.apex_trace)
            is None
        ]
        models[model] = {
            "reason": "" if apexes else "no_allowed_apex",
            "r_edge_db": place.r_edge_db(patch, context),
            "apex_sample": float(patch.apex_sample),
            "half_width_20db": float(patch.half_width_20db),
            "n_allowed_apex": len(apexes),
        }
        candidates += [
            place.HyperbolaCandidate(
                job.line.line_id,
                job.line.dataset,
                model,
                float(apex),
                float(patch.half_width_20db),
                float(context.period_traces),
            )
            for apex in apexes
        ]
    return candidates, models


def band_position_id(line_id: str, index: int) -> str:
    """``<line>:b<index>`` (shared by every arm, SNR, polarity, texture)."""
    return f"{line_id}:b{index:02d}"


def band_tables(
    job: LineJob,
    context: inject.LineContext,
    plan: place.BandLinePlan,
    groups: Mapping[str, responses_io.ModelGroup],
) -> tuple[list[dict], list[dict]]:
    """``band_positions.csv`` and ``band_arms.csv`` rows of one line."""
    positions, arms = [], []
    period = context.period_samples
    k_zero = context.zero.k_zero
    n_samples = context.background.n_samples
    w_eff = inject.effective_median_window(context)
    for position in plan.positions:
        position_id = band_position_id(job.line.line_id, position.index)
        long_arm = position.arm_of(place.SHALLOW_ROW_PX, place.LONG_ARM)
        assert long_arm is not None  # every position has the 28 px long arm
        positions.append(
            {
                "position_id": position_id,
                "line_id": job.line.line_id,
                "dataset": job.line.dataset,
                "cluster_id": job.line.cluster_id,
                "shares_r3_activity": job.line.shares_r3_activity,
                "model": position.model,
                "model_group": groups[position.model].value,
                "center_trace": float(position.center_trace),
                "taper_traces": position.taper_traces,
                "native_y1_px": float(long_arm.shift.native_y1_px),
                "has_deep_row": position.has_deep_row,
                "long_length_traces": plan.lengths[place.LONG_ARM],
                "short_length_traces": plan.lengths[place.SHORT_ARM],
                "w_eff": w_eff,
                "period_bg_samples": float(period),
                "period_bg_traces": float(context.period_traces),
            }
        )
        for arm in position.arms:
            truth = inject.band_truth_box(arm.band, arm.center_trace, n_samples)
            arms.append(
                {
                    "position_id": position_id,
                    "line_id": job.line.line_id,
                    "bin_target_px": arm.row_px,
                    "length_arm": arm.arm,
                    "center_trace": float(arm.center_trace),
                    "length_traces": arm.length_traces,
                    "length_P": arm.length_traces / context.period_traces,
                    "l_over_w_eff": arm.length_traces / w_eff,
                    "taper_traces": arm.band.taper_traces,
                    "shift_samples_first": arm.shift.shift_samples_first,
                    "shift_samples": arm.shift.shift_samples,
                    "shift_correction_samples": arm.shift.shift_correction_samples,
                    "shift_step_samples": arm.shift.shift_step_samples,
                    "shift_gap_samples": float(arm.band.row0)
                    - place.target_sample(context, arm.row_px),
                    "short_center_offset_traces": arm.center_trace
                    - position.center_trace,
                    "shift_P": float(arm.shift.shift_periods),
                    "delta_t_ns": float(arm.shift.delta_t_ns),
                    "d_prime_m": place.d_prime_m(context, arm.band),
                    "apex_sample": float(arm.band.apex_sample),
                    "apex_P_below_zero_target": (
                        ""
                        if k_zero is None
                        else (arm.band.apex_sample - k_zero) / period
                    ),
                    "rows_to_record_end_P": (n_samples - 1 - arm.band.row1) / period,
                    "realized_y1_px_plan": place.row_px_of(context, arm.band.row0),
                    "i0": truth.i0,
                    "i1": truth.i1,
                    "k0": truth.k0,
                    "k1": truth.k1,
                }
            )
    return positions, arms


def plan_line(
    job: LineJob,
    hyperbola_library: responses_io.ResponseLibrary,
    band_library: responses_io.ResponseLibrary,
) -> LinePlanPart:
    """Hyperbola candidates and band positions of one line (CPU, seeded)."""
    line = runner.load_line(job.line.ref)
    context = inject.prepare_line_context(line.radargram, job.exclusions)
    candidates, models = hyperbola_candidates(
        job, context, line_responses(hyperbola_library, line, HYPERBOLA_MODELS)
    )
    band_plan = place.plan_band_positions_e3x(
        context, line_responses(band_library, line, BAND_MODELS)
    )
    positions, arms = band_tables(job, context, band_plan, band_library.groups)
    report = {
        "line_id": job.line.line_id,
        "excluded_reason": band_plan.excluded_reason,
        "model_reasons": dict(band_plan.model_reasons),
        "arm_failures": dict(band_plan.arm_failures),
        "shift_mismatches": [
            {"line_id": job.line.line_id, **item} for item in band_plan.shift_mismatches
        ],
        "lengths": dict(band_plan.lengths),
        "n_positions": len(band_plan.positions),
    }
    return LinePlanPart(
        line_row(job, line, context),
        tuple(candidates),
        models,
        tuple(positions),
        tuple(arms),
        report,
        band_plan.n_shift_mismatch,
    )


_WORKER: dict[str, object] = {}


def _init_plan_worker() -> None:
    _WORKER["hyperbola"] = load_hyperbola_library()
    _WORKER["band"] = responses_io.load_library()


def _plan_worker(job: LineJob) -> LinePlanPart:
    return plan_line(job, _WORKER["hyperbola"], _WORKER["band"])  # type: ignore[arg-type]


def map_ordered(
    workers: int,
    function: Callable,
    jobs: Sequence,
    initializer: Callable[[], None],
    label: str,
) -> list:
    """``function`` over ``jobs`` in job order; in process with one worker."""
    results = []
    if workers == 1:
        initializer()
        for number, job in enumerate(jobs, start=1):
            results.append(function(job))
            print(f"{label} {number}/{len(jobs)}", flush=True)
        return results
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer) as pool_:
        # map keeps the job order, so the tables do not depend on the workers.
        for number, result in enumerate(pool_.map(function, jobs), start=1):
            results.append(result)
            print(f"{label} {number}/{len(jobs)}", flush=True)
    return results


def hyperbola_target_map(
    lines: Sequence[PoolLine],
    groups: Mapping[str, responses_io.ModelGroup],
) -> dict[tuple[str, str], int]:
    """``(background, model) -> target`` for the backgrounds present."""
    backgrounds = {line.dataset for line in lines}
    targets = {}
    for model in HYPERBOLA_MODELS:
        for background in sorted(backgrounds):
            target = HYPERBOLA_TARGETS.get((background, groups[model]), 0)
            if target:
                targets[(background, model)] = target
    return targets


HYPERBOLA_COLUMNS = (
    "position_id",
    "line_id",
    "dataset",
    "cluster_id",
    "shares_r3_activity",
    "model",
    "model_group",
    "group",
    "index_in_group",
    "apex_trace",
    "apex_sample",
    "apex_depth_P",
    "apex_y_px",
    "half_width_20db",
    "r_edge_db",
    "edge_stratum",
    "edge_snr_set",
    "second_on_line",
    "period_bg_samples",
    "period_bg_traces",
)
BAND_POSITION_COLUMNS = (
    "position_id",
    "line_id",
    "dataset",
    "cluster_id",
    "shares_r3_activity",
    "model",
    "model_group",
    "center_trace",
    "taper_traces",
    "native_y1_px",
    "has_deep_row",
    "long_length_traces",
    "short_length_traces",
    "w_eff",
    "period_bg_samples",
    "period_bg_traces",
)
BAND_ARM_COLUMNS = (
    "position_id",
    "line_id",
    "bin_target_px",
    "length_arm",
    "center_trace",
    "length_traces",
    "length_P",
    "l_over_w_eff",
    "taper_traces",
    "shift_samples_first",
    "shift_samples",
    "shift_correction_samples",
    "shift_step_samples",
    "shift_gap_samples",
    "short_center_offset_traces",
    "shift_P",
    "delta_t_ns",
    "d_prime_m",
    "apex_sample",
    "apex_P_below_zero_target",
    "rows_to_record_end_P",
    "realized_y1_px_plan",
    "i0",
    "i1",
    "k0",
    "k1",
)
LINE_COLUMNS = (
    "line_id",
    "dataset",
    "cluster_id",
    "shares_r3_activity",
    "role",
    "n_traces",
    "n_samples",
    "dt_ns",
    "dx_m",
    "time_window_ns",
    "period_samples",
    "period_traces",
    "samples_per_px",
    "traces_per_px",
    "k_zero",
    "t_zero_ns",
    "w_eff",
)


def hyperbola_rows(
    plan: place.HyperbolaPlan,
    parts: Mapping[str, LinePlanPart],
    lines: Mapping[str, PoolLine],
    groups: Mapping[str, responses_io.ModelGroup],
) -> list[dict]:
    """``hyperbola_positions.csv`` rows (line, group, apex order)."""
    rows = []
    ordered = sorted(plan.positions, key=lambda p: (p.line_id, p.group, p.apex_trace))
    index_in_group: Counter[tuple[str, int]] = Counter()
    for position in ordered:
        part = parts[position.line_id]
        model = part.hyperbola_models[position.model]
        line_info = part.line_row
        period = line_info["period_samples"]
        k_zero = line_info["k_zero"]
        index = index_in_group[(position.line_id, position.group)]
        index_in_group[(position.line_id, position.group)] += 1
        r_edge = model["r_edge_db"]
        rows.append(
            {
                "position_id": hyperbola_position_id(
                    position.line_id, position.group, index
                ),
                "line_id": position.line_id,
                "dataset": position.background,
                "cluster_id": lines[position.line_id].cluster_id,
                "shares_r3_activity": lines[position.line_id].shares_r3_activity,
                "model": position.model,
                "model_group": groups[position.model].value,
                "group": position.group,
                "index_in_group": index,
                "apex_trace": position.apex_trace,
                "apex_sample": model["apex_sample"],
                "apex_depth_P": (
                    "" if k_zero == "" else (model["apex_sample"] - k_zero) / period
                ),
                "apex_y_px": model["apex_sample"] / line_info["samples_per_px"],
                "half_width_20db": position.half_width_20db,
                "r_edge_db": r_edge,
                "edge_stratum": place.edge_stratum(r_edge).value,
                "edge_snr_set": ",".join(f"{s:g}" for s in place.edge_snr_set(r_edge)),
                "second_on_line": position.second_on_line,
                "period_bg_samples": period,
                "period_bg_traces": line_info["period_traces"],
            }
        )
    return rows


def hyperbola_position_id(line_id: str, group: int, index: int) -> str:
    """``<line>:h<group>:<index>`` (shared by both arms and every condition)."""
    return f"{line_id}:h{group:02d}:{index:02d}"


def run_plan(
    settings: Settings, started: float, context: Context
) -> tuple[str, Counts]:
    """Zone (c), per-line candidates and bands, round-robin, tables, seal draft."""
    hashes = pool_input_hashes(settings)
    boxes, zone_hashes = zone_c_boxes(settings)
    hashes.update(zone_hashes)
    candidates_apex = inputs.candidate_apexes()
    lines = background_pool(settings)
    prepare_stage_dir(settings)
    jobs = [
        LineJob(line, runner.line_exclusions(line.line_id, boxes, candidates_apex))
        for line in lines
    ]
    plan_started = time.perf_counter()
    parts = map_ordered(settings.workers, _plan_worker, jobs, _init_plan_worker, "plan")
    by_line = {part.line_row["line_id"]: part for part in parts}
    groups = dict(responses_io.MODEL_GROUPS)
    targets = hyperbola_target_map(lines, groups)
    choice = place.plan_hyperbola_positions(
        [c for part in parts for c in part.candidates], targets
    )
    plan_seconds = time.perf_counter() - plan_started
    print(f"timing plan: {plan_seconds:.1f} s", flush=True)
    identity = new_identity(settings, context, hashes)
    mismatches = sum(part.n_shift_mismatch for part in parts)
    base = STATUS_OK if mismatches == 0 else STATUS_DEFECT
    status, judgment = close_run(settings, context, identity, base)
    rows = hyperbola_rows(
        choice, by_line, {line.line_id: line for line in lines}, groups
    )
    positions = [row for part in parts for row in part.band_positions]
    arms = [row for part in parts for row in part.band_arms]
    stage_dir = settings.stage_dir
    write_rows(stage_dir / "hyperbola_positions.csv", HYPERBOLA_COLUMNS, rows)
    write_rows(stage_dir / "band_positions.csv", BAND_POSITION_COLUMNS, positions)
    write_rows(stage_dir / "band_arms.csv", BAND_ARM_COLUMNS, arms)
    write_rows(stage_dir / "lines.csv", LINE_COLUMNS, [p.line_row for p in parts])
    inputs.write_exclusions(stage_dir / "exclusions.csv", boxes)
    report = plan_report(rows, positions, arms, parts, choice, lines)
    report.update(
        {
            "schema": SCHEMA,
            "stage": Stage.PLAN.value,
            "status": status,
            "judgment": judgment,
            "run_identity": {**identity, "finished_at": common.local_now()},
            "n_shift_mismatch": mismatches,
            "timings": {"plan": plan_seconds},
            "seconds": round(time.time() - started, 1),
        }
    )
    write_json(stage_dir / "plan_report.json", report)
    write_json(
        stage_dir / "seal2_record_draft.json",
        seal2_draft(stage_dir, settings, context, identity),
    )
    return status, Counts(len(parts), len(rows) + len(positions), 0, 0)


def pool_input_hashes(settings: Settings) -> dict[str, str]:
    """LF sha of the pool, the line lists and the apex candidates."""
    hashes = pool.assert_inputs()
    hashes["pool_csv"] = inputs.sha256_file_lf(settings.pool_csv)
    hashes["apex_candidates"] = inputs.sha256_file_lf(inputs.APEX_CANDIDATES)
    hashes["k_apexes"] = inputs.checked_text_sha256(
        inputs.K_APEXES, inputs.EXPECTED_SHA256["k_apexes"], "k_apexes"
    )
    return hashes


# ------------------------------------------------------------ plan report


def _kish(counts: Iterable[int]) -> float | None:
    values = [n for n in counts if n > 0]
    if not values:
        return None
    return sum(values) ** 2 / sum(n * n for n in values)


def cluster_counts(rows: Sequence[Mapping]) -> dict:
    """Positions, clusters and Kish effective clusters per background."""
    report = {}
    for background in sorted({row["dataset"] for row in rows}):
        members = [row for row in rows if row["dataset"] == background]
        per_cluster = Counter(row["cluster_id"] for row in members)
        report[background] = {
            "positions": len(members),
            "clusters": len(per_cluster),
            "g_eff": _kish(per_cluster.values()),
        }
    report["clusters_total"] = len({(r["dataset"], r["cluster_id"]) for r in rows})
    report["positions_total"] = len(rows)
    return report


def _arm_set(arms: Sequence[Mapping]) -> dict[str, set[tuple[float, str]]]:
    found: dict[str, set[tuple[float, str]]] = {}
    for arm in arms:
        found.setdefault(arm["position_id"], set()).add(
            (float(arm["bin_target_px"]), arm["length_arm"])
        )
    return found


def band_hypothesis_positions(
    positions: Sequence[Mapping], arms: Sequence[Mapping]
) -> dict[str, list[Mapping]]:
    """Planned positions of HB1, HB1b and HB2 (arms planned, before results)."""
    by_position = _arm_set(arms)
    long_rows = {
        "HB1": (place.SHALLOW_ROW_PX, *place.IN_RANGE_ROWS_PX),
        "HB1b": (*place.IN_RANGE_ROWS_PX, place.DEEP_ROW_PX),
    }
    selected: dict[str, list[Mapping]] = {"HB1": [], "HB1b": [], "HB2": []}
    for row in positions:
        present = by_position.get(row["position_id"], set())
        for name, rows_px in long_rows.items():
            if all((r, place.LONG_ARM) in present for r in rows_px):
                selected[name].append(row)
        if all(
            (r, arm) in present
            for r in place.IN_RANGE_ROWS_PX
            for arm in (place.LONG_ARM, place.SHORT_ARM)
        ):
            selected["HB2"].append(row)
    return selected


def plan_report(
    hyperbola: Sequence[Mapping],
    band_positions: Sequence[Mapping],
    band_arms: Sequence[Mapping],
    parts: Sequence[LinePlanPart],
    choice: place.HyperbolaPlan,
    lines: Sequence[PoolLine],
) -> dict:
    """Detector-free plan counts for seal 2 and the MDE evaluation (§3-4)."""
    strata = Counter(
        (row["model"], row["edge_stratum"], row["edge_snr_set"]) for row in hyperbola
    )
    by_model: dict[str, dict[str, int]] = {}
    for (model, stratum, snr_set), count in sorted(strata.items()):
        key = f"{stratum}:{snr_set}" if snr_set else stratum
        by_model.setdefault(model, {})[key] = count
    band_sets = band_hypothesis_positions(band_positions, band_arms)
    fourtu = [line for line in lines if line.dataset == inputs.DATASET_4TU]
    shared_lines = [line for line in fourtu if line.shares_r3_activity]
    return {
        "lines": {
            "by_dataset": dict(Counter(line.dataset for line in lines)),
            "fourtu_r3_shared_activity_lines": len(shared_lines),
            "fourtu_r3_shared_activities": len(
                {line.cluster_id for line in shared_lines}
            ),
        },
        "hyperbola": {
            "positions_by_dataset_model": runner.count_by(
                hyperbola, ("dataset", "model")
            ),
            "shortfall": {f"{bg}/{m}": n for (bg, m), n in choice.shortfall.items()},
            "edge_strata_before_model_condition": dict(
                Counter(row["edge_stratum"] for row in hyperbola)
            ),
            "edge_composition_by_model": by_model,
            "r_edge_db_m02_m07_m08": {
                row["position_id"]: row["r_edge_db"]
                for row in hyperbola
                if row["model"] in ("M02", "M07", "M08")
            },
            "clusters_all": cluster_counts(hyperbola),
            "clusters_manipulated": cluster_counts(
                [r for r in hyperbola if r["edge_stratum"] == "manipulated"]
            ),
            "clusters_negative_control": cluster_counts(
                [r for r in hyperbola if r["edge_stratum"] == "negative_control"]
            ),
            "second_on_line": sum(bool(r["second_on_line"]) for r in hyperbola),
        },
        "band": {
            "positions_by_dataset_model": runner.count_by(
                band_positions, ("dataset", "model")
            ),
            "arms_by_row_length": runner.count_by(
                band_arms, ("bin_target_px", "length_arm")
            ),
            "hypothesis_clusters": {
                name: cluster_counts(rows) for name, rows in band_sets.items()
            },
            "hb1b_models": runner.count_by(band_sets["HB1b"], ("dataset", "model")),
            "hb1_models": runner.count_by(band_sets["HB1"], ("dataset", "model")),
            "shift_P_by_row_model": shift_distribution(band_arms, band_positions),
            "lines_excluded": {
                part.band_report["line_id"]: part.band_report["excluded_reason"]
                for part in parts
                if part.band_report["excluded_reason"]
            },
            "lines_without_position": [
                part.band_report["line_id"]
                for part in parts
                if not part.band_report["n_positions"]
            ],
            "model_reasons": dict(
                Counter(
                    reason.split(":")[-1]
                    for part in parts
                    for reason in part.band_report["model_reasons"].values()
                )
            ),
            "arm_failures": dict(
                sum(
                    (Counter(part.band_report["arm_failures"]) for part in parts),
                    Counter(),
                )
            ),
            "shift_mismatches": [
                item for part in parts for item in part.band_report["shift_mismatches"]
            ],
            "shift_mismatch_checks": {
                "target_row_i": sum(
                    1
                    for part in parts
                    for item in part.band_report["shift_mismatches"]
                    if item["arm_offsets_parallel"]
                ),
                "arm_offsets_ii_models": len(
                    {
                        (part.band_report["line_id"], item["model"])
                        for part in parts
                        for item in part.band_report["shift_mismatches"]
                        if not item["arm_offsets_parallel"]
                    }
                ),
            },
            "shift_correction_samples": runner.count_by(
                band_arms, ("shift_correction_samples",)
            ),
            "shift_step_samples": runner.count_by(band_arms, ("shift_step_samples",)),
            "short_arm_center_offset": short_offset_positions(band_arms),
        },
    }


def short_offset_positions(band_arms: Iterable[Mapping]) -> int:
    """Band positions whose short arm sits 0.5 trace left (v0.4 §2-4, §6 3).

    Counted per position: the short arm has two rows (47 and 68 px) with
    the same centre, so counting arms would give about twice the number.
    """
    return len(
        {
            arm["position_id"]
            for arm in band_arms
            if arm["length_arm"] == place.SHORT_ARM
            and float(arm["short_center_offset_traces"]) != 0
        }
    )


def shift_distribution(
    arms: Sequence[Mapping], positions: Sequence[Mapping]
) -> dict[str, dict]:
    """Median and maximum ``|shift|`` in P per row and model (plan §6 3)."""
    models = {row["position_id"]: row["model"] for row in positions}
    groups: dict[str, list[float]] = {}
    for arm in arms:
        if arm["length_arm"] != place.LONG_ARM:
            continue
        key = f"{arm['bin_target_px']:g}/{models[arm['position_id']]}"
        groups.setdefault(key, []).append(float(arm["shift_P"]))
    return {
        key: {"n": len(v), "median": statistics.median(v), "max": max(v)}
        for key, v in sorted(groups.items())
    }


# ============================================================ plan reading


@dataclass(frozen=True)
class PlannedHyperbolaRow:
    """One planned hyperbola position as read from the plan table."""

    position_id: str
    model: str
    group: int
    index_in_group: int
    apex_trace: float
    apex_sample: float
    row: Mapping[str, str]


@dataclass(frozen=True)
class PlannedArm:
    """One planned band arm as read from the plan table."""

    position_id: str
    model: str
    row_px: float
    arm: str
    center_trace: float
    length_traces: int
    taper_traces: int
    shift_samples: int
    truth: GridBox
    row: Mapping[str, str]
    position: Mapping[str, str]


@dataclass(frozen=True)
class LinePlan:
    """Everything the plan fixed for one line."""

    line: PoolLine
    hyperbolas: tuple[PlannedHyperbolaRow, ...]
    arms: tuple[PlannedArm, ...]
    exclusions: inject.LineExclusions

    @property
    def band_position_ids(self) -> tuple[str, ...]:
        return tuple(sorted({arm.position_id for arm in self.arms}))


def read_plan(
    plan_dir: Path, lines: Sequence[PoolLine], candidates: inputs.CandidateApexes
) -> list[LinePlan]:
    """The plan of every given line (tables as written by the plan stage)."""
    hyperbolas = inputs.read_csv(plan_dir / "hyperbola_positions.csv")
    positions = {
        row["position_id"]: row
        for row in inputs.read_csv(plan_dir / "band_positions.csv")
    }
    arms = inputs.read_csv(plan_dir / "band_arms.csv")
    boxes = inputs.read_exclusions(plan_dir / "exclusions.csv")
    plans = []
    for line in lines:
        h_rows = tuple(
            PlannedHyperbolaRow(
                row["position_id"],
                row["model"],
                int(row["group"]),
                int(row["index_in_group"]),
                float(row["apex_trace"]),
                float(row["apex_sample"]),
                row,
            )
            for row in hyperbolas
            if row["line_id"] == line.line_id
        )
        a_rows = tuple(
            PlannedArm(
                row["position_id"],
                positions[row["position_id"]]["model"],
                float(row["bin_target_px"]),
                row["length_arm"],
                float(row["center_trace"]),
                int(row["length_traces"]),
                int(row["taper_traces"]),
                int(row["shift_samples"]),
                GridBox(int(row["i0"]), int(row["i1"]), int(row["k0"]), int(row["k1"])),
                row,
                positions[row["position_id"]],
            )
            for row in arms
            if row["line_id"] == line.line_id
        )
        exclusions = runner.line_exclusions(line.line_id, boxes, candidates)
        plans.append(LinePlan(line, h_rows, a_rows, exclusions))
    return plans


# ============================================================ detection


class WarningCounter(logging.Handler):
    """Counts log records whose message matches a pattern (NMS time limit)."""

    def __init__(self, pattern: re.Pattern[str] = NMS_WARNING) -> None:
        super().__init__(level=logging.WARNING)
        self.pattern = pattern
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if self.pattern.search(record.getMessage()):
            self.count += 1

    def take(self) -> int:
        """The count since the last call, then reset."""
        value, self.count = self.count, 0
        return value


def attach_nms_counter() -> WarningCounter:
    """A counter on the Ultralytics logger (lazy import, AGPL boundary)."""
    from ultralytics.utils import LOGGER  # lazy (ADR-001)

    counter = WarningCounter()
    LOGGER.addHandler(counter)
    return counter


def load_detectors(settings: Settings, confidence: float) -> dict[int, object]:
    """N2-a' ``last.pt`` of every fold at one conf (weights asserted before)."""
    from gpr_detection.detect.yolo import load_yolo_detector  # lazy (ADR-001)
    from gpr_detection.detect.yolo_logging import quiet_ultralytics_logger

    quiet_ultralytics_logger()
    return {
        fold: load_yolo_detector(
            runner.weights_path(settings.run_root, fold),
            input_size=INPUT_PX,
            confidence_threshold=confidence,
            iou_threshold=NMS_IOU,
            device=settings.device,
        )
        for fold in settings.folds
    }


def operating_boxes(tile_boxes: Sequence[LineBox]) -> list[LineBox]:
    """Tile boxes over the operating conf, cut **before** the merge (§3-1)."""
    return [box for box in tile_boxes if box.score > CONF_OPERATING]


@dataclass(frozen=True)
class FoldBoxes:
    """Boxes of one image and fold on every path the records need."""

    tile_boxes: tuple[LineBox, ...]
    merged: tuple[tuple[LineBox, GridBox], ...]
    merged_s3: tuple[tuple[LineBox, GridBox], ...]
    n_tiles_maxdet: int
    n_tiles_operating_maxdet: int


def _to_grid(
    boxes: Iterable[LineBox], image: DetectionImage
) -> tuple[tuple[LineBox, GridBox], ...]:
    pairs = []
    for box in boxes:
        grid = image_box_to_grid_box(box.box, image.grid_map)
        if grid is not None:
            pairs.append((box, grid))
    return tuple(pairs)


def fold_boxes(
    image: DetectionImage,
    tile_boxes: Sequence[LineBox],
    zero: ZeroResult,
    config: PPConfig,
) -> FoldBoxes:
    """Operating cut, merge, S0 and S3 from the same merged boxes, max-det tiles."""
    merged = merge_line_boxes(operating_boxes(tile_boxes), image, DEFAULT_MERGE_CONFIG)
    outcomes = apply_pp(merged, content_rows_for(image, zero), config)
    per_tile = Counter(box.tile_index for box in tile_boxes)
    operating_tile = Counter(box.tile_index for box in operating_boxes(tile_boxes))
    return FoldBoxes(
        tile_boxes=tuple(tile_boxes),
        merged=_to_grid(merged, image),
        merged_s3=_to_grid(kept_after_pp(outcomes), image),
        n_tiles_maxdet=sum(count >= MAX_DET for count in per_tile.values()),
        n_tiles_operating_maxdet=sum(
            count >= MAX_DET for count in operating_tile.values()
        ),
    )


def scored(pairs: Iterable[tuple[LineBox, GridBox]]) -> list[match.ScoredBox]:
    return [match.ScoredBox(box.class_id, box.score, grid) for box, grid in pairs]


def best_tile_score(
    tile_boxes: Sequence[LineBox],
    image: DetectionImage,
    accepts: Callable[[GridBox], bool],
    class_id: int | None,
) -> float | None:
    """Largest pre-merge tile score whose L0 box (rounded outward) is accepted."""
    best = None
    for box in tile_boxes:
        if class_id is not None and box.class_id != class_id:
            continue
        grid = image_box_to_grid_box(box.box, image.grid_map)
        if grid is None or not accepts(grid):
            continue
        if best is None or box.score > best:
            best = float(box.score)
    return best


def best_scores(
    tile_boxes: Sequence[LineBox],
    image: DetectionImage,
    accepts: Callable[[GridBox], bool],
    class_id: int,
    prefix: str,
) -> dict:
    """``best_score_001_<prefix>`` (one class) and ``_any`` with their ``no_box``."""
    own = best_tile_score(tile_boxes, image, accepts, class_id)
    anything = best_tile_score(tile_boxes, image, accepts, None)
    return {
        f"best_score_001_{prefix}": _exact(own),
        f"best_score_001_{prefix}_no_box": own is None,
        "best_score_001_any": _exact(anything),
        "best_score_001_any_no_box": anything is None,
    }


def apex_accepts(
    apex: match.TruthApex, period_samples: float
) -> Callable[[GridBox], bool]:
    """The apex containment rule (+-3 traces, +-0.5 P, background period)."""
    return lambda grid: match.box_contains_apex(grid, apex, period_samples)


def band_accepts(truth: GridBox) -> Callable[[GridBox], bool]:
    """``can_cover``: shared rows >= 50 % of the truth and a trace in ``[i0, i1]``."""
    return lambda grid: p4.can_cover(truth, grid)


def band_cover(
    truth: GridBox, pairs: Sequence[tuple[LineBox, GridBox]], class_id: int | None
) -> int:
    """Covered truth traces by the merged boxes of a class (``None``: any)."""
    grids = [g for box, g in pairs if class_id is None or box.class_id == class_id]
    return p4.covered_traces(truth, p4.recall_boxes(truth, grids))


# ============================================================ realizations


P42_CONFIG = replace(
    inputs.recipe_config(RECIPE_R0), p_target_px=P42_TARGET_PX, recipe_id=RECIPE_P42
)
"""R-b: r0 with ``P*`` 4.2 only (tile 256, overlap from ``overlap_for``, §2-6)."""
RECIPE_CONFIGS = {RECIPE_R0: inputs.recipe_config(RECIPE_R0), RECIPE_P42: P42_CONFIG}

KIND_HYPERBOLA = "hyperbola"
KIND_BAND = "band"
KIND_UNINJECTED = "uninjected"


@dataclass(frozen=True)
class Truth:
    """One planned truth of an image: where to score and what to write."""

    injection_id: str
    position_id: str
    accepted: bool
    apex_trace: float
    apex_sample: float
    box: GridBox | None
    edge_traces: tuple[int, ...]
    fields: Mapping[str, object]


@dataclass(frozen=True, eq=False)
class ImageEntry:
    """One image to infer: a realization (or a background) and its truths."""

    realization_id: str
    kind: str
    recipe: str
    image: DetectionImage | None
    truths: tuple[Truth, ...]
    fields: Mapping[str, object]


@dataclass(frozen=True, eq=False)
class LineRealizations:
    """Everything the CPU side built for one line."""

    plan: LinePlan
    line_fields: Mapping[str, object]
    period_samples: float
    period_traces: float
    zero: ZeroResult
    entries: tuple[ImageEntry, ...]
    defects: Mapping[str, int]
    texture_checks: Mapping[str, bool]
    off_on_equal: Mapping[str, bool]
    zero_test: bool
    r_edge_matches: Mapping[str, bool] = field(default_factory=dict)
    n_checked: Mapping[str, int] = field(default_factory=dict)
    """Checks made per ``CPU_ASSERTIONS`` key (precheck 10 count guard, v0.5)."""
    failed_items: tuple[dict, ...] = ()
    """``{"assertion", "item_id", "error"}`` of each failed CPU check (v0.5)."""
    texture_details: Mapping[str, place.TextureCheck] = field(default_factory=dict)


def zero_and_period_fields(
    context: inject.LineContext, injected: Radargram, image: DetectionImage
) -> dict:
    """[ZD] keys of one realization (``m33_change_defs``)."""
    filtered = inject.linear_steps(injected.data, injected)
    measured = find_time_zero(injected, filtered)
    zero = change_defs.zero_change(context.zero, measured, context.period_samples)
    period = change_defs.period_change(context.period_samples, image.period_samples)
    return {
        "dk_zero_samples": _exact(zero.dk_samples),
        "dk_zero_over_p": _exact(zero.dk_over_p),
        "zero_label_changed": zero.label_changed,
        "zero_failed_changed": zero.failed_changed,
        "zero_rule_changed": zero.rule_changed,
        "zero_events_changed": zero.events_changed,
        "zero_changed": zero.changed,
        "zero_changed_exact": zero.exact,
        "dp_samples": _exact(period.dp_samples),
        "dp_rel": _exact(period.dp_rel),
        "period_failed": period.failed,
        "period_changed": period.changed,
        "period_changed_exact": period.exact,
        "period_bg": float(context.period_samples),
        "period_inj": float(image.period_samples),
    }


def zero_test_passes(context: inject.LineContext) -> bool:
    """§5-2 check 2: zero and period re-measured on the background are unchanged."""
    background = context.background
    measured = find_time_zero(
        background, inject.linear_steps(background.data, background)
    )
    zero = change_defs.zero_change(context.zero, measured, context.period_samples)
    image = make_detection_image(background, RECIPE_CONFIGS[RECIPE_R0])
    return (
        not zero.changed
        and not zero.exact
        and image.period_samples == context.period_samples
    )


def injected_radargram(context: inject.LineContext, data: np.ndarray) -> Radargram:
    """The background line with injected data (float32, as the R runners)."""
    return replace(context.background, data=data.astype(np.float32))


def _line_fields(plan: LinePlan) -> dict:
    return {
        "line_id": plan.line.line_id,
        "dataset": plan.line.dataset,
        "cluster_id": plan.line.cluster_id,
        "shares_r3_activity": plan.line.shares_r3_activity,
    }


# ---------------------------------------------------------------- hyperbola


def hyperbola_specs(
    plan: LinePlan, group: int, snr_db: float, polarity: int
) -> list[tuple[inject.InjectionSpec, PlannedHyperbolaRow]]:
    """Table rows of one hyperbola group under one condition, with their position.

    Both arms share the injection ids; the realization id of the on arm ends
    in ``:on`` (the off arm's in ``:off``).
    """
    stem = f"{plan.line.line_id}:h{group:02d}:snr{snr_db:g}:p{polarity:+d}"
    members = sorted(
        (h for h in plan.hyperbolas if h.group == group), key=lambda h: h.index_in_group
    )
    return [
        (
            inject.InjectionSpec(
                injection_id=f"{stem}:{h.index_in_group:02d}",
                realization_id=f"{stem}:on",
                line_id=plan.line.line_id,
                apex_trace=h.apex_trace,
                response_id=h.model,
                snr_db=snr_db,
                polarity=polarity,
                k=inject.K_MAIN,
                seed=place.PLAN_SEED_E3X,
            ),
            h,
        )
        for h in members
    ]


def _hyperbola_static(
    planned: PlannedHyperbolaRow, taper: str, spec: inject.InjectionSpec
) -> dict:
    row = planned.row
    return {
        "position_id": planned.position_id,
        "model": planned.model,
        "model_group": row["model_group"],
        "group": planned.group,
        "apex_trace": planned.apex_trace,
        "apex_sample": planned.apex_sample,
        "apex_depth_P": row["apex_depth_P"],
        "apex_y_px": row["apex_y_px"],
        "r_edge_db": row["r_edge_db"],
        "edge_stratum": row["edge_stratum"],
        "edge_snr_set": row["edge_snr_set"],
        "taper": taper,
        "snr_db": spec.snr_db,
        "polarity": spec.polarity,
    }


def hyperbola_entries(
    plan: LinePlan,
    context: inject.LineContext,
    responses: Mapping[str, inject.PipeResponse],
    images: bool = True,
) -> tuple[list[ImageEntry], dict[str, bool]]:
    """On and off arms of every group, SNR and polarity (plan §2-3).

    Args:
        images: ``False`` (precheck 10): the on / off fields and their check
            only, in the same order; no arm image (no entries).
    """
    patches = hyperbola_patches(plan, context, responses)
    entries, equal = [], {}
    for group in sorted({h.group for h in plan.hyperbolas}):
        for snr_db in HYPERBOLA_SNR_DB:
            for polarity in POLARITIES:
                pairs = hyperbola_specs(plan, group, snr_db, polarity)
                specs = [spec for spec, _ in pairs]
                on = inject.inject_realization(context, specs, responses)
                off = place.inject_untapered(context, on, patches)
                equal[specs[0].realization_id] = _off_matches_on(
                    context, on, off, patches
                )
                if images:
                    entries += _hyperbola_arm_entries(
                        plan, context, pairs, on, off, patches
                    )
    return entries, equal


R_EDGE_RELATIVE_TOLERANCE = 1e-9


def r_edge_matches(
    plan: LinePlan,
    context: inject.LineContext,
    patches: Mapping[tuple[str, float], inject.ResponsePatch],
) -> dict[str, bool]:
    """§5-2 check 4 (v0.4): the planned ``r_edge(p)`` from the injected arms.

    Re-measured on the fields the two arms add (on: ``patch.field``, off:
    ``untapered_field(patch)``, before they meet the line); equal to the
    plan value within a relative 1e-9.
    """
    found = {}
    for h in plan.hyperbolas:
        patch = patches.get((h.model, inject.K_MAIN))
        if patch is None:
            found[h.position_id] = False
            continue
        measured = place.r_edge_db(patch, context)
        recorded = float(h.row["r_edge_db"])
        found[h.position_id] = abs(measured - recorded) <= (
            R_EDGE_RELATIVE_TOLERANCE * abs(recorded)
        )
    return found


def r_edge_failures(
    plan: LinePlan,
    context: inject.LineContext,
    patches: Mapping[tuple[str, float], inject.ResponsePatch],
    matches: Mapping[str, bool],
) -> list[dict]:
    """Error records of the positions :func:`r_edge_matches` failed (precheck 10).

    ``error`` holds the re-measured and the planned ``r_edge_db`` (``None``
    when the patch could not be built); the pass rule stays the one of
    :func:`r_edge_matches`.
    """
    failures = []
    for h in plan.hyperbolas:
        if matches.get(h.position_id, True):
            continue
        patch = patches.get((h.model, inject.K_MAIN))
        measured = None if patch is None else place.r_edge_db(patch, context)
        recorded = float(h.row["r_edge_db"])
        error = {"measured_db": measured, "planned_db": recorded}
        failures.append(_failure("r_edge_differs_from_plan", h.position_id, error))
    return failures


def _failure(assertion: str, item_id: str, error: object = None) -> dict:
    return {"assertion": assertion, "item_id": item_id, "error": error}


def hyperbola_patches(
    plan: LinePlan,
    context: inject.LineContext,
    responses: Mapping[str, inject.PipeResponse],
) -> dict[tuple[str, float], inject.ResponsePatch]:
    """S4 patches (k 1.3) of the planned hyperbola models of a line."""
    patches = {}
    for h in plan.hyperbolas:
        built = inject.build_response_patch(responses[h.model], context, inject.K_MAIN)
        if isinstance(built, inject.ResponsePatch):
            patches[(h.model, inject.K_MAIN)] = built
    return patches


def _off_matches_on(
    context: inject.LineContext,
    on: inject.InjectionResult,
    off: place.UntaperedInjection,
    patches: Mapping[tuple[str, float], inject.ResponsePatch],
) -> bool:
    """§5-2 check 4 on one realization (rejected off fields masked out whole)."""
    n_traces = context.background.n_traces
    tapered = [(patches[(t.response_id, t.k)], t.apex_trace) for t in on.truths]
    mask = place.aperture_taper_columns(tapered, n_traces)
    rejected = {key for key, _ in off.rejected}
    for truth in on.truths:
        if truth.injection_id in rejected:
            first, last = place.aperture_end_traces(
                patches[(truth.response_id, truth.k)], truth.apex_trace
            )
            mask[max(0, first) : last + 1] = True
    return place.off_equals_on_outside_tapers(on.data, off.data, mask)


def _hyperbola_arm_entries(
    plan: LinePlan,
    context: inject.LineContext,
    pairs: Sequence[tuple[inject.InjectionSpec, PlannedHyperbolaRow]],
    on: inject.InjectionResult,
    off: place.UntaperedInjection,
    patches: Mapping[tuple[str, float], inject.ResponsePatch],
) -> list[ImageEntry]:
    truths_by_id = {t.injection_id: t for t in on.truths}
    on_rejected = dict(on.rejected)
    off_rejected = dict(off.rejected)
    stem = pairs[0][0].realization_id.removesuffix(":on")
    entries = []
    for taper, data in ((TAPER_ON, on.data), (TAPER_OFF, off.data)):
        realization_id = f"{stem}:{taper}"
        truths = []
        for spec, planned in pairs:
            fields = {**_line_fields(plan), **_hyperbola_static(planned, taper, spec)}
            truth = truths_by_id.get(spec.injection_id)
            reason = on_rejected.get(spec.injection_id, "")
            if taper == TAPER_OFF and not reason:
                reason = off_rejected.get(spec.injection_id, "")
            if truth is None or reason:
                fields |= {"accepted": False, "reject_reason": reason or "rejected"}
                truths.append(
                    Truth(
                        spec.injection_id,
                        planned.position_id,
                        False,
                        planned.apex_trace,
                        planned.apex_sample,
                        None,
                        (),
                        fields,
                    )
                )
                continue
            patch = patches[(truth.response_id, truth.k)]
            clipped = (
                truth.n_clipped_cells
                if taper == TAPER_ON
                else off.n_clipped_cells[spec.injection_id]
            )
            fields |= {
                "accepted": True,
                "reject_reason": "",
                "gain": float(truth.gain),
                "noise_rms": float(truth.noise_rms),
                "n_clipped_cells": clipped,
            }
            truths.append(
                Truth(
                    spec.injection_id,
                    planned.position_id,
                    True,
                    float(truth.apex_trace),
                    float(truth.apex_sample),
                    truth.box,
                    place.aperture_end_traces(patch, truth.apex_trace),
                    fields,
                )
            )
        image, fields = _realization_image(context, data, RECIPE_R0, any_truth=truths)
        entries.append(
            ImageEntry(
                realization_id, KIND_HYPERBOLA, RECIPE_R0, image, tuple(truths), fields
            )
        )
    return entries


def _realization_image(
    context: inject.LineContext,
    data: np.ndarray,
    recipe: str,
    any_truth: Sequence[Truth],
) -> tuple[DetectionImage | None, dict]:
    """The image of a realization (``None`` when no truth was accepted)."""
    if not any(truth.accepted for truth in any_truth):
        return None, {}
    injected = injected_radargram(context, data)
    image = make_detection_image(injected, RECIPE_CONFIGS[recipe])
    fields = zero_and_period_fields(context, injected, image)
    fields["compression"] = float(image.compression)
    fields["infer_i0"], fields["infer_i1"] = band_runner.tile_trace_range(
        image, tuple(range(len(image.tiles)))
    )
    return image, fields


# ------------------------------------------------------------------- bands


@dataclass(frozen=True)
class BuiltArm:
    """A planned band arm rebuilt on the line (shift re-measured)."""

    planned: PlannedArm
    band: inject.BandPatch


def rebuild_arm(
    context: inject.LineContext,
    response: inject.PipeResponse,
    planned: PlannedArm,
) -> inject.BandPatch | None:
    """The plan's band again: same shift, length and taper; ``None`` on a mismatch."""
    moved = place.shifted_response(response, planned.shift_samples * context.dt_ns)
    patch = inject.build_response_patch(moved, context, inject.K_MAIN)
    if isinstance(patch, inject.RejectReason):
        return None
    band = inject.build_band_patch(
        patch, context, planned.length_traces, planned.taper_traces
    )
    if isinstance(band, inject.RejectReason):
        return None
    n_samples = context.background.n_samples
    truth = inject.band_truth_box(band, planned.center_trace, n_samples)
    return band if truth == planned.truth else None


def band_spec(
    line_id: str, arm: PlannedArm, texture: str, snr_db: float, polarity: int
) -> inject.InjectionSpec:
    """Band table row of one arm under one condition."""
    stem = f"{line_id}:b:{arm.arm}{arm.row_px:g}:{texture}:snr{snr_db:g}:p{polarity:+d}"
    index = int(arm.position_id.rsplit(":b", 1)[1])
    return inject.InjectionSpec(
        injection_id=f"{stem}:{index:02d}",
        realization_id=stem,
        line_id=line_id,
        apex_trace=arm.center_trace,
        response_id=arm.model,
        snr_db=snr_db,
        polarity=polarity,
        k=inject.K_MAIN,
        seed=place.PLAN_SEED_E3X,
        mode=inject.InjectionMode.BAND,
        band_length_traces=arm.length_traces,
        band_center_trace=arm.center_trace,
    )


def _band_static(arm: PlannedArm, texture: str, recipe: str, spec) -> dict:
    row, position = arm.row, arm.position
    return {
        "position_id": arm.position_id,
        "model": arm.model,
        "model_group": position["model_group"],
        "native_y1_px": position["native_y1_px"],
        "bin_target_px": arm.row_px,
        "shift_samples_first": row["shift_samples_first"],
        "shift_samples": arm.shift_samples,
        "shift_correction_samples": row["shift_correction_samples"],
        "shift_step_samples": row["shift_step_samples"],
        "shift_gap_samples": row["shift_gap_samples"],
        "short_center_offset_traces": row["short_center_offset_traces"],
        "shift_P": row["shift_P"],
        "d_prime_m": row["d_prime_m"],
        "apex_P_below_zero_target": row["apex_P_below_zero_target"],
        "rows_to_record_end_P": row["rows_to_record_end_P"],
        "realized_y1_px_plan": row["realized_y1_px_plan"],
        "apex_depth_P": row["apex_P_below_zero_target"],
        "length_arm": arm.arm,
        "length_traces": arm.length_traces,
        "length_P": row["length_P"],
        "l_over_w_eff": row["l_over_w_eff"],
        "taper_traces": arm.taper_traces,
        "texture": texture,
        "recipe": recipe,
        "snr_db": spec.snr_db,
        "polarity": spec.polarity,
    }


@dataclass
class _Textures:
    """Shared noise field and amplitude multipliers of each band position.

    The first arm that asks for a position's fields (``band_entries`` order)
    also gives its texture check: ``checks`` keeps the pass flag, ``details``
    the whole ``TextureCheck`` (precheck 10 record, v0.5).
    """

    fields: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, place.TextureCheck] = field(default_factory=dict)

    def of(self, built: BuiltArm, period_traces: float, period: float):
        key = built.planned.position_id
        if key not in self.fields:
            band = built.band
            w = place.texture_field(
                band.n_columns,
                band.flat_columns,
                period_traces,
                place.TEXTURE_CORRELATION_PERIODS,
                place.texture_rng(key),
            )
            m = place.amplitude_texture(
                w, band.flat_columns, place.AMPLITUDE_TEXTURE_DB
            )
            self.fields[key] = (w, m)
            check = place.texture_check(band, w, m, period)
            self.details[key] = check
            self.checks[key] = check.passes()
        return self.fields[key]


def textured(
    built: BuiltArm, texture: str, textures: _Textures, context
) -> inject.BandPatch:
    """The arm's band with one texture (``none``: the plain band)."""
    if texture == TEXTURE_NONE:
        return built.band
    w, m = textures.of(built, context.period_traces, context.period_samples)
    if texture == TEXTURE_AMPLITUDE:
        return place.amplitude_textured_band(built.band, m)
    return place.jittered_band(
        built.band, w, TEXTURE_JITTER[texture], context.period_samples
    )


def band_conditions(arm: str, row_px: float) -> list[tuple[str, float, int]]:
    """``(texture, SNR, polarity)`` realizations of one arm (§2-2, R-c)."""
    conditions = [(TEXTURE_NONE, s, p) for s in BAND_SNR_DB for p in POLARITIES]
    if arm == place.LONG_ARM and row_px in place.IN_RANGE_ROWS_PX:
        conditions += [
            (t, s, p) for t in TEXTURES for s in TEXTURE_SNR_DB for p in POLARITIES
        ]
    return conditions


def arm_id(arm: PlannedArm) -> str:
    """``<position>:<arm><row px>`` of one planned arm (precheck 10 failures)."""
    return f"{arm.position_id}:{arm.arm}{arm.row_px:g}"


def band_entries(
    plan: LinePlan,
    context: inject.LineContext,
    responses: Mapping[str, inject.PipeResponse],
    background_images: Mapping[str, DetectionImage],
    images: bool = True,
) -> tuple[list[ImageEntry], list[str], _Textures]:
    """Every band realization of a line: ``(entries, mismatched arms, textures)``.

    Args:
        images: ``False`` (precheck 10): the arms are rebuilt and every
            textured condition asks ``textured`` for its band in the same
            order, so ``_Textures.of`` runs the texture checks exactly as in
            the main run; no realization, image or contrast is made.
    """
    built: dict[tuple[float, str], list[BuiltArm]] = {}
    mismatched = []
    for arm in plan.arms:
        band = rebuild_arm(context, responses[arm.model], arm)
        if band is None:
            mismatched.append(arm_id(arm))
            continue
        built.setdefault((arm.row_px, arm.arm), []).append(BuiltArm(arm, band))
    textures = _Textures()
    section = None
    if images and not context.background.amplitude_processed:
        section = inject.background_step4_section(context)
    entries = []
    for (row_px, arm_name), arms in sorted(built.items()):
        arms = sorted(arms, key=lambda a: a.planned.position_id)
        for texture, snr_db, polarity in band_conditions(arm_name, row_px):
            if not images:
                for built_arm in arms:
                    textured(built_arm, texture, textures, context)
                continue
            entries += _band_realization(
                plan,
                context,
                arms,
                texture,
                snr_db,
                polarity,
                textures,
                section,
                background_images,
            )
    return entries, mismatched, textures


def _band_realization(
    plan: LinePlan,
    context: inject.LineContext,
    arms: Sequence[BuiltArm],
    texture: str,
    snr_db: float,
    polarity: int,
    textures: _Textures,
    section: np.ndarray | None,
    background_images: Mapping[str, DetectionImage],
) -> list[ImageEntry]:
    line_id = plan.line.line_id
    specs = [band_spec(line_id, a.planned, texture, snr_db, polarity) for a in arms]
    placements = [
        (spec, textured(a, texture, textures, context))
        for spec, a in zip(specs, arms, strict=True)
    ]
    result = place.inject_bands(context, placements)
    survival = band_runner.survival_ratios(context, result, section)
    accepted = {
        t.injection_id: (t, s) for t, s in zip(result.truths, survival, strict=True)
    }
    rejected = dict(result.rejected)
    recipes = [RECIPE_R0]
    is_p42_arm = arms[0].planned.arm == place.LONG_ARM and texture == TEXTURE_NONE
    if is_p42_arm and snr_db == P42_SNR_DB:
        recipes.append(RECIPE_P42)
    entries = []
    for recipe in recipes:
        truths = [
            _band_truth(plan, a.planned, spec, accepted, rejected, texture, recipe)
            for spec, a in zip(specs, arms, strict=True)
        ]
        image, fields = _realization_image(context, result.data, recipe, truths)
        if image is not None:
            truths = [
                _with_contrast(t, image, background_images[recipe]) for t in truths
            ]
        realization_id = f"{specs[0].realization_id}:{recipe}"
        entries.append(
            ImageEntry(realization_id, KIND_BAND, recipe, image, tuple(truths), fields)
        )
    return entries


def _band_truth(
    plan: LinePlan,
    arm: PlannedArm,
    spec: inject.InjectionSpec,
    accepted: Mapping[str, tuple[inject.BandTruth, float | str]],
    rejected: Mapping[str, str],
    texture: str,
    recipe: str,
) -> Truth:
    fields = {**_line_fields(plan), **_band_static(arm, texture, recipe, spec)}
    if spec.injection_id not in accepted:
        reason = rejected.get(spec.injection_id, "rejected")
        fields |= {"accepted": False, "reject_reason": reason}
        return Truth(
            spec.injection_id,
            arm.position_id,
            False,
            arm.center_trace,
            math.nan,
            None,
            (),
            fields,
        )
    truth, survival = accepted[spec.injection_id]
    fields |= {
        "accepted": True,
        "reject_reason": "",
        "gain": float(truth.gain),
        "noise_rms": float(truth.noise_rms),
        "survival_ratio": survival,
        "truth_length": truth.box.n_traces,
    }
    return Truth(
        spec.injection_id,
        arm.position_id,
        True,
        arm.center_trace,
        float(truth.apex_sample),
        truth.box,
        (),
        fields,
    )


def _contrast(image: DetectionImage, box: GridBox, taper_traces: int) -> dict | None:
    grid = image.grid_map
    q = place.centred_grey(image.line_image, image.content_height_px)
    box_px = place.truth_box_px(box, grid.traces_per_px, grid.samples_per_px)
    side_gap = math.ceil(taper_traces / grid.traces_per_px)
    return place.band_contrast(q, box_px, image.content_height_px, side_gap)


def _with_contrast(
    truth: Truth, image: DetectionImage, background: DetectionImage
) -> Truth:
    """Contrast of the injected band and of the same box on the background."""
    if not truth.accepted or truth.box is None:
        return truth
    taper = int(truth.fields["taper_traces"])
    fields = dict(truth.fields)
    fields["realized_y1_px_image"] = truth.box.k0 / image.grid_map.samples_per_px
    for suffix, source in (("", image), ("_bg", background)):
        values = _contrast(source, truth.box, taper) or {}
        for name in ("c_lat", "c_vert"):
            fields[f"{name}{suffix}"] = _exact(values.get(f"{name}_db"))
        fields[f"coh{suffix}"] = _exact(values.get("coh"))
    return replace(truth, fields=fields)


# ----------------------------------------------------------- one line (CPU)


def build_line(
    plan: LinePlan,
    hyperbola_library: responses_io.ResponseLibrary,
    band_library: responses_io.ResponseLibrary,
    images: bool = True,
) -> LineRealizations:
    """Every realization image of one line, with the CPU-side checks.

    Args:
        images: ``False`` is precheck 10 (v0.5): the same body counts the five
            CPU assertions (``CPU_ASSERTIONS``) through the same calls in the
            same order, but makes no realization image, background image,
            band realization or contrast (the zero test still makes its one
            background image); ``entries`` is then empty.
    """
    line = runner.load_line(plan.line.ref)
    context = inject.prepare_line_context(line.radargram, plan.exclusions)
    background = {}
    if images:
        background = {
            recipe: make_detection_image(line.radargram, config)
            for recipe, config in RECIPE_CONFIGS.items()
        }
    hyperbola_responses = line_responses(hyperbola_library, line, HYPERBOLA_MODELS)
    hyperbolas, equal = hyperbola_entries(plan, context, hyperbola_responses, images)
    patches = hyperbola_patches(plan, context, hyperbola_responses)
    edge_checks = r_edge_matches(plan, context, patches)
    bands, mismatched, textures = band_entries(
        plan,
        context,
        line_responses(band_library, line, BAND_MODELS),
        background,
        images,
    )
    defects = Counter()
    defects["shift_mismatch"] = len(mismatched)
    defects["off_differs_from_on"] = sum(not value for value in equal.values())
    defects["texture_check"] = sum(not value for value in textures.checks.values())
    zero_ok = zero_test_passes(context)
    defects["zero_test"] = int(not zero_ok)
    defects["r_edge_differs_from_plan"] = sum(not v for v in edge_checks.values())
    failures = [
        *(_failure("shift_mismatch", item) for item in mismatched),
        *(_failure("off_differs_from_on", k) for k, ok in equal.items() if not ok),
        *r_edge_failures(plan, context, patches, edge_checks),
        *_texture_failures(textures),
    ]
    if not zero_ok:
        failures.append(_failure("zero_test", plan.line.line_id))
    return LineRealizations(
        plan=plan,
        line_fields=_line_fields(plan),
        period_samples=float(context.period_samples),
        period_traces=float(context.period_traces),
        zero=context.zero,
        entries=tuple(hyperbolas + bands),
        defects=dict(defects),
        texture_checks=dict(textures.checks),
        off_on_equal=equal,
        r_edge_matches=edge_checks,
        zero_test=zero_ok,
        n_checked={
            "shift_mismatch": len(plan.arms),
            "off_differs_from_on": len(equal),
            "r_edge_differs_from_plan": len(edge_checks),
            "texture_check": len(textures.checks),
            "zero_test": 1,
        },
        failed_items=tuple(failures),
        texture_details=dict(textures.details),
    )


def _texture_failures(textures: _Textures) -> list[dict]:
    return [
        _failure("texture_check", key, textures.details[key].as_record())
        for key, passed in textures.checks.items()
        if not passed
    ]


def _init_line_worker() -> None:
    _WORKER["hyperbola"] = load_hyperbola_library()
    _WORKER["band"] = responses_io.load_library()


def _line_worker(plan: LinePlan) -> LineRealizations:
    return build_line(plan, _WORKER["hyperbola"], _WORKER["band"])  # type: ignore[arg-type]


# ========================================================= inference (GPU)


@dataclass(frozen=True, eq=False)
class LineDetections:
    """Tile boxes of every image and fold, and the NMS warnings per fold."""

    boxes: Mapping[tuple[int, int], list[LineBox]]
    nms_warnings: Mapping[int, int]


def infer_entries(
    entries: Sequence[ImageEntry],
    detectors: Mapping[int, object],
    counter: WarningCounter | None,
    batch_size: int = BATCH_SIZE,
) -> LineDetections:
    """One conf for every image of a line, fold by fold, in shared batches."""
    jobs = [
        runner.ImageJob(
            entry.recipe,
            runner.ORIGINAL,
            entry.image,
            tuple(range(len(entry.image.tiles))),
            index,
        )
        for index, entry in enumerate(entries)
        if entry.image is not None
    ]
    boxes: dict[tuple[int, int], list[LineBox]] = {}
    warnings: dict[int, int] = {}
    for fold, detector in detectors.items():
        if counter is not None:
            counter.take()
        found = runner.run_jobs(jobs, {fold: detector}, batch_size)
        for (j, _), tile_boxes in found.items():
            boxes[(jobs[j].tag, fold)] = tile_boxes
        warnings[fold] = 0 if counter is None else counter.take()
    return LineDetections(boxes, warnings)


def hyperbola_fold_fields(
    entry: ImageEntry,
    boxes: FoldBoxes,
    line: LineRealizations,
    config_counts: Mapping[str, int],
) -> list[dict]:
    """Fold fields of every accepted hyperbola truth of one image."""
    accepted = [t for t in entry.truths if t.accepted]
    apexes = [
        match.TruthApex(t.injection_id, t.apex_trace, t.apex_sample) for t in accepted
    ]
    s0, s3 = scored(boxes.merged), scored(boxes.merged_s3)
    period_s, period_t = line.period_samples, line.period_traces
    outcomes = match.match_one_to_one(apexes, s0, period_t, period_s)
    outcomes_s3 = match.match_one_to_one(apexes, s3, period_t, period_s)
    merged_grids = [grid for _, grid in boxes.merged]
    rows = []
    for truth, apex, outcome, outcome_s3 in zip(
        accepted, apexes, outcomes, outcomes_s3, strict=True
    ):
        best = best_scores(
            boxes.tile_boxes,
            entry.image,
            apex_accepts(apex, period_s),
            DetectionClass.PIPE_HYPERBOLA,
            "hyp",
        )
        hit_score = (
            "" if outcome.box_index is None else float(s0[outcome.box_index].score)
        )
        rows.append(
            {
                "injection_id": truth.injection_id,
                "hit": outcome.hit,
                "hit_score": hit_score,
                "n_assigned_boxes": outcome.n_boxes,
                "hit_contain": match.apex_hit_d3(
                    truth.apex_trace, truth.apex_sample, s0, period_s
                ),
                "hit_s3": outcome_s3.hit,
                "n_edge_boxes": place.boxes_near_traces(
                    merged_grids, truth.edge_traces, period_t
                ),
                **best,
                **config_counts,
            }
        )
    return rows


def band_fold_fields(
    entry: ImageEntry, boxes: FoldBoxes, config_counts: Mapping[str, int]
) -> list[dict]:
    """Fold fields of every accepted band truth of one image."""
    rows = []
    for truth in entry.truths:
        if not truth.accepted or truth.box is None:
            continue
        length = truth.box.n_traces
        covered = band_cover(truth.box, boxes.merged, DetectionClass.PIPE_BAND)
        rows.append(
            {
                "injection_id": truth.injection_id,
                "covered_traces": covered,
                "truth_length": length,
                "coverage": covered / length,
                "covered_any_class": band_cover(truth.box, boxes.merged, None),
                "coverage_s3": band_cover(
                    truth.box, boxes.merged_s3, DetectionClass.PIPE_BAND
                )
                / length,
                **best_scores(
                    boxes.tile_boxes,
                    entry.image,
                    band_accepts(truth.box),
                    DetectionClass.PIPE_BAND,
                    "band",
                ),
                **config_counts,
            }
        )
    return rows


def _fold_defects(kind: str, rows: Sequence[Mapping], boxes: FoldBoxes) -> Counter:
    """§5-4 assertions on one image and fold."""
    defects: Counter[str] = Counter()
    defects["tile_with_300_operating_boxes"] += boxes.n_tiles_operating_maxdet
    for row in rows:
        if kind == KIND_HYPERBOLA:
            best = row["best_score_001_hyp"]
            if row["hit"] and not (best != "" and best > CONF_OPERATING):
                defects["hit_without_operating_box"] += 1
        else:
            best = row["best_score_001_band"]
            if row["covered_traces"] > 0 and not (best != "" and best > CONF_OPERATING):
                defects["cover_without_operating_box"] += 1
            if row["covered_traces"] > row["truth_length"]:
                defects["coverage_over_length"] += 1
    return defects


def line_records(
    line: LineRealizations,
    detections: LineDetections,
    folds: Sequence[int],
    config: PPConfig,
) -> tuple[dict[str, list[dict]], Counter]:
    """``hyperbola``, ``band`` and ``realization`` rows of one line, and defects."""
    records: dict[str, list[dict]] = {
        KIND_HYPERBOLA: [],
        KIND_BAND: [],
        "realizations": [],
    }
    defects: Counter[str] = Counter(line.defects)
    for index, entry in enumerate(line.entries):
        records["realizations"].append(
            {
                "realization_id": entry.realization_id,
                "line_id": line.plan.line.line_id,
                "kind": entry.kind,
                "recipe": entry.recipe,
                "n_truths": len(entry.truths),
                "n_accepted": sum(t.accepted for t in entry.truths),
                "n_tiles": 0 if entry.image is None else len(entry.image.tiles),
                **entry.fields,
            }
        )
        for fold in folds:
            records[entry.kind] += _entry_fold_rows(
                line, entry, index, fold, detections, config, defects
            )
    for fold, count in detections.nms_warnings.items():
        defects["nms_time_limit_warning"] += count
        if count:
            logger.error("fold %d: %d NMS time-limit warnings", fold, count)
    return records, defects


def _entry_fold_rows(
    line: LineRealizations,
    entry: ImageEntry,
    index: int,
    fold: int,
    detections: LineDetections,
    config: PPConfig,
    defects: Counter,
) -> list[dict]:
    base = {"realization_id": entry.realization_id, "fold": fold}
    if entry.image is None:
        return [base | dict(t.fields) | entry.fields for t in entry.truths]
    boxes = fold_boxes(entry.image, detections.boxes[(index, fold)], line.zero, config)
    counts = {
        "n_tiles_maxdet": boxes.n_tiles_maxdet,
        "n_nms_timeout_warnings": detections.nms_warnings.get(fold, 0),
    }
    if entry.kind == KIND_HYPERBOLA:
        fold_rows = hyperbola_fold_fields(entry, boxes, line, counts)
    else:
        fold_rows = band_fold_fields(entry, boxes, counts)
    defects.update(_fold_defects(entry.kind, fold_rows, boxes))
    by_id = {row["injection_id"]: row for row in fold_rows}
    rows = []
    for truth in entry.truths:
        extra = by_id.get(truth.injection_id, {})
        rows.append(base | dict(truth.fields) | entry.fields | extra | counts)
    return rows


# ------------------------------------------------------------ record files


ZD_COLUMNS = (
    "dk_zero_samples",
    "dk_zero_over_p",
    "zero_label_changed",
    "zero_failed_changed",
    "zero_rule_changed",
    "zero_events_changed",
    "zero_changed",
    "zero_changed_exact",
    "dp_samples",
    "dp_rel",
    "period_failed",
    "period_changed",
    "period_changed_exact",
)
HYPERBOLA_RECORD_COLUMNS = (
    "realization_id",
    "line_id",
    "dataset",
    "cluster_id",
    "shares_r3_activity",
    "position_id",
    "model",
    "model_group",
    "group",
    "apex_trace",
    "apex_sample",
    "apex_depth_P",
    "apex_y_px",
    "r_edge_db",
    "edge_stratum",
    "edge_snr_set",
    "taper",
    "snr_db",
    "polarity",
    "accepted",
    "reject_reason",
    "gain",
    "noise_rms",
    "n_clipped_cells",
    "fold",
    "hit",
    "hit_score",
    "n_assigned_boxes",
    "hit_contain",
    "hit_s3",
    "best_score_001_hyp",
    "best_score_001_hyp_no_box",
    "best_score_001_any",
    "best_score_001_any_no_box",
    "n_edge_boxes",
    "period_bg",
    "period_inj",
    "infer_i0",
    "infer_i1",
    "n_tiles_maxdet",
    "n_nms_timeout_warnings",
    *ZD_COLUMNS,
)
BAND_RECORD_COLUMNS = (
    "realization_id",
    "line_id",
    "dataset",
    "cluster_id",
    "shares_r3_activity",
    "position_id",
    "model",
    "model_group",
    "native_y1_px",
    "bin_target_px",
    "shift_samples_first",
    "shift_samples",
    "shift_correction_samples",
    "shift_step_samples",
    "shift_gap_samples",
    "short_center_offset_traces",
    "shift_P",
    "d_prime_m",
    "apex_P_below_zero_target",
    "rows_to_record_end_P",
    "realized_y1_px_plan",
    "realized_y1_px_image",
    "compression",
    "apex_depth_P",
    "length_arm",
    "length_traces",
    "length_P",
    "l_over_w_eff",
    "taper_traces",
    "texture",
    "recipe",
    "snr_db",
    "polarity",
    "accepted",
    "reject_reason",
    "gain",
    "noise_rms",
    "fold",
    "covered_traces",
    "truth_length",
    "coverage",
    "covered_any_class",
    "coverage_s3",
    "best_score_001_band",
    "best_score_001_band_no_box",
    "best_score_001_any",
    "best_score_001_any_no_box",
    "survival_ratio",
    "c_lat",
    "c_vert",
    "coh",
    "c_lat_bg",
    "c_vert_bg",
    "coh_bg",
    "period_bg",
    "period_inj",
    "n_tiles_maxdet",
    "n_nms_timeout_warnings",
    *ZD_COLUMNS,
)
REALIZATION_COLUMNS = (
    "realization_id",
    "line_id",
    "kind",
    "recipe",
    "n_truths",
    "n_accepted",
    "n_tiles",
    "compression",
    "infer_i0",
    "infer_i1",
    "period_bg",
    "period_inj",
    *ZD_COLUMNS,
)
UNINJECTED_COLUMNS = (
    "line_id",
    "dataset",
    "cluster_id",
    "fold",
    "kind",
    "position_id",
    "bin_target_px",
    "length_arm",
    "best_score_001_hyp",
    "best_score_001_hyp_no_box",
    "best_score_001_band",
    "best_score_001_band_no_box",
    "best_score_001_any",
    "best_score_001_any_no_box",
    "n_tiles_maxdet",
    "n_nms_timeout_warnings",
)


def write_record_files(
    stage_dir: Path, parts: Sequence[Mapping], folds: Sequence[int]
) -> dict[str, str]:
    """Concatenate the line parts; returns the raw sha256 of each file."""
    files = {
        "hyperbola_records.csv": (HYPERBOLA_RECORD_COLUMNS, KIND_HYPERBOLA),
        "band_records.csv": (BAND_RECORD_COLUMNS, KIND_BAND),
        "realizations.csv": (REALIZATION_COLUMNS, "realizations"),
    }
    hashes = {}
    for name, (columns, key) in files.items():
        rows = [row for part in parts for row in part["records"][key]]
        write_rows(stage_dir / name, columns, rows)
        hashes[name] = inputs.sha256_file(stage_dir / name)
    first_fold = min(folds)
    rejections = [
        {
            "realization_id": row["realization_id"],
            "position_id": row["position_id"],
            "kind": key,
            "reason": row["reject_reason"],
        }
        for part in parts
        for key in (KIND_HYPERBOLA, KIND_BAND)
        for row in part["records"][key]
        if row["fold"] == first_fold and not row["accepted"]
    ]
    write_rows(
        stage_dir / "rejections.csv",
        ("realization_id", "position_id", "kind", "reason"),
        rejections,
    )
    hashes["rejections.csv"] = inputs.sha256_file(stage_dir / "rejections.csv")
    truths = [
        {"kind": key, **row}
        for part in parts
        for key in (KIND_HYPERBOLA, KIND_BAND)
        for row in part["records"][key]
        if row["fold"] == first_fold and row["accepted"]
    ]
    write_rows(stage_dir / "truths.csv", TRUTH_COLUMNS, truths)
    hashes["truths.csv"] = inputs.sha256_file(stage_dir / "truths.csv")
    return hashes


TRUTH_COLUMNS = (
    "kind",
    "realization_id",
    "line_id",
    "dataset",
    "position_id",
    "model",
    "taper",
    "bin_target_px",
    "length_arm",
    "texture",
    "recipe",
    "snr_db",
    "polarity",
    "gain",
    "noise_rms",
    "n_clipped_cells",
    "survival_ratio",
)
"""``truths.csv``: one fold-free row per accepted truth (A-3)."""


# ============================================================ stage runs


@dataclass(frozen=True)
class StageInputs:
    """Inputs asserted before a detector stage, and their hashes."""

    plans: list[LinePlan]
    hashes: dict[str, str]
    seal: dict


def detector_stage_inputs(settings: Settings, context: Context) -> StageInputs:
    """Pool, seal-2 record, plan and weights of a detector stage."""
    hashes = pool_input_hashes(settings)
    seal = assert_seal2(settings, context)
    hashes["seal_record"] = seal["seal_record_sha256_lf"]
    hashes.update(seal["plan_files_sha256_lf"])
    hashes.update(runner.assert_weights(settings.run_root, settings.folds))
    plans = read_plan(
        SEALED_PLAN_DIR,
        background_pool(settings),
        inputs.candidate_apexes(),
    )
    return StageInputs(plans, hashes, seal)


def _part_path(settings: Settings, line: PoolLine) -> Path:
    return settings.stage_dir / "parts" / f"{line.ref.key}.json"


def _reusable_part(settings: Settings, line: PoolLine) -> dict | None:
    path = _part_path(settings, line)
    if not (settings.resume and path.is_file()):
        return None
    part = json.loads(path.read_text(encoding="utf-8"))
    if part.get("head") != runner.recorded_head(settings):  # type: ignore[arg-type]
        raise inputs.InputError(f"part {path.name} has another HEAD")
    return part


def run_injected(
    settings: Settings, started: float, context: Context
) -> tuple[str, Counts]:
    """Build, infer and record every realization of the plan (§5-1)."""
    assert_retry_output(settings)
    _, config = sealed_config(context)
    staged = detector_stage_inputs(settings, context)
    prepare_stage_dir(settings)
    check_resume(settings, staged.hashes, context.start_tree)
    identity = new_identity(settings, context, staged.hashes)
    identity["seal2"] = staged.seal
    detectors = load_detectors(settings, CONF_AP)
    counter = attach_nms_counter()
    progress = Progress(
        settings.stage_dir / PROGRESS_FILE, Stage.INJECTED.value, len(staged.plans)
    )
    parts, defects = evaluate_lines(
        settings, staged.plans, detectors, counter, config, started, progress=progress
    )
    status = STATUS_OK if not sum(defects.values()) else STATUS_DEFECT
    status, judgment = close_run(settings, context, identity, status)
    hashes = write_record_files(settings.stage_dir, parts, settings.folds)
    counts = summary_counts(parts)
    summary = {
        "schema": SCHEMA,
        "stage": Stage.INJECTED.value,
        "status": status,
        "judgment": judgment,
        "run_identity": {**identity, "finished_at": common.local_now()},
        "folds": list(settings.folds),
        "defects": dict(defects),
        "nms_time_limit_warnings": defects.get("nms_time_limit_warning", 0),
        "tiles_maxdet": sum(part["n_tiles_maxdet"] for part in parts),
        "outputs_sha256": hashes,
        "zd_summary": zd_summary(parts),
        "counts": asdict(counts),
        "resources": runner.resources(settings, started),  # type: ignore[arg-type]
    }
    write_json(settings.stage_dir / "summary.json", summary)
    print(
        f"defect checks: {sum(defects.values())}; "
        f"NMS warnings: {summary['nms_time_limit_warnings']}; "
        f"max-det tiles: {summary['tiles_maxdet']}",
        flush=True,
    )
    return status, counts


def evaluate_lines(
    settings: Settings,
    plans: Sequence[LinePlan],
    detectors: Mapping[int, object],
    counter: WarningCounter | None,
    config: PPConfig,
    started: float,
    keep_built: list[LineRealizations] | None = None,
    progress: Progress | None = None,
) -> tuple[list[dict], Counter]:
    """CPU realizations in workers, GPU inference here, one part per line.

    Args:
        keep_built: When given, every built line is appended (precheck).
        progress: When given, updated after every line (reused parts count
            as done).
    """
    head = runner.recorded_head(settings)  # type: ignore[arg-type]
    parts: dict[str, dict] = {}
    todo = []
    for plan in plans:
        reused = _reusable_part(settings, plan.line)
        if reused is None:
            todo.append(plan)
        else:
            parts[plan.line.line_id] = reused
    if progress is not None:
        progress.update(len(parts))
    for number, built in enumerate(_built_lines(settings, todo), start=1):
        if keep_built is not None:
            keep_built.append(built)
        line_started = time.time()
        detections = infer_entries(
            built.entries, detectors, counter, settings.batch_size
        )
        records, defects = line_records(built, detections, settings.folds, config)
        part = {
            "line_id": built.plan.line.line_id,
            "head": head,
            "records": records,
            "defects": dict(defects),
            "n_positions": len(built.plan.hyperbolas)
            + len(built.plan.band_position_ids),
            "n_realizations": len(built.entries),
            "n_inferences": sum(e.image is not None for e in built.entries),
            "n_tiles_maxdet": sum(
                row.get("n_tiles_maxdet", 0) or 0
                for key in (KIND_HYPERBOLA, KIND_BAND)
                for row in records[key]
            ),
            "seconds": round(time.time() - line_started, 2),
        }
        runner._write_part(_part_path(settings, built.plan.line), part)
        parts[built.plan.line.line_id] = part
        if progress is not None:
            progress.update(len(parts))
        print(
            f"injected {number}/{len(todo)}: {part['n_inferences']} images, "
            f"{part['seconds']} s (elapsed {time.time() - started:.0f} s)",
            flush=True,
        )
    ordered = [parts[plan.line.line_id] for plan in plans]
    total: Counter[str] = Counter()
    for part in ordered:
        total.update(part["defects"])
    return ordered, total


PREFETCH_LINES_PER_WORKER = 2
"""Lines built ahead per worker; bounds the images waiting for the GPU."""


def _built_lines(
    settings: Settings, plans: Sequence[LinePlan]
) -> Iterator[LineRealizations]:
    """Line realizations in plan order (worker processes when ``workers > 1``).

    At most ``PREFETCH_LINES_PER_WORKER x workers`` lines are built ahead,
    so the images of a line wait in memory only until the GPU side takes it.
    """
    if settings.workers == 1:
        _init_line_worker()
        for plan in plans:
            yield _line_worker(plan)
        return
    ahead = PREFETCH_LINES_PER_WORKER * settings.workers
    remaining = iter(plans)
    with ProcessPoolExecutor(
        max_workers=settings.workers, initializer=_init_line_worker
    ) as executor:
        pending = deque(
            executor.submit(_line_worker, plan) for plan in islice(remaining, ahead)
        )
        while pending:
            built = pending.popleft().result()
            following = next(remaining, None)
            if following is not None:
                pending.append(executor.submit(_line_worker, following))
            yield built


def summary_counts(parts: Sequence[Mapping]) -> Counts:
    return Counts(
        n_lines=len(parts),
        n_positions=sum(p["n_positions"] for p in parts),
        n_realizations=sum(p["n_realizations"] for p in parts),
        n_inferences=sum(p["n_inferences"] for p in parts),
    )


def zd_summary(parts: Sequence[Mapping]) -> dict:
    """[ZD] counts over the realizations (``n_zero_changed`` and the rest)."""
    rows = [row for part in parts for row in part["records"]["realizations"]]
    with_image = [row for row in rows if "zero_changed" in row]

    def count(key: str) -> int:
        return sum(bool(row[key]) for row in with_image)

    return {
        "n_realizations_with_image": len(with_image),
        "n_zero_changed": count("zero_changed"),
        "n_zero_changed_exact": count("zero_changed_exact"),
        "n_period_changed": count("period_changed"),
        "n_period_changed_exact": count("period_changed_exact"),
        "n_period_failed": count("period_failed"),
    }


# ------------------------------------------------------------ uninjected


def uninjected_line(
    plan: LinePlan,
    detectors: Mapping[int, object],
    counter: WarningCounter | None,
    settings: Settings,
    config: PPConfig,
) -> dict:
    """Best pre-merge 0.001 scores at every planned truth of one un-injected line."""
    line = runner.load_line(plan.line.ref)
    context = inject.prepare_line_context(line.radargram, plan.exclusions)
    image = make_detection_image(line.radargram, RECIPE_CONFIGS[RECIPE_R0])
    entry = ImageEntry(plan.line.line_id, KIND_UNINJECTED, RECIPE_R0, image, (), {})
    detections = infer_entries([entry], detectors, counter, settings.batch_size)
    rows, merged = [], {}
    operating_maxdet = 0
    for fold in settings.folds:
        boxes = fold_boxes(image, detections.boxes[(0, fold)], context.zero, config)
        operating_maxdet += boxes.n_tiles_operating_maxdet
        merged[fold] = [(int(b.class_id), g) for b, g in boxes.merged]
        base = {
            **_line_fields(plan),
            "fold": fold,
            "n_tiles_maxdet": boxes.n_tiles_maxdet,
            "n_nms_timeout_warnings": detections.nms_warnings.get(fold, 0),
        }
        for h in plan.hyperbolas:
            apex = match.TruthApex(h.position_id, h.apex_trace, h.apex_sample)
            rows.append(
                base
                | {"kind": KIND_HYPERBOLA, "position_id": h.position_id}
                | best_scores(
                    boxes.tile_boxes,
                    image,
                    apex_accepts(apex, context.period_samples),
                    DetectionClass.PIPE_HYPERBOLA,
                    "hyp",
                )
            )
        for arm in plan.arms:
            rows.append(
                base
                | {
                    "kind": KIND_BAND,
                    "position_id": arm.position_id,
                    "bin_target_px": arm.row_px,
                    "length_arm": arm.arm,
                }
                | best_scores(
                    boxes.tile_boxes,
                    image,
                    band_accepts(arm.truth),
                    DetectionClass.PIPE_BAND,
                    "band",
                )
            )
    return {
        "line_id": plan.line.line_id,
        "rows": rows,
        "merged": merged,
        "nms_warnings": dict(detections.nms_warnings),
        "n_tiles": len(image.tiles),
        "operating_maxdet": operating_maxdet,
    }


def reference_boxes(settings: Settings) -> dict[tuple[str, int], Counter]:
    """R2 / R3 un-injected merged boxes (PP off): ``(line, fold) -> boxes``."""
    found: dict[tuple[str, int], Counter] = {}
    for root in (settings.r2_root, settings.r3_root):
        for path in sorted(root.glob("fold*/boxes.csv")):
            for row in inputs.read_csv(path):
                if row.get("orientation", runner.ORIGINAL) != runner.ORIGINAL:
                    continue
                if row.get("pp", band_runner.PP_PRE) != band_runner.PP_PRE:
                    continue
                if row["in_grid"] != "True" or not float(row["score"]) > CONF_OPERATING:
                    continue
                key = (row["line_id"], int(row["fold"]))
                box = (
                    int(row["class_id"]),
                    int(row["i0"]),
                    int(row["i1"]),
                    int(row["k0"]),
                    int(row["k1"]),
                )
                found.setdefault(key, Counter())[box] += 1
    return found


def agreement_counts(
    parts: Sequence[Mapping], reference: Mapping[tuple[str, int], Counter]
) -> dict:
    """Merged operating boxes equal to the R2 / R3 boxes (report only)."""
    same = only_here = only_there = 0
    for part in parts:
        for fold, boxes in part["merged"].items():
            here = Counter((c, g.i0, g.i1, g.k0, g.k1) for c, g in boxes)
            there = reference.get((part["line_id"], int(fold)), Counter())
            common_ = here & there
            same += sum(common_.values())
            only_here += sum((here - common_).values())
            only_there += sum((there - common_).values())
    return {
        "same": same,
        "only_e3x": only_here,
        "only_r2_r3": only_there,
        "note": "batch composition differs from R2/R3; bit equality is not expected",
    }


def run_uninjected001(
    settings: Settings, started: float, context: Context
) -> tuple[str, Counts]:
    """conf 0.001 on the background lines, best scores at the planned truths."""
    assert_retry_output(settings)
    _, config = sealed_config(context)
    staged = detector_stage_inputs(settings, context)
    prepare_stage_dir(settings)
    check_resume(settings, staged.hashes, context.start_tree)
    identity = new_identity(settings, context, staged.hashes)
    identity["seal2"] = staged.seal
    detectors = load_detectors(settings, CONF_AP)
    counter = attach_nms_counter()
    progress = Progress(
        settings.stage_dir / PROGRESS_FILE,
        Stage.UNINJECTED001.value,
        len(staged.plans),
    )
    progress.update(0)
    parts = []
    for number, plan in enumerate(staged.plans, start=1):
        part = uninjected_line(plan, detectors, counter, settings, config)
        parts.append(part)
        progress.update(number)
        print(
            f"uninjected001 {number}/{len(staged.plans)}: {part['n_tiles']} tiles",
            flush=True,
        )
    warnings = sum(sum(p["nms_warnings"].values()) for p in parts)
    operating_maxdet = sum(p["operating_maxdet"] for p in parts)
    status = STATUS_OK if warnings == 0 and operating_maxdet == 0 else STATUS_DEFECT
    status, judgment = close_run(settings, context, identity, status)
    rows = [row for part in parts for row in part["rows"]]
    write_rows(
        settings.stage_dir / "uninjected001_records.csv", UNINJECTED_COLUMNS, rows
    )
    summary = {
        "schema": SCHEMA,
        "stage": Stage.UNINJECTED001.value,
        "status": status,
        "judgment": judgment,
        "run_identity": {**identity, "finished_at": common.local_now()},
        "folds": list(settings.folds),
        "nms_time_limit_warnings": warnings,
        "tiles_with_300_operating_boxes": operating_maxdet,
        "agreement_with_r2_r3": agreement_counts(parts, reference_boxes(settings)),
        "outputs_sha256": {
            "uninjected001_records.csv": inputs.sha256_file(
                settings.stage_dir / "uninjected001_records.csv"
            )
        },
        "resources": runner.resources(settings, started),  # type: ignore[arg-type]
    }
    write_json(settings.stage_dir / "summary.json", summary)
    n_positions = sum(
        len(p.hyperbolas) + len(p.band_position_ids) for p in staged.plans
    )
    return status, Counts(len(parts), n_positions, 0, len(parts))


# ================================================================ precheck


PRECHECK_TARGETS = 1
"""Precheck hyperbola positions per model and background (two lines only)."""


def _precheck_plan(
    settings: Settings, lines: Sequence[PoolLine], plan_dir: Path
) -> dict:
    """Plan the two precheck lines with the main-plan functions (tables kept)."""
    boxes, _ = zone_c_boxes(settings)
    candidates = inputs.candidate_apexes()
    jobs = [
        LineJob(line, runner.line_exclusions(line.line_id, boxes, candidates))
        for line in lines
    ]
    _init_plan_worker()
    parts = [_plan_worker(job) for job in jobs]
    by_line = {part.line_row["line_id"]: part for part in parts}
    groups = dict(responses_io.MODEL_GROUPS)
    targets = {key: PRECHECK_TARGETS for key in hyperbola_target_map(lines, groups)}
    choice = place.plan_hyperbola_positions(
        [c for part in parts for c in part.candidates], targets
    )
    rows = hyperbola_rows(
        choice, by_line, {line.line_id: line for line in lines}, groups
    )
    positions = [row for part in parts for row in part.band_positions]
    arms = [row for part in parts for row in part.band_arms]
    write_rows(plan_dir / "hyperbola_positions.csv", HYPERBOLA_COLUMNS, rows)
    write_rows(plan_dir / "band_positions.csv", BAND_POSITION_COLUMNS, positions)
    write_rows(plan_dir / "band_arms.csv", BAND_ARM_COLUMNS, arms)
    inputs.write_exclusions(plan_dir / "exclusions.csv", boxes)
    return {
        "n_shift_mismatch": sum(part.n_shift_mismatch for part in parts),
        "arms": arms,
        "positions": positions,
    }


def shift_check(plan_info: Mapping) -> bool:
    """§5-2 check 3 (v0.4) on the planned arms.

    Every corrected top within one sample of its target, one
    ``k1'(n2) - n2`` (= ``k0 - shift_samples``) per position, ``|n2| <= 12 P``
    and one ``T`` per position. The correction and step counts are written
    to ``precheck.json`` by the caller (:func:`shift_summary`).
    """
    if plan_info["n_shift_mismatch"]:
        return False
    periods = {
        row["position_id"]: float(row["period_bg_samples"])
        for row in plan_info["positions"]
    }
    tapers: dict[str, set[int]] = {}
    offsets: dict[str, set[int]] = {}
    for arm in plan_info["arms"]:
        # The plan's own product-form rule (exactly 12 P allowed).
        shift = int(arm["shift_samples"])
        if not place.within_shift_limit(shift, periods[arm["position_id"]]):
            return False
        if abs(float(arm["shift_gap_samples"])) > place.SHIFT_MATCH_SAMPLES:
            return False
        position = arm["position_id"]
        tapers.setdefault(position, set()).add(int(arm["taper_traces"]))
        offsets.setdefault(position, set()).add(
            int(arm["k0"]) - int(arm["shift_samples"])
        )
    same_taper = all(len(values) == 1 for values in tapers.values())
    return same_taper and all(len(values) == 1 for values in offsets.values())


def shift_summary(plan_info: Mapping) -> dict:
    """Correction and step counts of the planned arms (precheck report, v0.4)."""
    arms = plan_info["arms"]
    return {
        "shift_correction_samples": runner.count_by(
            arms, ("shift_correction_samples",)
        ),
        "shift_step_samples": runner.count_by(arms, ("shift_step_samples",)),
    }


def _same_boxes(first: Sequence[LineBox], second: Sequence[LineBox]) -> bool:
    def key(box: LineBox) -> tuple:
        b = box.box
        return (box.tile_index, box.class_id, box.score, b.x1, b.y1, b.x2, b.y2)

    return [key(b) for b in first] == [key(b) for b in second]


def conf_paths_equal(
    entries: Sequence[ImageEntry],
    low: LineDetections,
    operating: LineDetections,
    zero: ZeroResult,
    config: PPConfig,
) -> bool:
    """§5-2 check 6: 0.001 cut at 0.25 before the merge == the 0.25 path."""
    for (index, fold), boxes in low.boxes.items():
        image = entries[index].image
        assert image is not None
        reference = operating.boxes[(index, fold)]
        if not _same_boxes(operating_boxes(boxes), reference):
            return False
        a = fold_boxes(image, boxes, zero, config)
        b = fold_boxes(image, reference, zero, config)
        if [(x.class_id, x.score, x.box, g) for x, g in a.merged] != [
            (x.class_id, x.score, x.box, g) for x, g in b.merged
        ]:
            return False
    return True


NMS_DEFECT_KEYS = ("nms_time_limit_warning", "tile_with_300_operating_boxes")
"""Check 9: the defects counted by the NMS handler and the max-det counter."""


def _precheck_identity(settings: Settings, context: Context) -> tuple[bool, dict]:
    """Check 1: seal-2 record (code blobs, ``src`` tree, plan), weights, inputs.

    The settings hash, r0 identity lines, ``inject.py`` blob and sealed PP
    were asserted by :func:`start_context` before this point.
    """
    try:
        hashes = pool_input_hashes(settings)
        hashes["seal2"] = assert_seal2(settings, context)["seal_record_sha256_lf"]
        hashes.update(runner.assert_weights(settings.run_root, settings.folds))
    except inputs.InputError as error:
        logger.error("precheck 1: %s", error)
        return False, {}
    return True, hashes


def _precheck_conf_paths(
    built: Sequence[LineRealizations],
    detectors: tuple[Mapping[int, object], Mapping[int, object]],
    counter: WarningCounter,
    config: PPConfig,
    progress: Progress | None = None,
) -> bool:
    """Check 6 on every arm image and the un-injected image of each line."""
    low, operating = detectors
    for number, line in enumerate(built):
        if progress is not None:
            progress.update(number)
        radargram = runner.load_line(line.plan.line.ref).radargram
        background = make_detection_image(radargram, RECIPE_CONFIGS[RECIPE_R0])
        entries = [entry for entry in line.entries if entry.image is not None]
        entries.append(
            ImageEntry(
                line.plan.line.line_id, KIND_UNINJECTED, RECIPE_R0, background, (), {}
            )
        )
        first = infer_entries(entries, low, counter)
        second = infer_entries(entries, operating, counter)
        if not conf_paths_equal(entries, first, second, line.zero, config):
            return False
    if progress is not None:
        progress.update(len(built))
    return True


# ------------------------------------------- precheck 10: CPU assertions


PLAN_COUNT_TABLES = (
    "hyperbola_positions.csv",
    "band_positions.csv",
    "band_arms.csv",
    "exclusions.csv",
)
"""Plan tables precheck 10 reads (the precheck plan has no ``lines.csv``)."""


@dataclass(frozen=True)
class CpuAssertionJobs:
    """Precheck 10 input: every line with its plan, and the planned counts.

    Attributes:
        jobs: ``(plan, source)`` per line; ``source`` is ``main`` (sealed
            plan) or ``precheck`` (the precheck folder's plan).
        expected: Planned checks per line and ``CPU_ASSERTIONS`` key.
        sources: Plan folder and table sha (LF) per source.
    """

    jobs: tuple[tuple[LinePlan, str], ...]
    expected: dict[str, dict[str, int]]
    sources: dict[str, dict]


@dataclass(frozen=True)
class LineCpuChecks:
    """Precheck 10 result of one line (``build_line(..., images=False)``)."""

    line_id: str
    plan_source: str
    n_checked: dict[str, int]
    n_failed: dict[str, int]
    failures: tuple[dict, ...]
    textures: dict[str, place.TextureCheck]
    seconds: float


def planned_check_counts(
    plan_dir: Path, line_ids: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Checks each CPU assertion must make per line, counted on the plan tables.

    ``shift_mismatch``: band arms; ``off_differs_from_on``: hyperbola groups
    x ``HYPERBOLA_SNR_DB`` x polarities; ``r_edge_differs_from_plan``:
    hyperbola positions; ``texture_check``: band positions with a long arm at
    a textured row (``band_conditions``: long arm, 47 or 68 px);
    ``zero_test``: one per line. Read from the csv rows, not from
    ``read_plan``, so the guard does not trust the reader it checks.
    """
    hyperbolas = inputs.read_csv(plan_dir / "hyperbola_positions.csv")
    arms = inputs.read_csv(plan_dir / "band_arms.csv")
    conditions = len(HYPERBOLA_SNR_DB) * len(POLARITIES)
    counts = {}
    for line_id in line_ids:
        line_hyperbolas = [row for row in hyperbolas if row["line_id"] == line_id]
        line_arms = [row for row in arms if row["line_id"] == line_id]
        textured_positions = {
            row["position_id"]
            for row in line_arms
            if row["length_arm"] == place.LONG_ARM
            and float(row["bin_target_px"]) in place.IN_RANGE_ROWS_PX
        }
        counts[line_id] = {
            "shift_mismatch": len(line_arms),
            "off_differs_from_on": len({r["group"] for r in line_hyperbolas})
            * conditions,
            "r_edge_differs_from_plan": len(line_hyperbolas),
            "texture_check": len(textured_positions),
            "zero_test": 1,
        }
    return counts


def _display_path(path: Path) -> str:
    """Repo-relative posix path when inside the repository, else absolute."""
    resolved = path.resolve()
    if resolved.is_relative_to(REPO.resolve()):
        return resolved.relative_to(REPO.resolve()).as_posix()
    return str(resolved)


def _plan_source(plan_dir: Path) -> dict:
    return {
        "plan_dir": _display_path(plan_dir),
        "files_sha256_lf": {
            name: inputs.sha256_file_lf(plan_dir / name) for name in PLAN_COUNT_TABLES
        },
    }


def cpu_assertion_jobs(
    settings: Settings,
    precheck_plans: Sequence[LinePlan],
    precheck_plan_dir: Path,
    candidates: inputs.CandidateApexes,
) -> CpuAssertionJobs:
    """The 153 background lines (sealed plan) and the two precheck lines.

    Raises:
        inputs.InputError: When a line id is in both sets (the counts are
            kept per line id).
    """
    main_lines = background_pool(settings)
    shared = {line.line_id for line in main_lines} & {
        plan.line.line_id for plan in precheck_plans
    }
    if shared:
        raise inputs.InputError(f"lines both background and precheck: {sorted(shared)}")
    main_plans = read_plan(SEALED_PLAN_DIR, main_lines, candidates)
    expected = planned_check_counts(
        SEALED_PLAN_DIR, [line.line_id for line in main_lines]
    )
    expected |= planned_check_counts(
        precheck_plan_dir, [plan.line.line_id for plan in precheck_plans]
    )
    jobs = [(plan, PLAN_SOURCE_MAIN) for plan in main_plans]
    jobs += [(plan, PLAN_SOURCE_PRECHECK) for plan in precheck_plans]
    sources = {
        PLAN_SOURCE_MAIN: _plan_source(SEALED_PLAN_DIR),
        PLAN_SOURCE_PRECHECK: _plan_source(precheck_plan_dir),
    }
    return CpuAssertionJobs(tuple(jobs), expected, sources)


def _cpu_check_worker(job: tuple[LinePlan, str]) -> LineCpuChecks:
    plan, source = job
    line_started = time.perf_counter()
    built = build_line(
        plan,
        _WORKER["hyperbola"],  # type: ignore[arg-type]
        _WORKER["band"],  # type: ignore[arg-type]
        images=False,
    )
    return LineCpuChecks(
        line_id=plan.line.line_id,
        plan_source=source,
        n_checked={key: int(built.n_checked[key]) for key in CPU_ASSERTIONS},
        n_failed={key: int(built.defects.get(key, 0)) for key in CPU_ASSERTIONS},
        failures=tuple(built.failed_items),
        textures=dict(built.texture_details),
        seconds=time.perf_counter() - line_started,
    )


def cpu_check_results(
    workers: int, jobs: Sequence[tuple[LinePlan, str]], progress: Progress
) -> list[LineCpuChecks]:
    """Precheck 10 over the lines (``workers`` processes), in job order."""
    found: list[LineCpuChecks] = []

    def finished(result: LineCpuChecks) -> None:
        found.append(result)
        progress.update(len(found))
        print(f"precheck 10: {len(found)}/{len(jobs)} lines", flush=True)

    progress.update(0)
    if workers == 1:
        _init_line_worker()
        for job in jobs:
            finished(_cpu_check_worker(job))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_line_worker
        ) as executor:
            futures = [executor.submit(_cpu_check_worker, job) for job in jobs]
            for future in as_completed(futures):
                finished(future.result())
    order = {plan.line.line_id: index for index, (plan, _) in enumerate(jobs)}
    return sorted(found, key=lambda result: order[result.line_id])


def cpu_assertion_guard(
    results: Sequence[LineCpuChecks], expected: Mapping[str, Mapping[str, int]]
) -> list[str]:
    """Count guard of precheck 10: problems (empty when the counts are right).

    Every line of the plan is checked, every line makes exactly its planned
    number of checks per assertion, and every assertion makes at least one
    check over all lines (an empty set is not a pass).
    """
    problems = []
    if sorted(r.line_id for r in results) != sorted(expected):
        problems.append("checked lines differ from the planned lines")
    for result in results:
        planned = expected.get(result.line_id, {})
        for key in CPU_ASSERTIONS:
            if result.n_checked.get(key) != planned.get(key):
                problems.append(
                    f"{key} {result.line_id}: checked {result.n_checked.get(key)}"
                    f" planned {planned.get(key)}"
                )
    for key in CPU_ASSERTIONS:
        if sum(result.n_checked.get(key, 0) for result in results) <= 0:
            problems.append(f"{key}: no check")
    return problems


def cpu_assertion_counts(
    results: Sequence[LineCpuChecks], expected: Mapping[str, Mapping[str, int]]
) -> dict[str, dict[str, int]]:
    """``n_checked``, ``n_expected`` and ``n_failed`` per assertion (all lines)."""
    counts = {}
    for key in CPU_ASSERTIONS:
        counts[key] = {
            "n_checked": sum(r.n_checked[key] for r in results),
            "n_expected": sum(planned.get(key, 0) for planned in expected.values()),
            "n_failed": sum(r.n_failed[key] for r in results),
        }
    return counts


def _distribution(values: Sequence[float]) -> dict:
    """Minimum, median, 90th percentile (``numpy`` linear) and maximum."""
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, RECORD_PERCENTILE)),
        "max": float(array.max()),
    }


def texture_positions(results: Sequence[LineCpuChecks]) -> list[dict]:
    """Every texture position's ``TextureCheck`` values (errors and records)."""
    rows = []
    for result in results:
        for position_id, check in sorted(result.textures.items()):
            rows.append(
                {"line_id": result.line_id, "position_id": position_id}
                | check.as_record()
            )
    return rows


def texture_distributions(rows: Sequence[Mapping]) -> dict[str, dict]:
    """:func:`_distribution` of every ``TextureCheck`` value over the positions."""
    if not rows:
        return {}
    names = [name for name in rows[0] if name not in ("line_id", "position_id")]
    return {name: _distribution([float(row[name]) for row in rows]) for name in names}


def failure_rows(results: Sequence[LineCpuChecks]) -> list[dict]:
    """Every failed check: line, assertion, item id and error value."""
    rows = []
    for result in results:
        for failure in result.failures:
            rows.append({"line_id": result.line_id, **failure})
    return rows


def _cpu_line_columns() -> tuple[str, ...]:
    columns = ["line_id", "plan_source"]
    for key in CPU_ASSERTIONS:
        columns += [f"{key}_checked", f"{key}_expected", f"{key}_failed"]
    return (*columns, "n_checks", "n_failed", "seconds")


def cpu_line_rows(
    results: Sequence[LineCpuChecks], expected: Mapping[str, Mapping[str, int]]
) -> list[dict]:
    """``cpu_assertions_lines.csv`` rows: counts and seconds per line."""
    rows = []
    for result in results:
        planned = expected.get(result.line_id, {})
        row: dict[str, object] = {
            "line_id": result.line_id,
            "plan_source": result.plan_source,
        }
        for key in CPU_ASSERTIONS:
            row[f"{key}_checked"] = result.n_checked[key]
            row[f"{key}_expected"] = planned.get(key)
            row[f"{key}_failed"] = result.n_failed[key]
        row["n_checks"] = sum(result.n_checked.values())
        row["n_failed"] = sum(result.n_failed.values())
        row["seconds"] = result.seconds
        rows.append(row)
    return rows


def run_cpu_assertions(
    settings: Settings, cpu_jobs: CpuAssertionJobs, identity: Mapping
) -> bool:
    """Precheck 10 (§5-2, v0.5): counts, guard, the two record files, one line.

    Writes ``cpu_assertions.json`` and ``cpu_assertions_lines.csv`` to the
    precheck folder; stdout gets the progress lines and one summary line
    with counts only. Returns ``True`` when no assertion failed and the
    count guard holds.
    """
    started = time.perf_counter()
    progress = Progress(
        settings.stage_dir / PROGRESS_FILE, "precheck_10", len(cpu_jobs.jobs)
    )
    results = cpu_check_results(settings.workers, cpu_jobs.jobs, progress)
    problems = cpu_assertion_guard(results, cpu_jobs.expected)
    counts = cpu_assertion_counts(results, cpu_jobs.expected)
    n_failed = sum(item["n_failed"] for item in counts.values())
    passed = not problems and n_failed == 0
    elapsed_s = time.perf_counter() - started
    positions = texture_positions(results)
    write_rows(
        settings.stage_dir / CPU_ASSERTIONS_LINES,
        _cpu_line_columns(),
        cpu_line_rows(results, cpu_jobs.expected),
    )
    write_json(
        settings.stage_dir / CPU_ASSERTIONS_JSON,
        {
            "schema": "m33_e3x_cpu_assertions/1",
            "status": STATUS_OK if passed else STATUS_PRECHECK_FAILED,
            "passed": passed,
            "run_identity": {**identity, "finished_at": common.local_now()},
            "plan_sources": cpu_jobs.sources,
            "workers": settings.workers,
            "n_lines": len(results),
            "assertions": counts,
            "guard_problems": problems,
            "failures": failure_rows(results),
            "texture_fields": {
                "asserted": list(place.TEXTURE_ASSERTED_FIELDS),
                "record_only": list(place.TEXTURE_RECORD_ONLY_FIELDS),
            },
            "texture_distribution": texture_distributions(positions),
            "texture_positions": positions,
            "elapsed_s": elapsed_s,
        },
    )
    summary = {
        "precheck": "10_cpu_assertions",
        "status": STATUS_OK if passed else STATUS_PRECHECK_FAILED,
        "n_lines": len(results),
        "n_checks": sum(item["n_checked"] for item in counts.values()),
        "n_failed": n_failed,
        "elapsed_s": round(elapsed_s, 1),
    }
    print(json.dumps(summary), flush=True)
    return passed


def cpu_assertion_file_hashes(folder: Path) -> dict[str, dict[str, str]]:
    """Raw and LF sha256 of the two precheck 10 record files."""
    return {
        name: {
            "raw": inputs.sha256_file(folder / name),
            "lf": inputs.sha256_file_lf(folder / name),
        }
        for name in (CPU_ASSERTIONS_JSON, CPU_ASSERTIONS_LINES)
    }


class _Tee(io.TextIOBase):
    """Writes to several text streams (check 10 output also reaches check 8)."""

    def __init__(self, *streams: io.TextIOBase) -> None:
        super().__init__()
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


# ------------------------------------------------------------ precheck run


def _close_precheck(
    settings: Settings, context: Context, identity: dict, report: Mapping
) -> str:
    """Measure the end tree and write ``precheck.json``; returns the status."""
    checks = report["checks"]
    status = STATUS_OK if all(checks.values()) else STATUS_PRECHECK_FAILED
    status, judgment = close_run(settings, context, identity, status)
    write_json(
        settings.stage_dir / "precheck.json",
        {
            "schema": SCHEMA,
            "stage": "precheck",
            "status": status,
            "judgment": judgment,
            **report,
            "run_identity": {**identity, "finished_at": common.local_now()},
        },
    )
    failed = [name for name, passed in checks.items() if not passed]
    print(f"precheck failed items: {failed}", flush=True)
    return status


def run_precheck(
    settings: Settings, started: float, context: Context
) -> tuple[str, Counts]:
    """§5-2 check 1, check 10, then checks 2-9 (v0.5; refuses an existing folder).

    Order: 1 -> precheck plan -> 10 -> 3 -> 2, 4-9. A failed check 1 stops
    right there (no precheck plan, no check 10, no check 2-9); a failed
    check 10 skips checks 3 and 2, 4-9 (no detector is loaded). Either way
    ``precheck.json`` lists the checks not run in ``not_run`` and the status
    is ``precheck_failed`` (exit 1).

    The folder holds ``plan/`` (the two precheck lines planned with the
    main-plan functions), ``cpu_assertions.json`` and
    ``cpu_assertions_lines.csv`` (check 10 on every pool line), ``w1/`` and
    ``w2/`` (records with one and two workers), ``progress.json`` and
    ``precheck.json``. Running again
    needs a new folder (``--precheck-dir runs/eval/m33/e3x_precheck_r<k>``):
    a folder that exists, even empty, is refused. Keep a copy of the folder
    made right before the main run (plan §5-2).
    """
    if settings.stage_dir.exists():
        raise inputs.InputError(f"output exists: {settings.stage_dir}")
    _, config = sealed_config(context)
    prepare_stage_dir(settings)
    checks: dict[str, bool] = {}
    checks["1_identity"], hashes = _precheck_identity(settings, context)
    identity = new_identity(settings, context, hashes)
    identity["started_at"] = _local_time(started)
    if not checks["1_identity"]:
        report = {
            "checks": checks,
            "not_run": ["10_cpu_assertions", *GPU_PRECHECK_CHECKS],
        }
        return _close_precheck(settings, context, identity, report), Counts()
    lines = precheck_pool(settings)
    plan_dir = settings.stage_dir / PLAN_DIR
    plan_info = _precheck_plan(settings, lines, plan_dir)
    candidates = inputs.candidate_apexes()
    plans = read_plan(plan_dir, lines, candidates)
    captured = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, captured)):
        cpu_jobs = cpu_assertion_jobs(settings, plans, plan_dir, candidates)
        checks["10_cpu_assertions"] = run_cpu_assertions(settings, cpu_jobs, identity)
    cpu_files = cpu_assertion_file_hashes(settings.stage_dir)
    if not checks["10_cpu_assertions"]:
        report = {
            "checks": checks,
            "not_run": list(GPU_PRECHECK_CHECKS),
            "cpu_assertions_sha256": cpu_files,
            "shift_summary": shift_summary(plan_info),
        }
        status = _close_precheck(settings, context, identity, report)
        return status, Counts(len(lines), 0, 0, 0)
    checks["3_shift"] = shift_check(plan_info)
    low = load_detectors(settings, CONF_AP)
    operating = load_detectors(settings, CONF_OPERATING)
    counter = attach_nms_counter()
    outputs: dict[int, dict[str, str]] = {}
    defects: dict[int, dict[str, int]] = {}
    built: list[LineRealizations] = []
    with contextlib.redirect_stdout(captured):
        for workers in (1, 2):
            run_settings = replace(
                settings,
                workers=workers,
                precheck_dir=settings.stage_dir / f"w{workers}",
            )
            prepare_stage_dir(run_settings)
            parts, found = evaluate_lines(
                run_settings,
                plans,
                low,
                counter,
                config,
                started,
                built if workers == 1 else None,
                Progress(
                    settings.stage_dir / PROGRESS_FILE,
                    f"precheck_w{workers}",
                    len(plans),
                ),
            )
            outputs[workers] = write_record_files(
                run_settings.stage_dir, parts, settings.folds
            )
            defects[workers] = dict(found)
        checks["6_conf_paths_equal"] = _precheck_conf_paths(
            built,
            (low, operating),
            counter,
            config,
            Progress(settings.stage_dir / PROGRESS_FILE, "precheck_6", len(built)),
        )
    checks["2_zero_test"] = all(line.zero_test for line in built)
    checks["4_off_on_equal"] = all(
        all(line.off_on_equal.values()) and all(line.r_edge_matches.values())
        for line in built
    )
    checks["5_textures"] = all(all(line.texture_checks.values()) for line in built)
    checks["7_workers_bytes_equal"] = outputs[1] == outputs[2]
    checks["8_stdout_blind"] = not METRIC_WORDS.search(captured.getvalue())
    checks["9_nms_and_maxdet"] = not any(
        found.get(key, 0) for found in defects.values() for key in NMS_DEFECT_KEYS
    )
    report = {
        "checks": checks,
        "not_run": [],
        "cpu_assertions_sha256": cpu_files,
        "defect_counts": {f"w{k}": v for k, v in defects.items()},
        "shift_summary": shift_summary(plan_info),
        "outputs_sha256": {f"w{k}": v for k, v in outputs.items()},
    }
    status = _close_precheck(settings, context, identity, report)
    n_images = sum(len(line.entries) for line in built)
    return status, Counts(len(lines), 0, n_images, n_images)


# ===================================================================== CLI


def parse_args(argv: Sequence[str] | None) -> Settings:
    """Command line -> ``Settings``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=[s.value for s in Stage], default=None)
    parser.add_argument("--precheck", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="0", help="'0' (GPU) or 'cpu'")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--precheck-dir", type=Path, default=DEFAULT_PRECHECK_DIR)
    parser.add_argument("--folds", default=",".join(map(str, FOLDS)))
    parser.add_argument("--only-lines", default="")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="plan only")
    parser.add_argument("--run-root", type=Path, default=common.RUN_ROOT)
    parser.add_argument("--r2-root", type=Path, default=m33_inputs.R2_OUT_ROOT)
    parser.add_argument("--r3-root", type=Path, default=R3_UNINJECTED_IDS)
    parser.add_argument("--pool", type=Path, default=pool.DEFAULT_OUT)
    parser.add_argument("--seal-record", type=Path, default=SEAL2_RECORD)
    args = parser.parse_args(argv)
    if args.precheck == (args.stage is not None):
        parser.error("give exactly one of --stage or --precheck")
    if args.overwrite and args.stage != Stage.PLAN.value:
        parser.error("--overwrite is for the plan stage only")
    try:
        folds = tuple(int(item) for item in args.folds.split(",") if item.strip())
    except ValueError:
        parser.error(f"--folds must be integers: {args.folds!r}")
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be 1-{MAX_WORKERS}")
    return Settings(
        stage=None if args.stage is None else Stage(args.stage),
        precheck=args.precheck,
        out_root=args.out_root,
        precheck_dir=args.precheck_dir,
        device=args.device,
        workers=args.workers,
        resume=args.resume,
        overwrite=args.overwrite,
        allow_dirty=args.allow_dirty,
        folds=folds,
        only_lines=tuple(i.strip() for i in args.only_lines.split(",") if i.strip()),
        run_root=args.run_root,
        r2_root=args.r2_root,
        r3_root=args.r3_root,
        pool_csv=args.pool,
        seal_record=args.seal_record,
    )


STAGE_RUNNERS = {
    Stage.PLAN: run_plan,
    Stage.UNINJECTED001: run_uninjected001,
    Stage.INJECTED: run_injected,
}


def run(settings: Settings, started: float) -> tuple[str, Counts]:
    """Start checks, then the stage (or the precheck)."""
    context = start_context(settings)
    if settings.precheck:
        return run_precheck(settings, started, context)
    assert settings.stage is not None
    return STAGE_RUNNERS[settings.stage](settings, started, context)


def print_final_line(
    status: str, settings: Settings | None, counts: Counts, started: float
) -> None:
    """Closing stdout JSON line; never carries metric values."""
    line = {
        "status": status,
        "out_dir": None if settings is None else str(settings.stage_dir),
        **asdict(counts),
        "elapsed_s": round(time.time() - started, 1),
    }
    print(
        json.dumps({key: line[key] for key in FINAL_KEYS}, ensure_ascii=False),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry; returns the exit code (module docstring table)."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    started = time.time()
    settings = parse_args(argv)
    counts = Counts()
    try:
        status, counts = run(settings, started)
    except inputs.InputError as error:
        logger.error("input error: %s", error)
        status = STATUS_INPUT_ERROR
    except Exception:
        logger.exception("internal error")
        status = STATUS_ERROR
    print_final_line(status, settings, counts, started)
    return EXIT_CODES[status]


if __name__ == "__main__":
    # m32_recipe_eval and the R3 module are imported by name; keep one copy.
    sys.modules.setdefault("m33_e3x", sys.modules[__name__])
    sys.exit(main())
