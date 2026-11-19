"""M3-3 R3 change report: zero and period changes under the 2026-11-20 rule.

Spec: ``docs/processing/m33_r3_zero_change_definition.md`` §5-2 (branch B,
user decision 2026-11-20), §6-2 and §6-4; definitions in
``m33_change_defs``. The sealed runner (``m32_recipe_eval.py``,
``m33_r3_band.py``) and ``src/`` are imported, never changed. No inference,
no PP, no GPU: the P4 value is not recomputed (it stays the stored one).

What a run does

1. Input checks (exit 2): the output folder is new and outside the R3
   folder; ``PERIOD_CHANGE_TOLERANCE_REL`` equals
   ``inject.PERIOD_CHECK_TOLERANCE``; the work-tree blobs of the C1 code
   equal ``C1_CODE_BLOBS`` (C1 seal §1) and ``inject.py`` is
   ``m33_inputs.C1_INJECT_BLOB``; every file under ``runs/eval/m33/r3/``
   matches the tracked hash lists (``m33_r3_result_hashes.csv`` for
   ``plan``/``injected``, ``docs/experiments/m33_r3_upstream_hashes.csv``
   for ``uninjected_ids``; no file missing, changed or new); the stored
   injected run's ``code_blobs`` equal ``C1_CODE_BLOBS``; the plan tables
   match the C1b record on LF sha (``m33_r3_band.read_c1b_inputs``, the
   runner's function); the stage lists, the plan report, the C1b record and
   the response library hash as the stored run recorded them.
2. Per line, in worker processes (also for ``--workers 1``, so the result
   does not depend on N): the calls of ``evaluate_band_line`` up to the
   flags. A null test first: the background cast to ``float32`` through the
   flag path must give exactly ``context.zero.k_zero`` and
   ``context.period_samples``. Then per realization: inject, re-measure the
   zero (``inject.linear_steps`` -> ``find_time_zero``), build the r0 image,
   ``zero_change``, ``period_change`` and the inferred trace range.
3. Reproduction checks (exit 1): the realization ids equal the stored ones
   (``injected/parts/*.json``) and, per realization, the old flags
   (``exact``) and ``(infer_i0, infer_i1)`` equal the stored values.
4. End checks (exit 1): the R3 file list and the code blobs are measured
   again and must equal the start.

Outputs (``--out-dir``, default ``runs/eval/m33/r3_change_report/``; an
existing folder is refused, there is no overwrite option)

* ``r3_tree_sha256_start.csv``, ``r3_tree_sha256_end.csv``: every file under
  the R3 folder (``root,path,sha256,bytes,mtime_utc``).
* ``realizations.csv``: one row per realization, sorted by line and
  realization id; every float is ``repr(float)``.
* ``lines.csv``: one row per line with its null test (both differences 0).
* ``line_seconds.csv``: seconds per line, kept apart so that the two tables
  above are byte-identical for any ``--workers``.
* ``summary.json``: the counts of spec §3-2 and §3-3 (all, per dataset, per
  ``ids_freq_stratum`` and for the P4 scope ``gz_ids``/``main_400``), the
  reproduction and null-test results and the run identity. Only
  ``identity`` and ``timings`` depend on the run.

Statuses and exit codes of a report run (stdout last line
``{"status", "out_dir", "n_lines", "n_realizations", "elapsed_s"}``)

==============  ====  =====================================================
status          exit  meaning
==============  ====  =====================================================
ok              0     full run, every check passed: this is the report
partial         1     ``--limit-lines`` or one background only, every check
                      passed on that subset (smoke; never the report)
empty           1     no realization was measured (an empty set is no pass)
not_reproduced  1     a null test was not exact, or an id, an old flag or
                      an inferred range differs from the stored run
tree_changed    1     an R3 file or a code blob changed during the run
input_error     2     an input check failed or the output folder exists;
                      nothing was written
error           3     internal error (git failed at the end included)
==============  ====  =====================================================

After any exit other than 2 the output folder holds evidence; run again
into a new folder (or remove it after reading it). The counts are report
values, not judgment inputs; stdout still carries no value (same format as
the runner).

``--write-result-hash-list PATH`` (sub-command) writes the hash list of
``<r3>/plan`` and ``<r3>/injected`` and exits: 0 written, 2 the file exists
or a folder is missing, 3 internal error; last line ``{"status", "out",
"n_files", "elapsed_s"}``. It made ``analysis/detection/m33_r3_result_hashes.csv``
(2026-11-20) before the report run.

Run (gpr-dev, CPU only, about 10-12 minutes with four workers):
    conda run -n gpr-dev python -I -B analysis/detection/m33_r3_change_report.py \
        --workers 4
"""

import argparse
import csv
import json
import logging
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common
import m32_pipe_responses as responses_io
import m32_recipe_eval as runner
import m32_recipe_inputs as inputs
import m33_change_defs as defs
import m33_inputs
import m33_r3_band as band
import m33_r3_p4 as p4

from gpr_detection.eval import inject
from gpr_detection.io.radargram import Radargram
from gpr_detection.preprocess.timezero import ZeroResult, find_time_zero
from gpr_detection.preprocess.to_image import DetectionImage, make_detection_image

logger = logging.getLogger("m33_r3_change_report")

REPO = inputs.REPO
SCHEMA = "m33_r3_change_report/1"
DEFAULT_OUT_DIR = REPO / "runs" / "eval" / "m33" / "r3_change_report"
RESULT_HASH_LIST = Path(__file__).resolve().parent / "m33_r3_result_hashes.csv"
"""sha256 of every file of ``r3/plan`` and ``r3/injected`` (2026-11-20)."""
UPSTREAM_HASH_LIST = REPO / "docs" / "experiments" / "m33_r3_upstream_hashes.csv"
"""C1b-time list; its ``r3/uninjected_ids`` rows cover the third R3 folder."""
RESULT_STAGES = ("plan", "injected")
UNINJECTED_STAGE = runner.BAND_UNINJECTED_DIR
HASH_COLUMNS = ("root", "path", "sha256", "bytes", "mtime_utc")
QUANTILES = (0.0, 0.5, 0.9, 0.99, 1.0)
P4_SCOPE = (inputs.DATASET_IDS, band.FREQ_MAIN)
"""``(dataset, ids_freq_stratum)`` of the P4 value ([RUN] §3-2)."""

C1_CODE_BLOBS = {
    "analysis/detection/m32_recipe_eval.py": "ded84a93b95003a5920bf41f73ad1d0436b9ef14",
    "analysis/detection/m31_d1_common.py": "0ba0675c8fc6bd632ed2b1e6794702b60f9b5d72",
    "analysis/detection/m32_aihub_perturb.py": (
        "320ce955a7acbcd8956cde7901234cf61e19ba45"
    ),
    "analysis/detection/m32_pipe_responses.py": (
        "5d065b3a807b41df04ff81098df86ba9ec747f58"
    ),
    "analysis/detection/m32_recipe_inputs.py": (
        "7f4815c5582086408a56a2002bf51d6002826a62"
    ),
    "analysis/detection/m32_recipe_match.py": (
        "7e17f060947073dbfcf4e3701a6e0505bffa9547"
    ),
    "analysis/detection/m33_inputs.py": "ababaa50bc0371f90c55b18b0f0e7192fc267a74",
    "analysis/detection/m33_r3_p4.py": "888f3dc9ac209400c71bbb29cf337cc4581f8f3b",
    "analysis/detection/m33_r3_band.py": "b499a456fe423b7d8a72fa516a0f2123ae3bd7f4",
    "analysis/detection/m33_r2_eval.py": "06d95b592549d5eb27fcd15347906fa0d74a4530",
    "src/gpr_detection/preprocess/to_image.py": (
        "233e47ef22f825dc24449b7e5567fe8a47719d9a"
    ),
    "src/gpr_detection/eval/inject.py": "2d88ff0a25c5cbc1bb7ca8698ba422c633f5b531",
    "src/gpr_detection/pipeline.py": "dd5e7bcba4397efd732e7bf3ca1c5c92fb2404a1",
    "src/gpr_detection/detect/yolo.py": "505ff04d228c9e41272144fc263b7912b7568065",
}
"""Copy of ``docs/experiments/m33_c1_seal.md`` §1 for the paths the R3 runner
records in ``run_identity.code_blobs`` (a test checks the copy)."""
REPORT_CODE_PATHS = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parent / "m33_change_defs.py",
)
"""This tool and its definitions: blobs recorded and checked start = end."""

RUN_INPUT_KEYS_RAW = ("plan_report", "c1b_record", "responses")
"""Hashes the stored run kept in ``run_identity.inputs_sha256``."""
RUN_INPUT_KEYS_LF = ("injection_table", "positions", "exclusions")
"""Hashes the stored run kept in ``run_identity.inputs_sha256_lf``."""

STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_EMPTY = "empty"
STATUS_NOT_REPRODUCED = "not_reproduced"
STATUS_TREE_CHANGED = "tree_changed"
STATUS_INPUT_ERROR = m33_inputs.STATUS_INPUT_ERROR
STATUS_ERROR = "error"
EXIT_CODES = {
    STATUS_OK: 0,
    STATUS_PARTIAL: 1,
    STATUS_EMPTY: 1,
    STATUS_NOT_REPRODUCED: 1,
    STATUS_TREE_CHANGED: 1,
    STATUS_INPUT_ERROR: 2,
    STATUS_ERROR: 3,
}

REALIZATION_COLUMNS = (
    "line_id",
    "dataset",
    "ids_freq_stratum",
    "realization_id",
    "snr_db",
    "polarity",
    "dt_ns",
    "period_bg_samples",
    "period_inj_samples",
    "k_zero_bg",
    "k_zero_inj",
    "zero_rule_bg",
    "zero_rule_inj",
    "zero_label_bg",
    "zero_label_inj",
    "n_events_bg",
    "n_events_inj",
    "zero_dk_samples",
    "zero_dk_ns",
    "zero_dk_over_p",
    "zero_label_changed",
    "zero_failed_changed",
    "zero_rule_changed",
    "zero_events_changed",
    "zero_exceeds_tolerance",
    "zero_changed",
    "zero_changed_exact",
    "period_dp_samples",
    "period_dp_rel",
    "period_exceeds_tolerance",
    "period_failed",
    "period_changed",
    "period_changed_exact",
    "infer_i0",
    "infer_i1",
    "stored_zero_changed",
    "stored_period_changed",
    "stored_infer_i0",
    "stored_infer_i1",
)
LINE_COLUMNS = (
    "line_id",
    "dataset",
    "ids_freq_stratum",
    "n_traces",
    "n_samples",
    "dt_ns",
    "period_bg_samples",
    "k_zero_bg",
    "zero_rule_bg",
    "zero_label_bg",
    "n_events_bg",
    "null_k_zero",
    "null_dk_samples",
    "null_zero_equal",
    "null_period_samples",
    "null_dp_samples",
    "null_period_equal",
    "line_zero_dk_samples",
    "n_realizations",
)
"""``line_zero_dk_samples``: ``runner.load_line`` zero vs the context zero
(report only; both come from the same steps, so it should be 0)."""
LINE_SECONDS_COLUMNS = ("line_id", "seconds")


# ================================================================ settings


@dataclass(frozen=True)
class ReportSettings:
    """Where the report reads and writes, and how many processes it uses."""

    r3_root: Path = m33_inputs.R3_OUT_ROOT
    out_dir: Path = DEFAULT_OUT_DIR
    c1b_record: Path = m33_inputs.R3_C1B_RECORD
    result_hash_list: Path = RESULT_HASH_LIST
    upstream_hash_list: Path = UPSTREAM_HASH_LIST
    workers: int = 1
    limit_lines: int | None = None
    backgrounds: tuple[str, ...] = runner.BAND_BACKGROUNDS

    @property
    def is_full(self) -> bool:
        """Every line of both backgrounds: the only run that is the report."""
        return self.limit_lines is None and sorted(self.backgrounds) == sorted(
            runner.BAND_BACKGROUNDS
        )

    def band_settings(self) -> runner.RunSettings:
        """Runner settings of the injected stage this report re-measures."""
        return runner.parse_args(
            [
                "--injectant",
                runner.INJECTANT_BAND,
                "--stage",
                "injected",
                "--out-root",
                str(self.r3_root),
                "--c1b-record",
                str(self.c1b_record),
                "--backgrounds",
                ",".join(self.backgrounds),
            ]
        )


LineLoader = Callable[[inputs.LineRef], runner.LineData]


@dataclass(frozen=True)
class LineJob:
    """One line for a worker; everything that depends on module state is here.

    ``loader`` is a module-level function (picklable); the report passes
    ``runner.load_line``.
    """

    ref: inputs.LineRef
    realizations: Mapping[str, Sequence[inject.InjectionSpec]]
    exclusions: inject.LineExclusions
    freq: str
    loader: LineLoader


@dataclass(frozen=True)
class LineReport:
    """What a worker returns for one line."""

    line_row: dict
    realization_rows: list[dict]
    seconds: float


@dataclass(frozen=True)
class ReportInputs:
    """Checked inputs of a report run."""

    lines: list[inputs.LineRef]
    realizations_by_line: dict[str, dict[str, list[inject.InjectionSpec]]]
    exclusions: dict[str, inject.LineExclusions]
    strata: dict[str, str]
    library: responses_io.ResponseLibrary
    stored: dict[str, Mapping]
    injected_summary: dict
    sha256: dict[str, str]


@dataclass
class ReportCounts:
    """What the closing stdout line reports (no values)."""

    n_lines: int = 0
    n_realizations: int = 0


# ============================================================ small helpers


def _display_path(path: Path) -> str:
    """Repo-relative posix path, or the absolute posix path outside the repo."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _mtime_utc(path: Path) -> str:
    moment = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _cell(value: object) -> str:
    """CSV text of a value: full-precision floats, ``True``/``False``, "" for None."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float | np.floating):
        return repr(float(value))
    return str(value)


def _as_bool(value: object) -> bool:
    """A stored flag: JSON ``true`` or the csv text ``"True"``."""
    return value is True or str(value) == "True"


def _difference(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return value - reference


def _write_rows(path: Path, columns: Sequence[str], rows: Iterable[Mapping]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _cell(row.get(column)) for column in columns})


# ================================================================ file hashes


def hash_rows(root: Path) -> list[dict[str, str]]:
    """One row per file under ``root``, sorted by path (``HASH_COLUMNS``)."""
    root_text = _display_path(root)
    rows = [
        {
            "root": root_text,
            "path": _display_path(path),
            "sha256": inputs.sha256_file(path),
            "bytes": str(path.stat().st_size),
            "mtime_utc": _mtime_utc(path),
        }
        for path in root.rglob("*")
        if path.is_file()
    ]
    return sorted(rows, key=lambda row: row["path"])


def write_hash_rows(path: Path, rows: Iterable[Mapping]) -> None:
    """Hash rows as an LF csv (the format of ``m33_r3_upstream_hashes.csv``)."""
    _write_rows(path, HASH_COLUMNS, rows)


def read_hash_rows(path: Path) -> list[dict[str, str]]:
    """Rows of a hash list.

    Raises:
        inputs.InputError: For a missing file or other columns.
    """
    if not path.is_file():
        raise inputs.InputError(f"hash list missing: {path}")
    rows = inputs.read_csv(path)
    if rows and tuple(rows[0]) != HASH_COLUMNS:
        raise inputs.InputError(f"hash list {path} has columns {tuple(rows[0])}")
    return rows


def _fingerprints(rows: Iterable[Mapping]) -> dict[str, tuple[str, str]]:
    return {row["path"]: (row["sha256"], row["bytes"]) for row in rows}


def tree_differences(
    expected: Mapping[str, tuple[str, str]], measured: Mapping[str, tuple[str, str]]
) -> dict[str, list[str]]:
    """Paths whose sha or size differ, that are missing, or that are new."""
    return {
        "changed": sorted(
            path
            for path in expected.keys() & measured.keys()
            if expected[path] != measured[path]
        ),
        "missing": sorted(expected.keys() - measured.keys()),
        "new": sorted(measured.keys() - expected.keys()),
    }


def expected_r3_tree(settings: ReportSettings) -> dict[str, tuple[str, str]]:
    """Fingerprints of every R3 file from the two tracked hash lists.

    Raises:
        inputs.InputError: For a missing list, a result list that does not
            name exactly ``plan`` and ``injected``, or no ``uninjected_ids`` row.
    """
    result_rows = read_hash_rows(settings.result_hash_list)
    result_roots = {_display_path(settings.r3_root / stage) for stage in RESULT_STAGES}
    listed_roots = {row["root"] for row in result_rows}
    if listed_roots != result_roots:
        raise inputs.InputError(
            f"result hash list roots {sorted(listed_roots)} != {sorted(result_roots)}"
        )
    upstream_root = _display_path(settings.r3_root / UNINJECTED_STAGE)
    upstream_rows = [
        row
        for row in read_hash_rows(settings.upstream_hash_list)
        if row["root"] == upstream_root
    ]
    if not upstream_rows:
        raise inputs.InputError(f"no {upstream_root} rows in the upstream hash list")
    return _fingerprints(result_rows + upstream_rows)


def check_r3_tree(settings: ReportSettings, measured_rows: Sequence[Mapping]) -> None:
    """The R3 files must be exactly those of the tracked hash lists.

    Raises:
        inputs.InputError: For a changed, missing or new file (counts in the
            message).
    """
    differences = tree_differences(
        expected_r3_tree(settings), _fingerprints(measured_rows)
    )
    if any(differences.values()):
        counts = {kind: len(paths) for kind, paths in differences.items()}
        first = next(paths[0] for paths in differences.values() if paths)
        raise inputs.InputError(
            f"R3 files differ from the hash lists {counts}: {first}"
        )


def write_result_hash_list(r3_root: Path, path: Path) -> int:
    """Write the hash list of ``<r3_root>/plan`` and ``/injected``; returns rows.

    Raises:
        inputs.InputError: If ``path`` exists or a folder is missing.
    """
    if path.exists():
        raise inputs.InputError(f"hash list exists (never overwritten): {path}")
    rows: list[dict[str, str]] = []
    for stage in RESULT_STAGES:
        stage_dir = r3_root / stage
        if not stage_dir.is_dir():
            raise inputs.InputError(f"R3 folder missing: {stage_dir}")
        rows += hash_rows(stage_dir)
    write_hash_rows(path, rows)
    return len(rows)


# ================================================================ code blobs


def measure_code_blobs() -> dict[str, str | None]:
    """Work-tree blobs of the C1 code and of this tool (``None``: git failed)."""
    paths = [REPO / name for name in C1_CODE_BLOBS] + list(REPORT_CODE_PATHS)
    return m33_inputs.git_blobs(paths)


def check_c1_code_blobs(blobs: Mapping[str, str | None], where: str) -> None:
    """The C1 paths of ``blobs`` must hold the C1 seal values.

    Raises:
        inputs.InputError: For an unmeasured or another blob.
    """
    for path, sealed in C1_CODE_BLOBS.items():
        measured = blobs.get(path)
        if measured != sealed:
            raise inputs.InputError(f"{where}: {path} blob {measured} != C1 {sealed}")


def check_tolerance_link() -> None:
    """The period tolerance must stay the project's "same period" value.

    Raises:
        inputs.InputError: If the two constants differ.
    """
    if defs.PERIOD_CHANGE_TOLERANCE_REL != inject.PERIOD_CHECK_TOLERANCE:
        raise inputs.InputError(
            f"PERIOD_CHANGE_TOLERANCE_REL {defs.PERIOD_CHANGE_TOLERANCE_REL} != "
            f"inject.PERIOD_CHECK_TOLERANCE {inject.PERIOD_CHECK_TOLERANCE}"
        )


def check_arguments(settings: ReportSettings) -> None:
    """Worker count, line limit and backgrounds the runner accepts.

    Raises:
        inputs.InputError: For a value ``runner.parse_args`` would refuse or
            a count below 1.
    """
    if settings.workers < 1:
        raise inputs.InputError(f"--workers must be >= 1, got {settings.workers}")
    if settings.limit_lines is not None and settings.limit_lines < 1:
        raise inputs.InputError(f"--limit-lines must be >= 1: {settings.limit_lines}")
    unknown = set(settings.backgrounds) - set(runner.BAND_BACKGROUNDS)
    if unknown or not settings.backgrounds:
        raise inputs.InputError(
            f"--backgrounds must name {list(runner.BAND_BACKGROUNDS)}, "
            f"got {list(settings.backgrounds)}"
        )


def check_output_dir(settings: ReportSettings) -> None:
    """A new folder outside the R3 folder.

    Raises:
        inputs.InputError: If it exists or lies inside ``r3_root``.
    """
    if settings.out_dir.exists():
        raise inputs.InputError(
            f"output exists (never overwritten): {settings.out_dir}"
        )
    if settings.out_dir.resolve().is_relative_to(settings.r3_root.resolve()):
        raise inputs.InputError(f"output inside the R3 folder: {settings.out_dir}")


# =============================================================== inputs


def stored_realizations(parts_dir: Path) -> dict[str, Mapping]:
    """``realization_id -> stored row`` from the injected line parts."""
    stored: dict[str, Mapping] = {}
    for path in sorted(parts_dir.glob("*.json")):
        part = json.loads(path.read_text(encoding="utf-8"))
        for row in part["realizations"]:
            stored[row["realization_id"]] = row
    return stored


def _check_same_inputs_as_run(sha256: Mapping[str, str], identity: Mapping) -> None:
    """The plan, record and responses must hash as the stored run recorded.

    Raises:
        inputs.InputError: For a missing or different hash.
    """
    recorded_raw = identity.get("inputs_sha256") or {}
    recorded_lf = identity.get("inputs_sha256_lf") or {}
    checks = [(key, recorded_raw.get(key)) for key in RUN_INPUT_KEYS_RAW]
    checks += [(key, recorded_lf.get(key)) for key in RUN_INPUT_KEYS_LF]
    for key, recorded in checks:
        if sha256.get(key) != recorded:
            raise inputs.InputError(
                f"{key} sha256 {sha256.get(key)} != stored run {recorded}"
            )


def _read_injected_summary(r3_root: Path) -> dict:
    path = r3_root / "injected" / "summary.json"
    if not path.is_file():
        raise inputs.InputError(f"injected summary missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_report_inputs(settings: ReportSettings) -> ReportInputs:
    """Check and read everything the line jobs and the reproduction need.

    Raises:
        inputs.InputError: For any input that is not the one of the stored
            injected run (see the module docstring, step 1).
    """
    band_settings = settings.band_settings()
    injected_summary = _read_injected_summary(settings.r3_root)
    identity = injected_summary.get("run_identity") or {}
    check_c1_code_blobs(identity.get("code_blobs") or {}, "stored injected run")
    strata = band.ids_freq_strata()
    p4.assert_ids_freq_strata(strata)
    sha256 = dict(inputs.assert_list_hashes())
    c1b = band.read_c1b_inputs(band_settings)
    sha256.update(c1b.sha256)
    library = responses_io.load_library()
    sha256["responses"] = responses_io.library_sha256(library)
    _check_same_inputs_as_run(sha256, identity)
    summary_path = settings.r3_root / "injected" / "summary.json"
    sha256["injected_summary"] = inputs.sha256_file(summary_path)
    sha256["result_hash_list_lf"] = inputs.sha256_file_lf(settings.result_hash_list)
    sha256["upstream_hash_list_lf"] = inputs.sha256_file_lf(settings.upstream_hash_list)

    by_line = runner.group_specs(c1b.specs)
    refs = {
        ref.line_id: ref for ref in band.band_background_lines(band_settings, strata)
    }
    lines = [refs[line_id] for line_id in by_line if line_id in refs]
    if settings.limit_lines is not None:
        lines = lines[: settings.limit_lines]
    candidates = inputs.candidate_apexes()
    exclusions = {
        ref.line_id: runner.line_exclusions(ref.line_id, c1b.exclusions, candidates)
        for ref in lines
    }
    return ReportInputs(
        lines=lines,
        realizations_by_line=by_line,
        exclusions=exclusions,
        strata=strata,
        library=library,
        stored=stored_realizations(settings.r3_root / "injected" / "parts"),
        injected_summary=injected_summary,
        sha256=sha256,
    )


def line_jobs(report_inputs: ReportInputs, loader: LineLoader) -> list[LineJob]:
    """One job per chosen line, in table order."""
    return [
        LineJob(
            ref=ref,
            realizations=report_inputs.realizations_by_line[ref.line_id],
            exclusions=report_inputs.exclusions[ref.line_id],
            freq=report_inputs.strata.get(ref.line_id, ""),
            loader=loader,
        )
        for ref in report_inputs.lines
    ]


# ============================================================ line measure


def _r0_image(radargram: Radargram) -> DetectionImage:
    return make_detection_image(radargram, inputs.recipe_config(inputs.R0))


def _zero_on_flag_path(radargram: Radargram) -> ZeroResult:
    """The zero as ``m33_r3_band.zero_changed`` measures it."""
    return find_time_zero(radargram, inject.linear_steps(radargram.data, radargram))


def null_test_row(
    job: LineJob, line: runner.LineData, context: inject.LineContext
) -> dict:
    """Line row: the background through the flag path must equal the context."""
    radargram = line.radargram
    background32 = dataclass_replace(
        radargram, data=np.asarray(radargram.data).astype(np.float32)
    )
    zero32 = _zero_on_flag_path(background32)
    period32 = _r0_image(background32).period_samples
    zero = context.zero
    return {
        "line_id": job.ref.line_id,
        "dataset": job.ref.dataset,
        "ids_freq_stratum": job.freq,
        "n_traces": radargram.n_traces,
        "n_samples": radargram.n_samples,
        "dt_ns": radargram.dt_ns,
        "period_bg_samples": context.period_samples,
        "k_zero_bg": zero.k_zero,
        "zero_rule_bg": zero.zero_rule,
        "zero_label_bg": zero.zero_event_label,
        "n_events_bg": len(zero.events_ns),
        "null_k_zero": zero32.k_zero,
        "null_dk_samples": _difference(zero32.k_zero, zero.k_zero),
        "null_zero_equal": bool(zero32.k_zero == zero.k_zero),
        "null_period_samples": period32,
        "null_dp_samples": _difference(period32, context.period_samples),
        "null_period_equal": bool(period32 == context.period_samples),
        "line_zero_dk_samples": _difference(line.zero.k_zero, zero.k_zero),
    }


def realization_row(
    job: LineJob, context: inject.LineContext, result: inject.BandInjectionResult
) -> dict:
    """One realization: both definitions of both changes and its trace range.

    The injected record is re-measured as ``evaluate_band_line`` does it: a
    ``float32`` copy of the line, the zero through the flag path and the r0
    image (its ``period_samples`` and inferred tiles).
    """
    injected = dataclass_replace(
        context.background, data=result.data.astype(np.float32)
    )
    injected_zero = _zero_on_flag_path(injected)
    image = _r0_image(injected)
    spec = job.realizations[result.realization_id][0]
    zero = defs.zero_change(context.zero, injected_zero, context.period_samples)
    period = defs.period_change(context.period_samples, image.period_samples)
    first, last = band.tile_trace_range(image, band.inference_tiles(image))
    dt_ns = context.background.dt_ns
    return {
        "line_id": job.ref.line_id,
        "dataset": job.ref.dataset,
        "ids_freq_stratum": job.freq,
        "realization_id": result.realization_id,
        "snr_db": spec.snr_db,
        "polarity": spec.polarity,
        "dt_ns": dt_ns,
        "period_bg_samples": context.period_samples,
        "period_inj_samples": image.period_samples,
        "k_zero_bg": context.zero.k_zero,
        "k_zero_inj": injected_zero.k_zero,
        "zero_rule_bg": context.zero.zero_rule,
        "zero_rule_inj": injected_zero.zero_rule,
        "zero_label_bg": context.zero.zero_event_label,
        "zero_label_inj": injected_zero.zero_event_label,
        "n_events_bg": len(context.zero.events_ns),
        "n_events_inj": len(injected_zero.events_ns),
        "zero_dk_samples": zero.dk_samples,
        "zero_dk_ns": None if zero.dk_samples is None else zero.dk_samples * dt_ns,
        "zero_dk_over_p": zero.dk_over_p,
        "zero_label_changed": zero.label_changed,
        "zero_failed_changed": zero.failed_changed,
        "zero_rule_changed": zero.rule_changed,
        "zero_events_changed": zero.events_changed,
        "zero_exceeds_tolerance": zero.exceeds_tolerance,
        "zero_changed": zero.changed,
        "zero_changed_exact": zero.exact,
        "period_dp_samples": period.dp_samples,
        "period_dp_rel": period.dp_rel,
        "period_exceeds_tolerance": period.exceeds_tolerance,
        "period_failed": period.failed,
        "period_changed": period.changed,
        "period_changed_exact": period.exact,
        "infer_i0": first,
        "infer_i1": last,
    }


def evaluate_line(job: LineJob, library: responses_io.ResponseLibrary) -> LineReport:
    """Null test and every realization of one line (runner call order)."""
    started = time.perf_counter()
    line = job.loader(job.ref)
    radargram = line.radargram
    context = inject.prepare_line_context(radargram, job.exclusions)
    responses = library.for_line(job.ref.dataset, runner.line_time_window_ns(line))
    rows = []
    for specs in job.realizations.values():
        result = inject.inject_band_realization(context, specs, responses)
        if result.truths:  # the runner skips a realization without truths too
            rows.append(realization_row(job, context, result))
    line_row = null_test_row(job, line, context) | {"n_realizations": len(rows)}
    return LineReport(line_row, rows, time.perf_counter() - started)


_WORKER_LIBRARY: list[responses_io.ResponseLibrary] = []


def _init_worker(library: responses_io.ResponseLibrary) -> None:
    _WORKER_LIBRARY.clear()
    _WORKER_LIBRARY.append(library)


def _run_job(job: LineJob) -> LineReport:
    return evaluate_line(job, _WORKER_LIBRARY[0])


def run_line_jobs(
    jobs: Sequence[LineJob], library: responses_io.ResponseLibrary, workers: int
) -> list[LineReport]:
    """Every job in ``workers`` processes (``map`` keeps the job order)."""
    reports = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(library,)
    ) as executor:
        for number, report in enumerate(executor.map(_run_job, jobs), start=1):
            reports.append(report)
            print(
                f"line {number}/{len(jobs)} {report.line_row['line_id']}: "
                f"{len(report.realization_rows)} realizations, "
                f"{report.seconds:.1f} s",
                flush=True,
            )
    return reports


# =========================================================== reproduction


def attach_stored(rows: Iterable[dict], stored: Mapping[str, Mapping]) -> None:
    """Add the stored flags and trace range to each measured row."""
    for row in rows:
        stored_row = stored.get(row["realization_id"])
        if stored_row is None:
            continue
        row["stored_zero_changed"] = _as_bool(stored_row["zero_changed"])
        row["stored_period_changed"] = _as_bool(stored_row["period_changed"])
        row["stored_infer_i0"] = int(stored_row["infer_i0"])
        row["stored_infer_i1"] = int(stored_row["infer_i1"])


def _mismatched(rows: Sequence[Mapping], measured: str, stored: str) -> list[str]:
    return sorted(
        row["realization_id"]
        for row in rows
        if stored in row and row[measured] != row[stored]
    )


def reproduction_check(rows: Sequence[Mapping], expected_ids: set[str]) -> dict:
    """Ids, old flags and inferred ranges against the stored run."""
    measured_ids = {row["realization_id"] for row in rows}
    zero_ids = _mismatched(rows, "zero_changed_exact", "stored_zero_changed")
    period_ids = _mismatched(rows, "period_changed_exact", "stored_period_changed")
    range_ids = sorted(
        set(_mismatched(rows, "infer_i0", "stored_infer_i0"))
        | set(_mismatched(rows, "infer_i1", "stored_infer_i1"))
    )
    missing = sorted(expected_ids - measured_ids)
    extra = sorted(measured_ids - expected_ids)
    return {
        "n_expected": len(expected_ids),
        "n_measured": len(measured_ids),
        "missing_ids": missing,
        "extra_ids": extra,
        "zero_exact_mismatch_ids": zero_ids,
        "period_exact_mismatch_ids": period_ids,
        "infer_range_mismatch_ids": range_ids,
        "reproduced": not (missing or extra or zero_ids or period_ids or range_ids),
    }


def expected_realization_ids(
    settings: ReportSettings, report_inputs: ReportInputs
) -> set[str]:
    """Stored ids a run must reproduce: all, or those of the chosen lines."""
    if settings.is_full:
        return set(report_inputs.stored)
    chosen = {
        realization_id
        for ref in report_inputs.lines
        for realization_id in report_inputs.realizations_by_line[ref.line_id]
    }
    return set(report_inputs.stored) & chosen


def null_test_summary(line_rows: Sequence[Mapping]) -> dict:
    """Lines whose flag-path background differs from the context (must be 0)."""
    zero_ids = sorted(r["line_id"] for r in line_rows if not r["null_zero_equal"])
    period_ids = sorted(r["line_id"] for r in line_rows if not r["null_period_equal"])
    line_zero_ids = sorted(
        r["line_id"] for r in line_rows if r["line_zero_dk_samples"] not in (0.0, None)
    )
    return {
        "n_lines": len(line_rows),
        "zero_not_equal_lines": zero_ids,
        "period_not_equal_lines": period_ids,
        "line_zero_differs_lines": line_zero_ids,
        "passed": not (zero_ids or period_ids),
    }


# ================================================================ counts


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {f"q{q:g}": float(np.quantile(values, q)) for q in QUANTILES}


def _count(rows: Iterable[Mapping], key: str) -> int:
    return sum(bool(row[key]) for row in rows)


def _absolute(rows: Iterable[Mapping], key: str) -> list[float]:
    return [abs(row[key]) for row in rows if row[key] is not None]


def change_counts(rows: Sequence[Mapping]) -> dict:
    """The §3-2 and §3-3 report keys over a set of realization rows."""
    zero_dt_ns = _absolute(rows, "zero_dk_ns")
    return {
        "n_realizations": len(rows),
        "n_zero_changed": _count(rows, "zero_changed"),
        "n_zero_changed_exact": _count(rows, "zero_changed_exact"),
        "n_zero_label_changed": sum(
            row["zero_label_changed"] or row["zero_failed_changed"] for row in rows
        ),
        "n_zero_rule_changed": _count(rows, "zero_rule_changed"),
        "n_zero_events_changed": _count(rows, "zero_events_changed"),
        "zero_abs_dk_over_p_quantiles": _quantiles(_absolute(rows, "zero_dk_over_p")),
        "zero_abs_dt_ns_max": max(zero_dt_ns) if zero_dt_ns else None,
        "n_period_changed": _count(rows, "period_changed"),
        "n_period_changed_exact": _count(rows, "period_changed_exact"),
        "n_period_failed": _count(rows, "period_failed"),
        "period_abs_dp_rel_quantiles": _quantiles(_absolute(rows, "period_dp_rel")),
    }


def _counts_by(rows: Sequence[Mapping], key: str) -> dict[str, dict]:
    groups: dict[str, list[Mapping]] = {}
    for row in rows:
        if row[key]:
            groups.setdefault(row[key], []).append(row)
    return {name: change_counts(groups[name]) for name in sorted(groups)}


def grouped_counts(rows: Sequence[Mapping]) -> dict:
    """Counts over all rows, per dataset, per IDS stratum and the P4 scope."""
    p4_rows = [
        row for row in rows if (row["dataset"], row["ids_freq_stratum"]) == P4_SCOPE
    ]
    return {
        "all": change_counts(rows),
        "by_dataset": _counts_by(rows, "dataset"),
        "by_ids_freq_stratum": _counts_by(rows, "ids_freq_stratum"),
        "p4_scope_gz_ids_main_400": change_counts(p4_rows),
    }


# ================================================================ the run


def decide_status(
    settings: ReportSettings,
    n_rows: int,
    reproduction: Mapping,
    null_test: Mapping,
    end_checks: Mapping,
) -> str:
    """The status of a finished run (module docstring table, first match)."""
    if end_checks["code_blob_unmeasured"]:
        return STATUS_ERROR
    if end_checks["tree_changes"] or end_checks["code_blob_changes"]:
        return STATUS_TREE_CHANGED
    if not (reproduction["reproduced"] and null_test["passed"]):
        return STATUS_NOT_REPRODUCED
    if n_rows == 0:
        return STATUS_EMPTY
    if not settings.is_full:
        return STATUS_PARTIAL
    return STATUS_OK


def end_check(
    start_rows: Sequence[Mapping],
    end_rows: Sequence[Mapping],
    start_blobs: Mapping[str, str | None],
    end_blobs: Mapping[str, str | None],
) -> dict:
    """R3 files and code blobs at the end against the start.

    A ``None`` end blob is a failed git call, kept apart from a change.
    """
    differences = tree_differences(_fingerprints(start_rows), _fingerprints(end_rows))
    unmeasured = sorted(path for path, blob in end_blobs.items() if blob is None)
    changed = sorted(
        path
        for path, blob in end_blobs.items()
        if blob is not None and blob != start_blobs.get(path)
    )
    return {
        "tree_changes": {kind: paths for kind, paths in differences.items() if paths},
        "code_blob_changes": changed,
        "code_blob_unmeasured": unmeasured,
    }


def _injected_run_identity(summary: Mapping) -> dict:
    identity = summary.get("run_identity") or {}
    return {
        "run_id": identity.get("run_id"),
        "git_head": (identity.get("git") or {}).get("head"),
        "src_tree_id": identity.get("src_tree_id"),
        "status": summary.get("status"),
    }


def report_identity(
    settings: ReportSettings,
    report_inputs: ReportInputs,
    started_at: str,
    blobs: tuple[Mapping, Mapping],
) -> dict:
    """Who ran what on which inputs (the only run-dependent block)."""
    return {
        "started_at": started_at,
        "finished_at": common.local_now(),
        "argv": list(sys.argv),
        "git": common.git_state(),
        "versions": common.library_versions(),
        "workers": settings.workers,
        "limit_lines": settings.limit_lines,
        "backgrounds": list(settings.backgrounds),
        "r3_root": _display_path(settings.r3_root),
        "out_dir": _display_path(settings.out_dir),
        "code_blobs_start": dict(blobs[0]),
        "code_blobs_end": dict(blobs[1]),
        "inputs_sha256": dict(sorted(report_inputs.sha256.items())),
        "injected_run": _injected_run_identity(report_inputs.injected_summary),
    }


def _write_outputs(
    settings: ReportSettings,
    reports: Sequence[LineReport],
    rows: Sequence[Mapping],
    summary: Mapping,
) -> None:
    out = settings.out_dir
    line_rows = sorted((r.line_row for r in reports), key=lambda r: r["line_id"])
    seconds = sorted(
        ({"line_id": r.line_row["line_id"], "seconds": r.seconds} for r in reports),
        key=lambda r: r["line_id"],
    )
    _write_rows(out / "realizations.csv", REALIZATION_COLUMNS, rows)
    _write_rows(out / "lines.csv", LINE_COLUMNS, line_rows)
    _write_rows(out / "line_seconds.csv", LINE_SECONDS_COLUMNS, seconds)
    common.write_json(out / "summary.json", dict(summary))


@dataclass(frozen=True)
class StartState:
    """What the input checks measured before anything is written."""

    code_blobs: dict[str, str | None]
    r3_rows: list[dict[str, str]]
    report_inputs: ReportInputs


def check_start(settings: ReportSettings) -> StartState:
    """Every input check of the module docstring, step 1.

    Raises:
        inputs.InputError: For the first check that fails (exit 2).
    """
    check_output_dir(settings)
    check_arguments(settings)
    check_tolerance_link()
    code_blobs = measure_code_blobs()
    check_c1_code_blobs(code_blobs, "work tree")
    m33_inputs.assert_inject_blob(code_blobs)
    r3_rows = hash_rows(settings.r3_root)
    check_r3_tree(settings, r3_rows)
    return StartState(code_blobs, r3_rows, read_report_inputs(settings))


def sorted_realization_rows(reports: Iterable[LineReport]) -> list[dict]:
    """Realization rows of every line, by line id and realization id."""
    return sorted(
        (row for report in reports for row in report.realization_rows),
        key=lambda row: (row["line_id"], row["realization_id"]),
    )


def summary_document(
    settings: ReportSettings,
    status: str,
    rows: Sequence[Mapping],
    checks: Mapping[str, Mapping],
    identity: Mapping,
) -> dict:
    """``summary.json``: definitions, counts, checks, then identity and timings."""
    return {
        "schema": SCHEMA,
        "status": status,
        "full_run": settings.is_full,
        "spec": "docs/processing/m33_r3_zero_change_definition.md §3, §5-2",
        "zero_change_tolerance_p": defs.ZERO_CHANGE_TOLERANCE_P,
        "period_change_tolerance_rel": defs.PERIOD_CHANGE_TOLERANCE_REL,
        "comparison": "strict: |dk| > 0.1 P, |P_inj - P| > 0.05 P (product form)",
        "n_lines": checks["null_test"]["n_lines"],
        "counts": grouped_counts(rows),
        **checks,
        "identity": dict(identity),
    }


def run_report(
    settings: ReportSettings, loader: LineLoader
) -> tuple[str, ReportCounts]:
    """Checks, line jobs, reproduction and end checks; writes the outputs.

    Raises:
        inputs.InputError: Before anything is written (exit 2).
    """
    started, started_at = time.perf_counter(), common.local_now()
    start = check_start(settings)
    report_inputs = start.report_inputs
    settings.out_dir.mkdir(parents=True)
    write_hash_rows(settings.out_dir / "r3_tree_sha256_start.csv", start.r3_rows)

    jobs = line_jobs(report_inputs, loader)
    reports = run_line_jobs(jobs, report_inputs.library, settings.workers)
    rows = sorted_realization_rows(reports)
    attach_stored(rows, report_inputs.stored)
    expected_ids = expected_realization_ids(settings, report_inputs)

    end_rows = hash_rows(settings.r3_root)
    write_hash_rows(settings.out_dir / "r3_tree_sha256_end.csv", end_rows)
    end_blobs = measure_code_blobs()
    checks = {
        "reproduction": reproduction_check(rows, expected_ids),
        "null_test": null_test_summary([report.line_row for report in reports]),
        "end_checks": end_check(start.r3_rows, end_rows, start.code_blobs, end_blobs),
    }
    status = decide_status(settings, len(rows), **checks)
    identity = report_identity(
        settings, report_inputs, started_at, (start.code_blobs, end_blobs)
    )
    summary = summary_document(settings, status, rows, checks, identity)
    summary["timings"] = {"elapsed_s": round(time.perf_counter() - started, 1)}
    _write_outputs(settings, reports, rows, summary)
    return status, ReportCounts(len(reports), len(rows))


# ================================================================== CLI


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Command line of the report and of the hash-list sub-command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--r3-root", type=Path, default=m33_inputs.R3_OUT_ROOT)
    parser.add_argument("--c1b-record", type=Path, default=m33_inputs.R3_C1B_RECORD)
    parser.add_argument("--result-hash-list", type=Path, default=RESULT_HASH_LIST)
    parser.add_argument("--upstream-hash-list", type=Path, default=UPSTREAM_HASH_LIST)
    parser.add_argument(
        "--limit-lines",
        type=int,
        default=None,
        help="first N lines of the table (smoke; status partial, exit 1)",
    )
    parser.add_argument(
        "--backgrounds",
        default=",".join(runner.BAND_BACKGROUNDS),
        help="4tu, gz_ids (both for the report)",
    )
    parser.add_argument(
        "--write-result-hash-list",
        type=Path,
        default=None,
        metavar="PATH",
        help="write the plan/injected hash list to PATH and exit",
    )
    return parser.parse_args(argv)


def settings_from(args: argparse.Namespace) -> ReportSettings:
    """``ReportSettings`` of parsed arguments."""
    backgrounds = tuple(
        item.strip() for item in args.backgrounds.split(",") if item.strip()
    )
    return ReportSettings(
        r3_root=args.r3_root,
        out_dir=args.out_dir,
        c1b_record=args.c1b_record,
        result_hash_list=args.result_hash_list,
        upstream_hash_list=args.upstream_hash_list,
        workers=args.workers,
        limit_lines=args.limit_lines,
        backgrounds=backgrounds,
    )


def _print_final_line(document: Mapping, started: float) -> None:
    line = dict(document) | {"elapsed_s": round(time.time() - started, 1)}
    print(json.dumps(line, ensure_ascii=False), flush=True)


def _hash_list_command(args: argparse.Namespace, started: float) -> int:
    n_files = 0
    try:
        n_files = write_result_hash_list(args.r3_root, args.write_result_hash_list)
        status = STATUS_OK
    except inputs.InputError as error:
        logger.error("input error: %s", error)
        status = STATUS_INPUT_ERROR
    except Exception:
        logger.exception("internal error")
        status = STATUS_ERROR
    document = {
        "status": status,
        "out": str(args.write_result_hash_list),
        "n_files": n_files,
    }
    _print_final_line(document, started)
    return {STATUS_OK: 0, STATUS_INPUT_ERROR: 2, STATUS_ERROR: 3}[status]


def main(argv: Sequence[str] | None = None, loader: LineLoader | None = None) -> int:
    """CLI entry; returns the exit code (module docstring tables).

    Args:
        argv: Command line (``sys.argv[1:]`` when ``None``).
        loader: Line loader for the workers; ``runner.load_line`` when
            ``None`` (tests pass a module-level stand-in).
    """
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    started = time.time()
    args = parse_args(argv)
    if args.write_result_hash_list is not None:
        return _hash_list_command(args, started)
    settings = settings_from(args)
    counts = ReportCounts()
    try:
        status, counts = run_report(settings, loader or runner.load_line)
    except inputs.InputError as error:
        logger.error("input error: %s", error)
        status = STATUS_INPUT_ERROR
    except Exception:
        logger.exception("internal error")
        status = STATUS_ERROR
    document = {
        "status": status,
        "out_dir": str(settings.out_dir),
        "n_lines": counts.n_lines,
        "n_realizations": counts.n_realizations,
    }
    _print_final_line(document, started)
    return EXIT_CODES[status]


if __name__ == "__main__":
    sys.exit(main())
