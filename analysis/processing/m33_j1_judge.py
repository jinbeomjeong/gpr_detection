# The verdict phrases copy [U3] §2-5 word for word (en dash, prime, minus).
# ruff: noqa: RUF001
"""M3-3 J1-2 / J1-3: 4TU end-to-end depth judgment on detector-box candidates.

Spec: ``docs/processing/m33_j1_preregistration.md`` v0.3.4 (§1 run unit and
order, §2 verdict, substantive metric and report items, §5 J1-2 assertions
and report computations, J1-3 report runs), [U3]
``docs/processing/preregistration_4tu_reeval_v1.md`` §2 (phrases, k, borders),
runner spec ``docs/experiments/m33_runner_spec.md`` §5 (R2 J1 inputs).

Judgment run (once per M3-3)::

    conda run -n gpr-dev python -I analysis/processing/m33_j1_judge.py ^
      --boxes runs/eval/m33/r2/fold0/j1_boxes_post.csv --boxes-sha256 <C2> ^
      --pp-config-sha256 <C0b> --prefix depth_4tu_j1

1. Assertions 1-4 and 6 of J1-2 (tree and C2 blobs, input sha, R2
   judgeability and identity, box file content, no ``depth_4tu_j1*`` output
   yet). Any failure: exit 2 before a file under ``docs/`` is written; the
   refusal is recorded under ``data/processing/m33_j1/``.
2. J1-1 (``m33_j1_candidates.py``) -> ``depth_4tu_j1_candidates_input.csv``;
   its sha is recorded and re-read before the depth run (assertion 5).
3. ``scripts/eval_depth_4tu.py`` ``main(["--candidates", ..., "--prefix",
   "depth_4tu_j1"])`` in this process, unchanged; its log and stdout go to a
   file (assertion 7, blind).
4. §1-1 3 counts (``n_pass`` from the matches csv, ``k`` from
   ``baseline_p``) with their rounding assertions (exit 2 on failure), the
   §2-3 phrase, the §2-4 substantive metric and the J1-2 report computations
   -> ``depth_4tu_j1_verdict.json`` and ``.md``. A baseline that cannot be
   reproduced leaves the report items ``null`` with the reason (exit 3).

Report runs (J1-3, only after the verdict file exists, each once):
``--report pre|fold1|fold2|fold3|fold4|signal`` with the same ``--boxes`` and
``--pp-config-sha256``; prefixes ``depth_4tu_j1_pre``, ``_f1``-``_f4``,
``_signal``; results in ``<prefix>_report.json`` and ``.md``. They never
write the verdict files.

``--measure-seal`` prints the C2 values (blobs and sha only) to fill
``m33_j1_seal.py`` at the C2 commit.

Blind: judgment values go to files only. stdout carries phase names and the
last line ``{"status", "out", "n_candidates", "elapsed_s"}``.
Exit codes: 0 done, 2 input refused, 3 internal error or report failure.
"""

import os

# eval_depth_4tu sets this before numpy loads MKL (bit-stable depths); numpy
# is imported here first, so the same default must be set here.
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import argparse
import contextlib
import json
import logging
import math
import platform
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import beta

REPO = Path(__file__).resolve().parents[2]
for _folder in ("scripts", "analysis/detection", "analysis/processing"):
    if str(REPO / _folder) not in sys.path:
        sys.path.insert(0, str(REPO / _folder))

import eval_depth_4tu as depth_eval  # noqa: E402
import m31_d1_common as common  # noqa: E402
import m33_inputs  # noqa: E402
import m33_j1_candidates as j1_candidates  # noqa: E402
import m33_j1_seal as seal  # noqa: E402
from m32_recipe_inputs import InputError  # noqa: E402

from gpr_detection.io.fourtu import METADATA_FOLDER, site_of_activity  # noqa: E402
from gpr_detection.io.paths import load_dataset_paths  # noqa: E402

LOGGER = logging.getLogger("m33_j1_judge")

SCHEMA = "m33_j1_judge/1"
VERDICT_PREFIX = "depth_4tu_j1"
OUT_DIR = REPO / "docs" / "processing"
RUN_LOG_DIR = REPO / "data" / "processing" / "m33_j1"
"""Logs, eval stdout and refusal records (git-ignored, never judgment files)."""

N_LINES_FULL = 754
EXPECTED_FOLDS = [0, 1, 2, 3, 4]
N_GT = 388
"""Ground truths of the main set (exclude_x, 93 activities, [U3] §2-2)."""
TARGET_N_PASS = 97
"""``n_T = ceil(0.25 * 388)`` ([U3] §2-4)."""
P_MAX_K = 49
"""``p < 0.05 <=> k <= 49`` for 1,000 trials ([U3] §2-3)."""
P_BORDER_K = (37, 64)
"""Clopper-Pearson border of ``k`` ([U3] §6, D6)."""
TARGET_BORDER_N = 9
"""``|n_pass - 97| <= 9``: one binomial SE ([U3] §6)."""
K_INTEGER_TOL = 1e-6
"""``|baseline_p * 1000 - k| < 1e-6`` (J1 §1-1 3)."""
CLOPPER_PEARSON_LEVEL = 0.95
SUMMARY_DECIMALS = 4
"""Decimals of the summary csv; the reproduction is compared at this place."""
N_REQ_RANK = 50
"""``n_req = T_(50) + 1``: 50th largest baseline total (§2-4 (ii))."""
BOOTSTRAP_SEED = 20261115
BOOTSTRAP_REPLICATES = 2000
"""Site bootstrap (§1-4 보고용 난수, §2-4)."""

MAIN_COMBINATION = {
    "gt_set": "exclude_x",
    "assumption": "top",
    "zero_variant": "detected",
    "variant": "main",
}
"""The verdict combination ([U3] §2-1)."""
SCOPE_ALL = "all"
SITE_SCOPE_PREFIX = "site_"
BASELINE_COLUMNS = (
    "baseline_pass_b_mean",
    "baseline_pass_b_p025",
    "baseline_pass_b_p975",
    "baseline_p",
)
"""Summary columns the reproduced baseline must equal (J1-2 보고 계산 2)."""

SEAL_MODULE_PATH = "analysis/processing/m33_j1_seal.py"
"""The commit that last changed this file is the C2 commit."""
DOCS_ONLY_PREFIXES = ("docs/", ".claude/agent-memory/")
"""Commits after C2 may change only files below these folders (assertion 1,
J1 v0.3.4: agent memory counts as documentation)."""
SRC_PACKAGE_PATH = "src/gpr_detection"
RECORDED_CODE_PATHS = (
    SEAL_MODULE_PATH,
    "analysis/detection/m33_inputs.py",
    "analysis/detection/m31_d1_common.py",
    "analysis/detection/m32_recipe_inputs.py",
    "analysis/detection/m32_aihub_perturb.py",
    "analysis/detection/m33_r2_lines.csv",
    m33_inputs.PIPELINE_PATH,
)
"""Code and tables recorded beside the sealed ones (blobs, not asserted)."""

CANDIDATE_SOURCE_TEXT = " · 후보: 검출기 N2-a′ fold 0, PP(C0b) 후 박스"
"""J1 §2-3 (v0.3.4): candidate source, inside the [U3] parenthesis."""
U3_PARENTHESIS = "(판정률 B, `main`·관 상단·검출 영점·X 제외)"
J1_PARENTHESIS = U3_PARENTHESIS[:-1] + CANDIDATE_SOURCE_TEXT + ")"
"""``(판정률 B, … X 제외 · 후보: … 박스)``: the particle after it stays [U3]'s."""

STATUS_OK = "ok"
STATUS_INPUT_ERROR = "input_error"
STATUS_REPORT_FAILED = "report_failed"
STATUS_INTERNAL_ERROR = "internal_error"
EXIT_CODES = {
    STATUS_OK: 0,
    STATUS_INPUT_ERROR: 2,
    STATUS_REPORT_FAILED: 3,
    STATUS_INTERNAL_ERROR: 3,
}


class ReportKind(StrEnum):
    """J1-3 report runs (``--report``)."""

    PRE = "pre"
    FOLD1 = "fold1"
    FOLD2 = "fold2"
    FOLD3 = "fold3"
    FOLD4 = "fold4"
    SIGNAL = "signal"


class VerdictCase(StrEnum):
    """The four [U3] §2-5 cases."""

    PASS = "pass"
    FAIL_BASELINE = "fail_baseline"
    FAIL_TARGET = "fail_target"
    FAIL_BOTH = "fail_both"


PHRASES = {
    VerdictCase.PASS: (
        "판정: **합격** — 합격률 {n_pass}/388{parenthesis}로 절대 목표 T = 0.25"
        "(97/388)를 충족했고, 활동 범위 무작위 기준선(같은 추정 수, 1,000회, "
        "`default_rng(0)`) 대비 단측 p = {k}/1000 < 0.05다({baseline})."
    ),
    VerdictCase.FAIL_BASELINE: (
        "판정: **불합격(기준선)** — 합격률 {n_pass}/388{parenthesis}로 절대 목표 "
        "T = 0.25(97/388)는 충족했으나, 활동 범위 무작위 기준선(같은 추정 수, "
        "1,000회, `default_rng(0)`) 대비 단측 p = {k}/1000 ≥ 0.05다({baseline}). "
        "이 기준선은 짝의 약 95 %를 맞힐 만큼 강하며, 그 때문에 이 조건으로 "
        "불합격할 가능성이 높다는 것을 알고 실행 전에 유지한 기준이다"
        "(사전 등록 §2-6, D5)."
    ),
    VerdictCase.FAIL_TARGET: (
        "판정: **불합격(목표)** — 활동 범위 무작위 기준선(같은 추정 수, 1,000회, "
        "`default_rng(0)`) 대비 단측 p = {k}/1000 < 0.05는 충족했으나, 합격률 "
        "{n_pass}/388{parenthesis}이 절대 목표 T = 0.25(97/388)에 못 미친다"
        "({baseline})."
    ),
    VerdictCase.FAIL_BOTH: (
        "판정: **불합격(둘 다)** — 합격률 {n_pass}/388{parenthesis}이 절대 목표 "
        "T = 0.25(97/388)에 못 미치고, 활동 범위 무작위 기준선(같은 추정 수, "
        "1,000회, `default_rng(0)`) 대비 단측 p = {k}/1000 ≥ 0.05다({baseline})."
    ),
}
"""[U3] §2-5 phrases word for word; ``{parenthesis}`` carries the J1 §2-3 text."""

SITE_CLUSTER_NOTE = (
    "목표 경계(±9)는 짝을 독립으로 본 이항 1 SE이고 현장 군집을 반영하지 않는다. "
    "현장이 10곳뿐이라 현장 부트스트랩 구간은 실제 불확도보다 좁게 나올 수 있다."
)
"""J1 §2-4 public sentence (copied to §6)."""


# ================================================================= checks


@dataclass
class Checks:
    """Named assertion results, kept in order for the output json."""

    records: list[dict] = field(default_factory=list)

    def require(self, name: str, passed: bool, detail: str = "") -> bool:
        self.records.append({"name": name, "passed": bool(passed), "detail": detail})
        return bool(passed)

    def failed(self) -> list[str]:
        return [record["name"] for record in self.records if not record["passed"]]

    def raise_if_failed(self) -> None:
        failed = self.failed()
        if failed:
            raise InputError("assertions failed: " + ", ".join(failed))


@dataclass(frozen=True)
class RunSpec:
    """What one invocation evaluates (the verdict or one report run)."""

    prefix: str
    report: ReportKind | None
    fold: int | None
    pp: str | None

    @property
    def is_verdict(self) -> bool:
        return self.report is None

    @property
    def uses_boxes(self) -> bool:
        return self.fold is not None

    def boxes_path(self, r2_root: Path) -> Path:
        return r2_root / f"fold{self.fold}" / f"j1_boxes_{self.pp}.csv"

    @property
    def candidates_path(self) -> Path:
        if not self.uses_boxes:
            return depth_eval.DEFAULT_APEX_CANDIDATES
        return OUT_DIR / f"{self.prefix}_candidates_input.csv"

    @property
    def result_json(self) -> Path:
        suffix = "verdict" if self.is_verdict else "report"
        return OUT_DIR / f"{self.prefix}_{suffix}.json"


def run_spec(report: ReportKind | None) -> RunSpec:
    """Prefix, fold and PP side of the verdict or of one report run (J1-3)."""
    if report is None:
        return RunSpec(VERDICT_PREFIX, None, 0, "post")
    if report is ReportKind.PRE:
        return RunSpec(f"{VERDICT_PREFIX}_pre", report, 0, "pre")
    if report is ReportKind.SIGNAL:
        return RunSpec(f"{VERDICT_PREFIX}_signal", report, None, None)
    fold = int(report.value.removeprefix("fold"))
    return RunSpec(f"{VERDICT_PREFIX}_f{fold}", report, fold, "post")


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=common.GIT_TIMEOUT_S,
        check=True,
    )
    return completed.stdout.strip()


def is_ancestor(ancestor: str, head: str) -> bool:
    """``git merge-base --is-ancestor`` (a commit is its own ancestor)."""
    completed = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", ancestor, head],
        capture_output=True,
        timeout=common.GIT_TIMEOUT_S,
        check=False,
    )
    return completed.returncode == 0


def _prefix_outputs(prefix: str) -> list[str]:
    return sorted(path.name for path in OUT_DIR.glob(f"{prefix}*"))


def check_no_prior_outputs(checks: Checks, spec: RunSpec) -> None:
    """Assertion 6 (one run): no output of the prefix; reports need the verdict."""
    if spec.is_verdict:
        existing = _prefix_outputs(VERDICT_PREFIX)
        checks.require("6.no_depth_4tu_j1_output", not existing, ", ".join(existing))
        return
    verdict = run_spec(None).result_json
    checks.require("J1-3.verdict_written_first", verdict.is_file(), verdict.name)
    existing = _prefix_outputs(spec.prefix)
    checks.require("J1-3.no_report_output", not existing, ", ".join(existing))


def check_seal_filled(checks: Checks) -> None:
    missing = seal.unfilled_names()
    checks.require("1.c2_seal_filled", not missing, ", ".join(missing))


# ------------------------------------------------------------ assertion 1


def _sealed_paths() -> list[Path]:
    return [REPO / name for name in seal.SEALED_BLOBS]


def _head_blob(name: str) -> str | None:
    try:
        return _git("rev-parse", f"HEAD:{name}")
    except (OSError, subprocess.SubprocessError):
        return None


def _same_known(measured: object, *expected: object) -> bool:
    """All values equal and known (an unfilled seal never matches a missing file)."""
    return measured is not None and all(value == measured for value in expected)


def _sha_lf_or_none(path: Path) -> str | None:
    return common.sha256_file_lf(path) if path.is_file() else None


def _check_sealed_files(checks: Checks, work_blobs: Mapping[str, str | None]) -> None:
    for name, expected in seal.SEALED_BLOBS.items():
        head_blob = _head_blob(name)
        checks.require(
            f"1.blob.{name}", _same_known(head_blob, expected, work_blobs.get(name))
        )
        lf_sha = _sha_lf_or_none(REPO / name)
        checks.require(
            f"1.sha_lf.{name}", _same_known(lf_sha, seal.SEALED_SHA256_LF[name])
        )


def _check_head_after_c2(checks: Checks) -> None:
    try:
        c2_commit = _git("log", "-1", "--format=%H", "--", SEAL_MODULE_PATH)
        changed = _git("diff", "--name-only", c2_commit, "HEAD").splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        checks.require("1.head_is_c2_or_docs_only", False, repr(error))
        return
    not_docs = [name for name in changed if not name.startswith(DOCS_ONLY_PREFIXES)]
    detail = f"c2={c2_commit}; non-docs changes: {not_docs}"
    checks.require(
        "1.head_is_c2_or_docs_only", bool(c2_commit) and not not_docs, detail
    )


def check_tree(checks: Checks) -> dict:
    """Assertion 1: clean tree, C2 blobs and LF sha, ``src`` tree, HEAD rule.

    Also asserts the sealed PP (C0b ``d0_gate.json`` sha, ``PPConfig`` hash,
    C0a ``pipeline.py``) with ``m33_inputs.assert_sealed_pp``.

    Returns:
        The tree record (HEAD, problems, blobs) for the output json.
    """
    recorded = [REPO / name for name in RECORDED_CODE_PATHS]
    tree = m33_inputs.tree_state(_sealed_paths() + recorded, _sealed_paths() + recorded)
    checks.require("1.tree_clean", not tree["dirty"], "; ".join(tree["problems"]))
    _check_sealed_files(checks, tree["code_blobs"])
    try:
        src_tree = _git("rev-parse", f"HEAD:{SRC_PACKAGE_PATH}")
    except (OSError, subprocess.SubprocessError):
        src_tree = None
    checks.require(
        "1.src_tree_id", _same_known(src_tree, seal.SRC_TREE_ID), str(src_tree)
    )
    _check_head_after_c2(checks)
    try:
        sealed_pp = m33_inputs.assert_sealed_pp(tree["code_blobs"])
        checks.require("3.sealed_pp_c0b", True, sealed_pp["pp_set_id"])
    except InputError as error:
        checks.require("3.sealed_pp_c0b", False, str(error))
    return {**tree, "src_gpr_detection_tree_id": src_tree}


def fourtu_meta_paths() -> dict[str, Path]:
    """The four 4TU metadata csv files (read-only dataset)."""
    folder = load_dataset_paths().fourtu / METADATA_FOLDER
    return {name: folder / name for name in seal.FOURTU_META_SHA256}


def check_fourtu_meta(checks: Checks) -> None:
    """Assertion 1: raw sha of the 4TU metadata csv files."""
    for name, path in fourtu_meta_paths().items():
        measured = common.sha256_file(path) if path.is_file() else None
        expected = seal.FOURTU_META_SHA256[name]
        checks.require(f"1.fourtu_meta.{name}", _same_known(measured, expected))


# ------------------------------------------------------------ assertion 3


def read_r2_summary(r2_root: Path) -> dict:
    """R2 ``summary.json``.

    Raises:
        InputError: When it is missing or not json.
    """
    path = r2_root / "summary.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise InputError(f"R2 summary unreadable: {path}") from error


def _check_judgeability(checks: Checks, summary: Mapping) -> None:
    judgment = summary.get("judgment") or {}
    j1 = summary.get("j1_inputs") or {}
    checks.require("3.status_ok", summary.get("status") == m33_inputs.STATUS_OK)
    checks.require("3.judgeable", judgment.get("judgeable") is True)
    checks.require("3.full_run", summary.get("full_run") is True)
    checks.require("3.no_blockers", judgment.get("blockers") == [])
    folds_run, expected = judgment.get("folds_run"), judgment.get("expected_folds")
    checks.require("3.all_folds", folds_run == expected == EXPECTED_FOLDS)
    checks.require("3.j1_n_lines_754", j1.get("n_lines") == N_LINES_FULL)
    checks.require("3.n_lines_754", summary.get("n_lines") == N_LINES_FULL)


def _check_weights(checks: Checks, summary: Mapping) -> None:
    j1_weights = (summary.get("j1_inputs") or {}).get("weights_sha256") or {}
    identity = summary.get("run_identity") or {}
    run_weights = identity.get("inputs_sha256") or {}
    from_j1, from_identity = j1_weights.get("0"), run_weights.get("0")
    both_present = bool(from_j1) and bool(from_identity)
    same = from_j1 == from_identity == seal.FOLD0_WEIGHTS_SHA256
    checks.require("3.fold0_weights_sha", both_present and same)


def _expected_r0_sha() -> dict[str, str]:
    return {f"{d}/{line}": sha for d, line, sha in m33_inputs.R0_IDENTITY_LINES}


def _check_r_checks(checks: Checks, identity: Mapping) -> None:
    r_checks = identity.get("r_checks") or {}
    settings_sha = (r_checks.get("settings") or {}).get("settings_sha256")
    checks.require("3.settings_sha_c1", settings_sha == m33_inputs.R_SETTINGS_SHA256)
    r0_sha = r_checks.get("r0_identity_sha256")
    checks.require("3.r0_11_lines_c1", r0_sha == _expected_r0_sha())


def _check_r2_head(checks: Checks, identity: Mapping) -> None:
    head = (identity.get("git") or {}).get("head")
    tree_end = identity.get("tree_end") or {}
    same_head = bool(head) and head == tree_end.get("head")
    after_c1 = same_head and is_ancestor(seal.C1_SEAL_COMMIT, str(head))
    checks.require("3.r2_head_after_c1", after_c1, str(head))
    clean = identity.get("dirty") is False and tree_end.get("dirty") is False
    checks.require("3.r2_tree_clean", clean)


def check_r2_summary(checks: Checks, summary: Mapping, pp_config_sha: str) -> None:
    """Assertion 3: judgeability (가)-(라), weights, r0, settings, PP, HEAD."""
    _check_judgeability(checks, summary)
    _check_weights(checks, summary)
    identity = summary.get("run_identity") or {}
    _check_r_checks(checks, identity)
    j1 = summary.get("j1_inputs") or {}
    pp_sha = j1.get("pp_config_sha256")
    sealed_sha = m33_inputs.C0B_PP_CONFIG_SHA256
    checks.require("3.pp_config_sha_c0b", pp_sha == pp_config_sha == sealed_sha)
    sealed_set = m33_inputs.C0B_PP_SET_ID.value
    checks.require("3.pp_set_id_c0b", j1.get("pp_set_id") == sealed_set)
    _check_r2_head(checks, identity)


# ---------------------------------------------------------- assertions 2, 4


def load_line_set() -> set[tuple[str, str]]:
    """The 754 R2 lines (``m33_r2_lines.csv``, LF sha asserted).

    Raises:
        InputError: When the list sha is not the C1 one.
    """
    path = m33_inputs.R2_LINES_CSV
    if common.sha256_file_lf(path) != m33_inputs.R2_LINES_SHA256_LF:
        raise InputError(f"R2 line list changed: {path}")
    table = _read_text_csv(path)
    return set(zip(table["activity_id"], table["path"], strict=True))


def _fold_entry(summary: Mapping, fold: int) -> Mapping:
    folds = (summary.get("j1_inputs") or {}).get("folds") or {}
    return folds.get(str(fold)) or {}


def _expected_box_sha(spec: RunSpec) -> str | None:
    if spec.is_verdict:
        return seal.R2_J1_POST_SHA256
    if spec.report is ReportKind.PRE:
        return seal.R2_J1_PRE_SHA256
    return None


def check_box_sha(
    checks: Checks, spec: RunSpec, path: Path, summary: Mapping, cli_sha: str | None
) -> str | None:
    """Assertion 2: file sha = C2 value (verdict, pre) = R2 summary (= CLI)."""
    measured = common.sha256_file(path) if path.is_file() else None
    recorded = _fold_entry(summary, spec.fold or 0).get(f"{spec.pp}_sha256")
    checks.require(f"2.{spec.pp}_sha_r2_summary", _same_known(measured, recorded))
    if spec.is_verdict or spec.report is ReportKind.PRE:
        expected = _expected_box_sha(spec)
        checks.require(f"2.{spec.pp}_sha_c2", _same_known(measured, expected))
    if spec.is_verdict:
        checks.require("2.post_sha_cli", _same_known(measured, cli_sha))
    return measured


def _check_line_ids(checks: Checks, rows: Sequence[Mapping[str, str]]) -> None:
    def splits(row: Mapping[str, str]) -> bool:
        activity_id, _, path = row["line_id"].partition("/")
        return (activity_id, path) == (row["activity_id"], row["path"])

    checks.require("4.line_id_split", all(splits(row) for row in rows))


def check_box_rows(
    checks: Checks,
    spec: RunSpec,
    rows: Sequence[Mapping[str, str]],
    line_set: set[tuple[str, str]],
) -> None:
    """Assertion 4: fold, PP action (post), class, ``line_id`` and line list."""
    checks.require("4.fold", all(row["fold"] == str(spec.fold) for row in rows))
    if spec.pp == "post":
        removed = [row for row in rows if row["pp_action"].startswith("removed_")]
        checks.require("4.no_removed_rows", not removed)
    hyperbola = str(int(j1_candidates.DetectionClass.PIPE_HYPERBOLA))
    checks.require(
        "4.class_pipe_hyperbola", all(r["class_id"] == hyperbola for r in rows)
    )
    _check_line_ids(checks, rows)
    outside = [r for r in rows if (r["activity_id"], r["path"]) not in line_set]
    checks.require("4.lines_in_754", not outside, f"{len(outside)} rows")


# ================================================================== inputs


@dataclass(frozen=True)
class RunInputs:
    """What the pre-run assertions established."""

    spec: RunSpec
    r2_root: Path
    summary: dict
    boxes_path: Path | None
    boxes_sha256: str | None
    line_set: set[tuple[str, str]]
    tree: dict


def _check_boxes(
    checks: Checks, spec: RunSpec, inputs: dict, cli_sha: str | None
) -> tuple[Path, str | None]:
    path = spec.boxes_path(inputs["r2_root"])
    sha = check_box_sha(checks, spec, path, inputs["summary"], cli_sha)
    try:
        rows = j1_candidates.read_box_rows(path)
    except (OSError, ValueError) as error:
        checks.require("4.box_file_readable", False, str(error))
        return path, sha
    check_box_rows(checks, spec, rows, inputs["line_set"])
    return path, sha


def _check_signal_candidates(checks: Checks) -> None:
    path = depth_eval.DEFAULT_APEX_CANDIDATES
    measured = _sha_lf_or_none(path)
    expected = seal.SIGNAL_CANDIDATES_SHA256_LF
    checks.require("signal.candidates_sha_u3", _same_known(measured, expected))


def assert_run_inputs(
    checks: Checks, spec: RunSpec, args: argparse.Namespace
) -> RunInputs:
    """J1-2 assertions 1-4 and 6 (all recorded, then one refusal).

    Raises:
        InputError: When any assertion failed (exit 2, nothing under docs/).
    """
    check_no_prior_outputs(checks, spec)
    check_seal_filled(checks)
    checks.require("args.prefix", args.prefix == VERDICT_PREFIX, args.prefix)
    out_dir_ok = depth_eval.DEFAULT_OUT_DIR.resolve() == OUT_DIR.resolve()
    checks.require("args.eval_out_dir", out_dir_ok)
    tree = check_tree(checks)
    check_fourtu_meta(checks)
    boxes = args.boxes.resolve()
    r2_root = boxes.parent.parent
    is_fold0_post = (boxes.parent.name, boxes.name) == ("fold0", "j1_boxes_post.csv")
    checks.require("args.boxes_is_fold0_post", is_fold0_post)
    summary = read_r2_summary(r2_root)
    check_r2_summary(checks, summary, args.pp_config_sha256)
    inputs = {"r2_root": r2_root, "summary": summary, "line_set": load_line_set()}
    boxes_path, boxes_sha = None, None
    if spec.uses_boxes:
        boxes_path, boxes_sha = _check_boxes(checks, spec, inputs, args.boxes_sha256)
    else:
        _check_signal_candidates(checks)
    checks.raise_if_failed()
    return RunInputs(
        spec, r2_root, summary, boxes_path, boxes_sha, inputs["line_set"], tree
    )


# ============================================================ candidates


@dataclass(frozen=True)
class CandidateRecord:
    """The candidate csv a run fed to the depth evaluation."""

    path: Path
    sha256: str
    counts: dict


def write_candidates(inputs: RunInputs) -> CandidateRecord:
    """J1-1, then assertion 5 (sha recorded now, re-read before the depth run)."""
    spec = inputs.spec
    assert inputs.boxes_path is not None and spec.fold is not None and spec.pp
    built = j1_candidates.build_candidate_set(
        inputs.boxes_path, spec.fold, spec.pp, inputs.line_set
    )
    j1_candidates.write_candidates_csv(built.table, spec.candidates_path)
    sha = common.sha256_file(spec.candidates_path)
    counts = {
        "n_boxes": built.n_boxes,
        "n_candidates_before_dedupe": built.n_before_dedupe,
        "n_candidates": built.n_candidates,
        "n_boxes_without_candidate": built.skipped_by_reason(),
        "n_candidates_tau_blank": built.n_zero_failed,
    }
    return CandidateRecord(spec.candidates_path, sha, counts)


def recheck_candidates(checks: Checks, record: CandidateRecord) -> None:
    """Assertion 5: the csv is unchanged right before ``eval_depth_4tu.main``."""
    checks.require(
        "5.candidates_sha_unchanged", common.sha256_file(record.path) == record.sha256
    )
    checks.raise_if_failed()


def signal_candidates() -> CandidateRecord:
    path = depth_eval.DEFAULT_APEX_CANDIDATES
    n_rows = len(pd.read_csv(path, dtype=str))
    return CandidateRecord(path, common.sha256_file(path), {"n_candidates": n_rows})


# ============================================================ depth run


@contextlib.contextmanager
def blind_output(log_path: Path) -> Iterator[None]:
    """Send logging (INFO) and stdout / stderr of the block to ``log_path``.

    ``eval_depth_4tu.main`` calls ``logging.basicConfig``; with a handler
    already on the root logger that call does nothing, so its progress log
    lands in the file, never on the console (assertion 7).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    with log_path.open("a", encoding="utf-8") as stream:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        root.handlers = [handler]
        root.setLevel(logging.INFO)
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                yield
        finally:
            root.handlers = saved_handlers
            root.setLevel(saved_level)


def run_depth_eval(candidates_path: Path, prefix: str) -> None:
    """``eval_depth_4tu.main`` in this process with the spec's two arguments.

    Raises:
        RuntimeError: When it returns a non-zero code.
    """
    argv = ["--candidates", str(candidates_path), "--prefix", prefix]
    with blind_output(RUN_LOG_DIR / f"{prefix}_eval.log"):
        code = depth_eval.main(argv)
    if code != 0:
        raise RuntimeError(f"eval_depth_4tu.main returned {code}")


@dataclass(frozen=True)
class EvalOutputs:
    """The depth evaluation csv files of one prefix (all cells as text)."""

    summary: pd.DataFrame
    matches: pd.DataFrame
    clusters: pd.DataFrame
    candidates: pd.DataFrame


MATCH_COLUMNS = ("activity_id", *MAIN_COMBINATION, "within_30cm")
CLUSTER_COLUMNS = ("activity_id", "zero_variant", "kept_by_support")
"""Columns read from files that ``write_csv`` leaves empty when it has no row."""


def _read_text_csv(path: Path, columns: Sequence[str] = ()) -> pd.DataFrame:
    """A csv with every cell as text; an empty file gives ``columns`` and no row."""
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=list(columns), dtype=str)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


EVAL_OUTPUT_SUFFIXES = ("summary", "matches", "clusters", "candidates")
"""The depth-run csv files the judge reads (``<prefix>_<suffix>.csv``)."""


def eval_output_sha256(prefix: str) -> dict[str, dict[str, str]]:
    """Raw and LF sha256 of the depth-run csv files the judge reads.

    A record only (main decision 2026-11-16): the post-judgment audit can
    check that the files a verdict cites are the ones of this run. The raw
    sha is of the bytes as eval wrote them (CRLF rows of ``csv``); the LF sha
    does not depend on a later checkout.
    """
    paths = {
        f"{prefix}_{suffix}.csv": OUT_DIR / f"{prefix}_{suffix}.csv"
        for suffix in EVAL_OUTPUT_SUFFIXES
    }
    return {
        "raw": {name: common.sha256_file(path) for name, path in paths.items()},
        "lf": {name: common.sha256_file_lf(path) for name, path in paths.items()},
    }


def read_eval_outputs(prefix: str) -> EvalOutputs:
    return EvalOutputs(
        summary=_read_text_csv(OUT_DIR / f"{prefix}_summary.csv"),
        matches=_read_text_csv(OUT_DIR / f"{prefix}_matches.csv", MATCH_COLUMNS),
        clusters=_read_text_csv(OUT_DIR / f"{prefix}_clusters.csv", CLUSTER_COLUMNS),
        candidates=_read_text_csv(OUT_DIR / f"{prefix}_candidates.csv"),
    )


def main_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Rows of the verdict combination (``exclude_x``, top, detected, main)."""
    mask = np.ones(len(table), dtype=bool)
    for column, value in MAIN_COMBINATION.items():
        mask &= (table[column] == value).to_numpy()
    return table[mask]


# ================================================================ verdict


@dataclass(frozen=True)
class MainResult:
    """§1-1 3 counts of the main combination, scope ``all``."""

    n_pass: int
    k: int
    n_pairs: int
    pass_b: float
    baseline_p: float
    baseline_mean: float
    baseline_p025: float
    baseline_p975: float
    row: dict


def _number(text: str) -> float | None:
    return None if text == "" else float(text)


def _main_matches_counts(matches: pd.DataFrame) -> tuple[int, int]:
    rows = main_rows(matches)
    n_pass = int((rows["within_30cm"] == "True").sum())
    return n_pass, len(rows)


def read_main_result(outputs: EvalOutputs, checks: Checks) -> MainResult | None:
    """§1-1 3: ``n_pass`` from matches, ``k`` from ``baseline_p``, all asserted."""
    summary_rows = main_rows(outputs.summary)
    all_rows = summary_rows[summary_rows["scope"] == SCOPE_ALL]
    if not checks.require("1-1.3.one_main_row", len(all_rows) == 1, str(len(all_rows))):
        return None
    row = all_rows.iloc[0].to_dict()
    n_pass, n_match_rows = _main_matches_counts(outputs.matches)
    pass_b, baseline_p = float(row["pass_b"]), float(row["baseline_p"])
    k = round(baseline_p * 1000)
    checks.require("1-1.3.n_gt_388", int(row["n_gt"]) == N_GT)
    checks.require("1-1.3.n_pass_round_pass_b", n_pass == round(pass_b * N_GT))
    checks.require("1-1.3.match_rows_n_pairs", n_match_rows == int(row["n_pairs"]))
    if row.get("n_pass", "") != "":
        checks.require("1-1.3.summary_n_pass", int(row["n_pass"]) == n_pass)
    checks.require("1-1.3.k_integer", abs(baseline_p * 1000 - k) < K_INTEGER_TOL)
    if checks.failed():
        return None
    return MainResult(
        n_pass=n_pass,
        k=k,
        n_pairs=int(row["n_pairs"]),
        pass_b=pass_b,
        baseline_p=baseline_p,
        baseline_mean=float(row["baseline_pass_b_mean"]),
        baseline_p025=float(row["baseline_pass_b_p025"]),
        baseline_p975=float(row["baseline_pass_b_p975"]),
        row=row,
    )


def decide_case(n_pass: int, k: int) -> VerdictCase:
    """[U3] §2-5: ``k <= 49`` (p < 0.05) and ``n_pass >= 97``."""
    meets_p = k <= P_MAX_K
    meets_target = n_pass >= TARGET_N_PASS
    if meets_p and meets_target:
        return VerdictCase.PASS
    if meets_target:
        return VerdictCase.FAIL_BASELINE
    if meets_p:
        return VerdictCase.FAIL_TARGET
    return VerdictCase.FAIL_BOTH


def clopper_pearson(
    k: int, n: int, level: float = CLOPPER_PEARSON_LEVEL
) -> tuple[float, float]:
    """Exact binomial interval of ``k / n``."""
    alpha = 1.0 - level
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return low, high


def border_markers(n_pass: int, k: int) -> list[str]:
    """[U3] §2-5 border notes appended to the phrase (the verdict is unchanged)."""
    markers = []
    if P_BORDER_K[0] <= k <= P_BORDER_K[1]:
        low, high = clopper_pearson(k, depth_eval.BASELINE_TRIALS)
        interval = f"{low:.3f}–{high:.3f}"
        markers.append(
            f" [p 경계: 몬테카를로 불확도 안, Clopper–Pearson 95 % {interval}]"
        )
    if abs(n_pass - TARGET_N_PASS) <= TARGET_BORDER_N:
        markers.append(" [목표 경계: 표본 오차 안]")
    return markers


def verdict_phrase(case: VerdictCase, result: MainResult) -> str:
    """The [U3] phrase of ``case`` with the J1 candidate text and the markers."""
    baseline = (
        f"기준선 판정률 B 평균 {result.baseline_mean:.4f}, "
        f"2.5–97.5 % {result.baseline_p025:.4f}–{result.baseline_p975:.4f}"
    )
    phrase = PHRASES[case].format(
        n_pass=result.n_pass,
        k=result.k,
        parenthesis=J1_PARENTHESIS,
        baseline=baseline,
    )
    return phrase + "".join(border_markers(result.n_pass, result.k))


def _row_value(row: Mapping[str, str], column: str) -> float | None:
    return _number(row.get(column, ""))


def accompanying_values(result: MainResult) -> dict:
    """[U3] §2-5: values written under the phrase (never used for the verdict)."""
    row = result.row
    return {
        "n_pass": result.n_pass,
        "n_gt": N_GT,
        "pass_b": result.n_pass / N_GT,
        "k": result.k,
        "p": result.k / depth_eval.BASELINE_TRIALS,
        "n_pairs": result.n_pairs,
        "pass_a": _row_value(row, "pass_a"),
        "b_max": result.n_pairs / N_GT,
        "baseline_pass_a_mean": _row_value(row, "baseline_pass_a_mean"),
        "mae_m": _row_value(row, "mae_m"),
        "bias_m": _row_value(row, "bias_m"),
        "missed": int(row["missed"]),
        "false_est": int(row["false_est"]),
        "n_est": int(row["n_est"]),
    }


def substantive_metric(result: MainResult) -> dict:
    """§2-4: ``Delta_B = B - mean baseline B`` with the baseline interval."""
    return {
        "pass_b": result.n_pass / N_GT,
        "baseline_pass_b_mean": result.baseline_mean,
        "baseline_pass_b_p025": result.baseline_p025,
        "baseline_pass_b_p975": result.baseline_p975,
        "delta_b": result.n_pass / N_GT - result.baseline_mean,
    }


# =============================================================== report


@dataclass(frozen=True)
class BaselineInputs:
    """Ground truth (main set, top) and lines per activity, as the depth run."""

    gt_depths: dict[str, list[float]]
    """``exclude_x`` depths per activity, in the depth run's activity order."""
    lines_per_activity: dict[str, int]


def load_baseline_inputs() -> BaselineInputs:
    """The depth run's own loaders: 4TU ground truth and the line counts."""
    root = load_dataset_paths().fourtu
    ground_truth = depth_eval.load_ground_truth_sets(root)["exclude_x"]
    lines = pd.read_csv(
        depth_eval.DEFAULT_LINES, dtype={"activity_id": str, "path": str}
    )
    table = depth_eval.load_duplicate_table(depth_eval.DEFAULT_DUPLICATES)
    lines_per_activity = depth_eval.unique_line_counts(lines, table)
    gt_depths = {
        activity: [item.depth_for(MAIN_COMBINATION["assumption"]) for item in items]
        for activity, items in ground_truth.items()
        if activity in lines_per_activity
    }
    return BaselineInputs(gt_depths, lines_per_activity)


def kept_estimate_counts(clusters: pd.DataFrame) -> dict[str, int]:
    """``n_e,a``: ``main`` clusters kept by support at the detected zero."""
    kept = clusters[
        (clusters["zero_variant"] == MAIN_COMBINATION["zero_variant"])
        & (clusters["kept_by_support"] == "True")
    ]
    return kept.groupby("activity_id").size().astype(int).to_dict()


def reproduce_baseline(
    clusters: pd.DataFrame, inputs: BaselineInputs
) -> depth_eval.BaselineDraws:
    """J1-2 보고 계산 2 (가): the main activity-range draws, same rng and order."""
    counts = kept_estimate_counts(clusters)
    n_est = {activity: counts.get(activity, 0) for activity in inputs.gt_depths}
    return depth_eval.draw_baseline(
        n_est,
        inputs.gt_depths,
        depth_eval.BASELINE_TRIALS,
        depth_eval.BASELINE_SEED,
        "activity",
    )


def weighted_columns(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Columns with a positive weight (in draw order) and their weights."""
    columns = np.flatnonzero(weights > 0)
    return columns, weights[columns]


def weighted_pass_b(draws: depth_eval.BaselineDraws, weights: np.ndarray) -> np.ndarray:
    """Per-trial baseline pass B with activity weights ``w_a``."""
    columns, column_weights = weighted_columns(weights)
    n_pass = (draws.n_pass[:, columns] * column_weights).sum(axis=1)
    total_gt = int((draws.n_gt[columns] * column_weights).sum())
    return depth_eval._ratio_per_trial(n_pass, np.full(n_pass.shape, total_gt))


def weighted_baseline_stats(
    draws: depth_eval.BaselineDraws, weights: np.ndarray
) -> depth_eval.BaselineStats | None:
    """``baseline_stats`` with activity weights (J1-2 보고 계산 6).

    Columns with weight 0 are left out and the others are multiplied by their
    integer weight, so ``w = 1`` reproduces ``baseline_stats`` bit for bit.
    """
    columns, column_weights = weighted_columns(weights)
    if columns.size == 0:
        return None
    n_pass = (draws.n_pass[:, columns] * column_weights).sum(axis=1)
    n_pairs = (draws.n_pairs[:, columns] * column_weights).sum(axis=1)
    total_gt = int((draws.n_gt[columns] * column_weights).sum())
    error_sum = (draws.abs_error_sum[:, columns] * column_weights).sum(axis=1)
    maes = depth_eval._ratio_per_trial(error_sum, n_pairs)
    pass_a = depth_eval._ratio_per_trial(n_pass, n_pairs)
    pass_b = depth_eval._ratio_per_trial(n_pass, np.full(n_pass.shape, total_gt))
    return depth_eval.BaselineStats(
        mae_mean=float(np.nanmean(maes)),
        mae_p025=float(np.nanpercentile(maes, 2.5)),
        mae_p975=float(np.nanpercentile(maes, 97.5)),
        pass_a_mean=float(np.nanmean(pass_a)),
        pass_b_mean=float(np.nanmean(pass_b)),
        pass_b_p025=float(np.nanpercentile(pass_b, 2.5)),
        pass_b_p975=float(np.nanpercentile(pass_b, 97.5)),
        pass_a_p025=float(np.nanpercentile(pass_a, 2.5)),
        pass_a_p975=float(np.nanpercentile(pass_a, 97.5)),
        pass_b_values=pass_b,
        gate_pairs_values=(draws.n_gate_pairs[:, columns] * column_weights).sum(axis=1),
        n_est_total=int((draws.n_est[columns] * column_weights).sum()),
        n_gt_total=total_gt,
    )


@dataclass(frozen=True)
class SiteTable:
    """Observed and baseline counts arranged by site (sites ascending)."""

    sites: tuple[int, ...]
    activity_site_index: np.ndarray
    """Site index of every draw column (``draws.activities`` order)."""
    n_pass_by_site: np.ndarray
    n_gt_by_site: np.ndarray


def observed_pass_by_activity(matches: pd.DataFrame) -> dict[str, int]:
    rows = main_rows(matches)
    passing = rows[rows["within_30cm"] == "True"]
    return passing.groupby("activity_id").size().astype(int).to_dict()


def site_table(
    draws: depth_eval.BaselineDraws, n_pass_by_activity: Mapping[str, int]
) -> SiteTable:
    activity_sites = [site_of_activity(activity) for activity in draws.activities]
    sites = tuple(sorted(set(activity_sites)))
    index = np.array([sites.index(site) for site in activity_sites], dtype=int)
    n_pass = np.array(
        [n_pass_by_activity.get(a, 0) for a in draws.activities], dtype=int
    )
    return SiteTable(
        sites=sites,
        activity_site_index=index,
        n_pass_by_site=np.bincount(index, weights=n_pass, minlength=len(sites)).astype(
            int
        ),
        n_gt_by_site=np.bincount(
            index, weights=draws.n_gt, minlength=len(sites)
        ).astype(int),
    )


def _rounded_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, SUMMARY_DECIMALS)


def _scope_reproduction(
    draws: depth_eval.BaselineDraws, weights: np.ndarray, n_pass: int
) -> dict[str, float | None]:
    stats = weighted_baseline_stats(draws, weights)
    if stats is None:
        return dict.fromkeys(BASELINE_COLUMNS)
    total_gt = stats.n_gt_total
    observed_b = n_pass / total_gt if total_gt else None
    p_value = (
        None
        if observed_b is None
        else float(np.mean(stats.pass_b_values >= observed_b))
    )
    values = (stats.pass_b_mean, stats.pass_b_p025, stats.pass_b_p975, p_value)
    return {
        column: _rounded_or_none(v)
        for column, v in zip(BASELINE_COLUMNS, values, strict=True)
    }


def reproduction_mismatches(
    draws: depth_eval.BaselineDraws, summary: pd.DataFrame, sites: SiteTable
) -> list[str]:
    """J1-2 보고 계산 2: every main scope row equals the reproduced baseline."""
    mismatches = []
    for row in main_rows(summary).to_dict("records"):
        scope = row["scope"]
        if scope == SCOPE_ALL:
            weights = np.ones(len(draws.activities), dtype=int)
            n_pass = int(sites.n_pass_by_site.sum())
        else:
            site_index = sites.sites.index(int(scope.removeprefix(SITE_SCOPE_PREFIX)))
            weights = (sites.activity_site_index == site_index).astype(int)
            n_pass = int(sites.n_pass_by_site[site_index])
        reproduced = _scope_reproduction(draws, weights, n_pass)
        for column, value in reproduced.items():
            if _number(row[column]) != value:
                mismatches.append(f"{scope}.{column}")
    return mismatches


def reachability(draws: depth_eval.BaselineDraws) -> dict:
    """§2-4 (i)-(iii): ``k_perfect``, ``n_req``, ``A_req`` and ``E_+``."""
    totals = draws.n_pass.sum(axis=1)
    n_pairs = int(np.minimum(draws.n_est, draws.n_gt).sum())
    k_perfect = int(np.sum(totals == n_pairs))
    t_rank = int(np.sort(totals)[::-1][N_REQ_RANK - 1])
    n_req = t_rank + 1
    return {
        "n_pairs": n_pairs,
        "trial_pairs_constant": bool(np.all(draws.n_pairs.sum(axis=1) == n_pairs)),
        "k_perfect": k_perfect,
        "perfect_observation_cannot_reach_p": k_perfect >= N_REQ_RANK,
        "t_50": t_rank,
        "n_req": n_req,
        "a_req": n_req / n_pairs if n_pairs else None,
        "n_req_exceeds_n_pairs": n_req > n_pairs,
        "excess_estimates": int(np.maximum(draws.n_est - draws.n_gt, 0).sum()),
        "n_activities_more_estimates_than_gt": int(np.sum(draws.n_est > draws.n_gt)),
        "n_single_gt_activities_with_estimate": int(
            np.sum((draws.n_gt == 1) & (draws.n_est >= 1))
        ),
    }


def leave_one_site_out(draws: depth_eval.BaselineDraws, sites: SiteTable) -> list[dict]:
    """§2-4 현장 하나씩 빼기: ``B_-s``, ``Delta_B,-s`` and ``k_-s`` per site."""
    n_pass_total, n_gt_total = (
        int(sites.n_pass_by_site.sum()),
        int(sites.n_gt_by_site.sum()),
    )
    rows = []
    for site_index, site in enumerate(sites.sites):
        weights = (sites.activity_site_index != site_index).astype(int)
        n_pass = n_pass_total - int(sites.n_pass_by_site[site_index])
        n_gt = n_gt_total - int(sites.n_gt_by_site[site_index])
        baseline_b = weighted_pass_b(draws, weights)
        trial_pass = (draws.n_pass * weights).sum(axis=1)
        b_without = n_pass / n_gt if n_gt else None
        baseline_mean = float(np.nanmean(baseline_b))
        rows.append(
            {
                "site": site,
                "n_pass": n_pass,
                "n_gt": n_gt,
                "pass_b": b_without,
                "baseline_pass_b_mean": baseline_mean,
                "delta_b": None if b_without is None else b_without - baseline_mean,
                "k": int(np.sum(trial_pass >= n_pass)),
            }
        )
    return rows


def site_bootstrap(
    draws: depth_eval.BaselineDraws,
    sites: SiteTable,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """§2-4 현장 부트스트랩: resample sites, re-weight the same baseline trials.

    Each replicate draws ``|S|`` site indices with ``rng.integers(0, |S|)``
    (sites ascending) and counts how often each site was drawn (``w_s``).
    """
    rng = np.random.default_rng(seed)
    n_sites = len(sites.sites)
    b_star = np.empty(replicates)
    delta_star = np.empty(replicates)
    for replicate in range(replicates):
        site_weights = np.bincount(
            rng.integers(0, n_sites, size=n_sites), minlength=n_sites
        )
        b_star[replicate] = (site_weights @ sites.n_pass_by_site) / (
            site_weights @ sites.n_gt_by_site
        )
        activity_weights = site_weights[sites.activity_site_index]
        baseline_mean = float(np.nanmean(weighted_pass_b(draws, activity_weights)))
        delta_star[replicate] = b_star[replicate] - baseline_mean
    return {
        "seed": seed,
        "replicates": replicates,
        "n_sites": n_sites,
        "pass_b_p025": float(np.percentile(b_star, 2.5)),
        "pass_b_p975": float(np.percentile(b_star, 97.5)),
        "delta_b_p025": float(np.percentile(delta_star, 2.5)),
        "delta_b_p975": float(np.percentile(delta_star, 97.5)),
    }


@dataclass(frozen=True)
class ReportResult:
    """J1-2 보고 계산 2-5, or ``None`` items with the reason."""

    items: dict | None
    failure: str | None


def compute_report(
    outputs: EvalOutputs, result: MainResult, inputs: BaselineInputs
) -> ReportResult:
    """J1-2 보고 계산 2-5 on the trials of the depth run (no new baseline draw)."""
    draws = reproduce_baseline(outputs.clusters, inputs)
    sites = site_table(draws, observed_pass_by_activity(outputs.matches))
    mismatches = reproduction_mismatches(draws, outputs.summary, sites)
    if mismatches:
        return ReportResult(None, "baseline not reproduced: " + ", ".join(mismatches))
    reach = reachability(draws)
    if reach["excess_estimates"] != int(result.row["false_est"]):
        return ReportResult(None, "E_+ differs from the summary false_est")
    if reach["n_pairs"] != result.n_pairs or not reach["trial_pairs_constant"]:
        return ReportResult(
            None, "baseline pair count differs from the summary n_pairs"
        )
    items = {
        "sites": list(sites.sites),
        "reachability": reach,
        "leave_one_site_out": leave_one_site_out(draws, sites),
        "site_bootstrap": site_bootstrap(draws, sites),
        "note": SITE_CLUSTER_NOTE,
    }
    return ReportResult(items, None)


# ================================================================ records


def candidate_diagnostics(candidates: pd.DataFrame) -> dict:
    """§4 5: zero failures and QC exclusion counts of the depth run's rows."""

    def counts(column: str) -> dict[str, int]:
        if column not in candidates:
            return {}
        return {str(k): int(v) for k, v in candidates[column].value_counts().items()}

    return {
        "status": counts("status"),
        "qc_verdict": counts("qc_verdict"),
        "apex_at_record_end": counts("apex_at_record_end"),
        "apex_on_trace_seam": counts("apex_on_trace_seam"),
        "qc_excluded": counts("qc_excluded"),
    }


def r2_report_values(inputs: RunInputs) -> dict:
    """R2 values the J1 report repeats (§4 5): box counts and PP tallies."""
    fold = inputs.spec.fold
    entry = _fold_entry(inputs.summary, fold) if fold is not None else {}
    keys = (
        "n_boxes_post",
        "n_boxes_pre",
        "n_padding_only_dropped",
        "n_padding_only_dropped_pre",
    )
    return {
        "fold": fold,
        "j1_file": {key: entry.get(key) for key in keys},
        "pp_actions": inputs.summary.get("pp_actions"),
        "pp_a_top_in_content": inputs.summary.get("pp_a_top_in_content"),
    }


@dataclass(frozen=True)
class Clock:
    """Wall-clock start (for the record) and a monotonic start (for seconds)."""

    started_at: str
    started: float

    @classmethod
    def start(cls) -> "Clock":
        return cls(common.local_now(), time.perf_counter())

    def elapsed_s(self) -> float:
        return round(time.perf_counter() - self.started, 1)


def run_record(clock: Clock, inputs: RunInputs, argv: Sequence[str]) -> dict:
    return {
        "started_at": clock.started_at,
        "finished_at": common.local_now(),
        "elapsed_s": clock.elapsed_s(),
        "argv": list(argv),
        "head": inputs.tree.get("head"),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
    }


def _project_relative(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    if resolved.is_relative_to(REPO.resolve()):
        return resolved.relative_to(REPO.resolve()).as_posix()
    return resolved.as_posix()


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not json serialisable: {type(value).__name__}")


def write_json(path: Path, document: Mapping) -> None:
    """UTF-8 json with LF line ends (numpy scalars as python numbers)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=1, ensure_ascii=False, default=_json_default)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def base_document(
    spec: RunSpec,
    inputs: RunInputs,
    record: CandidateRecord,
    checks: Checks,
    pp_config_sha: str,
) -> dict:
    """Identity part of the verdict / report json (filled before any value)."""
    return {
        "schema": SCHEMA,
        "status": None,
        "kind": "verdict" if spec.is_verdict else f"report:{spec.report}",
        "prefix": spec.prefix,
        "inputs": {
            "boxes": _project_relative(inputs.boxes_path),
            "boxes_sha256": inputs.boxes_sha256,
            "r2_summary": _project_relative(inputs.r2_root / "summary.json"),
            "pp_config_sha256": pp_config_sha,
            "candidates_csv": _project_relative(record.path),
            "candidates_sha256": record.sha256,
        },
        "candidate_counts": record.counts,
        "r2": r2_report_values(inputs) if spec.uses_boxes else None,
        "assertions": checks.records,
        "tree": inputs.tree,
    }


# =============================================================== markdown


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _md_table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(cell) for cell in row) + " |" for row in rows]
    return [*lines, ""]


def _md_header(document: Mapping) -> list[str]:
    inputs, run = document["inputs"], document["run"]
    return [
        f"- 생성: `analysis/processing/m33_j1_judge.py`, {run['started_at']}, "
        f"HEAD {run['head']}, {run['elapsed_s']} s",
        f"- 박스 입력: `{inputs['boxes']}` (sha256 {inputs['boxes_sha256']})",
        f"- 후보 csv: `{inputs['candidates_csv']}` "
        f"(sha256 {inputs['candidates_sha256']})",
        "- 깊이 계산: `scripts/eval_depth_4tu.py` (무수정), "
        f"prefix `{document['prefix']}`",
        f"- 상태: `{document['status']}`",
        "",
    ]


def _md_verdict(verdict: Mapping) -> list[str]:
    return ["## 판정", "", verdict["phrase"], ""]


def _md_main_values(values: Mapping) -> list[str]:
    rows = [
        ("합격 짝 n_pass / 388", f"{values['n_pass']} / {values['n_gt']}"),
        ("판정률 B", values["pass_b"]),
        ("k (기준선 시행 1,000회 중 관측 이상)", values["k"]),
        ("p = k / 1000", values["p"]),
        ("짝 수 n_pairs", values["n_pairs"]),
        ("판정률 A", values["pass_a"]),
        ("B_max = n_pairs / 388", values["b_max"]),
        ("기준선 판정률 A 평균", values["baseline_pass_a_mean"]),
        ("MAE (m)", values["mae_m"]),
        ("bias (m)", values["bias_m"]),
        ("누락 / 오검출", f"{values['missed']} / {values['false_est']}"),
    ]
    return ["## 함께 적는 값 (판정에 쓰지 않음)", "", *_md_table(("항목", "값"), rows)]


def _md_substantive(values: Mapping) -> list[str]:
    rows = [
        ("B", values["pass_b"]),
        ("기준선 B 평균", values["baseline_pass_b_mean"]),
        ("기준선 B 2.5 %", values["baseline_pass_b_p025"]),
        ("기준선 B 97.5 %", values["baseline_pass_b_p975"]),
        ("Δ_B = B − 기준선 평균", values["delta_b"]),
    ]
    return ["## 실질 지표 (보고, 판정 아님)", "", *_md_table(("항목", "값"), rows)]


def _md_reachability(reach: Mapping) -> list[str]:
    rows = [
        ("k_perfect", reach["k_perfect"]),
        ("관측이 완벽해도 p < 0.05 불가", reach["perfect_observation_cannot_reach_p"]),
        ("n_req = T_(50) + 1", reach["n_req"]),
        ("A_req = n_req / n_pairs", reach["a_req"]),
        ("초과 추정 E_+ (= summary false_est)", reach["excess_estimates"]),
        ("n_e,a > n_g,a 활동 수", reach["n_activities_more_estimates_than_gt"]),
        (
            "정답 1개이면서 추정 있는 활동 수",
            reach["n_single_gt_activities_with_estimate"],
        ),
    ]
    return ["### p 도달 가능성", "", *_md_table(("항목", "값"), rows)]


def _md_sites(items: Mapping) -> list[str]:
    header = ("빠진 현장", "n_pass", "N_g", "B_-s", "기준선 평균", "Δ_B,-s", "k_-s")
    rows = [
        (
            r["site"],
            r["n_pass"],
            r["n_gt"],
            r["pass_b"],
            r["baseline_pass_b_mean"],
            r["delta_b"],
            r["k"],
        )
        for r in items["leave_one_site_out"]
    ]
    boot = items["site_bootstrap"]
    boot_line = (
        f"- 현장 부트스트랩(`default_rng({boot['seed']})`, {boot['replicates']}회, "
        f"현장 {boot['n_sites']}곳): B* 2.5–97.5 % {_fmt(boot['pass_b_p025'])}–"
        f"{_fmt(boot['pass_b_p975'])}, Δ_B* 2.5–97.5 % {_fmt(boot['delta_b_p025'])}–"
        f"{_fmt(boot['delta_b_p975'])}"
    )
    return [
        "### 현장 단위 불확도",
        "",
        *_md_table(header, rows),
        boot_line,
        "",
        f"- {items['note']}",
        "",
    ]


def _md_report(document: Mapping) -> list[str]:
    lines = ["## 보고 항목 (J1-2 보고 계산, 판정 아님)", ""]
    if document["report"] is None:
        return [*lines, f"- 계산하지 못함: {document['report_failure']}", ""]
    items = document["report"]
    return [*lines, *_md_reachability(items["reachability"]), *_md_sites(items)]


def _md_counts(document: Mapping) -> list[str]:
    lines = ["## 후보 수", ""]
    lines += [
        f"- {key}: {value}" for key, value in document["candidate_counts"].items()
    ]
    if document.get("r2") is not None:
        lines += [
            f"- R2 J1 파일(fold {document['r2']['fold']}): {document['r2']['j1_file']}"
        ]
    lines += [f"- 깊이 계산 후보 진단: {document['depth_run_diagnostics']}", ""]
    return lines


def _md_assertions(records: Sequence[Mapping]) -> list[str]:
    rows = [
        (r["name"], "통과" if r["passed"] else "실패", r["detail"]) for r in records
    ]
    return ["## 단언", "", *_md_table(("단언", "결과", "비고"), rows)]


def markdown_document(document: Mapping) -> str:
    """The ``.md`` twin of the verdict / report json (values from the json)."""
    if document["kind"] == "verdict":
        title = (
            "# M3-3 J1 판정 — 4TU 종단 깊이 (검출기 N2-a′ fold 0, PP(C0b) 후 박스 후보)"
        )
    else:
        title = f"# M3-3 J1 보고 실행 `{document['kind']}` (판정 아님)"
    lines = [title, "", *_md_header(document)]
    if document.get("verdict"):
        lines += _md_verdict(document["verdict"])
    if document.get("main_result"):
        lines += _md_main_values(document["main_result"])
        lines += _md_substantive(document["substantive"])
        lines += _md_report(document)
    lines += _md_counts(document) + _md_assertions(document["assertions"])
    return "\n".join(lines)


def write_result(spec: RunSpec, document: Mapping) -> Path:
    """``<prefix>_verdict.json`` / ``_report.json`` and the ``.md`` beside it."""
    write_json(spec.result_json, document)
    markdown = markdown_document(document)
    spec.result_json.with_suffix(".md").write_text(
        markdown, encoding="utf-8", newline="\n"
    )
    return spec.result_json


# =================================================================== run


@dataclass(frozen=True)
class Outcome:
    """What the last stdout line reports."""

    status: str
    out: Path | None
    n_candidates: int | None


def print_phase(name: str) -> None:
    """Progress on stdout: phase names only (blind)."""
    print(f"phase {name}", flush=True)


def judged_sections(spec: RunSpec, result: MainResult) -> dict:
    """Verdict phrase (verdict run only), accompanying values, ``Delta_B``."""
    sections = {
        "main_result": accompanying_values(result),
        "substantive": substantive_metric(result),
    }
    if spec.is_verdict:
        case = decide_case(result.n_pass, result.k)
        sections["verdict"] = {
            "case": case.value,
            "phrase": verdict_phrase(case, result),
            "markers": border_markers(result.n_pass, result.k),
        }
    return sections


def evaluate_outputs(spec: RunSpec, checks: Checks, document: dict) -> str:
    """Read the depth outputs, judge and report into ``document``; the status."""
    document["eval_outputs_sha256"] = eval_output_sha256(spec.prefix)
    outputs = read_eval_outputs(spec.prefix)
    document["depth_run_diagnostics"] = candidate_diagnostics(outputs.candidates)
    result = read_main_result(outputs, checks)
    if result is None:
        document["report"], document["report_failure"] = None, "§1-1 3 assertion failed"
        return STATUS_INPUT_ERROR
    document.update(judged_sections(spec, result))
    report = compute_report(outputs, result, load_baseline_inputs())
    document["report"], document["report_failure"] = report.items, report.failure
    return STATUS_OK if report.failure is None else STATUS_REPORT_FAILED


def run_judgment(
    args: argparse.Namespace, spec: RunSpec, checks: Checks, clock: Clock
) -> Outcome:
    """Assertions, candidates, depth run, verdict / report files (one run)."""
    print_phase("assertions")
    inputs = assert_run_inputs(checks, spec, args)
    print_phase("candidates")
    record = write_candidates(inputs) if spec.uses_boxes else signal_candidates()
    recheck_candidates(checks, record)
    print_phase("depth_eval")
    run_depth_eval(record.path, spec.prefix)
    print_phase("verdict" if spec.is_verdict else "report")
    document = base_document(spec, inputs, record, checks, args.pp_config_sha256)
    document["status"] = evaluate_outputs(spec, checks, document)
    document["run"] = run_record(clock, inputs, sys.argv)
    out = write_result(spec, document)
    return Outcome(document["status"], out, record.counts.get("n_candidates"))


def record_refusal(spec: RunSpec, checks: Checks, error: Exception) -> Path:
    """Write the failed assertions outside ``docs/`` (the run may be repeated)."""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    path = RUN_LOG_DIR / f"{spec.prefix}_refused_{stamp}.json"
    write_json(
        path,
        {
            "status": STATUS_INPUT_ERROR,
            "error": str(error),
            "assertions": checks.records,
        },
    )
    return path


@contextlib.contextmanager
def judge_log(prefix: str) -> Iterator[Path]:
    """Attach a file handler to this module's logger for one run."""
    path = RUN_LOG_DIR / f"{prefix}_judge.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    saved_propagate = LOGGER.propagate
    LOGGER.addHandler(handler)
    LOGGER.propagate = False
    try:
        yield path
    finally:
        LOGGER.removeHandler(handler)
        LOGGER.propagate = saved_propagate
        handler.close()


def run_guarded(args: argparse.Namespace, clock: Clock) -> Outcome:
    """:func:`run_judgment` with the exit-2 / exit-3 mapping (CLI boundary)."""
    spec = run_spec(args.report)
    checks = Checks()
    with judge_log(spec.prefix) as log_path:
        try:
            return run_judgment(args, spec, checks, clock)
        except InputError as error:
            refusal = record_refusal(spec, checks, error)
            LOGGER.error("input refused: %s", error)
            print(
                f"input refused: {error} (record: {_project_relative(refusal)})",
                file=sys.stderr,
            )
            return Outcome(STATUS_INPUT_ERROR, None, None)
        except Exception as error:
            # CLI boundary: exit 3; the traceback goes to the log file only,
            # because an exception text could carry a judgment value.
            LOGGER.exception("internal error")
            log_name = _project_relative(log_path)
            print(
                f"internal error: {type(error).__name__} (see {log_name})",
                file=sys.stderr,
            )
            return Outcome(STATUS_INTERNAL_ERROR, None, None)


# ============================================================ C2 values


def measure_seal_values(boxes: Path) -> dict:
    """Current blobs and sha for ``m33_j1_seal.py`` (C2); hashes only."""
    sealed = list(seal.SEALED_BLOBS)
    work_blobs = m33_inputs.git_blobs(REPO / name for name in sealed)
    summary = read_r2_summary(boxes.resolve().parent.parent)
    fold0 = _fold_entry(summary, 0)
    src_changes = _git(
        "status", "--porcelain", "--untracked-files=no", "--", SRC_PACKAGE_PATH
    )
    return {
        "SEALED_BLOBS (work tree, git hash-object)": work_blobs,
        "SEALED_SHA256_LF": {
            name: common.sha256_file_lf(REPO / name) for name in sealed
        },
        "SRC_TREE_ID (HEAD)": _git("rev-parse", f"HEAD:{SRC_PACKAGE_PATH}"),
        "src_uncommitted_changes": src_changes.splitlines(),
        "FOURTU_META_SHA256": {
            name: common.sha256_file(path) for name, path in fourtu_meta_paths().items()
        },
        "R2_J1_POST_SHA256 (file)": common.sha256_file(
            boxes.parent / "j1_boxes_post.csv"
        ),
        "R2_J1_POST_SHA256 (summary)": fold0.get("post_sha256"),
        "R2_J1_PRE_SHA256 (file)": common.sha256_file(
            boxes.parent / "j1_boxes_pre.csv"
        ),
        "R2_J1_PRE_SHA256 (summary)": fold0.get("pre_sha256"),
    }


# ================================================================== main


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--boxes",
        type=Path,
        default=m33_inputs.R2_OUT_ROOT / "fold0" / "j1_boxes_post.csv",
        help="R2 fold 0 j1_boxes_post.csv (the R2 folder is its grandparent)",
    )
    parser.add_argument(
        "--boxes-sha256", default=None, help="C2 value of the boxes file"
    )
    parser.add_argument("--pp-config-sha256", default=None, help="C0b PPConfig hash")
    parser.add_argument("--prefix", default=VERDICT_PREFIX, help="must be depth_4tu_j1")
    parser.add_argument(
        "--report",
        type=ReportKind,
        choices=list(ReportKind),
        default=None,
        help="J1-3 report run (after the verdict): pre, fold1-fold4 or signal",
    )
    parser.add_argument(
        "--measure-seal", action="store_true", help="print the C2 values only"
    )
    args = parser.parse_args(argv)
    if args.measure_seal:
        return args
    if args.pp_config_sha256 is None:
        parser.error("--pp-config-sha256 is required")
    if args.report is None and args.boxes_sha256 is None:
        parser.error("the judgment run needs --boxes-sha256")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.measure_seal:
        print(json.dumps(measure_seal_values(args.boxes), indent=1, ensure_ascii=False))
        return 0
    clock = Clock.start()
    outcome = run_guarded(args, clock)
    last_line = {
        "status": outcome.status,
        "out": _project_relative(outcome.out),
        "n_candidates": outcome.n_candidates,
        "elapsed_s": clock.elapsed_s(),
    }
    print(json.dumps(last_line, ensure_ascii=False), flush=True)
    return EXIT_CODES[outcome.status]


if __name__ == "__main__":
    sys.exit(main())
