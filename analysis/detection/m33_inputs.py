"""M3-3 R runners: fixed inputs and the run checks every runner shares.

Spec: ``docs/experiments/m33_runner_spec.md`` v0.3.6 §1 ([RUN] common
requirements) and §3-1 4 (sealed PP set), ``docs/experiments/m33_pp_spec.md``
v0.2.4 appendix A ([PP] full precision, judgeability, end-of-run tree, LF
sha). New M3-3 constants live here; the M3-2 modules keep theirs ([RUN] §0).

* Sealed PP: the C0b ``d0_gate.json`` (sha asserted) names the selected set;
  its ``PPConfig`` hash must equal the C0b value and ``pipeline.py`` must be
  the C0a blob. The rule list is read from the set id, never written here.
* Judgeability: only folds 0-4, every line and a tree clean at the start can
  be judged; any other run is ``not_judgeable`` (exit 1).
* Tree at the start and at the end: HEAD, problems, code blobs and the
  ``src`` tree id; a change means the code that ran is not the code recorded
  (``input_error``, exit 2).
* Text inputs are asserted on their LF sha (raw sha recorded beside it).
"""

import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import m31_d1_common as common
import m32_aihub_perturb as perturb
from m32_recipe_inputs import InputError

from gpr_detection.pipeline import PP_SETS, PPSetId

REPO = common.REPO

EXPECTED_FOLDS = (0, 1, 2, 3, 4)
"""Folds a judgeable run covers ([RUN] §1 판정 가능성)."""

C0B_D0_GATE_JSON = REPO / "runs" / "eval" / "m33_d0" / "d0_gate.json"
C0B_D0_GATE_SHA256 = "107d7b53f30667684baf7c2236bdf338c861309531ff17497e2836193bea08c6"
"""sha256 of the C0b ``d0_gate.json`` (0f11c34); the selection is read from it."""
C0B_PP_SET_ID = PPSetId.S3
"""``selection.pp_set_id`` of C0b (an assertion value, not the rule list)."""
C0B_PP_CONFIG_SHA256 = (
    "eb2796ddfaa9611e5030342c4ed66b00c91ff2ed6a6698ee5979b9afe080602f"
)
"""``PPConfig.sha256()`` of the C0b selected set."""
C0A_PIPELINE_BLOB = "dd5e7bcba4397efd732e7bf3ca1c5c92fb2404a1"
"""``git rev-parse 53c4e16:src/gpr_detection/pipeline.py`` (C0a)."""
PIPELINE_PATH = "src/gpr_detection/pipeline.py"
"""Repo-relative key of the PP implementation in ``code_blobs``."""

R3_OUT_ROOT = REPO / "runs" / "eval" / "m33" / "r3"
R3_SMOKE_OUT_ROOT = REPO / "runs" / "eval" / "m33" / "r3_smoke"
R2_OUT_ROOT = REPO / "runs" / "eval" / "m33" / "r2"
"""R2 4TU un-injected boxes (zone (c') and ghost pairs of the 4TU bands)."""
R3_C1B_RECORD = REPO / "analysis" / "detection" / "m33_r3_c1b.json"
"""Tracked C1b record: the band table, positions and exclusions sha (LF)."""

STATUS_OK = "ok"
STATUS_NOT_JUDGEABLE = "not_judgeable"
STATUS_INPUT_ERROR = "input_error"

BLOCKER_SMOKE = "smoke"
BLOCKER_PARTIAL_FOLDS = "partial_folds"
BLOCKER_LINE_LIMIT = "line_limit"
BLOCKER_PARTIAL_RECIPES = "partial_recipes"
BLOCKER_PROVISIONAL = "provisional"
BLOCKER_DIRTY_TREE = "dirty_tree"
"""``judgment.blockers`` values: why a finished run is not a judgment input."""

TREE_KEYS = ("head", "dirty", "problems", "code_blobs", "src_tree_id")
"""``tree_state`` fields compared at the start and the end of a run."""


# ---------------------------------------------------------------- text sha


def sha256_text_lf(path: Path) -> str:
    """sha256 of a text file with CRLF read as LF (checkout independent)."""
    return common.sha256_file_lf(path)


def text_sha256_record(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    """Raw and LF sha of each text input: ``{"raw": {...}, "lf": {...}}``."""
    return {
        "raw": {name: common.sha256_file(path) for name, path in paths.items()},
        "lf": {name: sha256_text_lf(path) for name, path in paths.items()},
    }


# ------------------------------------------------------------------- folds


def check_folds(folds: Sequence[int]) -> tuple[int, ...]:
    """Fold ids of a run; a repeated fold or one outside 0-4 is refused.

    Raises:
        InputError: For a repeated or unknown fold (exit 2).
    """
    folds = tuple(folds)
    if len(set(folds)) != len(folds):
        raise InputError(f"repeated fold in {list(folds)}")
    unknown = sorted(set(folds) - set(EXPECTED_FOLDS))
    if unknown:
        raise InputError(f"unknown folds {unknown} (expected a subset of 0-4)")
    return folds


def has_all_folds(folds: Iterable[int]) -> bool:
    """The run covers exactly the expected folds 0-4."""
    return tuple(sorted(folds)) == EXPECTED_FOLDS


# -------------------------------------------------------------------- tree


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=common.GIT_TIMEOUT_S,
        check=True,
    )
    return completed.stdout


def git_blobs(paths: Iterable[Path]) -> dict[str, str | None]:
    """``git hash-object`` of each file (repo-relative keys, ``None`` on failure)."""
    blobs: dict[str, str | None] = {}
    for path in paths:
        relative = path.resolve().relative_to(REPO.resolve()).as_posix()
        try:
            blobs[relative] = _git("hash-object", "--", relative).strip()
        except (OSError, subprocess.SubprocessError):
            blobs[relative] = None
    return blobs


def _src_tree_id() -> str | None:
    try:
        return _git("rev-parse", "HEAD:src").strip()
    except (OSError, subprocess.SubprocessError):
        return None


def tree_state(required_paths: Sequence[Path], blob_paths: Sequence[Path]) -> dict:
    """HEAD, tracked-tree problems, code blobs and the ``src`` tree id now.

    Args:
        required_paths: Files git must track unchanged (code and tables).
        blob_paths: Code files whose blobs identify the run.
    """
    problems = perturb.tracked_tree_problems(required_paths, REPO)
    return {
        "head": common.git_state()["head"],
        "dirty": bool(problems),
        "problems": problems,
        "code_blobs": git_blobs(blob_paths),
        "src_tree_id": _src_tree_id(),
    }


def tree_changes(start: Mapping[str, Any], end: Mapping[str, Any]) -> list[str]:
    """``TREE_KEYS`` that differ between two snapshots (an unknown HEAD counts)."""
    changed = [key for key in TREE_KEYS if start.get(key) != end.get(key)]
    if "head" not in changed and start.get("head") is None:
        changed.insert(0, "head")
    return changed


# --------------------------------------------------------------- judgment


def judgment_record(
    folds_run: Sequence[int] | None,
    scope_blockers: Sequence[str],
    start_tree: Mapping[str, Any],
    end_tree: Mapping[str, Any],
) -> dict:
    """``judgment`` block of a result json ([RUN] §1 판정 가능성).

    Blockers are the scope reasons plus a tree dirty at the start; a tree
    that changed during the run is listed apart (``tree_changes``).

    Args:
        folds_run: Folds of the run, ``None`` for a stage without folds
            (plan).
        scope_blockers: Why the run scope is not a full run.
        start_tree: :func:`tree_state` at the start.
        end_tree: :func:`tree_state` right before the results are written.
    """
    blockers = list(scope_blockers)
    if start_tree["dirty"]:
        blockers.append(BLOCKER_DIRTY_TREE)
    changes = tree_changes(start_tree, end_tree)
    return {
        "judgeable": not blockers and not changes,
        "blockers": blockers,
        "expected_folds": list(EXPECTED_FOLDS),
        "folds_run": None if folds_run is None else list(folds_run),
        "tree_stable": not changes,
        "tree_changes": changes,
    }


def final_status(base_status: str, judgment: Mapping[str, Any]) -> str:
    """Status of a finished run from its own status and its judgment.

    A changed tree is ``input_error`` (exit 2) whatever the run found. A
    gate or sample failure keeps its own status (exit 1); a finished run
    with a blocker is ``not_judgeable`` (exit 1).
    """
    if judgment["tree_changes"]:
        return STATUS_INPUT_ERROR
    if base_status != STATUS_OK:
        return base_status
    if judgment["blockers"]:
        return STATUS_NOT_JUDGEABLE
    return STATUS_OK


# --------------------------------------------------------------- sealed PP


def _read_c0b_selection(d0_gate_path: Path, expected_sha256: str) -> dict:
    if not d0_gate_path.is_file():
        raise InputError(f"C0b d0_gate.json missing: {d0_gate_path}")
    digest = common.sha256_file(d0_gate_path)
    if digest != expected_sha256:
        raise InputError(f"C0b d0_gate.json sha256 {digest} != {expected_sha256}")
    document = json.loads(d0_gate_path.read_text(encoding="utf-8"))
    selection = document.get("selection")
    if not isinstance(selection, Mapping):
        raise InputError("C0b d0_gate.json has no selection block")
    return dict(selection)


def assert_sealed_pp(
    code_blobs: Mapping[str, str | None],
    d0_gate_path: Path | None = None,
    expected_d0_gate_sha256: str | None = None,
) -> dict:
    """Assert the PP set and the PP code are the sealed C0b / C0a ones.

    The set id comes from the C0b ``d0_gate.json`` selection; its
    ``PPConfig`` hash must equal both the recorded and the sealed value, the
    id must be the sealed one, and ``pipeline.py`` must be the C0a blob
    ([RUN] §1 D0 실행기 blob 행, §3-1 4).

    Args:
        code_blobs: ``tree_state`` code blobs (``pipeline.py`` included).
        d0_gate_path: C0b ``d0_gate.json`` (default ``C0B_D0_GATE_JSON``).
        expected_d0_gate_sha256: Its sha (default ``C0B_D0_GATE_SHA256``).

    Returns:
        ``pp_set_id``, ``pp_config_sha256``, the rule list and the checks.

    Raises:
        InputError: When any check fails (exit 2, before inference).
    """
    path = C0B_D0_GATE_JSON if d0_gate_path is None else d0_gate_path
    expected_sha = (
        C0B_D0_GATE_SHA256
        if expected_d0_gate_sha256 is None
        else expected_d0_gate_sha256
    )
    selection = _read_c0b_selection(path, expected_sha)
    try:
        set_id = PPSetId(selection.get("pp_set_id"))
    except ValueError as error:
        raise InputError(f"C0b selection names no PP set: {selection}") from error
    config = PP_SETS[set_id]
    measured_sha = config.sha256()
    checks = {
        "set_id_is_sealed": set_id is C0B_PP_SET_ID,
        "config_sha_matches_record": measured_sha == selection.get("pp_config_sha256"),
        "config_sha_is_sealed": measured_sha == C0B_PP_CONFIG_SHA256,
        "pipeline_blob_is_c0a": code_blobs.get(PIPELINE_PATH) == C0A_PIPELINE_BLOB,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise InputError(f"PP differs from the C0b / C0a seal: {failed}")
    return {
        "pp_set_id": set_id.value,
        "pp_config_sha256": measured_sha,
        "rules": [rule.value for rule in config.rules],
        "d0_gate_sha256": expected_sha,
        "pipeline_blob": code_blobs.get(PIPELINE_PATH),
        "checks": checks,
        "result": "pass",
    }
