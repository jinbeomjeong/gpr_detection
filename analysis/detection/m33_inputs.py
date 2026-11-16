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

import hashlib
import json
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import m31_d1_common as common
import m32_aihub_perturb as perturb
import numpy as np
from m32_recipe_inputs import InputError

from gpr_detection.io.fourtu import load_fourtu_radargram
from gpr_detection.io.gprsd import DtHypothesis, load_gprsd_radargram
from gpr_detection.io.guangzhou import load_guangzhou_radargram
from gpr_detection.io.paths import DatasetName, load_dataset_paths
from gpr_detection.io.radargram import Radargram
from gpr_detection.pipeline import DEFAULT_MERGE_CONFIG, PP_SETS, PPSetId
from gpr_detection.preprocess.to_image import (
    DEFAULT_DETECTION_IMAGE_CONFIG,
    M3_2_ASSUMED,
    make_detection_image,
)

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
C1_INJECT_BLOB = "2d88ff0a25c5cbc1bb7ca8698ba422c633f5b531"
"""``git rev-parse 826aa41:src/gpr_detection/eval/inject.py`` (C1, S4)."""
INJECT_PATH = "src/gpr_detection/eval/inject.py"
"""Repo-relative key of the injection code (bands, ``band_conditions``)."""

R3_OUT_ROOT = REPO / "runs" / "eval" / "m33" / "r3"
R3_SMOKE_OUT_ROOT = REPO / "runs" / "eval" / "m33" / "r3_smoke"
R2_OUT_ROOT = REPO / "runs" / "eval" / "m33" / "r2"
"""R2 4TU un-injected boxes (zone (c') and ghost pairs of the 4TU bands)."""
R3_C1B_RECORD = REPO / "analysis" / "detection" / "m33_r3_c1b.json"
"""Tracked C1b record: the band table, positions and exclusions sha (LF)."""

R2_OUT_ROOT_SMOKE = REPO / "runs" / "eval" / "m33" / "r2_smoke"
R2_LINES_CSV = REPO / "analysis" / "detection" / "m33_r2_lines.csv"
R2_LINES_SHA256_LF = "57de56c64d488eab0bc2316f816177317958a40060c47227787523d0f97c4b8d"
"""LF sha of the R2 754-line list built by ``m33_r2_lines.py`` (2026-11-16)."""
K_APEX_SNR_SHA256_LF = (
    "55d54bb15a84f4acccdd51d4ace82b4fef23265956f7ab3f73b1dea94c29bd16"
)
"""LF sha of ``analysis/processing/m33_k_apex_snr.csv`` (X6, tracked)."""

R1_OUT_ROOT = REPO / "runs" / "eval" / "m33" / "r1"
R1_SMOKE_OUT_ROOT = REPO / "runs" / "eval" / "m33" / "r1_smoke"
MOROCCO_DIR = REPO / "data" / "morocco_eval_v1"
MOROCCO_MANIFEST = REPO / "data" / "splits" / "morocco_eval_v1.csv"
MOROCCO_SHA256_LF = {
    "manifest": "db2ebc301b68aff70cd448c99b4b0338ece3f7d240bd46f20b2a1d7661513fd5",
    "eval_images": "85b1666d18eeca0c9e1b5108e34797e6503bf22431989eae16bde0410c1f0cef",
}
"""LF sha of the Morocco manifest and list (the assertion basis, [RUN] §1).

Both files hold CRLF; their raw sha are the X7 values of
``m33_data_assertions.md`` §1 (``MOROCCO_SHA256_RAW``), checked as well.
"""
MOROCCO_SHA256_RAW = {
    "manifest": "b140c7b465e8bf4fd0dd1701356862984531176578c1c9482b9caca2c0562327",
    "eval_images": "0e91de3529635f8d1df7903c840c68df63a7cbb2853072fd7b81a1bf484eaa1a",
}
MOROCCO_IMAGES_SHA256 = (
    "95ad76444681d9b82e3a70bf66e48b05e2800df72e3c6643de348c998c448958"
)
"""sha256 of the 233 image sha256 joined by newlines in file-name order (X7)."""
MOROCCO_LABELS_SHA256 = (
    "12b0fbe2f2bf0cc5d191f830e9c10da622543f361f3669043e1e7081b58ef7f3"
)
"""The same over the 233 label files (X7)."""

R0_IDENTITY_LINES = (
    (
        "fourtu",
        "03.9/Path5",
        "a0891dbfd7efda9a862ddc8a3b0a0ebfd898f821750c07ba85e0a68275d94c79",
    ),
    (
        "fourtu",
        "01.1/Path1",
        "10e8bc9ad7c671a766f9797d9a385dbf8b46b04867ba7d79b61d6d600019a62f",
    ),
    (
        "fourtu",
        "09.1/Path2",
        "66863f4caa028cbca8ee93f173cb7cb5692733a38f61cf66d464316f9cc33ffd",
    ),
    (
        "fourtu",
        "09.4/Path1",
        "78fad5249da1ceb1cf83c752eb8d7096f22be009c471ca190819ca15f075b660",
    ),
    (
        "guangzhou",
        "pipe/001_190321AA/LID10001",
        "b1de7022cc398c91bfa3d34c329938868dbc1559fa878eeef75ae12192d13d78",
    ),
    (
        "guangzhou",
        "pipe/1030_190117AF/LID10001",
        "a3a40315e42ac81602759c7490b98b6c0306bbb4461f0cc536c51031bd7699da",
    ),
    (
        "guangzhou",
        "pipe/190321A_190321AH/LID10001",
        "788f8d22fcccb6b26015d4b699739122922afdd7898d94a8cad0639039e22e46",
    ),
    (
        "guangzhou",
        "pipe/MALA_07011/LA010001",
        "2a0707a491e844808cb29bc0450eba977004e27a76d96dc458e5c75a12d98596",
    ),
    (
        "guangzhou",
        "pipe/1030_181224AB/LID10001",
        "d73a8135a5c9c86075b319c3fd711584af08fe9f88e8e38c069fa6f4ab4e049e",
    ),
    (
        "gprsd",
        "CementRoad2_1",
        "292ede9cdffdef2507692a9418aed58107546832c28af417ebae357a6ce9c9e6",
    ),
    (
        "gprsd",
        "AsphaltRoad2_1",
        "30b732908110f45f581673bd0dd6765f5b4ca68c492dee603957bfa8a9b9fb89",
    ),
)
"""[SP] §9 11 lines: ``(dataset, line, r0 line_image sha256)`` ([RUN] §1 r0 정체).

The M3-2 S6 (a) values (recorded before S1 at a1a9895, unchanged since);
``tests/test_detection_image_recipes.py`` reads them from here. GPR-SD
lines load with ``DtHypothesis.A``. Never recomputed: the point is that r0
is frozen.
"""
R_SETTINGS_SHA256 = "b9c5c905c5fec3ea9840e9644b177aeb6689ab4191d1e2e1dfce17e9a4518b6a"
"""``settings_sha256()`` of the C1 code ([RUN] §1 설정 해시, 2026-11-16)."""

STATUS_OK = "ok"
STATUS_NOT_JUDGEABLE = "not_judgeable"
STATUS_INPUT_ERROR = "input_error"

BLOCKER_SMOKE = "smoke"
BLOCKER_PARTIAL_FOLDS = "partial_folds"
BLOCKER_LINE_LIMIT = "line_limit"
BLOCKER_PARTIAL_RECIPES = "partial_recipes"
BLOCKER_PROVISIONAL = "provisional"
BLOCKER_DIRTY_TREE = "dirty_tree"
BLOCKER_NO_FLIP = "no_flip"
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


def assert_inject_blob(code_blobs: Mapping[str, str | None]) -> dict:
    """R2 / R3: ``inject.py`` must be the C1 blob (verif S4, user 2026-11-16).

    Args:
        code_blobs: ``tree_state`` code blobs (``inject.py`` included).

    Raises:
        InputError: For another or an unmeasured blob (exit 2).
    """
    measured = code_blobs.get(INJECT_PATH)
    if measured != C1_INJECT_BLOB:
        raise InputError(f"{INJECT_PATH} blob {measured} != C1 {C1_INJECT_BLOB}")
    return {"path": INJECT_PATH, "blob": measured, "result": "pass"}


# --------------------------------------------------------- settings hash


CONF_OPERATING = 0.25
CONF_AP = 0.001
NMS_IOU = 0.7


def settings_document(
    conf_operating: float = CONF_OPERATING,
    conf_ap: float = CONF_AP,
    nms_iou: float = NMS_IOU,
) -> dict:
    """The settings the R runners share ([RUN] §1 설정), data-free.

    conf 0.25 (operating) and 0.001 (AP), NMS 0.7, ``MergeConfig()`` and the
    r0 ``DetectionImageConfig`` as the image records it (``assumed_params``
    and ``assumed_values``). The runners pass the values they use.
    """
    config = DEFAULT_DETECTION_IMAGE_CONFIG
    return {
        "conf_operating": conf_operating,
        "conf_ap": conf_ap,
        "nms_iou": nms_iou,
        "merge_config": asdict(DEFAULT_MERGE_CONFIG),
        "r0_assumed_params": list(M3_2_ASSUMED),
        "r0_assumed_values": [[name, getattr(config, name)] for name in M3_2_ASSUMED],
    }


def settings_sha256(document: Mapping | None = None) -> str:
    """sha256 of the sorted-key JSON of the document (enums as ``str``)."""
    document = settings_document() if document is None else document
    text = json.dumps(document, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_settings(conf_operating: float, conf_ap: float, nms_iou: float) -> dict:
    """The settings a runner uses must be the C1 ones.

    Raises:
        InputError: When their sha differs from ``R_SETTINGS_SHA256`` (exit 2).
    """
    document = settings_document(conf_operating, conf_ap, nms_iou)
    measured = settings_sha256(document)
    if measured != R_SETTINGS_SHA256:
        raise InputError(f"settings sha256 {measured} != C1 {R_SETTINGS_SHA256}")
    return {"settings_sha256": measured, "document": document}


# --------------------------------------------------------------- r0 frozen


def _load_identity_line(dataset: str, line: str, paths) -> Radargram:
    root = paths.root(DatasetName(dataset))
    if dataset == DatasetName.FOURTU:
        activity_id, path = line.split("/", 1)
        return load_fourtu_radargram(root, activity_id, path)
    if dataset == DatasetName.GUANGZHOU:
        return load_guangzhou_radargram(root, line)
    return load_gprsd_radargram(root, line, DtHypothesis.A)


def line_image_sha256(radargram: Radargram) -> str:
    """sha256 of the default (r0) ``line_image`` bytes."""
    image = make_detection_image(radargram)
    return hashlib.sha256(np.ascontiguousarray(image.line_image).tobytes()).hexdigest()


def assert_r0_identity(loader: Callable | None = None) -> dict[str, str]:
    """Rebuild the 11 [SP] §9 ``line_image`` and compare with the S6 (a) sha.

    Args:
        loader: ``(dataset, line) -> Radargram`` (default: the read-only
            dataset packages of ``load_dataset_paths``).

    Raises:
        InputError: For a line that cannot be read or another sha (exit 2).
    """
    if loader is None:
        paths = load_dataset_paths()

        def loader(dataset: str, line: str) -> Radargram:
            return _load_identity_line(dataset, line, paths)

    measured = {}
    for dataset, line, expected in R0_IDENTITY_LINES:
        try:
            digest = line_image_sha256(loader(dataset, line))
        except (OSError, ValueError) as error:
            raise InputError(f"r0 identity line {dataset}/{line}: {error}") from error
        if digest != expected:
            raise InputError(f"r0 line_image of {dataset}/{line} changed: {digest}")
        measured[f"{dataset}/{line}"] = digest
    return measured
