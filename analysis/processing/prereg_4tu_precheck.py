"""Pre-run checks 2-4 of the 4TU re-evaluation pre-registration.

Source of truth: docs/processing/preregistration_4tu_reeval_v1.md (commit
267a7b8, published final version). Every pinned value below is copied from that
document by hand and kept as a constant on purpose: the document is not parsed,
so a later edit of the document cannot move the registered values.

Checks (section 1-5 of the document):
  2. clean work tree, HEAD descends from the registration commit, pinned blob
     ids (section 1-2), src tree id (section 1-2-1 (i) or (ii)) and SHA-256 of
     the four 4TU metadata csv files (section 1-2; hashed only, never parsed).
  3. no docs/processing/depth_4tu_v2* output exists yet.
  4. the evaluation dry-run prints exactly the expected JSON (only totals).
Check 1 (resources) is done by the runner's wait condition, not here.
The gpr-dev package versions (section 1-2) are recorded for information only.

No ground-truth depth file is opened by this script; the dry-run subprocess
reads the ground truth but prints totals only.

Run (exit 0 = all checks pass, 1 = at least one check fails):
  conda run -n gpr-dev python -I analysis/processing/prereg_4tu_precheck.py
Writes data/processing/depth_4tu_v2_precheck.json (git-ignored).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from gpr_detection.io.paths import load_dataset_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORD_PATH = PROJECT_ROOT / "data" / "processing" / "depth_4tu_v2_precheck.json"

# Section 1-2 / 1-5 item 2: commit holding the final pre-registration.
REGISTRATION_COMMIT = "267a7b8325ffef83152319b65beee7e5d80139d1"

# Section 1-2-1: allowed tree ids of src/gpr_detection at the run commit.
SRC_TREE_PATH = "src/gpr_detection"
ALLOWED_SRC_TREES = {
    "93021c9a85571fc84d1458f65ad7590ec103ac1e": "(i) filters.py comment reverted",
    "63b16374034de554aef2afd97451a6b8a028d4f3": "(ii) filters.py comment committed",
}

# Section 1-5 item 3: outputs of the run must not exist yet.
OUTPUT_GLOB_DIR = PROJECT_ROOT / "docs" / "processing"
OUTPUT_GLOB = "depth_4tu_v2*"

# Section 1-3: inputs passed to the dry-run exactly as in the run command line.
EVAL_SCRIPT = "scripts/eval_depth_4tu.py"
DRY_RUN_ARGS = (
    "--candidates",
    "docs/processing/fourtu_apex_candidates_v1.csv",
    "--lines",
    "docs/processing/fourtu_lines_v1.csv",
    "--dry-run",
)
DRY_RUN_TIMEOUT_S = 900
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_THREADING_LAYER": "TBB",
}
DEFAULT_CONDA_EXE = r"C:\ProgramData\miniforge3\Scripts\conda.exe"
CONDA_ENV_NAME = "gpr-dev"

# Section 1-5 item 4: expected dry-run JSON (2026-11-07 judge run).
EXPECTED_DRY_RUN: dict[str, Any] = {
    "candidates": 3615,
    "candidates_by_source": {"auto_v1": 3576, "manual_v0": 39},
    "candidates_without_tau": 0,
    "activities": 125,
    "lines": 759,
    "lines_with_candidates": 745,
    "activities_with_gt": 93,
    "activities_without_gt": 32,
    "gt_items": 388,
    "gt_items_include_x": 389,
    "gt_radius_known": 98,
    "multi_trench_activities": ["011.7"],
    "long_lines": 50,
    "duplicate_within_copies": 5,
    "duplicate_pairs_across": 34,
    "lines_unique": 754,
    "candidates_on_within_copies": 30,
}

# Section 1-2 bullet 3: environment measured on 2026-11-07. A difference is
# reported but is not a reason to withhold the judgement.
EXPECTED_VERSIONS = {
    "python": "3.13.18",
    "numpy": "2.5.4",
    "scipy": "1.18.1",
    "pandas": "3.1.0",
}

# Section 1-2 bullet 2: original-byte SHA-256 of the 4TU metadata csv files,
# relative to the 4TU package root (load_dataset_paths().fourtu).
FOURTU_METADATA_DIR = "01_메타데이터"
FOURTU_METADATA_SHA256 = {
    "utilities_groundtruth.csv": (
        "18e217b1fa890151423e10aa08dc1104c1240d51c4e2828b7b104b5b2777a41e"
    ),
    "utilities_metadata_list.csv": (
        "d7c8ec9e3898caecbc09774d28b08b48c5320074338f29ef475b55ac06a0dc7c"
    ),
    "radargrams_index.csv": (
        "28a550e190cbba31dc8d504d3a29a80779eec1b544ee73df07223e03fcc6b106"
    ),
    "activities_metadata_ko.csv": (
        "91bca3447ed861655ca996b18c42719c81196fb118576d0d20fe628aca3899a9"
    ),
}
HASH_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class PinnedBlob:
    """A tracked file pinned by section 1-2 of the pre-registration.

    Attributes:
        path: Repository-relative path.
        blob_id: Git blob id at bbf484b.
        sha256_lf: SHA-256 of the LF-normalised content, or None when the
            document pins the blob id only.
    """

    path: str
    blob_id: str
    sha256_lf: str | None


# Section 1-2 table (src tree row is handled by ALLOWED_SRC_TREES).
PINNED_BLOBS = (
    PinnedBlob(
        "scripts/eval_depth_4tu.py",
        "4c444153bce56f45feb36368e9682ed41d28f6b7",
        "1eca748bd46775ae661823401a5f4f10a6311e8f7aa42971725d100d372cd272",
    ),
    PinnedBlob(
        "docs/processing/fourtu_apex_candidates_v1.csv",
        "a0e7ff11d1ba9b24e35c7d8ec212f08f06621b4f",
        "b348a2ffe9fe11f139be4fa8512c68a6eba449c37c99ba284f2154144968d43f",
    ),
    PinnedBlob(
        "docs/processing/fourtu_lines_v1.csv",
        "fbc6da72ab89e48b7463dd0becdfe3e38a7b40d7",
        "475f0a71fdd79fcca3fde035bce0876778f4771b61e4dc50195868e085104cc6",
    ),
    PinnedBlob(
        "docs/datasets/fourtu_duplicate_radargrams.csv",
        "b6516bbcf8f98b33256067528d4222d8efd5a0a6",
        "7602d94fbd5824f3536d463e2b248e4ea474770f852519db3ea30097312c488a",
    ),
    PinnedBlob(
        "docs/processing/m29c_qc_constants.json",
        "13f2fd3ac76d6996ac9c590a2cadcd988d34a376",
        "6a88bd8d2501132b0db43eeec2dec9ef347de6e260a9f98aef9a2322ec56ba8c",
    ),
    PinnedBlob(
        "docs/processing/fourtu_apex_candidate_lines_v1.csv",
        "3efec82f574352590731cb18b1c386c220fc6c99",
        None,
    ),
)


@dataclass
class CheckResult:
    """Outcome of one comparison.

    Attributes:
        item: Item number of section 1-5 ("2", "3", "4") or "info".
        name: Short identifier of the comparison.
        passed: True/False, or None for information-only entries.
        expected: Registered value.
        actual: Value observed now.
    """

    item: str
    name: str
    passed: bool | None
    expected: Any
    actual: Any


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _git_output(*args: str) -> str:
    completed = _run_git(*args)
    if completed.returncode != 0:
        return f"<git error {completed.returncode}: {completed.stderr.strip()}>"
    return completed.stdout.strip()


def _sha256_lf_of_head_blob(path: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "cat-file", "blob", f"HEAD:{path}"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return f"<git error {completed.returncode}>"
    return hashlib.sha256(completed.stdout.replace(b"\r\n", b"\n")).hexdigest()


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def check_git_state() -> list[CheckResult]:
    """Item 2: clean tree, ancestry, pinned blob ids and the src tree id."""
    porcelain = _git_output("status", "--porcelain")
    status_lines = porcelain.splitlines() if porcelain else []
    is_ancestor = _run_git(
        "merge-base", "--is-ancestor", REGISTRATION_COMMIT, "HEAD"
    ).returncode
    results = [
        CheckResult("2", "git_status_clean", not status_lines, [], status_lines),
        CheckResult(
            "2",
            "head_descends_from_registration",
            is_ancestor == 0,
            REGISTRATION_COMMIT,
            {"head": _git_output("rev-parse", "HEAD"), "is_ancestor_exit": is_ancestor},
        ),
    ]
    for pinned in PINNED_BLOBS:
        blob_id = _git_output("rev-parse", f"HEAD:{pinned.path}")
        results.append(
            CheckResult(
                "2",
                f"blob:{pinned.path}",
                blob_id == pinned.blob_id,
                pinned.blob_id,
                blob_id,
            )
        )
        if pinned.sha256_lf is None:
            continue
        sha256_lf = _sha256_lf_of_head_blob(pinned.path)
        results.append(
            CheckResult(
                "2",
                f"sha256_lf:{pinned.path}",
                sha256_lf == pinned.sha256_lf,
                pinned.sha256_lf,
                sha256_lf,
            )
        )
    src_tree = _git_output("rev-parse", f"HEAD:{SRC_TREE_PATH}")
    results.append(
        CheckResult(
            "2",
            f"tree:{SRC_TREE_PATH}",
            src_tree in ALLOWED_SRC_TREES,
            ALLOWED_SRC_TREES,
            {"tree": src_tree, "variant": ALLOWED_SRC_TREES.get(src_tree)},
        )
    )
    return results


def check_fourtu_metadata() -> list[CheckResult]:
    """Item 2: SHA-256 of the 4TU metadata csv files (bytes only, not parsed)."""
    metadata_dir = load_dataset_paths().fourtu / FOURTU_METADATA_DIR
    results = []
    for file_name, expected_sha256 in FOURTU_METADATA_SHA256.items():
        path = metadata_dir / file_name
        actual = _sha256_of_file(path) if path.is_file() else f"<missing: {path}>"
        results.append(
            CheckResult(
                "2",
                f"fourtu_sha256:{file_name}",
                actual == expected_sha256,
                expected_sha256,
                actual,
            )
        )
    return results


def check_no_outputs() -> list[CheckResult]:
    """Item 3: the run must not overwrite existing depth_4tu_v2 outputs."""
    existing = sorted(p.name for p in OUTPUT_GLOB_DIR.glob(OUTPUT_GLOB))
    return [
        CheckResult(
            "3", f"no_docs/processing/{OUTPUT_GLOB}", not existing, [], existing
        )
    ]


def _conda_command(*python_args: str) -> list[str]:
    conda_exe = os.environ.get("CONDA_EXE", DEFAULT_CONDA_EXE)
    return [
        conda_exe,
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV_NAME,
        "python",
        "-I",
        *python_args,
    ]


def _run_in_gpr_dev(*python_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _conda_command(*python_args),
        cwd=PROJECT_ROOT,
        env={**os.environ, **SINGLE_THREAD_ENV},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=DRY_RUN_TIMEOUT_S,
        check=False,
    )


def _last_json_line(stdout: str) -> dict[str, Any] | None:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def check_dry_run() -> list[CheckResult]:
    """Item 4: dry-run JSON equals the registered one, keys and values."""
    completed = _run_in_gpr_dev(EVAL_SCRIPT, *DRY_RUN_ARGS)
    report = _last_json_line(completed.stdout)
    actual: Any = report
    if report is None:
        actual = {
            "exit": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    passed = completed.returncode == 0 and report == EXPECTED_DRY_RUN
    results = [CheckResult("4", "dry_run_json", passed, EXPECTED_DRY_RUN, actual)]
    if report is not None and not passed:
        results.append(
            CheckResult("4", "dry_run_diff", False, None, _dict_diff(report))
        )
    return results


def _dict_diff(report: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(report) | set(EXPECTED_DRY_RUN))
    return {
        key: {"expected": EXPECTED_DRY_RUN.get(key), "actual": report.get(key)}
        for key in keys
        if report.get(key) != EXPECTED_DRY_RUN.get(key)
    }


def record_versions() -> list[CheckResult]:
    """Information only: gpr-dev package versions against section 1-2."""
    probe = (
        "import json, platform, numpy, scipy, pandas; print(json.dumps({"
        "'python': platform.python_version(), 'numpy': numpy.__version__, "
        "'scipy': scipy.__version__, 'pandas': pandas.__version__}))"
    )
    completed = _run_in_gpr_dev("-c", probe)
    versions = _last_json_line(completed.stdout) or {"error": completed.stderr[-500:]}
    return [CheckResult("info", "gpr_dev_versions", None, EXPECTED_VERSIONS, versions)]


def _print_result(result: CheckResult) -> None:
    label = {True: "PASS", False: "FAIL", None: "INFO"}[result.passed]
    print(f"[{label}] item {result.item} {result.name}")
    if result.passed is not True:
        print(f"       expected: {json.dumps(result.expected, ensure_ascii=True)}")
        print(f"       actual:   {json.dumps(result.actual, ensure_ascii=True)}")


def write_record(results: list[CheckResult], all_passed: bool) -> None:
    """Store the full comparison as the run record required by item 2."""
    record = {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "preregistration": "docs/processing/preregistration_4tu_reeval_v1.md",
        "all_passed": all_passed,
        "results": [asdict(result) for result in results],
    }
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    results = [
        *check_git_state(),
        *check_fourtu_metadata(),
        *check_no_outputs(),
        *check_dry_run(),
        *record_versions(),
    ]
    all_passed = all(result.passed is not False for result in results)
    for result in results:
        _print_result(result)
    write_record(results, all_passed)
    failed = [r.name for r in results if r.passed is False]
    print(f"record: {RECORD_PATH}")
    print(json.dumps({"all_passed": all_passed, "failed": failed}, ensure_ascii=True))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
