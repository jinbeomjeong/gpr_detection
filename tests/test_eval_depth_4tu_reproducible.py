"""Run-to-run reproducibility of ``scripts/eval_depth_4tu.py`` on real 4TU lines.

Without MKL conditional numerical reproducibility the depths change in their
last bits between processes, and the exact-tie matchings of the auxiliary
variants (b, bd, noqc) of 011.2 and 04.6 then pick other pairs (M2-10,
2026-11-11). The script sets ``MKL_CBWR=COMPATIBLE`` itself before numpy loads;
two fresh processes must write byte-identical matches.

Opt-in and slow (two runs of about 15 s): ``pytest -m data
tests/test_eval_depth_4tu_reproducible.py``. Skipped without the 4TU data.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpr_detection.io.paths import DatasetName, load_dataset_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "eval_depth_4tu.py"
TIE_ACTIVITIES = "011.2,04.6"
"""Activities whose auxiliary matchings changed between processes."""
RUN_TIMEOUT_S = 600


def _require_opt_in(request: pytest.FixtureRequest) -> None:
    if "data" not in (request.config.getoption("markexpr") or ""):
        pytest.skip("opt-in: run with -m data")


def _fourtu_root() -> Path:
    root = load_dataset_paths().root(DatasetName.FOURTU)
    if not root.is_dir():
        pytest.skip(f"4TU data not found at {root}")
    return root


def _run(root: Path, out_dir: Path) -> None:
    """One evaluation in a fresh process, MKL_CBWR left to the script."""
    env = {k: v for k, v in os.environ.items() if k != "MKL_CBWR"}
    command = [
        sys.executable,
        "-I",
        str(SCRIPT),
        "--root", str(root),
        "--activities", TIE_ACTIVITIES,
        "--out-dir", str(out_dir),
        "--prefix", "t",
    ]  # fmt: skip
    subprocess.run(command, env=env, check=True, timeout=RUN_TIMEOUT_S)


@pytest.mark.data
def test_two_processes_write_identical_matches(request, tmp_path) -> None:
    _require_opt_in(request)
    root = _fourtu_root()
    first, second = tmp_path / "first", tmp_path / "second"
    _run(root, first)
    _run(root, second)
    for name in ("matches", "summary", "candidates"):
        csv_name = f"t_{name}.csv"
        assert (first / csv_name).read_bytes() == (second / csv_name).read_bytes(), name
    md = (first / "t.md").read_text(encoding="utf-8").splitlines()
    (run_line,) = [line for line in md if line.startswith("- 실행:")]
    assert run_line.endswith("MKL_CBWR COMPATIBLE")
