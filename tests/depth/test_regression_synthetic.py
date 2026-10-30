"""Synthetic depth regression R1-R13 (spec_depth_v1 section 11-2, CHECKLIST M2-5).

Runs ``scripts/eval_depth_synthetic.py`` once per session on the 20 gprMax
models and asserts every check of one ID. Excluded from the default run
(``-m "not synthetic"`` in pyproject); run with ``pytest -m synthetic``.
Skipped when ``data/synthetic_gprmax`` is missing.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from gpr_detection.io.paths import DEFAULT_SYNTHETIC_ROOT

pytestmark = pytest.mark.synthetic

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "eval_depth_synthetic.py"
)

EXPECTED_CHECK_COUNTS = {
    "R1": 15,
    "R2": 1,
    "R3": 4,
    "R4": 20,
    "R5": 20,
    "R6": 1,
    "R7": 4,
    "R8": 15,
    "R9": 15,
    "R10": 15,
    "R11": 15,
    "R12": 1,
    "R13a-1": 1,
    "R13a-2": 1,
    "R13b": 1,
}
"""Models per check: 15 ground, h 0.1 (M12, M13, M15, M18), h 0.3 (M20)."""


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "script_eval_depth_synthetic", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses look the module up in sys.modules while the class is built.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checks() -> list:
    if not (DEFAULT_SYNTHETIC_ROOT / "manifest.csv").is_file():
        pytest.skip(f"synthetic data not found at {DEFAULT_SYNTHETIC_ROOT}")
    script = _load_script()
    try:
        return script.regression_checks(script.evaluate_all())
    finally:
        sys.modules.pop("script_eval_depth_synthetic", None)


@pytest.mark.parametrize("check_id", list(EXPECTED_CHECK_COUNTS))
def test_regression_check(checks: list, check_id: str) -> None:
    group = [check for check in checks if check.check_id == check_id]
    assert len(group) == EXPECTED_CHECK_COUNTS[check_id]
    failures = [
        f"{check.model_id}: {check.measured} (need {check.criterion})"
        for check in group
        if not check.passed
    ]
    assert failures == []
