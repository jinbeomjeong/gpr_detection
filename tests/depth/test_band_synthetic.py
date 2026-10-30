"""Pseudo-band regression BR1-BR7 (spec_band_depth_v1 section 9-2, M2-6b).

Runs ``scripts/eval_band_depth_synthetic.py`` once and asserts every check of
one ID. Opt-in (``pytest -m synthetic``); skipped without
``data/synthetic_gprmax``.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from gpr_detection.io.paths import DEFAULT_SYNTHETIC_ROOT

pytestmark = pytest.mark.synthetic

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "eval_band_depth_synthetic.py"
)
MODULE_NAME = "script_eval_band_depth_synthetic"

EXPECTED_CHECK_COUNTS = {
    "BR1": 40,
    "BR2": 30,
    "BR3": 8,
    "BR4": 2,
    "BR5": 20,
    "BR6": 40,
    "BR7": 40,
}
"""20 models x 2 coverages; BR2 15 ground, BR3 4 at h 0.1, BR4 M20, BR5 cover 1.0."""


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
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
        sys.modules.pop(MODULE_NAME, None)


@pytest.mark.parametrize("check_id", list(EXPECTED_CHECK_COUNTS))
def test_band_regression_check(checks: list, check_id: str) -> None:
    group = [check for check in checks if check.check_id == check_id]
    assert len(group) == EXPECTED_CHECK_COUNTS[check_id]
    failures = [
        f"{check.model_id}@{check.coverage}: {check.measured} (need {check.criterion})"
        for check in group
        if not check.passed
    ]
    assert failures == []
