"""Shared fixtures: load a ``scripts/*.py`` file as a module for in-process tests.

Also registers the opt-in ``realdata`` marker: tests that read many real
dataset files (M3-1 T7(e)) are skipped unless the ``-m`` expression names
``realdata`` (e.g. ``pytest -m realdata``).
"""

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
REALDATA_MARKER = "realdata"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{REALDATA_MARKER}: real-data round trips (opt-in, -m {REALDATA_MARKER})",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if REALDATA_MARKER in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason=f"opt-in: run with -m {REALDATA_MARKER}")
    for item in items:
        if REALDATA_MARKER in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def load_script(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], ModuleType]:
    """Return a loader that imports ``scripts/<name>.py`` as a fresh module.

    Scripts are not a package; loading them by path lets tests call ``main``
    and monkeypatch module globals (output folders, the training call).
    """

    def load(name: str) -> ModuleType:
        spec = importlib.util.spec_from_file_location(
            f"script_{name}", SCRIPTS_DIR / f"{name}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # dataclasses look the module up in sys.modules while the class is built.
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    return load
