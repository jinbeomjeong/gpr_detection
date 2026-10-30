"""Shared fixtures: load a ``scripts/*.py`` file as a module for in-process tests."""

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


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
