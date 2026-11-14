"""Static checks of the module dependency rules (architecture README section 2-2).

1. ``detect`` and ``depth`` never import each other (they meet in ``pipeline``).
2. ``io`` imports no other ``gpr_detection`` subpackage.
3. ``ultralytics`` / ``rfdetr`` are imported only under ``detect/`` and only
   inside functions (lazy), never at module level (ADR-001, CRN-4).
4. ``src`` never imports ``scripts``.
8. ``eval`` imports only the standard library, numpy, scipy,
   ``gpr_detection.io``, ``gpr_detection.preprocess`` and itself (no
   ``detect``, ``pipeline``, ``depth``, detector libraries or torch).
9. No module of ``gpr_detection`` outside ``eval/`` imports ``gpr_detection.eval``.

Imports are read with ``ast`` so that imports nested in functions count too.
"""

import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "gpr_detection"
DETECTOR_LIBRARIES = ("ultralytics", "rfdetr")
NEUTRAL_DETECT_MODULES = ("__init__.py", "base.py", "band_metrics.py")


@dataclass(frozen=True)
class ImportRecord:
    path: Path
    module: str
    line: int
    is_module_level: bool

    @property
    def top_level_name(self) -> str:
        return self.module.split(".")[0]

    @property
    def relative_path(self) -> str:
        return self.path.relative_to(PACKAGE_ROOT).as_posix()


def _package_of(path: Path) -> list[str]:
    """Dotted package parts that contain ``path`` (``__init__.py`` is its own)."""
    module_parts = path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts
    return list(module_parts[:-1])


def _absolute_base_module(node: ast.ImportFrom, path: Path) -> str:
    """Resolve ``from <level dots><module>`` to an absolute dotted name."""
    if node.level == 0:
        return node.module or ""
    package_parts = _package_of(path)
    levels_up = node.level - 1
    if levels_up >= len(package_parts):
        raise ValueError(f"{path}:{node.lineno} relative import beyond top package")
    base_parts = package_parts[: len(package_parts) - levels_up]
    if node.module:
        base_parts.append(node.module)
    return ".".join(base_parts)


def _imported_names(node: ast.Import | ast.ImportFrom, path: Path) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    base_module = _absolute_base_module(node, path)
    # ``from pkg import sub`` may bind a submodule, so record ``pkg.sub`` too.
    names = [base_module]
    for alias in node.names:
        if alias.name != "*":
            names.append(f"{base_module}.{alias.name}")
    return names


def _imports_in_source(source: str, path: Path) -> list[ImportRecord]:
    tree = ast.parse(source, filename=str(path))
    module_level_nodes = {id(node) for node in tree.body}
    records = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        is_module_level = id(node) in module_level_nodes
        for name in _imported_names(node, path):
            records.append(ImportRecord(path, name, node.lineno, is_module_level))
    return records


def _imports_of(path: Path) -> list[ImportRecord]:
    return _imports_in_source(path.read_text(encoding="utf-8"), path)


def _all_imports() -> list[ImportRecord]:
    records = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        records.extend(_imports_of(path))
    return records


def _in_subpackage(record: ImportRecord, subpackage: str) -> bool:
    return record.relative_path.startswith(f"{subpackage}/")


ALL_IMPORTS = _all_imports()


@pytest.mark.parametrize(
    ("relative_file", "source", "expected_name"),
    [
        ("detect/fake.py", "from gpr_detection import depth", "gpr_detection.depth"),
        ("detect/fake.py", "from .. import depth", "gpr_detection.depth"),
        ("detect/fake.py", "from ..depth.fit import f", "gpr_detection.depth.fit"),
        ("depth/fake.py", "from ..detect import yolo", "gpr_detection.detect.yolo"),
        ("detect/__init__.py", "from . import yolo", "gpr_detection.detect.yolo"),
        ("fake.py", "from scripts import train", "scripts.train"),
        ("augment.py", "from ultralytics import YOLO", "ultralytics.YOLO"),
    ],
)
def test_collector_resolves_from_imports(relative_file, source, expected_name) -> None:
    records = _imports_in_source(source, PACKAGE_ROOT / relative_file)
    assert expected_name in {record.module for record in records}


def test_collector_marks_function_level_imports() -> None:
    source = "def load():\n    from ultralytics import YOLO\n"
    records = _imports_in_source(source, PACKAGE_ROOT / "detect" / "fake.py")
    assert {record.module for record in records} == {"ultralytics", "ultralytics.YOLO"}
    assert not any(record.is_module_level for record in records)


def test_collector_rejects_relative_import_beyond_top_package() -> None:
    with pytest.raises(ValueError, match="beyond top package"):
        _imports_in_source("from ... import x", PACKAGE_ROOT / "fake.py")


def test_imports_were_found() -> None:
    assert len(ALL_IMPORTS) > 50


def test_detector_libraries_only_in_detect_backends() -> None:
    offenders = [
        f"{record.relative_path}:{record.line} {record.module}"
        for record in ALL_IMPORTS
        if record.top_level_name in DETECTOR_LIBRARIES
        and not _in_subpackage(record, "detect")
    ]
    assert offenders == []


def test_detector_libraries_are_imported_lazily() -> None:
    offenders = [
        f"{record.relative_path}:{record.line} {record.module}"
        for record in ALL_IMPORTS
        if record.top_level_name in DETECTOR_LIBRARIES
        and (record.is_module_level or record.path.name in NEUTRAL_DETECT_MODULES)
    ]
    assert offenders == []


@pytest.mark.parametrize(
    ("importer", "forbidden"),
    [("detect", "gpr_detection.depth"), ("depth", "gpr_detection.detect")],
)
def test_detect_and_depth_do_not_import_each_other(importer, forbidden) -> None:
    offenders = [
        f"{record.relative_path}:{record.line} {record.module}"
        for record in ALL_IMPORTS
        if _in_subpackage(record, importer) and record.module.startswith(forbidden)
    ]
    assert offenders == []


def test_io_imports_only_itself_from_the_package() -> None:
    offenders = [
        f"{record.relative_path}:{record.line} {record.module}"
        for record in ALL_IMPORTS
        if _in_subpackage(record, "io")
        and record.top_level_name == "gpr_detection"
        and not record.module.startswith("gpr_detection.io")
    ]
    assert offenders == []


def test_augment_module_is_backend_neutral() -> None:
    names = {
        record.top_level_name for record in _imports_of(PACKAGE_ROOT / "augment.py")
    }
    assert names.isdisjoint({*DETECTOR_LIBRARIES, "torch", "cv2"})


AUGMENT_GEOMETRY_ALLOWED_THIRD_PARTY = frozenset({"numpy", "cv2"})
AUGMENT_GEOMETRY_FORBIDDEN = frozenset({*DETECTOR_LIBRARIES, "torch"})


def test_augment_geometry_imports_only_the_allow_list() -> None:
    """``augment_geometry.py``: stdlib, numpy, cv2 and gpr_detection only.

    An allow list (not a deny list) so a new heavy dependency fails here;
    torch, ultralytics and rfdetr are named for a clear message.
    """
    path = PACKAGE_ROOT / "augment_geometry.py"
    if not path.is_file():
        pytest.skip("augment_geometry.py not present")
    allowed = {
        *sys.stdlib_module_names,
        *AUGMENT_GEOMETRY_ALLOWED_THIRD_PARTY,
        "gpr_detection",
    }
    names = {record.top_level_name for record in _imports_of(path)}
    assert names.isdisjoint(AUGMENT_GEOMETRY_FORBIDDEN)
    assert names <= allowed, sorted(names - allowed)


def test_augment_geometry_allow_list_rejects_detector_imports() -> None:
    source = "import numpy\nfrom torch import nn\nimport ultralytics\nimport rfdetr\n"
    records = _imports_in_source(source, PACKAGE_ROOT / "augment_geometry.py")
    names = {record.top_level_name for record in records}
    assert names & AUGMENT_GEOMETRY_FORBIDDEN == {"torch", "ultralytics", "rfdetr"}


EVAL_ALLOWED_THIRD_PARTY = frozenset({"numpy", "scipy"})
EVAL_ALLOWED_PACKAGE_PREFIXES = (
    "gpr_detection.io",
    "gpr_detection.preprocess",
    "gpr_detection.eval",
)
EVAL_FORBIDDEN = frozenset({*DETECTOR_LIBRARIES, "torch"})


def _is_within(module: str, package: str) -> bool:
    """True for ``package`` itself and its submodules (not ``package_x``)."""
    return module == package or module.startswith(f"{package}.")


def _violates_eval_allow_list(record: ImportRecord) -> bool:
    """Rule 8 for one import of an ``eval/`` module.

    The bare package ``gpr_detection`` (``from gpr_detection import io``) is
    allowed: its ``__init__`` imports nothing, and the submodule it binds is a
    separate record that must pass the prefix check.
    """
    top_level = record.top_level_name
    if top_level in EVAL_FORBIDDEN:
        return True
    if top_level == "gpr_detection":
        return record.module != "gpr_detection" and not any(
            _is_within(record.module, prefix)
            for prefix in EVAL_ALLOWED_PACKAGE_PREFIXES
        )
    allowed = {*sys.stdlib_module_names, *EVAL_ALLOWED_THIRD_PARTY}
    return top_level not in allowed


def test_eval_imports_only_the_allow_list() -> None:
    """Rule 8: ``eval/`` stays on stdlib, numpy, scipy, io and preprocess."""
    if not (PACKAGE_ROOT / "eval").is_dir():
        pytest.skip("eval/ not present")
    offenders = [
        f"{record.relative_path}:{record.line} {record.module}"
        for record in ALL_IMPORTS
        if _in_subpackage(record, "eval") and _violates_eval_allow_list(record)
    ]
    assert offenders == []


@pytest.mark.parametrize(
    ("source", "is_violation"),
    [
        ("from gpr_detection import depth", True),
        ("from ..pipeline import x", True),
        ("import gpr_detection.detect", True),
        ("import ultralytics", True),
        ("import torch", True),
        ("import h5py", True),
        ("from ..preprocess.to_image import make_detection_image", False),
        ("from gpr_detection.io.radargram import GridBox", False),
        ("from . import inject", False),
        ("import scipy.signal", False),
        ("import csv", False),
    ],
)
def test_eval_allow_list_rejects_forbidden_imports(source, is_violation) -> None:
    records = _imports_in_source(source, PACKAGE_ROOT / "eval" / "fake.py")
    assert any(_violates_eval_allow_list(record) for record in records) is is_violation


def test_package_never_imports_eval() -> None:
    """Rule 9: ``eval`` is evaluation-only; the rest of ``src`` never imports it."""
    offenders = [
        f"{record.relative_path}:{record.line} {record.module}"
        for record in ALL_IMPORTS
        if not _in_subpackage(record, "eval")
        and _is_within(record.module, "gpr_detection.eval")
    ]
    assert offenders == []


def test_eval_collector_sees_relative_imports_from_outside() -> None:
    records = _imports_in_source(
        "from ..eval import inject", PACKAGE_ROOT / "detect" / "fake.py"
    )
    assert "gpr_detection.eval.inject" in {record.module for record in records}


def test_package_never_imports_scripts() -> None:
    offenders = [
        f"{record.relative_path}:{record.line} {record.module}"
        for record in ALL_IMPORTS
        if record.top_level_name == "scripts"
    ]
    assert offenders == []


def test_importing_contract_modules_does_not_load_detector_libraries() -> None:
    probe = (
        "import sys\n"
        "import gpr_detection.detect, gpr_detection.detect.yolo\n"
        "import gpr_detection.detect.yolo_train, gpr_detection.pipeline\n"
        "import gpr_detection.augment, gpr_detection.detect.band_metrics\n"
        "import gpr_detection.eval.inject\n"
        "loaded = [n for n in ('ultralytics', 'rfdetr', 'torch') if n in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert completed.stdout.strip() == ""
