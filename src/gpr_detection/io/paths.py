"""Locate the five read-only dataset roots from a single ``configs/paths.yaml``."""

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

PATHS_CONFIG_ENV_VAR = "GPR_PATHS_CONFIG"
# src/gpr_detection/io/paths.py -> project root is three levels above the package.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATHS_CONFIG = _PROJECT_ROOT / "configs" / "paths.yaml"
DEFAULT_SYNTHETIC_ROOT = _PROJECT_ROOT / "data" / "synthetic_gprmax"
"""gprMax synthetic set: a derived project folder (``data/``), not a read-only
dataset under ``gpr_data_root``, so it is not listed in paths.yaml."""


class DatasetName(StrEnum):
    """Keys under ``datasets:`` in paths.yaml."""

    AIHUB = "aihub"
    MOROCCO = "morocco"
    FOURTU = "fourtu"
    GUANGZHOU = "guangzhou"
    GPRSD = "gprsd"


@dataclass(frozen=True)
class DatasetPaths:
    """Root folder of each dataset package (they are never written to).

    Attributes:
        aihub: AIHub 71327 ``01-1.정식개방데이터`` folder (Training/Validation zips).
        morocco: Morocco processed package root.
        fourtu: 4TU processed package root.
        guangzhou: Guangzhou University processed package root.
        gprsd: GPR-SD processed package root.
    """

    aihub: Path
    morocco: Path
    fourtu: Path
    guangzhou: Path
    gprsd: Path

    def root(self, dataset: DatasetName) -> Path:
        """Return the root folder of ``dataset``."""
        return getattr(self, dataset.value)


def _resolve_config_path(config_path: Path | None) -> Path:
    if config_path is not None:
        return config_path
    from_env = os.environ.get(PATHS_CONFIG_ENV_VAR)
    return Path(from_env) if from_env else DEFAULT_PATHS_CONFIG


def load_dataset_paths(config_path: Path | None = None) -> DatasetPaths:
    """Read dataset roots from paths.yaml.

    Args:
        config_path: YAML file to read. Defaults to ``$GPR_PATHS_CONFIG`` or
            ``<project>/configs/paths.yaml``.

    Returns:
        Absolute dataset roots. Existence is not checked so that callers can
        skip datasets that are not present on this machine.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a dataset entry is missing.
    """
    path = _resolve_config_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset path config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    data_root = Path(config["gpr_data_root"])
    entries: dict[str, str] = config["datasets"]

    roots: dict[str, Path] = {}
    for dataset in DatasetName:
        if dataset.value not in entries:
            raise KeyError(f"{path}: datasets.{dataset.value} is missing")
        # An absolute entry wins over gpr_data_root because "/" drops the left side.
        roots[dataset.value] = data_root / entries[dataset.value]
    return DatasetPaths(**roots)
