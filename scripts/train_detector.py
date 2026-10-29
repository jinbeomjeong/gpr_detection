"""Train YOLO11m (one-channel, from scratch) on one AIHub fold with GPR augmentations.

Checklist M1-7 / M1-8. Reads a training config (default
``configs/train_yolo11m_base.yaml``), copies the chosen fold yaml to
``runs/detect/_data/<name>.yaml`` with ``channels: 1`` and starts the
Ultralytics trainer with the project augmentations attached
(:mod:`gpr_detection.detect.yolo_train`). Results go to ``runs/detect/<name>``
together with ``gpr_train_settings.yaml`` (the resolved settings).

Examples:
    conda run -n gpr-dev python -I scripts/train_detector.py --fold 0 --dry-run
    conda run -n gpr-dev python -I scripts/train_detector.py --fold 0 \\
        --epochs 3 --augment-strength 1.0 --seed 0
"""

import argparse
import dataclasses
import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

# gpr-dev's conda numpy uses MKL; with torch loaded, MKL's default OpenMP layer
# aborts on the first numpy matmul (0xC06D007F, or "OMP: Error #15" under
# `conda run`), which Ultralytics' affine augmentation triggers. The TBB layer
# works (checked 2026-10-29). Must be set before numpy is imported (the
# gpr_detection imports below load it); DataLoader workers inherit it.
os.environ.setdefault("MKL_THREADING_LAYER", "TBB")

from gpr_detection.augment import (
    AugmentConfig,
    augment_config_from_dict,
)
from gpr_detection.detect.yolo_train import (
    build_gpr_trainer,
    check_augmentation_policy,
    write_training_data_yaml,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_yolo11m_base.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
DATA_YAML_DIR = RUNS_DIR / "_data"
FOLD_CHOICES = ("0", "1", "2", "3", "4", "test")
SETTINGS_FILE_NAME = "gpr_train_settings.yaml"

LOGGER = logging.getLogger("train_detector")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--fold",
        choices=FOLD_CHOICES,
        default=None,
        help="validation fold 0-4, or 'test' (train on all folds); default: config",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="explicit Ultralytics data yaml instead of the fold yaml (smoke tests)",
    )
    parser.add_argument(
        "--augment-strength",
        type=float,
        default=None,
        help="multiplier of all GPR augmentation intensities (0 = off)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu")
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="share of the training images to use (Ultralytics 'fraction')",
    )
    parser.add_argument("--name", default=None, help="run folder name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved settings and exit without training",
    )
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _fold_yaml(dataset_dir: Path, fold: str) -> Path:
    file_name = "test.yaml" if fold == "test" else f"fold{fold}.yaml"
    return dataset_dir / file_name


def _resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _default_run_name(fold: str, strength: float, seed: int) -> str:
    return f"yolo11m_fold{fold}_aug{strength:g}_seed{seed}"


def _ultralytics_overrides(
    config: Mapping[str, Any], args: argparse.Namespace, data_yaml: Path, name: str
) -> dict[str, Any]:
    overrides = dict(config["ultralytics"])
    overrides["model"] = config["model"]
    overrides["data"] = str(data_yaml)
    overrides["project"] = str(RUNS_DIR)
    overrides["name"] = name
    optional_flags = {
        "epochs": args.epochs,
        "seed": args.seed,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "fraction": args.fraction,
    }
    for key, value in optional_flags.items():
        if value is not None:
            overrides[key] = value
    return overrides


def _augment_config(config: Mapping[str, Any], strength: float) -> AugmentConfig:
    settings = dict(config.get("gpr_augment", {}))
    settings["strength"] = strength
    return augment_config_from_dict(settings)


def _plain(value: Any) -> Any:
    """Convert dataclasses and enums to YAML/JSON-friendly values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {key: _plain(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Enum):
        return value.value
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve settings and train (or only print them with ``--dry-run``).

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    config = _load_config(args.config)

    if args.data is not None:
        fold = "custom"
    else:
        fold = args.fold if args.fold is not None else str(config["fold"])
    strength = (
        args.augment_strength
        if args.augment_strength is not None
        else float(config.get("gpr_augment", {}).get("strength", 1.0))
    )
    seed = args.seed if args.seed is not None else int(config["ultralytics"]["seed"])
    name = args.name or _default_run_name(fold, strength, seed)

    source_yaml = args.data or _fold_yaml(
        _resolve_project_path(config["dataset_dir"]), fold
    )
    data_yaml = DATA_YAML_DIR / f"{name}.yaml"
    overrides = _ultralytics_overrides(config, args, data_yaml, name)
    augment_config = _augment_config(config, strength)
    for warning in check_augmentation_policy(overrides):
        LOGGER.warning(warning)

    settings = {
        "config_file": str(args.config),
        "fold": fold,
        "source_data_yaml": str(source_yaml),
        "augment_strength": strength,
        "augment_seed": seed,
        "gpr_augment": _plain(augment_config),
        "ultralytics": overrides,
    }
    if args.dry_run:
        print(json.dumps(settings, indent=2, ensure_ascii=False))
        return 0

    write_training_data_yaml(source_yaml, data_yaml)
    trainer = build_gpr_trainer(overrides, augment_config, augment_seed=seed)
    save_dir = Path(trainer.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / SETTINGS_FILE_NAME).write_text(
        yaml.safe_dump(settings, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    LOGGER.info("training run folder: %s", save_dir)
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
