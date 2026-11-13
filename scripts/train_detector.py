"""Train YOLO11m (one-channel, from scratch) on one AIHub fold with GPR augmentations.

Checklist M1-7 / M1-8. Reads a training config (default
``configs/train_yolo11m_base.yaml``), copies the chosen fold yaml to
``runs/detect/_data/<name>.yaml`` with ``channels: 1`` and starts the
Ultralytics trainer with the project augmentations attached
(:mod:`gpr_detection.detect.yolo_train`). Results go to ``runs/detect/<name>``
together with ``gpr_train_settings.yaml`` (the resolved settings plus git commit,
torch/Ultralytics versions, train/val image counts and start time).

An existing ``runs/detect/<name>`` is never trained into again (Ultralytics would
otherwise silently create ``<name>2``). ``--on-existing`` decides what happens:

* ``error`` (default): exit 2 whenever the folder exists.
* ``skip-complete``: exit 0 without training if the run is complete, exit 2 if
  it is incomplete.
* ``archive``: move an incomplete folder to
  ``runs/detect/_incomplete/<name>_<YYYYmmddTHHMMSS>`` and train from scratch;
  a complete run is skipped (exit 0).

A run is complete when ``results.csv`` has at least ``epochs`` data rows and
``weights/last.pt`` exists. Complete folders are never moved or overwritten.

A ``gpr_geometric:`` block in the config turns on the N2-a geometric
augmentations (``docs/experiments/m31_n2a_retrain_plan.md`` section 5-3); without
it nothing changes (M1-10 configs train exactly as before). The settings file
also records the config sha256, and ``train_record.json`` (written after
training) the sha256 of ``last.pt`` / ``best.pt`` for the A1 gate.

``--blind`` (plan section 4-2): Ultralytics and project logs only at WARNING and
above (this also hides the Ultralytics progress bars, which turn themselves off
above INFO), one metric-free JSON progress line per epoch on stderr
(``{"stage", "fold", "epoch", "elapsed_s"}``) and a last stdout line
``{"status", "exit_code", "out"}``. An unexpected error then exits 3.

Examples:
    conda run -n gpr-dev python -I scripts/train_detector.py --fold 0 --dry-run
    conda run -n gpr-dev python -I scripts/train_detector.py --fold 0 \\
        --epochs 3 --augment-strength 1.0 --seed 0
    conda run -n gpr-dev python -I scripts/train_detector.py \\
        --config configs/train_yolo11m_m110.yaml --fold 0 \\
        --name yolo11m_aihubv1_20261030_fold0 --on-existing archive
"""

import argparse
import csv
import dataclasses
import json
import logging
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum, StrEnum
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
from gpr_detection.augment_geometry import (
    GeometricAugmentConfig,
    geometric_config_from_dict,
)
from gpr_detection.detect.yolo_logging import quiet_ultralytics_logger
from gpr_detection.detect.yolo_train import (
    build_gpr_trainer,
    check_augmentation_policy,
    geometric_settings,
    write_training_data_yaml,
)
from gpr_detection.run_info import (
    git_state,
    local_timestamp,
    package_versions,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_yolo11m_base.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
DATA_YAML_DIR = RUNS_DIR / "_data"
INCOMPLETE_DIR_NAME = "_incomplete"
FOLD_CHOICES = ("0", "1", "2", "3", "4", "test")
SETTINGS_FILE_NAME = "gpr_train_settings.yaml"
TRAIN_RECORD_FILE_NAME = "train_record.json"
TRAIN_RECORD_SCHEMA = "gpr_train_record/1"
RESULTS_FILE_NAME = "results.csv"
LAST_WEIGHTS_RELATIVE = Path("weights") / "last.pt"
RECORDED_WEIGHTS = ("last", "best")
ARCHIVE_TIME_FORMAT = "%Y%m%dT%H%M%S"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
VERSIONED_PACKAGES = ("torch", "ultralytics")

EXIT_OK = 0
EXIT_EXISTING_RUN = 2
"""Refused because of the existing run folder (see ``--on-existing``)."""
EXIT_INTERNAL_ERROR = 3
"""Unexpected error in a ``--blind`` run (without ``--blind`` it propagates)."""

LOGGER = logging.getLogger("train_detector")


class OnExisting(StrEnum):
    """What to do when ``runs/detect/<name>`` already exists."""

    ERROR = "error"
    SKIP_COMPLETE = "skip-complete"
    ARCHIVE = "archive"


class RunState(StrEnum):
    """State of a run folder before training."""

    ABSENT = "absent"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


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
        "--on-existing",
        choices=[choice.value for choice in OnExisting],
        default=OnExisting.ERROR.value,
        help="existing runs/detect/<name>: error (exit 2), skip-complete "
        "(complete -> exit 0, incomplete -> exit 2) or archive (move an "
        "incomplete folder to runs/detect/_incomplete and train)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved settings and exit without training",
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help="no metric output: logs at WARNING+, metric-free progress lines on "
        "stderr, last stdout line {status, exit_code, out}",
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


def _geometric_config(config: Mapping[str, Any]) -> GeometricAugmentConfig | None:
    """The ``gpr_geometric`` block, or ``None`` (off) when the config has none."""
    block = config.get("gpr_geometric")
    return None if block is None else geometric_config_from_dict(block)


def _plain(value: Any) -> Any:
    """Convert dataclasses and enums to YAML/JSON-friendly values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {key: _plain(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Enum):
        return value.value
    return value


def _write_train_record(run_dir: Path, config_sha256: str) -> None:
    """Hash the written weights so later gates can check them (A1, section 3-1)."""
    weights_sha256 = {}
    for weights_name in RECORDED_WEIGHTS:
        path = run_dir / "weights" / f"{weights_name}.pt"
        weights_sha256[weights_name] = sha256_file(path) if path.is_file() else None
    record = {
        "schema": TRAIN_RECORD_SCHEMA,
        "finished_at": local_timestamp(),
        "config_sha256": config_sha256,
        "weights_sha256": weights_sha256,
    }
    with (run_dir / TRAIN_RECORD_FILE_NAME).open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


def _fold_for_progress(fold: str) -> int | str:
    return int(fold) if fold.isdigit() else fold


def _add_blind_progress(trainer: Any, fold: str, started_s: float) -> None:
    """One metric-free JSON line per finished epoch on stderr."""

    def report_epoch(active_trainer: Any) -> None:
        line = {
            "stage": "train",
            "fold": _fold_for_progress(fold),
            "epoch": int(active_trainer.epoch) + 1,
            "elapsed_s": round(time.monotonic() - started_s, 1),
        }
        print(json.dumps(line), file=sys.stderr, flush=True)

    trainer.add_callback("on_train_epoch_end", report_epoch)


def _print_final_line(exit_code: int, out: Path | None) -> int:
    status = "ok" if exit_code == EXIT_OK else "error"
    line = {
        "status": status,
        "exit_code": exit_code,
        "out": None if out is None else str(out),
    }
    print(json.dumps(line), flush=True)
    return exit_code


def _count_results_rows(results_csv: Path) -> int:
    """Number of epoch rows in Ultralytics ``results.csv`` (header excluded)."""
    with results_csv.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
    return max(len(rows) - 1, 0)


def run_state(run_dir: Path, epochs: int) -> RunState:
    """Classify a run folder: absent, complete or incomplete.

    Complete means ``results.csv`` has at least ``epochs`` data rows and
    ``weights/last.pt`` exists.
    """
    if not run_dir.exists():
        return RunState.ABSENT
    results_csv = run_dir / RESULTS_FILE_NAME
    has_all_epochs = (
        results_csv.is_file() and _count_results_rows(results_csv) >= epochs
    )
    has_last_weights = (run_dir / LAST_WEIGHTS_RELATIVE).is_file()
    if has_all_epochs and has_last_weights:
        return RunState.COMPLETE
    return RunState.INCOMPLETE


def _archive_incomplete_run(run_dir: Path) -> Path:
    stamp = datetime.now().strftime(ARCHIVE_TIME_FORMAT)
    destination = RUNS_DIR / INCOMPLETE_DIR_NAME / f"{run_dir.name}_{stamp}"
    if destination.exists():
        raise FileExistsError(f"archive destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(run_dir, destination)
    return destination


def _handle_existing_run(
    run_dir: Path, state: RunState, policy: OnExisting
) -> int | None:
    """Apply ``--on-existing``; return an exit code to stop, ``None`` to train."""
    if state is RunState.ABSENT:
        return None
    if policy is OnExisting.ERROR:
        LOGGER.error(
            "run folder exists (%s): %s; choose --on-existing skip-complete or "
            "archive, or another --name",
            state.value,
            run_dir,
        )
        return EXIT_EXISTING_RUN
    if state is RunState.COMPLETE:
        LOGGER.info("run already complete, not training again: %s", run_dir)
        return EXIT_OK
    if policy is OnExisting.SKIP_COMPLETE:
        LOGGER.error(
            "incomplete run folder: %s; use --on-existing archive to restart it",
            run_dir,
        )
        return EXIT_EXISTING_RUN
    destination = _archive_incomplete_run(run_dir)
    LOGGER.warning("incomplete run moved to %s; training from scratch", destination)
    return None


def _image_count(entry: str | Path, dataset_root: Path) -> int:
    """Images named by one data-yaml entry (a list file or an image folder)."""
    path = Path(entry)
    if not path.is_absolute():
        path = dataset_root / path
    if path.is_dir():
        return sum(
            1 for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES
        )
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return sum(1 for line in lines if line.strip())


def _split_image_counts(data_yaml: Path) -> dict[str, int]:
    """Count the train and val images of an Ultralytics data yaml."""
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(data["path"])
    counts = {}
    for split in ("train", "val"):
        entries = data[split] if isinstance(data[split], list) else [data[split]]
        counts[split] = sum(_image_count(entry, dataset_root) for entry in entries)
    return counts


def _provenance() -> dict[str, Any]:
    return {
        "git": git_state(PROJECT_ROOT),
        "versions": package_versions(VERSIONED_PACKAGES),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve settings and train (or only print them with ``--dry-run``).

    Returns:
        Process exit code: 0 trained or skipped as complete, 2 refused because
        of the existing run folder, 3 unexpected error (``--blind`` only).
    """
    args = _build_parser().parse_args(argv)
    log_level = logging.WARNING if args.blind else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")
    if not args.blind:
        exit_code, _ = _train(args)
        return exit_code
    try:
        exit_code, run_dir = _train(args)
    except Exception:  # blind runs report through the exit code (plan §4-2)
        LOGGER.exception("training failed")
        return _print_final_line(EXIT_INTERNAL_ERROR, None)
    return _print_final_line(exit_code, run_dir)


def _train(args: argparse.Namespace) -> tuple[int, Path]:
    """Body of :func:`main`; returns the exit code and the run folder."""
    started_s = time.monotonic()
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
    geometric_config = _geometric_config(config)
    for warning in check_augmentation_policy(overrides, geometric_config):
        LOGGER.warning(warning)

    config_sha256 = sha256_file(args.config)
    settings = {
        "config_file": str(args.config),
        "config_sha256": config_sha256,
        "fold": fold,
        "source_data_yaml": str(source_yaml),
        "augment_strength": strength,
        "augment_seed": seed,
        "gpr_augment": _plain(augment_config),
        "gpr_geometric": (
            None if geometric_config is None else geometric_settings(geometric_config)
        ),
        "ultralytics": overrides,
    }
    run_dir = RUNS_DIR / name
    state = run_state(run_dir, int(overrides["epochs"]))
    if args.dry_run:
        print(
            json.dumps(
                {**settings, "existing_run": state.value}, indent=2, ensure_ascii=False
            )
        )
        return EXIT_OK, run_dir

    stop_code = _handle_existing_run(run_dir, state, OnExisting(args.on_existing))
    if stop_code is not None:
        return stop_code, run_dir

    write_training_data_yaml(source_yaml, data_yaml)
    settings["image_counts"] = _split_image_counts(data_yaml)
    settings.update(_provenance())
    if args.blind:
        quiet_ultralytics_logger()  # before the trainer logs its arguments
    trainer = build_gpr_trainer(
        overrides,
        augment_config,
        augment_seed=seed,
        geometric_config=geometric_config,
    )
    save_dir = Path(trainer.save_dir)
    if save_dir.resolve() != run_dir.resolve():
        raise RuntimeError(
            f"Ultralytics chose {save_dir} instead of {run_dir}; not training"
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    settings["started_at"] = local_timestamp()
    (save_dir / SETTINGS_FILE_NAME).write_text(
        yaml.safe_dump(settings, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    LOGGER.info("training run folder: %s", save_dir)
    if args.blind:
        _add_blind_progress(trainer, fold, started_s)
    trainer.train()
    _write_train_record(save_dir, config_sha256)
    return EXIT_OK, run_dir


if __name__ == "__main__":
    sys.exit(main())
