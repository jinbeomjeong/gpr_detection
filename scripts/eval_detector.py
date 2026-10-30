"""Evaluate a trained YOLO run on its validation fold and save the numbers (M1-10).

Plan: docs/experiments/m110_baseline_plan.md section 5-1. For each requested
weights file (``last``, ``best``) and unit (``crop``, ``image``) this writes
``<run>/eval/<weights>_<unit>/``:

* ``eval_images.txt`` - absolute paths of the evaluated images
* ``eval_data.yaml`` - one-channel data yaml pointing at that list (never train)
* ``metrics.json`` - per-class and overall metrics, confusion matrix, weights
  and list hashes, settings, versions, git commit
* ``per_class.csv``, ``confusion_matrix.csv`` - the same numbers as tables
* Ultralytics plots (PR curves, confusion matrix PNG)

Ultralytics 8.4.192 builds the confusion matrix at the validation ``conf``
(0.001 here). ``--confusion-conf 0.25`` validates once more at that threshold
for the matrix only: ``confusion_matrix_conf0.25.csv`` next to the files above,
its plots in the subfolder ``confusion_conf0.25/``; the AP metrics and
``confusion_matrix.csv`` stay those of ``--conf``.

Units: ``image`` evaluates every image of the fold (all gain copies, as during
training); ``crop`` keeps one gain copy per crop (1.0 > 0.5 > 0.25), the unit
the M1-10 judgement uses because gain copies are not independent samples.

The fold and the dataset come from ``<run>/gpr_train_settings.yaml`` unless
given. The test hold-out is sealed until M4-3 (user decision 2026-10-30):
``--fold test`` (or a run trained with fold test) is refused with exit 2
unless ``--allow-test`` is given.

Exit codes: 0 done, 1 input error (missing file, list mismatch), 2 test refused.

Examples:
    conda run -n gpr-dev python -I scripts/eval_detector.py \\
        --run-dir runs/detect/yolo11m_aihubv1_20261030_fold0 --device 0
    conda run -n gpr-dev python -I scripts/eval_detector.py \\
        --run-dir runs/detect/m17_aug0 --weights last --unit crop --device 0
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

# Same reason as in train_detector.py: MKL's default OpenMP layer aborts once
# torch is loaded; must be set before numpy is imported.
os.environ.setdefault("MKL_THREADING_LAYER", "TBB")

from gpr_detection.data.splits import (
    DEFAULT_GAIN_PRIORITY,
    FILE_NAME_COLUMN,
    FOLD_COLUMN,
    GAIN_COLUMN,
    select_crop_representatives,
)
from gpr_detection.detect.yolo_eval import evaluate_yolo
from gpr_detection.run_info import (
    git_state,
    local_timestamp,
    package_versions,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "splits" / "aihub_yolo_v1.csv"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "aihub_yolo_v1"
SETTINGS_FILE_NAME = "gpr_train_settings.yaml"
EVAL_DIR_NAME = "eval"
WEIGHTS_CHOICES = ("last", "best")
UNIT_CHOICES = ("crop", "image")
FOLD_CHOICES = ("0", "1", "2", "3", "4", "test")
TEST_FOLD_NAME = "test"
MANIFEST_TEST_FOLD = "-1"
IMAGES_DIR_NAME = "images"
GRAY_CHANNELS = 1
METRICS_SCHEMA = "gpr_eval/1"
VERSIONED_PACKAGES = ("torch", "ultralytics")

EXIT_OK = 0
EXIT_INPUT_ERROR = 1
EXIT_TEST_SEALED = 2

LOGGER = logging.getLogger("eval_detector")


@dataclass(frozen=True)
class EvalSettings:
    """Ultralytics validation settings shared by every evaluation of one call."""

    imgsz: int
    batch: int
    conf: float
    iou: float
    device: str | None
    confusion_conf: float | None = None
    """Extra validation run used only for its confusion matrix (``None`` = off)."""


@dataclass(frozen=True)
class EvalTarget:
    """One evaluation: a weights file on one unit's image list."""

    weights_name: str
    weights_path: Path
    unit: str
    out_dir: Path


class EvalInputError(Exception):
    """A required input is missing or inconsistent (exit code 1)."""


def _comma_list(choices: Sequence[str]) -> Any:
    def parse(text: str) -> list[str]:
        values = [value.strip() for value in text.split(",") if value.strip()]
        unknown = sorted(set(values) - set(choices))
        if not values or unknown:
            raise argparse.ArgumentTypeError(
                f"expected a comma list of {', '.join(choices)}; got {text!r}"
            )
        return list(dict.fromkeys(values))

    return parse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--weights", type=_comma_list(WEIGHTS_CHOICES), default=list(WEIGHTS_CHOICES)
    )
    parser.add_argument(
        "--unit", type=_comma_list(UNIT_CHOICES), default=list(UNIT_CHOICES)
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="folder with images/, labels/ and data yamls; default: the folder "
        "of the run's source data yaml",
    )
    parser.add_argument(
        "--fold", choices=FOLD_CHOICES, default=None, help="default: the run's fold"
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="unseal the test hold-out (only for the final M4-3 measurement)",
    )
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--confusion-conf",
        type=float,
        default=None,
        help="also validate once more at this score threshold for the confusion "
        "matrix only (e.g. 0.25); AP metrics always use --conf",
    )
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu")
    return parser


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_train_settings(run_dir: Path) -> dict[str, Any]:
    settings_path = run_dir / SETTINGS_FILE_NAME
    if not settings_path.is_file():
        raise EvalInputError(f"missing {settings_path}")
    return yaml.safe_load(settings_path.read_text(encoding="utf-8"))


def _resolve_dataset_dir(
    explicit_dir: Path | None, train_settings: Mapping[str, Any]
) -> Path:
    if explicit_dir is not None:
        return _resolve_project_path(explicit_dir)
    source_yaml = train_settings.get("source_data_yaml")
    if source_yaml:
        return Path(source_yaml).parent
    return DEFAULT_DATASET_DIR


def _manifest_fold(fold: str) -> str:
    return MANIFEST_TEST_FOLD if fold == TEST_FOLD_NAME else fold


def _fold_file_stem(fold: str) -> str:
    return TEST_FOLD_NAME if fold == TEST_FOLD_NAME else f"fold{fold}"


def _fold_list_name(fold: str) -> str:
    return f"{_fold_file_stem(fold)}_images.txt"


def _fold_yaml_name(fold: str) -> str:
    return f"{_fold_file_stem(fold)}.yaml"


def _read_fold_rows(manifest: Path, fold: str) -> list[dict[str, str]]:
    if not manifest.is_file():
        raise EvalInputError(f"missing manifest {manifest}")
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row[FOLD_COLUMN] == _manifest_fold(fold)
        ]
    if not rows:
        raise EvalInputError(f"no manifest rows for fold {fold} in {manifest}")
    return rows


def _check_rows_match_fold_list(
    rows: Sequence[Mapping[str, str]], dataset_dir: Path, fold: str
) -> None:
    """The manifest fold must be exactly the fold list the run validated on."""
    list_path = dataset_dir / _fold_list_name(fold)
    if not list_path.is_file():
        raise EvalInputError(f"missing fold list {list_path}")
    lines = list_path.read_text(encoding="utf-8-sig").splitlines()
    listed = {Path(line.strip()).name for line in lines if line.strip()}
    from_manifest = {row[FILE_NAME_COLUMN] for row in rows}
    if listed != from_manifest:
        raise EvalInputError(
            f"{list_path} and the manifest fold {fold} differ: "
            f"{len(listed - from_manifest)} only in the list, "
            f"{len(from_manifest - listed)} only in the manifest"
        )


def _rows_for_unit(rows: Sequence[Mapping[str, str]], unit: str) -> list[Mapping]:
    if unit == "crop":
        return select_crop_representatives(rows, DEFAULT_GAIN_PRIORITY)
    return sorted(rows, key=lambda row: row[FILE_NAME_COLUMN])


def _write_eval_list(rows: Sequence[Mapping], dataset_dir: Path, out_dir: Path) -> Path:
    """Write absolute image paths, LF line ends (stable hash on any platform)."""
    images_dir = (dataset_dir / IMAGES_DIR_NAME).resolve()
    lines = [str(images_dir / str(row[FILE_NAME_COLUMN])) for row in rows]
    list_path = out_dir / "eval_images.txt"
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return list_path


def _write_eval_data_yaml(
    dataset_dir: Path, fold: str, list_path: Path, out_dir: Path
) -> Path:
    fold_yaml = dataset_dir / _fold_yaml_name(fold)
    if not fold_yaml.is_file():
        raise EvalInputError(f"missing fold yaml {fold_yaml}")
    names = yaml.safe_load(fold_yaml.read_text(encoding="utf-8"))["names"]
    data = {
        "path": str(dataset_dir.resolve()),
        # Ultralytics requires a train key; it points at the evaluation list.
        "train": str(list_path),
        "val": str(list_path),
        "names": names,
        "channels": GRAY_CHANNELS,
    }
    data_yaml = out_dir / "eval_data.yaml"
    data_yaml.write_text(
        "# EVALUATION ONLY - never train on this file. Written by "
        f"scripts/eval_detector.py for {_fold_yaml_name(fold)}; adds channels: 1\n"
        + yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return data_yaml


def _write_per_class_csv(per_class: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    columns = [
        "class_id",
        "name",
        "instances",
        "images",
        "precision",
        "recall",
        "ap50",
        "ap50_95",
    ]
    with (out_dir / "per_class.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in per_class:
            writer.writerow({column: row[column] for column in columns})


def _confusion_suffix(confusion_conf: float) -> str:
    """``conf0.25`` style tag for the second confusion matrix's files."""
    return f"conf{confusion_conf:g}"


def _write_confusion_matrix_csv(
    confusion: Mapping[str, Any] | None,
    out_dir: Path,
    file_name: str = "confusion_matrix.csv",
) -> None:
    """Rows = predicted class, columns = true class (as in metrics.json)."""
    if confusion is None:
        return
    labels = confusion["labels"]
    with (out_dir / file_name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["predicted\\true", *labels])
        for label, counts in zip(labels, confusion["matrix"], strict=True):
            writer.writerow([label, *counts])


def _gain_counts(rows: Sequence[Mapping]) -> dict[str, int]:
    return dict(sorted(Counter(str(row[GAIN_COLUMN]) for row in rows).items()))


def _metrics_document(
    target: EvalTarget,
    fold: str,
    run_dir: Path,
    eval_rows: Sequence[Mapping],
    list_path: Path,
    settings: EvalSettings,
    result: Mapping[str, Any],
    confusion_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    epoch_index = result.get("checkpoint_epoch")
    second_matrix = (
        None if confusion_result is None else confusion_result["confusion_matrix"]
    )
    return {
        "schema": METRICS_SCHEMA,
        "evaluated_at": local_timestamp(),
        "run_dir": str(run_dir),
        "fold": fold,
        "unit": target.unit,
        "weights": {
            "name": target.weights_name,
            "path": str(target.weights_path),
            "sha256": sha256_file(target.weights_path),
            "checkpoint_epoch_zero_based": epoch_index,
            "epochs_completed": None if epoch_index is None else epoch_index + 1,
        },
        "eval_images": {
            "path": str(list_path),
            "sha256": sha256_file(list_path),
            "count": len(eval_rows),
            "gain_counts": _gain_counts(eval_rows),
        },
        "class_names": result["class_names"],
        "classes_present": result["classes_present"],
        "per_class": result["per_class"],
        "map50": result["map50"],
        "map50_95": result["map50_95"],
        "pipe_ap50_mean": result["pipe_ap50_mean"],
        "conf_threshold": settings.conf,
        "confusion_conf_threshold": settings.confusion_conf,
        "confusion_matrix": result["confusion_matrix"],
        # Second validation run used only for this matrix (--confusion-conf).
        "confusion_matrix_at_confusion_conf": second_matrix,
        "settings": {
            "imgsz": settings.imgsz,
            "batch": settings.batch,
            "conf": settings.conf,
            "confusion_conf": settings.confusion_conf,
            "iou": settings.iou,
            "device": settings.device,
            "rect": True,  # YOLO.val default
            "values_are_fractions": True,
        },
        "versions": {
            "python": sys.version.split()[0],
            **package_versions(VERSIONED_PACKAGES),
        },
        "git": git_state(PROJECT_ROOT),
    }


def _evaluate_target(
    target: EvalTarget,
    fold: str,
    run_dir: Path,
    fold_rows: Sequence[Mapping[str, str]],
    dataset_dir: Path,
    settings: EvalSettings,
) -> dict[str, Any]:
    eval_rows = _rows_for_unit(fold_rows, target.unit)
    target.out_dir.mkdir(parents=True, exist_ok=True)
    list_path = _write_eval_list(eval_rows, dataset_dir, target.out_dir)
    data_yaml = _write_eval_data_yaml(dataset_dir, fold, list_path, target.out_dir)
    LOGGER.info(
        "evaluating %s on %d %s-level images -> %s",
        target.weights_path,
        len(eval_rows),
        target.unit,
        target.out_dir,
    )
    result = _validate(target.weights_path, data_yaml, target.out_dir, settings)
    confusion_result = None
    if settings.confusion_conf is not None:
        confusion_result = _validate(
            target.weights_path,
            data_yaml,
            _confusion_run_dir(target.out_dir, settings.confusion_conf),
            replace(settings, conf=settings.confusion_conf),
        )
    document = _metrics_document(
        target,
        fold,
        run_dir,
        eval_rows,
        list_path,
        settings,
        result,
        confusion_result,
    )
    (target.out_dir / "metrics.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_per_class_csv(document["per_class"], target.out_dir)
    _write_confusion_matrix_csv(document["confusion_matrix"], target.out_dir)
    if settings.confusion_conf is not None:
        _write_confusion_matrix_csv(
            document["confusion_matrix_at_confusion_conf"],
            target.out_dir,
            f"confusion_matrix_{_confusion_suffix(settings.confusion_conf)}.csv",
        )
    return document


def _confusion_run_dir(out_dir: Path, confusion_conf: float) -> Path:
    """Subfolder for the second run, so its PNGs never replace the first run's."""
    return out_dir / f"confusion_{_confusion_suffix(confusion_conf)}"


def _validate(
    weights: Path, data_yaml: Path, out_dir: Path, settings: EvalSettings
) -> dict[str, Any]:
    return evaluate_yolo(
        weights,
        data_yaml,
        out_dir,
        imgsz=settings.imgsz,
        batch=settings.batch,
        conf=settings.conf,
        iou=settings.iou,
        device=settings.device,
    )


def _targets(
    run_dir: Path, weights: Sequence[str], units: Sequence[str]
) -> list[EvalTarget]:
    targets = []
    for weights_name in weights:
        weights_path = run_dir / "weights" / f"{weights_name}.pt"
        if not weights_path.is_file():
            raise EvalInputError(f"missing weights {weights_path}")
        for unit in units:
            out_dir = run_dir / EVAL_DIR_NAME / f"{weights_name}_{unit}"
            targets.append(EvalTarget(weights_name, weights_path, unit, out_dir))
    return targets


def _is_sealed_test(fold: str | None, allow_test: bool) -> bool:
    if fold != TEST_FOLD_NAME or allow_test:
        return False
    LOGGER.error(
        "the test hold-out is sealed until M4-3 (user decision 2026-10-30); "
        "pass --allow-test only for the final measurement"
    )
    return True


def _run(args: argparse.Namespace) -> int:
    # Checked before anything is read, and again for the fold the run used.
    if _is_sealed_test(args.fold, args.allow_test):
        return EXIT_TEST_SEALED
    run_dir = _resolve_project_path(args.run_dir)
    train_settings = _read_train_settings(run_dir)
    fold = args.fold if args.fold is not None else str(train_settings["fold"])
    if _is_sealed_test(fold, args.allow_test):
        return EXIT_TEST_SEALED
    if fold not in FOLD_CHOICES:
        raise EvalInputError(f"fold {fold!r} is not one of {FOLD_CHOICES}")

    dataset_dir = _resolve_dataset_dir(args.dataset_dir, train_settings)
    fold_rows = _read_fold_rows(_resolve_project_path(args.manifest), fold)
    _check_rows_match_fold_list(fold_rows, dataset_dir, fold)
    settings = EvalSettings(
        args.imgsz,
        args.batch,
        args.conf,
        args.iou,
        args.device,
        confusion_conf=args.confusion_conf,
    )

    for target in _targets(run_dir, args.weights, args.unit):
        document = _evaluate_target(
            target, fold, run_dir, fold_rows, dataset_dir, settings
        )
        LOGGER.info(
            "%s_%s: mAP50 %s, mAP50-95 %s, pipe AP50 mean %s",
            target.weights_name,
            target.unit,
            document["map50"],
            document["map50_95"],
            document["pipe_ap50_mean"]["value"],
        )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate the requested weights and units of one run.

    Returns:
        Process exit code: 0 done, 1 input error, 2 test hold-out refused.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    except EvalInputError as error:
        LOGGER.error("%s", error)
        return EXIT_INPUT_ERROR


if __name__ == "__main__":
    sys.exit(main())
