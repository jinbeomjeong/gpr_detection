"""Tests of ``scripts/aggregate_folds.py`` on five synthetic fold runs."""

import csv
import json
import statistics
from pathlib import Path
from types import ModuleType

import pytest
import yaml

CLASS_NAMES = ["pipe_hyperbola", "pipe_band", "cavity", "manhole"]
MAP50_BY_FOLD = [0.84, 0.80, 0.86, 0.79, 0.82]
TRAINING_BLOCK = {"epochs": 50, "seed": 0, "imgsz": 256, "optimizer": "AdamW"}
EVAL_SETTINGS = {"imgsz": 256, "batch": 64, "conf": 0.001, "iou": 0.7, "rect": True}


def _metrics(fold: int, map50: float, *, hyperbola_ap50: float | None) -> dict:
    per_class = []
    for class_id, name in enumerate(CLASS_NAMES):
        ap50 = hyperbola_ap50 if class_id == 0 else map50
        per_class.append(
            {
                "class_id": class_id,
                "name": name,
                "instances": 10 + fold,
                "images": 5,
                "ap50": ap50,
                "ap50_95": None if ap50 is None else ap50 / 2,
            }
        )
    return {
        "fold": str(fold),
        "unit": "crop",
        "weights": {"name": "last"},
        "class_names": CLASS_NAMES,
        "eval_images": {"count": 1000 + fold},
        "per_class": per_class,
        "map50": map50,
        "map50_95": map50 / 2,
        "pipe_ap50_mean": {"value": map50 - 0.05, "classes": ["pipe_band"]},
        "settings": {**EVAL_SETTINGS, "device": "0"},
    }


def _write_fold_run(
    root: Path, fold: int, *, training_block: dict | None = None
) -> Path:
    run_dir = root / f"run_fold{fold}"
    eval_dir = run_dir / "eval" / "last_crop"
    eval_dir.mkdir(parents=True)
    block = {
        **(training_block or TRAINING_BLOCK),
        "name": run_dir.name,
        "data": f"data_fold{fold}.yaml",
    }
    settings = {"fold": str(fold), "ultralytics": block}
    (run_dir / "gpr_train_settings.yaml").write_text(
        yaml.safe_dump(settings), encoding="utf-8"
    )
    hyperbola = None if fold == 3 else 0.6
    metrics = _metrics(fold, MAP50_BY_FOLD[fold], hyperbola_ap50=hyperbola)
    (eval_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run_dir


@pytest.fixture
def aggregate_script(load_script) -> ModuleType:
    return load_script("aggregate_folds")


def _run(module: ModuleType, runs: list[Path], out: Path) -> int:
    return module.main(["--runs", *map(str, runs), "--out", str(out)])


def test_summary_statistics_over_five_folds(aggregate_script, tmp_path: Path) -> None:
    runs = [_write_fold_run(tmp_path, fold) for fold in (4, 2, 0, 3, 1)]
    out = tmp_path / "summary" / "m110_fold_summary"

    assert _run(aggregate_script, runs, out) == 0

    document = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert [row["fold"] for row in document["folds"]] == ["0", "1", "2", "3", "4"]
    assert document["folds"][1]["images"] == 1001
    assert document["folds"][1]["instances"] == 4 * 11
    assert document["folds"][0]["seed"] == 0
    map50 = document["summary"]["map50"]
    assert map50["mean"] == pytest.approx(statistics.fmean(MAP50_BY_FOLD))
    assert map50["std"] == pytest.approx(statistics.stdev(MAP50_BY_FOLD))
    assert (map50["min"], map50["max"]) == (min(MAP50_BY_FOLD), max(MAP50_BY_FOLD))
    assert map50["n"] == 5
    hyperbola = document["summary"]["ap50_pipe_hyperbola"]
    assert hyperbola["n"] == 4 and hyperbola["std"] == pytest.approx(0.0)
    assert "name" not in document["ultralytics"]

    with out.with_suffix(".csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["fold"] for row in rows] == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "mean",
        "std",
        "min",
        "max",
    ]
    assert float(rows[6]["map50"]) == pytest.approx(statistics.stdev(MAP50_BY_FOLD))


@pytest.mark.parametrize(
    "folds", [(0, 1, 2, 3), (0, 1, 2, 3, 3)], ids=["missing", "duplicate"]
)
def test_rejects_wrong_fold_set(aggregate_script, tmp_path: Path, folds) -> None:
    runs = []
    for index, fold in enumerate(folds):
        runs.append(_write_fold_run(tmp_path / str(index), fold))
    out = tmp_path / "summary"

    assert _run(aggregate_script, runs, out) == 1
    assert not out.with_suffix(".json").exists()


def test_rejects_different_training_settings(aggregate_script, tmp_path: Path) -> None:
    runs = [_write_fold_run(tmp_path, fold) for fold in range(4)]
    runs.append(
        _write_fold_run(tmp_path, 4, training_block={**TRAINING_BLOCK, "epochs": 30})
    )
    assert _run(aggregate_script, runs, tmp_path / "summary") == 1


def test_rejects_different_eval_settings(aggregate_script, tmp_path: Path) -> None:
    runs = [_write_fold_run(tmp_path, fold) for fold in range(5)]
    metrics_path = runs[2] / "eval" / "last_crop" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["settings"]["conf"] = 0.25
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    assert _run(aggregate_script, runs, tmp_path / "summary") == 1


def test_device_difference_is_allowed(aggregate_script, tmp_path: Path) -> None:
    runs = [_write_fold_run(tmp_path, fold) for fold in range(5)]
    metrics_path = runs[0] / "eval" / "last_crop" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["settings"]["device"] = None
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    assert _run(aggregate_script, runs, tmp_path / "summary") == 0


def test_missing_metrics_file_is_an_error(aggregate_script, tmp_path: Path) -> None:
    runs = [_write_fold_run(tmp_path, fold) for fold in range(5)]
    (runs[1] / "eval" / "last_crop" / "metrics.json").unlink()
    assert _run(aggregate_script, runs, tmp_path / "summary") == 1


def test_rejects_different_confusion_conf(aggregate_script, tmp_path: Path) -> None:
    runs = [_write_fold_run(tmp_path, fold) for fold in range(5)]
    metrics_path = runs[4] / "eval" / "last_crop" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["settings"]["confusion_conf"] = 0.25
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    assert _run(aggregate_script, runs, tmp_path / "summary") == 1
