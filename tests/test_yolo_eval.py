"""Tests of the M1-10 evaluation path without a GPU or Ultralytics run.

* crop-level image selection (``gpr_detection.data.splits``)
* conversion of a stand-in Ultralytics metrics object to plain values
* ``scripts/eval_detector.py`` with the validation call replaced by a fake
"""

import csv
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

from gpr_detection.data.splits import select_crop_representatives
from gpr_detection.detect.yolo_eval import metrics_to_dict
from gpr_detection.run_info import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AIHUB_MANIFEST = PROJECT_ROOT / "data" / "splits" / "aihub_yolo_v1.csv"
CLASS_NAMES = {0: "pipe_hyperbola", 1: "pipe_band", 2: "cavity", 3: "manhole"}
FOLD0_CROPS = 1298
FOLD0_GAIN_PICKS = {"1.0": 1160, "0.5": 130, "0.25": 8}
HIGH_CONFUSION_CONF = 0.25


def _row(crop: str, gain: str, fold: str = "0") -> dict[str, str]:
    return {
        "file_name": f"{crop}_g{gain}.jpg",
        "crop_key": crop,
        "gain": gain,
        "fold": fold,
    }


# --- crop representatives -------------------------------------------------------


def test_crop_selection_prefers_high_gain_and_handles_missing_gains() -> None:
    rows = [
        _row("a", "0.25"),
        _row("a", "0.5"),
        _row("a", "1.0"),
        _row("b", "0.25"),
        _row("b", "0.5"),  # no 1.0 copy
        _row("c", "0.25"),  # only the lowest gain
    ]
    chosen = select_crop_representatives(rows)
    assert [(row["crop_key"], row["gain"]) for row in chosen] == [
        ("a", "1.0"),
        ("b", "0.5"),
        ("c", "0.25"),
    ]


def test_crop_selection_is_independent_of_input_order() -> None:
    rows = [
        _row(f"crop{i:03d}", gain) for i in range(60) for gain in ("0.25", "0.5", "1.0")
    ]
    expected = select_crop_representatives(rows)
    shuffled = rows.copy()
    random.Random(7).shuffle(shuffled)
    assert select_crop_representatives(shuffled) == expected
    file_names = [row["file_name"] for row in expected]
    assert file_names == sorted(file_names)


def test_crop_selection_custom_priority_and_ties() -> None:
    rows = [
        _row("a", "1"),  # numeric match with "1.0"
        _row("b", "0.25"),
        {**_row("c", "0.5"), "file_name": "c_second.jpg"},
        {**_row("c", "0.5"), "file_name": "c_first.jpg"},
    ]
    chosen = select_crop_representatives(rows, gain_priority=("1.0", "0.5"))
    assert [row["file_name"] for row in chosen] == ["a_g1.jpg", "c_first.jpg"]
    with pytest.raises(ValueError, match="at least one gain"):
        select_crop_representatives(rows, gain_priority=())


def test_fold0_crop_level_matches_m17_counts() -> None:
    if not AIHUB_MANIFEST.is_file():
        pytest.skip(f"derived manifest not built: {AIHUB_MANIFEST}")
    with AIHUB_MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        fold0 = [row for row in csv.DictReader(handle) if row["fold"] == "0"]
    chosen = select_crop_representatives(fold0)
    assert len(chosen) == FOLD0_CROPS
    assert Counter(row["gain"] for row in chosen) == FOLD0_GAIN_PICKS


# --- metrics conversion ---------------------------------------------------------


def _fake_metrics(*, with_matrix: bool = True) -> SimpleNamespace:
    """DetMetrics stand-in: pipe_hyperbola has no ground truth."""
    box = SimpleNamespace(
        p=np.array([0.8, 0.9, 0.7]),
        r=np.array([0.6, 0.85, 0.65]),
        ap50=np.array([0.5, 0.95, 0.75]),
        ap=np.array([0.2, 0.6, 0.4]),
        map50=np.mean([0.5, 0.95, 0.75]),
        map=np.mean([0.2, 0.6, 0.4]),
    )
    matrix = np.arange(25, dtype=float).reshape(5, 5)
    return SimpleNamespace(
        names=CLASS_NAMES,
        ap_class_index=np.array([1, 2, 3]),
        nt_per_class=np.array([0, 10, 4, 6]),
        nt_per_image=np.array([0, 7, 4, 5]),
        box=box,
        confusion_matrix=SimpleNamespace(matrix=matrix) if with_matrix else None,
        save_dir=Path("unused"),
    )


def test_metrics_to_dict_maps_classes_and_nulls_absent_ones() -> None:
    result = metrics_to_dict(_fake_metrics(), conf=0.001)

    by_name = {row["name"]: row for row in result["per_class"]}
    hyperbola = by_name["pipe_hyperbola"]
    assert hyperbola["instances"] == 0
    assert all(
        hyperbola[key] is None for key in ("precision", "recall", "ap50", "ap50_95")
    )
    assert by_name["pipe_band"]["ap50"] == pytest.approx(0.5)
    assert by_name["cavity"]["precision"] == pytest.approx(0.9)
    assert by_name["manhole"]["ap50_95"] == pytest.approx(0.4)
    assert by_name["pipe_band"]["images"] == 7
    assert result["classes_present"] == [1, 2, 3]
    assert result["map50"] == pytest.approx((0.5 + 0.95 + 0.75) / 3)
    assert result["pipe_ap50_mean"] == {
        "value": pytest.approx(0.5),
        "classes": ["pipe_band"],
    }
    json.dumps(result)  # plain values only


def test_metrics_to_dict_pipe_mean_over_both_and_confusion_axes() -> None:
    metrics = _fake_metrics()
    metrics.ap_class_index = np.array([0, 1, 2])
    metrics.nt_per_class = np.array([3, 10, 4, 0])
    result = metrics_to_dict(metrics, conf=0.25)

    assert result["pipe_ap50_mean"]["value"] == pytest.approx((0.5 + 0.95) / 2)
    assert result["per_class"][3]["ap50"] is None
    confusion = result["confusion_matrix"]
    assert confusion["labels"] == [*CLASS_NAMES.values(), "background"]
    assert confusion["rows"].startswith("predicted")
    assert confusion["columns"].startswith("true")
    assert confusion["conf_threshold"] == 0.25
    assert confusion["matrix"][4][0] == 20


def test_metrics_to_dict_without_plots_or_classes() -> None:
    assert (
        metrics_to_dict(_fake_metrics(), conf=0.001, has_plots=False)[
            "confusion_matrix"
        ]
        is None
    )
    empty = _fake_metrics(with_matrix=False)
    empty.ap_class_index = np.array([], dtype=int)
    result = metrics_to_dict(empty, conf=0.001)
    assert result["map50"] is None
    assert result["pipe_ap50_mean"] == {"value": None, "classes": []}


def test_yolo_eval_import_does_not_load_ultralytics() -> None:
    probe = (
        "import sys, gpr_detection.detect.yolo_eval\n"
        "print(','.join(n for n in ('ultralytics', 'torch') if n in sys.modules))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert completed.stdout.strip() == ""


# --- scripts/eval_detector.py ---------------------------------------------------


@pytest.fixture
def eval_script(load_script, monkeypatch) -> ModuleType:
    module = load_script("eval_detector")
    module.calls = []

    def fake_evaluate(weights, data_yaml, out_dir, **settings):
        module.calls.append((Path(weights), Path(data_yaml), Path(out_dir), settings))
        metrics = _fake_metrics()
        if settings["conf"] >= HIGH_CONFUSION_CONF:  # tell the two runs apart
            metrics.confusion_matrix.matrix = np.ones((5, 5))
        result = metrics_to_dict(metrics, conf=settings["conf"])
        return {**result, "checkpoint_epoch": 49, "save_dir": str(out_dir)}

    monkeypatch.setattr(module, "evaluate_yolo", fake_evaluate)
    return module


def _write_fold_dataset(root: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
    """Dataset folder (fold list + yaml) and manifest for fold 0."""
    dataset = root / "dataset"
    dataset.mkdir()
    fold0 = sorted(row["file_name"] for row in rows if row["fold"] == "0")
    (dataset / "fold0_images.txt").write_text(
        "".join(f"./images/{name}\n" for name in fold0), encoding="utf-8"
    )
    fold_yaml = {"path": str(dataset), "train": [], "val": "fold0_images.txt"}
    fold_yaml["names"] = CLASS_NAMES
    (dataset / "fold0.yaml").write_text(yaml.safe_dump(fold_yaml), encoding="utf-8")
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dataset, manifest


def _write_run(root: Path, dataset: Path, fold: str = "0") -> Path:
    run_dir = root / "run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "last.pt").write_bytes(b"last weights")
    settings = {"fold": fold, "source_data_yaml": str(dataset / "fold0.yaml")}
    (run_dir / "gpr_train_settings.yaml").write_text(
        yaml.safe_dump(settings), encoding="utf-8"
    )
    return run_dir


@pytest.fixture
def eval_inputs(tmp_path: Path) -> tuple[Path, Path]:
    rows = [
        _row("a", "0.25"),
        _row("a", "1.0"),
        _row("b", "0.5"),
        _row("x", "1.0", fold="1"),
    ]
    dataset, manifest = _write_fold_dataset(tmp_path, rows)
    return _write_run(tmp_path, dataset), manifest


def test_eval_script_writes_numbers_per_weights_and_unit(
    eval_script, eval_inputs
) -> None:
    run_dir, manifest = eval_inputs
    code = eval_script.main(
        ["--run-dir", str(run_dir), "--weights", "last", "--manifest", str(manifest)]
    )
    assert code == 0
    assert [call[2].name for call in eval_script.calls] == ["last_crop", "last_image"]

    crop_dir = run_dir / "eval" / "last_crop"
    metrics = json.loads((crop_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["fold"] == "0" and metrics["unit"] == "crop"
    assert metrics["eval_images"]["count"] == 2
    assert metrics["eval_images"]["gain_counts"] == {"0.5": 1, "1.0": 1}
    assert metrics["eval_images"]["sha256"] == sha256_file(crop_dir / "eval_images.txt")
    assert metrics["weights"]["sha256"] == sha256_file(run_dir / "weights" / "last.pt")
    assert metrics["weights"]["epochs_completed"] == 50
    assert metrics["pipe_ap50_mean"]["value"] == pytest.approx(0.5)
    assert set(metrics["versions"]) >= {"torch", "ultralytics", "python"}
    assert len(metrics["git"]["commit"]) == 40

    listed = (crop_dir / "eval_images.txt").read_bytes().decode("utf-8").splitlines()
    assert [Path(line).name for line in listed] == ["a_g1.0.jpg", "b_g0.5.jpg"]
    assert b"\r\n" not in (crop_dir / "eval_images.txt").read_bytes()
    data_yaml_text = (crop_dir / "eval_data.yaml").read_text(encoding="utf-8")
    assert "never train" in data_yaml_text
    assert yaml.safe_load(data_yaml_text)["channels"] == 1

    with (crop_dir / "per_class.csv").open(encoding="utf-8", newline="") as handle:
        per_class = list(csv.DictReader(handle))
    assert [row["name"] for row in per_class] == list(CLASS_NAMES.values())
    assert per_class[0]["ap50"] == ""
    with (crop_dir / "confusion_matrix.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        confusion = list(csv.reader(handle))
    assert len(confusion) == 6 and confusion[0][-1] == "background"

    image_metrics = json.loads(
        (run_dir / "eval" / "last_image" / "metrics.json").read_text(encoding="utf-8")
    )
    assert image_metrics["eval_images"]["count"] == 3


def test_eval_script_refuses_test_fold(
    eval_script, eval_inputs, tmp_path: Path
) -> None:
    _, manifest = eval_inputs
    assert (
        eval_script.main(["--run-dir", str(tmp_path / "absent"), "--fold", "test"]) == 2
    )

    test_run = _write_run(tmp_path / "test_trained", tmp_path / "dataset", fold="test")
    assert (
        eval_script.main(["--run-dir", str(test_run), "--manifest", str(manifest)]) == 2
    )
    assert eval_script.calls == []


def test_eval_script_rejects_list_mismatch(eval_script, eval_inputs) -> None:
    run_dir, manifest = eval_inputs
    fold_list = Path(
        yaml.safe_load(
            (run_dir / "gpr_train_settings.yaml").read_text(encoding="utf-8")
        )["source_data_yaml"]
    ).with_name("fold0_images.txt")
    fold_list.write_text("./images/a_g1.0.jpg\n", encoding="utf-8")

    assert (
        eval_script.main(["--run-dir", str(run_dir), "--manifest", str(manifest)]) == 1
    )
    assert eval_script.calls == []


def test_confusion_conf_adds_second_matrix_without_touching_first(
    eval_script, eval_inputs
) -> None:
    run_dir, manifest = eval_inputs
    argv = ["--run-dir", str(run_dir), "--weights", "last", "--unit", "crop"]
    argv += ["--manifest", str(manifest), "--confusion-conf", "0.25"]
    assert eval_script.main(argv) == 0

    crop_dir = run_dir / "eval" / "last_crop"
    out_dirs = [call[2] for call in eval_script.calls]
    assert out_dirs == [crop_dir, crop_dir / "confusion_conf0.25"]
    assert [call[3]["conf"] for call in eval_script.calls] == [0.001, 0.25]

    metrics = json.loads((crop_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["conf_threshold"] == 0.001
    assert metrics["confusion_conf_threshold"] == 0.25
    assert metrics["settings"]["confusion_conf"] == 0.25
    assert metrics["confusion_matrix"]["conf_threshold"] == 0.001
    second = metrics["confusion_matrix_at_confusion_conf"]
    assert second["conf_threshold"] == 0.25 and second["matrix"][0][0] == 1
    assert metrics["map50"] == pytest.approx((0.5 + 0.95 + 0.75) / 3)

    with (crop_dir / "confusion_matrix.csv").open(encoding="utf-8") as handle:
        first_rows = list(csv.reader(handle))
    with (crop_dir / "confusion_matrix_conf0.25.csv").open(encoding="utf-8") as handle:
        second_rows = list(csv.reader(handle))
    assert first_rows[2][1:] == [str(value) for value in range(5, 10)]
    assert second_rows[2][1:] == ["1"] * 5


def test_without_confusion_conf_no_second_run(eval_script, eval_inputs) -> None:
    run_dir, manifest = eval_inputs
    argv = ["--run-dir", str(run_dir), "--weights", "last", "--unit", "crop"]
    assert eval_script.main([*argv, "--manifest", str(manifest)]) == 0

    crop_dir = run_dir / "eval" / "last_crop"
    metrics = json.loads((crop_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(eval_script.calls) == 1
    assert metrics["confusion_conf_threshold"] is None
    assert metrics["confusion_matrix_at_confusion_conf"] is None
    assert not list(crop_dir.glob("confusion_matrix_conf*.csv"))
