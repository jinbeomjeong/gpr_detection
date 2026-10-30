"""Tests of ``scripts/train_detector.py --on-existing`` and the run settings file.

Training is replaced by a fake trainer; every folder lives in ``tmp_path``.
"""

from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
M110_CONFIG = PROJECT_ROOT / "configs" / "train_yolo11m_m110.yaml"
RUN_NAME = "fold0_run"
EPOCHS = 50
RESULTS_HEADER = "epoch,time,metrics/mAP50(B)\n"
TRAIN_IMAGES = 3
VAL_IMAGES = 2


class FakeTrainer:
    """Stands in for the Ultralytics trainer: records whether it trained."""

    def __init__(self, save_dir: Path) -> None:
        self.save_dir = save_dir
        self.trained = False

    def train(self) -> None:
        self.trained = True


@pytest.fixture
def train_script(load_script, tmp_path: Path, monkeypatch) -> ModuleType:
    module = load_script("train_detector")
    runs_dir = tmp_path / "runs" / "detect"
    monkeypatch.setattr(module, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(module, "DATA_YAML_DIR", runs_dir / "_data")
    module.trainers = []

    def fake_build_trainer(overrides, augment_config, augment_seed) -> FakeTrainer:
        trainer = FakeTrainer(Path(overrides["project"]) / overrides["name"])
        module.trainers.append(trainer)
        return trainer

    monkeypatch.setattr(module, "build_gpr_trainer", fake_build_trainer)
    return module


@pytest.fixture
def data_yaml(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    train_lines = [f"./images/t{i}.jpg" for i in range(TRAIN_IMAGES)]
    (dataset / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_lines = [f"./images/v{i}.jpg" for i in range(VAL_IMAGES)]
    (dataset / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    data = {
        "path": str(dataset),
        "train": ["train.txt"],
        "val": "val.txt",
        "names": {0: "pipe_hyperbola"},
    }
    path = dataset / "data.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _make_run(run_dir: Path, result_rows: int, *, has_last: bool) -> None:
    run_dir.mkdir(parents=True)
    rows = "".join(f"{epoch},1.0,0.5\n" for epoch in range(1, result_rows + 1))
    (run_dir / "results.csv").write_text(RESULTS_HEADER + rows, encoding="utf-8")
    if has_last:
        (run_dir / "weights").mkdir()
        (run_dir / "weights" / "last.pt").write_bytes(b"weights")


def _main(module: ModuleType, data_yaml: Path, *extra: str) -> int:
    return module.main(
        [
            "--config",
            str(M110_CONFIG),
            "--data",
            str(data_yaml),
            "--name",
            RUN_NAME,
            "--epochs",
            str(EPOCHS),
            *extra,
        ]
    )


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("result_rows", "has_last", "expected"),
    [
        (49, True, "incomplete"),
        (50, True, "complete"),
        (51, True, "complete"),
        (50, False, "incomplete"),
        (0, True, "incomplete"),
    ],
)
def test_run_state_boundaries(
    train_script, tmp_path: Path, result_rows: int, has_last: bool, expected: str
) -> None:
    run_dir = tmp_path / "run"
    _make_run(run_dir, result_rows, has_last=has_last)
    assert train_script.run_state(run_dir, EPOCHS).value == expected


def test_run_state_absent_and_missing_results(train_script, tmp_path: Path) -> None:
    assert train_script.run_state(tmp_path / "nothing", EPOCHS).value == "absent"
    run_dir = tmp_path / "only_weights"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "last.pt").write_bytes(b"w")
    assert train_script.run_state(run_dir, EPOCHS).value == "incomplete"


def test_new_run_trains_and_records_provenance(train_script, data_yaml: Path) -> None:
    assert _main(train_script, data_yaml) == 0

    assert [trainer.trained for trainer in train_script.trainers] == [True]
    settings_path = train_script.RUNS_DIR / RUN_NAME / "gpr_train_settings.yaml"
    settings: dict[str, Any] = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    assert settings["image_counts"] == {"train": TRAIN_IMAGES, "val": VAL_IMAGES}
    assert set(settings["git"]) == {"commit", "dirty"}
    assert len(settings["git"]["commit"]) == 40
    assert set(settings["versions"]) == {"torch", "ultralytics"}
    assert settings["started_at"]
    assert settings["ultralytics"]["epochs"] == EPOCHS


@pytest.mark.parametrize(("result_rows", "has_last"), [(50, True), (10, True)])
def test_error_mode_refuses_any_existing_folder(
    train_script, data_yaml: Path, result_rows: int, has_last: bool
) -> None:
    run_dir = train_script.RUNS_DIR / RUN_NAME
    _make_run(run_dir, result_rows, has_last=has_last)
    before = _snapshot(run_dir)

    assert _main(train_script, data_yaml) == 2

    assert train_script.trainers == []
    assert _snapshot(run_dir) == before


def test_skip_complete_skips_complete_run(train_script, data_yaml: Path) -> None:
    run_dir = train_script.RUNS_DIR / RUN_NAME
    _make_run(run_dir, EPOCHS, has_last=True)
    before = _snapshot(run_dir)

    assert _main(train_script, data_yaml, "--on-existing", "skip-complete") == 0

    assert train_script.trainers == []
    assert _snapshot(run_dir) == before


def test_skip_complete_refuses_incomplete_run(train_script, data_yaml: Path) -> None:
    run_dir = train_script.RUNS_DIR / RUN_NAME
    _make_run(run_dir, EPOCHS - 1, has_last=True)
    before = _snapshot(run_dir)

    assert _main(train_script, data_yaml, "--on-existing", "skip-complete") == 2

    assert train_script.trainers == []
    assert _snapshot(run_dir) == before


def test_archive_moves_incomplete_run_and_trains(train_script, data_yaml: Path) -> None:
    run_dir = train_script.RUNS_DIR / RUN_NAME
    _make_run(run_dir, EPOCHS, has_last=False)
    before = _snapshot(run_dir)

    assert _main(train_script, data_yaml, "--on-existing", "archive") == 0

    archived = list((train_script.RUNS_DIR / "_incomplete").iterdir())
    assert len(archived) == 1
    assert archived[0].name.startswith(f"{RUN_NAME}_")
    stamp = archived[0].name.removeprefix(f"{RUN_NAME}_")
    assert len(stamp) == len("20261030T120000") and stamp[8] == "T"
    assert _snapshot(archived[0]) == before
    assert [trainer.trained for trainer in train_script.trainers] == [True]
    assert not (run_dir / "results.csv").exists()


def test_archive_leaves_complete_run_alone(train_script, data_yaml: Path) -> None:
    run_dir = train_script.RUNS_DIR / RUN_NAME
    _make_run(run_dir, EPOCHS, has_last=True)
    before = _snapshot(run_dir)

    assert _main(train_script, data_yaml, "--on-existing", "archive") == 0

    assert train_script.trainers == []
    assert _snapshot(run_dir) == before
    assert not (train_script.RUNS_DIR / "_incomplete").exists()


def test_dry_run_reports_state_and_changes_nothing(
    train_script, data_yaml: Path, capsys
) -> None:
    run_dir = train_script.RUNS_DIR / RUN_NAME
    _make_run(run_dir, 3, has_last=False)

    assert _main(train_script, data_yaml, "--on-existing", "archive", "--dry-run") == 0

    assert '"existing_run": "incomplete"' in capsys.readouterr().out
    assert run_dir.is_dir()
    assert train_script.trainers == []


def test_unexpected_save_dir_stops_before_training(
    train_script, data_yaml: Path, monkeypatch
) -> None:
    def renamed_trainer(overrides, augment_config, augment_seed) -> FakeTrainer:
        trainer = FakeTrainer(Path(overrides["project"]) / f"{overrides['name']}2")
        train_script.trainers.append(trainer)
        return trainer

    monkeypatch.setattr(train_script, "build_gpr_trainer", renamed_trainer)

    with pytest.raises(RuntimeError, match="instead of"):
        _main(train_script, data_yaml)
    assert [trainer.trained for trainer in train_script.trainers] == [False]
