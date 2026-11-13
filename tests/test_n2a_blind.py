"""N2-a blind outputs and the A1 judge: tests B-1 and B-2.

docs/experiments/m31_n2a_retrain_plan.md v0.2 section 4-2, 5-5, 5-6.

B-1 trains a tiny one-channel YOLO11n for one epoch on CPU (four synthetic
64 px images, geometric augmentations on with sizes scaled to 64 px), then
evaluates and exports it, all with ``--blind``, and checks that stdout and
stderr carry no metric words or numeric tables. Every file lives in a
temporary folder outside ``tmp_path``: the export script refuses any path
with "test" in it (sealed test hold-out), and pytest's base folder is named
``pytest-of-<user>``.

B-2 judges synthetic metrics with ``n2a_a1_judge.py`` against a throw-away git
repository that holds the sealed config.
"""

import csv
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "detection"
N2A_CONFIG = PROJECT_ROOT / "configs" / "train_yolo11m_n2a.yaml"
CLASS_NAMES = {0: "pipe_hyperbola", 1: "pipe_band", 2: "cavity", 3: "manhole"}
METRIC_WORDS = re.compile(r"map|ap50|recall|precision", re.IGNORECASE)
NUMERIC_TABLE_ROW = re.compile(r"(\d+\.\d+\s+){3,}")
FOLDS = (0, 1, 2, 3, 4)
SEALED_MARKER = "test"
IMAGE_SIZE_PX = 64
RUN_NAME = "n2a_blind_fold0"


def _load_module(path: Path, name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _assert_blind(text: str, root: Path) -> None:
    """No metric words or numeric table rows; last line is the status JSON."""
    scrubbed = text.replace(str(root), "<ROOT>").replace(root.as_posix(), "<ROOT>")
    assert not METRIC_WORDS.search(scrubbed), scrubbed
    assert not NUMERIC_TABLE_ROW.search(scrubbed), scrubbed


def _last_json_line(stdout: str) -> dict:
    return json.loads(stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------- B-1


@pytest.fixture
def blind_root() -> Iterator[Path]:
    root = Path(tempfile.mkdtemp(prefix="gpr_n2a_b1_"))
    if any(SEALED_MARKER in part.lower() for part in root.parts):
        shutil.rmtree(root)
        pytest.skip(f"temporary folder {root} contains '{SEALED_MARKER}'")
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _write_dataset(root: Path) -> Path:
    """Fold-0 dataset: images, labels, fold lists, fold yaml and manifest."""
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    rng = np.random.default_rng(0)
    names = []
    for index in range(4):
        name = f"img{index}.png"
        image = rng.integers(60, 120, (IMAGE_SIZE_PX, IMAGE_SIZE_PX), dtype=np.uint8)
        image[20:40, 16:48] = 230
        cv2.imwrite(str(images / name), image)
        (labels / f"img{index}.txt").write_text(
            f"{index % 4} 0.5 0.47 0.5 0.31\n", encoding="utf-8"
        )
        names.append(name)
    listing = "\n".join(f"./images/{name}" for name in names) + "\n"
    (root / "train_images.txt").write_text(listing, encoding="utf-8")
    (root / "fold0_images.txt").write_text(listing, encoding="utf-8")
    (root / "fold0.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": ["train_images.txt"],
                "val": "fold0_images.txt",
                "names": CLASS_NAMES,
            }
        ),
        encoding="utf-8",
    )
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_name", "fold", "gain", "crop_key"])
        for index, name in enumerate(names):
            writer.writerow([name, "0", "1.0", f"crop{index}"])
    return root


def _write_tiny_config(path: Path, dataset_dir: Path) -> Path:
    config = yaml.safe_load(N2A_CONFIG.read_text(encoding="utf-8"))
    config["model"] = "yolo11n.yaml"
    config["dataset_dir"] = str(dataset_dir)
    config["ultralytics"].update(imgsz=IMAGE_SIZE_PX, plots=False, amp=False)
    # The N2-a sizes (96 px shift, 128 px band) need 256 px images; scale to 64.
    config["gpr_geometric"]["horizontal_shift"]["max_shift_px"] = 16
    config["gpr_geometric"]["edge_fill"].update(
        width_min_px=4, width_max_px=16, height_min_px=4, height_max_px=24
    )
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


BLIND_CPU_FLAGS = ("--device", "cpu", "--blind")


def _train_blind(load_script, monkeypatch, root: Path, config: Path) -> Path:
    train_script = load_script("train_detector")
    runs_dir = root / "runs" / "detect"
    monkeypatch.setattr(train_script, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(train_script, "DATA_YAML_DIR", runs_dir / "_data")
    arguments = [
        *("--config", str(config), "--fold", "0", "--name", RUN_NAME),
        *("--epochs", "1", "--batch", "4", "--workers", "0"),
        *BLIND_CPU_FLAGS,
    ]
    assert train_script.main(arguments) == 0
    return runs_dir / RUN_NAME


def _evaluate_blind(load_script, run_dir: Path, manifest: Path) -> None:
    eval_script = load_script("eval_detector")
    arguments = [
        *("--run-dir", str(run_dir), "--weights", "last", "--unit", "crop"),
        *("--manifest", str(manifest), "--imgsz", "64", "--batch", "4"),
        *BLIND_CPU_FLAGS,
    ]
    assert eval_script.main(arguments) == 0


def _export_blind(monkeypatch, run_dir: Path, export_dir: Path) -> None:
    export_script = _load_module(
        ANALYSIS_DIR / "export_val_predictions.py", "n2a_export", monkeypatch
    )
    data_yaml = run_dir / "eval" / "last_crop" / "eval_data.yaml"
    arguments = [
        *("--weights", str(run_dir / "weights" / "last.pt")),
        *("--data-yaml", str(data_yaml), "--out-dir", str(export_dir)),
        *("--imgsz", "64", "--batch", "4"),
        *BLIND_CPU_FLAGS,
    ]
    assert export_script.main(arguments) == 0


def _progress_lines(stderr: str) -> list[dict]:
    return [json.loads(line) for line in stderr.splitlines() if '"stage"' in line]


def test_b1_train_eval_export_blind_outputs(
    blind_root: Path, load_script, monkeypatch, capfd
) -> None:
    pytest.importorskip("ultralytics")
    dataset = _write_dataset(blind_root / "dataset")
    config = _write_tiny_config(blind_root / "tiny_n2a.yaml", dataset)
    export_dir = blind_root / "export" / "fold0_crop"

    run_dir = _train_blind(load_script, monkeypatch, blind_root, config)
    train_out, train_err = capfd.readouterr()
    _evaluate_blind(load_script, run_dir, dataset / "manifest.csv")
    eval_out, eval_err = capfd.readouterr()
    _export_blind(monkeypatch, run_dir, export_dir)
    export_out, export_err = capfd.readouterr()

    ok = {"status": "ok", "exit_code": 0}
    assert _last_json_line(train_out) == {**ok, "out": str(run_dir)}
    assert _last_json_line(eval_out) == {**ok, "out": str(run_dir / "eval")}
    assert _last_json_line(export_out) == {**ok, "out": str(export_dir)}
    train_progress = _progress_lines(train_err)
    assert [
        (line["stage"], line["fold"], line["epoch"]) for line in train_progress
    ] == [("train", 0, 1)]
    assert [line["stage"] for line in _progress_lines(eval_err)] == ["eval"]
    assert [line["stage"] for line in _progress_lines(export_err)] == ["export"]
    for produced in (
        run_dir / "aug_stats.json",
        run_dir / "train_record.json",
        run_dir / "eval" / "last_crop" / "metrics.json",
        export_dir / "val_metrics.json",
    ):
        assert produced.is_file(), produced
    for text in (train_out, train_err, eval_out, eval_err, export_out, export_err):
        _assert_blind(text, blind_root)


# ------------------------------------------------------------------- B-2


GOOD_ARGS = {"scale": 0.0, "translate": 0.1, "flipud": 0.0, "epochs": 50, "seed": 0}
N2A_PREFIX = "yolo11m_aihubv1n2a_20261114_fold"
M110_PREFIX = "yolo11m_aihubv1_20261101_fold"


@pytest.fixture
def judge_script(monkeypatch) -> ModuleType:
    return _load_module(ANALYSIS_DIR / "n2a_a1_judge.py", "n2a_a1_judge", monkeypatch)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _metrics(fold: int, weights: Path, eval_list: Path, map50: float, hyp: float):
    return {
        "fold": str(fold),
        "unit": "crop",
        "weights": {"name": "last", "sha256": _sha256(weights)},
        "eval_images": {"sha256": _sha256(eval_list), "count": 2},
        "map50": map50,
        "map50_95": map50 / 2,
        "per_class": [
            {"name": "pipe_hyperbola", "ap50": hyp},
            {"name": "pipe_band", "ap50": 0.5},
            {"name": "cavity", "ap50": 0.6},
            {"name": "manhole", "ap50": 0.7},
        ],
    }


def _build_judge_inputs(
    root: Path,
    map50: float = 0.8,
    hyperbola: float = 0.95,
    args_override: dict | None = None,
) -> tuple[Path, Path, str]:
    """Synthetic N2-a and M1-10 run folders plus a git repo with the config."""
    repo = root / "repo"
    (repo / "configs").mkdir(parents=True)
    config = repo / "configs" / "n2a.yaml"
    config.write_text("ultralytics: {scale: 0.0}\n", encoding="utf-8", newline="\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "configs/n2a.yaml")
    _git(repo, "commit", "-q", "-m", "seal")
    seal = _git(repo, "rev-parse", "HEAD")

    run_root = root / "runs"
    for k in FOLDS:
        run = run_root / f"{N2A_PREFIX}{k}"
        weights = run / "weights" / "last.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(f"weights fold {k}".encode())
        eval_list = run / "eval" / "last_crop" / "eval_images.txt"
        eval_list.parent.mkdir(parents=True)
        eval_list.write_text(f"a{k}.png\nb{k}.png\n", encoding="utf-8", newline="\n")
        _write_json(
            eval_list.parent / "metrics.json",
            _metrics(k, weights, eval_list, map50, hyperbola),
        )
        m110_metrics = _metrics(k, weights, eval_list, 0.79, 0.93)
        _write_json(
            run_root / f"{M110_PREFIX}{k}" / "eval" / "last_crop" / "metrics.json",
            m110_metrics,
        )
        args = dict(GOOD_ARGS)
        if args_override and k in args_override:
            args.update(args_override[k])
        (run / "args.yaml").write_text(yaml.safe_dump(args), encoding="utf-8")
        (run / "gpr_train_settings.yaml").write_text(
            yaml.safe_dump({"fold": str(k), "config_sha256": _sha256(config)}),
            encoding="utf-8",
        )
        _write_json(
            run / "train_record.json",
            {"weights_sha256": {"last": _sha256(weights), "best": None}},
        )
    return run_root, config, seal


def _run_judge(judge_script, root: Path, out: Path, *extra: str) -> int:
    run_root, config, seal = _build_judge_inputs(root)
    return _judge(judge_script, run_root, config, seal, out, *extra)


def _judge(judge_script, run_root, config, seal, out, *extra: str) -> int:
    arguments = [
        *("--run-prefix", N2A_PREFIX, "--m110-prefix", M110_PREFIX),
        *("--run-root", str(run_root), "--config", str(config)),
        *("--seal-commit", seal, "--repo", str(config.parents[1])),
        *("--out", str(out), "--blind", *extra),
    ]
    return judge_script.main(arguments)


def _verdict(out: Path) -> dict:
    return json.loads(out.read_text(encoding="utf-8"))


def test_b2_passing_inputs_are_judged(judge_script, tmp_path, capsys) -> None:
    out = tmp_path / "verdict.json"
    assert _run_judge(judge_script, tmp_path, out) == 0
    stdout = capsys.readouterr().out
    assert stdout.strip().count("\n") == 0  # status line only
    assert _last_json_line(stdout) == {"status": "ok", "exit_code": 0, "out": str(out)}
    verdict = _verdict(out)
    assert verdict["gates_pass"] and verdict["verdict"] == "A1 met"
    assert verdict["a1"]["map50"]["mean_pp_exact"] == "80"
    assert b"\r\n" not in out.read_bytes()
    assert set(verdict["run_identity"]["weights_sha256"]) == {"0", "1", "2", "3", "4"}


def test_b2_eval_list_mismatch_fails_gate(judge_script, tmp_path) -> None:
    run_root, config, seal = _build_judge_inputs(tmp_path)
    m110 = run_root / f"{M110_PREFIX}2" / "eval" / "last_crop" / "metrics.json"
    document = json.loads(m110.read_text(encoding="utf-8"))
    document["eval_images"]["sha256"] = "0" * 64
    m110.write_text(json.dumps(document), encoding="utf-8")
    out = tmp_path / "verdict.json"
    assert _judge(judge_script, run_root, config, seal, out) == 1
    verdict = _verdict(out)
    assert verdict["failures"] == ["fold 2"]
    assert verdict["verdict"] == "no verdict: gate failed"
    assert "a1" not in verdict


def test_b2_args_scale_not_zero_fails_gate(judge_script, tmp_path) -> None:
    run_root, config, seal = _build_judge_inputs(
        tmp_path, args_override={4: {"scale": 0.2}}
    )
    out = tmp_path / "verdict.json"
    assert _judge(judge_script, run_root, config, seal, out) == 1
    fold4 = _verdict(out)["gates"]["per_fold"][4]
    assert fold4["checks"]["args_scale_is_0"] is False


def test_b2_existing_output_is_refused(judge_script, tmp_path) -> None:
    out = tmp_path / "verdict.json"
    out.write_text("{}", encoding="utf-8")
    assert _run_judge(judge_script, tmp_path, out) == 2
    assert out.read_text(encoding="utf-8") == "{}"


def test_b2_changed_config_fails_gate(judge_script, tmp_path) -> None:
    run_root, config, seal = _build_judge_inputs(tmp_path)
    config.write_text("ultralytics: {scale: 0.2}\n", encoding="utf-8", newline="\n")
    out = tmp_path / "verdict.json"
    assert _judge(judge_script, run_root, config, seal, out) == 1
    assert _verdict(out)["failures"] == ["config file"]


@pytest.mark.parametrize(
    ("map50", "hyperbola", "is_met"),
    [
        (0.768, 0.95, True),  # mean exactly 76.8 passes
        (0.7679999, 0.95, False),  # 76.79999 fails, no rounding
        (0.8, 0.909, True),  # mean exactly 90.9 passes
        (0.8, 0.9089999, False),
    ],
)
def test_b2_threshold_boundary_without_rounding(
    judge_script, tmp_path, map50, hyperbola, is_met
) -> None:
    run_root, config, seal = _build_judge_inputs(tmp_path, map50, hyperbola)
    out = tmp_path / "verdict.json"
    assert _judge(judge_script, run_root, config, seal, out) == 0
    verdict = _verdict(out)
    assert verdict["a1"]["pass"] is is_met
    assert verdict["verdict"] == ("A1 met" if is_met else "A1 not met")
