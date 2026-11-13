"""M3-1 D1 r2 scripts (m31_detection_input_tiling.md v0.5 §8-4, tests S1-S8).

Synthetic run folders cover the CLI contracts (dry labels, overwrite refusal,
LF output, blind stdout, identity block, judge exit codes and rule O order).
The ``data`` tests use the git-ignored M1-10 outputs (labels, eval lists and
the validator detections of ``runs/analysis/m110_band_aux``) and are skipped
when those are absent. No test runs a model or opens a D1 result file.
"""

import csv
import importlib.util
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "detection"
REAL_RUN_ROOT = PROJECT_ROOT / "runs" / "detect"
REAL_EXPORT_ROOT = PROJECT_ROOT / "runs" / "analysis" / "m110_band_aux"
REAL_LABEL_DIR = PROJECT_ROOT / "data" / "aihub_yolo_v1" / "labels"
PREFIX = "yolo11m_aihubv1_20261101_fold"
FOLDS = (0, 1, 2, 3, 4)
METRIC_WORDS = re.compile(r"recall|ap50|map50|delta|p_star|overlap", re.IGNORECASE)
SUBDIR = "m31_d1_r2"
N_HYPERBOLA = 100
OBJECTS_HEADER = (
    "fold",
    "condition",
    "image",
    "class",
    "gt_index",
    "x_center",
    "w",
    "h",
    "hit_conf025",
    "score",
)

# Synthetic crops: two per fold, one GT box of each class in all.
SYNTHETIC_LABELS = {
    "a": ((0, 100.0, 20.0, 150.0, 50.0), (2, 60.0, 100.0, 80.0, 130.0)),
    "b": ((1, 10.0, 60.0, 200.0, 75.0), (3, 180.0, 100.0, 200.0, 150.0)),
}
SYNTHETIC_PREDICTIONS = {  # stem prefix -> rows (cls, conf, x1, y1, x2, y2)
    "a": ((0, 0.9, 100.0, 20.0, 150.0, 50.0), (2, 0.8, 60.0, 100.0, 80.0, 130.0)),
    "b": (
        (1, 0.7, 10.0, 60.0, 200.0, 75.0),
        (3, 0.6, 180.0, 100.0, 200.0, 150.0),
        (0, 0.3, 0.0, 0.0, 10.0, 10.0),
    ),
}


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ANALYSIS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_script() -> ModuleType:
    return _load("m31_d1_shift_scale")


@pytest.fixture(scope="module")
def judge_script(run_script: ModuleType) -> ModuleType:
    return _load("m31_d1_judge")


@pytest.fixture(scope="module")
def common(run_script: ModuleType) -> ModuleType:
    return sys.modules["m31_d1_common"]


@pytest.fixture
def workdir() -> Iterator[Path]:
    # Not pytest's tmp_path: its folder names contain "test", which the run
    # script refuses in eval-list paths.
    path = Path(tempfile.mkdtemp(prefix="m31d1_"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


# ------------------------------------------------------- synthetic folders


@dataclass(frozen=True)
class SyntheticRepo:
    run_root: Path
    export_root: Path
    label_dir: Path


def _yolo_line(class_id: int, x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f"{class_id} {(x1 + x2) / 512:.6f} {(y1 + y2) / 512:.6f} "
        f"{(x2 - x1) / 256:.6f} {(y2 - y1) / 256:.6f}"
    )


def _write_export(folder: Path, stems: list[str], weights: Path) -> None:
    folder.mkdir(parents=True)
    image_rows, prediction_rows = [], []
    for order, stem in enumerate(stems):
        rows = SYNTHETIC_PREDICTIONS[stem[0]]
        image_rows.append([order, f"{stem}.png", 256, 256, len(rows)])
        prediction_rows.extend([f"{stem}.png", *row] for row in rows)
    with (folder / "images.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order", "file_name", "width", "height", "n_predictions"])
        writer.writerows(image_rows)
    with (folder / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_name", "cls", "conf", "x1", "y1", "x2", "y2"])
        writer.writerows(prediction_rows)
    (folder / "val_metrics.json").write_text(
        json.dumps({"weights": str(weights), "per_class": {}}), encoding="utf-8"
    )


def _write_m110_metrics(
    common: ModuleType, run: Path, export: Path, label_dir: Path, n_images: int
) -> None:
    scored = common.score_val_export(common.load_val_export(export), label_dir)
    eval_list = run / "eval" / "last_crop" / "eval_images.txt"
    instances = {cls: scored[cls]["n_gt"] for cls in common.CLASS_NAMES}
    instances["pipe_hyperbola"] = N_HYPERBOLA  # r2 metrics carry 100 hyperbolas
    metrics = {
        "weights": {"sha256": common.sha256_file(run / "weights" / "last.pt")},
        "eval_images": {"sha256": common.sha256_file(eval_list), "count": n_images},
        "per_class": [
            {"name": cls, "instances": instances[cls], "ap50": scored[cls]["ap50"]}
            for cls in common.CLASS_NAMES
        ],
    }
    (eval_list.parent / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )


def build_synthetic(root: Path, common: ModuleType) -> SyntheticRepo:
    """Five folds of two crops each, with M1-10 files consistent with the export."""
    label_dir, image_dir = root / "labels", root / "images"
    label_dir.mkdir()
    image_dir.mkdir()
    run_root, export_root = root / "runs", root / "export"
    for k in FOLDS:
        stems = [f"a{k}", f"b{k}"]
        for stem in stems:
            lines = [_yolo_line(*box) for box in SYNTHETIC_LABELS[stem[0]]]
            (label_dir / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            cv2.imwrite(
                str(image_dir / f"{stem}.png"), np.full((256, 256), 90, np.uint8)
            )
        run = run_root / f"{PREFIX}{k}"
        (run / "weights").mkdir(parents=True)
        (run / "weights" / "last.pt").write_bytes(f"weights of fold {k}".encode())
        (run / "eval" / "last_crop").mkdir(parents=True)
        eval_list = run / "eval" / "last_crop" / "eval_images.txt"
        eval_list.write_text(
            "".join(f"{image_dir / s}.png\n" for s in stems), encoding="utf-8"
        )
        export = export_root / f"fold{k}_crop"
        _write_export(export, stems, run / "weights" / "last.pt")
        _write_m110_metrics(common, run, export, label_dir, len(stems))
    return SyntheticRepo(run_root, export_root, label_dir)


@pytest.fixture
def synthetic(
    workdir: Path, common: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> SyntheticRepo:
    monkeypatch.setattr(common, "LABEL_DIR", workdir / "labels")
    repo = build_synthetic(workdir, common)
    monkeypatch.setattr(common, "EXPORT_ROOT", repo.export_root)
    sha8 = tuple(
        common.sha256_file(repo.run_root / f"{PREFIX}{k}" / "weights" / "last.pt")[:8]
        for k in FOLDS
    )
    monkeypatch.setattr(common, "EXPECTED_WEIGHTS_SHA8", sha8)
    expected = {cls: (1,) * 5 for cls in common.CLASS_NAMES}
    expected["pipe_hyperbola"] = (N_HYPERBOLA,) * 5
    monkeypatch.setattr(common, "EXPECTED_N_GT", expected)
    return repo


def _hyperbola_hits(recall: float | None, n_lost: int) -> list[int]:
    """Hit flags of the kept hyperbola GT: the first round(recall x 100) of 100
    objects are hits, and a shift loses the last ``n_lost`` objects."""
    n_hits = 0 if recall is None else round(recall * N_HYPERBOLA)
    flags = [int(i < n_hits) for i in range(N_HYPERBOLA)]
    return flags[: N_HYPERBOLA - n_lost]


def write_r2_metrics(
    repo: SyntheticRepo,
    run_script: ModuleType,
    common: ModuleType,
    hyperbola_recall: dict[str, float | None] | None = None,
    edit=None,
) -> None:
    """Synthetic r2 metrics.json and objects.csv per fold.

    pipe_hyperbola: 100 GT, recall 0.95 of the full set unless overridden;
    gp6_dx+-64 lose the expected numbers of objects (the last ones, misses).
    """
    hyperbola_recall = hyperbola_recall or {}
    for k in FOLDS:
        run = repo.run_root / f"{PREFIX}{k}"
        conditions, object_rows = {}, []
        for name in run_script.default_conditions():
            classes = {
                cls: {"n_gt": 1, "n_ignored": 0, "ap50": 0.9, "recall_conf025": 0.9}
                for cls in common.CLASS_NAMES
            }
            n_lost = common.EXPECTED_GP6_64_HYP_IGNORED.get(name, (0,) * 5)[k]
            target = hyperbola_recall.get(name, 0.95)
            flags = _hyperbola_hits(target, n_lost)
            classes["pipe_hyperbola"] = {
                "n_gt": len(flags),
                "n_ignored": n_lost,
                "ap50": 0.9,
                "recall_conf025": None if target is None else sum(flags) / len(flags),
            }
            conditions[name] = {"classes": classes, "map50": 0.9}
            for index, hit in enumerate(flags):
                object_rows.append(
                    [k, name, f"a{k}", "pipe_hyperbola", index, 1, 1, 1, hit, 0.5]
                )
        metrics = {
            "schema": "m31_d1/2",
            "n_images": 2,
            "conditions": conditions,
            "weights": str(run / "weights" / "last.pt"),
            "run_identity": {
                "run_id": "run-1",
                "git": {"head": "abc"},
                "weights_sha256": {
                    str(k): common.sha256_file(run / "weights" / "last.pt")
                },
                "g1_inputs_sha256": {
                    str(k): common.export_sha256(repo.export_root / f"fold{k}_crop")
                },
            },
        }
        if edit is not None:
            edit(k, metrics)
        out_dir = run / "eval" / SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        common.write_csv(out_dir / "objects.csv", OBJECTS_HEADER, object_rows)


def _judge(
    judge_script: ModuleType, repo: SyntheticRepo, out: Path, *extra: str
) -> int:
    return judge_script.main(
        [
            "--subdir",
            SUBDIR,
            "--out",
            str(out),
            "--run-root",
            str(repo.run_root),
            "--export-root",
            str(repo.export_root),
            *extra,
        ]
    )


def _assert_blind_stdout(text: str) -> None:
    assert not METRIC_WORDS.search(text), text
    last = json.loads(text.strip().splitlines()[-1])
    assert set(last) == {"status", "exit_code", "out"}


# --------------------------------------------------------- S1 label edges


def test_s1b_label_overshoot_is_clipped_and_kept(run_script, common, workdir) -> None:
    label = workdir / "edge.txt"
    x1, x2 = 250.0, 256.000128
    label.write_text(
        f"0 {(x1 + x2) / 512!r} {40 / 512!r} {(x2 - x1) / 256!r} {20 / 256!r}\n"
        f"2 {(250 + 256.5) / 512!r} {40 / 512!r} {6.5 / 256!r} {20 / 256!r}\n",
        encoding="utf-8",
    )
    boxes, classes = common.read_yolo_label(label)
    assert boxes[:, 2].max() == 256.0
    sample = run_script.Sample("edge", None, boxes, classes)
    _, kept_classes, ignore = run_script.map_labels(
        sample, run_script.place(1.0, 0, None)
    )
    assert kept_classes.tolist() == [0, 2]
    assert len(ignore) == 0


def test_s1b_edge_tolerance_without_clip(run_script) -> None:
    boxes = np.array([[250.0, 10.0, 256.000128, 30.0]])
    sample = run_script.Sample("edge", None, boxes, np.array([0]))
    _, kept_classes, _ = run_script.map_labels(sample, run_script.place(1.0, 0, None))
    assert kept_classes.tolist() == [0]


def test_s1b_real_cut_becomes_ignore(run_script) -> None:
    sample = run_script.Sample(
        "cut", None, np.array([[240.0, 10.0, 256.0, 30.0]]), np.array([0])
    )
    _, kept_classes, ignore = run_script.map_labels(
        sample, run_script.place(1.0, 10, None)
    )
    assert kept_classes.tolist() == []
    np.testing.assert_allclose(ignore, [[250.0, 10.0, 256.0, 30.0]])


def _first_run_counts(samples: list, run_script: ModuleType, cond: tuple) -> np.ndarray:
    """n_ignored per class with the 1st-run rule (no clip, 1e-6 px tolerance)."""
    placement = run_script.place(*cond)
    x0, y0, x1, y1 = placement.valid
    ignored = np.zeros(4, dtype=int)
    for unclipped, classes in samples:
        boxes = unclipped * placement.scale + np.array(
            [placement.off_x, 0] * 2, dtype=float
        )
        inside = (
            (boxes[:, 0] >= x0 - 1e-6)
            & (boxes[:, 1] >= y0 - 1e-6)
            & (boxes[:, 2] <= x1 + 1e-6)
            & (boxes[:, 3] <= y1 + 1e-6)
        )
        ignored += np.bincount(classes[~inside], minlength=4)
    return ignored


def _unclipped_labels(stem: str) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    label = REAL_LABEL_DIR / f"{stem}.txt"
    if label.is_file():
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                c, xc, yc, w, h = int(parts[0]), *map(float, parts[1:])
                rows.append(
                    (
                        c,
                        (xc - w / 2) * 256,
                        (yc - h / 2) * 256,
                        (xc + w / 2) * 256,
                        (yc + h / 2) * 256,
                    )
                )
    table = np.asarray(rows, dtype=float).reshape(-1, 5)
    return table[:, 1:5], table[:, 0].astype(int)


def _real_eval_list(fold: int) -> Path:
    path = REAL_RUN_ROOT / f"{PREFIX}{fold}" / "eval" / "last_crop" / "eval_images.txt"
    if not (path.is_file() and REAL_LABEL_DIR.is_dir()):
        pytest.skip(f"M1-10 eval list or labels not found ({path})")
    return path


@pytest.mark.data
def test_s1a_s1c_dry_label_counts(run_script, common) -> None:
    edge_only = ("g0.50_dx+0", "g0.74_dx+0", "g0.87_dx+0", "g1.00_dx+0")
    conds = run_script.default_conditions()
    old_total = {name: np.zeros(4, dtype=int) for name in conds}
    for k in FOLDS:
        eval_list = _real_eval_list(k)
        samples = run_script.read_samples(eval_list, None, load_images=False)
        old_samples = [_unclipped_labels(s.stem) for s in samples]
        for name, cond in conds.items():
            counts = run_script.count_labels(samples, cond)["classes"]
            new = np.array([counts[cls]["n_ignored"] for cls in common.CLASS_NAMES])
            old = _first_run_counts(old_samples, run_script, cond)
            old_total[name] += old
            assert (new <= old).all(), (k, name)
            if name in edge_only:
                assert new.sum() == 0, (k, name)
            if name == "g1.00_dx+0":
                n_gt = [counts[cls]["n_gt"] for cls in common.CLASS_NAMES]
                assert n_gt == [
                    common.EXPECTED_N_GT[cls][k] for cls in common.CLASS_NAMES
                ]
    # 1st-run record (§8-3 (2)): pipe_band 64 and manhole 32 ignored at g1.00_dx+0.
    assert old_total["g1.00_dx+0"].tolist() == [0, 64, 0, 32]


# ------------------------------------------------------ S2 scoring match


def test_ap_equals_ultralytics_ap_per_class(common) -> None:
    metrics = pytest.importorskip("ultralytics.utils.metrics")
    rng = np.random.default_rng(20261113)
    n_pred = 400
    # Scores on a coarse grid so that ties occur, as in real conf values.
    conf = (rng.integers(1, 60, n_pred) / 64).astype(np.float32)
    pred_classes = rng.integers(0, 4, n_pred)
    is_tp = rng.random(n_pred) < 0.4
    target_classes = np.repeat(np.arange(4), (70, 90, 60, 80))
    expected = metrics.ap_per_class(
        np.repeat(is_tp[:, None], 10, axis=1), conf, pred_classes, target_classes
    )[5][:, 0]
    ours = common.ap50_per_class(conf, pred_classes, is_tp, [70, 90, 60, 80])
    np.testing.assert_allclose(ours, expected, rtol=0, atol=1e-12)


def test_perfect_class_gives_the_ultralytics_0995(common) -> None:
    # The (last recall, 0) sentinel makes a perfect detector score 0.995, not 1.
    ap = common.ap50_per_class(
        np.array([0.9, 0.8]), np.array([0, 0]), np.array([True, False]), [1]
    )
    assert ap == [pytest.approx(0.995, abs=1e-12)]


def test_ap_none_without_gt_or_prediction(common) -> None:
    ap = common.ap50_per_class(
        np.array([0.9]), np.array([0]), np.array([False]), [1, 3]
    )
    assert ap[1] is None
    assert common.ap50_per_class(
        np.zeros(0), np.zeros(0, int), np.zeros(0, bool), [0]
    ) == [None]


def test_match_claims_best_unclaimed_gt(common) -> None:
    gt = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 12.0]])
    preds = np.array([[0.0, 0.0, 10.0, 11.0], [0.0, 0.0, 10.0, 10.0]])
    is_tp, claim = common.match_image(preds, np.array([0.5, 0.9]), gt)
    assert is_tp.tolist() == [True, True]
    assert claim.tolist() == [0.9, 0.5]


@pytest.mark.data
@pytest.mark.parametrize("fold", FOLDS)
def test_s2_validator_detections_rescore_to_m110(common, fold: int) -> None:
    export_dir = REAL_EXPORT_ROOT / f"fold{fold}_crop"
    m110_path = (
        REAL_RUN_ROOT / f"{PREFIX}{fold}" / "eval" / "last_crop" / "metrics.json"
    )
    if not (export_dir.is_dir() and m110_path.is_file() and REAL_LABEL_DIR.is_dir()):
        pytest.skip(f"M1-10 validator detections not found ({export_dir})")
    scored = common.score_val_export(common.load_val_export(export_dir), REAL_LABEL_DIR)
    m110 = json.loads(m110_path.read_text(encoding="utf-8"))
    for row in m110["per_class"]:
        assert abs(scored[row["name"]]["ap50"] - row["ap50"]) <= 1e-6, row["name"]


# ---------------------------------------------------- S3 G1 assertions


@pytest.mark.data
@pytest.mark.parametrize("fold", FOLDS)
def test_s3_validator_export_assertions(common, workdir, fold: int) -> None:
    export_dir = REAL_EXPORT_ROOT / f"fold{fold}_crop"
    eval_list = _real_eval_list(fold)
    if not export_dir.is_dir():
        pytest.skip(f"M1-10 validator detections not found ({export_dir})")
    weights = REAL_RUN_ROOT / f"{PREFIX}{fold}" / "weights" / "last.pt"
    stems = common.eval_list_stems(eval_list)
    checks = common.check_val_export(common.load_val_export(export_dir), stems, weights)
    assert all(checks.values()), checks
    copy = workdir / f"fold{fold}_crop"
    shutil.copytree(export_dir, copy)
    lines = (
        (copy / "predictions.csv").read_text(encoding="utf-8").splitlines(keepends=True)
    )
    (copy / "predictions.csv").write_text("".join(lines[:-1]), encoding="utf-8")
    tampered = common.check_val_export(common.load_val_export(copy), stems, weights)
    assert not tampered["n_predictions_sum_equals_rows"]


def test_s3_judge_exits_1_on_tampered_export(
    judge_script, run_script, common, synthetic, workdir
) -> None:
    write_r2_metrics(synthetic, run_script, common)
    predictions = synthetic.export_root / "fold2_crop" / "predictions.csv"
    lines = predictions.read_text(encoding="utf-8").splitlines(keepends=True)
    predictions.write_text("".join(lines[:-1]), encoding="utf-8")
    out = workdir / "verdict.json"
    assert _judge(judge_script, synthetic, out) == 1
    verdict = json.loads(out.read_text(encoding="utf-8"))
    assert "G1 fold 2" in verdict["failures"]
    assert "rule_p" not in verdict


# ------------------------------------- S4-S7 run script on synthetic folds


class FakeDetector:
    """Stands in for the YOLO backend: one fixed hyperbola box per image."""

    def predict(self, canvas: np.ndarray):
        from gpr_detection.detect.base import (
            Detection,
            DetectorOutput,
            InputTransform,
            PixelBox,
        )

        box = PixelBox(100.0, 20.0, 150.0, 50.0)
        detection = Detection(0, "pipe_hyperbola", 0.9, box)
        transform = InputTransform(1.0, 1.0, 0.0, 0.0, 256, 256)
        return DetectorOutput((detection,), transform)


def _run(
    run_script: ModuleType, repo: SyntheticRepo, summary: Path, *extra: str
) -> int:
    return run_script.main(
        [
            "--run-root",
            str(repo.run_root),
            "--device",
            "cpu",
            "--out-subdir",
            SUBDIR,
            "--summary",
            str(summary),
            *extra,
        ]
    )


def test_s4_s5_s7_run_outputs(
    run_script, common, synthetic, workdir, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        run_script, "_load_detector", lambda weights, device: FakeDetector()
    )
    summary = workdir / "summary.json"
    assert _run(run_script, synthetic, summary, "--save-examples") == 0
    _assert_blind_stdout(capsys.readouterr().out)
    out_dir = synthetic.run_root / f"{PREFIX}0" / "eval" / SUBDIR
    for path in (summary, out_dir / "metrics.json", out_dir / "objects.csv"):
        assert b"\r\n" not in path.read_bytes(), path
    with (out_dir / "objects.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == OBJECTS_HEADER
    # k1: an object keeps its label line number in every condition it survives.
    index_by_box = {(r["image"], r["class"], r["gt_index"]) for r in rows}
    assert index_by_box == {
        ("a0", "pipe_hyperbola", "0"),
        ("a0", "cavity", "1"),
        ("b0", "pipe_band", "0"),
        ("b0", "manhole", "1"),
    }
    identity = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))[
        "run_identity"
    ]
    for key in (
        "run_id",
        "started_at",
        "finished_at",
        "argv",
        "git",
        "versions",
        "gpu",
        "weights_sha256",
        "eval_list_sha256",
        "g1_inputs_sha256",
    ):
        assert identity[key] is not None, key
    assert set(identity["git"]) == {"head", "dirty", "status", "error"}
    assert set(identity["versions"]) == {
        "python",
        "numpy",
        "opencv",
        "torch",
        "torch_cuda",
        "ultralytics",
        "gpr_detection",
    }
    assert identity["gpu"] == "cpu"
    for k in FOLDS:
        assert identity["weights_sha256"][str(k)].startswith(
            common.EXPECTED_WEIGHTS_SHA8[k]
        )
        assert set(identity["g1_inputs_sha256"][str(k)]) == set(common.EXPORT_FILES)


def test_s6_run_refuses_existing_output(run_script, synthetic, workdir, capsys) -> None:
    summary = workdir / "summary.json"
    assert _run(run_script, synthetic, summary, "--dry-labels") == 0
    metrics = synthetic.run_root / f"{PREFIX}0" / "eval" / SUBDIR / "metrics.json"
    first = metrics.read_bytes()
    assert _run(run_script, synthetic, summary, "--dry-labels") == 2
    assert metrics.read_bytes() == first
    summary.unlink()
    assert (
        _run(run_script, synthetic, summary, "--dry-labels") == 2
    )  # fold folders exist
    assert _run(run_script, synthetic, summary, "--dry-labels", "--overwrite") == 0
    assert b"\r\n" not in metrics.read_bytes()
    dry = json.loads(metrics.read_text(encoding="utf-8"))
    assert dry["schema"] == "m31_d1_dry_labels/2"
    assert set(dry["conditions"]["g1.00_dx+0"]["classes"]["cavity"]) == {
        "n_gt",
        "n_ignored",
    }
    _assert_blind_stdout(capsys.readouterr().out)


def test_s6_overwrite_never_touches_first_run_names(
    run_script, synthetic, workdir
) -> None:
    first_run = synthetic.run_root / f"{PREFIX}0" / "eval" / "m31_d1"
    first_run.mkdir()
    (first_run / "metrics.json").write_text("{}", encoding="utf-8")
    code = run_script.main(
        [
            "--run-root",
            str(synthetic.run_root),
            "--out-subdir",
            "m31_d1",
            "--dry-labels",
            "--overwrite",
        ]
    )
    assert code == 2
    assert (first_run / "metrics.json").is_file()


def test_s1a_dry_labels_need_no_images(run_script, synthetic, workdir) -> None:
    for image in (workdir / "images").iterdir():
        image.unlink()
    assert _run(run_script, synthetic, workdir / "summary.json", "--dry-labels") == 0


# ------------------------------------------------------------ S8 judge


def _verdict(out: Path) -> dict:
    return json.loads(out.read_text(encoding="utf-8"))


def test_s8_normal_run_is_judged(
    judge_script, run_script, common, synthetic, workdir, capsys
) -> None:
    write_r2_metrics(synthetic, run_script, common)
    out = workdir / "verdict.json"
    assert _judge(judge_script, synthetic, out) == 0
    _assert_blind_stdout(capsys.readouterr().out)
    verdict = _verdict(out)
    assert verdict["gates_pass"]
    assert verdict["verdict"] == {"p_star": 4.06, "overlap": "50 % (O 128)"}
    assert verdict["source_runs"]["per_fold"]["0"] == {
        "run_id": "run-1",
        "git_head": "abc",
    }
    assert b"\r\n" not in out.read_bytes()


def test_s8_wrong_n_gt_fails(
    judge_script, run_script, common, synthetic, workdir
) -> None:
    def edit(k, metrics):
        if k == 3:
            metrics["conditions"]["g1.00_dx+0"]["classes"]["cavity"]["n_gt"] = 2

    write_r2_metrics(synthetic, run_script, common, edit=edit)
    out = workdir / "verdict.json"
    assert _judge(judge_script, synthetic, out) == 1
    verdict = _verdict(out)
    assert verdict["failures"] == ["G3 fold 3"]
    assert "rule_p" not in verdict and "per_fold_and_mean_pp" not in verdict


def test_s8_none_recall_fails(
    judge_script, run_script, common, synthetic, workdir
) -> None:
    write_r2_metrics(synthetic, run_script, common, {"g1.00_dx+32": None})
    out = workdir / "verdict.json"
    assert _judge(judge_script, synthetic, out) == 1
    assert len(_verdict(out)["failures"]) == 5
    assert "rule_o" not in _verdict(out)


@pytest.mark.parametrize(("offset", "expected_code"), [(0.015, 1), (0.009, 0)])
def test_s8_g1_tolerance(
    judge_script,
    run_script,
    common,
    synthetic,
    workdir,
    offset: float,
    expected_code: int,
) -> None:
    write_r2_metrics(synthetic, run_script, common)
    path = synthetic.run_root / f"{PREFIX}1" / "eval" / "last_crop" / "metrics.json"
    m110 = json.loads(path.read_text(encoding="utf-8"))
    m110["per_class"][2]["ap50"] -= offset  # cavity, 1.5 or 0.9 %p
    path.write_text(json.dumps(m110), encoding="utf-8")
    out = workdir / "verdict.json"
    assert _judge(judge_script, synthetic, out) == expected_code


def test_s8_rule_o_non_monotone_is_both_fail(
    judge_script, run_script, common, synthetic, workdir
) -> None:
    recalls = {
        "g1.00_dx+64": 0.93,
        "g1.00_dx-64": 0.93,  # delta(64) = 2 %p
        "g1.00_dx+32": 0.90,
        "g1.00_dx-32": 0.90,
    }  # delta(32) = 5 %p
    write_r2_metrics(synthetic, run_script, common, recalls)
    out = workdir / "verdict.json"
    assert _judge(judge_script, synthetic, out) == 0
    rule_o = _verdict(out)["rule_o"]
    assert rule_o["series"] == "X1"
    assert rule_o["overlap"] == "both_fail (provisional 75 %)"


def test_s6_judge_refuses_existing_out(
    judge_script, run_script, common, synthetic, workdir
) -> None:
    write_r2_metrics(synthetic, run_script, common)
    out = workdir / "verdict.json"
    out.write_text("{}", encoding="utf-8")
    assert _judge(judge_script, synthetic, out) == 2
    assert out.read_text(encoding="utf-8") == "{}"
    assert _judge(judge_script, synthetic, out, "--overwrite") == 0


def test_judge_missing_input_exits_2(judge_script, synthetic, workdir) -> None:
    assert _judge(judge_script, synthetic, workdir / "verdict.json") == 2


# -------------------------------------------------- S5 identity on real weights


@pytest.mark.data
def test_s5_real_weights_match_audit(common) -> None:
    for k in FOLDS:
        weights = REAL_RUN_ROOT / f"{PREFIX}{k}" / "weights" / "last.pt"
        if not weights.is_file():
            pytest.skip(f"M1-10 weights not found ({weights})")
        assert common.sha256_file(weights).startswith(common.EXPECTED_WEIGHTS_SHA8[k])


# ------------------------------------------------- S10 v0.5.1 additions (k)


def test_s10_1_paired_delta_equals_old_formula_without_loss(judge_script) -> None:
    base = {("a", i): int(i < 8) for i in range(10)}
    plus = {("a", i): int(i < 6) for i in range(10)}
    minus = {("a", i): int(i < 7) for i in range(10)}
    hits = {"b": base, "p": plus, "m": minus}
    paired = judge_script.paired_shift_delta(hits, "b", "p", "m")
    assert paired == pytest.approx(80.0 - 0.5 * (60.0 + 70.0), abs=1e-9)


def test_s10_2_paired_delta_with_three_lost(judge_script) -> None:
    flags = [1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
    base = {("a", i): f for i, f in enumerate(flags)}
    plus_flags = [1, 1, 1, 0, 1, 1, 0]  # objects 7, 8, 9 lost at +64
    plus = {("a", i): f for i, f in enumerate(plus_flags)}
    minus = dict(base)
    hits = {"b": base, "p": plus, "m": minus}
    # +: R0 on {0..6} = 7/7, R(+64) = 5/7 -> 200/7 %p; -: 0 -> mean 100/7 %p.
    paired = judge_script.paired_shift_delta(hits, "b", "p", "m")
    assert paired == pytest.approx(100.0 / 7.0, abs=1e-12)


def test_s10_full_judge_with_expected_losses_passes(
    judge_script, run_script, common, synthetic, workdir
) -> None:
    write_r2_metrics(synthetic, run_script, common)
    out = workdir / "verdict.json"
    assert _judge(judge_script, synthetic, out) == 0
    x6_d64 = json.loads(out.read_text(encoding="utf-8"))["per_fold_and_mean_pp"][
        "X6_d64"
    ]["per_fold"]
    # Lost objects are misses here, so on the kept set the base recall is
    # 95/97 like the shifted one: delta 0 although 95 - 95/97 x 100 != 0.
    assert x6_d64 == [0.0] * 5


def test_s10_3_wrong_loss_count_fails(
    judge_script, run_script, common, synthetic, workdir
) -> None:
    def edit(k, metrics):
        if k == 2:
            hyp = metrics["conditions"]["gp6_dx+64"]["classes"]["pipe_hyperbola"]
            hyp["n_ignored"] = 2

    write_r2_metrics(synthetic, run_script, common, edit=edit)
    out = workdir / "verdict.json"
    assert _judge(judge_script, synthetic, out) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["failures"] == ["G3 fold 2"]


def test_s10_4_run_id_per_fold_fails(
    judge_script, run_script, common, synthetic, workdir
) -> None:
    def edit(k, metrics):
        metrics["run_identity"]["run_id"] = f"run-{k}"

    write_r2_metrics(synthetic, run_script, common, edit=edit)
    out = workdir / "verdict.json"
    assert _judge(judge_script, synthetic, out) == 1
    verdict = json.loads(out.read_text(encoding="utf-8"))
    assert verdict["failures"] == [f"G2 fold {k}" for k in FOLDS]
    assert not verdict["gates"]["G2"][0]["checks"]["g2b_single_run_id"]


def test_s10_5_changed_g1_input_fails(
    judge_script, run_script, common, synthetic, workdir
) -> None:
    write_r2_metrics(synthetic, run_script, common)
    val_metrics = synthetic.export_root / "fold4_crop" / "val_metrics.json"
    val_metrics.write_text(val_metrics.read_text(encoding="utf-8") + "\n", "utf-8")
    out = workdir / "verdict.json"
    assert _judge(judge_script, synthetic, out) == 1
    verdict = json.loads(out.read_text(encoding="utf-8"))
    assert verdict["failures"] == ["G2 fold 4"]
    assert not verdict["gates"]["G2"][4]["checks"]["g2c_g1_inputs_match_run_record"]


def test_g2a_weights_record_mismatch_fails(
    judge_script, run_script, common, synthetic, workdir
) -> None:
    def edit(k, metrics):
        if k == 1:
            metrics["run_identity"]["weights_sha256"]["1"] = "0" * 64

    write_r2_metrics(synthetic, run_script, common, edit=edit)
    out = workdir / "verdict.json"
    assert _judge(judge_script, synthetic, out) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["failures"] == ["G2 fold 1"]


def test_objects_without_gt_index_exits_2(
    judge_script, run_script, common, synthetic, workdir
) -> None:
    write_r2_metrics(synthetic, run_script, common)
    objects = synthetic.run_root / f"{PREFIX}3" / "eval" / SUBDIR / "objects.csv"
    lines = objects.read_text(encoding="utf-8").splitlines()
    stripped = [
        ",".join(p for i, p in enumerate(ln.split(",")) if i != 4) for ln in lines
    ]
    objects.write_text("\n".join(stripped) + "\n", encoding="utf-8")
    assert _judge(judge_script, synthetic, workdir / "verdict.json") == 2


# ------------------------------------- D1 r3 arguments (n2a plan §4-1, §5-5)


def test_explicit_expected_weights_sha8_is_used(
    judge_script, run_script, common, synthetic, workdir
) -> None:
    write_r2_metrics(synthetic, run_script, common)
    right = ",".join(common.EXPECTED_WEIGHTS_SHA8)
    wrong = ",".join((*common.EXPECTED_WEIGHTS_SHA8[:4], "00000000"))
    sha_flag = "--expected-weights-sha8"
    right_out = workdir / "right.json"
    wrong_out = workdir / "wrong.json"

    assert _judge(judge_script, synthetic, right_out, sha_flag, right) == 0
    assert _judge(judge_script, synthetic, wrong_out, sha_flag, wrong) == 1
    verdict = json.loads(wrong_out.read_text(encoding="utf-8"))
    assert verdict["failures"] == ["G2 fold 4"]
    assert verdict["gates"]["G2"][4]["weights_sha8_expected"] == "00000000"


BAD_SHA8_LISTS = ["a,b,c,d,e", "5300f34b,cf9423f1", "zzzzzzzz," * 4 + "1"]


@pytest.mark.parametrize("text", BAD_SHA8_LISTS)
def test_expected_weights_sha8_must_be_five_hex_prefixes(judge_script, text) -> None:
    with pytest.raises(SystemExit):
        judge_script._parse_args(["--out", "x.json", "--expected-weights-sha8", text])


def test_run_prefix_selects_other_run_folders(
    judge_script, run_script, common, synthetic, workdir
) -> None:
    write_r2_metrics(synthetic, run_script, common)
    for k in FOLDS:
        source = synthetic.run_root / f"{PREFIX}{k}"
        shutil.copytree(source, synthetic.run_root / f"n2a_copy_fold{k}")
    copied_out = workdir / "verdict.json"
    absent_out = workdir / "none.json"
    explicit_out = workdir / "explicit.json"
    assert _judge(judge_script, synthetic, explicit_out, "--run-prefix", PREFIX) == 0
    # The copies are read (G2 passes on them) but the export still names the
    # original weights path, so G1 fails: an export must come from these runs.
    code = _judge(judge_script, synthetic, copied_out, "--run-prefix", "n2a_copy_fold")
    verdict = json.loads(copied_out.read_text(encoding="utf-8"))
    assert code == 1
    assert verdict["failures"] == [f"G1 fold {k}" for k in FOLDS]
    assert all(row["pass"] for row in verdict["gates"]["G2"])
    assert "n2a_copy_fold0" in verdict["gates"]["G2"][0]["weights"]
    assert (
        _judge(judge_script, synthetic, absent_out, "--run-prefix", "absent_fold") == 2
    )


def _dry_g1_sha(synthetic, run_script, workdir, name: str, *extra: str) -> dict:
    summary = workdir / f"{name}_summary.json"
    args = ("--dry-labels", "--overwrite", *extra)
    assert _run(run_script, synthetic, summary, *args) == 0
    metrics = synthetic.run_root / f"{PREFIX}0" / "eval" / SUBDIR / "metrics.json"
    identity = json.loads(metrics.read_text(encoding="utf-8"))["run_identity"]
    return identity["g1_inputs_sha256"]


def test_run_export_root_sets_recorded_g1_inputs(
    run_script, common, synthetic, workdir
) -> None:
    default = _dry_g1_sha(synthetic, run_script, workdir, "default")
    explicit = _dry_g1_sha(
        synthetic,
        run_script,
        workdir,
        "explicit",
        "--export-root",
        str(synthetic.export_root),
    )
    other_root = workdir / "other_export"
    shutil.copytree(synthetic.export_root, other_root)
    images_csv = other_root / "fold3_crop" / "images.csv"
    images_csv.write_bytes(images_csv.read_bytes() + b"\n")
    other = _dry_g1_sha(
        synthetic, run_script, workdir, "other", "--export-root", str(other_root)
    )

    assert explicit == default
    assert other["3"] == common.export_sha256(common.export_folder(other_root, 3))
    assert other["3"] != default["3"]
    assert other["0"] == default["0"]
