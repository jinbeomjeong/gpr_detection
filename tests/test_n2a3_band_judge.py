"""N2-a″ B / B-p judge (analysis/detection/n2a3_band_judge.py), synthetic JSON.

Plan docs/experiments/m31_n2a3_retrain_plan.md v1.0 §6-3: threshold edges
(mean exactly 0.641 passes, a fold equal to N2-a' is not better) and the fail
path of each gate G-b1..G-b4. CPU only.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JUDGE_SCRIPT = PROJECT_ROOT / "analysis" / "detection" / "n2a3_band_judge.py"
REAL_REFERENCE = PROJECT_ROOT / "analysis" / "detection" / "n2a2_band_aux.json"
REFERENCE_SHA256 = "bf42403e1eedfa95fc556f1710c461d2cdb146cbe164072003a14742860a6145"
REFERENCE_RECALL = [
    0.646922995795787,
    0.5401378301936709,
    0.5778940392773931,
    0.5797035944530312,
    0.7197308631979048,
]
RUN_PREFIX = "yolo11m_aihubv1n2a3_20261120_fold"
FOLDS = [0, 1, 2, 3, 4]
# Mean exactly 0.641 (3.205 / 5); every fold above REFERENCE_RECALL but fold 4.
RECALL_AT_THRESHOLD = [0.66, 0.60, 0.62, 0.65, 0.675]
PRECISION_AT_THRESHOLD = [0.5, 0.49, 0.497, 0.501, 0.497]  # mean 0.497


def _load_judge(monkeypatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("n2a3_band_judge", JUDGE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "n2a3_band_judge", module)
    spec.loader.exec_module(module)
    return module


def _band_document(recall, precision, fold_ids=FOLDS, failing_fold=None) -> dict:
    folds = [
        {
            "fold": fold,
            "unit": "crop",
            "checks": {
                "confusion_matches": fold != failing_fold,
                "ap50_matches": True,
            },
        }
        for fold in fold_ids
    ]
    return {
        "schema": "m110_band_aux/1",
        "units": {
            "crop": {
                "fold_ids": list(fold_ids),
                "statistics": {
                    "recall": {"folds": list(recall)},
                    "precision": {"folds": list(precision)},
                },
                "folds": folds,
            }
        },
    }


def _a1_verdict(gates_pass=True, run_prefix=RUN_PREFIX, a1_pass=True) -> dict:
    return {
        "gates_pass": gates_pass,
        "inputs": {"run_prefix": run_prefix},
        "verdict": "A1 met" if a1_pass else "A1 not met",
        "a1": {"pass": a1_pass, "map50": {"mean_pp": 79.0}},
    }


def _write(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _inputs(
    tmp_path: Path,
    recall=RECALL_AT_THRESHOLD,
    precision=PRECISION_AT_THRESHOLD,
    reference_fold_ids=FOLDS,
    failing_fold=None,
    a1: dict | None = None,
    reference_sha256: str | None = None,
) -> list[str]:
    band = _write(
        tmp_path / "n2a3_band_aux.json",
        _band_document(recall, precision, failing_fold=failing_fold),
    )
    reference = _write(
        tmp_path / "n2a2_band_aux.json",
        _band_document(REFERENCE_RECALL, [0.75] * 5, fold_ids=reference_fold_ids),
    )
    a1_path = _write(tmp_path / "n2a3_a1_verdict.json", a1 or _a1_verdict())
    sha = reference_sha256 or hashlib.sha256(reference.read_bytes()).hexdigest()
    return [
        *("--band-aux", str(band), "--reference", str(reference)),
        *("--reference-sha256", sha, "--a1-verdict", str(a1_path)),
        *("--run-prefix", RUN_PREFIX, "--out", str(tmp_path / "verdict.json")),
    ]


def _verdict(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8"))


def test_mean_exactly_at_thresholds_passes(monkeypatch, tmp_path, capsys) -> None:
    judge = _load_judge(monkeypatch)
    assert judge.main(_inputs(tmp_path)) == 0
    verdict = _verdict(tmp_path)
    assert verdict["verdict"] == "B and B-p met" and verdict["failed"] == []
    assert verdict["b"]["mean_recall"]["mean_exact"] == "0.641"
    assert verdict["b"]["folds_better"] == 4
    assert verdict["b_p"]["mean_precision"]["mean_exact"] == "0.497"
    assert verdict["b_p"]["pass"] and verdict["adoption_pending_a2"]
    assert verdict["a1"]["verdict"] == "A1 met"
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["exit_code"] == 0
    assert "0.641" not in lines[0]  # blind: no metric on stdout


def test_mean_just_below_threshold_fails(monkeypatch, tmp_path) -> None:
    judge = _load_judge(monkeypatch)
    recall = [*RECALL_AT_THRESHOLD[:4], 0.67499999]
    precision = [*PRECISION_AT_THRESHOLD[:4], 0.49699999]
    assert judge.main(_inputs(tmp_path, recall=recall, precision=precision)) == 0
    verdict = _verdict(tmp_path)
    assert verdict["verdict"] == "B or B-p not met"
    assert verdict["failed"] == ["B", "B-p"]
    assert not verdict["b_p"]["pass"] and not verdict["adoption_pending_a2"]


def test_fold_equal_to_reference_is_not_better(monkeypatch, tmp_path) -> None:
    judge = _load_judge(monkeypatch)
    # Folds 0-2 better, fold 3 equal, fold 4 worse: 3 / 5 even with a high mean.
    recall = [0.70, 0.70, 0.70, REFERENCE_RECALL[3], 0.70]
    assert judge.main(_inputs(tmp_path, recall=recall)) == 0
    verdict = _verdict(tmp_path)
    assert verdict["b"]["mean_recall"]["pass"]
    assert verdict["b"]["fold_better_than_reference"] == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert verdict["b"]["folds_better"] == 3
    assert verdict["verdict"] == "B or B-p not met" and verdict["failed"] == ["B"]


RECALL_BELOW = [*RECALL_AT_THRESHOLD[:4], 0.67499999]
PRECISION_BELOW = [*PRECISION_AT_THRESHOLD[:4], 0.49699999]


@pytest.mark.parametrize(
    ("recall", "precision", "verdict_text", "failed"),
    [
        (RECALL_AT_THRESHOLD, PRECISION_AT_THRESHOLD, "B and B-p met", []),
        (RECALL_BELOW, PRECISION_AT_THRESHOLD, "B or B-p not met", ["B"]),
        (RECALL_AT_THRESHOLD, PRECISION_BELOW, "B or B-p not met", ["B-p"]),
        (RECALL_BELOW, PRECISION_BELOW, "B or B-p not met", ["B", "B-p"]),
    ],
)
def test_verdict_and_failed_for_the_four_combinations(
    monkeypatch, tmp_path, recall, precision, verdict_text, failed
) -> None:
    """Plan v1.0 §6-3: the judging unit is B and B-p; exit 0 whenever judged."""
    judge = _load_judge(monkeypatch)
    assert judge.main(_inputs(tmp_path, recall=recall, precision=precision)) == 0
    verdict = _verdict(tmp_path)
    assert verdict["verdict"] == verdict_text
    assert verdict["failed"] == failed
    assert verdict["b"]["pass"] is ("B" not in failed)
    assert verdict["b_p"]["pass"] is ("B-p" not in failed)
    assert verdict["adoption_pending_a2"] is (not failed)


def test_a1_not_met_blocks_adoption(monkeypatch, tmp_path) -> None:
    judge = _load_judge(monkeypatch)
    arguments = _inputs(tmp_path, a1=_a1_verdict(a1_pass=False))
    assert judge.main(arguments) == 0
    verdict = _verdict(tmp_path)
    assert verdict["verdict"] == "B and B-p met" and verdict["failed"] == []
    assert not verdict["adoption_pending_a2"]


@pytest.mark.parametrize(
    ("gate", "options"),
    [
        ("G-b1_reference_sha256", {"reference_sha256": "0" * 64}),
        ("G-b2_fold_ids", {"reference_fold_ids": [0, 1, 2, 3, 5]}),
        ("G-b3_reproduction", {"failing_fold": 2}),
        ("G-b4_a1_verdict", {"a1": _a1_verdict(gates_pass=False)}),
        ("G-b4_a1_verdict", {"a1": _a1_verdict(run_prefix="other_fold")}),
    ],
)
def test_each_gate_failure_gives_no_verdict(monkeypatch, tmp_path, gate, options):
    judge = _load_judge(monkeypatch)
    assert judge.main(_inputs(tmp_path, **options)) == 1
    verdict = _verdict(tmp_path)
    assert verdict["failures"] == [gate]
    assert verdict["verdict"] == "no verdict: gate failed"
    assert "b" not in verdict and "adoption_pending_a2" not in verdict


def test_existing_out_and_missing_input_are_refused(monkeypatch, tmp_path) -> None:
    judge = _load_judge(monkeypatch)
    arguments = _inputs(tmp_path)
    assert judge.main(arguments) == 0
    assert judge.main(arguments) == 2  # --out exists
    assert judge.main([*arguments, "--overwrite"]) == 0
    (tmp_path / "n2a3_a1_verdict.json").unlink()
    assert judge.main([*arguments, "--overwrite"]) == 2


def test_malformed_fold_value_is_an_input_error(monkeypatch, tmp_path) -> None:
    judge = _load_judge(monkeypatch)
    recall = [*RECALL_AT_THRESHOLD[:4], None]
    assert judge.main(_inputs(tmp_path, recall=recall)) == 2


def test_real_reference_file_is_the_sealed_one(monkeypatch) -> None:
    """Plan §4-1: the N2-a' reference file and the recall values it holds."""
    if not REAL_REFERENCE.is_file():
        pytest.skip("analysis/detection/n2a2_band_aux.json absent")
    judge = _load_judge(monkeypatch)
    assert hashlib.sha256(REAL_REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA256
    document = json.loads(REAL_REFERENCE.read_text(encoding="utf-8"))
    assert document["units"]["crop"]["fold_ids"] == FOLDS
    recall = judge.fold_values(document, "reference", "recall")
    assert [float(value) for value in recall] == REFERENCE_RECALL
