"""Regression of the AP50 in ``band_metrics`` on the saved M1-10 predictions.

Recomputes, from ``runs/analysis/m110_band_aux/fold<k>_<unit>/predictions.csv``,
the per-class AP50 (must equal the validator's ``metrics.json``), the conf 0.25
confusion matrix (must equal ``confusion_matrix_conf0.25.csv``) and the fold
and per-session ``pipe_band`` AP50 recorded in
``analysis/detection/m110_band_aux.json``. Skipped when the run outputs (git
excluded) are absent. CPU only, about 4 s per case.
"""

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = PROJECT_ROOT / "analysis" / "detection" / "m110_band_aux.py"
STORED_RESULTS = ANALYSIS_SCRIPT.with_suffix(".json")
PREDICTION_ROOT = PROJECT_ROOT / "runs" / "analysis" / "m110_band_aux"
MANIFEST = PROJECT_ROOT / "data" / "splits" / "aihub_yolo_v1.csv"
FOLDS = (0, 1, 2, 3, 4)
UNITS = ("crop", "image")


def _load_analysis_module():
    spec = importlib.util.spec_from_file_location("m110_band_aux", ANALYSIS_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stored_fold(unit: str, fold: int) -> dict:
    document = json.loads(STORED_RESULTS.read_text(encoding="utf-8"))
    (result,) = [r for r in document["units"][unit]["folds"] if r["fold"] == fold]
    return result


@pytest.mark.data
@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("fold", FOLDS)
def test_saved_predictions_give_the_recorded_ap50(unit: str, fold: int) -> None:
    prediction_dir = PREDICTION_ROOT / f"fold{fold}_{unit}"
    needed = (ANALYSIS_SCRIPT, STORED_RESULTS, MANIFEST, prediction_dir)
    if not all(path.exists() for path in needed):
        pytest.skip(f"M1-10 saved predictions not found ({prediction_dir})")
    analysis = _load_analysis_module()
    sessions = analysis._read_sessions(MANIFEST)

    result = analysis.evaluate_fold(fold, unit, PREDICTION_ROOT, sessions)

    assert result["checks"]["ap50_max_abs_diff_vs_metrics_json"] == 0.0
    assert result["checks"]["confusion_matches"]
    stored = _stored_fold(unit, fold)
    assert result["band_ap50"] == stored["band_ap50"]
    session_ap50 = {row["session"]: row["band_ap50"] for row in result["sessions"]}
    stored_session_ap50 = {
        row["session"]: row["band_ap50"] for row in stored["sessions"]
    }
    assert session_ap50 == stored_session_ap50
