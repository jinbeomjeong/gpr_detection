"""pipe_band auxiliary metrics of the N2-a models (report only, M3-1 A1 judgement).

Runs ``m110_band_aux.py`` unchanged on the N2-a validator exports by
rebinding its two module constants before ``main``:

* ``RUN_NAME``  -> ``yolo11m_aihubv1n2a_20261113_fold{fold}`` (the run whose
  ``eval/last_crop`` confusion matrix and metrics.json the predictions must
  reproduce),
* ``OUT_STEM``  -> ``analysis/detection/n2a_band_aux`` (so the M1-10 outputs
  are never overwritten).

Prediction root defaults to ``runs/analysis/n2a_val_export`` (crop unit only).
CPU only. Usage:
    python -I analysis/detection/n2a_band_aux.py
Writes ``analysis/detection/n2a_band_aux.json``/``.csv``; the last stdout line
is the summary JSON of ``m110_band_aux.main``. Refuses to overwrite.
"""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = "yolo11m_aihubv1n2a_20261113_fold{fold}"
OUT_STEM = PROJECT_ROOT / "analysis" / "detection" / "n2a_band_aux"
PREDICTION_ROOT = PROJECT_ROOT / "runs" / "analysis" / "n2a_val_export"


def main() -> int:
    if OUT_STEM.with_suffix(".json").exists():
        print('{"status": "error", "exit_code": 2, "out": "exists"}')
        return 2
    path = Path(__file__).with_name("m110_band_aux.py")
    spec = importlib.util.spec_from_file_location("m110_band_aux", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.RUN_NAME = RUN_NAME
    module.OUT_STEM = OUT_STEM
    return module.main(["--units", "crop", "--prediction-root", str(PREDICTION_ROOT)])


if __name__ == "__main__":
    sys.exit(main())
