"""pipe_band auxiliary metrics of the N2-a″ (n2a3) models (M3-1, blind).

Plan docs/experiments/m31_n2a3_retrain_plan.md v1.0 §6-2. Runs
``m110_band_aux.py`` unchanged on the N2-a″ validator exports
(``runs/analysis/n2a3_val_export``, crop unit only) by rebinding its two module
constants before ``main``:

* ``RUN_NAME`` -> ``yolo11m_aihubv1n2a3_<D>_fold{fold}`` (``--date``; the run
  whose ``eval/last_crop`` confusion matrix and metrics.json the predictions
  must reproduce, the binding check of §6-2),
* ``OUT_STEM`` -> ``analysis/detection/n2a3_band_aux`` (other outputs are never
  overwritten).

Blind (§7-1): the metric line ``m110_band_aux.main`` prints is captured and
not shown; stdout carries only {"status", "exit_code", "out"}. Refuses to
overwrite an existing json or csv. CPU only. Exit codes: 0 written and every
fold reproduced its evaluation, 1 written but a reproduction check failed,
2 input missing / output exists / bad --date, 3 internal error.

Usage:
    python -I analysis/detection/n2a3_band_aux.py --date <YYYYMMDD>
"""

import argparse
import contextlib
import importlib.util
import io
import json
import logging
import re
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_NAME_TEMPLATE = "yolo11m_aihubv1n2a3_{date}_fold{{fold}}"
OUT_STEM = PROJECT_ROOT / "analysis" / "detection" / "n2a3_band_aux"
PREDICTION_ROOT = PROJECT_ROOT / "runs" / "analysis" / "n2a3_val_export"
M110_SCRIPT = Path(__file__).with_name("m110_band_aux.py")
DATE_PATTERN = re.compile(r"\d{8}")

logger = logging.getLogger("n2a3_band_aux")


def _load_m110() -> ModuleType:
    spec = importlib.util.spec_from_file_location("m110_band_aux", M110_SCRIPT)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"cannot load {M110_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _existing_outputs(out_stem: Path) -> list[Path]:
    candidates = (out_stem.with_suffix(".json"), out_stem.with_suffix(".csv"))
    return [path for path in candidates if path.exists()]


def _final_line(exit_code: int, out: Path) -> int:
    line = {
        "status": "ok" if exit_code == 0 else "error",
        "exit_code": exit_code,
        "out": str(out),
    }
    print(json.dumps(line, ensure_ascii=False), flush=True)
    return exit_code


def run(date: str, out_stem: Path, prediction_root: Path) -> int:
    """Rebind the constants and run ``m110_band_aux.main`` with stdout captured."""
    module = _load_m110()
    module.RUN_NAME = RUN_NAME_TEMPLATE.format(date=date)
    module.OUT_STEM = out_stem
    captured = io.StringIO()
    arguments = ["--units", "crop", "--prediction-root", str(prediction_root)]
    with contextlib.redirect_stdout(captured):
        return int(module.main(arguments))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--date", required=True, help="run date <D> of yolo11m_aihubv1n2a3_<D>_fold<k>"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry; see the module docstring for the exit codes."""
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
    args = _parse_args(argv)
    out = OUT_STEM.with_suffix(".json")
    if not DATE_PATTERN.fullmatch(args.date):
        logger.error("input error: --date must be YYYYMMDD, got %r", args.date)
        return _final_line(2, out)
    existing = _existing_outputs(OUT_STEM)
    if existing:
        logger.error("input error: output exists: %s", ", ".join(map(str, existing)))
        return _final_line(2, out)
    try:
        code = run(args.date, OUT_STEM, PREDICTION_ROOT)
    except FileNotFoundError as error:
        logger.error("input error: %s", error)
        return _final_line(2, out)
    except Exception:
        # The traceback names files only; m110_band_aux prints metrics to stdout,
        # which stays captured.
        logger.exception("internal error")
        return _final_line(3, out)
    return _final_line(code, out)


if __name__ == "__main__":
    sys.exit(main())
