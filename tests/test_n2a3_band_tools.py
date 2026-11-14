"""N2-a″ helper scripts: band-aux wrapper (blind) and the S-check diagnostic.

``analysis/detection/n2a3_band_aux.py`` must hide the metric line of
``m110_band_aux.main`` and refuse to overwrite; ``n2a3_band_label_diag.py``
accepts only a config pair that differs by the K32 key and bins widths as the
plan §5-1 lists. CPU only, no data.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "detection"
N2A2_CONFIG = PROJECT_ROOT / "configs" / "train_yolo11m_n2a2.yaml"
N2A3_CONFIG = PROJECT_ROOT / "configs" / "train_yolo11m_n2a3.yaml"

FAKE_M110 = """
import json
RUN_NAME = "unset"
OUT_STEM = None


def main(argv):
    OUT_STEM.with_suffix(".json").write_text(
        json.dumps({"run_name": RUN_NAME, "argv": argv}), encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "recall": {"mean": 0.123456}}))
    return 0
"""


def _load(name: str, monkeypatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ANALYSIS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ------------------------------------------------------ band-aux wrapper


def test_band_aux_wrapper_hides_metrics_and_refuses_overwrite(
    monkeypatch, tmp_path, capsys
) -> None:
    wrapper = _load("n2a3_band_aux", monkeypatch)
    fake = tmp_path / "m110_band_aux.py"
    fake.write_text(FAKE_M110, encoding="utf-8")
    monkeypatch.setattr(wrapper, "M110_SCRIPT", fake)
    monkeypatch.setattr(wrapper, "OUT_STEM", tmp_path / "n2a3_band_aux")
    monkeypatch.setitem(sys.modules, "m110_band_aux", None)  # restored afterwards

    assert wrapper.main(["--date", "20261120"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and "recall" not in lines[0]
    assert json.loads(lines[0])["exit_code"] == 0
    written = json.loads((tmp_path / "n2a3_band_aux.json").read_text("utf-8"))
    assert written["run_name"] == "yolo11m_aihubv1n2a3_20261120_fold{fold}"
    assert written["argv"][:2] == ["--units", "crop"]
    assert written["argv"][3].endswith(str(Path("runs/analysis/n2a3_val_export")))

    assert wrapper.main(["--date", "20261120"]) == 2  # output exists
    assert wrapper.main(["--date", "2026-11-20"]) == 2  # not YYYYMMDD


# -------------------------------------------------------- S-check diag


def test_diag_accepts_only_the_k32_pair(monkeypatch) -> None:
    diag = _load("n2a3_band_label_diag", monkeypatch)
    baseline, k32 = _yaml(N2A2_CONFIG), _yaml(N2A3_CONFIG)
    diag.check_config_pair(baseline, k32)  # fine
    with pytest.raises(diag.common.InputError, match="--config needs"):
        diag.check_config_pair(baseline, baseline)
    with pytest.raises(diag.common.InputError, match="--baseline-config"):
        diag.check_config_pair(k32, k32)
    other = _yaml(N2A3_CONFIG)
    other["gpr_context"]["p_context"] = 0.4
    with pytest.raises(diag.common.InputError, match="beyond K32"):
        diag.check_config_pair(baseline, other)


def test_diag_width_bins_follow_the_plan(monkeypatch) -> None:
    diag = _load("n2a3_band_label_diag", monkeypatch)
    assert [diag.width_bin(value) for value in (0, 7.99, 8, 31.9, 32, 64)] == [
        "0-8",
        "0-8",
        "8-16",
        "24-32",
        "32-64",
        "64-128",
    ]
    assert diag.width_bin(256) == "192-256"  # the window width is the top edge
    assert diag.width_bin(256.5) == ">256"


def test_diag_refuses_existing_out(monkeypatch, tmp_path) -> None:
    diag = _load("n2a3_band_label_diag", monkeypatch)
    out = tmp_path / "diag.json"
    out.write_text("{}", encoding="utf-8")
    arguments = [
        *("--baseline-config", str(N2A2_CONFIG), "--config", str(N2A3_CONFIG)),
        *("--images", "1", "--seeds", "0", "--out", str(out)),
    ]
    assert diag.main(arguments) == 2
    assert out.read_text(encoding="utf-8") == "{}"
