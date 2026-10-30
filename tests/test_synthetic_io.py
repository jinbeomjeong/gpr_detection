"""Tests for gpr_detection.io.synthetic (gprMax models, CHECKLIST M2-4b).

Reads the derived ``data/synthetic_gprmax`` folder and is skipped when it is
missing. Depth estimation is not run here.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from gpr_detection.io.paths import DEFAULT_SYNTHETIC_ROOT
from gpr_detection.io.radargram import Radargram
from gpr_detection.io.synthetic import (
    SyntheticTruth,
    list_synthetic_models,
    load_synthetic,
    load_synthetic_background,
    load_synthetic_truth,
)

EXPECTED_MODEL_COUNT = 20
BACKGROUND_TRACES = 3


def _require_root() -> Path:
    if not (DEFAULT_SYNTHETIC_ROOT / "manifest.csv").is_file():
        pytest.skip(f"synthetic data not found at {DEFAULT_SYNTHETIC_ROOT}")
    return DEFAULT_SYNTHETIC_ROOT


def _json(model_id: str) -> dict:
    path = DEFAULT_SYNTHETIC_ROOT / f"{model_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model_ids() -> list[str]:
    return list_synthetic_models(_require_root())


@pytest.mark.data
def test_manifest_lists_twenty_models(model_ids: list[str]) -> None:
    assert len(model_ids) == EXPECTED_MODEL_COUNT


@pytest.mark.data
def test_every_model_loads_with_survey_metadata_only(model_ids: list[str]) -> None:
    for model_id in model_ids:
        meta = _json(model_id)
        radargram = load_synthetic(model_id)
        assert isinstance(radargram, Radargram)
        assert radargram.data.dtype == np.float32
        assert radargram.data.shape == (meta["n_samples"], meta["n_traces"])
        assert radargram.dt_ns == meta["dt_ns"]
        assert radargram.dx_m == meta["dx_m"]
        assert radargram.f_c_hz == meta["center_freq_hz"]
        expected_gap = "air" if meta["air_gap_m"] > 0 else "ground"
        assert radargram.air_gap == expected_gap, model_id
        assert radargram.antenna_offset_m == meta["antenna_offset_m"]
        assert radargram.eps_r_meta == meta["eps_r"]
        assert radargram.eps_r_meta_source == "synthetic_truth"
        assert radargram.pipe_radius_m == meta["pipe_radius_m"]
        assert not radargram.amplitude_processed
        assert (radargram.dataset, radargram.line_id) == ("synthetic_gprmax", model_id)
        # No ground-truth time or depth reaches the radargram.
        assert dict(radargram.t0_candidates_ns) == {}
        assert set(radargram.metadata) == {
            "model_id",
            "trace_x_m_first",
            "time_of_sample0_ns",
        }


@pytest.mark.data
def test_air_and_ground_models_are_both_present(model_ids: list[str]) -> None:
    gaps = {load_synthetic_truth(model_id).air_gap_m > 0 for model_id in model_ids}
    assert gaps == {True, False}


@pytest.mark.data
def test_truth_matches_the_json(model_ids: list[str]) -> None:
    for model_id in model_ids:
        meta = _json(model_id)
        truth = load_synthetic_truth(model_id)
        assert isinstance(truth, SyntheticTruth)
        assert truth.pipe_top_depth_m == meta["pipe_top_depth_m"]
        assert truth.velocity_m_per_ns == meta["velocity_m_per_ns"]
        assert truth.pipe_radius_m == meta["pipe_radius_m"]
        assert truth.air_gap_m == meta["air_gap_m"]
        assert truth.t0_top_ns == meta["t0_top_ns"]
        assert truth.apex_trace_index == meta["apex"]["trace_index"]
        assert truth.apex_envelope_peak_ns == meta["apex"]["envelope_peak_ns"]
        zeros = meta["time_zero_candidates_ns"]
        assert truth.source_peak_ns == zeros["source_peak"]
        assert truth.direct_wave_peak_ns == zeros["direct_wave_peak"]
        assert truth.surface_reflection_peak_ns == zeros["surface_reflection_peak"]
        assert len(truth.ray_reflection_time_ns) == meta["n_traces"]
        assert truth.pipe_material == meta.get("pipe_material")


@pytest.mark.data
def test_water_filled_pipe_material_is_read(model_ids: list[str]) -> None:
    assert "M18" in model_ids
    assert load_synthetic_truth("M18").pipe_material == "pvc_water"


@pytest.mark.data
def test_eps_and_radius_can_be_withheld(model_ids: list[str]) -> None:
    radargram = load_synthetic(model_ids[0], include_eps=False, include_radius=False)
    assert radargram.eps_r_meta is None
    assert radargram.eps_r_meta_source == ""
    assert radargram.pipe_radius_m is None


@pytest.mark.data
def test_background_is_three_no_pipe_traces(model_ids: list[str]) -> None:
    model_id = model_ids[0]
    background = load_synthetic_background(model_id)
    assert background.dtype == np.float32
    assert background.shape == (_json(model_id)["n_samples"], BACKGROUND_TRACES)


@pytest.mark.data
def test_missing_required_field_is_an_error(tmp_path: Path) -> None:
    root = _require_root()
    model_id = list_synthetic_models(root)[0]
    shutil.copy(root / f"{model_id}.npy", tmp_path / f"{model_id}.npy")
    meta = _json(model_id)
    del meta["t0_top_ns"]
    del meta["apex"]["trace_index"]
    (tmp_path / f"{model_id}.json").write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match=r"t0_top_ns.*apex\.trace_index"):
        load_synthetic_truth(model_id, root=tmp_path)
    with pytest.raises(ValueError, match="missing required fields"):
        load_synthetic(model_id, root=tmp_path)
