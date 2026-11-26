"""Operational PP default of ``detect_line`` (M3-3 close, user decision 2026-11-16).

``docs/processing/m33_j1_judgment.md`` section 8: J1 judged the PP the R2
runner applied (C0b set S3, ``PPConfig`` ``eb2796dd...``); its conclusion
carries over to the operational default only if ``detect_line`` without
``pp_mode`` gives, bit for bit, what that runner computes with S3 named.

The reference is built from the R2 runner's own calls: ``load_line`` (zero),
``uninjected_jobs`` (r0 image), ``run_jobs`` (tile boxes), the R2 merge and
``apply_pp`` with the sealed set, and the R2 ``post`` boxes on the L0 grid
(as its J1 rows keep them). No model, stored result or dataset is read.
"""

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "detection"))
import m32_recipe_eval as runner
import m32_recipe_inputs as inputs
import m33_inputs
import m33_r2_eval as r2

import m33_close_blob
from fake_line_detector import TileStatisticsDetector
from gpr_detection import pipeline
from gpr_detection.io.radargram import GridBox
from gpr_detection.pipeline import (
    DEFAULT_MERGE_CONFIG,
    DEFAULT_PP_SET,
    PP_SETS,
    LineBox,
    LineDetections,
    PPAction,
    PPOutcome,
    PPSetId,
    apply_pp,
    detect_line,
    merge_line_boxes,
)
from gpr_detection.preprocess.timezero import given_zero
from gpr_detection.preprocess.to_image import DEFAULT_DETECTION_IMAGE_CONFIG
from test_pipeline_tiles import VisiblePartDetector
from test_pp import _boundary_boxes, _pp_fixture_image

SEALED_S3_SHA256 = "eb2796ddfaa9611e5030342c4ed66b00c91ff2ed6a6698ee5979b9afe080602f"
"""``PPConfig`` of the C0b selection (``m33_c0b_d0_judgment.md``, J1 section 1)."""
C0A_SEALED_PIPELINE_BLOB = "dd5e7bcba4397efd732e7bf3ca1c5c92fb2404a1"
"""``pipeline.py`` blob of C0a (53c4e16), the value the runners assert."""
FIXTURE_ZERO_K = 63.2
"""Zero sample of the PP fixture line (dt 1 ns), as ``test_pp._use_fixture``."""
LINE_REF = inputs.LineRef(inputs.DATASET_4TU, "01.1/Path3", "k")
SINGLE_FOLD = 0


@pytest.fixture(scope="module")
def inject_helpers() -> ModuleType:
    """``tests/test_inject.py`` (its synthetic 4TU-like air line)."""
    path = Path(__file__).resolve().parent / "test_inject.py"
    spec = importlib.util.spec_from_file_location("inject_pp_default_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _r2_reference(
    data: runner.LineData, detector: object
) -> tuple[tuple[PPOutcome, ...], tuple[tuple[LineBox, GridBox], ...]]:
    """R2 runner with the sealed S3 named: outcomes and ``post`` boxes on L0."""
    settings = SimpleNamespace(flip=False, recipes=(inputs.R0,))
    jobs, _ = runner.uninjected_jobs(data, settings)  # type: ignore[arg-type]
    (job,) = jobs
    tile_boxes = runner.run_jobs(jobs, {SINGLE_FOLD: detector}, batch_size=1)
    geometry = r2._geometry(job.image, data)
    merged = merge_line_boxes(
        tile_boxes[(0, SINGLE_FOLD)], job.image, DEFAULT_MERGE_CONFIG
    )
    outcomes = apply_pp(merged, geometry.pp, PP_SETS[PPSetId.S3])
    on_grid = []
    for box in r2.pp_versions(outcomes)[r2.PP_POST]:
        grid = r2._grid(box, job.image)
        if grid is not None:
            on_grid.append((box, grid))
    return outcomes, tuple(on_grid)


def _assert_bit_identical(
    detections: LineDetections,
    outcomes: Sequence[PPOutcome],
    boxes: Sequence[tuple[LineBox, GridBox]],
) -> None:
    """Equal values and equal ``repr`` (float bits, e.g. -0.0 vs 0.0)."""
    assert detections.pp_outcomes == tuple(outcomes)
    assert detections.boxes == tuple(boxes)
    assert repr(detections.pp_outcomes) == repr(tuple(outcomes))
    assert repr(detections.boxes) == repr(tuple(boxes))


def test_default_set_is_the_sealed_c0b_s3() -> None:
    config = PP_SETS[DEFAULT_PP_SET]
    assert DEFAULT_PP_SET is PPSetId.S3 is m33_inputs.C0B_PP_SET_ID
    assert config.sha256() == SEALED_S3_SHA256 == m33_inputs.C0B_PP_CONFIG_SHA256
    assert not config.is_off
    assert inputs.recipe_config(inputs.R0) == DEFAULT_DETECTION_IMAGE_CONFIG


def test_default_equals_the_r2_s3_path_on_the_boundary_fixture(monkeypatch) -> None:
    """Every S3 rule acts (a, b clip, d); the default matches R2 bit for bit."""
    image = _pp_fixture_image()
    zero = given_zero(FIXTURE_ZERO_K, 1.0)
    monkeypatch.setattr(pipeline, "make_detection_image", lambda *a, **k: image)
    monkeypatch.setattr(pipeline, "line_zero", lambda radargram: zero)
    monkeypatch.setattr(runner, "make_detection_image", lambda *a, **k: image)
    radargram = object()

    default = detect_line(
        radargram,  # type: ignore[arg-type]
        VisiblePartDetector(image, _boundary_boxes()),
    )
    named = detect_line(
        radargram,  # type: ignore[arg-type]
        VisiblePartDetector(image, _boundary_boxes()),
        pp_mode=PPSetId.S3,
    )
    data = runner.LineData(LINE_REF, radargram, zero)  # type: ignore[arg-type]
    outcomes, boxes = _r2_reference(data, VisiblePartDetector(image, _boundary_boxes()))

    assert {outcome.action for outcome in outcomes} == {
        PPAction.NONE,
        PPAction.REMOVED_A,
        PPAction.CLIPPED_B,
        PPAction.REMOVED_D,
    }
    _assert_bit_identical(default, outcomes, boxes)
    _assert_bit_identical(named, outcomes, boxes)
    assert default.pp_config is PP_SETS[PPSetId.S3]
    assert default.pp_config.sha256() == SEALED_S3_SHA256
    assert default.tile_seam_box == named.tile_seam_box


def test_default_equals_the_r2_s3_path_on_a_synthetic_line(
    monkeypatch, inject_helpers
) -> None:
    """Unpatched image and zero: a synthetic air line through both paths."""
    radargram = inject_helpers._radargram(inject_helpers._background_data())
    monkeypatch.setattr(runner, "load_radargram", lambda ref: radargram)
    data = runner.load_line(LINE_REF)

    default = detect_line(radargram, TileStatisticsDetector())
    outcomes, boxes = _r2_reference(data, TileStatisticsDetector())

    assert outcomes and any(o.action.is_processed for o in outcomes)
    _assert_bit_identical(default, outcomes, boxes)
    assert default.pp_config is not None
    assert default.pp_config.sha256() == SEALED_S3_SHA256


def test_close_blob_stand_in_needs_the_close_pipeline(monkeypatch) -> None:
    """Runner fixtures may swap in the close blob only while it is the tree's."""
    close = m33_close_blob.CLOSE_PIPELINE_BLOB
    assert m33_close_blob.working_tree_pipeline_blob() == close
    with monkeypatch.context() as patch:
        assert m33_close_blob.accept_close_pipeline_blob(patch) == close
        assert close == m33_inputs.C0A_PIPELINE_BLOB
    assert m33_inputs.C0A_PIPELINE_BLOB == C0A_SEALED_PIPELINE_BLOB

    monkeypatch.setattr(m33_close_blob, "working_tree_pipeline_blob", lambda: "0" * 40)
    with pytest.raises(AssertionError, match="M3-3 마감 뒤 바뀜"):
        m33_close_blob.accept_close_pipeline_blob(monkeypatch)
    assert close != m33_inputs.C0A_PIPELINE_BLOB  # the refusal set nothing


def test_s0_still_turns_pp_off(inject_helpers) -> None:
    radargram = inject_helpers._radargram(inject_helpers._background_data())
    off = detect_line(radargram, TileStatisticsDetector(), pp_mode=PPSetId.S0)
    assert off.pp_outcomes == () and off.pp_config is None
