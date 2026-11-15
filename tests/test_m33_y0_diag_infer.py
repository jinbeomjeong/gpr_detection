"""M3-3 Y0 inference experiments E0-E4 (``analysis/processing/m33_y0_diag_infer.py``).

No model and no GPU: images come from small analytic fixtures (a ground
coupled record from ``synthetic_bscan`` and an air-coupled 4TU-like line with
an analytic pipe response, as in ``test_inject``), detections from the
``fake_line_detector`` stand-in or a fixed-box detector. One test compares
E0 with the D0 Y0 image on the real M04 model (``data`` marker).
"""

import csv
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import convolve1d

from fake_line_detector import TileStatisticsDetector
from gpr_detection.detect.base import Detection, DetectorOutput, PixelBox, letterbox
from gpr_detection.eval import inject
from gpr_detection.io.radargram import Radargram
from gpr_detection.pipeline import LineBox
from synthetic_bscan import (
    T_SRC_NS,
    BScanSpec,
    cylinder_times,
    make_bscan,
    make_radargram,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "analysis" / "processing" / "m33_y0_diag_infer.py"
M04_RECORD = PROJECT_ROOT / "data" / "synthetic_gprmax" / "M04.npy"
TILE_PX = 256

UNIT_SPEC = BScanSpec(snr_db=None, n_traces=111, n_samples=800)
UNIT_APEX_TRACE = UNIT_SPEC.n_traces // 2

LINE_DT_NS = 0.098
LINE_DX_M = 0.02
F_C_HZ = 400e6
LINE_SAMPLES = 512
LINE_TRACES = 235
LINE_ID = "01.1/Path3"
SYN_DT_NS = 0.0118
SYN_SAMPLES = 3224
SYN_TRACES = 111
SYN_APEX_TRACE = 55.0
SYN_REFERENCE_NS = 4.0
SYN_APEX_NS = 16.0
E3_APEX_TRACE = 117.0  # whole-trace offset 62 from the synthetic apex


@pytest.fixture(scope="module")
def diag():
    spec = importlib.util.spec_from_file_location("m33_y0_diag_infer_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def unit_model(diag):
    """Ground-coupled record (direct wave + pipe) and its pipe-only field."""
    pipe_only = make_bscan(replace(UNIT_SPEC, direct_amplitude=0.0))
    apex_ns = cylinder_times(
        UNIT_SPEC.x_m,
        UNIT_SPEC.x0_m,
        UNIT_SPEC.d_top_m,
        UNIT_SPEC.velocity,
        UNIT_SPEC.radius_m,
    )[UNIT_APEX_TRACE]
    return diag.SyntheticModel(
        model_id="UNIT",
        radargram=make_radargram(UNIT_SPEC),
        pipe_only=pipe_only.astype(np.float64),
        apex_trace=float(UNIT_APEX_TRACE),
        apex_sample=(T_SRC_NS + apex_ns) / UNIT_SPEC.dt_ns,
    )


def _ricker(times_ns: np.ndarray) -> np.ndarray:
    arg = (np.pi * F_C_HZ * 1e-9 * times_ns) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def _fourtu_line(n_traces: int = LINE_TRACES) -> Radargram:
    """Air-coupled 4TU-like line: flat band plus laterally random texture."""
    rng = np.random.default_rng(0)
    times_ns = np.arange(LINE_SAMPLES) * LINE_DT_NS
    band = 10000.0 * _ricker(times_ns - 5.0)
    wavelet = _ricker(np.arange(-40, 41) * LINE_DT_NS)
    spikes = rng.normal(0.0, 1.0, (LINE_SAMPLES, n_traces))
    texture = 100.0 * convolve1d(spikes, wavelet, axis=0)
    return Radargram(
        data=(band[:, None] + texture).astype(np.float32),
        dt_ns=LINE_DT_NS,
        dx_m=LINE_DX_M,
        air_gap="air",
        amplitude_processed=False,
        dataset="4tu",
        line_id=LINE_ID,
        f_c_hz=F_C_HZ,
    )


@pytest.fixture(scope="module")
def context():
    return inject.prepare_line_context(_fourtu_line())


@pytest.fixture(scope="module")
def response() -> inject.PipeResponse:
    """Analytic symmetric hyperbola on a gprMax-like grid (as in test_inject)."""
    times_ns = np.arange(SYN_SAMPLES) * SYN_DT_NS
    offsets_m = (np.arange(SYN_TRACES) - SYN_APEX_TRACE) * LINE_DX_M
    depth_m = 0.1 * (SYN_APEX_NS - SYN_REFERENCE_NS) / 2.0
    arrivals_ns = SYN_REFERENCE_NS + 2.0 / 0.1 * np.sqrt(depth_m**2 + offsets_m**2)
    beam = np.exp(-((offsets_m / 0.4) ** 2))
    data = beam[None, :] * _ricker(times_ns[:, None] - arrivals_ns[None, :])
    return inject.PipeResponse(
        response_id="M12",
        data=data,
        dt_ns=SYN_DT_NS,
        dx_m=LINE_DX_M,
        apex_trace=SYN_APEX_TRACE,
        apex_time_ns=SYN_APEX_NS,
        reference_time_ns=SYN_REFERENCE_NS,
        coupling="air",
        air_gap_m=0.1,
        eps_r=9.0,
        f_c_hz=F_C_HZ,
        top_depth_m=depth_m,
        od_m=0.2,
        material="pec",
        quality={inject.MIRROR_DIFF_KEY: 0.0, inject.EARLY_ENERGY_KEY: 0.0},
    )


def _spec(
    apex_trace: float = E3_APEX_TRACE,
    *,
    response_id: str = "M12",
    line_id: str = LINE_ID,
    snr_db: float = 12.0,
    polarity: int = 1,
    k: float = inject.K_MAIN,
) -> inject.InjectionSpec:
    realization_id = f"{line_id}:test:snr{snr_db:g}:k{k}:p{polarity}"
    return inject.InjectionSpec(
        injection_id=f"{realization_id}:{response_id}:{apex_trace}",
        realization_id=realization_id,
        line_id=line_id,
        apex_trace=apex_trace,
        response_id=response_id,
        snr_db=snr_db,
        polarity=polarity,
        k=k,
        seed=0,
    )


class FixedBoxDetector:
    """Reports one box at fixed tile coordinates on every tile."""

    class_names = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")

    def __init__(self, box: PixelBox, score: float, class_id: int = 0) -> None:
        self.box = box
        self.score = score
        self.class_id = class_id

    def predict(self, image: np.ndarray) -> DetectorOutput:
        _, transform = letterbox(image, TILE_PX)
        detection = Detection(
            self.class_id,
            self.class_names[self.class_id],
            self.score,
            transform.apply(self.box),
        )
        return DetectorOutput((detection,), transform)


def _apex_px(item) -> tuple[float, float]:
    """Truth apex in line L1 (column, row) of the item's image."""
    grid_map = item.image.grid_map
    return (
        item.apex_trace / grid_map.traces_per_px,
        item.apex_sample / grid_map.samples_per_px,
    )


def _box_around(column: float, row: float, half_width: float = 5.0) -> PixelBox:
    return PixelBox(column - half_width, row - 3.0, column + half_width, row + 10.0)


# ------------------------------------------------------------- E0, D0 path


@pytest.mark.data
def test_e0_is_the_d0_y0_image_and_apex(diag):
    if not M04_RECORD.is_file():
        pytest.skip("synthetic model M04 missing")
    model = diag.load_synthetic_model("M04", diag.InputGuard())
    e0 = diag.e0_image(model)
    y0 = diag.d0.synthetic_lines("M04", diag.d0.InputGuard())[0]
    assert y0.condition == "Y0"
    assert np.array_equal(e0.image.line_image, y0.image.line_image)
    assert e0.image.tiles == y0.image.tiles
    assert e0.image.grid_map == y0.image.grid_map
    assert e0.image.period_samples == y0.image.period_samples
    assert (e0.apex_trace, e0.apex_sample) == (y0.apex_trace, y0.apex_sample)


# ----------------------------------------------------------------- E1, E2


def test_e1_changes_only_the_gain(diag, unit_model):
    e0 = diag.e0_image(unit_model)
    e1 = diag.e1_images(unit_model)
    assert [item.variant for item in e1] == ["G4", "G2_b0.5"]
    for item in e1:
        assert item.image.recipe_id == item.variant
        assert item.image.grid_map == e0.image.grid_map
        assert item.image.tiles == e0.image.tiles
        assert item.image.period_samples == e0.image.period_samples
        assert not np.array_equal(item.image.line_image, e0.image.line_image)
    assert diag.GAIN_RECIPES["G4"].gain_mode == "none"
    g2 = diag.GAIN_RECIPES["G2_b0.5"]
    assert (g2.gain_mode, g2.agc_floor_beta) == ("agc_capped", 0.5)


def test_unit_noise_is_fixed_per_model(diag):
    first = diag.unit_noise("M01", (50, 7))
    assert np.array_equal(first, diag.unit_noise("M01", (50, 7)))
    assert not np.array_equal(first, diag.unit_noise("M02", (50, 7)))


def test_e2_noise_gives_the_snr_after_the_linear_steps(diag, unit_model):
    e0 = diag.e0_image(unit_model)
    images = diag.e2_images(unit_model, e0.image.period_samples)
    assert [item.snr_db for item in images] == list(diag.SNR_STEPS_DB)
    assert [item.variant for item in images] == ["snr6", "snr12", "snr18", "snr30"]
    noise = diag.unit_noise(unit_model.model_id, unit_model.pipe_only.shape)
    for item in images:
        scaled = item.details["noise_scale"] * noise
        filtered = diag.linear_steps(unit_model.radargram, scaled)
        noise_rms = math.sqrt(float(np.mean(filtered**2)))
        snr_db = 20.0 * math.log10(item.details["signal_amplitude"] / noise_rms)
        assert snr_db == pytest.approx(item.snr_db, abs=1e-4)
        # The E0 period is held: same scale and tiles, only the content differs.
        assert item.image.period_samples == e0.image.period_samples
        assert item.image.period_source == "override"
        assert item.image.grid_map == e0.image.grid_map


def test_signal_is_the_apex_envelope_peak_of_the_pipe(diag, unit_model):
    period = diag.e0_image(unit_model).image.period_samples
    filtered = diag.linear_steps(unit_model.radargram, unit_model.pipe_only)
    peak = diag.apex_envelope_peak(
        filtered, unit_model.apex_trace, unit_model.apex_sample, period
    )
    # The apex column carries the largest envelope of the whole pipe field.
    far_column = diag.apex_envelope_peak(filtered, 5.0, unit_model.apex_sample, period)
    assert peak > far_column > 0


# --------------------------------------------------------------------- E4


def test_e4_centres_the_content_and_keeps_the_bottom_padding(diag, unit_model):
    e0 = diag.e0_image(unit_model)
    e4 = diag.e4_image(e0)
    width = e0.image.width_px
    shift = e4.content_shift_px
    assert e4.experiment == "E4" and e4.image is e0.image
    assert shift == (TILE_PX - width) // 2 > 0
    pixels = diag.tile_pixels(e4, 0)
    pad = e0.image.pad_gray
    assert np.all(pixels[:, :shift] == pad)
    assert np.all(pixels[:, shift + width :] == pad)
    assert np.array_equal(
        pixels[:, shift : shift + width], e0.image.tile_image(0)[:, :width]
    )
    assert np.all(pixels[e0.image.content_height_px :] == pad)


def test_e4_boxes_are_moved_back_before_the_match(diag, unit_model):
    e0 = diag.e0_image(unit_model)
    e4 = diag.e4_image(e0)
    column, row = _apex_px(e0)
    detector = FixedBoxDetector(_box_around(column + e4.content_shift_px, row), 0.4)
    images = [e0, e4]
    jobs = diag.tile_jobs(images)
    pixels = [diag.tile_pixels(images[j.image_index], j.tile_index) for j in jobs]
    e0_row, e4_row = diag.score_fold(detector, images, jobs, pixels, fold=3)
    # The same tile box sits on the apex only in the centred tile.
    assert e4_row["best_apex_hyperbola_score"] == 0.4
    assert e0_row["best_apex_hyperbola_score"] is None
    assert (e0_row["fold"], e4_row["variant"]) == (3, "centred")


def test_a_line_wider_than_a_tile_has_no_e4(diag):
    wide = diag.DiagImage(
        diag.Experiment.E0,
        "r0",
        "W",
        "W",
        diag.make_detection_image(_fourtu_line(n_traces=700)),
        350.0,
        200.0,
    )
    assert len(wide.image.tiles) > 1
    assert diag.e4_image(wide) is None


# ------------------------------------------------------------------- match


def test_apex_scores_take_the_best_box_containing_the_apex(diag, unit_model):
    e0 = diag.e0_image(unit_model)
    column, row = _apex_px(e0)

    def line_box(class_id: int, score: float, box: PixelBox) -> LineBox:
        name = diag.common.CLASS_NAMES[class_id]
        return LineBox(class_id, name, score, box, tile_index=0)

    boxes = [
        line_box(0, 0.3, _box_around(column, row)),
        line_box(0, 0.6, _box_around(column, row, half_width=12.0)),
        line_box(3, 0.9, _box_around(column, row, half_width=8.0)),
        line_box(0, 0.95, _box_around(column + 40.0, row)),
    ]
    scores = diag.apex_scores(e0, boxes)
    assert scores.n_line_boxes == 4
    assert scores.n_apex_hyperbola_boxes == 2
    assert scores.best_apex_hyperbola_score == 0.6
    assert (scores.best_apex_any_class_score, scores.best_apex_any_class) == (
        0.9,
        "manhole",
    )


def test_inferred_tiles_reach_the_apex_margin(diag):
    image = diag.make_detection_image(_fourtu_line(n_traces=700))
    traces_per_px = image.grid_map.traces_per_px
    for apex_trace in (3.0, 350.0, 696.0):
        item = diag.DiagImage(
            diag.Experiment.E3, "snr6", "M12", LINE_ID, image, apex_trace, 200.0
        )
        assert item.tile_indices
        first_px = (apex_trace - 3) / traces_per_px
        end_px = (apex_trace + 4) / traces_per_px
        for tile in image.tiles:
            reaches = (
                tile.x0_px < end_px and first_px < tile.x0_px + tile.content_width_px
            )
            assert (tile.index in item.tile_indices) == reaches


# --------------------------------------------------------------------- E3


def test_e3_keeps_the_stage_c_gain_and_undoes_only_the_taper(diag, context, response):
    spec = _spec()
    tapered = inject.inject_realization(context, [spec], {"M12": response})
    untapered = diag.inject_without_taper(context, response, spec)
    truth = tapered.truths[0]
    assert untapered.truth.gain == truth.gain
    assert (untapered.truth.apex_trace, untapered.truth.apex_sample) == (
        truth.apex_trace,
        truth.apex_sample,
    )
    patch = inject.build_response_patch(response, context)
    taper = patch.n_aperture_taper
    assert untapered.n_aperture_taper == taper > 0
    offset = round(E3_APEX_TRACE - SYN_APEX_TRACE)
    inner = slice(offset + taper, offset + SYN_TRACES - taper)
    data = untapered.radargram.data
    assert data.dtype == np.float32
    assert np.array_equal(data[:, inner], tapered.data.astype(np.float32)[:, inner])
    background = context.background.data
    assert np.array_equal(data[:, :offset], background[:, :offset])
    assert np.array_equal(
        data[:, offset + SYN_TRACES :], background[:, offset + SYN_TRACES :]
    )
    # The first aperture column carried weight ~0.006; now it carries 1.
    edge = offset
    added = data[:, edge].astype(np.float64) - background[:, edge]
    expected = truth.gain * diag.untapered_field(patch)[:LINE_SAMPLES, 0]
    assert np.allclose(added, expected, rtol=1e-5, atol=0.05)
    assert (
        np.abs(expected).max()
        > 20 * np.abs(tapered.data[:, edge] - background[:, edge]).max()
    )


def test_e3_images_cover_every_snr_step(diag, context, response):
    found = diag.e3_images(diag.StageCPosition(_spec(snr_db=0.0)), context, response)
    assert found.skipped == []
    assert [item.snr_db for item in found.images] == list(diag.SNR_STEPS_DB)
    gains = [item.details["gain"] for item in found.images]
    for item, gain in zip(found.images, gains, strict=True):
        assert (item.experiment, item.line_id) == ("E3", LINE_ID)
        assert item.apex_trace == E3_APEX_TRACE
        expected = 10.0 ** (item.snr_db / 20.0) * item.details["noise_rms"]
        assert gain == pytest.approx(expected / item.details["signal_amplitude"])
        assert item.tile_indices


def test_e3_rejection_is_listed_as_skipped(diag, context, response):
    position = diag.StageCPosition(_spec(apex_trace=10.0))
    found = diag.e3_images(position, context, response)
    assert found.images == []
    assert {entry["reason"] for entry in found.skipped} == {"aperture_outside_line"}
    assert len(found.skipped) == len(diag.SNR_STEPS_DB)


def test_stage_c_positions_take_the_first_4tu_rows_per_model(diag):
    fourtu = {"A/Path1", "B/Path1"}
    ids_line = "pipe/1/LID1"
    specs = [
        _spec(50.0, line_id=ids_line, response_id="M01"),
        _spec(84.0, line_id="A/Path1", response_id="M01", polarity=-1),
        _spec(84.0, line_id="A/Path1", response_id="M01", k=0.8),
        _spec(84.0, line_id="A/Path1", response_id="M01", snr_db=0.0),
        _spec(84.0, line_id="A/Path1", response_id="M01", snr_db=6.0),
        _spec(141.0, line_id="A/Path1", response_id="M02"),
        _spec(90.0, line_id="B/Path1", response_id="M01"),
        _spec(60.0, line_id="B/Path1", response_id="M99"),
    ]
    models = ("M02", "M01", "M03")
    first = diag.stage_c_positions(specs, fourtu, models, per_model=1)
    assert [(p.model_id, p.line_id, p.spec.apex_trace) for p in first] == [
        ("M02", "A/Path1", 141.0),
        ("M01", "A/Path1", 84.0),
    ]
    two = diag.stage_c_positions(specs, fourtu, models, per_model=2)
    assert [(p.model_id, p.spec.apex_trace) for p in two] == [
        ("M02", 141.0),
        ("M01", 84.0),
        ("M01", 90.0),
    ]


# --------------------------------------------------------- guard, outputs


def test_guard_refuses_aihub_and_the_d0_forbidden_inputs(diag, tmp_path):
    guard = diag.InputGuard()
    for relative in (
        "data/aihub_yolo_v1/images/a_g1.0_x.jpg",
        "data/aihub_yolo_v1/test.yaml",
        "data/processing/d3/lines.csv",
        "data/morocco_eval_v1/x.csv",
    ):
        with pytest.raises(diag.common.InputError):
            guard.check(PROJECT_ROOT / relative)
    with pytest.raises(diag.common.InputError):
        guard.check(tmp_path / "line.npy")
    assert guard.check(diag.STAGE_C_TABLE) == diag.STAGE_C_TABLE.resolve()
    outside = diag.InputGuard(external_roots=(tmp_path,))
    assert outside.check(tmp_path / "line.npy") == (tmp_path / "line.npy").resolve()


@pytest.mark.parametrize(
    "relative",
    [
        "runs/eval/m33_d0",
        "runs/eval/m33_d0r",
        "RUNS/eval/M33_D0R/sub",
        "runs/eval/m33_d0/x",
    ],
)
def test_the_d0_folders_are_never_an_output(diag, relative):
    with pytest.raises(diag.common.InputError, match="read-only"):
        diag.check_out_root(PROJECT_ROOT / relative, overwrite=True)


def test_an_existing_output_needs_overwrite(diag, tmp_path):
    with pytest.raises(diag.common.InputError, match="overwrite"):
        diag.check_out_root(tmp_path, overwrite=False)
    diag.check_out_root(tmp_path, overwrite=True)
    diag.check_out_root(tmp_path / "new", overwrite=False)


def test_main_refuses_a_d0_folder_before_any_input(diag, monkeypatch, capsys):
    monkeypatch.setattr(diag, "_prepare", lambda settings: pytest.fail("inputs opened"))
    code = diag.main(["--out-root", str(PROJECT_ROOT / "runs" / "eval" / "m33_d0r")])
    final = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 2
    assert final["status"] == "input_error"


@pytest.mark.parametrize(
    "argv",
    [["--folds", "0,0"], ["--models", "M01,M21"], ["--experiments", "E5"]],
)
def test_bad_choices_are_input_errors(diag, argv, tmp_path, capsys):
    code = diag.main([*argv, "--out-root", str(tmp_path / "out")])
    assert code == 2
    assert not (tmp_path / "out").exists()
    capsys.readouterr()


# ------------------------------------------------------------------- run


@pytest.fixture
def fixture_images(diag, unit_model, context, response):
    found = diag.synthetic_model_images(
        unit_model, [e for e in diag.Experiment if e != "E3"]
    )
    found.add(diag.e3_images(diag.StageCPosition(_spec()), context, response))
    return found


def _settings(diag, out_root: Path, **changes):
    settings = diag.RunSettings(
        out_root=out_root,
        folds=(0, 1),
        models=("UNIT",),
        experiments=tuple(diag.Experiment),
        device="cpu",
        e3_positions_per_model=1,
        overwrite=False,
        images_only=False,
    )
    return replace(settings, **changes)


def _patch_inputs(diag, monkeypatch, tmp_path, images) -> None:
    monkeypatch.setattr(diag, "build_images", lambda settings, guard: images)
    monkeypatch.setattr(diag, "new_run_identity", lambda settings, weights: {})
    monkeypatch.setattr(diag, "D0_APEX_CSV", tmp_path / "no_d0.csv")


def test_run_writes_one_record_per_image_and_fold(
    diag, monkeypatch, tmp_path, fixture_images, capsys
):
    _patch_inputs(diag, monkeypatch, tmp_path, fixture_images)
    out_root = tmp_path / "out"
    counts = diag.run(
        _settings(diag, out_root),
        diag.InputGuard(),
        lambda fold: TileStatisticsDetector(),
        {},
    )
    images = fixture_images.images
    assert [item.experiment for item in images].count("E2") == 4
    assert {item.experiment for item in images} == set(diag.Experiment)
    assert counts["n_images"] == len(images)
    assert counts["n_records"] == 2 * len(images)
    with (out_root / "records.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == diag.RECORD_HEADER
    assert [row["fold"] for row in rows] == ["0"] * len(images) + ["1"] * len(images)
    assert [row["image_id"] for row in rows[: len(images)]] == [
        item.image_id for item in images
    ]
    # Scores and coordinates are written as repr: they read back exactly.
    jobs = diag.tile_jobs(images)
    pixels = [diag.tile_pixels(images[j.image_index], j.tile_index) for j in jobs]
    direct = diag.score_fold(TileStatisticsDetector(), images, jobs, pixels, 0)
    for row, record in zip(rows, direct, strict=False):
        for name in ("apex_sample", "period_samples", "best_apex_hyperbola_score"):
            value = record[name]
            assert row[name] == ("" if value is None else repr(value))
    document = json.loads((out_root / "diag_infer.json").read_text(encoding="utf-8"))
    assert document["schema"] == diag.SCHEMA
    assert len(document["images"]) == len(images)
    assert document["e0_vs_d0"] == {"available": False}
    counted = sum(cell["records"] for cell in document["variant_counts"].values())
    assert counted == 2 * len(images)
    assert "E3/snr30" in document["variant_counts"]
    capsys.readouterr()


def test_images_only_writes_the_image_list_without_records(
    diag, monkeypatch, tmp_path, fixture_images, capsys
):
    _patch_inputs(diag, monkeypatch, tmp_path, fixture_images)
    out_root = tmp_path / "out"
    settings = _settings(diag, out_root, images_only=True)
    counts = diag.run(
        settings, diag.InputGuard(), lambda fold: pytest.fail("detector"), {}
    )
    assert counts["n_records"] == 0
    lines = (out_root / "records.csv").read_text(encoding="utf-8").splitlines()
    assert lines == [",".join(diag.RECORD_HEADER)]
    document = json.loads((out_root / "diag_infer.json").read_text(encoding="utf-8"))
    assert len(document["images"]) == len(fixture_images.images)
    assert document["resources"]["peak_vram_reserved_mb"] is None
    capsys.readouterr()


def test_e0_is_compared_with_the_d0_hits(diag, tmp_path, monkeypatch):
    d0_csv = tmp_path / "synthetic_apex.csv"
    d0_csv.write_text(
        "fold,condition,model,hit_S0\n0,Y0,M01,1\n0,Y0,M02,0\n0,Y1_1P,M02,1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(diag, "D0_APEX_CSV", d0_csv)
    records = [
        {
            "fold": 0,
            "experiment": "E0",
            "model": "M01",
            "best_apex_hyperbola_score": 0.3,
        },
        {
            "fold": 0,
            "experiment": "E0",
            "model": "M02",
            "best_apex_hyperbola_score": 0.26,
        },
        {
            "fold": 0,
            "experiment": "E1",
            "model": "M02",
            "best_apex_hyperbola_score": 0.9,
        },
        {
            "fold": 1,
            "experiment": "E0",
            "model": "M01",
            "best_apex_hyperbola_score": None,
        },
    ]
    guard = diag.InputGuard(external_roots=(tmp_path,))
    result = diag.compare_e0_with_d0(records, guard)
    assert result["n_compared"] == 2
    assert result["mismatches"] == [
        {"fold": 0, "model": "M02", "d0_hit": False, "e0_best_score": 0.26}
    ]
