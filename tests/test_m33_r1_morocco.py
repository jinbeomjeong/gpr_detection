"""M3-3 R1 Morocco evaluation (``analysis/detection/m33_morocco_eval.py``).

Synthetic 224 x 224 images and a stand-in detector; no model and no stored
image is read. Covers the input checks, the condition transforms, the
Ultralytics-equal AP, P5 (structural zero and its failure), the run-once
rule and the stdout contract ([RUN] §7 4, 5, 7, 8), and the C1 stage-2
additions: 5-fold-mean bootstrap, case PNGs and the s5 period assertion.
"""

import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "detection"))
import m31_d1_common as common
import m32_recipe_inputs as inputs
import m33_inputs
import m33_morocco_eval as r1

from gpr_detection.detect.base import (
    Detection,
    DetectorOutput,
    InputTransform,
    PixelBox,
)
from gpr_detection.pipeline import PP_SETS, PPSetId
from m33_close_blob import accept_close_pipeline_blob

METRIC_WORDS = re.compile(r"ap50|recall|precision|p5", re.IGNORECASE)
FINAL_KEYS = {"status", "out_dir", "n_images", "elapsed_s"}


def _image(stem: str, subset: str, boxes_norm, classes) -> r1.MoroccoImage:
    rng = np.random.default_rng(len(stem))
    pixels = rng.integers(0, 255, (224, 224), dtype=np.uint8)
    return r1.MoroccoImage(
        stem=stem,
        subset=subset,
        group_id=stem,
        pixels=pixels,
        boxes_norm=np.asarray(boxes_norm, dtype=float).reshape(-1, 4),
        classes=np.asarray(classes, dtype=int),
        gt_index=np.arange(len(classes)),
    )


def _rows(n: int = r1.N_IMAGES) -> list[dict]:
    return [
        {"image_id": f"U_{i}_aug_1", "group_id": f"U_{i}", "aug_transform": "잡음"}
        for i in range(n)
    ]


# ------------------------------------------------------------ input checks


def test_manifest_needs_233_rows_no_forbidden_copy_one_per_group() -> None:
    r1.check_manifest(_rows())
    with pytest.raises(inputs.InputError, match="rows"):
        r1.check_manifest(_rows(232))
    rotated = _rows()
    rotated[5]["aug_transform"] = "회전 ±15°"
    with pytest.raises(inputs.InputError, match="forbidden"):
        r1.check_manifest(rotated)
    twice = _rows()
    twice[7]["group_id"] = twice[8]["group_id"]
    with pytest.raises(inputs.InputError, match="groups"):
        r1.check_manifest(twice)


# -------------------------------------------------------------- conditions


@pytest.mark.parametrize(
    ("condition", "scale", "offset"),
    [
        ("main", 256 / 224, 0.0),
        ("s1_pad", 1.0, 16.0),
        ("s2_scale_0.75", 192 / 224, 32.0),
        ("s3_top_mask", 256 / 224, 0.0),
    ],
)
def test_ground_truth_follows_the_image_transform(condition, scale, offset) -> None:
    image = _image("U_1_aug_1", "utilities", [[0.25, 0.25, 0.5, 0.75]], [0])
    canvas = r1.condition_image(image, condition)
    assert canvas.pixels.shape == (256, 256) and canvas.pixels.dtype == np.uint8
    expected = np.array([56, 56, 112, 168], dtype=float) * scale + offset
    assert canvas.boxes == pytest.approx(expected[None, :])


def test_large_scale_crops_and_clips_the_ground_truth() -> None:
    image = _image(
        "U_1_aug_1", "utilities", [[0.0, 0.0, 0.1, 0.1], [0.4, 0.4, 0.6, 0.6]], [0, 2]
    )
    canvas = r1.condition_image(image, "s2_scale_1.33")
    # side 340, offset -42: the corner box leaves the canvas, the centre stays.
    assert len(canvas.boxes) == 1 and canvas.classes.tolist() == [2]
    assert canvas.boxes[0] == pytest.approx([136 - 42, 136 - 42, 204 - 42, 204 - 42])
    assert canvas.n_dropped == 1
    assert r1.condition_image(image, "main").n_dropped == 0


def test_top_mask_greys_the_first_28_source_rows() -> None:
    image = _image("U_1_aug_1", "intact", [], [])
    masked = r1.condition_image(image, "s3_top_mask").pixels
    assert np.all(masked[:31, :] == 128)  # 28 x 256 / 224 = 32 rows (edge mixes)


# ------------------------------------------------------------------ scoring


def _predictions(boxes, scores, classes) -> r1.Predictions:
    return r1.Predictions(
        np.asarray(boxes, dtype=float).reshape(-1, 4),
        np.asarray(scores, dtype=float),
        np.asarray(classes, dtype=int),
    )


def test_ap50_equals_the_ultralytics_equal_accumulator() -> None:
    truth = r1.CanvasImage(
        np.empty(0),
        np.array([[10, 10, 50, 50], [100, 100, 140, 140]], dtype=float),
        np.array([0, 0]),
        np.array([0, 1]),
    )
    prediction = _predictions(
        [[12, 12, 50, 50], [0, 0, 20, 20], [101, 99, 139, 141]],
        [0.9, 0.8, 0.3],
        [0, 0, 0],
    )
    matches = [r1.image_matches(truth, prediction)]
    accumulator = common.ScoreAccumulator()
    accumulator.add_image(
        prediction.boxes, prediction.scores, prediction.classes,
        truth.boxes, truth.classes,
    )  # fmt: skip
    expected = accumulator.class_results()["pipe_hyperbola"]["ap50"]
    result = r1.ap_from_matches(matches, r1.HYPERBOLA)
    assert result["ap50"] == expected
    assert result["ap50_95"] <= result["ap50"]
    scored = r1.Scored()
    scored.add(_image("U_1_aug_1", "utilities", [], []), truth, prediction)
    rp = r1.recall_precision(scored, r1.HYPERBOLA)
    assert (rp["recall_conf025"], rp["precision_conf025"]) == (1.0, 2 / 3)


def test_confusion_matrix_counts_background_both_ways() -> None:
    truth = r1.CanvasImage(
        np.empty(0), np.array([[10, 10, 50, 50]], float), np.array([2]), np.array([0])
    )
    prediction = _predictions(
        [[10, 10, 50, 50], [200, 200, 220, 220]], [0.9, 0.9], [0, 1]
    )
    scored = r1.Scored()
    scored.add(_image("C_1_aug_1", "cavities", [], []), truth, prediction)
    matrix = r1.confusion_matrix(scored)
    assert matrix["pipe_hyperbola"]["cavity"] == 1
    assert matrix["pipe_band"]["background"] == 1
    assert matrix["background"]["cavity"] == 0


def test_cavities_stay_out_of_the_main_metrics() -> None:
    truth = r1.CanvasImage(
        np.empty(0), np.array([[10, 10, 50, 50]], float), np.array([2]), np.array([0])
    )
    prediction = _predictions([[10, 10, 50, 50]], [0.9], [2])
    scored = r1.Scored()
    scored.add(_image("C_1_aug_1", "cavities", [], []), truth, prediction)
    metrics = r1.condition_metrics(scored)
    assert metrics["utilities_intact"]["n_images"] == 0
    assert metrics["cavities"]["cavity"]["ap50"] == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------- P5


def test_pp_on_square_crops_leaves_the_hyperbola_boxes() -> None:
    """S3 cannot act on a square crop (y_bot 256, no zero); S4 removes a
    low-score band near the bottom only."""
    raw = _predictions([[10, 10, 80, 80], [0, 240, 256, 255]], [0.9, 0.3], [0, 1])
    pre, post = r1.apply_crop_pp(raw, PP_SETS[PPSetId.S3])
    assert post.boxes.tolist() == raw.boxes.tolist() and set(pre.actions) == {"none"}
    _, post_s4 = r1.apply_crop_pp(raw, PP_SETS[PPSetId.S4])
    assert post_s4.classes.tolist() == [0]


def test_p5_structural_zero_and_its_failure() -> None:
    def metrics(pre: float, post: float) -> dict:
        def one(value):
            return {"utilities_intact": {"pipe_hyperbola": {"ap50": value}}}

        return {"main/pre": one(pre), "main/post": one(post)}

    zero = r1.p5_record({0: metrics(0.4, 0.4), 1: metrics(0.5, 0.5)})
    assert zero["p5_structural_zero"] is True and zero["p5_adoption_input"] == 0.0
    changed = r1.p5_record({0: metrics(0.4, 0.4), 1: metrics(0.5, 0.49)})
    assert changed["p5_structural_zero"] is False


def test_p5_without_hyperbola_predictions_is_unchanged_not_failed() -> None:
    """Smoke 2026-11-16: no ``pipe_hyperbola`` prediction in scope gives AP50
    ``None`` before and after PP; that is the same value, not a P5 break."""

    def metrics(pre, post) -> dict:
        def one(value):
            return {"utilities_intact": {"pipe_hyperbola": {"ap50": value}}}

        return {"main/pre": one(pre), "main/post": one(post)}

    none_both = r1.p5_record({0: metrics(None, None)})
    assert none_both["p5_structural_zero"] is True
    assert none_both["p5_adoption_input"] is None
    assert none_both["p5_folds_without_ap50"] == 1
    mixed = r1.p5_record({0: metrics(None, None), 1: metrics(0.3, 0.3)})
    assert mixed["p5_structural_zero"] is True and mixed["p5_adoption_input"] == 0.0
    one_side = r1.p5_record({0: metrics(0.3, None)})
    assert one_side["p5_structural_zero"] is False


def test_smoke_status_with_no_hyperbola_prediction_is_not_judgeable(
    offline_r1, monkeypatch, tmp_path, capsys
) -> None:
    """The smoke images may hold no ``pipe_hyperbola`` prediction at all."""

    class CavityOnly(FakeDetector):
        def predict(self, image):
            size = image.shape[0]
            box = Detection(2, "cavity", 0.7, PixelBox(40, 40, 120, 120))
            return DetectorOutput((box,), InputTransform(1, 1, 0, 0, size, size))

    monkeypatch.setattr(r1, "load_detector", lambda *_: CavityOnly())
    argv = ["--smoke", "--device", "cpu", "--out-root", str(tmp_path / "s")]
    code, final, _ = _main(argv, capsys)
    assert (code, final["status"]) == (1, "not_judgeable")
    summary = json.loads((tmp_path / "s" / "summary.json").read_text(encoding="utf-8"))
    assert summary["p5_structural_zero"] is True
    assert summary["s5_periods"]["asserted"] is False
    assert offline_r1 == []  # smoke reports the periods only


# --------------------------------------------------------------- bootstrap

GT_BOX = [10.0, 10.0, 50.0, 50.0]


def _truth(boxes, classes) -> r1.CanvasImage:
    return r1.CanvasImage(
        np.full((256, 256), 128, dtype=np.uint8),
        np.asarray(boxes, dtype=float).reshape(-1, 4),
        np.asarray(classes, dtype=int),
        np.arange(len(classes)),
    )


def _scored(stems, prediction_of) -> r1.Scored:
    scored = r1.Scored()
    for stem in stems:
        image = _image(stem, "utilities", [], [])
        scored.add(image, _truth([GT_BOX], [0]), prediction_of(stem))
    return scored


def test_bootstrap_scores_the_same_draw_in_every_fold_and_averages() -> None:
    """A fold that finds every pipe and one that finds none: the recall of
    every draw is their mean, 0.5, only if both folds score the same draw."""
    stems = ["U_1_aug_1", "U_2_aug_1", "U_3_aug_1"]
    found = _scored(stems, lambda _: _predictions([GT_BOX], [0.9], [0]))
    empty = _scored(stems, lambda _: _predictions([], [], []))

    result = r1.bootstrap_fold_mean({0: found, 1: empty}, seed=1, n_draws=50)

    recall = result["utilities_intact_pipe_hyperbola"]["recall_conf025"]
    assert (recall["low"], recall["high"], recall["n"]) == (0.5, 0.5, 50)
    assert result["cavities_cavity"] == {"ap50": None, "recall_conf025": None}
    assert (result["folds"], result["n_draws"], result["seed"]) == ([0, 1], 50, 1)
    again = r1.bootstrap_fold_mean({0: found, 1: empty}, seed=1, n_draws=50)
    assert again == result


def test_bootstrap_refuses_folds_with_different_images() -> None:
    def predict(_):
        return _predictions([GT_BOX], [0.9], [0])

    first = _scored(["U_1_aug_1", "U_2_aug_1"], predict)
    other = _scored(["U_1_aug_1", "U_3_aug_1"], predict)
    with pytest.raises(RuntimeError, match="different image lists"):
        r1.bootstrap_fold_mean({0: first, 1: other}, n_draws=2)


# ------------------------------------------------------------------- cases


def _case_scored() -> r1.Scored:
    """A: hit 0.6, B: hit 0.9, C: miss, D: false alarm only, E: nothing."""
    hit = [GT_BOX]
    rows = [
        ("A", [GT_BOX], [0], _predictions(hit, [0.6], [0])),
        ("B", [GT_BOX], [0], _predictions(hit, [0.9], [0])),
        ("C", [GT_BOX], [0], _predictions([], [], [])),
        ("D", [], [], _predictions([[100, 100, 140, 140]], [0.5], [1])),
        ("E", [], [], _predictions([], [], [])),
    ]
    scored = r1.Scored()
    for stem, boxes, classes, prediction in rows:
        image = _image(stem, "utilities", [], [])
        scored.add(image, _truth(boxes, classes), prediction)
    return scored


def test_case_ranks_follow_errors_then_score_then_name() -> None:
    ranks = r1.rank_cases(_case_scored(), n_cases=3)
    assert ranks == {"top": [1, 0, 2], "bottom": [2, 3, 0]}


def test_case_pngs_are_written_into_the_cases_folder(tmp_path) -> None:
    case_dir = r1.write_cases(tmp_path, _case_scored())
    names = sorted(path.name for path in case_dir.iterdir())
    assert case_dir == tmp_path / r1.CASE_DIR
    assert names[:5] == [
        "bottom_01_C.png",
        "bottom_02_D.png",
        "bottom_03_A.png",
        "bottom_04_B.png",
        "bottom_05_E.png",
    ]
    assert names[5:] == ["top_01_B.png", "top_02_A.png", "top_03_C.png"]


# ------------------------------------------------------------------ periods


def test_period_quantiles_follow_the_x6_rule() -> None:
    quantiles = r1.period_quantiles([10.0, None, 20.0])
    assert quantiles is not None
    assert (quantiles["p50"], quantiles["n"]) == (15.0, 2)
    assert r1.period_quantiles([None]) is None


def test_period_record_reads_the_x6_quantiles_per_subset() -> None:
    images = [
        replace(_image("U_1_aug_1", "utilities", [], []), period_px=20.0),
        replace(_image("I_1_aug_1", "intact", [], []), period_px=None),
    ]
    record = r1.period_record(images)
    assert record["all"]["x6"]["n"] == r1.N_IMAGES
    assert record["all"]["n_no_period"] == 1 and record["intact"]["measured"] is None
    assert set(record) >= {"all", "utilities", "intact", "x6_sha256_lf"}


def test_full_run_periods_must_equal_x6() -> None:
    quantiles = {"p5": 7.0, "p25": 16.0, "p50": 20.364, "p75": 24.889, "p95": 44.8}
    same = {"all": {"measured": quantiles, "n_no_period": 0, "x6": quantiles}}
    r1.assert_periods_match_x6(same)
    shifted = {"all": same["all"] | {"measured": quantiles | {"p50": 20.365}}}
    with pytest.raises(inputs.InputError, match="differ from X6"):
        r1.assert_periods_match_x6(shifted)
    missing = {"all": same["all"] | {"n_no_period": 1}}
    with pytest.raises(inputs.InputError, match="differ from X6"):
        r1.assert_periods_match_x6(missing)


def test_only_a_run_over_every_image_asserts_the_periods() -> None:
    assert r1.parse_args(["--folds", "0"]).uses_every_image
    assert not r1.parse_args(["--smoke"]).uses_every_image
    assert not r1.parse_args(["--limit", "5"]).uses_every_image


# ---------------------------------------------------------------------- run


class FakeDetector:
    class_names = common.CLASS_NAMES

    def __init__(self, change_hyperbola: bool = False) -> None:
        self.change_hyperbola = change_hyperbola

    def predict(self, image: np.ndarray) -> DetectorOutput:
        size = image.shape[0]
        detections = (
            Detection(0, "pipe_hyperbola", 0.7, PixelBox(40, 40, 120, 120)),
            Detection(1, "pipe_band", 0.3, PixelBox(0, 230, 256, 250)),
        )
        return DetectorOutput(detections, InputTransform(1, 1, 0, 0, size, size))


@pytest.fixture
def offline_r1(monkeypatch, tmp_path_factory):
    folder = tmp_path_factory.mktemp("r1in")
    c0b = folder / "d0_gate.json"
    selection = {"pp_set_id": "S3", "pp_config_sha256": PP_SETS[PPSetId.S3].sha256()}
    c0b.write_text(json.dumps({"selection": selection}), encoding="utf-8")
    monkeypatch.setattr(m33_inputs, "C0B_D0_GATE_JSON", c0b)
    monkeypatch.setattr(m33_inputs, "C0B_D0_GATE_SHA256", inputs.sha256_file(c0b))
    accept_close_pipeline_blob(monkeypatch)
    images = [
        _image("U_1_aug_1", "utilities", [[0.15, 0.15, 0.47, 0.47]], [0]),
        _image("I_1_aug_1", "intact", [], []),
        _image("C_1_aug_1", "cavities", [[0.1, 0.1, 0.3, 0.3]], [2]),
    ]
    monkeypatch.setattr(r1, "assert_inputs", lambda *_: {"manifest": "m"})
    monkeypatch.setattr(r1, "read_images", lambda *_: images)
    monkeypatch.setattr(r1, "load_detector", lambda *_: FakeDetector())
    monkeypatch.setattr(r1.inputs, "checked_sha256", lambda *_: "w")
    monkeypatch.setattr(r1, "BOOTSTRAP_B", 20)
    # The synthetic images have no X6 periods; record the assertion calls.
    asserted_records: list[dict] = []
    monkeypatch.setattr(r1, "assert_periods_match_x6", asserted_records.append)
    return asserted_records


def _main(argv, capsys) -> tuple[int, dict, list[str]]:
    code = r1.main(argv)
    lines = capsys.readouterr().out.strip().splitlines()
    return code, json.loads(lines[-1]), lines


def test_partial_run_writes_files_and_prints_no_metric(
    offline_r1, tmp_path, capsys
) -> None:
    out = tmp_path / "r1"
    argv = ["--folds", "0", "--device", "cpu", "--allow-dirty", "--out-root", str(out)]
    code, final, lines = _main(argv, capsys)

    assert (code, final["status"]) == (1, "not_judgeable")
    assert set(final) == FINAL_KEYS and final["n_images"] == 3
    assert not any(METRIC_WORDS.search(text) for text in lines)
    predictions = inputs.read_csv(out / "fold0" / "predictions.csv")
    assert {row["pp"] for row in predictions} == {"pre", "post"}
    assert {row["condition"] for row in predictions} == set(r1.CONDITIONS)
    objects = inputs.read_csv(out / "fold0" / "objects.csv")
    assert objects and all(len(row["x1"]) > 0 for row in objects)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["p5_structural_zero"] is True
    assert summary["p5_adoption_input"] == 0.0
    assert "partial_folds" in summary["judgment"]["blockers"]
    assert (
        "s4_near_duplicate/pre"
        in json.loads((out / "metrics.json").read_text(encoding="utf-8"))["0"]
    )
    assert summary["timings"]["infer_main"] >= 0
    assert summary["s5_periods"]["asserted"] is True
    assert len(offline_r1) == 1 and offline_r1[0]["all"]["x6"]["n"] == r1.N_IMAGES
    assert summary["n_gt_dropped_by_condition"]["s2_scale_1.33"] == 0
    assert set(summary["bootstrap_fold_mean"]["folds"]) == {0}
    assert f"case png: {out / r1.CASE_DIR}" in lines
    assert sorted(path.name for path in (out / r1.CASE_DIR).iterdir()) == [
        "bottom_01_C_1_aug_1.png",
        "bottom_02_I_1_aug_1.png",
        "bottom_03_U_1_aug_1.png",
        "top_01_U_1_aug_1.png",
        "top_02_C_1_aug_1.png",
    ]  # errors: C 3 (miss, two false), I 2 (two false), U 1 (low band box)

    code, final, _ = _main(argv, capsys)  # the run happens once
    assert (code, final["status"]) == (2, "input_error")


def test_a_changed_hyperbola_box_fails_p5(offline_r1, monkeypatch, tmp_path, capsys):
    """[RUN] §7 8: a PP that changes a hyperbola box ends with exit 1."""
    original = r1.apply_crop_pp

    def shrink_hyperbolas(predictions, config):
        pre, post = original(predictions, config)
        boxes = post.boxes.copy()
        boxes[post.classes == 0, 3] = boxes[post.classes == 0, 1] + 5  # IoU < 0.5
        return pre, r1.Predictions(boxes, post.scores, post.classes, post.actions)

    monkeypatch.setattr(r1, "apply_crop_pp", shrink_hyperbolas)
    out = tmp_path / "r1"
    argv = ["--folds", "0,1,2,3,4", "--device", "cpu", "--out-root", str(out)]
    monkeypatch.setattr(
        m33_inputs,
        "tree_state",
        lambda *_: {
            "head": "a",
            "dirty": False,
            "problems": [],
            "code_blobs": {m33_inputs.PIPELINE_PATH: m33_inputs.C0A_PIPELINE_BLOB},
            "src_tree_id": "b",
        },
    )
    code, final, _ = _main(argv, capsys)
    assert (code, final["status"]) == (1, "gate_failed")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["p5_structural_zero"] is False
    assert summary["judgment"]["judgeable"] is True


def test_smoke_takes_eight_images_and_fold_0() -> None:
    settings = r1.parse_args(["--smoke"])
    assert settings.folds == (0,) and settings.smoke and settings.allow_dirty
    assert settings.out_root == m33_inputs.R1_SMOKE_OUT_ROOT
    images = [
        _image(f"{s[0].upper()}_{i}_aug_1", s, [], [])
        for s in ("utilities", "intact", "cavities")
        for i in range(5)
    ]
    chosen = r1.smoke_selection(images)
    assert len(chosen) == r1.SMOKE_IMAGES
    assert [i.subset for i in chosen].count("cavities") == 2
