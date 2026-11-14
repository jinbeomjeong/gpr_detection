"""M3-2 S3 stage-B script (``analysis/detection/m32_aihub_perturb.py``).

Covers the perturbations (determinism, polarity, padding, identity), the
condition list (count, stage-A recipes), the split assertions, the identity
gate, the paired common-set recall and the blind CLI contract. A stand-in
detector replaces YOLO; no test runs a model or reads a stage-B result.
"""

import csv
import hashlib
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

from gpr_detection.detect.base import (
    Detection,
    DetectorOutput,
    InputTransform,
    PixelBox,
)
from gpr_detection.preprocess import to_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "analysis" / "detection" / "m32_aihub_perturb.py"
METRIC_WORDS = re.compile(r"recall|ap50|map50|delta|n_gt", re.IGNORECASE)
FINAL_LINE_KEYS = {"status", "out_dir", "n_conditions", "n_images", "elapsed_s"}
CANVAS = 256

# Every synthetic crop has the same three boxes; the manhole sits low, so a
# tall padding puts its centre into the fill (ignore path).
LABEL_BOXES = (
    (0, 100.0, 20.0, 150.0, 50.0),
    (2, 60.0, 100.0, 80.0, 130.0),
    (3, 180.0, 220.0, 220.0, 250.0),
)
CROPS_PER_FOLD = 3
FOLDS = (0, 1)


@pytest.fixture(scope="module")
def s3() -> ModuleType:
    spec = importlib.util.spec_from_file_location("m32_aihub_perturb", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["m32_aihub_perturb"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def workdir() -> Iterator[Path]:
    # Not tmp_path: keep "test" out of the synthetic data paths, as the other
    # analysis tests do.
    path = Path(tempfile.mkdtemp(prefix="m32s3_"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _noise_crop(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (CANVAS, CANVAS), np.uint8)


# ------------------------------------------------------------- perturbations


def test_identity_returns_the_crop_bit_for_bit(s3: ModuleType) -> None:
    image = _noise_crop(1)
    identity = s3.build_conditions()[0]
    assert identity.name == s3.IDENTITY and identity.is_identity
    perturbed = s3.perturb_image(image, identity, pad_height_px=120)
    assert np.array_equal(perturbed, image)
    assert perturbed is not image


def test_keep_mapping_without_steps_is_the_identity_image(s3: ModuleType) -> None:
    # Guards the amplitude round trip q - 127.5 + 127.5 used by all mimics.
    image = _noise_crop(2)
    plain = s3.Perturbation("plain_keep", recipe_id="x")
    assert not plain.is_identity
    assert np.array_equal(s3.perturb_image(image, plain), image)


@pytest.mark.parametrize("index", range(12))
def test_perturbations_are_deterministic(s3: ModuleType, index: int) -> None:
    image = _noise_crop(3)
    condition = s3.build_conditions()[index]
    cdf = np.linspace(1 / 256, 1.0, 256)
    first = s3.perturb_image(image, condition, 90, cdf)
    second = s3.perturb_image(image.copy(), condition, 90, cdf)
    assert first.dtype == np.uint8 and first.shape == image.shape
    assert np.array_equal(first, second)


def test_polarity_inverts_content_and_keeps_padding(s3: ModuleType) -> None:
    image = _noise_crop(4)
    polarity = next(c for c in s3.build_conditions() if c.invert_polarity)
    assert np.array_equal(s3.perturb_image(image, polarity), 255 - image)
    padded = s3.Perturbation("pad_invert", pad=s3.PadMimic.FLAT, invert_polarity=True)
    result = s3.perturb_image(image, padded, pad_height_px=40)
    assert np.array_equal(result[:-40], 255 - image[:-40])
    assert np.all(result[-40:] == s3.PAD_GRAY)


def test_flat_padding_fills_the_bottom_rows_only(s3: ModuleType) -> None:
    image = _noise_crop(5)
    pad_p0 = next(c for c in s3.build_conditions() if c.name == "pad_p0")
    result = s3.perturb_image(image, pad_p0, pad_height_px=100)
    assert np.array_equal(result[:156], image[:156])
    assert np.all(result[156:] == 128)
    assert np.array_equal(s3.perturb_image(image, pad_p0, 0), image)


def test_taper_reaches_the_pad_grey_and_keeps_upper_rows(s3: ModuleType) -> None:
    image = _noise_crop(6)
    taper = next(c for c in s3.build_conditions() if c.name == "pad_p1_n2")
    result = s3.perturb_image(image, taper, pad_height_px=56).astype(int)
    taper_px = 2 * s3.PERIOD_AIHUB_PX
    content_rows = CANVAS - 56
    untouched = content_rows - int(np.ceil(taper_px))
    assert np.array_equal(result[:untouched], image[:untouched])
    last_row_step = np.abs(result[content_rows - 1] - 128)
    assert last_row_step.max() <= 2  # sin^2 weight of the last row is ~0.02
    assert np.all(result[content_rows:] == 128)


def test_mappings_match_the_s1_implementation(s3: ModuleType) -> None:
    amplitudes = _noise_crop(7).astype(float) - 127.5
    cdf = np.cumsum(np.full(256, 1 / 256))
    cdf[-1] = 1.0
    clip = float(np.percentile(np.abs(amplitudes), 98))
    assert np.array_equal(
        s3.symmetric_grey_levels(amplitudes, 98),
        to_image._symmetric_gray_levels(amplitudes, clip),
    )
    assert np.array_equal(
        s3.hist_matched_grey_levels(amplitudes, cdf),
        to_image._hist_matched_gray_levels(amplitudes, tuple(cdf)),
    )


def test_only_the_gain_only_conditions_remove_the_trace_mean(s3: ModuleType) -> None:
    conditions = s3.build_conditions(["r0", "r2", "r5", "r7"])
    with_mean_removal = {c.name for c in conditions if c.remove_trace_mean}
    assert with_mean_removal == {"gain_g0_agc8", "gain_g2_b0.5"}
    assert not any(m.remove_trace_mean for m in s3.RECIPE_MIMICS.values())


@pytest.mark.parametrize("name", ["gain_g0_agc8", "gain_g2_b0.5"])
def test_gain_only_conditions_ignore_a_column_grey_offset(
    s3: ModuleType, name: str
) -> None:
    # An NM crop centred on grey 125 instead of 127.5 must give the same image
    # as a centred one: the offset is removed before the AGC can amplify it.
    rng = np.random.default_rng(9)
    centred = rng.integers(20, 230, (CANVAS, CANVAS)).astype(np.uint8)
    column_offsets = rng.integers(-3, 4, CANVAS)
    shifted = (centred.astype(int) + column_offsets[None, :]).astype(np.uint8)
    condition = next(c for c in s3.build_conditions() if c.name == name)
    expected = s3.perturb_image(centred, condition)
    assert np.array_equal(s3.perturb_image(shifted, condition), expected)
    without_removal = s3.replace(condition, remove_trace_mean=False)
    assert not np.array_equal(s3.perturb_image(shifted, without_removal), expected)


def test_trace_mean_uses_content_rows_only(s3: ModuleType) -> None:
    # A padded variant: the mean must not see the rows the padding replaces.
    image = _noise_crop(10)
    padded = s3.Perturbation(
        "gain_pad",
        remove_trace_mean=True,
        gain=s3.GainMimic.AGC,
        pad=s3.PadMimic.FLAT,
        mapping=s3.GreyMimic.SYMMETRIC,
    )
    changed_bottom = image.copy()
    changed_bottom[-50:] = 0
    assert np.array_equal(
        s3.perturb_image(image, padded, 50),
        s3.perturb_image(changed_bottom, padded, 50),
    )


def test_pad_height_depends_on_seed_and_name_only(s3: ModuleType) -> None:
    heights = np.arange(0, 231)
    names = [f"crop_{i}" for i in range(40)]
    forward = [s3.draw_pad_height(n, heights) for n in names]
    backward = [s3.draw_pad_height(n, heights) for n in reversed(names)]
    assert forward == backward[::-1]
    assert len(set(forward)) > 10
    assert forward != [s3.draw_pad_height(n, heights, seed=1) for n in names]


def test_pad_table_needs_the_documented_tile_count(
    s3: ModuleType, workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = workdir / "heights.csv"
    table.write_text(
        "dataset,line,pad_px,n_tiles\n4tu,a,10,2\ngz_ids,b,30,1\nmala,c,99,5\n",
        encoding="utf-8",
    )
    with pytest.raises(s3.common.InputError):
        s3.load_pad_heights(table)
    monkeypatch.setattr(s3, "EXPECTED_PAD_TILES", 3)
    assert s3.load_pad_heights(table).tolist() == [10, 10, 30]  # MALA left out


def test_real_pad_table_has_the_documented_distribution(s3: ModuleType) -> None:
    heights = s3.load_pad_heights(s3.PAD_HEIGHTS_CSV)
    assert len(heights) == 14749
    # Plan §1-5 table: p50 129, maximum 230, share above 128 50.0 %.
    assert np.percentile(heights, 50) == 129
    assert heights.max() == 230
    assert abs(float(np.mean(heights > 128)) - 0.5) <= 0.0005


def test_ground_truth_with_centre_in_the_padding_is_ignored(s3: ModuleType) -> None:
    boxes = np.array([box[1:] for box in LABEL_BOXES])
    sample = s3.Sample(
        "x", "1.0", _noise_crop(8), boxes, np.array([0, 2, 3]), np.arange(3), 30
    )
    _, kept_classes, kept_index, ignore = s3.split_ground_truth(sample, 226)
    assert kept_classes.tolist() == [0, 2] and kept_index.tolist() == [0, 1]
    assert np.array_equal(ignore, boxes[2:])
    predictions = (boxes.copy(), np.array([0.9, 0.8, 0.7]), np.array([0, 2, 3]))
    _, scores, classes = s3.drop_ignored(*predictions, ignore)
    assert classes.tolist() == [0, 2] and scores.tolist() == [0.9, 0.8]


# ---------------------------------------------------------------- conditions


def test_condition_count_and_names(s3: ModuleType) -> None:
    base = s3.build_conditions()
    assert len(base) == 12 and base[0].is_identity
    full = s3.build_conditions(["r0", "r2", "r5", "r7"])
    assert len(full) == s3.MAX_CONDITIONS == 16
    assert len({c.name for c in full}) == 16
    assert sum(c.invert_polarity for c in full) == 1
    assert [c.recipe_id for c in full[12:]] == ["r0", "r2", "r5", "r7"]


@pytest.mark.parametrize(
    "recipes", [["r0", "r1", "r2", "r5", "r6"], ["r3"], ["r4"], ["r9"], ["r0", "r0"]]
)
def test_bad_stage_a_recipe_lists_are_refused(
    s3: ModuleType, recipes: list[str]
) -> None:
    with pytest.raises(s3.common.InputError):
        s3.build_conditions(recipes)


def test_recipe_mimics_follow_the_plan_table(s3: ModuleType) -> None:
    mimics = s3.RECIPE_MIMICS
    assert mimics["r0"].background == s3.BackgroundMimic.ROW_MEDIAN
    assert mimics["r0"].gain == s3.GainMimic.AGC
    assert mimics["r1"].background == s3.BackgroundMimic.NONE
    assert mimics["r2"].background_alpha == 0.5
    assert mimics["r6"].background_alpha == 0.8
    assert mimics["r6"].agc_window_periods == 32.0
    assert mimics["r6"].pad == s3.PadMimic.FLAT
    assert mimics["r7"].mapping == s3.GreyMimic.HIST
    assert all(m.recipe_id == key for key, m in mimics.items())


def test_condition_sha_changes_with_the_list(s3: ModuleType) -> None:
    base = s3.build_conditions()
    assert s3.conditions_sha256(base) == s3.conditions_sha256(s3.build_conditions())
    assert s3.conditions_sha256(base) != s3.conditions_sha256(base[::-1])


def test_gain_mimic_needs_a_grey_mapping(s3: ModuleType) -> None:
    with pytest.raises(ValueError, match="grey mapping"):
        s3.Perturbation("bad", gain=s3.GainMimic.AGC)


# -------------------------------------------------------------- split checks


def _manifest_row(name: str, session: str, crop: str, subset: str, fold: int) -> dict:
    return {
        "file_name": name,
        "session": session,
        "crop_key": crop,
        "gain": "1.0",
        "subset": subset,
        "fold": str(fold),
    }


def test_split_checks_pass_and_catch_each_leak(s3: ModuleType) -> None:
    manifest = [
        _manifest_row("a_g1.0", "s1", "a", "trainpool", 0),
        _manifest_row("a_g0.5", "s1", "a", "trainpool", 0),
        _manifest_row("b_g1.0", "s2", "b", "trainpool", 1),
        _manifest_row("t_g1.0", "s9", "t", "test", -1),
    ]
    clean = s3.check_fold_split(manifest, ["a_g1.0"], 0)
    assert all(clean["checks"].values())
    leaked_session = [*manifest, _manifest_row("c_g1.0", "s1", "c", "trainpool", 1)]
    assert not s3.check_fold_split(leaked_session, ["a_g1.0"], 0)["checks"][
        "sessions_absent_from_training"
    ]
    split_copy = [*manifest, _manifest_row("a_g0.25", "s1", "a", "trainpool", 1)]
    assert not s3.check_fold_split(split_copy, ["a_g1.0"], 0)["checks"][
        "gain_copies_same_side"
    ]
    with_test = s3.check_fold_split(manifest, ["a_g1.0", "t_g1.0"], 0)
    assert not with_test["checks"]["test_seal_zero_test_crops"]
    assert with_test["counts"]["n_test"] == 1
    assert not s3.check_fold_split(manifest, ["b_g1.0"], 0)["checks"][
        "all_in_this_fold"
    ]
    assert not s3.check_fold_split(manifest, ["zz"], 0)["checks"]["all_in_manifest"]


# ---------------------------------------------------------------------- gate


def _classes(ap50: float, recall: float, n_gt: int = 5, n_ignored: int = 0) -> dict:
    names = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")
    return {
        name: {
            "n_gt": n_gt,
            "n_ignored": n_ignored,
            "ap50": ap50,
            "recall_conf025": recall,
        }
        for name in names
    }


def _gate(s3: ModuleType, identity: dict, reference: dict) -> dict:
    expected = dict.fromkeys(identity, 5)
    validator = dict.fromkeys(identity, 0.5)
    return s3.check_identity_gate(identity, reference, validator, expected)


def test_gate_passes_on_equal_values_despite_validator_difference(
    s3: ModuleType,
) -> None:
    gate = _gate(s3, _classes(0.8, 0.6), _classes(0.8 + 0.5e-4, 0.6))
    assert gate["pass"]
    assert gate["validator_ap50_diff_pp"]["cavity"] == pytest.approx(30.0)


@pytest.mark.parametrize(
    ("identity", "reference", "failed_check"),
    [
        ({"ap50": 0.8 + 2e-4}, {}, "ap50_equals_d1_r4"),
        ({"recall_conf025": 0.61}, {}, "recall_equals_d1_r4"),
        ({"n_gt": 4}, {}, "n_gt"),
        ({"n_ignored": 1}, {}, "n_ignored_zero"),
        ({"ap50": None}, {}, "ap50_equals_d1_r4"),
    ],
)
def test_gate_fails_on_each_check(
    s3: ModuleType, identity: dict, reference: dict, failed_check: str
) -> None:
    identity_classes = _classes(0.8, 0.6)
    identity_classes["pipe_hyperbola"].update(identity)
    gate = _gate(s3, identity_classes, _classes(0.8, 0.6))
    assert not gate["pass"]
    failed = {check["check"] for check in gate["checks"] if not check["pass"]}
    assert failed == {failed_check}


def test_gate_does_not_gate_recall_of_other_classes(s3: ModuleType) -> None:
    identity = _classes(0.8, 0.6)
    identity["pipe_band"]["recall_conf025"] = 0.1
    assert _gate(s3, identity, _classes(0.8, 0.6))["pass"]


# --------------------------------------------------------------- aggregation


def _object(condition: str, gt_index: int, hit: int) -> list:
    # fold, condition, image, gain, class, gt_index, x, y, w, h, hit, score
    return [0, condition, "a", "1.0", "pipe_hyperbola", gt_index, 0, 0, 1, 1, hit, 0]


# identity keeps objects 0 and 1 and hits both; "pad" lost object 1 to the
# padding and misses object 0; "flip" keeps both and misses object 1.
PAIRED_ROWS = [
    _object("identity", 0, 1),
    _object("identity", 1, 1),
    _object("pad", 0, 0),
    _object("flip", 0, 1),
    _object("flip", 1, 0),
]
PAIRED_NAMES = ["identity", "pad", "flip"]


def test_paired_recall_pairs_each_condition_with_the_identity(s3: ModuleType) -> None:
    paired = s3.paired_recall(PAIRED_ROWS, PAIRED_NAMES)
    flip = paired["flip"]["pipe_hyperbola"]
    # The unpadded condition keeps all objects even though "pad" lost one.
    assert flip["n_common"] == 2
    assert flip["recall"] == 0.5 and flip["identity_recall"] == 1.0
    assert flip["delta_vs_identity"] == -0.5
    pad = paired["pad"]["pipe_hyperbola"]
    assert pad["n_common"] == 1 and pad["delta_vs_identity"] == -1.0
    assert paired["identity"]["pipe_hyperbola"]["delta_vs_identity"] == 0.0
    assert paired["identity"]["cavity"]["recall"] is None


def test_paired_recall_all_common_uses_objects_kept_everywhere(
    s3: ModuleType,
) -> None:
    all_common = s3.paired_recall_all_common(PAIRED_ROWS, PAIRED_NAMES)
    flip = all_common["flip"]["pipe_hyperbola"]
    assert flip["n_common"] == 1  # only object 0 survives every condition
    assert flip["recall"] == 1.0 and flip["delta_vs_identity"] == 0.0


def test_n_ignored_is_counted_per_gain_copy(s3: ModuleType) -> None:
    boxes = np.array([box[1:] for box in LABEL_BOXES])
    classes = np.array([box[0] for box in LABEL_BOXES])
    samples = [
        s3.Sample(f"c{i}", gain, _noise_crop(i), boxes, classes, np.arange(3), 60)
        for i, gain in enumerate(("1.0", "1.0", "0.5"))
    ]
    pad_p0 = next(c for c in s3.build_conditions() if c.name == "pad_p0")
    result = s3.evaluate_condition(PerfectDetector(), samples, pad_p0, None, 0, [])
    # The manhole centre (y 235) lies in the 60 px padding of every crop.
    assert result["classes"]["manhole"]["n_ignored"] == 3
    assert result["by_gain"]["1.0"]["classes"]["manhole"]["n_ignored"] == 2
    assert result["by_gain"]["0.5"]["classes"]["manhole"]["n_ignored"] == 1
    assert result["by_gain"]["0.25"]["classes"]["manhole"]["n_ignored"] == 0
    assert result["by_gain"]["1.0"]["classes"]["cavity"]["n_ignored"] == 0


def test_mean_and_sd_use_ddof_one_and_skip_missing(s3: ModuleType) -> None:
    result = s3.mean_and_sd([0.1, 0.3, None])
    assert result["n"] == 2
    assert result["mean"] == pytest.approx(0.2)
    assert result["sd"] == pytest.approx(np.std([0.1, 0.3], ddof=1))
    assert s3.mean_and_sd([None])["mean"] is None


# ------------------------------------------------------------- CLI (blind)


class PerfectDetector:
    """Stand-in detector: always reports the three label boxes."""

    def predict(self, image: np.ndarray) -> DetectorOutput:
        assert image.shape == (CANVAS, CANVAS) and image.dtype == np.uint8
        names = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")
        detections = tuple(
            Detection(cls, names[cls], 0.9 - 0.1 * i, PixelBox(x1, y1, x2, y2))
            for i, (cls, x1, y1, x2, y2) in enumerate(LABEL_BOXES)
        )
        transform = InputTransform(1.0, 1.0, 0.0, 0.0, CANVAS, CANVAS)
        return DetectorOutput(detections=detections, transform=transform)


@dataclass(frozen=True)
class SyntheticRepo:
    run_root: Path
    label_dir: Path
    manifest: Path
    pad_heights: Path
    hist_cdf: Path
    d1_verdict: Path
    out_root: Path
    weights_sha256: tuple[str, ...]

    def d1_reference(self, fold: int) -> Path:
        return self.run_root / f"prefix{fold}" / "eval" / "m31_d1_r4" / "metrics.json"

    def eval_list(self, fold: int) -> Path:
        return (
            self.run_root / f"prefix{fold}" / "eval" / "last_crop" / "eval_images.txt"
        )


def _yolo_line(class_id: int, x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f"{class_id} {(x1 + x2) / 512:.6f} {(y1 + y2) / 512:.6f} "
        f"{(x2 - x1) / 256:.6f} {(y2 - y1) / 256:.6f}"
    )


def _write_fold(root: Path, label_dir: Path, fold: int, rows: list) -> str:
    run_dir = root / "runs" / f"prefix{fold}"
    (run_dir / "weights").mkdir(parents=True)
    weights = run_dir / "weights" / "last.pt"
    weights.write_bytes(f"weights {fold}".encode())
    list_dir = run_dir / "eval" / "last_crop"
    list_dir.mkdir(parents=True)
    paths = []
    for index in range(CROPS_PER_FOLD):
        stem = f"f{fold}c{index}_g1.0"
        image_path = root / "images" / f"{stem}.png"
        image_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(image_path), _noise_crop(10 * fold + index))
        labels = "\n".join(_yolo_line(*box) for box in LABEL_BOXES)
        (label_dir / f"{stem}.txt").write_text(labels + "\n", encoding="utf-8")
        paths.append(str(image_path))
        rows.append(
            _manifest_row(image_path.name, f"s{fold}{index}", stem, "trainpool", fold)
        )
    (list_dir / "eval_images.txt").write_text("\n".join(paths) + "\n", "utf-8")
    per_class = [{"name": n, "ap50": 0.5} for n in _classes(0, 0)]
    (list_dir / "metrics.json").write_text(json.dumps({"per_class": per_class}))
    d1_dir = run_dir / "eval" / "m31_d1_r4"
    d1_dir.mkdir(parents=True)
    _write_d1_reference(d1_dir / "metrics.json", _classes(0.0, 0.0))
    return hashlib.sha256(weights.read_bytes()).hexdigest()


def _write_d1_reference(path: Path, classes: dict, gpu: str = "cpu") -> None:
    reference = {
        "run_identity": {"gpu": gpu},
        "conditions": {"g1.00_dx+0": {"classes": classes}},
    }
    path.write_text(json.dumps(reference), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin_references(
    s3: ModuleType, repo: "SyntheticRepo", monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fix the D1 r4 reference sha constants to the current synthetic files."""
    pinned = tuple(_sha256(repo.d1_reference(fold)) for fold in FOLDS)
    monkeypatch.setattr(s3, "EXPECTED_D1_REFERENCE_SHA256", pinned)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def repo(workdir: Path) -> SyntheticRepo:
    label_dir = workdir / "labels"
    label_dir.mkdir()
    rows: list[dict] = []
    hashes = tuple(_write_fold(workdir, label_dir, fold, rows) for fold in FOLDS)
    rows.append(_manifest_row("t.png", "s99", "t", "test", -1))
    manifest = workdir / "manifest.csv"
    _write_csv(manifest, rows)
    pad_heights = workdir / "pad.csv"
    pad_rows = [
        {"dataset": "4tu", "line": "a", "pad_px": "0", "n_tiles": "2"},
        {"dataset": "gz_ids", "line": "b", "pad_px": "60", "n_tiles": "2"},
    ]
    _write_csv(pad_heights, pad_rows)
    hist_cdf = workdir / "cdf.json"
    cdf = np.linspace(1 / 256, 1.0, 256).tolist()
    hist_cdf.write_text(json.dumps({"cdf": cdf}), encoding="utf-8")
    repo = SyntheticRepo(
        workdir / "runs",
        label_dir,
        manifest,
        pad_heights,
        hist_cdf,
        workdir / "verdict.json",
        workdir / "out",
        hashes,
    )
    records = [
        {"fold": fold, "eval_list_sha256": _sha256(repo.eval_list(fold))}
        for fold in FOLDS
    ]
    repo.d1_verdict.write_text(json.dumps({"gates": {"G2": records}}), "utf-8")
    return repo


@pytest.fixture
def patched(s3: ModuleType, repo: SyntheticRepo, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(s3, "EXPECTED_WEIGHTS_SHA256", repo.weights_sha256)
    _pin_references(s3, repo, monkeypatch)
    monkeypatch.setattr(s3, "EXPECTED_PAD_TILES", 4)
    monkeypatch.setattr(s3, "_load_detector", lambda *_: PerfectDetector())
    n_gt = {"pipe_hyperbola": (3, 3), "pipe_band": (0, 0)}
    n_gt |= {"cavity": (3, 3), "manhole": (3, 3)}
    monkeypatch.setattr(s3.common, "EXPECTED_N_GT", n_gt)
    return s3


def _argv(repo: SyntheticRepo, *extra: str) -> list[str]:
    return [
        "--run-root",
        str(repo.run_root),
        "--run-prefix",
        "prefix",
        "--label-dir",
        str(repo.label_dir),
        "--manifest",
        str(repo.manifest),
        "--pad-heights",
        str(repo.pad_heights),
        "--hist-cdf",
        str(repo.hist_cdf),
        "--d1-verdict",
        str(repo.d1_verdict),
        "--out-root",
        str(repo.out_root),
        "--folds",
        "0,1",
        "--device",
        "cpu",
        "--allow-dirty",
        *extra,
    ]


def _final_line(captured: str) -> dict:
    return json.loads(captured.strip().splitlines()[-1])


def _assert_blind(captured: str) -> None:
    assert not METRIC_WORDS.search(captured), captured


def test_cli_gate_failure_exits_1_and_writes_values_to_files_only(
    patched: ModuleType, repo: SyntheticRepo, capsys: pytest.CaptureFixture
) -> None:
    code = patched.main(_argv(repo))
    captured = capsys.readouterr().out
    assert code == 1
    _assert_blind(captured)
    final = _final_line(captured)
    assert set(final) == FINAL_LINE_KEYS and final["status"] == "gate_failed"
    summary = json.loads((repo.out_root / "summary.json").read_text("utf-8"))
    assert summary["gate"]["0"]["pass"] is False
    assert list(summary["gate"]) == ["0"]  # stopped before fold 1
    fold0 = json.loads((repo.out_root / "fold0" / "metrics.json").read_text("utf-8"))
    assert list(fold0["conditions"]) == ["identity"]


def test_cli_gate_fails_on_a_different_gpu(
    patched: ModuleType,
    repo: SyntheticRepo,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_d1_reference(repo.d1_reference(0), _classes(0.0, 0.0), gpu="RTX")
    _pin_references(patched, repo, monkeypatch)
    assert patched.main(_argv(repo)) == 1
    capsys.readouterr()
    summary = json.loads((repo.out_root / "summary.json").read_text("utf-8"))
    failed = {c["check"] for c in summary["gate"]["0"]["checks"] if not c["pass"]}
    assert "same_gpu_as_d1_r4" in failed


def test_cli_full_run_passes_the_gate_with_matching_references(
    patched: ModuleType,
    repo: SyntheticRepo,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The run stops at the first failing fold, so fix the references fold by
    # fold from the measured identity values (and re-pin their sha256).
    for fold in FOLDS:
        patched.main(_argv(repo, "--overwrite"))
        capsys.readouterr()
        metrics_path = repo.out_root / f"fold{fold}" / "metrics.json"
        measured = json.loads(metrics_path.read_text("utf-8"))
        classes = measured["conditions"]["identity"]["classes"]
        _write_d1_reference(repo.d1_reference(fold), classes)
        _pin_references(patched, repo, monkeypatch)
    code = patched.main(_argv(repo, "--overwrite"))
    captured = capsys.readouterr().out
    assert code == 0
    _assert_blind(captured)
    final = _final_line(captured)
    assert final == {**final, "status": "ok", "n_conditions": 12, "n_images": 6}
    summary = json.loads((repo.out_root / "summary.json").read_text("utf-8"))
    assert all(summary["gate"][str(fold)]["pass"] for fold in FOLDS)
    assert len(summary["aggregate"]) == 12
    identity = summary["run_identity"]
    assert set(identity["weights_sha256"]) == {"0", "1"}
    assert identity["conditions_sha256"] and identity["crop_list_sha256"]
    assert len(identity["inputs_sha256"]["label_dir"]) == 64
    disclosure = summary["settings"]["disclosure"]
    assert {"m2_reference_cdf", "units", "paired_recall"} <= set(disclosure)
    fold0 = json.loads((repo.out_root / "fold0" / "metrics.json").read_text("utf-8"))
    assert {"paired_recall", "paired_recall_all_common"} <= set(fold0)
    with (repo.out_root / "fold0" / "objects.csv").open(encoding="utf-8") as handle:
        objects = list(csv.DictReader(handle))
    assert tuple(objects[0]) == patched.OBJECTS_HEADER
    assert {row["condition"] for row in objects} == {
        c.name for c in patched.build_conditions()
    }


def test_cli_refuses_wrong_weights_and_existing_output(
    patched: ModuleType,
    repo: SyntheticRepo,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(patched, "EXPECTED_WEIGHTS_SHA256", ("0" * 64, "1" * 64))
    assert patched.main(_argv(repo)) == 2
    assert _final_line(capsys.readouterr().out)["status"] == "input_error"
    repo.out_root.mkdir()
    assert patched.main(_argv(repo)) == 2  # output exists, no --overwrite


@pytest.mark.parametrize("changed", ["d1_reference", "eval_list"])
def test_cli_refuses_a_changed_gate_reference_or_crop_list(
    patched: ModuleType,
    repo: SyntheticRepo,
    capsys: pytest.CaptureFixture,
    changed: str,
) -> None:
    # [CVR] A-1: both are fixed by sha256 and checked before any inference.
    path = repo.d1_reference(1) if changed == "d1_reference" else repo.eval_list(1)
    path.write_text(path.read_text("utf-8") + "\n", encoding="utf-8")
    assert patched.main(_argv(repo)) == 2
    final = _final_line(capsys.readouterr().out)
    assert final["status"] == "input_error" and final["n_images"] is None
    assert not repo.out_root.exists()  # refused before fold 0 ran


def test_cli_smoke_skips_the_gate(
    patched: ModuleType, repo: SyntheticRepo, capsys: pytest.CaptureFixture
) -> None:
    argv = _argv(repo, "--smoke", "--limit", "2")
    assert patched.main(argv) == 0
    captured = capsys.readouterr().out
    _assert_blind(captured)
    summary = json.loads((repo.out_root / "summary.json").read_text("utf-8"))
    assert summary["settings"]["gated"] is False
    assert summary["gate"] == {"0": None}
    assert _final_line(captured)["n_images"] == 2
