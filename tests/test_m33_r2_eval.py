"""M3-3 R2 4TU report (``m32_recipe_eval.py --lines``, ``m33_r2_eval``).

The synthetic line of ``tests/test_inject.py`` stands in for a K line; a
stand-in detector replaces YOLO (one whole-tile ``pipe_hyperbola`` box and a
``pipe_band`` box low in the tile, which PP removes or clips). No model and
no stored result are read; the line-list builder test reads the tracked
lists and the K bundle when ``data/`` is present.
"""

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "detection"))
import m32_recipe_eval as runner
import m32_recipe_inputs as inputs
import m33_inputs
import m33_r2_eval as r2
import m33_r2_lines as r2_lines
import m33_r3_band as band

from gpr_detection.detect.base import (
    Detection,
    DetectorOutput,
    InputTransform,
    PixelBox,
)
from gpr_detection.pipeline import (
    PP_SETS,
    ContentRows,
    LineBox,
    PPSetId,
    apply_pp,
)
from gpr_detection.preprocess.to_image import ImageGridMap

CLASS_NAMES = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")
METRIC_WORDS = re.compile(r"recall|hit|m4|agree|ap50|inclusion", re.IGNORECASE)


@pytest.fixture(scope="module")
def helpers() -> ModuleType:
    path = Path(__file__).resolve().parent / "test_inject.py"
    spec = importlib.util.spec_from_file_location("inject_r2_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDetector:
    """A whole-tile ``pipe_hyperbola`` box and a ``pipe_band`` box low in the tile."""

    class_names = CLASS_NAMES

    def predict(self, image: np.ndarray) -> DetectorOutput:
        size = image.shape[0]
        detections = (
            Detection(0, "pipe_hyperbola", 0.6, PixelBox(0, 0, size, size)),
            Detection(1, "pipe_band", 0.3, PixelBox(10, size - 40, 60, size - 10)),
        )
        return DetectorOutput(detections, InputTransform(1, 1, 0, 0, size, size))


def _lines_csv(path: Path, line_id: str, keep_one: bool = True) -> Path:
    activity_id, line_path = line_id.split("/", 1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, r2_lines.COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "line_id": line_id,
                "activity_id": activity_id,
                "path": line_path,
                "site": "1",
                "dataset": "4tu",
                "keep_one": keep_one,
                "role": "k",
                "n_traces": 235,
            }
        )
    return path


@pytest.fixture
def offline_r2(monkeypatch, helpers, tmp_path_factory):
    radargram = helpers._radargram(helpers._background_data())
    folder = tmp_path_factory.mktemp("r2in")
    lines = _lines_csv(folder / "lines.csv", helpers.LINE_ID)
    c0b = folder / "d0_gate.json"
    selection = {"pp_set_id": "S3", "pp_config_sha256": PP_SETS[PPSetId.S3].sha256()}
    c0b.write_text(json.dumps({"selection": selection}), encoding="utf-8")
    monkeypatch.setattr(m33_inputs, "C0B_D0_GATE_JSON", c0b)
    monkeypatch.setattr(m33_inputs, "C0B_D0_GATE_SHA256", inputs.sha256_file(c0b))
    monkeypatch.setattr(m33_inputs, "R2_LINES_SHA256_LF", inputs.sha256_file_lf(lines))
    monkeypatch.setattr(m33_inputs, "assert_r0_identity", lambda: {"stub": "r0"})
    monkeypatch.setattr(inputs, "assert_list_hashes", lambda: {})
    monkeypatch.setattr(
        inputs,
        "known_apexes",
        lambda: [inputs.KnownApex("G114", helpers.LINE_ID, 100, 200.0)],
    )
    monkeypatch.setattr(runner, "assert_weights", lambda *_: {"0": "w"})
    monkeypatch.setattr(runner, "load_radargram", lambda ref: radargram)
    monkeypatch.setattr(runner, "load_detectors", lambda settings: {0: FakeDetector()})
    return lines


def _argv(lines: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--stage",
        "uninjected",
        "--lines",
        str(lines),
        "--folds",
        "0",
        "--device",
        "cpu",
        "--allow-dirty",
        "--out-root",
        str(out),
        *extra,
    ]


def _final(capsys) -> tuple[dict, list[str]]:
    lines = capsys.readouterr().out.strip().splitlines()
    return json.loads(lines[-1]), lines


def test_r2_writes_the_r3_box_format_j1_files_and_summary(
    offline_r2, tmp_path, capsys
) -> None:
    code = runner.main(_argv(offline_r2, tmp_path))

    final, lines = _final(capsys)
    assert (code, final["status"]) == (1, "not_judgeable")
    assert not any(METRIC_WORDS.search(text) for text in lines)
    fold = tmp_path / "fold0"
    with (fold / "boxes.csv").open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert tuple(header[: len(band.BOX_COLUMNS)]) == band.BOX_COLUMNS
    boxes = inputs.read_csv(fold / "boxes.csv")
    assert {row["pp"] for row in boxes} == {"pre", "post"}
    assert {row["orientation"] for row in boxes} == {"original", "flipped"}
    removed = [
        r for r in boxes if r["pp"] == "pre" and r["pp_action"].startswith("removed")
    ]
    for row in removed:  # no post row for a removed box
        assert not any(
            r["pp"] == "post"
            and r["box_index"] == row["box_index"]
            and r["orientation"] == row["orientation"]
            for r in boxes
        )
    apex = inputs.read_csv(fold / "apex_hits.csv")
    assert {row["pp"] for row in apex} == {"pre", "post"}
    assert apex[0]["snr_quartile"] != ""  # G114 is in the X6 table
    for name in ("j1_boxes_pre", "j1_boxes_post", "j1_band_boxes_pre"):
        rows = inputs.read_csv(fold / f"{name}.csv")
        assert all(float(row["score"]) > 0.25 for row in rows)
        assert all(
            row["activity_id"] + "/" + row["path"] == row["line_id"] for row in rows
        )
    post = inputs.read_csv(fold / "j1_boxes_post.csv")
    assert not any(row["pp_action"].startswith("removed") for row in post)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["judgment"]["blockers"]
    j1 = summary["j1_inputs"]
    assert j1["pp_set_id"] == "S3" and j1["n_lines"] == 1
    assert j1["folds"]["0"]["post_sha256"] == inputs.sha256_file(
        fold / "j1_boxes_post.csv"
    )
    assert j1["folds"]["0"]["n_boxes_post"] == len(post)
    assert j1["folds"]["0"]["n_padding_only_dropped"] >= 0
    for row in summary["tables"]["keep_one_720"]["pre"]["missed_apex_widths"]:
        assert 1 <= row["n_folds_missed"] <= 1  # one fold in this run
    checks = summary["run_identity"]["r_checks"]
    assert checks["settings"]["settings_sha256"] == m33_inputs.R_SETTINGS_SHA256
    assert checks["inject_blob"]["blob"] == m33_inputs.C1_INJECT_BLOB
    assert summary["pp_actions"]["by_fold"]["0"] == summary["pp_actions"]["total"]
    assert "0" in summary["pp_a_top_in_content"]["by_fold"]
    main = summary["tables"]["keep_one_720"]
    assert main["n_lines"] == 1 and "k_inclusion" in main["pre"]
    assert main["pre"]["k_inclusion"]["n_apexes"] == 1
    assert {r2.PHASE_INFER, r2.PHASE_SUMMARY} <= set(summary["timings"])
    line = inputs.read_csv(tmp_path / "lines.csv")[0]
    assert {"two_event", "y_bot_px", "zero_px", "keep_one"} <= set(line)


def test_r3_reads_the_r2_boxes(offline_r2, tmp_path, capsys) -> None:
    runner.main(_argv(offline_r2, tmp_path))
    capsys.readouterr()
    rows = [
        row
        for row in band._box_rows_of(tmp_path)
        if row["orientation"] == runner.ORIGINAL
    ]
    zone = band.zone_boxes(rows)
    pairs = band.pp_boxes_from_rows(rows, "line_id")
    line_id = inputs.read_csv(offline_r2)[0]["line_id"]
    assert zone[line_id] and pairs[(line_id, 0)]
    plan = ["--injectant", "band", "--stage", "plan", "--out-root", str(tmp_path / "x")]
    partial = runner.parse_args([*plan, "--only-lines", line_id])
    assert band.require_upstream(tmp_path / "summary.json", partial)
    with pytest.raises(inputs.InputError, match="not judgeable"):
        band.require_upstream(tmp_path / "summary.json", runner.parse_args(plan))


def test_r2_smoke_is_lines_plus_smoke_without_a_stage(
    offline_r2, tmp_path, capsys
) -> None:
    """Smoke 2026-11-16: ``--stage uninjected --smoke`` is refused; the R2
    smoke is ``--lines <csv> --smoke`` (fold 0, r0, its own folder)."""
    with pytest.raises(SystemExit):
        runner.parse_args(["--stage", "uninjected", "--smoke", "--lines", "x.csv"])
    settings = runner.parse_args(["--smoke", "--lines", str(offline_r2)])
    assert settings.is_r2 and settings.smoke and settings.folds == (0,)
    assert settings.out_root == m33_inputs.R2_OUT_ROOT_SMOKE
    line_id = inputs.read_csv(offline_r2)[0]["line_id"]
    argv = ["--smoke", "--lines", str(offline_r2), "--device", "cpu"]
    argv += ["--only-lines", line_id, "--out-root", str(tmp_path)]
    code = runner.main(argv)
    final, _ = _final(capsys)
    assert (code, final["status"]) == (1, "not_judgeable")
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "smoke" in summary["judgment"]["blockers"]
    assert (tmp_path / "fold0" / "boxes.csv").is_file()


def test_r2_line_list_sha_is_asserted(offline_r2, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(m33_inputs, "R2_LINES_SHA256_LF", "0" * 64)
    assert runner.main(_argv(offline_r2, tmp_path)) == 2
    assert _final(capsys)[0]["status"] == "input_error"


def test_r2_refuses_another_inject_blob(offline_r2, monkeypatch, tmp_path, capsys):
    """verif S4 (user 2026-11-16): R2 holds ``inject.py`` in its code blobs."""
    monkeypatch.setattr(m33_inputs, "C1_INJECT_BLOB", "0" * 40)
    assert runner.main(_argv(offline_r2, tmp_path)) == 2
    assert _final(capsys)[0]["status"] == "input_error"
    blobs = {m33_inputs.INJECT_PATH: "1" * 40}
    with pytest.raises(inputs.InputError, match=r"inject\.py blob"):
        m33_inputs.assert_inject_blob(blobs)
    with pytest.raises(inputs.InputError, match=r"inject\.py blob"):
        m33_inputs.assert_inject_blob({})


def test_no_flip_blocks_a_full_r2_run_only(tmp_path) -> None:
    full = ["--stage", "uninjected", "--out-root", str(tmp_path)]
    r2_run = runner.parse_args([*full, "--lines", "x.csv", "--no-flip"])
    assert r2_run.scope_blockers() == [m33_inputs.BLOCKER_NO_FLIP]
    assert runner.parse_args([*full, "--lines", "x.csv"]).scope_blockers() == []
    m32_run = runner.parse_args([*full, "--no-flip"])
    assert m33_inputs.BLOCKER_NO_FLIP not in m32_run.scope_blockers()


def test_r2_settings_run_r0_and_need_the_uninjected_stage(tmp_path) -> None:
    settings = runner.parse_args(
        ["--stage", "uninjected", "--lines", str(tmp_path / "l.csv")]
    )
    assert settings.is_r2 and settings.recipes == ("r0",)
    assert settings.expected_recipes == ("r0",)
    assert settings.out_root == m33_inputs.R2_OUT_ROOT
    assert settings.stage_dir == settings.out_root
    full = runner.parse_args(
        ["--stage", "uninjected", "--lines", "x.csv", "--out-root", str(tmp_path)]
    )
    assert full.scope_blockers() == []
    only = runner.parse_args(
        ["--stage", "uninjected", "--lines", "x.csv", "--only-lines", "a/b,c/d"]
    )
    assert only.only_lines == ("a/b", "c/d") and "line_limit" in only.scope_blockers()
    with pytest.raises(SystemExit):
        runner.parse_args(["--stage", "plan", "--lines", "x.csv"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--stage", "uninjected", "--lines", "x", "--recipes", "r5"])


def test_line_ids_split_at_the_first_slash(tmp_path) -> None:
    assert r2.split_line_id("011.17/Profile_6409") == ("011.17", "Profile_6409")
    good = _lines_csv(tmp_path / "good.csv", "01.1/Path5")
    (line,) = r2.read_r2_lines(good)
    assert (line.activity_id, line.path, line.keep_one) == ("01.1", "Path5", True)
    bad = tmp_path / "bad.csv"
    bad.write_text(
        good.read_text(encoding="utf-8").replace("Path5,1,", "Path6,1,", 1),
        encoding="utf-8",
    )
    with pytest.raises(inputs.InputError):
        r2.read_r2_lines(bad)
    with pytest.raises(inputs.InputError):
        r2.check_full_list([line])


def test_j1_post_drops_removed_boxes_and_keeps_clipped_bottoms() -> None:
    grid = ImageGridMap(0, 0, 1.0, 1.0, 200, 120)
    image = SimpleNamespace(grid_map=grid)
    content = ContentRows(y_bot_px=100.0, period_px=6.0, zero_px=None)
    boxes = [
        LineBox(0, "pipe_hyperbola", 0.9, PixelBox(10, 90, 30, 120), 0),  # removed_a
        LineBox(0, "pipe_hyperbola", 0.8, PixelBox(40, 50, 60, 110), 0),  # clipped_b
        LineBox(0, "pipe_hyperbola", 0.2, PixelBox(70, 20, 90, 40), 0),  # under 0.25
        LineBox(1, "pipe_band", 0.5, PixelBox(0, 30, 200, 40), 0),
        LineBox(0, "pipe_hyperbola", 0.9, PixelBox(10, 125, 30, 140), 0),  # padding
    ]
    outcomes = apply_pp(boxes, content, PP_SETS[PPSetId.S3])
    line = r2.R2Line(
        inputs.LineRef("4tu", "01.1/Path1", "k"), True, "01.1", "Path1", "1", 200
    )
    geometry = r2.LineGeometry(image, content, None)  # type: ignore[arg-type]

    files, dropped = r2.j1_rows(line, 0, geometry, outcomes)

    assert dropped == {"j1_boxes_pre": 1}  # the padding box has no L0 box
    record = r2.padding_dropped_by_fold([{"j1_padding_dropped": {"0": dropped}}])
    assert record == {"0": {"j1_boxes_pre": 1}}
    assert [row["pp_action"] for row in files["j1_boxes_pre"]] == [
        "removed_a",
        "clipped_b",
    ]
    (clipped,) = files["j1_boxes_post"]
    assert clipped["pp_action"] == "clipped_b" and clipped["y2_px"] == 100.0
    assert clipped["k1"] == 99  # outward L0 rounding of the clipped bottom
    assert len(files["j1_band_boxes_pre"]) == len(files["j1_band_boxes_post"]) == 1


def test_missed_apex_widths_count_the_missing_folds() -> None:
    """cv decision 2026-11-16: every K apex missed by at least one fold."""

    def row(blind_id: str, fold: int, hit: bool) -> dict:
        widths = {column: f"{blind_id}-{column}" for column in r2.SNR_COLUMNS}
        return {"blind_id": blind_id, "line_id": "01.1/Path3", "fold": fold,
                "pp": "pre", "hit": str(hit)} | widths  # fmt: skip

    rows = [
        row("G1", 0, True), row("G1", 1, False), row("G1", 2, False),
        row("G2", 0, True), row("G2", 1, True), row("G2", 2, True),
        row("G3", 0, False), row("G3", 1, False), row("G3", 2, False),
    ]  # fmt: skip
    rows.append(row("G2", 0, False) | {"pp": "post"})

    missed = r2.missed_apex_widths(rows, "pre")

    assert [(m["blind_id"], m["n_folds_missed"]) for m in missed] == [
        ("G1", 2),
        ("G3", 3),
    ]
    assert missed[0]["snr_db"] == "G1-snr_db"
    assert r2.missed_apex_widths(rows, "post")[0]["blind_id"] == "G2"


# ------------------------------------------------------- C1 shared checks


def test_settings_hash_is_the_c1_constant_and_guards_the_values() -> None:
    document = m33_inputs.settings_document(0.25, 0.001, 0.7)
    assert m33_inputs.settings_sha256(document) == m33_inputs.R_SETTINGS_SHA256
    checked = m33_inputs.assert_settings(0.25, 0.001, 0.7)
    assert checked["settings_sha256"] == m33_inputs.R_SETTINGS_SHA256
    with pytest.raises(inputs.InputError, match="settings sha256"):
        m33_inputs.assert_settings(0.3, 0.001, 0.7)
    with pytest.raises(inputs.InputError, match="settings sha256"):
        m33_inputs.assert_settings(0.25, 0.001, 0.6)


def test_r0_identity_holds_the_eleven_spec9_lines_and_refuses_a_change(
    helpers, monkeypatch
) -> None:
    assert len(m33_inputs.R0_IDENTITY_LINES) == 11
    radargram = helpers._radargram(helpers._background_data())
    digest = m33_inputs.line_image_sha256(radargram)
    monkeypatch.setattr(
        m33_inputs, "R0_IDENTITY_LINES", (("fourtu", "01.1/Path1", digest),)
    )

    measured = m33_inputs.assert_r0_identity(lambda *_: radargram)

    assert measured == {"fourtu/01.1/Path1": digest}
    monkeypatch.setattr(
        m33_inputs, "R0_IDENTITY_LINES", (("fourtu", "01.1/Path1", "0" * 64),)
    )
    with pytest.raises(inputs.InputError, match="changed"):
        m33_inputs.assert_r0_identity(lambda *_: radargram)

    def unreadable(*_):
        raise OSError("missing")

    with pytest.raises(inputs.InputError, match="r0 identity line"):
        m33_inputs.assert_r0_identity(unreadable)


@pytest.mark.skipif(
    not inputs.K_LINES.is_file(), reason="K bundle (data/processing/d3) not present"
)
def test_line_list_builder_counts_754_and_720(tmp_path) -> None:
    result = r2_lines.build(tmp_path / "lines.csv")
    assert (result["n_lines"], result["n_keep_one"], result["n_k"]) == (754, 720, 27)
    assert result["sha256_lf"] == m33_inputs.R2_LINES_SHA256_LF
    tracked = m33_inputs.R2_LINES_CSV.read_bytes()
    assert (tmp_path / "lines.csv").read_bytes() == tracked
