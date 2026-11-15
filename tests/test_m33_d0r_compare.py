"""D0' ↔ C0b reproducibility check (``analysis/detection/m33_d0r_compare.py``).

Synthetic ``d0_gate.json`` documents exercise every tolerance edge, missing
keys, nulls, selection changes and cell-count errors; the real C0b file is
only read, for a self-comparison.
"""

import copy
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "analysis" / "detection" / "m33_d0r_compare.py"
C0B_JSON = PROJECT_ROOT / "runs" / "eval" / "m33_d0" / "d0_gate.json"
D0R_JSON = PROJECT_ROOT / "runs" / "eval" / "m33_d0r" / "d0_gate.json"
METRIC_WORDS = re.compile(r"loss|recall|lost|h_s0|increment|cumulative", re.I)
ALL_SETS = ("S1", "S2", "S3", "S4", "S5")
EVALUATED = ["S1", "S2", "S3", "S4"]


@pytest.fixture(scope="module")
def cmp():
    spec = importlib.util.spec_from_file_location("m33_d0r_compare_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _loss_cells() -> dict:
    cell = {"lost": 2.0, "gained": 0.0, "total": 100.0, "loss_pp": 0.7, "gain_pp": 0.0}
    return {
        condition: {str(fold): dict(cell) for fold in range(5)}
        for condition in ("A0", "A1", "A2", "A3")
    }


def _gate() -> dict:
    return {
        "G1": {"cells": _loss_cells()},
        "G2": {"cells": _loss_cells()},
        "G3": {"loss": 0, "h_s0": 10, "h_sj": 10, "exposed_boxes_s0": 3},
        "G4": {
            "increment": 40,
            "cumulative": 200,
            "by_rule": {"removed_a": 150, "clipped_b": 50},
            "by_condition": {
                "A1": {"removed_a": 100},
                "A3": {"removed_d": 4},
                "Y0": {},
            },
        },
        "checks": {"G1": True, "G2": True, "G3": True, "G4": True},
        "pass": True,
    }


Y1_CUTS = ("Y1_0.5P", "Y1_1P", "Y1_2P", "Y1_3P")


def _report() -> dict:
    """Report-only Y1 values: apex hits per cut and set, rule counts per cut."""
    hits = {cut: {f"S{k}": 9 for k in range(6)} for cut in Y1_CUTS}
    rules = {
        set_id: {"by_condition": {cut: {"removed_a": 3} for cut in Y1_CUTS}}
        for set_id in ALL_SETS
    }
    return {"synthetic_apex_hits": hits, "rule_counts_outside_gate": rules}


def _document() -> dict:
    """A C0b-shaped document: S1-S4 evaluated, S3 selected, stop at S4.

    It carries a judgeable ``judgment`` block, as a revised D0 run does
    (the real C0b file has none; the base never needs one).
    """
    gates = {set_id: _gate() for set_id in ALL_SETS}
    gates["S4"]["checks"]["G1"] = False
    gates["S4"]["pass"] = False
    return {
        "selection": {
            "pp_set_id": "S3",
            "evaluated": list(EVALUATED),
            "stopped_at": "S4",
            "stop_reason": "gate_failed",
        },
        "gates": gates,
        "report": _report(),
        "judgment": {"judgeable": True, "blockers": []},
    }


def _changed(edit) -> dict:
    document = _document()
    edit(document)
    return document


# ---------------------------------------------------------- self match


def test_a_document_matches_itself_with_the_expected_counts(cmp):
    base = _document()
    result = cmp.compare(base, copy.deepcopy(base))
    assert result.status == "match"
    assert result.outside == [] and result.differing_within == []
    expected = cmp.expected_counts(len(EVALUATED))
    assert {k: result.counts[k] for k in expected} == expected
    assert expected["G1"] == expected["G2"] == 4 * 3 * 5
    assert result.counts["G4_rules"] == 4 * (2 + 2)  # by_rule 2, by_condition 2


# ------------------------------------------------------ tolerance edges


def _status(cmp, edit) -> tuple[str, list]:
    result = cmp.compare(_document(), _changed(edit))
    return result.status, [cell.key for cell in result.differing_within]


def _set(path: tuple, value):
    def edit(document):
        node = document
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return edit


G1_A0 = ("gates", "S2", "G1", "cells", "A0", "3", "lost")
G2_A3 = ("gates", "S3", "G2", "cells", "A3", "1", "loss_pp")


@pytest.mark.parametrize(
    ("path", "value", "status"),
    [
        (G1_A0, 3.0, "match"),  # +1 GT
        (G1_A0, 1.0, "match"),  # -1 GT
        (G1_A0, 4.0, "mismatch"),  # +2 GT
        (G2_A3, 0.9, "match"),  # +0.2 %p (0.9 - 0.7 = 0.20000000000000007)
        (G2_A3, 0.5, "match"),  # -0.2 %p
        (G2_A3, 0.91, "mismatch"),
        (("gates", "S1", "G3", "h_s0"), 11, "match"),
        (("gates", "S1", "G3", "h_sj"), 8, "mismatch"),
        (("gates", "S4", "G3", "exposed_boxes_s0"), 4, "match"),
        (("gates", "S2", "G4", "cumulative"), 210, "match"),  # 5 % of 200
        (("gates", "S2", "G4", "cumulative"), 211, "mismatch"),
        (("gates", "S2", "G4", "increment"), 45, "match"),  # floor of 5
        (("gates", "S2", "G4", "increment"), 46, "mismatch"),
        (("gates", "S3", "G4", "by_rule", "removed_a"), 157, "match"),  # 7.5
        (("gates", "S3", "G4", "by_rule", "removed_a"), 158, "mismatch"),
        (("gates", "S3", "G4", "by_condition", "Y0", "removed_a"), 5, "match"),
        (("gates", "S3", "G4", "by_condition", "Y0", "removed_a"), 6, "mismatch"),
    ],
)
def test_tolerance_edges(cmp, path, value, status):
    found, within = _status(cmp, _set(path, value))
    assert found == status
    if status == "match":
        assert len(within) == 1  # a difference inside the band is listed


def test_a_count_key_missing_on_one_side_counts_zero(cmp):
    def drop(document):
        del document["gates"]["S1"]["G4"]["by_condition"]["A3"]["removed_d"]

    assert _status(cmp, drop) == ("match", ["S1/G4/by_condition/A3/removed_d"])

    def drop_big(document):
        del document["gates"]["S1"]["G4"]["by_rule"]["removed_a"]

    assert _status(cmp, drop_big)[0] == "mismatch"


def test_null_equals_only_null(cmp):
    both_null = _set(G2_A3, None)
    base = _changed(both_null)
    assert cmp.compare(base, copy.deepcopy(base)).status == "match"
    assert cmp.compare(base, _changed(_set(G2_A3, 0.7))).status == "mismatch"
    assert cmp.compare(_document(), base).status == "mismatch"


# --------------------------------------------------- exact (selection)


@pytest.mark.parametrize(
    "edit",
    [
        _set(("selection", "pp_set_id"), "S2"),
        _set(("selection", "stop_reason"), None),
        _set(("selection", "evaluated"), ["S1", "S2", "S3"]),
        _set(("gates", "S2", "checks", "G4"), False),
        _set(("gates", "S4", "pass"), True),
    ],
)
def test_selection_and_checks_must_be_equal(cmp, edit):
    assert _status(cmp, edit)[0] == "mismatch"


def test_record_only_cells_and_s5_never_change_the_result(cmp):
    def edit(document):
        document["gates"]["S2"]["G1"]["cells"]["A2"]["0"]["lost"] = 50.0
        document["gates"]["S5"]["G4"]["cumulative"] = 10_000
        document["gates"]["S5"]["checks"]["G2"] = False

    result = cmp.compare(_document(), _changed(edit))
    assert result.status == "match"
    record = {cell.key: cell for cell in result.cells if cell.record_only}
    assert not record["S2/G1/A2/fold0/lost"].within  # recorded, not counted
    assert all(not key.startswith("S5") for key in (c.key for c in result.cells))


# ------------------------------------------------------- input errors


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d["gates"]["S1"]["G1"]["cells"]["A3"].pop("4"),
        lambda d: d["gates"]["S2"]["G2"]["cells"].pop("A1"),
        lambda d: d["gates"]["S3"]["G4"]["by_condition"].pop("Y0"),
        lambda d: d["gates"]["S4"]["G3"].pop("exposed_boxes_s0"),
        lambda d: d["gates"].pop("S2"),
    ],
)
def test_a_missing_cell_is_an_input_error(cmp, edit):
    with pytest.raises(cmp.CompareInputError, match="missing"):
        cmp.compare(_document(), _changed(edit))
    with pytest.raises(cmp.CompareInputError, match="missing"):
        cmp.compare(_changed(edit), _document())


def test_cell_counts_off_the_expected_are_a_mismatch(cmp, monkeypatch):
    """Tool update 2026-11-15: an off count is ``mismatch``, not an input error."""
    cells = cmp.build_cells(_document(), _document())
    _, off = cmp.check_counts(cells, n_sets=5)
    assert off["G1"] == (60, 75)
    without_one = [cell for cell in cells if cell.key != "S2/G1/A0/fold3/lost"]
    assert cmp.check_counts(without_one, n_sets=4)[1] == {"G1": (59, 60)}
    assert cmp.check_counts(cells, n_sets=4)[1] == {}

    wanted = cmp.expected_counts(len(EVALUATED))
    monkeypatch.setattr(
        cmp, "expected_counts", lambda n_sets: dict(wanted, G3=wanted["G3"] + 1)
    )
    result = cmp.compare(_document(), _document())
    assert result.status == "mismatch" and result.outside == []
    assert result.count_problems == {"G3": (12, 13)}


def test_an_empty_comparison_is_an_input_error(cmp):
    empty = _changed(_set(("selection", "evaluated"), []))
    with pytest.raises(cmp.CompareInputError, match="empty"):
        cmp.compare(empty, empty)
    record_only = [
        c for c in cmp.build_cells(_document(), _document()) if c.record_only
    ]
    with pytest.raises(cmp.CompareInputError, match="no compared cells"):
        cmp.check_counts(record_only, n_sets=4)


def test_a_not_judgeable_new_run_is_not_compared(cmp):
    new = _changed(_set(("judgment",), {"judgeable": False, "blockers": ["smoke"]}))
    with pytest.raises(cmp.CompareInputError, match="not judgeable"):
        cmp.compare(_document(), new)
    assert cmp.compare(_document(), _document()).judgment == "judgeable"


def test_a_new_run_without_a_judgment_block_is_refused(cmp):
    """Tool update 2026-11-15: an old-format json (e.g. C0b itself) is refused."""
    old_format = _document()
    del old_format["judgment"]
    with pytest.raises(cmp.CompareInputError, match="no judgment block"):
        cmp.compare(_document(), old_format)
    assert cmp.compare(old_format, _document()).status == "match"  # base may lack it


# ------------------------------------------------------- Y1 record only


def test_y1_cells_are_recorded_and_never_change_the_result(cmp):
    def edit(document):
        document["report"]["synthetic_apex_hits"]["Y1_1P"]["S3"] = 2
        by_condition = document["report"]["rule_counts_outside_gate"]["S2"]
        by_condition["by_condition"]["Y1_3P"] = {"removed_a": 300, "clipped_b": 1}

    result = cmp.compare(_document(), _changed(edit))
    assert result.status == "match"
    y1 = {c.key: c for c in result.cells if c.group == "record_Y1"}
    assert all(cell.record_only for cell in y1.values())
    assert (y1["S3/Y1/Y1_1P/apex_hits"].base, y1["S3/Y1/Y1_1P/apex_hits"].new) == (9, 2)
    assert not y1["S2/Y1/Y1_3P/by_rule/removed_a"].within
    assert y1["S2/Y1/Y1_3P/by_rule/clipped_b"].base == 0  # union of keys
    hit_keys = [key for key in y1 if key.endswith("apex_hits")]
    assert len(hit_keys) == len(Y1_CUTS) * (1 + len(EVALUATED))  # S0 + evaluated
    assert "S0/Y1/Y1_0.5P/apex_hits" in y1


def test_missing_y1_keys_are_recorded_as_null(cmp):
    new = _document()
    del new["report"]
    result = cmp.compare(_document(), new)
    assert result.status == "match"
    hits = [c for c in result.cells if c.key.endswith("apex_hits")]
    assert hits and all(cell.new is None for cell in hits)


# ----------------------------------------------------------------- CLI


def _write(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _main(cmp, capsys, *argv) -> tuple[int, dict]:
    exit_code = cmp.main([str(arg) for arg in argv])
    final = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    return exit_code, final


def _sealed_base(cmp, monkeypatch, path: Path, document: dict) -> Path:
    """Write a stand-in C0b file and make its sha the sealed one."""
    _write(path, document)
    monkeypatch.setattr(cmp, "C0B_D0_GATE_SHA256", cmp.sha256_file(path))
    return path


def test_cli_writes_values_to_files_only(cmp, capsys, tmp_path, monkeypatch):
    base = _sealed_base(
        cmp, monkeypatch, tmp_path / "c0b" / "d0_gate.json", _document()
    )
    new = _write(tmp_path / "d0r" / "d0_gate.json", _changed(_set(G1_A0, 3.0)))
    out = tmp_path / "report" / "compare.json"
    exit_code, final = _main(cmp, capsys, "--base", base, "--new", new, "--out", out)
    assert (exit_code, final["status"]) == (0, "match")
    assert set(final) == {"status", "n_cells", "n_record_only", "out"}
    assert not METRIC_WORDS.search(json.dumps(final))
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [row["key"] for row in report["within_tolerance_differences"]] == [
        "S2/G1/A0/fold3/lost"
    ]
    assert report["outside_tolerance"] == []
    assert report["inputs"]["base"]["sha256"] != report["inputs"]["new"]["sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", report["script"]["sha256"])
    assert out.with_suffix(".md").read_text(encoding="utf-8").startswith("# D0'")
    assert final["n_cells"] == len(report["cells"])


def test_cli_exit_codes(cmp, capsys, tmp_path, monkeypatch):
    base = _sealed_base(
        cmp, monkeypatch, tmp_path / "c0b" / "d0_gate.json", _document()
    )
    bad = _write(tmp_path / "d0r" / "d0_gate.json", _changed(_set(G1_A0, 9.0)))
    out = tmp_path / "report" / "compare.json"
    assert _main(cmp, capsys, "--base", base, "--new", bad, "--out", out)[0] == 1
    missing = tmp_path / "none.json"
    code, final = _main(cmp, capsys, "--base", base, "--new", missing, "--out", out)
    assert (code, final["status"]) == (2, "input_error")
    inside = tmp_path / "c0b" / "compare.json"  # never write next to C0b
    code, _ = _main(cmp, capsys, "--base", base, "--new", bad, "--out", inside)
    assert code == 2 and not inside.exists()


def test_a_base_other_than_the_c0b_record_is_refused_first(
    cmp, capsys, tmp_path, monkeypatch
):
    """Tool update 2026-11-15: the base sha is asserted before anything else."""
    base = _sealed_base(
        cmp, monkeypatch, tmp_path / "c0b" / "d0_gate.json", _document()
    )
    other = _write(tmp_path / "other" / "d0_gate.json", _changed(_set(G1_A0, 3.0)))
    new = _write(tmp_path / "d0r" / "d0_gate.json", _document())
    out = tmp_path / "report" / "compare.json"
    code, final = _main(cmp, capsys, "--base", other, "--new", new, "--out", out)
    assert (code, final["status"]) == (2, "input_error") and not out.exists()
    assert cmp.check_base_sha(base) == cmp.C0B_D0_GATE_SHA256
    with pytest.raises(cmp.CompareInputError, match="not the C0b record"):
        cmp.check_base_sha(other)


def test_boundary_rows_are_recorded_from_the_predictions(cmp, tmp_path):
    fold_dir = tmp_path / "fold1"
    fold_dir.mkdir()
    header = ["fold", "condition", "image", "class", "score", "category"]
    header += [f"pp_action_S{k}" for k in range(1, 6)]
    removed_from_s2 = ["none", "removed_d", "removed_d", "removed_d", "removed_d"]
    untouched = ["none"] * 5
    crop = cmp.BOUNDARY_CROP
    rows = [
        [1, "A3", crop, "pipe_band", "0.2500000149", "d", *removed_from_s2],
        [1, "A0", crop, "pipe_band", "0.2500000149", "d", *untouched],  # other view
        [1, "A3", crop, "pipe_band", "0.9", "", *untouched],  # far from 0.25
    ]
    with (fold_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    (record,) = cmp.boundary_rows(tmp_path)
    assert record["score_above_conf_recall"] and record["category"] == "d"
    assert record["pp_action_S2"] == "removed_d"
    assert cmp.boundary_rows(tmp_path / "missing") is None


# ------------------------------------------------- real C0b, self check


@pytest.mark.skipif(not C0B_JSON.is_file(), reason="C0b d0_gate.json absent")
def test_c0b_is_the_sealed_base_and_not_a_new_run(cmp, capsys, tmp_path):
    """Read only: C0b passes the base sha check but has no judgment block."""
    out = tmp_path / "c0b_self.json"
    before = C0B_JSON.stat().st_mtime_ns
    assert cmp.check_base_sha(C0B_JSON) == cmp.C0B_D0_GATE_SHA256
    code, final = _main(
        cmp, capsys, "--base", C0B_JSON, "--new", C0B_JSON, "--out", out
    )
    assert (code, final["status"]) == (2, "input_error") and not out.exists()
    assert C0B_JSON.stat().st_mtime_ns == before


@pytest.mark.skipif(
    not (C0B_JSON.is_file() and D0R_JSON.is_file()), reason="C0b or D0' json absent"
)
def test_d0r_matches_c0b(cmp, capsys, tmp_path):
    """Read only: the recorded D0' result still matches C0b with zero differences."""
    out = tmp_path / "d0r_vs_c0b.json"
    before = (C0B_JSON.stat().st_mtime_ns, D0R_JSON.stat().st_mtime_ns)
    code, final = _main(
        cmp, capsys, "--base", C0B_JSON, "--new", D0R_JSON, "--out", out
    )
    assert (code, final["status"]) == (0, "match")
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["outside_tolerance"] == []
    assert report["cell_count_problems"] == {}
    expected = report["expected_counts"]
    assert {k: report["cell_counts"][k] for k in expected} == expected
    assert report["evaluated_sets"] == EVALUATED
    assert report["cell_counts"]["record_Y1"] > 0
    assert (C0B_JSON.stat().st_mtime_ns, D0R_JSON.stat().st_mtime_ns) == before
