"""M3-3 D0' reproducibility check: a re-run ``d0_gate.json`` against C0b.

Rule: ``docs/experiments/m33_pp_spec.md`` v0.2.4 appendix A-1 (user
decision 2026-11-15, fixed before the D0' result). This is not a gate
judgment; it asks whether the re-run reproduces the C0b one.

* Equal or mismatch: ``selection.pp_set_id``, ``selection.stop_reason``,
  ``selection.evaluated`` and, for every set of the C0b ``evaluated`` list,
  the ``checks`` G1-G4 and ``pass``.
* Within a tolerance, per condition (A0, A1, A3) x fold (0-4) cell of every
  evaluated set: G1 ``lost`` <= 1 GT; G2 ``loss_pp`` <= 0.2 %p; G3
  ``h_s0``, ``h_sj``, ``exposed_boxes_s0`` <= 1; G4 ``increment``,
  ``cumulative``, ``by_rule`` and ``by_condition`` (A1, A3, Y0) counts each
  <= max(5 % of the C0b value, 5).
* Count dicts are compared over the union of their keys (a missing key is
  0). ``null`` equals ``null`` only.
* Record only (never changes the result): the A2 cells and the C0b note-1
  boundary box (fold 1, A3, ``pipe_band``) from ``fold1/predictions.csv``
  next to each json, when present.

Usage::

    python -I -B analysis/detection/m33_d0r_compare.py \
        --base runs/eval/m33_d0/d0_gate.json \
        --new runs/eval/m33_d0r/d0_gate.json \
        --out runs/eval/m33_d0r/compare_c0b.json

``--out`` names the json report; a markdown table is written beside it
(same stem, ``.md``). Values go to these files only. The last stdout line is
``{"status", "n_cells", "n_record_only", "out"}`` with ``status`` ``match``,
``mismatch``, ``input_error`` or ``error``. Exit codes: 0 match, 1 mismatch,
2 input error (missing file or key, zero cells, cell count off the expected
count, D0' not judgeable, report inside the C0b folder), 3 internal error.
"""

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "m33_d0r_compare/1"
RULE = "docs/experiments/m33_pp_spec.md v0.2.4 A-1 (user decision 2026-11-15)"

GATE_CONDITIONS = ("A0", "A1", "A3")
RECORD_CONDITIONS = ("A2",)
FOLDS = (0, 1, 2, 3, 4)
G4_CONDITIONS = ("A1", "A3", "Y0")
CHECK_NAMES = ("G1", "G2", "G3", "G4")
SELECTION_KEYS = ("pp_set_id", "stop_reason", "evaluated")

G1_TOLERANCE_GT = 1.0
G2_TOLERANCE_PP = 0.2
G3_TOLERANCE = 1.0
G4_RELATIVE_TOLERANCE = 0.05
G4_MIN_TOLERANCE = 5.0
FLOAT_SLACK = 1e-9
"""Absorbs binary rounding of a difference such as 0.9 - 0.7 (0.2000...07)."""

BOUNDARY_FOLD = 1
BOUNDARY_CROP = "NM_20221102(142627)_N00_B10_g1.0_x(10892-11148)"
BOUNDARY_CONDITION = "A3"
BOUNDARY_CLASS = "pipe_band"
BOUNDARY_SCORE_WINDOW = (0.2, 0.3)
"""C0b note 1: the ``pipe_band`` box of this crop near conf 0.25 (record only)."""
CONF_RECALL = 0.25

STATUS_MATCH = "match"
STATUS_MISMATCH = "mismatch"
STATUS_INPUT_ERROR = "input_error"
STATUS_ERROR = "error"
EXIT_CODES = {
    STATUS_MATCH: 0,
    STATUS_MISMATCH: 1,
    STATUS_INPUT_ERROR: 2,
    STATUS_ERROR: 3,
}

logger = logging.getLogger("m33_d0r_compare")


class CompareInputError(Exception):
    """A missing file or key, or cell counts that are off (exit 2)."""


@dataclass(frozen=True)
class Cell:
    """One compared value of the two runs.

    Attributes:
        key: ``<set>/<gate>/<condition>/fold<k>/<item>`` style path.
        group: Count group (``selection``, ``checks``, ``G1`` ... ``G4_rules``,
            ``record_A2``).
        base: C0b value.
        new: D0' value.
        tolerance: Largest allowed ``|new - base|``; ``None`` = must be equal.
        record_only: Written to the report, never part of the result.
    """

    key: str
    group: str
    base: Any
    new: Any
    tolerance: float | None
    record_only: bool = False

    @property
    def diff(self) -> float | None:
        """``new - base`` for two numbers (bools excluded), else ``None``."""
        if _is_number(self.base) and _is_number(self.new):
            return self.new - self.base
        return None

    @property
    def within(self) -> bool:
        """Equal (no tolerance) or ``|new - base| <= tolerance``; null == null only."""
        if self.tolerance is None or self.base is None or self.new is None:
            return self.base == self.new
        diff = self.diff
        return diff is not None and abs(diff) <= self.tolerance + FLOAT_SLACK

    @property
    def differs(self) -> bool:
        return self.base != self.new

    def as_row(self) -> dict:
        return asdict(self) | {"diff": self.diff, "within": self.within}


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


# ------------------------------------------------------------ json access


def _require(document: Mapping, path: Sequence[str], source: str) -> Any:
    """``document[path[0]][path[1]]...``; a missing key is an input error."""
    value: Any = document
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            joined = "/".join(path)
            raise CompareInputError(f"{source}: missing {joined}")
        value = value[key]
    return value


def load_document(path: Path, source: str) -> dict:
    """Read one ``d0_gate.json``.

    Raises:
        CompareInputError: For a missing file or a file that is not JSON.
    """
    if not path.is_file():
        raise CompareInputError(f"{source}: no file {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CompareInputError(f"{source}: not JSON ({error})") from error


def evaluated_sets(base: Mapping) -> list[str]:
    """The C0b ``selection.evaluated`` list: the sets compared cell by cell."""
    sets = list(_require(base, ("selection", "evaluated"), "base"))
    if not sets:
        raise CompareInputError("base: selection.evaluated is empty (no cells)")
    return sets


# ------------------------------------------------------------- cells


def selection_cells(base: Mapping, new: Mapping) -> list[Cell]:
    return [
        Cell(
            f"selection/{key}",
            "selection",
            _require(base, ("selection", key), "base"),
            _require(new, ("selection", key), "new"),
            None,
        )
        for key in SELECTION_KEYS
    ]


def check_cells(base: Mapping, new: Mapping, set_id: str) -> list[Cell]:
    """Pass or fail of G1-G4 and the set ``pass`` flag (must be equal)."""
    cells = []
    for name in CHECK_NAMES:
        path = ("gates", set_id, "checks", name)
        cells.append(
            Cell(
                f"{set_id}/checks/{name}",
                "checks",
                _require(base, path, "base"),
                _require(new, path, "new"),
                None,
            )
        )
    path = ("gates", set_id, "pass")
    base_pass, new_pass = _require(base, path, "base"), _require(new, path, "new")
    cells.append(Cell(f"{set_id}/pass", "checks", base_pass, new_pass, None))
    return cells


def loss_cells(
    base: Mapping,
    new: Mapping,
    set_id: str,
    gate: str,
    conditions: Sequence[str] = GATE_CONDITIONS,
) -> list[Cell]:
    """G1 ``lost`` or G2 ``loss_pp`` of every condition x fold cell."""
    if gate == "G1":
        item, tolerance = "lost", G1_TOLERANCE_GT
    else:
        item, tolerance = "loss_pp", G2_TOLERANCE_PP
    record_only = tuple(conditions) == RECORD_CONDITIONS
    group = f"record_{conditions[0]}" if record_only else gate
    cells = []
    for condition in conditions:
        for fold in FOLDS:
            path = ("gates", set_id, gate, "cells", condition, str(fold), item)
            cells.append(
                Cell(
                    f"{set_id}/{gate}/{condition}/fold{fold}/{item}",
                    group,
                    _require(base, path, "base"),
                    _require(new, path, "new"),
                    tolerance,
                    record_only,
                )
            )
    return cells


def g3_cells(base: Mapping, new: Mapping, set_id: str) -> list[Cell]:
    cells = []
    for item in ("h_s0", "h_sj", "exposed_boxes_s0"):
        path = ("gates", set_id, "G3", item)
        cells.append(
            Cell(
                f"{set_id}/G3/{item}",
                "G3",
                _require(base, path, "base"),
                _require(new, path, "new"),
                G3_TOLERANCE,
            )
        )
    return cells


def g4_tolerance(base_value: Any) -> float:
    """max(5 % of the C0b value, 5) for a count."""
    magnitude = abs(base_value) if _is_number(base_value) else 0.0
    return max(G4_RELATIVE_TOLERANCE * magnitude, G4_MIN_TOLERANCE)


def _count_cells(prefix: str, base_counts: Mapping, new_counts: Mapping) -> list[Cell]:
    """One cell per key of the union; a key missing on one side counts 0."""
    cells = []
    for key in sorted(set(base_counts) | set(new_counts)):
        base_value = base_counts.get(key, 0)
        new_value = new_counts.get(key, 0)
        cells.append(
            Cell(
                f"{prefix}/{key}",
                "G4_rules",
                base_value,
                new_value,
                g4_tolerance(base_value),
            )
        )
    return cells


def g4_cells(base: Mapping, new: Mapping, set_id: str) -> list[Cell]:
    """Increment, cumulative and the per-rule counts (A1, A3, Y0)."""
    cells = []
    for item in ("increment", "cumulative"):
        path = ("gates", set_id, "G4", item)
        base_value = _require(base, path, "base")
        cells.append(
            Cell(
                f"{set_id}/G4/{item}",
                "G4",
                base_value,
                _require(new, path, "new"),
                g4_tolerance(base_value),
            )
        )
    rule_path = ("gates", set_id, "G4", "by_rule")
    cells += _count_cells(
        f"{set_id}/G4/by_rule",
        _require(base, rule_path, "base"),
        _require(new, rule_path, "new"),
    )
    for condition in G4_CONDITIONS:
        path = ("gates", set_id, "G4", "by_condition", condition)
        cells += _count_cells(
            f"{set_id}/G4/by_condition/{condition}",
            _require(base, path, "base"),
            _require(new, path, "new"),
        )
    return cells


def build_cells(base: Mapping, new: Mapping) -> list[Cell]:
    """Every compared and record-only cell, set by set in C0b order."""
    cells = selection_cells(base, new)
    for set_id in evaluated_sets(base):
        cells += check_cells(base, new, set_id)
        cells += loss_cells(base, new, set_id, "G1")
        cells += loss_cells(base, new, set_id, "G2")
        cells += g3_cells(base, new, set_id)
        cells += g4_cells(base, new, set_id)
        cells += loss_cells(base, new, set_id, "G1", RECORD_CONDITIONS)
        cells += loss_cells(base, new, set_id, "G2", RECORD_CONDITIONS)
    return cells


# ------------------------------------------------------- cell counts


def expected_counts(n_sets: int) -> dict[str, int]:
    """Cells per fixed group; ``G4_rules`` depends on the keys, not checked."""
    n_cells = len(GATE_CONDITIONS) * len(FOLDS)
    n_record = len(RECORD_CONDITIONS) * len(FOLDS)
    return {
        "selection": len(SELECTION_KEYS),
        "checks": (len(CHECK_NAMES) + 1) * n_sets,
        "G1": n_cells * n_sets,
        "G2": n_cells * n_sets,
        "G3": 3 * n_sets,
        "G4": 2 * n_sets,
        "record_A2": 2 * n_record * n_sets,
    }


def count_groups(cells: Iterable[Cell]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell.group] = counts.get(cell.group, 0) + 1
    return counts


def check_counts(cells: Sequence[Cell], n_sets: int) -> dict[str, int]:
    """Cell counts per group; they must equal :func:`expected_counts`.

    Raises:
        CompareInputError: For zero compared cells or a group count off the
            expected count.
    """
    counts = count_groups(cells)
    if not any(not cell.record_only for cell in cells):
        raise CompareInputError("no compared cells")
    expected = expected_counts(n_sets)
    off = {
        group: (counts.get(group, 0), n)
        for group, n in expected.items()
        if counts.get(group, 0) != n
    }
    if off:
        raise CompareInputError(f"cell counts off the expected (got, want): {off}")
    return counts


# ------------------------------------------------- record-only extras


def boundary_rows(run_dir: Path) -> list[dict] | None:
    """C0b note-1 boundary box rows of ``fold1/predictions.csv`` (or ``None``).

    Rows of the crop, A3, ``pipe_band`` with a score in (0.2, 0.3); the
    report keeps the score, ``score > 0.25``, category and PP actions.
    """
    path = run_dir / f"fold{BOUNDARY_FOLD}" / "predictions.csv"
    if not path.is_file():
        return None
    low, high = BOUNDARY_SCORE_WINDOW
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not _is_boundary_row(row):
                continue
            score = float(row["score"])
            if low < score < high:
                rows.append(_boundary_record(row, score))
    return rows


def _is_boundary_row(row: Mapping[str, str]) -> bool:
    return (
        row.get("image") == BOUNDARY_CROP
        and row.get("condition") == BOUNDARY_CONDITION
        and row.get("class") == BOUNDARY_CLASS
    )


def _boundary_record(row: Mapping[str, str], score: float) -> dict:
    actions = {key: value for key, value in row.items() if key.startswith("pp_action_")}
    return {
        "score": row["score"],
        "score_above_conf_recall": score > CONF_RECALL,
        "category": row.get("category", ""),
        **actions,
    }


def judgment_state(new: Mapping) -> str:
    """``judgeable`` / ``absent`` (old format, e.g. a C0b self-check).

    Raises:
        CompareInputError: When the D0' json says it is not judgeable
            ([PP] A-1: such a run is not compared).
    """
    judgment = new.get("judgment")
    if judgment is None:
        return "absent"
    if judgment.get("judgeable") is not True:
        raise CompareInputError(f"new: not judgeable ({judgment.get('blockers')})")
    return "judgeable"


# ------------------------------------------------------------ identity


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def script_identity() -> dict:
    """``git hash-object`` and sha256 of this script (blob ``None`` without git)."""
    script = Path(__file__).resolve()
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO), "hash-object", "--", str(script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=True,
        )
        blob: str | None = completed.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("git hash-object failed: %r", error)
        blob = None
    return {
        "path": script.relative_to(REPO).as_posix(),
        "git_blob": blob,
        "sha256": sha256_file(script),
    }


# ------------------------------------------------------------ report


@dataclass(frozen=True)
class Comparison:
    """Everything the report holds (values stay in the files)."""

    status: str
    cells: list[Cell]
    counts: dict[str, int]
    evaluated: list[str]
    judgment: str

    @property
    def compared(self) -> list[Cell]:
        return [cell for cell in self.cells if not cell.record_only]

    @property
    def outside(self) -> list[Cell]:
        return [cell for cell in self.compared if not cell.within]

    @property
    def differing_within(self) -> list[Cell]:
        return [cell for cell in self.compared if cell.within and cell.differs]


def compare(base: Mapping, new: Mapping) -> Comparison:
    """Compare two ``d0_gate.json`` documents under the A-1 rule.

    Raises:
        CompareInputError: For a missing key, zero cells, off cell counts or
            a D0' document that is not judgeable.
    """
    judgment = judgment_state(new)
    evaluated = evaluated_sets(base)
    cells = build_cells(base, new)
    counts = check_counts(cells, len(evaluated))
    is_match = all(cell.within for cell in cells if not cell.record_only)
    status = STATUS_MATCH if is_match else STATUS_MISMATCH
    return Comparison(status, cells, counts, evaluated, judgment)


def report_document(comparison: Comparison, base_path: Path, new_path: Path) -> dict:
    record_only = [cell for cell in comparison.cells if cell.record_only]
    return {
        "schema": SCHEMA,
        "rule": RULE,
        "status": comparison.status,
        "inputs": {
            "base": {"path": str(base_path), "sha256": sha256_file(base_path)},
            "new": {"path": str(new_path), "sha256": sha256_file(new_path)},
        },
        "script": script_identity(),
        "new_judgment": comparison.judgment,
        "evaluated_sets": comparison.evaluated,
        "cell_counts": comparison.counts,
        "expected_counts": expected_counts(len(comparison.evaluated)),
        "outside_tolerance": [cell.as_row() for cell in comparison.outside],
        "within_tolerance_differences": [
            cell.as_row() for cell in comparison.differing_within
        ],
        "cells": [cell.as_row() for cell in comparison.compared],
        "record_only": {
            "cells": [cell.as_row() for cell in record_only],
            "boundary_box": {
                "crop": BOUNDARY_CROP,
                "fold": BOUNDARY_FOLD,
                "base": boundary_rows(base_path.parent),
                "new": boundary_rows(new_path.parent),
            },
        },
    }


def _markdown_table(rows: Sequence[dict]) -> list[str]:
    lines = [
        "| cell | base | new | diff | tolerance | within |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        values = [
            row[key] for key in ("key", "base", "new", "diff", "tolerance", "within")
        ]
        lines.append("| " + " | ".join(_cell_text(value) for value in values) + " |")
    return lines


def _cell_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return str(value)


def markdown_report(document: Mapping) -> str:
    """The report as markdown: summary, the two difference lists, every cell."""
    lines = [
        f"# D0' ↔ C0b 대조 (`{SCHEMA}`)",
        "",
        f"- 규칙: {document['rule']}",
        f"- status: **{document['status']}**",
        f"- base sha256: `{document['inputs']['base']['sha256']}`",
        f"- new sha256: `{document['inputs']['new']['sha256']}`",
        f"- script blob: `{document['script']['git_blob']}`",
        f"- new judgment: {document['new_judgment']}",
        f"- 칸 수: {document['cell_counts']}",
        "",
        "## 폭 밖 차이",
        *_markdown_table(document["outside_tolerance"]),
        "",
        "## 폭 안 차이",
        *_markdown_table(document["within_tolerance_differences"]),
        "",
        "## 모든 칸",
        *_markdown_table(document["cells"]),
        "",
        "## 기록만 (A2)",
        *_markdown_table(document["record_only"]["cells"]),
        "",
        "## 기록만 (C0b 주 1 경계 박스)",
        "```json",
        json.dumps(
            document["record_only"]["boundary_box"], ensure_ascii=False, indent=1
        ),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_report(document: Mapping, out: Path) -> tuple[Path, Path]:
    """``<out>.json`` and ``<out>.md`` (LF, UTF-8)."""
    json_path, markdown_path = out.with_suffix(".json"), out.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    markdown_path.write_text(markdown_report(document), encoding="utf-8", newline="\n")
    return json_path, markdown_path


# ----------------------------------------------------------------- CLI


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True, help="C0b d0_gate.json")
    parser.add_argument("--new", type=Path, required=True, help="D0' d0_gate.json")
    parser.add_argument(
        "--out", type=Path, required=True, help="report json (a .md is written beside)"
    )
    return parser.parse_args(argv)


def _check_out_path(out: Path, base_path: Path) -> None:
    """The report never goes into the C0b result folder (read only)."""
    base_dir = base_path.resolve().parent
    if base_dir in out.resolve().parents:
        raise CompareInputError(f"--out inside the base folder {base_dir}")


def run(args: argparse.Namespace) -> tuple[str, dict]:
    """Compare and write the report; return the status and the line counts."""
    _check_out_path(args.out, args.base)
    base = load_document(args.base, "base")
    new = load_document(args.new, "new")
    comparison = compare(base, new)
    document = report_document(comparison, args.base, args.new)
    json_path, _ = write_report(document, args.out)
    counts = {
        "n_cells": len(comparison.compared),
        "n_record_only": len(comparison.cells) - len(comparison.compared),
        "out": str(json_path),
    }
    return comparison.status, counts


def main(argv: list[str] | None = None) -> int:
    """CLI entry; returns 0 match, 1 mismatch, 2 input error, 3 error."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    counts: dict = {"n_cells": None, "n_record_only": None, "out": None}
    try:
        status, counts = run(args)
    except CompareInputError as error:
        logger.error("input error: %s", error)
        status = STATUS_INPUT_ERROR
    except Exception:
        logger.exception("internal error")
        status = STATUS_ERROR
    print(json.dumps({"status": status, **counts}, ensure_ascii=False), flush=True)
    return EXIT_CODES[status]


if __name__ == "__main__":
    sys.exit(main())
