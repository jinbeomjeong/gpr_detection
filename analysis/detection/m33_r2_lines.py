"""M3-3 R2 line list: the 4TU 754 unique lines with ``keep_one`` and role.

Spec: ``docs/experiments/m33_runner_spec.md`` v0.3.6 §2-1 1 ([RUN]) and the
[U3] duplicate rules (``docs/processing/spec_eval_4tu_v1.md`` §4-5, counts in
``docs/datasets/m33_data_assertions.md`` §2-1).

* ``keep_both`` (754): the 759 lines of ``docs/processing/fourtu_lines_v1.csv``
  without the five within-activity copies of
  ``docs/datasets/fourtu_duplicate_radargrams.csv``.
* ``keep_one`` (720): the [U3] rule of ``scripts/eval_depth_4tu.py``
  (``keep_one_plan``: strict superset wins, else the lower activity number)
  drops 34 shared lines.
* ``role``: ``k`` for the 27 K-bundle lines of ``data/processing/d3/lines.csv``
  (the M6 gate lines), else ``background``.

The table is written to ``analysis/detection/m33_r2_lines.csv`` (LF) and its
LF sha is what R2 records. When ``data/audit/m33_x7/
fourtu_keep_one_dropped_lines.csv`` (curator X7) is present the dropped set
is checked against it. Only file names, activity numbers and line lists are
read; no depth table and no radargram.

Usage::

    python -I -B analysis/detection/m33_r2_lines.py [--out PATH]

The last stdout line is ``{"status", "out", "sha256_lf", "n_lines",
"n_keep_one"}``. Exit codes: 0 written, 2 an input or count is off, 3 error.
"""

import argparse
import csv
import importlib.util
import json
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m32_recipe_inputs as inputs

REPO = inputs.REPO
FOURTU_LINES = REPO / "docs" / "processing" / "fourtu_lines_v1.csv"
DUPLICATES = REPO / "docs" / "datasets" / "fourtu_duplicate_radargrams.csv"
EVAL_DEPTH_SCRIPT = REPO / "scripts" / "eval_depth_4tu.py"
X7_DROPPED = REPO / "data" / "audit" / "m33_x7" / "fourtu_keep_one_dropped_lines.csv"
DEFAULT_OUT = REPO / "analysis" / "detection" / "m33_r2_lines.csv"

N_ALL_LINES = 759
N_LINES_UNIQUE = 754
N_KEEP_ONE = 720
DATASET = inputs.DATASET_4TU
COLUMNS = (
    "line_id",
    "activity_id",
    "path",
    "site",
    "dataset",
    "keep_one",
    "role",
    "n_traces",
)
EXIT_CODES = {"ok": 0, "input_error": 2, "error": 3}

logger = logging.getLogger("m33_r2_lines")


@dataclass(frozen=True)
class LineRow:
    """One row of the R2 line list."""

    activity_id: str
    path: str
    site: str
    n_traces: int
    keep_one: bool
    role: str

    @property
    def line_id(self) -> str:
        """``<activity_id>/<path>`` (split at the first ``/``, [RUN] §5)."""
        return f"{self.activity_id}/{self.path}"

    def as_row(self) -> dict:
        return {
            "line_id": self.line_id,
            "activity_id": self.activity_id,
            "path": self.path,
            "site": self.site,
            "dataset": DATASET,
            "keep_one": self.keep_one,
            "role": self.role,
            "n_traces": self.n_traces,
        }


def _eval_depth_module() -> ModuleType:
    """``scripts/eval_depth_4tu.py`` as a module (its main does not run)."""
    spec = importlib.util.spec_from_file_location("eval_depth_4tu", EVAL_DEPTH_SCRIPT)
    if spec is None or spec.loader is None:
        raise inputs.InputError(f"cannot import {EVAL_DEPTH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def keep_one_dropped(
    all_lines: Sequence[Mapping[str, str]], duplicates_path: Path
) -> tuple[frozenset[tuple[str, str]], frozenset[tuple[str, str]]]:
    """Within-activity copies and the ``keep_one`` dropped lines ([U3]).

    Returns:
        ``(within_copies, keep_one_dropped)`` as ``(activity_id, path)`` sets.
    """
    import pandas as pd

    eval_depth = _eval_depth_module()
    table = eval_depth.load_duplicate_table(duplicates_path)
    frame = pd.DataFrame(
        {
            "activity_id": [row["activity_id"] for row in all_lines],
            "path": [row["path"] for row in all_lines],
        }
    )
    lines_per_activity = eval_depth.unique_line_counts(frame, table)
    dropped = {
        (decision.dropped, path)
        for decision in eval_depth.keep_one_plan(table, lines_per_activity)
        for path in decision.dropped_paths
    }
    return frozenset(table.within_copies()), frozenset(dropped)


def k_bundle(k_lines: Iterable[inputs.LineRef]) -> frozenset[str]:
    return frozenset(ref.line_id for ref in k_lines)


def build_rows(
    all_lines: Sequence[Mapping[str, str]],
    within_copies: frozenset[tuple[str, str]],
    dropped: frozenset[tuple[str, str]],
    k_line_ids: frozenset[str],
) -> list[LineRow]:
    """The 754 lines in the table order of ``fourtu_lines_v1.csv``."""
    rows = []
    for line in all_lines:
        key = (line["activity_id"], line["path"])
        if key in within_copies:
            continue
        line_id = f"{key[0]}/{key[1]}"
        rows.append(
            LineRow(
                activity_id=key[0],
                path=key[1],
                site=line["site"],
                n_traces=int(line["n_traces"]),
                keep_one=key not in dropped,
                role=inputs.ROLE_K if line_id in k_line_ids else inputs.ROLE_BACKGROUND,
            )
        )
    return rows


def check_rows(rows: Sequence[LineRow], k_line_ids: frozenset[str]) -> dict:
    """754 unique lines, 720 ``keep_one``, every K line present (and kept).

    Raises:
        inputs.InputError: For any other count.
    """
    n_keep = sum(row.keep_one for row in rows)
    ids = [row.line_id for row in rows]
    k_present = {row.line_id for row in rows if row.role == inputs.ROLE_K}
    problems = []
    if len(rows) != N_LINES_UNIQUE or len(set(ids)) != len(ids):
        problems.append(f"{len(rows)} lines ({len(set(ids))} unique), want 754")
    if n_keep != N_KEEP_ONE:
        problems.append(f"{n_keep} keep_one lines, want 720")
    if k_present != k_line_ids:
        problems.append(f"K lines missing: {sorted(k_line_ids - k_present)}")
    if problems:
        raise inputs.InputError("; ".join(problems))
    return {"n_lines": len(rows), "n_keep_one": n_keep, "n_k": len(k_present)}


def check_against_x7(dropped: frozenset[tuple[str, str]]) -> str:
    """``match`` / ``absent``; a different X7 dropped list is refused.

    Raises:
        inputs.InputError: When the curator's list differs.
    """
    if not X7_DROPPED.is_file():
        return "absent"
    listed = {(row["activity_id"], row["path"]) for row in inputs.read_csv(X7_DROPPED)}
    if listed != set(dropped):
        raise inputs.InputError("keep_one dropped lines differ from the X7 list")
    return "match"


def write_rows(path: Path, rows: Sequence[LineRow]) -> str:
    """Write the list (LF) and return its LF sha."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.as_row() for row in rows)
    return inputs.sha256_file_lf(path)


def build(out: Path) -> dict:
    """Build, check and write the R2 line list."""
    all_lines = inputs.read_csv(FOURTU_LINES)
    if len(all_lines) != N_ALL_LINES:
        raise inputs.InputError(f"{FOURTU_LINES.name} has {len(all_lines)} lines")
    within_copies, dropped = keep_one_dropped(all_lines, DUPLICATES)
    k_line_ids = k_bundle(inputs.k_lines())
    rows = build_rows(all_lines, within_copies, dropped, k_line_ids)
    counts = check_rows(rows, k_line_ids)
    x7 = check_against_x7(dropped)
    sha = write_rows(out, rows)
    return {"status": "ok", "out": str(out), "sha256_lf": sha, **counts, "x7": x7}


def main(argv: list[str] | None = None) -> int:
    """CLI entry; see the module docstring."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        line = build(args.out)
    except inputs.InputError as error:
        logger.error("input error: %s", error)
        line = {"status": "input_error", "out": str(args.out)}
    except Exception:
        logger.exception("internal error")
        line = {"status": "error", "out": str(args.out)}
    print(json.dumps(line, ensure_ascii=False), flush=True)
    return EXIT_CODES[line["status"]]


if __name__ == "__main__":
    sys.exit(main())
