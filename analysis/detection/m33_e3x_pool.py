"""M3-3 E3x background lines: the seeded 4TU sample and the IDS 53 lines.

Spec: ``docs/processing/m33_e3x_plan.md`` v0.3 (seal 1, 72b0140) §2-1 and
appendix A-1 (rule and ``POOL_SEED`` sealed at seal 1; the list itself is
sealed at seal 2). Only line lists are read: no radargram, depth table or
detector output.

4TU candidates: the R2 ``keep_one`` lines (720) without the 177 R3 / stage-C
lines (stage A 150 + the 27 K lines, ``role == k`` in the R2 list), 176 of
which are ``keep_one`` (720 - 176 = 544 lines, 110 activities). An activity
is *shared* when it holds one of the 177 lines.

Sample rule (§2-1, seed ``POOL_SEED``):

1. every line of the unshared activities, at most 3 per activity, in seed
   order;
2. the rest up to 100 lines from the shared activities, at most 3 per
   activity, in seed order, the 100 ns records first until the whole sample
   holds at least 12 of them;
3. within-activity copies are already reduced by ``keep_one`` (R2 list).

"Seed order" is one permutation of the 544 candidates sorted by line id,
drawn once from ``numpy.random.default_rng(POOL_SEED)``; every step walks
that order. A record is 100 ns when ``m32_pipe_responses.is_long_4tu_window``
holds for ``n_samples x dt_ns`` of ``docs/processing/fourtu_lines_v1.csv``
(the M15 rule of the runners).

IDS: the 53 ``main_400`` lines of the stage-A list (``antenna_MHz`` 400).
The two precheck lines of §2-1 are written with ``role = precheck``: 4TU
``02.5/Path1`` (an R3 line) and the one 600 MHz ``report_other`` IDS line.

Columns: ``line_id, dataset, cluster_id, n_traces, dt_ns, time_window_ns,
shares_r3_activity, role``. ``cluster_id`` is the 4TU activity (the part of
the id before ``/``) or the IDS zone (the middle part). IDS ``n_traces`` and
``dt_ns`` are left empty: they are not in any line list and reading the
records is not needed to draw the sample.

Usage::

    conda run -n gpr-dev python -I -B analysis/detection/m33_e3x_pool.py [--out PATH]

The last stdout line is ``{"status", "out", "sha256_lf", "n_lines",
"n_fourtu", "n_ids", "n_precheck", "n_fourtu_shared_activity_lines",
"n_fourtu_shared_activities", "n_fourtu_100ns", "elapsed_s"}`` (counts only,
no detector value). Exit codes: 0 written, 2 an input or count is off,
3 internal error.
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common
import m32_pipe_responses as responses_io
import m32_recipe_inputs as inputs
import m33_inputs

REPO = inputs.REPO
POOL_SEED = 20261120
DEFAULT_OUT = REPO / "analysis" / "detection" / "m33_e3x_pool.csv"
FOURTU_LINES = REPO / "docs" / "processing" / "fourtu_lines_v1.csv"
"""Tracked 4TU line table (``n_samples``, ``dt_ns``); its LF sha is logged."""

N_KEEP_ONE = 720
N_R3_LINES = 177
N_R3_LINES_KEEP_ONE = 176
N_CANDIDATES = 544
N_CANDIDATE_ACTIVITIES = 110
N_FOURTU_SAMPLE = 100
MAX_LINES_PER_ACTIVITY = 3
MIN_LONG_WINDOW_LINES = 12
N_IDS_MAIN = 53
IDS_MAIN_ANTENNA_MHZ = 400.0
PRECHECK_FOURTU_LINE = "02.5/Path1"
PRECHECK_IDS_ANTENNA_MHZ = 600.0

ROLE_BACKGROUND = "background"
ROLE_PRECHECK = "precheck"
COLUMNS = (
    "line_id",
    "dataset",
    "cluster_id",
    "n_traces",
    "dt_ns",
    "time_window_ns",
    "shares_r3_activity",
    "role",
)
EXIT_CODES = {"ok": 0, "input_error": 2, "error": 3}

logger = logging.getLogger("m33_e3x_pool")


@dataclass(frozen=True)
class FourtuCandidate:
    """One 4TU ``keep_one`` line outside the 177 R3 lines."""

    line_id: str
    activity_id: str
    n_traces: int
    dt_ns: float
    time_window_ns: float
    shares_r3_activity: bool

    @property
    def is_long_window(self) -> bool:
        """A nominal 100 ns record (the M15 rule of the runners)."""
        return responses_io.is_long_4tu_window(self.time_window_ns)


def activity_of(line_id: str) -> str:
    """4TU activity id: the part before ``/``."""
    return line_id.split("/", 1)[0]


def ids_zone_of(line_id: str) -> str:
    """IDS zone: the middle part of ``pipe/<zone>/<line>``."""
    parts = line_id.split("/")
    if len(parts) != 3:
        raise inputs.InputError(f"IDS line id is not pipe/<zone>/<line>: {line_id}")
    return parts[1]


def r3_line_ids(r2_rows: Sequence[Mapping[str, str]]) -> set[str]:
    """The 177 R3 / stage-C 4TU lines: stage A 150 and the 27 K lines.

    Raises:
        inputs.InputError: For another count or a stage-A line missing from R2.
    """
    stage_a = {row["line"] for row in inputs.read_csv(inputs.STAGE_A_4TU)}
    k_lines = {row["line_id"] for row in r2_rows if row["role"] == inputs.ROLE_K}
    r2_ids = {row["line_id"] for row in r2_rows}
    missing = stage_a - r2_ids
    if missing:
        raise inputs.InputError(f"stage-A lines not in the R2 list: {sorted(missing)}")
    lines = stage_a | k_lines
    if len(lines) != N_R3_LINES:
        raise inputs.InputError(f"R3 lines {len(lines)} != {N_R3_LINES}")
    return lines


def fourtu_line_scales() -> dict[str, tuple[int, int, float]]:
    """``line_id -> (n_samples, n_traces, dt_ns)`` of the 4TU line table."""
    scales = {}
    for row in inputs.read_csv(FOURTU_LINES):
        line_id = f"{row['activity_id']}/{row['path']}"
        scales[line_id] = (
            int(row["n_samples"]),
            int(row["n_traces"]),
            float(row["dt_ns"]),
        )
    return scales


def fourtu_candidates(r2_rows: Sequence[Mapping[str, str]]) -> list[FourtuCandidate]:
    """The 544 ``keep_one`` lines outside the R3 lines, with their activity flag.

    Raises:
        inputs.InputError: For counts that differ from the design (§2-1) or a
            line whose scale is missing or disagrees with the R2 list.
    """
    r3_lines = r3_line_ids(r2_rows)
    keep_one = [row for row in r2_rows if row["keep_one"] == "True"]
    if len(keep_one) != N_KEEP_ONE:
        raise inputs.InputError(f"keep_one lines {len(keep_one)} != {N_KEEP_ONE}")
    removed = [row for row in keep_one if row["line_id"] in r3_lines]
    if len(removed) != N_R3_LINES_KEEP_ONE:
        raise inputs.InputError(f"R3 keep_one lines {len(removed)} != 176")
    shared = {activity_of(line_id) for line_id in r3_lines}
    scales = fourtu_line_scales()
    candidates = []
    for row in keep_one:
        line_id = row["line_id"]
        if line_id in r3_lines:
            continue
        if line_id not in scales:
            raise inputs.InputError(f"no scale for {line_id} in {FOURTU_LINES.name}")
        n_samples, n_traces, dt_ns = scales[line_id]
        if n_traces != int(row["n_traces"]):
            raise inputs.InputError(f"{line_id}: n_traces differs between lists")
        candidates.append(
            FourtuCandidate(
                line_id=line_id,
                activity_id=activity_of(line_id),
                n_traces=n_traces,
                dt_ns=dt_ns,
                time_window_ns=n_samples * dt_ns,
                shares_r3_activity=activity_of(line_id) in shared,
            )
        )
    activities = {candidate.activity_id for candidate in candidates}
    if (len(candidates), len(activities)) != (N_CANDIDATES, N_CANDIDATE_ACTIVITIES):
        raise inputs.InputError(
            f"candidates {len(candidates)} / activities {len(activities)} "
            f"!= {N_CANDIDATES} / {N_CANDIDATE_ACTIVITIES}"
        )
    return sorted(candidates, key=lambda candidate: candidate.line_id)


def seed_order(
    candidates: Sequence[FourtuCandidate], seed: int = POOL_SEED
) -> list[FourtuCandidate]:
    """One seeded permutation of the candidates (sorted by line id first)."""
    ordered = sorted(candidates, key=lambda candidate: candidate.line_id)
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    return [ordered[int(index)] for index in permutation]


def draw_fourtu_sample(
    candidates: Sequence[FourtuCandidate],
    seed: int = POOL_SEED,
    n_sample: int = N_FOURTU_SAMPLE,
) -> list[FourtuCandidate]:
    """The §2-1 stratified sample, in the order the lines were drawn.

    Raises:
        inputs.InputError: If the rule cannot reach ``n_sample`` lines.
    """
    order = seed_order(candidates, seed)
    chosen: list[FourtuCandidate] = []
    per_activity: Counter[str] = Counter()

    def take(candidate: FourtuCandidate) -> bool:
        if len(chosen) >= n_sample or candidate in chosen:
            return False
        if per_activity[candidate.activity_id] >= MAX_LINES_PER_ACTIVITY:
            return False
        chosen.append(candidate)
        per_activity[candidate.activity_id] += 1
        return True

    for candidate in order:
        if not candidate.shares_r3_activity:
            take(candidate)
    for candidate in order:
        if sum(c.is_long_window for c in chosen) >= MIN_LONG_WINDOW_LINES:
            break
        if candidate.shares_r3_activity and candidate.is_long_window:
            take(candidate)
    for candidate in order:
        if candidate.shares_r3_activity:
            take(candidate)
    if len(chosen) != n_sample:
        raise inputs.InputError(f"the rule drew {len(chosen)} lines, not {n_sample}")
    return chosen


def ids_rows() -> tuple[list[dict], list[dict]]:
    """IDS 53 ``main_400`` lines and the 600 MHz precheck line (stage-A list).

    Raises:
        inputs.InputError: For another count of 400 MHz or 600 MHz lines.
    """
    rows = inputs.read_csv(inputs.STAGE_A_IDS)
    main = [r for r in rows if float(r["antenna_MHz"]) == IDS_MAIN_ANTENNA_MHZ]
    other = [r for r in rows if float(r["antenna_MHz"]) == PRECHECK_IDS_ANTENNA_MHZ]
    if len(main) != N_IDS_MAIN or len(other) != 1:
        raise inputs.InputError(f"IDS 400 / 600 MHz lines {len(main)} / {len(other)}")

    def row_of(row: Mapping[str, str], role: str) -> dict:
        return {
            "line_id": row["line"],
            "dataset": inputs.DATASET_IDS,
            "cluster_id": ids_zone_of(row["line"]),
            "n_traces": "",
            "dt_ns": "",
            "time_window_ns": float(row["time_window_ns"]),
            "shares_r3_activity": True,
            "role": role,
        }

    return [row_of(r, ROLE_BACKGROUND) for r in main], [row_of(other[0], ROLE_PRECHECK)]


def fourtu_row(candidate: FourtuCandidate, role: str = ROLE_BACKGROUND) -> dict:
    """Pool row of a 4TU line."""
    return {
        "line_id": candidate.line_id,
        "dataset": inputs.DATASET_4TU,
        "cluster_id": candidate.activity_id,
        "n_traces": candidate.n_traces,
        "dt_ns": candidate.dt_ns,
        "time_window_ns": candidate.time_window_ns,
        "shares_r3_activity": candidate.shares_r3_activity,
        "role": role,
    }


def precheck_fourtu_row(r2_rows: Sequence[Mapping[str, str]]) -> dict:
    """``02.5/Path1``: an R3 line, outside the sample (§2-1 precheck lines)."""
    if PRECHECK_FOURTU_LINE not in r3_line_ids(r2_rows):
        raise inputs.InputError(f"{PRECHECK_FOURTU_LINE} is not an R3 line")
    n_samples, n_traces, dt_ns = fourtu_line_scales()[PRECHECK_FOURTU_LINE]
    candidate = FourtuCandidate(
        PRECHECK_FOURTU_LINE,
        activity_of(PRECHECK_FOURTU_LINE),
        n_traces,
        dt_ns,
        n_samples * dt_ns,
        True,
    )
    return fourtu_row(candidate, ROLE_PRECHECK)


def build_pool(r2_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict], dict]:
    """All pool rows (4TU sample in line order, IDS, precheck) and the counts."""
    sample = draw_fourtu_sample(fourtu_candidates(r2_rows))
    fourtu = [fourtu_row(c) for c in sorted(sample, key=lambda c: c.line_id)]
    ids_main, ids_precheck = ids_rows()
    shared = [c for c in sample if c.shares_r3_activity]
    counts = {
        "n_fourtu": len(fourtu),
        "n_ids": len(ids_main),
        "n_precheck": 2,
        "n_fourtu_shared_activity_lines": len(shared),
        "n_fourtu_shared_activities": len({c.activity_id for c in shared}),
        "n_fourtu_activities": len({c.activity_id for c in sample}),
        "n_fourtu_100ns": sum(c.is_long_window for c in sample),
    }
    rows = fourtu + ids_main + [precheck_fourtu_row(r2_rows), *ids_precheck]
    return rows, counts


def assert_inputs() -> dict[str, str]:
    """LF sha of every list the pool reads; the R2 and stage-A ones asserted."""
    return {
        "r2_lines": inputs.checked_text_sha256(
            m33_inputs.R2_LINES_CSV, m33_inputs.R2_LINES_SHA256_LF, "R2 line list"
        ),
        "stage_a_4tu": inputs.checked_text_sha256(
            inputs.STAGE_A_4TU, inputs.EXPECTED_SHA256["stage_a_4tu"], "stage_a_4tu"
        ),
        "stage_a_ids": inputs.checked_text_sha256(
            inputs.STAGE_A_IDS, inputs.EXPECTED_SHA256["stage_a_ids"], "stage_a_ids"
        ),
        "fourtu_lines": inputs.sha256_file_lf(FOURTU_LINES),
    }


def write_pool(path: Path, rows: Sequence[Mapping]) -> str:
    """Write the pool (LF csv, full-precision floats) and return its LF sha."""
    common.write_csv(path, COLUMNS, ([row[c] for c in COLUMNS] for row in rows))
    return inputs.sha256_file_lf(path)


def read_pool(path: Path = DEFAULT_OUT) -> list[dict[str, str]]:
    """Rows of a pool file (the runner reads it through this)."""
    return inputs.read_csv(path)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry; returns the exit code (see module docstring)."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    started = time.time()
    final: dict = {"status": "error", "out": str(args.out)}
    try:
        input_sha = assert_inputs()
        rows, counts = build_pool(inputs.read_csv(m33_inputs.R2_LINES_CSV))
        sha = write_pool(args.out, rows)
        final = {
            "status": "ok",
            "out": str(args.out),
            "sha256_lf": sha,
            "n_lines": len(rows),
            **counts,
        }
        logger.info("inputs LF sha256: %s", json.dumps(input_sha, sort_keys=True))
    except inputs.InputError as error:
        logger.error("input error: %s", error)
        final["status"] = "input_error"
    except Exception:
        logger.exception("internal error")
    final["elapsed_s"] = round(time.time() - started, 2)
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return EXIT_CODES[final["status"]]


if __name__ == "__main__":
    sys.exit(main())
