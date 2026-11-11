"""S1g harness (M2-10, spec_eval_4tu_v1 v1.9 section 9-9-1 (c), 12-4-3).

Runs the 4TU code path on the gprMax S1g scan lines and writes the candidate
table with each QC-passing candidate attached to an expected event. **No verdict
is made here.**

K1 was rejected by the S1g verdict and its code removed (user decision
2026-11-11): this version writes no ``cluster_mu`` column and no K1 estimates.
The S1g verdict (``docs/processing/m210_k1_s1g_judgment.md``) is reproduced at
commit 90845b4. Event attachment and the G0 record stay for later re-tests
(K7).

Per line (``S1G_D060``, ``S1G_D100``, ``S1G_C1``, ``S1G_BGL``):

1. ``load_synthetic(id, include_eps=True, include_radius=False,
   include_height=True)``: metadata eps_r only, no radius (as 4TU), antenna
   height = json ``air_gap_m``; ``line_id`` ``S1G/<id>``.
2. ``prepare_depth_inputs(rg, DepthConfig())`` -> the generator wrapper
   ``apex_candidates_for_radargram`` -> ``eval_depth_4tu.run_line(rg,
   candidates, apex_qc=True)`` (the evaluation's own depth and QC calls).
3. QC-passing candidates (``ok`` and not excluded by B+D+E) are attached to the
   ``expected_events`` of ``models_s1g.json`` (``attach_event``) and clustered
   by depth (complete linkage, one-line support filter).

Outputs (``--out-dir``, default ``data/processing/m210_s1g``):

* ``candidates.csv``: one row per candidate (``CANDIDATE_COLUMNS``).
* ``run_info.json``: seal commit and spec blob, input SHA-256, zero rule and
  t_z per line, ``m29c_qc_constants.json`` SHA-256 and the code constants
  against it, ``DepthConfig()`` defaults, the ``MKL_CBWR`` value and the
  generator wrapper regression (G0 (i), 4TU apex candidates regenerated into
  ``<out-dir>/wrapper_regression``; no 4TU ground truth is read).

Run (one core, BelowNormal):
    conda run -n gpr-dev python -I analysis/processing/m210_s1g_pipeline.py
    ... --regression-activities none   (skip the 4TU regeneration, ~4 min)
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# Reproducible MKL results across processes, before numpy loads (M2-10); an
# explicit value from the caller wins. Same setting as scripts/eval_depth_4tu.py.
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "analysis" / "processing"))

import eval_depth_4tu as ev  # noqa: E402
import fourtu_apex_candidate_select as generator  # noqa: E402

from gpr_detection.depth.config import DEFAULT_DEPTH_CONFIG, DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402
from gpr_detection.io.radargram import Radargram  # noqa: E402
from gpr_detection.io.synthetic import load_synthetic  # noqa: E402

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "synthetic_gprmax_s1g"
DEFAULT_MODELS_JSON = (
    PROJECT_ROOT / "analysis" / "synthetic" / "s1g" / "models_s1g.json"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processing" / "m210_s1g"
SPEC_PATH = "docs/processing/spec_eval_4tu_v1.md"
QC_CONSTANTS_JSON = PROJECT_ROOT / "docs" / "processing" / "m29c_qc_constants.json"
QC_CONSTANTS_LOCKED_SHA256 = ("6a88bd8d", "ec56ba8c")
"""Prefix and suffix of the locked SHA-256 quoted in spec 11 (v1.4)."""
QC_CONSTANT_NAMES = (
    "QC_R_LO",
    "QC_R_HI",
    "QC_R_TOL",
    "QC_UNSTABLE_SIGMA_REL",
    "QC_UNSTABLE_INLIER",
    "QC_BOUND_REL",
    "QC_RADIUS_M",
    "RECORD_END_PERIODS",
    "SEAM_TIME_PERIODS",
    "SEAM_AMP_RATIO",
    "SEAM_NOISE_K",
    "SEAM_ZERO_HALF_PERIODS",
    "SEAM_LAG_HALF_PERIODS",
    "SEAM_REF_HALF_M",
    "SEAM_APEX_HALF_M",
)
"""``eval_depth_4tu`` constants locked in ``m29c_qc_constants.json`` (same list
as the test ``test_qc_constants_match_the_locked_json``)."""
SEAL_COMMIT = "b3e68fa"
SEAL_SPEC_BLOB = "0f368eec"
"""Sealed S1g criteria: the commit and spec blob given for spec 12-4 (2026-11-11)."""
S1G_MODELS = ("S1G_D060", "S1G_D100", "S1G_C1", "S1G_BGL")
LINE_ACTIVITY = "S1G"

ATTACH_X_M = 0.10
"""|x_apex - x_event| limit (5 traces, spec 12-4-3), boundary included."""
ATTACH_PERIODS = 0.5
"""|t_apex - t_event| limit in periods of f_c (1 ns at 500 MHz), boundary included."""
ATTACH_TOL = 1e-9
"""Float slack of both attach limits (m and ns) so the boundary is inside."""
KIND_PRIMARY = "primary"
KIND_UNEXPLAINED = "unexplained"
EVENT_KINDS = (KIND_PRIMARY, "multiple_surface", "multiple_pipe", "multiple_other")

CANDIDATE_COLUMNS = (
    "model_id",
    "candidate_trace",
    "candidate_sample",
    "apex_trace",
    "apex_x_m",
    "apex_t_ns",
    "depth_top_m",
    "status",
    "qc_verdict",
    "qc_reason",
    "qc_excluded",
    "event_kind",
    "event_pipe_index",
    "cluster_id",
    "cluster_depth_m",
    "zero_rule",
    "t_zero_ns",
)
"""``candidates.csv`` columns: spec 9-9-1 (c) without ``cluster_mu`` (K1 removed)."""
GENERATOR_COLUMNS = ("trace", "sample", "tau_ns", "source")
GENERATOR_FILES = (
    "fourtu_apex_candidates_v1.csv",
    "fourtu_apex_candidate_lines_v1.csv",
)


# ----------------------------------------------------------------- event attachment


@dataclass(frozen=True)
class ExpectedEvent:
    """One ``expected_events`` entry (EM, recorded before the full runs)."""

    event_id: str
    kind: str
    x_m: float
    """Apex x on the export axis ``trace_x_m_first + i dx``."""
    t_ns: float
    """Apex two-way time on the record axis ``time_of_sample0_ns + k dt``."""
    pipe_index: int | None = None
    """Pipe of a ``primary`` event; ``None`` otherwise."""


def parse_events(entries: Sequence[Mapping[str, object]]) -> list[ExpectedEvent]:
    """``expected_events[<id>]`` of ``models_s1g.json``, json order kept.

    Raises:
        ValueError: An unknown ``kind``.
    """
    events = []
    for entry in entries:
        kind = str(entry["kind"])
        if kind not in EVENT_KINDS:
            raise ValueError(f"{entry.get('event_id')}: unknown event kind {kind!r}")
        pipe_index = entry.get("pipe_index")
        events.append(
            ExpectedEvent(
                event_id=str(entry["event_id"]),
                kind=kind,
                x_m=float(entry["x_m"]),
                t_ns=float(entry["t_ns"]),
                pipe_index=None if pipe_index is None else int(pipe_index),
            )
        )
    return events


def attach_event(
    apex_x_m: float, apex_t_ns: float, events: Sequence[ExpectedEvent], f_c_hz: float
) -> ExpectedEvent | None:
    """The expected event of a fitted apex, or ``None`` (unexplained, 12-4-3).

    Candidates are the events with ``|dx| <= 0.10 m`` and ``|dt| <= 0.5 / f_c``.
    Among them a ``primary`` wins, then the nearest time, then the smaller
    ``|dx|``, then the earlier json entry.
    """
    t_limit_ns = ATTACH_PERIODS * 1e9 / f_c_hz
    best_key, best_event = None, None
    for order, event in enumerate(events):
        dx_m = abs(apex_x_m - event.x_m)
        dt_ns = abs(apex_t_ns - event.t_ns)
        if dx_m > ATTACH_X_M + ATTACH_TOL or dt_ns > t_limit_ns + ATTACH_TOL:
            continue
        key = (event.kind != KIND_PRIMARY, dt_ns, dx_m, order)
        if best_key is None or key < best_key:
            best_key, best_event = key, event
    return best_event


# ----------------------------------------------------------------- one line


@dataclass(frozen=True)
class LineAxes:
    """Export axes of one line (``Radargram.metadata`` of ``load_synthetic``)."""

    trace_x_m_first: float
    dx_m: float
    time_of_sample0_ns: float
    dt_ns: float

    def x_m(self, trace: float) -> float:
        return self.trace_x_m_first + trace * self.dx_m

    def t_ns(self, sample: float) -> float:
        return self.time_of_sample0_ns + sample * self.dt_ns


def _line_axes(radargram: Radargram) -> LineAxes:
    assert radargram.dx_m is not None and radargram.dt_ns is not None
    return LineAxes(
        trace_x_m_first=float(radargram.metadata["trace_x_m_first"]),
        dx_m=radargram.dx_m,
        time_of_sample0_ns=float(radargram.metadata.get("time_of_sample0_ns", 0.0)),
        dt_ns=radargram.dt_ns,
    )


def _generator_table(kept: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """The wrapper's kept rows as the ``run_line`` input (spec 9-9-1 (c))."""
    return pd.DataFrame(list(kept), columns=list(GENERATOR_COLUMNS))


def _passes_qc(candidate: ev.CandidateResult) -> bool:
    """``ok`` and not dropped by the main rule B+D+E (12-4-3 "QC 통과 후보")."""
    is_ok = candidate.result.status == "ok"
    return is_ok and candidate.qc is not None and not candidate.qc.excluded


@dataclass(frozen=True)
class ClusterLabel:
    cluster_id: int
    depth_m: float


def _cluster_labels(
    passing: Sequence[ev.CandidateResult], radargram: Radargram
) -> dict[int, ClusterLabel]:
    """Depth clusters of the QC-passing candidates, per index into ``passing``.

    The line is read as a one-line activity (complete linkage, the support
    filter keeps every cluster), as in the evaluation's ``main`` path.
    """
    depths = [float(c.result.depth_top_m) for c in passing]
    clusters = ev.cluster_depths(depths, [radargram.line_id] * len(depths))
    kept = ev.supported(clusters, 1, ev.MIN_SUPPORT_LINES)
    labels: dict[int, ClusterLabel] = {}
    for cluster_id, cluster in enumerate(clusters):
        if cluster not in kept:
            continue
        for index in cluster.member_indices:
            labels[index] = ClusterLabel(cluster_id, cluster.depth_m)
    return labels


def _blank_or_rounded(value: float | None, digits: int) -> object:
    return "" if value is None or not math.isfinite(value) else round(value, digits)


def _candidate_row(
    model_id: str,
    candidate: ev.CandidateResult,
    axes: LineAxes,
    event: ExpectedEvent | None,
    label: ClusterLabel | None,
    is_attached: bool,
) -> dict[str, object]:
    result, qc = candidate.result, candidate.qc
    has_apex = result.apex_trace is not None and result.apex_sample is not None
    if not is_attached:
        event_kind = ""
    elif event is None:
        event_kind = KIND_UNEXPLAINED
    else:
        event_kind = event.kind
    pipe_index = "" if event is None or event.pipe_index is None else event.pipe_index
    return {
        "model_id": model_id,
        "candidate_trace": candidate.candidate_trace,
        "candidate_sample": round(candidate.candidate_sample, 2),
        "apex_trace": "" if result.apex_trace is None else result.apex_trace,
        "apex_x_m": round(axes.x_m(result.apex_trace), 4) if has_apex else "",
        "apex_t_ns": round(axes.t_ns(result.apex_sample), 4) if has_apex else "",
        "depth_top_m": _blank_or_rounded(result.depth_top_m, 4),
        "status": result.status.value,
        "qc_verdict": "" if qc is None else qc.curvature.verdict.value,
        "qc_reason": "" if qc is None else qc.curvature.reason,
        "qc_excluded": "" if qc is None else qc.excluded,
        "event_kind": event_kind,
        "event_pipe_index": pipe_index,
        "cluster_id": "" if label is None else label.cluster_id,
        "cluster_depth_m": "" if label is None else round(label.depth_m, 4),
        "zero_rule": result.zero_rule or "",
        "t_zero_ns": _blank_or_rounded(result.t_zero_ns, 4),
    }


def run_model(
    model_id: str, radargram: Radargram, events: Sequence[ExpectedEvent]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Candidate rows and the ``run_info`` entry of one S1g line.

    Args:
        model_id: e.g. ``"S1G_D100"``.
        radargram: ``load_synthetic`` result; its ``line_id`` must be ``S1G/<id>``.
        events: The line's expected events (json order).

    Raises:
        ValueError: ``line_id`` is not ``S1G/<model_id>``.
    """
    if radargram.line_id != f"{LINE_ACTIVITY}/{model_id}":
        raise ValueError(f"line_id {radargram.line_id!r} is not S1G/{model_id}")
    inputs = prepare_depth_inputs(radargram, DepthConfig())
    line, kept, _ = generator.apex_candidates_for_radargram(
        radargram, inputs, act=LINE_ACTIVITY, path=model_id
    )
    candidates = ev.run_line(radargram, _generator_table(kept), apex_qc=True)
    axes = _line_axes(radargram)
    passing = [c for c in candidates if _passes_qc(c)]
    labels = _cluster_labels(passing, radargram)
    passing_index = {id(c): index for index, c in enumerate(passing)}
    rows = []
    for candidate in candidates:
        index = passing_index.get(id(candidate))
        event = None
        if index is not None:
            event = attach_event(
                axes.x_m(candidate.result.apex_trace),
                axes.t_ns(candidate.result.apex_sample),
                events,
                radargram.f_c_hz,
            )
        label = None if index is None else labels.get(index)
        rows.append(
            _candidate_row(
                model_id, candidate, axes, event, label, is_attached=index is not None
            )
        )
    info = {
        "line_id": radargram.line_id,
        "zero_rule": None
        if inputs.zero.zero_rule is None
        else str(inputs.zero.zero_rule),
        "t_zero_ns": inputs.zero.t_zero_ns,
        "antenna_height_m": radargram.antenna_height_m,
        "eps_r_meta": radargram.eps_r_meta,
        "pipe_radius_m": radargram.pipe_radius_m,
        "axes": dataclasses.asdict(axes),
        "generator_line": {k: v for k, v in line.items() if not k.startswith("_")},
        "n_candidates": len(candidates),
        "n_qc_passing": len(passing),
        "n_expected_events": len(events),
    }
    return rows, info


# ----------------------------------------------------------------- run record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    """Read-only git query in the project; ``None`` when git fails."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"git {' '.join(args)} failed: {error}", file=sys.stderr)
        return None
    return completed.stdout.strip()


def seal_record() -> dict[str, object]:
    """Seal commit, spec blob at that commit and now (facts only, no verdict)."""
    return {
        "seal_commit": SEAL_COMMIT,
        "seal_spec_blob": SEAL_SPEC_BLOB,
        "seal_commit_full": _git("rev-parse", SEAL_COMMIT),
        "spec_blob_at_seal_commit": _git("rev-parse", f"{SEAL_COMMIT}:{SPEC_PATH}"),
        "spec_blob_working_tree": _git("hash-object", SPEC_PATH),
        "head_commit": _git("rev-parse", "HEAD"),
        "dirty_files": (_git("status", "--porcelain") or "").splitlines(),
    }


def qc_constants_record() -> dict[str, object]:
    """``m29c_qc_constants.json`` SHA-256 and the code constants against it."""
    sha = _sha256(QC_CONSTANTS_JSON)
    locked = json.loads(QC_CONSTANTS_JSON.read_text(encoding="utf-8"))
    differing = [
        name for name in QC_CONSTANT_NAMES if getattr(ev, name) != locked[name]
    ]
    prefix, suffix = QC_CONSTANTS_LOCKED_SHA256
    return {
        "path": QC_CONSTANTS_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": sha,
        "matches_locked_sha_quote": sha.startswith(prefix) and sha.endswith(suffix),
        "code_constants_checked": list(QC_CONSTANT_NAMES),
        "code_constants_differing": differing,
    }


def depth_config_record() -> dict[str, object]:
    config = DepthConfig()
    return {
        "equals_default": config == DEFAULT_DEPTH_CONFIG,
        "values": json.loads(json.dumps(dataclasses.asdict(config), default=str)),
    }


def input_record(data_root: Path, models_json: Path, model_ids: Sequence[str]) -> dict:
    files = {"models_json": {"path": str(models_json), "sha256": _sha256(models_json)}}
    for model_id in model_ids:
        for suffix in (".npy", ".json"):
            path = data_root / f"{model_id}{suffix}"
            files[f"{model_id}{suffix}"] = {"path": str(path), "sha256": _sha256(path)}
    return files


def _read_csv_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def _auto_rows(rows: list[list[str]], activities: set[str] | None) -> list[list[str]]:
    source = rows[0].index("source")
    return [
        row
        for row in rows[1:]
        if row[source] == "auto_v1" and (activities is None or row[0] in activities)
    ]


def wrapper_regression(out_dir: Path, activities: str) -> dict[str, object]:
    """G0 (i): regenerate the 4TU apex candidates with the wrapper (spec 9-9-1 (b)).

    The generator's ``main`` runs with its ``OUT`` constant pointed at
    ``out_dir``; the committed csvs are only read. ``activities`` is ``all``,
    ``none`` (not run) or a comma list.
    """
    if activities == "none":
        return {"run": False}
    out_dir.mkdir(parents=True, exist_ok=True)
    original_out, original_argv = generator.OUT, sys.argv
    wanted = "" if activities == "all" else activities
    try:
        generator.OUT = out_dir
        sys.argv = ["fourtu_apex_candidate_select", "--activities", wanted]
        generator.main()
    finally:
        generator.OUT, sys.argv = original_out, original_argv
    selected = None if activities == "all" else set(activities.split(","))
    record: dict[str, object] = {"run": True, "activities": activities}
    for name in GENERATOR_FILES:
        committed, regenerated = original_out / name, out_dir / name
        record[name] = {
            "committed_sha256": _sha256(committed),
            "regenerated_sha256": _sha256(regenerated),
            "byte_identical": committed.read_bytes() == regenerated.read_bytes(),
        }
    committed_rows = _read_csv_rows(original_out / GENERATOR_FILES[0])
    regenerated_rows = _read_csv_rows(out_dir / GENERATOR_FILES[0])
    old, new = (
        _auto_rows(committed_rows, selected),
        _auto_rows(regenerated_rows, selected),
    )
    record["auto_rows"] = {
        "committed": len(old),
        "regenerated": len(new),
        "identical": old == new,
    }
    return record


# ----------------------------------------------------------------- entry point


def load_line(model_id: str, data_root: Path) -> Radargram:
    """S1g line as the harness reads it (spec 9-9-1 (c) "불러오기")."""
    radargram = load_synthetic(
        model_id,
        root=data_root,
        include_eps=True,
        include_radius=False,
        include_height=True,
    )
    return dataclasses.replace(radargram, line_id=f"{LINE_ACTIVITY}/{model_id}")


def write_candidates(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CANDIDATE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--models-json", type=Path, default=DEFAULT_MODELS_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--models", default=",".join(S1G_MODELS))
    parser.add_argument(
        "--regression-activities",
        default="all",
        help="4TU activities for the wrapper regression: all, none or a comma list",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_ids = args.models.split(",")
    models = json.loads(args.models_json.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    lines: dict[str, object] = {}
    for model_id in model_ids:
        events = parse_events(models["expected_events"][model_id])
        line_rows, lines[model_id] = run_model(
            model_id, load_line(model_id, args.data_root), events
        )
        rows.extend(line_rows)
        print(f"{model_id}: {len(line_rows)} candidates", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_candidates(args.out_dir / "candidates.csv", rows)
    run_info = {
        "spec": "spec_eval_4tu_v1 v1.9 section 9-9-1 (c); no verdict here",
        "seal": seal_record(),
        "inputs": input_record(args.data_root, args.models_json, model_ids),
        "qc_constants": qc_constants_record(),
        "depth_config": depth_config_record(),
        "mkl_cbwr": os.environ.get("MKL_CBWR"),
        "attach": {
            "x_m": ATTACH_X_M,
            "periods": ATTACH_PERIODS,
            "tolerance": ATTACH_TOL,
            "order": "primary, nearest time, smaller |dx|, json order",
        },
        "lines": lines,
        "wrapper_regression": wrapper_regression(
            args.out_dir / "wrapper_regression", args.regression_activities
        ),
    }
    (args.out_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=1, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"wrote {args.out_dir / 'candidates.csv'} and run_info.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
