"""M2-9 improvement E pre-survey: implausible antenna heights of the 4TU zero rule.

No ground-truth depth is read. Per line of ``docs/processing/fourtu_lines_v1.csv``
the default depth path (``prepare_depth_inputs``, ``DepthConfig()``) finds the
time zero; the early events of the mean trace are re-detected with the same
settings to keep their amplitudes and the search-window end.

For lines with ``zero_rule == "air_surface_late"`` the antenna height
``h_est = c * (t_last - t_first) / 2`` (offset unknown, ``timezero._antenna_height_m``)
is compared with the searched maximum ``h_max = 0.5 m`` (``ZeroConfig.h_max_m``).
Two alternative zeros are reported for those lines (diagnostic only, nothing
here changes the depth code):

* ``single``: the single-event rule (first event, ``_air_zero`` with one event).
* ``within_hmax``: the latest event whose gap to the first event implies
  ``h <= h_max``; if only the first event remains, the single-event zero.

The implied depth change of every candidate on the line is
``v_meta * (t_zero - t_zero_alt) / 2`` (positive = deeper) with
``v_meta = c / sqrt(eps_r_used)`` (the candidate depth itself is not read).

Ground truth: only *membership* of an activity in the main ground-truth set
(``exclude_x``, as in the ``--dry-run`` counts) is used, to tell whether a line
can influence the 4TU metrics. No depth value is read or printed.

Outputs (docs/processing/):
* ``m29e_zero_survey_lines.csv``: one row per line.
* ``m29e_zero_survey_sites.csv``: per-site counts.
* ``m29e_zero_survey.json``: totals and run conditions.

Run (single process):
    conda run -n gpr-dev python -I analysis/processing/m29e_zero_survey.py
"""

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gpr_detection.depth.config import DEFAULT_DEPTH_CONFIG
from gpr_detection.depth.estimate import prepare_depth_inputs
from gpr_detection.io.fourtu import T0Choice, load_fourtu_radargram
from gpr_detection.io.paths import load_dataset_paths
from gpr_detection.preprocess.timezero import detect_zero_events

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "processing"
LINES_CSV = DOCS / "fourtu_lines_v1.csv"
CANDIDATES_CSV = DOCS / "fourtu_apex_candidates_v1.csv"
EVAL_SCRIPT = ROOT / "scripts" / "eval_depth_4tu.py"
C_M_PER_NS = 0.299792458
H_MAX_M = DEFAULT_DEPTH_CONFIG.h_max_m


def _gt_activities() -> frozenset[str]:
    """Activities in the main ground-truth set (membership only, no depth)."""
    spec = importlib.util.spec_from_file_location("eval_depth_4tu", EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_depth_4tu"] = module
    spec.loader.exec_module(module)
    gt_sets = module.load_ground_truth_sets(load_dataset_paths().fourtu)
    return frozenset(gt_sets["exclude_x"])


def _height_m(gap_ns: float) -> float:
    return C_M_PER_NS * gap_ns / 2.0


def survey_line(root: Path, activity: str, path: str) -> dict[str, object]:
    radargram = load_fourtu_radargram(root, activity, path, T0Choice.BOTH)
    config = DEFAULT_DEPTH_CONFIG
    inputs = prepare_depth_inputs(radargram, config)
    zero = inputs.zero
    events = detect_zero_events(
        inputs.branch.filtered,
        radargram.dt_ns,
        radargram.f_c_hz,
        pick_mode=config.pick_mode,
        antenna_offset_m=radargram.antenna_offset_m,
        config=config.zero_config(),
    )
    events_ns = list(events.events_ns)
    if [round(t, 9) for t in events_ns] != [round(t, 9) for t in zero.events_ns]:
        raise RuntimeError(f"{activity}/{path}: re-detected events differ")
    amplitudes = np.asarray(events.amplitudes, dtype=float)
    rel = amplitudes / amplitudes.max() if amplitudes.size else amplitudes
    strongest = int(np.argmax(amplitudes)) if amplitudes.size else -1
    v_meta = C_M_PER_NS / math.sqrt(radargram.eps_r_meta)
    row: dict[str, object] = {
        "activity_id": activity,
        "path": path,
        "dt_ns": radargram.dt_ns,
        "eps_r_used": radargram.eps_r_meta,
        "zero_rule": zero.zero_rule,
        "n_events": len(events_ns),
        "events_ns": ";".join(f"{t:.3f}" for t in events_ns),
        "events_rel_amp": ";".join(f"{a:.3f}" for a in rel),
        "strongest_index": strongest,
        "window_end_ns": events.window_end_k * radargram.dt_ns,
        "t_zero_ns": zero.t_zero_ns,
        "h_est_m": zero.h_est_m,
        "h_est_gt_hmax": bool(zero.h_est_m is not None and zero.h_est_m > H_MAX_M),
        "meta_t0_est": "{:.3f}/{}".format(*zero.meta_t0["t0_sample_est"])
        if "t0_sample_est" in zero.meta_t0
        else "",
        "meta_t0_act": "{:.3f}/{}".format(*zero.meta_t0["t0_sample_act_median"])
        if "t0_sample_act_median" in zero.meta_t0
        else "",
        "flags": " ".join(zero.flags),
    }
    if zero.zero_rule == "air_surface_late" and events_ns:
        first = events_ns[0]
        within = [t for t in events_ns if _height_m(t - first) <= H_MAX_M]
        t_single = first
        t_within = within[-1]
        row |= {
            "h_first_to_strongest_m": _height_m(events_ns[strongest] - first),
            "dt_single_ns": t_single - zero.t_zero_ns,
            "dt_within_hmax_ns": t_within - zero.t_zero_ns,
            "within_hmax_rule": "air_surface_late" if len(within) >= 2 else "single",
            "dd_single_m": v_meta * (zero.t_zero_ns - t_single) / 2.0,
            "dd_within_hmax_m": v_meta * (zero.t_zero_ns - t_within) / 2.0,
        }
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--activities", default=None)
    args = parser.parse_args()
    started = time.perf_counter()
    root = load_dataset_paths().fourtu
    lines = pd.read_csv(LINES_CSV, dtype={"activity_id": str, "path": str})
    if args.activities:
        lines = lines[lines["activity_id"].isin(args.activities.split(","))]
    candidates = pd.read_csv(CANDIDATES_CSV, dtype={"activity_id": str, "path": str})
    n_cand = candidates.groupby(["activity_id", "path"]).size().to_dict()
    gt_activities = _gt_activities()

    rows = []
    for record in lines.itertuples(index=False):
        row = survey_line(root, record.activity_id, record.path)
        row["site"] = int(record.site)
        row["activity_has_gt"] = record.activity_id in gt_activities
        row["n_candidates"] = int(n_cand.get((record.activity_id, record.path), 0))
        rows.append(row)
    table = pd.DataFrame(rows)
    front = ["activity_id", "site", "path", "activity_has_gt", "n_candidates"]
    table = table[front + [c for c in table.columns if c not in front]]
    table.to_csv(DOCS / "m29e_zero_survey_lines.csv", index=False, float_format="%.4f")

    late = table[table["zero_rule"] == "air_surface_late"]
    implausible = late[late["h_est_gt_hmax"]]
    per_site = (
        table.groupby("site")
        .agg(
            lines=("path", "size"),
            activities=("activity_id", "nunique"),
            surface_late=("zero_rule", lambda s: int((s == "air_surface_late").sum())),
            h_gt_hmax=("h_est_gt_hmax", "sum"),
        )
        .reset_index()
    )
    per_site["h_gt_hmax_with_gt"] = per_site["site"].map(
        implausible[implausible["activity_has_gt"]].groupby("site").size()
    ).fillna(0).astype(int)
    per_site["candidates_on_h_gt_hmax"] = per_site["site"].map(
        implausible.groupby("site")["n_candidates"].sum()
    ).fillna(0).astype(int)
    with_gt = implausible[implausible["activity_has_gt"]]
    per_site["candidates_on_h_gt_hmax_with_gt"] = per_site["site"].map(
        with_gt.groupby("site")["n_candidates"].sum()
    ).fillna(0).astype(int)
    per_site.to_csv(DOCS / "m29e_zero_survey_sites.csv", index=False)

    def quantiles(values: pd.Series) -> list[float]:
        values = values.dropna()
        if values.empty:
            return []
        return np.percentile(values, [0, 25, 50, 75, 100]).round(3).tolist()

    summary = {
        "lines": len(table),
        "zero_rules": table["zero_rule"].value_counts().to_dict(),
        "surface_late_lines": len(late),
        "h_gt_hmax_lines": len(implausible),
        "h_gt_hmax_lines_with_gt": int(implausible["activity_has_gt"].sum()),
        "candidates_on_h_gt_hmax": int(implausible["n_candidates"].sum()),
        "candidates_on_h_gt_hmax_with_gt": int(
            implausible.loc[implausible["activity_has_gt"], "n_candidates"].sum()
        ),
        "h_est_quantiles_surface_late": quantiles(late["h_est_m"]),
        "h_est_quantiles_implausible": quantiles(implausible["h_est_m"]),
        "n_events_implausible": implausible["n_events"].value_counts().to_dict(),
        "strongest_index_implausible": implausible["strongest_index"]
        .value_counts()
        .to_dict(),
        "within_hmax_rule_implausible": implausible["within_hmax_rule"]
        .value_counts()
        .to_dict(),
        "dd_single_m_quantiles_implausible": quantiles(implausible["dd_single_m"]),
        "dd_within_hmax_m_quantiles_implausible": quantiles(
            implausible["dd_within_hmax_m"]
        ),
        "dd_single_m_quantiles_plausible_late": quantiles(
            late.loc[~late["h_est_gt_hmax"], "dd_single_m"]
        ),
        "h_max_m": H_MAX_M,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "gt_use": "activity membership in exclude_x only (no depth read)",
    }
    (DOCS / "m29e_zero_survey.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(per_site.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
