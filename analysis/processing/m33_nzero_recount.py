"""Read-only recount of the R3 ``zero_changed`` / ``period_changed`` report.

M3-3 item 1 (2026-11-20): the R3 injected summary counted every realization
(2,159) as ``n_zero_changed``. This script rebuilds each injected record with
the sealed C1b table (same functions as ``m33_r3_band.evaluate_band_line``,
no inference, no PP) and writes the signed zero and period differences, so a
tolerance-based definition can be proposed.

Per line it also runs two null tests:

* ``null_same_path``: the background run through the exact code path of
  ``zero_changed`` (float32 cast, ``inject.linear_steps``) -> difference from
  ``context.zero``. Nonzero would mean the comparison itself is biased.
* ``null_line_zero``: ``context.zero`` vs ``runner.load_line(...).zero``.

Outputs (CSV, one row per realization / line) go to ``--out-dir``. Inputs
(``runs/eval/m33/r3/plan``, ``.../injected/parts``) are only read.

Run (gpr-dev, one process):
    conda run -n gpr-dev python -I -B analysis/processing/m33_nzero_recount.py \
        --start 0 --stop 40 --out-dir analysis/processing/m33_nzero
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "detection"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

import m32_pipe_responses as responses_io  # noqa: E402
import m32_recipe_eval as runner  # noqa: E402
import m32_recipe_inputs as inputs  # noqa: E402
import m33_r3_band as band  # noqa: E402
from gpr_detection.eval import inject  # noqa: E402
from gpr_detection.preprocess.timezero import find_time_zero  # noqa: E402
from gpr_detection.preprocess.to_image import make_detection_image  # noqa: E402

R3 = REPO / "runs" / "eval" / "m33" / "r3"
PLAN = R3 / "plan"
PARTS = R3 / "injected" / "parts"

REAL_COLUMNS = [
    "line_id", "dataset", "ids_freq_stratum", "realization_id", "snr_db", "polarity",
    "dt_ns", "f_c_hz", "period_bg_samples", "k_zero_bg", "rule_bg", "n_events_bg",
    "k_zero_inj", "rule_inj", "n_events_inj", "events_k_bg", "events_k_inj",
    "dk_zero_samples", "dk_zero_ns", "dk_zero_over_p", "rule_changed",
    "event_count_changed", "min_apex_minus_zero_over_p", "period_inj_samples",
    "dp_samples", "dp_rel", "stored_zero_changed", "stored_period_changed",
    "exact_zero_changed", "exact_period_changed",
]
LINE_COLUMNS = [
    "line_id", "dataset", "n_traces", "n_samples", "dt_ns", "f_c_hz", "period_bg_samples",
    "k_zero_bg", "rule_bg", "n_events_bg", "null_same_path_dk", "null_line_zero_dk",
    "null_period_dp", "seconds",
]


def _f(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def _stored_rows() -> dict[str, dict]:
    stored = {}
    for path in sorted(PARTS.glob("*.json")):
        part = json.loads(path.read_text(encoding="utf-8"))
        for row in part["realizations"]:
            stored[row["realization_id"]] = row
    return stored


def _line_refs() -> list[inputs.LineRef]:
    strata = band.ids_freq_strata()
    refs = [r for r in inputs.background_lines() if r.dataset == inputs.DATASET_4TU]
    refs += inputs.k_lines()
    refs += [r for r in band._ids_lines() if strata.get(r.line_id) != band.FREQ_EXCLUDED]
    return refs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--no-period", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    strata = band.ids_freq_strata()
    specs, _ = inject.read_injection_table(PLAN / "injection_table.csv")
    by_line = runner.group_specs(specs)
    exclusions = inputs.read_exclusions(PLAN / "exclusions.csv")
    candidates = inputs.candidate_apexes()
    library = responses_io.load_library()
    stored = _stored_rows()
    refs = {ref.line_id: ref for ref in _line_refs()}
    lines = [refs[line_id] for line_id in by_line if line_id in refs]
    chosen = lines[args.start : args.stop]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.start:03d}_{(args.stop if args.stop is not None else len(lines)):03d}"
    real_path = args.out_dir / f"realizations_{tag}.csv"
    line_path = args.out_dir / f"lines_{tag}.csv"
    n_real = 0
    with real_path.open("w", newline="", encoding="utf-8") as real_file, line_path.open(
        "w", newline="", encoding="utf-8"
    ) as line_file:
        real_writer = csv.DictWriter(real_file, REAL_COLUMNS, lineterminator="\n")
        line_writer = csv.DictWriter(line_file, LINE_COLUMNS, lineterminator="\n")
        real_writer.writeheader()
        line_writer.writeheader()
        for ref in chosen:
            t_line = time.perf_counter()
            line = runner.load_line(ref)
            rg = line.radargram
            line_excl = runner.line_exclusions(ref.line_id, exclusions, candidates)
            context = inject.prepare_line_context(rg, line_excl)
            zero_bg = context.zero
            period_bg = context.period_samples
            # null test 1: background through the zero_changed code path
            same = replace(rg, data=np.asarray(rg.data).astype(np.float32))
            z_same = find_time_zero(same, inject.linear_steps(same.data, same))
            null_same = (
                None
                if z_same.k_zero is None or zero_bg.k_zero is None
                else z_same.k_zero - zero_bg.k_zero
            )
            null_line = (
                None
                if line.zero.k_zero is None or zero_bg.k_zero is None
                else line.zero.k_zero - zero_bg.k_zero
            )
            null_dp = None
            if not args.no_period:
                img0 = make_detection_image(same, inputs.recipe_config(inputs.R0))
                null_dp = img0.period_samples - period_bg
            responses = library.for_line(ref.dataset, runner.line_time_window_ns(line))
            for realization_id, rspecs in by_line[ref.line_id].items():
                result = inject.inject_band_realization(context, rspecs, responses)
                if not result.truths:
                    continue
                injected = replace(rg, data=result.data.astype(np.float32))
                z_inj = find_time_zero(injected, inject.linear_steps(injected.data, injected))
                dk = (
                    None
                    if z_inj.k_zero is None or zero_bg.k_zero is None
                    else z_inj.k_zero - zero_bg.k_zero
                )
                p_inj = dp = None
                if not args.no_period:
                    image = make_detection_image(injected, inputs.recipe_config(inputs.R0))
                    p_inj = image.period_samples
                    dp = p_inj - period_bg
                apex_gap = None
                if zero_bg.k_zero is not None:
                    apex_gap = min(
                        (t.apex_sample - zero_bg.k_zero) / period_bg for t in result.truths
                    )
                row_stored = stored.get(realization_id, {})
                spec0 = rspecs[0]
                real_writer.writerow(
                    {
                        "line_id": ref.line_id,
                        "dataset": ref.dataset,
                        "ids_freq_stratum": strata.get(ref.line_id, ""),
                        "realization_id": realization_id,
                        "snr_db": spec0.snr_db,
                        "polarity": spec0.polarity,
                        "dt_ns": _f(rg.dt_ns),
                        "f_c_hz": _f(rg.f_c_hz),
                        "period_bg_samples": _f(period_bg),
                        "k_zero_bg": _f(zero_bg.k_zero),
                        "rule_bg": str(zero_bg.zero_rule),
                        "n_events_bg": len(zero_bg.events_ns),
                        "k_zero_inj": _f(z_inj.k_zero),
                        "rule_inj": str(z_inj.zero_rule),
                        "n_events_inj": len(z_inj.events_ns),
                        "events_k_bg": ";".join(
                            f"{e / rg.dt_ns:.4f}" for e in zero_bg.events_ns
                        ),
                        "events_k_inj": ";".join(
                            f"{e / rg.dt_ns:.4f}" for e in z_inj.events_ns
                        ),
                        "dk_zero_samples": _f(dk),
                        "dk_zero_ns": _f(None if dk is None else dk * rg.dt_ns),
                        "dk_zero_over_p": _f(None if dk is None else dk / period_bg),
                        "rule_changed": z_inj.zero_rule != zero_bg.zero_rule,
                        "event_count_changed": len(z_inj.events_ns) != len(zero_bg.events_ns),
                        "min_apex_minus_zero_over_p": _f(apex_gap),
                        "period_inj_samples": _f(p_inj),
                        "dp_samples": _f(dp),
                        "dp_rel": _f(None if dp is None else dp / period_bg),
                        "stored_zero_changed": row_stored.get("zero_changed", ""),
                        "stored_period_changed": row_stored.get("period_changed", ""),
                        "exact_zero_changed": z_inj.k_zero != zero_bg.k_zero,
                        "exact_period_changed": ""
                        if p_inj is None
                        else p_inj != period_bg,
                    }
                )
                n_real += 1
            line_writer.writerow(
                {
                    "line_id": ref.line_id,
                    "dataset": ref.dataset,
                    "n_traces": rg.n_traces,
                    "n_samples": rg.n_samples,
                    "dt_ns": _f(rg.dt_ns),
                    "f_c_hz": _f(rg.f_c_hz),
                    "period_bg_samples": _f(period_bg),
                    "k_zero_bg": _f(zero_bg.k_zero),
                    "rule_bg": str(zero_bg.zero_rule),
                    "n_events_bg": len(zero_bg.events_ns),
                    "null_same_path_dk": _f(null_same),
                    "null_line_zero_dk": _f(null_line),
                    "null_period_dp": _f(null_dp),
                    "seconds": f"{time.perf_counter() - t_line:.2f}",
                }
            )
            real_file.flush()
            line_file.flush()
    print(
        json.dumps(
            {
                "status": "ok",
                "n_lines": len(chosen),
                "n_lines_total": len(lines),
                "n_realizations": n_real,
                "seconds": round(time.perf_counter() - started, 1),
                "out": str(real_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
