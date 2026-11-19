"""Summarize ``m33_nzero_recount.py`` outputs (read-only).

Reads every ``realizations_*.csv`` / ``lines_*.csv`` in ``--in-dir`` and writes
``summary.json``: |dk_zero| quantiles (samples, ns, P), counts above candidate
tolerances, rule / event-count changes, period bin hops and relative changes,
null-test results and the cross-check against the stored R3 flags.

Run (gpr-dev):
    conda run -n gpr-dev python -I -B analysis/processing/m33_nzero_summary.py \
        --in-dir analysis/processing/m33_nzero
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ZERO_TOLS_P = (1e-6, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5)
ZERO_TOLS_NS = (0.01, 0.05, 0.1, 0.2, 0.25)
PERIOD_TOLS_REL = (0.0, 0.01, 0.02, 0.05, 0.1)
QUANTILES = (0.0, 0.5, 0.9, 0.99, 1.0)
_LABEL = {  # zero_event_label of each rule (timezero.select_zero)
    "air_single_event": "merged",
    "air_first_event": "merged",
    "air_surface_late": "surface_reflection",
    "ground_direct_peak": "direct_wave",
    "ground_direct_shifted": "direct_wave",
    "meta_sample0": "meta",
    "zero_failed": "failed",
}


def _rows(in_dir: Path, prefix: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(in_dir.glob(f"{prefix}_*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows += list(csv.DictReader(handle))
    return rows


def _num(value: str) -> float:
    return float("nan") if value in ("", None) else float(value)


def _quant(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {f"q{q:g}": float(np.quantile(values, q)) for q in QUANTILES}


def _by_group(rows: list[dict], key: str) -> dict:
    out = {}
    for name in sorted({r[key] for r in rows}):
        sub = [r for r in rows if r[key] == name]
        adk = np.array([abs(_num(r["dk_zero_over_p"])) for r in sub])
        adp = np.array([abs(_num(r["dp_rel"])) for r in sub])
        out[name or "(none)"] = {
            "n": len(sub),
            "n_period_exact": int(np.sum(adp > 0)),
            "n_period_gt_5pct": int(np.sum(adp > 0.05)),
            "n_period_gt_10pct": int(np.sum(adp > 0.1)),
            "abs_dp_rel_max": float(np.nanmax(adp)) if adp.size else None,
            "abs_dk_over_p": _quant(adk),
            "n_gt_0.1P": int(np.sum(adk > 0.1)),
            "n_rule_or_events_changed": sum(
                r["rule_changed"] == "True" or r["event_count_changed"] == "True"
                for r in sub
            ),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    reals = _rows(args.in_dir, "realizations")
    lines = _rows(args.in_dir, "lines")

    dk_s = np.array([_num(r["dk_zero_samples"]) for r in reals])
    dk_ns = np.array([_num(r["dk_zero_ns"]) for r in reals])
    dk_p = np.array([_num(r["dk_zero_over_p"]) for r in reals])
    dp_rel = np.array([_num(r["dp_rel"]) for r in reals])
    dp_s = np.array([_num(r["dp_samples"]) for r in reals])
    rule_changed = np.array([r["rule_changed"] == "True" for r in reals])
    events_changed = np.array([r["event_count_changed"] == "True" for r in reals])
    structural = rule_changed | events_changed
    label_changed = np.array(
        [_LABEL.get(r["rule_bg"], r["rule_bg"]) != _LABEL.get(r["rule_inj"], r["rule_inj"])
         for r in reals]
    )

    stored_zero = Counter(r["stored_zero_changed"] for r in reals)
    agree_zero = sum(r["stored_zero_changed"] == r["exact_zero_changed"] for r in reals)
    agree_period = sum(
        r["stored_period_changed"] == r["exact_period_changed"]
        for r in reals
        if r["exact_period_changed"] != ""
    )
    n_period_known = sum(r["exact_period_changed"] != "" for r in reals)

    null_same = np.array([_num(r["null_same_path_dk"]) for r in lines])
    null_line = np.array([_num(r["null_line_zero_dk"]) for r in lines])
    null_dp = np.array([_num(r["null_period_dp"]) for r in lines])

    finite_p = np.isfinite(dk_p)
    summary = {
        "n_lines": len(lines),
        "n_realizations": len(reals),
        "n_zero_missing": int(np.sum(~finite_p)),
        "cross_check": {
            "stored_zero_changed_values": dict(stored_zero),
            "recount_exact_zero_matches_stored": agree_zero,
            "recount_exact_period_matches_stored": agree_period,
            "n_period_recounted": n_period_known,
        },
        "null_tests": {
            "same_path_dk_nonzero_lines": int(np.sum(np.abs(null_same) > 0)),
            "same_path_dk_max_abs": float(np.nanmax(np.abs(null_same))) if len(lines) else None,
            "line_zero_dk_nonzero_lines": int(np.sum(np.abs(null_line) > 0)),
            "period_dp_nonzero_lines": int(np.sum(np.abs(null_dp[np.isfinite(null_dp)]) > 0)),
        },
        "zero": {
            "abs_dk_samples": _quant(np.abs(dk_s)),
            "abs_dk_ns": _quant(np.abs(dk_ns)),
            "abs_dk_over_p": _quant(np.abs(dk_p)),
            "n_exact_nonzero": int(np.sum(dk_s != 0)),
            "n_gt_tol_p": {f"{t:g}": int(np.sum(np.abs(dk_p) > t)) for t in ZERO_TOLS_P},
            "n_gt_tol_ns": {f"{t:g}": int(np.sum(np.abs(dk_ns) > t)) for t in ZERO_TOLS_NS},
            "n_gt_1_sample": int(np.sum(np.abs(dk_s) > 1.0)),
            "n_rule_changed": int(rule_changed.sum()),
            "n_event_count_changed": int(events_changed.sum()),
            "n_structural": int(structural.sum()),
            "n_proposed_0.1P_or_structural": int(
                np.sum(structural | (np.abs(dk_p) > 0.1))
            ),
            "n_proposed_0.1P_or_rule": int(np.sum(rule_changed | (np.abs(dk_p) > 0.1))),
            "n_label_or_failed_changed": int(label_changed.sum()),
            "n_proposed_final": int(np.sum(label_changed | (np.abs(dk_p) > 0.1))),
            "n_events_changed_by_rule_bg": dict(
                Counter(r["rule_bg"] for r in reals if r["event_count_changed"] == "True")
            ),
            "events_changed_abs_dk_over_p_max": float(
                np.max(np.abs(dk_p[events_changed]))
            )
            if events_changed.any()
            else None,
            "rule_pairs": dict(
                Counter(
                    f"{r['rule_bg']}->{r['rule_inj']}|{r['n_events_bg']}->{r['n_events_inj']}"
                    for r in reals
                    if r["rule_changed"] == "True" or r["event_count_changed"] == "True"
                )
            ),
        },
        "period": {
            "abs_dp_samples": _quant(np.abs(dp_s)),
            "abs_dp_rel": _quant(np.abs(dp_rel)),
            "n_gt_tol_rel": {
                f"{t:g}": int(np.sum(np.abs(dp_rel[np.isfinite(dp_rel)]) > t))
                for t in PERIOD_TOLS_REL
            },
            "min_nonzero_abs_dp_rel": float(
                np.min(np.abs(dp_rel[np.isfinite(dp_rel) & (dp_rel != 0)]))
            )
            if np.any(np.isfinite(dp_rel) & (dp_rel != 0))
            else None,
        },
        "tau_0.1P_ns_by_dataset": {
            name: _quant(
                np.array(
                    [
                        0.1 * _num(r["period_bg_samples"]) * _num(r["dt_ns"])
                        for r in lines
                        if r["dataset"] == name
                    ]
                )
            )
            for name in sorted({r["dataset"] for r in lines})
        },
        "period_bin_rel_by_dataset": {
            # bin spacing dP/P ~ P / n_fft, n_fft = 2^ceil(log2(8 n_s))
            name: _quant(
                np.array(
                    [
                        _num(r["period_bg_samples"])
                        / float(1 << int(np.ceil(np.log2(8 * int(r["n_samples"])))))
                        for r in lines
                        if r["dataset"] == name
                    ]
                )
            )
            for name in sorted({r["dataset"] for r in lines})
        },
        "by_dataset": _by_group(reals, "dataset"),
        "by_snr": _by_group(reals, "snr_db"),
        "by_freq_stratum": _by_group(reals, "ids_freq_stratum"),
        "top_abs_dk_over_p": [
            {
                k: reals[i][k]
                for k in (
                    "realization_id", "dk_zero_samples", "dk_zero_ns", "dk_zero_over_p",
                    "rule_bg", "rule_inj", "n_events_bg", "n_events_inj",
                    "events_k_bg", "events_k_inj", "min_apex_minus_zero_over_p",
                    "period_bg_samples",
                )
            }
            for i in np.argsort(-np.nan_to_num(np.abs(dk_p), nan=-1.0))[:15]
        ],
    }
    out = args.in_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "n_realizations": len(reals), "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
