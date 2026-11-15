"""M3-2 stage C pre-check failure: signal-processing diagnosis on recipe r0 only (no detector, read only).

Scope (main session 2026-11-15): r0 only. Reads only runs/eval/m32_recipe/injected/r0/fold0/inj_matches.csv
(hit flags of r0 fold 0) and the plan injection table / exclusions. r5 / r6 results are not opened.

Positions: k 1.3, in_distribution True. Groups
  drop    : 4TU/A0, hit at 12 dB and missed at 18 dB (same line, apex trace, response, polarity);
  keep    : 4TU/A0, hit at both 12 and 18 dB;
  a1      : 4TU/A1 at 18 dB (floor cell), up to 30 positions;
  ids     : IDS/A0 hit at both 12 and 18 dB, up to 30 positions (control, no drop there).
For each position and SNR (12, 18) the realization is rebuilt with the S4 code (same table rows, exclusions,
responses) and the r0 detection branch is redone on L0:
  B0 = prepare_fitting_branch(., 301).background_removed, G0 = agc(., dt, f_c), M0 clip c = p99 |a| of the line.
Measured (target injection only):
  clip_ratio      c(injected) / c(uninjected line);
  sat_core        share of |G0| >= c inside the -6 dB core box (L0) -> pixels at 0 / 255;
  sat_line        share of |G0| >= c over the whole line;
  b0_keep         apex-trace envelope peak of B0(injected) - B0(background), over the unit-field (1)-(3)
                  apex envelope x gain (share of the injected apex that survives the median background);
  wing_over_apex  after G0: envelope at the -6 dB half-width traces (mean of both sides, at their own envelope
                  maximum within the core rows +- 3 P) over the apex-trace envelope; same ratio before G0 (B0 only);
  apex_over_c     apex envelope after G0 divided by c (>1 means the apex is clipped flat).
Output: analysis/processing/m32_stageC_r0_signal_diag.json (aggregates per group and SNR, no per-file names).
Run: conda run -n gpr-dev python -I analysis/processing/m32_stageC_r0_signal_diag.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis" / "detection"))

import m32_pipe_responses as responses_mod  # noqa: E402
import m32_recipe_eval as recipe_eval  # noqa: E402
import m32_recipe_inputs as inputs  # noqa: E402
from gpr_detection.eval import inject  # noqa: E402
from gpr_detection.preprocess.filters import agc, prepare_fitting_branch  # noqa: E402

RUN = ROOT / "runs" / "eval" / "m32_recipe"
MATCHES = RUN / "injected" / "r0" / "fold0" / "inj_matches.csv"
TABLE = RUN / "plan" / "injection_table.csv"
EXCL = RUN / "plan" / "exclusions.csv"
OUT = ROOT / "analysis" / "processing" / "m32_stageC_r0_signal_diag.json"
SEED = 20261114
N_CONTROL = 30


def env(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    pad = np.concatenate([x, np.zeros_like(x)], axis=0)
    return np.abs(hilbert(pad, axis=0))[:n]


def r0_branch(rg) -> tuple[np.ndarray, np.ndarray, float]:
    b0 = prepare_fitting_branch(rg, 301).background_removed
    g0 = agc(b0, rg.dt_ns, rg.f_c_hz)
    return b0, g0, float(np.percentile(np.abs(g0), 99.0))


def select_positions():
    spec_by_id = {s.injection_id: s for s in inject.read_injection_table(TABLE)[0]}
    rows = [r for r in csv.DictReader(open(MATCHES, encoding="utf-8"))
            if r["k"] == "1.3" and r["in_distribution"] == "True"]
    by_pos = defaultdict(dict)
    for r in rows:
        s = spec_by_id[r["injection_id"]]
        key = (s.line_id, s.apex_trace, s.response_id, s.polarity, r["dataset"], r["model_group"])
        by_pos[key][float(r["snr_db"])] = (r["hit"] == "True", s)
    groups = defaultdict(list)
    for key, d in by_pos.items():
        if 12.0 not in d or 18.0 not in d:
            continue
        cell = f"{key[4]}/{key[5]}"
        h12, h18 = d[12.0][0], d[18.0][0]
        if cell == "4tu/A0" and h12 and not h18:
            groups["drop"].append((key, d))
        elif cell == "4tu/A0" and h12 and h18:
            groups["keep"].append((key, d))
        elif cell == "4tu/A1":
            groups["a1"].append((key, d))
        elif cell == "gz_ids/A0" and h12 and h18:
            groups["ids"].append((key, d))
    rng = np.random.default_rng(SEED)
    for g in ("a1", "ids", "keep"):
        if len(groups[g]) > N_CONTROL:
            idx = sorted(rng.choice(len(groups[g]), N_CONTROL, replace=False))
            groups[g] = [groups[g][i] for i in idx]
    return groups, spec_by_id


def measure(line, context, responses, specs_all, spec, base):
    b0_bg, g0_bg, c_bg = base
    realization = [s for s in specs_all if s.realization_id == spec.realization_id]
    result = inject.inject_realization(context, realization, responses)
    truth = next((t for t in result.truths if t.injection_id == spec.injection_id), None)
    if truth is None:
        return None
    rg = line.radargram
    from dataclasses import replace
    inj = replace(rg, data=result.data.astype(np.float32))
    b0, g0, c = r0_branch(inj)
    core = truth.core_box
    k0, k1, i0, i1 = core.k0, core.k1 + 1, core.i0, core.i1 + 1
    sat_core = float(np.mean(np.abs(g0[k0:k1, i0:i1]) >= c))
    sat_line = float(np.mean(np.abs(g0) >= c))
    apex_tr = int(round(truth.apex_trace))
    apex_k = int(round(truth.apex_sample))
    p = context.period_samples
    r0w, r1w = max(0, int(apex_k - p)), min(rg.n_samples, int(apex_k + p) + 1)
    patch_scale = abs(truth.gain) * truth.signal_amplitude
    diff = env(b0[:, apex_tr] - b0_bg[:, apex_tr])
    b0_keep = float(diff[r0w:r1w].max() / patch_scale) if patch_scale > 0 else None
    hw = max(1, int(round((core.i1 - core.i0) / 2)))
    lo_k, hi_k = max(0, int(apex_k - p)), min(rg.n_samples, int(apex_k + 3 * p) + 1)

    def wing_ratio(sec):
        e_apex = env(sec[:, apex_tr])[r0w:r1w].max()
        side = []
        for tr in (apex_tr - hw, apex_tr + hw):
            if 0 <= tr < rg.n_traces:
                side.append(env(sec[:, tr])[lo_k:hi_k].max())
        return float(np.mean(side) / e_apex) if side and e_apex > 0 else None

    apex_g = env(g0[:, apex_tr])[r0w:r1w].max()
    return {
        "clip_ratio": c / c_bg if c_bg > 0 else None,
        "sat_core": sat_core,
        "sat_line": sat_line,
        "b0_keep": b0_keep,
        "wing_over_apex_b0": wing_ratio(b0),
        "wing_over_apex_g0": wing_ratio(g0),
        "apex_over_c": float(apex_g / c),
        "line_traces": rg.n_traces,
        "core_traces": core.i1 - core.i0 + 1,
    }


def summarise(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return None
    return {"p10": round(float(np.percentile(arr, 10)), 3), "p50": round(float(np.median(arr)), 3),
            "p90": round(float(np.percentile(arr, 90)), 3), "n": int(arr.size)}


def main() -> None:
    groups, _ = select_positions()
    specs_all = inject.read_injection_table(TABLE)[0]
    boxes = inputs.read_exclusions(EXCL)
    candidates = inputs.candidate_apexes()
    library = responses_mod.load_library()
    refs = {r.line_id: r for r in inputs.background_lines()}
    cache = {}
    out = {}
    for gname, items in groups.items():
        agg = defaultdict(lambda: defaultdict(list))
        for key, d in sorted(items, key=lambda x: (x[0][0], x[0][1])):
            line_id = key[0]
            if line_id not in cache:
                ref = refs[line_id]
                line = recipe_eval.load_line(ref)
                excl = recipe_eval.line_exclusions(line_id, boxes, candidates)
                context = inject.prepare_line_context(line.radargram, excl)
                cache = {line_id: (line, context, library.for_line(ref.dataset, ref.time_window_ns),
                                   r0_branch(line.radargram))}
            line, context, responses, base = cache[line_id]
            for snr in (12.0, 18.0):
                m = measure(line, context, responses, specs_all, d[snr][1], base)
                if m is None:
                    continue
                for k, v in m.items():
                    agg[snr][k].append(v)
        out[gname] = {"n_positions": len(items),
                      **{f"snr{int(s)}": {k: summarise(v) for k, v in vals.items()} for s, vals in agg.items()}}
        print(f"{gname}: {len(items)} positions", flush=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "groups": {g: v["n_positions"] for g, v in out.items()}}))


if __name__ == "__main__":
    main()
