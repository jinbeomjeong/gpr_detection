"""M3-2 S4/S5 review (exploratory, read only, not a verdict): why the injection stratum is short.

For a few stage A background lines (4TU dt 0.098 x3, dt 0.195 x1, Guangzhou IDS 400 MHz x2, first lines of the
lists in order), builds every response patch with the S4 code (eval/inject.py, no exclusions) for k 1.3 and 0.8 and
records: reject reason; for period-check failures the measured / line period ratio; for built patches the L1 size in
P of the -20 dB wing box (current stratum), of the -6 dB core box, and of -10 / -12 dB boxes of the same
(untapered) envelope; whether the -20 dB wing reaches the synthetic aperture edge (aperture-limited).
AIHub pipe_hyperbola label p5-p95: width 7.79-14.67 P, height 5.22-8.35 P (m32_s1_recheck.json).
Output: analysis/processing/m32_inject_stratum_check.json.
Run: conda run -n gpr-dev python -I analysis/processing/m32_inject_stratum_check.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis" / "detection"))

import m32_pipe_responses as responses_mod  # noqa: E402
from gpr_detection.eval import inject  # noqa: E402
from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.guangzhou import load_guangzhou_radargram  # noqa: E402
from gpr_detection.io.paths import DatasetName, load_dataset_paths  # noqa: E402
from gpr_detection.preprocess.to_image import measure_period_samples  # noqa: E402

HERE = ROOT / "analysis" / "processing"
OUT = HERE / "m32_inject_stratum_check.json"
LEVELS = {"m20": 0.1, "m12": 10 ** (-12 / 20), "m10": 10 ** (-10 / 20), "m6": 0.5}
W_RANGE, H_RANGE = (7.79, 14.67), (5.22, 8.35)


def pick_lines():
    l4 = list(csv.DictReader(open(HERE / "m32_stageA_lines_4tu.csv", encoding="utf-8")))
    li = list(csv.DictReader(open(HERE / "m32_stageA_lines_ids.csv", encoding="utf-8")))
    a = [r["line"] for r in l4 if r["dt_ns"] in ("0.098",)][:3]
    b = [r["line"] for r in l4 if r["dt_ns"] == "0.195"][:1]
    c = [r["line"] for r in li if r["antenna_MHz"] == "400.0"][:2]
    return [("4tu", x) for x in a + b] + [("gz_ids", x) for x in c]


def size_p(bbox, context):
    col0, col1, row0, row1 = bbox
    gm = context.grid_map
    p_l1 = context.period_samples / gm.samples_per_px
    return ((col1 - col0 + 1) / gm.traces_per_px / p_l1, (row1 - row0 + 1) / gm.samples_per_px / p_l1)


def main() -> None:
    paths = load_dataset_paths()
    lib = responses_mod.load_library()
    rows = []
    for ds, fid in pick_lines():
        if ds == "4tu":
            act, path = fid.split("/")
            rg = load_fourtu_radargram(paths.root(DatasetName.FOURTU), act, path)
            pool = lib.air
        else:
            rg = load_guangzhou_radargram(paths.root(DatasetName.GUANGZHOU), fid)
            pool = lib.ground
        ctx = inject.prepare_line_context(rg, inject.NO_EXCLUSIONS)
        for rid, resp in pool.items():
            for k in (1.3, 0.8):
                rec = {"dataset": ds, "line": fid, "response": rid, "group": lib.groups[rid].value, "k": k,
                       "P_line": ctx.period_samples}
                patch = inject.build_response_patch(resp, ctx, k)
                tapered, untapered, ntap = inject._filtered_fields(resp, ctx, k, True)
                meas = measure_period_samples(tapered, pad_factor=ctx.image_config.period_pad_factor,
                                              min_period=ctx.image_config.period_min_samples)
                rec["period_ratio"] = None if meas is None else meas / ctx.period_samples
                if isinstance(patch, inject.RejectReason):
                    rec["reason"] = patch.value
                    rows.append(rec)
                    continue
                rec["reason"] = "ok"
                env = inject._envelope(inject._linear_steps(untapered, rg))
                peak = inject._apex_peak(env, round(resp.apex_trace), patch.apex_sample, ctx.period_samples)
                for name, lev in LEVELS.items():
                    mask = inject._seed_component(env >= lev * peak.amplitude, peak.seed)
                    bb = inject._bbox(mask)
                    w, h = size_p(bb, ctx)
                    rec[f"{name}_W"], rec[f"{name}_H"] = round(w, 2), round(h, 2)
                    rec[f"{name}_edge"] = bool(bb[0] == 0 or bb[1] == env.shape[1] - 1)
                    rec[f"{name}_in"] = bool(W_RANGE[0] <= w <= W_RANGE[1] and H_RANGE[0] <= h <= H_RANGE[1])
                rows.append(rec)
        print(f"{ds} {fid} done", flush=True)
    summary = defaultdict(lambda: defaultdict(int))
    for r in rows:
        key = f"{r['dataset']}/{r['group']}/k{r['k']}"
        summary[key]["n"] += 1
        summary[key][r["reason"]] += 1
        if r["reason"] == "ok":
            for name in LEVELS:
                summary[key][f"{name}_in"] += int(r[f"{name}_in"])
                summary[key][f"{name}_edge"] += int(r[f"{name}_edge"])
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "n": len(rows)}))


if __name__ == "__main__":
    main()
