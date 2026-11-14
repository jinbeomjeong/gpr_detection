"""M3-2 S4 review (exploratory, read only): period check on the whole patch vs on the apex window.

NOTE (2026-11-14): exploration only, inconclusive. The apex window used here (+-2 traces x -1 P .. +3 P) is NOT
the adopted region. The plan (m32_image_quality_plan.md v0.2.3, continuations 5-6) adopted the [EMR] B-2
first-reflection region (inside the -20 dB wing component, apex - 2 P .. apex + tau_m - 1 P). The short window
here often returns None because of the estimator's n/4 period limit.

Same lines and responses as m32_inject_stratum_check.py (k 1.3 and 0.8). For each patch the waveform-replaced
tapered field is measured with measure_period_samples (a) over the whole patch (current S4 check) and (b) over the
apex window only: apex column +- 2 traces, apex sample -1 P .. +3 P. Reports the ratio to P_line and how many
patches pass +-5 % under each. Output: analysis/processing/m32_period_check_window.json.
Run: conda run -n gpr-dev python -I analysis/processing/m32_period_check_window.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis" / "detection"))
sys.path.insert(0, str(ROOT / "analysis" / "processing"))

import m32_inject_stratum_check as base  # noqa: E402
import m32_pipe_responses as responses_mod  # noqa: E402
from gpr_detection.eval import inject  # noqa: E402
from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.guangzhou import load_guangzhou_radargram  # noqa: E402
from gpr_detection.io.paths import DatasetName, load_dataset_paths  # noqa: E402
from gpr_detection.preprocess.to_image import measure_period_samples  # noqa: E402

OUT = ROOT / "analysis" / "processing" / "m32_period_check_window.json"


def main() -> None:
    paths = load_dataset_paths()
    lib = responses_mod.load_library()
    summary = defaultdict(lambda: defaultdict(int))
    ratios = defaultdict(list)
    for ds, fid in base.pick_lines():
        if ds == "4tu":
            act, path = fid.split("/")
            rg = load_fourtu_radargram(paths.root(DatasetName.FOURTU), act, path)
            pool = lib.air
        else:
            rg = load_guangzhou_radargram(paths.root(DatasetName.GUANGZHOU), fid)
            pool = lib.ground
        ctx = inject.prepare_line_context(rg, inject.NO_EXCLUSIONS)
        p = ctx.period_samples
        for rid, resp in pool.items():
            for k in (1.3, 0.8):
                tapered, _, _ = inject._filtered_fields(resp, ctx, k, True)
                whole = measure_period_samples(tapered, pad_factor=8, min_period=4.0)
                shift = inject._alignment_shift_ns(resp, ctx)
                apex = (resp.apex_time_ns + shift) / ctx.dt_ns
                col = round(resp.apex_trace)
                r0, r1 = max(0, math.floor(apex - p)), min(tapered.shape[0], math.ceil(apex + 3 * p))
                c0, c1 = max(0, col - 2), min(tapered.shape[1], col + 3)
                win = tapered[r0:r1, c0:c1] if r1 - r0 >= 16 else None
                local = None if win is None else measure_period_samples(win, pad_factor=8, min_period=4.0)
                key = f"{ds}/{lib.groups[rid].value}/k{k}"
                summary[key]["n"] += 1
                for name, val in (("whole", whole), ("apex", local)):
                    ok = val is not None and abs(val / p - 1) <= 0.05
                    summary[key][f"{name}_pass"] += int(ok)
                    if val is not None:
                        ratios[f"{key}/{name}"].append(val / p)
        print(f"{ds} {fid} done", flush=True)
    rstat = {k: [round(float(np.min(v)), 3), round(float(np.median(v)), 3), round(float(np.max(v)), 3)]
             for k, v in ratios.items()}
    OUT.write_text(json.dumps({"summary": summary, "ratio_min_med_max": rstat}, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(OUT)}))


if __name__ == "__main__":
    main()
