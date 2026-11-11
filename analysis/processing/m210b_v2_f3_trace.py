"""M2-10 (b) v2: re-run single realisations of the F3-failing condition (DIAGNOSTIC ONLY).

Already-seen v2 results. Reproduces S4N_D15_S045, rho 20, Delta 0 for a few seeds
with the sealed harness functions (imported, unchanged) and prints every detection
peak of current and B_full with its crest-moved point, so the judgment report can
describe WHY the candidate rules lose the apex there. It does not change or recount
any judgment item. Output: docs/processing/m210b_v2_f3_trace.json.

Run: conda run -n gpr-dev python -I analysis/processing/m210b_v2_f3_trace.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "processing"))

import fourtu_apex_candidate_select as generator  # noqa: E402
import m210b_s4s6_pipeline as hp  # noqa: E402

from gpr_detection.depth.estimate import DepthConfig, prepare_depth_inputs  # noqa: E402

MODEL = "S4N_D15_S045"
RHO = 20
SEEDS = (20273220, 20273221, 20273223, 20273224, 20273226, 20273230, 20273231)
RULES = ("current", "B_full")


def main() -> None:
    plan = hp.V2_PLAN
    models = hp.ModelSet(ROOT / "data" / "synthetic_gprmax_s4s6_v2", plan)
    clean = models.clean(MODEL, hp.GRID_HARNESS)
    cal = models.calibration(hp.calibration_pair(MODEL, plan), hp.GRID_HARNESS)
    target = clean.target
    dt = clean.radargram.dt_ns
    out = {"note": "diagnostic only (already-seen v2); harness functions unchanged",
           "model": MODEL, "rho": RHO, "apex_trace": target.trace, "t_a0_ns": target.t_a0_ns,
           "apex_row": round(target.t_a0_ns / dt, 2), "seeds": {}}
    shape = clean.radargram.data.shape
    for seed in SEEDS:
        noise = cal.sigma_raw_by_rho[RHO] * hp.white_noise(shape, seed)
        noisy = hp.with_noise(clean.radargram, noise)
        inputs = prepare_depth_inputs(noisy, DepthConfig())
        rec = {}
        for rule in RULES:
            line, kept, E = generator.apex_candidates_for_radargram(
                noisy, inputs, act=hp.LINE_ACTIVITY, path=MODEL, floor_rule=rule, mute_offset_ns=0.0)
            peaks = []
            for p in line.get("_peaks", []):
                k0 = int(round(p["sample0"]))
                peaks.append({
                    "trace0": p["trace0"], "t0_ns": round(p["sample0"] * dt, 3),
                    "E0": float(E[k0, p["trace0"]]),
                    "trace": p["trace"], "t_ns": round(p["sample"] * dt, 3),
                    "status": p.get("status"),
                    "in_window": bool(hp.is_in_apex_window(p["trace"], p["sample"] * dt, target.trace,
                                                           target.t_a0_ns, noisy.f_c_hz)),
                })
            near = [p for p in peaks if abs(p["trace0"] - target.trace) <= 25 and abs(p["t0_ns"] - target.t_a0_ns) <= 3]
            ka = int(round(target.t_a0_ns / dt))
            rec[rule] = {
                "n_peaks": line.get("n_peaks"), "F": line.get("_floor", {}).get("F"),
                "E_at_apex": float(E[ka, target.trace]),
                "E_max_window": float(E[ka - 10:ka + 11, target.trace - 3:target.trace + 4].max()),
                "peaks_near_apex": near,
                "all_peaks": peaks if rule != "current" else None,
            }
        out["seeds"][str(seed)] = rec
        print(seed, {r: (v["n_peaks"], len(v["peaks_near_apex"])) for r, v in rec.items()}, flush=True)
    p = ROOT / "docs" / "processing" / "m210b_v2_f3_trace.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("written", p)


if __name__ == "__main__":
    main()
