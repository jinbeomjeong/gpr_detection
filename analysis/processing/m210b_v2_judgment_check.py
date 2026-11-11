"""M2-10 (b) v2: read-only checks for writing the judgment report (no judgment here).

Prints the judgment json items the report quotes, re-derives a few of them from
runs_v2.csv / run_info.json (independent recount, already-seen v2 results), and
gives the F3-failure breakdown used for the physical description. Nothing is written
except docs/processing/m210b_v2_judgment_check.json.

Run: conda run -n gpr-dev python -I analysis/processing/m210b_v2_judgment_check.py
"""

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data" / "processing" / "m210b_s4s6_v2"
J = json.loads((ROOT / "docs" / "processing" / "m210b_v2_judgment.json").read_text(encoding="utf-8"))
INFO = json.loads((RUN / "run_info.json").read_text(encoding="utf-8"))
OUT = ROOT / "docs" / "processing" / "m210b_v2_judgment_check.json"


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=True).stdout.strip()


def main():
    out = {"note": "read-only recount for the v2 judgment report (already-seen v2 results)"}
    runs = pd.read_csv(RUN / "runs_v2.csv", dtype=str, keep_default_na=False)
    out["runs_sha256"] = hashlib.sha256((RUN / "runs_v2.csv").read_bytes()).hexdigest()
    out["runs_sha256_info"] = INFO.get("runs_csv_sha256") or INFO.get("runs_v2_csv_sha256")
    out["n_rows"] = len(runs)
    out["judge_blob_now"] = git("hash-object", "analysis/processing/m210b_v2_judge.py")
    out["judge_blob_seal"] = git("rev-parse", "752e783:analysis/processing/m210b_v2_judge.py")
    out["spec_blob_seal"] = git("rev-parse", "752e783:docs/processing/m210b_selector_floor_v2.md")
    out["spec_blob_now"] = git("hash-object", "docs/processing/m210b_selector_floor_v2.md")
    out["v14_blob_seal"] = git("rev-parse", "752e783:docs/processing/m210b_selector_floor.md")
    out["interp_blob_seal"] = git("rev-parse", "752e783:docs/processing/m210b_harness_interpretation.md")
    out["seal_commit_time"] = git("show", "-s", "--format=%cI", "752e783")
    # R table: main (self) vs P-F3 (cal)
    out["R_table_main"] = [
        {k: r.get(k) for k in ("model", "rho", "delta_e", "delta_l", "rate_e", "rate_l", "e", "Q", "rho_min",
                               "Lambda", "Lambda_l", "floor_share_e", "G", "G_fail_reasons", "control_gap", "C", "R_pair")}
        for r in J["R_table"]
    ]
    out["R_table_PF3"] = [
        {k: r.get(k) for k in ("model", "rho", "delta_e", "delta_l", "e", "Lambda", "Lambda_l", "G", "R_pair")}
        for r in J["P_F3"]["R_table"]
    ] if "R_table" in J["P_F3"] else None
    out["G_excluded_no_escape"] = J["G_excluded_no_escape"]
    out["harness_Lambda_self"] = {m: v["harness"].get("Lambda") for m, v in INFO["models"].items()}
    # judgment rows
    R = runs[(runs["dt_ns"].astype(float).sub(0.098).abs() < 1e-9) & (runs["ramp_noise"] == "0")]
    pipe = R[R["model_id"] != "S6P_NP_S040_J2"]
    out["pipe_blank_accepted_floor"] = {c: int((pipe[c] == "").sum()) for c in ("accepted_apex", "floor_held_at_apex")}
    # F3 breakdown: S4N_D15_S045 rho 20 delta 0, per rule
    g = R[(R["model_id"] == "S4N_D15_S045") & (R["rho"] == "20") & (R["delta_ns"].astype(float) == 0.0)]
    br = {}
    for u, gg in g.groupby("rule"):
        br[u] = {
            "n": len(gg),
            "accepted_rate": round(float((gg["accepted_apex"] == "True").mean()), 3),
            "peak_on_crest_rate": round(float((gg["peak_on_crest"] == "True").mean()), 3),
            "kept_rate": round(float((gg["kept_apex"] == "True").mean()), 3),
            "lost_reason": gg["lost_reason"].value_counts().to_dict(),
            "F_median": float(pd.to_numeric(gg["F"]).median()),
            "L_median": float(pd.to_numeric(gg["L"]).median()),
            "sigma_n_median": float(pd.to_numeric(gg["sigma_n"]).median()),
            "n_peaks_median": float(pd.to_numeric(gg["n_peaks"]).median()),
            "floor_held_rate": round(float((gg["floor_held_at_apex"] == "True").mean()), 3),
            "rho_real_median": float(pd.to_numeric(gg["rho_real"]).median()),
        }
    out["F3_S4N_D15_S045_rho20_d0"] = br
    # same for rho 10 and all deltas, rates only
    out["S4N_D15_S045_rates"] = {
        f"{u}|{rho}|{d}": round(float((gg["accepted_apex"] == "True").mean()), 3)
        for (u, rho, d), gg in R[R["model_id"] == "S4N_D15_S045"].groupby(["rule", "rho", "delta_ns"])
    }
    m = INFO["models"].get("S4N_D15_S045", {}).get("harness", {})
    out["S4N_D15_S045_harness"] = {k: m.get(k) for k in ("t_a0_ns", "A0_self", "apex_trace", "t_zero_ns")}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    main()
