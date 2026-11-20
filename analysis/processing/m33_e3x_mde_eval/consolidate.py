"""Consolidate E3x plan-structure MDE evaluation into one json + csv.

Usage: consolidate.py <work_dir> <out_json> <out_csv>
"""
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

work, out_json, out_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
REPO = Path("C:/workspace/gpr_detection")
PLAN = REPO / "runs/eval/m33/e3x/plan"


def sha_lf(p: Path) -> str:
    b = p.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def sha_raw(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


inputs = {f"runs/eval/m33/e3x/plan/{n}": sha_lf(PLAN / n) for n in
          ("hyperbola_positions.csv", "band_positions.csv", "plan_report.json")}
inputs["docs/processing/m33_e3x_plan.md"] = sha_raw(REPO / "docs/processing/m33_e3x_plan.md")
seal = json.loads((REPO / "analysis/detection/m33_e3x_seal2.json").read_text(encoding="utf-8"))
seal_match = {}
for k, v in inputs.items():
    if not k.startswith("runs/"):
        continue
    name = k.split("/")[-1]
    ref = seal["plan_report_sha256_lf"] if name == "plan_report.json" else seal["plan_files_sha256_lf"][name]
    seal_match[k] = v == ref

# cluster definition check (principle 3): 4TU activity = before '/', IDS zone = middle token
h = pd.read_csv(PLAN / "hyperbola_positions.csv", dtype=str)
b = pd.read_csv(PLAN / "band_positions.csv", dtype=str)


def expect_cluster(row):
    parts = row["line_id"].split("/")
    return parts[0] if row["dataset"] == "4tu" else parts[1]


cluster_ok = bool(all(h.apply(expect_cluster, axis=1) == h["cluster_id"]) and
                  all(b.apply(expect_cluster, axis=1) == b["cluster_id"]))

scen = {}
for p in sorted(work.glob("res_P*.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    scen[r["config"]["name"]] = (p, r)

rows = []


def approx_mde(s):
    """Normal-theory approximation: mean half-width + z_0.8 * true SD."""
    return s["halfwidth_mean"] + 0.8416 * s["sd_est_true"]


for name, (p, r) in scen.items():
    for tag, d in r["hyp"].items():
        for hyp in ("HH1", "NC", "HH3", "HH1_full"):
            s = d.get(hyp)
            if not s:
                continue
            rows.append({"scenario": name, "mstar_set": tag, "hypothesis": hyp, "R": s["R"], "B": r["config"]["B"],
                         "seed": r["config"]["seed"], "n_pos_mean": round(s["n_pos_mean"], 2),
                         "n_clu_mean": round(s["n_clu_mean"], 2), "n_models_mean": round(s.get("n_models_mean", np.nan), 2),
                         "nu_median": round(s["nu_median"], 2), "mde80_sat": round(s["mde80"], 4),
                         "mde80_mcse": round(s["mde80_mcse"], 4), "mde80_approx": round(approx_mde(s), 4),
                         "mde80_sat_negative_dir": round(s["mde80_neg"], 4),
                         "false_claim_two_sided_at0": round(s["false_claim_two_sided_at0"], 4),
                         "false_claim_mcse": round(s["false_claim_mcse"], 4),
                         "p_negligible_at0": round(s["p_negligible_at0"], 4),
                         "p_false_negligible_at_delta": round(s["p_false_negligible_at_delta"], 4),
                         "boot_se_over_true_sd": round(s["boot_se_over_true_sd"], 3) if s["boot_se_over_true_sd"] else None,
                         "mde80_percentile_report_only": round(s["mde80_percentile_report_only"], 4)})
    for hyp in ("HB1", "HB1b", "HB2"):
        s = r[hyp]
        rows.append({"scenario": name, "mstar_set": "-", "hypothesis": hyp, "R": s["R"], "B": r["config"]["B"],
                     "seed": r["config"]["seed"], "n_pos_mean": s["n_pos_mean"], "n_clu_mean": s["n_clu_mean"],
                     "n_models_mean": None, "nu_median": round(s["nu_median"], 2), "mde80_sat": round(s["mde80"], 4),
                     "mde80_mcse": round(s["mde80_mcse"], 4), "mde80_approx": round(approx_mde(s), 4),
                     "mde80_sat_negative_dir": round(s["mde80_neg"], 4),
                     "false_claim_two_sided_at0": round(s["false_claim_two_sided_at0"], 4),
                     "false_claim_mcse": round(s["false_claim_mcse"], 4),
                     "p_negligible_at0": round(s["p_negligible_at0"], 4),
                     "p_false_negligible_at_delta": round(s["p_false_negligible_at_delta"], 4),
                     "boot_se_over_true_sd": round(s["boot_se_over_true_sd"], 3),
                     "mde80_percentile_report_only": round(s["mde80_percentile_report_only"], 4)})

hh2 = json.loads((work / "res_hh2_plan.json").read_text(encoding="utf-8"))
pts = sorted((v["truth_eta2p"], v["p_confirmed"]) for v in hh2["scenarios"].values())
tr = np.array([x for x, _ in pts])
pw = np.array([y for _, y in pts])
hh2_mde80 = float(np.interp(0.8, pw, tr))
enum = json.loads((work / "enum_minimums.json").read_text(encoding="utf-8"))
extra = {k: {"joint_false_claim_5_linear_null": r["joint_false_claim_5_linear_null"],
             "joint_mcse": round(r["joint_false_claim_mcse"], 4), "natural_mstar": r["natural_mstar"],
             "nc_trigger_natural": r["hyp"]["natural"]["NC_trigger"], "config": r["config"],
             "elapsed_s": r["elapsed_s"], "result_file": p.name, "result_sha256": sha_raw(p)}
         for k, (p, r) in scen.items()}
scripts = {n: sha_raw(work / n) for n in ("e3x_plan_mde.py", "hh2_plan.py", "gen_cfgs.py", "summarize.py", "consolidate.py")}
res = {
    "schema": "m33_e3x_mde_eval/1",
    "computed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "purpose": "plan-structure MDE80 under the sealed v0.4 rule (report only, rule unchanged); synthetic outcomes",
    "inputs_sha256_lf": inputs, "inputs_match_seal2": seal_match, "seal2_commit": "729d19c",
    "cluster_definition_ok": cluster_ok,
    "not_opened": ["runs/eval/m33/e3x/uninjected001/**", "runs/eval/m33/e3x/injected/**", "runs/eval/m33/e3x/judgment/**",
                   "runs/eval/m33/e3x_retry/**", "runs/eval/m33/e3x_precheck*/**"],
    "level": "1 - 0.05/6 (two-sided), Satterthwaite t", "main_scenario": "P0_base",
    "scripts_sha256": scripts,
    "hh2": {"rule": "basic interval a = 0.05/12, confirmed iff lower > 0.5", "R": hh2["R"], "B": hh2["B"],
            "curve": [{"truth_eta2p": round(x, 4), "p_confirmed": round(y, 4)} for x, y in pts],
            "mde80_truth_eta2p_interp": round(hh2_mde80, 4), "scenarios": hh2["scenarios"],
            "result_sha256": sha_raw(work / "res_hh2_plan.json")},
    "minimum_counts_enumeration": enum,
    "scenario_extra": extra,
    "rows": rows,
}
out_json.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(json.dumps({"status": "ok", "rows": len(rows), "cluster_ok": cluster_ok, "seal_match": all(seal_match.values()),
                  "hh2_mde80": round(hh2_mde80, 4)}))
