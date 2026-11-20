"""Collect scenario results into one table (json + csv). Usage: summarize.py <out.json> <out.csv> <res...>"""
import csv
import json
import sys
from pathlib import Path

rows = []
allres = {}
for p in sys.argv[3:]:
    r = json.loads(Path(p).read_text(encoding="utf-8"))
    name = r["config"]["name"]
    allres[name] = r
    for tag, d in r["hyp"].items():
        for h in ("HH1", "NC", "HH3", "HH1_full"):
            s = d.get(h)
            if not s:
                continue
            rows.append({"scenario": name, "mstar": tag, "hyp": h, "R": s["R"], "B": r["config"]["B"],
                         "n_pos_mean": round(s["n_pos_mean"], 2), "n_clu_mean": round(s["n_clu_mean"], 2),
                         "n_models_mean": round(s.get("n_models_mean", float("nan")), 2),
                         "nu_median": round(s["nu_median"], 2) if s["nu_median"] == s["nu_median"] else None,
                         "mde80": round(s["mde80"], 4), "mde80_mcse": round(s["mde80_mcse"], 4),
                         "mde80_neg": round(s["mde80_neg"], 4), "mde50": round(s["mde50"], 4),
                         "false_claim_at0": round(s["false_claim_two_sided_at0"], 4),
                         "false_claim_mcse": round(s["false_claim_mcse"], 4),
                         "p_negligible_at0": round(s["p_negligible_at0"], 4),
                         "p_false_negl_at_delta": round(s["p_false_negligible_at_delta"], 4),
                         "boot_se_over_true_sd": round(s["boot_se_over_true_sd"], 3) if s["boot_se_over_true_sd"] else None,
                         "mde80_percentile_report_only": round(s["mde80_percentile_report_only"], 4)})
    for h in ("HB1", "HB1b", "HB2"):
        s = r[h]
        rows.append({"scenario": name, "mstar": "-", "hyp": h, "R": s["R"], "B": r["config"]["B"],
                     "n_pos_mean": s["n_pos_mean"], "n_clu_mean": s["n_clu_mean"], "n_models_mean": None,
                     "nu_median": round(s["nu_median"], 2), "mde80": round(s["mde80"], 4),
                     "mde80_mcse": round(s["mde80_mcse"], 4), "mde80_neg": round(s["mde80_neg"], 4),
                     "mde50": round(s["mde50"], 4), "false_claim_at0": round(s["false_claim_two_sided_at0"], 4),
                     "false_claim_mcse": round(s["false_claim_mcse"], 4),
                     "p_negligible_at0": round(s["p_negligible_at0"], 4),
                     "p_false_negl_at_delta": round(s["p_false_negligible_at_delta"], 4),
                     "boot_se_over_true_sd": round(s["boot_se_over_true_sd"], 3),
                     "mde80_percentile_report_only": round(s["mde80_percentile_report_only"], 4)})
extra = {k: {"joint_false_claim_5_linear_null": v["joint_false_claim_5_linear_null"],
             "joint_mcse": v["joint_false_claim_mcse"], "natural_mstar": v["natural_mstar"],
             "nc_trigger_natural": v["hyp"]["natural"]["NC_trigger"],
             "elapsed_s": v["elapsed_s"], "seed": v["config"]["seed"]} for k, v in allres.items()}
Path(sys.argv[1]).write_text(json.dumps({"rows": rows, "scenario_extra": extra}, indent=1), encoding="utf-8")
with open(sys.argv[2], "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(json.dumps({"status": "ok", "rows": len(rows)}))
