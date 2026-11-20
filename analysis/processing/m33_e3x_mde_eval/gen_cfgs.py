"""Write scenario configs for e3x_plan_mde.py."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = {"plan_dir": "C:/workspace/gpr_detection/runs/eval/m33/e3x/plan", "B": 2000,
        "tau_h": 0.8, "s_w2": 0.1, "rho_f": 0.85, "icc_h_4tu": 0.28, "icc_h_ids": 0.42,
        "rho_man": 0.0, "rho_nc": 0.9, "tau_b": 1.5, "s_w2_b": 0.3, "rho_f_b": 0.5,
        "icc_b_4tu": 0.05, "icc_b_ids": 0.06, "band_mu_ids": 0.19, "band_mu_4tu": 0.12,
        "rho_band": 0.0, "sd_cluster_diff": 0.2, "forced_exclusions": [], "forced_named": {}}
SC = {
    "P0_base": {"seed": 20261114, "R": 4000, "forced_exclusions": [0, 2, 3, 4, 5, 6, 7],
                "forced_named": {"Me10_noM12": ["M15", "M16", "M09", "M04", "M12"]}},
    "P1_base_B20k": {"seed": 20261115, "R": 800, "B": 20000, "forced_exclusions": [4]},
    "P2_cd04": {"seed": 20261116, "R": 2000, "sd_cluster_diff": 0.4, "forced_exclusions": [3, 4, 5]},
    "P3_cd0": {"seed": 20261117, "R": 2000, "sd_cluster_diff": 0.0, "forced_exclusions": [3, 4, 5]},
    "P4_rhoman05": {"seed": 20261118, "R": 2000, "rho_man": 0.5, "forced_exclusions": [3, 4, 5]},
    "P5_shift_m05": {"seed": 20261119, "R": 2000, "q_shift": -0.5},
    "P6_shift_m10": {"seed": 20261120, "R": 2000, "q_shift": -1.0},
    "P7_band_low": {"seed": 20261121, "R": 2000, "band_mu_ids": 0.10, "band_mu_4tu": 0.06},
    "P8_band_high": {"seed": 20261122, "R": 2000, "band_mu_ids": 0.35, "band_mu_4tu": 0.25},
    "P9_selfcheck_iid": {"seed": 20261123, "R": 4000, "icc_h_4tu": 0.0, "icc_h_ids": 0.0, "icc_b_4tu": 0.0,
                         "icc_b_ids": 0.0, "sd_cluster_diff": 0.0, "rho_f": 0.0, "rho_f_b": 0.0},
    "P10_selfcheck_rhonc1": {"seed": 20261124, "R": 50, "rho_nc": 1.0},
}
for k, v in SC.items():
    (HERE / f"cfg_{k}.json").write_text(json.dumps(BASE | v | {"name": k}), encoding="utf-8")
print(json.dumps({"status": "ok", "n": len(SC)}))
