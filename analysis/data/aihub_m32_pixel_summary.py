"""Print a compact summary of m32_pixel_facts.json (median [p10,p90])."""
import json
from pathlib import Path

d = json.loads(Path(r"C:\workspace\gpr_detection\data\audit\aihub_71327\m32_pixel_facts.json").read_text(encoding="utf-8"))
f = lambda v: f"{v[0]:.3g} [{v[1]:.3g},{v[2]:.3g}]"
for eq, o in d["eq"].items():
    print("==", eq)
    for g in ("0.25", "0.5", "1.0"):
        s = o[g]
        print(g, "mean", f(s["mean"]), "std", f(s["std"]), "sat lo/hi", f"{s['lo'][0]:.3f}/{s['hi'][0]:.3f}",
              "top8", f(s["top8_over_all"]), "top16", f(s["top16_over_all"]), "bot32", f(s["bottom32_over_all"]),
              "peakrow_med", s["peak_row_median"], "hist16", s["peak_row_hist16"])
    for k in ("0.25_vs_1.0", "0.5_vs_1.0"):
        print(k, "slope", f(o[k]["slope"]), "r", f(o[k]["r"]))
print("row energy first 24 rows (gain1.0):")
for k, v in d["row_energy_first48"].items():
    if k.endswith("g1.0") or k.startswith("NM"):
        print(k, v[:24])
