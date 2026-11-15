"""M3-3 D0 follow-up (3): the same gprMax models with a real background.

Reads the M3-2 stage C pre-check records (r0, fold 0, conf 0.25 path,
runs/eval/m32_recipe_v2/injected_precheck/parts/*.json) and tabulates the
pipe_hyperbola hit rate per response model M01-M20 and SNR, next to the D0 Y0
stand-alone hits (runs/eval/m33_d0/synthetic_apex.csv, fold 0 and all folds).
Read only; no detector run. Output: analysis/processing/m33_y0_vs_stagec_inject.json
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARTS = ROOT / "runs" / "eval" / "m32_recipe_v2" / "injected_precheck" / "parts"
D0_APEX = ROOT / "runs" / "eval" / "m33_d0" / "synthetic_apex.csv"
OUT = ROOT / "analysis" / "processing" / "m33_y0_vs_stagec_inject.json"
MODELS = {f"M{i:02d}" for i in range(1, 21)}


def main() -> None:
    cell = defaultdict(lambda: {"n": 0, "hit": 0, "hit_any": 0, "n_in": 0, "hit_in": 0})
    for path in sorted(PARTS.glob("*.json")):
        for m in json.loads(path.read_text(encoding="utf-8"))["inj_matches"]:
            if m["response_id"] not in MODELS or m["recipe"] != "r0" or not m.get("main_cell", True):
                continue
            for key in ((m["response_id"], m["snr_db"]), ("ALL", m["snr_db"]),
                        (m["dataset"], m["snr_db"])):
                c = cell[key]
                c["n"] += 1
                c["hit"] += int(m["hit"])
                c["hit_any"] += int(m["hit_any_class"])
                c["n_in"] += int(m["in_distribution"])
                c["hit_in"] += int(m["hit"] and m["in_distribution"])
    y0 = defaultdict(lambda: {"fold0": 0, "all_folds": 0})
    with open(D0_APEX, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["condition"] == "Y0":
                y0[r["model"]]["all_folds"] += int(r["hit_S0"])
                if r["fold"] == "0":
                    y0[r["model"]]["fold0"] += int(r["hit_S0"])
    table = {}
    for (key, snr), c in sorted(cell.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        table.setdefault(key, {})[f"{snr:g}dB"] = {
            **c, "recall": c["hit"] / c["n"] if c["n"] else None,
            "recall_in": c["hit_in"] / c["n_in"] if c["n_in"] else None}
    out = {"stage_c_r0_fold0_by_model": table, "d0_y0_hits": dict(y0)}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for key in ["ALL", "4tu", "guangzhou", *sorted(MODELS)]:
        if key not in table:
            continue
        cells = " ".join(f"{s}:{v['hit']}/{v['n']}" for s, v in table[key].items())
        extra = f" | Y0 f0 {y0[key]['fold0']}/1, all {y0[key]['all_folds']}/5" if key in MODELS else ""
        print(f"{key:9s} {cells}{extra}")
    print(json.dumps({"status": "ok", "out": str(OUT)}))


if __name__ == "__main__":
    main()
