"""Summarise the M3-3 Y0 diagnostic inference E0-E4 (runs/eval/m33_y0_diag_infer).

Diagnostic only, read only, CPU. Counts per (experiment, variant):
  * hits: apex-containing pipe_hyperbola box with score > 0.25 (D0 line rule),
  * same on a common model set (models present in every variant: E3 skips
    M13, M20 and, at 30 dB, M01, M14),
  * per-model fold hits, median best apex score (missing box = 0),
  * paired discordance against E0 per (model, fold),
  * E3 fold-0 hits next to the M3-2 stage C fold-0 12 dB recall per model.
Outputs: analysis/processing/m33_y0_infer_summary.json,
docs/processing/figures/m33_y0_diag/infer_hits_by_model.png (no AIHub pixels).
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "eval" / "m33_y0_diag_infer"
STAGE_C = ROOT / "analysis" / "processing" / "m33_y0_vs_stagec_inject.json"
OUT = ROOT / "analysis" / "processing" / "m33_y0_infer_summary.json"
FIG = ROOT / "docs" / "processing" / "figures" / "m33_y0_diag" / "infer_hits_by_model.png"
CONF = 0.25
MODELS = [f"M{i:02d}" for i in range(1, 21)]
ORDER = ["E0/r0", "E1/G4", "E1/G2_b0.5", "E4/centred", "E2/snr6", "E2/snr12", "E2/snr18",
         "E2/snr30", "E3/snr6", "E3/snr12", "E3/snr18", "E3/snr30"]


def score(value: str) -> float:
    return float(value) if value else 0.0


def main() -> None:
    if os.name == "nt":
        import ctypes

        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)
    rec: dict[tuple[str, str, int], dict] = {}
    with open(RUN / "records.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = f"{r['experiment']}/{r['variant']}"
            rec[(key, r["model"], int(r["fold"]))] = r
    present = defaultdict(set)
    for key, model, _ in rec:
        present[key].add(model)
    common = sorted(set.intersection(*(present[k] for k in ORDER)))

    def best(key, model, fold):
        r = rec.get((key, model, fold))
        return None if r is None else score(r["best_apex_hyperbola_score"])

    def best_any(key, model, fold):
        r = rec.get((key, model, fold))
        return None if r is None else score(r["best_apex_any_class_score"])

    summary = {}
    for key in ORDER:
        rows = [(m, f, best(key, m, f)) for m in MODELS for f in range(5)
                if best(key, m, f) is not None]
        rows_c = [x for x in rows if x[0] in common]
        per_model = {m: sum(1 for mm, f, s in rows if mm == m and s > CONF) for m in present[key]}
        disc = {"hit_only_here": 0, "hit_only_E0": 0}
        for m, f, s in rows:
            s0 = best("E0/r0", m, f)
            disc["hit_only_here"] += int(s > CONF and s0 <= CONF)
            disc["hit_only_E0"] += int(s <= CONF and s0 > CONF)
        summary[key] = {
            "n": len(rows),
            "hits": sum(s > CONF for _, _, s in rows),
            "n_common": len(rows_c),
            "hits_common": sum(s > CONF for _, _, s in rows_c),
            "models_hit_any_fold": sorted(m for m, n in per_model.items() if n > 0),
            "median_best_score": float(np.median([s for _, _, s in rows])),
            "median_best_score_common": float(np.median([s for _, _, s in rows_c])),
            "any_class_hits": sum(best_any(key, m, f) > CONF for m, f, _ in rows),
            "per_model_fold_hits": dict(sorted(per_model.items())),
            "hits_by_fold": [sum(1 for mm, f, s in rows if f == k and s > CONF) for k in range(5)],
            "hits_by_fold_common": [sum(1 for mm, f, s in rows_c if f == k and s > CONF)
                                    for k in range(5)],
            "paired_vs_E0": disc,
        }
    stage_c = json.loads(STAGE_C.read_text(encoding="utf-8"))["stage_c_r0_fold0_by_model"]
    e3_vs_c = {}
    for m in MODELS:
        c12 = stage_c.get(m, {}).get("12dB")
        e3_vs_c[m] = {
            "stage_c_12dB_recall": None if c12 is None else round(c12["recall"], 3),
            **{f"E3_{s}_fold0": (None if best(f"E3/snr{s}", m, 0) is None
                                 else int(best(f"E3/snr{s}", m, 0) > CONF))
               for s in (6, 12, 18, 30)},
            **{f"E3_{s}_folds": summary[f"E3/snr{s}"]["per_model_fold_hits"].get(m)
               for s in (6, 12, 18, 30)},
        }
    out = {"conf": CONF, "common_models": common, "by_variant": summary, "e3_vs_stage_c": e3_vs_c}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    grid = np.full((len(MODELS), len(ORDER)), np.nan)
    for j, key in enumerate(ORDER):
        for i, m in enumerate(MODELS):
            if m in present[key]:
                grid[i, j] = summary[key]["per_model_fold_hits"][m]
    ramp = ListedColormap(["#f3f2ee", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap=ramp, vmin=-0.5, vmax=5.5, aspect="auto")
    ax.set_facecolor("#ffffff")
    for i in range(len(MODELS)):
        for j in range(len(ORDER)):
            if np.isnan(grid[i, j]):
                ax.text(j, i, "-", ha="center", va="center", fontsize=7, color="#52514e")
            elif grid[i, j] > 0:
                ax.text(j, i, f"{int(grid[i, j])}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if grid[i, j] >= 3 else "#0b0b0b")
    ax.set_xticks(range(len(ORDER)), [k.replace("/", "\n") for k in ORDER], fontsize=7)
    ax.set_yticks(range(len(MODELS)), MODELS, fontsize=7)
    totals = [f"{summary[k]['hits']}/{summary[k]['n']}" for k in ORDER]
    for j, t in enumerate(totals):
        ax.text(j, -0.9, t, ha="center", va="bottom", fontsize=7, color="#52514e")
    ax.set_ylim(len(MODELS) - 0.5, -1.6)
    cb = fig.colorbar(im, ax=ax, ticks=range(6), fraction=0.03)
    cb.set_label("folds with apex pipe_hyperbola score > 0.25 (of 5)", fontsize=7)
    ax.set_title("Y0 diagnostic inference E0-E4 (top: hits / records; '-' = not run)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    plt.close(fig)
    print(json.dumps({"status": "ok", "common": common,
                      "hits": {k: (summary[k]["hits"], summary[k]["n"], summary[k]["hits_common"],
                                   summary[k]["n_common"]) for k in ORDER},
                      "by_fold_common": {k: summary[k]["hits_by_fold_common"] for k in ORDER},
                      "paired": {k: summary[k]["paired_vs_E0"] for k in ORDER}}))


if __name__ == "__main__":
    main()
