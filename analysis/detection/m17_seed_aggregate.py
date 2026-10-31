"""Aggregate M1-7 augmentation-strength runs (seed 0 x4, seed 1 x2) and apply the decision rule.

Reads runs/detect/<run>/eval/{last,best}_{crop,image}/metrics.json (scripts/eval_detector.py, schema gpr_eval/1).
Decision (user, 2026-10-30/31): for strength 0 and 1, mean over seeds 0 and 1 of the pipe 2-class AP50
(pipe_hyperbola, pipe_band) on last.pt / crop unit; pick the highest; if within 1 %p pick the weaker (0).

Usage: python -I analysis/detection/m17_seed_aggregate.py [--root C:/workspace/gpr_detection] [--out <json>]
Writes a JSON summary (default analysis/detection/m17_seed_aggregate.json); prints markdown tables and
a final one-line JSON on stdout.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

RUNS = {
    # (strength, seed): run folder
    ("0", 0): "m17_aug0",
    ("0.5", 0): "m17_aug0.5",
    ("1", 0): "m17_aug1",
    ("2", 0): "m17_aug2",
    ("0", 1): "m17_aug0_s1",
    ("1", 1): "m17_aug1_s1",
}
CLASSES = ["pipe_hyperbola", "pipe_band", "cavity", "manhole"]
VARIANTS = ["last_crop", "last_image", "best_crop", "best_image"]
TIE_MARGIN_PP = 1.0


def load(root: Path, run: str, variant: str) -> dict:
    p = root / "runs" / "detect" / run / "eval" / variant / "metrics.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    ap = {c["name"]: c["ap50"] * 100 for c in m["per_class"]}
    inst = {c["name"]: c["instances"] for c in m["per_class"]}
    cm = m["confusion_matrix_at_confusion_conf"]
    assert cm["conf_threshold"] == 0.25, p
    labels = cm["labels"]
    mat = cm["matrix"]
    bg = labels.index("background")
    band = labels.index("pipe_band")
    conf = {
        "band_tp": mat[band][band],
        "band_fn_to_bg": mat[bg][band],
        "bg_fp_band": mat[band][bg],
        "bg_fp": {labels[i]: mat[i][bg] for i in range(len(labels)) if i != bg},
        "gt_missed": {labels[j]: mat[bg][j] for j in range(len(labels)) if j != bg},
    }
    cm001 = m["confusion_matrix"]
    conf["bg_fp_band_conf0.001"] = cm001["matrix"][band][bg]
    return {
        "path": str(p),
        "count": m["eval_images"]["count"],
        "eval_images_sha256": m["eval_images"]["sha256"],
        "weights_sha256": m["weights"]["sha256"],
        "ap50": ap,
        "instances": inst,
        "pipe": m["pipe_ap50_mean"]["value"] * 100,
        "map50": m["map50"] * 100,
        "map50_95": m["map50_95"] * 100,
        "confusion_conf0.25": conf,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "analysis" / "detection" / "m17_seed_aggregate.json"

    data: dict = {}
    try:
        for (s, seed), run in RUNS.items():
            for v in VARIANTS:
                data.setdefault(v, {})[f"{s}|{seed}"] = {"strength": s, "seed": seed, "run": run,
                                                       **load(root, run, v)}
    except (OSError, KeyError, AssertionError) as e:
        print(json.dumps({"status": "input_error", "error": repr(e)}))
        return 2

    # all runs must be evaluated on the same image list per variant
    for v in VARIANTS:
        shas = {r["eval_images_sha256"] for r in data[v].values()}
        if len(shas) != 1:
            print(json.dumps({"status": "mismatch", "variant": v, "eval_images_sha256": sorted(shas)}))
            return 1

    summary: dict = {"variants": {}, "runs": data}
    for v in VARIANTS:
        d = data[v]
        per = {}
        for s in ("0", "1"):
            r0, r1 = d[f"{s}|0"], d[f"{s}|1"]
            keys = {"pipe": (r0["pipe"], r1["pipe"]), "map50": (r0["map50"], r1["map50"]),
                    "map50_95": (r0["map50_95"], r1["map50_95"])}
            keys.update({c: (r0["ap50"][c], r1["ap50"][c]) for c in CLASSES})
            per[s] = {k: {"seed0": x0, "seed1": x1, "mean": (x0 + x1) / 2, "diff_s0_minus_s1": x0 - x1,
                          "sd": statistics.stdev([x0, x1])} for k, (x0, x1) in keys.items()}
        m0, m1 = per["0"]["pipe"]["mean"], per["1"]["pipe"]["mean"]
        best = "0" if m0 >= m1 else "1"
        gap = abs(m0 - m1)
        pick = "0" if gap <= TIE_MARGIN_PP else best
        summary["variants"][v] = {"per_strength": per, "pipe_mean_gap_pp": m0 - m1, "highest": best,
                                  "decision": pick}

    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown printout
    for v in VARIANTS:
        print(f"\n## {v}  (n={next(iter(data[v].values()))['count']})")
        print("| strength | seed | hyp | band | pipe | cavity | manhole | mAP50 | mAP50-95 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for key, r in data[v].items():
            print(f"| {r['strength']} | {r['seed']} | " + " | ".join(f"{r['ap50'][c]:.2f}" for c in CLASSES[:2])
                  + f" | {r['pipe']:.2f} | " + " | ".join(f"{r['ap50'][c]:.2f}" for c in CLASSES[2:])
                  + f" | {r['map50']:.2f} | {r['map50_95']:.2f} |")
        sv = summary["variants"][v]
        for s in ("0", "1"):
            p = sv["per_strength"][s]
            print(f"strength {s} mean: pipe {p['pipe']['mean']:.3f} (s0-s1 {p['pipe']['diff_s0_minus_s1']:+.2f}), "
                  f"mAP50 {p['map50']['mean']:.3f} (s0-s1 {p['map50']['diff_s0_minus_s1']:+.2f}), "
                  + ", ".join(f"{c} {p[c]['mean']:.2f} ({p[c]['diff_s0_minus_s1']:+.2f})" for c in CLASSES))
        print(f"gap(0-1) {sv['pipe_mean_gap_pp']:+.3f} pp, highest {sv['highest']}, decision {sv['decision']}")
        if v.endswith("crop"):
            print("confusion conf0.25:")
            for key, r in data[v].items():
                c = r["confusion_conf0.25"]
                print(f"  {key}: band TP {c['band_tp']} FN {c['band_fn_to_bg']} bgFP band {c['bg_fp_band']} "
                      f"(conf0.001 {c['bg_fp_band_conf0.001']}) bgFP {c['bg_fp']} missed {c['gt_missed']}")

    lc = summary["variants"]["last_crop"]
    print(json.dumps({"status": "ok", "decision_last_crop": lc["decision"],
                      "pipe_gap_pp": round(lc["pipe_mean_gap_pp"], 3), "out": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
