"""Read-only: condense the M3-2 stage-B S3 outputs into one table (judge use).

Reads runs/eval/m32_aihub_perturb/summary.json and fold<k>/metrics.json and
writes a Markdown-ready JSON to the path given as the only argument. Prints
only a status line (blind rule: no metric values on stdout).
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "runs" / "eval" / "m32_aihub_perturb"
CLASSES = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")


def pp(value):
    return None if value is None else round(100.0 * value, 2)


def main(out_path: str) -> int:
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    folds = {
        k: json.loads((ROOT / f"fold{k}" / "metrics.json").read_text(encoding="utf-8"))
        for k in range(5)
    }
    gate = {
        k: {
            "pass": v["pass"],
            "n_checks": len(v["checks"]),
            "failed": [c for c in v["checks"] if not c["pass"]],
            "validator_ap50_diff_pp": v.get("validator_ap50_diff_pp"),
        }
        for k, v in summary["gate"].items()
    }
    table = {}
    for name, agg in summary["aggregate"].items():
        row = {"map50": {"mean": pp(agg["map50"]["mean"]), "sd": pp(agg["map50"]["sd"])}}
        for cls in CLASSES:
            c = agg["classes"][cls]
            per_fold = []
            per_fold_recall = []
            for k in range(5):
                pr = folds[k]["paired_recall"][name][cls]
                per_fold.append(pp(pr["delta_vs_identity"]))
                per_fold_recall.append(pp(folds[k]["conditions"][name]["classes"][cls]["recall_conf025"]))
            row[cls] = {
                "ap50": pp(c["ap50"]["mean"]),
                "ap50_sd": pp(c["ap50"]["sd"]),
                "recall": pp(c["recall_conf025"]["mean"]),
                "paired_delta": pp(c["paired_recall_delta"]["mean"]),
                "paired_delta_sd": pp(c["paired_recall_delta"]["sd"]),
                "paired_delta_folds": per_fold,
                "recall_folds": per_fold_recall,
                "paired_delta_all_common": pp(c["paired_recall_delta_all_common"]["mean"]),
                "n_ignored_total": c["n_ignored_total"],
            }
        table[name] = row
    by_gain = {}
    for name in ("identity", "polarity_m3", "bg_b0_row_median", "recipe_r0", "recipe_r5", "recipe_r6"):
        by_gain[name] = {}
        for gain in ("0.25", "0.5", "1.0"):
            vals = [folds[k]["conditions"][name]["by_gain"][gain]["classes"]["pipe_hyperbola"]["recall_conf025"] for k in range(5)]
            ns = [folds[k]["conditions"][name]["by_gain"][gain]["classes"]["pipe_hyperbola"]["n_gt"] for k in range(5)]
            by_gain[name][gain] = {"recall_folds": [pp(v) for v in vals], "n_gt": ns}
    out = {
        "status": summary["status"],
        "gated": summary["settings"]["gated"],
        "n_images_total": summary["n_images_total"],
        "gate": gate,
        "resources": summary.get("resources"),
        "table": table,
        "hyperbola_by_gain": by_gain,
    }
    Path(out_path).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "out": out_path}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
