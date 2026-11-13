"""Diagnose the failed D1 baseline check (m31_detection_input_tiling.md §8-2 (3)).

Separates the two possible sources of |AP50(D1 g1.00_dx+0) - AP50(M1-10)| > 1 %p
without any GPU run:
  (A) metric implementation: score the *validator* detections that
      export_val_predictions.py saved for M1-10 (runs/analysis/m110_band_aux/
      fold<k>_crop/predictions.csv, same weights, same crop list) with the D1
      AP / recall code, and compare with M1-10 metrics.json.
  (B) inference path: compare per-object hits at conf > 0.25 between the
      validator detections and the D1 backend run (objects.csv, condition
      g1.00_dx+0), object by object.
CPU only, read-only. Writes JSON (--out); last stdout line is JSON.
"""

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO / "runs" / "detect"
EXPORT_ROOT = REPO / "runs" / "analysis" / "m110_band_aux"
PREFIX = "yolo11m_aihubv1_20261101_fold"
CLASS_NAMES = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")


def load_d1_module():
    path = Path(__file__).with_name("m31_d1_shift_scale.py")
    spec = importlib.util.spec_from_file_location("m31_d1_shift_scale", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    d1 = load_d1_module()
    result = {"folds": {}}
    for k in range(5):
        run = RUN_ROOT / f"{PREFIX}{k}"
        m110 = json.loads((run / "eval" / "last_crop" / "metrics.json").read_text(encoding="utf-8"))
        m110_ap = {pc["name"]: pc["ap50"] for pc in m110["per_class"]}
        d1m = json.loads((run / "eval" / "m31_d1" / "metrics.json").read_text(encoding="utf-8"))
        d1_base = d1m["conditions"]["g1.00_dx+0"]["classes"]
        stems = [Path(line.strip()).stem for line in (run / "eval" / "last_crop" / "eval_images.txt")
                 .read_text(encoding="utf-8").splitlines() if line.strip()]
        preds: dict[str, list] = defaultdict(list)
        with (EXPORT_ROOT / f"fold{k}_crop" / "predictions.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                preds[Path(row["file_name"]).stem].append(
                    (int(row["cls"]), float(row["conf"]), float(row["x1"]), float(row["y1"]),
                     float(row["x2"]), float(row["y2"])))
        acc = {c: {"scores": [], "tps": [], "n_gt": 0, "hit": 0} for c in range(4)}
        val_hits: dict[tuple, int] = {}
        n_pred_total = 0
        for stem in stems:
            label = d1.LABEL_DIR / f"{stem}.txt"
            gt_rows = []
            if label.is_file():
                for row in label.read_text(encoding="utf-8").splitlines():
                    p = row.split()
                    if len(p) == 5:
                        c, xc, yc, w, h = int(p[0]), *map(float, p[1:])
                        gt_rows.append((c, (xc - w / 2) * 256, (yc - h / 2) * 256,
                                        (xc + w / 2) * 256, (yc + h / 2) * 256))
            gt = np.asarray(gt_rows, dtype=float).reshape(-1, 5)
            pr = np.asarray(preds.get(stem, []), dtype=float).reshape(-1, 6)
            n_pred_total += len(pr)
            for c in range(4):
                g = gt[gt[:, 0] == c, 1:5]
                p = pr[pr[:, 0] == c]
                tp, _ = d1.match(p[:, 2:6], p[:, 1], g)
                acc[c]["scores"].append(p[:, 1]); acc[c]["tps"].append(tp); acc[c]["n_gt"] += len(g)
                sel = p[:, 1] > d1.CONF_RECALL
                _, gs = d1.match(p[sel, 2:6], p[sel, 1], g)
                acc[c]["hit"] += int((gs >= 0).sum())
                for b, s in zip(g, gs):
                    key = (stem, CLASS_NAMES[c], round((b[0] + b[2]) / 2, 2))
                    val_hits[key] = int(s >= 0)
        fold = {"n_val_predictions": n_pred_total, "classes": {}}
        for c, a in acc.items():
            name = CLASS_NAMES[c]
            ap = d1.ap_101(np.concatenate(a["scores"]), np.concatenate(a["tps"]), a["n_gt"])
            fold["classes"][name] = {
                "ap50_m110": round(100 * m110_ap[name], 3),
                "ap50_d1code_on_val_preds": round(100 * ap, 3),
                "diff_A_metric_pp": round(100 * (ap - m110_ap[name]), 3),
                "ap50_d1_backend": round(100 * d1_base[name]["ap50"], 3),
                "diff_B_path_pp": round(100 * (d1_base[name]["ap50"] - ap), 3),
                "recall025_val_preds": round(100 * a["hit"] / a["n_gt"], 3) if a["n_gt"] else None,
                "recall025_d1_backend": round(100 * d1_base[name]["recall_conf025"], 3)
                if d1_base[name]["recall_conf025"] is not None else None,
            }
        # object-level agreement at conf > 0.25
        agree = defaultdict(lambda: [0, 0, 0, 0])  # both, val_only, d1_only, neither
        with (run / "eval" / "m31_d1" / "objects.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["condition"] != "g1.00_dx+0":
                    continue
                key = (row["image"], row["class"], round(float(row["x_center"]), 2))
                if key not in val_hits:
                    continue
                v, d = val_hits[key], int(row["hit_conf025"])
                idx = 0 if v and d else 1 if v else 2 if d else 3
                agree[row["class"]][idx] += 1
        fold["object_hits_both_valonly_d1only_neither"] = dict(agree)
        result["folds"][k] = fold
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    summ = {k: {c: (v["classes"][c]["diff_A_metric_pp"], v["classes"][c]["diff_B_path_pp"])
                for c in ("pipe_hyperbola", "cavity", "manhole")} for k, v in result["folds"].items()}
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
