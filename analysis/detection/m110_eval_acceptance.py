"""Acceptance check of scripts/eval_detector.py against the M1-7 numbers (CPU, read-only).

Run eval_detector.py on runs/detect/m17_aug0 first (GPU, ~1 min):
  conda run -n gpr-dev python -I scripts/eval_detector.py --run-dir runs/detect/m17_aug0 \
      --weights last --unit crop,image
then this script compares runs/detect/m17_aug0/eval/last_{crop,image}/metrics.json with
docs/experiments/m17_augmentation_strength.md (strength 0, last.pt, conf 0.001, iou 0.7):
  crop : n 1298, mAP50 0.8435, pipe mean 0.7810, mAP50-95 0.3986
  image: n 3525, mAP50 0.8352, pipe mean 0.7668, mAP50-95 0.3885
Tolerance 0.001 (0.1 %p); the doc values are rounded to 0.01-0.1 %p.
Exit 0 = pass, 1 = mismatch, 2 = metrics.json missing. Last stdout line is JSON.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "runs" / "detect" / "m17_aug0" / "eval"
EXPECTED = {
    "crop": {"count": 1298, "map50": 0.8435, "pipe": 0.7810, "map50_95": 0.3986},
    "image": {"count": 3525, "map50": 0.8352, "pipe": 0.7668, "map50_95": 0.3885},
}
TOL = 0.001


def main() -> int:
    report = {}
    fails = []
    for unit, exp in EXPECTED.items():
        path = EVAL / f"last_{unit}" / "metrics.json"
        if not path.exists():
            print(json.dumps({"verdict": "input_missing", "path": str(path)}))
            return 2
        doc = json.loads(path.read_text(encoding="utf-8"))
        got = {"count": doc["eval_images"]["count"], "map50": doc["map50"],
               "pipe": doc["pipe_ap50_mean"]["value"], "map50_95": doc["map50_95"]}
        report[unit] = got
        if got["count"] != exp["count"]:
            fails.append(f"{unit} count {got['count']} != {exp['count']}")
        for key in ("map50", "pipe", "map50_95"):
            if got[key] is None or abs(got[key] - exp[key]) > TOL:
                fails.append(f"{unit} {key} {got[key]} vs {exp[key]}")
    print(json.dumps({"verdict": "pass" if not fails else "fail", "fails": fails, "got": report}))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
