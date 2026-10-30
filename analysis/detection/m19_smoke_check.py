"""M1-9 smoke pass/fail check for the M1-10 training path (CPU, read-only).

Criteria (docs/experiments/m110_baseline_plan.md section 4):
1. Log has "AdamW(lr=0.00125, momentum=0.9)" and no "'optimizer=auto' found".
2. args.yaml: optimizer AdamW, lr0 0.00125, momentum 0.9, warmup_bias_lr 0.0,
   patience 0, epochs 3, flipud 0, seed 0.
3. gpr_train_settings.yaml: augment_strength 0, fold "0".
4. results.csv has exactly `--epochs` data rows; weights/last.pt exists.
5. Epoch-1 val mAP50 within [0.20, 0.40] (M1-7 fold 0 epoch 1: 0.273-0.301).

Usage:
  conda run -n gpr-dev python -I analysis/detection/m19_smoke_check.py \
      --run-dir runs/detect/yolo11m_aihubv1_<D>_smoke_fold0 --log runs/logs/m19_smoke.log
Exit 0 = pass, 1 = fail, 2 = input missing. Last stdout line is JSON.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

EXPECTED_ARGS = {"optimizer": "AdamW", "lr0": 0.00125, "momentum": 0.9, "warmup_bias_lr": 0.0,
                 "patience": 0, "flipud": 0.0, "seed": 0, "imgsz": 256, "batch": 64}
MAP50_BAND = (0.20, 0.40)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    run = args.run_dir
    if not run.is_dir() or not args.log.is_file():
        print(json.dumps({"verdict": "input_missing", "run_dir": str(run), "log": str(args.log)}))
        return 2

    fails: list[str] = []
    log = args.log.read_text(encoding="utf-8", errors="replace")
    if "AdamW(lr=0.00125, momentum=0.9)" not in log:
        fails.append("log: AdamW(lr=0.00125, momentum=0.9) missing")
    if "'optimizer=auto' found" in log:
        fails.append("log: optimizer=auto used")

    train_args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    expected = dict(EXPECTED_ARGS, epochs=args.epochs)
    for key, value in expected.items():
        if train_args.get(key) != value:
            fails.append(f"args.yaml {key}={train_args.get(key)!r} != {value!r}")

    settings = yaml.safe_load((run / "gpr_train_settings.yaml").read_text(encoding="utf-8"))
    if float(settings.get("augment_strength", -1)) != 0.0:
        fails.append(f"augment_strength={settings.get('augment_strength')}")
    if str(settings.get("fold")) != "0":
        fails.append(f"fold={settings.get('fold')}")

    rows: list[dict[str, str]] = []
    csv_path = run / "results.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(handle)]
    if len(rows) != args.epochs:
        fails.append(f"results.csv rows {len(rows)} != {args.epochs}")
    if not (run / "weights" / "last.pt").exists():
        fails.append("weights/last.pt missing")
    map50_e1 = float(rows[0]["metrics/mAP50(B)"]) if rows else None
    if map50_e1 is None or not MAP50_BAND[0] <= map50_e1 <= MAP50_BAND[1]:
        fails.append(f"epoch1 mAP50 {map50_e1} outside {MAP50_BAND}")

    result = {"verdict": "pass" if not fails else "fail", "fails": fails,
              "epoch1_mAP50": map50_e1,
              "mAP50_by_epoch": [float(r["metrics/mAP50(B)"]) for r in rows],
              "epoch_time_s": [float(r["time"]) for r in rows]}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
