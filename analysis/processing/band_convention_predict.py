"""Export pipe_band prediction boxes of the v1 / v2 band-convention models (fold 0 crops).

Input: the 1,298 fold-0 crop images listed in
``runs/detect/band_v2_eval/<run>__gt_v2/eval/last_crop/eval_images.txt`` (same list for all runs),
weights ``runs/detect/<run>/weights/last.pt`` for the four runs of docs/experiments/band_merge_v2_compare.md.

Inference: Ultralytics ``YOLO.predict`` (deployment path, not ``val``), imgsz 256, conf 0.25,
NMS IoU 0.7, single GPU (device 0), batch 64. AGPL: ultralytics is imported only in this analysis script.

Output (local only - AIHub file names and box coordinates):
``data/audit/band_convention_preds.csv`` with run, file_name, cls, conf, x1, y1, x2, y2 (original pixels).
stdout last line: JSON summary.

Example:
    conda run -n gpr-dev python -I analysis/processing/band_convention_predict.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "v1_s0": "m17_aug0",
    "v1_s1": "m17_aug0_s1",
    "v2_s0": "yolo11m_bandv2_aug0_20261031_fold0_s0",
    "v2_s1": "yolo11m_bandv2_aug0_20261031_fold0_s1",
}
PIPE_BAND = 1
CONF = 0.25
IOU = 0.7
IMGSZ = 256


def image_list() -> list[Path]:
    path = ROOT / "runs" / "detect" / "band_v2_eval" / f"{RUNS['v2_s0']}__gt_v2" / "eval" / "last_crop" / "eval_images.txt"
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "audit" / "band_convention_preds.csv")
    a = ap.parse_args()
    try:
        images = image_list()
        if a.limit:
            images = images[: a.limit]
        for name, run in RUNS.items():
            assert (ROOT / "runs" / "detect" / run / "weights" / "last.pt").exists(), run
    except (OSError, AssertionError) as exc:
        print(json.dumps({"status": "input_error", "error": repr(exc)}))
        return 2

    import cv2
    from ultralytics import YOLO

    rows: list[dict[str, object]] = []
    timing: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, run in RUNS.items():
        model = YOLO(str(ROOT / "runs" / "detect" / run / "weights" / "last.pt"))
        start = time.perf_counter()
        n_band = 0
        for begin in range(0, len(images), a.batch):
            chunk = images[begin : begin + a.batch]
            arrays = []
            for path in chunk:
                gray = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise OSError(f"cannot read {path}")
                arrays.append(gray[:, :, None])
            results = model.predict(arrays, imgsz=IMGSZ, conf=CONF, iou=IOU, device=a.device,
                                    verbose=False, batch=len(arrays))
            for path, res in zip(chunk, results):
                boxes = res.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy()
                for b, c, s in zip(xyxy, cls, conf):
                    if c != PIPE_BAND:
                        continue
                    n_band += 1
                    rows.append({"run": name, "file_name": path.name, "cls": int(c), "conf": round(float(s), 4),
                                 "x1": round(float(b[0]), 2), "y1": round(float(b[1]), 2),
                                 "x2": round(float(b[2]), 2), "y2": round(float(b[3]), 2)})
        timing[name] = round(time.perf_counter() - start, 1)
        counts[name] = n_band
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "file_name", "cls", "conf", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "ok", "images": len(images), "band_boxes": counts, "seconds": timing,
                      "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
