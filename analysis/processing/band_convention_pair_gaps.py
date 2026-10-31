"""Gap distribution of depth-stage merge candidates (band spec 3-4 item 1) per box source.

Box geometry only (no image processing): for the fold-0 band crops of band_box_convention_depth.py,
count candidate pairs (y overlap >= 50 % of the smaller height, gap <= G) by gap bin for the GT
boxes and the detector boxes (data/audit/band_convention_preds.csv). gap < 0 means the two boxes
overlap in x (duplicate-like detections); the facing-end check of group_band_boxes then compares
picks at the far ends of the overlap, not at a common trace.

Output: stdout JSON (last line). Example:
    conda run -n gpr-dev python -I analysis/processing/band_convention_pair_gaps.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_box_convention_depth as base  # noqa: E402


def main() -> int:
    from gpr_detection.depth.band import candidate_pairs
    from gpr_detection.depth.config import BandConfig
    cfg = replace(BandConfig(), merge_gap_px=32)
    names = [Path(p.strip()).name for p in base.LIST_TXT.read_text(encoding="utf-8").splitlines() if p.strip()]
    preds = defaultdict(lambda: defaultdict(list))
    with base.PRED_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            preds[row["file_name"]][row["run"]].append(tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2")))
    out = defaultdict(Counter)
    for name in names:
        stem = Path(name).stem
        v1 = base.yolo_band_boxes(base.V1_DIR / "labels" / f"{stem}.txt", 256, 256)
        if not v1:
            continue
        sources = {"gt_v1": v1, "gt_v2": base.yolo_band_boxes(base.V2_DIR / "labels" / f"{stem}.txt", 256, 256)}
        for run in base.PRED_RUNS:
            sources[f"pred_{run}"] = [b for b in (base.xyxy_box(*xy, 256, 256) for xy in preds[name][run]) if b]
        for src, boxes in sources.items():
            out[src]["boxes"] += len(boxes)
            for i, j in candidate_pairs(boxes, None, cfg):
                a, b = (boxes[i], boxes[j]) if boxes[i].i0 <= boxes[j].i0 else (boxes[j], boxes[i])
                gap = b.i0 - a.i1 - 1
                key = "gap_lt0" if gap < 0 else ("gap_0_8" if gap <= 8 else "gap_9_32")
                out[src][key] += 1
    print(json.dumps({k: dict(v) for k, v in out.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
