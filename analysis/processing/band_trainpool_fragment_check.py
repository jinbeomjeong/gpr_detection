"""Why fragment_of_group went 6 -> 8 on the AIHub train_pool band check after band spec v1.4.

Same 40 crops as scripts/check_band_picks_aihub.py (seed 0, imported read-only). For every crop
with >= 2 band boxes, list the merge-candidate pairs (spec 3-4 item 1) at G 8 and G 32 with their
gap, the v1.4 pick difference (pair_pick_difference, px and / P_px), the tolerance and whether the
depth stage merged them; also results / fragment_of_group counts at G 8 vs G 32.
No file names or box coordinates are printed (AIHub card rule). stdout last line: JSON.

Example:
    conda run -n gpr-dev python -I analysis/processing/band_trainpool_fragment_check.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import check_band_picks_aihub as chk  # noqa: E402  (read-only reuse of the sampler and box reader)


def main() -> int:
    from gpr_detection.depth import band as B
    from gpr_detection.depth.config import BandConfig
    from gpr_detection.io._common import read_grayscale_file
    from gpr_detection.io.radargram import Radargram
    from gpr_detection.preprocess.filters import prepare_fitting_branch
    from gpr_detection.preprocess.timezero import find_time_zero

    cfgs = {"G8": replace(BandConfig(), merge_gap_px=8), "G32": replace(BandConfig(), merge_gap_px=32)}
    pairs_out, totals = [], {k: {"results": 0, "fragment_of_group": 0} for k in cfgs}
    for crop_no, name in enumerate(chk.sample_images(40)):
        image = read_grayscale_file(chk.DATASET_DIR / "images" / name)
        h, w = image.shape
        boxes = chk.band_boxes(chk.DATASET_DIR / "labels" / f"{Path(name).stem}.txt", w, h)
        radargram = Radargram(data=image.astype(np.float32), dt_ns=None, dx_m=None, air_gap="unknown",
                              amplitude_processed=False, dataset="aihub", line_id=Path(name).stem)
        branch = prepare_fitting_branch(radargram)
        zero = find_time_zero(radargram, branch.filtered,
                              exclude_traces=B.band_exclude_traces(boxes, w, None))
        for key, cfg in cfgs.items():
            results = B.estimate_band_depths(radargram, boxes, zero, cfg=cfg, branch=branch)
            totals[key]["results"] += len(results)
            totals[key]["fragment_of_group"] += sum("fragment_of_group" in r.flags for r in results)
        if len(boxes) < 2:
            continue
        context = B._context(radargram, zero, branch)
        picks = B._fragment_picks(context, boxes, cfgs["G32"])
        tol = B.pixel_limits(context.period_px).merge_tolerance
        groups = {k: B._merged_groups(context, boxes, cfg, picks) for k, cfg in cfgs.items()}
        for i, j in B.candidate_pairs(boxes, None, cfgs["G32"]):
            a, b = (i, j) if (boxes[i].i0, i) <= (boxes[j].i0, j) else (j, i)
            gap = boxes[b].i0 - boxes[a].i1 - 1
            diff = B.pair_pick_difference(a, b, boxes, picks)
            merged = {k: any(a in g and b in g for g in gs) for k, gs in groups.items()}
            pairs_out.append({"crop": crop_no, "gap": gap,
                              "overlap_rule": gap < 0 and min(boxes[a].i1, boxes[b].i1)
                              - max(boxes[a].i0, boxes[b].i0) + 1 >= B.OVERLAP_MIN_TRACES,
                              "diff_px": None if math.isnan(diff) else round(diff, 2),
                              "diff_P": None if math.isnan(diff) or not context.period_px
                              else round(diff / context.period_px, 2),
                              "tol_px": round(tol, 2), "merged_G8": merged["G8"], "merged_G32": merged["G32"]})
    for p in pairs_out:
        print(p)
    print(json.dumps({"status": "ok", "totals": totals, "pairs": len(pairs_out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
