---
name: aihub-crop-geometry
description: AIHub B-scan crops are object-centered (hyperbola never touches crop edge) and overlap within session/channel — matters for tiling real lines and for test beds
metadata:
  type: project
---

AIHub train_pool crops (crop unit 5,473) are cut around the object: pipe_hyperbola center x p5–p95 116–135 px of 256, min gap to left/right edge 81.7 px, 0/503 touch the edge; cavity center 125.5–131. pipe_band touches the edge in 35.6 %. Within one session+channel (gain stripped) 352 groups have ≥2 crops and 1,196 consecutive pairs overlap in x (median 152 px). Measured 2026-11-13 by `analysis/detection/m31_tiling_stats.py`.

Measured consequence (D1 r2, 2026-11-13, M1-10 YOLO11m last.pt, 5 folds): hyperbola recall@0.25 drops 9–15 %p at ±32 px shift, 58–72 %p at ±64 px, 35 %p when a 64-px grey(128) side strip is added with the object still centred; scale tolerance only ~0.87–1.28x (0.74x → 63 %, 0.5x → 0 %, 1.67x → 53 %). Manhole is robust to shift. Root cause attributed to centred crops + translate 0.1 / scale 0.2.

**Why:** a scratch YOLO trained only on centered objects may lose recall for objects near tile edges when real lines are tiled (M3-1/M3-3), and has never seen truncated hyperbolas.

**How to apply:** when choosing tile overlap or judging edge misses, run a horizontal-shift sensitivity eval first (design in `docs/experiments/m31_detection_input_tiling.md` §8-2 D1). Overlapping crops can be stitched into labeled pseudo-lines for merge tests, but pixel identity of overlaps is unverified.
