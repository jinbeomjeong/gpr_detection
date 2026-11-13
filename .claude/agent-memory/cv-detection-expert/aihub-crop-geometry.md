---
name: aihub-crop-geometry
description: AIHub B-scan crops are object-centered (hyperbola never touches crop edge) and overlap within session/channel — matters for tiling real lines and for test beds
metadata:
  type: project
---

AIHub train_pool crops (crop unit 5,473) are cut around the object: pipe_hyperbola center x p5–p95 116–135 px of 256, min gap to left/right edge 81.7 px, 0/503 touch the edge; cavity center 125.5–131. pipe_band touches the edge in 35.6 %. Within one session+channel (gain stripped) 352 groups have ≥2 crops and 1,196 consecutive pairs overlap in x (median 152 px). Measured 2026-11-13 by `analysis/detection/m31_tiling_stats.py`.

**Why:** a scratch YOLO trained only on centered objects may lose recall for objects near tile edges when real lines are tiled (M3-1/M3-3), and has never seen truncated hyperbolas.

**How to apply:** when choosing tile overlap or judging edge misses, run a horizontal-shift sensitivity eval first (design in `docs/experiments/m31_detection_input_tiling.md` §8-2 D1). Overlapping crops can be stitched into labeled pseudo-lines for merge tests, but pixel identity of overlaps is unverified.
