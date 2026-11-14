---
name: n2a2-perturbation-sensitivity
description: N2-a' AIHub perturbation (M3-2 stage B) sensitivity pattern and S3/S5 eval timing for estimating future perturbation/injection runs
metadata:
  type: project
---

N2-a' YOLO11m on AIHub val crops (2026-11-14, M3-2 stage B): pipe_hyperbola recall is nearly invariant to
polarity inversion, row-median bg removal, AGC mimic, padding (all |Δ| ≤ 1.5 %p mean); pipe_band, manhole,
cavity are the fragile classes (AGC mimic manhole −30 %p, row-median band −11 %p, 255−q flips cavity −35 %p
because it also inverts region brightness).

**Why:** hyperbola is the selection metric of M3-2; band/manhole are where image-recipe changes actually bite.
**How to apply:** do not expect AIHub-side perturbation runs to discriminate recipes on hyperbola; read
polarity sensitivity from hyperbola only. Hyperbola GT is only 45–147 per fold (1 object = 0.7–2.2 %p).

Timing: S3 predict path one image per call, ~14 s per condition per ~1,000 crops (15 cond × 5,473 crops
= 1,325 s wall), VRAM reserved 372 MB. D3 real-line inference ~66 tiles/s unbatched — batch for injection runs
([[yolo11m-timing-and-fp]]).
