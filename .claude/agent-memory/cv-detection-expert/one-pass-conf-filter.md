---
name: one-pass-conf-filter
description: When a conf 0.001 inference filtered to score > 0.25 equals the conf 0.25 path (Ultralytics 8.4.192 NMS + project tile merge), and the traps
metadata:
  type: project
---

One inference at conf 0.001, then `score > 0.25` on **pre-merge tile boxes**, equals the conf 0.25 path (checked 2026-11-20 for M3-3 E3x, code reading only).

- Holds because Ultralytics `utils/nms.py` is class-offset greedy NMS (a box is suppressed only by a higher-score same-class box), `max_det` 300 keeps the top by score, `max_nms` 30000 never binds at 256 px input (1344 anchors), and the threshold is a strict `>` on float32 (0.25 is exact).
- Traps:
  - Filtering **after** `merge_line_boxes` is not equivalent. `_representative_key` ranks truncated/owned before score, low-score boxes bridge R2 components, and low-score bands clip others in `_arbitrate_bands`.
  - R3 `p4.evaluate_unit` and `match_units` filter > 0.25 post-merge. That is harmless at conf 0.25 inference but wrong at 0.001.
  - NMS `time_limit = 2 + 0.05*bs` s (5.2 s at batch 64) silently empties the later images of a batch. It logs a WARNING, which `quiet_ultralytics_logger` does not hide.
  - Exact score ties under an unstable CUDA sort can differ with the candidate set size.
  - A bit-identity reference must use the same `run_jobs` jobs and batch composition. Another batch size changes the last bits.

**How to apply:** for any "one pass, several conf levels" design, require the pre-merge filter, a time-limit warning counter as a defect, and a same-batch conf 0.25 reference in precheck.
