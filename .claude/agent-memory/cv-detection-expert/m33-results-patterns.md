---
name: m33-results-patterns
description: M3-3 result patterns — N2-a′ does not transfer to Morocco (AP50 ~0.01, hyperbolas read as cavity), IDS pseudo-band recall only 6.5 %, PP-b turns M4b into M4c, n_zero_changed counts every realization
metadata:
  type: project
---

M3-3 main runs (2026-11-16, judgment `docs/experiments/m33_p4p5_judgment.md`):
- Morocco (233 img, 224→256): N2-a′ `pipe_hyperbola` AP50 ≈ 0.008, conf>0.25 recall 0/158 in every fold; true pipes are matched by `cavity` predictions (17–26 per fold) or missed. Folds 0 and 4 had no hyperbola predictions in the utilities+intact subset → AP None (P5 checked on 3/5 folds).
- R3 IDS 400 MHz pseudo-band recall before PP was only 6.5 % (4TU background ≤ 2.3 %), versus ~61 % AIHub crop band recall. Pseudo bands are nearly invisible to the detector; P4 had little to lose.
- R2 4TU: PP S3 removes all M4a (36) and clips all M4b (1,817), but clipped band boxes land in the M4c window, so M4c jumps 519 → 2,292 (+1,773 = exactly the clipped M4b `pipe_band` boxes; the other 44 M4b were hyperbola/cavity). That is a re-category, not new false positives. Summary `pp_actions` counts 754 lines, original orientation only (720: clipped_b 1,817).
- `n_zero_changed` in R3 equals all realizations (2,159) — exact float comparison of re-measured zero; treat as a reporting artefact.

**Why:** these surprised the reviewer and will shape M3-7 (IDS) and any band-depth work.

**How to apply:** do not quote M4c alone as a PP harm metric; when planning pseudo-band tests, check baseline recall first; for Morocco treat the current detector as non-transferring.
