---
name: synthetic-standalone-detection-facts
description: Why stand-alone gprMax lines are nearly invisible to the r0 line detector (D0 Y0 3/100) and why record-cut Y1 lines give manhole false positives; which M01-M20 models are detectable at all
metadata:
  type: project
---

Facts from the 2026-11-15 M3-3 D0 follow-up (docs/processing/m33_y0_detection_diag.md, scripts analysis/processing/m33_y0_detect_diag.py, m33_y1_artifact_diag.py, m33_y0_vs_stagec_inject.py):

- Stand-alone gprMax lines under r0: no background -> AGC (8/f_c) normalises on the pipe itself -> post-AGC -6 dB box spans the whole line in 20/20 models, ringing lifted 14-37 dB into ghost arcs, apex above p99 clip (saturated). Period/scale are NOT the problem (P_L1 6.0, compression 1).
- Same models injected into real 4TU/IDS backgrounds (M3-2 stage C precheck, r0 fold 0): M14/M17 93 %, M01 75 % at 12 dB vs stand-alone 2/5, 0/5, 1/5 folds. M03, M04, M09, M15, M16, M18 are ~0 % even with background (deep/flat arcs, short eps-4 line) -> exclude or report separately in detection gates. Stage C precheck parts json has per-injection `response_id`, `hit` -> reusable per-model recall table.
- Cutting a record then re-running r0 (D0 Y1) makes AGC window/record 0.8-1.07 -> per-trace normalisation -> vertical column blocks read as manhole (66 of 75 Y1 boxes, mostly near zero). Real 4TU window/record ~0.32. Emulate record-end pipes by processing the full record then cutting amplitudes.
- 2.2 m gprMax lines = 13-29 P wide in L1 -> L-shaped right+bottom pad beyond training bands (14/20).

**Why:** stand-alone synthetic detection counts look like detector failure but are mostly an imaging-regime artefact; reusing them as gate denominators gives no information.
**How to apply:** for detection-side tests use injection into real backgrounds (or add seeded noise first), pick informative models (M01, M14, M17, M10, M11), log conf-0.001 scores. Related: [[detection-image-scale-facts]].
