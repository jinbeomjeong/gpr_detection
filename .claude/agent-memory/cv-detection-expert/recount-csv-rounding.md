---
name: recount-csv-rounding
description: Independent recounts from M3-3 runner CSVs hit 4-dp score / 2-dp box rounding; treat score "0.25" as conf>0.25 and expect boundary category flips; synthetic Y0 apex hit rate ~3/100
metadata:
  type: project
---

M3-3 D0 (2026-11-15) runner CSVs (`predictions.csv`) store score at 4 dp and boxes at 2 dp. In an independent recount, one fold-1 A3 band box printed as `0.25` had a raw score just above 0.25: strict `> 0.25` on the printed value gave G4 S2 69 vs 70 and G2 A3 0.458 vs 0.495 %p. Treating printed 0.25 as confident reproduced all 401 compared items. Box rounding also flips ~7 low-score boxes that sit exactly on PP boundaries (y_m = y_bot, y2 = y_bot ± e, y2 = y_bot − w_c, y_m = y_z + 0.5P).

Also: the conf-0.25 line detector hit only 3/100 synthetic Y0 apexes (gprMax M01–M20 × 5 folds), so G3-type synthetic gates carry little information.

**Why:** These artefacts look like real disagreements on first comparison and can waste an audit pass.
**How to apply:** In R2/J1 recounts, run strict and inclusive-0.25 variants together, use ~0.1 px / 1e-4 %p tolerances for length sums, and attribute residual diffs to boundary rows before you call anything a defect. See [[framework-aug-defaults]] for val/predict differences.
