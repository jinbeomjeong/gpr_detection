---
name: synthetic-standalone-detection-facts
description: Why stand-alone gprMax lines are nearly invisible to the line detector (D0 Y0 3/100; E0-E4 inference shows real-background context, not AGC, is the cause) and why record-cut Y1 lines give manhole false positives; which M01-M20 models are informative
metadata:
  type: project
---

Facts from the 2026-11-15 M3-3 D0 follow-up (docs/processing/m33_y0_detection_diag.md §7; inference runs/eval/m33_y0_diag_infer, summary analysis/processing/m33_y0_infer_summary.py):

- Stand-alone gprMax lines under r0: E0 conf-0.001 best apex score median 0.001 (not "just below 0.25"). Common 16 models x 5 folds hits: stand-alone 0, AGC off/capped 0 (AGC hypothesis REJECTED), centred pad 4, white noise 4-10 (12/18 dB), real 4TU background 37/34 (12/18 dB). The detector reads pipes only inside real-line context (texture + surface band + width). White noise recovers only 1/4-1/3.
- Even with background, 30 dB < 12 dB (37 -> 25): high-SNR AGC saturation, same as M3-2 stage C 18 dB drop. Keep recall-test SNR <= ~18 dB.
- Models: M09, M15, M18 = 0 in every fold/SNR even injected; M03/M04/M16 fold/position dependent (stage C fold-0-only zero was misleading); M10, M14, M17, M19 5/5 folds at 12-18 dB. One injection position per model mixes position and model effects (M01 0 at its E3 position vs 75 % in stage C).
- Cutting a record then re-running r0 (D0 Y1) makes AGC window/record 0.8-1.07 -> per-trace normalisation -> vertical column blocks read as manhole (66 of 75 Y1 boxes). Process the full record, then cut amplitudes.

**Why:** stand-alone synthetic detection counts look like detector failure but carry no information; my first (CPU-only) diagnosis wrongly blamed AGC - only the inference ablation separated it.
**How to apply:** detection-side synthetic tests = injection into real backgrounds at several positions, informative models only, SNR <= 18 dB, log conf-0.001 scores; verify image-regime hypotheses with an ablation before asserting a mechanism. Related: [[detection-image-scale-facts]].
