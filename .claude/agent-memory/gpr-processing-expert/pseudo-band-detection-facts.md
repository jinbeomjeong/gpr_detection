---
name: pseudo-band-detection-facts
description: Why injected pseudo bands (R3) are rarely called pipe_band by the N2-a' detector - placement (length, L1 row/depth) not SNR scale, matching or class confusion; period-change confound trap; contrast metric in detector-input space
metadata:
  type: project
---

From the M3-3 pre-close item 2 review (2026-11-20, docs/processing/m33_r3_band_recall_review.md, scripts analysis/processing/m33_r3_recall_diag.py, m33_r3_band_contrast.py):

- R3 pseudo-band recall (P4 6.5 %) is NOT matching (any-row overlap gives the same 0.0653), NOT class confusion (any-class 0.0657), NOT median erasure for L <= 100 (survival p50 0.93). The detector draws almost nothing at the band (boxes touch 317/5,440 truth x fold).
- Strong association with placement: AIHub lpipe labels have top row >= 36.9 px (p5/p50/p95 46/57/79) and width p5 ~12.4 P (P_AIHub 4.2 px). R3 put 59 % of P4 positions above 37 px and 63 % shorter; long & deep positions (20 %) gave 68 % of covered length, 18 dB 42 % vs 2.3 % for short & shallow. Model differences (M01/M17 = 0) were depth, not waveform. Un-injected detector band boxes sit at 10 P (IDS) / 13.7 P (4TU) below zero.
- SNR scale is not the binding cause: at 12-18 dB the band's lateral L1 contrast (4.1/7.5 dB) already exceeds AIHub labels (1.2 dB) and detector-found real bands (0.6 dB); AIHub-level contrast (0-6 dB injections) gives ~0 recall. Bands show no 18 dB drop (unlike hyperbolas). Injected bands are far flatter (coh 0.90 vs AIHub 0.28) - untested OOD hypothesis.
- L 150 at W 301 = median tipping point: ~0.6 A survives plus -0.42 A inverted ghost flanks -> low recall; keep it as a separate "weak band" case.
- Period-change trap: realizations whose L1 period moved > 5 % sit at low-recall positions and mostly at 18 dB; the naive comparison (1.7 vs 6.9 %) is confounded. Compare polarity twins at the same position and SNR (2.3 vs 1.4 %, no harm).
- R3 runner keeps only conf > 0.25 boxes, so "seen weakly" cannot be separated from "not seen" - ask for conf-0.001 top score per truth in new runs.

**Why:** pseudo-band recall numbers look like a detector failure on longitudinal pipes but mostly encode the placement distribution.
**How to apply:** for any injection-based recall (bands or hyperbolas), fix vertical position (L1 row / depth in P) and length as design factors, compare within position, and measure contrast in the detector-input grey image before blaming SNR. Related: [[synthetic-standalone-detection-facts]], [[detection-image-scale-facts]], [[band-depth-pitfalls]].
