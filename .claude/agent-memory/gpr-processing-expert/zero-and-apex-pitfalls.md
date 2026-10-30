---
name: zero-and-apex-pitfalls
description: Time-zero/apex picking traps verified on gprMax synthetics (h=0.1 merged events, argmax|mean| picks direct wave, global-max apex fails on water pipes, mean vs median background, boxcar dewow ghost lobes, polarity-dependent two-event split, window QC event tracking, surface-event interference bias, weak Z1)
metadata:
  type: project
---

Verified 2026-10-29 with analysis/processing/zero_reference_check.py (results in docs/processing/zero_reference_*.csv, spec docs/processing/spec_depth_v1.md §12).

- Air gap h=0.1 m: direct wave and surface reflection are ~0.45 ns apart -> merge into ONE envelope peak at the direct-wave time (M12/M13/M15/M18). Depth then +1.8..+3.6 cm deep. h=0.3 (M20) separates (gap 1.9 ns). 4TU does NOT look like M20: see [[fourtu-zero-band]].
- Blind "late event" on mean trace sits +0.18 ns after the true surface reflection (direct-wave tail overlap) -> -0.9 cm. Acceptable.
- 4TU-style rule argmax|mean trace| picked the DIRECT wave in M20 (+9.6 cm) and is 0.27 ns before the envelope peak in M02 (phase mismatch). Treat meta t0 as comparison only.
- Apex candidate must be the FIRST envelope peak >= 0.3*box max, not per-trace global max: global max picked the water-pipe bottom/internal echo in M18 (2.17 m vs 1.0 m).
- Mean-trace background removal delays apex +0.022 ns; median trace ~0.
- Straight-ray curvature fit on air-coupled data has small RMS (0.044 ns) yet v'/v 1.82 -> residuals cannot detect the failure; must be forbidden by rule. Refraction-model eps grid recovered eps 9.00 on M20.
- Ground-coupled point fit (R=0) can look "right" (+1.8 cm on M02) only because v over (+9.8 %) cancels shallow direct-wave zero (-3.3 cm).
- (2026-10-30 M2-1 review) Boxcar dewow (T_w = 3/f_c) makes envelope ghost lobes at +-T_w/2 (5.5-6.9 % on gprMax, 11.7 % with a slow non-zero-mean tail) -> can pass the 10 % event threshold and become a fake "late" surface event (~+3 ns, ~-15 cm). Gaussian dewow sigma = 1/f_c has the same -3 dB (0.25 f_c) and no ghosts. spec v1.1 switched.
- Bandpass/dewow move the M20 late (surface) event: raw +0.18, boxcar +0.23, Gaussian/no-dewow +0.29 ns vs json. Do not tune R2 on raw-data numbers.
- Two-event separation is polarity-dependent (analytic Ricker after (1)-(3)): opposite polarity splits at gap >= 0.75/f_c (h_sep = 0.375 c/f_c, 0.22 m @500 MHz); same polarity re-merges at 0.65-1.0/f_c. "1.2/f_c, h<=0.15 m" was wrong.
- Window QC must track the selected event, not "last peak" (ground/single zero is the first peak).
- (2026-10-30, analysis/processing/surface_zero_shift.py) The late-event delay is pure direct-wave INTERFERENCE: an isolated surface reflection moves <=0.003 ns through (1)-(3) (zero-phase). Opposite-polarity envelopes repel; bandpass roughly doubles it (analytic h0.3: raw +0.055 -> +0.12; M20 +0.18 -> +0.29), dewow adds only ~0.01. Bias depends on gap/amp ratio/wavelet and flips sign near the split limit (h 0.25, amp -3: -0.08) -> no constant correction; test windows must be bias-aware, not "processing-redefined" (processed isolated peak == geometric truth).
- (2026-10-30 M2-3) "Earliest first peak >= 0.3 Emax" apex candidate is a min over ~100 traces -> one broadband-noise exceedance wins. No-scale branch (DC+median only) put the apex candidate at the box top (-28 px) at SNR 10-20 dB; trace smoothing (3/5) did not help; a band-pass in pixel units around the data's dominant period (f_px from mean |FFT|) fixed it (<=1.1 px). Spec v1.3 `bandpass_px`.
- (2026-10-30 M2-4) Refraction-QC eps is dominated by the antenna height: exact ray picks h 0.300/0.329/0.349 -> eps 9.00/11.08/13.0 (~+8 %/cm); a -0.12 ns zero shift moves it only -0.1. h_est carries the one-sided interference bias (+0.03 analytic, +0.05 M20), so single-value "eps within 30 %" gives false meta_eps_suspect. Spec v1.4: interval QC over [h_est-0.06, h_est].
- (2026-10-30 M2-5) v1 "refraction QC eps 9.00 on M20" was a cancellation: it used h_est 0.330 (+0.03). With json h 0.3 the QC gives eps 7.7-8.0 for every background/filter/zero variant, because gprMax envelope picks arrive relatively earlier on the wings (pick - ray: +0.018 ns at apex, -0.093 ns at 0.7-0.9 m). Model itself is exact (reflection_times == json ray time). Judge QC algorithm on json ray times, not on picks.
- Zero-phase band-pass moves the envelope peak of an asymmetric (dispersed) wavelet: M15 (eps 16, d 1.5 m, json lag +0.237) apex -0.055 ns from the band-pass alone; median bg only -0.007. R-checks against raw json envelope peaks need a processing-consistent reference.
- Robust fit covariance from all raw residuals turns outlier size into sigma_v (2.69 % vs 0.05 % inliers-only at 20 % outliers); use inliers.
- Z1 tail 0.6 does NOT discriminate box vs Gaussian dewow with current thresholds (box ghost 10.0 % height, just under); tail 0.8 does (box 2 events at ~+1.9 ns, Gaussian 1).

**Why:** these are non-obvious failure modes that recur when verifying python-expert implementations.
**How to apply:** check them first in verify/review of src/gpr_detection/preprocess/timezero.py and depth/picks.py; see [[spec-depth-v1]].
