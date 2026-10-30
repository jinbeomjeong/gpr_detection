---
name: zero-and-apex-pitfalls
description: Time-zero/apex picking traps verified on gprMax synthetics (h=0.1 merged events, argmax|mean| picks direct wave, global-max apex fails on water pipes, mean vs median background, boxcar dewow ghost lobes, polarity-dependent two-event split, window QC event tracking)
metadata:
  type: project
---

Verified 2026-10-29 with analysis/processing/zero_reference_check.py (results in docs/processing/zero_reference_*.csv, spec docs/processing/spec_depth_v1.md §12).

- Air gap h=0.1 m: direct wave and surface reflection are ~0.45 ns apart -> merge into ONE envelope peak at the direct-wave time (M12/M13/M15/M18). Depth then +1.8..+3.6 cm deep. h=0.3 (M20) separates (gap 1.9 ns). 4TU does NOT look like M20: see [[fourtu-zero-band]].
- Blind "late event" on mean trace sits +0.18 ns after the true surface reflection (direct-wave tail overlap) -> -0.9 cm. Acceptable.
- 4TU-style rule argmax|mean trace| picked the DIRECT wave in M20 (+9.6 cm) and is 0.27 ns before the envelope peak in M02 (phase mismatch). Treat meta t0 as comparison only.
- Apex seed must be the FIRST envelope peak >= 0.3*box max, not per-trace global max: global max picked the water-pipe bottom/internal echo in M18 (2.17 m vs 1.0 m).
- Mean-trace background removal delays apex +0.022 ns; median trace ~0.
- Straight-ray curvature fit on air-coupled data has small RMS (0.044 ns) yet v'/v 1.82 -> residuals cannot detect the failure; must be forbidden by rule. Refraction-model eps grid recovered eps 9.00 on M20.
- Ground-coupled point fit (R=0) can look "right" (+1.8 cm on M02) only because v over (+9.8 %) cancels shallow direct-wave zero (-3.3 cm).
- (2026-10-30 M2-1 review) Boxcar dewow (T_w = 3/f_c) makes envelope ghost lobes at +-T_w/2 (5.5-6.9 % on gprMax, 11.7 % with a slow non-zero-mean tail) -> can pass the 10 % event threshold and become a fake "late" surface event (~+3 ns, ~-15 cm). Gaussian dewow sigma = 1/f_c has the same -3 dB (0.25 f_c) and no ghosts. spec v1.1 switched.
- Bandpass/dewow move the M20 late (surface) event: raw +0.18, boxcar +0.23, Gaussian/no-dewow +0.29 ns vs json. Do not tune R2 on raw-data numbers.
- Two-event separation is polarity-dependent (analytic Ricker after (1)-(3)): opposite polarity splits at gap >= 0.75/f_c (h_sep = 0.375 c/f_c, 0.22 m @500 MHz); same polarity re-merges at 0.65-1.0/f_c. "1.2/f_c, h<=0.15 m" was wrong.
- Window QC must track the selected event, not "last peak" (ground/single zero is the first peak).

**Why:** these are non-obvious failure modes that recur when verifying python-expert implementations.
**How to apply:** check them first in verify/review of src/gpr_detection/preprocess/timezero.py and depth/picks.py; see [[spec-depth-v1]].
