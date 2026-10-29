---
name: zero-and-apex-pitfalls
description: Time-zero/apex picking traps verified on gprMax synthetics (h=0.1 merged events, argmax|mean| picks direct wave, global-max apex fails on water pipes, mean vs median background)
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

**Why:** these are non-obvious failure modes that recur when verifying python-expert implementations.
**How to apply:** check them first in verify/review of src/gpr_detection/preprocess/timezero.py and depth/picks.py; see [[spec-depth-v1]].
