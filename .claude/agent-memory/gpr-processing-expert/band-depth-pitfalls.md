---
name: band-depth-pitfalls
description: pipe_band (parallel pipe) depth pitfalls - median background erases long bands, band = apex-trace equivalence, oblique-crossing curvature bias, no parallel-pipe truth in 4TU/2D gprMax
metadata:
  type: project
---

Facts found while writing the pipe_band depth spec (2026-10-30, M2-6):

- Moving-median background removal erases a horizontal band longer than half the window. With a band over the whole profile, only 0.7-1.6 % of the amplitude is left and depth moves -8 to -30 cm (2D gprMax pseudo-band). The band branch must use a masked median (traces outside all band boxes) or no background removal. The same applies to the detection-branch image: long bands can vanish there too.
- A pipe parallel to the line and directly below it gives the apex-trace response on every trace. A pseudo-band (apex trace tiled + `<id>_bg.npy` no-pipe traces) reproduces apex-rule depth within 0.35 cm, so the 2D set can regress band rules without new gprMax runs. 2D vs 3D waveform equivalence is still unverified.
- A hyperbola from an oblique crossing (angle theta) gives curvature v and depth both too large by 1/sin(theta): +6.4 % at 70 deg, +15.5 % at 60 deg. spec_depth_v1 does not mention this. It matters whenever a neighbor-hyperbola v is borrowed.
- A lateral offset y between the line and the pipe axis makes the band deeper by about y^2/(2 d_c). This cannot be observed from a single line.
- 4TU lines are designed perpendicular to the utilities and its truth comes from trench cross-sections, so parallel-pipe truth is essentially absent. Candidates for a look: the 11 activities flagged as crossing utilities. 2D TMz cannot model parallel pipes; 3D is required.
- Shallow bands inside the zero search window are mistaken for the surface reflection by the air-coupled "late event" rule. Exclude band-box traces from the mean trace.

**Why:** these are not obvious from spec_depth_v1, and they change which preprocessing and validation are valid for pipe_band.
**How to apply:** when reviewing or verifying `depth/band.py` (M2-6b) or the detection preprocessing for scaled data, check bg mode, zero exclusion mask, and v source. Related: [[zero-and-apex-pitfalls]], [[fourtu-zero-band]].
