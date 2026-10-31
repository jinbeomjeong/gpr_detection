---
name: band-depth-pitfalls
description: pipe_band (parallel pipe) depth pitfalls - median background erases long bands, band = apex-trace equivalence, oblique-crossing curvature bias, no parallel-pipe truth in 4TU/2D gprMax
metadata:
  type: project
---

Facts found while writing the pipe_band depth spec (2026-10-30, M2-6):

- Moving-median background removal erases a horizontal band longer than half the window. With a band over the whole profile, only 0.7-1.6 % of the amplitude is left and depth moves -8 to -30 cm (2D gprMax pseudo-band). The band branch must use a masked median (traces outside all band boxes) or no background removal. The same applies to the detection-branch image: long bands can vanish there too.
- A pipe parallel to the line and directly below it gives the apex-trace response on every trace. A pseudo-band (apex trace tiled + `<id>_bg.npy` no-pipe traces) reproduces apex-rule depth within 0.35 cm, so the 2D set can regress band rules without new gprMax runs. 2D vs 3D partly confirmed (band_top_synthetic_v1.md 2-3, 10): weak layer 1 P above, wet bed, plates match within <= 0.12 ns and the same pick class; large-ratio PVC and other geometries still unchecked.
- A hyperbola from an oblique crossing (angle theta) gives curvature v and depth both too large by 1/sin(theta): +6.4 % at 70 deg, +15.5 % at 60 deg. spec_depth_v1 does not mention this. It matters whenever a neighbor-hyperbola v is borrowed.
- A lateral offset y between the line and the pipe axis makes the band deeper by about y^2/(2 d_c). This cannot be observed from a single line.
- 4TU lines are designed perpendicular to the utilities and its truth comes from trench cross-sections, so parallel-pipe truth is essentially absent. Candidates for a look: the 11 activities flagged as crossing utilities. 2D TMz cannot model parallel pipes; 3D is required.
- (2026-10-30 M2-6b) AIHub 256x256 JPG crops: the vertical dominant period is ~4.2 px (2.44-4.83) = the radar carrier (vertical only, same inside band boxes, no JPEG 8x8 lines), band boxes span 4-5 cycles. So "2 px" fixed thresholds were ~0.5 period, not tiny. AIHub band picks scatter 0.25-0.5 period; do not tune thresholds on AIHub. Check with analysis/processing/aihub_period_px_check.py before assuming a px scale.
- Shallow bands inside the zero search window are mistaken for the surface reflection by the air-coupled "late event" rule. Exclude band-box traces from the mean trace.

- (2026-10-31 box convention check, docs/processing/band_box_convention_depth.md) AIHub bands are stacked layers; the band-top pick sits ~0.7 P below the box top (p10 -1 P), so depth follows the box TOP edge more than split/merge. A merged group uses the union box (min k0) window for every trace: traces whose own fragment top is >=5 px lower jump to an earlier layer 39 % of the time. Label-level merge (v2 auto-32) skips the facing-end check: 46 % of its merged pairs fail it and cannot be split back. Detector boxes give mostly x-OVERLAPPING merge candidates (180-352 pairs vs 4 in GT); the facing-end check compares non-shared traces there. Measure P-normalized (P_px ~4.3 px; 1 P ~ 10 cm at 500 MHz eps 9).

- (2026-10-31 gprMax band-top judgement, band_box_convention_depth.md 8) Current pick (box top - 0.4P pad - 1P) takes any layer within ~1.4 P above the pipe (Ricker) / ~2 P above, ~1.5 P below (ringing); only a lateral (masked) background that sees the layer outside the band fixes it, label top placement cannot. S3 per-fragment windows never worse but +0..6 %p < +10 %p rule -> not adopted. Steel protective plate: band = plate (2D and 3D), depth shallower by plate height (0.1-0.3 m). AIHub wavelet is ringing-like (spectral half-amp rel. bandwidth 0.33 vs Ricker 0.92 / ring 0.34) -> use ringing results for AIHub policy. Judge with decimated gprMax traces (5089 = 7 x 727 samples, a large prime factor, so the FFT of the zero-padded 10178-sample trace is very slow) and zero-padded Hilbert.

**Why:** these are not obvious from spec_depth_v1, and they change which preprocessing and validation are valid for pipe_band.
**How to apply:** when reviewing or verifying `depth/band.py` (M2-6b) or the detection preprocessing for scaled data, check bg mode, zero exclusion mask, and v source. Related: [[zero-and-apex-pitfalls]], [[fourtu-zero-band]].
