---
name: detection-image-scale-facts
description: L1 detection-image scale facts (M3-1, 2026-11-13) - measure carrier period after bg removal, AIHub ~isotropic in depth units, native aspect makes pipes look like cavity/manhole, GPR-SD 51-sample period conflict, letterbox upscales
metadata:
  type: project
---

Facts from the L1 spec draft (docs/processing/spec_detection_image_v1.md, scripts analysis/processing/detection_image_{scale_survey,period_check}.py):

- Carrier period must be measured on the background-removed (step 4) section with the AIHub estimator (trace-mean |rfft| peak). Raw DC-removed data is biased long: 4TU 3.1 vs 2.51 ns (+24 %), Guangzhou 400 MHz up to 2.5x; the Guangzhou header `spectrum_peak_MHz` is unusable (p10-p90 2.9-14 ns). 4TU P = 2.51 ns both whole-line and at true apexes (matches +-lobe spacing 1.2-1.4 ns = half period).
- AIHub tpipe apex crops P_px p50 4.06 (band crops 4.20). tpipe width = 11.4 P; 4TU/Guangzhou hyperbolas are 8-15 P wide, so matching P puts them at AIHub size. AIHub box shape is only consistent with near-isotropic depth units (horizontal compression <= 1.6x) [estimate].
- Native 1 trace = 1 sample images stretch hyperbolas vertically 2-8x vs depth-isotropic -> w/h 0.15-0.69, i.e. AIHub cavity (0.74) / manhole (0.54) shapes. Scale rule: s_y = P_line/P*, s_x = s_y dt v/(2 dx) (45 deg asymptote).
- GPR-SD dominant period ~51 samples (whole line and pipe-mark windows) = 62/73 MHz under dt hyp A/B, contradicting the documented 200-400 MHz; with v dt/2 = 0.012 m it implies ~80 MHz. Hyperbolas only 1-2.6 P wide -> no scale matches AIHub. Exploratory, not a verdict.
- `detect/base.py letterbox` upscales small inputs (ratio = min(256/H, 256/W)); keep scale by padding inside L1 (right/bottom, maps outside the grid -> clipped).
- Overlap lower bound must be capped in periods, not only metres: in depth-isotropic L1, lambda/2 = P* px, and the widest AIHub tpipe box is ~20 P (80.7 px). A 2 a_max = 2 m bound blows up for high-frequency lines (Guangzhou 2000 MHz 325 px). Per-line eps also matters (4TU O_phys 64-96 px uncapped).
- Content height > 256 px does happen at default P*: Guangzhou MALA (dt 0.391 ns, 190-200 ns window = 40-80 periods). MALA measured period scatters 4.45-36.6 samples between folders (processed data), so counts depend on the period assumption (8 vs 75 lines).
- 4TU GT neighbour utility spacing p50 0.20 m (63 % <= 0.30 m) vs Fresnel radius 0.19-0.43 m: most neighbours are one GPR event anyway.

- (2026-11-14 M3-2 plan pilot, analysis/processing/m32_{baseline_texture,horizontal_coherence}.py) AIHub crops keep horizontally coherent energy (row-mean share R_h p50 0.23-0.40 by class, top 15 % rows 0.34-0.65) -> AIHub looks not (or weakly) background-removed; 4TU L1 after median bg R_h 0.017, and a longer window does not help (0.019) because most 4TU lines (<=301 traces) already use the whole-line median. AIHub decays with depth (bottom/top RMS 0.30-0.41) while AGC 8/f_c makes 4TU flat (0.91). 20/26 4TU S lines have H_c < 128 -> bottom pad taller than the trained pad band max 128 px; AIHub flat rows p50 1-2 %. AIHub whole-crop period (no band-pass) is 8-14 px (horizontal structure), so compare R2 periods only with identical definitions. 4TU dt 0.195: P_line min 12.15 samples vs compression threshold 11.98 (n_s P*/256.5), scale_out_of_range needs < 8.89 -> no issue. Keep period measurement on the fitting-branch (median 301) section even if the detection bg recipe changes. Full survey (m32_content_height.py, P* 6): 50 % of 4TU+IDS tiles have bottom pad > 128 px; Guangzhou IDS with MEASURED periods has 20 compressed lines but only 1 with c > 1.35 (spec's "2000 MHz 2 + 600 MHz 1" came from 1/f assumed periods). AIHub gain value = inverse contrast factor (gain 1.0 linear); no-bg/no-AGC evidence only for RA/NM equipment (pipes, manhole), MA (cavity) undecidable (curator m32_aihub_processing_facts.md). MALA 8x period spread is processing residue/harmonics, ~250 MHz antenna (EM m32_em_review.md).

**Why:** these decide the L1 scale/tiling rule and are easy to get wrong by measuring on raw data or trusting headers.
**How to apply:** when verifying `DetectionImage` (M3-2) or judging cross-domain detection, re-measure P per line on bg-removed data; never treat L1 curvature as velocity. Related: [[band-depth-pitfalls]], [[fourtu-zero-band]].
