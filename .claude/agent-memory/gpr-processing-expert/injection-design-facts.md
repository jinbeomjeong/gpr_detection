---
name: injection-design-facts
description: Real-background injection design levers (inject.py) - band shift via reference_time_ns (pure translation), shift limit acts as model filter, coupling groups, CPU-bound cost, fresh pools, texture vs AGC, r_edge per line, CI method for few IDS zones
metadata:
  type: project
---

Facts found while designing M3-3 E3x (2026-11-20, `docs/processing/m33_e3x_plan.md` v0.3). Re-check against code before use.

- Band/hyperbola vertical position is fixed by the model: `build_response_patch` aligns `reference_time_ns` to the line zero `t_z`. Move a band without touching `src/` by `dataclasses.replace(response, reference_time_ns=ref - dt)` (pure translation, EM-approved). Keep `top_depth_m` and fix the end taper per position from the native response; recomputing the Fresnel taper at the shifted depth mixes taper ratio (0.83 -> 0.70) into the row effect.
- Shift limit is a design constant, not physics (band field independent of shift; 2.5D tail error <= 1.2 dB at 12 P). It works as a model filter: on R3 plan draws, of three-row-eligible (28/47/68 px) draws only 0.125 (4TU) / 0.26 (IDS) also reach 90 px at 8 P, vs 0.70 / 0.77 at 12 P (`analysis/processing/m33_e3x_design_feasibility.json` `band_shift_limit_eligibility`). Check this before picking a limit.
- Coupling: A0 models (M01-M11, M14, M16, M17, M19) are built with `coupling="air"` on 4TU and "ground" on IDS; A1 (M12, M15, M18, K7) are 4TU only; M15 only on 100 ns 4TU records; M13/M20 not loaded (`m32_pipe_responses`).
- R3 injected cost per realization (5 folds): CPU about 0.75 s, GPU about 0.22 s -> CPU-bound; parallel CPU workers are the lever.
- Fresh backgrounds: R2 inferred all 754 4TU lines (zone c' boxes exist). IDS 400 MHz lines outside stage A are the M3-7 domain -> new exposure needs a user decision.
- Texture: r0 per-trace AGC (8/f_c) cancels amplitude-only texture (2.5 dB -> 0.5-0.9 dB effective, coh only -0.01..-0.035). Time jitter sigma 0.2 P survives AGC and reaches AIHub-level coh (~0.2). coh is also driven by contrast.
- Aperture-edge manipulation strength r_edge (edge/apex envelope, dB) depends on the line (dx taper columns, 2.5D timing, P_line wavelet) -> define manipulated / negative-control strata per position, with a gray band between. High-recall models M01/M12/M14/M17 have no visible edge at 12-18 dB.
- Stats (stats-design review): stratified cluster percentile bootstrap at 1-0.05/6 undercovers when IDS has few zones (53 lines in 16 zones, Kish ~6.5) -> family error ~0.09. Use per-stratum bootstrap variance + Satterthwaite t (df from Kish effective clusters); BCa is worse. Partial eta^2 with ~8 positions/model is biased up -> basic (reverse percentile) interval. Add a floor (arm mean >= 2 delta) before declaring "negligible".

Related: [[pseudo-band-detection-facts]], [[synthetic-standalone-detection-facts]]
