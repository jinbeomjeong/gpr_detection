---
name: fourtu-zero-band
description: "4TU field facts (2026-10-29, M2-7) — single merged zero band, meta t0 est/act are +/- lobes not direct/surface, hyperbola shape implies h <~ 0.1 m, eps_r_used often = header default 9"
metadata:
  type: project
---

Measured on 4TU activities 01.1, 04.1, 010.14, 011.14 with analysis/processing/fourtu_explore.py (doc docs/processing/fourtu_explore.md).

- Zero band is ONE envelope peak in 41/41 windows (no separate direct wave / surface reflection) -> h cannot come from event gap.
- `t0_sample_est` vs `t0_sample_act_median` flips (e.g. 27 vs 39) are the + and - lobes of the SAME band (gap 1.2-1.4 ns, per-trace corr 0.87-0.95), not direct vs surface. Whole index |est-act| clusters at 1.0-1.5 ns, almost none near 1.95 ns. Lobe-vs-envelope phase offset is only -0.7..+0.6 ns (~ +-3.5 cm).
- Discriminator that worked: shift-coherence test (align traces by k * band shift, k=0 stationary / 1 surface / 2 multiple). Main lobes follow the surface (k=1); weak leading lobe (~-1.9 ns) slightly prefers k=0 (direct coupling).
- Refraction QC: straight-ray eps of good 01.1 apexes ~8.7 (= meta 9); any h >= 0.1 m forces eps -> >40. RMS min at h <= 0.1 m for 22/27 apexes. Earlier "4TU h ~0.29 m" premise (depth_physics §7, spec §12) is unsupported.
- eps_r_used == eps_r_header (device default 9) in sites 01/04/010 -> likely not measured. Site 11 (dt 0.195, samples 0-47 muted) eps 15.18 gives apex depths ~0.22 m shallower than GT.
- Per-line Hungarian with dense GT sets (010.14: 19 pipes) is very lenient; always add a random-depth baseline.

**Why:** these overturn assumptions written into spec_depth_v1 and the agent definition; next 4TU work (M2-8) must not re-derive them.
**How to apply:** treat 4TU as `air_single_event` with zero = band envelope peak; label meta t0 as band lobes; recheck h/eps on more activities before relaxing the curvature-fit ban. Related: [[zero-and-apex-pitfalls]].
