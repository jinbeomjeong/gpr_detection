---
name: fractional-shift-texture-facts
description: FFT sub-sample shift loses energy at the even-length Nyquist bin (r_N sin^2(pi delta)); line-grid columns are not band-limited in their own DFT; flat-normalised short-window texture fields extrapolate in tapers; 2-line precheck misses per-line defects
metadata:
  type: project
---

- Circular rfft phase-rotation shift of an even-length column keeps only the real part of the Nyquist bin: E'/E = 1 - r_N sin^2(pi delta) exactly (r_N = |X_{n/2}|^2/n / E). This is a sampling property, not a bug. An "energy conserved" assertion therefore needs this identity (residual ~1e-14), not a fixed tolerance. (E3x injected run 1 2026-11-20 failed on 1e-9 with r_N up to 1.03e-8.)
- Patch columns are effectively band-limited by the Gabor target of the waveform filter, not by the 0.8-Nyquist low-pass (which never acts). The leak in the length-n DFT comes from the hard zero outside the synthetic window [0, window_ns]: mainly the start step (up to 1.8 % A), plus a non-periodic end when the column runs past the record. r_N ~ (s/2)^2/(nE), so r_N <= 1e-6 only catches steps larger than ~11-29 % of max. Never write "already low-passed, so the shift is exact". (EM check `m33_e3x_texture_em_check.md`.)
- Texture field normalised to SD 1 on the flat columns, flat length only ~3.6 kernel SDs. Per-position realisation is mostly a smooth bend or tilt (about 56 % of box variance is linear trend). That comes from the geometry, not from the normalisation. The normalisation pins flat SD (amplifies up to ~12x) and inflates extrapolated |w| in the tapers (up to ~8.5). Effective jitter is ~1.15-1.3x the EM prediction. The EMV Monte Carlo normalisation basis is unrecorded, so do not attribute it.
- A precheck on 2 lines cannot catch per-line numeric defects: the precheck lines had r_N 1e-17. CPU-only implementation assertions should run on every line before the GPU stage.

**Why:** one sealed main run was lost (3,119 s) to a false premise in a report-only branch.
**How to apply:** when specifying FFT shifts or energy checks, state the parity/Nyquist identity. When designing a precheck, put every CPU-decidable assertion on all lines. Review: `docs/processing/m33_e3x_texture_defect_review.md`. Related: [[injection-design-facts]], [[eval-4tu-verify-method]].
