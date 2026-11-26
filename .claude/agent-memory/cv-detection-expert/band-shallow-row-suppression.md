---
name: band-shallow-row-suppression
description: N2-a' pipe_band misses flat bands in shallow px rows (y1 < ~37 px); px-tied (E3x R-b), not period-scaled zero/AGC proximity; hyperbolas fine at shallow rows
metadata:
  type: project
---

N2-a' YOLO11m suppresses flat bands whose top row is shallow (E3x HB1 confirmed 2026-11-26: r0 28 px coverage 0.02-0.04 vs 0.22-0.34 at 47/68 px, no_box at conf 0.001 ~58 %).

- Training side: train_pool lpipe box y1 min 36.9 px (0.05 % < 37). After N2-a' geometry (top-anchored scale 0.55-1.5 + translate 0.1) ~10 % < 37, density at 28 px ~1/3 of 47/68 px; observed recall ratio is steeper (0.07-0.18), so an extra learned negative is likely (AIHub NM crops carry unlabeled direct-wave ringing in rows 8-30).
- R-b (P* 4.2) moved the drop with px rows while period geometry was unchanged -> px-tied; but a non-constant scale penalty exists (p42 48 px 0.15 vs r0 47 px 0.38), so I_px is "px component exists", not its size.
- Hyperbola apex < 28 px still Y_on 0.73: suppression is class/shape-specific, not image-level masking near the zero band.
- In r0 the zero row sits at image row ~0-3, so "px from top" and "px below zero band" cannot be separated with r0/R-b.

**Why:** recurring FN pattern; any fix (shallow band labels / synthetic pseudo-band training) will trade against FP on the real zero band.
**How to apply:** in M4 band work, gate shallow-row recall together with zero-band-row FP; script analysis/detection/m33_e3x_hb1_mechanism.py. Related: [[n2a2-perturbation-sensitivity]], [[m33-results-patterns]].
