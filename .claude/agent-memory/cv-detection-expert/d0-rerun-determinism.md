---
name: d0-rerun-determinism
description: M3-3 timing facts — D0 full run ~5 min and bit-identical on rerun; R3 band smoke 3 IDS lines x fold 0 took 544 s (spec's 20-30 min R3 estimate was far off)
metadata:
  type: project
---

D0 gate executor (`m33_d0_gate.py`, N2-a′ 5 fold, batch 64, RTX 4070 SUPER) full run: 295 s, 16,919 images (2026-11-15 D0′). Re-run with the revised executor reproduced every C0b gate value bit-for-bit (183 cells, float repr identical) — no GPU nondeterminism observed on the same GPU, driver and package versions.

R3 band runner smoke (2026-11-16): 1st 544 s (3 IDS lines, fold 0); 2nd timed smoke after fixes 374 s = uninjected 7.7 s, plan 269 s (CPU), injected 69 s (GPU inference only 5.8 s). Full R3 upper bound ~1.9 h, dominated by plan CPU (~83 min). r0 has three line-global statistics (grey clip 99th pct of |a|, period from mean rfft of step-(4) section, AGC floor = line max power x 1e-12), so injected realizations must be inferred over the whole line — a local "median window" tile range is wrong.

**Why:** strict-vs-wide tolerance debate before D0′ assumed GPU nondeterminism might occur; it did not. The original R3 estimate (20-30 min) was written before any band timing existed.

**How to apply:** for re-runs on the same machine/env, expect exact equality; numeric drift points to a code/input change. For injection runners, time a stage-split smoke before quoting run time; do not reuse hyperbola-era estimates.
