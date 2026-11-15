---
name: d0-rerun-determinism
description: M3-3 D0 gate executor full run takes ~5 min (16,919 images) and reproduced C0b gate values bit-for-bit on the same GPU/env
metadata:
  type: project
---

D0 gate executor (`m33_d0_gate.py`, N2-a′ 5 fold, batch 64, RTX 4070 SUPER) full run: 295 s, 16,919 images (2026-11-15 D0′). Re-run with the revised executor reproduced every C0b gate value bit-for-bit (183 cells, float repr identical) — no GPU nondeterminism observed on the same GPU, driver and package versions.

**Why:** strict-vs-wide tolerance debate before D0′ assumed GPU nondeterminism might occur; it did not.

**How to apply:** for re-runs on the same machine/env, expect exact equality; any numeric drift points to a code/input change rather than GPU noise. Use ~5 min as the D0-scale inference time estimate.
