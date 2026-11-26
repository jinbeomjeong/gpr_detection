---
name: e3x-judgment-facts
description: M3-3 E3x judgment (2026-11-26) reusable facts - on/off arm near-identity shrinks HH1 CI far below simulated MDE, M* floor drops strongest-cut models, pandas cluster_id float trap, judge bootstrap reproducible only with the sealed judge's draw form (rng.integers, not rule text)
metadata:
  type: project
---

- Paired arms that differ only in a sub-sigma edge (HH1 taper on/off) give identical hits at most positions (61/80 D_p = 0). The stats simulator assumed arm noise correlation <= 0.5, so it predicted CI half-width ~0.06-0.07; the real one was 0.011. **Why:** simulated MDE/`negligible` probability for paired image manipulations is pessimistic when the manipulation barely changes the image. **How to apply:** when reading MDE reports for near-identical arms, expect much narrower CIs; say so in the judgment as interpretation, not as a rule change.
- M* (model mean >= 0.10) removed M04/M16, the models where the abrupt cut is most visible (r_edge about -1..+0.7 dB). An HH1 "negligible" then covers r_edge -15..-2.7 dB models only. **How to apply:** state the M*_e scope next to any edge-treatment conclusion.
- pandas `read_csv` turns cluster ids like `01.10`/`010.17` into floats (collisions `01.1` = `01.10`). Always pass `dtype={"cluster_id": str, "line_id": str}`.
- The sealed E3x judge bootstrap is reproducible once the sealed judge's draw form is known (not from the rule text alone): `default_rng(20261121)`, fixed order HH1 -> neg control -> HH2 -> HH3 -> HB1 -> HB1b -> HB2, per background 4tu then gz_ids `rng.integers(0, G, (B, G))` over sorted cluster ids (R-g reuses replicates). The plan (`m33_e3x_plan.md:400`) fixes only seed, B, (B, G_g) shape and calls the order "implementation-fixed"; the `integers` draw form comes from the sealed judge code (verif audit `m33_e3x_judgment_audit.md` finding 6). **How to apply:** when claiming reproducibility of a sealed bootstrap, cite the judge blob, not the rule text. Re-check script: `analysis/processing/m33_e3x_judgment_recheck.py`.

Related: [[apex-candidate-qc-facts]], [[pseudo-band-detection-facts]], [[band-depth-pitfalls]]
