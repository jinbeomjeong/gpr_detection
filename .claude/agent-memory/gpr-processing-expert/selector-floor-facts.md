---
name: selector-floor-facts
description: M2-10 (b) apex-candidate selector detection floor 0.05 max(level) - zero coupling facts, 16 no-peak deep truths mostly NOT floor, B_full exact decoupling, B_min floods under non-stationary noise, gprMax harness needs dt 0.098 resample + noise
metadata:
  type: project
---

Measured 2026-11-11 (docs/processing/m210b_selector_floor.md, analysis/processing/m210b_floor_*.py; 4TU = already-seen, exploration).

- Floor coupling is structural: S and E do not depend on t_z, only k_lo does; earlier zero -> superset rows -> L = max level can only rise. max level sits in the first period below the mute on 690/759 4TU lines, so the floor is set by zero-band tail / near-surface clutter. Floor-held rows ~58 % of the search region (first at nominal ~0.95 m): there the test is the absolute E >= 0.1 L (20 dB below the strongest median clutter).
- The 16 "no peak" deep truths of m210d are mostly NOT the floor: 1 floor (peak only, never accepted), 6 global num_peaks=300 cap on long lines (site 10, 011.7), 8 noise-level (window max < 5 sigma_n), 1 definition artefact. m210d "peak" depth is AFTER the crest move (m210d_generator_stages.py 177-183) - compare peak depths with that in mind.
- Rule facts: A_skip1 (skip one period) keeps the coupling (4TU two-event loss 1230 vs current 937). B_full (floor 2.5 sigma_n, MAD over the whole record S[0:k_hi]) is exactly zero-independent (0 lost) but line MAD includes shallow clutter -> 13 % of current kept candidates are below its 5 sigma (deep, local depth-path sigma is smaller). B_min (min of row-wise MAD) is monotone-safe but 4TU sigma_min/sigma median 0.107 -> peaks x2; floods under gain-like non-stationary noise (proto P7b 177 vs 17). Current rule floods pure noise (~285 peaks/line) unless strong shallow clutter sets the floor.
- Prototype apex hit must allow the crest move: constant-amplitude hyperbolas tie on plateaus; use d/r wing decay and "peak -> crest_apex -> accept within +-3 traces, +-0.5 period".
- gprMax S4/S6 for the selector: resample to 0.098 ns (min_distance 8, trim 16, smoothing are in samples) and add noise (rho = apex envelope / sigma_n 10, 20), same sigma_raw and seeds for S4N/S6P pairs; noise-free synthetic puts every deep row on the floor.

- Judge a floor rule at the ACCEPTED stage (before ranking/NMS/cap 6): the floor only acts on peaks, while S1g showed kept-stage losses from early-lobe wins (lobe 1.02 ns early, higher detection SNR, NMS kills the true apex) and the cap 6; a later mute can also free cap slots and fake a Delta dependence. EM pilot apex/max-clutter 0.136 gives Lambda = L/A0 <= 7.4 < 10, so S6P at Delta 0 is not in the floor-loss regime; an earlier mute (Delta -1 ns) mirrors the 4TU defect.

**Why:** these decide M2-10 (b) rule choice and how to read selector-stage losses; re-deriving needs the 5-min 759-line diagnosis.
**How to apply:** before blaming the floor for a deep loss, check num_peaks cap (long lines) and 5 sigma energy; judge floor rules on synthetic F1-F5/F4b (doc 4-2), not on 4TU counts. Related: [[fourtu-zero-band]], [[fourtu-eval-baseline-and-failure-pattern]].
