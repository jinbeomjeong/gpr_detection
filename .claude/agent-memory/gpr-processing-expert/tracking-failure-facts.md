---
name: tracking-failure-facts
description: Why 4TU true-apex wing tracking stops one-sided (M2-9 C follow-up 2026-11-07) - reverse rule 0.1 ns vs near-apex jitter, hint-window top truncation, kappa 0.258, prototype method, v1.9 verify facts (descent bound, rise threshold, X->V, unknown as false evidence)
metadata:
  type: project
---

Diagnosis of 15 true apexes with curvature unknown (docs/processing/m29c_track_diagnosis.md; T1+T2 approved 2026-11-07 and written into spec_depth v1.9 - spec_depth is canonical over the diagnosis section 9 table):

- 27/30 failing wings stop on the spec 5-3 reverse rule (pick earlier than previous by > 0.1 ns), 17 within 4 traces of the apex; known-curvature T: 0/52 that close. Box edge never ends a 4TU track (box +-0.88 m, amplitude/reverse end first).
- Near-apex ridge step jitter on 4TU T: robust sigma 0.079 ns, 13 % of steps < -0.1 ns. Zero-band (surface) step sigma 0.0098 ns, corr 0.03 -> NOT timebase jitter; it is event-local interference (flat-topped / doublet envelopes: equal find_peaks maxima 2 samples apart).
- Hint-window top truncation: find_peaks never reports index 0, so when the hint sample is late (manual hints median -0.74 ns vs found apex) the "earliest visible first peak" is a locally LATE trace -> neighbours earlier -> step-1 reverse stop. Signature: candidate <= 2.5 samples below the hint-window top (9/15 unknown vs 4/26 known), apex at the +-3 trace edge.
- One same-wavelet interferer of relative amplitude a shifts the processed envelope peak by up to kappa*a*P, kappa 0.258 (Ricker 0.2581, gprMax M02 0.2563), linear to a 0.5, worst at |gap| ~ 0.45 P. The 0.1 ns rule trips at a ~ 0.1 (-20 dB) while the tracker treats < 0.2 E_a as no event.
- Point-scatterer clutter MC (dx 0.02) does NOT reproduce near-apex jitter stops with hint at the apex (scatterer wings are smooth); it reproduces truncation (hint 0.45 P late: v1.8 fails 10-22 % at clutter 0.1-0.2). Use deterministic alternating +-shifts for jitter tests.
- In-process prototype method (no src edits): replace picks._track_one_side / find_apex_candidate / _hint_window and wrap extract_picks in estimate, eval_depth_synthetic and eval_depth_4tu modules (they import by name). gprMax regression 129 checks ~35 s per variant.
- Prototype descent details that the handoff table left implicit (now spec v1.9 text): skipped without dt (no-scale hint path), time window clipped only to record bounds [1, N-2] - NOT to mute/box top; amplitude floor is r_stop x the PRE-descent candidate. stop_left/right live only on HyperbolaPicks, not DepthResult, so eval csv columns come from the QC re-call of extract_picks (blank with --apex-qc off; apex_descended comes from DepthResult.flags and is always filled).
- Since v1.9 is in src, m29c_track_diagnose/proto/repro.py are v1.8-pinned (SystemExit); reproduce via git worktree f179e23 + PYTHONPATH without -I. m29c_track_jitter.py has no guard but its outputs are v1.8-based.
- v1.9 verify (2026-11-07, pass; analysis/processing/m29c_track_descent_check.py, v1.9-only): descent can never pass the box top (start >= hint top = k_s-0.5P, box top k_s-1.5P, monotone reach 2*hint_traces*b = 1.8 ns - NOT hint_traces*b); only the mute can be crossed, and 4TU review 0/169 (gprMax 0/20) did. Descent has no top-guard: G122 ends inside the 2-sample guard band (verdict unchanged) - record only.
- Median-of-3 reverse rule: sustained-rise detection threshold is tau_rev/2 per trace (0.103 ns, 5.16 ns/m at dx 0.02) ~ v1.8 0.100 ns; T2 only relaxed single-step jitter. Side effect: synthetic X crossings become V picks (later branch both sides), kept out of "in" only by inlier_frac < 0.5 (10 ns/m margin 0.07).
- On auto 4TU candidates curvature-unknown is evidence for false (T 3/63 = 4.8 % < prior 10.8 %); manual unknown 7/14.
- Residual causes not fixable by tracking: apex outside the +-3-trace hint window (T definition allows 0.10 m), first-peak vs window-argmax lobe ambiguity, neighbour-event capture, R=0 fit instability on shallow broad tops.

**Why:** next steps (spec v1.9, implementation verify, E decision, 4TU re-evaluation diagnostics) depend on these mechanisms and numbers.
**How to apply:** when verifying the v1.9 implementation, check D1a/D2/D4 analogues and that hint-at-apex gives zero descent; do not tune kappa or tolerances on 4TU jitter numbers. Related: [[apex-candidate-qc-facts]], [[zero-and-apex-pitfalls]].
