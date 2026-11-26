---
name: aihub-tpipe-box-habit
description: AIHub tpipe label box habit (top ~0.5 P above apex lobe, one box per crop, wings cut) and why per-box apex picking fails on AIHub crops
metadata:
  type: project
---

AIHub train_pool tpipe boxes (503 boxes = 503 crop units, 9 sessions, measured 2026-11-26 for M3-4 box convention):
exactly ONE pipe_hyperbola box per crop (no overlap pairs at all; adjacent second arcs are merged or left out, seen in ~4/36 zoom tiles);
box top sits ~0.5 P above the first strong centre lobe peak (p25-p75 0.05-0.85 P, ~46 % >= 0.6 P);
height median 6.65 P (bottom ~apex + 6 P, centre envelope ~0.1 of box max at y1);
sides cut the wings while still strong (ridge amp at edge ~0.8 of apex, still ~1.0 16 px beyond), apex arc ~2.5-3.6 P down at the edge;
apex column within +-0.1 w of box centre. No box touches the crop edge.

**Why:** per-box automatic apex picking (envelope first-peak, even with local-median background and an arc-descent test) locks onto the shallow flat clutter above the box in most AIHub crops (gain JPG, shallow pipes) — 79-94 % "apex above box" was an artefact. Box-aligned mean profiles with a short search window [-1, +2.5] P were the usable measure.

**How to apply:** for AIHub label geometry questions use `analysis/detection/m34_aihub_box_profiles.py` (box-aligned profiles), not per-box apex pickers; verify with zoom tiles (`m34_box_zoom_tiles.py --ticks`). AIHub cannot teach multi-pipe/overlap or truncation conventions. See [[aihub-crop-geometry]].

GPR-SD transfer trap (r1, 2026-11-26): the visible ~14.6-sample (and ~51-sample) stripes in GPR-SD stored-amplitude views are flat, time-fixed equipment components, NOT the carrier (M3-6: carrier 19.3-22.3, fixed P 20.0). Converting P-unit rules with the visible stripe period gave a box top of 0.45-0.55 P. Always convert P rules with the measured reflected carrier and tell labelers to count only arc-following stripes.
