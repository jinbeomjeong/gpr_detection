---
name: aihub-crop-geometry
description: AIHub B-scan crops are object-centered (hyperbola never touches crop edge) and overlap within session/channel — matters for tiling real lines and for test beds
metadata:
  type: project
---

AIHub train_pool crops (crop unit 5,473) are cut around the object: pipe_hyperbola center x p5–p95 116–135 px of 256, min gap to left/right edge 81.7 px, 0/503 touch the edge; cavity center 125.5–131. pipe_band touches the edge in 35.6 %. Within one session+channel (gain stripped) 352 groups have ≥2 crops and 1,196 consecutive pairs overlap in x (median 152 px). Measured 2026-11-13 by `analysis/detection/m31_tiling_stats.py`.

Measured consequence (D1 r2, 2026-11-13, M1-10 YOLO11m last.pt, 5 folds): hyperbola recall@0.25 drops 9–15 %p at ±32 px shift, 58–72 %p at ±64 px, 35 %p when a 64-px grey(128) side strip is added with the object still centred; scale tolerance only ~0.87–1.28x (0.74x → 63 %, 0.5x → 0 %, 1.67x → 53 %). Manhole is robust to shift. Root cause attributed to centred crops + translate 0.1 / scale 0.2.

N2-a retrain (2026-11-13: project isotropic scale 0.55–1.5 with AREA shrink, horizontal shift ±96 with grey-128 fill, edge-fill strips) fixed scale (flat 91–95 % over 0.74–1.67x) and shift-with-fill (Δ64 68 → 6 %p), but shift WITHOUT fill (crop window moved inside real context, D1 X6) still lost 16 %p at ±32 and 42 %p at ±64, and a 64-px fill strip alone still costs 12 %p. Lesson: shift augmentation must move a window inside real context, not pad with fill, or the model ties "off-centre" to "fill strip present".

N2-a′ (2026-11-14, context-window shift inside real content: c1 enlarged-view window + stitched overlapping crops) cut X6 to Δ32 2.4 / Δ64 8.7 %p and passed D1 (P* 6, overlap 75 %). Costs: 0.5x recall 68 → 48 % (shrink share halved), band recall 69 → 61 %, hyperbola FP up. CAUTION: the "30 % < 32 px band slivers" figure (n2a2_band_drop_diag `band_length_px_entering_window`) is the strip-px width of ALL band boxes at window-stage INPUT, incl. those dropped outside the window — not the kept-label share. Kept-label width distribution was unmeasured as of N2-a″ v0.2 (S-0 pre-check added). Original AIHub crops have 0 band labels < 32 px (p1 51 px). Always read diag scripts before quoting their counters. N2-a″ (K32: drop context-clipped band < 32 px, 2026-11-14) did NOT recover band recall (59.6 vs 61.3, 1/5 folds better; band TP/FN unchanged, only band FP 990→857). K32 touched only ~4–5 % of kept band labels. Remaining suspects: centre-band loss, neighbour label XOR, context-mode distribution shift itself.
Windows/git: `.gitattributes eol=lf` + text-mode writers → pinned sha256 of worktree JSON (CRLF) ≠ committed blob (LF). Pin LF-normalised sha or note both. Overlapping-crop pairs (R1) give no hyperbola offsets ≥ 32 px — not usable as an independent position test.

**Why:** a scratch YOLO trained only on centered objects may lose recall for objects near tile edges when real lines are tiled (M3-1/M3-3), and has never seen truncated hyperbolas.

**How to apply:** when choosing tile overlap or judging edge misses, run a horizontal-shift sensitivity eval first (design in `docs/experiments/m31_detection_input_tiling.md` §8-2 D1). Overlapping crops can be stitched into labeled pseudo-lines for merge tests, but pixel identity of overlaps is unverified.
