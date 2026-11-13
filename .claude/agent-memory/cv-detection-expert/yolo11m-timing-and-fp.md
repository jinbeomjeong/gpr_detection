---
name: yolo11m-timing-and-fp
description: YOLO11m scratch imgsz256 1ch fold0 epoch time/VRAM (b64,w2,0xFF) and recurring pipe_band false-positive pattern
metadata:
  type: project
---

YOLO11m scratch, imgsz 256, 1 channel, AIHub fold0 (train 11,276 img), batch 64, workers 2, mask 0xFF (2026-10-30 M1-7):
- steady epoch incl. val ~29.7 s (first epoch 42-46 s); 30 epoch run ~934-948 s wall (1,068 s with CPU overlap). Probe estimate of 46 s/epoch was too high.
- VRAM peak ~7.5 GB; GPU util ~75 % while running. CPU overlap with other sessions slowed to ~34 s/epoch (strength 2 mean).
- Per-fold run wall ~= E*29.7 s*(N_train/11276) + ~54 s (first-epoch extra + startup + final best.pt val); matched 934-948 s at E=30. Folds 0-4 train 11,276/11,591/11,900/12,311/12,126.
- Val curve near-saturated by ~epoch 20 of 30 but mAP50 still +0.1-0.25 %p/epoch over 21-30 (val box loss flat from ~13, val cls min at 28-30) -> M1-10 budget set to 50 epoch (user, 2026-10-30).

Recurring FP pattern: pipe_band (AIHub lpipe) has ~7x more background->pipe_band FPs than GT in the M1-7 confusion matrix (crop-level fold0) — but that matrix was at conf 0.001 (Ultralytics 8.4.192 fills the CM with the val `conf`, val.py:266; match IoU 0.45), so the ratio is NOT an operating-threshold FP rate; re-measure at conf 0.25. AP50 only 49-59 %; suspected fragmented/inconsistent band boxes. Pipe_hyperbola has only 71 crop-level instances in fold0 val -> noisy AP.
- Re-measured 2026-10-31 at conf 0.25 (scripts/eval_detector.py --confusion-conf 0.25, 6 M1-7 runs): background->pipe_band FP = 0.39-0.57x GT (377-553 of 966), band recall 0.65-0.71.
- Cause traced 2026-10-31 (m17_aug0, conf 0.25): of 377 band FPs, 59 % fragment (overlaps band GT, IoU<0.45), 25 % duplicate (GT already taken), 16 % no-GT (58/59 in images that have band GT), 0 % other-class. 356/377 come from ONE session NM_20220922(111046), which also holds 934/966 fold0 band GTs; band GTs are often split into several short boxes along one band. So fold0 band metrics ≈ one session's labeling convention. FP list analysis/detection/m17_pipe_band_fp_list.csv for curator.
- Seed noise (fold0, 30 ep, same config, seed 0 vs 1): pipe 2-class AP50 mean differs 4.3-5.5 %p, pipe_band up to 7.9 %p, pipe_hyperbola up to 6.2 %p, overall mAP50 2.0-2.5 %p. Strength ranking flipped between seeds.
- Label-convention effect (2026-10-31): same m17_aug0 last.pt, only GT swapped to band auto-merge v2 (gap<=32 px): band AP50 59.1 -> 22.5, overall mAP50 84.4 -> 75.2 (crop). Cross-GT numbers are not comparable; judging on the new GT favours models trained on it. eval_detector always writes <run>/eval/, so evaluate against a second GT via a view folder (analysis/detection/band_v2_eval_view.ps1). Eval of one run, last.pt, crop+image, conf-cm 0.25 ~2.3 min.
- Symmetric bias confirmed (M1-7b, 2026-10-31): training on v2 (only 14.7 % of train band boxes changed) flips band AP50 by ~±27 %p — wins on own GT, loses equally on the other. IoU-AP cannot separate "better detection" from "learned box convention"; need a convention-free metric (x-interval coverage) for band label-rule decisions. v2 train run 942-947 s, VRAM 7.38 GB (same as v1).

- Full eval_detector default (last,best x crop,image, --confusion-conf 0.25) on fold0 val: 244-271 s per run (M1-7 seed runner 2026-10-31). Use ~4.2 min/fold for M1-10 eval estimates.

- M1-10 measured (2026-11-01, 50 ep, 5 folds): epoch 29.6-29.9 s for ALL folds — epoch time tracks train+val total (14,801 every fold), not train size; run wall 1,512-1,543 s; VRAM target 7.31-7.37 GiB (nvidia-smi 7.44-7.50 GB); eval 229-253 s/fold.
- 50 ep overfits: val mAP50 peaks epoch 24-36, val box loss min epoch 14-26 then rises; best.pt beats last.pt by 2.1-4.6 %p on every fold. Fold0 50 ep was not better than 30 ep.
- Band AP across folds tracks val-session label style (single long box sessions fold1/2 -> AP 35-39, FP/GT ~1.0), not instance count, not train/val convention mismatch.
- eval metrics.json weights.checkpoint_epoch_zero_based = -1 / epochs_completed 0 for last & best (stripped ckpt); take epochs from results.csv.
- Real-data band FP (D3 4TU, N2-a′ fold0, 2026-11-14): 152/243 pipe_band boxes sit within 8 px of the content bottom (time-window end -> gray pad 128), 140 on one line 09.7/Path5. Check "box touches content/pad boundary" before reading band counts or R3 pair stats on field lines. Missed hyperbolas on 01.1 look wider (~100-135 L1 px) than the fold0 box width max 88.6 px.

**How to apply:** use ~29.7 s/epoch for train+val ~14.8k images (scale by train+val total, not train only). Treat pipe-AP differences under ~5 %p from single runs as noise; need multiple seeds/folds before claiming a difference.
