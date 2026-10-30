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

**How to apply:** use 30 s/epoch (fold0 size) for time estimates; scale by train size for other folds. Treat pipe_band AP differences of a few %p as noise until label audit.
