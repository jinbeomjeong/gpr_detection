---
name: yolo11m-timing-and-fp
description: YOLO11m scratch imgsz256 1ch fold0 epoch time/VRAM (b64,w2,0xFF) and recurring pipe_band false-positive pattern
metadata:
  type: project
---

YOLO11m scratch, imgsz 256, 1 channel, AIHub fold0 (train 11,276 img), batch 64, workers 2, mask 0xFF (2026-10-30 M1-7):
- steady epoch incl. val ~29.7 s (first epoch 42-46 s); 30 epoch run ~934-948 s wall (1,068 s with CPU overlap). Probe estimate of 46 s/epoch was too high.
- VRAM peak ~7.5 GB; GPU util ~75 % while running. CPU overlap with other sessions slowed to ~34 s/epoch (strength 2 mean).
- Val curve saturates by ~epoch 20 of 30 (val box loss flat from ~13).

Recurring FP pattern: pipe_band (AIHub lpipe) has ~7x more background->pipe_band FPs than GT at conf 0.25 (crop-level fold0), AP50 only 49-59 %; suspected fragmented/inconsistent band boxes. Pipe_hyperbola has only 71 crop-level instances in fold0 val -> noisy AP.

**How to apply:** use 30 s/epoch (fold0 size) for time estimates; scale by train size for other folds. Treat pipe_band AP differences of a few %p as noise until label audit.
