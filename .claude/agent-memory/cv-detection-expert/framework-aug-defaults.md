---
name: framework-aug-defaults
description: Ultralytics 8.4.192 and rfdetr 1.11.2 augmentation/resize defaults that conflict with GPR policy (mosaic, hsv, square resize)
metadata:
  type: project
---

Verified 2026-10-29 from installed sources.
- Ultralytics 8.4.192: flipud=0, degrees=0 by default, but mosaic=1.0, scale=0.5, hsv_s=0.7 are on. Albumentations block in augment.py is inactive because albumentations is not installed in gpr-dev (and must stay so). No built-in gaussian noise/contrast.
- rfdetr 1.11.2: `square_resize_div_64=True` does Resize((s,s)) (aspect stretch, not letterbox); `scale_jitter=True` adds RandomSizedCrop to square; `multi_scale=PER_BATCH`. AUG_AGGRESSIVE/AUG_AERIAL presets enable VerticalFlip and big rotations. Default AUG_CONFIG is hflip only.
- RFDETRMedium(pretrain_weights=None) builds offline (33.7M params); transformers RTDetrV2/DFine random init with use_pretrained_backbone=False works offline.
- Ultralytics 8.4.192 `optimizer: auto` (trainer.build_optimizer): iterations = ceil(N/max(batch,nbs))*epochs; >10000 -> MuSGD lr0 0.01, else AdamW lr 0.002*5/(4+nc) (=0.00125 for nc=4), and warmup_bias_lr forced 0. AIHub folds b64 (177-193 it/epoch) switch at epoch 52-57, so budgets 52-56 mix optimizers across folds. Pin `optimizer`/`lr0` explicitly when epochs > 50. Auto uses AdamW beta1 0.9, but explicit `optimizer: AdamW` takes beta1 from `momentum` (default 0.937) -> also pin `momentum: 0.9` to reproduce auto (done in configs/train_yolo11m_m110.yaml).
- `patience` 0 disables EarlyStopping (`patience or inf`); default 100. EarlyStopping and best.pt both use fitness = mAP50-95 only (w [0,0,0,1]).

**Why:** these silently violate the no-vflip / letterbox policy in comparisons.
**How to apply:** write explicit values in every train config; for RF-DETR confirm aspect-preserving path with an input-tensor-size smoke before compare runs. Policy table: docs/experiments/augmentation_policy.md.
