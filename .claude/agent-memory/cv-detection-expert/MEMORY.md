- [YOLO11m timing & pipe_band FP](yolo11m-timing-and-fp.md) — ~30 s/epoch b64 fold0, 7.5 GB VRAM; band FP 0.4-0.57x GT at conf 0.25; seed noise pipe AP ~5 %p; band FP = split labels in one session; GT-swap drops band AP 59->22
- [Framework aug defaults](framework-aug-defaults.md) — Ultralytics mosaic/hsv on, optimizer auto AdamW->MuSGD >10k iters, explicit AdamW needs momentum 0.9; rfdetr square stretch + vflip; val≠predict, CM matching by conf

