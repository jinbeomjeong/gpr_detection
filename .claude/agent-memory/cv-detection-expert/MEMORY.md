- [YOLO11m timing & pipe_band FP](yolo11m-timing-and-fp.md) — 29.7 s/epoch per 14.8k train+val, 7.4 GB; 50 ep overfits (peak ep 24-36); band AP = val-session label style; seed noise ~5 %p
- [Framework aug defaults](framework-aug-defaults.md) — Ultralytics mosaic/hsv on, optimizer auto AdamW->MuSGD >10k iters, explicit AdamW needs momentum 0.9; rfdetr square stretch + vflip; val≠predict, CM matching by conf
- [AIHub crop geometry](aihub-crop-geometry.md) — crops object-centered (hyperbola 0/503 at edge), crops overlap within session; tiling edge-recall risk

