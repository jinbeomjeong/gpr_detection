- [YOLO11m timing & pipe_band FP](yolo11m-timing-and-fp.md) — 29.7 s/epoch per 14.8k train+val, 7.4 GB; 50 ep overfits (peak ep 24-36); band AP = val-session label style; seed noise ~5 %p; 4TU band FP at content/pad bottom
- [Framework aug defaults](framework-aug-defaults.md) — Ultralytics mosaic/hsv on, optimizer auto AdamW->MuSGD >10k iters, explicit AdamW needs momentum 0.9; rfdetr square stretch + vflip; val≠predict, CM matching by conf
- [N2-a′ perturbation sensitivity](n2a2-perturbation-sensitivity.md) — hyperbola invariant to polarity/bg/AGC/pad; band·manhole fragile; S3 timing 14 s/cond/1k crops
- [AIHub crop geometry](aihub-crop-geometry.md) — crops object-centered (hyperbola 0/503 at edge), crops overlap within session; tiling edge-recall risk
- [placeholder_scan prose trap](placeholder-scan-prose.md) — scan flags `{NAME}` even in descriptive prose; write slot names without braces outside fill sites
- [D0 rerun determinism](d0-rerun-determinism.md) — D0 full run 295 s/16,919 img; rerun reproduced gate values bit-for-bit on same GPU/env
- [Recount CSV rounding](recount-csv-rounding.md) — runner CSV score 4 dp/box 2 dp: count printed 0.25 as conf>0.25, boundary flips; synthetic Y0 apex hits only 3/100

