- [YOLO11m timing & pipe_band FP](yolo11m-timing-and-fp.md) — 29.7 s/epoch per 14.8k train+val, 7.4 GB; 50 ep overfits (peak ep 24-36); band AP = val-session label style; seed noise ~5 %p; 4TU band FP at content/pad bottom
- [Framework aug defaults](framework-aug-defaults.md) — Ultralytics mosaic/hsv on, optimizer auto AdamW->MuSGD >10k iters, explicit AdamW needs momentum 0.9; rfdetr square stretch + vflip; val≠predict, CM matching by conf
- [N2-a′ perturbation sensitivity](n2a2-perturbation-sensitivity.md) — hyperbola invariant to polarity/bg/AGC/pad; band·manhole fragile; S3 timing 14 s/cond/1k crops
- [AIHub crop geometry](aihub-crop-geometry.md) — crops object-centered (hyperbola 0/503 at edge), crops overlap within session; tiling edge-recall risk
- [placeholder_scan prose trap](placeholder-scan-prose.md) — scan flags `{NAME}` even in descriptive prose; write slot names without braces outside fill sites
- [M3-3 timing & determinism](d0-rerun-determinism.md) — D0 295 s, bit-identical rerun; R3 band ~1.9 h upper (plan CPU dominant); r0 has line-global stats → infer whole line
- [M3-3 result patterns](m33-results-patterns.md) — Morocco no transfer, IDS pseudo-band recall 6.5 %, PP-b M4b→M4c, n_zero_changed artefact
- [One-pass conf filter](one-pass-conf-filter.md) — conf 0.001 + pre-merge >0.25 = conf 0.25 path; post-merge filter wrong; NMS time_limit silent drop; same-batch reference
- [Recount CSV rounding](recount-csv-rounding.md) — runner CSV score 4 dp/box 2 dp: count printed 0.25 as conf>0.25, boundary flips; synthetic Y0 apex hits only 3/100

