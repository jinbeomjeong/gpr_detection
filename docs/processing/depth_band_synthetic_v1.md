# 의사 띠 깊이 회귀 v1 (M2-6b, spec_band_depth_v1 §9-2)

- 생성: `scripts/eval_band_depth_synthetic.py`, 2026-10-30T23:36:29+09:00, commit 4c07fb77ee1312427e67f988a9311a98b639b3b5 (dirty True)
- 구성: 20개 모델 x 덮음 (0.4, 1.0), 측선 160 트레이스, 잡음 0.05 x max|관 단독 꼭짓점 트레이스|, 시드 0
- 영점: 띠 트레이스(+여유) 제외 검출값, v: `meta_fixed`(json eps_r)
- 실행 시간: 55.5 s (단일 프로세스)
- BR1 범위(띠 - 꼭짓점 규칙): -0.29 ~ +0.16 cm

## BR1-BR7 판정

| check | criterion | passed | verdict | failed |
|---|---|---|---|---|
| BR1 | \|d_band - d_apex_rule\| <= 0.5 cm | 40/40 | PASS | - |
| BR2 | -5 <= d - d_top < 0 cm | 30/30 | PASS | - |
| BR6 | coverage >= 0.95 | 40/40 | PASS | - |
| BR7 | zero_rule as spec_depth_v1 R1-R3 (by air gap) | 40/40 | PASS | - |
| BR5 | bg_mode == "none" (< 30 outside traces) | 20/20 | PASS | - |
| BR3 | 0 < d - d_top <= 5 cm | 8/8 | PASS | - |
| BR4 | \|d - d_top\| <= 2 cm | 2/2 | PASS | - |

## 모델 x 덮음 결과

| model_id | coverage | air_gap_m | status | zero_rule | d_err_cm | band_minus_apex_rule_cm | model | bg_mode | band_coverage | confidence | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M01 | 0.4 | 0.0 | ok | ground_direct_peak | -3.61 | -0.0 | flat | masked_median | 1.0 | 1.0 | meta_eps_unverified |
| M01 | 1.0 | 0.0 | ok | ground_direct_peak | -3.63 | -0.02 | flat | none | 1.0 | 0.799 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M02 | 0.4 | 0.0 | ok | ground_direct_peak | -3.52 | -0.04 | flat | masked_median | 1.0 | 0.994 | meta_eps_unverified |
| M02 | 1.0 | 0.0 | ok | ground_direct_peak | -3.52 | -0.04 | flat | none | 1.0 | 0.799 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M03 | 0.4 | 0.0 | ok | ground_direct_peak | -3.39 | -0.04 | flat | masked_median | 1.0 | 1.0 | meta_eps_unverified |
| M03 | 1.0 | 0.0 | ok | ground_direct_peak | -3.46 | -0.11 | flat | none | 1.0 | 0.8 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M04 | 0.4 | 0.0 | ok | ground_direct_peak | -3.31 | -0.12 | flat | masked_median | 1.0 | 0.999 | meta_eps_unverified |
| M04 | 1.0 | 0.0 | ok | ground_direct_peak | -3.34 | -0.15 | flat | none | 1.0 | 0.799 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M05 | 0.4 | 0.0 | ok | ground_direct_peak | -3.78 | -0.12 | flat | masked_median | 1.0 | 1.0 | meta_eps_unverified |
| M05 | 1.0 | 0.0 | ok | ground_direct_peak | -3.68 | -0.02 | flat | none | 1.0 | 0.8 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M06 | 0.4 | 0.0 | ok | ground_direct_peak | -3.45 | 0.02 | flat | masked_median | 1.0 | 1.0 | meta_eps_unverified |
| M06 | 1.0 | 0.0 | ok | ground_direct_peak | -3.5 | -0.03 | flat | none | 1.0 | 0.8 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M07 | 0.4 | 0.0 | ok | ground_direct_peak | -2.13 | 0.01 | flat | masked_median | 1.0 | 1.0 | meta_eps_unverified |
| M07 | 1.0 | 0.0 | ok | ground_direct_peak | -1.98 | 0.16 | flat | none | 1.0 | 0.799 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M08 | 0.4 | 0.0 | ok | ground_direct_peak | -2.45 | 0.06 | flat | masked_median | 1.0 | 1.0 | surface_multiple_candidate meta_eps_unverified |
| M08 | 1.0 | 0.0 | ok | ground_direct_peak | -2.41 | 0.1 | flat | none | 1.0 | 0.8 | zero_band_overlap surface_multiple_candidate touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M09 | 0.4 | 0.0 | ok | ground_direct_peak | -4.1 | -0.02 | flat | masked_median | 1.0 | 1.0 | meta_eps_unverified |
| M09 | 1.0 | 0.0 | ok | ground_direct_peak | -4.39 | -0.29 | flat | none | 1.0 | 0.8 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M10 | 0.4 | 0.0 | ok | ground_direct_peak | -3.23 | -0.06 | flat | masked_median | 1.0 | 0.999 | meta_eps_unverified |
| M10 | 1.0 | 0.0 | ok | ground_direct_peak | -3.25 | -0.08 | flat | none | 1.0 | 0.799 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M11 | 0.4 | 0.0 | ok | ground_direct_peak | -3.43 | -0.1 | flat | masked_median | 1.0 | 1.0 | meta_eps_unverified |
| M11 | 1.0 | 0.0 | ok | ground_direct_peak | -3.38 | -0.05 | flat | none | 1.0 | 0.799 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M12 | 0.4 | 0.1 | ok | air_single_event | 1.82 | -0.07 | flat | masked_median | 1.0 | 1.0 | meta_eps_unverified |
| M12 | 1.0 | 0.1 | ok | air_single_event | 1.89 | -0.0 | flat | none | 1.0 | 0.8 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M13 | 0.4 | 0.1 | ok | air_single_event | 3.55 | -0.1 | flat | masked_median | 1.0 | 1.0 | meta_eps_unverified |
| M13 | 1.0 | 0.1 | ok | air_single_event | 3.68 | 0.03 | flat | none | 1.0 | 0.8 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M14 | 0.4 | 0.0 | ok | ground_direct_peak | -3.31 | -0.14 | flat | masked_median | 1.0 | 1.0 | inner_reflection_candidate meta_eps_unverified |
| M14 | 1.0 | 0.0 | ok | ground_direct_peak | -3.34 | -0.17 | flat | none | 1.0 | 0.799 | zero_band_overlap inner_reflection_candidate touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M15 | 0.4 | 0.1 | ok | air_single_event | 1.95 | -0.17 | flat | masked_median | 1.0 | 1.0 | meta_eps_unverified |
| M15 | 1.0 | 0.1 | ok | air_single_event | 1.91 | -0.2 | flat | none | 1.0 | 0.8 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M16 | 0.4 | 0.0 | ok | ground_direct_peak | -4.24 | -0.16 | flat | masked_median | 1.0 | 0.999 | inner_reflection_candidate meta_eps_unverified |
| M16 | 1.0 | 0.0 | ok | ground_direct_peak | -4.2 | -0.12 | flat | none | 1.0 | 0.799 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M17 | 0.4 | 0.0 | ok | ground_direct_peak | -2.58 | 0.05 | flat | masked_median | 1.0 | 1.0 | meta_eps_unverified |
| M17 | 1.0 | 0.0 | ok | ground_direct_peak | -2.59 | 0.03 | flat | none | 1.0 | 0.799 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M18 | 0.4 | 0.1 | ok | air_single_event | 2.82 | -0.02 | flat | masked_median | 1.0 | 1.0 | meta_eps_unverified |
| M18 | 1.0 | 0.1 | ok | air_single_event | 2.85 | 0.01 | flat | none | 1.0 | 0.8 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M19 | 0.4 | 0.0 | ok | ground_direct_peak | -3.03 | -0.13 | flat | masked_median | 1.0 | 1.0 | meta_eps_unverified |
| M19 | 1.0 | 0.0 | ok | ground_direct_peak | -3.05 | -0.15 | flat | none | 1.0 | 0.8 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
| M20 | 0.4 | 0.3 | ok | air_surface_late | -1.6 | -0.18 | flat | masked_median | 1.0 | 1.0 | inner_reflection_candidate meta_eps_unverified |
| M20 | 1.0 | 0.3 | ok | air_surface_late | -1.55 | -0.13 | flat | none | 1.0 | 0.799 | zero_band_overlap touches_profile_edge_left touches_profile_edge_right bg_none meta_eps_unverified |
