# AIHub train_pool `pipe_band` 픽 품질 점검 v1 (M2-6b, band spec §9-4)

- 대상: `data/splits/aihub_yolo_v1.csv` trainpool(fold 0-4), gain 1.0, `pipe_band` 포함 crop에서 시드 0 표본 40장, 박스 53개 (test 분할은 M4-3까지 봉인이라 쓰지 않음)
- 스케일 없음(dt·dx 없음): 판정 없음. 파일명·박스 좌표는 싣지 않는다(AIHub 카드 규칙)
- 조각 묶기: 박스 53개 → 결과 51개
- 픽이 나온 결과: 51/51
- QA-4 m·ns 필드가 채워진 결과: 0개

| 항목 | 값 |
|---|---|
| status `no_scale` | 51 |
| coverage 분위(최소/25/50/75/최대) | 0.63 / 0.86 / 0.95 / 1.00 / 1.00 |
| confidence 분위 | 0.00 / 0.00 / 0.00 / 0.00 / 0.29 |
| k_ref - k0 (px) 분위 | -2.38 / 0.76 / 3.18 / 5.00 / 8.63 |
| sigma_res / P_px 분위 (문턱 0.25) | 0.08 / 0.35 / 0.52 / 0.84 / 1.52 |
| flag `air_gap_unknown` | 51 |
| flag `zero_px_unverified` | 51 |
| flag `air_gap_unknown_multi_event` | 50 |
| flag `inconsistent_picks` | 42 |
| flag `band_concave` | 20 |
| flag `bg_none` | 10 |
| flag `zero_band_overlap` | 9 |
| flag `touches_profile_edge_right` | 8 |
| flag `fragment_of_group` | 6 |
| flag `touches_profile_edge_left` | 4 |
| flag `crossing_outside_box` | 2 |
| P_px 적용 영상 수 / 분위(px) | 40 / 2.44 / 3.82 / 4.06 / 4.27 / 4.83 |
| `bandpass_px_skipped` 영상 수 / 1/f_px 분위(px) | 0 / - |

스케일 없는 입력의 px 기준은 band spec §3-5대로 우세 주기 P_px에 묶었다(f_scale 0.05P, 잔차 문턱 0.25P, sloped 0.1P, 묶기 max(0.5P, 3), q_res 0.125P; `bandpass_px_skipped`면 고정 1/2/2/3/1 px). P_px 범위 하한은 [S] §3 v1.7의 2.2 px다.

## 버전 비교 (같은 40장, 시드 0)

| 항목 | v1.0 (고정 1/2/2/3/1 px) | v1.1 (P_px 기준, 하한 4 px) | v1.2 (P_px 기준, 하한 2.2 px, 이번) |
|---|---|---|---|
| flag `inconsistent_picks` | 30/51 | 39/51 | 42/51 |
| flag `band_concave` | 19/51 | 21/51 | 20/51 |
| confidence 분위 | 0.00 / 0.00 / 0.00 / 0.00 / 0.40 | 0.00 / 0.00 / 0.00 / 0.00 / 0.29 | 0.00 / 0.00 / 0.00 / 0.00 / 0.29 |
| `bandpass_px_skipped` | 16/51 결과 (영상 수 미기록) | 16/51 결과, 14장 (1/f_px 2.44-3.82 px) | 0/51 결과, 0장 |
| sigma_res / P_px 분위 | - | - | 0.08 / 0.35 / 0.52 / 0.84 / 1.52 |
