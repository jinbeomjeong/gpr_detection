<!-- generated: scripts/check_band_picks_aihub.py begin -->
# AIHub train_pool `pipe_band` 픽 품질 점검 v1 (M2-6b, band spec §9-4)

- 대상: `data/splits/aihub_yolo_v1.csv` trainpool(fold 0-4), gain 1.0, `pipe_band` 포함 crop에서 시드 0 표본 40장, 박스 53개 (test 분할은 M4-3까지 봉인이라 쓰지 않음)
- 스케일 없음(dt·dx 없음): 판정 없음. 파일명·박스 좌표는 싣지 않는다(AIHub 카드 규칙)
- 조각 묶기: 박스 53개 → 결과 50개
- 픽이 나온 결과: 50/50
- QA-4 m·ns 필드가 채워진 결과: 0개

| 항목 | 값 |
|---|---|
| status `no_scale` | 50 |
| coverage 분위(최소/25/50/75/최대) | 0.63 / 0.85 / 0.94 / 1.00 / 1.00 |
| confidence 분위 | 0.00 / 0.00 / 0.00 / 0.00 / 0.29 |
| k_ref - k0 (px) 분위 | -2.38 / 0.72 / 2.95 / 5.03 / 8.63 |
| sigma_res / P_px 분위 (문턱 0.25) | 0.08 / 0.36 / 0.55 / 0.86 / 1.52 |
| flag `air_gap_unknown` | 50 |
| flag `zero_px_unverified` | 50 |
| flag `air_gap_unknown_multi_event` | 49 |
| flag `inconsistent_picks` | 41 |
| flag `band_top_multi_event` | 22 |
| flag `band_concave` | 20 |
| flag `bg_none` | 9 |
| flag `touches_profile_edge_right` | 8 |
| flag `zero_band_overlap` | 8 |
| flag `fragment_of_group` | 8 |
| flag `touches_profile_edge_left` | 4 |
| flag `crossing_outside_box` | 2 |
| P_px 적용 영상 수 / 분위(px) | 40 / 2.44 / 3.82 / 4.06 / 4.27 / 4.83 |
| `bandpass_px_skipped` 영상 수 / 1/f_px 분위(px) | 0 / - |

스케일 없는 입력의 px 기준은 band spec §3-5대로 우세 주기 P_px에 묶었다(f_scale 0.05P, 잔차 문턱 0.25P, sloped 0.1P, 묶기 max(0.5P, 3), q_res 0.125P; `bandpass_px_skipped`면 고정 1/2/2/3/1 px). P_px 범위 하한은 [S] §3 v1.7의 2.2 px다.

## 버전 비교 (같은 40장, 시드 0)

| 항목 | v1.0 (고정 1/2/2/3/1 px) | v1.1 (P_px 기준, 하한 4 px) | v1.3 (P_px 기준, 하한 2.2 px) | v1.4 (S1 겹침 판정 + G 32 px) | v1.5 (R2 다중 사건 표시 + R3 보호판 문구, 이번) |
|---|---|---|---|---|---|
| flag `inconsistent_picks` | 30/51 | 39/51 | 42/51 | 41/50 | 41/50 |
| flag `band_concave` | 19/51 | 21/51 | 20/51 | 20/50 | 20/50 |
| confidence 분위 | 0.00 / 0.00 / 0.00 / 0.00 / 0.40 | 0.00 / 0.00 / 0.00 / 0.00 / 0.29 | 0.00 / 0.00 / 0.00 / 0.00 / 0.29 | 0.00 / 0.00 / 0.00 / 0.00 / 0.29 | 0.00 / 0.00 / 0.00 / 0.00 / 0.29 |
| `bandpass_px_skipped` | 16/51 결과 (영상 수 미기록) | 16/51 결과, 14장 (1/f_px 2.44-3.82 px) | 0/51 결과, 0장 | 0/50 결과, 0장 | 0/50 결과, 0장 |
| sigma_res / P_px 분위 | - | - | 0.08 / 0.35 / 0.52 / 0.84 / 1.52 | 0.08 / 0.36 / 0.55 / 0.86 / 1.52 | 0.08 / 0.36 / 0.55 / 0.86 / 1.52 |
| 박스 53개 → 결과 / `fragment_of_group` | - | - | 51 / 6 | 50 / 8 | 50 / 8 |
<!-- generated: scripts/check_band_picks_aihub.py end (hand-written notes go below this line) -->

## v1.4 재측정 (band spec v1.4, 2026-10-31)

- (v1.4 실행 당시 기록) 위쪽 본문 표는 v1.4 실행값이었다(`scripts/check_band_picks_aihub.py --count 40`, band spec v1.4 코드: S1 겹친 후보 판정, `merge_gap_px` 32). 지금 본문 표는 v1.5 실행값이다(아래 v1.5 절). v1.4 값은 비교 표의 v1.4 열에 있다.
- 직전 실행 대비 본문 값 중 바뀐 것은 다음과 같다(괄호 안이 직전 값, 아래에 없는 값은 같음).
  - 결과 50(51)
  - coverage 분위 0.63 / 0.85 / 0.94 / 1.00 / 1.00
  - k_ref − k0 분위 −2.38 / 0.72 / 2.95 / 5.03 / 8.63 px
  - 플래그 `bg_none` 9(10), `zero_band_overlap` 8(9), `fragment_of_group` 8(6)
  - QA-4 위반 0
  - 결과 수가 51 → 50으로 줄어 함께 바뀐 개수: `status no_scale`·`air_gap_unknown`·`zero_px_unverified` 50(51), `air_gap_unknown_multi_event` 49(50), `inconsistent_picks` 41(42), σ_res/P 분위 0.36 / 0.55 / 0.86(0.35 / 0.52 / 0.84)
- **`fragment_of_group` 6 → 8의 원인**(`analysis/processing/band_trainpool_fragment_check.py`, 같은 40장, 시드 0):
  - G 8에서 v1.4 코드는 51 결과·플래그 6으로 직전 실행(하한 2.2 px 열)과 같다. 이 표본에는 x로 5 트레이스 이상 겹친 후보가 없어 S1이 작동하지 않았다.
  - G 32에서 후보가 두 쌍 늘었다.
    - (a) 간격 10 px, 픽 차 +0.40 $P$(1.9 px, 허용 3 px) → **묶임**. 결과 51 → 50.
    - (b) 간격 30 px, 픽 차 −2.26 $P$(−9.5 px) → **거부**. 양쪽 결과에 `fragment_of_group`이 붙어 +2.
  - 판정: 증가는 S2가 넓힌 후보 중 다른 사건(2.3 주기 차)을 깊이 단계가 제대로 걸러내고 그 사실을 정보 플래그로 남긴 것이다. 오작동이 아니다. 같은 사건(0.4 주기 차)은 묶였다.

## v1.5 재측정 (band spec v1.5 R2·R3, 2026-10-31)

- 위쪽 본문 표는 **v1.5 실행값**이다(같은 명령, 같은 40장, 시드 0). 직전 v1.4 값은 비교 표의 v1.4 열에 있다.
- 픽·묶기 수치(결과 50, `inconsistent_picks` 41, `band_concave` 20, σ_res/P, `fragment_of_group` 8)는 v1.4와 같다. R2·R3은 플래그·문구만 더한다.
- 새 항목
  - `band_top_multi_event` **22/50**
  - `PROTECTIVE_PLATE_NOTE` **50/50**(모든 결과, R3)
  - `LAYER_AMBIGUITY_NOTE` **28/50** = 다중 사건 22 + `bg_none` 9 − 겹침 3
  - QA-4 위반 0
- 해석
  - `band_top_multi_event` 22/50은 AIHub 띠의 44 %에서 첫 픽 뒤 2 주기 안에 0.3 $E_\text{box}$ 이상 사건이 또 있다는 뜻이다. 겹 구조 띠와 울림 쪽 장비 파형(`band_box_convention_depth.md` §8-6)에서 예상되는 결과다.
  - 이 플래그는 층·관 내부 반사·울림을 구분하지 않는다(명세 §13-4). 그러므로 층이 있는 띠 비율로 읽지 않는다. 판정은 없다.
