# 명세 v1 — 검출 영상(L1) 생성, 타일 병합, 측선 단위 깊이 호출 (M3-1 · M3-2 `DetectionImage` 좌표 부분)

- 상태: **v0.3.6** (2026-11-13, gpr-processing-expert). 사용자 결정(2026-11-13)을 반영한 **구현 위임 명세**다. §4·§5·§6·§8을 python-expert가 그대로 구현한다. v0.1(같은 날 초안)의 대안 비교·근거는 §2·§3에 남겼다.
- v0.3 정정은 [CV] v0.2의 발견 3건·선택 4건을 반영한 것이고, 요약은 §0-1에 있다. 시험 목록은 §8이 한 벌이고, [CV] C1–C5와의 대응도 §8에 적었다.
- 범위
  - 실측 npy(4TU·광저우대·GPR-SD) → 측선 L1 영상 → 256×256 겹침 타일 → 검출 → 측선 연속 좌표에서 병합 → L0 `GridBox`.
  - `pipeline`의 측선 단위 깊이 호출 순서(ADR-004 결정 7): 영점·`exclude_traces` → 쌍곡선 → 속도 풀 → 띠.
  - M3-1 왕복 시험과 깊이 순서 시험.
- 범위 밖
  - 영상 품질(채움 값·회색 매핑·검출 영상 배경 제거 창)은 M3-2로 넘긴다. M3-1은 §4-2의 **가정값**으로 구현하고, 결과에 그 사실을 표시한다(`assumed_params`).
  - 목표 주기 $P^*$와 겹침 폭은 cv-detection-expert의 D1 GPU 평가로 정한다. 여기서는 파라미터로만 둔다. D1 r2(커밋 ea7f447, [CV] §8-5) 결과 기본값은 4.06 px·**192 px(잠정)**이다.
  - GPR-SD 라벨 축척은 M3-4 라벨이 나온 뒤 본다.
- 바꾸지 않는 것
  - 좌표 계약(ADR-004 결정 1–8)과 `ImageGridMap`·`InputTransform` 스키마.
  - 깊이 갈래 입력(L0 `GridBox`만)과 깊이 함수 계약(Q6: 박스 위 끝 여유를 더하지 않는다).
  - 전처리 (1)–(4)의 정의(`spec_depth_v1.md` §3).
- 짝 문서: cv-detection-expert의 [`docs/experiments/m31_detection_input_tiling.md`](../experiments/m31_detection_input_tiling.md) ([CV]). 검출기 쪽(타일 배치·잘림 표시·R2·띠 x 중재)의 원본은 [CV]다. 이 문서는 그 규칙을 구현 명세로 옮기고 물리 쪽 조건을 더했다. 두 문서가 어긋나면 [CV]가 검출 규칙의 원본이고, 이 문서가 물리 거부권·축척·깊이 연결의 원본이다. 이 문서는 [CV]를 고치지 않는다.
- 근거 산출물(v0.1에서 실행; 단일 프로세스·1코어·BelowNormal, 난수 없음)
  - `analysis/processing/detection_image_scale_survey.py` → `docs/processing/detection_image_scale_survey_v1.json` [SV]
  - `analysis/processing/detection_image_period_check.py` → `docs/processing/detection_image_period_check_v1.json` [PC]
- 출처 약어: [ADR4] `docs/architecture/ADR-004-coordinate-contract.md`, [D] `docs/processing/spec_depth_v1.md`, [B] `docs/processing/spec_band_depth_v1.md`, [AH] `docs/datasets/aihub_71327.md`, [AW] `analysis/processing/aihub_wavelet_length.json`, [AP] `docs/experiments/augmentation_policy.md`, [R4] `docs/datasets/readme_4tu.md`, [RG] `docs/datasets/readme_guangzhou.md`, [RS] `docs/datasets/readme_gprsd.md`, [MR] `docs/processing/m29c_apex_review.csv`.

## 0. 사용자 결정(2026-11-13)과 요약

| 항목 | 결정 | 이 문서 |
|---|---|---|
| Q1 L1 방식 | **C′ 채택**: 주기 맞춤 세로 배율, 깊이 등방 가로 배율, 256×256 겹침 타일, 겹침 = max($O_\text{phys}$, 128)(v0.3.4: $O_\text{cv}$ 잠정 192, Q2 행). 시간창 전체를 쓰고 세로는 자르지 않는다 | §4 |
| 병합 | [CV] R2(IoU ≥ 0.7, 잘린 박스가 끼면 IoMin ≥ 0.8, 대표 1개, 평균 없음) + 물리 거부권 V1·V2. `pipe_band`는 트레이스 구간만 중재하고 $k_0,k_1$은 그대로 두며 `group_band_boxes`로 넘긴다. 짝 없이 잘린 쌍곡선은 `tile_truncated` 플래그를 달아 깊이로 넘긴다 | §5, §6 |
| Q2 $P^*$, 겹침 50/75 % | D1 r2(ea7f447): $P^*$ **4.06 유지**. 겹침은 규칙 O "둘 다 불충족"이라 **잠정 192 px(75 %)**. 사용자 선택 N2-a(위치·축척 쏠림 보정 증강 재학습) 뒤 D1 r3에서 다시 판정 | §4-2 |
| Q4·Q7 | $v_s$ = 메타 v, 광저우 0.10 m/ns 가정. GPR-SD는 M3-1에서 기본안(주기 맞춤)으로 좌표 왕복 시험만 한다. 라벨 축척은 M3-4 뒤 | §4-4, §9 |
| Q3·Q5 | M3-2(영상 품질)로 넘긴다. M3-1은 가정값으로 구현하고 표시한다 | §4-2 `assumed_params` |
| Q6 | 깊이 계약을 바꾸지 않는다 | §7 (a) |
| Q8·U6 | 측선 L1 정수 px `x0` + 측선당 `ImageGridMap` 1개([CV] 제안과 같음) | §4-3 |
| Q9 내용 높이 > 256 (v0.3.1) | **(A) 압축 유지**: 두 축을 같은 비율로 줄이고 `compression`·`scale_out_of_range`를 남긴다. D3·M3-2에서 다시 본다 | §4-4 3, §10 |

### 0-1. v0.3 정정 ([CV] v0.2 발견·선택 반영)

| [CV] 항목 | 판단 | 반영 |
|---|---|---|
| 발견 1: 측선별 εr로 $O_\text{phys}$가 4TU 최대 96 px, $P^*$ 6이면 141 px | 맞다. v0.2의 "64–78 px"는 εr 9·400 MHz 기준이었다. 아래 발견 2의 상한식으로 바꾸면 4TU는 64–82 px($P^*$ 4.06), 91–120 px($P^*$ 6)이고 모두 128 이하다. 그래서 실제 겹침은 언제나 $O_\text{cv}$이고 상한 224는 작동하지 않는다 | §4-4 9, §5-4 |
| 발견 2: 광저우 2000 MHz 24측선은 $O_\text{phys}$ > 192($P^*$ 6이면 900 MHz도) | **$O_\text{phys}$ 식을 정정한다(사용자 결정 불필요).** 박스 폭 상한을 미터(2 $a_\max$)만으로 잡은 것이 원인이다. 고주파에서는 2 m가 80 주기를 넘어, 검출기가 낼 수 있는 박스 폭(AIHub `tpipe` 최대 80.7 px = 19.9 주기)과 동떨어진다. 새 식은 $\min(2a_\max,\ 20\,\lambda/2)$를 L1로 환산한다. 깊이 단위 등방이라 $\lambda/2$는 L1에서 정확히 $P^*$ px다. 결과는 2000 MHz 82 px, 전 측선 ≤ $20P^*$다. 이 측선들을 빼지 않고 플래그 `overlap_period_capped`만 단다 | §4-4 9, §4-6 |
| 발견 3: 광저우 내용 높이 > 256 px 측선 75개 | **v0.2 §3의 "큰 $P^*$에서만 생긴다"는 틀렸다.** 광저우 MALA(Δt 0.391 ns, 시간창 190–200 ns ≈ 40–80 주기)에서 기본값으로도 생긴다. 개수는 MALA 주기 가정에 따라 다르다: [CV]의 2.5 ns 가정이면 75개, 폴더별 실측 주기(4.45–36.6 샘플, 불안정)면 8개(`analysis/processing/detection_image_gz_height_check.py`, 압축비 $c$ 1.82). [CV] §4-4 3은 이 측선들이 "세로 겹침 타일 경로를 탄다"고 했는데, 이는 v0.1 기준이다. v0.2부터는 두 축을 함께 줄이는 압축(`period_compressed`)이라 **두 문서가 어긋난다** | §3, §4-4 3. M3-1 기본은 압축이고, 압축비 $c$ > 1.25(학습 등방 증강 0.8–1.2 밖)면 `scale_out_of_range`를 붙인다. 세로 분할은 하지 않는다(Q9 결정 (A), 2026-11-13, §10) |
| 선택 (a) 마지막 타일 측선 끝 맞춤 | v0.2와 같다 | – |
| 선택 (b) 대표 박스 트레이스 구간 그대로(합집합 없음) | v0.2와 같다(§5-3 3-5) | – |
| 선택 (c) 짧은 측선도 256×256 L1(왼쪽 위 + 오른쪽·아래 채움), letterbox 항상 항등 | 따른다. 물리 쪽으로도 이득이다. 백엔드 좌우 채움이 없어져 `InputTransform`이 언제나 항등이다 | §4-3, §4-4 10·11, T6 |
| 선택 (d) 병합 시험 C1–C5, 축척·기하는 [SP] T1–T10 정본 | 시험 목록을 한 벌로 합쳤다. ID는 [SP]가 정본이고 [CV] ID를 옆에 적는다 | §8 |
| (덧붙임) 진단 이름 `seams_px`·`merged_from_tiles`·`tile_seam_box`·`short_line_padded` | [CV]와 같은 이름을 쓴다 | §4-3, §5-1 |

요약
1. 측선 하나는 **측선 L1 영상 하나**(높이 256 px, 폭 $W_1$)가 된다.
   - 세로 배율은 $s_y=P_\text{line}/P^*$다. 주기는 배경 제거 후 단면에서 잰다.
   - 가로 배율은 $s_x=s_y\,(\Delta t\,v_s/2)/\Delta x$다(깊이 단위 등방). 이러면 쌍곡선 점근 기울기가 45°가 된다.
   - 내용은 위에 두고 아래를 0 진폭 회색으로 채운다.
2. 타일은 측선 L1의 정수 px 원점 $x_0$에서 256 px씩 자른다. 보폭은 $256-O$이고 마지막 타일은 측선 끝에 맞춘다. $W_1\le256$이면 타일 한 장이고, 256×256 안 왼쪽 위에 두고 오른쪽을 채운다(letterbox 항상 항등).
3. 검출 박스는 타일 L1 → 측선 L1(연속)로 옮긴다. 거기서 잘림 표시, R2 + V1·V2 병합, 띠 x 중재를 하고, L0 바깥쪽 반올림은 마지막에 한 번만 한다.
4. 깊이는 측선마다 다음 순서로 부른다: `band_exclude_traces` → `prepare_depth_inputs` → `estimate_depth`(쌍곡선마다) → `velocity_estimate_from` → `estimate_band_depths`.
5. 기본값에서의 결과(εr 9 기준, §4-7)
   - 4TU: $s_x$ 1.54 트레이스/px, $s_y$ 6.3 샘플/px, 내용 높이 81 px(Δt 0.195는 161 px).
   - 광저우 30 ns: 1.50 트레이스/px, 12.3 샘플/px, 42 px.
   - GPR-SD: 5.04 트레이스/px, 12.6 샘플/px, 25 px.
   - 4TU·광저우 쌍곡선은 23–63 px로 AIHub `tpipe`(46×27 px)와 비슷한 크기가 된다.
   - GPR-SD는 주기가 51 샘플이라 쌍곡선이 8×14 px로 AIHub와 맞지 않는다(데이터셋 카드 기록 후보, §11).

## 1. 기호

| 기호 | 뜻 | 단위 |
|---|---|---|
| $\Delta t$, $\Delta x$ | 샘플 간격, 트레이스 간격(`Radargram.dt_ns`, `dx_m`) | ns, m |
| $n_s$, $N$ | 측선의 샘플 수, 트레이스 수 | 개 |
| $P_\text{line}$ | 측선 우세 주기(§4-4 2) | 샘플 |
| $P^*$ | L1 목표 주기 `p_target_px` | px |
| $\delta_z$ | 샘플당 깊이 $v_s\Delta t/2$ (`depth_m_per_sample`) | m/샘플 |
| $v_s$ | 가로 배율용 속도(깊이에 쓰지 않음) | m/ns |
| $s_y$, $s_x$ | 공칭 배율 | 샘플/px, 트레이스/px |
| $H_c$, $W_1$ | 측선 L1 내용 높이, 측선 L1 폭 | px |
| $\sigma_y=n_s/H_c$, $\sigma_x=N/W_1$ | 실제 배율(`samples_per_px`, `traces_per_px`) | 샘플/px, 트레이스/px |
| $L$ = 256 | 타일 한 변(`tile_px`) | px |
| $O$, $O_\text{phys}$ | 실제 겹침, 물리 하한 | px |
| $S=L-O$ | 보폭 | px |
| $P_{L1}=P_\text{line}/\sigma_y$ | L1에서의 주기 | px |
| $\lambda=2\delta_z P_\text{line}$ | 토양 속 파장(Δt 없이 계산) | m |
| $r_F=\sqrt{\lambda d/2}$ | 제1 프레넬 반경 | m |

## 2. 데이터셋 축척과 쌍곡선 크기 (v0.1 근거, 변경 없음)

### 2-1. 격자 (근거: 로더 json·색인 csv, [SV] `fourtu_grid`·`gz_grid`·`gprsd_grid`)

| 데이터 | 측선 수 | $n_s$ | $\Delta t$ (ns) | $\Delta x$ (m) | $N$ (트레이스) p5 / p50 / p95 / 최대 | 진폭 |
|---|---|---|---|---|---|---|
| 4TU | 759 | 512(전부) | 0.098 ×582, 0.097 ×92, 0.195 ×83, 0.117 ×2 | 0.02 | 89 / 235 / 2,250 / 14,516 (최소 9: 03.9 Path5) | int16 원시 |
| 광저우 관로 | 523 (IDS 448, MALA 75) | 512 ×448, 2048 ×6, 256 ×6, 384 ×2, MALA 479–499 | 시간창 ÷ $n_s$: 400 MHz 30 ns 0.0586 (177개), 50 ns 0.0977 (85), 40 ns 0.0781 (47) 등; MALA 0.391 | 0.024 대부분 (0.01–0.044, MALA 0.0256) | 110 / 472 / 6,077 / 11,067 | IDS int16 원시, MALA 처리본 |
| GPR-SD | 12 | 321 | 가설 A 0.3022 / B 0.2553 (미확정) | 0.03 | 9,360–27,628 | float32 처리본 |
| AIHub `tpipe` (비교) | 크롭 1,522 (gain 3벌) | 256 px | 미상 | 미상 | 256 px | JPG |

- 4TU 타일 수(εr 9 일괄)는 겹침 128 px에서 2,209장, 192 px(v0.3.4 잠정 기본)에서는 3,530장([CV] §4-4 표 4,263장은 측선별 εr 기준), 물리 하한 65 px에서 1,780장이다. εr 9·$P$ 2.51 ns 일괄 가정이고, 실제는 측선별 $P_\text{line}$·εr로 조금 달라진다. [CV] §4-4의 3,297장은 1 트레이스 = 1 px 기준이다.

### 2-2. 우세 주기 (근거: [PC], AIHub는 [AW])

추정기는 AIHub $P$_px와 같다(트레이스 평균 $|\mathrm{rfft}|$의 최대). 실측선에는 8배 영 채움을 더했다.

| 데이터 | 단면 | $P$ p10 / p50 / p90 | 비고 |
|---|---|---|---|
| AIHub `tpipe` 꼭짓점 크롭(88) | DC 제거 JPG | 3.82 / **4.06** / 4.57 px | 띠 크롭 137개는 p50 4.20 px |
| 4TU (T 꼭짓점이 있는 27측선) | 원시 DC 제거 | 2.73 / 3.10 / 3.86 ns | 영점 띠·wow가 섞여 길게 나온다 |
| 〃 | (1)–(4) 후 측선 전체 | 2.38 / **2.51** / 2.84 ns | 영점 띠 ± 엽 간격 1.2–1.4 ns(반주기, [R4] 정정 주석 2)와 맞다 |
| 〃 | (1)–(4) 후 꼭짓점 창(41개) | 2.28 / 2.51 / 3.43 ns | 관 반사도 같은 주기 |
| 광저우 400 MHz 30 ns (8측선) | 원시 / (1)–(4) 후 / 헤더 `spectrum_peak_MHz` | 3.44–7.50 / **2.14–3.43 (p50 2.93)** / 2.85–14.3 ns | 원시·헤더는 쓸 수 없다 |
| 광저우 400 MHz 50 ns (8측선) | 〃 | 3.07–6.25 / **1.74–2.46 (p50 2.11)** / 2.66–7.14 ns | |
| GPR-SD (관로 표식 17건 창) | 저장 진폭 | 48.8 / **51.2** / 52.5 샘플 | 측선 전체 4개 p50 50.6. ns로는 A 15.5 / B 13.1 ns (62 / 73 MHz) |

- 주기는 **배경 제거 후(4) 단면**에서 잰다. 원시 단면에서는 4TU가 +24 %, 광저우가 최대 2.5배 길게 나온다.
- GPR-SD의 51 샘플 주기는 문서의 200–400 MHz와 dt 가설 A·B 어느 쪽과도 맞지 않는다([RS] §4). **[탐색, 판정 아님]** 데이터셋 카드 기록 후보로만 둔다(§11).

### 2-3. 쌍곡선 박스 크기 (근거: [SV] `analytic`, [MR], [AH])

박스 모형(**가정**)은 [D] §11-2 박스 공급 규칙(M2-5)과 같은 꼴의 점산란 쌍곡선이다. 반폭 $a$, 꼭짓점 위 1.5 주기, 날개 끝 아래 1.5 주기를 둔다.

$$h_\text{ns}=\frac{2}{v}\left(\sqrt{d^2+a^2}-d\right)+3P,\qquad w_\text{tr}=\frac{2a}{\Delta x}+1$$

- 반폭 근거: 4TU 진짜 꼭짓점 41개의 추적 반폭은 p50 0.34 m, p90 0.58 m, 최대 0.62 m다([MR] type T). 깊이 공급 박스는 0.8 m([D] §11-2), 추적 상한 $a_\max$는 1.0 m([D] §5)다.

| 데이터 (v, $P$) | $a$ (m) | 원래 격자 w × h (트레이스 × 샘플), $d$ 0.3 / 0.7 / 1.5 m | 원래 w/h | 폭 (주기) | C′ L1 w × h (px), $d$ 0.7 | C′ w/h |
|---|---|---|---|---|---|---|
| 4TU Δt 0.098 (0.1, 2.51 ns) | 0.35 | 36 × 110 / 94 / 85 | 0.33–0.42 | 5.6 | 23 × 15 | 1.34–1.73 |
| | 0.6 | 61 × 153 / 122 / 100 | 0.40–0.61 | 9.6 | 40 × 19 | 1.63–2.49 |
| | 0.8 | 81 × 190 / 151 / 118 | 0.43–0.69 | 12.8 | 52 × 24 | 1.74–2.80 |
| 4TU Δt 0.195 | 0.6 | 61 × 77 / 61 / 50 | 0.79–1.22 | 9.6 | 40 × 19 | 같음 |
| 광저우 30 ns (0.1 가정, 2.93 ns) | 0.6 | 51 × 277 / 226 / 189 | 0.18–0.27 | 8.2 | 34 × 18 | 1.51–2.21 |
| 광저우 50 ns (0.1 가정, 2.11 ns) | 0.6 | 51 × 141 / 110 / 88 | 0.36–0.58 | 11.4 | 47 × 21 | 1.78–2.85 |
| GPR-SD ($\delta_z$ 0.012 m, 51 샘플) | 0.6 | 41 × 185 / 172 / 163 | 0.22–0.25 | 1.95 | 8 × 14 | 0.55–0.63 |
| AIHub `tpipe` 라벨(고유 564) | – | w 37–58 px, h 22–33 px (p10–p90), 중앙 46 × 27 | **1.71** (1.41–2.13) | **11.4** | – | – |
| AIHub `cavity` / `manh` | – | 중앙 22 × 29 / 28 × 46 px | 0.74 / 0.54 | – | – | – |

- "폭 (주기)" = $2a/(\lambda/2)$는 배율로 바꿀 수 없는 물리량이다. 4TU·광저우는 AIHub와 같은 범위이고 GPR-SD만 크게 작다.
- AIHub 쌍곡선 라벨은 크롭 경계에 닿은 것이 0 %다([CV] §2: 0/503). 그래서 잘린 쌍곡선은 학습 분포 밖이고, 겹침이 필요하다.
- AIHub 등방성 [추정]: `tpipe` 중앙 박스를 위 모형으로 풀면, 가로가 깊이 단위보다 1.6배 넘게 압축된 경우 해가 없다. AIHub는 깊이 단위 등방에서 크게 벗어나지 않는다.

## 3. 대안 비교 (v0.1, Q1에서 C′ 채택)

| 대안 | 쌍곡선 종횡비 | 작은 관로 소실 | 판정 |
|---|---|---|---|
| A. 측선 전체 비율 유지 리사이즈 | 원래 격자 비율이라 깊이 단위 대비 세로 2.0–8.2배, w/h 0.15–0.69(`cavity`·`manh` 모양) | 81 트레이스 쌍곡선이 4TU p5 9 px, 광저우 IDS p10 6.7 px, GPR-SD 0.5–1.5 px | 기각 |
| B-native. 고정 폭 타일, 원래 격자 1:1 | A와 같음 | 없음 | 기각 |
| B-matched. 맞춤 배율, 겹침 없음 | 45° 점근, w/h 1.2–3.2 | 4TU 23–52 px | 경계에서 잘림 → C′ |
| **C′. 맞춤 배율 + 겹침 타일** | 위와 같음 | 위와 같음, 경계 잘림은 병합으로 해소 | **채택(Q1)** |

- 세로: 시간창 전체를 쓰고 자르지 않는다.
  - 기본값($P^*$ 4.06)에서 4TU는 80–161 px, 광저우 IDS 400 MHz는 42–96 px, GPR-SD는 25 px라 256 안에 든다.
  - **광저우 MALA는 예외다**(v0.3 정정). Δt 0.391 ns·시간창 190–200 ns가 주기 40–80개라 256 px를 넘을 수 있다. [CV]는 MALA 2.5 ns 가정으로 75측선을 셌고, 폴더별 실측 주기로는 8측선이다(압축비 1.82). $P^*$ 6에서는 IDS 일부(2000·600 MHz, 400 MHz 96 ns 창)도 넘는다.
  - 넘치면 자르지 않고 두 축을 함께 줄인다(§4-4 3, `period_compressed`). 압축비가 1.25를 넘으면 객체 축척이 학습 범위 밖이므로 `scale_out_of_range`를 붙인다. Q9 확정 (A)(2026-11-13): 세로 분할하지 않는다. 재검토는 D3·M3-2.

## 4. 구현 명세 — `DetectionImage` (`src/gpr_detection/preprocess/to_image.py`)

### 4-1. 모듈과 의존

- 위치: `preprocess/to_image.py`(기존 `ImageGridMap` 옆). `ImageGridMap`은 **바꾸지 않는다**.
- import 허용: `numpy`, `gpr_detection.io.radargram`(`Radargram`, `velocity_from_eps`), `gpr_detection.preprocess.filters`(`prepare_fitting_branch`, `agc`, `remove_dc`). `filters.dominant_period_px`·`estimate_period_px`는 import하지 않는다(아래).
- 주기 추정(v0.3.3): `measure_period_samples`는 `to_image.py`에 **새로 만든다**. 기존 `filters.estimate_period_px`·`dominant_period_px`(166–195줄)는 고치지도 재사용하지도 않는다.
  - 공통점: 트레이스 평균 $|\mathrm{rfft}|$의 최대(DC 제외)라는 정의가 같다. AIHub $P^*$ 4.06 px는 기존 함수로 잰 값이다.
  - 차이: 기존 함수는 영 채움이 없고(512 샘플에서 주기 해상도 약 ±5 %), 유효 범위가 2.2–128 px이며, 스케일 없는 입력의 `bandpass_px` 경로(깊이 갈래, AIHub 픽 시험 기준)에 묶여 있다. 새 함수는 8배 영 채움, 범위 $[4, n_s/4]$ 샘플, 탐색 구간 양끝 빈이면 `None`이다.
  - 이유: 기존 함수를 바꾸면 깊이 갈래의 `bandpass_px`·AIHub 회귀가 함께 움직이고, 그대로 쓰면 해상도가 모자라고 2,048 샘플 측선($P$ ≈ 128 샘플)이 범위 상한에 걸린다.
- `cv2`는 쓰지 않는다. 리샘플은 §4-4 6의 numpy 면적 평균으로 한다. 결정적이고 시험식이 닫혀 있기 때문이다. `detect`·`depth`·`pipeline`은 import하지 않는다(README §2-2 규칙).
- 스케일 없는 입력(`dt_ns is None` 또는 `dx_m is None`, AIHub·모로코)은 이 경로를 쓰지 않는다. 지금처럼 `ImageGridMap.identity`를 쓴다. `make_detection_image`는 이런 입력에 `ValueError`를 낸다.

### 4-2. 설정 `DetectionImageConfig`

```python
@dataclass(frozen=True)
class DetectionImageConfig:
    p_target_px: float = 4.06          # P*  (D1 r2: 4.06 kept)
    overlap_px: int = 192              # O_cv (D1 r2: provisional 192, re-judged in r3 after N2-a)
    tile_px: int = 256                 # L
    overlap_min_px: int = 64           # O_min of O_phys
    a_max_m: float = 1.0               # O_phys: tracking limit (spec_depth 5)
    overlap_width_periods: float = 20.0  # O_phys: widest AIHub tpipe box (19.9 P)
    overlap_max_px: int = 224          # cap, S >= 32 (inactive with the defaults)
    compress_max: float = 1.25         # c above this -> scale_out_of_range (1/0.8)
    period_pad_factor: int = 8
    period_min_samples: float = 4.0    # valid P range [4, n_s/4]
    clip_percentile: float = 99.0
    v_assumed_m_per_ns: float = 0.10   # Guangzhou (README), Q7
    # --- M3-2 assumptions (Q3, Q5); listed in DetectionImage.assumed_params ---
    pad_gray: int = 128                # zero amplitude grey
    gray_mapping: str = "symmetric"    # q = rint(127.5 + 127.5 * clip(a/c, -1, 1))
    detection_bg_window: int = 301     # same as the fitting branch
DEFAULT_DETECTION_IMAGE_CONFIG = DetectionImageConfig()
DATASET_DEFAULT_PERIOD: dict[str, tuple[float, str]] = {
    "4tu": (2.51, "ns"), "guangzhou": (2.5, "ns"), "gprsd": (51.2, "samples"),
}
M3_2_ASSUMED = ("pad_gray", "gray_mapping", "detection_bg_window")
```

| 파라미터 | 기본값 | 근거 / 결정 상태 |
|---|---|---|
| `p_target_px` | 4.06 | [AW] `tpipe` 꼭짓점 크롭 p50. **D1 r2 결과 4.06 유지** |
| `overlap_px` | 192 | **잠정**(D1 r2 규칙 O 둘 다 불충족 → 75 %). N2-a 재학습 뒤 D1 r3에서 128/192 재판정 |
| `tile_px` | 256 | 학습 영상 256×256, imgsz 256([AP], [CV] U7) |
| `overlap_min_px`, `a_max_m`, `overlap_width_periods` | 64, 1.0 m, 20 | $O_\text{phys}$: AIHub `tpipe` 폭 p95 60.5 px, [D] §5 추적 상한, AIHub `tpipe` 최대 폭 80.7 px = 19.9 주기(v0.3) |
| `overlap_max_px` | 224 | 보폭 ≥ 32 px 보장(안전 상한, 가정). v0.3 식에서는 $O_\text{phys}\le 20P^*$(4.06 → 82, 6 → 120)라 $O_\text{cv}$ 128·192에서 작동하지 않는다 |
| `compress_max` | 1.25 | 압축 뒤 객체 축척 ≥ 0.8 = 학습 등방 증강 하한([AP] `scale 0.2`) |
| `period_*` | ×8, 4 샘플 | §2-2 측정과 같은 추정기 |
| `clip_percentile` | 99 | [D] §3 (6) |
| `v_assumed_m_per_ns` | 0.10 | 광저우 [RG] §4, Q7 |
| `pad_gray`, `gray_mapping`, `detection_bg_window` | 128, 대칭, 301 | **가정, M3-2(Q3·Q5)에서 정함** |

### 4-3. 출력 자료형

```python
@dataclass(frozen=True)
class LineTile:
    index: int
    x0_px: int                 # origin in line L1 (integer px)
    content_width_px: int      # L, or W1 when W1 <= L (the tile image is always L x L)
    owner_px: tuple[float, float]   # half-open [lo, hi) in line L1
    inner_left: bool           # left edge touches a neighbour tile
    inner_right: bool

    @property
    def seams_px(self) -> tuple[int, ...]:   # line-L1 columns of inner edges ([CV] name)
        # (x0_px,) if inner_left, + (x0_px + L,) if inner_right

@dataclass(frozen=True)
class DetectionImage:
    line_image: np.ndarray     # uint8 [L, W1]; rows >= content_height_px are pad_gray
    grid_map: ImageGridMap     # trace_offset 0, sample_offset 0, traces_per_px N/W1,
                               # samples_per_px n_s/H_c, n_traces N, n_samples n_s
    tiles: tuple[LineTile, ...]
    content_height_px: int     # H_c
    period_samples: float      # P_line
    period_source: str         # "measured" | "dataset_default" | "f_c"
    depth_m_per_sample: float  # delta_z = v_s dt / 2
    dx_m: float                # trace spacing of the line (V2 needs it, v0.3.3)
    v_scale_m_per_ns: float
    v_scale_source: str        # "eps_r_meta" | "depth_per_sample_rough" | "assumed"
    p_target_px: float
    overlap_px: int            # O actually used
    overlap_phys_px: int
    compression: float         # c (1.0 when not compressed)
    flags: tuple[str, ...]     # period_fallback, period_compressed, scale_out_of_range,
                               # overlap_period_capped, v_scale_assumed,
                               # amplitude_processed, single_tile, short_line_padded, flat_line
    assumed_params: tuple[str, ...]   # == M3_2_ASSUMED in M3-1

    def tile_image(self, k: int) -> np.ndarray:   # uint8 [L, L], C-contiguous copy;
        # columns >= content_width_px are pad_gray (short line only)
    @property
    def period_l1_px(self) -> float:              # P_line / samples_per_px
```

- 측선당 `ImageGridMap` 1개(Q8·U6)다. 오른쪽 끝을 지나거나 $H_c$ 아래(채움)에 있는 픽셀은 $u\,\sigma_x\ge N$ 또는 $v\,\sigma_y\ge n_s$에 대응한다. 그래서 기존 `image_box_to_grid_box`가 잘라 내거나 `None`으로 버린다([ADR4] 결정 4).

### 4-4. 알고리즘 (`make_detection_image`)

```python
def make_detection_image(
    radargram: Radargram,
    config: DetectionImageConfig = DEFAULT_DETECTION_IMAGE_CONFIG,
) -> DetectionImage
```

(1)–(6)은 **측선 전체 L0에서 한 번만** 한다([CV] P-3). 타일마다 하면 겹침 구역의 같은 반사가 두 타일에서 다른 픽셀이 된다.

1. **단면**
   - 원시 자료면 `prepare_fitting_branch(radargram, config.detection_bg_window).background_removed`([D] §3 (1)–(4))를 쓴다.
   - `amplitude_processed`면 저장 진폭에 트레이스별 DC 제거(`remove_dc`)만 한다. 플래그 `amplitude_processed`를 붙인다.
2. **주기** `measure_period_samples(section, pad_factor, min_period) -> float | None`
   - DC 제거한 트레이스마다 $\mathrm{rfft}$(길이 $2^{\lceil\log_2(8n_s)\rceil}$, 영 채움)를 하고 $|\cdot|$의 트레이스 평균을 낸다.
   - 주파수 $f\in[1/(n_s/4),\ 1/P_\min]$ cycle/샘플 구간에서 최대를 찾는다. 최대가 구간 양 끝 빈이거나 스펙트럼이 평탄하면(최댓값 0) `None`이다. 그 밖에는 $1/f_\text{peak}$다.
   - 없으면 `DATASET_DEFAULT_PERIOD[radargram.dataset]`(ns이면 $\div\Delta t$)를 쓰고 `period_source="dataset_default"`, 플래그 `period_fallback`을 붙인다.
   - 그것도 없으면 $1/(f_c\Delta t)$(`"f_c"`, `period_fallback`)를 쓴다. 모두 없으면 `ValueError`다.
   - 4TU는 1/f_c = 2.0 ns보다 실측 2.51 ns가 맞으므로 자료셋 기본값을 먼저 쓴다(§2-2).
3. **샘플당 깊이와 배율**
   - $\delta_z$ 출처
     - `eps_r_meta`가 있으면 $v_s$ = `velocity_from_eps(eps_r_meta)`, $\delta_z=v_s\Delta t/2$다(`"eps_r_meta"`, 4TU).
     - 아니면 `metadata["depth_per_sample_m_rough"]`가 있으면 $\delta_z$ = 그 값, $v_s=2\delta_z/\Delta t$다(`"depth_per_sample_rough"`, GPR-SD; dt 가설과 무관하게 $\delta_z$만 쓴다).
     - 아니면 $v_s$ = `v_assumed_m_per_ns`(`"assumed"`, 플래그 `v_scale_assumed`, 광저우)다.
   - 공칭 배율: $s_y=P_\text{line}/P^*$, $s_x=s_y\,\delta_z/\Delta x$.
   - 압축: $\operatorname{round}(n_s/s_y)>L$이면 $c=(n_s/L)/s_y$로 두 배율에 같은 $c$를 곱한다(등방 유지, 자르지 않음, 플래그 `period_compressed`). `compression` = $c$다. $c$ > `compress_max`(1.25)이면 `scale_out_of_range`를 더한다. 좌표 계약에는 영향이 없고 검출 축척만 학습 범위 밖이 된다(§10 Q9). 해당 측선은 광저우 MALA(기본값), $P^*$ 6에서는 광저우 IDS 일부다(§3).
   - 크기: $H_c=\max(1,\operatorname{round}(n_s/s_y))$, $W_1=\max(1,\operatorname{round}(N/s_x))$. 실제 배율은 $\sigma_y=n_s/H_c$, $\sigma_x=N/W_1$이고 이 값을 `ImageGridMap`에 넣는다.
   - 두 축 반올림 때문에 등방이 최대 $\pm0.5/\min(H_c,W_1)$ 상대로 어긋난다. 허용한다.
4. **gain**: 원시 자료는 `agc(section, dt_ns, f_c_hz)`([D] §3 (5))를 적용한다. `amplitude_processed`면 생략한다.
5. **클립 문턱**: $c$ = `np.percentile(|a|, clip_percentile)`, 측선 전체 L0 기준이다. $c=0$이면 플래그 `flat_line`을 붙이고 영상 전체를 `pad_gray`로 채운다.
6. **리샘플** `resample_area(a, (H_c, W_1)) -> float64`. 축마다 따로 한다(시간축 → 트레이스축).
   - 축소(출력 ≤ 입력): 적분 면적 평균이다. 셀 경계 $e_j=j\cdot n/m$에서 누적합 $C(x)=\sum_{i<\lfloor x\rfloor}a_i+(x-\lfloor x\rfloor)a_{\lfloor x\rfloor}$를 쓰고 $\text{out}_j=(C(e_{j+1})-C(e_j))/(n/m)$이다. 정수 배율이면 `cv2.INTER_AREA`와 같다.
   - 확대(출력 > 입력, 큰 $P^*$): 셀 중심 $(j+0.5)\,n/m-0.5$에서 `np.interp` 선형 보간을 하고 양 끝은 고정한다.
   - 앨리어싱: 반송파는 $1/P^*$ = 0.246 cycle/px로 새 나이퀴스트 0.5 아래다. 나이퀴스트 1.5배 성분의 상자 응답은 0.30–0.33이다(T5).
7. **양자화**: $q=\operatorname{rint}(127.5+127.5\operatorname{clip}(a/c,-1,1))$, uint8이다. 0 진폭 → 128 = `pad_gray`다. 클립 문턱은 5의 L0 값이고 리샘플한 값에 적용한다.
8. **측선 L1 영상**: `line_image = full((L, W1), pad_gray)`에 `line_image[:H_c, :] = q`를 넣는다. 내용은 위, 채움은 아래다([CV] P-2).
9. **겹침**(v0.3 정정)
   $$O_\text{phys}=\max\!\left(O_\min,\ \left\lceil\min\!\left(\frac{2a_\max}{\sigma_x\Delta x},\ w_P\,P^*\right)-10^{-9}\right\rceil\right),\qquad w_P=\texttt{overlap\_width\_periods}=20$$
   - 첫 항은 깊이 추적 상한(±$a_\max$)의 L1 폭이다. 둘째 항은 검출기가 학습에서 본 가장 넓은 쌍곡선 박스(20 주기)다. 깊이 단위 등방이라 L1에서 $\lambda/2$ = $P^*$ px이므로 $w_P\lambda/2$가 $w_P P^*$ px가 된다.
   - 박스는 둘 중 좁은 쪽보다 넓을 필요가 없다. 추적은 $a_\max$ 밖을 쓰지 않고, 검출기는 20 주기보다 넓은 박스를 학습하지 않았다.
   - 둘째 항이 이기면 `overlap_period_capped`를 붙인다. 측선은 빼지 않는다. 조건은 $2a_\max>w_P\,\lambda/2$, 곧 토양 속 반파장 $\lambda/2=v_sP_\text{ns}/2<0.1$ m다. $P^*$와 무관하고(L1 반올림 차만 있음), **고주파 또는 저속(큰 εr)** 측선에서 생긴다(v0.3.2 정정).
     - 광저우: 600·900·2000 MHz 51측선([CV] v0.3 `m31_cprime_tiles.json`).
     - 4TU: $P$ 2.51 ns이면 $v_s$ < 0.0797 m/ns, 곧 εr > 14.16인 측선이다. 메타 εr 15.18·16.89·17.36·19.46의 91측선이고, $P^*$ 4.06에서는 반올림 때문에 εr 14.06 측선 1개가 더해져 92측선이다(`analysis/processing/detection_image_capped_count.py`).
     - 플래그가 붙어도 실제 겹침은 바뀌지 않는다. 상한 값이 $20P^*$ = 82 / 120 px로 $O_\text{cv}$(128·192)보다 작기 때문이다. 4TU 759측선 모두 $O=O_\text{cv}$(128이면 128, 192이면 192)이다(같은 스크립트, $P^*$ 4.06·6).
     - 4TU의 메타 εr는 실측이 아니므로(헤더 기본값 포함) 이 플래그는 정보용이다. 판정에는 쓰지 않는다.
     - 실행 때는 측정 $P_\text{line}$을 쓰므로 위 건수(2.51 ns 가정)와 다르다. 주기가 짧게 재진 측선은 εr가 작아도 붙는다. 예: 03.9 Path5는 $P_\text{line}$ 21.33 샘플(2.09 ns), εr 11.11이라 $\lambda/2$ = 0.094 m < 0.1 m이고, 플래그가 붙는 것이 맞다(구현 `detection_image_flags.csv`, v0.3.5 확인).
   - $O=\min(O_\text{max},\ \max(O_\text{phys}, O_\text{cv}))$, $S=L-O$.
   - 값: 4TU(측선별 εr) 64–82 px($P^*$ 4.06), 91–120 px($P^*$ 6). 광저우 전 측선 ≤ 82 / ≤ 120 px(2000 MHz 2 $a_\max$ 환산 325 px → 82 px). 그래서 기본값에서 $O=O_\text{cv}$다(근거: `analysis/processing/detection_image_gz_height_check.py`, `detection_image_expected_values.py`).
10. **타일 배치** `tile_layout(width_px, tile_px, overlap_px) -> tuple[LineTile, ...]`([CV] §5-2와 같음)
    - $W_1\le L$이면 타일 하나다: $x_0$ = 0, `content_width_px` $W_1$, 소유 $[0, W_1)$, 내부 경계 없음, 플래그 `single_tile`. $W_1<L$이면 `short_line_padded`도 붙인다.
    - 그 밖에는 $x_0^{(k)}=kS$로 $kS+L<W_1$인 동안 늘어놓고, 마지막 타일을 $x_0=W_1-L$에 둔다(앞 타일과 같으면 중복하지 않음). 타일 수는 $1+\lceil (W_1-L)/S\rceil$다.
    - 소유 경계: $b_0$ = 0, $b_n=W_1$, $b_k=(x_0^{(k-1)}+L+x_0^{(k)})/2$(겹침 가운데). 타일 $k$는 $[b_k, b_{k+1})$을 소유한다.
    - `inner_left = k > 0`, `inner_right = k < n−1`.
11. **백엔드 입력**: 타일 영상은 `tile_image(k)`(uint8 $[L,L]$)이다. 짧은 측선은 `line_image[:, :W1]`을 왼쪽에 두고 나머지 열을 `pad_gray`로 채운다([CV] 선택 (c)).
    - 모든 타일이 YOLO `letterbox`에서 비율 1·채움 0이다(`InputTransform` **언제나 항등**).
    - 채움 열의 박스는 $u\,\sigma_x\ge N$이라 `image_box_to_grid_box`가 자르거나 버린다.

### 4-5. 공개 보조 함수 (단위 시험 대상)

| 함수 | 시그니처 | 반환 |
|---|---|---|
| `measure_period_samples` | `(section: np.ndarray, *, pad_factor: int = 8, min_period: float = 4.0) -> float \| None` | §4-4 2 |
| `line_scale` | `(n_samples: int, n_traces: int, dx_m: float, period_samples: float, depth_m_per_sample: float, p_target_px: float, tile_px: int = 256) -> LineScale` | `LineScale(width_px, content_height_px, traces_per_px, samples_per_px, s_x_nominal, s_y_nominal, compression)` (`compression` = $c$, 1.0이면 압축 없음) |
| `overlap_for` | `(traces_per_px: float, dx_m: float, p_target_px: float, config: DetectionImageConfig) -> tuple[int, int, bool]` | `(overlap_px, overlap_phys_px, period_capped)` |
| `tile_layout` | `(width_px: int, tile_px: int, overlap_px: int) -> tuple[LineTile, ...]` | §4-4 10 |
| `resample_area` | `(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray` | §4-4 6 |
| `to_gray` | `(a: np.ndarray, clip_level: float) -> np.ndarray` (uint8) | §4-4 7 |

### 4-6. 경계 조건

| 조건 | 처리 |
|---|---|
| `dt_ns` 또는 `dx_m`가 None | `ValueError("DetectionImage needs a physical scale; use ImageGridMap.identity")` |
| $N$ = 1 등 아주 짧은 측선 | $W_1\ge1$, 타일 하나(256×256, 오른쪽 채움). 03.9 Path5(9 트레이스) → $W_1$ = 6 px |
| 주기 측정 실패 | 자료셋 기본값 → $1/f_c$ → `ValueError` 순서, `period_fallback` |
| $H_c>L$ | 등방 압축, `period_compressed`. $c$ > 1.25이면 `scale_out_of_range` |
| 2 $a_\max$의 L1 폭 > $20P^*$(고주파 또는 큰 εr: 광저우 600·900·2000 MHz 51측선, 4TU εr > 14.16의 91–92측선) | $O_\text{phys}=20P^*$, `overlap_period_capped`(정보용), 측선 포함. 실제 겹침은 $O_\text{cv}$ 그대로 |
| MALA 실측 주기가 폴더마다 4.45–36.6 샘플로 흩어짐 | 측정값을 그대로 쓴다(범위 [4, $n_s$/4] 안). 불안정성은 §11-1 위험 R1로 기록하고 M3-2에서 본다 |
| 자료가 평탄 | `flat_line`, 영상 전부 `pad_gray` |
| `amplitude_processed` | (1)–(4)·(5) 생략, DC 제거와 (6)만 |

### 4-7. 기본값에서의 기대값 (εr 9, [PC] 주기; 시험 T3·T7의 기대값)

계산: `analysis/processing/detection_image_expected_values.py`(이 표·T2 타일 수·T5 상자 응답과 같은 식). $\sigma$는 반올림 뒤 실제 배율이다.
- **기대값의 정본은 이 스크립트의 출력이다**(v0.3.3). 표의 $P_\text{line}$·$\delta_z$·배율은 표시용 반올림값이라, 표 값을 다시 입력하면 $H_c$·$W_1$이 1 px 달라질 수 있다(예: 161/162, 7,890/7,891).
- T3 입력은 정밀식으로 준다. $c$ = 0.299792458 m/ns, $v=c/\sqrt{\varepsilon_r}$, 4TU $P_\text{line}=2.51/\Delta t$, $\delta_z=\Delta t\,v/2$. 광저우 30 ns는 $\Delta t=30/512$, $P_\text{line}=2.93/\Delta t$, $\delta_z=0.10\,\Delta t/2$다. GPR-SD는 $P_\text{line}$ 51.2, $\delta_z$ 0.012다.
- 4TU 행은 모두 εr 9로 계산했다. 03.9 Path5의 메타 εr 11.11이 아니라 순수 함수 시험용 값이다.
- 정수($H_c$, $W_1$, $O_\text{phys}$, 타일 수)는 완전 일치, 실수는 상대 $10^{-9}$로 비교한다.

| 사례 | $P_\text{line}$ (샘플) | $\delta_z$ (m) | $s_y$ | $s_x$ | $H_c$ | $W_1$ | $\sigma_x$ | $\sigma_y$ | $O_\text{phys}$ | 타일 수 ($O$ 128 / 192) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4TU 01.1 Path1 (304 tr, Δt 0.097) | 25.88 | 0.004847 | 6.3735 | 1.5445 | 80 | 197 | 1.54315 | 6.4 | 65 | 1 / 1 |
| 4TU 09.1 Path2 (12,186 tr, Δt 0.195) | 12.87 | 0.009743 | 3.1704 | 1.5445 | 161 | 7,890 | 1.54449 | 3.18012 | 65 | 61 / 121 |
| 같은 측선, $P^*$ 8 (압축) | 12.87 | 0.009743 | 2.0 | 0.9743 | 256 | 12,507 | 0.97433 | 2.0 | 103 | 97 / 193 |
| 광저우 1030_190117AF/LID10001 (710 tr, 30 ns, $P$ 2.93 ns 가정) | 50.01 | 0.002930 | 12.3166 | 1.5035 | 42 | 472 | 1.50424 | 12.19048 | 64 | 3 / 5 |
| GPR-SD CementRoad2_1 (9,411 tr) | 51.2 | 0.012 | 12.6108 | 5.0443 | 25 | 1,866 | 5.04341 | 12.84 | 64 | 14 / 27 |
| 4TU 03.9 Path5 (9 tr, Δt 0.098) | 25.61 | 0.004897 | 6.3084 | 1.5445 | 81 | 6 | 1.5 | 6.32099 | 67 | 1 / 1 |

- 실측 시험(T7 `data`)에서는 $P_\text{line}$을 측선에서 재므로 값이 달라진다. 이 표는 `line_scale`·`tile_layout` 순수 함수 시험에 주기를 직접 넣어 쓴다.

## 5. 구현 명세 — 타일 박스 병합 (`src/gpr_detection/pipeline.py`)

### 5-1. 자료형

```python
@dataclass(frozen=True)
class LineBox:
    class_id: int
    class_name: str
    score: float
    box: PixelBox              # line L1, continuous, before rounding
    tile_index: int
    tile_truncated: bool = False   # left/right edge within edge_px of an inner tile edge
    tile_clipped: bool = False     # pipe_band trimmed by the x arbitration
    n_tiles_seen: int = 1
    merged_from_tiles: tuple[int, ...] = ()   # sorted tile indices of the merged group ([CV] name)

@dataclass(frozen=True)
class MergeConfig:
    iou_min: float = 0.7           # [CV] R2
    iomin_truncated: float = 0.8   # [CV] R2
    edge_px: float = 2.0           # [CV] 5-3
    band_layer_overlap: float = 0.5
    physics_veto: bool = True
    veto_period_factor: float = 1.0    # V1: |dy1| > factor * P_L1
    veto_min_center_px: float = 2.0    # V2 lower bound
DEFAULT_MERGE_CONFIG = MergeConfig()
```

### 5-2. 함수

| 함수 | 시그니처 | 일 |
|---|---|---|
| `tile_box_to_line_box` | `(box: PixelBox, tile: LineTile) -> PixelBox` | $x$에 `tile.x0_px`를 더한다. $y$는 그대로 |
| `is_tile_truncated` | `(box: PixelBox, tile: LineTile, edge_px: float) -> bool` | `inner_left`이고 $x_1-x_0\le\delta$, 또는 `inner_right`이고 $(x_0+L)-x_2\le\delta$(내부 경계가 있는 타일은 늘 꽉 찬 타일). 측선 시작·끝과 짧은 측선의 채움 경계는 보지 않는다 |
| `detect_tiles` | `(image: DetectionImage, detector: Detector, edge_px: float = 2.0) -> list[LineBox]` | 타일마다 `detector.predict(image.tile_image(k))` → `model_box_to_image_box`(L2 → 타일 L1) → `tile_box_to_line_box` → 잘림 표시 |
| `physics_veto` | `(a: LineBox, b: LineBox, image: DetectionImage, config: MergeConfig) -> bool` | §5-3 V1·V2. $\Delta x$는 `image.dx_m`, $\sigma_x$·$\sigma_y$는 `image.grid_map`, $\delta_z$·$P_\text{line}$는 `image.depth_m_per_sample`·`image.period_samples`에서 얻는다 |
| `merge_line_boxes` | `(boxes: Sequence[LineBox], image: DetectionImage, config: MergeConfig = DEFAULT_MERGE_CONFIG) -> tuple[LineBox, ...]` | §5-3 전부. 출력 순서는 (class_id, box.x1, box.y1, tile_index) |
| `line_boxes_to_grid` | `(boxes: Sequence[LineBox], image: DetectionImage) -> list[tuple[LineBox, GridBox]]` | 기존 `image_box_to_grid_box(box, image.grid_map)`. 바깥쪽 반올림은 **여기서 한 번만**. `None`이면 버린다 |
| `detect_line` | `(radargram: Radargram, detector: Detector, image_config: DetectionImageConfig = ..., merge_config: MergeConfig = ...) -> LineDetections` | `make_detection_image` → `detect_tiles` → `merge_line_boxes` → `line_boxes_to_grid` |

`LineDetections = (image: DetectionImage, boxes: tuple[tuple[LineBox, GridBox], ...], tile_seam_box: int)`이다. `tile_seam_box`는 병합 뒤에도 `tile_truncated`인 출력 박스 수다([CV] 감시 지표 이름). 짝 있는 잘림이 모두 병합됐으면 0이고, 짝 없는 잘림(T8c)만 센다. 기존 함수(`model_box_to_image_box`, `image_box_to_grid_box`, `detections_to_grid_boxes`)는 그대로 둔다. 단일 영상 경로(AIHub·모로코)는 지금처럼 쓴다.

### 5-3. 병합 규칙

1. **잘림 표시**(`detect_tiles`): [CV] §5-3이다. 위·아래 변은 보지 않는다.
2. **비교 대상**: 서로 다른 타일에서 온 **같은 클래스** 박스 쌍만 본다(같은 타일 안은 백엔드 NMS를 거쳤다). 클래스가 다르면 합치지 않는다.
3. **`pipe_hyperbola`·`cavity`·`manhole`** (R2 + 거부권)
   1. 중복 간선: IoU ≥ `iou_min`, 또는 한쪽이라도 `tile_truncated`이고 IoMin(교집합 ÷ 작은 넓이) ≥ `iomin_truncated`이다.
   2. **물리 거부권**(`pipe_hyperbola`이고 `physics_veto`일 때만): `physics_veto(a, b)`가 참이면 간선을 만들지 않는다.
      - (V1) $|y_1^{(a)}-y_1^{(b)}|>$ `veto_period_factor`$\cdot P_{L1}$이다. 위 끝 차가 한 주기를 넘으면 다른 반사체(위아래로 놓인 관)로 본다. 수직 분해능은 $P/2$이고, 위 끝 검출 잡음은 ±1–2 px(0.25–0.5 P) [추정]이다.
      - (V2) 두 박스가 모두 잘리지 않았고 중심 간격 $|c_x^{(a)}-c_x^{(b)}|>\max(r_F/(\sigma_x\Delta x),\ \texttt{veto\_min\_center\_px})$다.
        - $r_F=\sqrt{\lambda d/2}$, $\lambda=2\delta_z P_\text{line}$, $d=\tfrac12(y_1^{(a)}+y_1^{(b)})\,\sigma_y\,\delta_z$(영점을 빼지 않아 크게 잡히므로 $r_F$가 넉넉한 쪽).
        - 잘린 박스는 중심이 치우치므로 V2를 쓰지 않는다.
   3. **연결 요소**를 만들고 대표 1개를 고른다. 우선순위는 다음과 같다([CV] §5-4 5).
      1. 잘리지 않은 박스
      2. 박스 중심 $c_x$가 자기 타일의 `owner_px` 안인 박스
      3. 높은 점수
      4. 낮은 `tile_index`
      5. 입력 순서
   4. **거부권 재확인**: 성분 안에서 대표와 `physics_veto`가 참인 구성원은 성분에서 빼고, 뺀 것끼리 3–4를 되풀이한다(사슬로 이어져 거부된 박스가 합쳐지는 것을 막는다). 거부권이 없으면 [CV] R2와 결과가 같다.
   5. 출력은 대표 박스 그대로다(좌표·점수 평균 없음, 트레이스 구간 합집합도 없음 — [CV] §5-4 5). `n_tiles_seen` = 성분의 서로 다른 타일 수, `merged_from_tiles` = 그 타일 번호(정렬), `tile_truncated` = 대표의 값이다. 세로 구간은 대표 것이다. 위 끝을 합집합(더 이른 쪽)으로 넓히면 꼭짓점 탐색이 얕은 층·영점 띠를 잡기 때문이다([D] §5 탐색 범위).
   6. 짝 없이 남은 잘린 `pipe_hyperbola`는 `tile_truncated=True`로 남긴다(§6에서 깊이 플래그).
4. **`pipe_band`** ([CV] §5-5, 물리 조건 추가)
   1. **층**: 같은 측선 띠 박스 가운데 세로 겹침이 작은 쪽 높이의 `band_layer_overlap`(50 %) 이상인 것을 연결 요소로 묶는다.
   2. **x 중재**: 층마다, 박스 $b$(타일 $k$)의 x 구간 중 다른 타일 $j$의 소유 구간에 있으면서 같은 층의 타일 $j$ 박스가 실제로 덮는 부분을 뺀다. 한 타일만 덮는 x는 남긴다(이웃이 놓친 길이를 잃지 않음).
      - 빼서 구간이 둘로 나뉘면 둘 다 남긴다. 폭이 0이 되면 버린다.
      - 줄어든 박스는 `tile_clipped=True`다.
   3. **$y_1, y_2$($k_0,k_1$)는 절대 바꾸지 않는다.** 띠 깊이는 박스 위 끝을 따라가기 때문이다(`band_box_convention_depth.md`).
   4. 결과 조각은 모두 깊이로 넘긴다. 묶기는 `group_band_boxes`([B] §3-4)가 한다.
      - 중재 뒤 조각은 맞닿거나(gap 0) 떨어져 있으므로 끝 5 픽 중앙값 비교 경로로 묶인다.
      - [CV] §5-5 "확인 필요"(위 끝이 1–2 px 다른 맞닿은 조각이 묶이는가)의 답: 묶음 후보 조건은 시간 겹침 ≥ 50 %와 gap ≤ $G$다. 판정은 박스 위 끝이 아니라 **픽** 시각 차(≤ 1.25 ns, UB18 기준)로 한다. 묶이는 조건은 두 가지다(v0.3.5 정정). (i) 위 끝 차가 작은 박스 높이의 절반 이하라 시간 겹침이 50 % 이상이어야 한다. (ii) 두 조각이 같은 사건을 픽해야 한다(창이 다른 겹을 잡으면 묶이지 않고 각각 `fragment_of_group`). v0.3.4의 "1–2 px 달라도 묶인다"는 얇은 박스(높이 1.2 P ≈ 5 px)에서만 맞는 표현이었다. 그 높이에서는 3 px 차부터 후보에서 빠진다(구현 시험 결과). AIHub형 띠 박스 높이(p5 13 / p50 16 / p95 21 px, [CV] §2)라면 6–10 px 차까지 (i)을 만족한다. 실측 이웃 타일 띠의 위 끝 차와 박스 높이 분포는 D3에서 보고한다(§11-1 R3). T8k·T8k-2로 확인한다.

### 5-4. 물리 근거 (v0.1 §5-1·§5-2)

- 겹침 하한
  - 타일 $k$가 $[kS, kS+L)$를 덮으면 폭 $w\le O$인 박스는 $k=\lfloor x_1/S\rfloor$ 타일 안에 온전히 들어간다. 마지막 타일은 측선 끝에 맞추므로 성립이 유지된다.
  - 박스 폭 상한은 v0.3 식이다: $\max(O_\min,\ \min(2a_\max, 20\lambda/2))$의 L1 환산이고, $O_\min$ 64 px는 AIHub `tpipe` 폭 p95 60.5 px보다 크다. 측선별 εr로 4TU 64–82 px($P^*$ 6이면 91–120 px), 광저우 고주파 측선도 ≤ $20P^*$다. 128 px는 모든 측선에서 이를 만족한다.
  - v0.2 식(미터 상한만)은 고주파에서 2 m = 80 주기 이상을 요구했다. 검출기 박스 폭(최대 20 주기)과 무관한 값이라 정정했다([CV] 발견 2).
  - 위치 민감도(AIHub 객체 중앙 쏠림)는 D1이 정한다. 물리 쪽 요구는 하한뿐이다.
- 수평 분해능: 같은 깊이의 두 관이 $r_F$보다 가까우면 한 쌍곡선이다.
  - 4TU($\lambda$ 0.25 m)의 $r_F$는 $d$ 0.3 / 0.7 / 1.5 m에서 0.19 / 0.30 / 0.43 m다.
  - 4TU 정답의 이웃 관 간격(289쌍)은 p50 0.20 m이고 63 %가 0.30 m 이하다([SV] `fourtu_gt_spacing`). V2가 $r_F$ 안의 쌍을 거부하지 않아도 분해 가능한 관로를 잃지 않는다.
  - [CV] C3d(폭 60 px 두 박스, 20 px 간격)는 4TU에서 0.62 m > $r_F$라 R2·V2 모두 2개다. 두 규칙은 충돌하지 않는다.
- 수직 분해능은 $P/2$다. V1 문턱 1 P는 검출 잡음 여유를 둔 값이다(**가정**, T8 기대값과 함께 D3 4TU 정성 확인에서 다시 본다).

### 5-5. cv-detection-expert와의 경계

| 이 문서(물리) | [CV] |
|---|---|
| 축척($P^*$·$s_x$ 식), 물리 거부권 V1·V2, 대표 박스 세로 불변, 띠 $k_0,k_1$ 불변, 깊이 연결 | 타일 배치·소유 구간·잘림 표시, R2 문턱, 대표 우선순위, 띠 x 중재 규칙, D1·D2·D3 평가, $P^*$·겹침 최종값 |

## 6. 구현 명세 — 측선 단위 깊이 호출 (`pipeline.py`, ADR-004 결정 7)

### 6-1. 자료형과 함수

```python
@dataclass(frozen=True)
class DepthBox:
    class_id: int                 # DetectionClass value
    grid_box: GridBox             # L0
    flags: tuple[str, ...] = ()   # "tile_truncated", "tile_clipped"

def depth_boxes_from(detections: LineDetections) -> list[DepthBox]

@dataclass(frozen=True)
class LineDepths:
    line_id: str
    inputs: DepthInputs                       # one zero for the whole line
    exclude_traces: np.ndarray | None         # bool [N] or None
    hyperbola: tuple[tuple[DepthBox, DepthResult], ...]   # input order
    pool: tuple[VelocityEstimate, ...]
    band: tuple[DepthResult, ...]             # estimate_band_depths order
    skipped: tuple[DepthBox, ...]             # cavity, manhole

def estimate_line_depths(
    radargram: Radargram,
    boxes: Sequence[DepthBox],
    depth_config: DepthConfig = DEFAULT_DEPTH_CONFIG,
    band_config: BandConfig = DEFAULT_BAND_CONFIG,
    *,
    line_id: str | None = None,              # default radargram.line_id
    survey_key: str | None = None,
    survey_pool: Sequence[VelocityEstimate] = (),
) -> LineDepths

def estimate_survey_depths(
    lines: Sequence[tuple[Radargram, Sequence[DepthBox]]],
    depth_config: DepthConfig = DEFAULT_DEPTH_CONFIG,
    band_config: BandConfig = DEFAULT_BAND_CONFIG,
    *,
    survey_key: str,
) -> list[LineDepths]
```

- import: `pipeline`은 `depth.estimate`(`prepare_depth_inputs`, `estimate_depth`, `DepthInputs`), `depth.velocity`(`velocity_estimate_from`, `VelocityEstimate`), `depth.band`(`band_exclude_traces`, `estimate_band_depths`), `depth.config`(`DepthConfig`·`BandConfig`·기본값), `io.classes.DetectionClass`를 쓴다.
- `depth`는 `pipeline`·`detect`를 import하지 않는다(기존 규칙, `tests/test_architecture_rules.py`).

### 6-2. 순서 (`estimate_line_depths`)

0. `depth_config.pick_mode != band_config.pick_mode`이면 어떤 계산도 하기 전에 `ValueError`다(영점·꼭짓점 위상 기준 일치, [ADR4] 결정 7).
1. **분류**: `class_id`가 `PIPE_HYPERBOLA`이면 `hyper`, `PIPE_BAND`이면 `band`, 나머지는 `skipped`다. 각 목록은 입력 순서를 유지한다.
2. **영점(측선당 한 번)**
   - `band`가 있으면 `exclude = band_exclude_traces([b.grid_box for b in band], N, radargram.dx_m, band_config)`, 없으면 `None`이다.
   - `inputs = prepare_depth_inputs(radargram, depth_config, exclude_traces=exclude)`. 측선 전체가 띠면 `find_time_zero`가 빼지 않고 `zero_band_overlap`을 붙인다(기존 동작).
3. **쌍곡선**: `hyper`의 $j$번째 박스마다 `r = estimate_depth(radargram, b.grid_box, depth_config, inputs=inputs)`. 모두 같은 `inputs`를 쓴다.
   - `"tile_truncated" in b.flags`이면 `r = dataclasses.replace(r, flags=r.flags + ("tile_truncated",))`(맨 끝에 덧붙임, 중복 없음)이다.
4. **속도 풀**: `pool = [e for j, (b, r) in enumerate(hyperbola) if (e := velocity_estimate_from(r, line_id=line_id, source_index=j, survey_key=survey_key)) is not None]`.
5. **띠**: `band`가 있으면 `estimate_band_depths(radargram, [b.grid_box for b in band], inputs.zero, pool, survey_pool, band_config, branch=inputs.branch)`, 없으면 `()`이다.
   - 결과 `r`의 `r.band.member_indices` 가운데 하나라도 `"tile_clipped" in band[m].flags`이면 `tile_clipped`를 덧붙인다(같은 `replace` 방식).
6. `cavity`·`manhole`은 깊이로 넘기지 않는다(ADR-002 결정 8). `skipped`에 남긴다.

`estimate_survey_depths`(두 단계, 지면결합 조사 단위 풀)
- 단계 1: 측선마다 0–4를 하고 측선별 `pool`을 모은다.
- 단계 2: `survey_pool` = 모든 측선 `pool`의 합(입력 측선 순서)을 쓰고, 측선마다 5–6을 한다. 같은 `inputs`를 재사용하고 영점을 다시 구하지 않는다.
- 조사 단위 키 추출(광저우 `zone`, GPR-SD 도로)은 호출자 몫이다(M5-1). 4TU(공기결합)는 쓰지 않는다.
- M3-1에서는 함수와 시험 TD6만 둔다.

### 6-3. `tile_truncated` 깊이 쪽 해석

- 짝 없는 잘린 쌍곡선은 깊이로 넘기고 플래그만 단다. 4TU 평가에서 빼지 않고 건수를 보고한다.
- 꼭짓점이 박스 안이면 4TU 깊이(메타 v 고정)는 유효하다.
- 꼭짓점이 잘린 쪽 3 트레이스 안이면 기존 `apex_at_box_edge`가 함께 붙는다([D] §5 6항).
- 한쪽 추적 → `insufficient_picks` → 곡률 v 없음이다. 광저우는 메타·풀 규칙으로 간다.

## 7. L1 방식이 깊이에 닿는 경로

깊이는 L0 `GridBox`와 피팅 갈래(gain 없음)만 쓴다. gain·클립·리샘플·회색 매핑은 깊이에 닿지 않는다(T9). 닿는 경로는 박스 창뿐이다.

| 경로 | 크기(기본값) | 영향 | 조치 |
|---|---|---|---|
| (a) 박스 위 끝 오차 | L1 1 px = 4TU 6.3 샘플(0.62 ns, 0.25 P, 깊이 환산 3.1 cm), 광저우 30 ns 12.2 샘플, GPR-SD 12.8 샘플(약 15 cm) | 꼭짓점 후보는 박스 안 첫 피크이고 위로 넓히지 않는다([D] §5). 위 끝이 꼭짓점 주엽(±0.6 P) 안으로 내려오면 주엽이 잘려 깊어지거나 `apex_at_box_top`이 붙는다 | **Q6: 계약 불변**. T10으로 측정만 한다. 위 끝이 꼭짓점보다 ≥ 0.6 P(L1 2.4 px) 위여야 한다는 요구를 cv에 전달(AIHub 라벨 위 여유 측정) |
| (b) 바깥쪽 반올림 | 각 변 < 1 셀, 병합 뒤 1회 | 무시할 수 있음 | T1·T8j |
| (c) 박스 폭 | AIHub형 46 px → 71 트레이스 | 좁으면 곡률 픽 수 감소(4TU 깊이는 무관) | M3 평가에서 폭 ≤ 0.6 m 비율 보고 |
| (d) 타일 경계 잘림 | 짝 있으면 병합에서 해소 | 짝 없으면 `tile_truncated`(§6-3) | 건수 보고 |
| (e) 병합 실패(중복) | – | 한 관로에 결과 2개 → 4TU 매칭에서 남는 추정 | T8 |

## 8. 시험 (python-expert)

공통
- 시드 20261113(`numpy.random.default_rng`).
- 합성은 `tests/synthetic_bscan.py`(`BScanSpec`, `make_radargram`)를 쓴다.
- 실측은 `@pytest.mark.data`, gprMax는 `@pytest.mark.synthetic`이다.
- 파일
  - `tests/preprocess/test_detection_image.py`: T1–T7 순수·합성 부분
  - `tests/test_pipeline_tiles.py`: T8·T1(타일 왕복)
  - `tests/test_pipeline_depth_order.py`: TD1–TD6·T9
  - `tests/test_detection_image_data.py`: T7 실측(`data`)
  - `tests/depth/test_l1_box_sensitivity.py`: T10(`synthetic`)

### 8-1. 영상·좌표 (T1–T7, T9–T10)

| ID | 내용 | 기대값 |
|---|---|---|
| T1 (C2) | **정확 왕복**(QA-8). §4-7 표의 6개 사례 `line_scale`로 `ImageGridMap`을 만든다. 무작위 격자 정렬 `GridBox` 200개(시드 고정, 타일 경계를 넘는 것 포함)를 L0 → 측선 L1(`grid_box_to_image_box`) → 그 박스를 온전히 담는 모든 타일(경계를 넘는 박스는 두 타일 모두)의 타일 L1(−$x_0$) → L2(`letterbox`의 `InputTransform`, 언제나 항등) → 타일 L1 → 측선 L1(+$x_0$) → L0(`image_box_to_grid_box`) | 모든 경로에서 **정확히 같은 `GridBox`**. 연속 박스 L1 ↔ L2 ≤ 0.5 px |
| T2 (C1) | **타일 배치** `tile_layout`. $W_1$ ∈ {1, 97, 255, 256, 257, 300, 383, 512, 2,250, 14,516, 27,628} × $O$ ∈ {0, 64, 128, 141, 192}, $L$ = 256 | 타일 수가 다음 표와 같다. 모든 열이 덮이고, 소유 구간이 $[0,W_1)$을 빈틈·겹침 없이 나누며, 마지막 타일 오른쪽 끝 = $W_1$이다. $W_1\le256$이면 1장·`single_tile`. 폭 ≤ $O$인 모든 정수 위치 구간 $[x, x+w)$이 어떤 타일 안에 온전히 든다 |
| T3 | **실제 배율**: `line_scale`의 6개 사례 | §4-7 표 값과 상대 $10^{-9}$ 안에서 일치. L0 트레이스 $i$에 세로선(진폭 1)을 그린 합성(DC·gain 끔 — `resample_area`만)을 측선 L1로 바꾸고, 최대 열 $u$에서 $\lfloor u\sigma_x\rfloor$ ∈ {i−1, i, i+1}. 샘플 축도 같다 |
| T4 | **주기 정규화**(v0.3.5 개정): `BScanSpec(f_c_hz=500e6, dt_ns=0.098, n_traces=401, n_samples=512, air_gap_m=0)`, `snr_db` ∈ {None, 20.0}(시드 고정) → `make_detection_image`. (a) `period_l1_px`. (b) **꼭짓점 창**에서 잰 주기: 꼭짓점은 `prepare_fitting_branch(...).background_removed`의 가운데 트레이스 포락선 최대(앞 60 샘플 제외)다. 이를 측선 L1로 옮긴 행 $r$·열 $c$에서 `line_image[r−2P*:r+4P*+1, c−7:c+8] − 127.5`에 `measure_period_samples(..., min_period=2.0)`를 쓴다 | (a) $P^*$ ±1 %. (b) $P^*$ ±10 %. 기대값(`analysis/processing/detection_image_t4_check.py`)은 잡음 없음 4.06 / 5.95 px, SNR 20 dB 4.41 / 6.48 px($P^*$ 4.06 / 6.0). xfail을 없앤다. 내용 전체에서 잰 주기는 판정하지 않는다(AGC가 잡음 구간을 키워 바뀜, R2) |
| T5 | **앨리어싱**: 길이 4,096 샘플의 정현파 $f$ = 0.75/s cycle/샘플(새 나이퀴스트의 1.5배)을 `resample_area`로 $s$ ∈ {3, 6.3, 12.6}배 축소 | 출력 RMS / 입력 RMS ≤ **0.36**(상자 응답 0.333 / 0.307 / 0.302). 반송파 $f$ = 0.246/s에서는 ≥ 0.85 |
| T6 | **letterbox 항등**: 모든 `tile_image(k)`가 shape (256, 256)이고 `letterbox(tile, 256)`의 `InputTransform`이 `scale_x == scale_y == 1`, `pad_x == pad_y == 0`(짧은 측선 포함, [CV] 선택 (c)). 짧은 측선 타일은 열 ≥ $W_1$이 모두 `pad_gray`이고, 그 열 안의 박스는 `image_box_to_grid_box`가 `None` | 예외 없음 |
| T7 | **경계·실측**: (a) `dt_ns=None` → `ValueError`. (b) 03.9 Path5 형(9 트레이스) → $W_1$ = 6, 타일 1장. (c) $P^*$ 8, 4TU Δt 0.195형($P$ 12.87 샘플) → `period_compressed`, $H_c$ = 256, `compression` = 1.243(≤ 1.25라 `scale_out_of_range` 없음). MALA형($n_s$ 512, $P$ 4.45 샘플, $P^*$ 4.06) → `compression` = 1.825, `scale_out_of_range`. (d) 평탄 자료 → `flat_line`. (f) 고주파형(Δt 0.039, Δx 0.01, $P$ 12.82 샘플, $\delta_z$ 0.00195) → `overlap_for`가 $O_\text{phys}$ 82, `period_capped` True, 플래그 `overlap_period_capped`. 4TU형: εr 9($P$ 25.61 샘플, Δt 0.098)은 2 $a_\max$ 환산 약 65 px라 False·$O_\text{phys}$ 65, εr 17.36(같은 Δt, $\delta_z$ 0.003526)은 약 90 px > 81.2라 True·$O_\text{phys}$ 82. 세 경우 모두 $O=O_\text{cv}$(기본값 192; `overlap_px=128`로 주면 128). (g) 메타데이터 회귀(원본 불필요): `docs/processing/fourtu_lines_v1.csv` 759측선에 $P$ 2.51 ns·메타 εr로 `line_scale`·`overlap_for`를 돌리면 `period_capped`가 $P^*$ 4.06에서 92측선, 6.0에서 91측선이고, 모든 측선의 `overlap_px` = 설정한 $O_\text{cv}$(기본 192, 128로 주면 128)이다(`analysis/processing/detection_image_capped_count.py`와 같은 값). 실측 실행에서는 $P_\text{line}$을 재므로 건수가 달라질 수 있다(T7(e)는 건수를 판정하지 않음). (e) `data`: §9 후보 실측 파일 전부에 `make_detection_image`를 하고 T1을 반복, `assumed_params == M3_2_ASSUMED`, GPR-SD는 `v_scale_source == "depth_per_sample_rough"`(dt 가설 A·B로 읽은 `line_image`가 같음), 광저우 IDS는 `"assumed"`·`v_scale_assumed`, 4TU는 `"eps_r_meta"`, MALA는 `amplitude_processed`, 광저우 2000 MHz는 `overlap_period_capped`. 압축된 측선은 `compression`을 보고한다(MALA_07011은 실측 주기 4.45 샘플이면 약 1.8). 실행 결과로 `overlap_period_capped`·`period_compressed`·`scale_out_of_range` 세 플래그의 측선별 값과 건수를 csv(`data/processing/detection_image_flags.csv`, git 제외)와 로그에 남긴다(판정 없이 보고) | 각 항목대로 |
| T9 | **깊이 불변**: 같은 L0 `GridBox`·합성 측선에서 `estimate_line_depths` 결과가 `DetectionImageConfig`($P^*$ 4.06/6.0/8.0, 겹침 128/192)와 무관 | `DepthResult` 전 필드 동일(`==`) |
| T10 | **박스 위 끝 민감도**(`synthetic`, 보고용): gprMax 20개 모델 정답 박스(M2-5 규칙) → 측선 L1 연속 박스 → (i) 정수 px로 바깥쪽 양자화, (ii) 위 끝 ±1·±2 L1 px 이동 → L0 → `estimate_depth` | (i) $\lvert\Delta d\rvert$ ≤ 1 cm(**가정** 기준, 넘으면 보고). (ii) 판정 없이 $\Delta d$·플래그 분포를 csv로 남긴다(`data/processing/l1_box_sensitivity.csv`, git 제외). Q6 재검토 근거 |

T2 기대 타일 수(행 $W_1$, 열 $O$ = 0 / 64 / 128 / 141 / 192)

| $W_1$ | 1 | 97 | 255 | 256 | 257 | 300 | 383 | 512 | 2,250 | 14,516 | 27,628 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 타일 수 | 1/1/1/1/1 | 1/1/1/1/1 | 1/1/1/1/1 | 1/1/1/1/1 | 2/2/2/2/2 | 2/2/2/2/2 | 2/2/2/3/3 | 2/3/3/4/5 | 9/12/17/19/33 | 57/76/113/125/224 | 108/144/215/240/429 |

### 8-2. 병합 (T8, `tests/test_pipeline_tiles.py`)

[CV] v0.2 §8-2의 C1–C5를 이 문서 ID로 합친 **한 벌**이다. python-expert는 아래 ID로 구현하고, 시험 함수 docstring에 두 ID를 함께 적는다.

| 이 문서 | [CV] | 주제 |
|---|---|---|
| T2 | C1 | 타일 배치·소유 구간($O$ 집합은 두 문서의 합집합 {0, 64, 128, 141, 192}, $W_1$은 27,628 추가) |
| T1 | C2 | 타일 경계를 넘는 박스의 정확 왕복 |
| T8a–T8e | C3a–C3e | R2 기본 사례 |
| T8f–T8i | C3j | 물리 거부권 V1·V2·사슬 |
| T8j | C3i | 반올림 1회 |
| T8k, T8l | C3f, C3g·C3h | 띠 중재, 한쪽 검출·순서·멱등 |
| T8n | C4 | 단일 타일 동등성 |
| T8m | C5 | 종단 가짜 검출기(`tile_seam_box` 0) |
| T3–T7, T9, T10, TD1–TD6 | – | 이 문서에만 있음 |

박스는 `LineBox`를 직접 만든다. `DetectionImage`는 `dx_m` = 0.02를 포함해 아래 값으로 직접 만든다(V2가 `image.dx_m`을 쓴다). 측선은 4TU형이다($\sigma_x$ 1.5445, $\sigma_y$ 6.32, $P_\text{line}$ 25.61 샘플 → $P_{L1}$ = 4.05 px, $\delta_z$ 0.004897, $\Delta x$ 0.02, $W_1$ = 1,000, $O$ = 128 → 타일 7장. 기본값과 무관하게 `overlap_px=128`로 **고정**한 픽스처다).

| ID | 내용 | 기대값 |
|---|---|---|
| T8a (C3a) | 같은 쌍곡선을 이웃 두 타일이 ±1 px 흔들림으로 냄 | 1개, 소유 타일 박스 |
| T8b (C3b) | 온전 + 잘림, 보이는 비율 f ∈ {0.3, 0.5, 0.7, 0.9} | 1개 = 온전한 박스, `tile_truncated` False, `n_tiles_seen` 2 |
| T8c (C3c) | 잘림만(이웃이 놓침) | 1개, `tile_truncated` True, `LineDetections.tile_seam_box` = 1 |
| T8d (C3d) | 폭 60 px 두 박스가 20 px 떨어짐(IoU 0.50) | 2개 |
| T8e (C3e) | 겹친 `pipe_hyperbola` + `cavity` | 둘 다 남음 |
| T8f (C3j-1) | **V1**: 같은 x, 폭 50 × 높이 60 px 두 박스(타일 0·1, 둘 다 온전), 위 끝 차 1.5 $P_{L1}$ = 6.1 px(IoU 0.82) | 2개(V1). `physics_veto=False`이면 1개(R2). 대조: 높이 25 px면 IoU 0.61이라 거부권과 무관하게 2개 |
| T8g (C3j-2) | V1 경계: 위 끝 차 0.5 $P_{L1}$ = 2.0 px(높이 60) | R2대로 1개 |
| T8h (C3j-3) | **V2**: 위 끝 $y_1$ = 40 px($d\approx$ 1.24 m → $r_F\approx$ 0.39 m = 12.7 px), 폭 120 px 두 박스 중심 간격 20 px(IoU ≈ 0.71) | 2개(V2). 간격 8 px → 1개 |
| T8i (C3j-4) | 사슬: 같은 x, 폭 50 × 높이 60 px, 위 끝 A 40, B 42.5, C 45 px(A–B·B–C는 2.5 px라 IoU 0.92·거부 없음, A–C는 5 px > $P_{L1}$로 V1 거부). A·C는 타일 0, B는 타일 1(A–C는 같은 타일이라 비교 안 함). 점수 A > B > C, 모두 온전, 중심은 각자 소유 구간 안 | 대표 A(B 합침, `n_tiles_seen` 2), C는 재확인 4로 빠져 따로 남음 → 출력 2개 |
| T8j (C3i) | **반올림 1회**: 병합 결과의 `line_boxes_to_grid` = 반올림 전 대표 박스에 `image_box_to_grid_box`를 한 번 한 값 | 같음. 안쪽으로 줄어든 변 없음 |
| T8k (C3f) | **띠**: 5개 타일에 걸친 띠 하나(위 끝이 타일마다 0/±1/±2 px 다름) + 6 ns 아래 다른 띠 | 층 2개. 층마다 조각이 겹치지 않고 맞닿으며 x 합집합 길이 보존. 모든 조각의 $y_1,y_2$가 입력과 같다. 합성 측선(평탄 띠 진폭 1)에 넣어 `estimate_band_depths`를 부르면 첫 층이 결과 1개로 묶이고 깊이 ±2 cm |
| T8k-2 (v0.3.5) | 띠 묶음 경계: 박스 높이 16 px(AIHub형)에서 이웃 조각 위 끝 차 3 / 6 / 9 px, 높이 5 px(1.2 P)에서 2 / 3 px. 평탄 띠 사건(진폭 1, 다른 층 없음)은 두 박스 모두의 세로 구간 안에 둔다 | 높이 16: 3·6 px는 결과 1개(묶임), 9 px(겹침 44 %)는 후보 아님. 높이 5: 2 px는 묶임, 3 px는 후보 아님. 모든 경우 $y_1,y_2$는 입력 그대로 |
| T8l (C3g·C3h) | 띠 한쪽만 검출, 타일 순서 섞기, 병합 두 번(멱등) | 그 길이가 남음(`tile_clipped` False), 결과 같음 |
| T8m (C5) | **종단 가짜 검출기**: 측선 L1 2,250 × 256에 사각형 50개(경계 위 포함, 폭 ≤ 120 px) → 타일 → 타일 안에서 보이는 부분만 내는 가짜 `Detector`(M1-8 `test_detect_contract.py` 방식) → 병합 → L0 | 사각형마다 정확히 1개, 각 변 오차 < 1 트레이스·1 샘플(바깥쪽만), 중복 0, 누락 0, `tile_seam_box` = 0 |
| T8n (C4) | **단일 타일 동등성**: $W_1\le256$ 측선(예: 03.9 Path5형 $W_1$ 6, 01.1 Path1형 197) | `detect_line`의 결과 = 타일 1장에 `detect_tiles`만 하고 병합 없이 `line_boxes_to_grid`를 한 결과(같은 `GridBox`·점수·순서) |

### 8-3. 깊이 호출 순서 (TD1–TD6, `tests/test_pipeline_depth_order.py`)

합성 측선은 `BScanSpec(eps_r=9, d_top_m=0.6, n_traces=401, air_gap_m=0)`(지면결합, 곡률 v 사용)이고, 필요하면 `extra_events`로 평탄 띠를 둔다. 쌍곡선 박스는 M2-5 규칙이다.

| ID | 내용 | 기대값 |
|---|---|---|
| TD1 | **띠 트레이스 제외**: 띠 박스 `GridBox(0, 199, k0, k1)`(얕은 평탄 띠, 진폭 2, 트레이스 0–199) + 쌍곡선 박스 1개. `prepare_depth_inputs`를 감싼 spy로 인자를 잡는다 | (a) spy가 **정확히 한 번** 불리고 `exclude_traces`가 `band_exclude_traces([band], 401, dx, band_config)`와 `np.array_equal`. (b) `LineDepths.inputs.zero.t_zero_ns` == `find_time_zero(radargram, prepare_fitting_branch(radargram, cfg.bg_window).filtered, pick_mode=..., exclude_traces=<같은 마스크>, config=cfg.zero_config()).t_zero_ns`(완전 일치). (c) 쌍곡선 결과와 띠 결과의 `t_zero_ns`가 모두 같은 값 |
| TD2 | 띠가 측선 전체(`GridBox(0, 400, ...)`) | `inputs.zero.flags`에 `zero_band_overlap`. 띠 박스를 빼고 부른 경우(spy `exclude_traces is None`)와 영점이 같다 |
| TD3 | 띠 없음 | spy `exclude_traces is None`. 결과가 `prepare_depth_inputs(radargram, cfg)` + `estimate_depth`를 직접 부른 값과 같다 |
| TD4 | **순서·풀**: 쌍곡선 1개 + 띠 1개. `estimate_depth`·`velocity_estimate_from`·`estimate_band_depths`에 spy | 호출 순서 = prepare → estimate_depth(쌍곡선 수만큼) → velocity_estimate_from(같은 수) → estimate_band_depths(1회). `estimate_band_depths`의 `pool` 인자 = `LineDepths.pool`, `branch is inputs.branch`, `zero is inputs.zero`. 전제로 `pool[0].eligible`을 먼저 확인(아니면 픽스처 오류로 실패)하고, 띠 결과 `v_rule == "neighbor_hyperbola"`, v = 쌍곡선 v |
| TD5 | **분류·플래그**: `cavity`·`manhole` 박스 각 1개, `tile_truncated` 쌍곡선 1개, `tile_clipped` 띠 조각 1개 | `skipped`에 2개이고 깊이 함수에 넘어가지 않음(spy). 쌍곡선 결과 `flags[-1] == "tile_truncated"`. 띠 결과 `"tile_clipped" in flags` |
| TD6 | `pick_mode` 불일치 → `ValueError`(spy 호출 0회). `estimate_survey_depths`: 두 측선(각 쌍곡선 1 + 띠 1) | 측선마다 `prepare_depth_inputs` 1회(단계 2에서 다시 부르지 않음). 각 측선의 `estimate_band_depths` `survey_pool` 인자 = 두 측선 `pool`의 합(측선 순서) |

## 9. M3-1 왕복 시험 실측 후보 (T7 (e))

| 데이터 | 파일 | 크기 [$n_s$ × N] | Δt (ns) / Δx (m) | 고른 이유 |
|---|---|---|---|---|
| 4TU | `05_radargrams_npy/03.9/Path5` | 512 × 9 | 0.098 / 0.02 | 최소 측선, 타일 1장·오른쪽 채움 |
| 4TU | `05_radargrams_npy/01.1/Path1` | 512 × 304 | 0.097 / 0.02 | 타일 1장, T 꼭짓점 있음(trace 105, [MR] G114) |
| 4TU | `05_radargrams_npy/09.1/Path2` | 512 × 12,186 | 0.195 / 0.02 | 긴 지그재그, $H_c$ 161, 타일 약 61장 |
| 4TU (선택) | `05_radargrams_npy/09.4/Path1` | 512 × 1,892 | 0.117 / 0.02 | 드문 dt |
| 광저우 | `pipe/001_190321AA/LID10001` | 512 × 119 | 0.0977 / 0.024 | 50 ns, 짧은 측선 |
| 광저우 | `pipe/1030_190117AF/LID10001` | 512 × 710 | 0.0586 / 0.024 | 30 ns, 타일 약 3장 |
| 광저우 | `pipe/190321A_190321AH/LID10001` | 2048 × 765 | 0.0195 / 0.024 | 큰 $n_s$, 주기 범위 상한 근처 |
| 광저우 | `pipe/MALA_07011/LA010001` | 512 × 208 | 0.391 / 0.0256 | MALA 처리본(`amplitude_processed`). 폴더 실측 주기 4.45 샘플이면 $H_c$ > 256 → 압축 약 1.8·`scale_out_of_range`(v0.3) |
| 광저우 | `pipe/1030_181224AB/LID10001` | 512 × 11,067 | 0.0391 / 0.01 | 2000 MHz, `overlap_period_capped`(v0.3), 가로 확대($s_x$ < 1) |
| GPR-SD | `CementRoad2_1` | 321 × 9,411 | A 0.3022 / 0.03 | 기본안(주기 맞춤), 타일 약 14장 |
| GPR-SD | `AsphaltRoad2_1` | 321 × 27,628 | A 0.3022 / 0.03 | 최장 측선 |

- 4TU 010.15·010.16 Path1은 바이트 동일 중복이라 뺐다(`docs/datasets/fourtu_duplicates.md`).
- GPR-SD는 dt 가설 A로 읽는다($\delta_z$만 쓰므로 B로 읽어도 L1은 같다. T7(e)에서 A·B 두 번 읽어 `line_image`가 같은지 함께 확인).
- 원본은 로더로 읽기만 한다.

## 10. 결정 상태

| 항목 | 상태 | 남은 일 |
|---|---|---|
| Q1 L1 방식 | 확정(C′) | – |
| 병합(R2 + V1·V2, 띠 중재, `tile_truncated`) | 확정 | V1 문턱 1 P는 D3(4TU 정성)에서 확인 |
| Q2 $P^*$ 4.06/6.0, 겹침 128/192 | $P^*$ 4.06 확정(D1 r2). 겹침 잠정 192(규칙 O 둘 다 불충족) | N2-a 재학습 → D1 r3에서 겹침 재판정. 구현은 기본값 4.06·192, 파라미터 유지 |
| Q3 채움 값·회색 매핑 | M3-2 | 가정값 128·대칭, `assumed_params` |
| Q4 GPR-SD 축척 | M3-1은 기본안 왕복만 | 라벨 축척은 M3-4 뒤 |
| Q5 검출 영상 배경 창 | M3-2 | 가정값 301 |
| Q6 박스 위 끝 여유 | 계약 불변 | T10 측정만 |
| Q7 $v_s$ | 메타 v, 광저우 0.10 가정 | – |
| Q8·U6 | 측선 L1 `x0` + 측선당 `ImageGridMap` | – |
| $O_\text{phys}$ 식(v0.3) | 정정 확정: $\min(2a_\max, 20\lambda/2)$, 고주파 측선 포함·`overlap_period_capped` | 사용자 결정 불필요. 현재 파라미터($O_\text{cv}$ 128·192)에서 실제 겹침이 바뀌는 측선 없음 |
| Q9 내용 높이 > 256 | **확정(A), 사용자 결정 2026-11-13**: 등방 압축, `compression` 기록, $c$ > 1.25면 `scale_out_of_range` | D3·M3-2에서 해당 측선(광저우 MALA, $P^*$ 6의 IDS 일부)의 검출 품질을 보고 다시 본다. 검토했던 선택지: (A) 압축 유지 / (B) 세로 겹침 타일(축척 보존, 타일 2차원화·위아래 잘림 표시·세로 소유 구간 추가, 깊은 타일에 지표 행 없음) / (C) $c$ ≤ 1.25만 압축, 넘으면 (B) |

## 11. 핸드오프

- **python-expert**: §4(`to_image.py`), §5(`pipeline.py` 병합), §6(`pipeline.py` 깊이 순서), §8 시험. 기본값은 §4-2. 실행은 `conda run -n gpr-dev python -I`, PowerShell 도구다. 구현 뒤 gpr-processing-expert verify(T1–T10·TD1–TD6 결과 대조).
- **cv-detection-expert**: D1 r3(N2-a 재학습 뒤 겹침 128/192 재판정; $P^*$는 r2에서 4.06), D3에서 V1 문턱과 `tile_truncated` 건수 확인, AIHub `tpipe` 라벨 위 여유(P 단위) 측정(Q6 근거). [CV] §5-5 "확인 필요"의 답은 §5-3 4-4다.
- **add-architect**: ADR-004 "위험" 갱신. L1 = 측선 L1 + 정수 px 타일(Q8), 병합은 `pipeline`의 측선 연속 좌표, 결정 7 연결 구현 위치는 이 문서 §6. 스키마 변경 없음.
- **cv-detection-expert(v0.3 어긋남 통지)**: 해소(CV v0.3.1). 높이 > 256 처리(Q9 확정 (A) 압축), 2000 MHz 측선 포함, 2,048 샘플 측선 높이, 절 번호 인용 5건 모두 [CV] v0.3.1에 반영됐다.
- **gpr-dataset-curator**: **데이터셋 카드 기록 후보**. GPR-SD 우세 주기 약 51 샘플(배경 처리본, 측선 전체·관로 표식 창 모두)이 문서의 200–400 MHz·dt 가설 A(62 MHz)·B(73 MHz)와 맞지 않는다([PC], 탐색). 판정 아님.

### 11-1. 위험 (M3-2 이후에 볼 것)

| ID | 위험 | 근거 | 다룰 곳 |
|---|---|---|---|
| R2 (v0.3.5) | 검출 영상의 AGC((5), 창 $8/f_c$)가 잡음·필터 끝 구간을 키워, L1 **전체**에서 잰 주기가 $P^*$보다 길어진다. 실측 4TU 6측선 4.57–5.25 px(+13–29 %, AIHub 크롭 전체 기준 p50 4.06·p90 4.57). 잡음 없는 합성은 17 px, SNR 20 dB는 7.4 px다. 관 반사 자리(꼭짓점 창)의 주기는 3.37–4.41 px로 $P^*$ 근처다. 그래서 축척 규칙(배경 제거 후·gain 전 $P_\text{line}$)은 맞고, 바뀌는 것은 영상 질감이다 | `analysis/processing/detection_image_t4_check.py` | M3-2 영상 품질: gain 종류·창과 AIHub 질감(크롭 전체 주기) 맞춤. M3-1 알고리즘은 바꾸지 않는다 |
| R3 (v0.3.5) | 얇은 띠 박스(높이 ≤ 2 × 위 끝 차)는 이웃 타일 조각이 묶이지 않아 띠 결과가 둘로 나뉜다 | 구현 시험(높이 1.2 P, 3 px) | D3: 실측 이웃 타일 띠 조각의 위 끝 차·높이 분포와 미묶음 건수 보고 |
| R1 | 광저우 MALA의 측정 주기가 폴더마다 4.45–36.6 샘플로 흩어진다. 처리본이라 저주파 잔여나 gain이 섞인 것으로 보인다. 그래서 $s_y$와 압축 여부가 측선마다 크게 바뀌고, 높이 > 256 측선 수가 주기 가정에 따라 8–75개로 달라진다 | `analysis/processing/detection_image_gz_height_check.py`(폴더 첫 파일 6개) | M3-2 영상 품질: MALA 주기 추정(창 제한·폴더 단위 대표값)과 D3에서 `compression`·`scale_out_of_range` 측선의 검출 품질 확인. M3-1은 측정값을 그대로 쓴다 |

### 11-2. N2-a 학습 축척 증강의 물리 조건 (v0.3.6, 2026-11-13 의견 기록)

N2-a 재학습(`docs/experiments/m31_n2a_retrain_plan.md`)이 Ultralytics `scale`을 0.2에서 0.5(등방 배율 0.5–1.5)로 넓히는 경우의 물리 쪽 의견이다. 판정이 아니라 조건이다.

1. **등방 축척은 곡률(속도) 단서를 보존한다.**
   - 깊이 단위 등방 영상에서 점산란 쌍곡선은 $z=\sqrt{z_0^2+x^2}$다. 배율 $s$를 걸면 $z_0\to s z_0$인 같은 꼴이 되고, 점근 기울기(45°)는 그대로다. 꼭짓점 깊이와 크기만 바뀐다.
   - 이는 εr을 고정한 채 길이와 파장을 함께 늘린 전자기 상사와 같다. 쌍곡선 폭을 주기로 잰 값(AIHub `tpipe` 11.4 P)도 보존된다.
   - 곡률을 망가뜨리는 것은 비등방 배율뿐이다([AP] 금지 항목).
2. **C′ 전제와 충돌하지 않는다.**
   - C′는 추론 입력의 주기를 $P^*$에 맞추는 정규화다(§4-4 3).
   - 학습 주기가 $s\cdot4.06$ px로 퍼져도 $P^*$ 4.06이 그 안에 들어간다. 그래서 주기 추정 오차(§11-1 R1)를 견디는 쪽으로 돕는다.
3. **하한 ≥ 0.55를 권고한다.**
   - $s$ = 0.5이면 반송파 주기가 2.0 px로 나이퀴스트(2 px)에 닿는다. 축소할 때 반송파가 앨리어싱되거나 뭉개져, 실측에 없는 질감이 생긴다.
   - $s\ge0.55$이면 $P\ge2.2$ px로, 기존 픽셀 밴드패스의 유효 하한(0.45 cycle/px, [D] §3)과 같다.
   - 하한을 더 낮추려면 축소할 때 앨리어싱 방지 보간을 쓰는지 먼저 확인한다.
4. **상한 1.5는 겹침과 맞는다.**
   - AIHub `tpipe` 최대 폭 80.7 px는 1.5배면 약 120 px다. 겹침 192 px(§4-2 잠정 기본) 안에 들어간다.
   - 크롭 경계 잘림이 늘어나는지는 cv 쪽에서 본다.
   - 학습 박스 폭이 주기로 보존되므로 $O_\text{phys}$의 주기 상한($20P^*$, §4-4 9)은 바꾸지 않는다.
5. **`compress_max`(1.25)는 재학습 뒤 다시 본다.**
   - 현재 값은 학습 축척 하한 0.8의 역수다(§4-2).
   - 재학습 하한이 $s_\min$이면 1/$s_\min$(0.55 → 1.82)으로 고치는 것을 D1 r3 뒤에 판정한다. 광저우 MALA의 압축비(최대 약 1.8)가 이 범위에 든다.
6. 깊이는 L0만 쓰므로 이 증강의 영향을 받지 않는다(§7).
## 12. [CV] 가정·결정 후보에 대한 물리 쪽 답

| [CV] 항목 | 답 |
|---|---|
| P-1 축척 | 세로 주기 맞춤 + 가로 깊이 등방(§4-4 3). 쌍곡선 기준 주기는 4.06 px다(띠 크롭은 4.20). 4.3 px와의 차이는 등방 증강 0.8–1.2 안이다 |
| P-2 높이 256, 지표 위 | 같다. 내용 4TU 80–161·광저우 IDS 42–96·GPR-SD 25 px, 아래 채움(§4-4 8). 광저우 MALA는 256 초과 → 압축(Q9) |
| P-3 측선 단위 정규화 | 같다. (1)–(6) 모두 측선 L0에서 한 번(§4-4) |
| P-4 한 측선 한 배율 | 같다. 측선당 `ImageGridMap` 1개 |
| U1 | C′ |
| U2 | 물리 하한(v0.3 식) 4TU 64–82 px($P^*$ 6: 91–120), 전 측선 ≤ $20P^*$. 128·192 모두 만족. D1이 정한다 |
| U3 | R2 + V1·V2, 대표 세로 불변(§5-3 3) |
| U4 | 조각 전달 확정. $k_0,k_1$ 불변. 맞닿은 조각 묶임 조건은 §5-3 4-4, T8k |
| U5 | 플래그와 함께 깊이로(§6-3) |
| U6 | 측선 L1 `x0`(Q8) |
| U7 | 256×256, imgsz 256 |

## 변경 이력

| 날짜 | 판 | 내용 |
|---|---|---|
| 2026-11-13 | v0.1 초안 | 축척 조사([SV]), 주기 측정([PC]), 대안 비교, 추천안 C′, 병합 물리 요구, 깊이 영향 경로, 시험 초안, 후보 파일, Q1–Q8, [CV] 맞춤 |
| 2026-11-13 | v0.2 | 사용자 결정 반영(Q1 C′, R2 + V1·V2, 띠 $k_0,k_1$ 불변, `tile_truncated` 깊이로, Q2·겹침은 D1 파라미터, Q3·Q5 M3-2 가정값, Q4·Q7, Q6 불변, Q8 측선 L1 `x0`). §4·§5·§6을 구현 명세로 다시 씀(시그니처·모듈·기본값). §6 측선 단위 깊이 호출 순서(ADR-004 결정 7) 추가. 시험 T1–T10·T8a–m·TD1–TD6 기대값. v0.1의 칸 정렬 타일 방식과 세로 타일은 Q8 결정과 압축 규칙으로 대체 |
| 2026-11-13 | v0.3 | [CV] v0.2 반영: $O_\text{phys}$ 식 정정($\min(2a_\max,20\lambda/2)$, 고주파 측선 포함·`overlap_period_capped`), 높이 > 256 정정(광저우 MALA는 기본값에서도 생김, `compression`·`scale_out_of_range`, Q9 신설), 짧은 측선도 256×256 L1(letterbox 항상 항등), 진단 이름 [CV]와 통일(`seams_px`·`merged_from_tiles`·`tile_seam_box`·`short_line_padded`), 시험 한 벌로 통합(T8n = C4, C1–C5 대응표, T2에 $O$ 141), 실측 후보에 MALA·2000 MHz 추가. 근거 `detection_image_gz_height_check.py`, `detection_image_expected_values.py` |
| 2026-11-13 | v0.3.1 | Q9 사용자 결정 (A) 기록(§0 결정 표, §0-1 발견 3 반영 칸, §10). 다른 내용 변경 없음 |
| 2026-11-13 | v0.3.2 | `overlap_period_capped` 조건 정정: 고주파만이 아니라 반파장 < 0.1 m인 측선 전부(4TU εr > 14.16의 91–92측선 포함, [CV] v0.3 관찰). §4-4 9·§4-6·T7(f)(g) 갱신, 겹침 결과 불변 확인(4TU 759측선 모두 128). 근거 `detection_image_capped_count.py` |
| 2026-11-13 | v0.3.3 | integrity-checker 감사 반영: $P^*$ 6의 4TU $O_\text{phys}$ 92–120 → 91–120(4곳), `measure_period_samples`를 새로 만드는 이유와 기존 `dominant_period_px`·`estimate_period_px`와의 차이(§4-1), `DetectionImage.dx_m` 추가(V2, T8h), §4-7 기대값 정본 = 스크립트 출력·정밀 입력식, 해소된 [CV] 어긋남 통지 정리, Q9 확정 문구(§3), §11-1 위험 R1(MALA 주기), [CV] 인용 갱신(§5-4 5, C3d), T7(e) 플래그 건수 보고, 변경 이력 정렬 |
| 2026-11-13 | v0.3.4 | D1 r2(ea7f447, [CV] §8-5) 반영: `p_target_px` 4.06 유지, `overlap_px` 기본 128 → **192(잠정)**, N2-a 재학습 뒤 D1 r3에서 재판정(§0 Q2, §4-2, §10 Q2, §11). 기본값에 묶인 기대값만 고침(T7(f)(g)는 $O=O_\text{cv}$). T2·§4-7·T8·T9는 고정 파라미터라 그대로다 |
| 2026-11-13 | v0.3.5 | 구현 확인 3건 반영: T4 (b)를 꼭짓점 창 주기로 개정(`min_period=2.0`, 잡음 없음·SNR 20 dB 기대값, xfail 제거)하고 전체 주기의 AGC 영향은 R2(M3-2)로 둠. §5-3 4-4 띠 묶음 조건 정정(위 끝 차 ≤ 작은 높이의 절반)과 T8k-2·R3 추가. 03.9 Path5 `overlap_period_capped`가 맞음을 확인(§4-4 9). 근거 `detection_image_t4_check.py` |
| 2026-11-13 | v0.3.6 | §11-2 신설: N2-a 학습 축척 증강의 물리 조건(등방 곡률 보존, 하한 ≥ 0.55, 상한 1.5와 겹침 192, `compress_max` 재검토). 2026-11-13 리포트 의견을 본문에 기록. 다른 내용 변경 없음 |
