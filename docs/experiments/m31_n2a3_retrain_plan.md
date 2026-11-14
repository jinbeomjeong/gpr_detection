# M3-1 후속 N2-a″ — 띠 라벨 규칙(K32) 재학습 계획

- 상태: **v1.0 봉인 판** (2026-11-14, cv-detection-expert). **판정 완료(2026-11-14): A1·B-p 충족, B 미충족 → 불채택**(N2-a′ 기준선 유지). 판정서는 [`m31_n2a3_a1b_judgment.md`](m31_n2a3_a1b_judgment.md)다. 아래 상태 줄과 규칙 본문은 봉인 당시(학습 전) 그대로 둔다.
  - 사용자 결정 U3-a–U3-f(§9)와 python-expert 구현 결과(§7-0: 수정 뒤 전체 시험 1,410 passed, N2-a·N2-a′ 동작 비트 동일, 확인 필요 5건 결정, 구현 차이 승인)를 반영했다.
  - 사전 확인 S-0–S-4는 두 seed 모두 통과했다(§5-3, dirty 트리 실행). 봉인 HEAD에서 다시 돌려 확정한다(§7-2).
  - 봉인은 두 커밋이다. **C1** = 이 문서 v1.0 + 설정 + 구현 + 시험. **C2** = C1 HEAD에서 다시 만든 진단 json만 더한 커밋이다. `<SEAL>`은 C2다(§7-2).
- 선택 경위: 2026-11-14 사용자 결정. M3-1 완료(a24ab98) 뒤 다음 작업으로 N2-a″를 골랐다(D1 r4 판정서 §8 N6-b의 재학습 부분).
- 현 기준선: N2-a′ [`2026-11-14_n2a2_baseline.md`](2026-11-14_n2a2_baseline.md). 이전 계획·판정서 약칭은 다음과 같다.
  - [P2] [`m31_n2a2_retrain_plan.md`](m31_n2a2_retrain_plan.md) v1.1
  - [J1] [`m31_n2a2_a1_judgment.md`](m31_n2a2_a1_judgment.md)
  - [J2] [`m31_n2a2_d1r4_judgment.md`](m31_n2a2_d1r4_judgment.md)
  - [T] [`m31_detection_input_tiling.md`](m31_detection_input_tiling.md)
  - [SP] [`spec_detection_image_v1.md`](../processing/spec_detection_image_v1.md)(검출 영상·타일·병합 명세, v0.4 기본값 $P^*$ 6·$O$ 192·`compress_max` 1.35)
- 데이터: 매니페스트 `data/splits/aihub_yolo_v1.csv`, AIHub 세션 그룹 5-fold. fold 0 학습 영상은 11,276장이다. test는 쓰지 않는다.

## 0. 요약

| 항목 | 내용 |
|---|---|
| 목적 | N2-a′의 띠 보조 재현율 하락(61.29 %, U-3 목표 64.1 미만)이 맥락 모드의 짧은 band 조각 라벨 때문인지 가르고, 맞다면 회복한다 |
| 바꾸는 것 (U3-a: K32) | 맥락 모드에서 **이 단계가 자른** `pipe_band` 박스 중 출력 창에서 보이는 폭이 32 px 미만이면 버린다. 영상 픽셀, 다른 클래스, 기본 모드, 다른 단계는 N2-a′와 같다 |
| 인과 분리 | 같은 seed에서 증강 영상이 N2-a′와 **바이트 동일**하고, band 라벨은 N2-a′의 부분집합이다(시험 K-1·K-2, 사전 확인 S-2) |
| 채택 (사전 등록) | 다음 넷을 모두 만족해야 한다. A1(76.8 / 90.9), B(띠 보조 재현율 평균 ≥ 64.1, N2-a′보다 높은 fold 4개 이상), B-p(띠 보조 정밀도 평균 ≥ 49.7), A2′(D1 r5 규칙 P = 6, 규칙 O = 75 % 또는 50 %) |
| 사전 확인 | S-0–S-4(CPU, 학습 전, §5). S-0은 v0.2에서 새로 넣은 전제 확인이다(§1-3) |
| 자동 진행 | N2-a′와 같다. 채택 판정 커밋까지 간다. A1·B·B-p·A2′ 중 하나라도 미충족이거나, 게이트 실패, 감사 차단, S 위반, 러너 비정상 종료가 나오면 그 자리에서 멈춘다(§7-3) |
| 비용 | GPU 약 3.5 h, 감사 4회 |

## 1. 근거

### 1-1. 증상 (N2-a′, crop, `last.pt`, 5 fold, seed 0)

| 지표 | M1-10 | N2-a | N2-a′ | U-3 목표 |
|---|---|---|---|---|
| 전체 mAP50 | 79.78 | 80.71 | 79.59 | ≥ 76.8 |
| pipe_hyperbola AP50 | 93.95 | 92.26 | 91.61 | ≥ 90.9 |
| 띠 보조 재현율(conf > 0.25, 길이 가중) | 69.52 | 69.38 | **61.29** ± 7.10 | ≥ 64.1 |
| 띠 보조 정밀도 | 71.93 | 75.10 | 75.66 | ≥ 49.7 |
| 쌍곡선 오검출(conf 0.25, 5 fold 합) | 68 | 91 | 127 | – |

fold별 띠 보조 재현율(%, fold 0–4)과 짝지은 차

| 모델 | fold 0 / 1 / 2 / 3 / 4 | 짝지은 차 평균 ± SD (ddof=1) |
|---|---|---|
| M1-10 | 73.40 / 72.26 / 69.94 / 60.07 / 71.92 | – |
| N2-a | 72.70 / 67.12 / 73.18 / 58.25 / 75.63 | N2-a − M1-10: −0.14 ± 3.69 |
| N2-a′ | 64.69 / 54.01 / 57.79 / 57.97 / 71.97 | N2-a′ − M1-10: −8.23 ± 7.45, N2-a′ − N2-a: −8.09 ± 6.31 |

### 1-2. 원 crop의 band 폭 (v0.1 작성 때 잰 값, fold 0 학습 집합 v1 라벨, band 2,748개)

| p1 | p5 | p25 | p50 | p95 | < 32 px | 32–64 px | crop 가장자리에 닿음 |
|---|---|---|---|---|---|---|---|
| 51.2 | 62.0 | 89.0 | 114.1 | 210.0 | **0** | 155 (5.6 %) | 687 (25 %) |

- 원 crop에는 32 px 미만 band 라벨이 없다. 가장자리에서 잘린 band는 있지만(25 %) 모두 51 px 이상이다.
- **정의(v1.0에서 명시, v0.1 측정 스크립트로 재확인)**
  - 폭: v1 라벨 txt의 정규화 폭 × 256이다. 반올림하지 않은 실수이고, 구간은 반열림 [하한, 상한)이다.
  - 그래서 32–64 px 155개는 32 ≤ w < 64다. 0.1 px로 반올림해도 155개로 같다. 64 px를 닫힌 경계로 넣으면 169개다(w = 64 근처 14개).
  - 가장자리: x1 ≤ 2 px 또는 x2 ≥ 254 px(허용치 2 px)이면 닿은 것이다. 허용치 0.5 px로 세면 271개다.
  - 진단 json(`original_crop_band_widths`)의 `counts_plan_convention`(155 / 687)은 이 정의와 같다. `counts_raw`(164 / 271)는 다른 정의다: 폭을 xyxy 뺄셈(`x2 − x1`, loader float64)으로 구하고 가장자리 허용치 0.5 px. txt 폭이 정확히 64 px인 박스 9개가 뺄셈의 부동소수 오차로 64 미만이 되어 [32, 64)에 들어가므로 155 + 9 = 164다(json `width_64_counted_below_64_in_raw` = 9). 보고 값이며 판정과 무관하다(2026-11-14 python-expert 확인, 메인 세션 기록).

### 1-3. 정정: "창에 들어온 band의 30.5 %"의 실제 뜻 (v0.2)

- [J1] §4와 v0.1 §1-2는 `n2a2_band_drop_diag.json`의 `band_length_px_entering_window`(0–32 px 416 / 1,362)를 "창에 들어온(남은) band의 30.5 %"로 읽었다.
- 스크립트(`n2a2_band_drop_diag.py` 306–308행)를 확인하니 이 값은 다르다.
  - 창 단계 **입력** band 박스 전부의 **띠(strip) 좌표 폭**이다. 열 소유 구간으로 자른 뒤이고, 축척과 창 자르기 전이다.
  - 창 밖으로 버려진 박스도 포함한다. 1,362 = 남김 600 + 버림 762다.
- 그래서 **학습에 실제로 남은 band 라벨 중 32 px 미만 비율은 아직 재지 않았다.** 짧은 조각의 상당수는 창 밖에 있어 이미 버려졌을 수 있다.
- 결과: 가설 (1)의 정량 근거가 v0.1에서 적은 것보다 약하다. 그래서 §5에 **S-0(전제 확인)** 을 넣는다. N2-a′ 규칙에서 남은 band 라벨의 짧은 조각 비율이 작으면, K32는 학습 라벨을 거의 바꾸지 못하므로 학습하지 않고 멈춘다.
- [J1]은 커밋된 판정서라 고치지 않는다. 판정(A1 충족)에는 영향이 없고 원인 해석 문단만 해당한다. 정정은 이 계획과 N2-a″ 판정서에 공개한다.

### 1-4. 가설과 범위

- (1) **주 가설.** 맥락 모드가 원 분포에 없는 짧은 band 조각을 정답으로 만든다. K32가 겨누는 대상이다.
- (2) 중심 band 손실(맥락 모드 5.5–8.7 %)과 (3) 이웃 라벨 불일치(XOR 0.41)는 건드리지 않는다.
- K32를 맥락 모드에만 거는 이유가 있다. N2-a는 채움 이동 0.5로 짧은 조각을 만들고도 띠 재현율 69 %였다. 채움 이동 조각은 회색 채움에 닿아 끝나지만, 맥락 조각은 띠가 실제로 이어지는 자리에서 잘린다.
- D3 띠 경계 오검출([`m31_d3_4tu_qualitative.md`](m31_d3_4tu_qualitative.md) §8)은 범위 밖이다. 채택 판정 뒤 보고 전용으로 다시 본다(§7-4).
- 쌍곡선 대책도 넣지 않는다. 넣으면 원인을 가를 수 없다(위험은 §8 1번에 공개).

## 2. 가져오는 것 (N2-a′ 그대로)

- 설정: 새 설정 `configs/train_yolo11m_n2a3.yaml`은 `configs/train_yolo11m_n2a2.yaml`(sha256 b9416b57e8877ba8c6b696fc6a3e63ba87b18bc608664e2613ff4c3d3ee142ef)의 사본이다. 바꾸는 것은 다음 둘뿐이다.
  - `gpr_context:` 블록에 `band_min_clipped_width_px: 32` 한 줄을 더한다.
  - 머리 주석의 계획 문서 경로를 바꾼다.
- 증강: [P2] §2–§3 전부와 [`augmentation_policy.md`](augmentation_policy.md)의 금지 항목.
  - 맥락 모드 0.5(축척 U[1.0, 1.5]) / 기본 모드 0.5(축척 U[0.55, 1.5]), 창 이동 ±96
  - 채움 이동 0.15, 채움 띠 0.15(아래 0.5)
  - Ultralytics `scale` 0, translate 0.1, fliplr 0.5
  - **flipud**·degrees·shear·perspective·mosaic 0, GPR 증강 강도 0
- 맥락 인덱스: `data/aihub_context_index/`. crops.csv sha256 e10192bfc42c8781769acb81556225996f0b069744b91179a787cc5fe923bbbf, groups.csv sha256 f1fc261d18eb91d0bfdc6fdbf252a97d0e80aa2ece4ac2a8764c9296a9a38570. 누수 단언은 [P2] §4 그대로다. K32는 라벨을 줄이기만 하므로 새 누수 경로가 없다. 봉인 전에 fold 0–4 `--dry-run`이 단언 0건인지 확인한다.
- 학습 고정값: scratch(`pretrained: false`), seed 0, `augment_seed` 0, 50 epoch, batch 64, imgsz 256, AdamW lr0 0.00125·momentum 0.9, workers 2, `val: true`, 라벨 v1(`data/aihub_yolo_v1`).
- 평가: crop 단위 `last.pt`, AP는 conf 0.001, 혼동 행렬은 conf 0.25. 띠 보조 지표는 M1-10 §6-1 정의다(§6-2).
- run 이름: `yolo11m_aihubv1n2a3_<D>_fold<k>`. 내보내기 폴더는 `runs/analysis/n2a3_val_export/fold<k>_crop`이다.

## 3. K32 구현 명세 (python-expert)

### 3-1. 규칙

1. **적용 범위**: `augment_context_scale`에서 **맥락 모드**(`is_context` true, 출처 a·c1 모두)일 때만 적용한다.
   - 기본 모드에는 적용하지 않는다.
   - `augment_shift_fill`(채움 이동·채움 띠)과 Ultralytics 단계에도 적용하지 않는다.
2. **"잘림"의 정의**: 박스의 출력(256 창) 보이는 폭 $w_\text{vis}$가 그 박스의 **원 라벨 폭**을 출력 배율로 옮긴 값 $w_\text{full}$보다 1 px 넘게 작으면 잘린 것이다. 식은 $w_\text{vis} < w_\text{full} - 1.0$이다.
   - 원 라벨 폭: 그 박스가 나온 crop(중심 또는 이웃)의 라벨 txt에 적힌 폭(crop px)이다. 열 소유 구간 자르기와 띠 끝($b + E$) 자르기 **전** 값이다.
   - 출력 배율: `factor_x` = `round(W_s · s) / W_s`다($W_s$는 띠 폭). `place_window`의 박스 변환과 같은 식이다.
   - 따라서 열 소유 구간 자르기와 창 자르기 둘 다 "잘림"으로 센다.
3. **버림**: `keep_mask`가 참이고, 클래스가 `pipe_band`(id 1)이고, 잘렸고, $w_\text{vis} < 32.0$이면 `keep_mask`를 거짓으로 바꾼다. $w_\text{vis} = 32.0$은 남긴다.
4. **순서**: `place_window` 안의 기존 규칙(보이는 넓이 ≥ 50 %, 잘린 변 ≥ 4 px)을 먼저 적용하고, 그 뒤 K32를 적용한다. K32는 버림만 늘리고 남김을 늘리지 않는다.
5. **난수**: K32는 난수를 뽑지 않는다. 같은 rng 상태에서 영상·`s`·`dx`·다음 영상의 rng 상태가 N2-a′와 같다.
6. **끄기**: `band_min_clipped_width_px`가 없거나 0이면 K32를 적용하지 않는다. 동작은 N2-a′와 바이트 단위로 같다(하위 호환).

### 3-2. 코드 변경 위치 (`src/gpr_detection/augment_geometry.py`, `src/gpr_detection/detect/yolo_train.py`)

| 위치 | 변경 |
|---|---|
| `ContextScaleConfig` | 필드 `band_min_clipped_width_px: float = 0.0`. `__post_init__`에서 유한 실수이고 ≥ 0인지 검사한다. 아니면 `ValueError` |
| `context_config_from_dict` | 키 `band_min_clipped_width_px`를 읽는다. 없으면 0.0 |
| `merge_strip_labels` | 박스별 **원 라벨 폭**(crop px, 열 자르기 전)과 **출처**(0 = 중심, 1 = 이웃)를 함께 돌려줄 수 있게 한다. 기존 반환형을 쓰는 호출과 시험은 그대로 두고, 키워드 인자(예: `with_meta=True`)나 새 함수로 더한다. 출처 규칙: 비-band 박스가 규칙 (가)로 다른 박스를 대체하면 대체한 박스의 폭·출처를 쓴다 |
| `ContextStrip` | 선택 필드 `source_widths_px: np.ndarray \| None = None`, `origins: np.ndarray \| None = None`(길이 M). `assemble_context_strip`이 채운다. `crop_only_strip`은 박스 폭 그대로, 출처 0으로 채운다 |
| 새 함수 `drop_short_clipped_bands(boxes_out, classes, keep_mask, full_widths_out, min_width_px, band_class_id=PIPE_BAND_CLASS_ID, tolerance_px=1.0) -> (keep_mask_new, dropped_index)` | §3-1 2–3을 계산한다. 입력 배열을 바꾸지 않는다(새 배열 반환). `min_width_px` ≤ 0이면 입력 마스크 사본과 빈 인덱스를 돌려준다 |
| `augment_context_scale` | `place_window` 뒤, 맥락 모드이고 `config.band_min_clipped_width_px` > 0이면 `full_widths_out = strip.source_widths_px * factor_x`로 `drop_short_clipped_bands`를 부른다. 결과를 `ContextRecord`의 새 필드 `band_k32_dropped: tuple[tuple[float, int], ...] = ()`((보이는 폭, 출처) 목록)에 담는다 |
| `YoloContextScale`(yolo_train.py) | `aug_stats`의 맥락 단계 기록에 `band_k32` 블록을 더한다. 항목: `dropped`(건수), `by_origin`({centre, neighbour}), `width_bins`({0–8, 8–16, 16–24, 24–32}). `stages.context_scale`의 남김·버림 카운터는 K32 버림을 포함한 최종 `keep_mask`로 센다. 학습 기록 `gpr_context` 블록(sha 포함)에 새 키가 들어가는지 확인한다 |

- `place_window`의 시그니처와 동작은 바꾸지 않는다. 기본 모드 경로와 `scale_isotropic` 동일성(C-14)도 그대로 둔다.
- `src/gpr_detection/detect/` 밖에서 ultralytics를 import하지 않는다(AGPL 경계).

### 3-3. 시험 (`tests/test_context_augment.py`에 추가, CPU)

기존 C-1–C-25·R-1과 저장소 전체 시험이 그대로 통과해야 한다. 합성 영상·인덱스는 기존 C-1의 세로 줄무늬 설정(중심 (1000, 1256), 이웃 (1104, 1360))을 쓴다.

| ID | 내용 | 기대 |
|---|---|---|
| K-1 | **영상·라벨 동일성.** 같은 합성 인덱스와 라벨(band·쌍곡선 섞음)에서 rng = `default_rng([7, i])`로 i = 0..999를 두 설정으로 각각 처리한다. 두 설정은 `band_min_clipped_width_px` 0과 32다. 모드는 설정 확률 0.5 그대로 둔다 | 1,000장 모두 영상 sha256 동일. 비-band 박스·클래스 완전 동일. 32 설정의 남은 band 박스 집합 ⊆ 0 설정의 남은 band 박스 집합(행 단위 완전 일치). 전체에서 K32 버림이 1건 이상(시험이 실제로 규칙을 건드리는지 확인) |
| K-2 | **난수 소비.** K-1의 각 영상 처리 뒤 `rng.bit_generator.state`를 비교한다 | 두 설정에서 모든 i에 대해 같다 |
| K-3 | **경계값.** `drop_short_clipped_bands`에 band 박스 2개를 넣는다. 보이는 폭 31.9와 32.0이고, 원 폭(출력 px)은 둘 다 100이다 | 31.9는 버림, 32.0은 남김 |
| K-4 | **안 잘린 짧은 band.** 보이는 폭 20.0 / 원 폭 20.0인 경우와, 보이는 폭 19.2 / 원 폭 20.0인 경우(1 px 허용 오차 안) | 둘 다 남김 |
| K-5 | **기본 모드 미적용.** 기본 모드를 강제(`p_context` 0)하고, band가 crop 양 끝에 닿는 라벨로 s를 U[0.55, 1.5]에서 1,000회 뽑는다. 32 설정과 0 설정을 비교한다 | 영상·라벨·`keep_mask`가 모두 같다. `band_k32_dropped`는 비어 있다 |
| K-6 | **끄기.** 설정에 키가 없을 때와 0일 때 | `context_config_from_dict` 결과가 0.0이다. K-1과 같은 1,000장에서 N2-a′ 경로와 출력이 같다. 기존 C-1–C-25가 통과한다 |
| K-7 | (§7-0 3번 참조) **다른 클래스 미적용.** 맥락 모드, 쌍곡선 박스가 창에 잘려 보이는 폭 20 px(원 폭 60 px, 보이는 넓이 ≥ 50 %가 되도록 높이를 둠) | 쌍곡선은 남김. K32는 band만 대상이다 |
| K-8 | (§7-0 4번 참조) **열 소유 자르기도 잘림.** C-13과 같은 배치에서 이웃 band (1150, 72, 1270, 88)를 [1256, 1270)으로 자르면 14 px이다. 맥락 모드, s = 1.0, dx = 0으로 띠 안 위치가 창 안에 온전히 들어오게 둔다 | 0 설정은 남김(보이는 넓이 = 자른 박스 전체), 32 설정은 버림(보이는 14 < 원 120 − 1). `band_k32_dropped` = ((14.0, 1),) |
| K-9 | **설정 검사.** −1, NaN, 문자열 | `ValueError` |
| K-10 | **기록·피클.** `YoloContextScale`을 피클했다가 되살린 뒤 같은 seed로 실행한다 | 결과 동일. `aug_stats`에 `band_k32` 블록이 있고 `dropped`가 `by_origin` 합과 같다 |

## 4. 채택 조건 (사전 등록, U3-b·U3-c·U3-d 결정)

### 4-1. 지표와 문턱

모든 지표는 crop 단위, `last.pt`, 5 fold 비가중 평균이다. Fraction으로 정확히 비교하고 반올림하지 않는다.

| ID | 지표 | 문턱 | 비교 대상 |
|---|---|---|---|
| A1 | 전체 mAP50 / pipe_hyperbola AP50 | ≥ 76.8 / ≥ 90.9 | U-3 절대값 |
| B | 띠 보조 가로 구간 재현율 | 평균 ≥ 0.641 **그리고** $\#\{k : r^{(k)}_\text{N2-a″} > r^{(k)}_\text{N2-a′}\}$ ≥ 4 (같으면 개선이 아님) | U-3 절대값 + N2-a′ fold 값 |
| B-p | 띠 보조 가로 구간 정밀도 | 평균 ≥ 0.497 | U-3 절대값 |
| A2′ | D1 r5 | `rule_P` = 6 **그리고** `rule_O` ∈ {75 %, 50 %} | [T] v0.5.2 §8-2 |

- **채택** = A1 그리고 B 그리고 B-p 그리고 A2′다.
  - 채택되면 기준선을 N2-a′ → N2-a″로 바꾼다(사용자 승인 뒤).
  - `rule_O` = 50 %이면 [SP] 기본값(`overlap_px` 192) 변경은 따로 사용자 결정으로 한다(U3-d).
  - 불채택이면 N2-a′가 기준선으로 남고, 다음 단계는 결과를 본 뒤 정한다(U3-e).
- **N2-a′ 기준값 출처(고정)**: `analysis/detection/n2a2_band_aux.json`, sha256 **bf42403e1eedfa95fc556f1710c461d2cdb146cbe164072003a14742860a6145**(2026-11-14 `Get-FileHash`, 커밋됨).
  - 쓰는 값: `units.crop.statistics.recall.folds` = [0.646922995795787, 0.5401378301936709, 0.5778940392773931, 0.5797035944530312, 0.7197308631979048].
- 근거
  - B 평균 문턱은 U-3 띠 목표다.
  - 4/5 fold 조건을 두는 이유: fold 간 SD가 3.7–7.5 %p라 평균 하나로는 흔들림을 거르지 못한다. 효과가 0일 때 우연히 통과할 확률은 6/32 ≈ 19 %다.
  - B-p는 짧은 조각을 버린 뒤 정밀도가 무너지지 않았는지 막는 하한(U-3 띠 정밀도 목표)이다.
  - A2′에 규칙 P = 6을 넣는 이유: [SP] v0.4 기본값이 이미 $P^*$ 6이다.

### 4-2. 보고 항목 (판정 조건 아님)

- 클래스별 AP50, mAP50-95, `best.pt` 값, conf 0.25 혼동 행렬(쌍곡선 오검출 수 포함)
- 띠 AP50, 조각 수
- fold별 짝지은 차: N2-a″ − N2-a′, N2-a″ − M1-10
- `aug_stats.band_k32`
- R1(겹친 crop 실제 위치 차, N2-a″만 새로 계산, 다른 모델은 r4 결과 재사용)
- 쌍곡선 AP50 여유

## 5. 사전 확인 (학습 전, CPU, 블라인드 안전)

### 5-1. 스크립트 `analysis/detection/n2a3_band_label_diag.py` (신규, python-expert)

- 바탕은 `n2a2_band_drop_diag.py`(sha256 98f19f0869d17bd72ed9d782e73b7e4c2b7d21239949f6a155a6fdbc62731a50)다. 표본 추출(fold 0 학습 영상에서 seed로 2,000장)과 단계 재현 방식을 재사용한다.
- 증강 함수(`augment_context_scale`, `augment_shift_fill`)를 직접 부르고, 학습·GPU·모델 가중치는 쓰지 않는다.
- CLI: `--baseline-config configs/train_yolo11m_n2a2.yaml --config configs/train_yolo11m_n2a3.yaml --validation-fold 0 --images 2000 --seeds 0,1 --out analysis/detection/n2a3_band_label_diag.json [--overwrite]`
- 출력 규약
  - 덮어쓰기는 `--overwrite` 없이는 거부한다(종료 2).
  - stdout은 상태 줄만 쓰고, 마지막 줄은 요약 JSON 한 줄이다.
  - 종료 코드: 0 = S 전부 통과, 1 = S 위반 있음, 2 = 입력 오류, 3 = 내부 오류.
  - `run_identity`: git head·dirty, 두 설정 sha256, 인덱스 sha256, 이 스크립트 sha256, argv, 패키지 버전.
- 실행 방식(seed마다, 두 설정을 같은 영상·같은 rng로)
  - **R-ctx**: 맥락 모드 강제(`p_context` = 1). 영상 i의 rng는 `default_rng([seed, i])`다. `augment_context_scale`만 적용한다.
  - **R-nat**: 설정 확률 그대로. 같은 rng로 `augment_context_scale`을 적용한 뒤 `augment_shift_fill`을 적용한다.
- 재는 값: **남은 band 라벨의 출력 보이는 폭**(창 자르기 뒤, 출력 px)이다. 구간은 0–8 / 8–16 / 16–24 / 24–32 / 32–64 / 64–128 / 128–192 / 192–256이다.
  - 잘림 여부, 출처(중심 / 이웃)별로도 나눈다.
  - 원 crop 분포(fold 0 학습 집합의 v1 band 라벨 폭)를 같은 구간으로 다시 계산해 기록한다. §1-2 값과 같아야 한다.
  - KS 거리(남은 band 폭 ↔ 원 crop 폭)와 중심 band 버림률도 기록한다.

### 5-2. 판정 (U3-f: 권고값, S-0은 v0.2 신설)

| ID | 대상 | 조건(두 seed 모두) | 어기면 |
|---|---|---|---|
| **S-0** | N2-a′ 설정, R-ctx | 남은 band 라벨 중 보이는 폭 < 32 px 비율 ≥ 5 % | 학습하지 않는다. 가설 (1)의 전제(학습 라벨에 짧은 조각이 의미 있게 있음)가 성립하지 않으므로 메인 세션에 보고하고 사용자 결정을 받는다 |
| S-1 | N2-a″ 설정, R-ctx | 남은 band 라벨 중 보이는 폭 < 32 px 비율 ≤ 2 % | 학습하지 않는다. 규칙을 고친 판을 다시 사전 등록한다 |
| S-2 | 두 설정, R-nat | 2,000장 모두 영상 sha256 동일, 비-band 라벨 동일, N2-a″ band ⊆ N2-a′ band | 학습하지 않는다(구현 결함, python-expert) |
| S-3 | N2-a″ 설정, R-ctx | 중심 crop에 band가 있던 영상 중 남은 band가 0개인 영상 비율 ≤ 1 % | 학습하지 않는다. 규칙 재검토 |
| S-4 | 두 설정, R-ctx | (K32로 버린 band의 보이는 폭 합) ÷ (N2-a′ 설정에서 남은 band의 보이는 폭 합) ≤ 5 % | 학습하지 않는다. 규칙 재검토 |

- S-0의 5 %는 v0.2에서 정한 값이다. 근거는 다음과 같다. 맥락 모드 비율이 0.5이므로 학습 전체에서 바뀌는 band 라벨은 S-0 값의 절반이다. 5 %면 학습 band 라벨의 약 2.5 %가 바뀐다. 이보다 작으면 띠 재현율 8 %p 하락을 설명하기 어렵다고 본다.
- 이 진단은 학습·평가 결과를 보지 않는다. 결과에 따라 규칙을 고쳐도 블라인드가 깨지지 않는다. 다만 고친 규칙은 새 판으로 사전 등록한다.
- 진단 결과 json은 봉인 C1 HEAD에서 다시 만들어 C2로 커밋한다(§7-2, 확인 필요 4 결정).

### 5-3. 결과 (2026-11-14, dirty 트리 실행, 봉인 전 참고값)

- 실행 정보: HEAD a24ab98, dirty true(구현 미커밋). 스크립트 sha256 60fd9a6a…d489, N2-a″ 설정 sha256 14327dab…0dc1. seed 0 / seed 1 순서로 적는다.
- **보관 위치**: 이 dirty 실행의 json은 작업 트리 밖 scratchpad `n2a3_dirty_diag\n2a3_band_label_diag.json`(메인 세션 scratchpad)으로 옮겼다. 커밋하지 않는 참고값이다.
- 그 뒤 진단 스크립트 sha256이 60fd9a6a…d489 → 32e35c5c…로 바뀌었다. `counts_raw` 정의 필드를 더한 것이고 S 판정 로직은 그대로다. 봉인 C1 HEAD에서 다시 돌린 S 값은 아래 표의 dirty 값과 대조한다(§7-2 3).
- **C1 HEAD 재실행 확정 (2026-11-14, 판정자 cv)**: `analysis/detection/n2a3_band_label_diag.json`(C2에 커밋)이 확정값이다. 확인 내용은 다음과 같다.
  - 실행 정체성: head 2e6526dc25662769a0310dd8f2402054779abff3, dirty false, 종료 0. 스크립트 sha256 32e35c5cb15a1dc7b91e1bd4e950506f2d5defb7293ff5064638d2d7d65a0bcb, N2-a″ 설정 14327dabfa8288234f0576fc8e2a0628c7172f03068ed7ac223259ca51a30dc1, N2-a′ 설정 b9416b57…42ef, 인덱스 sha는 §2 고정값이다. 앞의 두 sha는 C1 작업 트리 파일의 재해시와 같다.
  - `s_checks` 전체(정확 분수 포함)와 `seeds` 세부값이 dirty 실행과 **완전히 같다**. 예: S-0 56/573 · 11/115, S-3 0/1 · 1/396. 같은 seed·같은 2,000장·같은 판정 로직이므로 기대대로다.
  - dirty 실행과 다른 최상위 필드는 `plan`·`run_identity`·`original_crop_band_widths`뿐이다. 마지막은 정의 필드를 더한 차이다(`definitions`, `width_64_counted_below_64_in_raw`).
  - 정의 필드 값: `counts_plan_convention` 155 / 687(`matches_plan_1_2` true), `counts_raw` 164 / 271, `width_64_counted_below_64_in_raw` 9. 모두 §1-2와 같다.
  - **S-0–S-4 확정 통과.**

| ID | 값 (seed 0 / 1) | 문턱 | 결과 |
|---|---|---|---|
| S-0 | 9.77 % / 9.57 % | ≥ 5 % | 통과 |
| S-1 | 0 % / 0 % | ≤ 2 % | 통과 |
| S-2 | 차이 0 / 0 | 동일 | 통과 |
| S-3 | 0 % / 0.25 % | ≤ 1 % | 통과 |
| S-4 | 1.37 % / 1.56 % | ≤ 5 % | 통과 |

- 짧은 조각과 K32 버림(56건 / 55건)은 **모두 이웃 출처**다. 중심 band 버림률은 두 설정이 같다. 그래서 K32는 사실상 "맥락 열의 짧은 이웃 band 조각 버림"이다.
- 원 crop 분포와의 KS 거리는 N2-a″ 0.156, N2-a′ 0.148로 N2-a″가 조금 크다. v0.2 §5-1의 "원 분포에 가까워짐" 기대는 KS 기준으로는 나오지 않았다.
  - 해석: 판정 조건이 아니다. KS는 분포 전체의 최대 차이라, 맥락 모드 확대(s ≥ 1)로 긴 쪽이 늘어난 차이가 지배한다. 32 px 미만 꼬리를 지우면 상대 분포가 긴 쪽으로 더 쏠려 KS가 조금 커질 수 있다. 짧은 꼬리는 S-1로 없어졌음을 확인했다.
- **효과 크기에 대한 해석(위험, §8 7번)**
  - K32가 바꾸는 것은 맥락 모드에서 남은 band 라벨의 약 10 %(S-0)다.
  - 맥락 모드는 영상의 절반이다. 그래서 학습 전체 band 라벨로는 약 5 % 이하, 보이는 길이로는 약 0.7–0.8 %(S-4의 절반)만 바뀐다.
  - 띠 재현율 8 %p 하락을 이 정도 변화로 되돌리기에는 작을 수 있다. B가 실패해도 가설 (1)이 틀렸다고 단정하지 않고, "K32 크기의 변화로는 회복하지 않음"으로 해석한다.

## 6. 판정 경로 (A1·B·B-p·A2′)

### 6-1. A1: `analysis/detection/n2a_a1_judge.py`를 바꾸지 않고 재사용한다

- sha256 7377d00dffad96746b1eb49a97e220ef31a72d5e88e99ea006661af53f0f3676(a24ab98 기준).
- 이 스크립트는 다음 게이트를 이미 확인한다.
  - 설정 봉인(봉인 커밋의 설정 blob sha = 학습 기록), `gpr_context` 블록 sha = 봉인 설정의 블록. 새 키 `band_min_clipped_width_px: 32`가 블록에 들어가므로 별도 단언 없이 보증된다.
  - `args.yaml` 단언(scale 0·translate 0.1·flipud 0·epochs 50·seed 0)
  - 인덱스 sha의 fold 간 동일성, 평가 목록 = M1-10
- python-expert 확인 결과(v1.0): 학습 기록 dump에 새 키가 기록된다(§7-0 6). `n2a_a1_judge.py`는 바꾸지 않는다. `RULES` 문자열은 N2-a 계획을 가리키는 상수이고 판정 계산에는 쓰이지 않는다(N2-a′ 판정서 §6 주석과 같은 성격).
- 판정자 손 확인: 인덱스 sha = §2 고정값, 학습 기록 `gpr_context.band_min_clipped_width_px` = 32.0.

### 6-2. 띠 보조 지표: 새 래퍼 `analysis/detection/n2a3_band_aux.py`

- `n2a2_band_aux.py`와 같은 방식이다. `m110_band_aux.py`(sha256 690f97459f80dfc278c1266ed042335998e59bdb474494fdd1d8ce51a73fee64, 수정 없음)를 불러 두 상수를 바꿔 끼운다.
  - `RUN_NAME` = `yolo11m_aihubv1n2a3_<D>_fold{fold}`. `<D>`는 `--date` 인자로 받는다.
  - `OUT_STEM` = `analysis/detection/n2a3_band_aux`
- 예측 폴더는 `runs/analysis/n2a3_val_export`, crop 단위만 쓴다. CPU만 쓰고 덮어쓰기는 거부한다.
- `m110_band_aux`의 내부 검산이 결속 장치다. 예측으로 그 run의 `eval/last_crop` 혼동 행렬(conf 0.25, 25칸)과 `metrics.json` band AP50을 재현해야 한다. 그래서 띠 지표는 A1이 가중치 sha를 확인한 같은 평가에 묶인다.

### 6-3. B·B-p 판정: 새 스크립트 `analysis/detection/n2a3_band_judge.py`

- 띠 판정을 A1 스크립트에 넣지 않는 이유: N2-a′ A1 판정에 쓴 봉인된 스크립트를 그대로 재사용해 A1 경로를 바꾸지 않기 위해서다.
- CLI: `--band-aux analysis/detection/n2a3_band_aux.json --reference analysis/detection/n2a2_band_aux.json --reference-sha256 bf42403e1eedfa95fc556f1710c461d2cdb146cbe164072003a14742860a6145 --a1-verdict analysis/detection/n2a3_a1_verdict.json --run-prefix yolo11m_aihubv1n2a3_<D>_fold --out analysis/detection/n2a3_band_verdict.json [--overwrite]`
- 게이트(하나라도 어기면 종료 1, 판정 값 없음)
  - G-b1: `--reference` 파일 sha256 = `--reference-sha256`.
  - G-b2: 두 띠 지표 파일의 `units.crop.fold_ids` = [0, 1, 2, 3, 4].
  - G-b3: `n2a3_band_aux.json`의 5 fold 모두 `checks.confusion_matches` true이고 `checks.ap50_matches` true.
  - G-b4: `n2a3_a1_verdict.json`의 `gates_pass` true, run-prefix 일치. A1 판정 값은 이 스크립트가 다시 계산하지 않고 그대로 옮긴다.
  - 결속의 한계(확인 필요 2 결정): `n2a3_band_aux.json`에는 run 이름이 없다. 그래서 띠 지표 ↔ run 결속은 G-b3(그 run의 평가 혼동 행렬·AP 재현)뿐이고, 래퍼의 `--date`가 틀리면 판정 스크립트가 알아채지 못한다. **코드는 바꾸지 않고 판정서 손 확인 항목으로 둔다.**
    - (h1) 러너 로그의 `n2a3_band_aux.py` argv `--date <D>`가 run-prefix의 `<D>`와 같다.
    - (h2) `n2a3_band_aux.json`의 fold별 `n_images`·`n_gt_band_boxes`가 같은 run의 `eval/last_crop/metrics.json` 평가 목록·band 인스턴스 수와 같다.
    - (h3) 래퍼 파일 sha256이 봉인 커밋의 blob과 같다.
- 계산
  - B: 재현율 fold 값 5개의 평균(Fraction) ≥ 641/1000, 그리고 기준 fold 값보다 큰 fold 수(엄격한 >) ≥ 4.
  - B-p: 정밀도 평균 ≥ 497/1000.
- 출력: `verdict`, `failed`, `b`·`b_p`의 값·문턱·통과 여부, fold별 짝지은 차, `a1`(옮긴 값), `adoption_pending_a2` = A1 and B and B-p, 그리고 `run_identity`(git head·dirty, 입력 파일 sha256 3개, argv).
- **verdict 규칙(확인 필요 1 결정, 구현 완료 — python-expert, 봉인 C1 전, 네 조합 시험 포함)**: 처음 구현은 `verdict`가 "B met" / "B not met"으로 B만 반영했다. 사전 등록 판정 단위는 "B 그리고 B-p"이므로 다음처럼 바꿨다.
  - `verdict` ∈ {"B and B-p met", "B or B-p not met"}. `b.pass and b_p.pass`일 때만 앞쪽이다.
  - `failed`: 통과하지 못한 항목 목록(순서 고정 `["B", "B-p"]` 중 해당하는 것). 둘 다 통과하면 빈 목록이다.
  - `b.pass`·`b_p.pass`·`adoption_pending_a2`는 그대로 둔다.
  - 시험: (B, B-p) 네 조합 각각의 `verdict`·`failed`. 기존 경계 시험은 그대로 통과해야 한다.
  - 종료 코드는 바꾸지 않는다(판정했으면 충족 여부와 무관하게 0).
- 종료 코드: 0 = 판정함(충족·미충족 모두), 1 = 게이트 실패, 2 = 입력 오류, 3 = 내부 오류. stdout 마지막 줄은 요약 JSON이다.
- 시험(`tests/` CPU, 합성 json)
  - 문턱 경계: 평균이 정확히 0.641이면 통과한다. 4/5 fold 중 하나가 기준과 같으면(>가 아님) 3/5로 실패한다.
  - G-b1–G-b4 각각 실패 경로를 시험한다.

### 6-4. A2′: D1 r5 (부록 자리, A1·B 충족 뒤 v1.1에서 채움)

- 규칙은 [T] v0.5.2 §8-2(봉인 3c81aca, 규칙 blob eacf9126376c199d9486bb8a98b13fa2b411d7bd) 그대로다. 스크립트도 `m31_d1_shift_scale.py`·`m31_d1_judge.py` 그대로다.
- A2′ 값은 `m31_d1_r5_verdict.json`의 규칙 P·규칙 O 결과에서 판정자가 손으로 옮긴다. N2-a′ 때와 같은 방식이다.

| 항목 | 값 (v1.1에서 채움) |
|---|---|
| N2-a″ 학습 봉인 | `<SEAL>`(설정 sha256 `<CFG_SHA>`) |
| run-prefix / export-root | `yolo11m_aihubv1n2a3_<D>_fold` / `runs/analysis/n2a3_val_export` |
| 기대 sha8(fold 0–4) | `<SHA8 × 5>` |
| 가중치 sha256 5개, G1 입력 sha256 15개 | `<64자 × 20>` |
| G3 기대 | r4와 같은 n_gt·결손 기대값([P2] §12) |
| 출력 | `runs/detect/yolo11m_aihubv1n2a3_<D>_fold<k>/eval/m31_d1_r5/`, `analysis/detection/m31_d1_r5_summary.json`, `analysis/detection/m31_d1_r5_verdict.json`, R1 `analysis/detection/m31_r1_n2a3.json` |

## 7. 러너·감사 순서

### 7-0. 구현 결과 반영 (v1.0, python-expert 구현, 처음 1,406 passed → verdict 수정 뒤 1,410 passed, N2-a·N2-a′ 설정에서 동작 비트 동일)

봉인 판에서는 이 표가 §3·§5·§6 명세보다 우선한다.

| # | 구현(보고) | 판단 |
|---|---|---|
| 1 | `merge_strip_labels`를 고치지 않고 새 함수 `merge_strip_labels_with_meta`로 원 폭·출처를 돌려줌 | 승인. 명세가 허용한 방식(새 함수)이고 기존 호출·시험이 그대로다 |
| 2 | 원 라벨 폭은 `_clip_to_image`(crop 경계 자르기) **뒤** 값 | 승인. crop 밖으로 나간 라벨 좌표(부동소수 오차 수준)를 crop 안 폭으로 본다. "이 단계가 자른 것"과 "원래 crop 경계"를 구분하는 명세 뜻과 같다 |
| 3 | K-7을 두 시험으로 나눔: 명세 값(보이는 20 / 원 60)은 함수 단위 시험, 단계 전체는 쌍곡선 36 → 20 px 경우 | 승인. 단계 전체에서 50 % 규칙을 통과하는 배치를 만들려면 값이 달라야 한다. 확인하는 뜻(band만 대상)은 같다 |
| 4 | K-8의 창 이동을 dx −40으로 둠 | 승인. 이웃 조각 [1256, 1270)이 창 안에 온전히 들어오게 하는 값이다. 기대(0 설정 남김, 32 설정 버림)는 같다 |
| 5 | `aug_stats`의 `band_k32`를 맥락 단계 아래가 아니라 최상위 키로 둠 | 승인. 판정 조건이 아니고 보고용이다. 판정서와 스모크 확인은 최상위 `band_k32`를 읽는다 |
| 6 | 학습 기록 dump의 `gpr_context`에 `band_min_clipped_width_px: 0.0`이 들어감(키가 없는 N2-a·N2-a′ 설정에서도). 블록 sha는 불변 | 승인. sha는 설정 파일 블록에서 계산하므로 N2-a′ 봉인 비교에 영향이 없다. N2-a″는 32.0이 기록되어야 한다(§6-1 손 확인) |
| 7 | `n2a3_band_aux.py`가 stdout을 블라인드 처리하고, 종료 코드 0/1/2/3을 씀 | 승인. §7-1 블라인드 규약과 같다 |
| 8 | `n2a3_band_judge.py`에 종료 코드 3(내부 오류)을 추가 | 승인. §6-3에 반영했다 |

확인 필요 5건 결정

| # | 질문 | 결정 |
|---|---|---|
| 1 | `n2a3_band_judge.py` verdict "B met"이 B만 반영 | **바꾼다.** verdict = "B and B-p met" / "B or B-p not met", `failed` 목록 추가. 명세는 §6-3. 구현 완료(네 조합 시험 포함) |
| 2 | G-b4는 A1 json의 run-prefix만 보고, 띠 지표 json에는 run 이름이 없음 | **코드는 그대로 두고 판정서 손 확인 (h1)–(h3)으로 둔다**(§6-3). G-b3이 그 run의 평가를 재현해야 통과하므로, 남는 위험은 `--date`를 잘못 넣는 운영 실수뿐이다. 러너 로그 argv로 확인할 수 있다 |
| 3 | §1-2 값의 정의 | v0.1 값은 반올림 없는 반열림 구간(155)과 가장자리 2 px 허용치(687)로 잰 것이다. 이번에 같은 스크립트로 다시 확인했다. 정의는 §1-2에 적었다. `counts_raw` 164의 정의는 json에 기록했다(구현 완료, 보고 값) |
| 4 | 진단 json이 dirty 트리에서 만들어짐 | **봉인 C1 HEAD에서 다시 돌려 확정하고 C2로 커밋한다**(§7-2 3–4). S 판정은 C1 실행값으로 확정한다 |
| 5 | §7-1 진단 명령에 `-B` | 넣었다 |

### 7-1. 명령 (메인 세션 러너, PowerShell, 작업 폴더 `C:\workspace\gpr_detection`)

```powershell
$py = "C:\ProgramData\miniforge3\Scripts\conda.exe"
# [seal C1 HEAD] pre-check rerun, CPU (step 3 below). The earlier dirty-tree json is moved out of the
# working tree before C1, so this run starts from a clean tree and needs no --overwrite.
& $py run -n gpr-dev --no-capture-output python -I -B analysis/detection/n2a3_band_label_diag.py --baseline-config configs/train_yolo11m_n2a2.yaml --config configs/train_yolo11m_n2a3.yaml --validation-fold 0 --images 2000 --seeds 0,1 --out analysis/detection/n2a3_band_label_diag.json
# [runner A] smoke, then per fold k = 0..4 (each step preceded by a HEAD check)
& $py run -n gpr-dev --no-capture-output python -I scripts/train_detector.py --config configs/train_yolo11m_n2a3.yaml --fold 0 --epochs 2 --device 0 --name yolo11m_aihubv1n2a3_smoke_fold0 --blind
& $py run -n gpr-dev --no-capture-output python -I scripts/train_detector.py --config configs/train_yolo11m_n2a3.yaml --fold <k> --device 0 --name yolo11m_aihubv1n2a3_<D>_fold<k> --blind
& $py run -n gpr-dev --no-capture-output python -I scripts/eval_detector.py --run-dir runs/detect/yolo11m_aihubv1n2a3_<D>_fold<k> --confusion-conf 0.25 --device 0 --blind
& $py run -n gpr-dev --no-capture-output python -I analysis/detection/export_val_predictions.py --device 0 --weights runs/detect/yolo11m_aihubv1n2a3_<D>_fold<k>/weights/last.pt --data-yaml runs/detect/yolo11m_aihubv1n2a3_<D>_fold<k>/eval/last_crop/eval_data.yaml --out-dir runs/analysis/n2a3_val_export/fold<k>_crop --blind
# [runner A] judgement inputs (CPU)
& $py run -n gpr-dev --no-capture-output python -I analysis/detection/n2a_a1_judge.py --config configs/train_yolo11m_n2a3.yaml --run-prefix yolo11m_aihubv1n2a3_<D>_fold --seal-commit <SEAL> --out analysis/detection/n2a3_a1_verdict.json --blind
& $py run -n gpr-dev --no-capture-output python -I analysis/detection/n2a3_band_aux.py --date <D>
& $py run -n gpr-dev --no-capture-output python -I analysis/detection/n2a3_band_judge.py --band-aux analysis/detection/n2a3_band_aux.json --reference analysis/detection/n2a2_band_aux.json --reference-sha256 bf42403e1eedfa95fc556f1710c461d2cdb146cbe164072003a14742860a6145 --a1-verdict analysis/detection/n2a3_a1_verdict.json --run-prefix yolo11m_aihubv1n2a3_<D>_fold --out analysis/detection/n2a3_band_verdict.json
# [runner B, after appendix seal + audit 3] D1 r5 and R1
& $py run -n gpr-dev --no-capture-output python -I analysis/detection/m31_d1_shift_scale.py --run-prefix yolo11m_aihubv1n2a3_<D>_fold --export-root runs/analysis/n2a3_val_export --folds 0,1,2,3,4 --device 0 --fill 128 --save-examples --out-subdir m31_d1_r5 --summary analysis/detection/m31_d1_r5_summary.json
& $py run -n gpr-dev --no-capture-output python -I analysis/detection/m31_d1_judge.py --run-prefix yolo11m_aihubv1n2a3_<D>_fold --export-root runs/analysis/n2a3_val_export --expected-weights-sha8 <SHA8> --subdir m31_d1_r5 --out analysis/detection/m31_d1_r5_verdict.json
& $py run -n gpr-dev --no-capture-output python -I analysis/detection/m31_r1_overlap_shift.py --run-prefix yolo11m_aihubv1n2a3_<D>_fold --folds 0,1,2,3,4 --device 0 --out analysis/detection/m31_r1_n2a3.json --blind
```

- `n2a3_band_aux.py`·`n2a3_band_judge.py`는 블라인드 출력 규약을 따른다. stdout에 지표를 쓰지 않고 상태와 경로만 쓴다. 판정 json은 판정자만 연다.

### 7-2. 순서

1. 구현 완료(2026-11-14, §7-0). §6-3 verdict 수정과 `counts_raw` 정의 기록까지 구현했다. 수정 뒤 실행 결과는 다음과 같다.
   - 전체 시험: 1,410 passed, 14 skipped, 26 deselected, 2 xfailed. 판정 시험 네 조합이 들어 있다.
   - fold 0–4 `--dry-run`: 종료 0, 누수 단언 0건, K32 = 32.
2. ~~dirty 트리 사전 확인~~ 완료(§5-3, 참고값). integrity-checker 봉인 전 점검을 받는다.
3. **봉인 C1**: 이 문서 v1.0, 설정, 구현, 시험을 한 커밋으로 만든다. dirty 트리에서 만든 진단 json은 C1에 넣지 않고 작업 트리 밖(scratch)으로 옮긴다.
   - 그다음 **C1 HEAD, 깨끗한 트리에서** §7-1 진단 명령을 다시 돌린다. 판정자(cv)가 S-0–S-4를 확정한다. **위반이 있으면 멈춘다.**
   - 확인할 것: 새 json의 `run_identity.git.head` = C1, `dirty` false, `script_sha256`·`config_sha256`이 C1 blob과 같다. S 값은 dirty 실행(§5-3)과 같아야 한다(같은 seed·같은 코드). 다르면 멈추고 원인을 본다.
4. **봉인 C2**: 진단 json 하나만 더해 커밋한다. `<SEAL>` = C2다.
   - 확인: `git diff --name-only C1 C2` = `analysis/detection/n2a3_band_label_diag.json`과 이 문서(§5-3 확정 기록·변경 이력만) 두 개뿐. 설정·코드·시험은 바뀌지 않는다. 설정 blob이 C1과 같으므로 `n2a_a1_judge.py --seal-commit C2`의 설정 봉인 비교는 C1 내용과 같다.
   - 감사 ①(실행 전)을 C2에서 받는다. **차단이면 멈춘다.**
5. 러너 A: 스모크(판정자는 `aug_stats.band_k32`가 0이 아닌지만 본다) → 5 fold 학습·평가·내보내기 → A1 판정 → 띠 지표 → B 판정.
6. 판정자가 A1·B 판정서(`m31_n2a3_a1b_judgment.md`)를 쓴다. **A1·B·B-p 중 하나라도 미충족이거나 게이트가 실패하면, 판정서를 쓰고 감사 ②를 거쳐 커밋한 뒤 멈춘다.**
7. 감사 ②(판정 뒤)를 받는다. 차단이면 멈추고, 아니면 커밋한다.
8. §6-4 D1 r5 부록을 채워 v1.1로 커밋한다. 감사 ③(실행 전)을 받는다. 차단이면 멈춘다.
9. 러너 B: D1 r5 → D1 판정 → R1.
10. 판정자가 A2′·채택 판정서(`m31_n2a3_d1r5_judgment.md`)를 쓴다. 감사 ④(판정 뒤)를 받고 커밋한다. **여기서 멈춘다.** 기준선 교체와 `rule_O` = 50 %일 때의 기본값 변경은 사용자 결정이다.

### 7-3. 운영 (N2-a′와 같음, [P2] §6)

- **블라인드**: 학습·평가·판정 단계는 `--blind`로 돌린다. 메인 세션은 종료 줄·`.done`·자원만 보고, 판정자만 verdict·metrics·띠 지표를 연다.
- **HEAD 확인**: 다음 각 단계 앞에서 HEAD = 봉인 커밋, 추적 파일 변경 0, 인덱스 sha 일치를 단언한다. 러너 B에서는 봉인 커밋을 부록 커밋으로 바꿔 같은 단언을 한다. 어긋나면 그 단계에서 멈춘다.
  - 러너 A: 학습 fold마다, 평가·내보내기·A1 판정·띠 지표·B 판정 앞마다
  - 러너 B: D1 실행 앞, D1 판정 앞, R1 앞
- **빈 코어만**: 각 러너는 실행 직전 `~/.claude/tools/cpu-scheduler/status.py`로 다른 세션 예약과 부하를 보고 빈 코어만 affinity로 준다. compute-scheduler-expert 배치표를 실행 시점에 받는다. make_runner에 `--reserve-owner gpr_detection:n2a3 --reserve-ttl-min`(가장 긴 단계보다 길게)을 붙인다. GPU 작업이므로 `--gpu-log --gpu-counters`도 붙인다.
- **러너가 도는 동안 저장소에 커밋하지 않는다.**
- 러너 단계 하나라도 종료 코드가 0이 아니면(판정 스크립트의 정의된 0/1 제외), 또는 `log_errors`에 오류가 있으면 멈춘다.

### 7-4. 보고 전용 후속 (채택 판정 뒤, 자동 진행 밖)

- D3 재실행(같은 59측선, `lines.csv` sha256 f6fe08a93808555727acb3ecd864abc1cd555f0c6900a753a8c598be4543ca0a)으로 띠 수, 내용 아래 끝 띠 수, 알려진 꼭짓점 포함 수를 N2-a′와 비교한다. 사용자 승인 뒤 한다.

### 7-5. 시간·자원 (N2-a′ 실측 기준)

| 단계 | 추정 |
|---|---|
| 사전 확인 | 1분 미만(N2-a′ 진단 9 s × 설정 2 × seed 2 × 실행 방식 2) |
| 스모크 + 학습 5 fold | 약 2.4 h(1,644–1,700 s/fold, 스모크 117 s) |
| 평가·내보내기·판정·띠 지표 | 약 0.5 h |
| D1 r5 + R1 | 약 30분(r4 1,798 s, R1 14 s) |
| VRAM | 학습 약 7.4 GB, D1 2.5 GB 이하 |

## 8. 위험·한계 (판정서에 공개)

1. **쌍곡선 A1 여유(U3-c: 90.9 유지, 위험 공개).** 영상이 같아도 라벨이 바뀌면 학습 경로가 seed를 바꾼 것처럼 흔들린다. 효과가 0이어도 쌍곡선 AP50이 90.9 아래로 내려갈 확률을 약 30 %로 본다 **[추정]**. 근거는 seed에 따른 fold 0 쌍곡선 AP 흔들림 최대 6.2 %p(M1-7)이고, fold 평균의 흔들림 SD는 약 1.5 %p로 가정했다.
2. **주 가설의 정량 근거가 약해졌다(§1-3).** S-0으로 전제를 학습 전에 확인한다.
3. **K32의 비용.** 버린 조각 자리에 보이는 띠가 라벨 없이 남는다. S-4로 양을 제한한다.
4. **가설 (2)·(3)은 그대로다.** 일부만 회복되면 다음 후보는 NBc(이웃 band 세로 연속 연결) 또는 W32(band 유지 규칙 대체)다(v0.1 §2-1 비교).
5. **4/5 fold 조건은 약한 검정이다.** 효과가 0이어도 19 % 확률로 통과한다.
6. **D3 경계 오검출은 범위 밖이다.**
7. **효과가 작을 수 있다(§5-3).** K32가 바꾸는 것은 맥락 모드 band 라벨의 약 10 %(S-0 9.6–9.8 %)다. 학습 전체 band 라벨로는 약 5 % 이하이고, 보이는 길이로는 약 0.7–0.8 %다. 버린 조각은 모두 이웃 출처이고 중심 band 손실은 그대로다. B가 실패하면 "K32 크기의 변화로는 회복하지 않음"으로 해석하고, 가설 (1) 자체의 기각으로 보지 않는다.
8. **KS 거리는 원 분포에 가까워지지 않았다(0.148 → 0.156).** 판정 조건은 아니다. 짧은 꼬리는 S-1(0 %)로 없어졌다.

## 9. 사용자 결정 (2026-11-14, 기록)

| ID | 결정 |
|---|---|
| U3-a | K32 |
| U3-b | 띠 보조 재현율 평균 ≥ 64.1, N2-a′보다 높은 fold 4/5 이상, B-p 정밀도 ≥ 49.7 |
| U3-c | 쌍곡선 90.9 유지, 위험 공개 |
| U3-d | D1 r5 규칙 O 75 % 또는 50 %. 50 %이면 기본값 변경은 따로 결정 |
| U3-e | 불채택이면 결과를 본 뒤 정한다 |
| U3-f | 사전 확인 S-1–S-4는 권고값 |
| 자동 진행 | N2-a′와 같다. 채택 판정 커밋까지 가고, A1·B·A2′ 중 하나라도 미충족이거나 감사 차단이면 멈춘다 |

- S-0(N2-a′ 남은 band 라벨의 32 px 미만 비율 ≥ 5 %, 두 seed)은 v0.2에서 §1-3 정정 때문에 넣었다. 메인 세션이 이 문턱으로 사전 확인을 진행했으므로 v1.0에 그대로 둔다. **사용자 확인(2026-11-14, 메인 세션 기록): 사용자가 메시지 박스에서 "S-0 추가"를 선택해 승인했다.**

## 10. 위임

- **python-expert**: 구현 완료(§7-0, §7-2 1번).
  - (1) §6-3 verdict 수정과 시험: 구현 완료.
  - (2) 진단 json `original_crop_band_widths.counts_raw`의 정의 기록: 구현 완료(진단 스크립트 sha256 32e35c5c…, §5-3).
- **integrity-checker**: 봉인 전 점검.
- **verif-protocol-auditor**: 감사 ①–④.
- **compute-scheduler-expert**: 러너 A·B 실행 시점 배치표.

## 변경 이력

| 날짜 | 판 | 내용 |
|---|---|---|
| 2026-11-14 | v0.1 초안 | 원인 가설, 후보 7개 비교와 권고 K32, 가져올 것, 채택 조건 제안, 사전 확인 S-1–S-4, 운영, 사용자 결정 U3-a–U3-f |
| 2026-11-14 | v0.2 | 사용자 결정 반영(K32, B 4/5 fold + B-p, 90.9 유지, 규칙 O 75·50 %, S 권고값, 자동 진행 정지 조건). 구현 명세(§3: "잘림" = 원 라벨 폭 대비, 맥락 모드만, 코드 위치, 시험 K-1–K-10), 진단 스크립트 명세와 S-0–S-4 판정(§5), 판정 경로(A1은 `n2a_a1_judge.py` 무수정 재사용, 띠 지표 래퍼 `n2a3_band_aux.py`, B 판정 `n2a3_band_judge.py`, N2-a′ 기준 `n2a2_band_aux.json` sha256 bf42403e… 고정), D1 r5 부록 자리, 러너·감사 순서. **정정**(§1-3): `band_length_px_entering_window`는 창 단계 입력 전부(창 밖 포함)의 띠 좌표 폭이라 "남은 라벨의 30.5 %"가 아니다 → S-0 신설(사용자 확인 필요). v0.1의 후보 비교 표(§2-1)는 v0.2에서 뺐고 위험 4번에 다음 후보만 남겼다 |
| 2026-11-14 | v1.0 | 봉인 판. 구현 결과 §7-0(구현 차이 8건 승인, 확인 필요 5건 결정): (1) B 판정 verdict를 "B and B-p met"으로 바꾸고 `failed` 추가(명세 §6-3), (2) 띠 지표 ↔ run 결속은 G-b3 + 판정서 손 확인 (h1)–(h3), (3) §1-2 폭·가장자리 정의 명시(반올림 없음, [32, 64) 155개, 가장자리 2 px 687개), (4) 진단을 봉인 C1 HEAD에서 다시 돌려 C2로 커밋, `<SEAL>` = C2, (5) 진단 명령에 `-B`. 사전 확인 결과 §5-3(S-0–S-4 두 seed 통과, dirty 실행 참고값, 버림은 모두 이웃 출처, KS 0.148 → 0.156). 위험 7(효과 크기: 맥락 모드 band 라벨의 약 10 %, 전체 약 5 % 이하)·8(KS) 추가. 판정 문턱·규칙은 v0.2와 같다 |
| 2026-11-14 | v1.0(봉인 전 점검 반영) | integrity-checker C1 봉인 전 점검(차단 0) 반영, 뜻 불변: verdict 수정·`counts_raw` 정의 기록을 "구현 완료"로(§6-3, §7-0, §7-2 1, §10), 시험 수 갱신(1,410 passed, 14 skipped, 26 deselected, 2 xfailed, 판정 네 조합 포함, `--dry-run` fold 0–4 종료 0·누수 단언 0·K32 = 32), §5-3 dirty 진단 json 보관 위치(scratchpad `n2a3_dirty_diag`, 미커밋)와 진단 스크립트 sha 60fd9a6a… → 32e35c5c…(정의 필드 추가, 판정 로직 불변), 약칭 [SP] 정의, K-7·K-8에 §7-0 참조 |
| 2026-11-14 | v1.0(C1 재실행 확정) | §5-3에 C1(2e6526d) HEAD 깨끗한 트리 진단 재실행 확정 기록을 넣었다. dirty 0, sha 일치, `s_checks`가 dirty 실행과 정확히 같음, 정의 필드 155 / 687 / 164 / 271 / 9. S-0–S-4 확정 통과. C2에 진단 json과 함께 커밋한다. 규칙·문턱 불변 |
| 2026-11-14 | v1.0(판정 완료) | 상태 줄에 판정 결과만 추가: A1·B-p 충족, **B 미충족 → 불채택**, N2-a′ 기준선 유지(U3-e (a)), 판정서 [`m31_n2a3_a1b_judgment.md`](m31_n2a3_a1b_judgment.md). 규칙 본문 불변 |
