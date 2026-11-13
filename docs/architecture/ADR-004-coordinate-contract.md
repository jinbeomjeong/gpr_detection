# ADR-004: 검출↔깊이 좌표는 3층(L0·L1·L2) 계약으로 고정하고, 역변환은 `pipeline` 한 곳에서 한다

- 상태: Accepted (2026-10-30 문서화. 결정은 2026-10-29 사용자 답변 Q2로 확정, M1-8에서 구현). 결정 7 개정 2026-10-30(`pipe_band` 전달·측선 단위 처리 순서, 아래 변경 이력). M3-1 반영 2026-11-14: 검출 영상(L1) 방식 C′·타일 원점·병합 위치 하위 결정 추가, 결정 1·4 보완(측선 L1과 타일 원점), 결정 7 `pipeline` 연결 구현됨, QA-8 실측 합격
- 날짜: 2026-10-30
- 관련 드라이버: QA-8(좌표 계약), QA-1(검출기 교체), FR-2, FR-3, QA-4, CRN-2, CRN-4 — 정의는 [README](README.md) §1
- 반복: 1회차 결정의 기록 보완 (README §2-3에만 있던 결정을 ADR로 옮김, 2026-10-30 `adr_lint.py` 고아 드라이버 경고 대응)

## 맥락

- 깊이 갈래는 원본 격자 (trace `i`, sample `k`)만 받는다(spec_depth_v1 §1-2, ADR-002). 검출 갈래는 영상화·리사이즈(L1)를 거친 뒤, 백엔드가 다시 letterbox(Ultralytics) 또는 정사각 리사이즈(RF-DETR `square_resize_div_64`)를 한 모델 입력(L2)에서 박스를 낸다.
- 박스가 한 trace·한 sample만 어긋나도 쌍곡선 피팅 창이 바뀐다. 4TU 512×N 측선처럼 가로세로 배율이 다른 입력에서는 축별 배율을 하나로 뭉뚱그리면 오차가 커진다(README §2-3).
- 검출기는 M4에 YOLO11m에서 RF-DETR로 바뀐다(ADR-001). 역변환이 백엔드마다 따로 있으면 교체 때마다 좌표 코드를 새로 쓰고 시험해야 한다(QA-1).
- 스케일 없는 입력(AIHub·모로코 JPG)도 박스 좌표는 필요하다. 이때는 이미지 픽셀이 곧 격자다(FR-3, QA-4).
- 사용자 결정(2026-10-29, README §5 Q2): 좌표 역변환은 **pipeline 공통 처리**, 백엔드는 L2 좌표 + `InputTransform`을 낸다.

## 고려한 대안

| 대안 | 장점 | 단점 | 드라이버 충족 |
|---|---|---|---|
| **A. 백엔드는 L2 박스 + `InputTransform`(축별 배율·패딩)만 내고, L2 → L1 → L0 역변환은 `pipeline` 한 곳에서** (채택) | 역변환 코드가 하나라 왕복 시험도 하나(QA-8). 백엔드 래퍼가 얇다. 백엔드를 바꿔도 좌표 코드는 그대로(QA-1) | 백엔드가 라이브러리 내부 전처리 값(Ultralytics `ratio_pad` 등)을 꺼내 노출해야 한다. "축별 배율 + 패딩"으로 표현되지 않는 전처리(크롭·타일링·비선형 리사이즈)가 오면 스키마와 `pipeline`을 같이 고쳐야 한다 | QA-8 ○, QA-1 ○(스키마 안일 때) |
| B. 백엔드가 L2 → L1까지 되돌려 L1 박스를 낸다(Ultralytics `scale_boxes` 기본 동작) | 라이브러리 기본 출력을 그대로 써 래퍼 코드가 적다 | 역변환이 백엔드 수만큼 생기고, 반올림·클리핑 규칙이 백엔드마다 달라질 수 있다. RF-DETR 정사각 늘림처럼 비등방 변환은 라이브러리 기본 역변환이 맞는지 따로 확인해야 한다 | QA-8 △(백엔드마다 시험), QA-1 △ |
| C. 백엔드가 `ImageGridMap`까지 받아 L0 `GridBox`를 바로 낸다 | `pipeline`이 가장 단순 | `detect`가 격자(`io`·`preprocess`) 개념을 알아야 해 의존 규칙(README §2-2 규칙 1·2)이 약해지고, 백엔드마다 L0 반올림을 다시 구현한다 | QA-8 △, QA-1 ✕ |
| D. `to_image`가 L1을 모델 입력 크기로 미리 맞춰 L1 = L2로 만든다(백엔드는 리사이즈 안 함) | 층이 둘로 줄어 변환이 하나 | 모델 입력 크기가 전처리에 박혀 백엔드 교체 때 `preprocess`도 바뀐다. 정사각 입력이면 쌍곡선 종횡비가 바뀐다(augmentation_policy §4, ADR-001 위험) | QA-1 ✕, QA-8 ○ |

L1 → L0 반올림 규칙(대안 A 안의 하위 결정):

| 규칙 | 장점 | 단점 | 판정 |
|---|---|---|---|
| **바깥쪽 반올림**: 연속 구간 [t1, t2] → 닫힌 인덱스 `floor(t1+1e-6) .. ceil(t2−1e-6)−1`, 영폭은 1셀 (채택) | 박스가 덮은 셀을 하나도 잃지 않아 쌍곡선 꼭짓점·날개가 잘리지 않는다. 격자에 맞는 박스는 왕복이 정확(오차 0) | 연속 박스는 각 변에서 1셀 미만만큼 넓어질 수 있다 | 채택 |
| 가장 가까운 정수로 반올림 | 대칭, 평균 편향 없음 | 0.5 경계에서 셀을 잃을 수 있고 영폭 박스가 0셀이 된다 | 기각 |
| 안쪽 반올림 | 배경을 덜 포함 | 작은 박스가 사라지고 꼭짓점이 잘릴 수 있다 | 기각 |

깊이 갈래 호출 순서(결정 7의 하위 결정, 2026-10-30 추가):

| 대안 | 장점 | 단점 | 판정 |
|---|---|---|---|
| **P1. `pipeline`이 측선 단위로 영점(띠 트레이스 제외) → 쌍곡선 → 속도 풀 → 띠 순서로 부른다. `depth` 함수는 측선 하나·박스 목록만 안다** (채택) | 영점을 한 번 구해 두 클래스가 같은 영점을 쓴다(위상 기준 일치, ADR-002 결정 4). 조사 단위 풀처럼 측선을 넘는 상태가 `depth` 밖에 있어 `depth`는 순수 함수로 시험된다(CRN-2) | `pipeline`에 순서 규칙과 조사 단위 묶기가 생긴다. 조사 단위 풀이 필요하면 두 단계 처리 | 채택 (FR-2, QA-2, CRN-2) |
| P2. `depth`에 측선 단위 진입점 하나(두 클래스 박스 목록을 받아 내부에서 순서 처리) | `pipeline`이 단순 | 조사 단위 풀은 여러 측선을 봐야 해 결국 `depth` 밖 상태가 필요하다. 클래스 이름을 `depth`가 알아야 한다 | 보류(측선 내부만 감싸는 편의 함수로는 M2-6b에서 허용) |
| P3. 박스마다 영점을 따로 구한다 | 박스 사이 의존 없음 | 띠 트레이스가 섞인 영점이 생기고(band §3-1, d ≲ 0.21 m 띠를 지표 반사로 오인) 같은 측선에서 영점이 달라진다 | 기각 |

검출 영상(L1) 생성 방식(결정 1·4의 하위 결정, M3-1. 사용자 결정 Q1 2026-11-13, 비교 원본은 [`spec_detection_image_v1.md`](../processing/spec_detection_image_v1.md) §3):

| 대안 | 장점 | 단점 | 판정 |
|---|---|---|---|
| A. 측선 전체를 비율 유지로 한 장에 리사이즈 | 타일·병합이 없다 | 쌍곡선 종횡비가 깊이 단위 대비 세로 2.0–8.2배로 바뀌고, 작은 관로가 4TU p5 9 px·GPR-SD 0.5–1.5 px로 사라진다 | 기각 |
| B-native. 고정 폭 타일, 원래 격자 1:1 | 리샘플이 없다 | 종횡비 문제가 A와 같다 | 기각 |
| B-matched. 맞춤 배율 타일, 겹침 없음 | 쌍곡선 45° 점근, 작은 관로 유지 | 타일 경계에서 쌍곡선이 잘린다 | C′로 대체 |
| **C′. 주기 맞춤 세로 배율 + 깊이 등방 가로 배율 + 256×256 겹침 타일**(시간창 전체, 넘치면 두 축 압축) (채택) | B-matched의 장점에 더해 경계 잘림을 겹침과 병합으로 푼다. 타일이 학습 crop(256×256)과 같은 크기라 letterbox가 항등이다 | 한 열이 여러 타일에 들어가 추론 횟수가 는다(겹침 192 px면 보폭 64 px, 열마다 최대 4장). 중복 박스 병합 규칙이 생긴다 | 채택 (QA-8, FR-1, FR-2) |

타일 원점과 `ImageGridMap` 단위(사용자 결정 Q8·U6, 2026-11-13):

| 대안 | 장점 | 단점 | 판정 |
|---|---|---|---|
| **T1. 측선 전체를 한 번 리샘플한 측선 L1 + 정수 px 타일 원점 `x0`, `ImageGridMap`은 측선당 1개**(`trace_offset = sample_offset = 0`) (채택) | 겹친 두 타일의 같은 열은 픽셀이 같다. 타일 L1 → 측선 L1은 정수 평행 이동(`u_line = u_tile + x0`)이라 오차가 없다. `ImageGridMap` 스키마를 바꾸지 않는다 | 측선 L1 영상 전체를 메모리에 둔다 | 채택 (QA-8, QA-1) |
| T2. 타일마다 원본 격자에서 따로 리샘플, 타일마다 `ImageGridMap`(`trace_offset` = 타일 원점, 명세 v0.1의 칸 정렬 방식) | 타일 하나만 만들 수 있다 | 타일 원점을 격자 칸에 맞추려면 배율이 제약된다. 겹친 열 픽셀이 타일마다 다를 수 있다. 타일마다 L0 반올림이 생긴다 | 기각 |

중복 박스 병합 위치(사용자 결정 2026-11-13, 명세 §5):

| 대안 | 장점 | 단점 | 판정 |
|---|---|---|---|
| **M1. `pipeline`이 반올림 전 측선 L1 연속 좌표에서 병합(R2: IoU ≥ 0.7, 잘린 박스가 끼면 IoMin ≥ 0.8, 대표 1개, 평균 없음 + 물리 거부권 V1·V2, `pipe_band`는 트레이스 구간만 중재), L0 바깥쪽 반올림은 병합 뒤 한 번** (채택) | 반올림이 한 번이라 결정 8의 오차 한계(각 변 1셀 미만)가 병합 뒤에도 그대로다. 병합이 좌표 변환과 같은 모듈에 있다 | `pipeline`에 병합 규칙과 물리 거부권이 더해져 커진다 | 채택 (QA-8, FR-2) |
| M2. 타일마다 L0로 반올림한 뒤 `GridBox`끼리 병합 | 병합이 정수 격자에서 단순하다 | 반올림이 두 번 일어나 1셀 넘게 넓어질 수 있다(결정 8 위반 가능). 겹친 영역의 박스가 1셀 차로 IoU가 흔들린다 | 기각 |
| M3. 백엔드(`detect/`)가 타일을 받아 병합까지 한다 | `pipeline`이 단순하다 | 병합 규칙이 백엔드마다 생기고 물리 거부권(축척·`dx_m`)을 `detect`가 알아야 한다(README §2-2 규칙 1, QA-1) | 기각 |

## 결정

1. 좌표는 세 층이다. 층마다 정의 위치를 고정한다.
   - L0 원본 격자: (trace `i`, sample `k`), 정수 인덱스, `data[k, i]`. `io/radargram.py`의 `Radargram`·`GridBox`(닫힌 구간 `i0..i1`, `k0..k1`). **깊이는 L0만 받는다.**
   - L1 검출 이미지: letterbox 전 영상 픽셀 (u, v). `preprocess/to_image.py`의 `ImageGridMap`(L1 원점의 L0 위치 `trace_offset`·`sample_offset`, 축별 배율 `traces_per_px`·`samples_per_px`, 격자 크기).
     - 물리 스케일이 있는 입력(M3-1, C′): L1은 **측선 L1**(측선 전체를 한 번 리샘플한 `DetectionImage.line_image`)이고 `ImageGridMap`은 측선당 1개다. `make_detection_image`는 `trace_offset = sample_offset = 0`으로 둔다. 곧 `trace_offset`은 타일 원점이 아니라 측선 L1 기준(0)이다.
     - 타일은 측선 L1을 자르기만 한다. 타일 원점은 `LineTile.x0_px`(측선 L1의 정수 px)이고, 타일 L1 → 측선 L1은 `u_line = u_tile + x0_px`(`pipeline.tile_box_to_line_box`)다.
     - 스케일 없는 입력(AIHub·모로코)은 `ImageGridMap.identity`(결정 5)다.
   - L2 모델 입력: 백엔드가 만든 텐서 좌표. `detect/base.py`의 `InputTransform`(`x_L2 = x_L1·scale_x + pad_x`, 축별).
2. 픽셀 규약은 모든 층에서 연속 좌표, 픽셀 `j`는 [j, j+1)이다(Ultralytics `xyxy`와 같다).
3. `Detector.predict`는 박스를 **L2 그대로** 돌려주고 자기가 적용한 L1 → L2 변환을 `InputTransform`으로 함께 낸다(`DetectorOutput`). 백엔드는 역변환하지 않는다(ADR-001 결정 3).
4. 역변환 L2 → L1 → L0는 `pipeline.py`에만 둔다(`model_box_to_image_box` → `image_box_to_grid_box`, 묶음 `model_box_to_grid_box`·`detections_to_grid_boxes`). 반올림은 위 표의 바깥쪽 규칙(허용 오차 `GRID_ROUNDING_TOLERANCE = 1e-6`), 격자 범위로 자르고, 격자와 겹치지 않으면(letterbox 패딩 안 등) 버린다(`None`).
   - 타일 경로(M3-1): L2 →(`InputTransform`) 타일 L1 →(`tile_box_to_line_box`, 정수 `x0_px` 더하기) 측선 L1 연속 박스 `LineBox` →(`merge_line_boxes`, 위 M1 병합) →(`line_boxes_to_grid`) L0 `GridBox`. 묶음 함수는 `detect_tiles`·`detect_line`이다. **L0 반올림은 병합 뒤 한 번만** 한다.
   - 기본값: 목표 주기 $P^*$ **6 px**, 겹침 **192 px(75 %)**. D1 r4(2026-11-14)로 정했고 기준선 교체(사용자 결정 N6-a, ADR-001)로 확정했다. 정본은 `spec_detection_image_v1.md` §4-2다. 값은 파라미터(`DetectionImageConfig`)라 이 ADR의 구조 결정과 무관하다.
5. `ImageGridMap`은 등방 여부와 무관하게 축별 배율을 따로 가진다. 스케일 없는 입력(AIHub·모로코)은 `ImageGridMap.identity`(이미지 픽셀 = 격자)를 쓴다.
6. 세로축은 모든 층에서 시간 증가 방향이 아래다. 상하 반전은 어느 층에도 없다(augmentation_policy §1).
7. `depth`는 `Detection`을 모른다. 박스의 클래스 판별은 `pipeline`이 하고, `pipe_hyperbola`와 `pipe_band` 박스를 `GridBox`(L0)로 바꿔 각각 `depth`(쌍곡선)와 `depth.band`로 넘긴다. `cavity`·`manhole`은 깊이로 넘기지 않는다(ADR-002 결정 8, 개정 2026-10-30, 근거 spec_band_depth_v1 §2·§8-1). `depth/band.py`·`depth/velocity.py`도 `io.radargram`·`preprocess`만 import하고 `detect`·`pipeline`은 import하지 않는다(README §2-2 규칙 1). 처리 순서는 측선(radargram) 단위로 고정한다.
   1. **영점(측선당 한 번)**: `prepare_depth_inputs(radargram, config, exclude_traces=<pipe_band 박스 트레이스(좌우 여유 포함)>)` → `DepthInputs(branch, zero)`. 띠 박스 트레이스를 평균 트레이스에서 빼고(`find_time_zero`의 `exclude_traces`), 측선 전체가 띠면 빼지 않고 `zero_band_overlap`을 붙인다(band §3-1). 제외 마스크는 `pipeline`이 L0 `GridBox`에서 만든다.
   2. **쌍곡선**: `pipe_hyperbola` 박스마다 `estimate_depth(radargram, box, config, inputs=inputs)`. 같은 `inputs`(같은 영점)를 재사용한다(M2-4 구조).
   3. **속도 풀**: 2의 `DepthResult`마다 `velocity_estimate_from(result, *, line_id, source_index, survey_key=None)` → 측선 풀(`pool`, v가 없는 결과는 `None`이라 빠진다). 지면결합의 조사 단위 풀(`survey_pool`)은 같은 조사 단위 키(`survey_key`)의 모든 측선에서 2를 끝낸 뒤 모은다.
   4. **띠**: `estimate_band_depths(radargram, band_boxes, inputs.zero, pool, survey_pool, cfg, branch=inputs.branch)`. 조각 박스 묶기는 여기서 한다(Q3). `branch`를 넘기지 않으면 `band`가 (1)–(3)을 다시 계산한다.
   - 영점·쌍곡선·띠는 같은 (1)–(3) 전처리 자료(`FittingBranch`, `inputs.branch`)를 공유한다. 영점은 `filtered`, 쌍곡선은 `background_removed`(측선 median), 띠는 `filtered`에서 박스 밖 트레이스로 masked median을 따로 구한다(spec_band_depth_v1 §3-2).
   - 조사 단위 풀이 필요하면 조사 단위 안의 모든 측선에 1–3을 먼저 하고 4를 나중에 하는 두 단계 처리가 된다. 조사 단위 풀이 없으면(4TU 공기결합·합성) 측선 하나로 끝난다.
   - 함수 이름·인자는 spec_band_depth_v1 §8-2를 따르고 `depth/band.py`·`depth/velocity.py`에 구현됨(2026-10-31, M2-6b).
   - `pipeline`의 이 연결은 **M3-1에서 구현됨**(커밋 `abbe8dc`, 명세 `spec_detection_image_v1.md` §6). 측선 하나는 `estimate_line_depths(radargram, boxes, depth_config, band_config, *, line_id, survey_key, survey_pool)`가 `band_exclude_traces` → `prepare_depth_inputs`(영점 한 번) → 쌍곡선마다 `estimate_depth` → `velocity_estimate_from` → `estimate_band_depths` 순서로 부른다. 조사 단위 두 단계 처리는 `estimate_survey_depths(lines, ..., survey_key=...)`다(1단계 모든 측선의 ①–③, 2단계 조사 단위 풀로 ④). 입력 `DepthBox`는 병합 결과(`LineDetections`)에서 `depth_boxes_from`으로 만든다. 제외 트레이스 마스크는 `pipeline`이 L0 `GridBox`로 `depth.band.band_exclude_traces`를 불러 만든다.
   - 남은 것: `gpr-detect infer`(M5-1)에서 이 함수를 부르는 연결과 조사 단위 키를 로더 메타에서 꺼내는 일이다.
8. QA-8 판정 기준: 격자에 맞춘 박스의 L0 → L1 → L2 → L0 왕복은 **정확히 같은 `GridBox`**, 연속 박스의 L1 → L2 → L1 왕복은 ≤ 0.5 px, 검출 박스의 L0 복원 오차는 바깥쪽 반올림 때문에 각 변 1 trace·1 sample 미만(바깥쪽으로만 넓어짐). README QA-8 척도 ≤ 1 trace·1 sample은 사용자가 2026-10-30에 확정했다.

## 결과

얻는 것
- 좌표 변환 코드가 한 곳이라 QA-8 시험이 `pipeline` 함수 묶음 하나만 본다. 백엔드를 바꿔도 이 시험은 그대로다(QA-1).
- 깊이 갈래가 검출 라이브러리·영상 크기를 몰라도 된다(README §2-2 규칙 1).
- 비정사각·축별 배율이 다른 입력(4TU 측선 타일)도 같은 식으로 처리한다.

잃는 것 / 비용
- 백엔드 래퍼가 라이브러리 내부 전처리 값을 정확히 노출해야 한다. 현재 YOLO 래퍼는 letterbox를 직접 수행해(`detect/base.py`의 `letterbox`, Ultralytics `LetterBox(center=True)` 재현) 값을 얻는다. 라이브러리 버전이 패딩 반올림 규칙을 바꾸면 래퍼가 어긋난다.
- `InputTransform` 스키마는 "축별 배율 + 패딩"만 표현한다. 크롭·타일링·비선형 리사이즈를 하는 백엔드가 오면 스키마와 `pipeline`을 함께 고쳐야 하고, 그때는 QA-1의 "`detect/`·`configs/` 밖 0줄"이 깨진다.
- 바깥쪽 반올림으로 깊이 피팅 창이 각 변 최대 1셀 미만 넓어진다(배경 점이 조금 더 들어온다).
- (M3-1) 겹침 타일 때문에 측선당 추론 횟수가 늘고(겹침 192 px면 보폭 64 px), `pipeline`에 병합(R2·물리 거부권·띠 구간 중재)이 더해져 이 모듈의 책임이 커졌다. 측선 L1 영상 전체를 메모리에 둔다.

검증 상태 (코드 읽기로 확인, 이 ADR 작성 때 시험은 실행하지 않음)
- M1-8 구현·시험이 있다: `tests/test_detect_contract.py`
  - L1 → L2 → L1 왕복 ≤ 0.5 px: L1 크기 6종(256×256, 224×224, 512×300, 512×1400, 180×700, 97×61) × 박스 50개, 입력 256 px.
  - 그린 사각형을 letterbox 픽셀에서 찾아 되돌리기(`InputTransform`이 실제 픽셀 위치와 맞는지).
  - L0 → L1 → L2 → L0 정확 왕복: 원점 (1200, 40), 축별 배율 2.5·1.75인 4TU형 타일, 무작위 `GridBox` 200개.
  - 바깥쪽 반올림, 클리핑, 격자 밖 `None`, 영폭 1셀, 패딩 안 검출 버림.
- **M3-1 완료**(2026-11-13, `abbe8dc`): 실측 npy에서 만든 측선 `ImageGridMap`(`DetectionImage`, C′)으로 4TU·광저우·GPR-SD 11측선을 왕복했다. 격자 정렬 `GridBox` 3,280개는 정확히 같았고, 연속 검출 박스 9,840개는 바깥쪽으로만 1셀 미만 넓어졌다(최대 트레이스 0.99999·샘플 0.99995셀, 안쪽으로 준 변 0). QA-8 합격. 근거는 `P*` 4.06 판정([`docs/processing/m31_qa8_roundtrip.md`](../processing/m31_qa8_roundtrip.md))과 `P*` 6.0 재확인(같은 문서 추가 절, gpr-processing-expert, 2026-11-14)이다. 재확인 결과는 격자 정렬 `GridBox` 3,740개 정확 왕복, 연속 박스 11,220개 안쪽으로 준 변 0·1셀 이상 넓어진 변 0, realdata 시험 13/13 통과다.
- 남은 것: RF-DETR 래퍼의 `InputTransform` 왕복(**M4-4**)뿐이다. 실제 Ultralytics 추론 내부 변환이 `detect/base.py` `letterbox` 재현과 같은지도 이 판정 범위 밖이다(판정 문서 "남은 한계" 3).

위험
- **결정됨**(M3-1, 2026-11-13. 겹침·`P*`는 D1 r4로 2026-11-14 확정): 측선 L1 1장(주기 맞춤 세로·깊이 등방 가로 배율) + 256×256 겹침 타일(정수 px `x0`, 측선당 `ImageGridMap` 1개). 중복 박스 병합은 `pipeline`이 측선 연속 좌표에서 R2 + 물리 거부권 V1·V2로 하고, 반올림은 한 번만 한다(`docs/processing/spec_detection_image_v1.md`, 위 하위 결정 표 C′·T1·M1). 겹침 폭은 192 px(75 %), `P*`는 6 px다(D1 r4, [`m31_n2a2_d1r4_judgment.md`](../experiments/m31_n2a2_d1r4_judgment.md) §1).
- **측정됨**: 정수 px 바깥쪽 양자화의 깊이 변화 ≤ 1.0 mm(gprMax 20, 명세 T10, `data/processing/l1_box_sensitivity.csv`). 검출기 박스 위 끝 오차(±1–2 L1 px)의 깊이 영향은 QA-8 밖이고, 명세 §7 (a)·Q6에서 계속 본다.
- (닫음 2026-11-14) 코드 기본값과 docstring: **반영됨**(2026-11-14 M3-1 마감 커밋, `P*` 6.0·`compress_max` 1.35, `ImageGridMap` docstring을 측선 L1 기준으로 수정). 계약 변경은 아니다.

후속 과제
- (완료 2026-11-13) M3-1: 실측 경로 L0 왕복 시험과 QA-8 판정 — 합격(위 검증 상태). 결정 7의 `pipeline` 연결도 같은 커밋에서 구현됨.
- M4-4: RF-DETR 래퍼의 `InputTransform`(정사각 늘림이면 `scale_x ≠ scale_y`) 왕복 시험 ≤ 0.5 px.
- (닫음 2026-11-14) M3-2의 "타일 분할·겹침과 중복 박스 병합 규칙"은 M3-1에서 정했다(위 위험 첫 항목). M3-2에 남은 것은 영상 품질(채움 값·회색 매핑·검출 영상 배경 제거 창, `assumed_params`)이고 이 ADR의 계약과 무관하다.
- M5-1(결정 7): `gpr-detect infer`에서 `detect_line` → `depth_boxes_from` → `estimate_line_depths`/`estimate_survey_depths`를 연결한다. 지면결합 조사 단위 풀을 쓰려면 조사 단위 키(광저우 `zone`, GPR-SD 도로)가 필요하다(band §5-1). `estimate_survey_depths`는 키를 인자로 받는다. 로더에는 조사 단위 키 필드가 따로 없고, `file_id`(광저우 `<target>/<mission_zone>/<file>`, GPR-SD `<road>_<line>`)에서 뽑을 수 있다 `[추정]`(코드 읽기, 2026-10-30).

## 근거 문서

- [README](README.md) §2-3 — 3층 좌표 표, 인터페이스 스케치, 계약 규칙, §5 Q2(사용자 답변 2026-10-29)
- [ADR-001](ADR-001-detector-migration.md) 결정 3 — `Detector` 프로토콜과 `InputTransform` 반환
- [ADR-002](ADR-002-physics-based-depth.md) 결정 7·8 — 피팅 갈래, 깊이 대상 클래스(`pipe_band` 규칙 2026-10-30)
- [`docs/processing/spec_band_depth_v1.md`](../processing/spec_band_depth_v1.md) §2(입력 계약·ADR 반영 요청), §3-1(띠 트레이스 제외 영점), §5(속도 풀), §8-1·§8-2(모듈 배치·시그니처)
- [`docs/processing/spec_depth_v1.md`](../processing/spec_depth_v1.md) §13(`prepare_depth_inputs`·`estimate_depth`, `ZeroResult` 정본)
- [`docs/experiments/augmentation_policy.md`](../experiments/augmentation_policy.md) §1, §4 — 상하 반전 금지, RF-DETR 정사각 리사이즈
- 코드: `src/gpr_detection/pipeline.py`, `src/gpr_detection/preprocess/to_image.py`, `src/gpr_detection/detect/base.py`, `src/gpr_detection/io/radargram.py`, `tests/test_detect_contract.py`
- [`docs/CHECKLIST.md`](../CHECKLIST.md) M1-8(완료), M2-6b, M3-1, M3-2, M4-4, M5-1
- [`docs/processing/spec_detection_image_v1.md`](../processing/spec_detection_image_v1.md) §3(L1 대안), §4(`DetectionImage`·측선당 `ImageGridMap`), §5(병합 R2·V1·V2), §6(측선 단위 깊이 호출), §8(시험 T1–T10) — M3-1
- [`docs/experiments/m31_detection_input_tiling.md`](../experiments/m31_detection_input_tiling.md) — 검출기 쪽 타일·병합 규칙과 D1, [`m31_n2a2_d1r4_judgment.md`](../experiments/m31_n2a2_d1r4_judgment.md) — `P*` 6·겹침 192 px 확정
- [`docs/processing/m31_qa8_roundtrip.md`](../processing/m31_qa8_roundtrip.md) — QA-8 실측 왕복 판정(합격)과 이 ADR 갱신 문구 제안
- 코드(M3-1, 커밋 `abbe8dc`): `src/gpr_detection/preprocess/to_image.py`(`DetectionImage`, `LineTile`, `make_detection_image`), `src/gpr_detection/pipeline.py`(`tile_box_to_line_box`, `merge_line_boxes`, `line_boxes_to_grid`, `detect_line`, `estimate_line_depths`, `estimate_survey_depths`)
- 코드(결정 7): `src/gpr_detection/depth/estimate.py`(`prepare_depth_inputs`·`estimate_depth`·`DepthInputs`), `src/gpr_detection/preprocess/timezero.py`(`find_time_zero`의 `exclude_traces`), `src/gpr_detection/depth/band.py`(`estimate_band_depths`)·`velocity.py`(`velocity_estimate_from`·속도 풀)·`result.py`(`band` 필드) — 구현됨(2026-10-31, M2-6b)

## 변경 이력

| 날짜 | 변경 | 근거 |
|---|---|---|
| 2026-10-30 | 최초 문서화(결정 1–8) | README §2-3, 사용자 답변 Q2(2026-10-29), M1-8 |
| 2026-10-30 | 결정 7 개정: `pipe_band` 박스도 `depth.band`로 전달, 측선 단위 처리 순서(영점·띠 트레이스 제외 → 쌍곡선 → 속도 풀 → 띠), 하위 결정 표 P1–P3, `pipeline` 연결은 M3-1·M5-1 몫 명시 | spec_band_depth_v1 §2·§8, ADR-002 결정 8 개정, M2-6b와 함께 개정(사용자 결정 2026-10-30) |
| 2026-10-31 | 결정 7 정정(M2-6b 구현 대조): ③ `velocity_estimate_from` 실제 시그니처, ④ `branch=inputs.branch` 추가, 같은 `FittingBranch` 공유 명시, "구현 중"·"`band` 필드 자리" → 구현됨 | `depth/band.py`·`velocity.py`·`estimate.py` 코드, integrity-checker 지적 |
| 2026-11-14 | M3-1 반영: 하위 결정 표 3개(L1 방식 C′, 타일 원점 T1, 병합 위치 M1) 추가. 결정 1 보완(`trace_offset`은 측선 L1 기준 0, 타일 원점은 정수 `LineTile.x0_px`), 결정 4에 타일 경로·반올림 1회·기본값(`P*` 6, 겹침 192) 추가, 결정 7 `pipeline` 연결 구현됨(`estimate_line_depths`·`estimate_survey_depths`, 남은 것 M5-1). 검증 상태(M3-1 완료, M4-4만 남음), 위험(L1 방식 결정됨, 1셀 깊이 영향 측정됨 ≤ 1.0 mm, 코드 기본값·docstring 반영 위험 항목), 후속 과제(M3-1 완료, M3-2 타일·병합 항목 닫음) 갱신 | `m31_qa8_roundtrip.md` 갱신 문구 제안, spec_detection_image_v1 v0.3.6, D1 r4 판정서, 사용자 결정 Q1·Q8(2026-11-13)·N6-a(2026-11-14), 코드 `abbe8dc` |
| 2026-11-14 | M3-1 마감 반영: 기본값·docstring 위험 항목 닫음(마감 커밋에 `P*` 6.0·`compress_max` 1.35·`ImageGridMap` docstring 측선 L1 기준 반영). QA-8 근거에 `P*` 6.0 재확인(정렬 3,740개 정확 왕복, 연속 11,220개 안쪽 0·1셀 이상 0, realdata 13/13) 추가, `[추정]` 삭제 | M3-1 마감 정합 점검 지적, `m31_qa8_roundtrip.md` 추가 절(gpr-processing-expert) |
