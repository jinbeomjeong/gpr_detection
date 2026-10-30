# ADR-004: 검출↔깊이 좌표는 3층(L0·L1·L2) 계약으로 고정하고, 역변환은 `pipeline` 한 곳에서 한다

- 상태: Accepted (2026-10-30 문서화. 결정은 2026-10-29 사용자 답변 Q2로 확정, M1-8에서 구현)
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

## 결정

1. 좌표는 세 층이다. 층마다 정의 위치를 고정한다.
   - L0 원본 격자: (trace `i`, sample `k`), 정수 인덱스, `data[k, i]`. `io/radargram.py`의 `Radargram`·`GridBox`(닫힌 구간 `i0..i1`, `k0..k1`). **깊이는 L0만 받는다.**
   - L1 검출 이미지: letterbox 전 영상 픽셀 (u, v). `preprocess/to_image.py`의 `ImageGridMap`(타일 원점, 축별 배율 `traces_per_px`·`samples_per_px`, 격자 크기).
   - L2 모델 입력: 백엔드가 만든 텐서 좌표. `detect/base.py`의 `InputTransform`(`x_L2 = x_L1·scale_x + pad_x`, 축별).
2. 픽셀 규약은 모든 층에서 연속 좌표, 픽셀 `j`는 [j, j+1)이다(Ultralytics `xyxy`와 같다).
3. `Detector.predict`는 박스를 **L2 그대로** 돌려주고 자기가 적용한 L1 → L2 변환을 `InputTransform`으로 함께 낸다(`DetectorOutput`). 백엔드는 역변환하지 않는다(ADR-001 결정 3).
4. 역변환 L2 → L1 → L0는 `pipeline.py`에만 둔다(`model_box_to_image_box` → `image_box_to_grid_box`, 묶음 `model_box_to_grid_box`·`detections_to_grid_boxes`). 반올림은 위 표의 바깥쪽 규칙(허용 오차 `GRID_ROUNDING_TOLERANCE = 1e-6`), 격자 범위로 자르고, 격자와 겹치지 않으면(letterbox 패딩 안 등) 버린다(`None`).
5. `ImageGridMap`은 등방 여부와 무관하게 축별 배율을 따로 가진다. 스케일 없는 입력(AIHub·모로코)은 `ImageGridMap.identity`(이미지 픽셀 = 격자)를 쓴다.
6. 세로축은 모든 층에서 시간 증가 방향이 아래다. 상하 반전은 어느 층에도 없다(augmentation_policy §1).
7. `depth`는 `Detection`을 모른다. `pipeline`이 `pipe_hyperbola`만 `GridBox`로 바꿔 넘긴다(ADR-002 결정 8).
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

검증 상태 (코드 읽기로 확인, 이 ADR 작성 때 시험은 실행하지 않음)
- M1-8 구현·시험이 있다: `tests/test_detect_contract.py`
  - L1 → L2 → L1 왕복 ≤ 0.5 px: L1 크기 6종(256×256, 224×224, 512×300, 512×1400, 180×700, 97×61) × 박스 50개, 입력 256 px.
  - 그린 사각형을 letterbox 픽셀에서 찾아 되돌리기(`InputTransform`이 실제 픽셀 위치와 맞는지).
  - L0 → L1 → L2 → L0 정확 왕복: 원점 (1200, 40), 축별 배율 2.5·1.75인 4TU형 타일, 무작위 `GridBox` 200개.
  - 바깥쪽 반올림, 클리핑, 격자 밖 `None`, 영폭 1셀, 패딩 안 검출 버림.
- 아직 없는 것: 실측 npy에서 만든 `ImageGridMap`(M3-2 `DetectionImage`)으로 하는 왕복(M3-1), RF-DETR 래퍼의 `InputTransform` 왕복(M4-4).

위험
- M3-2에서 L1을 만드는 방식(리사이즈만 / 타일 분할 / 겹침 타일)이 아직 정해지지 않았다. 겹침 타일이면 같은 관로가 두 타일에서 검출돼 L0에서 중복 박스가 생긴다. 이것은 좌표 계약이 아니라 `pipeline`의 병합 규칙 문제로, M3-2 설계 때 정한다.
- QA-8 허용 오차(≤ 1 trace·1 sample)는 확정됐지만(2026-10-30), 깊이 피팅 창 1셀 차이가 깊이 오차에 주는 영향은 측정하지 않았다(gpr-processing-expert 판정 대상).

후속 과제
- M3-1: 실측 경로(M3-2 `DetectionImage` + `ImageGridMap`)로 L0 왕복 시험을 추가하고 QA-8을 판정한다. 판정 기준(확정 2026-10-30): 격자에 맞춘 `GridBox`는 L0 → L1 → L2 → L0 왕복 오차 0(정확히 같은 `GridBox`), 검출 박스의 L0 복원은 바깥쪽 반올림으로 각 변 1 trace·1 sample 미만만 넓어짐(안쪽으로 줄어들면 실패).
- M4-4: RF-DETR 래퍼의 `InputTransform`(정사각 늘림이면 `scale_x ≠ scale_y`) 왕복 시험 ≤ 0.5 px.
- M3-2: 타일 분할·겹침과 중복 박스 병합 규칙을 정하면 이 ADR의 위험 항목을 갱신한다.

## 근거 문서

- [README](README.md) §2-3 — 3층 좌표 표, 인터페이스 스케치, 계약 규칙, §5 Q2(사용자 답변 2026-10-29)
- [ADR-001](ADR-001-detector-migration.md) 결정 3 — `Detector` 프로토콜과 `InputTransform` 반환
- [ADR-002](ADR-002-physics-based-depth.md) 결정 7·8 — 피팅 갈래, 깊이 대상 클래스
- [`docs/experiments/augmentation_policy.md`](../experiments/augmentation_policy.md) §1, §4 — 상하 반전 금지, RF-DETR 정사각 리사이즈
- 코드: `src/gpr_detection/pipeline.py`, `src/gpr_detection/preprocess/to_image.py`, `src/gpr_detection/detect/base.py`, `src/gpr_detection/io/radargram.py`, `tests/test_detect_contract.py`
- [`docs/CHECKLIST.md`](../CHECKLIST.md) M1-8(완료), M3-1, M3-2, M4-4
