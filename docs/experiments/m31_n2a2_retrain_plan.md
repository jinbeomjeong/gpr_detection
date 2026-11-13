# M3-1 N2-a′ — 문맥 이동 증강 재학습 계획

- 상태: **v1.0 봉인 판** (2026-11-14, cv-detection-expert). 구현은 C-22–C-25까지 끝났다(`tests/test_context_augment.py` 43개 통과, 전체 1,361 passed, python-expert 기준). 이 문서 v1.0과 설정·구현을 한 커밋으로 봉인한다. v1.0은 v0.3(python-expert 구현 결과 §7-0, 확인 필요 4건 결정 §7-5)에 integrity-checker 봉인 전 점검을 반영한 것이고, 뜻은 바뀌지 않았다.
- 배경: D1 r3 판정서 [`m31_detection_input_tiling.md`](m31_detection_input_tiling.md) §8-6(이하 [T]). N2-a는 축척 손실과 채움 이동 손실을 거의 없앴다. 그러나 **채움 없이 실제 맥락 안에서 옮겨진 객체**(X6, 1.478배 확대 영상 안에서 창만 옮김)는 Δ(32) 15.7 %p, Δ(64) 42.1 %p를 잃었다. 채움 띠만으로도 11.6 %p가 떨어졌다. 그 결과 A2 미충족으로 불채택됐다.
- 이전 계획: [`m31_n2a_retrain_plan.md`](m31_n2a_retrain_plan.md) v0.3.4(이하 [P]), N2-a A1 판정서 [`m31_n2a_a1_judgment.md`](m31_n2a_a1_judgment.md).
- 자료: `data/aihub_context_index/`(`crops.csv`, `groups.csv`, `adjacent_pairs.csv`), 스크립트 `analysis/data/aihub_context_groups.py`(gpr-dataset-curator, 2026-11-13). 매니페스트 `data/splits/aihub_yolo_v1.csv`.

## 0. 요약

| 항목 | 결정·계획 |
|---|---|
| 맥락 출처(U2-a) | **(c1) 확대 영상 안 창 이동**(모든 클래스) + **(a) 겹친 crop 이어 붙이기**(v1 라벨, 같은 line·같은 gain·같은 fold의 train_pool crop만). (b) 원 세션 B-scan은 AIHub zip에 없다(curator: B-scan 74,838개 모두 256×256 crop) |
| 새 단계 | `YoloContextScale`: 맥락 띠 조립 → 등방 축척 → 256 창 배치(맥락 모드는 실제 내용 안에서 옮김, 기본 모드는 중앙). N2-a의 `YoloIsotropicScale`을 대체한다 |
| 확률(U2-b) | 맥락 모드 0.5(축척 U[1.0, 1.5], 창 이동 U[가능 범위 ∩ ±96]) / 기본 모드 0.5(축척 U[0.55, 1.5], 중앙). 채움 이동 0.15(±96), 채움 띠 0.15(좌 0.25 / 우 0.25 / 아래 0.5). §2-3 근거 |
| 그대로 | 학습 고정 설정(scratch, seed 0, 50 epoch, U2-d), 가려진 박스 규칙, A1 문턱(76.8 / 90.9), A2 정의, D1 r4 규칙 = [T] v0.5.2 그대로, 블라인드, 감사 ①–④ |
| 학습 밖 시험(U2-c) | R1(겹친 crop 쌍의 실제 위치 차)만 보고 전용으로 한다. R2(원 세션)는 원본이 없어 하지 않는다 |
| 공개할 한계 | (c1)은 D1 X6와 같은 종류의 변환이다. 그래서 X6 개선은 "학습에서 본 변환에 대한 강건성"에 가깝다. 쌍곡선은 사실상 (c1)에만 의존하므로 실제 맥락 이동 폭이 확대율에 묶인다(\|dx\| ≤ 128 (s − 1) ≤ 64 px) |
| 누수 | 이웃은 같은 line·gain·fold·subset=trainpool이어야 하고, 학습 fold여야 한다. 아니면 `ValueError`. 현재 인덱스에서 fold·test를 넘는 그룹은 0개다(curator) |

## 1. 맥락 가용성 (train_pool, curator 집계 + 인덱스)

| 클래스 | 한쪽 ≥ 64 px 맥락(겹친 crop) | 사슬 맥락이 조금이라도 있는 파일 / 전체 파일 | 인접 쌍(같은 클래스, train_pool) |
|---|---|---|---|
| pipe_band | 93.2 % | 3,342 / 3,576 | 2,787 |
| pipe_hyperbola | 8.3 % | 336 / 1,357 | 166 |
| manhole | 2.3 % | 221 / 4,934 | 112 |
| cavity | 0.3 % | 20 / 4,934 | 10 |

- 인접 쌍 stride 중앙값은 104 px, 겹침 폭 중앙값은 152 px다. 같은 세션·채널이라도 gain 사본끼리는 x 시작이 다를 수 있다. 그래서 다른 gain은 잇지 않는다.
- 결론: (a)는 사실상 **pipe_band 전용**이다. 쌍곡선·공동·맨홀은 맥락 모드에서 (c1)(확대 여유)에만 의존한다.

## 2. 증강 설계

### 2-1. 단계 순서 (학습 transforms, Ultralytics 8.4.192)

`YoloGprAugment`(강도 0) → **`YoloContextScale`**(신규, N2-a `YoloIsotropicScale` 자리, transforms 인덱스 1) → Ultralytics 전처리(`RandomPerspective`: translate 0.1, scale 0) → 좌우 반전 0.5 → `YoloGeometricAugment`(채움 이동 → 채움 띠, 확률만 변경) → `Format`.

### 2-2. `YoloContextScale` 동작 (영상 1장당)

1. **모드 선택**: 확률 `p_context` = 0.5로 맥락 모드, 아니면 기본 모드.
2. **맥락 띠 조립**(맥락 모드만, §3): 이 crop과 같은 line·gain·fold·train_pool인 겹친 이웃으로 원본 x 구간 [a − E, b + E] 안의 실제 열을 모은다(E = 128 px). 이웃이 없거나 쓸 수 없으면 띠 = crop 자체(폭 256)다.
3. **축척 s**: 맥락 모드 s ~ U[1.0, 1.5], 기본 모드 s ~ U[0.55, 1.5]. 보간은 N2-a와 같다(축소 `INTER_AREA`, 확대 `INTER_LINEAR`, s = 1이면 리샘플 없음). 띠 전체에 같은 s를 적용한다.
4. **창 배치**: 출력은 256 × 256이고, 위 행 기준(off_y = 0)이다.
   - 기본 모드: 띠 안 crop 중심을 출력 중심에 둔다(dx = 0). 축소로 빈 곳은 128.
   - 맥락 모드: 가능 범위 [dx_min, dx_max]는 창(256 px)이 **실제 내용(축척된 띠) 안에 완전히 들어가는** dx 범위다. 그 범위와 [−96, 96]의 교집합에서 dx를 균등 정수로 뽑는다. 교집합이 {0}이면 dx = 0이다.
   - 부호 규약: dx > 0이면 창이 왼쪽으로 가서 객체가 출력에서 오른쪽으로 dx만큼 이동한다.
5. **라벨**: 띠 라벨(§3-2)을 같은 사상으로 옮긴 뒤, 창으로 잘리면 [P] §1-1-a 규칙을 적용한다(보이는 넓이 ≥ 50 %, 잘린 박스만 변 ≥ 4 px, 단계 전 프레임 자르기).
6. **기록**: 모드, s, dx, 가능 범위, 사용한 이웃 수, 맥락 출처(none / c1 / a)를 `aug_stats`에 남긴다.

- (c1)은 따로 구현하지 않는다. 이웃이 없을 때 맥락 모드의 가능 범위가 ±128 (s − 1)이 되는 것이 곧 (c1)이다.

### 2-3. 확률 근거 (U2-b, cv 결정)

- **맥락 모드 0.5**: 실제 측선 안쪽 타일에서는 객체가 중앙을 벗어나고 옆이 실제 자료다. 그런데 쌍곡선은 맥락 모드에서 확대 여유에만 기대므로, 실제 이동 폭은 \|dx\| ≤ 128 (s − 1)이다.
  - s ~ U[1.0, 1.5]에서 가능 반폭 r은 U[0, 64]다. dx ~ U[−r, r]이면 \|dx\| ≥ 32가 될 확률은 (1/64)∫₃₂⁶⁴(1 − 32/r)dr = 0.153이다.
  - 따라서 쌍곡선 학습 영상 중 실제 맥락에서 32 px 이상 벗어난 것은 0.5 × 0.153 ≈ 7.7 %다. 64 px는 거의 없다.
  - 이 비율을 더 올리려면 맥락 모드를 늘려야 하는데, 그만큼 축소(s < 1) 학습이 줄어든다. 축소는 4TU 작은 쌍곡선을 위한 것이다(N2-a r3에서 0.5배 재현율 0.4 → 68 %).
  - 그래서 반반으로 둔다. 축소 비율은 0.5 × 0.474 = 23.7 %다(N2-a 47.4 %).
- **채움 이동 0.15, 채움 띠 0.15**: 측선 끝·짧은 측선·C′ 아래 채움은 실제로 있다. 그러나 "이탈 = 채움" 상관을 약하게 하려고 N2-a(0.5, 0.3)보다 낮춘다.
  - 채움 띠는 아래 비중을 0.5로 올린다. C′ 아래 채움은 4TU 타일의 37–69 %라 모든 타일에 있고, 좌우 채움은 측선 끝 타일에만 있기 때문이다.
- **띠(pipe_band)**: 맥락 모드에서 93 %가 (a) 맥락을 가져 ±96 이동을 대부분 받는다.

### 2-4. 다른 증강·고정 설정 (U2-d와 [P] §1-2 그대로)

scratch(`pretrained: false`), seed 0, `augment_seed` 0, 50 epoch, batch 64, imgsz 256, AdamW lr0 0.00125·momentum 0.9 등 고정값, translate 0.1, Ultralytics scale 0, fliplr 0.5, **flipud·degrees·shear·perspective 0**, mosaic 0, GPR 증강 강도 0, `val: true`, workers 2, 매니페스트·fold 정의, 라벨 v1(`data/aihub_yolo_v1`).
- band v2 자동 병합 라벨(`aihub_yolo_v2_auto32`)은 쓰지 않는다. M1-10·N2-a와 같은 라벨이어야 A1·D1을 비교할 수 있다.

## 3. (a) 이어 붙이기 규칙

### 3-1. 이웃 선택과 밝기 보정

- **후보**: `crops.csv`에서 중심 crop과 `line`·`gain`·`fold`·`subset`이 같은 행이다(`subset == "trainpool"`). 그중 x 구간이 [a − E, b + E]와 겹치는 것을 고른다.
- **사슬**: 중심에서 바깥으로, 이미 받아들인 crop과 **겹치는(맞닿기만 한 것 제외)** 이웃만 잇는다. 겹침 폭 < 16 px이면 쓰지 않는다.
- **밝기 보정**: 이웃 N을 이미 받아들인 기준 crop R(처음은 중심 C)에 맞춘다.
  - 겹침 열 O(원본 x 기준, 모든 행)에서 R ≈ α·N + β를 최소제곱으로 푼다.
  - α ∈ [0.8, 1.25]를 벗어나거나, 보정 전 상관 corr(N_O, R_O) < 0.98이면 그 이웃과 그 너머 사슬을 버린다.
  - 보정값은 clip(round(α·N + β), 0, 255) uint8이다.
  - 근거: curator·cv 사전 조사에서 같은 gain 겹침 상관 p50은 0.9965였고, 평균 밝기 차 p90은 15.9였다.
- **픽셀 우선순위**: 겹친 열은 **이미 받아들인 쪽(중심에 가까운 crop)의 픽셀**을 쓴다. 섞지 않는다. 중심 crop 열 [a, b]는 언제나 원래 픽셀 그대로다.
- **gain**: 이웃은 중심과 같은 gain만 쓴다. gain 사본끼리는 x 시작이 다를 수 있어 섞지 않는다.

### 3-2. 라벨 합침

- 모든 박스를 띠 좌표(원본 x)로 옮긴다.
- **pipe_band 이외**(쌍곡선·공동·맨홀), v0.3 확정(§7-5 (1))
  1. **중심 crop 박스는 항상 남긴다**(보로노이로 지우지 않음). 중심 crop의 라벨이 평가 영상과 같은 라벨이고, 지우면 그 객체가 배경으로 학습될 위험이 크기 때문이다.
  2. 이웃 박스는 중심이 중심 crop 열 [a, b) 안이면 버린다(그 열은 중심 라벨이 정본). 나머지 이웃 박스는 이웃 crop 중심끼리의 보로노이 분할로 소유를 정해, 자기 crop 소유 구간에 중심이 있는 것만 남긴다.
  3. 중복 정리(같은 클래스), 순서대로 적용한다.
     - (가) 포함: IoMin ≥ 0.7 **그리고** 넓이 비 ≥ 1.25이면 **넓은 쪽**을 남긴다.
     - (나) 그 밖에 IoU ≥ 0.5이면 중심 crop 박스, 그다음 중심에 가까운 crop의 박스를 남긴다.
     - (가)는 중심 crop 가장자리에서 잘려 그려진 박스가, 띠에서 온전히 보이는 같은 객체의 이웃 박스로 바뀌는 경우다. 중심 박스는 지워지지 않고 같은 객체의 온전한 박스로 대체된다.
- **pipe_band**(curator 위험: 겹침 구간 x 커버리지 XOR 중앙값 0.41로, crop마다 띠를 다르게 그림)
  - 중심 crop 열 [a, b] 안에서는 **중심 crop의 band 박스만** 쓴다(평가 영상과 같은 라벨 규약).
  - [a, b] 밖의 맥락 열에서는 이웃의 band 박스를 그 열 범위로만 잘라 쓴다. 세로 $y_1$·$y_2$는 바꾸지 않는다. 바깥 이웃끼리 겹치면 중심에 가까운 이웃이 우선한다.
  - 결과로 띠가 a 또는 b에서 조각으로 맞닿을 수 있다. 이것은 v1의 조각 라벨 규약과 같은 종류라 그대로 둔다.
- **창 자르기 뒤** [P] §1-1-a 규칙을 적용한다.
- **한계**: AIHub crop 라벨이 crop 안의 모든 객체를 덮는지는 확인되지 않았다(crop은 객체 중심으로 잘림). 이웃 맥락에 라벨 없는 객체가 보일 수 있다. 중심 crop 자체에도 같은 위험이 있으므로 새로 생기는 위험은 아니지만, 맥락 영역이 넓어 노출이 커진다.

## 4. 누수·안전 단언 (구현이 매 영상 또는 시작 시 검사)

- **맥락 인덱스 고정값(v1.0 봉인)**: `data/aihub_context_index/crops.csv` sha256 **e10192bfc42c8781769acb81556225996f0b069744b91179a787cc5fe923bbbf**, `groups.csv` sha256 **f1fc261d18eb91d0bfdc6fdbf252a97d0e80aa2ece4ac2a8764c9296a9a38570**(2026-11-14 `Get-FileHash`). 학습 기록의 인덱스 sha256이 이 값과 다르면 A1 판정을 하지 않는다(판정자 손 확인, §5).

- 시작 시
  - `crops.csv`·`groups.csv` sha256을 기록한다.
  - `crops.csv`의 (line → fold, subset)이 1:1인지 단언한다(curator: line은 세션과 1:1).
  - 학습 fold 목록(`fold<k>.yaml`의 train 리스트)에 들어 있는 crop만 맥락 후보로 둔다.
- 이웃마다
  - `subset == "trainpool"`이 아니면(예: `test`) `ValueError`.
  - `fold`가 중심과 다르거나, 이번 학습의 검증 fold이면 `ValueError`.
  - `line`·`gain`이 중심과 다르면 `ValueError`.
- 결과: 검증 fold·test 세션의 영상이 학습 증강에 들어가지 않는다. 인덱스상 fold·test를 넘는 그룹은 0개이므로 정상이면 단언은 걸리지 않는다. 걸리면 학습을 멈춘다.

## 5. 평가와 판정 (사전 등록)

- **A1**(AIHub 원 crop, [P] §3-1 그대로): crop 단위 `last.pt` 5 fold 평균, 전체 mAP50 ≥ 76.8 그리고 pipe_hyperbola AP50 ≥ 90.9, 반올림 없음. 게이트는 `n2a_a1_judge.py`와 같다(설정 봉인·`args.yaml` 단언에 `gpr_context` 블록 sha256 추가).
  - A1 게이트 추가 줄: `context_index_same_in_all_folds`(5 fold 학습 기록의 `gpr_context_index.file_sha256`이 모두 같음, 코드에 이미 있음).
  - A1 판정서 손 확인 항목: 학습 기록의 맥락 인덱스 sha256 = §4의 고정값(`crops.csv` e10192bf…bbbf, `groups.csv` f1fc261d…8570). 판정 스크립트는 fold 사이 동일성만 보므로, 고정값 대조는 판정자가 손으로 한다.
- **A2**: D1 r4에서 규칙 O가 "둘 다 불충족"이 아니다.
- **D1 r4**: [T] v0.5.2 §8-2(봉인 3c81aca, 규칙 blob eacf9126…)를 그대로 쓴다. 변형 20개, P·O, 문턱 3.0 %p, 집계 순서, G1–G3, 채움 128. 입력 부록(가중치 sha·G1 입력 sha)만 A1 뒤 새로 봉인한다.
- **채택** = A1 그리고 A2. 채택되면 U-c대로 기준선을 교체하고 U-3 절대값은 유지한다(사용자 승인 뒤). 불채택이면 M1-10이 기준선으로 남는다.
- **학습 밖 시험(U2-c, 보고 전용)**
  - **R1**: 검증 fold에서 `crops.csv` 순서로 구성한 이웃 쌍(같은 line·gain, 대표 gain만; `adjacent_pairs.csv`는 집계 참고용) 가운데 같은 객체가 두 crop에 모두 온전히 들어 있는 쌍을 쓴다(겹침 구간 안, 같은 gain, crop 단위 대표 gain). 두 crop에서 그 객체의 가로 위치(crop 중심 대비 오프셋)가 다르다.
    - 객체 단위 재현율(conf > 0.25, IoU ≥ 0.5)을 "중앙에 가까운 쪽" 대 "먼 쪽"으로 짝지어, 오프셋 구간(0–32, 32–64, 64–96, ≥ 96 px)별로 낸다.
    - 변형(리샘플·채움)이 전혀 없는 실제 맥락이라 학습 변환과 독립이다.
    - 표본: train_pool 인접 쌍 중 쌍곡선–쌍곡선 166, 띠–띠 2,787, 맨홀 112, 공동 10. fold당 쌍곡선은 수십 개라 불확실성이 크고, 이 점을 함께 적는다.
    - N2-a′와 함께 M1-10·N2-a 모델에도 같은 R1을 돌려 비교한다.
  - **R2**(원 세션): 원본이 없어 하지 않는다. 실측 일반화의 최종 확인은 M3-3·D3(4TU)다.
- **공개할 한계(판정서에 그대로)**: 맥락 모드의 (c1)은 D1 X6(1.478배 확대 + 창 이동)와 같은 종류의 변환이다. 그래서 D1 r4의 X6 개선은 "학습에서 본 변환에 대한 강건성"을 재는 쪽에 가깝다. R1과 M3-3·D3로 따로 확인한다. 쌍곡선의 실제 맥락 이동 폭은 128 (s − 1) ≤ 64 px로 제한된다.

## 6. 블라인드·감사·자원

- **블라인드**: [P] §4-2 그대로다. 모든 단계는 `--blind`이고, 메인 세션은 `.done`·자원만 본다. 판정자만 verdict·metrics를 연다. 결과를 본 뒤 설정·규칙을 바꾸는 것은 금지다(바꾸려면 새 사전 등록).
- **감사**: [P] §6과 같다.
  - ① 실행 전(봉인 뒤, 학습 전) → 스모크 → 5-fold·평가·내보내기 → A1 판정 → ② 판정 뒤
  - D1 r4 부록 봉인 → ③ 실행 전 → D1 r4 + R1 → A2·채택 판정 → ④ 판정 뒤
- **러너 마스크**: 각 러너는 **실행 직전** `~/.claude/tools/cpu-scheduler/status.py`로 다른 세션 예약과 부하를 보고, **빈 코어만** affinity로 준다. N2-a 배치표의 마스크를 재사용하지 않는다(r3 공개 사항 재발 방지). GPU 작업이라 compute-scheduler-expert 배치표를 실행 시점에 받고, `reserve.py`로 예약한다.
- **HEAD 확인**: 학습 fold마다, 평가·내보내기·판정 단계 앞마다 HEAD = 봉인 커밋, 추적 파일 변경 0을 단언한다(N2-a 학습 중 다른 커밋이 끼어든 일 재발 방지). 장시간 러너가 도는 동안에는 저장소에 커밋하지 않는다(메인 세션 운영 규칙).

## 7. python-expert 구현 명세

### 7-0. 구현 결과 반영 (v0.3, python-expert 구현, 새 시험 39개·전체 1,355개 통과, N2-a 설정으로 돌리면 증강 결과가 HEAD와 바이트 동일)

명세와 다르게 구현한 곳(python-expert 보고 원문, 메인 세션 전달 그대로)과 판단이다. 봉인 판에서는 이 표가 §7-1–§7-2 명세보다 우선한다.

| # | 구현(보고 원문) | 판단 |
|---|---|---|
| 1 | 함수 형태: ContextStrip·StripPiece dataclass, place_window 반환 4개, fit_brightness NamedTuple | 승인(뜻 같음) |
| 2 | ContextIndex 키를 (line, gain)으로 묶고, 단언을 먼저 한 뒤 거르며, 시작할 때 단언 2개를 더함 | 승인. fold·subset 단언이 거르기 전에 걸리므로 §4보다 엄격하다 |
| 3 | gpr_context가 있으면 geometric.scale을 꺼진 것으로 해석하고, 켜져 있으면 ValueError | 승인(C-19와 같은 뜻) |
| 4 | C-2 float 검증, C-4 float 2배 경로, C-18 seed 2 | 승인. 시험 입력 형식의 차이일 뿐 기대값은 같다 |
| 5 | R1 이웃 쌍은 crops.csv 순서로 정하고 대표 gain만 씀 | 승인. 대표 gain은 crop 단위 평가와 같다(1.0 > 0.5 > 0.25) |
| 6 | 기록 항목 정의(none/c1/a, dx_span_px, overlap 사유) | 승인. `aug_stats` 보고용 |

- 메인 세션 전달은 "9건"이었으나 원문 항목은 위 6줄로 왔다(한 줄에 여러 건이 묶임). 각 줄 단위로 승인했다.
- **실데이터 점검**(fold 0 학습, 맥락 모드 강제, 보고용): 이웃을 1개 이상 이은 비율은 band 91.3 %, hyperbola 26.0 %, manhole 4.4 %, cavity 0.2 %다. N2-a′ 설정으로 fold 0–4 `--dry-run`이 모두 종료 0, 단언 0건이다.

### 7-5. 확인 필요 4건 결정 (v0.3, 봉인 전)

1. **중심 crop 박스의 보로노이 예외 — 예외로 둔다(항상 유지).** 이웃이 그 객체를 라벨하지 않았으면 라벨이 사라져 배경으로 학습되므로 위험이 크다. 규칙은 §3-2(v0.3)다.
   - **python-expert 수정 명세**: `merge_strip_labels`
     - (i) 중심 crop 박스는 소유 판정에서 빼고 모두 남긴다.
     - (ii) 이웃 박스 중 중심 x ∈ [a, b)인 것은 버린다.
     - (iii) 나머지 이웃 박스는 이웃 crop 중심끼리의 보로노이로 거른다.
     - (iv) 같은 클래스 중복: 먼저 IoMin ≥ 0.7이고 넓이 비 ≥ 1.25이면 넓은 박스를 남긴다. 그 밖에 IoU ≥ 0.5이면 중심 → 가까운 crop 순으로 하나를 남긴다. 비슷한 크기의 같은 객체 박스(C-12, IoU 0.9)는 (나)로 중심 박스가 남고, 잘려 그려진 중심 박스(C-23, 넓이 비 2)는 (가)로 넓은 이웃 박스로 바뀐다.
     - 시험 C-22–C-24를 추가하고, C-12는 그대로 통과해야 한다.
2. **C-21 정의 — 겹침 ≥ 16 px 기준, fold 0 학습 집합, 정확값 306 / 1,172(26.11 %).** 구현이 실제로 잇는 조건(겹침 ≥ 16 px, 맞닿기 제외)과 같고, 학습이 실제로 쓰는 모집단이다. 맞닿기 포함 trainpool 전체 값(24.8 %)은 §1 표의 참고값으로만 둔다. 밝기 검사 뒤 비율은 단언하지 않는다(C-21 행 수정).
3. **밝기 기준 — 승인.** 기준은 `YoloGprAugment` 뒤 중심 픽셀(강도 0이면 원본과 같음)이고, 이웃은 바로 앞 고리의 보정된 픽셀에 맞춘다.
   - 조건 추가: 강도 > 0인 GPR 증강은 중심에만 잡음을 넣어 기준을 흐리게 하므로 `gpr_context`와 함께 쓰지 못하게 한다.
   - **python-expert 수정 명세**: 설정 검사에서 `gpr_context`가 켜져 있고 `gpr_augment.strength > 0`이면 `ValueError`. 시험 C-25를 추가한다.
4. **이웃 선택·keep 기준 — 승인.**
   - 다음 이웃은 가장 가까운 것부터 고른다.
   - 띠 끝(b + E)에서 잘린 다른 클래스 박스는 **원래(자르기 전) 넓이**를 기준으로 keep 규칙을 적용한다. 보이지 않게 된 것이기 때문이다.
   - band 조각은 소유 열로 자른 박스를 기준으로 한다. 열 자르기는 중재이지 가림이 아니기 때문이다. 이 기준 박스가 창 자르기에서 다시 가려질 때만 50 % 규칙이 걸린다.

### 7-1. 모듈

- `src/gpr_detection/augment_geometry.py`에 넣을 것(cv2 허용 모듈, [P] §5-0 #1)
  - `ContextIndex`: `crops.csv`를 읽고 (line, gain, fold) → 정렬된 (a, b, file_name) 목록을 만든다. 단언은 §4.
  - `assemble_context_strip(center_file, image_loader, index, extend_px=128, min_overlap_px=16, corr_min=0.98, alpha_range=(0.8, 1.25)) -> (strip uint8 [H, W_s], strip_x0, boxes_strip, classes, used_neighbors)`. 밝기 보정·픽셀 우선순위·라벨 합침은 §3이다. `image_loader`·`label_loader`는 주입한다(시험용).
  - `fit_brightness(ref_overlap, nb_overlap) -> (alpha, beta, corr, ok)`
  - `merge_strip_labels(crops_boxes, centers, center_index) -> (boxes, classes)`: §3-2 보로노이 + band 규칙.
  - `place_window(strip, boxes, s, mode, rng, max_shift_px=96, fill=128) -> (image256, boxes, record)`: 축척·창 배치·가능 범위, §2-2.
- `src/gpr_detection/detect/yolo_train.py`
  - `YoloContextScale(config, index, seed)`: `YoloIsotropicScale` 자리(인덱스 1)에 들어간다.
  - Ultralytics labels의 `im_file`로 crop을 찾는다. 이웃 영상·라벨은 데이터셋 폴더에서 읽는다(`cv2.IMREAD_GRAYSCALE`, 라벨 txt).
  - 피클·워커 RNG는 기존과 같은 방식이다(SeedSequence 스트림 k = 3).
  - `aug_stats`에 모드·출처·s·dx·이웃 수를 추가한다.
- 설정 블록 `gpr_context:` 예시

```yaml
gpr_context:            # N2-a' (docs/experiments/m31_n2a2_retrain_plan.md)
  index_dir: data/aihub_context_index
  p_context: 0.5
  context_scale: [1.0, 1.5]
  plain_scale: [0.55, 1.5]
  max_shift_px: 96
  extend_px: 128
  min_overlap_px: 16
  corr_min: 0.98
  alpha_range: [0.8, 1.25]
  fill: 128
gpr_geometric:
  horizontal_shift: {probability: 0.15, max_shift_px: 96, fill: 128}
  edge_fill: {probability: 0.15, side_weights: [0.25, 0.25, 0.50],
              width_min_px: 8, width_max_px: 96, height_min_px: 16, height_max_px: 128, fill: 128}
  keep_rule: {min_visible_area_fraction: 0.5, min_side_px: 4}
# gpr_geometric.scale is removed: YoloContextScale owns isotropic scaling (assert both are not enabled)
```

- 설정 파일 `configs/train_yolo11m_n2a2.yaml`(N2-a 사본에서 `gpr_context` 추가, `gpr_geometric.scale` 제거, 확률 변경). 단언: `gpr_context`가 켜져 있으면 `gpr_geometric.scale`이 없어야 하고, Ultralytics `scale`은 0이어야 한다.
- `analysis/detection/m31_r1_overlap_shift.py`(신규, R1)
  - 입력: `--run-prefix`, `--folds`, `--device`, `--out`, `--blind`.
  - 검증 fold `adjacent_pairs`에서 두 crop에 모두 온전히 들어 있는 객체를 짝지어, 오프셋 구간별 짝지은 재현율을 낸다.
  - 원 crop을 그대로 넣는다(`YoloDetector`, conf 0.001 → 0.25 기준).
  - stdout은 status만 쓰고, 덮어쓰기는 거부한다. `run_identity`를 남긴다.

### 7-2. 시험 기대값 (`tests/test_context_augment.py`, CPU)

합성 영상: 값 = (원본 x 좌표 mod 251)인 세로 줄무늬. 그래서 이어 붙인 띠의 열 값으로 원본 x를 확인할 수 있다.

| ID | 내용 | 기대 |
|---|---|---|
| C-1 | 중심 (1000, 1256), 이웃 (1104, 1360)(겹침 152), 이웃 = 중심과 같은 원본 | 띠 x 범위 [1000, 1360), 폭 360, 열 j의 값 = (1000 + j) mod 251. 중심 열은 중심 픽셀 그대로 |
| C-2 | 밝기: 이웃 = clip(0.9·원본 + 10) | α ≈ 1/0.9, β ≈ −11.11(\|오차\| < 1e-3). 보정 뒤 겹침 MAD < 1 |
| C-3 | 상관 미달: 이웃 겹침에 잡음을 넣어 corr 0.95 | 이웃 버림, 띠 = 중심만(폭 256) |
| C-4 | α 범위 밖: 이웃 = 2·원본 | 버림 |
| C-5 | 맞닿기만 함: 이웃 (1256, 1512) | 버림(겹침 0) |
| C-6 | 겹침 10 px | 버림(< 16) |
| C-7 | 사슬: (1000,1256)–(1104,1360)–(1208,1464), E = 128 | 띠 [1000, 1384)(b + E = 1384에서 자름). 세 번째 crop은 둘째에 맞춰 보정 |
| C-8 | 누수: 이웃 fold ≠ 중심 fold | `ValueError` |
| C-9 | 누수: 이웃 subset = test | `ValueError` |
| C-10 | 누수: 이웃이 이번 학습의 검증 fold | `ValueError` |
| C-11 | gain 다름 | 후보에서 제외(단언이 아니라 선택 단계에서 거름) |
| C-12 | 라벨(쌍곡선): 중심 박스 중심 x 1128, 이웃 박스 중심 x 1130(같은 객체, IoU 0.9) | 하나만 남음(중심 crop 것) |
| C-13 | 라벨(band): 중심 band (1050, 70, 1200, 86), 이웃 band (1150, 72, 1340, 88) | 결과: 중심 band 그대로 + 이웃 band를 [1256, 1340)으로 자른 (1256, 72, 1340, 88). y는 입력 그대로 |
| C-14 | 창(기본 모드) s = 0.55, 이웃 없음 | N2-a S-1과 같은 결과(박스 (112.078125, 27.5390625, 139.6171875, 44.0625)) |
| C-15 | 창(맥락 모드) s = 1.25, 이웃 없음 | 가능 범위 dx ∈ [−32, 32]. 10,000회에서 모두 범위 안이고 균등(카이제곱 p > 0.01). 출력에 128 채움 없음 |
| C-16 | 창(맥락 모드) s = 1.0, 이웃 없음 | dx = 0, 출력 = 원본 |
| C-17 | 창(맥락 모드) s = 1.0, 띠 폭 360(C-1) | 가능 범위 dx ∈ [−104, 0] ∩ [−96, 96] = [−96, 0](객체가 왼쪽으로만 감). 채움 없음 |
| C-18 | 모드 비율 | 10,000회에서 맥락 모드 0.5 ± 0.02, s 분포가 각 모드 범위에서 균등 |
| C-19 | 이중 축척 방지 | `gpr_context`와 `gpr_geometric.scale`이 함께 켜지면 `ValueError` |
| C-20 | 삽입 위치·피클 | `YoloContextScale`이 인덱스 1에 하나만 있음. 피클 뒤 같은 seed 같은 결과 |
| C-21 | 실제 인덱스 스모크(v0.3 정의) | `crops.csv`로 fold 0 학습 집합(fold 1–4 train_pool, pipe_hyperbola 파일 1,172개)을 구성하면 단언 0건. **인덱스만으로 센** "같은 line·gain·fold에서 겹침 ≥ 16 px인 이웃이 1개 이상인 쌍곡선 파일"이 **정확히 306개(26.11 %)**. 밝기 검사(상관·α) 뒤 실제로 이은 비율(실데이터 점검 26.0 %)은 보고만 하고 단언하지 않는다 |
| C-22 | (v0.3) 중심 박스 유지 | 중심 쌍곡선 박스 중심 x 1250(중심 crop 열 끝 근처), 이웃 (1104, 1360)에 그 객체 라벨 없음 | 중심 박스가 남는다(보로노이로 지워지지 않음) |
| C-23 | (v0.3) 중심 가장자리 잘린 박스 교체 | 중심 박스 (1236, 50, 1256, 80)(잘려 그림), 이웃 박스 (1236, 50, 1276, 80)(같은 객체 온전, 중심 x 1256 ∉ [1000, 1256)) | IoMin 1.0, 넓이 비 2.0 → (가)로 넓은 이웃 박스 하나만 남는다(IoU 0.5 규칙보다 먼저) |
| C-24 | (v0.3) 이웃 박스가 중심 열 안 | 이웃 박스 중심 x 1200 ∈ [1000, 1256) | 버린다(중심 라벨이 정본) |
| C-25 | (v0.3) GPR 증강 강도 > 0과 `gpr_context` 동시 사용 | `ValueError` |
| R-1 | R1 짝짓기 | 합성 쌍(같은 객체, 오프셋 0과 80 px)에서 구간 0–32와 64–96에 각 1개, 겹침 밖 객체는 제외 |

### 7-3. 명령 초안 (메인 세션 러너, PowerShell, 작업 폴더 `C:\workspace\gpr_detection`, `<D>` 실행일, `<SEAL>` 봉인 커밋)

```powershell
$py = "C:\ProgramData\miniforge3\Scripts\conda.exe"
# smoke (fold 0, 2 epochs, metrics not opened)
& $py run -n gpr-dev --no-capture-output python -I scripts/train_detector.py --config configs/train_yolo11m_n2a2.yaml --fold 0 --epochs 2 --device 0 --name yolo11m_aihubv1n2a2_smoke_fold0 --blind
# 5-fold (k = 0..4), each preceded by a HEAD check step
& $py run -n gpr-dev --no-capture-output python -I scripts/train_detector.py --config configs/train_yolo11m_n2a2.yaml --fold <k> --device 0 --name yolo11m_aihubv1n2a2_<D>_fold<k> --blind
& $py run -n gpr-dev --no-capture-output python -I scripts/eval_detector.py --run-dir runs/detect/yolo11m_aihubv1n2a2_<D>_fold<k> --confusion-conf 0.25 --device 0 --blind
& $py run -n gpr-dev --no-capture-output python -I analysis/detection/export_val_predictions.py --device 0 --weights runs/detect/yolo11m_aihubv1n2a2_<D>_fold<k>/weights/last.pt --data-yaml runs/detect/yolo11m_aihubv1n2a2_<D>_fold<k>/eval/last_crop/eval_data.yaml --out-dir runs/analysis/n2a2_val_export/fold<k>_crop --blind
# A1
& $py run -n gpr-dev --no-capture-output python -I analysis/detection/n2a_a1_judge.py --config configs/train_yolo11m_n2a2.yaml --run-prefix yolo11m_aihubv1n2a2_<D>_fold --seal-commit <SEAL> --out analysis/detection/n2a2_a1_verdict.json --blind
# D1 r4 (after the appendix seal and audit 3)
& $py run -n gpr-dev --no-capture-output python -I analysis/detection/m31_d1_shift_scale.py --run-prefix yolo11m_aihubv1n2a2_<D>_fold --export-root runs/analysis/n2a2_val_export --folds 0,1,2,3,4 --device 0 --fill 128 --save-examples --out-subdir m31_d1_r4 --summary analysis/detection/m31_d1_r4_summary.json
& $py run -n gpr-dev --no-capture-output python -I analysis/detection/m31_d1_judge.py --run-prefix yolo11m_aihubv1n2a2_<D>_fold --export-root runs/analysis/n2a2_val_export --expected-weights-sha8 <SHA8> --subdir m31_d1_r4 --out analysis/detection/m31_d1_r4_verdict.json
# R1 (report only), also for M1-10 and N2-a prefixes
& $py run -n gpr-dev --no-capture-output python -I analysis/detection/m31_r1_overlap_shift.py --run-prefix yolo11m_aihubv1n2a2_<D>_fold --folds 0,1,2,3,4 --device 0 --out analysis/detection/m31_r1_n2a2.json --blind
```

- `n2a_a1_judge.py`가 새 설정(`gpr_context` 블록)의 args 단언을 처리하는지 python-expert가 확인한다. 필요하면 단언 목록에 `gpr_context` sha256을 더한다.

### 7-4. 예상 시간·자원

| 단계 | 추정 | 근거 |
|---|---|---|
| 학습 5 fold | 2.5–3.5 h | N2-a 1,489–1,670 s/fold. 맥락 모드에서 이웃 JPG를 더 읽는다(band 위주, 영상당 평균 1장 미만 추가) |
| 평가·내보내기·A1 | 약 0.5 h | N2-a 실측 |
| D1 r4 | 25–35분 | r2 1,430 s, r3 2,005 s(CPU 겹침) |
| R1(모델 3개) | 수 분 | 검증 fold 인접 쌍만 |
| VRAM | 약 7.4 GB(학습), 2.5 GB 이하(D1) | N2-a 실측 |

## 8. 사용자 결정 (기록)

| ID | 결정(2026-11-13) |
|---|---|
| U2-a | (c1) 확대 영상 안 창 이동(모든 클래스) + (a) 겹친 crop 이어 붙이기(v1 안, 같은 line·gain만, v1 밖·다른 gain 이웃 금지) |
| U2-b | 확률은 cv가 정한다 → 맥락 모드 0.5, 채움 이동 0.15, 채움 띠 0.15(§2-3 근거) |
| U2-c | 학습 밖 시험은 보고 전용. R2는 원본이 없어 하지 않고 R1만 남는다 |
| U2-d | scratch, seed 0, 50 epoch |

남은 사용자 결정은 없다. 확률 근거에 이의가 있으면 봉인 전에 고친다.

## 9. 실행 순서

1. python-expert가 §7을 구현했다. C-1–C-25·R-1과 기존 시험이 통과했다(2026-11-14, `test_context_augment.py` 43개, 전체 1,361 passed).
2. integrity-checker 봉인 전 점검을 마쳤다(Critical·High 0, 2026-11-14). 이 문서 v1.0 + 설정 + 구현을 한 커밋으로 봉인한다.
3. 감사 ①을 받는다.
4. 스모크: `train_batch*.jpg`에서 맥락 이동이 채움 없이 되는지, 이음 경계가 보이는지 확인한다. `aug_stats`의 모드·출처 비율도 본다.
5. 러너가 실행 직전 빈 코어 마스크를 정하고 5-fold → 평가 → 내보내기를 돌린다.
6. A1 판정 → 감사 ②.
7. D1 r4 부록 봉인 → 감사 ③.
8. D1 r4와 R1(M1-10·N2-a·N2-a′)을 실행한다.
9. A2·채택 판정 → 감사 ④.

## 10. 위임 필요

- **python-expert**: §7 구현·시험.
- ~~**gpr-dataset-curator**: `crops.csv`의 `cL`·`cR`(사슬 확장)과 `dL`·`dR`(직접 확장) 정의를 인덱스 README에 적기~~ → 완료. 인덱스 sha256은 §4에 고정했다.
- **compute-scheduler-expert**: 실행 시점 배치표(빈 코어만).
- **verif-protocol-auditor**: 감사 ①–④.

## 11. 공개 사항

- **절차 위반 1건(구현 단계)**: python-expert가 시험 코드를 파일에 덧붙일 때 Bash heredoc(`cat >> file <<'PYEOF'`)을 한 번 썼다. Python 실행은 아니었지만, 호출 지시의 heredoc 금지에 어긋난다. 결과 파일은 시험 통과(1,355개)와 integrity-checker 봉인 전 점검으로 확인한다. 코드 내용에 대한 영향은 없는 것으로 본다.

## 변경 이력

| 날짜 | 판 | 내용 |
|---|---|---|
| 2026-11-13 | v0.1 초안 | 맥락 출처 비교, 증강 틀, 가져올 것·바꿀 것, 학습 밖 시험, 누수, 사용자 결정 U2-a–U2-d |
| 2026-11-13 | v0.2 | 사용자 결정(U2-a: c1 + a, U2-b: cv 결정, U2-c: R1만 보고 전용, U2-d: scratch·seed 0·50 epoch)과 curator 결과((b) 없음, 맥락 인덱스, band 라벨 XOR 0.41, 밝기 차 p90 15.9) 반영. `YoloContextScale` 설계(모드 0.5/0.5, 맥락 모드 축척 U[1.0, 1.5]·가능 범위 안 창 이동), 밝기 보정·픽셀 우선순위·라벨 합침(band는 중심 열은 중심 라벨, 밖은 이웃 라벨 열 자르기), 누수 단언, A1·A2·D1 r4(v0.5.2 그대로)·R1, 블라인드·감사·빈 코어 마스크·HEAD 확인, 구현 명세·시험 C-1–C-21·R-1, 명령 초안 |
| 2026-11-14 | v0.3 | 봉인 대상 판. 구현 결과 표 §7-0(보고 원문 6줄 승인, 실데이터 점검 band 91.3·hyp 26.0·man 4.4·cav 0.2 %), 확인 필요 결정 §7-5: (1) 중심 crop 박스 보로노이 예외(항상 유지, 이웃 박스 중심 열 안 버림, IoMin ≥ 0.7이면 넓은 박스) — §3-2 수정·C-22–C-24, (2) C-21 = 겹침 ≥ 16 px·fold 0 학습 집합·정확값 306/1,172, (3) 밝기 기준 승인 + GPR 증강 강도 > 0과 `gpr_context` 동시 사용 금지(C-25), (4) 이웃 가까운 순·keep 기준(다른 클래스 원래 넓이, band 소유 열 박스) 승인. 공개 §11(heredoc 절차 위반 1건) |
| 2026-11-14 | v1.0 | 봉인 판. integrity-checker 봉인 전 점검 반영(뜻 불변): §4에 맥락 인덱스 sha256 고정(crops e10192bf…bbbf, groups f1fc261d…8570), §5 A1 게이트에 `context_index_same_in_all_folds`와 인덱스 고정값 손 확인, 머리말·§9를 구현 완료(C-22–C-25, 43개·전체 1,361 passed)와 한 커밋 봉인으로, R1 구간 '≥ 96'·쌍 구성(crops.csv 순서), §10 curator 요청 완료 처리 |
