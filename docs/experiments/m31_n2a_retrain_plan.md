# M3-1 N2-a — 위치·축척 증강 재학습 계획 (사전 등록)

- 상태: **v0.3.3** (2026-11-13, cv-detection-expert). **봉인 대상 판**이다. v0.3.3은 [SP] 인용을 v0.3.6 §11-2로 맞춘 것뿐이다. v0.3.2는 integrity-checker 봉인 전 점검(Critical·High 0)의 문서 지적만 반영했다(뜻 불변). v0.3.1은 v0.3.1은 (다) 구현 중 생긴 명세 밖 처리(단계 전 영상 프레임 자르기) 승인, G-10 시험 정정, 명령 초안 보완만 더했다. v0.3은 v0.2(사용자 결정 U-a·U-b·U-c)에 python-expert 구현 결과(§5-0)와 확인 필요 4건의 결정(§5-7)을 반영했다. 구현과 시험(1,302개 통과, 실패 0)은 끝났고, 학습·평가는 아직 없다. 봉인 커밋 → 감사 ① → 스모크 → 5-fold 순서로 진행한다(§6).
- 배경: D1 r2 판정서 [`m31_detection_input_tiling.md`](m31_detection_input_tiling.md) §8-5(이하 [T]). M1-10 YOLO11m(scratch)은 위치·축척 허용 폭이 매우 좁다.
  - 객체를 64 px 옮기면 쌍곡선 재현율(conf > 0.25)이 평균 68 %p 떨어진다.
  - 객체 옆 64 열만 회색(128)으로 덮어도 35 %p 떨어진다.
  - 축척은 0.87–1.28배에서만 유지된다(0.5배에서 약 0 %, 0.41 %).
  - 원인은 학습 분포로 본다. AIHub crop은 객체가 가로 중앙 ±12 px에 몰려 있고(쌍곡선이 crop 끝에 닿은 경우 0/503), 증강은 `translate 0.1`·`scale 0.2`뿐이었다([T] §2, §8-5 (5)).
- 관련: [증강 정책](augmentation_policy.md), [M1-10 기준선](2026-11-01_m110_baseline.md), [M1-10 설계](m110_baseline_plan.md), `configs/train_yolo11m_m110.yaml`, `src/gpr_detection/augment.py`, `src/gpr_detection/augment_geometry.py`(cv2 증강, §5-0 #1), `src/gpr_detection/detect/yolo_train.py`, `scripts/train_detector.py`, `scripts/eval_detector.py`, ADR-001, CHECKLIST U-3, [SP] = `docs/processing/spec_detection_image_v1.md`(현재 v0.3.6). 축척 하한 0.55 의견은 gpr-processing-expert 의견 2026-11-13, [SP] v0.3.6 §11-2에 기록.

## 0. 요약

| 항목 | 계획 (v0.2 고정) |
|---|---|
| 바꾸는 것 | (1) **등방 축척을 프로젝트 증강으로 옮긴다.** s ~ U[0.55, 1.5]를 한 번 뽑는다. 축소(s < 1)는 `INTER_AREA`, 확대(s > 1)는 `INTER_LINEAR`이고, 위·가로중앙 기준이며 빈 곳은 128로 채운다. Ultralytics `scale`은 0(배율 1 고정)이다(U-b). (2) **가로 이동**: 확률 0.5, dx ~ U[−96, 96] px, 빈 열은 128. (3) **가장자리 채움 띠**: 확률 0.3, 좌·우 폭 U[8, 96] px 또는 아래 높이 U[16, 128] px, 값 128. (2)(3)과 축척 확대에서 가려진 박스는 보이는 부분으로 자르고, 보이는 넓이 50 % 미만이면 라벨을 버린다 |
| 그대로 두는 것 | `translate` 0.1, 금지 증강 전부 0(`flipud`·`degrees`·`shear`·`perspective`), `mosaic` 0, GPR 증강 강도 0, seed 0, 50 epoch, 옵티마이저 고정값, imgsz 256, scratch, 1채널, 매니페스트·fold, 평가 방식(crop, `last.pt`), `val: true` |
| 규모 | **5-fold 한 번에**(U-a). 먼저 지표를 보지 않는 2 epoch 스모크. 학습 2.2–2.6 h + 평가 약 0.4 h + D1 r3 약 0.4 h |
| 채택 조건(사전 등록) | **A1** 전체 mAP50 fold 평균 ≥ 76.8 % 그리고 pipe_hyperbola AP50 fold 평균 ≥ 90.9 %. **A2** D1 r3 규칙 O가 "둘 다 불충족"이 아님. 둘 다 만족하면 **기준선 교체**, U-3 절대값은 유지(U-c) |
| 감사 | 실행 전 감사(학습 전), A1 판정 뒤 감사, D1 r3 실행 전 감사, A2·채택 판정 뒤 감사(§6) |
| 블라인드 | 학습·평가·판정 스크립트는 `--blind`로 지표를 stdout·로그에 쓰지 않는다. 메인 세션은 종료 코드와 진행 줄만 본다. 지표 파일은 판정자만 연다(§4-2) |

## 1. 재학습 설정과 근거

- **[T] §8-5 N2-a 초안에서 바뀐 점**(v0.3.2 기록): 초안은 `translate` 0.4–0.5, `scale` 0.5, 채움 띠 0–96 px였다. 이 계획은 다음처럼 바꿨다.
  - 위치: `translate`는 0.1 그대로 두고 가로 이동 ±96 px를 따로 둔다. Ultralytics `translate`는 세로에도 같은 비율로 걸려, 지표 행이 크게 움직이고 위쪽 객체가 잘리기 때문이다.
  - 축척: Ultralytics `scale`은 0으로 두고 프로젝트 등방 축척 U[0.55, 1.5]를 쓴다. 축소는 면적 평균으로 하고 하한은 0.55(주기 ≥ 2.2 px)다. 0.5배에서는 주기가 나이퀴스트에 닿고, Ultralytics 축소에는 앨리어싱 방지가 없기 때문이다(사용자 결정 U-b).
  - 띠: 좌·우 8–96 px에 아래 16–128 px를 더한다. C′ 아래쪽 채움을 재현하기 위해서다.
  - [T] §10 N2 행은 봉인 뒤 [T]를 개정할 때 고친다.

### 1-1. 바꾸는 증강

| 증강 | M1-10 | N2-a | 근거 |
|---|---|---|---|
| **등방 축척**(프로젝트, 신규) | Ultralytics `scale` 0.2(0.8–1.2배, 선형, 영상 중심 기준) | Ultralytics `scale` **0**. 프로젝트 단계에서 s ~ U[0.55, 1.5], 확률 1. 축소는 `INTER_AREA`, 확대는 `INTER_LINEAR`, s = 1이면 원본 그대로. 위·가로중앙 기준, 빈 곳 128 | D1 r2 축척 곡선에서 재현율이 0.87–1.28배 밖에서 무너졌다. 4TU 쌍곡선은 C′ $P^*$ 4.06에서 AIHub 중앙 폭 대비 0.50–1.13배다([T] §8-2 (2)). 하한 0.55와 축소 보간 방식은 §1-3 |
| **가로 이동**(프로젝트, 신규) | 없음(`translate` 0.1이 가로·세로 ±25.6 px) | 확률 0.5, dx ~ 균등 정수 U[−96, 96] px, 빈 열 128 | 실제 타일에서 객체는 소유 구간(50 % 겹침 ±64, 75 % ±32 px)과 이웃 타일 위치(±96 px 안팎)에 놓인다. Ultralytics `translate`는 가로·세로에 같은 비율로 걸린다. 그래서 크게 키우면 지표 행이 위아래로 움직이고 위쪽 객체(쌍곡선 박스 아래 끝 p99 95.6 px)가 잘린다. 가로만 따로 구현한다 |
| **가장자리 채움 띠**(프로젝트, 신규) | 없음 | 확률 0.3. 위치는 왼쪽 0.35 / 오른쪽 0.35 / 아래 0.30. 좌·우 폭 U[8, 96] px, 아래 높이 U[16, 128] px, 값 128 | D1 채움 대조에서 64 px 띠만으로 35 %p가 떨어졌다. 실제 L1에는 짧은 측선의 오른쪽 채움, C′ 아래쪽 채움(4TU 타일의 37–69 %), 측선 끝이 있다. 값 128은 [SP] `pad_gray` 가정값이다(Q3는 M3-2에서 확정) |

**1-1-a. 가려진 박스 규칙**(축척 확대·이동·채움 공통)
- 박스를 보이는 영역(캔버스 안, 채움 띠 밖)으로 자른다.
- 다음 셋을 모두 만족할 때만 라벨을 남긴다.
  - 보이는 넓이 ≥ 기준 넓이의 50 %. 기준 넓이는 그 단계 직전 박스 넓이이고, 축척 단계는 배율을 곱한 넓이다.
  - 보이는 폭 ≥ 4 px
  - 보이는 높이 ≥ 4 px
- **v0.3 수정**: 폭·높이 ≥ 4 px 조건은 **보이는 영역에 의해 잘린 박스에만** 적용한다. 잘리지 않고 축척만 된 박스는 넓이 비율 1이므로 그대로 남긴다(§5-7 (다)). **v0.3.1**: 축척·이동·채움 단계에 들어가기 전에 입력 박스를 영상 프레임 [0, W] × [0, H]로 먼저 자른다(`_clip_to_image`). 그래서 라벨 끝 결함(부동소수점 오차로 1.28 × 10⁻⁴ px 넘침)은 "잘린 박스"로 판정되지 않는다(§5-0 #8).
- 이유: Ultralytics `box_candidates`는 넓이 10 %만 남아도 라벨을 남긴다. 쌍곡선 날개 조각에 `pipe_hyperbola` 라벨을 주면 꼭짓점 없는 박스를 학습시킨다. 절반 이상 보이는 잘린 쌍곡선은 타일 경계에서 실제로 생기고, 병합 R2의 잘림 짝 맞추기([T] §5-4)에 필요하다.
- 한계: 50 % 미만 조각은 라벨 없는 배경이 된다. 이 조각들은 이웃 타일이 온전히 본다(겹침).

**1-1-b. 적용 순서**(학습 transforms, Ultralytics 8.4.192)

`YoloGprAugment`(잡음 등 4종, 강도 0) → **`YoloIsotropicScale`(프로젝트 축척)** → Ultralytics 전처리 블록(Mosaic p 0 → `RandomPerspective`: translate 0.1, scale 0, 테두리 114) → (Albumentations 없음) → `RandomHSV`(1채널 무효) → 상하 반전 p 0 → 좌우 반전 p 0.5 → **`YoloGeometricAugment`(가로 이동 → 채움 띠)** → `Format`.

- 축척을 affine 앞에 두는 이유: `INTER_AREA` 축소는 Ultralytics affine 안에서 할 수 없다. 축척 기준점(위·가로중앙)도 Ultralytics affine(영상 중심)과 다르다.
- 이동·채움 띠를 affine 뒤에 두는 이유: 띠 폭과 이동량이 최종 입력 프레임 px 단위로 정확해야 하기 때문이다.
- 이 경우 Ultralytics affine에서는 `translate`만 움직이므로, M1-10 축척과 N2-a 축척이 둘 다 적용되는 일(배율 곱)은 생기지 않는다. 구현이 이를 단언한다(§5-2).

### 1-2. 그대로 두는 설정 (M1-10과 같음)

| 항목 | 값 | 출처 |
|---|---|---|
| 모델·초기화 | `yolo11m.yaml`, `pretrained: false`(scratch), 1채널(`channels: 1`) | `configs/train_yolo11m_m110.yaml` |
| 데이터·분할 | `data/aihub_yolo_v1`, 매니페스트 `data/splits/aihub_yolo_v1.csv`(분할 seed 20261029), `fold<k>.yaml`(k = 0–4) | M1-10 §1 |
| 학습 | epochs 50, batch 64, imgsz 256, seed 0, `deterministic`, amp, patience 0, AdamW lr0 0.00125, momentum 0.9, warmup_bias_lr 0, lrf 0.01, weight_decay 0.0005, warmup_epochs 3, nbs 64, `val: true` | 같은 yaml |
| 증강 고정 | fliplr 0.5, **flipud 0**, **degrees 0**, **shear 0**, **perspective 0**, translate 0.1, mosaic 0, close_mosaic 0, mixup 0, cutmix 0, copy_paste 0, hsv_h 0, hsv_s 0, hsv_v 0.3(1채널 무효), multi_scale 0, rect false. **바뀌는 것은 `scale` 0.2 → 0 하나** | 같은 yaml, `check_augmentation_policy`가 금지 키를 단언 |
| GPR 증강 4종 | `gpr_augment.strength` 0(M1-7 결정) | 같은 yaml |
| 평가 | `scripts/eval_detector.py` 기본(imgsz 256, batch 64, conf 0.001, iou 0.7) + `--confusion-conf 0.25`, crop 단위, `last.pt`(판정), `best.pt`는 참고 | M1-10 §1 |
| workers | 2(오류 시 0으로 재시도하고 보고) | M1-10 |

- 50 epoch은 비교 조건(ADR-001 예산)이라 그대로 둔다. 증강이 세지면 수렴이 늦을 수 있다. 그러나 M1-10은 검증 최고점이 epoch 24–36이었으므로 여유가 있다고 본다(추정).
- `val: true`를 유지하는 이유: Ultralytics 8.4.192 `trainer.py` 656행은 `val`이 false여도 마지막 epoch에 검증하므로, 끄더라도 지표가 숨겨지지 않는다. 학습 곡선(과적합 진단)도 M1-10과 같은 방식으로 남는다. 블라인드는 로그 수준으로 처리한다(§4-2).
- 새 설정 파일: `configs/train_yolo11m_n2a.yaml`(M1-10 yaml 사본, §1-1과 `scale` 0만 다름, 머리에 "N2-a" 표시).

### 1-3. 축척 증강 — 곡률 단서, 하한, 보간

- **곡률**: 등방 축척은 쌍곡선 꼴을 보존한다. 점근 45°(깊이 단위 등방 L1에서 속도 단서)와 주기 단위로 잰 폭이 그대로이므로 곡률 단서는 왜곡되지 않는다. 비등방 축척만 문제다. [SP] v0.3.6 §11-2의 gpr-processing-expert 의견과 같고, 이 계획은 비등방 축척을 쓰지 않는다.
- **하한 0.55**: 0.5배면 반송파 주기(AIHub 약 4.06 px)가 2.0 px로 나이퀴스트에 닿는다. 0.55배에서는 주기가 2.2 px 이상이다(gpr-processing-expert 의견 2026-11-13, [SP] v0.3.6 §11-2에 기록). 4TU의 가장 작은 쌍곡선 일부(0.50–0.55배)는 학습 범위 밖에 남는다.
- **축소 보간(U-b = `INTER_AREA`)**
  - Ultralytics `RandomPerspective`는 `cv2.warpAffine`을 `flags` 없이 부르므로 `INTER_LINEAR`이고, 축소 전 저역 통과가 없다(설치본 8.4.192 `data/augment.py` 1212행).
  - 프로젝트 단계의 `INTER_AREA`는 D1 변형·C′ 리샘플(면적 평균)과 같은 방식이다.
  - 사전 확인(2026-11-13, CPU, 저장소 밖 임시 스크립트): 2 px 주기 세로 줄무늬(0/255)를 0.55배로 줄이면 행 표준편차가 `INTER_AREA` 11.7, `INTER_LINEAR` 73.6이다(LINEAR는 접힌 무늬가 남음). 4.06 px 반송파를 0.55배로 줄이면 `INTER_AREA` 출력의 주파수 최댓값은 0.447 cycle/px(= 0.246/0.55, 접힘 없음)이고, 진폭은 원래의 0.64배다.
- **확대(s > 1)**: `INTER_LINEAR`(업샘플이라 앨리어싱 없음, M1-10 Ultralytics 방식과 같음).
- **기준점**: 위쪽 행을 y = 0에 고정하고 가로는 중앙에 맞춘다(지표 행이 위에 남음, C′·D1과 같은 기하). 확대 때는 아래쪽과 좌우가 잘리고, 축소 때는 아래쪽과 좌우를 128로 채운다.
- **채움 값 128**: 축소로 생기는 빈 곳은 실제 L1의 채움(짧은 측선·C′ 아래쪽)과 같은 의미라 [SP] `pad_gray` 128을 쓴다. Ultralytics `translate`가 만드는 좁은 테두리(최대 ±25.6 px)는 114 그대로다.
- **상한 1.5**: 쌍곡선 최대 폭은 71 × 1.5 ≈ 107 px다([SP]는 AIHub 최대 80.7 px 기준 약 120 px). 겹침 128·192 px 안에 든다.
- **crop 경계 잘림(cv 판단)**: 가로 중앙 기준 확대에서 쌍곡선 박스 끝은 128 ± (12 × 1.5 + 54) ≈ 128 ± 72 px로 캔버스 안이다. 세로는 위 기준이라 쌍곡선 아래 끝(p99 95.6 px)이 1.5배에서 약 143 px다. 그래서 축척만으로 잘리는 쌍곡선은 거의 없다. 잘림은 주로 가로 이동에서 생기고, 이동을 적용한 영상의 약 10–20 %, 전체로 약 5–10 %로 추정한다(스모크 라벨 통계로 확인). 공동(아래 끝 최대 195 px)·맨홀(최대 145 px)은 1.5배에서 아래가 잘리는 경우가 생긴다(1-1-a 규칙 적용).
- **후속(D1 r3 뒤)**: 채택되면 [SP] `compress_max`(현재 1.25 = 1/0.8)를 1/0.55 ≈ 1.82로 고칠지 gpr-processing-expert가 판정한다.
- **위임 필요(gpr-processing-expert)**: 아래쪽 채움 띠·축소 채움(128)이 C′ 아래쪽 채움과 같은 의미인지 판단해야 한다. AIHub crop 아래쪽 회색 영역의 값·비율(육안 2장, [T] §2)과 맞추는 일은 Q3(M3-2)와 겹친다. 이 계획에서는 128로 고정하고, 답이 다르면 새 사전 등록으로 다룬다.

### 1-4. augmentation_policy.md에서 바뀌는 행 (채택되면 반영, 그 전에는 N2-a 실험 전용)

| 위치 | 지금 | 바꿀 내용 |
|---|---|---|
| §2 `scale` 행 | 권장 0.2 | N2-a: Ultralytics `scale` 0. 등방 축척은 프로젝트 증강 U[0.55, 1.5](축소 `INTER_AREA`, 확대 `INTER_LINEAR`, 위·가로중앙 기준). 근거 [T] §8-5, gpr-processing-expert 의견 2026-11-13, [SP] v0.3.6 §11-2에 기록 |
| §2 `translate` 행 | 0.1 유지 | 유지. 가로 위치 다양화는 프로젝트 가로 이동이 맡는다는 주석 |
| §2.1 표 | 대비·가우시안 잡음 등 | 새 행 3개: 등방 축척(프로젝트), 가로 이동(±96 px, 128, 확률 0.5), 가장자리 채움 띠(좌·우·아래, 128, 확률 0.3). 구현 위치 `augment_geometry.py` + `detect/yolo_train.py` |
| §3 설정 블록 | `scale: 0.2` | N2-a 예시(`scale: 0.0`, `gpr_geometric:` 블록) |
| §1 원칙 표 | 허용: 작은 이동·스케일 | "가로 이동은 ±96 px까지(타일 위치 재현), 세로 이동은 작게, 등방 축척 0.55–1.5배(축소는 면적 평균)" 추가. 금지 행(상하 반전·큰 회전·전단·원근·세로 비율 리사이즈)은 그대로 |

## 2. 학습 규모·시간·산출 경로

- **5-fold 한 번에**(U-a, 사용자 결정 2026-11-13). 채택 판정(A1·A2)과 D1 r3는 5-fold 평균이 판정 단위다. 학습 중 설정을 바꾸지 않는다.
- **사전 스모크**(지표를 보지 않음): fold 0, `epochs 2`, `--blind`, 산출 `runs/detect/yolo11m_aihubv1n2a_smoke_fold0/`. 확인하는 것은 다음 넷뿐이다.
  1. 종료 0(DataLoader·증강 오류 없음)
  2. `train_batch0–2.jpg`(학습 입력과 라벨만 그린 그림)에서 축척·이동·채움 띠·박스 정합
  3. epoch 시간과 VRAM(`sample.py`)
  4. 라벨 통계(증강 단계별 남긴·버린 박스 수, §5-2의 `aug_stats.json`)
  - `results.csv`·`results.png`·`val_batch*`·로그는 열지 않는다.
- **GPU 시간 추정**(M1-10 실측, `2026-11-01_m110_baseline.md` §1-1)

| 단계 | M1-10 실측 | N2-a 추정 |
|---|---|---|
| 학습 1 fold(50 epoch) | wall 1,512–1,543 s(epoch 2–50 평균 29.6–29.9 s), VRAM 약 7.3 GB | 26–32분. 프로젝트 증강(영상당 cv2.resize 1회 + 배열 복사)이 DataLoader workers 2에서 병목이 될 수 있어 +0–25 %. VRAM 같음 |
| 학습 5 fold | 2.1 h | **2.2–2.7 h** |
| 스모크(2 epoch) | — | 약 3분 |
| 평가(`eval_detector.py`, fold당 4변형 + 혼동 행렬) | fold당 229–253 s | 5 fold 약 20분 |
| G1용 검증기 검출 내보내기(`export_val_predictions.py`, crop) | — | 5 fold 약 5–10분 |
| D1 r3 | r2 wall 1,430 s | 약 24분 |
| 합계 | | **약 3.3–3.8 h**(30분 초과라 메인 세션 러너, `reserve.py`, `sample.py`) |

- **실행 명령 초안**(python-expert 제공 인자 기준, PowerShell, 작업 폴더 `C:\workspace\gpr_detection`, `<D>` = 실행일 `YYYYMMDD`, `<SEAL>` = 봉인 커밋 id). 모든 단계에 `--blind`를 붙이고, 러너가 stdout·stderr를 파일로 보낸다.

```powershell
$py = "C:\ProgramData\miniforge3\Scripts\conda.exe"
# 1) smoke: fold 0, 2 epochs, metrics not opened
& $py run -n gpr-dev --no-capture-output python -I scripts/train_detector.py --config configs/train_yolo11m_n2a.yaml --fold 0 --epochs 2 --device 0 --name yolo11m_aihubv1n2a_smoke_fold0 --blind
# 2) 5-fold training (k = 0..4), one runner step per fold
& $py run -n gpr-dev --no-capture-output python -I scripts/train_detector.py --config configs/train_yolo11m_n2a.yaml --fold <k> --device 0 --name yolo11m_aihubv1n2a_<D>_fold<k> --blind
# 3) evaluation (same defaults as M1-10: last/best x crop/image) + confusion matrix
& $py run -n gpr-dev --no-capture-output python -I scripts/eval_detector.py --run-dir runs/detect/yolo11m_aihubv1n2a_<D>_fold<k> --confusion-conf 0.25 --device 0 --blind
# 4) validator detections for D1 r3 gate G1
& $py run -n gpr-dev --no-capture-output python -I analysis/detection/export_val_predictions.py --device 0 --weights runs/detect/yolo11m_aihubv1n2a_<D>_fold<k>/weights/last.pt --data-yaml runs/detect/yolo11m_aihubv1n2a_<D>_fold<k>/eval/last_crop/eval_data.yaml --out-dir runs/analysis/n2a_val_export/fold<k>_crop --blind
# 5) A1 judgement (judge only opens the output)
& $py run -n gpr-dev python -I analysis/detection/n2a_a1_judge.py --config configs/train_yolo11m_n2a.yaml --run-prefix yolo11m_aihubv1n2a_<D>_fold --seal-commit <SEAL> --out analysis/detection/n2a_a1_verdict.json --blind
```

- 스모크의 `--epochs 2`는 명령줄 덮어쓰기다(설정 파일은 50 그대로). 스모크 run 폴더는 판정에 쓰지 않는다.
- (v0.3.1 보완) 학습·평가·내보내기는 `--device 0`을 명시한다. `--no-capture-output`은 러너 로그 파일에 실시간으로 쓰이게 하려는 것이다(블라인드 단계에서도 지표는 출력되지 않는다). 5-fold 학습은 `train_detector.py`의 기본값 `--on-existing error`를 그대로 둔다(같은 이름 run이 있으면 멈춤). 중단 뒤 재시작할 때만 `--on-existing archive`로 기존 폴더를 보관하고 다시 돌리며, 그 사실을 판정서에 공개한다.

- **산출 경로**(기존 `runs/` 보존)
  - 학습: `runs/detect/yolo11m_aihubv1n2a_<YYYYMMDD>_fold<k>/`
  - 평가: 같은 폴더 `eval/last_crop/` 등
  - 검증기 검출: `runs/analysis/n2a_val_export/fold<k>_crop/`
  - A1 판정: `analysis/detection/n2a_a1_verdict.json`(판정자만 연다)
  - D1 r3: 같은 run 폴더 `eval/m31_d1_r3/`, `analysis/detection/m31_d1_r3_summary.json`·`m31_d1_r3_verdict.json`
  - 설정 `configs/train_yolo11m_n2a.yaml`, 러너 로그 `runs/logs/n2a_runner_<날짜>/`

## 3. 기준선 영향과 채택 조건 (사전 등록)

### 3-1. 판정

1. **A1 — AIHub 원 crop 성능**(학습·평가 직후, D1 r3 전)
   - 지표: crop 단위, `last.pt`, 5 fold 비가중 평균, %, 반올림 없이 비교한다. 값은 `eval_detector.py`의 `metrics.json`(`map50`, `per_class[pipe_hyperbola].ap50`)에서 읽는다.
   - 조건: 전체 mAP50 평균 ≥ **76.8** 그리고 pipe_hyperbola AP50 평균 ≥ **90.9**.
   - 근거: 두 수는 U-3 검출 목표(M1-10 fold 평균 − 1 SD)와 같다. fold 평균이 이미 그 아래면 test에서 목표를 맞출 근거가 없다.
   - 게이트(판정 전 단언, `n2a_a1_judge.py`가 계산)
     - 평가 목록 sha256 = M1-10 같은 fold `eval/last_crop/metrics.json`의 `eval_images.sha256` 키 값(파일 이름이 아니라 metrics.json 안의 키).
     - 가중치 sha256 = 학습 기록. **v0.3 확정(§5-7 (가))**: 학습 기록은 `train_detector.py`가 쓰는 run 폴더의 `train_record.json`(`last`·`best` sha256)이다. 판정 때 `last.pt`를 다시 해시해 같아야 한다.
     - 설정 파일 sha256 = 봉인 커밋의 같은 경로 파일. **v0.3 확정(§5-7 (나))**: `configs/train_yolo11m_n2a.yaml`을 봉인 커밋에 넣는다. 판정자는 `--seal-commit <봉인 id>`를 넘기고, 스크립트는 `git show <봉인>:configs/train_yolo11m_n2a.yaml` 내용의 sha256과 각 fold `gpr_train_settings.yaml`의 `config_sha256`이 같은지 본다. 저장소는 `.gitattributes` `* text=auto eol=lf`이고 작업 파일도 LF라(2026-11-13 확인) 작업 파일과 blob 내용이 같다. 봉인 직후 이 둘이 같은지 한 번 확인한다.
     - 동일성 4개(구현 추가, 승인): 평가 가중치가 `last`, 단위가 `crop`, 평가 fold = 학습 fold, 평가 목록이 그 fold의 검증 목록.
     - 5 fold `args.yaml`의 `scale` 0, `translate` 0.1, `flipud` 0, `epochs` 50, `seed` 0.
     - 하나라도 어기면 판정 없음(종료 1).
   - 평균은 `Fraction`으로 정확히 계산한다(76.8은 통과, 76.79999는 불통과). 보고 전용 값은 `report_only` 블록에 둔다(판정 조건 아님).
2. **A2 — D1 r3**(§4): 규칙 O가 "50 %" 또는 "75 %"면 충족이다.
3. **채택** = A1 그리고 A2.

| A1 | A2 | 결과 | 다음 |
|---|---|---|---|
| 충족 | 충족 | **채택** | §3-2대로 기준선 교체 |
| 충족 | 미충족 | 불채택(위치 민감도 미해결) | 사용자 결정(N2-b·N2-c 결합 등). M1-10이 기준선으로 남음 |
| 미충족 | — | 불채택(AIHub 성능 손실) | D1 r3는 돌리지 않는다. M1-10 유지. 사용자 결정 |

- 보고만 하는 값(판정 조건 아님): pipe_band 띠 보조 지표(M1-10 §6-1), cavity·manhole AP50, mAP50-95, fold별 짝지은 차(N2-a − M1-10), conf 0.25 혼동 행렬, `best.pt` 값.

### 3-2. 기준선 교체 (U-c, 사용자 결정 2026-11-13)

- **채택되면**
  - ADR-001 이전 판정의 YOLO11m 기준값을 N2-a 5-fold 평균(crop, `last.pt`, 전체 mAP50)으로 바꾼다. 이전 가능 경계는 그 값 − 5 %p다.
  - M4 RF-DETR은 같은 증강 정책(등방 축척 0.55–1.5·가로 이동·채움 띠)으로 학습한다(같은 조건 비교, M1-10 §5 "맞춰야 할 조건").
  - **U-3 목표 절대값은 유지한다**: 전체 mAP50 ≥ 76.8 %, pipe_hyperbola AP50 ≥ 90.9 %, 띠 재현율 ≥ 64.1 %, 정밀도 ≥ 49.7 %.
  - 문서 갱신: ADR-001(add-architect), augmentation_policy(§1-4), CHECKLIST.
- **M1-10 기록은 지우지 않는다.** 불채택이면 M1-10이 기준선으로 남는다.

## 4. D1 r3 사전 등록과 블라인드

### 4-1. D1 r3 규칙

- [T] v0.5.2(봉인 3c81aca, 규칙 blob eacf9126376c199d9486bb8a98b13fa2b411d7bd)의 §8-2를 그대로 쓴다. 변형 20개, 지표, 짝지은 차, 집계 (3)(a)(b)(d)(e)(f)(g), 문턱 3.0 %p, 규칙 P·O와 조항 순서, 게이트 G1–G3, 채움 128.
- **바꾸는 것(새 모델에 맞춘 입력 값뿐)**
  - 가중치 `runs/detect/yolo11m_aihubv1n2a_<날짜>_fold<k>/weights/last.pt`.
  - G1 기준 검출: 새 모델의 `export_val_predictions.py` 출력(`runs/analysis/n2a_val_export/fold<k>_crop/`)과 새 `eval/last_crop/metrics.json`.
  - G2 기대 sha256: A1 통과 뒤 **D1 r3 봉인 부록**(이 문서 §4-3, 그때 추가)에 5 fold 가중치 sha256과 G1 입력 sha256을 적고 커밋한다.
  - G3 기대값은 r2와 같다(라벨·평가 목록·변형 기하 불변, `gp6_dx±64` 결손 0/0, 0/0, 0/3, 3/3, 2/0).
  - 실행·판정 명령(v0.3). `<SHA8>` = 부록의 5 fold `last.pt` sha256 앞 8자리를 쉼표로 이은 값이다.

```powershell
& $py run -n gpr-dev python -I analysis/detection/m31_d1_shift_scale.py --run-prefix yolo11m_aihubv1n2a_<D>_fold --export-root runs/analysis/n2a_val_export --folds 0,1,2,3,4 --device 0 --fill 128 --save-examples --out-subdir m31_d1_r3 --summary analysis/detection/m31_d1_r3_summary.json
& $py run -n gpr-dev python -I analysis/detection/m31_d1_judge.py --run-prefix yolo11m_aihubv1n2a_<D>_fold --export-root runs/analysis/n2a_val_export --expected-weights-sha8 <SHA8> --subdir m31_d1_r3 --out analysis/detection/m31_d1_r3_verdict.json
```

  - **공백(§5-7 (라))**: 지금 `m31_d1_shift_scale.py`에는 `--export-root`가 없다. 그래서 위 실행은 G1 입력 sha를 M1-10 내보내기에서 기록하고, 판정 때 G2-c가 실패한다. python-expert가 실행 스크립트에 `--export-root`(기본값 = M1-10 내보내기 `runs/analysis/m110_band_aux`)를 추가한다. 이 수정이 끝나고 감사 ③을 통과해야 r3를 실행한다. `m31_d1_judge.py`의 `--run-prefix`·`--export-root`·`--expected-weights-sha8`은 이미 있다(기본값 M1-10, r2 재현 유지).
- **바꾸지 않는 것**: 문턱·규칙·집계·게이트 정의, D1 스크립트 판(3c81aca 이후 인자 추가만 허용).
- **알려 둘 한계(판정서에 공개)**: 학습 증강(가로 이동 ±96, 채움 띠 128, 면적 평균 축소)은 D1 변형과 같은 종류다. 그래서 D1 r3는 "학습에서 본 변환에 대한 강건성"을 재는 쪽에 가깝다. 실제 타일 이음매·실측 도메인 일반화는 M3-3·D3에서 따로 확인한다.

### 4-2. 블라인드 절차 (학습·평가·A1·D1 r3 공통)

- **스크립트**(§5-5 구현): `scripts/train_detector.py`·`scripts/eval_detector.py`·`export_val_predictions.py`·A1 판정 스크립트에 `--blind`를 둔다. `--blind`이면 다음과 같이 동작한다.
  - Ultralytics LOGGER(`ultralytics.utils.LOGGER`)와 프로젝트 지표 로그를 WARNING 이상만 내보내 epoch별 val mAP 표·평가 요약 줄을 막는다. tqdm도 끈다(§5-0 #4). `verbose=False`·`plots`는 그대로 둔다.
  - 진행 상황은 지표 없는 한 줄(`{"stage":…, "fold":k, "epoch":e, "elapsed_s":…}`)로 stderr에 쓴다.
  - 마지막 줄 JSON은 `{"status", "exit_code", "out"}`만 담는다.
  - D1 스크립트는 이미 블라인드다([T] §8-4 (d)).
- **러너**: 모든 단계의 stdout·stderr를 `runs/logs/n2a_runner_<날짜>/` 파일로 보낸다. 메인 세션은 `.done`의 `exit=`와 `gpu.csv`·`sample.json`(자원)만 본다. 로그 본문·`tail`·`runner_summary.py` 로그 발췌는 열지 않는다.
- **열지 않는 파일**(판정자 외): `results.csv`, `results.png`, `*_curve.png`, `confusion_matrix*.png`, `val_batch*_pred.jpg`, `eval/**/metrics.json`·`per_class.csv`, `n2a_a1_verdict.json`, D1 r3 출력 전부. 스모크의 `train_batch*.jpg`는 라벨·입력 그림이라 열어도 된다.
- **판정자**(cv-detection-expert)만 A1·D1 r3 판정 파일을 연다. 판정자는 A1 판정 때 AIHub 지표를 보지만, 이것은 D1 판정 입력(변형 재현율)과 다른 값이다.
- **사후 변경 금지**: A1·D1 r3 결과를 본 뒤 증강·규칙을 바꾸지 않는다. 바꾸려면 새 사전 등록과 새 학습이 필요하고, 그 사실을 공개한다.
- **공개**: 블라인드가 깨지면(로그를 연 경우 등) 판정서에 누가 무엇을 언제 봤는지 그대로 적는다([T] §8-3 (6) 방식).

## 5. python-expert 구현 명세

**위치**
- 증강 함수: `src/gpr_detection/augment.py`(백엔드 공통, numpy + cv2).
- Ultralytics 연결: `src/gpr_detection/detect/yolo_train.py`(`YoloGprAugment`·`attach_gpr_augment`·`build_gpr_trainer`가 이미 있음).
- 설정 읽기: `scripts/train_detector.py`(`gpr_augment` 블록과 같은 방식).
- ultralytics import는 `detect/` 안에서만 한다(ADR-001).

### 5-0. 구현 결과 반영 (v0.3, python-expert 구현 2026-11-13, 시험 1,302개 통과)

아래는 §5-1–§5-6 명세와 다르게 구현된 곳이다. **봉인 판에서는 이 표가 우선한다.**

| # | 명세(v0.2) | 구현 | 판단 |
|---|---|---|---|
| 1 | cv2 증강을 `augment.py`에 | 새 모듈 `src/gpr_detection/augment_geometry.py`(축척·이동·채움·박스 필터). `augment.py`는 numpy만 유지. 추가로 `src/gpr_detection/detect/aug_stats.py`(증강 통계), `src/gpr_detection/detect/yolo_logging.py`(블라인드 로그 수준) | 승인. 아키텍처 규칙 5(`augment.py` cv2 금지)를 지킨다. 아키텍처 README §2-2의 `augment_geometry` 행은 M3-1 마감 때 add-architect가 추가한다(기록만) |
| 2 | 함수 반환 | 증강 함수는 모두 `keep_mask`를 함께 돌려준다. `augment_scale`은 건너뛸 때 `s=None` | 승인(뜻 같음) |
| 3 | `aug_stats.json` | worker별 파일을 epoch 끝에 합친다. worker 값은 batch(64장) 미만만큼 늦게 반영될 수 있다 | 승인. 보고 전용 통계라 판정에 영향 없음 |
| 4 | `--blind` 진행 줄 | tqdm도 끈다. 진행은 stderr에 지표 없는 JSON(train은 epoch마다) | 승인 |
| 5 | `n2a_a1_judge.py` 인자 | `--seal-commit`(필수), `--run-root`, `--repo` 추가. 동일성 검사 4개(last·crop·평가 fold·학습 fold) 추가. `Fraction` 정확 평균. `report_only` 블록 | 승인. §3-1 게이트에 반영 |
| 6 | 정체성 기록 | `train_detector.py`가 `gpr_train_settings.yaml`에 `config_sha256`, run 폴더 `train_record.json`에 `last`·`best` sha256을 남긴다 | 승인. §3-1 게이트 근거(§5-7 (가)(나)) |
| 7 | `m31_d1_judge.py` | `--expected-weights-sha8`(쉼표 목록)·`--run-prefix` 추가, 기본값 M1-10 | 승인. §4-1 명령에 반영 |
| 8 | (v0.3.1) §5-7 (다) 식만 사용 | `augment_scale`·`augment_shift_fill`이 단계 전에 입력 박스를 [0, W] × [0, H]로 자른다(`_clip_to_image`). 이유: 식만 쓰면 S-12에서 `RA_20210525(111537)_N00_B12_x(834-1090)`의 manhole 1개 × gain 3장(폭 6.15 px)이 버려졌다. 원인은 라벨 끝 결함(y1 = −0.000128 px)으로 축척 단계에서 "잘린 박스"로 판정된 것이다 | **승인**. D1 §8-4 (a)의 "라벨을 읽을 때 [0, 256]으로 자르기"와 같은 처리이고, Ultralytics도 이 박스를 정답으로 센다. px 허용 오차로 잘림을 판정하는 대안보다 규칙이 하나 적고, 넘침 크기(≤ 1.28 × 10⁻⁴ px)와 무관하게 결정적이다. 실제로 프레임 밖에 걸친 박스(라벨 결함이 아닌 경우)는 AIHub train_pool에 없다(넘침 최대 1.28 × 10⁻⁴ px, [T] §8-3 (7)) |
### 5-1. `augment.py` 추가 (v0.2 명세. 실제 위치는 §5-0 #1에 따라 `augment_geometry.py`)

설정(dataclass, frozen, `__post_init__` 검사)
- `IsotropicScaleConfig(probability: float = 1.0, scale_min: float = 0.55, scale_max: float = 1.5, fill: int = 128)`. 검사: 0 < scale_min ≤ 1 ≤ scale_max ≤ 2, fill ∈ [0, 255].
- `HorizontalShiftConfig(probability: float = 0.5, max_shift_px: int = 96, fill: int = 128)`. 검사: 0 ≤ max_shift_px ≤ 128.
- `EdgeFillConfig(probability: float = 0.3, side_weights: tuple[float, float, float] = (0.35, 0.35, 0.30), width_min_px: int = 8, width_max_px: int = 96, height_min_px: int = 16, height_max_px: int = 128, fill: int = 128)`. 검사: 가중치 합 1(허용 1e-9), 범위 정합.
- `BoxKeepRule(min_visible_area_fraction: float = 0.5, min_side_px: float = 4.0)`
- `GeometricAugmentConfig(scale: IsotropicScaleConfig, shift: HorizontalShiftConfig, edge_fill: EdgeFillConfig, keep_rule: BoxKeepRule)`. 기존 `AugmentConfig`·`strength`와 독립이다.

함수(모두 입력을 바꾸지 않음, 2-D uint8 단언)
- `scale_isotropic(image, boxes_xyxy, s, fill) -> (image, boxes_scaled, visible_rect)`
  - `H, W = image.shape`, `new_w = round(W*s)`, `new_h = round(H*s)`, `kx = new_w/W`, `ky = new_h/H`.
  - s = 1이면 원본 사본과 박스 사본을 그대로 돌려준다(리샘플하지 않음).
  - 보간: s < 1이면 `cv2.INTER_AREA`, s > 1이면 `cv2.INTER_LINEAR`.
  - 배치: `off_x = (W - new_w) // 2`(축소 양수, 확대 음수), `off_y = 0`. 출력 `[H, W]`는 `fill`로 채운 뒤, 리샘플 영상의 겹치는 부분을 복사한다.
  - 박스: `x' = off_x + x*kx`, `y' = y*ky`.
  - `visible_rect = (max(0, off_x), 0, min(W, off_x+new_w), min(H, new_h))`.
- `shift_horizontal(image, boxes_xyxy, dx, fill) -> (image, boxes, visible_rect)`: 정수 열 이동, 보간 없음, 빈 열 `fill`.
- `apply_edge_fill(image, side, size_px, fill) -> (image, visible_rect)`. side ∈ {"left", "right", "bottom"}.
- `filter_boxes(boxes_ref_area, boxes, visible_rect, rule) -> (boxes_clipped, keep_mask)`: 1-1-a 규칙. 기준 넓이는 그 단계 직전 넓이이고, 축척 단계는 `area * kx * ky`다.
- `augment_scale(image, boxes, config, rng) -> (image, boxes, keep_mask, s)`: 확률 판정 → `s = rng.uniform(scale_min, scale_max)` → `scale_isotropic` → `filter_boxes`.
- `augment_shift_fill(image, boxes, config, rng) -> (image, boxes, keep_mask, record)`: 이동(확률) → 채움 띠(확률, 위치는 `rng.choice(3, p=side_weights)`, 크기는 정수 균등) 순서다. 두 단계의 `keep_mask`를 합치고, `record`에 dx·side·size를 담는다.
- 세로 이동 함수는 만들지 않는다.

### 5-2. `detect/yolo_train.py` 추가

- `YoloIsotropicScale(config, seed)`·`YoloGeometricAugment(config, seed)`: `YoloGprAugment`와 같은 피클·워커별 RNG 방식이다. 스트림은 `SeedSequence([seed, worker_seed, k])`(k = 1 축척, 2 이동·채움)로 기존 증강과 분리한다.
  - `__call__`에서 `labels["instances"]`를 xyxy px로 읽는다(`_boxes_xyxy_px`와 같은 방식, 정규화·형식 처리).
  - 증강 뒤 `keep_mask`로 `instances`·`cls`를 거르고, 그 시점 Ultralytics 형식(`bbox_format`, `normalized`)으로 되돌린다.
  - 영상 shape `[H, W, 1]`을 유지한다.
- 삽입 위치
  - `YoloIsotropicScale`: `dataset.transforms`의 인덱스 1(`YoloGprAugment` 바로 뒤, Ultralytics 전처리 블록 앞).
  - `YoloGeometricAugment`: `Format` 바로 앞.
  - 실제 Compose 순서를 설치본에서 확인해 docstring에 적는다. `_close_dataloader_mosaic` 재구성 때도 둘 다 다시 넣는다. 검증 데이터셋에는 넣지 않는다.
- 단언(설정 검사, `check_augmentation_policy` 확장)
  - `gpr_geometric.scale`이 켜져 있으면 Ultralytics `scale`이 0이어야 한다. 아니면 `ValueError`(배율 이중 적용 방지).
  - 금지 키는 지금처럼 단언한다.
  - `max_shift_px` ≤ 128, fill ∈ [0, 255].
- 기록: 학습 폴더 `gpr_train_settings.yaml`에 `gpr_geometric` 블록과 시드를 남긴다. 워커별로 증강 통계(축척 적용 수와 s 히스토그램 10칸, 이동·채움 적용 수, 단계별 남긴·버린 박스 수, 클래스별)를 모아 epoch 끝에 `aug_stats.json`으로 쓴다. 지표 아님. 워커 합산이 어려우면 epoch 0만 주 프로세스에서 표본 1,000장으로 계산해도 된다.

### 5-3. `scripts/train_detector.py`

- 설정 yaml의 `gpr_geometric:` 블록을 `GeometricAugmentConfig`로 읽어 `build_gpr_trainer`에 넘긴다. 블록이 없으면 끈다(M1-10 설정 재현 보장).
- `--blind` 추가(§4-2).

### 5-4. 설정 파일 `configs/train_yolo11m_n2a.yaml` (M1-10 사본에서 다른 부분)

```yaml
# N2-a (docs/experiments/m31_n2a_retrain_plan.md v0.2): M1-10 copy, only these differ
ultralytics:
  scale: 0.0          # N2-a: isotropic scale moved to gpr_geometric.scale (was 0.2)
  translate: 0.1      # unchanged: keeps vertical shift small
gpr_geometric:
  scale: {probability: 1.0, scale_min: 0.55, scale_max: 1.5, fill: 128}   # area down / linear up, top-centre anchor
  horizontal_shift: {probability: 0.5, max_shift_px: 96, fill: 128}
  edge_fill: {probability: 0.3, side_weights: [0.35, 0.35, 0.30],
              width_min_px: 8, width_max_px: 96,
              height_min_px: 16, height_max_px: 128, fill: 128}
  keep_rule: {min_visible_area_fraction: 0.5, min_side_px: 4}
```

### 5-5. 블라인드·판정 지원

- `scripts/eval_detector.py`, `analysis/detection/export_val_predictions.py`: `--blind`(§4-2). 지표는 파일에만 쓴다.
- `analysis/detection/n2a_a1_judge.py`(신규): 5 fold `eval/last_crop/metrics.json`을 읽고 §3-1 게이트와 A1을 계산한다.
  - 입력: `--run-prefix`, `--config`, `--m110-prefix`, `--out`.
  - 출력: `n2a_a1_verdict.json`. stdout은 status만 쓴다.
  - 종료 코드: 0 판정 완료, 1 게이트 실패, 2 입력 없음·덮어쓰기 거부, 3 내부 오류.
  - 출력 파일이 있으면 종료 2(`--overwrite` 있을 때만 덮어씀). json은 `newline="\n"`.
  - `run_identity` 블록을 남긴다([T] §8-4 (e)와 같음).
- `analysis/detection/m31_d1_judge.py`: 가중치 감사 기대값을 인자 `--expected-weights-sha8 a,b,c,d,e`(또는 부록 json 경로)로 받게 한다. 기본값은 지금 값(M1-10)으로 두어 r2 재현을 유지한다.

### 5-6. 시험 기대값 (`tests/test_augment_geometry.py`, CPU)

256 × 256 합성 영상(값 = 열 번호 mod 256인 그라디언트, 또는 지정 무늬), 박스 xyxy px.

| ID | 내용 | 기대 |
|---|---|---|
| S-1 | 축소 s = 0.55, 박스 (100, 50, 150, 80) | `new` 141, `off_x` 57. 박스 (112.078125, 27.5390625, 139.6171875, 44.0625). 열 0–56·198–255, 행 141–255는 모두 128 |
| S-2 | 확대 s = 1.5, 박스 (100, 50, 150, 80) | `new` 384, `off_x` −64. 박스 (86, 75, 161, 120). 남음 |
| S-3 | 확대 s = 1.5, 박스 (20, 50, 50, 80) | (−34, 75, 11, 120) → 보이는 0–11, 11/45 = 0.244 → 버림 |
| S-4 | 확대 s = 1.5, 박스 (100, 140, 150, 180) | (86, 210, 161, 270) → 보이는 행 210–256, 46/60 = 0.767 → (86, 210, 161, 256) 남음 |
| S-5 | 확대 s = 1.5, 박스 (100, 150, 150, 200) | (86, 225, 161, 300) → 31/75 = 0.413 → 버림 |
| S-6 | s = 1.0 | 영상·박스 바이트 동일, 리샘플 호출 없음 |
| S-7 | 앨리어싱: 2 px 주기 세로 줄무늬(짝수 열 0, 홀수 열 255)를 s = 0.55로 축소 | 0행 표준편차 < 20(사전 확인 11.7). 같은 영상을 `INTER_LINEAR`로 줄인 대조는 > 60(73.6). 구현이 축소에 AREA를 쓰는지 확인 |
| S-8 | 확대 보간 | s = 1.2 결과가 `cv2.resize(..., INTER_LINEAR)` 후 같은 배치와 바이트 동일 |
| S-9 | 분포: 10,000회 | s가 U[0.55, 1.5](KS p > 0.01), s < 1 비율 0.474 ± 0.02 |
| S-10 | 이중 축척 방지 | `gpr_geometric.scale` 켜짐 + Ultralytics `scale` 0.2 → `ValueError` |
| G-1 | 이동 dx = +40, 박스 (100, 50, 150, 80) | 박스 (140, 50, 190, 80). 열 0–39 = 128. 열 j ≥ 40 = 원래 j − 40 |
| G-2 | dx = −40 | (60, 50, 110, 80). 열 216–255 = 128 |
| G-3 | dx = +120 | 보이는 220–256(36/50 = 0.72) → (220, 50, 256, 80) |
| G-4 | dx = +130 | 26/50 = 0.52 → (230, 50, 256, 80) |
| G-5 | dx = +140 | 16/50 = 0.32 → 버림 |
| G-6 | 왼쪽 띠 64, 박스 (40, 50, 90, 80) | 열 0–63 = 128. (64, 50, 90, 80)(0.52) |
| G-7 | 왼쪽 띠 64, 박스 (30, 50, 80, 80) | 16/50 → 버림 |
| G-8 | 아래 띠 100(행 156–255), 박스 (10, 120, 60, 180) | (10, 120, 60, 156)(36/60 = 0.6) |
| G-9 | 아래 띠 100, 박스 (10, 170, 60, 200) | 버림 |
| G-10 | (v0.3.1 정정) 잘린 3 px 조각: 박스 (−2, 50, 3, 80)을 단계 안에서 보이는 영역 [0, 256]으로 자름 → 보이는 폭 3, 넓이 비율 0.6 | 넓이 비율은 통과하지만 잘렸고 폭 < 4 px → 버림(`min_side_px`) |
| G-10b | (v0.3.1) 잘리지 않은 3 px 박스 (100, 50, 103, 80), 모든 확률 0 또는 s = 1 | 남음 |
| G-11 | 모든 확률 0 | 영상·박스·mask 바이트 동일 |
| G-12 | 같은 seed 두 번 / 다른 worker seed | 같음 / 다름 |
| G-13 | 분포 10,000회 | 이동 0.5 ± 0.02, dx 균등(카이제곱 p > 0.01), 채움 0.3 ± 0.02, 위치 0.35/0.35/0.30 ± 0.02, 크기 범위 지킴 |
| G-14 | 어댑터: 가짜 labels(정규화 xywh, xyxy px 두 형식) | 박스·`cls` 길이 일치, 원래 형식 복귀 |
| G-15 | 삽입 위치: 작은 합성 데이터셋으로 `build_dataset`(CPU) | 순서 `YoloGprAugment` → `YoloIsotropicScale` → … → 좌우 반전 → `YoloGeometricAugment` → `Format`. 각각 하나씩 |
| G-16 | 정책 | `flipud` ≠ 0 → `ValueError`. `max_shift_px` 129 → `ValueError` |
| G-17 | 피클 | 직렬화 뒤 같은 seed 같은 결과 |
| S-11 | (v0.3, §5-7 (다)) 잘리지 않은 얇은 박스: 폭 4.79 px 박스 (100, 50, 104.79, 120)에 s = 0.55 | 폭 2.64 px(141/256 배)로 줄지만 잘리지 않았으므로 남음(`keep_mask` True). 잘린 얇은 조각을 버리는 대조는 G-10 |
| S-12 | (v0.3) AIHub 실제 라벨 전수(train_pool 19,967개)에 s = 0.55 축척만 적용(이동·채움 없음) | 버려지는 라벨 0개(v0.2 규칙에서는 manhole 17개) |
| B-1 | `--blind`(train·eval·export·A1 판정) | stdout·stderr에 `map`, `mAP`, `ap50`, `recall`, `precision` 문자열과 숫자 표가 없음(작은 합성 데이터 1 epoch, CPU) |
| B-2 | `n2a_a1_judge.py` | 합성 metrics로 (1) 통과 → 0, (2) 평가 목록 sha 불일치 → 1, (3) `args.yaml` `scale` ≠ 0 → 1, (4) 출력 파일 있음 → 2. 76.8 경계값(76.79999·76.8)을 반올림 없이 판정 |

### 5-7. 확인 필요 4건 결정 (v0.3, 봉인 전)

- **(가) 가중치 학습 기록 = `train_record.json`**: 승인한다. `train_detector.py`가 학습 직후 run 폴더에 남기는 `last`·`best` sha256이 "학습 러너 기록"이다. A1 판정 때 `last.pt`를 다시 해시해 같아야 한다(§3-1).
- **(나) 설정 sha256 = 봉인 커밋의 파일**: 확인한다. `configs/train_yolo11m_n2a.yaml`(지금 미추적)을 이 문서 v0.3과 같은 봉인 커밋에 넣는다. 판정자는 `--seal-commit <봉인 id>`를 넘긴다. 비교 대상은 git 객체 id가 아니라 `git show <봉인>:<경로>` **내용**의 sha256이다. `.gitattributes`(`* text=auto eol=lf`)와 작업 파일 LF 상태(2026-11-13 확인)로 작업 파일과 같다. 봉인 직후 메인 세션이 둘이 같은지 한 번 확인한다.
- **(다) 축척 뒤 변 4 px 하한 — 규칙을 고친다.** 폭·높이 ≥ 4 px 조건은 보이는 영역에 의해 **잘린 박스에만** 적용한다.
  - 이유: 이 조건은 잘림으로 생긴 조각(날개 끝 등)을 거르려는 것이다. 축척 단계는 확률 1이라, 지금 규칙대로면 AIHub manhole 17개(최소 폭 4.79 px)가 s < 약 0.84에서 늘 버려진다. 보이는 객체를 배경으로 학습시키게 된다.
  - 영향: 축척 0.55배에서 이 박스들은 폭 약 2.6 px가 된다. Ultralytics `box_candidates`(폭·높이 > 2 px)는 통과하므로 Ultralytics 쪽에서도 남는다.
  - **python-expert 수정 명세**: `augment_geometry.filter_boxes`에서 박스별 `clipped = visible_area < pre_area * (1 - 1e-9)`를 계산한다. `min_side_px` 검사는 `clipped`인 박스에만 적용한다. 넓이 비율 검사(≥ 0.5)는 모든 박스에 그대로 적용한다. 시험 S-11·S-12를 추가하고, 기존 G-10(잘린 3 px 조각 버림)은 그대로 통과해야 한다.
- **(라) D1 r3 실행 스크립트 `--export-root` 공백**: §4-1에 기록했다. python-expert가 `m31_d1_shift_scale.py`에 `--export-root`(기본값 M1-10 내보내기)를 추가한다. r3 명령에는 이미 넣었다. 감사 ③에서 확인한다.

## 6. 실행 순서와 감사

1. python-expert가 §5-7 (다)·(라) 수정을 하고 S-11·S-12를 더해 전체 시험을 다시 통과시킨다(§5 나머지 구현과 S-1–S-10·G-1–G-17·B-1·B-2는 2026-11-13 완료, 1,302개 통과).
2. integrity-checker로 점검(이 문서 ↔ [T] §8-2·§8-5, augmentation_policy, 설정, 구현 파일)한 뒤 **이 문서 v0.3 + `configs/train_yolo11m_n2a.yaml` + 구현·시험 파일을 한 커밋으로 봉인**한다. 봉인 직후 설정 파일 작업본과 blob 내용의 sha256이 같은지 확인한다.
3. **감사 ①(실행 전, verif-protocol-auditor)**: 봉인 규칙(§3 A1·게이트, §4 블라인드) ↔ 구현(`n2a_a1_judge.py`, `--blind`, 설정 파일, 러너 명령) 정합을 본다. 차단이 있으면 고치고 다시 받는다.
4. 메인 세션이 2 epoch 스모크를 한다(§2). `train_batch*.jpg`·`aug_stats.json`·자원만 본다.
5. 메인 세션 러너가 5-fold 학습 → 평가 → 검증기 검출 내보내기를 한다(compute-scheduler-expert 배치표, `reserve.py`, `sample.py`, 모두 `--blind`).
6. cv-detection-expert가 `n2a_a1_judge.py`로 A1을 판정하고 판정서를 쓴다.
7. **감사 ②(판정 뒤, verif-protocol-auditor)**: A1 판정이 봉인 규칙대로인지(게이트·반올림·시점) 본다. A1이 미충족이면 여기서 멈추고 사용자 결정으로 간다.
8. A1이 충족이면 §4-3 D1 r3 봉인 부록(가중치·G1 입력 sha256)을 쓰고 커밋한다.
9. **감사 ③(실행 전)**: D1 r3 인자·부록·게이트 기대값을 본다.
10. 메인 세션이 D1 r3를 실행한다(블라인드).
11. cv-detection-expert가 D1 r3 판정서를 쓰고 A2와 채택을 판정한다.
12. **감사 ④(판정 뒤)**: A2·채택 판정을 감사한다.
13. 채택이면 기준선을 교체한다(add-architect: ADR-001, augmentation_policy·CHECKLIST 반영). gpr-processing-expert가 `compress_max`를 판정한다.

## 7. 결정 상태

| ID | 결정 | 상태 |
|---|---|---|
| U-a | 학습 규모 | **확정**: 5-fold 한 번에, 지표를 보지 않는 2 epoch 스모크 먼저(2026-11-13) |
| U-b | 축소 보간 | **확정**: 축소는 `INTER_AREA`(2026-11-13). 구현 방식: Ultralytics `scale` 0, 프로젝트 단일 단계 U[0.55, 1.5], 확대 `INTER_LINEAR`(§1-3, §5) |
| U-c | 기준선 | **확정**: A1·A2 모두 충족 시 기준선 교체, U-3 절대값 유지(2026-11-13) |
| U-d | 불채택 시 다음 | 결과가 나오면 묻는다(판정에는 영향 없음) |

## 8. 위임 필요

- **gpr-processing-expert**: 축소·아래 채움(128)이 C′ 아래 채움과 같은 의미인지(§1-3, Q3와 겹침, 메인 세션 판단으로 넘김). D1 r3 뒤 `compress_max` 판정.
- **python-expert**: §5-7 (다) `filter_boxes` 수정 + 시험 S-11·S-12, (라) `m31_d1_shift_scale.py --export-root`. 나머지 §5는 완료.
- **compute-scheduler-expert**: 약 3.5 h GPU 배치. VRAM 약 7.4 GB, batch 64, workers 2.
- **verif-protocol-auditor**: §6 감사 ①–④.
- **add-architect**: 채택 시 ADR-001 갱신. M3-1 마감 때 아키텍처 README §2-2에 `augment_geometry`(cv2 증강 모듈) 행 추가(§5-0 #1).
- **integrity-checker**: 봉인 전 정합 점검.

## 변경 이력

| 날짜 | 판 | 내용 |
|---|---|---|
| 2026-11-13 | v0.1 초안 | N2-a(사용자 결정 2026-11-13) 계획. 증강 3종, 고정 설정, 규모·시간, 채택 조건 A1·A2, D1 r3 사전 등록, 구현 명세·시험, 사용자 결정 U-a–U-d |
| 2026-11-13 | v0.1.1 | gpr-processing-expert 의견([SP] v0.3.4): 축척 하한 0.55, Ultralytics 선형 보간 확인, 상한 1.5의 잘림 판단, `compress_max` 후속 |
| 2026-11-13 | v0.2 | 사용자 결정 U-a(5-fold 한 번에 + 블라인드 스모크), U-b(축소 `INTER_AREA`: Ultralytics `scale` 0, 프로젝트 단일 축척 단계 U[0.55, 1.5], 축소 AREA·확대 LINEAR, 위·가로중앙 기준, 빈 곳 128, transforms 인덱스 1), U-c(기준선 교체, U-3 절대값 유지) 반영. A1 게이트, 감사 ①–④ 순서, 블라인드 절차(`--blind`, 로그 미열람, `val: true` 유지 이유), 구현 명세 §5-1–§5-6(S-1–S-10, B-1·B-2 추가). AREA·LINEAR 축소 사전 확인 수치 기록 |
| 2026-11-13 | v0.3 | 봉인 대상 판. python-expert 구현 결과 반영(§5-0: `augment_geometry.py`·`aug_stats.py`·`yolo_logging.py` 신설, `keep_mask`·`s=None`, `aug_stats` 병합, 블라인드 tqdm 끔, A1 판정 인자·동일성 4개·Fraction·report_only, `config_sha256`·`train_record.json`, D1 판정 인자). 확인 필요 결정(§5-7): (가) 가중치 기록 = `train_record.json` 승인, (나) 설정 파일을 봉인 커밋에 넣고 내용 sha256 비교, (다) 변 4 px 하한을 잘린 박스에만 적용하도록 규칙 수정(S-11·S-12 추가), (라) D1 r3 `--export-root` 공백 기록·명령 반영. 스모크·5-fold·평가·내보내기·A1·D1 r3 명령 초안(§2·§4-1). 실행 순서 §6 갱신 |
| 2026-11-13 | v0.3.1 | (다) 구현 반영: 단계 전 영상 프레임 자르기(`_clip_to_image`) 승인 — §5-0 #8, §1-1-a. 이유: 라벨 끝 결함(y1 = −0.000128 px) manhole 1개 × gain 3장이 S-12에서 버려짐, D1 §8-4 (a)와 같은 처리. G-10을 잘린 3 px 조각(−2, 50, 3, 80; 넓이 0.6)으로 정정하고 G-10b(잘리지 않은 3 px 박스 남음) 추가. 명령 초안에 `--device 0`·`--no-capture-output`, `--on-existing` 운용(기본 error, 재시작 때만 archive) 추가. 나머지 불변 |
| 2026-11-13 | v0.3.2 | integrity-checker 봉인 전 점검 반영(뜻 불변): §1 머리에 [T] §8-5 N2-a 초안 대비 변경과 이유 한 줄, [SP] v0.3.4 인용 4곳을 'gpr-processing-expert 의견 2026-11-13, [SP] v0.3.6 기록 예정'으로, 관련 목록·§1-4의 구현 위치에 `augment_geometry.py`, §4-2 tqdm 문장을 '끈다(§5-0 #4)'로, §3-1 `eval_images.sha256`이 metrics.json 키임을 명시, 0.5배 재현율 '약 0 %(0.41 %)' |
| 2026-11-13 | v0.3.3 | [SP] 축척 하한 인용(관련 목록, §1-3 곡률·하한, §1-4)을 '[SP] v0.3.6 §11-2'로 맞춤. 뜻 불변 |
