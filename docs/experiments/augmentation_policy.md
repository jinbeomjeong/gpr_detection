# GPR B-scan 검출 증강 정책 (Ultralytics YOLO11m 기준)

- 작성: 2026-10-29, cv-detection-expert (review 모드, 학습 없음)
- **개정: 2026-11-14 — 검출 기준선을 M1-10에서 N2-a′로 교체**(사용자 결정 N6-a)한 데 맞췄다.
  - 바뀐 것: 등방 축척을 프로젝트 증강으로 옮기고 범위를 넓혔다(0.55–1.5배, 축소는 면적 평균). 실제 맥락 안의 창 이동(문맥 이동)·채움 이동·가장자리 채움 띠를 추가했다.
  - 바뀌지 않은 것: 금지 증강(상하 반전·큰 회전·전단·원근·세로 비율 리사이즈·mosaic·mixup·cutmix)은 모두 그대로다.
  - 채택 근거: [`m31_n2a2_d1r4_judgment.md`](m31_n2a2_d1r4_judgment.md)(D1 r4, A2 충족·채택), [`m31_n2a2_a1_judgment.md`](m31_n2a2_a1_judgment.md)(A1), 계획 [`m31_n2a2_retrain_plan.md`](m31_n2a2_retrain_plan.md) v1.1, 이전 단계 [`m31_n2a_retrain_plan.md`](m31_n2a_retrain_plan.md) §1-4. 기준선 요약은 [`2026-11-14_n2a2_baseline.md`](2026-11-14_n2a2_baseline.md)다.
- 확인 환경: conda `gpr-dev`, ultralytics 8.4.192, torch 2.14.1+cu130
- 기본값 출처: `ultralytics.cfg.get_cfg()` 실행 결과와 `ultralytics/cfg/default.yaml`(115–132행)
- 관련 문서: `docs/research/prior_art.md` §0(gain 적용 영상만 검출 입력, letterbox), `docs/datasets/readme_morocco.md`

## 1. 원칙

B-scan의 세로축은 **시간(깊이)**, 가로축은 **주사 거리**다. 쌍곡선의 꼭짓점 위치와 곡률(전파 속도 정보)이 형태 단서이므로 다음을 지킨다.

| 구분 | 내용 | 이유 |
|---|---|---|
| 허용 | 좌우 반전, 밝기·대비, 가우시안 잡음 | 주사 방향 반전은 물리적으로 가능한 영상이다 |
| 허용(2026-11-14 확대) | **등방 축척 0.55–1.5배**(축소는 면적 평균 `INTER_AREA`, 확대는 선형, 위쪽 행 기준) | 등방 축척은 쌍곡선 꼴과 점근 45°를 보존한다(곡률 단서 불변). 하한 0.55는 반송파 주기 ≥ 2.2 px를 지키기 위해서다([SP] `spec_detection_image_v1.md` v0.3.6 §11-2). 실측 4TU 쌍곡선 크기(AIHub 대비 0.5–1.1배)를 덮는다 |
| 허용(2026-11-14 신설) | **가로 이동 ±96 px** — 실제 맥락 안의 창 이동(문맥 이동)을 우선하고, 채움(128) 이동은 소량만 | 실제 측선 타일에서 객체는 중앙을 벗어나고 옆은 실제 자료다. 채움 이동만 쓰면 "중앙 이탈 = 옆 채움"을 학습한다(D1 r3 판정서 §8-6). 세로 이동은 `translate` 0.1로 작게 둔다(지표 행 위치 유지) |
| 허용(2026-11-14 신설) | **가장자리 채움 띠**(좌·우 8–96 px, 아래 16–128 px, 값 128) | 실제 검출 영상(L1)의 채움(짧은 측선·C′ 아래 채움)을 재현한다 |
| 허용(조건부, 2026-11-14) | 같은 세션·같은 line(채널)·같은 gain에서 **x가 겹친 crop 이어 붙이기**(겹침 ≥ 16 px, 밝기 선형 보정, 상관 ≥ 0.98) | 같은 원본에서 잘린 연속 자료라 가짜 경계가 없다. 맥락 모드 창 이동의 출처다. 다른 세션·다른 gain·겹치지 않은 crop은 붙이지 않는다(아래 금지) |
| 금지 | 상하 반전 | 시간축 역전. 쌍곡선이 아래로 볼록해지고 지표 반사가 바닥으로 간다 |
| 금지 | 큰 회전, 전단, 원근 | 쌍곡선 대칭축이 기울어져 실제로 나올 수 없는 형태가 된다 |
| 금지 | 세로 비율을 크게 바꾸는 리사이즈(비등방 축척) | 곡률(속도) 단서가 왜곡된다. 입력은 비율 유지 letterbox |
| 금지 | mosaic, 겹치지 않은 crop·다른 세션 crop 붙이기, mixup, cutmix | 가짜 수평·수직 경계(반사처럼 보이는 불연속)나 가짜 쌍곡선 교차를 만든다 |
| 무의미 | 색상(hue)·채도(saturation)·채널 교환 | GPR 영상은 1채널 그레이이며 학습은 data yaml `channels: 1`로 1채널로 읽는다(M1-7 결정) |

## 2. Ultralytics 기본값과 권장값

| 키 | 기본값 (8.4.192 실측) | 권장값 (N2-a′ 기준선) | 판정 | 근거 |
|---|---|---|---|---|
| `flipud` | 0.0 | **0.0** (명시) | 기본값 유지, 명시 고정 | 시간축 역전 금지. 버전이 바뀌어도 꺼지도록 설정 파일에 적는다 |
| `fliplr` | 0.5 | 0.5 | 유지 | 주사 방향 반전은 허용 |
| `degrees` | 0.0 | **0.0** (명시) | 유지 | 회전은 쌍곡선 대칭축을 기울인다 |
| `shear` | 0.0 | **0.0** (명시) | 유지 | 쌍곡선 비대칭 왜곡 |
| `perspective` | 0.0 | **0.0** (명시) | 유지 | 세로·가로 비율이 위치마다 달라진다 |
| `translate` | 0.1 | 0.1 | 유지 | 세로 이동을 작게 유지한다. 가로 위치 다양화는 프로젝트 문맥 이동·채움 이동이 맡는다(§2.1) |
| `scale` | 0.5 (배율 0.5–1.5) | **0.0**(배율 1 고정) | 변경(2026-11-14) | 등방 축척은 프로젝트 단계 `YoloContextScale`이 맡는다(§2.1). Ultralytics `RandomPerspective` 축소는 선형 보간이라 앨리어싱 방지가 없고(`cv2.warpAffine` flags 없음), 기준점이 영상 중심이다. 둘이 함께 켜지면 `ValueError`(배율 이중 적용 방지). M1-10은 0.2였다 |
| `mosaic` | 1.0 | **0.0** | 끔 | 가짜 경계. 켜려면 별도 ablation 실험(`mosaic=0.5`)으로만 |
| `close_mosaic` | 10 | 0 | mosaic=0이면 무관 | mosaic ablation 시에는 10 유지 |
| `mixup` | 0.0 | 0.0 | 유지 | 영상 중첩은 가짜 쌍곡선 교차를 만든다 |
| `cutmix` | 0.0 | 0.0 | 유지 | 위와 같음 |
| `copy_paste` | 0.0 | 0.0 | 유지 | detect 작업에는 적용되지 않음(세그먼트 전용) |
| `hsv_h` | 0.015 | **0.0** | 끔 | 그레이 영상에서 무의미 |
| `hsv_s` | 0.7 | **0.0** | 끔 | 위와 같음 |
| `hsv_v` | 0.4 | **0.3** | 축소 | **1채널에서는 무효** — Ultralytics 8.4.192 `RandomHSV`가 3채널이 아니면 건너뜀(`augment.py:1512`). 설정에는 0.3을 남기되 무효임을 주석으로 표시 |
| `bgr` | 0.0 | 0.0 | 유지 | 그레이 영상에서 무의미 |
| `auto_augment` | randaugment | (무관) | 해당 없음 | 분류 작업 전용 |
| `erasing` | 0.4 | (무관) | 해당 없음 | 분류 작업 전용 |
| `imgsz` | 640 | **256** | 변경 | AIHub 원본 256×256을 업스케일하지 않음(M1 결정, dataset_spec_m1) |
| `rect` | False | False(학습), 검증은 Ultralytics가 rect 사용 | 유지 | 둘 다 비율 유지 |
| `multi_scale` | 0.0 | 0.0 | 유지 | 비교 실험의 입력 크기 통일 |
| `seed` / `deterministic` | 0 / True | 0 / True(모든 fold) | 5-fold | AIHub 세션 그룹 5-fold(ADR-003), 비교는 fold 평균±표준편차 |
| `workers` | 8 | 2 (Windows, 오류 시 0) | 축소 | Windows DataLoader 안정성, CPU 포화 시 경합 회피 |

### 2.1 프로젝트 증강 (Ultralytics에 없음)

| 증강 | 상태 | 구현 |
|---|---|---|
| 대비·감마, 가우시안 잡음, 수평 띠 잡음, 트레이스 누락(4종) | 구현됨(M1-7). **강도 0**(M1-7 결정, N2-a′도 0) | `src/gpr_detection/augment.py`(numpy), 어댑터 `detect/yolo_train.py` `YoloGprAugment`(transforms 맨 앞). 강도 > 0과 `gpr_context`는 함께 쓰지 못한다(밝기 보정 기준이 흐려짐) |
| **등방 축척 + 문맥 이동**(`YoloContextScale`, 2026-11-14 기준선) | 확률 0.5로 맥락 모드: s ~ U[1.0, 1.5], 창(256 px)을 실제 내용(확대 여유 + 겹친 crop 이어 붙인 띠) 안에서 dx ~ U[가능 범위 ∩ ±96]로 옮긴다(채움 없음). 나머지 0.5는 기본 모드: s ~ U[0.55, 1.5], 중앙, 축소로 빈 곳은 128. 축소 `INTER_AREA`, 확대 `INTER_LINEAR`, 위쪽 행 기준 | `src/gpr_detection/augment_geometry.py`(cv2 허용 모듈), 어댑터 `YoloContextScale`(transforms 인덱스 1). 맥락 인덱스 `data/aihub_context_index/`(sha256 고정: `crops.csv` e10192bfc42c8781769acb81556225996f0b069744b91179a787cc5fe923bbbf, `groups.csv` f1fc261d18eb91d0bfdc6fdbf252a97d0e80aa2ece4ac2a8764c9296a9a38570). 이웃은 같은 line·gain·fold·train_pool만 쓰고, 다르면 `ValueError`(누수 방지) |
| **채움 이동**(2026-11-14) | 확률 0.15, dx ~ U[−96, 96] 정수, 빈 열 128 | `augment_geometry.py`, 어댑터 `YoloGeometricAugment`(좌우 반전 뒤, `Format` 앞) |
| **가장자리 채움 띠**(2026-11-14) | 확률 0.15, 위치 좌 0.25 / 우 0.25 / 아래 0.5, 좌우 폭 U[8, 96], 아래 높이 U[16, 128], 값 128 | 위와 같음 |
| 가려진 박스 규칙(위 단계 공통) | 단계 전에 박스를 영상 프레임으로 자르고, 보이는 영역으로 다시 자른다. 보이는 넓이 ≥ 50 %이고, 잘린 박스만 변 ≥ 4 px일 때 남긴다. 맥락 띠 라벨 합침에서는 중심 crop 박스를 항상 남긴다. band는 중심 열 [a, b)에서 중심 라벨만 쓰고, 밖은 이웃 라벨을 열 범위로 잘라 쓰며 y는 바꾸지 않는다 | `augment_geometry.filter_boxes`·`merge_strip_labels` |

- 알려진 대가(기준선 판정서 §7): 맥락 모드 band 라벨의 약 30 %가 폭 32 px 미만 조각이 되고, 띠 보조 재현율이 61 %로 떨어졌다(M1-10 70 %). 축소 학습 비율이 47 → 24 %로 줄어 0.5배 재현율이 48 %다. 띠 조각 규칙은 개정 후보로 둔다(새 사전 등록 필요).

## 3. 설정 파일에 넣을 블록 (영문, 기준선 `configs/train_yolo11m_n2a2.yaml`)

기준선 설정 파일은 `configs/train_yolo11m_n2a2.yaml`(sha256 b9416b57e8877ba8c6b696fc6a3e63ba87b18bc608664e2613ff4c3d3ee142ef, 봉인 fa4b91c)이다. 증강 부분은 아래와 같다.

```yaml
# Augmentation policy for GPR B-scans (docs/experiments/augmentation_policy.md, baseline N2-a' 2026-11-14)
ultralytics:
  fliplr: 0.5
  flipud: 0.0        # never flip the time axis
  degrees: 0.0       # no rotation (distorts hyperbola symmetry)
  shear: 0.0
  perspective: 0.0
  translate: 0.1     # small vertical/horizontal jitter only
  scale: 0.0         # isotropic scaling is done by gpr_context (YoloContextScale)
  mosaic: 0.0
  close_mosaic: 0
  mixup: 0.0
  cutmix: 0.0
  copy_paste: 0.0
  hsv_h: 0.0         # grayscale input
  hsv_s: 0.0
  hsv_v: 0.3         # no effect on one-channel images
  bgr: 0.0
  multi_scale: 0.0
  imgsz: 256
  rect: false
  deterministic: true
gpr_augment:
  strength: 0.0      # M1-7 decision; must stay 0 with gpr_context
gpr_context:
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
```

- 이전 기준선(M1-10, `configs/train_yolo11m_m110.yaml`)은 Ultralytics `scale: 0.2`이고 프로젝트 기하 증강이 없었다. 그 설정은 기록용으로 남긴다.

## 4. RF-DETR(`gpr-detr`, rfdetr 1.11.2) 쪽 주의

비교 실험에서 같은 증강 정책을 맞추려면 아래 기본값을 바꿔야 한다(설치된 소스 확인 결과).

| 항목 | rfdetr 기본 | 문제 | 권장 |
|---|---|---|---|
| `aug_config` 기본(`AUG_CONFIG`) | HorizontalFlip p=0.5 | 없음 | 유지 |
| 프리셋 `AUG_AGGRESSIVE`·`AUG_AERIAL` | VerticalFlip p=0.5, Rotate ±45°/90° | 상하 반전·큰 회전 | 쓰지 않는다 |
| `square_resize_div_64` | True → `Resize((s, s))` | **비율 무시 정사각 늘리기**. letterbox 원칙 위반 | 비교 실험 전에 비율 유지 경로(letterbox 전처리 또는 `square_resize_div_64=False`) 확인 필요. 256 × 256 입력이면 항등(검출 입력 계약 C′ 타일) |
| `scale_jitter` | True → `RandomResize([400,500,600])` + `RandomSizedCrop(384–600)` → 정사각 | 잘라 낸 영역을 정사각으로 늘려 종횡비가 바뀐다 | False |
| `multi_scale` | PER_BATCH, `expanded_scales=True` | 입력 크기가 YOLO와 달라진다 | 비교 실험은 끔 |
| 밝기·대비 | `AUG_CONSERVATIVE`의 RandomBrightnessContrast는 Albumentations 백엔드 | albumentations 금지 | 사용자 정의 변환 또는 Kornia 경로 확인 |

**M4 요구사항(2026-11-14, 기준선 교체 뒤)**: ADR-001 이전 판정의 YOLO11m 기준이 N2-a′(5-fold 평균 mAP50 79.59 %, 경계 74.59 %)로 바뀌었다. 같은 조건 비교 원칙에 따라 M4-2 RF-DETR 학습에도 **§2.1의 N2-a′ 증강을 같게 적용**해야 한다. 적용 대상은 다음과 같다.
- 맥락 모드 등방 축척·문맥 이동(같은 맥락 인덱스와 sha256, 같은 누수 단언)
- 채움 이동 0.15
- 채움 띠 0.15
- 가려진 박스 규칙·라벨 합침
- 축소 `INTER_AREA`
- 강도 0인 GPR 증강
- Ultralytics 쪽 `translate` 0.1·`fliplr` 0.5에 해당하는 rfdetr 변환

rfdetr의 `scale_jitter`·`multi_scale`·정사각 늘림은 끈다. 실제 구현은 M4에서 python-expert가 하고, 학습 전 smoke에서 입력 텐서 크기와 `train_batch` 그림으로 확인한다.

이 표의 비율 유지 경로는 실제 학습 smoke에서 입력 텐서 크기로 다시 확인한다.

## 5. 데이터 쪽에 이미 들어 있는 금지 증강 (모로코)

`docs/datasets/readme_morocco.md` 66–71행에 따르면 모로코 세트에는 **회전 ±15°, 탄성 변형**이 미리 적용된 사본이 있고, 공동은 원본을 비율 무시하고 224×224로 늘려 쌍곡선이 비등방으로 찌그러져 있다. 학습 설정의 증강을 꺼도 이 사본은 그대로 들어온다. 사본은 같은 원본 그룹의 증강본이므로 독립 표본으로 세지 않는다. **결정(2026-10-29, 사용자)**: 회전·탄성 사본은 학습·평가에서 뺀다(`aug_transform` 기준, 제외 목록은 gpr-dataset-curator가 매니페스트와 함께 남긴다). 공동 하위셋의 비등방 늘림은 사본 제외로 해결되지 않으므로 공동 클래스 지표를 도메인별로 따로 본다.

## 6. 점검 목록 (학습 전)

- [ ] `flipud: 0.0`, `degrees: 0.0`, `shear: 0.0`, `perspective: 0.0`이 설정 파일에 명시돼 있다
- [ ] Ultralytics `scale: 0.0`이고 `gpr_context`가 켜져 있다(둘이 함께 축척하지 않음)
- [ ] `gpr_augment.strength` 0이다(`gpr_context`와 함께 쓸 때)
- [ ] 맥락 인덱스 sha256이 §2.1 고정값과 같고, 학습 기록 `gpr_context_index`의 `train_folds`·`validation_folds`가 fold 정의와 맞다
- [ ] 학습 로그 첫 줄(`train: ... hyperparameters`)과 `gpr_train_settings.yaml`에서 위 값이 반영됐는지 확인했다
- [ ] 입력은 gain 적용 영상만 쓴다
- [ ] 비교 실험(YOLO11m ↔ RF-DETR)은 같은 분할, 같은 `imgsz`, 같은 epoch 예산, **같은 증강 정책(§2.1 포함)**이다

## 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-10-29 | 최초 작성(M1 기준) |
| 2026-11-14 | 기준선 N2-a′ 교체(사용자 결정 N6-a) 반영: 허용 증강에 등방 축척 0.55–1.5(축소 AREA)·문맥 이동·채움 이동·채움 띠·겹친 crop 이어 붙이기(조건부) 추가, Ultralytics `scale` 0.2 → 0, §2.1 프로젝트 증강 표, §3 기준선 설정 블록, §4 M4 RF-DETR 동일 증강 요구, §6 점검 항목. 금지 증강 불변 |
