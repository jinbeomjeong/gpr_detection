# GPR B-scan 검출 증강 정책 (Ultralytics YOLO11m 기준)

- 작성: 2026-10-29, cv-detection-expert (review 모드, 학습 없음)
- 확인 환경: conda `gpr-dev`, ultralytics 8.4.192, torch 2.14.1+cu130
- 기본값 출처: `ultralytics.cfg.get_cfg()` 실행 결과와 `ultralytics/cfg/default.yaml`(115–132행)
- 관련 문서: `docs/research/prior_art.md` §0(gain 적용 영상만 검출 입력, letterbox), `docs/datasets/readme_morocco.md`

## 1. 원칙

B-scan의 세로축은 **시간(깊이)**, 가로축은 **주사 거리**다. 쌍곡선의 꼭짓점 위치와 곡률(전파 속도 정보)이 형태 단서이므로 다음을 지킨다.

| 구분 | 내용 | 이유 |
|---|---|---|
| 허용 | 좌우 반전, 작은 이동·스케일(등방), 밝기·대비, 가우시안 잡음 | 주사 방향 반전은 물리적으로 가능한 영상이다. 등방 스케일은 쌍곡선 종횡비를 바꾸지 않는다 |
| 금지 | 상하 반전 | 시간축 역전. 쌍곡선이 아래로 볼록해지고 지표 반사가 바닥으로 간다 |
| 금지 | 큰 회전, 전단, 원근 | 쌍곡선 대칭축이 기울어져 실제로 나올 수 없는 형태가 된다 |
| 금지 | 세로 비율을 크게 바꾸는 리사이즈 | 곡률(속도) 단서가 왜곡된다. 입력은 비율 유지 letterbox |
| 무의미 | 색상(hue)·채도(saturation)·채널 교환 | GPR 영상은 1채널 그레이(3채널 복제로 로드됨) |

## 2. Ultralytics 기본값과 권장값

| 키 | 기본값 (8.4.192 실측) | 권장값 | 판정 | 근거 |
|---|---|---|---|---|
| `flipud` | 0.0 | **0.0** (명시) | 기본값 유지, 명시 고정 | 시간축 역전 금지. 버전이 바뀌어도 꺼지도록 설정 파일에 적는다 |
| `fliplr` | 0.5 | 0.5 | 유지 | 주사 방향 반전은 허용 |
| `degrees` | 0.0 | **0.0** (명시) | 유지 | 회전은 쌍곡선 대칭축을 기울인다 |
| `shear` | 0.0 | **0.0** (명시) | 유지 | 쌍곡선 비대칭 왜곡 |
| `perspective` | 0.0 | **0.0** (명시) | 유지 | 세로·가로 비율이 위치마다 달라진다 |
| `translate` | 0.1 | 0.1 | 유지 | 작은 이동(±10%)은 허용 범위 |
| `scale` | 0.5 (배율 0.5–1.5) | **0.2** (0.8–1.2) | 축소 | 등방이라 종횡비는 보존되지만 0.5배 축소는 작은 쌍곡선을 수 px로 만든다. "작은 스케일"로 제한 |
| `mosaic` | 1.0 | **0.0** (기준 실험) | 끔 | 서로 다른 B-scan을 이어 붙여 가짜 수평·수직 경계(반사처럼 보이는 불연속)를 만든다. 허용 목록에 없다. 켜려면 별도 ablation 실험(`mosaic=0.5`)으로만 |
| `close_mosaic` | 10 | 0 | mosaic=0이면 무관 | mosaic ablation 시에는 10 유지 |
| `mixup` | 0.0 | 0.0 | 유지 | 영상 중첩은 가짜 쌍곡선 교차를 만든다 |
| `cutmix` | 0.0 | 0.0 | 유지 | 위와 같음 |
| `copy_paste` | 0.0 | 0.0 | 유지 | detect 작업에는 적용되지 않음(세그먼트 전용) |
| `hsv_h` | 0.015 | **0.0** | 끔 | 그레이 영상에서 무의미. 색상 맵 영상이 섞일 경우 가짜 색 변화 방지 |
| `hsv_s` | 0.7 | **0.0** | 끔 | 위와 같음 |
| `hsv_v` | 0.4 | **0.3** | 축소 | 밝기(곱셈 이득) 증강. AIHub는 gain 0.25/0.5/1.0 사본이 이미 강도 변화를 주므로 약하게 |
| `bgr` | 0.0 | 0.0 | 유지 | 그레이 영상에서 무의미 |
| `auto_augment` | randaugment | (무관) | 해당 없음 | 분류 작업 전용. detect 학습에는 쓰이지 않는다 |
| `erasing` | 0.4 | (무관) | 해당 없음 | 분류 작업 전용 |
| `imgsz` | 640 | 640 | 유지 | 학습 시 letterbox(비율 유지 + 패딩). 정사각 강제 늘리기 없음 |
| `rect` | False | False(학습), 검증은 Ultralytics가 rect 사용 | 유지 | 둘 다 비율 유지 |
| `multi_scale` | 0.0 | 0.0 | 유지 | 켜도 등방이지만 비교 실험의 입력 크기 통일을 위해 끈다 |
| `seed` / `deterministic` | 0 / True | fold별 고정·기록 / True | 5-fold | AIHub 세션 그룹 5-fold(ADR-003), 비교는 fold 평균±표준편차 |
| `workers` | 8 | 0–2 (Windows 첫 시도는 2, 오류 시 0) | 축소 | Windows DataLoader 안정성, CPU 포화 시 경합 회피 |

### 2.1 Ultralytics에 없는 허용 증강

| 증강 | 상태 | 조치 |
|---|---|---|
| 대비(contrast) | Ultralytics 기본 증강에 없음(`hsv_v`는 곱셈 이득이라 밝기·대비가 함께 변함) | 필요하면 사용자 정의 변환으로 추가 |
| 가우시안 잡음 | 없음. Albumentations 연동 블록(`augment.py` 2150행: Blur·MedianBlur·ToGray·CLAHE, 각 p=0.01)은 albumentations가 설치돼 있지 않아 **비활성**(gpr-dev에 미설치 확인) | albumentations는 이 프로젝트에서 쓰지 않는다. 잡음·대비 변환은 `src/gpr_detection/detect/` 안의 사용자 정의 변환으로 구현(python-expert 위임) |

## 3. 설정 파일에 넣을 블록 (영문, `configs/train_yolo11m_<실험>.yaml`)

```yaml
# Augmentation policy for GPR B-scans (see docs/experiments/augmentation_policy.md)
fliplr: 0.5
flipud: 0.0        # never flip the time axis
degrees: 0.0       # no rotation (distorts hyperbola symmetry)
shear: 0.0
perspective: 0.0
translate: 0.1
scale: 0.2         # isotropic 0.8-1.2, keeps aspect ratio
mosaic: 0.0        # baseline; mosaic=0.5 only as a separate ablation
close_mosaic: 0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
hsv_h: 0.0         # grayscale input
hsv_s: 0.0
hsv_v: 0.3
bgr: 0.0
multi_scale: 0.0
imgsz: 640
rect: false
deterministic: true
```

## 4. RF-DETR(`gpr-detr`, rfdetr 1.11.2) 쪽 주의

비교 실험에서 같은 증강 정책을 맞추려면 아래 기본값을 바꿔야 한다(설치된 소스 확인 결과).

| 항목 | rfdetr 기본 | 문제 | 권장 |
|---|---|---|---|
| `aug_config` 기본(`AUG_CONFIG`) | HorizontalFlip p=0.5 | 없음 | 유지 |
| 프리셋 `AUG_AGGRESSIVE`·`AUG_AERIAL` | VerticalFlip p=0.5, Rotate ±45°/90° | 상하 반전·큰 회전 | 쓰지 않는다 |
| `square_resize_div_64` | True → `Resize((s, s))` | **비율 무시 정사각 늘리기**. letterbox 원칙 위반 | 비교 실험 전에 비율 유지 경로(letterbox 전처리 또는 `square_resize_div_64=False`) 확인 필요 |
| `scale_jitter` | True → `RandomResize([400,500,600])` + `RandomSizedCrop(384–600)` → 정사각 | 잘라 낸 영역을 정사각으로 늘려 종횡비가 바뀐다 | False |
| `multi_scale` | PER_BATCH, `expanded_scales=True` | 입력 크기가 YOLO와 달라진다 | 비교 실험은 끔 |
| 밝기·대비 | `AUG_CONSERVATIVE`의 RandomBrightnessContrast는 Albumentations 백엔드 | albumentations 금지 | 사용자 정의 변환 또는 Kornia 경로 확인 |

이 표의 비율 유지 경로는 실제 학습 smoke에서 입력 텐서 크기로 다시 확인한다.

## 5. 데이터 쪽에 이미 들어 있는 금지 증강 (모로코)

`docs/datasets/readme_morocco.md` 66–71행에 따르면 모로코 세트에는 **회전 ±15°, 탄성 변형**이 미리 적용된 사본이 있고, 공동은 원본을 비율 무시하고 224×224로 늘려 쌍곡선이 비등방으로 찌그러져 있다. 학습 설정의 증강을 꺼도 이 사본은 그대로 들어온다. 사본은 같은 원본 그룹의 증강본이므로 독립 표본으로 세지 않는다. **결정(2026-10-29, 사용자)**: 회전·탄성 사본은 학습·평가에서 뺀다(`aug_transform` 기준, 제외 목록은 gpr-dataset-curator가 매니페스트와 함께 남긴다). 공동 하위셋의 비등방 늘림은 사본 제외로 해결되지 않으므로 공동 클래스 지표를 도메인별로 따로 본다.

## 6. 점검 목록 (학습 전)

- [ ] `flipud: 0.0`, `degrees: 0.0`, `shear: 0.0`, `perspective: 0.0`이 설정 파일에 명시돼 있다
- [ ] 학습 로그 첫 줄(`train: ... hyperparameters`)에서 위 값이 반영됐는지 확인했다
- [ ] 입력은 gain 적용 영상만 쓴다
- [ ] 비교 실험(YOLO11m ↔ RF-DETR)은 같은 분할, 같은 `imgsz`, 같은 epoch 예산, 같은 증강 정책이다
