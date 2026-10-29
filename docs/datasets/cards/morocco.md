---
pretty_name: "Morocco GPR Utilities and Voids Dataset (Mojahid et al., Mendeley Data)"
license: cc-by-4.0
language:
  - en
task_categories:
  - object-detection
size_categories:
  - 1K<n<10K
tags:
  - ground-penetrating-radar
  - gpr
  - b-scan
  - buried-pipe
  - cavity
  - morocco
  - gssi
  - yolo
source_datasets:
  - original
annotations_creators:
  - expert-generated
---

# 모로코 GPR 관로·공동 영상 데이터셋

| Item | Value |
|---|---|
| Summary | Morocco construction-site GPR B-scan screen captures cut into 224x224 patches, with boxes for utility pipes and cavities (plus intact/background patches) |
| Format | 8-bit grayscale JPG + YOLO txt (VOC xml in the original); processed copy has YOLO folders, `images_index.csv`, `boxes.csv`, `split_groups.csv` |
| Size | 2,524 images = 285 unique originals + 2,239 augmented copies; 1,607 boxes (6 degenerate, 1,601 used) |
| Labels | pipe (Utility), cavity; intact images have empty labels |
| Physical scale | None (screen captures: no ns, m or raw waveform) |
| Role in this project | Cross-domain evaluation only (`morocco_eval_v1`, 233 images, never used for training or tuning) |
| License | CC BY 4.0 (original); attribution required |

## Dataset Summary

모로코 여러 도시의 공사현장(2019–2024)에서 GSSI 400 MHz·200 MHz 안테나로 찍은 B-scan 화면을 224x224 패치로 잘라 관로(pipe)·공동(cavity) 박스를 단 영상 탐지용 데이터다. 하위셋은 utilities(관로), cavities(공동), intact(이상 없음, 빈 라벨 = 배경)이다. 고유 영상은 285장뿐이고 나머지는 이를 6–12배 증강한 사본이다([readme_morocco.md §1](../readme_morocco.md)).

## Dataset Structure

### 형식

- 로컬 위치: `C:\workspace\GPR_Data\모로코_관로공동GPR영상_가공본_261016\03_yolo_dataset`(`images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`).
- 메타데이터 `01_메타데이터/`: `images_index.csv`(영상 2,524행: 새 이름·하위셋·증강 종류 `aug_transform`·크기·박스 수·그룹·분할), `boxes.csv`(박스 1,607개), `split_groups.csv`(분할 그룹 233개와 묶은 사유).
- 영상 이름: `{U|C|I}_{원본 stem}_aug_{번호}` (U 관로, C 공동, I 이상 없음).
- 영상 크기: 224x224가 2,089장, 이상 없음의 `_aug_3`은 268x268, `_aug_9`는 336x336(각 75장). 전부 8비트 1채널.

### 제공 분할 (그룹 단위, 약 70/15/15)

| 분할 | 영상 | 관로 / 공동 / 이상 없음 영상 | 관로 박스 | 공동 박스 |
|---|---:|---|---:|---:|
| train | 1,598 | 570 / 392 / 636 | 703 | 418 |
| val | 329 | 108 / 77 / 144 | 145 | 90 |
| test | 312 | 108 / 84 / 120 | 154 | 91 |
| 합계 | 2,239 | 786 / 553 / 900 | 1,002 | 599 |

- 고유 원본: 관로 131장(근접 중복을 합치면 120그룹), 공동 79장(75그룹), 이상 없음 75장(38그룹) = 285장, 분할 그룹 233개.
- 분할 그룹 묶음: 관로 근접 중복 5군, 공동 근접 중복 4쌍, 이상 없음 37쌍(`N.JPG`와 `N (2).JPG`).

### 라벨

- 원본 YOLO txt는 두 하위셋 모두 클래스 0이라 가공본이 관로 0·공동 1로 다시 매겼다. **이 프로젝트는 다시 `pipe` → 0 `pipe_hyperbola`, `cavity` → 2 `cavity`로 매핑한다**([ADR-003](../../architecture/ADR-003-data-split-and-classes.md) 결정 1).
- 폭 또는 높이 3 px 미만 퇴화 박스 6개는 YOLO 라벨에서 제외(boxes.csv에 `degenerate_excluded=예`로 남음).
- 박스 1,607개 중 639개가 영상 경계에 닿는다(`touches_border`).
- 박스는 증강본마다 사람이 따로 그린 것이라 같은 원본의 증강본끼리 박스 수·위치가 다를 수 있다.

### 증강 종류 (`aug_transform`)

| 하위셋 | 번호 | 변환 |
|---|---|---|
| 관로 | 1–6 | 가우시안 잡음 / 가로 이동 +20px / **회전 +15° (3)** / 좌우 반전 / **탄성 변형 (5)** / 주파수 이동(원본과 거의 동일, 6) |
| 공동 | 7–13 | 가우시안 잡음 / 가로 이동 +20px / 중앙 1.5배 확대 / **회전 +15° (10)** / 좌우 반전 / **탄성 변형 (12)** / 주파수 이동(원본과 거의 동일, 13) |
| 이상 없음 | 1–12 | 잡음 / 가로 이동 +10px / 268 리사이즈 / **회전 +15° (4)** / 좌우 반전 / **탄성 변형 (6)** / 강한 잡음 / 가로 이동 +20px / 336 리사이즈 / **회전 −15° (10)** / **탄성 변형 (11)** / 원본과 거의 동일(12) |

굵게 표시한 회전·탄성 사본은 이 프로젝트에서 금지 증강 사본이다(관로 3·5, 공동 10·12, 이상 없음 4·6·10·11). 번호가 아니라 `images_index.csv`의 `aug_transform` 열로 걸렀고 위 번호와 일치함을 확인했다([audit_morocco_eval_v1.md](../audit_morocco_eval_v1.md)).

## Dataset Creation

- **출처**: Mojahid 외, *Intelligent recognition of subsurface utilities and voids: A Ground Penetrating Radar dataset for Deep Learning applications*, Mendeley Data V1, DOI 10.17632/ww7fd9t325.1 (https://data.mendeley.com/datasets/ww7fd9t325/1). 논문 Data in Brief 59 (2025) 111338. 처리 스크립트: https://github.com/LCSkhalid/GPR_Data (저장소 라이선스 표기 미확인, 참고용).
- **가공(받은 가공본)**: 클래스 재매핑, 퇴화 박스 6개 제외, 파일명 변경, 그룹 기반 분할, YOLO 폴더 구성. 변환 종류는 원본과 증강본을 상관 대조해 확인한 것이며 원 저자 `augmentation.py` 목록과 이상 없음 쪽이 다르다.
- **라벨링**: 원 저자의 사람 라벨(VOC xml → YOLO txt). 가공본이 VOC와 1,607개 박스를 전부 대조해 최대 차이 0.5 px를 확인했다.

## Considerations (알려진 결함·편향·한계)

1. **물리 축척이 없다.** 화면 캡처라 ns·m로 바꿀 수 없다. 깊이 추정에는 쓰지 않는다.
2. **고유 영상은 285장뿐**이고 증강 사본은 독립 표본이 아니다. 같은 프로파일을 5 px씩 밀어 자른 8장 같은 근접 중복 그룹이 있다.
3. **공동 하위셋은 원본 전체가 비율을 무시하고 224x224로 늘려져 있다**(쌍곡선이 비등방으로 찌그러짐). 회전·탄성 사본을 빼도 해결되지 않는다. **공동 클래스 평가 시 주의**: 공동 지표는 따로 보고하고 관로·이상 없음과 합산하지 않는다(매니페스트 `anisotropic_stretch=True` 75장).
4. 제공 분할을 넘는 근접 중복 1쌍(`U_233` ↔ `U_239`, pHash 해밍 6)이 평가 세트 안에 있다.
5. 관로 영상 위쪽에 흰 세로선 두 줄이 보이는 영상이 많다. 원본 패치의 표식으로 보이며 AIHub 영상에는 없다. 검출 모델이 이를 단서로 삼는지 평가 때 확인해야 한다.
6. GSSI 단채널 화면 캡처라 국내 장비 영상과 질감이 다르다. 가공본 설명서는 이 데이터를 단독 평가셋으로 쓰지 않을 것을 권하며, 이 프로젝트는 **교차 도메인 평가 전용**으로만 쓴다.
7. 이상 없음 38그룹은 빈 라벨이라 오검출 평가에만 쓸 수 있다. 라벨 정확도 전수 검수는 하지 않았다(표본 18장 육안 확인).
8. 원본 패치 285장에는 라벨이 없다(관로 `_aug_6`, 공동 `_aug_13`이 원본과 거의 같아 그 박스를 대용).

## Licensing & Citation

- **라이선스: CC BY 4.0.** 출처 표시와 변경 사실 명시 조건으로 복제·수정·상업 이용 가능. 받은 가공본은 클래스 재매핑·퇴화 박스 제거·파일명 변경·분할을 한 변경본이다.
- 데이터: Mojahid, A., El Ouai, D., El Amraoui, K., El Hami, K., Aitbenamer, H. (2024). Intelligent recognition of subsurface utilities and voids: A Ground Penetrating Radar dataset for Deep Learning applications. Mendeley Data, V1, doi:10.17632/ww7fd9t325.1
- 논문: Mojahid, A. et al. (2025). Intelligent recognition of subsurface utilities and voids: A ground penetrating radar dataset for deep learning applications. Data in Brief, 59, 111338. doi:10.1016/j.dib.2025.111338

## 이 프로젝트에서의 사용

- **역할**: 교차 도메인 평가 전용(`morocco_eval_v1`). 학습·튜닝에 쓰지 않는다. 그룹(233개)마다 "원본과 거의 동일" 사본 1장(관로 `_aug_6` 120, 공동 `_aug_13` 75, 이상 없음 `_aug_12` 38)만 골라 224x224 그대로 두고 평가 때 256 letterbox한다.
- **쓰는 파생 데이터**: `data/morocco_eval_v1/`(images·labels 각 233, `eval_images.txt`, `eval.yaml`), 매니페스트 `data/splits/morocco_eval_v1.csv`(git 제외). 제공 분할은 참고 열 `split_src`(163/35/35)로만 보존.
- **개수 흐름**: 라벨 있는 2,239장 − 회전·탄성 720장 − 기타 사본 1,286장 = 233장(박스 239: 관로 158, 공동 81). 라벨 없는 원본 패치 285장은 쓰지 않음.
- **감사**: 판정 통과(그룹당 1장, 회전·탄성 0장, 이미지 SHA256 원본과 불일치 0, 퇴화·범위 밖 박스 0). AIHub 16,887장과 근접 중복 없음(최소 해밍 6, 육안 확인).
- **관련 체크리스트**: M1-4(평가 세트), M1-5(감사), M3-3(교차 도메인 평가).
- 근거 문서: [readme_morocco.md](../readme_morocco.md), [audit_morocco_eval_v1.md](../audit_morocco_eval_v1.md), [dataset_spec_m1.md](../dataset_spec_m1.md), [ADR-003](../../architecture/ADR-003-data-split-and-classes.md).
