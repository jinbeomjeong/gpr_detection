<!-- 원본: C:\workspace\GPR_Data\모로코_관로공동GPR영상_가공본_261016\README_사용법.docx | docx_dump(R1)·python-docx로 추출, 2026-10-29. 원문 그대로 옮김(서식만 Markdown으로 변환). -->

# 모로코 GPR 관로·공동 영상 데이터셋 — 가공본 사용 설명서

- 작성: 데이터 가공본 제공처 (공개용으로 과제 정보 삭제)
- 작성일: 2026-10-16
- 원본: *Intelligent recognition of subsurface utilities and voids: A Ground Penetrating Radar dataset for Deep Learning applications* — Mojahid 외(모로코), Mendeley Data 2024, DOI 10.17632/ww7fd9t325.1, 논문 Data in Brief 59(2025) 111338
- 원본 다운로드: https://data.mendeley.com/datasets/ww7fd9t325/1 (처리 스크립트: https://github.com/LCSkhalid/GPR_Data)
- 라이선스: CC BY 4.0(원본). 가공본도 같은 조건으로 사용·수정·재배포할 수 있으며 **출처 표시가 필수**입니다(인용 문구는 9절)

## 1. 한 줄 요약

모로코 여러 도시의 공사현장(2019\~2024)에서 GSSI 400 MHz·200 MHz 안테나로 찍은 B-scan 화면을 224×224 패치로 잘라 관로(pipe)·공동(cavity) 박스를 단 영상 탐지용 데이터입니다. 이 가공본은 ① 두 하위셋에서 모두 0번이던 클래스를 관로 0·공동 1로 나누고, ② 퇴화 박스 6개를 빼고, ③ 같은 프로파일에서 나온 패치를 한 그룹으로 묶어 학습·검증·시험으로 나누고, ④ YOLO 학습 폴더(data.yaml 포함)로 바로 쓸 수 있게 정리한 것입니다.

가장 큰 가치: 라벨이 이미 있어 탐지 모델(YOLO 등) 학습에 바로 넣을 수 있고, 국내 자료에 적은 공동(cavity) 박스 600개가 있습니다.

가장 큰 한계: **물리 축척이 전혀 없습니다.** 화면 캡처를 자른 영상이라 시간(ns)·깊이(m)·거리(m)로 바꿀 수 없고 원시 파형도 없습니다. 깊이 추정에는 쓸 수 없고 박스 탐지 학습·검증 보조용입니다. 또 고유 영상은 285장뿐이고 나머지는 이를 6\~12배 증강한 것입니다.

## 2. 폴더 구성

```
모로코_관로공동GPR영상_가공본_261016\
├ README_사용법.docx              ← 이 문서
├ 00_원본참고\                     원 저자 GitHub의 README.md·crop.py·augmentation.py·dataset.png
├ 01_메타데이터\
│   ├ images_index.csv            ★ 영상 2,524장(증강 2,239 + 원본 285)의 새 이름·하위셋·증강 종류·크기·박스 수·그룹·분할
│   ├ boxes.csv                   ★ 박스 1,607개(퇴화 6개 포함, 표시)의 클래스·좌표
│   └ split_groups.csv            분할 그룹 233개와 묶은 사유
├ 02_참고이미지\                   미리보기 PNG 5장
├ 03_yolo_dataset\                ★ 바로 학습용: images\{train,val,test}, labels\{train,val,test}, data.yaml
├ 04_검토용_박스표시\{하위셋}\     {stem}.png — 원본 1장 + 그 증강본 전부를 박스와 함께 한 장에(사람 확인용)
├ 05_원본패치\{하위셋}\            증강 전 원본 285장(원래 파일명, 라벨 없음, YOLO 폴더에는 넣지 않음)
└ 06_scripts\                      build_morocco_package.py(이 패키지 생성), example_load.py
```

- 새 영상 이름: {U|C|I}_{원본 stem}_aug_{번호} (U 관로·C 공동·I 이상 없음). 예: U_001_aug_1, C_10_aug_7, I_1-2_aug_3(원본 1 (2)_aug_3)
- 하위셋: utilities(관로), cavities(공동), intact(이상 없음 — 빈 라벨 = 배경 샘플)

## 3. 데이터 규모와 분할

| **분할** | **영상** | **관로 / 공동 / 이상 없음 영상** | **관로 박스** | **공동 박스** |
|---|---|---|---|---|
| train | 1,598 | 570 / 392 / 636 | 703 | 418 |
| val | 329 | 108 / 77 / 144 | 145 | 90 |
| test | 312 | 108 / 84 / 120 | 154 | 91 |
| 합계 | 2,239 | 786 / 553 / 900 | 1,002 | 599 |

- 고유 원본: 관로 패치 131장(근접 중복을 합치면 120그룹), 공동 79장(75그룹), 이상 없음 75장(38그룹)
- 분할은 **그룹 단위**(약 70/15/15, 하위셋별)이므로 같은 원본의 증강본이나 같은 프로파일을 조금씩 밀어 자른 패치가 학습과 검증에 동시에 들어가지 않습니다. 파일 단위로 다시 섞으면 검증 점수가 부풀려지니 주의하세요
- 영상 크기: 224×224가 2,089장, 이상 없음의 _aug_3은 268×268, _aug_9는 336×336(각 75장). 전부 8비트 1채널 그레이

## 4. 라벨

- 클래스: 0 pipe(원본 VOC 이름 Utility), 1 cavity(원본 cavities). 원본 YOLO txt는 두 하위셋 모두 0번이라 합치면 구분이 안 돼서 다시 매겼습니다
- 좌표: 원본 YOLO txt(소수점 좌표)를 썼습니다. VOC xml과 1,607개 박스 전부 매칭 대조해 최대 차이 0.5 px(정수 반올림 차)임을 확인했습니다
- 제외: 폭 또는 높이가 3 px 미만인 퇴화 박스 6개(U_128_aug_1·U_204_aug_1은 높이 0, U_222_aug_1·U_232_aug_1·U_265_aug_1은 높이 1\~2 px, C_34_aug_10은 0.4 px)는 YOLO 라벨에서 뺐고 boxes.csv에는 degenerate_excluded=예로 남겨 두었습니다
- 원본 패치(05)에는 라벨이 없습니다. 관로 _aug_6, 공동 _aug_13은 원본과 거의 같은 영상이라 그 박스를 원본 라벨 대신 쓸 수 있습니다(공동은 224 기준이라 원본 크기로 비율 환산 필요)
- 박스는 증강본마다 사람이 따로 그린 것이라, 같은 원본의 증강본끼리도 박스 수·위치가 다를 수 있습니다(04 검토 시트에서 확인 가능)
- 박스 1,607개 중 639개는 영상 경계에 닿아 있습니다(쌍곡선이 패치 밖으로 잘린 경우). touches_border 열로 걸러 볼 수 있습니다

## 5. 증강 종류(images_index.csv `aug_transform`)

| **하위셋** | **번호** | **변환** |
|---|---|---|
| 관로 | 1\~6 | 가우시안 잡음 / 가로 이동 +20px / 회전 +15° / 좌우 반전 / 탄성 변형 / 주파수 이동(원본과 거의 동일) |
| 공동 | 7\~13 | 가우시안 잡음 / 가로 이동 +20px / 중앙 1.5배 확대 / 회전 +15° / 좌우 반전 / 탄성 변형 / 주파수 이동(원본과 거의 동일) |
| 이상 없음 | 1\~12 | 잡음 / 가로 이동 +10px / 268 리사이즈 / 회전 +15° / 좌우 반전 / 탄성 변형 / 강한 잡음 / 가로 이동 +20px / 336 리사이즈 / 회전 −15° / 탄성 변형 / 원본과 거의 동일 |

- 변환 종류는 원본과 증강본을 상관 대조해 확인한 것입니다(원 저자 augmentation.py의 목록과 이상 없음 쪽은 다름)
- 공동은 원본(크기 제각각)을 가로세로 비율을 무시하고 224×224로 늘린 뒤 증강했으므로 쌍곡선 모양이 비등방으로 찌그러져 있습니다
- 학습 시 각자 증강을 다시 걸 계획이면 aug_transform이 "원본과 거의 동일"인 영상(관로 _aug_6, 공동 _aug_13, 이상 없음 _aug_12)만 골라 쓰는 방법도 있습니다

## 6. 분할 그룹(split_groups.csv)

| **묶음** | **내용** |
|---|---|
| 관로 근접 중복 5군 | 015·173 / 177·179·181·183·185·186·188·190(같은 쌍곡선을 5 px씩 밀어 자른 8장) / 192·194 / 263·266 / 289·301 |
| 공동 근접 중복 4쌍 | 18·23 / 32·79 / 42·77 / 43·78(이동 허용 템플릿 상관 0.95 이상) |
| 이상 없음 37쌍 | N.JPG와 N (2).JPG(같은 프로파일을 다르게 자른 것) |
| 나머지 | 원본 1장 = 1그룹 |

## 7. 사용 예

### 7-1. YOLO 학습(ultralytics)

```
yolo detect train data="<패키지폴더>\03_yolo_dataset\data.yaml" model=yolov8n.pt imgsz=224 epochs=100
```

data.yaml의 경로는 이 파일이 있는 폴더 기준이라 폴더를 옮겨도 그대로 쓸 수 있습니다.

### 7-2. 라벨 읽기

```
import cv2, numpy as np
iid, sp = 'U_001_aug_1', 'train'          # 분할은 images_index.csv의 split 열
img = cv2.imdecode(np.fromfile(f'03_yolo_dataset/images/{sp}/{iid}.jpg', np.uint8), cv2.IMREAD_GRAYSCALE)
for line in open(f'03_yolo_dataset/labels/{sp}/{iid}.txt'):
    cls, xc, yc, w, h = line.split()      # 0~1 정규화, cls 0 관로 / 1 공동
```

06_scripts\example_load.py \<패키지폴더\> U_001_aug_1을 실행하면 박스를 그려 보여 줍니다. 한글 경로에서는 cv2.imread 대신 위처럼 cv2.imdecode를 쓰세요.

## 8. 권장 활용 순서

1. 04_검토용_박스표시에서 관로·공동 시트를 몇 장 열어 박스 품질과 증강 모양을 확인
2. 03_yolo_dataset으로 기준 모델을 학습해 val/test 성능 확인(그룹 분할 유지)
3. 국내 자료(AI Hub 260·261)·4TU·광저우대 가공본과 섞거나 사전학습용으로 써서 교차 도메인 성능 비교
4. 국내 실측 GPR 영상으로 미세조정·검증. 이 데이터는 GSSI 단채널 화면 캡처라 국내 장비 영상과 질감이 다르므로 단독 평가셋으로는 쓰지 않는 것을 권장

## 9. 라이선스·인용

- 라이선스: Creative Commons Attribution 4.0 International(CC BY 4.0). 출처 표시와 변경 사실 명시 조건으로 복제·수정·상업 이용 가능. 이 가공본은 클래스 재매핑·퇴화 박스 제거·파일명 변경·분할을 한 변경본입니다
- 데이터 인용: Mojahid, A., El Ouai, D., El Amraoui, K., El Hami, K., Aitbenamer, H. (2024). Intelligent recognition of subsurface utilities and voids: A Ground Penetrating Radar dataset for Deep Learning applications. Mendeley Data, V1, doi:10.17632/ww7fd9t325.1
- 논문 인용: Mojahid, A. et al. (2025). Intelligent recognition of subsurface utilities and voids: A ground penetrating radar dataset for deep learning applications. Data in Brief, 59, 111338. doi:10.1016/j.dib.2025.111338
- 00_원본참고의 GitHub 스크립트는 저장소에 라이선스 표기가 확인되지 않았으니 참고용으로만 쓰세요

## 10. 문의

가공본 제공처에 문의(공개용으로 연락처 삭제).
