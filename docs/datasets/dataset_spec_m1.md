# 학습 데이터셋 명세 — `aihub_yolo_v1` (M1)

- 작성: 2026-10-29, 메인 세션. 체크리스트 M1-3 산출물의 명세.
- 수치 출처: `scripts/build_yolo_dataset.py`(AIHub 전체, seed 20261029). 2026-10-29 전체 변환(약 69 s)을 실행했고, 결과가 dry-run과 전부 일치함을 감사로 확인했다 — [audit_aihub_yolo_v1.md](audit_aihub_yolo_v1.md) (판정: 경고, 치명 결함 없음).
- 관련 결정: [ADR-003](../architecture/ADR-003-data-split-and-classes.md), [CHECKLIST "M1 결정 사항"](../CHECKLIST.md), AIHub 전수 집계 [aihub_71327.md](aihub_71327.md).

## 1. 용도와 구성

| 분할 | 용도 | 데이터 |
|---|---|---|
| **train / val** (train_pool, 세션 그룹 5-fold) | 모델 학습·검증, 하이퍼파라미터·epoch 선택, YOLO11m ↔ RF-DETR 비교(M4-3) | AIHub B-scan |
| **test** (AIHub 홀드아웃) | 최종 성능 1회 측정(같은 장비·도로 조건). 학습·튜닝에 쓰지 않는다 | AIHub B-scan, 세션 단위로 고정 |
| **교차 도메인 test** | 다른 장비·지역에서 버티는지 평가(M3-3) | 모로코: 회전·탄성 제외 뒤 그룹당 원본에 가장 가까운 사본 1장(233그룹·233장, 관로·공동·이상 없음), 선택 사본이 전부 224×224(원본 크기 유지)·평가 시 256 letterbox, 공동은 따로 보고(M1-4, [아래 소절](#모로코-평가-세트-morocco_eval_v1)). GPR-SD 관로 수동 라벨(M3-4 이후) |

### 모로코 평가 세트 (`morocco_eval_v1`)

교차 도메인 평가 전용(학습·튜닝 금지). 감사: [audit_morocco_eval_v1.md](audit_morocco_eval_v1.md) (판정: 통과). 출력 `data/morocco_eval_v1/`(git 제외), 매니페스트 `data/splits/morocco_eval_v1.csv`.

| 하위셋 | 그룹 | 이미지 | 박스 | 클래스 | 선택 사본 | 비고 |
|---|---:|---:|---:|---|---|---|
| 관로 | 120 | 120 | 158 | 0 pipe_hyperbola | `_aug_6` | |
| 공동 | 75 | 75 | 81 | 2 cavity | `_aug_13` | `anisotropic_stretch=True`, 따로 집계 |
| 이상 없음 | 38 | 38 | 0 | (빈 라벨) | `_aug_12` | 오검출 평가용 |
| 합계 | 233 | 233 | 239 | | | 전부 224×224 |

- **선택 규칙**: ① 라벨 있는 YOLO 이미지에서 `images_index.csv`의 `aug_transform`이 회전·탄성인 사본을 제외한다. ② 그룹(`split_groups.csv`)마다 "원본과 거의 동일" 사본(`_aug_6`·`_aug_13`·`_aug_12`) 1장을 고른다. ③ 다중 stem 그룹(46개)은 `stems` 열의 첫 stem을 쓴다(결정적, 숫자 최소 stem과 같음). ④ **대체 규칙**: 선호 사본이 없는 그룹은 기하를 보존하는 변환(원본과 거의 동일·주파수 이동·가우시안 잡음·잡음·강한 잡음) 중 stem 순서 → 사본 번호 순으로 첫 장을 쓰고, 그것도 없으면 그룹을 제외하고 `summary.json`에 기록한다. 이번에는 233그룹 전부 선호 사본이 있어 대체 0건.
- **제외 개수**: 회전·탄성 720장 + 그 밖의 사본 1,286장 = 라벨 있는 2,239장 − 선택 233장. 라벨 없는 원본 패치 285장은 쓰지 않는다(index 2,524행 = 2,239 + 285). 제외 번호는 관로 3·5, 공동 10·12, 이상 없음 4·6·10·11(`aug_transform` 열과 대조해 일치).
- **공동**: 원본이 비율 무시 224×224로 늘려져 있어(사본 제외로 해결되지 않음) 공동 지표는 따로 보고하고, 관로·이상 없음과 합산하지 않는다. 클래스는 모로코 pipe → 0, cavity → 2.
- **`eval.yaml`의 `train:` 키 주의**: Ultralytics `check_det_dataset`이 `train` 키를 요구해 `train`·`val`·`test` 모두 `eval_images.txt`를 가리킨다(파일 머리에 "Never train on it" 주석). 학습 설정에는 쓰지 않는다. 분할 열 `split_src`(train 163/val 35/test 35)는 제공 분할의 참고용이다.
- **알려진 사항**: 제공 분할을 넘는 근접 중복 1쌍(`U_233`↔`U_239`, pHash 6), AIHub 학습 데이터와 근접 중복 없음(최소 해밍 6, 육안으로 다른 장면).
- **재생성**: `conda run -n gpr-dev python -I scripts/build_morocco_eval.py` — 옵션 `--out`(기본 `data/morocco_eval_v1`), `--manifest`(기본 `data/splits/morocco_eval_v1.csv`), `--morocco-root`(기본 `configs/paths.yaml` 값), `--dry-run`(개수만 출력). dry-run이 아닐 때 출력 폴더가 비어 있지 않으면 중단.

## 2. 원천과 범위
- 원천: AIHub 「3D 지표투과레이더 탐사 데이터」(dataSetSn 71327, 로컬 폴더 `261.3D 지표투과레이더 탐사 데이터`), Training·Validation zip을 **합쳐서** 다시 나눈다(제공 분할은 세션이 100 % 겹쳐 쓰지 않음).
- 포함: **B-scan만**(C-scan 제외). 이미지 256×256 그레이 JPG(원본 바이트 그대로 복사, 재인코딩 없음).
- 이미지-라벨 짝: zip **멤버 이름** 기준(XML `<filename>` 불일치 1,870건 회피).
- 물리 스케일: 없음(세로축 ns/px 미상) → 이 데이터로는 m 단위 깊이를 내지 않는다.

## 3. 클래스

| ID | 이름 | AIHub 라벨 | B-scan 형상 | 깊이 처리 |
|---|---|---|---|---|
| 0 | `pipe_hyperbola` | `tpipe`(횡배관) | 쌍곡선 | 쌍곡선 피팅 대상 |
| 1 | `pipe_band` | `lpipe`(종배관) | 수평 띠 | 띠 상단 규칙(M2-6) |
| 2 | `cavity` | `cavity`(대칭·비대칭·기타) | — | 없음 |
| 3 | `manhole` | `manh` | — | 없음 |

## 4. 구성 규칙
1. **관로 이미지는 전부** 쓴다.
2. **공동·맨홀은 축소**: test와 train_pool 각각에서, 그 부분집합의 관로 이미지 수와 비슷한 규모(balance-ratio 1.0)로 고른다. gain 제거 crop 키 단위·세션 라운드로빈으로 뽑아 세션 균형을 유지한다.
3. **gain 사본**: 0.25/0.5/1.0 세 벌을 모두 쓴다. 같은 crop 키의 사본은 같은 분할·같은 fold에 둔다(약 27 %의 crop은 사본이 1–2벌뿐). 평가는 crop 단위 1회 집계를 기준으로 하고 이미지 단위 값도 함께 보고한다.
4. **분할 그룹**: `<source>` 세션, 이름에서 유도한 세션, crop 키를 union-find로 묶은 연결 성분(1,801개). `<source>` 불일치 224건 때문에 여러 세션이 한 그룹이 되는 경우가 2개 있다(`summary.json`의 `split_groups_with_several_sessions`; 구현 보고 기준 공동 MA 21세션·NM 5세션, 축소 뒤 선택분에 남은 것은 그룹 1개).
5. **test 선정**: 그룹 단위로 약 15 %. tpipe·lpipe 세션 각 2개 이상, **관로 종류별 test 이미지 비중 10–25 %**. `"{seed}:test:{attempt}"` 스트림으로 attempt를 늘려 가며 처음 만족하는 조합을 채택한다(이번 채택 attempt **0**).
6. **5-fold**: train_pool에 `StratifiedGroupKFold(k=5, shuffle=True, random_state=seed)`, 그룹 = 분할 그룹, 층화 = 이미지의 주 클래스(4수준). 관로 포함 여부로 층화하면 tpipe가 없는 fold가 생겨(test 홀드아웃 전 전체 AIHub 기준, 21개 seed 시험에서 21/21) 주 클래스를 택했다.
7. **누수 단언**: test↔train_pool, fold 상호 간 `session`·`session_from_name`·`crop_key` 교집합 0을 계획 단계와 매니페스트 재읽기 단계에서 두 번 검사한다.
8. **시드**: 20261029(분할·축소·test 탐색 공통). 매니페스트 `seed`·`test_attempt` 열에 기록.

## 5. 개수 (seed 20261029, 전체 변환·감사로 확인)

칸은 `이미지 / 박스 / 세션`. 클래스 칸의 이미지는 그 클래스를 포함한 이미지 수.

| 구분 | 이미지 | 박스 | crop | 세션 | pipe_hyperbola | pipe_band | cavity | manhole | gain 0.25/0.5/1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 전체 | 16,887 | 19,967 | 6,243 | 1,802 | 1522/1522/11 | 4106/6193/13 | 5630/5630/1482 | 5661/6622/395 | 5625/5626/5636 |
| **test** | 2,086 | 2,504 | 770 | 271 | 165/165/2 | 530/796/2 | 696/696/221 | 712/847/61 | 699/691/696 |
| **train_pool** | 14,801 | 17,463 | 5,473 | 1,531 | 1357/1357/9 | 3576/5397/11 | 4934/4934/1261 | 4949/5775/334 | 4926/4935/4940 |
| fold 0 | 3,525 | 4,938 | 1,298 | 308 | 185/185/3 | 1367/2649/2 | 986/986/253 | 987/1118/67 | 1181/1184/1160 |
| fold 1 | 3,210 | 3,625 | 1,186 | 308 | 125/125/2 | 1111/1370/1 | 987/987/252 | 987/1143/67 | 1064/1064/1082 |
| fold 2 | 2,901 | 3,069 | 1,074 | 304 | 260/260/1 | 667/700/2 | 987/987/251 | 987/1122/67 | 967/953/981 |
| fold 3 | 2,490 | 2,752 | 922 | 307 | 405/405/1 | 112/166/4 | 987/987/253 | 986/1194/66 | 823/836/831 |
| fold 4 | 2,675 | 3,079 | 993 | 304 | 382/382/2 | 319/512/2 | 987/987/252 | 1002/1198/67 | 891/898/886 |

- test 관로 이미지 비중: tpipe 165/1,522 = **10.8 %**, lpipe 530/4,106 = **12.9 %**.
- fold k의 학습 = 나머지 4 fold 합, 검증 = fold k. 최종 test 학습은 train_pool 전체(5 fold 합).

### test 관로 세션
| 클래스 | 세션(이미지 수) |
|---|---|
| tpipe | `NM_2022-10-19_(12_45_21)` (9), `NM_2022-10-19_(12_47_20)` (156) |
| lpipe | `NM_2022-10-19_(11_03_51)` (173), `NM_2022-10-19_(12_47_20)` (357) |

공동·맨홀 포함 test 세션 전체는 `summary.json`의 `test_sessions`, 매니페스트 `subset=test` 행.

## 6. 파일 구성(`data/aihub_yolo_v1/`, git 제외)
| 파일 | 내용 |
|---|---|
| `images/*.jpg`, `labels/*.txt` | 원본 JPG, YOLO 라벨(정규화 xywh, 6자리) |
| `fold{0..4}_images.txt`, `fold{k}.yaml` | fold 목록, Ultralytics 설정(train=나머지 4 fold, val=fold k, names 4개) |
| `test_images.txt`, `test.yaml` | test 목록, 설정(train=5 fold 전체, val·test=test) |
| `summary.json` | 클래스·fold·세션 분포, 축소 개수, `test_selection`(attempt, 비중, test 관로 세션) |
| `data/splits/aihub_yolo_v1.csv` | 매니페스트: `file_name, member, split_src, session, session_from_name, crop_key, gain, classes, n_boxes, subset, fold(test=-1), seed, test_attempt, category, primary_class, split_group` |

재생성: `conda run -n gpr-dev python -I scripts/build_yolo_dataset.py` (기본 seed·제약). 개수만 보기: `--dry-run`.

| CLI 옵션 | 기본값 | 의미 |
|---|---|---|
| `--out` | `data/aihub_yolo_v1` | 출력 폴더(비어 있지 않으면 중단) |
| `--manifest` | `data/splits/<out 이름>.csv` | 매니페스트 경로 |
| `--aihub-root` | `configs/paths.yaml` 값 | AIHub 루트 |
| `--seed` | 20261029 | 분할·축소·test 탐색 공통 시드 |
| `--k` | 5 | train_pool fold 수 |
| `--balance-ratio` | 1.0 | 공동·맨홀 축소 비율(관로 이미지 수 대비) |
| `--test-fraction` | 0.15 | test 세션 비율 |
| `--min-test-pipe-sessions` | 2 | test 관로 종류별 최소 세션 |
| `--pipe-share-min` / `--pipe-share-max` | 0.10 / 0.25 | test 관로 이미지 비중 범위 |
| `--max-test-attempts` | 1000 | test 탐색 최대 시도 |
| `--limit` | 없음 | zip마다 앞 N개 라벨만(시험용) |
| `--dry-run` | 끔 | 라벨만 읽고 summary를 stdout에 |

## 7. 알려진 한계와 해석 주의
- **관로 세션이 적고 fold에 쏠린다**(NM 장비만). lpipe는 fold 0·1이 각각 세션 하나(1,281장·1,111장)에 지배되고 fold 3은 112장뿐이다. tpipe는 fold 2·3이 세션 1개다. fold별 관로 지표는 "세션 1–2개 기준"으로 읽고, fold 평균±표준편차로 판단한다.
- **test 관로도 세션 2–3개**(같은 날 NM 세션)다. test 관로 지표는 신뢰구간이 넓다.
- 혼합 클래스 이미지 32장(lpipe + manh): 관로 crop으로 처리돼 전부 포함되며, manhole 이미지 수가 목표보다 약간 많다.
- `<filename>` 불일치 1,870건은 멤버 이름으로 짝지었다. 라벨 내용이 실제 이미지와 맞는지는 감사 문서에서 확인한다.
- 축소한 공동·맨홀은 가용분의 약 15–20 %만 쓴다(train_pool 공동 4,934/31,635, 맨홀 4,934/25,762). 오검출이 문제가 되면 balance-ratio를 키운 v2를 만든다.
- 모로코(교차 도메인 test)는 공동 하위셋 전체가 비율 무시 224×224로 늘려져 있어 공동 지표를 도메인별로 따로 본다.
- **test도 클래스 균형 축소를 거친 표본**이라 자연 분포가 아니다. test의 precision은 실제 현장보다 낙관적으로 나올 수 있다(공동·맨홀이 관로와 비슷한 수로 줄어 있음).
- **분할 경계의 근접 중복**(pHash, 같은 gain끼리): 해밍 0인 쌍 15개 중 3개, 해밍 ≤2인 쌍 86개 중 21개(test↔train_pool 3개)가 분할을 넘는다. 모두 세션이 달라 세션 키 누수는 아니며, ≤2 경계 쌍은 대부분 `MA_2016…` 맨홀끼리다. 목록: `analysis/data/out/phash_cross_le4.csv`.

## 8. 감사 결과 요약 (2026-10-29)
| 검사 | 결과 |
|---|---|
| 누수 | test↔train_pool, fold 상호 간 `session`·`session_from_name`·`crop_key`·`split_group` 교집합 0(64개 조합, 매니페스트로 독립 재계산) |
| 짝·형식 | images 16,887 = labels 16,887, 짝 없음·빈 라벨·형식 오류 0, 클래스 0–3만, 좌표 범위 밖 0, 퇴화 박스 0, 전수 256×256 |
| Ultralytics | `check_det_dataset`로 fold0–4·test yaml 모두 읽힘(nc=4). fold k train/val = 11,276/3,525 · 11,591/3,210 · 11,900/2,901 · 12,311/2,490 · 12,126/2,675, test.yaml 14,801/2,086 |
| `<filename>` 불일치 | 이번 데이터에 427건, 그중 418건은 `.jpg` 확장자 유무만 다름(맨홀), 실제로 이름이 다른 것 9건(공동). 무작위 20건 + 9건 전수 몽타주에서 박스가 이미지 내용과 맞음 |
| 혼합 클래스 | 32장 전부 lpipe+manhole: `NM_2022-10-19_(11_03_51)` 17장(test), `(12_33_02)` 15장(fold 4) |
