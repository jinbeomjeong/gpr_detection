# 모로코 평가 세트 `morocco_eval_v1` 감사 리포트

- 작성: 2026-10-29, gpr-dataset-curator (audit + card, 체크리스트 M1-5 모로코 분)
- 대상: `data/morocco_eval_v1/`(images 233, labels 233, `eval_images.txt`, `eval.yaml`, `summary.json`), 매니페스트 `data/splits/morocco_eval_v1.csv`. 생성 코드 `scripts/build_morocco_eval.py`, `src/gpr_detection/data/morocco_eval.py`는 읽기만 했다.
- 독립 대조 근거: 원본 `01_메타데이터/images_index.csv`(`aug_transform`), `split_groups.csv`, 원본 `03_yolo_dataset` 이미지·라벨. 프로젝트 코드는 import하지 않았다.
- 감사 스크립트(`analysis/data/`): `audit_morocco_eval_v1.py`, `audit_morocco_eval_v1_pairs.py`, `audit_morocco_eval_v1_yamlcheck.py`. 중간 산출물은 `analysis/data/out/morocco_eval_audit.json`, `morocco_eval_phash_internal_le8.csv`, `morocco_eval_vs_aihub_nearest.csv`.
- 실행: `conda run -n gpr-dev python -I <스크립트>` (단일 프로세스, 시드 20261029는 몽타주 표본 추출에만 사용).

## 판정: 통과 (참고 사항 2건)

| 검사 | 대상 | 결과 |
|---|---|---|
| 그룹당 1장 | split_groups 233그룹(관로 120·공동 75·이상 없음 38) | 매니페스트 233행 = 고유 그룹 233 = 고유 파일 233, 누락·초과 그룹 0 |
| 회전·탄성 사본 | 매니페스트 233행 `aug_transform` | 0장 (index 전체 회전·탄성 = 720장: 관로 131×2, 공동 79×2, 이상 없음 75×4) |
| 제외 번호와 `aug_transform` 대조 | index의 회전·탄성 열 | 관로 {3,5}, 공동 {10,12}, 이상 없음 {4,6,10,11}로 지시문의 번호와 일치 |
| 선택 사본 | 매니페스트 vs index | 관로 `_aug_6` 120, 공동 `_aug_13` 75, 이상 없음 `_aug_12` 38. 전부 "원본과 거의 동일" 변환. index의 변환·번호·그룹·분할·크기와 불일치 0 |
| 선택 규칙 | selection_rule | 233 전부 preferred, fallback 0 (index에서 "원본과 거의 동일" 번호는 하위셋당 하나뿐: 관로 6 / 공동 13 / 이상 없음 12) |
| 동률(다중 stem) 그룹 | 46그룹(관로 5·공동 4·이상 없음 37) | 선택 stem은 `split_groups.csv` `stems` 열의 **첫 stem** 46/46(해당 stem에 선호 사본 존재). 숫자 최소 stem과도 46/46 일치(차이 0). 후보 순서가 CSV 위치→사본 번호라 set·해시 순서에 의존하지 않아 결정적 |
| 이미지 SHA256 | 233장 | 원본과 불일치 0 (재인코딩 없음) |
| 이미지 크기·모드 | 233장 | 233장 전부 224×224 그레이(L) |
| images/labels/목록 | 폴더·`eval_images.txt` | images 233 = labels 233 = 목록 233 줄, 목록 경로 전부 존재 |
| 라벨 클래스 | 박스 239 | 관로 158 전부 클래스 0, 공동 81 전부 클래스 2, 이상 없음 38장 전부 빈 파일 |
| 박스 수 | 233장 | 원본 YOLO 라벨·변환본·매니페스트 `n_boxes` 불일치 0, 좌표값도 원본과 차이 0건(클래스만 0→0, 1→2로 재매핑) |
| 좌표 범위·이미지 밖 | 박스 239 | 0–1 밖 0, 이미지 밖 0 |
| 퇴화 박스(폭·높이 < 3 px) | 박스 239 | 0 (최소 폭·높이: 관로 18.2·26.9 px, 공동 37.7·18.8 px) |
| Ultralytics | `check_det_dataset(eval.yaml)` | 읽힘, nc=4, names 0–3 = pipe_hyperbola·pipe_band·cavity·manhole, val 233줄 |
| 근접 중복(평가 세트 내부, pHash 64비트) | 233장, 27,028쌍 | 해밍 ≤4: 0쌍, ≤6: 1쌍, ≤8: 2쌍 |
| 근접 중복(AIHub 16,887장 ↔ 평가 233장) | 233 × 16,887 | 최소 해밍 6. ≤4: 0, ≤6: 2, ≤8: 5, ≤10: 20, ≤12: 53 (중앙값 14) |

## 구성

| 하위셋 | 그룹 | 이미지 | 박스 | 빈 라벨 | 제공 분할 train/val/test(이미지) |
|---|---:|---:|---:|---:|---|
| 관로(utilities) | 120 | 120 | 158 | 0 | 84 / 18 / 18 |
| 공동(cavities) | 75 | 75 | 81 | 0 | 52 / 11 / 12 |
| 이상 없음(intact) | 38 | 38 | 0 | 38 | 27 / 6 / 5 |
| 합계 | 233 | 233 | 239 | 38 | 163 / 35 / 35 |

- 개수 흐름: 라벨 있는 YOLO 이미지 2,239 − 회전·탄성 720 − 기타 사본 1,286 = 233. 원본 패치 285장(라벨 없음)은 쓰지 않았다(index 2,524행 = 2,239 + 285).
- 제공 분할 열 `split_src`는 참고용이며 이 세트는 전체가 평가 전용이다.

## 참고 사항

1. **근접 중복 1쌍이 제공 분할을 넘는다(영향 작음)**: `U_233_aug_6`(val) ↔ `U_239_aug_6`(train), 해밍 6. 눈으로 보면 같은 쌍곡선 두 개가 보이는 거의 같은 영상이다(제공 그룹은 서로 다른 그룹으로 처리됨). 이 세트는 학습에 쓰지 않으므로 누수는 아니고, 평가 세트 안에서 사실상 같은 표본이 2장 들어 있는 것이다. 다른 1쌍(`U_156` ↔ `U_195`, 해밍 8)은 서로 다른 영상이다. 몽타주: `docs/datasets/figures/morocco_eval_v1_phash_pairs_montage.png`(원본 이미지 포함, 로컬 보관·공개 저장소 제외). 제외하지 않고 남겨 둔다(그룹 규칙은 제공 분할을 그대로 따른다).
2. **AIHub ↔ 모로코 근접 중복은 없다**: 최소 해밍 6의 2쌍(`C_68`·`C_70`과 RA 장비 이미지)과 8의 쌍들을 육안으로 확인했다. 모로코 쪽은 강한 줄무늬 영상이고 AIHub 쪽은 거의 회색이거나 다른 장면이라 같은 신호가 아니다(pHash가 큰 단색·줄무늬 영상에서 가깝게 나오는 우연). 교차 도메인 가정(장비 상이)과 일치한다.

## 관찰·주의

- **`eval.yaml`의 `train:` 키**: Ultralytics `check_det_dataset`은 data yaml에 `train`(과 `val`/`test` 중 하나)이 없으면 오류를 낸다. 그래서 `train`, `val`, `test`가 모두 `eval_images.txt`를 가리킨다. 파일 머리 주석에 "Never train on it: 'train' points at the eval list only because Ultralytics requires the key."가 있다(경고 문구 있음, 확인함). 학습 설정은 `aihub_yolo_v1`의 `fold*.yaml`·`test.yaml`만 쓰고 이 yaml은 `val`/`test` 평가 호출에만 쓴다.
- **관로 영상 위쪽의 흰 세로선**: 관로 하위셋 영상 상단에 흰 세로선 두 줄이 보이는 영상이 많다(몽타주 `docs/datasets/figures/morocco_eval_v1_samples_montage.png`, 원본 이미지 포함·공개 제외). 원본 패치의 표식으로 보이며, 검출 모델이 이 선을 단서로 삼을 수 있는지는 평가 때 확인 필요(AIHub 영상에는 없음). gpr-processing-expert 확인 후보.
- **공동 하위셋은 비등방 늘림**: 원본이 가로세로 비율 무시 224×224로 늘려져 있어 쌍곡선 모양이 찌그러져 있다. 사본 제외로 해결되지 않으므로 공동 지표는 따로 보고한다(매니페스트 `anisotropic_stretch=True` 75장).
- 이상 없음 38그룹은 빈 라벨(배경)이라 오검출 평가에만 쓸 수 있다(박스 지표 없음). 하위셋 클래스 구성이 단순(관로=클래스 0, 공동=클래스 2만)해서 하위셋별 AP를 따로 내야 한다.
- 사례 몽타주: `docs/datasets/figures/morocco_eval_v1_samples_montage.png`(하위셋별 6장, 박스 겹침 확인 — 박스가 쌍곡선·반사 위에 놓임).
- 이 감사는 라벨 정확도(박스가 실제 대상인지) 전수 검수가 아니라 표본 18장 육안 확인이다. 박스는 증강본마다 사람이 따로 그린 것이라 원본 대비 위치 오차가 있을 수 있다(`readme_morocco.md` §4).
