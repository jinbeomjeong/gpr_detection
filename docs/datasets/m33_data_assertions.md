# M3-3 X7 데이터 재단언

- 작성: 2026-11-15, gpr-dataset-curator. 계획: `docs/experiments/m33_cross_domain_plan.md` v0.2.2 §7 X7 (저장소 HEAD 9b28694).
- 판정: **통과** (모로코·4TU·광저우·AIHub test 모두 단언 충족, 불일치 0). 참고 사항 2건(근접 중복 1쌍, 공동 하위셋 비등방 늘림)은 아래에 공개한다.
- 스크립트(`analysis/data/`, `conda run -n gpr-dev python -I`, 단일 프로세스·E코어 affinity·BelowNormal, 난수 없음): `m33_x7_assertions.py`(sha 27328b41…, 본 단언), `m33_assert_no_aihub_test.py`(sha 358a19f9…, AIHub test 단언). 산출물(git 제외): `data/audit/m33_x7/x7_assertions.json`, `morocco_eval_sha.csv`, `fourtu_keep_one_dropped_lines.csv`.
- **정답 값은 열지 않았다.** 모로코 라벨 파일은 sha·바이트 크기만(빈 파일 수 = 크기 0 개수), 4TU 깊이표·`utilities_groundtruth.csv`는 열지 않았고(M3-3 계획이 정한 원 바이트 sha는 [U3]에 기록됨), `known_apexes.csv`는 sha와 행 수·고유 측선 수만 셌다(좌표 출력·저장 없음). 이미지·레이더그램 npy는 해시·크기·pHash에 읽었다.

## 1. 모로코 평가 세트 (`data/morocco_eval_v1`, 근거 `docs/datasets/audit_morocco_eval_v1.md`)

| 단언 | 결과 |
|---|---|
| 매니페스트 행 = 디스크 이미지 = 라벨 = 목록 줄 | 233 = 233 = 233 = 233 |
| 이미지 sha256: 평가 사본 ↔ 원본 `03_yolo_dataset` | 불일치 0 (233장, 재인코딩 없음) |
| 금지 증강 사본(`images_index.csv`의 `aug_transform`에 "회전" 또는 "탄성") | **0장**. 매니페스트 열 기준도 0, 매니페스트 `aug_transform`과 index 불일치 0, 그룹·분할 불일치 0, index에 없는 행 0 |
| index 전체 회전·탄성 | 720장. 번호: 관로 {3,5}, 공동 {10,12}, 이상 없음 {4,6,10,11} (지시 번호와 일치) |
| 선택된 사본 번호 | 관로 {6}, 공동 {13}, 이상 없음 {12}. 금지 번호와 교집합 0 |
| 그룹당 1장 | `split_groups.csv` 233그룹 = 매니페스트 고유 그룹 233. 1장이 아닌 그룹 0, 양방향 누락 0 |
| 하위셋 | 관로 120 / 공동 75 / 이상 없음 38. 제공 분할 참고 열 train 163 / val 35 / test 35 |
| 빈 라벨 파일(크기 0) | 38 (이상 없음 38장과 같음) |

### sha 목록 (SHA-256, 원 바이트)
| 파일 | sha256 |
|---|---|
| `data/splits/morocco_eval_v1.csv` | `b140c7b465e8bf4fd0dd1701356862984531176578c1c9482b9caca2c0562327` |
| `data/morocco_eval_v1/eval_images.txt` | `0e91de3529635f8d1df7903c840c68df63a7cbb2853072fd7b81a1bf484eaa1a` |
| `data/morocco_eval_v1/eval.yaml` | `7f27f94d0995ed2fc2305d9c7a06b3bbf26b3ba69e83226bb8a96ad171da5c4f` |
| `data/morocco_eval_v1/summary.json` | `9fbaeb52859121e283fdaa47525fdaa1d901d322b49985dae8b7f10da68ee95e` |
| 원본 `01_메타데이터/images_index.csv` | `0f8a66d3e4045166c255202e8e2409374c2631589cd7dee472e85dc02c5b0f17` |
| 원본 `01_메타데이터/split_groups.csv` | `93fec4f52eff03643e97d8da31b5679b731ce5229cdb2a27b7255dac6bbe19a4` |
| 이미지 233장 개별 sha 표 `data/audit/m33_x7/morocco_eval_sha.csv`(파일명·평가 sha·원본 sha·일치·라벨 sha·라벨 바이트) | `bd6927f61c71d068fe334c974a0599afbb2374bb876359bf4e536cbbda154084` |
| 이미지 sha 233개를 파일명 순으로 줄바꿈 연결한 것의 sha256 | `95ad76444681d9b82e3a70bf66e48b05e2800df72e3c6643de348c998c448958` |
| 라벨 sha 233개 같은 방식 | `12b0fbe2f2bf0cc5d191f830e9c10da622543f361f3669043e1e7081b58ef7f3` |

### 공개 사항
1. **근접 중복 1쌍(U_233·U_239)**: `U_233_aug_6`(제공 분할 val) ↔ `U_239_aug_6`(train), pHash 해밍 **6**. 재계산 결과 233장 내부 해밍 ≤8은 정확히 2쌍이다: 이 쌍과 `U_156_aug_6` ↔ `U_195_aug_6`(해밍 8, 서로 다른 영상, 감사 문서 판정). ≤4는 0쌍. 제외하지 않았다. 계획 §2-1 민감도 "근접 중복 쌍을 하나로 센 값"이 이 쌍(U_233·U_239) 하나만 대상이다.
2. **공동 하위셋 75장은 224×224 비등방 늘림**(`anisotropic_stretch=True` 75, 관로·이상 없음은 0). 매니페스트의 `subset == cavities` 또는 `anisotropic_stretch`로 도메인을 분리한다. 사본 제외로는 해결되지 않으므로 공동 지표는 전체 지표에 합치지 않고 따로 낸다(계획 §2-1과 같음). 박스는 증강본마다 사람이 따로 그린 것이라 원본 대비 위치 오차가 있을 수 있다(`readme_morocco.md` §4, 감사 문서와 같음).

## 2. 4TU

### 2-1. 측선 집합 (754 / 720)
- 목록: `docs/datasets/fourtu_duplicate_radargrams.csv` sha256 `6cdef12c55fe49ec1d865f0f513485b4dd51cfbe4e81b9171956b22821430552` (39쌍: `across_activities` 34 + `within_activity` 5). 사전 등록 [U3] `docs/processing/preregistration_4tu_reeval_v1.md` sha256 `6f9f638548d3ac761ce7c516833ea83179492a6459894de85bf71fda3a655eba`(작업 사본 바이트, 이 문서 작성 시점. [U3] §1-2의 LF 정규화·blob id 대조 규칙은 별도).
- 원본 `01_메타데이터/radargrams_index.csv` sha256 `28a550e190cbba31dc8d504d3a29a80779eec1b544ee73df07223e03fcc6b106` (759행, [U3]이 기록한 값과 같음).
- **독립 재계산**: `05_radargrams_npy` 759개 전부 SHA-256 해시 → 바이트 동일 그룹 39개(전부 크기 2) = 목록 39쌍. 목록 39쌍의 `sha256_npy`가 양쪽 파일 모두와 일치 39/39. **목록에 없는 바이트 동일 파일 0**(목록 누락 0).
- `keep_both` = 759 − 활동 안 사본 5(쌍의 `file_b`) = **754**.
- `keep_one` = 754 − 34 = **720**. [U3]의 규칙(한쪽 측선 집합이 엄격히 포함하면 큰 쪽, 아니면 활동 번호가 작은 쪽)을 `scripts/eval_depth_4tu.py`를 쓰지 않고 다시 구현해 계산했다: 활동 쌍 결정 7건(동일 집합 4, 엄격 포함 1, 번호 작은 쪽 2), 제거 측선 34, **측선이 남지 않아 평가에서 빠지는 활동 5개: 02.4, 03.5, 010.12, 010.16, 013.2**(`tests/test_eval_depth_4tu.py`의 기대값과 같음). 제거 측선 목록 `data/audit/m33_x7/fourtu_keep_one_dropped_lines.csv` sha256 `5ba2abcd0466e1af9cd8c7b50947efcab10428c3c51030a25771e8032561a54b`. 이 계산은 파일 해시·측선 목록·활동 번호만 썼고 정답 깊이는 쓰지 않았다.

### 2-2. K 꼭짓점 정본
| 파일 | sha256 | 확인 |
|---|---|---|
| `data/processing/d3/known_apexes.csv` | `0fec12dba4efa504a69a8319e52eb854dac6359296beb391f6b83b177d49db61` | 41행(K 꼭짓점 41개), 고유 측선 **27** |
| `data/processing/d3/lines.csv` | `f6fe08a93808555727acb3ecd864abc1cd555f0c6900a753a8c598be4543ca0a` | 59행. `bundles` 열: K 26, S+K 1, S 25, X 4, L 3. **K를 가진 측선 26 + 1 = 27**로 `known_apexes.csv`의 고유 측선 27과 맞는다 |

주의: `lines.csv`는 27측선 목록이 아니라 D3의 평가 측선 59개 표다(K 27은 그 일부). 지시문의 "27측선의 정본은 두 파일"은 이렇게 읽으면 맞다.

## 3. 광저우대 IDS 60측선
| 파일 | sha256 | 확인 |
|---|---|---|
| `analysis/processing/m32_stageA_lines_ids.csv` | `a52e467dfed80e64885b3ced9e5fb5e9e9617305b39e7d7e1d44e4d1f5bedfb3` | 60행(열 line, antenna_MHz, time_window_ns), 고유 60 |
| 원본 `광저우대…\01_메타데이터\radargrams_index.csv` | `964df6fcdce3a0c88f5c4ba5868fc417e3f62c9c66ad9ee50e4f1520353d887e` | 648행(IDS 573 + MALA 75) |

- 목록 60개가 전부 index의 `file_id`에 있다(60/60). 해당 행의 `format`은 전부 `IDS .dt`, `target`은 전부 `pipe`다. 안테나 MHz: 400 53, 2000 4, 200 1, 600 1, 900 1.

## 4. AIHub test가 평가·개발 입력에 0건임을 단언하는 방법
- 스크립트: `python -I analysis/data/m33_assert_no_aihub_test.py [--expect-hits] <목록>...` (sha256 `358a19f9899d9137e92b006cff20f3ce0a152f5828b546b8d7b0aadbc069db73`). 목록은 .txt(영상 경로), .yaml(`path`·train·val·test가 가리키는 .txt까지 따라감), .csv(file_name·file·crop_key 열).
- 기준: `data/splits/aihub_yolo_v1.csv`(sha256 `c548cc0b64a91b6622dc2f6713ea628cfb88704306b39db24b3d17552a00f8a1`)의 `subset == test` 2,086행. 세 키로 센다: 파일명, gain 제거 크롭 키, 세션(`source`). 하나라도 걸리면 종료 1. 매니페스트에 없는 이름은 `not_in_manifest`로 따로 센다(0이어야 함).
- **양성 대조**: `--expect-hits`로 `test_images.txt`를 넣으면 3키 모두 2,086건이 걸려야 하고(종료 0), 단언이 실제로 걸림을 보인다.
- 오늘 실행(2026-11-15): `data/aihub_yolo_v1/fold0–4_images.txt`(3,525 + 3,210 + 2,901 + 2,490 + 2,675 = 14,801 = train_pool 전체, 중복 0)와 `fold0.yaml`(이름이 가리키는 목록 전체 14,801)에서 파일·크롭 키·세션 적중 **0**, 매니페스트 밖 이름 0 → 통과. 양성 대조 `test_images.txt`는 2,086 / 2,086 / 2,086건 적중, 종료 0(기대대로).
- 계획에서의 쓰임: D0·R1–R3·J1 실행 전 입력 목록마다 위 스크립트를 한 줄로 돌려 종료 0과 `total_hits: 0`을 실행 기록 md에 넣는다. 새 입력 목록(예: `aihub_yolo_v2_auto32`의 목록, D0 `5,473 crop` 목록)이 생기면 같은 방식으로 추가 실행해야 한다(이번에는 `aihub_yolo_v1`의 목록만 확인함. `aihub_yolo_v2_auto32`와 합성 데이터는 확인하지 않았다).

## 5. 위임 필요·권고
없음. 실행 전 감사(verif)가 필요하면 위 sha 표를 사전 등록 값으로 쓰면 된다. 변경이 생기면(특히 `fourtu_duplicate_radargrams.csv`, `known_apexes.csv`) 이 문서의 sha를 갱신한다.
