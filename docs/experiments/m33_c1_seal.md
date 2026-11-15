# M3-3 C1 봉인 기록 (초안)

- 상태: **초안**(2026-11-16, cv-detection-expert). 아직 봉인하지 않았다. 규칙은 [RUN] [`m33_runner_spec.md`](m33_runner_spec.md) v0.3.6 §1, 순서는 [PLAN] [`m33_cross_domain_plan.md`](m33_cross_domain_plan.md) v0.2.8 §4 C1 행이다.
- **권위 출처는 러너 상수다(러너 단언 행에 한함).** 러너가 단언하는 값은 `analysis/detection/m33_inputs.py`(와 `m32_recipe_inputs.py`·`m32_aihub_perturb.py`)의 상수다. 이 문서의 표는 그 사본이다. §2의 "간접 고정"·"기록 전용" 행에는 러너 상수가 없으므로 이 문장이 해당하지 않는다. 그 행의 값은 이 문서가 기록이다. C1 실행 전 감사에서 verif-protocol-auditor가 상수와 이 표가 같은지 대조한다.
- 텍스트 입력은 **LF sha가 단언 기준**이고 원시 sha를 함께 적는다([RUN] §1 텍스트 입력 sha). 바이너리는 원시 sha다.
- 아래 값은 2026-11-16 작업 트리(HEAD `995665f`)에서 scratchpad 스크립트로 직접 잰 것이다(읽기 전용, 판정 자료를 열지 않음). 러너 상수에 이미 있는 값은 모두 일치했다.

## 1. 코드 blob

C1 커밋 뒤에 채운다. 자리 `C1_BLOB`은 각 행의 C1 커밋 blob, `C1_COMMIT`은 C1 커밋 id다. 행 목록은 R1·R2·R3 `run_identity.code_blobs`의 합집합과 D0 계보·r0 기대값 출처다([RUN] §1 정체성 기록).

- C1 커밋: `{C1_COMMIT}`

| 경로 | C1 커밋 blob | 쓰는 곳 | 비고 |
|---|---|---|---|
| `analysis/detection/m32_recipe_eval.py` | `{C1_BLOB}` | R2·R3 | 러너 본체 |
| `analysis/detection/m33_r2_eval.py` | `{C1_BLOB}` | R2 | |
| `analysis/detection/m33_r2_lines.py` | `{C1_BLOB}` | R2 목록 빌더 | 출력은 §2 R2 목록 |
| `analysis/detection/m33_r3_band.py` | `{C1_BLOB}` | R3 | |
| `analysis/detection/m33_r3_p4.py` | `{C1_BLOB}` | R3 | P4 계산 |
| `analysis/detection/m33_morocco_eval.py` | `{C1_BLOB}` | R1 | |
| `analysis/detection/m33_inputs.py` | `{C1_BLOB}` | R1·R2·R3 | 단언 상수의 권위 출처 |
| `analysis/detection/m32_recipe_inputs.py` | `{C1_BLOB}` | R1·R2·R3 | |
| `analysis/detection/m32_recipe_match.py` | `{C1_BLOB}` | R2·R3 | |
| `analysis/detection/m32_pipe_responses.py` | `{C1_BLOB}` | R2·R3 | |
| `analysis/detection/m32_aihub_perturb.py` | `{C1_BLOB}` | R1·R2·R3 | 가중치 sha 상수 |
| `analysis/detection/m31_d1_common.py` | `{C1_BLOB}` | R1·R2·R3 | |
| `analysis/detection/m33_d0_gate.py` | `{C1_BLOB}` | D0 계보 | §5와 함께 |
| `analysis/detection/m33_d0r_compare.py` | `{C1_BLOB}` | D0 계보 | §5와 함께 |
| `src/gpr_detection/pipeline.py` | `{C1_BLOB}` | 전체 | C0a 값 `dd5e7bcb…`과 같아야 한다 |
| `src/gpr_detection/preprocess/to_image.py` | `{C1_BLOB}` | R2·R3 | |
| `src/gpr_detection/eval/inject.py` | `{C1_BLOB}` | R3 | S4 코드(띠 함수 포함) |
| `src/gpr_detection/detect/yolo.py` | `{C1_BLOB}` | R1·R2·R3 | |
| `src/gpr_detection/data/morocco_eval.py` | `{C1_BLOB}` | R1 | |
| `src/gpr_detection/preprocess/filters.py` | `{C1_BLOB}` | R1 | |
| `tests/test_detection_image_recipes.py` | `{C1_BLOB}` | r0 정체 기대값 출처 | §4 |

- src tree id(C1 커밋): `{C1_BLOB}`. C0a·D0′ 값 `164f0990…`과 다르다. 882e881 뒤 바뀐 src는 `eval/inject.py` 하나다(띠 함수 추가, 995665f에서 +305/−1, `git diff --stat 882e881 995665f -- src`). 995665f의 src tree는 `d0e54933…`이다. `pipeline.py` blob은 바뀌지 않았다. 그래서 R 러너의 봉인 PP 단언(`pipeline.py` blob·`PPConfig`)은 유지되고, src tree id 단언은 D0 실행기에만 있다(D0′ 재실행은 이제 종료 2로 멈춘다. 다시 돌릴 계획은 없다).

## 2. 텍스트 입력

| 이름 | 경로 | LF sha256 (단언 기준) | 원시 sha256 | 줄 수(머리 포함) | 줄끝 | 러너에서의 쓰임 |
|---|---|---|---|---|---|---|
| R2 4TU 측선 목록(754 / `keep_one` 720 / K 27) | `analysis/detection/m33_r2_lines.csv` | `57de56c64d488eab0bc2316f816177317958a40060c47227787523d0f97c4b8d` | 같음 | 755 | LF | 러너 단언(`R2_LINES_SHA256_LF`) |
| 활동 간 중복 목록 | `docs/datasets/fourtu_duplicate_radargrams.csv` | `7602d94fbd5824f3536d463e2b248e4ea474770f852519db3ea30097312c488a` | `6cdef12c55fe49ec1d865f0f513485b4dd51cfbe4e81b9171956b22821430552` | 40 | CRLF | 간접 고정(R2 목록 빌더 입력, 754 목록 sha로 고정) |
| K 꼭짓점 | `data/processing/d3/known_apexes.csv` | `0fec12dba4efa504a69a8319e52eb854dac6359296beb391f6b83b177d49db61` | 같음 | 42 | LF | 러너 단언(`EXPECTED_SHA256["k_apexes"]`) |
| K 측선 | `data/processing/d3/lines.csv` | `f6fe08a93808555727acb3ecd864abc1cd555f0c6900a753a8c598be4543ca0a` | 같음 | 60 | LF | 러너 단언(`EXPECTED_SHA256["k_lines"]`) |
| K 꼭짓점 SNR(X6) | `analysis/processing/m33_k_apex_snr.csv` | `55d54bb15a84f4acccdd51d4ace82b4fef23265956f7ab3f73b1dea94c29bd16` | 같음 | 42 | LF | 러너 단언(`K_APEX_SNR_SHA256_LF`) |
| X6 주기·SNR | `analysis/processing/m33_x6_period_snr.json` | `2ee86a06a1c020c6462b75a33d06bc7f2e6e497b0bf0aa36e7c6d9c4fd696178` | `8a6e189b644d0fef9c06268f131f4739b70e95f74cf3e03aa85c4e8172823761` | 479 | CRLF | 기록 전용(R1은 분위수 값만 단언, sha는 기록) |
| IDS 60측선 목록 | `analysis/processing/m32_stageA_lines_ids.csv` | `a52e467dfed80e64885b3ced9e5fb5e9e9617305b39e7d7e1d44e4d1f5bedfb3` | 같음 | 61 | LF | 러너 단언(`EXPECTED_SHA256["stage_a_ids"]`) |
| 4TU 단계 A 150측선 목록 | `analysis/processing/m32_stageA_lines_4tu.csv` | `2d590cdab6055df22d7294ddaa11676fee39774e9b95a705b4c9884e3787cdf2` | 같음 | 151 | LF | 러너 단언(`EXPECTED_SHA256["stage_a_4tu"]`) |
| 채움 높이 표 | `analysis/processing/m32_content_height_lines.csv` | `c11bbd3780dc42a6245941cd8b2a9ac49db90ee0f0412fe267996ec469edfb41` | `65fb6bf4b2a4d8ac6f801b241e5d8ee988454a9e86d8072470d9c7245f7076db` | 1169 | CRLF | 기록 전용(M3-2 smoke 선택에만 쓰임) |
| 모로코 매니페스트 | `data/splits/morocco_eval_v1.csv` | `db2ebc301b68aff70cd448c99b4b0338ece3f7d240bd46f20b2a1d7661513fd5` | `b140c7b465e8bf4fd0dd1701356862984531176578c1c9482b9caca2c0562327` | 234 | CRLF | 러너 단언(`MOROCCO_SHA256_LF`·`_RAW`) |
| 모로코 평가 목록(233) | `data/morocco_eval_v1/eval_images.txt` | `85b1666d18eeca0c9e1b5108e34797e6503bf22431989eae16bde0410c1f0cef` | `0e91de3529635f8d1df7903c840c68df63a7cbb2853072fd7b81a1bf484eaa1a` | 233 | CRLF | 러너 단언(`MOROCCO_SHA256_LF`·`_RAW`) |
| 모로코 영상 233장(이름순 sha 목록의 sha) | `data/morocco_eval_v1/` | `95ad76444681d9b82e3a70bf66e48b05e2800df72e3c6643de348c998c448958` | – | – | – | 러너 단언(`MOROCCO_IMAGES_SHA256`) |
| 모로코 라벨 233개(같은 방식) | `data/morocco_eval_v1/` | `12b0fbe2f2bf0cc5d191f830e9c10da622543f361f3669043e1e7081b58ef7f3` | – | – | – | 러너 단언(`MOROCCO_LABELS_SHA256`) |
| C0b `d0_gate.json` | `runs/eval/m33_d0/d0_gate.json` | `107d7b53f30667684baf7c2236bdf338c861309531ff17497e2836193bea08c6` | 같음 | 27,137 | LF | 러너 단언(`C0B_D0_GATE_SHA256`) |
| IDS `radargrams_index.csv` | 읽지 않음 | – | – | – | – | 단언하지 않음(R 러너가 읽지 않음) |

- 모로코 영상·라벨 두 값은 X7(`m33_data_assertions.md` §1) 방식의 묶음 sha이고, 러너 상수 `MOROCCO_IMAGES_SHA256`·`MOROCCO_LABELS_SHA256`에서 옮겼다(이번에 다시 재지 않음).
- `data/`·`runs/`는 git 밖이다. 그래서 blob이 아니라 sha만 적는다.
- IDS `radargrams_index.csv`는 [RUN] §1 입력 목록에 있었지만 R 러너가 읽지 않는다(python-expert 확인, 2026-11-16). 그래서 sha를 적지 않는다. IDS 측선 정보는 `m32_stageA_lines_ids.csv`에서 온다.

## 3. 바이너리 입력 — 검출기 가중치 (N2-a′ 5 fold `last.pt`)

| fold | 경로 | sha256 | 크기(B) |
|---|---|---|---|
| 0 | `runs/detect/yolo11m_aihubv1n2a2_20261114_fold0/weights/last.pt` | `7ead71c7fa62d85edb11f5614e78b11619062b455dca62f5cc66c93b2dcf8d92` | 40,471,724 |
| 1 | `runs/detect/yolo11m_aihubv1n2a2_20261114_fold1/weights/last.pt` | `f4b7e0443799d66983adb21356a390bf5cf68ec98964e9425f0981cd3b079b15` | 40,471,724 |
| 2 | `runs/detect/yolo11m_aihubv1n2a2_20261114_fold2/weights/last.pt` | `32c0e854ed7e6ac6e4847265f956f038030bb968f4d01ab8a550a40e2d98a99b` | 40,471,724 |
| 3 | `runs/detect/yolo11m_aihubv1n2a2_20261114_fold3/weights/last.pt` | `0238d4d62722ff7318e8c8fa8c283471b008f01905e95d326510547882b74de8` | 40,471,724 |
| 4 | `runs/detect/yolo11m_aihubv1n2a2_20261114_fold4/weights/last.pt` | `98922d2e79b547fd7cf4714b400e693b131d9082c58d293b9e0f5163e039c6b2` | 40,471,724 |

- 다섯 값 모두 `m32_aihub_perturb.EXPECTED_WEIGHTS_SHA256`과 같다(앞 8자는 [`m31_n2a2_retrain_plan.md`](m31_n2a2_retrain_plan.md)의 기대 sha8과 같다). `m31_d1_common.EXPECTED_WEIGHTS_SHA8`은 M1-10 가중치용이라 N2-a′와 무관하다.
- 가중치는 AGPL 대상이라 공개 저장소에 넣지 않는다(sha만 적는다).

## 4. 설정·정체

| 항목 | 값 | 출처·계산 |
|---|---|---|
| 설정 해시(conf 0.25·AP 0.001·NMS 0.7·`MergeConfig()`·r0 `assumed_params`·`assumed_values`) | `b9c5c905c5fec3ea9840e9644b177aeb6689ab4191d1e2e1dfce17e9a4518b6a` | 상수 `m33_inputs.R_SETTINGS_SHA256`(`m33_inputs.py`:169). 문서 `settings_document`(:417) = conf_operating 0.25, conf_ap 0.001, nms_iou 0.7, `asdict(DEFAULT_MERGE_CONFIG)`, r0 `M3_2_ASSUMED` 이름 목록과 그 이름들의 `DEFAULT_DETECTION_IMAGE_CONFIG` 값. 해시 = `json.dumps(sort_keys=True, default=str)`의 sha256. 단언 `assert_settings`(:446). 줄 번호는 미커밋 작업 트리(2026-11-16) 기준이다 |
| `pp_set_id` | S3 = {a, b, d} | C0b `selection.pp_set_id`, `m33_inputs.C0B_PP_SET_ID` |
| `PPConfig` 해시(S3) | `eb2796ddfaa9611e5030342c4ed66b00c91ff2ed6a6698ee5979b9afe080602f` | `m33_inputs.C0B_PP_CONFIG_SHA256`, C0b `d0_gate.json` |
| `pipeline.py` blob(C0a) | `dd5e7bcba4397efd732e7bf3ca1c5c92fb2404a1` | `m33_inputs.C0A_PIPELINE_BLOB`, `git rev-parse 53c4e16:src/gpr_detection/pipeline.py` |

**r0 정체 — [SP] §9 11측선 `line_image` sha256(기대값)**. 출처는 M3-2 S6 (a) 시험 고정값, `tests/test_detection_image_recipes.py`의 `RealLine` 목록(S1 전 기본값과 비트 동일)이다. **이관 완료**: 상수 `m33_inputs.R0_IDENTITY_LINES`(`m33_inputs.py`:105), 단언 `assert_r0_identity`(:478). 시험 파일도 이 상수를 import하므로 출처가 하나다. C1 실행 전 감사에서 상수와 이 표를 대조한다.

| # | 측선 | sha256 |
|---|---|---|
| 1 | `4tu/03.9/Path5` | `a0891dbfd7efda9a862ddc8a3b0a0ebfd898f821750c07ba85e0a68275d94c79` |
| 2 | `4tu/01.1/Path1` | `10e8bc9ad7c671a766f9797d9a385dbf8b46b04867ba7d79b61d6d600019a62f` |
| 3 | `4tu/09.1/Path2` | `66863f4caa028cbca8ee93f173cb7cb5692733a38f61cf66d464316f9cc33ffd` |
| 4 | `4tu/09.4/Path1` | `78fad5249da1ceb1cf83c752eb8d7096f22be009c471ca190819ca15f075b660` |
| 5 | `gz/pipe/001_190321AA/LID10001` | `b1de7022cc398c91bfa3d34c329938868dbc1559fa878eeef75ae12192d13d78` |
| 6 | `gz/pipe/1030_190117AF/LID10001` | `a3a40315e42ac81602759c7490b98b6c0306bbb4461f0cc536c51031bd7699da` |
| 7 | `gz/pipe/190321A_190321AH/LID10001` | `788f8d22fcccb6b26015d4b699739122922afdd7898d94a8cad0639039e22e46` |
| 8 | `gz/pipe/MALA_07011/LA010001` | `2a0707a491e844808cb29bc0450eba977004e27a76d96dc458e5c75a12d98596` |
| 9 | `gz/pipe/1030_181224AB/LID10001` | `d73a8135a5c9c86075b319c3fd711584af08fe9f88e8e38c069fa6f4ab4e049e` |
| 10 | `gprsd/CementRoad2_1` | `292ede9cdffdef2507692a9418aed58107546832c28af417ebae357a6ce9c9e6` |
| 11 | `gprsd/AsphaltRoad2_1` | `30b732908110f45f581673bd0dd6765f5b4ca68c492dee603957bfa8a9b9fb89` |

- 같은 시험 파일의 `DT0195_LINE`(`4tu/011.17/Profile_6409`, Δt 0.195)은 §9 11측선 밖의 추가 시험 측선이라 이 표에 넣지 않는다.

## 5. D0 계보

[`m33_d0r_record.md`](m33_d0r_record.md) §6을 옮기고, C1 커밋 blob 열을 더했다.

| 대상 | C0a·C0b 쪽 (53c4e16) | D0′ 쪽 (882e881) | C1 커밋 |
|---|---|---|---|
| `m33_d0_gate.py` blob | `a2741259bddbeaa26daf3b091ac76ea753679918` | `1775f9a27f9e798e8fc36364fb723cb836fb22ee` | `{C1_BLOB}` |
| `m31_d1_common.py` blob | `bdfb7f0f2ff08ced10b7f281d646d717dee34621` | `0ba0675c8fc6bd632ed2b1e6794702b60f9b5d72` | `{C1_BLOB}` |
| `m33_d0r_compare.py` blob | – | `ee802e8b8e665264ccc95dc0f9944a8f661b4097` | `{C1_BLOB}` |
| `pipeline.py` blob | `dd5e7bcba4397efd732e7bf3ca1c5c92fb2404a1` | 같음 | `{C1_BLOB}`(같아야 함) |
| src tree id | `164f09909ad51c1a8d26545cadb0417b1da3acc6` | 같음 | §1 |
| `d0_gate.json` sha256 | `107d7b53f30667684baf7c2236bdf338c861309531ff17497e2836193bea08c6` | `16d7d2ade37a79ef24f00603628d864b5ddd396bd468738dbe9ab3f1970aedf1` | – |
| 대조 | – | `compare_c0b.json` sha256 `39fac6049d632addfd8328605c2425caa40f96e675cf785fcb7c0c7d9db57058`, status `match`, verif 판정 후 감사 완료(2026-11-15) | – |
| `PPConfig` 해시(S3) | `eb2796ddfaa9611e5030342c4ed66b00c91ff2ed6a6698ee5979b9afe080602f` | 같음 | §4 |

- C1 1단계 도구 보강으로 `m33_d0_gate.py`·`m33_d0r_compare.py`가 882e881 뒤에 다시 바뀌었다. 그래서 D0′ 출처 blob(882e881)과 C1 커밋 blob이 다르다([PP] [`m33_pp_spec.md`](m33_pp_spec.md) 부록 A-1).

## 6. 나중에 채울 값 (이 문서가 아니라 다른 곳에 기록)

| 단계 | 값 | 기록 위치 | 시점 |
|---|---|---|---|
| C1b | R3 주입 표·위치 표·제외 표 LF sha | `analysis/detection/m33_r3_c1b.json`(plan의 `plan/c1b_record_draft.json`을 옮김)과 C1b 커밋 | R3 plan 뒤, injected 전 |
| C2 | R2 fold 0 `j1_boxes_post.csv`·`j1_boxes_pre.csv` sha, J1 코드 blob | J1 사전 등록 문서의 C2·R2 박스 sha 자리 | R2 뒤, J1 전 |
| 판정서 | `mde_pp`(stats-design, 길이 가중 개수 모형) | P4 판정서 | C1b 뒤, injected 전 |

- 이 문서에는 C1b·C2 자리를 두지 않는다. 그래서 C1 실행 전 감사의 허용 목록은 바뀌지 않는다.

## 7. 자리 이름과 placeholder_scan 허용 목록

| 자리 | 뜻 | 채우는 시점 |
|---|---|---|
| `C1_COMMIT` | C1 커밋 id | 코드·문서 커밋 직후 |
| `C1_BLOB` | 그 행 경로의 C1 커밋 blob(또는 src tree id) | 같음 |

- **초안 단계**(지금, C1 커밋 전): 이 문서는 `placeholder_scan.py docs/experiments/m33_c1_seal.md --allow C1_COMMIT C1_BLOB`로 종료 0이어야 한다(설정 해시 자리는 2026-11-16 값으로 채웠다).
- **C1 실행 전 감사**: 두 자리 모두 채워져 있어야 한다. 이 문서는 `--allow` 없이 종료 0이어야 한다. J1 문서는 [PLAN] §4 C1 행대로 `--allow C2 R2_J1_POST_SHA R2_J1_PRE_SHA`다.
- 설명 문장에서는 자리 이름을 중괄호 없이 쓴다(스캔이 설명 문장도 잡음).

## 8. 남은 일 (대기)

- gpr-processing-expert: X6 주기 측정 함수 확인은 참고로 남긴다(R1이 s5 주기의 X6 일치를 정식 실행에서 이미 단언하므로, 어긋나면 R1 단언이 실패해 멈춘다).
- 메인 세션: 코드·문서 커밋 → 이 문서의 §1·§5 자리(`C1_COMMIT`·`C1_BLOB`) 채우기 → C1 실행 전 감사(verif) → C1 봉인.
- (끝남, 2026-11-16) python-expert: 설정 해시 상수(§4), r0 11측선 기대값 이관(§4), IDS `radargrams_index.csv` 정리(§2, 읽지 않음), C1 2단계 보강 6건.
- (끝남, 2026-11-16) gpr-processing-expert: J1 채움 전용 박스 제외와 "파일의 모든 행" 규칙의 정합 확인. J1 §3·§4 5 문구로 반영했다(J1 판 번호는 v0.3.2 그대로, 문구 정리).
