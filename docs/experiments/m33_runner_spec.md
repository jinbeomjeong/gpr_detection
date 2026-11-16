# M3-3 X4 — 평가 러너 명세 (R1 모로코, R2 4TU, R3 의사 띠, J1 입력)

- 상태: **v0.3.6 — C1 봉인됨(2026-11-16), C1 코드 커밋 f4ad45e**(cv-detection-expert). 봉인 기록은 [`m33_c1_seal.md`](m33_c1_seal.md)다. 이후 이 문서의 규칙 변경은 새 판과 공개가 필요하다. v0.3.6에서 stats-design 검토와 사용자 결정(D0 결과 뒤·C1 전·측정 전)에 따라 P4 상태 결정을 재현율 집합 R의 제거 작용으로 한정하고, `ghost_delta`를 보고 전용(유령 구현 단언)으로, 판정 기준을 ≥ −1.0 %p − 1e-9로 정했다. v0.3.5에서 D0 결과(C0b, S3) 뒤·C1 전·측정 전 사용자 결정에 따라 P4 PP 노출 위치를 선택 집합 규칙이 작동하는 위치로 한정하고 노출 0 처리(`p4_status` 네 상태)를 정했다. v0.3.4에서 [EMR] 부록 D-8(v1.9.1) P4 물리 검토를 반영했다(IDS 층 구성 정정, L 150 유지, 노출 기하 공개, 노출 부분집합 보고 열, 배치 순서 명시, 층 경계 시험). v0.3.3에서 stats-design 검토에 따른 사용자 결정(P5 구조상 0 단언, PP 후 유령 짝, P4 노출 위치 수·작은 층 처리, IDS 층 구성 공개)을 반영했다(C1 전). v0.3.2에서 띠 모드 (c′)·(d) 해석을 정했다. v0.2에서 R3를 [EMR] 부록 D에 맞췄고, v0.3에서 X6·사용자 결정을, v0.3.1에서 무결성 점검과 [EMR] 부록 D-6(v1.6)을 반영했다. [PLAN] [`m33_cross_domain_plan.md`](m33_cross_domain_plan.md) v0.2.8(S0 9b28694는 v0.2.2)와 [PP] [`m33_pp_spec.md`](m33_pp_spec.md) v0.2.4(규칙 본문은 C0a v0.2.3, 부록 A 실행기 개정)와 [EMR] 부록 D v1.9.1을 따른다. 구현은 python-expert가 한다.
- 이 러너들은 C1 봉인 뒤에만 정식 실행한다. 봉인 순서는 [PLAN] §4 표다(… → C0b → 개정 실행기 커밋 → 실행 전 감사 → D0′ → 대조 → C1 → R1 → R2 → R3 uninjected(IDS) → R3 plan → **C1b** → R3 injected → **C2** → J1. 코드는 plan이 IDS 미주입 결과를 요구한다). C1 전에는 `--smoke`만 허용하고, smoke는 별도 폴더에 쓰며 판정에 쓰지 않는다.

## 0. 재사용 판단

| 대상 | 재사용 | 판단 |
|---|---|---|
| R2 4TU 미주입 | **`analysis/detection/m32_recipe_eval.py` `uninjected` 단계를 확장해 재사용** | 같은 검출 경로가 이미 있다(r0 L1, batch 64, 병합, M4a–d, `apex_hits`, `conf_counts`, `flip_pairs`, batch 1 정체 게이트). 바꿀 것은 §2-1의 확장 항목이다 |
| R3 의사 띠 | **`plan`·`injected` 단계를 확장해 재사용** | 배치·제외 구역(c′)·실현·이어 돌리기가 그대로 쓰인다(주입 주변 타일 추론은 r0 측선 전체 통계 때문에 측선 전체 추론으로 바꿨다, §3-1 4). 바꿀 것은 주입체(S4 띠 모드, `make_pseudo_band` 아님)와 일치 규칙(가로 구간)이다. (v0.3.6 C1 1단계 배선) 띠 전용 로직은 새 모듈 **`analysis/detection/m33_r3_band.py`**에 두고 `m32_recipe_eval.py --injectant band`가 uninjected(IDS) → plan → injected 세 단계를 부른다 |
| J1 입력 | R2 실행의 박스 내보내기로 만든다 | 추론을 따로 하지 않는다 |
| R1 모로코 | **새 러너 `analysis/detection/m33_morocco_eval.py`** | S5는 측선 npy 경로다. 모로코는 스케일 없는 정사각 JPG라 맞지 않는다. 대신 S3(`m32_aihub_perturb.py`)의 predict 경로(`gpr_detection.detect.yolo`, conf 0.001, NMS 0.7, 한 장씩)와 `m31_d1_common.ScoreAccumulator`(Ultralytics 8.4.192와 같은 AP)를 재사용한다. validator 경로(`scripts/eval_detector.py`)는 PP를 걸 수 없고 S3와 경로가 달라 쓰지 않는다 |

- 공통 모듈: `m32_recipe_inputs`(sha 단언·목록), `m32_recipe_match`(M4·일치), `m31_d1_common`(채점), **`m33_inputs`**(v0.3.6 C1 1단계: M3-3 공통 상수와 검사 `check_folds`·`tree_state`·`judgment_record`·`final_status`·`assert_sealed_pp`·`text_sha256_record`), **`m33_r3_p4`**(P4 계산: 노출·R·G·G′·상태·판정·덮임 불변·유령 단언).
- 새 상수·목록은 `analysis/detection/m33_inputs.py`에 둔다. M3-2 상수는 바꾸지 않는다.

## 1. 공통 요구 (R1–R3·J1 내보내기)

| 항목 | 요구 |
|---|---|
| 검출기 | N2-a′ 5 fold `last.pt`. sha 5개를 단언한다([BL] §1). 앙상블 없음, fold별 독립 |
| 설정 | conf 0.25(운용), AP용 0.001, NMS 0.7, `MergeConfig()`, r0 `DetectionImageConfig`. **설정 해시**(conf·병합·r0 `assumed_params`·`assumed_values`)를 C1 값과 단언한다. (C1 2단계 판단, cv 2026-11-16) 기대값은 C1 커밋 코드에서 한 번 계산해 `m33_inputs`에 상수로 고정하고 C1 sha 표에 적는다. 자료를 읽지 않는 결정적 값이라 판정 자료와 무관하다. C1 뒤 설정이 바뀌면 종료 2다. (구현) 상수 `m33_inputs.R_SETTINGS_SHA256` = `b9c5c905…`, 문서 `settings_document`(conf_operating·conf_ap·nms_iou·`asdict(DEFAULT_MERGE_CONFIG)`·r0 `M3_2_ASSUMED` 이름과 `DEFAULT_DETECTION_IMAGE_CONFIG` 값), 해시는 `json.dumps(sort_keys=True, default=str)`의 sha256, 단언 `assert_settings`. R2·R3는 결과를 `run_identity.r_checks`에, R1은 `run_identity.settings_assertion`에 남긴다. 값은 [`m33_c1_seal.md`](m33_c1_seal.md) §4 |
| r0 정체 | [SP] §9 11측선 `line_image` sha256을 첫 단계에서 다시 만들어 단언한다. (C1 2단계 판단) 기대값은 M3-2 S6 (a)가 고정한 값, 곧 `tests/test_detection_image_recipes.py`의 `RealLine` sha256 11개다(S1 전 기본값과 비트 동일, M3-2 판정 커밋 기준). 이를 `m33_inputs` 상수로 옮겨 단일 출처로 두고, C1 sha 표에 시험 파일 blob과 함께 적는다. 새로 계산한 값을 기대값으로 쓰지 않는다(r0 동결 확인이 목적). (구현) 상수 `m33_inputs.R0_IDENTITY_LINES`(시험 파일도 이것을 import, 단일 출처), 단언 `assert_r0_identity`. R2·R3가 시작할 때 `r_runner_checks`로 설정과 함께 단언하고 `run_identity.r_checks`에 남긴다. R1은 측선 영상을 쓰지 않아 r0 정체를 단언하지 않는다 |
| PP | 같은 병합 박스에 **PP 꺼짐(S0)과 C0b 선택 집합**을 둘 다 적용해 두 행으로 저장한다(`pp` 열 `pre`/`post`). `PPConfig` 해시 = C0b를 단언한다. 추론은 1회다 |
| 입력 sha 단언(C1 표) | 가중치 5 · r0 11측선 · 설정 해시 · C0b(`PPConfig`, `d0_gate.json`) · 모로코 매니페스트·`eval_images.txt` · 4TU 754측선 목록·`keep_one` 표(720) · 중복 목록 `fourtu_duplicate_radargrams.csv` · K 꼭짓점(`known_apexes.csv`)·K 측선(`d3/lines.csv`) · IDS 목록(`m32_stageA_lines_ids.csv`) · `radargrams_index.csv` · S4 코드 blob(`src/gpr_detection/eval/inject.py`. 상수 `m33_inputs.C1_INJECT_BLOB` = C1 코드 커밋 f4ad45e의 blob `2d88ff0a…`(826aa41과 같음)로 R2·R3 시작 때 단언, 다르면 종료 2. 결과는 `run_identity.r_checks.inject_blob`에 남는다) · R3 주입 표(C1b). 하나라도 다르면 종료 2. (C1 2단계) `radargrams_index.csv`는 R 러너가 읽지 않아 단언하지 않는다. 중복 목록은 754 목록 sha(`R2_LINES_SHA256_LF`)로 간접 고정한다. X6 json은 값 단언(분위수), sha는 기록. 행별 구분(러너 단언 / 간접 고정 / 기록 전용)은 [`m33_c1_seal.md`](m33_c1_seal.md) §2 |
| D0 실행기 blob(C1 표, v0.3.6) | C1 sha 표에 D0 실행기 `m33_d0_gate.py`(와 `m31_d1_common.py`) blob을 **두 개 나란히** 적는다: C0a 실행기 blob(53c4e16, C0b 결과의 출처)과 개정 실행기 blob(D0′ 결과의 출처). **D0′ 출처 blob은 882e881 값**(`m33_d0_gate.py` `1775f9a2…`, `m31_d1_common.py` `0ba0675c…`, `m33_d0r_compare.py` `ee802e8b…`, [`m33_d0r_record.md`](m33_d0r_record.md) §6)으로 적는다. C1 1단계에서 `m33_d0_gate.py`·`m33_d0r_compare.py`가 도구 보강으로 다시 바뀌었으므로 C1 커밋의 blob과 다르며, 그 차이와 C1 커밋 blob을 함께 적는다. D0′ `d0_gate.json` sha와 대조 결과도 적는다([PP] 부록 A-1). R 러너는 실행기 blob을 단언하지 않고 `PPConfig` 해시·`pipeline.py` blob을 단언한다. D0′ 실행기는 같은 값을 모듈 상수(`assert_sealed_pp`)로 단언하고 `src/` tree id도 C0a 값과 대조하므로, R 러너 구현 중 `src/`를 바꾸는 작업은 D0′ 뒤에 한다([PP] 부록 A-1) |
| 깨끗한 트리 | 정식 실행은 추적 파일 변경 0, 필수 입력 추적을 요구한다(`dirty` false). (구현, C1 실행 전 감사 반영) 판정 가능성은 **시작 시점의 트리가 dirty인지**로 정한다. dirty이면 `dirty_tree` 사유로 `not_judgeable`(종료 1)이다. `--allow-dirty`는 dirty 트리에서 시작을 허용할 뿐이고, 판정 가능하게 만들지 않는다. smoke·진단 전용이며 정식 실행에서는 금지 인자다(아래 "확정 명령·금지 인자") |
| 확정 명령·금지 인자·실패 규칙(C1 실행 전 감사 반영, 사용자 결정 2026-11-16, R1–R3 결과 전) | **확정 명령**(저장소 루트, `conda run -n gpr-dev python -I -B …`): R1 `analysis/detection/m33_morocco_eval.py`(인자 없음). R2 `analysis/detection/m32_recipe_eval.py --stage uninjected --lines analysis/detection/m33_r2_lines.csv`. R3 `m32_recipe_eval.py --injectant band --stage uninjected` → `--injectant band --stage plan [--workers N]`(N은 compute-scheduler-expert 배치표대로) → C1b 커밋 → `--injectant band --stage injected`. **금지 인자**: `--smoke`, `--limit*`, `--only-lines`, `--folds`, `--no-flip`, `--overwrite`, `--allow-dirty`, `--out-root`, `--run-root`, `--r2-root`, `--c1b-record`, `--f2-record`, `--provisional`(이미 판정 불가 사유로 막히지만 함께 적는다). 예외는 아래 실패 규칙의 예비 폴더(`--out-root`로 `r1_retry`·`r2_retry`·`r3_retry`를 주는 것)뿐이다. **실패 규칙**(D0′와 같은 예비 폴더 방식): (1) 출력 폴더가 만들어지기 전에 예외가 나면 원인을 고쳐 같은 명령으로 다시 돌린다. (2) 부분 출력이 생겼으면 그 폴더는 열지 않고 남겨 두고, 예비 폴더 `runs/eval/m33/r1_retry`·`r2_retry`·`r3_retry`에서 다시 돌린다. 이 사실(실패 폴더, 종료 코드, 원인)을 판정서에 공개한다. (3) 코드를 고쳤으면 새 blob이므로 실행 전 감사를 다시 받는다. (4) 예비 폴더를 쓰면 하류는 그 경로를 `--r2-root`로 받는다(금지 인자의 허용 예외). R3는 실패한 단계부터가 아니라 세 단계(uninjected·plan·injected)와 C1b를 모두 `r3_retry`에서 다시 한다. 러너가 만든 빈 폴더나 표지(`stage_settings.json`·`parts/`)만 있어도 부분 출력으로 보고 예비 폴더로 간다. 같은 명령으로 다시 돌릴 수 있는 경우는 출력 폴더가 아직 만들어지지 않은 때뿐이다(C1 실행 전 감사 재확인 기록 1). (5) `--resume`(이어 돌리기)은 같은 HEAD·코드 blob·src tree에서만 쓴다. 단계 폴더의 resume 표지에 `head`·`code_blobs`·`src_tree_id`를 남기고, 이어 돌릴 때 지금 값과 하나라도 다르면 거부한다(사용자 결정, 구현됨) |
| 끝 시점 트리 재측정(v0.3.6) | git HEAD·`dirty`·코드 blob·src tree id를 시작할 때와 결과 파일을 쓰기 직전에 두 번 잰다. 다르면 `input_error`, 종료 2다. 두 값과 차이를 `run_identity.tree_end`·`tree_problems`에 남긴다(D0 실행기 개정과 같은 방식, [PP] 부록 A) |
| 전체 정밀도(v0.3.6) | CSV·json에 쓰는 점수·좌표·시간·주기·IoU 등 모든 실수는 반올림하지 않고 `repr(float)`(왕복 가능한 전체 정밀도)로 쓴다. 표시용 반올림은 md 보고서에만 쓴다. 근거: D0 재집계에서 점수 소수 4자리 기록 때문에 conf > 0.25 경계 박스의 판정이 뒤집혔다([`m33_c0b_d0_judgment.md`](m33_c0b_d0_judgment.md) 주 1) |
| 판정 가능성(v0.3.6) | 기대 fold는 0–4다. 중복되거나 범위 밖인 fold 인자는 종료 2로 거부한다. smoke, 일부 fold, 측선·영상·crop 수 제한, 일부 모델, 시작 시 dirty 가운데 하나라도 해당하면 `status` = `not_judgeable`, 종료 1이다. 결과 json에 `judgment` 블록(`judgeable`, `blockers`, `expected_folds`, `folds_run`, `tree_stable`, `tree_changes`)을 싣는다. P4·P5·J1 입력은 `judgeable` true인 실행에서만 쓴다 |
| 텍스트 입력 sha(v0.3.6) | csv·json·txt 같은 텍스트 입력은 원시 sha와 **LF 정규화 sha를 함께** 기록한다(`run_identity.inputs_sha256_lf`). 단언과 C1 sha 표는 **LF sha를 기준**으로 한다(Windows 체크아웃의 CRLF 변환으로 원시 sha가 달라질 수 있음). 바이너리(가중치·npy)는 원시 sha 그대로다 |
| 블라인드 | 지표(재현율·AP·포함률·M4 수·판정률)는 파일에만 쓴다. stdout에는 단계, 측선·타일·영상·주입 건수, 경과 시간, 경로, sha만 찍는다. 마지막 줄은 `{"status","out_dir","n_lines","n_tiles","n_images","n_injections","elapsed_s"}`(해당 키만)이다. Ultralytics 로거는 끈다 |
| 종료 코드 | 0 완료, 1 게이트 실패, 2 입력 거부, 3 내부 오류 |
| 정체성 기록 | `run_identity`: git HEAD·`dirty`·status, 코드 blob(러너·inputs·match·common·`m33_inputs.py`·`m33_r3_p4.py`·`m32_recipe_eval.py`·`m32_recipe_inputs.py`·`pipeline.py`·`to_image.py`·`inject.py`·`yolo.py`), 패키지 버전, GPU, argv, 위 sha 전부, 배치 크기·타일 순서 규칙. C1 sha 표에도 새 코드 blob(`m33_inputs.py`, `m33_r3_p4.py`, `m33_r3_band.py`, 바뀐 `m32_recipe_eval.py`·`m32_recipe_inputs.py`, 띠 함수를 더한 `src/gpr_detection/eval/inject.py`(S4 코드))을 넣는다. (C1 2단계) 러너별 `run_identity.code_blobs` 목록: **R2·R3** `m32_recipe_eval.py`·`m31_d1_common.py`·`m32_aihub_perturb.py`·`m32_pipe_responses.py`·`m32_recipe_inputs.py`·`m32_recipe_match.py`·`m33_inputs.py`·`m33_r3_p4.py`·`m33_r3_band.py`·`m33_r2_eval.py`와 src 4개(`pipeline.py`·`preprocess/to_image.py`·`eval/inject.py`·`detect/yolo.py`). **R1** `m33_morocco_eval.py`·`m31_d1_common.py`·`m32_aihub_perturb.py`·`m32_recipe_inputs.py`·`m33_inputs.py`와 src `pipeline.py`·`detect/yolo.py`·`morocco_eval.py`·`filters.py`. C1 sha 표에는 두 목록의 합집합을 적는다 |
| C1 봉인 기록 | **C1 sha 표의 값은 C1 봉인 기록 [`m33_c1_seal.md`](m33_c1_seal.md)에 둔다**(코드 blob, 텍스트 입력 LF·원시 sha, 가중치, 설정·r0 정체, D0 계보). 러너가 단언하는 값의 권위 출처는 `m33_inputs.py` 등의 상수이고, 봉인 기록은 그 사본이다 |
| C1 sha 표 추가 입력(C1 2단계) | R2 측선 목록 `analysis/detection/m33_r2_lines.csv` LF `57de56c6…`(빌더 `m33_r2_lines.py`), K 꼭짓점 SNR 표 `m33_k_apex_snr.csv` LF `55d54bb1…`, 모로코 매니페스트 LF `db2ebc30…`(원시 `b140c7b4…`), 모로코 목록 `eval_images.txt` LF `85b1666d…`(원시 `0e91de35…`). 전체 값은 C1 sha 표에 적는다 |
| 판정 불가 사유(v0.3.6 C1 1단계) | `judgment.blockers` 값: `smoke`, `partial_folds`, `line_limit`(측선 수 제한), `partial_recipes`(레시피 일부), `provisional`(plan을 P5a 구역 (c′) 없이 돈 비용 추정), `no_flip`(R2를 반전 방향 없이 돈 실행), `dirty_tree`. 하나라도 있으면 `not_judgeable`(종료 1)이다. fold가 없는 단계(R3 plan, CPU)는 `judgment.folds_run` = `null`이다 |
| 이어 돌리기 | (단계, 측선, fold) 부분 파일 + 설정 sha 검사(M3-2 S5 방식) |
| 자원 | GPU 단일 프로세스, batch 64. CPU 전처리는 허용 코어 affinity·BelowNormal·BLAS 1스레드. 30분 넘는 실행은 메인 세션 러너·예약 |
| AGPL | `ultralytics`는 `gpr_detection.detect` 안에서만 쓴다 |

- **C1 구현 항목(v0.3.6, 남은 위험)**: D0 실행기 개정(python-expert, 2026-11-15) 뒤에도 아래가 남아 있다. R1–R3 러너가 C1 봉인 전에 고쳐야 하고, 시험으로 확인한다.
  1. `m32_recipe_eval.py`(R2·R3 기반)는 아직 `period_samples`·`t_zero_ns`·`iou` 등을 4자리로 반올림해 쓴다. 전체 정밀도 요구에 맞게 바꾼다.
  2. `m32_recipe_eval.py`에는 fold 완전성 검사(판정 가능성)와 끝 시점 트리 재측정이 없다. R2·R3로 확장할 때 위 두 행을 넣는다.
  3. `m32_aihub_perturb._object_rows`는 값을 반올림한다. R1(`m33_morocco_eval.py`)은 이 함수를 그대로 가져다 쓰지 않는다.
  4. C1 sha 표의 텍스트 입력은 LF sha로 적는다(원시 sha 병기).
  - (C1 1단계, python-expert 2026-11-16) 1·2·4는 `m32_recipe_eval.py`·`m32_recipe_inputs.py`에 반영됐다(정밀도, fold·판정 가능성, 끝 트리, LF sha, 봉인 PP 단언). 3은 R1 단계에서 확인한다. **R2의 기대 레시피를 r0로 바꾸는 일은 R2 단계로 넘긴다.**
  - (R3 배선 뒤, 2026-11-16) R2·R1으로 넘길 것: R2는 박스 CSV를 R3의 `BOX_COLUMNS` 형식으로 쓰고 `summary.json`에 `status`와 `judgment`를 넣는다(R3 4TU plan이 `--r2-root`로 읽고 판정 가능한 R2만 받는다). R2 기대 레시피를 r0로 바꾼다. R1은 `m33_inputs`를 재사용한다.
  - 순서상 R3 4TU 배경은 R2 출력이 있어야 plan을 돌 수 있다. R3 IDS 배경만으로는 `--backgrounds` 부분 실행이라 `not_judgeable`이다.
  - **M3-2 영향(받아들임, 메인 판단)**: 공통 검사 때문에 M3-2 precheck 부분집합과 smoke 실행이 이제 `not_judgeable`(종료 1)로 끝난다. M3-2는 이미 끝났고 그 판정은 당시 코드로 봉인됐으므로 이 변화를 받아들인다.

## 2. R2 — 4TU 검출 보고 (`m32_recipe_eval.py --stage uninjected` 확장)

### 2-1. 확장 항목
1. **측선 목록 인자** `--lines <csv>`: 4TU 754측선(`lines_unique`, 활동 안 사본 5 제외, curator X7 확정 목록). 열은 `line_id, site, dataset, keep_one(bool), role(k/background)`이다. 지금의 단계 A 150 + K 27 고정 목록 대신 쓴다.
   - (C1 2단계 구현) R2 전용 로직은 새 모듈 `analysis/detection/m33_r2_eval.py`이고 `m32_recipe_eval.py --lines analysis/detection/m33_r2_lines.csv`로 부른다. 목록 파일은 빌더 `analysis/detection/m33_r2_lines.py`가 만들고 754측선 / `keep_one` 720 / K 27, LF sha `57de56c6…`이다. X7 탈락 목록과 일치한다. 열은 명세 5열에 `activity_id`·`path`·`n_traces`를 더한 8열이다. 레시피는 r0 하나다.
   - `boxes.csv` 앞부분 열은 R3 `BOX_COLUMNS`와 같다(R3 4TU plan이 `--r2-root`로 읽음). `summary.json`에 `status`·`judgment`·`timings`를 싣는다.
   - `--only-lines <line_id,…>`(smoke·진단용)를 주면 `line_limit` 판정 불가 사유가 붙어 `not_judgeable`(종료 1)이다. `--stage`와 `--smoke`는 함께 쓸 수 없다.
   - 출력: 정식 `runs/eval/m33/r2/`, smoke `runs/eval/m33/r2_smoke/`.
2. **레시피 고정** `--recipes r0`: M3-2 후보 r5·r6은 쓰지 않는다.
3. **PP 전·후 행**(§1): `boxes.csv`·`apex_hits.csv`·`m4` 집계에 `pp` 열을 더한다.
4. **집계 키**: 모든 행에 `keep_one`을 싣는다. 요약은 **720(`keep_one` true)이 주**이고 754는 민감도다.
5. **SNR 대리값 층**: X6(gpr-processing)이 주는 K 꼭짓점별 국소 SNR 4분위 표(`m33_k_apex_snr.csv`, 검출 무관)를 읽어 `apex_hits`에 붙인다.
6. **01.1형 폭**: 놓친 K 꼭짓점에 X6 `m33_k_apex_snr.csv`의 `width_vis_P`(보이는 범위 폭, 사용자 결정 2026-11-15)와 끝 닿음 플래그를 붙인다. 비교 기준은 AIHub p5–p95 7.79–14.67 P이고, 출처는 `m32_s1_recheck.py` `aihub_box_p()`(확인됨)다. `width_m6_P`·`width_m20_P`는 보고 열로 함께 싣는다.
7. **2사건 측선 표시**: `zero.n_events ≥ 2`인 측선을 `two_event` 열로 표시한다.
8. **J1 내보내기**(§5).
9. 정체 게이트: M3-2와 같은 r0 × K 27측선 × 5 fold batch 1 → D3 값(90 / 30 / 15). PP 전 박스로 본다. batch 64 대 1 경로 차는 보고만 한다.

### 2-2. 출력
- `runs/eval/m33/r2/`
  - `fold<k>/{boxes,apex_hits,conf_counts,flip_pairs}.csv`(`pp` 열 포함)
  - `lines.csv`(측선별 플래그·`keep_one`·`two_event`·$y_\text{bot}$·$y_z$)
  - `gate.json`
  - `summary.json`: 720 주 집계와 754 민감도로 K 포함률(5 fold 합/205, 아무 fold/41, 활동별), M4a–d·M4c·D3 8 px 값과 M4 측선 수, 클래스별 박스 수·점수 분위, $n_\text{agree}$, 반전 일관성, SNR 층, 2사건 층을 PP 전·후로 낸다. 함께 **PP 단계별 처리 수**(`removed_a`·`clipped_b`·`removed_b`·`removed_d`·`removed_c`)와 **PP-a 진단**(PP-a로 제거된 박스 중 위 끝이 내용 안, 즉 $y_1 < y_\text{bot} - P_ell$인 수, [PP] §4-4)을 fold별·5 fold 합으로 낸다. 키는 `pp_actions`, `pp_a_top_in_content`다.
- 지표 정의는 M3-2와 같다(M4는 `m4_category`, K 포함은 ±3 트레이스·±0.5 $P_\text{line}$).
- (C1 2단계 구현 해석, cv 판단 2026-11-16, 결과 없음)
  - $n_\text{agree}$는 D3 방식이다(fold 0 기준, 같은 클래스, IoU ≥ 0.5).
  - 2사건 층은 `two_event` 측선 층별 K 맞춤 수다.
  - 01.1형 폭 목록: 한 fold라도 놓친 꼭짓점(`n_folds_missed` > 0) 전부에 X6 열(`width_vis_P`·끝 닿음·`width_m6_P`·`width_m20_P`)을 붙이고 `n_folds_missed`(1–5) 열을 싣는다(§2-1 6의 "놓친 K 꼭짓점"은 fold를 가리지 않음, 보고 전용, 구현됨).

## 3. R3 — 의사 띠 (`plan`·`injected` 확장, `--injectant band`)

물리와 주입 구현은 [EMR] 부록 D(v1.9.1, D-6–D-8 포함)를 따른다. 띠 모드 구현(D-5)은 python-expert가 `inject.py`에 한다. 이 절은 러너와 지표 쪽 요구만 적는다.

**실행 인터페이스(v0.3.6 C1 1단계, python-expert 배선)**
- 명령: `m32_recipe_eval.py --injectant band`(`--stage`로 한 단계만 돌릴 수 있다. smoke는 `--stage` 없이 세 단계를 차례로 돌았다). 띠 전용 옵션은 `--backgrounds`(`4tu`·`gz_ids`, 하나만 주면 `not_judgeable`), `--r2-root`(4TU 미주입 박스를 R2 출력에서 읽음), `--c1b-record`다.
- C1b 기록: plan이 `plan/c1b_record_draft.json`(주입 표·위치 표·제외 표 LF sha)을 쓴다. C1b 커밋 때 이것을 `analysis/detection/m33_r3_c1b.json`으로 옮겨 고정하고, injected는 그 파일의 sha를 단언한다(정식 실행은 기본 경로, `--c1b-record`는 금지 인자). injected는 상류 plan의 `plan_report.json`도 확인하고 그 sha를 `run_identity.inputs_sha256.plan_report`에 남긴다.
- 출력(`runs/eval/m33/r3/<stage>/`): uninjected `fold<k>/boxes.csv`·`lines.csv`·`summary.json`, plan `injection_table.csv`·`positions.csv`·`exclusions.csv`·`plan_report.json`·`c1b_record_draft.json`, injected `truths.csv`·`rejections.csv`·`fold<k>/` 박스·일치 파일·`summary.json`(P4 값, §3-2).
- status: §1 공통값에 더해 `p4_defect`(P4 구현 단언 실패: 구조상 0 위반, 덮임 불변 위반, 설명되지 않는 유령, PP 결과 불일치)가 있고 종료 1이다.
- **실데이터 smoke 공개(2026-11-16, 사용자가 블라인드 조건으로 미리 허용)**: `conda run -n gpr-dev python -I -B analysis/detection/m32_recipe_eval.py --injectant band --smoke`, IDS 400 MHz 3측선·fold 0, 출력 `runs/eval/m33/r3_smoke`. 세 단계 모두 `not_judgeable`·종료 1(smoke라 정상), 측선 3, 타일 1,920, 주입 152, 544 s. 메인 세션은 종료 코드·status·건수만 보았고 값은 열지 않았다. 판정에 쓰지 않는다.
  - **시간(2026-11-16 정정)**: 1차 smoke(`r3_smoke`, 544 s, 추론 범위가 띠 주변 타일이던 구현)는 단계별 시간이 없었다. 보강 뒤 **2차 시간 smoke**(`--injectant band --smoke --out-root runs/eval/m33/r3_smoke2`, 사용자 사전 허용, 블라인드)는 세 단계 모두 `not_judgeable`·종료 1(정상), 3측선, 타일 2,548, 주입 152, 374 s였다. 단계별로 uninjected 7.7 s, plan 269 s, injected 69 s(주입 17, step (4)·영점 22, L1 23, 추론 5.8 s)다. 메인 세션은 종료 코드·status·건수·시간만 보았다. 두 smoke 모두 판정에 쓰지 않는다. 전체 추정은 §6이다. 각 단계 json에 `timings`, smoke는 `smoke_estimate.json`을 남긴다.

### 3-1. 확장 항목
1. **배경**
   - 4TU: 단계 A 150 + K 27측선. 모델은 A1 주, A0 민감도이고 `model_group`으로 나눈다. 보고 P6이다.
   - IDS: 단계 A 60측선, A0만, 30 ns 창은 상단 ≤ 1.0 m 모델만. 채택 근거 P4다.
   - **IDS 주파수 층**([EMR] 부록 D-6, v1.6). IDS 60측선은 주입 56(주 53 + 보고 3)과 주입 제외 4로 나눈다. 측선 표에 `antenna_MHz`와 `ids_freq_stratum`을 싣는다.

     | `ids_freq_stratum` | 측선 | 주입 | 쓰임 |
     |---|---|---|---|
     | `main_400` | 400 MHz 53 | 400 MHz H 맞춤(기존) | **P4 주 값** |
     | `report_other` | 200·600·900 MHz 각 1(3) | 측선마다 $P_\text{line}$으로 H를 맞춘다. 900 MHz 측선은 500 MHz 모델(M11·M14·M16)을 먼저 쓴다 | 보고 전용(주파수마다 1측선이라 채택 근거에 섞지 않음) |
     | `excluded_2ghz` | 2000 MHz 4 | **주입하지 않는다.** 합성 400 MHz 응답으로 흉내 낼 수 없다 | 미주입 추론과 구역 (c′)에만 쓴다 |
   - 2 GHz용 새 합성은 이번 범위 밖이다.
   - 셀은 `4tu`·`gz_ids`로 나눈다.
2. **미주입 IDS**: R3 안에서 r0 × 5 fold로 IDS 60측선 미주입 추론을 한다. 4TU는 R2 출력을 쓴다. 두 배경 모두 **띠 클래스를 포함한 모든 클래스** 미주입 박스를 구역 (c′)에 넣는다.
3. **주입체**: S4 새 띠 모드(`mode = "band"`, [EMR] D-5 `build_band_patch`)를 쓴다. `make_pseudo_band`는 쓰지 않는다. `_bg`는 배경에 더하지 않는다.
4. **배치**
   - 길이 표본: {40, 64, 100, 150, 200, 300} 트레이스. 4TU는 $L \le N - 2r_F$로 제한한다.
   - 제외 구역: (a)·(b)·(b′)·(c′)·(d)·바닥을 띠 가로 구간 전체에 적용한다([EMR] D-2). (c′)와 (d)의 정확한 형태는 다음과 같다(cv 결정 2026-11-15).
     - **(c′) 띠는 2D로 본다.** 미주입 박스(모든 클래스, 띠 포함, 모든 레시피 r0 × 5 fold 합집합)를 가로 ±1 P(트레이스)·세로 ±1 $P_\text{line}$(샘플)로 넓힌다. 그 사각형이 **띠 정답 박스**(가로 = 띠 구간 [i0, i1], 세로 = 꼭짓점 열 −20 dB 성분)와 겹치면 막는다.
       - 근거 1: 일치(시간 겹침 ≥ 정답 높이 50 %)와 유령 판정(미주입 짝 IoU)은 같은 깊이의 박스만 건드린다.
       - 근거 2: 이동 median은 행마다 따로 작용하므로, 다른 깊이의 실측 띠는 주입 띠의 생존을 바꾸지 않는다.
       - 근거 3: 가로 전용은 짧은 4TU 측선을 대부분 막아 표본과 대표성을 줄인다. M3-2 쌍곡선 (c) 측정에서 중앙값 53 %가 막혔다([CVR] 부록 B-1).
       - 다른 깊이의 강반사층은 (b′)가 따로 막는다.
       - [EMR] D-2의 "가로로 겹치지 않음"은 쌍곡선 규칙을 띠 가로 구간으로 넓힌다는 뜻으로 읽었다. 이 읽기는 분모·표본 정의(cv 몫)이고 물리 사슬은 바뀌지 않는다.
     - **보고**: 가로 전용 규칙이었다면 막혔을 위치를 `in_horizontal_box_zone` 열로 표시하고, 그 부분집합의 재현율을 따로 낸다(M3-2 `delta_old_horizontal_zone_c_subset`와 같은 방식).
     - **(d)는 구현 해석을 받아들인다.** 띠 구간 양 끝([i0, i1])에서 반경을 잰다. K 꼭짓점은 −20 dB 반폭, 자동 후보(`fourtu_apex_candidates_v1.csv`)는 −6 dB 반폭 + 1 P다. 쌍곡선 (d)와 같은 반경이다([PLAN] §3-2, [CVR] 부록 C-2). 자동 후보에 −20 dB 반폭을 쓰면 4TU 짧은 측선이 더 막히는데, 얻는 것이 없다. 자동 후보는 실관이 아닐 수도 있고, 띠와 겹쳐도 띠 일치에는 영향이 없다(띠 클래스만 셈).
     - `model_group`은 러너가 응답 표에서 붙인다(M3-2와 같음).
   - 간격: $|c_1-c_2| \ge (L_1+L_2)/2 + W_\text{eff}$, $W_\text{eff}=\min(301,N)$. 쌍곡선과 띠는 같은 실현에 섞지 않는다.
   - SNR 0/6/12/18 dB, 극성 ±1 짝(같은 위치·길이), 시드 20261115.
   - **배치 구현 해석(v0.3.6 C1 1단계, python-expert 구현 → cv 판단 2026-11-16, R3 결과 없음)**
     - 배치 함수는 `inject.plan_band_positions`다. 길이 표본 {40, 64, 100, 150, 200, 300} 트레이스, 시드 20261115.
     - 상한은 쌍곡선과 같은 측선당 위치 15, 실현 그룹 2다. 같은 그룹 안 간격은 위 $W_\text{eff}$ 식이다. 그룹이 다른 띠끼리는 주입 열(정답 구간 + 양 끝 테이퍼 $T/2$)이 겹치지 않게 $|c_1-c_2| \ge (L_1+T_1+L_2+T_2)/2$로 둔다. 위치끼리 배경 열을 나누지 않게 하는 보수적 선택이다. 위치 수가 줄면 §3-2 작은 층 처리가 받는다.
     - 4TU $L \le N - 2r_F$는 $r_F$를 테이퍼 트레이스 $T$(프레넬 반경을 짝수 트레이스로 반올림한 값)로 보고 $L + 2T \le N$으로 구현했다. 실제 주입 폭은 $L + T$이므로 이 조건은 그보다 $T$만큼 보수적이다.
     - 조건은 SNR 4 × 극성 2이고 파형 교체는 $k$ 1.3 한 갈래다([EMR] D-4 띠 모드는 H($k$ 1.3)). 쌍곡선에 있던 $k$ 0.8 보고 조건은 띠에 넣지 않는다.
     - `report_other`의 900 MHz 측선은 500 MHz 모델(M11·M14·M16)을 먼저 쓴다(위 표, [EMR] D-6). `plan_band_positions(preferred_models)`가 이 셋을 적격 모델 순서의 앞에 둔다(각 묶음 안은 시드 순서). 다른 측선의 plan은 바뀌지 않는다. plan 보고에 `lines[].preferred_models`를 남긴다. 보고 전용 층이라 P4에는 영향이 없다. 같은 그룹 간격 $(L_1+L_2)/2 + W_\text{eff}$와 다른 그룹 간격 $(L_1+T_1+L_2+T_2)/2$는 식을 직접 계산하는 시험으로 확인한다.
     - **주입 실현의 추론 범위는 측선 전체다**(2026-11-16 정정, 판정 경로 영향). r0 사슬에 측선 전체 통계가 세 개 있다: 회색 클립 수준(측선 전체 |a|의 99번째 백분위), 주기 측정(단계 (4) 단면 전체 평균 rfft 최대), AGC 바닥값(측선 전체 `mean_power.max()` × 1e-12). 그래서 띠를 넣으면 이동 median 창보다 먼 열의 화소도 바뀐다(시험 `test_a_band_changes_the_r0_image_far_beyond_the_median_window`). 이전의 "띠 열 ± $W_\text{eff}/2$ + 1 stride" 타일 범위는 이 전제가 틀려 버렸다. `inference_tiles`는 측선 전체 타일을 돌려주고, 실현별 추론 범위는 `tile_trace_range`로 기록한다. **채택하지 않은 대안**: 클립 수준·$P_\text{line}$·AGC 바닥값을 배경 값으로 고정하는 방식이다. src 변경과 판정 경로 변경이 필요해 쓰지 않았다. 측선 전체 추론은 운용 경로와 같아서 판정 경로를 바꾸지 않는다. 실현 영상은 fold와 무관하게 한 번만 만들어 모든 fold에서 쓴다(시험으로 확인).
     - 유령 짝(미주입 IoU ≥ 0.5)은 미주입 박스를 주입 실현의 추론 열 범위로 자른 뒤 잰다(`clip_pair_pool`). 지금은 범위가 측선 전체라 잘리는 박스가 없다. 범위를 줄이는 구현이 다시 생길 때를 위한 장치다.
     - summary 재현율·유령 표는 C3-6 주 칸 정답만으로 만든다. 바닥 칸(주 칸 밖) 정답은 같은 지표로 따로 낸 보고 표 summary `bottom`(`report_only`)으로 둔다.
     - 유령은 정답의 층 키로 묶고 묶음 안에서 실현별로 서로 다른 박스를 센다. P4의 `ghost_delta`(두 층 합)는 두 층을 합친 범위에서 실현별로 다시 중복을 뺀다(구현됨). 층별 표의 합은 P4 값과 다를 수 있다(보고 전용).
     - px 층 경계는 < 64 / 64–192(192 포함) / > 192 L1 px다([PLAN] §2-3과 같은 값. 가운데 층은 해석하지 않는다).
     - PP 내용 범위($y_\text{bot}$, $P_\ell$)는 주입 실현의 r0 영상 자신의 값이다. 박스 좌표가 그 영상에서 나오므로 같은 영상의 값을 써야 한다. 영점 $y_z$는 배경 측선 값을 쓴다(구역 (b)가 띠를 영점 + 2 P 아래에 둠). summary에 `n_realizations`, 주입으로 주기 추정이 바뀐 실현 수 `n_period_changed`, 주입 자료로 다시 잰 영점이 배경 영점과 다른 실현 수 `n_zero_changed`를 싣는다.
     - 내용 범위 밖 채움 영역에만 있어 L0 박스가 없는 박스는 일치(R·G·G′·재현율)에서 뺀다. 정답과 겹칠 수 없기 때문이다. 박스 CSV에는 남긴다. `pp_acted_boxes_all`에는 넣지 않고, 그런 박스의 작용 수를 `pp_acted_padding_only`(보고 전용, summary 전체 값과 P4 범위 값)로 따로 낸다.
   - **층 최소 수**: 배경마다 $L/W_\text{eff}$ 층마다 위치 ≥ 30을 목표로 한다. 모자라면 수를 공개하고 멈추지 않는다.
   - plan 보고에는 층별 위치 수를 낸다. P4 두 층(< 0.33, 0.33–0.5)의 IDS 위치 수도 따로 낸다.
   - **PP 노출 위치 수**(v0.3.5 개정. 사용자 결정 2026-11-15: D0 결과(C0b, S3 선택)를 안 뒤, C1 전, R1–R3 측정 전에 정함)
     - plan 보고에 P4 두 층의 PP 노출 위치 수 `pp_exposed_positions`(층별과 두 층 합)를 낸다.
     - **노출 위치**는 **C0b 선택 집합 $S_\text{sel}$에 든 규칙만** 띠 정답 박스(주입 표의 정답 박스, L1 측선 px)에 [PP] §2 조건(원래 박스 기준, 범주 순서 a > b > d > c)대로 적용해서 하나 이상에 해당하는 위치다. C0b 봉인값은 `pp_set_id` = `S3` = {a, b, d}다.
       - PP-a($\in S_\text{sel}$일 때): $y_m > y_\text{bot}$.
       - PP-b($\in S_\text{sel}$일 때): a가 아니고 $y_1 < y_\text{bot}-e$이고 $y_2 > y_\text{bot}+e$.
       - PP-d($\in S_\text{sel}$일 때): a·b가 아니고, 그 측선의 $y_z$가 있고, $y_m < y_z + 0.5P_\text{line}$.
       - PP-c($\in S_\text{sel}$일 때만): a·b·d가 아니고 $y_2 \ge y_\text{bot}-w_c$. 점수 조건은 검출 쪽 값이라 노출 정의에서 뺀다.
     - 러너는 `pp_set_id`와 `PPConfig` 해시가 C0b 값과 같은지 단언한다. 규칙 목록은 `pp_set_id`에서 읽고, S3를 코드에 박지 않는다. 노출은 검출과 무관하며 C1b에 고정한다. 규칙별 수 `pp_exposed_by_rule`도 낸다.
     - 보고 전용: PP-c 창(아래 끝 ≥ $y_\text{bot}-w_c$)에 드는 위치 수 `pp_c_window_positions`를 함께 낸다. $S_\text{sel}$에 c가 없으면 이 수는 노출에 넣지 않는다.
     - 공개(개정 이유): v0.3.3–v0.3.4 정의는 선택 집합과 상관없이 (i) PP-c 창, (ii) PP-d, (iii) PP-a·b 가운데 하나에 드는 위치였다. [EMR] D-8-4 기하 추정은 (ii)·(iii)을 구조상 0(0/695)으로 보았다. 그래서 옛 정의의 노출은 사실상 (i)뿐이었는데, S3에는 PP-c가 없다. 즉 작동하지 않는 규칙의 창이 노출로 세어져 `p4_status`가 `no_effect`로 갈 수 없었다. v0.3.5 개정은 이 불일치만 없앴고 문턱·범위·층·가중 규칙은 바꾸지 않았다. 유령 조건을 보고 전용으로 바꾸고 경계 비교를 정한 것은 v0.3.6이다(§3-2). 옛 정의에서 쓴 [EMR] D-8-4 수치(PP-c 창 57/695, 약 8–11 %, "−1.0 %p ≈ 노출 위치 −10 %p")는 PP-c가 든 집합(S4·S5)에만 해당하는 이력이다.
   - **노출 0·작용 0 처리**(v0.3.5 신설, v0.3.6 개정: stats-design 검토와 사용자 결정 2026-11-15, D0 결과 뒤·C1 전·R1–R3 측정 전. 결과와 무관하게 미리 정함)
     - **작용 집합**: P4 범위(`gz_ids`, `main_400`, A0, 두 층)에서 **PP 전 박스**를 기준으로 셋을 나눈다. 모두 검출 쪽 값이고 5 fold 합이다.
       - (R) 재현율 집합: 주입 실현의 conf > 0.25 `pipe_band` 박스 중 정답과 시간 겹침 ≥ 정답 높이 50 %이고 [i0, i1]과 가로로 겹치는 박스. 곧 6의 재현율에 들어가는 박스다. 한 박스가 두 정답에 해당하면 각 정답에서 센다.
       - (G) 유령 후보: 같은 시간 겹침 조건이고 일부가 [i0 − 테이퍼, i1 + 테이퍼] 밖에 있는 박스.
       - (G′) G 박스와 PP 전 또는 PP 후 IoU ≥ 0.5인, 같은 측선·fold의 미주입 `pipe_band` 박스. 짝 후보 풀은 conf > 0.25인 미주입 박스로 거른다(운용 문턱. C1 실행 전 감사에서 드러난 미문서 해석을 적은 것이다).
       - 층: R·G는 그 박스가 시간 겹침 조건을 만족한 **정답 띠의** `lw_stratum`을 쓴다. 한 실현에 길이가 다른 띠가 여럿 들어 있으므로 실현 단위가 아니다. G′는 짝지어진 G 박스의 층을 따른다. 미주입 박스는 측선·fold마다 하나뿐이고 그 측선의 여러 실현(SNR × 극성 × 실현)과 짝지어질 수 있다. 그래서 G′는 (실현, G 박스, 미주입 박스) 짝 단위로 세고, 중복을 뺀 미주입 박스 수도 함께 낸다.
     - **상태 결정용** `pp_acted_boxes`: R 박스 가운데 `pp_action` ∈ {`removed_a`, `removed_b`, `removed_d`}인 수다. 층별과 두 층 합으로 낸다. `pp_acted_by_rule`(규칙별)과 `pp_acted_lines`(R 안 제거(`removed_a`·`removed_b`·`removed_d`)가 1개 이상인 측선 수, 두 층 합은 중복 없이)를 함께 낸다. `n_r_boxes`(R 박스 수, `pp_acted_boxes`와 같은 계수 기준)와 `n_r_lines`(PP 전 R 박스가 1개 이상인 측선 수)를 층별과 두 층 합으로 낸다.
     - `pp_acted_clip_recall_set`: R 안의 `clipped_b` 수다. 러너는 해당 정답마다 덮인 길이가 PP 전·후로 같은지 단언한다(정수 트레이스 일치). 다르면 구현 결함으로 종료 1이다. 이 단언은 "PP-b 자르기는 재현율을 바꾸지 않는다"는 전제를 실행 중에 확인한다. 전제의 근거: C3-6 주 칸은 정답 마지막 행 ≤ $n_s - 1 - P_\text{line}$(L0)이다. 측선 영상($\text{sample\_offset}=0$)에서 $y_\text{bot} = n_s/\sigma_y$(PP, L1)이므로 정답 아래 끝은 L1에서 $y_\text{bot}-P_\ell$ 이하다. 자르기는 $y_2$를 $y_\text{bot}$까지만 줄이므로 정답과의 시간 겹침이 바뀌지 않는다. 같은 이유로 `removed_b`(자른 높이 < $P_\ell$)는 정답과 시간이 겹치지 않아 R에 들 수 없다. 그래도 R 작용 집합에는 넣어 둔다.
     - `pp_acted_ghost_set`(보고 전용): G·G′의 작용 수(`pp_action` ≠ `none`)다. 유령 증가를 설명하는 대조에만 쓴다.
     - `pp_acted_boxes_all`(보고 전용): P4 범위 모든 `pipe_band` 박스(주입·미주입)의 작용 수다. 정답과 짝이 없는 박스는 층이 정해지지 않으므로 층별 값 없이 `total`과 `by_rule`만 낸다.
     - P4 범위는 C3-6 주 칸(`main_cell` true)인 위치만이다. 노출·R·G·G′·재현율 모두 같다. 바닥 칸은 보고 층으로 간다.
     - 좌표: 일치(시간 겹침, 가로 덮임)와 IoU는 L0 정수 격자의 닫힌 구간에서, 노출은 PP와 같은 L1 px(`ContentRows`, 정답 L1 박스는 L0 셀 가장자리 변환)에서 계산한다.
     - `p4_status`는 아래 순서로 정한다.
       1. `not_evaluable`: 두 층의 IDS 위치가 모두 0이다(§3-2 작은 층 처리).
       2. `no_effect`("작용 없음(정보 없음)"): `pp_exposed_positions`(두 층 합) = 0이고 `pp_acted_boxes` = 0이다. **R에 제거 작용이 없으므로 `recall_delta`는 구조상 정확히 0이다(단언). `ghost_delta`는 0이 아닐 수 있으며 보고하고 유령 구현 단언(6)을 따른다.** 러너는 `recall_delta` == 0(부동소수 완전 일치)을 단언하고 결과를 `p4_structural_zero`(bool)에 싣는다. 어기면 구현 결함으로 종료 1이다. P4는 채택 근거에서 빠진다. 다만 차가 0이므로 규칙을 적용해도 통과하며, 상태 분기는 권고 확률을 바꾸지 않는다([PLAN] §3-5). 판정서에는 0/`n_r_lines` 상한을 적는다([PLAN] §3-5). 분모는 53이 아니다.
       3. `acted_unexposed`: `pp_exposed_positions` = 0이지만 `pp_acted_boxes` > 0이다. 정답 박스는 규칙 조건 밖이지만, 정답보다 길거나 어긋난 예측 박스를 PP가 지운 경우다. **P4 판정 규칙을 그대로 적용하고 채택 근거로 남긴다.** 작용 박스 목록 `p4_pp_acted.csv`(측선, fold, 실현, 박스, 집합 R·G·G′, `pp_action`)를 낸다. `recall_delta_exposed`는 `null`이다. (v0.3.6 C1 1단계) `p4_pp_acted.csv`는 상태와 상관없이 항상 쓰고, R 안 `clipped_b`와 G·G′ 작용도 집합 열로 구분해 싣는다. 판정서에 옮기는 것은 `acted_unexposed`·`evaluated`일 때다.
       4. `evaluated`: `pp_exposed_positions` > 0이다. P4 판정 규칙을 그대로 적용한다.
     - **S3에서 예상되는 경로(공개, [EMR] D-8-4 기하 추정)**: S3에서는 정답 띠 박스 노출이 구조상 0이다. 구역 (b)가 꼭짓점 ≥ 영점 + 2 P를, C3-6이 기록 끝 걸침을 막기 때문이다. 그래서 상태는 `no_effect`나 `acted_unexposed` 가운데 하나로 예상하고, `recall_delta_exposed`는 `null`이다. PP-b 자르기는 정답 시간 겹침을 바꾸지 않아 `recall_delta`에 영향이 없다. 재현율 손실은 a·d가 정답과 겹친 박스를 지울 때만 생긴다. 이 문단은 경로만 말하고 어느 상태가 나올지는 예측하지 않는다.
     - 문턱 비교와 상태 결정은 R3 injected 결과가 나온 뒤 기계적으로 한다. 상태 이름과 순서는 C1에 고정한다.
   - **띠 배치 순서**: a0dc987 `inject.py` 띠 모드 구현 그대로 쌍곡선과 같은 바닥 우선 순서(`_bottom_first_order`, 꼭짓점 ≥ $n_s$ − 3 P 후보를 측선당 1개 먼저)를 쓴다. 바닥 우선 후보 36쌍 중 30쌍이 C3-6에 걸려 주 칸 밖(바닥 층 보고)으로 빠진다([EMR] D-8-4). 이 순서는 노출을 늘리는 장치가 아니며, 그 사실을 공개한다. 순서는 C1b 주입 표 sha로 고정된다.
   - **IDS 400 MHz 층 구성(공개)**: IDS 400 MHz 53측선은 89–6,665트레이스(중앙 426)이고 16측선이 301 미만이라 $W_\text{eff} = N$이다. 그래서 층은 L과 N으로 정해진다. N ≥ 301인 37측선에서는 `lt033` = {40, 64}, `033_05` = {100, 150}이고, 짧은 측선에서는 `033_05`에 L 40·64·100이 들어오고 L 100·150 일부는 긴 두 층으로 간다(길이별 들어갈 수 있는 층의 측선 수: L 40 — `lt033` 49 / `033_05` 4, L 64 — 45 / 3 / `05_067` 4 / `ge067` 1, L 100 — `033_05` 44 / 긴 두 층 6 / 불가 3, L 150 — `033_05` 37 / 긴 층 10 / 불가 6, [EMR] D-8-1). L = 100은 0.3322로 경계 바로 위다(경계 리터럴 0.33, 시험으로 단언). L = 150은 0.498로 생존 갈림점 W/2(0.5) 바로 아래(0.5트레이스 차)이고, 끝 테이퍼 때문에 생존 비가 SNR과 거의 무관한 약 0.56–0.62인 약한 띠다([EMR] D-8-2). **L = 150은 유지한다**(cv 결정 2026-11-15: median이 띠 값으로 넘어가지 않는 마지막 정수 길이라 배정이 타당하고, 약한 띠라 PP-c에 오히려 민감한 사례다. 대안 L 136은 쓰지 않는다). fold SD는 다섯 fold가 같은 위치를 공유하므로 표본 SE가 아니다.
5. **정답 표 열**(위치 표·`inj_matches`): `mode`, `band_length_traces`, `band_center_trace`, `length_px`, `w_eff`, `l_over_w_eff`, `lw_stratum`(`lt033`·`033_05`·`05_067`·`ge067`), `px_stratum`, `l_over_n`, `taper_traces`, 정답 박스(L0), `gain`, `noise_rms`, `signal_amplitude`, 잘린 칸 수·최대 주입 비율, 극성, SNR, $k$, `response_id`, `model_group`.
6. **일치**(PP 전·후 각각)
   - **가로 구간 재현율**: conf > 0.25 `pipe_band` 박스 가운데 정답과 시간 겹침 ≥ 정답 높이 50 %인 박스의 가로 합집합을 본다. 그것이 정답 구간 [i0, i1] 안에서 덮는 길이 / 정답 길이이고, 길이 가중이다. 정답 구간 밖 부분은 재현율에 넣지 않는다.
   - **유령 띠 오검출**: 다음 세 조건을 모두 만족하는 박스는 주입이 만든 오검출로 센다.
     - 같은 실현의 `pipe_band` 박스로, 정답과 시간 겹침 ≥ 50 %다.
     - 가로로 [i0 − 테이퍼, i1 + 테이퍼] 밖에 있다.
     - 같은 측선·fold 미주입 박스(같은 클래스)와 IoU ≥ 0.5인 짝이 없다.
     - 정답 구간과 걸친 박스는 구간 밖 부분 길이를 유령 길이로 센다.
     - 출력: 실현별 `ghost_n_boxes`, `ghost_length_traces`.
     - (v0.3.6 C1 1단계 해석, cv) "밖"과 "구간 밖 부분"의 구간은 테이퍼를 넣은 [i0 − T, i1 + T]다. 일부라도 그 밖에 걸치면 유령 후보다. `ghost_n_boxes`와 `ghost_delta`는 **실현마다 서로 다른 박스 수**로 센다. 한 박스가 같은 실현의 두 정답에 대해 유령이어도 1이다. (정답, 박스) 짝은 `ghost_pp_artifacts.csv`의 상세로만 둔다. `ghost_length_traces`는 실현마다 유령 박스들의 [i0 − T, i1 + T] 밖 부분의 합집합 길이(L0 트레이스)다. 둘 다 보고 전용이다.
     - **PP 후 유령 판정의 짝**: PP 후 박스의 미주입 짝은 **같은 PP를 적용한** 미주입 박스에서 찾는다(PP 전은 PP 전끼리). PP는 박스를 만들지 않으므로, PP 후에 새로 유령이 된 박스는 후보 자신의 자르기나 짝 박스의 제거·자르기로만 생긴다.
     - **유령 구현 단언**(v0.3.6, 사용자 결정 2026-11-15): PP 후 유령이고 PP 전 유령이 아니던 박스를 모두 `ghost_pp_artifacts.csv`에 싣는다(측선, fold, 실현, 박스, 후보의 `pp_action`, 짝이던 미주입 박스와 그 `pp_action`). 러너는 이 박스마다 후보 또는 짝의 `pp_action` ≠ `none`인지 단언한다. 하나라도 그렇지 않으면(설명되지 않는 증가) 구현 결함으로 종료 1이다. 설명되는 증가는 보고만 한다.
   - 클래스 무시 덮임(다른 클래스로 읽은 비율)은 보고만 한다.
7. **생존 진폭 비**: [EMR] D-5 2의 식 $s$를 r0 (4) 단면에서 잰다. 검출기와 무관하므로 fold와도 무관하고, 실현마다 1회 잰다.
8. **주입 표 고정**: plan 단계 뒤 주입 표 sha를 **C1b** 커밋에 적고, 그 뒤 injected를 실행한다. 규칙 변경은 없다(M3-2 F2 방식).

### 3-2. 출력
- `runs/eval/m33/r3/{plan,uninjected_ids,injected}/`
- `summary.json`
  - 배경(`4tu`·`gz_ids`) × 모델 묶음 × `lw_stratum` × SNR × 극성마다: 가로 구간 재현율과 유령 띠 오검출(수·길이). 모두 PP 전·후, 5 fold 평균 ± SD다.
  - 생존 진폭 비: SNR × `lw_stratum` × 배경.
  - px 층 보고.
- **P4 값**(`p4_adoption_input`, 키 이름 고정)
  - `recall_delta`: `gz_ids`, **`ids_freq_stratum` = `main_400`(53측선)**, A0, `lw_stratum` ∈ {`lt033`, `033_05`}. 층 같은 가중, 층 안에서 SNR 4 × 극성 2 균등, PP 후 − PP 전, 5 fold 평균. **판정 기준은 `recall_delta` ≥ −1.0 %p − 1e-9다**(v0.3.6: 정확히 −1.0이면 통과, 부동소수 오차 허용 1e-9 %p). P4 판정은 이 재현율 조건 하나다.
    - 개수 모형(공개, stats-design 검토): 두 층 위치가 n씩이면 층마다 실현-fold는 n × 8(SNR × 극성) × 5 fold = 40n개다. 한 실현-fold의 정답을 완전히 잃으면(덮임 1 → 0) −1.25/n %p이고, 문턱 −1.0 %p는 전체 실현-fold의 1 %, 곧 0.8n개에 해당한다. 길이가 같을 때의 근사다. 길이 가중이므로 정답 하나를 잃을 때의 값은 −1.25·(L/L̄)/n %p다(L̄: 그 셀 정답 길이의 평균, 사용자 결정 (A) 2026-11-16).
  - `ghost_delta`(v0.3.6부터 **보고 전용**): 같은 범위의 유령 띠 박스 수 5 fold 합, PP 후 − PP 전. 판정 기준이 아니다. 증가분은 §3-1 6 유령 구현 단언을 따른다(설명되면 보고, 설명되지 않으면 종료 1).
  - `n_positions_by_stratum`: 두 층의 위치 수.
  - `n_lines`: 53(`main_400`)을 단언한다.
  - **작은 층 처리**(사용자 결정 2026-11-15, C1b의 위치 수로 정해지고 결과와 무관)
    - 두 층 모두 IDS 위치 ≥ 30이면 위 정의(층 같은 가중) 그대로다(`stratum_mode` = `two_strata`).
    - 한 층이라도 위치 < 30이면 두 층을 합친 $L/W_\text{eff}$ < 0.5 한 층(층 가중 없이 두 층 단위를 합침, 셀 안 길이 가중)으로 낸다(`stratum_mode` = `pooled`). 한 층이 0이면 합친 층(= 남은 층)만 쓴다.
    - 둘 다 0이면 `recall_delta`·`ghost_delta`는 `null`이고 `stratum_mode` = `not_evaluable`이다.
    - 합친 층을 쓰면 그 위치 수의 최소 검출 효과(MDE)를 `mde_pp`로 적는다. `mde_pp`는 위 개수 모형(실현-fold 단위 손실, 측선 군집 반영) 기준으로 계산한다. (v0.3.6 C1 1단계, cv 판단) 러너는 `mde_pp` = `null`과 계산에 필요한 수(층별 위치 수, 측선 수, 측선별 위치 수)만 싣는다. 값은 C1b plan 뒤·injected 전에 stats-design-expert가 길이 가중을 반영한 개수 모형 시뮬레이션으로 계산해 판정서에 적는다. 검출과 무관한 위치 수만 쓰므로 블라인드를 깨지 않고, 보고 전용이라 판정에 쓰지 않는다. 참고: 실제 −2 %p여도 통과 확률은 약 0.1–0.6이다(손실이 소수 측선에 몰릴수록 높음, [추정]). 검정력 80 % 최소 효과는 1.5–7.25 %p다(stats-design 합성 시뮬레이션, 위치 30–100·합친 40).
  - `pp_set_id`·`pp_config_sha256`: C0b 값(단언 결과 포함).
  - `pp_exposed_positions`: §3-1 4(v0.3.5) 노출 위치 수(층별과 두 층 합). `pp_exposed_by_rule`(규칙별), `pp_c_window_positions`(보고 전용)를 함께 싣는다.
  - `pp_acted_boxes`(상태 결정용, R 안 제거 수), `pp_acted_by_rule`, `pp_acted_lines`, `pp_acted_clip_recall_set`(R 안 `clipped_b` 수와 덮임 불변 단언 결과), `pp_acted_ghost_set`(보고 전용, G·G′ 작용 수, 짝 단위와 중복 뺀 미주입 박스 수), `pp_acted_boxes_all`(보고 전용): §3-1 4(v0.3.6), 5 fold 합, 층별과 두 층 합(`pp_acted_boxes_all`만 `total`·`by_rule`).
  - `n_r_boxes`, `n_r_lines`(§3-1 4, 층별과 두 층 합, 측선은 중복 없이), `pp_acted_lines_by_stratum`, `recall_pre`(같은 범위·같은 가중의 PP 전 가로 구간 재현율, **0–1 비율**, 5 fold 평균. 공개 기준 "`recall_pre` < 1.0 %p"는 비율로 0.01이다), `action_line_rate_upper95`(Clopper–Pearson 양측 95 % 상한, x = `pp_acted_lines`, n = `n_r_lines`, n = 0이면 `null`, 예 0/53 → 0.0672(6.7 %), 보고 전용)와 `action_line_rate_note`(n = 0이면 "상한 없음(R 없음)", 0 < n < 30이면 "상한이 넓다(X %)", 그 밖에는 보고 전용 문구. python-expert 구현과 같은 뜻). 상태와 상관없이 싣는다(사용자 결정 2026-11-16, R1–R3 결과 전).
  - `n_truth_folds_by_cell`: P4 범위 층 × SNR·극성 셀 × fold별 정답×fold 단위 수(`by_stratum_cell_fold`), 기대 셀 8개(`expected_cells`), 층별 빈 셀(`missing_cells_by_stratum`), `cells_complete`. 빈 셀은 보고만 한다(판정을 멈추지 않음).
  - `p4_status`: `evaluated`·`acted_unexposed`·`no_effect`·`not_evaluable`(정하는 순서는 §3-1 4). `no_effect`이면 R에 제거 작용이 없으므로 `recall_delta`는 구조상 정확히 0이다(단언, `p4_structural_zero`). `ghost_delta`는 0이 아닐 수 있으며 보고하고 유령 구현 단언을 따른다. `acted_unexposed`이면 `p4_pp_acted.csv` 경로와 sha를 싣는다.
  - `ghost_pp_artifacts`: `ghost_pp_artifacts.csv` 경로·sha·행 수와 설명되지 않는 증가 0 단언 결과.
  - `recall_delta_exposed`(보고 전용, 판정 기준 없음): 같은 범위·같은 가중 규칙에서 노출 위치(v0.3.5 정의)만으로 낸 PP 후 − PP 전 가로 구간 재현율, 5 fold 평균. 노출 위치 수(`n_exposed_positions`)와 측선 수를 함께 싣는다. 노출 0이면 `null`이다.
- 긴 두 층(`05_067`, `ge067`), 4TU 배경, IDS `report_other` 3측선은 `report_only`에 둔다. `summary.json`에 IDS 측선 구분 `{"main_400": 53, "report_other": 3, "excluded_2ghz": 4}`를 싣는다.

## 4. R1 — 모로코 (`m33_morocco_eval.py`, 새로 작성)

| 항목 | 요구 |
|---|---|
| 입력 | `data/morocco_eval_v1/eval_images.txt`(233), 라벨, 매니페스트 `data/splits/morocco_eval_v1.csv`. sha와 사본 0·그룹당 1장을 다시 단언한다 |
| 경로 | 주 경로는 letterbox 256(224 → 256 리사이즈, 1채널)이다. S3와 같은 predict 경로를 쓰고 fold 0–4 모두 돈다 |
| 클래스 | 정답 클래스 0(`pipe_hyperbola`)·2(`cavity`). 예측 4클래스를 그대로 쓴다. 하위셋(`utilities`·`cavities`·`intact`)은 매니페스트 열로 나눈다 |
| 지표 | 관로 + 이상 없음 하위셋 `pipe_hyperbola` AP50·AP50-95, conf 0.25 재현율·정밀도. 이상 없음 장당 오검출(클래스별). 공동 하위셋 `cavity` AP50·재현율(도메인 분리, 전체 지표에 넣지 않음). 영상 부트스트랩 95 % CI(B 2000, 시드 20261115). 혼동 행렬(배경 포함) |
| 민감도(보고) | (s1) 리사이즈 없이 중앙 패딩만. (s2) 축척 0.75·1.0·1.33배(주 경로 대비 등방). (s3) 상단 흰 세로선 영역 가림: X6에 행 범위가 없어 cv가 고정한다. 원 영상 행 0–27(위 1/8)을 128로 채운다. (s4) 근접 중복 쌍 U_233·U_239를 하나로 센다. (s5) **주기 맞춤 축소**: 영상별 $P_\text{native}$(X6 `m33_x6_period_snr.json`)로 $P_\text{in}$ = **$P_A$ 4.06 px**가 되게 등방 축소(INTER_AREA)한다. 관로 기준 배율은 중앙 0.16배(0.08–0.19)다. $P_A$를 기준으로 삼은 것은 AIHub 학습 분포에 맞추려는 취지이기 때문이다. $P^*$ 6 기준(0.23배)은 쓰지 않는다([PLAN] §2-1). 256 캔버스 가운데에 두고 나머지는 128로 채우며, 정답 박스도 같은 변환을 한다. P5에는 쓰지 않는다 |
| PP | 정사각 입력이라 채움이 없고($y_\text{bot}$ = 256) $y_z$도 없다. 그래서 PP-a·b·d는 작용하지 않는다. PP-c는 $y_\text{bot}$ = 256 기준으로 작용할 수 있다. PP 전·후를 모두 내고, **P5 값** = 관로 + 이상 없음 `pipe_hyperbola` AP50(PP 후 − PP 전, 5 fold 평균)이다. 키 이름은 `p5_adoption_input`이다(이름은 유지하지만 [PLAN] v0.2.5에서 P5는 채택 근거가 아니라 작용 확인이다). PP-c는 `pipe_band`만 지우고 AP는 클래스별이므로 P5는 구조상 0이다. 러너는 fold마다 PP 전·후 `pipe_hyperbola` AP50이 같음(`p5_adoption_input == 0`, 부동소수 완전 일치)을 단언하고, 어기면 구현 결함으로 종료 1(게이트 실패)이다. 단언 결과는 `summary.json`의 `p5_structural_zero`(bool)에 싣는다. (C1 2단계, smoke 1차에서 고친 결함) 단언은 명세 문구대로 "fold마다 PP 전 AP50 == PP 후 AP50"이고, 양쪽 모두 None(그 fold에 정답 또는 예측이 없어 AP가 정의되지 않음)이면 같음으로 본다. 그런 fold 수를 `p5_folds_without_ap50`로 보고한다. `p5_adoption_input`은 AP50이 정의된 fold만으로 평균한다. 판정서에 `p5_folds_without_ap50`을 싣고, 0이 아니면 "P5 작용 확인은 일부 fold에서만 확인됐다"고 공개한다 |
| 1회 실행 | 정식 실행은 한 번만 한다. 출력 폴더가 있으면 거부한다(`--overwrite` 없음). smoke는 8장, 별도 폴더 |
| 출력 | `runs/eval/m33/r1/fold<k>/{predictions,objects}.csv`, `metrics.json`, `summary.json`, 사례 PNG(상·하위 12장, `runs/` 아래에만 두고 공개 저장소에 넣지 않음). smoke는 `runs/eval/m33/r1_smoke/` |
| 사례 PNG 순위(C1 2단계, cv 결정) | fold 0, 주 경로, PP 전, conf > 0.25, 클래스 일치 IoU ≥ 0.5 탐욕 매칭에서 영상별 오류 수 $e$ = 미검출 + 오검출을 센다. **하위 12** = $e$가 큰 순(동률은 영상 이름순). **상위 12** = 정답이 있는 영상 가운데 $e$가 작은 순(동률은 맞춘 박스 점수 평균이 높은 순, 다음 이름순). 보고 그림 전용이다 |
| 구현 해석(C1 2단계, cv 판단 2026-11-16, 결과 없음) | (0) PP 내용 범위는 모든 조건에서 `CROP_CONTENT`($y_\text{bot}$ = 256)를 쓴다. s1(패딩만)·s5(주기 맞춤 축소)도 같다. 이 두 조건은 256 캔버스 안에 128 채움이 생기지만 PP 내용 범위로 보지 않는다(C1 실행 전 감사에서 드러난 미문서 해석, 보고 조건이라 P5에는 쓰지 않음). (1) 주 경로 리사이즈는 명시적 `INTER_LINEAR`(letterbox와 같은 보간)다. (2) s2 1.33배는 가운데를 잘라 256에 맞추고 정답 박스도 같이 자른다. 잘린 면적이 0인 박스는 버리고 그 수를 보고한다. (3) s4는 근접 중복 쌍에서 `U_239_aug_6`을 빼고 `U_233` 쪽을 남긴다. (4) 혼동 행렬은 클래스 무시 탐욕 매칭(IoU ≥ 0.5, conf > 0.25, 점수 내림차순)이다. 배경 행·열을 포함한다. (5) **부트스트랩은 5 fold 평균값에 대해 낸다**: 영상을 복원 추출한 같은 표본에서 fold마다 지표를 계산해 평균한다(B 2000, 시드 20261115, 주 경로, PP 전). summary 키 `bootstrap_fold_mean`(구현됨). 보고 전용이다. (6) s5의 영상별 $P_\text{native}$는 X6 json에 분위수만 있어 `prepare_fitting_branch(...).period_px`로 영상마다 다시 잰다(결정적, 233장 모두 측정됨). summary `s5_periods`에 영상별 값과 분위수를 싣고, 정식 실행에서는 X6 json 분위수와의 일치를 단언한다(smoke는 `asserted: false`). 그 밖의 summary 키: `n_gt_dropped_by_condition`(s2 자르기 등으로 버린 정답 수), `case_rule`(사례 PNG 순위 규칙), 사례 그림 폴더 `cases/` |

## 5. J1 입력 생성 (R2 안)

- R2 실행이 fold마다 **`j1_boxes_post.csv`(PP 후, 판정 입력)**와 `j1_boxes_pre.csv`(PP 전, 보고)를 쓴다.
  - 행은 754측선 전체다(`keep_both`, [U3] 규칙).
  - 열은 `line_id, fold, class_id, score, i0, i1, k0, k1`(L0 격자), `x0_line_px, y1_px, y2_px`, `pp_action`, `tile_truncated`, `n_tiles_seen`이다.
  - `pipe_hyperbola`만 쓴다. `pipe_band`는 따로 `j1_band_boxes_{pre,post}.csv`로 쓴다(띠 깊이 규칙용이고 J1 판정에는 쓰지 않는다).
  - (C1 2단계 구현 해석) `x0_line_px`는 박스 왼쪽 끝 $x_1$(측선 L1 px)이다. 내용 범위 밖 채움에만 있어 L0 박스가 없는 박스는 J1 파일에 싣지 않는다(J1은 L0 `i0, i1, k0, k1`만 쓰므로 후보를 만들 수 없다). 뺀 수를 fold별로 `summary.json`의 `j1_inputs.folds[k].n_padding_only_dropped`(post·pre, 띠 파일 포함)에 싣는다(구현됨). J1 문서의 "파일의 모든 행" 규칙과 맞는지 gpr-processing-expert가 확인한다.
- 두 파일의 sha와 `PPConfig` 해시를 R2 `summary.json`의 `j1_inputs`에 적는다. 구조는 fold별 사전으로 고정한다.
  ```
  "j1_inputs": {
    "pp_set_id": "S<j>", "pp_config_sha256": "<…>",
    "weights_sha256": {"0": "<…>", …, "4": "<…>"},
    "n_lines": 754,
    "folds": {
      "0": {"post_path": "runs/eval/m33/r2/fold0/j1_boxes_post.csv", "post_sha256": "<…>",
            "pre_path": "…/fold0/j1_boxes_pre.csv", "pre_sha256": "<…>",
            "n_boxes_post": <int>, "n_boxes_pre": <int>},
      "1": {…}, …, "4": {…}
    }
  }
  ```
- **`line_id` ↔ `(activity_id, path)` 대응**: 박스 파일의 `line_id`는 `"<activity_id>/<path>"` 형식이다(예: `01.1/Path5`, `011.17/Profile_6409`). J1-1은 첫 `/`에서 나눠 `activity_id`·`path`를 만든다. 러너는 같은 규칙으로 만든 두 열 `activity_id`, `path`도 함께 싣는다.
- J1 판정 스크립트(X3, gpr-processing + python)는 이 sha를 단언하고, 판정에는 `post`만 쓴다.
- `j1_boxes_*.csv`에는 운용 conf 필터(점수 > 0.25, 병합 경로와 같은 필터)를 통과한 병합 박스만 싣는다. J1 쪽은 다시 거르지 않는다.
- `post` 파일에는 제거된 박스(`pp_action` = `removed_*`)를 싣지 않는다. `clipped_b`는 잘린 좌표로 싣는다.
- 파일 머리 또는 `summary.json`에 `fold`, `pp_set_id`, `PPConfig` 해시, 가중치 sha, 측선 수(754)를 적는다.
- 박스 → 꼭짓점 후보 변환은 J1 쪽(X3) 몫이다. R2는 박스만 낸다.
- cv 러너는 깊이표를 열지 않는다(허용 목록에 없음).

## 6. 실행 순서·시간 (GPU)

| 순서 | 실행 | 추정 |
|---|---|---|
| 1 | R1 모로코(주 + 민감도) | 약 3분(S3 기반 추정 유지). smoke 8장 10.5 s는 모델 적재 고정 비용이 대부분이라 외삽 근거가 부족하다. 단순 비례 상한은 233/8 × 10.5 s × 5 fold ≈ 26분이다 |
| 2 | R2 4TU 754측선 × 5 fold × 2방향 + batch 1 게이트 | 약 15–30분(M3-2 P5a 583 s 기반 추정 유지). smoke 3측선·타일 12·9.0 s는 고정 비용이 대부분이라 외삽 근거가 부족하다. 정식 실행 `timings`로 기록한다 |
| 3 | R3 uninjected(IDS) → plan(4TU + IDS) → C1b(주입 표 sha) → injected | 2차 시간 smoke(3측선, 374 s) 기준 전체 상한 약 6,899 s ≈ **1.9 h**(측선당 실현 16 가정): plan CPU 약 83분, 주입 CPU 약 19분, GPU 추론 약 9분, 미주입 약 4분. 사용자 결정(2026-11-16, R3 결과 전): IDS와 4TU 배경을 모두 C1 범위에서 돈다(4TU 분리 안 함) |
| 4 | J1 판정(CPU, X3) | gpr-processing 추정 |

- 실행 전 `status.py`를 확인한다. 메인 세션 러너(`make_runner.py --gpu-log --gpu-counters`)와 예약 `gpr_detection:m33`을 쓰고, 30분 넘으면 long-running-task-monitor로 감시한다. **R3는 30분을 넘으므로(상한 약 1.9 h) 세션 간 예약(`reserve.py`, TTL은 상한보다 길게)과 주기 감시(`/loop` 점검)를 반드시 건다.** plan은 CPU가 대부분이라 compute-scheduler-expert 배치표로 CPU 묶음을 정한다. 확정 명령·금지 인자·실패 규칙은 §1 표다.
- **운영 주의(C1 실행 전 감사 반영)**: 실행 중에 추적 파일을 바꾸거나(에이전트 메모리 `.claude/agent-memory/**` 포함) 커밋하면, 끝 시점 트리 재측정에서 HEAD·`dirty`·blob이 달라져 그 실행은 `input_error`(종료 2)로 무효가 된다. R1–R3 실행 중에는 저장소의 추적 파일을 건드리는 작업(문서 수정, 메모리 기록, 커밋)을 하지 않는다.
- smoke(측선 3, fold 0)로 시간·VRAM을 먼저 잰다.
- **C1 2단계 블라인드 smoke 공개(2026-11-16, 사용자 사전 허용, 판정 무관. 메인 세션은 종료 코드·status·건수·시간만 보았다)**
  - 1차: R1 `gate_failed`(구현 결함: P5 단언이 PP 전·후 AP50이 둘 다 None인 fold를 실패로 셌다. §4 PP 행대로 고침. PP 작용은 0건, 예측 638행 모두 `none`), R2 명령 오류(`--stage`와 `--smoke` 동시 사용), R3 4TU uninjected 정상(1측선, 타일 165), R3 plan은 R2 출력이 없어 `input_error`.
  - 2차(수정 뒤): R1 `not_judgeable`·8장·10.5 s, R2 `not_judgeable`·4TU 3측선(`--only-lines 06.1/Path8,01.4/Path10,01.1/Path3`)·타일 12·9.0 s, R3 4TU plan(`--injectant band --stage plan --r2-root runs/eval/m33/r2_smoke`) `not_judgeable`·4측선·주입 152·plan 320.8 s. 900 MHz 측선 `pipe/1.15_200115AF/LID10003`의 `preferred_models`는 [M11, M14, M16]이고 4TU 3측선은 비어 있다.
  - smoke 상태 우선순위는 `input_error` > `gate_failed` > `not_judgeable`이다.
  - 3차(보강 6건 뒤, 같은 명령, 2026-11-16): 모두 `not_judgeable`·종료 1. R1 8장 11.0 s(`cases/` 생성), R2 3측선·타일 12·17.0 s(시작 때 r0 11측선 단언으로 길어짐), R3 4TU plan 4측선·주입 152·337.7 s. 메인 세션은 값을 열지 않았다.
  - **smoke 폴더와 부분 판정량 공개(C1 실행 전 감사 반영)**: smoke 출력은 `runs/eval/m33/` 아래 다섯 폴더 `r1_smoke`·`r2_smoke`·`r3_smoke`·`r3_smoke2`·`r3_smoke4tu`(R3 4TU plan smoke)에 있다. 이 가운데에는 판정량의 부분값이 들어 있다: R3 injected `summary.json`의 `p4_adoption_input`(IDS 3측선·fold 0), R1 8장의 AP와 P5 값. 메인 세션은 이 값을 열지 않았고, P4·P5 판정서를 쓸 때까지 열지 않는다. 판정에는 쓰지 않는다.
  - **smoke 파일 해시 목록**: 다섯 폴더의 파일 91개(r1 17, r2 15, r3 24, r3_smoke2 25, r3_smoke4tu 10)의 sha256·크기·수정 시각을 [`m33_smoke_hashes.csv`](m33_smoke_hashes.csv)에 남겼다(2026-11-16, cv, 내용은 열지 않고 바이트 해시만 잼). 지금 판은 4차 뒤 판(`674afa93…`)이다(아래 4차 참고). 판정서 단계에서 smoke 폴더가 바뀌지 않았는지 이 목록으로 대조한다.
  - 4차(C1 실행 전 감사 보강 뒤, 2026-11-16, 사용자 사전 허용, 판정 무관, 메인은 종료 코드·status·건수만 봄): R2 `--lines … --smoke --overwrite --only-lines 06.1/Path8,01.4/Path10,01.1/Path3` → `not_judgeable`, 3측선, 타일 12, 17.2 s. R3 IDS 전체 단계 `--injectant band --smoke --overwrite --out-root runs/eval/m33/r3_smoke2` → 세 단계 모두 `not_judgeable`, 타일 2,548, 주입 152, 390.0 s. R3 4TU uninjected(900 MHz 1측선) → `not_judgeable`, 타일 165, 18.4 s. R3 4TU plan(4측선) → `not_judgeable`, 주입 152, 338.8 s. R1은 코드(`m33_morocco_eval.py` blob)가 바뀌지 않아 다시 돌리지 않았다. 4차 smoke는 커밋 전 작업 트리 코드로 돌렸다(C1 실행 전 감사 재확인 기록 2). 커밋된 코드(f4ad45e)의 근거는 verif 재확인의 R1·R2·R3·P4 시험 99 passed다.
    - 4차는 `--overwrite`로 `r2_smoke`·`r3_smoke2`·`r3_smoke4tu`를 다시 썼다. 그래서 1·2차 `r3_smoke2`의 원래 내용과 3차 R2 출력은 덮어써져 사라졌다.
    - 사라진 출력의 이전 해시는 첫 해시 목록 판(파일 sha256 `d84cf08a…`, 2026-11-16)에 있었다. **그 판은 커밋되지 않았고 지금 판으로 덮어써졌으므로, 사라진 출력의 파일별 해시는 남아 있지 않다.** 남은 것은 그 판 파일 자체의 sha256(`d84cf08a5dfcde92975f79262f0e35dd02c8000a509dbdd1e0bcee8304233ff1`)과 폴더별 파일 수(r1 17, r2 15, r3 24, r3_smoke2 25, r3_smoke4tu 10)뿐이다.
    - 해시 목록은 4차 뒤 같은 스크립트로 다시 만들었다(91개, 폴더별 파일 수는 같음, 파일 sha256 `674afa93…`). 판정서 단계 대조는 이 판으로 한다. 이 판은 다음 커밋에 넣는다.
  - **smoke 출력 열람 기록(2026-11-16)**: gpr-processing-expert가 J1 v0.3.3 단언 3의 키 구조를 확인하려고 `runs/eval/m33/r2_smoke/summary.json`을 열었다. 본 키는 `run_identity.weights_sha256`(빈 사전, `m31_d1_common.new_run_identity`가 만드는 키)과 `j1_inputs.weights_sha256["0"]`(가중치 sha)다. 이 파일에는 3측선·fold 0의 P2·P3 보고값(K 포함률, M4)과 `j1_inputs`의 박스 수가 들어 있다(C1 실행 전 감사 재확인 기록 3). P4·P5 판정값은 없다. 정식 J1 입력은 정식 R2 출력이므로 판정에는 영향이 없다. 메인 세션은 이 파일을 열지 않았다.
  - R3 4TU plan 4측선 320.8 s는 IDS 3측선 plan 269 s와 비슷하다. plan 시간은 고정 비용이 크다는 뜻으로 읽고, §6 R3 상한(약 1.9 h)은 유지한다. 4TU 177측선 plan의 측선당 시간은 정식 실행 `timings`로 확인한다.

## 7. 시험 (python-expert)

1. PP 전 행은 M3-2 S5 r0 출력과 같다. 같은 측선 3개 fixture로 확인한다.
2. `keep_one` 집계: 720 측선 수와 중복 쌍 처리가 [U3] 규칙과 같다.
3. 띠 가로 구간 일치: 합집합 덮임, 시간 겹침 경계, 박스 없음이면 0. 정답 구간 밖 부분은 재현율에서 빠지고 유령 길이로 세어진다. 미주입 짝이 있는 박스는 유령에서 빠진다.
3a. `lw_stratum` 경계값: 0.33, 0.5, 0.67은 위쪽 층에 넣는다(구간이 [a, b) 형태). (v0.3.4) 경계 리터럴이 0.33(1/3 아님)임을 단언하고, $(L, N)$ = (100, 301) → `033_05`, (64, 199) → `lt033`, (40, 89) → `033_05`를 단언한다($W_\text{eff} = \min(301, N)$).
4. 모로코 하위셋 분리와 공동이 전체 지표에서 빠지는지.
5. 블라인드: stdout 마지막 줄에 지표 키가 없다.
6. 입력 sha 하나를 바꾸면 종료 2.
7. 모로코 출력 폴더가 있으면 거부한다.
8. (v0.3.3) P5 구조상 0: `pipe_band` 박스만 다른 PP 전·후 예측에서 `pipe_hyperbola` AP50이 같고, 일부러 쌍곡선 박스를 바꾼 fixture에서는 종료 1이다.
9. (v0.3.3) PP 후 유령 짝: 미주입 짝 박스를 PP가 지운 fixture에서 PP 후 유령 수가 1 늘고 `ghost_pp_artifacts.csv`에 실린다. 짝이 PP 후에도 남으면 늘지 않는다.
10. (v0.3.3) P4 작은 층: 위치 수 (30, 30) → `two_strata`, (29, 40)·(0, 12) → `pooled`, (0, 0) → `not_evaluable`. (v0.3.5) `not_evaluable`은 노출·작용 값과 무관하게 먼저 정해진다.
11. (v0.3.5) 노출 정의가 선택 집합을 따른다: 정답 아래 끝만 PP-c 창에 드는 fixture에서 `pp_set_id` = `S3`이면 노출 0·`pp_c_window_positions` 1이고, `S4`이면 노출 1이다. 중심이 $y_\text{bot}$ 아래(a), $y_\text{bot}$ 걸침(b), $y_z + 0.5P_\text{line}$ 위(d)인 정답은 S3에서 각각 노출 1이고 `pp_exposed_by_rule`이 맞다. `pp_set_id`·`PPConfig` 해시가 C0b 값과 다르면 종료 2다.
12. (v0.3.6) `p4_status`와 단언(모두 노출 0인 fixture, 마지막 둘 제외)
    - 아무 작용 없음 → `no_effect`, `p4_structural_zero` true, `recall_delta` 정확히 0.
    - G 박스의 짝인 미주입 박스 하나만 `removed_d` → `no_effect`(`pp_acted_ghost_set` 1, `pp_acted_boxes_all` 1, `pp_acted_boxes` 0). 늘어난 유령은 `ghost_pp_artifacts.csv`에 실리고 종료 0이다.
    - 짝이 아닌 미주입 박스 하나만 `removed_d` → `no_effect`(`pp_acted_boxes_all` 1, `pp_acted_ghost_set` 0, `pp_acted_boxes` 0).
    - R 안에 `clipped_b`만 있음 → `no_effect`이고 자르기 단언 통과(`pp_acted_clip_recall_set` 1, 덮임 불변).
    - R 안에 `removed_d` 1개 → `acted_unexposed`, `p4_pp_acted.csv`에 1행.
    - 정답 덮인 길이가 줄어드는 자르기(정답 아래 끝을 $y_\text{bot}$ 아래로 둔 인공 fixture) → 종료 1.
    - 짝과 후보 모두 `pp_action` = `none`인데 유령이 늘어난 fixture → 종료 1.
    - `no_effect` 조건인데 PP 후 R 박스를 일부러 바꾼 fixture → 종료 1.
    - 노출 > 0 → `evaluated`.
    - 경계: 두 층 위치 50/50에서 길이가 모두 같은 fixture에서 한 위치의 실현-fold 40개를 모두 잃음(덮임 1 → 0) → `recall_delta` = −1.0 %p(−1.25/50 × 40)이고 통과한다. 41개를 잃으면 실패한다.

## 변경 이력

| 날짜 | 판 | 내용 |
|---|---|---|
| 2026-11-15 | v0.1 | 최초 작성(X4): 재사용 판단(S5 확장: R2·R3, S3 경로 새 러너: R1), 공통 sha·블라인드·PP 전·후, R2 754 추론/720 집계, R3 의사 띠(4TU 보고, IDS P4), R1 민감도·1회 실행·P5, J1 박스 내보내기 |
| 2026-11-15 | v0.2 | [EMR] 부록 D(X5)에 맞춰 R3를 고쳤다. 주입체는 S4 새 띠 모드로 바꿨다(`make_pseudo_band` 아님, `_bg` 미가산). 배경별 모델, (c′)에 미주입 띠 박스 포함, 간격 $W_\text{eff}$ 식, 길이 표본, $L/W_\text{eff}$ 층과 층 최소 수, 정답 표 열, 유령 띠 오검출 정의, 생존 비 [EMR] 식, P4 키(IDS·짧은 두 층·`ghost_delta`)를 정했다. 시험 3·3a를 추가했다 |
| 2026-11-15 | v0.3 | X6·사용자 결정 반영: R2 01.1형 폭을 `width_vis_P`로, R1 (s3) 행 범위를 0–27로 고정, (s5) 주기 맞춤 축소 추가(보고 전용), J1 박스 파일 규칙(conf 필터 통과분만, post에는 제거 박스 없음, 머리 정보) |
| 2026-11-15 | v0.3.1 | 무결성 점검 반영. 상위 판 번호를 계획 v0.2.4·PP v0.2로 고치고 봉인 순서(C1b·C2)를 적었다. R2 `summary.json`에 PP 처리 수와 PP-a 진단(`pp_a_top_in_content`)을 넣었다. `j1_inputs` fold별 구조와 `line_id` ↔ `(activity_id, path)` 대응을 정했다. IDS 비-400 MHz 7측선은 [EMR] D-6에 따라 `ids_freq_stratum`(주 400 MHz 53 / 보고 3 / 2 GHz 주입 제외 4)으로 나누고 P4를 `main_400`으로 한정했다. (s5)는 $P_A$ 기준이다 |
| 2026-11-15 | v0.3.2 | 띠 모드 구현 해석 확인. (c′) 띠는 2D로 본다(넓힌 미주입 박스 ±1 P × ±1 $P_\text{line}$가 띠 정답 박스와 겹칠 때만 막음, 가로 전용 부분집합은 `in_horizontal_box_zone`으로 보고). (d)는 구현 해석을 받아들였다(띠 구간 양 끝 기준, K −20 dB 반폭, 자동 후보 −6 dB 반폭 + 1 P). `model_group`은 러너가 붙인다 |
| 2026-11-15 | v0.3.3 | stats-design 검토에 따른 사용자 결정(C1 전, 측정 전). R1: P5는 구조상 0이라 러너가 `p5_adoption_input == 0`을 단언(어기면 종료 1, `p5_structural_zero`). R3: PP 후 유령 판정은 같은 PP를 적용한 미주입 박스와 짝짓고, PP 후 − PP 전 > 0이면 `ghost_pp_artifacts.csv` 사례 목록. plan 보고에 P4 두 층의 PP 노출 위치 수(검출 무관, C1b 고정, 0이면 `no_effect`). P4 작은 층 처리(한 층이라도 위치 < 30이면 합친 한 층·MDE, 둘 다 0이면 `not_evaluable`). IDS 400 MHz 층 구성과 fold SD 성질 공개. 시험 8–10 추가. 상위 판 번호를 계획 v0.2.5·PP v0.2.3으로 고쳤다 |
| 2026-11-15 | v0.3.4 | [EMR] D-8(v1.9) 반영(C1 전, 측정 전). §3-1 4: IDS 400 MHz 층 구성 공개 문장을 정정했다(53측선 89–6,665트레이스, 16측선 $W_\text{eff}=N$, 층은 L과 N으로 정해짐, 길이별 층 측선 수). L = 150은 유지(cv 결정, 약한 띠·갈림점 공개, L 136 대안 미채택). PP 노출 기하(사실상 PP-c 창뿐, 약 8–11 %, 노출 0 확률)를 공개했다. 띠 배치 순서를 a0dc987 구현(바닥 우선) 그대로로 명시하고 C3-6 소비를 공개했다. §3-2에 `recall_delta_exposed`(보고 전용)를 넣었다. 시험 3a에 경계 리터럴 0.33과 (L, N) 단언 3건을 넣었다. P4 감도는 사용자 결정 (A) 2026-11-15(공개만, `recall_delta_exposed` 보고 전용, 판정 규칙 불변). C0a 전 무결성 보강: [EMR] 참조를 v1.9.1로 맞췄다 |
| 2026-11-15 | v0.3.5 | 사용자 결정(2026-11-15, **D0 결과(C0b 0f11c34, S3 = {a, b, d})를 안 뒤·C1 전·R1–R3 측정 전**, [`m33_c0b_d0_judgment.md`](m33_c0b_d0_judgment.md) §9 2). §3-1 4 PP 노출 위치를 C0b 선택 집합의 규칙(S3: a·b·d)이 정답 띠 박스에 작용할 수 있는 위치로 한정했다(옛 정의: 집합과 무관한 PP-c 창 ∪ PP-d ∪ PP-a·b). PP-c 창 위치 수는 보고 전용 `pp_c_window_positions`로 남겼다. 노출 0일 때의 처리를 정했다: 검출 쪽 작용 수 `pp_acted_boxes`를 더해 `p4_status`를 `not_evaluable` → `no_effect`(노출 0·작용 0, 차 구조상 0 단언, 어기면 종료 1, P4는 채택 근거에서 빠짐) → `acted_unexposed`(노출 0·작용 > 0, 판정 규칙 적용·작용 목록) → `evaluated` 순서로 정한다. §3-2 키 추가(`pp_set_id`·`pp_config_sha256`·`pp_exposed_by_rule`·`pp_c_window_positions`·`pp_acted_boxes`·`p4_structural_zero`·`p4_pp_acted.csv`). 시험 10 보강, 11·12 추가. 문턱·범위·층·가중 규칙은 바꾸지 않았다. 상위 판 번호를 [PLAN] v0.2.7로 고쳤다 |
| 2026-11-15 | v0.3.6 | stats-design-expert 검토(v0.3.5 개정안)와 사용자 결정 3건(2026-11-15, 모두 권고안, **D0 결과(C0b 0f11c34, S3)를 안 뒤·C1 전·R1–R3 측정 전**)과 후속 권고 4건. (1) 작용 집합을 R(재현율 박스)·G(유령 후보)·G′(G의 미주입 짝)로 나누고, 상태 결정용 `pp_acted_boxes`를 R 안 제거(`removed_a`·`removed_b`·`removed_d`)로 한정했다. 빈 통과를 막으려는 취지다. R 안 `clipped_b`는 `pp_acted_clip_recall_set`로 따로 내고 덮임 불변을 단언한다(어기면 종료 1). G·G′는 `pp_acted_ghost_set`, 전체는 `pp_acted_boxes_all`(둘 다 보고 전용)이다. `pp_acted_by_rule`·`pp_acted_lines`를 추가했다. 층은 정답 띠 단위이고 G′는 짝 G의 층을 따른다(짝 단위·중복 뺀 수). (2) `ghost_delta`는 보고 전용이다. 새 유령은 모두 `ghost_pp_artifacts.csv`(후보·짝 `pp_action`)에 싣고, 설명되지 않는 증가는 종료 1이다. P4 판정은 재현율 조건 하나다. (3) 기준은 `recall_delta` ≥ −1.0 %p − 1e-9(동률 통과)다. `no_effect` 단언 문구를 "R에 제거 작용이 없으므로 `recall_delta`는 구조상 0(단언), `ghost_delta`는 보고"로 고쳤다. S3 예상 경로(노출 구조상 0, PP-b 자르기 재현율 불변, 손실은 a·d 제거로만)를 공개했고, PP-b 전제를 확인했다(같은 기록 끝: L0 $n_s$ ↔ L1 $y_\text{bot}=n_s/\sigma_y$, sample_offset 0, 여유 1 $P_\ell$). 개수 모형(−1.25/n %p, 0.8n)과 작은 층 참고 문장(−2 %p 통과 0.1–0.6, MDE80 1.5–7.25 %p [추정], `mde_pp`도 같은 모형)을 고쳤다. 시험 12를 다시 썼다(짝/비짝 미주입 제거, R `clipped_b`, R `removed_d`, 덮임 감소 자르기, 설명 안 되는 유령, −1.0 동률 경계). 상위 판 번호를 [PLAN] v0.2.8로 고쳤다. (같은 판, 이어서) python-expert의 D0 실행기·`m32_recipe_eval.py` 결함 수정(2026-11-15, 미커밋)을 공통 요구로 옮겼다: §1에 끝 시점 트리 재측정(다르면 종료 2), 전체 정밀도(`repr(float)`), 판정 가능성(기대 fold 0–4, `not_judgeable` 종료 1, `judgment` 블록), 텍스트 입력 LF sha 기준(C1 sha 표 포함) 행을 넣고, 남은 위험 4건(`m32_recipe_eval.py` 4자리 반올림·fold 완전성·끝 시점 재측정 없음, `_object_rows` 반올림, LF sha 표)을 C1 구현 항목으로 적었다. 규칙·문턱은 바뀌지 않았다. (이어서) 사용자 결정(2026-11-15, C1 전 D0′)과 integrity 권고: §1에 D0 실행기 blob 두 개(C0a·개정)를 C1 표에 나란히 적는 행을 넣고, "깨끗한 트리" 행을 "`--allow-dirty`는 smoke·진단 전용(판정 불가, `not_judgeable`)"으로 고쳤다. (이어서) D0 실행기 blob 행에 D0′ 봉인 PP 단언(`assert_sealed_pp`)과 `src/` 변경은 D0′ 뒤라는 순서 제약을 적었다. (이어서, 2026-11-16, C1 1단계 구현 검토) python-expert의 반영 사항: 공통 모듈 `m33_inputs`·`m33_r3_p4`, 판정 불가 사유 값(`line_limit`·`partial_recipes`·`provisional` 등)과 fold 없는 단계의 `folds_run` null, C1 sha 표에 새 코드 blob과 D0′ 출처 blob(882e881 값) 규칙, R2 레시피 r0 전환은 R2 단계로, M3-2 precheck·smoke가 `not_judgeable`이 된 영향 수용. cv 해석(결과 전): 유령은 [i0 − T, i1 + T] 기준·실현별 서로 다른 박스 수·길이는 합집합, P4 범위는 C3-6 주 칸, `pp_acted_boxes_all`은 total·by_rule, `p4_pp_acted.csv`는 항상 씀, 일치·IoU는 L0 닫힌 격자·노출은 L1 px, `mde_pp`는 러너 null·stats-design이 C1b 뒤 계산. 판정 규칙(문턱·범위·가중)은 바꾸지 않았다. (이어서) **사용자 결정 (A)(2026-11-16, C1 봉인 전, R3 결과 없음): 판정값 정의 확정 — 셀 안 길이 가중 유지(코드 변경 없음).** §3-2 합친 층 문구를 "층 가중 없이 두 층 단위를 합침, 셀 안 길이 가중"으로 고쳤다(이전 "위치 같은 가중"은 구현·§3-1 6과 어긋났음). 개수 모형에 "길이가 같을 때의 근사, 정답 하나 손실 = −1.25·(L/L̄)/n %p"를 적었다. `mde_pp`는 stats-design-expert가 C1b plan 뒤 길이 가중을 반영한 개수 모형 시뮬레이션으로 계산한다. 시험 12 경계 사례는 길이가 같은 fixture로 한정했다. (이어서, 2026-11-16, R3 띠 러너 배선 검토, R3 결과 없음) 새 모듈 `m33_r3_band.py`, 실행 인터페이스(`--injectant band`·`--backgrounds`·`--r2-root`·`--c1b-record`, C1b 기록 `m33_r3_c1b.json`·초안 `plan/c1b_record_draft.json`, 출력 파일, status `p4_defect`), 실데이터 smoke 공개(3측선·fold 0·544 s, 값 미열람)와 §6 R3 시간 재추정 필요, §3-1 4 배치 구현 해석 9건(상한·그룹 간격, $L+2T \le N$, $k$ 1.3만, 900 MHz 500 MHz 모델 우선 구현 필요, 추론 범위, 유령 짝은 미주입 박스를 추론 범위로 잘라 비교, 바닥 칸 보고 표, P4 유령 중복 제거, px 층 경계, PP 내용 범위·영점 출처와 `n_zero_changed`, 채움 전용 박스 처리와 `pp_acted_padding_only`), C1 sha 표에 `m33_r3_band.py`·`inject.py`, R2·R1으로 넘길 일을 적었다. 판정 규칙(문턱·범위·가중)은 바꾸지 않았다. (이어서, 2026-11-16, R3 보강 8건 반영과 2차 시간 smoke, R3 결과 없음) **추론 범위를 측선 전체로 정정했다**(r0 측선 전체 통계 3개: 회색 클립 99번째 백분위, 주기 rfft 평균, AGC 바닥값. 띠 주변 타일 범위의 전제가 틀림, 시험으로 확인. 배경 값 고정 대안은 src·판정 경로 변경이 필요해 채택 안 함). 900 MHz `preferred_models`, `clip_pair_pool`, summary `bottom`·`n_realizations`·`n_period_changed`·`n_zero_changed`·`pp_acted_padding_only`, P4 유령 두 층 합 중복 제거, 간격 식 시험이 구현됐다. 2차 smoke(374 s, uninjected 7.7·plan 269·injected 69 s)를 공개하고 §6 R3 시간을 상한 약 1.9 h로 고쳤다. 사용자 결정(2026-11-16, R3 결과 전): IDS·4TU 배경 모두 C1 범위(4TU 분리 안 함), 실행 때 세션 간 예약과 주기 감시. (이어서, 2026-11-16, C1 2단계 R1·R2 구현 검토, 결과 없음) 설정 해시 기대값은 C1 커밋 코드에서 계산해 상수로 고정하고, r0 11측선 기대값은 M3-2 S6 (a) 시험 고정값을 쓴다. 러너별 코드 blob 목록과 C1 sha 표 추가 입력(R2 목록·K SNR 표·모로코 매니페스트·목록의 LF/원시 sha)을 적었다. §2-1 R2 모듈·목록 파일·`--only-lines`·출력 경로, §2-2 R2 요약 해석(01.1형 폭은 한 fold라도 놓친 꼭짓점 + `n_folds_missed`로 넓힘), §4 R1 P5 단언 None 규칙·사례 PNG 순위·구현 해석 6건(부트스트랩은 5 fold 평균으로 고침, s5 분위수는 X6와 나란히 보고), §5 J1 `x0_line_px`·채움 전용 박스 제외와 `n_padding_only_dropped`, §6 R1·R2 시간과 C1 2단계 smoke 공개를 넣었다. 판정 규칙은 바꾸지 않았다. (이어서) §1에 "C1 sha 표의 값은 C1 봉인 기록 `m33_c1_seal.md`에 둔다" 행을 넣었다. (이어서, 보강 6건 구현 뒤) §1 설정·r0 정체 행에 구현(`R_SETTINGS_SHA256` `b9c5c905…`, `R0_IDENTITY_LINES`, `r_checks`·`settings_assertion`), §4 R1 `bootstrap_fold_mean`·`s5_periods`(정식 실행에서 X6 일치 단언)·`n_gt_dropped_by_condition`·`case_rule`·`cases/`, §2-2 `n_folds_missed`, §5 `n_padding_only_dropped`(post·pre·띠), §6 3차 smoke 공개를 넣었다. (이어서, integrity 점검 반영) §1 입력 sha 단언 행에 `radargrams_index.csv`(읽지 않음)·중복 목록(간접 고정)·X6 json(값 단언, sha 기록)을 밝혔고, §2-2의 옛 "5 fold 모두 놓친" 문장을 지웠다. (이어서, C1 실행 전 감사(차단 0, 실행 전 수정 6, 기록 13) 반영, 사용자 결정 2026-11-16, R1–R3 결과 전) 순서를 R1 → R2 → R3 uninjected(IDS) → plan → C1b → injected → C2 → J1로 통일했다. §1 입력 sha 행의 S4 blob을 R3 시작 때 상수로 단언(종료 2)하는 문구로 바꿨다. "깨끗한 트리" 행을 시작 시점 dirty 판정과 `--allow-dirty`의 뜻으로 고쳤다. §1에 확정 명령·금지 인자·실패 규칙(예비 폴더 `r1_retry`·`r2_retry`·`r3_retry`, 코드 수정 시 감사 재실행, `--resume`은 같은 HEAD·blob에서만) 행을 넣었다. §4 P5 None 뜻(정답 또는 예측 없음)과 `p5_folds_without_ap50` 공개 문구, R1 `CROP_CONTENT` $y_\text{bot}$ 256을 s1·s5에도 쓴다는 해석, §3-1 4 G′ 짝 풀의 conf > 0.25 필터를 적었다. §6에 운영 주의(실행 중 추적 파일 변경·커밋은 끝 트리 검사로 실행 무효), smoke 폴더 다섯 개와 부분 판정량 공개, smoke 파일 해시 목록 `m33_smoke_hashes.csv`를 넣었다. (이어서, stats-design 검토에 따른 사용자 결정 2026-11-16, R1–R3 결과 전, 판정 규칙·결정 확률 불변) `pp_acted_lines` 정의(R 안 제거가 있는 측선, 두 층 합은 중복 없이)를 고치고 `n_r_boxes`·`n_r_lines`·`recall_pre`·`action_line_rate_upper95`(분모 `n_r_lines`, n = 0이면 null)를 출력에 넣었다. `no_effect` 행에 "판정서에는 0/`n_r_lines` 상한, 분모는 53이 아님"을 적었다. (이어서, C1 실행 전 감사 보강 구현과 4차 smoke 뒤) 구현 키에 맞췄다: `r_checks.inject_blob`, resume 표지 `head`·`code_blobs`·`src_tree_id`(구현됨), 판정 불가 사유 `no_flip`, injected의 `plan_report.json` 상류 확인과 `inputs_sha256.plan_report`, P4 `n_truth_folds_by_cell`(셀 결측 보고)·`pp_acted_lines_by_stratum`·`recall_pre`(0–1 비율)·`action_line_rate_note`, 0/53 → 0.0672. §6에 4차 smoke와 `--overwrite`로 사라진 출력, 커밋되지 않은 첫 해시 목록 판, 다시 만든 해시 목록(`674afa93…`)을 공개했다. (이어서, integrity 권고) `action_line_rate_note` 세 경우 문구, `n_truth_folds_by_cell` 구조(`by_stratum_cell_fold`·`expected_cells`·`missing_cells_by_stratum`·`cells_complete`), 금지 인자 `--f2-record`·`--provisional`. (이어서, C1 봉인 2026-11-16, verif 재확인 기록 4건 반영) 상태 줄을 C1 봉인으로 바꿨다. 실패 규칙에 예비 폴더의 하류 경로(`--r2-root` 허용 예외), R3는 세 단계와 C1b를 모두 `r3_retry`에서 다시 함, 빈 폴더·표지만 있어도 부분 출력이라는 규칙을 넣었다. smoke 열람 공개 문구(P2·P3 보고값과 `j1_inputs` 박스 수)와 4차 smoke의 코드 시점을 고쳤다. `C1_INJECT_BLOB` 표기를 "C1 코드 커밋 f4ad45e의 blob(826aa41과 같음)"으로 고쳤다 |
