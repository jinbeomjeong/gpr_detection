# M3-3 마감 전 ③ E3x 확장 실험 판정서

- 작성: gpr-processing-expert, 2026-11-26. 판정 결과(`runs/eval/m33/e3x_retry/`)를 처음 연 사람이 작성자다. 메인 세션은 종료 코드와 마지막 줄만 보았다(§5-3).
- 판정 규칙: `docs/processing/m33_e3x_plan.md` v0.5. 커밋 `50a28a3434dfa9387e7931cdc29c4bb21e34b2a7`. sha256 `89fff477c1233123cfe2ffe19d5102a33cec96bf468d3b8c4705497887e5cb50`(작성 시 파일에서 직접 계산. 408df3a·bc57ef1에서 바뀌지 않음. `.sha256` 파일은 없고, 커밋과 이 64자리 값으로 고정한다. verif 재확인 기록 10).
- 봉인 기록: `analysis/detection/m33_e3x_seal2_r1.json`. sha256(LF = raw) `81e57d3e4c0c36a4dda991bb33230ae7388da86511363c5aa03a2fef470625fe`, blob `151d995b17d9ebc01fbf6b7be9b3a5de2a072f34`, 커밋 50a28a3.
- 판정 입력: `runs/eval/m33/e3x_retry/injected/`, `runs/eval/m33/e3x_retry/uninjected001/`(R-d 기준값만 씀). 판정 도구 출력: `runs/eval/m33/e3x_retry/judgment/`. 재시도 차수는 둘 다 `e3x_retry`(k = 1)다.
- 출처 약어(수치마다 붙임)
  - [J] `runs/eval/m33/e3x_retry/judgment/judgment.json`
  - [SI] `runs/eval/m33/e3x_retry/injected/summary.json`
  - [SU] `runs/eval/m33/e3x_retry/uninjected001/summary.json`
  - [PR] `runs/eval/m33/e3x/plan/plan_report.json`(봉인 2 plan 보고), [BA] `runs/eval/m33/e3x/plan/band_arms.csv`
  - [PC] `runs/eval/m33/e3x_precheck_r1/precheck.json`, [CA] `runs/eval/m33/e3x_precheck_r1/cpu_assertions.json`
  - [RL] 재시도 러너 `C:\Users\jinbeom\.claude\runners\gpr_detection-m33_e3x_retry\`(`logs_P\`, `runner_summary.json`). 저장소 사본 `docs/processing/evidence/m33_e3x_retry_runner/`(SHA256SUMS 22항목, 원본은 그대로 둠)
  - [RC] `analysis/processing/m33_e3x_judgment_recheck.json`, [RR] `analysis/processing/m33_e3x_judgment_recheck_report.json`(이 판정서의 독립 재집계, §11)
  - [MDE] `docs/processing/m33_e3x_mde_eval.md`

## 0. 판정 요약

수준은 양측 정확히 $1-0.05/6$이다([J] `hypotheses.*.ci_level` = 0.9916666666666667). 판정 구간은 Satterthwaite t, HH2만 기본(역백분위) 부트스트랩 구간이다. δ = 0.05, 바닥 기준은 $F \ge 0.10$이다.

| 가설 | 추정량 | 판정 구간 | 결정 라벨 | 판정서 문구 | 출처 |
|---|---|---|---|---|---|
| HH1 개구 끝 처리 | $\hat\Delta$ = 0.009375 | [−0.001466, 0.020216] | `negligible` | **기각(실질 효과 없음)**. 음성 대조는 통과(표지 없음) | [J] `hypotheses.HH1.{estimate,ci_low,ci_high,decision,wording}` |
| HH1 음성 대조(점검) | $\hat\Delta_\text{neg}$ = −0.005357 | [−0.016141, 0.005427] | `pass` | – | [J] `hypotheses.HH1.manipulation_checks.negative_control` |
| HH2 모델 몫 > 0.5 | $\hat\eta^2_p$ = 0.65900 | [0.49798, 0.73910](기본 구간) | `inconclusive` | **추정 유지(구간이 넓음)** | [J] `hypotheses.HH2` |
| HH3 30 dB 포화 | $\hat\Delta$ = 0.09123 | [0.03654, 0.14593] | `confirmed` | **확인** | [J] `hypotheses.HH3` |
| HB1 얕은 행 | $\hat\Delta$ = 0.23733 | [0.14934, 0.32532] | `confirmed` | **확인** | [J] `hypotheses.HB1` |
| HB1b 깊은 행 | $\hat\Delta$ = 0.09759 | [−0.01090, 0.20607] | `inconclusive` | **추정 유지(구간이 넓음)** | [J] `hypotheses.HB1b` |
| HB2 길이 | $\hat\Delta$ = 0.23592 | [0.16253, 0.30930] | `confirmed` | **확인** | [J] `hypotheses.HB2` |

- 판정 가능성: injected `judgment.judgeable` true, blockers 없음, fold 0–4 모두 돎([SI] `judgment`). [J] `judgeable` true, `status` ok.
- 독립 재집계([RC], §11): 7개 추정량, 바닥 양 $F$, 위치·군집 수, $G_g$·$G_{\text{eff},g}$, $v_g$, $\nu$, 판정 구간 끝, $M^\ast$·$M^\ast_e$를 기록 csv에서 다시 셌다. 모두 [J]와 같다(구간 끝 차 ≤ 3×10⁻¹⁷, 결정 라벨 7/7 일치).

## 1. 실행 정체성 (§6 1)

**공통 정체성**(재시도 세 단계 같음. [SI]·[SU] `run_identity`, [J] `identity`)

| 항목 | 값 | 출처 |
|---|---|---|
| HEAD(재시도) | `bc57ef1dd50a9984956b7efca6df6fd56a323002`, dirty false, 시작 = 끝 | [SI] `run_identity.git.head`, `run_identity.tree_end.head`. [SU] 같음. [J] `identity.git.head` |
| src tree id | `d0e54933f3a0c87cf2b8d43d255e97bd6bb97122`(시작·끝) | [SI] `run_identity.src_tree_id`, `tree_end.src_tree_id` |
| 봉인 기록 sha(LF) | `81e57d3e4c0c36a4dda991bb33230ae7388da86511363c5aa03a2fef470625fe`. injected·uninjected001·판정 도구가 같은 값 | [SI] `run_identity.seal2.seal_record_sha256_lf`, [SU] 같은 키, [J] `identity.seal_record_sha256_lf` |
| 코드 blob(18개, 봉인 2 r1 기록과 같음) | `analysis/detection/m33_e3x.py` `d13d6e770fffd359b4f3dd7f38f699d1ecdd8460`, `m33_e3x_place.py` `0f1491a1e5cc6369f90dac3cf02f376106090ea3`, `m33_e3x_pool.py` `a3452bd59c7886a0944a1c217714b5cf1b2c3e93`, `m33_e3x_judge.py` `d9ed14b675fcdce8d467a3b6643b4545ec12bcb5`, `m31_d1_common.py` `0ba0675c8fc6bd632ed2b1e6794702b60f9b5d72`, `m32_aihub_perturb.py` `320ce955a7acbcd8956cde7901234cf61e19ba45`, `m32_pipe_responses.py` `5d065b3a807b41df04ff81098df86ba9ec747f58`, `m32_recipe_eval.py` `ded84a93b95003a5920bf41f73ad1d0436b9ef14`, `m32_recipe_inputs.py` `7f4815c5582086408a56a2002bf51d6002826a62`, `m32_recipe_match.py` `7e17f060947073dbfcf4e3701a6e0505bffa9547`, `m33_change_defs.py` `7a411d644f28275daaa4076ae9bd5456fd15e945`, `m33_inputs.py` `ababaa50bc0371f90c55b18b0f0e7192fc267a74`, `m33_r3_band.py` `b499a456fe423b7d8a72fa516a0f2123ae3bd7f4`, `m33_r3_p4.py` `888f3dc9ac209400c71bbb29cf337cc4581f8f3b`, `src/gpr_detection/preprocess/to_image.py` `233e47ef22f825dc24449b7e5567fe8a47719d9a`, `src/gpr_detection/eval/inject.py` `2d88ff0a25c5cbc1bb7ca8698ba422c633f5b531`, `src/gpr_detection/pipeline.py` `dd5e7bcba4397efd732e7bf3ca1c5c92fb2404a1`, `src/gpr_detection/detect/yolo.py` `505ff04d228c9e41272144fc263b7912b7568065` | [SI] `run_identity.code_blobs`(시작), `tree_end.code_blobs`(끝), [SR] `code_blobs` |
| 가중치 sha256(fold 0–4, `last.pt`) | 0 `7ead71c7fa62d85edb11f5614e78b11619062b455dca62f5cc66c93b2dcf8d92`, 1 `f4b7e0443799d66983adb21356a390bf5cf68ec98964e9425f0981cd3b079b15`, 2 `32c0e854ed7e6ac6e4847265f956f038030bb968f4d01ab8a550a40e2d98a99b`, 3 `0238d4d62722ff7318e8c8fa8c283471b008f01905e95d326510547882b74de8`, 4 `98922d2e79b547fd7cf4714b400e693b131d9082c58d293b9e0f5163e039c6b2` | [SI] `run_identity.inputs_sha256."0"`–`"4"`. `assert_weights`가 기준값과 대조한 결과다(`m32_recipe_eval.py:534`). `run_identity.weights_sha256`은 빈 dict다(이 러너는 가중치 sha를 `inputs_sha256`에 넣음) |
| 입력 sha256 | r2_lines `57de56c64d488eab0bc2316f816177317958a40060c47227787523d0f97c4b8d`, stage_a_4tu `2d590cdab6055df22d7294ddaa11676fee39774e9b95a705b4c9884e3787cdf2`, stage_a_ids `a52e467dfed80e64885b3ced9e5fb5e9e9617305b39e7d7e1d44e4d1f5bedfb3`, fourtu_lines `475f0a71fdd79fcca3fde035bce0876778f4771b61e4dc50195868e085104cc6`, pool_csv `06f0078374704d8ce95e3bdc499caa3d967feeb25eecf8d24a623ea544007c11`, apex_candidates `b348a2ffe9fe11f139be4fa8512c68a6eba449c37c99ba284f2154144968d43f`, k_apexes `0fec12dba4efa504a69a8319e52eb854dac6359296beb391f6b83b177d49db61`, hyperbola_positions.csv `ad4ac3b544b2a0dc3a9aaa71f85d99ee106564e67c403fa2e3a8f9e5c2b1eaff`, band_positions.csv `6817075541fda4fcc5dccd45168b21b7d3ffd6e43aa96629a43b55a1c8af415e`, band_arms.csv `66aad2fba14d6161286b8eefaf7088c76c2e1c172a153f83df00db8cec8db609`, lines.csv `a079a5cc2488878e1bbf80a76651ea15ac7fb4269e3ad6f4a6d90f530b69a75b`, exclusions.csv `4369750153e2b3059884be74284c614a1007059a9d07fdfcad6914e31407a875` | [SI] `run_identity.inputs_sha256`, `run_identity.seal2.plan_files_sha256_lf`(`result` pass) |
| 설정 해시 | `R_SETTINGS_SHA256` `b9c5c905c5fec3ea9840e9644b177aeb6689ab4191d1e2e1dfce17e9a4518b6a`(r0 설정 문서). 단계 설정 `stage_settings.json`의 `settings_sha256`은 `301c829e64d9ae18c0b358a2c24bb23ca89a1e00abe73a555d579219e4a0a669`(injected·uninjected001 같은 내용) | [SI] `run_identity.r_checks.settings.settings_sha256`, `runs/eval/m33/e3x_retry/injected/stage_settings.json` |
| `pp_set_id`·`PPConfig` sha | S3, `eb2796ddfaa9611e5030342c4ed66b00c91ff2ed6a6698ee5979b9afe080602f`. D0 게이트 sha `107d7b53f30667684baf7c2236bdf338c861309531ff17497e2836193bea08c6`. 단언 4개 모두 참, `result` pass | [SI] `run_identity.sealed_pp_assertion` |
| inject blob 단언 | `2d88ff0a25c5cbc1bb7ca8698ba422c633f5b531` pass | [SI] `run_identity.r_checks.inject_blob` |
| r0 정체 11측선 sha | 11개 기록(값은 출처 참조) | [SI] `run_identity.r_checks.r0_identity_sha256` |
| 시드 | pool 20261120, plan 20261122, texture 20261123, 부트스트랩 20261121(B = 20,000, `n_boot_is_sealed` true) | [SI] `run_identity.seeds`, [J] `constants`, `n_boot_is_sealed` |
| 버전 | python 3.13.18, numpy 2.5.4, opencv 4.14.0, torch 2.14.1+cu130(CUDA 13.0), ultralytics 8.4.192, gpr_detection 0.1.0, GPU RTX 4070 SUPER | [SI] `run_identity.versions`, `gpu` |

**단계별**

| 단계 | run_id | 시작 → 끝(KST) | 시간 | 작업자 | VRAM 최대 | NMS 경고 · max_det 타일 | 출처 |
|---|---|---|---|---|---|---|---|
| uninjected001 | `6808b113-7983-4cb0-a2cd-287ed4c94af2` | 2026-11-26T12:58:04 → 12:59:58 | 120.5 s(러너 wall 129 s) | 1 | 예약 1,988 MB(러너 측정 2,191 MB) | 0 · 0(300박스 타일 0, 기록 열 `n_tiles_maxdet` 합 0) | [SU] `run_identity`, `resources`, `nms_time_limit_warnings`, `tiles_with_300_operating_boxes`; [RL] `runner_summary.json`; `uninjected001_records.csv` 열 합 |
| injected | `1c1d32c6-4cbd-4ac3-a528-cef96d9f166a` | 2026-11-26T13:00:09 → 13:48:34 | 2,911.4 s(러너 wall 2,916 s) | 5 | 예약 1,640 MB(러너 측정 1,843 MB) | 0 · 0 | [SI] `run_identity`, `resources`, `nms_time_limit_warnings`, `tiles_maxdet`, `defects` |
| judge | `69c26b1d-4e35-451e-8d70-1da95866c728` | 2026-11-26T13:49:06 → (`finished_at` null). 러너 `[start]`는 13:48:35, `.done` 13:49:07이라 [J] `started_at`은 실제 시작보다 늦은 기록 시각으로 보인다(의미 미상, 다음 판에서 판정 도구 수정) | 28.8 s(러너 wall 33 s) | – | 0 | – | [J] `identity`; [RL] `logs_P\e3x-retry-judge.log` |

- 건수: 측선 153, 위치 242(쌍곡선 144 + 띠 98), 실현 5,852, 추론 5,816([SI] `counts`). `defects` 7개 키 모두 0이다(`shift_mismatch`, `off_differs_from_on`, `texture_check`, `zero_test`, `r_edge_differs_from_plan`, `nms_time_limit_warning`, `tile_with_300_operating_boxes`).
- 실행 마스크(정체성 필드 밖, verif 재확인 기록 8): 재시도 세 단계 모두 P코어 `0xFFFF`·Normal([RL] `logs_P\*.log` 첫 줄 `[start] … mask=0xFFFF priority=Normal`). stop-on-fail 러너 `run_retry_P.ps1`(sha256 `dd37e2424d12cd704588a993b57ca0fed925e4dd1d3f45fe2fdcda7646430b65`). 11:11에 러너를 띄웠고, 다른 세션 예약이 끝나기를 6,373 s 기다렸다(`logs_P\_wait_until.log` `waited_s` 6373.3). 예약 `r-20261126125749-98ec5f`(owner `gpr_detection:m33_e3x_retry`, mask 0xFFFF)는 끝난 뒤 풀었다(`_runner.log`).
- 판정 도구 입력 정체성: `identity.injected_run_id` = injected run_id이다. `identity.injected_summary_sha256` `358ad13fb94c87c5fa48f929cf848c28cb2b15e09ce007e18e640ba5372b4e78`은 [SI]의 실제 sha256(raw = LF)과 같다. 단위 표 sha 7개([J] `units_sha256`)도 실제 파일과 같다. injected 출력 sha 5개([SI] `outputs_sha256`)와 uninjected001 기록 sha([SU] `outputs_sha256`)도 실제 파일과 같다(이 작성에서 `sha256sum`으로 대조).
- 기록 결함(판정 영향 없음): [J] `identity.finished_at`이 null이다. 판정 도구가 끝 시각을 쓰지 않는다. 끝 시각은 러너 `.done`의 2026-11-26T13:49:07에서 본다.

## 2. 판정 가능성, 실패·재실행 이력 (§6 2)

- 판정 가능성 블록
  - injected: `judgeable` true, `blockers` [], `expected_folds` = `folds_run` = [0–4], `tree_stable` true, `tree_changes` []([SI] `judgment`).
  - uninjected001: 같음([SU] `judgment`).
  - 판정 도구: `judgeable` true([J]).
- 재시도 폴더 차수: injected·uninjected001 모두 `runs/eval/m33/e3x_retry/`(k = 1)다. 두 `run_identity`의 봉인 기록 sha와 코드 blob이 같다(A-4 조건).
- 판정 입력 확인: 판정 도구 argv는 기본 인자 `["analysis/detection/m33_e3x_judge.py"]`이고, 기본 입력은 `e3x_retry/{injected,uninjected001}`이다([J] `identity.argv`, verif 재확인 기록 6). 재집계(§11)도 `e3x_retry/injected/`의 기록 csv와 `e3x_retry/judgment/`의 단위 표만 읽었다. 1회차 폴더는 입력이 아니다.
- **injected 1회차(실패, 열지 않고 보존)**
  - run_id `73fe0456-29e6-438f-bb56-a6c915158f2d`(규칙 공개 22에서 옮김).
  - HEAD `d720fe369322cfe2e63b0ef7eaa34ce338c59647`, 봉인 2 커밋 `729d19ced8674039efac0cfe906fcb41f45c31e3`.
  - 코드 blob(봉인 2 기록 `analysis/detection/m33_e3x_seal2.json` `code_blobs`. 재시도와 다른 것은 셋뿐): `m33_e3x.py` `b601daf6d54d5717ae5386c5a1bdd4288f0c1e0b`, `m33_e3x_place.py` `e22665181b010e45e0a8fdb5ecfe5efcf84e64d0`, `m33_e3x_judge.py` `0da693fda4ded97212b31c5831ebcbf3be9eb5c4`. 나머지 15개는 위 표와 같다.
  - src tree `d0e54933f3a0c87cf2b8d43d255e97bd6bb97122`(봉인 2 기록).
  - 종료 1 `e3x_defect`, 3,119 s. 실패 단언은 `texture_check`의 `jitter_energy`(σ_δ 0.1·0.2)였다. 위치는 4TU `010.18/Path2` b00·b01, 4TU `011.16/Profile_6453` b00, IDS `pipe/1030_181230AB/LID10001` b00·b01의 5개다(공개 22).
  - 1회차 uninjected001은 ok(122 s)였고, judge는 돌지 않았다(메인 세션 인계 기록).
  - 시작·끝 시각(메인 세션이 1회차 러너 로그의 `[start]`·`[end]` 줄에서 옮김, 2026-11-26): uninjected001 `2026-11-20T14:26:57` 시작 exit 0 wall 122 s, injected `2026-11-20T14:28:59` 시작 exit 1 wall 3,119 s(끝 시각은 로그에 따로 없고 `.done`도 없어 시작 + wall ≈ `2026-11-20T15:20:58`로 적는다, 추정; `e3x-injected.log` mtime 15:20과 맞음), 이어서 `[stop] e3x-injected failed (exit=1); later steps NOT started`(judge 안 돎). 마스크 0x7FFF·Normal. 러너 로그는 다른 세션의 scratchpad에만 있었으므로 폴더째 `docs/processing/evidence/m33_e3x_try1_runner/`(SHA256SUMS)로 복사했다. 판정서 작성자는 1회차 폴더와 이 로그를 열지 않았고, 메인은 제어 줄만 읽었다.
- **1회차 폴더를 열지 않았다는 확인**: 판정서 작성자는 `runs/eval/m33/e3x/{uninjected001,injected,judgment}/`를 Read·Grep·Glob·나열·해시 어느 방식으로도 열지 않았다. 검색 범위로 `runs/eval/m33/`·`runs/eval/m33/e3x/`를 쓰지 않았다. 연 것은 `runs/eval/m33/e3x/plan/`(블라인드 아님)뿐이다.
- **원인 확인 열람 범위(공개 23)**: 증거는 `docs/processing/evidence/m33_e3x_prerun_r1/disclosure23/`에 있다.
  - `reconstructed_from_transcript.md` sha256 `2010ea043753a7d7e83be66b3e2389105300f3e8d9d29d2664bc500af5877fae`
  - `transcript_excerpt_lines_50_52_54_55_59_60.jsonl` sha256 `fed36bbadff4040e290a63b9c1c20d6761c07179ad5aa0119254f2d0b3cb5905`
  - 폴더 `SHA256SUMS`(14항목)는 이 작성 때 `sha256sum -c`로 모두 통과했다. SUMS 자체의 sha256은 `a5725eefed9e918d018852c6ba4dd47b6be635cf62e3b21fa4b9d4e2068593ab`이다.
  - verif 재확인(기록 5)의 성격 판단을 그대로 적는다. 이 예외 규정(`plan.md:654`)은 열람(2026-11-20 15:24) 뒤에 v0.5(17:48 커밋)에서 만든 **사후 규정**이다. 스크립트·stdout 원본이 없어 **transcript 전사로 대신한 증거**다. 스크립트는 `summary.json`과 `parts\*.json` 전체를 메모리로 읽었다. 출력된 것은 파일 이름 8개, `defects` 키 이름 7개, 0이 아닌 키 `texture_check`와 그 측선 3개다. 판정 지표 값은 나오지 않았다.
- **봉인 2 r1 기록**([SR])
  - `supersedes`: path `analysis/detection/m33_e3x_seal2.json`, sha256_lf `94c671edb27ba5077551855b529aa4d5370bb8e0c8dbc36ea6b0d49b518bb776`, commit `729d19ced8674039efac0cfe906fcb41f45c31e3`.
  - plan 재사용 근거(`plan_reuse`): ast 기준으로 바뀐 최상위 정의는 71개다. plan 경로(`main`·`parse_args`·`run`·`start_context`·`run_plan`·`print_final_line`, 디스패치 `run` → `start_context`·`run_plan`)에서 도달하는 정의는 158개이고, 그중 바뀐 것은 `m33_e3x:DEFAULT_PRECHECK_DIR`·`m33_e3x:SEAL2_RECORD` 두 개(argparse 기본값)뿐이다.
  - plan 표 5개·`plan_report.json`(`eaa9682414b61d1f35d52ada3923bd1877c5d251badf683f98d4e81ef8136c50`)·pool csv sha는 봉인 2 값 그대로다. plan을 만든 blob(`plan_code_blobs`)도 봉인 2 값이다. plan run_id `aea5035a-31ff-4f02-855d-d6b4d1bf89b1`, plan HEAD `38c05cbf69f179f94bcb7ae25082dd4e01dca8ae`.
  - verif 재확인(기록 1)이 독립 AST 대조로 같은 수를 얻었다.
- **precheck r1**
  - HEAD `408df3a49ea1a2539b9025e21ef29283c3ff1adb`(50a28a3에서 `.gitignore` 2줄만 바뀜, `git diff --stat 50a28a3 408df3a`).
  - run_id `04c09681-b000-4fb6-b557-080200fd5a56`. 러너 시작 2026-11-26 02:27:39, python 시작 02:27:46, 끝 03:05:23([PC] `run_identity`).
  - E코어 마스크 `0x0FFF0000`·BelowNormal·`--workers 4`, 종료 0, wall 2,267 s, `status` ok. P코어는 다른 세션이 예약 중이어서 E코어로 돌렸다(compute-scheduler 권고). 마스크·우선순위는 정체성 필드가 아니다(verif 재확인 기록 8).
  - `checks` 10개가 모두 참이고 `not_run` []이다([PC] `checks`). 실행 순서는 1 → plan → 10 → 3 → 6 → 2·4·5·7·8·9다(verif 재확인 기록 3).
  - 10번 `cpu_assertions.json`([CA] `assertions`): 측선 155(본 표본 153 + precheck 2), 검사 1,742, 실패 0, `guard_problems` [], `failures` []. 단언별 검사 수 = 기대 수는 다음과 같다.

    | 단언 | 검사 수 = 기대 수 | 실패 |
    |---|---|---|
    | `shift_mismatch` | 569 | 0 |
    | `off_differs_from_on` | 768 | 0 |
    | `r_edge_differs_from_plan` | 149 | 0 |
    | `texture_check` | 101 | 0 |
    | `zero_test` | 155 | 0 |

  - 질감 단언 분포([CA] `texture_distribution`): `jitter_energy_identity` 최대 7.54×10⁻¹⁵(문턱 10⁻¹²), `jitter_nyquist_fraction` 최대 1.025×10⁻⁸(문턱 10⁻⁶).
  - **실측 precheck 2측선에서 두 경로(w1 영상 경로와 10번 무영상 경로)의 검사 수를 직접 대조하지 않았다.** 대조는 합성 대역 시험에서만 했다(verif 재확인 기록 9). 본 실행이 같은 단언을 다시 세었고 모두 0이므로 판정 영향은 없다.
  - 폴더와 사본: `runs/eval/m33/e3x_precheck_r1/`, `runs/eval/m33/e3x_precheck_r1_copy/`(각 22파일). 파일별 sha256 목록은 `docs/processing/evidence/m33_e3x_prerun_r1/precheck_r1_runner/precheck_r1_sha256.txt`(44줄, 파일 sha256 `98cd3797b9919598b80dfa1ba7e8f7b3e8cee99c9522cbee7dac4f63f74c9daa`)이다. 이 작성 때 44/44가 `sha256sum -c`로 통과했다. 두 폴더 공통 sha256은 다음과 같다.

    | 파일 | sha256 |
    |---|---|
    | cpu_assertions_lines.csv | `4a6cd43b25959cfbe5c533a6de59962052f919ec0d3f4b611ebe18aadaa8dcb8` |
    | cpu_assertions.json | `17408607bdb71eafd0e28b68a2cb2d46c2b3c508ec3861b952e4c05fd9d5a1ec` |
    | precheck.json | `a374e30151bdb3cf63158a178256d8cae0c27e32d08cb4ecf11453c96f99f260` |
    | progress.json | `e24f5ff9f7ac30da3a774400116ea7bb4ce435139bf3e7a003eefef7e604bf3b` |
    | plan/band_arms.csv | `16ce7467aedb9ecdd7fcae5d55221f100959ea9845b3e21b5b2119531c163087` |
    | plan/band_positions.csv | `33b6871b0290d63e99b10d027887727dedec27dc83cba9e550758acb07f85abe` |
    | plan/exclusions.csv | `4369750153e2b3059884be74284c614a1007059a9d07fdfcad6914e31407a875` |
    | plan/hyperbola_positions.csv | `d2235c97f6c62c5f7a9a67279148fda4e092621211340ccc68e8e88c0276c731` |
    | w1/·w2/band_records.csv | `ff27d3d0422a9d86a9d86e4b2532f0da1602407f28e2800f10127a2d4f18b2ce` |
    | w1/·w2/hyperbola_records.csv | `1265787d5cda217dae9f5d6b52e9bd09248172770a8f62111e2de5b76aa5add6` |
    | w1/·w2/realizations.csv | `1135e1cb8c107b05d0ee62e7f969a1592411ae140ecb9965ff754abbbbcbf734` |
    | w1/·w2/rejections.csv | `d4b71d1ff43678d0e43ffcff88e4d8f8ad624a94aa947e9604418aebcc19094e` |
    | w1/·w2/truths.csv | `1cfb37aa9ac5ba833470b88d5b2b70a9a409663473f1b8579bd1413f5faa7555` |
    | w1/parts/4tu__02.5__Path1.json | `b21790d3d15ca309fee5b2ad7ec5c68c81824ccc8c6a768145d102fd22c305cc` |
    | w1/parts/gz_ids__pipe__1.15_200116AC__LID10003.json | `cc983e60fad3a9dd3d60a39569c0b0d862a00f1c7305c39de5b75101b64e4119` |
    | w2/parts/4tu__02.5__Path1.json | `a3375cc350755a6260196588e48612cf020a5b8afad58734395959534931e2cc` |
    | w2/parts/gz_ids__pipe__1.15_200116AC__LID10003.json | `8b209bf27aefc6a791019f2843dbaa3cb0db19036b25fa9b4c05fabe716bf978` |

    7번(workers 1 ↔ 2 바이트 동일)의 비교 대상은 기록 csv 5개다([PC] `outputs_sha256.w1` = `w2`). `parts/*.json`은 작업자별 부분 파일이라 w1·w2가 다르고, 비교 대상이 아니다.
- **실행 중 사용자 결정(재확인 뒤, 결과 전)**: 아래 1·2는 실제로 발동하지 않았다.
  1. 재시도 uninjected001이 0이 아닌 종료로 끝나면 멈춘다. 규칙 표 `plan.md:649`의 "injected를 막지 않는다"보다 이 결정이 우선한다. 재시도 uninjected001은 종료 0이었다. 이 결정 기록(`m33_e3x_prerun_audit_r1.md` 메인 덧붙임 1)은 "판정 도구에 성공한 uninjected 폴더를 `--uninjected`로 명시해 넘긴다"고 적었으나, 러너 judge 명령에는 이 인자가 들어가지 않았다(공개). 판정 도구는 기본값 `runs/eval/m33/e3x_retry/uninjected001`으로 돌았고, 이것은 성공한 재시도 폴더와 같다([RL] `steps.json`, [J] `identity.argv`, [SU] `status` ok·`judgeable` true·봉인 sha와 code_blobs가 injected와 같음, verif 판정 후 감사 발견 1). 결과 영향은 없다.
  2. 재시도 단계가 종료 3 등 부분 출력으로 끝나면 `--resume`을 쓰지 않고 `e3x_retry2/`로 간다. 모두 종료 0이었다.
  3. 감사 문서를 재시도 전에 커밋한다. 그래서 precheck r1 HEAD(408df3a)와 재시도 HEAD(bc57ef1)가 다르다. `git diff --stat 408df3a bc57ef1`은 `.gitattributes`, `docs/processing/evidence/m33_e3x_prerun_r1/**` 15파일, `docs/processing/m33_e3x_prerun_audit_r1.md`만 더했다. 봉인 blob·src tree는 같다.

## 3. plan 수 (§6 3)

출처는 [PR](봉인 2 plan 보고, 재시도가 재사용)와 [BA]다. 판정 도구는 같은 값을 [J] `planned_band_positions`·`report_only.R-e.shift_P_by_row_model`로 다시 냈고, 내용이 같다.

| 항목 | 값 | 출처 |
|---|---|---|
| 배경 측선 | 4TU 100, IDS `main_400` 53 | [PR] `lines.by_dataset` |
| **실제로 뽑힌** R3 공유 4TU | 23활동, 24측선 | [PR] `lines.fourtu_r3_shared_activities`, `fourtu_r3_shared_activity_lines` |
| 쌍곡선 위치·군집 | 4TU 84위치·30군집(Kish 19.6), IDS 60위치·15군집(11.39), 합 144·45. 위치는 66측선에 있다. 측선 둘째 위치 0, 부족분 없음(`shortfall` {}) | [PR] `hyperbola.clusters_all`, `second_on_line`, `shortfall` |
| 쌍곡선 모델별 위치 | A0 15모델: 4TU 4 + IDS 4. A1(M12·M15·M18): 4TU 8 | [PR] `hyperbola.positions_by_dataset_model` |
| HH1 층(모델 조건 전) | 조작 층 105(4TU 57·IDS 48, 군집 37), 회색 띠 7, 음성 대조 32(4TU 20·15군집, IDS 12·7군집, 합 22군집) | [PR] `hyperbola.edge_strata_before_model_condition`, `clusters_manipulated`, `clusters_negative_control` |
| 모델별 $S_p^\text{edge}$ | {12, 18}: M03·M04·M06·M09·M16·M19 각 8, M07 1. {18}: M02·M05·M08·M10·M11 각 8, M07 7, M15 6, M18 3. 회색: M15 2, M18 5. 음성 대조: M01·M12·M14·M17 각 8 | [PR] `hyperbola.edge_composition_by_model` |
| M02·M07·M08 위치별 $r_\text{edge}(p)$(dB) | 24위치 −13.65 ~ −10.94. M07 `pipe/1030_181230AB/LID10001:h00:00` −10.94만 {12, 18}이고, 나머지는 −13.65 ~ −12.06이라 {18}이다 | [PR] `hyperbola.r_edge_db_m02_m07_m08` |
| 띠 위치 | 98(4TU 60, IDS 38), 67측선 | `band_positions.csv` |
| 띠 행·길이별 팔 | 28 긴 98, 47 긴 98, 47 짧은 98, 68 긴 98, 68 짧은 98, 90 긴 61(합 551) | [PR] `band.arms_by_row_length`, [BA] |
| 가설별 위치·군집(plan) | HB1·HB2: 4TU 60·30(Kish 23.38), IDS 38·14(9.89). HB1b: 4TU 38·23(19.51), IDS 23·12(10.37). `single_background` 없음 | [PR] `band.hypothesis_clusters` |
| 띠에서 빠진 측선과 사유 | 20측선 `long_arm_over_0.45_w_eff`. 위치 없는 측선은 이 20을 포함해 86 | [PR] `band.lines_excluded`, `lines_without_position` |
| (측선, 모델) 거부 사유 | `apex_outside_record` 112, `no_model` 47, `beyond_synthetic_window` 16, `shift_limit` 7 | [PR] `band.model_reasons` |
| 팔 배치 실패(후보 탐색 중) | 28 px `band_zone` 566,673, `strong_layer_zone` 369, `uninjected_box_zone` 143,350, `candidate_apex_zone_auto` 111,856, `saturation_zone` 104,724. 47 px `uninjected_box_zone` 9,797, `wing_past_record_end` 44, `strong_layer_zone` 14. 68 px `wing_past_record_end` 25,945, `uninjected_box_zone` 10,444, `box_clipped` 9,010, `strong_layer_zone` 231 | [PR] `band.arm_failures` |
| 빠진 행 | 90 px 행은 98위치 중 37위치에서 빠짐(`has_deep_row` False 37). 해당 위치는 HB1b에서만 빠진다 | `band_positions.csv` |
| 세로 이동량 $\lvert n_2\rvert/P$(행별) | 28 px: 중앙 1.84, p95 6.66, 최대 8.12(> 8 P 2팔). 47 px: 2.91 / 5.19 / 6.10. 68 px: 5.73 / 8.68 / 9.58(> 8 P 21). 90 px: 9.24 / 11.79 / 11.91(> 8 P 40). 전체 최대 11.91 P ≤ 12 P | [BA] `shift_P` |
| 모델·행별 이동량 | 중앙·최대 표 전체는 [PR] `band.shift_P_by_row_model`(= [J] `report_only.R-e.shift_P_by_row_model`). 예: 90 px에서 M01 중앙 11.64·최대 11.91, M17 11.29·11.88 | 같음 |
| 재측정 보정 `shift_correction_samples` | 551팔 모두 0 | [PR] `band.shift_correction_samples`, [BA] |
| 계단 `shift_step_samples` | 551팔 모두 0. 목표와의 차 `shift_gap_samples` 최대 0.497 샘플(≤ 1) | [PR] `band.shift_step_samples`, [BA] |
| `shift_mismatch` 검사 | 목표 행 (i) 0, 팔 사이 (ii) 0 | [PR] `band.shift_mismatch_checks` |
| 짧은 팔 중심 −0.5 트레이스 | 40위치(짧은 팔 80) | [PR] `band.short_arm_center_offset`, [BA] |
| HB1 모델 구성 | 4TU: K7_P1_D060 4, K7_P1_D110_S002 1, K7_P1_D150_S003 4, K7_PVC_D070 1, M01 6, M02 2, M04 4, M06 3, M07 2, M08 2, M09 1, M10 4, M11 1, M14 8, M17 4, M18 5, M19 8. IDS: M01 7, M02 1, M03 3, M05 4, M06 3, M07 4, M09 1, M10 7, M14 5, M16 2, M17 1 | [PR] `band.hb1_models` |
| **HB1b 모델 구성(HB1과 비교)** | 4TU: K7_P1_D060 4, K7_P1_D110_S002 1, K7_P1_D150_S003 2(HB1 4), K7_PVC_D070 1, M02 2, M04 4, M06 2(3), M07 2, M08 2, M10 3(4), M11 1, M14 4(8), M17 1(4), M18 4(5), M19 5(8). **M01 6·M09 1은 4TU HB1b에 없음.** IDS: M01 2(7), M02 1, M03 2(3), M05 2(4), M06 3, M07 2(4), M10 5(7), M14 5, M17 1. **M09 1·M16 2는 IDS HB1b에 없음** | [PR] `band.hb1b_models` |
| 가설별 $G_g$·$G_{\text{eff},g}$(판정 대상, 결과 반영 뒤) | §4 각 가설 표 | [J] `hypotheses.*.g_by_background`, `g_eff_by_background` |

## 4. 가설별 판정 (§6 4)

공통 사항
- `single_background`·`single_cluster_background`·`degenerate_ci`는 모든 가설에서 false다. `zero_variance_background`는 모두 []다([J] 각 가설 키).
- 재집계 범위: 아래 표의 추정량·구간·$\nu$·$v_g$·$\nu_g$($= G_{\text{eff},g}-1$)·$G_g$·$G_{\text{eff},g}$·위치·군집·$F$·결정·$M^\ast$는 **기록 csv에서 다시 셌다**([RC]). 구간 끝은 [J]와 3×10⁻¹⁷ 안에서 같다. 단위 표 7개(`units_*.csv`)도 위치 집합이 같고, 위치값 차가 ≤ 2.2×10⁻¹⁶이다([RC] `units_csv_comparison`). 배경별 값과 HH2 보조 구간은 json에서 옮겼다(각 표에 표시).
- 판정서 문구는 §3-3 결정표의 문구이고, 판정 도구 [J] `wording`과 같다.

### 4-1. $M^\ast$와 $M^\ast_e$ (결과 의존, §3-4 결과 의존 분모)

- 모델 평균 $(Y_{p,\text{on}}+Y_{p,\text{off}})/2$([RC] `model_mean_on_off`): M01 0.811, M02 0.350, M03 0.278, **M04 0.091**, M05 0.725, M06 0.281, M07 0.563, M08 0.234, M09 0.191, M10 0.691, M11 0.534, M12 0.313, M14 0.906, **M15 0.006**, **M16 0.047**, M17 0.993, **M18 0.069**, M19 0.409.
- $M^\ast$(≥ 0.10, 14모델): M01·M02·M03·M05·M06·M07·M08·M09·M10·M11·M12·M14·M17·M19([J] `hypotheses.HH1.m_star`). 빠진 모델은 M04·M15·M16·M18이다.
- $M^\ast_e$(조작 층 위치가 있는 $M^\ast$ 모델, 10모델): M02·M03·M05·M06·M07·M08·M09·M10·M11·M19([J] `hypotheses.HH1.m_star_e`). 모델마다 8위치다([RR] `HH1_positions_by_model`).
- 층(모델 조건 뒤): 조작 층 80위치, 음성 대조 32위치, 회색 띠 0위치(M15·M18이 모두 빠짐)([RC] `counts.strata_mstar`).
- **조작 층의 {18}만 비중(실제 $M^\ast$ 뒤)**: 80위치 중 47위치(58.75 %)다. {12, 18}은 M03·M06·M09·M19 각 8과 M07 1, 합 33위치다([RC] `HH1.snr_composition`). [MDE] §5 공개 2의 예상(60–67 %, $\lvert M^\ast_e\rvert$ 9–11)과 같은 범위다. 규칙 §3-7 (iv)의 "약 53 %"보다 높다. §3-7 (iv)는 M09·M15·M18이 빠지는 경우를 가정했는데, 실제로는 M04·M16({12, 18})이 빠졌다.
- int16 거부로 빠진 위치: 2위치(`010.14/Path1:h00:01`, `010.24/Path2:h00:00`)다. 18 dB 켬·끔 쌍이 함께 거부돼 두 팔 평균을 만들 수 없다(`runs/eval/m33/e3x_retry/injected/rejections.csv`). 두 위치는 음성 대조 모델 M01·M17의 위치이고, HH2 대상에서도 빠진다(142위치, R-f M01·M17 7위치).

### 4-2. HH1 — 개구 끝 처리(양측, 조작 층)

| 항목 | 값 | 출처 |
|---|---|---|
| $\hat\Delta$(모델 같은 가중, 켬 − 끔) | 0.009375 | [J] `hypotheses.HH1.estimate`, [RC] |
| 판정 구간(Satterthwaite, 수준 $1-0.05/6$) | [−0.001466, 0.020216] | `ci_low`·`ci_high` |
| $\nu$ | 21.136 | `nu` |
| $v_g$ | 4TU 6.932×10⁻⁶, IDS 6.944×10⁻⁶ | `v_by_background` |
| $\nu_g$ | 4TU 12.559, IDS 9.127 | `nu_by_background` |
| $G_g$ / $G_{\text{eff},g}$ | 4TU 17 / 13.559, IDS 13 / 10.127 | `g_by_background`, `g_eff_by_background` |
| 배경별 값 | 4TU 0.00875, IDS 0.01000 | `by_background`(json에서 옮김. 모델 같은 가중과 단순 평균이 같음, [RR]) |
| 위치 / 군집 | 80 / 30 | `n_positions`, `n_clusters` |
| 최소 수(§3-4) | 조작 층 위치 80 ≥ 30, $\lvert M^\ast_e\rvert$ 10 ≥ 5, 군집 30 ≥ 15. 끔 int16 거부 비율 0.0(≤ 0.20) | `off_int16_rejection_share` |
| δ / 바닥 양 $F$ | 0.05 / 0.41781(`floor_ok` true) | `delta`, `floor_mean` |
| 결정 | `negligible`(구간 ⊂ (−0.05, +0.05), $F$ ≥ 0.10) | `decision` |
| 판정서 문구 | **기각(실질 효과 없음)** | `wording` |
| 음성 대조 점검 1 | $\hat\Delta_\text{neg}$ = −0.005357, 구간 [−0.016141, 0.005427](같은 수준), $\nu$ 6.517, 30위치·21군집(4TU 14군집, IDS 7군집, $G_{\text{eff}}$ 12.46 / 5.14), 모델 M01·M12·M14·M17. 구간이 0을 포함하므로 **(c) 통과, 표지 없음**(`outcome` pass) | [J] `manipulation_checks.negative_control`, [RC] `HH1_negative_control` |
| fold 방향 일치 | 4/5 | [J] `report_only.R-e.fold_direction_agreement.HH1` |

- 위치별 차 $D_p$: 80위치 중 61위치가 정확히 0이고, 양 14·음 5다(범위 −0.10 ~ +0.20, `units_HH1.csv`). 모델 평균 $D$는 M07 +0.050, M03·M09 +0.019, M05·M11 +0.0125, M19 +0.006, M02·M06·M10 0, M08 −0.025다([RR] `HH1_model_mean_D`).
- 음성 대조 점검의 성질(§3-3 점검 1 (d)가 판정서에 적으라고 한 문장): 문턱 근처(10위치·5군집)에서 이 점검은 $\lvert\Delta_\text{neg}\rvert$ ≳ 0.06–0.08만 잡는 약한 점검이다. 이번 실제 수는 30위치·21군집이고, [MDE] §2의 음성 대조 MDE80은 0.028이다.
- 음성 대조 위치에서 켬·끔 팔의 `period_inj`가 다른 단위 6개(R-e, [J] `report_only.R-e.negative_control_period_inj_mismatch_units`).

### 4-3. HH2 — 모델 몫 > 0.5

| 항목 | 값 | 출처 |
|---|---|---|
| $\hat\eta^2_p$(배경 + 모델 가법, 모델 Type II) | 0.658997 | [J] `hypotheses.HH2.estimate`, [RC] |
| 판정 구간(기본 구간, $a$ = 0.05/12) | [0.497979, 0.739101] | `ci_low`·`ci_high` |
| 문턱 | 0.5 | `threshold` |
| 위치 / 모델 / 군집 | 142 / 18(모두 위치 ≥ 4) / 45(4TU 30, IDS 15) | `n_positions`, `models`, `g_by_background` |
| 최소 수 | 위치 ≥ 4인 모델 18 ≥ 12, 군집 45 ≥ 20 | – |
| NaN 복제 | 0 | `n_nan_replicates` |
| 결정 | `inconclusive`(하한 0.49798 ≤ 0.5, 상한 0.7391 ≥ 0.5) | `decision` |
| 판정서 문구 | **추정 유지(구간이 넓음)** | `wording` |
| 보고 전용 $\omega^2_p$ | 0.6085, 기본 구간 [0.4243, 0.7013], 백분위 [0.5157, 0.7927] | `omega2_p`(json에서 옮김. 점추정은 [RC]에서 같음) |
| 보고 전용 $M^\ast$만 $\eta^2_p$ | 0.5291, 기본 구간 [0.2988, 0.6287], 백분위 [0.4294, 0.7593] (110위치·14모델·38군집) | `eta2_p_mstar`(json에서 옮김. 점추정은 [RC]에서 같음. 구간은 규칙대로 HH2 복제를 다시 쓴 값이다. [RC]는 새로 뽑아 [0.3043, 0.6277]이었다) |
| 보고 전용 백분위 구간(R-g) | [0.5789, 0.8200] | `ci_percentile_low`·`high` |

- 하한이 문턱보다 0.0020 낮다. 백분위 구간(보고 전용)은 0.5를 넘지만, 규칙은 기본 구간으로 판정한다(v3-2). 백분위 구간은 부분 $\eta^2$의 위 편향을 한 번 더 얹는다([STV] 6). 라벨은 바뀌지 않는다.

### 4-4. HH3 — 배경 있는 30 dB 포화(12 − 30 dB, 테이퍼 켬)

| 항목 | 값 | 출처 |
|---|---|---|
| $\hat\Delta$ | 0.091233 | [J] `hypotheses.HH3.estimate`, [RC] |
| 판정 구간 | [0.036540, 0.145926] | `ci_low`·`ci_high` |
| $\nu$ / $v_g$ / $\nu_g$ | 19.307 / 4TU 1.2946×10⁻⁴·IDS 2.1695×10⁻⁴ / 4TU 13.126·IDS 9.530 | `nu`, `v_by_background`, `nu_by_background` |
| $G_g$ / $G_{\text{eff},g}$ | 4TU 18 / 14.126, IDS 14 / 10.530 | 같음 |
| 배경별 값 | 4TU 0.12683, IDS 0.06471(판정 도구는 위치 단순 평균. 모델 같은 가중으로는 0.12986 / 0.06410, [RR] `HH3_by_background`) | `by_background`(json에서 옮김) |
| 위치 / 군집 | 92 / 32 | – |
| 최소 수·거부 | 92 ≥ 40, 32 ≥ 20. 30 dB 거부: $M^\ast$ 112위치 중 20위치(0.1786 ≤ 0.30) | `rejection_30db_share`, [RC] `HH3.positions_missing_30db`·`n_mstar_positions` |
| δ / $F$ | 0.05 / 0.46129 | `floor_mean` |
| 결정 / 문구 | `confirmed` / **확인**(점추정 0.091 ≥ δ라 "5 %p 미만" 덧붙임 없음, `small_effect` false) | `decision`, `wording` |
| $M^\ast$ 모델 | 14모델(HH1과 같은 $M^\ast$) | `models` |
| fold 방향 일치 | 4/5 | [J] `report_only.R-e.fold_direction_agreement.HH3` |

- 위치별 차는 양 49·음 14·0 29다(`units_HH3.csv`). 모델 평균이 음수인 모델은 M08(−0.0375)과 M14(−0.08)다([RR] `HH3_model_mean_D`).

### 4-5. HB1 — 얕은 행(47·68 px 평균 − 28 px, 긴 팔)

| 항목 | 값 | 출처 |
|---|---|---|
| $\hat\Delta$(배경 같은 가중) | 0.237333 | [J] `hypotheses.HB1.estimate`, [RC] |
| 판정 구간 | [0.149342, 0.325324] | `ci_low`·`ci_high` |
| $\nu$ / $v_g$ / $\nu_g$ | 20.952 / 4TU 3.6426×10⁻⁴·IDS 5.4831×10⁻⁴ / 4TU 22.377·IDS 8.890 | – |
| $G_g$ / $G_{\text{eff},g}$ | 4TU 30 / 23.377, IDS 14 / 9.890 | – |
| 배경별 값 | 4TU 0.28459, IDS 0.19007 | `by_background`([RC]에서 같음) |
| 위치 / 군집 | 98(4TU 60, IDS 38) / 44 | – |
| 최소 수 | 98 ≥ 30, 배경별 ≥ 15, 44 ≥ 20, `shift_mismatch` 0 | – |
| δ / $F$ | 0.05 / 0.14986 | `floor_mean` |
| 결정 / 문구 | `confirmed` / **확인** | – |
| 대비 점검 2 | injected 띠 $c_\text{lat}$ 중앙(dB): 28 px 4.305, 47 px 4.205, 68 px 4.067, 90 px 3.622. 28 px가 47·68 px보다 3 dB 넘게 낮지 않으므로 **표지 없음**(`flag` false) | [J] `manipulation_checks.contrast`, [RR] `HB1_contrast_c_lat_median_db`(긴 팔·r0·평탄·12·18 dB 단위 392개에서 같음) |
| 위치별 부호 | 양 85, 음 4, 0 9 | [RR] |
| fold 방향 일치 | 5/5 | [J] `report_only.R-e.fold_direction_agreement.HB1` |

### 4-6. HB1b — 깊은 행(47·68 px 평균 − 90 px, 긴 팔)

| 항목 | 값 | 출처 |
|---|---|---|
| $\hat\Delta$ | 0.097587 | [J] `hypotheses.HB1b.estimate`, [RC] |
| 판정 구간 | [−0.010901, 0.206074] | – |
| $\nu$ / $v_g$ / $\nu_g$ | 14.056 / 4TU 2.4442×10⁻⁴·IDS 1.00693×10⁻³ / 4TU 18.514·IDS 9.373 | – |
| $G_g$ / $G_{\text{eff},g}$ | 4TU 23 / 19.514, IDS 12 / 10.373 | – |
| 배경별 값 | **4TU 0.02132, IDS 0.17386** | `by_background`([RC]에서 같음) |
| 위치 / 군집 | 61(4TU 38, IDS 23) / 35 | – |
| 최소 수 | 61 ≥ 20, 35 ≥ 12, 배경별 ≥ 15 | – |
| δ / $F$ | 0.05 / 0.21182 | – |
| 결정 / 문구 | `inconclusive` / **추정 유지(구간이 넓음)** | – |
| 모델 구성 | §3 표(HB1과의 차 공개) | [PR] `band.hb1b_models` |
| 위치별 부호 | 양 38, 음 17, 0 6 | [RR] |
| fold 방향 일치 | 5/5 | [J] |

### 4-7. HB2 — 길이(47·68 px에서 긴 팔 − 짧은 팔)

| 항목 | 값 | 출처 |
|---|---|---|
| $\hat\Delta$ | 0.235917 | [J] `hypotheses.HB2.estimate`, [RC] |
| 판정 구간 | [0.162532, 0.309301] | – |
| $\nu$ / $v_g$ / $\nu_g$ | 21.713 / 4TU 2.6624×10⁻⁴·IDS 3.7310×10⁻⁴ / 4TU 22.377·IDS 8.890 | – |
| $G_g$ / $G_{\text{eff},g}$ | 4TU 30 / 23.377, IDS 14 / 9.890 | – |
| 배경별 값 | 4TU 0.26207, IDS 0.20976 | `by_background` |
| 위치 / 군집 | 98 / 44 | – |
| δ / $F$ | 0.05 / 0.15057 | – |
| 결정 / 문구 | `confirmed` / **확인** | – |
| 위치별 부호 | 양 88, 음 1, 0 9 | [RR] |
| fold 방향 일치 | 5/5 | [J] |

## 5. 마감 판정서 문장 변경안 (§6 5, §3-6 대응표)

| 결과 | 고칠 문장(변경안) |
|---|---|
| HB1 `confirmed` 그리고 HB2 `confirmed` | [R2R] §5 "재현율은 띠의 배치와 강하게 연관된다" → "**세로 행(같은 파형·같은 국소 SNR)**과 길이가 재현율을 정한다(③ 확인)". 대비 점검 표지가 없으므로 "(r0 대비 차 포함)"은 붙이지 않는다 |
| HB1b `inconclusive` | "AIHub 위치 범위 아래의 세로 행도 낮다"는 '추정' 그대로 둔다. 사유: 구간이 넓음(99.17 % 구간 [−0.011, 0.206]). 배경별 값(4TU 0.021, IDS 0.174)과 HB1b 모델 구성(§3)을 함께 적는다 |
| HH1 `negligible` | "**개구 끝 처리가 보이는 모델에서** 영향 없음(보이지 않는 모델 M01·M12·M14·M17 등은 구조상 차이 없음)". 음성 대조 표지는 없다(점검 통과). 그래서 E3(끔)와 단계 C(켬)의 수치 차를 개구 끝 처리로 돌리지 않는다. 함께 적을 범위(§10-1 해석): 이 결론의 모델은 $M^\ast_e$ 10모델이고, 조작 층 위치의 58.75 %가 18 dB 단위만 들어갔다. 끝 칸이 가장 강하게 보이는 M04·M16은 재현율이 바닥이라 $M^\ast$에 들지 못했다 |
| HH2 `inconclusive` | [Y0] §0 H1 "부분 기여(모델 간 차이를 정함)"는 '추정' 그대로 둔다. 사유: 구간이 넓음(기본 구간 하한 0.4980으로 경계 0.5와 0.0020 차). $M^\ast$만 $\eta^2_p$ 0.529(보고 전용)를 옆에 적는다 |
| HH3 `confirmed` | [Y0] §5 권고 7(SNR 상한 18 dB 근처)을 **확인**: "실측 배경이 있어도 30 dB 주입은 12 dB보다 쌍곡선 재현율이 낮다(차 0.091, 99.17 % 구간 [0.037, 0.146], ③ 확인)" |

## 6. 보고 전용 R-a–R-g (§6 6)

모든 95 % CI는 판정과 같은 Satterthwaite 방식을 수준 0.95로 쓴 값이고 **다중 비교 보정 없음**이다. 미리 고정한 해석 문장은 모두 '추정'이다(공개 13: 모두 귀무여도 방향 문장이 약 0.5개 거짓일 수 있음). 점추정 재집계 범위는 각 줄에 적었다. 나머지 CI·분포는 json에서 옮겼다.

- **R-a 띠 24 − 18 dB**: 0.06592, CI [0.04275, 0.08909], 98위치·44군집 → 하한 > 0이므로 "**24 dB까지 포화 없음**(추정)"([J] `report_only.R-a`). 판정 도구의 단위는 47·68 px 긴 팔이다. [RR] `R-a_rows_47_68_long`이 같은 점추정을 냈다. 네 행 긴 팔로 넓히면 점추정 0.0564다([RR] `R-a_rows_all_long`, 공개 사항 §9-2 (1)).
- **R-b $P^*$ 4.2 재영상화**([J] `report_only.R-b`)
  - $I_{px}$ = 0.2276, CI [0.1515, 0.3038](61위치·35군집, 90 px가 있는 위치)
  - $R'_{90}-R_{90}$ = −0.0537, CI [−0.0926, −0.0149](61위치)
  - $R'_{47}-R_{47}$ = −0.2529, CI [−0.3130, −0.1929](98위치)
  - 해석(§2-6 규칙의 세 경우를 적용): $I_{px}$ > 0(CI 하한 > 0)은 "절대 px 행 사전" 쪽이다. 그러나 90 px 띠가 범위 안(실제 중앙 63.0 px)으로 들어와도 오르지 않고 조금 내려갔다. 두 차가 모두 음수라 "축척 자체가 해로운" 몫도 함께 있다. 종합하면 "절대 px 행 사전 몫과 축척 손해 몫이 함께 있음(추정)"이다. 판정 가족 밖이다.
  - 실제 행(중앙, px): e3x_p42 28 → 19.62, 47 → 32.98, 68 → 47.66, 90 → 62.99. r0은 28.06 / 47.02 / 67.99 / 90.00이다([J] `R-b.realized_rows`).
  - `compression`: e3x_p42 3,550기록 모두 1.0이다. r0은 28,280기록 중앙 1.0·최대 1.178이다([RR] `R-b_compression`, 기록 csv에서 다시 셈). 4.2/6 = 0.70은 확인 축척 범위(0.74) 밖이다(공개 8).
- **R-c 질감**(질감 − 평탄, 긴 팔·47·68 px·12·18 dB, 98위치·44군집. 점추정·$coh$·$c_\text{lat}$ 중앙은 [RR]에서 다시 셌다. CI는 [J] `report_only.R-c`)
  - (i) 진폭 질감: −0.03963, CI [−0.04949, −0.02977]. CI가 ±δ 안이므로 "**AGC 뒤 실효 0.5–0.9 dB의 진폭 질감은 무관**(추정)"이다. 단, CI가 0을 포함하지 않고 크기(약 −4 %p)가 정규화 부수 효과 예상(−0.7 ~ −1.4 %p, 공개 6)보다 크다. 하한이 −δ와 0.0005 차다(§10 해석).
  - (ii) σ_δ 0.1 P: +0.01562, CI [0.00428, 0.02696]. 미리 고정한 해석 문장이 없는 갈래다. 실현은 주로 굽거나 기운 띠이고, 정답 행은 흔들리지 않은 열 기준이다(공개 26).
  - (iii) σ_δ 0.2 P: −0.00146, CI [−0.02594, 0.02302]. CI가 0을 포함하므로 "**시간 질감 효과는 정하지 못함**(추정)". 실현은 주로 굽거나 기운 띠이고, 정답 행은 흔들리지 않은 열 기준이다(공개 26).
  - 측정 $coh$ 중앙: 평탄 0.856, 진폭 0.806, 0.1 P 0.480, 0.2 P 0.140. $c_\text{lat}$ 중앙(dB): 4.131, 3.919, 4.528, 4.827. $coh$는 대비와 섞여 있으므로 $c_\text{lat}$과 함께 읽는다.
- **R-d 최고 점수와 주입 전 기준값**(놓친 단위의 fold별 값. 판정 도구는 단위 × fold로 센다. [J] `report_only.R-d`). 기준값은 같은 폴더 차수 `e3x_retry/uninjected001`(종료 0, `status` ok)에서 왔다. "기준값 없음"이 아니다.
  - 쌍곡선(1,669 단위·fold): `no_box` 0.313, ≤ 0.01 0.200, (0.01, 0.1] 0.320, (0.1, 0.25] 0.167, (0.25, 1] 0.0006. `no_box` + ≤ 0.01 = 0.513 > 0.5 → "**못 봤다**(추정)". 같은 자리 주입 전: `no_box` 0.942, ≤ 0.01 0.041, (0.01, 0.1] 0.013, (0.1, 0.25] 0.005, (0.25, 1] 0. 주입 뒤 − 주입 전 평균 +0.0410(`no_box`는 0으로 둠).
  - 띠(9,486 단위·fold): 0.390 / 0.215 / 0.253 / 0.142 / 0.0001 → 0.605 > 0.5 → "**못 봤다**(추정)". 주입 전: 0.752 / 0.149 / 0.084 / 0.015 / 0. 주입 뒤 − 주입 전 +0.0293.
  - 미주입 병합 전 > 0.25 박스와 R2·R3 미주입 박스의 일치: 같음 3,552, E3x만 19, R2·R3만 19(배치 구성이 달라 비트 일치는 기대하지 않음, [SU] `agreement_with_r2_r3`).
- **R-e**([J] `report_only.R-e`)
  - 6 dB: $M^\ast$ 위치 평균 재현율 0.3732(`recall_6db_mstar_position_mean`).
  - 배경별 효과: §4 각 표. fold 방향 일치: HH1 4, HH3 4, HB1 5, HB1b 5, HB2 5(5 fold 중).
  - PP S3: S3에서 결정 라벨이 달라진 가설 0개(`s3_label_changes`). S3 결정은 S0와 같다(HH1 `negligible`, HH2 `inconclusive`, HH3·HB1·HB2 `confirmed`, HB1b `inconclusive`). S3 HH1 구간 [−0.00143, 0.02018], HH2 [0.49919, 0.73657], HB1b [−0.00960, 0.20477] (`s3_estimates`).
  - HH1 전 단위($M^\ast$ 전부, 110위치·38군집): 0.00740, CI [0.00190, 0.01289].
  - HH1 조작 층 12 dB 단위만(33위치): 0.00500, CI [−0.00808, 0.01808]. 18 dB 단위만(80위치): 0.01125, CI [0.00179, 0.02071].
  - 경계 단위($\lvert r_\text{edge}(p)+s\rvert$ < 1.5 dB) 뺀 HH1: 0.00875, CI [0.00114, 0.01636](80위치).
  - 음성 대조 `period_inj` 불일치 단위 6. 개구 끝 열 ±1 P 안 conf > 0.25 박스 수: 켬 1,924, 끔 2,006.
  - 이동량 민감도(보정 뒤 $n_2$)

    | 가설 | ≤ 4 P | > 4 P | ≤ 8 P | > 8 P |
    |---|---|---|---|---|
    | HB1 | `not_evaluable`(2위치·1군집) | 0.2416 [0.1738, 0.3093](96) | 0.2109 [0.1466, 0.2751](39) | 0.2604 [0.1583, 0.3625](59) |
    | HB1b | 위치 없음 | 0.0976 [0.0222, 0.1730](61) | 0.0515 [−0.0340, 0.1370](20) | 0.1194 [0.0093, 0.2295](41) |
    | HB2 | `not_evaluable`(2·1) | 0.2400 [0.1852, 0.2949](96) | 0.1891 [0.1387, 0.2395](39) | 0.2720 [0.1907, 0.3534](59) |

  - R3 공유 활동을 뺀 4TU 부분집합(`r3_unshared_4tu`): HH1 0.01125 [−0.00093, 0.02343](37위치), HH3 0.1264 [0.0648, 0.1880](40), HB1 0.2913 [0.2024, 0.3802](53), HB1b 0.0158 [−0.0565, 0.0880](33), HB2 0.2644 [0.1896, 0.3393](53).
  - 생존 진폭 비 중앙: 긴 팔 12 dB 0.88–0.91, 18 dB 0.94–0.95, 24 dB 0.97. 짧은 팔 12 dB 0.957, 18 dB 0.976, 24 dB 0.987(`survival_ratio`).
  - 구간·모델별 이동량 분포: §3 표와 `shift_P_by_row_model`.
- **R-f 모델 선택표**([J] `report_only.R-f`): `informative` M01(7위치)·M05·M10·M14·M17(7위치). 나머지 13모델은 `variable`이다. `undetectable`은 없다(M15도 모든 조건에서 0은 아님). 이후 게이트가 이 표를 쓰면 "4TU 새 배경을 포함한 주입으로 정한 선택"이라고 공개한다(§3-5).
- **R-g 통계 보조**(판정과 같은 수준의 백분위 구간, [J] `report_only.R-g`): HH1 [−0.00040, 0.02138], HH2 [0.5789, 0.8200], HH3 [0.03639, 0.14593], HB1 [0.15572, 0.30942], HB1b [0.01305, 0.19143], HB2 [0.17112, 0.30253]. $\nu$·$v_g$·$G_{\text{eff},g}$는 §4 표와 같다. HH2 $\omega^2_p$·$M^\ast$만 $\eta^2_p$는 §4-3에 있다. 라벨·문구는 바뀌지 않는다. HB1b는 백분위 구간이 0을 넘지만 판정 구간이 아니다.

## 7. [ZD] 키, 주기 민감도, R-b 축척 (§6 7)

- [ZD] 키([SI] `zd_summary` = [J] `report_only.R-e.zd_summary`, 영상이 있는 실현 5,816개)
  - `n_zero_changed` 0: 라벨·실패 변화 또는 $\lvert\Delta k_z\rvert$ > 0.1 P.
  - `n_zero_changed_exact` 5,816: 이전 `!=` 정의다. 분수 샘플 영점이 띠 진폭에 선형인 누설로 아주 조금 움직인 것이다([ZD] §3·§5).
  - `n_period_changed` 230(> 5 %), `n_period_changed_exact` 2,775, `n_period_failed` 0.
- 주기가 5 % 넘게 바뀐 단위를 뺀 민감도(`period_change_units_excluded`)

  | 가설 | 값 | 95 % CI | 위치 |
  |---|---|---|---|
  | HH1 | 0.01082 | [0.00170, 0.01995] | 68 |
  | HH3 | 0.09039 | [0.04674, 0.13405] | 79 |
  | HB1 | 0.25386 | [0.16258, 0.34515] | 64 |
  | HB1b | 0.10178 | [0.02264, 0.18093] | 58 |
  | HB2 | 0.24278 | [0.18836, 0.29719] | 92 |

  95 % CI이고 다중 비교 보정 없음이다.
- R-b 레시피별 `compression`과 실제 행: §6 R-b.

## 8. plan 수 기반 MDE (§6 8, 보고 전용)

| 항목 | 값 |
|---|---|
| 보고 | `docs/processing/m33_e3x_mde_eval.md`(커밋 `d720fe369322cfe2e63b0ef7eaa34ce338c59647`, 파일 sha256 `7a7aaf0656226d0660b1fac04753b0c70946f202a20cb09a5cee2a30f92d9734`) |
| 계산 산출물 | `analysis/processing/m33_e3x_mde_eval.json` sha256 `2cc2b93363bf9afd27972f415bb3b6ce57d54ac5e3b869658b83eeb70a8c1138`, `m33_e3x_mde_eval.csv` sha256 `13ba5532c2e2f81b50b54914584bcd7b1779d042c44a7ea23b1b9c2b1bc57ad7`([MDE] §1에서 옮김) |
| 시각 근거 | 계산 2026-11-20 13:24–13:57 KST(json `computed_at_utc` 2026-11-20T04:57:43Z). 봉인 2 커밋(13:18:16) 뒤이고, 보고 커밋 d720fe3(14:14:56)은 injected 1회차 실행 전이다. 재시도 결과는 2026-11-26 13:48에 생겼다. 결과 개봉 전에 계산했다 |
| MDE80(Satterthwaite, 주 시나리오) | HH1 0.071, HH1 음성 대조 0.028, HH3 0.066, HB1 0.048, HB1b 0.056, HB2 0.039. HH2: 참 $\eta^2_p$ ≈ 0.649에서 검정력 80 %. 선형 5개 결합 거짓 주장 0.051 ± 0.0035 |

[MDE] §5 공개 1·2(사용자 결정 Q1 (A) "규칙 유지, 판정서에 공개"에 따라 여기에 옮김)
1. **HH1·HH3 구간의 과소 포괄**
   - plan 구조에서 HH1·HH3 Satterthwaite 구간의 가설당 양측 거짓 주장은 0.015–0.019다(명목 0.0083의 약 2배).
   - 원인은 모델 같은 가중의 모델 내 평균(모델당 약 8위치, 약 30군집)에서 부트스트랩 분산이 작게 잡히는 것이다(SE 비 0.885, 위치가 독립이어도 0.92). Kish 자유도로는 보정되지 않는다.
   - 띠 가설(배경 같은 가중)은 명목 안이다(SE 비 0.96–1.0).
   - 선형 5가설 결합 거짓 주장은 0.051 ± 0.0035로 목표 0.05와 같은 수준이다(군집×팔 SD 0–0.4에서 0.034–0.059, B 20,000 점검 0.041 ± 0.007). 규칙은 바꾸지 않았다.
2. **HH1 {18}만 비중**
   - 실제 $M^\ast$에서 빠질 가능성이 큰 모델은 M15·M16·M09·M04(단계 C 0.00–0.07)이고, 그중 M16·M04는 {12, 18} 모델이다.
   - 그래서 HH1 조작 층의 {18}만 비중은 $M^\ast$ 뒤 약 60–67 %($\lvert M^\ast_e\rvert$ 9–11)로 오를 수 있다.
   - §3-7 (iv)의 "$M^\ast$에서 빠져도 약 53 %" 추정은 M09·M15·M18이 빠지는 경우다.
   - 판정서에는 실제 $M^\ast$로 센 비중을 적는다. → **실제 $M^\ast$로 센 비중은 47/80 = 58.75 %다**(§4-1). 실제로 빠진 모델은 M04·M15·M16·M18이고, M09는 $M^\ast$에 남았다.

- 실제와의 비교(보고 전용, 해석): HH1 실제 구간 반폭은 0.0108로 MDE80 0.071의 약 1/6이다. 생성 모형은 켬·끔 팔이 독립이거나 상관 0.5라고 가정했다. 실제로는 80위치 중 61위치에서 두 팔 적중이 같았다($D_p$ = 0). 과소 포괄(SE 비 0.885)을 거꾸로 보정해 반폭을 1/0.885배로 넓혀도 약 [−0.003, 0.022]이고, ±δ(0.05) 안이다. 그러므로 HH1 `negligible`은 공개 1의 과소 포괄에 민감하지 않다(추정, 보고 전용, 규칙 밖 점검). HH3은 반폭 0.0547이고 하한 0.0365다. 같은 1/0.885배 보정으로도 하한은 약 0.029로 0보다 크다(추정).

## 9. 공개 사항 (§6 9)

### 9-1. 규칙 §4 공개 사항 1–28 (판정서에 그대로 옮김)

  1. 설계자(gpr-processing-expert)는 R3 결과(② 검토)와 Y0 E0–E4 결과를 보고 가설을 만들었다. 가설은 그 결과에서 나왔고, 이 실험은 위치·행·길이를 새로 정한 개입(짝 비교)으로 확인한다.
  2. 구간 경계 37·57·79 px와 길이 12.4 P는 AIHub 라벨 통계에서 왔고 R3 결과를 보고 고르지 않았다([R2R] §5 공개). 목표 행 28·47·68·90 px, 길이 13·6 P, 여유 δ 5 %p, 질감 값(σ_g 2.5 dB, σ_δ 0.1·0.2 P, $\ell_c$ 3 P), 이동 한도 12 P(v0.2의 8 P를 통계 검토 뒤 v0.3에서 바꿈, 결과 전), 위 끝 단언 1 샘플과 재측정 보정 1회(v0.4, 결과 전), 조작 층 문턱(0 dB, 음성 대조 −6 dB), 음성 대조 점검 문턱 0.05·0.02, 바닥 조건 0.10, 음성 대조 최소 수(10위치·5군집)는 이 문서에서 정한 값이다(가정, 결과 전).
  3. 띠 세로 이동은 꼭짓점 열의 순수 평행 이동이다. 깊이 증가에 따른 국소 SNR 감소와 주파수 하강(일정 Q 추정 0.26·Δt/(P·Q), Q 30에서 28 → 90 px 약 9 %)은 넣지 않았다. HB1·HB1b는 같은 파형·같은 국소 SNR에서의 세로 행 효과다. 2.5D 가중은 원래 깊이 기준이다(정답 상자 안 오차 ≤ 0.5 dB, 최악 1.1 dB). 관–지표 다중 반사는 2Δt가 아니라 Δt만 이동한다(크기 ≤ −21 dB). 끝 테이퍼는 원래 깊이 값으로 고정했다. 실제 더 깊은 띠는 끝 전이가 $\sqrt{d'/d}$배 길다([EMV] 6절 1–3).
  4. 이동 한도 12 P는 설계 상수다(물리 한도 아님). 로컬 R3 plan 표(저장소 밖, 판정 완료 자료) 재집계로 8 P였다면 HB1b 위치가 약 17개로 최소 20에 못 미쳤을 것이라 넓혔다. 12 P에서도 원래 위 끝이 아주 얕은 모델과 90 px가 기록 끝에 걸리는 측선은 HB1b에서 빠지므로 HB1b 모델 구성은 HB1과 완전히 같지 않다(모델 구성 공개). 12 P에서는 2.5D 가중 오차 최악 사례(원래 꼭짓점 2.7 P → 90 px, 정답 상자 아래 끝 −1.1 dB)가 적격이 된다. 이동량 ≤ 8 P 대 > 8 P 민감도를 보고 전용으로 낸다.
  5. HH1은 개구 끝 처리 효과이고 날개 길이 효과가 아니다. 조작 강도가 모델마다 −37 ~ +1 dB로 다르고, 단계 C 고재현 모델 네 개(M01·M12·M14·M17)는 조작이 보이지 않는다.
  6. R-c 진폭 질감은 r0 AGC 뒤 실효 0.5–0.9 dB다. $coh$는 대비와 섞여 있다. 평탄 열 평균 제곱 정규화로 진폭 질감 갈래에 −0.7 ~ −1.4 %p의 진폭 몫이 섞인다.
  7. IDS 배경은 가설을 만든 R3와 같은 53측선이다. 4TU 배경은 새 측선이지만 최대 24측선이 R3와 활동을 공유한다(plan 보고에 실제 수). 결론은 이 배경 집합에 대한 것이다.
  8. $P^*$ 4.2(0.70배)는 확인 축척 범위(0.74) 밖이다(R-b).
  9. 설계 집계에 M3-2 단계 C와 R3의 로컬 plan 표(저장소 밖, 판정 완료 자료, 검출 출력 아님, 입력 sha256은 feasibility json `inputs`), R2 측선 목록을 읽었다. 통계 검토(stats-design-expert)는 분산 근거로 판정이 끝난 공개 자료 `runs/eval/m32_recipe_v2/injected/r0/fold*/inj_matches.csv`(단계 C R1, 2026-11-15 판정 완료)와 `analysis/processing/m33_r3_recall_diag_units.csv`(R3, P4 공개)를 읽었다([STV] 오염 보고). E3x 결과는 누구도 열지 않았다(v0.3 작성 때 `runs/eval/m33/e3x*` 폴더 없음. v0.4 작성 때 `runs/eval/m33/e3x/`에는 검출 무관 `plan/`만 있었고 §5-3 블라인드 경로는 하나도 없었다. v0.4 진단 [PDD]는 plan 표·배경 측선·응답 라이브러리만 읽었다. v0.5 작성 때의 열람 범위는 공개 23·25).
  10. 판정 구간은 Satterthwaite t 구간이다(v3-1). 시뮬레이션에서 백분위 층화 군집 부트스트랩 구간(99.17 %)은 IDS 유효 존 수가 약 6.5로 작아 선형 5가설 결합 거짓 주장이 약 0.094(목표 0.05)였고, Satterthwaite는 0.041(0.036–0.048)이었다. 대가로 MDE가 약 10–33 % 크다. BCa는 과소 포괄이라 쓰지 않았다. 수준은 정확히 $1-0.05/6$이고 `negligible`도 같은 구간을 쓴다(CI 교정 여유 때문에 유지, [STV] 2).
  11. $|M^\ast_e|$, HH1·HH3·음성 대조 대상 위치 수, HH3 30 dB 거부 비율의 분모는 $M^\ast$에 의존하므로 결과 의존이다. HH3은 선택 통계에 한 팔이 들어가 원리상 편향될 수 있으나 시뮬레이션 편향 +0.0009로 무시할 만하다.
  12. 검정력(시뮬레이션 추정, [STV]): HH1 `negligible`은 사실상 닿기 어렵다(Δ = 0에서도 확률 0.08–0.17). HH1 결과는 사실상 `confirmed_*` 또는 `inconclusive`다. HH2는 참값이 약 0.65를 넘어야 확인될 가능성이 크다. plan 수 기반 MDE는 판정서 §6 8항에 적는다.
  13. 보고 전용 95 % CI에는 "다중 비교 보정 없음"을 붙이고 해석 문구는 '추정'이다. 미리 고정한 해석이 있는 항목이 10개 안팎이라 모두 귀무여도 방향 문장이 약 0.5개 거짓으로 나올 수 있다.
  14. HH1 추정 대상은 끝 칸이 σ 위로 보이는 단위($r_\text{edge}(p)+s\ge 0$ dB, $r_\text{edge}$는 켬·끔 차 장 기준)의 모델 같은 가중 평균이다. 조작 층 위치의 절반 남짓(plan 1회차 위치를 v0.4 정의로 다시 잰 값 53 %, $M^\ast$ 조건 전)은 18 dB 단위만 들어가므로 이 평균은 18 dB 쪽에 무게가 실린 혼합이며, 특정 SNR의 효과가 아니다. 실제 구성은 plan 보고의 모델별 $S_p^\text{edge}$로, SNR별 값은 보고 전용으로 낸다(§3-7 (iv)).
  15. 경계 근처 단위(plan 1회차 위치를 v0.4 정의로 다시 잰 값: 조작 층 단위 154개 중 12개, 주로 M06의 12 dB 단위와 M15의 18 dB 단위)는 측선에 따라 조작 층에 들어가거나 빠진다. 설계 때 경계로 본 M02·M07·M08은 −13.7 ~ −10.9 dB로 대부분 {18}이다. 배정은 측선 격자에서 잰 위치별 값으로 결과 전에 정해졌고, 경계 근처 단위의 조작은 약하다(§3-7 (v)).
  16. HH2의 모델 몫에는 어느 배경에서도 검출되지 않는 모델과 검출되는 모델의 차, 그리고 모델 고유 깊이의 효과가 들어 있다. 위치 몫에는 모델 × 배경 상호작용(A1 모델은 4TU에만 있음, Type II)과 위치 × 조건 변동이 들어 있다. $M^\ast$만의 $\eta^2_p$와 $\omega^2_p$를 보고 전용으로 함께 낸다(§3-7 (iii)).
  17. 음성 대조는 위치별 $r_\text{edge}(p)+18<-6$ dB이고 모델 ∈ $M^\ast$로 제한했다. $|\hat\Delta_\text{neg}|\ge 0.05$(구간이 0 제외)일 때만 HH1을 `not_evaluable`로 두고, 0.02–0.05는 문구 표지만 붙인다. 작은 음성 대조 효과는 끔 장 구성 결함보다 σ 아래 구조의 결맞은 검출이나 측선 전체 r0 통계 차일 가능성이 크기 때문이다(§3-7 (i)).
  18. `negligible`은 대상 위치의 두 팔 평균이 0.10 이상일 때만 낸다. 못 채우면 `inconclusive`(사유 `floor`)다.
  19. **plan 1회차와 v0.4 개정(결과 전·검출 무관)**: 봉인 1(v0.3, 72b0140) 뒤 구현으로 돌린 plan 1회차(run_id `e3689ada-4071-4f2a-9d5f-d9defc68b64c`, 2026-11-20, CPU, 검출기 없음)가 종료 1 `e3x_defect`로 끝났다. 원인은 띠 세로 이동의 `shift_mismatch` 7건(목표와 차 1.03–1.48 샘플, M01 4·M08 2·M09 1, 원래 위 끝이 기록 위 끝 0.8–2.5 P)이다. 진단([PDD]): 장의 평행 이동은 정확했고(상대 1.8e-14), 원래 패치의 위 끝 재측정만 단계 (1)–(3)의 기록 시작 경계 처리와 합성 창 잘림 때문에 1 샘플 달랐다. v0.4는 재측정 보정 1회($n_2$)와 팔 사이 평행 이동 단언을 넣고 1 샘플 단언을 유지했다(§2-4, 사용자 결정 v4-1). 이 개정으로 띠 선택이 바뀔 수 있어 plan을 다시 돌린다(§5-4). plan 1회차 표는 봉인 대상이 아니다.
  20. **v0.3 $r_\text{edge}$ 식 결함과 v0.4 정정**: v0.3 식(개구 테이퍼 열에서 비테이퍼 포락선 최대)은 가중 ≈ 1인 안쪽 테이퍼 열(±0.9 m)의 날개를 재서 조작 강도를 +0.5 ~ +9.6 dB(중앙 +3.9) 과대로 쟀다. 구현은 문구를 그대로 따랐다. v0.4는 켬·끔 차 장 $f_\text{unt}(1-w)$의 포락선 최대(모든 행·열)로 정의를 고쳤다(§2-3, 사용자 결정 v4-2). plan 1회차 위치 144개(쌍곡선 위치 선택은 $r_\text{edge}$와 무관)를 두 정의로 잰 층 수(M* 조건 전): 조작 층 112 → 105위치(군집 40 → 37), {12, 18}/{18} 109/3 → 49/56, 회색 띠 12 → 7, 음성 대조 20 → 32위치(군집 16 → 22, 모델 M12·M14·M17 일부 → M01·M12·M14·M17). v0.4 값은 [EMV] 4절 참고표와 [STV] 시뮬레이션의 층 가정(음성 대조 약 32위치·24군집)에 맞는다. 통계 재검토는 봉인 전에 하지 않고 봉인 2 뒤 `[모드] 평가`(새 plan 수, §6 8항)에서 반영한다. 판정 규칙·문턱(0 dB, −6 dB, §3-2–§3-4)은 바꾸지 않았다.
  21. v0.4의 문구 명시(규칙 의미 변경 없음): 짧은 팔 중심 −0.5 트레이스(열 수 홀짝이 다른 위치, plan 1회차 98위치 중 80), T 조건의 (측선, 모델) 단위 적용(plan 1회차 해당 0건), 병합 출력 보존 시험(A-5 13, cv 권고).
  22. **injected 1회차 실패와 v0.5 개정(결과 미열람)**
      - 실행: 봉인 2(`729d19ced8674039efac0cfe906fcb41f45c31e3`) 뒤 본 실행(HEAD `d720fe369322cfe2e63b0ef7eaa34ce338c59647`, injected run_id `73fe0456-29e6-438f-bb56-a6c915158f2d`(UUID 36자, 1회차 `summary.json` `run_identity.run_id`를 메인 세션이 정체성 키만 읽어 그대로 옮김))의 injected 단계가 종료 1 `e3x_defect`(3,119 s)로 끝났다.
      - 실패 단언: 질감 단언 `texture_check`의 하위 필드 `jitter_energy`(σ_δ 0.1·0.2 둘 다) 하나였다.
      - 위치: 4TU `010.18/Path2` b00·b01, 4TU `011.16/Profile_6453` b00, IDS `pipe/1030_181230AB/LID10001` b00·b01.
      - 원인: 짝수 길이 꼭짓점 열의 나이퀴스트 빈 에너지 손실($r_N\sin^2(\pi\delta)$, $r_N\le1.03\times10^{-8}$)이다. 단언의 전제("0.8 나이퀴스트 저역통과가 걸린 열이라 정확함")가 틀렸다. 이 전제는 [EMV] 5절 문장에서 왔고, 설계자가 v0.2 §2-7에 검증 없이 옮겼다. EM 검토도 그 문장에 근거 계산을 붙이지 않았다([EMC]).
      - 1회차 폴더 `runs/eval/m33/e3x/injected/`와 1회차 `runs/eval/m33/e3x/uninjected001/`(있으면)은 열지 않고 남겼다. 재시도는 v0.5(봉인 2 r1)로 `runs/eval/m33/e3x_retry/`에서 했다.
      - 판정 가설·규칙·문턱·시드·실현 비트·plan 표는 바꾸지 않았다.
  23. **원인 확인 열람 범위**
      - 무엇을 열었나: python-expert가 원인을 확인하려고 1회차 injected 폴더의 `summary.json`·`parts\*.json`에서 `defects` 키의 0이 아닌 키 이름과 `line_id`만 스크립트로 출력했다. 재현율·적중·덮임·점수·대비·영점 키는 출력하지 않았다.
      - 공개하는 이유: 이 열람은 §5-3이 메인 세션에 허용한 범위(종료 코드·마지막 줄·`run_identity`·`judgeable`)를 넘는다. `defects` 이름 자체는 §5-4가 "단언 이름 공개"로 정한 정보다.
      - 결과 열람으로 치지 않는 근거: 0이 아닌 키가 `texture_check`뿐이었다는 출력은 검출 의존 단언(`hit_without_operating_box` 등)이 0이었다는 것도 알려 준다. 그러나 이 단언들은 구현이 맞으면 늘 0인 값이라 판정 정보가 아니다. 판단은 gpr-processing-expert가 했고 verif 재확인 대상이다.
      - 증거: 원본 스크립트(`defect_names.py`)와 stdout 파일은 남아 있지 않다. 스크립트는 scratchpad 정리 때(2026-11-20 15:25:31) 지워졌고, stdout은 도구 출력으로만 받았다. 다시 돌리지 않았다. 열람은 두 건이다. 하나는 결과 폴더 파일 이름 목록(15:24:03–04)이고, 다른 하나는 `defect_names.py` 실행(15:24:15 명령, 15:24:29 결과)이다. 두 건 모두 python-expert 서브에이전트 transcript에서 전사했다. 전사본은 저장소 밖 `C:\Users\jinbeom\.claude\runners\gpr_detection-m33_e3x\diag_texture_20261120\h6_disclosure23\`에 있다.
        - `reconstructed_from_transcript.md`: 스크립트 본문·명령·stdout·시각, "재실행 아님·transcript 전사". sha256 `2010ea043753a7d7e83be66b3e2389105300f3e8d9d29d2664bc500af5877fae`
        - `transcript_excerpt_lines_50_52_54_55_59_60.jsonl`: 원 transcript 해당 6행 원문 사본. sha256 `fed36bbadff4040e290a63b9c1c20d6761c07179ad5aa0119254f2d0b3cb5905`
  24. **precheck 표본 한계**
      - v0.4 precheck는 질감 단언을 precheck 2측선에서만 돌렸다. 그 두 측선의 $r_N$은 e-17–e-18이라 결함을 잡을 수 없었다(본 표본에서 실패한 위치는 e-8–e-9).
      - 실행 전 감사도 precheck에 음성 대조·회색 층 위치가 없다고 기록했다.
      - v0.5는 CPU 구현 단언 다섯을 전 측선 precheck 10으로 넓혔다. 검출이 필요한 단언은 여전히 precheck 2측선(6·9번)과 본 실행에서만 확인한다.
  25. **진단 CPU 재계산(검출 없음)**
      - 무엇을 돌렸나: 원인 확인과 수정안 비교를 위해 다음을 돌렸다.
        - (i) python-expert `texture_scan.py`(155측선 질감 단언 재계산과 (a) 모의, 226.6 s, 저장소 밖 `C:\Users\jinbeom\.claude\runners\gpr_detection-m33_e3x\diag_texture_20261120\`)
        - (ii) gpr-processing-expert `wrap_scan.py`·`w_scan.py`([TDR], `analysis/processing/m33_e3x_texture_defect/`)
        - (iii) electromagnetic-simulation-expert `edge_scan.py`·`leak_attrib.py`·`norm_mc.py`([EMC] §5, 스크립트는 지웠고 sha만 남음)
      - 읽은 것: 모두 plan 표·precheck plan 표·배경 측선·응답 라이브러리만 읽었다. 영상·추론·박스는 만들지 않았다. 출력은 합성 장의 에너지 지수, 지연 크기, 절단 크기다.
      - 문턱: v0.5의 두 문턱(항등식 1e-12, $r_N\le10^{-6}$)은 이 분포를 본 뒤 정했다. 판정 문턱이 아니라 구현 단언이다. $r_N$ 한계는 큰 절단·구현 결함만 잡는다(§2-7).
  26. **시간 흔들림 실현 성질**([EMC] §4-3)
      - 굽음·기울기: 평탄 구간이 핵 표준편차의 약 3.6배(3.1–3.9, 과정 상관 SD의 약 2.5배)뿐이다. 그래서 위치마다 δ(x)는 주로 완만한 굽음·기울기다(상자 열 δ 분산의 중앙 약 56 %가 1차 추세). 이 성질은 정규화 방식과 무관하다.
      - 정규화의 효과: $w$를 평탄 열 위에서 정규화하므로 모든 위치에서 평탄 열 δ의 SD가 정확히 σ_δ이고, 평탄 밖으로 외삽된 |δ|가 커진다. σ_δ 0.2에서 상자 열 최대는 중앙 0.39 P·최대 1.17 P, 테이퍼 열 최대는 1.69 P다.
      - [EMV] 예측과의 관계: [EMV] 5절 coh 예측 계산의 정규화 기준은 기록에 없다(띠 안 표본 질감은 명목의 약 0.76배). EM 확인 몬테카를로(101위치 기하)에서 띠 전체(L+T) 정규화는 0.86배였다. 그래서 구현의 띠 안 실효 흔들림은 예측 계산보다 약 1.15–1.3배 크다(추정).
      - coh: 예측(0.2 P에서 0.21–0.23)보다 낮을 수 있다(잡음 없는 대리 계산에서 약 0.7배, [EMC] §3). coh는 고정 상자의 행 평균으로 재므로 기울기도 무작위 흔들림만큼 coh를 낮춘다. 그러나 검출기에게는 기운 띠도 일관된 띠다. 따라서 이 갈래는 "거친 띠"보다 "굽거나 기운 띠"의 효과에 가깝다.
      - 덮임 판정 몫: 정답 세로 범위는 흔들리지 않은 열 기준이다. 그래서 짝 차에는 "정답 행에서 벗어난 띠"의 덮임 판정 몫(시간 겹침 ≥ 정답 높이 50 %, 상자 열 최대 1.17 P에서 겹침 약 51 %)이 섞일 수 있다.
  27. **합성 창 절단(기존 성질, 판정 무관)**([EMC] §4-6)
      - 응답 장은 합성 창 [0, `window_ns`] 밖을 0으로 둔다. 기록 안에는 두 가지가 있다.
        - (i) 창 시작 계단: 행에 따라 34–100 %의 팔에서 기록 안에 있고, 최대 1.8 % A이며, 정답 위 끝보다 ≥ 2.08 P 위다.
        - (ii) 창 끝의 1 P 감쇠: 정답 아래 끝보다 ≥ 3.37 P 아래다. 감쇠 전 꼬리는 `09.2/Path7` 두 위치에서 최대 0.143 A이고 나머지는 −24 dB 아래다.
      - 둘 다 정답 상자 밖이라 재현율·덮임 지표에 들어가지 않는다.
      - 다만 창 끝이 기록 안에 있는 비율은 행에 따라 다르다(28 px 83/101, 47 px 63/101, 68 px 21/101). 그래서 HB1·HB1b의 "같은 파형"은 정답 구간에 대한 말이다(§2-4).
      - R3·단계 C에도 같은 성질이 있었다.
  28. **재시도 출력 루트와 v0.4 잠재 결함(봉인 2 r1 전, 결과 무관)**
      - v0.5 재시도 명령은 `--out-root runs/eval/m33/e3x_retry`를 준다(§5-1). v0.5 러너는 1회차 루트를 detection 단계 출력으로 받으면 거부한다.
      - v0.4 러너는 plan 표를 `--out-root` 아래(`<out-root>/plan/`)에서 찾았다. 그래서 v0.4로 `--out-root e3x_retry`를 주었다면 `e3x_retry/plan`이 없어 종료 2로 멈췄을 것이다. v0.4의 §5-4 재시도 경로(`e3x_retry/`)가 코드로는 실행될 수 없었던 잠재 결함이다.
      - v0.5 구현은 plan을 봉인 경로 `SEALED_PLAN_DIR`(`runs/eval/m33/e3x/plan/`, 봉인 2 sha)에서 읽도록 고쳤다. python-expert 구현 보고에서 드러났고 결과와 무관하다.

### 9-2. 결과를 본 뒤 읽은 자료, 정한 해석, 개정

- **결과를 본 것으로 치는 시점**: 판정서 작성자가 2026-11-26 13:52 KST(의존성 점검 직후)에 [J]를 처음 열었다. 그 전에 규칙 문서 v0.5 전문을 읽었다.
- **결과를 본 뒤 읽은 자료**
  - 재시도 결과: [SI], [SU], 두 `stage_settings.json`·`progress.json`, `injected/hyperbola_records.csv`·`band_records.csv`·`rejections.csv`, `uninjected001/uninjected001_records.csv`(카운터 열 합), `judgment/units_*.csv`.
  - 봉인·plan: [SR], `analysis/detection/m33_e3x_seal2.json`, [PR], `runs/eval/m33/e3x/plan/` 표 5개.
  - precheck r1: [PC], [CA](요약 키와 분포), `cpu_assertions_lines.csv`(머리 3줄), 증거 폴더 SHA256SUMS·`precheck_r1_sha256.txt`(sha 대조).
  - 문서: [MDE], `docs/processing/m33_e3x_prerun_audit_r1.md`, `m33_e3x_prerun_audit.md`·`m33_e3x_texture_defect_review.md`·`m33_r3_zero_change_definition.md`(해당 줄 검색), 메인 세션 인계 메모리(1회차 사실 확인).
  - 코드: `analysis/detection/m33_inputs.py`·`m33_e3x.py`·`m32_recipe_eval.py`(가중치 sha 위치 확인 줄 검색).
  - 러너: [RL]의 로그·요약·steps·ps1(마스크·우선순위·시각).
  - 1회차 블라인드 폴더는 읽지 않았다.
- **결과 뒤 정한 해석**(규칙 문구를 바꾸지 않음)
  1. R-a의 단위: 규칙은 24 − 18 dB 짝 차의 행 범위를 정하지 않았다. 판정 도구는 47·68 px 긴 팔로 냈고, 이 판정서는 그 값을 옮겼다. 네 행으로 넓힌 점추정 0.056도 같은 해석(하한 > 0인지는 구간 미계산)을 줄 것으로 보인다(보고 전용).
  2. 배경별 값의 정의: 규칙은 "배경별 값"의 가중을 정하지 않았다. 판정 도구는 HH1·HH3에서 배경 안 위치 단순 평균을 냈다. 모델 같은 가중 값은 §4-4에 함께 적었다. 판정 라벨과 무관하다.
  3. R-d의 집계 단위: 판정 도구는 "놓친 단위"를 단위 × fold로 셌다. 해석 문장(과반 규칙)은 그 비율에 적용했다.
  4. R-b 해석: §2-6의 세 경우가 섞여 나왔다($I_{px}$ > 0이면서 두 차가 모두 음수). 그래서 "두 몫이 함께 있음(추정)"으로 적었다.
  5. HH1 문구 범위: §3-6 문구 "개구 끝 처리가 보이는 모델에서"에 $M^\ast_e$ 구성과 M04·M16 제외를 덧붙였다(§5, §10-1). 라벨은 그대로다.
  6. §8 끝의 과소 포괄 민감도(1/0.885배)는 규칙 밖 보고 전용 점검이다.
- **개정**: 결과를 본 뒤 규칙 개정은 없다. 판정은 v0.5 원판 규칙으로 냈다.
- **v0.5의 성격**: v0.5는 결과 미열람 개정이다. 다만 본 실행 1회차(injected `e3x_defect`) 뒤의 개정이다(공개 22–25).
- **실행 중 결정과 감사 기록**: §2 "실행 중 사용자 결정" 1–3. precheck r1 마스크(E코어 0x0FFF0000·BelowNormal)와 재시도 마스크(P코어 0xFFFF·Normal). 공개 23의 사후 규정·전사 증거 성격. 실측 2측선의 두 경로 검사 수 대조 없음(verif 재확인 기록 9). 규칙 v0.5 sha 64자리·커밋(머리). 두 HEAD 408df3a(precheck r1)·bc57ef1(재시도).

## 10. 물리·검출 해석 (보고 전용, gpr-processing-expert, 판정 라벨을 바꾸지 않음)

1. **HH1(끝 처리 무관).** 조작 층에서 켬·끔 팔은 80위치 중 61위치에서 fold 평균 적중까지 같았다. 개구 끝 칸이 σ 위로 보이더라도(대부분 18 dB에서 σ의 1–2배), 0.2 m 코사인과 급단절의 차는 꼭짓점 배정 결과를 거의 바꾸지 않는다. 검출기는 꼭짓점 주변 ±수 P의 날개로 쌍곡선을 잡는다. 꼭짓점보다 13–15 dB 낮은 ±1.1 m 끝 칸은 그 판단에 들어가지 않는다고 본다(추정).
   - 범위의 한계: 끝 칸이 꼭짓점과 거의 같은 세기인 M04(+0.6 dB)·M16(−1 dB)은 $M^\ast$ 평균 0.091·0.047로 빠졌다. 그래서 "급단절이 강하게 보이는 쌍곡선"에서의 효과는 이 판정에 들어 있지 않다. 강한 절단 모델로는 M03(−5 ~ −2.7 dB)만 남는다(M03 평균 $D$ +0.019).
   - 보고 전용 전 단위·18 dB 단위 95 % CI는 0을 겨우 배제한다(약 +0.7–1.1 %p). 효과가 있더라도 δ보다 한 자릿수 작다.
2. **HB1(얕은 행 손실).** 같은 파형·같은 국소 SNR에서 28 px 행(정답 위 끝 L1 행)만 덮임이 약 24 %p 낮다. 대비 점검에서 28 px의 $c_\text{lat}$이 오히려 높았으므로(4.31 dB 대 4.20·4.07), 대비 차로 설명되지 않는다.
   - 후보 기제는 둘이다. (a) 학습 라벨 위치 사전(AIHub y1 ≥ 36.9 px). (b) 영점 띠 잔차와 AGC 창(4TU ±3.2 P)의 근접.
   - R-b에서 축척만 바꾸면(주기 단위 기하는 같음) 47 px → 33 px 띠의 덮임이 0.25 떨어지고, 90 px → 63 px는 0.05만 떨어졌다. 주기 단위로 같은 배치에서 화소 행만 얕아졌는데 손실이 컸으므로 (a) 화소 행 사전 쪽을 지지한다(추정, 보고 전용, 축척 손해 몫이 섞임).
   - 이동량 ≤ 8 P와 > 8 P 부분집합 모두 0.21–0.26이라, 이동 크기에 따른 2.5D 가중 오차로 설명되지 않는다.
3. **HB2(길이).** 긴 팔(13 P)이 짧은 팔(6 P)보다 약 24 %p 높다. 중앙값 배경 제거(301 트레이스)는 긴 띠를 더 깎는다. 실제로 생존 진폭 비 중앙이 긴 팔 0.88–0.95, 짧은 팔 0.96–0.99였다. 그런데도 긴 팔이 더 잘 검출됐으므로 HB2는 배경 제거 손실의 역효과가 아니라 길이(가로 지지 폭) 효과다(추정). 끝 전이 비율($T/L$ 0.4–0.5)이 다르다는 점은 규칙 §1-1대로 "같은 끝 전이폭에서 길이" 효과로 읽는다.
4. **HB1b(깊은 행).** 4TU는 0.021, IDS는 0.174로 배경 차가 크다. IDS는 30–40 ns 창이 많아 90 px 행이 기록 끝에 가깝다. 행 효과와 기록 끝 근접(AGC 창 끝, 창 끝 감쇠 꼬리, 공개 27)이 IDS에서 섞였을 가능성이 있다(추정, 이 실험으로는 가르지 않음). 4TU만 보면 깊은 행 손실은 거의 없다.
5. **HH3(30 dB 포화).** 배경이 있어도 30 dB가 12 dB보다 9 %p 낮다. 높은 SNR 주입은 트레이스별 AGC(8/$f_c$ 창)에서 주변 배경을 눌러 영상 문맥을 바꾸고, 99 % 백분위 클립으로 쌍곡선이 평평해진다. 그래서 학습 분포(AIHub 중간 대비)에서 벗어난다고 본다(추정).
   - 30 dB int16 거부로 $M^\ast$ 위치의 17.9 %가 빠졌다. 거부는 측선 진폭 범위와 이득으로 정해지는 검출 무관 선택이지만 모집단을 바꾼다.
   - M14(−0.08)·M08(−0.04)은 반대 방향이다.
6. **R-c 진폭 질감.** 규칙 문장으로는 "무관"이지만, 효과는 −4 %p로 0이 아니고 정규화 몫 예상의 약 3배다. AGC 뒤 실효 0.5–0.9 dB의 가로 진폭 변화가 띠 박스의 가로 연속성(덮임 합집합)을 조금 끊는다고 본다(추정).
   - 시간 흔들림 0.2 P의 $coh$ 0.14는 AIHub 수준(0.28)보다 낮다. [EMC] 예상(0.21–0.23의 약 0.7배 ≈ 0.15)과 맞는다. 그런데도 덮임은 평탄과 같아서(CI ±0.025), 굽거나 기운 띠는 검출기에게 여전히 일관된 띠다(공개 26과 일치).

## 11. 재집계 범위와 근거 파일

| 범위 | 방법 | 파일 |
|---|---|---|
| 기록 csv에서 다시 셈 | 규칙 §3-2–§3-4 문장으로 다시 구현했다(판정 도구 import 없음). 단위 $y_u$ = 5 fold 평균, $Y_{p,a}$, $M^\ast$, 층, HH1·음성 대조·HH3·HB1·HB1b·HB2 추정량과 $F$, HH2 $\eta^2_p$·$\omega^2_p$(배경 + 모델 가법, 가중 최소제곱), 배경별 층화 군집 부트스트랩(`default_rng(20261121)`, B 20,000, 가설 순서 HH1 → 음성 대조 → HH2 → HH3 → HB1 → HB1b → HB2, 배경 순서 4TU → IDS, `rng.integers(0, G, (B, G))`, 군집 id 정렬), Satterthwaite 구간·기본 구간. 그 결과 판정 구간 끝이 [J]와 ≤ 3×10⁻¹⁷로 같았다. HH2 $M^\ast$만 구간은 새로 뽑아서 조금 다르다(규칙은 복제 재사용) | `analysis/processing/m33_e3x_judgment_recheck.py`(sha256 `ddbbed0332e3c6789c58b5dd1be2722987fa7f380d67ca756c73109f8d90725b`) → [RC] (`2620245fa6bbac2c12bbdca0059e286d912ad1f145d774dafbea650f5265c74c`) |
| 기록 csv에서 다시 셈(보고 전용 점값) | R-a·R-c 점추정과 $coh$·$c_\text{lat}$ 중앙, HB1 대비 중앙, `compression` 분포, 배경별 값 두 정의, 위치별 부호 | `analysis/processing/m33_e3x_judgment_recheck_report.py`(`a1d321def7711d78b1925ca80205af4ff0ae29653ab60f8a4bf394fcd5baf79e`) → [RR] (`28b07e1d05bf33110fee28f62490f650e8a4b75928c5900367c870e95906355f`) |
| json에서 옮김 | R-a·R-b·R-c 구간, R-d 비율, R-e 전부, R-f, R-g, S3, [ZD] 키, HH2 보조 구간, 이동량 분포(plan 보고와 같음) | [J], [SI], [SU], [PR] |
| 카운터 | NMS 경고·max_det 타일: 기록 열 `n_nms_timeout_warnings`·`n_tiles_maxdet`의 합이 hyperbola 11,520행, band 31,840행, uninjected 3,475행 모두 0 | 즉석 `-c` 한 줄(출력 없음 파일) |

- 재집계에서 발견한 함정: pandas 기본 읽기는 `cluster_id`(예 `01.10`)를 실수로 바꾼다. 두 스크립트는 문자열 dtype을 강제한다.
- 실행: `conda run -n gpr-dev python -I -B analysis/processing/m33_e3x_judgment_recheck.py`(58 s, BLAS 스레드 1), `…_report.py`(수 초).

## 12. 대조표 (§6, §4, §5-3, §5-4)

**§6 판정서에 적을 것**

| 항목 | 위치 | 상태 |
|---|---|---|
| 1 실행 정체성(HEAD·src tree·blob·입력 sha 64자리, 가중치 sha, 설정 해시, pp_set_id·PPConfig sha, run_id, 시작·끝, 단계별 시간, VRAM 최대, 작업자 수, NMS 경고·max_det) | §1 | 채움. 판정 도구 `finished_at` null은 기록 결함으로 공개 |
| 2 판정 가능성·이력·재시도 차수 | §2 | 채움 |
| 2 (v0.5) 1회차 정체성·실패 단언·위치 5개 | §2 | 채움(시작·끝 시각은 1회차 러너 로그 제어 줄에서 메인이 옮김) |
| 2 1회차 폴더 미열람 확인, 공개 23 범위·증거·sha | §2 | 채움 |
| 2 봉인 2 r1 `supersedes`·plan 재사용 근거 | §2 | 채움 |
| 2 precheck r1·사본 경로·파일별 sha256, 10번 단언별 수 | §2 | 채움 |
| 2 판정 입력 = `e3x_retry/`뿐, 재집계 범위 | §2, §11 | 채움 |
| 3 plan 수(배경별 측선·군집·위치, R3 공유, 모델별, 행·길이별, 거부 사유, 이동량, 보정량·계단, −0.5 위치, 빠진 행, HB1b 구성, 층 수 전·후, $r_\text{edge}$·$S_p^\text{edge}$, M02·M07·M08, $G_g$·$G_\text{eff}$) | §3, §4-1 | 채움 |
| 4 가설별 전 항목, 조작 점검, 재집계 범위 | §4 | 채움 |
| 5 §3-6 문장 변경안 | §5 | 채움 |
| 6 R-a–R-g와 해석 적용 | §6 | 채움 |
| 7 [ZD] 키·주기 민감도·R-b compression | §7 | 채움 |
| 8 plan 수 MDE(경로·sha·MDE80·시각 근거) + [MDE] 공개 1·2 | §8 | 채움 |
| 9 공개 1–28, 결과 뒤 자료·개정, v0.5 성격 | §9 | 채움 |
| 10 대조표 | §12 | 이 표 |

**§4 데이터 범위와 공개 사항**

| 항목 | 이번 실행·판정서 |
|---|---|
| AIHub 열지 않음(공개 상수만) | 판정서 작성자도 열지 않음 |
| 4TU 깊이표 열지 않음, 4TU는 채택 근거 아님 | 열지 않음. 깊이(m)를 내지 않음 |
| IDS `main_400` 53측선만 | [PR] `lines.by_dataset.gz_ids` 53 |
| 모로코·GPR-SD·광저우 MALA 쓰지 않음 | 쓰지 않음 |
| 공개 1–28 | §9-1에 그대로 옮김 |
| `single_background`·`zero_variance_background`·`degenerate_ci` 공개(§3-2·§3-3) | 모두 해당 없음(§4 공통) |
| 음성 대조 약한 점검 문장(§3-3 (d)) | §4-2 |
| HB1b 모델 구성 공개(§2-4) | §3 |
| 결과 의존 분모와 $M^\ast$ 구성(§3-4) | §4-1 |

**§5-3 블라인드**

| 항목 | 확인 |
|---|---|
| 판정값은 파일에만 | 러너 로그 마지막 줄의 키: injected·uninjected001 `status`·`out_dir`·`n_lines`·`n_positions`·`n_realizations`·`n_inferences`·`elapsed_s`, judge `status`·`out_dir`·`n_hypotheses`·`elapsed_s`. 그 밖의 줄은 진행 건수·카운터뿐이다([RL] `logs_P\*.log`) |
| 메인은 판정서 전 종료 코드·마지막 줄만 | 지시문 사실. 판정서 작성자가 처음 열었다 |
| 블라인드 경로·진행 파일 예외 | 1회차 폴더 미열람. `progress.json` 6키 확인 |
| 결과를 본 시점 | §9-2(2026-11-26 13:52 KST, 판정서 작성자) |

**§5-4 실패 처리표**

| 행 | 이번 일 |
|---|---|
| pool·plan 0 / plan 1 `e3x_defect` | plan 1회차 실패와 v0.4(공개 19). 재실행 plan ok가 봉인 2 |
| precheck 0 | 1회차 precheck(v0.4)와 precheck r1 ok. r1 사본·sha는 §2 |
| precheck 1 | 발생하지 않음 |
| uninjected001 2 / 1·2·3 | 발생하지 않음. 실패 시 멈춤 결정은 §2(규칙 `:649`보다 우선, 발동 안 함) |
| injected 1 `e3x_defect`(1회차) | 폴더 보존·미열람, 단언 이름 공개(공개 22), 새 blob → 봉인 2 r1 → verif 재확인 → `e3x_retry/` |
| 재시도 행(0/1/2/3) | 재시도 세 단계 모두 0, 한 번씩. `e3x_retry2` 불필요(종료 3 처리 결정, 발동 안 함) |
| 실패 폴더 열람 | 공개 23 범위만. 재시도와 비교하려고 1회차 폴더를 열거나 해시하지 않았다 |
| injected 3 / injected 0 | 3은 발생하지 않음. 0은 다시 돌리지 않음 |
| judge 0 | 다시 돌리지 않음. 판정서 작성자가 처음 엶 |
| judge 2·3 | 발생하지 않음 |

## 보고 전용 권고 (규칙 밖, 판정에 쓰지 않음)

1. python-expert: `m33_e3x_judge.py`가 `identity.finished_at`을 채우게 한다. 이전 감사부터 이월된 uninjected001 `status`·`judgeable` 확인도 다음 실험 판에서 닫는다.
2. 다음 설계 후보
   - (a) HH1의 빈칸(M04·M16처럼 끝 칸이 꼭짓점과 같은 세기인 모델)은 재현율 바닥 때문에 이번 판정 밖이다. 이를 보려면 그 모델이 검출되는 조건(SNR·배경)을 먼저 찾아야 한다.
   - (b) HB1b의 배경 차(IDS만 깊은 행 손실)는 기록 끝 거리(`rows_to_record_end_P`)로 층을 나눠 보는 것이 다음 진단이다.
   - (c) HH2는 경계에 붙었다(하한 0.498). 위치를 늘리면 판정될 가능성이 크다(MDE80 0.649 대 점추정 0.659).


---

작성 경위: 이 판정서는 gpr-processing-expert hand-back 원문(sha256 LF `3e28dc9c1e8614527eb6330d2071b6541553b61ed9d37bf2ac8925a41f85c360`)을 `handback_save.py`로 저장한 것이다. 메인 세션은 뒤에 붙은 '메인 세션용 메모' 절(판정 밖)을 떼어 내고 §2 1회차 시작·끝 시각 줄과 §12 대조표 한 칸을 채웠다.

## 판정 후 감사 반영 (메인 세션 보충, 2026-11-26)

verif-protocol-auditor 판정 후 감사(`docs/processing/m33_e3x_judgment_audit.md`): 차단 0, 독립 재집계로 가설 6 + 음성 대조의 추정량·구간·결정 라벨 7/7 일치(차 ≤ 3.5×10⁻¹⁷). 감사 지적을 다음과 같이 반영했다. 판정 라벨은 바뀌지 않는다.
1. (발견 1) 결정 1의 `--uninjected` 명시 미이행을 §2 결정 1 줄에 공개했다.
2. (발견 2) 재시도 러너 사본 `docs/processing/evidence/m33_e3x_retry_runner/`(SHA256SUMS)를 만들고 [RL] 정의에 적었다. 이 러너는 r1 실행 전 감사 뒤에 만들어져 실행 전 감사를 받지 않았다. 판정 후 감사가 `ps1_lint` 종료 0, `$StopOnFail = $true`, 단계 인자가 봉인 문구와 같음을 확인했다.
3. (발견 3) §2에 1회차 injected 끝 시각(시작 + wall, 추정)을 보충했다.
4. (발견 5, 결과 뒤 확인한 해석 차이) HB1b 대상 위치를 규칙 `plan.md:385`는 "네 행(28·47·68·90) 모두 있는 위치"로, 판정 도구 `m33_e3x_judge.py:1066-1070`은 47·68·90만 요구한다. 이번 자료에서는 HB1 대상이 띠 위치 전체(98)라 모든 위치에 28 px 단위가 있어 두 정의의 HB1b 61위치가 같다. 다음 판에서 코드와 규칙 문구를 맞춘다.
5. (발견 7) 이동량 민감도(R-e)는 팔 단위가 아니라 위치마다 모든 팔의 \|n₂\| 최댓값으로 위치를 나눈다(`m33_e3x_judge.py:1799-1807`, `:1891-1892`). §6 R-e 표는 이 기준으로 읽는다.
6. (발견 6) [RC] 스크립트 `analysis/processing/m33_e3x_judgment_recheck.py` docstring의 "봉인 도구의 추출 순서를 일부러 재현하지 않음, 약 1e-2 일치"는 초기 의도다. 실제 재집계는 같은 추출 순서(`rng.integers(0, G, (B, G))`, 규칙 `plan.md:400` "구현이 정한 고정 순서")를 재현했고 구간 끝 차는 ≤ 3.5×10⁻¹⁷이다. 스크립트는 sha로 인용했으므로 고치지 않는다.
7. (integrity 점검) judge 행의 [J] `started_at`(13:49:06)과 러너 `[start]`(13:48:35) 차이를 §1 표에 적었다. 렌더에서 링크로 잘못 해석되던 `[구간](…)`·`[RC](…)`·`[RR](…)` 네 곳에 공백을 넣었다(내용 불변). 작성자 재집계([RC])의 구간 끝 차 "≤ 3×10⁻¹⁷"과 verif 독립 재집계의 "≤ 3.5×10⁻¹⁷"(최대 3.47×10⁻¹⁷)은 서로 다른 재집계 값이다.
8. 이월(다음 판, python-expert): 판정 도구 identity에 uninjected run_id·summary sha·status·자기 code_blobs·`finished_at`을 넣고, uninjected `status`·`judgeable`을 확인한다.
