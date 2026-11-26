# M3-3 ③ E3x verif 실행 전 감사 재확인 (verif-protocol-auditor `[시점] 실행 전`, 2026-11-26, 봉인 2 r1 + precheck r1 뒤, GPU 재시도 전)

## 감사 요약
- 의존성 점검: 모두 있음(numpy 2.5.3, pyyaml 6.0.3, git)
- 본 결과: 판정 값은 보지 않았습니다. 연 것은 아래뿐이고, 내용 값은 이 리포트에 옮기지 않았습니다(통과·실패와 건수만 적음).
  - `runs/eval/m33/e3x_precheck_r1/precheck.json`: `status`·`checks`(10항목 통과 여부)·`not_run`·`judgment`·`run_identity`(정체성 키)·`cpu_assertions_sha256`. `defect_counts`는 키 이름만 봤습니다.
  - `cpu_assertions.json`: `status`·`passed`·`assertions`(단언별 n_checked/n_expected/n_failed)·`guard_problems`·`failures` 건수·`run_identity`·`plan_sources`. 질감 기록 값(`texture_positions`·`texture_distribution`)은 출력하지 않았습니다.
  - `cpu_assertions_lines.csv`: 측선별 검사 수·기대 수·실패 수 열
  - `progress.json` 전체(6키)
  - `e3x_precheck_r1/plan/*.csv`, `runs/eval/m33/e3x/plan/`(봉인 2 plan 표, 블라인드 아님): 측선별 행 수와 열 이름
  - `w1/`·`w2/` 기록 파일과 `e3x_precheck_r1_copy/`: sha256만 봤습니다.
  - 1회차 블라인드 폴더(`e3x/injected`·`uninjected001`·`judgment`)와 `e3x_retry*`: 열지도, 나열하지도, 해시하지도 않았습니다.
- 대상: gpr_detection M3-3 마감 전 ③ E3x. 봉인 2 r1(50a28a3) 뒤 precheck r1을 마쳤고, uninjected001 → injected → judge 재시도(`e3x_retry/`) 직전입니다.
- 기준 문서
  - `docs/processing/m33_e3x_plan.md` v0.5: sha256 89fff477, 커밋 50a28a3, 408df3a에서 바뀌지 않음, `eol=lf`
  - `analysis/detection/m33_e3x_seal2_r1.json`: sha256 81e57d3e(raw = LF), blob 151d995b
  - 이전 봉인 기록 `m33_e3x_seal2.json`: sha256 94c671ed, 바뀌지 않음
  - 이전 감사 `docs/processing/m33_e3x_prerun_audit.md`(d720fe3)
- 판정: 차단 0 / 실행 전 수정 1 / 기록 11, 확인 필요 3. **재시도 본 실행을 시작해도 됩니다.** Critical·High에 해당하는 막는 지적은 없습니다. 실행 전 수정 1은 러너 단계와 판정 도구 호출 인자를 정하는 절차로 닫을 수 있고, 코드를 바꿀 필요가 없습니다.

## 감사 범위별 결론(호출 항목 1–6)

| # | 항목 | 결론 | 근거(아래 상세) |
|---|---|---|---|
| 1 | 50a28a3 blob diff가 질감 단언(b2)·precheck 10·재시도 경로에 한정되는가, 바뀐 함수가 plan 경로에 닿지 않는가 | 한정됨. plan 경로에 닿는 변경은 CLI 기본 경로 상수 2개뿐이고, plan 단계는 이 둘을 읽지 않음. plan 표 재사용은 정당함 | 기록 1 |
| 2 | precheck 10 건수 가드 | 단언 5개 모두 검사 수 = 기대 수 > 0. 내가 plan 표에서 독립적으로 다시 센 기대 수와도 5/5, 측선별 155/155 일치. 실패 0, 가드 문제 0 | 기록 2 |
| 3 | 1–10 전부 통과, 실행 순서, run_identity ↔ 봉인 2 r1 | 10/10 통과, `not_run` 없음. 순서는 코드·checks 기록 순서·파일 mtime이 1 → plan → 10 → 3 → (6) 2·4–9로 맞음. 정체성은 전 항목 일치. HEAD 408df3a는 봉인 blob·정체성 단언에 영향 없음 | 기록 3·4 |
| 4 | 공개 23(1회차 진단 열람)의 성격 | 결과 열람이 아님(판정 지표 비노출). 증거 사본은 원 transcript와 바이트 일치. 다만 사후에 만든 예외 규정이고 파일 전체를 메모리로 읽었다는 점은 공개 그대로 둠 | 기록 5 |
| 5 | 재시도 경로 블라인드 | detection 단계와 판정 도구 모두 1회차 루트를 읽기 전에 거부함(시험 통과). plan은 봉인 경로에서 sha로 단언함. 1회차 폴더를 읽는 코드 경로는 없음 | 기록 6·7 |
| 6 | 실패 처리표의 재시도 완결성 | 대체로 완결. 빈틈 2개: uninjected001이 실패했을 때 injected를 계속할지와 판정 도구가 그 기준값을 어떻게 다룰지, 재시도의 종료 3에서 `--resume`을 쓸지 `e3x_retry2`로 갈지 | 실행 전 수정 1, 확인 필요 1·2 |

## 발견 사항 (심각도순)

[실행 전 수정][E·R] uninjected001 재시도가 실패하면 처리 규칙과 판정 도구 동작이 어긋납니다. 이전 감사의 같은 지적이 코드에서 아직 닫히지 않았습니다
- 근거
  - 규칙 `m33_e3x_plan.md:649`: "그 폴더는 열지 않고 남긴다. … 판정 가설에는 쓰이지 않으므로 injected를 막지 않는다. 다만 R-d 기준값이 없으면 판정서에 '기준값 없음'을 공개한다"
  - 규칙 `:653`: 재시도도 "부분 출력으로 끝나면 그 폴더는 열지 않고 남긴다"
  - 판정 도구 `m33_e3x_judge.py:2055-2058`: `summary.json`과 `uninjected001_records.csv`가 있기만 하면 R-d 기준값으로 읽습니다.
  - `assert_run_identities`(`:1959-1979`)는 봉인 기록 sha와 code blob만 대조합니다. uninjected의 `status`·`judgment.judgeable`은 보지 않습니다.
  - 러너 `run_uninjected001`(`m33_e3x.py:3337-3360`)은 `e3x_defect`일 때도 기록과 summary를 씁니다.
  - 이전 감사(`m33_e3x_prerun_audit.md:45-53`, `:164`)는 "판정 도구가 status ok와 judgeable을 확인"을 다음 판의 수정으로 권했습니다. v0.5는 판정 도구 `run`을 고쳤지만(`:2049`, `:2053`) 이 확인은 넣지 않았습니다.
  - 1회차에는 사용자 결정으로 "러너대로 멈춤(`--stop-on-fail`)"을 썼습니다(`m33_e3x_prerun_audit.md:235`). v0.5 표 `:649`는 여전히 "막지 않는다"입니다. 재시도 러너는 아직 없습니다(`~/.claude/runners/gpr_detection-m33_e3x*`에 precheck_r1만 있음).
- 실패 시나리오: 재시도 uninjected001이 `e3x_defect`(NMS 경고, 300박스 타일)로 끝나고, 메인이 injected와 판정 도구를 기본 인자로 이어 돌립니다. 그러면 판정 도구가 실패한 `e3x_retry/uninjected001`의 기록을 R-d 기준값으로 씁니다. 판정 가설 6개에는 영향이 없지만, 보고 전용 R-d가 오염되고 "열지 않고 남긴다"는 폴더를 도구가 엽니다.
- 권장 조치(절차로 닫음, 담당 메인 세션)
  - 재시도 러너를 만들기 전에 정합니다: uninjected001이 0이 아니면 (a) 1회차처럼 멈출지, (b) 표 `:649`대로 injected를 이어갈지. (b)면 make_runner의 전역 `--stop-on-fail`을 그 단계에 걸지 않아야 합니다. 정한 내용은 판정서에 공개합니다.
  - uninjected001이 0이 아닌 종료 코드로 끝나면, 판정 도구는 `--uninjected <성공한 재시도 폴더, 예: e3x_retry2/uninjected001>` 또는 없는 경로로 명시해 부릅니다. 기본값 `e3x_retry/uninjected001`을 그대로 쓰지 않습니다.
  - 코드 수정(status·judgeable 확인)은 지금 하면 새 blob → 봉인 2 r2 → precheck r2가 필요하므로, 이번 실행에서는 하지 않는 것을 권합니다.

[기록][S·R] 1. 50a28a3 diff 범위와 plan 재사용 근거: 확인했습니다
- blob 비교(`git ls-tree`)
  - 729d19c = d720fe3에서 50a28a3 사이에 바뀐 코드 blob은 `m33_e3x.py`(b601daf → d13d6e7), `m33_e3x_place.py`(e226651 → 0f1491a), `m33_e3x_judge.py`(0da693f → d9ed14b) 셋뿐입니다.
  - `m33_e3x_pool.py`(a3452bd)와 나머지 재사용 모듈 14개, `src` tree d0e54933은 같습니다.
  - 408df3a는 `.gitignore` 2줄만 바꿨습니다(`git diff 50a28a3 408df3a`).
- 독립 AST 대조(내 스크립트. 판정 도구 함수는 import하지 않음)
  - 최상위 정의 기준으로 바뀐 정의는 71개이고, 봉인 기록 `plan_reuse.changed` 목록과 같습니다.
  - plan 단계 디스패치(`run` → `start_context`, `run_plan`. `m33_e3x.py:4109-4115` `STAGE_RUNNERS[Stage.PLAN] = run_plan`을 읽어 확인)로 좁히면, 도달 정의는 158개(기록 `n_plan_path_definitions` 158)입니다. 그 가운데 바뀐 것은 `DEFAULT_PRECHECK_DIR`·`SEAL2_RECORD` 2개로 기록 `reachable_changed`와 같습니다.
  - 디스패치를 좁히지 않은 그래프에서는 `STAGE_RUNNERS` dict를 거쳐 71개가 모두 도달합니다. 그래서 이 좁힘은 필요한 해석이고, 코드로 정당합니다.
  - 두 상수는 argparse 기본값뿐입니다. plan 단계는 `settings.seal_record`를 `tracked_inputs`에 넣지 않습니다(`m33_e3x.py:463-464`). `stage_dir`도 precheck가 아니면 `out_root/plan`입니다(`:307-312`). 초안 도구의 `REACHABLE_NOTES`(`m33_e3x_seal2_r1_draft.py:79-90`)와 같습니다.
  - 모듈 수준 비정의 문장과 데코레이터는 바뀌지 않았습니다. import는 표준 라이브러리 이름 추가(`as_completed`, `UTC, datetime`, `asdict`)뿐입니다.
- 바뀐 코드 정의의 성격(docstring을 뺀 AST 비교)
  - `time_jitter`는 docstring만 바뀌었습니다. 실현 비트가 그대로라는 v5-1 전제와 맞습니다.
  - `texture_check`·`TextureCheck`(`m33_e3x_place.py:1134-1242`)는 §2-7·§5-2 5와 같습니다: 항등식을 모든 띠 열에서 1e-12로, $r_N\le10^{-6}$으로 보고, 기록 전용 5필드를 남깁니다.
  - `m33_e3x.py`에서 바뀐 정의는 `images=False` 경로(`build_line`·`hyperbola_entries`·`band_entries`), 검사 수·실패 목록 필드, `assert_seal2`·`detector_stage_inputs`의 plan 경로를 `SEALED_PLAN_DIR`로 고정한 것, 진행 파일, `assert_retry_output` 호출, precheck 10·순서입니다.
  - 판정 도구는 기본 경로 3개, `assert_retry_paths`·`assert_run_identities`, `--plan` 기본값만 바뀌었습니다. 통계 로직 diff는 0입니다.
- 초안 도구 `m33_e3x_seal2_r1_draft.py --check`를 다시 돌렸습니다: 종료 0, `sha256_lf` 81e57d3e…, `n_changed` 71, `n_reachable_changed` 2.
- plan 표 실재 대조: `runs/eval/m33/e3x/plan/`의 표 5개와 `plan_report.json`의 LF sha가 봉인 r1 값과 같습니다. `plan_report.run_identity`의 run_id·head(38c05cb)·code_blobs(시작·끝)는 `plan_code_blobs`·`plan_run_id`·`plan_head`와 같습니다.

[기록][E·C] 2. precheck 10 건수 가드: 규칙대로 작동했고 독립 재집계와 같습니다
- 코드
  - `planned_check_counts`(`m33_e3x.py:3572-3605`)는 기대 수를 csv 행에서 셉니다(`read_plan`을 믿지 않음).
  - `cpu_assertion_guard`(`:3706-3729`)는 세 가지를 봅니다: 측선 집합이 같은지, 측선·단언마다 검사 수 = 기대 수인지, 단언마다 전체 검사 수 > 0인지.
  - `passed = not problems and n_failed == 0`(`:3836`)
  - 규칙 `plan.md:613` "단언마다 검사 수가 plan 표에서 센 기대 수와 같고 0보다 크다"와 같습니다.
- 실측(`cpu_assertions.json`): 상태 ok, 측선 155(배경 153 + precheck 2), 가드 문제 0, 실패 0. 단언별 검사 수 = 기대 수 > 0, 실패 0:

  | 단언 | 검사 수 = 기대 수 |
  |---|---|
  | shift_mismatch | 569 |
  | off_differs_from_on | 768 |
  | r_edge_differs_from_plan | 149 |
  | texture_check | 101 |
  | zero_test | 155 |

- 독립 재집계(규칙 문장에서 다시 구현): 봉인 plan 표와 precheck plan 표에서 센 값은 다음과 같고 5개 모두 도구 기대 수와 같습니다. `cpu_assertions_lines.csv` 155행도 측선별로 검사·기대·실패 수가 모두 맞습니다(불일치 0).
  - 띠 팔 행 수
  - 쌍곡선 group 수 × SNR 4 × 극성 2
  - 쌍곡선 위치 수(층 구분 없이 전부)
  - 긴 팔이 47·68 px에 있는 띠 위치 수
  - 측선당 1
- 이전 감사의 기록 "precheck에 음성 대조·회색 층 위치가 없어 4번 재측정이 조작 층에서만 확인됨"(`m33_e3x_prerun_audit.md:112`)은 10번이 전 측선의 쌍곡선 149위치를 층 구분 없이 세므로 닫혔습니다.
- 질감 단언의 `all(...)`(`m33_e3x_place.py:1169`, gate_scan 경고)은 키가 상수 `JITTER_PERIODS`(0.1, 0.2)라 빈 입력이 될 수 없습니다. 오탐입니다.

[기록][I·T] 3. precheck r1 전부 통과, 순서, 정체성 일치
- `precheck.json`: `status` ok. `checks` 10개가 모두 참이고 `not_run`은 []입니다. `judgment.judgeable`은 참, blockers []. `tree_stable`은 참, `tree_changes` [].
- 순서
  - 코드 `run_precheck`(`m33_e3x.py:3932-4043`)는 1 → precheck plan → 10 → 3 → 탐지기 로드 → w1·w2 → 6 → 2·4·5·7·8·9 순서입니다.
  - 1이 실패하면 즉시 닫고(`:3960-3965`), 10이 실패하면 탐지기를 부르지 않습니다(`:3976-3984`).
  - `checks`의 기록 순서는 `1_identity, 10_cpu_assertions, 3_shift, 6_conf_paths_equal, 2_zero_test, 4…, 5…, 7…, 8…, 9…`입니다.
  - 파일 mtime도 같은 순서입니다: plan 02:30:00 → cpu_assertions 02:33:01 → w1 02:44:54 → w2 02:56:36 → precheck.json 03:05:23.
  - 6번은 w1·w2 실행 안에서 계산하고 2·4·5는 그 뒤에 w1 결과로 집계합니다. "10 → 3 → 2, 4–9" 조건을 어기지 않습니다.
- 러너 로그: precheck 10 요약 줄이 규칙 `:624`의 6키(`precheck`·`status`·`n_lines`·`n_checks`·`n_failed`·`elapsed_s`)로 나왔고, 마지막 줄 키 집합은 §5-3과 같습니다. `[end] exit=0 wall_s=2267`, `.done exit=0`.
- run_identity ↔ 봉인 2 r1(즉석 스크립트로 대조)
  - `code_blobs` 18개가 시작·끝 모두 기록 값과 같습니다.
  - `src_tree_id`(시작·끝)가 같습니다.
  - `seeds`가 같습니다.
  - `inputs_sha256.seal2` = 봉인 r1 LF sha, `inputs_sha256.pool_csv` = 기록 값입니다.
  - head = 408df3a(시작 = 끝), `dirty` false, `tree_problems` [], `pre_seal2_untracked` [], `allow_dirty` false
  - argv는 `--precheck --workers 4 --precheck-dir runs/eval/m33/e3x_precheck_r1`이고 workers 4입니다.
  - `cpu_assertions.json`의 run_identity는 `finished_at`·`tree_end`를 빼면 precheck.json과 같습니다.
  - `plan_sources.main` 표 sha가 봉인 `plan_files_sha256_lf`와 같고, `plan_sources.precheck`는 `e3x_precheck_r1/plan` 파일의 LF sha와 같습니다.
- `started_at`(02:27:46)은 python 시작 시각이고(`m33_e3x.py:3959`), 러너 `[start]` 02:27:39(conda 기동 7 s 포함)와 맞습니다. 이전 감사 기록(정체성 생성 시각)은 고쳐졌습니다.
- 폴더와 사본: 두 폴더가 각 22파일로 이름 목록이 같습니다. `precheck_r1_sha256.txt` 44줄이 raw와 LF 모두 44/44 일치합니다(CRLF 파일 0). `precheck.json`에 기록된 `cpu_assertions_sha256` raw·lf도 실제 파일과 같습니다.

[기록][I] 4. HEAD 408df3a(`.gitignore`만 변경)는 봉인 단언·정체성에 영향이 없습니다
- 봉인 단언 `assert_seal2`(`m33_e3x.py:772-811`)와 판정 도구 `read_inputs`·`assert_run_identities`(`judge.py:1959-2014`)는 pool·plan 표·plan_report sha, code blob, src tree, 봉인 기록 sha만 봅니다. HEAD는 보지 않습니다.
- HEAD는 같은 실행의 시작·끝 비교(`close_run`, `:511-522`)와 `--resume` 조건에만 쓰입니다.
- `tracked_inputs`(`:452-465`)에 `.gitignore`는 들지 않습니다. 408df3a가 `.claude/blind_paths.json`을 무시하게 해서 작업 트리가 clean입니다(`git status` 빈 출력). 이 커밋 전이었다면 미추적 파일 하나가 생겼을 것입니다.
- 결론: precheck r1(408df3a)과 재시도는 같은 봉인 blob으로 묶입니다. 판정 도구는 injected와 uninjected의 HEAD 일치를 요구하지 않습니다. 다만 재시도 각 단계 안에서 커밋하면 그 단계가 `input_error`가 됩니다(아래 메인 세션 권고 2).

[기록][T] 5. 공개 23(1회차 injected 폴더 열람): 결과 열람이 아니라고 판단합니다. 증거는 원본과 일치합니다
- 증거 대조
  - `transcript_excerpt_lines_50_52_54_55_59_60.jsonl` 6줄이 원 transcript `…\8552c4b2-…\subagents\agent-a69e5d6b43ae10991.jsonl`의 50·52·54·55·59·60행과 바이트 단위로 같습니다.
  - 원 transcript는 지금 295행입니다. 앞 270행의 sha가 `reconstructed_from_transcript.md:9`에 기록된 0e20ee93…과 같아, 전사 뒤 앞부분이 바뀌지 않았습니다.
  - 규칙 `plan.md:547-548`에 인용된 sha 2010ea04·fed36bba는 seal_check `--quoted`로 두 파일과 raw 일치합니다.
- 범위
  - 사람(에이전트)에게 나온 것은 세 가지입니다: 1회차 injected 폴더의 파일 이름 8개(`ls`), `summary.defects` 키 이름 7개, 0이 아닌 키 `texture_check`와 그 측선 3개.
  - 재현율·적중·덮임·점수·대비·영점 값은 출력되지 않았습니다.
  - `hit_without_operating_box`·`cover_without_operating_box`·`coverage_over_length`는 `Counter`에 조건부로만 더해집니다(`m33_e3x.py:2671-2681`). 그래서 키 목록에 없다는 것은 0이라는 뜻입니다. 공개 23의 "검출 의존 단언이 0이었다는 것도 알려 준다"는 이 경로로 맞는 설명입니다. 이 값은 구현이 맞으면 늘 0인 단언이라 판정 정보가 아닙니다.
- 남는 점(공개 문구는 이미 정확함)
  - 이 예외(`plan.md:654`)는 열람(2026-11-20 15:24) 뒤에 v0.5(17:48 커밋)에서 만든 규정입니다. 스크립트와 stdout 원본이 없어서 "스크립트·stdout·실행 시각·sha256을 남겨"는 transcript 전사로 대신했습니다. 판정서에 "사후 규정·전사 증거"로 그대로 적으면 됩니다.
  - 스크립트는 `summary.json`과 `parts\*.json` **전체를** 메모리로 읽었습니다(`reconstructed_from_transcript.md:75-77`에 공개됨).
- 추가 스캔: 2026-11-20 뒤에 수정된 이 프로젝트 transcript 33개(내 transcript 제외)에서, 1회차 블라인드 폴더를 직접 건드린 도구 호출(Read·Grep·Glob·Bash·PowerShell)을 찾았습니다.
  - 공개 23의 2건(`ls`, `defect_names.py`)
  - 메인 세션 8552c4b2의 837·840행: `summary.json`을 읽고 `run_identity.run_id`·`head`만 출력. §5-3이 허용한 정체성 블록이고, 공개 22의 run_id 출처입니다.
  - 그 밖에는 없습니다.

[기록][R·E] 6. 재시도 경로 블라인드: 코드 경로가 규칙과 같습니다
- `assert_retry_output`(`m33_e3x.py:354-373`)은 `run_injected`·`run_uninjected001`의 첫 줄(`:3022`, `:3311`)에서 출력 폴더가 1회차 루트 안이면 거부합니다(`input_error`, 종료 2). 입력을 읽거나 폴더를 만들기 전입니다.
- plan은 `--out-root`와 무관하게 `SEALED_PLAN_DIR`(`:160`, `:786`, `:2996`)에서 읽고 sha를 단언합니다. 1회차 루트에서 읽는 것은 블라인드가 아닌 `e3x/plan/`뿐입니다.
- 판정 도구의 기본값은 `e3x_retry/{injected,uninjected001,judgment}`입니다(`judge.py:115-117`). `assert_retry_paths`(`:1936-1950`)는 1회차 경로를 입력·출력 어디에 주든 경로만 보고 거부합니다.
- 시험 `test_first_try_outputs_are_refused`(`tests/test_m33_e3x.py:1429-1464`) 통과.
- `e3x_retry`는 경로 부분 비교(`is_relative_to`)라 `e3x` 아래로 잡히지 않습니다.

[기록][R] 7. 진단용 인자 제한이 docstring보다 느슨합니다(이전 감사에서 이월, 이번 명령에는 영향 없음)
- docstring `m33_e3x.py:88-93`은 "정식 실행에서 `--out-root`는 `e3x_retry`·`e3x_retry<k>`만, `--precheck-dir`는 `e3x_precheck_r<k>`만"이라고 적었습니다. 코드는 1회차 루트만 거부하고, 다른 임의 경로는 판정 가능 실행으로 받습니다(`scope_blockers` `:319-326`에 없음).
- `--stage plan --overwrite`는 봉인 뒤에도 기본 루트의 봉인 plan 폴더를 지웁니다(`:566-574`). 보존 사본 `e3x_plan_copy/`가 있고 재시도 명령은 이 인자를 쓰지 않습니다.
- 명령을 봉인 문구대로(`--out-root runs/eval/m33/e3x_retry`) 주면 문제없습니다.

[기록][I] 8. E코어 마스크 0x0FFF0000·BelowNormal 실행은 규칙상 문제가 없습니다. 판정서에 기록합니다
- 규칙에서 실행 정체성 필드는 작업자 수뿐입니다(§6 1 "작업자 수", §5-1 "작업자 수는 status.py와 배치표로 정한다"). 마스크·우선순위·BLAS 스레드는 정체성 필드나 단언이 아닙니다.
- precheck 판정 항목은 모두 같은 실행 안의 자기 일관성 단언입니다(7번 workers 1↔2 바이트 동일, 10번의 plan 대조). 다른 마스크의 저장 출력과 비교하는 항목은 없습니다.
- plan 표(다른 시점·배치에서 생성)와의 대조인 `shift_mismatch` 정확 일치와 `r_edge` 상대 1e-9가 E코어 마스크에서 569·149건 모두 통과했습니다. 따라서 마스크 차이가 plan 대조를 바꾸지 않았다는 실측 근거가 있습니다.
- 러너 단계 인자는 초안(.bak)과 같습니다. diff는 생성 시각, Priority, RunnerMask, wait 조건, 예약 마스크, `Write-Reserve` 함수뿐입니다.
- 재시도도 마스크가 자유이지만, 실측 마스크·우선순위·스레드 환경 변수는 `run_identity`에 남지 않습니다. 러너 로그(`[start] … mask=0x… priority=…`)를 판정서 §6 1에 옮깁니다.

[기록][R] 9. A-5 시험 4의 "precheck 2측선"은 합성 대역(offline 픽스처)에서만 단언됩니다
- `test_offline_precheck_runs_every_check`(`tests/test_m33_e3x.py:1586-1597`)와 `test_offline_precheck10_equals_build_line_and_guards_counts`(`:1647-1675`)는 합성 측선에서 두 경로의 defects·검사 수가 같음을 단언합니다. 통과했습니다.
- 실제 precheck r1에서는 w1 경로(이미지 있음)의 검사 수와 10번(이미지 없음)의 검사 수를 직접 대조하지 않습니다. w1의 CPU 단언 통과(2·4·5번 참)만 있습니다.
- 본 실행도 같은 단언을 다시 세므로 판정 영향은 없습니다. 판정서에 "실측 2측선의 두 경로 검사 수 대조는 없음"으로 적습니다.

[기록][S] 10. 봉인 형태와 저장소 밖 증거
- 규칙 v0.5에는 `.sha256`이 없습니다(seal_check `seal_missing`). 고정 근거는 커밋 50a28a3(origin/main에 있음)과 `eol=lf`입니다. 봉인 r1 json에는 규칙 문서 sha가 들어 있지 않습니다. 이전 감사와 같은 방식이므로, 판정서에 규칙 sha 64자리와 커밋을 적습니다.
- 봉인 plan 표 6개는 `/runs/` 무시 규칙에 걸려 커밋에 없습니다(`cited_ignored` 6). sha는 추적되는 봉인 기록에 고정돼 있고, 탐지 단계와 판정 도구가 단언합니다. 이전 감사와 같습니다.
- 다음 증거는 저장소 밖 `C:\Users\jinbeom\.claude\runners\…`에만 있습니다.
  - precheck r1 sha 목록 `precheck_r1_sha256.txt`
  - 공개 23 증거 `h6_disclosure23\` 2파일
  - 진단 `texture_scan.py`(plan `:555`)

[기록][E] 11. 도구 실행 결과와 오탐 판별
- `log_errors.py`(precheck_r1 logs): 종료 0. 단 이 러너는 "errors N" 줄을 찍지 않아(info `no_errors_line` 3) 사실상 빈 통과입니다. 판정은 종료 코드와 `status`로 했습니다.
- `placeholder_scan.py --pattern-set both`(ps1, steps.json): 종료 1, `<LogDir>` 1건. `run_precheck_r1.ps1:71`의 주석 줄(make_runner가 생성한 설명)이라 오탐입니다. 단계 인자 줄에는 자리표시가 없습니다.
- `ps1_lint.py`: 종료 0(info `tee_captured_with_return` 1).
- `gate_scan.py`(E3x 3파일): 종료 1, warning 2·info 5. 모두 오탐이거나 기존 항목입니다.
  - `judge.py:796` `fold_means`: 이전 감사의 오탐과 같습니다(FOLDS 상수).
  - `place.py:1169`: 위 기록 2
  - `Progress.update` 경고 뒤 진행: 진행 파일은 기록 전용이라 설계대로입니다(`m33_e3x.py:386-393` docstring).
  - `judge.py:1892` `max(default=0.0)`: 보고 전용 이동량 민감도이고 v0.5에서 바뀌지 않았습니다.
- `ref_check.py plan.md --rev 408df3a`(블라인드 경로 제외 인자 사용, 제외된 참조 10): 종료 1. `ref_missing` 39는 모두 앞으로 생길 출력, 일반 파일 이름, glob, 저장소 밖 도구입니다. `ref_untracked` 6은 `runs/` 등 무시된 입력입니다(이전 감사에서 실재 확인). 재시도가 읽는 추적 입력 가운데 빠진 것은 없습니다.
- `timeline.py`: 비교 2건 모두 before(규칙 git 시각 < precheck `started_at`). warning `seal_time_missing` 1(`.sha256` 없음).
- `junit_skips.py`: 시험 48개 통과, skip 0, xfail 0, critical skip 0(아래 실행한 검증).

## 규칙↔도구 대조표 (v0.5 변경분)

| 항목 | 규칙 원문 | 도구 | 판정 |
|---|---|---|---|
| 질감 단언 (b2): 모든 띠 열 항등식 1e-12, $r_N\le10^{-6}$ | `plan.md:337-340`, `:608` | `m33_e3x_place.py:88-91`, `:1163-1175`, `:1189-1194`, `:1220-1229` | 같음 |
| 기록 전용 5필드(위치별 $r_N$, 0.8 나이퀴스트 위 에너지, 원형↔4n 선형 차, 시작 계단, 상자·테이퍼 열 최대 |δ|) | `plan.md:608` | `place.py:92-111`, `:1097-1131`, `:1226-1229` | 같음($r_N$은 단언 겸 기록) |
| 실현은 v0.4 그대로 | `plan.md:65`, `:541` | `time_jitter`는 docstring만 바뀜(AST 비교) | 같음 |
| precheck 10: 전 측선 153 + 2, 같은 함수·같은 순서, 영상화 없음 | `plan.md:613-626` | `m33_e3x.py:2454-2523`(`images=False`), `:3625-3656`, `:3659-3676` | 같음 |
| 건수 가드 | `plan.md:613` | `:3572-3605`, `:3706-3729` | 같음(측선별 일치 + 단언별 합 > 0) |
| 실행 순서 1 → plan → 10 → 3 → 2, 4–9, 실패 시 생략 | `plan.md:600` | `:3952-4043` | 같음 |
| 10번 기록·stdout 6키 | `plan.md:622-624` | `:3844-3874` | 같음 |
| precheck 폴더 새로 만들기, 있으면 거부 | `plan.md:627` | `:3952-3953` | 같음 |
| `started_at` = 실제 시작 | `plan.md:627` | `:3959` | 같음 |
| 재시도 출력 `e3x_retry`, 1회차 루트 거부 | `plan.md:593`, `:653` | `m33_e3x.py:153`, `:354-373`, `:3022`, `:3311` | 같음(다른 임의 루트는 거부하지 않음, 기록 7) |
| plan은 봉인 경로에서 | `plan.md:593`, 공개 28 | `:160`, `:786`, `:2996` | 같음 |
| 판정 도구 입력·출력 = `e3x_retry`, 1회차 거부, 봉인 sha·blob 일치(차수가 달라도 됨) | `plan.md:776-777` | `judge.py:115-117`, `:1936-1979`, `:2049-2053` | 같음. uninjected `status`를 보지 않음(실행 전 수정 1) |
| 진행 파일 6키 | `plan.md:636`, `:771` | `m33_e3x.py:384-421` | 같음(precheck r1 `progress.json` 실측 6키) |
| 봉인 r1 기록 키(`supersedes`, `plan_code_blobs`, `plan_reuse`) | `plan.md:734` | `m33_e3x_seal2_r1.json` | 같음. `supersedes` sha 94c671ed·커밋 729d19c 일치 |
| precheck 1: E3x blob = 봉인 r1 기록 | `plan.md:604` | `m33_e3x.py:3494`(seal2 sha → inputs), `assert_seal2` | 같음 |

## 확인 필요
1. 재시도에서 uninjected001이 실패하면 injected를 이어갈지(`plan.md:649` "막지 않는다") 아니면 1회차처럼 멈출지(사용자 결정, `m33_e3x_prerun_audit.md:235`), 재시도 러너를 만들기 전에 사용자가 정해야 합니다. 어느 쪽이든 판정 도구에 `--uninjected`를 명시하는 규칙은 같습니다(실행 전 수정 1).
2. 재시도 injected·uninjected001이 종료 3(`error`, GPU 메모리 부족 포함)으로 끝났을 때 처리가 두 행으로 갈립니다.
   - `plan.md:655`: "같음(`--resume`은 같은 HEAD·blob·src tree에서만)"
   - `:653`(재시도 행, 0/1/2/3): 부분 출력이면 "그 폴더는 열지 않고 남긴다. 그다음 `e3x_retry2/`"
   - 권고: 재시도 행이 더 구체적이므로 `e3x_retry2`로 갑니다. `--resume`을 쓰려면 실행 전에 결정 기록으로 남깁니다.
   - 같은 맥락: 재시도에서 출력 폴더를 만들기 전의 종료 2는 `:648`·`:650`의 "같은 명령으로 다시"가 맞습니다(`:653`의 2는 부분 출력이 있을 때로 읽음).
3. 재시도 직전에 이 감사 문서를 커밋할지 정합니다. 커밋하면 재시도 HEAD가 precheck r1 HEAD(408df3a)와 달라집니다. 규칙에는 두 HEAD가 같아야 한다는 요구가 없고 blob 단언에는 영향이 없습니다(1회차도 precheck 729d19c ↔ 본 실행 d720fe3였음). 다만 판정서에 두 HEAD를 함께 적습니다. 실행 중 커밋은 금지입니다.

## 위임 필요
- python-expert(프로젝트 코드, 이번 실행에서는 고치지 않음. 고치면 새 blob → 봉인 2 r2 → precheck r2 → verif 재확인)
  - `m33_e3x_judge.py:2055-2058`, `:1959-1979`: uninjected001 `summary.json`의 `status == "ok"`와 `judgment.judgeable`을 확인합니다. 아니면 기준값을 쓰지 않거나(`null`, 판정서 공개) 종료 2로 멈춥니다(실행 전 수정 1). 이전 감사부터 이월된 항목입니다.
  - `m33_e3x.py:88-93` 대 `:319-326`, `:566-574`: 정식 실행의 `--out-root`를 `e3x_retry`·`e3x_retry<k>`로 제한하거나 그 밖을 `not_judgeable`로 표시합니다. 봉인 2 뒤의 `--stage plan --overwrite`도 거부합니다(기록 7).
  - precheck 10에 "실측 precheck 2측선에서 w1 경로 검사 수 = 10번 검사 수" 대조를 추가합니다(기록 9).
- python-expert(`tools/verif` 후보, 반복 검사)
  - plan 재사용 도달성 독립 재계산(입력: 두 커밋, 모듈 목록, 디스패치 간선 json; 출력: 바뀐 정의·도달 정의 수·도달한 변경 목록; 종료 0 일치 / 1 불일치). 이번에는 즉석 스크립트 2개로 했습니다.
  - transcript 블라인드 접근 스캔(입력: transcript 글롭, 블라인드 경로 정규식; 출력: 도구 호출 줄·시각·도구 이름·입력 앞부분, Agent·Write 프롬프트 언급 제외; 종료 0 없음 / 1 발견). 공개 문구 재확인에 반복해서 쓰입니다.
  - `baseline_check sums`가 "<sha>  <상위 폴더 기준 상대 경로>" 꼴(`precheck_r1_sha256.txt`처럼 목록 파일이 저장소 밖에 있고 경로 기준이 `runs/eval/m33`)을 `--base <폴더>`로 받게 합니다. 지금은 `sums-dir`·`repo-root`만 있어 즉석 대조로 했습니다.
  - (이전 감사 이월) `seal_check --quoted`가 40자리 git blob·tree id를 `git hash-object`·`rev-parse`로 풉니다(이번에도 `quoted_unresolved` 39건 = 전부 blob·tree id, `git ls-tree`로 손 대조 일치).

## 메인 세션 권고
1. 재시도 러너를 만들기 전에 확인 필요 1·2를 정해 결정 기록에 남깁니다. 단계 인자는 봉인 문구 그대로 씁니다.
   - `--stage uninjected001 --out-root runs/eval/m33/e3x_retry`
   - `--stage injected --workers N --out-root runs/eval/m33/e3x_retry`
   - 판정 도구 기본 인자. uninjected001이 0이 아니면 `--uninjected` 명시
   - 확인 필요 1에서 "이어감"을 고르면 uninjected001 단계에 전역 stop-on-fail을 걸지 않습니다.
2. 재시도 시작부터 판정 도구가 끝날 때까지 커밋과 추적 파일 변경을 하지 않습니다(`close_run` HEAD 비교). `.claude/agent-memory/**`도 쓰지 않게 지시합니다. 이 감사 문서를 커밋하려면 러너를 띄우기 전에 합니다. 그 뒤 `git diff --stat 408df3a HEAD`가 문서만 바꿨는지 확인합니다.
3. 띄우기 직전에 `Test-Path`로 `runs/eval/m33/e3x_retry`(와 `e3x_retry2`)가 없는지 존재만 확인합니다. 내용은 보지 않습니다. 비어 있지 않은 폴더가 있으면 도구가 종료 2로 거부합니다.
4. 저장소 밖 증거를 프로젝트 근거 폴더로 복사하고 SHA256SUMS를 만듭니다(`tools\verif\baseline_check.py write-sums`). 원본은 그대로 둡니다. 판정서 §6 2의 "precheck r1과 사본의 경로·파일별 sha256"·"공개 23 증거 경로·sha"의 근거가 됩니다. 대상:
   - `C:\Users\jinbeom\.claude\runners\gpr_detection-m33_e3x_precheck_r1\precheck_r1_sha256.txt`(+ 러너 ps1·steps.json·logs)
   - `C:\Users\jinbeom\.claude\runners\gpr_detection-m33_e3x\diag_texture_20261120\h6_disclosure23\`
5. 15분 감시(long-running-task-monitor)에는 블라인드 경로로 `runs/eval/m33/e3x/{uninjected001,injected,judgment}/`, `e3x_retry*/`, `e3x_precheck*/`를 넘깁니다. 예외는 `e3x_retry/{uninjected001,injected}/progress.json`뿐입니다(`plan.md:635-636`). 1회차 injected 폴더는 재시도와 비트 단위로 같은 검출 기록을 담고 있을 것이 거의 확실합니다(실현·검출 결정적, 단언만 바뀜). 그래서 판정서 작성 전까지 해시·비교도 하지 않습니다(`plan.md:654`).
6. 판정서 §6에 적을 것(이번 감사에서 나온 것)
   - precheck r1의 실행 마스크 0x0FFF0000·BelowNormal(정체성 밖, 러너 로그 근거)과 재시도 마스크
   - 공개 23의 사후 규정·전사 증거 성격(transcript 행 일치 확인됨)
   - 실측 2측선의 두 경로 검사 수 대조 없음(기록 9)
   - 규칙 v0.5 sha 89fff477…(64자리)·커밋 50a28a3

## 사용자 조치 필요
없음

## 검사하지 못한 것
- uninjected001·injected 재시도의 실제 출력(아직 없음). 판정 도구가 읽을 injected `summary.json` 키(`run_identity.seal2.seal_record_sha256_lf`, `code_blobs`, `outputs_sha256`, `judgment.judgeable`, `zd_summary`)는 건식 출력이 없어 `key_presence.py`로 보지 못했습니다. 대신 코드(`m33_e3x.py:3028`, `:3041-3056`)와 통과한 offline 시험 `test_offline_seal2_injected_stage_and_judge_run`으로 확인했습니다.
- `baseline_check.py sums`·`identity`: 이 실험에는 SHA256SUMS와 identity 파일이 없어 해당하지 않습니다. 대신 sha 목록 44줄 즉석 대조(44/44)와 run_identity 즉석 대조(위 기록 3)를 했습니다.
- 마스크(P코어 ↔ E코어)가 GPU 추론 결과 비트에 미치는 영향: 규칙 항목이 아니라 보지 않았습니다.
- `inject_untapered`·`_band_realization` 등 `images=True` 경로의 바뀌지 않은 본체는 다시 전수로 읽지 않았습니다(diff 0). 바뀐 정의만 읽었습니다.
- 진단 `texture_scan.py`, EM 진단 스크립트(지워짐)의 입력 범위: 공개 25의 문장과 transcript 스캔(블라인드 접근 없음)에 기댔습니다.

## 실행한 검증
- 인터프리터
  - 검사·재집계: python-dev `C:\ProgramData\miniforge3\envs\python-dev\python.exe -I -B`(MKL_THREADING_LAYER=TBB, 한글 출력 `-X utf8`). 판정 도구 함수는 import하지 않았습니다.
  - 프로젝트 시험: `conda run --no-capture-output -n gpr-dev python -I -B -m pytest tests/test_m33_e3x.py -p no:cacheprovider --basetemp=<scratchpad> --junitxml=<scratchpad>\junit.xml`. E코어 마스크 FFF0000, BelowNormal, OMP·MKL·OPENBLAS 스레드 1(P코어 0–15는 다른 세션 `ed7-l1ref`가 예약 중). 결과: 48 passed(137 s), 종료 0, GPU 미사용(`--device cpu` 픽스처).
  - 봉인 초안 대조: `conda run -n gpr-dev python -I -B analysis/detection/m33_e3x_seal2_r1_draft.py --check`. 종료 0, sha 81e57d3e…, 71·2.
- verif 도구와 종료 코드
  - `depcheck.ps1`: CHECKED
  - `log_errors.py`(precheck_r1 logs): 0
  - `placeholder_scan.py --pattern-set both`(ps1, steps.json): 1(주석 오탐 1)
  - `ps1_lint.py`: 0
  - `gate_scan.py`(E3x 3파일): 1(오탐·기존 항목)
  - `seal_check.py --repo … plan.md --code <3파일> --quoted plan.md --quoted seal2_r1.json --hash-files <공개 23 증거 2, seal2.json, pool, plan 표 6, feasibility 2> --check-tracked --rev 408df3a --min-hex-len 13`: 1
    - error 6(`cited_ignored` = 봉인 plan 표, 이월)
    - warning 40(`quoted_unresolved` 39 = git blob·tree id, 손 대조 일치. `seal_missing` 1)
    - 64자리 인용 sha는 모두 풀림(2010ea04, fed36bba, 94c671ed, 06f00783, plan 표·report, feasibility 2)
  - `ref_check.py plan.md --rev 408df3a --exclude <블라인드 6 + .claude>`: 1(위 기록 11)
  - `timeline.py --rules plan.md --results precheck.json --results cpu_assertions.json --time-field run_identity.started_at`: 1(`seal_time_missing`만, 2건 before)
  - `junit_skips.py junit.xml --critical "test_4|precheck|first_try|seal2_r1|offline|test_11|test_12|test_13"`: 0
- 즉석 스크립트(scratchpad `verif-e3x-prerun-r1\`, 모두 지움)
  - `seal_keys.py`: 봉인 r1 키·changed 목록
  - `ast_diff.py`: 독립 AST diff·도달성, 디스패치 좁힘 유무 두 번
  - `ast_other.py`: 모듈 수준 문장·데코레이터·import
  - `doc_only.py`: docstring만 바뀐 정의 판별
  - `keys.py`: precheck.json 키 트리, 값은 상태·정체성만
  - `identity_check.py`: 정체성 ↔ 봉인 대조, 독립 기대 수 재집계, sha 목록 44줄, mtime 순서
  - `plan_blobs.py`: plan_report 정체성 ↔ `plan_code_blobs`
  - `excerpt_check.py`: 공개 23 사본 ↔ transcript 6행, 앞 270행 sha
  - `transcript_scan.py`: transcript 33개 블라인드 접근 스캔. 도구 입력만 보고 결과 내용은 보지 않음
- 조회: `git ls-tree`(5개 커밋), `git diff --stat`·`-U`, `git ls-files --eol`, `git check-attr`, `git log -1`, `git branch -r --contains 408df3a`(origin/main), `git status`(시험 뒤 clean, 새 `__pycache__` 없음), `status.py`
- 만든 뒤 지운 파일: scratchpad `C:\Users\jinbeom\AppData\Local\Temp\claude\8d33c416c891208f\e25399d9-3470-4376-a080-46da64520752\scratchpad\verif-e3x-prerun-r1\` 폴더 전체(스크립트 9개, seal·ref·gate·timeline 보고서, junit.xml, pytest basetemp, guard.json)
- 지우지 못한 부산물: 하네스가 만든 도구 출력 사본 `C:\Users\jinbeom\.claude\projects\C--workspace-gpr-detection\e25399d9-3470-4376-a080-46da64520752\tool-results\b169xwoq7.txt`. transcript 스캔 1차 출력(도구 입력 명령 앞부분)이고, 결과 값은 없습니다.
- 최종 확인: 보고 직전에 HEAD가 408df3a이고 `git status`가 비어 있음을 확인했습니다. source_guard `--strict-additions` 검증이 종료 0이었습니다(감사 중 입력 파일 변경 없음).
- 원본 불변: 확인 — 원본 쓰기 0회, 원본 디렉터리 쓰기 0건 (sha m33_e3x_plan.md 89fff477…, m33_e3x_seal2_r1.json 81e57d3e…, m33_e3x_seal2.json 94c671ed…, m33_e3x.py ac00f412…, m33_e3x_judge.py 7e6a110a…, m33_e3x_place.py 898dd80b…, m33_e3x_seal2_r1_draft.py d362c2ae…, cpu_assertions.json 17408607…, precheck.json a374e301…, cpu_assertions_lines.csv 4a6cd43b…, run_precheck_r1.ps1 3337a6b5…, steps.json c8aac138…, precheck_r1_sha256.txt 98cd3797…, reconstructed_from_transcript.md 2010ea04…, transcript_excerpt_lines_50_52_54_55_59_60.jsonl fed36bba…, 디렉터리 목록 동일, ~$ 임시 파일 없음)
- 에이전트 메모리: `C:\Users\jinbeom\.claude\agent-memory\verif-protocol-auditor\audit_pitfalls.md`에 일반 함정 1항목을 덧붙이고 `MEMORY.md` 색인 줄을 갱신했습니다(Edit). 저장소 안 파일은 건드리지 않았습니다.
- 사용한 스킬: 없음

---

## 메인 세션 덧붙임: 결정·조치 (2026-11-26, 감사 원문 뒤, 재시도 전)

위 절까지는 hand-back 원문 그대로다(`handback_save.py`, hand-back sha256 LF `d8c98ee0629d08ed076bd7f1f96c43decde06e98218e396237a0514f67712706`).

사용자 결정(확인 필요 1–3, 결과 전):
1. 재시도 uninjected001이 0이 아닌 종료 코드로 끝나면 **멈춘다**(1회차와 같음, 규칙 표 `:649`의 "막지 않는다"보다 우선). injected를 시작하지 않고 보고한다. 판정 도구에는 성공한 uninjected 폴더를 `--uninjected`로 명시해 넘긴다(실행 전 수정 1을 절차로 닫음). 판정서에 공개한다.
2. 재시도 단계가 종료 3(GPU 메모리 부족 포함) 등 부분 출력으로 끝나면 `--resume`을 쓰지 않고 그 폴더를 열지 않은 채 남긴 뒤 새 폴더 `e3x_retry2/`로 간다(`:653` 재시도 행). 출력 폴더를 만들기 전의 종료 2는 같은 명령으로 다시 한다.
3. 이 감사 문서는 재시도 러너를 띄우기 전에 커밋한다. 재시도 HEAD는 precheck r1 HEAD(408df3a)와 달라지며, 판정서에 두 HEAD를 함께 적는다.

메인 조치:
- 저장소 밖 증거 사본: `docs/processing/evidence/m33_e3x_prerun_r1/`(precheck r1 러너 ps1·.bak·steps.json·로그·요약·`precheck_r1_sha256.txt`, 공개 23 증거 2파일), 폴더 `SHA256SUMS`(14항목, `baseline_check.py write-sums --recursive`). 원본은 그대로 둔다. 바이트 보존을 위해 `.gitattributes`에 `docs/processing/evidence/** -text`를 넣었다.
- precheck r1 실측: E코어 0x0FFF0000·BelowNormal, wall 2,267 s, 작업 VRAM 정점 1,977 MB(`vram_source` target-counters), `record.py` 기록.
