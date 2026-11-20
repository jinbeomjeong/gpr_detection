# M3-3 ③ E3x verif 실행 전 감사 (verif-protocol-auditor `[시점] 실행 전`, 2026-11-20)

> 메인 세션 주: 아래는 verif-protocol-auditor 핸드백 원문 전체다(요약하지 않음). 정정·사용자 결정은 끝 절에 덧붙인다.

## 감사 요약
- 의존성 점검: 모두 있음(numpy 2.5.3, pyyaml 6.0.3, git)
- 본 결과: 판정 값은 보지 않았습니다. 연 것은 다음과 같습니다.
  - `runs/eval/m33/e3x_precheck/precheck.json`: `status`·`checks`(9항목 통과 여부)·`judgment`·`run_identity` 일부 키
  - `e3x_precheck/plan/*.csv`: 검출과 무관한 행 수, 모델·층 구성
  - `e3x_precheck`와 `e3x_precheck_copy`의 19개 파일: sha256만(내용은 출력하지 않음)
  - `runs/eval/m33/e3x/plan/`(블라인드 아님): 위치·층·배경별 건수, `plan_report.json`의 `lines` 블록과 정체성 키
  - `w1/`·`w2/` 기록 파일과 주입·검출 값: 열지 않았습니다.
- 대상: gpr_detection M3-3 ③ E3x. 시점은 실행 전(GPU 본 실행 직전)입니다.
- 기준 문서
  - `docs/processing/m33_e3x_plan.md` v0.4: sha256 ec5da8f4, 커밋 38c05cb, 봉인 2 커밋 729d19c에서 바뀌지 않음
  - 봉인 1 검토 문서 3개와 feasibility py·json: 72b0140 뒤 diff 없음
  - `analysis/detection/m33_e3x_seal2.json`: sha256 94c671ed, plan 폴더의 `seal2_record_draft.json`과 바이트 동일
- 판정: 차단 0 / 실행 전 수정 4 / 기록 10. **본 실행을 시작해도 됩니다.** 실행 전 수정 1–4는 모두 절차 문서·운영 조치로 닫을 수 있고 코드를 바꿀 필요가 없습니다. 실행을 막지 않습니다.

## 발견 사항 (심각도순)

[실행 전 수정][R·E] 실패 처리표의 재시도 경로 `runs/eval/m33/e3x_retry/`는 코드로 실행되지 않습니다
- 근거
  - 규칙 `m33_e3x_plan.md:549`·`:551`: "`runs/eval/m33/e3x_retry/`에서 다시"
  - 코드에서 `RETRY_OUT_ROOT`(`analysis/detection/m33_e3x.py:127`)는 정의만 되고 어디서도 쓰이지 않습니다(Grep 결과 127·67행 docstring뿐).
  - 탐지 단계는 봉인된 plan을 `settings.out_root / PLAN_DIR`에서 읽습니다(`m33_e3x.py:654` `assert_seal2`, `:2760` `read_plan`).
  - `--out-root`에는 제한이 없고, `scope_blockers`(`:261-268`)에도 들지 않아 어느 경로든 판정 가능 실행이 됩니다. docstring `:64-68`의 "diagnostic" 설명과 다릅니다.
- 실패 시나리오: injected나 uninjected001이 실패한 뒤 표대로 `--out-root runs/eval/m33/e3x_retry`로 다시 돌리면 `e3x_retry/plan/`이 없어 입력 오류(종료 2 또는 3)로 멈춥니다. 표에 적힌 재시도 절차를 그대로 따를 수 없습니다.
- 권장 조치: 코드 수정 없이 절차로 닫을 수 있습니다.
  - 재시도가 필요하면 `e3x/plan/` 7개 파일을 `e3x_retry/plan/`에 복사합니다. 복사본은 `assert_seal2`가 sha로 단언하므로 그대로 통과합니다.
  - 판정 도구는 `--injected runs/eval/m33/e3x_retry/injected --plan runs/eval/m33/e3x/plan`으로 부릅니다.
  - 이 명령을 실행 전에 결정 기록(결과 전)으로 남깁니다. 담당: 메인 세션, 문구는 gpr-processing-expert.

[실행 전 수정][T·I] stats-design-expert의 MDE 산출물이 저장소 안 미추적 파일로 생겼습니다. 실행 중에 커밋하면 HEAD가 바뀌어 injected가 무효가 됩니다
- 근거
  - `git status --porcelain`에 다음이 새로 나타났습니다: `?? analysis/processing/m33_e3x_mde_eval.csv`·`.json`·`m33_e3x_mde_eval/`(13:24–13:57) 그리고 `docs/processing/m33_e3x_mde_eval.md`(14:01:37). scratchpad의 `stats-e3x-mde` 폴더는 없어졌습니다. 호출문의 "scratchpad에서, 저장소 무변경"과 사실이 다릅니다.
  - 러너 `close_run`(`m33_e3x.py:383-394`)은 시작과 끝의 HEAD를 비교합니다(`m33_inputs.py:189` `TREE_KEYS`에 head 포함). 다르면 `final_status`가 `input_error`(`m33_inputs.py:333-334`)가 됩니다.
  - 미추적 파일 자체는 `tracked_tree_problems`(필수 경로만 봄)에 걸리지 않아 시작은 막지 않습니다.
- 실패 시나리오: 본 실행 중(약 1–3 h) MDE 보고를 커밋·push하면 injected가 끝날 때 `input_error`가 됩니다. GPU 실행 한 번이 날아가고 재시도 절차가 필요해집니다.
- 권장 조치
  - 둘 중 하나를 고릅니다. (a) 본 실행 전에 MDE 파일을 커밋(+push)하고 HEAD가 새 커밋이 된 뒤 러너를 띄웁니다. seal2는 HEAD가 아니라 blob을 단언하므로 문제없습니다. (b) 판정 도구가 끝날 때까지 커밋하지 않습니다.
  - (a)가 낫습니다. §6 8항의 "결과 개봉 전 시각 근거"가 git 시각으로 남기 때문입니다(mtime만으로는 `[추정]`).
  - 판정서에 stats가 저장소에 직접 쓴 사실을 적습니다. 담당: 메인 세션.

[실행 전 수정][E·R] 판정 도구가 uninjected001 기준값을 status 확인 없이 씁니다. 러너는 실패 처리표와 달리 uninjected001이 실패하면 injected를 막습니다
- 근거
  - 판정 도구 `m33_e3x_judge.py:1996-1999`: `summary.json`과 `uninjected001_records.csv`가 있기만 하면 읽습니다. `status`나 `judgment.judgeable`은 보지 않습니다.
  - 러너 `run_uninjected001`(`m33_e3x.py:3081-3104`)는 `e3x_defect`일 때도 기록과 summary를 씁니다.
  - 실패 처리표 `plan.md:549`: "그 폴더는 열지 않고 남긴다 … injected를 막지 않는다"
  - 러너 스크립트 `run_e3x.ps1`: `$StopOnFail = $true`라서 uninjected001이 실패하면 injected와 판정을 시작하지 않습니다.
  - `conda.bat run`의 종료 코드 전달을 실측했습니다: `sys.exit(3)` → `$LASTEXITCODE` 3.
- 실패 시나리오: uninjected001이 `e3x_defect`로 끝나고 메인이 injected와 판정을 수동으로 이어 돌리면, 판정 도구 기본값이 실패한 폴더의 기록을 R-d 기준값으로 그대로 씁니다. 판정 가설에는 영향이 없지만 보고 전용 R-d가 오염되고, "열지 않는다"는 폴더를 도구가 엽니다.
- 권장 조치: 절차로 닫습니다. uninjected001이 0이 아닌 종료 코드로 끝나면 판정 도구를 `--uninjected runs/eval/m33/e3x_retry/uninjected001`(재시도가 성공했을 때) 또는 없는 경로로 부르고, 판정서에 "기준값 없음"을 적습니다. 다음 판에서는 판정 도구가 status ok와 judgeable을 확인하게 합니다(python-expert). 표와 러너 중 어느 쪽을 따를지는 "확인 필요" 5를 보십시오.

[실행 전 수정][S] 봉인된 plan 표 7개가 `.gitignore`(`/runs/`)에 걸려 커밋에 없고, 보존 사본이 없습니다
- 근거: `seal_check --check-tracked --rev 729d19c` 결과 `cited_ignored` 7건(`runs/eval/m33/e3x/plan/*.csv`·`plan_report.json`·`seal2_record_draft.json`, `.gitignore:3:/runs/`).
- sha는 추적되는 `m33_e3x_seal2.json`에 고정돼 있고 모두 일치합니다(raw = LF). injected(`m33_e3x.py:656-662`)와 판정 도구(`judge.py:1948`)가 이를 단언하므로 바뀌면 멈춥니다. 안전 쪽입니다.
- 실패 시나리오: plan 폴더를 잃거나 실수로 `--stage plan --overwrite`를 돌리면(`m33_e3x.py:442-443`, 봉인 2 뒤에도 거부하지 않음) 판정이 입력 오류로 멈춥니다. plan은 결정적이라 재생성은 가능하지만 증명 단계가 하나 늘어납니다.
- 권장 조치: 본 실행 전에 `runs/eval/m33/e3x/plan/`을 `runs/eval/m33/e3x_plan_copy/`로 복사하고 sha를 대조해 남깁니다(추적 트리 변화 없음). 담당: 메인 세션.

[기록][S] 봉인 기록 방식(`.sha256` 없이 `m33_e3x_seal2.json`)이 충분한가
- seal2.json에 들어 있는 것(모두 실측 일치)
  - pool csv LF sha, plan 표 5개와 plan_report의 LF sha
  - 코드 blob 18개: 커밋 트리·작업 트리·봉인 값 모두 OK
  - src tree `d0e54933`: 729d19c와 HEAD 모두 같음
  - plan_head, run_id, 시드 3개
- 하류 단언
  - uninjected001·injected: `detector_stage_inputs` → `assert_seal2`(`m33_e3x.py:2753-2764`, `:644-679`)가 pool·plan 표·plan_report·blob·src tree·plan status를 봅니다. seal2.json 자체도 추적 무변경이어야 합니다(`:335-336`).
  - 판정 도구: plan 표·현재 blob·injected 정체성 blob을 봅니다(`judge.py:1945-1957`).
- 빠진 것: 규칙 문서 v0.4의 sha(ec5da8f4…), [PDD] 진단 8파일, 검토 문서, 시험 파일 blob. 이들은 seal2.json에 없고, 저장소 어디에도 ec5da8f4가 적혀 있지 않습니다(`git grep` 0건). 근거는 git 커밋 38c05cb·72b0140뿐입니다.
- 판단: 커밋이 origin/main에 push되어 있고(`git branch -r --contains 729d19c` → origin/main) 규칙 파일이 `eol=lf`라 git 기준 봉인으로 충분합니다. 다만 `plan.md:9`의 "봉인 2 = 이 문서 v0.4 + … + [PDD]" 가운데 문서와 [PDD]는 seal2.json이 아니라 커밋으로만 고정된다는 점을 판정서 §6 1항에 적습니다(규칙 sha 64자리, 커밋 38c05cb).
- 판정 도구 쪽 빈틈: 판정 도구는 pool sha와 src tree를 다시 보지 않습니다. 자기가 읽은 seal2 sha(`judgment.json identity.seal_record_sha256_lf`)와 injected `run_identity.seal2.seal_record_sha256_lf`도 비교하지 않습니다. docstring `judge.py:27-29`는 "src tree"를 본다고 적었지만 코드에는 없습니다. 판정 뒤 감사에서 두 값을 대조합니다.

[기록][S] [PDD] 진단 json·csv 6개는 작업 트리가 전부 CRLF이고 인덱스는 LF입니다
- 근거: `git ls-files --eol`에서 6개 모두 `i/lf w/crlf attr/text=auto eol=lf`. 줄 단위 CRLF 67–424개. 파일마다 CR을 지운 LF sha가 `git show 38c05cb:<파일>`의 sha와 같습니다(예 diag.json raw ad696711 ↔ LF eee423aa = 커밋). 원인은 `write_text` 기본 줄바꿈입니다(`m33_e3x_plandefect_diag.py:263` 등).
- 판단: LF(커밋 blob) 기준이 맞습니다. 이 파일들을 인용하는 sha는 없고 러너도 읽지 않아 판정 영향은 없습니다. 인용하려면 LF sha나 blob을 씁니다. 작업 트리 정규화(`git checkout -- <파일>`)를 하려면 실행 뒤에 합니다.

[기록][I] plan·precheck·판정 json의 `run_identity.started_at`은 시작 시각이 아니라 정체성을 만든 시각입니다
- 근거
  - plan: `started_at` 13:08:16. 실제 시작은 stdout·stderr 생성 시각 12:57:37이고, 걸린 시간 625.9 s(plan stdout).
  - precheck: `started_at` 13:45:36. 실제 시작은 13:18:45(stderr 생성)이고 1610 s. 원인은 `m33_e3x.py:3333`(점검을 마친 뒤 `new_identity`)입니다.
  - 판정 도구: `judge.py:2068`(끝에 생성, `finished_at` null)
  - injected와 uninjected001은 실행 전에 만들어 정상입니다(`:2789`, `:3067`).
- 권장 조치: 판정서 §6 1항의 "시작 시각"은 로그 생성 시각, `stage_settings.json`, `elapsed_s`로 적고 그 근거를 밝힙니다.

[기록][T] 순서는 규칙 → v0.4 → 봉인 2 → precheck로 맞습니다. 다만 precheck가 stats MDE보다 먼저 돌았습니다
- 시각
  - 봉인 1: 10:10:25
  - pool csv: 10:54(pool.py 10:36보다 뒤)
  - v0.4 커밋: 12:57:10(코드 최종 12:55:32, 문서 최종 12:56:14)
  - plan 재실행 시작: 12:57:37, `e3x/plan` 폴더 새로 생성 12:57:50, 끝 13:08:21, head 38c05cb, E3x 코드 5개는 `pre_seal2_untracked`
  - 봉인 2: 13:18:16
  - precheck 시작: 13:18:45, head 729d19c, dirty false, tree_end 같음
  - 사본: 13:46:01
  - 러너 생성: 13:47:14
- 블라인드 대상 폴더 `e3x/uninjected001`·`injected`·`judgment`·`e3x_retry`는 없습니다(확인만).
- `plan.md:9`의 순서("봉인 2 → stats 평가 → precheck")와 달리 precheck(13:18)가 MDE(13:23–14:01)보다 먼저 돌았습니다. 둘 다 결과 전이고 검출과 무관해 판정 영향은 없습니다. 판정서 이력에 적습니다.

[기록][R] 문서 상태 줄과 docstring이 낡았습니다
- `plan.md:3`: "봉인 2 전의 새 판"이 그대로 남았습니다(봉인 2는 끝남).
- `m33_e3x_judge.py:3`: "v0.3 (seal 1)". v0.4는 §3 판정 규칙을 바꾸지 않았으므로(diff 확인, 수치 문맥만 바뀜) 동작 영향은 없습니다.
- 다음 판이나 판정서에서 정정합니다(gpr-processing-expert / python-expert).

[기록][R] 진단용 인자 제한이 docstring과 다릅니다
- `--out-root`는 어느 경로든 받고 판정 가능 실행이 됩니다.
- `--overwrite`는 봉인 2 뒤에도 plan 폴더를 지웁니다(`m33_e3x.py:438-446`, `:3380-3381`).
- 본 실행 명령(steps.json)은 두 인자를 쓰지 않으므로 이번 실행에는 영향이 없습니다.

[기록][R] precheck 실패 처리표와 precheck 집계
- 표 `plan.md:546-547`에는 precheck 0과 1만 있습니다. 코드에는 2(`input_error`: 폴더 있음, 시작 점검 실패)와 3(`error`)도 있습니다(`m33_e3x.py:3448-3455`).
- precheck 3·4·5 집계(`:3321-3326`)는 `all(... for line in built)`이고 건수 가드가 없습니다. 이번 precheck plan에는 쌍곡선 5위치(모두 조작 층)와 띠 3위치·18팔이 있어 빈 통과는 아니었습니다.
- 다만 음성 대조와 회색 층 위치는 precheck에 없었습니다. 그래서 4번의 $r_\text{edge}$ 재측정(상대 1e-9)은 조작 층에서만 확인됐습니다.

[기록][운영] 예약 TTL과 단계 실패 처리
- 예약 TTL이 규칙 §5-1(`plan.md:509`)의 180분이 아니라 480분입니다(배치 담당이 늘림). 판정서 운영 기록에 적습니다.
- 러너 Stop-on-fail과 실패 처리표의 차이는 "실행 전 수정" 3번에 적었습니다.

[기록][도구] verif 도구 오탐 3건(코드를 직접 읽어 확정)
- `ps1_lint` `tee_with_return`(`run_e3x.ps1:47`): Tee-Object 뒤에 `| Out-Host`가 있어 파이프라인으로 아무것도 나가지 않습니다. 같은 꼴의 함수를 재현해 보니 반환형이 Int32 하나였습니다. 오탐입니다.
- `gate_scan` `all_without_count_guard` 2건
  - `judge.py:785`: `all(f in by_fold for f in folds)`이고, folds는 비지 않는 상수 FOLDS입니다.
  - `place.py:1059`: jitter dict 키가 0.1·0.2로 고정돼 있습니다.
  - 둘 다 빈 입력이 될 수 없어 오탐입니다.

## 규칙↔도구 대조표

| 항목 | 규칙 원문 | 도구 | 판정 |
|---|---|---|---|
| 판정 구간 Satterthwaite: $v_g=\frac{G_g}{G_g-1}\mathrm{Var}_b$(분모 B−1), Kish $G_\text{eff}$, $\nu_g=\max(G_\text{eff}-1,1)$, $\nu$ 식, $t_{\nu,1-0.05/12}$ | `plan.md:361-376`, A-4 `:660` | `judge.py:221-223`, `313-354`(`np.var ddof=1`, `stats.t.ppf(1-tail,nu)`, ν는 반올림 안 함) | 같음 |
| 부분 복제: 다른 배경 가중 1, 같은 추출 재사용 | `plan.md:371` | `judge.py:234-255` | 같음 |
| 수준 상수: 정확한 분수 | `plan.md:381`, `:658` | `judge.py:66` `CI_LEVEL = 1 - 0.05 / 6`, `:68` `TAIL = 0.05 / 12`, `:70-71` | 같음(반올림값 없음) |
| HH2: $\hat\eta^2_p$ 기본(역백분위) 구간, `numpy.quantile` 선형, 자르지 않음, Type II 가법 모형 | `plan.md:339`, `:380`, `:661` | `judge.py:545-550`, `598-638`, `690-758` | 같음. NaN 복제를 빼는 처리는 규칙에 없음(확인 필요 3) |
| 경계 (1) $G_g<2$ → `single_cluster_background`, 음성 대조는 `not_run` | `plan.md:378` | `judge.py:408-417`, `:528-534` | 같음 |
| 경계 (3) $v_g=0$ → ν 분모에서 빼고 표지 | 같음 | `judge.py:348-351`, `1156-1158` | 같음 |
| 경계 (4) `degenerate_ci`: $\hat\Delta\ne0$ → `inconclusive`, =0 → 결정표 | 같음 | `judge.py:349-350`, `489-490` | 같음(음성 대조 쪽은 확인 필요 2) |
| 경계 (5) `single_background` 배경 제외(plan 수 15 미만) | `plan.md:350`, `:378` | `judge.py:1082-1086`, `1299-1315` | 같음. plan 기준 HB1·HB1b·HB2 모두 두 배경이 15 이상이라 발동하지 않음 |
| 추출 순서 HH1 → 음성 대조 → HH2 → HH3 → HB1 → HB1b → HB2 → 보고 전용, 가설 안은 4TU → IDS, 생성기 하나 | `plan.md:357`, A-4 `:657` | `judge.py:1372-1389`, `88`, `177-189`, `2002-2017`(보고 전용과 S3 재판정은 같은 생성기로 이어감) | 같음 |
| MIN_COUNTS와 모델 수 | `plan.md:411-419` | `judge.py:90-102`, `1112-1118`, `1214-1224`, `707-712` | 같음(HH1 (30,15)·모델 5·끔 거부 20 %, HH2 모델 12·군집 20, HH3 (40,20)·30 % 등) |
| 결정표·바닥 조건 $F\ge0.10$(경계 포함), 열린 구간 ±δ | `plan.md:385-398` | `judge.py:468-503` | 같음(HH1 음의 방향 '5 %p 미만' 문구는 확인 필요 1) |
| 음성 대조 (a)–(d): 0.05 이상은 (a), 0.02 이상은 (b) | `plan.md:401-406` | `judge.py:522-542`, `1203-1230` | 같음 |
| HB1 대비 점검(3 dB) | `plan.md:407` | `judge.py:1089-1106` | 같음(평가 안 됨은 표지 없음, 확인 필요 4) |
| 층 배정: $r+s\ge0$ 조작, $r+18<-6$ 음성 대조, 그 사이 회색 | `plan.md:173-175` | `place.py:86-91`, `126-140` | 같음 |
| $r_\text{edge}$ v0.4 차 장 정의 | `plan.md:155-168` | `place.py:149-188` | 같음 |
| 띠 이동 $n_1\to n_2$ 보정 1회, 12 P 곱 형식, 1 샘플 | `plan.md:191-194`, `224-227` | `place.py:54-56`, `459-535` | 같음 |
| 블라인드 stdout | `plan.md:531-535` | `m33_e3x.py:193-206`, `2816-2821`, `2874-2878`, `3434-3437`, `judge.py:2101-2107` | 같음(수만 출력) |
| 실패 처리표 status ↔ 종료 코드 | `plan.md:539-557` | `m33_e3x.py:46-62`, `179-192`, `m33_inputs.py:326-339`, `judge.py:107` | 대체로 같음. 재시도 경로 없음(실행 전 수정 1), precheck 2·3 행 없음(기록) |
| `PRE_SEAL2_FILES` 허용 | `plan.md:9`, `:543-545` | `m33_e3x.py:170-176`, `345-380`(`Stage.PLAN`이고 precheck가 아닐 때만), `387`(탐지 단계는 빈 튜플) | 봉인 2 뒤에는 효력 없음. 탐지 단계는 seal2.json까지 추적 무변경을 요구함 |
| 봉인 2 하류 단언 | `plan.md:517`, A-4 `:655` | `m33_e3x.py:644-679`, `2753-2764`, `3230-3243`, `judge.py:1925-1960` | 있음(빈틈은 위 기록) |

## 확인 필요
1. HH1 `confirmed_negative`에서 $|\hat\Delta|<\delta$일 때 "효과는 있으나 5 %p 미만"을 붙일지 정해야 합니다. 도구는 양의 방향에만 붙이고 판정 결과에 `small_effect`를 남깁니다(`judge.py:491-498`). 규칙 `plan.md:390`·`:397`은 방향 가설의 2행에만 덧붙임을 정했습니다. 해석 문구 문제라 판정서 작성자(gpr-processing-expert)가 정하면 되고, 코드 수정은 필요 없습니다.
2. 음성 대조의 구간이 `degenerate`([Δ,Δ], Δ≠0)일 때 도구는 "0을 포함하지 않음"으로 보고 fail·flag로 갑니다(`judge.py:535-541`). 규칙 경계 (4)는 가설에만 `inconclusive`를 정했고 음성 대조는 "Δ=0이면 통과"만 정했습니다. 발생 확률은 아주 낮습니다(모든 위치의 차가 같아야 함).
3. HH2 복제 가운데 분산 분해가 NaN(합 0)인 복제를 분위수에서 빼고 `n_nan_replicates`로 기록하는 것은 규칙에 없는 도구의 해석입니다(`judge.py:677-686`, `747-757`). 판정서에 그 수를 적을지 정합니다.
4. HB1 대비 점검에 필요한 행(28·47·68)이 비면 `evaluated: false`, 표지 없음으로 끝납니다(`judge.py:1103-1104`). 판정서에 "평가 안 됨"을 적을지 정합니다.
5. uninjected001이 실패했을 때 실패 처리표대로 injected를 이어갈지(`plan.md:549`), 러너대로 멈출지(`StopOnFail`) 사용자가 정해야 합니다.

## 위임 필요
- python-expert(`tools/verif`)
  - `ps1_lint.py`: Tee-Object 뒤에 `| Out-Host`(또는 `$null =`)가 이어지면 `tee_with_return`을 내지 않게 고칩니다. 입력은 `run_e3x.ps1:47-59`(make_runner 표준 `Invoke-Logged`)이고, 기대 결과는 경고 0입니다.
  - `seal_check.py`: 40자리 git blob id(`code_blobs`)와 src tree id를 `--quoted`에서 `git hash-object`·`rev-parse`로 풀게 합니다. 지금은 `quoted_unresolved` 19건이 나와 손으로 대조했습니다.
  - `baseline_check.py identity`: 이 프로젝트의 `run_identity` 블록(`*identity*.json` 파일이 아니라 summary·plan_report·precheck json 안의 키)을 읽는 모드를 추가합니다. 지금은 종료 2로 쓸 수 없습니다.
- python-expert(프로젝트 코드, 다음 판·다음 실험용. 이번 실행을 위해 고치지는 않습니다. 고치면 새 blob이 되어 봉인 2 새 판이 필요합니다)
  - `m33_e3x.py`: `--out-root`가 `RETRY_OUT_ROOT`가 아니면 거부하거나 판정 불가로 표시합니다. 재시도 단계의 plan 경로를 `DEFAULT_OUT_ROOT/plan`으로 고정하고, 봉인 2 뒤의 `--overwrite`를 거부합니다.
  - `m33_e3x_judge.py`: uninjected001 summary의 status ok와 judgeable을 확인하고, seal2 sha를 injected와 대조하며, docstring을 v0.4에 맞게 고칩니다.
  - precheck: `started_at`을 시작할 때 기록하고, 3·4·5 집계에 건수 가드를 넣습니다.

## 메인 세션 권고
1. 실행 전 수정 1–4를 정해 결정 기록에 남긴 뒤 러너를 띄웁니다.
   - 재시도 명령(plan 사본을 `e3x_retry/plan`에)
   - MDE 파일을 먼저 커밋할지, 판정 뒤에 커밋할지
   - uninjected001 실패 시 판정 도구 인자
   - plan 보존 사본
2. **git 자식 Suspended 위험.** 내가 띄운 시험이 git에서 멈췄습니다.
   - 첫 pytest(전체 40개, 백그라운드 작업 `bk8oudju5`)가 `test_offline_precheck_runs_every_check` 안의 `git -C C:\workspace\gpr_detection status --porcelain`에서 멈췄습니다. 이 git은 PID 49664, TID 46056, 시작 `2026-11-20T13:59:30.9682785+09:00`이고 Suspended입니다. `Stop-Process`는 액세스가 거부됐습니다.
   - 위로 이어진 프로세스는 python 68524(pytest), python 34176, conda.exe 25076, cmd.exe 43784로, 모두 내가 띄운 것입니다. `.git/index.lock`은 없습니다.
   - `resume_thread.py` dry-run은 `ready 1`이었습니다. 사용자 승인을 받은 뒤 다음을 실행하거나 TaskStop `bk8oudju5`로 정리해 주십시오: `C:\ProgramData\miniforge3\envs\python-dev\python.exe -I -B C:\Users\jinbeom\.claude\tools\monitor\resume_thread.py --pid 49664 --tid 46056 --started "2026-11-20T13:59:30.9682785+09:00" --apply`
   - 본 러너도 시작과 끝에 git(`hash-object`, `status`, `rev-parse`)을 부릅니다(`m33_inputs.py:235-280`). 15분 감시(long-running-task-monitor, 블라인드 경로 §5-3)에서 Suspended git 자식을 꼭 봐야 합니다.
3. 본 실행 중에는 커밋과 추적 파일 변경을 하지 않습니다. 프로젝트 에이전트 메모리(`.claude/agent-memory/**`)도 쓰지 않게 지시합니다.

## 사용자 조치 필요
없음

## 검사하지 못한 것
- 시험 40개 가운데 offline 3개(`test_offline_plan_realizations_records_and_judge`, `test_offline_seal2_injected_stage_and_judge_run`, `test_offline_precheck_runs_every_check`)와 `test_runner_settings_and_pre_seal_problems`는 이번에 완료를 확인하지 못했습니다(git Suspended로 멈춤, 다시 돌릴 때는 제외). 봉인 커밋 메시지의 "40 passed"는 이번 감사로 다시 증명하지 못했습니다.
- `baseline_check sums`, `seal_check --sums`: 이 실험에는 SHA256SUMS 파일이 없어 해당하지 않습니다. 대신 seal2.json을 손과 seal_check `--quoted`로 대조했습니다.
- `baseline_check identity`: 종료 2(identity 파일 없음). 대신 plan과 precheck의 `run_identity`를 직접 읽었습니다.
- uninjected001·injected 단계의 실제 코드 경로(`line_records`, `infer_entries`, 띠 덮임 계산)는 전수로 읽지 않았습니다. 이 부분은 precheck 6·7과 시험에 기댑니다.
- 보고 전용(R-a–R-g)의 세부 식은 표본으로만 대조했습니다.
- C축 재집계: 실행 전이라 해당 없음.

## 실행한 검증
- 인터프리터
  - 검사: python-dev `python.exe -I -B`(MKL_THREADING_LAYER=TBB, 한글 출력은 `-X utf8`)
  - 시험: gpr-dev `conda run --no-capture-output -n gpr-dev python -B -m pytest tests/test_m33_e3x.py -p no:cacheprovider`(PYTHONDONTWRITEBYTECODE=1). 판정 도구 함수는 import하지 않았습니다.
- 도구별 종료 코드
  - `depcheck.ps1`: CHECKED
  - `seal_check.py --repo . m33_e3x_plan.md --code <E3x 4파일> --quoted plan.md --quoted seal2.json --hash-files <plan 표·pool·processing·입력 csv> --check-tracked --rev 729d19c`: 종료 1. error 7(`cited_ignored` plan 표), warning 20(`quoted_unresolved` 19 = git blob·tree id라 손으로 전부 일치 확인, `seal_missing` 1), info 2. 인용된 64자리 sha 9개는 raw 일치.
  - `git ls-tree 729d19c`·`git hash-object`로 blob 18개와 src tree를 봉인 값과 대조: 전부 OK.
  - `gate_scan.py`(E3x 4파일): 종료 1, warning 2(오탐, 위 기록)
  - `ps1_lint.py run_e3x.ps1`: 종료 1, warning 1(오탐, 재현 함수 반환 Int32)
  - `log_errors.py`
    - `main-e3x-precheck`: 종료 0
    - `main-e3x-plan`: 종료 0
    - 둘 다 info `no_errors_line`입니다. 이 러너는 "errors N" 줄을 찍지 않으므로 이 점검은 사실상 빈 통과입니다. precheck stderr는 0바이트입니다.
  - `placeholder_scan.py`(steps.json, seal2.json, plan_report.json, seal2_record_draft.json): 종료 0
  - `ref_check.py plan.md --rev 729d19c`: 종료 1
    - `ref_missing` 24: 앞으로 생길 출력 경로, 도구 이름, 일반 파일 이름, glob
    - glob 입력 `runs/eval/m33/r2/fold*/boxes.csv`와 `r3/uninjected_ids/fold*/boxes.csv`는 10개 모두 있고, plan 정체성에 LF sha가 기록돼 있습니다.
    - `ref_untracked` 7: `runs/` 등
    - 명세가 참조하는 추적 입력 `m33_r2_lines.csv`, `analysis/processing/m32_stageA_lines_{4tu,ids}.csv`, `docs/processing/fourtu_lines_v1.csv`(작업 트리 CRLF, 러너는 LF sha), `fourtu_apex_candidates_v1.csv`는 모두 추적됩니다.
  - `baseline_check.py identity runs/eval/m33/e3x/plan --expect-commit 38c05cb`: 종료 2(해당 없음)
  - `timeline.py --rules plan.md --results plan_report.json --results precheck.json`: 종료 1(`seal_time_missing`, 비교 2건 모두 before)
  - `junit_skips.py junit2.xml --critical <판정 근거 시험 정규식>`: 종료 0(시험 36개 통과, 건너뜀 0, 4개 제외)
- 그 밖의 조회
  - 즉석 스크립트: 정체성 키 추출, precheck와 사본 sha 대조(19/19 동일), plan 건수(검출 무관)
  - `conda.bat run` 종료 코드 전달 실측(3 → 3)
  - `py-spy dump`(멈춘 pytest), `resume_thread.py` dry-run
  - `git status --porcelain --ignored`, `git branch -r --contains`, `status.py`
- 만든 뒤 지운 파일: scratchpad `verif-e3x-prerun\` 폴더 전체(스크립트 3개, seal·gate·ref·timeline 보고서, junit2.xml, pytest2.txt, tee_test.log, guard.json).
  - 공개: `tee_test.log`(Tee-Object 재현)와 `pytest2.txt`(Out-File)는 정의의 "출력 리디렉션 금지"를 어기고 자기 scratchpad 폴더에 쓴 것이며, 둘 다 지웠습니다.
- 지우지 못한 부산물
  - pytest 임시 폴더 `C:\Users\jinbeom\AppData\Local\Temp\pytest-of-jinbeom\pytest-2822`(멈춘 첫 실행이 쓰는 중)와 `C:\Users\jinbeom\AppData\Local\Temp\pytest-of-jinbeom\pytest-2823`
  - 멈춘 프로세스(위 메인 세션 권고 2)
  - 저장소에 새 `__pycache__`는 없습니다.
- 최종 확인: 보고 직전에 HEAD가 729d19c 그대로이고, 블라인드 출력 폴더 4개가 없고, 감사 대상 파일 22개가 바뀌지 않았음을 다시 확인했습니다(source_guard).
- 원본 불변: 위반 — 입력 폴더에 항목 추가 1건(m33_e3x_mde_eval.md)
  - 설명: 내가 쓴 파일이 아닙니다. stats-design-expert가 14:01:37에 `docs/processing/`에 새로 쓴 미추적 파일입니다(스냅숏 13:49 뒤). 입력 파일 22개는 내용 변경이 0건입니다.
- 에이전트 메모리: `C:\Users\jinbeom\.claude\agent-memory\verif-protocol-auditor\audit_pitfalls.md`와 `MEMORY.md`에 일반 함정 1항목을 덧붙였습니다(Edit).
- 사용한 스킬: 없음

## 메인 세션 정정·결정 (2026-11-20, 본 실행 결과 전)
- 정정: `docs/processing/m33_e3x_mde_eval.md`는 stats-design-expert가 아니라 메인 세션이 그 리포트의 "저장할 내용"을 원문 그대로 저장한 파일이다(14:01). `analysis/processing/m33_e3x_mde_eval.{json,csv}`와 폴더는 stats-design-expert가 직접 썼다(호출문의 "scratchpad에서"와 다름, 판정서에 공개).
- 실행 전 수정 1(재시도 경로): 절차로 닫는다. 재시도가 필요하면 `runs/eval/m33/e3x/plan/` 7파일을 `runs/eval/m33/e3x_retry/plan/`에 복사하고(sha는 `assert_seal2`가 단언) `--out-root runs/eval/m33/e3x_retry`로 해당 단계를 다시 돌린다. 판정 도구는 `--injected runs/eval/m33/e3x_retry/injected --plan runs/eval/m33/e3x/plan`(필요하면 `--uninjected`도 재시도 폴더)으로 부른다.
- 실행 전 수정 2(MDE 산출물 커밋): 사용자 결정 — **본 실행 전에 커밋**(이 문서와 함께). 본 실행 중에는 커밋하지 않는다.
- 실행 전 수정 3(uninjected001 실패 시): 사용자 결정 — **러너대로 멈춤**(`--stop-on-fail`). 멈추면 보고하고 다음 조치를 정한다. 판정 도구는 실패한 uninjected001 폴더를 기준값으로 쓰지 않도록 `--uninjected`에 재시도 성공 폴더 또는 없는 경로를 주고, 판정서에 "기준값 없음"을 적는다. 실패 처리표 549행("injected를 막지 않는다")과 다르게 운용한 것은 판정서에 공개한다.
- 실행 전 수정 4(plan 보존 사본): `runs/eval/m33/e3x_plan_copy/`에 7파일 복사, 해시 7/7 동일(2026-11-20).
- 메인 세션 권고 2(git Suspended): 사용자 승인 뒤 `resume_thread.py --pid 49664 --tid 46056 … --apply` 실행, resumed 1·exited 1.
- 확인 필요 1–4: 판정서 작성자(gpr-processing-expert)가 정하고 판정서에 적는다. 위임 필요(도구·다음 판 코드)는 판정 뒤로 미룬다.
