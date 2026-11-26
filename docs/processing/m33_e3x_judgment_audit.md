# M3-3 ③ E3x verif 판정 후 감사 (verif-protocol-auditor `[시점] 판정 후`, 2026-11-26, 재시도 본 실행·판정서 작성 뒤, 판정서 커밋 전)

## 감사 요약
- 본 결과: `runs/eval/m33/e3x_retry/injected/{hyperbola_records,band_records,rejections}.csv`(적중·덮임·c_lat·거부 사유 열을 독립 재집계에 씀), `injected/summary.json`·`uninjected001/summary.json`(정체성·status·defects·counts·resources), `judgment/judgment.json`(추정량·구간·결정·문구·정체성 키), 재시도 러너 로그 `C:\Users\jinbeom\.claude\runners\gpr_detection-m33_e3x_retry\logs_P\*.log`(진행 줄·마지막 줄). 1회차 블라인드 폴더 `runs/eval/m33/e3x/{uninjected001,injected,judgment}/`는 Read·Grep·Glob·해시 어느 것으로도 열지 않았다. 상위 `runs/eval/m33/`·`runs/eval/m33/e3x/`는 검색 범위로 쓰지 않았다. `ref_check`에는 블라인드 세 폴더와 `.claude`를 `--exclude`로 넣었다.
- 대상: E3x 재시도(`e3x_retry/`), 판정서 `docs/processing/m33_e3x_judgment.md`(미커밋, sha 95311c10…)
- 기준 문서: 규칙 `docs/processing/m33_e3x_plan.md` v0.5(89fff477…, 커밋 50a28a3, blob bcbe46a0 = HEAD = 작업 트리). 봉인 기록 `analysis/detection/m33_e3x_seal2_r1.json`(81e57d3e…, raw = LF, blob 151d995b, 50a28a3 = HEAD). 판정 도구 `analysis/detection/m33_e3x_judge.py`(blob d9ed14b6 = 봉인 기록 값)
- 판정: 차단 0 / 실행 전 수정 0 / 기록 8. 판정서 수정은 3건(아래 목록의 1·2·3). 실질 수정이 필요한 것은 1·2이고, 재집계 결과는 바뀌지 않는다.
- **결론: 판정서 커밋 가능.** 단, 아래 "판정서 수정 목록"의 1·2(공개 문장 추가)와 메인 조치 M1(재시도 러너 증거 사본)을 커밋 전에 하기를 권한다. 막는 지적(Critical/High)은 없다. 판정 가설 6개와 음성 대조의 추정량·구간·ν·v_g·G_g·G_eff·F·위치·군집 수·M*·M*_e·결정 라벨·문구를 규칙 문장에서 독립 구현해 다시 셌다. 모두 [J]와 같았다(차 ≤ 3.5×10⁻¹⁷, 라벨 7/7).

## 발견 사항 (심각도순)

[기록 → 판정서 수정 필요][I·R] 1. 사용자 결정 1의 "판정 도구에 `--uninjected` 명시" 부분이 지켜지지 않았는데, 판정서에 공개되지 않았다(결과 영향 없음)
- 근거
  - 결정 원문 `docs/processing/m33_e3x_prerun_audit_r1.md:293`: "판정 도구에는 성공한 uninjected 폴더를 `--uninjected`로 명시해 넘긴다(실행 전 수정 1을 절차로 닫음). 판정서에 공개한다."
  - 실제 명령 `C:\Users\jinbeom\.claude\runners\gpr_detection-m33_e3x_retry\steps.json:10`: `… python -I -B analysis/detection/m33_e3x_judge.py`(인자 없음). [J] `identity.argv` = `["analysis/detection/m33_e3x_judge.py"]`.
  - 판정서 `m33_e3x_judgment.md:73`은 기본 인자를 "verif 재확인 기록 6"으로 정당화한다. `:133-136`의 결정 1–3 서술에는 `--uninjected` 명시 조항이 빠져 있다.
- 영향이 없는 근거
  - 기본값 `DEFAULT_UNINJECTED = e3x.RETRY_OUT_ROOT / "uninjected001"`(`judge.py:116`)가 성공한 재시도 폴더와 같다.
  - 그 폴더의 [SU]는 `status` ok, `judgeable` true, blockers []이다. 봉인 sha 81e57d3e…와 code_blobs가 injected와 같다. `outputs_sha256`도 실제 파일과 같다(즉석 스크립트로 확인).
- 실패 시나리오: uninjected 재시도가 `e3x_retry2/`로 갔다면 기본값이 옛 폴더를 R-d 기준값으로 읽었을 것이다. 이번에는 해당하지 않는다.
- 권장 조치(gpr-processing-expert 또는 메인): §2 "실행 중 사용자 결정" 1에 다음 문장을 더한다. "결정 1의 `--uninjected` 명시는 러너 명령에 들어가지 않았고, 판정 도구는 기본값(`e3x_retry/uninjected001`, 성공한 폴더와 같음)으로 돌았다(steps.json, [J] `identity.argv`)." §9-2 "실행 중 결정과 감사 기록"에도 같은 내용을 한 줄로 넣는다.

[기록 → 메인 조치][S·T] 2. 재시도 러너 증거([RL])가 저장소 밖에만 있다. 앞 단계 러너는 증거 폴더로 복사됐다
- 근거
  - 판정서 `:13`·`:57-62`가 VRAM 측정값, 마스크·우선순위, 대기 6,373 s, 예약 ID, `run_retry_P.ps1` sha를 `C:\Users\jinbeom\.claude\runners\gpr_detection-m33_e3x_retry\`에서 인용한다.
  - 같은 실험의 precheck r1 러너는 `docs/processing/evidence/m33_e3x_prerun_r1/precheck_r1_runner/`(추적, SUMS 14 통과)로 복사됐다. 1회차 러너는 `docs/processing/evidence/m33_e3x_try1_runner/`(미추적, SUMS 10항목 일치)로 복사됐다. 판정 대상인 재시도 러너만 저장소 밖이다.
  - 인용 sha는 맞다: `run_retry_P.ps1` = dd37e242…(직접 계산). `.done` 3개: uninj 12:59:58 exit=0 129 s, injected 13:48:35 exit=0 2,916 s, judge 13:49:07 exit=0 33 s.
- 이 러너는 실행 전 감사(r1)를 받지 않았다. r1 감사 당시 러너가 아직 없었다(`m33_e3x_prerun_audit_r1.md:42`). 이번에 확인한 내용:
  - `ps1_lint` 종료 0(info `tee_captured_with_return` 1)
  - `placeholder_scan --pattern-set both`: `<LogDir>` 1건. make_runner 주석이라 오탐이다(r1 감사 기록 11과 같은 꼴).
  - `$StopOnFail = $true`(`run_retry_P.ps1:14`): 결정 1과 맞다.
  - `log_errors` 종료 0. 단 `errors N` 줄이 없어 빈 통과다(info 5). 판정은 `.done` exit와 마지막 줄 `status`로 했다.
- 권장 조치(메인): `~\.claude\runners\gpr_detection-m33_e3x_retry\`를 통째로 `docs/processing/evidence/m33_e3x_retry_runner/`에 복사한다(원본은 둔다). `baseline_check.py write-sums --recursive`로 SUMS를 만든다. 판정서 [RL] 정의에 사본 경로를 함께 적는다. `docs/processing/evidence/** -text` 규칙이 이미 있다(try1 폴더 전 파일 `text: unset` 확인).

[기록 → 판정서 수정][R] 3. §6 2항 "(v0.5) 1회차 injected의 시작·끝 시각"에서 끝 시각이 빠졌다
- 근거: 규칙 `m33_e3x_plan.md:675`은 "시작·끝 시각"을 요구한다. 판정서 `:81`에는 시작 `2026-11-20T14:28:59`, exit 1, wall 3,119 s만 있다. 1회차 증거 폴더 `logs/`에는 `e3x-injected.done`이 없다. `e3x-injected.log`의 mtime은 2026-11-20 15:20이다.
- 권장 조치(메인, 제어 줄만 읽음): 끝 시각을 1회차 로그 제어 줄이나 `run_e3x.out`에서 옮긴다. 없으면 "시작 + wall ≈ 15:20:58(`.done` 없음, 추정)"으로 적는다.

[기록][I·E] 4. 판정 도구 정체성의 빈칸과, 이월된 uninjected status 미확인. 이번 결과에는 영향이 없다
- `judge.py:2126-2133` identity에 다음이 없다.
  - uninjected001의 run_id·summary sha. `uninjected_run_id` 키 자체가 없다.
  - 판정 도구 자신의 code_blobs. 다만 `read_inputs:2007-2014`가 작업 트리 blob과 봉인 기록이 같음을 단언한다.
  - `finished_at`(null). 판정서 `:64`에 공개돼 있다.
- `assert_run_identities`(`:1959-1979`)와 `run`(`:2055-2058`)은 uninjected의 `status`·`judgeable`·`outputs_sha256`을 보지 않는다. 앞 두 감사에서 이월된 항목이다.
- 이번 값: [SU] status ok, judgeable true, tree_stable true, head bc57ef1 시작 = 끝, dirty false, uninjected001_records.csv sha가 기록값과 같다. 따라서 R-d 기준값 입력은 유효했다. 판정서의 보고 전용 권고 1(`:568`)이 이미 이 항목을 적었다.
- 권장(python-expert, 다음 판): identity에 uninjected run_id·summary sha·status를 넣는다. status가 ok가 아니면 `rd_baseline = null`로 공개한다. `finished_at`을 채운다.

[기록][R] 5. HB1b 대상 위치 정의가 규칙 문장과 코드에서 다르다. 이번 자료에서는 결과가 같다
- 규칙 `plan.md:385`: HB1b 대상은 "네 행 모두 있는 위치"(28·47·68·90)다.
- 코드 `judge.py:1066-1070`은 47·68·90만 요구하고 28은 요구하지 않는다.
- 결과가 같은 근거: HB1 대상이 98위치 = 띠 위치 전체다(독립 재집계, `band_positions` 98). 따라서 모든 위치에 28 px S_J 단위가 있고, HB1b 61위치는 두 정의에서 같다. 거부된 띠 단위는 `013.3/Path1:b00` 28 px 24 dB 두 극성뿐이라 S_J 밖이다.
- 권장: 판정서 §9-2 "결과 뒤 정한 해석"에 한 줄을 넣는다(선택). 다음 판에서는 코드를 규칙에 맞추거나 규칙 문구를 고친다(python-expert / gpr-processing-expert).

[기록][C] 6. 판정서 작성자 재집계 스크립트의 docstring이 판정서 서술과 맞지 않는다
- `analysis/processing/m33_e3x_judgment_recheck.py:14-16`: "The draw order of the sealed tool is not reproduced on purpose; agreement is judged … to about 1e-2."
- 판정서 `:32`·`:498`은 같은 순서를 재현했고 구간 끝 차가 ≤ 3×10⁻¹⁷이라고 적었다.
- 이 감사의 독립 재집계(별도 구현, 같은 추출 규약)로는 최대 차가 3.47×10⁻¹⁷(HH3 하한)이다. 판정서의 수치 주장은 사실이고 "≤ 3×10⁻¹⁷"은 반올림 표현 차다.
- 판정서가 sha로 인용한 파일이라 고치면 sha가 바뀐다. 판정서 §11에 "스크립트 docstring은 초기 의도이고, 실제로는 같은 순서를 재현했다"를 한 줄 넣는 쪽을 권한다.
- `.claude/agent-memory/gpr-processing-expert/e3x-judgment-facts.md:11`의 "reproducible from the rule text"도 정확하지 않다. 추출 꼴 `rng.integers(0, G, (B, G))`는 규칙이 아니라 구현이 정했다(`plan.md:400` "구현이 정한 고정 순서"). 담당: gpr-processing-expert, 선택.

[기록][R] 7. 보고 전용 해석 중 판정서 §9-2 목록에 없는 것
- 이동량 민감도(R-e)는 위치마다 모든 팔(28·47·68·90 긴 팔, 47·68 짧은 팔)의 \|n₂\| 최댓값으로 위치를 나눈다. 팔 단위로 나누지 않는다(`judge.py:1799-1807`, `:1891-1892`).
- 판정서 `:331-337` 표는 이 기준을 적지 않았다. 판정에는 영향이 없다. §9-2 해석 목록에 한 줄을 넣기를 권한다.

[기록][S·T] 8. 봉인 형태·추적 상태(이전 감사와 같음)
- 규칙에 `.sha256`이 없다(`timeline` `seal_time_missing`). 고정 근거는 커밋 50a28a3(2026-11-20T17:48:02)과 판정서 `:4`의 64자리, `eol=lf`(`git ls-files --eol` i/lf w/lf, `core.autocrlf` true지만 eol=lf가 보호한다)다.
- 판정서가 인용하는 `runs/` 결과 48개는 `.gitignore`에 걸려 있다(`seal_check` `cited_ignored` 48). 프로젝트 규약("runs/ git 제외")대로이고, sha는 판정서와 [J]·[SI]에 고정돼 있다.
- 커밋에 함께 넣어야 할 미추적 파일
  - `docs/processing/m33_e3x_judgment.md`
  - 재집계 4파일. sha 일치: recheck.py ddbbed03…, recheck.json 2620245f…, report.py a1d321de…, report.json 28b07e1d…
  - `docs/processing/evidence/m33_e3x_try1_runner/`(11파일, 모두 -text)
  - (메인 판단) 에이전트 메모리 2파일
- 에이전트 메모리 변경은 실행 뒤다. MEMORY.md 14:06:45와 e3x-judgment-facts.md 14:06:41이 judge 끝(13:49:07)보다 늦다. 재시도 동안 커밋은 없다(bc57ef1 10:31:56 뒤 커밋 0). injected·uninjected의 `tree_end` dirty false, judge `git.dirty` false. 따라서 "실행 중 금지" 위반이 아니다.

## 규칙↔도구 대조표 (판정 경로)

| 규칙 | 규칙 위치 | 도구 위치 | 판단 |
|---|---|---|---|
| y_u = 5 fold 평균(전 fold 필요) | plan.md:363 | judge.py:779-797 | 같음 |
| Y_{p,a} = S×극성 평균, 단위 하나라도 없으면 위치 제외 | :364, :379-386 "두 팔 모두 실현" | :823-834 | 같음 |
| M* = 모델 평균 (Y_on+Y_off)/2 ≥ 0.10 | :388 | :911-924 | 같음(재집계 14모델 일치) |
| HH1 조작 층·S_p^edge, r_edge+s ≥ 0 | :186, :379 | :938-971, place.py:154-168 | 같음. records `edge_stratum`·`edge_snr_set`과 재계산 층 불일치 0 |
| 음성 대조 r_edge+18 < −6, M* 안, S_J | :187, :380 | :1250-1256 | 같음 |
| 음성 대조 (a)–(d), 경계 포함 | :445-449 | :533-553, :1214-1241 | 같음 |
| HH2 η²_p Type II, 위치 ≥ 4 모델, 기본 구간 | :382, :423 | :599-769 | 같음 |
| HH3 12−30 dB 켬, 30 dB 거부 비율 ≤ 0.30 | :383, :459 | :998-1030, :1288-1298 | 같음 |
| HB1·HB2 | :384, :386 | :1055-1079 | 같음 |
| HB1b "네 행 모두 있는 위치" | :385 | :1069-1070(28 요구 안 함) | 다름. 이번 자료에서 결과는 같다(발견 5) |
| F 바닥 ≥ 0.10(경계 포함), 같은 가중 | :392, :435-439 | :411, :510-513 | 같음 |
| Satterthwaite: v_g = G/(G−1)·Var(B−1), G_eff Kish, ν_g = max(G_eff−1, 1), t(1−0.05/12) | :404-421 | :335-365, :232-234 | 같음 |
| 부분 복제(다른 배경 가중 1, 같은 추출) | :414 | :245-266 | 같음 |
| 부트스트랩 B 20,000, seed 20261121, 순서 HH1→음성→HH2→HH3→HB1→HB1b→HB2, 배경 4TU→IDS | :400 | :188-200, :1383-1400 | 같음(독립 구현이 비트 수준까지 재현) |
| 결정표(열린 구간, 하한 0은 확인 아님), HH1 양측 | :428-441 | :479-530 | 같음 |
| 최소 수 | :454-462 | :101-113, :1123-1129 | 같음 |
| 끔 int16 거부 ≤ 20 % | :457 | :974-995(사유 `int16_overflow_untapered`만 셈) | 같음. 켬 거부를 물려받은 끔 행은 켬 사유를 갖는다(`m33_e3x.py:2067-2071`). 끔만 거부된 단위는 0이다(재집계 패턴: on/off 둘 다 채택 1,110, 둘 다 거부 42, 끔만 거부 0) |
| HB1 대비 점검 3 dB | :450 | :1100-1117 | 같음(중앙 4.305 / 4.205 / 4.067 dB, 각 392) |
| 판정서 문구 | §3-3, §3-6 | :151-165 | 같음([J] wording = 판정서 §0·§4) |

## 재집계 표 (판정 대상, 독립 스크립트, 판정 도구 import 없음)

| 항목 | 판정서·[J] 값 | 재집계 값 | 차이 | 판단 |
|---|---|---|---|---|
| M* | 14모델(M04·M15·M16·M18 제외) | 같음 | 0 | 일치 |
| M*_e | 10모델 | 같음 | 0 | 일치 |
| HH1 Δ̂ / CI | 0.009375 / [−0.001466, 0.020216] | 0.009375 / [−0.0014656, 0.0202156] | ≤ 3.5e-18 | 일치 |
| HH1 ν, v_g, G_g, G_eff | 21.136; 6.932e-6 / 6.944e-6; 17/13; 13.559/10.127 | 같음 | ≤ 4e-15 | 일치 |
| HH1 위치/군집/F/결정 | 80 / 30 / 0.41781 / negligible | 같음 | 0 | 일치 |
| HH1 D_p 부호 | 0: 61, +: 14, −: 5 | 같음 | 0 | 일치 |
| {18}만 비중 | 47/80 | 47/80 | 0 | 일치 |
| 음성 대조 Δ̂ / CI / ν | −0.005357 / [−0.016141, 0.005427] / 6.517 | 같음 | 0 | 일치, (c) pass |
| 음성 대조 위치/군집 | 30 / 21(14 + 7) | 같음 | 0 | 일치 |
| HH2 η²_p / 기본 구간 | 0.658997 / [0.497979, 0.739101] | 같음 | 0 | 일치, inconclusive(하한 < 0.5) |
| HH2 ω²_p·M*만 η²_p(보고) | 0.6085 [0.4243, 0.7013]; 0.5291 [0.2988, 0.6287] | 같음 | 0 | 일치(복제 재사용) |
| HH2 위치/모델/군집, NaN | 142 / 18 / 30 + 15, 0 | 같음 | 0 | 일치 |
| HH3 Δ̂ / CI | 0.091233 / [0.036540, 0.145926] | 같음 | ≤ 3.5e-17 | 일치, confirmed |
| HH3 위치/군집/F, 30 dB 거부 | 92 / 32 / 0.46129, 20/112 | 같음 | ≤ 6e-17 | 일치 |
| HB1 Δ̂ / CI | 0.237333 / [0.149342, 0.325324] | 같음 | ≤ 2.8e-17 | 일치, confirmed |
| HB1 F, 위치/군집 | 0.14986, 98 / 44 | 같음 | ≤ 2.8e-17 | 일치 |
| HB1b Δ̂ / CI | 0.097587 / [−0.010901, 0.206074] | 같음 | ≤ 1.4e-17 | 일치, inconclusive |
| HB1b 배경별, F, 위치/군집 | 4TU 0.02132, IDS 0.17386; 0.21182; 61 / 35 | 같음 | 0 | 일치 |
| HB2 Δ̂ / CI | 0.235917 / [0.162532, 0.309301] | 같음 | 0 | 일치, confirmed |
| HB2 F, 위치/군집 | 0.15057, 98 / 44 | 같음 | ≤ 2.8e-17 | 일치 |
| 백분위 구간(R-g) 6가설 | 판정서 :343 | 같음 | ≤ 2.8e-17 | 일치 |
| ci_level | 0.9916666666666667 | == 1 − 0.05/6 | 0 | 일치 |

- 재집계 범위: 판정 가설 6개 + 음성 대조 전부, HB1 대비 점검, int16 거부 구조. 보고 전용 R-a–R-f(R-g 제외), S3, [ZD] 민감도, fold 방향 일치는 다시 세지 않았다(아래 "검사하지 못한 것").

## 확인 필요
1. 발견 1: 사용자 결정 1의 "`--uninjected` 명시"를 지키지 않은 것을 판정서에 공개 사항으로 넣을지, 메인 덧붙임 절로 둘지(권고: §2 결정 1 줄에 한 문장을 넣는다).
2. 발견 2: 재시도 러너 사본을 판정서와 같은 커밋에 넣을지(권고: 넣는다. precheck r1·1회차 러너와 같은 방식).

## 위임 필요
- python-expert(다음 판, 이번 판정 영향 없음): `m33_e3x_judge.py`
  - (a) `:2126-2133` identity에 uninjected run_id·summary sha·status, 판정 도구 code_blobs, `finished_at`을 넣는다.
  - (b) `:1974-1979`·`:2055-2058`에서 uninjected `status == ok`·`judgeable`·`outputs_sha256`을 확인한다(앞 감사에서 이월).
  - (c) `:1069-1070`에서 HB1b가 28 px도 요구하게 하거나, 규칙 문구를 바꾸는 개정을 gpr-processing-expert에 맡긴다.
- `tools/verif` 후보(python-expert): `timeline.py --time-field`가 중첩 키(`run_identity.started_at`)를 읽지 못해 `mtime_only_basis`로 떨어진다. 점 경로 지원 후보다. 입력: summary.json 3개. 종료 1(warning `seal_time_missing` 1, info 3).

## 메인 세션 권고
- M1: 재시도 러너 폴더 사본과 SUMS(발견 2). 실행 명령 예: 복사 뒤 `& C:\ProgramData\miniforge3\envs\python-dev\python.exe -I -B $HOME\.claude\tools\verif\baseline_check.py write-sums C:\workspace\gpr_detection\docs\processing\evidence\m33_e3x_retry_runner --recursive`.
- M2: 1회차 injected 끝 시각 보충(발견 3, 제어 줄만 읽는다).
- 그다음 integrity-checker → 커밋·push(CLAUDE.md 순서).

## 검사하지 못한 것
- 보고 전용 R-a·R-b·R-c·R-d·R-e·R-f 값, S3 결정, [ZD] 주기 민감도, fold 방향 일치는 다시 세지 않았다. 판정서도 대부분 "json에서 옮김"으로 적었다. 판정 라벨과 무관하다.
- precheck r1 `cpu_assertions.json` 단언별 수(569·768·149·101·155)는 이번에 다시 열지 않았다. r1 감사(기록 2)가 독립 재집계로 확인한 값이다. 이번에는 `precheck_r1_sha256.txt` 44/44(`runs/eval/m33` 기준)만 다시 대조했다.
- 1회차 러너 증거 폴더의 로그 본문은 열지 않았다(`baseline_check sums`로 해시만 대조).
- 가중치 sha, 입력 sha(r2_lines 등), r0 정체 11측선 sha는 run_identity끼리만 대조했다. 원 파일은 다시 해시하지 않았다(봉인 단언 `seal2.result` pass, `r_checks` pass에 기댐).
- 판정서 작성자의 [RC]·[RR] json 내용은 열지 않았다(sha만 대조). 독립성을 지키려고 재집계는 그 스크립트와 별도로 했다.

## 실행한 검증
- 인터프리터: `C:\ProgramData\miniforge3\envs\python-dev\python.exe -I -B`(`MKL_THREADING_LAYER=TBB`, BLAS 스레드 1). 프로젝트 모듈 import 없음. 판정 도구 import 없음. 판정 도구 코드는 규칙↔도구 대조와 추출 규약 확인을 위해 읽기만 했다.
- 자원 확인: `cpu-scheduler/status.py`(P 평균 9 %, 예약 없음) → 1프로세스.
- 즉석 스크립트(scratchpad `verif-e3x-post\`, 보고 전에 폴더째 지움)
  - `explore.py`: 열 값 분포
  - `recount.py`: 규칙 §3-2–§3-4 독립 구현 → M*·층·가설 7개 추정량·Satterthwaite·기본 구간·결정, 그리고 [J] 대조. 36 s.
  - `rej.py`: int16 거부의 on/off 구조
  - `ident.py`: 두 단계 run_identity, 봉인 sha·blob·src tree를 `git rev-parse bc57ef1:`·`hash-object`와 대조, plan 표 sha, outputs·units sha
  - `disc.py`: 공개 1–28 원문 대조(규칙 :514-576 ↔ 판정서 :391-453, diff 0줄)
  - `sumchk.py`: precheck_r1 sha 목록 44/44
- tools/verif
  - `baseline_check sums` 1회차 증거: mismatch 0, missing 0, `cited_untracked` 11(미커밋, 정상)
  - `baseline_check sums` prerun_r1 `--rev bc57ef1`: 종료 0
  - `timeline`: `rule_after_results` 0. 규칙 커밋 11-20 17:48 < precheck 11-26 02:27 < uninj 12:58:04 < injected 13:00:09 < judge 13:49:06 < 판정서 작성자가 결과를 처음 연 시각 13:52
  - `judgement_checklist`: items 146, found 75, missing 0, ambiguous 71. §6 1–10항은 직접 대조했다.
  - `seal_check --quoted --hash-files`: 미해결 43건은 모두 git blob·tree, 가중치·입력·설정 sha, 저장소 밖 ps1·hand-back sha다. ps1은 직접 해시해 일치했고, 나머지는 run_identity 대조로 확인했다. ambiguous 8건은 w1/w2·사본 동일 파일이다.
  - `ref_check --exclude …`: missing 34건은 모두 맥락 속 짧은 이름이라 오탐이다. untracked 24건은 runs/·미커밋 파일이다.
  - `gate_scan judge.py`: `all_without_count_guard` `:796`은 오탐이다(FOLDS가 비지 않는 상수). `min_default` `:1892`는 보고 전용이다.
  - `log_errors logs_P`: 종료 0, `errors N` 줄 없음.
  - `ps1_lint` 종료 0, `placeholder_scan` `<LogDir>` 주석 1건(오탐).
- git 조회: `ls-files --eol`, `check-attr`, `check-ignore -v`, `log`, `diff --stat 50a28a3 bc57ef1`(규칙·봉인·analysis/detection·src 변경 없음), `rev-parse`·`hash-object`
- 부산물: 없음. `analysis/detection/__pycache__`는 2026-11-07 생성으로, 이 감사와 무관하다.
- 에이전트 메모리: `audit_pitfalls.md`에 함정 1항(결정의 명시 인자 ↔ 러너 기본값, 끔 거부 사유 상속, 늦게 만든 러너 미감사, 대상 행 정의)을 더했고 `MEMORY.md` 색인에 반영했다.
- 원본 불변: 확인 — 원본 쓰기 0회, 원본 디렉터리 쓰기 0건 (sha m33_e3x_plan.md 89fff477…, m33_e3x_seal2_r1.json 81e57d3e…, m33_e3x_seal2.json 94c671ed…, m33_e3x_judgment.md 95311c10…, m33_e3x_judge.py 7e6a110a…, m33_e3x.py ac00f412…, m33_e3x_place.py 898dd80b…, m33_e3x_prerun_audit.md 0dc32379…, m33_e3x_prerun_audit_r1.md 640321e4…, band_records.csv f44500ed…, hyperbola_records.csv eab4428b…, progress.json a1d9a96c…, realizations.csv 8d5483b3…, rejections.csv 46d5d1b4…, stage_settings.json ad74c229…, summary.json 358ad13f…, truths.csv 2a0eee85…, progress.json 0c61f44b…, stage_settings.json f201535b…, summary.json db879e82…, uninjected001_records.csv a65cef0e…, judgment.json a7d492e2…, units_HB1.csv daec62a0…, units_HB1b.csv 5584e3be…, units_HB2.csv a846b0ae…, units_HH1_negative_control.csv 7eb1e9b9…, units_HH1.csv a465ac40…, units_HH2.csv 5b0269d6…, units_HH3.csv bbad58d2…, 디렉터리 목록 동일, ~$ 임시 파일 없음)
- 사용한 스킬: 없음

## 판정서 수정 목록 (담당)
1. (gpr-processing-expert 또는 메인) §2 "실행 중 사용자 결정" 1과 §9-2: 결정 1의 `--uninjected` 명시가 러너 명령에 없었고, 판정 도구가 기본값(성공한 `e3x_retry/uninjected001`과 같음)으로 돌았다는 공개 문장을 넣는다(발견 1).
2. (메인) [RL] 사본 경로 `docs/processing/evidence/m33_e3x_retry_runner/`(SUMS)를 만들고 판정서 `:13` 정의에 함께 적는다(발견 2).
3. (메인) §2 `:81`에 1회차 injected 끝 시각을 보충한다(발견 3).
4. (선택, gpr-processing-expert) §9-2 해석 목록에 두 줄을 더한다: HB1b 대상 정의 차이(발견 5), 이동량 민감도의 위치별 최대 \|n₂\| 기준(발견 7). §11에 재집계 스크립트 docstring과 실제 재현 여부의 차를 한 줄 적는다(발견 6).
