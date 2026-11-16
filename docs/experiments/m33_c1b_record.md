# M3-3 C1b 기록 — R3 주입 표 고정·P4 `mde_pp`·상류 sha (injected 전)

- 상태: **C1b 고정(2026-11-16)**. 이 문서는 R3 injected 실행 **전에** 커밋한다. 규칙은 바꾸지 않으며 봉인 기록 [`m33_c1_seal.md`](m33_c1_seal.md)(C1 봉인 68f4719, C1 코드 커밋 f4ad45e)의 값도 고치지 않는다. 메인 세션 작성(값은 각 담당 에이전트 리포트에서 옮김).
- 근거 규칙: 러너 명세 [`m33_runner_spec.md`](m33_runner_spec.md) §1(확정 순서 plan → C1b 커밋 → injected)·§3-2(`mde_pp`는 C1b plan 뒤·injected 전에 계산), 계획 [`m33_cross_domain_plan.md`](m33_cross_domain_plan.md) §4 C1b 행.

## 1. C1b 주입 표 고정

| 항목 | 값 |
|---|---|
| C1b 기록 | `analysis/detection/m33_r3_c1b.json` = `runs/eval/m33/r3/plan/c1b_record_draft.json` 바이트 그대로, sha256 `65eec135dc5d8ed7874b63d734d84de9f840374c38a07642571caed51caddbb8` |
| 고정 대상(기록 안) | `injection_table.csv`(주입 4,600), `positions.csv`, `exclusions.csv`의 LF sha256 |
| plan_report.json | sha256 `0387d78ed77ed3dbb068ea9bf539e7f2a6fe286ce3eb13af6780b25d71b2a046`(raw = LF) — injected의 `run_identity.inputs_sha256.plan_report`와 판정서에서 대조 |
| plan 실행 | HEAD 68f4719, `--injectant band --stage plan --workers 3`, status ok, judgeable, provisional false, 233측선, 1,653.7 s(시작 약 09:56:31) |
| apex 후보 표 | `docs/processing/fourtu_apex_candidates_v1.csv` LF sha `b348a2ff…`(plan `inputs_sha256_lf.apex_candidates`) — C1b 커밋은 이 파일을 건드리지 않음, 판정서에서 plan·injected 값 대조 |

C1 봉인 기록 §9의 "봉인 뒤 `analysis/`는 f4ad45e와 같아야 한다"에 대해: f4ad45e 뒤 `analysis/` 차이는 봉인 기록 §6·계획 §4·러너 상수 `R3_C1B_RECORD`가 지정한 `analysis/detection/m33_r3_c1b.json` 한 파일뿐이다(지정된 예외).

verif C1b 커밋 전 확인(2026-11-16): 차단 0. C1b 기록 sha 3개 독립 재계산 일치, plan 판정 가능(정체성·봉인 단언·code_blobs 14개 = 봉인 기록 §1), 상류 IDS uninjected·R2 judgeable, `read_c1b_inputs` 키·기본 경로 일치, injected 끝 시점 트리 검사는 자기 실행 시작·끝만 비교(C1b 커밋 뒤 HEAD 변경 허용).

## 2. 상류 박스 sha 목록

injected는 IDS uninjected와 R2의 `fold*/boxes.csv`를 다시 읽지만(유령 짝·P4 범위) 그 sha는 injected `inputs_sha256`에도 plan_report에도 없다(verif 기록). 그래서 injected 전 상태를 고정한다.

- 파일: [`m33_r3_upstream_hashes.csv`](m33_r3_upstream_hashes.csv) — `runs/eval/m33/r2`(798개)·`runs/eval/m33/r3/uninjected_ids`(68개) 파일별 sha256·크기·mtime, 합 866행, 파일 sha256 `d837db07026465b3e9bfdb0069d4b9769f749945d09a84e511954f1061f92ae9`. 바이트 해시만 쟀고 내용은 열지 않았다.
- P4 판정서 단계에서 이 목록과 다시 대조한다.

## 3. P4 `mde_pp` (stats-design-expert, injected 전 계산) [추정]

입력은 결과와 무관한 plan 구조값뿐이다(`plan_report.json`의 `p4_plan`, `positions.csv` 구조 열). 오염 없음(stats-design 보고).

- P4 범위: 400 MHz 136위치(lt033 108, 033_05 28) → `stratum_mode` pooled, 50측선, 길이 L 40·64·100·150이 79·33·14·10개. 노출(S3) 0, PP-c 창 15.
- 모형: 개수 모형 시뮬레이션, 셀 안 길이 가중, 측선 군집(영향 측선 기대 수 m), 손실 단위 3종(실현-fold·실현·위치), 10,000회, 시드 20261114. δ = 0이면 통과 확률 1(PP는 재현율을 올릴 수 없음).

| 영향 측선 기대 수 m | MDE80 (%p), 실현-fold · 실현 · 위치 |
|---|---|
| 50(군집 없음) | 1.15 · 1.35 · 2.40 |
| 25 | 1.30 · 1.45 · 2.53 |
| 10 | 1.70 · 1.80 · 2.80 |
| 5 | 2.40 · 2.52 · 3.51 |
| 3 | 3.45 · 3.55 · 4.35 (최대 가능 6 %p) |
| 2 | 도달 불가 |

- 대표값 1.8 %p(측선 10개, 실현 단위). δ = 2 %p에서 통과 확률 0.00–0.49.
- 길이 가중으로 MDE가 같은 길이 가정보다 0–0.85 %p 커짐. Kish 유효 위치 106.6/136, 측선 무게 기준 유효 측선 20.2/50.
- 개수 환산(합친 층): 실현-fold 하나 손실 = −2.5·(L/L̄)/136 %p(L̄ 60.1), 문턱 −1.0 %p = L̄ 환산 실현-fold 54.4개(0.4·n_p). 위치 하나를 모든 실현-fold에서 잃으면 L 40: 0.49, 64: 0.78, 100: 1.22, 150: 1.84 %p — L 100·150 위치 하나만으로 문턱을 넘는다. 50측선 중 24측선은 그 측선 하나를 완전히 잃는 것만으로 문턱을 넘는다.
- 해석: 노출 0이므로 상태는 `no_effect` 또는 `acted_unexposed`이고, 이 MDE는 `acted_unexposed`에서 뜻을 갖는다. 손실은 `recall_pre`를 넘을 수 없으므로 `recall_pre` < δ인 효과는 도달할 수 없다(계획 공개 사항 9).
- 판정서 문구(그대로 인용): "`mde_pp`(합친 층, 400 MHz 136위치·50측선, 셀 안 길이 가중, 개수 모형 시뮬레이션 10,000회·시드 20261114, 결과 무관 구조값만 사용) [추정]: 검정력 80 % 최소 효과는 손실이 측선 전반에 퍼지면 1.2–2.4 %p, 측선 5–10개에 몰리면 1.7–3.5 %p, 3개면 3.5–4.4 %p, 2개면 도달 불가(대표값 1.8 %p). 보고 전용이고 판정에 쓰지 않는다."
- 이 값은 러너 명세의 이전 합성 범위(MDE80 1.5–7.25 %p, "−2 %p여도 통과 0.1–0.6")를 이 위치 구조로 대체하는 보고값이다. 판정 규칙(`recall_delta` ≥ −1.0 %p − 1e-9)은 그대로다.
- 산출물(저장소 밖 scratchpad, sha256): `p4_mde_sim.py` `743c04fa890da224075fe349c46fb684a7b3fd58281fe63dbe62d9e21f29d9b2`, `p4_mde_results.json` `97cd9c74305fc25ba6fd82871dda19f38e3d3a6bfa1ee2816a4111281e8c8a39`, `p4_mde_table.md` `2d26af4b0609fda5e029391ef39816375103b191fe0a740c3c07774384aa6796`.

## 4. injected 실행 조건 (verif 권고)

- 확정 명령 `--injectant band --stage injected`(금지 인자 없음, 기본 `--c1b-record`), `runs/eval/m33/r3/injected` 없음 확인, `git status --porcelain --untracked-files=no` 비어 있음.
- 실행 중 커밋·추적 파일 편집·프로젝트 에이전트 호출 금지.
- 실행 직후 키만 확인: `inputs_sha256.plan_report` = 0387d78e…, C1b 기록 sha, `judgment.judgeable`, `tree_changes` [].
