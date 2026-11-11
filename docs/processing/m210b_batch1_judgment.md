# M2-10 (b) 묶음 1 판정 — 선정기 검출 정규화 바닥 (gprMax S4N·S6P) — 확정본

**판정(확정, 사용자 결정 (가) 2026-11-11): 결정 순위 5 — 후보 전부 불충분. 현행 바닥 규칙을 유지한다.**
- R(결함 재현) 성립.
- B_full·B_min·B_noise·C_none은 F5만 실패했다.
- A_skip1은 F1·F2·F5가 실패했다.
- 결과를 본 뒤 기준·셈법·결정표는 바꾸지 않았다.
- F5의 비교 대상을 정의한 v2를 봉인한 뒤, 새 합성으로 다시 시험한다([`m210b_selector_floor_v2.md`](m210b_selector_floor_v2.md), 초안).

- 판정: 2026-11-11, gpr-processing-expert. 판정은 하네스 출력만으로 했다. 4TU 정답은 쓰지 않았다.
- 기준: 봉인 커밋 `bc9a474`의 [`m210b_selector_floor.md`](m210b_selector_floor.md) v1.4 §4-2(판정 항목 M0·R·F1–F6·F4b, R 정의 Q·G·C·e, 결정 순위 1–9).
- 결과 전 결정: [`m210b_harness_interpretation.md`](m210b_harness_interpretation.md)(커밋 `7e843d6`)의 D1·D2와 P-F1–P-F3. **D1·D2·P-F1–P-F3는 봉인 뒤·결과 전 결정이다.** 결과를 본 뒤 기준·셈법·결정표를 바꾸지 않았다.
- 계산: `analysis/processing/m210b_batch1_judge.py` → `docs/processing/m210b_batch1_judgment.json`(gpr-dev, E코어 25, BelowNormal, BLAS 1, 난수 없음).

## 0. 판정 요약

| 항목 | 결과 |
|---|---|
| P-F1 단언 | **모두 통과**(아래 §1) → 순위 1 아님 |
| 순위 2(S4N ρ 20 수락 단계 성공률 < 0.5) | 해당 없음(S4N ρ 20 current Δ 0: D12_S010 0.9, D15_S010 1.0, D15_S030 0.8, D15_S040 0.8) |
| 순위 3(자격 Q 없음) | 해당 없음(16개 (c, π) 모두 Q) |
| **R(결함 재현)** | **성립.** S6P_D15_S040, ρ 10·20, 두 짝 모두. current 수락 단계 성공률 Δ −1.0·0·+1.0 = **0/10**, Δ +2.35 = **9/10**(e = 0.9). Q(ρ_real 최소 18.5·36.9), G(Λ 37.06·32.38 > 10, 바닥에 걸린 시드 10/10), C(S4N_D15_S040 차 0) |
| 후보별(순위 4/5) | **모두 순위 5 불충분(확정).** A_skip1은 F1·F2·F5 실패다. B_full·B_min·B_noise·C_none은 F1–F4·F4b 통과, F5만 실패다(봉인 셈법; §4 사용자 결정 (가)) |
| P-F3 민감도(Λ_cal·ρ_cal) | 같은 결론(R 성립, 후보별 같은 실패 항목). G가 추가로 S6P_D15_S030 Δ 0에서 성립(Λ_cal 10.96)하지만 R 짝은 여전히 S040뿐 |
| 사전 예상(§4-2-7) | **맞음.** 봉인 v1.4 §4-2-7이 명시한 사전 예상은 Δ 0·−1.0에서 잃고 +2.35에서 잡는 것이었다. current가 그대로였다. Δ +1.0은 명시된 예상이 아니었고, 결과는 0/10(잃음)이다(점검 Λ_est(+1) 20.1 > 10과 같은 방향, 보고만) |

## 1. 실행 정체성과 P-F1

| 항목 | 값 |
|---|---|
| 봉인 커밋 | `bc9a474452c29b9644cfe0b7a60a89c90189f758`(2026-11-11 19:29:19 +09:00) |
| 봉인 명세 blob | `docs/processing/m210b_selector_floor.md` = `64004278548f4091f0801035745af03fca1541ec` |
| 해석 문서 blob(7e843d6) | `docs/processing/m210b_harness_interpretation.md` = `1598818a6a76128a722e1e55da411a97b56bf719` |
| 봉인 파일 | 44개, 모두 봉인 blob과 작업 트리 blob 일치(`run_info.json` `seal.sealed_files`; 경로·blob 표는 아래). 판정 후 감사에서 `bc9a474` 트리와 3중 대조해 불일치 0 |
| 실행 커밋 | HEAD `7e843d6a2b9c539a4af07fc335020c850227688f`, 실행 전후 dirty 0(`pre_run.txt`, `post_run.txt`), 입력 npy·json 36개 |
| 시각 순서 | 봉인 19:29:19 < gprMax 전체 실행 19:32–20:18 < 하네스 시작 20:27:02(run_info; 메인 세션 기록 20:26:54) < 종료 20:39:23 |
| `runs.csv` SHA-256 | `7470514950902d8fd8815e5c7e099d04199987e027fd3698a1fd523b205fd187`(run_info 기록 = 판정 시 재계산) |
| 환경 | MKL_CBWR COMPATIBLE, numpy 2.5.4, Python 3.13.18. 4TU 래퍼 회귀 두 csv 바이트 동일 |

봉인 파일 44개 — `bc9a474` 시점 blob(`run_info.json` `seal.sealed_files.<경로>.blob_at_seal`):

| 경로 | blob(SHA-1, LF 기준) |
|---|---|
| `analysis/processing/m210b_floor_proto.py` | `860cd3ada5b898e611ea24b3e16d4725105a7776` |
| `analysis/processing/m210b_floor_rules.py` | `bad0f95cdcebe3d193fb7aad6511dbbafcac8327` |
| `analysis/processing/m210b_lambda_pilot.py` | `aee1b0a24ffde10ef37c3f1c65a43ca52347d4aa` |
| `analysis/synthetic/s4s6/inputs/BG_N_S005.in` | `0375e7039dd4c91badf58bf315e6e55e2e4d1b95` |
| `analysis/synthetic/s4s6/inputs/BG_N_S010.in` | `1d249114f069ba0067efc6c79a2a814052e92130` |
| `analysis/synthetic/s4s6/inputs/BG_N_S030.in` | `ff07589b3827ac7043f6e4816eb5185aaf6e8828` |
| `analysis/synthetic/s4s6/inputs/BG_N_S040.in` | `e8c317eb3c10e0f36f0506e20536c1c1e4c5fcb6` |
| `analysis/synthetic/s4s6/inputs/BG_PF_S010.in` | `0577b1897c9809761351ea5920097deb75ac9831` |
| `analysis/synthetic/s4s6/inputs/BG_PF_S030.in` | `a54c95d7ecfcee21fd1c8dbd1944a5ee2f9e90d1` |
| `analysis/synthetic/s4s6/inputs/BG_PF_S040.in` | `194a8b238a595d606c8d2e33f63adab0058bcf97` |
| `analysis/synthetic/s4s6/inputs/FS.in` | `e826698de4e44c0a357e722802dde5121629d2df` |
| `analysis/synthetic/s4s6/inputs/S4N_D12_S005.in` | `6b15f781bb1ef425f932e4a07372b310fb31e4af` |
| `analysis/synthetic/s4s6/inputs/S4N_D12_S010.in` | `c5c02a274190f7d94fd4b96198b588924444fb24` |
| `analysis/synthetic/s4s6/inputs/S4N_D12_S030.in` | `bb56632450fd07fd17608c0291c45d1f4523643e` |
| `analysis/synthetic/s4s6/inputs/S4N_D15_S005.in` | `2eb4f0c91b2bf204674a2ac5f29e5e9ab4758e7a` |
| `analysis/synthetic/s4s6/inputs/S4N_D15_S010.in` | `ad36b3e971d36088c9f8c63ecbc7d2f974a9fe35` |
| `analysis/synthetic/s4s6/inputs/S4N_D15_S030.in` | `23faa51021c361bc9dc661018cabf1b2529b17c0` |
| `analysis/synthetic/s4s6/inputs/S4N_D15_S040.in` | `d84eb3099a22fe3adc9bf5d4bc32609e5e3b64e3` |
| `analysis/synthetic/s4s6/inputs/S4S_D12_S005.in` | `77b7322f5c3191891af894b98e61504ebf457d25` |
| `analysis/synthetic/s4s6/inputs/S4S_D12_S010.in` | `5a1d5a55b44522ec4fe7cef7a9021259999b1d0c` |
| `analysis/synthetic/s4s6/inputs/S4S_D12_S030.in` | `6ebdaa966c04bfbf686a26c222f929f0fdadc16d` |
| `analysis/synthetic/s4s6/inputs/S4S_D15_S005.in` | `628c11300bd6e576f28a3abfbaae186fd0f32db1` |
| `analysis/synthetic/s4s6/inputs/S4S_D15_S010.in` | `90aad372bd44ddf7933d2e7ff46baaa69e2f49b3` |
| `analysis/synthetic/s4s6/inputs/S4S_D15_S030.in` | `e8a8fc532e777b3e2bf25d56aa4abcbc5eab3f17` |
| `analysis/synthetic/s4s6/inputs/S6P_D12_S010.in` | `9c28520312dbd954637909fb6cc287bc5695eefd` |
| `analysis/synthetic/s4s6/inputs/S6P_D15_S010.in` | `dbcd7758aa2d552bfac501aae50db8fe555f4357` |
| `analysis/synthetic/s4s6/inputs/S6P_D15_S030.in` | `46931dc4b75a65700fca7fa7cdb1f2dca58f9de8` |
| `analysis/synthetic/s4s6/inputs/S6P_D15_S040.in` | `fcbbe53cbcc351872d5f8d34f4267628bcdadea5` |
| `analysis/synthetic/s4s6/inputs/S6P_NP_S010.in` | `0f7a85ddf3a7e75ba676d0978df305812a44045d` |
| `analysis/synthetic/s4s6/models_s4s6.json` | `73d8bdce152577359cfe30a51aeed2e1fa24d9fa` |
| `analysis/synthetic/s4s6/pilot/L_S6P_NP_S040_T15.in` | `887c18142d76b293b07d72c7b5ff1002df3bd043` |
| `analysis/synthetic/s4s6/pilot/P_S4N_D15_S030.in` | `8e4ca906468692776bca137baa4b7d509284bcaa` |
| `analysis/synthetic/s4s6/pilot/P_S4N_D15_S040.in` | `65337f98a2631b63bb396f0d1af4e8da5fcd4dcb` |
| `analysis/synthetic/s4s6/pilot/P_S6P_D15_S030.in` | `66c299964a86ce339f1bf5cede7bb8aa2f1782a7` |
| `analysis/synthetic/s4s6/pilot/P_S6P_D15_S040.in` | `690bcd669cc6a7a7383624ca379c7584bc2ca9d8` |
| `analysis/synthetic/s4s6/pilot/P_S6P_NP_S040.in` | `5f3f70dc8b85fbe4a2a8ad3329e56fa7790fac01` |
| `analysis/synthetic/s4s6/pilot/lambda_S6P_D15_S040.json` | `8c6018e9b9370f0da3af17d1819833b0caeac3c1` |
| `analysis/synthetic/s4s6/pilot/pilot_check.json` | `fa75f7b4477353a0696b76b42dff0d05a24fba34` |
| `analysis/synthetic/s4s6/pilot/pilot_envelopes.png` | `ca0c0499756ca6b237e2ee8f191c2f86306f4d1d` |
| `analysis/synthetic/s4s6/run_s4s6.cmd` | `b561467dbd7346bc1f5e74bc24592a106ec13870` |
| `docs/processing/m210b_a0_S4N_D15_S030.json` | `ce4f33b5ae20ff9400ee3aaf82157d9d1f0cfc70` |
| `docs/processing/m210b_a0_S6P_D15_S030.json` | `f6ff56e0e43432049b5e386aea03faf2c5e27db1` |
| `docs/processing/m210b_floor_proto.json` | `a6e9db068d28e788677760fadd7b4ed23ef95b33` |
| `docs/processing/m210b_selector_floor.md` | `64004278548f4091f0801035745af03fca1541ec` |

P-F1 단언(모두 참):
- `n_runs` 4,740(행 4,740 = run_info), `n_runs_without_zero` 0
- 판정 행(`dt_ns` 0.098 ∧ `ramp_noise` 0) 4,320. 칸 432개(모델 9 × ρ 2 × 규칙 6 × Δ 4)가 빠짐·남음 없이 칸마다 시드 정확히 10개
- F4b 행(`ramp_noise` 1) 24칸 × 10시드, 모델은 S6P_NP_S010뿐
- 관 있는 8모델 모두 `L_noise_free_current` 키 4개, `apex_trace` ≠ null
- 빈칸은 불충족으로 셌다. 판정 행에서 판정 열이 빈 관 모델 행은 없었다

## 2. R — Q·G·C·e (current, 수락 단계, D1·D2 기준)

| S6P 조건 | ρ | π(Δe→Δl) | 성공률 e→l | e | Q(ρ_real 최소) | Λ(Δe) | 바닥 시드 | G | S4N 차 | C | R 짝 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S6P_D12_S010 | 10·20 | 두 짝 | 0.8→0.8 / 0.9→0.9 | 0 | 참(11.7 / 23.5) | 0.12 / 0.11 | 0 | 거짓 | 0 | 참 | 아니오 |
| S6P_D15_S010 | 10·20 | 두 짝 | 0.5→0.5 / 0.8→0.8 | 0 | 참(11.8 / 23.4) | 0.21 / 0.19 | 0 | 거짓 | 0 | 참 | 아니오 |
| S6P_D15_S030 | 10 | 두 짝 | 0.5→0.8 | **0.3** | 참(16.0) | 6.68 / 5.83 | 10 | 거짓(Λ ≤ 10) | 0 | 참 | 아니오 |
| S6P_D15_S030 | 20 | 두 짝 | 0.9→0.9 | 0 | 참(31.8) | 6.68 / 5.83 | 10 | 거짓 | 0 | 참 | 아니오 |
| **S6P_D15_S040** | **10** | (0→+2.35), (−1→+2.35) | **0.0→0.9** | **0.9** | 참(18.5) | **37.06 / 32.38** | **10** | **참** | 0 | 참 | **예** |
| **S6P_D15_S040** | **20** | 두 짝 | **0.0→0.9** | **0.9** | 참(36.9) | 37.06 / 32.38 | 10 | 참 | 0 | 참 | **예** |

- R 성립(Q∧G∧C∧e ≥ 0.5인 (c, π) 4개) → 결정은 순위 4/5(후보별)로 간다.
- S6P_D15_S030 ρ 10은 e 0.3, 곧 약한 결합이다. 그러나 Λ_self 6.68 ≤ 10이라 영역 G 밖이고 R에 들지 않는다. P-F3에서는 Λ_cal 10.96으로 G가 성립하지만 e 0.3 < 0.5라 R 짝이 아니다.

## 3. 후보별 F1–F5·F4b (수락 단계 성공률; 기준 = current)

| 규칙 | F1 분리 | F2 회복(S040) | F3 대조 무손실 | F4 거짓 | F4b 경사 잡음 | F5 위치 불변 | 봉인 셈법 결과 |
|---|---|---|---|---|---|---|---|
| **B_full** | 통과(실패 0) | 통과(0.9 ≥ 0.9 − 0.1) | 통과(최소 여유 S4N_D15_S040 ρ 20: 0.7 대 0.8, 경계) | 통과 | 통과(깊은 거짓 0.0 대 0.7) | **실패**: 짝 337 중 58(|Δtrace| 최대 5, |Δsample| 최대 1.8) | 5 불충분(F5) |
| **B_min** | 통과 | 통과 | 통과(같은 경계) | 통과 | 통과(0.3 대 0.7) | **실패**: 337 중 54 | 5 불충분(F5) |
| B_noise | 통과 | 통과 | 통과(같은 경계) | 통과 | 통과(0.0 대 0.7) | **실패**: 337 중 58 | 5 불충분(F5) |
| C_none | 통과 | 통과 | 통과 | 통과 | 통과(0.7 대 0.7) | **실패**: 356 중 21 | 5 불충분(F5) |
| A_skip1 | **실패 6**(S6P_D15_S040 ρ 10·20 Δ +1·+2.35 차 0.9, S6P_D15_S030 ρ 10 차 0.3) | **실패 4**(S040 Δe 0/10) | 통과 | 통과 | 통과 | 실패 5 | **5 불충분**(F1·F2, F5와 무관) |

- 성공률 표(수락 단계): S6P_D15_S040은 B_full·B_min·B_noise·C_none 모두 Δ 네 값에서 0.9다(ρ 10·20). current는 0/0/0/0.9, A_skip1은 0/0/0.9/0.9다. 전체 표는 json `rates`에 있다.
- F4b는 Δ 0으로 판정했고(F4와 같은 Δ), 네 Δ 모두 같은 결론이다(`F4b_delta_dependent` 거짓).
- F6(보고):
  - kept 단계 성공률(관 있는 8모델 판정 행 평균): current 0.570, B_full·B_min·B_noise 0.675, C_none 0.656, A_skip1 0.600
  - 사유: `early_lobe_nms` 0건(모든 모델 $t_e$ 없음), `cap6` 1건(B_full), `other_nms` 7–40
  - 상한 300 도달 0행
  - 바닥 $F$(Δ 0 중앙): S6P_NP_S010에서 current 0.083, B_full 1.92, B_min 1.71. S6P_D15_S040에서 current 0.0732, B_full 0.0077
  - 원 Δt 보조 실행(180행, current·B_full, ρ 10, Δ 0)은 `runs.csv`에만 있고 판정에 쓰지 않았다(이전 판의 "json에 있다"는 틀린 서술이었다). 판정 json `F6_original_dt_report`에 요약을 더했다.
    - 검출 피크 수 중앙(원 격자): S4N_D12_S010 B_full 18; S4N_D12_S010 current 262; S4N_D15_S010 B_full 22; S4N_D15_S010 current 263; S4N_D15_S030 B_full 18; S4N_D15_S030 current 266; S4N_D15_S040 B_full 17; S4N_D15_S040 current 261; S6P_D12_S010 B_full 21; S6P_D12_S010 current 262; S6P_D15_S010 B_full 30; S6P_D15_S010 current 264; S6P_D15_S030 B_full 38; S6P_D15_S030 current 21; S6P_D15_S040 B_full 87; S6P_D15_S040 current 11; S6P_NP_S010 B_full 11; S6P_NP_S010 current 258
    - 원 격자에서 current는 관 없는 줄과 클러터 없는 줄에서 상한 300에 걸렸다(10/10행). B_full은 0행이다. 샘플 단위 상수가 4TU dt 기준이라 원 격자 피크 수는 4TU 영역과 다르다(봉인 규약 2의 재표본 근거와 같음)
- 보고만(판정 무관): S6P_D15_S030 ρ 10에서 B_full·B_min·B_noise는 0.5로 늦은 뮤트 current(0.8)와 C_none(0.8)보다 낮다. 그 조건은 G 밖이라 F2 대상이 아니다. 잡음 기준 바닥의 깊은 쪽 감도 비용(봉인 §3-3에서 예상한 위험)과 같은 방향의 신호다.

## 4. F5와 사용자 결정 (가)

- 봉인 F5는 "두 규칙 모두 꼭짓점 후보가 있는 시드에서 kept 꼭짓점 차 |Δ트레이스| ≤ 1, |Δ샘플| ≤ 0.5"다. 하네스 열 `kept_apex_trace`·`kept_apex_sample`은 "창 안 kept 가운데 순위가 가장 높은 것"이다(결과 전 커밋 3500121의 열 정의). 해석 확인 문서(D1·D2, P-F1–P-F3)는 이 선택을 따로 다루지 않았다.
- 창(±3 트레이스·±1 ns) 안에 kept가 여럿일 수 있다. 그러면 규칙에 따라 "가장 높은 순위"가 다른 후보로 바뀔 수 있다. 순위 키의 SNR이 바닥에 따라 달라지기 때문이다. 같은 꼭짓점을 다른 시작 피크에서 능선 이동해 끝점이 달라지는 경우도 있다.
- **진단(판정 아님, 이미 본 결과에서 셈; 판정 json `F5_diagnostic`)**:
  - B_full F5 실패 58쌍은 16개 실현(모델·ρ·시드)에서 Δ를 가리지 않고 반복된다. 클러터 없는 S4N(바닥이 걸리지 않는 곳)에서도 나온다.
  - 그 58쌍에서 B_full의 후보가 참 꼭짓점 시각 $t_{a0}$에 더 가깝다. 시각 차 중앙은 0.069 ns 대 0.147 ns이고, B_full이 더 가까운 쌍 48 대 10이다. 트레이스 차 중앙은 둘 다 1이다.
  - 곧 위치 변화는 "바닥이 검출 피크 집합을 바꿔 능선 이동의 시작점이 달라진 것"이고, 참 꼭짓점에서 멀어지는 쪽이 아니다.
- **F5 범위(공개)**: 판정의 F5는 관 있는 8모델 × Δ 4 × ρ 2에서 같은 Δ끼리 짝지은 쌍 전체로 셌다. 봉인 문구는 범위를 정하지 않았고, 판정 코드가 고른 읽기다. 글자에 가장 가깝고("두 규칙 모두 꼭짓점 후보가 있는 시드", 조건 제한 없음) 보수적인 읽기다(짝이 많을수록 실패가 쉽다).
- **범위별 F5 실패 수(판정 아님, 판정 json `F5_by_scope_report`)**:

| 규칙 | 전체(판정 범위) | R 조건(S6P_D15_S040) | S4N만 | G 밖 S6P |
|---|---|---|---|---|
| B_full | 58/337 | 0/17 | 28/168 | 30/152 |
| B_min | 54/337 | 0/17 | 28/168 | 26/152 |
| A_skip1 | 5/362 | 0/17 | 0/180 | 5/165 |
| C_none | 21/356 | 3/17 | 0/180 | 18/159 |
| B_noise | 58/337 | 0/17 | 28/168 | 30/152 |

  B_full·B_min·B_noise의 실패는 모두 S4N과 G 밖 S6P에서 나왔다. R 조건만 보면 0/17이다. C_none은 R 조건에서도 3/17 실패했다. **사용자는 이 사실을 본 뒤에도 (가)를 유지했다(2026-11-11).**
- **시간 순서(공개)**: 판정 코드 `m210b_batch1_judge.py`는 `runs.csv`가 생긴 뒤 작성됐다. 그때 열 구조만 확인했고, 값은 코드 실행으로 처음 보았다. F5 범위는 그 코드가 선택한 것이다. 결과 전 문서에 범위가 적혀 있지 않았다. 범위 선택이 결과를 보고 한 것이 아님을 확인할 길은 이 진술뿐이다. v2에서는 판정 스크립트를 결과 전에 커밋한다(v2 §8-2).
- 판정자가 제시한 선택지는 셋이었다: (가) 봉인 셈법 그대로 순위 5, (나) 결과 뒤 "같은 후보 위치"로 다시 셈(예외), (다) 보류.
- **사용자 결정 (가)(2026-11-11, 판정 뒤)**: 봉인 문구와 결과 전 하네스 열 정의 그대로 F5 실패를 받아들인다.
  - B_full·B_min·B_noise·C_none은 순위 5다. U1(합성 통과 후보로 교체)의 근거가 없으므로 현행 바닥 규칙을 유지한다.
  - (나)의 다시 셈은 하지 않는다. 위 진단은 v2 설계 근거로만 쓴다.
- **한계**:
  - F5가 실패한 이유는 "바닥이 꼭짓점 위치를 나쁘게 했다"가 아니다. 비교 대상이 규칙마다 다른 후보가 될 수 있는 정의 때문이다.
  - 그래서 이 판정은 B 계열이 나쁘다는 증거가 아니다. **봉인 기준으로 채택을 입증하지 못했다**는 결과다.
  - 이 문서는 규칙 교체를 권하지도 금하지도 않는다.
- **교훈**: "위치 불변" 같은 비교 기준은 후보가 여럿일 때 **무엇과 무엇을 비교하는지**를 결과 전에 정의해야 한다. 이번에는 결과 전 하네스 열 정의(순위 최상위)가 그 역할을 했지만 검토 대상에서 빠졌다. 해석 확인 문서도 같은 빠짐이 있었다. v2에서 F5를 정의한다.

## 5. P-F3 민감도 (봉인 글자대로: Λ_cal = L/A0_cal, ρ_cal)

- Λ_cal은 S6P_D15_S040 Δ 0에서 71.79, Δ −1.0에서 62.73이다. S6P_D15_S030 Δ 0에서는 10.96이다(S6P/S4N A0 비 1.94·1.64배만큼 크다).
- R은 같고 짝도 같다(S040 ρ 10·20 × 두 짝). S030 Δ 0은 G가 성립하지만 e 0.3·0이라 R 짝이 아니다.
- 후보별 결과는 모두 5이고, 실패 항목은 D1·D2 판정과 같다. 결론은 바뀌지 않는다.

## 6. M0 (보고)

| S6P | Λ_self(Δ −1 / 0 / +1 / +2.35) | Λ_cal(같은 순서) | $t_{a0}$ (ns) |
|---|---|---|---|
| D12_S010 | 0.11 / 0.12 / 0.07 / 0.06 | 0.13 / 0.14 / 0.08 / 0.07 | 26.85 |
| D15_S010 | 0.19 / 0.21 / 0.12 / 0.08 | 0.22 / 0.25 / 0.14 / 0.10 | 32.83 |
| D15_S030 | 5.83 / 6.68 / 3.61 / 1.42 | 9.58 / 10.96 / 5.94 / 2.33 | 32.83 |
| D15_S040 | 32.38 / 37.06 / 20.07 / 7.95 | 62.73 / 71.79 / 38.88 / 15.40 | 32.73 |

- 점검 실행 추정 Λ_est(S040, Δ −1 / 0 / +1 / +2.35) 32.4 / 37.0 / 20.1 / 7.9와 전체 실행 Λ_self 32.38 / 37.06 / 20.07 / 7.95가 같다.
잡음 실행 $\arg\max\ell$(current, ρ 10, 시드 10개의 10/50/90 %, 뮤트 아래 주기 수; json `M0.<id>.argmax_periods_noisy_current`):

| S6P | Δ −1.0 | Δ 0 | Δ +1.0 | Δ +2.35 |
|---|---|---|---|---|
| D12_S010 | 11.61 / 11.93 / 12.10 | 0.00 / 0.05 / 11.30 | 10.58 / 10.90 / 11.07 | 9.94 / 10.27 / 10.44 |
| D15_S010 | 0.73 / 0.73 / 0.78 | 0.00 / 0.00 / 0.01 | 13.47 / 13.74 / 13.93 | 12.83 / 13.11 / 13.29 |
| D15_S030 | 0.78 / 0.78 / 0.78 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.54 / 0.54 |
| D15_S040 | 0.78 / 0.78 / 0.78 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |

- 잡음 없는 argmax는 하네스 출력에 없어 보고하지 못했다(P-F2 보고 항목; v2에서 출력 추가). 잡음 실행 분포만 적었다. $\arg\max\ell$은 대부분 탐색 구간 첫 행이다(봉인 §2-2 v1.3 정정과 같음).

## 7. 결론과 다음 (사용자 결정 (가) 확정)

- **결함은 gprMax에서 재현됐다.** 바닥 영역 모델 S6P_D15_S040에서 현행 선정기는 뮤트가 뒤로 2.35 ns 갈 때만 꼭짓점을 찾았다(0/10 → 9/10). 클러터 없는 대조는 Δ와 무관했다.
- 잡음 기준 바닥(B_full·B_min·B_noise)과 바닥 없음(C_none)은 분리(F1)·회복(F2)·대조 무손실(F3)·거짓 후보(F4·F4b)를 모두 통과했다. A_skip1은 결합이 남았다(F1·F2 실패, 순위 5).
- **확정: 순위 5, 현행 유지.** B 계열은 F5(봉인 정의)를 넘지 못했다(§4).

사용자 결정 (가) 뒤의 다음 단계:
1. 현행 바닥 규칙 유지. 4TU 후보 재생성·재평가는 하지 않는다(U1 교체 근거 없음). 봉인 v1.4 §10형 한계는 그대로 둔다.
2. F5 비교 대상을 정의한 v2 명세([`m210b_selector_floor_v2.md`](m210b_selector_floor_v2.md))를 결과 전에 봉인한 뒤, **새 합성 실현**(새 줄눈 배치·깊이·σ, 새 잡음 시드)으로 다시 시험한다. 묶음 1의 h5·시드는 v2 판정에 쓰지 않는다.
3. 묶음 2(S4S 돌)는 바닥 기작이 아니라 v2 판정에 넣지 않는다(보고 전용, 실행 여부는 따로 결정).

**재확인**: 이 판정서는 봉인 기준(bc9a474)과 결과 전 결정(7e843d6)만으로 셌다. 결과를 본 뒤 기준·셈법·결정표를 바꾸지 않았다. 확정본·감사 반영으로 고친 것은 사용자 결정 기록, 공개 사항, 문구다. 수치는 보고 표 반올림 3곳 정정(§6 Λ_self 3.62→3.61, 20.06→20.07, 7.96→7.95) 외에는 같다. 판정 json은 보고 키(`F5_diagnostic`, `F5_by_scope_report`, `F6_original_dt_report`)를 더한 것 외에 첫 판정 실행과 같다(대조함).
