# M2-10 (b) v2 판정 — 선정기 검출 정규화 바닥 재시험 (gprMax 새 실현 9모델)

**판정: 결정 순위 5 — 판정 후보 B_full·B_min 모두 불충분. U1 = 현행(current) 바닥 규칙 유지.**
- R(결함 재현)은 성립했다. P0 S6P_D14_S040_J2에서 current 수락 단계 성공률이 Δ 0에서 0, Δ +2.35에서 0.55(ρ 10)·1.00(ρ 20)이었다.
- B_full·B_min은 F1·F2·F4·F4b·F5-B를 모두 통과했고 **F3 하나만** 실패했다. 실패 조건은 S4N_D15_S045 ρ 20, Δ 0이고 성공률 0.55 대 current 0.85다.
- 보고 후보도 순위 5다. B_noise는 F3, C_none은 F4·F4b·F5-B 실패다.
- 기준은 봉인 명세 [`m210b_selector_floor_v2.md`](m210b_selector_floor_v2.md) v2.1(유효 봉인 `752e783`)과 이어받는 [`m210b_selector_floor.md`](m210b_selector_floor.md) v1.4, [`m210b_harness_interpretation.md`](m210b_harness_interpretation.md)다. **결과를 본 뒤 기준·셈법·결정표를 바꾸지 않았다.** 수치는 `docs/processing/m210b_v2_judgment.json`(판정 스크립트 출력)과 `data/processing/m210b_s4s6_v2/runs_v2.csv`·`run_info.json`에서만 옮겼다.
- 판정자: gpr-processing-expert, 2026-11-12. 4TU 자료는 쓰지 않았다.

## 1. 실행 정체성 (§4 판정 보고 요구)

| 항목 | 값 | 근거 |
|---|---|---|
| 유효 봉인 커밋 | `752e783cc7649b950504e06ef447c68f7b16cbeb`(2026-11-11 23:52:38 +09:00) | `git`, run_info `seal` |
| 대체된 첫 시도 | `e7ffa7f`(실행 전 감사가 봉인 뒤에 열려 대체; 실행은 e7ffa7f로 하지 않음, v2 §4-1) | 봉인 명세 §4-1 |
| **C.** 판정 인자 | `--expected-seal-commit 752e783` = 유효 봉인 커밋 | 메인 세션 실행 기록, 판정 json `rank1_identity.seal_commit_is_expected` = 참 |
| **D.** 판정 스크립트 | `git hash-object analysis/processing/m210b_v2_judge.py` = `9fae7ae7eed7594bcbe37f95616e41a41384a700` = 봉인 blob(`752e783:` 같은 경로) | `docs/processing/m210b_v2_judgment_check.json` |
| v2 명세 blob | `72e1ee0e31b76c4e1dde2243412eb2cab215b5cd`(봉인 = 작업 사본) | 〃 |
| v1.4·해석 문서 blob | `64004278548f4091f0801035745af03fca1541ec`·`1598818a6a76128a722e1e55da411a97b56bf719`(`bc9a474`·`7e843d6`과 같음) | 〃, `rank1_identity` |
| 바닥 규칙·래퍼·하네스 blob | `m210b_floor_rules.py` `bad0f95c…`, `fourtu_apex_candidate_select.py` `7cc2bf2b…`, `m210b_s4s6_pipeline.py` `7346ea8d…` | 판정 json `identity.sealed_files` |
| 봉인 파일 | 81개, 실행 시작·끝 모두 봉인 blob과 일치 | run_info `seal`·`seal_at_end`, `rank1_identity.sealed_blobs_match_run` |
| 실행 HEAD·dirty | 시작·끝 HEAD = `752e783`, dirty 0, git 조회 실패 0 | run_info, `pre_run.txt`·`post_run.txt` |
| `runs_v2.csv` | 7,960행, SHA-256 `c57401354ef377d7edb2903ce73b6f50dd13ae4b403d0644ec712341d0125bea`(run_info 기록 = 판정 시 재계산) | `m210b_v2_judgment_check.json` |
| 시각 순서 | 봉인 23:52:38 < gprMax 러너 로그 첫 생성 23:53:03(`runner_logs/S4N_D14_S040.log`·`S6P_D14_S040_J2.log`) < 마지막 모델 `.done` 00:35:25(S6P_NP_S040_J2) < 하네스 시작 00:38:10 < 종료 00:55:18 | 러너 로그 파일 시각(수기 대조, v2 §4), run_info `started_at`·`finished_at` |
| 입력 원본 | 하네스 입력 해시 실행 전후 같음(`inputs_changed_during_run` 없음). source_guard verify(입력 41개) 통과, 원본 쓰기 0 | run_info, 메인 세션 확인 |
| 시드 | 공식 `20271120 + 1000 p + 100 q + s`, 순잡음 20271199. v2 시드 201개(20271120–20275239), 묶음 1·시제품 시드와 겹침 0. 공식 밖 행 0 | run_info `seeds`, 판정 json `rank1_seeds` |
| 환경·회귀 | MKL_CBWR COMPATIBLE, numpy 2.5.4, Python 3.13.18. 4TU 래퍼 회귀 두 csv 바이트 동일(all 활동) | run_info |
| 하네스 가드 | pass, failures 없음 | run_info `guards` |

- **결과 전 결정**: D1·D2·P-F1–P-F3는 봉인 뒤·결과 전 결정이다(해석 문서 `m210b_harness_interpretation.md`, 커밋 `7e843d6`). v2의 W1–W10·V9는 v2 봉인(`752e783`) 전 결정이다.
- **봉인판 명세의 이력**: 실행 전 재감사가 본 판(작업 사본 SHA-256 앞 8자리 `528A3EFD`)에 지적 B·C·D 문구를 더했다(2026-11-11 23:42, scratchpad `apply_ff.py`; 판정 기준 문장은 바꾸지 않음). 그 결과가 봉인판이다(작업 사본 SHA-256 `0707A611…`, blob `72e1ee0e…`).
- **운영 순서 편차(기록)**: 하네스 로그 폴더 `data/processing/m210b_s4s6_v2/`는 00:37:53에 생성됐다. gprMax 실행(23:53:03–00:35:25)보다 늦어, 명세 §4-2의 순서(로그 폴더를 미리 만든 뒤 gprMax)와 다르다. 하네스 시작(00:38:10) 전에 만들어졌으므로 하네스 로그 손실은 없다(`harness.err` 0바이트). gprMax 로그는 입력 폴더 하위 `runner_logs/`·`h5/`에 있었다.

봉인 파일 81개 — `752e783` 시점 blob(판정 json `identity.sealed_files` 그대로; `git ls-tree -r 752e783`과 대조해 불일치 0):

| 경로 | blob(SHA-1, LF 기준) |
|---|---|
| `analysis/processing/fourtu_apex_candidate_select.py` | `7cc2bf2b3db80a626bd6f81fabc00e53b9805d8a` |
| `analysis/processing/m210b_floor_proto.py` | `860cd3ada5b898e611ea24b3e16d4725105a7776` |
| `analysis/processing/m210b_floor_rules.py` | `bad0f95cdcebe3d193fb7aad6511dbbafcac8327` |
| `analysis/processing/m210b_lambda_pilot.py` | `aee1b0a24ffde10ef37c3f1c65a43ca52347d4aa` |
| `analysis/processing/m210b_s4s6_pipeline.py` | `7346ea8df7e2a29c0072dca9d511fb052cbe0515` |
| `analysis/processing/m210b_v2_judge.py` | `9fae7ae7eed7594bcbe37f95616e41a41384a700` |
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
| `analysis/synthetic/s4s6/inputs_v2/BG_N_S045.in` | `756a03b9171f01b4fe5e5e8102fdfae6fe342e57` |
| `analysis/synthetic/s4s6/inputs_v2/BG_PF_S045.in` | `9eabcafac85157cfc88e74f6bd1924e22be0886b` |
| `analysis/synthetic/s4s6/inputs_v2/S4N_D13_S010.in` | `9debe0018303ad31a017d634231ae1744366b37a` |
| `analysis/synthetic/s4s6/inputs_v2/S4N_D14_S040.in` | `94a91bba2afba8ea6d7b92adf09106077c8c5a2b` |
| `analysis/synthetic/s4s6/inputs_v2/S4N_D15_S045.in` | `a0192ef0fd6eb3cc4ff26545e50e5b588958c5a3` |
| `analysis/synthetic/s4s6/inputs_v2/S4N_D16_S040.in` | `21c4bf67adbc7c4b26373e0c7d96e94a316f9cd3` |
| `analysis/synthetic/s4s6/inputs_v2/S6P_D13_S010_J5.in` | `ce404228b5665052cfb03239c2a1b1c05695e7ab` |
| `analysis/synthetic/s4s6/inputs_v2/S6P_D14_S040_J2.in` | `f557f9f29d4c46ee46f518e2db312454b263e163` |
| `analysis/synthetic/s4s6/inputs_v2/S6P_D15_S045_J4.in` | `f896f180b027e8944bbf1b33fd7ccc19249eae85` |
| `analysis/synthetic/s4s6/inputs_v2/S6P_D16_S040_J3.in` | `b7e2bf6779b893ee4a22084efd7ca682bc13798d` |
| `analysis/synthetic/s4s6/inputs_v2/S6P_NP_S040_J2.in` | `5fe1c3c23cf911bc3472fb97fe56778102f9adb6` |
| `analysis/synthetic/s4s6/models_s4s6.json` | `73d8bdce152577359cfe30a51aeed2e1fa24d9fa` |
| `analysis/synthetic/s4s6/models_s4s6_v2.json` | `d1d9f52315ec636756208563d791af38b94867a0` |
| `analysis/synthetic/s4s6/pilot/L_S6P_NP_S040_T15.in` | `887c18142d76b293b07d72c7b5ff1002df3bd043` |
| `analysis/synthetic/s4s6/pilot/P_S4N_D15_S030.in` | `8e4ca906468692776bca137baa4b7d509284bcaa` |
| `analysis/synthetic/s4s6/pilot/P_S4N_D15_S040.in` | `65337f98a2631b63bb396f0d1af4e8da5fcd4dcb` |
| `analysis/synthetic/s4s6/pilot/P_S6P_D15_S030.in` | `66c299964a86ce339f1bf5cede7bb8aa2f1782a7` |
| `analysis/synthetic/s4s6/pilot/P_S6P_D15_S040.in` | `690bcd669cc6a7a7383624ca379c7584bc2ca9d8` |
| `analysis/synthetic/s4s6/pilot/P_S6P_NP_S040.in` | `5f3f70dc8b85fbe4a2a8ad3329e56fa7790fac01` |
| `analysis/synthetic/s4s6/pilot/lambda_S6P_D15_S040.json` | `8c6018e9b9370f0da3af17d1819833b0caeac3c1` |
| `analysis/synthetic/s4s6/pilot/pilot_check.json` | `fa75f7b4477353a0696b76b42dff0d05a24fba34` |
| `analysis/synthetic/s4s6/pilot/pilot_envelopes.png` | `ca0c0499756ca6b237e2ee8f191c2f86306f4d1d` |
| `analysis/synthetic/s4s6/pilot_v2/L_S6P_NP_S040_J2_T15.in` | `c058e7f252d60009553f3fdf5b5f98fb09dfa405` |
| `analysis/synthetic/s4s6/pilot_v2/L_S6P_NP_S040_J3_T15.in` | `aa378c13ca9319f55553a522dab58c16daf8ea0a` |
| `analysis/synthetic/s4s6/pilot_v2/L_S6P_NP_S045_J4_T15.in` | `2e890a5d722d6e4fdfbece822523bb9e2bd8f159` |
| `analysis/synthetic/s4s6/pilot_v2/P_S4N_D14_S040.in` | `c9d5dde5f1167f7084ad2b0247de93b18857008f` |
| `analysis/synthetic/s4s6/pilot_v2/P_S4N_D15_S045.in` | `9f3626acf9614d7cc207467d99a56195c302a9ce` |
| `analysis/synthetic/s4s6/pilot_v2/P_S4N_D16_S040.in` | `8fe761451a6203e44c38708ac69554a46eb6ce93` |
| `analysis/synthetic/s4s6/pilot_v2/P_S6P_D14_S040_J2.in` | `0f0bfc2f253997efa3c4a0cd3015179f6431db02` |
| `analysis/synthetic/s4s6/pilot_v2/P_S6P_D15_S045_J4.in` | `fe57879d2a5df2083f69ae442e0ec7dab9218ac6` |
| `analysis/synthetic/s4s6/pilot_v2/P_S6P_D16_S040_J3.in` | `840e48bcfaa9637f12effe1f7a33bb5b2ab848c8` |
| `analysis/synthetic/s4s6/pilot_v2/P_S6P_NP_S040_J2.in` | `8333256903b3b9b2149a92998210cb43a9352e76` |
| `analysis/synthetic/s4s6/pilot_v2/P_S6P_NP_S040_J3.in` | `eb199c538616a20749a09c62bdd96f42e00b37ba` |
| `analysis/synthetic/s4s6/pilot_v2/P_S6P_NP_S045_J4.in` | `8a81376d1d1322a56a9149ea49ebf30205c67092` |
| `analysis/synthetic/s4s6/pilot_v2/lambda_S6P_D14_S040_J2.json` | `cfda7bb74aa4ffc4a15da3dbb028bcc7ceb12049` |
| `analysis/synthetic/s4s6/pilot_v2/lambda_S6P_D15_S045_J4.json` | `f9d403f5cc9b10d9b7bf178c93ac157e2bf6744e` |
| `analysis/synthetic/s4s6/pilot_v2/lambda_S6P_D16_S040_J3.json` | `db142c3a36f3342da4f662fa70e55c2968e01a90` |
| `analysis/synthetic/s4s6/run_s4s6.cmd` | `b561467dbd7346bc1f5e74bc24592a106ec13870` |
| `analysis/synthetic/s4s6/run_s4s6_v2.cmd` | `2c3d95e361a07b358c0a83831d0d4b97a84c927f` |
| `docs/processing/m210b_a0_S4N_D15_S030.json` | `ce4f33b5ae20ff9400ee3aaf82157d9d1f0cfc70` |
| `docs/processing/m210b_a0_S6P_D15_S030.json` | `f6ff56e0e43432049b5e386aea03faf2c5e27db1` |
| `docs/processing/m210b_floor_proto.json` | `a6e9db068d28e788677760fadd7b4ed23ef95b33` |
| `docs/processing/m210b_harness_interpretation.md` | `1598818a6a76128a722e1e55da411a97b56bf719` |
| `docs/processing/m210b_selector_floor.md` | `64004278548f4091f0801035745af03fca1541ec` |
| `docs/processing/m210b_selector_floor_v2.md` | `72e1ee0e31b76c4e1dde2243412eb2cab215b5cd` |
| `scripts/synthetic/export_dataset.py` | `1bc6160986ae1471746b826c7e9f794fd5617342` |
| `tests/test_m210b_s4s6_pipeline.py` | `65faa8251aba9c74289daed2f32791a3bb7d5291` |
| `tests/test_m210b_v2_judge.py` | `648d245ecb6b3893bc20ca70ceba9e5598558037` |
| `tests/test_synthetic_export.py` | `4379838fc7bc54b05bf4a2916bf81b69d0c72591` |

## 2. 순위 1 단언과 P-F1 (모두 참)

- `rank1_identity`의 모든 항목이 참이다: 봉인 파일 목록, 봉인 blob = 실행 blob, 봉인 커밋 = 기대값, 실행 HEAD = 봉인, v2 문서 blob, git 조회, v1.4·해석 문서·바닥 규칙·래퍼 blob, HEAD 시작 = 끝, dirty 0(시작·끝), 봉인 < 하네스 시작, runs SHA-256, 시드 겹침 0, 하네스 가드, 래퍼 회귀.
- P-F1(v2):
  - 판정 행 7,200, 칸 360개에 빠짐·남음 없음, 칸마다 시드 정확히 20개. F4b 칸도 같다.
  - `n_runs` 7,960 = 기대 = run_info, 영점 실패 0.
  - 모델 집합 = 기대 9개 ID(json `batch == "v2"`, `runs_v2.csv`, `V2_EXPECTED_MODELS` 모두 일치).
  - 관 모델 `L_noise_free_current`·`apex_trace` 이상 0.
  - **W10**: NP의 `calibration_pair` = D14_S040이고, `sigma_raw`(ρ 10 0.0092359, ρ 20 0.0046179)가 P0 보정값과 같다.

## 3. 실행 전 감사 기록 5–12 (v2 §4 요구)

| # | 기록 | 이 판정에서 |
|---|---|---|
| 5 | 빈 `accepted_apex`를 실패로 읽어 효과 e가 부풀 위험 | 관 모델 판정 행의 `accepted_apex`·`floor_held_at_apex` 빈칸 **0건**(센 값, `pipe_judgment_blanks`). 영향 없음 |
| 6 | F4b는 Δ 0에서만 판정(봉인 코드의 읽기, 묶음 1과 같음) | F4b는 Δ 0으로 판정했다. Δ 4개 결과는 §5 표에 있고, 네 Δ 모두 같은 결론이다(B_full·B_noise 0.0 대 current 0.0, B_min 0.05 대 0.0으로 통과; C_none 0.55–0.65 대 0.0으로 실패) |
| 7 | 빈 집합 경로 넷 | 기대 모델 집합 단언 참(9개, NP 포함), git 실패 = 가드 실패(실패 0), 끝 `sealed_files` 대조 참(81개). 영향 없음 |
| 8 | 감사가 첫 봉인 29초 뒤에 열림 | 재봉인(`752e783`)으로 처리했다(§1). 실행은 유효 봉인으로만 했다 |
| 9 | `run_s4s6_v2.cmd` 작업 사본 LF(속성 eol=crlf) | blob 대조 일치(봉인 파일 81개 일치에 포함). 판정 무관 |
| 10 | W10 단언 없음 | 이제 단언한다. 결과 참(§2) |
| 11 | BG_N·BG_PF_S045 h5가 봉인 전 생성 | 러너 로그 기준 `h5/BG_N_S045.log` 21:14:55–21:15:07, `h5/BG_PF_S045.log` 21:15:16–21:15:33. h5 파일 시각은 `BG_N_S045_merged.h5` 21:15:10, `BG_PF_S045_merged.h5` 21:15:35–21:15:36(2026-11-11)이다. 판정 입력이 아니다(영점 참고) |
| 12 | `Lambda` 4자리 반올림이 경계 10과 비교됨 | 판정은 `Lambda` 열로 했다. \|Λ − 10\| < 1e-3인 (c, π)는 없다(`lambda_rounding_check` 빈 목록). 열 값과 반올림 전 값의 결론 차이 없음 |

## 4. R — Q·G·C·e (current, 수락 단계, 판정 열 `rho_real`·`Lambda`; D1·D2)

| S6P 조건 | ρ | π | 성공률 e→l | e | Q(ρ_real 최소) | Λ(Δe) / Λ(Δl) | Δe 바닥 비율 | G(사유) | C | R 짝 |
|---|---|---|---|---|---|---|---|---|---|---|
| **S6P_D14_S040_J2**(P0) | 10 | (0, +2.35) | 0.00→0.55 | **0.55** | 참(18.40) | 21.24 / 4.45 | 1.0 | **참** | 참(0) | **예** |
| 〃 | 10 | (−1, +2.35) | 0.00→0.55 | 0.55 | 참 | 18.67 / 4.45 | 1.0 | 참 | 참 | **예** |
| 〃 | 20 | (0, +2.35) | 0.00→1.00 | **1.00** | 참(36.17) | 21.24 / 4.45 | 1.0 | 참 | 참 | **예** |
| 〃 | 20 | (−1, +2.35) | 0.00→1.00 | 1.00 | 참 | 18.67 / 4.45 | 1.0 | 참 | 참 | **예** |
| S6P_D16_S040_J3(P1) | 10·20 | 두 짝 | 0.00→0.00 | 0 | 참(17.88 / 35.23) | 68.28·59.23 / 14.09 | 1.0 | 거짓(`lambda_l_gt_10`) | 참 | 아니오 |
| S6P_D15_S045_J4(P2) | 10·20 | 두 짝 | 0.00→0.00 | 0 | 참(19.53 / 38.24) | 87.44·77.12 / 18.46 | 1.0 | 거짓(`lambda_l_gt_10`) | 참 | 아니오 |
| S6P_D13_S010_J5(P3) | 10 / 20 | 두 짝 | 0.65→0.65 / 0.90→0.90 | 0 | 참(11.54 / 23.19) | 0.16·0.14 / 0.06 | 0.0 | 거짓(`lambda_e_le_10`, `floor_frac_lt_0.5`) | 참 | 아니오 |

- R 성립(R 짝 4개) → 결정은 순위 4/5(규칙별)로 간다(판정 json `decision`). 순위 2는 해당 없다(S4N ρ 20 current Δ 0: D13 0.85, D14 0.90, D15_S045 0.85, D16 0.85, 모두 ≥ 0.5). 순위 3(Q 없음)도 해당 없다.
- V9의 효과(보고, `G_excluded_no_escape`): P1·P2의 8개 짝은 Λ(Δe) > 10이고 바닥에 걸렸지만, Λ(Δl)이 14.09·18.46 > 10이라 G 밖이다. 그 짝의 e는 모두 0이다. 기작의 예측(늦은 뮤트로도 바닥을 벗어나지 못함)과 같다. V9가 없었다면 R은 P0로 성립했으므로 이 판정은 V9와 무관하게 같다.
- **보고 불일치 확인(메인 세션 질의)**: `G_excluded_no_escape`의 Λ 68.28/14.09는 판정 표(`R_table`)와 같은 **Lambda_self**(판정 열, D2) 값이다. 질의에 나온 131.94/27.23은 `P_F3.R_table`의 **Lambda_cal**(봉인 글자대로 읽은 민감도, P-F3)이다. 판정 G는 봉인 규칙대로 `Lambda` 열(self)로 계산됐다(`m210b_v2_judge.py` `r_table`, `use_cal=False`). **불일치가 아니며 판정은 바뀌지 않는다. 위임할 것 없음.**

## 5. 후보별 F1–F5·F4b (수락 단계; 기준 = current)

| 규칙 | F1 분리 | F2 회복(P0 R 짝) | F3 대조 무손실 | F4 거짓(Δ 0) | F4b 경사 잡음(Δ −1 / 0 / +1 / +2.35, X 대 cur) | F5-B(S1, 428쌍) | 순위 |
|---|---|---|---|---|---|---|---|
| **B_full**(판정) | 통과 | 통과(ρ 10 0.55 ≥ 0.55 − 0.1, ρ 20 1.00 ≥ 1.00 − 0.1) | **실패**: S4N_D15_S045 ρ 20 0.55 대 0.85 | 통과 | 0.00/0.00/0.00/0.00 대 0.00(통과) | 통과: 악화 0, 개선 60, 시각 오차 중앙 0.060 대 0.0675 ns | **5** |
| **B_min**(판정) | 통과 | 통과 | **실패**(같은 조건, 0.55 대 0.85) | 통과 | 0.05 ×4 대 0.00(통과) | 통과(같은 값) | **5** |
| B_noise(보고) | 통과 | 통과 | 실패(같은 조건) | 통과 | 0.00 ×4 대 0.00(통과) | 통과(같은 값) | 5 |
| C_none(보고) | 통과 | 통과 | 통과 | **실패**: NP ρ 10·20 0.65 대 0.00 | 0.55/0.65/0.65/0.65 대 0.00(실패) | **실패**: 악화 9, 개선 0(459쌍) | 5 |

- F3은 판정 후보 둘 다 S4N_D15_S045 ρ 20 한 조건에서만 실패했다. 같은 모델 ρ 10은 모든 규칙이 0.25로 같다. 나머지 S4N 조건의 Δ 0 성공률(B_full = B_min 대 current)은 D13 0.40 대 0.45·0.80 대 0.85, D14 0.25 대 0.25·0.85 대 0.90, D16 0.25 대 0.25·0.85 대 0.85(ρ 10·20)이고, 모두 차 ≤ 0.1이다.
- F5-B 대칭 점검(보고): current를 X 자리에 놓으면 악화 60·개선 0으로 비열등이 아니다. 곧 위치 정확도는 **B 계열이 current보다 낫다**(판정 json `reading` "B_full better"). 범위별(`F5B_by_scope`, 보고):

  | 범위 | 쌍 | 악화 | 개선 |
  |---|---|---|---|
  | S2(R 조건) | 32 | 0 | 0 |
  | S3(G 조건) | 32 | 0 | 0 |
  | S4N만 | 304 | 0 | 44 |
  | G 밖 S6P | 92 | 0 | 16 |

- U1: B_full 순위 5, B_min 순위 5 → **current**(현행 유지).

## 6. F3 불충족의 물리적 기술 (판정 아님 — 진단, 규칙·기준 변경 없음)

이미 본 v2 결과에서, 봉인된 하네스 함수를 그대로 불러 S4N_D15_S045 ρ 20 Δ 0의 실현 7개를 다시 돌렸다(`analysis/processing/m210b_v2_f3_trace.py` → `docs/processing/m210b_v2_f3_trace.json`). 재실행의 검출 피크 수는 `runs_v2.csv`와 7/7 같았다. 실현 선택은 $s$ = 0, 1, 3, 4, 6, 10, 11이다(B_full 성공 1 + 실패 6). 선택 규칙은 미리 정하지 않았다. 시드별 표를 앞에서부터 보며 비교용 성공 1개와 실패를 골랐고, 실패인 $s$ = 7·16·18은 특별한 이유 없이 빠졌다(선택 편향 가능성을 기록한다; 진단일 뿐 판정과 무관). 피크 수(current / B_full): 172/3, 165/2, 172/1, 166/3, 169/1, 165/2, 154/3.

- **꼭짓점 에너지는 충분하다.**
  - 잡음 없는 꼭짓점 포락선은 $A_{0,\text{self}}$ = 0.00802, 실현 $\rho_\text{real}$ 중앙은 19.79다.
  - B_full 바닥은 $F$ 중앙 0.00107($=2.5\hat\sigma$; 탐색 구간 $\hat\sigma_n$ 중앙 0.000424)이다. 꼭짓점의 검출값은 $E/F\approx7.5$로 문턱 2를 넘는다.
  - 실제로 7개 실현 모두에서 B_full의 검출 피크가 꼭짓점 근처(트레이스 ±25·±3 ns 안, 1–3개, 시작 시각 33.48–34.13 ns)에 있었다.
  - 클러터 없는 이 측선에서는 $L$(0.00081) < $F$라 탐색 구간 전체가 바닥에 걸린다(`floor_held_at_apex` B 1.0, current 0.0).
- **잃는 곳은 능선 이동이다.** 깊이 1.5 m 꼭짓점은 옆으로 거의 평평하다(±3 트레이스 = ±0.06 m에서 왕복 시각 차 약 0.02 ns). `crest_apex`는 ±0.5 m 안에서 가장 이른 점으로 옮긴다. 그런데 평평한 능선에서는 잡음이 "가장 이른 점"을 옆으로 옮겨, 끝점이 꼭짓점에서 4–6 트레이스 떨어진다.
  - B_full의 끝점 예: 트레이스 84·86·74·75·85, 꼭짓점 80. 창 ±3 트레이스 밖이다.
  - 같은 트레이스에서 출발한 피크라도 시작 시각이 규칙마다 다르다. current는 $D=E/\ell(k)$의 행 극대에서, B는 $E/F$의 행 극대에서 출발한다(예: 33.467 대 33.521 ns). 그래서 끝점이 갈린다.
- **current가 더 자주 맞히는 이유는 시작점이 많아서다.**
  - current는 클러터 없는 깊은 행에서 바닥이 거의 0($F$ 중앙 4.1e-5)이다. 그래서 잡음 피크가 측선당 154–172개 생기고, 그 가운데 꼭짓점 능선 위 여러 피크가 저마다 능선 이동을 시작한다. 하나라도 창 안에 끝나면 수락 단계 성공이다.
  - B는 잡음 피크를 거의 없애 시작점이 1–3개뿐이다. 그 한두 번의 이동이 옆으로 빗나가면 실패다(`no_peak` 7/20, `not_accepted` 2/20, 대 current 2/20·1/20).
  - C_none(바닥 없음)도 피크 166.5개(중앙)로 current와 같은 0.85다. 이것이 이 설명과 맞는다.
- 곧 F3 실패는 잡음 기준 바닥이 깊은 꼭짓점을 "못 보는" 감도 손실이 아니다. **현행의 높은 성공률이 잡음 피크가 주는 여러 번의 능선 이동 기회에 기대고 있고, 잡음 피크를 없앤 규칙은 그 기회를 잃는다**는 상호작용이다. 같은 조건 ρ 10에서는 모든 규칙이 0.25로 같다.
- W5(깊은 쪽 감도, 보고) S6P_D13_S010_J5 ρ 10의 손실 0.2(B 0.45 대 늦은 뮤트 current 0.65)도 같은 유형으로 보인다. 다만 단일 실현 추적은 S4N_D15_S045만 했다(추정).
- **결과로 규칙을 바꾸지 않는다.** 이 기술은 다음 판(v3) 설계의 근거로만 쓴다.

## 7. 보고 항목 (판정 무관)

- **F5-C 깊이**(관 상단, 최근접 kept 힌트): \|d − d_true\| 10/50/90 % 분위.

  | 규칙 | X | current |
  |---|---|---|
  | B_full | 0.0134 / 0.0165 / 0.0201 m | 0.0136 / 0.0165 / 0.0198 m |
  | C_none | 0.0133 / 0.0165 / 0.0198 m | 같음 |

  깊이 정확도에는 실질 차가 없다.
- **F6**:
  - kept 단계 성공률(관 모델 판정 행): current 0.359, B_full·B_min·B_noise 0.650, C_none 0.624
  - lost_reason(current): no_peak 543, not_accepted 209, other_nms 68
  - lost_reason(B_full): no_peak 228, not_accepted 216, other_nms 4
  - 상한 300 도달 0행
- **바닥 영역 밖 대비(보고)**: P1·P2(Λ(+2.35) > 10)에서 current는 Δ 네 값 모두 0/20이다. B_full은 0.70·0.60(ρ 10), 1.00(ρ 20)이다. 늦은 뮤트로도 바닥을 벗어나지 못하는 깊은 관을 B 계열은 잡는다. V9로 판정에서 뺀 영역이라 판정에 쓰지 않는다.
- **M0**:
  - 잡음 없는 argmax(뮤트 아래 주기, Δ −1 / 0 / +1 / +2.35): S6P_D14·D16·D15_S045가 0.784 / 0 / 0 / 0.539, S6P_D13_S010_J5가 0.784 / 0 / 0 / 11.711이다.
  - 잡음 실행 분포(ρ 10)도 같은 값에 몰려 있다(판정 json `M0`).
  - Λ_self 값은 §4, Λ_cal은 §8에 있다.
- **원 Δt 보조**(360행, current·B_full, ρ 10, Δ 0): 원 격자에서도 S6P P0–P2의 current는 0.00, B_full은 0.70·0.85·0.55다. current는 클러터 없는 줄과 S6P_D13에서 상한 300에 20/20행 걸렸다(판정 json `original_dt`).
- `lambda_rounding_check`: 해당 없음(빈 목록).

## 8. P-F3 민감도 (Λ_cal·ρ_cal, 봉인 글자대로)

- Λ_cal(Δ 0 / +2.35): P0 41.06 / 8.61, P1 131.94 / 27.23, P2 184.40 / 38.93, P3 0.19 / 0.08.
- G·R은 D2 판정과 같은 짝에서 성립한다(P0 4짝). 결정 4/5와 규칙별 순위도 같다. 결론은 바뀌지 않는다.

## 9. 결론

- **결정 순위 5의 귀결**: U1에 따라 현행 바닥 규칙($0.05\max\ell$)을 유지한다. 4TU 후보 재생성·재평가(U4)는 하지 않는다.
- **묶음 1 대비**:

  | 항목 | 묶음 1(v1.4) | v2(새 실현) |
  |---|---|---|
  | 결함 재현 | S6P_D15_S040, 0/10 → 9/10 | P0 S6P_D14_S040_J2, 0/20 → 11/20(ρ 10)·20/20(ρ 20) — 다른 줄눈 배치·깊이에서 다시 재현됨 |
  | 실패 항목 | F5(순위 최상위 비교) | F5-B 통과(위치는 오히려 개선) → 새 실패 항목 F3(한 조건) |
  | 공통점 | 분리·회복·거짓 후보는 통과 | 같음 |

- 두 판정 모두 "결함은 실재하고, 잡음 기준 바닥은 결함을 고친다(F1·F2)". 그러나 "봉인 기준 전부를 넘지는 못했다"는 같은 형태로 끝났다.
- **한계**:
  1. 2D 합성(원통 확산)이라 성공률의 크기는 정성 판단이다. 판정은 같은 모델 안의 Δ·규칙 비교로만 했다.
  2. F3 문턱 0.1은 20시드에서 2시드 차다. 실패는 0.30 차(6시드)로 문턱보다 크다.
  3. §6의 F3 기작 기술은 실현 7개 추적에 근거한 진단이고, 정량 일반화가 아니다.
  4. current의 수락 단계 성공에는 잡음 피크가 주는 여러 번의 시작점이 섞여 있다. 그래서 F3("클러터 없는 곳에서 잃지 않는가")은 바닥의 감도만이 아니라 시작점 수의 차이도 잰다. 이 점은 봉인 기준의 성질이다.
  5. 4TU의 영향은 재지 않았다(U4는 채택 때만).
- 다음 단계(사용자 결정 사항, 판정 아님): 현행 유지로 M2-10 (b)를 닫을지, 또는 v3로 다시 시험할지를 정한다. v3는 §6의 사실을 설계 근거로 삼는다. 예를 들어 능선 이동 시작점 수를 통제한 F3 정의나, 능선 이동 규칙 자체의 평평한 꼭짓점 안정화를 생각할 수 있다. v3를 하면 이 결과를 보고 만든 기준이므로 새 실현으로만 판정한다.

## 10. 재현

세 명령 모두 git 제외 폴더 `data/`(gprMax 내보내기 `data/synthetic_gprmax_s4s6_v2/`, 하네스 출력 `data/processing/m210b_s4s6_v2/`)가 필요하다. 저장소만으로는 재현할 수 없고, 특히 세 번째 명령은 내보낸 npy가 있어야 한다.

```
conda run -n gpr-dev python -I analysis/processing/m210b_v2_judge.py --expected-seal-commit 752e783   # 판정(봉인판)
conda run -n gpr-dev python -I analysis/processing/m210b_v2_judgment_check.py                        # 판정서 대조용 재집계(읽기 전용)
conda run -n gpr-dev python -I analysis/processing/m210b_v2_f3_trace.py                              # §6 진단(판정 아님)
```
