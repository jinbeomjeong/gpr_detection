# M2-10 K7 생성기 결함 — 판정서 (봉인 명세 v1.2, 재봉인 8642e29)

- 작성: gpr-processing-expert, 2026-11-12. 판정 대상: 명세 [`m210_k7_generator.md`](m210_k7_generator.md) v1.2(봉인 커밋 `8642e29`, 문서 blob `2ccd168e`)의 후보 a1(시작점 오르막)·a1b2(a1 + 순위 SNR).
- 판정 입력(이 문서의 모든 수치는 아래 파일에서만 옮겼다):
  - 판정 스크립트 출력 [`m210_k7_judgment.json`](m210_k7_judgment.json)(`analysis/processing/m210_k7_judge.py --expected-seal-commit 8642e29`, `MKL_CBWR=COMPATIBLE`, 16:11:08 종료 0)
  - 하네스 출력 `data/processing/m210_k7/run_info.json`·`runs_k7.csv`(git 제외)
  - 읽기 전용 수집 [`m210_k7_judgment_check.json`](m210_k7_judgment_check.json)(`analysis/processing/m210_k7_judgment_check.py`, gpr-dev, E코어 24–27, BelowNormal, 난수 없음): git blob, source_guard 재verify, h5_compare 보고서 요약, runs 행 진단
  - h5_compare 보고서 `<scratchpad>/h5cmp_k7/*.json`, source_guard 상태 `<scratchpad>/k7_run1_h5_guard.json`·`k7v12_h5_guard.json`·`k7v12_export_guard.json`
- **고정점**: 판정 json [`m210_k7_judgment.json`](m210_k7_judgment.json) SHA-256 = `c779d42dc89ca5d2c7fd1866cd1a95890962f813486b296a53ba30f5a28a0aae`(16:11:08 생성, 이 판정서의 판정 값은 모두 이 파일에서 옮겼다).
- 수집 json 이력: `m210_k7_judgment_check.json`은 16:14:51에 처음 만들었고, 수집 스크립트를 16:16:17에 고친 뒤(진단 항목 추가) 16:16:48에 다시 만들었다. 최종 json은 최종 스크립트의 산출이다. 수집 json은 판정 json과 무관하다(판정에 쓰지 않는 사실 대조·진단용).
- **결과를 본 뒤 바꾼 것: 없음.** 명세·판정 기준·결정표는 봉인 커밋 그대로 적용했다. §4의 물리 해석과 §5–§6의 진단은 판정에 쓰지 않는다.

## 0. 판정

**결정 순위 2 — 보류(재현 조건 미달). 채택 = 현행 생성기(`current`).**

| 후보 | 순위 | 사유(판정 json `ranks`) |
|---|---|---|
| a1 | **2** | K-R(a) 미성립 |
| a1b2 | 판정 안 함(a1이 순위 7이 아님) | 참고: K-R(b) 미성립으로 순위 2에 해당 |

- 순위 1 사유는 0건이다(`rank1_reasons` 빈 목록, 정체성·시드·고정 집합 검사 모두 pass).
- K-R은 봉인 정의대로 ρ 20, `cur`에서 (수락 단계 성공률 − kept 단계 성공률) ≥ 0.5를 본다. 세 결함 모델 모두 0.5에 못 미쳤다(§3).
- 결정표(명세 §4-6)에 따라 순위 2는 "모델 재설계"로 가는 보류다. 4TU 재생성·재평가는 하지 않는다.

## 1. 판정서 첫머리 목록 (명세 §8, 19항목)

공개 항목(10의 대조 결과·영점, 11, 13, 15, 17, 18, 19)은 값과 무관하게 순위 1 사유가 아니다. 항목이 빠지면 순위 1이다.

| # | 항목 | 값 | 출처 |
|---|---|---|---|
| 1 | 봉인 커밋과 파일별 blob | 재봉인 `8642e29a9d19dad98997690d8d572b70173e503b`(15:03:59 KST). 봉인 파일 42개 blob은 §1-1 표. 봉인 폴더 `analysis/synthetic/k7` 트리 `cb81d405c817cbfd25157fa4909a0b145db98bdd`(23파일, `d6aaf0f`와 같음). 작업 트리 blob은 42개 모두 봉인과 같다(불일치 0). LF 기준 blob 대조 | check json `seal` |
| 2 | `--expected-seal-commit` | `8642e29a9d19dad98997690d8d572b70173e503b` | 판정 json `expected_seal_commit` |
| 3 | 판정 스크립트 blob | 작업 트리 `9773ecb0c72b2b7102b037f57c94f8f7b211b107` = 봉인 커밋 blob(판정 스크립트 자기 단언 `judge_blob_is_seal_blob` true, 수기 대조 같음) | 판정 json·check json `judge_blob` |
| 4 | HEAD·dirty 전후 | 하네스 `pre_run.txt` 15:40:23 HEAD `8642e29…` dirty 0, `post_run.txt` 16:10:32 HEAD `8642e29…` dirty 0. `seal_at_end.dirty_files` 빈 목록, `git_failures` 빈 목록 | run_info, check json `identity_files` |
| 5 | `runs_k7.csv` SHA-256 | `641b8256c018a96f9aa4b4078e45f995481f757ccabca6d1626142e79b1898c5`(run_info 기록 = 판정 때 재계산 = 수집 재계산) | run_info, 판정 json, check json |
| 6 | 시각 순서 | 봉인 15:03:59 < gprMax 시작 15:04:25(러너 로그 첫 줄 `K7_REF_S001`·`K7_REF_L085_S001`) < 하네스 시작 15:40:23(판정 단언 `seal_before_harness_start` true). gprMax 13단계 모두 `exit=0`, 마지막 `.done` 15:38:11 | check json `runner_logs`, run_info |
| 7 | 시드 겹침 | 0. 대조 집합: 묶음 1 101개, v2 201개, v3 481개, 시제품 1,710개, K7 MC 581개. K7 시드 641개(20301112–20308251), 순잡음 20301199 | run_info `seeds` |
| 8 | G0와 시험 | 4TU 두 csv 재생성 SHA-256 = 커밋값 `1afd23cf…3ef268`·`485d4c22…c01bb21b`(작업 트리 CRLF 바이트 기준). `g0_s1g` identical(19행, 열 단위, `stored_only_columns` = [`cluster_mu`]). (b) v3 부분 집합(S4N_D14_S038, 시드 2) 68행 차이 0. 생성기 blob `2318deae01fe31d43485a87ee7b5ea53732fedd6`. 하네스 시작 자식 pytest(K7 시험 3파일) 94 passed, skip 0, 실패 0 | 판정 json `published.g0`, run_info `tests` |
| 9 | 고정 집합 단언 | `rank1_fixed_sets` 117검사 모두 pass: 기대 모델 8, 역할 집합(json과 같음, `pvc`·`c_diag` ⊂ `control`, 분할 서로소·8모델), 변형 4, ρ 2, 시드 40, 관 수(`pipes` 길이), 조건·관별 행 수, 잡음 없는 행, 보정 짝, `models_k7.json` SHA. 대상 셀 수: K-F1 a1 4/4, a1b2 6/6, K-F2 8/8, K-F3 16/16. 행 3,888(영점 실패 0) | 판정 json `rank1_fixed_sets`, `items` |
| 10 | h5 출처·snapshot·영점(공개) | snapshot ↔ verify: 재실행 h5 13개 + `FS.h5`(14), 내보낸 33파일, run1 h5 13개 모두 pass(판정서 작성 때 재verify도 pass, §1-2). `FS.h5` = `ae2c3844077f9ba566c443a7215a8f911f1950261386c0a1597b1b599f943bfa`(v3 판정서 참조 h5 표와 같음). 점검 산출 대조: `K7_REF_S001`–`S003` 재실행 = `pilot_prerun_h5` **배열 비트 동일**(파일 SHA는 S002만 같음, `track_times` 예상 차이). 영점: 점검 S001 3.6162·S002 3.6156·S003 3.6151 ns, 전체 실행 내보내기 `surface_reflection_peak` 3.6162·3.6156·3.6151 ns(같음). 하네스 검출 영점(`air_single_event`)은 3.2026–3.2031 ns | check json `source_guard`·`h5_compare`·`export_time_zero_candidates_ns`·`harness_detected_zero_ns` |
| 11 | K-F2 자명 통과·K-F4 해당 없음(공개) | K-F2 자명 통과 셀: a1 없음, a1b2 없음. K-F4 해당 없음(짝 행 < 20): a1 — P3_D080 ρ20(12), P1_D150_S003 ρ10(19), P1_D060 ρ10(19), PVC_D070 ρ10(15). a1b2 — P3_D080 ρ20(10), P1_D150_S003 ρ10(16), P1_D110_S002 ρ10(18), P1_D060 ρ10(19), PVC_D070 ρ10(15). PVC ρ20 셀은 두 후보 모두 해당(35·34쌍)이라 K-F4 판정 불가 조건은 아니다 | 판정 json |
| 12 | run_info 정체성 | `protocol` = `"k7"`, `sealed_files` ⊇ `K7_SEALED_FILES`(판정 단언 true, 42개 모두 일치), `mkl_cbwr` = `COMPATIBLE`. `run_info.spec`의 "v1.1" 문자열과 판정 json `note`의 "v1.1"("criteria m210_k7_generator.md v1.1 4-5, 4-6")은 명세 v1.2 §6대로 하네스·판정 스크립트 blob 보존 때문에 남은 것이다(계약 불변) | run_info |
| 13 | 실행 전 감사 기록(공개) | §1-3 | 감사·무결성 지적(메인 세션 전달) |
| 14 | 정체성 기록 파일 | 메인 세션은 `main_session_pre.txt`(15:40:15)·`main_session_post.txt`(16:10:50)에만 썼다. 하네스 `pre_run.txt`·`post_run.txt`는 하네스 형식이다(판정 단언 `pre_run_txt_harness_format`·`post_run_txt_harness_format` true). `harness.err` 0바이트 | check json `identity_files` |
| 15 | c 진단(공개) | §5-3 | 판정 json `K_F6.c_diagnostics` |
| 16 | §9 마지막 행 < 봉인 | 명세 §9 마지막 행 15:02 KST < 재봉인 `git log -1 --format=%cI 8642e29` = 15:03:59 | 명세, check json |
| 17 | 해석 주의 셋(공개) | PVC primary +0.22 ns: K-F4는 차라 영향 없음, K-F5 절대 오차에 들어감(PVC ρ20 `cur` 깊이 오차 중앙 0.0295 m, 다른 단일 관 0.0167–0.0198 m). K7_P1_D060 ≈ 이미 본 S1G_C1: 대조(K-F2) 역할로만 썼고 결과는 a1·a1b2 모두 통과. 층 사건 `ambiguous_with`: U는 알려진 사건 여부만 보므로 영향 없음 | 판정 json `items.*.K_F4`, 명세 §4-3 |
| 18 | 첫 봉인 대체 이력(공개) | §1-4 | 명세 §6-2, check json `seal` |
| 19 | run1 재현 대조(공개) | run1 h5 13개 ↔ 재실행 h5 13개 모두 **배열 비트 동일**(`--expect bit --all-datasets`). 파일 SHA 같음 8개(BG_L·P1_D150_S003·P2_D070_L·P2_D070·REF_L130_S001·REF_S001·REF_S002·REF_S003), 다름 5개(P1_D060·P1_D110_S002·P3_D080·PVC_D070·REF_L085_S001, `track_times`). §1-2 표 | check json `h5_file_sha`·`h5_compare` |

### 1-1. 봉인 파일 blob (항목 1)

| 파일 | blob @8642e29 | blob @d6aaf0f | 첫 봉인 뒤 바뀜 | 작업 트리 = 봉인 |
|---|---|---|---|---|
| `analysis/processing/fourtu_apex_candidate_select.py` | `2318deae01fe31d43485a87ee7b5ea53732fedd6` | `2318deae01fe31d43485a87ee7b5ea53732fedd6` | 아니오 | 예 |
| `analysis/processing/m210_k7_judge.py` | `9773ecb0c72b2b7102b037f57c94f8f7b211b107` | `9773ecb0c72b2b7102b037f57c94f8f7b211b107` | 아니오 | 예 |
| `analysis/processing/m210_k7_pipeline.py` | `3c45d9923a00abb134611be8edece5a0aeae4235` | `3c45d9923a00abb134611be8edece5a0aeae4235` | 아니오 | 예 |
| `analysis/processing/m210_k7_probe.py` | `fc32cfe866b0a8af25cfe03fc8f42e76f74f8b9d` | `fc32cfe866b0a8af25cfe03fc8f42e76f74f8b9d` | 아니오 | 예 |
| `analysis/processing/m210_k7_rank_replay.py` | `04961f8cef645336aa5dc5b19f5af7b9634d9b84` | `04961f8cef645336aa5dc5b19f5af7b9634d9b84` | 아니오 | 예 |
| `analysis/processing/m210_s1g_pipeline.py` | `6a987955ffed671ecd7f6e6841c9d39943c3a117` | `6a987955ffed671ecd7f6e6841c9d39943c3a117` | 아니오 | 예 |
| `analysis/processing/m210b_floor_rules.py` | `bad0f95cdcebe3d193fb7aad6511dbbafcac8327` | `bad0f95cdcebe3d193fb7aad6511dbbafcac8327` | 아니오 | 예 |
| `analysis/processing/m210b_s4s6_pipeline.py` | `ae63d3f04327628ad36425a10b31d8c9ba09798a` | `ae63d3f04327628ad36425a10b31d8c9ba09798a` | 아니오 | 예 |
| `analysis/synthetic/k7/inputs/K7_BG_L.in` | `c1f8e22b0a2e4b32da2b8f733abd67ef070f08d4` | `c1f8e22b0a2e4b32da2b8f733abd67ef070f08d4` | 아니오 | 예 |
| `analysis/synthetic/k7/inputs/K7_P1_D060.in` | `d2fa64294af342b3d4d08d985ed513fdbcce2cc0` | `d2fa64294af342b3d4d08d985ed513fdbcce2cc0` | 아니오 | 예 |
| `analysis/synthetic/k7/inputs/K7_P1_D110_S002.in` | `bd72d9173673a550377fc417f14a742ce99b4acd` | `bd72d9173673a550377fc417f14a742ce99b4acd` | 아니오 | 예 |
| `analysis/synthetic/k7/inputs/K7_P1_D150_S003.in` | `a49e4dd285d9c5d0b969dd6f75cc4c07b3c8a37e` | `a49e4dd285d9c5d0b969dd6f75cc4c07b3c8a37e` | 아니오 | 예 |
| `analysis/synthetic/k7/inputs/K7_P2_D070.in` | `071bc6282e4ac7f2c49816bdefce2bd890eaff45` | `071bc6282e4ac7f2c49816bdefce2bd890eaff45` | 아니오 | 예 |
| `analysis/synthetic/k7/inputs/K7_P2_D070_L.in` | `699421e7ac62621e0fcde3d18dc438946d2d966c` | `699421e7ac62621e0fcde3d18dc438946d2d966c` | 아니오 | 예 |
| `analysis/synthetic/k7/inputs/K7_P3_D080.in` | `8fbf993a95fccf7bb7035bae4c010f5fe9856a4f` | `8fbf993a95fccf7bb7035bae4c010f5fe9856a4f` | 아니오 | 예 |
| `analysis/synthetic/k7/inputs/K7_PVC_D070.in` | `8eddd8cb48224eebdbb731d7f3c87efe79491e1b` | `8eddd8cb48224eebdbb731d7f3c87efe79491e1b` | 아니오 | 예 |
| `analysis/synthetic/k7/inputs/K7_REF_L085_S001.in` | `9ad62f7bb2f412335b805ef2982420e6a2dfae45` | `9ad62f7bb2f412335b805ef2982420e6a2dfae45` | 아니오 | 예 |
| `analysis/synthetic/k7/inputs/K7_REF_L130_S001.in` | `5f8cd7861b4862c791ed0b4a90bfd2daa4a71b33` | `5f8cd7861b4862c791ed0b4a90bfd2daa4a71b33` | 아니오 | 예 |
| `analysis/synthetic/k7/inputs/K7_REF_S001.in` | `e571bec737038db85a38120d586b13194fccd7cc` | `e571bec737038db85a38120d586b13194fccd7cc` | 아니오 | 예 |
| `analysis/synthetic/k7/inputs/K7_REF_S002.in` | `f7a4813fc601fcc2a8bf05c97be5758a2683bda1` | `f7a4813fc601fcc2a8bf05c97be5758a2683bda1` | 아니오 | 예 |
| `analysis/synthetic/k7/inputs/K7_REF_S003.in` | `0d0ed6b92c4fc66cbb863177e3d1c674ce553e08` | `0d0ed6b92c4fc66cbb863177e3d1c674ce553e08` | 아니오 | 예 |
| `analysis/synthetic/k7/models_k7.json` | `a9255e4ec5dc7fb9c2c17f559e1c1bc66af843d2` | `a9255e4ec5dc7fb9c2c17f559e1c1bc66af843d2` | 아니오 | 예 |
| `analysis/synthetic/k7/pilot/P_K7_BG_L.in` | `f887e3cd299e0f833eb913540c5a25f7f4007f24` | `f887e3cd299e0f833eb913540c5a25f7f4007f24` | 아니오 | 예 |
| `analysis/synthetic/k7/pilot/P_K7_P1_D060.in` | `c3d0d268ee28a892e5e4e5a5ba5483f5da9d8d86` | `c3d0d268ee28a892e5e4e5a5ba5483f5da9d8d86` | 아니오 | 예 |
| `analysis/synthetic/k7/pilot/P_K7_P1_D110_S002.in` | `2344327fc9d16bbc9fd43b67dbe1fadae8e1ccb6` | `2344327fc9d16bbc9fd43b67dbe1fadae8e1ccb6` | 아니오 | 예 |
| `analysis/synthetic/k7/pilot/P_K7_P1_D150_S003.in` | `5dcf9bcecb9945faf3062552fba6fb310738eb90` | `5dcf9bcecb9945faf3062552fba6fb310738eb90` | 아니오 | 예 |
| `analysis/synthetic/k7/pilot/P_K7_P2_D070.in` | `f4aaf0beb7f2820e8fda4c2159e5d5048c74baf5` | `f4aaf0beb7f2820e8fda4c2159e5d5048c74baf5` | 아니오 | 예 |
| `analysis/synthetic/k7/pilot/P_K7_P2_D070_L.in` | `cf9b758b74ed631a446914c43b2f74aa893bcbba` | `cf9b758b74ed631a446914c43b2f74aa893bcbba` | 아니오 | 예 |
| `analysis/synthetic/k7/pilot/P_K7_P3_D080.in` | `c695ed15523bbc87dc6858866c5a2aacb21c3c94` | `c695ed15523bbc87dc6858866c5a2aacb21c3c94` | 아니오 | 예 |
| `analysis/synthetic/k7/pilot/P_K7_PVC_D070.in` | `eee48550c8fbf18f19f5dcff69d07c9ee8ddbca1` | `eee48550c8fbf18f19f5dcff69d07c9ee8ddbca1` | 아니오 | 예 |
| `analysis/synthetic/k7/run_k7.cmd` | `8a3d7606f78aada9f0528d0d6dee7429bcfe9358` | `8a3d7606f78aada9f0528d0d6dee7429bcfe9358` | 아니오 | 예 |
| `docs/processing/m210_k7_generator.md` | `2ccd168e177c202f2478cc5cb80611ddd25ded65` | `d532dc99e5b791aa749287050aeb9fc621dfd852` | 예 | 예 |
| `docs/processing/m210_k7_probe.json` | `fda58ce7827fc442bd5d6d82d46f579223363e77` | `fda58ce7827fc442bd5d6d82d46f579223363e77` | 아니오 | 예 |
| `docs/processing/m210_k7_rank_replay.json` | `e99955a7a50582f17b51266398a1b223e6dc2874` | `e99955a7a50582f17b51266398a1b223e6dc2874` | 아니오 | 예 |
| `docs/processing/m29c_qc_constants.json` | `13f2fd3ac76d6996ac9c590a2cadcd988d34a376` | `13f2fd3ac76d6996ac9c590a2cadcd988d34a376` | 아니오 | 예 |
| `scripts/eval_depth_4tu.py` | `2c2f3fc958001d1f5af5a9ea727349d50724ca0e` | `2c2f3fc958001d1f5af5a9ea727349d50724ca0e` | 아니오 | 예 |
| `scripts/synthetic/export_dataset.py` | `00069282273dfce75f1f5db31d9521ff6b6773f6` | `03e78e3bdb1f0094b2b05b3a08390f38af03e2d2` | 예 | 예 |
| `tests/test_m210_k7_generator.py` | `4033f3e12e536bd90c97234cb9178fb0512cb668` | `4033f3e12e536bd90c97234cb9178fb0512cb668` | 아니오 | 예 |
| `tests/test_m210_k7_judge.py` | `653421bb11af522f350c7d217da7ca76d6673366` | `653421bb11af522f350c7d217da7ca76d6673366` | 아니오 | 예 |
| `tests/test_m210_k7_pipeline.py` | `a91f6ce848e0f1038c09da6732e61893bd1f081c` | `a91f6ce848e0f1038c09da6732e61893bd1f081c` | 아니오 | 예 |
| `tests/test_m210b_s4s6_pipeline.py` | `0c654fe8f1c188915158a39c2cbef9ee11c6bd52` | `0c654fe8f1c188915158a39c2cbef9ee11c6bd52` | 아니오 | 예 |
| `tests/test_synthetic_export.py` | `c4b20f9c3041fe29ce7d57d83863ec62047222ca` | `649cffc381d929d8594977dd718493425b3f6bb0` | 예 | 예 |

- 첫 봉인 `d6aaf0f` 뒤 바뀐 파일은 3개다(명세, `scripts/synthetic/export_dataset.py`, `tests/test_synthetic_export.py`). 명세 v1.2 §6의 "재봉인에서 바뀌는 봉인 대상"과 같다.

### 1-2. h5 대조와 source_guard (항목 10·19)

| 보고서 | 기준(reference) | 후보(candidate) | 판정 |
|---|---|---|---|
| `<scratchpad>/h5cmp_k7/pilot_K7_REF_S001.json` | `data/synthetic_gprmax_k7/pilot_prerun_h5/K7_REF_S001_merged.h5` | `data/synthetic_gprmax_k7/h5/K7_REF_S001_merged.h5` | `bit_identical` |
| `<scratchpad>/h5cmp_k7/pilot_K7_REF_S002.json` | `data/synthetic_gprmax_k7/pilot_prerun_h5/K7_REF_S002_merged.h5` | `data/synthetic_gprmax_k7/h5/K7_REF_S002_merged.h5` | `bit_identical` |
| `<scratchpad>/h5cmp_k7/pilot_K7_REF_S003.json` | `data/synthetic_gprmax_k7/pilot_prerun_h5/K7_REF_S003_merged.h5` | `data/synthetic_gprmax_k7/h5/K7_REF_S003_merged.h5` | `bit_identical` |
| `<scratchpad>/h5cmp_k7/run1_K7_BG_L_merged.json` | `data/synthetic_gprmax_k7/run1_d6aaf0f/h5/K7_BG_L_merged.h5` | `data/synthetic_gprmax_k7/h5/K7_BG_L_merged.h5` | `bit_identical` |
| `<scratchpad>/h5cmp_k7/run1_K7_P1_D060_merged.json` | `data/synthetic_gprmax_k7/run1_d6aaf0f/h5/K7_P1_D060_merged.h5` | `data/synthetic_gprmax_k7/h5/K7_P1_D060_merged.h5` | `bit_identical` |
| `<scratchpad>/h5cmp_k7/run1_K7_P1_D110_S002_merged.json` | `data/synthetic_gprmax_k7/run1_d6aaf0f/h5/K7_P1_D110_S002_merged.h5` | `data/synthetic_gprmax_k7/h5/K7_P1_D110_S002_merged.h5` | `bit_identical` |
| `<scratchpad>/h5cmp_k7/run1_K7_P1_D150_S003_merged.json` | `data/synthetic_gprmax_k7/run1_d6aaf0f/h5/K7_P1_D150_S003_merged.h5` | `data/synthetic_gprmax_k7/h5/K7_P1_D150_S003_merged.h5` | `bit_identical` |
| `<scratchpad>/h5cmp_k7/run1_K7_P2_D070_L_merged.json` | `data/synthetic_gprmax_k7/run1_d6aaf0f/h5/K7_P2_D070_L_merged.h5` | `data/synthetic_gprmax_k7/h5/K7_P2_D070_L_merged.h5` | `bit_identical` |
| `<scratchpad>/h5cmp_k7/run1_K7_P2_D070_merged.json` | `data/synthetic_gprmax_k7/run1_d6aaf0f/h5/K7_P2_D070_merged.h5` | `data/synthetic_gprmax_k7/h5/K7_P2_D070_merged.h5` | `bit_identical` |
| `<scratchpad>/h5cmp_k7/run1_K7_P3_D080_merged.json` | `data/synthetic_gprmax_k7/run1_d6aaf0f/h5/K7_P3_D080_merged.h5` | `data/synthetic_gprmax_k7/h5/K7_P3_D080_merged.h5` | `bit_identical` |
| `<scratchpad>/h5cmp_k7/run1_K7_PVC_D070_merged.json` | `data/synthetic_gprmax_k7/run1_d6aaf0f/h5/K7_PVC_D070_merged.h5` | `data/synthetic_gprmax_k7/h5/K7_PVC_D070_merged.h5` | `bit_identical` |
| `<scratchpad>/h5cmp_k7/run1_K7_REF_L085_S001_merged.json` | `data/synthetic_gprmax_k7/run1_d6aaf0f/h5/K7_REF_L085_S001_merged.h5` | `data/synthetic_gprmax_k7/h5/K7_REF_L085_S001_merged.h5` | `bit_identical` |
| `<scratchpad>/h5cmp_k7/run1_K7_REF_L130_S001_merged.json` | `data/synthetic_gprmax_k7/run1_d6aaf0f/h5/K7_REF_L130_S001_merged.h5` | `data/synthetic_gprmax_k7/h5/K7_REF_L130_S001_merged.h5` | `bit_identical` |
| `<scratchpad>/h5cmp_k7/run1_K7_REF_S001_merged.json` | `data/synthetic_gprmax_k7/run1_d6aaf0f/h5/K7_REF_S001_merged.h5` | `data/synthetic_gprmax_k7/h5/K7_REF_S001_merged.h5` | `bit_identical` |
| `<scratchpad>/h5cmp_k7/run1_K7_REF_S002_merged.json` | `data/synthetic_gprmax_k7/run1_d6aaf0f/h5/K7_REF_S002_merged.h5` | `data/synthetic_gprmax_k7/h5/K7_REF_S002_merged.h5` | `bit_identical` |
| `<scratchpad>/h5cmp_k7/run1_K7_REF_S003_merged.json` | `data/synthetic_gprmax_k7/run1_d6aaf0f/h5/K7_REF_S003_merged.h5` | `data/synthetic_gprmax_k7/h5/K7_REF_S003_merged.h5` | `bit_identical` |

| 상태 파일 | snapshot 시각 | 입력 수 | 판정서 작성 때 재verify |
|---|---|---|---|
| `<scratchpad>/k7_run1_h5_guard.json` | 2026-11-12T15:04:19+09:00 | 13 | 종료 0, pass True (2026-11-12T16:16:30+09:00) |
| `<scratchpad>/k7v12_h5_guard.json` | 2026-11-12T15:39:05+09:00 | 14 | 종료 0, pass True (2026-11-12T16:16:30+09:00) |
| `<scratchpad>/k7v12_export_guard.json` | 2026-11-12T15:40:15+09:00 | 33 | 종료 0, pass True (2026-11-12T16:16:30+09:00) |

| h5 | 재실행 SHA-256 | run1 SHA-256 | 파일 SHA 같음 |
|---|---|---|---|
| `FS.h5` | `ae2c3844077f9ba566c443a7215a8f911f1950261386c0a1597b1b599f943bfa` | –(run1 없음, 재사용 FS) | – |
| `K7_BG_L_merged.h5` | `01f150812b9401740c7327a91563ca09fe84b0a64f93a58739a9530cb0a7653c` | `01f150812b9401740c7327a91563ca09fe84b0a64f93a58739a9530cb0a7653c` | 예 |
| `K7_P1_D060_merged.h5` | `ae62a7cd644ffa470128f44d1dbffd9cfb6260d39186af1056c28bd50809a233` | `27cd2b2b170772c1c0d98108155512d01db4530b81c36c923edc2e198dd1a07f` | 아니오 |
| `K7_P1_D110_S002_merged.h5` | `5def41ee2cf99c92b3051395a17bfe1cc06b66f5377dbe0c2121f7beccb3d50e` | `8dbe08dafef238d5fef4319f1625ac08fb7737ae0d7c8b7e5ae183ac6bbe3eb9` | 아니오 |
| `K7_P1_D150_S003_merged.h5` | `e7b228279d251c22f66869f74d6e6b015e8e9dd21341aad3933091931dd7c0f1` | `e7b228279d251c22f66869f74d6e6b015e8e9dd21341aad3933091931dd7c0f1` | 예 |
| `K7_P2_D070_L_merged.h5` | `1c2a2c20959646af015ccdb10d21796e92c628d4e42c51cb3e3165bedf11a6fc` | `1c2a2c20959646af015ccdb10d21796e92c628d4e42c51cb3e3165bedf11a6fc` | 예 |
| `K7_P2_D070_merged.h5` | `a5484045a1aee7b2aa02ce68b1f5ea258461a78e0ba614b5026bbcf840d92c90` | `a5484045a1aee7b2aa02ce68b1f5ea258461a78e0ba614b5026bbcf840d92c90` | 예 |
| `K7_P3_D080_merged.h5` | `0dd0f020ed46a7bb7ed9feafd1986465848aa859b5178b15ad359bc415ea644d` | `289bad49060f5f2b92e8a3ed7ad04cce61aa9f7e6be859f9100a720e52874467` | 아니오 |
| `K7_PVC_D070_merged.h5` | `ab383c8ef40724146cb0219a43082d6e9bb666a9a4664b4cc3071dc90c6e3ab1` | `fc944df0e5325604131cf36754e6638bf16d12dcc41fbf519470802bff55128f` | 아니오 |
| `K7_REF_L085_S001_merged.h5` | `e3d304d23fd8b1546229d674b98fd8113969e2ed58198abdbd7229ff8356aadd` | `89669619b618aff052916a87c5aed7dcd0ee95d0d903565c39368d1c48f30e55` | 아니오 |
| `K7_REF_L130_S001_merged.h5` | `dbddd445ba52ac82e99aaeac9b90e15ea4fe58fef5e60290b287342d513fdbcf` | `dbddd445ba52ac82e99aaeac9b90e15ea4fe58fef5e60290b287342d513fdbcf` | 예 |
| `K7_REF_S001_merged.h5` | `f7e733eafa39f44260e3f108b8e0d7b52ec19322e97192bdaf3cf5de1e6332fc` | `f7e733eafa39f44260e3f108b8e0d7b52ec19322e97192bdaf3cf5de1e6332fc` | 예 |
| `K7_REF_S002_merged.h5` | `3da57642274c1663353843176d02f77b23296302f9f116b32d42568b7324c3a9` | `3da57642274c1663353843176d02f77b23296302f9f116b32d42568b7324c3a9` | 예 |
| `K7_REF_S003_merged.h5` | `761c2c4c5aedd71f2aee8be5638d8bfd0859fb7bd19b5df39647d08345713ef2` | `761c2c4c5aedd71f2aee8be5638d8bfd0859fb7bd19b5df39647d08345713ef2` | 예 |

### 1-3. 실행 전 감사 기록 (항목 13, 공개 — 감사 원문 요지, 원문은 메인 세션 전달이며 파일로 남지 않음)

원문 번호를 함께 적는다. 표의 내용은 값과 무관하게 순위 1 사유가 아니다.

**실행 전 감사(v1.1 봉인 전, 원문 1–17)**

| 원문 # | 요지 표지 | 지적 요지 | 판정서에 적을 것 / 반영 |
|---|---|---|---|
| 1 | A1 | §9 행마다 시각 | 명세 §9에 KST 시각(항목 16) |
| 2 | A2 | 상속 봉인 변경 공개에 export·export 시험 추가 | 명세 §4-2, (b) 판정은 각 봉인 커밋에서 재현 |
| 3 | A3 | "v1.0 봉인판" 표기 | 명세 §6 v1.1 → v1.2 |
| 4 | A4 | 순위 SNR 위치 확정 | 끝점의 D, 현행과 같은 위치(명세 §2-1·§5) |
| 5 | A5 | 항목 10 대조 기준, 모델 h5 snapshot, 비트 대조는 S001–S003만 | 이 판정서 항목 10 |
| 6 | A6 | 시제품 스크립트 mtime이 출력보다 늦음 | 명세 §9 공개(문서 문자열만 바뀜) |
| 7 | A7 | 층 참조 입력이 점검 도중 추가됨 | 명세 §9 11:20 행(점검 출력은 열지 않음) |
| 8 | A8 | 비교 명령 `--out` 필수·옮긴 경로 | 명세 §6-1 4 |
| 9 | A9 | 관 수 출처, a1 동률, τ ≤ 0 대체, 시험 수, pytest 수 기록 방식 | 명세 반영. pytest 실행 시점 값은 항목 8 |
| 10 | A10 | 봉인 전 G0 근거(python-expert 구현 보고)는 보고일 뿐이고, 구속력 있는 G0는 하네스 끝의 G0다 | 하네스 끝 G0 결과는 항목 8(모두 동일) |
| 11 | A11 | `docs/processing/m29c_qc_constants.json` 작업 트리가 CRLF라 `run_info.environment.qc_constants.matches_locked_sha_quote` = false(원시 바이트 SHA-256 `cec7615a1a1c19cd5ce1a48c7aa7fb09202229368b26eb70eac8993df88d35a5`) | LF 기준 SHA-256 `6a88bd8d…ec56ba8c`는 명세와 같고, blob `13f2fd3ac76d6996ac9c590a2cadcd988d34a376`도 봉인 커밋과 같다. `code_constants_differing`는 빈 목록(QC 상수 15개 일치). 판정 스크립트는 이 값을 보지 않는다. **정체성 불일치가 아니다** |
| 12 | A12 | 관 수 출처 | 모델 json `pipes` 길이(항목 9) |
| 13 | A13 | 관별 행 수 단언 | 항목 9(고정 집합 단언 pass) |
| 14 | A14 | 자식 pytest의 skip 수 기록 | 항목 8: 94 passed, skip 0 |
| 15 | A15 | [R] a1 오르막의 동률 처리(늦은 쪽 = 뒤쪽 우선)가 구현 해석 목록에 없음 | 명세 §4-4 구현 해석 (i)·§2-1에 반영 |
| 16 | A16 | [R] export K7 시험 개수 표기(명세 "3개", 실제 4개) | v1.1에서 4개로, v1.2에서 6개(`test_k7*` 5 + `test_a_lone_pipe_without_picks_still_fails`)로 정정 |
| 17 | A17 | 시간 순서 | 정상(항목 6·16) |

**무결성 점검(v1.1 봉인 전)**

| # | 지적 요지 | 반영 |
|---|---|---|
| I10–I14 | 홈 경로 제거, CSV CRLF 기준, 구현 해석 표지, 메모리 쓰기 금지, 인용 출처 | 명세 반영. CRLF 기준은 항목 8 |

**v1.1 재감사 기록(원문 1–3)**

| 원문 # | 지적 요지 | 판정서에 적을 것 / 반영 |
|---|---|---|
| 1 | 미커밋 메모리·CHECKLIST를 봉인 커밋에 포함 | 포함됨(봉인 커밋에서 dirty 0, 항목 4) |
| 2 | 명세를 고치면 §9에 행 추가 | 명세 §9 반영(마지막 행 15:02, 항목 16) |
| 3 | `models_k7.json`의 "봉인 blob 내용" 조건 | 간접 사슬로 성립: 하네스가 HEAD = 봉인·dirty 0·봉인 blob 일치 상태에서 원시 SHA-256을 기록하고, 판정이 그 값을 재계산 값과 맞춘다(`models_json_sha256_matches_run` true). 작업 트리가 LF라 원시 SHA-256 = blob 내용의 SHA-256이다 |

**v1.2 재봉인 전 감사 기록(원문 1–4)**

| 원문 # | 분류 | 지적 요지 | 반영 |
|---|---|---|---|
| 1 | T | 마른 내보내기에서 본 범위 공개 | python-expert 사실 보고를 명세 §6-2에 공개(이 판정서 §1-4) |
| 2 | T | 명세 §9의 14:37 결정 행이 내보내기 수정 mtime 14:36:56보다 늦음 | 결정 시각(14:36 이전, 기록 없음)과 기록 시각(14:37)을 나눠 정정 |
| 3 | R | 명세의 상속 변경·시험 개수 표기가 v1.2 기준으로 낡음 | 정정(+197/−17줄, +195/−0줄, export K7 시험 6개; 명세 §4-2·§4-4·§5) |
| 4 | I·S | 불변 파일의 "v1.1" 판 문자열, 내보내기 `MKL_CBWR=COMPATIBLE`, CHECKLIST | 판 문자열은 blob 보존 때문에 유지(항목 12). `MKL_CBWR=COMPATIBLE`은 운영 규칙(하네스가 검사하지 않음, json 보고 값 끝자리만 영향; 명세 §6-1 6). CHECKLIST는 재봉인 커밋에 포함 |

- 같은 시기의 문구 정정(봉인판 표기 v1.2, 콘솔 요약 `vg/vt=n/a (no usable picks)`, §9 마지막 행)도 명세에 반영됐다.

### 1-4. 첫 봉인 대체 이력 (항목 18, 공개)

- 첫 봉인 `d6aaf0f`(v1.1, 12:48:54). run1 gprMax 13단계 12:50–14:26 모두 exit 0(레인 B 12:59–14:00 conda `reg.exe` Suspended → 사용자 승인 뒤 재개). `K7_REF_S001`–`S003`는 점검 산출과 배열 비트 동일.
- 내보내기 14:27:00–14:27:41 실패: `K7_P3_D080 pipe 1: no usable per-trace picks for the velocity check`. 하네스·판정은 돌지 않아 run1에는 판정 결과가 없다.
- 판정 입력이 아닌 내보내기 진단 키(`velocity_check`)의 예외 처리를 고쳤다(이웃 관이 있고 쓸 점 0이면 `velocity_check: null` + `velocity_check_note: "no_usable_picks_neighbours"`). 그다음 v1.2로 재봉인(`8642e29`)했다. blob이 바뀐 봉인 파일은 §1-1의 3개다.
- 재봉인 전 마른 내보내기(보관 run1 h5)에서는 내보내기 진단값과 하네스 쪽 읽기 결과만 봤다. 생성기·하네스 행 함수·판정은 부르지 않았고 성공률·깊이·후보 위치는 보지 않았다(명세 §6-2).
- run1 h5는 판정에 쓰지 않았다. 재실행 h5와 배열 비트 동일임만 확인했다(항목 19).

## 2. 재봉인 실행 요약

| 단계 | 시각(KST) | 결과 |
|---|---|---|
| gprMax 13단계(2레인) | 15:04:25–15:38:11 | 모두 exit 0, 단계당 21–498 s |
| 배열 비트 대조 | 15:38 | 재실행 = run1(13), S001–S003 = 점검 산출(3) |
| h5 snapshot(14) | 15:39:05 | – |
| 내보내기 | 15:39:05–15:39:59 | exit 0, 33파일, `MKL_CBWR=COMPATIBLE` |
| 내보내기 snapshot(33) | 15:40:15 | – |
| 하네스 | 15:40:23–16:10:32 | 3,888행, guards pass, 입력 변경 없음 |
| verify | 하네스 뒤 | 두 snapshot pass(14·33, 변경 0) |
| 판정 | 16:11:08 | 종료 0 |

## 3. K-R — 결함 재현 (봉인 정의, 판정에 씀)

| 기작 | 모델 | ρ | `cur` 수락 단계 | `cur` kept 단계 | gap | 기준 | 재현 |
|---|---|---|---|---|---|---|---|
| a | K7_P2_D070 | 20 | 0.950 | 0.875 | 0.075 | ≥ 0.5 | 아니오 |
| a | K7_P2_D070_L | 20 | 0.9125 | 0.775 | 0.1375 | ≥ 0.5 | 아니오 |
| b | K7_P3_D080 | 20 | 0.550 | 0.2917 | 0.2583 | ≥ 0.5 | 아니오 |

- K-R(a)는 `defect_a` 가운데 하나 이상, K-R(b)는 `defect_b`에서 gap ≥ 0.5여야 성립한다. 둘 다 성립하지 않는다.

### 3-1. 사전 예측 대비 왜 재현되지 않았나 (물리 기술 — 진단, 판정 무관, 기준 변경 없음)

출처: check json `diagnostics`(행 집계). `kept_offset` = kept 후보 시각 − $t_{a0}$(ns).

1. **잡음 없는 행에서는 결함이 그대로 재현됐다.**
   - K7_P2_D070_L(층 0.85 m): `cur`는 두 관 모두 kept 실패(`early_lobe_nms`), 시작점이 $E$ 국소 최대가 아님. a1은 두 관 모두 kept, 꼭짓점과 +0.008 ns. S1g D100과 같은 기작이다.
   - K7_P2_D070(층 없음): `cur`도 kept 성공(+0.029 ns). 층이 행 수준 $\ell$의 기울기를 만든다는 v1.0 설계 가정(§4-3 "a(ℓ 기울기 강화)")과 맞는다.
   - K7_P3_D080: `cur`는 바깥 두 관을 `other_nms`로 잃고 가운데 관만 kept(−0.859 ns). a1b2만 세 관을 모두 kept.
2. **잡음이 있으면 결함은 남지만 창 손실로 거의 이어지지 않는다.**
   - `cur` kept 후보의 시작점은 대부분 $E$ 국소 최대가 아니다(국소 최대 비율 P2_D070 ρ20 0.057, P2_D070_L ρ20 0.129). 곧 오르는 날 시작점은 잡음에서도 흔하다.
   - 그러나 그 끝점은 꼭짓점 앞 1 ns 안에 든다. `cur` kept 시각 차(10/50/90 %)는 P2_D070_L ρ20에서 −0.637/−0.441/−0.012 ns, P2_D070 ρ20에서 −0.637/−0.115/−0.005 ns다. 봉인 성공 창은 $t_{a0}\pm0.5/f_c$(±1 ns)라 이 후보들은 kept 성공으로 센다.
   - a1은 같은 셀에서 −0.094/−0.046/0.000 ns, −0.086/−0.045/−0.001 ns로 끝점을 꼭짓점에 붙인다. 위치 결함(오르는 날)은 a1이 고치지만, K-R이 재는 "창 밖으로 밀려 지워짐"은 잡음 조건에서 드물었다.
   - 기작 해석(추정, 직접 측정 안 함): 검출 정규화 $D=E/\ell$의 극대는 $\partial_k\ln E=\partial_k\ln\ell$인 곳이다. 잡음이 없으면 $\ell$(행 중앙값)은 신호 사건이 정하고 바닥 $F$ 아래로는 잘리므로 꼭짓점 행에서 급하게 오를 수 있다(S1g D100 1.33 → 8.73). 잡음을 더하면 161트레이스 행 중앙값의 대부분이 잡음 포락선이 되어 $\ell$이 시간에 따라 평평해진다. 그러면 $\partial_k\ln\ell$이 작아져 $D$ 극대의 앞당김이 1 ns보다 작아진다. 위 시각 차 분포가 이 방향과 맞는다.
3. **기작 b는 이 모델에서 주된 손실이 아니었다.**
   - K7_P3_D080 ρ20 `cur` 손실 사유(관 120행): `no_peak` 54, `nms` 26(그중 `early_lobe_nms` 2), `cap6` 5. 측선당 상한 6에 걸린 것은 5행뿐이다. 0.6 m 간격 세 관과 1.30 m 층에서는 수락 전 단계(`no_peak`)의 손실이 컸다.
   - S1g D060에서 primary를 자른 것은 상한 6이었다(이미 본 자료). 새 모델에서는 같은 상황(긴 날개 경쟁자가 6개 넘게 수락됨)이 잡음 아래에서 드물게만 생겼다.
4. 정리: 사전 예측(명세 v1.0 §2·§4-5)은 "잡음 ρ 20에서도 결함이 창 손실(gap ≥ 0.5)로 재현된다"였다. 실제로는 결함의 원인(오르는 날 시작점)은 잡음에서도 흔했지만, 손실 크기가 봉인 창 기준(±1 ns)을 넘는 경우가 적었다. 이 판정은 봉인 기준대로 보류(순위 2)이고, 위 해석으로 기준을 바꾸지 않는다.

## 4. 후보별 F 항목 (참고 — 순위는 순위 2에서 이미 정해짐)

결정표는 위에서부터 처음 맞는 행으로 정한다. a1은 순위 2에서 멈췄으므로 아래 값은 순위를 바꾸지 않는다.

| 항목 | a1 | a1b2(참고) |
|---|---|---|
| K-F1 회복 | 실패 3/4셀(P2_D070 ρ10 0.500 < 0.5375, P2_D070_L ρ10 0.4125 < 0.475, ρ20 0.750 < 0.8125; P2_D070 ρ20 0.850 = 0.85 통과) | 실패 4/6셀(P2_D070 ρ20 0.8375 < 0.85, P2_D070_L ρ20 0.775 < 0.8125, P3_D080 ρ10 0.300 < 0.383, ρ20 0.100 < 0.450) |
| K-F2 무손실 | 통과 8/8 | 통과 8/8 |
| K-F3 거짓 | 통과 16/16(P3_D080 ρ20 U 0.775 → 0.925, +0.15) | 통과 16/16 |
| K-F4 깊이 | 통과(해당 셀 10개, 차 최대 0.0003 m) | 통과(해당 셀 9개, 차 최대 0.0002 m) |

- 주목할 것(진단): a1은 K7_P3_D080에서 수락 단계 성공률을 크게 낮췄다(ρ10 0.483 → 0.342, ρ20 0.550 → 0.117). 손실 사유는 `no_peak`(능선 이동 뒤 창 안에 드는 검출 피크 없음)이 54 → 106행으로 늘었다. 0.6 m 간격에서는 시작점을 원 포락선 국소 최대로 올리면 이웃 관이나 다른 사건의 극대로 올라가, 능선 이동 끝이 창 밖으로 가는 경우가 많아진 것으로 본다(추정). S1g·시제품에서는 보지 못한 부작용이다.

## 5. 보고 항목 (판정 무관, 판정 json에서 그대로 옮김)

### 5-1. K-F5 깊이 성공($\lvert\hat d-d_\text{top}\rvert\le0.10$ m인 QC 통과 primary 비율)

| 모델 | ρ10 cur / a1 / a1b2 / a1b1 | ρ20 cur / a1 / a1b2 / a1b1 |
|---|---|---|
| K7_P2_D070 | 0.6625 / 0.625 / 0.625 / 0.625 | 0.9375 / 0.9375 / 0.95 / 0.9375 |
| K7_P2_D070_L | 0.5125 / 0.5125 / 0.60 / 0.5125 | 0.875 / 0.875 / 0.8875 / 0.875 |
| K7_P3_D080 | 0.35 / 0.30 / 0.275 / 0.30 | 0.25 / 0.10 / 0.10 / 0.10 |
| K7_P1_D150_S003 | 0.475 / 0.475 / 0.40 / 0.475 | 0.875 / 0.90 / 0.925 / 0.90 |
| K7_P1_D110_S002 | 0.50 / 0.50 / 0.475 / 0.50 | 0.925 / 0.925 / 0.90 / 0.925 |
| K7_P1_D060 | 0.475 / 0.475 / 0.50 / 0.475 | 0.975 / 0.975 / 0.975 / 0.975 |
| K7_PVC_D070 | 0.375 / 0.375 / 0.375 / 0.375 | 0.90 / 0.875 / 0.85 / 0.875 |

### 5-2. K-F6 후보 수·손실·시작점

- 평균 kept 수(측선당): 변형 사이 차가 작다. 가장 큰 것은 K7_P3_D080 ρ20 `a1b1` 8.075(상한 제거, 다른 변형 5.925)다. `a1b1` 후보 수 비용: kept 합 2,206(a1) → 2,318(a1b1), 1.051배.
- 시작점 수(`n_window_starts`)별 kept 성공률: 0개면 모든 변형 0, 1–9개에서 a1 0.48–1.0(판정 json `K_F6.kept_rate_by_window_starts`).
- 손실 사유 전체는 판정 json `K_F6.lost_reasons`.

### 5-3. c 진단 (L3, 공개, 판정 무관)

| 모델 | 변형 | 항목 있는 행 | 끝점 옆 오차 10/50/90 %(트레이스) | ±3 안 | 시작점 국소 최대 | 끝점 국소 최대 | τ² 꼭짓점 ±3 안 |
|---|---|---|---|---|---|---|---|
| K7_P2_D070 | cur | 127 | −2 / 0 / 2.4 | 1.00 | 0.079 | 0.520 | 0.961 |
| K7_P2_D070 | a1 | 123 | −2 / 0 / 2.8 | 1.00 | 1.00 | 1.00 | 0.976 |
| K7_P2_D070_L | cur | 119 | −2 / 0 / 2 | 1.00 | 0.101 | 0.454 | 0.941 |
| K7_P2_D070_L | a1 | 114 | −2 / 0 / 2 | 1.00 | 1.00 | 1.00 | 0.956 |
| K7_P3_D080 | cur | 124 | −2 / 0 / 2.7 | 1.00 | 0.129 | 0.242 | 0.863 |
| K7_P3_D080 | a1 | 55 | −2.6 / 0 / 2.6 | 1.00 | 1.00 | 1.00 | 0.655 |

- 나머지 모델·변형은 판정 json `K_F6.c_diagnostics`.
- 읽기: 창 안 항목의 옆 오차는 모든 변형에서 ±3 트레이스 안이다. K7-c(옆 흔들림)는 이 모델들(깊이 0.7–0.8 m)에서 kept 창을 잃게 하지 않았다. `cur`의 끝점이 $E$ 국소 최대인 비율은 0.24–0.52로, 오르는 날 위 끝점이 흔하다는 §3-1의 관찰과 같다.

## 6. S1g 원 결함과의 관계 (이미 본 자료)

- S1g D100(잡음 없음, 관 2개 1.0 m 간격, 0.60 m)의 손실은 K7_P2_D070_L의 **잡음 없는 행**에서 같은 형태로 재현됐다. a1이 그것을 고친다는 것도 같다.
- 그러나 판정 조건(ρ 10·20의 잡음)에서는 같은 결함이 창 손실로 거의 이어지지 않았다(§3-1). S1g의 손실은 잡음이 없는 합성에서 $\ell$이 신호로만 정해진 조건에 크게 기대고 있었던 것으로 본다(추정).
- S1g D060(상한 6)형은 K7_P3_D080에서 드물었다(`cap6` 5/120행).
- 4TU 관점(이미 본 자료, 탐색): 명세 §1-1에서 4TU 현행 kept 가운데 끝점이 $E$ 국소 최대가 아닌 것은 2.5 %였다. 실측 자료의 잡음·클러터가 $\ell$을 평평하게 만드는 쪽이라면 K7-a의 4TU 영향은 작을 것이다. 이 판정은 4TU를 재지 않았다.

## 7. 결론과 다음 선택지

- **판정**: 순위 2(보류, 재현 조건 미달). 채택 = 현행 생성기. 플래그 기본값은 현행과 바이트 동일(G0)이므로 4TU 산출물은 바뀌지 않는다.
- 배운 것(진단):
  1. 오르는 날 시작점(K7-a의 원인)은 잡음에서도 흔하지만 대개 꼭짓점 앞 1 ns 안이다. a1은 그 위치를 꼭짓점에 붙인다(중앙 −0.44 → −0.05 ns, P2_D070_L ρ20).
  2. a1은 0.6 m 간격 다중 관에서 수락 단계 성공률을 크게 낮추는 부작용이 있다(P3_D080 ρ20 0.55 → 0.12).
  3. 깊이 정확도(K-F4·K-F5)에는 a1·a1b2의 이득이 없다.
- 다음 선택지:
  - (가) **K7 닫고 현행 생성기 유지.** a1·a1b2는 채택하지 않는다. 생성기 플래그 코드는 기본값이 현행이므로 그대로 둘 수 있다.
  - (나) 재현 조건을 결과 전에 다시 정한 새 실현(v2): 예를 들어 K-R을 "kept 위치 오차"로 정의하거나 잡음을 낮춘 조건을 넣는다. 이번 결과를 본 뒤의 착상이므로 새 판·새 실현·새 시드로만 할 수 있다. a1의 다중 관 부작용(§4)을 먼저 다뤄야 한다.
  - (다) M2-10의 다른 항목으로 간다: (a) 메타 εr 잔여 bias, (c) P3 이동 정합 QC·P4·P5, (f) 첫 사건이 약한 정답 없는 28측선, (g) U-2 상단/중심 미결 기록.

## 8. 가정·한계

- 2D 합성(원통 확산)이라 진폭에 기대는 단계(검출 정규화, SNR 순위, 정지 0.2)는 정성 판단만 가능하다. §3-1의 $\ell$ 해석은 행 수준을 직접 재지 않은 추정이다.
- 성공 창 ±1 ns는 봉인 정의다. 창을 좁히면 결과가 달라질 수 있지만, 그것은 결과를 본 뒤의 착상이라 이 판정에 쓰지 않는다.
- K7_P1_D060은 이미 본 S1G_C1과 물리적으로 거의 같다(대조로만 씀).
- 깊이 기준점은 관 상단(json `pipe_top_depth_m`)이다.
- 감사 원문(항목 13)은 메인 세션 대화로 전달됐고 파일로 남지 않았다. 표는 요지다.
