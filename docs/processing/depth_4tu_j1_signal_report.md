# M3-3 J1 보고 실행 `report:signal` (판정 아님)

- 생성: `analysis/processing/m33_j1_judge.py`, 2026-11-16T15:15:20+09:00, HEAD 1a73ba84a6ce80a75171e17f8b7040e2245f26c3, 509.7 s
- 박스 입력: `None` (sha256 None)
- 후보 csv: `docs/processing/fourtu_apex_candidates_v1.csv` (sha256 1afd23cfab30b12aa128a1abe9fa4053ab5126fd3513b8b4a1dcf539fb3ef268)
- 깊이 계산: `scripts/eval_depth_4tu.py` (무수정), prefix `depth_4tu_j1_signal`
- 상태: `ok`

## 함께 적는 값 (판정에 쓰지 않음)

| 항목 | 값 |
|---|---|
| 합격 짝 n_pass / 388 | 140 / 388 |
| 판정률 B | 0.3608 |
| k (기준선 시행 1,000회 중 관측 이상) | 1000 |
| p = k / 1000 | 1.0000 |
| 짝 수 n_pairs | 184 |
| 판정률 A | 0.7609 |
| B_max = n_pairs / 388 | 0.4742 |
| 기준선 판정률 A 평균 | 0.9513 |
| MAE (m) | 0.2096 |
| bias (m) | -0.1596 |
| 누락 / 오검출 | 204 / 21 |

## 실질 지표 (보고, 판정 아님)

| 항목 | 값 |
|---|---|
| B | 0.3608 |
| 기준선 B 평균 | 0.4511 |
| 기준선 B 2.5 % | 0.4381 |
| 기준선 B 97.5 % | 0.4639 |
| Δ_B = B − 기준선 평균 | -0.0903 |

## 보고 항목 (J1-2 보고 계산, 판정 아님)

### p 도달 가능성

| 항목 | 값 |
|---|---|
| k_perfect | 0 |
| 관측이 완벽해도 p < 0.05 불가 | False |
| n_req = T_(50) + 1 | 180 |
| A_req = n_req / n_pairs | 0.9783 |
| 초과 추정 E_+ (= summary false_est) | 21 |
| n_e,a > n_g,a 활동 수 | 13 |
| 정답 1개이면서 추정 있는 활동 수 | 6 |

### 현장 단위 불확도

| 빠진 현장 | n_pass | N_g | B_-s | 기준선 평균 | Δ_B,-s | k_-s |
|---|---|---|---|---|---|---|
| 1 | 115 | 338 | 0.3402 | 0.4157 | -0.0754 | 1000 |
| 3 | 120 | 342 | 0.3509 | 0.4425 | -0.0916 | 1000 |
| 4 | 130 | 364 | 0.3571 | 0.4537 | -0.0966 | 1000 |
| 5 | 133 | 362 | 0.3674 | 0.4587 | -0.0913 | 1000 |
| 6 | 116 | 347 | 0.3343 | 0.4328 | -0.0985 | 1000 |
| 8 | 131 | 368 | 0.3560 | 0.4399 | -0.0839 | 1000 |
| 10 | 115 | 302 | 0.3808 | 0.4714 | -0.0906 | 1000 |
| 11 | 127 | 320 | 0.3969 | 0.5043 | -0.1074 | 1000 |
| 12 | 135 | 369 | 0.3659 | 0.4459 | -0.0800 | 1000 |
| 13 | 138 | 380 | 0.3632 | 0.4527 | -0.0896 | 1000 |

- 현장 부트스트랩(`default_rng(20261115)`, 2000회, 현장 10곳): B* 2.5–97.5 % 0.2743–0.4564, Δ_B* 2.5–97.5 % -0.1451–-0.0466

- 목표 경계(±9)는 짝을 독립으로 본 이항 1 SE이고 현장 군집을 반영하지 않는다. 현장이 10곳뿐이라 현장 부트스트랩 구간은 실제 불확도보다 좁게 나올 수 있다.

## 후보 수

- n_candidates: 3615
- 깊이 계산 후보 진단: {'status': {'ok': 3515, 'fit_failed': 100}, 'qc_verdict': {'unknown': 1667, 'in': 1016, 'out': 932}, 'apex_at_record_end': {'False': 3450, 'True': 165}, 'apex_on_trace_seam': {'False': 3478, 'True': 137}, 'qc_excluded': {'True': 2615, 'False': 1000}}

## 단언

| 단언 | 결과 | 비고 |
|---|---|---|
| J1-3.verdict_written_first | 통과 | depth_4tu_j1_verdict.json |
| J1-3.no_report_output | 통과 |  |
| 1.c2_seal_filled | 통과 |  |
| args.prefix | 통과 | depth_4tu_j1 |
| args.eval_out_dir | 통과 |  |
| 1.tree_clean | 통과 |  |
| 1.blob.scripts/eval_depth_4tu.py | 통과 |  |
| 1.sha_lf.scripts/eval_depth_4tu.py | 통과 |  |
| 1.blob.analysis/processing/m33_j1_candidates.py | 통과 |  |
| 1.sha_lf.analysis/processing/m33_j1_candidates.py | 통과 |  |
| 1.blob.analysis/processing/m33_j1_judge.py | 통과 |  |
| 1.sha_lf.analysis/processing/m33_j1_judge.py | 통과 |  |
| 1.blob.docs/processing/fourtu_lines_v1.csv | 통과 |  |
| 1.sha_lf.docs/processing/fourtu_lines_v1.csv | 통과 |  |
| 1.blob.docs/datasets/fourtu_duplicate_radargrams.csv | 통과 |  |
| 1.sha_lf.docs/datasets/fourtu_duplicate_radargrams.csv | 통과 |  |
| 1.blob.docs/processing/m29c_qc_constants.json | 통과 |  |
| 1.sha_lf.docs/processing/m29c_qc_constants.json | 통과 |  |
| 1.src_tree_id | 통과 | 84b02beda30544960d4d2ac02cf461fe66556c4f |
| 1.head_is_c2_or_docs_only | 통과 | c2=b6d5d8f7d8d0afad5e21509c9915feddb46cc8a2; non-docs changes: [] |
| 3.sealed_pp_c0b | 통과 | S3 |
| 1.fourtu_meta.utilities_groundtruth.csv | 통과 |  |
| 1.fourtu_meta.utilities_metadata_list.csv | 통과 |  |
| 1.fourtu_meta.radargrams_index.csv | 통과 |  |
| 1.fourtu_meta.activities_metadata_ko.csv | 통과 |  |
| args.boxes_is_fold0_post | 통과 |  |
| 3.status_ok | 통과 |  |
| 3.judgeable | 통과 |  |
| 3.full_run | 통과 |  |
| 3.no_blockers | 통과 |  |
| 3.all_folds | 통과 |  |
| 3.j1_n_lines_754 | 통과 |  |
| 3.n_lines_754 | 통과 |  |
| 3.fold0_weights_sha | 통과 |  |
| 3.settings_sha_c1 | 통과 |  |
| 3.r0_11_lines_c1 | 통과 |  |
| 3.pp_config_sha_c0b | 통과 |  |
| 3.pp_set_id_c0b | 통과 |  |
| 3.r2_head_after_c1 | 통과 | 68f4719db43d2d9c2ec45e10d8232ddd39597f44 |
| 3.r2_tree_clean | 통과 |  |
| signal.candidates_sha_u3 | 통과 |  |
| 5.candidates_sha_unchanged | 통과 |  |
| 1-1.3.one_main_row | 통과 | 1 |
| 1-1.3.n_gt_388 | 통과 |  |
| 1-1.3.n_pass_round_pass_b | 통과 |  |
| 1-1.3.match_rows_n_pairs | 통과 |  |
| 1-1.3.summary_n_pass | 통과 |  |
| 1-1.3.k_integer | 통과 |  |
