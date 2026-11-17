# M3-3 J1 보고 실행 `report:fold3` (판정 아님)

- 생성: `analysis/processing/m33_j1_judge.py`, 2026-11-16T15:04:25+09:00, HEAD 1a73ba84a6ce80a75171e17f8b7040e2245f26c3, 362.3 s
- 박스 입력: `runs/eval/m33/r2/fold3/j1_boxes_post.csv` (sha256 23cd2668c20ca14775932652a483653794487d2a8668ae881b96dcc1807a11dd)
- 후보 csv: `docs/processing/depth_4tu_j1_f3_candidates_input.csv` (sha256 5ddf31834be1eca18e2ab1da1acd33480ca2be7a308adbc4507feda705cb5a1a)
- 깊이 계산: `scripts/eval_depth_4tu.py` (무수정), prefix `depth_4tu_j1_f3`
- 상태: `ok`

## 함께 적는 값 (판정에 쓰지 않음)

| 항목 | 값 |
|---|---|
| 합격 짝 n_pass / 388 | 42 / 388 |
| 판정률 B | 0.1082 |
| k (기준선 시행 1,000회 중 관측 이상) | 1000 |
| p = k / 1000 | 1.0000 |
| 짝 수 n_pairs | 56 |
| 판정률 A | 0.7500 |
| B_max = n_pairs / 388 | 0.1443 |
| 기준선 판정률 A 평균 | 0.9459 |
| MAE (m) | 0.2161 |
| bias (m) | -0.0449 |
| 누락 / 오검출 | 332 / 10 |

## 실질 지표 (보고, 판정 아님)

| 항목 | 값 |
|---|---|
| B | 0.1082 |
| 기준선 B 평균 | 0.1365 |
| 기준선 B 2.5 % | 0.1289 |
| 기준선 B 97.5 % | 0.1418 |
| Δ_B = B − 기준선 평균 | -0.0283 |

## 보고 항목 (J1-2 보고 계산, 판정 아님)

### p 도달 가능성

| 항목 | 값 |
|---|---|
| k_perfect | 14 |
| 관측이 완벽해도 p < 0.05 불가 | False |
| n_req = T_(50) + 1 | 56 |
| A_req = n_req / n_pairs | 1.0000 |
| 초과 추정 E_+ (= summary false_est) | 10 |
| n_e,a > n_g,a 활동 수 | 5 |
| 정답 1개이면서 추정 있는 활동 수 | 6 |

### 현장 단위 불확도

| 빠진 현장 | n_pass | N_g | B_-s | 기준선 평균 | Δ_B,-s | k_-s |
|---|---|---|---|---|---|---|
| 1 | 38 | 338 | 0.1124 | 0.1419 | -0.0295 | 1000 |
| 3 | 39 | 342 | 0.1140 | 0.1403 | -0.0262 | 1000 |
| 4 | 42 | 364 | 0.1154 | 0.1455 | -0.0301 | 1000 |
| 5 | 32 | 362 | 0.0884 | 0.1132 | -0.0248 | 1000 |
| 6 | 39 | 347 | 0.1124 | 0.1411 | -0.0287 | 1000 |
| 8 | 42 | 368 | 0.1141 | 0.1439 | -0.0298 | 1000 |
| 10 | 31 | 302 | 0.1026 | 0.1246 | -0.0219 | 1000 |
| 11 | 34 | 320 | 0.1062 | 0.1472 | -0.0409 | 1000 |
| 12 | 40 | 369 | 0.1084 | 0.1307 | -0.0223 | 1000 |
| 13 | 41 | 380 | 0.1079 | 0.1368 | -0.0289 | 1000 |

- 현장 부트스트랩(`default_rng(20261115)`, 2000회, 현장 10곳): B* 2.5–97.5 % 0.0652–0.1587, Δ_B* 2.5–97.5 % -0.0544–0.0001

- 목표 경계(±9)는 짝을 독립으로 본 이항 1 SE이고 현장 군집을 반영하지 않는다. 현장이 10곳뿐이라 현장 부트스트랩 구간은 실제 불확도보다 좁게 나올 수 있다.

## 후보 수

- n_boxes: 795
- n_candidates_before_dedupe: 795
- n_candidates: 753
- n_boxes_without_candidate: {}
- n_candidates_tau_blank: 0
- R2 J1 파일(fold 3): {'n_boxes_post': 795, 'n_boxes_pre': 797, 'n_padding_only_dropped': 0, 'n_padding_only_dropped_pre': 0}
- 깊이 계산 후보 진단: {'status': {'ok': 733, 'fit_failed': 20}, 'qc_verdict': {'unknown': 403, 'in': 231, 'out': 119}, 'apex_at_record_end': {'False': 752, 'True': 1}, 'apex_on_trace_seam': {'False': 722, 'True': 31}, 'qc_excluded': {'True': 524, 'False': 229}}

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
| 2.post_sha_r2_summary | 통과 |  |
| 4.fold | 통과 |  |
| 4.no_removed_rows | 통과 |  |
| 4.class_pipe_hyperbola | 통과 |  |
| 4.line_id_split | 통과 |  |
| 4.lines_in_754 | 통과 | 0 rows |
| 5.candidates_sha_unchanged | 통과 |  |
| 1-1.3.one_main_row | 통과 | 1 |
| 1-1.3.n_gt_388 | 통과 |  |
| 1-1.3.n_pass_round_pass_b | 통과 |  |
| 1-1.3.match_rows_n_pairs | 통과 |  |
| 1-1.3.summary_n_pass | 통과 |  |
| 1-1.3.k_integer | 통과 |  |
