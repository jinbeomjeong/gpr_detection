# M3-3 J1 보고 실행 `report:fold2` (판정 아님)

- 생성: `analysis/processing/m33_j1_judge.py`, 2026-11-16T14:58:24+09:00, HEAD 1a73ba84a6ce80a75171e17f8b7040e2245f26c3, 357.7 s
- 박스 입력: `runs/eval/m33/r2/fold2/j1_boxes_post.csv` (sha256 da91b122f4e489cceeeacc5743f9ad03417b32cbe700c389c89f226fdd6d5b98)
- 후보 csv: `docs/processing/depth_4tu_j1_f2_candidates_input.csv` (sha256 0f9728c1e39af218d81df58fda20b94cfe043b927326061b9d69c31fa0ab4706)
- 깊이 계산: `scripts/eval_depth_4tu.py` (무수정), prefix `depth_4tu_j1_f2`
- 상태: `ok`

## 함께 적는 값 (판정에 쓰지 않음)

| 항목 | 값 |
|---|---|
| 합격 짝 n_pass / 388 | 47 / 388 |
| 판정률 B | 0.1211 |
| k (기준선 시행 1,000회 중 관측 이상) | 1000 |
| p = k / 1000 | 1.0000 |
| 짝 수 n_pairs | 57 |
| 판정률 A | 0.8246 |
| B_max = n_pairs / 388 | 0.1469 |
| 기준선 판정률 A 평균 | 0.9551 |
| MAE (m) | 0.1731 |
| bias (m) | -0.1319 |
| 누락 / 오검출 | 331 / 6 |

## 실질 지표 (보고, 판정 아님)

| 항목 | 값 |
|---|---|
| B | 0.1211 |
| 기준선 B 평균 | 0.1403 |
| 기준선 B 2.5 % | 0.1340 |
| 기준선 B 97.5 % | 0.1469 |
| Δ_B = B − 기준선 평균 | -0.0192 |

## 보고 항목 (J1-2 보고 계산, 판정 아님)

### p 도달 가능성

| 항목 | 값 |
|---|---|
| k_perfect | 30 |
| 관측이 완벽해도 p < 0.05 불가 | False |
| n_req = T_(50) + 1 | 57 |
| A_req = n_req / n_pairs | 1.0000 |
| 초과 추정 E_+ (= summary false_est) | 6 |
| n_e,a > n_g,a 활동 수 | 3 |
| 정답 1개이면서 추정 있는 활동 수 | 5 |

### 현장 단위 불확도

| 빠진 현장 | n_pass | N_g | B_-s | 기준선 평균 | Δ_B,-s | k_-s |
|---|---|---|---|---|---|---|
| 1 | 37 | 338 | 0.1095 | 0.1316 | -0.0222 | 1000 |
| 3 | 43 | 342 | 0.1257 | 0.1446 | -0.0188 | 1000 |
| 4 | 47 | 364 | 0.1291 | 0.1496 | -0.0204 | 1000 |
| 5 | 38 | 362 | 0.1050 | 0.1229 | -0.0179 | 1000 |
| 6 | 42 | 347 | 0.1210 | 0.1425 | -0.0214 | 1000 |
| 8 | 46 | 368 | 0.1250 | 0.1452 | -0.0202 | 1000 |
| 10 | 38 | 302 | 0.1258 | 0.1369 | -0.0111 | 1000 |
| 11 | 41 | 320 | 0.1281 | 0.1561 | -0.0280 | 1000 |
| 12 | 45 | 369 | 0.1220 | 0.1341 | -0.0122 | 1000 |
| 13 | 46 | 380 | 0.1211 | 0.1406 | -0.0196 | 1000 |

- 현장 부트스트랩(`default_rng(20261115)`, 2000회, 현장 10곳): B* 2.5–97.5 % 0.0808–0.1733, Δ_B* 2.5–97.5 % -0.0449–0.0039

- 목표 경계(±9)는 짝을 독립으로 본 이항 1 SE이고 현장 군집을 반영하지 않는다. 현장이 10곳뿐이라 현장 부트스트랩 구간은 실제 불확도보다 좁게 나올 수 있다.

## 후보 수

- n_boxes: 692
- n_candidates_before_dedupe: 692
- n_candidates: 672
- n_boxes_without_candidate: {}
- n_candidates_tau_blank: 0
- R2 J1 파일(fold 2): {'n_boxes_post': 692, 'n_boxes_pre': 692, 'n_padding_only_dropped': 0, 'n_padding_only_dropped_pre': 0}
- 깊이 계산 후보 진단: {'status': {'ok': 647, 'fit_failed': 25}, 'qc_verdict': {'unknown': 313, 'in': 257, 'out': 102}, 'apex_at_record_end': {'False': 672}, 'apex_on_trace_seam': {'False': 644, 'True': 28}, 'qc_excluded': {'True': 417, 'False': 255}}

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
