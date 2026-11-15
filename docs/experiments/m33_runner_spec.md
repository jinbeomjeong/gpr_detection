# M3-3 X4 — 평가 러너 명세 (R1 모로코, R2 4TU, R3 의사 띠, J1 입력)

- 상태: **v0.3.1**(2026-11-15, cv-detection-expert). v0.2에서 R3를 [EMR] 부록 D에 맞췄고, v0.3에서 X6·사용자 결정을, v0.3.1에서 무결성 점검과 [EMR] 부록 D-6(v1.6)을 반영했다. [PLAN] [`m33_cross_domain_plan.md`](m33_cross_domain_plan.md) v0.2.4(S0 9b28694는 v0.2.2)와 [PP] [`m33_pp_spec.md`](m33_pp_spec.md) v0.2.1과 [EMR] 부록 D v1.6을 따른다. 구현은 python-expert가 한다.
- 이 러너들은 C1 봉인 뒤에만 정식 실행한다. 봉인 순서는 [PLAN] §4 표다(… → C1 → R1 → R2 → R3 plan → **C1b** → R3 injected → **C2** → J1). C1 전에는 `--smoke`만 허용하고, smoke는 별도 폴더에 쓰며 판정에 쓰지 않는다.

## 0. 재사용 판단

| 대상 | 재사용 | 판단 |
|---|---|---|
| R2 4TU 미주입 | **`analysis/detection/m32_recipe_eval.py` `uninjected` 단계를 확장해 재사용** | 같은 검출 경로가 이미 있다(r0 L1, batch 64, 병합, M4a–d, `apex_hits`, `conf_counts`, `flip_pairs`, batch 1 정체 게이트). 바꿀 것은 §2-1의 확장 항목이다 |
| R3 의사 띠 | **`plan`·`injected` 단계를 확장해 재사용** | 배치·제외 구역(c′)·실현·주입 주변 타일 추론·이어 돌리기가 그대로 쓰인다. 바꿀 것은 주입체(S4 `make_pseudo_band`)와 일치 규칙(가로 구간)이다 |
| J1 입력 | R2 실행의 박스 내보내기로 만든다 | 추론을 따로 하지 않는다 |
| R1 모로코 | **새 러너 `analysis/detection/m33_morocco_eval.py`** | S5는 측선 npy 경로다. 모로코는 스케일 없는 정사각 JPG라 맞지 않는다. 대신 S3(`m32_aihub_perturb.py`)의 predict 경로(`gpr_detection.detect.yolo`, conf 0.001, NMS 0.7, 한 장씩)와 `m31_d1_common.ScoreAccumulator`(Ultralytics 8.4.192와 같은 AP)를 재사용한다. validator 경로(`scripts/eval_detector.py`)는 PP를 걸 수 없고 S3와 경로가 달라 쓰지 않는다 |

- 공통 모듈: `m32_recipe_inputs`(sha 단언·목록), `m32_recipe_match`(M4·일치), `m31_d1_common`(채점).
- 새 상수·목록은 `analysis/detection/m33_inputs.py`에 둔다. M3-2 상수는 바꾸지 않는다.

## 1. 공통 요구 (R1–R3·J1 내보내기)

| 항목 | 요구 |
|---|---|
| 검출기 | N2-a′ 5 fold `last.pt`. sha 5개를 단언한다([BL] §1). 앙상블 없음, fold별 독립 |
| 설정 | conf 0.25(운용), AP용 0.001, NMS 0.7, `MergeConfig()`, r0 `DetectionImageConfig`. **설정 해시**(conf·병합·r0 `assumed_params`·`assumed_values`)를 C1 값과 단언한다 |
| r0 정체 | [SP] §9 11측선 `line_image` sha256을 첫 단계에서 다시 만들어 단언한다 |
| PP | 같은 병합 박스에 **PP 꺼짐(S0)과 C0b 선택 집합**을 둘 다 적용해 두 행으로 저장한다(`pp` 열 `pre`/`post`). `PPConfig` 해시 = C0b를 단언한다. 추론은 1회다 |
| 입력 sha 단언(C1 표) | 가중치 5 · r0 11측선 · 설정 해시 · C0b(`PPConfig`, `d0_gate.json`) · 모로코 매니페스트·`eval_images.txt` · 4TU 754측선 목록·`keep_one` 표(720) · 중복 목록 `fourtu_duplicate_radargrams.csv` · K 꼭짓점(`known_apexes.csv`)·K 측선(`d3/lines.csv`) · IDS 목록(`m32_stageA_lines_ids.csv`) · `radargrams_index.csv` · S4 코드 blob · R3 주입 표(C1b). 하나라도 다르면 종료 2 |
| 깨끗한 트리 | 정식 실행은 추적 파일 변경 0, 필수 입력 추적을 요구한다(`dirty` false). `--allow-dirty`는 smoke 전용 |
| 블라인드 | 지표(재현율·AP·포함률·M4 수·판정률)는 파일에만 쓴다. stdout에는 단계, 측선·타일·영상·주입 건수, 경과 시간, 경로, sha만 찍는다. 마지막 줄은 `{"status","out_dir","n_lines","n_tiles","n_images","n_injections","elapsed_s"}`(해당 키만)이다. Ultralytics 로거는 끈다 |
| 종료 코드 | 0 완료, 1 게이트 실패, 2 입력 거부, 3 내부 오류 |
| 정체성 기록 | `run_identity`: git HEAD·`dirty`·status, 코드 blob(러너·inputs·match·common·`pipeline.py`·`to_image.py`·`inject.py`·`yolo.py`), 패키지 버전, GPU, argv, 위 sha 전부, 배치 크기·타일 순서 규칙 |
| 이어 돌리기 | (단계, 측선, fold) 부분 파일 + 설정 sha 검사(M3-2 S5 방식) |
| 자원 | GPU 단일 프로세스, batch 64. CPU 전처리는 허용 코어 affinity·BelowNormal·BLAS 1스레드. 30분 넘는 실행은 메인 세션 러너·예약 |
| AGPL | `ultralytics`는 `gpr_detection.detect` 안에서만 쓴다 |

## 2. R2 — 4TU 검출 보고 (`m32_recipe_eval.py --stage uninjected` 확장)

### 2-1. 확장 항목
1. **측선 목록 인자** `--lines <csv>`: 4TU 754측선(`lines_unique`, 활동 안 사본 5 제외, curator X7 확정 목록). 열은 `line_id, site, dataset, keep_one(bool), role(k/background)`이다. 지금의 단계 A 150 + K 27 고정 목록 대신 쓴다.
2. **레시피 고정** `--recipes r0`: M3-2 후보 r5·r6은 쓰지 않는다.
3. **PP 전·후 행**(§1): `boxes.csv`·`apex_hits.csv`·`m4` 집계에 `pp` 열을 더한다.
4. **집계 키**: 모든 행에 `keep_one`을 싣는다. 요약은 **720(`keep_one` true)이 주**이고 754는 민감도다.
5. **SNR 대리값 층**: X6(gpr-processing)이 주는 K 꼭짓점별 국소 SNR 4분위 표(`m33_k_apex_snr.csv`, 검출 무관)를 읽어 `apex_hits`에 붙인다.
6. **01.1형 폭**: 놓친 K 꼭짓점에 X6 `m33_k_apex_snr.csv`의 `width_vis_P`(보이는 범위 폭, 사용자 결정 2026-11-15)와 끝 닿음 플래그를 붙인다. 비교 기준은 AIHub p5–p95 7.79–14.67 P이고, 출처는 `m32_s1_recheck.py` `aihub_box_p()`(확인됨)다. `width_m6_P`·`width_m20_P`는 보고 열로 함께 싣는다.
7. **2사건 측선 표시**: `zero.n_events ≥ 2`인 측선을 `two_event` 열로 표시한다.
8. **J1 내보내기**(§5).
9. 정체 게이트: M3-2와 같은 r0 × K 27측선 × 5 fold batch 1 → D3 값(90 / 30 / 15). PP 전 박스로 본다. batch 64 대 1 경로 차는 보고만 한다.

### 2-2. 출력
- `runs/eval/m33/r2/`
  - `fold<k>/{boxes,apex_hits,conf_counts,flip_pairs}.csv`(`pp` 열 포함)
  - `lines.csv`(측선별 플래그·`keep_one`·`two_event`·$y_\text{bot}$·$y_z$)
  - `gate.json`
  - `summary.json`: 720 주 집계와 754 민감도로 K 포함률(5 fold 합/205, 아무 fold/41, 활동별), M4a–d·M4c·D3 8 px 값과 M4 측선 수, 클래스별 박스 수·점수 분위, $n_\text{agree}$, 반전 일관성, SNR 층, 2사건 층을 PP 전·후로 낸다. 함께 **PP 단계별 처리 수**(`removed_a`·`clipped_b`·`removed_b`·`removed_d`·`removed_c`)와 **PP-a 진단**(PP-a로 제거된 박스 중 위 끝이 내용 안, 즉 $y_1 < y_\text{bot} - P_ell$인 수, [PP] §4-4)을 fold별·5 fold 합으로 낸다. 키는 `pp_actions`, `pp_a_top_in_content`다.
- 지표 정의는 M3-2와 같다(M4는 `m4_category`, K 포함은 ±3 트레이스·±0.5 $P_\text{line}$).

## 3. R3 — 의사 띠 (`plan`·`injected` 확장, `--injectant band`)

물리와 주입 구현은 [EMR] 부록 D(v1.6, D-6 포함)를 따른다. 띠 모드 구현(D-5)은 python-expert가 `inject.py`에 한다. 이 절은 러너와 지표 쪽 요구만 적는다.

### 3-1. 확장 항목
1. **배경**
   - 4TU: 단계 A 150 + K 27측선. 모델은 A1 주, A0 민감도이고 `model_group`으로 나눈다. 보고 P6이다.
   - IDS: 단계 A 60측선, A0만, 30 ns 창은 상단 ≤ 1.0 m 모델만. 채택 근거 P4다.
   - **IDS 주파수 층**([EMR] 부록 D-6, v1.6). IDS 60측선은 주입 56(주 53 + 보고 3)과 주입 제외 4로 나눈다. 측선 표에 `antenna_MHz`와 `ids_freq_stratum`을 싣는다.

     | `ids_freq_stratum` | 측선 | 주입 | 쓰임 |
     |---|---|---|---|
     | `main_400` | 400 MHz 53 | 400 MHz H 맞춤(기존) | **P4 주 값** |
     | `report_other` | 200·600·900 MHz 각 1(3) | 측선마다 $P_\text{line}$으로 H를 맞춘다. 900 MHz 측선은 500 MHz 모델(M11·M14·M16)을 먼저 쓴다 | 보고 전용(주파수마다 1측선이라 채택 근거에 섞지 않음) |
     | `excluded_2ghz` | 2000 MHz 4 | **주입하지 않는다.** 합성 400 MHz 응답으로 흉내 낼 수 없다 | 미주입 추론과 구역 (c′)에만 쓴다 |
   - 2 GHz용 새 합성은 이번 범위 밖이다.
   - 셀은 `4tu`·`gz_ids`로 나눈다.
2. **미주입 IDS**: R3 안에서 r0 × 5 fold로 IDS 60측선 미주입 추론을 한다. 4TU는 R2 출력을 쓴다. 두 배경 모두 **띠 클래스를 포함한 모든 클래스** 미주입 박스를 구역 (c′)에 넣는다.
3. **주입체**: S4 새 띠 모드(`mode = "band"`, [EMR] D-5 `build_band_patch`)를 쓴다. `make_pseudo_band`는 쓰지 않는다. `_bg`는 배경에 더하지 않는다.
4. **배치**
   - 길이 표본: {40, 64, 100, 150, 200, 300} 트레이스. 4TU는 $L \le N - 2r_F$로 제한한다.
   - 제외 구역: (a)·(b)·(b′)·(c′)·(d)·바닥을 띠 가로 구간 전체에 적용한다([EMR] D-2).
   - 간격: $|c_1-c_2| \ge (L_1+L_2)/2 + W_\text{eff}$, $W_\text{eff}=\min(301,N)$. 쌍곡선과 띠는 같은 실현에 섞지 않는다.
   - SNR 0/6/12/18 dB, 극성 ±1 짝(같은 위치·길이), 시드 20261115.
   - **층 최소 수**: 배경마다 $L/W_\text{eff}$ 층마다 위치 ≥ 30을 목표로 한다. 모자라면 수를 공개하고 멈추지 않는다.
   - plan 보고에는 층별 위치 수를 낸다. P4 두 층(< 0.33, 0.33–0.5)의 IDS 위치 수도 따로 낸다.
5. **정답 표 열**(위치 표·`inj_matches`): `mode`, `band_length_traces`, `band_center_trace`, `length_px`, `w_eff`, `l_over_w_eff`, `lw_stratum`(`lt033`·`033_05`·`05_067`·`ge067`), `px_stratum`, `l_over_n`, `taper_traces`, 정답 박스(L0), `gain`, `noise_rms`, `signal_amplitude`, 잘린 칸 수·최대 주입 비율, 극성, SNR, $k$, `response_id`, `model_group`.
6. **일치**(PP 전·후 각각)
   - **가로 구간 재현율**: conf > 0.25 `pipe_band` 박스 가운데 정답과 시간 겹침 ≥ 정답 높이 50 %인 박스의 가로 합집합을 본다. 그것이 정답 구간 [i0, i1] 안에서 덮는 길이 / 정답 길이이고, 길이 가중이다. 정답 구간 밖 부분은 재현율에 넣지 않는다.
   - **유령 띠 오검출**: 다음 세 조건을 모두 만족하는 박스는 주입이 만든 오검출로 센다.
     - 같은 실현의 `pipe_band` 박스로, 정답과 시간 겹침 ≥ 50 %다.
     - 가로로 [i0 − 테이퍼, i1 + 테이퍼] 밖에 있다.
     - 같은 측선·fold 미주입 박스(같은 클래스)와 IoU ≥ 0.5인 짝이 없다.
     - 정답 구간과 걸친 박스는 구간 밖 부분 길이를 유령 길이로 센다.
     - 출력: 실현별 `ghost_n_boxes`, `ghost_length_traces`.
   - 클래스 무시 덮임(다른 클래스로 읽은 비율)은 보고만 한다.
7. **생존 진폭 비**: [EMR] D-5 2의 식 $s$를 r0 (4) 단면에서 잰다. 검출기와 무관하므로 fold와도 무관하고, 실현마다 1회 잰다.
8. **주입 표 고정**: plan 단계 뒤 주입 표 sha를 **C1b** 커밋에 적고, 그 뒤 injected를 실행한다. 규칙 변경은 없다(M3-2 F2 방식).

### 3-2. 출력
- `runs/eval/m33/r3/{plan,uninjected_ids,injected}/`
- `summary.json`
  - 배경(`4tu`·`gz_ids`) × 모델 묶음 × `lw_stratum` × SNR × 극성마다: 가로 구간 재현율과 유령 띠 오검출(수·길이). 모두 PP 전·후, 5 fold 평균 ± SD다.
  - 생존 진폭 비: SNR × `lw_stratum` × 배경.
  - px 층 보고.
- **P4 값**(`p4_adoption_input`, 키 이름 고정)
  - `recall_delta`: `gz_ids`, **`ids_freq_stratum` = `main_400`(53측선)**, A0, `lw_stratum` ∈ {`lt033`, `033_05`}. 층 같은 가중, 층 안에서 SNR 4 × 극성 2 균등, PP 후 − PP 전, 5 fold 평균. 판정 기준은 ≥ −1.0 %p다.
  - `ghost_delta`: 같은 범위의 유령 띠 박스 수 5 fold 합, PP 후 − PP 전. 기준은 ≤ 0이다.
  - `n_positions_by_stratum`: 두 층의 위치 수.
  - `n_lines`: 53(`main_400`)을 단언한다.
- 긴 두 층(`05_067`, `ge067`), 4TU 배경, IDS `report_other` 3측선은 `report_only`에 둔다. `summary.json`에 IDS 측선 구분 `{"main_400": 53, "report_other": 3, "excluded_2ghz": 4}`를 싣는다.

## 4. R1 — 모로코 (`m33_morocco_eval.py`, 새로 작성)

| 항목 | 요구 |
|---|---|
| 입력 | `data/morocco_eval_v1/eval_images.txt`(233), 라벨, 매니페스트 `data/splits/morocco_eval_v1.csv`. sha와 사본 0·그룹당 1장을 다시 단언한다 |
| 경로 | 주 경로는 letterbox 256(224 → 256 리사이즈, 1채널)이다. S3와 같은 predict 경로를 쓰고 fold 0–4 모두 돈다 |
| 클래스 | 정답 클래스 0(`pipe_hyperbola`)·2(`cavity`). 예측 4클래스를 그대로 쓴다. 하위셋(`utilities`·`cavities`·`intact`)은 매니페스트 열로 나눈다 |
| 지표 | 관로 + 이상 없음 하위셋 `pipe_hyperbola` AP50·AP50-95, conf 0.25 재현율·정밀도. 이상 없음 장당 오검출(클래스별). 공동 하위셋 `cavity` AP50·재현율(도메인 분리, 전체 지표에 넣지 않음). 영상 부트스트랩 95 % CI(B 2000, 시드 20261115). 혼동 행렬(배경 포함) |
| 민감도(보고) | (s1) 리사이즈 없이 중앙 패딩만. (s2) 축척 0.75·1.0·1.33배(주 경로 대비 등방). (s3) 상단 흰 세로선 영역 가림: X6에 행 범위가 없어 cv가 고정한다. 원 영상 행 0–27(위 1/8)을 128로 채운다. (s4) 근접 중복 쌍 U_233·U_239를 하나로 센다. (s5) **주기 맞춤 축소**: 영상별 $P_\text{native}$(X6 `m33_x6_period_snr.json`)로 $P_\text{in}$ = **$P_A$ 4.06 px**가 되게 등방 축소(INTER_AREA)한다. 관로 기준 배율은 중앙 0.16배(0.08–0.19)다. $P_A$를 기준으로 삼은 것은 AIHub 학습 분포에 맞추려는 취지이기 때문이다. $P^*$ 6 기준(0.23배)은 쓰지 않는다([PLAN] §2-1). 256 캔버스 가운데에 두고 나머지는 128로 채우며, 정답 박스도 같은 변환을 한다. P5에는 쓰지 않는다 |
| PP | 정사각 입력이라 채움이 없고($y_\text{bot}$ = 256) $y_z$도 없다. 그래서 PP-a·b·d는 작용하지 않는다. PP-c는 $y_\text{bot}$ = 256 기준으로 작용할 수 있다. PP 전·후를 모두 내고, **P5 값** = 관로 + 이상 없음 `pipe_hyperbola` AP50(PP 후 − PP 전, 5 fold 평균)이다. 키 이름은 `p5_adoption_input` |
| 1회 실행 | 정식 실행은 한 번만 한다. 출력 폴더가 있으면 거부한다(`--overwrite` 없음). smoke는 8장, 별도 폴더 |
| 출력 | `runs/eval/m33/r1/fold<k>/{predictions,objects}.csv`, `metrics.json`, `summary.json`, 사례 PNG(상·하위 12장, `runs/` 아래에만 두고 공개 저장소에 넣지 않음) |

## 5. J1 입력 생성 (R2 안)

- R2 실행이 fold마다 **`j1_boxes_post.csv`(PP 후, 판정 입력)**와 `j1_boxes_pre.csv`(PP 전, 보고)를 쓴다.
  - 행은 754측선 전체다(`keep_both`, [U3] 규칙).
  - 열은 `line_id, fold, class_id, score, i0, i1, k0, k1`(L0 격자), `x0_line_px, y1_px, y2_px`, `pp_action`, `tile_truncated`, `n_tiles_seen`이다.
  - `pipe_hyperbola`만 쓴다. `pipe_band`는 따로 `j1_band_boxes_{pre,post}.csv`로 쓴다(띠 깊이 규칙용이고 J1 판정에는 쓰지 않는다).
- 두 파일의 sha와 `PPConfig` 해시를 R2 `summary.json`의 `j1_inputs`에 적는다. 구조는 fold별 사전으로 고정한다.
  ```
  "j1_inputs": {
    "pp_set_id": "S<j>", "pp_config_sha256": "<…>",
    "weights_sha256": {"0": "<…>", …, "4": "<…>"},
    "n_lines": 754,
    "folds": {
      "0": {"post_path": "runs/eval/m33/r2/fold0/j1_boxes_post.csv", "post_sha256": "<…>",
            "pre_path": "…/fold0/j1_boxes_pre.csv", "pre_sha256": "<…>",
            "n_boxes_post": <int>, "n_boxes_pre": <int>},
      "1": {…}, …, "4": {…}
    }
  }
  ```
- **`line_id` ↔ `(activity_id, path)` 대응**: 박스 파일의 `line_id`는 `"<activity_id>/<path>"` 형식이다(예: `01.1/Path5`, `011.17/Profile_6409`). J1-1은 첫 `/`에서 나눠 `activity_id`·`path`를 만든다. 러너는 같은 규칙으로 만든 두 열 `activity_id`, `path`도 함께 싣는다.
- J1 판정 스크립트(X3, gpr-processing + python)는 이 sha를 단언하고, 판정에는 `post`만 쓴다.
- `j1_boxes_*.csv`에는 운용 conf 필터(점수 > 0.25, 병합 경로와 같은 필터)를 통과한 병합 박스만 싣는다. J1 쪽은 다시 거르지 않는다.
- `post` 파일에는 제거된 박스(`pp_action` = `removed_*`)를 싣지 않는다. `clipped_b`는 잘린 좌표로 싣는다.
- 파일 머리 또는 `summary.json`에 `fold`, `pp_set_id`, `PPConfig` 해시, 가중치 sha, 측선 수(754)를 적는다.
- 박스 → 꼭짓점 후보 변환은 J1 쪽(X3) 몫이다. R2는 박스만 낸다.
- cv 러너는 깊이표를 열지 않는다(허용 목록에 없음).

## 6. 실행 순서·시간 (GPU)

| 순서 | 실행 | 추정 |
|---|---|---|
| 1 | R1 모로코(주 + 민감도) | 약 3분 |
| 2 | R2 4TU 754측선 × 5 fold × 2방향 + batch 1 게이트 | 약 15–30분 |
| 3 | R3 plan(4TU + IDS) → C1b(주입 표 sha) → IDS 미주입 → injected | 약 20–30분 |
| 4 | J1 판정(CPU, X3) | gpr-processing 추정 |

- 실행 전 `status.py`를 확인한다. 메인 세션 러너(`make_runner.py --gpu-log --gpu-counters`)와 예약 `gpr_detection:m33`을 쓰고, 30분 넘으면 long-running-task-monitor로 감시한다.
- smoke(측선 3, fold 0)로 시간·VRAM을 먼저 잰다.

## 7. 시험 (python-expert)

1. PP 전 행은 M3-2 S5 r0 출력과 같다. 같은 측선 3개 fixture로 확인한다.
2. `keep_one` 집계: 720 측선 수와 중복 쌍 처리가 [U3] 규칙과 같다.
3. 띠 가로 구간 일치: 합집합 덮임, 시간 겹침 경계, 박스 없음이면 0. 정답 구간 밖 부분은 재현율에서 빠지고 유령 길이로 세어진다. 미주입 짝이 있는 박스는 유령에서 빠진다.
3a. `lw_stratum` 경계값: 0.33, 0.5, 0.67은 위쪽 층에 넣는다(구간이 [a, b) 형태).
4. 모로코 하위셋 분리와 공동이 전체 지표에서 빠지는지.
5. 블라인드: stdout 마지막 줄에 지표 키가 없다.
6. 입력 sha 하나를 바꾸면 종료 2.
7. 모로코 출력 폴더가 있으면 거부한다.

## 변경 이력

| 날짜 | 판 | 내용 |
|---|---|---|
| 2026-11-15 | v0.1 | 최초 작성(X4): 재사용 판단(S5 확장: R2·R3, S3 경로 새 러너: R1), 공통 sha·블라인드·PP 전·후, R2 754 추론/720 집계, R3 의사 띠(4TU 보고, IDS P4), R1 민감도·1회 실행·P5, J1 박스 내보내기 |
| 2026-11-15 | v0.2 | [EMR] 부록 D(X5)에 맞춰 R3를 고쳤다. 주입체는 S4 새 띠 모드로 바꿨다(`make_pseudo_band` 아님, `_bg` 미가산). 배경별 모델, (c′)에 미주입 띠 박스 포함, 간격 $W_\text{eff}$ 식, 길이 표본, $L/W_\text{eff}$ 층과 층 최소 수, 정답 표 열, 유령 띠 오검출 정의, 생존 비 [EMR] 식, P4 키(IDS·짧은 두 층·`ghost_delta`)를 정했다. 시험 3·3a를 추가했다 |
| 2026-11-15 | v0.3 | X6·사용자 결정 반영: R2 01.1형 폭을 `width_vis_P`로, R1 (s3) 행 범위를 0–27로 고정, (s5) 주기 맞춤 축소 추가(보고 전용), J1 박스 파일 규칙(conf 필터 통과분만, post에는 제거 박스 없음, 머리 정보) |
| 2026-11-15 | v0.3.1 | 무결성 점검 반영. 상위 판 번호를 계획 v0.2.4·PP v0.2로 고치고 봉인 순서(C1b·C2)를 적었다. R2 `summary.json`에 PP 처리 수와 PP-a 진단(`pp_a_top_in_content`)을 넣었다. `j1_inputs` fold별 구조와 `line_id` ↔ `(activity_id, path)` 대응을 정했다. IDS 비-400 MHz 7측선은 [EMR] D-6에 따라 `ids_freq_stratum`(주 400 MHz 53 / 보고 3 / 2 GHz 주입 제외 4)으로 나누고 P4를 `main_400`으로 한정했다. (s5)는 $P_A$ 기준이다 |
