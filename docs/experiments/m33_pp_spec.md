# M3-3 X1 — 경계 오검출 후처리(PP) 명세와 개발 게이트 D0

- 상태: **v0.2.4**(2026-11-15, cv-detection-expert). C0a(53c4e16)로 봉인한 판은 v0.2.3이다. v0.2.4는 C0b 뒤 D0 실행기 정밀도·판정 가능성 개정을 부록 A로 덧붙인 것이고, §1–§5 본문(게이트·문턱·선택)은 v0.2.3 그대로다. D0 재실행 때 적용한다. 판별 내용은 변경 이력을 본다. [X2] [`m33_x2_pp_physics_review.md`](../processing/m33_x2_pp_physics_review.md)(gpr-processing 물리 검토)를 반영했다. [PLAN] [`m33_cross_domain_plan.md`](m33_cross_domain_plan.md) v0.2.6 §3을 구체화한다(S0 9b28694는 v0.2.2, 현재 계획 v0.2.8).
- 이 문서는 **J1 판정 감사 대상**이다(QA (b)). 봉인 순서는 [PLAN] §4 표다: S0 → **C0a**(이 문서 §6 고정 항목, D0 전) → D0 → **C0b**(D0 결과·선택 집합) → C1 → …
- 물리 기준($t_z$, $y_\text{bot}$, $P_ell$)은 gpr-processing-expert가 X2로 검토했다. 그 의견을 반영해 v0.2로 올렸고, v0.2.1은 표기 정리다(규칙 불변).
- 구현은 python-expert가 한다.
  - 위치: `src/gpr_detection/pipeline.py`의 병합 뒤 단계(`pp_mode`, 기본 꺼짐 — C0a 당시. M3-3 마감 2026-11-26에 운용 기본값을 S3로 바꿈(`pipeline.DEFAULT_PP_SET`), 끄려면 `pp_mode=PPSetId.S0`. 꺼짐 비트 동일성 시험은 `tests/test_pp.py` test_1이 S0 명시로 유지, 기본값 = R2 명시 S3 비트 동일은 `tests/test_pp_default.py`), 상수는 `PPConfig`에 둔다.
  - 시험: `tests/test_pp.py`.
  - D0 실행기: `analysis/detection/m33_d0_gate.py`.

## 1. 좌표와 영역

- PP는 **병합된 측선 단위 박스**(`merge_line_boxes` 출력, L1 측선 영상 px, $y$ 아래로 증가)에 적용한다. 측선 연속 좌표에서 L0 격자로 바꾸기(`line_boxes_to_grid`) **전**이다.
- 영역 정의는 M3-2 `m32_recipe_match.ContentRange`·`m4_category`와 같다. 화소값은 쓰지 않는다.

| 기호 | 정의 | 출처 |
|---|---|---|
| $y_\text{bot}$ | 내용 아래 끝(이 행부터 채움) | `ImageGridMap` 내용 범위 |
| $P_ell$ | 측선 L1 주기(px). 압축 측선은 $P^*/c$ | `DetectionImage` |
| $y_z$ | 영점 행 = `ZeroResult.k_zero`(분수 샘플) / `samples_per_px`. `k_zero`가 없으면(영점 실패) 없음 | `select_zero` 기본 규칙, `ZeroConfig()` 기본값. src tree id를 C0a에 고정한다([X2] §1-3·§3 1) |
| $e$ | `EDGE_PX` = 2.0 px | M3-2 `m32_recipe_match` |
| $w_c$ | `M4C_WINDOW_PX` = 18.0 px(= 3 $P^*$, 고정 px) | 같음 |

## 2. 규칙 정의

박스 $B=(x_1,y_1,x_2,y_2)$, 중심 $y_m=(y_1+y_2)/2$, 클래스 $k$, 점수 $s$.

- 범주는 **원래 박스**로 한 번만 정한다. 순서는 a > b > d이고, `m4_category`와 같은 정의다. 구현은 src(`pipeline.pp_category`)에 두고 `m32_recipe_match.m4_category`와 같은지를 §5 시험 2로 확인한다(§4-7 5).
- c는 범주가 없는 박스에만 본다.

| 규칙 | 조건(원래 박스) | 동작 |
|---|---|---|
| **PP-a** | $y_m > y_\text{bot}$ (모든 클래스) | 제거 |
| **PP-b** | a가 아니고 $y_1 < y_\text{bot}-e$ 그리고 $y_2 > y_\text{bot}+e$ (모든 클래스) | $y_2 \leftarrow y_\text{bot}$로 자른다. 자른 높이 $y_\text{bot}-y_1 < P_ell$이면 제거한다 |
| **PP-d** | a·b가 아니고 $y_z$가 있으며 $y_m < y_z + 0.5\,P_ell$ (모든 클래스) | 제거. 영점 실패 측선에는 적용하지 않는다 |
| **PP-c($s_c$)** | a·b·d가 아니고 $k$ = `pipe_band`, $y_2 \ge y_\text{bot} - w_c$, $s < s_c$ | 제거 |

- 자르기(b)는 $y_2$만 바꾼다. $x$, 점수, 클래스, `n_tiles_seen`, 플래그는 유지한다. 출력 박스에는 `pp_action` 필드(`none`·`removed_a`·`clipped_b`·`removed_b`·`removed_d`·`removed_c`)를 단다.
- PP는 병합 결과에만 작용한다. 병합(R2·V1·V2·띠 중재)은 다시 돌리지 않는다.
- 결정성: 입력 박스 순서와 무관하다(박스별 독립 규칙).
- `pp_mode` 기본값은 꺼짐이고(C0a 당시; M3-3 마감 2026-11-26부터 기본 S3), 꺼짐 출력은 지금과 비트 단위로 같아야 한다(시험; 마감 뒤에는 `pp_mode=PPSetId.S0` 명시로 유지, `tests/test_pp.py` test_1).

## 3. 포함 순서열과 선택 규칙

$$S_1=\{a\}\subset S_2=\{a,d\}\subset S_3=\{a,b,d\}\subset S_4=\{a,b,d,c(0.35)\}\subset S_5=\{a,b,d,c(0.50)\}$$

- PP-c는 "점수 < $s_c$이면 제거"이므로 $s_c$가 높을수록 더 많이 지운다. $c(0.35)$가 지우는 박스는 $c(0.50)$가 지우는 박스에 모두 들어 있다.
- **선택**: $S_1$부터 차례로 게이트(§4-4)를 본다. 통과하는 가장 큰 $S_j$를 고르고, 처음 실패한 집합의 뒤는 보지 않는다.
- $S_1$이 실패하면 멈추고 사용자에게 묻는다. 실패 사유는 둘로 나눠 기록한다.
  - G1–G3 중 하나라도 실패: **구현 결함 신호**(상태 `s1_defect`). G4도 함께 0이어도 이쪽으로 친다.
  - G1–G3는 통과하고 G4 = 0으로만 실패: **"자료에서 작동하지 않음"**(상태 `s1_no_effect`). G1–G3 실패와 구분해 보고한다.
  - 두 경우 모두 종료 코드 1이다. 구분은 `d0_gate.json`의 `selection.stop_reason`(`s1_defect` / `s1_no_effect`)과 stdout 마지막 줄 `status` 값으로 한다(§4-5).
- 선택 결과는 `pp_set_id`(S1–S5)와 `PPConfig` 해시로 C0b에 봉인한다.

## 4. 개발 게이트 D0

### 4-1. 쓰는 자료 (입력 허용 목록 — 실행기가 열려는 모든 경로를 이 목록과 대조해 단언한다)

| 자료 | 경로 | 비고 |
|---|---|---|
| N2-a′ 가중치 5개 | `runs/detect/yolo11m_aihubv1n2a2_20261114_fold<k>/weights/last.pt` | sha [BL] §1 |
| AIHub 검증 crop 목록 | `runs/detect/yolo11m_aihubv1n2a2_20261114_fold<k>/eval/last_crop/eval_images.txt` | fold별, 합 5,473. sha는 D1 r4 verdict와 같음 |
| AIHub 영상·라벨 | `data/aihub_yolo_v1/` | 라벨 v1 |
| 분할 매니페스트 | `data/splits/aihub_yolo_v1.csv` | 누수·test 0 단언 |
| 채움 높이 분포 | `analysis/processing/m32_content_height_lines.csv` | 실측 측선 **내용 높이 기하 통계**(검출 없음, M3-2 §1-5 공개). 공개 사항에 적는다 |
| gprMax 합성 | `data/synthetic_gprmax/M01–M20{.npy,_bg.npy,.json}` | 정답 꼭짓점은 json의 관 위치·깊이 |
| crop 목록 sha 기준 | `analysis/detection/m31_d1_r4_verdict.json` | 커밋된 M3-1 판정 문서. crop 목록 sha만 읽고 지표 값은 읽어 쓰지 않는다(v0.2.2 추가, §4-7 5) |
| 코드 | `src/gpr_detection/**`, `analysis/detection/m31_d1_common.py`, `analysis/detection/m32_aihub_perturb.py`(채움 흉내 함수 재사용), `analysis/detection/m33_d0_gate.py`, `analysis/data/m33_assert_no_aihub_test.py`(X7 test 0 단언, 자식 프로세스로 매니페스트와 crop 목록만 읽음) | git blob 기록. `m32_recipe_match.py`는 실행기가 import하지 않는다(시험에서만 비교 기준으로 씀) |

### 4-2. 쓰지 않는 자료 (금지 목록 — 열면 종료 2)

- `C:\workspace\GPR_Data` 아래 4TU·광저우·GPR-SD·모로코 원자료 전부.
- `data/morocco_eval_v1/**`, `data/processing/d3/**`, `data/processing/**`의 4TU·IDS 산출물.
- `runs/eval/m32_recipe*/**`(M3-2 P5a·P5b 박스·주입), `runs/eval/m32_aihub_perturb/**` 결과 파일. 함수 재사용은 허용하지만 결과는 읽지 않는다.
- 4TU 깊이표(`utilities_groundtruth.csv` 등), `docs/processing/fourtu_*` 후보·측선 파일.
- AIHub test 세션 crop(`test.yaml`).

### 4-3. 조건

| ID | AIHub 조건(검증 fold, fold k 모델) | 내용 범위 | 영점 |
|---|---|---|---|
| A0 | 변형 없음 | $y_\text{bot}$ = 256 | $y_z$ = 0(crop 위 끝) |
| A1 | S3 `pad_p0`: crop마다 시드 20261114로 실측 높이 분포에서 뽑은 $h_\text{pad}$만큼 아래를 평탄 128로 채움 | $y_\text{bot}$ = 256 − $h_\text{pad}$ | $y_z$ = 0 |
| A2(보고) | A1 + 위쪽 평탄 128 띠 $h_\text{top}$ = 2 $P_A$(8 px)를 덧붙이고 같은 높이만큼 아래를 잘라 256 유지 | $y_\text{bot}$ = 256 − max($h_\text{pad}$ − $h_\text{top}$, 0)(아래 8 px를 잘라 낸 뒤 남은 채움) | $y_z$ = $h_\text{top}$ |
| **A3** | A0 crop 그대로(추가 추론 없음, A0 예측 재사용) | $y_\text{bot}$ = 256 | $y_z$ = **지표 띠 행**: 행 0–63(위 1/4) 안에서 행 에너지 $\mathrm{mean}_x(q-\bar q)^2$가 최대인 행. $\bar q$는 crop 전체 평균이다. 동률이면 작은 행을 고른다. 라벨·검출기와 무관한 결정적 규칙이다([X2] §2-2). $y_z$는 행 번호 $j$ 그대로(행 위 끝)다. AIHub는 항등 격자(`samples_per_px` 1)이므로 §1 정의 `k_zero` / `samples_per_px`에서 $k = j$와 같다 |

- $P_A$ = 4.06 px이다(AIHub crop 주기 상수, M3-2 §1-6). AIHub에서는 $P_ell$ 대신 $P_A$를 쓴다.
- 정답 처리
  - 중심이 채움 안(또는 A2에서 잘려 나간 곳)이면 무시 박스다(S3 규칙: 무시 박스와 IoU ≥ 0.5 또는 IoA ≥ 0.5인 예측은 버림).
  - A2의 무시 판정은 $+h_\text{top}$ 옮긴 뒤 **자르기 전** 박스 중심으로 한다. 순서는 (i) $y_1' = y_1 + 8$, $y_2' = y_2 + 8$(자르지 않음), (ii) $(y_1'+y_2')/2 \ge y_\text{bot}$이면 무시(IoU·IoA용 좌표는 $[y_1', \min(y_2', 256)]$), (iii) 남은 정답만 $y_2' \leftarrow \min(y_2', y_\text{bot})$이다. 256 자르기와 $y_\text{bot}$ 자르기는 남은 정답에만 적용한다. 따라서 $h_\text{pad} \ge 8$이면 A2 유지 집합은 A1과 같고 좌표만 +8이다(A1↔A2 불변성, 시험으로 확인).
  - 남은 정답은 PP 전·후 모두 $y_2 \leftarrow \min(y_2, y_\text{bot})$으로 **같이 자른다**. PP-b 자르기가 정답과의 IoU를 인위적으로 바꾸지 않게 하려는 것이다.
- A2는 [X2] §2-2의 지적대로 시험력이 약하다. AIHub 실제 지표 띠가 덧붙인 띠 아래에 있기 때문이다. 그래서 **게이트에서 빼고 보고**로 둔다. PP-d 게이트 근거는 A3와 Y0다.
- **합성 기록 끝 변형(Y1)**([X2] §2-1)
  - Y0와 같은 20모델의 L0 기록을 꼭짓점 샘플 + $\{0.5, 1, 2, 3\}\,P_\text{line}$에서 잘라 시간창을 줄인다. $P_\text{line}$은 그 측선의 r0 `period_samples`다.
  - 그 뒤 C′ 경로로 측선을 만든다. 주기는 Y0의 r0 `period_samples`로 고정하고(짧은 기록의 재측정은 최대 3.8배 어긋남), 영점·내용 범위·AGC·영상은 자른 기록에서 다시 만든다. 남기는 샘플은 0 … floor($k_\text{apex}$ + f·$P_\text{line}$)(포함)이고, $k_\text{apex}$는 json `apex.envelope_peak_ns`의 샘플 위치다. 꼭짓점 일치 여유 ±0.5 $P_\text{line}$도 고정 P를 쓴다. 정답 꼭짓점은 그대로다.
  - 주기 고정은 src `make_detection_image(..., period_samples_override: float | None = None)`로 한다. `None`이면 지금과 비트 단위로 같아야 한다(시험).
  - `synthetic_apex.csv`에 `period_samples`, `period_source`(`measured` / `Y0_fixed`), `p_over_p_y0`(재측정했을 때의 값, 보고), `zero_px`, `y_bot_px`, `(y_bot − apex)/P_ell` 열을 싣는다.
  - 지표는 꼭짓점 포함 손실(맞힌 수 차)을 자르는 위치별로 낸다. 함께 규칙별 처리 수를 낸다.
  - **보고 + 손실 수 공개**로 두고 게이트에는 넣지 않는다. 사유: 기록 끝에서 잘린 관은 날개가 없어 깊이를 낼 수 없으므로 J1 손실과 직결되지 않는다([X2] §2-1 권고).
- 합성(Y0): M01–M20 각각 C′ 경로(r0, $P^*$ 6)로 L1 측선 영상을 만들어 5 fold 추론·병합한다.
  - 짧은 측선이라 아래 채움이 자연스럽게 생긴다.
  - 영점은 `select_zero`다(공기결합 모델은 지표 반사 띠).
  - 정답은 json의 관 꼭짓점을 L0(트레이스, 샘플)로 바꾼 값이다. 일치는 M3-2 규칙(±3 트레이스, ±0.5 $P_\text{line}$, `pipe_hyperbola`, 일대일)이다.
- 추론은 조건마다 1회다. PP 집합 $S_0$(없음)–$S_5$는 같은 예측에 사후 적용한다.

### 4-4. 게이트 (집합 $S_j$마다, 같은 예측에서 $S_0$ 대비)

| ID | 지표 | 통과 조건 |
|---|---|---|
| G1 | 쌍곡선 conf > 0.25 재현율 **총손실** = $S_0$에서 맞고 $S_j$에서 놓친 정답 수 / 정답 수(짝지은 객체, IoU 0.5). $S_j$에서만 맞는 정답(이득)은 빼지 않는다. 5 fold 비가중 평균, **A0·A1·A3 중 가장 나쁜 조건** | ≤ **0.5 %p** |
| G2 | 띠 가로 구간 재현율 **총손실** = 띠 정답마다 덮인 길이의 감소분(증가분은 0으로 침)의 합 / 정답 길이 합. 덮임은 `band_metrics.py` 정의(conf > 0.25, 세로 겹침 ≥ 정답 높이 50 %, 정답 구간 안 가로 합집합)다. 5 fold 비가중 평균, A0·A1·A3 중 가장 나쁜 조건 | ≤ **1.0 %p** |
| G3 | 합성 Y0 꼭짓점 포함 손실(M01–M20 × 5 fold, 맞힌 수의 차). $H(S_0) \ge 1$일 때만 정의된다(§4-7 4) | = **0** |
| G4 | **새 규칙 작동 수**: $S_j$의 처리 수 − $S_{j-1}$의 처리 수($S_0$ = 0). 처리 = `removed_*` 또는 `clipped_b`, 집계 범위는 §4-7 2(A1·A3·Y0 합) | > **0** |

- 보고(게이트 아님): A2의 G1·G2 값, Y1의 자르는 위치별 꼭짓점 손실 수와 규칙별 처리 수, cavity·manhole 재현율 손실, mAP50(conf 0.001) 차, 규칙별 처리 수(a·b·d·c), gain 사본별 분해, fold별 값, A3의 $y_z$ 행 분포.
- **PP-a 진단**(D0와 R2·J1 공통, [X2] §1-1 (ii)): PP-a로 제거된 박스 가운데 위 끝이 내용 안($y_1 < y_\text{bot} - P_ell$)인 수다. "잘린 실관일 수 있는 제거"를 뜻하고, 보고만 한다. D0 출력에는 조건별로, R2 `summary.json`과 J1 보고(§4 5)에는 측선 합으로 낸다.
- G1·G2는 총손실이라 음수가 없다. 이득(S_j에서만 맞는 정답 수, 늘어난 덮임 길이)은 집합·조건·fold별로 보고한다. 총손실로 정한 근거: PP-b가 정답도 함께 잘라(§4-3 정답 처리) IoU를 올릴 수 있으므로, 순손실로 보면 이득이 손실을 가린다.
- G4는 이번에 더한 규칙이 D0 자료에서 실제로 작동하는지 보는 상식 점검이다. 증분이 0이면 그 집합은 실패다(앞 집합과 출력이 같아 새 규칙의 TP 손실 근거가 없다). 누적 처리 수와 규칙별 처리 수는 보고한다.

### 4-5. 출력 형식

- `runs/eval/m33_d0/`
  - `fold<k>/predictions.csv`: **AIHub crop 예측만**(A0–A3, 무시 박스 제거 뒤). 열: fold, 조건, crop, gain, 박스(crop px), 점수, 클래스, 범주, 집합별 `pp_action`.
  - `fold<k>/synthetic_boxes.csv`(v0.2.3 추가): **합성 Y0·Y1 병합 박스 전부**(conf 0.25 검출기, PP 전 원래 박스). 좌표계가 crop과 달라(L1 측선 px) 파일을 나눈다. 열: fold, 조건(`Y0`·`Y1_<f>P`), 모델, 박스(L1 px `x1,y1,x2,y2`), L0 박스(`i0,i1,k0,k1`, 격자에 닿지 않으면 빈칸), 점수, 클래스, 범주, `tile_truncated`, `n_tiles_seen`, 집합별 `pp_action`, 집합별 PP 후 `y2`(자른 경우). 재감사 때 `synthetic_apex.csv`의 맞춤 여부와 처리 수를 이 파일에서 다시 만들 수 있어야 한다.
  - `fold<k>/objects.csv`: 정답별 집합별 맞춤 여부.
  - `synthetic_apex.csv`: 모델·조건·fold별 꼭짓점 맞춤 여부(집합별), 처리 수(집합별·**규칙별**), §4-3 Y1 진단 열.
  - `d0_gate.json`: 집합별 G1–G4 값과 통과 여부(G1·G2는 총손실과 이득을 조건·fold별로, G3는 $H(S_0)$·$H(S_j)$·노출 수, G4는 증분·누적·규칙별 수 — 키 `increment`·`cumulative`·`by_condition`·`by_rule`; 게이트 밖 A0·A2·Y1 규칙별 수는 `report.rule_counts_outside_gate`), 선택 `pp_set_id`, 멈춤 사유 `selection.stop_reason`(`null`·`s1_defect`·`s1_no_effect`·`gate_failed`), `PPConfig` 해시, `run_identity`(git HEAD·`dirty` false, 코드 blob, 가중치 sha 5, crop 목록 sha 5, 매니페스트·채움 표·합성 파일 sha, 허용 목록 단언 결과, 열린 경로 목록).
- stdout에는 단계·건수·경과 시간·출력 경로만 찍는다. 마지막 줄은 `{"status","out_dir","n_conditions","n_images","elapsed_s"}`이고 값은 넣지 않는다.
- 마지막 줄 `status`: `ok`(선택 완료, $S_2$ 이후에서 멈춘 경우 포함), `s1_defect`, `s1_no_effect`, `input_error`, `error`.
- 종료 코드: 0 완료, 1 $S_1$ 실패(멈춤, `s1_defect`·`s1_no_effect` 둘 다), 2 입력 거부(금지 경로, sha 불일치, dirty), 3 내부 오류.

### 4-6. 자원

- GPU: 추론 3조건(A0, A1, A2; A3는 A0 예측 재사용) × 5,473 crop + 합성 20모델 × (Y0 1 + Y1 4) × 5 fold로 약 6–10분(S3 약 14 s/조건/1,000 crop).
- 단일 프로세스, batch 64(M3-2 S5와 같은 배치 경로)다.

### 4-7. 구현 해석 (v0.2.2 확정, v0.2.3 게이트 정의 개정 반영, C0a 전, PP 규칙·문턱 불변)

명세가 비어 있던 곳을 아래처럼 고정한다. 모두 결과를 보기 전에 정했다.

1. **무시 박스 제거 순서.** 조건별 무시 박스(§4-3 정답 처리, S3 규칙: IoU ≥ 0.5 또는 IoA ≥ 0.5)와 겹치는 예측은 PP **전에** 원래 예측에서 한 번만 버린다. $S_0$–$S_5$는 남은 같은 후보 예측에서 출발한다. 버린 예측은 PP 처리 수·G4·PP-a 진단에 넣지 않는다(채움 속 실물체를 "정의상 오검출 처리"로 세지 않는 보수적 선택). A3는 A0의 무시 박스(없음)를 쓴다.
2. **G4 집계 범위.**
   - AIHub A1·A3: **원래 점수 $s$ > 0.25**인 예측만 센다(conf 0.001까지 세면 G4가 늘 참이 되어 상식 점검이 되지 않는다). G1·G2·J1 박스와 같은 운용 문턱이다.
   - Y0: conf 0.25 측선 검출기의 병합 박스 전부(L0 격자에 닿지 않는 채움 속 박스 포함).
   - 처리 = `removed_*` 또는 `clipped_b`. conf 0.001 전체 처리 수와 규칙별 수는 보고(§4-4)에 따로 낸다.
   - 규칙별 처리 수(`by_rule`: `removed_a`·`clipped_b`·`removed_b`·`removed_d`·`removed_c`)는 **누적 처리 수와 같은 범위(A1·A3·Y0 합)**로 센다. 집합마다 `by_rule`의 합 = 누적 처리 수여야 한다(실행기가 단언). 조건별 분해(A1·A3·Y0 각각)도 함께 낸다. 게이트 밖 조건(A0·A2·Y1)의 규칙별 수는 같은 형식으로 보고에만 낸다(Y1은 자르는 위치별).
   - (v0.2.3) G4 값은 같은 범위에서 센 $S_j$ 처리 수 − $S_{j-1}$ 처리 수다. 범주가 배타적이고 집합이 포함 순서열이라 누적 처리 수는 $j$에 대해 줄지 않는다. 증분이 음수면 구현 결함이다.
3. **G1·G2 값이 없을 때.** 게이트 조건 A0·A1·A3 × fold 0–4의 15칸 가운데 **하나라도** 해당 클래스 정답이 없어 재현율(총손실의 분모)이 정의되지 않으면 그 게이트는 실패다. 남은 칸만으로 평균·최악을 내지 않는다("5 fold 평균, 세 조건 중 최악"의 정의를 지키기 위함). 정식 실행(5,473 crop)에서는 생기지 않을 것으로 보고, 생기면 공개한다.
4. **G3.** 맞힌 수 차 $H(S_0)-H(S_j)$가 정확히 0일 때만 통과다. PP는 박스를 지우거나 $y_2$를 줄이기만 하므로 음수는 구조상 나올 수 없다. 음수가 나오면 구현 결함으로 보고 실패로 친다. 또 G3는 $H(S_0) \ge 1$일 때만 정의되고, $H(S_0)=0$이면 실패다. $H(S_0)$와 노출 수($S_0$에서 맞힌 꼭짓점 박스 가운데 범주 a·b·d에 드는 수)를 함께 낸다.
5. **허용 목록 보강과 재구현.** §4-1 표에 `m31_d1_r4_verdict.json`(crop 목록 sha 기준)과 `m33_assert_no_aihub_test.py`(매니페스트 test 행 대비 파일·crop 키·세션 0 단언)를 넣었다. 둘 다 AIHub 검증 fold와 매니페스트 범위 안이고 M3-3 판정 대상 결과(4TU·모로코·IDS, `runs/eval/**`)를 읽지 않으므로 블라인드·누수 원칙과 맞는다. 판정 문서의 지표 값은 실행기 출력·게이트에 쓰지 않는다. `m32_recipe_match.py`는 허용 목록 밖이라 실행기가 import하지 않는다. M4 범주는 src `pp_category`, 단일 꼭짓점 일치(±3 트레이스, ±0.5 $P_\text{line}$, `pipe_hyperbola`, 일대일 — 꼭짓점이 하나이면 "포함하는 박스가 있음"과 같다)는 D0에 두고, 원본과 같은지를 시험으로 확인한다.
6. **추론 문턱·배치.** AIHub crop은 conf 0.001·NMS 0.7(S3와 같은 문턱, AP 보고용; conf > 0.25 재현율은 같은 예측을 거른 값)이다. Y0·Y1 측선은 M3-2 S5 측선 경로와 같은 conf 0.25 검출기를 fold마다 따로 쓴다(앙상블 없음). 둘 다 batch 64다. crop의 batch 64는 S3의 한 장씩 경로와 부동소수 차가 있을 수 있으나 $S_0$와 $S_j$가 같은 예측이라 게이트에는 영향이 없다.

## 5. 시험 (python-expert, `tests/test_pp.py`)

1. `pp_mode` 꺼짐은 비트 단위로 같다. 저장된 M3-2 결과 파일은 열지 않는다(블라인드). 다음 둘로 확인한다.
   - (1a) [SP] §9 r0 11측선 원자료 + 결정적 가짜 검출기로 `pp_mode` 기본(C0a 당시 꺼짐; 마감 뒤 기본은 S3라 비교에서 빼고 S0 명시만)·S0·PP 없는 경로(`detect_tiles` → `merge_line_boxes` → `line_boxes_to_grid`)의 출력이 같다(opt-in `-m realdata`). 실제 가중치를 쓰지 않는 것은 C1 전에 판정 도메인(4TU 등) 검출 결과를 만들지 않기 위함이다.
   - (1b) HEAD a0dc987(PP 도입 전) 코드와 작업 트리 코드가 **11측선 전부**에서 같은 가짜 검출기로 같은 병합 박스·L0 박스를 낸다. 비교 스크립트·출력 sha를 C0a 시험 통과 기록에 넣는다.
   - C0a 시험 통과 기록에는 `-m realdata`를 포함한 실행(건너뜀 0)을 적는다.
2. 범주 배타성: 한 박스는 a·b·d 중 하나이고, `m4_category`와 같다.
3. b 자르기: 자른 뒤 $y_2 = y_\text{bot}$이고, 높이 < $P_ell$이면 제거된다. 경계 $\pm e$ 안의 박스는 그대로다.
4. d: 영점 실패면 작용하지 않는다. 경계 $y_z + 0.5P_ell$ 바로 위·아래에서 동작한다.
5. c: 순서열 포함 관계(c(0.35)의 제거 집합 ⊂ c(0.50)의 제거 집합)가 성립하고, 띠 외 클래스에는 작용하지 않는다.
6. 박스 순서를 섞어도 결과가 같다.
7. D0 실행기가 금지 경로를 열면 종료 2로 거부한다.

## 6. 결과 전 고정 항목 (C0a로 커밋, D0 실행 전)

- 이 문서의 §1–§5 전체: 정의, 순서열, 선택 규칙, 조건 A0–A3·Y0·Y1, $h_\text{top}$ = 8 px, A3 지표 띠 행 규칙, Y1 자르는 위치 {0.5, 1, 2, 3} P와 보고 전용 구분, Y1 주기 고정(Y0 P), A2 무시 판정 순서(옮긴 뒤 자르기 전 중심), 시드 20261114, 게이트 지표 정의(G1·G2 총손실, G3 $H(S_0) \ge 1$, G4 증분)와 문턱, $S_1$ 멈춤 사유 구분(`s1_defect`·`s1_no_effect`), 허용·금지 목록, 출력 형식, PP-a 진단.
- $y_z$ 정의(`ZeroResult.k_zero` / `samples_per_px`), `select_zero`·`ZeroConfig()` 기본값, src tree id.
- `PPConfig` 상수 값($e$ 2.0, $w_c$ 18.0, d 여유 0.5 P, b 최소 높이 1 P, $s_c$ {0.35, 0.50}).
- D0 실행기와 `pp` 구현의 코드 blob, 시험 통과 기록.
- C0b(D0 뒤): `d0_gate.json` sha, 선택된 `pp_set_id`, `PPConfig` 해시. 이후 M3-3의 모든 실행은 이 해시를 단언한다.
- D0 뒤 규칙·문턱을 바꾸면 C0a를 새로 만들고 D0를 다시 돌린다. 앞 결과는 공개한다.

## 7. 공개 사항 (판정서에 옮김)

1. D0는 AIHub 검증 fold와 gprMax 합성만 썼다. 채움 높이 분포는 실측 측선 **기하 통계**(내용 높이, 검출 무관, M3-2 §1-5)에서 왔다.
2. AIHub에는 $t_z$가 없다.
   - A3는 위 1/4 안 행 에너지 최대 행을 지표 띠로 보고 영점으로 썼다. 그 행이 강한 얕은 반사이면 $y_z$가 아래로 가서 손실을 크게 본다(보수적 시험).
   - A2(덧붙인 평탄 띠)는 시험력이 약해 보고만 했다.
   - PP-d의 TP 손실 근거는 A3, Y0, [X2] §1-3 기하 논증이다.
   - **영점 바로 아래(< 0.3 m) 관은 합성에 없다.** 새 gprMax가 필요해 이번에는 넣지 않았다([X2] §2-1). 기하 논증과 J1 진단으로 대신한다.
2a. 기록 끝 근처 관(PP-a·b 위험)은 Y1로 쟀다. Y1은 보고 전용이고 손실 수를 공개한다.
3. PP-c 창 $w_c$는 M4c와 같은 고정 18 px다. 압축 측선($P_ell$ < 6)에서도 px 고정이다.
4. 규칙의 동기는 공개 집계값([D3] §8-1, [SC] R-3·R-4)이다. 문턱 값은 그 값들에 맞추지 않았다.
5. **게이트의 통계적 성질**(stats-design 검토, v0.2.3)
   - G1 0.5 %p는 사실상 개수 규칙이다. 5 fold 비가중 평균에서 503개 가운데 최대 3개 손실을 허용하고, 손실이 fold 0·1(쌍곡선이 적은 fold)에 끼면 1–2개로 줄어든다. fold별 쌍곡선 수는 71/45/98/147/142다([CVR] [`m32_cv_review.md`](m32_cv_review.md) §4-1 S3 입력 행).
   - G2에서 fold 3의 띠 1개는 0.32 %p다. 비가중 fold 평균의 분산은 합산 대비 G1 1.21배, G2 2.24배다.
   - G3: 합성 20모델에서 손실 0/20이어도 손실률의 95 % 상한은 약 16 %다. 영점 바로 아래 관이 합성에 없어(§7 2) PP-d에 대한 G3의 검정력은 0이다. 이 한계는 공개만 하고 게이트는 바꾸지 않는다.
   - 순차 선택(§3)은 집합이 포함 순서열이고 지표가 단조라 다중 비교 보정이 필요 없다. 다만 선택된 집합의 D0 값은 "통과해서 고른 값"이라 경계 근처에서 작게 치우칠 수 있다.
   - 경계 부호가 다르다. 정답 무시 판정은 중심 ≥ $y_\text{bot}$(§4-3)이고, PP-a와 `m4_category`는 $y_m > y_\text{bot}$(§2)이다. 중심이 정확히 $y_\text{bot}$인 예측만 차이가 나며, 그 예측은 무시 정답과 겹쳐 PP 전에 버려지는 경우가 대부분이라 영향은 거의 없다. PP-a를 M3-2 `m4_category`와 같게 두기 위해 통일하지 않고 공개한다.

## 부록 A. C0b 뒤 D0 실행기 개정 (v0.2.4, 게이트 규칙 불변)

- **성격**: C0b(0f11c34) 뒤, C1 전에 python-expert가 D0 실행기(`analysis/detection/m33_d0_gate.py`, 함께 쓰는 `m31_d1_common.py`)의 기록 정밀도와 판정 가능성 결함을 고쳤다(2026-11-15). §1–§4의 정의·조건·게이트 지표·문턱·선택 규칙과 `PPConfig`, PP 구현(`src/gpr_detection/pipeline.py`)은 바뀌지 않았다. §6이 말하는 "규칙·문턱 변경"이 아니므로 C0a를 새로 만들지 않는다. 바뀐 것은 §4-5 출력 형식(status·종료 코드·json 키)과 기록 방식뿐이다.
  - **§6 고정 목록의 해석**: §6의 "출력 형식"과 "D0 실행기와 `pp` 구현의 코드 blob", 그리고 **src tree id**(`164f0990…`)는 **C0a 시점 값**(53c4e16)을 가리킨다. src tree는 C1 1단계에서 `eval/inject.py`(띠 함수)만 바뀌어 달라졌고 `pipeline.py` blob은 그대로다(C1 실행 전 감사 기록 반영, [`m33_c1_seal.md`](m33_c1_seal.md) §1). C0b의 D0 결과는 그 값으로 만든 것이다. D0 재실행(D0′, 아래)에는 개정 실행기 blob과 이 부록의 개정 출력 형식을 쓴다. `pp` 구현 blob(`pipeline.py`)과 `PPConfig` 해시는 두 실행에서 같아야 한다(D0′가 단언).
- **적용 범위**: 이 부록은 **D0를 다시 돌릴 때부터** 적용한다. C0b에 봉인된 D0 결과(`runs/eval/m33_d0/`, C0a 53c4e16 실행기 출력)는 바꾸기 전 코드의 출력이다. 선택(S3)과 C0b 판정서는 고치지 않는다.
  - **"판정·선택 불변"은 아직 재실행으로 확인된 값이 아니다.** C0b 판정서 주 1의 불변은 4자리 기록값의 재집계와 실행기 값의 비교에서 나온 추론이다. 사용자 결정(2026-11-15)에 따라 C1 전에 D0′로 확인한다.
- **바뀐 것**(§4-5 보충)
  1. **전체 정밀도**: `predictions.csv`·`synthetic_boxes.csv`·`synthetic_apex.csv` 등의 점수·좌표, 합성 행의 `apex_sample`·`period_samples`·`zero_px`·`y_bot_px` 등을 반올림하지 않고 `repr(float)`로 쓴다. 이전에는 점수가 소수 4자리였고, 재집계에서 conf > 0.25 경계 박스 1개의 판정이 뒤집혔다(C0b 판정서 주 1. 판정·선택 불변은 추론이며 D0′로 확인한다, A-1).
  2. **판정 가능성**: 기대 fold는 `EXPECTED_FOLDS` = 0–4다. 중복이거나 범위 밖인 fold는 종료 2로 거부한다. smoke·일부 fold·crop 수 제한·일부 모델·시작 시 dirty 가운데 하나라도 해당하면 `status` = `not_judgeable`, 종료 1이다. `d0_gate.json`에 `judgment` 블록(`judgeable`, `blockers`, `expected_folds`, `folds_run`, `tree_stable`, `tree_changes`), `selection.judgeable`, `settings.expected_folds`를 넣는다.
  3. **끝 시점 트리 재측정**: git HEAD·`dirty`·코드 blob·src tree id를 시작할 때와 결과를 쓰기 직전에 잰다. 다르면 `input_error`, 종료 2다. `run_identity.tree_end`·`tree_problems`에 남긴다.
  4. **텍스트 입력 LF sha**: `run_identity.inputs_sha256_lf`에 텍스트 입력의 LF 정규화 sha를 원시 sha와 함께 적는다. 예: 채움 표는 원시 `65fb6bf4…`, LF `c11bbd37…`.
  5. `run_identity`에서 `g1_inputs_sha256`를 뺐다.
- **§4-5 status·종료 코드의 보충**: 마지막 줄 `status`에 `not_judgeable`을 더한다. 종료 코드 1은 $S_1$ 실패(`s1_defect`·`s1_no_effect`)와 `not_judgeable`, 종료 2는 입력 거부(금지 경로, sha 불일치, dirty, fold 인자 오류, 끝 시점 트리 변경)다.
- 같은 요구를 R1–R3 러너 공통 요구로 옮겼다([RUN] [`m33_runner_spec.md`](m33_runner_spec.md) v0.3.6 §1).

### A-1. D0 재실행 D0′ (사용자 결정 2026-11-15, C1 전, 결과 전에 정의)

- **순서**: 개정 실행기 커밋 → verif-protocol-auditor 실행 전 감사(`log_errors.py`·`placeholder_scan.py` 종료 0 포함) → **D0′ 실행** → **대조** → C1([PLAN] §4 표).
- **실행 조건**
  - 개정 실행기를 커밋한 뒤 깨끗한 트리에서 돌린다.
  - **확정 명령**(저장소 루트에서, 다른 인자 없음): `conda run -n gpr-dev python -I -B analysis/detection/m33_d0_gate.py --out-root runs/eval/m33_d0r`
  - 기본 경로(`runs/eval/m33_d0`)는 C0b 결과라 덮어쓰지 않는다. 실행기는 출력 폴더가 있으면 `--overwrite` 없이 거부하므로, 실수로 기본 경로를 쓰면 종료 2로 멈춘다. `--overwrite`·`--smoke`·`--limit`·`--synthetic-limit`·`--folds`·`--allow-dirty`는 쓰지 않는다.
  - **C0b 폴더 보호**: `--out-root`가 C0b 결과 폴더 `runs/eval/m33_d0`(기본값, 생략 포함)로 해석되면 실행기는 `--overwrite`가 있어도 `input_error`·종료 2로 멈춘다(`C0B_OUT_ROOT`·`is_c0b_out_root`, 경로 resolve·대소문자 무시 비교, git 측정·추론 전). 다른 폴더의 `--overwrite`는 기존처럼 동작하나 D0′에서는 쓰지 않는다(사용자 결정 2026-11-15). (D0′ 뒤 C1 1단계 도구 보강, 2026-11-16) C0b 폴더의 **하위 폴더**를 `--out-root`로 주는 것도 거부한다.
  - `judgment.judgeable` = true여야 한다. `not_judgeable`·`input_error`·`error`이면 대조하지 않고 원인을 고친 뒤 다시 돌린다(결과값은 보지 않는다).
  - **봉인 PP 단언**(실행기 모듈 상수로 구현, 인자 아님): `assert_sealed_pp`를 시작 트리 측정 직후, 입력을 읽거나 추론하기 전에 부른다. 하나라도 다르면 `input_error`, 종료 2다. 통과하면 `run_identity.sealed_pp_assertion`(`checks`, `expected`, `measured`, `result`)에 남긴다. 단언 상수는 다음과 같다.
    - `C0B_PP_SET_SHA256`: S0–S5 집합별 `PPConfig.sha256()`(C0b `d0_gate.json`, S3 = `eb2796dd…`).
    - `C0B_PP_SETS_SHA256` = `1f2b0cf2…`(집합 전체).
    - `C0B_SELECTED_SET` = S3.
    - `C0A_SRC_TREE_ID` = `164f0990…`(src tree id).
    - `C0A_PIPELINE_BLOB` = `dd5e7bcb…`(`src/gpr_detection/pipeline.py` blob).
  - **순서 제약**: D0′는 `src/`를 바꾸기 전에 돌린다. `src/`가 바뀌면 src tree id 단언에 걸려 실행기가 종료 2로 멈춘다. D0′와 C1 사이에 `src/`를 바꾸는 작업(R 러너 구현 등)이 있으면 D0′를 먼저 끝낸다.
- **대조 항목**(D0′ `d0_gate.json` ↔ C0b `runs/eval/m33_d0/d0_gate.json`, sha는 C0b 기록값으로 먼저 단언)
  0. **대조 범위**(사용자 결정 2026-11-15, D0′ 결과 전, verif 실행 전 감사 반영): 판정 범위만 본다.
     - 집합: C0b `selection.evaluated`(S1–S4). S5는 대조하지 않는다(해석 결정 1).
     - 조건: 게이트 조건 A0·A1·A3의 조건·fold별 칸. A2(보고 조건)와 Y1은 기록만 하고 불일치 사유로 쓰지 않는다.
     - **키 처리**: 두 json 키의 합집합으로 비교하고, 한쪽에 없는 키는 0으로 본다. null↔null은 같음, null↔수는 불일치다. 비교 칸 수가 기대 수(예: G1·G2는 평가 집합 4 × 게이트 조건 3 × 5 fold = 각 60칸)와 다르면 불일치다. 비교 칸이 0이면 입력 오류다.
  1. 선택 `pp_set_id`(= S3)와 `selection.stop_reason`.
  2. 평가된 집합(S1–S4)마다 G1–G4 게이트별 통과·실패와 어느 집합까지 평가했는지(`selection.evaluated`).
  3. C0b 주 1의 경계 박스(fold 1 A3 `pipe_band`, crop `NM_20221102(142627)_N00_B10_g1.0_x(10892-11148)`): D0′ `predictions.csv`의 전체 정밀도 점수가 0.25보다 큰지, 범주가 d인지, $S_2$부터 `removed_d`인지.
  4. 수치: G3의 $H(S_0)$·$H(S_j)$·노출 수, G4 증분·누적·규칙별 수(`by_rule`)·조건별 수(`by_condition`: A1·A3·Y0), G1·G2 조건(A0·A1·A3)·fold별 잃은 정답 수·덮임 감소 길이.
- **불일치 판정(결과 전 고정)**
  - (사용자 결정 2026-11-15, D0′ 실행 전. 아래 허용 폭이 엄격한 초안을 대신한다.)
  - 1 또는 2가 하나라도 다르면 허용 폭과 상관없이 **불일치**다. C1로 넘어가지 않고 사용자에게 보고한다.
  - 3은 **확인해서 기록만** 한다(점수, 범주, 집합별 `pp_action`). 불일치 사유가 아니다. 넓은 허용 폭이 박스 몇 개의 변화를 이미 흡수하기 때문이다.
  - 4는 **같은 집합·조건 칸**끼리 비교한다. G1·G2는 `d0_gate.json`에 기록된 조건·fold별 칸이다. 허용 폭은 다음과 같다.
    - G1: 잃은 정답 수 차이 ≤ 1개.
    - G2: 덮임 손실 차이 ≤ ±0.2 %p.
    - G4: 증분·누적·`by_rule`·`by_condition`(A1·A3·Y0) 각각 차이 ≤ max(5 %, 5개). 5 %는 C0b 값 기준이다.
    - G3: $H(S_0)$·$H(S_j)$·노출 수 차이 ≤ 1.
    - 폭 안의 차이는 모두 목록(집합, 조건, fold, 항목, C0b 값, D0′ 값)과 추정 원인(경계 박스, GPU 비결정성 등)을 적어 공개한다.
    - 폭을 넘는 차이가 하나라도 있으면 **불일치**다. C1로 넘어가지 않고 사용자에게 보고한다.
  - 1·2가 같고 4가 모두 폭 안이면 "D0′가 C0b 선택(S3)을 확인했다"로 기록한다. 이때도 C0b 결과와 판정서는 그대로 두고, D0′ `d0_gate.json` sha와 대조 결과를 C1 커밋에 적는다.
- **대조 주체와 절차**(사용자 결정 2026-11-15, D0′ 결과 전)
  - 대조 스크립트 `analysis/detection/m33_d0r_compare.py`(python-expert, 결과 전 작성)가 위 0–4와 허용 폭을 기계적으로 적용한다. C0b `d0_gate.json`끼리 비교하는 자체 시험(차이 0, 칸 수 = 기대값)을 통과한 뒤 커밋해 sha를 고정한다. 커밋은 D0′ 실행 전이다.
  - D0′ 뒤 이 스크립트를 돌리고 cv-detection-expert가 출력을 해석한다(폭 안 차이의 추정 원인 포함). 마지막으로 verif-protocol-auditor 판정 후 감사를 받는다.
- **재실행 대비(예비 폴더)**: D0′가 판정 불가(종료 1·2·3, `not_judgeable`·`input_error`·`error`)로 끝나면 실패 폴더 `runs/eval/m33_d0r`는 열지 않고 남긴다. 원인을 고친 뒤 `--out-root runs/eval/m33_d0r2`로 다시 돌린다. 실행기 코드를 고쳤으면 새 blob이므로 실행 전 감사를 다시 받는다. 재실행이 있었으면 그 사실과 실패 폴더의 종료 코드를 대조 기록에 적는다.
- **C1 감사 확인 항목**: §6 고정 목록의 '출력 형식'은 문자 그대로는 바뀌었다(부록 A). 부록 A "성격"의 해석(§6 목록은 C0a 시점 값을 가리키고 규칙·문턱은 불변)이 받아들여지는지를 C1 실행 전 감사에서 명시적으로 확인한다.
- **C1 sha 표**: D0 실행기 blob 두 개를 나란히 적는다. C0a 실행기 blob(53c4e16, C0b 결과의 출처)과 개정 실행기 blob(D0′의 출처, 개정 커밋)이다. `m31_d1_common.py`도 같은 방식으로 두 blob을 적는다. D0′ 출처 blob은 **882e881 값**(`m33_d0_gate.py` `1775f9a2…`, `m31_d1_common.py` `0ba0675c…`, `m33_d0r_compare.py` `ee802e8b…`)이다. C1 1단계 도구 보강으로 `m33_d0_gate.py`·`m33_d0r_compare.py`가 다시 바뀌었으므로 C1 커밋 blob을 별도 열로 함께 적는다.
- **대조 스크립트 개정(D0′ 뒤, 2026-11-16, C1 1단계 도구 보강, verif 판정 후 감사 권고 반영)**: base sha를 기록값과 단언한다. Y1 기록 칸(`report.synthetic_apex_hits`·`report.rule_counts_outside_gate`의 Y1)을 기록 전용으로 더했다. `judgment` 블록이 없는 입력은 거부한다. 칸 수가 기대값과 다르면 `mismatch`다(이전에는 보고만). 이 개정은 이미 끝난 D0′ 대조(882e881 스크립트, `match`)를 바꾸지 않는다. 다시 대조할 일이 생기면 개정 스크립트를 쓴다.
- **상태(기록, 규칙 아님)**: D0′ 완료(2026-11-15, HEAD 882e881, `runs/eval/m33_d0r/d0_gate.json` sha256 `16d7d2ad…`, 대조 status `match`, 183칸 차이 0). "D0′가 C0b 선택(S3)을 확인했다." 기록은 [`m33_d0r_record.md`](m33_d0r_record.md)다. verif 판정 후 감사 2026-11-15 완료(차단 0, 기록 6, gates·report 전체와 CSV 전 행 독립 재집계 일치).

## 변경 이력

| 날짜 | 판 | 내용 |
|---|---|---|
| 2026-11-15 | v0.1 | 최초 작성(X1): 정의, 순서열·선택, D0 자료 허용·금지 목록, 조건 A0–A2·Y0, 게이트 G1–G4, 출력, 시험, 결과 전 고정 항목(C0a/C0b) |
| 2026-11-15 | v0.2 | [X2] 반영. $y_z$ = `ZeroResult.k_zero` / `samples_per_px`(tree id 고정). A3(위 1/4 안 행 에너지 최대 행을 영점으로, A0 예측 재사용)를 게이트에 넣고 A2는 보고로 내렸다. Y1 기록 끝 변형(꼭짓점 + 0.5/1/2/3 P)은 보고 + 손실 수 공개로 두고 게이트에 넣지 않았다(사유: 잘린 관은 깊이를 낼 수 없어 J1 손실과 직결되지 않음, [X2] 권고와 같음). PP-a 진단(위 끝이 내용 안인 제거 수)을 D0·R2·J1에 추가했다. **반영하지 않은 것**: 영점 바로 아래(< 0.3 m) 관 합성 추가(새 gprMax 필요, [X2]도 이번에는 넣지 않기를 권고, 한계로 공개). 규칙 정의·순서열·게이트 문턱은 바뀌지 않았다 |
| 2026-11-15 | v0.2.1 | 무결성 반영(규칙 불변): 상위 판 번호를 계획 v0.2.4로 고치고, 봉인 순서를 [PLAN] §4 표(S0 → C0a → D0 → C0b → C1 → …)로 적었다. X2 검토 문구를 과거형으로 바꿨다 |
| 2026-11-15 | v0.2.2 | C0a 전 구현 해석 확정(규칙·문턱 불변). §4-7 신설: 무시 박스는 PP 전 1회 제거, G4는 AIHub conf > 0.25·Y0 conf 0.25 병합 박스 전부, G1·G2는 15칸 중 하나라도 값이 없으면 실패, G3는 정확히 0, 추론 문턱·배치. §4-1에 `m31_d1_r4_verdict.json`·`m33_d0_gate.py`·`m33_assert_no_aihub_test.py`를 넣었고 `m32_recipe_match.py`는 import하지 않음(시험 비교만)을 적었다. §2 범주는 "같은 함수"를 "같은 정의(시험 2)"로 고쳤다. §5 시험 1을 가짜 검출기 11측선(1a)·HEAD↔작업 트리 11측선(1b)으로 바꿨다(저장된 M3-2 결과는 열지 않음). gpr-processing 검토 반영: A3 $y_z$ = 행 $j$(항등 격자) 명시, A2 무시 판정을 +8 옮긴 뒤 자르기 전 중심으로(구현 결함 수정, A1↔A2 불변), Y1 주기를 Y0 r0 `period_samples`로 고정(고정안 채택, 짧은 기록 재측정은 최대 3.8배 어긋남)하고 진단 열을 추가했다. §6에 Y1 주기 고정·A2 무시 순서를 넣었다 |
| 2026-11-15 | v0.2.3 | stats-design 검토에 따른 사용자 결정(결과 전, 문턱 값 불변). G1·G2를 총손실로 바꿨다(S_0에서 맞고 S_j에서 놓친 정답 수 / 정답 수, 띠는 정답별 덮임 감소분 합 / 길이 합, 이득은 빼지 않고 따로 보고. 근거: PP-b가 정답도 잘라 IoU를 올릴 수 있음). G4를 증분(S_j − S_{j−1} 처리 수)으로 바꿨다. G3는 H(S_0) ≥ 1일 때만 정의되고 H(S_0)와 노출 수를 함께 낸다. S_1 실패를 `s1_defect`(G1–G3)와 `s1_no_effect`(G4만)로 나눴다(종료 1 공통, `stop_reason`·`status`로 구분). §7 5에 게이트의 통계적 성질(G1 개수 규칙, G2 fold 3 띠 1개 0.32 %p, 분산비 1.21·2.24, G3 상한 16 %·PP-d 검정력 0, 순차 선택의 선택 편향, 경계 부호 ≥/> 차이 — 통일하지 않고 공개)을 넣었다 |
| 2026-11-15 | v0.2.3 (C0a 전 무결성 보강) | 규칙·문턱 불변. §4-3 A2 내용 범위를 $y_\text{bot}$ = 256 − max($h_\text{pad}$ − 8, 0)으로 바로잡았다(코드·A1↔A2 불변성과 같은 식). §4-5 출력: `predictions.csv`는 crop 전용으로 적고, 합성 Y0·Y1 병합 박스는 새 `fold<k>/synthetic_boxes.csv`(L1·L0 좌표, 집합별 `pp_action`)에 쓰게 했다(재감사·재현용). §4-7 2: 규칙별 처리 수 `by_rule`을 누적과 같은 범위(A1·A3·Y0)로 세고 합 일치를 단언, 게이트 밖 조건(A0·A2·Y1)의 규칙별 수는 보고. 상태 줄의 판 이력 반복을 줄였다 |
| 2026-11-15 | v0.2.4 | **C0b(0f11c34) 뒤·C1 전·R1–R3 측정 전.** D0 실행기 정밀도·판정 가능성 개정(python-expert 수정, 미커밋)을 부록 A로 남겼다. 전체 정밀도(`repr(float)`), 기대 fold 0–4·`not_judgeable`(종료 1)·`judgment` 블록, 끝 시점 트리 재측정(다르면 종료 2), 텍스트 입력 LF sha(`inputs_sha256_lf`), `g1_inputs_sha256` 제거. **게이트 규칙·문턱·선택·`PPConfig`·PP 구현 불변**이라 §6의 C0a 재작성 대상이 아니다. D0 재실행 때 적용하고, C0b의 기존 D0 결과(53c4e16 실행기 출력)와 선택 S3는 그대로다. §1–§5 본문은 고치지 않았다. (같은 판, 이어서) 사용자 결정(2026-11-15, C1 전에 개정 실행기로 D0를 다시 돌림)과 integrity 권고: 부록 A "성격"에 §6 고정 목록('출력 형식', 코드 blob)은 C0a 시점 값을 가리키고 D0′는 개정 blob·출력 형식을 쓴다는 해석을 적었다. "판정·선택 불변"은 아직 재실행으로 확인된 값이 아님을 적었다. A-1 D0′(순서, `--out-root runs/eval/m33_d0r` 한 인자, judgeable 필수, `PPConfig`·`pipeline.py` 단언, 대조 항목 4, 결과 전 불일치 기준, C1 sha 표에 실행기 blob 두 개)를 신설했다. 상태 줄에 현재 계획 v0.2.8을 적었다. (이어서) **사용자 결정(2026-11-15, D0′ 실행 전)**: A-1 불일치 기준을 엄격한 초안(수치는 같아야 하고, 차이는 점수 0.25 ± 5e-5 경계 박스로만 설명될 때 허용, 주 1 경계 박스 처리가 다르면 불일치)에서 미리 정한 넓은 허용 폭으로 바꿨다. 같은 집합·조건 칸에서 G1 ≤ 1개, G2 ≤ ±0.2 %p, G4 ≤ max(5 %, 5개), G3 ≤ 1이다. 폭 안의 차이는 목록과 추정 원인을 공개한다. 항목 3(주 1 경계 박스)은 기록만 한다. 항목 1·2(선택·`stop_reason`·게이트별 통과·실패·평가 범위)는 그대로 불일치 사유다. (이어서) python-expert의 D0′ 단언 구현(미커밋)을 A-1 실행 조건에 반영했다: 확정 명령(`--out-root runs/eval/m33_d0r` 한 인자), 모듈 상수 단언 `assert_sealed_pp`(`C0B_PP_SET_SHA256`·`C0B_PP_SETS_SHA256`·`C0B_SELECTED_SET`·`C0A_SRC_TREE_ID`·`C0A_PIPELINE_BLOB`, 입력·추론 전, 어기면 종료 2, `run_identity.sealed_pp_assertion`), `src/` 변경 전 실행 제약, C0b 폴더는 실행기가 `--overwrite`도 거부(`is_c0b_out_root`, 하위 폴더는 다음 개정에서). (이어서) **사용자 결정(2026-11-15, D0′ 결과 전, verif 실행 전 감사의 실행 전 수정 2건)**: A-1 대조 범위를 판정 범위로 한정(집합 S1–S4 = C0b `selection.evaluated`, S5 제외, 조건 A0·A1·A3, A2·Y1은 기록만)하고, G4 허용 폭에 `by_condition`(A1·A3·Y0)을 넣고, 키 처리(합집합·없는 키 0·null 규칙·칸 수 기대값 불일치·0칸 입력 오류)를 정했다. 대조 스크립트 `m33_d0r_compare.py`(결과 전 작성·자체 시험·커밋 고정, cv 해석, verif 판정 후 감사)와 예비 폴더 `runs/eval/m33_d0r2`(판정 불가 시, 실패 폴더는 열지 않음, 코드 수정 시 감사 재실행)를 적었다. C1 감사 확인 항목(§6 '출력 형식' 문자 변경과 부록 A 해석)을 넣었다. (이어서, D0′ 뒤) A-1 끝에 상태 기록 한 줄(D0′ 완료, `match`, 기록 `m33_d0r_record.md`)을 넣었다. 규칙은 바꾸지 않았다. (이어서, 2026-11-16, C1 1단계 도구 보강) C0b 하위 폴더 거부, 대조 스크립트 개정(base sha 단언, Y1 기록 칸, judgment 필수, 칸 수 불일치 = mismatch, 끝난 D0′ 대조에는 영향 없음), C1 sha 표의 D0′ 출처 blob은 882e881 값이라는 규칙을 적었다. (이어서, C1 실행 전 감사 반영, 2026-11-16, R1–R3 결과 전) 부록 A §6 해석에 src tree id도 C0a 시점 값이라는 점을 더했다 |
