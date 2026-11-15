# M3-3 X1 — 경계 오검출 후처리(PP) 명세와 개발 게이트 D0

- 상태: **v0.2.1**(2026-11-15, cv-detection-expert). [X2] [`m33_x2_pp_physics_review.md`](../processing/m33_x2_pp_physics_review.md)(gpr-processing 물리 검토)를 반영했다. [PLAN] [`m33_cross_domain_plan.md`](m33_cross_domain_plan.md) v0.2.4 §3을 구체화한다(S0 9b28694는 v0.2.2).
- 이 문서는 **J1 판정 감사 대상**이다(QA (b)). 봉인 순서는 [PLAN] §4 표다: S0 → **C0a**(이 문서 §6 고정 항목, D0 전) → D0 → **C0b**(D0 결과·선택 집합) → C1 → …
- 물리 기준($t_z$, $y_\text{bot}$, $P_ell$)은 gpr-processing-expert가 X2로 검토했다. 그 의견을 반영해 v0.2로 올렸고, v0.2.1은 표기 정리다(규칙 불변). C0a는 v0.2.1로 커밋한다.
- 구현은 python-expert가 한다.
  - 위치: `src/gpr_detection/pipeline.py`의 병합 뒤 단계(`pp_mode`, 기본 꺼짐), 상수는 `PPConfig`에 둔다.
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

- 범주는 **원래 박스**로 한 번만 정한다. 순서는 a > b > d이고, `m4_category`와 같은 함수를 쓴다.
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
- `pp_mode` 기본값은 꺼짐이고, 꺼짐 출력은 지금과 비트 단위로 같아야 한다(시험).

## 3. 포함 순서열과 선택 규칙

$$S_1=\{a\}\subset S_2=\{a,d\}\subset S_3=\{a,b,d\}\subset S_4=\{a,b,d,c(0.35)\}\subset S_5=\{a,b,d,c(0.50)\}$$

- PP-c는 "점수 < $s_c$이면 제거"이므로 $s_c$가 높을수록 더 많이 지운다. $c(0.35)$가 지우는 박스는 $c(0.50)$가 지우는 박스에 모두 들어 있다.
- **선택**: $S_1$부터 차례로 게이트(§4-4)를 본다. 통과하는 가장 큰 $S_j$를 고르고, 처음 실패한 집합의 뒤는 보지 않는다.
- $S_1$이 실패하면 멈추고 사용자에게 묻는다(구현 결함 신호).
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
| 코드 | `src/gpr_detection/**`, `analysis/detection/m31_d1_common.py`, `analysis/detection/m32_aihub_perturb.py`(채움 흉내 함수 재사용) | git blob 기록 |

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
| A2(보고) | A1 + 위쪽 평탄 128 띠 $h_\text{top}$ = 2 $P_A$(8 px)를 덧붙이고 같은 높이만큼 아래를 잘라 256 유지 | $y_\text{bot}$ = 256 − $h_\text{pad}$(자른 뒤) | $y_z$ = $h_\text{top}$ |
| **A3** | A0 crop 그대로(추가 추론 없음, A0 예측 재사용) | $y_\text{bot}$ = 256 | $y_z$ = **지표 띠 행**: 행 0–63(위 1/4) 안에서 행 에너지 $\mathrm{mean}_x(q-\bar q)^2$가 최대인 행. $\bar q$는 crop 전체 평균이다. 동률이면 작은 행을 고른다. 라벨·검출기와 무관한 결정적 규칙이다([X2] §2-2) |

- $P_A$ = 4.06 px이다(AIHub crop 주기 상수, M3-2 §1-6). AIHub에서는 $P_ell$ 대신 $P_A$를 쓴다.
- 정답 처리
  - 중심이 채움 안(또는 A2에서 잘려 나간 곳)이면 무시 박스다(S3 규칙: 무시 박스와 IoU ≥ 0.5 또는 IoA ≥ 0.5인 예측은 버림).
  - 남은 정답은 PP 전·후 모두 $y_2 \leftarrow \min(y_2, y_\text{bot})$으로 **같이 자른다**. PP-b 자르기가 정답과의 IoU를 인위적으로 바꾸지 않게 하려는 것이다.
- A2는 [X2] §2-2의 지적대로 시험력이 약하다. AIHub 실제 지표 띠가 덧붙인 띠 아래에 있기 때문이다. 그래서 **게이트에서 빼고 보고**로 둔다. PP-d 게이트 근거는 A3와 Y0다.
- **합성 기록 끝 변형(Y1)**([X2] §2-1)
  - Y0와 같은 20모델의 L0 기록을 꼭짓점 샘플 + $\{0.5, 1, 2, 3\}\,P_\text{line}$에서 잘라 시간창을 줄인다. $P_\text{line}$은 그 측선의 r0 `period_samples`다.
  - 그 뒤 C′ 경로로 측선을 만든다. 정답 꼭짓점은 그대로다.
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
| G1 | 쌍곡선 conf > 0.25 재현율 손실 $R(S_0)-R(S_j)$. 짝지은 객체, IoU 0.5, 5 fold 비가중 평균. **A0·A1·A3 중 가장 나쁜 조건** | ≤ **0.5 %p** |
| G2 | 띠 가로 구간 재현율 손실(`band_metrics.py` 정의: conf > 0.25, 세로 겹침 ≥ 정답 높이 50 %, 길이 가중), 5 fold 평균, A0·A1·A3 중 가장 나쁜 조건 | ≤ **1.0 %p** |
| G3 | 합성 Y0 꼭짓점 포함 손실(M01–M20 × 5 fold, 맞힌 수의 차) | = **0** |
| G4 | 정의상 오검출 처리 수: $S_j$가 제거하거나 자른 박스 수(A1·A3·Y0 합) | > **0** |

- 보고(게이트 아님): A2의 G1·G2 값, Y1의 자르는 위치별 꼭짓점 손실 수와 규칙별 처리 수, cavity·manhole 재현율 손실, mAP50(conf 0.001) 차, 규칙별 처리 수(a·b·d·c), gain 사본별 분해, fold별 값, A3의 $y_z$ 행 분포.
- **PP-a 진단**(D0와 R2·J1 공통, [X2] §1-1 (ii)): PP-a로 제거된 박스 가운데 위 끝이 내용 안($y_1 < y_\text{bot} - P_ell$)인 수다. "잘린 실관일 수 있는 제거"를 뜻하고, 보고만 한다. D0 출력에는 조건별로, R2 `summary.json`과 J1 보고(§4 5)에는 측선 합으로 낸다.
- G1·G2는 손실이 음수(이득)여도 통과다.
- G4는 "규칙이 실제로 작동하는 자료인가"를 보는 상식 점검이다. 0이면 그 집합은 실패로 친다.

### 4-5. 출력 형식

- `runs/eval/m33_d0/`
  - `fold<k>/predictions.csv`: 조건, crop 또는 모델, 박스, 점수, 클래스, 범주, 집합별 `pp_action`.
  - `fold<k>/objects.csv`: 정답별 집합별 맞춤 여부.
  - `synthetic_apex.csv`
  - `d0_gate.json`: 집합별 G1–G4 값과 통과 여부, 선택 `pp_set_id`, `PPConfig` 해시, `run_identity`(git HEAD·`dirty` false, 코드 blob, 가중치 sha 5, crop 목록 sha 5, 매니페스트·채움 표·합성 파일 sha, 허용 목록 단언 결과, 열린 경로 목록).
- stdout에는 단계·건수·경과 시간·출력 경로만 찍는다. 마지막 줄은 `{"status","out_dir","n_conditions","n_images","elapsed_s"}`이고 값은 넣지 않는다.
- 종료 코드: 0 완료, 1 $S_1$ 실패(멈춤), 2 입력 거부(금지 경로, sha 불일치, dirty), 3 내부 오류.

### 4-6. 자원

- GPU: 추론 3조건(A0, A1, A2; A3는 A0 예측 재사용) × 5,473 crop + 합성 20모델 × (Y0 1 + Y1 4) × 5 fold로 약 6–10분(S3 약 14 s/조건/1,000 crop).
- 단일 프로세스, batch 64(M3-2 S5와 같은 배치 경로)다.

## 5. 시험 (python-expert, `tests/test_pp.py`)

1. `pp_mode` 꺼짐은 비트 단위로 같다. M3-2 r0 11측선 병합 박스로 확인한다.
2. 범주 배타성: 한 박스는 a·b·d 중 하나이고, `m4_category`와 같다.
3. b 자르기: 자른 뒤 $y_2 = y_\text{bot}$이고, 높이 < $P_ell$이면 제거된다. 경계 $\pm e$ 안의 박스는 그대로다.
4. d: 영점 실패면 작용하지 않는다. 경계 $y_z + 0.5P_ell$ 바로 위·아래에서 동작한다.
5. c: 순서열 포함 관계(c(0.35)의 제거 집합 ⊂ c(0.50)의 제거 집합)가 성립하고, 띠 외 클래스에는 작용하지 않는다.
6. 박스 순서를 섞어도 결과가 같다.
7. D0 실행기가 금지 경로를 열면 종료 2로 거부한다.

## 6. 결과 전 고정 항목 (C0a로 커밋, D0 실행 전)

- 이 문서의 §1–§5 전체: 정의, 순서열, 선택 규칙, 조건 A0–A3·Y0·Y1, $h_\text{top}$ = 8 px, A3 지표 띠 행 규칙, Y1 자르는 위치 {0.5, 1, 2, 3} P와 보고 전용 구분, 시드 20261114, 게이트 문턱, 허용·금지 목록, 출력 형식, PP-a 진단.
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

## 변경 이력

| 날짜 | 판 | 내용 |
|---|---|---|
| 2026-11-15 | v0.1 | 최초 작성(X1): 정의, 순서열·선택, D0 자료 허용·금지 목록, 조건 A0–A2·Y0, 게이트 G1–G4, 출력, 시험, 결과 전 고정 항목(C0a/C0b) |
| 2026-11-15 | v0.2 | [X2] 반영. $y_z$ = `ZeroResult.k_zero` / `samples_per_px`(tree id 고정). A3(위 1/4 안 행 에너지 최대 행을 영점으로, A0 예측 재사용)를 게이트에 넣고 A2는 보고로 내렸다. Y1 기록 끝 변형(꼭짓점 + 0.5/1/2/3 P)은 보고 + 손실 수 공개로 두고 게이트에 넣지 않았다(사유: 잘린 관은 깊이를 낼 수 없어 J1 손실과 직결되지 않음, [X2] 권고와 같음). PP-a 진단(위 끝이 내용 안인 제거 수)을 D0·R2·J1에 추가했다. **반영하지 않은 것**: 영점 바로 아래(< 0.3 m) 관 합성 추가(새 gprMax 필요, [X2]도 이번에는 넣지 않기를 권고, 한계로 공개). 규칙 정의·순서열·게이트 문턱은 바뀌지 않았다 |
| 2026-11-15 | v0.2.1 | 무결성 반영(규칙 불변): 상위 판 번호를 계획 v0.2.4로 고치고, 봉인 순서를 [PLAN] §4 표(S0 → C0a → D0 → C0b → C1 → …)로 적었다. X2 검토 문구를 과거형으로 바꿨다 |
