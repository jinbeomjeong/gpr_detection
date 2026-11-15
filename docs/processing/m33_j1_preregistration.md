# M3-3 J1 — 4TU 종단 판정 사전 등록 (초안 v0.3)

- 상태: **초안 v0.3**(2026-11-15). gpr-processing-expert가 v0.1을 썼다. cv-detection-expert가 공동 검토로 v0.2를 고쳤고, gpr-processing 재검토의 탐색 창 보완과 무결성 지적을 v0.3에 반영했다(X3).
  - 봉인 순서는 [PLAN] §4 표다: … → C0b → C1 → R1 → R2 → R3 → **C2** → J1 판정.
  - `{C2}`·`{R2_J1_POST_SHA}`·`{R2_J1_PRE_SHA}` 자리는 **C2 커밋**에서 채운다. C1 실행 전 감사의 `placeholder_scan.py`는 이 자리들을 허용 자리(`--allow`)로 받는다. C2 뒤에는 허용 없이 종료 0이어야 한다.
  - `{C0B}` 자리는 C0b 값으로 C1에서 채운다.
  - 실행 전 감사는 verif-protocol-auditor가 한다. `placeholder_scan.py`로 중괄호로 감싼 대문자 이름 형식의 자리표시가 남지 않았는지 확인한다.
- 형식은 [U3] [`preregistration_4tu_reeval_v1.md`](preregistration_4tu_reeval_v1.md)를 따른다.
- **J1에서 [U3]와 달라지는 것은 꼭짓점 후보 공급원 하나뿐이다.** [U3]의 신호처리 후보 `fourtu_apex_candidates_v1.csv` 대신, 검출기 fold 0의 PP 후 `pipe_hyperbola` 병합 박스에서 만든 후보를 쓴다([PLAN] [`m33_cross_domain_plan.md`](../experiments/m33_cross_domain_plan.md) §2-4).
- 출처 약어: [U3], [PLAN], [PP] [`m33_pp_spec.md`](../experiments/m33_pp_spec.md), [RS] [`m33_runner_spec.md`](../experiments/m33_runner_spec.md), [X2] [`m33_x2_pp_physics_review.md`](m33_x2_pp_physics_review.md), [EV] [`spec_eval_4tu_v1.md`](spec_eval_4tu_v1.md), [D] [`spec_depth_v1.md`](spec_depth_v1.md).

## 0. 요약

| 항목 | 내용 |
|---|---|
| 실행 | (1) 후보 생성 `analysis/processing/m33_j1_candidates.py` 1회 → (2) `scripts/eval_depth_4tu.py --candidates docs/processing/depth_4tu_j1_candidates_input.csv --prefix depth_4tu_j1` 1회 → (3) 판정 스크립트 `analysis/processing/m33_j1_judge.py`가 문구를 만든다. 모두 CPU, 단일 프로세스, BelowNormal |
| 판정 입력 | R2 실행의 **fold 0** `j1_boxes_post.csv`(PP 후, C0b 집합). sha를 단언한다 |
| 보고 입력 | fold 0 `j1_boxes_pre.csv`(PP 전), fold 1–4의 post·pre |
| 주 조합 | [U3]와 같다: `main`(B+D+E) · 묶기 0.10 m 완전 연결 · 지지 측선 ≥ 2 · 활동 간 중복 `keep_both` · 정답 `exclude_x` · **관 상단** · **검출 영점** · 범위 전체 · 기준선 1,000회 `default_rng(0)` |
| 측선 | 754측선(`keep_both`: 759 − 활동 안 사본 5). 활동 안 사본은 `eval_depth_4tu.py`가 지금처럼 뺀다 |
| 합격(U-3 4TU, 사용자 결정 2026-11-15로 그대로) | $p<0.05$ **그리고** $n_\text{pass}\ge97$ (판정률 B ≥ 0.25 = 97/388) |
| 판정 문구 | [U3] §2-5 표 4종을 그대로 쓴다. "신호처리 후보" 자리만 "검출기 fold 0 PP 후 박스 후보"로 바꾼다(§2-3) |
| 실질 지표(보고) | **B − 기준선 평균**, 기준선 2.5–97.5 % 구간과 함께. PP 전 박스로도 같은 값을 낸다 |
| 미리 알아 둘 성질 | [U3] §2-6: 활동 범위 기준선은 짝의 약 95 %를 맞힐 만큼 강하다. 그래서 불합격(기준선) 가능성이 높다는 것을 알고 기준을 유지한다(사용자 결정 2026-11-15, [PLAN] §2-4) |
| 감사 | verif-protocol-auditor 실행 전·판정 뒤 감사([PLAN] J1 행) |

## 1. 무엇을 한 번 돌리는가

### 1-1. 실행 단위

1. **후보 생성**(§3): fold 0 `j1_boxes_post.csv` → `docs/processing/depth_4tu_j1_candidates_input.csv`.
   - 열은 `fourtu_apex_candidates_v1.csv`와 같다.
   - `eval_depth_4tu.py`가 읽는 열은 `activity_id`, `path`, `trace`, `sample`, `tau_ns`, `source` 여섯 개다. 나머지 열(`site`, `t_ns`, `snr`, `n_left`, `n_right`, `v_straight_qc`, `note`)은 채우거나 빈 값으로 둔다.
2. **깊이·판정 계산**: `scripts/eval_depth_4tu.py --candidates docs/processing/depth_4tu_j1_candidates_input.csv --prefix depth_4tu_j1`.
   - 나머지 인자는 기본값이다(주 조합 = [U3]).
   - 출력은 `docs/processing/depth_4tu_j1_*.csv`와 `depth_4tu_j1.md`다.
3. **판정 문구**: `m33_j1_judge.py`가 summary의 주 조합 행에서 $n_\text{pass}$·$k$·기준선 분위를 읽어 §2-3 문구와 실질 지표를 `docs/processing/depth_4tu_j1_verdict.json`·`.md`에 쓴다.
- 1–3은 러너 한 번으로 묶는다. 판정용 값은 stdout에 찍지 않는다(memory `blind-prereg-runs`).
- 끝까지 돌면 그 결과가 최종이다. 같은 M3-3 안에서 다시 돌리지 않는다.
- **순서**(cv 공동 검토 추가): 판정 실행(fold 0 post)이 `depth_4tu_j1_verdict.json`을 쓴 **뒤에** 보고 실행을 한다. 보고 실행은 PP 전, fold 1–4, 신호처리 후보 재실행이다(J1-3).
  - 보고 실행은 prefix가 달라(`depth_4tu_j1_pre`, `_f1`–`_f4`, `_signal`) 판정 파일을 덮지 못한다.
  - 판정 래퍼는 판정 prefix 출력이 이미 있으면 거부한다(1회 실행).

### 1-2. 고정할 코드·입력 (C2 커밋 때 채움)

| 대상 | git blob/tree id | SHA-256(LF) | 쓰임 |
|---|---|---|---|
| `scripts/eval_depth_4tu.py` | `{C2}`(지금 HEAD blob 9289f6df…) | `{C2}` | 깊이·묶기·매칭·기준선 |
| `src/gpr_detection/` (tree) | `{C2}` | – | 깊이 경로·영점·로더 |
| `analysis/processing/m33_j1_candidates.py` | `{C2}` | `{C2}` | 박스 → 후보(§3) |
| `analysis/processing/m33_j1_judge.py` | `{C2}` | `{C2}` | 판정 문구·실질 지표 |
| R2 fold 0 `j1_boxes_post.csv` | – | `{R2_J1_POST_SHA}`(R2 summary `j1_inputs.folds["0"].post_sha256`) | **판정 입력** |
| R2 fold 0 `j1_boxes_pre.csv` | – | `{R2_J1_PRE_SHA}`(같은 곳의 `pre_sha256`) | 보고 |
| C0b `PPConfig` 해시, `d0_gate.json` | – | `{C0B}` | PP 집합 정체 |
| `docs/processing/fourtu_lines_v1.csv` | `{C2}` | `{C2}` | 측선 메타 |
| `docs/datasets/fourtu_duplicate_radargrams.csv` | `{C2}` | `{C2}` | 중복 규칙 |
| `docs/processing/m29c_qc_constants.json` | `{C2}` | `{C2}` | QC 상수 |
| 4TU 원본 메타 csv 4개 | – | `{C2}` | 정답·메타([U3] §1-2와 같은 대상) |
| 후보 csv `depth_4tu_j1_candidates_input.csv` | – | 생성 직후 기록 | 1단계 산출 |

- 판정 스크립트는 `j1_boxes_post.csv`의 sha와 `PPConfig` 해시를 C2 값과 단언한다. 다르면 종료 2다.
- 후보 생성 직후 후보 csv의 sha를 기록하고, 깊이 계산은 그 sha를 단언한 뒤 시작한다.

### 1-3. 명령줄 (초안)

```
conda run -n gpr-dev python -I analysis/processing/m33_j1_judge.py ^
  --boxes runs/eval/m33/r2/fold0/j1_boxes_post.csv --boxes-sha256 {R2_J1_POST_SHA} ^
  --pp-config-sha256 {C0B} --prefix depth_4tu_j1
```

`m33_j1_judge.py`는 후보 생성 함수를 부르고 `eval_depth_4tu.main([...])`을 같은 프로세스에서 부른 뒤 문구를 쓴다. `eval_depth_4tu.py`는 고치지 않는다.

### 1-4. 난수와 기준선

- [U3] §1-4와 같다. `numpy.random.default_rng(0)`, 주 조합 활동 범위 기준선 1,000회, 보조 200회.
- **기준선을 새로 계산하는 방법**: 기준선은 활동마다 J1의 추정 수 $n_{e,a}$(지지 필터 뒤 묶음 수)를 그대로 쓴다. 그 수만큼 $U[\min d_\text{gt},\max d_\text{gt}]$에서 깊이를 뽑아 같은 헝가리안 매칭을 한다([U3] §2-3, `draw_baseline`).
  - 그래서 J1 기준선은 J1 출력의 활동별 추정 수에 따라 [U3] 값(0.397, M2-9 0.451)과 달라진다.
  - 계산은 `eval_depth_4tu.py`가 한 실행 안에서 하므로 따로 구현하지 않는다.

### 1-5. 실행 전 점검 (모두 통과해야 실행)

1. 깨끗한 트리(`dirty` false). §1-2의 blob·sha가 C2 값과 같아야 한다.
2. R2 `summary.json`의 `j1_inputs`가 판정 입력 sha·`PPConfig` 해시와 같아야 한다.
3. `--dry-run`: `eval_depth_4tu.py --dry-run --candidates docs/processing/depth_4tu_j1_candidates_input.csv`가 후보·측선·정답·중복 표를 읽고 종료 0이어야 한다.
4. 후보 csv 열 형식과 측선 범위: 모든 `(activity_id, path)`가 754측선 안에 있어야 한다.
5. `log_errors.py`(러너 로그)와 `placeholder_scan.py`(이 문서)가 종료 0이어야 한다.

### 1-6. 실행 실패 처리

[U3] §1-6과 같다.
- 출력 전에 예외가 나면 원인을 고친 뒤 다시 돌린다(결과를 보지 않은 상태).
- 출력이 일부라도 쓰였으면 결과를 본 것으로 친다.
- 끝까지 돌았으면 그 결과가 최종이다.

## 2. 판정 기준

### 2-1. 판정률 B

[U3] §2-2와 같다. 기호(활동 93개, $N_g$ = 388, 헝가리안 짝, ±0.30 m, $\tau=10^{-12}$)도 같다.

### 2-2. 무작위 기준선 $p$와 절대 목표

[U3] §2-3·§2-4와 같다. $p=k/1000$이고 $k=\#\{k':n^{(k')}_\text{pass}\ge n_\text{pass}\}$이다(동률은 관측에 불리). T = 0.25(97/388)다.

### 2-3. 판정 문구

- [U3] §2-5 표 4종(합격 / 불합격(기준선) / 불합격(목표) / 불합격(둘 다))을 그대로 쓴다.
- 문구의 "(판정률 B, `main`·관 상단·검출 영점·X 제외)" 뒤에 "**· 후보: 검출기 N2-a′ fold 0, PP(C0b) 후 박스**"를 붙인다.
- 경계 표시(p 경계, 목표 경계)도 [U3] §2-5와 같다.
- "제186조 준수"라고 쓰지 않는다.

### 2-4. 실질 지표 (보고, 판정 아님)

- $\Delta_B=B-\bar B_\text{base}$. $\bar B_\text{base}$는 주 조합 기준선 판정률 B의 1,000회 평균이다. 기준선 B의 2.5·97.5 % 분위와 함께 낸다.
- 같은 값을 PP 전 박스(fold 0 `j1_boxes_pre.csv`)로도 낸다. 후보 생성과 깊이 계산을 같은 규칙으로 한 번 더 돌린다(보고 실행, 판정 아님).

## 3. 박스 → 꼭짓점 후보 규칙 (물리 명세)

**입력**: fold 0 `j1_boxes_post.csv`의 **모든 행**(`pipe_hyperbola`, L0 격자 `i0, i1, k0, k1`).
- 러너가 이미 운용 conf 필터(점수 > 0.25, 병합 경로와 같은 필터)와 PP(C0b)를 적용했고, 제거된 박스는 싣지 않는다([RS] §5). 그래서 J1 쪽에서 다시 거르지 않는다. 문턱 경계(≥ / >)가 두 곳에서 달라지는 일을 막기 위해서다.

**측선 준비**(측선마다 1회, 깊이 경로와 같은 값)
- 단면은 피팅 갈래 (1)–(4) 배경 제거 단면이다(`prepare_fitting_branch` 기본, median 301). 포락선 $e$는 시간 방향 Hilbert이고 2N 영 채움을 쓴다.
- 영점 $t_z$는 `prepare_depth_inputs`(기본 `DepthConfig`)의 `zero.t_zero_ns`다. `eval_depth_4tu.py`의 검출 영점과 같은 함수다.
- $P_\text{line}$은 r0 `make_detection_image`의 `period_samples`다.

**박스마다 후보 1개**

| 단계 | 규칙 | 물리 근거 |
|---|---|---|
| 1 탐색 창 | 행 $[k_\text{lo},\ k_0+2.5P]$. $k_\text{lo}=\max(k_0-0.5P,\ k_z+0.5P)$이고, 영점이 없으면 $k_\text{lo}=k_0-0.5P$다. $k_z$는 `zero.k_zero`(L0 샘플)이고 여유 0.5 P는 PP-d와 같다. 창 전체를 $[0,\ n_s-1]$로 자른다. 창이 비면($k_0+2.5P \le k_z+0.5P$) 후보를 만들지 않고 사유 `window_above_zero`를 기록한다(v0.3, gpr-processing 보완: 남은 지표 띠 포락선이 최댓값으로 잡혀 깊이 약 0인 후보가 생기는 것을 막는다). 트레이스 $[i_0+\lfloor w/4\rfloor,\ i_1-\lfloor w/4\rfloor]$ ($w=i_1-i_0+1$; $w<8$이면 $[i_0, i_1]$ 전체) | 검출 박스 위 끝은 꼭짓점 근처다(꼭짓점이 위 끝보다 0.5 P 넘게 위인 경우 41개 중 2개, [D3] §13-6). 꼭짓점은 쌍곡선에서 가장 이른 점이라 박스 위쪽 띠에 있다. 가운데 절반 트레이스로 제한하면 날개 쪽 다른 반사를 피한다 |
| 2 꼭짓점 | 창 안 $e$의 최댓값 칸 $(i^*, k^*)$. 행은 3점 포물선 보간으로 분수 샘플 $\hat k$ | [U3] 신호처리 후보와 같은 "포락선 피크" 위상 기준이다(영점도 포락선 피크) |
| 3 시간 | `t_ns` $=\hat k\,\Delta t$, `tau_ns` $=\hat k\,\Delta t-t_z$. 영점 실패면 `tau_ns` 빈 값(NaN) | `eval_depth_4tu.py`가 `candidate_tau_missing`로 처리한다 |
| 4 중복 제거 | 같은 측선에서 두 후보가 ±3 트레이스·±0.5 $P_\text{line}$ 안이면 박스 점수가 높은 쪽 하나만 남긴다. 점수가 같으면 (트레이스, 샘플)이 작은 쪽을 남긴다. 처리 순서는 점수 내림차순 탐욕이다(결정적) | 겹친 박스가 같은 꼭짓점을 두 번 세면 활동별 추정 수와 기준선이 바뀐다. M3-2 일치 규칙과 같은 반경이다 |
| 5 기록 | `source` = `j1_fold0_post`(보고 실행은 `j1_fold0_pre`), `note` = 박스 점수·`pp_action`·`tile_truncated` | 보고용 |

- 후보마다 깊이 박스는 `eval_depth_4tu.py`가 [EV] §4-1 규칙(꼭짓점 ±0.8 m, 메타 속도 날개)으로 다시 만든다. 검출 박스 자체는 깊이 창으로 쓰지 않는다. 이렇게 해서 [U3]와 달라지는 것을 후보 공급원 하나로 한정한다.
- **경로 확정**(사용자 결정 2026-11-15, [PLAN] v0.2.4 §2-4): 박스 → 꼭짓점 후보 → `eval_depth_4tu.py`(무수정)다. `estimate_line_depths`는 [U3]의 QC·묶기·지지 필터를 거치지 않아 J1에 쓰지 않는다. 보고 변형으로도 이번에는 돌리지 않는다.
- 띠(`pipe_band`) 박스는 J1에 쓰지 않는다.

## 4. 보고할 보조 지표 (판정에 쓰지 않음)

1. [U3] §3 S1–S9에 해당하는 표: 상단·중심 × 영점 3, 보조 변형, X 포함, 중복 민감도, 게이트, 현장별, 2사건 측선.
2. PP 전 박스의 B·$p$·$\Delta_B$(§2-4).
3. fold 1–4의 post 후보 B·$p$·$\Delta_B$(fold별 독립 실행, 같은 규칙).
4. **신호처리 후보 비교**: 같은 커밋의 `eval_depth_4tu.py`를 `fourtu_apex_candidates_v1.csv`로 다시 돌린 B·$p$·$\Delta_B$. 같은 코드에서의 공정 비교이고, M2-9 140/388은 옛 코드 값으로 함께 적는다.
5. 후보 수
   - 박스 수 → 후보 수(중복 제거 전·후).
   - 영점 실패, QC B·D·E 각 제외 수.
   - PP 단계별 제거 박스 수와, 그중 위 끝이 내용 안인 PP-a 제거 수([X2] §1-1 진단).
6. K 꼭짓점 SNR 4분위별 포함 여부는 R2가 보고하므로 여기서는 하지 않는다.

## 5. 구현 명세 (python-expert)

| ID | 내용 | 위치 |
|---|---|---|
| J1-1 | 후보 생성 `build_j1_candidates(boxes_csv, fold, pp, lines) -> DataFrame`. §3 단계 1–5를 따른다. 정답 파일은 읽지 않는다(허용 목록: 박스 csv, 측선 목록, 4TU 원자료, `src`). 결정적이고 난수를 쓰지 않는다 | `analysis/processing/m33_j1_candidates.py` |
| J1-2 | 판정 래퍼: (1) 단언 — 아래 "J1-2 단언 목록" 전부, (2) J1-1 → 후보 csv와 sha 기록, (3) `eval_depth_4tu.main(["--candidates", …, "--prefix", "depth_4tu_j1"])`, (4) summary 주 조합 행 → §2-3 문구, §2-4 실질 지표 → `depth_4tu_j1_verdict.json`·`.md`. stdout 마지막 줄은 `{"status","out","n_candidates","elapsed_s"}`이고 지표를 넣지 않는다. `eval_depth_4tu.main`은 `logging.basicConfig(level=INFO)`로 진행 로그를 낸다. 래퍼는 이 로그를 파일 핸들러로 돌려 stdout에 지표가 나가지 않게 한다. 종료 코드 0 / 2(입력 거부) / 3 | `analysis/processing/m33_j1_judge.py` |
| J1-3 | 보고 실행 옵션 `--report pre|fold1..4|signal`: 같은 경로로 prefix만 바꿔 돌린다(`depth_4tu_j1_pre` 등). 판정 파일은 쓰지 않는다 | 같음 |
| J1-4 | 시험 `tests/test_m33_j1_candidates.py` 다섯 가지. (e) 박스 위 끝이 영점 띠 위에 있어도 후보 행 $\hat k > k_z+0.5P$이고, 창이 비면 후보 없이 `window_above_zero`가 기록된다. 나머지 네 가지: (a) 합성 쌍곡선 박스 → 후보 꼭짓점이 정답 ±1 트레이스·±0.25 P 안. (b) 중복 제거(점수 높은 쪽 남김). (c) 영점 실패면 `tau_ns` NaN. (d) 열 형식이 `fourtu_apex_candidates_v1.csv`와 같고 `eval_depth_4tu.py --dry-run`을 통과 | `tests/` |

**J1-2 단언 목록**(cv 공동 검토 추가, 하나라도 어기면 종료 2. 단언 결과는 `depth_4tu_j1_verdict.json`의 `assertions`에 쓴다)
1. 깨끗한 트리(`dirty` false). HEAD는 C2 커밋이거나 그 뒤의 문서 전용 커밋이다. §1-2 표의 blob·sha(`eval_depth_4tu.py`, `src` tree, J1-1·J1-2, 측선·중복·QC 상수, 4TU 메타 csv 4개)가 C2 값과 같다.
2. 판정 입력 `j1_boxes_post.csv`의 sha가 C2 값과 같고, R2 `summary.json`의 `j1_inputs.folds["0"].post_sha256`과도 같다(키 구조는 [RS] §5).
3. R2 `run_identity`
   - 가중치 fold 0 sha = 7ead71c7fa62d85edb11f5614e78b11619062b455dca62f5cc66c93b2dcf8d92.
   - r0 11측선 `line_image` sha·설정 해시(conf·병합·r0)가 C1 값과 같다.
   - `PPConfig` 해시와 `pp_set_id`가 C0b 값과 같다.
   - R2 실행 당시 git HEAD가 C1 이후이고 `dirty` false다.
4. 박스 파일 내용
   - `fold` 열이 모두 0이다.
   - `pp_action`에 `removed_*`가 없다.
   - 클래스가 모두 `pipe_hyperbola`다.
   - `line_id`는 `activity_id`, `/`, `path`를 이어 붙인 형식이고 첫 `/`에서 나눈다. 파일의 `activity_id`·`path` 열이 그 분해와 같다([RS] §5).
   - 모든 `(activity_id, path)`가 754측선 목록 안에 있다.
5. 후보 csv를 만든 직후 sha를 기록하고, `eval_depth_4tu.main` 호출 전에 다시 읽어 같은지 확인한다.
6. **실행 전, 단계 1(후보 생성) 이전에** 출력 prefix `depth_4tu_j1*` 파일이 하나도 없다. 보고 실행은 판정 뒤에 돌므로 이 시점에는 보고 파일도 없어야 한다.
7. 블라인드: `eval_depth_4tu.main`의 `logging`과 `print`(stdout)를 둘 다 파일로 돌린다(`contextlib.redirect_stdout` + 파일 핸들러). 래퍼 stdout에는 마지막 줄 JSON만 남긴다.

- **기존 도구 재사용 판단**: `scripts/eval_depth_4tu.py`(M2-8·M2-9 판정 도구)를 **고치지 않고 그대로 쓴다.**
  - `--candidates` 인자가 이미 있다.
  - 후보 csv에서 여섯 열만 읽고, 상자 공급·QC·묶기·지지 필터·중복 처리·매칭·기준선·summary를 모두 한 실행에서 한다.
  - 판정 문구 자동화와 sha 단언만 새 래퍼(J1-2)에 둔다.

## 6. 공개 사항 (판정서에 옮김)

1. 후보 공급원만 [U3]와 다르다. 깊이 파라미터는 4TU로 튜닝하지 않았다. 검출 쪽도 4TU 깊이표를 보고 바꾸지 않았다([PLAN] §2-4).
2. fold 0을 판정 모델로 둔 규칙은 D3 때 4TU를 보기 전에 정한 것이다. A1 판정서 crop mAP50의 중앙값 fold다.
3. PP 집합은 AIHub·합성만 쓴 D0로 골랐다(C0b). [X2]의 Y1·A3 권고를 반영했는지 적는다.
4. K 꼭짓점은 M3-2 가드에 쓰였고 R2에서 보고된다. J1 매칭에는 쓰지 않는다(정답은 깊이표다).
5. 기준선의 성질([U3] §2-6)을 미리 알고 기준을 유지했다.

## 변경 이력

| 날짜 | 판 | 내용 |
|---|---|---|
| 2026-11-15 | v0.1 초안 | X3 초안(gpr-processing): [U3] 형식, 후보 공급원만 교체, 박스 → 꼭짓점 규칙(위쪽 띠·가운데 절반·포락선 피크·포물선 보간·중복 제거), 기준선 재계산 방식, 실질 지표 B − 기준선 평균, `eval_depth_4tu.py` 무수정 재사용 + 래퍼, 보고 항목, 구현 명세 J1-1–J1-4, [PLAN] `estimate_line_depths` 문구와의 차이(확인 필요) |
| 2026-11-15 | v0.2 초안 | cv 공동 검토(cv-detection-expert). §3 입력을 "파일의 모든 행"으로 바꿨다(러너가 conf > 0.25·PP를 이미 적용, 문턱 경계 이중 정의 방지). 중복 제거에 동률 규칙과 처리 순서를 넣었다. [PLAN] 문구 차이는 사용자 결정으로 확정했다(`eval_depth_4tu.py` 경로, `estimate_line_depths` 미사용). §1-1에 판정 → 보고 실행 순서와 1회 실행 거부를 넣었다. J1-2 단언 목록 7개(트리·blob, 입력 sha 이중 대조, R2 정체성: fold 0 가중치·r0·설정·`PPConfig`·HEAD, 박스 내용, 후보 sha 재확인, prefix 부재, stdout·logging 이중 차단)를 넣었다. 박스 → 꼭짓점 규칙, 기준선 재계산, 실질 지표는 그대로 받아들였다 |
| 2026-11-15 | v0.3 초안 | gpr-processing 재검토: 단계 1 탐색 창 위쪽 경계를 $\max(k_0-0.5P,\ k_z+0.5P)$로 묶고 $[0, n_s-1]$로 자르며, 창이 비면 `window_above_zero`로 기록한다. 시험 (e)를 추가했다. 무결성 반영: 제목 판 번호, 봉인 순서와 C2 자리 처리(`placeholder_scan` 허용 → C2 뒤 0), 단언 2의 `j1_inputs.folds["0"]` 키, 단언 4의 `line_id` 분해 규칙, 단언 6 시점(단계 1 이전). 판단 규칙은 바뀌지 않았다 |
