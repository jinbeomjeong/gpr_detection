# 명세 v1 — M2-8 4TU 깊이 검증 (준비 단계: 입력·꼭짓점 후보·박스·매칭·지표·기준선·출력)

- 작성: gpr-processing-expert, 2026-10-31 (spec 모드, 체크리스트 M2-8 준비)
- 받는 쪽: python-expert(`scripts/eval_depth_4tu.py` 구현·시험), 메인 세션(실행 승인)
- 기준 문서: `docs/processing/spec_depth_v1.md`(이하 [S], v1.7), `docs/processing/fourtu_explore.md`(이하 [FX]), `docs/datasets/readme_4tu.md`(이하 [R4])
- 상태: **준비 단계**. 깊이 실행·판정은 하지 않았다. 이 문서의 수치는 입력 목록과 꼭짓점 후보뿐이다(정답 깊이는 꼭짓점 후보 선정에 쓰지 않았다).
- 변경 이력: 2026-10-31 v1.2(커밋 전 정합) — 사용자 결정 5(실행 시점·점검·튜닝 없이 보고), 현장 11 진단 csv `depth_4tu_v1_site11.csv`, 기준선 MAE·판정률 A 열 추가와 전역 기준선 실제 열 이름, `multi_trench` 열, 정답 함수 실제 이름, CLI `--lines`·`--dry-run`, 시험 E1b·E5/E6 변형·E8, `n_peaks`. 2026-10-31 v1.1(구현 대조) — 정답 활동 93·정답 388행·무정답 32(08.1·013.6 깊이 전부 빈칸), `candidate_box`에 `tau_ns`, 기준선 시행 횟수 구분(주 1,000·보조 200), X 포함·제외를 한 번 실행으로, 현장 11 εr 8 진단은 주 결과 경로. 2026-10-31 용어 변경: 씨앗 → 꼭짓점 후보(영문 식별자 `apex_candidate`; 파일 `fourtu_seeds_v1*` → `fourtu_apex_candidates_v1*`, 열 `seed_*` → `candidate_*`. 난수 시드는 그대로)
- 사용자 결정(2026-10-31):
  1. 범위는 13개 현장 전체 활동이다.
  2. 매칭은 활동 단위 헝가리안이고 게이트가 없다. 누락(짝 없는 정답)과 오검출(짝 없는 추정)은 따로 센다. 지표는 MAE와 제186조 ±30 cm 판정률이다. 무작위 깊이 기준선(P8)을 같은 방식으로 돌린다. 상단/중심 두 가정과 t0 두 값(`t0_sample_est`, `t0_sample_act_median`)을 모두 보고한다. **4TU로 튜닝하지 않는다**(합성 기본값 그대로).
  3. 리포트는 `docs/processing/`에 md와 csv로 둔다(4TU CC0).
  4. (준비 결과 질문 1–4, 모두 권고안) **주 결과 = 평행 측선 반복 지지 묶음**(같은 활동 평행 측선 ≥2에서 깊이 ±0.10 m 반복; 지지 필터 없음·묶지 않음은 보조, 무작위 기준선은 같은 추정 수) · 정답 처리 §2 제안대로(빈 깊이 8행 제외, code X는 주 결과 제외·보조 포함, U 포함, 011.7 두 트렌치 합집합+표시) · 무작위 기준선 주 = 활동별 정답 범위 균등, 보조 = 전역 0.2–2.0 m · 정답 없는 활동 30개(v1.1: 32 = 정답표 없음 30 + 깊이 전부 빈칸 2)는 추정 개수만 보고(오검출로 세지 않음).
  5. (2026-10-31) 실행은 사용자가 다시 지시할 때 한다. 30분을 넘으면 15분 간격으로 점검한다. **결과가 나빠도 튜닝하지 않고 있는 그대로 보고·커밋한다.** 원인 분석은 gpr-processing-expert가 하고, 개선은 M2-9·후속 과제로 제안한다.

---

## 1. 입력

| 항목 | 출처 | 규모(2026-10-31 실측) |
|---|---|---|
| radargram | `io/fourtu.py` `load_fourtu_radargram(root, activity_id, path, T0Choice.BOTH)` | 활동 125, 측선 759, 트레이스 459,308. npy 누락 0. 측선 30 m 이상 50개(지그재그 다측선) |
| 측선 목록 | `radargrams_index.csv` → `docs/processing/fourtu_lines_v1.csv`(메타만) | 현장별 표는 §3-3 |
| 정답 깊이 | `utilities_groundtruth.csv` | 397행, 행이 있는 활동 95개. `depth_m` 빈칸 8행, code `X`(관로 아님) 1행(010.5). 빈칸을 빼면(§2) **정답 활동 93개·정답 388행**이다(08.1·013.6은 깊이가 전부 빈칸이라 빠짐; 2026-10-31 확인). 정답 없는 활동은 30 + 2 = **32개**다([R4] §5-3-4). 관경을 아는 정답 98행 |
| 관경 | `utilities_metadata_list.csv` `diameter_mm` | `정답표와 순서 일치 == 예`인 활동만 `seq`로 대응한다([FX] §4와 같음) |
| 꼭짓점 후보 | `docs/processing/fourtu_apex_candidates_v1.csv` | §3 |

- εr은 `eps_r_used`(메타)다. 실측 보증이 없다([S] §9). 현장별 값: 1·4·7·12·13은 9(헤더 기본값과 같음), 3·6은 11.11, 5는 8.16, 8은 12.17, 9는 14.06, 10은 9/12.17/13.06/16.89, 11은 15.18/17.36/19.46. 현장 2는 8.16/9다.
- `distance_m`(트렌치 수평거리)을 트레이스 위치로 쓰지 않는다([R4] §5-3-2). 깊이 집합끼리만 비교한다.

## 2. 정답 준비

1. `depth_m` 빈칸 행은 뺀다(8행).
2. code `X`(관로 아님)는 주 결과에서 빼고, 포함한 결과를 보조로 낸다. `U`(불명)는 포함한다. v1.1: 정답 집합만 바뀌므로 **한 번 실행 안에서** 매칭 단계에서 두 정답 집합(X 제외·포함)으로 모두 낸다(깊이 재계산 없음; `--include-code-x` 옵션 대신 summary의 `gt_set` 열). 해당은 010.5의 1행뿐이다.
3. `trench_no`가 둘인 활동(011.7)은 두 트렌치의 정답을 합집합으로 쓰고 `multi_trench` 표시를 단다(같은 관이 두 번 들어갈 수 있다). v1.2: 표시는 `depth_4tu_v1_matches.csv`의 `multi_trench` 열(bool, 활동 단위)과 md ⑦ 가정·한계에 나온다.
4. 기준점 두 가정:
   - 상단 가정: $d_\text{gt}=$ `depth_m`.
   - 중심 가정: $d_\text{gt}=$ `depth_m` $-R$, $R$ = `diameter_mm`/2000. 관경을 알 수 없는 행은 $R=0$(두 가정이 같다)이고, 행별 `radius_known`을 기록한다.
5. 정답이 없는 활동(32개 = 정답표 없음 30 + 깊이 전부 빈칸 2)은 매칭하지 않고 추정 개수만 보고한다(음성 보장 없음, [R4]).

## 3. 꼭짓점 후보 (`docs/processing/fourtu_apex_candidates_v1.csv`)

### 3-1. 선정 규칙 (정답을 보지 않음, 재현 가능)

스크립트: `analysis/processing/fourtu_apex_candidate_select.py`(단일 프로세스, BelowNormal, 759측선 약 5분). 측선마다 다음을 한다.

1. [S] 기본 설정으로 `prepare_depth_inputs`를 돌려 median 배경 제거 자료와 검출 영점 $t_z$를 얻는다.
2. 포락선 $E$에서 뮤트($t_z+1/f_c$) 아래 2 샘플부터 기록 끝 16 샘플 전까지를 본다.
3. **검출용 정규화**: 샘플별로 트레이스 방향 median을 한 주기로 평활하고, 그것으로 나눈 값에서 검출 피크를 찾는다. 이는 검출 갈래 gain에 해당한다. 시각은 항상 gain 없는 $E$에서 잡는다.
4. 검출 피크: `peak_local_max(min_distance=8, threshold_abs=2.0)`, 상위 300개. 측선 양 끝 0.2 m 안은 뺀다.
5. **능선 꼭짓점 이동**: 검출 피크에서 ±0.5 m 능선을 따라가며(더 이른 쪽으로 가는 것도 허용) 가장 이른 점으로 옮긴다. 포락선 최대는 날개에 있을 수 있기 때문이다.
6. 받아들이는 조건:
   - [S] §5-3 추적에서 양쪽 각각 3점 이상, 합계 9점 이상
   - ±0.10 m 안에서 가장 이른 점일 것(날개가 아님)
   - ±0.3 m 포물선의 $p_2>0$
   - 직선 광선 $v=\sqrt{2/(p_2\tau_a)}\in[0.04,0.20]$ m/ns(넓은 물리 범위, QC 전용, 깊이에 쓰지 않음)
   - ±0.4 m에서 V자(두 날개 교차) 모형이 쌍곡선보다 2배 이상 잘 맞으면 거부. 다만 실측상 이 조건으로 거부된 검출 피크는 없었다(§3-4).
7. 같은 측선에서 0.3 m·2 ns 안의 중복은 하나만 남긴다. 순위는 양쪽 날개 점 수의 작은 쪽, 다음이 검출 SNR이다. 측선당 최대 6개다.
8. [FX]의 수동 꼭짓점 후보 39개(01.1·04.1·010.14·011.14)는 그대로 덧붙인다(`source = manual_v0`, 사람이 고른 것). 가까운 자동 꼭짓점 후보 유무와, 수동 꼭짓점 후보에 같은 규칙을 적용한 결과를 `note`에 적는다.

### 3-2. 꼭짓점 후보 파일 열

`activity_id, site, path, trace, sample, t_ns, tau_ns, snr, n_left, n_right, v_straight_qc, source, note`. `trace`·`sample`은 L0 격자 인덱스이고 `sample`은 실수다. `tau_ns` = $t-t_z$(검출 영점)다. 측선별 결과(규칙, 영점, 검출 피크 수 `n_peaks`, 꼭짓점 후보 수, 꼭짓점 후보가 없는 사유)는 `fourtu_apex_candidate_lines_v1.csv`, 활동별 개수는 `fourtu_apex_candidates_v1_summary.csv`에 있다.

### 3-3. 개수 (2026-10-31)

| 현장 | 활동 | 측선 | 꼭짓점 후보 있는 측선 | 자동 꼭짓점 후보 |
|---|---|---|---|---|
| 1 | 9 | 92 | 88 | 458 |
| 2 | 7 | 74 | 70 | 307 |
| 3 | 9 | 104 | 103 | 455 |
| 4 | 8 | 28 | 28 | 138 |
| 5 | 10 | 27 | 27 | 123 |
| 6 | 6 | 64 | 63 | 230 |
| 7 | 4 | 29 | 29 | 120 |
| 8 | 8 | 37 | 37 | 200 |
| 9 | 9 | 53 | 53 | 295 |
| 10 | 24 | 125 | 123 | 601 |
| 11 | 17 | 81 | 81 | 444 |
| 12 | 8 | 31 | 30 | 129 |
| 13 | 6 | 14 | 14 | 72 |
| **합계** | **125** | **759** | **746** | **3,572** (+ 수동 39) |

- 꼭짓점 후보가 없는 측선은 13개다(활동 단위로는 모든 활동에 꼭짓점 후보가 있음).
  - 01.9 Path12·13·16·23, 02.6 Path3·9·15·19, 06.5 Path8, 010.5 Path4·6, 012.2 Path3: 검출 피크가 측선 끝·뮤트에 붙거나 날개 점이 부족해 거부됐다.
  - 03.9 Path5: 검출 피크 없음.
- 꼭짓점 후보가 있는 측선의 꼭짓점 후보 수 중앙값은 6이다(상한에 걸림). 자동 꼭짓점 후보 $\tau$의 5–95 %는 3.7–33.5 ns다.
- 영점 규칙: `air_single_event` 709측선, **`air_surface_late` 50측선**(현장 9가 33개, 7이 5개, 나머지 1–3개). [FX] 4개 활동의 41/41 단일 사건과 달리 현장 9는 두 사건으로 갈린다. 실행 단계에서 $h_\text{est}$와 2a QC를 따로 보고한다.

### 3-4. 꼭짓점 후보 품질 (정답 없이 볼 수 있는 것만)

- **수동 꼭짓점 후보 재현율**: 39개 중 20개는 자동 꼭짓점 후보가 12 트레이스·15 샘플 안에 있었다. 나머지 19개 중 9개는 같은 규칙에 통과하지만 검출 피크 단계에서 빠졌고, 1개는 측선당 상한에 밀렸다. 나머지는 날개 부족(`few_picks`)·볼록 아님·v 범위 밖·꼭짓점 아님으로 거부됐다. 010.14(클러터 많음)는 0/7이었다.
- **눈 점검**(`fourtu_apex_candidates_v1_pilot.png`, `fourtu_apex_candidates_v1_overview_a.png`, `_b.png`: 현장마다 첫 측선, 표시용 AGC 영상)에서 확인한 점:
  - 뚜렷한 쌍곡선 꼭짓점(01.1, 03.1, 013.1 등)은 잡는다.
  - 기울어진 층의 굴곡점, 두 날개의 교차점, 수평 층의 요철에도 꼭짓점 후보가 놓인다(07.1, 09.1, 012.1).
  - **정밀도는 측정하지 않았다.** 대략 절반 수준으로 보인다(**가정**, 눈 점검 13측선).
- V자 판별 시험은 4TU의 얕고 뾰족한 실제 꼭짓점도 V로 판정했다(1배 비교 시 수동 꼭짓점 후보 14개 거부). 그래서 2배로 완화했고, 결과적으로 효과가 없다.
- 대책은 실행 단계에서 정한다(§4-3 지지 측선 수 필터, 확인 필요 질문 1).

## 4. 박스 공급과 깊이 실행

### 4-1. 꼭짓점 후보 → `GridBox` ([S] §11-2 합성 규칙과 같은 모양)

꼭짓점 후보 $(i_s,k_s)$, $\Delta t$, $\Delta x$, $f_c$ 500 MHz, $v_m=c/\sqrt{\varepsilon_{r,\text{used}}}$(메타)를 쓴다.

- 트레이스: $i_s\pm\lceil0.8/\Delta x\rceil$(40 트레이스)를 측선 안으로 자른다.
- 위: $k_0=\lfloor k_s-1.5/(f_c\Delta t)\rfloor$.
- 아래: $k_1=\lceil k_s+\big(\tfrac{2}{v_m}(\sqrt{d_s^2+0.8^2}-d_s)+1.5/f_c\big)/\Delta t\rceil$, $d_s=v_m\tau_s/2$. 기록 끝에서 자른다.
  - 메타 v로 날개 끝 시각을 잡는다. 메타는 입력이고 정답이 아니다.
- 합성 규칙과 다른 점: 정답 광선 시각 대신 메타 v 쌍곡선을 쓴다.

### 4-2. `estimate_depth` 설정

- 측선마다 `prepare_depth_inputs(radargram, DepthConfig())`를 한 번 한다(띠 박스가 없으므로 `exclude_traces=None`). 꼭짓점 후보마다 `estimate_depth(radargram, box, DepthConfig(), inputs=...)`를 부른다. **설정은 전부 기본값이고 4TU로 바꾸지 않는다.**
- 공기결합이므로 v는 `meta_fixed`이고 곡률 피팅은 금지다([S] §7 순서 2). 영점은 [S] §4-2 규칙을 따른다(대부분 `air_single_event` = 띠 포락선 피크, 현장 9 등은 `air_surface_late`).
- 2a 굴절 QC는 `air_surface_late` 측선에서 자동으로 돈다. 구간과 `meta_eps_suspect`를 보고만 하고 깊이에는 쓰지 않는다.
- t0 세 가지(보고 대상):
  - 주 결과 `depth_top_m`(검출 영점)
  - `alt_depths["t0_sample_est"]`, `alt_depths["t0_sample_act_median"]`(같은 꼭짓점·같은 v, 메타 영점). 라벨은 [S] §4-3(`merged` 등)
- 꼭짓점 후보 QC: 결과 `apex_trace`가 꼭짓점 후보에서 3 트레이스보다 멀거나 `apex_sample`이 $0.5/(f_c\Delta t)$ 샘플보다 멀면 `candidate_moved` 표시를 단다(제외하지 않고 개수만 보고).
- `status != "ok"`인 꼭짓점 후보는 깊이가 없다. 상태별 개수를 보고한다.

### 4-3. 활동 단위 묶기 (측선 간 같은 관)

측선은 대개 관로와 직각이라, 한 활동의 측선들에 같은 관이 비슷한 깊이로 반복해서 나온다([R4] §5-3-2).

1. 활동 안의 `ok` 깊이를 정렬하고, 이웃 차가 ≤ 0.10 m이면 같은 묶음으로 한다(단일 연결, [FX] P8). 묶음 깊이는 구성원의 중앙값이다. 측선 수 `n_lines`와 꼭짓점 후보 수 `n_apex_candidates`를 기록한다.
2. **지지 필터**(주 결과, 확인 필요 질문 1): 측선이 2개 이상인 활동에서는 `n_lines ≥ 2`인 묶음만 추정으로 쓴다. 측선이 1개인 활동은 모두 쓴다. 근거는 실제 관은 평행 측선에 반복되고, 층 굴곡·교차점 같은 거짓 꼭짓점 후보는 대개 한 측선에만 있다는 점이다(**가정**).
3. 보조 결과:
   - (a) 지지 필터 없음
   - (b) 묶지 않음(꼭짓점 후보 깊이 전부)
   - 묶기는 깊이만으로 하므로 0.10 m 안의 다른 관 두 개는 하나로 합쳐진다. 정답에 둘이 있으면 하나가 누락된다.
4. t0 세 가지와 기준점 두 가정마다 같은 묶기를 따로 한다.

## 5. 매칭과 지표

- 활동마다 추정 묶음 $\{\hat d_a\}$와 정답 $\{d_b\}$를 놓고 비용 $|\hat d_a-d_b|$로 `scipy.optimize.linear_sum_assignment`를 돌린다(직사각 허용, **게이트 없음**). 짝 수는 $\min(n_\text{est},n_\text{gt})$다.
- 지표(활동을 모은 쌍 기준이 주 결과, 현장별·활동별도 냄):

| 기호 | 정의 | 단위 |
|---|---|---|
| $e$ | $\hat d-d_\text{gt}$ (짝마다) | m |
| MAE | $\overline{|e|}$ | m |
| bias | $\bar e$ | m |
| med, P90 | $|e|$의 중앙값, 90 % 분위 | m |
| 판정률 A (짝 기준) | $\#\{|e|\le0.30\}/\#\text{짝}$ — 제186조 ±30 cm | – |
| 판정률 B (정답 기준) | $\#\{|e|\le0.30\}/\#\text{정답}$ (누락을 실패로 셈) | – |
| 누락 | $n_\text{gt}-\#\text{짝}$ | 개 |
| 오검출 | $n_\text{est}-\#\text{짝}$ | 개 |

- 판정 문구는 "합격률 n/N"이다(정의서 verify 규칙). 판정률 A·B와 n/N을 함께 쓴다.
- 보고 조합: {상단, 중심} × {검출 영점, est, act} × {주 결과(지지 필터), 보조 a, b}. 표는 주 결과를 먼저 보이고 나머지는 csv에 둔다.

## 6. 무작위 깊이 기준선 (P8)

- 활동마다 추정 묶음 수 $n_\text{est}$(주 결과와 같은 수)만큼 깊이를 균등분포 $U[\min d_\text{gt},\ \max d_\text{gt}]$에서 뽑는다. 정답이 1개면 $U[d-0.3,\ d+0.3]$이고 0 아래는 0으로 자른다.
  - 정답 범위를 아는 기준선이므로 기준선 쪽에 유리하다. 이를 이기면 의미가 있다.
- 같은 헝가리안·지표를 적용한다. `numpy.random.default_rng(0)`.
- 시행 횟수(v1.1, 비용 구분):
  - **주 결과 조합**(지지 필터 묶음 × 상단/중심 × 영점 3 = 6조합): 활동 범위 기준선 **1,000회**. 현장별 값은 같은 시행에서 현장별로 집계한다(추가 시행 없음).
  - 보조 조합(지지 필터 없음, 묶지 않음): 200회.
  - 전역 범위 보조 기준선: 주 결과 조합에만 200회, 전체와 현장별.
  - 시행 횟수를 md 실행 조건에 적는다.
- 보고: 기준선 MAE·판정률 A·B의 평균과 2.5/97.5 % 분위, 그리고 $p$ = (기준선 판정률 B ≥ 관측 판정률 B인 시행 비율). v1.2: 활동 범위 기준선은 세 지표를 모두 csv 열로 낸다(§8). 이미 계산되는 값이고, 게이트 없는 매칭의 관대함은 MAE·판정률 A에서도 드러나기 때문이다.
- 보조 기준선: 전역 범위 $U[0.2,\ 2.0]$ m(현장 공통, **가정**). 비용 때문에 판정률 B의 평균·분위와 $p$만 낸다(v1.2).

## 7. 별도 보고

- **현장 11**(dt 0.195 ns, 앞 약 48 샘플 뮤트, εr 15.18/17.36/19.46): [FX]에서 꼭짓점 깊이가 정답보다 약 0.22 m 얕았고 메타 εr 과대 후보였다. 현장 11은 표를 따로 내고 bias 부호를 본다. 진단 전용으로 "εr 8로 바꾼 깊이"(×$\sqrt{\varepsilon_\text{used}/8}$)도 함께 내되, **판정과 기본값에는 쓰지 않는다**(튜닝 금지). v1.1: 진단은 주 결과와 같은 경로로 한다(꼭짓점 후보 깊이를 먼저 배율 조정 → §4-3 묶기·지지 필터 → 매칭). 그래야 현장 11 주 결과 표와 바로 비교된다. 묶지 않은 결과는 보조 열로만 낸다.
- **현장 9**(`air_surface_late` 33측선, dt 0.098/0.117/0.195 혼재): $h_\text{est}$ 분포, 2a 구간, `meta_eps_suspect` 개수를 보고한다.
- εr 메타가 헤더 기본값과 같은 현장(1·4·7·12·13, 2의 85 %): [FX] P5 `eps_meta_equals_header_default` 후보로 표시만 한다.
- 측선 30 m 이상(50개): 결과 csv에 `long_line` 표시만 한다(깊이에는 영향 없음, x 미사용).

## 8. 출력

`docs/processing/` 아래에 둔다(4TU CC0 → 파일명·좌표 공개 가능).

| 파일 | 행 | 열 |
|---|---|---|
| `depth_4tu_v1_candidates.csv` | 꼭짓점 후보 | `activity_id, site, path, candidate_trace, candidate_sample, source, status, depth_top_m, depth_est_m, depth_act_m, label_est, label_act, eps_r_used, zero_rule, t_zero_ns, h_est_m, apex_trace, apex_sample, candidate_moved, eps_qc_lo, eps_qc_hi, flags`. `flags`에 `candidate_tau_missing`(수동 꼭짓점 후보의 `tau_ns` 빈칸 → `zero_failed`, v1.1) |
| `depth_4tu_v1_clusters.csv` | 활동 × t0 × 묶음 | `activity_id, site, zero_variant, cluster_id, depth_m, n_lines, n_apex_candidates, kept_by_support` |
| `depth_4tu_v1_matches.csv` | 짝 | `activity_id, site, gt_set, multi_trench, assumption, zero_variant, variant, est_depth_m, gt_depth_m, gt_code, radius_known, error_m, within_30cm` |
| `depth_4tu_v1_summary.csv` | 조합 × (전체·현장) | `scope, gt_set, assumption, zero_variant, variant, n_activities, n_gt, n_est, n_pairs, mae_m, bias_m, med_abs_m, p90_abs_m, pass_a, pass_b, missed, false_est, baseline_mae_mean, baseline_mae_p025, baseline_mae_p975, baseline_pass_a_mean, baseline_pass_a_p025, baseline_pass_a_p975, baseline_pass_b_mean, baseline_pass_b_p025, baseline_pass_b_p975, baseline_p, baseline_global_pass_b_mean, baseline_global_pass_b_p025, baseline_global_pass_b_p975, baseline_global_p`. `baseline_mae_*`·`baseline_pass_a_*`는 v1.2 추가, `baseline_global_*`는 주 결과 조합에만 채운다(v1.1) |
| `depth_4tu_v1_site11.csv`(v1.2) | 현장 11 진단 짝 | `activity_id, gt_set, assumption, zero_variant, variant(main_path / unclustered), est_depth_m, est_depth_eps8_m, gt_depth_m, error_eps8_m, within_30cm_eps8`. 진단 전용이고 판정에 쓰지 않는다(§7). md에는 요약 표만 둔다 |
| dry-run JSON(v1.1, 파일 아님·stdout) | – | 위 개수에 더해 `activities_without_gt`(32), `gt_items_include_x`(389), `candidates_without_tau` |
| `depth_4tu_v1.md` | – | ① 실행 조건(커밋, 설정, 시드) ② 주 결과 표(상단/중심 × 영점 3) ③ 기준선 비교 ④ 현장별 표 ⑤ 현장 11·9 별도 ⑥ 상태·`candidate_moved`·꼭짓점 후보 출처(auto/manual)별 개수 ⑦ 가정·한계 |

## 9. python-expert 구현 명세

### 9-1. `scripts/eval_depth_4tu.py`

```text
python -I scripts/eval_depth_4tu.py
    --root  <4TU package root>                    (default io.paths 4TU root)
    --candidates docs/processing/fourtu_apex_candidates_v1.csv
    --lines docs/processing/fourtu_lines_v1.csv
    --out-dir docs/processing  --prefix depth_4tu_v1
    [--dry-run]                                   (counts only, JSON to stdout, no depth run)
    [--activities 01.1,04.1] [--cluster-gap-m 0.10] [--min-support-lines 2]
    [--baseline-trials 1000] [--baseline-trials-aux 200] [--baseline-seed 0]
```

함수(스크립트 안, `src/` 변경 없음):
- `candidate_box(radargram, trace, sample, tau_ns) -> GridBox` — §4-1. $\tau_s$는 꼭짓점 후보 csv의 `tau_ns`(같은 기본 설정의 검출 영점 기준)를 쓴다. 영점 규칙이 바뀌면 후보 csv를 다시 만든다. 수동 꼭짓점 후보는 `tau_ns`가 영점 실패로 비어 있으면 박스를 만들지 않고 `status="zero_failed"`로 센다(v1.1 확인)
- `run_line(radargram, candidates) -> list[CandidateResult]` — 측선당 `prepare_depth_inputs` 한 번, 꼭짓점 후보별 `estimate_depth`, §4-2 `candidate_moved`
- 정답(§2, 빈칸·X·두 트렌치·R; v1.2 실제 이름): `load_ground_truth_sets(root)`(두 `gt_set`을 한 번에), `build_ground_truth(..., include_code_x)`(한 집합), `build_ground_truth_sets(...)`(두 집합 묶음)
- `cluster_depths(depths, lines, gap_m) -> list[Cluster]` — §4-3 단일 연결, 중앙값, `n_lines`
- `match_activity(est, gt) -> MatchResult` — 헝가리안, 누락·오검출
- `summarize(matches, ...)` — §5 지표
- `random_baseline(n_est_by_act, gt_by_act, trials, seed, mode) -> BaselineStats` — §6
- 출력 §8. 난수는 기준선에만 쓰고, 시드를 md에 적는다.

### 9-2. 단위 시험 (`tests/test_eval_depth_4tu.py`, 합성·작은 표만, 4TU 원본 없이)

| ID | 내용 | 기준 |
|---|---|---|
| E1 | 해석 B-scan([S] §11-1 생성기, 공기결합 500 MHz h 0.05 m, εr 9, d 0.6)에서 정답 꼭짓점을 꼭짓점 후보로 `candidate_box` → `estimate_depth` | `status=="ok"`, `candidate_moved` 없음, 깊이 ±3 cm |
| E2 | `cluster_depths([0.50, 0.55, 0.62, 1.00], lines=[1,2,3,1], gap=0.10)` | 묶음 2개(0.55: n_lines 3, 1.00: n_lines 1). 지지 필터 2이면 1개 |
| E3 | `match_activity(est=[0.5, 1.0, 1.6], gt=[0.55, 1.05])` | 짝 2, 오검출 1, 누락 0, 오차 −0.05·−0.05. est와 gt를 바꾸면 누락 1 |
| E4 | 게이트 없음 확인: est=[3.0], gt=[0.5] | 짝 1, $|e|$ 2.5, within False |
| E5 | `random_baseline` 같은 시드 두 번 | 결과 동일. 정답 1개 활동의 범위가 [d−0.3, d+0.3], 0에서 잘림 |
| E6 | 작은 정답 표 고정 데이터(빈 깊이·X·두 트렌치·순서 일치 예/아니오) | 빈칸 제외, X 기본 제외·옵션 포함, 두 트렌치 합집합 + 표시, R은 "예"만 |
| E7 | 판정률 정의: 짝 3(|e| 0.1, 0.3, 0.31), 정답 4 | A = 2/3, B = 2/4(경계 0.30은 포함) |
| E1b (v1.2) | 수동 꼭짓점 후보의 `tau_ns` 빈칸 | 박스 없음, `status=="zero_failed"`, flag `candidate_tau_missing` |
| E5 변형 (v1.2) | 주 1,000회·보조 200회, 같은 시드 두 번 | 두 횟수 모두 재현, 보조 기준선은 판정률 B·p만 |
| E6 변형 (v1.2) | 같은 고정 표에서 `build_ground_truth_sets` | `gt_set` 두 개(X 제외·포함) 개수 차 = X 행 수, 두 트렌치 활동 `multi_trench` True |
| E8 (v1.2) | 현장 11 진단: 작은 고정 입력에서 εr 배율 → 묶기·지지 필터 → 매칭 | 배율 $\sqrt{\varepsilon_\text{used}/8}$ 적용 확인, `main_path`·`unclustered` 두 variant 행, 주 결과 지표에는 들어가지 않음 |

### 9-3. 실행 비용

꼭짓점 후보 3,611개 × `estimate_depth`(측선당 전처리 1회)는 꼭짓점 후보 선정 실측(약 5분)과 같은 규모에 피팅이 더해진다. 단일 프로세스로 **약 10–20분으로 추정**한다(가정). 30분을 넘으면 메인 세션 규칙(BelowNormal, 장시간 점검)을 따른다.

## 10. 가정과 미결

| 항목 | 처리 | 확인 |
|---|---|---|
| 자동 꼭짓점 후보 정밀도 | 측정 안 함, 눈 점검상 절반 수준 추정 | 확인 필요 질문 1, 실행 뒤 묶음 지지 분포 |
| 지지 필터(측선 ≥ 2) | 주 결과로 제안(**가정**) | 보조 a와 비교 |
| 묶기 간격 0.10 m | [FX] P8 그대로(**가정**) | 0.05·0.15 민감도는 보조로만 |
| 무작위 기준선 범위 | 활동 정답 범위(유리한 기준선) | 보조 전역 범위 |
| 정답 기준점 | 상단·중심 둘 다 | – |
| 메타 εr | 실측 보증 없음 | 현장 11 별도, 2a(현장 9) |
| 박스 반폭 0.8 m, 위·아래 1.5 주기 | 합성 규칙과 같음(**가정**) | – |
