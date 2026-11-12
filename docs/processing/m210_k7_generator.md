# M2-10 K7 생성기 결함 — 원인 정리와 판정 명세 (확정판, 미봉인)

- 판: **v1.2 확정판(재봉인 대상, 미봉인)**, 2026-11-12, gpr-processing-expert. v1.2 = 첫 봉인 `d6aaf0f`(v1.1) 실행이 내보내기에서 멈춰 대체됨(판정 결과 없음, §6-2·§9). 내보내기의 관별 속도 점검(판정 입력 아님) 처리 변경, run1 h5 배열 비교·대체 이력의 공개 항목 추가, 봉인 대상 blob 변경(`scripts/synthetic/export_dataset.py`·시험). 기준·문턱·결정표·모델·시드는 v1.1과 같다. v1.1 = 봉인 전 구현·EM 산출 반영(실제 파일 이름, `K7_SEALED_FILES`, 구현 해석 (a)–(i)(§4-4 목록), 사건 붙임 `attach_k7_event`, 공유 참조 h5·점검 영점, 잡음 없는 행 수 정정, 상속 봉인 파일 변경 공개, 실행 전 감사·무결성 지적 반영; §4-2–§4-6·§5·§6·§9·§10, 기준·문턱·결정표 순위는 불변). v1.0 = 사용자 결정 L1–L4 반영(§0), K7-c 진단 전용, 봉인 문구·봉인 대상·운영 순서·판정서 첫머리 목록·빈 경우와 고정 집합 단언 추가(§4-6·§6–§9). v0.1 = 원인 정리·후보·초안(질문 Q1–Q5).
- 범위: 4TU 꼭짓점 후보 생성기(`analysis/processing/fourtu_apex_candidate_select.py`, 래퍼 `apex_candidates_for_radargram`, HEAD blob `7cc2bf2b`)의 후보 손실 기작 세 가지(K7-a·b·c)다. 정규화 바닥(M2-10 (b))은 현행 규칙 유지로 닫혔다(사용자 결정 2026-11-12, [`m210b_v3_judgment.md`](m210b_v3_judgment.md) §9). 이 문서는 바닥 규칙을 다시 다루지 않는다.
- 자료 규율
  - 이 문서의 수치는 모두 **이미 본 자료**(S1g gprMax, M2-10 (b) 묶음 1·v2·v3, 4TU)이거나 순수 함수 몬테카를로다. 개발 근거일 뿐이고 채택 판정에 쓰지 않는다.
  - 4TU는 탐색 보고만 한다. 4TU 값으로 후보·상수를 고르지 않는다.
  - 판정은 §4의 새 gprMax 실현으로만 하고, 기준(§4-5·§4-6)은 그 실행 **전에** 봉인한다(§7). 봉인 뒤 결과를 보고 기준을 바꾸지 않는다.
  - S1g·S4S6(묶음 1·v2·v3)의 h5·npy·시드·runs·판정 결과는 K7 판정에 쓰지 않는다. 개발·회귀 고정물로만 쓴다(§3).
  - **참조 h5 예외**: 자유 공간 `data/synthetic_gprmax_s4s6/h5/FS.h5`(S4/S6 묶음 1)는 영점 기준(배경 − FS)에 다시 쓴다. 같은 영역·안테나라 K7 사본은 같은 FDTD 입력이 되기 때문이다. 관이 없는 자료라 결함 판정의 결과를 담지 않는다. snapshot 대상에 넣는다(§6-1).
- 근거 스크립트(gpr-dev, 1 프로세스, BelowNormal, 난수 시드 20261112 계열)
  - [`analysis/processing/m210_k7_probe.py`](../../analysis/processing/m210_k7_probe.py) → [`m210_k7_probe.json`](m210_k7_probe.json)
  - [`analysis/processing/m210_k7_rank_replay.py`](../../analysis/processing/m210_k7_rank_replay.py) → [`m210_k7_rank_replay.json`](m210_k7_rank_replay.json)
  - 두 스크립트 모두 생성기 파일을 고치지 않는다. 프로세스 안에서 `crest_apex`·`_rank_and_suppress`·상한 상수만 바꿔 끼워 변형을 재현한다(시제품).

## 0. 사용자 결정 (확정 2026-11-12, K7 판정 실현 결과 전)

| ID | 결정 | 확정 내용 | 반영 위치 |
|---|---|---|---|
| L1 | K7-a 고침 | **a1 시작점 오르막**: 시작 샘플을 원 포락선 $E$의 국소 최대까지 올린다. a2(NMS 승자를 원 $E$로)와 엽 병합은 불채택 | §2-1, §5 |
| L2 | K7-b 고침 | **b2 순위 키 = 검출 SNR 하나, 측선당 상한 6 유지**, a1 위에서만. 판정 순서 a1 → a1b2. b1(상한 제거)은 판정하지 않고 후보 수 비용만 보고(`a1b1`) | §2-2, §4-4, §4-6, §5 |
| L3 | K7-c | **이번 판은 고치지 않고 진단만 출력**(끝점 옆 오차, 끝점 $E$ 국소 최대 여부, $\tau^2$ 꼭짓점 위치). c1·c1-post는 판정하지 않는다. v0.1의 c 문턱 0.8·0.95는 **쓰지 않는다** | §2-3, §4-4, §4-5, §5 |
| L4 | 판정 실현 | **새 gprMax 8모델 + 조건당 잡음 시드 40**. 기존 S1g·S4S6 h5는 판정에 쓰지 않는다 | §3, §4-3, §4-4 |

- v0.1 질문과의 대응: Q1 → (1) a + b 판정, c 진단만(L1–L3). Q2 → (1) a1(L1). Q3 → (1) b2(L2). Q4 → (1) 진단만(L3). Q5 → (1) 새 gprMax 8모델 + 시드 40(L4). 모두 v0.1 권고안이다.
- 결정 근거(이미 본 자료, 판정 무관): a1은 S1g D100 두 관을 되살리고 4TU kept의 약 1.5 %만 옮긴다. b2는 a1 위에서 S1g 세 모델을 되살리고 kept 수를 늘리지 않지만 4TU kept의 약 32 %를 바꾼다. c1은 4TU kept의 약 40 %를 가리지 않고 바꾼다(§1·§2).
- 이 결정은 K7 새 실현의 결과가 하나도 없을 때 내렸다. 결정 때 본 자료는 §1–§2의 이미 본 자료와 순수 함수 MC뿐이다(§9).
## 1. 결함 기작과 증거 (모두 이미 본 자료)

### 1-0. 생성기 단계와 기호

| 기호 | 뜻 | 단위 |
|---|---|---|
| $E(k,j)$ | 배경 제거 분기의 포락선(샘플 $k$, 트레이스 $j$) | 진폭 |
| $\ell(k)$ | 탐색 구간 행 중앙값(트레이스 방향)을 한 주기로 평활하고 바닥 $F$로 자른 값 | 진폭 |
| $D(k,j)=E/\ell$ | 검출 정규화 맵(검출에만 쓰는 이득) | 무차원 |
| $f_c$ | 중심 주파수(4TU·합성 500 MHz, $1/f_c$ = 2 ns) | Hz |
| $n_l, n_r$ | `accept`의 왼·오른 날개 추적 픽 수(최대 $A_\text{max}/\Delta x$ = 50) | 개 |
| $t_a$ | 잡음 없는 처리 꼭짓점 시각(json `envelope_peak_ns`) | ns |

생성기 순서: $D$의 국소 최대(`peak_local_max`, SNR ≥ 2) → 시작점 정밀화 `parabolic_peak(E[:, j0], k)`(±0.5 샘플) → `crest_apex`(±0.5 m 능선을 ±1 ns 창으로 걸으며 **가장 이른 점**) → `accept`(날개·꼭짓점·곡률·V자 검사) → 순위 $(-\min(n_l,n_r))$, −SNR → NMS(0.3 m·2 ns) → 측선당 상한 6.

### 1-1. K7-a: 검출 극대가 원 포락선 극대가 아니다 (시작점 위치 결함)

- **기작**: $D=E/\ell$의 시간 방향 극대는 $\partial_k\ln E=\partial_k\ln\ell$인 곳이다. $\ell$이 꼭짓점 시각 근처에서 시간에 따라 커지면 $D$의 극대는 $E$의 오르는 날(flank)로 앞당겨진다. 시작점 정밀화는 ±0.5 샘플만 움직여 $E$의 극대까지 가지 못한다. 그 뒤 `crest_apex`는 "가장 이른 점"을 고르므로, 이웃 트레이스의 능선 점(모두 더 늦음)보다 이른 이 시작점을 그대로 남긴다. 순위의 SNR은 $D$ 값이라 낮은 $\ell$ 행에 놓인 이 점이 이긴다. 코드 주석("Pick times always come from the un-normalised envelope E")의 의도와 어긋난다.
- **증거(S1g, [`m210_k7_probe.json`](m210_k7_probe.json) `A_s1g`)**

| 모델·관 | 남은 후보(시각, ns) | 원 $E$ / 꼭짓점 $E$ | $\ell$(함의) | SNR | $E$ 국소 최대? | 능선 이동 | 결과 |
|---|---|---|---|---|---|---|---|
| D100 관 55·105 | 14.547(꼭짓점 −1.021) | 22.61 / 52.39 = 0.43 | 1.33 | 17.0 | **아님** | 없음(시작 = 끝) | kept → 깊이 `no_apex` |
| D100 같은 관 | 15.556(꼭짓점) | 52.39 | 8.73 | 6.0 | 맞음 | 트레이스 30·80, 18.27 ns에서 이동 | NMS로 빠짐 |
| C1 관 80 | 15.290(−0.277) | 49.50 / 52.40 = 0.94 | 0.117 | 423.9 | **아님** | 없음 | 13.344에 NMS(Δt 1.946 ≤ 2) |
| C1 관 80 | 13.344(−2.223) | 1.64 / 52.40 = 0.03 | 0.059 | 27.9 | **아님** | 없음 | kept(min 날개 50) |
| C1 관 80 | 15.585(트레이스 77) | 52.01 | 0.134 | 389.0 | 맞음 | 80, 17.74 ns에서 이동 | kept(Δt 2.24 > 2, 여유 0.24 ns) |
| D060 관 65·95 | 14.948(−0.624) | 38.89 / 52.81 = 0.74 | 0.70 | 55.3 | **아님** | 없음 | cap6 |

- 꼭짓점 트레이스 포락선의 국소 최대는 $[t_a-3,\,t_a+1.5]$ ns에서 하나뿐이다(D100 15.556 ns, C1 15.549 ns, D060 15.565 ns; 상대 크기 ≥ 0.05 기준). 곧 14.547 ns는 **엽이 아니다**.
- **정정**: [`m210_k1_s1g_judgment.md`](m210_k1_s1g_judgment.md) §3-1과 M2-10 (b) v1.4 §4-5 (a)는 이 점을 "이른 엽"이라 불렀다. 실제로는 $D$ 정규화가 만든 오르는 날 위 시작점이다. K1 판정 결과(순위 6 기각)와 (b) 판정은 바뀌지 않는다. 바뀌는 것은 기작 이름과 고칠 곳뿐이다(NMS가 아니라 시작점).
- (b) v1.4 P2의 "이른 엽 상자" $[t_{a0}-1.5/f_c,\,t_{a0}-0.5/f_c)$에 D100 점은 여유 0.009 ns로 겨우 든다. 그래서 상자 기반 고침(엽 병합)은 권하지 않는다.
- **4TU 범위(탐색, [`m210_k7_probe.json`](m210_k7_probe.json) `D_variants_4tu`)**: 현행 kept 2,629개 중 끝점이 $E$의 시간 국소 최대가 아닌 것이 66개(2.5 %)다. a1은 550측선 중 80측선의 kept를 바꾸고, ±3 트레이스·±1 ns 밖으로 바뀌는 것은 37·40개다. NMS 승자가 "이른 엽 상자" 안인 패턴은 7개(4측선)뿐이다(`B_4tu_replay`).
- `ℓ`이 그 행에서 왜 크게 오르는지(어떤 사건이 측선 전체로 퍼졌는지)는 이 문서에서 확정하지 않는다(가정 표시). 기작은 $\ell$의 기울기만 있으면 생기므로, 원인 사건과 무관하게 시험할 수 있다(§4-1 T2).

### 1-2. K7-b: 순위 키 $\min(n_l,n_r)$와 측선당 상한 6

- **기작**: 날개 픽 수는 최대 50에서 포화한다. 평평하고 긴 사건(층, 다중 반사, 측선 전체로 퍼진 약한 띠)은 양쪽 날개가 50이 된다. 이웃 관이 안쪽 날개를 끊은 진짜 꼭짓점(D060 min 9)은 순위가 밀린다. 수락 후보가 6개를 넘는 측선에서는 상한이 그 꼭짓점을 자른다.
- **증거**
  - S1g D060: primary 수락 7개(관 65·95, min 날개 9, 순위 36–42) 모두 `cap6`. 남은 6개는 min 날개 37·34·22·22·20·17인 다중 반사·깊은 사건이다. 원 $E$는 0.42–10.3으로 primary(52.8)의 1–20 %다([`m210_k7_rank_replay.json`](m210_k7_rank_replay.json) `s1g.S1G_D060.cur`).
  - S1g C1: 원 $E$ 1.64(꼭짓점의 3 %)인 13.344 ns 띠(날개 50/50)가 순위 0이다.
  - 4TU(탐색, 541측선, `B_4tu_replay`): 수락 8,914개 중 kept 2,629개. 상한 손실 4,988개(285측선), NMS 손실 1,297개. 오프라인 재생이 541/541측선에서 생성기 kept와 같다. m210d i-gen2(순위·NMS·상한 6에서 잘린 정답) 4개([`m210d_event_selection_diagnosis.md`](m210d_event_selection_diagnosis.md) §2 표).
  - M2-9 C 블라인드 검토: 자동 후보 130개 중 층 48개([`m29c_apex_review.md`](m29c_apex_review.md)) — 긴 날개를 우대하는 순위와 같은 방향의 관찰이다.

### 1-3. K7-c: 능선 이동의 "가장 이른 점" 규칙과 평평한 꼭짓점

- **기작**: 깊은 꼭짓점은 옆으로 평평하다. 왕복 시각 차는 $\Delta t(x)=\frac{2}{v}\left(\sqrt{d^2+x^2}-d\right)\approx\frac{x^2}{v\,d}$이다. $d$ 1.5 m, $v$ 0.1 m/ns, ±3 트레이스(±0.06 m)에서 0.024 ns로, 잡음이 흔드는 픽 시각보다 작다. ±0.5 m 능선 점 가운데 "가장 이른 점"은 잡음의 최솟값을 따라 옆으로 간다.
- **증거**
  - M2-10 (b) v2 진단(실현 7개, S4N_D15_S045 ρ 20): B_full 끝점 트레이스 84·86·74·75·85(꼭짓점 80), ±3 창 밖([`m210b_v2_judgment.md`](m210b_v2_judgment.md) §6).
  - v3 F3-S: current 성공률이 시작점 수에 기대 있음(0.90 → 강한 시작점만 0.70, H-C, [`m210b_v3_judgment.md`](m210b_v3_judgment.md) §6). 가장 이른 점 규칙이 시작점마다 다른 끝점을 내므로, 시작점이 많을수록 창 안에 끝날 기회가 늘어난다.
  - kept 단계 손실 `other_nms`: v2 current 68행·B_full 4행([`m210b_v2_judgment.md`](m210b_v2_judgment.md) §7), v3 current 204행·B_full 16행([`m210b_v3_judgment.md`](m210b_v3_judgment.md) §7).
  - 순수 함수 몬테카를로(`C_crest_monte_carlo`, 조건당 200회, Ricker 500 MHz, 대역 통과 백색 잡음, 시작 트레이스 꼭짓점 ±10 무작위, $\rho=A_0/\hat\sigma_u$):

| $d$ (m) | ρ | ±3 트레이스 거리 $\Delta t$ (ns) | 가장 이른 점: ±3 안 비율 / 중앙 / 90 % | $\tau^2$ 꼭짓점: ±3 안 비율 / 중앙 / 90 % |
|---|---|---|---|---|
| 0.6 | 10 | 0.060 | 0.915 / 2 / 3 | 1.000 / 0 / 0 |
| 0.6 | 20 | 0.060 | 0.990 / 1 / 2 | 1.000 / 0 / 0 |
| 1.0 | 10 | 0.036 | 0.805 / 2 / 4 | 1.000 / 0 / 1 |
| 1.0 | 20 | 0.036 | 0.950 / 1 / 3 | 1.000 / 0 / 0 |
| 1.5 | 10 | 0.024 | **0.680** / 3 / 5 | **1.000** / 0 / 1 |
| 1.5 | 20 | 0.024 | 0.890 / 1 / 4 | 1.000 / 0 / 0 |

  - 4TU(탐색): kept의 능선 이동 거리(검출 피크 → 끝점) 중앙 15 트레이스, 90 % 25, ≥ 4 트레이스 87 %. 이동 자체는 설계 의도(날개 위 검출 피크를 꼭짓점으로 옮김)라 결함 증거가 아니다. 4TU에는 옆 위치 정답이 없어 끝점 오차를 잴 수 없다.
- **깊이 영향**: 작다. v2·v3 F5-C 깊이 오차는 규칙과 무관하게 1.3–2.2 cm였다(최근접 kept 힌트). 깊이 단계가 힌트 주변에서 꼭짓점을 다시 맞추기 때문이다. K7-c는 주로 옆 위치(꼭짓점 4 트레이스 안 정지 비율, (e) 출력)와 시작점 수 의존(바닥 재시험의 공정성)에 영향을 준다.

### 1-4. 기작 요약

| ID | 기작 | 손실 단계 | 핵심 증거 | 자료 상태 |
|---|---|---|---|---|
| K7-a | $D$ 극대가 $E$ 극대가 아님 → 오르는 날 시작점이 남고 순위에서 이김 | kept(NMS), 깊이(`no_apex`) | S1g D100 0.43·$E$ 국소 최대 아님, C1 여유 0.24 ns | 이미 본 자료 |
| K7-b | 포화하는 날개 순위 + 상한 6 | kept(cap6) | S1g D060 primary 7개 cap6, 4TU cap 손실 4,988 | 이미 본 자료 |
| K7-c | 가장 이른 점이 평평한 꼭짓점에서 잡음을 따라 옆으로 감 | kept 창(위치), 수락(`not_apex`) 일부 | v2 끝점 4–6 트레이스, MC 0.68 | 이미 본 자료 + 순수 함수 |

## 2. 고칠 후보 (새 상수 없음, 플래그로 켬, 기본값 = 현행 바이트 동일; 채택안 = L1 a1, L2 b2)

### 2-1. K7-a 후보

| 후보 | 규칙 | 재사용 상수 | 사전 예측(새 실현) | S1g 재현(이미 본 자료, 잡음 없음) |
|---|---|---|---|---|
| **a1 시작점 오르막(채택, L1)** | 시작 정수 샘플 $k$에서 $E(\cdot,j_0)$가 엄격히 커지는 이웃으로 한 칸씩 옮겨 국소 최대에 닿으면 멈춘다(양쪽이 크면 큰 쪽, 양쪽이 같으면 늦은 쪽 = 큰 샘플 색인). 그다음 기존 `parabolic_peak`과 `crest_apex`. 순위 SNR은 **능선 이동 뒤 끝점(출력 `trace`·`sample`)의 $D$**로, 현행과 같은 위치다 | 없음(오르막은 매개변수가 없음) | $\ell$ 기울기가 있는 꼭짓점에서 kept 단계 성공률이 수락 단계 성공률까지 오름. 오르막이 뮤트 위로 가면 기존 `apex_at_edge_or_mute` | D100 두 관 kept 0 → 1·1. C1 그대로 1. D060 그대로 0(상한 탓). BGL kept 3 그대로 |
| a2 NMS 대표를 원 $E$로 | NMS 충돌 때 원 $E$가 큰 쪽이 자리를 차지 | 없음 | D100형은 고치나, 시작점이 날에 남은 채라 위치가 0.3 m·2 ns 안 다른 사건으로 바뀔 위험 | D100 1·1, C1 그대로(오프라인 재생) |
| (불채택) 엽 병합 | (b) v1.4 P2 상자 안 이른 kept를 늦은 쪽으로 | 상자 경계 | 상자 경계 여유 0.009 ns(D100)로 불안정 | – |

- a1의 위험: 공기 채움 플라스틱 관처럼 위·아래 반사가 1 주기 안에 둘 있으면, 날 위 시작점이 더 큰 아래 반사로 올라갈 수 있다(M18형 "트레이스 전역 최대가 관 아래·내부 반사" 함정, [`spec_depth_v1.md`](spec_depth_v1.md) §12의 첫 피크 ≥ 0.3 규칙 근거). 그러면 깊이가 깊게 나온다(외경 0.10 m 공기 관: 아래 반사가 관 안 공기 왕복으로 약 1.2–1.3 ns 뒤 → 흙 속도로 읽으면 $v\Delta t/2$ ≈ 6–7 cm). §4 모델 K7_PVC가 이것을 잰다.

### 2-2. K7-b 후보 (a1 위에서)

| 후보 | 규칙 | S1g(a1 위, 잡음 없음; `s1g_combo`) | 4TU(탐색) | 판단 |
|---|---|---|---|---|
| **b2 SNR 순위(채택, L2)** | 순위 키 −SNR(하나). 상한 6 유지 | C1 1, D060 1·1, D100 1·1, kept 수 그대로(4·6·6·3) | kept 2,629 → 2,628, 352측선 바뀜, 멀리 바뀜 835개(32 %) | 후보 수 불변. 검출 정규화가 이미 깊이 보정 강도라는 점(m210d K7 원안 "깊이 보정 강도")과 맞음 |
| b1 상한 제거(보고만, L2) | 상한 없음 | D060 kept 6 → 13, D100 6 → 10 | kept 2,629 → 5,905(현행 시작점, 오프라인) | 후보 2.25배 → QC·묶음 부담, 같은 추정 수 기준선도 오름 |
| b3 원 $E$ 순위 | $(-E)$ | 세 모델 모두 회복 | – | 깊은 약한 꼭짓점이 얕은 클러터에 밀림(검출 정규화를 둔 이유와 반대). 불채택 |
| b4 날개 합 | $(-(n_l+n_r))$, −SNR | D060 0·0, D100 0·1 | 320측선 바뀜, 멀리 바뀜 443개 | 실패. 불채택 |

- b2는 a1 없이 쓰면 안 된다: 현행 시작점에서 SNR 순위만 바꾸면 D100은 날 위 점(SNR 17.0)이 계속 이긴다(`s1g.S1G_D100.b2` 0·0).
- b2의 위험: 날개가 짧은 교차(limb crossing)·짧은 호가 SNR이 높으면 위로 온다. 현행 키의 주석 근거("short limb crossings rarely have long wings on both sides")가 바로 이것이다. `accept`의 날개 각 ≥ 3·합 ≥ 9·V자 검사가 1차로 막는다. §4-5 K-F3(거짓 후보)로 잰다.

### 2-3. K7-c 후보 (a1 위에서; L3에 따라 이번 판은 판정하지 않음 — 기록용)

| 후보 | 규칙 | 재사용 상수 | 사전 예측 | S1g(a1 위) |
|---|---|---|---|---|
| c1 $\tau^2$ 꼭짓점 | `crest_apex`가 걸은 능선 점 $(j,k)$ 가운데 가장 이른 점 $j_e$ 기준 $\lvert\xi\rvert\le0.30$ m인 점으로 $\tau^2=a+b\xi+c\xi^2$를 최소 제곱으로 맞춘다($\tau=k\,\Delta t-t_z$). 점 ≥ 5, 창 안 모든 $\tau>0$, $c>0$, $\lvert\xi_0\rvert\le0.30$ m이면 $\xi_0=-b/(2c)$의 가장 가까운 능선 점을 시작점으로 쓰고, 아니면 가장 이른 점 | 0.30 m·5점(`accept`의 곡률 창), $\tau^2$ 2차식(`accept` V자 검사의 직선 광선 쌍곡선) | MC에서 ±3 안 비율 1.5 m·ρ 10 0.68 → 1.00. 공기층·관경이 있어도 쌍곡선은 꼭짓점 대칭이라 $\xi_0$는 치우치지 않음 | a1+b2+c1: C1 1, D060 1·1, D100 1·1, BGL kept 3 → 1 |
| c1-post 보고 위치만 | `accept`는 현행 끝점으로 하고, 출력 `trace`·`sample`만 $\xi_0$ 점으로 | 같음 | 위치 오차는 고치나 끝점이 꼭짓점에서 멀어 생기는 `not_apex`·`few_picks` 손실은 그대로 | – |

- c1은 a1 없이 쓰면 안 된다: 현행 시작점에서 c1만 켜면 C1이 kept 1 → 0으로 나빠졌다(`D_variants_s1g.S1G_C1.c1`). 날 위 시작점에서 걸은 능선 점이 여러 사건에 걸쳐 맞춤이 흔들리기 때문으로 본다(추정).
- $\tau^2$ 식은 깊이 피팅이 아니다. 꼭짓점 **옆 위치**만 고르고, 속도·깊이에는 쓰지 않는다(공기결합 곡률 피팅 금지 규약과 충돌하지 않음).

### 2-4. 조합과 4TU 출력 범위 (탐색, 이미 본 자료)

[`m210_k7_probe.json`](m210_k7_probe.json) `D_variants_4tu`·[`m210_k7_rank_replay.json`](m210_k7_rank_replay.json) `fourtu_combo`(m210d 생성기 측선 중 영점이 잡힌 550측선, 생성기를 프로세스 안에서 바꿔 끼워 다시 생성):

| 변형 | kept 수 | 바뀐 측선(550 중) | 멀리 새로 생김 | 멀리 사라짐 | 출처 |
|---|---|---|---|---|---|
| cur(현행) | 2,629 | 0 | 0 | 0 | – |
| a1 | 2,626 | 80 | 37 | 40 | `D_variants_4tu` |
| **a1b2** | 2,628 | 352 | 834 | 835 | `fourtu_combo` |
| a1b4(불채택) | 2,627 | 320 | 441 | 443 | `fourtu_combo` |
| b1(현행 시작점) | 5,905 | 266 | 3,276 | 0 | `D_variants_4tu` |
| a1b1 | 5,895 | 306 | 3,298 | 32 | `D_variants_4tu` |
| c1(현행 시작점) | 2,311 | 519 | 725 | 1,043 | `D_variants_4tu` |
| a1c1 | 2,310 | 518 | 738 | 1,057 | `D_variants_4tu` |
| a1b1c1 | 4,419 | 520 | 2,824 | 1,034 | `D_variants_4tu` |

- a1b2c1은 4TU에서 돌리지 않았다.
- a1은 4TU kept의 약 1.5 %만 옮긴다. 현행 kept 가운데 끝점이 $E$의 시간 국소 최대가 아닌 것은 66개(2.5 %)다.
- b2는 kept 수를 지키지만 약 32 %(835/2,629)를 다른 후보로 바꾼다. 어느 쪽이 참 꼭짓점인지는 4TU에서 가를 수 없다(자동 후보 정밀도 약 11 %, [`m29c_apex_review.md`](m29c_apex_review.md)).
- **c1은 4TU kept의 약 40 %(1,057/2,629)를 지운다.** 지운 후보와 전체 kept의 성질이 같다(min 날개 ≥ 40 비율 0.015 대 0.015, 직선 곡률 v 중앙 0.0864 대 0.0873 m/ns, τ 중앙 11.07 대 11.06 ns; [`m210_k7_rank_replay.json`](m210_k7_rank_replay.json) `fourtu_c1_loss`). 곧 층·평평한 사건을 골라 지우는 것이 아니라 가리지 않고 바꾼다. 4TU kept의 대부분이 쌍곡선이 아니어서 $\tau^2$ 꼭짓점이 ±0.3 m 안에서 임의로 움직이고, 그 점에서 `accept`가 달라지는 것으로 본다(추정). 이 변화 폭 때문에 c1은 이번 판에서 판정하지 않는다(L3).

- "멀리 새로 생김/사라짐" = 현행 kept와 ±3 트레이스·±1 ns 안 짝이 없는 후보 수다. 정확 키(트레이스, 샘플) 차이는 c1의 1 트레이스 이동도 세므로 크게 나온다.
- 이 표는 4TU 출력이 얼마나 바뀌는지의 범위일 뿐이다. 판정률·MAE는 재지 않았다(4TU 맞춤 방지).

## 3. 기존 h5를 판정에 쓸 수 있는가

| 자료 | 개발·회귀 | 판정 | 이유 |
|---|---|---|---|
| S1g C1·D060·D100·BGL | 쓸 수 있음(§4-1 회귀 고정물, 하네스 점검) | **쓸 수 없음** | K7-a·b를 찾아낸 자료다. 잡음 없이도 결과가 결정된다(D100·D060). 새 잡음만 얹으면 같은 기하의 같은 결정을 다시 보는 셈이다 |
| S4N·S6P(묶음 1·v2·v3, 27모델) | 쓸 수 있음 | **잡음 시드를 새로 해도 판정에는 쓰지 않음(L4)** | K7-c 증거(v2 §6, v3 F3-S)가 나온 자료다. 새 시드는 독립 실현이지만 기하·σ가 같아 "결함을 본 조건"을 고른 셈이 된다 |
| 새 gprMax 모델(§4-3) | – | **판정용** | 기하·깊이·사건 구성을 새로 정하고 결과 전에 봉인 |

- L4에 따라 기존 h5에 새 시드만 얹는 안(v0.1 Q5 (2))은 쓰지 않는다.

## 4. 합성 시험 설계 (판정 기준 확정, 봉인 전)

### 4-1. 순수 함수 시험 (python-expert, gprMax 없음)

| ID | 대상 | 고정물 | 기대 |
|---|---|---|---|
| T1 | a1 오르막 함수 | 1-D 열: 오르는 날 위 시작, 국소 최대 위 시작, 배열 끝, 평평한 꼭대기 | 국소 최대 색인, 국소 최대 위면 그대로, 끝에서 멈춤, 평평하면 움직이지 않음(엄격한 증가만) |
| T2 | K7-a 재현 | 합성 B-scan: Ricker 쌍곡선(꼭짓점 0.6 m) + 꼭짓점 시각 뒤 0.5 ns부터 측선 전체 평평한 띠(행 수준 $\ell$을 올림). 고정물이 결함을 만든다는 사실(현행 시작점이 $E$ 국소 최대가 아님)을 먼저 단언 | 현행: 그 시작점이 kept로 남음. a1: kept 끝점이 $E$ 국소 최대이고 꼭짓점 ±3 트레이스·±0.5/f_c 안 |
| T3 | K7-b 재현 | 0.6 m 간격 두 쌍곡선(안쪽 날개 9픽에서 끊김) + 날개 50/50인 약한 평평 사건 6개 | 현행: primary `cap6`. a1+b2: primary kept, kept 수 ≤ 6 |
| T4 | c 진단 함수(L3, 출력만) | 잡음 없는 깊이 1.5 m 쌍곡선, 시작을 꼭짓점에서 5 트레이스 옆에 둠. 그리고 같은 고정물 + 고정 시드 잡음 1개 | 진단 $\tau^2$ 꼭짓점 트레이스 = 참 꼭짓점 ±1. 생성기 출력(`trace`·`sample`)은 바뀌지 않음 |
| T5 | c 진단 대체 경로 | 점 < 5, 창 안 $\tau\le0$, $c\le0$, $\lvert\xi_0\rvert>0.30$ | 진단 값 = 빈칸(None), 오류 없음 |
| T6 | 기본값 | 래퍼 기본 인자 | 기존 키의 값이 현행과 모두 같음(`_peaks`·`_accepted_all`·`_accepted_fate` 포함). 새 진단 키는 추가만 허용. csv 출력 바이트 동일 |
| T7 | 허용 조합 | §5의 허용 조합 4개 밖(예: `rank_rule="snr"` + `start_rule="current"`, `per_line_cap=None` + `start_rule="current"`) | `ValueError` |
| T8 | 위·아래 반사 | 1 주기 안 두 극대(아래가 큼) 사이 날 위 시작 | a1이 오르는 쪽 극대로 감을 기록(위험 문서화, 기대값은 오르막 정의 그대로) |

### 4-2. 기존 래퍼 기본값의 바이트 회귀 (G0)

- 생성기 파일은 플래그 추가로 blob이 `7cc2bf2b…`에서 바뀐다. 기본 경로가 현행과 같다는 증거는 이 회귀와 T6이다. 새 blob은 봉인 대상에 넣는다(§6).
- 하네스가 끝에서 G0 세 가지(아래)와 `sealed_files` 대조를 한다. 봉인 전 구현 확인(python-expert, 2026-11-12): 4TU 두 csv 바이트 동일, S1g 열 단위 동일, (b) v3 부분 집합 행 동일. 전체 pytest 수는 봉인 직전 값을 봉인 커밋 메시지에 적고(구현 보고 때 1085 passed), 판정서에는 하네스 시작 때 자식 pytest의 실행 시점 통과·skip 수를 적는다(§8 항목 8).
- 플래그를 모두 기본값으로 두고 4TU 759측선을 다시 생성한다. `fourtu_apex_candidates_v1.csv`(SHA-256 `1afd23cf…`)·`fourtu_apex_candidate_lines_v1.csv`(`485d4c22…`)가 커밋 파일과 **바이트 동일**해야 한다(S1g 판정 G0 (i)와 같은 방법). 이 SHA-256은 **작업 트리(CRLF) 바이트 기준**이다. 그래서 봉인 커밋부터 verify까지 이 두 csv를 checkout·clone하지 않는다(줄바꿈이 바뀌면 해시가 달라짐).
- M2-10 (b) 하네스(`m210b_s4s6_pipeline.py`)의 기본 경로가 v3 `runs` 한 부분 집합(모델 1·시드 2)에서 같은 값을 내야 한다.
- S1g 하네스(`m210_s1g_pipeline.py`) 기본 경로의 `candidates.csv`가 저장본과 **열 단위로** 같아야 한다. 저장본에는 K1 시절 열 `cluster_mu`가 더 있어 바이트 비교가 아니라 현재 `CANDIDATE_COLUMNS` 열만 비교한다. 판정 키 `g0_s1g` = `identical`, 저장본에만 있는 열은 `stored_only_columns`에 적는다.
- **상속 봉인 파일 변경 공개**: (b) v1.4·v2·v3의 봉인 파일 가운데 다음이 K7 구현으로 작업 트리에서 바뀐다.
  - 생성기 `fourtu_apex_candidate_select.py`(플래그·c 진단 추가, blob `7cc2bf2b…`에서 바뀜)
  - `tests/test_m210b_s4s6_pipeline.py`의 `test_floor_rules_and_wrapper_are_unchanged_since_752e783`(생성기는 752e783 blob 확인만 하고, 기본 경로 동일성은 K7 T6·G0가 맡는다)
  - `scripts/synthetic/export_dataset.py`(`--k7` 경로와 v1.2 속도 점검 처리, +197/−17줄, `git diff d6aaf0f~1 --numstat` 기준)
  - `tests/test_synthetic_export.py`(K7 시험 6개 = `test_k7*` 5개 + `test_a_lone_pipe_without_picks_still_fails`, +195/−0줄, `git diff d6aaf0f~1 --numstat` 기준)
  - 이 넷 모두 그 판정들은 각 봉인 커밋에서 재현하고, K7 변경이 그 판정을 바꾸지 않는다.

### 4-3. 새 gprMax 모델 (electromagnetic-simulation-expert 위임, L4 확정)

공통: S1g·S4S6과 같은 틀(2D TMz, 4.0 × 2.45 m, 공기층 0.10 m, 오프셋 0.10 m, Ricker 500 MHz, 셀 5 mm, 161트레이스 × 0.02 m, 55 ns). 기존 모델과 입력이 하나도 같지 않음을 생성 때 단언한다.

| ID | 구성 | 겨냥 기작 | 역할 |
|---|---|---|---|
| K7_P2_D070 | PEC 관 2개(R 0.05, 상단 0.70 m, 중심 x 1.5·2.5 m), εr 9, σ 0.001 | a | 결함 모델(D100의 새 깊이판) |
| K7_P2_D070_L | 위 + 평평한 경계(εr 9 → 12) 상단 0.85 m | a(ℓ 기울기 강화) | 결함 모델 |
| K7_P3_D080 | PEC 관 3개(R 0.05, 상단 0.80 m, 간격 0.6 m) + 평평한 경계 1.30 m | b | 결함 모델(D060형, 긴 날개 경쟁자 포함) |
| K7_P1_D150_S003 | PEC 관 1개(R 0.10, 상단 1.50 m), σ 0.003 | c 진단, b2 위험 | c 진단(L3, 보고) + a·b 대조(깊은 약한 꼭짓점이 SNR 순위에서 밀리지 않는가, K-F2) |
| K7_P1_D110_S002 | PEC 관 1개(R 0.10, 상단 1.10 m), σ 0.002 | c 진단, b2 위험 | 위와 같음(중간 깊이) |
| K7_P1_D060 | PEC 관 1개(R 0.05, 상단 0.60 m) | – | 대조(무손실 K-F2) |
| K7_PVC_D070 | 공기 채움 PVC(외경 R 0.10, 벽 0.01 m, εr 3) 상단 0.70 m | a1 위험 | 대조(아래 반사 편향 K-F4) |
| K7_BG_L | 관 없음 + 평평한 경계 0.85 m | – | 거짓 후보(K-F3) |

- 비용(추정): v2·v3 실측(9모델 약 42분, 2개 동시) 기준 8모델 약 40분 + 점검 실행 약 15분. 4070 SUPER 한 대, 다른 GPU 작업과 겹치지 않게.
- 다중 반사·경계 반사의 기대 시각(사건 목록)은 EM이 json에 낸다(S1g `events` 형식). K-F3의 "알려진 사건"은 이 목록으로 정한다.
- **EM 산출 경로**(electromagnetic-simulation-expert, 2026-11-12): 입력 `analysis/synthetic/k7/inputs/`, 점검 입력 `analysis/synthetic/k7/pilot/`, 모델 json `analysis/synthetic/k7/models_k7.json`, 실행 `analysis/synthetic/k7/run_k7.cmd`. 전체 실행 h5 `data/synthetic_gprmax_k7/h5/`, 점검 h5 `data/synthetic_gprmax_k7/pilot/`(점검 로그 `data/synthetic_gprmax_k7/pilot_run.log`).
- **공유 참조 h5**(영점 기준 배경 − FS, 3트레이스): `K7_REF_S001`(K7_P2_D070·K7_P1_D060·K7_PVC_D070), `K7_REF_S002`(K7_P1_D110_S002), `K7_REF_S003`(K7_P1_D150_S003), `K7_REF_L085_S001`(K7_P2_D070_L·K7_BG_L), `K7_REF_L130_S001`(K7_P3_D080). 자유 공간 `fs_ref` = S4/S6 `FS.h5` 재사용(머리말 참조 h5 예외). 6개 모두 snapshot 대상이다(§6-1). 잡음 보정용 순잡음 단면은 하네스가 시드로 만든다(§4-4).
- **점검 영점**(봉인 전 점검 실행, 지표 반사 피크): S001 3.6162 ns, S002 3.6156 ns, S003 3.6151 ns.
- **판정 해석 주의**(EM, 결과 전 공개):
  - K7_PVC_D070의 primary는 아래 벽 반사와 합쳐져 포락선 피크가 약 +0.22 ns 늦다(흙 속도로 약 +1.1 cm). K-F4는 `cur`와의 차라 영향이 없고, K-F5 절대 오차 보고에 이 몫이 들어간다.
  - K7_P1_D060은 이미 본 S1G_C1과 물리적으로 거의 같다(단일 PEC 관 0.60 m). 대조(K-F2) 역할에는 문제가 없지만, 새 기하가 아니라는 점을 판정서에 적는다.
  - 층 모델(K7_P2_D070_L·K7_P3_D080·K7_BG_L)의 층 관련 사건은 기대 시각이 서로 가까워 사건 목록에 `ambiguous_with`로 표시돼 있다. 붙임은 §4-4의 `attach_k7_event` 규칙을 따르고 `ambiguous_with`는 기록만 한다. 애매한 사건끼리의 붙임 차이는 U 수(K-F3)를 바꾸지 않는다(어느 쪽이든 알려진 사건).
- **역할 집합(판정 스크립트가 고정 단언, §4-6)**: `defect_a` = (K7_P2_D070, K7_P2_D070_L), `defect_b` = (K7_P3_D080), `control` = (K7_P1_D060, K7_P1_D150_S003, K7_P1_D110_S002, K7_PVC_D070), `pvc` = (K7_PVC_D070, control의 부분 집합), `bg` = (K7_BG_L), `c_diag` = (K7_P1_D150_S003, K7_P1_D110_S002, 보고만). 기대 모델 집합은 위 표의 8개 ID다.
- **내보내기 속도 점검(v1.2)**: 내보내기 json의 관별 `velocity_check`(트레이스별 포락선 픽 기울기로 본 겉보기 속도 비, S1g 형식)는 **판정 입력이 아니다**. K7 하네스·판정 스크립트·`load_synthetic`은 이 키를 읽지 않는다(코드 검색으로 확인). 다른 사용처는 `scripts/synthetic/theory_table.py`(M01–M20 보고)와 내보내기 자신뿐이다. **이웃 관이 있고**, 이웃 제외를 통과한 트레이스가 0개라 쓸 점이 0개인 관(K7_P3_D080 가운데 관, 0.6 m 간격)만 `velocity_check: null`과 `velocity_check_note: "no_usable_picks_neighbours"`를 기록하고 내보내기를 계속한다. 관이 하나뿐인데 점이 0개이면 여전히 `ValueError`로 멈춘다. 이 경우 manifest의 `apparent_v_ratio_pick_slope`만 빈 값(None)이고, 콘솔 요약은 `vg/vt=n/a (no usable picks)`로 찍힌다.
- 점검 실행(봉인 전)은 측선 일부·단일 트레이스로 했다(점검 입력 `P_<모델>`). 그 h5와 `pilot_run.log`는 전체 실행 전에 보관 폴더로 옮긴다(§6-1).

### 4-4. 하네스 규약 (확정)

- **파일 이름(구현 확정)**: 하네스 `analysis/processing/m210_k7_pipeline.py --protocol k7 --seal-commit <봉인 커밋>`, 판정 `analysis/processing/m210_k7_judge.py --expected-seal-commit <봉인 커밋>`, 시험 `tests/test_m210_k7_generator.py`·`tests/test_m210_k7_pipeline.py`·`tests/test_m210_k7_judge.py`와 `tests/test_synthetic_export.py`의 K7 시험 6개(`test_k7*` 5개 + `test_a_lone_pipe_without_picks_still_fails`), 내보내기 `scripts/synthetic/export_dataset.py --k7 analysis/synthetic/k7/models_k7.json` → `data/synthetic_gprmax_k7/`, 하네스 출력 `data/processing/m210_k7/`(`runs_k7.csv` 37열, `run_info.json`, 하네스만 쓰는 `pre_run.txt`·`post_run.txt`).
- **시작·끝 가드**: 하네스는 시작 때 K7 시험 파일 3개(`test_m210_k7_generator.py`·`test_m210_k7_pipeline.py`·`test_m210_k7_judge.py`)를 자식 pytest로 돌려 하나라도 실패하면 즉시 끝낸다. 끝에서 G0 세 가지(§4-2)와 `sealed_files` 대조를 한다.
- **경로**: 재표본 0.098 ns((b) v1.4 규약 2와 같음) → `prepare_depth_inputs` → 래퍼(변형 인자) → `ev.run_line(apex_qc=True)` → 주 QC 통과 → 사건 붙임(`attach_k7_event`, 아래).
- **사건 목록과 붙임**(봉인 전 확정, 하네스 `m210_k7_pipeline.py`; S1g 하네스 파일은 바뀌지 않음):
  - 사건 목록은 `models_k7.json`의 최상위 `expected_events`를 읽는다. 모델 항목 안에 사본이 있으면 최상위와 같아야 하고, 다르면 시작 가드 실패다.
  - 사건 종류 `K7_EVENT_KINDS` = S1g 4종(`primary`·`multiple_surface`·`multiple_pipe`·`multiple_other`) + `flat_layer`·`multiple_layer`·`pvc_internal`. json `expected_events_meta.kinds`와 같다. 모르는 종류는 오류다. `x_m: null`은 `flat_layer`에만 허용한다.
  - `attach_k7_event`: S1g `attach_event` 규칙 그대로다. $\lvert\Delta x\rvert\le0.10$ m이고 $\lvert\Delta t\rvert\le0.5/f_c$인 사건 가운데 `primary`를 먼저, 그다음 가까운 시각 → 작은 $\lvert\Delta x\rvert$ → json 순서로 하나를 고른다. 단 `flat_layer`는 모든 트레이스에 걸친 사건이라 $\lvert\Delta x\rvert=0$으로 본다.
  - 관 정답(성공·깊이·K-F4)은 `kind == "primary"`인 사건만이다. `pvc_internal`은 관 색인(`pipe_index` 0)을 갖지만 primary가 아니다. 알려진 사건이므로 붙으면 U에 들지 않는다.
  - `ambiguous_with`는 기록만 하고 붙임에 쓰지 않는다.
  - U = QC 통과 후보 가운데 어느 사건에도 붙지 않은 것.
- **꼭짓점 기준 시각** $t_{a0}$: 관마다, 같은 모델의 잡음 없는 처리 결과(하네스 경로, `cur`)에서 꼭짓점 트레이스 포락선의 json `envelope_peak_ns` ±0.5/f_c 안 최대 시각((b) v1.4 P2와 같음).
- **성공 창**: 꼭짓점 트레이스 ±3, $t_{a0}$ ±0.5/f_c(±1 ns).
- **잡음**: $\sigma_\text{raw}=A_0/(\rho\,\hat\sigma_u)$, ρ ∈ {10, 20}. $A_0$ = 잡음 없는 처리 결과에서 모델 안 관들의 꼭짓점 포락선(꼭짓점 트레이스 ±3·$t_{a0}$ ±0.5/f_c 최대) 가운데 **최솟값**. K7_BG_L은 관이 없으므로 같은 층 구조인 K7_P2_D070_L의 **보정 전체**($A_0$와 $\hat\sigma_u$에서 나온 $\sigma_\text{raw}$)를 그대로 쓴다((b) v3 W10 방식, 보정 짝, 판정 스크립트가 단언). $\hat\sigma_u$ = 순잡음 단면(시드 20301199)의 처리 결과에서 탐색 구간 1.4826·MAD.
- **시드**: 조건(모델 × ρ)당 40. 시드식 `20301112 + 1000·m + 100·q + s`. $m$ = §4-3 표 순서의 모델 색인 0–7, $q$ = ρ 색인(0: 10, 1: 20), $s$ = 0–39. 하네스 시작 때 다음 집합과 겹침 0을 단언한다: 묶음 1·v2·v3 잡음·순잡음 시드, (b) 시제품 시드, 이 문서 §1-3 MC 시드(20261112와 20261113 + n + 1000ρ + ⌊100d⌋, n 0–199, ρ {10, 20}, d {0.6, 1.0, 1.5}).
- **잡음 없는 행**: 모델 × 변형 × 관마다 1행(`noise_free = true`). 관이 없는 K7_BG_L은 모델 × 변형마다 1행(`pipe_index` 빈칸). $t_{a0}$·$A_0$와 보고에만 쓰고 성공률에는 넣지 않는다.
- **변형 집합**(고정): `cur`(기준), `a1`(판정), `a1b2`(판정), `a1b1`(보고: 후보 수 비용).
- **행 수**(행 = 관 하나, 관 없으면 측선 하나): 잡음 행은 조건(모델 × ρ × 변형)마다 40 × max(관 수, 1). 관 수 합 11(2 + 2 + 3 + 1 + 1 + 1 + 1 + 0)에 K7_BG_L 1을 더한 Σ max(관 수, 1) = 12이므로 잡음 행 = 12 × ρ 2 × 변형 4 × 40 = 3,840행, 잡음 없는 행 = 12 × 변형 4 = **48행**. 합계 3,888행(v1.0의 "2,560행 + 32행"은 관 단위 행을 세지 않은 값이라 정정). 깊이 단계 포함 1코어 약 40–70분(추정). 30분을 넘으므로 메인 세션 권고 + long-running-task-monitor 규칙.
- **행별 출력**(관이 여럿이면 관마다): 수락 단계 성공, kept 단계 성공, QC 통과 primary 유무와 $\hat d$(관 상단), kept 수, QC 통과 미붙음(U) 수, 손실 사유(`no_peak`·`not_accepted`·`cap6`·`nms`).
- **구현 해석 (a)–(i)(봉인 전 확인, 구현과 같음)**:
  - (a) K7_BG_L 보정은 K7_P2_D070_L의 보정 전체($A_0$·$\hat\sigma_u$ → $\sigma_\text{raw}$)를 쓴다(위 **잡음**).
  - (b) 조건 행 수는 40 × max(관 수, 1)이다(관 없는 K7_BG_L은 측선 하나에 1행; 위 **행 수**).
  - (c) 잡음 없는 행은 모델 × 변형마다 max(관 수, 1)행, 합 48행이다(위 **잡음 없는 행**·**행 수**).
  - (d) 같은 관에 QC 통과 primary가 여럿이면 사건 기대 시각에 가장 가까운 것 → 꼭짓점과의 $\lvert\Delta x\rvert$가 작은 것 → 후보 순서로 하나를 고른다. 깊이 $\hat d$는 그 하나로 잰다.
  - (e) 손실 사유는 (b) 하네스의 `early_lobe_nms`·`other_nms`를 `nms` 하나로 묶는다. 원래 값은 `lost_reason_detail`에 남긴다(보고).
  - (f) K-F6의 "시작점 수"는 `n_window_starts` = 능선 이동 뒤 끝점이 성공 창 안에 드는 검출 피크 수다.
  - (g) b2(`rank_rule="snr"`)의 SNR 동률은 검출 피크 순서(안정 정렬)로 가른다. 시제품(`m210_k7_rank_replay.py`)은 동률을 날개 키로 갈랐으나 S1g 결과는 같다.
  - (h) S1g G0는 바이트가 아니라 현재 `CANDIDATE_COLUMNS` 열 단위 비교다(§4-2).
  - (i) a1 오르막에서 양쪽 이웃이 같은 크기로 크면 늦은 쪽(큰 샘플 색인)으로 간다(`climb_to_local_max`, §2-1).
- **c 진단 열(L3, 보고만)**: 창 안 kept(없으면 창 안 수락 항목)의 끝점 옆 오차(트레이스, 부호 포함), 시작점·끝점이 $E$ 시간 국소 최대인지, $\tau^2$ 꼭짓점 트레이스(§2-3 c1 식, 실패 시 빈칸). 이 열은 판정 항목에 들어가지 않는다.
- **정체성·가드**: (b) v2.1 H7과 같다. 입력 해시는 행 루프 전에, 시작 가드 실패 시 즉시 종료, git 실패 = 가드 실패, 끝에서 `sealed_files` 대조. `run_info`에 `protocol = "k7"`, `sealed_files`, `mkl_cbwr`, 입력 SHA-256, 생성기 blob, 시드 겹침 단언 결과를 적는다.

### 4-5. 판정 항목 (확정)

성공률은 조건(모델 × ρ)마다 40시드 비율이다. 관이 여럿인 모델은 관마다 따로 세고 관 평균을 쓴다(관별 값도 보고). 분모는 언제나 40 × 관 수다.

| ID | 범위 | 정의 | 기준 |
|---|---|---|---|
| K-R(a) | `defect_a`, `cur`, ρ 20 | gap = 수락 단계 성공률 − kept 단계 성공률 | `defect_a` 모델 중 **하나 이상**에서 gap ≥ 0.5 → 기작 a 재현 |
| K-R(b) | `defect_b`, `cur`, ρ 20 | 같은 gap | gap ≥ 0.5 → 기작 b 재현 |
| K-F1 회복 | a1: `defect_a` × ρ. a1b2: `defect_a` ∪ `defect_b` × ρ | 후보의 kept 단계 성공률 | 모든 셀에서 ≥ `cur` 수락 단계 성공률 − 0.1 |
| K-F2 무손실 | `control` × ρ(8셀) | kept 단계 성공률, QC 통과 primary 비율 | 모든 셀에서 둘 다 후보 ≥ `cur` − 0.1 |
| K-F3 거짓 | 8모델 × ρ(16셀) | QC 통과 U의 측선당 평균 | 모든 셀에서 후보 ≤ `cur` + 0.5 |
| K-F4 깊이 | 8모델 중 관 있는 7모델 × ρ, 짝 행 | 짝 행 = 같은 모델·ρ·시드·관에서 `cur`와 후보 모두 QC 통과 primary가 있는 행. 셀마다 $\hat d-d_\text{top}$ 중앙값 | 짝 행 ≥ 20인 모든 셀에서 $\lvert$중앙값(후보) − 중앙값(`cur`)$\rvert\le0.02$ m |
| K-F5 깊이 성공(보고) | 관 있는 7모델 × ρ | QC 통과 primary가 $\lvert\hat d-d_\text{top}\rvert\le0.10$ m인 비율 | 보고만 |
| K-F6 보고 | 전체 | kept 수, 손실 사유(`lost_reason_detail` 포함), `a1b1` 후보 수 비용, 시작점 수(`n_window_starts`)별 성공률(F3-S형), c 진단 열의 분포(모델·변형별 옆 오차 분위, ±3 안 비율, 국소 최대 비율) | 보고만 |

- 문턱 재사용: 0.1(v2.1 F 항목), 0.5(R의 e), 0.02 m·0.10 m(합성 일치·허용), 짝 행 20(= 40시드의 0.5, G 바닥 비율 0.5와 같은 비율). **새 상수 없음.** v0.1의 c 문턱 0.8·0.95는 쓰지 않는다(L3).
- K7_PVC에서 a1이 아래 반사로 오르면 K-F4 차가 약 +6–7 cm로 나야 한다(§2-1). K-F4가 이것을 잡는다.

### 4-6. 결정표 (확정; 위에서부터 처음 맞는 행; 후보마다 따로)

| 순위 | 조건 | 결과 |
|---|---|---|
| 1 | 정체성 실패: §8 첫머리 목록 누락·불일치(공개 항목의 **값**은 제외), G0 바이트 회귀 실패, 순수 함수·하네스·판정 시험 실패, 시드 겹침, 고정 집합 단언 실패(아래), 판정 대상 셀 수가 정의와 다름 | 보류 — 구현·실행 정체성 수정 |
| 2 | 후보가 겨냥한 기작의 K-R 미성립(a1 → K-R(a), a1b2 → K-R(b)), 또는 K-F4 판정 불가(K7_PVC 두 셀 모두 짝 행 < 20) | 보류 — 재현 조건 미달 또는 위험 측정 불가(모델 재설계) |
| 3 | K-F2 실패 | 기각 — 대조 손상 |
| 4 | K-F3 실패 | 기각 — 거짓 후보 증가 |
| 5 | K-F4 실패 | 기각 — 깊이 편향 |
| 6 | K-F1 실패 | 불충분 — 결함을 못 고침 |
| 7 | 1–6 어디에도 해당 없음 | 채택 가능 |

- **판정 순서**: a1 → a1b2. a1b2는 a1이 순위 7일 때만 순위를 매긴다. a1이 7이 아니면 a1b2는 "판정 안 함(선행 미충족)"으로 적고 값만 보고한다.
- **최종 채택**: 순위 7에 닿은 마지막 후보. 셋 중 하나다 — 현행 유지(a1이 7 아님), a1, a1b2. 채택 뒤 4TU 재생성·재평가는 사용자 결정 뒤 탐색 보고로만 한다(판정 규칙 불변, 4TU 결과로 이 판정을 바꾸지 않음).
- **고정 집합 단언**(판정 스크립트, 실패 시 순위 1):
  - 기대 모델 집합 = §4-3 표 8개 ID와 같음(빠짐·더함 모두 실패)
  - 역할 집합 = §4-3의 `defect_a`·`defect_b`·`control`·`pvc`·`bg`·`c_diag` 정의와 같음. `pvc` ⊂ `control`, `defect_a`·`defect_b`·`control`·`bg`는 서로소이고 합집합이 8모델
  - 변형 집합 = {`cur`, `a1`, `a1b2`, `a1b1`}, ρ 집합 = {10, 20}, 시드 = 0–39 전부, 조건마다 잡음 행 40 × max(관 수, 1), 모델 × 변형마다 잡음 없는 행 max(관 수, 1)
  - 관 수 = 모델 json의 `pipes` 길이(K7_BG_L 0)
  - 보정 짝 K7_BG_L → K7_P2_D070_L
  - `models_k7.json`의 SHA-256 = `run_info` 기록 = 봉인 커밋 blob의 내용(판정 때 재계산)
  - 관별 행 수: 관이 있는 모델은 조건마다 관 색인(`pipe_index`)마다 정확히 40행, K7_BG_L은 `pipe_index` 빈칸 40행
- **빈 경우 규칙**:
  - 성공률 분모는 언제나 40 × 관 수라 비지 않는다(관 없는 K7_BG_L은 성공률이 없고 K-F3에만 쓴다). 조건 행 수가 40 × max(관 수, 1)와 다르면 고정 집합 실패(순위 1)다.
  - `cur` 수락 단계 성공률이 0이면 그 모델의 gap ≤ 0이다. 곧 그 모델에서 K-R 미성립으로 센다.
  - K-F1·K-F2·K-F3의 대상 셀 수는 정의로 정해진다(K-F1: a1 4셀, a1b2 6셀; K-F2 8셀; K-F3 16셀). 판정 스크립트는 셀 수를 출력하고, 정의와 다르면(0 포함) 통과가 아니라 순위 1이다. **빈 집합은 통과가 아니다.**
  - K-F2에서 `cur` 값이 0인 셀은 기준이 자명하게 맞는다. 판정에는 그대로 쓰되 "자명 통과" 셀 목록을 공개한다(§8 항목 11).
  - K-F3에서 QC 통과가 없는 행의 U는 0이다(빈칸 아님).
  - K-F4에서 짝 행 < 20인 셀은 "해당 없음"으로 공개하고 기준에서 뺀다. 단 K7_PVC 두 셀이 모두 해당 없음이면 순위 2다. 해당 없음이 아닌 셀이 하나도 없으면(모든 7모델) 역시 순위 2다.
- **망라성**: 순위 7이 나머지라 모든 경우를 덮는다. 순위 1·2의 조건은 판정 스크립트가 기계 출력하고(`rank_reasons`), 판정서는 그 값을 옮긴다.

## 5. python-expert 위임 명세 (확정, 봉인 전 착수 가능)

| # | 대상 | 변경 | 계약 |
|---|---|---|---|
| K7-1 | `analysis/processing/fourtu_apex_candidate_select.py` | 래퍼 인자 `start_rule: "current" \| "envelope_max"`(a1), `rank_rule: "wings_snr" \| "snr"`(b2), `per_line_cap: int \| None = MAX_APEX_CANDIDATES_PER_LINE`(a1b1 보고용) | 기본값 = 현행. 허용 조합은 4개뿐: `cur` = (`current`, `wings_snr`, 6), `a1` = (`envelope_max`, `wings_snr`, 6), `a1b2` = (`envelope_max`, `snr`, 6), `a1b1` = (`envelope_max`, `wings_snr`, None). 그 밖은 `ValueError`(T7). a1 오르막은 §2-1 정의(엄격 증가 이웃으로 이동, 양쪽이 크면 큰 쪽, 같으면 늦은 쪽, 그다음 기존 `parabolic_peak`). 순위 SNR은 능선 이동 뒤 끝점(출력 `trace`·`sample`)의 $D$로 현행과 같은 위치(`fourtu_apex_candidate_select.py`의 `accepted` 행 `snr`) |
| K7-2 | 〃 | c 진단(L3) | `_peaks`에 `sample_snapped`, `start_is_local_max`, `end_is_local_max`, `tau2_vertex_trace`(§2-3 c1 식; 점 < 5, $\tau\le0$, $c\le0$, $\lvert\xi_0\rvert>0.30$ m이면 None)를 **추가만** 한다. 생성기 결정(수락·순위·NMS·출력 `trace`·`sample`)에 쓰지 않는다. `crest_apex`의 결정 로직은 바꾸지 않는다(진단은 능선 점 목록을 따로 받아 계산) |
| K7-3 | `analysis/processing/m210_k7_pipeline.py`(새, `--protocol k7 --seal-commit`) | 하네스 | §4-4 전부. `K7_SEALED_FILES`·`K7_SEALED_FOLDERS`, 시작 때 K7 시험 3파일 자식 pytest(실패 시 즉시 종료), 시작·끝 정체성, 끝 G0 세 가지, 시드 겹침 단언, `pre_run.txt`·`post_run.txt`는 하네스만 쓴다 |
| K7-4 | `analysis/processing/m210_k7_judge.py`(새) | 판정 | §4-5·§4-6. `--expected-seal-commit`. 자기 blob = 봉인 커밋 blob, `run_info.protocol == "k7"`, `sealed_files ⊇ K7_SEALED_FILES`를 **스스로 단언**한다((b) v3 실행 전 감사 6의 공백 해소). 고정 집합 단언, 셀 수 출력, 빈 경우 규칙, 후보별 순위와 `rank_reasons`, 최종 채택, 보고 키(K-F5·K-F6, c 진단, 자명 통과·해당 없음 셀 목록) |
| K7-5 | `scripts/synthetic/export_dataset.py` | `--k7 <models_k7.json>` | `data/synthetic_gprmax_k7/`로 내보내기. 기존 폴더와 섞지 않음 |
| K7-6 | 시험 | `tests/test_m210_k7_generator.py`, `tests/test_m210_k7_pipeline.py`, `tests/test_m210_k7_judge.py`, `tests/test_synthetic_export.py`(K7 6개) | §4-1 T1–T8, 허용 조합 밖 → `ValueError`, 고정 집합 누락(모델 하나 뺌) → 순위 1, 대상 셀 0 → 순위 1, K7_PVC 짝 행 부족 → 순위 2, 시드 하나 겹침 → 실패, 결정표 순위 1–7 각 경계 |
| K7-7 | G0 | §4-2 | 기본 경로 4TU 두 csv 바이트 동일, (b)·S1g 하네스 기본 경로 회귀 |

- 4TU 커밋 산출물(`fourtu_apex_candidates_v1.csv` 등)을 덮어쓰지 않는다. 변형 실행은 다른 출력 폴더에 쓴다.
- 새 상수 없음. 재사용: `NMS_M`·`NMS_NS`, `accept`의 0.30 m·5점(c 진단), `STOP_REL`, `MAX_APEX_CANDIDATES_PER_LINE`.
- 바닥 규칙 `m210b_floor_rules.py`는 바꾸지 않는다(blob `bad0f95c…`).

## 6. 봉인 대상과 순서

- **봉인 대상**:
  - 이 문서(v1.2 봉인판)
  - `K7_SEALED_FILES`(하네스 상수, 판정 스크립트가 `run_info.sealed_files ⊇` 단언):
    - 이 문서, 결정 근거 `analysis/processing/m210_k7_probe.py`·`analysis/processing/m210_k7_rank_replay.py`·`docs/processing/m210_k7_probe.json`·`docs/processing/m210_k7_rank_replay.json`
    - `analysis/synthetic/k7/models_k7.json`
    - 생성기 `analysis/processing/fourtu_apex_candidate_select.py`(새 blob), 하네스 `analysis/processing/m210_k7_pipeline.py`, 판정 `analysis/processing/m210_k7_judge.py`, 내보내기 `scripts/synthetic/export_dataset.py`
    - 시험 `tests/test_m210_k7_generator.py`·`tests/test_m210_k7_pipeline.py`·`tests/test_m210_k7_judge.py`·`tests/test_synthetic_export.py`·`tests/test_m210b_s4s6_pipeline.py`
    - 상속: `analysis/processing/m210b_floor_rules.py`(`bad0f95c…`, 변경 없음), `analysis/processing/m210b_s4s6_pipeline.py`, `analysis/processing/m210_s1g_pipeline.py`, `scripts/eval_depth_4tu.py`, QC 상수 `docs/processing/m29c_qc_constants.json`(LF 기준 SHA-256 `6a88bd8d…ec56ba8c`)
  - `K7_SEALED_FOLDERS` = `analysis/synthetic/k7`(폴더 통째: 모델 json, `inputs/`, `pilot/`, `run_k7.cmd`)
  - `DepthConfig()` 기본값은 `run_info`에 기록한다(파일 봉인이 아니라 기록 대조)
- **명세 판 문자열**: 하네스 `run_info.spec`("… v1.1 sections 4-4 and 5 …")과 판정 스크립트 출력 `note`("… v1.1 4-5, 4-6")의 "v1.1"은 하네스·판정 스크립트 blob을 `d6aaf0f`와 같게 두려고 고치지 않는다. 두 스크립트의 계약(§4-4–§4-6)은 v1.2에서 바뀌지 않았다. 판정서는 이 문자열이 아니라 봉인 커밋과 blob(§8 항목 1·18)으로 판을 가린다.
- **v1.2 재봉인에서 바뀌는 봉인 대상**: 이 문서, `scripts/synthetic/export_dataset.py`(작업 트리 blob `00069282…`, 봉인 커밋에서 재확인), `tests/test_synthetic_export.py`(시험 2개 추가: `test_k7_middle_pipe_between_neighbours_has_no_velocity_check`, `test_a_lone_pipe_without_picks_still_fails`)의 blob. 하네스·판정 스크립트·생성기는 바뀌지 않는다. 봉인 전 확인(python-expert): 전체 pytest 1099 passed, 1 skipped. 나머지 `K7_SEALED_FILES`·`K7_SEALED_FOLDERS`는 `d6aaf0f`와 blob이 같아야 한다(판정서 §8 항목 1에서 파일별로 대조). §8의 "봉인 커밋"은 v1.2 재봉인 커밋이다(`d6aaf0f` 아님).
- **순서**: 사용자 결정(완료, §0) → EM(모델·점검)과 python-expert(코드·시험)를 봉인 전에 병행 → 무결성 점검 → **실행 전 감사(봉인 전)** → 지적 반영 → 봉인 커밋 → §6-1 운영 → verify → 판정.

### 6-1. 실행 운영 순서 (봉인 커밋 뒤)

**run1 보관(v1.2)**: 첫 봉인 실행 산출은 `data/synthetic_gprmax_k7/run1_d6aaf0f/`(h5 13개, `runner_logs/`, 부분 내보내기 `export_partial/`, `export_k7.err`)에 있다. 이 폴더는 **판정에 쓰지 않는다.** 재실행부터 verify까지 이 폴더에 쓰지 않는다. 재실행 전에 run1 h5 13개의 파일 SHA-256을 기록한다.

1. 하네스 로그 폴더 `data/processing/m210_k7/`와 러너 로그 폴더 `data/synthetic_gprmax_k7/runner_logs/`(새로)를 **gprMax 전에** 만든다.
2. **점검 산출 보관**: 봉인 전 점검 실행의 `data/synthetic_gprmax_k7/pilot/`(h5·`.log`)와 `data/synthetic_gprmax_k7/pilot_run.log`를 전체 실행 전에 `data/synthetic_gprmax_k7/pilot_prerun_h5/`로 **이동**한다(복사 아님, mtime 유지). 입력 폴더 맨 위에 점검 로그를 남기지 않는다((b) v3 감사 10 재발 방지).
3. gprMax 전체 실행: 8모델과 참조 자료(`_bg`·`_fs`). gprMax 로그와 러너 로그는 새 파일로 쓴다. GPU는 한 계열 작업만.
4. **점검 산출 재현 대조**: 점검과 전체 실행이 같은 입력으로 만든 h5는 `K7_REF_S001`·`K7_REF_S002`·`K7_REF_S003` 셋뿐이다. 층 참조 `K7_REF_L085_S001`·`K7_REF_L130_S001`은 점검 실행에 없었고(§9), 모델 점검 `P_<모델>`은 입력이 달라 대상이 아니다. 셋은 python-dev로 `~/.claude/tools/gprmax/h5_compare.py <pilot_prerun_h5 안의 점검 h5> <새 h5> --expect bit --all-datasets --format json --out <scratchpad 보고서 경로>`(`--out` 필수, 점검 h5는 2에서 옮긴 경로)로 `rxs/rx1/Ez`·dt·Iterations를 **배열 비트 단위로** 비교한다. 파일 SHA-256은 기록만 한다(HDF5 `track_times` 때문에 배열이 같아도 다름, 예상된 차이). 배열이 다르면 판정서에 공개하고 전체 실행 것을 쓴다.
   - **run1 재현 대조(v1.2, 공개 항목)**: 재실행 h5 13개를 같은 이름의 run1 h5와 4와 같은 명령(`--expect bit --all-datasets --format json --out <scratchpad 보고서 경로>`)으로 배열 비트 비교한다. 같은 입력의 결정적 계산이라 같아야 한다. 결과(같음/다름, 다르면 데이터셋 이름)는 판정서에 공개하고, 판정에는 언제나 재실행 h5를 쓴다. 파일 SHA-256은 기록만 한다(`track_times`).
5. **h5 snapshot**: 내보내기 전에 참조 h5 6개(`K7_REF_S001`·`S002`·`S003`·`L085_S001`·`L130_S001`과 재사용 `data/synthetic_gprmax_s4s6/h5/FS.h5`)와 **모델 h5 8개**(`data/synthetic_gprmax_k7/h5/K7_*_merged.h5`)에 `source_guard` snapshot을 건다. 파일 SHA-256은 기록만 하고, 대조 기준은 snapshot ↔ verify 값이다.
6. 내보내기(`MKL_CBWR=COMPATIBLE`로 실행) → 내보낸 npy·json에 `source_guard` snapshot → 하네스 → 하네스 종료 뒤 **두 snapshot 모두 verify**.
   - `MKL_CBWR=COMPATIBLE`가 필요한 이유: S1g 내보내기 json 3개가 `velocity_check` polyfit 값의 끝자리만 MKL 실행 경로에 따라 달라졌고, `MKL_CBWR=COMPATIBLE`에서는 77f5f96 스크립트와 현재 스크립트가 바이트 동일했다(코드 무관, python-expert 확인). 판정 입력 npy는 h5 원값을 그대로 옮긴 것이라 영향이 없다. 흔들릴 수 있는 것은 json의 보고 값 끝자리뿐이다. S4S6 묶음 1·v2·v3 내보내기는 각 37파일이 바이트 동일했다.
7. 봉인 커밋부터 verify까지 어느 세션도 커밋·저장소 쓰기를 하지 않는다. 저장소 쓰기에는 `.claude/agent-memory/**`(에이전트 메모리)도 든다. 그래서 이 구간에는 메모리를 쓰는 프로젝트 에이전트를 부르지 않는다(작업 트리가 dirty가 되면 순위 1). 입력 폴더 맨 위에는 하네스 시작부터 끝까지 쓰지 않는다.
8. **하네스 출력 폴더의 `pre_run.txt`·`post_run.txt`는 하네스만 쓴다.** 메인 세션의 정체성 기록은 같은 폴더의 `main_session_pre.txt`·`main_session_post.txt`(또는 다른 폴더)에 쓴다((b) v3 판정서의 `post_run.txt` 덮어쓰기 재발 방지).
9. 하네스가 30분을 넘으면 long-running-task-monitor 규칙을 따른다.

### 6-2. 첫 봉인 실행의 대체 이력 (v1.2)

- 첫 봉인 `d6aaf0f`(v1.1, 12:48:54) 뒤 §6-1대로 gprMax 13단계(8모델 + 참조 5)를 12:50–14:26에 돌렸고 모두 exit 0이었다. 레인 B는 12:59–14:00에 conda `reg.exe`가 Suspended 상태로 멈췄다가 사용자 승인 뒤 재개됐다.
- §6-1 4의 점검 산출 대조: `K7_REF_S001`–`S003`가 점검 산출과 배열 비트 동일이었다.
- 내보내기(14:27:00–14:27:41)가 `K7_P3_D080 pipe 1: no usable per-trace picks for the velocity check`로 실패했다. 0.6 m 간격 관 3개 가운데 관은 양쪽 이웃 관 때문에 트레이스별 픽이 모두 빠져 점이 0개였다. P2 두 모델만 내보내졌다.
- 하네스·판정은 돌지 않았다. 성공률·깊이 등 **판정 결과는 하나도 없다.** 그래서 v1.1 판정은 성립하지 않고, v1.2 재봉인 실행으로 대체한다.
- **마른 내보내기에서 본 범위(공개, python-expert 사실 보고)**: 보관 run1 h5로 내보내기를 끝까지 돌려(출력 scratchpad, 뒤에 삭제) 다른 실패가 없는지 봤다.
  - 부른 것: 내보내기 `main(--k7 … --h5-dir <run1> … --out-dir <scratchpad>)`, 하네스 모듈의 `json_models`·`model_set_failures`·`load_model`(`load_synthetic` + 관 정답)·`floor_harness.resample_to`·`floor_harness.input_record`뿐이다.
  - 부르지 않은 것: 생성기(`apex_candidates_for_radargram`·`_peaks`), `prepare_depth_inputs`, `clean_model`, `calibrate`, `model_rows`·`line_rows`·`variant_rows`, `qc_summary`·`run_line`, 하네스 `main`, 판정 스크립트. 하네스를 import하면서 생성기·`eval_depth_4tu` 모듈이 import만 됐다.
  - 화면에 본 값: `line_id`, 배열 크기(4665, 161) → 재표본 뒤 (562, 161), dt 0.098 ns, 꼭짓점 트레이스(json 입력값과 같음), 안테나 높이 0.1 m, εr 9, 관별 `velocity_check_note`(K7_P3_D080 관 1만 사유), `model_set_failures` 빈 목록, `input_record` 34파일.
  - 내보내기 json의 꼭짓점 포락선 시각·파형 지연·속도 점검 기울기와 콘솔 요약은 scratch 로그에 쓰였으나 필터로 걸러 화면에 띄우지 않았고, 폴더는 지웠다.
  - 성공률·깊이·후보 위치는 출력하지도 보지도 않았다.
- 이 결함은 판정 입력이 아닌 내보내기 진단 키(`velocity_check`)의 예외 처리다. 기준·문턱·결정표·모델·시드는 바꾸지 않는다.

## 7. 봉인 문구

"이 문서 v1.2(§0 결정 L1–L4, §2-1·§2-2 채택 후보 a1·b2의 정의, §4-1 시험, §4-2 G0, §4-3 모델·역할 집합, §4-4 하네스 규약·시드·변형 집합, §4-5 판정 항목, §4-6 결정표·판정 순서·고정 집합 단언·빈 경우 규칙, §5 위임 계약, §6 봉인 대상·운영, §8 판정서 첫머리 목록)은 K7 새 실현 전체 실행 결과를 보기 전에 고정한다. **결과를 본 뒤 기준·셈법·결정표를 바꾸지 않는다.** 결과 뒤 착상은 새 판(v2)으로 적고 이 판정에 쓰지 않는다. S1g·S4S6(묶음 1·v2·v3)의 h5·npy·시드·runs·판정 결과와 4TU 결과는 K7 판정에 쓰지 않는다(참조 h5 예외 `FS.h5` 제외). K7-c는 진단으로 보고만 하고 판정에 쓰지 않는다. 첫 봉인 `d6aaf0f` 실행(run1)은 내보내기에서 멈춰 판정 결과가 없으며, 그 h5는 재실행 h5와의 배열 대조에만 쓰고 판정에 쓰지 않는다."

## 8. 판정서 첫머리에 적을 것 — 하나라도 없거나 맞지 않으면 결정 순위 1(보류)

- 단, 공개 항목은 예외다. 항목 10의 대조 결과(같음/다름)·영점 값, 항목 11의 셀 목록, 항목 13 감사 기록 표의 내용, 항목 15의 c 진단 값, 항목 17의 영향 기술, 항목 18의 이력 기술, 항목 19의 대조 결과는 **값이 무엇이든 순위 1 사유가 아니다.** 그 항목이 판정서에서 **빠지면** 순위 1이다.

1. 봉인 커밋 해시와 파일별 blob SHA-1: `K7_SEALED_FILES` 목록과 EM 폴더의 봉인 대상(§6). `git rev-parse <커밋>:<경로>`, LF 기준. CRLF 작업 사본은 blob으로만 대조한다.
2. 판정 인자 `--expected-seal-commit` = 봉인 커밋
3. `git hash-object analysis/processing/m210_k7_judge.py` = 봉인 커밋의 같은 파일 blob(판정 스크립트 자기 단언 결과와 수기 대조)
4. 실행 HEAD·dirty 전후(HEAD = 봉인 커밋, dirty 0, 시작·끝 같음)
5. `runs_k7.csv` SHA-256(run_info 기록 = 판정 때 재계산)
6. 시각 순서: 봉인 커밋 < gprMax 시작(러너 로그 시각, 수기 대조) < 하네스 시작(판정 스크립트 단언)
7. 시드 겹침 0(§4-4의 집합 전부)
8. G0 결과: 4TU 두 csv SHA-256 = `1afd23cf…`·`485d4c22…`(작업 트리 CRLF 바이트 기준, §4-2), `g0_s1g` = `identical`과 `stored_only_columns`, (b) v3 부분 집합 행 동일, 생성기 새 blob, 하네스 시작 자식 pytest의 실행 시점 통과·skip 수
9. 고정 집합 단언 결과: 기대 모델 8, 역할 집합, 변형 4, ρ 2, 시드 40, 관 수, 보정 짝, 판정 항목별 대상 셀 수
10. h5 출처와 source_guard snapshot ↔ verify 대조 결과(참조 h5 6개 + 모델 h5 8개). `FS.h5` SHA-256은 `ae2c3844077f9ba566c443a7215a8f911f1950261386c0a1597b1b599f943bfa`([`m210b_v3_judgment.md`](m210b_v3_judgment.md) 첫머리 참조 h5 표, 묶음 1)와 같아야 한다. §6-1 4의 배열 비트 대조 결과(S001–S003, 같음/다름)와 파일 SHA-256(기록), 점검 영점(§4-3)과 전체 실행 영점 — 대조 결과·영점 값은 공개 항목
11. K-F2 자명 통과 셀 목록, K-F4 해당 없음 셀 목록 — 공개 항목
12. `run_info.protocol` = `"k7"`, `sealed_files` ⊇ `K7_SEALED_FILES`, `mkl_cbwr` 값
13. 실행 전 감사 기록(감사 원문 요지, 번호대로, 항목마다 "판정서에 적을 것") — 공개 항목
14. 메인 세션 정체성 기록 파일 이름과, 하네스의 `pre_run.txt`·`post_run.txt`를 메인 세션이 쓰지 않았다는 확인(파일 내용이 하네스 형식인지)
15. c 진단 보고(K-F6의 c 열 분포) — 공개 항목
16. 결과 전 결정 기록(§9)의 마지막 행 시각(KST) < 봉인 커밋 시각(`git log -1 --format=%cI <봉인 커밋>`)
17. §4-3 판정 해석 주의 세 가지(PVC +0.22 ns, K7_P1_D060 ≈ S1G_C1, 층 사건 `ambiguous_with`)와 그 영향 — 공개 항목
18. 첫 봉인 대체 이력(§6-2): `d6aaf0f`, 실패 단계(내보내기)와 메시지, 하네스·판정 미실행(판정 결과 없음), v1.2 재봉인 커밋, v1.2에서 blob이 바뀐 봉인 파일 목록 — 공개 항목
19. run1 재현 대조(§6-1 4의 하위 항목): run1 h5 13개 ↔ 재실행 h5 배열 비트 비교 결과(같음/다름)와 양쪽 파일 SHA-256(기록) — 공개 항목

## 9. 결과 전 결정 기록

봉인 전에 기준·모델·하네스를 바꾸면 이 표에 한 줄씩 더한다. 그때 본 자료를 함께 적는다. 봉인 뒤에는 더하지 않는다. 시각은 KST(HH:MM)이고 근거는 파일 mtime 또는 작업 기록이다. **봉인 직전 마지막 수정 행의 시각까지 채운 뒤** 봉인한다(§8 항목 16).

| 시각(2026-11-12 KST) | 결정·변경 | 결정 때 본 자료 | K7 판정 실현 결과를 봤는가 |
|---|---|---|---|
| 10:47–11:05 v0.1 작성(시제품 실행 10:47–11:03, 문서 마지막 쓰기 11:05:32) | 기작 K7-a·b·c 정리, 후보 a1·a2·b1–b4·c1 제안, 질문 Q1–Q5 | S1g(이미 본 자료), 4TU 550측선 재생성(탐색), 순수 함수 MC(§1–§2, `m210_k7_probe.json`·`m210_k7_rank_replay.json`) | 아니오(실현 없음) |
| 11:12 사용자 결정(CHECKLIST mtime 11:12:23) | L1 a1, L2 b2(a1 위, a1 → a1b2), L3 c 진단만(c 문턱 미사용), L4 새 gprMax 8모델 + 시드 40 | 위와 같음 | 아니오 |
| 11:14–11:20 EM 입력 첫판(`.in` 11:14:30, `run_k7.cmd` 11:14:57), 점검 실행 시작 11:15:15 | 모델 8개·공유 참조 S001–S003 입력과 점검 실행 | 명세 v0.1·결정 L1–L4 | 아니오 |
| 11:13–12:02 v1.0 작성(v0.1 사본 11:13:39, 꼬리 초안 11:17:32; 끝 시각은 파일 기록이 없어 상한 = v1.1 첫 편집 12:02:50으로 적음) | 결정 반영, 봉인 문구·봉인 대상·운영 순서·첫머리 목록·빈 경우·고정 집합 추가 | 위와 같음 | 아니오 |
| 11:20 층 참조 입력 추가(`K7_REF_L085_S001`·`K7_REF_L130_S001` `.in` 11:20:11) | 첫판(11:14:30)에 없던 층 참조 둘을 명세 v1.0 알림("모델마다 같은 층 구조의 `_bg`")에 맞춰 추가. 내용은 모델 정의 값 그대로. 다른 `.in` 19개와 `run_k7.cmd`는 11:14:30·11:14:57 그대로. 이 두 참조는 아직 실행된 적 없음(그래서 §6-1 4의 비트 대조 대상이 아님) | 명세 v1.0 알림. 점검 출력은 열지 않음(첫 점검 h5 읽기는 11:30경 사건 목록 스크립트) | 아니오 |
| 12:00 `models_k7.json`(12:00:14; 그 뒤 S003 점검 실측 반영·`role_sets.source` 수정) | 사건 목록·역할 집합·참조 대응 확정 | EM 점검 실행(영점·사건 시각·기하) | 아니오 |
| 12:01 하네스 `m210_k7_pipeline.py`(12:01:05) | 구현 | – | 아니오 |
| 12:03 v1.1: 파일 이름 | 하네스·판정·시험·내보내기·출력 이름을 구현 이름으로(§4-4·§5) | python-expert 구현 보고(pytest 1085 passed, G0 확인) | 아니오 |
| 12:03 v1.1: 봉인 목록 | `K7_SEALED_FILES`·`K7_SEALED_FOLDERS`로 §6 목록 교체, 상속 파일 4개 추가(`m210b_s4s6_pipeline.py`·`eval_depth_4tu.py`·`test_m210b_s4s6_pipeline.py`·`test_synthetic_export.py`) | 위와 같음 | 아니오 |
| 12:03 v1.1: 구현 해석 | BG 보정 전체 짝, 관 없는 행 수 max(관 수, 1), primary 고르기 순서, `nms` 묶음·`lost_reason_detail`, `n_window_starts`, b2 동률 = 피크 순서, S1g G0 열 단위 | 위와 같음 | 아니오 |
| 12:03 v1.1: 행 수 정정 | 잡음 행 3,840, 잡음 없는 행 48(v1.0 "2,560 + 32" 정정; 구현 보고의 "44"는 K7_BG_L 행을 빠뜨린 셈이다. 하네스 `model_rows`는 관 없는 모델도 변형마다 1행을 내고, 판정 단언 `noise_free_rows_*` = max(관 수, 1)이라 48이다) | 하네스·판정 스크립트 코드 | 아니오 |
| 12:03 v1.1: EM 산출 | §4-3에 경로, 공유 참조 h5 5개 + FS 재사용(참조 h5 예외), 점검 영점 S001 3.6162·S002 3.6156·S003 3.6151 ns, 해석 주의 셋 | EM 점검 실행(영점·사건 시각·기하, 성공률 아님) | 아니오 |
| 12:03 v1.1: 상속 변경 공개 | 생성기 blob 변경, `test_m210b_s4s6_pipeline.py` 수정, `export_dataset.py`·`test_synthetic_export.py` 변경(v1.1 당시 값; 현재 합계는 §4-2)으로 (b) 봉인 파일이 작업 트리에서 바뀜 — 그 판정은 각 봉인 커밋에서 재현(§4-2) | python-expert 구현 보고 | 아니오 |
| 12:13 v1.1: 사건 붙임 | §4-4 붙임 문구를 `attach_k7_event`로: 최상위 `expected_events`(모델별 사본 불일치 = 시작 가드 실패), `K7_EVENT_KINDS` 7종(모르는 종류 오류, `x_m` null은 `flat_layer`만), `flat_layer` $\lvert\Delta x\rvert=0$, 관 정답 = `primary`만(`pvc_internal`은 알려진 사건, U 아님), `ambiguous_with` 기록만 | python-expert 마른 점검(실제 json 구조, 결과 없음) | 아니오 |
| 12:29 v1.1: 실행 전 감사·무결성 지적 반영 | §9 시각, 상속 변경 4파일, 판 표시 v1.1, 순위 SNR 위치(끝점의 $D$, 현행과 같음), §8 항목 10 대조 기준(`FS.h5` SHA)·모델 h5 snapshot·비트 대조 S001–S003만, 비교 명령, 관 수 = `pipes` 길이, a1 동률 늦은 쪽, $\tau\le0$ 대체, export 시험 4개, pytest 수 기록 방식, 홈 경로 제거, CSV CRLF 기준, 구현 해석 (a)–(i) 표지, 메모리 쓰기 금지, 인용 출처, 고정 집합 단언 2개 추가(`models_k7.json` SHA, 관별 행 수) — 기준·문턱·결정표 불변 | 실행 전 감사·무결성 점검 지적, 생성기 코드(`climb_to_local_max`·`tau2_vertex_trace`·`accepted` 행 `snr`) | 아니오 |
| 12:48 첫 봉인 `d6aaf0f`(12:48:54) | v1.1 봉인 | 위 행들 | 아니오 |
| 12:50–14:26 run1 gprMax 13단계 | 모두 exit 0. 레인 B 12:59–14:00 conda `reg.exe` Suspended → 사용자 승인 뒤 재개. `K7_REF_S001`–`S003` 점검 산출과 배열 비트 동일 | 러너 로그·`h5_compare` 결과 | 아니오(h5만, 성공률 없음) |
| 14:27 run1 내보내기 실패(14:27:00–14:27:41) | `K7_P3_D080 pipe 1: no usable per-trace picks for the velocity check`. P2 두 모델만 내보내짐. 하네스·판정 미실행 | `export_k7.err` | 아니오(판정 결과 없음) |
| 14:36 이전 사용자 결정(정확한 시각 기록 없음; 상한 = 내보내기 수정 mtime 14:36:56, 이 행 기록 14:37) | python-expert가 내보내기를 고침(속도 점검이 판정 입력이 아니면 점 0개 관은 `velocity_check: null` + 사유로 기록하고 계속, 보관 h5로 마른 내보내기를 끝까지 돌려 다른 실패 확인), 명세 v1.2 재봉인, gprMax 재실행, run1 산출은 `run1_d6aaf0f/`에 보관 | run1 실패 메시지, 보관 h5 목록 | 아니오 |
| 14:37 v1.2 틀 작성 | 판 표시, §4-3 속도 점검 줄(키 이름 확정 대기), §6 재봉인 대상, §6-1 run1 보관·run1 재현 대조, §6-2 대체 이력, §7 봉인 문구, §8 항목 18·19, 이 행들 — 기준·문턱·결정표·모델·시드 불변 | 위와 같음, 하네스·판정·로더의 `velocity_check` 사용 여부 코드 검색 | 아니오 |
| 14:59 v1.2 마무리(봉인 대상 확정) | 사유 키 `velocity_check_note` = `"no_usable_picks_neighbours"`(이웃 관이 있고 쓸 점 0일 때만, 관 하나에 점 0이면 여전히 오류), 바뀐 봉인 파일 2개(`export_dataset.py` blob `00069282…`, `test_synthetic_export.py` 시험 2개 추가), 내보내기 `MKL_CBWR=COMPATIBLE`. 보관 run1 h5로 마른 내보내기(출력 scratchpad, 삭제): 8모델 모두 내보내짐, K7_P3_D080 관 1만 null + 사유, 하네스 쪽 읽기(고정 집합 단언·`load_synthetic`·관 정답·꼭짓점 트레이스·사건 두 벌·`input_record` 34파일) 통과, 다른 실패 없음. 하네스 본실행·판정 미실행 — 기준·문턱·결정표·모델·시드 불변 | python-expert 수정·마른 내보내기 보고(pytest 1099 passed, 1 skipped; 기존 내보내기 바이트 대조) | 아니오(성공률·깊이 없음) |
| 15:02 v1.2 재봉인 직전 문구 정정 | export K7 시험 수 6개·줄 수(+197/−17, +195/−0), 봉인판 표시 v1.2, 콘솔 요약 `vg/vt=n/a (no usable picks)`, 사용자 결정 행 시각 구분, 하네스·판정 판 문자열 v1.1 유지 사유 — 기준·문턱·결정표·모델·시드 불변 | 무결성 점검·실행 전 감사 지적, `git diff d6aaf0f~1 --numstat`, `tests/test_synthetic_export.py` 시험 이름. 같은 시각에 §6-2에 마른 내보내기에서 본 범위(부른 함수·화면에 본 값, 성공률·깊이·후보 위치는 안 봄)를 공개 | 아니오 |

- **시제품 출력 시각 공개**: `analysis/processing/m210_k7_probe.py`의 mtime(11:05:44)이 출력 `docs/processing/m210_k7_probe.json`(10:59:12)보다 늦다. json을 만든 전체 실행(10:52–10:59) 뒤 이 스크립트에 한 변경은 **모듈 머리말 문서 문자열 추가뿐**이다("D. Prototype variants …" 문단과 `K7_PROBE_LINES` 설명). 코드·상수·출력 키는 바꾸지 않았으므로 json 수치에는 영향이 없다(gpr-processing-expert 작업 기록). `m210_k7_rank_replay.py`(11:00:45)는 그 json(11:03:00)보다 먼저다. 새 생성기에서 시제품을 다시 돌려 확인하는 것은 불가능하다(§10, 752e783 생성기 사본 필요).
- EM 점검 실행(봉인 전)은 모델 확인(진폭·사건 시각·넣는 조건)에만 쓴다. 점검 실행으로 성공률을 재지 않는다. 점검 결과로 모델·기준을 바꾸면 그 사실과 본 값(예: $A_0$, 사건 시각)을 이 표에 적는다((b) v3 실행 전 감사 8과 같은 공개).

## 10. 가정·한계

- 2D 합성(원통 확산)이라 진폭에 기대는 단계(SNR 순위, 정지 0.2, 검출 정규화)는 정성 판단만 할 수 있다. b2는 SNR에 기대므로 이 한계가 특히 크다.
- $\ell$이 S1g 꼭짓점 행에서 오르는 원인 사건은 확정하지 않았다(가정). T2와 K7_P2_D070_L은 원인과 무관하게 $\ell$ 기울기를 만든다.
- c1이 C1을 나쁘게 한 이유(a1 없이)는 추정이다.
- 4TU 수치는 이미 본 자료의 탐색이다. 550측선(재생성)·541측선(수락 후보가 있는 측선, 오프라인 재생)은 m210d 생성기 측선(정답 있는 활동)이고 759측선 전체가 아니다.
- 이 문서의 MC 시드(20261113 + n + 1000ρ + ⌊100d⌋, n < 200)는 수치 범위가 v3 하네스 시드(20281120 + …)와 겹친다. MC는 순수 함수 개발 근거라 판정 실현에 영향이 없지만, K7 하네스의 겹침 단언에는 둘 다 넣는다(§4-4 시드 단언).
- 몬테카를로 잡음은 대역 통과 백색 잡음이고 실측 클러터가 아니다. 시작 트레이스는 꼭짓점 ±10 무작위 하나다.
- 깊이 기준점은 관 상단으로 본다(합성 json `pipe_top_depth_m`). 4TU 재평가는 상단·중심 두 가정을 모두 보고한다.
- 시제품 `m210_k7_probe.py`·`m210_k7_rank_replay.py`는 생성기를 프로세스 안에서 바꿔 끼우는 방식이라, 플래그가 들어간 새 생성기에서는 그대로 재현되지 않는다. 재현하려면 752e783의 생성기 사본(blob `7cc2bf2b…`)을 함께 둔다. 그 결과는 결정 근거로만 봉인되고 판정에 쓰지 않는다.
- K7-c를 고치지 않으므로, 꼭짓점 4 트레이스 안 정지 비율((e) 출력)과 시작점 수 의존은 이번 판에서 그대로 남는다. c 진단 열이 다음 판의 근거가 된다.
- b2는 4TU kept의 약 32 %를 바꾼다. 합성 판정이 채택 가능이어도 4TU에서 좋아진다는 보장은 없다. 4TU는 채택 뒤 탐색 보고로만 본다.
