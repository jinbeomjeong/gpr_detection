# M2-10 (c) QC P3–P5 — 명세 (v1.2, 봉인 판정 생략)

- 판: **v1.2**, 2026-11-13, gpr-processing-expert.
  - v1.2: 구현 결과 반영(결정 C6–C9 불변). P3 판정 함수 $C_1>C_0$의 한계와 플래그의 실제 뜻(§1-3), 띠 이동의 두 정의(상호상관 11–12 % 대 포락선 피크 36–41 %)와 참고 배율 2.5의 기준(§1-1), 순잡음 시드 20311199(§4-2), 실제 이름(§5·§7), G0 결과(§4-3), 4TU 진단 결과 대 예측(§1-4).
  - v1.1: 사용자 결정 C6–C9. P3 보정을 보류하고 플래그·진단만 남김(깊이 불변). 봉인 판정 생략(C5 철회). 합성은 플래그 동작 보고. 4TU는 진단 csv만. 띠 약 40 %·관 약 100 % 물리 사실 기록.
  - v1.0: C1–C5·R1·R2. P3 보정 + 봉인 판정 설계(대체됨, 이력은 §11·§12).
  - v0.1: 설계 초안.
- 범위: CHECKLIST M2-10 (c).
  - P3: 이동 정합 QC 플래그·진단(`zero_shift_large`·`zero_tracks_surface`·`zero_shift_not_surface`). **영점 보정은 하지 않는다.**
  - P4: `h_scan_qc` — 구현 보류(§2)
  - P5: `eps_meta_equals_header_default` 플래그(§3)
- 출처: 제안 원문 [`fourtu_explore.md`](fourtu_explore.md) §1 표 3·§5·§8([FX]). 상태·계획 [`spec_depth_v1.md`](spec_depth_v1.md) §4-1·§4-4·§7·§9·§14. M2-9 한계 [`depth_4tu_v2.md`](depth_4tu_v2.md) §8 한계 5. [DP] = [`depth_physics.md`](../research/depth_physics.md). (a) 결과 [`m210_a_eps_bias.md`](m210_a_eps_bias.md). 점검 물리 [`m210c_pilot_physics.json`](m210c_pilot_physics.json).
- 자료 규율
  - 4TU는 이미 본 자료다. 4TU에서 센 수는 탐색이고, 규칙·문턱을 4TU로 고르지 않는다.
  - 이번 판의 모든 변경은 깊이 값과 판정에 영향을 주지 않는다. 기본값(모든 새 옵션 꺼짐)에서 기존 출력이 바이트 동일해야 한다(§4-3 G0).
- 설계 근거 스캔(기존 csv만): [`analysis/processing/m210c_design_scan.py`](../../analysis/processing/m210c_design_scan.py) → [`m210c_design_scan.json`](m210c_design_scan.json).

## 0. 결정

### 0-1. 현행 결정 (v1.1)

| ID | 결정 | 내용 | 반영 |
|---|---|---|---|
| C2 | P4 구현 보류 | (a) 결과와 E1(공기층–εr 퇴화)로 갈음하고 문서에 기록 | §2 |
| C3 | P5 플래그만 | `eps_meta_equals_header_default`, $\sigma_\varepsilon/\varepsilon$ 0.15 유지, 로더에 `eps_r_header` 필드 | §3 |
| C6 | P3 = (다) | 영점 보정은 보류하고 플래그·진단만 남긴다(깊이 불변, C1 축소). 띠 약 40 %·관 약 100 % 물리 사실을 §1-1과 [DP] 기록 대상에 명시 | §1 |
| C7 | 봉인 판정 생략 | C5 철회. 단위 시험·G0·합성 플래그 동작 보고로 끝낸다. 무결성 점검은 거친다 | §4, §8 |
| C8 | gprMax 전체 실행 | 명세대로 실행한다. 점검 산출은 `pilot_prerun_h5/`로 옮기고 전체 실행 h5와 배열 비트 비교 | §4-1, §8 |
| C9 | 4TU 진단 csv만 | 현장별 P3·P5 플래그 수, $\sigma_s$ 분포, 참고값 "띠 std × 약 2.5 = 관 시각 변화 추정"(깊이·판정에 쓰지 않음). 판정률 재계산 없음 | §7 |

### 0-2. 대체·철회된 결정 (기록)

| ID | 원래 결정(2026-11-12 23:4x KST) | 지금 |
|---|---|---|
| C1 | P3 플래그 + 꼭짓점 트레이스 영점 보정, 합성 시험 먼저 | C6으로 축소(보정 없음) |
| C4 | 구현 뒤 4TU 재평가 1회(판정률 포함) | C9로 대체(진단 csv만) |
| C5 | 보정 채택은 봉인 판정, 자동 진행 | C7로 철회 |
| R1 | Z-F1 문턱 1.0 cm(2026-11-13 00:0x) | 판정이 없으므로 쓰지 않음 |
| R2 | 효과 2 + 재사용 대조 2, Z_TOPO 없음(2026-11-13 00:0x) | 모델 구성은 플래그 동작 보고(§4-1)에 그대로 씀 |

## 1. P3 — 이동 정합 QC (플래그·진단)

### 1-1. 물리 (C6: [DP] 기록 대상)

공기결합 안테나 높이가 측선 안에서 바뀌면(수레 흔들림, 노면 단차) 공기 왕복 경로가 $\Delta t_j=2\Delta h_j/c$만큼 바뀐다. 측선 하나에 영점 하나를 쓰는 지금 규칙에서는, 공기층이 평균보다 두꺼운 곳의 관은 깊게, 얇은 곳의 관은 얕게 나온다($\Delta d_j=v\,\Delta t_j/2$).

**점검 실행으로 확인한 사실**(EM 점검 h5, 2D TMz, Ricker 500 MHz, 오프셋 0.10 m, εr 9, $h$ 0.05–0.15 m; [`m210c_pilot_physics.json`](m210c_pilot_physics.json)과 EM 보고):

| 양 | 값 | 기하 기대 대비 |
|---|---|---|
| 합쳐진 띠(직접 결합 + 지표 반사) 포락선 피크의 높이 기울기 | 2.41 ns/m(EM 라이브러리) · 2.65–2.71 ns/m(효과 모델 측선) | **약 36–41 %**($2/c$ = 6.67 ns/m). $h$ 0.14–0.15 m에서 포화 |
| 따로 뗀 지표 반사(배경 − 자유 공간)의 기울기 | 5.61 ns/m | 약 84 % |
| 관 꼭짓점 시각 차(높은 쪽 − 낮은 쪽) | Z_HSTEP 0.513 ns(기하 0.534) · Z_HWAVE 0.639 ns(기하 0.667) | **약 96 %**(EM: 기대와 ±0.005 ns) |
| 측선 띠 피크 std | 0.109 ns(Z_HSTEP) · 0.098 ns(Z_HWAVE) | 0.2 ns 문턱 아래 |
| 보정 없는 두 관 깊이 차(전역 영점) | 2.56 cm · 3.19 cm | 명세 예측 2.7·3.3 cm와 맞음 |

- 뜻: 띠는 직접 결합(안테나 높이와 무관하게 정지)과 지표 반사(공기층 따라 움직임)의 합이라, 띠 피크는 공기층 변화의 약 40 %만 따라간다. 관 반사는 거의 100 % 따라간다. 그래서 **띠 이동을 영점 보정에 그대로 쓰면 관 시각 변화의 약 40 %만 상쇄된다.** 이것이 C6(보정 보류)의 근거다.
- 4TU 참고(해석, 이미 본 자료): 4TU 영점도 대부분 합쳐진 띠다(709/759측선). 띠 std가 $\sigma_s$이면 관 시각 변화의 크기는 대략 $\sigma_s/0.4\approx2.5\,\sigma_s$로 추정할 수 있다. 예를 들어 011.14(띠 std 0.31–0.42 ns)는 관 시각 std 약 0.8–1.0 ns, 현장 11 $v$ 0.077 m/ns에서 깊이 std 약 3–4 cm다. 이 2.5는 2D 합성(위 조건)에서 나온 **참고 배율**이고, 실제 InfraRadar 안테나·오프셋에 맞는다는 근거는 없다. 깊이·판정에 쓰지 않는다(§7).
- **(v1.2) 띠 이동의 두 정의와 2.5의 기준**:
  - 위 36–41 %와 2.5는 **띠 포락선 피크 시각**의 이동 기준이다(EM 라이브러리·[`m210c_pilot_physics.json`](m210c_pilot_physics.json)).
  - §1-3의 $\sigma_s$는 **띠 창 상호상관 지연**이다. 합성 플래그 보고([`m210c_flag_report.json`](m210c_flag_report.json))에서 이 지연은 기하 이동 $2\Delta h/c$의 **약 11–12 %**만 따라갔다(라이브러리 띠 피크 이동의 28–34 %). 효과 모델의 $\sigma_s$ 중앙은 0.033·0.030 ns였다(포락선 피크 std 0.10–0.11 ns보다 작음). 띠 창 안의 에너지 대부분이 정지한 직접 결합이라, 상관은 정지 성분에 끌린다.
  - 그래서 v1.1 §7의 참고값 `pipe_time_std_ref_ns` = $2.5\,\sigma_s$는 **정의가 맞지 않는다**(포락선 기준 배율을 상관 기준 $\sigma_s$에 곱함). 같은 정의로 맞추면 배율은 약 $1/0.115\approx8$–9가 된다. 그러나 이를 4TU 현장 9($\sigma_s$ 중앙 0.40 ns)에 곱하면 관 시각 std가 3 ns를 넘어 공기층 변화로는 비현실적이다. 4TU의 큰 $\sigma_s$에는 공기층 말고 다른 원인이 섞여 있다(§1-4: 두 사건 측선). 결론적으로 이 참고값은 **관 시각 변화의 크기로 읽을 수 없다.** 열은 진단 csv에 그대로 있지만(결정 C9 불변), 해석에 쓰지 않는다. 다시 정의하는 일은 다음 항목 후보로만 기록한다(§12).
- [DP] 반영 요청: 위 표와 "합쳐진 띠는 공기층 변화의 약 40 %만 따라감(2D 합성, 조건부)"을 [DP] §7-2에 더하는 일은 electromagnetic-simulation-expert 몫이다(§9 핸드오프).

### 1-2. 지금 상태

- **미구현.** 있는 것은 창별 영점 QC `zero_window_drift`뿐이다(창 사이 영점 시각 std > 0.2 ns, `src/gpr_detection/preprocess/timezero.py` `_window_qc`). `ZeroResult.t_zero_per_trace_ns` 필드는 있지만 늘 `None`이다. 이번 판에서도 채우지 않는다.
- 4TU(이미 본 자료, 설계 스캔): 745측선 중 `zero_window_drift` 26측선(현장 11: 10, 10: 7, 9·12: 3, 5: 2, 7: 1).
- [FX] 표 3: 01·04·010 띠 이동 std 0.07–0.17 ns, 011.14 0.31–0.42 ns. 주 엽은 지표를 따라 움직였다($k=1$ 최적).

### 1-3. 알고리즘 (확정, 진단만)

입력은 처리 (1)–(3)까지 한 부호 있는 자료 $a_j(t)$다(배경 제거 전, 영점 검출과 같은 입력). 트레이스는 영점 검출의 포함 트레이스(`pipe_band` 박스 제외)만 쓴다.

| 기호 | 뜻 | 단위 | 값·출처 |
|---|---|---|---|
| $t_z$ | 지금 규칙의 측선 영점(띠 포락선 피크) | ns | `ZeroResult.t_zero_ns` |
| $W_b$ | 띠 창 $[t_z-0.75/f_c,\ t_z+0.75/f_c]$ | ns | [FX] 표 3의 띠 ±1.5 ns(500 MHz) |
| $L$ | 지연 탐색 범위 $\lvert\ell\rvert\le0.5/f_c$ | ns | 반 주기(엽 건너뜀 방지) |
| $r(t)$ | 기준 트레이스 = 포함 트레이스의 시각별 중앙값 | 진폭 | 배경 제거의 median과 같은 선택 |
| $s_j$ | 트레이스 $j$의 띠 이동 | ns | 아래 1 |
| $\tilde s_j$ | 평균을 뺀 이동 $s_j-\overline{s}$ | ns | 아래 2 |
| $\sigma_s$ | $\mathrm{std}_j(\tilde s_j)$ | ns | – |
| $C_0, C_1$ | 정지·지표 추종 가정의 쌓기 결맞음 | 무차원 | 아래 3 |
| $\tau_\text{drift}$ | 0.2 ns | ns | `ZeroConfig.window_drift_ns`(재사용) |

1. $s_j=\arg\max_{\lvert\ell\rvert\le L}\sum_{t\in W_b}a_j(t+\ell)\,r(t)$. 정수 샘플 지연에서 최대를 찾고 `parabolic_peak`로 정밀화한 뒤 $\Delta t$를 곱한다. 최대가 탐색 끝에 걸리면 그 트레이스는 `nan`(집계에서 뺌).
2. $\tilde s_j=s_j-\overline{s}$, $\overline{s}$ = 포함 트레이스 평균.
3. $C_k=\dfrac{\sum_{t\in W_b}\big(\tfrac1N\sum_j a_j(t+k\tilde s_j)\big)^2}{\sum_{t\in W_b}\tfrac1N\sum_j a_j(t+k\tilde s_j)^2}$, $k\in\{0,1\}$. 소수 샘플 이동은 선형 보간(`np.interp`).
4. 플래그:
   - $\sigma_s\le\tau_\text{drift}$: 플래그 없음.
   - $\sigma_s>\tau_\text{drift}$이고 $C_1>C_0$: `zero_shift_large`, `zero_tracks_surface`.
   - $\sigma_s>\tau_\text{drift}$이고 $C_1\le C_0$: `zero_shift_large`, `zero_shift_not_surface`.
   - **(v1.2) 판정 함수의 한계**: $s_j$를 기준 트레이스와의 상관이 최대인 지연으로 고르므로, $\tilde s_j$만큼 옮겨 쌓으면 정의상 거의 언제나 결맞음이 오른다($C_1>C_0$). 구현 확인(python-expert): 정지 + 이동 섞임, 극성 반전, 잡음만인 띠 모두 $C_1\le C_0$가 나오지 않았다. 4TU에서도 `zero_shift_not_surface`는 0/759측선이다. 시험 T2는 결정 함수에 값을 직접 넣어 분기를 시험했다. 따라서 **`zero_tracks_surface`는 사실상 "$\sigma_s>0.2$ ns(띠가 흔들림)"와 같고, 띠가 지표를 따른다는 확인이 아니다.** 이번 판에서 기준을 바꾸지 않는다(C6 그대로). 지표 추종을 실제로 가리는 기준(예: 상관 기준 트레이스를 자기 자신과 독립인 반쪽 측선에서 만들기, 지연과 띠 진폭 변화의 관계)은 다음 항목 후보로만 기록한다(§12).
5. 진단 필드: `band_shift_std_ns` = $\sigma_s$, `band_coherence` = $(C_0, C_1)$, `band_shift_n_valid`(`nan`이 아닌 트레이스 수). **`t_zero_per_trace_ns`는 채우지 않고, 깊이는 바꾸지 않는다.**
6. 새 상수: 없음(0.75/f_c는 [FX] ±1.5 ns를 주기 단위로, 0.5/f_c·0.2 ns는 기존 값).
7. 기본값: `ZeroConfig.band_shift_qc: bool = False`. 꺼지면 계산하지 않고 기존 출력이 바이트 동일하다. `amplitude_processed`·`dt_ns=None`·`zero_failed`에서는 켜져 있어도 계산하지 않는다.

### 1-4. 사전 예측

- 합성(점검 사실 기준): 효과 모델의 $\sigma_s$는 약 0.10–0.11 ns라 `zero_shift_large`가 걸리지 않을 것이다. 대조 모델은 잡음 수준이다. 곧 합성에서 플래그는 대개 비어 있고, $\sigma_s$·$C_0$·$C_1$ 진단값이 정보다(§4).
- 4TU(예측, v1.1): 측선 대부분은 $\sigma_s\le0.2$ ns라 플래그가 없고, 걸리는 측선은 `zero_window_drift` 26측선 근처(현장 11·10)일 것이다.
- **4TU 결과(v1.2, 이미 본 자료·탐색, [`m210_c_4tu_diag.md`](m210_c_4tu_diag.md))**: `zero_shift_large` = `zero_tracks_surface` **179/759측선**(현장 9: 48/53, 10: 44, 11: 31, 3: 17, 12: 13, 7·13: 9), `zero_shift_not_surface` 0. $\sigma_s$ 10/50/90 % = 0.045/0.124/0.302 ns. `zero_window_drift` 26측선 중 11측선만 겹친다. 예측(26측선 근처)보다 훨씬 많다.
  - 영점 규칙별 $\sigma_s$ 중앙(정정, `m210c_4tu_line_diag.csv`): 전체는 두 사건(`air_first_event`, 50측선) 0.31 ns, 단일 띠(`air_single_event`, 709측선) 0.12 ns다. 그러나 **현장 9 안에서는 두 사건 0.39 ns(33측선), 단일 띠 0.44 ns(20측선)로 단일 띠도 높다.** 현장 9를 빼면 두 사건 0.16 ns(17측선), 단일 띠 0.12 ns(689측선)다. 곧 전체의 두 사건 대 단일 띠 차이는 대부분 현장 9(두 사건 50측선 중 33측선)에서 온 것이고, 현장 9의 높은 $\sigma_s$는 두 사건 띠 때문이 아니다. **원인은 확인하지 않았다**(현장 9의 표본화 간격 혼재 0.098/0.117/0.195 ns, 띠 모양, 실제 안테나 높이 변화 등은 후보일 뿐이다). 어느 경우든 공기층 변화의 증거로 읽지 않는다.
  - §1-3의 한계 때문에 179측선은 "띠가 흔들림"일 뿐 지표 추종의 확인이 아니다.

## 2. P4 — 구현 보류 (C2)

- 하지 않는다. 곡률은 지금처럼 꼭짓점 후보 QC 전용이고, 깊이 곡률 금지는 그대로다([`spec_depth_v1.md`](spec_depth_v1.md) §7).
- 근거(기록):
  - [DP] §7-2 E1: 공기층 $h\le0.1$ m에서는 곡률만으로 "εr 12–17이고 $h\approx0.1$"과 "메타 εr 9이고 $h\approx0$"을 가르지 못한다.
  - (a) §7-5: 현장 배율 $k$와 곡률 비 중앙의 Spearman −0.31(6점, $p$ 0.56), 판단 불가.
  - 자동 후보의 참 꼭짓점 비율이 낮다(약 11 %, E 적용 뒤 약 36 %, [`depth_4tu_v2.md`](depth_4tu_v2.md) §8 한계 2). 이 플래그를 쓰는 규칙도 없다.
  - 4TU 현장별 `in` 후보 곡률 비 중앙은 대부분 0.93–1.07, 현장 9·11이 1.20·1.24다(설계 스캔). 공기층 때문인지 εr 때문인지 가를 수 없다.
- 다시 열 조건: εr을 따로 잰 실측 자료나, 공기층을 아는 4TU 장비 메타가 생길 때.

## 3. P5 — 헤더 기본값 플래그 (C3)

- `Radargram.eps_r_header: float | None = None`을 더한다. 4TU 로더(`src/gpr_detection/io/fourtu.py`)가 측선 json의 `eps_r_header`(SEG-Y 텍스트 헤더의 장비 설정값 7/9/11)를 채운다. 다른 로더는 `None`이다.
- `DepthConfig.flag_header_eps: bool = False`. 켜져 있고 `eps_r_header`와 `eps_r_meta`가 모두 있으며 $\lvert\varepsilon_\text{meta}-\varepsilon_\text{header}\rvert<10^{-6}$이면 `flags`에 `eps_meta_equals_header_default`를 더한다.
- $\sigma_\varepsilon/\varepsilon$는 모든 경우 0.15 그대로다. 근거: (a)에서 헤더 기본값 현장과 실측 εr 현장의 깊이 비가 같았다(0.867 대 0.866, $p$ 0.98). 이 플래그는 출처 투명성 정보다.
- 사전 예측(4TU, 이미 본 자료): 메타 εr 9.0 활동 51개(현장 1: 9, 2: 6, 4: 8, 7: 4, 10: 10, 12: 8, 13: 6) 근처. 실제 수는 `eps_r_header`와 같은지로 다시 센다.

## 4. 합성 — 플래그 동작 보고 (C7·C8, 문턱 판정 없음)

### 4-1. 모델과 실행

| ID | 역할 | 구성 | 관 |
|---|---|---|---|
| Z_HSTEP | 효과 | 평평한 지면, εr 9, σ 0.001. 안테나 높이 $x<2.0$ m 0.06 m, $x\ge2.0$ m 0.14 m | PEC R 0.05, 상단 0.60 m, $x$ 1.2(`low_gap`)·2.8 m(`high_gap`) |
| Z_HWAVE | 효과 | $h(x)=0.10+0.05\sin\big(2\pi(x-1.0)/2.0\big)$ m | PEC R 0.05, 상단 0.60 m, $x$ 1.5(`high_gap`)·2.5 m(`low_gap`) |
| K7_P1_D060 | 대조(재사용) | K7 모델, 평평, $h$ 0.10 | 0.60 m 하나 |
| K7_P2_D070_L | 대조(재사용) | K7 모델, 평평, $h$ 0.10, εr 12 경계 0.85 m | 0.70 m 둘 |

- 전체 실행(C8): 효과 모델 2개와 높이 라이브러리 참조(`Z_REF_H`·`Z_FS_H`)를 명세대로 돌린다. 점검 산출은 전체 실행 전에 `data/synthetic_gprmax_c_qc/pilot_prerun_h5/`로 옮기고, 같은 입력의 h5를 배열 비트 비교한다(§8).
- 대조 2개는 K7 실행 산출(`data/synthetic_gprmax_k7/`)을 그대로 쓴다.

### 4-2. 보고할 것 (판정 없음)

- 스크립트: `analysis/processing/m210c_flag_report.py`(§5 C-7). 잡음 없는 1행 + 잡음 ρ 10·20 × 시드 20(시드 `20311112 + 1000·m + 100·q + s`, m 0–3, q ρ 색인, s 0–19; 이전 시드와 겹침 0 확인). 잡음 규약은 K7과 같다. 순잡음 단면 시드는 **20311199**(v1.1에 빠졌던 값, 구현에서 정함, 겹침 확인 대상에 포함).
- 모델 × ρ마다: $\sigma_s$ 분포(중앙·10/90 %), $C_1-C_0$ 분포, 플래그 비율(`zero_shift_large`·`zero_tracks_surface`·`zero_shift_not_surface`), `band_shift_n_valid`.
- 효과 모델: json 기대 띠 이동(`expected_band_shift_ns`, 기하 $2\Delta h/c$)과 실측 띠 피크(`expected_band_peak_ns`) 대비 $\tilde s_j$의 기울기(따름 비율). 점검 사실(약 36–41 %)과 맞는지.
- 대조 모델: 플래그가 걸리지 않는지(보고).
- 깊이는 계산하지 않는다(보정이 없으므로 대상이 아님).
- 문턱·결정표·봉인 문구·판정서 첫머리 목록: **C7로 생략.** 결과는 보고서로만 남기고, 이 결과로 P3 규칙을 바꾸지 않는다.

### 4-3. 바이트 회귀 (G0)

- 새 옵션(`ZeroConfig.band_shift_qc`, `DepthConfig.flag_header_eps`, 평가 스크립트 진단 옵션)을 모두 끈 기본값에서 다음이 같아야 한다.
  - 4TU 후보 생성 두 csv(`fourtu_apex_candidates_v1.csv` `1afd23cf…`, `fourtu_apex_candidate_lines_v1.csv` `485d4c22…`, 작업 트리 CRLF 바이트 기준)
  - S1g 하네스 기본 경로 `candidates.csv`(열 단위, `cluster_mu` 저장본 열 예외)
  - K7 하네스 기본 경로 부분 집합(모델 1·시드 2·변형 `cur`)
- **(v1.2) G0 결과(python-expert)**: 전체 pytest 1123 passed. 4TU 후보 두 csv 바이트 동일. K7·S1g·(b) v3 부분 집합 동일. M2-9 평가 8 csv는 HEAD 코드로 다시 만든 것과 바이트 동일. 다만 **커밋된** `depth_4tu_v2_matches.csv`·`_summary.csv`·`_site11.csv`·`_dup_sensitivity.csv`는 HEAD 코드로 다시 돌려도 커밋본과 다르다. `MKL_CBWR` 고정 전에 커밋된 것이라 동률 처리 차이로 추정한다(이번 변경과 무관, 원인 미확정).
  - `ZeroResult`·`DepthResult`의 기존 필드 값(새 필드는 `None`)

## 5. python-expert 위임 명세 (축소판)

| # | 대상 | 변경 | 계약 |
|---|---|---|---|
| C-1 | `src/gpr_detection/preprocess/timezero.py` | `ZeroConfig.band_shift_qc: bool = False`. 함수 `band_shift_qc(filtered, dt_ns, f_c_hz, t_zero_ns, included_mask, config=DEFAULT_ZERO_CONFIG) -> BandShift`(§1-3 1–4: $\tilde s_j$ ns 배열, $\sigma_s$, $C_0$, $C_1$, 유효 트레이스 수, 플래그). `select_zero`/`find_time_zero`가 켜졌을 때만 호출해 플래그와 새 필드(`band_shift_std_ns`, `band_coherence`, `band_shift_n_valid`)를 채움 | 꺼지면 계산 안 함. `t_zero_per_trace_ns`는 채우지 않음. `amplitude_processed`·`dt_ns=None`·`zero_failed`에서는 호출 안 함. 깊이 불변 |
| C-2 | `src/gpr_detection/io/radargram.py`·`io/fourtu.py` | `Radargram.eps_r_header: float \| None = None`, 4TU 로더가 json `eps_r_header`를 채움 | 다른 로더 불변 |
| C-3 | `src/gpr_detection/depth/config.py`·`estimate.py` | `DepthConfig.flag_header_eps: bool = False`, §3 플래그 | 꺼지면 불변. 깊이·$\sigma$ 불변 |
| C-4 | `scripts/eval_depth_4tu.py` | 진단 옵션 `--band-shift-qc`, `--flag-header-eps`, `--diag-out <폴더>`, `--prefix <이름>`(출력 `<prefix>_line_diag.csv`). 켜면 측선 진단 csv(§7 열)만 따로 쓴다. 판정·매칭·요약 출력은 만들지 않거나, 만들더라도 기본 산출물을 덮어쓰지 않음 | 기본 실행 출력 바이트 동일 |
| C-5 | `scripts/synthetic/export_dataset.py` | `--c-qc analysis/synthetic/c_qc/models_c_qc.json` → `data/synthetic_gprmax_c_qc/`. EM json 필드(§6)를 내보낸 json에 옮김. 트레이스별 참조는 높이 라이브러리 색인으로 고름 | 기존 내보내기 바이트 동일 |
| C-6 | (삭제) | v1.0의 깊이 보정(`estimate.py` $\delta_a$·`zero_per_trace_applied`·`zero_shift_apex_ns`), 봉인 하네스(`m210c_zero_pipeline.py`)·판정 스크립트(`m210c_zero_judge.py`)는 **만들지 않는다** | – |
| C-7 | `analysis/processing/m210c_flag_report.py`(새) | §4-2 보고 스크립트(읽기 전용, 판정 없음). 출력 `docs/processing/m210c_flag_report.json`·`.csv`(LF, 상대 경로) | 시드 겹침 0 확인을 출력에 기록 |
| C-8 | 시험 | `tests/test_m210c_band_shift.py`(T1–T5), 로더·플래그 시험(T6), G0 시험 | – |

단위 시험(T):

| ID | 고정물 | 기대 |
|---|---|---|
| T1 | 합성 띠(Ricker 500 MHz) 트레이스별 이동 계단, 잡음 없음. 계단 0.38 ns와 0.44 ns(반반) | $\tilde s_j$ = ±계단/2(± 0.01 ns). $\sigma_s$ 0.19 ns → 플래그 없음, 0.22 ns → `zero_shift_large` |
| T2 | 정지 띠와 움직이는 띠가 섞여 $C_1\le C_0$ | `zero_shift_not_surface` |
| T3 | 이동이 반 주기를 넘는 트레이스 | 탐색 끝 → `nan`, 집계에서 빠짐, `band_shift_n_valid` 감소 |
| T4 | `band_shift_qc=True`인데 `zero_failed`·`amplitude_processed` | 계산 안 함, 새 필드 `None` |
| T5 | `band_shift_qc=False` | 기존 `ZeroResult`·`DepthResult` 필드 값 동일, 깊이 동일 |
| T6 | `eps_r_header` = `eps_r_meta`(9.0)·다름·`None` | 플래그는 첫 경우에만, 깊이·$\sigma$ 불변 |

## 6. electromagnetic-simulation-expert 위임 명세

- 폴더: 입력·모델 json `analysis/synthetic/c_qc/`(`models_c_qc.json`, `inputs/`, 실행 `run_cqc.cmd`), h5 `data/synthetic_gprmax_c_qc/`(점검 `pilot/`, 전체 `h5/`).
- 모델: §4-1 Z_HSTEP·Z_HWAVE. 안테나 높이는 트레이스별 모델(`#python`과 `current_model_run`으로 Tx·Rx 높이 설정, `--geometry-fixed` 없음)로 바꾼다.
- 참조: 높이 라이브러리 `Z_REF_H`(배경)·`Z_FS_H`(자유 공간) 21개. 트레이스 $j$의 참조 색인 = $\mathrm{round}\big((h_j-0.05)/0.005\big)$.
- json 필드(모델별): `antenna_height_m`(트레이스별), `antenna_height_mean_m`, `expected_band_shift_ns`(기하 $2(h_j-\bar h)/c$), `expected_band_peak_ns`(라이브러리 실측 띠 피크), `expected_surface_reflection_ns`, `pipes[]`(`side` = `high_gap`/`low_gap`, `apex_trace_index`, `antenna_height_at_apex_m`, `top_depth_below_surface_m`), `bg_ref`/`fs_ref`(라이브러리와 색인 규칙), `height_rule`, `height_snapped_to_cell_m`.
- 전체 실행: 점검과 같은 입력. 점검 h5와 배열 비트 비교(§8).
- [DP] §7-2에 §1-1 표의 물리 사실(합쳐진 띠 약 36–41 %, 분리 지표 반사 약 84 %, 관 약 96–100 %, 포화 0.14–0.15 m, 조건)을 더한다(§9).

## 7. 4TU 진단 csv (C9, 이미 본 자료·탐색, 판정 아님)

- 실행: `scripts/eval_depth_4tu.py --band-shift-qc --flag-header-eps --diag-out data/processing/m210c_4tu_diag --prefix m210c_4tu`(`MKL_CBWR=COMPATIBLE`) → `data/processing/m210c_4tu_diag/m210c_4tu_line_diag.csv`. 요약은 `analysis/processing/m210c_4tu_diag_summary.py` → `docs/processing/m210_c_4tu_diag.md`. 기본 산출물은 건드리지 않는다.
- 측선 진단 csv 열: `activity_id`, `path`, `site`, `zero_rule`, `band_shift_std_ns`, `band_c0`, `band_c1`, `band_shift_n_valid`, P3 플래그 3종, `zero_window_drift`, `eps_r_meta`, `eps_r_header`, `eps_meta_equals_header_default`, 참고값 `pipe_time_std_ref_ns` = $2.5\,\sigma_s$(v1.2: 정의 불일치로 해석에 쓰지 않음, §1-1).
- 요약 md `docs/processing/m210_c_4tu_diag.md`(머리에 "이미 본 자료·탐색 보고, 판정 아님"):
  - 현장별 P3 플래그 수와 $\sigma_s$ 분포, `zero_window_drift`와의 겹침
  - 현장별 P5 플래그 수(설계 스캔 예측 51활동 근처와 대조)
  - 참고: $2.5\,\sigma_s$와 현장 $v$로 본 깊이 std 추정. **2.5는 2D 합성 조건(§1-1)의 참고 배율이고, 깊이·판정·규칙에 쓰지 않는다.**
- 판정률·MAE는 다시 계산하지 않는다(깊이가 같으므로).

## 8. 운영 순서

1. 사용자 결정 C6–C9 기록(§0·§11, 완료).
2. 병행 진행: EM(전체 실행 준비)과 python-expert(C-1–C-8).
3. EM 실행 운영:
   - 러너 로그 폴더 `data/synthetic_gprmax_c_qc/runner_logs/`를 gprMax 전에 만든다.
   - 점검 산출(`data/synthetic_gprmax_c_qc/pilot/`의 h5·`.log`, `pilot_run.log`)을 전체 실행 전에 `data/synthetic_gprmax_c_qc/pilot_prerun_h5/`로 **이동**한다(mtime 유지).
   - gprMax 전체 실행(GPU 한 계열만).
   - 같은 입력 h5를 `~/.claude/tools/gprmax/h5_compare.py <pilot_prerun_h5 안의 h5> <새 h5> --expect bit --all-datasets --format json --out <scratchpad 보고서>`로 배열 비트 비교. 파일 SHA는 기록만.
4. 내보내기(`MKL_CBWR=COMPATIBLE`) → 보고 스크립트(§4-2) → 4TU 진단 csv(§7).
5. 무결성 점검(정의↔코드↔이 문서) → 커밋. 봉인·실행 전 감사·판정 후 감사는 C7로 하지 않는다.
6. 보고 결과로 P3·P5 규칙이나 문턱을 바꾸지 않는다. 바꾸려면 새 판과 사용자 결정이 필요하다.

## 9. 핸드오프

- electromagnetic-simulation-expert: [DP] §7-2에 §1-1 물리 사실 추가. §6 전체 실행.
- python-expert: §5 C-1–C-8.
- 메인 세션: 4TU 진단 실행(§7)과 커밋.

## 10. 공개할 점 (판정은 없지만 이력으로)

- v1.0(봉인 판정 계획) 아래에서 봉인 전 점검을 하던 중, gpr-processing-expert가 점검 h5를 읽기 전용으로 분석했다(`analysis/processing/m210c_pilot_physics.py`). 이때 관 시각 차와 띠 이동 차가 계산되어, **잡음 없는 보정 잔차(약 1.5–1.9 cm)가 산술로 드러났다.** 이 값은 v1.0 §4-4가 봉인 전에 보지 말라고 한 `cand` 쪽 양(Z-F1 대상)의 잡음 없는 근사다. EM 보고의 "띠 36 %"로도 유추할 수 있었다.
- 이 노출 뒤 사용자 결정 C6·C7로 보정과 봉인 판정을 함께 거뒀다. 그래서 판정 오염 문제는 남지 않는다. 다만 C6 결정이 이 물리 사실을 본 뒤 내려졌다는 점을 기록한다.

## 11. 결정 기록 (시각 KST, 표시 없으면 2026-11-12)

| 시각 | 결정·변경 | 결정 때 본 자료 |
|---|---|---|
| v0.1 작성 | 설계 초안, 질문 Q1–Q4 | 설계 스캔(4TU 기존 csv 셈), (a) 결과 |
| 23:4x 사용자 결정(메인 세션 기록, CHECKLIST mtime 23:54 이전) | C1–C4 | v0.1 |
| 23:4x 사용자 결정(같은 무렵) | C5(봉인 판정, 자동 진행 범위) | v0.1 |
| 23:56 v1.0 작성 | 보정 알고리즘·합성 시험(봉인 판정)·위임·운영 | 위와 같음, 코드 |
| 2026-11-13 00:0x 사용자 결정 | R1(Z-F1 1.0 cm), R2(효과 2 + 대조 2) | v1.0 질문 |
| 2026-11-13 00:03 R1·R2 반영 | v1.0 문구 | 위와 같음 |
| 2026-11-13 00:2x–00:3x EM 점검 실행 보고 | 띠 약 36 %, 관 ±0.005 ns, 폴더 이름 `c_qc` | 점검 h5(EM) |
| 2026-11-13 00:37 점검 물리 분석(gpr-processing-expert, `m210c_pilot_physics.json`) | 전역 영점 짝 차 2.56·3.19 cm, 띠 std 0.10–0.11 ns, 보정 잔차 산술 노출(§10) | 점검 h5(raw Ez) |
| 2026-11-13 00:3x 폴더 이름 반영 | `c_qc` | EM 보고 |
| 2026-11-13 (메인 세션 기록 시각) 사용자 결정 | C6–C9(보정 보류, 봉인 판정 생략, 전체 실행, 4TU 진단 csv만) | 위 분석 |
| 2026-11-13 00:41 v1.1 작성 | C6–C9 반영 | 위와 같음 |
| 2026-11-13 (메인 세션 기록 시각) python-expert 구현 보고 | pytest 1123, G0, 합성 플래그 보고, 4TU 진단 | 구현 산출물 |
| 2026-11-13 01:08 v1.2 작성 | 구현 결과 반영(결정 불변): $C_1>C_0$ 한계, 띠 이동 정의, 순잡음 시드, 이름, G0, 4TU 결과 | `m210c_flag_report.json`, `m210_c_4tu_diag.md`, `m210c_4tu_line_diag.csv`(영점 규칙별 $\sigma_s$ 확인) |
| 2026-11-13 01:11 v1.2 정정 | §1-4 현장 9 해석을 "두 사건 띠 때문"에서 "원인 미확인"으로(현장 9 안 두 사건 0.39·단일 띠 0.44 ns), §12 후보 3 수정 | python-expert 계산, `m210c_4tu_line_diag.csv` 재확인 |

## 12. 가정·한계

- §1-1의 약 40 %·약 100 %·2.5 배율은 2D 합성(Ricker 500 MHz, 오프셋 0.10 m, εr 9, $h$ 0.05–0.15 m)에서 나왔다. 실제 InfraRadar 4TU 안테나의 직접 결합 크기·오프셋은 모른다(Tx–Rx 간격 미확인, [`spec_depth_v1.md`](spec_depth_v1.md) §14).
- P3 플래그는 깊이를 바꾸지 않는다. 4TU에서 공기층 변화로 생기는 깊이 오차(현장 11에서 대략 수 cm, 해석)는 그대로 남는다.
- P4 보류로 4TU 메타 εr 의심을 자동 표시하지 못한다(지금과 같음).
- P5의 0.15는 가정 그대로다.
- 합성 보고(§4)는 판정이 아니다. 4TU 진단 csv는 이미 본 자료의 탐색이다.
- (v1.2) 다음 항목 후보(기록만, 결정 아님):
  1. 지표 추종을 실제로 가리는 P3 판정 기준(§1-3 한계). 지금의 $C_1>C_0$는 정의상 거의 늘 참이다.
  2. 띠 이동 정의의 통일(상호상관 대 포락선 피크)과 참고 배율의 재정의. 지금의 `pipe_time_std_ref_ns`는 해석 불가다.
  3. 현장 9의 높은 $\sigma_s$(단일 띠·두 사건 모두 약 0.4 ns) 원인 확인. 두 사건 측선의 띠 창 정의는 현장 9 밖에서 차이가 작아(0.16 대 0.12 ns) 우선순위가 낮다.
  4. 커밋된 M2-9 평가 csv 4개가 HEAD 재실행과 다른 원인 확인(`MKL_CBWR` 고정 전 동률 추정).
