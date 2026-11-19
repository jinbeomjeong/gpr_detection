# M3-3 R3 `n_zero_changed` 보고 정의 점검 (2026-11-20)

- 작성: gpr-processing-expert. M3-3 마감 전 3건 중 ①이다. **명세·판정 전용**이고 코드(`src/`·`scripts/`·`analysis/detection/`)는 고치지 않았다. 구현은 §5 명세로 python-expert에 넘긴다.
- 대상: R3 injected 결과 `runs/eval/m33/r3/injected/`(C1 코드 커밋 `f4ad45e`, `m33_r3_band.py` blob `b499a456fe423b7d8a72fa516a0f2123ae3bd7f4`). P4 판정은 이미 공개됐다([`m33_p4p5_judgment.md`](../experiments/m33_p4p5_judgment.md) §3-3·§9-1). 그래서 이 결과를 열었다.
- 판정 영향: **없다.** `zero_changed`·`period_changed`는 실현 행에만 쓰이고 `realization_counts`(summary의 세 키)만 읽는다(§1-3). P4 판정값과 J1 판정값은 이 보고 정의와 무관하다.
- **사용자 결정 2026-11-20**: §7 질문 1은 갈래 **B**(봉인 러너·`src/` 불변, 갈래 A는 구현하지 않음), 질문 2는 대칭 키(`n_zero_changed`·`n_period_changed`가 새 정의, 이전 정의는 `n_zero_changed_exact`·`n_period_changed_exact`), 질문 3은 $\tau_z$ = 0.1 P·$\tau_P$ = 5 %(엄격 `>`, 규칙 이름·사건 수 변화는 보고만), 질문 4는 P4·P5 판정서 §10 공개(cv-detection-expert)·verif 감사 생략. 구현: [`m33_change_defs.py`](../../analysis/detection/m33_change_defs.py), [`m33_r3_change_report.py`](../../analysis/detection/m33_r3_change_report.py), 기준 해시 목록 [`m33_r3_result_hashes.csv`](../../analysis/detection/m33_r3_result_hashes.csv)(`r3/plan`·`r3/injected` 187개), 시험 `tests/test_m33_change_defs.py`. 비교는 구현에서 곱 형태(영점 $|Delta k_z| > 0.1P$, 주기 $|Delta P| > 0.05P$)로 썼다 — 비 형태와는 경계 1 ulp 안에서만 다르다(P4·P5 판정서 §10-2).
- 재집계 스크립트(읽기 전용, 이 문서와 함께 둠): [`m33_nzero_recount.py`](../../analysis/processing/m33_nzero_recount.py), [`m33_nzero_summary.py`](../../analysis/processing/m33_nzero_summary.py). 출력 `analysis/processing/m33_nzero/`(실현별·측선별 csv, `summary.json`).

## 0. 결론

| 항목 | 판정 | 근거(§) |
|---|---|---|
| 원인 | **비교 정의의 산물(보고 결함)이다.** 영점 재측정은 분수 샘플이고 비교는 `!=`다. 띠가 DC 제거·필터·Hilbert 포락선을 거쳐 영점 구간에 새는 1차 섭동(띠 진폭에 선형)까지 "바뀜"으로 셌다 | 1 |
| 같은 경로 영 시험 | 161측선 모두 $\Delta k_z$ = 0, $\Delta P$ = 0(정확히). 비교 경로 자체는 치우치지 않았다 | 2-1 |
| 재현 | 재집계가 저장 플래그를 2,159/2,159(영점), 2,159/2,159(주기) 그대로 재현했다(`n_zero_changed` 2,159, `n_period_changed` 1,245) | 2-1 |
| $|\Delta k_z|$ 크기 | 중앙 7.2e-5 샘플(7.0e-6 ns), 99 % 0.041 샘플(0.0041 ns), 최대 0.33 샘플 = **0.032 ns = 0.012 P**. 1 샘플을 넘은 실현 0, 0.05 P를 넘은 실현 0 | 2-1 |
| 영점 사건 종류 | 라벨·실패 변화 0. 규칙 이름 변화 2(`air_single_event` → `air_first_event`, 같은 첫 띠), 초기 사건 수 변화 12(모두 4TU 공기결합, 띠가 영점 탐색 창 안의 뒤 사건을 바꿈) | 2-1 |
| 제안 정의 | 라벨·실패 변화 또는 $|\Delta k_z| > 0.1P$ → **`n_zero_changed` = 0/2,159** | 3-2 |
| 주기와의 일관성 | 일관되지 않다. 주기는 rfft bin으로 양자화돼 완전 일치 비교가 "bin 한 칸(0.26–2.1 %)"부터 센다. 영점은 비트 단위다. 둘 다 "측정 분해능·불확도를 넘는 변화"로 맞추면 주기는 > 5 %(`PERIOD_CHECK_TOLERANCE`) → **140/2,159** | 3-3 |
| 주기의 실제 변화 | 영점과 달리 주기는 크게 바뀐 실현이 있다: > 10 % 68, 최대 50 %(IDS `LID10062` SNR 18). 몇 측선에 몰린다. ② 재현율 검토에 넘길 사실이다 | 2-2 |
| 판정 영향 | 없다(두 플래그는 `realization_counts`만 읽음). P4·J1 판정값 불변 | 1-3, 6-4 |
| 재실행 | 권고 갈래 B: 봉인 러너를 고치지 않는 CPU 보고 도구로 `runs/eval/m33/r3_change_report/`에 다시 낸다(약 10분, `--workers 4`). 대안 A는 러너 수정 + GPU 재추론(약 36분, C1 봉인 새 판) | 5, 6 |

## 1. 원인

### 1-1. 코드가 하는 일

`m33_r3_band.zero_changed`(b499a456, 1093–1097행)는 다음과 같다.

```python
filtered = inject.linear_steps(injected.data, injected)
measured = find_time_zero(injected, filtered).k_zero
return measured != context.zero.k_zero
```

- `context.zero`는 `prepare_line_context`가 배경 측선에 `find_time_zero(background, prepare_fitting_branch(background).filtered)`로 잰 영점이다.
- `k_zero`는 **분수 샘플**이다. 평균 트레이스 포락선(`pick_mode="envelope"`)의 `find_peaks` 정수 피크를 `parabolic_peak`로 세분한 값이다(`timezero._early_peaks`). 4TU는 `air_single_event`(단일 띠 포락선 피크), IDS GZ는 지면결합 규칙이다.
- 비교는 부동소수 완전 일치(`!=`)다. 허용 오차가 없다.
- 주기 `period_changed`(1206행)도 `image.period_samples != context.period_samples`로 완전 일치 비교다. 다만 $P$는 `measure_period_samples`가 0 채움 rfft의 **이산 bin**에서 고른 값($P = n_\text{fft}/j$)이라 값이 양자화돼 있다(§3-3).

### 1-2. 영점이 왜 "조금" 움직이는가

띠는 구역 (b)에 따라 영점 + 2 P 아래에 놓이고 영점 탐색 창(`[0, t_max + 2h_max/c + 0.5/f_c]`)과 직접 겹치지 않는다. 그래도 영점 사슬에 측선·트레이스 전체에 작용하는 단계가 있다.

| 단계 | 띠가 영점 구간에 새는 경로 |
|---|---|
| (1) DC 제거 | 트레이스 평균이 띠 진폭만큼 바뀌어 트레이스 전체에 상수 차가 생긴다 |
| (2) dewow·(3) 밴드패스 | `sosfiltfilt`(IIR, 양방향)의 임펄스 응답 꼬리 |
| 포락선 | `hilbert`는 FFT 기반 전역 연산이라 $1/t$ 꼴 꼬리가 영점 구간에 닿는다 |

이 누설은 띠 진폭에 **선형**이다. 재집계에서 같은 위치의 SNR 0 → 6 → 12 → 18 dB(진폭 ×2씩)마다 $\Delta k$가 거의 정확히 두 배가 되고, 극성 ±1에서 부호만 바뀐다(예: 4TU `02.5/Path1` g00, $\Delta k$ = 6.07e-5 → 1.21e-4 → 2.41e-4 → 4.82e-4 샘플, 극성 −1은 같은 크기에 부호 반대). 새 사건이 생기거나 규칙이 바뀐 것이 아니라 1차 섭동이다.

### 1-3. 판정 경로와의 관계

- `zero_changed`·`period_changed`는 `evaluate_band_line`의 `realization_rows`에만 들어간다. 그 행을 읽는 곳은 `inference_ranges`(`infer_i0`·`infer_i1`만)와 `realization_counts`(세 summary 키)뿐이다(`grep -n "zero_changed\|period_changed" analysis/detection/m33_r3_band.py`: 1093·1199·1206·1207·1363·1367·1368행).
- PP의 영점 $y_z$는 배경 측선 값이다([RUN] [`m33_runner_spec.md`](../experiments/m33_runner_spec.md) §3-1 4). 주입 자료로 다시 잰 영점은 어떤 판정 계산에도 들어가지 않는다.

### 1-4. 원인 판정

| 가설 | 판정 | 근거 |
|---|---|---|
| H1 비교 경로가 배경과 다름(dtype·필터 경로 차) | **기각** | 배경을 `float32`로 바꿔 `zero_changed`와 같은 경로(`inject.linear_steps` → `find_time_zero`)로 재면 161측선 모두 `context.zero.k_zero`와 정확히 같다. `runner.load_line(...).zero`와도 같다. 주기도 `make_detection_image(bg32)`가 `context.period_samples`와 정확히 같다 |
| H2 띠가 영점 사건을 실제로 옮기거나 바꿈 | **기각** | 라벨·실패 변화 0, $|\Delta k_z|$ 최대 0.012 P(0.032 ns). 영점 표준 불확도 $\sigma_z$ 0.15–0.25 ns의 약 1/5–1/8 |
| H3 분수 샘플 완전 일치 비교가 1차 섭동까지 셈 | **채택** | $|\Delta k_z| > 0$이 2,159/2,159. 크기는 띠 진폭에 선형(SNR 6 dB마다 2배, 극성에 따라 부호 반대, §1-2). 1e-6 P를 넘는 것도 1,392개뿐이고 나머지는 그보다 작다 |

## 2. 재집계 결과 (|Δ영점|, |Δ주기| 분포)

출처: `analysis/processing/m33_nzero/summary.json`(재현 명령 §8). 161측선(4TU 105, IDS GZ 56), 실현 2,159개(4TU 1,303, IDS 856, 이 가운데 `main_400` 808, `report_other` 48).

### 2-1. 영점

| 분위수 | $|\Delta k_z|$ (샘플) | $|\Delta t_z|$ (ns) | $|\Delta k_z|/P$ |
|---|---|---|---|
| 0 % | 5.4e-11 | 5.2e-12 | 2.1e-12 |
| 50 % | 7.2e-5 | 7.0e-6 | 2.9e-6 |
| 90 % | 3.0e-3 | 2.9e-4 | 1.1e-4 |
| 99 % | 4.1e-2 | 4.1e-3 | 1.5e-3 |
| 100 % | 0.328 | 0.0321 | 0.0118 |

| 허용 오차 | 넘는 실현 수 (/2,159) |
|---|---|
| > 0 (이전 정의) | 2,159 |
| > 1e-6 P | 1,392 |
| > 1e-3 P | 39 |
| > 0.01 P | 2 |
| > 0.05 P / 0.1 P / 0.25 P / 0.5 P | 0 / 0 / 0 / 0 |
| > 0.01 ns / 0.05 ns / 0.1 ns / 0.2 ns / 0.25 ns | 4 / 0 / 0 / 0 / 0 |
| > 1 샘플 | 0 |

- 데이터셋별 최대 $|\Delta k_z|/P$: 4TU 0.0118, IDS 0.0022(`main_400` 0.0022, `report_other` 2.6e-6). SNR별 최대: 0 dB 0.0015, 6 dB 0.0029, 12 dB 0.0059, 18 dB 0.0118 — SNR 6 dB마다 거의 정확히 2배로, 선형 누설과 맞는다.
- 사건 종류: 라벨 변화 0, `zero_failed` 0. 규칙 이름 변화 2개는 모두 4TU `09.7/Path6` g00 SNR 18 dB(극성 ±1)이고 `air_single_event` → `air_first_event`다. 영점 사건은 그대로이고($\Delta k_z$ = ∓0.076 샘플 = 0.003 P), 새로 생긴 사건은 144–145 샘플로 띠 꼭짓점(영점 + 2.82 P ≈ 139.8 샘플) 바로 뒤다. 띠 자신이 영점 탐색 창에 들어온 것이다.
- 초기 사건 수 변화 12개(위 2개 포함)는 모두 4TU 공기결합 측선이다(배경 규칙 `air_first_event` 10, `air_single_event` 2). 그중 영점 이동 최대는 0.0118 P(`09.5/Path6`, 사건 48.0·101.0·138.3 → 47.7·116.7)다. 탐색 창 끝은 $t_\max + 2h_\max/c + 0.5/f_c$이고 $t_\max$는 앞 1/4 구간의 포락선 최대다. 뒤 사건이 앞 1/4 안에서 가장 크면 창이 영점 + 2 P(구역 (b))를 넘어 띠까지 닿는다고 본다(**추정**, 창 끝 값은 기록하지 않았다). 영점에는 영향이 없으므로 보고 전용으로 둔다.

### 2-2. 주기

| 분위수 | $|\Delta P|$ (샘플) | $|\Delta P|/P$ |
|---|---|---|
| 50 % | 0.123 | 0.58 % |
| 90 % | 0.985 | 3.7 % |
| 99 % | 5.17 | 21.1 % |
| 100 % | 18.5 | 50.0 % |

| 허용 오차 | 넘는 실현 수 (/2,159) | 4TU (/1,303) | IDS `main_400` (/808) | IDS `report_other` (/48) |
|---|---|---|---|---|
| > 0 (이전 정의, bin ≥ 1칸) | 1,245 | 807 | 413 | 25 |
| > 1 % | 624 | 374 | 238 | 12 |
| > 2 % | 357 | 217 | 138 | 2 |
| > 5 % (제안) | 140 | 93 | 47 | 0 |
| > 10 % | 68 | 53 | 15 | 0 |

- bin 한 칸의 상대 크기 $P/n_\text{fft}$($n_\text{fft} = 2^{\lceil \log_2 8n_s \rceil}$): 4TU 중앙 0.63 %(0.30–0.88 %), IDS 중앙 0.80 %(0.26–2.1 %). 0이 아닌 $|\Delta P|/P$의 최솟값 0.26 %가 bin 한 칸과 맞는다.
- SNR별 > 5 %: 0 dB 13, 6 dB 20, 12 dB 33, 18 dB 74.
- 큰 변화는 몇 측선에 몰린다(> 5 % 실현 수: 4TU `011.16/Profile_6454` 8, `09.5/Path2` 7, IDS `LID10074`(1030_190118AA) 6, `LID10001`(1030_190118AC) 6 등). `LID10074`는 SNR 0 dB 극성 −1에서도 $P$ 46.5 → 65.0 샘플(+40 %)로 뛴다. 평균 스펙트럼에 크기가 비슷한 봉우리가 둘 있어 작은 띠가 최대 bin을 옮기는 측선으로 본다(**추정**, 스펙트럼은 그리지 않았다). 이런 실현은 배경 측선과 30–50 % 다른 L1 세로 축척으로 추론됐다.

## 3. 제안 정의

### 3-1. 기호

| 기호 | 뜻 | 단위 |
|---|---|---|
| $k_z^{bg}$ | 배경 측선 영점(`context.zero.k_zero`) | 샘플(분수) |
| $k_z^{inj}$ | 주입 실현 자료로 같은 경로로 다시 잰 영점 | 샘플(분수) |
| $\Delta k_z$ | $k_z^{inj}-k_z^{bg}$ | 샘플 |
| $\Delta t_z$ | $\Delta k_z \cdot \Delta t$ | ns |
| $\Delta t$ | 측선 샘플 간격(`dt_ns`, 측선 json/목록 값) | ns |
| $P$ | 배경 측선 검출 영상 주기 `P_line`(`context.period_samples`) | 샘플 |
| $P^{inj}$ | 주입 실현 r0 영상의 `image.period_samples` | 샘플 |
| $\tau_z$ | 영점 허용 오차 | P(무차원) |
| $\tau_P$ | 주기 허용 오차 | 무차원(상대) |
| $\sigma_z$ | 규칙별 영점 표준 불확도([`spec_depth_v1.md`](spec_depth_v1.md) §9, `SIGMA_ZERO_NS`) | ns |

### 3-2. 영점 바뀜

실현이 다음 가운데 하나라도 만족하면 "영점 바뀜"(`zero_changed` true)이다.

1. **사건 종류 변화**: 영점 사건 라벨(`zero_event_label`: `merged`·`surface_reflection`·`direct_wave`·`meta`)이 배경과 다르다. 또는 한쪽만 `zero_failed`다.
2. **크기 변화**: $|\Delta k_z| > \tau_z \cdot P$, $\tau_z = 0.1$.

규칙 이름(`zero_rule`)과 초기 사건 수가 바뀐 것은 **영점 바뀜에 넣지 않고 따로 보고한다.** 재집계에서 그런 실현은 모두 영점 사건이 그대로였다(§2-1). `air_single_event` → `air_first_event`는 4TU에서 둘 다 첫(합쳐진) 띠를 영점으로 쓰므로([`spec_depth_v1.md`](spec_depth_v1.md) v2.1 §4-2-1, 높이 메타 없음) 영점의 정의가 같고, 늘어난 사건은 띠 자신이 영점 탐색 창 안에 들어온 뒤 사건이다. 다른 사건으로 영점이 옮겨 가면 사건 간격(분리 하한 0.75/$f_c$ ≈ 0.75 P)이 0.1 P보다 훨씬 크므로 2로 잡힌다.

$\tau_z = 0.1\,P$의 근거:

- 영점을 쓰는 하류 규칙이 모두 P 단위다. 구역 (b)는 꼭짓점 ≥ 영점 + 2 P, PP-d는 $y_m < y_z + 0.5P_\ell$다. 0.1 P는 가장 좁은 여유(PP-d 0.5 P)의 1/5이다. 이보다 작은 이동은 경계에서 0.1 P 안에 있는 박스만 뒤집을 수 있다.
- R3 측선에서 0.1 P는 4TU 0.13–0.36 ns(중앙 0.26), IDS 0.14–0.50 ns(중앙 0.20)다. 영점 자체의 표준 불확도 $\sigma_z$(공기결합 0.25 ns, 지면결합 0.15 ns, [`spec_depth_v1.md`](spec_depth_v1.md) §9)와 같은 크기이고, 영점 창 이동 QC 문턱 `window_drift_ns` 0.2 ns(`ZeroConfig`)와도 같은 크기다. 영점 불확도보다 작은 변화는 "바뀌었다"고 말할 근거가 없다.
- 깊이로는 $\Delta d = v\,\Delta t_z/2$다. $\varepsilon_r = 9$(**가정**, 4TU 헤더 기본값과 같은 값이라 실측 아님)에서 $v$ = 0.1 m/ns이면 0.1 P ≈ 1.3 cm(4TU 중앙 0.26 ns 기준, 범위 0.7–1.8 cm)다. 4TU 제186조 ±30 cm보다 두 자릿수 작다.
- 단위를 P로 둔 것은 주기 정의(§3-3)와 같은 무차원 척도로 맞추기 위해서다.

summary에 함께 낼 보고 키(정의 투명성):

| 키 | 뜻 |
|---|---|
| `n_zero_changed` | 위 정의(라벨·실패 변화 또는 $|\Delta k_z| > 0.1P$) |
| `n_zero_changed_exact` | 이전 정의(`!=`). 2026-11-16 공개값 2,159와 같은 정의 |
| `n_zero_label_changed` | 1만(라벨·실패 변화) |
| `n_zero_rule_changed` | 규칙 이름이 바뀐 실현(보고 전용) |
| `n_zero_events_changed` | 초기 사건 수가 바뀐 실현(보고 전용, 띠가 영점 탐색 창에 들어온 표지) |
| `zero_change_tolerance_p` | 0.1 |
| `zero_abs_dk_over_p_quantiles` | $|\Delta k_z|/P$의 0·50·90·99·100 % 분위수 |
| `zero_abs_dt_ns_max` | $|\Delta t_z|$ 최댓값(ns) |

### 3-3. 주기 바뀜과의 일관성

- 현재 `period_changed`도 완전 일치 비교다. 그러나 $P$는 rfft bin에서 고른 이산 값이라, 완전 일치 비교는 "bin이 하나 이상 바뀜"을 센다. bin 간격은 $\delta P \approx P^2/n_\text{fft}$(상대값 4TU 중앙 0.63 %, IDS 중앙 0.80 %, 전체 0.26–2.1 %)다. 부동소수 잡음이 아니라 측정 분해능 한 칸이다.
- 그래서 두 정의는 **일관되지 않다.** 영점은 "비트 하나"까지, 주기는 "분해능 한 칸"부터 센다. 공통 원칙을 "측정 자체의 분해능·불확도를 넘는 변화"로 두면 다음이 된다.
  - 영점: §3-2(0.1 P ≈ $\sigma_z$).
  - 주기: $|P^{inj}/P - 1| > \tau_P$, $\tau_P$ = 0.05. 근거는 프로젝트가 이미 쓰는 "같은 주기" 기준 `inject.PERIOD_CHECK_TOLERANCE` = 0.05(응답 주기가 $P_\text{line}$의 ±5 % 안이면 같은 주기로 받아들임)다.
- 주기 변화는 영점과 달리 **실제로 큰 경우가 있다**(§2-2: > 5 % 140개, > 10 % 68개, 최대 50 %). L1 세로 축척이 $P$에 반비례하므로 이 실현들은 배경 측선과 다른 축척의 영상으로 추론됐다. 판정에는 영향이 없지만(박스와 정답이 같은 영상 좌표에서 비교됨, [RUN] §3-1 4 PP 내용 범위 문단) ② R3 의사 띠 재현율 검토에서 볼 변수다.
- 제안 보고 키: `n_period_changed`(> 5 %), `n_period_changed_exact`(이전 정의, 공개값 1,245), `period_change_tolerance_rel` 0.05, `period_abs_dp_rel_quantiles`. 이름을 대칭으로 둘지는 §7 질문 2다.

## 4. 경계 조건

| 경우 | 처리 |
|---|---|
| 한쪽 `k_zero`가 `None`(`zero_failed`) | 바뀜(true). 둘 다 `None`이면 false |
| `amplitude_processed` 측선(`meta_sample0`, $k_z$ = 0) | 규칙이 같고 $\Delta k_z$ = 0이므로 false. R3 측선에는 없다 |
| $P$ 또는 $P^{inj}$가 `None` | 주기 바뀜 true(측정 실패도 변화). R3 주입 실현에서는 0건이어야 하고, 생기면 건수를 보고한다 |
| 허용 오차 경계 | 엄격 부등호(`>`). 0.1 P와 정확히 같으면 바뀜이 아니다 |
| 단위 | $\Delta k_z$·$P$ 모두 같은 측선의 L0 샘플이다. px로 바꾸지 않는다 |

## 5. python-expert 위임 명세

두 갈래 가운데 무엇을 구현할지는 §7 질문 1(재실행 방식)이 정한다. 5-1은 두 갈래 공통이다.

### 5-1. 공통: 변화 정의 함수 (새 모듈)

- 새 파일 `analysis/detection/m33_change_defs.py`(봉인 목록 밖 새 파일). 의존: `numpy`, `gpr_detection.preprocess.timezero.ZeroResult`.
- 상수
  - `ZERO_CHANGE_TOLERANCE_P = 0.1` — §3-2 근거.
  - `PERIOD_CHANGE_TOLERANCE_REL = 0.05` — §3-3 근거. 시험에서 `inject.PERIOD_CHECK_TOLERANCE`와 같은 값임을 단언한다(근거의 연결을 깨지 않게).
- `zero_change(bg: ZeroResult, inj: ZeroResult, period_samples: float) -> ZeroChange`
  - `ZeroChange`(frozen dataclass): `dk_samples: float | None`, `dk_over_p: float | None`, `label_changed: bool`(`zero_event_label` 다름), `failed_changed: bool`(한쪽만 `failed`), `rule_changed: bool`(보고 전용), `events_changed: bool`(`len(events_ns)` 다름, 보고 전용), `exceeds_tolerance: bool`(`abs(dk_over_p) > ZERO_CHANGE_TOLERANCE_P`), `changed: bool`(= label ∨ failed ∨ exceeds_tolerance), `exact: bool`(이전 정의 `inj.k_zero != bg.k_zero`).
  - **규칙 이름·사건 수 변화는 `changed`에 넣지 않는다.** 재집계에서 그런 실현은 모두 영점 사건이 그대로였다(§2-1, §3-2). `air_single_event` ↔ `air_first_event`는 라벨이 둘 다 `merged`이고 영점이 같은 첫 띠다.
  - 경계: §4 표. `period_samples`가 0 이하이거나 `None`이면 `ValueError`.
- `period_change(p_bg: float | None, p_inj: float | None) -> PeriodChange`
  - `PeriodChange`: `dp_samples`, `dp_rel`(`p_inj / p_bg - 1`), `exceeds_tolerance`(`abs(dp_rel) > 0.05`), `failed`(`p_inj is None`), `changed`(= exceeds ∨ failed), `exact`(`p_inj != p_bg`).
- 모든 실수는 출력 시 `repr(float)`([RUN] §1 전체 정밀도).

### 5-2. 갈래 B(권고): 별도 보고 도구, 봉인 러너 불변

- 새 파일 `analysis/detection/m33_r3_change_report.py`. 봉인 러너(`m32_recipe_eval.py`·`m33_r3_band.py`)와 `src/`는 **고치지 않는다.** 러너 함수는 import만 한다.
- 입력과 단언(어기면 종료 2)
  - `runs/eval/m33/r3/plan/injection_table.csv`·`exclusions.csv`·`positions.csv`의 LF sha = `analysis/detection/m33_r3_c1b.json` 값(`inputs.checked_text_sha256`, 러너와 같은 함수).
  - `runs/eval/m33/r3/injected/summary.json`의 `run_identity.code_blobs`가 C1 봉인 기록 §1 값과 같고, 지금 작업 트리의 같은 파일 blob과 같다(같은 코드로 다시 만드는지 확인).
  - `inject.py` blob = `m33_inputs.C1_INJECT_BLOB`.
- 측선마다(러너 `evaluate_band_line`과 같은 호출 순서, 추론·PP·생존 비 없음)
  1. `runner.load_line(ref)` → `runner.line_exclusions(...)` → `inject.prepare_line_context(radargram, exclusions)` → `library.for_line(dataset, runner.line_time_window_ns(line))`.
  2. 영 시험: 배경을 `float32`로 바꿔 `find_time_zero(bg32, inject.linear_steps(bg32.data, bg32))`와 `make_detection_image(bg32, recipe_config(R0)).period_samples`를 잰다. `context.zero.k_zero`·`context.period_samples`와 **완전 일치**해야 한다(어기면 종료 1, "비교 경로 자체가 치우침").
  3. 실현마다 `inject_band_realization` → 정답 없으면 건너뜀 → `replace(radargram, data=result.data.astype(np.float32))` → 영점 재측정(러너 `zero_changed`와 같은 경로) → `make_detection_image(..., recipe_config(R0))` → `zero_change`·`period_change`·`tile_trace_range(image, inference_tiles(image))`.
- 재현 단언(어기면 종료 1): 실현 id 집합이 `injected/parts/*.json`의 `realizations` 집합과 같다(2,159). 실현마다 `ZeroChange.exact`·`PeriodChange.exact`·`(infer_i0, infer_i1)`가 저장값과 같다. 이로써 같은 주입 자료를 다시 만들었음을 보인다.
- 원본 불변: 시작과 끝에 `runs/eval/m33/r3/` 아래 모든 파일의 sha256 목록을 재고 같아야 한다(종료 1). 목록은 출력 폴더에 `r3_tree_sha256_start.csv`·`_end.csv`로 남긴다.
- 출력 `runs/eval/m33/r3_change_report/`(있으면 종료 2, 덮어쓰기 인자 없음)
  - `realizations.csv`: 측선, 데이터셋, `ids_freq_stratum`, 실현 id, SNR, 극성, `dt_ns`, $P$, $k_z^{bg}$·$k_z^{inj}$, 규칙 둘, 사건 수 둘, `ZeroChange`·`PeriodChange` 전 필드, 저장값 둘.
  - `lines.csv`: 측선별 영 시험 값(모두 0이어야 함), 초.
  - `summary.json`: §3-2·§3-3 키(전체, `dataset`별, `ids_freq_stratum`별, `main_400`의 P4 범위(`gz_ids`·`main_400`) 따로), 재현 단언 결과, 정체성(HEAD·dirty·도구 blob·러너 blob·입력 LF sha·`r3/injected/summary.json` sha·패키지 버전·argv·시작·끝 시각).
  - stdout 마지막 줄 `{"status","out_dir","n_lines","n_realizations","elapsed_s"}`. 판정값이 아니므로 블라인드는 필요 없지만 형식은 러너와 맞춘다.
- `--workers N`(측선 단위 `ProcessPoolExecutor`, 기본 1). 출력은 측선·실현 id 순으로 정렬해 쓰므로 N과 무관하게 바이트가 같아야 한다(시험).

### 5-3. 갈래 A(대안): 러너 수정 + injected 재실행

- `m33_r3_band.zero_changed`를 `zero_change(...)` 호출로 바꾸고, 실현 행에 `dk_zero_samples`·`dk_zero_over_p`·`zero_label_changed`·`zero_rule_changed`·`zero_events_changed`·`zero_changed_exact`·`dp_rel`·`period_changed_exact`를 더한다. `zero_changed`·`period_changed` 열은 새 정의(§3)다.
- `realization_counts`에 §3-2·§3-3 키를 더하고 `SCHEMA`를 `m33_r3_band/2`로 올린다.
- 시험 `tests/test_m33_r3_band.py` 399–409행(`realization_counts` 기대값)과 544행을 새 키로 고친다.
- 이 갈래는 `m33_r3_band.py` blob을 바꾸므로 §6-3 봉인 영향이 생긴다.

### 5-4. 시험 (두 갈래 공통, `tests/test_m33_change_defs.py`)

| # | 입력 | 기대 |
|---|---|---|
| 1 | 같은 `ZeroResult` 둘 | `changed` false, `exact` false, `dk` 0 |
| 2 | `k_zero` 차 1e-6 샘플, $P$ = 25 | `exact` true, `changed` false |
| 3 | 차 = 0.1·P 정확히 | `changed` false(엄격 부등호) |
| 4 | 차 = 0.1·P + 1e-9 | `changed` true |
| 5 | 규칙 `air_single_event` → `air_first_event`(라벨 둘 다 `merged`), 차 0.003 P | `rule_changed` true, `changed` false |
| 5b | 라벨 `merged` → `surface_reflection`, 차 0 | `label_changed` true, `changed` true |
| 6 | 사건 수 3 → 2, 규칙 같음, 차 0.01 P | `events_changed` true, `changed` false |
| 7 | 한쪽 `zero_failed`(`k_zero` None) | `failed_changed` true, `changed` true, `dk` None |
| 8 | 둘 다 `zero_failed` | `changed` false |
| 9 | $P^{inj}/P$ = 1.05 정확히 / 1.0500001 | false / true |
| 10 | $P^{inj}$ None | `failed` true, `changed` true |
| 11 | `PERIOD_CHANGE_TOLERANCE_REL == inject.PERIOD_CHECK_TOLERANCE` | 참 |
| 12 | (갈래 B) 합성 작은 측선: 띠 주입 → 보고 도구 1측선 실행 | 영 시험 완전 일치, `exact` 재현, 원본 sha 목록 시작 = 끝 |
| 13 | (갈래 B) `--workers 1`과 `2` 출력 | 바이트 같음 |
| 14 | (실데이터, `realdata` 표지) 4TU `02.5/Path1` g00 8실현 | `exact` 모두 true, `changed` 모두 false, $|\Delta k_z|$ ≤ 1e-3 샘플, SNR 6 dB마다 $|\Delta k_z|$ 비 2 ± 0.05 |

## 6. R3 보고 재실행 명세

### 6-1. 어느 단계를 다시 돌리는가

| 단계 | `zero_changed`·`period_changed`와의 관계 | 다시 돌림 |
|---|---|---|
| `uninjected`(IDS 60, r0 × 5 fold) | 계산하지 않음 | 아니오 |
| `plan`(배치·구역·주입 표) | 계산하지 않음(구역 (b)는 배경 영점을 씀) | 아니오 — C1b 봉인 표를 그대로 쓴다 |
| `injected` | `evaluate_band_line`에서 실현마다 계산 | 갈래 A만 예(추론 포함). 갈래 B는 추론 없이 보고 부분만 |

C1b 주입 표·배치·제외 구역은 바뀌지 않는다. 주입 자료는 결정적이다(시드 20261115, 잡음 없음, `inject_band_realization`은 배경 사본에 띠를 더할 뿐). 그래서 plan을 다시 돌릴 이유가 없다.

### 6-2. 갈래별 실행

**갈래 B(권고)** — 5-2 도구, CPU만.

```powershell
conda run -n gpr-dev python -I -B analysis/detection/m33_r3_change_report.py --workers 4
```

- 출력: `runs/eval/m33/r3_change_report/`(새 경로, 기존 `r3/`·C1b 폴더를 덮지 않음).
- 예상 시간: 이 점검의 재집계(같은 계산, 1 프로세스)가 161측선·2,159실현에 2,176 s(약 36분)였다. 측선 하나의 최장 시간은 92.6 s다. `--workers 4`이면 약 10–12분(합 ÷ 4 = 9분에 긴 측선 꼬리), 1이면 약 36분이다. GPU 없음, 메모리 프로세스당 0.5 GB 안팎(재집계 실측 작업 집합 455–470 MB).
- `--workers 4`면 30분을 넘지 않아 세션 간 예약·감시가 필요 없다. 1로 돌리면 약 36분이라 전역 감시 규칙(15분 간격 점검)을 따른다. 시작 전 `status.py`로 빈 코어를 보고 affinity·BelowNormal로 띄운다.

**갈래 A(대안)** — 5-3 러너 수정 뒤 injected 재실행, GPU.

- 출력 폴더: 새 out-root `runs/eval/m33/r3_report_v2/`. injected 단계는 `<out-root>/plan`과 `<out-root>/uninjected_ids`를 읽으므로 기존 `r3/plan`·`r3/uninjected_ids`를 **바이트 그대로 복사**하고 대조한 뒤 돌린다. `uninjected_ids`는 [`m33_r3_upstream_hashes.csv`](../experiments/m33_r3_upstream_hashes.csv)(68행)와, `plan`은 C1b 기록의 세 LF sha와 기존 injected `summary.json`의 `run_identity` 입력 sha(`plan_report`)와 대조한다. 4TU 미주입 박스는 기본 `--r2-root`(R2 결과) 그대로다.
- 명령: `conda run -n gpr-dev python -I -B analysis/detection/m32_recipe_eval.py --injectant band --stage injected --out-root runs/eval/m33/r3_report_v2`. `--out-root`는 [RUN] §1 금지 인자다(예외는 실패용 예비 폴더뿐). **보고 재실행용 예외를 사용자 결정으로 새로 두어야 한다**(§7 질문 1).
- 예상 시간: 2026-11-16 injected 실측 2,148 s(약 36분, VRAM 예약 1.9 GB) 그대로. 30분을 넘으므로 메인 세션 러너(`make_runner.py --gpu-log --gpu-counters`), 세션 간 예약(`--reserve-owner gpr_detection:m33_r3_report --reserve-ttl-min 90`), 15분 간격 감시가 필요하다.

### 6-3. 봉인 영향

| 대상 | 갈래 B | 갈래 A |
|---|---|---|
| C1 봉인 기록 [`m33_c1_seal.md`](../experiments/m33_c1_seal.md) §1 코드 blob 21행 | 안 바뀜(새 파일 2개는 봉인 목록 밖) | `m33_r3_band.py` 행(`b499a456…`)이 바뀜. 봉인 기록 머리말대로 "C1 코드 blob이 바뀌면 봉인이 깨지므로 새 판과 공개"가 필요하다. 새 판에는 새 blob, 바뀐 함수 두 개, "판정 경로 불변" 근거(§1-3)를 적는다 |
| 러너 단언 | 영향 없음 | 러너는 `m33_r3_band.py` blob을 상수로 단언하지 않는다(단언은 `inject.py` `C1_INJECT_BLOB`와 `pipeline.py` C0a 값뿐, `m33_inputs.py` 414–425행). 그래서 실행은 거부되지 않고 `run_identity.code_blobs`에 새 blob이 남는다 |
| C1b 기록 `analysis/detection/m33_r3_c1b.json` | 안 바뀜 | 안 바뀜(키: 주입 표·위치·제외 sha, 시드, `pp_set_id`, `pp_config_sha256`만) |
| C2·J1 봉인(`analysis/processing/m33_j1_seal.py` `SEALED_BLOBS`) | 안 바뀜 | 안 바뀜. 목록은 `scripts/eval_depth_4tu.py`·J1 도구 2개·csv/json 3개이고 `analysis/detection/` 파일이 없다. J1 입력은 R2 박스(`r2/fold0/j1_boxes_post.csv`)이고 src tree도 안 바뀐다(`src/` 수정 없음) |
| R2 결과의 `run_identity.code_blobs` | 해당 없음 | R2 기록에는 옛 `m33_r3_band.py` blob이 남는다(R2 코드 경로 목록에 들어 있음, `m32_recipe_eval.code_paths`). R2를 다시 돌리지 않으므로 기록만 공개한다 |
| 재감사 | 사용자 결정(2026-11-13)대로 판정 결론이 없는 보고 실행이라 verif 감사 대상이 아니다. integrity-checker만 거친다 | 같은 이유로 verif 실행 전 감사는 필수가 아니다. 다만 C1 봉인 새 판이 생기므로 integrity-checker로 봉인 기록 ↔ 코드 blob 정합을 본다. verif를 거칠지는 §7 질문 4 |

### 6-4. P4·J1 판정값이 바뀌지 않음을 보이는 방법

**구조 근거(두 갈래 공통).** §1-3의 grep 결과대로 두 플래그는 `realization_counts`만 읽는다. P4 입력(`p4_inputs`·`match_units`·`evaluate_p4_value`)과 J1 입력(R2 박스)은 이 값을 읽지 않는다.

**갈래 B 대조 기준**

1. 원본 불변: `runs/eval/m33/r3/` 아래 모든 파일 sha256이 도구 시작·끝에서 같고, `uninjected_ids`는 [`m33_r3_upstream_hashes.csv`](../experiments/m33_r3_upstream_hashes.csv)(68행)와도 같다. **이 목록에는 `r3/plan`·`r3/injected`가 없다**(r2 798행 + `r3/uninjected_ids` 68행). 그래서 도구 실행 **전에** 메인 세션이 `r3/plan`·`r3/injected` 전 파일의 sha256 목록을 만들어 커밋해 두기를 권한다(예: `docs/experiments/m33_r3_result_hashes.csv` — 실제 위치는 `analysis/detection/m33_r3_result_hashes.csv`, 같은 열 `root,path,sha256,bytes,mtime_utc`). 그러면 도구의 시작 목록을 커밋된 기준과 대조할 수 있다(갈래 A도 이 목록이 비교 기준이 된다).
2. 재현: 실현 2,159개 전부에서 `exact` 플래그와 `(infer_i0, infer_i1)`가 저장값과 같다. `n_zero_changed_exact` = 2,159, `n_period_changed_exact` = 1,245.
3. P4 판정값은 다시 계산하지 않으므로 기존 `r3/injected/summary.json`의 `p4_adoption_input`이 그대로 판정값이다(1 덕분에 바이트 불변).

**갈래 A 대조 기준**(새 폴더 vs 기존 `r3/injected/`, `tools\pyinspect\output_compare.py`)

| 파일 | 기준 |
|---|---|
| `truths.csv`, `rejections.csv` | LF sha 같음 |
| `fold0`–`fold4`의 `inj_boxes.csv`·`inj_matches.csv`·`ghosts.csv` | LF sha 같음 |
| `p4_pp_acted.csv`, `ghost_pp_artifacts.csv` | LF sha 같음 |
| `summary.json` | `run_identity`·`timings`·`resources`·`schema`와 새 보고 키(§3)를 뺀 나머지를 정렬 JSON으로 비교해 같음. `p4_adoption_input`은 통째로 같음 |
| `parts/*.json` | `realizations[*]`의 새 열·`zero_changed`·`period_changed`, `timings`, `seconds`, `head`를 뺀 나머지가 같음 |

- 박스 점수는 `repr(float)` 전체 정밀도로 쓰이므로 GPU 추론이 비트 단위로 재현되지 않으면 박스 csv sha가 달라질 수 있다. 이 경우 판정값 키(`p4_status`, `recall_delta`, `pp_exposed_positions`, `pp_acted_boxes`, `n_r_boxes`, `n_r_lines`, `recall_pre`)의 완전 일치로 대신하고, 다른 칸 수를 공개한다. 앞선 재실행 사례(D0′, 183칸 차이 0, [`m33_d0r_record.md`](../experiments/m33_d0r_record.md))는 지표 수준 일치였고 박스 바이트 일치는 확인된 적이 없다.
- 이 위험이 갈래 B를 권하는 이유 가운데 하나다.

### 6-5. 공개 문구(판정서에 붙일 것)

P4·P5 판정서 [`m33_p4p5_judgment.md`](../experiments/m33_p4p5_judgment.md)는 판정 문서라 본문을 고치지 않고 뒤에 "§10 보고 정의 개정(2026-11-20)"을 덧붙이는 방식을 권한다(작성 주체는 판정서 작성자 cv-detection-expert, §7 질문 4). 넣을 문장:

- `n_zero_changed` 2,159는 분수 샘플 영점의 완전 일치 비교에서 나온 값이다. 같은 경로의 영 시험은 0이었고, 차는 띠 진폭에 선형인 필터·포락선 누설(최대 0.33 샘플 = 0.032 ns = 0.012 P)이었다.
- 새 정의(라벨·실패 변화 또는 $|\Delta k_z| > 0.1P$)로는 0/2,159다. 주기는 이전 정의 1,245(bin 한 칸 이상), 새 정의(> 5 %) 140/2,159다.
- 판정은 바뀌지 않는다. 근거는 §6-4.
- 이 개정은 결과를 본 뒤 정한 보고 정의 변경이다(공개 사항).

## 7. 확인 필요 질문(결정됨 2026-11-20, 머리말 6행)

1. **재실행 방식**: 사용자 결정(2026-11-20) "고친 코드로 R3 보고를 다시 실행"을 어느 갈래로 읽을까?
   - B(권고): 새 보고 도구(5-2)가 C1b 표로 주입 자료를 다시 만들어 영점·주기만 다시 잰다. 봉인 blob 불변, GPU 없음, `--workers 4`로 약 10–12분. P4는 다시 계산하지 않으므로 바이트 불변이 자명하다. E3 확장에서 러너를 다시 쓰면 그때 5-1 함수를 러너에 넣는다(E3 봉인에 포함).
   - A: 러너(`m33_r3_band.py`)를 고치고 injected를 새 폴더에서 다시 돌린다. C1 봉인 새 판, `--out-root` 예외 결정, GPU 약 36분, 박스 바이트 재현 위험(§6-4).
2. **키 이름**: 영점·주기 모두 주 키(`n_zero_changed`·`n_period_changed`)를 새 정의로 두고 이전 정의를 `_exact`로 옮길까(권고, 대칭)? 아니면 주기는 이미 이산값이라 `n_period_changed`를 그대로 두고 `n_period_changed_gt5pct`만 더할까?
3. **허용 오차 값**: 영점 $\tau_z$ = 0.1 P(≈ $\sigma_z$), 주기 $\tau_P$ = 5 %(`PERIOD_CHECK_TOLERANCE`)로 정해도 될까? 대안은 영점을 규칙별 $\sigma_z$(ns, 0.15/0.25)로 두는 것이다. R3 결과에서는 두 값 모두 같은 수를 준다(§2-1).
4. **공개 위치·작성자**: P4·P5 판정서 끝에 §10을 덧붙이는 방식(작성 cv-detection-expert, 이 문서 인용)으로 할까? 갈래 A면 C1 봉인 기록 새 판도 함께 필요하다. verif 감사는 사용자 결정(2026-11-13)대로 생략해도 되는지(갈래 A에서 봉인 새 판이 생기는 경우 포함) 확인한다.

## 8. 재현

- 작업 트리: HEAD `c1a2163591fa49dfa9995acc4c38121365acfafa`(깨끗, 새 파일은 이 문서와 `analysis/processing/m33_nzero*`뿐). `analysis/detection/m33_r3_band.py` blob `b499a456fe423b7d8a72fa516a0f2123ae3bd7f4`(C1 봉인값과 같음), `src/` 변경 없음.
- R3 injected 결과의 실행 정체성: `stage_settings.json` `head` `e5b108fbd308a13a169737003ba8454b3cc4226b`, `src_tree_id` `d0e54933f3a0c87cf2b8d43d255e97bd6bb97122`, 2026-11-16 10:37–11:13, `seconds_total` 2,148.4.
- 명령(gpr-dev, 1 프로세스, BelowNormal, BLAS 스레드 1, GPU 없음, 4구간으로 나눠 실행):

```powershell
conda run -n gpr-dev python -I -B analysis/processing/m33_nzero_recount.py --start 0 --stop 40 --out-dir analysis/processing/m33_nzero
#   ... --start 40 --stop 80 / 80 120 / 120 161
conda run -n gpr-dev python -I -B analysis/processing/m33_nzero_summary.py --in-dir analysis/processing/m33_nzero
```

- 구간별 시간(s): 0–40 453.7, 40–80 345.6, 80–120 638.6, 120–161 738.5(합 2,176.4). 난수 없음(주입은 C1b 표의 결정적 값).
- 입력: `runs/eval/m33/r3/plan/injection_table.csv`·`exclusions.csv`(읽기만), `runs/eval/m33/r3/injected/parts/*.json`(저장 플래그 대조용, 읽기만), 응답 라이브러리(`m32_pipe_responses.load_library()`), 측선 원본(`C:\workspace\GPR_Data`, 읽기만).
- 출력: `analysis/processing/m33_nzero/realizations_*.csv`(2,159행), `lines_*.csv`(161행), `summary.json`.
