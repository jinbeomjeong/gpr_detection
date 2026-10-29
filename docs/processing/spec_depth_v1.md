# 명세 v1 — 쌍곡선 피팅 기반 관로 깊이 추정 (전처리 · 영점 · 피팅 · 속도 규칙 · 회귀 시험)

- 작성: gpr-processing-expert, 2026-10-29 (spec 모드)
- 받는 쪽: python-expert (구현), cv-detection-expert (박스 → 시료 좌표 계약만)
- 1차 근거: `docs/research/depth_physics.md`(이하 [DP]), `docs/research/prior_art.md`(이하 [PA], 출처 번호는 그 문서의 [n]), `docs/datasets/readme_*.md`·`codebook_4tu.md`, `data/synthetic_gprmax/*.json`·`manifest.csv`
- 이 명세를 위한 탐색 실행: `analysis/processing/zero_reference_check.py` → `docs/processing/zero_reference_check.csv`, `zero_reference_curvature.csv`, `zero_reference_airgap_models.csv`, `zero_reference_check.png` (§12에 요약)

---

## 0. 범위와 원칙

1. 깊이는 **물리 스케일(dt_ns, dx_m)이 있는 입력**에서만 m 단위로 낸다. AIHub(71327, 폴더명 "261")와 모로코는 꼭짓점 위치를 픽셀로만 내고 `scale="none"`을 붙인다.
2. 깊이는 **관 상단** 깊이 $d_\text{top}$다. 반경 $R$을 알면 $d_\text{center}=d_\text{top}+R$도 함께 낸다. 정답 깊이의 기준점(상단/중심)은 원본에 없으므로 비교 단계에서 두 가정을 모두 쓴다([DP] §4, readme_4tu §5-3).
3. 피팅과 영점 검출은 **gain을 적용하지 않은** 시간축에서 한다. gain은 검출 입력 쪽 갈래에만 둔다.
4. 영점과 꼭짓점은 **같은 위상 기준**으로 잡는다. 기본값은 둘 다 Hilbert 포락선 피크다. first break 영점과 피크 꼭짓점을 섞으면 εr 9에서 +8 cm(지면결합)~+17 cm(공기결합)가 어긋난다(§12 표 1).
5. 반경 $R$은 피팅 미지수로 넣지 않는다. 0 또는 알려진 관경/2로 고정한다(Giannakis 2022 [PA 10]).
6. 공기결합 자료(4TU)에서는 직선 광선 쌍곡선의 곡률로 v를 구하지 않는다([DP] §0-5, §12 표 3).

### 기호

| 기호 | 뜻 | 단위 |
|---|---|---|
| $t$ | 기록 시각(샘플 $k$의 시각 $k\,\Delta t$) | ns |
| $\Delta t$, $\Delta x$ | 샘플 간격, 트레이스 간격 | ns, m |
| $t_z$ | 깊이 영점(선택된 time-zero 사건의 포락선 피크 시각) | ns |
| $t_\text{dir}$, $t_\text{surf}$ | 직접파 피크, 지표 반사 피크 | ns |
| $t_a$ | 꼭짓점 반사 피크 시각 | ns |
| $t_0$ | 영점 기준 꼭짓점 왕복시간 $t_a-t_z$ | ns |
| $x$, $x_0$ | 트레이스 중점 위치, 꼭짓점 위치 | m |
| $v$ | 토양 전파속도 | m/ns |
| $\varepsilon_r$ | 비유전율, $v=c/\sqrt{\varepsilon_r}$ | – |
| $c$ | 진공 광속 0.299792458 | m/ns |
| $R$ | 관 반경(외경/2) | m |
| $d_\text{top}$ | 지표에서 관 상단까지 깊이 | m |
| $h$ | 안테나 높이(공기층) | m |
| $s$ | Tx–Rx 간격(공통 오프셋) | m |
| $f_c$ | 안테나 중심주파수 | Hz |

---

## 1. 입력 계약

### 1-1. `Radargram` (불변 dataclass, `src/gpr_detection/io/radargram.py` 구현됨 b0311ed; 코드에는 이 표 외에 `metadata: Mapping` 필드가 더 있다)

| 필드 | 형 | 단위 | 필수 | 설명 |
|---|---|---|---|---|
| `data` | `np.ndarray` float32, shape `[n_samples, n_traces]` | 원 진폭 | 예 | 원본 dtype(int16 등)은 로더에서 float32로 바꾼다. 축 순서는 [시간, 트레이스] 고정 |
| `dt_ns` | float \| None | ns | 예 | None이면 깊이 금지 |
| `dx_m` | float \| None | m | 예 | None이면 깊이 금지 |
| `f_c_hz` | float \| None | Hz | 아니오 | 밴드패스·창 길이에 쓴다. None이면 밴드패스 생략, 창 길이는 기본 ns 값 |
| `eps_r_meta` | float \| None | – | 아니오 | 메타 비유전율. **실측 보증 없음**(§9) |
| `eps_r_meta_source` | str | – | 아니오 | 예: `"4tu_metadata_csv"`, `"synthetic_truth"`, `"assumed"` |
| `air_gap` | `Literal["air","ground","unknown"]` | – | 예 | 4TU = `"air"`, 광저우 IDS·합성 h=0 = `"ground"` |
| `antenna_offset_m` | float \| None | m | 아니오 | $s$. 4TU는 미확인 → None |
| `t0_candidates_ns` | `dict[str, float]` | ns | 아니오 | 메타가 준 영점(4TU `t0_sample_est`·`t0_sample_act_median`은 **샘플 번호 × dt_ns로 ns로 바꿔** 넣는다. 광저우 `t0_sample_used` 동일) |
| `amplitude_processed` | bool | – | 예 | GPR-SD(MATGPR 처리)·광저우 MALA = True |
| `pipe_radius_m` | float \| None | m | 아니오 | 알려진 관경/2. 모르면 None |
| `dataset`, `line_id` | str | – | 예 | 보고·분할 추적용 |

데이터셋별 채움 규약:

| 데이터셋 | dt_ns | dx_m | f_c | air_gap | eps_r_meta | 비고 |
|---|---|---|---|---|---|---|
| gprMax 합성 | json `dt_ns` | json `dx_m` (트레이스 중점 $x=0.4+0.02i$) | json `center_freq_hz` | `air_gap_m>0`이면 air | json `eps_r` (`synthetic_truth`) | s = 0.10 m |
| 4TU | json `dt_ns` (0.098 / 0.195 / 0.117) | 0.02 | 500 MHz | air | `eps_r_used` (`4tu_metadata_csv`) | int16 [512, N]. 09.5 Path8·Path13은 헤더 εr 2.03 비정상 → `eps_r_used` 사용 |
| 광저우 IDS | **시간창 ÷ 샘플 수** (헤더 `data_y_cell` 금지) | `trace_dx_m` | `antenna_MHz` | ground | None | `t0_sample_used` 기준이 섞임(직접파 시작 vs 정점) |
| 광저우 MALA | 동일 | 동일 | None | ground | None | 처리된 자료, t0=0, `amplitude_processed=True` |
| GPR-SD | **가설 A 0.3022 / B 0.2553 둘 다 실행** | 0.03 | None | ground | None | float32 [321, N], 지표 에코 제거·gain 적용됨 |
| AIHub 71327, 모로코 | None | None | – | unknown | – | 깊이 금지 |

### 1-2. 검출 박스 → 시료 좌표

검출기 박스는 백엔드가 모델 입력 좌표(L2)와 `InputTransform`으로 내고, **pipeline이 L2→L1→L0로 공통 역변환**해 원본 `(trace i0..i1, sample k0..k1)`(`GridBox`, 제안)으로 넘긴다(`docs/architecture/README.md` §2-3, ADR-001). 깊이 모듈은 **원본 격자 좌표만** 받는다. 4TU의 트렌치 거리(`distance_m`)를 트레이스 위치로 바로 대응시키지 않는다. 측선 30 m 이상인 약 50개는 지그재그 다측선이라 $x$는 측선 안에서만 $x=i\,\Delta x$로 쓴다.

---

## 2. 출력 계약 — `DepthResult`

| 필드 | 형 | 단위 | 설명 |
|---|---|---|---|
| `status` | `"ok" \| "no_scale" \| "insufficient_picks" \| "fit_failed" \| "zero_failed"` | – | |
| `depth_top_m` | float \| None | m | 주 결과 |
| `depth_center_m` | float \| None | m | $R$을 알 때만 $d_\text{top}+R$ |
| `sigma_depth_m` | float \| None | m | §9 결합 표준불확도 |
| `bias_notes` | list[str] | – | 부호가 정해진 계통 편향(예: `"ground_direct_zero: shallow 0.53–1.03 ns"`) |
| `v_m_per_ns`, `sigma_v` | float | m/ns | |
| `eps_r_used` | float | – | |
| `x0_m`, `t_apex_ns`, `t_zero_ns`, `t0_ns` | float | m, ns | $t_0=t_a-t_z$ |
| `apex_trace`, `apex_sample` | int, float | – | 스케일 없는 입력도 채움 |
| `zero_rule` | str | – | §4 규칙 이름 |
| `zero_event_label` | `"surface_reflection" \| "direct_wave" \| "merged" \| "meta"` | – | 영점이 어떤 사건인지 |
| `v_rule` | str | – | §7 규칙 이름 |
| `radius_used_m` | float | m | 0 또는 기지값 |
| `n_picks`, `inlier_frac`, `rms_ns` | int, float, float | –, –, ns | |
| `h_est_m` | float \| None | m | 공기결합에서 두 사건 간격으로 추정 |
| `alt_depths` | `dict[str, AltDepth]` | m | 메타 영점별 깊이와 사건 라벨(§4-3, 4TU 필수) |
| `flags` | list[str] | – | `air_single_event`, `meta_eps_unverified`, `dt_hypothesis_A/B`, `amplitude_processed`, `apex_at_box_edge` 등 |

`AltDepth = {t_zero_ns, label: "direct_wave"|"surface_reflection"|"unclassified", depth_top_m}`.

---

## 3. 전처리 순서

두 갈래로 나눈다. **피팅 갈래**는 시간 정보를 보존하는 처리만 쓰고, **검출 갈래**만 gain·정규화·리사이즈를 쓴다.

```
raw ─► (1) DC ─► (2) dewow ─► (3) bandpass ─┬─► [영점 검출: 배경 제거 전 평균 트레이스]  (§4)
                                            ├─► (4) 배경 제거 ─► [쌍곡선 점 추출·피팅]   (§5–6)
                                            └─► (4) 배경 제거 ─► (5) gain ─► (6) clip/정규화 ─► (7) letterbox ─► 검출기
```

| 단계 | 식 / 방법 | 기본값 | 근거 |
|---|---|---|---|
| (1) DC | 트레이스별 평균 제거 $a_j(t)\leftarrow a_j(t)-\overline{a_j}$ | 전 구간 평균 | [PA 4][PA 20] |
| (2) dewow | 중심 이동평균 차감, 창 $T_w=3/f_c$ (홀수 샘플로 반올림). $f_c$ 없으면 $T_w=6$ ns | 500 MHz → 6 ns | 컷오프 약 $0.15 f_c$로 대역 밖. 중심 창이라 위상 지연 0 |
| (3) bandpass | Butterworth 4차 SOS, `scipy.signal.sosfiltfilt`(영위상), 통과대역 $[0.25 f_c,\ \min(2f_c,\ 0.45/\Delta t)]$ | 500 MHz → 125 MHz–1 GHz | [PA 4] 100–600 MHz, 정의서 0.25–2배. 영위상이라 영점·꼭짓점 시각이 같은 방식으로 바뀐다 |
| (4) 배경 제거 | 이동 **median** 트레이스 차감, 창 $W_\text{bg}$ 트레이스. 측선이 $W_\text{bg}$보다 짧으면 측선 전체 median | $W_\text{bg}=301$ (4TU 6 m) | mean 차감은 꼭짓점을 +0.022 ns 늦게, median은 −0.0007 ns(§12 표 2). 창은 쌍곡선 폭(±1 m ≈ 100 트레이스)보다 충분히 길게 |
| (5) gain | AGC(RMS 창 $8/f_c$) 또는 SEC. **검출 갈래 전용** | 가정 값, cv 쪽 조정 | 정의서 범위 1 |
| (6) clip | |amp| 99 백분위로 클립 후 [0,255] | GPR-SD는 필수 | readme_gprsd |
| (7) letterbox | 비율 유지 패딩, `ns_per_px`, `m_per_px`, 패딩 오프셋을 메타로 보존 | – | [PA] §0 |

- `amplitude_processed=True`(GPR-SD, 광저우 MALA): (1)–(4)를 다시 하지 않는다(이미 지표 에코 제거·gain 적용). 시간 가변 gain은 포락선 피크를 gain 기울기 방향으로 조금 옮기므로 `flags`에 `amplitude_processed`를 넣는다.
- 공기결합 자료에서 배경 제거는 수평인 지표 반사를 지운다. 그래서 영점은 반드시 (4) **전** 자료로 잡는다([DP] §7-5).

---

## 4. 영점 선택 규칙

### 4-1. 사건 검출 (`detect_zero_events`, 공통)

입력: (3)까지 처리한 자료. 배경 제거 전이다.

1. 평균 트레이스 $\bar a(t)=\frac1N\sum_j a_j(t)$, 포락선 $e(t)=|\mathcal H[\bar a](t)|$. 측선이 $W_\text{bg}$보다 길면 $W_\text{bg}$ 트레이스 창마다 따로 구하고, 창 사이 $t_\text{surf}$ 표준편차를 QC 값으로 낸다.
2. 탐색 창: 기록 앞 25 % 안에서 $e$가 최대인 시각을 $t_\text{max}$로 둔다. 창은 $[0,\ t_\text{max}+2h_\text{max}/c+0.5/f_c]$이고 $h_\text{max}=0.5$ m(가정)다.
3. `scipy.signal.find_peaks(e, height=0.1·max, prominence=0.05·max)`를 쓰고 포물선 보간으로 시각을 잡는다.
4. 첫 피크가 $t_\text{dir}$이다. 창 안 마지막 피크를 `late`로 둔다. 피크가 2개 이상이면 $t_\text{surf}=t_\text{late}$다.
5. 공기층 추정(두 사건일 때): $t_\text{surf}-t_\text{dir}=\dfrac{2\sqrt{h^2+s^2/4}-s}{c}$를 $h$로 푼다. $s$를 모르면 $s=0$으로 두고 $h=c\,(t_\text{surf}-t_\text{dir})/2$로 계산한 뒤 `flags`에 남긴다.

### 4-2. 데이터셋별 규칙

| `air_gap` | 조건 | `zero_rule` | $t_z$ | 라벨 | 편향·불확도 |
|---|---|---|---|---|---|
| air | 사건 2개 이상 | `air_surface_late` | $t_\text{surf}$ (늦은 쪽) | surface_reflection | M20: $t_\text{surf}$가 기준(bg−fs)보다 +0.18 ns 늦음 → 깊이 −0.9 cm. $\sigma_z=0.2$ ns |
| air | 사건 1개(두 사건 간격 < 포락선 폭 ≈ $1.2/f_c$, 즉 $h\lesssim0.15$ m) | `air_single_event` | 그 피크 | merged | 합쳐진 피크가 직접파 위치에 놓여 **깊게** 나온다. M12·M13·M15·M18(h=0.1)에서 +1.8~+3.6 cm(§12 표 4). 편향 상한 $h/\sqrt{\varepsilon_r}$을 `bias_notes`에 넣는다 |
| ground | 항상 | `ground_direct_peak` | $t_\text{dir}$ | direct_wave | 직접파 피크는 송신보다 0.53–1.03 ns 늦다(εr 4–16, s=0.1 m 합성) → **얕게** −1.9~−3.9 cm([DP] 표 D, M02 −3.3 cm) |
| ground | 선택: `ground_zero_shift="synthetic_table"` | `ground_direct_shifted` | $t_\text{dir}-\Delta_\text{dw}(\varepsilon_r)$ | direct_wave | $\Delta_\text{dw}$: εr 4 → 0.53, 9 → 0.725, 16 → 0.96 ns(400 MHz; 500 MHz는 0.53/0.727/1.03). $\sqrt{\varepsilon_r}$에 선형 보간한다. **s = 0.1 m 2D 합성에서 얻은 값이라 기본값은 끈다**(장비 s가 다르면 값이 다름) |
| – | GPR-SD | `meta_sample0` | 0 (가정) | meta | 지표 에코가 제거돼 사건 검출이 불가하다. 깊이에 `zero_unverified` 플래그를 붙인다 |
| – | 광저우 MALA | `meta_sample0` | 0 | meta | 처리본(t0=0). 위상 기준 미상 |

- 광저우 IDS의 `t0_sample_used`(직접파 시작 또는 정점)는 주 영점으로 쓰지 않는다. 직접 검출한 $t_\text{dir}$(포락선 피크)를 쓰고, `t0_sample_used`로 낸 깊이는 `alt_depths["t0_sample_used"]`에만 넣는다(최대 약 1 ns·5 cm 차이, readme_guangzhou).
- first break(5 % 문턱)는 영점으로 쓰지 않는다. M02에서 +7.9 cm, M20에서 +16.9 cm가 나왔다(§12 표 1).

### 4-3. 4TU 메타 영점 두 개의 분류와 보고 (CLAUDE.md 분할 규칙)

1. 메타 영점을 ns로 바꾼다. $t^\text{est}=\texttt{t0\_sample\_est}\cdot\Delta t$, $t^\text{act}=\texttt{t0\_sample\_act\_median}\cdot\Delta t$.
2. 각 값을 §4-1의 $t_\text{dir}$, $t_\text{surf}$ 중 가까운 쪽 사건으로 라벨링한다. 허용폭은 0.5 ns(0.098 ns 자료에서 약 5 샘플)이고, 벗어나면 `unclassified`로 둔다.
3. 깊이를 세 개 낸다. 주 결과는 $t_z=t_\text{surf}$ 결과이고, `alt_depths["t0_sample_est"]`와 `alt_depths["t0_sample_act_median"]`에는 각 메타 영점 결과와 라벨을 넣는다.
4. 메타 t0는 "처음 100 샘플 중 평균 진폭 최대"로 잡은 값이다. 포락선 피크와 위상 기준이 다르다. 합성에서 $\arg\max|\bar a|$는 M02에서 포락선보다 0.27 ns 이르게 잡혔고, M20에서는 **직접파**를 잡아 +9.6 cm가 나왔다(§12 표 1). 그래서 메타 영점은 비교용으로만 쓴다.
5. 라벨이 `direct_wave`인 메타 영점으로 낸 깊이는 체계적으로 깊다. 크기는 εr 9, h 0.3 m에서 약 +8.6 cm다([DP] §3).

---

## 5. 쌍곡선 점 추출 (`extract_picks`)

입력은 배경 제거 후·gain 전 자료(§3 (4)), 박스 `(i0..i1, k0..k1)`, $t_z$다.

1. 박스를 트레이스 방향으로 좌우 10 %씩 넓힌다. 포락선은 $E=|\mathcal H[a]|$를 시간축으로 구한다.
2. **꼭짓점 씨앗**: 박스 안 최대 포락선을 $E_\text{max}$라 한다. 각 트레이스에서 $k>k_\text{mute}$ 구간의 **첫** 피크 가운데 $E\ge0.3E_\text{max}$인 것을 찾는다. 그중 시각이 가장 이른 트레이스를 $j_a$, 그 시각을 $t_a$로 둔다. $k_\text{mute}$는 $t_z+1/f_c$에 해당하는 샘플이다.
   - 근거: 트레이스별 **전역 최대**를 쓰면 물 찬 PVC(M18)에서 바닥·내부 반사를 잡아 깊이가 2.17 m(정답 1.0 m)로 나왔다. 첫 피크 규칙으로 바꾸자 1.027 m가 됐다(§12 표 4).
3. **추적**: $j_a$에서 좌우로 한 트레이스씩 나간다. 탐색 창은 $[t_\text{prev}-0.3,\ t_\text{prev}+2\Delta x/v_\text{min}+0.3]$ ns이고 $v_\text{min}=0.05$ m/ns다. 창 안 포락선 최대를 포물선 보간으로 잡는다. 다음 중 하나면 멈춘다.
   - $E<0.2\,E(t_a)$
   - $|x-x_a|>a_\text{max}$ ($a_\text{max}=1.0$ m, 박스 경계도 적용)
   - 바깥으로 갈 때 $t$가 0.1 ns 넘게 줄어듦(다른 사건으로 넘어간 경우)
4. 점 수가 $n\ge9$이고 양쪽 각각 3개 이상이어야 곡률 피팅을 허용한다. 모자라면 `insufficient_picks`이고, v는 메타 고정만 쓴다.
5. `pick_mode`는 `"envelope"`(기본), `"signed_max"`, `"signed_min"` 중 하나다. **영점 검출과 같은 모드를 강제한다**(API에서 하나의 인자로 공유).
6. 꼭짓점이 박스 좌우 경계 3 트레이스 안에 있으면 `apex_at_box_edge` 플래그를 붙인다.

---

## 6. 피팅

### 6-1. 모델

시각은 모두 영점 기준 $\tau_i=t_i-t_z$ (ns)이다.

- 직선 광선, 반경 $R$ 고정(지면결합):
  $$\tau(x)=\frac{2}{v}\left(\sqrt{\left(\tfrac{v t_0}{2}+R\right)^2+(x-x_0)^2}-R\right),\qquad d_\text{top}=\frac{v\,t_0}{2}$$
  $R=0$이면 점산란 식 $\tau=\frac2v\sqrt{d^2+(x-x_0)^2}$다.
- 굴절(공기층) 모델: Tx $(x-s/2,\,h)$ → 관 상단 정반사점 → Rx $(x+s/2,\,h)$의 Fermat 최소 시간에서 지표 반사 기하 시간 $2\sqrt{h^2+s^2/4}/c$를 뺀다. 알고리즘은 `scripts/synthetic/ray_geometry.py`의 `leg_time`(굴절점 이분법)·`reflection_time`(원주 각 탐색)과 같다. **`src/gpr_detection/depth/refraction.py`로 옮겨 구현**하고 `scripts/`를 import하지 않는다. 각 격자는 721점, 이분법은 40회면 충분하다(M20에서 εr 9.0 복원).

### 6-2. RANSAC 초기값 (직선 광선, $R=0$)

$$\left(\frac{\tau}{2}\right)^2=a+b\,\xi+c\,\xi^2,\qquad \xi=x-x_a$$
$$v=\frac{1}{\sqrt c},\quad x_0=x_a-\frac{b}{2c},\quad t_0=2\sqrt{a-\frac{b^2}{4c}}$$

| 파라미터 | 기본값 | 근거 |
|---|---|---|
| 추정기 | `sklearn.linear_model.RANSACRegressor(LinearRegression())`, 특징 $[\xi,\xi^2]$ | [PA 4] Canny/C3+RANSAC 최저 오차 |
| `residual_threshold` | $(\tau_a/2)\cdot0.3$ ns² (시간 약 0.3 ns) | 0.098 ns 자료 3 샘플 |
| `min_samples` | 3 | |
| `max_trials` | 200 | |
| `random_state` | 0 | 재현 |
| 거부 | $c\le0$ 또는 $v\notin[0.03, c_0]$이면 $(x_a, \tau_a, v_\text{meta}\ \text{or}\ 0.1)$로 대체 | |

### 6-3. 정밀화

`scipy.optimize.least_squares`, 잔차 $r_i=\tau(x_i;x_0,t_0,v)-\tau_i$.

| 파라미터 | 기본값 |
|---|---|
| `loss` | `"soft_l1"` (`"huber"` 선택 가능) |
| `f_scale` | 0.1 ns (약 1 샘플) |
| `bounds` | $x_0\in[x_\text{min},x_\text{max}]$, $t_0\in[0,t_\text{max}]$, $v\in[0.03,\ 0.2998]$ m/ns |
| `x_scale` | `"jac"` |
| 미지수 | $(x_0,t_0,v)$. **$R$은 고정** |
| 공분산 | $\mathrm{Cov}=s^2(J^\top J)^{-1}$, $s^2=\sum r_i^2/(n-3)$ → $\sigma_v,\sigma_{t_0}$ |
| 보고 | `rms_ns`, `inlier_frac`(|r| < 3·f_scale 비율) |

### 6-4. v 고정 시 꼭짓점 시각

$v$를 고정하면 깊이에 필요한 것은 $t_0$뿐이다. 공기결합 날개는 직선 광선 식과 맞지 않으므로 날개로 $t_0$를 맞추지 않는다. $|x-x_a|\le0.10$ m 안의 점(5개 이상)에 $\tau=p_0+p_1\xi+p_2\xi^2$를 맞추고, $t_0=p_0-p_1^2/(4p_2)$, $x_0=x_a-p_1/(2p_2)$로 둔다. $p_2\le0$이거나 점이 5개 미만이면 $t_0=\tau_a$를 쓴다.

---

## 7. 속도 결정 규칙

| 순서 | 조건 | `v_rule` | v | 곡률 피팅 |
|---|---|---|---|---|
| 1 | `dt_ns` 또는 `dx_m` 없음 | `none` | – | 깊이 금지. `status="no_scale"`, 꼭짓점 픽셀만 |
| 2 | `air_gap="air"` (4TU, 합성 M12·M13·M15·M18·M20) | `meta_fixed` | $c/\sqrt{\varepsilon_{r,\text{meta}}}$ | **직선 광선 곡률 피팅 금지.** 호출 시 `ValueError` |
| 2a | 위 + 사건 2개(h 추정 가능) + 점 충분 | `refraction_qc` (부가) | 굴절 모델로 εr 격자 3–30, 간격 0.25 → 최소 RMS 주변을 `minimize_scalar`로 정밀화. $d$는 꼭짓점 시각에 묶는다 | QC 전용이다. $|\varepsilon_\text{fit}-\varepsilon_\text{meta}|/\varepsilon_\text{meta}>0.3$이면 `meta_eps_suspect` 플래그. 주 깊이는 `meta_fixed` 결과로 낸다 |
| 3 | `air_gap="ground"`, 점 충분 | `curvature_R_fixed` | §6-3 결과 | $R$ = 기지 관경/2, 없으면 0 |
| 3a | 위 + `eps_r_meta` 있음 | 둘 다 보고 | 주 결과는 곡률. `alt_depths["meta_fixed"]` | |
| 4 | `air_gap="ground"`, 점 부족, `eps_r_meta` 있음 | `meta_fixed` | | |
| 5 | `air_gap="ground"`, 점 부족, 메타 없음 | `none` | – | `insufficient_picks` (가정 v 0.1 m/ns로 깊이를 내지 않는다) |

- $R$을 모른 채 $R=0$으로 곡률 피팅하면 $v'/v$가 1.01–1.14배로 커진다. $d'$는 상단보다 0.2R–0.5R 깊게 나온다([DP] 표 B). `bias_notes`에 `"radius_unknown: v,d over by up to +14 %"`를 넣는다.
- GPR-SD: 지면결합 점산란 식에서 시간축을 $k$배로 늘리면 $t\to kt$, $v\to v/k$가 되고 $d=vt_0/2$는 변하지 않는다. 그래서 **dt 가설 A/B는 곡률 기반 깊이에 영향을 주지 않고 $\varepsilon_r$만 바꾼다.** 두 가설을 모두 돌려 깊이가 같은지 확인한다(단위 시험 U5). 다만 영점이 미확정(`meta_sample0`)이라 깊이에는 `zero_unverified` 플래그를 붙인다.

---

## 8. 깊이 계산

| 경우 | 식 |
|---|---|
| 지면결합, $s$ 모름 | $d_\text{top}=v\,t_0/2$ |
| 지면결합, $s$ 앎 | $d_\text{top}=\sqrt{(v t_0/2)^2-(s/2)^2}$ (영향 $\approx s^2/(8d)$, s=0.1 m에서 2–4 mm) |
| 공기결합 | $d_\text{top}=v\,(t_a-t_\text{surf})/2$ (꼭짓점의 굴절·오프셋 영향은 M20에서 −0.9 cm 안) |
| 반경 앎 | $d_\text{center}=d_\text{top}+R$ |

---

## 9. 불확도

$$\sigma_d=\sqrt{\left(d\,\frac{\sigma_v}{v}\right)^2+\left(\frac v2\right)^2\left(\sigma_z^2+\sigma_a^2\right)}$$

| 성분 | 기본값 | 근거 |
|---|---|---|
| $\sigma_v/v$ (`meta_fixed`) | $\tfrac12\,\sigma_{\varepsilon}/\varepsilon$, $\sigma_\varepsilon/\varepsilon=0.15$ (**가정**) | 4TU `eps_r_used`는 조사자 기록이다. Reflex-W 쌍곡선 피팅에서 $\varepsilon=(c/v)^2$로 얻었을 가능성이 있다([PA] §5). 공기결합 곡률 피팅값이면 $v$가 과대이고 $\varepsilon$은 과소이므로 **깊이가 깊게** 나오는 한쪽 편향이 된다. `flags`에 `meta_eps_unverified`를 넣는다 |
| $\sigma_v/v$ (`curvature_R_fixed`) | $\sqrt{(\sigma_v^\text{fit}/v)^2+0.03^2}$ | 실측 픽 곡률 편향 1.001–1.052([DP] 표 F 반경 열) |
| $\sigma_z$ | `air_surface_late` 0.2 ns, `air_single_event` 0.25 ns(+편향), `ground_direct_*` 0.15 ns(+편향), `meta_sample0` 0.5 ns | §12 표 1·4 |
| $\sigma_a$ | $\max(0.05\ \text{ns},\ \text{rms}/\sqrt{n_\text{apex}})$ | 5 mm 격자 지연 0.05–0.1 ns([DP] §6) |
| 계통(합산하지 않음) | 기준점 상단/중심 $\pm R$, 영점 편향 부호 | `bias_notes` |

---

## 10. 경계 조건·실패 처리

| 상황 | 처리 |
|---|---|
| 박스가 영점 사건과 겹침($k_0\Delta t<t_z+1/f_c$) | 박스 위쪽을 $t_z+1/f_c$로 자르고, 남은 높이가 $2/f_c$보다 작으면 `fit_failed` |
| 꼭짓점 2개 이상(인접 관) | 씨앗은 가장 이른 것 하나만 쓴다. 다중 관 분리는 v2 |
| 꼭짓점이 박스 경계 | 플래그만 붙이고 계속 |
| RANSAC 실패($c\le0$) | 메타 v로 §6-4 |
| `least_squares` 미수렴 또는 경계 값에 붙음 | `fit_failed`, 메타 v가 있으면 §6-4 결과를 `alt_depths`에 |
| 공기결합에서 사건 0개 | `zero_failed`, 깊이 없음 |
| int16 포화(±32767) | 포화 샘플이 꼭짓점 ±2 샘플 안에 있으면 `saturated_apex` |
| $\Delta t$가 커서 포락선 피크가 3 샘플 이하 폭 | 포물선 보간은 유지하고 $\sigma_a\ge\Delta t/2$ |
| 스케일 없음 | `no_scale`, `apex_trace`·`apex_sample`만 |

---

## 11. 시험

난수 시드는 모두 0이다. 시간 단위는 ns, 거리 단위는 m다.

### 11-1. 단위 시험 (해석 합성, `tests/depth/test_*.py`)

합성 B-scan 생성기(시험용): 트레이스 $x_j=j\Delta x$, $\Delta x=0.02$, $\Delta t=0.05$ ns, 400 MHz Ricker를 $t_j=t_\text{src}+\tau(x_j)$에 놓는다. 직접파는 $t_\text{src}+0.7$, 공기결합이면 지표 반사를 $t_\text{src}+2h/c$에 둔다. 백색잡음 SNR 20 dB(꼭짓점 피크 대비).

| ID | 내용 | 통과 기준 |
|---|---|---|
| U1 | 점산란 $R=0$, εr 9, d 0.6 m, 지면결합, 영점=송신 | $|v/v_\text{true}-1|\le5\%$, $|d-0.6|\le2$ cm |
| U2 | 원통 $R=0.1$, $R$ 고정 피팅 | 같음. $R=0$으로 피팅하면 $v'/v>1$ (편향 부호 확인) |
| U3 | U1 픽의 20 %를 ±3 ns 이상치로 교체 | U1 기준 유지, `inlier_frac`≥0.75 |
| U4 | 공기결합 h 0.3 m(굴절 모델로 생성), `air_gap="air"` | 직선 광선 곡률 호출은 `ValueError`. `meta_fixed` 깊이 ±2 cm. `refraction_qc` εr 9±0.5 |
| U5 | U1 자료의 시간축을 1.18배(GPR-SD 가설 B/A 비)로 늘림 | $d$ 변화 ≤ 1 mm, $v$는 $1/1.18$배 |
| U6 | `pick_mode` 불일치(영점 envelope, 꼭짓점 signed) | 호출 거부 |
| U7 | 스케일 없음(`dt_ns=None`) | `status="no_scale"`, 깊이 None |
| U8 | 늦은 강한 사건(바닥 반사 진폭 2배)이 있는 원통 | 씨앗이 상단 반사(첫 피크)를 잡음 |

### 11-2. 회귀 시험 (gprMax 합성 20개, json 정답)

정답 필드:
- `pipe_top_depth_m`, `velocity_m_per_ns`, `pipe_radius_m`, `air_gap_m`, `t0_top_ns`
- `time_zero_candidates_ns.{source_peak, direct_wave_peak, surface_reflection_peak}`
- `apex.envelope_peak_ns`, `apex.trace_index`
- `ray_reflection_time_ns` (트레이스별 픽 검증용: 픽 − (source_peak + wavelet_lag) 와 비교)

`signed_peak_polarity`는 극성 판정에 쓰지 않는다(2D 합성, [DP] §8).

| ID | 대상 | 검사 | 기준 | 현재 근거값 |
|---|---|---|---|---|
| R1 | h=0 (M01–M11, M14, M16, M17, M19) | $t_\text{dir}$ vs `direct_wave_peak` | ≤0.05 ns | M02 0.0000 ns |
| R2 | M20 (h 0.3) | 사건 2개, $t_\text{surf}$ vs `surface_reflection_peak` | ≤0.25 ns, $|h_\text{est}-0.3|\le0.05$ m | +0.184 ns, 0.330 m |
| R3 | h 0.1 (M12, M13, M15, M18) | `zero_rule == "air_single_event"`, 단일 피크 vs `direct_wave_peak` | ≤0.05 ns | 4개 모두 단일 사건, 편차 0 |
| R4 | 20개 전부 | 꼭짓점 트레이스 vs `apex.trace_index` | ±3 트레이스 | 측정한 6개(M02·M12·M13·M15·M18·M20) 52–55 |
| R5 | 20개 전부 | $t_a$(median 배경) vs `apex.envelope_peak_ns` | ≤0.05 ns | M02 +0.0003, M20 −0.0007 |
| R6 | M20 | `meta_fixed` 깊이 | $|d-d_\text{top}|\le2$ cm (일치) | −0.88 cm |
| R7 | M12, M13, M15, M18 | `meta_fixed` 깊이, 편향 부호 + | $0<d-d_\text{top}\le5$ cm | +1.8, +3.6, +2.0, +2.8 cm |
| R8 | h=0 전부 | `meta_fixed` + `ground_direct_peak` | $-5\le d-d_\text{top}<0$ cm | [DP] 표 D −1.9~−3.9, M02 −3.26 |
| R9 | h=0 전부 | `meta_fixed` + `ground_direct_shifted`(표 $\Delta_\text{dw}$) | $|d-d_\text{top}|\le2$ cm | M02 송신 영점 +0.36 cm |
| R10 | h=0 전부 | `curvature_R_fixed`($R$=json), 영점 `ground_direct_peak` | $|v/v_\text{true}-1|\le6\%$, $|d-d_\text{top}|\le5$ cm. 5 %·2 cm 안이면 "일치", 아니면 "허용 일치"로 기록 | M02 +3.4 %, −1.6 cm. [DP] 표 F 최대 +5.2 %(M09), −2.4 cm |
| R11 | h=0 전부 | $R=0$ 곡률 피팅 | $v'/v>1$ (편향 부호) | M02 1.098 |
| R12 | M20 | 직선 광선 곡률 피팅 요청 | `ValueError`. 참고로 강제 실행 시 $v'/v$ 1.82, +47.8 cm | §12 표 3 |
| R13 | M20 | `refraction_qc` | εr 9±0.5, $d$ ±2 cm | εr 9.00, −0.83 cm |

- R10 기준을 5 %가 아니라 6 %로 둔 이유: 직접파 영점과 반경 모델을 함께 써도 M09(εr 4)에서 +5.2 %가 실측됐다([DP] 표 F). 판정 등급은 정의서 verify 기준(일치 ≤5 %·2 cm, 허용 일치 d ≤10 cm)으로 따로 기록한다.
- 공기층 모델(M12·M13·M15·M18·M20)은 곡률 v 기준을 적용하지 않고 고정 v의 d 기준만 본다(정의서 verify 규칙).
- 회귀 시험은 합성 원본 npy를 읽기만 한다. 20개 전체 실행 비용은 Hilbert 2D 20회 + 피팅이라 단일 프로세스 수십 초다(굴절 QC는 M20만).

---

## 12. 근거: analyze 결과 요약

실행: `conda run -n gpr-dev python -I analysis/processing/zero_reference_check.py` (단일 프로세스, 수 초, 시드 0). 꼭짓점은 트레이스 55(x = 1.5 m), v는 json 정답(εr 9 → 0.09993 m/ns)이다.

### 표 1. 영점별 깊이 (꼭짓점 = 이상적 배경 차감 포락선 피크)

| 모델 | 영점 | $t_z$ (ns) | $t_a-t_z$ (ns) | json $t_{0,\text{top}}$ (ns) | 차 (ns) | $d$ (m) | 오차 (cm) |
|---|---|---|---|---|---|---|---|
| M02 (h 0) | 송신 피크(이상) | 3.536 | 12.080 | 12.008 | +0.072 | 0.604 | +0.4 |
| M02 | 직접파 포락선 피크(블라인드) | 4.260 | 11.355 | 12.008 | −0.653 | 0.567 | **−3.3** |
| M02 | argmax\|평균 트레이스\| (4TU 메타 규칙) | 3.986 | 11.630 | 12.008 | −0.379 | 0.581 | −1.9 |
| M02 | first break 5 % | 2.017 | 13.599 | 12.008 | +1.591 | 0.680 | +7.9 |
| M20 (h 0.3) | 송신 피크 | 2.828 | 14.065 | 12.008 | +2.056 | 0.703 | +10.3 |
| M20 | 직접파(블라인드 첫 피크) | 3.164 | 13.729 | 12.008 | +1.720 | 0.686 | **+8.6** |
| M20 | **늦은 사건(블라인드) = 지표 반사** | 5.060 | 11.833 | 12.008 | −0.176 | 0.591 | **−0.9** |
| M20 | 지표 반사 기준(no-pipe − free-space) | 4.877 | 12.016 | 12.008 | +0.008 | 0.600 | +0.04 |
| M20 | argmax\|평균 트레이스\| | 2.972 | 13.921 | 12.008 | +1.913 | 0.696 | +9.6 |
| M20 | first break 5 % | 1.498 | 15.395 | 12.008 | +3.387 | 0.769 | +16.9 |

결론:
- **지면결합**에서는 데이터에서 잡을 수 있는 최선의 영점이 직접파 포락선 피크다. 그래도 −3.3 cm 얕다. 원인은 직접파 피크가 송신보다 0.725 ns 늦다는 점이고, 송신 시각을 알면 +0.4 cm로 맞는다.
- **공기결합**에서는 늦은 사건(지표 반사)이 맞는 영점이다(−0.9 cm). 직접파·송신·4TU 메타 규칙을 쓰면 +8.6~+10.3 cm 깊게 나온다.

### 표 2. 배경 제거 방식별 꼭짓점 시각 (ns)

| 모델 | 이상(무관 모델 차감) | 평균 트레이스 | median 트레이스 |
|---|---|---|---|
| M02 | 15.6158 | 15.6381 (+0.022) | 15.6161 (+0.0003) |
| M20 | 16.8929 | 16.9151 (+0.022) | 16.8922 (−0.0007) |

### 표 3. 곡률 피팅 (|x−x_a| ≤ 1.0 m, 이상 배경 차감, 포락선 픽)

| 모델 | 영점 | 피팅 | 픽 | $v'/v$ | εr | $d$ (m) | 오차 (cm) | RMS (ns) |
|---|---|---|---|---|---|---|---|---|
| M02 | 직접파 | 직선 R=0, (x0,t0,v) | 101 | 1.098 | 7.47 | 0.618 | +1.8 | 0.098 |
| M02 | 직접파 | 직선 R=0.1 고정 | 101 | 1.034 | 8.42 | 0.584 | −1.6 | 0.069 |
| M02 | 직접파 | v 고정(메타) | 1 | 1 | 9 | 0.567 | −3.3 | – |
| M20 | 지표 반사 | 직선 R=0 | 85 | **1.816** | 2.73 | 1.078 | **+47.8** | 0.044 |
| M20 | 지표 반사 | 직선 R=0.1 고정 | 85 | **1.742** | 2.97 | 1.035 | **+43.5** | 0.046 |
| M20 | 지표 반사 | v 고정(메타) | 1 | 1 | 9 | 0.591 | −0.9 | – |
| M20 | 지표 반사 | 굴절 격자(h_est 0.330 m, R 0.1, s 0.1) | 43 | 1.000 | 9.00 | 0.592 | −0.8 | 0.010 |

- M02 R=0 결과(+1.8 cm)가 정답에 가까워 보이는 것은 두 편향이 상쇄된 결과다. v 과대(+9.8 %)와 영점 얕음(−3.3 cm)이 서로 지웠다. 맞는 모델이라서가 아니다.
- M20 직선 광선 결과는 [DP] 표 C·F(1.81, +0.49 m)를 재현한다. 직선 광선에서는 RMS가 0.044 ns로 작아 **잔차로는 실패를 알 수 없다**. 그래서 규칙으로 금지한다(§7 순서 2).

### 표 4. 공기층 모델 블라인드 영점 (v = 정답, 꼭짓점 = median 배경, 첫 피크 씨앗)

| 모델 | h (m) | f (MHz) | 검출 사건 (ns) | 늦은 사건 − json 지표 반사 (ns) | $h_\text{est}$ (m) | $d$ 오차 (cm) | json 지표 반사 영점일 때 (cm) |
|---|---|---|---|---|---|---|---|
| M12 | 0.1 | 400 | [3.890] | −0.449 | – | +1.8 | −0.4 |
| M13 | 0.1 | 500 | [3.193] | −0.429 | – | +3.6 | +0.3 |
| M15 | 0.1 | 400 | [3.910] | −0.423 | – | +2.0 | +0.4 |
| M18 | 0.1 | 400 | [3.890] | −0.450 | – | +2.8 | +0.5 |
| M20 | 0.3 | 500 | [3.164, 5.060] | +0.184 | 0.330 | −0.9 | +0.04 |

- h = 0.1 m에서는 직접파와 지표 반사의 간격이 약 0.45 ns로 포락선 폭보다 짧다. 그래서 피크가 하나로 합쳐진다(`air_single_event`). 4TU는 두 사건 간격이 약 20 샘플(1.95 ns, h ≈ 0.29 m)이라 M20처럼 분리될 것으로 예상한다([DP] §7-2). 현장 확인은 analyze 다음 단계다.
- 꼭짓점 씨앗을 전역 최대로 잡으면 M18이 +116.7 cm로 실패했고, 첫 피크 규칙으로 +2.8 cm가 됐다(§5-2).

그림: `docs/processing/zero_reference_check.png`(꼭짓점 트레이스 포락선과 영점 후보 표시, M02·M20).

---

## 13. 구현 배치 제안 (python-expert)

| 모듈 | 내용 |
|---|---|
| `src/gpr_detection/io/radargram.py` | `Radargram` dataclass와 `sample_to_ns`(구현됨). 데이터셋별 로더는 별도 파일 `io/{fourtu,guangzhou,gprsd}.py`(구현됨)와 `io/synthetic.py`(예정, CHECKLIST M2-4b) |
| `src/gpr_detection/preprocess/filters.py` | `remove_dc`, `dewow`, `bandpass`, `remove_background(median, window)`, `agc` |
| `src/gpr_detection/preprocess/timezero.py` | `detect_zero_events`, `select_zero(radargram, events) -> (t_z, zero_rule, label)`, `classify_meta_t0` |
| `src/gpr_detection/depth/picks.py` | `extract_picks(box, ...)`, 첫 피크 씨앗, 추적 |
| `src/gpr_detection/depth/hyperbola.py` | `ransac_init`, `fit_hyperbola(R fixed)`, `apex_vertex` |
| `src/gpr_detection/depth/refraction.py` | `leg_time`, `reflection_time`, `refraction_qc` (ray_geometry 알고리즘 이식, MIT 호환 자체 코드) |
| `src/gpr_detection/depth/estimate.py` | `estimate_depth(radargram, box, config) -> DepthResult`, §7 결정표, §9 불확도 |
| `tests/depth/` | §11-1 단위 시험, §11-2 회귀 시험(합성 npy 읽기, `@pytest.mark.synthetic`) |

설정 기본값은 `DepthConfig` dataclass 하나에 모은다: `a_max_m=1.0`, `seed_rel=0.3`, `track_stop_rel=0.2`, `v_min=0.05`, `ransac_dt_ns=0.3`, `ransac_trials=200`, `f_scale_ns=0.1`, `apex_halfwidth_m=0.10`, `bg_window=301`, `h_max_m=0.5`, `ground_zero_shift="none"`, `sigma_eps_rel=0.15`, `seed=0`.

---

## 14. 가정과 미결

| 항목 | 현재 처리 | 확인 방법 |
|---|---|---|
| 4TU `eps_r_used`의 출처(실측 vs 곡률 피팅값) | 가정: 미검증, $\sigma_\varepsilon/\varepsilon=0.15$ | 활동 몇 개에서 `refraction_qc`와 비교(다음 analyze) |
| 4TU Tx–Rx 간격 $s$ | 미확인 → $h$ 추정에서 $s=0$ | 장비 사양(InfraRadar) 조사 |
| 4TU 정답 깊이 기준(상단/중심) | 미정 → 두 가정 모두 보고 | – |
| 지면결합 $\Delta_\text{dw}$ 보정 | 2D 합성(s=0.1 m)에서만 얻은 값, 기본값 끔 | 광저우 깊이 기지 표적이 생기면 보정 |
| GPR-SD 영점 | 샘플 0 가정 | 원 저장소 처리 내역 |
| $h_\text{max}=0.5$ m, AGC 창 $8/f_c$, $\sigma_\varepsilon$ 0.15 | 가정 | 4TU analyze |
