# 명세 v1 — 쌍곡선 피팅 기반 관로 깊이 추정 (전처리 · 영점 · 피팅 · 속도 규칙 · 회귀 시험)

- 작성: gpr-processing-expert, 2026-10-29 (spec 모드)
- 받는 쪽: python-expert (구현), cv-detection-expert (박스 → 시료 좌표 계약만)
- 1차 근거: `docs/research/depth_physics.md`(이하 [DP]), `docs/research/prior_art.md`(이하 [PA], 출처 번호는 그 문서의 [n]), `docs/datasets/readme_*.md`·`codebook_4tu.md`, `data/synthetic_gprmax/*.json`·`manifest.csv`, `docs/processing/spec_band_depth_v1.md`(이하 [B], v1.1부터 참조)
- 이 명세를 위한 탐색 실행: `analysis/processing/zero_reference_check.py` → `docs/processing/zero_reference_check.csv`, `zero_reference_curvature.csv`, `zero_reference_airgap_models.csv`, `zero_reference_check.png` (§12에 요약)
- v1.2 지표 반사 영점 지연 분해: `conda run -n gpr-dev python -I analysis/processing/surface_zero_shift.py` (단일 프로세스, 수 초, 생성기 시드 0) → `docs/processing/surface_zero_shift.csv`(단계·사건 구성·픽 방식별 이동), `surface_zero_shift_hsweep.csv`(공기층·진폭비별 편향), `surface_zero_shift_z1.csv`(후보 처리별 Z1 사건 수), `surface_zero_shift.png`(단독/직접파/합성 포락선, 해석 Ricker·M20)

### 변경 이력

| 날짜 | 판 | 내용 |
|---|---|---|
| 2026-10-30 | v1.1 | M2-1·M2-2 구현 review 반영: dewow 커널을 상자형에서 가우스($\sigma=1/f_c$)로 바꿈(상자형 유령 엽 ±$T_w/2$, §3), 스케일 없음·`f_c` 없음·배경 창 끝·AGC 기본값 명시(§3, §4-1), 창별 영점 QC를 선택 사건 추적으로 바꿈(§4-1), `air_single_event` 분리 한계·편향 상한 식 수정과 메타 영점 라벨 규칙(`merged`, 지면결합 `direct_wave`만) 보완(§4-2·§4-3), `air_gap="unknown"`·$\Delta_\text{dw}$ 적용 범위(§4-2), R2 허용폭(§11-2), `select_zero`→`ZeroResult`와 필드 정본(§13-1) |
| 2026-10-30 | v1.2 | 공기결합 지표 반사 영점 지연 원인 조사 반영(사용자 결정 2026-10-30, 권고 (b) 수정안). 지연은 처리 편향이 아니라 **직접파 간섭 편향**이다(단독 지표 반사는 (1)–(3) 뒤 ≤0.003 ns). `air_surface_late` 편향 문구(§4-2)와 $\sigma_z$ 0.2→0.3 ns(§9), Z1 꼬리 진폭 0.6→0.8·Z6(단독 무편향)·Z7(두 사건 간섭 창, 기존 두 사건·UB14 마스크 시험의 시각 기준 대체) 추가(§11-1), R2 근거값 분해·R6 근거값 v1.1 처리 기준 −1.42 cm(§11-2). 근거: `analysis/processing/surface_zero_shift.py` → [`surface_zero_shift.csv`](surface_zero_shift.csv), [`surface_zero_shift_hsweep.csv`](surface_zero_shift_hsweep.csv), [`surface_zero_shift_z1.csv`](surface_zero_shift_z1.csv)(Z1 꼬리 0.6 한 조건뿐. 꼬리 0.8 판별은 시험 Z1(a·b), §11-1), [`surface_zero_shift.png`](surface_zero_shift.png) |
| 2026-10-30 | v1.2 (M2-2 대조) | M2-2 구현 확인 반영: Z1 상자형 판별 대조를 상시 시험으로(사용자 결정), Z3·Z5 얕은 반사 값($\tau$ 3.5 ns, 진폭 −1.5, 반대 극성)과 분리 한계 근거, Z6 잡음 없음 명시(§11-1), `ZeroEvents` 필드(§13-2), `classify_meta_t0` 시그니처·`ZeroConfig`·$\Delta_\text{dw}$ 범위 상수(§13-3) |
| 2026-10-30 | v1.3 (M2-3 대조) | M2-3 구현 판정 반영:<br>- 스케일 없는 입력에 픽셀 단위 밴드패스 `bandpass_px` 추가(§3, 꼭짓점 후보 잡음 대책)<br>- 꼭짓점 후보·추적 탐색 범위와 뮤트 규칙(§5-4)<br>- RANSAC 거부는 초기값 대체 후 계속 + `ransac_rejected`(§6-2·§10, v1 §10 "메타 v로 §6-4" 행과의 충돌 해소)<br>- $t_\text{max}=\max\tau$·경계 판정 $10^{-4}$·공분산을 정상점과 유사역행렬로(§6-3)<br>- §6-4 꼭짓점 창 밖이면 꼭짓점 후보로 대체<br>- 공기결합 거부를 점 수 검사보다 먼저(§7)<br>- 플래그 `box_in_zero_zone`·`no_apex_candidate`·`mute_unavailable`·`ransac_rejected`·`bandpass_px_skipped`(§2·§10)<br>- U5 $f_c$ 함께 축척·U7 잡음 자료 유지(§11-1)<br>- `DepthConfig` 상수 확인과 `single_event_h_max_m` 삭제(§13-1 아래)<br>- 픽·피팅 객체 필드 정본(§13-4)<br>- (integrity 반영) §5-2 꼭짓점 후보 문구를 "문턱을 넘는 첫 피크"로 명확화, §10 `bandpass_px_skipped` 행, §13 모듈 표 시그니처, §13-4 `CurvatureFit` status 규칙(꼭짓점 없음 → `fit_failed`) |
| 2026-10-30 | v1.4 (M2-4 대조) | M2-4 구현 판정 반영:<br>- `status="no_velocity"`, `target_class`, `eps_r_refraction_qc(_range)`, `sigma_zero_ns`, `sigma_apex_ns`, `alt_depths` 채우는 규칙(§2)<br>- 굴절 격자 한 단·꼭짓점 $d$ 결정(§6-1)<br>- §7 2a를 $h_\text{est}$ 간섭 편향에 견디는 **구간 판정**으로 바꿈, 2a 세부, 순서 2b(공기결합·메타 없음)·6(`unknown` → `no_velocity`, 곡률 금지)<br>- §9 $\sigma_a$ 정의, §10 좁은 피크 판정식·$s$ 보정 0 처리<br>- U4 조건(500 MHz)·기준 재정의, R13 두 경로(§11)<br>- §13 refraction·velocity·result·estimate 시그니처<br>- 근거: 정확한 광선 픽 εr 민감도. $h$ 0.300–0.349 m로 상한 쪽을, 0.25·0.269·0.289 m로 구간 하한 쪽을 쟀다(§7 2a, review 계산, scratchpad, CSV 없음)<br>- (구현 확인) 구간을 min/max로 정렬, U4 실측 구간 (6.85, 9.70)의 분해(R 불일치 −6~−9 %), U4(d) 메타값 5 → 2.7, §13 `refraction_h_margin_m`·`eps_r_refraction_qc_range` |
| 2026-10-30 | v1.5 (M2-4b 대조) | 합성 로더 반영: §1-1 합성 채움 규약(`eps_r_meta_source="synthetic_truth"`, `t0_candidates_ns` 비움, 끄기 옵션), §11-2 입력 규약(`SyntheticTruth` 필드 용도와 필수/선택, x축 비교 규약, **M2-5 박스 공급 규칙**, `_fs.npy` 불필요), §13 `io/synthetic.py` 행 |
| 2026-10-30 | v1.6 (M2-5 회귀 판정) | 사용자 결정 반영:<br>- R5 기준을 같은 (1)–(3)을 거친 이상 배경의 꼭짓점 포락선 피크로 바꾸고 필터 이동은 별도 기록(M15 밴드패스 −0.055 ns)<br>- R13을 R13a-1(json 광선 시각 픽, εr 9±0.1·d ±1 cm), R13a-2(현재 경로 픽·json h, εr ≤20 %·d ≤5 cm, 날개 지연 −13 % 기록, 기본 격자·전체 픽), R13b로 나눔<br>- R6 실측 −1.49 cm, R2·R13b $h_\text{est}$ +0.049 m<br>- §12 표 3 굴절 9.00에 "h_est 0.330 상쇄" 주석<br>- §7 R 미상 편향 문구 +14 → +17 %(R11 M01·M06 +16.5 %)<br>- §14 Δh 여유 0.011 m<br>- 근거: `analysis/processing/regression_failures_m2_5.py` |
| 2026-10-30 | v1.7 (band v1.1 반영 대조) | §3 `bandpass_px` 주기 범위 하한 4 → 2.2 px(AIHub 반송파 2.44–4.83 px 실측, 근거 `analysis/processing/aihub_period_px_check.py`), §10 해당 행, §13 filters 행에 `dominant_period_px`·`FittingBranch.period_px` |
| 2026-10-31 | v1.7 (용어) | 2026-10-31 용어 변경: 씨앗 → 꼭짓점 후보(`ApexSeed`→`ApexCandidate`, `find_apex_seed`→`find_apex_candidate`, `no_apex_seed`→`no_apex_candidate`, `seed_rel`→`candidate_rel`). 내용 변경 없음. 난수 시드(`seed=0`, `random_state`)는 그대로 |

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
| gprMax 합성 (`io/synthetic.py` `load_synthetic`, v1.5) | json `dt_ns` | json `dx_m`. `Radargram`에서는 $x=i\,\Delta x$(0부터)이고 json 축은 $0.4+0.02i$(`metadata["trace_x_m_first"]`, §11-2 x축 규약) | json `center_freq_hz` | `air_gap_m>0`이면 air, 아니면 ground | json `eps_r`, `eps_r_meta_source="synthetic_truth"`(`include_eps=False`면 None·`""`) | `antenna_offset_m` = json(0.10 m), `pipe_radius_m` = json(`include_radius=False`면 None), `amplitude_processed=False`. **`t0_candidates_ns`는 비운다**(정답 영점·꼭짓점·깊이는 `load_synthetic_truth`로만). `metadata`는 `model_id`, `trace_x_m_first`, `time_of_sample0_ns`(보고용; 20개 모두 0.0이고 로더는 시간축에 반영하지 않으므로 0이어야 한다). 로더는 정답용 필수 필드가 빠진 json도 거부한다(`load_synthetic_truth`와 같은 검사) |
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
| `status` | `"ok" \| "no_scale" \| "insufficient_picks" \| "fit_failed" \| "zero_failed" \| "no_velocity"` | – | `no_velocity`(v1.4, [B] §5와 같은 이름): 스케일·영점·꼭짓점은 있으나 v 출처가 없음. 공기결합 + `eps_r_meta` 없음, 또는 `air_gap="unknown"`(§7). 지면결합 순서 5(점 부족 + 메타 없음)는 `insufficient_picks`로 둔다 |
| `target_class` | str | – | `"pipe_hyperbola"`(M2-6b에서 `"pipe_band"`) (v1.4) |
| `depth_top_m` | float \| None | m | 주 결과 |
| `depth_center_m` | float \| None | m | $R$을 알 때만 $d_\text{top}+R$ |
| `sigma_depth_m` | float \| None | m | §9 결합 표준불확도 |
| `bias_notes` | list[str] | – | 부호가 정해진 계통 편향(예: `"ground_direct_zero: shallow 0.53–1.03 ns"`) |
| `v_m_per_ns`, `sigma_v` | float | m/ns | |
| `eps_r_used` | float | – | |
| `x0_m`, `t_apex_ns`, `t_zero_ns`, `t0_ns` | float | m, ns | $t_0=t_a-t_z$ |
| `apex_trace`, `apex_sample` | int, float | – | 스케일 없는 입력도 채움 |
| `zero_rule` | str | – | §4 규칙 이름 |
| `zero_event_label` | `"surface_reflection" \| "direct_wave" \| "merged" \| "meta" \| None` | – | 영점이 어떤 사건인지. `None`은 `zero_rule="zero_failed"`일 때만 |
| `v_rule` | str | – | §7 규칙 이름 |
| `radius_used_m` | float | m | 0 또는 기지값 |
| `n_picks`, `inlier_frac`, `rms_ns` | int, float, float | –, –, ns | |
| `h_est_m` | float \| None | m | 공기결합에서 두 사건 간격으로 추정 |
| `alt_depths` | `dict[str, AltDepth]` | m | 메타 영점별 깊이와 사건 라벨(§4-3, 4TU 필수) |
| `flags` | list[str] | – | `air_single_event`, `meta_eps_unverified`, `dt_hypothesis_A/B`, `amplitude_processed`, `apex_at_box_edge`, `box_in_zero_zone`, `no_apex_candidate`, `mute_unavailable`, `ransac_rejected`, `bandpass_px_skipped`(v1.3, §3·§5·§6-2·§10) 등 |

`AltDepth = {t_zero_ns, label: "direct_wave"|"surface_reflection"|"merged"|"unclassified", depth_top_m}` (`merged`는 v1.1, §4-3).

v1.4 추가 필드(M2-4 구현):
- `eps_r_refraction_qc`(float \| None): §7 2a QC의 εr($h_\text{est}$에서 구한 값). 플래그 판정용이며 깊이에는 쓰지 않는다.
- `eps_r_refraction_qc_range`(tuple[float, float] \| None): §7 2a 구간 $[\varepsilon_\text{lo},\varepsilon_\text{hi}]$.
- `sigma_zero_ns`, `sigma_apex_ns`(float \| None, ns): §9 계산에 실제로 쓴 $\sigma_z$와 $\sigma_a$다.

`alt_depths` 채우는 규칙(v1.4):
- (a) 메타 영점 이름(`t0_sample_est` 등): 주 v로, 주 v가 없으면 메타 v로 $d$를 구한다. 시각은 꼭짓점 $t_a$에서 메타 영점을 뺀 값이고, 라벨은 §4-3이다.
- (b) `"meta_fixed"`: 주 결과가 `meta_fixed`가 아닐 때(순서 3a, 곡률 실패)만 넣는다. 메타 v, 주 영점, **§6-4 꼭짓점**으로 구한다(곡률 해의 $t_0$가 아니다; v를 고정하면 꼭짓점은 §6-4로 정한다). 라벨은 `zero_event_label`이고, `meta`·None이면 `unclassified`다.

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
| (2) dewow | **가우스 커널** 이동평균 차감(v1.1): $a\leftarrow a-G_\sigma * a$, $\sigma=1/f_c$, `scipy.ndimage.gaussian_filter1d(axis=0, mode="reflect", truncate=4.0)`. $f_c$ 없으면 $\sigma=2$ ns | 500 MHz → σ 2 ns | 응답 $1-e^{-2\pi^2\sigma^2f^2}$: −3 dB가 $0.25f_c$(밴드패스 하한과 같음), 통과대역 리플 없음, 영위상. v1의 상자형 창($T_w=3/f_c$)도 −3 dB는 $0.25f_c$이지만 통과대역 리플 0.70–1.22배와 **±$T_w/2$ 유령 엽**(포락선 5.5–6.9 %, gprMax M02·M12·M13; 느린 꼬리가 있으면 11.7 %)을 만든다. 유령 엽이 사건 문턱 10 %를 넘으면 공기결합 `late` 규칙이 그것을 지표 반사로 잡아 영점이 약 +3 ns 늦고 깊이가 약 −15 cm 틀어진다(v1.1 review 계산) |
| (3) bandpass | Butterworth `N=4`(`butter(4, [lo, hi], btype="bandpass")`, 즉 대역통과 8차) SOS, `scipy.signal.sosfiltfilt`(영위상), 통과대역 $[0.25 f_c,\ \min(2f_c,\ 0.45/\Delta t)]$ | 500 MHz → 125 MHz–1 GHz | [PA 4] 100–600 MHz, 정의서 0.25–2배. 영위상이라 영점·꼭짓점 시각이 같은 방식으로 바뀐다 |
| (4) 배경 제거 | 이동 **median** 트레이스 차감, 창 $W_\text{bg}$ 트레이스(**홀수로 올림**). 측선 끝에서는 창을 안쪽으로 밀어 폭을 유지한다(첫·마지막 온전한 창의 median). 측선이 $W_\text{bg}$ 이하면 측선 전체 median | $W_\text{bg}=301$ (4TU 6 m) | mean 차감은 꼭짓점을 +0.022 ns 늦게, median은 −0.0007 ns(§12 표 2). 창은 쌍곡선 폭(±1 m ≈ 100 트레이스)보다 충분히 길게. `pipe_band` 갈래는 이 단계 대신 masked median([B] §3-2) |
| (5) gain | AGC(RMS 창 $8/f_c$, $f_c$ 없으면 16 ns) 또는 SEC. **검출 갈래 전용** | 가정 값, cv 쪽 조정 | 정의서 범위 1 |
| (6) clip | |amp| 99 백분위로 클립 후 [0,255] | GPR-SD는 필수 | readme_gprsd |
| (7) letterbox | 비율 유지 패딩, `ns_per_px`, `m_per_px`, 패딩 오프셋을 메타로 보존 | – | [PA] §0 |

- (6)·(7)과 (5)를 묶은 검출 영상 변환은 M3-2 `DetectionImage`에서 구현한다. M2-1 범위는 피팅 갈래 (1)–(4)와 `agc` 함수다.
- `amplitude_processed=True`(GPR-SD, 광저우 MALA): (1)–(4)를 다시 하지 않는다(이미 지표 에코 제거·gain 적용). 검출 갈래의 (5) gain도 기본값으로 생략하고 (6) clip만 한다(이중 gain 방지, cv 쪽에서 바꿀 수 있음). 시간 가변 gain은 포락선 피크를 gain 기울기 방향으로 조금 옮기므로 `flags`에 `amplitude_processed`를 넣는다.
- `dt_ns=None`(스케일 없는 JPG): ns 시간 창을 정할 수 없으므로 (2)는 생략한다. 창 $W_\text{bg}$는 픽셀(열) 단위 그대로다(AIHub 256 px 폭은 301 이하라 측선 전체 median).
  - **(3) 대신 픽셀 단위 밴드패스(v1.3).** (1) 뒤 자료의 트레이스 평균 진폭 스펙트럼 $\overline{|\mathcal F_k[a_j]|}$에서 DC를 뺀 최대 주파수 $f_\text{px}$(cycle/px)를 우세 주기로 잡는다. 통과대역 $[0.25f_\text{px},\ \min(2f_\text{px},\ 0.45)]$ cycle/px로 (3)과 같은 4차 Butterworth `sosfiltfilt`를 쓴다(단계 이름 `bandpass_px`). 영점(px)과 꼭짓점 후보가 같은 자료를 쓰게 하려고 (4) 전에 둔다.
  - 우세 주기 $1/f_\text{px}$가 **2.2–128 px**(v1.7; v1.3은 4–128 px) 밖이면 생략하고 `bandpass_px_skipped` 플래그를 붙인다.
    - 하한 2.2 px = 1/0.45: 상한 차단이 $0.45$ cycle/px에 묶이므로, 그보다 짧은 주기는 통과대역을 만들 수 없다.
    - v1.3의 4 px는 근거 없는 가정이었다. AIHub 실측 우세 주기가 2.44–4.83 px였고, 이 중 4 px 미만 14/40장이 생략됐다.
    - 이 주기는 레이다 반송파로 판정했다: 세로(시간축)에만 있고, 띠 박스 안에서도 같은 4.1–4.6 px이며, 가로에는 없고, JPEG 8×8 블록 선(1/8·1/4 cycle/px) 비가 0.8–1.2로 평탄하다. 근거는 `analysis/processing/aihub_period_px_check.py`(train_pool 10장)다.
  - 근거: (1)·(4)만 하면 SNR 20 dB 해석 합성에서 광대역 잡음 포락선이 $0.3E_\text{max}$를 넘는다. "가장 이른 첫 피크" 규칙(§5-2)은 120 트레이스 중 하나만 넘어도 그것을 고르므로 꼭짓점 후보가 박스 위 끝(정답보다 27–30 px 이른 곳)에 잡혔다(SNR 10–20 dB 전부). 트레이스 방향 3·5개 평균으로는 고쳐지지 않았다. 픽셀 밴드패스를 쓰면 $f_\text{px}$ 0.0187–0.0213(정답 0.0200)이었고, 꼭짓점 후보는 트레이스 75–76, 꼭짓점 차 −1.1~0.0 px였다(review 계산 2026-10-30, scratchpad, CSV 없음).
  - $f_\text{px}$는 물리 축척이 아니므로 ns·m로 바꾸지 않는다.
- `f_c_hz=None`, `dt_ns` 있음: (3)을 생략하고 (2)는 σ 2 ns(500 MHz 가정)를 쓴다. 영점 탐색 창의 $1/f_c$도 2 ns로 둔다(§4-1). 세 기본값(dewow σ 2 ns, AGC 16 ns, 주기 2 ns)은 모두 500 MHz 가정에서 나온 값이다.
- 기록 앞머리 효과: 사건이 기록 시작에서 dewow 커널 반폭 안에 있으면 포락선 피크가 조금 옮겨진다. 4TU 직접파 위치(약 2 ns, Δt 0.098)에서 +0.03 ns, 400 MHz에서 1.5 ns에 놓인 사건은 +0.18 ns(깊이 약 0.9 cm)였다. 보정하지 않는다.
- 공기결합 자료에서 배경 제거는 수평인 지표 반사를 지운다. 그래서 영점은 반드시 (4) **전** 자료로 잡는다([DP] §7-5).

---

## 4. 영점 선택 규칙

### 4-1. 사건 검출 (`detect_zero_events`, 공통)

입력: (3)까지 처리한 자료. 배경 제거 전이다.

1. 평균 트레이스 $\bar a(t)=\frac1N\sum_j a_j(t)$(`pipe_band` 박스 트레이스 제외, [B] §3-1), 포락선 $e(t)=|\mathcal H[\bar a](t)|$. **주 영점은 이 전역 평균 트레이스 하나로 정한다.** 측선이 $W_\text{bg}$보다 길면 $W_\text{bg}$ 트레이스 창마다 같은 사건 검출을 하되 QC에만 쓴다(v1.1):
   - 창마다 **선택된 영점 사건**(§4-2 규칙이 고른 사건)에 가장 가까운 피크를 0.5 ns 안에서 찾는다. 찾은 시각들의 표준편차가 `window_std_ns`다. 창 피크 목록에서 마지막 피크를 무조건 쓰지 않는다(지면결합·단일 사건에서는 영점이 첫 피크다).
   - 일치 피크가 없는 창이 있으면 `zero_window_missing`, 창 사건 수가 전역 사건 수와 다른 창이 있으면 `zero_window_event_count_mismatch`(공기결합에서 단일/두 사건 판정이 측선 안에서 바뀐다는 뜻), `window_std_ns`가 0.2 ns를 넘으면 `zero_window_drift` 플래그를 붙인다(0.2 ns는 [B] §3-1 항목 3의 트레이스별 영점 전환 기준과 같다. 전환 자체는 v2).
2. 탐색 창: 기록 앞 25 % 안에서 $e$가 최대인 시각을 $t_\text{max}$로 둔다. 창은 $[0,\ t_\text{max}+2h_\text{max}/c+0.5/f_c]$이고 $h_\text{max}=0.5$ m(가정)다. $f_c$가 없으면 $1/f_c=2$ ns(500 MHz 가정). `dt_ns=None`이면 ns 창을 만들 수 없으므로 기록 앞 25 % 전체(픽셀)를 창으로 쓰고 결과에 `zero_px_unverified`를 붙인다. AIHub crop은 지표가 영상에 없을 수도 있으므로 이 픽셀 영점은 상대 위치 참고값일 뿐이다.
3. `scipy.signal.find_peaks(e, height=0.1·max, prominence=0.05·max)`를 쓰고 포물선 보간으로 시각을 잡는다.
4. 첫 피크가 $t_\text{dir}$이다. 창 안 마지막 피크를 `late`로 둔다. 피크가 2개 이상이면 $t_\text{surf}=t_\text{late}$다.
5. 공기층 추정(두 사건일 때): $t_\text{surf}-t_\text{dir}=\dfrac{2\sqrt{h^2+s^2/4}-s}{c}$를 $h$로 푼다. $s$를 모르면 $s=0$으로 두고 $h=c\,(t_\text{surf}-t_\text{dir})/2$로 계산한 뒤 `flags`에 남긴다.

### 4-2. 데이터셋별 규칙

| `air_gap` | 조건 | `zero_rule` | $t_z$ | 라벨 | 편향·불확도 |
|---|---|---|---|---|---|
| air | 사건 2개 이상 | `air_surface_late` | $t_\text{surf}$ (늦은 쪽) | surface_reflection | **직접파 간섭 편향**(v1.2): $t_\text{surf}$가 단독 지표 반사 피크보다 (1)–(3) 뒤 0.12–0.29 ns 늦다(해석 Ricker h 0.3 m +0.12, gprMax M20 +0.29) → εr 9에서 깊이 0.6–1.5 cm **얕게**. 단독 지표 반사는 (1)–(3)으로 ≤0.003 ns만 움직이므로(영위상) 처리 자체의 편향이 아니고, 반대 극성 포락선이 서로 밀어내는 효과를 밴드패스가 약 2배로 키운 것이다(M20 필터 없음 +0.18 → 처리 후 +0.29, v1 상자형 +0.23). 크기·부호가 간격·진폭비·파형에 따라 −0.08 ~ +0.29 ns로 바뀌어 보정하지 않는다([`surface_zero_shift.csv`](surface_zero_shift.csv), [`surface_zero_shift_hsweep.csv`](surface_zero_shift_hsweep.csv)). $\sigma_z=0.3$ ns |
| air | 사건 1개 | `air_single_event` | 그 피크 | merged | 합쳐진 피크가 직접파 위치에 놓여 **깊게** 나온다. M12·M13·M15·M18(h=0.1)에서 +1.8~+3.6 cm(§12 표 4). 편향은 $\approx h/\sqrt{\varepsilon_r}$(영점 오차 $\approx 2h/c$)이고, 단일 사건이면 $h\le h_\text{sep}=0.375\,c/f_c$이므로 상한 $h_\text{sep}/\sqrt{\varepsilon_r}$(500 MHz, εr 9 → 7.5 cm)을 **수치로** `bias_notes`에 넣는다(εr 메타 없으면 식만). 분리 한계 근거는 아래 |
| ground | 항상 | `ground_direct_peak` | $t_\text{dir}$ | direct_wave | 직접파 피크는 송신보다 0.53–1.03 ns 늦다(εr 4–16, s=0.1 m 합성) → **얕게** −1.9~−3.9 cm([DP] 표 D, M02 −3.3 cm) |
| ground | 선택: `ground_zero_shift="synthetic_table"` | `ground_direct_shifted` | $t_\text{dir}-\Delta_\text{dw}(\varepsilon_r)$ | direct_wave | $\Delta_\text{dw}$: εr 4 → 0.53, 9 → 0.725, 16 → 0.96 ns(400 MHz; 500 MHz는 0.53/0.727/1.03). $\sqrt{\varepsilon_r}$에 선형 보간한다. **s = 0.1 m 2D 합성에서 얻은 값이라 기본값은 끈다**(장비 s가 다르면 값이 다름). 적용 범위(v1.1): $f_c$ 350–550 MHz(가까운 행), εr 4–16(바깥 외삽 금지), `antenna_offset_m`이 None이거나 $|s-0.1|\le0.02$ m. 하나라도 벗어나거나 $f_c$·`eps_r_meta`가 없으면 보정하지 않고 `ground_direct_peak` + `ground_zero_shift_unavailable` |
| unknown | 스케일 있음·없음 | 지면결합 규칙과 같음 | $t_\text{dir}$ | direct_wave | `air_gap_unknown` 플래그. 사건이 2개 이상이면 `air_gap_unknown_multi_event`도 붙인다(공기결합이면 영점이 늦은 사건이어야 해서 깊이가 깊게 나올 수 있음). 현재 스케일 있는 데이터셋은 모두 air/ground가 정해져 있다(§1-1) |
| – | GPR-SD (`amplitude_processed`) | `meta_sample0` | 0 (가정) | meta | 지표 에코가 제거돼 사건 검출이 불가하다. 깊이에 `zero_unverified` 플래그를 붙인다 |
| – | 광저우 MALA (`amplitude_processed`) | `meta_sample0` | 0 | meta | 처리본(t0=0). 위상 기준 미상. 규칙은 데이터셋 이름이 아니라 `amplitude_processed`로 고른다 |

- **두 사건 분리 한계(v1.1 review 계산, 해석 Ricker, (1)–(3) 처리 후, `find_peaks` 높이 0.1·돌출 0.05)**: 지표 반사가 직접파와 **반대 극성**(공기→토양 반사계수 음수, 진폭비 −2/3)이면 간격 $\ge0.75/f_c$(500 MHz 1.5 ns, 400 MHz 1.85 ns)부터 두 사건으로 갈린다. $s=0$이면 $h_\text{sep}=c\cdot0.75/(2f_c)=0.375\,c/f_c$(500 MHz 0.22 m). **같은 극성**이면 0.4–0.6/f_c에서 갈렸다가 0.65–1.0/f_c(500 MHz 1.3–2.0 ns)에서 다시 합쳐진다. gprMax M20(간격 1.71 ns)은 분리됐고 반대 극성 결과와 맞는다. v1의 "포락선 폭 ≈ $1.2/f_c$, $h\lesssim0.15$ m" 표현은 틀려서 지웠다.

- 광저우 IDS의 `t0_sample_used`(직접파 시작 또는 정점)는 주 영점으로 쓰지 않는다. 직접 검출한 $t_\text{dir}$(포락선 피크)를 쓰고, `t0_sample_used`로 낸 깊이는 `alt_depths["t0_sample_used"]`에만 넣는다(최대 약 1 ns·5 cm 차이, readme_guangzhou).
- first break(5 % 문턱)는 영점으로 쓰지 않는다. M02에서 +7.9 cm, M20에서 +16.9 cm가 나왔다(§12 표 1).

### 4-3. 4TU 메타 영점 두 개의 분류와 보고 (CLAUDE.md 분할 규칙)

1. 메타 영점을 ns로 바꾼다. $t^\text{est}=\texttt{t0\_sample\_est}\cdot\Delta t$, $t^\text{act}=\texttt{t0\_sample\_act\_median}\cdot\Delta t$.
2. 각 값을 **선택된 규칙의 사건 목록**으로 라벨링한다(v1.1). 벗어나면 `unclassified`로 둔다.

   | 규칙 | 라벨 후보 | 허용폭 |
   |---|---|---|
   | `air_surface_late` (사건 2개 이상) | $t_\text{dir}$ → `direct_wave`, $t_\text{surf}$ → `surface_reflection` (가까운 쪽) | 0.5 ns (0.098 ns 자료에서 약 5 샘플) |
   | `air_single_event` | 합쳐진 피크 → `merged` | $\max(0.5\ \text{ns},\ 0.5/f_c)$ (500 MHz 1.0 ns). 메타 t0는 $\arg\max|\bar a|$라서 같은 띠의 ± 엽에 놓이고, 엽은 포락선 피크에서 약 ±0.4/f_c(Ricker 0.78 ns @500 MHz, [FX] 실측 −0.7 ~ +0.6 ns) 떨어져 있다 |
   | `ground_*` | $t_\text{dir}$ → `direct_wave`만 | 0.5 ns. 지면결합에는 따로 떨어진 지표 반사가 없으므로 창 안 늦은 피크를 `surface_reflection`으로 부르지 않는다 |
   | `meta_sample0`, 실패 | 사건이 있으면 위 지면결합 규칙, 없으면 전부 `unclassified` | 0.5 ns |

   단일 사건의 엽 라벨(`band_lobe_pos`/`band_lobe_neg`/`band_envelope`, 허용폭 0.3 ns, 위상차 기록)은 [FX] P2로 v2(M2-9)에서 정한다.
3. 깊이를 세 개 낸다. 주 결과는 §4-2 규칙의 $t_z$ 결과이고(사건 2개면 $t_\text{surf}$, 단일 사건이면 합쳐진 피크 — 4TU는 [FX]에서 대부분 단일 사건), `alt_depths["t0_sample_est"]`와 `alt_depths["t0_sample_act_median"]`에는 각 메타 영점 결과와 라벨을 넣는다.
4. 메타 t0는 "처음 100 샘플 중 평균 진폭 최대"로 잡은 값이다. 포락선 피크와 위상 기준이 다르다. 합성에서 $\arg\max|\bar a|$는 M02에서 포락선보다 0.27 ns 이르게 잡혔고, M20에서는 **직접파**를 잡아 +9.6 cm가 나왔다(§12 표 1). 그래서 메타 영점은 비교용으로만 쓴다.
5. **사건이 둘로 분리된 경우에만**, 라벨이 `direct_wave`인 메타 영점으로 낸 깊이는 체계적으로 깊다. 크기는 εr 9, h 0.3 m에서 약 +8.6 cm다([DP] §3). `merged` 라벨이면 메타 영점과 주 영점의 차는 엽 위상차뿐이다(±0.4/f_c, 깊이 약 ±3.5 cm, [FX]).

---

## 5. 쌍곡선 점 추출 (`extract_picks`)

입력은 배경 제거 후·gain 전 자료(§3 (4)), 박스 `(i0..i1, k0..k1)`, $t_z$다.

1. 박스를 트레이스 방향으로 좌우 10 %씩 넓힌다. 포락선은 $E=|\mathcal H[a]|$를 시간축으로 구한다.
2. **꼭짓점 후보**: 박스 안 최대 포락선을 $E_\text{max}$라 한다. 각 트레이스에서 $k>k_\text{mute}$ 구간의 피크 가운데 $E\ge0.3E_\text{max}$를 넘는 **첫** 피크를 찾는다(v1.3 문구 명확화: 문턱 아래 작은 피크는 건너뛰고 문턱을 넘는 가장 이른 피크; `find_peaks(height=0.3E_max)`의 첫 원소). "첫 피크가 문턱 아래면 그 트레이스는 후보 없음"으로 읽지 않는다. 그렇게 읽으면 사건 앞의 작은 잡음·잔향 피크 하나로 꼭짓점 트레이스가 빠진다. M18 회피 의도(늦게 오는 더 강한 바닥·내부 반사 대신 상단)는 문턱을 넘는 첫 피크로 충분하다(§12 표 4, 1.027 m). 그중 시각이 가장 이른 트레이스를 $j_a$, 그 시각을 $t_a$로 둔다. $k_\text{mute}$는 $t_z+1/f_c$에 해당하는 샘플이다.
   - 근거: 트레이스별 **전역 최대**를 쓰면 물 찬 PVC(M18)에서 바닥·내부 반사를 잡아 깊이가 2.17 m(정답 1.0 m)로 나왔다. 첫 피크 규칙으로 바꾸자 1.027 m가 됐다(§12 표 4).
3. **추적**: $j_a$에서 좌우로 한 트레이스씩 나간다. 탐색 창은 $[t_\text{prev}-0.3,\ t_\text{prev}+2\Delta x/v_\text{min}+0.3]$ ns이고 $v_\text{min}=0.05$ m/ns다. 창 안 포락선 최대를 포물선 보간으로 잡는다. 다음 중 하나면 멈춘다.
   - $E<0.2\,E(t_a)$
   - $|x-x_a|>a_\text{max}$ ($a_\text{max}=1.0$ m, 박스 경계도 적용)
   - 바깥으로 갈 때 $t$가 0.1 ns 넘게 줄어듦(다른 사건으로 넘어간 경우)
4. 점 수가 $n\ge9$이고 양쪽 각각 3개 이상이어야 곡률 피팅을 허용한다. 모자라면 `insufficient_picks`이고, v는 메타 고정만 쓴다.
   - 탐색 범위(v1.3, M2-3 구현 확인):
     - 꼭짓점 후보의 $E_\text{max}$와 첫 피크는 좌우로 넓힌 박스 안에서, 시간 방향은 박스 $k_0..k_1$(뮤트로 위를 자른 뒤) 안에서만 찾는다. 위로 넓히지 않는다.
     - 추적의 시간 창은 박스 아래 끝에 묶이지 않는다. 날개가 박스 밖으로 내려가도 기록 끝까지 따라간다. 트레이스 방향은 넓힌 박스 경계와 $a_\text{max}$ 중 먼저 닿는 곳에서 멈춘다.
   - 뮤트(v1.3):
     - $k_\text{mute}=\lceil(t_z+1/f_c)/\Delta t\rceil$이다. $f_c$가 없으면 $1/f_c=2$ ns로 둔다.
     - 스케일이 없으면 뮤트하지 않는다([B] §3-3과 같다).
     - 영점이 실패해서 $t_z$가 없으면 뮤트 없이 추적하고 `mute_unavailable`를 붙인다. 이때 깊이는 §10대로 내지 않고 꼭짓점 위치만 남는다.
5. `pick_mode`는 `"envelope"`(기본), `"signed_max"`, `"signed_min"` 중 하나다. **영점 검출과 같은 모드를 강제한다**(API에서 하나의 인자로 공유).
6. 꼭짓점이 박스 좌우 경계 3 트레이스 안에 있으면 `apex_at_box_edge` 플래그를 붙인다.

---

## 6. 피팅

### 6-1. 모델

시각은 모두 영점 기준 $\tau_i=t_i-t_z$ (ns)이다.

- 직선 광선, 반경 $R$ 고정(지면결합):
  $$\tau(x)=\frac{2}{v}\left(\sqrt{\left(\tfrac{v t_0}{2}+R\right)^2+(x-x_0)^2}-R\right),\qquad d_\text{top}=\frac{v\,t_0}{2}$$
  $R=0$이면 점산란 식 $\tau=\frac2v\sqrt{d^2+(x-x_0)^2}$다.
- 굴절(공기층) 모델: Tx $(x-s/2,\,h)$ → 관 상단 정반사점 → Rx $(x+s/2,\,h)$의 Fermat 최소 시간에서 지표 반사 기하 시간 $2\sqrt{h^2+s^2/4}/c$를 뺀다. 알고리즘은 `scripts/synthetic/ray_geometry.py`의 `leg_time`(굴절점 이분법)·`reflection_time`(원주 각 탐색)과 같다. **`src/gpr_detection/depth/refraction.py`로 옮겨 구현**하고 `scripts/`를 import하지 않는다. 이분법은 40회면 충분하다(M20에서 εr 9.0 복원). 각 격자는 721점이다.
- v1.4 확인: 구현은 721점 격자 **한 단**만 쓴다(2단 정밀 격자 생략). 정반사점에서 시간이 정류값이라 반 격자(π/1440) 각 오차가 R 0.1 m에서 약 $10^{-6}$ ns만 바꾸므로 승인한다.
- 꼭짓점 시각으로 $d$를 정할 때(§7 2a): Tx·Rx가 관 중심에 대칭이므로 정반사점은 원 꼭대기다($R$이 들어가지 않는다). $d$는 0–10 m에서 이분법 40회로 찾는다.

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
| 거부 | $c\le0$, $v\notin[0.03, c_0]$, 꼭짓점 $a-b^2/(4c)\le0$, 또는 sklearn이 합의 집합을 못 찾음 → $(x_a, \tau_a, v_\text{meta}\ \text{or}\ 0.1)$을 **초기값으로만** 대체하고 §6-3을 계속한다. `ransac_rejected` 플래그(v1.3) | RANSAC은 초기값 단계일 뿐이다. 최종 성패는 §6-3 수렴·경계 판정이 정한다(§10) |

### 6-3. 정밀화

`scipy.optimize.least_squares`, 잔차 $r_i=\tau(x_i;x_0,t_0,v)-\tau_i$.

| 파라미터 | 기본값 |
|---|---|
| `loss` | `"soft_l1"` (`"huber"` 선택 가능) |
| `f_scale` | 0.1 ns (약 1 샘플) |
| `bounds` | $x_0\in[x_\text{min},x_\text{max}]$(픽 위치 범위), $t_0\in[0,t_\text{max}]$, $t_\text{max}=\max_i\tau_i$(v1.3; 꼭짓점은 가장 이른 픽보다 늦을 수 없으므로 느슨한 상한), $v\in[0.03,\ 0.2998]$ m/ns. 초기값은 경계 안쪽($10^{-6}$·범위)으로 자른다 |
| 경계 판정 | 해가 어느 경계에서 범위의 $10^{-4}$ 안이면 "경계에 붙음" → §10 `fit_failed` (v1.3) |
| `x_scale` | `"jac"` |
| 미지수 | $(x_0,t_0,v)$. **$R$은 고정** |
| 공분산 | $\mathrm{Cov}=s^2(J^\top J)^{+}$(유사역행렬). $s^2$와 $J$는 **정상점**($|r_i|<3\,f_\text{scale}$)만으로 계산한다: $s^2=\sum_{\text{in}} r_i^2/(n_\text{in}-3)$. $n_\text{in}\le3$이면 전체 점을 쓴다 → $\sigma_v,\sigma_{t_0}$ (v1.3). 근거: 이상치 20 %(±3–5 ns, U3형) 자료에서 전체 원잔차로 구하면 $\sigma_v/v$가 2.69 %였고 정상점만 쓰면 0.05 %였다(이상치 없을 때 둘 다 0.05 %; review 계산, scratchpad). 원잔차 방식은 이상치 크기를 피팅 불확도로 잘못 옮긴다. §9의 곡률 바닥값 3 %가 이미 계통 편향을 덮는다 |
| 보고 | `rms_ns`, `inlier_frac`(|r| < 3·f_scale 비율) |

### 6-4. v 고정 시 꼭짓점 시각

$v$를 고정하면 깊이에 필요한 것은 $t_0$뿐이다. 공기결합 날개는 직선 광선 식과 맞지 않으므로 날개로 $t_0$를 맞추지 않는다. $|x-x_a|\le0.10$ m 안의 점(5개 이상)에 $\tau=p_0+p_1\xi+p_2\xi^2$를 맞추고, $t_0=p_0-p_1^2/(4p_2)$, $x_0=x_a-p_1/(2p_2)$로 둔다. $p_2\le0$이거나, 점이 5개 미만이거나, 꼭짓점 $\xi_v=-p_1/(2p_2)$가 $|\xi_v|>0.10$ m(맞춘 창 밖, 외삽)이면 꼭짓점 후보 $(x_a,\tau_a)$를 쓴다(v1.3 마지막 조건 추가).

---

## 7. 속도 결정 규칙

| 순서 | 조건 | `v_rule` | v | 곡률 피팅 |
|---|---|---|---|---|
| 1 | `dt_ns` 또는 `dx_m` 없음 | `none` | – | 깊이 금지. `status="no_scale"`, 꼭짓점 픽셀만 |
| 2 | `air_gap="air"` (4TU, 합성 M12·M13·M15·M18·M20) | `meta_fixed` | $c/\sqrt{\varepsilon_{r,\text{meta}}}$ | **직선 광선 곡률 피팅 금지.** 호출 시 `ValueError`. 점 수 검사보다 먼저 거부한다(자료에 따라 결과가 달라지지 않게, v1.3) |
| 2a | 위 + 사건 2개(h 추정 가능) + 점 충분 | `refraction_qc` (부가) | 굴절 모델로 εr 격자 3–30, 간격 0.25 → 최소 RMS 주변을 `minimize_scalar`로 정밀화. $d$는 꼭짓점 시각에 묶는다 | QC 전용이다. 주 깊이는 `meta_fixed` 결과로 낸다. **v1.4 구간 판정**: QC를 $h_\text{est}$와 $h_\text{est}-\Delta h$($\Delta h=0.06$ m, **가정**: 간섭 편향 실측 +0.029 m 해석·+0.049 m M20에 여유를 둔 값)에서 두 번 돌려 $\varepsilon_\text{lo}=\min\{\varepsilon(h_\text{est}-\Delta h),\varepsilon(h_\text{est})\}$, $\varepsilon_\text{hi}=\max\{\cdot\}$를 낸다(v1.4 확인: 정렬해서 내므로 $h$에 대한 단조성을 가정하지 않는다). $\varepsilon_\text{meta}<0.7\,\varepsilon_\text{lo}$ 또는 $\varepsilon_\text{meta}>1.3\,\varepsilon_\text{hi}$일 때만 `meta_eps_suspect`를 붙인다(v1.3까지의 단일값 30 % 규칙을 대체). 이유: εr이 $h$에 매우 민감하다. 정확한 광선 픽에서 $h$ +1 cm마다 εr이 약 +8 % 오른다(점산란체 R 0, 참 h 0.3 m, εr 9: QC h 0.25/0.269/0.289/0.300/0.329/0.349 → εr 6.80/7.50/8.40/9.00/11.08/13.0). 영점 이동 −0.12 ns는 −0.1밖에 바꾸지 않는다. 그래서 $h_\text{est}$의 한쪽 간섭 편향이 단일값 판정을 거짓 의심으로 만든다. 4TU 메타 εr의 주된 오류 방향은 **과소**(공기결합 곡률값, §9)이고, 아래쪽 판정은 구간에서도 유지된다.<br>**판별 범위(v1.4 실측)**: U4 해석 합성(R 0.05인데 QC는 R 0으로 들어감)의 구간은 (6.85, 9.70)이다. 분해하면 정확한 광선 픽·R 일치에서 (7.50, 11.08), R 불일치(관경 모름 → 0)에서 (7.03, 10.04), 나머지 −0.2~−0.3은 검출 영점(+0.12 ns)·잡음 픽이다. 따라서 아래쪽 문턱은 $0.7\times6.85=4.8$이다. 공기결합 직선 광선 곡률로 얻은 메타값은 $v'/v$ 1.5–1.82(h 0.22–0.3 m, 2a가 도는 범위)에서 εr′ = 9/(v'/v)² = 2.7–4.0이므로 이 문턱으로 잡힌다. 관경을 모르면 구간이 약 10 % 아래로 밀리고(R 0.05 m, d 0.6 m), 큰 관에서는 더 밀린다 |
| 3 | `air_gap="ground"`, 점 충분 | `curvature_R_fixed` | §6-3 결과 | $R$ = 기지 관경/2, 없으면 0 |
| 3a | 위 + `eps_r_meta` 있음 | 둘 다 보고 | 주 결과는 곡률. `alt_depths["meta_fixed"]` | |
| 4 | `air_gap="ground"`, 점 부족, `eps_r_meta` 있음 | `meta_fixed` | | |
| 5 | `air_gap="ground"`, 점 부족, 메타 없음 | `none` | – | `insufficient_picks` (가정 v 0.1 m/ns로 깊이를 내지 않는다) |
| 2b | (v1.4) `air_gap="air"`, `eps_r_meta` 없음 | `none` | – | `status="no_velocity"`. 꼭짓점·영점·`t0_ns`는 채운다 |
| 6 | (v1.4) `air_gap="unknown"`, 스케일 있음 | `none` | – | **곡률 피팅을 하지 않는다.** 공기결합일 수 있고, 공기결합 곡률 실패는 잔차로 드러나지 않는다(§12 표 3: RMS 0.044 ns인데 $v'/v$ 1.82). `status="no_velocity"`이고 주 깊이는 없다. 메타 v가 있으면 `alt_depths`를 §2 규칙대로 채운다: (a) 메타 영점 항목(주 v가 없으므로 메타 v)과 (b) `"meta_fixed"`. 둘 다 대체값이고, 영점이 지면결합 규칙(직접파)이라 실제로 공기결합이면 깊게 나올 수 있다(`air_gap_unknown` 플래그로 표시됨). [B] §5-1의 `unknown → none`과 같은 규칙이다. 현재 스케일 있는 데이터셋에는 해당이 없다 |

- 2a 세부(v1.4 확인):
  - $h$ = `ZeroResult.h_est_m`, $s$ = `antenna_offset_m`(모르면 0), $R$ = 기지 관경/2(모르면 0).
  - 꼭짓점은 §6-4의 $(x_0,t_0)$이다.
  - 격자 최소점 양옆 칸 안에서 `minimize_scalar(method="bounded", xatol=1e-3)`로 정밀화한다.
  - $h_\text{est}$의 간섭 편향(§4-2, +0.03~+0.05 m)이 εr 결과를 키운다. 허용폭은 §11-1 U4.

- $R$을 모른 채 $R=0$으로 곡률 피팅하면 $v'/v$가 커지고, $d'$는 상단보다 0.2R–0.5R 깊게 나온다([DP] 표 B). 커지는 정도는 이상 픽([DP] 표 B)에서 1.01–1.14배였고, 처리된 픽(M2-5 R11, 직접파 영점)에서는 최대 1.165배(M01·M06 +16.5 %)였다. `bias_notes`에 `"radius_unknown: v,d over by up to +17 %"`를 넣는다(v1.6, +14 %에서 바꿈).
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
| $\sigma_z$ | `air_surface_late` 0.3 ns(v1.2, 0.2에서 올림), `air_single_event` 0.25 ns(+편향), `ground_direct_*` 0.15 ns(+편향), `meta_sample0` 0.5 ns. 코드 상수 `SIGMA_ZERO_NS`(`preprocess/timezero.py`)가 이 값을 따른다 | §12 표 1·4. `air_surface_late` 0.3 ns는 직접파 간섭 편향 M20 +0.29 ns를 덮는 값([`surface_zero_shift.csv`](surface_zero_shift.csv), §4-2) |
| $\sigma_a$ | $\max(0.05\ \text{ns},\ \text{rms}/\sqrt{n_\text{apex}})$. v1.4 정의: 곡률 경로는 rms = 곡률 피팅 전체 RMS, $n_\text{apex}$ = 피팅 $x_0$ ±0.10 m 안의 픽 수(최소 1). §6-4 경로는 포물선 RMS와 점 수(대체 시 rms 0, n 1 → 바닥값 0.05 ns) | 5 mm 격자 지연 0.05–0.1 ns([DP] §6) |
| 계통(합산하지 않음) | 기준점 상단/중심 $\pm R$, 영점 편향 부호 | `bias_notes` |

---

## 10. 경계 조건·실패 처리

| 상황 | 처리 |
|---|---|
| 박스가 영점 사건과 겹침($k_0\Delta t<t_z+1/f_c$) | 박스 위쪽을 $t_z+1/f_c$로 자르고, 남은 높이가 $2/f_c$보다 작으면 픽 없음 + `box_in_zero_zone` 플래그, `status="fit_failed"` |
| 꼭짓점 후보 없음(박스 안에 $0.3E_\text{max}$ 첫 피크 없음, 박스 전부 0 포함) | 픽 없음 + `no_apex_candidate`, `status="fit_failed"`(꼭짓점이 없어 어떤 v로도 깊이를 못 냄) (v1.3) |
| 스케일 없음, 우세 주기 $1/f_\text{px}$가 2.2–128 px 밖(v1.7) | `bandpass_px`를 생략하고 `bandpass_px_skipped` 플래그를 붙인 뒤 처리를 계속한다(§3, v1.3) |
| 영점 실패로 뮤트 불가 | 뮤트 없이 추적 + `mute_unavailable`(§5). 깊이는 아래 "사건 0개" 행대로 없음 (v1.3) |
| 꼭짓점 2개 이상(인접 관) | 꼭짓점 후보는 가장 이른 것 하나만 쓴다. 다중 관 분리는 v2 |
| 꼭짓점이 박스 경계 | 플래그만 붙이고 계속 |
| RANSAC 거부(§6-2) | v1.3: §6-2 대체 초기값으로 §6-3을 **계속**하고 `ransac_rejected` 플래그. §6-3이 수렴하고 경계에 붙지 않으면 곡률 결과를 쓴다. 실패하면 아래 행. (v1의 "메타 v로 §6-4"는 아래 행에 합쳤다. 메타 v 경로는 M2-4에서 반영됨: 곡률 실패 시 `v_rule="none"` + `alt_depths["meta_fixed"]`(§7, §2 (b))) |
| `least_squares` 미수렴 또는 경계 값에 붙음 | `fit_failed`, 메타 v가 있으면 §6-4 결과를 `alt_depths`에 |
| 공기결합에서 사건 0개 | `zero_failed`, 깊이 없음 |
| int16 포화(±32767) | 포화 샘플이 꼭짓점 ±2 샘플 안에 있으면 `saturated_apex` |
| $\Delta t$가 커서 포락선 피크가 3 샘플 이하 폭 | 포물선 보간은 유지하고 $\sigma_a\ge\Delta t/2$. v1.4 판정식: 주기 $1/f_c$(없으면 2 ns)가 $3\Delta t$ 이하. 근사이며, 포락선 폭을 한 주기로 본다. 이 조건이면 밴드패스 상한 $2f_c$가 나이퀴스트를 넘으므로 실제 자료(4TU 최소 10 샘플/주기)에서는 걸리지 않는다 |
| 지면결합, $s$ 앎, 경사거리 $vt_0/2<s/2$ | $d=0$(§8 식의 제곱근 안이 음수). 곡률 경로에도 §8 식을 쓴다 (v1.4) |
| 스케일 없음 | `no_scale`, `apex_trace`·`apex_sample`만 |

---

## 11. 시험

난수 시드는 모두 0이다. 시간 단위는 ns, 거리 단위는 m다.

### 11-1. 단위 시험 (해석 합성, `tests/depth/test_*.py`)

합성 B-scan 생성기(시험용): 트레이스 $x_j=j\Delta x$, $\Delta x=0.02$, $\Delta t=0.05$ ns, 400 MHz Ricker를 $t_j=t_\text{src}+\tau(x_j)$에 놓는다. 지면결합이면 직접파를 $t_\text{src}+0.7$ ns(진폭 +3)에 둔다. 공기결합이면 직접파를 $t_\text{src}+s/c$(진폭 +3)에, 지표 반사를 $t_\text{src}+2\sqrt{h^2+s^2/4}/c$(진폭 −2, 공기→토양 반사계수 음수라 반대 극성)에 둔다(`tests/synthetic_bscan.py` `make_bscan`). 백색잡음 SNR 20 dB(꼭짓점 피크 대비, 시드 0).

| ID | 내용 | 통과 기준 |
|---|---|---|
| U1 | 점산란 $R=0$, εr 9, d 0.6 m, 지면결합, 영점=송신 | $|v/v_\text{true}-1|\le5\%$, $|d-0.6|\le2$ cm |
| U2 | 원통 $R=0.1$, $R$ 고정 피팅 | 같음. $R=0$으로 피팅하면 $v'/v>1$ (편향 부호 확인) |
| U3 | U1 픽의 20 %를 ±3 ns 이상치로 교체 | U1 기준 유지, `inlier_frac`≥0.75 |
| U4 | 공기결합 **500 MHz**(400 MHz면 간격 1.69 ns가 분리 한계 1.85 ns보다 작아 사건이 합쳐짐, §4-2), h 0.3 m, s 0.1 m, $R$ 0.05 m(굴절 모델로 생성, 메타 `pipe_radius_m` 없음), SNR 20 dB, `air_gap="air"`, `eps_r_meta` 9 (v1.4) | (a) 직선 광선 곡률 호출은 `ValueError`. (b) `meta_fixed` 깊이 ±2 cm. (c) 2a 구간이 9를 포함하고(실측 (6.85, 9.70)) `meta_eps_suspect`가 없음. (d) 같은 자료에 `eps_r_meta` **2.7**(= 9/1.82², M20 공기결합 직선 광선 곡률값 $v'/v$ 1.82를 모사)을 주면 `meta_eps_suspect`가 붙음. 4.5도 붙어야 한다(문턱 4.8 아래, 선택 시험). v1.4 초안의 5는 문턱 위라 폐기한다. 시험 시간을 줄이려고 QC 격자를 εr 4–14로 좁혀도 된다. 이때 구간 양 끝이 격자 경계(4, 14)에 붙지 않았음을 함께 단언한다(붙으면 격자가 좁아서 생긴 값이다). **알고리즘 정확도**는 `test_refraction`(정확한 광선 픽, 참 $h$)에서 εr 9±0.1로 따로 본다. v1.3까지의 "εr 9±0.5"를 $h_\text{est}$ 자료에 적용하던 기준은 폐기한다(간섭 편향이 εr을 +8 %/cm로 옮김, §7 2a) |
| U5 | U1 자료(같은 샘플 배열)를 Δt 1.18배(GPR-SD 가설 B/A 비)로 다시 해석한다. 영점도 1.18배다. 같은 배열을 다른 Δt로 읽으면 파형의 실제 주파수도 1/1.18배가 되므로 $f_c$도 1/1.18배로 준다. 그래야 필터가 샘플 단위로 같다(v1.3) | $d$ 변화 ≤ 1 mm, $v$는 $1/1.18$배(±0.5 %). 실제 GPR-SD는 `amplitude_processed`·$f_c$ 없음이라 필터를 다시 돌리지 않는다. U5는 피팅 식의 축척 불변성만 본다 |
| U6 | `pick_mode` 불일치(영점 envelope, 꼭짓점 signed) | 호출 거부 |
| U7 | 스케일 없음(`dt_ns=None`), U1 자료 **SNR 20 dB 그대로**(v1.3; 잡음 없는 자료로 바꾸지 않는다) | `status="no_scale"`, 깊이 None. 픽은 꼭짓점 후보 1개이고, 꼭짓점 후보 트레이스는 75±1, 샘플은 정답 꼭짓점 ±2 px(§3 `bandpass_px`). `bandpass_px` 없이는 실패하는 것이 정상이다(트레이스 68, −28 px) |
| U8 | 늦은 강한 사건(바닥 반사 진폭 2배)이 있는 원통 | 꼭짓점 후보가 상단 반사(첫 피크)를 잡음 |

전처리·영점 단위 시험(v1.1 추가, `tests/preprocess/`):

| ID | 내용 | 통과 기준 |
|---|---|---|
| Z1 | 500 MHz, Δt 0.098 ns, 단일 Ricker + 느린 양의 꼬리($0.8\,e^{-(t-t_e)/1.5\,\text{ns}}$, $t\ge t_e$; v1.2에서 0.6→0.8), 공기결합 | 두 시험을 **상시** 둔다(사용자 결정 2026-10-30). (a) (1)–(3) 뒤 사건 1개, `air_single_event`. (b) 판별력 대조: 같은 자료에서 (2)만 v1 상자형 이동평균 차감($T_w=3/f_c$, 홀수 샘플, `mode="reflect"`)으로 바꾸고 (1)·(3)은 그대로 두면 사건 2개(약 +1.9 ns 유령 엽)여야 한다. (b)가 깨지면 Z1이 상자형 회귀를 못 잡는다는 뜻이므로 꼬리 진폭을 다시 정한다. v1.2 근거는 셋이다.<br>① [`surface_zero_shift_z1.csv`](surface_zero_shift_z1.csv)는 꼬리 0.6·Δt 0.098 ns·감쇠 1.5 ns 한 조건만 담는다. 이 조건에서는 상자형+밴드패스(`dc+box3+bp (v1)`)도 사건 1개였다. 같은 행의 `envelope_rel_at_pm3ns` 0.121은 $|t-t_e|$ 2.5–3.5 ns 구간 포락선 최댓값이다. 이 값은 사건 앞쪽 엽(2.06 ns)의 것이고, 그 엽은 돌출도 0.018로 돌출 문턱 0.05에 못 미쳐 사건이 되지 않았다.<br>② 사건 뒤쪽 유령 엽(7.55 ns)은 높이 0.100·돌출도 0.079로 높이 문턱 0.1 바로 아래였다. 이 값은 평균 트레이스 포락선(앞 100 샘플)에 `find_peaks(prominence=0)`를 돌려 얻은 review 계산이며 CSV 열이 아니다(scratchpad, CSV 없음).<br>③ 꼬리 0.8에서 가우스 1개·상자형 2개로 갈리는 판별은 시험 Z1(a·b)으로 상시 확인한다. Δt 0.05 ns·감쇠 3 ns 변형(모두 가우스 1·상자형 2)은 review 계산이다(scratchpad, CSV 없음) |
| Z2 | 두 사건 반대 극성(진폭비 −2/3), 간격 1.2 ns / 1.8 ns @500 MHz | 1개 / 2개 |
| Z3 | 지면결합(생성기 기본 400 MHz, 직접파 $t_\text{src}+0.7$ ns, 진폭 +3), 창 안 얕은 수평 반사 $\tau=3.5$ ns(직접파 뒤 2.8 ns = $1.12/f_c$, 진폭 **−1.5**, 반대 극성)가 모든 트레이스에 있고 메타 t0가 그 피크에 있음 | 사건 2개. 메타 라벨 `unclassified`(`surface_reflection` 아님), 직접파 위 메타 t0는 `direct_wave` |
| Z4 | 공기결합 단일 사건, 메타 t0 = 포락선 피크 ± 0.78 ns(500 MHz) | 라벨 `merged` |
| Z5 | 측선 700 트레이스, 지면결합, Z3과 같은 얕은 반사($\tau=3.5$ ns, 진폭 −1.5)가 앞쪽 350 트레이스에만 있음 | `ground_direct_peak`, `window_std_ns`가 직접파 기준으로 < 0.05 ns(마지막 피크를 따라가지 않음), `zero_window_event_count_mismatch` 플래그 |
| Z6 | (v1.2) 공기결합 지표 반사 **단독**(직접파 진폭 0, 지표 반사 −2), 500 MHz, Δt 0.05 ns, h 0.3 m, s 0.1 m, **잡음 없음**(`snr_db=None`; 관 반사는 영점 창 밖이라 둬도 됨) | 사건 1개, (1)–(3) 뒤 포락선 피크와 $t_\text{geom}$의 차가 ±0.01 ns 안(영위상 확인, 현재 0.000 ns). 잡음 없이 판정하는 이유: 시험 대상이 필터의 결정적 성질(영위상)이고, 기준 ±0.01 ns가 잡음 지터(151 트레이스 평균 후 약 0.008 진폭)보다 엄격해서 잡음을 넣으면 필터 회귀와 잡음을 구분할 수 없다 |
| Z7 | (v1.2) 직접파(+3) + 지표 반사(−2), h 0.3 m, s 0.1 m, 500 MHz, Δt 0.05 ns(생성기 기본). 기존 두 사건 시험과 [B] UB14 마스크 시험(`exclude_traces` 사용 시)의 시각 기준을 이것으로 바꾼다 | `air_surface_late`, 사건 2개, $0 < t_\text{surf}-t_\text{geom} \le 0.2$ ns(현재 +0.122, 마스크 사용 +0.120), $0.30\le h_\text{est}\le0.36$ m(현재 0.329) |

Z3·Z5 주석(v1.2, M2-2 구현 대조):
- $\tau$는 생성기 `extra_events` 규약대로 $t_\text{src}$ 기준이다. 직접파와의 간격은 $\tau-0.7$ ns다.
- 400 MHz, 진폭비 −1.5/3(반대 극성)에서 (1)–(3)과 §4-1 문턱을 거치면 간격 1.5·1.8 ns는 사건 1개, 2.0 ns부터 2개였다(review 계산, 2.0–3.2 ns에서 늦은 피크 +0.11 ~ +0.16 ns). §4-2의 반대 극성 분리 한계 $0.75/f_c$ = 1.875 ns와 맞는다.
- 그래서 $\tau$ 1.5·2.3 ns(간격 0.8·1.6 ns)가 합쳐진 것은 물리적으로 맞다. $\tau=3.5$(간격 2.8 ns = $1.12/f_c$)는 한계보다 0.9 ns 여유가 있다.
- 같은 극성(+1.5)이면 간격 2.5 ns까지 합쳐졌으므로 Z3·Z5의 반대 극성 조건을 바꾸지 않는다.
- 간격 2.8 ns는 εr 9에서 약 0.14 m 깊이의 얕은 층(포장 기층 등)에 해당한다.

Z6·Z7 주석(v1.2):
- $t_\text{geom}=t_\text{src}+2\sqrt{h^2+s^2/4}/c$(h 0.3, s 0.1 m → $t_\text{src}$ + 2.029 ns)는 단독 지표 반사의 포락선 피크다. 이 값은 (1)–(3)을 거쳐도 0.003 ns 안에서 변하지 않는다(Z6). 그러니 기대값을 "처리 후 단독 피크"로 다시 정의해도 값은 같다.
- 두 사건 시험에서 생기는 차이는 직접파 해석 신호 꼬리와의 **간섭 편향**이다. 반대 극성 포락선이 서로 밀어내고, 밴드패스가 이를 약 2배로 키운다(해석: 원 신호 +0.055 → DC·가우스 dewow +0.066 → 밴드패스 +0.120 ns).
- 편향은 간격·진폭비·파형에 따라 바뀐다. 해석 Ricker(h 0.25–0.5 m, 진폭비 −1/3~−1)에서 −0.08 ~ +0.17 ns이고 분리 한계 근처에서는 부호가 바뀐다. gprMax M20에서는 +0.29 ns다. 그래서 보정하지 않고 허용 창과 $\sigma_z$(§9, 0.3 ns)로 다룬다. εr 9에서 깊이로 −0.6 ~ −1.5 cm(얕게)다.
- Z7의 한쪽 창은 이 고정 기하에서만 성립한다(부호 회귀 검출용). 다른 기하에 일반화하지 않는다.
- 근거: [`surface_zero_shift.csv`](surface_zero_shift.csv), [`surface_zero_shift_hsweep.csv`](surface_zero_shift_hsweep.csv), [`surface_zero_shift.png`](surface_zero_shift.png).
- 처리 변경 후보(σ 0.5–2/f_c, dewow 없음, 밴드패스 하한 0.125–0.5 f_c, 2차, 영점만 DC 처리, 부호 피크)를 모두 시험했다. 이득은 최대 0.03 ns(≈1.5 mm)였다. 이보다 크게 줄이는 후보는 상자형(유령 엽)과 DC만 처리(Z1 실패)뿐이어서 둘 다 기각했다.

### 11-2. 회귀 시험 (gprMax 합성 20개, json 정답)

정답 필드:
- `pipe_top_depth_m`, `velocity_m_per_ns`, `pipe_radius_m`, `air_gap_m`, `t0_top_ns`
- `time_zero_candidates_ns.{source_peak, direct_wave_peak, surface_reflection_peak}`
- `apex.envelope_peak_ns`, `apex.trace_index`
- `ray_reflection_time_ns` (트레이스별 픽 검증용: 픽 − (source_peak + wavelet_lag) 와 비교)

`signed_peak_polarity`는 극성 판정에 쓰지 않는다(2D 합성, [DP] §8).

**입력 규약 (v1.5, M2-4b 로더)**:
- 자료: `load_synthetic(model_id)`(장비 메타만), 정답: `load_synthetic_truth(model_id) -> SyntheticTruth`. 정답은 `Radargram`에 넣지 않는다(누수 방지).
- 기본 실행은 `include_eps=True`·`include_radius=True`이다. R10은 $R$=json이 필요하므로 `include_radius=True`, R11($R=0$ 곡률)은 `include_radius=False`로 돌린다.
- 영점은 **검출값**(`find_time_zero`)을 쓴다. 정답 영점은 비교에만 쓴다.
- `SyntheticTruth` 필드 용도:

  | 필드 | 쓰는 시험 | 구분 |
  |---|---|---|
  | `pipe_top_depth_m`, `velocity_m_per_ns`, `eps_r`, `pipe_radius_m`, `air_gap_m` | R6–R13 | 필수 |
  | `apex_trace_index`, `apex_envelope_peak_ns` | R4, R5, 박스 공급 | 필수 |
  | `source_peak_ns` | 박스 공급, 픽 검증 | 필수 |
  | `ray_reflection_time_ns`(트레이스 $i$별, json 축) | 박스 공급(아래 끝), 트레이스별 픽 검증 | 필수 |
  | `trace_x_m_first`, `pipe_center_x_m` | $x_0$ 비교 | 필수 |
  | `direct_wave_peak_ns` | R1, R3 | 선택형이지만 R1·R3 대상 모델에서 None이면 시험 **실패**(건너뛰지 않음) |
  | `surface_reflection_peak_ns` | R2 | 같음(M20) |
  | `wavelet_lag_ns` | 박스 공급, 픽 검증 | 선택. None이면 0으로 둔다(0.04 ns 수준이라 박스 여유 $1.5/f_c$에 묻힘) |
  | `pipe_material` | 회귀 실패 진단(`pec`·`pvc_air`·`pvc_water`; 물 찬 관 M08·M14·M18은 하단·내부 반사) | 선택. 없으면 None (사용자 결정 2026-10-30 보완) |
  | `t0_top_ns` | 참고(깊이는 `pipe_top_depth_m`으로 판정) | 필수 |

  20개 json 모두 세 선택 필드가 채워져 있다(2026-10-30 확인).
- **x축 비교 규약**: `Radargram`의 트레이스 위치는 $x=i\,\Delta x$(0부터)이다. json 축은 $x_\text{json}=\texttt{trace\_x\_m\_first}+i\,\Delta x$(합성은 0.4 m부터)이다.
  - 비교는 **트레이스 인덱스**로 하거나, 추정 $x_0$에 `trace_x_m_first`를 더해 `pipe_center_x_m`과 비교한다.
  - `ray_reflection_time_ns[i]`, `apex.trace_index`, `_bg.npy` 열(json x 0.4/1.5/2.6 m = 트레이스 0/55/110)은 모두 인덱스 기준이다.
  - `metadata["trace_x_m_first"]`는 보고용으로만 쓰고 깊이 계산에는 넣지 않는다.
- **박스 공급 규칙(M2-5)**: 검출기 대신 정답으로 박스 `GridBox(i0, i1, k0, k1)`를 만든다.
  - 트레이스: $i_a$=`apex_trace_index`, 반폭 $W=\lceil 0.8\,\text{m}/\Delta x\rceil$(0.02 m → 40 트레이스), $i_0=\max(0,i_a-W)$, $i_1=\min(N-1,i_a+W)$. §5-1이 좌우 10 %를 더 넓히고, 추적은 $a_\text{max}$ 1.0 m까지 간다.
  - 시간 위: $k_0=\max\!\left(0,\ \lfloor(t_\text{apex,env}-1.5/f_c)/\Delta t\rfloor\right)$($t_\text{apex,env}$=`apex_envelope_peak_ns`; 현재 20개 모델은 모두 $t_\text{apex}>1.5/f_c$라 하한에 걸리지 않는다).
  - 시간 아래: $k_1=\lceil(\max_{i\in[i_0,i_1]}(t_\text{src}+\tau_\text{ray}[i]+\ell)+1.5/f_c)/\Delta t\rceil$, 기록 끝에서 자른다($t_\text{src}$=`source_peak_ns`, $\tau_\text{ray}$=`ray_reflection_time_ns`, $\ell$=`wavelet_lag_ns`).
  - 근거: 1.5 주기는 포락선 주엽(약 ±0.6 주기)에 여유를 둔 값이다(**가정**). 0.8 m는 합성 측선 ±1.1 m 안에서 날개가 충분히 들어오는 폭이다.
  - 박스 위쪽이 영점 구역에 걸리면 §10대로 잘린다. 물 찬 관(M08·M14·M18)의 하단·내부 반사가 박스 안에 들어와도 그대로 둔다. 첫 피크 꼭짓점 후보(§5-2)을 시험하는 조건이기 때문이다.
  - 박스는 정답에서 만들지만 꼭짓점 후보는 박스 전체(±40 트레이스)에서 찾으므로 R4(±3 트레이스)는 순환 판정이 아니다.
- `_fs.npy`(자유공간, 공기층 모델 M12·M13·M15·M18·M20만): M2-5에는 필요 없다. R2는 json `surface_reflection_peak`(= bg − fs 포락선 피크, 미리 계산됨)를 쓴다. 영점 분해 같은 분석은 `analysis/processing/`에서 npy를 직접 읽는다.

| ID | 대상 | 검사 | 기준 | 현재 근거값 |
|---|---|---|---|---|
| R1 | h=0 (M01–M11, M14, M16, M17, M19) | $t_\text{dir}$ vs `direct_wave_peak` | ≤0.05 ns | M02 0.0000 ns |
| R2 | M20 (h 0.3) | 사건 2개, $t_\text{surf}$ vs `surface_reflection_peak` | ≤0.35 ns, $|h_\text{est}-0.3|\le0.07$ m (v1.1: (1)–(3) 처리 후 값 기준) | 필터 없음 +0.184 ns·0.330 m. v1 상자형 처리 +0.227 ns·0.337 m, v1.1 가우스 처리 +0.292 ns·0.349 m(review 계산). 분해(v1.2): 지표 반사 단독(bg−fs)은 처리 후 +0.002 ns, 두 사건(bg)·코드 입력 간섭 +0.29 ns. 편향 전부가 직접파 간섭이다([`surface_zero_shift.csv`](surface_zero_shift.csv)). M2-5 실측 $h_\text{est}$ 0.349 m(+0.049) |
| R3 | h 0.1 (M12, M13, M15, M18) | `zero_rule == "air_single_event"`, 단일 피크 vs `direct_wave_peak` | ≤0.05 ns | 4개 모두 단일 사건, 편차 0 |
| R4 | 20개 전부 | 꼭짓점 트레이스 vs `apex.trace_index` | ±3 트레이스 | 측정한 6개(M02·M12·M13·M15·M18·M20) 52–55 |
| R5 | 20개 전부 | (v1.6) $t_a$(꼭짓점 후보 = 꼭짓점 트레이스의 첫 포락선 피크, median 배경) vs **같은 (1)–(3)을 거친 이상 배경(D − `_bg` 가운데 열) 꼭짓점 트레이스의 포락선 피크**. 기준 피크는 그 트레이스의 박스 시간 범위($k_0..k_1$) 안에서 $0.3\times$최대 이상인 **첫 피크**로 잡는다(§5-2 꼭짓점 후보 선정 규칙과 같다. 전역 최대를 쓰면 물 찬 관 M08·M14·M18에서 더 강한 하단 반사로 간다). `_bg`의 측방 차는 최대 1.84e-05라 가운데 열 하나로 빼도 된다(v1.6 구현 확인) | ≤0.05 ns. 필터 이동($t$((1)–(3) 이상 배경) − json `apex.envelope_peak_ns`)은 별도 열에 기록만 한다 | v1: M02 +0.0003, M20 −0.0007(json 기준). v1.6 근거: M15 json 대비 −0.059 ns였으나, 이 중 밴드패스 이동이 −0.055 ns이고(M15 꼭짓점 파형은 json 기하 기대보다 +0.237 ns 늦은 분산·비대칭 파형), 재정의 기준 대비는 −0.006 ns다(`analysis/processing/regression_failures_m2_5.py`) |
| R6 | M20 | `meta_fixed` 깊이 | $|d-d_\text{top}|\le2$ cm (일치) | **−1.49 cm**(v1.6, M2-5 실측; v1.2 예상 −1.42 cm. 차 0.07 cm는 §6-4 꼭짓점과 포락선 피크의 차). v1 근거값 −0.88 cm는 필터 없는 영점(+0.184 ns) 기준 |
| R7 | M12, M13, M15, M18 | `meta_fixed` 깊이, 편향 부호 + | $0<d-d_\text{top}\le5$ cm | +1.8, +3.6, +2.0, +2.8 cm |
| R8 | h=0 전부 | `meta_fixed` + `ground_direct_peak` | $-5\le d-d_\text{top}<0$ cm | [DP] 표 D −1.9~−3.9, M02 −3.26 |
| R9 | h=0 전부 | `meta_fixed` + `ground_direct_shifted`(표 $\Delta_\text{dw}$) | $|d-d_\text{top}|\le2$ cm | M02 송신 영점 +0.36 cm |
| R10 | h=0 전부 | `curvature_R_fixed`($R$=json), 영점 `ground_direct_peak` | $|v/v_\text{true}-1|\le6\%$, $|d-d_\text{top}|\le5$ cm. 5 %·2 cm 안이면 "일치", 아니면 "허용 일치"로 기록 | M02 +3.4 %, −1.6 cm. [DP] 표 F 최대 +5.2 %(M09), −2.4 cm |
| R11 | h=0 전부 | $R=0$ 곡률 피팅 | $v'/v>1$ (편향 부호) | M02 1.098 |
| R12 | M20 | 직선 광선 곡률 피팅 요청 | `ValueError`. 참고로 강제 실행 시 $v'/v$ 1.82, +47.8 cm | §12 표 3 |
| R13a-1 | M20 | (v1.6) `refraction_qc` **알고리즘**: 픽 = json `ray_reflection_time_ns`(트레이스 인덱스 기준, $\tau_i$ = `ray_reflection_time_ns[i]` − `surface_reflection_time(h, s)`; 같은 값인 json `surface_reflection_geom_ns`는 `SyntheticTruth`에 없으므로 식으로 계산), json $h$ 0.3·$s$ 0.1·$R$ 0.1, 기본 격자 3–30. 픽 범위는 json **전체 111 트레이스**(±1.1 m)다. 박스 ±40보다 넓어 더 엄격하고, 정확한 시각이라 박스가 필요 없다(v1.6 구현 확인) | εr 9±0.1, $|d-d_\text{top}|\le1$ cm (실측 9.000, −0.001 cm) | 모델 검증: `reflection_times`와 json 광선 시각의 최대 차 0.0000 ns(M2-5 review 계산) |
| R13a-2 | M20 | (v1.6) `refraction_qc` **자료**: 현재 경로의 픽(median 배경, (1)–(3), 검출 $t_\text{surf}$, §6-4 꼭짓점), json $h$ 0.3·$s$ 0.1·$R$ 0.1, **기본 격자·전체 픽**(축소·솎기 안 함) | $|\varepsilon/9-1|\le20\%$, $|d-d_\text{top}|\le5$ cm. 결과에 "날개 지연 편향"을 기록한다. 기록값은 이 행의 εr 상대오차(실측 −13.2 %)이고, 위치별 픽−광선 차는 회귀에서 다시 계산하지 않는다(근거 칸의 review 값으로 둔다) | εr 7.81(−13.2 %), d +2.88 cm. 원인: gprMax 포락선 픽이 날개에서 상대적으로 일찍 옴(픽 − 광선: 꼭짓점 쪽 +0.018 ns, 0.7–0.9 m −0.093 ns). 배경·필터·영점을 바꿔도 7.6–8.0(이상 배경 원 신호 7.72)이라 처리 문제가 아니다 |
| R13b | M20 | (v1.4) `estimate_depth` 경로($h_\text{est}$): 2a 구간이 9를 포함하고 `meta_eps_suspect` 없음 | 위와 같음 | 구간 [7.41, 10.33](M2-5 실측). $h_\text{est}-h$ = +0.049 m(Δh 0.06 안, 여유 0.011 m). 통과는 $h_\text{est}$ 편향(εr↑)과 날개 지연(εr↓)의 부분 상쇄 결과다 |

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
- (v1.6 주석) M20 굴절 격자 행의 εr 9.00·−0.8 cm는 **$h_\text{est}$ 0.330 m 조건에서 나온 상쇄 결과**다. $h_\text{est}$ +0.03 m 편향은 εr을 약 +24 % 올리고, gprMax 날개 픽 지연은 약 −13 % 내린다. json $h$ 0.3 m에서는 같은 이상 배경 원 신호 픽으로 εr 7.72가 나온다(M2-5 review, `analysis/processing/regression_failures_m2_5.py`). 알고리즘 근거로 쓰지 않는다(§11-2 R13a-1·a-2).
- M20 직선 광선 결과는 [DP] 표 C·F(1.81, +0.49 m)를 재현한다. 직선 광선에서는 RMS가 0.044 ns로 작아 **잔차로는 실패를 알 수 없다**. 그래서 규칙으로 금지한다(§7 순서 2).

### 표 4. 공기층 모델 블라인드 영점 (v = 정답, 꼭짓점 = median 배경, 첫 피크 꼭짓점 후보)

| 모델 | h (m) | f (MHz) | 검출 사건 (ns) | 늦은 사건 − json 지표 반사 (ns) | $h_\text{est}$ (m) | $d$ 오차 (cm) | json 지표 반사 영점일 때 (cm) |
|---|---|---|---|---|---|---|---|
| M12 | 0.1 | 400 | [3.890] | −0.449 | – | +1.8 | −0.4 |
| M13 | 0.1 | 500 | [3.193] | −0.429 | – | +3.6 | +0.3 |
| M15 | 0.1 | 400 | [3.910] | −0.423 | – | +2.0 | +0.4 |
| M18 | 0.1 | 400 | [3.890] | −0.450 | – | +2.8 | +0.5 |
| M20 | 0.3 | 500 | [3.164, 5.060] | +0.184 | 0.330 | −0.9 | +0.04 |

- h = 0.1 m에서는 직접파와 지표 반사의 간격이 약 0.45 ns로 포락선 폭보다 짧다. 그래서 피크가 하나로 합쳐진다(`air_single_event`). 4TU는 두 사건 간격이 약 20 샘플(1.95 ns, h ≈ 0.29 m)이라 M20처럼 분리될 것으로 예상한다([DP] §7-2). 현장 확인은 analyze 다음 단계다.
- 꼭짓점 후보를 전역 최대로 잡으면 M18이 +116.7 cm로 실패했고, 첫 피크 규칙으로 +2.8 cm가 됐다(§5-2).

그림: `docs/processing/zero_reference_check.png`(꼭짓점 트레이스 포락선과 영점 후보 표시, M02·M20).

---

## 13. 구현 배치 제안 (python-expert)

| 모듈 | 내용 |
|---|---|
| `src/gpr_detection/io/radargram.py` | `Radargram` dataclass와 `sample_to_ns`(구현됨). 데이터셋별 로더는 별도 파일 `io/{fourtu,guangzhou,gprsd}.py`(구현됨)와 `io/synthetic.py`(M2-4b 구현됨) |
| `src/gpr_detection/io/synthetic.py` | (v1.5) `load_synthetic(model_id, *, root, include_eps=True, include_radius=True) -> Radargram`(장비 메타만, §1-1), `load_synthetic_background(model_id, *, root) -> ndarray`(`_bg.npy` [n_samples, 3], 트레이스 0/55/110, 분석 전용), `load_synthetic_truth(model_id, *, root) -> SyntheticTruth`(§11-2 필드 표), `list_synthetic_models(root)`(manifest). 기본 경로는 `io/paths.py` `DEFAULT_SYNTHETIC_ROOT`. **x 오프셋 주의**: json x = `trace_x_m_first`(0.4) + i·dx, `Radargram` x = i·dx(§11-2). `_fs.npy` 로더는 두지 않는다(M2-5 불필요) |
| `src/gpr_detection/io/radargram.py`(추가, M2-1) | `SPEED_OF_LIGHT_M_PER_NS`, `velocity_from_eps`, `eps_from_velocity` — depth→io 의존 방향을 지키려고 c 상수를 io에 둔다 |
| `src/gpr_detection/preprocess/filters.py` | `remove_dc`, `dewow`(`dewow_sigma_samples`), `bandpass`(`bandpass_corners_hz`; 통과대역이 비면, 즉 0.45/Δt ≤ 0.25 f_c이면 `ValueError`), `remove_background(median, window)`, `agc`, `prepare_fitting_branch -> FittingBranch`((1)–(4) 적용). v1.3: `estimate_period_px`(우세 주기를 **px로** 반환, $1/f_\text{px}$, 범위 밖이면 None), 스케일 없을 때 `bandpass_px` 단계(§3), `FittingBranch.flags`(`bandpass_px_skipped`). v1.7: `dominant_period_px`(범위 검사 없는 $1/f_\text{px}$, 생략된 입력의 보고용), `FittingBranch.period_px`([B] §3-5 픽셀 기준) |
| `src/gpr_detection/preprocess/signals.py` | 영점·꼭짓점 공통 위상 기준: `PickMode`, `check_pick_mode`, `require_same_pick_mode`(U6), `pick_signal`(envelope / signed_max / signed_min), `parabolic_peak` |
| `src/gpr_detection/preprocess/timezero.py` | `detect_zero_events -> ZeroEvents`(§13-2), `select_zero(radargram, events, config) -> ZeroResult`(v1.1, 튜플 대신 §13-1 객체), `find_time_zero`, `classify_meta_t0`(§13-3), `given_zero`, `direct_wave_delay_ns`, `ZeroConfig` |
| `src/gpr_detection/depth/config.py` | `DepthConfig`(아래 기본값, v1.4 `refraction_h_margin_m=0.06` 추가 — §7 2a Δh, 가정), `zero_config()` |
| `src/gpr_detection/depth/picks.py` | `extract_picks(section, box, zero, *, dt_ns, dx_m, f_c_hz, config) -> HyperbolaPicks`(§13-4), `widen_box`, `find_apex_candidate`(첫 피크 꼭짓점 후보), `mute_sample`, 추적 |
| `src/gpr_detection/depth/hyperbola.py` | `hyperbola_time`(§6-1 식), `ransac_init -> RansacInit`, `fit_hyperbola(R fixed) -> HyperbolaFit`, `apex_vertex -> ApexVertex`, `fit_curvature(picks, *, radius_m, air_gap, v_fallback, config) -> CurvatureFit`, `fixed_velocity_apex(picks, config) -> ApexVertex` (§13-4) |
| `src/gpr_detection/depth/refraction.py` | `leg_time(x_antenna, h, px, pz, v)`, `reflection_times(x_mid_m, *, offset_m, h, x0_m, d_top_m, radius_m, v)`(송신 기준 ns), `surface_reflection_time(h, offset_m)`, `refraction_qc(x_m, tau_ns, *, x0_m, tau_apex_ns, h_m, offset_m, radius_m, config) -> RefractionQC`(`eps_fit`, `v_fit`, `d_top_m`, `rms_ns`, `n_picks`, `h_m`, `offset_m`) (ray_geometry 알고리즘 이식, MIT 호환 자체 코드) |
| `src/gpr_detection/depth/velocity.py` | (v1.4) `VelocityRule`(StrEnum: `none`, `meta_fixed`, `curvature_R_fixed`; 띠 규칙은 M2-6b에서 추가), `VelocityChoice`([B] §8-2 필드와 같음: `v_rule`, `v_m_per_ns`, `sigma_v`, `eps_r_used`, `v_sources`, `alt_meta_v`, `bias_notes`, `flags`; 속성 `has_velocity`), `meta_velocity(radargram)`, `select_hyperbola_velocity(radargram, curvature, config) -> VelocityChoice`(§7) |
| `src/gpr_detection/depth/result.py` | (v1.4) `DepthStatus`(§2 집합 + `no_velocity`), `AltDepth`, `DepthResult`(§2 필드 + `target_class`, `eps_r_refraction_qc`, `eps_r_refraction_qc_range`(lo, hi), `sigma_zero_ns`, `sigma_apex_ns`; 목록형 필드는 tuple) |
| `src/gpr_detection/depth/estimate.py` | `prepare_depth_inputs(radargram, config, *, exclude_traces=None) -> DepthInputs(branch, zero)`(측선당 한 번), `estimate_depth(radargram, box, config, *, inputs=None) -> DepthResult`(§7 결정표, §8 `depth_from_time`, §9 `depth_uncertainty`) |
| `tests/depth/` | §11-1 단위 시험, §11-2 회귀 시험(합성 npy 읽기, `@pytest.mark.synthetic`) |

### 13-1. `ZeroResult` 필드 정본 (v1.1)

`select_zero`·`find_time_zero`가 돌려주고 `estimate_depth`(위 표)와 `estimate_band_depths`([B])가 함께 받는다. [B] §8-2 블록은 이 표를 따른다. 필드 추가는 허용하고, 빼거나 이름을 바꾸려면 두 명세를 함께 고친다.

| 필드 | 형 | 단위 | 설명 |
|---|---|---|---|
| `t_zero_ns` | float \| None | ns | 스케일 없음·실패면 None |
| `k_zero` | float \| None | 샘플 | 스케일 없어도 채움(px). 실패면 None |
| `zero_rule` | str | – | §4-2 규칙 이름, 실패면 `"zero_failed"`, 호출자 지정이면 `"given"` |
| `zero_event_label` | `"surface_reflection" \| "direct_wave" \| "merged" \| "meta" \| None` | – | None은 `zero_failed`일 때만 |
| `pick_mode` | `"envelope" \| "signed_max" \| "signed_min"` | – | 꼭짓점·띠 픽이 같은 값이어야 함 |
| `events_ns` | tuple[float, ...] | ns | 검출된 앞쪽 사건 전부(스케일 없으면 빈 튜플) |
| `h_est_m` | float \| None | m | `air_surface_late`일 때만 |
| `t_zero_per_trace_ns` | ndarray \| None | ns | 선택 사항([B] §3-1 항목 3), v1.1은 None |
| `meta_t0` | Mapping[str, tuple[float, label]] | ns | §4-3, label ∈ {`direct_wave`, `surface_reflection`, `merged`, `unclassified`} |
| `excluded_traces` | int | – | 평균 트레이스에서 뺀 트레이스 수 |
| `flags` | tuple[str, ...] | – | `zero_failed`, `zero_band_overlap`, `zero_unverified`, `amplitude_processed`, `zero_px_unverified`, `air_gap_unknown`, `air_gap_unknown_multi_event`, `ground_zero_shift_unavailable`, `h_est_offset_unknown`, `zero_window_missing`, `zero_window_event_count_mismatch`, `zero_window_drift`, `zero_given`(`given_zero`가 붙임) |
| `sigma_zero_ns` | float | ns | §9 $\sigma_z$ (규칙별) |
| `bias_notes` | tuple[str, ...] | – | 규칙의 부호 있는 계통 편향(`air_single_event`는 수치 상한 포함) |
| `window_std_ns` | float \| None | ns | §4-1 항목 1 창별 QC(선택 사건 추적). v1.2: `ZeroEvents`에서 이 객체로 옮겼다(선택 사건을 알아야 계산되므로) |

### 13-2. `ZeroEvents` 필드 (v1.2, M2-2 구현 대조)

`detect_zero_events`가 돌려주고 `select_zero`가 받는 중간 객체다. §4-1 결과를 담고, 규칙 선택(§4-2) 전 값이다.

| 필드 | 형 | 단위 | 설명 |
|---|---|---|---|
| `events_k` | tuple[float, ...] | 샘플 | 전역 평균 트레이스의 사건 피크(포물선 보간), 이른 순 |
| `amplitudes` | tuple[float, ...] | 픽 신호 | 각 피크의 `pick_signal` 값 |
| `dt_ns` | float \| None | ns | 스케일 없으면 None |
| `f_c_hz` | float \| None | Hz | 창 폭·메타 허용폭에 쓴 값(None이면 500 MHz 가정) |
| `pick_mode` | `PickMode` | – | 영점 위상 기준(꼭짓점과 같아야 함) |
| `window_end_k` | int | 샘플 | §4-1 항목 2 탐색 창 끝 |
| `h_est_m` | float \| None | m | 사건 2개 이상·스케일 있을 때 §4-1 항목 5 |
| `window_events_k` | tuple[tuple[float, ...], ...] | 샘플 | 트레이스 창별 피크(QC). 측선 ≤ $W_\text{bg}$ 또는 스케일 없음이면 빈 튜플 |
| `excluded_traces` | int | – | 평균에서 뺀 트레이스 수 |
| `flags` | tuple[str, ...] | – | `zero_band_overlap`, `h_est_offset_unknown` |
| 속성 `n_events` | int | – | `len(events_k)` |
| 속성 `events_ns` | tuple[float, ...] | ns | `events_k`·`dt_ns`, 스케일 없으면 빈 튜플 |

### 13-3. 영점 함수 시그니처·설정 (v1.2 확인)

- `classify_meta_t0(candidates_ns, zero_rule, events_ns, f_c_hz=None, tolerance_ns=0.5) -> dict[str, tuple[float, label]]`
  - 라벨 후보는 §4-3 표의 **선택된 규칙** 기준이다.
  - `air_surface_late`: 첫 사건 → `direct_wave`, 마지막 사건 → `surface_reflection`.
  - `air_single_event`: `merged`, 허용폭 $\max(\text{tol},0.5/f_c)$.
  - 그 밖(`ground_*`, `meta_sample0`, `given`): 첫 사건 → `direct_wave`만. 사건이 없으면(`zero_failed`) 전부 `unclassified`.
- `ZeroConfig` 기본값: `h_max_m=0.5`, `early_fraction=0.25`, `height_rel=0.1`, `prominence_rel=0.05`, `window_traces=301`($W_\text{bg}$), `window_match_ns=0.5`(§4-1 항목 1 일치 거리), `window_drift_ns=0.2`(`zero_window_drift` 문턱), `meta_tolerance_ns=0.5`, `ground_zero_shift="none"`.
- $\Delta_\text{dw}$ 적용 범위 상수(§4-2):
  - `DELAY_TABLE_F_C_RANGE_HZ=(350e6, 550e6)`
  - `DELAY_TABLE_EPS_RANGE=(4, 16)`
  - `DELAY_TABLE_OFFSET_M=0.1`
  - `DELAY_TABLE_OFFSET_TOLERANCE_M=0.02`
  - `antenna_offset_m=None`은 적용 가능으로 본다.
- `SIGMA_ZERO_NS`: §9 값과 같다(`air_surface_late` 0.3).
- `SEPARATION_PERIODS=0.75`: §4-2 반대 극성 분리 한계, `air_single_event` 편향 상한에 쓴다.

### 13-4. 픽·피팅 객체 필드 정본 (v1.3, M2-3 구현 대조)

`depth/picks.py`·`depth/hyperbola.py`가 돌려주고 M2-4 `estimate_depth`가 받는다. 필드 추가는 허용하고, 빼거나 이름을 바꾸려면 이 표를 함께 고친다. 위치는 L0 격자(원본 트레이스·샘플)다.

**`ApexCandidate`**(§5-2): `trace: int`(L0), `k: float`(L0 샘플, 포물선 보간), `amplitude: float`(픽 신호 값).

**`PickScale`**: `dt_ns`, `dx_m`, `t_zero_ns`(각 float \| None). `require()`는 하나라도 None이면 `ValueError`다.

**`HyperbolaPicks`**(`extract_picks` 반환)

| 필드 | 형 | 단위 | 설명 |
|---|---|---|---|
| `box` | `GridBox` | L0 | 실제로 찾은 박스(좌우 확장, 위쪽은 뮤트로 자름). `box_in_zero_zone`이면 확장만 한 박스 |
| `trace_idx` | ndarray[int] | 트레이스 | 픽 트레이스, 오름차순 |
| `k_pick` | ndarray[float] | 샘플 | 픽 위치(L0) |
| `amplitude` | ndarray[float] | 픽 신호 | 픽 값 |
| `apex` | `ApexCandidate \| None` | – | 꼭짓점 후보. `no_apex_candidate`·`box_in_zero_zone`이면 None |
| `n_left`, `n_right` | int | – | 꼭짓점 좌·우 픽 수(꼭짓점 제외) |
| `scale` | `PickScale` | – | 변환에 쓴 축척·영점 |
| `flags` | tuple[str, ...] | – | `apex_at_box_edge`, `box_in_zero_zone`, `no_apex_candidate`, `mute_unavailable` |
| 속성 | `n_picks`, `x_m`(=trace·dx), `tau_ns`(=k·dt−t_z), `apex_x_m`, `apex_tau_ns`, `is_enough_for_curvature(config)` | m, ns | 축척이 없으면 `x_m`·`tau_ns`·`apex_*`가 `ValueError` |

**`RansacInit`**(§6-2): `x0_m`, `t0_ns`, `v_m_per_ns`(float), `inlier_mask: ndarray[bool]`(RANSAC을 안 돌렸으면 전부 False), `accepted: bool`(False면 대체 초기값).

**`HyperbolaFit`**(§6-3)

| 필드 | 형 | 단위 | 설명 |
|---|---|---|---|
| `x0_m`, `t0_ns`, `v_m_per_ns` | float | m, ns, m/ns | 해. $t_0$는 영점 기준 |
| `sigma_x0_m`, `sigma_t0_ns`, `sigma_v` | float | m, ns, m/ns | §6-3 공분산(정상점, 유사역행렬). 점 ≤ 3이면 NaN |
| `radius_m` | float | m | 고정 반경 |
| `rms_ns` | float | ns | 전체 원잔차 RMS |
| `inlier_frac` | float | – | $|r|<3f_\text{scale}$ 비율 |
| `n_picks` | int | – | |
| `converged`, `at_bound` | bool | – | `least_squares.success`, 경계 판정(§6-3) |
| 속성 | `is_usable`(= converged and not at_bound), `depth_top_m`(= $v t_0/2$, $s$ 보정 전) | m | §8의 $s$ 보정은 M2-4 |

**`FitStatus`**(StrEnum, 값은 `DepthResult.status`와 공유): `ok`, `insufficient_picks`, `fit_failed`.

**`CurvatureFit`**(`fit_curvature` 반환): `status: FitStatus`, `init: RansacInit | None`·`fit: HyperbolaFit | None`(시도하지 않았으면 None), `flags`(`ransac_rejected`). 속성 `is_ok`.
- status 규칙(v1.3, §10):
  - `picks.apex is None`(`box_in_zero_zone`·`no_apex_candidate`)이면 `fit_failed`이고 `flags`에 `picks.flags`를 그대로 담는다.
  - 꼭짓점은 있으나 §5-4 점 수가 모자라면 `insufficient_picks`다. 이때는 §7 순서 4로 `meta_fixed` + §6-4를 쓸 수 있다.
  - 수렴 실패·경계에 붙으면 `fit_failed`, 그 밖은 `ok`다.
- 두 실패를 구분하는 이유: `insufficient_picks`는 메타 v로 깊이를 낼 수 있고, 꼭짓점이 없으면 낼 수 없다.

**`ApexVertex`**(§6-4, `apex_vertex`·`fixed_velocity_apex` 반환): `x0_m`, `t0_ns`(대체 시 꼭짓점 후보 값), `n_points: int`(대체 시 1), `rms_ns`(대체 시 0), `used_fallback: bool`.

설정 기본값은 `DepthConfig` dataclass 하나에 모은다: `a_max_m=1.0`, `candidate_rel=0.3`, `track_stop_rel=0.2`, `v_min=0.05`, `ransac_dt_ns=0.3`, `ransac_trials=200`, `f_scale_ns=0.1`, `apex_halfwidth_m=0.10`, `bg_window=301`, `h_max_m=0.5`, `ground_zero_shift="none"`, `sigma_eps_rel=0.15`, `seed=0`.
- v1.3에서 확인하고 추가한 값(본문 상수와 같음):
  - `pick_mode="envelope"`(§0-4)
  - `box_expand_frac=0.1`(§5-1), `track_back_ns=0.3`·`track_reverse_ns=0.1`(§5-3), `min_picks=9`·`min_side_picks=3`(§5-4), `apex_edge_traces=3`(§5-6)
  - `apex_min_points=5`(§6-4), `loss="soft_l1"`(§6-3)
  - `curvature_sigma_floor=0.03`(§9)
  - `refraction_eps_min=3.0`·`refraction_eps_max=30.0`·`refraction_eps_step=0.25`·`meta_eps_suspect_rel=0.3`(§7 2a)
- v1.4 추가: `refraction_h_margin_m=0.06`(§7 2a의 Δh, **가정**, §14).
- `single_event_h_max_m`은 두지 않는다. v1의 "h ≲ 0.15 m"는 v1.1에서 틀린 표현으로 지웠고, `air_single_event` 상한은 `SEPARATION_PERIODS`(0.375 c/f_c, §4-2)로 계산한다.

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
| 2a 구간 판정의 Δh 0.06 m(`refraction_h_margin_m`) | 가정(v1.4): $h_\text{est}$ 간섭 편향 실측 +0.029 m(해석 Ricker)·+0.049 m(M20)에 여유를 둔 값 | (v1.6) M2-5 실측: M20 $h_\text{est}-h$ = +0.049 m로 Δh 안이지만 **여유 0.011 m로 작다**. 공기층 h·주파수가 다른 합성이 생기면 다시 잰다. R13(M20)과 M2-5 합성 세트에서 $h_\text{est}-h$ 분포를 재서 확인 |
| $f_c$ 없을 때 500 MHz 가정(dewow σ 2 ns, AGC 16 ns, 영점 창 주기 2 ns) | 가정(v1.1) | 해당 입력(GPR-SD·MALA)은 `amplitude_processed`라 현재 쓰이지 않음 |
| 가우스 dewow가 4TU 단일 띠 판정([FX] 41/41, 상자형으로 측정)을 바꾸지 않는지 | 미확인. 가우스는 유령 엽을 줄이는 쪽이라 단일 판정이 유지될 것으로 봄 | M2-7·M2-8 4TU 실행에서 사건 수 재확인 |
| 4TU 현장 11의 앞쪽 0 뮤트(약 48 샘플)와 DC 제거 | 전 구간 평균을 빼면 뮤트 구간이 상수로 바뀌고 뮤트 끝에 계단이 생긴다(영향 미측정) | M2-7에서 뮤트 끝 사건 여부 확인, 필요하면 DC를 뮤트 밖 샘플로만 계산 |
