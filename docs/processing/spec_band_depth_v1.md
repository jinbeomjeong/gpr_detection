# 명세 v1 — `pipe_band` 깊이 추정 (띠 상단 경계 · v 출처 · 출력 · 시험)

- 작성: gpr-processing-expert, 2026-10-30 (spec 모드, 체크리스트 M2-6)
- 받는 쪽: python-expert (M2-6b 구현·시험), add-architect (ADR-002 결정 8 근거 갱신, ADR-004 결정 7 문구 변경 — §2), electromagnetic-simulation-expert (3D 합성 요청 §9-3)
- 기준 문서: `docs/processing/spec_depth_v1.md`(이하 [S]). 이 명세는 [S]의 전처리·영점·불확도 규칙을 **그대로 물려받고**, 다른 점만 적는다.
- 1차 근거: `docs/research/depth_physics.md`([DP]), `docs/processing/fourtu_explore.md`([FX]), `docs/datasets/aihub_71327.md` §4, `docs/datasets/audit_pipe_band_shape_v1.md`([AU]), `docs/datasets/readme_4tu.md` §5, ADR-002, ADR-004
- **사용자 결정(2026-10-30, 확인 필요 질문 Q1–Q4 모두 권고안)**: Q1 가정 v(데이터셋 기본 εr) **허용 안 함**(`dataset_default_eps=None`, 출처 없으면 `no_velocity`·τ만) · Q2 측방 이격 편향은 **보정 없이** `bias_notes`·`lateral_bias_max_m` 상한만 · Q3 조각 박스 묶기는 라벨 v2 여부와 무관하게 **깊이 단계에서 항상** · Q4 경사 띠 대표 깊이는 **박스 중심**(양끝·경사율은 `depth_profile`·`grade`). Q5(4TU 끝단간 평가에서 `pipe_band`는 헝가리안 매칭에 넣지 않고 따로 보고)는 권고안대로 두고 M3 때 다시 본다.
- 이 명세를 위한 탐색 실행: `conda run -n gpr-dev python -I analysis/processing/pipe_band_check.py` (단일 프로세스, 약 30 s, 시드 0) → `docs/processing/pipe_band_check_synthetic.csv`, `pipe_band_check_geometry.csv`, `pipe_band_check_oblique.csv`, `pipe_band_check.png` (§10에 요약)

---

## 0. 결론 요약

1. 측선 바로 아래에서 측선과 평행하게 놓인 관은 **모든 트레이스에서 쌍곡선 꼭짓점 트레이스와 같은 응답**을 낸다. 그래서 띠 상단 시각은 꼭짓점 시각과 물리적으로 같은 양이다. 깊이 식·영점 규칙·위상 기준(포락선 피크)·불확도는 [S]의 꼭짓점 규칙(§6-4, §8, §9)을 그대로 쓴다. gprMax 20개 모델로 만든 의사 띠에서 띠 규칙 깊이와 꼭짓점 규칙 깊이의 차는 **−0.35 ~ +0.14 cm**였다(§10 표 A1).
2. **배경 제거가 가장 큰 함정이다.** [S] §3 (4)의 이동 median 배경 제거는 창의 절반보다 긴 띠를 지운다. 띠가 측선 전체를 덮으면 진폭이 **0.7–1.6 %**만 남고, 깊이가 **−7.6 ~ −29.6 cm** 틀어진다(표 A2). 띠 갈래는 박스 밖 트레이스로만 배경을 구하는 **masked median**을 쓴다. 박스 밖 트레이스가 모자라면 배경 제거를 하지 않는다.
3. 띠 상단은 **첫 반사의 포락선 피크**로 잡는다. 포락선 상승점(50 %)을 쓰고 영점은 피크로 두면 피크·피크보다 −2.8 ~ −8.4 cm 얕아진다(표 A3). 둘 다 상승점으로 잡아도 일관된 이득이 없다(지면결합 −0.5 ~ +1.3 cm로 엇갈림, M20은 −3.5 cm 대 −1.2 cm로 나빠짐).
4. 띠에는 곡률이 없어서 v를 스스로 구할 수 없다. v 출처 우선순위는 다음과 같다(§5). 같은 B-scan의 `pipe_hyperbola` 곡률 v(지면결합만) → 메타 εr → 같은 조사 단위의 쌍곡선 v 중앙값 → 없음. 공기결합(4TU)은 메타 εr 고정만 쓴다.
5. 단일 측선에서 보정할 수 없는 **한쪽 방향 편향** 두 가지가 있다.
   - 측선이 관 축에서 옆으로 $y$만큼 떨어지면 깊이가 깊게 나온다. $d_\text{top}=0.6$ m에서 $y=0.2$ m면 +2.8~+3.1 cm, $y=0.3$ m면 +6.2~+6.8 cm다(표 B1).
   - 이웃 쌍곡선이 비스듬히 교차한 것이면 곡률 v가 $1/\sin\theta$배로 커진다. 70°에서 +6.4 %, 60°에서 +15.5 %다(표 B2).
   두 편향 모두 `bias_notes`로만 보고하고 보정하지 않는다.
6. 띠는 교차각 $\theta\to0$인 쌍곡선의 극한이다. 박스 반폭 1 m, 500 MHz 기준으로 곡선이 띠처럼 보이는 상한각은 $d$ 0.3–1.0 m에서 **10–19°**다(표 B3). 그래서 띠에도 약한 곡률이 남을 수 있다. 곡률이 유의하면 꼭짓점(교차점) 시각을 쓴다. 꼭짓점 깊이 오차는 ≤0.2 cm이고, 중앙값을 쓰면 최대 +4.8 cm다(표 C1).
7. 스케일 없는 입력(AIHub·모로코)에는 **픽셀 좌표만** 낸다(`status="no_scale"`). 시간 단위도 내지 않는다(dt가 없다).
8. 검증 자료 현황:
   - 4TU에는 측선과 평행한 관의 정답이 사실상 없다. 측선은 관로와 직각으로 설계됐고, 정답은 그와 직각인 시굴 단면이다.
   - 2D gprMax 합성으로는 평행 관을 만들 수 없다. 3D 합성을 요청한다(§9-3).
   - M2-6b 판정은 의사 띠 회귀, 해석 합성 단위 시험, (가능해지면) 3D 합성으로 한다.

### 기호

| 기호 | 뜻 | 단위 |
|---|---|---|
| $t_{b,j}$ | 트레이스 $j$의 띠 상단(첫 반사) 포락선 피크 시각 | ns |
| $t_z$ ($t_{z,j}$) | 깊이 영점(전역 또는 트레이스별), [S] §4 | ns |
| $\tau_j=t_{b,j}-t_{z(,j)}$ | 영점 기준 띠 상단 왕복시간 | ns |
| $\tau_\text{ref}$ | 대표 왕복시간(§3-5 모델별 정의) | ns |
| $x_j=j\,\Delta x$ | 트레이스 위치(측선 안) | m |
| $v$, $\varepsilon_r$ | 토양 속도, 비유전율 $v=c/\sqrt{\varepsilon_r}$ | m/ns, – |
| $c$ | 진공 광속 0.299792458 | m/ns |
| $R$ | 관 반경 | m |
| $d_\text{top}$, $d_c$ | 관 상단 깊이, 관 중심 깊이 $d_c=d_\text{top}+R$ | m |
| $y$ | 측선과 관 축 사이 수평 거리(측방 이격) | m |
| $\theta$ | 관 축과 측선 사이 수평 교차각(90° = 직교, 0° = 평행) | ° |
| $\alpha$, $g$ | 측선 방향 관 경사각, 경사율 $g=\tan\alpha$ | °, m/m |
| $f_c$ | 안테나 중심주파수. 시간 식의 $1/f_c$는 ns로 쓴다(즉 $f_c$를 GHz로 넣는다. 500 MHz → $1/f_c$ = 2 ns) | Hz |
| $\Delta t$, $\Delta x$ | 샘플 간격, 트레이스 간격 | ns, m |
| $h$ | 공기층(안테나 높이) | m |

---

## 1. 물리 모델

### 1-1. 왜 띠가 되는가

관 축이 깊이 $d_c$(중심)의 수평선이라고 하자. 직선 광선·영 오프셋 근사에서 측선 위 $x$의 안테나와 관 표면 사이 최단 거리는 다음과 같다.

$$r(x)=\begin{cases}\sqrt{d_c^2+\big((x-x_0)\sin\theta\big)^2}-R, & \theta>0\ (\text{측선 위 }x_0\text{에서 관 축 위를 지남})\\[4pt]\sqrt{d_c^2+y^2}-R, & \theta=0\ (\text{측방 이격 }y)\end{cases}$$

왕복시간은 $\tau(x)=2r(x)/v$다.

| 경우 | $\tau(x)$ | B-scan 모양 |
|---|---|---|
| $\theta=90°$ (직교) | $\frac2v\big(\sqrt{d_c^2+(x-x_0)^2}-R\big)$ | 쌍곡선([S] §6-1) |
| $0<\theta<90°$ | 위 식에서 $(x-x_0)\to(x-x_0)\sin\theta$ | 넓은 쌍곡선. 겉보기 속도 $v/\sin\theta$ |
| $\theta=0$, $y=0$ | $\frac2v\,d_\text{top}$ (상수) | **수평 띠** |
| $\theta=0$, $y\ne0$ | $\frac2v\big(\sqrt{d_c^2+y^2}-R\big)$ (상수, 더 늦음) | 수평 띠, 깊게 보임 |
| $\theta=0$, 경사 $\alpha$ | $\frac2v\,d_\text{top}(x)\cos\alpha$ | 기울어진 띠, 기울기 $d\tau/dx=2\sin\alpha/v$ |

- $\theta=0,\,y=0$일 때 모든 트레이스의 기하는 직교 관의 **꼭짓점 트레이스**와 같다. 그래서 띠 상단 시각은 꼭짓점 시각 $t_a$와 같은 양이다. 2D 합성의 꼭짓점 트레이스를 이어 붙인 **의사 띠**(§10-1)로 이 명세의 영점·피크 규칙을 시험할 수 있다. 다만 2D 선원과 3D 점원의 파형 차이는 검증되지 않았다(**가정**, §9-3 P09로 확인).
- Tx–Rx 오프셋 $s$가 관 축과 나란해도 정반사점은 두 안테나의 가운데에 놓인다. 경로는 $2\sqrt{d_\text{top}^2+s^2/4}$로 직교 경우와 같다. 영향은 $s^2/(8d)$ ≈ 2–4 mm로 무시한다([S] §8).
- 편파: 안테나 쌍극자가 측선과 직교하면(흔한 배치) 평행 관에서는 전기장이 관 축과 직교해 세관·케이블의 반사가 약해진다. 이는 **진폭**(검출 가능성) 문제이고 시각에는 영향이 없다고 둔다(**가정**, P08로 확인).

### 1-2. 깊이 식

$$d_\text{top}=\frac{v\,\tau_\text{ref}}{2},\qquad \tau_\text{ref}=t_{b,\text{ref}}-t_z,\qquad d_c=d_\text{top}+R\ (R\text{ 기지일 때})$$

- **지면결합**: [S] §4-2 `ground_direct_peak` 영점이면 [DP] 표 D와 같은 편향(−1.9~−3.9 cm, 얕음)이 그대로 남는다. 의사 띠 실측은 −1.74 ~ −4.04 cm다(§10 표 A1).
- **공기결합**: [S] §4-2를 그대로 따른다(`air_surface_late`, 4TU는 [FX] 결론대로 대부분 `air_single_event`). 띠 상단은 수직 입사이므로 굴절이 없다. 곡률 피팅 금지 조항([S] §7 순서 2)은 띠와 상관없다. 띠에는 곡률 피팅이 없기 때문이다. 의사 띠에서 h 0.1 m는 +1.81~+3.71 cm, M20(h 0.3 m)은 −1.24 cm였다. [S] R6·R7과 같은 부호·크기다.
- **경사**: 기울어진 띠의 영 오프셋 시각은 $d_\text{top}(x)\cos\alpha$를 준다. 경사율 10 %에서 $\cos\alpha=0.995$이고 깊이 1 m당 −0.5 cm다(표 B4). **보정하지 않는다.** 경사율 20 %를 넘으면 `steep_band` 플래그를 붙인다(−1.9 cm/m).

### 1-3. 편향 요인과 처리

| 요인 | 크기 (근거) | 부호 | 처리 |
|---|---|---|---|
| 측방 이격 $y$ | $\Delta d=\sqrt{d_c^2+y^2}-d_c\approx y^2/(2d_c)$. $d$ 0.6 m에서 y 0.1/0.2/0.3 m → +0.7/+2.8–3.1/+6.2–6.8 cm (표 B1) | + (깊게) | 단일 측선에서 관측 불가. `bias_notes`에 상한만 적는다(§6-3). 보정 없음 |
| 이웃 쌍곡선 교차각 $\theta$ (v 출처 1) | $v'/v=d'/d=1/\sin\theta$. 80°/70°/60° → +1.5/+6.4/+15.5 % (표 B2) | + | `bias_notes`에 적는다. 보정 없음 |
| 관경 미지 (v 출처 1) | 이웃 쌍곡선을 $R=0$으로 피팅하면 v가 1.01–1.14배 ([S] §7, [DP] 표 B) | + | 이웃 쌍곡선 결과의 `radius_used_m`이 0이면 그대로 적는다 |
| 영점 위상 | [S] §4-2 표 그대로 | 데이터별 | [S] 규칙과 `bias_notes` 그대로 |
| 상단 피크 기준 | 첫 반사 포락선 피크. 상승점 혼용은 피크·피크 대비 −2.8~−8.4 cm (표 A3) | − | 포락선 피크로 고정. API가 혼용을 거부 |
| 관 재질(내부 다중반사) | 공기 찬 PVC는 상단·하단 반사가 $4R/c$ (R 0.1 m에서 1.33 ns) 간격으로 겹쳐 포락선이 옮겨 간다. M07 −1.74 cm 대 PEC M02 −3.21 cm | ± | [S] 꼭짓점과 같은 편향. 따로 보정하지 않는다 |
| 물 찬 관 하단 반사 | $4R\sqrt{\varepsilon_\text{in}}/c$ (R 0.1, εr 81 → 12 ns) 뒤의 강한 사건 | – | 첫 피크 규칙으로 배제([S] §5-2와 같은 근거) |
| 관 경사 $\alpha$ | $\cos\alpha$, 10 %에서 −0.5 %(표 B4) | − | 보정 없음, 경사 보고 |
| 배경 제거 | median이 띠를 지움(표 A2) | − (실패) | masked median(§3-2) |

---

## 2. 입력 계약

- `Radargram`([S] §1-1, `io/radargram.py`)과 `GridBox`(L0, 닫힌 구간 `i0..i1`, `k0..k1`)만 받는다. 깊이 모듈은 `Detection`·검출 라이브러리를 모른다(ADR-004 결정 7, README §2-2 규칙).
- 같은 radargram의 `pipe_band` 박스 목록(L0)과, 같은 radargram에서 먼저 계산한 `pipe_hyperbola` 결과로 만든 **속도 풀**(`VelocityEstimate` 목록, §5)을 함께 받는다.
- 영점은 pipeline이 radargram마다 한 번 계산해 넘긴다. 이때 **모든 `pipe_band` 박스의 트레이스를 평균 트레이스에서 뺀다**(§3-1).
- **ADR 반영 필요**(add-architect). 두 ADR은 상황이 다르다.
  - **ADR-002 결정 8**: 이미 "`pipe_band` 깊이는 M2-6 명세로 따로 정한다"로 열려 있다. 문구를 바꿀 필요는 없다. 이 명세(`spec_band_depth_v1.md`)를 근거 문서로 연결하도록 **갱신**만 한다.
  - **ADR-004 결정 7**: "`pipeline`이 `pipe_hyperbola`만 `GridBox`로 바꿔 넘긴다"로 되어 있다. **문구 변경**이 필요하다. `pipe_band` 박스도 `depth.band`로 넘기고, 처리 순서를 "영점(띠 트레이스 제외) → `pipe_hyperbola` → 속도 풀 → `pipe_band`"로 적어야 한다.

---

## 3. 알고리즘

### 3-1. 영점 (공통 영점과 같음, 띠 보호만 추가)

1. [S] §3 (1)–(3)(DC → dewow → 영위상 밴드패스)까지 처리한 자료로 [S] §4-1 사건 검출을 한다. **gain은 쓰지 않는다.**
2. **추가 규칙**: 평균 트레이스를 만들 때 `pipe_band` 박스(좌우 여유 $m$ 트레이스 포함)에 속한 트레이스는 뺀다. 측선 전체가 띠 박스면 빼지 않고 `zero_band_overlap` 플래그를 붙인다.
   - 이유: 수평 띠는 평균 트레이스에서 상쇄되지 않는다. 띠가 영점 탐색 창 $[0,\ t_\text{max}+2h_\text{max}/c+0.5/f_c]$ 안에 있으면 공기결합 규칙(`late` = 창 안 마지막 피크)이 띠를 지표 반사로 잡는다. 4TU에서 εr 9, $h_\text{max}$ 0.5 m면 창 끝이 영점 뒤 약 4.3 ns이고, $d\lesssim0.21$ m인 띠가 여기에 걸린다.
   - 이 규칙은 `pipe_hyperbola` 깊이에도 이롭다. `preprocess/timezero.py`와 `detect_zero_events`는 아직 없다(2026-10-30 현재 `preprocess/`에는 `__init__.py`·`to_image.py`뿐이다). 이 모듈은 **신규로 M2-2·M2-3과 함께 구현**하고, 처음부터 `exclude_traces` 인자를 포함한다.
3. 트레이스별 영점 $t_{z,j}$(이동 정합, [FX] P3)는 선택 사항이다. 기본값은 전역 $t_z$이고, 측선 영점 이동 std가 0.2 ns를 넘으면 트레이스별 영점을 쓴다([FX] 011.14 계단 근거). 기본값을 바꾸는 일은 M2-9 spec v2에서 [S]와 함께 정한다.

### 3-2. 띠 갈래 배경 제거 (masked median)

입력은 [S] §3 (3)까지 처리한 자료 $a$(gain 없음)다.

$$b(t)=\operatorname*{median}_{j\in\mathcal O}a_j(t),\qquad \mathcal O=\{j: |j-i_c|\le W_\text{bg}/2\}\setminus\bigcup_{\text{band boxes}}[i_0-m,\ i_1+m]$$

| 파라미터 | 기본값 | 근거 |
|---|---|---|
| $W_\text{bg}$ | 301 트레이스(띠 중심 $i_c$ 기준). 측선이 더 짧으면 측선 전체 | [S] §3 (4)와 같음 |
| 여유 $m$ | $\max(5,\ \lceil 0.1\ \text{m}/\Delta x\rceil)$ 트레이스. 스케일 없으면 5 | 박스 가장자리 바깥에 띠 꼬리가 남는 것을 피함 (**가정**) |
| 최소 바깥 트레이스 수 | 30 | 30개 미만이면 배경 제거를 생략하고(`bg_none`) 영점 사건은 뮤트로만 피한다 (**가정**) |
| 제외 대상 | 같은 radargram의 **모든** `pipe_band` 박스 | 다른 띠가 같은 시각에 있으면 median에 섞인다 |

- 의사 띠 실측(표 A1·A2): masked median과 배경 제거 생략은 꼭짓점 규칙 깊이와 −0.35 ~ +0.14 cm 차였다. 측선 전체를 median으로 빼면 띠가 사라졌다(진폭 0.7–1.6 %).
- 띠가 창의 40 %만 덮으면 전체 median도 띠를 지우지 않는다(진폭 0.967–0.999). 그래도 −0.9 cm(M07 공기 찬 PVC)까지 어긋났고, masked는 +0.11 cm였다. 그래서 masked를 기본값으로 둔다.
- `amplitude_processed=True`(GPR-SD, 광저우 MALA): 상류에서 배경 제거가 이미 됐을 수 있다. 다시 하지 않고 `band_bg_removed_upstream` 플래그를 붙인다. 긴 띠는 이미 지워졌을 수 있다. 이 경우 검출 자체가 없을 가능성이 크고, 깊이 신뢰도는 낮춘다(§6-2).

### 3-3. 띠 상단 시각 추출 (`extract_band_picks`)

1. **탐색 창**(샘플): $[k_\text{lo},k_\text{hi}]=[\max(k_0-p_\uparrow,\ k_\text{mute}),\ \min(k_1+p_\downarrow,\ N_s-1)]$
   - $p_\uparrow=\lceil 1/(f_c\Delta t)\rceil$ (한 주기), $p_\downarrow=\lceil 0.5/(f_c\Delta t)\rceil$
   - $f_c$를 모르면 $p_\uparrow=\max(2,\lceil0.25\,(k_1-k_0+1)\rceil)$, $p_\downarrow=\max(1,\lceil0.1\,(k_1-k_0+1)\rceil)$ 샘플(픽셀)
   - $k_\text{mute}$ = $t_z+1/f_c$에 해당하는 샘플([S] §5-2). 스케일 없으면 뮤트 없음
   - 이유: 검출·라벨 박스는 gain 영상에서 눈에 보이는 위쪽 경계(파형 시작)를 따라 그려지므로 포락선 피크가 박스 위 끝 근처에 걸릴 수 있다. 위로 한 주기를 넓힌다.
2. 트레이스 범위는 박스 그대로 $[i_0,i_1]$다(쌍곡선처럼 좌우로 넓히지 않는다. 이웃 조각과 섞이는 것을 막기 위해서다).
3. 포락선 $E=|\mathcal H[a-b]|$(시간축). $E_\text{box}$ = **원래 박스** $[k_0,k_1]\times[i_0,i_1]$ 안 $E$의 최댓값.
4. 트레이스마다 창 안 `find_peaks(E, height=0.3·E_box)`의 **첫 피크**를 고르고 포물선 보간으로 $t_{b,j}$를 잡는다. 0.3은 [S] §5-2 씨앗 문턱과 같다.
5. 고른 피크가 창 위 끝 2 샘플 안이면 그 트레이스에 `top_at_window_edge`를 기록한다. 이런 트레이스가 30 %를 넘으면 결과 플래그로 올린다.
6. **다중 사건 표시**(정보용): 같은 트레이스에서 첫 피크 뒤 $E\ge0.5E_\text{box}$인 피크 $t_2$가 있고, $|\tau_2/\tau_1-2|<0.1$이면 `surface_multiple_candidate`를 붙인다. $\tau_2-\tau_1<2/f_c$이면 `inner_reflection_candidate`를 붙인다. 깊이에는 첫 피크만 쓴다.
7. `pick_mode`는 영점과 같아야 한다([S] §5-5). 기본값은 `"envelope"`이고, `"onset"`은 API에 두지 않는다(표 A3).

### 3-4. 조각 박스 묶기 (`group_band_boxes`)

[AU]에서 band 이미지의 30.7 %가 같은 깊이에서 x로 끊긴 조각이었다(간격 ≤8 px 쌍 480개). 검출기도 같은 모양으로 낼 수 있으므로 깊이 단계에서 묶는다.

1. 후보: 두 박스의 시간 겹침이 작은 박스 높이의 50 % 이상이고, 트레이스 간격 $\text{gap}=i_0^{(B)}-i_1^{(A)}-1\le G$일 때 후보로 본다.
   - $G$ = 스케일 없음: 8 트레이스(=px, [AU] 권고 1의 보수값)
   - $G$ = 스케일 있음: $\lceil0.2\ \text{m}/\Delta x\rceil$ (4TU 10 트레이스) (**가정**)
2. 확인: 각 조각에서 §3-3 픽을 낸다. 맞닿는 쪽 끝 5 트레이스의 픽 중앙값 차가 $|\Delta\tau|\le\max(0.5/f_c,\ 3\Delta t)$(스케일 없으면 3 px)이면 같은 띠로 묶는다.
3. 묶음은 이행적으로 반복한다. 결과는 합집합 박스 하나와 구성 박스 인덱스 `member_indices`다. 간격 트레이스도 픽을 시도하되 coverage 분모에는 넣지 않는다.
4. 묶지 않은 조각은 각각 결과를 낸다. 같은 관의 조각이 따로 남을 수 있으므로 `fragment_of_group` 정보(이웃 조각 인덱스, $\Delta\tau$)를 남긴다.

### 3-5. 트레이스 방향 집계와 모델 선택 (`fit_band_profile`)

픽 $(x_j,\tau_j)$ (스케일 없으면 $(i_j,k_j)$ 픽셀)에 세 모델을 `scipy.optimize.least_squares(loss="soft_l1", f_scale=0.1 ns)`로 맞춘다. 스케일 없으면 f_scale은 1 px이다. $\xi=x-x_c$, $x_c$는 픽 범위 중심이다.

| 모델 | 식 | 채택 조건 | $\tau_\text{ref}$, $x_\text{ref}$ |
|---|---|---|---|
| `curved` | $\tau=p_0+p_1\xi+p_2\xi^2$ | $p_2>0$, $p_2/\sigma_{p_2}\ge3$, 꼭짓점 $\xi_v=-p_1/(2p_2)$가 픽 범위 안(양끝 5 트레이스 제외) | 꼭짓점 $p_0-p_1^2/(4p_2)$, $x_c+\xi_v$ |
| `sloped` | $\tau=q_0+q_1\xi$ | `curved` 아님, $|q_1|\,L\ge0.2$ ns이고 $|q_1|/\sigma_{q_1}\ge3$ ($L$ = 픽 범위 길이) | 박스 중심 $q_0$, $x_c$ (끝점 값도 함께) |
| `flat` | $\tau=$ 상수 | 위 둘 다 아님 | $\operatorname{median}_j\tau_j$, 픽 $x$ 중앙값 |

- 공분산은 [S] §6-3과 같다. $\mathrm{Cov}=s^2(J^\top J)^{-1}$.
- **`curved` 해석**: 작은 교차각 $\theta$에서 $p_2\approx2\sin^2\theta/(v^2\,\tau_c)$이고 $\tau_c=\tau_\text{ref}+2R/v$다. v를 알면
  $$\theta_\text{est}=\arcsin\!\Big(v\sqrt{p_2\,\tau_c/2}\Big)$$
  로 교차각을 낸다($R$ 모르면 0). 해석 합성(표 C1)에서 반폭 1 m 창, θ 5–30°, 픽 잡음 0.05 ns일 때 $|\theta_\text{est}-\theta|\le1.8°$였다(CSV 최대 1.74°, θ 30°; 시험 UB4 기준은 ≤2°). 꼭짓점 깊이 오차는 ≤0.2 cm, 중앙값 깊이 오차는 최대 +4.75 cm(θ 30°)였다.
  - $\theta_\text{est}>30°$이면 `band_oblique_large`를 붙인다. 쌍곡선 피팅 대상(검출 클래스 혼동)일 수 있다는 뜻이다. v는 여전히 외부 출처만 쓴다.
- **꼭짓점이 범위 밖**(최소 시각이 박스 끝): 교차점이 박스 밖에 있다. `sloped`로 처리하고 `crossing_outside_box`를 붙인다. 이른 쪽 끝 깊이가 깊이의 상한 쪽 추정이다.
- $p_2<0$이고 유의하면(가운데가 늦은 볼록) 관 기하로 설명되지 않는다(지층·지표 굴곡). `band_concave` 플래그를 붙이고 `flat`/`sloped`로 처리한다.
- `sloped`이면 경사율 $g=\frac{v}{2}q_1$ (m/m)과 `depth_profile`(박스 양끝과 중심의 $(x, d)$)을 낸다. 영점이 트레이스별이 아니면 겉보기 경사에 영점 이동(지표 단차)이 섞일 수 있다. 공기결합에서 영점 이동 std > 0.2 ns이면 `slope_may_include_zero_drift`를 붙인다.
- **잔차 일관성**: 채택 모델 잔차의 $1.4826\,\mathrm{MAD}$ = $\sigma_\text{res}$. $\sigma_\text{res}>0.25/f_c$(500 MHz 0.5 ns, 스케일 없으면 2 px)이면 서로 다른 사건이 섞였다고 보고 `inconsistent_picks` 실패로 처리한다.

### 3-6. 가장자리에서 잘린 띠

- 박스가 측선 끝(트레이스 0 또는 $N-1$)에 닿으면 `touches_profile_edge_left/right` 플래그를 붙인다. 깊이에는 영향이 없다. 띠 길이·교차점 판단에만 영향이 있다.
- AIHub crop 경계 접촉(32.9 %, [AU])은 학습 crop의 성질이다. 추론 입력(M3-2 타일)에서는 타일 경계가 생긴다. 타일 병합이 정해지기 전(M3-2)에는 L0 박스가 측선 끝에 닿을 때만 플래그를 붙인다.

### 3-7. 영점 구역과 겹친 띠

박스 아래 끝 $k_1$이 $k_\text{mute}$보다 이르면 띠가 영점 사건(직접파·지표 반사)과 분리되지 않는다. 이때는 `band_in_zero_zone` 실패다. 박스 일부만 뮤트 위에 있으면 창을 뮤트에서 자른다. 남은 높이가 $1/f_c$보다 작으면 같은 실패로 처리한다. AIHub 라벨 일부가 지표 근처 클러터 안에 있다는 관찰(aihub_71327.md §4)이 이 경우다.

---

## 4. 깊이 계산

| 경우 | 식 |
|---|---|
| 스케일 있음, v 있음 | $d_\text{top}=v\,\tau_\text{ref}/2$. 지면결합에서 $s$를 알면 $\sqrt{(v\tau_\text{ref}/2)^2-(s/2)^2}$ ([S] §8) |
| `sloped` | 위 식을 $x_c$와 양끝에 적용해 `depth_profile` |
| 반경 앎 | $d_c=d_\text{top}+R$ |
| v 없음 | $\tau_\text{ref}$(ns)만 내고 `status="no_velocity"` |
| 스케일 없음 | `ref_trace`·`ref_sample`(px, 실수)만. `status="no_scale"`. 영점 픽셀 `k_zero_px`는 검출되면 참고값으로 내고 `zero_px_unverified`를 붙인다(JPG는 gain 적용 영상이라 피크 시각이 gain 기울기로 옮겨 간다) |

---

## 5. 속도(v) 출처 규칙 (`select_band_velocity`)

### 5-1. 결정표

| 순서 | 조건 | `v_rule` | v | $\sigma_v/v$ |
|---|---|---|---|---|
| 0 | `dt_ns` 또는 `dx_m` 없음 | `none` | – | – (`no_scale`) |
| 1 | `air_gap="air"` (4TU, 합성 h>0) | `meta_fixed` | $c/\sqrt{\varepsilon_{r,\text{meta}}}$ | [S] §9: 0.075 (= ½·0.15). [FX] P5가 반영되면 `eps_meta_equals_header_default`일 때 0.125. 이웃 쌍곡선 v는 **쓰지 않는다**(공기결합 곡률 금지, [S] §7 순서 2). 메타 없으면 `no_velocity` |
| 2 | `air_gap="ground"`, 같은 radargram 풀에 적격 쌍곡선 ≥1 | `neighbor_hyperbola` | 적격 v의 가중 중앙값(가중 $1/\sigma_v^2$) | $\sqrt{(\sigma_\text{stat}/v)^2+(\mathrm{IQR}/1.349/v)^2+0.03^2+\delta_\tau^2}$ |
| 2a | 위 + `eps_r_meta` 있음 | 주 결과는 2. `alt_depths["meta_fixed"]` | | |
| 3 | `air_gap="ground"`, 적격 쌍곡선 없음, `eps_r_meta` 있음 | `meta_fixed` | | 0.075 |
| 4 | `air_gap="ground"`, 위 둘 다 없음, 같은 조사 단위 풀 적격 ≥3 | `survey_hyperbola_median` | 조사 단위 적격 v의 중앙값 | $\max(0.10,\ \mathrm{IQR}/1.349/v)$ (**가정**) |
| 5 | 위 모두 없음 | `none` | – | `no_velocity`. $\tau_\text{ref}$(ns)만 냄 |
| (선택) | `BandConfig.dataset_default_eps` 지정 | `dataset_default` | 지정값 | 0.15 (= ½·0.30), `v_assumed` 플래그. **기본값 None(끔)** — [S] §7 순서 5 "가정 v로 깊이를 내지 않는다"와 맞춘다. Q1 결정(2026-10-30): 끔 |
| `air_gap="unknown"` | | `none` | | 스케일이 있어도 공기층 판단 불가 → `no_velocity` (현재 해당 데이터 없음) |

- **조사 단위**(순서 4): 광저우 = 같은 존(`zone`), GPR-SD = 같은 도로(3측선), 합성 = 해당 없음. 4TU는 공기결합이라 순서 1에서 끝난다. 조사 단위 키는 로더 `metadata`에 두고, pipeline이 같은 키의 radargram 결과를 모아 풀을 만든다.
- **dt 불변성**: GPR-SD dt 가설 A/B에서 이웃 쌍곡선 v는 $1/k$배, $\tau$는 $k$배가 되므로 띠 깊이는 변하지 않는다([S] §7 U5와 같은 논리). 순서 1·3(메타 v)에서는 이 불변성이 없다.

### 5-2. 이웃 쌍곡선 적격 조건

`VelocityEstimate` 하나는 [S] `DepthResult` 하나에서 만든다.

| 조건 | 기준 | 근거 |
|---|---|---|
| 같은 radargram | `line_id` 같음 | 측선 간 속도 차를 피한다 |
| 규칙 | `v_rule == "curvature_R_fixed"`, `status == "ok"` | [S] §7 순서 3 |
| 점 | `n_picks` ≥ 9, 양쪽 ≥ 3, `inlier_frac` ≥ 0.75, `rms_ns` ≤ 0.3 | [S] §5-4, U3, [FX] 표 5 필터 |
| 경계 | `apex_at_box_edge` 없음 | [S] §5-6 |
| 범위 | $v\in[0.05,\ 0.20]$ m/ns (εr 2.2–36) | 물리 범위(**가정**) |
| 깊이 근접 | $|\tau_\text{hyp}-\tau_\text{band}|\le0.5\,\tau_\text{band}$인 것이 있으면 그것만 쓴다. 없으면 전부 쓰고 $\delta_\tau=0.05$ | 층상 매질에서 곡률 v는 꼭짓점 깊이까지의 RMS 속도다. 깊이가 다르면 대표성이 떨어진다(**가정** 크기) |

- 적격 쌍곡선의 `radius_used_m`이 0이면 `bias_notes`에 `"neighbor_v_radius_unknown: v up to +14 %"`를 넣는다. 교차각은 항상 미상이므로 `"neighbor_v_oblique_unknown: v,d over by 1/sin(theta)"`를 넣는다.
- 쓴 쌍곡선 인덱스와 각 v는 `v_sources`에 남긴다.

---

## 6. 출력 계약

### 6-1. 공통 결과와 띠 기하

[S] §2 `DepthResult`는 아직 구현 전(M2-3·M2-4)이다. 그래서 공통 필드를 하나로 두고 클래스별 하위 구조를 붙이도록 제안한다.

| 필드 | 형 | 단위 | 띠에서의 뜻 |
|---|---|---|---|
| `target_class` | `"pipe_hyperbola" \| "pipe_band"` | – | **신규**(공통) |
| `status` | §6-2 표 | – | 공통 집합 확장 |
| `depth_top_m`, `depth_center_m`, `sigma_depth_m` | float \| None | m | §4, §6-3 |
| `bias_notes`, `flags` | list[str] | – | §1-3, §3, §5 |
| `v_m_per_ns`, `sigma_v`, `eps_r_used`, `v_rule` | | m/ns, – | §5 |
| `t_zero_ns`, `zero_rule`, `zero_event_label` | | ns | [S] §4 그대로 |
| `t0_ns` | float | ns | $\tau_\text{ref}$ (꼭짓점 결과의 $t_0$와 같은 뜻) |
| `radius_used_m` | float | m | 기지 R 또는 0. 깊이 식에는 쓰지 않고 $d_c$에만 쓴다 |
| `alt_depths` | dict | m | 4TU 메타 영점 두 개([S] §4-3, CLAUDE.md 규칙), `meta_fixed`(순서 2a) |
| `apex_trace`, `apex_sample`, `x0_m`, `t_apex_ns` | | | 띠에서는 None. 대신 `band` 사용 |
| `band` | `BandGeometry \| None` | | 아래 표 |

`BandGeometry`:

| 필드 | 형 | 단위 | 설명 |
|---|---|---|---|
| `model` | `"flat" \| "sloped" \| "curved"` | – | §3-5 |
| `ref_trace`, `ref_sample` | float | px(L0 인덱스) | 대표점. 스케일 없는 입력도 채운다 |
| `x_ref_m`, `t_band_ns` | float \| None | m, ns | 스케일 있을 때만 |
| `trace_range` | (int, int) | – | 합집합 박스 `i0..i1` |
| `member_indices` | tuple[int, ...] | – | 묶인 입력 박스 번호(§3-4) |
| `coverage` | float | – | 픽 성공 트레이스 / 박스 트레이스 |
| `n_picks`, `sigma_res` | int, float | –, ns(또는 px) | §3-5 |
| `slope_ns_per_m`, `grade` | float \| None | ns/m, m/m | `sloped`만 |
| `theta_est_deg` | float \| None | ° | `curved`이고 v가 있을 때만 |
| `depth_profile` | tuple[(float, float), ...] \| None | (m, m) | `sloped`: 양끝·중심 |
| `bg_mode` | `"masked_median" \| "none" \| "upstream"` | – | §3-2 |
| `lateral_bias_max_m` | float \| None | m | §6-3 |
| `confidence` | float [0,1] | – | §6-4 |

### 6-2. 상태·실패 사유 코드

| `status` | 뜻 | 깊이 | 낼 것 |
|---|---|---|---|
| `ok` | 정상 | 있음 | 전부 |
| `no_scale` | dt 또는 dx 없음 | None | `band.ref_trace`·`ref_sample`·`model`·`coverage`·`confidence`(px) |
| `no_velocity` | v 출처 없음 | None | `t0_ns`, `band.*`(ns 단위) |
| `zero_failed` | 영점 사건 없음([S] §10) | None | `band.ref_*`(px) |
| `band_in_zero_zone` | §3-7 | None | 박스 정보 |
| `insufficient_picks` | coverage < 0.3 또는 `n_picks` < 5 (**가정**) | None | 있는 픽 요약 |
| `inconsistent_picks` | $\sigma_\text{res}>0.25/f_c$ (스케일 없으면 2 px) | None | 픽 요약 |

공통 플래그(일부): `zero_band_overlap`, `bg_none`, `band_bg_removed_upstream`, `top_at_window_edge`, `surface_multiple_candidate`, `inner_reflection_candidate`, `band_oblique_large`, `crossing_outside_box`, `band_concave`, `steep_band`, `slope_may_include_zero_drift`, `touches_profile_edge_left/right`, `fragment_of_group`, `zero_px_unverified`, `v_assumed`, `meta_eps_unverified`, `amplitude_processed`, `dt_hypothesis_A/B`.

### 6-3. 불확도

$$\sigma_d=\sqrt{\Big(d\,\frac{\sigma_v}{v}\Big)^2+\Big(\frac v2\Big)^2\big(\sigma_z^2+\sigma_b^2\big)}$$

| 성분 | 값 | 근거 |
|---|---|---|
| $\sigma_v/v$ | §5-1 표 | |
| $\sigma_z$ | [S] §9 그대로(`air_single_event` 0.25 ns 등) | |
| $\sigma_b$ | `flat`·`sloped`: $\max\big(0.05\ \text{ns},\ \sigma_\text{res}/\sqrt{n_\text{eff}}\big)$, $n_\text{eff}=\max(1,\ n_\text{picks}\Delta x/0.2\ \text{m})$. `curved`: 꼭짓점 시각의 공분산 전파값과 0.05 ns 중 큰 값 | 이웃 트레이스 픽은 발자국(약 0.2 m, **가정**) 안에서 상관된다 |
| 측방 이격(합산하지 않음) | `lateral_bias_max_m` $=\sqrt{d_c^2+y_\text{max}^2}-d_c$. $y_\text{max}=\lambda/4+d_\text{top}/\sqrt{\varepsilon_r+1}$ (안테나 발자국 반경, $\lambda=v/f_c$) | 발자국 식은 Conyers & Goodman(1997)의 경험식이다. **프로젝트 근거 문서에 아직 없음 → 가정**. 500 MHz, εr 9, $d$ 0.6 m에서 $y_\text{max}$ 0.24 m, 상한 약 +4–5 cm(R 0.1 → +4.0, R 0 → +4.6) |
| 계통(합산하지 않음) | 기준점 상단/중심 $\pm R$, 영점 편향 부호, 이웃 v 편향 | `bias_notes` |

### 6-4. 신뢰도 점수

깊이 정확도가 아니라 **띠 상단 픽의 품질**을 나타낸다(검출 점수와 별개).

$$q=q_\text{cov}\cdot q_\text{res}\cdot q_\text{snr},\quad q_\text{cov}=\min\!\Big(1,\frac{\text{coverage}}{0.6}\Big),\quad q_\text{res}=\exp\!\Big[-\Big(\frac{\sigma_\text{res}}{0.125/f_c}\Big)^2\Big],\quad q_\text{snr}=\mathrm{clip}\Big(\frac{\mathrm{SNR}_\text{dB}-6}{14},0,1\Big)$$

- $\mathrm{SNR}_\text{dB}=20\log_{10}\big(\operatorname{median}_jE(t_{b,j})/\hat\sigma_n\big)$. $\hat\sigma_n$은 탐색 창 바로 위 $[k_\text{lo}-2/f_c,\ k_\text{lo})$ 구간(뮤트 이후만)의 배경 제거 진폭의 $1.4826\,\mathrm{MAD}$다. 구간이 없으면 창 아래 같은 길이를 쓴다.
- 스케일 없으면 $0.125/f_c$ 대신 1 px을 쓴다.
- `band_bg_removed_upstream`·`bg_none`이면 $q$에 0.8을 곱한다. 계수와 문턱은 모두 **가정**이고, AIHub test 라벨 박스로 분포만 보고(§9-4) 4TU로는 조정하지 않는다.

---

## 7. 데이터셋별 적용

| 데이터셋 | 스케일 | 영점 | v | 출력 | 비고 |
|---|---|---|---|---|---|
| gprMax 합성(2D) | 있음 | [S] §4-2 | json εr = `meta_fixed`(`synthetic_truth`) | m | 평행 관 없음 → 의사 띠(§9-2) |
| 4TU | 있음 | `air_single_event`(대부분) / `air_surface_late` | `meta_fixed`만 | m + `alt_depths` 두 메타 영점 | 평행 관 정답 거의 없음(§9-5) |
| 광저우 IDS | 있음 | `ground_direct_peak` | 이웃 쌍곡선 → 존 중앙값 → 없음 | m 또는 `no_velocity` | 메타 εr 없음 |
| 광저우 MALA | 있음 | `meta_sample0` | 같음 | 같음 | `amplitude_processed` |
| GPR-SD | 있음(dt 가설 A/B) | `meta_sample0` | 이웃 쌍곡선 → 도로 3측선 중앙값 | m(`zero_unverified`) | 띠가 상류 처리로 지워졌을 수 있음 |
| AIHub 71327, 모로코 | 없음 | (px 참고만) | – | px만, `no_scale` | AIHub `permittivity` 6.0은 자리채움이라 쓰지 않음(aihub_71327.md §2) |

---

## 8. 구현 명세 (python-expert, M2-6b)

### 8-1. 모듈 배치

| 모듈 | 내용 |
|---|---|
| `src/gpr_detection/depth/band.py` | `BandConfig`, `BandPicks`, `BandFit`, `BandGeometry`, `group_band_boxes`, `masked_background`, `extract_band_picks`, `fit_band_profile`, `estimate_band_depth`, `estimate_band_depths` |
| `src/gpr_detection/depth/velocity.py` | `VelocityEstimate`, `VelocityChoice`, `velocity_estimate_from(result)`, `select_band_velocity` |
| `src/gpr_detection/depth/result.py` (또는 `estimate.py` 안) | 공통 `DepthResult`(+`target_class`, `band`) — M2-3/M2-4 구현과 함께 정한다 |
| `src/gpr_detection/preprocess/timezero.py` | **신규**(아직 없음, M2-2·M2-3과 함께 구현). `detect_zero_events(..., exclude_traces: np.ndarray \| None = None)`는 처음부터 이 인자를 포함한다. `ZeroResult` dataclass도 여기에 둔다 |
| `src/gpr_detection/pipeline.py` | 순서: 영점(띠 트레이스 제외) → `pipe_hyperbola` 깊이 → 속도 풀 → `pipe_band` 깊이. `pipe_band` `GridBox`도 depth로 넘긴다(ADR 개정 뒤) |
| `tests/depth/test_band_*.py` | §9-1·§9-2 |

- **import 규칙**: `depth/band.py`·`depth/velocity.py`는 `io.radargram`(`Radargram`, `GridBox`)과 `preprocess`만 import한다. `detect`·`pipeline`·검출 라이브러리는 import하지 않는다(ADR-004 결정 7, README §2-2). 클래스 이름 문자열 `"pipe_band"`은 결과 표시에만 쓴다. 박스 클래스 판별은 pipeline이 한다.
- 포락선·포물선 보간·영점 사건은 `depth/picks.py`·`preprocess/timezero.py`([S] §13 배치안, 둘 다 신규)와 같은 함수를 공유한다. 위상 기준이 같아야 하기 때문이다.

### 8-2. 시그니처

```python
@dataclass(frozen=True)
class BandConfig:
    seed_rel: float = 0.3            # first-peak threshold vs box envelope max
    pad_up_periods: float = 1.0      # window extension above the box, in 1/f_c
    pad_down_periods: float = 0.5
    bg_window: int = 301
    bg_margin_m: float = 0.1         # min 5 traces
    bg_min_outside: int = 30
    merge_gap_px: int = 8            # no-scale inputs
    merge_gap_m: float = 0.2         # scaled inputs
    merge_overlap_min: float = 0.5
    f_scale_ns: float = 0.1
    signif_sigma: float = 3.0
    slope_min_ns: float = 0.2
    res_max_periods: float = 0.25
    coverage_min: float = 0.3
    n_picks_min: int = 5
    oblique_flag_deg: float = 30.0
    footprint_corr_m: float = 0.2
    dataset_default_eps: float | None = None   # off by default (Q1)
    pick_mode: Literal["envelope", "signed_max", "signed_min"] = "envelope"

@dataclass(frozen=True)
class VelocityEstimate:
    v_m_per_ns: float
    sigma_v: float
    tau_ns: float                 # apex two-way time after zero
    line_id: str
    survey_key: str | None
    radius_known: bool
    source_index: int             # index of the hyperbola result
    eligible: bool                # section 5-2 checks already applied

@dataclass(frozen=True)
class ZeroResult:                 # NEW (preprocess/timezero.py); wraps the [S] section 13
    t_zero_ns: float | None       # select_zero tuple (t_z, zero_rule, label) + context.
    k_zero: float | None          # zero sample (float), also filled for no-scale input (px)
    zero_rule: str                # [S] section 4-2 rule name
    zero_event_label: Literal["surface_reflection", "direct_wave", "merged", "meta"]
    pick_mode: Literal["envelope", "signed_max", "signed_min"]
    events_ns: tuple[float, ...]  # all detected early events ([S] 4-1)
    h_est_m: float | None
    t_zero_per_trace_ns: np.ndarray | None   # optional per-trace zero (section 3-1 item 3)
    meta_t0: Mapping[str, tuple[float, str]] # e.g. {"t0_sample_est": (ns, label)} ([S] 4-3)
    excluded_traces: int          # number of traces masked out (band boxes)
    flags: tuple[str, ...]        # zero_failed, zero_band_overlap, ...

@dataclass(frozen=True)
class BandPicks:
    box: GridBox                  # merged (union) box actually searched
    member_indices: tuple[int, ...]
    trace_idx: np.ndarray         # int, traces with a valid pick
    k_pick: np.ndarray            # float sample index of the first envelope peak (L0)
    t_pick_ns: np.ndarray | None  # k_pick * dt_ns; None when no scale
    amp: np.ndarray               # envelope value at the pick
    n_box_traces: int             # coverage denominator (member traces, gaps excluded)
    window: tuple[int, int]       # (k_lo, k_hi) search window, samples
    bg_mode: Literal["masked_median", "none", "upstream"]
    noise_sigma: float | None     # robust noise estimate for q_snr (section 6-4)
    flags: tuple[str, ...]        # top_at_window_edge, surface_multiple_candidate, ...

@dataclass(frozen=True)
class BandFit:
    model: Literal["flat", "sloped", "curved"]
    params: tuple[float, ...]     # flat (p0,), sloped (q0, q1), curved (p0, p1, p2); xi = x - x_c
    cov: np.ndarray | None        # parameter covariance ([S] 6-3)
    x_c: float                    # centre of the pick range (m, or traces when no scale)
    ref_pos: float                # x_ref (m) or trace index when no scale
    tau_ref: float                # ns after zero, or samples after k_zero (px) when no scale
    sigma_tau_ref: float
    sigma_res: float              # 1.4826 * MAD of residuals (ns or px)
    slope: float | None           # ns/m (sloped)
    theta_est_deg: float | None   # curved and v known
    flags: tuple[str, ...]        # crossing_outside_box, band_concave, band_oblique_large, ...

@dataclass(frozen=True)
class VelocityChoice:
    v_rule: str                   # section 5-1: meta_fixed, neighbor_hyperbola, survey_hyperbola_median,
                                  # dataset_default, none
    v_m_per_ns: float | None
    sigma_v: float | None
    eps_r_used: float | None
    v_sources: tuple[int, ...]    # source_index of the hyperbolas used
    alt_meta_v: float | None      # rule 2a: meta v for alt_depths["meta_fixed"]
    bias_notes: tuple[str, ...]   # neighbor_v_radius_unknown, neighbor_v_oblique_unknown, ...
    flags: tuple[str, ...]        # v_assumed, meta_eps_unverified, ...

def group_band_boxes(boxes: Sequence[GridBox], dx_m: float | None,
                     cfg: BandConfig) -> list[tuple[int, ...]]: ...
def masked_background(a: np.ndarray, band_boxes: Sequence[GridBox], center_trace: int,
                      dx_m: float | None, cfg: BandConfig) -> tuple[np.ndarray | None, str]: ...
def extract_band_picks(a_bg: np.ndarray, box: GridBox, k_mute: int | None,
                       dt_ns: float | None, f_c_hz: float | None,
                       cfg: BandConfig) -> BandPicks: ...
def fit_band_profile(picks: BandPicks, dx_m: float | None, v_m_per_ns: float | None,
                     radius_m: float, cfg: BandConfig) -> BandFit: ...
def select_band_velocity(radargram: Radargram, tau_band_ns: float,
                         pool: Sequence[VelocityEstimate],
                         survey_pool: Sequence[VelocityEstimate],
                         cfg: BandConfig) -> VelocityChoice: ...
def estimate_band_depths(radargram: Radargram, band_boxes: Sequence[GridBox],
                         zero: ZeroResult, pool: Sequence[VelocityEstimate] = (),
                         survey_pool: Sequence[VelocityEstimate] = (),
                         cfg: BandConfig = BandConfig()) -> list[DepthResult]: ...
```

- `a`는 [S] §3 (3)까지 처리한 gain 없는 자료다. `ZeroResult`는 [S]에 없는 **신규 dataclass**다. [S] §13의 `select_zero`는 튜플 `(t_z, zero_rule, label)`을 반환한다. `ZeroResult`는 그 튜플을 감싸고 사건 목록·$h$ 추정·메타 영점 라벨·제외 트레이스 수를 더한다. 쌍곡선 깊이(`estimate_depth`)도 같은 객체를 받도록 M2-2·M2-3에서 함께 정한다. `BandPicks`·`BandFit`·`VelocityChoice`·`VelocityEstimate` 필드는 위 블록이 기준이다. 필드를 더하는 것은 허용하고, 빼거나 이름을 바꾸려면 이 명세를 함께 고친다.
- `radargram.has_physical_scale`이 False면 `dt_ns`·`f_c_hz` 없이 px로 진행한다. 결과에 m·ns 필드가 하나라도 채워지면 **시험 실패**다(QA-4).
- `pick_mode`가 영점의 모드와 다르면 `ValueError`를 낸다.
- 난수는 쓰지 않는다. RANSAC이 필요 없다. 이상치는 soft_l1과 MAD로 다룬다.

---

## 9. 검증 계획 (M2-6b)

난수 시드는 모두 0이다. 시간 단위는 ns, 거리 단위는 m다.

### 9-1. 단위 시험 (해석 합성, `tests/depth/test_band_unit.py`)

생성기는 [S] §11-1과 같다($\Delta x$ 0.02, $\Delta t$ 0.05 ns, 400 MHz Ricker, 직접파 $t_\text{src}+0.7$, 백색잡음 SNR 20 dB). 띠는 트레이스마다 $\tau(x)$(§1-1 식)에 놓는다. 별도 언급이 없으면 εr 9, $d_\text{top}$ 0.6, $R$ 0.05, 지면결합, 측선 200 트레이스, `meta_fixed`다.

| ID | 내용 | 통과 기준 |
|---|---|---|
| UB1 | 평탄 띠, 박스 = 트레이스 60–140 | $|d-d_\text{true}|\le2$ cm, `model=="flat"` |
| UB2 | 띠가 측선 100 % 덮음 | UB1과 같음. `bg_mode=="none"`. 비교: 전체 median 배경을 강제로 쓰면 진폭비 < 0.05(제거 확인) |
| UB3 | 경사율 3 % | `model=="sloped"`, $|g-0.03|\le0.005$, 중심 깊이 ±2 cm |
| UB4 | 교차각 θ 10°, 15°, 20° (반폭 1 m) | `model=="curved"`, $|\theta_\text{est}-\theta|\le2°$, 꼭짓점 깊이 ±1 cm |
| UB5 | 측방 이격 y 0.2 m | 깊이 = $\sqrt{d_c^2+y^2}-R$ ±1 cm (보정하지 않음 확인), `lateral_bias_max_m` > 0 |
| UB6 | 같은 τ 조각 두 개(간격 6 트레이스) / τ 2 ns 차 조각 두 개 | 앞은 결과 1개(`member_indices` 길이 2), 뒤는 2개 |
| UB7 | 물 찬 관: 상단 + 12 ns 뒤 2배 진폭 하단 반사 | 상단을 잡음(±2 cm) |
| UB8 | 지표 다중반사 $2\tau$ 사건 추가 | 상단을 잡음, `surface_multiple_candidate` |
| UB9 | 극성 반전 띠 | 깊이 차 ≤ 1 mm (envelope) |
| UB10 | 띠가 직접파와 겹침($d_\text{top}$ 0.05 m) | `status=="band_in_zero_zone"` |
| UB11 | 스케일 없음(`dt_ns=None`) | `status=="no_scale"`, m·ns 필드 전부 None, `ref_sample` 오차 ≤ 1 px |
| UB12 | v 출처: (a) 지면 + 적격 쌍곡선 풀 → `neighbor_hyperbola` (b) `air_gap="air"` + 풀 → `meta_fixed`(풀 무시) (c) 지면, 풀·메타 없음 → `no_velocity`, `t0_ns` 채움 (d) `dataset_default_eps=None`이면 순서 5에서 깊이 없음 | 규칙 이름 일치 |
| UB13 | UB1 시간축 1.18배 + 이웃 쌍곡선 v도 1/1.18배 (GPR-SD A/B) | 깊이 변화 ≤ 1 mm |
| UB14 | 영점 탐색 창 안의 얕은 띠(d 0.15 m, 공기결합 모의, 띠가 측선 60 %) | `exclude_traces` 사용 시 영점이 지표 반사로 남음. 미사용 시와 비교 기록 |
| UB15 | `pick_mode` 불일치 | `ValueError` |

### 9-2. 회귀 시험 (2D gprMax 의사 띠, `tests/depth/test_band_synthetic.py`, `@pytest.mark.synthetic`)

구성은 `analysis/processing/pipe_band_check.py` Part A와 같다.
- 꼭짓점 트레이스(json `apex.trace_index`)를 띠 영역에 반복한다. 나머지 트레이스는 `<id>_bg.npy`의 무관 모델 트레이스로 채운다.
- 측선은 160 트레이스다. 잡음 σ = 0.05·max|관 단독 꼭짓점 트레이스|이고 시드 0이다.
- 박스는 $[t_\text{apex}-0.4/f_c,\ t_\text{apex}+0.8/f_c]$ × 띠 트레이스다. 덮음 비율은 0.4와 1.0 두 가지다.

| ID | 대상 | 검사 | 기준 | 현재 근거값(분석 스크립트) |
|---|---|---|---|---|
| BR1 | 20개 × 덮음 2종 | 띠 깊이 − 꼭짓점 규칙 깊이(같은 영점, json `apex.envelope_peak_ns`) | ≤ 0.5 cm | −0.35 ~ +0.14 cm |
| BR2 | h = 0 (15개) | $d-d_\text{top}$ | $-5\le\Delta<0$ cm ([S] R8과 같음) | −4.04 ~ −1.74 |
| BR3 | h 0.1 (M12·M13·M15·M18) | $d-d_\text{top}$ | $0<\Delta\le5$ cm ([S] R7) | +1.81 ~ +3.71 |
| BR4 | M20 | $d-d_\text{top}$ | ≤ 2 cm ([S] R6) | −1.24 |
| BR5 | 덮음 1.0 | `bg_mode` | `"none"`(바깥 트레이스 < 30) | – |
| BR6 | 20개 | `coverage` | ≥ 0.95 | 1.0 |
| BR7 | 20개 | `zero_rule` | [S] R1–R3과 같은 규칙 | 일치 |

### 9-3. 3D 합성 요청 (electromagnetic-simulation-expert, 위임 필요)

2D TMz 모델에서 관은 항상 측선과 직교한다(평면 밖 무한 원통). 평행·사교 관은 **3D**에서만 만들 수 있다. 공통 조건: εr 9, σ 0.001 S/m, 500 MHz Ricker, 셀 5 mm, Tx–Rx 0.1 m(측선 방향), 지면결합(h 0). 관은 PML까지 관통시켜 관 끝 회절을 없앤다.

| ID | 관 | 기하 | 트레이스 | 목적 |
|---|---|---|---|---|
| P01 | PEC, OD 0.1 m | $\theta=0$, $y=0$, $d_\text{top}$ 0.6 | 5개(x 불변, 0.1 m 간격) | 띠 시각 기준. BR과 같은 판정 |
| P02 | PVC 공기, OD 0.2 | 같음 | 5 | 내부 반사 편향 |
| P03 | PVC 물, OD 0.2 | 같음 | 5 | 하단 반사(첫 피크 규칙) |
| P04 | PEC, OD 0.1 | $\theta=0$, $y$ = 0.1, 0.2, 0.3 m | 각 5 | 측방 이격 편향 표 B1 확인 |
| P05 | PEC, OD 0.1 | $\theta$ = 5°, 10°, 20°, 45°, 90° | 51개(0.04 m) | 곡률 → $\theta_\text{est}$, 이웃 v 편향 $1/\sin\theta$ |
| P06 | PEC, OD 0.1 | $\theta=0$, $y=0$, h = 0.05, 0.3 m | 5 | 공기결합 띠 영점 규칙 |
| P07 | PEC, OD 0.1 | $\theta=0$, 경사율 5 % | 26개(0.04 m) | `sloped`, $g$ 복원 |
| P08 | PEC, OD 0.05 | P01 기하, 쌍극자 방향 측선 평행 / 직교 | 5 + 5 | 편파에 따른 진폭·포락선 피크 이동 |
| P09 | PEC, OD 0.2 | $\theta=90°$ 3D vs M11(2D 같은 조건) | 51 | 2D↔3D 포락선 피크 차 → 의사 띠 회귀의 타당성 |

- 영역은 대략 x 2.0 × y 1.2 × z 1.5 m(공기 0.3 포함), 5 mm 셀(약 2.9×10⁷ 셀)이다. 실행 시간과 GPU 메모리 추정은 EM 에이전트가 낸다. 30분을 넘으면 메인 세션 규칙을 따른다.
- 내보내기는 2D 세트와 같은 json 스키마로 한다. 추가 필드는 `pipe_axis_angle_deg`, `lateral_offset_m`, `grade`, `dipole_orientation`이다.

### 9-4. AIHub (스케일 없음, 품질 점검만)

- 대상: test 분할의 `pipe_band` GT 박스 301개([AU] §3.2)를 원래 JPG(gain 1.0 사본)에 적용한다.
- 지표: `status` 분포, `coverage`·`confidence` 분포, $k_\text{ref}-k_0$(박스 위 끝 대비 픽 위치, px) 분포, 조각 묶기 전후 결과 수.
- 판정은 하지 않는다. 깊이 정답이 없기 때문이다. 목적은 픽 실패 유형(`band_in_zero_zone` 비율 등)을 찾는 것이다. 모든 결과에 m·ns 필드가 없는지 확인한다(QA-4).
- 박스 좌표·원본 이미지는 로컬 전용이다(AIHub 카드 공개 규칙).

### 9-5. 4TU

- 측선은 관로와 직각으로 설계됐고(readme_4tu §5-3-2), 정답은 측선과 나란한 시굴 단면의 관 깊이다. 01.7 survey map에서도 측선 5개가 시굴 방향과 나란했다. 그래서 **측선과 평행한 관의 정답은 사실상 없다.**
- 후보는 메타 `교차 관로 = Yes`인 11개 활동(01.7, 01.9, 03.5, 03.6, 05.9, 08.1, 010.21, 010.24, 011.7, 011.17, 012.2)과 30 m 이상 지그재그 측선의 연결 구간이다. M2-6b analyze에서 `04_radargrams_png_view`로 띠가 보이는지만 확인한다. 정답표에 해당 관이 있으면 깊이를 참고로 비교하되 **판정에는 넣지 않는다**.
- M2-8 깊이 판정(헝가리안·제186조)에는 `pipe_band` 결과를 섞지 않는다. 손 씨앗 꼭짓점만 쓴다.

### 9-6. 판정 기준 (M2-6b verify)

| 대상 | `일치` | `허용 일치` | `불일치` |
|---|---|---|---|
| 단위 시험 UB1–UB15 | 전부 통과 | – | 하나라도 실패 |
| 의사 띠 BR1–BR7 | 전부 통과 | BR1만 0.5–1 cm | 그 밖 |
| 3D P01–P07 (`meta_fixed`) | $|d-d_\text{top}|\le2$ cm (P04는 표 B1 예측 ±1 cm, P05는 $|\theta_\text{est}-\theta|\le2°$, θ ≤ 20°) | ≤ 10 cm | > 10 cm |

공기층 모델(P06)에서는 v 기준을 보지 않고 d 기준만 본다(정의서 verify 규칙).

---

## 10. 근거: analyze 결과 요약

실행: `conda run -n gpr-dev python -I analysis/processing/pipe_band_check.py` (단일 프로세스, CPU 약 30 s, 시드 0, GPU 없음). 영점은 [S] §4-1/4-2 블라인드 규칙이고, v는 json 정답이다.

### 10-1. 표 A1. 의사 띠 깊이 (masked median 또는 배경 제거 없음, 덮음 0.4·1.0, `pipe_band_check_synthetic.csv`)

| 그룹 | 모델 | 띠 − 꼭짓점 시각 (ns) | 픽 MAD (ns) | $d-d_\text{top}$ (cm) | 띠 − 꼭짓점 규칙 깊이 (cm) |
|---|---|---|---|---|---|
| 지면결합 h 0 | M01–M11, M14, M16, M17, M19 | −0.070 ~ +0.028 | 0.002–0.011 | −4.04 ~ −1.74 | −0.35 ~ +0.14 |
| 공기 h 0.1 | M12, M13, M15, M18 | −0.058 ~ +0.009 | 0.002–0.006 | +1.81 ~ +3.71 | −0.22 ~ +0.06 |
| 공기 h 0.3 | M20 | −0.025 ~ −0.022 | 0.005–0.007 | −1.25 ~ −1.24 | −0.12 ~ −0.11 |

- 편향 부호와 크기는 [S] §12 표 4, R7·R8과 같다. 원인은 영점 규칙이지 띠 규칙이 아니다.

### 10-2. 표 A2. 측선 전체 median 배경 제거 (`median_all`)

| 덮음 | 띠 진폭비(배경 제거 없음 대비) | 띠 − 꼭짓점 규칙 깊이 (cm) | $d-d_\text{top}$ (cm) |
|---|---|---|---|
| 0.4 | 0.967–0.999 | −0.90 ~ −0.01 | −4.02 ~ +3.40 |
| 1.0 | **0.007–0.016** | **−25.8 ~ −9.7** | **−29.6 ~ −7.6** |

그림: `docs/processing/pipe_band_check.png`(M02, 배경 제거 3가지 × 덮음 2가지 포락선).

### 10-3. 표 A3. 띠 상단 기준 (상승점 50 % vs 포락선 피크)

앞 세 열은 정답 대비 오차 $d-d_\text{top}$(cm, 그룹 안 모델별 최소~최대)이다. 뒤 두 열은 같은 행(모델·덮음·배경 방식)에서 피크·피크 결과를 뺀 차(cm)다. 대상은 배경 제거 `none`·`masked_median` 행이고, 출처는 `pipe_band_check_synthetic.csv`의 `d_err_cm`, `d_onset_mix_err_cm`, `d_onset_both_err_cm`다. 행별 차를 따로 계산했으므로 앞 열 범위끼리 빼서 뒤 열을 만들 수 없다.

| 그룹 | 피크·피크 오차 | 상승점 띠 + 피크 영점 오차 | 상승점·상승점 오차 | 혼용 − 피크·피크 | 상승점·상승점 − 피크·피크 |
|---|---|---|---|---|---|
| 지면결합 (15개) | −4.04 ~ −1.74 | −12.21 ~ −5.42 | −3.99 ~ −1.37 | **−8.38 (M09) ~ −2.76 (M14)** | −0.54 (M01) ~ +1.29 (M14) |
| 공기 h 0.1 (4개) | +1.81 ~ +3.71 | −3.78 ~ −2.39 | +1.79 ~ +3.84 | **−6.63 (M13) ~ −4.29 (M15)** | −0.11 ~ +0.13 |
| 공기 h 0.3 (M20) | −1.25 ~ −1.24 | −5.72 ~ −5.70 | −3.51 ~ −3.49 | **−4.48 ~ −4.46** | −2.27 ~ −2.25 |

상승점 띠와 피크 영점을 섞으면 피크·피크보다 **−2.8 ~ −8.4 cm** 얕아진다(전체 범위). 예를 들어 M14는 −2.66 → −5.42 cm(차 −2.76), M09는 −4.00 → −12.11 cm(차 −8.11)다. 둘 다 상승점으로 잡으면 지면결합에서 모델에 따라 −0.5 cm(M01, 나빠짐) ~ +1.3 cm(M14, 나아짐)로 엇갈린다. M20에서는 2.3 cm 나빠지고, [S]의 영점 규칙과 위상 기준도 달라진다. 그래서 포락선 피크로 고정한다.

### 10-4. 표 B. 기하 해석 (`pipe_band_check_geometry.csv`)

**B1 측방 이격 편향**($\sqrt{d_c^2+y^2}-d_c$, cm)

| $d_\text{top}$ \ y | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| 0.3 (R 0.025 / 0.1) | 0.38 / 0.31 | 1.50 / 1.23 | 5.66 / 4.72 | 11.73 / 10.00 | 27.13 / 24.03 |
| 0.6 | 0.20 / 0.18 | 0.79 / 0.71 | 3.12 / 2.80 | 6.83 / 6.16 | 17.54 / 16.02 |
| 1.0 | 0.12 / 0.11 | 0.49 / 0.45 | 1.93 / 1.80 | 4.30 / 4.02 | 11.54 / 10.83 |
| 1.5 | 0.08 / 0.08 | 0.33 / 0.31 | 1.31 / 1.25 | 2.92 / 2.79 | 7.99 / 7.63 |

**B2 사교 쌍곡선의 곡률 v·깊이 배율** $1/\sin\theta$: 90° 1.000, 80° 1.015, 70° 1.064, 60° 1.155, 45° 1.414, 30° 2.000.

**B3 띠로 보이는 상한각**(반폭 $a$에서 시각 처짐 < $0.5/f_c$, v 0.1 m/ns)

| $f_c$ | $d_\text{top}$ | $a$ 0.5 m | 1.0 m | 2.0 m |
|---|---|---|---|---|
| 500 MHz (1.0 ns) | 0.3 / 0.6 / 1.0 | 21.1° / 30.0° / 39.8° | 10.4° / 14.5° / 18.7° | 5.2° / 7.2° / 9.2° |
| 400 MHz (1.25 ns) | 0.3 / 0.6 / 1.0 | 24.0° / 34.2° / 45.9° | 11.7° / 16.3° / 21.0° | 5.8° / 8.1° / 10.3° |

**B4 관 경사**: 경사율 1/2/5/10/20 % → $\cos\alpha$ 0.99995/0.9998/0.99875/0.99504/0.98058. 깊이 1 m당 −0.005/−0.02/−0.13/−0.50/−1.94 cm. 시각 기울기(v 0.1)는 0.2/0.4/1.0/2.0/3.9 ns/m다.

### 10-5. 표 C1. 작은 교차각 띠의 곡률 → 교차각 (`pipe_band_check_oblique.csv`, 픽 잡음 0.05 ns, v·R 기지)

| $d_\text{top}$ | θ | 반폭 | $p_2/\sigma$ | $\theta_\text{est}$ | 꼭짓점 깊이 오차 (cm) | 중앙값 깊이 오차 (cm) |
|---|---|---|---|---|---|---|
| 0.6 | 3° | 1.0 | 1.0 | (비유의) | +0.06 | +0.08 |
| 0.6 | 5° | 1.0 | 6.9 | 4.9° | −0.01 | +0.17 |
| 0.6 | 10° | 1.0 | 24 | 10.0° | −0.04 | +0.62 |
| 0.6 | 20° | 1.0 | 92 | 19.3° | +0.08 | +2.40 |
| 0.6 | 30° | 1.0 | 190 | 28.3° | +0.20 | +4.75 |
| 1.0 | 5° | 0.5 | −1.65 | (비유의) | – | −0.00 |
| 1.0 | 10° | 1.0 | 21 | 9.9° | +0.01 | +0.40 |
| 1.0 | 30° | 1.0 | 144 | 29.2° | +0.06 | +2.93 |

- θ ≤ 3°이거나 창이 짧으면 곡률이 유의하지 않다. 그때는 `flat`으로 처리하고, 오차는 ≤ 0.2 cm로 작다. 곡률이 유의해지는 θ에서는 꼭짓점을 써야 편향이 없다.

---

## 11. 가정과 미결

| 항목 | 현재 처리 | 확인 방법 |
|---|---|---|
| 2D 의사 띠 = 3D 평행 관 띠(시각·포락선) | 가정 | 3D P01·P09 |
| 편파가 시각에 영향 없음 | 가정 | P08 |
| masked median 여유 $m$ 0.1 m, 최소 바깥 30 트레이스 | 가정 | UB2, 3D |
| 조각 묶기 간격 8 px / 0.2 m, $|\Delta\tau|$ 허용 | 가정([AU] 근거는 px만) | AIHub test 박스 점검(§9-4) |
| 발자국 반경 식(측방 이격 상한) | Conyers & Goodman(1997) 경험식, 프로젝트 근거 문서 미등록 | web-research-expert로 출처 확인 또는 P04 |
| 이웃 v 깊이 근접 조건(±50 %), $\delta_\tau$ 0.05 | 가정 | 광저우 다관 측선 |
| 신뢰도 점수 계수 | 가정 | AIHub 분포만 확인, 4TU로 조정 금지 |
| `survey_hyperbola_median` $\sigma_v/v\ge0.10$ | 가정 | 광저우 존 내 분산 |
| 상태 문턱(coverage 0.3, n 5) | 가정 | UB, AIHub |
| 쌍곡선 쪽(`curvature_R_fixed`) 사교 교차 편향 $1/\sin\theta$(표 B2) 보고 | 이 명세는 띠의 이웃 v 편향(`neighbor_v_oblique_unknown`)으로만 다룬다. `pipe_hyperbola` 결과 자체의 `bias_notes` 반영은 spec_depth_v1 범위 | **핸드오프: M2-9 spec v2**(CHECKLIST M2-9, gpr-processing-expert). 3D P05로 크기 확인 |
