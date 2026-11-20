# M3-3 ③ E3x injected 1회차 `e3x_defect` — 질감 단언 결함 검토와 수정 방향 권고

- 작성: gpr-processing-expert(설계 문서 [`m33_e3x_plan.md`](m33_e3x_plan.md) v0.4 작성자), 2026-11-20.
- 대상: 봉인 2 = 커밋 `729d19ced8674039efac0cfe906fcb41f45c31e3`, 본 실행 HEAD `d720fe369322cfe2e63b0ef7eaa34ce338c59647`. injected 1회차가 종료 1 `e3x_defect`(3,119 s)로 끝났다.
- 성격: **설계 판단 문서**다. 코드와 규칙 문서는 고치지 않았다. v0.5 규칙 문장은 §3에 초안으로만 둔다. 사용자 결정은 §8 질문으로 남긴다.
- 덧붙임(같은 날): 사용자 결정과 EM 확인([EMC])을 받아 규칙 문서를 v0.5로 개정했다. 결정과 [EMC]가 고친 이 문서의 문장은 §9에, python-expert 구현 명세는 §10에 있다. §0–§8 본문은 검토 당시 기록으로 남겼다. 효력 문장은 규칙 문서 v0.5의 것이다.
- 블라인드: `runs/eval/m33/e3x/injected/`와 `runs/eval/m33/e3x*` 결과 파일은 열지 않았다. 읽은 것은 plan 표와 precheck plan 표(진단 스크립트가 배치 재구성에 씀), 배경 측선 원자료, 응답 라이브러리, python-expert 진단 산출물(`C:\Users\jinbeom\.claude\runners\gpr_detection-m33_e3x\diag_texture_20261120\`)이다. 이 진단 산출물은 합성 장의 에너지 지수이고 검출 결과가 아니다. 판정 결과값은 아직 없다.
- 이 문서에서 돌린 계산: §1-3·§1-4 표. 검출은 없고, 새 난수도 없다(질감 잡음장은 봉인 시드 `TEXTURE_SEED` 20261123으로 재현). E 코어 1개(논리 20), BelowNormal, `conda run -n gpr-dev python -I -B <스크립트> --out-dir <scratch>`로 돌렸다. 스크립트와 출력은 [`analysis/processing/m33_e3x_texture_defect/`](../../analysis/processing/m33_e3x_texture_defect/)에 옮겼다(sha256은 LF 정규화 기준).
  - `wrap_scan.py` `b8a6441ba1962ba92546ef9bc914df41c03b568c58f8f1f3cec4502ab2b11d8c`
  - `wrap_scan_summary.json` `203a542750b3c4e5c5c951fb81f66d9bf928689528a0d5ac0afff45bdea1bd38`
  - `wrap_scan_positions.csv` `bd026c34ab1863ad39119ad3c565d54bf5e746a958894bc1e26464b2f4e007d8`
  - `w_scan.py` `7c9dbbe2f9c1c69091e85ce6f7694a9e286c7b90247bc5cdf7774a93dede2d43`
  - `w_scan_summary.json` `1e6a0a32b6db25c1cc65847dd606c669eec6d07754a4d4e1bb7d152fcff1d18e`
  - `w_scan_positions.csv` `c89ddf5ac136c693e69992dd3ee60a2ec5db7d189cb4040c5fb398f547a090c9`
  - 두 스크립트는 저장소 밖 python-expert 진단 `texture_scan.py`(sha256 `014d31b39e3509ad20b970320aa853d839d0e0e9b9e8426e67c25de05d378b1c`)의 배치 재구성 함수를 import한다. 그 파일이 없으면 다시 돌릴 수 없다.
  - `w_scan.py`·`wrap_scan.py`는 `C:\Users\jinbeom\.claude\runners\gpr_detection-m33_e3x\diag_texture_20261120` 경로를 하드코딩해 `texture_scan.py`를 import하며 이 PC 전용이다.

## 0. 요약

| 질문 | 답 |
|---|---|
| 결함의 물리 의미 | 짝수 길이 열의 나이퀴스트 빈은 위상 회전 뒤 실수부만 남는다. 그래서 열 에너지가 정확히 $r_N\sin^2(\pi\delta)$ 비율만큼 준다. 표본 격자에서 지연된 나이퀴스트 성분이 실제로 그렇게 행동하므로 구현 오류가 아니다. **틀린 것은 단언의 전제**("0.8 나이퀴스트 저역통과가 걸린 열이라 정확함", [EMV] 5절 문장을 §2-7이 옮김)다. 열은 합성 영역에서 대역 제한이지만, 측선 격자의 유한 행으로 자른 열이라 그 길이의 DFT에서는 엄밀히 대역 제한이 아니다. 손실 크기는 $r_N\le1.03\times10^{-8}$(진폭 환산 −4.5e-8 dB 이하)라 물리적으로 아무 의미가 없다 |
| 권고 | **(b2) 해석적 손실 식으로 단언을 바꾼다.** 실현 비트는 바뀌지 않는다. 단언은 두 가지로 둔다. (i) 항등식: 모든 열에서 $\lvert E'_x/E-1+r_N\sin^2(\pi\delta_x)\rvert\le10^{-12}$(관측 최대 7.5e-15). (ii) 물리 한계: $r_N\le10^{-6}$(관측 최대 1.03e-8). (a)는 흔들림 팔의 기준 열을 평탄 팔과 다르게 만든다. (b1)은 나이퀴스트 처리를 검사하지 않는 자명한 단언이 된다. (b3)은 근거 없는 여유값이다. §2 |
| 가설·보고 영향 | 판정 가설 6개(HH1–HH3, HB1·HB1b·HB2)와 그 팔은 어느 안에서도 바뀌지 않는다. 평탄 팔은 `time_jitter`를 부르지 않는다. 질감 갈래 R-c는 보고 전용이고 (b2)에서는 비트도 같다 |
| 함께 볼 물리 사항(새 발견) | (1) 원형 이동이라 열 끝 $\lvert\delta\rvert$ 행이 반대쪽 끝으로 돈다. 돈 성분은 꼭짓점 열 최대의 0.75 % 이하(−42 dB)이고, 판정 SNR의 σ보다 24 dB 이상 아래라 무시한다(§1-3). (2) 질감장 $w$를 평탄 열 위에서 정규화한다. 평탄 구간이 상관 길이와 비슷해서, 실현된 흔들림은 대부분 완만한 굽음·기울기이고 테이퍼 열에서 외삽돼 커진다(최대 $\lvert\delta\rvert$ 1.69 P, σ_δ 0.2). R-c 해석 문구에 공개할 사항이다(§1-4, 질문 Q3) |
| 판 번호 | 규칙 문서 **v0.5**, 봉인은 **봉인 2 r1**(봉인 2 새 판, `m33_e3x_seal2_r1.json`, 옛 `m33_e3x_seal2.json`은 그대로 둠). §3 |
| 넓힌 precheck | precheck 10을 새로 둔다. `build_line`이 세는 CPU 단언 다섯(`shift_mismatch`, `off_differs_from_on`, `r_edge_differs_from_plan`, `texture_check`, `zero_test`)을 155측선 전부에서 같은 함수로 돌린다. 영상·추론은 만들지 않고, GPU 항목보다 먼저 돈다. 질감 부분 실측은 226.6 s·229.9 s(1 프로세스)이고, 쌍곡선 켬·끔 주입을 더한 전체는 1 프로세스 약 10–15분, 작업자 4개면 약 3–5분으로 추정한다. §4 |
| EM 검토 | (b2) 결정에는 필요 없다. 다만 §2-7 문장이 [EMV] 5절에서 왔으므로 정정 문구와 §1-4의 정규화 문제는 짧게 확인받기를 권고한다(비차단). 질문 Q1에서 (a)를 고르면 필요하다. §6 |
| stats 재평가 | 필요 없다. R-c는 보고 전용이고 MDE 평가([`m33_e3x_mde_eval.json`](../../analysis/processing/m33_e3x_mde_eval.json), d720fe3)에 질감 항목이 없다. plan 표도 바뀌지 않는다(질감 코드는 plan 경로에 없음). §7 |

## 1. 결함 기전과 물리 의미

### 1-1. 식

`time_jitter`(`analysis/detection/m33_e3x_place.py:998-1015`)는 꼭짓점 열 $x[k]$($k=0..n-1$)을 `rfft` → $e^{-i2\pi f_m\delta}$ → `irfft`로 옮긴다. 실수 입력이면 짝수 $n$의 나이퀴스트 계수 $X_{n/2}$는 실수다. 회전 뒤 `irfft`는 그 허수부를 버리므로 $X_{n/2}\cos(\pi\delta)$만 남는다. 파스발 식으로 적으면 다음과 같다.

$$
\frac{E'(\delta)}{E} = 1 - r_N\,\sin^2(\pi\delta),\qquad
r_N=\begin{cases}\dfrac{|X_{n/2}|^2/n}{E}, & n\ \text{짝수}\\[4pt] 0, & n\ \text{홀수}\end{cases},\qquad
E=\sum_k x[k]^2
$$

| 기호 | 뜻 | 단위 |
|---|---|---|
| $x[k]$ | 띠의 꼭짓점 열(`apex_column(band)`, 단계 (1)–(3) 전 합성 장) | 진폭 |
| $n$ | 열 행 수(`_patch_rows`, 기록 길이 이상, 합성 창 끝까지) | 샘플 |
| $\delta$ | 한 띠 열의 지연 $\sigma_\delta P_\text{line} w(x)$ | 샘플 |
| $X_m$ | $x$의 `rfft` 계수, $m=0..\lfloor n/2\rfloor$ | 진폭 |
| $E$, $E'$ | 이동 전·후 열 에너지 | 진폭² |
| $r_N$ | 나이퀴스트 빈 에너지 비 | – |

- 물리적으로는 표본 격자에서 나이퀴스트 성분 $\cos(\pi k)$를 $\delta$만큼 늦추면 $\cos(\pi(k-\delta))=\cos(\pi k)\cos(\pi\delta)$가 된다. 사인 항은 정수 $k$에서 0이다. 표본 열 하나로는 나이퀴스트 성분의 위상을 나타낼 수 없으므로, 손실은 구현 오류가 아니라 표본화의 성질이다.
- 홀수 $n$에는 나이퀴스트 빈이 없어 $E'=E$가 정확히 성립한다(24위치, 오차 e-15).
- 전제가 틀린 지점: 열은 `_line_field`에서 **합성 영역** FFT 길이로 0.8 나이퀴스트 저역통과를 거친다(`inject.py:1086-1088`). 그 뒤 `_band_limited_resample`이 측선 시각 $n$개에서 값을 매기고 합성 창 밖($t<0$, $t>$ `window_ns`)은 0으로 둔다(`inject.py:1006-1028`). 유한 행으로 자른 열은 **그 길이 $n$의 원형 DFT**에서 대역 제한이 아니다. 열 끝 값의 불연속과 창 밖 0 처리가 누설을 만든다. python-expert는 누설 출처를 시간창 절단으로 추정했는데, 물리적으로 맞는 추정이다. 다만 아래 판단은 출처가 무엇이든 크기만으로 정해진다.

### 1-2. 수치 (python-expert 진단, 전 위치 CPU 재계산)

출처: `diag_texture_20261120\scan_all\summary.json`. 155측선(본 153 + precheck 2), 질감 위치 101개(69측선)다.

| 항목 | 값 |
|---|---|
| 실패 위치 | 5개(3측선): 4TU `010.18/Path2` b00·b01, 4TU `011.16/Profile_6453` b00, IDS `pipe/1030_181230AB/LID10001` b00·b01. 다른 4필드(`w_mean`, `w_sd`, `amplitude_mean_square`, `jitter_sd`)의 실패는 0 |
| 실패 위치의 예측/실제 손실 비 | 0.9999993–1.0000003 |
| 열 길이 홀짝 | 짝수 77 · 홀수 24 |
| $r_N$ 지수 분포 | e-8 2, e-9 3, e-10 2, e-11 2, e-12 6, e-13 9, e-14 5, e-15 22, e-16 8, e-17 12, e-18 6, 0 24 |
| 0.8 나이퀴스트 위 에너지 비 | 최대 e-6(2위치) |
| (a) 모의 | 에너지 오차 전 위치 e-15(1e-12 통과), 잘려 나가는 에너지 최대 e-6, 정수 이동 vs `np.roll` 오차 e-16 |
| 시간 | 226.6 s(1 프로세스, 측선 중앙 0.61 s, 최대 6.17 s) |

### 1-3. 원형 이동과 (b2)·(a)의 크기 (이 문서의 계산)

스크립트는 `analysis/processing/m33_e3x_texture_defect/wrap_scan.py`다. python-expert 진단 스크립트의 배치 재구성 함수를 그대로 썼고, 155측선·101위치를 229.9 s에 돌렸다. 표의 값은 위치별 최대의 최대다.

| 항목 | σ_δ 0.1 P | σ_δ 0.2 P | 뜻 |
|---|---|---|---|
| (b2) 항등식 잔차 $\max_x\lvert E'_x/E-1+r_N\sin^2(\pi\delta_x)\rvert$ | 5.3e-15 | 7.5e-15 | 1e-12 단언 여유 약 130배 |
| 봉인 단언 오차 $\max_x\lvert E'_x/E-1\rvert$ | 1.03e-8 | 1.03e-8 | = $r_N$ 최대 |
| $r_N$ 최대 | 1.03e-8 | | 1e-6 한계 여유 약 100배 |
| 최대 $\lvert\delta\rvert$ | 0.85 P(20.7 샘플) | 1.69 P(41.5 샘플) | §1-4 참고 |
| 원형 − 선형(4n 영 채움) 이동 차 최대 / 꼭짓점 열 최대 | 0.29 % | 0.75 %(−42 dB) | 열 끝이 반대쪽 끝으로 돈 성분 |
| 정답 위 끝 `row0` 위로 들어간 그 차의 에너지 / $E$ | 1.2e-6 | 1.0e-5 | |
| 열 위 끝·아래 끝 $\lceil\max\lvert\delta\rvert\rceil+1$ 행의 최대 / 열 최대 | 위 2.2e-5, 아래 0.29 % | 위 2.2e-5, 아래 0.91 % | 아래 끝(합성 창 끝)이 0이 아니다 |
| (a) 대역 제한으로 바뀐 열 최대 / 열 최대 | 0.14 %(−57 dB), 정답 행 안 5.5e-5(−85 dB) | | (a)가 평탄 팔과 다르게 만드는 크기 |

- 돈 성분은 판정 SNR 12·18 dB의 잡음 σ(신호 대비 −12·−18 dB)보다 24 dB 이상 아래다. 검출에 보일 수 없다. 모든 안이 원형 이동을 유지하므로 이 값은 안 선택과 무관하다. 그래서 단언이 아니라 **기록(보고)**으로 둔다(§3 precheck 5).
- (a)의 변경도 −57 dB(정답 행 안 −85 dB)라 물리적으로는 무시할 수 있다. 다만 0이 아니고, 흔들림이 없는 열($\delta=0$)에서도 평탄 팔과 같아지지 않는다(§2).

### 1-4. 질감장 정규화의 실현 성질 (새 발견, 보고 전용 갈래의 해석 사항)

§1-3에서 최대 $\lvert\delta\rvert$가 1.69 P(σ_δ 0.2)로 나왔다. 명목 σ의 8.5배다. 원인을 보려고 봉인 `texture_field`를 같은 시드로 다시 만들어 정규화 전후를 비교했다(`analysis/processing/m33_e3x_texture_defect/w_scan.py`). 155측선·101위치를 돌렸고 봉인 $w$와의 재현 차는 0이다. 값은 위치별 분포(최소 / 중앙 / p90 / 최대)다.

| 항목 | 최소 | 중앙 | p90 | 최대 |
|---|---|---|---|---|
| 평탄 열 수 / 핵 표준편차 $\ell_c$(트레이스) | 3.07 | 3.60 | 3.80 | 3.94 |
| 정규화 전 평탄 열 표본 SD / 앙상블 SD($\sqrt{\sum k^2}$) | 0.085 | 0.40 | 0.79 | 1.25 |
| $\max\lvert w\rvert$ 평탄 열 | 1.26 | 1.86 | 2.43 | 3.02 |
| $\max\lvert w\rvert$ 상자 열(−6 dB 점 사이) | 1.26 | 1.96 | 3.44 | 5.86 |
| $\max\lvert w\rvert$ 테이퍼 열(평탄 밖) | 0.82 | 2.06 | 3.88 | 8.46 |
| σ_δ 0.2: 상자 열 $\max\lvert\delta\rvert$(P) | 0.25 | 0.39 | 0.69 | 1.17 |
| σ_δ 0.2: 상자 열 δ의 SD(P) | 0.18 | 0.22 | 0.26 | 0.36 |
| σ_δ 0.2: 상자 열 δ의 1차 추세 제거 SD(P) | 0.035 | 0.145 | 0.24 | 0.36 |

- **기전**: 평탄 구간은 핵 SD의 약 3.6배다. 결과 과정의 상관 SD($\sqrt2\,\ell_c$)로는 약 2.5배다. 이렇게 짧은 창에서는 창 평균을 뺀 표본 SD가 앙상블 SD보다 작다(기대 제곱비 약 0.32, 가우스 상관 해석값). 평탄 열 정규화는 이것을 정확히 1로 끌어올린다(중앙 2.5배, 최대 약 12배 증폭). 증폭된 완만한 추세가 평탄 밖 테이퍼 열로 외삽되므로 $\lvert w\rvert$가 커진다.
- **물리 의미**: 위치마다 실현된 흔들림은 "AIHub 수준의 짧은 상관 질감"이라기보다 완만한 굽음과 기울기다. 상자 열 δ 분산의 중앙 약 56 %가 1차 추세다(추세 제거 SD 0.145 P 대 0.218 P). 관 처짐·기울기와 비슷한 모양이다. 설계 문장의 원인 후보 셋 중 "관 처짐"에는 맞고, "지표 요철·속도 불균질"의 짧은 상관 질감과는 덜 닮았다.
- **[EMV] 5절 예측과의 관계**: EM 몬테카를로는 띠 안 표본 질감이 명목의 약 0.76배(1.9 / 2.5 dB)였다. 앙상블 정규화로 계산했다는 뜻이다. 구현은 띠 안 평탄 열 SD를 명목에 정확히 맞추므로, 같은 "0.2 P"에서 실효 흔들림이 예측 계산보다 약 1.3배 크다(추정). 그래서 $coh$ 예측(0.2 P에서 0.21–0.23)은 구현에서 더 낮게 나올 수 있다.
- **정답·덮임 판정과의 관계**: 정답 세로 범위는 흔들리지 않은 열 기준이다. 정답 높이 중앙은 2.41 P다(§2-4 집계). 띠 덮임은 정답과 시간 겹침 ≥ 정답 높이 50 %인 박스만 센다(§3-1). 상자 열에서 띠가 정답 행에서 벗어나는 양은 중앙 0.39 P(겹침 손실 약 16 %)로 문제가 없다. 최대 1.17 P에서는 겹침이 약 51 %로 문턱 경계다. 따라서 R-c 시간 흔들림 짝 차에는 검출 효과에 더해 "정답 행 고정"에서 오는 판정 몫이 일부 위치에서 섞일 수 있다.
- **판단**: 이 성질은 v0.2에서 [EMV] 5절 권고("표준편차 정규화를 평탄 열 위에서")로 정한 규칙을 구현이 그대로 따른 결과다. 구현 결함이 아니다. 보고 전용 갈래이고 판정 가설과 무관하므로, 권고는 **모형 유지 + 공개**다(질문 Q3, 공개 26). 바꾸려면 앙상블 정규화(핵의 해석 SD로 나눔)가 맞는 쪽이다. 그러면 precheck 5의 `std_flat(δ)` 단언이 성립하지 않으므로 다른 단언이 필요하고 EM 검토도 필요하다.

## 2. 후보 비교와 권고

| 항목 | (a) 코드: 기준 열을 자기 DFT에서 대역 제한 | (b1) 기준 = 나이퀴스트 빈 뺀 에너지 | **(b2) 해석적 손실 식 + 물리 한계** | (b3) 허용오차 ≥ 1e-7 |
|---|---|---|---|---|
| "원 열" 정의 | 대역 제한 열(0.8 나이퀴스트 위·나이퀴스트 빈 0) | 그대로 | 그대로 | 그대로 |
| 흔들림 팔 실현 비트 | 전부 바뀜 | 그대로 | 그대로 | 그대로 |
| 평탄 팔과의 짝 | 기준 열이 다름(최대 −57 dB, 정답 행 안 −85 dB). $\delta=0$이어도 평탄 팔과 같지 않음 | $\delta\to0$에서 평탄 팔로 돌아감 | 같음 | 같음 |
| 단언이 잡는 것 | 위상 회전 구현(1e-12). 열이 대역 제한인지는 묻지 않음(강제로 만듦) | 크기 보존뿐(위상 회전이면 자명). 나이퀴스트 처리 잘못(예: 허수부 처리, 길이 `n` 누락)은 못 잡음 | 위상 회전과 나이퀴스트 처리를 식으로(1e-12), 그리고 "열이 사실상 대역 제한"이라는 물리 전제를 한계로($r_N\le10^{-6}$) | 1e-7 안의 오차 전부를 통과시킴. 근거 없는 여유(관측 최대의 10배) |
| 코드 변경 | `time_jitter`·`jittered_band`·`texture_check`·시험 4 | `texture_check`·시험 4 | `texture_check`(`TextureCheck` 필드)·시험 4 | 상수 하나 |
| 규칙 문장 | §2-7 원 열 정의 변경 + 공개(기준 열 차) | §2-7 전제 정정 + 기준 정의 | §2-7 전제 정정 + 식 | §2-7 전제 정정 + 허용오차 근거 |

**권고: (b2).** 근거는 다음과 같다.
1. **물리**: 손실은 표본화의 성질이고 크기는 진폭 −4.5e-8 dB 이하다. 고칠 대상은 실현이 아니라 거짓 전제의 단언이다. (a)는 거짓 전제를 참으로 만들려고 장을 바꾼다. 그 대가로 질감 − 평탄 짝 비교에 흔들림이 아닌 차(필터 차, −57 dB)를 섞는다. 크기는 무시할 만해도 짝 비교의 정의("한 요인만 바꿈")를 깨는 쪽이다. 평탄 팔에도 같은 필터를 걸면 짝은 깨끗해지지만, 평탄 팔은 HB1·HB1b·HB2의 주 팔(긴 팔 47·68 px, 12·18 dB)이라 판정 팔을 바꾸게 되므로 쓸 수 없다.
2. **단언의 검출력**: (b2)는 구현이 정확히 "열 길이의 원형 DFT 위상 회전"인지를 1e-12로 확인한다. 나이퀴스트 빈을 빼거나 길이를 잘못 넘기는 결함도 잡는다. 물리 전제는 별도 한계로 확인한다. (b1)은 크기 보존만 보므로 위상 회전이면 늘 통과한다. (b3)은 무엇을 확인하는지 말할 수 없다.
3. **변경 범위**: 실현 비트가 그대로라 재실행의 비질감 팔과 질감 팔이 모두 1회차와 같은 정의다(1회차 폴더는 비교용으로도 열지 않는다, §5). 바뀌는 blob은 `m33_e3x_place.py`(그리고 precheck 10·재시도 경로 때문에 `m33_e3x.py`·판정 도구 입력 경로) 뿐이다.
4. **문턱의 근거**: 항등식 1e-12는 float64 합의 최악 상대 오차($n\varepsilon\approx742\times1.1\times10^{-16}\approx8\times10^{-14}$)보다 10배 이상 크고 관측 최대(7.5e-15)의 약 130배다. $r_N\le10^{-6}$은 진폭 손실 $\le r_N/2$ = 5e-7(−4.3e-6 dB)이 R3 기울기(약 1.9 %p/dB) 기준 1e-5 %p로 보고 전용 해석에도 닿지 않는 값이다. 동시에 −60 dB를 넘는 나이퀴스트 성분이면 "원형 위상 회전 = 대역 제한 지연"이라는 모형 전제가 흔들린다는 뜻이라 구현 결함 신호로 쓴다. 두 문턱 모두 진단 분포를 본 뒤 정했다. 다만 판정 문턱이 아니라 구현 단언이고, 검출 결과는 누구도 보지 않았다(공개 25).

**"원 열"을 대역 제한 열로 바꾸면(= (a)) 가설·보고에 생기는 일** — 질문 Q1에서 (a)를 고를 때 v0.5에 넣을 공개 문장의 근거다.
- 판정 가설: 영향 없음. 흔들림 팔만 바뀐다.
- R-c (ii)·(iii): "질감 − 평탄" 짝 차가 "흔들림 + 기준 열 필터(최대 −57 dB)"의 효과가 된다. 해석 문구는 그대로 두고, 공개에 "기준 열이 평탄 팔과 최대 −57 dB 다름"을 넣어야 한다.
- 정답 세로 범위: v0.4처럼 **필터 전 흔들리지 않은 열**의 `row0`·`row1`(평탄 팔과 같은 정답)을 써야 한다. 필터 뒤 열로 다시 재면 −20 dB 문턱 통과가 1 샘플 바뀔 수 있다(v0.4 [PDD]: 얕은 위치의 문턱 통과 값 0.1003–0.1089). 그러면 질감 팔과 평탄 팔의 정답이 달라진다.
- 영점·주기 변경 열([ZD])과 $coh$·$c_\text{lat}$: 흔들림 팔의 값이 비트 단위로 바뀐다(보고 전용).

**다른 단언과의 상호작용**(모든 안에 공통, (a)에서 추가되는 것은 위 목록)
| 단언·규칙 | 상호작용 |
|---|---|
| 띠 위 끝 고정(`shift_mismatch`: 보정 뒤 1 샘플, 팔 사이 $k_1'(n_2)-n_2$ 같음) | 흔들림 **전** 띠에서 판정한다(`rebuild_arm`의 정답 상자 일치). 흔들림은 그 뒤 `jittered_band`에서만 걸리므로 서로 독립이다 |
| 순수 평행 이동(HB1·HB1b "같은 파형") | 평탄 팔에만 해당하고 영향 없다. 흔들림 팔은 47·68 px 긴 팔뿐이고 판정에 들어가지 않는다 |
| 끝 테이퍼 T | `jittered_band`가 이동 뒤에 `edge_taper_weights`를 곱한다. T와 무관하다 |
| SNR 정의 | 이득은 `band.signal_amplitude` = 흔들리지 않은 패치의 $A$로 정한다(`inject.py:2431`). 에너지 손실과 무관하게 이득·σ가 평탄 팔과 같다 |
| 정답 세로 범위 | 흔들리지 않은 열 기준(§2-7)이다. §1-4의 외삽 때문에 상자 열에서 실제 띠가 정답 행에서 벗어나는 양이 명목 σ보다 크다(해석 사항, Q3) |
| 끝 처리 켬·끔 비트 단언, $r_\text{edge}$ 재측정, 영 시험 | 쌍곡선·배경 경로라 무관하다 |

## 3. v0.5 개정 초안 (규칙 문서는 아직 고치지 않음)

판은 **v0.5**다. 효력은 **봉인 2 r1**(봉인 2 새 판) 커밋에서 생긴다. 봉인 기록은 새 파일 `analysis/detection/m33_e3x_seal2_r1.json`으로 둔다(키 `supersedes`: 옛 경로·sha256(LF)·커밋 `729d19c`). 옛 `m33_e3x_seal2.json`은 1회차 실행 정체성의 근거이므로 고치지 않는다. plan 표와 `plan_report.json`의 sha는 그대로 옮긴다(§7). 아래 초안의 `<…>` 자리는 메인 세션이 결과 json의 `run_identity` 블록(정체성 키만 허용, CLAUDE.md)에서 64자리 그대로 채운다.

### 3-1. 머리말·판 이력 (3–9행)

- 상태 줄: "**v0.5(2026-11-2x, gpr-processing-expert). 봉인 2(729d19c, v0.4) 뒤, injected 1회차 `e3x_defect` 뒤의 새 판(결과 미열람).** 봉인 2 r1에서 이 판이 효력을 갖는다."
- 판 이력에 추가: "**v0.5(2026-11-2x): injected 1회차(run_id `<run_id>`, HEAD d720fe3, 종료 1 `e3x_defect`, 3,119 s)의 실패 단언 `texture_check`(필드 `jitter_energy`)를 사용자 결정 v5-1–v5-n으로 고침.** 시간 흔들림 에너지 단언을 해석적 손실 식과 물리 한계로 바꾸고(§2-7, §5-2 5, A-5 4), CPU 구현 단언을 전 측선 precheck로 넓혔다(§5-2 10). 판정 가설·판정 규칙·문턱·시드·실현(비트)은 바꾸지 않았다. 결과는 아무도 열지 않았다(§4 공개 22–25). 근거: [TDR] [`m33_e3x_texture_defect_review.md`](m33_e3x_texture_defect_review.md)." (초안 자리 표시, 효력 문장은 규칙 문서)
- 남은 순서 줄: "… → 본 실행 1회차(injected `e3x_defect`, §4 공개 22) → **v0.5** → python-expert v0.5 구현 → 봉인 2 r1 → precheck r1(사본) → verif 실행 전 감사 재확인 → (uninjected001 재시도, 사용자 결정 v5-3) → injected 재시도(`runs/eval/m33/e3x_retry/`) → 판정 도구 → 판정서 → verif 판정 후 감사."
- 사용자 결정 표(셋째 표 뒤에 넷째 표):

| # | 문제(injected 1회차) | 결정 | 반영 |
|---|---|---|---|
| v5-1 | `texture_check.jitter_energy` 5위치 실패(짝수 길이 열의 나이퀴스트 빈 손실, $r_N\le1.03\times10^{-8}$) | (Q1 답. 권고 (b2)) 해석적 손실 식 1e-12 + $r_N\le10^{-6}$, 실현 불변 | §2-7, §5-2 5, A-2, A-5 4 |
| v5-2 | precheck 2측선이 결함을 못 잡음(두 측선 $r_N$ e-17–e-18) | CPU 구현 단언 전부를 전 측선 precheck 10으로, GPU 항목 전에(사용자 결정 2026-11-20) | §5-2 10, §5-4 |
| v5-3 | 1회차 uninjected001 출력의 정체성이 옛 blob | (Q2 답) | §5-4, A-4 |
| v5-4 | 질감장 평탄 열 정규화의 실현 성질(§1-4) | (Q3 답. 권고: 모형 유지 + 공개) | §2-7, §4 공개 26 |

### 3-2. §2-7 (ii)·(iii) 문단 (301행) 바꿀 문장

> - (ii)·(iii) 시간 흔들림: $\delta(x)=\sigma_\delta P_\text{line} w(x)$ 샘플, $\sigma_\delta \in \{0.1, 0.2\}$ P. 꼭짓점 열(평탄 팔과 같은 열, 길이 $n$ 행)을 띠 열마다 $\delta(x)$만큼, **그 열 길이의 원형 DFT에서 위상 회전**으로 옮긴다(`rfft` → $e^{-i2\pi f\delta}$ → `irfft`, 길이 $n$). 열은 합성 영역에서 0.8 나이퀴스트 저역통과를 거쳤다. 그러나 측선 격자의 유한 행으로 자른 열이라 길이 $n$의 DFT에서는 엄밀히 대역 제한이 아니다(v0.4의 "정확함"은 틀린 전제였다, 공개 22). 그래서 두 가지가 생긴다. (1) $n$이 짝수면 나이퀴스트 빈이 회전 뒤 실수부만 남아 열 에너지가 아래 식만큼 준다. 표본 격자에서 지연된 나이퀴스트 성분의 성질이다. (2) 원형 이동이라 열 끝 $\lvert\delta\rvert$ 행이 반대쪽 끝으로 돈다. injected 1회차 전 측선 진단([TDR] §1)에서 $r_N\le1.03\times10^{-8}$(진폭 −4.5e-8 dB 이하)였고, 돈 성분은 꼭짓점 열 최대의 0.75 % 이하(−42 dB, 판정 SNR의 σ보다 24 dB 이상 아래)였다. 둘 다 물리적으로 무시한다. 이득은 흔들리지 않은 열의 $A$(`signal_amplitude`)로 정하므로 SNR 정의는 바뀌지 않는다. 정답 세로 범위는 흔들리지 않은 열 기준으로 둔다.
>
> $$\frac{E'(x)}{E}=1-r_N\sin^2\!\big(\pi\delta(x)\big),\qquad r_N=\frac{|X_{n/2}|^2/n}{E}\ (n\ \text{짝수}),\quad r_N=0\ (n\ \text{홀수})$$
>
> (기호 표: [TDR] §1-1과 같음)
>
> - 실현 성질(v0.5 공개, Q3에 따라): $w$는 평탄 열 위에서 정규화한다. 평탄 구간이 핵 표준편차의 약 3.6배(3.1–3.9)라 위치마다 실현은 주로 완만한 굽음·기울기다(상자 열 δ 분산의 중앙 약 56 %가 1차 추세). 테이퍼 열에서는 외삽돼 $\lvert\delta\rvert$가 커진다(σ_δ 0.2에서 상자 열 최대 중앙 0.39 P·최대 1.17 P, 테이퍼 열 최대 1.69 P). [EMV] 5절의 $coh$ 예측은 앙상블 정규화(띠 안 표본 질감이 명목의 약 0.76배) 계산이라, 구현의 띠 안 실효 흔들림은 그 예측보다 약 1.3배 크다(추정, [TDR] §1-4).

### 3-3. §5-2 precheck 표 (521행 5번, 새 10번)

- 5번 바꿀 문장:

> | 5 | 질감(v0.5): 평탄 열 위에서 $w$의 평균 0·표준편차 1(상대 오차 1e-12), 진폭 질감 평탄 열 평균 제곱 1(1e-12), 시간 흔들림 $\operatorname{std}_\text{flat}(\delta) = \sigma_\delta P_\text{line}$(1e-12), **시간 흔들림 열 에너지가 §2-7 식과 같다(모든 띠 열, $\lvert E'_x/E-1+r_N\sin^2(\pi\delta_x)\rvert\le10^{-12}$)**, **나이퀴스트 빈 에너지 비 $r_N\le10^{-6}$**. 기록(단언 아님): 위치별 $r_N$, 0.8 나이퀴스트 위 에너지 비, 원형 이동과 선형(4n 영 채움) 이동의 차 최대(꼭짓점 열 최대 대비), 상자 열·테이퍼 열의 $\max\lvert\delta\rvert$(P) |

- 10번 새 줄:

> | 10 | **전 측선 CPU 구현 단언**(v0.5): 본 표본 153측선과 precheck 2측선 모두에서, injected 단계 `build_line`이 세는 CPU 단언 다섯을 **같은 함수·같은 순서**로 계산한다. `shift_mismatch`(띠 팔 재구성과 plan 정답 상자 일치, `rebuild_arm`), `off_differs_from_on`(끝 처리 끔 장이 개구 테이퍼 열 밖에서 켬과 비트 단위로 같음, 모든 쌍곡선 실현, `inject_realization`·`inject_untapered`·`_off_matches_on`), `r_edge_differs_from_plan`(상대 1e-9, 조작 층·회색 띠·음성 대조 위치 모두), `texture_check`(5번 전 필드, 질감 위치 전부), `zero_test`(측선마다)다. 영상화(`make_detection_image`는 영 시험의 배경 1장만)·추론·박스는 만들지 않는다. **건수 가드**: 단언마다 검사 수가 plan 표에서 센 기대 수와 같고 0보다 크다. 하나라도 어기면 `precheck_failed`. 10번은 1번 다음, 2–9번(GPU 포함) 전에 돌고, 실패하면 2–9번을 돌리지 않고 종료 1 |

- 표 아래 문단에 추가:

> - 10번은 검출기를 쓰지 않는다. 본 표본 측선에서 돌려도 판정 값이 드러나지 않는다(합성 장의 구현 단언과 그 오차 지수뿐). 기록은 precheck 폴더의 `cpu_assertions.json`(정체성, 단언별 검사 수·기대 수·실패 수, 실패 목록 (측선, 위치 또는 실현 id, 단언, 오차), 5번 기록 항목의 분포, 측선별 시간)과 `cpu_assertions_lines.csv`(측선별 단언 수·실패 수·시간)다. stdout 마지막 줄에는 수만 낸다(`status`, `n_lines`, `n_checks`, `n_failed`, `elapsed_s`).
> - 10번이 injected 단계의 단언을 대신하지는 않는다. 본 실행도 같은 단언을 다시 센다(`e3x_defect`). 검출이 필요한 단언(적중 ⇒ 최고 점수 > 0.25, 덮임 단언, NMS 경고, 0.25 초과 박스 300개 타일)은 CPU precheck로 옮길 수 없다. 이들은 precheck 2측선(6·9번)과 본 실행에서만 확인한다(공개 24).
> - precheck 폴더는 시도마다 새로 만든다(`runs/eval/m33/e3x_precheck_r1/`, 실패하면 `_r2` …). 있으면 거부한다. 1회차 폴더(`e3x_precheck/`, `e3x_precheck_copy/`)는 지우지도 덮어쓰지도 않는다(이전 증거). verif 실행 전 감사 재확인 직전에 한 번 돌린다. 끝나면 폴더째 사본(`e3x_precheck_r<k>_copy/`)과 두 폴더의 파일별 sha256(LF) 목록을 남긴다(CLAUDE.md 사전 등록 실행 규칙).

### 3-4. §5-3 블라인드 경로 (536행)

> - **블라인드 경로**: `runs/eval/m33/e3x/uninjected001/`, `runs/eval/m33/e3x/injected/`(1회차, 열지 않고 보존), `runs/eval/m33/e3x/judgment/`, `runs/eval/m33/e3x_retry*/`, `runs/eval/m33/e3x_precheck*/`.

### 3-5. §5-4 실패 처리표 (539–557행)

- precheck 1 행: "`precheck_failed`(10번 전 측선 CPU 단언 포함)" — 다시 돌려도 되나: "본 실행 금지. 코드 수정 → 새 blob → 봉인 2 새 판(r<k+1>) → verif 실행 전 감사 재확인 → 새 폴더 `e3x_precheck_r<k+1>`".
- injected `e3x_defect` 행의 보고 칸: "단언 이름 공개. 1회차(2026-11-20): `texture_check`의 `jitter_energy`(σ_δ 0.1·0.2) 5위치, 공개 22".
- 새 행 둘:

> | injected 재시도 | 0/1/2/3 | (같음) | `runs/eval/m33/e3x_retry/injected/`에서 v0.5(봉인 2 r1)로 한 번. 재시도도 부분 출력으로 끝나면 그 폴더는 열지 않고 남기고 `runs/eval/m33/e3x_retry2/`(k = 2, 3, …)에서 다시. 코드를 고쳤으면 새 blob → 봉인 2 새 판 → verif 재확인 | 열지 않았다면 아니다 | 이전 시도의 폴더·종료 코드·실패 단언 이름 |
> | 실패 폴더 | – | – | 이전 시도 폴더는 원인 확인을 위해서라도 기록 파일을 열지 않는다. 예외는 `summary.json`·`parts\*.json`의 `defects` 키(0이 아닌 키 이름)와 `line_id`뿐이다. 이 열람은 스크립트로 하고, 스크립트·stdout·실행 시각을 남겨 공개한다(공개 23). 재시도 결과와 비교하려고 이전 시도 폴더를 열거나 해시하지 않는다 | | 열람 범위 공개 |

### 3-6. 부록 A-2 (640행)와 A-5 시험 4 (672행)

- A-2 `time_jitter` 줄:

> - `time_jitter(column, w, sigma_periods, period_samples) -> np.ndarray`: 열마다 길이 $n$ 원형 DFT 위상 회전으로 $\delta(x)$ 샘플 이동(v0.4와 같은 계산). docstring의 "exact and its energy is kept"를 §2-7 식(짝수 $n$의 나이퀴스트 손실)으로 고친다.
> - `texture_check`(v0.5): `TextureCheck`의 `jitter_energy`를 `jitter_energy_identity`(σ마다 $\max_x\lvert E'_x/E-1+r_N\sin^2(\pi\delta_x)\rvert$, ≤ 1e-12)와 `jitter_nyquist_fraction`($r_N$, ≤ 1e-6, σ와 무관하게 한 값)으로 바꾼다. 기록 필드(통과 판정에 안 씀): `jitter_above_cutoff_fraction`, `jitter_wrap_rel`(σ마다), `jitter_delay_absmax_box_P`·`jitter_delay_absmax_taper_P`(σ마다). $r_N$은 `rfft` 파스발 가중(0번·나이퀴스트 1, 나머지 2)으로 계산한다.

- A-5 시험 4 바꿀 문장:

> | 4 | `texture_field`: 평탄 열 평균 0·표준편차 1, 시드 고정 재현. 생성기 단위 확인(열 ≥ 1,000개)에서 자기상관 표준편차가 $\sqrt2\,\ell_c$에 ±10 % 안. `amplitude_texture` 평탄 열 평균 제곱 1. `time_jitter`: 정수 이동에서 `np.roll`과 1e-9 안, $\delta=0$에서 원 열과 1e-12 안. **에너지(v0.5)**: 홀수 길이 열에서 모든 δ에 에너지 보존(1e-12). 짝수 길이 손 계산 fixture(예: $n=64$, 저주파 펄스 + $a(-1)^k$)에서 $E'/E=1-r_N\sin^2(\pi\delta)$(1e-12)이고, δ = 0.5 샘플에서 손실이 $r_N$과 같다. $r_N>10^{-6}$인 fixture에서 `texture_check`가 실패하고 $r_N\le10^{-6}$이면 통과한다. 나이퀴스트 빈을 0으로 만들고 비교하는 잘못된 구현(b1형)을 흉내 낸 fixture에서 항등식 단언이 실패한다. **precheck 10**: precheck 2측선에서 10번이 세는 단언별 실패 수·검사 수가 `build_line`의 `defects`·검사 수와 같다. 검사 수가 0인 fixture에서 건수 가드가 실패한다 |

### 3-7. §4 공개 사항 (22–26 추가)과 §6 판정서 항목

- §6 2항 뒤에 덧붙임: "injected 1회차의 정체성(`<run_id>`, HEAD·blob, 시작·끝 시각, 3,119 s), 실패 단언 이름과 위치 5개, 1회차 폴더를 열지 않았다는 확인, 원인 확인 열람 범위(공개 23), precheck r1과 그 사본의 경로·sha256(10번 기록 포함)." (초안 자리 표시, 효력 문장은 규칙 문서)
- §6 9항: "공개 22–26 포함".
- 공개 문장은 §5에 그대로 옮길 수 있게 적었다.

### 3-8. 변경 이력 (끝 표)

> | 2026-11-2x | v0.5 | 봉인 2(729d19c, v0.4)와 injected 1회차(종료 1 `e3x_defect`, 결과 미열람) 뒤 새 판. 사유: `texture_check`의 `jitter_energy` 단언 실패(짝수 길이 열의 나이퀴스트 빈 손실 $r_N\sin^2(\pi\delta)$, $r_N\le1.03\times10^{-8}$, [TDR]). 변경: §2-7 시간 흔들림 전제 정정·손실 식·원형 이동 성질·(Q3) 실현 성질 공개, §5-2 5(항등식 1e-12, $r_N\le10^{-6}$, 기록 항목), §5-2 10(전 측선 CPU 구현 단언, GPU 전, 건수 가드, 기록·사본), §5-3 블라인드 경로(`e3x_retry*`), §5-4 precheck·injected 재시도·실패 폴더 열람 행, §4 공개 22–26, §6 2·9항, A-2 `texture_check` 필드, A-5 시험 4. 판정 가설·규칙·문턱·시드·실현 비트·plan 표는 바꾸지 않음 |

## 4. 넓힌 precheck(10번)의 범위 명세

### 4-1. 단언별 범위

| 단언(`defects` 키) | 함수(`m33_e3x.py`) | CPU만으로 판정 | precheck 10 범위 | 비고 |
|---|---|---|---|---|
| `shift_mismatch` | `band_entries` → `rebuild_arm`(정답 상자 = plan) | 예 | 전 측선·모든 띠 팔 | plan 단계의 같은 단언(`arm_offsets_not_parallel` 포함)은 plan에서 이미 0 |
| `off_differs_from_on` | `hyperbola_entries` → `inject_realization`·`inject_untapered`·`_off_matches_on` | 예(주입 장만, 영상 불필요) | 전 측선·모든 (묶음, SNR, 극성) | 가장 비싼 부분(쌍곡선 주입 켬·끔 약 1,800회) |
| `r_edge_differs_from_plan` | `r_edge_matches`(`hyperbola_patches`) | 예 | 전 위치 | v0.4 precheck에는 음성 대조·회색 띠 위치가 없었다(실행 전 감사 [`m33_e3x_prerun_audit.md`](m33_e3x_prerun_audit.md) 112행). 이번에 처음 확인된다 |
| `texture_check` | `_Textures.of` → `texture_check` | 예 | 전 질감 위치(1회차 진단 101위치) | v0.5 필드 |
| `zero_test` | `zero_test_passes` | 예(배경 영상 1장) | 전 측선 | |
| `hit_without_operating_box`, `cover_without_operating_box`, `coverage_over_length`, `tile_with_300_operating_boxes`, `nms_time_limit_warning` | `_fold_defects`, NMS handler | **아니오**(검출 필요) | 넣지 않음 | 본 표본에서 돌리면 검출 결과가 되므로 블라인드 위반. precheck 2측선의 6·9번과 본 실행만 |

- **같은 함수 조건**: 10번은 `build_line`을 영상 생성 없이 부르는 경로다. 예: `build_line(..., images=False)`처럼 같은 본체에서 영상·`_band_realization`만 건너뛴다. 단언 계산부를 복사한 별도 함수는 쓰지 않는다. 시험(A-5 4)으로 precheck 2측선에서 두 경로의 `defects`가 같음을 단언한다.
- 띠 질감 단언은 `band_entries`가 영상을 만들 때 `_Textures.of`를 부르는 순서(행·팔 정렬, 위치 id 정렬, 질감 조건 순서)와 같은 순서로 부른다. python-expert 진단의 `first_built` 재구성이 봉인 결과와 5/5 같았으므로 이 순서는 재현된다.

### 4-2. 시간 근거와 자원

| 부분 | 근거 | 1 프로세스 |
|---|---|---|
| 측선 적재·문맥·띠 재구성·질감 단언 | python-expert `texture_scan.py` 226.6 s, 이 문서 `wrap_scan.py` 229.9 s(155측선, 같은 재구성) | 약 4분(실측) |
| 쌍곡선 켬·끔 주입과 `_off_matches_on` | R3 injected 주입 525 s / 2,159실현 ≈ 0.24 s/실현([`m33_e3x_plan.md`](m33_e3x_plan.md) §5-1 시간 근거) × E3x 쌍곡선 실현 약 1,800(켬·끔 합) | 약 7분(추정) |
| $r_\text{edge}$ 재측정(약 144위치 × 단계 (1)–(3)) + 영 시험(155 × 배경 영상 1장, 약 0.24 s) | R3 단면·영점 508 s / 2,159 | 약 1–2분(추정) |
| 합계 | | **약 10–15분**. 작업자 4개면 약 3–5분(추정) |

- "약 4분"은 질감 부분만의 실측이다. 쌍곡선 끔 장 단언을 넣으면 1 프로세스로는 4분을 넘는다. python-expert가 구현 뒤 precheck r1 실행의 `cpu_assertions_lines.csv` 측선별 시간으로 실측을 남긴다.
- 30분을 넘지 않을 것으로 보지만, precheck 전체(1–10)는 1회차 1,610 s에 10번이 더해진다. 시작 직전 `status.py`로 작업자 수를 정한다(CLAUDE.md 자원 유동 배분).

### 4-3. 기록·사본 규칙(요약, 문장은 §3-3)

- 출력: `runs/eval/m33/e3x_precheck_r<k>/cpu_assertions.json`, `cpu_assertions_lines.csv`(precheck의 다른 파일과 같은 폴더). `run_identity`(HEAD·blob·src tree, 64자리), `started_at`은 시작 시각(실행 전 감사 지적 78–81행 반영), 단언별 `n_checked`·`n_expected`·`n_failed`, 실패 목록, 기록 항목 분포.
- 메인 세션은 종료 코드·마지막 줄만 본다(precheck 폴더는 블라인드 경로 패턴 `e3x_precheck*`에 든다). 10번 기록은 검출 값이 아니므로 verif 감사가 열어도 된다.
- 사본: 실행 직후 폴더째 `_copy`, 파일별 sha256(LF) 목록. 판정서 §6 2항에 경로·sha를 적는다.

## 5. 공개 사항 목록 (v0.5 §4에 옮길 문장 초안)

22. **injected 1회차 실패**: 봉인 2(`729d19ced8674039efac0cfe906fcb41f45c31e3`) 뒤 본 실행(HEAD `d720fe369322cfe2e63b0ef7eaa34ce338c59647`, run_id `<run_id>`)의 injected 단계가 종료 1 `e3x_defect`(3,119 s)로 끝났다. 실패 단언은 질감 단언 `texture_check`의 하위 필드 `jitter_energy`(σ_δ 0.1·0.2 둘 다) 하나였다. 위치는 4TU `010.18/Path2` b00·b01, 4TU `011.16/Profile_6453` b00, IDS `pipe/1030_181230AB/LID10001` b00·b01이다. 원인은 짝수 길이 꼭짓점 열의 나이퀴스트 빈 에너지 손실($r_N\sin^2(\pi\delta)$, $r_N\le1.03\times10^{-8}$)이다. 단언의 전제("0.8 나이퀴스트 저역통과가 걸린 열이라 정확함")가 틀렸다. 이 전제는 [EMV] 5절 문장에서 왔고, 설계자가 v0.2 §2-7에 검증 없이 옮겼다. 1회차 폴더 `runs/eval/m33/e3x/injected/`는 열지 않고 남겼다. 재시도는 v0.5(봉인 2 r1)로 `runs/eval/m33/e3x_retry/`에서 했다. (초안 자리 표시, 효력 문장은 규칙 문서)
23. **원인 확인 열람 범위**: python-expert가 원인을 확인하려고 1회차 폴더의 `summary.json`·`parts\*.json`에서 `defects` 키의 0이 아닌 키 이름과 `line_id`만 스크립트로 출력했다. 재현율·적중·덮임·점수·대비·영점 키는 출력하지 않았다. 이 열람은 §5-3이 메인 세션에 허용한 범위(종료 코드·마지막 줄·`run_identity`·`judgeable`)를 넘으므로 공개한다. `defects` 이름은 §5-4가 "단언 이름 공개"로 정한 정보다. 다만 0이 아닌 키가 `texture_check`뿐이었다는 출력은 검출 의존 단언(`hit_without_operating_box` 등)이 0이었다는 것도 알려 준다. 이 단언들은 구현이 맞으면 늘 0인 값이라 판정 정보가 아니다. 그래서 결과를 본 것으로 치지 않는다(판단: gpr-processing-expert, verif 재확인 대상). 열람 스크립트·stdout·시각은 `<경로>`에 남겼다. (초안 자리 표시, 효력 문장은 규칙 문서)
24. **precheck 표본 한계**: v0.4 precheck는 질감 단언을 precheck 2측선에서만 돌렸다. 그 두 측선의 $r_N$은 e-17–e-18이라 결함을 잡을 수 없었다(본 표본에서 실패한 위치의 $r_N$은 e-8–e-9). 실행 전 감사도 precheck에 음성 대조·회색 층 위치가 없다고 기록했다. v0.5는 CPU 구현 단언 다섯을 전 측선 precheck 10으로 넓혔다. 검출이 필요한 단언은 여전히 precheck 2측선과 본 실행에서만 확인한다.
25. **진단 CPU 재계산(검출 없음)**: 원인 확인과 수정안 비교를 위해 다음을 돌렸다. (i) python-expert `texture_scan.py`(155측선 질감 단언 재계산과 (a) 모의, 226.6 s). (ii) gpr-processing-expert `wrap_scan.py`·`w_scan.py`(원형 이동 효과, (b2) 항등식 잔차, (a) 열 변화, 질감장 실현 성질). 셋 다 plan 표·precheck plan 표·배경 측선·응답 라이브러리만 읽었다. 영상·추론·박스는 만들지 않았다. 출력은 합성 장의 에너지 지수와 지연 크기다. v0.5의 두 문턱(항등식 1e-12, $r_N\le10^{-6}$)은 이 분포를 본 뒤 정한 구현 단언이고 판정 문턱이 아니다.
26. (Q3에 따라) **시간 흔들림 실현 성질**: $w$의 평탄 열 정규화 때문에 위치마다 실현은 주로 완만한 굽음·기울기다. 상자 열 최대 지연은 σ_δ 0.2에서 중앙 0.39 P·최대 1.17 P, 테이퍼 열 최대 1.69 P다. 띠 안 실효 흔들림은 [EMV] 5절 예측 계산보다 약 1.3배 크다(추정). 정답 세로 범위는 흔들리지 않은 열 기준이다. 그래서 R-c 시간 흔들림 갈래의 짝 차에는 "정답 행에서 벗어난 띠"의 덮임 판정 몫(시간 겹침 ≥ 정답 높이 50 % 조건)이 섞일 수 있다.

## 6. EM(electromagnetic-simulation-expert) 검토가 필요한가

- **(b2) 결정 자체에는 필요 없다.** 판단은 출처와 무관하게 크기($r_N\le1.03\times10^{-8}$, 돈 성분 −42 dB)로 정해진다. 1-1의 식은 표본화 이론의 항등식이다.
- **짧은 확인을 권고한다(비차단, 봉인 2 r1 전).**
  1. §2-7 정정 문장: 틀린 전제가 [EMV] 5절 "수정 문장"에서 왔다. 원 작성자가 정정 문구에 동의하는지 확인받는다.
  2. 누설 출처: `_band_limited_resample`의 합성 창 밖 0 처리와 유한 행 절단. 같은 기전이 v0.4의 얕은 위치 위 끝 계단(공개 19, "기록 시작 경계 처리와 합성 창 잘림")과도 이어진다. 평탄 팔(판정 팔)에 다른 영향이 없는지, 곧 열 아래 끝의 0.91 %(합성 창 끝 잔여, §1-3)가 기록 안 행인지 기록 밖 행인지를 확인받는다. 기록 밖이면 영향이 없다. 기록 안이면 이미 R3·단계 C에도 있던 성질이라 이번 판정을 바꾸지는 않지만, 공개 후보다.
  3. §1-4의 정규화 성질과 [EMV] 5절 $coh$ 예측의 관계(앙상블 대 평탄 열 정규화).
- 질문 Q1에서 (a)를 고르면 EM 검토가 필요하다. 기준 열을 바꾸는 것은 응답 장의 물리 변경이기 때문이다.

## 7. 다음 순서 제안

1. 사용자 결정: §8 Q1–Q4.
2. (권고) EM 짧은 확인: §6의 1–3(결과 무관, 지시문에 블라인드 경로를 넣는다).
3. gpr-processing-expert: v0.5를 규칙 문서에 반영한다(§3 초안 + 결정 + EM 확인).
4. python-expert 구현:
   - `texture_check` v0.5 필드와 docstring.
   - precheck 10(`build_line` 영상 없는 경로, 건수 가드, `cpu_assertions.*`, GPU 전 실행, `started_at` 시작 시각).
   - 재시도 출력 경로(`e3x_retry/`, 판정 도구 입력 경로 포함, 따라서 `m33_e3x_judge.py` blob도 바뀔 수 있음).
   - `m33_e3x_seal2_r1.json`과 러너의 봉인 기록 경로, A-5 시험 4.
   - 원인 확인 열람 스크립트·stdout 보존(공개 23).
   - 시험은 실행 전 감사가 지적한 git Suspended 위험이 있는 offline 시험을 포함해 끝까지 돈 것을 확인한다.
5. integrity-checker(여러 파일 변경) → 봉인 2 r1 커밋.
   - plan 표와 `plan_report.json`은 그대로 두고 sha만 새 기록에 옮긴다.
   - 근거: 질감 코드는 plan 경로에 없다(`m33_e3x.py`의 질감 호출은 모두 2052행 이후 injected 쪽 함수, `run_plan`은 `_plan_worker`만 씀).
   - 봉인 기록에 `plan_code_blobs`(plan을 만든 옛 blob)와 `code_blobs`(새 blob)를 함께 적고, 바뀐 함수가 plan 경로에 없다는 diff 근거를 적는다.
   - verif가 더 강한 증거를 원하면 새 blob으로 plan을 scratch 폴더에 다시 돌려 표 sha가 같은지 본다(30–60분, 선택).
6. **stats 재평가는 필요 없다.** 근거:
   - R-c 질감 갈래는 보고 전용이다(§1-2, 사용자 결정 4).
   - MDE 평가 산출물에 질감·흔들림 항목이 없다(`analysis/processing/m33_e3x_mde_eval/` 검색 0건).
   - plan 표가 같으면 MDE 입력도 같다.
   - plan 재실행으로 표 sha가 바뀌는 경우에만 다시 부른다(그때는 멈추고 보고).
7. precheck r1(1–10) → 사본 → verif 실행 전 감사 재확인. 재확인 범위:
   - blob diff, plan 재사용 근거, precheck r1과 10번 기록.
   - 공개 22–26, 특히 23의 열람이 "결과 열람"이 아닌지에 대한 판정.
   - 재시도 경로의 블라인드.
8. (Q2에 따라) uninjected001 재시도 → injected 재시도(`e3x_retry/`, 메인 세션 러너, 15분 감시, 블라인드 경로 `e3x_retry*`) → 판정 도구 → 판정서(gpr-processing-expert) → verif 판정 후 감사 → integrity-checker → 커밋·push.

## 8. 확인 필요 질문 (사용자 결정)

- **Q1. 수정 방향**: (a) / (b1) / **(b2, 권고)** / (b3). (b2)는 실현 비트를 바꾸지 않고, 단언을 해석적 손실 식(1e-12)과 물리 한계($r_N\le10^{-6}$)로 바꾼다.
- **Q2. 1회차 uninjected001 출력**(끝났다면): (가) **재시도에서 다시 돌림(권고)**. 3–5분이고, 판정 입력의 실행 정체성이 봉인 2 r1 하나로 통일된다. (나) 1회차 출력을 그대로 씀. uninjected001 경로는 질감 코드를 부르지 않아 값은 같을 것이다. 그러나 정체성의 blob이 옛 값이라 판정 도구 입력 단언에 예외를 둬야 한다. 1회차 uninjected001이 실패했거나 돌지 않았다면 (가)만 가능하다.
- **Q3. 질감장 정규화의 실현 성질(§1-4)**: (가) **모형 유지 + v0.5 §2-7·공개 26에 실현 성질 공개(권고)**. 보고 전용 갈래이고, 평탄 열 정규화는 v0.2에서 [EMV] 5절 권고로 정한 규칙이다. (나) 정규화를 앙상블 표준편차(핵의 해석값)로 바꿔 테이퍼 외삽을 없앰. 실현과 precheck 5의 `std_flat` 단언이 바뀌므로 EM 검토와 새 문턱이 필요하다. (다) R-c 시간 흔들림 갈래를 빼고 진폭 질감만 남김.
- **Q4. EM 짧은 확인(§6 1–3)**: 봉인 2 r1 전에 받을지(권고: 받음, 비차단·결과 무관), 생략할지.
- **Q5. plan 재사용 증거**: diff 근거만으로 plan 표를 재사용할지(권고), 새 blob으로 plan을 다시 돌려 sha 일치를 확인할지(30–60분 CPU).

## 9. 결정과 EM 확인 반영 (2026-11-20, v0.5 반영 뒤 덧붙임)

### 9-1. 사용자 결정

| 질문 | 결정 | 규칙 문서 반영 |
|---|---|---|
| Q1 수정 방향 | **(b2)**: 해석적 손실 식(1e-12) + $r_N\le10^{-6}$, 실현 불변 | v0.5 v5-1, §2-7, §5-2 5, A-2, A-5 4 |
| Q2 1회차 uninjected001 | (가): `runs/eval/m33/e3x_retry/`에서 다시 돌림 | v5-3, §5-1, §5-4, A-4 |
| Q3 질감장 정규화 | (가): 모형 유지 + 공개 26 | v5-4, §2-7, §1-2 R-c, 공개 26 |
| Q4 EM 짧은 확인 | 받음: [EMC] [`m33_e3x_texture_em_check.md`](m33_e3x_texture_em_check.md). 봉인 2 r1을 막는 사항 없음, 비차단 문구 수정 §4 | v5-6, 아래 9-2 |
| Q5 plan 재사용 | diff 근거로 재사용(plan 다시 돌리지 않음) | v5-5, §0 머리, A-1 `m33_e3x_seal2_r1.json` |
| [EMC] Q-a | 공개 27(합성 창 절단의 기존 성질)을 넣음 | §2-4, 공개 27 |
| [EMC] Q-b | $r_N$ 한계 $10^{-6}$ 유지, 근거 문장은 [EMC] §4-2대로 | §2-7, 공개 25 |
| (사용자 결정, 이 검토 전) | precheck를 넓혀 CPU 구현 단언을 전 측선에서 GPU 전에 검사 | v5-2, §5-2 10 |

### 9-2. 이 문서에서 [EMC]가 고친 문장 (본문은 기록으로 두고, 효력은 v0.5 규칙 문서 문장)

| 이 문서 위치 | 원 문장(요지) | 고친 내용([EMC]) | v0.5 위치 |
|---|---|---|---|
| §1-1 "전제가 틀린 지점" | 열은 0.8 나이퀴스트 저역통과로 대역 제한. 누설은 "열 끝 값의 불연속과 창 밖 0 처리" | 대역 제한의 실제 근거는 Gabor 목표 파형이다(0.6 나이퀴스트 위 약 1e-100). 0.8 나이퀴스트 저역통과는 작동하지 않는 조건이다. 누설은 합성 창 [0, `window_ns`] 밖 0 처리로 생긴 **시작·끝 계단(주로 시작)**이고, 열이 기록 끝을 넘으면 열 끝의 비주기 경계다(교대합 오일러 경계항, 추정/실제 중앙 1.09) | §2-7 대역 제한·누설 |
| §2 근거 4의 $r_N$ 문장 | "−60 dB를 넘는 나이퀴스트 성분이면 모형 전제가 흔들린다" | 틀린 근거다. $r_N\approx(s/2)^2/(nE)$라 $10^{-6}$은 열 최대의 약 11–29 % 계단에 해당한다. 큰 절단·구현 결함만 잡고, 절단이 보이지 않는다는 보장은 하지 않는다. 값 $10^{-6}$은 유지한다 | §2-7 단언 문단, 공개 25 |
| §1-4 "[EMV] 5절 예측과의 관계" | "앙상블 정규화로 계산했다는 뜻이다", 약 1.3배 | 귀속 근거가 없다(정규화 기준 기록 없음. L+T 0.86, 앙상블 0.44 어느 쪽도 0.76을 재현하지 못함). 실효 흔들림 비는 **약 1.15–1.3배**다. coh 대리값은 σ_δ 0.2에서 약 0.7배 | §2-7 예측 문단, 공개 26 |
| §1-4 "물리 의미"와 §5 공개 26 초안 | "평탄 열 정규화 때문에 주로 굽음·기울기" | 인과가 틀렸다. 굽음·기울기는 **평탄 구간 길이**(약 3.6 $\ell_c$)에서 오고 정규화와 무관하다. 정규화가 만드는 것은 (i) 평탄 열 SD 고정(최대 약 12배 증폭), (ii) 외삽 $\lvert w\rvert$ 증가다. "굽거나 기운 띠" 해석을 R-c 문구에 붙인다 | §2-7 실현 성질, §1-2 R-c, 공개 26 |
| §1-3 "아래 끝 0.91 %"·§6 2 | 기록 안인지 밖인지 확인 필요 | 기록 **밖** 행(IDS `LID10042`·`LID10062`, 512행 이후)이다. 평탄 팔은 `unit_field[:n_samples]`로 버리므로 영향이 없다. 흔들림 팔에서는 원형 이동으로 위 끝 몇 행에 들어오고 크기는 0.75 % 이하다 | §2-7 크기 문단 |
| §5 공개 22 | (선택) | 끝에 "EM 검토도 그 문장에 근거 계산을 붙이지 않았다([EMC])"를 덧붙임 | 공개 22 |
| (새) | – | HB1·HB1b의 "같은 파형"은 정답 구간에 대한 말이다. 창 시작 계단(최대 1.8 % A, 위 끝 ≥ 2.08 P 위)과 창 끝 감쇠(아래 끝 ≥ 3.37 P 아래)가 기록 안에 드는지는 행마다 다르다 | §2-4, 공개 27 |

- 바뀌지 않은 것: (b2)의 식·두 문턱 값, 실현 비트, plan 표, 판정 가설·규칙. [EMC]도 "봉인 2 r1을 막는 사항 없음"으로 판단했다.
- 공개 22–27의 효력 문장은 v0.5 규칙 문서 §4에 있다. 이 문서 §5는 초안 기록이다.
- 채움: 규칙 문서 v0.5 공개 22(1회차 injected run_id)·23(열람 증거 경로·sha256). 이 문서 §3·§5의 `<run_id>`·`<경로>`는 초안 자리 표시이고 효력 문장은 규칙 문서다.
- 반영하지 않은 integrity 지적(봉인 2 r1 전 점검, 사용자 결정 2026-11-20): `m33_e3x_judge.py`·`m33_e3x_pool.py` docstring의 v0.3 표기는 그대로 둔다(고치면 봉인 blob을 다시 만들어야 하므로 r2 때 함께 고친다). [EMV]에는 정오표를 달지 않는다(규칙 문서 공개 22로 충분하다).

## 10. python-expert 핸드오프 (v0.5 구현 명세)

규칙 원문은 [`m33_e3x_plan.md`](m33_e3x_plan.md) v0.5다. 이 절은 그 요약이고, 어긋나면 규칙 문서가 우선한다.

공통 조건
- 블라인드: `runs/eval/m33/e3x/injected/`, `runs/eval/m33/e3x/uninjected001/`, `runs/eval/m33/e3x/judgment/`, `runs/eval/m33/e3x_retry*/`, `runs/eval/m33/e3x_precheck*/`의 기록 파일은 열지 않는다. 예외는 공개 23 증거 보존(H6)뿐이다.
- `src/`는 고치지 않는다. src tree id는 봉인 2 값 그대로(`d0e54933f3a0c87cf2b8d43d255e97bd6bb97122`)다.
- plan을 다시 돌리지 않는다(v5-5). `runs/eval/m33/e3x/plan/`의 표와 `plan_report.json`, `m33_e3x_pool.csv`는 바꾸지 않는다.
- 실현 비트를 바꾸지 않는다. `time_jitter`·`jittered_band`·`texture_field`·`amplitude_texture`의 계산은 그대로다.
- 실행 규칙: 셸 heredoc과 `python -`는 쓰지 않는다. `conda run -n gpr-dev python -I -B <파일>`로 돌리고, Python·pytest는 PowerShell 도구로 돌린다. git Suspended 위험이 있는 offline 시험은 끝까지 돌았는지 확인한다(실행 전 감사 기록 174–184행).

H1. 질감 단언 v0.5 (`analysis/detection/m33_e3x_place.py`; 규칙 §2-7, §5-2 5, A-2)
- `TextureCheck.jitter_energy`를 없앤다. 대신 두 필드를 둔다.
  - `jitter_energy_identity: dict[float, float]`: σ마다 $\max_x\lvert E'(x)/E-1+r_N\sin^2(\pi\delta(x))\rvert$, 모든 띠 열(테이퍼 포함). $E'(x)$는 `time_jitter` 출력 열 $x$의 제곱합, $E$는 `apex_column(band)`의 제곱합이다.
  - `jitter_nyquist_fraction: float`: $r_N=(\lvert X_{n/2}\rvert^2/n)/E$($n$ 짝수), 홀수면 0.0. $X$는 `np.fft.rfft(column)`이다.
- 통과 조건: 기존 네 필드는 그대로 두고, `jitter_energy_identity` ≤ `JITTER_ENERGY_IDENTITY_TOL = 1e-12`, `jitter_nyquist_fraction` ≤ `JITTER_NYQUIST_FRACTION_MAX = 1e-6`를 더한다(경계 포함).
- 기록 필드(`passes()`에 쓰지 않음):
  - `jitter_above_cutoff_fraction`: `rfftfreq(n) > 0.4` 빈의 파스발 가중 에너지 / $E$.
  - `jitter_wrap_rel[σ]`: 봉인 `time_jitter`와 4n 영 채움 선형 위상 회전(열을 [n, 2n)에 두고 같은 회전 후 잘라냄)의 차 최대 / $\max\lvert x\rvert$.
  - `jitter_start_step_rel`: 첫 0 아닌 행의 $\lvert x\rvert$ / $\max\lvert x\rvert$.
  - `jitter_delay_absmax_box_P[σ]`·`jitter_delay_absmax_taper_P[σ]`: `box_columns` 안 / `flat_columns` 밖 열의 $\max\lvert\delta\rvert/P_\text{line}$.
- 계산 참고: [TDR] `analysis/processing/m33_e3x_texture_defect/wrap_scan.py`. 기존 101위치 관측 최대는 항등식 잔차 7.5e-15, $r_N$ 1.03e-8, wrap 0.75 %다.
- docstring: `time_jitter`의 "the column is already low-passed to 0.8 Nyquist, so the band-limited shift is exact and its energy is kept"를 고친다. 새 내용은 "circular length-n DFT phase rotation; for even n the Nyquist bin keeps only its real part, so E'/E = 1 - r_N sin^2(pi delta); circular wrap of the end rows"다. `texture_check` docstring도 같이 고친다.
- `_Textures.of`가 저장하는 `checks[key]`는 bool 그대로다. 대신 H2의 `cpu_assertions.json`이 위치별 `TextureCheck` 값 전체(오차·기록 필드)를 받을 수 있게, 같은 객체를 함께 보관한다(예: `details[key]`).

H2. precheck 10 — 전 측선 CPU 구현 단언 (`analysis/detection/m33_e3x.py`; 규칙 §5-2 10과 문단)
- 범위: 본 표본 153측선 + precheck 2측선. 단언은 `build_line`의 다섯 키다.
  - `shift_mismatch`
  - `off_differs_from_on`
  - `r_edge_differs_from_plan`
  - `texture_check`
  - `zero_test`
- 같은 함수 조건: `build_line`에 영상·띠 실현을 건너뛰는 경로를 둔다(예: `build_line(plan, hyp_lib, band_lib, images=False)`).
  - 쌍곡선은 `hyperbola_entries` 안의 `inject_realization`·`inject_untapered`·`_off_matches_on`을 같은 순서로 부르되, `_hyperbola_arm_entries`의 영상 생성은 건너뛴다.
  - 띠는 `band_entries`가 `_band_realization` 안에서 `textured()`→`_Textures.of`를 부르는 것과 같은 순서로 `_Textures.of`만 부른다. 순서는 (row_px, arm) 정렬 → 위치 id 정렬 → `band_conditions` 순서다.
  - 영 시험은 `zero_test_passes` 그대로다.
  - 단언 계산부를 복사한 새 함수는 만들지 않는다.
- 건수 가드: 단언마다 `n_checked`가 0보다 크고 plan 표에서 센 `n_expected`와 같다.
  - `shift_mismatch`: plan `band_arms.csv`의 팔 수(precheck 측선은 precheck plan)
  - `off_differs_from_on`: 쌍곡선 실현 수 = (측선, 묶음) × SNR 4 × 극성 2
  - `r_edge_differs_from_plan`: 쌍곡선 위치 수
  - `texture_check`: 질감 조건이 있는 띠 위치 수(긴 팔이고 47·68 px 팔이 만들어진 위치)
  - `zero_test`: 측선 수
  - 기대 수 계산식은 코드와 시험에 같이 둔다.
- 실행 순서: 1번(정체성) → 10번 → 2–9번. 10번이 실패하거나 가드가 어긋나면 `precheck_failed`(종료 1)이고, 2–9번(GPU)은 부르지 않는다. 작업자 수는 `--workers N`을 따른다(측선 단위 병렬, 순서 무관한 집계).
- 출력(precheck 폴더 안):
  - `cpu_assertions.json`: `run_identity`(64자리), `started_at`(실제 시작 시각), 단언별 `n_checked`·`n_expected`·`n_failed`, 실패 목록(`line_id`, 위치 또는 실현 id, 단언, 오차값), H1 기록 필드의 분포(최소·중앙·p90·최대).
  - `cpu_assertions_lines.csv`: 측선별 단언 수·실패 수·시간(s).
  - `precheck.json`의 `checks`에 `10_cpu_assertions`(bool)를 더한다.
- stdout: 진행 수만 낸다. 10번 요약 줄은 `n_lines`, `n_checks`, `n_failed`, `elapsed_s`이고, 마지막 줄 키 집합은 §5-3 규칙을 따른다. 검출 값은 없다.
- 폴더: 새 precheck 폴더 `runs/eval/m33/e3x_precheck_r1/`(인자로 받거나 차수 자동). 있으면 거부한다(종료 2). 1회차 `e3x_precheck/`·`e3x_precheck_copy/`는 건드리지 않는다.
- 사본: 메인 세션이 실행 직후 `e3x_precheck_r1_copy/`로 폴더째 복사하고 두 폴더 파일별 sha256(LF) 목록을 남긴다. 러너 기능으로 넣어도 된다.
- 시간: 질감 부분 실측 약 4분(1 프로세스), 전체는 1 프로세스 약 10–15분(추정). 측선별 시간을 기록해 실측을 남긴다.

H3. 재시도 경로 (`m33_e3x.py`, `m33_e3x_judge.py`; 규칙 §5-1, §5-4, A-4)
- `--stage uninjected001`·`--stage injected`의 출력 루트는 `runs/eval/m33/e3x_retry/`(`uninjected001/`, `injected/`)다. 판정 도구 출력은 `e3x_retry/judgment/`다. 재시도 차수 k ≥ 2는 `e3x_retry<k>/`다.
- 1회차 경로(`runs/eval/m33/e3x/uninjected001/`, `injected/`, `judgment/`)에 쓰기를 거부한다(종료 2). 1회차 폴더는 읽지도 않는다.
- 판정 도구 입력 단언:
  - injected·uninjected001 `run_identity`의 봉인 기록 sha(`m33_e3x_seal2_r1.json`)와 코드 blob이 같아야 한다.
  - 입력이 1회차 경로이면 종료 2다.
  - 1회차 uninjected001 출력을 R-d 기준값으로 쓰지 않는다(v5-3).
- `--resume`은 같은 HEAD·blob·src tree·같은 재시도 폴더에서만 된다.

H4. 봉인 2 r1 기록 (`analysis/detection/m33_e3x_seal2_r1.json`; 규칙 A-1)
- 키는 `m33_e3x_seal2.json`과 같다(`schema`는 `m33_e3x_seal2/2` 등 새 값). `code_blobs`는 새 blob이다. `pool_csv_sha256_lf`·`plan_files_sha256_lf`·`plan_report_sha256_lf`·`plan_head`·`plan_run_id`·`seeds`·`src_tree_id`는 봉인 2 값 그대로다.
- 더하는 키:
  - `supersedes`: `{"path": "analysis/detection/m33_e3x_seal2.json", "sha256_lf": <옛 기록>, "commit": "729d19ced8674039efac0cfe906fcb41f45c31e3"}`
  - `plan_code_blobs`: 봉인 2의 `code_blobs` 그대로
  - `plan_reuse`: 바뀐 함수 목록과, 그 함수가 `--stage plan` 경로(`run_plan`·`_plan_worker`와 그 호출)에서 불리지 않는다는 근거(호출 위치 행 번호)
- 러너의 `SEAL2_RECORD`를 새 파일로 바꾼다. `assert_seal2`는 새 기록으로 같은 대조를 한다. 옛 `m33_e3x_seal2.json`은 고치지 않는다.
- 봉인 2 r1 커밋 전에 integrity-checker가 규칙 문서 ↔ 코드를 대조한다. 그 뒤 verif 실행 전 감사 재확인을 받는다(메인 세션이 부름).

H5. 시험 (`tests/test_m33_e3x.py`; 규칙 A-5 4)
- `time_jitter`
  - 정수 이동 = `np.roll`(1e-9)
  - $\delta=0$ = 원 열(1e-12)
  - 홀수 $n$ 에너지 보존(1e-12)
  - 짝수 $n$ 손 계산 fixture($n=64$, 저주파 펄스 + $a(-1)^k$)에서 $E'/E=1-r_N\sin^2(\pi\delta)$(1e-12), δ = 0.5에서 손실 = $r_N$
  - 시작 계단 fixture에서 $r_N$이 $(s/2)^2/(nE)$와 같은 자릿수
- `texture_check`
  - $r_N>10^{-6}$ fixture에서 실패, $\le10^{-6}$에서 통과
  - 나이퀴스트 빈을 0으로 만들고 비교하는 b1형 모의 구현에서 항등식 단언 실패
  - 기존 네 필드 시험 유지
- precheck 10
  - precheck 2측선에서 10번의 단언별 `n_failed`·`n_checked`가 `build_line`의 `defects`·검사 수와 같다
  - 검사 수 0인 fixture에서 가드 실패
  - 10번 실패 fixture에서 2–9번이 불리지 않는다
- 재시도 경로
  - 판정 도구가 1회차 경로를 받으면 종료 2, 정체성이 다른 injected·uninjected001 짝을 받으면 종료 2
  - 러너가 1회차 출력 경로에 쓰기를 거부
- 봉인 기록: `m33_e3x_seal2_r1.json`의 `supersedes` sha가 옛 기록 파일과 같고, plan 표 sha가 봉인 2와 같다.
- `tests/`에서 ultralytics를 import하지 않는다(기존 규칙).

H6. 공개 23 증거 보존
- 1회차 injected `summary.json`·`parts\*.json`에서 `defects` 키 이름(0이 아닌 것)과 `line_id`를 출력한 스크립트를 보존한다. 위치는 `C:\Users\jinbeom\.claude\runners\gpr_detection-m33_e3x\diag_texture_20261120\` 또는 저장소 밖 같은 폴더이고, 원본 그대로 둔다.
- 함께 보존할 것은 그 stdout 전문, 실행 시각, 스크립트와 stdout의 sha256이다. 지금 남아 있지 않으면 다시 돌리지 말고, 남은 것(명령 기록·보고 문장)과 "재구성 불가"를 그대로 보고한다. 다시 열면 열람이 늘어난다.
- 보고에 경로·sha256을 적는다. 메인 세션이 규칙 문서 공개 23의 `<…>` 자리를 채운다.
- `texture_scan.py`(sha256 `014d31b39e3509ad20b970320aa853d839d0e0e9b9e8426e67c25de05d378b1c`)와 `scan_all\` 산출물도 지우지 않는다. [TDR]·[EMC]가 import·인용한다.

완료 보고에 넣을 것
- 바뀐 파일과 blob(40자리)
- 시험 결과(전체 수, skip 0 확인)
- precheck 10의 dry 확인(precheck 2측선에서만, 별도 scratch 폴더, 본 precheck r1 폴더는 만들지 않음)의 단언별 검사 수
- `m33_e3x_seal2_r1.json` 초안
- H6 증거 경로·sha256
- 본 precheck r1과 재시도 실행은 메인 세션이 띄운다.