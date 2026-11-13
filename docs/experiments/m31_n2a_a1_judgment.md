# M3-1 N2-a — A1 판정서 (AIHub 원 crop 성능 유지)

- 작성: 2026-11-13, cv-detection-expert(판정자). 규칙은 [`m31_n2a_retrain_plan.md`](m31_n2a_retrain_plan.md) v0.3.3(이하 [P]) §3-1이고, **봉인 커밋은 651cc10**(2026-11-13T19:38:06+09:00)이다.
- 다음 단계: 이 판정서를 커밋한 뒤 감사 ②(판정 뒤, verif-protocol-auditor)를 받는다. 그다음 D1 r3 봉인 부록(§6)을 커밋하고 감사 ③을 받는다([P] §6).

## 1. 판정

**A1 충족.** 게이트를 모두 통과했고, 두 문턱을 모두 넘었다.

| 지표 (crop, `last.pt`, 5 fold 비가중 평균, %) | N2-a fold 0 / 1 / 2 / 3 / 4 | 평균 ± SD (ddof=1) | 문턱 | 결과 |
|---|---|---|---|---|
| 전체 mAP50 | 82.30 / 76.92 / 77.79 / 83.60 / 82.94 | **80.71** ± 3.11 (정확값 80.710101496812938) | ≥ 76.8 | 통과 |
| pipe_hyperbola AP50 | 89.51 / 89.06 / 96.17 / 94.44 / 92.12 | **92.26** ± 3.08 (정확값 92.260584665038436) | ≥ 90.9 | 통과 |

- 판정 파일 `analysis/detection/n2a_a1_verdict.json`의 값은 `verdict` "A1 met", `gates_pass` true, `failures` 비어 있음, `a1.pass` true다.
- **판정 스크립트 종료 코드 0의 뜻**: `n2a_a1_judge.py` docstring의 정의는 "0 judged (A1 met or not), 1 gate failed, 2 input missing or …"다. 그래서 종료 0은 "게이트를 통과해 판정을 계산했다"는 뜻이고, 문턱 통과 여부는 따로 `verdict`·`a1.pass`로 본다. 이번 판정은 둘 다 통과다.
- 평균은 `Fraction`으로 정확히 계산했고, 문턱 비교에 반올림을 쓰지 않았다.
- **M1-10 대비**(보고용, fold별 짝지은 차, %p)
  - 전체 mAP50: −0.28 / +0.48 / +0.74 / +0.71 / +2.99, 평균 **+0.93**(M1-10 79.78 → 80.71).
  - pipe_hyperbola AP50: −6.99 / −2.53 / −0.74 / −0.30 / +2.11, 평균 **−1.69**(93.95 → 92.26).
  - 쌍곡선 AP50은 fold 0에서 크게 떨어졌다(96.50 → 89.51). 다만 평균은 U-3 목표 90.9 위에 있다.

## 2. 실행·정체성 기록 (감사 ① 권고 항목 포함)

| 항목 | 값 |
|---|---|
| 봉인 | `--seal-commit 651cc10`. 설정 `configs/train_yolo11m_n2a.yaml` sha256 7d5c6112…163d. 봉인 blob 내용과 일치(`config_file_matches_seal` true) |
| 학습 정체성 (`gpr_train_settings.yaml`) | 스모크 fold 0과 5 fold 모두 `commit` 651cc10e6b44db762e7febbc00a5ba08b3be99fc, `dirty` false, `config_sha256` 7d5c6112…163d, `seed` 0, `augment_seed` 0 |
| 러너 머리 점검 (`runs/n2a_runner_logs/headcheck.txt`) | fold 0–4 학습 직전(19:50, 20:20, 20:51, 21:22, 21:51) 모두 HEAD = 봉인 651cc10, 추적 파일 변경 0 |
| 학습 wall (s) | 1,558 / 1,519 / 1,587 / 1,489 / 1,670. 스모크 118 |
| 평가 wall (s) | 249 / 264 / 277 / 222 / 287. 내보내기 22–30 |
| 가중치 `last.pt` sha256 | fold 0 235ec15e…c338, 1 3907621c…e5, 2 8ae8cc60…5a94, 3 9d3f3b52…1304, 4 99f3bf95…9979. 판정 때 다시 해시한 값 = `train_record.json` = 평가 기록(`weights_match_train_record`·`weights_match_eval_record` true) |
| 평가 목록 | 5 fold 모두 M1-10 `metrics.json`의 `eval_images.sha256` 값과 같다(5350a569 / 4ef169c9 / 69a61b9a / 3f917aa7 / 06c10d22) |
| 동일성 4개 | `evaluated_last_pt`·`evaluated_crop_unit`·`evaluated_fold_k`·`trained_fold_k` 모두 true |
| 학습 인자 (`args.yaml`) | 5 fold 모두 `scale` 0.0, `translate` 0.1, `flipud` 0.0, `epochs` 50, `seed` 0 |
| env (gpr-dev) | python 3.13.18, numpy 2.5.4, opencv 4.14.0, torch 2.14.1+cu130(CUDA 13.0), ultralytics 8.4.192, gpr_detection 0.1.0 |
| 평가 설정 | 5 fold `metrics.json.settings` = imgsz 256, batch 64, conf 0.001, confusion_conf 0.25, iou 0.7, rect true. M1-10과 다른 것은 `device` 한 키뿐이다(N2-a "0", M1-10 null. null이면 Ultralytics가 GPU 0을 고르므로 같은 장치). 판정 지표와 관계된 설정은 같다 |
| 판정 실행 | `n2a_a1_judge.py`, run_id dbe593fd…0026, 2026-11-13T22:24:34, CPU, git HEAD d26a5fa(dirty false), argv `--config configs/train_yolo11m_n2a.yaml --run-prefix yolo11m_aihubv1n2a_20261113_fold --seal-commit 651cc10 --out analysis/detection/n2a_a1_verdict.json --blind` |

### 2-1. 공개 사항

- **실행 중 커밋 1건**: d26a5fa(2026-11-13T21:53:06, "M3-1 QA-8 실측 왕복 판정 합격")가 fold 4 학습 시작(머리 점검 21:51:26) 2분 뒤에 들어갔다.
  - 변경 파일은 `analysis/processing/m31_qa8_outward_check.json`·`.py`, `docs/processing/m31_qa8_roundtrip.md` 3개뿐이다. 검출 학습·평가·내보내기·판정 코드와 설정은 바뀌지 않았다(`git diff --stat 651cc10 d26a5fa`).
  - 그 결과 fold 4 학습 정체성은 651cc10이고, fold 4 평가·내보내기와 A1 판정 실행은 HEAD d26a5fa에서 돌았다. fold 4 `metrics.json.git.commit`이 d26a5fa로 기록돼 있다.
  - 판정 코드 경로가 같으므로 결과에는 영향이 없다고 본다. 사실 그대로 공개한다.
- **CPU 공유**: 21시 무렵 gpr-processing-expert의 QA-8 검증 스크립트가 학습 예약 범위 안의 P코어 1개(마스크 0x4)를 BelowNormal·BLAS 1스레드로 잠시 썼다. 학습은 deterministic이고 DataLoader 시드도 고정이라, 영향은 시간뿐이다(fold 3·4 wall). 재시작이나 `--on-existing archive`는 없었다.
- **블라인드**
  - 메인 세션은 종료 줄과 로그의 금지어 줄 수(지표 관련 0줄)만 확인했고, 지표·verdict를 열지 않았다(메인 세션 보고).
  - 판정자는 A1 판정과 이 판정서의 보고 항목을 위해 `n2a_a1_verdict.json`, 5 fold `eval/{last,best}_crop/metrics.json`, 혼동 행렬 값, `aug_stats.json`, 띠 보조 지표를 열었다.
  - D1 r3 판정 입력(변형 재현율)은 아직 존재하지 않는다.

## 3. 보고 항목 (판정 조건 아님, [P] §3-1)

### 3-1. `report_only`와 클래스별 (crop, `last.pt`, %)

| 지표 | fold 0 / 1 / 2 / 3 / 4 | N2-a 평균 | M1-10 평균 |
|---|---|---|---|
| pipe_band AP50 | 57.59 / 39.38 / 35.27 / 64.59 / 62.45 | 51.86 | 50.45 |
| cavity AP50 | 89.90 / 90.33 / 89.59 / 85.28 / 89.33 | 88.89 | 86.34 |
| manhole AP50 | 92.21 / 88.90 / 90.13 / 90.09 / 87.84 | 89.84 | 88.39 |
| mAP50-95 | 39.72 / 35.21 / 37.32 / 39.72 / 37.13 | 37.82 | 37.30 |

### 3-2. pipe_band 띠 보조 지표 (crop, conf > 0.25, 길이 가중, M1-10 §6-1과 같은 정의)

계산: `analysis/detection/n2a_band_aux.py`(새 파일, `m110_band_aux.py`를 고치지 않고 run 이름·출력 경로만 바꿔 실행) → `analysis/detection/n2a_band_aux.json`·`.csv`. 내부 검산 두 가지가 모두 통과했다(예측으로 다시 만든 conf 0.25 혼동 행렬 = 평가 csv, 다시 계산한 AP50 = `metrics.json`). 종료 0.

| 지표 | fold 0 / 1 / 2 / 3 / 4 | 평균 ± SD | 평균 − 1 SD | M1-10 평균 ± SD | U-3 띠 목표(test) |
|---|---|---|---|---|---|
| 가로 구간 재현율 | 72.70 / 67.12 / 73.18 / 58.25 / 75.63 | 69.38 ± 6.96 | 62.42 | 69.52 ± 5.43 | ≥ 64.1 |
| 가로 구간 정밀도 | 86.92 / 51.51 / 51.81 / 90.21 / 95.02 | 75.10 ± 21.58 | 53.51 | 71.93 ± 22.27 | ≥ 49.7 |

- `n2a_band_aux.json`의 `definition.weights` 문자열은 원 스크립트의 고정 문구다. 실제 가중치는 N2-a `last.pt`다(감사 ② 기록).

### 3-3. conf 0.25 혼동 행렬 (crop, `last.pt`, 매칭 IoU 0.45, 5 fold 합)

칸: 정검출 / 미검출(→ 배경) / 배경 → 해당 클래스 오검출 (오검출 ÷ 정답)

| 클래스 | N2-a | M1-10 |
|---|---|---|
| pipe_hyperbola | 473 / 30 / 91 (0.18) | 478 / 25 / 68 (0.14) |
| pipe_band | 1,328 / 650 / 1,200 (0.61) | 1,309 / 669 / 1,284 (0.65) |
| cavity | 1,644 / 183 / 260 (0.14) | 1,627 / 199 / 307 (0.17) |
| manhole | 1,980 / 153 / 454 (0.21) | 1,932 / 199 / 406 (0.19) |

- 클래스 사이 혼동은 manhole 정답을 cavity로 낸 것 3개뿐이다(fold 0·3·4 각 1). 쌍곡선 ↔ 띠 혼동은 0이다.
- 쌍곡선 오검출이 68 → 91로 늘었다(fold 3에서 41개). 위치·축척 증강으로 화면 가장자리·작은 크기의 쌍곡선 꼴 무늬에도 반응하게 된 것으로 보인다(추정, 사례 미확인).

### 3-4. `best.pt` (crop, %, 낙관 편향, 참고만)

| 지표 | fold 0 / 1 / 2 / 3 / 4 | 평균 |
|---|---|---|
| 전체 mAP50 | 82.39 / 79.63 / 78.74 / 83.36 / 85.28 | 81.88 |
| pipe_hyperbola AP50 | 88.72 / 92.55 / 97.98 / 96.00 / 93.45 | 93.74 |
| pipe_band AP50 | 58.38 / 43.73 / 36.86 / 61.71 / 69.83 | 54.10 |
| cavity AP50 | 89.18 / 91.99 / 88.79 / 84.10 / 90.93 | 89.00 |
| manhole AP50 | 93.27 / 90.24 / 91.34 / 91.65 / 86.91 | 90.68 |

`best.pt`는 M1-10과 마찬가지로 `last.pt`보다 높다(mAP50 +1.17 %p). 다만 M1-10의 차(+3.26)보다 작다. 증강이 후반 과적합을 줄인 것으로 보인다(추정).

## 4. 증강 적용 통계 (`aug_stats.json`, 보고용)

5 fold 모두 `epochs_completed` 50이고, 설정 블록은 [P] §5-4와 같다. 영상 수(축척 단계를 거친 학습 영상, 50 epoch 합)는 fold 0–4가 564,190 / 579,894 / 595,386 / 615,976 / 606,698이다.

| 항목 | fold 0 / 1 / 2 / 3 / 4 | 설정값 |
|---|---|---|
| 가로 이동 적용률 | 0.5001 / 0.5003 / 0.5005 / 0.5005 / 0.5006 | 0.5 |
| 채움 띠 적용률 | 0.3001 / 0.3002 / 0.3002 / 0.3001 / 0.3002 | 0.3 |
| 채움 띠 위치(fold 0, 좌 / 우 / 아래) | 0.350 / 0.350 / 0.300(59,226 / 59,325 / 50,739) | 0.35 / 0.35 / 0.30 |
| 축척 분포 | fold 0 10칸 히스토그램 56,144–56,618(균등) | U[0.55, 1.5] |

단계별 라벨 버림 비율(버림 ÷ (남김 + 버림), %, fold 0–4 범위)

| 단계 | pipe_hyperbola | pipe_band | cavity | manhole |
|---|---|---|---|---|
| 축척 | 0 | 1.3–4.1 | 0–0.07 | 1.07–1.38 |
| 가로 이동 | 0.04–0.11 | 3.9–7.8 | 0.18–0.23 | 2.35–2.69 |
| 채움 띠 | 4.5–4.8 | 5.5–7.1 | 4.7–4.8 | 5.7–5.8 |

- 쌍곡선 라벨은 축척에서 하나도 버려지지 않았다(§5-7 (다) 규칙 효과). 이동에서는 0.1 % 이하만 버려졌다. [P] §1-3의 "잘림 5–10 %" 추정은 "잘렸지만 50 % 이상 남아 라벨이 유지된 것"이 대부분이었다는 뜻이다. 잘린 박스 수 자체는 이 통계에 없다.
- worker 값은 64장 미만만큼 늦게 반영될 수 있다([P] §5-0 #3). 비율에 주는 영향은 무시할 수 있다.
- fold 0·3 학습 중 증강 통계 쓰기 경고(`PermissionError`)가 2건 있었다(epoch 44·39). 누적 원자 쓰기라 손실은 없다(감사 ② 기록).

## 5. 다음 단계

A1 충족이므로 [P] §6의 7–9를 진행한다.
1. 이 판정서와 `n2a_a1_verdict.json`·`n2a_band_aux.*`·`n2a_band_aux.py`를 커밋한다.
2. 감사 ②(판정 뒤)를 받는다.
3. 아래 §6 D1 r3 봉인 부록을 [P]에 붙여 커밋한다.
4. 감사 ③(실행 전)을 받는다.
5. D1 r3를 실행한다(블라인드).

## 6. D1 r3 봉인 부록 (준비본, [P] §4-3으로 옮겨 봉인)

| 항목 | 값 |
|---|---|
| 규칙 | [T] `m31_detection_input_tiling.md` v0.5.2 §8-2(봉인 3c81aca, 규칙 blob eacf9126376c199d9486bb8a98b13fa2b411d7bd) 그대로 |
| 가중치 | `runs/detect/yolo11m_aihubv1n2a_20261113_fold<k>/weights/last.pt` |
| 기대 sha8(fold 0–4) | **235ec15e,3907621c,8ae8cc60,9d3f3b52,99f3bf95** |
| 전체 sha256 | 235ec15e344dc9a314d9e444126c92e0aa9c0eecb8091157cec90e8a9b3ce338 / 3907621cf57048c61dbd23e66f7740582b573a8985cbb945d352313b5bbed3e5 / 8ae8cc60679db04ed576dee351d00cd898c8da682608be9522839822d1e05a94 / 9d3f3b52d67c588ed76d35a69cadbcc19940d6b6c6ee9c36736e595c51e61304 / 99f3bf95602a39aea7e4300bc1d9888a1585761f85647d4d8aa06bfc511c9979 |
| run-prefix | `yolo11m_aihubv1n2a_20261113_fold` |
| export-root | `runs/analysis/n2a_val_export`(`m31_d1_shift_scale.py`에 `--export-root` 있음, 438행 확인) |
| G1 입력 sha256 (predictions.csv / images.csv / val_metrics.json) | fold 0: 85b186c2…f2b5 / 1e13f9e5…c3fb / 77f194c1…3f3e. fold 1: a6441b14…6018 / 6de0bb2b…60c7 / 645d7929…94e7. fold 2: 16f39074…054e / cd2f1739…ad1a / c41a1125…d2b8. fold 3: 49e47921…b0f0 / 8636da99…9ad6 / 6fa0f0a6…166b. fold 4: 071329d3…a11a / da1cd9f5…0d07 / 8f0b3ddb…480e (전체 값은 `Get-FileHash`로 다시 계산 가능, 봉인 부록에는 전체 64자를 적는다) |
| G3 기대 | r2와 같다(n_gt 4클래스, `gp6_dx±64` 쌍곡선 결손 0/0, 0/0, 0/3, 3/3, 2/0) |
| 출력 | `eval/m31_d1_r3/`, `analysis/detection/m31_d1_r3_summary.json`, `analysis/detection/m31_d1_r3_verdict.json` |

명령([P] §4-1)

```powershell
$py = "C:\ProgramData\miniforge3\Scripts\conda.exe"
& $py run -n gpr-dev --no-capture-output python -I analysis/detection/m31_d1_shift_scale.py --run-prefix yolo11m_aihubv1n2a_20261113_fold --export-root runs/analysis/n2a_val_export --folds 0,1,2,3,4 --device 0 --fill 128 --save-examples --out-subdir m31_d1_r3 --summary analysis/detection/m31_d1_r3_summary.json
& $py run -n gpr-dev --no-capture-output python -I analysis/detection/m31_d1_judge.py --run-prefix yolo11m_aihubv1n2a_20261113_fold --export-root runs/analysis/n2a_val_export --expected-weights-sha8 235ec15e,3907621c,8ae8cc60,9d3f3b52,99f3bf95 --subdir m31_d1_r3 --out analysis/detection/m31_d1_r3_verdict.json
```

- 판정 스크립트의 G1(M1-10 기준 대신 N2-a 검증기 검출을 다시 채점한 값 = N2-a `eval/last_crop/metrics.json`)이 새 run에서 맞게 도는지는 감사 ③에서 확인한다. 예상 시간은 약 24분, VRAM은 2.5 GB 이하다(r2 실측 1,430 s).

## 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-11-13 | 최초 작성. A1 충족(mAP50 80.71, pipe_hyperbola AP50 92.26). 정체성·공개(실행 중 커밋 d26a5fa, CPU 공유), 보고 항목, 증강 통계, D1 r3 봉인 부록 준비본 |
| 2026-11-13 | 감사 ② 반영: fold 2 val_metrics 해시 끝자리 오기(b2b8 → d2b8), manhole 평균 89.83 → 89.84, §3-2 `definition.weights` 문구 주석, §4 `PermissionError` 2건 기록. 판정 불변 |
