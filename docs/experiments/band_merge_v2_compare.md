# pipe_band 자동 병합 라벨 v2 vs v1 비교 — 실험 설계 (2026-10-31)

상태: **설계·러너 준비 완료, 미실행**(실행 시점은 사용자가 나중에 지시한다. 그 뒤 메인 세션이 러너를 띄운다). M1-10은 보류 중이다.

## 1. 목적과 사용자 결정

- 질문: pipe_band 박스를 간격 ≤ 32 px로 자동 병합한 라벨(v2, `data/aihub_yolo_v2_auto32/`)로 학습하면 v1 라벨 학습보다 나은가.
- 사용자 결정(2026-10-31)
  1. 수동 재라벨은 이 비교 결과를 본 뒤 결정한다.
  2. 학습: v2를 seed 0·1로 두 번 학습한다. v1은 M1-7 `runs/detect/m17_aug0`(seed 0)·`m17_aug0_s1`(seed 1)을 재사용한다.
  3. 판정 정답(개정): **v1 정답과 v2 정답 양쪽을 모두 만족**해야 통과다. 앞서 정한 "v2 정답 기준, v1은 참고"를 대체한다. 근거는 §5-1의 편향(v1 seed 0 모델이 정답만 바뀌어도 band AP50 59.09 → 22.52 %)이다. 네 실행 모두 두 정답으로 평가한다. 정답별 조건은 §6-1.
  4. 판정 지표: 주 판정은 pipe_band AP50과 관로 평균 AP50, 전체 mAP50은 하락 가드다(eval_detector 산출 AP50). band 보조 지표는 판정에 넣지 않고 보고한다.
  5. fold: fold 0을 유지한다(학습 라벨 변화가 작다는 주의는 §6-3 해석에 둔다).
  6. 실행: 사용자가 나중에 지시한다. 가로 구간 겹침 지표는 실행 뒤 가중치로 사후 계산하는 보조 지표다(§6-4).
  7. 후속 행동: 통과·미통과 모두 **결과를 보고 사용자가 다시 결정**한다. §7의 행동은 후보일 뿐 확정이 아니다.

## 2. 데이터와 라벨 변경량

v2는 v1과 이미지·매니페스트 행·분할(fold/test)이 같고 pipe_band 줄만 다르다(curator `summary.json`·`leak_checks.json`, 아래 §3 재단언).

| 부분 | crop 단위 band 박스 v1 → v2 | 이미지 단위 band 박스 v1 → v2 | crop 변경 수 |
|---|---|---|---|
| fold0(검증) | 966 → 631 (−34.7 %) | 2,649 → 1,726 | 250 / 1,298 |
| fold1–4(fold0 학습 집합) | 1,012 → 863 (−14.7 %) | 2,748 → 2,342 | 129 / 4,175 |
| test(봉인) | 301 → 202 | 796 → 535 | 84 / 770 |

- **주의 1 — 학습 쪽 변경이 작다.** 병합은 fold0에 몰려 있다(crop 병합 583쌍 중 335쌍). fold 0 검증 실험에서 v2 학습 집합은 band 박스가 15 %만 줄었다. 학습 라벨 차이로 생기는 효과는 작게 나올 수 있고, 다른 fold(fold0이 학습 집합에 들어가는 경우)에서는 더 크게 나올 수 있다.
- test 정답은 v2 폴더 `labels/`에서 이미 병합되어 있고, v1 test 정답은 `test_v1/`에 보존되어 있다. 이 실험은 test를 쓰지 않는다(봉인 유지, `eval_detector.py`는 `--allow-test` 없이 test를 거부한다).

## 3. 누수 점검 — 통과 (2026-10-31, 학습 전)

curator `leak_checks.json`(전부 passed)과 별도로 독립 단언을 다시 돌렸다(읽기 전용 즉석 스크립트, gpr-dev).

| 항목 | 결과 |
|---|---|
| v1·v2 매니페스트 행(file_name, fold, session, crop_key, split_group) | 16,887행 동일 |
| session·crop_key·split_group 값이 두 부분(fold0–4, test)에 걸침 | 0건 |
| v2 fold·test 목록 = v1 목록 = 매니페스트 | 6개 부분 모두 동일 (fold0 3,525 / fold1 3,210 / fold2 2,901 / fold3 2,490 / fold4 2,675 / test 2,086장) |
| fold0.yaml 학습(11,276장)과 검증(3,525장) 공유 | 0장, test 미포함 |
| pipe_band 외 클래스 라벨 v1↔v2 차이 | 0파일 |

gain 사본(같은 crop의 0.25/0.5/1.0)은 crop_key 단위로 한 fold에만 있다. 판정 단위는 crop(gain 사본 1개, 1.0 > 0.5 > 0.25)이다.

## 4. 학습 조건 (v1과 다른 것은 데이터뿐)

| 항목 | v1 (재사용) | v2 (신규) |
|---|---|---|
| 실행 폴더 | `runs/detect/m17_aug0`, `m17_aug0_s1` | `runs/detect/yolo11m_bandv2_aug0_20261031_fold0_s0`, `..._s1` |
| 설정 파일 | `configs/train_yolo11m_base.yaml` | `configs/train_yolo11m_band_v2.yaml`(base 사본, `dataset_dir`만 `data/aihub_yolo_v2_auto32`) |
| 원본 data yaml | `data/aihub_yolo_v1/fold0.yaml` | `data/aihub_yolo_v2_auto32/fold0.yaml` |
| 모델·사전학습 | YOLO11m(`yolo11m.yaml`), 사전학습 없음 | 같음 |
| fold·epoch·imgsz·batch·workers | 0 · 30 · 256 · 64 · 2 | 같음 |
| GPR 증강 강도 | 0 | 0 |
| Ultralytics 증강 | fliplr 0.5, **flipud 0.0**, degrees 0, translate 0.1, scale 0.2, mosaic 0 | 같음 |
| 시드 | 0, 1 (deterministic true) | 0, 1 |
| CPU 마스크·우선순위 | 0xFF · Normal | 같음 |

- `train_detector.py --dry-run`으로 v1 seed 1과 v2 seed 1의 해석된 설정을 비교했다. 차이는 `config_file`, `source_data_yaml`, run 이름(`data`·`name`)뿐이다.
- `--data`가 아니라 설정 파일의 `dataset_dir`로 v2를 고른 이유: `--data`를 쓰면 `gpr_train_settings.yaml`의 fold가 `custom`이 되어 평가 때 fold를 따로 줘야 하고, 기존 `scripts/ops/run_m17_sweep.ps1`(`-Config` 지원, `--data` 미지원)을 그대로 못 쓴다. 결과로 들어가는 data yaml은 같은 `data/aihub_yolo_v2_auto32/fold0.yaml`이다.
- Ultralytics 읽기 smoke(학습 없음, 2026-10-31): v2 fold0을 `channels: 1`로 읽어 학습 11,276장(band 2,342), 검증 3,525장(band 1,726), 빈 라벨·손상 0, 표본 형상 [1, 256, 256], 캐시 `data/aihub_yolo_v2_auto32/labels.cache` 생성. 수치가 §3 독립 집계와 같다.

## 5. 평가 행렬

네 실행 × 두 정답 = 8회, 모두 `last.pt`, crop·image 단위, `--confusion-conf 0.25`, fold 0, imgsz 256, conf 0.001, NMS IoU 0.7.

| | v1 정답(`data/aihub_yolo_v1`, 매니페스트 v1) — **판정(비열세)** | v2 정답(`data/aihub_yolo_v2_auto32`, 매니페스트 v2) — **판정(우세)** |
|---|---|---|
| v1 학습 s0·s1 | 기준선. 기존 M1-7 수치와 같아야 함(재현 확인) | 기준선 |
| v2 학습 s0·s1 | 판정 대상 | 판정 대상 |

- `eval_detector.py`는 항상 `<run>/eval/<weights>_<unit>/`에 쓰므로 같은 실행을 두 정답으로 평가하면 덮어쓴다. 그래서 `analysis/detection/band_v2_eval_view.ps1`이 정답별 "보기 폴더" `runs/detect/band_v2_eval/<run>__gt_<v1|v2>/`(설정 yaml 사본 + `weights/last.pt` 하드링크)를 만들고, 그 폴더에 `--fold 0 --dataset-dir --manifest`를 명시해 평가한다. 원래 실행의 `eval/`(M1-7 수치)은 건드리지 않는다.
- 평가 smoke(2026-10-31, 스크래치 폴더): v1 seed 0 `last.pt`를 v2 정답 crop 단위로 평가 — 1,298장, 68 s.

### 5-1. 정답 기준이 바뀌면 수치를 직접 비교할 수 없다

- 같은 v1 seed 0 가중치인데 정답만 바꾸면 crop 단위 전체 mAP50 84.35 → 75.20, pipe_band AP50 59.09 → 22.52(P 0.349, R 0.401, GT 966 → 631), 관로 평균 78.10 → 59.79 %가 된다(위 smoke). 모델은 같고 박스 규약만 달라졌다.
- 따라서 (a) v1 정답 수치와 v2 정답 수치를 서로 비교하지 않는다. (b) 비교는 항상 한 정답 열 안에서 v1 학습과 v2 학습 사이에서만 한다.
- **(c) 각 정답은 그 정답으로 학습한 쪽으로 기운다.** v1 학습 모델은 쪼개진 박스를 내도록 배웠으므로 병합 박스와 IoU 0.5를 못 넘는 일이 구조적으로 생긴다. 반대로 v1 정답에서는 병합 박스를 내는 v2 학습 모델이 같은 이유로 불리하다. 한 열에서 이기는 것만으로는 "검출이 좋아졌다"와 "박스 규약을 따라 배웠다"를 가를 수 없다. 그래서 판정은 두 열을 함께 본다(§6-1). 규약에 무관한 확인은 §6-4의 가로 구간 지표(사후 보조)로 한다.

## 6. 지표와 판정 규칙 (사전 등록)

### 6-1. 판정 지표와 판정 규칙 — v1·v2 두 정답, crop 단위, `last.pt`

| 지표 | 정의 | 판정 역할 |
|---|---|---|
| pipe_band AP50 | eval_detector `per_class` | **주 판정** |
| 관로 평균 AP50 | (pipe_hyperbola + pipe_band)/2, `pipe_ap50_mean` | **주 판정** |
| 전체 mAP50 | 4클래스 평균 | 하락 가드 |
| pipe_hyperbola·cavity·manhole AP50, mAP50-95 | | 보고만 |

차이(정답별로 따로 계산, %p)
- v2 정답: Δ = mean(v2 s0, v2 s1) − mean(v1 s0, v1 s1)
- v1 정답: Δ' = 같은 식을 v1 정답 평가값으로 계산

시드 변동 임계 τ(지표별, 정답별) = max(M1-7 v1 시드 차, 이번 v1 s0·s1 차(그 정답), 이번 v2 s0·s1 차(그 정답)). M1-7 v1 시드 차(crop `last.pt`, v1 정답, §8-4 재계산)는 전체 mAP50 1.97, pipe_band 2.29, 관로 평균 4.26 %p이고, 두 정답 모두에서 하한으로 쓴다. 따라서 τ_map ≥ 1.97, τ_band ≥ 2.29, τ_pipe ≥ 4.26이다. v1 정답의 τ'는 같은 식에 v1 정답 값을 넣는다.

정답별 조건(사용자 결정 2026-10-31 개정: 두 정답 모두 만족해야 통과)

| 정답 | 조건 이름 | 조건 |
|---|---|---|
| v2 정답 | **우세(U)** | Δ_band > τ_band, Δ_pipe > τ_pipe, v2 두 시드의 band AP50이 모두 v1 두 시드보다 높음(범위 비겹침), Δ_map ≥ −τ_map |
| v1 정답 | **비열세(N)** | Δ'_band ≥ −τ'_band, Δ'_pipe ≥ −τ'_pipe, Δ'_map ≥ −τ'_map |

**v1 정답 조건은 "나빠지지 않음(Δ' ≥ −τ')"으로 정한다(권고안 채택).** 이유는 다음과 같다.
- v1 정답에서도 Δ' > τ'를 요구하면 같은 편향 때문에 사실상 통과할 수 없다. v1 정답에서는 v2 학습 모델이 병합 박스를 내는 만큼 쪼개진 v1 박스와 IoU 0.5를 못 넘는다(§5-1 (c)의 반대 방향).
- 그렇게 되면 이 실험은 "v2가 v1보다 낫다"가 아니라 "v2가 v1 규약까지 더 잘 따른다"를 묻게 된다.
- 비열세 조건이 막아야 할 것은 규약 교환이다. 규약 교환은 각 모델이 자기 정답에서만 이기는 경우이고, 그때 v2 학습은 v1 정답에서 τ'를 넘게 떨어진다.
- 대안(더 엄격): v1 정답에서도 Δ'_band > τ'_band를 요구한다. 이 대안은 채택하지 않았고, 결과 표에는 Δ'가 τ'를 넘었는지 함께 적어 둔다.

분류. 위에서부터 차례로 적용하고, 처음 맞는 칸이 결과다.

| 순서 | 결과 | 조건 |
|---|---|---|
| 1 | **열세** | v2 정답 Δ_band < −τ_band, 또는 v2 정답 Δ_map < −τ_map(mAP 가드 위반은 항상 열세), 또는 (v2 정답 Δ_band ≤ τ_band 이면서 N 불충족 — v1 정답의 band·관로·mAP 중 하나라도 τ' 넘게 하락) |
| 2 | **통과** | U 그리고 N |
| 3 | **부분 통과-규약 교환 의심** | v2 정답 Δ_band > τ_band이지만 N 불충족(v1 정답에서 τ' 넘게 하락) |
| 4 | **부분 통과-약함** | v2 정답 Δ_band > τ_band, N 충족, U의 나머지(Δ_pipe > τ_pipe, 범위 비겹침) 중 하나 이상 불충족(mAP 가드 위반은 순서 1에서 이미 열세) |
| 5 | **동등** | 나머지(v2 정답 \|Δ_band\| ≤ τ_band, Δ_map ≥ −τ_map, N 충족) |

> 2026-10-31 실행 전 정정(integrity-checker 지적): 순서 1·4의 mAP 가드 중복 귀속을 없애고, "v2 정답 band 변화 없음 + v1 정답에서 관로·mAP만 하락" 조합이 어느 칸에도 들지 않던 빈틈을 순서 1(열세)로 채웠다. 결과를 보기 전의 정정이다.

시드 2개라 표준편차 대신 두 값의 차를 변동폭으로 쓴다. 이 규칙은 결과를 보기 전에 정했고 바꾸지 않는다.

### 6-2. 결과 표 형식(실행 후 §10)

정답마다 v1 s0, v1 s1, v2 s0, v2 s1, Δ, τ, 조건 충족 여부를 적는다. 지표는 band AP50, 관로 평균, 전체 mAP50이다. 데이터 수는 fold0 crop 1,298장이고, band GT는 v1 966개, v2 631개다. image 단위(3,525장)는 참고로 같은 형식으로 적는다.

### 6-3. 해석 주의 — fold 0의 학습 라벨 변화는 작다

- fold 0 검증 실험의 학습 집합(fold1–4)에서 band 박스는 crop 단위로 1,012 → 863개(−14.7 %, 129/4,175 crop 변경)만 바뀐다. 검증(fold0)은 966 → 631개(−34.7 %)가 바뀐다(§2).
- 그래서 v2 학습 모델도 대부분 v1과 같은 라벨로 배운다. 이 실험의 "v2 학습 효과"는 작게 나오기 쉽다. **동등이 나와도 "병합이 쓸모없다"는 근거가 되지 않는다.** 이 경우에는 동등을 "fold 0 조건에서 차이를 확인하지 못함"으로 읽는다.
- 같은 이유로 통과가 나와도 효과 크기는 학습 라벨 변화가 큰 fold(fold0이 학습 집합에 들어가는 fold 1–4)에서 달라질 수 있다.
- fold는 사용자 결정으로 0을 유지한다. 다른 fold 재확인은 §7의 후보 행동으로만 둔다.

### 6-4. band 보조 지표(판정에 넣지 않음)

| 지표 | 정의 | 지금 코드로 | 비고 |
|---|---|---|---|
| band P·R·AP50·AP50-95(두 정답, crop·image) | `metrics.json` per_class | **가능** | 러너 산출물에서 바로 |
| conf 0.25 혼동 행렬: band→배경 FP, 배경→band 미검출, band↔다른 클래스 | `confusion_matrix_conf0.25.csv` | **가능** | M1-7 §8-5와 같은 읽기(매칭 IoU 0.45) |
| band FP/GT 비(conf 0.25) | 위 행렬의 band 행 배경 열 ÷ band GT 수 | **가능** | M1-7 v1: 0.4–0.57× |
| 세션별 band AP50(fold0 세션마다) | 이미지별 예측을 매니페스트 `session`으로 묶어 AP50 | **python-expert 필요** | eval_detector가 이미지별 예측을 저장하지 않음. 소수 세션 편중(M1-7: 한 세션의 쪼개진 라벨) 확인용 |
| 가로 구간 겹침 재현율·정밀도(규약 무관) | 이미지마다 GT band x구간 합집합과 예측 band(conf 0.25) x구간 합집합을 비교. 세로 겹침(GT 높이의 50 % 이상)인 쌍만. 재현율 = 덮인 GT 길이/GT 길이, 정밀도 = GT와 겹친 예측 길이/예측 길이 | **python-expert 필요**(실행 뒤 가중치로 사후 계산) | 박스를 몇 개로 쪼갰는지에 둔감 → v1·v2 정답에서 거의 같은 값이 나와야 하고, §5-1 (c)의 기울기를 피하는 지표. 판정에는 넣지 않고 부분 통과·동등 해석에 쓴다 |
| 분절 수(v2 GT band 1개당 겹치는 예측 수) | x구간이 겹치는 conf 0.25 예측 개수의 평균·분포 | **python-expert 필요** | 학습 모델이 병합 규약을 따랐는지 직접 확인 |
| FP 범주(duplicate/fragment/other_class/no_gt) | `m17_pipe_band_fp_gallery.py`의 범주 | **python-expert 필요(일반화)** | 현재 실행 폴더가 `m17_aug0`로 고정됨. `--run-dir --eval-dir --dataset-dir` 인자화하면 재사용 가능 |

python-expert 구현 명세(요약): 이미지별 예측 내보내기 1개(`analysis/detection/`, 기존 m17 FP 갤러리와 같은 YOLO.val 경로·conf 0.25·NMS 0.7, 출력 `predictions.csv`: file_name, cls, conf, x1,y1,x2,y2)와 그 위의 집계 1개(세션별 AP50, 가로 구간 재현율·정밀도, 분절 수). 입력은 이 문서의 보기 폴더, 정답은 `--dataset-dir`. AGPL: ultralytics import는 analysis 스크립트에만 두고 `src/`(detect/ 밖)·`scripts/`에 넣지 않는다.

## 7. 결과별 후보 행동 (확정 아님 — 결과를 보고 사용자가 다시 결정)

| 결과 | 후보 행동 |
|---|---|
| 통과 | (가) M1-10 기준선을 v2 라벨로 재개 (나) 수동 재라벨 범위를 v2로도 남는 문제(no_gt FP, 특정 세션)로 축소 (다) 학습 라벨 변화가 큰 fold 1개에서 재확인(§6-3) |
| 부분 통과-규약 교환 의심 | (가) §6-4 가로 구간 지표를 사후 계산한 뒤 재판단 (나) 띠 깊이 규칙(gpr-processing-expert)에 어느 박스 규약이 맞는지 먼저 정함 (다) 소표본 수동 검수로 규약 결정 |
| 부분 통과-약함 | (가) 시드 1개 추가(seed 2) (나) 가로 구간 지표 사후 계산 뒤 재판단 |
| 동등 | (가) 라벨 일관성·깊이 규칙 단순성을 근거로 v2 채택 (나) v1 유지 (다) 수동 재라벨 진행 (라) 학습 라벨 변화가 큰 fold에서 재확인(§6-3: fold 0 동등은 "차이 미확인") |
| 열세 | (가) v1 유지 (나) 병합 규칙 재검토(gpr-dataset-curator: 간격·높이비 조건) (다) 수동 재라벨 진행 |

## 8. 실행 (러너, 미실행 — 실행 시점은 사용자 지시)

- 러너: `runs/logs/band_v2_runner.ps1`(make_runner.py 생성, 단계 정의 `runs/logs/band_v2_steps.json`), 로그 `runs/logs/band_v2_runner_logs/`.
- 단계 10개, 순서대로, 실패하면 멈춤(`--stop-on-fail`), 끝난 단계는 건너뜀(`--skip-done`).
  1. `train_v2_s0`, `train_v2_s1`: `scripts/ops/run_m17_sweep.ps1 -Strengths 0 -Epochs 30 -Batch 64 -Workers 2 -Mask 255 -Config configs/train_yolo11m_band_v2.yaml -Prefix yolo11m_bandv2 -Seed <0|1> -Suffix _20261031_fold0_s<0|1>`. 학습 로그는 `runs/logs/yolo11m_bandv2_aug0_20261031_fold0_s<k>.log`에도 남는다.
  2. `eval_<v1s0|v1s1|v2s0|v2s1>_gt<v2|v1>` 8개: `analysis/detection/band_v2_eval_view.ps1 -RunDir <run> -Gt <v1|v2>`.
- 자원: 마스크 0xFF, Normal, batch 64, workers 2(M1-7과 같음). 대기 조건은 M1-7 seed 1 러너와 같다(CPU 0,2,4,6 < 20 %, GPU 사용률 < 10 %, 여유 RAM > 8 GB, CPU 0–7 예약 겹침 없음, 60 s 간격 5회 연속, 최대 12 h). 예약 `gpr_detection:band_v2`(마스크 0xFF, GPU 0, TTL 60분, 단계마다 갱신), Job kill-on-close, `--gpu-log --gpu-counters`.
- 예상 시간: 대기 ≥ 5분 + 학습 2 × 약 15.7분(M1-7 강도 0 실측 948·928 s) + 평가 8 × 약 2.3분 ≈ **50–55분**. VRAM 정점 약 7.5 GB(M1-7 b64).

## 9. 산출물(실행 후)

- 학습: `runs/detect/yolo11m_bandv2_aug0_20261031_fold0_s{0,1}/`
- 평가: `runs/detect/band_v2_eval/<run>__gt_{v1,v2}/eval/last_{crop,image}/`(`metrics.json`, `per_class.csv`, `confusion_matrix_conf0.25.csv`, PR 곡선·혼동 행렬 PNG), `view_info.txt`(원 실행·정답 기록)
- 결과 표·판정은 이 문서 §10에 추가한다(실행 후 evaluate 모드).

## 10. 결과

(미실행)
