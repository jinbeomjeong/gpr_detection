# M1-10 YOLO11m 기준선 — 실험 설계 (2026-10-30, 실행 전)

AIHub train_pool 세션 그룹 5-fold로 YOLO11m(scratch, 1채널)을 학습해 fold 평균±표준편차를 기준선으로 남긴다. 이 문서는 실행 전 설계이고, 결과는 `docs/experiments/<날짜>_m110_baseline.md`에 따로 쓴다.
근거: `docs/CHECKLIST.md` M1-10(사용자 결정 2026-10-30), `docs/experiments/m17_augmentation_strength.md`, `docs/experiments/m17_compute_probes.md`, `analysis/detection/m110_epoch_budget.json`, ADR-001·ADR-003, `docs/datasets/dataset_spec_m1.md`.

## 1. 설정

설정 파일: `configs/train_yolo11m_m110.yaml`(base를 복사한 단독 파일. `scripts/train_detector.py`는 설정 파일 하나만 읽고 상속은 지원하지 않는다).

| 항목 | 값 | 근거 |
|---|---|---|
| 모델 | `yolo11m.yaml`, `pretrained: false`(scratch), 입력 1채널(`channels: 1`, 학습 스크립트가 데이터 yaml 사본에 추가) | M1 결정 2026-10-29 |
| 입력 | imgsz 256(원본 256×256, letterbox 변화 없음), `rect: false` | dataset_spec_m1 |
| 예산 | **fold당 50 epoch**, `patience: 0`(조기 종료 끔) | 사용자 2026-10-30 |
| 옵티마이저 | `optimizer: AdamW`, `lr0: 0.00125`, **`momentum: 0.9`**(AdamW beta1), `warmup_bias_lr: 0.0`, `lrf 0.01`(선형 감소), `cos_lr false`, `weight_decay 0.0005`, `warmup_epochs 3`, `nbs 64` | M1-7에서 `auto`가 고른 값과 같게 맞춤(아래 주) |
| batch / workers / AMP | 64 / 2 / on | M1-7 probe(`m17_compute_probes.md`) |
| Ultralytics 증강 | fliplr 0.5, translate 0.1, scale 0.2, **flipud 0**, degrees·shear·perspective 0, mosaic·mixup·cutmix·copy_paste 0, hsv_h·hsv_s 0, hsv_v 0.3(1채널이라 효과 없음) | `augmentation_policy.md`, M1-7과 같음 |
| 사용자 정의 증강 | `gpr_augment.strength: 0`(4종 모두 끔) | M1-7 판정 2026-10-30 |
| 판정 가중치 | **`last.pt`**(50 epoch 끝). `best.pt`는 민감도용 | 사용자 2026-10-30 |
| 결정성 | `deterministic: true`, 학습 시드는 §6 | ADR-003 결정 6 |
| CPU·GPU | 러너 affinity 마스크 0xFF(코어 0–7), GPU 0 단독 | M1-7 probe |

**주 — `momentum: 0.9`를 넣은 이유.** Ultralytics 8.4.192 `build_optimizer`는 `optimizer: auto`일 때 `("AdamW", lr_fit, 0.9)`로 beta1을 0.9로 고정한다(M1-7 로그 `AdamW(lr=0.00125, momentum=0.9)`). AdamW를 명시하면 `args.momentum`(기본 0.937)이 beta1로 들어가므로, M1-7과 같은 옵티마이저를 쓰려면 `momentum: 0.9`도 명시해야 한다. 체크리스트 결정 문구에는 없던 값이지만 결정 의도(M1-7 auto 결과 고정)와 같다. Ultralytics `get_cfg`로 설정을 검사했다(optimizer AdamW, lr0 0.00125, momentum 0.9, warmup_bias_lr 0.0, patience 0, epochs 50).

## 2. 데이터와 fold별 규모

매니페스트 `data/splits/aihub_yolo_v1.csv`(분할 seed 20261029, test_attempt 0). fold k 학습 = 나머지 4 fold, 검증 = fold k. 집계: `analysis/detection/m110_leak_check.json`.

| fold | 학습 장수 | 검증 장수 | 검증 crop | 검증 세션 | 검증 인스턴스 crop 단위 (hyperbola / band / cavity / manhole) | 이미지 단위 | 예상 시간(분) |
|---|---|---|---|---|---|---|---|
| 0 | 11,276 | 3,525 | 1,298 | 308 | 71 / 966 / 365 / 408 | 185 / 2,649 / 986 / 1,118 | 25.6 |
| 1 | 11,591 | 3,210 | 1,186 | 308 | 45 / 503 / 367 / 425 | 125 / 1,370 / 987 / 1,143 | 26.3 |
| 2 | 11,900 | 2,901 | 1,074 | 304 | 98 / 258 / 364 / 418 | 260 / 700 / 987 / 1,122 | 27.0 |
| 3 | 12,311 | 2,490 | 922 | 307 | 147 / 62 / 367 / 439 | 405 / 166 / 987 / 1,194 | 27.9 |
| 4 | 12,126 | 2,675 | 993 | 304 | 142 / 189 / 364 / 446 | 382 / 512 / 987 / 1,198 | 27.5 |
| test(봉인) | — | 2,086 | 770 | 271 | 61 / 301 / 259 / 303 | 165 / 796 / 696 / 847 | — |

- 예상 시간 = 50 × 29.7 s × (학습 장수 / 11,276) + 54 s(첫 epoch 추가분·시작·마지막 검증). M1-7 fold 0 실측(30 epoch 906–908 s)으로 보정한 값이다. **5 fold 합 약 2.24 h**, 다른 세션과 CPU가 겹치면(M1-7 강도 2에서 +15 %) 약 2.6 h. VRAM 정점 약 7.5 GB(M1-7).
- crop 단위 검증 = crop마다 gain 사본 1장(1.0 > 0.5 > 0.25 우선). gain 사본은 독립 표본이 아니므로 판정은 crop 단위로 한다(dataset_spec_m1 §4 항목 3).
- **fold 간 관로 분포가 크게 다르다.** pipe_band crop 인스턴스는 fold 0에 966개, fold 3에 62개다. pipe_hyperbola는 fold마다 45–147개다. 클래스별 AP의 fold 표준편차는 클 것이고, fold 1·3의 관로 AP는 표본이 적어 흔들린다. 판정 지표(전체 mAP50)는 4클래스 평균이라 영향을 덜 받는다.

## 3. 누수 점검 — 통과 (2026-10-30 재단언)

`analysis/detection/m110_leak_check.py`(CPU, 읽기 전용) 결과 `analysis/detection/m110_leak_check.json`:
- fold 0–4·test(−1) 6개 집합의 15쌍 × 4키(`session`, `session_from_name`, `crop_key`, `split_group`)에서 겹침 **0건**.
- 모든 `crop_key`(gain 사본 묶음)가 fold 하나에만 있다(여러 fold에 걸친 crop 0건).
- `fold{k}_images.txt`·`test_images.txt`가 매니페스트 fold 집합과 정확히 같고, `fold{k}.yaml`의 train은 나머지 4 fold 목록, val은 fold k 목록이며 train∩val 0건, 클래스 이름 `0 pipe_hyperbola, 1 pipe_band, 2 cavity, 3 manhole`.
- 참고(누수 아님): pHash 근접 중복 경계 쌍은 `dataset_spec_m1.md` §한계에 있는 그대로다(세션이 다름).

## 4. 실행 순서와 재개

1. **선행 조건**: (a) **M1-9 smoke 통과 뒤 시작**(아래 3). (b) 체크리스트상 M1-0b(디코딩 재측정)는 M1-10 시작 전에 한다. (c) GPU에 다른 학습이 없을 것(M1-7 seed 1 반복은 사용자 결정으로 보류 — 재요청 시 실행, 현재 gpr_detection 예약 없음). (d) 학습 단계는 `--on-existing archive`로 실행한다(§8 A).
2. **한 fold씩 순차**: fold 0 → 1 → 2 → 3 → 4. make_runner 러너 하나에 5단계로 넣고 `--skip-done`, 실패하면 멈춘다. smoke는 이 러너에 넣지 않는다.
3. **M1-9 smoke — 별도 단독 실행(사용자가 지시할 때)**. 설정 `configs/train_yolo11m_smoke.yaml`(m110 설정과 epochs 3만 다름: AdamW·lr0 0.00125·momentum 0.9·warmup_bias_lr 0·patience 0·strength 0·batch 64·workers 2·fold 0·seed 0). run 이름 `yolo11m_aihubv1_<YYYYMMDD>_smoke_fold0`, 로그 `runs/logs/m19_smoke.log`. 예상 약 3–4분(첫 epoch 약 45 s + 2 × 30 s + 시작·마지막 검증 약 1분, conda 시작 포함).
   - 실행(cwd `C:\workspace\gpr_detection`, 마스크 0xFF 권장):
     `C:\ProgramData\miniforge3\Scripts\conda.exe run --no-capture-output -n gpr-dev python -I C:\workspace\gpr_detection\scripts\train_detector.py --config C:\workspace\gpr_detection\configs\train_yolo11m_smoke.yaml --fold 0 --seed 0 --name yolo11m_aihubv1_<YYYYMMDD>_smoke_fold0 2>&1 | Tee-Object C:\workspace\gpr_detection\runs\logs\m19_smoke.log`
   - 판정: `conda run -n gpr-dev python -I analysis/detection/m19_smoke_check.py --run-dir runs/detect/yolo11m_aihubv1_<YYYYMMDD>_smoke_fold0 --log runs/logs/m19_smoke.log` → 종료 0이면 통과.
   - 통과 기준: (1) 로그에 `AdamW(lr=0.00125, momentum=0.9)`가 있고 `'optimizer=auto' found`가 없음 (2) `args.yaml`이 optimizer AdamW·lr0 0.00125·momentum 0.9·warmup_bias_lr 0·patience 0·epochs 3·flipud 0·seed 0·imgsz 256·batch 64 (3) `gpr_train_settings.yaml`의 augment_strength 0·fold 0 (4) `results.csv` 데이터 3행, `weights/last.pt` 있음 (5) epoch 1 mAP50이 0.20–0.40(M1-7 fold 0 epoch 1: 0.273–0.301. 첫 epoch는 warmup 구간이라 LR 일정이 50 epoch 설정과 같다). epoch 2–3 값은 M1-7에서도 0.30–0.59로 흔들려 기준에 넣지 않는다.
   - 검사 스크립트는 M1-7 `m17_aug0`(optimizer auto)에 돌려 보면 불통과(종료 1)로 정상 동작함을 확인했다.
4. **재개 규칙(fold 단위로 원자적)**: 중간에 멈춘 fold는 이어 받지 않고 **처음부터 다시** 학습한다(Ultralytics resume은 데이터 순서·난수 상태가 끊긴 지점에서 달라져 다른 fold와 조건이 달라진다. 손실은 최대 약 28분). 절차: 학습 단계에 `--on-existing archive`를 붙여 두고 같은 러너를 다시 실행하면 된다. 완료 폴더는 종료 0으로 건너뛰고, 미완료 폴더는 `runs/detect/_incomplete/`로 옮긴 뒤 처음부터 학습한다(`.done`이 있는 단계는 러너가 먼저 건너뜀). 완료 판정: `results.csv` 데이터 행 수 ≥ epochs + `weights/last.pt` 존재. (옵션 없이 다시 돌리면 Ultralytics가 `<run>2`를 만들어 이름 규칙이 깨진다 — 기본 `error`는 이를 막고 종료한다.)
5. Windows DataLoader 오류가 나면 해당 fold를 `--workers 0`으로 다시 돌리고 결과 문서에 적는다(epoch 시간이 늘어난다).
6. **5 fold 학습이 모두 끝난 뒤** 평가 러너(§5)를 따로 실행한다. 학습 중 평가 병행 금지: `eval_detector.py`가 돌면 Ultralytics가 `data/aihub_yolo_v1/labels.cache`를 다시 써서 학습 중인 데이터로더와 겹칠 수 있다. 평가 러너 순서: ① 수용 확인(`m17_aug0` 재평가 + `analysis/detection/m110_eval_acceptance.py`) → ② fold 0–4 평가 → ③ 집계. ①이 실패하면 멈춘다.

## 5. 평가·판정

### 5-1. 평가 실행(fold마다)
- 명령: `scripts/eval_detector.py --run-dir <fold 폴더>`(기본 `last,best` × `crop,image`). 출력 `<run>/eval/<last|best>_<crop|image>/`. `--fold test`는 `--allow-test` 없이 종료 2(test 봉인).
- 가중치 2종 × 단위 2종: `last.pt`·`best.pt` × crop 단위·이미지 단위. 설정은 M1-7 재평가와 같게: `imgsz 256, batch 64, conf 0.001, iou 0.7`(`rect` 켜짐, YOLO.val 기본).
- **혼동 행렬 임계값 주의**: Ultralytics 8.4.192는 혼동 행렬을 `conf=self.args.conf`로 채운다(`models/yolo/detect/val.py:266`, `plots=True`일 때만, 매칭 IoU 0.45 고정 `utils/metrics.py:412`). 따라서 기본 평가의 `confusion_matrix.csv`는 **conf 0.001 행렬**이다(`metrics.json`의 `confusion_matrix.conf_threshold`에 기록). 저신뢰 박스까지 세므로 배경→클래스 오검출 수는 운용 임계값의 오검출로 읽으면 안 된다. 운용 임계값(conf 0.25) 행렬은 `--confusion-conf 0.25`로 같은 폴더에 `confusion_matrix_conf0.25.csv`로 따로 남긴다(AP와 기본 행렬은 conf 0.001 그대로, 두 번째 val 부산물은 하위 폴더 `confusion_conf0.25/`, 평가 시간 약 2배; §8 D).
- **수치 저장(M1-7 공백 보완)**: 각 평가 폴더에 `metrics.json`(필수)·`per_class.csv`·`confusion_matrix.csv`·`eval_images.txt`·`eval_data.yaml`, 그림(PNG)은 부수 산출물. M1-7 `m17_eval`은 PNG만 남아 수치를 다시 만들 수 없었다.
- 지표 값은 **분수(0–1)**로 저장한다(`settings.values_are_fractions: true`). 문서 표에서 %로 바꾼다.
- `metrics.json` 필드: 가중치 경로·sha256, `weights.checkpoint_epoch_zero_based`·`weights.epochs_completed`(last.pt면 50, best.pt는 고른 epoch), fold, 단위, 평가 이미지 목록 경로와 sha256, 이미지 수, 클래스별 {인스턴스 수, 이미지 수, P, R, AP50, AP50-95}, 전체 mAP50·mAP50-95(present class 평균), **관로 2클래스 AP50 평균**(pipe_hyperbola·pipe_band), 혼동 행렬(5×5, 배경 포함, 행·열 의미 명시), `conf_threshold`·`confusion_conf_threshold`(옵션 없으면 null)·`confusion_matrix_at_confusion_conf`, 평가 설정(`settings.confusion_conf` 포함 — fold마다 다르면 집계 종료 1), Ultralytics·torch 버전, git 커밋.

### 5-2. 집계(5 fold)
- 명령: `scripts/aggregate_folds.py --runs <fold 폴더 5개> --weights last --unit crop --out <경로>`. 요약에 n(fold 수)을 적고, fold 사이 평가 설정(imgsz·batch·conf·iou 등)이 다르면 종료 1로 막는다(device 차이만 허용).
- fold 평균 ± **표본 표준편차(ddof=1, n=5)**, fold를 같은 가중치로 평균(비가중). fold별 값과 fold별 검증 장수·인스턴스 수를 같은 표에 둔다.
- 집계 산출물은 git이 추적하는 `analysis/detection/m110_fold_summary.json`·`.csv`에 둔다(`runs/`는 git 제외라 수치가 사라지지 않게).
- 지표: 전체 mAP50(주 지표), mAP50-95, 관로 2클래스 AP50 평균, 클래스별 AP50·AP50-95. 단위: **crop 단위가 기준**, 이미지 단위는 참고. 가중치: **`last.pt`가 기준**, `best.pt`는 부록(best.pt는 같은 검증 fold의 mAP50-95로 고른 epoch라 낙관 편향).

### 5-3. 이 기준선이 쓰이는 곳
- M4-3 이전 판정(ADR-001): RF-DETR의 crop 단위 `last` 전체 mAP50 fold 평균이 이 값에서 −5 %p 이내면 `이전 가능`. 같은 매니페스트·같은 fold·같은 50 epoch 예산·같은 입력 크기·같은 시드 정책으로 맞춘다. 클래스별 AP50(특히 pipe_hyperbola)은 보고만 한다.
- U-3 목표치 확정의 기준값(fold 평균).

### 5-4. test 홀드아웃 — M4-3 판정까지 봉인(사용자 결정 2026-10-30)
- M1-10에서는 test를 보지 않는다. M4-3 판정이 끝난 뒤 채택 모델(필요하면 두 모델 모두)을 train_pool 전체(`test.yaml`의 train = 5 fold 합)로 같은 50 epoch 학습하고, `last.pt`로 test를 **1회** 측정한다(crop 단위 기준, 이미지 단위 참고).
- 이유: test 값을 일찍 보면 그 뒤의 결정(RF-DETR 설정, 목표치, 증강 재검토)이 test에 맞춰질 수 있다. 기준선 성능 보고는 5-fold 평균으로 충분하다(ADR-003 결정 6 "test는 최종 1회").
- 주의: `test.yaml`은 val도 test 목록을 가리킨다. 전체 학습 때 Ultralytics가 epoch마다 test로 검증하고 `best.pt`를 test 기준으로 고르므로, 그 실행의 `best.pt`는 쓰지 않는다(`last.pt`만). epoch별 test 곡선도 보지 않는 것이 원칙이다(가능하면 `val: false` — 이 경우에도 Ultralytics는 마지막 epoch에서 한 번 검증한다).

## 6. 시드 정책 (사용자 결정 2026-10-30)
- 분할 seed: 20261029(매니페스트 고정).
- 학습 seed: **모든 fold 0**. M1-7 fold 0 seed 0과 이어지고, fold 사이 차이가 데이터만으로 생긴다. 각 fold 폴더의 `gpr_train_settings.yaml`(`ultralytics.seed`, `augment_seed`)에 기록되고, 결과 표에도 적는다.
- 실행 간 변동(같은 fold, 다른 seed)은 아직 재지 않았다(M1-7 seed 1 반복은 보류, 사용자 재요청 시 실행). fold 표준편차에는 seed 변동이 섞여 있다는 점을 결과 문서에 적는다.
- M4-3 RF-DETR도 fold마다 같은 seed(0)를 쓴다.

## 7. 산출물 경로
| 산출물 | 경로 |
|---|---|
| 설정 | `configs/train_yolo11m_m110.yaml` |
| M1-9 smoke | 설정 `configs/train_yolo11m_smoke.yaml`, 폴더 `runs/detect/yolo11m_aihubv1_<YYYYMMDD>_smoke_fold0/`, 로그 `runs/logs/m19_smoke.log`, 판정 `analysis/detection/m19_smoke_check.py` |
| fold 학습 폴더 | `runs/detect/yolo11m_aihubv1_<YYYYMMDD>_fold<k>/`(날짜는 러너를 만든 날로 5 fold 공통). 안에 `weights/last.pt`·`best.pt`, `results.csv`, `args.yaml`, `gpr_train_settings.yaml` |
| 1채널 데이터 yaml | `runs/detect/_data/yolo11m_aihubv1_<YYYYMMDD>_fold<k>.yaml`(`train_detector.py`가 `data/aihub_yolo_v1/fold<k>.yaml`을 복사하고 `channels: 1` 추가) |
| 러너·로그(2026-10-30 생성, 미실행) | 학습 `runs/logs/m110_train_runner.ps1`(fold0–4, 마스크 0xFF Normal, `--skip-done` `--stop-on-fail`, 예약 `gpr_detection:m110` TTL 90분 GPU 0, `--job-kill-on-close`, 대기: 코어 0·2·4·6 < 20 %·GPU < 10 %·여유 RAM > 8 GB·0–7 예약 겹침 없음 5회 연속, 12 h 타임아웃). 평가 `runs/logs/m110_eval_runner.ps1`(accept_eval·accept_check·eval0–4·집계 3종 — last·crop, last·image, best·crop; 평가는 모두 `--confusion-conf 0.25`; 예약 `gpr_detection:m110_eval` TTL 30분; 대기에 `--proc-absent m110_train_runner.ps1` — **학습 러너를 띄운 뒤에만** 띄울 것). 로그 `runs/logs/m110_{train,eval}_runner_logs/<id>.log`·`.gpu.csv`·`.done`. 실행: `pwsh -NoProfile -File <러너>`. 배치 판단: compute-scheduler-expert(2026-10-30) |
| fold 평가 | `runs/detect/yolo11m_aihubv1_<YYYYMMDD>_fold<k>/eval/<last|best>_<crop|image>/`(`metrics.json`, `per_class.csv`, `confusion_matrix.csv`, `eval_images.txt`, `eval_data.yaml`, PNG) |
| 평가 수용 확인 | `runs/detect/m17_aug0/eval/last_<crop|image>/`, 판정 `analysis/detection/m110_eval_acceptance.py`(M1-7 문서 값과 0.1 %p 안) |
| 집계(git 추적) | 판정용 `analysis/detection/m110_fold_summary.json`·`.csv`(last·crop), 참고용 `m110_fold_summary_last_image.json`·`.csv`, `m110_fold_summary_best_crop.json`·`.csv` |
| 누수 점검 | `analysis/detection/m110_leak_check.py`, `analysis/detection/m110_leak_check.json` |
| 결과 문서 | `docs/experiments/<YYYY-MM-DD>_m110_baseline.md` |

## 8. 구현(python-expert, 2026-10-30 완료, 시험 54건 통과 — A·B·C 51건 + D 3건)
- **A. 실행 폴더 처리**: `scripts/train_detector.py --on-existing {error,skip-complete,archive}`(기본 `error`). `archive`: 완료 폴더면 종료 0으로 건너뜀, 미완료면 `runs/detect/_incomplete/`로 옮긴 뒤 학습. `skip-complete`: 완료면 종료 0, 미완료면 종료 2. 실행 정보(git·버전 등)는 `src/gpr_detection/run_info.py`.
- **B. 수치 평가**: `src/gpr_detection/data/splits.py`(`select_crop_representatives`, Ultralytics 없음), `src/gpr_detection/detect/yolo_eval.py`(Ultralytics 지연 import, `detect/` 안), `scripts/eval_detector.py`(CLI). 산출물 §5-1. 부수 효과: 평가가 `data/aihub_yolo_v1/labels.cache`를 다시 쓴다 → 학습과 병행 금지.
- **C. 집계**: `scripts/aggregate_folds.py`(Ultralytics 없음). §5-2.
- **D. 운용 임계값 혼동 행렬 — 구현 완료(2026-10-30, 시험 54건 통과)**: `eval_detector.py --confusion-conf 0.25`(AP는 conf 0.001로, 혼동 행렬만 별도 val 한 번 더 → `confusion_matrix_conf0.25.csv`). `--conf 0.25`로 다시 돌리면 출력 폴더가 같아 conf 0.001 결과를 덮어쓰므로 쓰지 않는다.

## 9. 결정 사항(사용자 2026-10-30)과 남은 질문
1. test 평가 시점 — **M4-3 판정까지 봉인**(§5-4).
2. 학습 seed — **모든 fold seed 0**(§6).
3. M1-9 smoke — **따로 실행**. M1-10 러너에 묶지 않고 사용자가 지시할 때 단독 실행(§4-3). M1-10은 smoke 통과 뒤 시작.
4. `momentum: 0.9` — **유지**(§1 주).
5. 구현 §8 A·B·C — 완료(2026-10-30). 평가는 5 fold 학습이 끝난 뒤 별도 러너로 한다(§4-6).
- 남은 질문: 없음.
