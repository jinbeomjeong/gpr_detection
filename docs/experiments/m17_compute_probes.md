# M1-7 증강 강도 비교 — 학습 자원 측정(probe)과 실행 준비 (2026-10-29)

증강 강도 비교 실험(M1-7) 전에 GPU 메모리·속도·CPU 배치를 1 epoch 측정으로 정했다. 실험 본체는 사용자 결정으로 보류했다가(다른 세션이 CPU를 쓰고 있어 "학습은 나중에") 2026-10-30에 실행·판정했다(결과 [m17_augmentation_strength.md](m17_augmentation_strength.md)). 배치 판단은 compute-scheduler-expert(구 cpu-scheduler-expert).

## 결정된 실행 설정
| 항목 | 값 | 근거 |
|---|---|---|
| batch | **64** | VRAM 최대 7,321 MiB(12 GB의 60 %), batch 32 대비 처리량 약 7 %↑. Ultralytics nbs=64라 grad 누적 없음. 5회 비교 안에서 고정 |
| DataLoader workers | **2** | worker 수를 늘린 이득이 없음. 마스크는 0xFF가 가장 빠름(w2 45.6 s 대 0xFFFF 계열 50.3–72.4 s; 같은 0xFFFF 안에서는 w3가 w2보다 빨랐고, 0xFF에서 w3·w4는 재지 않음, 아래 표) |
| CPU 배치 | P 논리 0–7(마스크 `0xFF`, 물리 4코어), 우선순위 Normal | 0xFFFF로 넓히면 느려짐 |
| 시작 조건 | P 짝수 논리 0·2·4·6 평균 < 20 %, GPU < 10 %, 여유 RAM > 8 GB가 5분 연속 | 다른 세션 Chrono 워커가 짝수 논리를 씀 |
| 시간(2026-10-30 실측) | 첫 epoch 42–46 s, 이후 약 30 s → 30 epoch 약 16분/회(wall 934–948 s, 다른 세션과 CPU가 겹치면 1,068 s), 4회 합 3,886 s(약 1.1시간) | 1 epoch probe의 46 s는 첫 epoch 캐시 생성이 섞여 과대 추정이었다 |

## 1 epoch 측정 결과 (fold 0, 증강 강도 1, imgsz 256, 1채널 scratch)
| 설정 | epoch(검증 포함) | train 구간 | GPU 사용률(동작 중 평균) | 최대 VRAM |
|---|---|---|---|---|
| b32, w2, 0xFF, Normal | 49.2 s | 31.0 s | 55.5 % | 4,025 MiB |
| **b64, w2, 0xFF, Normal** | **45.6 s** | 27.2 s | 64.6 % | 7,321 MiB |
| b64, w2, 0xFFFF, Normal | 72.4 s | 30.5 s | 42.3 % | 7,350 MiB |
| b64, w3, 0xFFFF, Normal | 50.3 s | 28.4 s | 61.0 % | 7,321 MiB |
| b64, w4, 0xFFFF, Normal | 55.5 s | 30.3 s | 64.2 % | 7,321 MiB |
| b64, w2, `0xAA`(P 홀수 논리 4), BelowNormal — 다른 세션 부하 중 | 109.1 s | 51.4 s | 38.8 % | 7,350 MiB |

- GPU 사용률이 60 %대에 머무는 병목은 worker 수가 아니라 메인 프로세스 쪽(배치 묶기·작은 모델 스텝 오버헤드)과 검증 구간으로 추정한다. 1 epoch 측정이라 첫 epoch 캐시 생성이 섞여 있다.
- **다른 세션과의 공존 시험(마지막 행)은 실패**: 다른 세션(outrigger Chrono 워커 22개, E 전부와 P 짝수 논리)의 CPU 점유가 검증 구간에서 평균 98 %→88 %, 최솟값 48 %까지 떨어졌다(BelowNormal인데도 검증 worker 4개가 겹침). E 전용은 E가 이미 100 %라 Normal이면 상대를 절반으로 늦추고, BelowNormal이면 학습이 멈추다시피 해 제외했다.

## 다시 시작하는 방법(재현·추가 실행용)
- 자원이 빌 때까지 기다렸다가 자동 시작: `pwsh -NoProfile -File scripts/ops/wait_then_m17.ps1` (강도 0, 0.5, 1, 2 순차, batch 64, workers 2, 마스크 0xFF, 첫 실패에서 멈춤, 로그 `runs/logs/`).
- 바로 실행: `pwsh -NoProfile -File scripts/ops/run_m17_sweep.ps1 -Strengths 0,0.5,1,2 -Epochs 30 -Batch 64 -Workers 2 -Mask 255 -Prefix m17` (`-Priority`로 우선순위, 완료된 실행은 건너뜀).
- ringing 끔 5회째: 4회 판정 뒤 `-Strengths <최적> -Suffix _noring -Config configs/train_yolo11m_noringing.yaml`.
- 실행 중 15분 간격 long-running-task-monitor 점검, 끝나면 compute-scheduler-expert `기록` 모드로 실측(시간·최대 RSS·VRAM) 기록.
- 측정 로그: `runs/logs/m17_probe_*` (git 제외).
- 대안(2026-10-30 전역 도구): 위 대기 스크립트 대신 `~/.claude/tools/cpu-scheduler/make_runner.py --mask 0xFF --reserve-owner gpr_detection:m17 --reserve-ttl-min <가장 긴 단계보다 길게>`(세션 간 예약 자동 등록·해제)로 러너를 만들 수 있다. 대기 조건은 `--wait-args "--cpu \"0,2,4,6<20\" --gpu-util \"<10\" --ram-free-gb \">8\""`로 러너에 넣는다(`wait_until.py` 조건, 기본 5회 연속 충족). 어느 쪽으로 할지는 재개 때 compute-scheduler-expert 배치표를 받아 정한다.
