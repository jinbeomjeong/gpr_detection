# gpr_detection 업무 체크리스트

단계 설명은 [`ROADMAP.md`](ROADMAP.md). 항목 하나가 끝나면 체크하고 커밋·push한다(메시지에 항목 번호). `→` 뒤는 담당 에이전트와 산출물.

## M0 초기화
- [x] M0-1 프로젝트 골격(git, `.gitignore`, `.gitattributes`, `CLAUDE.md`) → 메인 세션
- [x] M0-2 데이터셋 5종 조사·README/Codebook 추출 → Explore, docx-document-expert, pdf-document-export / `docs/datasets/`
- [x] M0-3 선행 연구 조사(정밀) → web-research-expert / `docs/research/prior_art.md`
- [x] M0-4 패키지 조사(정밀 + 에이전트별 절약) → web-research-expert / `docs/research/packages*.md`
- [x] M0-5 conda env 3개 설치·검증(gpr-dev, gpr-detr, gpr-label) → 메인 세션 / `environment.yml`, `envs/`
- [x] M0-6 프로젝트 에이전트 3개 설계·구현·시험 → 메인 세션 / `.claude/agents/`
- [x] M0-7 gprMax 합성 깊이 검증 세트 20개 + 물리 메모 → electromagnetic-simulation-expert / `scripts/synthetic/`, `docs/research/depth_physics.md`
- [x] M0-8 AIHub XML 전수 집계(lpipe 띠·tpipe 쌍곡선, Validation 세션 100 %가 Training과 중복) → gpr-dataset-curator / `docs/datasets/aihub_71327.md`
- [x] M0-9 증강 정책 → cv-detection-expert / `docs/experiments/augmentation_policy.md`
- [x] M0-10 깊이 명세 v1 → gpr-processing-expert / `docs/processing/spec_depth_v1.md`
- [x] M0-11 로드맵·체크리스트 → 메인 세션 / `docs/ROADMAP.md`, `docs/CHECKLIST.md`
- [x] M0-12 ADR 3건(YOLO11m→Apache 이전, 물리 기반 깊이, 데이터 분할·클래스 4개) + 아키텍처 개요, 확인 질문 Q1–Q5 사용자 답변 반영 → add-architect / `docs/architecture/`

## M1 데이터·검출 기준선

**M1 결정 사항(사용자, 2026-10-29)**: AIHub는 B-scan만(C-scan 제외), 4클래스 전체를 쓰되 공동·맨홀은 세션 균형을 유지하며 관로 이미지 수(약 5.6k장)와 비슷한 규모로 축소(총 약 17k장) · gain 0.25/0.5/1.0 세 벌 모두 학습에 사용(같은 fold, 평가는 crop 단위 1회) · 모로코는 교차 도메인 평가 전용(M3-3) · YOLO11m은 **사전학습 없이 scratch** · imgsz 256(원본) · fold당 epoch는 M1-9 smoke의 1 epoch 시간·수렴을 보고 사용자에게 다시 묻는다(→ 2026-10-30 M1-7 결과로 **50 epoch** 결정, M1-10 항목) · M1-0b는 M1-10 시작 전에 · AIHub test 세션 홀드아웃(약 15 %, 관로 종류별 이미지 비중 10–25 %), 나머지로 5-fold.

- [x] M1-1 패키징(`pyproject.toml`, src 레이아웃, `gpr-detect` 진입점) → python-expert
- [x] M1-2 io 로더 5종 + `Radargram`·`AnnotatedImage` 데이터 클래스 + 시험(20건 통과, 5종 전수 읽기 확인) → python-expert / `src/gpr_detection/io/`, `tests/test_io.py`
- [x] M1-3 AIHub zip → YOLO 변환(멤버 이름 짝짓기, 4클래스, test 세션 홀드아웃(약 15 %, 관로 비중 10–25 %) + train_pool 세션 그룹 5-fold `StratifiedGroupKFold` — 매니페스트에 fold 열 0–4, fold별 세션·gain 크롭 키 누수 단언, ADR-003) — 명세 `docs/datasets/dataset_spec_m1.md` → python-expert(구현), gpr-dataset-curator(실행·감사) / `scripts/build_yolo_dataset.py`, `data/splits/`
- [x] M1-4 모로코 통합(회전·탄성 사본을 `aug_transform`으로 제외한 목록, 그룹 분할 유지, 공동 비등방 늘림 주의 기록) — **결정(2026-10-29)**: 교차 도메인 평가 전용, 회전·탄성 제외 뒤 **그룹당 원본에 가장 가까운 사본 1장**(관로 _aug_6·공동 _aug_13·이상 없음 _aug_12, 독립 표본 약 233그룹), 이상 없음(빈 라벨)을 오검출 평가용으로 포함, 공동은 포함하되 비등방 늘림 주석과 함께 따로 보고, 파일은 원본 크기 유지·평가 때 imgsz 256 letterbox → gpr-dataset-curator(실행·감사), python-expert(변환 스크립트)
- [x] M1-5 누수 단언·근접 중복(imagehash) 감사 — AIHub 분은 M1-3 감사로 완료(`audit_aihub_yolo_v1.md`: 누수 0, 경계 넘는 근접 중복 해밍≤2 21쌍), 모로코 분도 감사 완료(`audit_morocco_eval_v1.md`: 통과, AIHub↔모로코 해밍 ≤2 쌍 없음(최소 6, 육안으로 다른 장면)) → gpr-dataset-curator / `docs/datasets/audit_*.md`
- [x] M1-6 데이터셋 카드(AIHub·모로코·4TU·광저우대·GPR-SD) — **결정(2026-10-29)**: 5종 모두 데이터셋마다 1장(`docs/datasets/cards/<이름>.md`, 기존 문서는 근거 링크), **Hugging Face 데이터셋 카드 형식**(YAML 메타데이터 머리 + 정해진 절), AIHub 이용약관은 "미확인"으로 적고 원본·라벨·파생 데이터 비공개 규칙을 명시, 본문 한국어 + 머리에 영어 요약 → gpr-dataset-curator / `docs/datasets/cards/`
- [ ] M1-7 (구현 완료 2026-10-29: `src/gpr_detection/augment.py`, `detect/yolo_train.py` 어댑터. 강도 비교 4회 완료·판정 2026-10-30: **최적 강도 0** — `docs/experiments/m17_augmentation_strength.md`. 최적이 0이라 "ringing 끈 5회째"는 강도 0과 같은 실험이 되므로, 사용자 결정(2026-10-30)으로 대신 **강도 0·1을 seed 1로 반복**(실행 간 변동 측정) — **실행 보류**(사용자 결정 2026-10-30, 재요청 시 `runs/logs/m17_seed1_runner.ps1`, M1-10과 GPU 겹치지 않게), 끝나면 판정 재확인 후 [x]. 실행 설정 batch 64·workers 2·마스크 0xFF, 방법은 `docs/experiments/m17_compute_probes.md`) 사용자 정의 증강(가우시안 잡음·대비, albumentations 없이) — **결정(2026-10-29)**: 가우시안 잡음·대비/감마·수평 띠 잡음(ringing)·트레이스 누락 4종, numpy 이미지·박스를 받는 **백엔드 공통 함수**(`detect/` 밖, 예: `src/gpr_detection/augment.py`) + YOLO·RF-DETR용 얇은 어댑터, **강도 비교 실험까지**(짧은 학습으로 강도 후보 비교 — M1-8 래퍼·M1-9 smoke 경로가 필요), 모델 입력 **1채널**(그레이). 수평 띠 잡음은 `pipe_band`(수평 띠 관로)와 혼동될 수 있어 진폭을 낮게·비교 실험에서 효과 확인 — **강도 비교 결정(2026-10-29)**: strength 0/0.5/1/2 + 최적 강도에서 ringing만 끈 1회(총 5회), fold 0·전체 데이터·30 epoch(시작 전 1 epoch 시간 측정, 30분 초과라 메인 세션 실행 + long-running-task-monitor), 판정은 관로 2클래스(pipe_hyperbola·pipe_band) AP50 평균 최고, 차이 1 %p 안이면 약한 강도, 전체 mAP50도 보고, M2는 계획대로 M1-10 때 시작 → python-expert(구현), cv-detection-expert(강도 비교 설계·판정) / `src/gpr_detection/augment.py`, `src/gpr_detection/detect/yolo_train.py`, `scripts/train_detector.py`, `configs/train_yolo11m_base.yaml`
- [x] M1-8 `Detector` 프로토콜 + YOLO 래퍼(ultralytics import는 `detect/` 안에만, 지연 import). 백엔드는 L2 박스 + `InputTransform`을 `DetectorOutput`으로 반환, pipeline 공통 좌표 역변환(L2→L1→L0) 함수와 왕복 시험(≤0.5 px), 정적 시험(`detect/` 밖 ultralytics·rfdetr import 0건, detect↔depth 상호 import 금지, src→scripts import 금지) — `docs/architecture/README.md` §2-3 + `GridBox`(원본 trace·sample 좌표 박스)를 `io/radargram.py`에 추가 → python-expert
- [x] M1-0a 이미지 디코딩 상대 비교(오염 상태 측정) → python-expert / `scripts/bench/decode_bench.py`, `docs/experiments/decode_benchmark.md`
- [ ] M1-0b 디코딩 판정 재측정(CPU가 빌 때, E코어·P코어 각각 `--affinity`) → python-expert, 시작 전 long-running-task-monitor로 측정 오염 점검
- [ ] M1-9 YOLO11m smoke 학습(3 epoch) — 설정·판정 스크립트 준비 완료(2026-10-30), **실행 대기(사용자 지시 시 단독 실행**, 명령·통과 기준은 `docs/experiments/m110_baseline_plan.md` §4) → cv-detection-expert / `configs/train_yolo11m_smoke.yaml`, `analysis/detection/m19_smoke_check.py`, `runs/logs/m19_smoke.log`, `runs/`
- [ ] M1-10 YOLO11m 기준선(AIHub 세션 그룹 5-fold, fold 평균±표준편차; 학습량 약 5배라 메인 세션 + long-running-task-monitor, 시작 전 cpu-scheduler/GPU 확인) — **epoch 결정(사용자, 2026-10-30)**: fold당 **50 epoch**(M1-9 smoke를 기다리지 않고 M1-7 30 epoch×4회 수렴을 근거로 결정 — 실행은 M1-9 통과 뒤(사용자 결정 2026-10-30), 집계 `analysis/detection/m110_epoch_budget.py`), `optimizer: AdamW`·`lr0: 0.00125`·`warmup_bias_lr: 0.0` 명시(Ultralytics 8.4.192 `optimizer: auto`는 학습 반복 10,000 초과 시 MuSGD로 바뀜 — fold 0–4는 52–57 epoch부터), `patience: 0`, 판정 가중치 `last.pt`(`best.pt`는 민감도용), 증강 강도 0·batch 64·workers 2, 예상 fold당 26–28분·5 fold 약 2.2 h(CPU 겹침 시 약 2.6 h), 실행 전 5 fold 누수 재단언. RF-DETR 비교(M4-3)도 같은 50 epoch 예산. **준비 완료(2026-10-30)**: 설계 `docs/experiments/m110_baseline_plan.md`(test는 M4-3까지 봉인, 모든 fold seed 0, momentum 0.9), 누수 재단언 통과(`analysis/detection/m110_leak_check.json`), 구현(`train_detector.py --on-existing`, `scripts/eval_detector.py`, `scripts/aggregate_folds.py`, 시험 54건), 러너 2개(`runs/logs/m110_train_runner.ps1`·`m110_eval_runner.ps1`) 생성·미실행 → cv-detection-expert / `docs/experiments/`

## M2 깊이 추정 구현·검증 (시작: M1-10 학습 중)
- [x] M2-1 전처리 모듈(dewow, median 배경 제거, 밴드패스, gain은 검출 갈래만) — 완료·사용자 승인 2026-10-30: `preprocess/filters.py`·`signals.py`, 단위 시험 `tests/preprocess/test_filters.py`, 합성 생성기 `tests/synthetic_bscan.py`. gpr-processing-expert 판정으로 dewow를 상자형에서 **가우스 σ=1/f_c**로 변경(유령 엽 방지, 명세 v1.1) → python-expert / `src/gpr_detection/preprocess/`
- [x] M2-2 영점 검출(지면결합 직접파, 공기결합 지표 반사, `air_single_event`) — 완료·사용자 승인 2026-10-30: `preprocess/timezero.py`(`exclude_traces`, `ZeroResult`, 창별 QC가 선택 사건 추적, 공기결합 단일 사건 메타 t0 라벨 `merged`, Δdw 적용 범위, `air_surface_late` σ_zero 0.3 ns), 시험 `tests/preprocess/test_timezero.py`·`test_zero_dewow_z1.py`(Z1–Z7, Z1 상자형 판별 상시). 지표 반사 +0.12 ns는 직접파 간섭 편향(보정 안 함, `docs/processing/surface_zero_shift.csv`), 명세 v1.2 → python-expert / `preprocess/timezero.py`
- [x] M2-3 꼭짓점·점 추출(첫 피크 씨앗) + RANSAC x²–t² + least_squares(R 고정) (M1-8의 `GridBox` 사용) — 완료·사용자 승인 2026-10-30: `depth/picks.py`·`hyperbola.py`·`config.py`, 단위 시험 U1–U3·U5–U8과 U4 거부(`tests/depth/test_hyperbola.py`). 판정 반영(명세 v1.3): 스케일 없는 입력은 우세 주기 픽셀 밴드패스(`preprocess/filters.py` `bandpass_px`, 잡음 속 씨앗 오인 방지), 공분산은 정상점만, RANSAC 거부 시 대체 초기값으로 계속 → python-expert / `depth/picks.py`, `depth/hyperbola.py`
- [x] M2-4 굴절 모델(공기층 h)과 v 결정 규칙 — 완료·사용자 승인 2026-10-30: `depth/refraction.py`(굴절 모델·`refraction_qc`), `velocity.py`(§7 결정표, M2-6b 속도 풀 자리), `result.py`(`DepthResult`·`target_class`·`no_velocity`), `estimate.py`(`estimate_depth`: 전처리→영점→픽→피팅→v→깊이·불확도·§10 실패 처리), 해석 합성 단위 시험(U4 깊이·εr 구간·메타 의심 2.7, U5, U7, 결정표, §10). 판정 반영(명세 v1.4): `refraction_qc`는 h_est·h_est−0.06 m εr **구간 판정**(h 민감도 +8 %/cm, Δh 0.06은 가정 — M2-5에서 확인), `air_gap="unknown"`은 곡률 금지·`no_velocity`, 메타 대체 깊이는 §6-4 꼭짓점 → python-expert / `depth/refraction.py`, `depth/velocity.py`, `depth/estimate.py`
- [x] M2-4b 합성 gprMax 로더(`io/synthetic.py`: `data/synthetic_gprmax/<id>.npy`+`.json` → `Radargram`, spec §1-1 표 첫 행) — 완료·사용자 승인 2026-10-30: `load_synthetic`(장비 메타만, 정답 시각·깊이·t0 후보는 넣지 않음, `include_eps`·`include_radius`), `load_synthetic_background`(`_bg.npy`), `load_synthetic_truth`→`SyntheticTruth`(`pipe_material` 포함), `list_synthetic_models`, 시험 `tests/test_synthetic_io.py`(모델 20개 전부 적재). 명세 v1.5 §11-2에 x축 비교 규약(json x = 0.4 m + i·dx)·M2-5 박스 공급 규칙 → python-expert
- [ ] M2-5 단위 시험 U1–U8, 합성 회귀 R1–R13 통과(R13 h_est 경로에서 `refraction_qc` 구간 판정의 Δh 0.06 m 가정도 확인)(시험별 허용치는 spec §11) + 합성 세트 전체 깊이 리포트 — 박스 공급: 정답 json 꼭짓점(`apex.trace_index`·`apex.envelope_peak_ns`) 주변 창(창 규칙은 gpr-processing-expert가 spec §11에 명시) → python-expert(구현), gpr-processing-expert(판정) / `tests/depth/`, `scripts/eval_depth_synthetic.py`
- [x] M2-6 `pipe_band` 깊이 규칙 명세(띠 상단 경계, v 출처) — 완료 2026-10-30 `docs/processing/spec_band_depth_v1.md`(띠 상단 = 꼭짓점 시각 규칙, 포락선 피크 기준, masked median 배경 제거, flat/sloped/curved 모델, v 출처 이웃 쌍곡선 → 메타 εr → 조사 단위 중앙값 → `no_velocity`; 사용자 결정 Q1–Q4 권고안). 후속 핸드오프: ADR-002 결정 8·ADR-004 결정 7 개정(add-architect), 3D 합성 P01–P09(electromagnetic-simulation-expert), 사교 교차 편향 1/sinθ는 M2-9 spec v2 → gpr-processing-expert / `docs/processing/`
- [ ] M2-6b `pipe_band` 깊이 구현·시험(스케일 있는 실측에서만 m 단위; AIHub는 스케일이 없어 픽셀·시간 단위만) → python-expert, gpr-processing-expert(판정)
- [x] M2-7 4TU analyze(01.1·04.1·010.14·011.14: 영점 사건 분리, 공기층 h, t0 두 값 라벨, 굴절 QC εr vs `eps_r_used`) → gpr-processing-expert / `docs/processing/fourtu_explore.md` — 결과: 영점은 단일 띠(공기층 ≈0), 메타 t0 두 값은 띠의 ± 엽, `eps_r_used` 9.0은 헤더 기본값과 같음
- [ ] M2-8 4TU 깊이 검증(활동 단위로 같은 관을 묶어 매칭, MAE, 제186조 ±30 cm 판정률, 상단/중심 두 가정, 무작위 깊이 기준선, **4TU로 튜닝하지 않음**) — 박스 공급: 손으로 고른 꼭짓점 씨앗(`fourtu_explore.md` §6) → python-expert(`scripts/eval_depth_4tu.py`), gpr-processing-expert(판정)
- [ ] M2-9 4TU 규약 정리(**M2-8 뒤 한꺼번에**, 사용자 결정 2026-10-29): `fourtu_explore.md` 제안 P1–P8로 spec v2, CLAUDE.md 깊이 규약·`depth_physics.md` §7-2·`readme_4tu.md` 주석·gpr-processing-expert 정의 갱신, 4TU dt 0.097 ns(현장 01, 92개) 반영, gprMax 공기층 0.02–0.05 m·오프셋 0.1–0.3 m 합성 추가 → gpr-processing-expert, electromagnetic-simulation-expert, 메인 세션

## M3 검출 고도화·교차 도메인
- [ ] M3-1 letterbox ↔ 원본 (트레이스, 샘플) 좌표 계약 구현·시험 → python-expert, cv-detection-expert
- [ ] M3-2 실측 B-scan을 검출 입력으로 바꾸는 경로(4TU·광저우대·GPR-SD npy → 전처리 → 영상) → python-expert
- [ ] M3-3 교차 도메인 평가(AIHub 학습 → 모로코·4TU·GPR-SD) → cv-detection-expert
- [ ] M3-4 GPR-SD 관로 17건 Label Studio 박스 라벨(서버는 메인 세션이 띄움, 라벨링은 사용자) → gpr-dataset-curator / `labeling/`
- [ ] M3-5 cleanlab 라벨 오류 후보 검토 → gpr-dataset-curator, cv-detection-expert
- [ ] M3-6 GPR-SD εr 해석 확정(dt 가설 A/B는 곡률 깊이 불변, εr만 바뀜)과 영점 점검 → gpr-processing-expert
- [ ] M3-7 광저우대 무라벨 추론·속도 추정 시험 → cv-detection-expert, gpr-processing-expert

## M4 Apache 모델 이전
- [ ] M4-1 RF-DETR 비율 유지 입력 경로 확인(입력 텐서 크기 스모크) → cv-detection-expert
- [ ] M4-2 RF-DETR(S/M 우선, 필요 시 N/L) 학습 설정·실행(gpr-detr env) — YOLO11m은 1채널 입력(M1-7 결정)이므로 RF-DETR 입력 채널(1채널 변환 또는 3채널 복제) 처리도 M4 시작 시 결정, RF-DETR 사전학습 가중치(Apache) 사용 여부는 M4 시작 시 결정(YOLO11m 기준선은 scratch) → cv-detection-expert
- [ ] M4-3 YOLO11m vs RF-DETR 비교 판정(M1-3과 같은 세션 그룹 5-fold 매니페스트, 기준: 전체 mAP50 fold 평균이 YOLO11m 대비 −5 %p 이내면 `이전 가능`, ADR-001) → cv-detection-expert / `docs/experiments/`
- [ ] M4-4 `detect/` 구현 교체(`Detector` 프로토콜 유지) — 백엔드 선택 설정(`detector.backend`)·추론 팩토리·백엔드별 학습 디스패처를 `detect/` 안에(ADR-001 후속 과제) → python-expert
- [ ] M4-5 ADR 갱신(이전 결과) → add-architect

## M5 통합 추론·배포
- [ ] M5-1 `gpr-detect infer`(입력 → 박스+깊이 JSON/PNG, 스케일 없는 입력은 "스케일 없음") → python-expert
- [ ] M5-2 ONNX 내보내기·onnxruntime-gpu 추론 일치 검증 → cv-detection-expert
- [ ] M5-3 README·사용 문서 → python-expert / docs
- [ ] M5-4 배포물 점검(AGPL 가중치 미포함, 데이터 미포함) → integrity-checker

## 사용자 조치·미해결
- [ ] U-1 AIHub 71327 「구축활용가이드」「데이터 설명서」 다운로드(https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71327) → 받으면 pdf-document-export / hwp-document-expert로 추출. **2026-10-29 결정: 나중에** — 그때까지 AIHub는 검출에만 쓰고 m 단위 깊이는 내지 않는다
- [ ] U-2 4TU 깊이 정답 기준점(관 상단/중심) 판단. **2026-10-29 결정: 두 가정 모두 보고**, 한쪽이 뚜렷이 잘 맞으면 그때 확정
- [ ] U-3 목표치 확정(합성 회귀 허용치는 spec §11로 확정, 4TU 판정률·검출 mAP 목표는 미정. 참고: AIHub 기준 EfficientDet mAP@0.5 0.8457). **2026-10-29 결정: M1-10 기준선과 M2-8 1차 검증을 본 뒤 정한다**
- [ ] U-4 배포 전 Ultralytics 라이선스 판단(사내 사용도 Enterprise 대상이라고 명시됨)
