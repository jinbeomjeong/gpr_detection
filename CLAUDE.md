# gpr_detection

GPR B-scan에서 매설 관로를 검출하고, 관로마다 지표에서의 수직 거리(깊이, m)를 추정하는 Python 프로젝트다.

## 목표와 결정 사항 (2026-10-29)
- 검출: Ultralytics **YOLO11m**으로 초안을 만든다. 성능이 확보되면 Apache/BSD 계열 모델로 옮긴다. YOLO 의존은 `src/gpr_detection/detect/` 안에만 둔다(`Detector` 프로토콜 뒤로).
- 깊이: **물리 기반 쌍곡선 피팅**을 쓴다. t(x) = (2/v)·√(d² + (x−x0)²)에서 v와 t0를 구하고 d = v·t0/2로 계산한다(공기결합 데이터의 예외는 아래 "깊이 규약"). 학습 회귀는 쓰지 않는다.
- 깊이는 물리 스케일(dt_ns, dx_m)이 있는 입력에서만 m 단위로 낸다. 스케일이 없는 JPG 입력에는 픽셀·시간 단위와 "스케일 없음" 표시만 낸다.
- 검출 클래스(2026-10-29 개정, AIHub 전수 집계 근거 `docs/datasets/aihub_71327.md`): 인덱스 순서대로 `0 pipe_hyperbola`(AIHub tpipe 횡배관, 모로코 pipe — B-scan에서 쌍곡선), `1 pipe_band`(AIHub lpipe 종배관 — 수평 띠), `2 cavity`, `3 manhole`(AIHub manh). 쌍곡선 피팅 깊이는 `pipe_hyperbola`에만 적용하고, `pipe_band`는 띠 상단 경계 시각으로 깊이를 추정하는 별도 규칙을 둔다(명세는 gpr-processing-expert).
- 모로코 데이터는 금지 증강 사본(회전 ±15°·탄성 변형, `images_index.csv`의 `aug_transform` 기준)을 학습·평가에서 뺀다. 남긴 사본은 원본 그룹(`split_groups.csv`) 단위 분할을 유지한다. 모로코 공동 하위셋은 원본 전체가 비율 무시 224×224로 늘려져 있으므로 공동 클래스 지표는 도메인별로 따로 본다.

## 환경
- conda env `gpr-dev`(Python 3.13, torch 2.14.1+cu130, ultralytics, opencv-python — headless 금지). env 전체 표는 아래 "env" 절. 실행 예: `conda run -n gpr-dev python -I scripts/<파일>.py`
- GPU: RTX 4070 SUPER 12 GB.
- gprMax 합성 데이터는 env `gprmax-analysis` / `gprmax-acceleration`에서 만든다.

## 데이터 (`C:\workspace\GPR_Data`, 읽기 전용 — 절대 수정하지 않는다)
| 데이터 | 경로 | 역할 |
|---|---|---|
| AIHub 261 | `261.3D 지표투과레이더 탐사 데이터\01-1.정식개방데이터` (zip, 압축 풀지 않고 읽음) | 검출 학습 주 데이터 |
| 모로코 | `모로코_관로공동GPR영상_가공본_261016\03_yolo_dataset` | 검출 보조·교차 시험 |
| 4TU | `4TU_네덜란드_매설관GPR_가공본_261015` (`05_radargrams_npy`, `01_메타데이터`) | 깊이 검증 |
| 광저우대 | `광저우대_중국_관로GPR_가공본_261016\05_radargrams_npy` | 미라벨 추론·속도 추정 시험 |
| GPR-SD | `GPR-SD_중국USTC_도로GPR_가공본_261016\05_radargrams_npy` | 현장 시험 |

파생 데이터는 `data/`에, 학습 결과는 `runs/`에 둔다(둘 다 git 제외).

## 분할 규칙 (누수 방지)
- AIHub: 제공된 Training/Validation 분할은 세션이 겹친다. XML의 `source`(세션) 단위로 다시 나누고, 같은 crop의 gain 0.25/0.5/1.0 사본은 같은 쪽에 둔다. 먼저 test 세션을 홀드아웃(약 15 %, 관로 종류별 이미지 비중 10–25 %)하고 나머지로 기준선을 포함한 모든 검출 학습을 **세션 그룹 5-fold**(ADR-003, `docs/datasets/dataset_spec_m1.md`)로 하며, 검출기 이전 판정은 전체 mAP50 fold 평균이 YOLO11m 대비 −5 %p 이내면 `이전 가능`(ADR-001)이다.
- 모로코: 제공 분할(원본 패치 그룹 단위)을 그대로 쓴다.
- 4TU: 현장(site 1–13) 단위로 나눈다. 깊이표는 활동 단위라서 검출 깊이 집합과 헝가리안 매칭으로 MAE를 구한다. t0는 `t0_sample_est`와 `t0_sample_act_median`을 둘 다 보고한다. 깊이 파라미터는 4TU로 튜닝하지 않고(합성 기본값) 13개 현장 전체를 시험에만 쓴다.

## 작업 운영
- 로드맵 단계나 체크리스트 단위 작업이 끝나면 먼저 **integrity-checker로 무결성 점검**(그 작업이 바꾼 파일 ↔ 관련 문서·코드·체크리스트)을 하고 지적을 반영한다. 그다음 git 커밋하고 바로 `origin`(**공개** 저장소, 주소는 git 설정에만 둔다)에 push한다(메시지: 한국어 요약 + 단계/항목). push 전에 비밀 정보·데이터 파일이 섞이지 않았는지 확인한다.
- 작업을 띄우기 전에 CPU·GPU·메모리 현황을 확인해 배치한다. 무거운 작업이 겹치거나 GPU 작업(학습 등)을 띄울 때는 compute-scheduler-expert 배치표(CPU 묶음 + GPU 순서·batch·worker 권고)를 따른다. GPU는 한 번에 한 계열 작업만.
- 사용자 대상 답변은 한국어.

## 프로젝트 전용 에이전트 (`.claude/agents/`, 2026-10-29)
| 에이전트 | 모델 / effort | env | 맡는 일 |
|---|---|---|---|
| gpr-processing-expert | claude-opus-5-5 / high | gpr-dev | 전처리·쌍곡선 피팅·깊이 추정 명세와 물리 판정(gprMax 정답, 4TU 제186조 판정률) |
| cv-detection-expert | claude-opus-5-5 / medium | gpr-dev, gpr-detr | 검출 학습 설정·평가·누수 점검·YOLO11m↔RF-DETR 이전 판정 |
| gpr-dataset-curator | claude-sonnet-5-5 / medium | gpr-dev, gpr-label | 변환 실행·분할 매니페스트·라벨 감사·Label Studio·데이터셋 카드 |

- 세 에이전트는 `src/`·`scripts/`를 고치지 않는다. 구현은 명세를 받아 python-expert가 한다.
- 리포트 처리: `메인 세션 권고`(30분 넘는 학습·변환, Label Studio 서버)는 메인 세션이 띄운다(30분 넘으면 long-running-task-monitor 규칙). `설치함:`이 오면 사용자에게 알리고, 함께 온 툴체인 표 갱신 줄을 해당 정의 파일과 메모리 `gpr-toolchain`에 반영한다. `위임 필요`는 이어서 넘긴다.
- 깊이 규약(2026-10-29 gprMax 검증, `docs/research/depth_physics.md`): 공기결합(4TU)은 지표 반사를 영점으로, v는 메타 εr 고정 또는 굴절 모델(곡률 피팅 금지). 반경은 피팅 미지수로 넣지 않는다.

## env
| env | 정의 | 용도 |
|---|---|---|
| gpr-dev | `environment.yml` | 프로젝트 실행·시험, 세 에이전트 공용 (torch 2.14.1+cu130, ultralytics 8.4.192, opencv-python 하나) |
| gpr-detr | `envs/gpr-detr.yml` | Apache 검출기 실험 (rfdetr 1.11.2, transformers 5.21.0, opencv-python-headless 하나, ultralytics 없음) |
| gpr-label | `envs/gpr-label.yml` | Label Studio 1.23.2 전용 |
| gprmax-acceleration | (기존) | gprMax 실행, electromagnetic-simulation-expert |

## 위임 규칙
- `.py` 작성·수정은 전부 **python-expert**에 맡긴다(메인 세션이 직접 고치지 않는다).
- gprMax 합성 데이터와 깊이 식의 물리 판정(t0, 공기층, 관경 보정)은 **electromagnetic-simulation-expert**가 한다.
- 아키텍처·ADR(`docs/architecture/`)은 **add-architect**가 맡는다.
- 선행 조사는 **web-research-expert**(모드는 전역 규칙대로 묻는다), 결과는 `docs/research/`에 둔다.
- .docx/.pdf/.hwp 원문은 docx-document-expert / pdf-document-export / hwp-document-expert로 추출한다.
- 여러 파일을 바꾼 뒤에는 **integrity-checker**로 문서↔코드 정합성을 감사한다.
- 30분 넘는 학습은 long-running-task-monitor 규칙을 따른다. 독립 CPU 작업을 3개 이상 동시에 돌릴 때는 compute-scheduler-expert로 배치표를 먼저 받는다.
- Python을 실행하는 에이전트를 부를 때는 매번 넣는다: "셸 heredoc과 `python -` 금지. 스크립트는 Write로 파일에 쓰고 `conda run -n gpr-dev python -I <파일>`로 실행. Python·pytest는 PowerShell 도구로 실행."
