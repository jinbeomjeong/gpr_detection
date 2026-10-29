# ADR-003: 검출 클래스는 4개로 두고, 데이터는 독립 그룹 키 단위로 분할한다

- 상태: Accepted (2026-10-29, 사용자 확정. AIHub k-fold와 4TU 튜닝 없음은 2026-10-29 사용자 답변으로 확정. 2026-10-29 개정: AIHub test 세션 홀드아웃 추가)
- 날짜: 2026-10-29
- 관련 드라이버: QA-5(누수 방지), QA-6(결정성), QA-3(4TU 깊이 보고), FR-1, CON-4 — 정의는 [README](README.md) §1
- 반복: 1회차 (체크리스트 M0-12)

## 맥락

- AIHub 71327 B-scan XML 전수 감사(`docs/datasets/aihub_71327.md`):
  - 제공 Validation 세션 1,599개가 **100 %** Training에도 있다. 파일명은 겹치지 않지만 gain 제거 크롭 키 7,260개가 겹친다.
  - 같은 크롭이 gain 0.25/0.5/1.0 세 벌이다(파일 74,838개 → 고유 크롭 27,680개).
  - `lpipe`는 수평 띠(w/h 중앙값 6.1), `tpipe`는 쌍곡선(w/h 1.7)이다. 쌍곡선 피팅은 tpipe에서만 성립한다.
  - pipe 독립 세션은 lpipe 13·tpipe 11개뿐이고 전부 NM 장비다.
  - XML `<filename>`이 멤버 이름과 다른 경우 1,870건, `<source>`가 이름 유도 세션과 다른 경우 224건.
- 모로코: 그룹 단위 분할(233그룹)이 제공된다. 증강 사본(회전 ±15°, 탄성)이 들어 있고, 공동 하위셋은 비율 무시 224×224 늘림이다(`docs/datasets/readme_morocco.md`, `docs/experiments/augmentation_policy.md` §5).
- 4TU: 13개 현장, 깊이표는 활동 단위이고 정답 깊이의 기준점(상단/중심)이 원본에 없다(spec_depth_v1 §0-2).
- 원래 계획(승인 계획서·CLAUDE.md 초판)은 `pipe` 하나로 합치는 3클래스였다. 2026-10-29 사용자 결정으로 4클래스가 됐고 CLAUDE.md "검출 클래스" 줄도 개정됐다(ROADMAP "확정된 설계 결정").

## 고려한 대안

### 클래스

| 대안 | 장점 | 단점 | 드라이버 충족 |
|---|---|---|---|
| **A. 4클래스: 0 `pipe_hyperbola`(tpipe·모로코 pipe), 1 `pipe_band`(lpipe), 2 `cavity`, 3 `manhole`** (채택) | 모양이 다른 두 관로 표현을 분리, 깊이 피팅 대상을 클래스로 선택 가능, 하위 클래스별 지표 보고 | pipe_band·pipe_hyperbola 표본이 더 작아짐 | FR-1 ○, FR-2 ◎ |
| B. 3클래스(`pipe` 병합) | 표본 많음, 원래 계획 | 띠와 쌍곡선이 한 클래스에 섞여 깊이 대상 필터가 불가, 띠 박스가 쌍곡선 피팅으로 들어감 | FR-2 ✕ |
| C. 관로 2클래스만(cavity·manhole 제외) | 단순 | 오검출을 줄이는 음성 클래스 정보 상실 | FR-1 △ |

### 분할

| 대안 | 장점 | 단점 | 드라이버 충족 |
|---|---|---|---|
| **A. 그룹 키 단위: AIHub `source` 세션 + gain 제거 크롭 키, 모로코 제공 그룹, 4TU 현장** (채택) | 누수 0을 단언으로 보장 | 분할 단위 수가 작아 분산이 큼(pipe 세션 11–13) | QA-5 ◎ |
| B. 제공 분할 그대로 | 작업 없음 | AIHub Validation 세션 100 % 누수 | QA-5 ✕ |
| C. 파일 단위 무작위 | 표본 균형 | gain 사본·같은 세션이 양쪽에 들어가 점수 부풀림 | QA-5 ✕ |
| D. A의 그룹 키 위에 AIHub 세션 그룹 k-fold(StratifiedGroupKFold) — **AIHub에 채택**(2026-10-29: test 세션 홀드아웃 뒤 train_pool에 적용) | 작은 세션 수(pipe 11–13)에서 분산 추정이 나음, 모든 세션이 한 번씩 평가에 쓰임 | 학습 비용 약 k배(k=5면 약 5배) | QA-5 ◎, QA-6 ◎ |
| E. AIHub 고정 분할 + 시드 3개 | 학습 비용 3배로 낮음 | 평가 세션이 2–3개뿐이라 분할 운에 좌우됨 | QA-6 △ — 기각 |

### 4TU 깊이 기준점

| 대안 | 장점 | 단점 |
|---|---|---|
| **A. 상단·중심 두 가정 모두 보고** (채택) | 근거 없는 선택을 피함, 한쪽이 뚜렷이 맞으면 그때 확정 | 보고 표가 두 배 |
| B. 상단으로 고정 | 단순, 피팅 정의와 일치 | 정답이 중심 기준이면 R만큼 계통 오차 |
| C. 중심으로 고정 | 시공 도면 관례일 수 있음 `[추정]` | R 미상 자료가 많음 |

## 결정

1. **클래스 4개**, ID 고정: `0 pipe_hyperbola`, `1 pipe_band`, `2 cavity`, `3 manhole`.
   - 매핑: AIHub `tpipe` → 0, `lpipe` → 1, `cavity` → 2, `manh` → 3. 모로코 `pipe` → 0, `cavity` → 2.
   - 깊이 피팅은 클래스 0에만 적용한다(ADR-002). 클래스 1 깊이는 M2-6 명세 뒤 결정.
   - 평가 지표는 클래스별로, 공동은 도메인(AIHub·모로코)별로 따로 보고한다(모로코 공동 비등방 늘림).
2. **AIHub 분할**: 그룹 키 = XML `<source>`(세션). gain 0.25/0.5/1.0 사본은 같은 쪽에 둔다(gain 접미 `_g<gain>_`를 뗀 크롭 키). 이미지 짝은 **zip 멤버 이름 기준**으로 맞추고, `<source>`와 이름 유도 세션이 다르면 매니페스트 별도 열에 기록한다. 제공 Training/Validation 구분은 무시하고 합친 뒤 다시 나눈다. 다시 나누는 방식(2026-10-29 개정, 사용자 결정): **먼저 AIHub test 세션을 홀드아웃**한다 — 분할 그룹(`<source>`·이름 유도 세션·crop 키의 union-find 연결 성분) 단위로 약 15 %, tpipe·lpipe 세션 각 2개 이상, 관로 종류별 test 이미지 비중 10–25 %를 만족하는 첫 시도를 채택(seed 20261029, attempt 0, tpipe 10.8 %·lpipe 12.9 %). **나머지(train_pool)를 세션 그룹 5-fold**(`StratifiedGroupKFold`, 주 클래스 층화)로 나눠 YOLO11m 기준선(M1-10)과 RF-DETR 비교(M4-3)를 포함한 모든 학습·검증에 같은 fold 매니페스트를 쓴다. test는 최종 1회 측정에만 쓴다. 구체 수치와 파일은 `docs/datasets/dataset_spec_m1.md`.
3. **모로코**: 제공 그룹 분할(train/val/test)의 그룹 단위를 유지한다. M1-4 결정(2026-10-29)으로 모로코는 교차 도메인 평가 전용이라 분할 없이 하나의 평가 집합(그룹당 1장)으로 쓰고, 제공 분할은 참고 열 `split_src`로만 보존한다. 회전·탄성 사본은 `aug_transform` 기준으로 학습·평가 모두에서 제외한다(제외 목록은 gpr-dataset-curator가 매니페스트와 함께 남긴다).
4. **4TU**: 현장(site 1–13) 단위로 나눈다. 깊이 파라미터는 4TU로 **튜닝하지 않는다** — 합성에서 정한 기본값으로 13개 현장 전체를 시험에만 쓴다(그래서 깊이 검증용 dev/test 분할은 없다. 현장 단위 분할은 검출 교차 도메인 평가 M3-3에만 쓴다). 깊이 비교는 활동 단위 헝가리안 매칭으로 MAE를 구한다. t0는 `t0_sample_est`·`t0_sample_act_median` 둘 다, 정답 기준점은 상단·중심 두 가정 모두 보고한다.
5. **누수 단언**: 분할 매니페스트(`data/splits/*.csv`)를 만들 때 분할 간 세션·크롭 키·그룹·현장 교집합이 0이 아니면 빌드를 실패시킨다. imagehash 근접 중복 감사(M1-5)를 함께 한다.
6. **시드**: fold 분할 시드는 매니페스트에 기록한다. 검출 지표는 train_pool fold 평균±표준편차로 보고하고, AIHub test는 최종 1회만 측정한다. 학습 시드는 fold마다 고정해 기록한다. 깊이 RANSAC은 `random_state=0`.

## 결과

얻는 것
- 검증 점수가 세션 누수로 부풀지 않는다(QA-5). 매니페스트가 바이트 단위로 재현된다(QA-6).
- 깊이 대상이 클래스 ID 하나로 정해져 `pipeline`의 필터가 단순하다.
- 4TU 깊이 결과가 기준점 가정에 따라 어떻게 바뀌는지 드러난다.

잃는 것 / 비용
- pipe 평가의 독립 단위가 세션 11–13개뿐이라 지표 분산이 크다. 단일 장비(NM) 편향이 남는다 → 모로코·4TU·GPR-SD 교차 도메인 평가(M3-3)로 보완.
- 모로코 회전·탄성 사본 제외로 YOLO 포함 이미지가 2,239장 → 약 1,519장(720장 제외)으로 준다. 남는 것도 이동·반전·잡음 등 다른 증강 사본이라 독립 표본으로 세지 않는다(원본 285장은 라벨이 없어 학습에 쓰이지 않음).
- AIHub 검출 학습 비용이 기준선부터 **약 5배**(5-fold)다. test 홀드아웃으로 관로 학습 데이터가 tpipe 1,357장·lpipe 3,576장으로 준다. GPU 1장(CON-2)에서 순차로 돌려야 하므로 M1-10·M4-3 일정이 길어진다.
- 4TU를 튜닝에 쓰지 않으므로 4TU 특성(예: 영점 규약)에 맞춘 파라미터 조정은 결과를 본 뒤 규약 개정으로만 한다.
- 클래스가 늘어 YOLO11m 기준선과 AIHub 공개 기준(EfficientDet mAP@0.5 0.8457, 클래스 정의 다름)의 직접 비교는 어렵다.

위험
- `<source>`와 이름 유도 세션이 다른 224건의 소속 판단이 틀리면 누수가 남을 수 있다 → 두 키 모두로 단언.
- 5-fold 학습 한 벌이 30분을 넘기 쉽다 → 메인 세션이 띄우고 long-running-task-monitor 규칙을 따른다. tpipe 세션이 11개라 fold당 평가 세션이 2–3개뿐이라서 fold 간 편차가 클 수 있다(층화로 완화).
- 클래스 ID·매핑의 코드 기준은 `src/gpr_detection/io/classes.py`(`DetectionClass`, 2026-10-29 구현됨, 커밋 b0311ed)다. 이 ADR과 어긋나면 integrity-checker 감사로 잡는다.

후속 과제
- (완료 2026-10-29) M1-3 변환·분할 구현·실행·감사, M1-4 모로코 평가 세트(`morocco_eval_v1`, 233장), M1-5 모로코 감사(`audit_morocco_eval_v1.md`, 통과).
- (완료 2026-10-29) M1-3 매니페스트에 fold 열(0–4, test=-1)과 test·fold 누수 단언(`docs/datasets/dataset_spec_m1.md`, `audit_aihub_yolo_v1.md`).
- (완료 2026-10-29) ROADMAP M1 완료 기준과 CHECKLIST M1-10을 "세션 그룹 5-fold"로 고쳤다.

## 근거 문서

- [`docs/datasets/aihub_71327.md`](../datasets/aihub_71327.md) §1–§5 — 세션 중복, gain 사본, lpipe/tpipe 형상, 이름 불일치
- [`docs/datasets/readme_morocco.md`](../datasets/readme_morocco.md) §5–§6 — `aug_transform`, 분할 그룹
- [`docs/datasets/readme_4tu.md`](../datasets/readme_4tu.md), [`codebook_4tu.md`](../datasets/codebook_4tu.md) — 현장, 깊이표, t0 메타
- [`docs/experiments/augmentation_policy.md`](../experiments/augmentation_policy.md) §5 — 모로코 사본 제외 결정
- [`docs/processing/spec_depth_v1.md`](../processing/spec_depth_v1.md) §0-2, §4-3 — 기준점, 4TU 메타 영점
- [`docs/ROADMAP.md`](../ROADMAP.md) "확정된 설계 결정", [`docs/CHECKLIST.md`](../CHECKLIST.md) M1-3–M1-5, M2-8, U-2
