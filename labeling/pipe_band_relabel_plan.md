# pipe_band(AIHub lpipe) 수동 재라벨 계획 (2026-10-31, 계획·준비 단계)

근거 감사: [`docs/datasets/audit_pipe_band_shape_v2.md`](../docs/datasets/audit_pipe_band_shape_v2.md). 이 단계에서는 **Label Studio 서버에 프로젝트를 만들거나 가져오지 않았고 API도 호출하지 않았다.** `data/aihub_yolo_v1`·`data/splits`는 읽기만 했다.

## 사용자 결정 (2026-10-31) — 아래 §0의 권고보다 우선

- **D1: 자동 병합 먼저(c)**, 수동 하이브리드 검수(a)는 v1↔v2 비교 결과(`docs/experiments/band_merge_v2_compare.md`)를 본 뒤 다시 결정한다.
- **D2: 조건부 32 px**(인접·y 겹침 ≥ 50 %·간격 ≤ 32 px·높이 ≤ 1.5배). 자동 전용 산출물도 16 px가 아니라 32 px로 만들었다: `data/aihub_yolo_v2_auto32/`(`scripts/build_yolo_v2_merged.py`, 병합 함수 `src/gpr_detection/data/band_merge.py`, 기본 G = 32). 아래 본문의 `v2_auto16`·`aihub_yolo_v2_auto16`(G = 16)은 결정 전 예시다.
- **D3: test 포함.** 자동 병합본에서는 test도 같은 규칙을 적용하고 v1 test 정답은 `data/aihub_yolo_v2_auto32/test_v1/`로 보존했다.
- **D4: 파일럿만 이중 라벨**(수동 검수를 하게 될 때).
- 자동 병합본 매니페스트의 `band_relabel` 값은 `auto_merged`/`auto_unchanged`/`not_in_scope`다(§4의 `relabeled`/`kept_v1_unclear`는 수동 검수본용). 이름 `data/aihub_yolo_v2/`는 수동 검수본용으로 남겨 둔다. §8-1·2(`make_relabel_tasks.py`, `build_yolo_v2_from_relabel.py`)는 수동 검수 결정까지 **보류**한다. 감사 v1 §145의 `--merge-band-gap` 명세(기본 0, 출력 `data/aihub_yolo_v2/`)는 별도 스크립트(기본 32, 출력 `aihub_yolo_v2_auto<gap>`)로 대체됐다.

## 0. 사용자 결정이 필요한 항목 (선택지와 권고)

| # | 결정 | 선택지 | 권고 |
|---|---|---|---|
| D1 | 재라벨 방식 | (a) **하이브리드**: 자동 병합 박스를 사전 주석으로 깔고 사람이 전수 검수(약 7.4 h, 6–11 h) / (b) 전면 재라벨: 사전 주석 없이 새로 그림(25–38 h) / (c) 자동 병합만(사람 0 h, `v2_auto16`) | **(a)**. (c)는 시험용 비교 기준(자동 병합 대 사람 검수)으로만 따로 만든다 |
| D2 | 병합 규칙(사전 주석용) | 8 px 유지 / 16 px / **조건부 32 px**(인접·y 겹침 ≥ 50 %, 간격 ≤ 32 px) | **조건부 32 px**. > 32 px 인접 16쌍은 사람이 정한다. 근거: 시각 검토 30장에서 같은 층 22·불확실 8·별개 0(감사 v2 §3) |
| D3 | test 세션(2개, 200 crop, 530장) 포함 | 포함(마지막 웨이브, 규칙 고정 뒤) / 제외(test는 v1 유지) | **포함**. 이유는 §1-2 |
| D4 | 라벨러 | (a) 사용자 1인 / (b) 사용자 + 두 번째 사람(파일럿 40장만 이중 라벨) / (c) 두 사람이 나눠서 | **(b)**. 파일럿 일치도로 지침을 확정한 뒤 본 작업은 1인이 하는 편이 일관된다. (c)는 세션 스타일 차를 다시 만들 위험 |
| D5 | Label Studio 로컬 파일 서빙 | 서버가 이미 `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true`, `DOCUMENT_ROOT=C:\workspace\gpr_detection\data`로 떠 있는지 확인 / 아니면 서버 재시작(메인 세션) 또는 이미지 업로드 방식 | 사용자가 띄운 서버의 환경변수를 큐레이터가 확인할 수 없다(API 호출 금지). **사용자 확인 필요**. 아니면 재시작 권고(§5-1) |
| D6 | 교차 crop 일관화 | crop별 독립 라벨 + 지침("보이는 구간 전체, 가장자리까지")으로 일치 / 절대 좌표 자동 병합 | **지침으로 일치**. 자동 병합 모의는 전폭 박스 41 %·높이 팽창 p95 1.53배라 띠 의미를 바꾼다(감사 v2 §5) |
| D7 | 평가 | 기존 IoU 0.45·mAP50 유지, band 보조 지표 추가(결정 4) — 이 계획 범위 밖 | cv-detection-expert. v1 정답 대 v2 정답 두 가지로 같은 fold 비교 권고 |

## 1. 대상·우선순위

### 1-1. 대상

v1 pipe_band crop **1,518장(gain 사본 포함 파일 4,106장)** 전부. 사람이 보는 단위는 crop 1장(gain 1.0 대표)이고, 결과는 같은 crop_key의 gain 0.25/0.5/1.0 사본에 복제한다(v1에서 6,082개 사본 2장 이상 crop의 라벨이 전부 같았음을 감사 v1에서 확인). 대상 밖: 공동·맨홀·쌍곡선 박스(v1 그대로, 라벨 설정에는 보이기만 한다). pipe_band 라벨이 없는 crop은 이번 범위가 아니다(v1 전체에서 band가 있는데 라벨이 없는 crop은 이 계획으로 찾지 못한다 — 한계).

### 1-2. 웨이브

난이도 층(tier): A = 같은 깊이에서 x로 끊긴 쌍이 있는 crop, B = 다박스지만 끊김 없음(적층 등), C = 박스 1개. 파일럿 40장(W0)은 test를 뺀 층화 표본(A 20·B 6·C 14, 세션 라운드로빈, 시드 20261031).

| 웨이브 | 내용 | crop | 파일 | A | B | C | 예상 검수 시간(H1) |
|---|---|---|---|---|---|---|---|
| W0 | 파일럿(규칙 보정, 이중 라벨) | 40 | 108 | 20 | 6 | 14 | 0.26 h × 2회 |
| W1 | 조각 많은 세션: `(11_10_46)` f0, `(12_33_02)`·`(14_16_20)` f4 (파일럿 제외) | 573 | 1,561 | 303 | 45 | 225 | 3.5 h |
| W2 | 나머지 train_pool 세션(f0 `(13_43_06)`, f1 `(14_26_27)`, f2 두 세션, f3 네 세션) | 705 | 1,907 | 59 | 32 | 614 | 2.2 h |
| W3 | test 세션 `(12_47_20)`·`(11_03_51)` (규칙 고정 뒤) | 200 | 530 | 84 | 1 | 115 | 1.0 h |
| 합계 | | 1,518 | 4,106 | 466 | 84 | 968 | 7.4 h(가장자리 보정 포함) |

세션·fold별 상세: `analysis/data/out/pipe_band_v2_workload_tiers.csv`(로컬 전용), 작업 목록 색인 `labeling/pipe_band_relabel/local/tasks_index.csv`.

우선순위 이유: 조각 라벨 crop 466장 중 W1에 303장(65 %)이 몰려 있고, M1-7 조각 FP 223개 중 216개가 W1의 `(11_10_46)`이다. W2는 단일 박스 비중이 높아(614/705) 짧은 검수가 대부분이다.

### 1-3. test 포함 권고와 근거

- **포함을 권고한다.** train_pool만 v2 규약으로 바꾸고 test를 v1로 두면, 모델은 긴 띠로 학습하는데 시험은 조각 정답으로 채점해 규약 불일치가 점수에 섞인다. 반대로 test만 v1이면 v1 대 v2 비교가 학습 효과와 채점 규약 효과를 분리하지 못한다.
- test 세션도 조각 비율이 40 %·45 %로 train_pool 큰 세션과 같은 수준이다(감사 v2 §2).
- **누수·편향 방지**: (i) 규칙은 W0–W2에서 고정하고 test는 마지막에 같은 규칙으로만 라벨한다, (ii) 사전 주석은 v1 정답에서 결정적으로 만든 병합 박스이고 **모델 예측을 쓰지 않는다**(test 모델 점수를 본 적이 없는 상태에서 진행, 라벨러에게 예측을 보여 주지 않는다), (iii) v1 test 라벨은 `test_v1`으로 보존해 v1·v2 두 정답 모두로 보고한다. test 평가가 이미 이뤄졌다면(M1-10 보류 중이라 아직 최종 test는 없음) 그 결과를 라벨러에게 알리지 않는다.
- 위험: test 세션이 2개뿐이라 band test 표본은 어차피 작다(301 박스). 재라벨로 박스는 약 −26 %가 된다.

## 2. 라벨 지침 초안

대상 클래스는 `pipe_band`뿐이다. 영상은 256×256 gain 1.0(대표) 한 장이다. 한 crop의 박스는 사전 주석을 출발점으로 수정한다.

1. **한 층 = 박스 하나.** crop 안에서 보이는 같은 수평 반사 줄(층)은 연속 구간 전체를 박스 하나로 그린다. 같은 층을 여러 박스로 나누지 않는다.
2. **끊김 판정.** 아래 중 하나일 때만 박스를 나눈다.
   - 반사가 육안으로 사라진 구간이 **32 px(crop 폭의 1/8)를 넘는다.**
   - 두 구간의 깊이 차가 박스 높이의 **절반을 넘는다**(박스를 위·아래로 따로 그리면 y 겹침 < 50 %).
   - 위층 박스 아래의 다른 행이다(별개 박스).
   그 외(간격 ≤ 32 px, 깊이 차 ≤ 절반)는 하나로 잇는다. 판정이 안 서면 `unclear` 플래그를 켜고 사전 주석을 그대로 둔다(그 crop은 v1 박스를 유지).
3. **상·하 경계.**
   - 상단 = 눈에 보이는 **첫 반사 파형의 시작(위쪽 가장자리)**. 파형 시작보다 안쪽(아래)으로 자르지 않는다. 깊이 명세(`docs/processing/spec_band_depth_v1.md` §3-3)는 박스 위로 한 주기(스케일 없으면 max(2, 0.25·박스 높이) px)를 더 보고 포락선 첫 피크를 잡으며, 라벨 박스가 눈에 보이는 위쪽 경계를 따른다고 가정한다.
   - 상단은 지표 반사(맨 위 검고 흰 반복 줄무늬 군) **아래**에 둔다. 직접파·지표 클러터를 박스에 넣지 않는다.
   - 하단 = 그 반사 묶음의 마지막 위상. 후미 잔향은 넣지 않는다.
   - 목표 높이 12–24 px(v1 p50 16, p95 21, 최대 32). 층이 기울어 높이가 24 px를 넘으면 구간을 나눈다.
4. **crop 가장자리.** 층이 crop 좌우 가장자리까지 이어지면 박스를 **x = 0 또는 x = 256까지** 그린다(꼬리를 남기지 않는다). 이는 M1-7 정답 없는 곳 FP 25개 중 19개가 "정답 박스 ↔ 가장자리 사이 꼬리"였던 것에 근거한다. 가장자리 바깥은 상상하지 않는다. 이웃 crop은 보지 않는다(규칙이 같으면 겹친 구간이 일치한다).
5. **너무 짧은 조각.** 폭 16 px 미만 박스는 만들지 않는다(가장자리에 닿은 8 px 이상 꼬리는 허용). 층이 아니라 점·곡선(쌍곡선 모양)이면 pipe_band로 그리지 않는다. 클래스 0(쌍곡선)·공동·맨홀 박스는 건드리지 않는다.
6. **라벨 없는 층·없는 띠.** 사전 주석에 없는 층이 뚜렷하면 추가하고, 사전 주석 박스가 띠가 아니면 삭제한 뒤 `no_band_here`를 켠다. 박스 0개로 끝나는 crop은 그대로 v2에 반영한다(해당 crop의 v1 band 박스 삭제). `no_band_here`가 파일럿에서 흔하면 지침을 보정한다.
7. **플래그.** `unclear`(판정 불가, v1 유지), `no_band_here`, `check_neighbor_crop`(가장자리 처리 의문). 메모는 선택.
8. **gain 사본은 라벨하지 않는다.** 변환이 crop_key 단위로 복제한다.

핫키: 1 = pipe_band, 2 = pipe_hyperbola, 3 = cavity, 4 = manhole, u/n/c = 플래그.

## 3. 작업량 추정

가정(측정값 아님, 파일럿 40장 실측으로 보정): 사전 주석 검수 시 A층 crop 30 s, B층 40 s, C층 8 s, 가장자리 접촉 C 95장은 +17 s. 전면 재라벨은 crop당 60–90 s(박스 평균 1.5개 + 끊김·가장자리 판단). 범위 = 가정값 ×0.8–×1.5.

| 안 | crop | 시간(점추정 → 범위) | 비고 |
|---|---|---|---|
| 하이브리드 전수 검수(D1-a) | 1,518 | **7.4 h → 6–11 h** | A 466×30 s = 3.9 h, B 84×40 s = 0.9 h, C 968×8 s = 2.2 h, 가장자리 보정 0.45 h |
| 하이브리드 표본(A+B+C 가장자리만) | 645 | 5.5 h | C 단일 박스 873장이 v1 그대로라 가장자리 꼬리·적층 누락이 남는다. M1-7 조각 띠 FP crop 164개 중 79개가 단일 박스 crop이므로 **비권고** |
| 전면 재라벨(D1-b) | 1,518 | 25–38 h | 사전 주석 효과 없음(검수보다 3–5배) |
| 자동 병합만(D1-c) | 0 | 0 h | `v2_auto16`(16 px, 자동 전용 권고값). 사람 검수 효과를 재는 기준선 |

고정 비용(사용자 시간): Label Studio 프로젝트 설정·파일럿 이중 라벨·지침 보정 약 1.5–2 h. 파일럿 40장을 두 사람이 하면(D4-b) 두 번째 사람 0.5 h 추가. 총 약 **9–13 h**(하이브리드, 하루 반~이틀). 변환·누수 단언·갱신은 python-expert/큐레이터(사람 시간 없음, 수 분).

파일럿 합격 기준(제안): 두 라벨러(또는 한 라벨러 두 회차) 사이 crop당 band 박스 수 일치 ≥ 80 %, IoU ≥ 0.5 매칭 F1 ≥ 0.8. 미달이면 지침(끊김 판정, 경계)을 고쳐 W0를 다시 한다. 파일럿 시간 실측이 점추정보다 1.5배 넘게 길면 D1을 다시 묻는다.

## 4. 산출물 버전 계획

- v1(`data/aihub_yolo_v1`, `data/splits/aihub_yolo_v1.csv`)은 **그대로 보존**한다. 새 폴더 `data/aihub_yolo_v2/`와 매니페스트 `data/splits/aihub_yolo_v2.csv`를 만든다. 선택으로 `data/aihub_yolo_v2_auto16/`(자동 병합만, D1-c).
- v2 구조는 v1과 같다: `images/`(v1과 같은 파일, 하드링크 우선·같은 볼륨이 아니면 복사), `labels/`, `fold0..4_images.txt`·`fold0..4.yaml`·`test_images.txt`·`test.yaml`·`summary.json`. 클래스 인덱스 `0 pipe_hyperbola, 1 pipe_band, 2 cavity, 3 manhole` 고정.
- **분할·매니페스트는 v1과 동일**: 같은 행 집합(`file_name`), 같은 `subset`·`fold`·`session`·`session_from_name`·`crop_key`·`split_group`·`seed`(20261029)·`test_attempt`. 새 열만 추가한다: `band_relabel`(`relabeled`/`kept_v1_unclear`/`not_in_scope`), `band_boxes_v1`, `band_boxes_v2`.
- 갱신되는 라벨은 클래스 1 박스뿐이다. 클래스 0·2·3 박스는 v1 바이트와 같아야 한다.
- **누수 재단언(v2 생성 뒤 매니페스트를 다시 읽어 독립 계산)**: (1) v2 행 집합·`fold`·`subset`이 v1과 완전히 같음, (2) test ↔ train_pool, fold 상호 간 `session`·`session_from_name`·`crop_key`·`split_group` 교집합 0(64개 조합), (3) 같은 crop_key의 gain 사본 라벨이 v2에서도 동일, (4) `fold{k}_images.txt` 합집합 = 매니페스트, 학습 목록에 검증 fold 파일 0건, (5) 클래스 0·2·3 라벨 불변, (6) 클래스 1 박스 합계(v1 2,279 → v2)·crop 수와 변경 목록을 `relabel_log.csv`로 남김.
- 문서 갱신(결정 뒤): `docs/datasets/dataset_spec_m1.md`에 v2 절, `docs/datasets/cards/aihub_71327.md`의 박스 규약·`aihub_yolo_v2` 항목, `audit_aihub_yolo_v1.md` 연결. 이 단계에서는 고치지 않았다.

## 5. Label Studio 구성안

### 5-1. 전제 (서버는 사용자가 이미 띄움)

- 서버 `http://127.0.0.1:8080`, 버전 1.23.2, 사용자 로그인 완료(메인 세션 전달). 이 단계에서 접속·API 호출·가져오기를 하지 않았다.
- 작업 JSON의 이미지 주소는 `/data/local-files/?d=aihub_yolo_v1/images/<URL 인코딩한 파일명>`이다. 이 주소가 열리려면 서버 프로세스에 다음이 있어야 한다.
  - `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true`
  - `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=C:\workspace\gpr_detection\data` (`images` 경로의 상위)
  사용자가 띄운 서버가 이 환경변수를 갖는지는 **확인하지 못했다**(D5). 가져온 뒤 첫 이미지가 안 보이면(404) 서버를 위 환경변수로 다시 띄워야 한다(서버 종료·재시작은 사용자/메인 세션 몫). 재시작을 피하려면 대안은 이미지를 가져오기 화면에서 직접 업로드하는 방식이다(crop 1,518장 × 수십 KB, 작업 JSON의 `image` 값을 업로드 경로로 바꿔야 해서 python-expert 수정 필요).
- 사용자가 지정한 브라우저(웨일)로 접속한다. 서버는 localhost 전용을 유지한다.

### 5-2. 프로젝트 구성 (사용자 결정 뒤 진행)

| 항목 | 값 |
|---|---|
| 프로젝트 이름 | `pipe_band_relabel_v2` |
| 라벨 설정 | `labeling/pipe_band_relabel/label_config.xml` (Settings → Labeling Interface → Code에 붙여넣기). 클래스 순서 `<Label category="0..3">` = YOLO 인덱스. Choices 3개, TextArea 1개 |
| 가져오기 | 파일럿: `labeling/pipe_band_relabel/local/tasks_pilot.json`(40), 본 작업: `tasks_all.json`(1,518, 웨이브 순). 둘 다 **로컬 전용**(AIHub 라벨 좌표 포함, `.gitignore`에 `labeling/pipe_band_relabel/local/` 추가함) |
| 사전 주석 형식 | 작업마다 `predictions[0]`(`model_version: v1_merge_g32`), 결과는 `rectanglelabels`(`from_name: label`, `to_name: image`, `original_width/height: 256`, `x,y,width,height`는 % 좌표). 클래스 1은 조건부 32 px 병합 박스, 클래스 0·2·3은 v1 박스 그대로. Settings → Annotation에서 prelabel에 이 model_version을 쓰도록 설정(1.23.2 화면 문구는 서버에서 확인). 안 되면 `annotations`로 가져와도 된다(제출 수가 달라지지 않음) |
| 작업 메타 | `crop_key, file, session, fold, tier, wave, n_files, v1_band_boxes, pre_band_boxes`(Data Manager 필터·정렬용) |
| 내보내기 | Data Manager Export → **JSON(전체)** (YOLO 내보내기는 플래그·메모·원본 파일명이 없어 쓰지 않는다). 저장 `labeling/pipe_band_relabel/local/export_<날짜>_<웨이브>.json` |
| 변환 경로 | export JSON → (§8 변환 스크립트) → `data/aihub_yolo_v2/labels/*.txt`(YOLO 정규화, gain 사본 복제) → 매니페스트·누수 재단언 |

### 5-3. 작업 목록 요약 (실제 생성함)

`analysis/data/make_pipe_band_relabel_tasks.py`가 만들었다(시드 20261031, 병합 G = 32 px). 작업 1,518개 = 파일럿 40 + W1 573 + W2 705 + W3 200. 사전 주석의 pipe_band 박스는 1,696개(v1 2,279개 대비 −25.6 %). 기존 가져오기 파일 위치는 위 표의 `local/`이다. 이 스크립트는 초안이며 운용용 변환 코드는 §8로 python-expert에 넘긴다.

### 5-4. 진행 절차 (결정 후)

1. D1–D6 결정, D5 서버 환경변수 확인. 2. 프로젝트 생성·설정 붙여넣기·파일럿 가져오기(1장 열어 이미지 로딩 확인). 3. 파일럿 40장 이중 라벨, 일치도·시간 측정, 지침 보정. 4. W1 → W2 가져오기·라벨(웨이브마다 export). 5. 규칙 고정 선언 뒤 W3(test). 6. 변환·누수 재단언·감사(큐레이터). 7. cv-detection-expert가 v1 대 v2 같은 fold 비교.

## 6. 위험·한계

- 시각 검토·FP 판정이 큐레이터 1인 육안이다. 파일럿 이중 라벨로 규칙을 재확인한다.
- v1은 균형화된 부분집합이라 pipe_band crop 1,518장이 전체 lpipe(6,193 객체)의 일부다.
- 사전 주석의 병합 규칙이 라벨러 판단을 끌 수 있다(앵커링). 파일럿에서 사전 주석 없는 판 5장과 비교해 편향을 점검하는 것을 권고한다(선택).
- 라벨러 피로로 C층(단일 박스) 검수가 형식적으로 흐를 수 있다. 8 s/crop 가정이 이를 반영하므로 파일럿 실측으로 확인한다.

## 7. 파일 목록 (이 계획이 만든 것)

| 파일 | 설명 | 공개 |
|---|---|---|
| `labeling/pipe_band_relabel/label_config.xml` | 라벨 설정 초안 | 공개 가능 |
| `labeling/pipe_band_relabel/local/tasks_all.json`, `tasks_pilot.json`, `tasks_index.csv` | 가져오기 파일·색인 | **로컬 전용**(.gitignore) |
| `analysis/data/make_pipe_band_relabel_tasks.py` | 작업 목록 초안 생성 | 공개 가능 |
| `analysis/data/audit_pipe_band_v2_*.py` | 감사 스크립트 7개 | 공개 가능 |
| `analysis/data/out/pipe_band_v2_*.csv` | 중간 CSV(박스 좌표) | 로컬 전용(`pipe_*.csv` 규칙) |
| `analysis/data/out/pipe_band_v2_*_report.txt` | 집계 보고(수치만, 좌표·파일명 없음) | 공개 가능 |
| `docs/datasets/figures/pipe_band_shape/pipe_band_v2_*.png` | 몽타주 5장(AIHub 영상 포함) | 로컬 전용 |

## 8. 위임 필요(python-expert) — 구현 명세

1. **`scripts/make_relabel_tasks.py`** (초안 `analysis/data/make_pipe_band_relabel_tasks.py`를 운용화): 입력 `data/aihub_yolo_v1`·`data/splits/aihub_yolo_v1.csv`, 옵션 `--merge-band-gap`(기본 32), `--pilot-size 40`, `--seed 20261031`. 병합: 같은 이미지·클래스 1, y 겹침 ≥ 0.5(작은 박스 높이 기준), **인접**(사이에 같은 행 박스 없음), x 간격 ≤ G, 가까운 쌍부터 반복. gain 사본에서 같은 결과. 출력은 tasks JSON(`data` 메타 + `predictions`)과 `tasks_index.csv`. 단위 시험: 합성 박스(간격 8/32/33 px, 단차 y 겹침 49/50 %, 사이 박스 있음/없음).
2. **`scripts/build_yolo_v2_from_relabel.py`**: 입력 Label Studio JSON export 1개 이상, v1 폴더, 매니페스트. 규칙:
   - 작업당 취소되지 않은 가장 최근 annotation 하나만 쓴다. 클래스 1(`pipe_band`) 사각형만 취하고 % → 픽셀(`x·original_width/100`) → [0,256]로 clip → 폭·높이 < 3 px 제거 → YOLO 정규화(소수 6자리). 정수 반올림하지 않는다.
   - 클래스 0·2·3은 v1 라벨을 그대로 복사한다. export의 해당 클래스 박스가 v1과 0.5 px 넘게 다르면 로그만 남기고 v1을 유지한다.
   - `unclear` 플래그 또는 annotation 없는 작업은 v1 라벨을 유지하고 `band_relabel = kept_v1_unclear`.
   - 결과를 crop_key의 모든 gain 사본에 복제한다(파일명은 매니페스트 기준).
   - 출력 `data/aihub_yolo_v2/`(§4 구조, 이미지는 하드링크 우선), `data/splits/aihub_yolo_v2.csv`, `relabel_log.csv`(crop_key, v1/v2 박스 수, 병합·삭제·추가 수, 플래그), `summary.json`.
   - **누수 재단언 §4 (1)–(6)**을 스크립트 끝에서 매니페스트를 다시 읽어 실행하고 결과를 stdout 마지막에 한 줄 JSON으로 낸다. 실패 시 종료 코드 1.
   - 시험: LS export 픽스처(정상·취소·플래그·중복 annotation), 클래스 0·2·3 불변, gain 사본 일치.
3. **`--merge-band-gap` 옵션(기존 v1 감사 명세)**은 1과 같은 병합 함수를 공유해 `data/aihub_yolo_v2_auto16/`(G = 16)를 만드는 데 쓴다. 인접 조건과 높이 보호(합집합 높이 ≤ 1.5배; 현재 막히는 쌍 0)를 포함한다.
