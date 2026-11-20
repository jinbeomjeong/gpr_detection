# M3-3 ③ E3x 설계 v0.1 — 검출 쪽 질문 5개 답변 (cv-detection-expert)

- 작성: cv-detection-expert(review 모드, 코드만 읽음), 2026-11-20. 서브에이전트 보고를 메인 세션이 저장했다.
- 대상: [`m33_e3x_plan.md`](../processing/m33_e3x_plan.md) v0.1. `runs/eval/m33/e3x*/**`는 열지 않았고 실험을 실행하지 않았다. GPU 미사용.
- 사용자 결정(2026-11-20, 이 검토 뒤): 미주입 측선(4TU 100 + IDS 53) conf 0.001 기준 추론 1회 **추가**(보고 전용).

## 요약
- **Q1**: 조건을 지키면 한 번 추론으로 된다. 단 0.25 거름은 **병합 전 타일 박스**에 한다.
- **Q2**: 맞다. 매칭 주기의 출처와 E3(Y0)와의 규칙 차이만 적으면 된다.
- **Q3**: E0 방식 그대로는 안 된다. 병합 전 타일 박스 기준으로 고친다.
- **Q4**: 256 / 192 유지. 192는 r0 기본값에서 자동으로 나온다. 압축 측선은 0.7배가 되지 않는다.
- **Q5**: 동의(조건 3개).

## Q1. conf 0.001 한 번 추론 후 0.25로 거르기
**근거 1. Ultralytics NMS** (`gpr-dev\Lib\site-packages\ultralytics\utils\nms.py`)
- 후보는 `amax > conf_thres`(86행), 최고 클래스는 `conf > conf_thres`(123–125행)로 거른다. 둘 다 엄격 `>`, 0.25는 float32 정확 표현.
- 클래스별 offset을 둔 탐욕 NMS(146·157행, `torchvision.ops.nms`). 박스는 점수가 더 높은 같은 클래스 박스에게만 억제되므로 0.25 초과 박스의 생존은 0.25 초과 박스끼리만으로 정해진다.
- `i[:max_det]`(160행) 상위 300. `YoloDetector`는 `max_det`을 넘기지 않으므로(`src/gpr_detection/detect/yolo.py` 114–121·143–150행) 두 경로 모두 300. `max_nms` 30000(140행)은 256 px 입력 앵커 1,344개에 닿지 않는다.

**근거 2. 병합은 낮은 점수 박스가 섞이면 결과가 달라진다** (`src/gpr_detection/pipeline.py`)
- `_representative_key`(375–387행)는 잘림 여부, 중심 소유, 점수 순으로 대표를 고른다(낮은 점수의 잘리지 않은 박스가 대표가 될 수 있음). R2 연결 성분(399–441행)은 낮은 점수 박스를 통해 이어질 수 있다. `_arbitrate_bands`/`_clip_band`(491–544행)에서는 낮은 점수 띠가 다른 띠를 자를 수 있다. 따라서 병합 뒤 거름은 같지 않다.
- 선례: 단계 C는 병합 전 keep 마스크로 걸렀다(`analysis/detection/m32_recipe_eval.py` 641–643·1506–1512행, `m32_recipe_match.evaluate_on_conf_grid` 264–289행).
- 위험: R3는 병합 **뒤에** 거른다(`m33_r3_p4.counts_for_match` 382–384·531행, `m33_r3_band._any_class_cover` 1300행). conf 0.25 추론이던 R3에서는 무해했지만 0.001 박스를 거르지 않고 `merged_with_pp`에 넣으면 조용히 틀린다.

**조건**
- (a) 병합 전에 `score > 0.25`로 거른다(전체 정밀도 float, 순서 보존).
- (b) 기준 경로는 `pipeline.detect_line`(타일마다 batch 1, 342–350행)이 아니라 **같은 `run_jobs` 작업 목록·같은 batch 64 묶음을 conf 0.25로 추론한 경로**다(`yolo.py` 129–131행: batch가 다르면 마지막 비트가 달라짐).
- (c) NMS `time_limit = 2 + 0.05·bs`초(91·165–166행, batch 64면 5.2초)를 넘으면 batch 뒤쪽 영상이 빈 출력이 되고 WARNING만 남는다. 0.001은 박스가 많아 더 위험. `quiet_ultralytics_logger`는 WARNING을 숨기지 않으므로 handler로 셀 수 있다.
- (d) 점수가 정확히 같은 같은 클래스 박스가 겹치면 CUDA 정렬이 불안정해 후보 수에 따라 남는 박스가 바뀔 수 있다. precheck #6이 이를 잡는 유일한 장치다.
- (e) 0.001에서 타일 박스가 300개에 닿으면 낮은 점수 박스가 잘린다. 판정에는 영향 없고 R-d 최저 구간만 과소 추정된다.

## Q2. 쌍곡선 적중을 `match_one_to_one`으로
맞다(`analysis/detection/m32_recipe_match.py` 117–162행: 박스마다 포함하는 꼭짓점 중 가장 가까운 하나에 배정, `realization_spacing_ok` 간격 규칙). 보완 두 가지:
1. **매칭 주기**: 단계 C 방식대로 배경 측선 주기 `context.period_traces/period_samples`(`m32_recipe_eval.py` 1501–1504행). Y0 E3는 주입 영상 주기를 썼다(`m33_y0_diag_infer.py` 799–806행).
2. **매칭 규칙**: E3는 포함 규칙(일대일 아님), 단계 C는 일대일. `apex_hit_d3`(185–200행) 값을 보고 전용 열 `hit_contain`으로 같이 기록한다.

## Q3. conf 0.001 최고 점수 정의
E0는 0.001 박스를 병합한 **뒤** 꼭짓점 포함 박스 최댓값을 잰다(`m33_y0_diag_infer.apex_scores` 809–829행). 대표 키 때문에 운용 적중(h = 1)인데 최고 점수가 0.25 이하인 모순이 생길 수 있고, 띠는 낮은 점수 띠가 덮임을 바꾼다.

**제안 정의**
- `best_score_001_hyp`: 0.001 **병합 전** 타일 `pipe_hyperbola` 박스(각각 L0로 바깥 반올림) 중 `box_contains_apex`(±3 트레이스, ±0.5 P, 배경 주기)를 만족하는 점수 최댓값. 없으면 빈 값(`no_box`). `_any`는 모든 클래스.
- 띠 `best_score_001_band`: 병합 전 `pipe_band` 중 `can_cover`(시간 겹침 ≥ 정답 높이 50 % **그리고** [i0, i1]과 가로 겹침) 박스. 가로 겹침 조건이 없으면 측선 다른 곳의 같은 깊이 실측 띠가 잡힌다.
- 불변식 `hit = 1 ⇒ best > 0.25`는 구현 단언(위반 시 `e3x_defect`).
- R-d 구간에 `no_box`(0.001 이하 또는 max_det 잘림)와 `(0.25, 1]`(병합·배정 손실)을 추가.
- 보고 전용 권고: 같은 자리 주입 전 0.001 기준값(미주입 측선 0.001 추론 1회) — **사용자 채택**.

## Q4. P* 4.2 재영상화에서 타일 256 / 겹침 192
유지한다. 타일 256은 학습 imgsz이고 letterbox가 항등(`to_image.py` 308–310행). 겹침은 `overlap_for` = `min(224, max(O_phys, 192))`(1079–1087행)라 P* 4.2에서도 192. 레시피 `e3x_p42`는 `p_target_px`(와 `recipe_id`)만 바꾸고 `overlap_px`는 지정하지 않는다. 긴 팔 13 P = 54.6 px는 겹침 안에 온전히 들어간다. 주의:
1. **압축 측선은 0.7배가 아니다**(`line_scale` 1042–1049행): 실제 행 비 0.7·c0/max(1, 0.7·c0). 레시피별 `compression`과 `realized_y1_px_image`를 기록하고 I_px는 실제 행과 함께 낸다.
2. 4.2/6 = 0.70은 `compress_max` 1.35(0.74)로 정한 확인 축척 범위 바로 밖이다(263–265행) — 판정서 공개.
3. 내용 높이가 줄어 패딩이 커진다. PP 꺼짐 판정에는 영향 없음, R-b는 보고 전용.

## Q5. 판정 S0, S3 보고 전용
동의. 조건: (i) S0와 S3는 **같은 병합 박스**(병합 전 거른 것)에서 낸다(`pipeline.pp_decision` 808행). (ii) R3처럼 `sealed_config` 방식으로 `pp_set_id = S3`와 `PPConfig` sha(C0b)를 단언한다. (iii) ③ 뒤 src 기본값이 S3로 바뀌어도 E3x 판정은 S0 기준이라고 명시하고, S3에서 결정 라벨이 바뀌는 가설 수는 보고 전용으로 낸다. 참고: 구역 (b)와 C3-6 때문에 정답 박스의 PP 노출은 구조상 거의 0일 것([RUN] 166행과 같은 논리).

## 기존 판정서·러너 명세와 어긋나는 곳, 누수 위험
1. §3-1 띠 행의 "`m33_r3_band.match_units`(같은 함수)" 재사용은 병합 뒤 거름 경로 — 병합 전 거름 명시(Q1).
2. A-5 시험 10은 같은 입력을 병합하는 동어반복이라 NMS 쪽을 검증하지 못한다. 음성 대조로 바꾼다: 낮은 점수의 잘리지 않은 박스가 대표가 되는 fixture에서 병합 뒤 거름은 다르고 병합 전 거름은 같음을 보인다. NMS 동일성은 실제 모델로 precheck #6이 맡는다(`tests/`에서 ultralytics import는 AGPL 경계 문제라 피함).
3. AIHub: §4 "열지 않는다(test·train 모두)"는 맞지만 상수 출처를 적는다. `detection_image_scale_survey_v1.json`은 `data/audit/aihub_71327/objects.csv`(전수, test 세션 포함)의 라벨 기하 통계다. 영상 추론·`test.yaml`·모델 출력은 쓰지 않으므로 test 개봉은 아니다.
4. 4TU 독립성: 새 측선은 R3 177측선과 측선 단위로만 겹치지 않는다. 활동(부트스트랩 군집)은 겹칠 수 있다 — 공유 활동 수를 plan 보고에 내거나 표본이 허락하면 R3 활동을 뺀다.
5. R2 미주입 박스 방향: R2는 2방향([RUN] 262행). 구역 (c) 입력의 방향(`original`만인지 합집합인지)을 봉인 1에 고정한다.
6. A-3 기록 열 추가: `best_score_001_*`의 `no_box` 구분, `hit_contain`, 0.001 박스 수가 300에 닿은 타일 수, NMS time-limit 경고 수, 레시피별 `compression`.

## 설계에 넣을 수정 문장(안)
- §3-1 끝 문단 대체: "conf 0.001로 한 번 추론하고, 판정 지표는 **병합 전 타일 박스**를 `score > 0.25`(전체 정밀도, 순서 보존)로 거른 뒤 `merge_line_boxes`·`line_boxes_to_grid`·(보고용) `apply_pp`를 거친 박스로만 계산한다. 병합 뒤 거름(`p4.counts_for_match`에만 기대는 것)은 쓰지 않는다. 운용 기준 경로는 같은 작업 목록·같은 batch 64 묶음을 conf 0.25로 추론한 경로다(`pipeline.detect_line`의 batch 1 경로가 아님)."
- §5-2 #6 대체: "precheck 측선 2개의 모든 팔·fold 0–4에서 conf 0.001 경로(병합 전 > 0.25)와 같은 `run_jobs` 묶음의 conf 0.25 경로가 타일 박스(클래스·점수·좌표)와 병합 박스에서 비트 단위로 같다."
- §5-2 #9 추가: "Ultralytics `NMS time limit` 경고 수를 logging handler로 세어 0이 아니면 `e3x_defect`(본 실행도 같음). 0.001 출력이 300개인 타일 수와, 0.25 초과 박스가 300개인 타일(있으면 `e3x_defect`)을 센다."
- §1-2 R-d 대체: "최고 점수 = conf 0.001 **병합 전** 타일 박스 중 꼭짓점 포함 규칙(±3 트레이스, ±0.5 P, 배경 주기)을 만족하는 `pipe_hyperbola` 점수의 최댓값(띠는 `can_cover` 조건). 구간은 `no_box` / ≤ 0.01 / (0.01, 0.1] / (0.1, 0.25] / (0.25, 1](병합·배정 손실). 단언: 적중이면 최고 점수 > 0.25."
- §3-1 쌍곡선 행 추가: "매칭 주기는 배경 측선 값(단계 C와 같음). Y0 E3(포함 규칙, 주입 영상 주기)와의 비교용으로 `hit_contain`(`apex_hit_d3`)을 보고 전용으로 기록한다."
- §2-6 추가: "`e3x_p42`는 `p_target_px` 4.2만 바꾼다(타일 256, 겹침은 `overlap_for`가 정하는 192). r0 압축 c0 > 1인 측선은 행 비가 0.7이 아니므로 레시피별 `compression`과 실제 행을 기록한다. 4.2/6 = 0.70은 확인 축척 범위(0.74) 밖이다(공개)."
- §3-1 공통 PP 행 추가: "S0와 S3는 같은 병합 박스에서 내고, `pp_set_id` S3와 `PPConfig` sha를 단언한다. ③ 뒤 src 기본값이 바뀌어도 E3x 판정은 S0 기준이며, S3에서 결정 라벨이 달라지는 가설 수는 보고 전용이다."
- §4 AIHub 행 추가: "상수는 AIHub 전수(test 세션 포함) 라벨 기하 통계(`objects.csv`)에서 왔다. 영상·모델 출력·`test.yaml`은 쓰지 않는다."
- §2-1 추가: "새 4TU 표본과 R3 표본이 공유하는 활동 수를 plan 보고에 낸다."
- §5-5 또는 R-d 보고 전용(사용자 채택): "같은 자리 주입 전 0.001 기준값(미주입 측선 0.001 추론 1회)."

## 위임 필요
- gpr-processing-expert: 위 수정 문장 v0.2 반영.
- python-expert(구현 단계): 병합 전 거름 경로, 같은 묶음 conf 0.25 기준 경로(precheck #6; 같은 YOLO 객체를 `YoloDetector(model, ...)` 두 개로 감싸거나 `load_yolo_detector` 두 번), NMS time-limit handler·max_det 카운터, `no_box`·`(0.25, 1]` 구간, `hit_contain` 열, 레시피별 `compression` 기록, A-5 시험 10 음성 대조로 교체.
- stats-design-expert: R-d 구간 두 칸 추가는 보고 전용이라 판정 가족에 영향 없음(전달만).
