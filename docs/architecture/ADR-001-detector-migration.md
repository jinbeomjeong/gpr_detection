# ADR-001: 검출기는 YOLO11m 초안 → Apache 계열(RF-DETR S/M)로 이전하고, 의존을 `Detector` 프로토콜 뒤로 격리한다

- 상태: Accepted (2026-10-29, 사용자 확정. 이전 판정 기준·좌표 역변환 위치는 2026-10-29 사용자 답변으로 확정)
- 날짜: 2026-10-29
- 관련 드라이버: QA-1(검출기 교체), QA-7(라이선스), FR-1, CON-1, CON-3, CRN-4 — 정의는 [README](README.md) §1
- 반복: 1회차 (체크리스트 M0-12). 이전 결과 반영은 M4-5

## 맥락

- 검출 기준선이 빨리 필요하다(M1-10). Ultralytics YOLO11m은 Windows·Python 3.13에서 공식 지원되고, 소규모 실측 GPR에서 RT-DETR-X보다 높았다(Štifanić 2025: YOLO11x mAP50 0.817 vs RT-DETR-X 0.735, `docs/research/prior_art.md` §1).
- Ultralytics는 AGPL-3.0이고 학습 산출 모델에도 적용되며, 사내 사용도 Enterprise 라이선스 대상이라고 명시돼 있다(`docs/research/packages.md` 34행). 배포물에 넣을 수 없다(CON-3).
- Apache-2.0 후보 중 RF-DETR N/S/M/L은 코드·가중치 모두 Apache-2.0이고 RF100-VL AP50 85.0–88.2(YOLO11-M 82.5)다. XL·2XL 등은 PML 1.0이라 제외한다(prior_art §4, packages.md 35행).
- `gpr-detr` env에는 ultralytics가 없다(CON-1). 패키지가 두 env에서 모두 import돼야 한다(CRN-4).
- RF-DETR 기본값은 정사각 리사이즈·scale jitter로 쌍곡선 종횡비를 바꾼다(`docs/experiments/augmentation_policy.md` §4).

## 고려한 대안

| 대안 | 장점 | 단점 | 드라이버 충족 |
|---|---|---|---|
| **A. YOLO11m 초안 → RF-DETR S/M 이전, `Detector` 프로토콜로 격리** (채택) | 기준선을 빨리 얻음. 교체 범위가 `detect/`로 한정됨. 두 모델을 같은 분할로 비교 가능 | 학습을 두 번 함. 이전 판정 전까지 AGPL 산출물이 로컬에 존재 | QA-1 ○, QA-7 ○(M4 완료 조건), FR-1 ○ |
| B. 처음부터 RF-DETR만 사용 | AGPL 노출 없음 | 정사각 리사이즈 우회·Windows 학습 경로가 미검증이라 기준선이 늦어짐. 소규모 GPR에서 DETR 계열이 약하다는 선행 사례 | QA-7 ◎, FR-1 일정 위험 |
| C. YOLO11m 유지 + Enterprise 라이선스 구매 | 코드 변경 없음 | 비용·법무 판단이 필요(U-4), 사용자 목표("Apache/BSD 계열로 이전")와 다름 | QA-7 조건부 |
| D. 다른 Apache 모델(D-FINE/DEIM, transformers RT-DETRv2) 1순위 | transformers 경로로 Windows 가능 | D-FINE은 COCO 가중치만 허용, DEIMv2는 DINOv3 라이선스. RF-DETR보다 문서화된 미세조정 경로가 약함 | 2순위 후보로 유지 |
| E. 격리 없이 ultralytics API를 파이프라인에서 직접 사용 | 초기 코드 적음 | 교체 시 파이프라인 전체 수정, ultralytics 없는 `gpr-detr`에서 import 실패 | QA-1 ✕, CRN-4 ✕ |

## 결정

1. M1–M3의 검출기는 Ultralytics **YOLO11m**(gpr-dev, ultralytics 8.4.192)이다.
2. 이전 1순위는 **RF-DETR S/M**(필요 시 N/L, Apache-2.0 크기만), 2순위는 transformers의 RT-DETRv2 / D-FINE(COCO 가중치)이다. 실험은 `gpr-detr` env에서 한다.
3. 모든 검출 백엔드는 `src/gpr_detection/detect/`의 `Detector` 프로토콜을 구현한다(계약은 README §2-3).
   - `predict(image)`는 L2(모델 입력) 좌표의 `Detection` 목록과, 백엔드가 적용한 L1 → L2 변환 파라미터 `InputTransform`(축별 배율·패딩·입력 크기)을 함께 돌려준다. letterbox·정사각 리사이즈의 역변환(L2 → L1 → L0)은 `pipeline`이 한 곳에서 공통으로 한다.
   - `ultralytics`·`rfdetr` import는 `detect/` 아래 해당 백엔드 파일의 함수 안(지연 import)에만 둔다. `detect/__init__.py`는 백엔드를 import하지 않는다.
   - 백엔드 선택은 설정 값(예: `detector.backend: yolo | rfdetr`)으로 한다.
4. 비교 실험은 같은 분할(ADR-003의 AIHub 세션 그룹 k-fold), 같은 증강 정책, 같은 `imgsz`·epoch 예산으로 한다. RF-DETR은 비율 유지 입력 경로를 확인한 뒤에만 비교한다(M4-1).
   - **이전 가능 판정 기준(M4-3)**: RF-DETR의 전체 mAP50(fold 평균)이 YOLO11m 대비 **−5 %p 이내**이면 이전 가능으로 판정한다. 클래스별 AP50은 함께 보고하되 판정 조건에는 넣지 않는다.
5. M5 배포물에는 Ultralytics 코드와 YOLO 학습 가중치를 넣지 않는다(M5-4 점검).

## 결과

얻는 것
- 검출기 교체가 `detect/`와 설정 파일 변경으로 끝난다(QA-1 척도: `detect/`·`configs/` 밖 변경 0줄).
- 패키지가 `gpr-dev`·`gpr-detr` 양쪽에서 import된다.
- 같은 조건의 정량 비교로 이전 여부를 판단한다.

잃는 것 / 비용
- 학습·평가를 두 계열로 반복한다(GPU 1장, CON-2 — 순차 실행).
- 프로토콜이 두 모델의 공통 분모만 노출한다(모델별 고유 출력, 예: 세그먼트 마스크는 쓰지 않는다).
- 역변환을 pipeline에 모은 대가로, 백엔드는 라이브러리 내부 전처리 값(Ultralytics letterbox 배율·패딩, RF-DETR 리사이즈 크기)을 `InputTransform`으로 꺼내 노출해야 한다. 새 백엔드의 전처리가 "축별 배율 + 패딩"으로 표현되지 않으면 스키마와 pipeline을 함께 고쳐야 한다(README §2-3).
- 판정 마진 −5 %p는 넓은 편이라, 이전 뒤 검출 성능이 최대 5 %p 낮아지는 것을 감수한다(라이선스를 우선).
- M4 전까지 AGPL 산출물(`runs/`의 YOLO 가중치)이 로컬에 남는다.

위험
- RF-DETR이 소규모·단일 장비 pipe 데이터(세션 11–13개)에서 YOLO11m보다 낮을 수 있다 → 2순위 후보, 또는 Enterprise 라이선스(U-4) 판단으로 넘어간다.
- 정사각 리사이즈 우회가 안 되면 비교가 불공정하다(augmentation_policy §4).
- 공개 저장소(CON-6)에 YOLO 가중치가 실수로 올라갈 위험 → 대응: `.gitignore`(`*.pt`, `*.onnx`, `/runs/`)와 커밋 전 비밀·데이터 점검. 별도 규칙은 두지 않는다(사용자 결정 2026-10-29).
- 백엔드가 `InputTransform`을 잘못 노출하면 깊이 박스가 어긋난다 → 백엔드 래퍼 시험에 L1 → L2 → L1 왕복 오차 ≤ 0.5 px를 넣는다.

후속 과제
- M1-8: 프로토콜(`DetectorOutput`·`InputTransform`)·YOLO 래퍼 구현, "`detect/` 밖 ultralytics import 0건" 정적 시험, `InputTransform` 왕복 시험(python-expert).
- M4-3: 결정 4의 기준(전체 mAP50 −5 %p 이내, k-fold 평균)으로 판정(cv-detection-expert).
- M4-5: 비교 결과로 이 ADR을 갱신(상태를 "대체됨" 또는 결과 절 추가).

## 근거 문서

- [`docs/research/prior_art.md`](../research/prior_art.md) §1, §4 — 검출 사례, Apache 후보 비교
- [`docs/research/packages.md`](../research/packages.md) — ultralytics·rfdetr·transformers 라이선스와 버전
- [`docs/research/packages_agents.md`](../research/packages_agents.md) — `gpr-detr` env 구성
- [`docs/experiments/augmentation_policy.md`](../experiments/augmentation_policy.md) §2–4 — 증강 정책, RF-DETR 기본값 문제
- [`docs/ROADMAP.md`](../ROADMAP.md) M1·M4·M5, [`docs/CHECKLIST.md`](../CHECKLIST.md) M1-8, M4-1–M4-5, U-4
