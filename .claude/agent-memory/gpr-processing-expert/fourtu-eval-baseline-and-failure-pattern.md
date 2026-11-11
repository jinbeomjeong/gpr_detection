---
name: fourtu-eval-baseline-and-failure-pattern
description: 4TU depth eval — activity-range random baseline pairs ~95% correct, so p<0.05 needs ~97% of pairs within 0.30 m; shallow-tail failures are wrong-event, not eps_r
metadata:
  type: project
---

4TU 깊이 평가(eval_depth_4tu.py)의 활동 범위 무작위 기준선은 관측과 같은 짝 수를 만들어 짝 기준 판정률 A가 약 0.95다(M2-8, M2-9 모두). 그래서 p<0.05는 짝의 약 97 % 이상이 ±0.30 m 안이어야 한다. 전역 U[0.2,2.0] 기준선은 M2-9에서 p=0으로 이겼다 — 두 기준선 차이가 "정답 범위를 아는" 이점의 크기.

M2-9 재평가(2026-11-11, depth_4tu_v2) 실패 짝은 대부분 얕은 쪽(40/44)이고 d_est/d_gt 중앙값 약 0.45 — εr 배율로는 불가능(εr ≥ 3 가정 시 하한 √(3/ε_meta)≈0.58). 깊은 꼭짓점 누락 + 게이트 없는 헝가리안이 얕은 거짓 추정과 짝지은 것으로 읽음. 메타 εr 보정(개선 F)의 기대 몫은 작다.

**Why:** M2-9는 U-3 불합격(기준선)으로 종료, 후속은 M2-10 "4TU 후속 탐색"(탐색만, 4TU는 이미 본 자료).
**How to apply:** 4TU 개선 제안을 판정할 때 A 0.95 문턱을 먼저 보고, εr 보정 효과는 d_est/d_gt 비 분포로 상한을 잡는다. 4TU 결과로 규칙을 검증하지 말고 GPR-SD·광저우대·gprMax로 검증한다. 4TU csv를 pandas로 읽을 때는 `activity_id`·`path`를 `dtype=str`로 읽는다. 숫자로 읽으면 "05.1"과 "05.10"이 5.1로 합쳐져 활동 93개가 91개로 세진다(M2-9 판정 첫 재집계에서 추정 0개 활동을 19로 잘못 셈, 정답 18). 생성 md에 "꼭짓점 4 트레이스 안 정지 비율"이 없으면 candidates csv의 min(qc_n_left, qc_n_right) ≤ 4로 근사(사후 표시).
