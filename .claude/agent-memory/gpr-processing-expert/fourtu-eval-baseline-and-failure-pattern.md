---
name: fourtu-eval-baseline-and-failure-pattern
description: 4TU depth eval — activity-range baseline A ~0.95; shallow failed pairs = surplus shallow event + truth without own estimate (not eps_r); depth-only clustering merges lateral pipes; ungated L1 Hungarian ties move pass count 133-148
metadata:
  type: project
---

4TU 깊이 평가(eval_depth_4tu.py)의 활동 범위 무작위 기준선은 관측과 같은 짝 수를 만들어 짝 기준 판정률 A가 약 0.95다(M2-8, M2-9 모두). 그래서 p<0.05는 짝의 약 97 % 이상이 ±0.30 m 안이어야 한다. 전역 U[0.2,2.0] 기준선은 M2-9에서 p=0으로 이겼다.

M2-10 (d) 진단(2026-11-11, `docs/processing/m210d_event_selection_diagnosis.md`, 이미 본 자료):
- 얕은 실패 짝 40/44는 "그 관을 얕게 잼"이 아니다. 남는 얕은 사건(추정 중앙 0.43 m)이 자기 추정 없는 깊은 정답(중앙 0.985 m)과 짝지어진 것이다(비 0.437 ≈ d̂/d 중앙 0.447). 판정률 B에서 실패 짝과 누락은 같은 손실이다.
- 비통과 248개 중 가장 큰 몫은 "근처 지지 추정을 통과한 다른 정답이 이미 씀"(ii-a 114)이다. 깊이 1차원 묶기가 같은 깊이대의 옆 관들을 추정 하나로 합친다(묶음 안 같은 측선 0.3 m 넘게 떨어진 꼭짓점 = lateral multi 71). >1.1 m는 생성기 단계(피크 없음·수락)와 기록 끝에서 잃는다.
- 게이트 없는 L1 헝가리안은 동률 최적해가 많다. 같은 입력에서 통과 수가 133–148로 달라진다(커밋 140). 게이트 매칭 상한은 +9다.
- "근처 항목 있음"은 우연 덮임이 크다(비통과 활동 평균 덮임: peak 0.64, supported 0.28). 손실 단계는 상한으로만 읽는다.

**Why:** M2-9는 U-3 불합격(기준선)으로 끝났고, M2-10은 탐색 전용이다(4TU 튜닝 금지, 검증은 합성·다른 실측).
**How to apply:** 4TU 개선 제안은 B_max·추정 수 부족 쪽을 먼저 본다. εr 보정이나 매칭 게이트의 몫은 작다. 기준선은 n_est를 따라 오르므로, 추정 수를 늘리는 개선은 기준선도 올린다. 진단 재현에는 `analysis/processing/m210d_generator_stages.py`(생성기 단계 재실행 104 s, 자동 후보 2,629개 정확 재현)와 커밋된 matches를 쓴다. 헝가리안을 다시 풀면 동률로 다른 짝이 나온다. pandas로 4TU csv를 읽을 때 `activity_id`·`path`는 `dtype=str`로 읽는다("05.1"과 "05.10" 충돌).
