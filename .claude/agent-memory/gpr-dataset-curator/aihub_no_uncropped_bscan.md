---
name: aihub-no-uncropped-bscan
description: AIHub 71327 원천 zip에는 crop 이전 B-scan이 없다(전부 256x256 crop, 객체가 있는 crop만); 맥락은 같은 line·gain의 겹친 crop뿐
metadata:
  type: project
---

TS_/VS_ B scan zip 12개 전수 목록(74,838, 전부 256x256 L JPG, 이름 `<장비>_<날짜(시각)>_N00_B<ch>_g<gain>_x(a-b)`)에 측선 전체·넓은 영상 없음. C scan zip은 256x15~32 깊이 단면 crop이라 B-scan 맥락이 아님. 모든 crop은 객체 ≥1 라벨 XML 짝이 있어 객체 없는 구간은 어떤 crop에도 없다. 같은 line(세션+채널)·gain 안에서 x가 겹치는 crop만이 맥락이며, 게인별 x 시작이 서로 다를 수 있다.
**Why:** 이동 증강 맥락 출처 질문(2026-11-13)에 재조사 없이 답하려고. **How to apply:** 결과표·인덱스는 data/aihub_context_index/(crops.csv, groups.csv, crops_fullset_ext.csv)와 analysis/data/aihub_context_groups.py·aihub_context_fullset.py. 맥락은 v1 매니페스트 line 안에서만(line→세션→fold 1:1, 누수 0).
