---
name: aihub-xml-pitfalls
description: AIHub 71327 label zip/XML 함정 - filename·source가 멤버 이름과 어긋남, 모든 Val 세션이 Train과 겹침
metadata:
  type: project
---

AIHub 71327 B scan 라벨 zip(TL_/VL_)의 XML 멤버 이름(=원천 JPG 이름)과 XML `<filename>`이 1,870건 다르고, `<source>`가 이름 유도 세션과 다른 것이 224건(공동 3종). 짝짓기는 멤버 stem 기준으로 한다.
**Why:** `<filename>` 기준이면 이미지를 못 찾는다. **How to apply:** 변환·분할 스크립트 검토 시 이 점을 확인.
zip 멤버 이름은 앞에 `/`가 붙는다(`/NM_..._x(a-b).jpg`). Validation 세션 1,599개가 전부 Training과 겹친다. lpipe=수평 띠, tpipe=쌍곡선(몽타주 확인), 둘 다 NM 장비·세션 11~13개. 상세는 docs/datasets/aihub_71327.md.
