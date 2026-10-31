---
name: aihub-crop-overlap-band
description: AIHub 256px crop은 같은 측선에서 서로 겹치고(stride 중앙값 103 px) pipe_band 라벨이 crop마다 따로라 불일치; v1 간격>8px 쌍 223 중 인접은 115
metadata:
  type: project
---

crop_key `<세션>_N00_B<n>_x(a-b)`의 같은 측선 crop은 겹친다(맞닿음 0). 이웃 crop을 stride만큼 옮기면 화소가 같다(평균 절대차 2-9/255). 같은 띠를 crop마다 다르게 그려 공유 구간 x 커버리지 XOR 중앙값 0.41.
**Why:** pipe_band 박스 끝이 영상 증거가 아니라 라벨러 습관이라 재라벨 지침(보이는 구간 전체, 가장자리까지)이 필요. **How to apply:** 간격/조각 통계는 "인접 쌍"(사이에 같은 행 박스 없음)으로 세야 한다(v1 >32px 124쌍 중 인접은 16). 재라벨 계획: labeling/pipe_band_relabel_plan.md. Label Studio 서버는 사용자가 띄움(127.0.0.1:8080), 프로젝트 생성·API 호출은 사용자 결정 뒤.
