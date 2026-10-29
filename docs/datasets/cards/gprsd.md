---
pretty_name: "GPR-SD (GPR Subsurface Diagnosis) Road GPR Dataset, USTC"
license: unknown
language:
  - en
task_categories:
  - other
size_categories:
  - n<1K
tags:
  - ground-penetrating-radar
  - gpr
  - b-scan
  - road
  - buried-pipe
  - cavity
  - weak-labels
  - china
source_datasets:
  - original
annotations_creators:
  - expert-generated
---

# GPR-SD 도로 GPR 데이터셋 (중국 USTC)

| Item | Value |
|---|---|
| Summary | Long road GPR profiles (2 cement + 2 asphalt roads, 3 parallel lines each, 200-400 MHz) with 73 rough anomaly marks (pipeline, cavity, loose, crack) |
| Format | Original MATLAB .mat + Mark.xlsx; processed copy has npy (float32, [321 samples, traces]) + json, `marks.csv` |
| Size | 12 profiles, 275,885 traces (about 8.28 km); 73 marks: pipeline 17, cavity 18, loose 23, crack 15 |
| Labels | Trace-number marks only (no depth, width or box; position error +-50-100 traces = +-1.5-3 m) |
| Physical scale | Partial: trace spacing 0.03 m, time window 97 ns documented; sample interval unresolved (hypothesis A 0.302 ns vs B 0.255 ns, +-16 %); no velocity |
| Role in this project | Field test and manual pipe-box labeling (M3-4); dt hypotheses A/B test |
| License | unknown - no license statement in the original repository; the authors only ask to cite two papers |

## Dataset Summary

시멘트 도로 2곳·아스팔트 도로 2곳을 200–400 MHz GPR로 각각 평행 3측선씩 탐사한 B-scan 12개(측선당 약 280–830 m)와, 관로·공동·이완층·균열 위치를 대략 표시한 표(73건)다. 원본은 중국과학기술대(USTC) Huanhuan Chen 연구실이 2023-02에 GitHub로 공개했다(https://github.com/ubriustc/GPR_data). 실제 도로의 연속 장거리 측선과 관로 17건·공동 18건 등의 이상체 위치가 있어 쌍곡선 탐지 모델의 교차 데이터셋 검증, 약지도 학습에 쓸 수 있다. 그러나 표식이 트레이스 번호 하나뿐이라 그대로는 학습 정답이 되지 못하고, 깊이는 거친 추정만 가능하다([readme_gprsd.md §1](../readme_gprsd.md)).

## Dataset Structure

### 형식

- 로컬 위치: `C:\workspace\GPR_Data\GPR-SD_중국USTC_도로GPR_가공본_261016`.
- `05_radargrams_npy/{도로}_{측선}.npy`(float32 `[321 샘플, 트레이스]`) + `.json`(파라미터와 해당 도로 표식). 도로 CementRoad1·CementRoad2·AsphaltRoad1·AsphaltRoad2, 측선 1–3(같은 도로의 평행 측선).
- `01_메타데이터/radargrams_index.csv`(측선 12개), `marks.csv`(표식 73건). marks 열: `xlsx_row, road_xlsx, road, segment_xlsx, location_trace, location_m, type, 유형, type_xlsx, segment_mismatch, note, files, window_png`.
- 진폭은 MATGPR로 지표 에코 제거·게인이 적용된 **후처리 자료**다(극값 ±1.4x10^5, 정규화 시 99 % 백분위로 자름). npy는 원본 float64를 float32로 저장.

### 개수

| 항목 | 값 |
|---|---|
| 측선 | 12 = 도로 4곳 x 평행 3측선 |
| 트레이스 | 275,885, 간격 0.03 m, 합계 약 8.28 km |
| 측선 길이 | CementRoad2 약 281–282 m, 나머지 822–829 m |
| 샘플 | 트레이스당 321 (원 문서는 380으로 적혀 있으나 실제 파일은 전부 321) |
| 표식 | 73 = 관로 17 + 공동 18 + 이완층(loose) 23 + 균열(crack) 15 |

### 분할

분할 없음(시험 전용). 같은 도로의 3측선에 같은 이상체가 반복되므로 측선 단위로 나누면 누수가 되고, 도로 단위가 독립 단위다.

## 물리 스케일

| 항목 | 값 | 상태 |
|---|---|---|
| 트레이스 간격 | 0.03 m | 원 저장소 문서 값 |
| 시간창 | 97 ns | 문서 값, 파일에는 시간축 정보 없음 |
| 샘플 간격 | 0.302 ns(가설 A: 321샘플 = 97 ns) 또는 0.255 ns(가설 B: 원래 380샘플 = 97 ns에서 앞 59샘플 절단) | **미확정, +-16 % 차이.** json·CSV에 `dt_ns_hypA_321`, `dt_ns_hypB_380` 둘 다 기록 |
| 안테나 | 200–400 MHz | 모델명·중심주파수 미기재 |
| 속도·유전율 | 없음 | - |
| 샘플당 깊이 | 약 0.012 m(1.0–1.4 cm) | 관로 쌍곡선 맞춤으로 거칠게 추정(적합도 낮음). 시간영점 = 샘플 0 가정, 가설 B면 약 30 % 과대 |

## Dataset Creation

- **출처**: GPR-SD (GPR Subsurface Diagnosis), 중국과학기술대(USTC) Huanhuan Chen 연구실, GitHub 공개 2023-02. 원본 .mat은 분할 압축 `GPR-SD.zip.001`–`.033`을 이어 풀어 얻는다.
- **라벨링**: 원 저자가 표식을 "대략 표시(roughly marked)"했다. 쌍곡선 꼭짓점과 비교하면 +-50–100 트레이스 오차가 있고, 표식 1개 근처에 관로 쌍곡선 2개가 있는 경우도 있다. 원본 표에 측선 구분이 없어 한 표식이 같은 도로의 3측선 모두에 적용되는 것으로 본다(측선마다 트레이스 수가 최대 108개 달라 정렬은 근사).
- **가공(받은 가공본)**: .mat → npy/png, 위치표 CSV 정리(도로 이름 통일, 오타 정정 `losse`→`loose` 2건, 구간 불일치 표시), 표식 주변 +-150트레이스 창 이미지 생성(박스 라벨 작업용).

## Considerations (알려진 결함·편향·한계)

1. **표식은 트레이스 번호 1개뿐**이라 깊이·폭·박스가 없다. 그대로는 학습 정답이 못 된다.
2. **시간축이 미확정**(가설 A/B, +-16 %)이라 깊이는 거친 추정이다. 쌍곡선 맞춤으로 직접 보정해야 한다.
3. 표식 `segment_mismatch` 5건은 원본 구간 범위 밖이다(원본 오류로 보임, 트레이스 번호를 우선 신뢰).
4. 균열(crack) 표식은 영상에서 구분하기 어려워 신뢰도가 낮다. 공동 일부는 물이 찬 공동이라는 원본 주석이 있다.
5. 지표 에코 제거·게인이 이미 적용된 후처리 자료이며 CementRoad2는 수평 성분이 일부 남아 있다.
6. 이 프로젝트(관로 검출) 관점에서는 관로 17건이 직접 쓸모 있고 공동·이완층은 지반침하 쪽 참고용이다.
7. 측선이 매우 길다(최대 27,628 트레이스). 학습·추론에는 창 단위로 잘라 쓴다.

## Licensing & Citation

- **라이선스: 알 수 없음(`unknown`).** 원 저장소에 LICENSE 파일이나 라이선스 문구가 없다. 연구 목적 내부 사용에 한정하고, 외부 재배포·상업 이용 전에는 원 저자 확인이 필요하다. 이 저장소에는 데이터를 올리지 않는다.
- 원 저자가 사용 시 요청하는 인용 2편:
  - [1] X. Zhou, H. Chen, J. Li, "An Automatic GPR B-Scan Image Interpreting Model," IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 6, pp. 3398-3412, 2018, doi: 10.1109/TGRS.2018.2799586
  - [2] X. Wang et al., "Accurate Label Refinement From Multiannotator of Remote Sensing Data," IEEE Transactions on Geoscience and Remote Sensing, vol. 61, Art. no. 4700413, 2023, doi: 10.1109/TGRS.2023.3241402

## 이 프로젝트에서의 사용

- **역할**: 현장 시험(교차 도메인). 관로 표식 17건 × 3측선 = 표식창 51장(계산값, 표식창 PNG `pipeline*` 51개로 확인)에 Label Studio로 박스를 수동으로 그려(M3-4, 라벨링은 사용자) 소량 검증셋을 만들고, AIHub로 학습한 모델을 시험한다(M3-3). 시간축 가설 A/B(dt)를 쌍곡선 맞춤으로 비교한다.
- **쓰는 파생 데이터**: 지금은 가공본을 읽기만 한다. 수동 라벨 완료 후 `labeling/` 내보내기를 프로젝트 형식으로 변환해 사용할 예정(미작성).
- **관련 체크리스트**: M3-3(교차 도메인 평가), M3-4(수동 라벨).
- 근거 문서: [readme_gprsd.md](../readme_gprsd.md), [CHECKLIST.md](../../CHECKLIST.md) M3-4.
