---
pretty_name: "Guangzhou University GPR Dataset (pipe and rebar, Zenodo 14637589)"
license: cc-by-4.0
language:
  - en
task_categories:
  - other
size_categories:
  - n<1K
tags:
  - ground-penetrating-radar
  - gpr
  - radargram
  - buried-pipe
  - rebar
  - unlabeled
  - ids-georadar
  - china
source_datasets:
  - original
annotations_creators:
  - no-annotation
---

# 광저우대 GPR 데이터셋 (중국)

| Item | Value |
|---|---|
| Summary | Raw IDS ground-coupled GPR lines over buried pipes on a university-town road (mostly 400 MHz) plus rebar in concrete (2.0/2.6 GHz); no labels |
| Format | Original IDS K2 `.dt` (and MALA `.rd3`); processed copy has npy (int16, [samples, traces]) + json scale, CSV indexes |
| Size | 648 converted line files: pipe IDS 448 (10.41 km) + pipe MALA 75 (1.21 km) + rebar IDS 125 (1.65 km); 147 zones; 268 files excluded |
| Labels | None (no boxes, depths, pipe types, line maps or GPS) |
| Physical scale | Yes for time/distance (dt, trace spacing, t0 in headers); velocity NOT measured (0.10 m/ns assumed) |
| Role in this project | Unlabeled inference and velocity-estimation test (M3-7) |
| License | CC BY 4.0 (original); attribution required |

## Dataset Summary

중국 광저우 대학성(대학 캠퍼스 도로)에서 IDS 지면결합 GPR(주로 400 MHz)로 지하관로 위를 측정한 원시 B-scan과, 포산 철근콘크리트를 2.0/2.6 GHz로 측정한 자료다. 원본은 IDS K2 전용 이진 형식(.dt)이고 설명 문서가 없었다. 시간·거리 축척이 헤더에 들어 있는 실제 도시 도로 관로 B-scan이 있어 쌍곡선 곡률로 속도·심도를 계산하는 연습과 탐지 모델 사전학습·추론 시험에 적합하다. **정답이 전혀 없으므로 깊이 정확도 검증용으로는 쓸 수 없다**([readme_guangzhou.md §1](../readme_guangzhou.md)).

## Dataset Structure

### 형식

- 로컬 위치: `C:\workspace\GPR_Data\광저우대_중국_관로GPR_가공본_261016\05_radargrams_npy`.
- `01_메타데이터/`: `radargrams_index.csv`(변환한 측선 648행), `zones_summary.csv`(존 147개), `excluded_files.csv`(변환하지 않은 268개와 사유).
- npy 경로: `{대상 pipe|rebar}/{미션_존}/{파일명}.npy`(int16 `[샘플, 트레이스]`) + 같은 이름의 `.json`. 파일 ID는 `대상/미션_존/파일명`(예: `pipe/2020wate_200113AA/LID10002`). MALA 파일은 `pipe/MALA_{원본 폴더명}/`.

### 개수

| 항목 | 값 |
|---|---|
| 관로 pipe, IDS .dt | 448 파일 · 77 존, 트레이스 485,205, 길이 합계 10.41 km. 안테나(MHz, 파일 수) 400(391) · 2000(24) · 600(16) · 900(11) · 200(4) · 80(2) |
| 관로 pipe, MALA .rd3 | 75 파일 · 6 폴더, 트레이스 47,397, 1.21 km. 시간창 약 190 ns(일부 50 ns), 처리본으로 보임 |
| 철근 rebar, IDS .dt | 125 파일 · 64 존, 트레이스 452,556, 1.65 km. 2000 MHz(포산 현장) 84, 2600 MHz(시편) 41 |
| 합계 | 448 + 75 + 125 = 648 파일 (147 존) |
| 제외 | 268 파일(ReflexW 파생·사본 114, 터널 라이닝 144 등, 사유별 표는 readme §6) |

### 분할

분할 없음. 라벨이 없어 학습·평가 분할을 만들지 않고, 추론·속도 추정 시험 전용으로 쓴다.

## 물리 스케일

| 파라미터 | 값 | 비고 |
|---|---|---|
| 트레이스 간격 | 관로 0.024 m(대부분), 0.01·0.016·0.02·0.044 m / 철근 0.002·0.004 m / MALA 0.0256·0.024 m | .dt 헤더, 바퀴 엔코더 |
| 샘플 수 | 512 대부분(일부 256·384·1024·2048) | |
| 시간창 | 관로 400 MHz는 30·50·40 ns가 대부분, 철근 6–10 ns | dt = 시간창 ÷ 샘플 수 (예: 30 ns/512 = 0.0586 ns) |
| 시간영점 | IDS는 K2 기록값(`t0_sample_k2`), 없으면 자동 추정(`t0_sample_peak`) | 기준이 섞여 있고 차이는 최대 약 1 ns(깊이 약 5 cm @0.1 m/ns) |
| 속도 | **0.10 m/ns 가정** | 실측값 없음. 헤더 PropVel은 화면 표시용 설정값 |

IDS TR 안테나는 지면결합형이라 4TU 같은 공기층이 없고, 화면 맨 위의 굵은 띠는 직접파(송수신 결합파 + 지표 반사)다.

## Dataset Creation

- **출처**: GPR Group of Guangzhou University(Hai Liu 교수팀), *GPR DATASET*, Zenodo 2025-01-13, DOI 10.5281/zenodo.14637589 (원본 Data Set.zip 3.78 GB).
- **장비**: IDS GeoRadar(K2 소프트웨어 v02.02) 지면결합 TR 안테나, 바퀴 엔코더 거리 트리거 / MALA RAMAC(안테나 미기록).
- **측정 장소·시기**: 관로 - 광저우 대학성 캠퍼스 도로, 2018-09–2020-06(MALA는 날짜 미기록) / 철근 - 포산 주거지 현장·시편, 2018-01–2019-04.
- **가공(받은 가공본)**: `.dt` → npy/png 변환, 파일마다 물리 파라미터 json, 파일·존 목록 CSV, 중복·손상·과제 범위 밖(터널 라이닝) 파일 제외. 진폭은 K2가 저장한 int16 그대로(처리 없음).
- **라벨링**: 없음.

## Considerations (알려진 결함·편향·한계)

1. **정답이 전혀 없다**(관로 위치·심도·관종, 박스, 측선 위치도, GPS). 깊이 검증에 쓸 수 없다.
2. **속도(유전율) 실측값이 없다.** 깊이는 쌍곡선 맞춤으로 측선별 속도를 추정해야 한다.
3. **시간영점 기준이 섞여 있다**(직접파 시작점 또는 정점). 정밀 비교 전에 통일한다.
4. **MALA 75개는 원시 자료가 아닌 것으로 보인다**: 배경 제거·게인이 이미 적용된 것으로 보이고, 직접파가 없어 시간영점을 0으로 두었으며, 안테나·날짜 기록이 없다. 학습·시험에 섞을 때는 따로 구분한다.
5. 약한 정답으로 쓸 수 있는 것: 철근(규칙 간격 쌍곡선), `pipe/2020wate_*`(수도관으로 보이며 측선 간 일관성 검사), `pipe/1.15_*`(같은 구간의 200/400/600/900 MHz 반복 측정으로 추정, 주파수별 비교). 모두 추정이다.
6. GPS·측선 위치도 없음. 같은 존 안의 평행 측선 번호(`line_no_in_zone`)만 있다.
7. 진폭이 raw int16이고 ±32767 포화 비율(`saturated_ratio`)이 파일마다 기록돼 있다.

## Licensing & Citation

- **라이선스: CC BY 4.0.** 사용·수정·재배포 가능, **출처 표시 필수**. 가공본도 같은 조건.
- 데이터: GPR Group of Guangzhou University (2025). GPR DATASET [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14637589
- 원 저자가 인용을 요청한 논문:
  - Liu, H., Lin, C., Cui, J., Fan, L., Xie, X., & Spencer, B. F. (2020). Detection and localization of rebar in concrete by deep learning using ground penetrating radar. Automation in Construction, 118, 103279
  - Liu, H., Yue, Y., Lai, S., Meng, X., Du, Y., Cui, J., & Spencer, B. F. (2023). Evaluation of the antenna parameters for inspection of hidden defects behind a reinforced shield tunnel using GPR. Tunnelling and Underground Space Technology, 140, 105265
  - Liu, H., Yue, Y., Lian, Y., Meng, X., Du, Y., & Cui, J. (2024). Reverse-time migration of GPR data for imaging cavities behind a reinforced shield tunnel. Tunnelling and Underground Space Technology, 146, 105649
  - Wang, X., Chen, J., & Liu, H. (2024). Enhancing Full Waveform Inversion of Field GPR Data: A Source-Independent Approach with Dynamic Reference Selection via SE-Wave-U-Net. Geophysics, 90(1)

## 이 프로젝트에서의 사용

- **역할**: 미라벨 추론과 쌍곡선 곡률 기반 속도 추정 시험(M3-7). 깊이 정확도 검증에는 쓰지 않는다(정답 없음 - 4TU 사용).
- **쓰는 파생 데이터**: 가공본의 `05_radargrams_npy`·`01_메타데이터`를 읽기만 한다. 400 MHz 관로 IDS 파일이 주 대상이고 MALA 처리본은 구분한다.
- **관련 체크리스트**: M3-7(cv-detection-expert, gpr-processing-expert).
- 근거 문서: [readme_guangzhou.md](../readme_guangzhou.md), [prior_art.md](../../research/prior_art.md).
