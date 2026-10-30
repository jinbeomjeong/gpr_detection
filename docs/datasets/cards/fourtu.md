---
pretty_name: "4TU GPR Dataset with Ground-Truthing from Utility Surveying in the Netherlands"
license: cc0-1.0
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
  - buried-utility
  - depth-validation
  - air-coupled
  - netherlands
  - segy
source_datasets:
  - original
annotations_creators:
  - expert-generated
---

# 4TU 네덜란드 매설관 GPR 데이터셋

| Item | Value |
|---|---|
| Summary | Air-coupled 500 MHz GPR lines at 13 Dutch construction sites with trial-trench ground truth (position, depth, discipline) of 396 utilities |
| Format | Original SEG-Y (759 radargrams, 13 zips) + 125 ground-truth PNG tables; processed copy has npy (int16, [512 samples, traces]) + json scale, and CSV transcriptions |
| Size | 13 sites, 125 activities (95 with ground truth), 759 radargrams, 396 ground-truth utilities (depth 0.22–4.57 m, median 0.70 m) |
| Labels | Depth lines only (no boxes). Discipline codes E / T / OGC / S / W / U |
| Physical scale | Yes: dt 0.098 ns (0.097 ns for site 01, 0.195 ns for site 11 and 09.1/09.2, 0.117 ns for 2 lines), 512 samples, 0.02 m per trace, per-activity relative permittivity (auto-estimated t0) |
| Role in this project | Depth validation (M2-8), no tuning on it; per-site detection cross-domain evaluation (M3-3) |
| License | CC0 1.0 (original); citation recommended |

## Dataset Summary

네덜란드 13개 공사현장에서 500 MHz **공기결합** GPR로 측선을 긋고, 같은 자리를 시험굴착(trial trench)해 실제 관로의 위치·깊이·관종을 기록한 데이터다. 원본은 SEG-Y 원시파일과 "그림으로 그려진" 정답표(png)라 그대로는 학습에 쓰기 어렵다. 쌍곡선 꼭짓점 깊이를 실제 굴착 깊이와 비교할 수 있는 공개 데이터라는 점이 가장 큰 가치이며, 전처리(time-zero·속도) 파이프라인과 깊이 추정 정확도 검증용으로 쓴다([readme_4tu.md §1](../readme_4tu.md)).

## Dataset Structure

### 형식

- 로컬 위치: `C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015`(원본은 `00_원본`에 보존).
- `01_메타데이터/`: `utilities_groundtruth.csv`(시굴 정답 396행), `radargrams_index.csv`(759행 물리 파라미터), `activities_metadata_ko.csv`(활동 125개 조사 조건), `utilities_metadata_list.csv`(관종·재질·관경 목록).
- `05_radargrams_npy/{활동}/PathN.npy`(int16 `[512 샘플, 트레이스]`) + `PathN.json`(물리 파라미터). 활동 ID는 `01.1`…`013.6`(현장 10–13은 앞에 0), 현장 11은 파일명이 `Profile_NNNN`.
- 정답표 열: `activity_id, site, activity_no, trench_no, seq, distance_m, depth_m, code, discipline, 관종, note`. 코드 E 전력, T 통신, W 상수, S 하수, OGC 가스·유류·화학, U 불명, X 관로 아님.
- 메타데이터 항목 정의는 원 코드북 추출본 [codebook_4tu.md](../codebook_4tu.md).

### 개수

| 항목 | 값 |
|---|---|
| 공사현장 | 13 |
| 조사활동 | 125 (정답 있는 활동 95, 없는 활동 30) |
| 측선 radargram | 759 (활동당 1–26개 — 원본 Readme는 2–26으로 적었으나 파일 1개뿐인 활동이 24개, 대부분 1–12 m). 길이 30 m 이상 약 50개는 여러 측선을 지그재그로 한 번에 기록 |
| 정답 관로 | 396 (전력 134 · 통신 125 · 가스유류화학 42 · 하수 36 · 상수 36 · 불명 23) |
| 정답 깊이 | 0.22–4.57 m, 중앙값 0.70 m (깊이 없는 관로 8, 거리 없는 관로 21) |

### 분할

- **현장(site 1–13) 단위**로 나눈다([ADR-003](../../architecture/ADR-003-data-split-and-classes.md) 결정 4). 깊이 검증용 dev/test 분할은 없다: 파라미터를 4TU로 튜닝하지 않고 13개 현장 전체를 시험에만 쓴다. 현장 단위 분할은 검출 교차 도메인 평가(M3-3)에만 쓴다.
- 깊이표는 활동 단위라서 검출 깊이 집합과 헝가리안 매칭으로 MAE를 구한다.

## Dataset Creation

- **출처**: ter Huurne 외(Univ. Twente), *A Ground Penetrating Radar dataset with ground-truthing from real-life utility surveying activities in the Netherlands*, 4TU.ResearchData 2023-09-05, DOI 10.4121/96303227-5886-41c9-8607-70fdd2cfe7c1.v1. 논문 Data in Brief 54 (2024) 110329.
- **장비**: InfraRadar 500 MHz 공기결합(air-launched) 안테나, 바퀴 엔코더, GNSS SP80. 속도는 Reflex-W 9.1.3 쌍곡선 피팅으로 구한 값(εr = (c/v)²)이며 원 논문은 깊이 오차를 보고하지 않았다([prior_art §5](../../research/prior_art.md)).
- **정답 생성**: 시험굴착으로 줄자·GNSS 측정. 원본 정답은 그림이라, 받은 가공본이 png 125장을 프로그램으로 읽어 CSV로 옮겼다(표 격자 검출 + 템플릿 정합 + OCR 교차 확인, 애매한 7건은 사람이 대조).
- **가공(받은 가공본)**: SEG-Y → npy/png 변환, 파일마다 물리 파라미터(샘플 간격·속도·지표면 위치) 부착, 메타데이터 한글화. SEG-Y는 little-endian·피코초 단위의 RadarMap 변종이라 일반 SEG-Y 라이브러리를 그대로 쓰면 값이 틀어질 수 있다.

## 물리 스케일

| 파라미터 | 값 | 비고 |
|---|---|---|
| 트레이스 간격 | 0.02 m | 바퀴 엔코더 |
| 샘플 수 | 512 | 전 파일 동일 |
| 샘플 간격 dt | 0.098 ns(582개) / 0.097 ns(92개, 현장 01 전부 — 같은 약 50 ns 창, 소수 셋째 자리만 다름) / 0.195 ns(83개, 100 ns: 현장 11 전부와 09.1·09.2) / 0.117 ns(2개) — readme_4tu는 0.098·0.097을 합쳐 674개로 적음, M2-9에서 정정 | 0.117 ns는 09.4 Path1·09.6 Path4 |
| 비유전율 | 활동별 8.16–19.46 | 조사자가 기록한 값. SEG-Y 헤더의 7/9/11은 장비 설정값 |
| 지표면 샘플 `t0_sample_est` | 파일마다 자동 추정(15–99, 중앙값 43) | 같은 활동 안에서도 오가는 경우가 있어 `t0_sample_act_median`을 함께 기록 |

깊이 환산식(가공본): `depth = (sample − t0) × dt × (0.3/√εr) / 2`. **공기결합 영점은 지표 반사**로 잡는 것이 현재 규약이다(M2-9에서 재검토 중 — M2-7 탐색에서 영점 사건이 단일 띠로 나타남)(직접파 영점이면 h/√εr만큼 깊어짐, 근거 [depth_physics.md §2·§7](../../research/depth_physics.md)).

## Considerations (알려진 결함·편향·한계)

1. **깊이 기준점(관 상단인지 중심인지)이 원본에 없다.** 상단·중심 두 가정을 모두 보고한다.
2. **측선(PathN)의 몇 m 지점이 굴착 단면인지 기록이 없다.** 깊이 비교는 측선·활동 단위로만 가능하고 수평 위치 비교는 어렵다(`distance_m`을 트레이스 위치로 쓰지 않는다).
3. **번들(bundle)**: 표의 관로 수와 Metadata의 관로 수(가닥 수)가 다를 수 있다.
4. **정답 없는 활동 30개**는 음성 샘플로 쓸 수 있으나 실제로 관로가 없다는 보장은 없다.
5. 트레이스 헤더의 GNSS 좌표는 측선 내내 거의 상수라 위치로 쓸 수 없다(원본 결함).
6. 정답표 전사 오류 가능성은 0이 아니다(중요한 비교는 원본 png를 확인).
7. 네덜란드 보도·모래 지반·소구경 케이블 위주라 국내 아스팔트·대구경 관과 환경이 다르다.
8. **라벨(박스)이 없다.** 박스가 필요하면 쌍곡선 꼭짓점을 손으로 표시해야 한다. M2-8은 손으로 고른 수동 꼭짓점 후보를 쓴다.
9. **영점·곡률 탐색 결과(활동 4개, 진행 중)**: 41개 창이 모두 단일 사건으로 직접파와 지표 반사가 분리되지 않았고, 쌍곡선 모양은 안테나 높이 h ≳ 0.1 m와 맞지 않았다. 메타 `t0` 두 값이 오가는 것은 직접파/지표 반사가 아니라 같은 띠 wavelet의 두 엽 차이로 해석됐다. 현장 11 활동 011.14는 앞 약 48샘플이 0인 mute 자료이고 메타 εr 15.18이 의심된다(검증 실패). 이 탐색은 판정이 아니며 npy로 확인한 것은 4개 활동뿐이다([fourtu_explore.md](../../processing/fourtu_explore.md)). **공기결합 영점 규약은 M2-9에서 재검토 중**이라 이 카드의 영점 서술은 개정될 수 있다.
10. 메타 εr은 실측 보증이 없다(01·04·010 활동은 헤더 기본값 9와 같음).
11. radargram 수는 759개다(가공본 색인 행 수 = 원본 zip 안 SEG-Y 수). 조사 문서 prior_art §5에 959로 적혔던 것은 오기라 2026-10-29에 759로 고쳤다.

## Licensing & Citation

- **라이선스: CC0 1.0** (원본). 가공본도 동일 조건. 논문 인용은 권장.
- 데이터: ter Huurne et al. (2023). 4TU.ResearchData, https://doi.org/10.4121/96303227-5886-41c9-8607-70fdd2cfe7c1.v1
- 논문: ter Huurne et al. (2024). *A Ground Penetrating Radar dataset with ground-truthing from real-life utility surveying activities in the Netherlands.* Data in Brief, 54, 110329.

## 이 프로젝트에서의 사용

- **역할**: 깊이 검증(M2-8). 활동 단위로 같은 관을 묶어 헝가리안 매칭, MAE, 공공측량 작업규정 제186조 기준(깊이 ±30 cm) 판정률, 상단/중심 두 가정, 무작위 깊이 기준선. **4TU로 튜닝하지 않는다.** 검출 교차 도메인 평가(M3-3)에서는 현장 단위로 쓴다.
- **쓰는 파생 데이터**: 가공본의 `05_radargrams_npy`와 `01_메타데이터`를 읽기만 한다. 프로젝트 안에 별도 파생 데이터셋은 없다.
- **영점 규약**: 공기결합이므로 지표 반사를 영점으로, v는 메타 εr 고정 또는 굴절 모델(곡률 피팅 금지)이 현재 프로젝트 규약이며 M2-9에서 재검토 중이다.
- **관련 체크리스트**: M2-7(탐색), M2-8(깊이 검증), M2-9(규약 정리), M3-3.
- 근거 문서: [readme_4tu.md](../readme_4tu.md), [codebook_4tu.md](../codebook_4tu.md), [fourtu_explore.md](../../processing/fourtu_explore.md), [depth_physics.md](../../research/depth_physics.md), [prior_art.md](../../research/prior_art.md), [ADR-003](../../architecture/ADR-003-data-split-and-classes.md).
