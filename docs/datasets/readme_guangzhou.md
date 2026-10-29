<!-- 원본: C:\workspace\GPR_Data\광저우대_중국_관로GPR_가공본_261016\README_사용법.docx | docx_dump(R1)·python-docx로 추출, 2026-10-29. 원문 그대로 옮김(서식만 Markdown으로 변환). -->

# 광저우대 GPR 데이터셋(중국) — 가공본 사용 설명서

- 작성: 데이터 가공본 제공처 (공개용으로 과제 정보 삭제)
- 작성일: 2026-10-16
- 원본: *GPR DATASET* — GPR Group of Guangzhou University(광저우대 Hai Liu 교수팀), Zenodo 2025-01-13, DOI 10.5281/zenodo.14637589
- 라이선스: CC BY 4.0(원본). 가공본도 같은 조건으로 사용·수정·재배포할 수 있으며 **출처 표시가 필수**입니다(인용 문구는 10절)

## 1. 한 줄 요약

중국 광저우 대학성(대학 캠퍼스 도로)에서 IDS 지면결합 GPR(주로 400 MHz)로 지하관로 위를 측정한 원시 B-scan, 그리고 포산 철근콘크리트를 2.0/2.6 GHz로 측정한 자료입니다. 원본은 IDS K2 전용 이진 형식(.dt)이고 설명 문서가 하나도 없어 그대로는 열기 어렵습니다. 이 가공본은 ① .dt를 npy/png로 바꾸고, ② 파일마다 물리 파라미터(시간창·샘플 간격·트레이스 간격·시간영점)를 json으로 붙이고, ③ 파일·존 목록을 CSV로 만들고, ④ 중복·손상·과제 범위 밖 파일을 걸러 낸 것입니다.

가장 큰 가치: 시간·거리 축척이 헤더에 다 들어 있는 실제 도시 도로 관로 B-scan이 10.41 km(이 중 400 MHz 391개 파일 6.86 km) 있습니다. AI Hub 자료(깊이 축척 없음)와 달리 쌍곡선 곡률로 속도·심도를 계산할 수 있어, 탐지 모델 사전학습·속도 추정·전처리 파이프라인 시험에 적합합니다.

가장 큰 한계: **정답이 전혀 없습니다.** 관로 위치·심도·관종, 라벨 박스, 측선 위치도, GPS가 모두 없습니다. 4TU 네덜란드 데이터처럼 깊이 정확도를 검증하는 용도로는 쓸 수 없습니다.

## 2. 폴더 구성

```
광저우대_중국_관로GPR_가공본_261016\
├ README_사용법.docx              ← 이 문서
├ 00_원본참고\                     동봉 MATLAB 리더·처리 스크립트(원본 GBK + _UTF8 사본), 데이터 분석 메모
├ 01_메타데이터\
│   ├ radargrams_index.csv        ★ 변환한 측선 파일 648개의 물리 파라미터(1행 1파일)
│   ├ zones_summary.csv           존(측정 묶음) 147개별 날짜·안테나·파일 수·길이
│   └ excluded_files.csv          변환하지 않은 레이더 파일 268개와 사유
├ 02_참고이미지\                   미리보기 PNG 5장, 원 저자 ReflexW 처리 단계 JPG 6장
├ 03_radargrams_png_raw\{대상}\{미션_존}\   파일명.png — 원시 진폭 8비트 그레이, 처리 없음(학습 입력용)
├ 04_radargrams_png_view\{대상}\{미션_존}\  파일명_segNN.png — 처리 후 거리·시간·깊이(가정 속도) 축(눈으로 검토용)
├ 05_radargrams_npy\{대상}\{미션_존}\       파일명.npy(int16, [샘플, 트레이스]) + 파일명.json(물리 파라미터)
└ 06_scripts\                      ids_dt.py(.dt/.rd3 리더), build_gz_package.py(이 패키지 생성), example_load.py
```

- 대상은 pipe(관로)와 rebar(철근) 두 가지이고, 미션_존은 원본 폴더명에서 .MIS·.ZON을 뗀 것입니다(예: 1030.MIS\181030AA.ZON → 1030_181030AA)
- MALA 형식 파일은 pipe\MALA_{원본 폴더명}\에 있습니다(예: MALA_0704)
- 파일 ID(file_id)는 대상/미션_존/파일명 형식이며(예: pipe/2020wate_200113AA/LID10002) CSV·json·폴더 경로가 모두 이 ID로 연결됩니다
- 넣지 않은 것: 원본 zip 전체(3.8 GB, Zenodo에서 받을 수 있음)와 터널 라이닝 자료(과제 범위 밖, 6절)

## 3. 데이터 규모

| **항목** | **값** |
|---|---|
| 관로 pipe — IDS .dt | 448개 파일·77개 존, 트레이스 485,205개, 길이 합계 10.41 km. 안테나(MHz, 파일 수) 400(391)·2000(24)·600(16)·900(11)·200(4)·80(2) |
| 관로 pipe — MALA .rd3 | 75개 파일·6개 폴더, 트레이스 47,397개, 길이 합계 1.21 km. 시간창 약 190 ns(일부 50 ns), 처리본으로 보임(4절) |
| 철근 rebar — IDS .dt | 125개 파일·64개 존, 트레이스 452,556개, 길이 합계 1.65 km. 2000 MHz(포산 현장 foshanJL) 84개, 2600 MHz(시편 yangben) 41개 |
| 측정 장소·시기 | 관로: 광저우 대학성 캠퍼스 도로, 2018-09\~2020-06(MALA는 날짜 미기록) / 철근: 포산 주거지 현장·시편, 2018-01\~2019-04 |
| 장비 | IDS GeoRadar(K2 소프트웨어 v02.02), 지면결합 TR 안테나, 바퀴 엔코더 거리 트리거 / MALA RAMAC(안테나 미기록) |
| 라벨·정답 | 없음(5절) |

## 4. 물리 축척(픽셀 → 미터) — 반드시 읽을 것

모든 값은 radargrams_index.csv와 각 파일명.json에 파일별로 들어 있습니다.

| **파라미터(열 이름)** | **값** | **비고** |
|---|---|---|
| 트레이스 간격 trace_dx_m | 관로 0.024 m(대부분)·0.01·0.016·0.02·0.044 m, 철근 0.002·0.004 m, MALA 0.0256·0.024 m | .dt 헤더 data_x_cell(= 바퀴 2 mm × 압축 배수). 바퀴 엔코더 구동이라 간격이 일정 |
| 안테나 antenna_MHz | 관로 80·200·400·600·900·2000 / 철근 2000·2600 | .dt 헤더 MaxFreq. MALA는 기록 없음 |
| 샘플 수 n_samples | 512가 대부분(일부 256·384·1024·2048) |  |
| 시간창 time_window_ns | 관로 400 MHz는 30·50·40 ns가 대부분, 철근 6\~10 ns | 헤더 sweepTi |
| 샘플 간격 dt_ns | 시간창 ÷ 샘플 수(예: 30 ns/512 = 0.0586 ns) | 헤더의 data_y_cell 대신 이 값을 쓸 것 |
| 시간영점 t0_sample_used | IDS는 K2 기록값(t0_sample_k2), 없으면 자동 추정(t0_sample_peak) | 아래 주의 참조 |
| 속도 velocity_assumed_m_per_ns | **0.10 m/ns(가정)** | 실측값 없음. 그림의 깊이 축과 depth_*_assumed 열은 이 가정값으로 계산 |
| 샘플당 깊이 depth_per_sample_m_assumed | dt × 0.10 / 2 | 예: 0.0586 ns → 2.9 mm |
| 최대 깊이 depth_max_m_assumed | (샘플 수 − t0) × 샘플당 깊이 | 400 MHz 30 ns 창 → 약 1.4 m, 50 ns 창 → 약 2.3 m |

변환식:

```
t[ns]      = (sample − t0_sample_used) × dt_ns          # 시간영점 이후 왕복시간
depth[m]   = t × v / 2                                   # v는 미정, 0.10 m/ns는 가정
distance[m] = trace_index × trace_dx_m
```

주의할 점

- **속도(유전율) 실측값이 없습니다.** 헤더의 PropVel(prop_vel_header_m_per_ns, 대부분 0.1 m/ns)은 K2 화면 표시용 설정값이라 실측 보정값이 아닙니다. 깊이를 다룰 때는 쌍곡선 맞춤으로 측선별 속도를 추정해 쓰길 권합니다(이 데이터의 가장 좋은 연습 문제입니다)
- **시간영점 기준이 섞여 있습니다.** K2 기록값(t0_sample_k2, Igr000N.Bkg 5행)은 원 저자 MATLAB 처리가 쓰는 값인데, 측정 설정에 따라 직접파 시작점(400 MHz 30 ns 창은 대부분 20\~21샘플) 또는 직접파 정점(50·96 ns 창은 대부분 35\~36샘플)을 잡아 두었습니다. 차이는 최대 약 1 ns(0.1 m/ns에서 깊이 약 5 cm)입니다. 직접파 정점 자동 추정값(t0_sample_peak)을 함께 적어 두었으니 정밀 비교 전에 둘을 비교해 기준을 통일하세요
- IDS TR 안테나는 지면결합형이라 4TU 데이터 같은 공기층이 없고, 화면 맨 위의 굵은 수평 띠는 직접파(송수신 결합파+지표 반사)입니다
- **진폭**: npy는 K2가 저장한 int16 값 그대로입니다. 원 저자 스크립트(00_원본참고\IDS_data_process_UTF8.m)는 시간영점에서 자르고 → Bkg 배경 트레이스를 빼고 → 대역통과 필터 → Stc 곡선으로 나눕니다. 이 두 곡선은 json의 k2_bkg_trace·k2_stc_curve에 넣어 두었습니다. 포화값(±32767) 비율은 saturated_ratio
- **MALA 75개는 원시 자료가 아닌 것으로 보입니다.** 평균 트레이스 성분 비율(mean_trace_ratio)이 IDS 원시는 약 0.9인데 MALA는 대부분 0.1 이하라 배경 제거가 이미 된 것으로 보이고, MALA_0704·0707·0806은 후반/초반 진폭비(late_early_amp_ratio)가 약 1이라 게인도 적용된 것으로 보입니다. 직접파가 없어 시간영점을 잡을 수 없으므로 0으로 두었습니다. 안테나·측정 날짜도 기록돼 있지 않습니다. 학습 데이터에 섞을 때는 따로 구분하세요
- GPS·측선 위치도는 없습니다. 같은 존 안의 평행 측선 번호(line_no_in_zone)만 있습니다
- spectrum_peak_MHz는 트레이스 진폭 스펙트럼의 최대 주파수(참고용)입니다. 지면결합 안테나는 보통 공칭 주파수보다 낮게 나옵니다

## 5. 정답이 없다는 점과 대안

- 라벨 파일(박스·마스크·CSV)·관로도·매설 심도표·측선 위치도가 하나도 없습니다. Zenodo 설명도 "다양한 관로 배치·심도"라고만 적고 수치는 없습니다
- 약한 정답(weak label)으로 쓸 수 있는 것
  - 철근(rebar): 쌍곡선이 규칙 간격으로 반복됩니다(예: rebar/foshanJL_190424AA/LID10001은 약 16 cm 간격 14개). 쌍곡선 자동 검출기의 검증·약라벨 생성에 쓰기 좋습니다
  - pipe/2020wate_*: 미션 이름상 수도관(water) 측정으로 보이며, 같은 관로를 여러 측선이 반복해 가로지르는 것으로 보여 측선 간 일관성 검사에 쓸 수 있습니다. 02_참고이미지에 원 저자가 ReflexW로 처리한 단계별 화면(raw → dewow → 시간영점 → 배경 제거 → 필터 → 게인)이 있습니다
  - pipe/1.15_*: 2020-01-15\~16에 존마다 안테나를 200·400·600·900 MHz로 바꿔 파일당 107\~175 m씩 측정했으므로 같은 구간의 주파수별 반복 측정으로 추정됩니다. 주파수별 탐지 성능 비교에 쓸 수 있습니다
- 박스 라벨이 필요하면 04_radargrams_png_view를 보며 쌍곡선 꼭짓점을 표시하는 작업이 필요합니다. 학습 입력은 03 raw png 또는 05 npy를 쓰고, 처리는 각자 파이프라인으로 하세요

## 6. 메타데이터 파일

### radargrams_index.csv (파일 1행)

| **열** | **설명** |
|---|---|
| file_id, target, format, mission, zone, file | 파일 ID와 분해값, 형식(IDS .dt / MALA .rd3) |
| source_path | 원본 Data Set\ 아래 경로 |
| header_version, k2_software, n_acq_header | .dt 헤더 버전(관로·철근은 모두 4), K2 버전, 존 안 취득 순번 |
| acq_datetime | 취득 일시(헤더, 현지 시각) |
| antenna_MHz, antenna_config_ini | 안테나 주파수(헤더 MaxFreq), K2 안테나 설정명(예: TR400_WHE50) |
| n_samples, n_traces, time_window_ns, dt_ns, trace_dx_m, length_m | 4절 |
| t0_sample_k2, t0_sample_peak, t0_sample_used, t0_source | 시간영점(4절) |
| velocity_assumed_m_per_ns, depth_per_sample_m_assumed, depth_max_m_assumed | 가정 속도 0.10 m/ns 기준 깊이 환산 |
| prop_vel_header_m_per_ns | K2 표시용 속도 설정값(실측 아님) |
| spectrum_peak_MHz | 스펙트럼 최대 주파수(참고) |
| stacking_ini | 스태킹 횟수(ini), MALA는 .rad STACKS |
| line_no_in_zone, zone_lines | 존 안 평행 측선 번호, 존의 측선 수(.ZON ScansioneL) |
| marker_traces | 마커 트레이스 수(관로·철근은 1개 파일 외 전부 0) |
| saturated_ratio | 포화 샘플(±32767) 비율 |
| mean_trace_ratio, late_early_amp_ratio | 처리 여부 지표(4절 MALA 항목) |
| n_view_segments | 04 검토용 PNG 장수(2,500트레이스씩 분할) |
| note | 특이사항(Bkg 없음, 헤더 NAcq 불일치, ReflexW 프로젝트 있음, MALA 처리본 등) |

### zones_summary.csv (존 1행)

존별 측정 날짜·안테나·시간창·트레이스 간격·파일 수·트레이스 수·길이 합계·측선 수, ReflexW 프로젝트 유무.

### excluded_files.csv (제외 파일 1행)

| **사유** | **개수** |
|---|---|
| ReflexW 가져오기 원시파일(.dt와 같은 측정의 파생본) | 82 |
| 터널 라이닝 .srd(GSSI DZT 계열 추정, 과제 범위 밖) | 76 |
| 터널 라이닝 측정(과제 범위 밖, 변환 안 함) | 67 |
| ReflexW ASCII 하위폴더 사본(존 폴더의 같은 이름 .dt와 바이트 동일) | 28 |
| ReflexW SEG-Y 내보내기(처리본) | 4 |
| 트레이스 0개(헤더만 있음) | 3 |
| IDS 측정(001.MIS\190321AA.ZON\같은 이름 .dt)을 처리해 MALA 형식으로 다시 내보낸 것(트레이스 수 동일) | 3 |
| 손상 파일(12,288 B, 헤더 15레코드 미만) | 3 |
| 같은 폴더 ASCII_707-01\ASCII의 MALA 파일 76개와 동일한 압축본 | 1 |
| 터널 .dt 변형(V0 헤더, 미해독) | 1 |

## 7. 사용 예

### 7-1. npy + json으로 읽기(가장 간단)

```
import numpy as np, json
fid = 'pipe/2020wate_200113AA/LID10002'
D = np.load(f'05_radargrams_npy/{fid}.npy').astype(np.float32)        # [512 샘플, 트레이스]
p = json.load(open(f'05_radargrams_npy/{fid}.json', encoding='utf-8'))
t = (np.arange(D.shape[0]) - p['t0_sample_used']) * p['dt_ns']        # ns
dist = np.arange(D.shape[1]) * p['trace_dx_m']                         # m
depth = t * 0.10 / 2                                                   # m(가정 속도, 직접 추정한 값으로 바꿀 것)
```

06_scripts\example_load.py \<패키지폴더\> pipe/2020wate_200113AA/LID10002 [속도]를 실행하면 시간영점 보정·배경 제거·게인 후 시간/깊이 축으로 보여 줍니다.

### 7-2. 원본 .dt를 직접 읽기

06_scripts\ids_dt.py의 read_ids_dt(경로)가 (헤더 dict, int16 [샘플, 트레이스], 트레이스 코드)를 돌려줍니다. 형식 요약: 고정 길이 레코드 연속, 레코드 크기 = 4 + 2×샘플 수(파일 바이트 4\~5의 uint16), 앞 15레코드가 헤더(15번째 H 레코드에 트레이스 수·샘플 수·안테나·시간창·트레이스 간격), 이후 트레이스마다 코드 4바이트 + int16 little-endian 샘플. 원본 643개 전부 이 규칙으로 크기가 맞는 것을 확인했습니다. 부속 파일은 read_bkg_stc()(배경·게인·시간영점), read_ini(), read_zon(), MALA는 read_mala().

### 7-3. 학습용 입력

- 03_radargrams_png_raw: 파일별 99.5% 진폭으로 정규화한 8비트 그레이(처리 없음). 세로 1 px = 1 샘플, 가로 1 px = 1 트레이스(관로 400 MHz는 대부분 2.4 cm)
- 04_radargrams_png_view: dewow → 배경 제거(501트레이스 이동평균) → AGC → 클립. 사람이 보기 위한 것이라 진폭이 왜곡돼 있으니 학습 입력으로는 쓰지 마세요. 가로 2,500트레이스마다 한 장(_seg01, _seg02…), x축은 측선 시작점 기준 거리

## 8. 권장 활용 순서

1. 04_radargrams_png_view에서 pipe/2020wate_200113AA, pipe/1030_181030AA, pipe/1.15_200115AA 등을 열어 관로 쌍곡선과 시간영점을 눈으로 확인
2. 쌍곡선 맞춤으로 측선별 속도를 추정해 깊이 환산(헤더 속도 대신)
3. 400 MHz 관로 B-scan으로 쌍곡선 검출기 사전학습(자기지도 또는 약라벨) → AI Hub·4TU·국내 실측으로 미세조정·검증
4. pipe/1.15_*로 주파수별(200/400/600/900 MHz) 탐지 성능 비교
5. 국내 실측(GPR, MALA rd3) 데이터가 들어오면 같은 리더·처리로 비교

## 9. 재현 방법

- 06_scripts\build_gz_package.py \<Data Set 폴더\> \<데이터메모.md\> \<출력폴더\> — 이 패키지 생성(numpy, scipy, matplotlib 필요)
- 06_scripts\ids_dt.py — IDS .dt·MALA .rd3 리더
- 06_scripts\example_load.py — 읽기 예제
- 원본: Zenodo 14637589의 Data Set.zip(3.78 GB, MD5 a20da497549c01f7f079e68e46ed7c87)을 풀면 Data Set\ 폴더가 나옵니다

## 10. 라이선스·인용

- 라이선스: Creative Commons Attribution 4.0 International(CC BY 4.0). 상업적 이용·수정·재배포가 가능하고 출처 표시가 필수입니다
- 데이터 인용: GPR Group of Guangzhou University (2025). GPR DATASET [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14637589
- 원 저자가 데이터와 함께 인용을 요청한 논문
  - Liu, H., Lin, C., Cui, J., Fan, L., Xie, X., & Spencer, B. F. (2020). Detection and localization of rebar in concrete by deep learning using ground penetrating radar. Automation in Construction, 118, 103279
  - Liu, H., Yue, Y., Lai, S., Meng, X., Du, Y., Cui, J., & Spencer, B. F. (2023). Evaluation of the antenna parameters for inspection of hidden defects behind a reinforced shield tunnel using GPR. Tunnelling and Underground Space Technology, 140, 105265
  - Liu, H., Yue, Y., Lian, Y., Meng, X., Du, Y., & Cui, J. (2024). Reverse-time migration of GPR data for imaging cavities behind a reinforced shield tunnel. Tunnelling and Underground Space Technology, 146, 105649
  - Wang, X., Chen, J., & Liu, H. (2024). Enhancing Full Waveform Inversion of Field GPR Data: A Source-Independent Approach with Dynamic Reference Selection via SE-Wave-U-Net. Geophysics, 90(1)
- 원 저자 연락처: Zenodo 레코드 14637589 참조(공개용으로 주소 삭제)
- 원본의 상세 구조·검증 결과는 00_원본참고\데이터메모_광저우대_Zenodo14637589.md 참조

## 11. 문의

가공본 제공처에 문의(공개용으로 연락처 삭제).
