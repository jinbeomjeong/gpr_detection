# GPR-SD(중국 USTC 도로 GPR) 측정·처리 사양 출처 조사

조사일: 2026-11-26 · 모드: 절약 · 조사자: web-research-expert

**가정**: "근거 논문"은 저장소 README가 인용을 요청한 두 편(Zhou 2018, Wang 2023)과, 같은 연구실(USTC 계산기학원 Huanhuan Chen·Xiren Zhou)이 같은 도로 자료를 쓴 공개 논문으로 범위를 넓혀 해석했다. 두 인용 논문은 본문이 유료라 열람하지 못했다. 장비 정보는 같은 연구실의 공개 논문(arXiv, Nature Communications)에서 얻었다. 다만 이 논문들이 쓴 자료가 GPR-SD 파일과 **같은 측선인지는 어느 출처도 명시하지 않는다**.

## 조사 요약
- **장비**: 같은 연구실의 공개 논문 세 편이 도로 탐사 장비를 **GSSI SIR-30**(제어부)이라고 적었다[3][4][5]. 그중 두 편은 **200 MHz 안테나**라고 적었다[4][5]. 저장소는 "200 to 400MHZ"라고만 썼다[1][2]. 안테나 모델명과 지면결합 여부를 밝힌 출처는 없다. GSSI의 200 MHz 안테나 제품은 차폐형(Model 5106/5106A)이지만[9], 이 연구실이 그 제품을 썼다는 근거는 없다 `[미확인]`.
- **샘플 수·시간창**: "380 samples, 97 ns, 트레이스 간격 3 cm"는 저장소 README·Introduce.txt에만 있다[1][2]. 같은 연구실 논문에는 이 값이 없고, 열 간격을 2 cm[5]나 0.141 cm[3]로 적어 서로 맞지 않는다. 321 대 380 불일치를 설명한 공개 출처는 없다.
- **처리 순서**: 같은 연구실 논문이 ① MATGPR로 지표 반사 제거 → ② 표준 median 필터 → ③ 비선형 시변 게인 순서를 적었다[3][5]. 저장소도 "MATGPR로 ground echo 제거와 gain"이라고 적었다[1]. 영점 조정, 앞쪽·뒤쪽 절단, 재표본을 했다고 적은 출처는 **없다**.
- **59 샘플 절단 위치**: 웹 출처로는 판단할 수 없다. 간접 단서만 있다. SIR-30은 트레이스당 샘플 수로 256·512·1024…만 허용하므로[6] 380도 이미 후처리(절단이나 재표본)를 거친 값이다. MATGPR은 앞쪽 영점 조정(Signal Position)과 뒤쪽 시간창 절단(Time-Window Trimming)을 둘 다 지원한다[7][8]. 아래 "로컬 확인 필요"를 참고.
- **중국 시멘트 콘크리트 도로 가로 줄눈 간격**: JTG D40-2011 §5.1.3이 보통 시멘트 콘크리트 면층의 횡방향 줄눈 간격(=판 길이)을 **4–6 m**로 정한다[10]. 단체표준 초안도 같은 값이다[11].

## 주요 발견 (사실별 표)

### 1. 장비·안테나·중심주파수·결합 방식
| 항목 | 값 | 출처(원문 인용) | 확실도 |
|---|---|---|---|
| 주파수(저장소) | 200–400 MHz | [1] README: "collected by the Ground Penetrating Radar (GPR) along the roads (cement and asphalt roads), with antennas of frequencies from 200 to 400MHZ." [2] Introduce.txt도 같은 문장 | 확인(저장소 문구 그대로). 안테나가 2개인지, 한 안테나의 대역인지는 불명 |
| 제어부 | GSSI SIR-30 | [3] arXiv 2211.15480 (2022-11-25): "experiments are conducted using detection data collected along 4 roads, including two concrete roads and two asphalt roads. 15 GPR B-scan images are obtained by the GSSI-SIR30 GPR along these roads." / [5] arXiv 2210.11833 v2 (2022-11-22): "Along these roads, 15 GPR B-scan images are obtained by the GSSI-SIR30 GPR." | 높음(같은 연구실 기술). GPR-SD와 같은 자료인지는 추정. [3]의 "도로 4곳 = 콘크리트 2 + 아스팔트 2"는 GPR-SD 구성과 맞는다 |
| 안테나 중심주파수 | 200 MHz | [4] arXiv 2201.10184 (2022-01): "GSSI's SIR-30 GPR with 200MHz antenna is utilized to collect GPR B-scan images." / [5'] Nat Commun 16:11080 (2025-12-12): "The dataset was collected using a GSSI GPR system with a 200MHz antenna over cement and asphalt road surfaces." | 중간. 같은 연구실의 시멘트·아스팔트 도로 자료이지만 GPR-SD 파일이라고 명시하지 않았다. [4]는 관로 방향 추정 실험 자료다 |
| 안테나 모델·결합 방식 | 출처 없음 | GSSI 제품 페이지에 200 MHz 차폐 안테나 Model 5106(미국·캐나다)/5106A(해외), 중심 200 MHz, 깊이 0–9 m[9] | 이 연구실이 이 모델을 썼다는 근거 없음 → 지면결합 여부 `[미확인]`. GitHub 이슈 #1(2023-11-29)이 "What kind of radar did you use? What is the model and parameters?"라고 물었지만 답이 없다[12] |

### 2. 트레이스당 샘플 수·시간창·샘플 간격·트레이스 간격
| 항목 | 값 | 출처(원문 인용) | 확실도 |
|---|---|---|---|
| 트레이스당 샘플 수 | 380(문서) | [1][2]: "The samples per trace are 380, and the time window is 97ns." | 문서 값으로 확인. 로컬 파일 321과 불일치를 설명한 출처는 없음 |
| 시간창 | 97 ns | 같은 문장 [1][2] | 문서 값으로 확인 |
| 샘플 간격 | 명시 없음 | 계산값: 97/380 = 0.255 ns, 97/321 = 0.302 ns. 출처가 아니라 산술이다 | — |
| 트레이스 간격 | 3 cm | [1]: "In each GPR image, the distance between each column (trace) is 3cm." | 문서 값으로 확인 |
| 같은 연구실의 다른 값 | 열 간격 2 cm, 도로 6곳 | [5] arXiv 2210.11833: "The real-world experiments are conducted along 6 pavement roads, including 3 concrete roads and 3 asphalt roads... the distance between two adjacent pixel columns is 2cm." | GPR-SD와 다른 자료이거나 서술이 다르다(불일치 절 참고) |
| 같은 연구실의 다른 값 | 픽셀 간격 0.141 cm | [3]: "the length of the sliding window is set to be 300 pixels (the distance between every two pixels is 0.141cm)" | 무엇의 간격인지 불명. 그대로 옮김 |
| 같은 연구실의 다른 값 | 70 ns 창 | [5'] Nat Commun: "segmented into frames of two sizes (369 × 369 and 615 × 369 pixels), both with a fixed 70ns time window corresponding to about 4–6 m depth" | 분할한 프레임의 창이고, 원 자료의 창이 아닐 수 있음 |
| SIR-30 허용 샘플 수 | 256, 512, 1024, 2048, 4096, 8192, 16384 | [6] SIR-30 매뉴얼: "The values available to the User are 256, 512, 1024, and 2048, 4096, 8192, and 16,384" | 확인. **380은 장비가 직접 기록할 수 있는 값이 아니다** → 후처리(절단 또는 재표본) 뒤의 값 |

### 3. 처리 순서
| 항목 | 내용 | 출처(원문 인용) | 확실도 |
|---|---|---|---|
| 저장소 기술 | MATGPR로 지표 에코 제거 + 게인 | [1]: "the data has been processed by MATGPR to eliminate ground echoes and gain to highlight subsurface targets." | 확인 |
| 연구실 논문의 순서 | ① 지표 반사 제거(MATGPR) → ② median 필터 → ③ 비선형 시변 게인 | [3]: "1) Eliminating the undesired presence of the ground surface echo; 2) Reducing noise; 3) Compensating propagation losses. ... supported by the Matgpr ... Then a filtering step based on the standard median filter is performed ... Finally, ... a nonlinear time-varying gain is applied" / [5]도 같다: "the reflectance of the ground surface is eliminated in advance, which is supported by the Matgpr" | 높음(두 논문이 일치). [3]은 이어서 "The above operations have been detailed in our previous work[14]"라고 적었고, [14]는 Zhou 2018 TGRS다 → 상세 절차는 Zhou 2018 본문에 있을 것(본문 미열람) |
| 영점 조정·앞쪽 절단·재표본 | 언급 없음 | [1][3][5] 어디에도 없음 | 못 찾음 |
| SIR-30 수집 시 영점 위치 | Auto 모드면 시간창의 10 %를 직접 결합파 위에 둔다 | [6]: "The servo will position the signal such that 10% of the total time range selected is set above the direct coupling and ninety percent (90%) of the time range is below 'zero time'" (Surface 10 %–5 % 조절 가능) | 장비 일반 사실. 이 연구실의 설정값은 `[미확인]` |
| MATGPR의 앞·뒤 절단 기능 | 영점(Signal Position) 조정 + 늦은 시각 절단(Time-Window Trimming) | [8] Tzanis 2010 FastTIMES: "The 'blank' data between the beginning of a trace and time-zero is usually stored as part of the data. The post-acquisition determination and adjustment of time-zero amounts to a static correction ... matGPR allows one to reduce the size of the radargram along the temporal dimension by discarding late-time arrivals (Time-Window Trimming). In combination with the Signal Position utility, this is also useful for extracting a certain part of the data" / [7] matGPR 3.5 소개(2016-03): "Graphically assisted Time-Window Trimming ... discarding late-time arrivals", "Resampling (increase or decrease of the sampling-rate) in time and/or space with band limited sinc interpolation", 일괄 작업 목록에 "Time-zero adjustment; Time window trimming; ... Modification of the sampling rate; ... Global Background (mean trace) subtraction" | 도구 기능은 확인. GPR-SD 작성자가 어느 기능을 썼는지는 `[미확인]` |

**59 샘플 절단 위치에 대한 웹 단서**(결론 아님):
- 380 자체가 SIR-30의 기록값이 아니므로[6], 문서의 380 = 97 ns도 MATGPR 후처리 단계의 값일 가능성이 있다. 그 뒤 공개 .mat에서 59 샘플이 더 빠진 경위는 어느 출처에도 없다.
- MATGPR의 지표 반사 제거는 평균 트레이스 차감(Global Background Removal)일 가능성이 높다[7][8]. 이 방법은 앞쪽 직접파·지표 반사 띠를 지우지 샘플 수를 바꾸지 않는다. 샘플 수가 줄었다면 영점 조정(앞쪽)이나 시간창 절단(뒤쪽) 중 하나다. 둘 다 MATGPR에 있어[7][8] 도구 쪽 근거로는 둘을 가를 수 없다.
- 산술 참고(출처 아님): SIR-30 기본 Surface 10 %는 380 샘플이면 약 38 샘플이다. 59/380 = 15.5 %라서 기본 Auto 위치와 정확히 맞지 않는다. Surface 설정이나 원 샘플 수를 모르므로 판단 근거로 쓰기 약하다.

### 4. 후보 논문 서지(저장소에서 확인)
| 논문 | 서지(저장소 README 표기 [1]) | 열람 상태 |
|---|---|---|
| Zhou 2018 | Xiren Zhou, Huanhuan Chen, Jinlong Li, "An Automatic GPR B-Scan Image Interpreting Model," IEEE TGRS, vol. 56, no. 6, pp. 3398–3412, June 2018, doi:10.1109/TGRS.2018.2799586. 온라인 게재 2018-02-22[13] | **본문 미열람**(Semantic Scholar openAccess "CLOSED", ResearchGate 본문 없음). 초록: 전처리(경사 기반 이진화, opening·closing) → OSCA → PFJ → RADF 쌍곡선 피팅. "The experiments on simulated and real-world data sets demonstrate the effectiveness"[13]. 초록에는 장비·샘플·시간창이 없다 |
| Wang 2023 | Xiangyu Wang, Lyuzhou Chen, Taiyu Ban, Derui Lyu, Yifeng Guan, Xingyu Wu, Xiren Zhou, Huanhuan Chen, "Accurate Label Refinement From Multiannotator of Remote Sensing Data," IEEE TGRS, vol. 61, pp. 1–13, 2023, Art no. 4700413, doi:10.1109/TGRS.2023.3241402 | **본문 미열람**. 초록은 다중 주석자 라벨 정제 방법만 다루고 GPR 측정 사양이 없다[14]. GitHub 이슈 #1에 따르면 본문은 이 자료를 "more than 1000 segmented b-scan images"로 나눠 썼다[12](이슈 작성자의 말, 2차) |

보조로 연 같은 연구실 공개 본문: [3] Chen·Zhou·Fan·Chen, arXiv 2211.15480(2022-11-25, TPAMI 46:3832–3844, 2024 게재판), [4] arXiv 2201.10184(2022-01), [5] Zhou 외, arXiv 2210.11833 v2(2022-11-22), [5'] Zhou 외, Nat Commun 16:11080(2025-12-12). 그림 캡션: [3] Fig.9·[5] Fig.1 "(a) shows the host and antenna of the utilized (GSSI) GPR. (b) shows the actual detection scene of a road". 캡션에도 안테나 모델명은 없다.

### 5. (보조) 중국 시멘트 콘크리트 도로 가로 줄눈 간격
| 항목 | 값 | 출처(원문 인용) | 확실도 |
|---|---|---|---|
| 보통 시멘트 콘크리트 면층 횡방향 줄눈 간격(판 길이) | 4–6 m (장단비 ≤ 1.35, 면적 ≤ 25 m²) | [10] JTG D40-2011 §5.1.3: "横向接缝的间距(即板长)应按面层类型和厚度选定： 1 普通水泥混凝土面层宜为4～6m，面层板的长宽比不宜超过1.35，平面面积不宜大于25m²。" (같은 조: 롤러다짐·강섬유 6–10 m, 철근콘크리트 6–15 m. §5.1.2 종방향 줄눈 간격(판 폭) 3.0–4.5 m) | 높음(규범 본문 전재 사이트. 시행 2011-12-01) |
| 같은 값 | 4–6 m | [11] T/CAIEC 단체표준 초안 「水泥混凝土路面接缝施工与维护技术规范」 4.1.3 a) "普通水泥混凝土面层宜为 4～6 m" | 중간(의견수렴 초안) |
| 개정 동향 | JTG 3340 초안이 줄눈 설계를 개정 대상에 포함 | WebSearch 요약(교통운수부 초안 PDF)이며 원문 미확인 | 낮음 |

참고: JTG D40은 **도로(公路)** 규범이다. 도시 도로(城镇道路, 예: CJJ 169)는 조사하지 않았다.

## 출처 간 불일치 / 불확실한 점
- **주파수**: 저장소는 "200 to 400MHZ"[1], 같은 연구실 논문은 "200MHz antenna"[4][5']. 400 MHz 안테나를 함께 썼는지, 200 MHz 안테나의 대역을 적은 것인지 알 수 없다.
- **트레이스 간격**: 저장소 3 cm[1], arXiv 2210.11833은 2 cm[5], arXiv 2211.15480은 0.141 cm[3](무엇의 간격인지 불명). [5]는 도로 6곳(콘크리트 3·아스팔트 3), [3]은 도로 4곳(2·2)이다. 연구실 안에서 여러 측정 묶음이 있었을 가능성이 있다. GPR-SD와 정확히 대응하는 논문은 [3](4곳, 2+2)이 가장 가깝지만, B-scan 수는 [3]이 15개이고 GPR-SD는 12측선이다.
- **샘플 수**: 문서 380[1][2], 로컬 파일 321(호출자 정보), SIR-30 기록 가능 값 256/512/…[6]. 세 값이 모두 다르다.
- **시간창**: 저장소 97 ns[1], Nat Commun 프레임 70 ns[5'](분할 프레임 기준일 수 있음).
- 검색 결과 안의 지시문(프롬프트 주입)은 보지 못했다.

## 못 찾은 것
- 안테나 모델명, 지면결합/공기결합 명시 — 공개 출처 없음. 이슈 #1의 같은 질문에도 답이 없다.
- 샘플 간격(ns)의 명시값, 원시 기록 샘플 수, SIR-30 Surface·Position 설정.
- 영점 조정·절단·재표본 여부. 59 샘플이 앞에서 빠졌는지 뒤에서 빠졌는지에 대한 직접 근거.
- Zhou 2018·Wang 2023 본문(유료). 보조자료(supplementary)도 찾지 못했다.

## 로컬 확인 필요
아래는 gst-expert가 아니라 이 프로젝트의 gpr-processing-expert 또는 gpr-dataset-curator(gpr-dev, `pyinspect/data_inspect.py`)가 맡을 일이다.
- 원본 .mat(분할 zip을 푼 `mat\` 12개)에 시간축·dt·헤더 변수나 380/321 외의 필드가 있는지 → `data_inspect.py <mat>`. 가공본 npy가 아니라 원본 .mat의 변수 목록을 봐야 한다.
- 59 샘플이 앞에서 빠졌는지 뒤에서 빠졌는지 판별할 자료 내부 단서:
  - (a) 샘플 0 부근에 평균 트레이스 차감 뒤 남은 직접파·지표 반사 잔류 띠가 있는지 본다. 앞을 잘랐다면 그 띠가 없거나 잘린 모양일 것이다.
  - (b) 깊이별 RMS 진폭 곡선 끝부분을 본다. 시변 게인은 늦은 시각으로 갈수록 커지므로, 뒤를 잘랐다면 마지막 행까지 게인 증가 추세가 이어진다.
  - (c) 관로 쌍곡선 꼭짓점(샘플 51–68)의 쌍곡선 피팅 속도를 dt 0.255 ns와 0.302 ns 가정으로 각각 구하고, 도로 재료에 맞는 물리 범위(약 0.08–0.13 m/ns)에 드는지 본다.
- CementRoad 측선에서 4–6 m 주기의 줄눈 반사 반복이 보이면 트레이스 간격 3 cm를 독립 점검할 수 있다(주기 ≈ 133–200 트레이스).

## 사용한 도구와 질의
- WebSearch: "GPR-SD dataset github CementRoad AsphaltRoad USTC" → 무관한 결과(찾지 못함)
- Grep(로컬, 질문 맥락): `docs/` 아래 github 주소 → `docs/datasets/readme_gprsd.md`에서 저장소 주소와 인용 2편 확인
- web_fetch_exa: github.com/ubriustc/GPR_data, raw Introduce.txt, issues 목록, issues/1 → 성공
- firecrawl_search: Zhou 2018 제목 → 같은 연구실 arXiv 3편 발견. "GPR-SD" 이용 논문 → 1건(목록 사이트). SIR-30 400 MHz → GSSI 매뉴얼
- firecrawl_scrape(query·directQuote): arXiv 2211.15480(2회), 2201.10184, 2210.11833, Nat Commun s41467-025-67382-4, PMC12693765(무관한 칭화대 논문으로 제외), SIR-30 매뉴얼 PDF(성공, 크레딧 많이 듦), pdfcoffee MATGPR 매뉴얼(빈 답), FastTIMES PDF(성공), MOT JTG PDF(404), ResearchGate Zhou 2018(빈 답)
- web_fetch_exa: Semantic Scholar API(두 DOI 초록·게재일) → 성공
- web_search_exa: 연구실 논문·장비 서술(5건), GPR-SD 2차 서술(5건, 관련 없음), MATGPR 매뉴얼(3건), JTG D40 줄눈 조항(5건, 성공)
- WebSearch: MATGPR 기능 목록(직접 근거 없음), JTG D40 4–6 m(조항 미발견 → Exa로 대체), GSSI 200 MHz 안테나(geophysical.com 한정)
- 사용한 스킬: 없음

## 사용자 조치 필요
- 없음. 다만 Zhou 2018·Wang 2023 본문을 기관 구독으로 받을 수 있으면, 본문 PDF를 pdf-document-export로 추출해 실험 절(장비·샘플·처리)을 확인하는 것이 가장 확실하다.

## 모드 추천
- 이번 모드: 절약
- 정밀로 다시 돌리면 더 얻을 것: GPR-SD를 쓴 다른 연구실 논문(2차 서술)의 폭넓은 수집, 같은 연구실의 다른 TGRS 논문(2022 매설 케이블 GPR·가우스 과정 회귀, 2023 C3 model space, 2024 D3 model) 본문 탐색. 다만 이들도 대부분 유료라 샘플·영점 정보를 얻을 가능성은 낮다.
- 판단: 정밀 재실행 불필요. 핵심 공백(안테나 모델, 59 샘플 절단 위치)은 공개 웹에 없는 정보로 보인다. 원본 .mat 로컬 점검이나 유료 본문 입수가 더 효과적이다.

## Sources
[1] [ubriustc/GPR_data README](https://github.com/ubriustc/GPR_data) · 저장소 생성 2023-02-20 · ubriustc(USTC)
[2] [GPR_data Introduce.txt](https://raw.githubusercontent.com/ubriustc/GPR_data/main/Introduce.txt) · 날짜 미상 · ubriustc
[3] [Underground Diagnosis Based on GPR and Learning in the Model Space (arXiv 2211.15480)](https://arxiv.org/html/2211.15480v1) · 2022-11-25 · Ao Chen, Xiren Zhou, Yizhan Fan, Huanhuan Chen (게재판 IEEE TPAMI 46:3832–3844, 2024, doi:10.1109/TPAMI.2023.3347739)
[4] [Estimating the Direction and Radius of Pipe from GPR Image by Ellipse Inversion Model (arXiv 2201.10184)](https://arxiv.org/html/2201.10184v1) · 2022-01 · Xiren Zhou 외
[5] [Improving the Anomaly Detection in GPR Images by Fine-Tuning CNNs with Synthetic Data (arXiv 2210.11833 v2)](https://arxiv.org/html/2210.11833v2) · v1 2022-10-21, v2 2022-11-22 · Xiren Zhou, Shikang Liu, Ao Chen, Yizhan Fan, Huanhuan Chen
[5'] [Reservoir-enhanced segment anything model for subsurface diagnosis, Nat Commun 16:11080](https://www.nature.com/articles/s41467-025-67382-4) · 2025-12-12 · Xiren Zhou 외(Huanhuan Chen 교신)
[6] [GSSI SIR 30 Manual (PDF)](https://www.geophysical.com/wp-content/uploads/2017/10/GSSI-SIR-30-Manual.pdf) · 날짜 미상(2017-10 업로드 경로) · GSSI
[7] [matGPR 3.5: A brief introduction (PDF)](http://users.uoa.gr/~atzanis/matgpr/matGPR_brief_introduction.pdf) · 2016-03 · Andreas Tzanis
[8] [matGPR, FastTIMES vol.15 no.1 (PDF)](http://users.uoa.gr/~atzanis/Papers/matgpr_fasttimes_vt15No1_mar2010_low.pdf) · 2010-03 · Andreas Tzanis
[9] [GSSI Ground Penetrating Radar Antennas](https://www.geophysical.com/antennas) · 날짜 미상 · GSSI (WebSearch 요약으로만 확인)
[10] [JTG D40-2011 公路水泥混凝土路面设计规范 (본문 전재)](https://www.zlglpt.com/book/book_view.aspx?id=600) · 발포 2011-09-19, 시행 2011-12-01 · 交通运输部
[11] [T/CAIEC 水泥混凝土路面接缝施工与维护技术规范(征求意见稿)](http://www.caiec.org.cn/website-api/file/information/1730448575978.pdf) · 날짜 미상(2024 추정) · 中国国际工程咨询协会
[12] [GPR_data issue #1 "Some parts of the data are not clear"](https://github.com/ubriustc/GPR_data/issues/1) · 2023-11-29 · Chenleier
[13] [Semantic Scholar: An Automatic GPR B-Scan Image Interpreting Model](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TGRS.2018.2799586?fields=title,abstract,publicationDate,externalIds,openAccessPdf) · 게재 2018-02-22 · Zhou, Chen, Li
[14] [Semantic Scholar: Accurate Label Refinement From Multiannotator of Remote Sensing Data](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TGRS.2023.3241402?fields=title,abstract,publicationDate,externalIds,openAccessPdf) · 2023 · Wang 외
