# GPR 관로 검출·깊이 추정 선행 조사 (prior art)

작성: 2026-10-29 · web-research-expert(정밀) · 출처 번호는 문서 끝 참고

## 0. 설계 권고 (요약)
| 항목 | 권고 | 근거 |
|---|---|---|
| 전처리 기본값 | DC/dewow → time-zero(first break 정렬) → 밴드패스 → 배경 제거(평균/median trace 차감) → 이득(exp/AGC, **검출 입력에만**) → letterbox 리사이즈(비율 유지), ns/px·m/trace 메타 보존 | [2][4][20] |
| 피팅 1순위 | 박스 내 trace별 최대/최소 위상점(또는 임계값+C3 중심 문자열) → RANSAC(x²–t² 선형) → 비선형 최소제곱(robust) 정밀화. 미지수 (x0, t0, v), R은 0 또는 기지 관경으로 고정 | [4][10] |
| 깊이 정의 | d = v·t0/2 = 관 **상단**까지. 얕은 표적은 Tx–Rx 간격 반영 | [1][2] |
| 이전 모델 1순위 | RF-DETR-M/S (Apache-2.0). 2순위 D-FINE/DEIM(COCO 전용 가중치), 대조군 torchvision Faster R-CNN v2 | [23][24][25][28] |
| 주의 | 소규모 실측에서 RT-DETR-X < YOLO11x(mAP50 0.735 vs 0.817) → YOLO11m 기준선과 k-fold 비교 후 확정 | [2] |
| **갱신(2026-10-29)** | 위 "first break 정렬"은 검출 입력 정렬용이다. **깊이 계산의 영점**은 gprMax 검증 결과를 따른다: 공기결합(4TU)은 지표 반사 피크, 영점과 꼭짓점은 같은 위상 기준(first break 영점 + 피크 꼭짓점이면 εr 9에서 10 cm 이상 어긋남). 공기결합에서 곡률 v 피팅 금지 | `depth_physics.md` §2·§7 |

## 1. 딥러닝 쌍곡선/관로 검출 사례 (2022+ 우선)
| 연구 | 검출기 | 데이터 | 성능 | 전처리 |
|---|---|---|---|---|
| Štifanić 2025 [2] | YOLO11x vs RT-DETR-X | GSSI UtilityScan 350 MHz 실측, 주석 1596개(완전정보 732), TWTT 9–43 ns, 깊이 0.4–2.0 m | YOLO11x mAP50 0.817±0.019 / 50-95 0.510; RT-DETR-X 0.735 / 0.440; 깊이 회귀 R² 0.963, MAE 0.041 | time-zero=first break, 밴드패스, 지수 이득 |
| Zhu & Ye 2023 [1] | YOLOv7 (vs v5) | 공개 이미지 209→증강 3012 | mAP 90.58% (v5 81.53%) | 대비 조정, 가로·세로 늘리기, 증강 |
| Wunderlich 2022 [4] | RetinaNet | 고고학 다채널 GPR | AP 0.58 | DC 제거, time-zero, 100–600 MHz BP, 80 ns 절단, 사용자 이득, wavenumber highpass; 256×256 (약 0.31 ns/px) |
| Kang 2025 [3] | YOLOv5+TDA, Sim2Real | 합성 300 + 실측 167 | mAP50 0.643 | median 배경 제거, AGC, 100–1500 MHz BP, 416² (agent 요약, 원문 미확인) |
| Wang 2023 [5] | ConvNeXt-YOLOv5 | 실측+FDTD+Mixup | YOLOv4/v5/v7/Faster R-CNN 대비 +1.9~3.8%p | 미상 |
| Bugarinović 2024 [6] | YOLO | gprMax 합성 2000장 학습 → 실측 | 정성 | 미상 |
| Lei 2019 [8] | Faster R-CNN + DCSE + CTFP + LS | 합성+현장 | — | 그레이 B-scan |
| Jaufer 2021 [7] | Faster R-CNN | gprMax + 현장 | — | — |
| GPR-Former 2024 [12] | Transformer, 파라미터 직접 회귀 | 해저·육상 관로 3종 | TP_Pre/Rec, C3 대비 +6~22% | — |
| AIHub 71327 기준 [15] | EfficientDet | 서울 도로 B/C-scan 213k | mAP@0.5 0.8457 (B 횡배관 0.995, 종배관 0.614) | gain 3종(0.25/0.5/1.0) |
| 국내 [32] | YOLOv3 + C3 | 서울 도로 공동 | AP50 정밀도 84%, 재현율 92%, 수평위치 오차 0.12–0.36 m | — |

- 서울시 멀티채널 GPR 장비 비교: zero-setting과 background removal은 모든 장비에 적용, 밴드패스는 선택 [20].
- LGRS 2022(Faster R-CNN+ACFPN): AP 0.9256, 위치 오차 11 cm 미만 (agent 요약, 원문 미확인) [34].

## 2. 박스 내 쌍곡선 피팅·깊이 추정
- 모델: t(x) = (2/v)·sqrt(d² + (x−x0)²); x²–t² 선형화 (t/2)² = (t0/2)² + (x−x0)²/v² → 기울기에서 v, 절편에서 t0 [4].
- Wunderlich 10방법 비교 [4]: 속도 RMS 0.02–0.045 m/ns. 최저는 Canny+RANSAC과 임계값+C3+(RANSAC 또는 x²–t²)이며 평균 오차 0.021 m/ns. t0 RMS 1–1.6 ns, 박스당 0.01–5 s. 오차 원인은 박스 정확도, 위상 선택, 소반경 가정, time-zero.
- 반경: Shihab 2004는 합성·통제 현장에서 1% 이내 [11]. Giannakis 2022는 잡음이 있으면 반경 오차가 최대 500%로, v·깊이만 추정하라고 권함 [10] → **R 동시추정 금지**.
- 깊이 정의: 관 상단 기준(Model 2)이면 깊이·반경 효과가 분리됨. 오차 0.6%/4.4%(균질), 4.8%/15%(비균질), 실측 TU1208 10% 미만 [1].
- time-zero: first break(첫 양의 피크) 정렬 [2]. v·t0 없이 깊이 추정해 오차 0.5% 미만, 표준편차 3%(350 MHz, 26개 표적) [13].
- Hough/RHT [14][4], C3 [9][4][32], CWT(경계 검출 용도뿐, 피팅 정확도 1차 근거 없음).
- 오픈소스: C3 MATLAB(lweileeds, **LICENSE 없음**) [9]; GPRPy (MIT, 쌍곡선 피팅 기능 미확인) [30]; RGPR (hyperbola fitting 기능, 라이선스 미확인) [31].

## 3. AIHub 3D 지표투과레이더 탐사 데이터
- **데이터셋 번호는 dataSetSn=71327** (261은 "한국어 이미지 설명" 데이터셋) [15][35]. 로컬 폴더명 "261."은 AIHub 데이터셋 번호가 아니다.
- 서울 도로 2015–2022, 셀파이엔씨·아이리스테크놀로지 구축, jpg + XML, 총 213,474건 [15].
- 클래스: 공동(대칭·비대칭·기타), 맨홀, 종배관, 횡배관. XML `name` = cavity, manh, lpipe, tpipe(페이지 표기 "tpipt") [15].
- 주요 XML 필드: permittivity(0~15), equipment, channels, channellayer, gainvalue, cavityclass(A/B/C) [15].
- B-scan 수: 종배관 4,566, 횡배관 1,692, 맨홀 34,227 [15]. 픽셀값은 0–255 정수 [16].
- **ns/픽셀·m/픽셀, 장비 코드(DS/MA/NM/RA) 정의: 웹에서 찾지 못함** (활용가이드는 로그인 다운로드).
- lpipe 정의는 출처 간 충돌: CODIL 발췌 "횡배관(lpipe)" [17] vs [16] Longitudinal/Transverse 표기 → 로컬 XML 확인 필요.
- 같은 계열 JIS 공개 데이터는 Kontur 3D-Radar 장비라고 밝힘 [19]. 71327과 같은 장비인지는 미확인.

## 4. YOLO11m 대체 후보 (Apache/BSD/MIT)
| 모델 | 코드 | 가중치 | 성능 | 비고 |
|---|---|---|---|---|
| RF-DETR N/S/M/L | Apache-2.0 | Apache-2.0 (A/F/P/XL/2XL은 PML 1.0) | COCO AP50 67.6–75.1, RF100-VL AP50 85.0–88.2 (YOLO11-M 82.5) | DINOv2 백본(Apache), 미세조정 지향 [23] |
| D-FINE | Apache-2.0 | COCO 가중치 사용 권장 | COCO AP 42.8–55.8 | Objects365 체크포인트는 데이터셋 약관 주의 [24] |
| DEIM (D-FINE/RT-DETRv2) | Apache-2.0 | 동일 | COCO AP 43.0–56.5 | 빠른 수렴 [25] |
| DEIMv2 | 자체 라이선스 | DINOv3 라이선스 | AP 50.9–57.8 | 상업 이용은 문의 → 제외 [26][29] |
| RT-DETR / v2 | Apache-2.0 | — | AP 46.5–54.8 | Ultralytics 구현은 패키지 라이선스(AGPL 추정, 미확인) 주의 [27] |
| torchvision Faster R-CNN v2 | BSD-3 | 명시 없음 | COCO 46.7 | 대조군 [28] |

## 5. 4TU "GPR utilities NL" 데이터와 논문
- 데이터: DOI 10.4121/96303227-5886-41c9-8607-70fdd2cfe7c1, CC0, 2023-09-05 [22].
- 논문: ter Huurne et al. 2024, Data in Brief 54:110329 [21].
- 장비: 500 MHz air-coupled, GNSS RTK, 0.02 m 간격, 512 samples/50 ns, SEG-Y 원시. 13개 현장, 125건, radargram 759개.
- 깊이·속도: Reflex-W 9.1.3 쌍곡선 피팅으로 v → εr=(c/v)². **깊이 오차는 보고 안 함.** 정답은 시험굴착(줄자·GNSS)이며 지리 좌표 없음.
- 후속 연구(ter Huurne 박사논문 등)는 GPR 사용 판단 의사결정 ML이며 깊이 오차 연구가 아니다.

## 미해결 (로컬·사용자 확인 필요)
- AIHub 71327 「구축활용가이드」「데이터 설명서」(로그인 다운로드): 장비 코드, ns/px, 라벨 정의의 1차 출처. https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71327
- AIHub XML의 equipment 분포, lpipe/tpipe 샘플이 쌍곡선인지 수평선인지 로컬 확인.

## Sources
[1] Zhu & Ye, A Modular Method for GPR Hyperbolic Feature Detection and Quantitative Parameter Inversion of Underground Pipelines, Remote Sens. 15:2114 — https://www.mdpi.com/2072-4292/15/8/2114 · 2023-04-16
[2] Štifanić et al., An Automated System for Underground Pipeline Parameter Estimation from GPR Recordings, Remote Sens. 17:3493 — https://www.mdpi.com/2072-4292/17/20/3493 · 2025-10-21
[3] Kang et al., Shape-Aware Topological Representation for Pipeline Hyperbola Detection in GPR Data (IEEE Sensors J / arXiv 2506.06311) — https://arxiv.org/html/2506.06311v3 · 2025
[4] Wunderlich et al., Hyperbola Detection with RetinaNet and Comparison of Hyperbola Fitting Methods, Remote Sens. 14:3665 — https://www.mdpi.com/2072-4292/14/15/3665 · 2022-07-30
[5] Wang et al., Identification of common underground pipelines by GPR based on improved YOLOv5, Chinese J. Geophys. — http://www.geophy.cn/en/article/doi/10.6038/cjg2023R0431?viewType=HTML · 2023-07
[6] Bugarinović et al., Detection of Hyperbolic Reflections on Real GPR Data Using YOLO Trained with Synthetic Data, TSP 2024 — https://doi.org/10.1109/tsp63128.2024.10605917 · 2024-07
[7] Jaufer et al., Deep learning based automatic hyperbola detection on GPR data for buried utility pipes mapping, IWAGPR 2021 — https://doi.org/10.1109/iwagpr50767.2021.9843151 · 2021-12
[8] Lei et al., Automatic hyperbola detection and fitting in GPR B-scan image, Autom. Constr. 106:102839 — https://doi.org/10.1016/j.autcon.2019.102839 · 2019-06
[9] Dou et al., Real-Time Hyperbola Recognition and Fitting in GPR Data (TGRS 2017) — https://eprints.whiterose.ac.uk/id/eprint/102528/1/hyperbola_fitting_final.pdf ; https://github.com/lweileeds/hyperbola_recognition · 2017 / repo 2022-11
[10] Giannakis et al., On the Limitations of Hyperbola Fitting for Estimating the Radius of Cylindrical Targets, IEEE GRSL — https://doi.org/10.1109/lgrs.2022.3195947 · 2022
[11] Shihab, Al-Nuaimy, Eriksen, Radius estimation for subsurface cylindrical objects detected by GPR — https://ieeexplore.ieee.org/document/1343435 · 2004-06
[12] Jin et al., GPR-Former, IEEE TGRS 62 — https://doi.org/10.1109/tgrs.2024.3406154 · 2024
[13] Peer, Depth Estimation and Ray Tracing Model Selection of Buried Utilities on GPR Data, TSP 2018 — https://doi.org/10.1109/tsp.2018.8441270 · 2018-07
[14] Xiao et al., Research on hyperbola detection and fitting in GPR B-scan image, CISCE 2021 — https://doi.org/10.1109/cisce52179.2021.9445915 · 2021-05
[15] AI-Hub 3D 지표투과레이더 탐사 데이터 (dataSetSn=71327) — https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71327 · v1.1 2023-11-21
[16] 김수윤·변중무, 기계학습 기반 회절파 분리…, 물리탐사 26(4):171-184 — https://koreascience.kr/article/JAKO202335272004879.pdf · 2023-11-30
[17] CODIL 보고서 OTKCRK240143 (지하공간통합지도 갱신 자동화…), 검색 발췌만 확인 — https://www.codil.or.kr/filebank/original/RK/OTKCRK240143/OTKCRK240143.pdf · 날짜 미상
[18] 황재홍·이기석, 서울시 GPR 탐사 데이터의 학습데이터 구축…, 대한공간정보학회지 32(1) — https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART003063647 · 2024
[19] 지하안전정보시스템, 지하안전 관리를 위한 GPR 공동탐지 AI 학습 데이터 공개 — https://www.jis.go.kr/community/boa01005_popup.do?board_no=1271 · 2025-12-31
[20] 윤진성 외, 도로공동 탐지를 위한 멀티채널 GPR 장비 비교 연구 — https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002413352 · 날짜 미상
[21] ter Huurne et al., Ground penetrating radar at work…, Data in Brief 54:110329 — https://pmc.ncbi.nlm.nih.gov/articles/PMC10973596/ · 2024-03-16
[22] 4TU.ResearchData, GPR dataset with ground-truth data of utility surveying activities — https://data.4tu.nl/datasets/96303227-5886-41c9-8607-70fdd2cfe7c1 · 2023-09-05
[23] roboflow/rf-detr README — https://github.com/roboflow/rf-detr · 확인 2026-10-29
[24] Peterande/D-FINE README — https://github.com/Peterande/D-FINE · 확인 2026-10-29
[25] DEIM README / LICENSE — https://github.com/ShihuaHuang95/DEIM · 확인 2026-10-29
[26] Intellindust-AI-Lab/DEIMv2 — https://github.com/Intellindust-AI-Lab/DEIMv2 · 2025-09
[27] lyuwenyu/RT-DETR README — https://github.com/lyuwenyu/RT-DETR · 확인 2026-10-29
[28] torchvision fasterrcnn_resnet50_fpn_v2 docs — https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html ; LibreYOLO mirror note https://huggingface.co/LibreYOLO/LibreFasterRCNNl · 날짜 미상
[29] Meta DINOv3 License — https://ai.meta.com/resources/models-and-libraries/dinov3-license · 2025-08-14
[30] NSGeophysics/GPRPy (MIT) — https://github.com/NSGeophysics/GPRPy · 날짜 미상
[31] emanuelhuber/RGPR — https://github.com/emanuelhuber/RGPR · 날짜 미상
[32] 최병훈 외, 도심지 도로 지하공동 탐지를 위한 딥러닝 기반 GPR 자료 해석 기법, 물리탐사 25(4) — https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002902847 · 2022
[34] A Study of Automatic Recognition and Localization of Pipeline for GPR Based on Deep Learning, IEEE GRSL (agent_run 요약) — https://doi.org/10.1109/LGRS.2022.3198439 · 2022
[35] AI-Hub dataSetSn=261 (한국어 이미지 설명 데이터셋) — https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=261 · 2022-02
