# gpr_detection 패키지 조사 (2026-10-29)

조사: web-research-expert (정밀 모드). 기준 환경: Windows 11, conda env `gpr-dev`, Python 3.13, torch 2.14.1+cu130, RTX 4070 SUPER.
버전·날짜는 PyPI 릴리스 RSS 기준(2026-10-29 조회). 추천: ◎ 채택 / ○ 조건부 / △ 참고만 / ✕ 제외.

## 1. GPR 입출력·처리

| 패키지 | 용도 | 최신 (날짜) | Py3.13 / Windows | 라이선스 | 유지보수 | 출처 | 추천 |
|---|---|---|---|---|---|---|---|
| segyio | 4TU SEG-Y(int16) 읽기 | 1.9.14 (2025-12-02); 2.0.0a1 알파 | cp313 win_amd64 휠 있음. conda-forge는 1.9.12, py313 빌드 안 보임 | LGPL-3.0 | Equinor, 2025-12 릴리스 | pip | ◎ |
| obspy | SEG-Y rev.1/SU 읽기(obspy.io.segy) | 1.5.1 (2026-08-28) | conda-forge win-64 py313 있음 | LGPL-3.0 | 활발 | conda-forge | ✕ (segyio로 충분, 무거움) |
| h5py | gprMax `.out`(HDF5) 읽기 | 3.16.0 (2026-03-06) | 있음 | BSD-3 | 활발 | conda-forge | ◎ |
| readgssi | GSSI .dzt 읽기·dewow·time-zero·BGR·gain (이동 없음) | 0.0.22 (2023-09-02) | sdist만, numpy 2 미확인, obspy 의존 | GPL-3.0(GitHub) / AGPL(PyPI) 불일치 | 2023 이후 릴리스 없음 | pip | △ (형식 참고) |
| GPRPy | DZT·DT1·rd3 읽기, dewow·time-zero·평균 제거·AGC·tpow·F-K(Stolt) 이동 | 1.0.13 (2023-04-04) | PyPI 없음(`pip install .`), numpy 2 미확인 | MIT | 2023 이후 릴리스 없음(포크 일부 2025~26) | 소스 | △ (MIT라 알고리즘 참고·발췌 가능) |
| ridal (구 rsgpr) | rd3·dzt·dt1 읽기, zero_corr·dewow·auto_gain, NetCDF/xarray 출력 | 0.11.0 (2026-10-27) | cp311–cp314 win_amd64 휠 | MIT | 활발(0.5에서 API 전면 변경) | pip | ○ (.dzt가 실제로 들어오면) |
| ImpDAR | Kirchhoff·Stolt·위상이동 이동, AGC, SEG-Y·gprMax 입력 | 1.2.1 (2025-06-20) | Py3.7–3.10에서 시험, 3.13 미확인 | GPL-3.0 | 느림 | pip | △ (이동 구현 참고만, 코드 복사 금지) |
| RAGU | 행성·빙하 레이더 GUI(tpow) | 0.1.56 (2025-01-30) | 미확인 | GPL-3.0 | 느림 | pip | ✕ |
| siina | DZT IO | 19.8.28 (2019-08-27) | Py3.6 시대 | Apache-2.0 | 방치 | pip | ✕ |
| RPGPRpy ("pyGPR") | R RGPR 래퍼 | — (2022 WIP) | R·rpy2 필요 | NOASSERTION | 방치 | — | ✕ |
| RGPR (R) | IDS .dt 포함 다형식, 쌍곡선 피팅 | — (2025-03 push) | R | GPL-3.0+ | 보통 | — | △ (IDS .dt 헤더 해석 참고) |
| gprMax 도구 | `.out` 병합·B-scan 그림 | 4.0.1 (2026-10-03) | Py3.11–3.13 | GPL-3.0+ | 활발 | 별도 env | ✕ (gpr-dev에서는 h5py로 직접) |

**판단: 직접 구현.** 이 프로젝트 입력의 대부분(SEG-Y·npy+json·VOC XML in zip·YOLO txt·HDF5)은 segyio/h5py/numpy/zipfile/xml.etree로 충분하다.
IDS .dt를 읽는 Python 라이브러리는 찾지 못했다(직접 파서, RGPR 참고). 전처리(dewow = 이동평균 제거, time-zero, 평균 트레이스 제거,
AGC/SEC/tpow gain)는 수십 줄짜리 numpy/scipy.signal 코드다. 이동(Kirchhoff/F-K Stolt)은 GPRPy(MIT) 구현을 참고해 직접 쓴다.
ImpDAR·readgssi는 GPL 계열이라 코드를 옮기지 않는다.

> 메인 세션 주: 4TU README(`docs/datasets/readme_4tu.md`)는 RadarMap SEG-Y 변종(little-endian, 시간 단위 ps)을 segyio 같은 일반 라이브러리로 읽으면 값이 틀어질 수 있다며 `06_scripts/sgy_radarmap.py`의 `read_sgy()`를 쓰라고 한다. 4TU는 이미 npy로 변환돼 있으므로 segyio는 선택 사항이다.

## 2. 검출

| 후보 | 코드 라이선스 | 가중치 라이선스 | 최신 (날짜) | torch 2.14 / Py3.13 / Windows | 유지보수 | 설치 | 추천 |
|---|---|---|---|---|---|---|---|
| ultralytics (YOLO11m) | AGPL-3.0 / Enterprise | AGPL(학습 산출 모델 포함) / Enterprise | 8.4.192 (2026-10-28) | Py3.8–3.14, Windows 공식 | 매우 활발 | pip | ○ 초안 전용 (내부 사용도 Enterprise 대상이라고 명시) |
| RF-DETR (rfdetr) | Apache-2.0 | N/S/M/L 검출·Seg 전 크기 Apache-2.0; XL·2XL·Atto/Femto/Pico 검출은 PML 1.0 | 1.11.2 (2026-10-04) | Py≥3.10. Windows는 PyTorch 경로로 가능하나 CUDA/FP8 extra는 Linux로 문서화 | 활발 | pip | ◎ 이전 1순위 (N–L만) |
| RT-DETR / RT-DETRv2 (transformers) | Apache-2.0 | 체크포인트별(원 구현 Apache-2.0 추정, 카드 확인) | transformers 5.21.0 (2026-10-26) | 순수 PyTorch, Windows 가능 | 활발 | pip | ◎ 이전 2순위 |
| D-FINE (원 저장소·transformers `DFineForObjectDetection`) | Apache-2.0 | `_coco`만 Apache-2.0. `_obj365`·`_obj2coco`는 상업 불가(Objects365) | 태그 없음(2026-10 커밋) | 공식 Py3.11/Linux. transformers판은 Windows 가능 | 보통 | pip(transformers) | ○ (COCO 가중치만) |
| DEIM / DEIMv2 | Apache-2.0 | DINOv3 백본 변형은 DINOv3 License 주의 | 태그 없음(2026-03 커밋) | Windows·torch 2.9+ 공식 표기 없음 | 보통 | 소스 | △ |
| torchvision detection (Faster R-CNN/RetinaNet/FCOS) | BSD-3 | COCO 가중치(데이터 약관 별도) | 0.29.1 (torch 2.14.1 짝) | 공식 Windows 휠 | 활발 | pip(cu130 인덱스) | ○ 기준선 |
| MMDetection 3.x / MMCV 2.x | Apache-2.0 | 모델별 | 3.3.0 (2024-01) / 2.2.0 (2024-04) | Windows 휠은 torch 2.2 세대까지 | 정체 | — | ✕ |
| detectron2 | Apache-2.0 | 모델별 | v0.6 (2021-11) | Windows 비공식 | 릴리스 정체 | — | ✕ |
| YOLOX | Apache-2.0 | 미확인 | 0.3.0 (2022-04) | 소스 빌드 | 저활동 | — | ✕ |

학습 보조:

| 패키지 | 용도 | 최신 (날짜) | 라이선스 | 비고 | 추천 |
|---|---|---|---|---|---|
| albumentations | 증강 | 2.0.8 (마지막 MIT) | MIT | 2025-07-10 아카이브, 수정 없음. `opencv-python-headless` 의존 | △ |
| albumentationsx | 증강(후속) | 2.5.1 | AGPL-3.0 / 상용 | MIT/Apache 프로젝트는 상용 라이선스 필요 | ✕ (이전 목표와 충돌) |
| torchvision.transforms.v2 | 증강(bbox 지원) | torchvision 0.29.1 | BSD-3 | 추가 의존 없음 | ◎ (이전 단계 증강) |
| torchmetrics | mAP | 1.9.0 (2026-03-09) | Apache-2.0 | 백엔드로 pycocotools 또는 faster-coco-eval 필요 | ◎ |
| pycocotools | COCO 평가 | 2.0.11 (2025-12-15) | BSD-2 | torchmetrics 기본 백엔드 | ◎ |
| faster-coco-eval | COCO 평가(빠름) | 1.8.0 (2026-08-24) | Apache-2.0 | 결과가 공식 구현과 동일하다고 문서화 | ○ |
| onnx / onnxslim | 내보내기 | 1.23.2 (2026-10-06) / 0.1.98 (2026-10-08) | Apache-2.0 / MIT | — | ○ |
| onnxruntime-gpu | 추론 | 1.31.1 (2026-10-26) | MIT | 1.27부터 PyPI 기본 빌드가 CUDA 13.0 + cuDNN 9. torch를 먼저 import하거나 `preload_dlls()` | ○ (배포 단계) |
| tensorrt | 최적화 추론 | 11.4.0.106 (2026-10-08) | NVIDIA 독점 | cu13 Windows 휠 있음 | ✕ (지금은 불필요) |
| supervision | 시각화·지표 유틸 | 0.31.3 (2026-10-28) | MIT | 선택 | △ |

## 3. 쌍곡선 검출·피팅

| 수단 | 용도 | 최신 | 라이선스 | 추천 |
|---|---|---|---|---|
| scipy.optimize.least_squares | t(x)=(2/v)·√(d²+(x−x0)²) 비선형 피팅(soft_l1/huber loss, 경계) | scipy 1.18.1 (2026-08-21) | BSD-3 | ◎ |
| scikit-learn RANSACRegressor | 외곽점 제거(선형화한 t² = (4/v²)(d²+(x−x0)²) 형태에 적용) | 1.9.1 (2026-09-10) | BSD-3 | ○ (선형화 RANSAC → least_squares 정밀화) |
| scikit-image | 피크·에지·이진화·Hough(원/직선) | 0.26.0 (2025-12-20), 0.27.0rc1 | BSD-3 | ○ (전처리·후보점 추출) |
| C3 hyperbola_recognition (Dou 2017) | column-connection clustering + 직교거리 피팅 | 2022 | 라이선스 파일 없음, MATLAB | △ (논문 보고 재구현) |

판단: 바로 쓸 만한 permissive Python 구현은 찾지 못했다. 검출 박스 안 피크 추출 → C3 방식 열 연결 → RANSAC → least_squares 순서로 직접 구현한다.

## 4. 데이터·실험 관리

| 패키지 | 최신 (날짜) | 라이선스 | 비고 | 추천 |
|---|---|---|---|---|
| pandas | 3.1.0 (2026-10-22) | BSD-3 | 3.x는 CoW·문자열 dtype 변경 | ◎ |
| pyarrow | 26.0.0 (2026-10-09) | Apache-2.0 | parquet 결과표 | ◎ |
| pyyaml | 6.0.3 (2025-09-25) | MIT | 설정 파일 | ◎ |
| omegaconf | 2.4.0 (2.4.1rc2 2026-10-28) | BSD-3 | 선택 | ○ |
| hydra-core | 1.3.8 (1.4.0rc3 2026-10-25) | MIT | 1.3.8은 omegaconf<2.4 고정, 저장소 이전 | ✕ (지금은 과함) |
| tqdm | 4.70.1 (2026-09-11) | MPL-2.0/MIT | — | ◎ |
| rich | 15.0.0 (2026-04-12) | MIT | — | ○ |
| tensorboard | 2.21.0 (2026-06-29) | Apache-2.0 | 가벼움, 로컬 전용 | ◎ |
| mlflow | 3.17.2 (2026-10-21) | Apache-2.0 | 서버·DB 의존이 무거움 | ✕ |

## 5. 시각화·시험·품질

| 패키지 | 최신 (날짜) | 라이선스 | 추천 |
|---|---|---|---|
| matplotlib | 3.11.2 (2026-09-11) | PSF 계열 | ◎ |
| opencv-python (pip) | 4.14.0.94 (2026-07-28); 5.0.0.93 (2026-07-02) | Apache-2.0 | ◎ (ultralytics 요구, 하나만 설치) |
| opencv-python-headless (pip) | 4.14.0.94 | Apache-2.0 | ✕ (ultralytics와 공존 시 cv2 충돌) |
| opencv (conda-forge) | — | Apache-2.0 | ✕ (pip판과 중복) |
| pytest | 9.2.0 (2026-10-27) | MIT | ◎ |
| hypothesis | 6.170.13 (2026-10-28) | MPL-2.0 | ◎ (피팅 함수 속성 시험) |
| ruff | 0.17.3 (2026-10-27) | MIT | ◎ |
| mypy | 2.4.1 (2026-10-28) | MIT | ◎ |

## 6. 충돌 주의

1. **OpenCV는 하나만 설치한다.** 공식 README: "SELECT ONLY ONE … all the packages use the same namespace (cv2)".
   ultralytics는 `opencv-python>=4.7.0,!=4.13.0.90`에 의존하므로 env에는 pip `opencv-python` 하나만 둔다.
   headless와 conda `opencv`는 넣지 않는다. albumentations(headless 의존)를 넣으면 다시 둘 다 설치되니 피한다.
   여러 개가 이미 설치돼 있으면 전부 `pip uninstall` 한 뒤 하나만 다시 설치한다.
2. **onnxruntime과 onnxruntime-gpu.** ultralytics export extra가 CPU판 `onnxruntime>=1.20`을 끌어온다.
   GPU판을 쓰려면 CPU판을 제거하고 `onnxruntime-gpu`만 남긴다(export는 `pip install ultralytics` 기본 설치로 하고, export extra는 쓰지 않는다).
3. **numpy 2.x.** ultralytics는 Windows에 numpy 상한이 없다. segyio 1.9.14는 cp313 휠(2025-12 빌드)이라 numpy 2 ABI일 가능성이 높다(실측 필요).
   readgssi·GPRPy·ImpDAR은 numpy 2 근거가 없어 env에 넣지 않는다.
4. **conda와 pip를 섞는 순서.** conda로 설치할 수 있는 것을 먼저 다 설치하고, pip는 마지막에 쓴다. pip 이후 conda로 추가 설치가 필요하면 env를 새로 만든다.
   environment.yml에는 pip 섹션을 마지막에 둔다. 같은 패키지(numpy, opencv 등)를 conda와 pip 양쪽에 두지 않는다.
5. **torch cu130 휠.** `pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130`.
   PyPI 기본 인덱스로 다시 설치하면 다른 CUDA 빌드로 바뀔 수 있으니, ultralytics 등을 설치할 때 torch가 재설치되지 않는지 확인한다.
   onnxruntime-gpu(CUDA 13 기본)와 같은 CUDA 13·cuDNN 9 계열이다.
6. **라이선스 경계.** YOLO 의존은 `detect/` 아래에만 둔다(CLAUDE.md 규칙). AlbumentationsX·RF-DETR XL/2XL·D-FINE obj365 가중치는 이전 목표(Apache/BSD/MIT)와 맞지 않는다.

## 권장 environment.yml 초안

> 메인 세션 주(2026-10-29): 이 초안은 조사 시점 기록이다. **최종본은 저장소의 `environment.yml`**이고, 에이전트별 보강(pywavelets·numba·numpy<2.6·imagehash·cleanlab 등)은 `packages_agents.md`를 본다.

```yaml
name: gpr-dev
channels:
  - conda-forge
  - nodefaults
dependencies:
  - python=3.13
  - pip
  # 수치·IO
  - numpy>=2.2
  - scipy>=1.15
  - scikit-learn>=1.6
  - scikit-image>=0.25
  - h5py>=3.12
  - pandas>=2.2
  - pyarrow>=18
  # 설정·로그
  - pyyaml>=6.0.2
  - omegaconf>=2.3
  - tqdm>=4.66
  - rich>=13.9
  - tensorboard>=2.18
  # 시각화·시험·품질
  - matplotlib>=3.9
  - pytest>=8.3
  - hypothesis>=6.120
  - ruff>=0.8
  - mypy>=1.13
  - pip:
      # torch는 아래 명령으로 먼저 따로 설치한다 (cu130 인덱스):
      #   pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
      - segyio>=1.9.14            # conda-forge에 py313 빌드 없음
      - opencv-python>=4.10,<5    # cv2는 이것 하나만
      - ultralytics>=8.3.0        # YOLO11 (AGPL, 초안 전용)
      - torchmetrics>=1.6
      - pycocotools>=2.0.8
      - onnx>=1.17
      - onnxslim>=0.1.80
      - onnxruntime-gpu>=1.27     # CUDA 13 기본 빌드. CPU판 onnxruntime은 제거
      # 이전 단계(선택):
      # - rfdetr>=1.11            # N/S/M/L만 (Apache-2.0)
      # - transformers>=5.0       # RT-DETRv2 / D-FINE(COCO 가중치)
      # - faster-coco-eval>=1.6
      # - ridal>=0.11             # GSSI .dzt가 실제로 들어올 때
```

## Sources
[1] PyPI 릴리스 RSS `https://pypi.org/rss/project/<name>/releases.xml` (각 패키지, 2026-10-29 조회)
[2] readgssi GitHub https://github.com/iannesbitt/readgssi (날짜 미상)
[3] readgssi PyPI https://pypi.org/project/readgssi/ (0.0.22, 2023-09-02)
[4] GPRPy releases https://github.com/NSGeophysics/GPRPy/releases (1.0.13, 2023-04-04)
[5] ridal PyPI https://pypi.org/project/ridal/ (0.11.0, 2026-10-27)
[6] segyio PyPI files https://pypi.org/project/segyio/#files (1.9.14, 2025-12-02)
[7] conda-forge segyio API https://api.anaconda.org/package/conda-forge/segyio
[8] conda-forge obspy API https://api.anaconda.org/package/conda-forge/obspy
[9] obspy.io.segy 문서 https://docs.obspy.org/packages/obspy.io.segy.html (ObsPy 1.5.0)
[10] ImpDAR https://impdar.readthedocs.io/en/latest/bin/impproc.html (agent 경유)
[11] RAGU README https://github.com/btobers/RAGU (agent 경유)
[12] siina https://github.com/ahartikainen/siina (agent 경유)
[13] gprMax releases https://github.com/gprMax/gprMax/releases (4.0.1, 2026-10-03)
[14] C3 hyperbola_recognition https://github.com/lweileeds/hyperbola_recognition (2022)
[15] Dou et al., Real-Time Hyperbola Recognition and Fitting in GPR Data, IEEE TGRS, https://doi.org/10.1109/tgrs.2016.2592679 (2016-09-21)
[16] Ultralytics License https://www.ultralytics.com/license (날짜 미상)
[17] ultralytics pyproject.toml https://raw.githubusercontent.com/ultralytics/ultralytics/main/pyproject.toml (main, 2026-10-29 조회)
[18] Roboflow Model Licenses https://docs.roboflow.com/models/model-licenses (날짜 미상)
[19] RF-DETR Model License https://roboflow.com/model-licenses/rf-detr (날짜 미상)
[20] rf-detr issue #666 https://github.com/roboflow/rf-detr/issues/666 (2026-02-12)
[21] D-FINE issue #357 https://github.com/Peterande/D-FINE/issues/357 (2026-08-19 답변, 2026-09-08 종료)
[22] HF ustc-community/dfine-medium-coco https://huggingface.co/ustc-community/dfine-medium-coco
[23] HF PekingU/rtdetr_v2_r50vd https://huggingface.co/PekingU/rtdetr_v2_r50vd
[24] albumentations GitHub README https://github.com/albumentations-team/albumentations (2025-07-10 아카이브)
[25] albumentations PyPI https://pypi.org/project/albumentations/ (2.0.8)
[26] ONNX Runtime CUDA EP https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html (2026-10-29 조회)
[27] PyTorch previous versions https://pytorch.org/get-started/previous-versions/ (2.14.1 항목)
[28] torchvision 호환표 https://github.com/pytorch/vision#installation
[29] torchmetrics mAP https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html (1.9.0)
[30] opencv-python README https://github.com/opencv/opencv-python
[31] conda Managing environments https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html ; Anaconda pip-install https://www.anaconda.com/docs/getting-started/working-with-conda/packages/pip-install
[32] hydra-core PyPI https://pypi.org/project/hydra-core/ (1.3.8)
[33] Exa agent_run 합성 결과(2026-10-29): MMDetection, MMCV 휠 인덱스, detectron2, DEIM, YOLOX, LW-DETR, TensorRT pip, AlbumentationsX 블로그 — 원문 직접 확인 안 함
[34] Ultralytics 포럼 opencv headless https://community.ultralytics.com/t/opencv-python-and-opencv-python-headless/1022 (날짜 미상)
