# 에이전트·env 보강 패키지 조사 (2026-10-29, 절약 모드)

조사: web-research-expert(절약). 기본 조사는 `packages.md`, 이 문서는 프로젝트 전용 에이전트 3개와 env 3개(gpr-dev, gpr-detr, gpr-label)에 필요한 빠진 부분만 다룬다. 버전·날짜는 PyPI 릴리스 RSS(2026-10-29) 기준. conda-forge win-64 py313 빌드는 메인 세션이 `conda search`로 확인했다(numba 0.68.0, pywavelets 1.10.0, imagehash 4.3.2 noarch, cleanlab 2.9.0 noarch).

## 핵심 발견
1. **cleanlab은 v2.8.0부터 Apache-2.0**이다(v2.7.1 이하는 AGPL-3.0). 2.8 이상으로 고정하면 gpr-dev에 넣어도 된다. 객체검출 라벨 오류 탐지 `cleanlab.object_detection.filter.find_label_issues`(labels: `{bboxes (L,4) xyxy, labels, image_name}`, predictions: 클래스별 `(M,5) [x1,y1,x2,y2,prob]`)를 쓴다. YOLO 출력 → 이 형식 어댑터는 직접 짠다.
2. **gpr-detr에서 `rfdetr[train]`은 `roboflow` → `opencv-python-headless`를 끌어온다.** gpr-detr에는 headless 하나만 둔다(opencv-python 금지). train extra는 `torchmetrics[detection]>=1.8.2,<1.9`로 고정한다. torch 상한이 없어 2.14.1+cu130과 공존 가능. classifiers에 Windows 표기는 없지만 의존성은 모두 Windows 휠이나 순수 Python이라 설치 가능으로 판단(추론). Windows에서는 DataLoader `num_workers=0` 권고 사례가 있다(단일 사례).
3. **numba 0.68은 numpy를 `<2.6`으로 묶는다.** gpr-dev에 `numpy<2.6` 고정이 필요하다.
4. **scipy.signal 1.18에는 `cwt`가 없다.** 웨이블릿은 PyWavelets(`pywt.cwt`, `pywt.wavedec`)로 한다. 그 밖(dewow·대역통과 `butter`+`sosfiltfilt`·`hilbert` 포락선·`find_peaks`·`medfilt`)은 scipy.signal로 충분하다.
5. **label-studio 1.23.2**(2026-09-29, Apache-2.0): `requires-python >=3.10,<4`, 공식 Dockerfile이 Python 3.13. 직접 의존 약 60개(Django 5.2, `pyarrow>=24,<25`, `pytz<2023` 등 고정 다수) → **반드시 별도 env**.

## env별 채택 목록

### gpr-dev (3개 에이전트 공용)
| 패키지 | 버전 | 출처 | 라이선스 | 쓰는 에이전트 | 비고 |
|---|---|---|---|---|---|
| numpy | >=2.2,<2.6 | conda-forge | BSD | 전체 | numba 상한 |
| pywavelets | >=1.10 | conda-forge | MIT/BSD-3 | processing | CWT·DWT 잡음 제거, imagehash 의존 |
| numba | >=0.68 | conda-forge | BSD | processing | Kirchhoff 이동·후보 탐색 루프 가속(선택) |
| pillow | >=12.3 | conda-forge | HPND | curator | |
| imagehash | >=4.3.2 | conda-forge | BSD-2 | curator | pHash·dHash 근접 중복 탐지 |
| cleanlab | >=2.8 | conda-forge | Apache-2.0 | curator, detection | extra 없이(datalab·image extra는 HF datasets를 끌어옴) |
| tensorboard | >=2.21 | conda-forge | Apache-2.0 | detection | protobuf>=6.31.1 |
| faster-coco-eval | >=1.8 | pip | Apache-2.0 | detection | **extra 없이**(`[extra]`는 headless를 끌어옴) |
| globox | >=2.9 | pip | MIT | curator | COCO·YOLO·VOC·CVAT 상호 변환 CLI |

넣지 않음: fiftyone(opencv-headless·MongoDB 의존, 무거움), cleanlab[image]/[datalab], albumentations.

### gpr-detr (Apache 이전 실험, ultralytics 없음)
| 패키지 | 버전 | 비고 |
|---|---|---|
| torch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 | cu130 인덱스 |
| rfdetr[train] | ==1.11.2 | 버전 고정(train extra 내부 고정: hotcoco==1.0.1, torchmetrics<1.9, roboflow) |
| transformers | >=5.1,<6 | RT-DETRv2(`RTDetrV2ForObjectDetection`), D-FINE(`DFineForObjectDetection`). timm은 불필요할 가능성 높음(미확인) |
| torchmetrics | >=1.8.2,<1.9 | rfdetr train과 맞춤(gpr-dev와 판이 다름) |
| faster-coco-eval | >=1.8 | |
| opencv-python-headless | >=4.10,<5 | cv2는 이것 하나만 |
| conda-forge | numpy, scipy, pillow, pyarrow>=15(vernier), pandas, matplotlib, tensorboard, pyyaml, tqdm, pytest | |

넣지 않음: `rfdetr[loggers]`(W&B·MLflow·ClearML 일괄), `rfdetr[augment]`(albumentations), `rfdetr[plus]`(PML 1.0), `rfdetr[cuda]`(Linux 전용). `accelerate`는 HF Trainer를 쓸 때만.

### gpr-label (Label Studio 전용)
| 패키지 | 버전 | 비고 |
|---|---|---|
| python | 3.13 | 막히면 3.12 |
| label-studio | ==1.23.2 | 다른 패키지를 넣지 않는다 |

실행 환경변수(PowerShell):
- `$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"`
- `$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="C:\workspace\gpr_detection\data"` (이미지 폴더의 상위)
- UI: Settings > Cloud Storage > Add Source Storage > Local Files, 절대 경로는 document root 아래.
- 내보내기: RectangleLabels는 JSON/JSON_MIN/CSV/COCO/Pascal VOC/YOLO. 클래스 순서는 기본 알파벳순이니 `<Label category="...">`로 고정한다. YOLO·COCO 내보내기에는 이미지가 빠진다. 큰 프로젝트는 동기 내보내기가 시간 초과될 수 있어 SDK 스냅샷이나 `label-studio export`를 쓴다.
- 로컬 파일 서빙은 보안 위험이 있으므로 localhost에서만 쓴다.

## 설치 뒤 로컬 확인 항목
- gpr-dev: numpy 2.5.x 유지(numba 상한), cv2 배포판 하나(opencv-python), `pip check`.
- gpr-detr: torch가 cu130 빌드로 유지되는지, cv2 배포판 하나(headless), simplejpeg cp313 win 휠 설치 여부.
- gpr-label: 소스 빌드 없이 설치되는지, `label-studio --version`.

## 불확실한 점
- cleanlab 2.8.0 릴리스 날짜·전환 공지 원문 미확인(LICENSE 파일로 확인).
- label-studio 1.23.2 태그의 pyproject version 문자열은 1.23.0으로 남아 있음. 설치 문서의 Python 하한(3.8+)은 낡음.
- rfdetr develop(1.12.0.dev)에서는 hotcoco·ultrafast-pycocotools 고정이 바뀜.

## Sources
[1] PyPI 릴리스 RSS `https://pypi.org/rss/project/<name>/releases.xml` (2026-10-29 조회)
[2] PyPI JSON pywavelets 1.10.0 https://pypi.org/pypi/pywavelets/1.10.0/json
[3] PyPI JSON numba 0.68.1 https://pypi.org/pypi/numba/0.68.1/json ; Numba 설치 문서 https://numba.readthedocs.io/en/stable/user/installing.html
[4] SciPy 1.18.0 scipy.signal https://docs.scipy.org/doc/scipy/reference/signal.html
[5] PyPI JSON imagehash 4.3.2 https://pypi.org/pypi/imagehash/4.3.2/json
[6] PyPI JSON cleanlab 2.9.0 https://pypi.org/pypi/cleanlab/2.9.0/json ; LICENSE v2.8.0 https://raw.githubusercontent.com/cleanlab/cleanlab/v2.8.0/LICENSE ; v2.7.1 https://raw.githubusercontent.com/cleanlab/cleanlab/v2.7.1/LICENSE
[7] cleanlab object_detection.filter https://docs.cleanlab.ai/stable/cleanlab/object_detection/filter.html
[8] PyPI JSON globox 2.9.0 https://pypi.org/pypi/globox/2.9.0/json ; fiftyone 1.22.1 https://pypi.org/pypi/fiftyone/1.22.1/json
[9] PyPI JSON faster-coco-eval 1.8.0 https://pypi.org/pypi/faster-coco-eval/1.8.0/json ; tensorboard 2.21.0 https://pypi.org/pypi/tensorboard/2.21.0/json
[10] rfdetr 1.11.2 pyproject https://raw.githubusercontent.com/roboflow/rf-detr/1.11.2/pyproject.toml ; 설치 문서 https://rfdetr.roboflow.com/latest/getting-started/install/
[11] PyPI JSON roboflow https://pypi.org/pypi/roboflow/json ; hotcoco 1.0.1, vernier 0.5.3, torch-hungarian 0.1.0rc0 (PyPI JSON)
[12] HF transformers RT-DETRv2 https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr_v2 ; D-FINE https://huggingface.co/docs/transformers/main/en/model_doc/d_fine
[13] label-studio 1.23.2 pyproject https://raw.githubusercontent.com/HumanSignal/label-studio/1.23.2/pyproject.toml ; Dockerfile https://raw.githubusercontent.com/HumanSignal/label-studio/1.23.2/Dockerfile
[14] Label Studio 설치 https://labelstud.io/guide/install.html ; 로컬 스토리지 https://labelstud.io/guide/storage_local ; 내보내기 https://labelstud.io/guide/export.html
