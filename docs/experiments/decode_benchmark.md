# 이미지 디코딩 방식별 처리 시간 비교 (상대 비교, 1차)

- 작성: 2026-10-29, python-expert
- 스크립트: `scripts/bench/decode_bench.py`
- 결과: `runs/bench/decode_20261029_163931/` (gpr-dev, 전체), `runs/bench/decode_20261029_164741/` (gpr-detr, simplejpeg 포함 CPU만)
  - 각 폴더에 `results.csv`(행 = 데이터셋·단계·방법), `results.json`(조건·부하·정확성·nvjpeg 초기화·DataLoader), `summary.md`(전체 표)
- **이 측정은 상대 비교다.** 다른 프로젝트가 CPU를 쓰는 중에 쟀으므로 절대값은 부풀려져 있다. CPU가 빌 때 아래 "재측정 명령"으로 판정 측정을 다시 한다.
- **판정 측정(M1-0b, 2026-10-31)은 §7에 있다.** 절대값과 판정은 §7을 기준으로 하고, §1–§4는 1차 상대 비교 기록으로 남긴다.

## 1. 측정 조건

| 항목 | 값 |
|---|---|
| CPU / GPU | i7-14700 (P 8코어 = 논리 0–15, E 12코어 = 논리 16–27) / RTX 4070 SUPER 12 GB |
| env | gpr-dev: OpenCV 4.14.0, Pillow 13.0.0(libjpeg-turbo 8.0), torch 2.14.1+cu130, torchvision 0.29.1, numpy 2.5.4 · gpr-detr: 같은 버전 + simplejpeg 1.9.0 |
| 코어 고정 | 프로세스 시작 직후 psutil로 affinity `0x0FFF0000`(E코어 12개) 적용. DataLoader worker도 같은 12코어를 물려받음을 확인(`results.json`의 `worker_affinities`) |
| 스레드 | `cv2.setNumThreads(1)`, `torch.set_num_threads(1)` — 단일 스레드 기준값 |
| 반복 | 워밍업 1회(시간은 따로 `warmup ms/img`로 기록) + 측정 5회. 이미지마다 `perf_counter_ns`로 재고, 5×N개 값의 중앙값·p5·p95를 낸다. img/s = N ÷ (5회 중 중앙값 패스 시간) |
| 표본 (seed 0) | AIHub: Training `TS_B scan_*.zip` 6개에서 크기 비례 무작위 500장(zip 안에서 읽고, 디스크 단계용으로 `runs/bench/.../samples/aihub/`에 복사) · 모로코: `images/train` 1,598장 중 500장(원위치에서 읽기만) · 4TU: `03_radargrams_png_raw`↔`05_radargrams_npy` 상대경로가 같은 쌍 200개 |
| GPU | `decode_jpeg(device="cuda")` 매 호출 뒤 `torch.cuda.synchronize()`. CUDA 컨텍스트·nvjpeg 첫 호출은 따로 잼 |
| OS 캐시 | 모든 읽기는 **OS 파일 캐시가 데워진 상태**다(표본을 고를 때 이미 한 번 읽음). 콜드 디스크 I/O는 재지 않았다(캐시 비우기에 관리자 권한 필요) |
| 소요 | gpr-dev 전체 470 s(이 중 DataLoader worker 기동이 약 4분), gpr-detr CPU만 79 s |

### 부하 상태 (오염 정도)

| 시점 | 전체 CPU | 고정한 E코어 평균 | GPU 사용률 / 메모리 |
|---|---:|---:|---|
| gpr-dev 시작 | 93.5 % | 100.0 % | 2 % / 127 MiB |
| gpr-dev 측정 중 (2 s 간격 222회) | 평균 93.8 % (92.7–97.2) | 평균 100.0 % (99.9–100) | 평균 2.3 % (최대 46 %, 우리 nvjpeg 구간) |
| gpr-dev 끝 | 93.6 % | 100.0 % | 2 % / 340 MiB |
| gpr-detr 측정 중 (36회) | 평균 93.3 % | 평균 100.0 % | 1.3 % |

- CPU 상위 프로세스는 다른 프로젝트의 python.exe 5개(각각 논리코어 2.5–4개분)였다. 요청 시점의 평균 ~73 %보다 높은 **~94 %**였고, **E코어는 측정 내내 100 %로 포화**됐다. "고정한 E코어 평균"에는 이 벤치마크 자신의 사용분도 들어 있다.
- 따라서 단일 스레드 값은 다른 프로세스와 시분할한 결과이고, 빈 CPU에서는 더 빠를 것이다. 방법 간 **비율**은 두 env 사이에서 1 % 안팎으로 재현됐다(아래 §2.4).

## 2. 결과

### 2.1 AIHub B-scan JPG (256×256 그레이, 평균 28.9 KB, zip 저장 방식 deflate)

| 단계 | 방법 | 중앙값 ms | p5 | p95 | img/s | cv2_gray 대비 |
|---|---|---:|---:|---:|---:|---:|
| (a) zip 읽기 | `ZipFile.read` (핸들 유지, inflate 포함) | 0.198 | 0.150 | 0.323 | 4,736 | — |
| (b) 메모리 디코딩 | cv2 IMREAD_GRAYSCALE | **0.425** | 0.298 | 0.580 | 2,311 | 1.00 |
| | simplejpeg GRAY (gpr-detr) | 0.428 | — | — | 2,301 | 1.00 |
| | tv decode_image GRAY | 0.490 | 0.361 | 0.654 | 2,016 | 1.15 |
| | tv decode_jpeg GRAY | 0.503 | 0.373 | 0.670 | 1,960 | 1.18 |
| | cv2 IMREAD_COLOR (BGR 3채널, Ultralytics 기본) | 0.510 | 0.383 | 0.663 | 1,935 | 1.20 |
| | PIL convert('L') | 0.562 | 0.428 | 0.725 | 1,766 | 1.32 |
| | cv2 IMREAD_COLOR → 그레이 변환 | 0.565 | 0.438 | 0.718 | 1,742 | 1.33 |
| | PIL draft('L', 원크기) + convert('L') | 0.568 | 0.437 | 0.732 | 1,742 | 1.34 |
| (c) 디스크 파일 | 파일 읽기만 (`read_bytes`) | 0.258 | 0.237 | 0.327 | 3,747 | — |
| | 읽기 + cv2 gray | 0.761 | 0.626 | 0.941 | 1,289 | — |
| | 읽기 + cv2 color BGR (Ultralytics 경로) | 0.816 | 0.679 | 0.986 | 1,204 | — |
| 사전 디코딩 캐시 | uint8 평면 memmap, 무작위 순서 | **0.015** | 0.015 | 0.017 | 63,291 | 0.04 |
| | 이미지별 uint8 .npy (`np.load`) | 0.370 | 0.340 | 0.455 | 2,641 | 0.87 |
| GPU (nvjpeg) | 한 장씩, 출력 GPU 유지 | 0.650 | 0.491 | 0.868 | 1,515 | 1.53 |
| | 한 장씩 + D2H | 0.729 | 0.568 | 0.970 | 1,350 | 1.72 |
| | 배치 32, 출력 GPU 유지 | 0.498 | 0.468 | 0.542 | 2,013 | 1.17 |
| | 배치 128, 출력 GPU 유지 | 0.486 | 0.476 | 0.506 | 2,041 | 1.14 |
| | 배치 128 + D2H | 0.582 | 0.565 | 0.604 | 1,727 | 1.37 |
| | 비교: CPU cv2 gray + H2D(pageable) | 0.548 | 0.418 | 0.712 | 1,787 | 1.29 |

- zip 열기(central directory 파싱) 1회 비용: 횡배관 7 ms, 종배관 20 ms, 공동_대칭형 112 ms, 맨홀 158 ms(엔트리 27,384개).
- nvjpeg 초기화: CUDA 컨텍스트 0.111 s, 첫 단일 호출 0.014 s, 첫 배치(32) 호출 0.016 s — 정상 상태 값에서 뺐다.

### 2.2 모로코 JPG (평균 18.9 KB)

표본 500장의 크기는 **224×224 463장, 268×268 18장, 336×336 19장**이다(224×224만 있다고 가정하면 안 된다).

| 단계 | 방법 | 중앙값 ms | p95 | img/s | cv2_gray 대비 |
|---|---|---:|---:|---:|---:|
| 메모리 디코딩 | cv2 gray | **0.262** | 0.477 | 3,356 | 1.00 |
| | simplejpeg GRAY (gpr-detr) | 0.262 | — | 3,369 | 1.00 |
| | cv2 color BGR | 0.325 | 0.545 | 2,749 | 1.24 |
| | tv decode_jpeg / decode_image | 0.337 | 0.57 | 2,644–2,673 | 1.29 |
| | PIL / PIL draft | 0.409 / 0.419 | 0.64 | 2,266 / 2,203 | 1.56 / 1.60 |
| 디스크 (원위치) | 읽기만 | 0.256 | 0.316 | 3,794 | — |
| | 읽기 + cv2 gray / color BGR | 0.601 / 0.652 | 0.84 / 0.90 | 1,593 / 1,459 | — |
| 캐시 | uint8 memmap / 이미지별 .npy | 0.015 / 0.389 | 0.021 / 0.501 | 62,658 / 2,486 | — |
| GPU | nvjpeg 한 장씩 / 배치 128 / 배치 128 + D2H | 0.626 / 0.358 / 0.446 | 0.92 / 0.37 / 0.49 | 1,480 / 2,776 / 2,257 | 2.39 / 1.37 / 1.70 |
| | CPU cv2 gray + H2D | 0.450 | 0.688 | 2,027 | 1.72 |

### 2.3 4TU radargram (raw PNG 평균 87 KB, npy int16 [512, traces] 평균 605 KB)

raw PNG는 RGBA(R=G=B, A=255)라서 모든 디코더가 그레이로 줄인다. PNG는 int16 원자료를 8비트로 줄인 것이므로 정보량이 npy와 다르다.

| 단계 | 방법 | 중앙값 ms | p5 | p95 | img/s |
|---|---|---:|---:|---:|---:|
| 메모리 디코딩 (PNG) | cv2 IMREAD_UNCHANGED → 채널 0 | 1.849 | 0.773 | 20.5 | 223 |
| | cv2 IMREAD_GRAYSCALE | 1.969 | 0.809 | 19.4 | 232 |
| | tv decode_image GRAY | 2.029 | 0.874 | 19.0 | 229 |
| | PIL convert('L') | 2.479 | 1.081 | 23.7 | 190 |
| 메모리 (npy) | `np.load(BytesIO)` | 0.155 | 0.105 | 1.37 | 3,005 |
| 디스크 | PNG 읽기 + cv2 gray | 2.336 | 1.199 | 20.0 | 211 |
| | **`np.load`** | **0.455** | 0.389 | 1.93 | 1,503 |
| | `np.load(mmap_mode="r")` 열기만(픽셀 안 읽음) | 1.258 | 1.141 | 1.46 | 788 |
| | `np.load(mmap_mode="r")` + `np.array` 복사 | 1.600 | 1.360 | 3.71 | 513 |

PNG의 p95가 중앙값의 10배인 것은 폭이 넓은 radargram(수천 trace) 때문이다. Windows에서는 파일마다 메모리 맵을 만드는 비용(~1.2 ms)이 600 KB를 통째로 읽는 것보다 크다.

### 2.4 반복성

같은 부하에서 두 env로 따로 실행한 cv2_gray 중앙값: AIHub 0.425 / 0.430 ms, 모로코 0.262 / 0.261 ms(차이 ≤ 1.2 %). 다른 방법들도 같은 범위로 맞았다.

### 2.5 DataLoader 처리량 (참고, batch 32, `persistent_workers`, 표본 목록을 epoch마다 4번 순회)

| 데이터 | 경로 | workers 0 | 2 | 4 | 첫 epoch(기동 포함) w=2 / w=4 |
|---|---|---:|---:|---:|---|
| AIHub | zip(worker별 핸들) + cv2 gray | 1,447 img/s | 2,457 | 4,261 | 16.2 s / 30.3 s |
| AIHub | 디스크 JPG + cv2 gray | 1,267 | 2,188 | 3,850 | 15.6 s / 30.3 s |
| 모로코 | 디스크 JPG + cv2 gray | 1,518 | 2,495 | 4,312 | 15.8 s / 29.8 s |
| 4TU | 디스크 PNG + cv2 gray | 210 | 395 | 725 | 17.2 s / 30.7 s |
| 4TU | 디스크 npy + `np.load` | 1,389 | 1,726 | 2,430 | 15.1 s / 29.9 s |

- 첫 epoch의 15–30 s는 Windows spawn으로 worker마다 torch·cv2를 import하는 시간이다(E코어 포화 상태라 길다). 짧은 실험을 반복하면 이 비용이 디코딩보다 크다.

### 2.6 정확성 (cv2 IMREAD_GRAYSCALE 대비)

| 데이터 | 방법 | 최대 절대차 | 다른 픽셀 비율 |
|---|---|---:|---:|
| AIHub / 모로코 | cv2 color(→gray, 채널 0), PIL, PIL draft, tv decode_jpeg, tv decode_image, simplejpeg(gpr-detr) | 0 | 0 |
| AIHub | nvjpeg (GPU) | **1** | 1.88 % |
| 모로코 | nvjpeg (GPU) | **1** | 1.63 % |
| 4TU PNG | cv2 unchanged 채널 0, PIL, tv decode_image | 0 | 0 |
| 4TU npy | mmap + 복사 vs `np.load` | 0 | 0 |

CPU 디코더는 모두 libjpeg-turbo 계열이라 비트 단위로 같다. nvjpeg는 IDCT 구현 차이로 픽셀의 약 2 %가 ±1 다르다. 학습과 추론에 서로 다른 디코더를 쓰면 이만큼 입력이 어긋난다.

## 3. 해석

### 3.1 1 epoch 디코딩 시간 추정 (이번 부하 기준, OS 캐시 warm)

| 데이터 (이미지 수) | zip 읽기 + cv2 gray, 1스레드 | 디스크 + cv2 color(Ultralytics), 1스레드 | DataLoader zip+cv2, w=4 | uint8 memmap, 1스레드 |
|---|---:|---:|---:|---:|
| AIHub 관로 (~5,600) | 3.5 s | 4.6 s | 1.3 s | 0.08 s |
| AIHub B-scan 전체 (74,838) | 46.7 s | 61.1 s | 17.6 s | 1.1 s |
| 모로코 전체 (2,239) | 디스크+cv2 gray 1.3 s | 1.5 s | 0.5 s (디스크 경로) | 0.03 s |
| 4TU 전체 (759) | PNG 1.8 s / npy 0.35 s | — | PNG 1.0 s / npy 0.3 s | — |

계산: 이미지당 중앙값 × 이미지 수(1스레드), 또는 이미지 수 ÷ 측정 img/s(DataLoader).

### 3.2 병목은 zip I/O인가 디코딩인가

- **디코딩이 크다.** AIHub zip 경로 0.623 ms 중 디코딩이 0.425 ms(68 %), zip 읽기+inflate가 0.198 ms(32 %)다.
- 핸들을 열어 둔 zip 읽기(0.198 ms)가 낱개 파일 읽기(0.258 ms)보다 빨랐다. Windows에서는 파일을 열 때마다 드는 비용(NTFS open, 백신 검사)이 deflate 해제보다 크다. 다만 zip을 **항목마다 다시 열면** 7–158 ms가 들어 디코딩의 수백 배가 된다. zip을 직접 읽는 로더는 worker마다 핸들을 한 번만 열어야 한다.
- 콜드 캐시(첫 디스크 읽기)는 재지 않았다. B-scan Training zip은 약 1.9 GB라 첫 epoch만 디스크 속도의 영향을 받고, 이후는 RAM 캐시에 남는다(메모리 여유가 있을 때).
- 단일 스레드 디코딩 ~0.4–0.8 ms/장이면 4 worker로 4,000 img/s 이상이다. YOLO11m 학습 스텝(GPU)과 Ultralytics 증강(mosaic 등)이 장당 이보다 훨씬 오래 걸릴 것으로 보이므로 **학습 전체에서 디코딩은 병목이 아닐 가능성이 높다**(추정: 학습 스텝과 증강 시간은 이번에 재지 않았다).

### 3.3 디코더 선택

- CPU에서는 **cv2 IMREAD_GRAYSCALE이 가장 빠르고**, simplejpeg와 같다. torchvision +15–18 %, cv2 IMREAD_COLOR +20 %(모로코 +24 %), PIL +32 %(모로코 +56 %).
- PIL `draft('L', 원크기)`는 효과가 없다. 원본이 이미 그레이이고 축소도 하지 않으므로 바꿀 것이 없다.
- Ultralytics는 IMREAD_COLOR로 3채널을 만든다. 그레이로 읽어 채널을 복제하는 것보다 20 % 느리지만 장당 0.085 ms 차이라 epoch당 AIHub 전체 기준 6 s 수준이다. 단, 이 프로젝트 학습은 data yaml `channels: 1`이라 Ultralytics가 IMREAD_GRAYSCALE로 직접 읽는다(M1-7) — 판정 재측정(M1-0b)은 그레이 직접 읽기를 기준으로 한다.

### 3.4 GPU 디코딩(nvjpeg)

- 이 크기(256/224 px, 20–30 KB)에서는 **CPU보다 빠르지 않다.** 배치 128로 GPU에 출력을 남겨도 0.486 ms로 cv2 gray(0.425 ms)보다 14 % 느리고, CPU 디코딩 후 H2D 복사(0.548 ms)와 비슷하다. GPU 사용률은 측정 중 최대 46 %였다.
- 이유(추정): RTX 4070 SUPER에는 A100/H100 같은 하드웨어 JPEG 디코더가 없고, 작은 이미지는 nvjpeg가 Huffman 단계를 CPU에서 처리하는 hybrid 경로를 쓴다. 그래서 CPU 경합의 영향도 그대로 받는다.
- Ultralytics 증강은 CPU(numpy)에서 하므로 GPU 디코딩 결과를 다시 내려야 한다(+D2H: 배치 128에서 0.582 ms, cv2보다 37 % 느림). 출력도 libjpeg-turbo와 ±1 다르다.

## 4. 학습 파이프라인 권고

| 선택지 | 판정 | 근거 |
|---|---|---|
| **그대로 JPG + cv2 (Ultralytics 기본)** | **기본값으로 권장** | AIHub 관로 5.6k는 1스레드에서도 epoch당 4.6 s, 4 worker면 1–2 s. 디코딩이 학습 병목일 가능성이 낮다. 추가 저장 공간과 변환 단계가 없다 |
| AIHub zip을 직접 읽는 로더 | Ultralytics 쓰는 동안은 비권장 | Ultralytics는 파일 경로를 요구하므로 학습용 JPG는 `data/` 아래로 풀어야 한다. 자체 로더를 쓸 때 zip 직접 읽기는 빠르다(핸들 유지 시 낱개 파일보다 23 % 빠름) |
| **사전 디코딩 uint8 memmap 캐시** | B-scan 전체(74,838장)로 반복 실험할 때 권장 | 0.015 ms/장(디코딩 대비 약 30배 빠름). 그레이 uint8 기준 크기는 74,838 × 65,536 B ≈ 4.9 GB(디스크), 관로 5.6k는 약 370 MB. 모로코처럼 크기가 섞이면 평면 배열 + (offset, H, W) 색인으로 저장해야 한다. Ultralytics `cache="ram"`은 3채널 BGR로 저장하므로 B-scan 전체면 약 14.7 GB가 필요하다(계산값) |
| 이미지별 .npy 캐시 (Ultralytics `cache="disk"` 방식) | 이득 작음 | 0.37 ms/장으로 cv2 디스크 경로(0.76–0.82 ms)의 절반이지만, 파일 열기 비용이 남아 memmap보다 25배 느리다 |
| GPU 디코딩 (nvjpeg) | 비권장 | §3.4 — 이 크기·이 GPU에서 이득이 없고, 출력이 ±1 다르며, CPU 증강과 맞지 않는다 |
| 4TU | **npy를 `np.load`(mmap 없이)로 읽기** | PNG 디코딩보다 5배 빠르고(0.455 vs 2.336 ms, p95 1.9 vs 20 ms) int16 원자료를 보존한다. 깊이 피팅에는 어차피 npy가 필요하다. 파일별 mmap은 Windows에서 오히려 느리다 |

DataLoader worker 기동(Windows spawn)에 15–30 s가 들므로 `persistent_workers=True`를 유지하고, 짧은 검증 실행이 잦으면 worker 수를 줄이는 편이 낫다.

## 5. 재측정 명령 (CPU가 빌 때 판정 측정)

PowerShell, 프로젝트 루트에서. 실행 전 `nvidia-smi`와 작업 관리자로 다른 무거운 작업이 없는지 확인한다(측정 시작·끝·중간 부하는 스크립트가 `results.json`에 남긴다).

```powershell
# 전체 (약 8분: 디코딩 ~3분 + DataLoader worker 기동 ~4분)
conda run --no-capture-output -n gpr-dev python -I scripts/bench/decode_bench.py

# simplejpeg 비교 (CPU만, 약 1.5분)
conda run --no-capture-output -n gpr-detr python -I scripts/bench/decode_bench.py --dataset aihub,morocco --no-gpu --threads 0
```

주요 옵션: `--dataset all|aihub,morocco,4tu`, `--n`(데이터셋별 표본 수, 기본 500/500/200), `--repeats 5`, `--warmup 1`, `--threads 2,4`(DataLoader worker 수, `0`이면 생략), `--out <폴더>`, `--affinity 0x0FFF0000|none`, `--no-gpu`, `--seed 0`. 결과 폴더 하나는 표본 복사본과 캐시를 포함해 약 130 MB다.

판정 측정에서는 P코어(`--affinity 0x0000FFFF`)로도 한 번 더 재서 E코어 값과 함께 적는 것을 권한다. 학습 worker가 실제로 어느 코어에서 돌지에 따라 기준이 달라진다.

## 6. 한계

- 부하 ~94 %, 고정한 E코어 100 % 포화 상태의 값이다. 절대값은 판정 측정으로 대체한다.
- 모든 읽기는 OS 캐시가 데워진 상태다. 콜드 디스크 I/O와 백신 실시간 검사의 첫 접근 비용은 따로 재지 않았다.
- YOLO11m 학습 스텝과 Ultralytics 증강 시간은 재지 않았으므로 §3.2의 "디코딩은 병목이 아닐 가능성이 높다"는 추정이다.
- nvjpeg 측정은 torchvision 0.29.1 API 기준이다. NVIDIA DALI 같은 다른 GPU 파이프라인은 gpr-dev에 없어 재지 않았다.

## 7. 판정 측정 (M1-0b, 2026-10-31)

- 측정: 메인 세션, 2026-10-31 11:18–11:28. 다른 작업이 없고 전 코어를 예약한 상태에서 4회를 차례로 돌렸다. 정리는 python-expert가 했다.
- 결과 폴더 `runs/bench/decode_m10b_20261031_111807_{E,P,detr_E,detr_P}/`(각각 `results.csv`·`results.json`·`summary.md`), 로그 `runs/logs/m10b_decode_bench.log`.
- 실행: §5 명령에 `--affinity`만 바꿨다. 표본·반복(seed 0, 워밍업 1 + 측정 5)·스레드 1·OS 캐시 warm 조건은 §1과 같다.

| 실행 | env | affinity | 소요 |
|---|---|---|---|
| E | gpr-dev, 전체 | `0x0FFF0000` (E 논리 16–27) | 302 s |
| P | gpr-dev, 전체 | `0x0000FFFF` (P 논리 0–15) | 169 s |
| detr_E | gpr-detr, AIHub·모로코 CPU만(simplejpeg 포함) | E | 60 s |
| detr_P | gpr-detr, 같은 범위 | P | 35 s |

### 7.1 측정 중 부하와 유효성

| 실행 | 전체 CPU 평균 (최소–최대) | 고정 코어 평균 (최대) | GPU 사용률 평균 (최대) |
|---|---:|---:|---:|
| E | 9.2 % (5.9–21.8) | 18.1 % (48.4) | 2.5 % (55) |
| P | 9.7 % (6.0–20.8) | 12.5 % (28.9) | 2.5 % (50) |
| detr_E | 8.1 % (6.0–11.1) | 15.6 % (18.2) | 1.3 % (2) |
| detr_P | 9.3 % (6.0–12.6) | 11.7 % (16.3) | 1.2 % (2) |

- **유효로 판단한다.** 전체 CPU 평균 8–10 %로 오염 기준(평균 20 %)의 절반 아래다. 1차 측정은 전체 94 %, 고정 E코어 100 %였다.
- 이 부하의 대부분은 벤치마크 자신이다.
  - 단일 스레드 측정 1개는 E 12코어 중 1개(8.3 %), P 16논리 중 1개(6.3 %)를 채운다.
  - 고정 코어 최댓값(E 48 %, P 29 %)은 DataLoader worker 4개 구간에서 나왔다.
- 시작·끝 시점 상위 프로세스는 Defender(MsMpEng)·claude.exe·시스템 서비스로, 각각 논리코어 0.1–0.7개분이다.
  - detr_E 끝 시점에만 xelatex.exe가 논리코어 약 1개분으로 잡혔다.
  - 그 측정의 cv2 gray 값은 같은 코어의 gpr-dev 측정과 AIHub 1.4 %, 모로코 3.8 % 안에서 맞고 오히려 조금 빠르다(아래 재현성). 영향은 보이지 않는다.
- **재현성**: 같은 코어 종류에서 env만 다른 두 실행의 cv2 gray 중앙값 차이가 작다.
  - E: AIHub 0.277 / 0.273 ms, 모로코 0.182 / 0.175 ms.
  - P: AIHub 0.177 / 0.176 ms, 모로코 0.100 / 0.101 ms.
  - 차이는 ≤ 4 %, P에서는 ≤ 1 %다.
- 07:00에 시작했다가 중단된 측정(`runs/bench/decode_m10b_20261031_065951_{E,P,detr_E}`)은 참고용으로만 남긴다. 사용자가 처음부터 다시 재라고 지시해 판정에 쓰지 않는다.
  - E·P는 끝까지 돌았지만, P의 측정 중 부하가 평균 17.8 %, 고정 코어 26 %로 이번보다 높았다.
  - detr_E는 표본만 있고 detr_P는 없다.

### 7.2 AIHub B-scan JPG (256×256 그레이, 500장)

단위는 중앙값 ms/장(1스레드)이다. "1차"는 §2.1(E코어, 부하 94 %) 값이다.

| 단계 | 방법 | E | P | 1차 (E) | E 변화 |
|---|---|---:|---:|---:|---:|
| zip 읽기 | `ZipFile.read` (핸들 유지) | 0.132 | 0.070 | 0.198 | −33 % |
| 디스크 읽기 | `read_bytes` | 0.169 | 0.073 | 0.258 | −34 % |
| 메모리 디코딩 | **cv2 IMREAD_GRAYSCALE** | **0.277** | **0.177** | 0.425 | −35 % |
| | simplejpeg GRAY (gpr-detr) | 0.274 | 0.178 | 0.428 | −36 % |
| | tv decode_image GRAY | 0.321 | 0.192 | 0.490 | −34 % |
| | cv2 IMREAD_COLOR (BGR) | 0.322 | 0.203 | 0.510 | −37 % |
| | PIL convert('L') | 0.358 | 0.220 | 0.562 | −36 % |
| 디스크 읽기 + 디코딩 | **cv2 gray** | **0.476** | **0.282** | 0.761 | −37 % |
| | cv2 color BGR | 0.522 | 0.308 | 0.816 | −36 % |
| 캐시 | uint8 memmap(무작위) | 0.011 | 0.004 | 0.015 | −31 % |
| | 이미지별 .npy | 0.242 | 0.108 | 0.370 | −35 % |
| GPU | nvjpeg 배치 128(출력 GPU) | 0.315 | 0.214 | 0.486 | −35 % |
| | nvjpeg 배치 128 + D2H | 0.423 | 0.241 | 0.582 | −27 % |
| | cv2 gray + H2D | 0.355 | 0.221 | 0.548 | −35 % |

DataLoader 처리량(batch 32, `persistent_workers`, img/s, 괄호는 기동을 포함한 첫 epoch):

| 경로 | 코어 | workers 0 | 2 | 4 | 1차 (E) w=0 / 2 / 4 |
|---|---|---:|---:|---:|---|
| zip + cv2 gray | E | 2,113 | 2,945 (9.0 s) | 6,253 (16.2 s) | 1,447 / 2,457 / 4,261 |
| | P | 3,654 | 6,502 (4.7 s) | 12,050 (8.8 s) | – |
| 디스크 JPG + cv2 gray | E | 1,917 | 2,826 (8.7 s) | 5,616 (16.0 s) | 1,267 / 2,188 / 3,850 |
| | P | 3,359 | 6,103 (4.6 s) | 10,943 (8.7 s) | – |

### 7.3 모로코 JPG와 4TU

| 데이터 | 방법 | E | P | 1차 (E) |
|---|---|---:|---:|---:|
| 모로코 | 메모리 cv2 gray | 0.182 | 0.100 | 0.262 |
| | 디스크 + cv2 gray | 0.385 | 0.203 | 0.601 |
| | DataLoader 디스크 w=0 / 2 / 4 (img/s) | 2,322 / 3,148 / 6,493 | 4,288 / 7,547 / 13,300 | 1,518 / 2,495 / 4,312 |
| 4TU | 디스크 PNG + cv2 gray | 1.563 | 1.030 | 2.336 |
| | **디스크 npy `np.load`** | **0.311** | **0.156** | 0.455 |
| | `np.load(mmap)` + 복사 | 1.010 | – | 1.600 |
| | DataLoader PNG w=0 / 2 / 4 (img/s) | 306 / 563 / 1,074 | 478 / 904 / 1,646 | 210 / 395 / 725 |
| | DataLoader npy w=0 / 2 / 4 (img/s) | 1,836 / 2,245 / 3,738 | 3,481 / 4,112 / 5,203 | 1,389 / 1,726 / 2,430 |

정확성(cv2 gray 대비)은 1차와 같다.
- CPU 디코더는 모두 비트 단위로 같다.
- nvjpeg는 최대 ±1이고, 다른 픽셀 비율은 AIHub 1.88 %, 모로코 1.63 %다.

### 7.4 1차 대비 변화

- **절대값**: 같은 E코어에서 1스레드 디코딩·읽기가 대부분 27–37 % 빨라졌다. 1차 값은 다른 프로세스와 시분할한 결과였다는 §1의 해석이 맞았다.
  - DataLoader 처리량은 20–54 % 늘었다(w=2가 +20–43 %로 가장 작다).
  - worker 기동을 포함한 첫 epoch는 w=2에서 16 s → 9 s, w=4에서 30 s → 16 s로 줄었다.
- **순위는 그대로다.**
  - CPU 디코더 중 cv2 gray가 가장 빠르고 simplejpeg와 같다.
  - cv2 gray 대비 E/P 기준으로 tv decode_image +16 % / +8 %, cv2 color BGR +16 % / +15 %, PIL +29 % / +24 %다.
  - 4TU는 npy가 PNG보다 5–7배 빠르다.
- **P코어는 E코어보다 1.6–3.1배 빠르다**(배율은 `results.csv` 원값 기준).
  - 디코딩은 AIHub 0.17715 / 0.2772 ms로 1.6배, 모로코 1.8배다.
  - 파일 읽기는 0.0735 / 0.1689 ms로 2.3배다.
  - memmap은 AIHub 0.0041 / 0.0107 ms로 2.6배, 모로코 0.0034 / 0.0105 ms로 3.1배다.
- **zip 대 낱개 파일**: P코어에서는 zip 읽기와 낱개 파일 읽기의 차이가 거의 없어졌다(0.070 / 0.073 ms). E코어에서는 zip이 22 % 빠르다. 1차의 "zip이 23 % 빠름"은 E코어에서만 유지된다.
- **nvjpeg는 여전히 CPU보다 느리다.**
  - 배치 128, 출력을 GPU에 둔 경우: E +14 %, P +21 %.
  - D2H 포함: E +53 %, P +36 %.
  - §3.4 판단(이 크기·이 GPU에서 이득 없음)이 그대로다.

### 7.5 판정

1. **디코딩 경로: 지금대로 디스크 JPG + cv2 IMREAD_GRAYSCALE**로 한다. Ultralytics data yaml `channels: 1` 경로, M1-7과 같다.
   - CPU 디코더 중 가장 빠르고, simplejpeg를 들여와도 이득이 없다.
   - nvjpeg와 zip 직접 로더는 쓰지 않는다. 이유는 §4와 같고, 판정 측정으로 확인됐다.
   - 4TU는 **npy `np.load`(mmap 없이)**로 읽는다.
   - uint8 memmap 캐시는 M1-10에 필요 없다(아래 2). B-scan 전체(74,838장)로 반복 실험할 때만 다시 검토한다.
2. **디코딩은 학습 병목이 아니다.** §3.2의 추정을 확인했다.
   - **M1-7 학습 처리량**: fold 0 학습 11,276장을 학습 구간 약 27 s/epoch에 처리했다(`m17_compute_probes.md`, b64·w2·마스크 0xFF). 약 **420 img/s**다.
   - **디코딩 수요**: 학습 코어(P)의 디스크 읽기 + cv2 gray 0.282 ms/장 기준으로 약 0.12 코어다.
   - **로더 여유**: 같은 경로 DataLoader w=2 처리량 약 6,100 img/s(P)의 약 7 %만 쓴다. E코어로 밀려나도(w=2 2,826 img/s) 약 15 %다.
   - **epoch 디코딩 시간**: 1스레드 기준 P 3.2 s, E 5.4 s로, epoch 약 30 s의 1/10–1/6이다. 이 시간도 worker 2개에 나뉜다.
   - M1-7에서 GPU 사용률이 64–77 %(동작 중 평균, `m17_augmentation_strength.md` §3)에 머문 원인은 디코딩이 아니라 Ultralytics 증강·배치 묶기와 메인 프로세스 스텝 오버헤드, 검증 구간으로 본다(`m17_compute_probes.md`의 추정과 같다).
   - 한계: 학습 스텝과 증강 시간을 단계별로 나눠 재지는 않았다. 위 수치는 디코딩 몫의 상한 비교다.

### 7.6 학습 DataLoader worker 권고

- **M1-10: `workers 2`, 마스크 `0xFF`(P 논리 0–7)를 유지한다.** `m110_baseline_plan.md`의 설정(batch 64·workers 2·마스크 0xFF) 그대로다.
  - worker는 Ultralytics `InfiniteDataLoader`가 epoch 사이에 재사용한다. 이는 기본 동작이다(8.4.192 `ultralytics/data/build.py`, `persistent_workers` 인자와는 별개).
  - 디코딩 수요(약 420 img/s)는 worker 1개로도 남는다. P 1스레드 디스크 + cv2 gray가 약 3,500 img/s다.
  - worker 수를 늘려 얻는 이득은 M1-7 probe에서 보이지 않았다. 마스크는 0xFF가 가장 빨랐다(w2 45.6 s 대 0xFFFF 계열 50–72 s).
    - 같은 0xFFFF 안에서는 w3(50.3 s)가 w2(72.4 s)보다 빨랐다.
    - 0xFF에서 w3·w4는 재지 않았다.
  - 디코딩만 보면 worker를 늘릴 이유가 없다.
- **worker 기동 비용**(Windows spawn, 첫 epoch):
  - P: w=2 약 4.7 s, w=4 약 8.8 s.
  - E: w=2 약 9 s, w=4 약 16 s.
  - 50 epoch 학습에서는 무시할 수준이다. 짧은 smoke·평가를 반복할 때는 w=2 이하가 낫다.
- **E코어에서 학습해야 할 때**(P를 다른 세션이 쓸 때): 디코딩은 P보다 1.6배 느리지만 수요의 약 15 %라 디코딩 자체는 문제가 아니다. 다만 다른 세션과 E코어를 나누면 증강·메인 프로세스가 느려진다(`m17_compute_probes.md` 공존 시험). 배치는 compute-scheduler-expert 배치표를 따른다.
- **4TU 평가 로더**: npy `np.load`면 w=0으로 충분하다. P 3,481 img/s, 759장 전체가 1초 안이다.
