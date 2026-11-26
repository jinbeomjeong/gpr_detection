# M3-4 GPR-SD 관로 라벨 키트 (Label Studio)

- 계획: [`docs/processing/m36_gprsd_plan.md`](../../docs/processing/m36_gprsd_plan.md) §5, 박스 규약(r1, 2026-11-26): [`docs/experiments/m34_gprsd_box_convention.md`](../../docs/experiments/m34_gprsd_box_convention.md)
- 사용자용 지침: [`guide_ko.md`](guide_ko.md) (라벨링 전에 읽는다)
- 이 폴더(공개 가능): `label_config.xml`(라벨 설정), `README.md`, `guide_ko.md`. 이미지·과제·창 목록은 모두 `data/labels/gprsd_m34/`(git 제외)에 있다.
- 사전 주석 없음: 과제 json에 `predictions`가 없다. 검출기·신호처리 결과(M3-6)는 보기 영상에도 지침에도 쓰지 않았다.

## 1. 키트 구성 (`data/labels/gprsd_m34/`, 시드 20261126)

| 파일 | 내용 |
|---|---|
| `windows.csv` | 창 48개(측선 12 × 창, 도로별 CR1 4·CR2 1·AR1 6·AR2 5 창을 평행 3측선에 각각) |
| `images/v01..v48.png` | 라벨 보기 영상. 번호는 섞은 제시 순서(도로 섞임) |
| `images_relabel/r01..r06.png` | 재라벨용 6창(본 라벨 영상과 같은 픽셀, 파일 이름·순서만 다름) |
| `tasks_main.json`, `tasks_relabel.json` | Label Studio 가져오기 과제 |
| `render_config.json` | 배율 2 px/트레이스·2 px/샘플, 측선별 클립 값, 창 원점, 영상 sha256 |
| `SHA256SUMS` | `images/` 48장·`images_relabel/` 6장·`windows.csv`·`render_config.json`·`tasks_main.json`·`tasks_relabel.json`의 sha256(LF, 58줄). `road_independence.json`과 `SHA256SUMS` 자신은 넣지 않는다 |
| `road_independence.json` | 도로 독립성 확인 수치(분석 산출물, SUMS 밖. 도로 독립성 근거는 계획 [`m36_gprsd_plan.md`](../../docs/processing/m36_gprsd_plan.md) §2-3) |

## 2. 서버 기동 (메인 세션, localhost 전용)

서버는 장시간 프로세스라 큐레이터가 띄우지 않는다. 메인 세션이 PowerShell 창 하나에서 띄운다(기존 AIHub 재라벨 서버와 같은 포트·같은 env).

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="C:\workspace\gpr_detection\data"
& C:\ProgramData\miniforge3\Scripts\conda.exe run -n gpr-label label-studio start --port 8080 --internal-host localhost --no-browser
```

- `--internal-host localhost`로 외부 인터페이스에 열지 않는다. 접속 주소는 `http://localhost:8080`.
- 자기 PID만 종료한다. 이미 8080에서 서버가 떠 있고 위 환경변수로 시작됐다면 다시 띄우지 않는다(환경변수를 확인할 수 없으면 재시작).

## 3. 프로젝트 만들기 (사용자 또는 메인, 브라우저)

Label Studio 1.23.2는 **Local Files 소스 스토리지를 프로젝트에 추가해야** `/data/local-files/?d=…` 이미지를 서빙한다(`localfiles_data`가 스토리지 경로 접두사로 권한을 본다). 동기화(Sync)는 하지 않는다. 과제는 json으로 가져온다.

본 라벨 프로젝트 `m34_gprsd_main`:

1. Create Project → 이름 `m34_gprsd_main`.
2. Labeling Setup → Custom template → Code에 [`label_config.xml`](label_config.xml) 전체를 붙여 넣는다 → Save.
3. Settings → Cloud Storage → Add Source Storage → **Local Files**
   - Absolute local path: `C:\workspace\gpr_detection\data\labels\gprsd_m34\images`
   - File Filter Regex: `.*png`
   - "Treat every bucket object as a source file"는 **끈다**. Add Storage만 하고 Sync는 누르지 않는다.
4. Data Manager → Import → `C:\workspace\gpr_detection\data\labels\gprsd_m34\tasks_main.json` 업로드 → 48개 과제가 순서대로 들어왔는지 확인(과제 번호 = 제시 순서). Settings → Annotation에서 "Show Skip"·"Allow empty annotations"는 기본값 유지, **Instructions**에 `guide_ko.md` 요약을 넣어도 된다.
5. 첫 과제를 열어 영상이 보이는지 확인한다(안 보이면 3단계 경로·서버 환경변수 확인).

재라벨 프로젝트 `m34_gprsd_relabel`(**본 라벨을 모두 끝낸 뒤**, 최소 하루 뒤 권장):

- 위와 같은 설정 xml, 스토리지 경로만 `C:\workspace\gpr_detection\data\labels\gprsd_m34\images_relabel`, 가져오기는 `tasks_relabel.json`(6과제).
- 별도 프로젝트로 두어 본 라벨 결과가 보이지 않게 한다.

## 4. 내보내기 → L0 변환

1. 각 프로젝트에서 Export → **JSON**(전체; JSON-MIN 아님) 저장. 예: `data/labels/gprsd_m34/ls_export_main_<날짜>.json`.
2. 변환(`gpr-dev`, CPU 1프로세스, 수 초):

```powershell
conda run -n gpr-dev python -I analysis/data/gprsd_m34_export_to_l0.py --export data/labels/gprsd_m34/ls_export_main_<날짜>.json --bundle main
conda run -n gpr-dev python -I analysis/data/gprsd_m34_export_to_l0.py --export data/labels/gprsd_m34/ls_export_relabel_<날짜>.json --bundle relabel
```

3. 산출(`data/labels/gprsd_m34/export_<bundle>/`): `labels_l0*.csv`(계획 §5-4 열 + `flags`·`qa`, 변환기가 더하는 열 `bundle`·`apex_trace_f`·`apex_sample_f`(꼭짓점 소수 좌표)·`annotation_id`(Label Studio 주석 번호)), `windows_status*.csv`(`window_complete`·메모), `qa_report*.csv`, `convert_summary*.json`.
4. 내보내기에 `predictions`가 하나라도 있으면 변환이 멈춘다. QA의 `ERROR`는 사용자에게 되묻는다(`apex_outside_box`, `no_apex`, `certainty_missing` 등), `WARN`은 보고용이다.
5. 위 여유·아래 길이 검사의 P는 **20.0 샘플**이다(M3-6 종료, [`m36_gprsd_scale.md`](../../docs/processing/m36_gprsd_scale.md) §13; M3-3b 주 경로와 같음). `< 0.6 P` 경고는 동결 전 재그리기 목록이 된다. `convert_summary`에는 P 22.3 기준 `< 0.6 P` 건수(`top_margin_lt_0.6P_at_P22.3`)를 보고 전용으로 함께 적는다. 변환기 기본값이 20.0이라 `--period`는 따로 줄 필요가 없다. 위·아래 여유 WARN은 `truncated` 박스에도 찍히므로, 재그리기 목록을 만들 때는 `labels_l0.csv`의 `flags`에 `truncated`가 있는 행을 거른다(규약 §3-4·§3-6: 잘린 박스는 고치지 않는다).
6. 변환기 자체 시험: `conda run -n gpr-dev python -I analysis/data/gprsd_m34_export_to_l0.py --selftest`
   (합성 박스 200개 L0↔보기 px↔Label Studio 퍼센트 왕복 정확 일치, `predictions` 중단, QA 오류 사례).

## 5. 동결

내보내기 → L0 변환 → QA → (위 여유 경고 박스 재그리기 → 다시 내보내기·변환) → sha256 매니페스트 커밋 순이다. 동결 전에 위 여유 경고를 해소하고, 남은 건수와 이유는 공개 사항으로 적는다. 이 커밋이 M3-3b 봉인보다 앞서야 한다. 커밋하는 것은 sha256과 건수뿐이다(`docs/datasets/gprsd_m34_label_manifest.json`). 영상·라벨·창 목록은 올리지 않는다(데이터 라이선스 미상).

## 6. 재현

```powershell
conda run -n gpr-dev python -I analysis/data/gprsd_m34_build_kit.py            # 창·영상·과제·SHA256SUMS·매니페스트
conda run -n gpr-label python -I analysis/data/gprsd_m34_check_label_config.py  # 설정 xml·과제 json 형식 점검(gpr-label)
conda run -n gpr-dev python -I analysis/data/gprsd_m34_road_independence.py     # 도로 독립성 수치
```

- 입력: `C:\workspace\GPR_Data\GPR-SD_중국USTC_도로GPR_가공본_261016`(읽기 전용, `01_메타데이터\marks.csv`, `05_radargrams_npy`). 시드 20261126(표시 순서·재라벨 선택).
