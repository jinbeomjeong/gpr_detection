---
name: label-studio-local-files
description: LS 1.23.2 local-files 이미지는 프로젝트에 Local Files 소스 스토리지(경로 접두사)를 추가해야 보임; 서버 옵션; GPR-SD M3-4 키트 위치
metadata:
  type: reference
---

`/data/local-files/?d=<document_root 기준 경로>`는 환경변수(SERVING_ENABLED, DOCUMENT_ROOT)만으로는 안 열린다. `localfiles_data`(label_studio/io_storages/localfiles/views.py)가 요청 파일의 폴더가 어떤 프로젝트의 LocalFilesImportStorage.path 아래인지 본다 -> 프로젝트마다 Settings > Cloud Storage > Local Files(Sync 안 함) 추가, 경로는 document root 바로 그 폴더가 아니라 하위 폴더여야 한다.
서버 인자(argparser.py): `--internal-host localhost --no-browser --port 8080`(바인딩 localhost). `label-studio --version`은 첫 실행에 2분 넘게 걸림(타임아웃 주의).
**How to apply:** LS 과제 json은 `predictions` 키를 아예 넣지 않고, 내보내기는 전체 JSON(JSON-MIN 아님). GPR-SD M3-4 키트: data/labels/gprsd_m34/(git 제외), 스크립트 analysis/data/gprsd_m34_*.py, 설정 labeling/m34_gprsd/. 퍼센트 좌표는 px로 되돌릴 때 1e-3 px 스냅(6자리 반올림 대비).
