# _vendor — 데이터 가공본 원본 리더 (공개 저장소 제외)

이 폴더의 리더 스크립트 2개는 데이터 가공본에 들어 있던 파일이다. 작성자가 데이터 가공본 제공처이고 파일에 라이선스 표기가 없어서 **공개 저장소에는 올리지 않는다**(사용자 결정 2026-10-29, `.gitignore`에 등록).

| 파일 | 원본 위치 (`C:\workspace\GPR_Data` 아래) | 용도 |
|---|---|---|
| `sgy_radarmap.py` | `4TU_네덜란드_매설관GPR_가공본_261015\06_scripts\sgy_radarmap.py` | 4TU RadarMap SEG-Y 원본 읽기(`read_sgy`) |
| `ids_dt.py` | `광저우대_중국_관로GPR_가공본_261016\06_scripts\ids_dt.py` | 광저우대 IDS `.dt` / MALA 원본 읽기 |

- 현재 `gpr_detection.io` 로더는 가공본의 `05_radargrams_npy`(npy+json)만 읽으므로 이 두 파일이 없어도 동작한다.
- 원본 SEG-Y·`.dt`를 직접 읽어야 할 때만 위 경로에서 이 폴더로 복사해 쓴다. 복사한 파일의 머리말에 출처와 SHA256을 남긴다.
