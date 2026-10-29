# 데이터셋 카드 (Hugging Face 데이터셋 카드 형식)

데이터셋마다 1장. 각 카드는 맨 위 YAML 메타데이터, 영어 요약 표, 한국어 본문 절(Dataset Summary / Dataset Structure / Dataset Creation / Considerations / Licensing & Citation / 이 프로젝트에서의 사용)로 구성한다. 수치는 근거 문서에서만 가져왔고 각 카드에 상대 링크가 있다.

| 카드 | 한 줄 설명 | 라이선스 | 이 프로젝트에서의 역할 |
|---|---|---|---|
| [aihub_71327.md](aihub_71327.md) | 서울 도로 GPR B-scan + VOC XML(공동·맨홀·종배관·횡배관), dataSetSn 71327 | `other` (이용약관 미확인, 원본·라벨·파생 데이터 비공개) | 검출 학습·검증·test (`aihub_yolo_v1`) |
| [morocco.md](morocco.md) | 모로코 GSSI 화면 캡처 224x224 패치, 관로·공동 박스(고유 285장 + 증강) | `cc-by-4.0` | 교차 도메인 평가 전용 (`morocco_eval_v1`) |
| [fourtu.md](fourtu.md) | 네덜란드 13개 현장 공기결합 500 MHz GPR + 시험굴착 정답(396개 관로) | `cc0-1.0` | 깊이 검증 (M2-8, 튜닝 없음) |
| [guangzhou.md](guangzhou.md) | 광저우대 IDS 지면결합 GPR 원시 B-scan(관로·철근), 정답 없음 | `cc-by-4.0` | 미라벨 추론·속도 추정 시험 (M3-7) |
| [gprsd.md](gprsd.md) | USTC 도로 GPR 장거리 측선 12개 + 이상체 표식 73건 | `unknown` (원 저장소 표기 없음, 인용 2편 요구) | 현장 시험·수동 박스 라벨 (M3-4) |

## 공개 규칙

- AIHub는 이용약관을 확인하지 못했으므로 원본 이미지·라벨·파생 데이터(`data/`)를 저장소에 올리지 않고 통계·파일명만 공개한다.
- 원본 이미지가 들어간 그림(몽타주 등)은 공개 저장소에 포함하지 않는다.
- GPR-SD는 라이선스가 불명이므로 데이터를 올리지 않는다.
- 가공본을 만든 기관·과제·연락처 정보는 공개 문서에서 제외했다.

## 클래스 매핑 (검출, 인덱스 고정)

| ID | 이름 | AIHub | 모로코 |
|---|---|---|---|
| 0 | `pipe_hyperbola` | `tpipe` | `pipe` |
| 1 | `pipe_band` | `lpipe` | - |
| 2 | `cavity` | `cavity` | `cavity` |
| 3 | `manhole` | `manh` | - |

근거: [ADR-003](../../architecture/ADR-003-data-split-and-classes.md). 상세 명세: [dataset_spec_m1.md](../dataset_spec_m1.md).
